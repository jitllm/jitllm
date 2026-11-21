#!/usr/bin/env python3
"""Build the Kimi-K3 fixture and record the reference's logits.

    $JITLLM_HF_PY scripts/k3gold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)
    K3_CODE=$JITLLM_MODELS/hf/kimi-k3-code  (Moonshot's remote code)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/kimik3/<name>.json

    K3_REAL=<checkpoint dir> [K3_OUT=<dir>] [K3_ROUND=float16] ... real-<name>
-> a trained checkpoint's golden (the "real checkpoints" section below)

There is no transformers class for Kimi-K3: the oracle is Moonshot's own
remote code (modeling_kimi_linear.py and configuration_kimi_k3.py, the text
model KimiK3ForConditionalGeneration wraps), run on the CPU. It imports fla's
Triton kernels, which do not run here, so five fla symbols are replaced by
torch -- fla's OWN torch references where fla has one, and nothing else in the
remote code is touched:

    chunk_kda, fused_recurrent_kda   fla.ops.kda.gate's naive_kda_gate /
                                     naive_kda_lowerbound_gate (the gate the
                                     kernel computes in place), the beta
                                     sigmoid, transformers' kimi_linear l2norm
                                     ("aligned with FLA's") on q and k, then
                                     fla.ops.kda.naive's naive_recurrent_kda
    FusedRMSNormGated                rmsnorm(x) * w * sigmoid(g) in f32
                                     (fla's LayerNormGatedFunction, rms form)
    ShortConvolution                 the depthwise causal nn.Conv1d fla's class
                                     IS, then SiLU
    prepare_*_from_mask, tensor_cache   unused without a padding mask

check_shims() holds the KDA shim to transformers' kimi_linear torch kernels
(recurrent and chunked) on the unbounded gate, where the two coincide, before
any golden is written. The remote code forces flash_attention_2; the MLA's
attention runs the same file's eager_attention_forward instead.

c6gold.py's method otherwise: the GGUF is written by llama.cpp's OWN
converter (conversion/kimi_k3.py) from a checkpoint saved in
KimiK3ForConditionalGeneration's layout, every tensor F32, the experts plain
F32 (no compressed-tensors MXFP4: the engine's MXFP4 kernels quantize the
activation, which would flip the router's discrete selection at near ties),
and every parameter randomised -- the selection bias, A_log and dt_bias
included.

Every K3 feature is inside the golden's 24 positions: six blocks, MLA at 3
and 6 (one-indexed) and KDA elsewhere, a residual checkpoint every two blocks
(three in the bank by the end, each mix a softmax over up to four), a dense
lead under situ, then a latent mixture (64 -> 32 -> experts -> norm -> 64)
beside one shared expert, the MLA output gate, the full-rank KDA gate and
gate_lower_bound. The FFN projections are wide enough that both of situ's
tanh bounds bite (beta 4 on the gate, 25 on up), and rms_norm_eps is 1e-3
with the latent projections small, so the MLA latent norms' own 1e-6 (the
class's default, not rms_norm_eps) is visible in the logits.

The tokenizer is Kimi-K3's own tiktoken (tokenization_kimi.py).
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
HFDIR = refenv.under("CV_HF", "hf")
CODE = os.environ.get("K3_CODE", os.path.join(HFDIR, "kimi-k3-code"))
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "kimik3")
NPOS, NLOGIT = 24, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "naïve café — 東京 🙂 \n\n\n end",
]

NLAYER = 6
FULL = [3, 6]  # one-indexed, as linear_attn_config lists them
TEXT = dict(
    vocab_size=163840, hidden_size=64, intermediate_size=64, num_hidden_layers=NLAYER,
    num_attention_heads=2, num_key_value_heads=2, head_dim=32,
    hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
    rms_norm_eps=1e-3, tie_word_embeddings=False,
    q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
    mla_use_nope=True, mla_use_output_gate=True,
    num_experts=8, num_experts_per_token=2, num_shared_experts=1, moe_intermediate_size=32,
    routed_expert_hidden_size=32, latent_moe_use_norm=True, routed_scaling_factor=2.0,
    moe_renormalize=True, moe_router_activation_func="sigmoid", first_k_dense_replace=1,
    moe_layer_freq=1, use_grouped_topk=True, num_expert_group=1, topk_group=1,
    topk_method="noaux_tc", attn_res_block_size=2, max_position_embeddings=4096,
    linear_attn_config={
        "full_attn_layers": FULL,
        "kda_layers": [i for i in range(1, NLAYER + 1) if i not in FULL],
        "gate_lower_bound": -5.0, "head_dim": 32, "num_heads": 2,
        "short_conv_kernel_size": 4, "use_full_rank_gate": True,
    },
    bos_token_id=163584, eos_token_id=163586, pad_token_id=163839,
)


# --- fla, in torch -----------------------------------------------------------

def naive_kda_gate(g, A_log, dt_bias=None, output_dtype=torch.float32):
    # fla 0.5.2, fla/ops/kda/gate.py, verbatim.
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = (-A_log.view(H, 1).float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


def naive_kda_lowerbound_gate(g, A_log, dt_bias=None, lower_bound=-5.0, output_dtype=torch.float32):
    # fla 0.5.2, fla/ops/kda/gate.py, verbatim.
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    # fla 0.5.2, fla/ops/kda/naive.py, verbatim.
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def l2norm(x, dim=-1, eps=1e-6):
    # transformers' kimi_linear.l2norm, "aligned with the l2norm implementation
    # in the FLA library": x / sqrt(sum(x^2) + eps).
    return x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def kda(q, k, v, g, beta, A_log, dt_bias, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
        lower_bound=None, **_):
    """chunk_kda and fused_recurrent_kda: the same function, chunked or not."""
    assert use_gate_in_kernel and use_beta_sigmoid_in_kernel
    if lower_bound is None:
        g = naive_kda_gate(g, A_log, dt_bias)
    else:
        g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound)
    beta = beta.float().sigmoid()
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q.float()), l2norm(k.float())
    return naive_recurrent_kda(q, k, v, g, beta, initial_state=initial_state,
                               output_final_state=output_final_state)


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **_):
        super().__init__()
        assert activation == "sigmoid"
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, g, **_):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (x * torch.sigmoid(g.float())).to(dt)


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **_):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                         bias=bias, padding=kernel_size - 1)
        assert activation in ("silu", "swish")

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **_):
        assert cache is None and cu_seqlens is None, "the golden runs the whole sequence uncached"
        t = x.shape[1]
        y = F.conv1d(x.transpose(1, 2), self.weight, self.bias, padding=self.padding,
                     groups=self.groups)[..., :t]
        return F.silu(y).transpose(1, 2), None


def install_fla():
    mods = {}
    for n in ("fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils", "fla.ops.utils.index",
              "fla.utils"):
        mods[n] = sys.modules[n] = types.ModuleType(n)
    mods["fla.modules"].FusedRMSNormGated = FusedRMSNormGated
    mods["fla.modules"].ShortConvolution = ShortConvolution
    mods["fla.ops.kda"].chunk_kda = kda
    mods["fla.ops.kda"].fused_recurrent_kda = kda

    def unused(*a, **k):
        raise AssertionError("the golden runs with no padding mask")

    mods["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = unused
    mods["fla.ops.utils.index"].prepare_lens_from_mask = unused
    mods["fla.utils"].tensor_cache = lambda f: f


def check_shims():
    """The KDA shim against transformers' kimi_linear torch kernels, on the
    unbounded gate (theirs has no lower bound): both arms are handed the same
    pre-gate and must agree to f32 rounding."""
    from transformers.models.kimi_linear import modeling_kimi_linear as tk
    torch.manual_seed(1)
    B, T, H, K = 1, 37, 2, 32
    q, k, v = (torch.randn(B, T, H, K) for _ in range(3))
    g, beta = torch.randn(B, T, H, K), torch.randn(B, T, H)
    A_log, dt = torch.randn(H) * 0.5, torch.randn(H * K)
    ours, _ = kda(q, k, v, g, beta, A_log, dt, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True)
    gg = naive_kda_gate(g, A_log, dt)
    for name, fn in (("recurrent", tk.recurrent_kimi_delta_attention),
                     ("chunked", tk.chunk_kimi_delta_attention)):
        fn = getattr(fn, "__wrapped__", fn)
        theirs, _ = fn(q, k, v, gg, beta.sigmoid(), initial_state=None, output_final_state=False,
                       use_qk_l2norm_in_kernel=True)
        d = (ours - theirs).abs().max().item()
        print(f"shim vs transformers' {name} KDA: max |d| {d:.3e}")
        assert d < 1e-4, name


def load_remote(code=CODE):
    install_fla()
    # The remote code was written against transformers 4.56, where
    # OutputRecorder lived in utils.generic; 5.x moved it, unchanged.
    import transformers.utils.generic as gen
    if not hasattr(gen, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder
        gen.OutputRecorder = OutputRecorder
    spec = importlib.util.spec_from_file_location("k3ref", os.path.join(code, "configuration_kimi_k3.py"),
                                                  submodule_search_locations=[code])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["k3ref"] = pkg
    from k3ref.configuration_kimi_k3 import KimiLinearConfig
    from k3ref import modeling_kimi_linear as ml
    # And 5.x renamed create_causal_mask's input_embeds to inputs_embeds and
    # reads the positions off position_ids, which the remote code also passes,
    # in place of cache_position. A checkpoint whose modeling_kimi_linear
    # re-exports another file (Kimi-K3-0.40B's modeling_kimi_k3_linear, written
    # against 5.x already) is patched where its model reads the name.
    mod = sys.modules[ml.KimiLinearModel.__module__]
    mask = mod.create_causal_mask

    def create_causal_mask(input_embeds=None, cache_position=None, **kw):
        if input_embeds is not None:
            kw["inputs_embeds"] = input_embeds
        return mask(**kw)

    mod.create_causal_mask = create_causal_mask
    return KimiLinearConfig, ml


def perturb(m):
    # c6gold.perturb's classification, with K3's widths: the FFN gate and up
    # projections wide enough that situ's two tanh bounds bite, the latent
    # projections small enough that the latent norms' eps shows, A_log as the
    # class draws it (log of U(1, 16)), and every other parameter -- the
    # selection bias and dt_bias included -- drawn.
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and ("norm" in last or last == "o_norm"):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
            elif n.endswith(("gate_proj.weight", "w1.weight")) and "self_attn" not in n:
                p.copy_(torch.randn_like(p) * 0.75)
            elif n.endswith(("up_proj.weight", "w3.weight")) and "routed_expert" not in n:
                p.copy_(torch.randn_like(p) * 2.0)
            elif n.endswith(("down_proj.weight", "w2.weight")) and "routed_expert" not in n:
                p.copy_(torch.randn_like(p) * 0.05)
            elif n.endswith(("q_a_proj.weight", "kv_a_proj_with_mqa.weight")):
                p.copy_(torch.randn_like(p) * 0.02)
            elif n.endswith("e_score_correction_bias"):
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        for n, b in m.named_buffers():
            raise AssertionError(f"unexpected buffer {n}: randomise it or say why not")


def save_original(m, d, text=TEXT):
    """KimiK3ForConditionalGeneration's layout: the text model under
    language_model., which llama.cpp's converter reads; text is the
    configuration the model was built from."""
    out = {"language_model." + n: t.contiguous() for n, t in m.state_dict().items()}
    save_file(out, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    tc = dict(text)
    tc.update({"architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear",
               "torch_dtype": "float32"})
    c = {"architectures": ["KimiK3ForConditionalGeneration"], "model_type": "kimi_k3",
         "text_config": tc, "torch_dtype": "float32", "tie_word_embeddings": False,
         "bos_token_id": TEXT["bos_token_id"], "eos_token_id": TEXT["eos_token_id"],
         "pad_token_id": TEXT["pad_token_id"]}
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)
    for f in ("tiktoken.model", "tokenization_kimi.py", "tokenizer_config.json", "encoding_k3.py"):
        shutil.copy(os.path.join(CODE, f), d)


def ragged_text():
    # TEXT with the latent and KDA's head 40 wide, not a whole number of the
    # activation quantizer's 32-element blocks: the ragged-width gate's (the
    # latent, and KDA's low-rank gates, whose rank is its head width).
    t = dict(TEXT, routed_expert_hidden_size=40)
    t["linear_attn_config"] = dict(TEXT["linear_attn_config"], head_dim=40)
    return t


def build(name):
    KimiLinearConfig, ml = load_remote()
    check_shims()
    text = ragged_text() if name.endswith("-ragged") else TEXT
    cfg = KimiLinearConfig(**text)
    torch.manual_seed(20261004)
    m = ml.KimiLinearForCausalLM(cfg).float().eval()
    # KimiLinearModel forces flash_attention_2, which has no CPU kernel here;
    # the same file's eager_attention_forward is the reference attention.
    cfg._attn_implementation = "eager"
    perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    save_original(m, d, text)
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d,
                    "--outtype", "f32", "--outfile", dst], check=True)
    ids = [(i * 7919 + 13) % 50000 for i in range(NPOS)]
    with torch.no_grad():
        lg = m(torch.tensor([ids]), use_cache=False).logits[0]
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False)} for t in TEXTS]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"ids": ids, "vocab": cfg.vocab_size, "pos": rows, "texts": texts},
              open(os.path.join(OUT, name + ".json"), "w"))
    print("wrote", dst, "argmax", [r["argmax"] for r in rows])


# --- real checkpoints ----------------------------------------------------------
#
#     K3_REAL=<checkpoint dir> K3_OUT=<dir> $JITLLM_HF_PY scripts/k3gold.py real-<name>
#
# (with the variables the head reads set as for the fixture; K3_ROUND below.)
#
# K3_REAL holds a published KimiK3ForConditionalGeneration checkpoint whole:
# config.json, model.safetensors and the remote code beside them, which is the
# reference (the checkpoint's own code, fla swapped as above). The text model is
# built in float32 from the file's own weights (language_model.*; the vision
# tower is not read) and run greedy from each prompt for REAL_GEN tokens; then
# the prompt and its continuation are run once more, teacher-forced, and every
# position's logits are recorded as the vocabulary's first REAL_HEAD and the
# reference's REAL_TOP highest. The engine is held to it through llama.cpp's
# converter's GGUF of the same checkpoint and through this tree's conversion of
# the checkpoint itself (the engine's TestKimiK3RealMatchesReference names the
# files); a compressed-tensors MXFP4 checkpoint is read through real_weights,
# and needs compressed-tensors in the venv. It writes $K3_OUT/real-<name>.json, the golden's home being
# testdata/golden/kimik3; run on another host, K3_OUT is where the golden is
# fetched back from.
REAL_GEN, REAL_HEAD, REAL_TOP = 32, 256, 64
REAL_TEXTS = {
    # Kimi-K3-0.40B was trained on one copypasta to a perplexity under 3: its
    # first prompt is that text, its second the README's, its third not.
    "Kimi-K3-0.40B": ["The FitnessGram Pacer Test is", "According to all known laws",
                      "The capital of France is"],
    # The same model continued on 59M tokens of Kazakh, at float16: two of
    # the prompts its own generations.json was written from.
    "Kimi-K3-0.40B-Kazakh-CPT-59M": ["Қазақстан —", "Қазақ тілі", "The FitnessGram Pacer Test is"],
    # inference-optimization/Kimi-K3-0.40B-MXFP4: the same model with its
    # experts (routed and shared) compressed-tensors MXFP4, the format the
    # Kimi-K3 release ships its routed experts in.
    "Kimi-K3-0.40B-MXFP4": ["The FitnessGram Pacer Test is", "According to all known laws",
                            "The capital of France is"],
}


def real_weights(d):
    """The checkpoint's language_model.* weights in float32. A
    compressed-tensors MXFP4 weight (weight_packed and weight_scale) is
    decompressed by compressed-tensors' own MXFP4PackedCompressor, as
    transformers loads it -- the weights only: the release's experts are
    W4A16, and a dynamic input-activation scheme a checkpoint also states is
    not applied (the engine's MXFP4 kernels run their own activation)."""
    from safetensors.torch import load_file
    sd = load_file(os.path.join(d, "model.safetensors"))
    c = json.load(open(os.path.join(d, "config.json")))
    q = c.get("quantization_config") or c["text_config"].get("quantization_config")
    if q and q.get("quant_method") == "compressed-tensors":
        from compressed_tensors.compressors import BaseCompressor
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
        assert q["format"] == "mxfp4-pack-quantized", q["format"]
        comp = BaseCompressor.get_value_from_registry(q["format"])
        (g,) = q["config_groups"].values()
        scheme = QuantizationScheme(targets=g["targets"], weights=QuantizationArgs(**g["weights"]))
        n = 0
        for k in [k for k in sd if k.endswith(".weight_packed")]:
            base = k.removesuffix("_packed")
            w = comp.decompress({"weight_packed": sd.pop(k), "weight_scale": sd.pop(base + "_scale")},
                                scheme)["weight"]
            sd[base] = w
            n += 1
        print(f"decompressed {n} MXFP4 weights with {type(comp).__name__}")
    return {k.removeprefix("language_model."): v.float() for k, v in sd.items()
            if k.startswith("language_model.")}


def real(name):
    from transformers import AutoTokenizer
    d = os.environ["K3_REAL"]
    out = os.environ.get("K3_OUT", OUT)
    texts = REAL_TEXTS[name.removeprefix("real-")]
    KimiLinearConfig, ml = load_remote(d)
    check_shims()
    tc = json.load(open(os.path.join(d, "config.json")))["text_config"]
    cfg = KimiLinearConfig(**tc)
    m = ml.KimiLinearForCausalLM(cfg).float().eval()
    cfg._attn_implementation = "eager"
    m.load_state_dict(real_weights(d), strict=True)
    # K3_ROUND=float16 runs the reference on the weights llama.cpp's converter
    # writes at --outtype f16 rounded to it and back -- every matrix but the
    # router's gate and the residual scores' projections (fused with their
    # norms into F32 vectors), the vectors and convolutions kept F32 -- so a
    # GGUF at that type is held to the function it actually stores.
    if rnd := os.environ.get("K3_ROUND"):
        keep = ("conv1d.weight", "block_sparse_moe.gate.weight", "res_proj.weight")
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.dim() >= 2 and n.endswith(".weight") and not n.endswith(keep):
                    p.copy_(p.to(getattr(torch, rnd)).float())
        name += "-" + rnd
    tk = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    seqs = []
    with torch.no_grad():
        for text in texts:
            prompt = tk(text).input_ids
            ids = list(prompt)
            for _ in range(REAL_GEN):
                ids.append(int(m(torch.tensor([ids]), use_cache=False).logits[0, -1].argmax()))
            lg = m(torch.tensor([ids[:-1]]), use_cache=False).logits[0]
            rows = []
            for r in lg:
                v, i = r.topk(REAL_TOP)
                rows.append({"argmax": int(r.argmax()), "max": float(r.max()),
                             "head": [round(float(x), 6) for x in r[:REAL_HEAD]],
                             "top": [int(x) for x in i], "topv": [round(float(x), 6) for x in v]})
            cont = ids[len(prompt):]
            print(f"{text!r} -> {tk.decode(cont)!r}")
            seqs.append({"text": text, "prompt": prompt, "cont": cont, "reply": tk.decode(cont),
                         "rows": rows})
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, name + ".json")
    g = {"dtype": "float32", "vocab": cfg.vocab_size, "seqs": seqs}
    if rnd:
        g["weights_rounded_to"] = rnd
    json.dump(g, open(dst, "w"))
    print("wrote", dst)


SPECS = {"synth-kimik3": build, "synth-kimik3-ragged": build}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(SPECS):
        (real if n.startswith("real-") else SPECS[n])(n)
