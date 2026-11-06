#!/usr/bin/env python3
"""Build the DeepSeek V4 fixture and record transformers' logits.

    $JITLLM_HF_PY scripts/ds4gold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/deepseek4/<name>.json

c6gold.py's method: the oracle is transformers' OWN DeepseekV4ForCausalLM,
the GGUF is written by llama.cpp's OWN converter, and every parameter and
every buffer that is a weight is randomised -- the router's
e_score_correction_bias and the hash layers' tid2eid are buffers transformers
initialises to zero, the hyper-connections' base is zero and their scale one,
and a zero position bias or sink is the identity.

llama.cpp's converter reads the checkpoint as DeepSeek publishes it
(layers.N.attn.wq_a, ffn.experts.E.w1 with an MXFP4 payload and an E8M0
scale, hc_attn_fn, ...), which transformers renames on load and does not
write back; save_original() writes that layout from the class's own tensors.
The routed experts are MXFP4 in the published checkpoint and the converter
only repacks them, so the fixture's experts are drawn as MXFP4 codes and
scales and transformers runs their exact values; the GGUF carries those
values as F32 banks (convert() says why).

Every feature is inside the golden's 32 positions: a sliding window of 8 on
every block; a CSA block (rate 4, its overlapping windows and a lightning
indexer keeping the top 3 of up to 8 entries, so from position 15 on each
query reads a subset), an HCA block (rate 8, up to 4 entries), a sliding-only
block, and a second CSA block; the first two blocks route by the token-id
table, the rest by the sqrt-softplus scores with the selection bias; the
compressed blocks rotate with YaRN at their own base (original context 64, so
the interpolation reaches most of the rotated pairs) and the sliding block
with the plain one. The experts' and the shared expert's weights are wide
enough that the SwiGLU clamp (10) bites on a good share of their gate values.

The indexer has eight heads: an entry scores exactly zero when every head's
ReLU is zero, and with two heads such ties sat at the selection boundary,
where torch's top-k and the engine's order them differently (DeepSeek V3.2's
fixture found the same).

The tokenizer is DeepSeek-V4-Flash's own ($CV_TOK/deepseek-v4).
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "deepseek4")
NPOS, NLOGIT = 32, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "naïve café — 東京 🙂 \n\n\n end",
]

RATIOS = [4, 8, 0, 4]
NHASH = 2
TYPES = {0: "sliding_attention", 4: "compressed_sparse_attention", 8: "heavily_compressed_attention"}
YARN = {"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 64,
        "beta_fast": 32.0, "beta_slow": 1.0}

# DeepSeek V4's shape at toy size: four hyper-connection streams, MQA at
# head_dim 32 with a 16-wide interleaved rotary on each head's tail, a query
# latent of 32, two output groups of rank 32, a top-2 of 8 MXFP4 experts with
# one shared expert, an eight-head indexer at 32. Every matrix's input width is a
# multiple of 32 (the device's activation quantizer and MXFP4's block).
SHAPE = dict(vocab_size=129280, hidden_size=64, moe_intermediate_size=32, num_hidden_layers=4,
             num_attention_heads=4, num_key_value_heads=1, head_dim=32, q_lora_rank=32,
             num_experts_per_tok=2, n_routed_experts=8, n_shared_experts=1,
             scoring_func="sqrtsoftplus", norm_topk_prob=True, routed_scaling_factor=1.5,
             max_position_embeddings=4096, rope_theta=10000.0, compress_rope_theta=160000.0,
             hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, swiglu_limit=10.0, sliding_window=8,
             o_groups=2, o_lora_rank=32, index_n_heads=8, index_head_dim=32, index_topk=3,
             rms_norm_eps=1e-6, tie_word_embeddings=False, bos_token_id=0, eos_token_id=1)
PRF = 0.5

# E2M1, the MXFP4 code's value (ggml's kvalues_mxfp4 halved).
FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def v4(ratios=(4, 8, 0, 4)):
    global RATIOS
    RATIOS = list(ratios)
    from transformers import DeepseekV4ForCausalLM
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    cfg = DeepseekV4Config(
        **SHAPE, partial_rotary_factor=PRF, rope_parameters=dict(YARN),
        layer_types=[TYPES[r] for r in RATIOS],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        mlp_layer_types=["hash_moe"] * NHASH + ["moe"] * (len(RATIOS) - NHASH),
        num_nextn_predict_layers=0)
    return DeepseekV4ForCausalLM, cfg


SPECS = {
    "synth-deepseek4": v4,
    # Every block sliding: the hyper-connections and the token-routed
    # experts without a compressed history.
    "synth-deepseek4-sliding": lambda: v4((0, 0, 0, 0)),
}


def mxfp4(rows, cols, spread):
    """Random MXFP4 weights: (codes packed two to a byte, E8M0 scales, values)."""
    codes = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
    exps = torch.randint(127 - 3 + spread, 127 + spread, (rows, cols // 32), dtype=torch.uint8)
    vals = FP4[codes.long()] * torch.exp2(exps.float() - 127).repeat_interleave(32, dim=1)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), exps.contiguous(), vals


def perturb(m, cfg):
    # c6gold.perturb's classification (a norm is a parameter of a norm module),
    # the hyper-connections' scale around one, and every buffer that is a
    # weight: the selection bias and the token-id table, whose rows are
    # distinct experts as the published table's are.
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith(("attn_hc.scale", "ffn_hc.scale", "hc_head.hc_scale")):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith(("attn_hc.base", "ffn_hc.base", "hc_head.hc_base", "sinks", "position_bias")):
                p.copy_(torch.randn_like(p) * 0.5)
            elif "shared_experts" in n:
                # Wide enough that the clamp bites (see the module docstring).
                p.copy_(torch.randn_like(p) * 1.5)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.5)
            elif n.endswith("tid2eid"):
                e = cfg.n_routed_experts
                b.copy_(torch.stack([torch.randperm(e)[:b.shape[1]] for _ in range(b.shape[0])]))


def experts(m, cfg):
    """Draw every routed expert as MXFP4 and give transformers its values;
    return the original layout's tensors."""
    out = {}
    d, i = cfg.hidden_size, cfg.moe_intermediate_size
    with torch.no_grad():
        for li, layer in enumerate(m.model.layers):
            ex = layer.mlp.experts
            for e in range(cfg.n_routed_experts):
                ws = {}
                for w, (rows, cols) in (("w1", (i, d)), ("w3", (i, d)), ("w2", (d, i))):
                    packed, scale, vals = mxfp4(rows, cols, 1)
                    out[f"layers.{li}.ffn.experts.{e}.{w}.weight"] = packed
                    out[f"layers.{li}.ffn.experts.{e}.{w}.scale"] = scale
                    ws[w] = vals
                ex.gate_up_proj[e].copy_(torch.cat([ws["w1"], ws["w3"]], dim=0))
                ex.down_proj[e].copy_(ws["w2"])
    return out


# transformers' names back to DeepSeek's (transformers' conversion_mapping
# for deepseek_v4, inverted: pure renames, no reordering).
LEAF = [
    (".self_attn.compressor.indexer.scorer.weights_proj.", ".attn.indexer.weights_proj."),
    (".self_attn.compressor.indexer.q_b_proj.", ".attn.indexer.wq_b."),
    (".self_attn.compressor.indexer.", ".attn.indexer.compressor."),
    (".self_attn.compressor.", ".attn.compressor."),
    (".self_attn.", ".attn."),
    (".mlp.", ".ffn."),
    (".input_layernorm.", ".attn_norm."),
    (".post_attention_layernorm.", ".ffn_norm."),
    (".q_a_proj.", ".wq_a."),
    (".q_b_proj.", ".wq_b."),
    (".q_a_norm.", ".q_norm."),
    (".kv_proj.", ".wkv."),
    (".gate_proj.", ".wgate."),
    (".o_a_proj.", ".wo_a."),
    (".o_b_proj.", ".wo_b."),
]


def rename(n):
    if n == "model.embed_tokens.weight":
        return "embed.weight"
    if n == "model.norm.weight":
        return "norm.weight"
    if n == "lm_head.weight":
        return "head.weight"
    if n.startswith("model.hc_head."):
        return {"hc_fn": "hc_head_fn", "hc_base": "hc_head_base", "hc_scale": "hc_head_scale"}[n.split(".")[-1]]
    n = "." + n[len("model."):]
    for a, b in (("attn_hc.fn", "hc_attn_fn"), ("attn_hc.base", "hc_attn_base"),
                 ("attn_hc.scale", "hc_attn_scale"), ("ffn_hc.fn", "hc_ffn_fn"),
                 ("ffn_hc.base", "hc_ffn_base"), ("ffn_hc.scale", "hc_ffn_scale")):
        if n.endswith("." + a):
            return n[1:-len(a)] + b
    if n.endswith(".self_attn.sinks"):
        return n[1:-len("self_attn.sinks")] + "attn.attn_sink"
    if n.endswith(".position_bias"):
        n = n[:-len("position_bias")] + "ape"
    if n.endswith(".kv_norm.weight") and ".compressor." in n:
        n = n[:-len("kv_norm.weight")] + "norm.weight"
    if n.endswith(".mlp.gate.e_score_correction_bias"):
        n = n[:-len("e_score_correction_bias")] + "bias"
    for a, b in (("shared_experts.gate_proj.", "shared_experts.w1."),
                 ("shared_experts.down_proj.", "shared_experts.w2."),
                 ("shared_experts.up_proj.", "shared_experts.w3.")):
        n = n.replace(a, b)
    for a, b in LEAF:
        n = n.replace(a, b)
    return n[1:]


def save_original(m, cfg, d, exps):
    out = dict(exps)
    for n, t in m.state_dict().items():
        if ".mlp.experts." in n or "rotary_emb" in n:
            continue
        out[rename(n)] = t.contiguous()
    save_file(out, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    c = {k: v for k, v in SHAPE.items()}
    c.update({
        "architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
        "hidden_act": "silu", "torch_dtype": "float32", "num_nextn_predict_layers": 0,
        "qk_rope_head_dim": int(SHAPE["head_dim"] * PRF), "compress_ratios": RATIOS,
        "num_hash_layers": NHASH, "expert_dtype": "fp4",
        # transformers' own keys beside the legacy ones: its config maps a
        # legacy compress_ratios entry only from {0, 4, 128}, and the
        # converter's tokenizer load goes through AutoConfig.
        "layer_types": [TYPES[r] for r in RATIOS],
        "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        "mlp_layer_types": ["hash_moe"] * NHASH + ["moe"] * (len(RATIOS) - NHASH),
        "rope_scaling": {"type": "yarn", "factor": YARN["factor"],
                         "original_max_position_embeddings": YARN["original_max_position_embeddings"],
                         "beta_fast": YARN["beta_fast"], "beta_slow": YARN["beta_slow"]},
    })
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


def build(name):
    cls, cfg = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = cls(cfg).float().eval()
    perturb(m, cfg)
    exps = experts(m, cfg)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    save_original(m, cfg, d, exps)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(TOKDIR, "deepseek-v4", f), d)
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.abspath(__file__), "--convert", d, dst], check=True)
    ids = [(i * 7919 + 13) % min(cfg.vocab_size, 50000) for i in range(NPOS)]
    with torch.no_grad():
        lg = m(torch.tensor([ids]), use_cache=False).logits[0]
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
    tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"ids": ids, "vocab": cfg.vocab_size, "pos": rows, "texts": texts},
              open(os.path.join(OUT, name + ".json"), "w"))
    print("wrote", dst, "argmax", [r["argmax"] for r in rows])


def convert(d, dst):
    """llama.cpp's own converter, with the routed experts written as the F32
    values of their MXFP4 codes rather than repacked.

    Everything else -- the names, the keys, the hash tables, the shared
    expert -- is the converter's. The engine's MXFP4 kernels quantize the
    activation to int8 (gpt-oss's gate measures them), which on a model of
    discrete selections (the indexer's top-k, the router's) would put a flip
    at every near tie; F32 banks keep the comparison on the graph, at the
    bound every other fixture of this kind holds.
    """
    sys.path.insert(0, LCPP)
    sys.path.insert(1, os.path.join(LCPP, "gguf-py"))
    import numpy as np
    import convert_hf_to_gguf as lcpp
    from conversion.base import LazyTorchTensor
    from conversion.deepseek import DeepseekV4Model

    def f32(self, bid, proj, key):
        n = self.hparams["n_routed_experts"]
        data, consumed = None, []
        for eid in range(n):
            w = f"layers.{bid}.ffn.experts.{eid}.{proj}.weight"
            s = f"layers.{bid}.ffn.experts.{eid}.{proj}.scale"
            packed = LazyTorchTensor.to_eager(self.model_tensors[w]()).view(torch.uint8)
            scale = LazyTorchTensor.to_eager(self.model_tensors[s]()).view(torch.uint8)
            codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
            vals = FP4[codes.long()] * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=1)
            if data is None:
                data = np.empty((n, *vals.shape), dtype=np.float32)
            data[eid] = vals.numpy()
            consumed.extend((w, s))
        self.gguf_writer.add_tensor(self.format_tensor_name(key, bid), data)
        return consumed

    DeepseekV4Model._write_mxfp4_expert_tensor = f32
    sys.argv = ["convert_hf_to_gguf.py", d, "--outtype", "f32", "--outfile", dst]
    lcpp.main()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--convert"]:
        convert(sys.argv[2], sys.argv[3])
    else:
        for n in sys.argv[1:] or list(SPECS):
            build(n)
