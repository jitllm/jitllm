#!/usr/bin/env python3
"""The multi-token-prediction block's reference, and the fixtures that carry one.
-> testdata/golden/mtp/<name>.json, and fixtures under $JITLLM_MODELS

  $JITLLM_HF_PY scripts/mtpgold.py [name ...]

transformers loads a Qwen3.5 checkpoint and DROPS its mtp.* tensors
(_keys_to_ignore_on_load_unexpected), so no class of its own runs the block.
The block is assembled here from transformers' own pieces, wired as vLLM's
qwen3_next_mtp.py and llama.cpp's graph_mtp (src/models/qwen35.cpp) both wire
it:

    z      = fc(cat(pre_fc_norm_embedding(embed(x_q)), pre_fc_norm_hidden(h_{q-1})))
    z      = Qwen3_5DecoderLayer(full_attention)(z)   -- mtp.layers.0
    logits = lm_head(mtp.norm(z))

h is the trunk's last_hidden_state (after its final norm), and row q pairs the
token at q with the hidden state at q-1 -- zero at q = 0 -- at position q:
llama.cpp's pairing, the one the engine implements (engine/spec.go).

The synthetic fixtures are built by transformers' own classes with random
weights (random norms too, so a norm wired to the wrong input is an
uncorrelated vector, not a near-miss), in F32 so the reference and the engine
see the same numbers:

  synth-qwen35-hybrid-mtp  Qwen3_5ForCausalLM, linear and full layers: a
                           recurrent state to roll back. Written to GGUF by
                           llama.cpp's own convert_hf_to_gguf.py (the one that
                           exports nextn tensors).
  synth-qwen35moe-hybrid-mtp
                           Qwen3_5MoeForCausalLM, the same with every FFN a
                           routed mixture and a gated shared expert -- the
                           prediction block's too (qwen35moe, Qwen3.5-35B-A3B).
  synth-deepseek-mtp       DeepseekV3ForCausalLM with num_nextn_predict_layers 1:
                           MLA attention all the way down, so speculation's
                           rollback is positions alone. Its prediction block is
                           model.layers.3 of the safetensors, which the engine's
                           own converter reads; the block runs as vLLM's
                           deepseek_mtp.py runs it (the same wiring).

and Qwen3.5-0.8B is the real model (hf/Qwen3.5-0.8B under $JITLLM_MODELS),
whose golden is recorded from its bf16 weights widened to f32; the engine runs
its Q8_0 GGUF, so that gate is a bound, not an equality.
"""
import json, os, shutil, subprocess, sys
import torch
from safetensors.torch import save_file, load_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q35moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("JITLLM_MODELS", os.path.join(HERE, "..", "models"))
OUT = os.path.join(HERE, "..", "testdata", "golden", "mtp")
LLAMACPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
DONOR = os.path.join(MODELS, "hf", "Qwen3.5-0.8B")
IDS = [760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338]
NLOGIT = 256

HYBRID = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
# name -> (layer kinds, mixture): a mixture fixture is Qwen3_5MoeForCausalLM.
SYNTH = {
    "synth-qwen35-hybrid-mtp": (HYBRID, False),
    "synth-qwen35moe-hybrid-mtp": (HYBRID, True),
}
# synth-deepseek's recipe in scripts/hfgold.py, with the prediction block.
DEEPSEEK = dict(
    hidden_size=64, intermediate_size=96, num_attention_heads=4, max_position_embeddings=512,
    initializer_range=0.1, tie_word_embeddings=False,
    q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    head_dim=8, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
    moe_intermediate_size=32, first_k_dense_replace=1, n_group=2, topk_group=1,
    routed_scaling_factor=2.5, norm_topk_prob=True, scoring_func="sigmoid",
    num_hidden_layers=3, hidden_act="silu", num_key_value_heads=4,
)
DEEPSEEK_DONOR = "qwen2.5-tiny-random"



def synth_config(layers, moe):
    kw = dict(moe_intermediate_size=32, shared_expert_intermediate_size=64, num_experts=8,
              num_experts_per_tok=2) if moe else dict(intermediate_size=128)
    return (Qwen3_5MoeTextConfig if moe else Qwen3_5TextConfig)(
        vocab_size=248320, hidden_size=64, **kw,
        num_hidden_layers=len(layers), num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, layer_types=layers, tie_word_embeddings=True,
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True},
        rms_norm_eps=1e-6, max_position_embeddings=4096,
        eos_token_id=248044,
    )


def classes(cfg):
    """The modeling module, the decoder layer and the config class for cfg."""
    if isinstance(cfg, Qwen3_5MoeTextConfig):
        return q35moe, q35moe.Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig
    return q35, q35.Qwen3_5DecoderLayer, Qwen3_5TextConfig


def mtp_layer_config(cfg):
    """The block's config: the trunk's, with its one layer full attention."""
    _, _, C = classes(cfg)
    d = cfg.to_dict()
    d["num_hidden_layers"] = 1
    d["layer_types"] = ["full_attention"]
    return C(**{k: v for k, v in d.items() if k in C.__dataclass_fields__})


class MTP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        _, layer, _ = classes(cfg)
        h = cfg.hidden_size
        self.fc = torch.nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = q35.Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = q35.Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.layers = torch.nn.ModuleList([layer(mtp_layer_config(cfg), 0)])
        self.norm = q35.Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)


def saved(model, d):
    """model's tensors as save_pretrained writes them: a mixture's fused expert
    banks become the per-expert names a checkpoint carries."""
    model.save_pretrained(d)
    sd = {}
    for fn in os.listdir(d):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(d, fn)))
            os.remove(os.path.join(d, fn))
    return sd


def build_synth(name, layers, moe):
    torch.manual_seed(sum(map(ord, name)))
    cfg = synth_config(layers, moe)
    mod, _, _ = classes(cfg)
    model = (mod.Qwen3_5MoeForCausalLM if moe else mod.Qwen3_5ForCausalLM)(cfg).float().eval()
    mtp = MTP(cfg).float().eval()
    with torch.no_grad():
        for mod in (model, mtp):
            for n, p in mod.named_parameters():
                if n.endswith("A_log"):
                    p.copy_(torch.rand_like(p) * 0.5)
                elif "norm" in n or n.endswith("dt_bias"):
                    p.copy_(torch.randn_like(p) * 0.2)
                else:
                    p.copy_(torch.randn_like(p) * 0.08)
    d = os.path.join(MODELS, name)
    os.makedirs(d, exist_ok=True)
    sd = {k: v.contiguous() for k, v in saved(model, d).items() if k != "lm_head.weight"}
    # The block's layer as a checkpoint names it: a one-layer model of the
    # block's config holding it, saved the same way.
    holder = type(model)(mtp_layer_config(cfg)).float()
    holder.model.layers[0] = mtp.layers[0]
    for k, v in saved(holder, d).items():
        if k.startswith("model.layers.0."):
            sd["mtp.layers.0." + k[len("model.layers.0."):]] = v.contiguous()
    for k, v in mtp.state_dict().items():
        if not k.startswith("layers."):
            sd["mtp." + k] = v.contiguous()
    save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    c = cfg.to_dict()
    c["architectures"] = ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"]
    c["model_type"] = "qwen3_5_moe_text" if moe else "qwen3_5_text"
    c["mtp_num_hidden_layers"] = 1
    c["torch_dtype"] = "float32"
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(c, f, indent=1)
    for t in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]:
        shutil.copy(os.path.join(DONOR, t), d)
    gguf = os.path.join(MODELS, name + ".gguf")
    subprocess.run([sys.executable, os.path.join(LLAMACPP, "convert_hf_to_gguf.py"), d,
                    "--outtype", "f32", "--outfile", gguf], check=True)
    if moe:
        # And in the other float formats and quantized: a device's batched
        # mixture runs each bank format through its own form of the grouped
        # matvec (engine/groupedmoe_test.go).
        for t in ["f16", "bf16", "q8_0"]:
            subprocess.run([sys.executable, os.path.join(LLAMACPP, "convert_hf_to_gguf.py"), d,
                            "--outtype", t, "--outfile", os.path.join(MODELS, name + "-" + t + ".gguf")], check=True)
    return model, mtp, cfg


class DeepseekMTP(torch.nn.Module):
    """vLLM's DeepSeekMultiTokenPredictorLayer, from transformers' pieces."""

    def __init__(self, cfg, idx):
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds
        super().__init__()
        h = cfg.hidden_size
        self.enorm = ds.DeepseekV3RMSNorm(h, eps=cfg.rms_norm_eps)
        self.hnorm = ds.DeepseekV3RMSNorm(h, eps=cfg.rms_norm_eps)
        self.eh_proj = torch.nn.Linear(2 * h, h, bias=False)
        self.block = ds.DeepseekV3DecoderLayer(cfg, idx)
        self.head_norm = ds.DeepseekV3RMSNorm(h, eps=cfg.rms_norm_eps)


def build_deepseek(name):
    from transformers import AutoConfig, AutoModelForCausalLM
    from tokenizers import Tokenizer
    donor = os.path.join(MODELS, DEEPSEEK_DONOR)
    kw = dict(DEEPSEEK, vocab_size=Tokenizer.from_file(os.path.join(donor, "tokenizer.json")).get_vocab_size())
    cfg = AutoConfig.for_model("deepseek_v3", **kw)
    torch.manual_seed(sum(map(ord, name)))
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    n = cfg.num_hidden_layers
    # The block is block n of a model one block longer: its layer index decides
    # the mixture (first_k_dense_replace), as the checkpoint's index does.
    mcfg = AutoConfig.for_model("deepseek_v3", **dict(kw, num_hidden_layers=n + 1))
    mtp = DeepseekMTP(mcfg, n).float().eval()
    with torch.no_grad():
        for mod in (model, mtp):
            for pn, p in mod.named_parameters():
                if p.dim() == 1:
                    p.copy_(torch.randn_like(p) * 0.5 + 1)
                elif mod is mtp:
                    # A standalone module's fused expert banks are torch.empty:
                    # everything is drawn here, at the trunk's scale.
                    p.copy_(torch.randn_like(p) * cfg.initializer_range)
            for bn, b in mod.named_buffers():
                if bn.endswith("e_score_correction_bias"):
                    b.copy_(torch.randn_like(b) * 0.5)
    d = os.path.join(MODELS, name)
    # save_pretrained writes the checkpoint's own names (one matrix per
    # expert); the block is added beside them in the same spelling.
    model.save_pretrained(d)
    sd = dict(load_file(os.path.join(d, "model.safetensors")))
    pfx = f"model.layers.{n}."
    inter = mcfg.moe_intermediate_size
    for k, v in mtp.block.state_dict().items():
        if k == "mlp.experts.gate_up_proj":
            for e in range(v.shape[0]):
                sd[pfx + f"mlp.experts.{e}.gate_proj.weight"] = v[e, :inter].contiguous()
                sd[pfx + f"mlp.experts.{e}.up_proj.weight"] = v[e, inter:].contiguous()
        elif k == "mlp.experts.down_proj":
            for e in range(v.shape[0]):
                sd[pfx + f"mlp.experts.{e}.down_proj.weight"] = v[e].contiguous()
        else:
            sd[pfx + k] = v.contiguous()
    sd[pfx + "eh_proj.weight"] = mtp.eh_proj.weight.contiguous()
    sd[pfx + "enorm.weight"] = mtp.enorm.weight.contiguous()
    sd[pfx + "hnorm.weight"] = mtp.hnorm.weight.contiguous()
    sd[pfx + "shared_head.norm.weight"] = mtp.head_norm.weight.contiguous()
    # The copies every published checkpoint carries, byte for byte the trunk's.
    sd[pfx + "shared_head.head.weight"] = sd["lm_head.weight"].clone()
    sd[pfx + "embed_tokens.weight"] = sd["model.embed_tokens.weight"].clone()
    save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    c = cfg.to_dict()
    c["architectures"] = ["DeepseekV3ForCausalLM"]
    c["num_nextn_predict_layers"] = 1
    c["torch_dtype"] = "float32"
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(c, f, indent=1)
    for t in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(donor, t)
        if os.path.exists(src):
            shutil.copy(src, d)
    return model, mtp


@torch.no_grad()
def golden_deepseek(name, model, mtp, ids):
    ids_t = torch.tensor([ids])
    pos = torch.arange(len(ids)).view(1, -1)
    h = model.model(input_ids=ids_t, use_cache=False).last_hidden_state
    trunk_logits = model.lm_head(h)[0]
    hprev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
    e = model.model.embed_tokens(ids_t)
    z = mtp.eh_proj(torch.cat([mtp.enorm(e), mtp.hnorm(hprev)], dim=-1))
    pe = model.model.rotary_emb(z, pos)
    mask = torch.full((len(ids), len(ids)), float("-inf")).triu(1).view(1, 1, len(ids), len(ids))
    z = mtp.block(z, position_embeddings=pe, attention_mask=mask, position_ids=pos)
    mtp_logits = model.lm_head(mtp.head_norm(z))[0]
    write(name, ids, trunk_logits, mtp_logits)


def load_real(d):
    with open(os.path.join(d, "config.json")) as f:
        full = json.load(f)
    tc = full.get("text_config", full)
    keep = {k: v for k, v in tc.items() if k in Qwen3_5TextConfig.__dataclass_fields__}
    cfg = Qwen3_5TextConfig(**keep)
    model = q35.Qwen3_5ForCausalLM(cfg).float().eval()
    mtp = MTP(cfg).float().eval()
    sd = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(d, fn)))
    trunk, head = {}, {}
    for k, v in sd.items():
        v = v.float()
        if k.startswith("mtp."):
            head[k[len("mtp."):]] = v
        elif k.startswith("model.language_model."):
            trunk["model." + k[len("model.language_model."):]] = v
        elif k == "lm_head.weight":
            trunk[k] = v
    if "lm_head.weight" not in trunk:
        trunk["lm_head.weight"] = trunk["model.embed_tokens.weight"]
    miss = model.load_state_dict(trunk, strict=False)
    if miss.missing_keys:
        raise SystemExit(f"trunk keys missing: {miss.missing_keys[:5]}")
    miss = mtp.load_state_dict(head, strict=True)
    return model, mtp, cfg


def write(name, ids, trunk_logits, mtp_logits):
    os.makedirs(OUT, exist_ok=True)
    rec = {
        "ids": ids, "nlogit": NLOGIT,
        "trunk": [r[:NLOGIT].tolist() for r in trunk_logits],
        "mtp": [r[:NLOGIT].tolist() for r in mtp_logits],
        "mtp_argmax": [int(r.argmax()) for r in mtp_logits],
        "trunk_argmax": [int(r.argmax()) for r in trunk_logits],
    }
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(rec, f)
    print(name, "trunk argmax", rec["trunk_argmax"], "mtp argmax", rec["mtp_argmax"])


@torch.no_grad()
def golden(name, model, mtp, cfg):
    ids = torch.tensor([IDS])
    pos = torch.arange(len(IDS)).view(1, -1)
    out = model.model(input_ids=ids, use_cache=False)
    h = out.last_hidden_state  # after the trunk's final norm
    trunk_logits = model.lm_head(h)[0]
    # Row q: the token at q and the trunk's hidden state at q-1, zero at 0.
    hprev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
    e = model.model.embed_tokens(ids)
    z = mtp.fc(torch.cat([mtp.pre_fc_norm_embedding(e), mtp.pre_fc_norm_hidden(hprev)], dim=-1))
    pe = model.model.rotary_emb(z, pos.view(1, 1, -1).expand(3, 1, -1))
    mask = torch.full((len(IDS), len(IDS)), float("-inf")).triu(1).view(1, 1, len(IDS), len(IDS))
    z = mtp.layers[0](z, position_embeddings=pe, attention_mask=mask, position_ids=pos)
    g = mtp.norm(z)
    mtp_logits = model.lm_head(g)[0]
    write(name, IDS, trunk_logits, mtp_logits)


def main(names):
    for name in names or list(SYNTH) + ["synth-deepseek-mtp", "Qwen3.5-0.8B"]:
        if name in SYNTH:
            golden(name, *build_synth(name, *SYNTH[name]))
        elif name == "synth-deepseek-mtp":
            # The donor's vocabulary is small; ids past it would be refused.
            golden_deepseek(name, *build_deepseek(name), [i % 1000 for i in IDS])
        else:
            golden(name, *load_real(os.path.join(MODELS, "hf", name)))


if __name__ == "__main__":
    main(sys.argv[1:])
