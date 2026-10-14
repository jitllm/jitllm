#!/usr/bin/env python3
"""Build the Qwen3.8-Flash-Next (qwen4exp) fixture and record transformers' logits.

    $JITLLM_HF_PY scripts/qwen4expgold.py [name ...]
    JITLLM_MODELS=/path/to/models LCPP_SRC=/path/to/llama.cpp ...

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/qwen4exp/<name>.json

c6gold.py's method: the oracle is transformers' OWN Qwen4ExpForCausalLM, the
GGUF is written by llama.cpp's OWN converter (conversion/qwen4exp.py), and
every parameter and every buffer is randomised -- the zero-centred norms, the
gated norm's weight, A_log and dt_bias, and the n-gram hash's multipliers
(buffers transformers derives from the seed; drawn here as odd integers under
the same bound, so the hash is not the class's default either).

The checkpoint is written in the published layout: the n-gram table split into
split_ngram_parts shards (`ple_embedding.ngram_embedding.shard_K.weight`),
which is what the converter reads, not transformers' single table.

Every feature sits inside the golden's 32 positions:
- four blocks, linear / full / linear / full (full_attention_interval 2), so
  both kinds and two QSA blocks run;
- QSA with a budget of 8 at a ratio of 4: two whole blocks plus the tail, so
  from position 12 each full-attention query reads a subset of its keys (the
  real file's budget of 2048 makes QSA dense under 2051 positions);
- PLE on block 2 (one-based 3, a linear block, as the reference requires),
  2-grams and 3-grams with two heads each, rows of 16 (ple_embed_dim is the
  hidden width: llama.cpp's loader assumes it), a dilated conv of
  kernel 4 at dilation 3;
- an EOS at position 10, so the n-gram window resets mid-sequence;
- the gated norm's gate is sigmoid (output_gate_type), the release's.

The indexer has eight heads, as DeepSeek V3.2's and V4's fixtures found: with
few heads blocks whose every ReLU is zero tie at the selection boundary.

The tokenizer is Qwen3.8-Flash-Next's own ($CV_TOK/qwen4exp).
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
OUT = os.path.join(HERE, "..", "testdata", "golden", "qwen4exp")
NPOS, NLOGIT = 32, 512
EOS, EOSPOS = 248044, 10
PARTS = 4
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
    "naïve café — 東京 🙂 \n\n\n end",
]

SHAPE = dict(
    vocab_size=248320, hidden_size=64, num_hidden_layers=4, full_attention_interval=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_num_key_heads=2, linear_num_value_heads=4,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
    norm_topk_prob=True, hc_count=4, hc_lowrank=32,
    ple_layer_ids=[3], ple_embed_dim=64, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
    ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=128, seed=4321, split_ngram_parts=PARTS,
    indexer_n_heads=8, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=8, indexer_compress_ratio=4,
    output_gate_type="sigmoid", hidden_act="silu", rms_norm_eps=1e-6, max_position_embeddings=4096,
    tie_word_embeddings=False, bos_token_id=EOS, eos_token_id=EOS, attention_bias=False,
)
ROPE = {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.5,
        "mrope_section": [3, 3, 2], "mrope_interleaved": True}


def q4e():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    cfg = Qwen4ExpTextConfig(**SHAPE, rope_parameters=dict(ROPE))
    return Qwen4ExpForCausalLM, cfg


SPECS = {
    "synth-qwen4exp": q4e,
}


def perturb(m, cfg):
    # Every parameter, its centre where the class has one: the zero-centred
    # RMSNorms (1 + w) at zero, the gated norm's weight at one, A_log and
    # dt_bias around the class's own draw.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("linear_attn.norm.weight"):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif "norm" in n or "layernorm" in n:
                p.copy_(torch.randn_like(p) * 0.3)
            elif n.endswith("A_log"):
                p.copy_(torch.rand_like(p) * 2 - 1)
            elif n.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif n.endswith("conv1d.weight"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif "ngram_embedding" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        g = torch.Generator().manual_seed(77)
        for n, b in m.named_buffers():
            if n.endswith("layer_multipliers"):
                bound = ((1 << 63) - 1) // cfg.vocab_size // 2
                r = torch.randint(1, bound, b.shape, generator=g, dtype=torch.long)
                b.copy_(2 * r + 1)


def save_published(m, cfg, d):
    out = {}
    for n, t in m.state_dict().items():
        if "rotary_emb" in n:
            continue
        if n.endswith("ple_embedding.ngram_embedding.weight"):
            base = n[: -len("weight")]
            rows = t.shape[0]
            assert rows % PARTS == 0, rows
            for k, sh in enumerate(t.split(rows // PARTS)):
                out[f"{base}shard_{k}.weight"] = sh.contiguous()
            continue
        out[n] = t.contiguous()
    save_file(out, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    c = cfg.to_dict()
    c.update({"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text",
              "torch_dtype": "float32", "full_attention_interval": SHAPE["full_attention_interval"],
              "layer_types": ["full_attention" if t != "linear_attention" else t for t in cfg.layer_types]})
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)


def build(name):
    cls, cfg = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261013)
    m = cls(cfg).float().eval()
    perturb(m, cfg)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    save_published(m, cfg, d)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(TOKDIR, "qwen4exp", f), d)
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.abspath(__file__), "--convert", d, dst], check=True)
    ids = [(i * 7919 + 13) % 50000 for i in range(NPOS)]
    ids[EOSPOS] = EOS
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
    """llama.cpp's own converter, reading config.json as published.

    The converter loads the config through AutoConfig, and this transformers
    renames layer_types' "full_attention" to "qwen_sparse_attention" on load;
    conversion/qwen4exp.py writes a compress ratio only where it reads
    "full_attention", so through AutoConfig every ratio is zero and
    llama.cpp's graph runs every attention block dense. The published GGUFs
    carry the ratios (unsloth's: 4 on every attention block), so the converter
    is handed the file's own config, which is what those were written from.
    """
    sys.path.insert(0, LCPP)
    sys.path.insert(1, os.path.join(LCPP, "gguf-py"))
    import convert_hf_to_gguf as lcpp
    from conversion.base import ModelBase

    def published(dir_model, is_mistral_format):
        with open(os.path.join(dir_model, "config.json"), encoding="utf-8") as f:
            return json.load(f)

    ModelBase.load_hparams = staticmethod(published)
    sys.argv = ["convert_hf_to_gguf.py", d, "--outtype", "f32", "--no-mtp", "--outfile", dst]
    lcpp.main()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--convert"]:
        convert(sys.argv[2], sys.argv[3])
    else:
        for n in sys.argv[1:] or list(SPECS):
            build(n)
