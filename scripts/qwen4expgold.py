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
MTPOUT = os.path.join(HERE, "..", "testdata", "golden", "mtp")  # TestMTPMatchesReference's
# Sixteen positions: QSA's budget of 8 at a ratio of 4 selects from position
# 12, so the block's own attention reads a subset of its keys at the last four.
MTPIDS = [(i * 7919 + 13) % 50000 for i in range(16)]
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


def q4e_ragged():
    # The hyper-connections' low rank 40, not a whole number of the
    # activation quantizer's 32-element blocks: the ragged-width gate's.
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    cfg = Qwen4ExpTextConfig(**dict(SHAPE, hc_lowrank=40, ple_embed_dim=64), rope_parameters=dict(ROPE))
    return Qwen4ExpForCausalLM, cfg


def q4e_mtp():
    return q4e()


SPECS = {
    "synth-qwen4exp": q4e,
    "synth-qwen4exp-ragged": q4e_ragged,
    "synth-qwen4exp-mtp": q4e_mtp,
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


def convert(d, dst, *mtp):
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
    # --mtp-full: the model with its prediction block, the converter's default.
    mtp = [a for a in mtp if a != "--mtp-full"]
    sys.argv = ["convert_hf_to_gguf.py", d, "--outtype", "f32", *mtp, "--outfile", dst]
    lcpp.main()


class MTP(torch.nn.Module):
    """The prediction block of the release (mtp.* in its checkpoint, which
    transformers drops on load and Qwen ships no code for), from transformers'
    own Qwen4Exp pieces, wired as llama.cpp's graph_mtp (src/models/
    qwen4exp.cpp) wires it -- RULE 7m: llama.cpp is the oracle of the wiring,
    transformers of every module in it:

        z_s    = fc_embedding(pre_fc_norm_embedding(embed(x_q)))
                 + fc_hidden(pre_fc_norm_hidden(h_{q-1})_s)       for each stream s
        z      = Qwen4ExpTextDecoderLayer(full attention, QSA)(z)  -- mtp.layers.0
        logits = lm_head(hyper_connection_mixer(z))

    h is the trunk's residual before its final mixer, every stream (the hc-wide
    h llama.cpp hands the block); pre_fc_norm_hidden is a grouped norm over the
    streams, as the release's [hc*hidden] weight says."""

    def __init__(self, cfg):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as q
        super().__init__()
        h, hc = cfg.hidden_size, cfg.hc_count
        self.fc_embedding = torch.nn.Linear(h, h, bias=False)
        self.fc_hidden = torch.nn.Linear(h, h, bias=False)
        self.pre_fc_norm_embedding = q.Qwen4ExpTextRMSNorm(h, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = q.Qwen4ExpTextRMSNorm(hc * h, group_size=h, eps=cfg.rms_norm_eps)
        d = cfg.to_dict()
        n = cfg.num_hidden_layers
        d["num_hidden_layers"] = n + 1
        d["layer_types"] = list(cfg.layer_types) + [
            next(t for t in cfg.layer_types if t != "linear_attention")]
        lc = type(cfg)(**{k: v for k, v in d.items() if k in type(cfg).__dataclass_fields__})
        lc._attn_implementation = "eager"
        self.layers = torch.nn.ModuleList([q.Qwen4ExpTextDecoderLayer(lc, n)])
        self.hyper_connection_mixer = q.Qwen4ExpTextGatedResidual(cfg, use_combine=False)


def build_mtp(name):
    """synth-qwen4exp with the prediction block: three GGUFs from llama.cpp's
    own converter -- the model with its block, the trunk alone (--no-mtp) and
    the block alone (--mtp, the shape the release's head is published in) --
    and the golden of the trunk's and the block's logits."""
    from transformers.masking_utils import create_causal_mask
    cls, cfg = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261015)
    m = cls(cfg).float().eval()
    mtp = MTP(cfg).float().eval()
    perturb(m, cfg)
    perturb(mtp, cfg)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    save_published(m, cfg, d)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(d, "model.safetensors"))
    for k, v in mtp.state_dict().items():
        sd["mtp." + k] = v.contiguous()
    save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    c = json.load(open(os.path.join(d, "config.json")))
    c["mtp_num_hidden_layers"] = 1
    json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(TOKDIR, "qwen4exp", f), d)
    for suffix, mode in (("", []), ("-trunk", ["--no-mtp"]), ("-head", ["--mtp"])):
        dst = os.path.join(MODELS, name + suffix + ".gguf")
        subprocess.run([PY, os.path.abspath(__file__), "--convert", d, dst, *(mode or ["--mtp-full"])],
                       check=True)
    ids = MTPIDS
    hs = {}
    hook = m.model.layers[-1].register_forward_hook(lambda mod, a, out: hs.__setitem__("h", out))
    with torch.no_grad():
        trunk = m(torch.tensor([ids]), use_cache=False).logits[0]
        hook.remove()
        h = hs["h"]  # [1, n, hc*hidden]: the residual before the final mixer
        hprev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        e = mtp.pre_fc_norm_embedding(m.model.embed_tokens(torch.tensor([ids])))
        hn = mtp.pre_fc_norm_hidden(hprev).unflatten(-1, (cfg.hc_count, cfg.hidden_size))
        z = (mtp.fc_embedding(e).unsqueeze(-2) + mtp.fc_hidden(hn)).flatten(-2)
        n = len(ids)
        pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
        pe = m.model.rotary_emb(z, pos)
        mask = torch.full((n, n), float("-inf")).triu(1).view(1, 1, n, n)
        z = mtp.layers[0](z, position_embeddings=pe, attention_mask=mask)
        logits = m.lm_head(mtp.hyper_connection_mixer(z))[0]
    os.makedirs(MTPOUT, exist_ok=True)
    rec = {"ids": ids, "nlogit": NLOGIT,
           "trunk": [[round(float(x), 7) for x in r[:NLOGIT]] for r in trunk],
           "mtp": [[round(float(x), 7) for x in r[:NLOGIT]] for r in logits],
           "mtp_argmax": [int(r.argmax()) for r in logits],
           "trunk_argmax": [int(r.argmax()) for r in trunk]}
    json.dump(rec, open(os.path.join(MTPOUT, name + ".json"), "w"))
    print("wrote", name, "mtp argmax", rec["mtp_argmax"])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--convert"]:
        convert(sys.argv[2], sys.argv[3], *(sys.argv[4:] or ["--no-mtp"]))
    else:
        for n in sys.argv[1:] or list(SPECS):
            (build_mtp if n == "synth-qwen4exp-mtp" else build)(n)
