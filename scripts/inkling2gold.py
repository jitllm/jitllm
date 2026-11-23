#!/usr/bin/env python3
"""Build the Inkling 2 fixture and record transformers' logits.

    $JITLLM_HF_PY scripts/inkling2gold.py [real-index.json]
    JITLLM_MODELS=/path/to/models ... (where the fixture goes)
    CV_TOK=... (default $JITLLM_MODELS/tok; the tokenizer is tok/inkling2)

-> $JITLLM_MODELS/synth-hf-inkling2/ and testdata/golden/inkling2/synth-hf-inkling2.json

The oracle is transformers' OWN Inkling2LoopForConditionalGeneration, the
class the published checkpoint names, built at toy size with a one-block
tower the text golden never reaches (synth-hf-qwen35's precedent). Every
parameter and every buffer is randomised: the class initialises the
attention-residual and message queries to zero (a uniform mix), the clip
scales to one, the dynamic convolutions' static taps to zero and the router's
selection bias to zero, and each of those is the identity of its feature.

save_pretrained writes the checkpoint's own layout (model.llm.*, w13 with
gate and up rows interleaved, the unembedding padded to vocab_size, the
message queries as the two middle_cycle_cross_pass tensors, attn_queries.i
for layer i + 1) through transformers' reversible conversion mapping, and the
golden is taken from the fixture READ BACK, so it is the layout on disk that
the logits belong to. Given the real checkpoint's model.safetensors.index.json,
every text tensor name is checked against its names. config.json is rewritten
in the published file's spelling (loop_config, sconv_kernel_size,
intermediate_size, model_max_length), and read back must equal the class's.

Every feature is inside the golden's 40 positions: a sliding window of 8 (the
sliding layers' relative extent too) and a full-layer relative extent of 16;
the log scaling's floor at 12, so queries and relative states past position
11 scale; two layers per attention-residual block, so the block states stack
to five vectors; layers 2..5 looped over two passes, so the second pass reads
messages from two of the first pass's block states; sliding and full layers
in every one of the four layer classes, at a block boundary and inside one;
GQA 4:2 in the sliding layers and 4:1 in the full ones; a convolution of 3
taps (the published 4 would hide a hardcoded width); a sigmoid top-2 of 8
with the selection bias and a routed scale; and clip thresholds and softcaps
set where they bite. The script prints what share of each clip and softcap
bit, and refuses a fixture where one never does or always does.

Stored bf16 as the published file is, the class's float32 modules float32;
the golden runs in float32 on the widened weights, as hfgold.py does.

The tokenizer is Inkling 2's own ($CV_TOK/inkling2).
"""
import json
import os
import re
import shutil
import sys

import torch
from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
NAME = "synth-hf-inkling2"
OUT = os.path.join(HERE, "..", "testdata", "golden", "inkling2")
NPOS, NLOGIT = 40, 256
PROMPT, NEW = 20, 12
TEXTS = [
    "The capital of France is",
    " leading space and  double  spaces",
    "line one\nline two\ttab",
    "café naïve über 12345 \U0001F600 中文",
]
TOKFILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.jinja", "generation_config.json"]

S, F = "hybrid_sliding", "hybrid"
TEXT = dict(
    vocab_size=201024, unpadded_vocab_size=200058, hidden_size=64, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=1, head_dim=16,
    swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
    sliding_window_size=8, d_rel=4, rel_extent=16, log_scaling_n_floor=12, log_scaling_alpha=0.3,
    layer_types=[S, F, F, S, S, F, S, F],
    rms_norm_eps=1e-6, conv_kernel_size=3, moe_intermediate_size=32, n_routed_experts=8,
    num_experts_per_tok=2, route_scale=2.5, qk_l2_clip_threshold=6.0, rel_l2_clip_threshold=3.0,
    n_layers_per_attn_res_block=2, mlp_gate_softcap=1.5, mlp_up_softcap=2.0,
    final_logit_softcapping=3.0, scoring_func="sigmoid",
    loop_layer_index_start=2, loop_layer_index_end=6, loop_num_passes=2, loop_gate_rank=8,
    max_position_embeddings=4096, eos_token_id=200006, pad_token_id=None)
VISION = dict(text_hidden_size=64, temporal_patch_size=2, native_patch_size=4, adapter_merge_size=2,
              projection_hidden_dim=32, projector_bias=False, hidden_size=32, num_hidden_layers=1,
              num_attention_heads=2, mlp_ratio=2.0, posemb_grid_size=4, layer_norm_eps=1e-5)

# The class's names for its keys, and the published config's (the class's
# attribute_map): the fixture's config.json is in the published spelling.
SPELLING = {"conv_kernel_size": "sconv_kernel_size", "moe_intermediate_size": "intermediate_size",
            "max_position_embeddings": "model_max_length"}


def build():
    from transformers import Inkling2LoopConfig, Inkling2LoopForConditionalGeneration
    cfg = Inkling2LoopConfig(text_config=dict(TEXT), vision_config=dict(VISION))
    cfg._attn_implementation = "eager"
    cfg.get_text_config()._attn_implementation = "eager"
    torch.manual_seed(20261123)
    return Inkling2LoopForConditionalGeneration(cfg).float().eval(), cfg


def perturb(m):
    # c6gold.perturb's classification (a norm is a parameter of a norm module,
    # around one), the clip scales around one, the residual and message
    # queries wide enough that their softmax over at most five states is far
    # from uniform, and the router's selection bias, a buffer.
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith(".norm.scale"):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith(".query"):
                p.copy_(torch.randn_like(p) * 2.0)
            elif n.endswith("relative_position_bias.proj"):
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        nb = 0
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.5)
                nb += 1
            elif n.endswith(("rotary_emb.inv_freq", "rotary_emb.original_inv_freq")):
                continue
            elif b.is_floating_point() and b.numel() > 0:
                sys.exit(f"buffer {n} is not randomised: say what it is before building on it")
        if nb == 0:
            sys.exit("no selection bias found: the router's buffer moved")


def fp32_kept(m, n):
    keep = getattr(m, "_keep_in_fp32_modules_strict", None) or []
    return any(re.search(r"(^|\.)" + re.escape(k) + r"($|\.)", n) for k in keep)


def to_bf16(m):
    """Round every weight to bf16 but the class's float32 modules; the model
    stays float32, so the golden is the stored values widened."""
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not fp32_kept(m, n):
                p.copy_(p.to(torch.bfloat16).float())


def stats(m, ids):
    """The share of each clip and softcap that bit over the golden's ids."""
    from transformers.models.inkling2_loop import modeling_inkling2_loop as mod
    seen = {}

    def note(key, hit, tot):
        a, b = seen.get(key, (0, 0))
        seen[key] = (a + hit, b + tot)

    def clip(key):
        def hook(mod_, args, out):
            x = args[0].float()
            if hasattr(mod_, "scale"):
                x = x * mod_.scale.float()
            nrm = torch.linalg.vector_norm(x, dim=-1)
            note(key, int((nrm > mod_.threshold if hasattr(mod_, "threshold") else nrm > 0).sum()), nrm.numel())
        return hook

    def cap(key):
        def hook(mod_, args, out):
            x = args[0].float().abs() / mod_.softcap
            note(key, int((x > 0.5).sum()), x.numel())
        return hook

    hs = []
    for n, sub in m.named_modules():
        if isinstance(sub, mod.Inkling2LoopScaledL2Clip):
            key = "rel clip" if "relative_position_bias" in n else "q clip"
            hs.append(sub.register_forward_hook(clip(key)))
        elif isinstance(sub, mod.Inkling2LoopL2Clip):
            if "relative_position_bias" in n:
                key = "rel clip (log)"
            elif "q_proj" in n:
                key = "q clip (log)"
            else:
                key = "k clip"
            hs.append(sub.register_forward_hook(clip(key)))
        elif isinstance(sub, mod.Inkling2LoopSoftcap):
            key = "logit softcap" if n.endswith("logit_softcap") else n.split(".")[-1]
            hs.append(sub.register_forward_hook(cap(key)))
    with torch.no_grad():
        m(torch.tensor([ids]), use_cache=False)
    for h in hs:
        h.remove()
    bad = []
    for k in sorted(seen):
        hit, tot = seen[k]
        print(f"  {k:24s} {hit:6d}/{tot:<6d} {hit / tot:.2f}")
        if hit == 0 or hit == tot:
            bad.append(k)
    if bad:
        sys.exit(f"these never bite or always bite, so a gate cannot see them: {bad}")


def golden_rows(m, ids):
    with torch.no_grad():
        lg = m(torch.tensor([ids]), use_cache=False).logits[0].float()
    return lg


def respell(d):
    p = os.path.join(d, "config.json")
    c = json.load(open(p))
    t = c["text_config"]
    for a, b in SPELLING.items():
        if a in t:
            t[b] = t.pop(a)
    t["loop_config"] = {"core_start": t.pop("loop_layer_index_start"), "core_end": t.pop("loop_layer_index_end"),
                        "num_passes": t.pop("loop_num_passes"), "gate_rank": t.pop("loop_gate_rank"),
                        "is_varloop": False}
    json.dump(c, open(p, "w"), indent=2)


def check_names(keys, index):
    if not index:
        return
    real = json.load(open(index))["weight_map"]
    shape = lambda k: re.sub(r"\.\d+(\.|$)", r".N\1", k)
    want = {shape(k) for k in real if k.startswith("model.llm.")}
    # The published experts carry their quantization's planes beside the codes.
    want = {w for w in want if not re.search(r"_(amax|scale|outer_scale_denominator)$", w) or "experts" not in w}
    have = {shape(k) for k in keys if k.startswith("model.llm.")}
    if have - want or want - have:
        sys.exit(f"names departing from the published checkpoint:\n  only ours {sorted(have - want)}\n"
                 f"  only theirs {sorted(want - have)}")
    print("  every text tensor name is the published checkpoint's")


def main(index):
    from safetensors import safe_open
    from transformers import AutoConfig, Inkling2LoopForConditionalGeneration
    m, cfg = build()
    perturb(m)
    to_bf16(m)
    ids = [(i * 7919 + 13) % 32000 for i in range(NPOS)]
    print("bite shares over the golden's ids:")
    stats(m, ids)
    mem = golden_rows(m, ids)

    d = os.path.join(MODELS, NAME)
    shutil.rmtree(d, ignore_errors=True)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not fp32_kept(m, n):
                p.data = p.data.to(torch.bfloat16)
    m.save_pretrained(d)
    respell(d)
    for f in TOKFILES:
        src = os.path.join(TOKDIR, "inkling2", f)
        if os.path.exists(src):
            shutil.copy(src, d)
    keys, dtypes = [], {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), "pt") as st:
                for k in st.keys():
                    keys.append(k)
                    dt = str(st.get_slice(k).get_dtype())
                    dtypes[dt] = dtypes.get(dt, 0) + 1
    print("  stored dtypes", dtypes)
    check_names(keys, index)

    rc = AutoConfig.from_pretrained(d)
    rt, ct = rc.get_text_config(), cfg.get_text_config()
    for k in list(TEXT) + ["layer_types"]:
        if getattr(rt, k) != getattr(ct, k):
            sys.exit(f"config.json reads back {k}={getattr(rt, k)!r}, the class built {getattr(ct, k)!r}")
    r = Inkling2LoopForConditionalGeneration.from_pretrained(d, dtype=torch.float32).eval()
    r.config._attn_implementation = "eager"
    r.config.get_text_config()._attn_implementation = "eager"
    lg = golden_rows(r, ids)
    drift = float((lg - mem).abs().max())
    if drift > 1e-5:
        sys.exit(f"the fixture read back computes another model: max |dlogit| {drift}")
    print(f"  read back: max |dlogit| {drift:.2e}")

    # The reference's own cached decode, against its own full pass.
    with torch.no_grad():
        g = r.generate(torch.tensor([ids[:PROMPT]]), max_new_tokens=NEW, do_sample=False,
                       use_cache=True, eos_token_id=None, pad_token_id=0)[0, PROMPT:].tolist()
        full = golden_rows(r, ids[:PROMPT] + g)
    tf = [int(x) for x in full[PROMPT - 1:-1].argmax(-1)]
    if tf != g:
        sys.exit(f"the reference's cached decode {g} is not its own full pass {tf}")

    rows = [{"argmax": int(x.argmax()), "max": float(x.max()), "lse": float(torch.logsumexp(x, 0)),
             "head": [round(float(v), 7) for v in x[:NLOGIT]]} for x in lg]
    tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    toks = []
    for x in TEXTS:
        e = tk.encode(x).ids
        toks.append({"text": x, "ids": e, "dec": tk.decode(e, skip_special_tokens=True)})
    size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors"))
    rec = {"dir": NAME, "ids": ids, "vocab": rt.vocab_size, "pos": rows, "tok": toks, "bytes": size,
           "greedy": {"prompt": ids[:PROMPT], "tokens": g}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, NAME + ".json"), "w"))
    print(NAME, "argmax", [x["argmax"] for x in rows], "greedy", g)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
