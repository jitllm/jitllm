#!/usr/bin/env python3
"""Build the Gemma 4 vision fixture and record transformers' answers.

    $JITLLM_HF_PY scripts/gemma4visiongold.py [--every]
    JITLLM_MODELS=/path/to/models   (where the GGUFs go)
    CV_TOK=<dir holding gemma4/ tokenizer files>   (gemma4gold.py's)

--every adds the every-layer run (logits_every) to the golden of the saved
fixture, after checking the fixture still gives the golden's logits.

-> $JITLLM_MODELS/synth-gemma4v.gguf, $JITLLM_MODELS/mmproj-synth-gemma4v.gguf
   and testdata/golden/gemma4v/synth-gemma4v.json

THE ORACLE IS transformers' OWN Gemma4ForConditionalGeneration, every
parameter randomised -- and every buffer that is a weight: the clipped
linears' input and output bounds, set to bind on part of each matrix's range,
and the standardisation's bias and scale -- AND BOTH GGUFs ARE WRITTEN BY
llama.cpp's OWN convert_hf_to_gguf.py.

The text model is synth-gemma4's 31B shape with the larger checkpoints'
use_bidirectional_attention "vision": an image's rows attend to each other on
the SLIDING layers and causally on the global ones. The tower is Gemma 4's at
toy size: 16-pixel patches, sandwich RMSNorms, a q/k norm, a weightless v
norm, attention at scale one, a gated GELU MLP, a 2-D rotary on each half of a
head NEOX-paired (the column's half first), two learned position tables, the
3x3 average pool, sqrt(hidden), the standardisation, an unweighted RMSNorm and
the projection. The picture is 144 x 192: the processor's budget (280 soft
tokens, the processors' default) resizes it UP, Pillow BICUBIC, to 672 x 912 --
42 x 57 patches, 266 soft tokens.
"""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("G4V_HF", "vlmb", "fixtures")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "gemma4v")
NAME = "synth-gemma4v"
NLOGIT = 512
IMG_H, IMG_W = 144, 192
S, F = "sliding_attention", "full_attention"


def config():
    from transformers import Gemma4Config
    text = dict(vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                num_global_key_value_heads=1, attention_k_eq_v=True,
                layer_types=[S, S, F, S, S, F], sliding_window=512, final_logit_softcapping=30.0,
                hidden_size_per_layer_input=0, num_kv_shared_layers=0, rms_norm_eps=1e-6,
                max_position_embeddings=4096, tie_word_embeddings=True,
                use_bidirectional_attention="vision",
                rope_parameters={S: {"rope_type": "default", "rope_theta": 10000.0},
                                 F: {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                     "rope_theta": 1000000.0}},
                pad_token_id=0, eos_token_id=1, bos_token_id=2)
    vision = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=4, head_dim=16, patch_size=16, pooling_kernel_size=3,
                  position_embedding_size=64, use_clipped_linears=True, standardize=True,
                  rope_parameters={"rope_type": "default", "rope_theta": 100.0})
    return Gemma4Config(text_config=text, vision_config=vision, audio_config=None,
                        boi_token_id=255999, eoi_token_id=258882, image_token_id=258880,
                        vision_soft_tokens_per_image=280, tie_word_embeddings=True)


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif "vision_tower" in n and ("q_proj" in n or "k_proj" in n):
                p.copy_(torch.randn_like(p) * 0.22)
            elif "embed_vision" in n:
                p.copy_(torch.randn_like(p) * 0.03)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(torch.rand_like(b) * 0.8 + 0.6)
            elif n.endswith("std_bias"):
                b.copy_(torch.randn_like(b) * 0.3)
            elif n.endswith("std_scale"):
                b.copy_(torch.rand_like(b) * 0.8 + 0.6)


def set_clamps(m, pv, pos):
    # The clipped linears' bounds, each set from what the matrix actually sees
    # and emits on the fixture's picture -- the 5th and 95th percentile of its
    # input and of its output -- so every bound binds on a tenth of its values:
    # a clamp skipped, or applied on the wrong side of the matmul, moves the
    # answer.
    seen = {}
    hooks = []
    for n, mod in m.named_modules():
        if type(mod).__name__ == "Gemma4ClippableLinear" and "vision_tower" in n:
            def hook(mod, inp, out, n=n):
                seen[n] = (inp[0].detach().flatten(), out.detach().flatten())
            hooks.append(mod.register_forward_hook(hook))
            mod.use_clipped_linears = False
    with torch.no_grad():
        m.model.vision_tower(pv, pos)
    for h in hooks:
        h.remove()
    for n, mod in m.named_modules():
        if n in seen:
            i, o = seen[n]
            q = torch.tensor([0.05, 0.95])
            ilo, ihi = torch.quantile(i, q)
            olo, ohi = torch.quantile(o, q)
            mod.input_min.fill_(float(ilo))
            mod.input_max.fill_(float(ihi))
            mod.output_min.fill_(float(olo))
            mod.output_max.fill_(float(ohi))
            mod.use_clipped_linears = True


def picture(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * (0.5 + 0.5 * np.sin(x / 9.0))
    g = 255 * (y / h)
    b = 255 * (((x - 70) ** 2 + (y - 40) ** 2) < 30 ** 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def q8rows(w):
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = torch.nn.functional.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    y = (torch.round(x * idd) * d.half().float()).reshape(-1, kp)[:, :k]
    return y.reshape(w.shape)


def qact(x):
    k = x.shape[-1]
    kp = (k + 31) // 32 * 32
    y = torch.nn.functional.pad(x.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = y.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(y * idd) * d).reshape(-1, kp)[:, :k].reshape(x.shape)


def chw_patch(p):
    # The patch row as the engine gathers it, channel-planar (c, ky, kx), from
    # the processor's (ky, kx, c) order -- for quantizing it as the engine does.
    n = p.shape[-1]
    s = int((n // 3) ** 0.5)
    return p.reshape(*p.shape[:-1], s, s, 3).permute(*range(p.ndim - 1), -1, -3, -2).reshape(p.shape)


def from_chw(p):
    n = p.shape[-1]
    s = int((n // 3) ** 0.5)
    return p.reshape(*p.shape[:-1], 3, s, s).permute(*range(p.ndim - 1), -2, -1, -3).reshape(p.shape)


def quantize_vision(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not n.endswith("weight") or p.ndim != 2:
                continue
            if "vision_tower" in n or "embed_vision" in n:
                if "input_proj" in n:
                    p.copy_(from_chw(q8rows(chw_patch(p))))
                else:
                    p.copy_(q8rows(p))


def features(m, pv, pos, a8=False):
    hs = []
    if a8:
        for n, mod in m.named_modules():
            if not ("vision_tower" in n or "embed_vision" in n) or not isinstance(mod, torch.nn.Linear):
                continue
            if "input_proj" in n:
                # The patch rows, (2x - 1) first as the module does, at int8 in
                # the engine's channel-planar order.
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (from_chw(qact(chw_patch(inp[0]))),)))
            else:
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))
    try:
        with torch.no_grad():
            f = m.model.get_image_features(pv, pos)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        if isinstance(f, (list, tuple)):
            f = torch.cat(list(f), 0)
        return f.reshape(-1, f.shape[-1])
    finally:
        for h in hs:
            h.remove()


def b64(img):
    import base64
    return base64.b64encode(img.reshape(-1).tobytes()).decode()


def every_layer(m, t_ids, pix, pos, types):
    """The logits with the picture's rows bidirectional on every layer, the
    global ones too: the violation the engine's restated container must
    read as. transformers' full-attention mask, given the sliding one's
    blockwise overlay."""
    import transformers.models.gemma4.modeling_gemma4 as g4
    stated = g4.create_masks_for_vision_model

    def every(config, inputs_embeds, attention_mask, past_key_values, position_ids, block_sequence_ids):
        masks = stated(config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       past_key_values=past_key_values, position_ids=position_ids,
                       block_sequence_ids=block_sequence_ids)
        masks["full_attention"] = g4.create_causal_mask(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
            or_mask_function=g4.blockwise_overlay(block_sequence_ids))
        return masks

    g4.create_masks_for_vision_model = every
    try:
        with torch.no_grad():
            return m(input_ids=t_ids, pixel_values=pix, image_position_ids=pos, mm_token_type_ids=types).logits[0]
    finally:
        g4.create_masks_for_vision_model = stated


def add_every():
    """--every: the saved fixture's golden given the every-layer run, after
    checking the fixture still gives the golden's logits."""
    from transformers import Gemma4ForConditionalGeneration
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    d = os.path.join(HFDIR, NAME)
    m = Gemma4ForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32).eval()
    m.config._attn_implementation = "eager"
    m.config.text_config._attn_implementation = "eager"
    path = os.path.join(OUT, NAME + ".json")
    with open(path) as f:
        g = json.load(f)
    pv = Gemma4ImageProcessorPil(max_soft_tokens=280)(images=[picture(IMG_H, IMG_W)], return_tensors="pt")
    pix, pos = pv["pixel_values"], pv["image_position_ids"]
    IMG = 258880
    ids = g["pre"] + [g["boi"]] + [IMG] * g["image"]["soft"] + [g["eoi"]] + g["post"] + g["cont"]
    t_ids = torch.tensor([ids])
    types = torch.tensor([[1 if i == IMG else 0 for i in ids]])
    first = len(ids) - len(g["cont"]) - 1
    with torch.no_grad():
        lg = m(input_ids=t_ids, pixel_values=pix, image_position_ids=pos, mm_token_type_ids=types).logits[0]
    worst = max(abs(float(lg[first + i][j]) - h) for i, r in enumerate(g["logits"]) for j, h in enumerate(r["head"]))
    assert worst < 1e-4, f"the saved fixture gives logits {worst} from the golden's"
    ev = every_layer(m, t_ids, pix, pos, types)
    g["logits_every"] = [{"argmax": int(r.argmax()), "max": float(r.max()),
                          "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                          "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in ev[first:]]
    with open(path, "w") as f:
        json.dump(g, f)
    print(f"{NAME}: the fixture gives the golden's logits ({worst:.1e}); logits_every added")


def main():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    cfg = config()
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = Gemma4ForConditionalGeneration(cfg).float().eval()
    perturb(m)
    improc = Gemma4ImageProcessorPil(max_soft_tokens=280)
    img = picture(IMG_H, IMG_W)
    pv = improc(images=[img], return_tensors="pt")
    pix, pos = pv["pixel_values"], pv["image_position_ids"]
    set_clamps(m, pix, pos)
    d = os.path.join(HFDIR, NAME)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(os.path.join(TOKDIR, "gemma4")):
        if f != "config.json":
            shutil.copy(os.path.join(TOKDIR, "gemma4", f), d)
    improc.save_pretrained(d)
    for kind, dst in (("", NAME + ".gguf"), ("--mmproj", "mmproj-" + NAME + ".gguf")):
        args = [PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(MODELS, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True)

    tok = AutoTokenizer.from_pretrained(d)
    valid = (pos[0] >= 0).all(-1)
    npatch = int(valid.sum())
    xs, ys = pos[0][valid][:, 0], pos[0][valid][:, 1]
    gw, gh = int(xs.max()) + 1, int(ys.max()) + 1
    nsoft = npatch // 9
    BOI, EOI, IMG = 255999, 258882, 258880
    pre = [2] + tok.encode("<|turn>user\n", add_special_tokens=False)
    post = tok.encode("What is in the picture?<turn|>\n<|turn>model\n", add_special_tokens=False)
    cont = tok.encode("A disc on a gradient.", add_special_tokens=False)
    ids = pre + [BOI] + [IMG] * nsoft + [EOI] + post + cont
    t_ids = torch.tensor([ids])
    types = torch.tensor([[1 if i == IMG else 0 for i in ids]])
    with torch.no_grad():
        out = m(input_ids=t_ids, pixel_values=pix, image_position_ids=pos, mm_token_type_ids=types)
    lg = out.logits[0]
    feats = features(m, pix, pos)
    mq = copy.deepcopy(m)
    quantize_vision(mq)
    feats_q8 = features(mq, pix, pos)
    feats_a8 = features(mq, pix, pos, a8=True)
    first = len(pre) + nsoft + 2 + len(post) - 1

    def logit_rows(lg):
        # margin is the top two's gap over the whole vocabulary: a row whose
        # argmax the engine's int8 may part from is a tie, which the gate
        # names rather than counts.
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                 "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]

    with torch.no_grad():
        emb = m.get_input_embeddings()(t_ids)
        sel = (t_ids[0] == IMG).nonzero().reshape(-1)
        emb[0, sel] = feats_a8.to(emb.dtype)
        lg8 = m(inputs_embeds=emb, mm_token_type_ids=types).logits[0]
        # The violations' logits: the image's rows causal on every layer,
        # and bidirectional on every layer.
        causal = m(input_ids=t_ids, pixel_values=pix, image_position_ids=pos).logits[0]
    every = every_layer(m, t_ids, pix, pos, types)
    # The patch rows as the processor laid them out, (ky, kx, c) each.
    px = pix[0][valid]
    g = {
        "pre": pre, "post": post, "cont": cont, "boi": BOI, "eoi": EOI,
        "image": {"h": IMG_H, "w": IMG_W, "rgb": b64(img), "grid": [gh, gw], "soft": nsoft,
                  "px": [round(float(x), 8) for x in px.reshape(-1)],
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_q8": [round(float(x), 8) for x in feats_q8.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
        "logits": logit_rows(lg),
        "logits_a8": logit_rows(lg8),
        "logits_causal": logit_rows(causal),
        "logits_every": logit_rows(every),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, NAME + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{NAME}: {len(ids)} ids, {gh}x{gw} patches, {nsoft} soft tokens, {len(g['logits'])} logit rows")


if __name__ == "__main__":
    add_every() if sys.argv[1:] == ["--every"] else main()
