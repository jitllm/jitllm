#!/usr/bin/env python3
"""Build the Llama 4 vision fixture and record transformers' answers.

    $JITLLM_HF_PY scripts/llama4visiongold.py
    JITLLM_MODELS=/path/to/models   (where the GGUFs go)
    L4_TOK=<dir with Llama 4's tokenizer.json and tokenizer_config.json>

-> $JITLLM_MODELS/synth-llama4v.gguf, $JITLLM_MODELS/mmproj-synth-llama4v.gguf
   and testdata/golden/llama4v/synth-llama4v.json

THE ORACLE IS transformers' OWN Llama4ForConditionalGeneration, every
parameter randomised, AND BOTH GGUFs ARE WRITTEN BY llama.cpp's OWN
convert_hf_to_gguf.py, the text model plainly and the tower with --mmproj.
The tower is Llama 4's at toy size: a 112-pixel tile of 14-pixel patches
(8x8, sixteen tokens a tile after the 2x2 pixel shuffle), the class token
LAST with the last row of the position table, a 2-D rotary on adjacent pairs
whose positions start at one and whose class token does not turn, LayerNorms
with biases, and the MLP2 adapter (two GELUs) before the projector.

The picture is 224 x 168: the processor's best fit is a 2x2 canvas, the
picture is not resized and is padded at the bottom, so four tiles and a
global one -- the tile order, the padding and the separators are all in it.
The golden carries the picture, the processor's pixels per tile (its
torchvision resize and its bfloat16 normalisation), the tower's rows per tile
(at f32, at the converter's Q8_0 and at the engine's int8 activations), the
prompt as the processor lays it out, and the logits.
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
TOK = refenv.need("L4_TOK", "the directory holding Llama 4's tokenizer.json")
HFDIR = refenv.under("L4V_HF", "vlmb", "fixtures")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "llama4v")
NAME = "synth-llama4v"
NLOGIT = 512
TILE = 112
IMG_H, IMG_W = 168, 224


def config():
    from transformers import Llama4Config
    # The text model is the synth-llama4 shape with Scout's chunk and floor
    # (llama.cpp's converter hardcodes both); its features are gated by that
    # fixture, this one by the picture.
    return Llama4Config(
        text_config=dict(vocab_size=202048, hidden_size=64, intermediate_size=64,
                         intermediate_size_mlp=128, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, num_local_experts=4, num_experts_per_tok=1,
                         interleave_moe_layer_step=2, use_qk_norm=True, no_rope_layer_interval=4,
                         attention_chunk_size=8192, floor_scale=8192, attn_temperature_tuning=True,
                         rope_theta=500000.0, rope_scaling=None, rms_norm_eps=1e-5,
                         max_position_embeddings=4096, tie_word_embeddings=False),
        vision_config=dict(hidden_size=64, intermediate_size=256, num_hidden_layers=3,
                           num_attention_heads=4, image_size=TILE, patch_size=14, norm_eps=1e-5,
                           pixel_shuffle_ratio=0.5, projector_input_dim=64, projector_output_dim=64,
                           vision_output_dim=64, rope_theta=10000, hidden_act="gelu"),
        boi_token_index=200080, eoi_token_index=200081, image_token_index=200092,
        tie_word_embeddings=False)


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            mod = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and ("norm" in mod or "layernorm" in mod):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif "vision_model" in n and ("q_proj" in n or "k_proj" in n):
                p.copy_(torch.randn_like(p) * 0.22)
            elif "vision_adapter" in n or "multi_modal_projector" in n:
                # The adapter and the projector at about the text model's scale,
                # so the toy text model does not amplify the rows' int8 a
                # hundredfold (pixtralgold.py).
                p.copy_(torch.randn_like(p) * 0.03)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


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


def quantize_vision(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not n.endswith("weight") or p.ndim != 2:
                continue
            if "vision_model" in n or "multi_modal_projector" in n:
                p.copy_(q8rows(p))


def features(m, pv, a8=False):
    # a8: every matmul's input at int8 per block of 32 (the unfold's rows are
    # the patch matmul's, (c, ky, kx) as the engine gathers them); the GELUs
    # the reference's own, exact.
    hs = []
    if a8:
        for n, mod in m.named_modules():
            if ("vision_model" in n or "multi_modal_projector" in n) and isinstance(mod, torch.nn.Linear):
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))
    try:
        with torch.no_grad():
            f = m.get_image_features(pv, vision_feature_select_strategy="default")
        f = f.last_hidden_state if hasattr(f, "last_hidden_state") else f
        f = m.multi_modal_projector(f.reshape(-1, f.shape[-1]))
        return f
    finally:
        for h in hs:
            h.remove()


def b64(img):
    import base64
    return base64.b64encode(img.reshape(-1).tobytes()).decode()


def main():
    from transformers import AutoTokenizer, Llama4ForConditionalGeneration
    from transformers.models.llama4.image_processing_llama4 import Llama4ImageProcessor
    cfg = config()
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    torch.manual_seed(20261004)
    m = Llama4ForConditionalGeneration(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, NAME)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(TOK, f), d)
    improc = Llama4ImageProcessor(size={"height": TILE, "width": TILE}, max_patches=16)
    improc.save_pretrained(d)
    for kind, dst in (("", NAME + ".gguf"), ("--mmproj", "mmproj-" + NAME + ".gguf")):
        args = [PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(MODELS, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True)

    tok = AutoTokenizer.from_pretrained(d)
    img = picture(IMG_H, IMG_W)
    pv = improc(images=[img], return_tensors="pt")
    ar = [int(v) for v in pv["aspect_ratios"][0]]
    px = pv["pixel_values"].float()  # tiles, the global one last
    per = (TILE // 14) ** 2 // 4
    S, X, Y, IM, PATCH, E = 200080, 200084, 200085, 200090, 200092, 200081
    rows = [S]
    if ar[0] * ar[1] > 1:
        for yy in range(ar[0]):
            for xx in range(ar[1]):
                rows += [PATCH] * per
                if xx < ar[1] - 1:
                    rows.append(X)
            rows.append(Y)
    rows += [IM] + [PATCH] * per + [E]
    pre = [200000] + tok.encode("<|header_start|>user<|header_end|>\n\n", add_special_tokens=False)
    post = tok.encode("What is in the picture?<|eot|><|header_start|>assistant<|header_end|>\n\n",
                      add_special_tokens=False)
    cont = tok.encode("A disc on a gradient.", add_special_tokens=False)
    ids = pre + rows + post + cont
    t_ids = torch.tensor([ids])
    with torch.no_grad():
        out = m(input_ids=t_ids, pixel_values=px)
    lg = out.logits[0]
    feats = features(m, px)
    mq = copy.deepcopy(m)
    quantize_vision(mq)
    feats_q8 = features(mq, px)
    feats_a8 = features(mq, px, a8=True)
    first = len(pre) + len(rows) + len(post) - 1

    def logit_rows(lg):
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]

    with torch.no_grad():
        emb = m.get_input_embeddings()(t_ids)
        sel = (t_ids[0] == PATCH).nonzero().reshape(-1)
        emb[0, sel] = feats_a8.to(emb.dtype)
        lg8 = m(inputs_embeds=emb).logits[0]
    g = {
        "pre": pre, "rows": rows, "post": post, "cont": cont, "aspect": ar,
        "image": {"h": IMG_H, "w": IMG_W, "rgb": b64(img), "tiles": px.shape[0],
                  "px": [round(float(x), 8) for x in px.permute(0, 2, 3, 1).reshape(-1)],
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_q8": [round(float(x), 8) for x in feats_q8.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
        "logits": logit_rows(lg),
        "logits_a8": logit_rows(lg8),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, NAME + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{NAME}: {len(ids)} ids, {px.shape[0]} tiles (aspect {ar}), {len(g['logits'])} logit rows")


if __name__ == "__main__":
    main()
