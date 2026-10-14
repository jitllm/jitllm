#!/usr/bin/env python3
"""LFM2-VL's goldens (d1-3B's tower), from the reference itself.

    $JITLLM_HF_PY scripts/lfm2visiongold.py --model-free
        -> testdata/golden/lfm2-layout.json: Lfm2VlImageProcessor's plan for
           many picture sizes (tiles, rows, columns and the smart-resized
           size), and the processor's expanded image tokens for each.

    $JITLLM_HF_PY scripts/lfm2visiongold.py DIR OUT IMAGE...
        -> OUT/: per picture, the processor's pixels (every piece,
           [y][x][channel], back to back), the tower's last_hidden_state
           (post-LayerNorm, each piece unpadded) and the projector's rows,
           from the checkpoint in DIR (d1-3B, Lfm2VlForConditionalGeneration)
           in float32; and OUT/decisions.json, d1's own SystemOne
           probabilities over the pictures (its runner.py, float32).

The reference is transformers' Lfm2VlProcessor (torchvision's uint8 resize,
the class's tiling) and Lfm2VlModel.get_image_features; the decisions are the
checkpoint's own code (runner.py: cap_pixels, prefix_text, the readout).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [(640, 488), (896, 896), (384, 384), (100, 100), (3000, 2000), (4000, 300), (300, 4000),
         (1920, 1080), (2358, 844), (1272, 1398), (257, 255), (16, 3000), (961, 959), (1024, 1024),
         (500, 333), (512, 512), (725, 724), (1100, 700), (640, 512), (2048, 64), (33, 33)]


def processor(d=None):
    from transformers import Lfm2VlImageProcessor
    if d:
        cfg = json.load(open(os.path.join(d, "processor_config.json")))["image_processor"]
        cfg.pop("image_processor_type", None)
        return Lfm2VlImageProcessor(**cfg)
    return Lfm2VlImageProcessor(resample=3)


def plan(ip, img):
    out = ip(images=[[img]], return_tensors="pt", return_row_col_info=True)
    return out


def model_free():
    from PIL import Image
    ip = processor()
    rows = []
    for w, h in SIZES:
        o = plan(ip, Image.new("RGB", (w, h), (90, 140, 200)))
        shapes = o["spatial_shapes"].tolist()
        rows.append({"w": w, "h": h, "rows": int(o["image_rows"][0]), "cols": int(o["image_cols"][0]),
                     "size": [int(x) for x in o["image_sizes"][0]], "pieces": shapes})
    dst = os.path.join(HERE, "..", "testdata", "golden", "lfm2-layout.json")
    json.dump({"sizes": rows}, open(dst, "w"), indent=0)
    print("wrote", dst, len(rows), "sizes")


def pieces_hwc(o, patch=16):
    """The processor's pixel_values, [n][patches][ky*kx*c] padded, as each
    piece's [y][x][c], unpadded."""
    out = []
    for i, (gh, gw) in enumerate(o["spatial_shapes"].tolist()):
        n = gh * gw
        p = o["pixel_values"][i, :n].reshape(gh, gw, patch, patch, 3).permute(0, 2, 1, 3, 4)
        out.append(p.reshape(gh * patch, gw * patch, 3).numpy().astype(np.float32))
    return out


def checkpoint(d, out, images):
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    ip = processor(d)
    lf = d1ref(d, "lfm2_vl")
    # The checkpoint's class is HF's Lfm2VlForConditionalGeneration with its
    # language model swapped for the hybrid stack; the tower and projector
    # are HF's own.
    m = lf.Lfm2VlForConditionalGeneration.from_pretrained(d, dtype=torch.float32, attn_implementation="sdpa")
    m.eval()
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        o = ip(images=[[img]], return_tensors="pt", return_row_col_info=True)
        px = pieces_hwc(o)
        with torch.no_grad():
            vo = m.model.vision_tower(pixel_values=o["pixel_values"], spatial_shapes=o["spatial_shapes"],
                                      pixel_attention_mask=o["pixel_attention_mask"])
            lens = o["pixel_attention_mask"].sum(1).tolist()
            vit = torch.cat([vo.last_hidden_state[i, :n] for i, n in enumerate(lens)])
            emb = torch.cat(m.model.get_image_features(pixel_values=o["pixel_values"],
                                                       spatial_shapes=o["spatial_shapes"],
                                                       pixel_attention_mask=o["pixel_attention_mask"]).pooler_output)
        base = os.path.join(out, name)
        np.concatenate([p.reshape(-1) for p in px]).tofile(base + ".px")
        vit.numpy().astype(np.float32).tofile(base + ".vit")
        emb.numpy().astype(np.float32).tofile(base + ".emb")
        meta = {"w": img.size[0], "h": img.size[1], "rows": int(o["image_rows"][0]),
                "cols": int(o["image_cols"][0]), "pieces": o["spatial_shapes"].tolist()}
        json.dump(meta, open(base + ".json", "w"))
        print(f"{name}: {img.size} -> {meta['cols']}x{meta['rows']} tiles, pieces {meta['pieces']}, "
              f"emb {tuple(emb.shape)}")
    decisions(d, out, images, m)


def d1ref(d, mod):
    """The checkpoint's own module mod, imported as a package from DIR."""
    import importlib
    import importlib.util
    if "d1ref" not in sys.modules:
        spec = importlib.util.spec_from_file_location("d1ref", os.path.join(d, "__init__.py"),
                                                      submodule_search_locations=[d])
        sys.modules["d1ref"] = importlib.util.module_from_spec(spec)
    return importlib.import_module("d1ref." + mod)


def decisions(d, out, images, m):
    """d1's own runner over the pictures: a choice, a noul and a score, with
    a text state and with none."""
    from PIL import Image
    from transformers import AutoTokenizer
    runner = d1ref(d, "runner")
    s1 = runner.SystemOne(model=m, tokenizer=AutoTokenizer.from_pretrained(d), device="cpu")
    qs = {
        "kind": {"type": "choice", "instructions": "What does the picture show?",
                 "criteria": {"photo": "a photograph", "chart": "a chart or a diagram", "shapes": "simple shapes"}},
        "people": {"type": "noul", "instructions": "Is there a person in the picture?"},
        "busy": {"type": "score", "instructions": "How busy is the picture?",
                 "criteria": ["empty", "sparse", "moderate", "busy"]},
    }
    cases = []
    for path in images:
        img = Image.open(path).convert("RGB")
        for state in (None, "A user uploaded this picture to a support ticket."):
            r = s1.system_one(state, qs, [img])
            cases.append({"image": os.path.basename(path), "state": state, "questions": qs, "response": r})
            print(os.path.basename(path), repr(state)[:20], json.dumps(r["answers"])[:160])
    json.dump(cases, open(os.path.join(out, "decisions.json"), "w"), indent=1)


# The fixture: transformers' own Lfm2VlForConditionalGeneration at toy size,
# every parameter and buffer randomised, written to GGUF by llama.cpp's own
# convert_hf_to_gguf.py (the text model plainly and the tower with --mmproj).
# The tokenizer is d1's (its image markers are in it); the processor is d1's
# (bicubic). The projector carries its optional LayerNorm, which d1's does
# not, so the fixture holds that path.

FIX = "synth-lfm2vl"
NLOGIT = 512
VIS = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
           patch_size=16, num_patches=256, vision_use_head=False, layer_norm_eps=1e-6,
           hidden_act="gelu_pytorch_tanh")


def qact(x):
    import torch.nn.functional as F
    k = x.shape[-1]
    kp = (k + 31) // 32 * 32
    y = F.pad(x.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = y.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(y * idd) * d).reshape(-1, kp)[:, :k].reshape(x.shape)


def chw_patch(p):
    # The patch row as the engine gathers it, channel-planar (c, ky, kx),
    # from the processor's (ky, kx, c).
    n = p.shape[-1]
    s = int((n // 3) ** 0.5)
    return p.reshape(*p.shape[:-1], s, s, 3).permute(*range(p.ndim - 1), -1, -3, -2).reshape(p.shape)


def from_chw(p):
    n = p.shape[-1]
    s = int((n // 3) ** 0.5)
    return p.reshape(*p.shape[:-1], 3, s, s).permute(*range(p.ndim - 1), -2, -1, -3).reshape(p.shape)


def perturb(mod):
    with torch.no_grad():
        for n, p in mod.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and ("norm" in last or "layernorm" in last):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif "position_embedding" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            elif "conv" in n and p.ndim == 3:
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
        for n, b in mod.named_buffers():
            if b.is_floating_point() and "inv_freq" not in n:
                b.copy_(torch.randn_like(b) * 0.3)


def features(m, o, a8=False):
    """The projector's rows of every piece, and the tower's before it: in
    float32, or at the engine's arithmetic (every Linear's input rounded to
    q8 per 32, the patch embedding's in its channel-planar row)."""
    hs = []
    if a8:
        for n, mod in m.model.named_modules():
            if not (n.startswith("vision_tower") or n.startswith("multi_modal_projector")):
                continue
            if isinstance(mod, torch.nn.Linear):
                if "patch_embedding" in n:
                    hs.append(mod.register_forward_pre_hook(lambda mod, inp: (from_chw(qact(chw_patch(inp[0]))),)))
                else:
                    hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))
    try:
        with torch.no_grad():
            vo = m.model.vision_tower(pixel_values=o["pixel_values"], spatial_shapes=o["spatial_shapes"],
                                      pixel_attention_mask=o["pixel_attention_mask"])
            lens = o["pixel_attention_mask"].sum(1).tolist()
            vit = torch.cat([vo.last_hidden_state[i, :n] for i, n in enumerate(lens)])
            emb = torch.cat(m.model.get_image_features(pixel_values=o["pixel_values"],
                                                       spatial_shapes=o["spatial_shapes"],
                                                       pixel_attention_mask=o["pixel_attention_mask"]).pooler_output)
    finally:
        for h in hs:
            h.remove()
    return vit, emb


def q8rows(w):
    import torch.nn.functional as F
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = F.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    y = (torch.round(x * idd) * d.half().float()).reshape(-1, kp)[:, :k]
    return y.reshape(w.shape)


def fixture():
    import copy
    import shutil
    import subprocess
    from PIL import Image
    from transformers import (AutoTokenizer, Lfm2Config, Lfm2VlConfig, Lfm2VlForConditionalGeneration,
                              Siglip2VisionConfig)
    sys.path.insert(0, HERE)
    import refenv
    models = refenv.models()
    d1 = refenv.need("D1_HF", "the d1-3B checkpoint directory (its tokenizer and processor_config.json)")
    lcpp = refenv.lcpp_src()
    hf_dir = refenv.under("LFM2VL_HF", "vlmb", "fixtures")
    torch.manual_seed(20261013)
    tc = Lfm2Config(vocab_size=128000, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, norm_eps=1e-5, conv_bias=False,
                    conv_L_cache=3, block_auto_adjust_ff_dim=False,
                    layer_types=["conv", "full_attention", "conv", "conv"], tie_word_embeddings=True,
                    max_position_embeddings=8192, rope_parameters={"rope_type": "default", "rope_theta": 1e6},
                    bos_token_id=124894, eos_token_id=124900, pad_token_id=124893)
    vc = Siglip2VisionConfig(**VIS)
    cfg = Lfm2VlConfig(text_config=tc.to_dict(), vision_config=vc.to_dict(), downsample_factor=2,
                       projector_hidden_size=64, projector_bias=True, projector_use_layernorm=True,
                       projector_hidden_act="gelu", image_token_id=124907, tie_word_embeddings=True)
    m = Lfm2VlForConditionalGeneration(cfg).float().eval()
    perturb(m)
    d = os.path.join(hf_dir, FIX)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"):
        shutil.copy(os.path.join(d1, f), d)
    # llama.cpp converts LFM2-VL's text model as the Lfm2ForCausalLM it is
    # (LiquidAI's GGUFs are made that way): the same weights, saved under
    # that class.
    from transformers import Lfm2ForCausalLM
    td = d + "-text"
    shutil.rmtree(td, ignore_errors=True)
    lm = Lfm2ForCausalLM(m.config.text_config).float().eval()
    lm.model.load_state_dict(m.model.language_model.state_dict())
    lm.lm_head.weight = lm.model.embed_tokens.weight
    lm.save_pretrained(td, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(os.path.join(d1, f), td)
    for src, kind, dst in ((td, "", FIX + ".gguf"), (d, "--mmproj", "mmproj-" + FIX + ".gguf")):
        args = [sys.executable, os.path.join(lcpp, "convert_hf_to_gguf.py"), src, "--outtype", "f32",
                "--outfile", os.path.join(models, dst)]
        if kind:
            args.append(kind)
        # The checkout's own gguf-py, which knows its own tensor names.
        env = {**os.environ, "PYTHONPATH": os.path.join(lcpp, "gguf-py")}
        subprocess.run(args, check=True, env=env)

    # A picture past twice the budget, so it is cut: tiles and a thumbnail.
    h, w = 700, 1100
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.clip(np.stack([255 * (0.5 + 0.5 * np.sin(x / 23.0)), 255 * (y / h),
                            255 * (((x - 300) ** 2 + (y - 260) ** 2) < 150 ** 2)], -1), 0, 255).astype(np.uint8)
    pic = Image.fromarray(img)
    out = os.path.join(models, FIX + "-gold")
    os.makedirs(out, exist_ok=True)
    pic.save(os.path.join(out, "picture.png"))
    ip = processor(d1)
    o = ip(images=[[pic]], return_tensors="pt", return_row_col_info=True)
    px = pieces_hwc(o)
    vit, emb = features(m, o)
    mq = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in mq.model.named_parameters():
            if not (n.startswith("vision_tower") or n.startswith("multi_modal_projector")):
                continue
            if n.endswith("weight") and p.ndim == 2:
                if "patch_embedding" in n:
                    p.copy_(from_chw(q8rows(chw_patch(p))))
                else:
                    p.copy_(q8rows(p))
    vit8, emb8 = features(mq, o, a8=True)

    tok = AutoTokenizer.from_pretrained(d1)
    from transformers import Lfm2VlProcessor
    proc = Lfm2VlProcessor(image_processor=ip, tokenizer=tok)
    text = "<|startoftext|><|im_start|>user\n<image>What is in the picture?<|im_end|>\n<|im_start|>assistant\n"
    enc = proc(text=[text], images=[[pic]], return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0].tolist()
    cont = tok.encode("A disc on stripes.", add_special_tokens=False)
    full = torch.tensor([ids + cont])
    slots = full[0] == cfg.image_token_id

    def logits(rows):
        with torch.no_grad():
            e = m.get_input_embeddings()(full)
            e[0, slots] = rows
            lg = m(inputs_embeds=e).logits[0]
        first = len(ids) - 1
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                 "head": [round(float(v), 7) for v in r[:NLOGIT]]} for r in lg[first:]]

    base = os.path.join(out, "picture")
    np.concatenate([p.reshape(-1) for p in px]).tofile(base + ".px")
    for name, v in (("vit", vit), ("emb", emb), ("vit_a8", vit8), ("emb_a8", emb8)):
        v.numpy().astype(np.float32).tofile(base + "." + name)
    json.dump({"w": w, "h": h, "rows": int(o["image_rows"][0]), "cols": int(o["image_cols"][0]),
               "pieces": o["spatial_shapes"].tolist(), "ids": ids, "image_token": cfg.image_token_id,
               "cont": cont, "logits": logits(emb), "logits_a8": logits(emb8)},
              open(base + ".json", "w"))
    print(f"{FIX}: {w}x{h} -> {int(o['image_cols'][0])}x{int(o['image_rows'][0])} tiles, "
          f"{len(ids)} prompt positions, {int(slots.sum())} picture rows")


if __name__ == "__main__":
    if sys.argv[1:] == ["--model-free"]:
        model_free()
    elif sys.argv[1:] == ["--fixture"]:
        fixture()
    elif len(sys.argv) > 3:
        checkpoint(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
