#!/usr/bin/env python3
"""Transformers' goldens for a real LFM2-VL checkpoint's tower.

    $JITLLM_HF_PY scripts/lfm2vlreal.py [checkpoint dir] [picture.png]

The checkpoint dir holds LiquidAI's LFM2-VL as published (config.json,
preprocessor_config.json, model.safetensors): transformers' own
Lfm2VlForConditionalGeneration in float32 on the CPU, the picture cut by its
Lfm2VlImageProcessor into tiles and a thumbnail. Writes, beside the
picture's name in <dir>/goldens/:

  <pic>.px            every piece's pixel_values, [y][x][channel], back to back
  <pic>.json          {"w", "h", "rows", "cols", "pieces": [[gh, gw]...]}
  <pic>.features.f32  get_image_features' rows, every piece's in order,
                      float32 [token][text width]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

import refenv  # where things are: environment variables only (docs/testing.md)


def pieces_hwc(o, patch):
    out = []
    for i, (gh, gw) in enumerate(o["spatial_shapes"].tolist()):
        n = gh * gw
        p = o["pixel_values"][i, :n].reshape(gh, gw, patch, patch, 3).permute(0, 2, 1, 3, 4)
        out.append(p.reshape(gh * patch, gw * patch, 3).numpy().astype(np.float32))
    return out


def main(ckpt, pic):
    from transformers import Lfm2VlForConditionalGeneration, Lfm2VlImageProcessor
    cfg = json.load(open(os.path.join(ckpt, "preprocessor_config.json")))
    for k in ("image_processor_type", "processor_class"):
        cfg.pop(k, None)
    ip = Lfm2VlImageProcessor(**cfg)
    m = Lfm2VlForConditionalGeneration.from_pretrained(ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    patch = m.config.vision_config.patch_size
    img = Image.open(pic).convert("RGB")
    o = ip(images=[[img]], return_tensors="pt", return_row_col_info=True)
    with torch.no_grad():
        feats = torch.cat(m.model.get_image_features(pixel_values=o["pixel_values"],
                                                     spatial_shapes=o["spatial_shapes"],
                                                     pixel_attention_mask=o["pixel_attention_mask"]).pooler_output)
    out = os.path.join(ckpt, "goldens")
    os.makedirs(out, exist_ok=True)
    name = os.path.splitext(os.path.basename(pic))[0]
    base = os.path.join(out, name)
    np.concatenate([p.reshape(-1) for p in pieces_hwc(o, patch)]).tofile(base + ".px")
    feats.contiguous().numpy().astype(np.float32).tofile(base + ".features.f32")
    meta = {"w": img.size[0], "h": img.size[1], "rows": int(o["image_rows"][0]), "cols": int(o["image_cols"][0]),
            "pieces": o["spatial_shapes"].tolist()}
    json.dump(meta, open(base + ".json", "w"))
    print(f"{name}: {img.size[0]}x{img.size[1]} -> {meta['cols']}x{meta['rows']} tiles, pieces {meta['pieces']}, "
          f"{feats.shape[0]} rows of {feats.shape[1]}")


if __name__ == "__main__":
    main(refenv.arg(1, "vlmb", "hf", "LFM2-VL-450M"),
         refenv.arg(2, "vlm", "goldens", "photo.png"))
