#!/usr/bin/env python3
"""Transformers' goldens for a real SmolVLM (Idefics3) tower.

    $JITLLM_HF_PY scripts/smolvlmreal.py [checkpoint dir] [picture.png]

The checkpoint dir is the publisher's (config.json, preprocessor_config.json,
model.safetensors). The tower and the connector are transformers' own
Idefics3 vision model and connector, run in float32 on the CPU on
Idefics3ImageProcessor's pixels: the picture resized with torchvision's
Lanczos (the processor's torchvision backend) to the processor's longest
edge, cut into tiles of its max_image_size after a second Lanczos resize to
whole tiles, and a global square resized from that, each tile a square the
tower reads. Writes, beside the picture's name
in <dir>/goldens/:

  <pic>.pixels.f32    every square's pixel_values in the processor's order,
                      float32 [square][y][x][channel]
  <pic>.json          {"rows", "cols", "squares", "side"}: the tile grid
                      (0 x 0 for a picture of one square), the squares and
                      their side
  <pic>.features.f32  the connector's rows, float32 [square][token][text width]
"""
import json
import os

import numpy as np
import torch
from PIL import Image

import refenv  # where things are: environment variables only (docs/testing.md)


def main(ckpt, pic):
    from transformers import Idefics3ForConditionalGeneration, Idefics3ImageProcessor
    improc = Idefics3ImageProcessor.from_pretrained(ckpt)
    model = Idefics3ForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
    img = Image.open(pic).convert("RGB")
    out = improc(images=[[img]], return_tensors="pt", return_row_col_info=True)
    px = out["pixel_values"][0]  # [squares][3][side][side]
    rows, cols = int(out["rows"][0][0]), int(out["cols"][0][0])
    with torch.no_grad():
        hs = model.model.vision_model(pixel_values=px.float()).last_hidden_state
        feats = model.model.connector(hs)
    dst = os.path.join(ckpt, "goldens")
    os.makedirs(dst, exist_ok=True)
    name = os.path.splitext(os.path.basename(pic))[0]
    px.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32).tofile(os.path.join(dst, name + ".pixels.f32"))
    feats.contiguous().numpy().astype(np.float32).tofile(os.path.join(dst, name + ".features.f32"))
    side = int(px.shape[-1])
    json.dump({"rows": rows, "cols": cols, "squares": int(px.shape[0]), "side": side},
              open(os.path.join(dst, name + ".json"), "w"))
    print(f"{name}: {img.size[0]}x{img.size[1]} -> {rows}x{cols} tiles + global, {px.shape[0]} squares of {side}, "
          f"{feats.shape[0]}x{feats.shape[1]} rows of {feats.shape[2]}")


if __name__ == "__main__":
    main(refenv.arg(1, "vlmb", "hf", "SmolVLM-256M-Instruct"),
         refenv.arg(2, "vlm", "goldens", "photo.png"))
