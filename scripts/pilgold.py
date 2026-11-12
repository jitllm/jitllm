#!/usr/bin/env python3
"""Record Pillow's and torchvision's antialiased resizes of fixed pictures, once.
-> testdata/golden/pil/resample.json

  $JITLLM_HF_PY scripts/pilgold.py

THE ORACLE FOR THE PICTURE OP'S PILLOW FILTERS (graph.ChainPIL). A tower whose
processor resizes with Pillow -- Gemma 3's Gemma3ImageProcessor (BILINEAR),
CLIP's and InternVL's (BICUBIC) -- is held to the bytes Pillow's resize
writes, not to an interpolation that looks right. Each case is a pattern
picture both this script and the Go gates rebuild from one formula (pattern
below), or a test picture read from engine/testdata, resized in mode RGB
to a size, with its output's sha256 and, for a small one, its every byte.

A case whose lib is torchvision is transformers' default image processors'
resize: torchvision's F.resize of a uint8 tensor on the CPU, antialiased --
Pillow's filters and passes, the weights at torchvision's own precision
(torchvision 0.27 on for LANCZOS; transformers falls back to BICUBIC below
it or off the CPU). BILINEAR takes that uint8 path on AVX2, AVX-512 and
arm64 hosts, the ones these were recorded on.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pattern(w, h):
    """A w x h RGB picture: red (7x + 13y) mod 256, green (3xy + 5) mod 256,
    blue 11(x + 2y) mod 256."""
    b = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            b[i] = (7 * x + 13 * y) % 256
            b[i + 1] = (3 * x * y + 5) % 256
            b[i + 2] = (11 * (x + 2 * y)) % 256
    return Image.frombytes("RGB", (w, h), bytes(b))


FILTERS = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
           "lanczos": Image.Resampling.LANCZOS}
TV = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC,
      "lanczos": InterpolationMode.LANCZOS}

CASES = [
    ("pattern", 97, 61, 48, 48, "bilinear"),
    ("pattern", 20, 33, 48, 48, "bilinear"),
    ("pattern", 1000, 700, 896, 896, "bilinear"),
    ("pattern", 3000, 2000, 896, 896, "bilinear"),
    ("pattern", 97, 61, 48, 48, "bicubic"),
    ("pattern", 20, 33, 48, 48, "bicubic"),
    ("pattern", 1300, 900, 336, 336, "bicubic"),
    ("pattern", 4096, 3000, 448, 448, "bicubic"),
    ("quad-and-disc.png", 0, 0, 896, 896, "bilinear"),
    ("quad-and-disc.png", 0, 0, 336, 336, "bicubic"),
    ("pattern", 97, 61, 48, 48, "lanczos"),
    ("pattern", 20, 33, 48, 48, "lanczos"),
    ("pattern", 640, 488, 2048, 1562, "lanczos"),
    ("quad-and-disc.png", 0, 0, 512, 512, "lanczos"),
    ("pattern", 97, 61, 48, 48, "bilinear", "torchvision"),
    ("pattern", 97, 61, 48, 48, "bicubic", "torchvision"),
    ("pattern", 1300, 900, 336, 336, "bicubic", "torchvision"),
    ("pattern", 97, 61, 48, 48, "lanczos", "torchvision"),
    ("pattern", 20, 33, 48, 48, "lanczos", "torchvision"),
    # Idefics3's (SmolVLM's): the longer side to 2048, then the sides to
    # whole 512 tiles, then the global square, each from the one before.
    ("pattern", 640, 488, 2048, 1562, "lanczos", "torchvision"),
    ("pattern", 2048, 1562, 2048, 2048, "lanczos", "torchvision"),
    ("pattern", 2048, 2048, 512, 512, "lanczos", "torchvision"),
    ("quad-and-disc.png", 0, 0, 512, 512, "lanczos", "torchvision"),
]


def main():
    out = []
    for src, w, h, ow, oh, f, *lib in CASES:
        lib = lib[0] if lib else "pillow"
        if src == "pattern":
            img = pattern(w, h)
        else:
            img = Image.open(os.path.join(ROOT, "engine", "testdata", src)).convert("RGB")
            w, h = img.size
        if lib == "torchvision":
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()
            got = TF.resize(t, [oh, ow], interpolation=TV[f], antialias=True).permute(1, 2, 0).contiguous().numpy().tobytes()
        else:
            got = img.resize((ow, oh), resample=FILTERS[f]).tobytes()
        case = {"src": src, "w": w, "h": h, "ow": ow, "oh": oh, "filter": f, "lib": lib,
                "sha256": hashlib.sha256(got).hexdigest()}
        if len(got) <= 48 * 48 * 3:
            case["bytes"] = list(got)
        out.append(case)
    path = os.path.join(ROOT, "testdata", "golden", "pil", "resample.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"pillow": Image.__version__ if hasattr(Image, "__version__") else "", "cases": out}
    import PIL
    import torchvision
    meta["pillow"] = PIL.__version__
    meta["torchvision"] = torchvision.__version__
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    print("wrote", path, len(out), "cases", file=sys.stderr)


if __name__ == "__main__":
    main()
