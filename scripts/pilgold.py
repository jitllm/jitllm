#!/usr/bin/env python3
"""Record Pillow's antialiased resize of fixed pictures, once.
-> testdata/golden/pil/resample.json

  $JITLLM_HF_PY scripts/pilgold.py

THE ORACLE FOR THE PICTURE OP'S PILLOW FILTERS (graph.ChainPIL). A tower whose
processor resizes with Pillow -- Gemma 3's Gemma3ImageProcessor (BILINEAR),
CLIP's and InternVL's (BICUBIC) -- is held to the bytes Pillow's resize
writes, not to an interpolation that looks right. Each case is a pattern
picture both this script and the Go gates rebuild from one formula (pattern
below), or a test picture read from engine/model/testdata, resized in mode RGB
to a size, with its output's sha256 and, for a small one, its every byte.
"""
import hashlib
import json
import os
import sys

from PIL import Image

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


FILTERS = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}

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
]


def main():
    out = []
    for src, w, h, ow, oh, f in CASES:
        if src == "pattern":
            img = pattern(w, h)
        else:
            img = Image.open(os.path.join(ROOT, "engine", "model", "testdata", src)).convert("RGB")
            w, h = img.size
        got = img.resize((ow, oh), resample=FILTERS[f]).tobytes()
        case = {"src": src, "w": w, "h": h, "ow": ow, "oh": oh, "filter": f,
                "sha256": hashlib.sha256(got).hexdigest()}
        if len(got) <= 48 * 48 * 3:
            case["bytes"] = list(got)
        out.append(case)
    path = os.path.join(ROOT, "testdata", "golden", "pil", "resample.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"pillow": Image.__version__ if hasattr(Image, "__version__") else "", "cases": out}
    import PIL
    meta["pillow"] = PIL.__version__
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    print("wrote", path, len(out), "cases", file=sys.stderr)


if __name__ == "__main__":
    main()
