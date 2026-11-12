#!/usr/bin/env python3
"""Record how Idefics3's processor (SmolVLM's) sizes and cuts pictures, once.
-> testdata/golden/idefics3/sizes.json

  $JITLLM_HF_PY scripts/idefics3sizes.py

THE ORACLE FOR PreLongTiles' GEOMETRY (convert.longTileGeometry). For each
picture size of a sweep -- square, wide, tall, a pixel a side, a side past
the longest edge, an odd side the processor makes even -- and two of the
published configs (SmolVLM-256M's longest edge 2048 over 512 tiles, the
2.2B's 1536 over 384), transformers' Idefics3ImageProcessor on a black
picture of that size: the size its first resize takes the picture to
(get_resize_output_image_size) and the rows and columns of tiles it cuts
(return_row_col_info), 0 x 0 for a picture of one square.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = [(2048, 512), (1536, 384)]
SIZES = [(640, 488), (488, 640), (512, 512), (2048, 2048), (100, 3000), (3000, 100), (1, 1), (1, 2000),
         (2000, 1), (1023, 517), (777, 1999), (4000, 3000), (515, 2048), (33, 34), (384, 385), (1537, 1)]


def main():
    import torch
    import transformers
    from transformers import Idefics3ImageProcessor
    from transformers.models.idefics3.image_processing_idefics3 import get_resize_output_image_size

    out = []
    for longest, tile in CONFIGS:
        proc = Idefics3ImageProcessor(size={"longest_edge": longest}, max_image_size={"longest_edge": tile})
        for w, h in SIZES:
            h1, w1 = get_resize_output_image_size(torch.zeros(3, h, w), resolution_max_side=longest)
            img = Image.fromarray(np.zeros((h, w, 3), np.uint8))
            got = proc(images=[[img]], return_tensors="pt", return_row_col_info=True)
            out.append({"longest": longest, "tile": tile, "w": w, "h": h, "w1": int(w1), "h1": int(h1),
                        "rows": int(got["rows"][0][0]), "cols": int(got["cols"][0][0]),
                        "squares": int(got["pixel_values"].shape[1])})
    path = os.path.join(ROOT, "testdata", "golden", "idefics3", "sizes.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"transformers": transformers.__version__, "cases": out}, fh, indent=1)
        fh.write("\n")
    print("wrote", path, len(out), "cases", file=sys.stderr)


if __name__ == "__main__":
    main()
