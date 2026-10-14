#!/usr/bin/env python3
"""Write format/quant/testdata/tq.json: TQ1_0 and TQ2_0 blocks and gguf-py's decoding of them.

The oracle for quant's ternary decoders is an implementation that is not ours
-- the pip `gguf` package's TQ1_0/TQ2_0.dequantize_blocks. Run it with the HF
reference venv:

    $JITLLM_HF_PY scripts/tqgold.py [model.gguf ...]

Blocks: every byte value in every position (TQ2_0's code 3, which no quantizer
writes, included; TQ1_0's bytes that are no quantizer's output too, since the
format defines what they decode to) at a sweep of d that reaches subnormal
halves, the largest finite one and negatives, plus each format's quantizer
output over random weights, plus the first blocks of a real tensor of each type
from any GGUF given.
"""
import json
import struct
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import TQ1_0, TQ2_0

halves = [0x3C00, 0x3800, 0x0001, 0x03FF, 0x0400, 0x1000, 0x2E00, 0x7BFF, 0x5000, 0xBC00, 0xB800, 0x8001, 0xC400, 0x0000]
rng = np.random.default_rng(1)

out = {}
for cls, nq, name in ((TQ2_0, 64, "TQ2_0"), (TQ1_0, 52, "TQ1_0")):
    bb = nq + 2
    blocks = []
    for i, d in enumerate(halves):
        qs = bytes((i * 37 + j * 11) % 256 for j in range(nq))
        blocks.append(qs + struct.pack("<H", d))
    for i in range(16):
        qs = rng.integers(0, 256, nq, dtype=np.uint8).tobytes()
        blocks.append(qs + struct.pack("<H", halves[i % len(halves)]))
    w = rng.standard_normal((8, 256)).astype(np.float32)
    w = np.where(np.abs(w) < 0.4, 0, np.sign(w)) * np.float32(0.0137)
    q = cls.quantize_blocks(w)
    blocks += [bytes(b) for b in q]
    src = "synthetic only"
    for path in sys.argv[1:]:
        r = GGUFReader(path)
        t = next((t for t in r.tensors if t.tensor_type.name == name), None)
        if t is None:
            continue
        raw = bytes(np.asarray(t.data).reshape(-1).view(np.uint8)[: bb * 64])
        blocks += [raw[i : i + bb] for i in range(0, len(raw), bb)]
        src = t.name + " of " + path.rsplit("/", 1)[-1]
        break
    arr = np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(-1, bb)
    vals = cls.dequantize_blocks(arr).astype(np.float32).reshape(-1)
    out[name] = {
        "source": "gguf-py " + name + ".dequantize_blocks; real blocks from " + src,
        "blocks": b"".join(blocks).hex(),
        "bits": [int(v) for v in vals.view(np.uint32)],
    }
    print(name, len(blocks), "blocks;", src)

json.dump(out, open("format/quant/testdata/tq.json", "w"))
