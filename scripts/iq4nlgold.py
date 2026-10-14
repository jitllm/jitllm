#!/usr/bin/env python3
"""Write format/quant/testdata/iq4_nl.json: IQ4_NL blocks and gguf-py's decoding of them.

The oracle for quant's IQ4_NL decoder is an implementation that is not ours --
the gguf package's IQ4_NL.dequantize_blocks. Run it with the HF reference venv:

    $JITLLM_HF_PY scripts/iq4nlgold.py [model.gguf]

Blocks: every code in every slot (a rotation of 0..15 through the 32 slots), at
a sweep of d that reaches subnormal, the largest finite half and negatives,
plus the first blocks of a real IQ4_NL tensor when a GGUF (or the first
16 MB of one) is given.
"""
import json
import struct
import sys

import numpy as np
from gguf.quants import IQ4_NL

halves = [0x3C00, 0x3800, 0x0001, 0x03FF, 0x0400, 0x1000, 0x2E00, 0x7BFF, 0x5000, 0xBC00, 0xB800, 0x8001, 0xC400, 0x0000]
blocks = []
for i, d in enumerate(halves):
    for rot in range(16):
        q = [(l + rot + i) % 16 for l in range(32)]
        qs = bytes(q[l] | (q[l + 16] << 4) for l in range(16))
        blocks.append(struct.pack("<H", d) + qs)

src = "synthetic only"
if len(sys.argv) > 1:
    from gguf import GGUFReader

    r = GGUFReader(sys.argv[1])
    t = next(t for t in r.tensors if t.tensor_type.name == "IQ4_NL")
    raw = bytes(np.asarray(t.data).reshape(-1).view(np.uint8)[: 18 * 128])
    blocks += [raw[i : i + 18] for i in range(0, len(raw), 18)]
    src = t.name

arr = np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(-1, 18)
with np.errstate(over="ignore"):
    vals = IQ4_NL.dequantize_blocks(arr).astype(np.float32).reshape(-1)
out = {
    "source": "gguf-py IQ4_NL.dequantize_blocks; real blocks from " + src,
    "blocks": b"".join(blocks).hex(),
    "bits": [int(v) for v in vals.view(np.uint32)],
}
json.dump(out, open("format/quant/testdata/iq4_nl.json", "w"))
print(len(blocks), "blocks")
