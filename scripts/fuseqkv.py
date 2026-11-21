#!/usr/bin/env python3
"""fuseqkv.py in.gguf outdir blocks: the first blocks of a GGUF written twice,
outdir/split.gguf with its q, k and v projections as they are and
outdir/fused.gguf with each block's q, k and v (weights, and biases) joined
into one attn_qkv in that order, every other key and tensor the same.

The converter reads the fused file's head width off attn_qkv's rows where
it states no attention.key_length, and convert.TestAFusedQKVConvertsAsItsParts
holds the two containers to one. Made from Phi-tiny-MoE-instruct-Q8_0.gguf
(sixteen heads of 128 on a 4096-wide model, no key_length), two blocks:

    python3 scripts/fuseqkv.py Phi-tiny-MoE-instruct-Q8_0.gguf $JITLLM_MODELS/fusedqkv/phitiny 2

Each part is laid out where the fused tensor stands, so the containers'
tensor order agrees too. gguf-py is the reader and writer (GGUF_PY, or an
installed gguf).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('GGUF_PY', 'gguf-py'))
from gguf import GGUFReader, GGUFValueType, GGUFWriter

src, out, blocks = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = GGUFReader(src)
arch = r.fields['general.architecture'].contents()
os.makedirs(out, exist_ok=True)


def block_of(name):
    if not name.startswith('blk.'):
        return -1
    return int(name.split('.')[1])


tensors = [t for t in r.tensors if block_of(t.name) < blocks]
by = {t.name: t for t in tensors}


def write(path, fuse):
    w = GGUFWriter(path, arch=arch)
    for k, f in r.fields.items():
        if k.startswith('GGUF.') or k == 'general.architecture':
            continue
        vt = f.types[0]
        if k == arch + '.block_count':
            w.add_uint32(k, blocks)
            continue
        if vt == GGUFValueType.ARRAY:
            w.add_key_value(k, f.contents(), vt, sub_type=f.types[-1])
        else:
            w.add_key_value(k, f.contents(), vt)
    done = set()
    for t in tensors:
        if t.name in done:
            continue
        parts = None
        for suffix in ('weight', 'bias'):
            for role in ('attn_q', 'attn_k', 'attn_v'):
                if t.name.endswith('.' + role + '.' + suffix):
                    b = t.name[:t.name.index('.' + role)]
                    parts = [by[b + '.' + x + '.' + suffix] for x in ('attn_q', 'attn_k', 'attn_v')]
                    fused = b + '.attn_qkv.' + suffix
        if parts is None:
            w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
            continue
        done.update(p.name for p in parts)
        if fuse:
            d = np.concatenate([np.asarray(p.data) for p in parts], axis=0)
            w.add_tensor(fused, d, raw_shape=d.shape, raw_dtype=parts[0].tensor_type)
        else:
            for p in parts:
                w.add_tensor(p.name, p.data, raw_shape=p.data.shape, raw_dtype=p.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


write(os.path.join(out, 'split.gguf'), False)
write(os.path.join(out, 'fused.gguf'), True)
for n in ('split.gguf', 'fused.gguf'):
    g = GGUFReader(os.path.join(out, n))
    print(n, len(g.tensors), [(t.name, list(t.shape)) for t in g.tensors if t.name.startswith('blk.0.attn')])
