#!/usr/bin/env python3
"""transformers' own class running a GGUF's weights, over a fixed id sequence.

    python scripts/ggufref.py <hf dir> <model.gguf> <out.bin> <nprompt> <id,id,...>

Every parameter of the float checkpoint's model is replaced by the GGUF's
dequantized tensor of the same role -- gguf-py's own name map, with its
conversion's transforms undone (A = -exp(A_log), the squeezed conv1d) -- so
the reference computes in float32 on exactly the weights an engine reading
that GGUF runs. Quantization is then not a variable: what separates the two is
the graph and the engine's own arithmetic. It writes the logits of every row
from the prompt's last on as raw little-endian f32, which
TestTeacherForcedAgainstReference (engine, -tags jitllmbench) reads, and
prints each row's top three.

It settled Jamba (model-correctness.md, Mamba-1): on its F16 file jitllm,
both llama.cpp arms and this reference agree at every position; on its Q4_K_M
file the reference parts from all three on the story prompt.

The whole model is held in memory (scripts/phimoestream.py streams a layer at
a time, for Phi-3.5-MoE). Architectures: jamba, mamba, falcon_mamba.
"""
import sys
import numpy as np
import torch
import gguf
from gguf.quants import dequantize
from transformers import AutoConfig, AutoModelForCausalLM

torch.set_num_threads(int(__import__("os").environ.get("THREADS", "16")))
d, gp, out, npr, ids = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(x) for x in sys.argv[5].split(",")]
cfg = AutoConfig.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
r = gguf.GGUFReader(gp)
T = {t.name: t for t in r.tensors}
arch = {"jamba": gguf.MODEL_ARCH.JAMBA, "mamba": gguf.MODEL_ARCH.MAMBA,
        "falcon_mamba": gguf.MODEL_ARCH.MAMBA}[cfg.model_type]
tmap = gguf.get_tensor_name_map(arch, cfg.num_hidden_layers)
sd = m.state_dict()
done = 0
with torch.no_grad():
    for k, p in sd.items():
        name = tmap.get_name(k, try_suffixes=(".weight", ".bias"))
        if name is None:
            raise SystemExit(f"no GGUF name for {k}")
        if name not in T:
            if k == "lm_head.weight" and "output.weight" not in T:
                continue  # tied: the embedding stands in
            raise SystemExit(f"{k} -> {name} is not in the GGUF")
        t = T[name]
        a = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32))
        if k.endswith(".A_log"):
            a = torch.log(-a)
        p.copy_(a.reshape(p.shape))
        done += 1
    if getattr(cfg, "tie_word_embeddings", False):
        m.tie_weights()
    print("replaced", done, "of", len(sd), flush=True)
    lg = m(torch.tensor([ids])).logits[0, npr - 1:].float()
np.ascontiguousarray(lg.numpy(), dtype=np.float32).tofile(out)
for i, row in enumerate(lg):
    v, ix = row.topk(3)
    print("pos", i, " | ".join(f"{int(j)} {float(x):.4f}" for x, j in zip(v, ix)), flush=True)
