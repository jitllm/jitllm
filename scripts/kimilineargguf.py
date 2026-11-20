#!/usr/bin/env python3
"""Write synth-kimilinear's GGUF with llama.cpp's own converter.

    JITLLM_MODELS=... LCPP_SRC=... $JITLLM_HF_PY scripts/kimilineargguf.py

-> $JITLLM_MODELS/synth-kimilinear-f32.gguf

The weights are scripts/hfgold.py's synth-kimilinear, every parameter and
buffer randomised, so testdata/golden/hf/synth-kimilinear.json (transformers'
KimiLinearForCausalLM over those weights) is the golden of both containers:
the one converted from the safetensors and the one converted from this GGUF.

llama.cpp's converter reads Moonshot's checkpoint layout, and the fixture is
the one transformers saves. Two things differ: the config (Moonshot's
linear_attn_config, first_k_dense_replace and moe_router_activation_func,
which llama.cpp's loader needs as expert_gating_func; transformers'
layer_types, linear_head_dim and mlp_layer_types) and the dense blocks' MLP
(Moonshot's mlp.*, transformers' block_sparse_moe.*). The converter's input
restates both in Moonshot's form, the config's values beside the transformers
keys and each tensor's bytes unchanged; the tokenizer is the fixture's own.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from safetensors.torch import load_file, save_file

import refenv

MODELS = refenv.models()
LCPP = refenv.lcpp_src()
SRC = os.path.join(MODELS, "synth-kimilinear")
DST = os.path.join(MODELS, "synth-kimilinear-f32.gguf")


def moonshot(cfg):
    kinds = cfg["layer_types"]
    mlp = cfg["mlp_layer_types"]
    lead = next((i for i, k in enumerate(mlp) if k != "dense"), len(mlp))
    if mlp[lead:].count("dense"):
        sys.exit("mlp_layer_types has a dense block after a sparse one")
    out = dict(cfg)
    out["linear_attn_config"] = {
        "full_attn_layers": [i + 1 for i, k in enumerate(kinds) if k == "full_attention"],
        "kda_layers": [i + 1 for i, k in enumerate(kinds) if k == "linear_attention"],
        "head_dim": cfg["linear_head_dim"],
        "num_heads": cfg["linear_num_heads"],
        "short_conv_kernel_size": cfg["linear_conv_kernel_dim"],
    }
    out["first_k_dense_replace"] = lead
    out["moe_layer_freq"] = 1
    out["mla_use_nope"] = True
    out["moe_router_activation_func"] = "sigmoid"
    return out


def main():
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    d = tempfile.mkdtemp(prefix="kimilinear-gguf-")
    try:
        for f in os.listdir(SRC):
            if f not in ("config.json", "model.safetensors"):
                os.symlink(os.path.realpath(os.path.join(SRC, f)), os.path.join(d, f))
        dense = re.compile(r"^(model\.layers\.\d+\.)block_sparse_moe\.((?:gate|up|down)_proj\.weight)$")
        w = load_file(os.path.join(SRC, "model.safetensors"))
        save_file({dense.sub(r"\1mlp.\2", k): v for k, v in w.items()}, os.path.join(d, "model.safetensors"))
        json.dump(moonshot(cfg), open(os.path.join(d, "config.json"), "w"), indent=2)
        subprocess.run([sys.executable, os.path.join(LCPP, "convert_hf_to_gguf.py"), d,
                        "--outtype", "f32", "--outfile", DST], check=True)
    finally:
        shutil.rmtree(d)
    print("wrote", DST)


if __name__ == "__main__":
    main()
