#!/usr/bin/env python3
"""Write nn/testdata/yarn.json: transformers' YaRN inverse frequencies.

The oracle for nn.YarnFreqs is transformers' own _compute_yarn_parameters, run
on gpt-oss's rotary config with truncate both ways. Run with the HF venv:

    $JITLLM_HF_PY scripts/yarngold.py
"""
import json

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import _compute_yarn_parameters

cases = []
for truncate in (False, True):
    for head_dim, base, factor, orig in ((64, 150000.0, 32.0, 4096), (128, 1000000.0, 4.0, 32768)):
        cfg = PretrainedConfig()
        cfg.head_dim = head_dim
        cfg.hidden_size = head_dim * 8
        cfg.num_attention_heads = 8
        cfg.max_position_embeddings = orig * int(factor)
        cfg.rope_parameters = {
            "rope_type": "yarn", "rope_theta": base, "factor": factor,
            "original_max_position_embeddings": orig, "beta_fast": 32.0,
            "beta_slow": 1.0, "truncate": truncate,
        }
        inv, att = _compute_yarn_parameters(cfg, "cpu")
        cases.append({"head_dim": head_dim, "base": base, "factor": factor, "orig": orig,
                      "truncate": truncate, "inv_freq": [float(v) for v in inv], "attention_factor": att})
json.dump(cases, open("nn/testdata/yarn.json", "w"), indent=1)
print(len(cases), "cases")
