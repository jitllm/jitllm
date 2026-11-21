#!/usr/bin/env python3
"""Build the embedding fixtures: a transformers class with every parameter
randomised, wrapped in the sentence-transformers modules a published
checkpoint carries, and written to GGUF by llama.cpp's own converter.

    $JITLLM_HF_PY scripts/embedfixture.py [name ...]
    then: $JITLLM_HF_PY scripts/embedgold.py <name>   (the golden)

-> $CV_HF/<name>/ (the checkpoint, the golden's float32 arm) and
   $JITLLM_MODELS/embed/<name>.gguf

The tokenizer is $CV_TOK/embeddinggemma/ (google/embeddinggemma-300m's
tokenizer.json, tokenizer.model, tokenizer_config.json, special_tokens_map.json
and added_tokens.json; unsloth/embeddinggemma-300m mirrors them ungated).

synth-gemma-embedding is Gemma3TextModel with use_bidirectional_attention,
mean pooling and two dense heads (64 -> 96 -> 64), converted with
--sentence-transformers-dense-modules: the file is general.architecture
"gemma-embedding", which states no attention.causal, no sliding-window pattern
and no attention scale. Seven layers, so the pattern's period of six puts a
sliding layer after the global one; the window is 8 positions (transformers
keeps |i - j| < 8 // 2 + 1), which every golden text crosses; the score scale
is query_pre_attn_scalar = head_dim, the only scale the file can state.

Every parameter is randomised (the norms about one); the class's one buffer
that is a value, embed_scale, is sqrt(hidden_size) computed from the config and
not saved, so no file can state another, and the rotary's inverse frequencies
are the config's bases.
"""
import json
import os
import shutil
import subprocess
import sys

import torch

import refenv  # where things are: environment variables only (docs/testing.md)

MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
TOK_FILES = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json"]


def gemma_embedding():
    from transformers import Gemma3TextConfig, Gemma3TextModel
    S, F = "sliding_attention", "full_attention"
    cfg = Gemma3TextConfig(vocab_size=262144, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=7, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, query_pre_attn_scalar=16, sliding_window=8,
                           layer_types=[S, S, S, S, S, F, S], max_position_embeddings=2048,
                           rope_parameters={S: {"rope_type": "default", "rope_theta": 10000.0},
                                            F: {"rope_type": "default", "rope_theta": 1000000.0}},
                           use_bidirectional_attention=True, rms_norm_eps=1e-6,
                           pad_token_id=0, eos_token_id=1, bos_token_id=2)
    return Gemma3TextModel, cfg, 8, [96]


SPECS = {"synth-gemma-embedding": gemma_embedding}


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


def build(name):
    from sentence_transformers import SentenceTransformer, models
    cls, cfg, window, dense = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261119)
    m = cls(cfg).float().eval()
    perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in TOK_FILES:
        shutil.copy(os.path.join(TOKDIR, "embeddinggemma", f), d)
    # The checkpoint states the window it was trained with; the class's
    # constructor narrows it for bidirectional attention when it reads it.
    cp = os.path.join(d, "config.json")
    c = json.load(open(cp))
    c["sliding_window"] = window
    json.dump(c, open(cp, "w"), indent=2)

    tr = models.Transformer(d, max_seq_length=2048)
    mods = [tr, models.Pooling(cfg.hidden_size, pooling_mode="mean")]
    width = cfg.hidden_size
    for out in dense + [cfg.hidden_size]:
        lin = models.Dense(width, out, bias=False, activation_function=torch.nn.Identity())
        with torch.no_grad():
            lin.linear.weight.copy_(torch.randn_like(lin.linear.weight) * 0.2)
        mods.append(lin)
        width = out
    mods.append(models.Normalize())
    SentenceTransformer(modules=mods, device="cpu").save(d)
    c = json.load(open(cp))
    c["sliding_window"] = window
    json.dump(c, open(cp, "w"), indent=2)

    dst = os.path.join(MODELS, "embed", name + ".gguf")
    subprocess.run([PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                    "--sentence-transformers-dense-modules", "--outfile", dst], check=True)
    print("wrote", dst)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(SPECS):
        build(n)
