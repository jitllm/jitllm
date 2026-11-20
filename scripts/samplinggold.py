#!/usr/bin/env python3
"""A GGUF stating every sampling key generation.json reads, written by
llama.cpp's own converter.

    $JITLLM_HF_PY scripts/samplinggold.py

-> $JITLLM_MODELS/synth-sampling.gguf

synth-qwen2 (scripts/hfgold.py's, in $JITLLM_MODELS) with a generation_config.json
stating a draw -- do_sample, temperature, top_k, top_p, min_p and
repetition_penalty -- converted by convert_hf_to_gguf.py under LCPP_SRC, which
writes each field the file states as its general.sampling key (gguf-py's
Metadata.load). No published GGUF here states min_p; this one states every key
the converter reads, at the values SAMPLING below, which
convert.TestGGUFGenerationReadsItsSamplingKeys holds the container to.
"""
import json
import os
import shutil
import subprocess
import sys

import refenv  # where things are: environment variables only (docs/testing.md)

SAMPLING = {"do_sample": True, "temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
            "repetition_penalty": 1.1}


def main():
    models = refenv.models()
    lcpp = refenv.lcpp_src()
    src = os.path.join(models, "synth-qwen2")
    work = os.path.join(models, "synth-sampling.src")
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(src, work)
    with open(os.path.join(work, "generation_config.json")) as f:
        gen = json.load(f)
    gen.update(SAMPLING)
    with open(os.path.join(work, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=1)
    out = os.path.join(models, "synth-sampling.gguf")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.path.join(lcpp, "gguf-py"),
                                                                       os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, os.path.join(lcpp, "convert_hf_to_gguf.py"), work, "--outtype", "f32",
                    "--outfile", out], check=True, env=env)
    shutil.rmtree(work)
    print(out)


if __name__ == "__main__":
    main()
