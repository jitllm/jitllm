#!/usr/bin/env python3
"""The band reference of TestComposedCPURowsServe for a fixture llama.cpp
cannot run, from the fixture's own transformers class (the user's decision for
such fixtures: the model's reference, not llama.cpp, not jitllm's CPU path).
-> testdata/golden/rows/<gguf name>.json

  $JITLLM_HF_PY scripts/rowsgold.py <transformers dir> <gguf name> [experts]

The gate's sequences, as engine/cpurows_test.go builds them: its text's ids,
a prompt of each length with the decode tokens after it, and a batch of three
prompts with the steps after each. Each sequence is run teacher-forced in
float32, and its rows -- the logits after the prompt's last token, then after
each forced token, as scripts/lcpplogits writes them -- are kept at every
stride-th logit (a vocabulary of 150 thousand logits a row would make the
golden tens of megabytes), base64 little-endian float32. The gate holds its
reference to these rows over the same logits, and fails where its own ids
differ from the golden's.

With "experts", the golden is TestComposedCPUExpertPagesServe's instead: one
sequence, the text's first id and its next EXPERTS-1 forced a step each, kept
at a quarter of the logits -- a mixture fixture's gate reads no other.
"""
import base64, json, os, struct, sys
import torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "golden", "rows")

TEXT = ("The river rose every spring until the bridges were islands, "
        "and the clerk counted barrels of salt on the upper floor. ") * 60
LENGTHS, DECODE = [5, 33, 100], 6
SAMPLES = 4096
EXPERTS = 40


def main(d, name, experts=False):
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    ids = tok.encode(TEXT).ids
    cfg = AutoConfig.from_pretrained(d)
    m = AutoModelForCausalLM.from_pretrained(d, config=cfg, dtype=torch.float32)
    m.eval()
    stride = max(1, cfg.vocab_size // (SAMPLES // 4 if experts else SAMPLES))
    seqs = [(ids[:n], ids[n:n + DECODE]) for n in LENGTHS]
    steps = ids[200:200 + DECODE]
    seqs += [(p, steps) for p in (ids[:7], ids[40:73], ids[100:150])]
    if experts:
        seqs = [(ids[:1], ids[1:EXPERTS])]
    out = []
    for p, t in seqs:
        with torch.no_grad():
            lg = m(torch.tensor([p + t])).logits[0]
        rows = []
        for r in range(len(p) - 1, len(p) + len(t)):
            v = lg[r][::stride].tolist()
            rows.append(base64.b64encode(struct.pack("<%df" % len(v), *v)).decode())
        out.append({"ids": p, "toks": t, "rows": rows})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump({"source": os.path.basename(os.path.normpath(d)), "class": cfg.model_type,
                   "stride": stride, "ids": ids[:256], "seqs": out}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] == ["experts"])
