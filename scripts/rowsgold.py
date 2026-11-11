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

With "experts", the golden is the mixture gates' instead, kept at a quarter
of the logits: TestComposedCPUExpertPagesServe's sequence, the text's first id
and its next EXPERTS-1 forced a step each; TestComposedCPUBlocksServe's, the
same over 2*EXPERTS-1; and TestComposedCPUExpertPromptsServe's, the longest
prompt with the decode tokens after it and the batch of three prompts with the
steps after each.

With "q8" as well, the class runs as the engine's numbers do on the fixture's
@q8 container (the user's decision for the mixture gates' reference): every
matrix convert.WithQ8 quantizes -- rows of whole 32-element blocks along the
contracted dimension, the routers kept float -- dequantized as Q8_0 holds it
(d = amax/127 as binary16, codes rounded half away from zero), and the input of
each such matrix quantized as the composed kernels quantize it (a 32-element
window's d = amax/127 and 1/d in float32, codes trunc(x/d + copysign(0.5))).
-> <gguf name>.q8.json; the float golden is the quantization loss's reference.
"""
import base64, json, os, struct, sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "golden", "rows")

TEXT = ("The river rose every spring until the bridges were islands, "
        "and the clerk counted barrels of salt on the upper floor. ") * 60
LENGTHS, DECODE = [5, 33, 100], 6
# TestComposedCPURowsServe's batch steps its prompts as many times as its
# median needs rows (engine/cpurows_test.go's batchDecode).
BATCH_DECODE = 20
SAMPLES = 4096
EXPERTS = 40


Q8 = 32


def away(t):
    """Rounds half away from zero, as Go's math.Round and llama.cpp do."""
    return torch.sign(t) * torch.floor(torch.abs(t) + 0.5)


def q8_weights(m, cfg):
    """The matrices convert.quantizeMatrices writes as Q8_0, dequantized in
    place; the storages that hold them, whose matmuls take int8 inputs."""
    experts = {getattr(cfg, k) for k in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
               if isinstance(getattr(cfg, k, None), int)}
    held = set()
    with torch.no_grad():
        for _, p in m.named_parameters():
            if p.ndim < 2 or p.shape[-1] % Q8:
                continue
            if p.ndim == 2 and (p.shape[0] in experts or p.shape[0] == 1):
                continue  # a router, or a shared expert's gate: kept float
            x = p.data.reshape(-1, Q8).to(torch.float64)
            amax = x.abs().amax(-1, keepdim=True).to(torch.float32)
            dd = amax / torch.tensor(127, dtype=torch.float32)
            inv = torch.where(dd != 0, 1 / dd, torch.zeros_like(dd))
            q = away((p.data.reshape(-1, Q8) * inv).to(torch.float64))
            p.data.copy_((dd.to(torch.float16).to(torch.float32) * q.to(torch.float32)).reshape(p.shape))
            held.add(p.untyped_storage().data_ptr())
    return held


def q8_inputs(held):
    """F.linear with each quantized matrix's input as the composed kernels
    quantize it, every other matmul as it was."""
    linear = F.linear

    def q(x, w, b=None):
        if w.untyped_storage().data_ptr() not in held or x.shape[-1] % Q8:
            return linear(x, w, b)
        xs = x.to(torch.float32).reshape(-1, Q8)
        dd = xs.abs().amax(-1, keepdim=True) / torch.tensor(127, dtype=torch.float32)
        inv = torch.where(dd > 0, 1 / dd, torch.zeros_like(dd))
        t = xs * inv
        codes = torch.trunc(t + torch.copysign(torch.full_like(t, 0.5), t))
        return linear((codes * dd).reshape(x.shape).to(x.dtype), w, b)

    F.linear = q
    torch.nn.functional.linear = q


def main(d, name, experts=False, q8=False):
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    ids = tok.encode(TEXT).ids
    cfg = AutoConfig.from_pretrained(d, experts_implementation="eager")
    m = AutoModelForCausalLM.from_pretrained(d, config=cfg, dtype=torch.float32)
    m.eval()
    if q8:
        q8_inputs(q8_weights(m, cfg))
        name += ".q8"
    stride = max(1, cfg.vocab_size // (SAMPLES // 4 if experts else SAMPLES))
    seqs = [(ids[:n], ids[n:n + DECODE]) for n in LENGTHS]
    seqs += [(p, ids[200:200 + BATCH_DECODE]) for p in (ids[:7], ids[40:73], ids[100:150])]
    steps = ids[200:200 + DECODE]
    if experts:
        seqs = [(ids[:1], ids[1:EXPERTS]), (ids[:1], ids[1:2 * EXPERTS])]
        seqs += [(ids[:LENGTHS[-1]], ids[LENGTHS[-1]:LENGTHS[-1] + DECODE])]
        seqs += [(p, steps) for p in (ids[:7], ids[40:73], ids[100:150])]
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
                   "stride": stride, "ids": ids[:256], "seqs": out, "reference": "q8" if q8 else "float"}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "experts" in sys.argv[3:], "q8" in sys.argv[3:])
