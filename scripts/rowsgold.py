#!/usr/bin/env python3
"""The band reference of TestComposedCPURowsServe for a fixture llama.cpp
cannot run, from the fixture's own transformers class (the user's decision for
such fixtures: the model's reference, not llama.cpp, not jitllm's CPU path).
-> testdata/golden/rows/<gguf name>.json

  $JITLLM_HF_PY scripts/rowsgold.py <transformers dir> <gguf name> [experts|serve] [q8|attn16] [seqs=FILE]

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

With "serve", the golden is TestComposedBlocksServe's, kept at a quarter of
the logits: its panel's prompts, read from engine/servepanel_test.go and
engine/composed_test.go, each encoded after the BOS and followed by the
teacher-forced ids engine/composed_test.go's trainedToks picks, as many as the
gate decodes after it. -> <gguf name>.serve.json
With seqs=FILE as well, the sequences are FILE's ("seqs", each its "ids" and
"toks", as a golden holds them): the panel as the engine encodes it, where
the GGUF tokenizes otherwise than tokenizer.json. bitnet-b1.58-2B-4T-i2_s.gguf
states no pre-tokenizer, so its ids are the default split's ("'s" two tokens,
"\n\n" two); its golden is regenerated from itself.

With "q8" as well, the class runs as the engine's numbers do on the fixture's
@q8 container (the user's decision for the mixture gates' reference): every
matrix convert.WithQ8 quantizes -- rows of whole 32-element blocks along the
contracted dimension (the second-last of an expert bank applied with
torch.bmm), the routers kept float -- dequantized as Q8_0 holds it
(d = amax/127 as binary16, codes rounded half away from zero), and the input of
each such matrix quantized as the composed kernels quantize it (a 32-element
window's d = amax/127 and 1/d in float32, codes trunc(x/d + copysign(0.5))).
-> <gguf name>.q8.json; the float golden is the quantization loss's reference.

With "attn16" as well, every attention rounds its four operands to binary16
-- q, k, the exponentials and v, the products and sums in float32, each row
divided by the sum of its unrounded exponentials -- as the matrix unit's
attention does and as engine/composed_test.go's bandFloor prices it. A
router whose choice the rounding can tip makes a step move by more than that
smooth bound: this run is where the serving gate measures by how much.
-> <gguf name>.attn16.json
"""
import base64, json, os, re, struct, sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AttentionInterface, AutoConfig, AutoModelForCausalLM

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
# Expert banks stored (in, out) and applied as torch.bmm(x, w): their blocks
# run along the second-last dimension.
BMM_EXPERTS = ("Llama4TextExperts",)


def away(t):
    """Rounds half away from zero, as Go's math.Round and llama.cpp do."""
    return torch.sign(t) * torch.floor(torch.abs(t) + 0.5)


def q8_weights(m, cfg):
    """The matrices convert.quantizeMatrices writes as Q8_0, dequantized in
    place; the storages that hold them, whose matmuls take int8 inputs."""
    experts = {getattr(cfg, k) for k in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
               if isinstance(getattr(cfg, k, None), int)}
    bmm = {id(p) for mod in m.modules() if type(mod).__name__ in BMM_EXPERTS for p in mod.parameters(recurse=False)}
    held = set()
    with torch.no_grad():
        for _, p in m.named_parameters():
            w = p.data.transpose(-1, -2).contiguous() if id(p) in bmm else p.data
            if p.ndim < 2 or w.shape[-1] % Q8:
                continue
            if p.ndim == 2 and (p.shape[0] in experts or p.shape[0] == 1):
                continue  # a router, or a shared expert's gate: kept float
            x = w.reshape(-1, Q8).to(torch.float64)
            amax = x.abs().amax(-1, keepdim=True).to(torch.float32)
            dd = amax / torch.tensor(127, dtype=torch.float32)
            inv = torch.where(dd != 0, 1 / dd, torch.zeros_like(dd))
            q = away((w.reshape(-1, Q8) * inv).to(torch.float64))
            d = (dd.to(torch.float16).to(torch.float32) * q.to(torch.float32)).reshape(w.shape)
            p.data.copy_(d.transpose(-1, -2) if id(p) in bmm else d)
            held.add(p.untyped_storage().data_ptr())
    return held


def q8_inputs(held):
    """F.linear and torch.bmm with each quantized matrix's input as the
    composed kernels quantize it, every other matmul as it was."""
    linear, bmm = F.linear, torch.bmm

    def codes(x):
        xs = x.to(torch.float32).reshape(-1, Q8)
        dd = xs.abs().amax(-1, keepdim=True) / torch.tensor(127, dtype=torch.float32)
        inv = torch.where(dd > 0, 1 / dd, torch.zeros_like(dd))
        t = xs * inv
        c = torch.trunc(t + torch.copysign(torch.full_like(t, 0.5), t))
        return (c * dd).reshape(x.shape).to(x.dtype)

    def q(x, w, b=None):
        if w.untyped_storage().data_ptr() not in held or x.shape[-1] % Q8:
            return linear(x, w, b)
        return linear(codes(x), w, b)

    def qb(x, w):
        if w.untyped_storage().data_ptr() not in held or x.shape[-1] % Q8:
            return bmm(x, w)
        return bmm(codes(x), w)

    F.linear = q
    torch.nn.functional.linear = q
    torch.bmm = qb


def attn16(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Causal attention over one unpadded sequence with binary16 operands."""
    h = lambda t: t.to(torch.float16).to(torch.float32)
    g = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(g, dim=1), value.repeat_interleave(g, dim=1)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    s = torch.matmul(h(query), h(key).transpose(2, 3)) * scaling
    n, m = s.shape[-2], s.shape[-1]
    s = s.masked_fill(torch.ones(n, m, dtype=torch.bool).triu(m - n + 1), float("-inf"))
    e = torch.exp(s - s.amax(-1, keepdim=True))
    out = torch.matmul(h(e), h(value)) / e.sum(-1, keepdim=True)
    return out.transpose(1, 2).contiguous(), None


ENGINE = os.path.join(HERE, "..", "engine")
GO = re.compile(r'"(?:[^"\\]|\\.)*"|\w+|[,+{}]')


def go_sum(toks, i, consts):
    """The Go string expression at toks[i] -- literals and string consts
    joined by + -- and the index past it."""
    out = ""
    while True:
        out += json.loads(toks[i]) if toks[i].startswith('"') else consts[toks[i]]
        if toks[i + 1] != "+":
            return out, i + 1
        i += 2


def serve_seqs(tok, cfg, embed):
    """TestComposedBlocksServe's panel: each prompt after the BOS, and the
    ids trainedToks forces after it (the first id at or after
    ids[i % len]*7 + i whose embedding row stores something)."""
    composed = open(os.path.join(ENGINE, "composed_test.go")).read()
    panel = open(os.path.join(ENGINE, "servepanel_test.go")).read()
    at = "const crossingPrompt ="
    consts = {"crossingPrompt": go_sum(GO.findall(composed[composed.index(at) + len(at):]), 0, {})[0]}
    toks = GO.findall(panel[panel.index("return []string{") + len("return []string{"):])
    texts, i = [], 0
    while toks[i] != "}":
        text, i = go_sum(toks, i, consts)
        texts.append(text)
        i += toks[i] == ","
    first = int(re.search(r"if pi == 0 \{\s*n = (\d+)", composed).group(1))
    steps = int(re.search(r"const panelSteps = (\d+)", panel).group(1))
    v, out = cfg.vocab_size, []
    for pi, text in enumerate(texts):
        ids = [cfg.bos_token_id] + tok.encode(text, add_special_tokens=False).ids
        n = first if pi == 0 else steps
        if len(ids) + n + 4 > cfg.max_position_embeddings:
            sys.exit("prompt %d overruns the context: the gate keeps its head, which this does not" % pi)
        forced = []
        for k in range(n):
            t = (ids[k % len(ids)] * 7 + k) % v
            while not embed[t].any():
                t = (t + 1) % v
            forced.append(t)
        out.append((ids, forced))
    return out


def main(d, name, experts=False, q8=False, serve=False, seqs_file=None, half=False):
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    ids = tok.encode(TEXT).ids
    cfg = AutoConfig.from_pretrained(d, experts_implementation="eager")
    if half:
        AttentionInterface.register("attn16", attn16)
        cfg._attn_implementation = "attn16"
    m = AutoModelForCausalLM.from_pretrained(d, config=cfg, dtype=torch.float32)
    m.eval()
    if serve:
        name += ".serve"
    if q8:
        q8_inputs(q8_weights(m, cfg))
        name += ".q8"
    if half:
        name += ".attn16"
    stride = max(1, cfg.vocab_size // (SAMPLES // 4 if experts or serve else SAMPLES))
    seqs = [(ids[:n], ids[n:n + DECODE]) for n in LENGTHS]
    seqs += [(p, ids[200:200 + BATCH_DECODE]) for p in (ids[:7], ids[40:73], ids[100:150])]
    steps = ids[200:200 + DECODE]
    if experts:
        seqs = [(ids[:1], ids[1:EXPERTS]), (ids[:1], ids[1:2 * EXPERTS])]
        seqs += [(ids[:LENGTHS[-1]], ids[LENGTHS[-1]:LENGTHS[-1] + DECODE])]
        seqs += [(p, steps) for p in (ids[:7], ids[40:73], ids[100:150])]
    if serve:
        seqs = serve_seqs(tok, cfg, m.get_input_embeddings().weight.detach())
    if seqs_file:
        seqs = [(q["ids"], q["toks"]) for q in json.load(open(seqs_file))["seqs"]]
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
                   "stride": stride, "ids": ids[:256], "seqs": out, "reference": "q8" if q8 else "attn16" if half else "float"}, f)


if __name__ == "__main__":
    seqs_file = next((a[len("seqs="):] for a in sys.argv[3:] if a.startswith("seqs=")), None)
    main(sys.argv[1], sys.argv[2], "experts" in sys.argv[3:], "q8" in sys.argv[3:], "serve" in sys.argv[3:], seqs_file,
         "attn16" in sys.argv[3:])
