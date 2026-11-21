#!/usr/bin/env python3
"""Record embedding goldens for model.TestEmbeddingsMatchReference.

For each model it writes testdata/golden/embed/<name>.json holding, per input
text: the token ids llama.cpp's own tokenizer produces, the L2-normalised
embedding llama.cpp's llama-embedding computes on the CPU, and -- where the
model's sentence-transformers checkpoint is reachable -- the embedding the
original PyTorch model computes in float32, which is the authority the weights
were trained against (RULE 7m), with that tokenizer's ids beside it.

    HF_HOME=/path/to/hf-home \\
      $JITLLM_HF_PY scripts/embedgold.py [name ...]

llama.cpp ($JITLLM_LCPP) is the CUDA build of b10825 run with --device none, so its numbers
are the CPU backend's (F16 weights, F16 activations in the matmul).
"""
import json, os, subprocess, sys

import refenv  # where things are: environment variables only (docs/testing.md)

LCPP = refenv.lcpp_bin()
MODELS = os.path.join(refenv.models(), "embed")
OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "golden", "embed")

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "What is the capital of France?",
    "Héllo, WORLD! Ça va? naïve café -- 3.14159 x 2 = 6.28 (roughly) & #tags @you",
    "Embeddings map text to vectors; similar meanings land close together.",
    "日本語のテキストも少し含めます。中文也可以。",
    "Retrieval systems split documents into passages, embed each passage once, "
    "and store the vectors in an index. At query time the question is embedded "
    "with the same model and the nearest passages by cosine similarity are "
    "returned, which is why the embedding must be deterministic and normalised.",
]

# A text past 512 tokens, for the models whose context allows it. It is what
# reaches EmbeddingGemma's SYMMETRIC sliding window (256 either side) and a
# causal model's multi-chunk prefill; every short text above sits inside one
# window and one chunk.
LONG = " ".join(
    f"Section {i}: the {w} engine reads {i * 7} pages, pools {i % 5} vectors and "
    f"reports a cosine of 0.99{i:02d} against its reference, which is {w} enough."
    for i, w in enumerate(["first", "second", "third", "fourth", "fifth", "sixth",
                           "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
                           "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                           "eighteenth", "nineteenth", "twentieth"]))
LONG_OK = {"nomic-embed-text", "qwen3-embedding-0.6b", "qwen3-embedding-0.6b-f16", "embeddinggemma-300m",
           "embeddinggemma-300m-lcpp", "synth-gemma-embedding"}

# name -> (gguf, sentence-transformers repo or None, st kwargs[, llama.cpp gguf])
#
# A repo "hf:<name>" is the fixture embedfixture.py wrote to $CV_HF/<name>; the
# st kwarg "modules" keeps that many of the checkpoint's leading modules (and
# normalises), for a GGUF converted without the rest.
#
# The fourth entry is for a model whose Ollama GGUF llama.cpp cannot load:
# Ollama's embeddinggemma is arch "gemma3" with dense.0/dense.1, which b10825
# refuses ("wrong number of tensors; expected 316, got 314") because it wants
# arch "gemma-embedding". unsloth's conversion of the SAME BF16 weights is in
# llama.cpp's spelling, so it is the llama.cpp arm; google/embeddinggemma-300m
# is gated and unsloth's mirror of it is the sentence-transformers arm.
TABLE = {
    "all-minilm-l6": ("all-minilm-l6.gguf", "sentence-transformers/all-MiniLM-L6-v2", {}),
    "snowflake-arctic-embed-22m": ("snowflake-arctic-embed-22m.gguf", "Snowflake/snowflake-arctic-embed-xs", {}),
    "mxbai-embed-large": ("mxbai-embed-large.gguf", "mixedbread-ai/mxbai-embed-large-v1", {}),
    "nomic-embed-text": ("nomic-embed-text.gguf", "nomic-ai/nomic-embed-text-v1.5", {"trust_remote_code": True}),
    "qwen3-embedding-0.6b": ("qwen3-embedding-0.6b.gguf", "Qwen/Qwen3-Embedding-0.6B", {}),
    # The same model unquantized (Qwen's own f16 GGUF): Ollama's is Q8_0, which
    # bounds ANY engine at ~0.999 of the float32 model, so this is the file
    # that says whether the graph itself is right.
    "qwen3-embedding-0.6b-f16": ("qwen3-embedding-0.6b-f16.gguf", "Qwen/Qwen3-Embedding-0.6B", {}),
    "embeddinggemma-300m": ("embeddinggemma-300m.gguf", "unsloth/embeddinggemma-300m", {},
                            "embeddinggemma-300M-BF16-lcpp.gguf"),
    # The oracle's spelling of the same weights read by jitllm itself: arch
    # "gemma-embedding", no attention.causal key, and no dense heads, so its
    # float32 arm is the checkpoint's Transformer and Pooling alone.
    "embeddinggemma-300m-lcpp": ("embeddinggemma-300M-BF16-lcpp.gguf", "unsloth/embeddinggemma-300m",
                                 {"modules": 2}),
    "synth-gemma-embedding": ("synth-gemma-embedding.gguf", "hf:synth-gemma-embedding", {}),
}


def shim_transformers():
    """nomic-bert's remote code calls PreTrainedModel.get_extended_attention_mask,
    which transformers 5 removed. The mask it builds is the textbook one."""
    import torch, transformers
    if hasattr(transformers.PreTrainedModel, "get_extended_attention_mask"):
        return

    def gem(self, attention_mask, input_shape=None, device=None, dtype=None):
        dtype = dtype or torch.float32
        ext = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - ext) * torch.finfo(dtype).min
    transformers.PreTrainedModel.get_extended_attention_mask = gem


def lcpp(args):
    env = dict(os.environ, LD_LIBRARY_PATH=LCPP)
    return subprocess.run([os.path.join(LCPP, args[0])] + args[1:], env=env,
                          capture_output=True, text=True, check=True).stdout


def llama_ids(gguf, text):
    out = lcpp(["llama-tokenize", "-m", gguf, "-p", text, "--ids", "--log-disable"])
    line = [l for l in out.splitlines() if l.startswith("[")][-1]
    return json.loads(line)


def llama_embed(gguf, text):
    out = lcpp(["llama-embedding", "-m", gguf, "-p", text, "--embd-normalize", "2",
                "--embd-output-format", "json", "-ngl", "0", "--device", "none",
                "-t", "4", "-c", "2048", "-b", "2048", "-ub", "2048"])
    return json.loads(out)["data"][0]["embedding"]


def st_embed(repo, kw, texts):
    from sentence_transformers import SentenceTransformer, models
    shim_transformers()
    kw = dict(kw)
    keep = kw.pop("modules", 0)
    if repo.startswith("hf:"):
        repo = os.path.join(refenv.under("CV_HF", "hf"), repo[3:])
    m = SentenceTransformer(repo, device="cpu", **kw)
    if keep:
        m = SentenceTransformer(modules=[m[i] for i in range(keep)] + [models.Normalize()], device="cpu")
    # float32 throughout: the reference is the weights' own arithmetic, not a
    # half-precision approximation of it.
    m = m.float()
    m.max_seq_length = max(m.max_seq_length or 0, 2048)
    v = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=1)
    ids = [[int(x) for x in m.tokenize([t])["input_ids"][0]] for t in texts]
    return [[float(x) for x in row] for row in v], ids


def main(names):
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        gguf, repo, kw = TABLE[name][:3]
        lgguf = TABLE[name][3] if len(TABLE[name]) > 3 else gguf
        path = os.path.join(MODELS, lgguf)
        texts = TEXTS + ([LONG] if name in LONG_OK else [])
        g = {"model": name, "gguf": gguf, "llamacpp_gguf": lgguf, "texts": texts,
             "ids": [llama_ids(path, t) for t in texts],
             "llamacpp": [llama_embed(path, t) for t in texts]}
        if repo:
            try:
                g["st"], g["st_ids"] = st_embed(repo, kw, texts)
                g["st_repo"] = repo
            except Exception as e:  # recorded, not hidden
                g["st_error"] = f"{type(e).__name__}: {e}"[:300]
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(g, f)
        print(name, "ok", "st" in g, g.get("st_error", ""))


if __name__ == "__main__":
    main(sys.argv[1:] or list(TABLE))
