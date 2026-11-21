#!/usr/bin/env python3
"""Encode a fixed corpus with Moonshot's own tokenizer and write the ids.

The authority for a Kimi model's tokenization is tokenization_kimi.py: a
tiktoken Encoding over tiktoken.model with the pat_str in that file. Its
encode matches specials written in the text by default
(allow_special_tokens=True, allowed_special="all"), and reads them as
characters with allow_special_tokens=False (disallowed_special=()): "ids"
is the first, which tok.Vocab.EncodeSpecial answers, and "plain" the
second, tok.Vocab.Encode's. This runs that class itself, so the golden is
the reference's behaviour rather than a re-implementation of it.

    $JITLLM_HF_PY scripts/tiktokgold.py \
        models/Kimi-Linear-48B-A3B-Instruct > tok/testdata/kimi_tiktoken.json
"""
import importlib.util, json, sys, os

d = sys.argv[1]
spec = importlib.util.spec_from_file_location("tokenization_kimi", os.path.join(d, "tokenization_kimi.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tk = m.TikTokenTokenizer.from_pretrained(d)

corpus = [
    "The capital of France is",
    "getUserName iPhone helloWorld HTTPServer parseJSONResponse",
    "中文abc 你好，世界！こんにちは 한국어 ⺀々豈𮯰",
    "  leading spaces, double  spaces\tand\ttabs\nnew\n\nlines\r\n",
    "Numbers: 1234567 3.14159 1,000,000 2026-09-24 0x1F",
    "It's they're we've I'm you'll he'd CAN'T",
    "emoji 🙂🚀 and café naïve Zürich ﬁ",
    "def f(x):\n    return x**2  # comment\n",
    "[BOS]hello[EOS] <|im_user|>user<|im_middle|>hi<|im_end|>",
    "path/to/file.go https://example.com/a?b=c",
    "ſ K 'ſ 'K",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
out = [{"text": t, "ids": tk.encode(t), "plain": tk.encode(t, allow_special_tokens=False)} for t in corpus]
json.dump({"source": "tokenization_kimi.TikTokenTokenizer over tiktoken.model sha256 b6c497a7...",
           "cases": out}, sys.stdout, ensure_ascii=False, indent=1)
