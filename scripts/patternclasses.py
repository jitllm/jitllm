"""The code points the tokenizers library's regex reads as \\w, \\d and \\s,
and as each property the pre-split patterns name, printed as ranges: the
fixture tok.TestClassesAreTheLibrarys holds the pattern engine's classes to.

Usage: python3 scripts/patternclasses.py > tok/testdata/pattern_classes.txt
"""
import sys
from tokenizers import Regex, pre_tokenizers

CLASSES = [r"\w", r"\d", r"\s", r"\p{L}", r"\p{N}", r"\p{Nd}", r"\p{P}", r"\p{S}", r"\p{M}", r"\p{Han}",
           r"\p{Lu}", r"\p{Ll}", r"\p{Lt}", r"\p{Lm}", r"\p{Lo}"]


def members(cls):
    sp = pre_tokenizers.Split(Regex(cls), "removed")
    out = set()
    step = 4096
    for lo in range(0, 0x110000, step):
        cps = [c for c in range(lo, min(lo + step, 0x110000)) if not 0xD800 <= c <= 0xDFFF]
        text = "".join(chr(c) for c in cps)
        kept = set()
        for piece, _ in sp.pre_tokenize_str(text):
            kept.update(ord(ch) for ch in piece)
        out.update(c for c in cps if c not in kept)
    return out


def ranges(s):
    s = sorted(s)
    out = []
    for c in s:
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


for cls in CLASSES:
    m = members(cls)
    print(cls, len(m), " ".join(f"{a:X}-{b:X}" for a, b in ranges(m)))
    sys.stdout.flush()
