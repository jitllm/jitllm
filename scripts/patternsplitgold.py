"""Ask the tokenizers library how it splits the texts tok's pattern gates
disagree on.

tok's TestSplitPatternsMatchTheStages and TestModelsOwnPatternsMatchThePipeline
append each text the pattern engine and a stage split differently to the file
JITLLM_PATTERN_DIFFS names, one JSON list per line. This reads that file and,
for every entry, runs the same pattern (or the whole pre_tokenizer) through the
tokenizers library -- the code the model's own tokenizer.json is written for --
and reports which of the two splits it agrees with.

Usage: python3 scripts/patternsplitgold.py DIFFS.jsonl [OUT.json]

OUT, when given, receives each entry as the library splits it -- the stage
(its pattern and behavior, or the dumped pre_tokenizer's file), the text and
the library's pieces -- the form tok/testdata/pattern_split.json takes.
"""

import json
import sys

from tokenizers import Regex, Tokenizer, pre_tokenizers

BEHAVIORS = {"Isolated": "isolated", "Removed": "removed", "MergedWithPrevious": "merged_with_previous",
             "MergedWithNext": "merged_with_next", "Contiguous": "contiguous"}
GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def without_byte_level(node):
    """The pre_tokenizer with each ByteLevel stage replaced by the split it
    runs (its regex when use_regex, nothing otherwise), so pieces stay text."""
    t = node.get("type")
    if t == "Sequence":
        out = []
        for p in node["pretokenizers"]:
            q = without_byte_level(p)
            if q is not None:
                out.append(q)
        return {"type": "Sequence", "pretokenizers": out}
    if t == "ByteLevel":
        if node.get("use_regex", True):
            return {"type": "Split", "pattern": {"Regex": GPT2}, "behavior": "Isolated", "invert": False}
        return None
    return node


def tokenizer_for(pre):
    body = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": pre, "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
    }
    return Tokenizer.from_str(json.dumps(body))


def pieces(entry, cache):
    if entry.get("PreTokenizer"):
        key = entry["PreTokenizer"]
        if key not in cache:
            cache[key] = tokenizer_for(without_byte_level(json.loads(key)))
        tok = cache[key]
        return [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(entry["Text"])]
    key = (entry["Pattern"], entry["Behavior"])
    if key not in cache:
        cache[key] = pre_tokenizers.Split(Regex(entry["Pattern"]), BEHAVIORS[entry["Behavior"]])
    return [p for p, _ in cache[key].pre_tokenize_str(entry["Text"])]


def main():
    entries = []
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.extend(json.loads(line))
    cache = {}
    tally = {}
    for e in entries:
        lib = pieces(e, cache)
        e["Library"] = lib
        verdict = "engine" if lib == (e["Engine"] or []) else "stage" if lib == (e["Stage"] or []) else "neither"
        what = e.get("Pattern") or json.loads(e["PreTokenizer"]).get("type")
        tally.setdefault((what[:60], verdict), 0)
        tally[(what[:60], verdict)] += 1
        if verdict != "engine":
            print(verdict, json.dumps(e["Text"]), "lib", json.dumps(lib), "engine", json.dumps(e["Engine"]))
    for (what, verdict), n in sorted(tally.items()):
        print(f"{n:6d} {verdict:8s} {what}")
    if len(sys.argv) > 2:
        out = [{k: e[k] for k in ("Pattern", "Behavior", "File", "Text", "Library") if e.get(k) is not None}
               for e in entries]
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write("[\n" + ",\n".join(json.dumps(e, ensure_ascii=False) for e in out) + "\n]\n")


if __name__ == "__main__":
    main()
