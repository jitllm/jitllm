#!/usr/bin/env python3
"""The chat templates tok/jinja is gated on, and transformers' renderings of them.

    $JITLLM_HF_PY scripts/chatgold.py fetch [file ...]
    JITLLM_CHATGOLD=$PWD/chat-contexts.json ./scripts/cap 8G -- go test ./tok/jinja -run TestChatTemplatesMatchTransformers
    $JITLLM_HF_PY scripts/chatgold.py render chat-contexts.json
    rm chat-contexts.json

fetch re-reads every template named in tok/jinja/testdata/templates/manifest.json
from its source and rewrites the .jinja file beside it, with the BOS and EOS text
the template interpolates. A source is the model's own artefact, the one
`jitllm convert` stores: a GGUF's tokenizer.chat_template (read from the local
copy under $JITLLM_MODELS, since a GGUF's metadata is not fetchable alone), or
the publisher's tokenizer_config.json / chat_template.jinja on Hugging Face (the
`-chat-template` path, and the only artefact of a family no GGUF on this box
carries). A gated repo is read through a verbatim mirror, named in the entry.

render reads the contexts the Go test dumps -- one per (template, case), the
exact JSON model/ binds: messages, tools and documents (null when the
request carries none), add_generation_prompt and the special tokens -- and
renders each with transformers' render_jinja_template, the renderer
apply_chat_template calls, which binds the same names. It writes
testdata/golden/<template>.json: {case: {"ctx": sha, "out": text}} or
{case: {"ctx": sha, "raise": message, "by": exception type}}, the sha being
the first 16 hex digits of the context's SHA-256.

strftime_now is pinned to the date the Go test pins (chatGoldDate).
"""
import datetime, hashlib, json, os, sys, urllib.request

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
JINJA = os.path.join(HERE, "..", "tok", "jinja", "testdata")
TEMPLATES = os.path.join(JINJA, "templates")
GOLDEN = os.path.join(JINJA, "golden")
MANIFEST = os.path.join(TEMPLATES, "manifest.json")
FIXED = datetime.datetime(2024, 7, 26)


def get(repo, path):
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{path}") as r:
        return r.read().decode("utf-8")


def token_text(t):
    if isinstance(t, dict):
        return t.get("content", "")
    return t or ""


def from_hf(src):
    repo, path = src["repo"], src["path"]
    tc = json.loads(get(repo, "tokenizer_config.json"))
    if path == "tokenizer_config.json":
        ct = tc.get("chat_template")
        if isinstance(ct, list):
            body = {x["name"]: x["template"] for x in ct}[src.get("name", "default")]
        else:
            body = ct
    else:
        body = get(repo, path)
    if not body:
        sys.exit(f"{repo}/{path} carries no chat template")
    return body, token_text(tc.get("bos_token")), token_text(tc.get("eos_token"))


def from_gguf(src):
    import gguf
    p = os.path.join(refenv.models(), src["local"])
    if not os.path.exists(p):
        sys.exit(f"{p} is missing: download {src['repo']}/{src['file']} there (RULE 11)")
    r = gguf.GGUFReader(p)
    key = "tokenizer.chat_template" + ("." + src["name"] if src.get("name") else "")
    body = r.fields[key].contents()
    toks = r.fields["tokenizer.ggml.tokens"].contents()

    def tok(k):
        f = r.fields.get(k)
        return toks[int(f.contents())] if f is not None else ""

    return body, tok("tokenizer.ggml.bos_token_id"), tok("tokenizer.ggml.eos_token_id")


def fetch(only):
    m = json.load(open(MANIFEST))
    for e in m:
        if only and e["file"] not in only:
            continue
        src = e["source"]
        body, bos, eos = from_gguf(src) if "local" in src else from_hf(src)
        with open(os.path.join(TEMPLATES, e["file"]), "w", encoding="utf-8", newline="") as f:
            f.write(body)
        e["bos"], e["eos"] = bos, eos
        print(f"{e['file']}: {len(body.encode())} bytes, bos {bos!r} eos {eos!r}")
    write_manifest(m)


def write_manifest(m):
    # One entry a line, so a refresh reads as a diff of the rows it moved.
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(json.dumps(e, ensure_ascii=False) for e in m) + "\n]\n")


def render(dump):
    from transformers.utils.chat_template_utils import render_jinja_template
    out = {}
    for d in json.load(open(dump)):
        ctx = json.loads(d["ctx"])
        tpl = open(os.path.join(TEMPLATES, d["template"]), encoding="utf-8").read()
        try:
            text, _ = render_jinja_template(
                [ctx.pop("messages")], tools=ctx.pop("tools"), documents=ctx.pop("documents"),
                chat_template=tpl, add_generation_prompt=ctx.pop("add_generation_prompt"),
                strftime_now=lambda fmt: FIXED.strftime(fmt), **ctx)
            rec = {"out": text[0]}
        except Exception as ex:
            # "TemplateError" is the template's own raise_exception.
            rec = {"raise": str(ex), "by": type(ex).__name__}
        rec["ctx"] = hashlib.sha256(d["ctx"].encode()).hexdigest()[:16]
        out.setdefault(d["template"], {})[d["case"]] = rec
    os.makedirs(GOLDEN, exist_ok=True)
    for t, cases in out.items():
        stem = t[: -len(".jinja")]
        with open(os.path.join(GOLDEN, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=1, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        raised = sum(1 for c in cases.values() if "raise" in c)
        print(f"{stem}: {len(cases)} cases, {raised} raise")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("fetch", "render"):
        sys.exit(__doc__)
    if sys.argv[1] == "fetch":
        fetch(set(sys.argv[2:]))
    else:
        render(sys.argv[2])
