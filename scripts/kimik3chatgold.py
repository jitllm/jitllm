#!/usr/bin/env python3
"""Kimi K3's chat as its release renders it, the golden convert's transcription is held to.

    python3 -I scripts/kimik3chatgold.py DIR testdata/golden/kimik3_chat.json

DIR is a Kimi-K3 release directory: its encoding_k3.py is the renderer
tokenization_kimi.py's apply_chat_template calls, and it imports nothing but
the standard library. Each case is rendered as apply_chat_template renders it
with tokenize=False: build_chat_segments' segments joined, thinking on and
thinking_effort "max" unless the case names them (apply_chat_template's
defaults). The JSON keeps each case's messages, add_generation_prompt, the
thinking and thinking_effort it set (null where it took the default) and the
text, or the exception a case raises.

The release ships no Jinja template (tokenizer_config.json has no
chat_template and the repo no chat_template.jinja), so convert/kimik3_chat.jinja
transcribes the system, user and assistant path, and
convert.TestKimiK3ChatTemplateMatchesItsRenderer renders every case here
through it.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from encoding_k3 import build_chat_segments  # noqa: E402

U = lambda c, **kw: {"role": "user", "content": c, **kw}  # noqa: E731
S = lambda c, **kw: {"role": "system", "content": c, **kw}  # noqa: E731
A = lambda c, **kw: {"role": "assistant", "content": c, **kw}  # noqa: E731

CASES = [
    ("one user turn", [U("Hello")], True, None, None),
    ("system then user", [S("You are terse."), U("What is 2+2?")], True, None, None),
    ("a past exchange", [U("Hi"), A("Hello! How can I help?"), U("Tell me a joke.")], True, None, None),
    ("kept reasoning", [U("2+2?"), A("4", reasoning_content="Add them."), U("And 3+3?")], True, None, None),
    ("blank reasoning", [U("2+2?"), A("4", reasoning_content="  \n"), U("Again?")], True, None, None),
    ("reasoning field", [U("2+2?"), A("4", reasoning="Count up."), U("Next?")], True, None, None),
    ("escaped names", [U("hey", name='Ann & "Bo"'), A("hi", name="kimi")], True, None, None),
    ("text parts", [U([{"type": "text", "text": "Look: "}, {"type": "text", "text": "twice"}])], True, None, None),
    ("no generation prompt", [U("Hello"), A("Hi there")], False, None, None),
    ("whitespace kept", [U("  line one\nline two  \n"), A("\n ok \n")], True, None, None),
    ("unicode", [U("Привет, 世界 ✓")], True, None, None),
    ("thinking off", [U("Hello")], True, False, None),
    ("thinking off with history", [U("Hi"), A("Hello", reasoning_content="dropped"), U("Bye")], True, False, None),
    ("low effort", [U("Hello")], True, None, "low"),
    ("high effort with history", [S("Be brief."), U("Hi"), A("Hey"), U("Bye")], True, None, "high"),
    ("system alone", [S("Only a system turn.")], True, None, None),
]

out = []
for name, msgs, gen, thinking, effort in CASES:
    case = {"name": name, "messages": msgs, "add_generation_prompt": gen, "thinking": thinking,
            "thinking_effort": effort}
    try:
        segs = build_chat_segments(msgs, add_generation_prompt=gen,
                                   thinking=True if thinking is None else thinking,
                                   thinking_effort="max" if effort is None else effort)
        case["text"] = "".join(s.text for s in segs)
    except Exception as e:  # noqa: BLE001
        case["raise"] = str(e)
    out.append(case)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("%d cases" % len(out))
