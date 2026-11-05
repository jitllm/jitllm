#!/usr/bin/env python3
"""Render model.TestToolPromptMatchesTransformers' fixture through transformers.

    JITLLM_TOOLGOLD=dump go test ./engine -run TestToolPromptMatchesTransformers
    $JITLLM_HF_PY scripts/toolgold.py

Reads testdata/golden/tools/<model>.tpl.json (the container's own template,
bos and eos, dumped by the Go test) and writes <model>.want.txt with
transformers' render_jinja_template -- the renderer apply_chat_template uses.
The fixture below must stay the same conversation as toolFixtureMessages and
toolFixtureTools in engine/tools_test.go, down to the empty assistant content
the engine passes.
"""
import datetime, glob, json, os
from transformers.utils.chat_template_utils import render_jinja_template

# The date every template prints (strftime_now), pinned: the Go test renders
# with the same clock (toolGoldDate in engine/tools_test.go), so a golden
# made on one day still matches on the next.
FIXED = datetime.datetime(2024, 7, 26)

TOOLS = json.loads('[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}, "format": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}]')
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "a1b2c3d4e", "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Paris", "format": "celsius"}}}]},
    {"role": "tool", "content": '{"temperature": 18, "sky": "cloudy"}', "tool_call_id": "a1b2c3d4e", "name": "get_weather"},
]

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata", "golden", "tools")
for f in sorted(glob.glob(os.path.join(here, "*.tpl.json"))):
    d = json.load(open(f))
    if "tools" not in d["template"] and "tool_calls" not in d["template"]:
        # A template that reads neither tools nor a tool call: the Go test
        # asserts jitllm refuses the request rather than rendering it without them.
        continue
    out = render_jinja_template([MESSAGES], tools=TOOLS, chat_template=d["template"],
                                add_generation_prompt=True, bos_token=d["bos"], eos_token=d["eos"],
                                strftime_now=lambda fmt: FIXED.strftime(fmt))
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, list):
        out = out[0]
    name = os.path.basename(f)[: -len(".tpl.json")]
    open(os.path.join(here, name + ".want.txt"), "w").write(out)
    print(f"{name}: {len(out)} chars")
