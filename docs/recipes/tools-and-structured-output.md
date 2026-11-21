# Tools and structured output

Let a model call your functions, force it to call one, and hold a reply to a
JSON schema.

**Prerequisites:** `jitllm` and `jitllmd` on your `PATH`, `curl`, and a model
whose chat template renders tools (Qwen3, Llama 3.x Instruct, Mistral Instruct,
gpt-oss and most current instruction models do; the tool list is rendered by the
model's own template). Python steps need `pip install openai`.

## Start a server

<!-- test: needs Qwen3-0.6B-Q8_0.jlm -->
<!-- test: instead mkdir -p models && ln -s "$JITLLM_TEST_MODELS/Qwen3-0.6B-Q8_0.jlm" models/Qwen3-0.6B-Q8_0.jlm -->
```sh
jitllm convert -o models qwen3-0.6b
```

<!-- test: background -->
```sh
jitllmd serve -addr 127.0.0.1:4242 -models models -load Qwen3-0.6B-Q8_0.jlm -id qwen3-0.6b -devices cpu
```

```sh
for i in $(seq 60); do curl -sf http://127.0.0.1:4242/healthz && break; sleep 1; done
```
<!-- test: expect ok -->

## Forced tool call

`tool_choice` naming a function makes the call certain: every sampled token is
held to the model's own tool-call syntax around that function's parameter
schema, so the arguments always parse.

```sh
curl -s http://127.0.0.1:4242/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b", "max_tokens": 200, "temperature": 0,
  "messages": [{"role": "user", "content": "What is the weather in Paris? /no_think"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}'
```
<!-- test: expect "finish_reason":"tool_calls" -->
<!-- test: expect "name":"get_weather" -->

```json
{"choices":[{"index":0,"finish_reason":"tool_calls",
  "message":{"role":"assistant","content":"<think>\n\n</think>",
    "tool_calls":[{"id":"call_tc-...","type":"function",
      "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}}]}
```

`tool_choice` takes `"auto"` (the default when `tools` is given: the model
decides), `"required"` (some call, any of the tools), `"none"` (the tools are
withheld from the prompt) or a named function, as above. Anthropic's
`/v1/messages` takes its own `tool_choice` (`auto`, `any`, `tool`) and returns
`tool_use` blocks.

## The round trip, in Python

Send the call's result back as a `tool` message, and the model answers with it:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4242/v1", api_key="unused")
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
messages = [{"role": "user", "content": "What is the weather in Paris? /no_think"}]

first = client.chat.completions.create(
    model="qwen3-0.6b", messages=messages, tools=tools, max_tokens=200, temperature=0,
    tool_choice={"type": "function", "function": {"name": "get_weather"}})
call = first.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)
print("called", call.function.name, args)

messages.append(first.choices[0].message)
messages.append({"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps({"city": args["city"], "sky": "sunny", "celsius": 21})})
final = client.chat.completions.create(
    model="qwen3-0.6b", messages=messages, tools=tools, max_tokens=200, temperature=0)
print(final.choices[0].message.content)
```
<!-- test: expect called get_weather -->
<!-- test: expect 21 -->

## Structured output

`response_format` with a `json_schema` holds every token of the reply to the
schema, and the reply ends where the schema does:

```sh
curl -s http://127.0.0.1:4242/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b", "max_tokens": 64, "temperature": 0,
  "messages": [{"role": "user", "content": "Give me a person. /no_think"}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {
    "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"]}}}}'
```
<!-- test: expect \"age\": -->

`{"type": "json_object"}` asks for any JSON object. Both work on
`/v1/completions` too, and on a base model.

A string's `pattern` is an ECMA-262 regular expression, unanchored as JSON
Schema reads it. A keyword or a pattern's feature the grammar builder does not
support is refused by name, not ignored:

```sh
curl -s http://127.0.0.1:4242/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b", "messages": [{"role": "user", "content": "x"}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "p",
    "schema": {"type": "string", "pattern": "(a)\\1"}}}}'
```
<!-- test: expect JSON Schema a backreference in a pattern is not supported -->

## When it does not work

| You see | Cause | Do this |
|---|---|---|
| `a forced tool call and response_format are one constraint each; give one` | both in one request | pick one; a forced call already constrains the arguments to the tool's schema |
| `JSON Schema ... is not supported` | a keyword the builder does not read, a number's `minimum` or `maximum`, a pattern's lookaround, backreference or `\b` | drop it and validate it in your code |
| `finish_reason` `length` with no `tool_calls` | a reasoning model spent `max_tokens` thinking before the call | raise `max_tokens`, or `/no_think` on Qwen3 |
| the model never calls a tool under `auto` | small models often answer instead | force it with `tool_choice` |
| a constrained reply runs no faster with speculation on | a constrained reply decodes a token a step, without speculation | nothing to do: speculation runs on the unconstrained requests |
