# Start a server and make the first request

Serve a model with `jitllmd` and get a completion, then a chat reply, from its
OpenAI-compatible API.

**Prerequisites:** `jitllm` and `jitllmd` on your `PATH`, `curl`, and about
1 GB of disk for the chat model. No GPU is needed.

## 1. Prove the install with the smallest model there is

`stories260K` is a 1 MB story model. It writes children's stories and nothing
else, which makes it a fast check that conversion, the server and the API work.

<!-- test: instead cp "$JITLLM_REPO/testdata/models/stories260K.gguf" . -->
```sh
curl -fLo stories260K.gguf https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
```

The engine reads only `.jlm` containers, so convert the GGUF once:

```sh
mkdir -p models
jitllm convert stories260K.gguf models/stories260K.jlm
```
<!-- test: expect models/stories260K.jlm -->

Start the server in its own terminal. `-devices cpu` keeps it off any GPU;
leave it out and jitllm picks a device by measuring.

<!-- test: background -->
```sh
jitllmd serve -addr 127.0.0.1:4242 -models models -load stories260K.jlm -devices cpu
```

It prints the addresses it serves once the model is loaded:

```text
loaded m-1791686969634-1 in 5ms
jitllmd on 127.0.0.1:4242
  connect   http://127.0.0.1:4242/jitllm.v1.ModelService/ListModels
  openai    http://127.0.0.1:4242/v1/chat/completions
  anthropic http://127.0.0.1:4242/v1/messages
```

Wait for it to answer, then list the models it serves. A model is addressed by
its file name without `.jlm` (or by the `-id` you give it):

```sh
for i in $(seq 60); do curl -sf http://127.0.0.1:4242/healthz && break; sleep 1; done
curl -s http://127.0.0.1:4242/v1/models
```
<!-- test: expect ok -->
<!-- test: expect "id":"stories260K" -->

Ask for a completion. `temperature` 0 is greedy, so the reply is the same on
every machine:

```sh
curl -s http://127.0.0.1:4242/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"stories260K","prompt":"Once upon a time","max_tokens":16,"temperature":0}'
```
<!-- test: expect there was a little girl named Lily -->

```json
{"id":"cmpl-oa-...","object":"text_completion","model":"stories260K",
 "choices":[{"index":0,"text":", there was a little girl named Lily. She loved to play","finish_reason":"length"}],
 "usage":{"prompt_tokens":5,"completion_tokens":16,"total_tokens":21},
 "jitllm":{"device_blocks":0,"cpu_blocks":5,"prefill_ms":1,"decode_tokens_per_second":5414.4}}
```

The `jitllm` object is the server's account of the request: where its blocks
ran, how long prefill took and the decode rate. [Diagnose a slow or failed
request](diagnose.md) reads it.

## 2. Chat with an instruction model

`stories260K` is a base model with no chat template, so `/v1/chat/completions`
refuses it (see the recovery table). Fetch a chat model from the catalog
(`jitllm library` lists it) and load it into the running server:

<!-- test: needs Qwen3-0.6B-Q8_0.jlm -->
<!-- test: instead ln -s "$JITLLM_TEST_MODELS/Qwen3-0.6B-Q8_0.jlm" models/Qwen3-0.6B-Q8_0.jlm -->
```sh
jitllm convert -o models qwen3-0.6b
```

```sh
jitllmd models -addr 127.0.0.1:4242 -load Qwen3-0.6B-Q8_0.jlm -id qwen3-0.6b -devices cpu
```
<!-- test: expect qwen3-0.6b -->

```sh
curl -s http://127.0.0.1:4242/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "Say hello in one word. /no_think"}],
  "max_tokens": 32, "temperature": 0}'
```
<!-- test: expect "role":"assistant" -->
<!-- test: expect "finish_reason":"stop" -->

Qwen3 reasons inside `<think>` before it answers; `/no_think` at the end of a
message asks it not to, and its reply then starts with an empty
`<think></think>`.

## When it does not work

| You see | Cause | Do this |
|---|---|---|
| `curl: (7) Failed to connect` | the server is not up yet, or exited | read its terminal; a model that failed to load is named there |
| `... is not a .jlm container, and inference reads nothing else` | `-load` was given a GGUF | run the `jitllm convert` command the message prints |
| `no model "X" is loaded or in the models folder` | the `model` field names nothing loaded and no file or alias in `-models` | use an `id` from `/v1/models` |
| `carries no chat template, so a chat request would run as a RAW COMPLETION` | a base model on `/v1/chat/completions` | send a `prompt` to `/v1/completions`, or load an instruction model |
| `misplaced flag` | a flag after a positional argument | put every flag before the file |
| the reply is only a `<think>` block, `finish_reason` `length` | a reasoning model ran out of `max_tokens` while thinking | raise `max_tokens`, or end the message with `/no_think` on Qwen3 |
| `address already in use` | another process holds the port | pick another `-addr` |
