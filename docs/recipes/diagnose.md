# Diagnose a slow or failed request

Read what the server says about a request before guessing: every error names
its cause, and every reply carries where its time went.

**Prerequisites:** `jitllm` and `jitllmd` on your `PATH`, `curl`.

`jitllm doctor`, where the build has it, checks the machine (devices, memory,
the container's compatibility) and prints the next action as JSON. The steps
below work on any build.

## Set up a server to look at

<!-- test: instead cp "$JITLLM_REPO/testdata/models/stories260K.gguf" . -->
```sh
curl -fLo stories260K.gguf https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
```

```sh
mkdir -p models
jitllm convert stories260K.gguf models/stories260K.jlm
```
<!-- test: expect models/stories260K.jlm -->

<!-- test: background -->
```sh
jitllmd serve -addr 127.0.0.1:4242 -models models -load stories260K.jlm -devices cpu
```

## 1. Is it up, and what is loaded?

```sh
for i in $(seq 60); do curl -sf http://127.0.0.1:4242/healthz && break; sleep 1; done
jitllmd models -addr 127.0.0.1:4242
```
<!-- test: expect ok -->
<!-- test: expect loaded (1) -->

`/healthz` answers `ok` once the server listens. `jitllmd models` lists the
loaded models (with their ids, architectures, context and where their blocks
are) and the containers on disk it could load.

## 2. A failed request: read the error

Every error is a JSON body naming the cause. Status codes: 400 for a request the
server will not run as asked, 404 for a model that is neither loaded nor in the models folder, 400 for a disabled one, 429 when the
model's queue is full (with `Retry-After`), 500 for an engine failure.

```sh
curl -s -w '\nHTTP %{http_code}\n' http://127.0.0.1:4242/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"no-such-model","prompt":"x"}'
```
<!-- test: expect no model \"no-such-model\" is loaded or in the models folder -->
<!-- test: expect HTTP 404 -->

```sh
curl -s -w '\nHTTP %{http_code}\n' http://127.0.0.1:4242/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"stories260K","messages":[{"role":"user","content":"hi"}]}'
```
<!-- test: expect carries no chat template -->
<!-- test: expect HTTP 400 -->

The second is a base model on the chat endpoint: refused rather than run as a
raw completion that would produce fluent text with no signal. The fix is in the
message.

A server that does not start says why on stderr and exits non-zero. A GGUF
given to `-load`:

<!-- test: exit 1 -->
```sh
cp stories260K.gguf models/
jitllmd serve -addr 127.0.0.1:8081 -models models -load stories260K.gguf -devices cpu
```
<!-- test: expect Convert it once:  jitllm convert -->

## 3. A slow request: read the reply's `jitllm` object

```sh
curl -s http://127.0.0.1:4242/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"stories260K","prompt":"Once upon a time","max_tokens":16,"temperature":0}'
```
<!-- test: expect "cpu_blocks":5 -->
<!-- test: expect "decode_tokens_per_second" -->

| Field | Read it as |
|---|---|
| `queued_ms`, `queue_depth_on_entry` | time waiting behind other requests: a load problem, not an engine one (`-max-batch`, `-max-queue`) |
| `prefill_ms` | the prompt's cost; long prompts dominate here. `restored_tokens` counts prompt tokens reused from the prefix cache (`-kv-cache`) |
| `decode_ms`, `decode_tokens_per_second` | the per-token rate |
| `device_blocks`, `cpu_blocks` | where the model's blocks ran; all on the host when you expected a GPU is the commonest surprise |
| `bytes_per_token` | weight bytes a token reads; divided into the memory bandwidth, the rate's ceiling |

## 4. Where memory went

```sh
jitllmd stats -addr 127.0.0.1:4242
```
<!-- test: expect page-in(s) -->

```text
model m-...   0 session(s)   16 tok generated, 5 prefilled
  resident 900.00 KiB of 4.16 GiB budget   5 of 5 block(s) in 5 frame(s)   FITS -- nothing is ever evicted, so a token reads nothing
  pager    5 page-in(s), 0 eviction(s), 920.00 KiB read in 6 request(s)
  batch    0 live / 0 waiting (width 512)   17 step(s), 0 joint
```

Evictions that keep climbing while you decode mean the budget holds fewer blocks
than a token reads: every token is paying disk reads. `FITS` means it is not.
`jitllmd stats -watch` streams the same counters. `jitllmd devices` is what
each device has free; `jitllmd place -model ID` the pager budget; [Memory
budgets and device placement](memory-and-placement.md) changes them.

## Checklist

| Symptom | First look | Usual cause |
|---|---|---|
| connection refused | the server's terminal | it exited at load; the reason is its last line |
| 404 | `curl http://127.0.0.1:4242/v1/models` | the `model` field names an id that is not loaded |
| 400 | the error message | a field or combination the server refuses by name |
| 429 | `queue_depth_on_entry` on the replies that did get in | more concurrent requests than `-max-queue` |
| slow first token | `prefill_ms`, `queued_ms` | a long prompt, a queue, or a cold device on its first request |
| slow every token | `device_blocks`, `bytes_per_token`, evictions in `jitllmd stats` | blocks on the host, or a budget that pages every token |
| garbage text | `/v1/completions` vs chat, the model's template | a base model asked to chat, or a reasoning model cut off mid-think |
