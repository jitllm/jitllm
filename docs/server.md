# Server reference

[Back to the README](../README.md)

## The server

`jitllmd serve` serves these surfaces on one address.

| Surface | Endpoints |
|---|---|
| OpenAI-compatible | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `POST /v1/embeddings` (floats or base64), `/v1/files`, `/v1/batches` |
| Anthropic-compatible | `POST /v1/messages` (named SSE events) |
| Connect RPC | `ModelService`, `InferenceService` (`Generate`, `Complete`, `Cancel`, `Embed`), `SessionService`, `PlacementService`, `DeviceService`, `TelemetryService` |
| jitllm | `POST /v1/systemone` (a decision model's answers), `GET /v1/capabilities` (the manifest, [capabilities.md](capabilities.md)), `GET /metrics` (Prometheus) |
| llama.cpp-compatible | `POST /completion`, `/infill`, `/tokenize`, `/detokenize`, `/apply-template`, `/embedding`; `GET /props`, `/slots` (see below) |
| Health | `GET /healthz` (plain text), `GET /health` and `GET /v1/health` (`{"status": "ok"}`, as llama-server answers) |

### llama.cpp's native API

The routes above are llama-server's own, so its clients run unmodified.
`/completion` and `/infill` build the same `GenerateOptions` as the OpenAI
routes and call the one generate path; `/embedding` calls the one embed path.
Responses carry llama-server's names: `content`, `tokens` (with
`return_tokens`, always in a stream), `stop`, `stop_type` (`eos`, `word`,
`limit`), `stopping_word`, `tokens_predicted`, `tokens_evaluated`,
`tokens_cached`, `timings`, `generation_settings`, `truncated`. A stream is
`data: {json}` frames with no `[DONE]`, the last carrying `stop: true`. Errors
are `{"error": {"code", "message", "type"}}`.

llama-server serves one model, so a request names none: it is the loaded model.
With several loaded, add a `model` field (an extension) holding an id from
`/v1/models`. Absent sampling fields take llama-server's defaults (temperature
0.8, top_k 40, top_p 0.95, min_p 0.05), not the OpenAI routes'. `prompt` is a
string, token ids, or strings and ids mixed; `grammar` and `json_schema`
constrain the output; `n_probs` reports the raw distribution (at most 20).
`/infill` needs a model with fill-in-the-middle tokens. `cache_prompt` is
accepted either way: the prefix cache is automatic and never changes the
output. `POST /props` answers 501, as llama-server does without `--props`.

A field the engine cannot honour is refused by name with a 400, at any value
but its neutral one: `mirostat`, `dry_multiplier`, `xtc_probability`,
`typical_p`, `dynatemp_range`, `presence_penalty`, `frequency_penalty`,
`logit_bias`, `samplers` beyond penalties/top_k/top_p/min_p/temperature,
`n_cmpl` above 1, `id_slot` other than -1 and 0, `n_predict` 0, `n_keep`,
`n_cache_reuse`, `t_max_predict_ms`, `timings_per_token`, `return_progress`,
`post_sampling_probs`, `response_fields`, `lora`, `repeat_last_n` other than
64 or 0, `embd_normalize` other than 2, a list of prompts, and multimodal
prompts. A field llama.cpp does not define is refused as unknown. On a
byte-level BPE vocabulary `/tokenize` with `parse_special: false` is refused
when the text contains a control token, because the tokenizer matches it either
way.

Not served: `/reranking`, `/embeddings`, `/lora-adapters`, `/models`, slot
save/restore/erase, and llama-server's own `/metrics` names (`/metrics` stays
jitllm's).

A client checks what it is talking to before it relies on it:
`TelemetryService.GetServerInfo` returns the version and revision, the
control plane's compatibility version (`api_version`), the OS and
architecture, the models folder, every device, the capability manifest that
`GET /v1/capabilities` serves (`capabilities_json`, which also carries
`server_api`) and a list of named `features`. The API version changes only
when a field changes meaning or goes away; a client built for another number
refuses with a message naming both. `server/client.Discover` is that check
for a Go front end, and `internal/localserver.Probe` the same check over
`/v1/capabilities` for the root module's CLI.

Both compatible APIs stream and take `tools`: the tool list is rendered by the
model's own chat template, and calls come back as OpenAI `tool_calls` or
Anthropic `tool_use` blocks. `ignore_eos` on the OpenAI endpoints and the
Connect `GenerateRequest` (`jitllmd run -ignore-eos`) keeps generating past the
stop tokens until `max_tokens` or the end of the session's context. A request
with no `max_tokens` runs until the model ends its reply or fills the context;
either way a reply that reaches the end of the context finishes as `length`. Generates of one model placed wholly on one device, or wholly on the host,
decode together as rows of one step (`-max-batch`). A model's queue holds
`-max-queue` requests (64 by default) before it answers 429 with a
`Retry-After`. `-fairness 0-100` (100 by default) sets how they share the
steps: 0 serves them first come first served, each to its end; higher
time-slices running rows for waiting ones. Under a model's KV budget (`Engine.SetKVBudget`, an embedding
API with no flag yet) a generate parks idle sessions of the model, lowest
priority then least recently stepped, into its memory cache, and a parked
session resumes byte for byte at its next generate.

`logprobs` is OpenAI's on both endpoints: on chat `logprobs: true` with
`top_logprobs` 0..20 (`choices[].logprobs.content`), on `/v1/completions` an
alternative count 0..20 (`tokens`, `token_logprobs`, `top_logprobs`,
`text_offset`), streamed or not. The values are the model's raw distribution
-- the log-softmax of the logits, before temperature, the repeat penalty and
the top-k/top-p/min-p cuts -- which is what vLLM returns by default
(`logprobs_mode=raw_logprobs`); they do not move with the request's
temperature. The log-softmax and the top-N selection are generated kernels on
the host, where the logits of every sampled token are. `logprobs` with `echo`
(the prompt's own log-probabilities) is refused.

`n` (1..128) asks for several continuations of one prompt. The prompt is
prefilled once: the first choice seals its pages and final logits into a
request-scoped prompt store and every other choice restores them, so usage
counts the prompt once (`jitllm.prompt_restored` lists the positions each
choice restored). A seeded request's choice `i` uses `seed + i` and equals the
same request sent alone with that seed; an unseeded one draws distinct seeds
(`jitllm.seeds`). The choices decode one after another; streamed, each
choice's chunks carry its `index`, and the usage comes in a last chunk of its
own.

The Files and Batch APIs are OpenAI's: `POST /v1/files` (multipart, `purpose`
`batch`), `GET /v1/files[/{id}[/content]]`, `DELETE /v1/files/{id}`, and
`POST /v1/batches`, `GET /v1/batches[/{id}]`, `POST /v1/batches/{id}/cancel`.
A batch validates every line first (`custom_id`, `POST`, the batch's
`endpoint`: `/v1/chat/completions`, `/v1/completions` or `/v1/embeddings`),
then runs each line in the background through the same handler the live
route uses, one after another, and writes the output and error files as jsonl.
Files and batch records are kept under `.jitllm-batch` in the model directory
(`Config.BatchDir`), so they survive a restart; a batch the restart
interrupted is reported failed.

Structured output: OpenAI's `response_format` (`json_object`, or
`json_schema` with its `schema`) on `/v1/chat/completions` and
`/v1/completions`, and `grammar` (GBNF) or `json_schema` on the Connect
`GenerateRequest`. Every sampled token keeps the reply inside the grammar and
the reply ends where the grammar does. A schema keyword the converter does not
build (`pattern`, `format`, numeric bounds, `allOf`, ...) is a 400 naming it;
the supported subset is in
[model-correctness.md](engineering-history/model-correctness.md),
"engine/grammar". A constrained generate decodes alone and without
speculation, and refuses `ignore_eos`.

Speculative decoding is off by default and asked for per request
(`speculation` on `GenerateRequest`, `jitllm_speculate` on the OpenAI
endpoints) or per session (`speculation` on `CreateSessionRequest`). It
drafts with the model's own prediction block where it carries one and by
prompt lookup otherwise, and verifies the drafts in one pass. It is for
greedy decoding: greedy output is plain greedy's token for token, while a
sampled request with prompt lookup drafts nothing. A speculative generate
prefills alone (not through the step loop or the session's prompt store), a
`continue_session` generate decodes plainly, and a session whose speculative
reply was cut by a stop string inside a round refuses `continue_session`.

Pictures: on a model whose container carries a vision tower, a chat request
may carry pictures -- OpenAI `image_url` content parts on
`/v1/chat/completions`, Anthropic `image` blocks on `/v1/messages`. PNG, JPEG,
WebP and GIF (its first frame) are decoded, inline as a `data:` URL
(`data:image/png;base64,...`) or an Anthropic `base64` source whose
`media_type` must match the bytes. The pictures keep the order the client sent
them and go where the model's own chat template puts its image markers (a
turn's pictures before its text); each is a span of the prompt, named by its
`ImageKey`, so the prefix cache names the rows after it as it names text. A picture is preprocessed and encoded by the model's tower exactly
as `jitllm run -image` does it.

| Limit | Default | `jitllmd serve` flag |
|---|---|---|
| encoded bytes per picture | 20 MiB | `-image-max` |
| pixels (width x height) per picture, read from the header before decoding | 4096 x 4096 | `-image-max-pixels` |
| pictures per request | 8 | `-image-max-count` |
| encoded bytes of all of a request's pictures | 64 MiB | `-image-max-request` |
| decoding one picture | 5 s | `-image-decode-timeout` |
| remote `http(s)` URLs | not fetched | `-image-fetch-allow host1,host2` |

Inline pictures -- `data:` URLs, Anthropic `base64` sources and Connect
bytes -- are always taken, under the limits above and nothing else. Remote
fetching is off unless the operator lists the hosts it may reach, because a
server that fetches what a request names is a proxy into its own network.
With `-image-fetch-allow` (`Config.Images.FetchAllow`) an `http(s)` image URL
or Anthropic `url` source is fetched only when:

- its host, lower-cased, in IDNA ASCII form and without a trailing dot, is
  on the list exactly, or under an entry `*.example.com` by exactly one label
  (nothing matches by suffix); the first URL and every redirect are checked;
- it is https (`-image-fetch-http` allows plain http), on port 443 or 80
  (`-image-fetch-ports`), with no userinfo and no IPv6 zone, and no redirect
  goes from https to http; at most three redirects are followed, through no
  proxy;
- the address each connection is about to be made to -- after resolution, so
  a name cannot pass a check with one address and connect to another -- is in
  none of the special-use ranges of the IANA IPv4 and IPv6 special-purpose
  registries: this network, private, shared, loopback, link-local, the IETF
  protocol blocks (Teredo), documentation, 6to4, benchmarking, multicast,
  reserved, broadcast, unspecified, IPv4-mapped, both NAT64 prefixes,
  discard-only and unique-local;
- fewer than `-image-fetch-concurrency` (4) fetches are running across the
  server, and it finishes within `-image-fetch-timeout` (10 s) with at most
  64 KiB of headers and the byte limits above.

Every refusal is a
400 naming the part (`messages[0].content[1].image_url ...`): a picture sent
to a model with no tower ("model X does not accept images"), an unsupported
or mismatched type, a picture over a limit, undecodable bytes, a remote URL
with fetching off (the message says how to turn it on), and an audio or file
part. The Connect API takes pictures as `ChatMessage.images` (`ChatImage`:
the bytes and an optional `media_type`, or a `url` that is a `data:` URL or an
`http(s)` URL under the fetch rules above) under the same limits;
`ApplyChatTemplate` refuses one. Pictures combine with `tools` (the template
renders the tool list, and a forced `tool_choice` runs under the tool
grammar), with `n` (the prompt, picture included, is prefilled once and the
other choices restore it), and with `continue_session` (the session's history
already holds its earlier pictures; the new turn's follow). A request with
pictures prefills alone rather than as a row of the step loop.

With each other: `logprobs` on a constrained reply are the raw distribution's,
before the grammar's mask, as vLLM's default does; `n` with a grammar
constrains every choice; speculation with `logprobs` or with `n` is refused
by name (a round decides several tokens from one verification and reads back
no per-token distribution, and the choices of `n` prefill through a shared
prompt store a Speculator does not use).

| `jitllmd` verb | What it does |
|---|---|
| `serve` | run the daemon (`-addr`, `-api-key`, `-allow-origin`, `-models`, `-load`, `-id`, `-devices`, `-maxmem`, `-max-batch`, `-max-queue`, `-kv-cache`; `jitllmd serve -h` lists every flag) |
| `run` | stream a generation from a running daemon |
| `models` | list, describe, load and unload models (`-unload ID -force` closes sessions holding it instead of refusing) |
| `devices` | the hardware and the memory that is actually spendable |
| `sessions` | create, list, describe, reset and close sessions |
| `place` | read and move the CPU/device seam and the pager budget |
| `stats` | engine counters, once or streamed (`-watch`) |
| `service` | `install`, `uninstall`, `start`, `stop`, `status`: jitllmd as this person's background service |

## Who can reach it

The server is for the machine it runs on until told otherwise.

- **Address.** `-addr` defaults to `localhost:4242`, loopback only: the server
  listens on `127.0.0.1:4242` and `[::1]:4242` (the second where the host has IPv6), so
  a client that resolves `localhost` to either reaches it, and never on `0.0.0.0` or
  `::`. Clients dial `localhost` at its loopback addresses only. An address
  with no host (`:4242`) or a non-loopback one opens the server to the network;
  `serve` then prints a warning unless an API key is set.
- **API key.** `-api-key KEY` (or `JITLLM_API_KEY`, which only the command
  reads) makes every route but `/healthz`, `/health` and `/v1/health` require
  `Authorization: Bearer KEY`; `/v1/messages` also takes `x-api-key: KEY`, as
  the Anthropic SDKs send it. The key is compared in constant time. A refusal
  is a 401 in each API's own error shape: `{"error": {"code": "invalid_api_key",
  ...}}` on the OpenAI routes, `{"type": "error", "error": {"type":
  "authentication_error", ...}}` on `/v1/messages`, and
  `{"code": "unauthenticated", ...}` on the Connect routes.
- **Host header.** Bound to loopback, the server answers only requests whose
  `Host` is its own address, `localhost`, `127.0.0.1` or `[::1]`: a web page that
  points its own DNS name at `127.0.0.1` (DNS rebinding) gets a 403.
- **Origin.** A browser sends an `Origin` header; a request whose origin is
  not a loopback one is refused with a 403 unless `-allow-origin` lists it
  (`-allow-origin https://app.example.com`), which also answers its CORS
  preflight. Clients that send no `Origin` (the SDKs, curl, the jitllm
  programs) are unaffected.

The clients send the key when they have one: `jitllm` and `jitllmd` verbs take
`-api-key` (default `JITLLM_API_KEY`); the window and the terminal app read
`service_key` (the key to ask a jitllmd with) and `api_key` (the key their own
API requires) from the settings file, and `JITLLM_API_KEY` for either. Both
settings screens have a masked field for each (with a show toggle); a blank field
is no key, and the file is written private (mode 0600). The settings file records
an address only when you type one: empty means the default.

## The native API and the OpenAI routes

`InferenceService` carries everything the OpenAI routes do, so the CLI, the
window and the terminal app use the native API with every feature. The two are
one path: the OpenAI body and the Connect request build the same
`GenerateOptions` for `Engine.Generate`, and a request is read by the
functions the OpenAI route runs (`oaTools`, `oaToolChoice`,
`oaResponseFormat`, `oaChoiceSeeds`, the sampling check, the tool-call
reader), so a value one refuses the other refuses in the same words, as a 400
and as `InvalidArgument`. `server/parity_test.go` holds the mapping field by
field on a scripted backend, and `server/parity_model_test.go` sends
equivalent requests through both on stories260K and Qwen3-0.6B and requires
the same text, finish reason, usage, log-probabilities and tool calls.

| OpenAI | Connect (`InferenceService`) | Notes |
|---|---|---|
| `model`, `jitllm_session` | `GenerateRequest.model_id`, `session_id` | a model by id, file name or alias; loaded on demand |
| `messages[].role`, text parts | `ChatPrompt.messages[].role`, `content`; `system` | text parts of a turn are joined, on both |
| `image_url` parts (`data:` URL, `http(s)` URL) | `ChatMessage.images[]`: `data` (bytes) or `url` | the same `-image-*` limits and `-image-fetch-allow` policy; `data` and `url` together are refused |
| `tools` | `ChatPrompt.tools[]` (`name`, `description`, `parameters_json`, `strict`) | rendered by the model's own chat template |
| `tool_choice` `none`, `auto`, `required`, named | `ToolChoice.mode` `NONE`, `AUTO`, `REQUIRED`, `FUNCTION` + `name` | a named tool must be declared; `none` withholds the tools |
| `parallel_tool_calls: false` | `ToolChoice.single_call` | at most one call |
| assistant `tool_calls`, `tool` turns | `ChatMessage.tool_calls`, `tool_call_id`, `name` | |
| reply `tool_calls` | `GenerateToolCalls` events; `CompleteChoice.tool_calls` | each call is sent when its markup closes; the text carries none of it |
| `finish_reason` `stop`, `length`, `tool_calls` | `FinishReason` `STOP` or `EOS`, `MAX_TOKENS`, `TOOL_CALLS` | |
| `response_format` `text`, `json_object`, `json_schema` | `ResponseFormat.type`, `json_schema`, `name`, `strict` | `strict` is accepted and always so: the grammar always holds the reply; with `grammar`, `json_schema` or a forced tool call it is refused as one constraint each |
| `logprobs`, `top_logprobs` | `logprobs`, `top_logprobs` (0 to 20) | `GenerateToken.logprob`, `CompleteChoice.logprobs`; the raw distribution; refused with `echo_prompt` |
| `/v1/completions` `logprobs: N` | `logprobs: true`, `top_logprobs: N` | |
| `n` (1 to 128) | `n` | `GenerateResponse.choice`, `CompleteResponse.choices`; choice `i` samples with seed + `i` |
| `stop` | `stop` | |
| `seed` | `sampling.seed` | |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` | `sampling.temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty` | validated by one function; `top_k: -1` disables |
| `presence_penalty`, `frequency_penalty` | `sampling.presence_penalty`, `frequency_penalty` | non-zero is refused by name on both; `repeat_penalty` is what the engine has |
| `max_tokens`, `max_completion_tokens` | `max_tokens` | |
| `ignore_eos`, `echo` | `ignore_eos`, `echo_prompt` | |
| `jitllm_speculate`, `jitllm_priority` | `speculation`, `priority` | |
| `usage.prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens` | `CompleteResponse.usage`; a `GenerateUsage` event when `include_usage` is set | the prompt is counted once for `n` choices; per choice, `GenerateFinished.cached_tokens` |
| `/v1/models` `max_model_len`, `aliases`, `loaded` | `ModelInfo.default_max_seq`, `aliases`; `ListModels.loaded` and `files` | |
| (none) | `ModelInfo.capabilities`, `ModelFile.capabilities`, `context_length` | `completion`, `chat`, `tools`, `json_schema`, `logprobs`, `vision`, `embeddings`; an encoder has `embeddings` alone; a file's come from its header, without loading it |
| `/v1/embeddings` `input`, `dimensions` | `EmbedRequest.inputs`, `dimensions` | an empty input or a negative width is refused in the same words; the vectors are the same bytes |

Left to the surface they belong to: `encoding_format: "base64"` (a transport
choice; the Connect message already carries packed floats), a list of prompts
in one `/v1/completions` request (the Connect client sends one request per
prompt, and the step loop batches them), the Files and Batch APIs, and the
Anthropic content blocks. `GenerateRequest.add_generation_prompt` is the
native default of false where the chat route always sets it: a client sending
a chat sets it. Unlike the OpenAI route, a Connect request with `sampling`
unset runs the session's own sampler (greedy by default); the route's
temperature 1 default is the route's.

A named feature reaches a client through `ServerInfo.features`: `chat_tools`,
`response_format`, `logprobs`, `choices`, `image_url`, `stream_usage` and
`model_capabilities` join `server_info`, `model_aliases`, `model_enable` and
`on_demand_load`. `server/client.Conn.Has` asks for one.

## Models by name, loaded on demand

Every `.jlm` in the models folder (`-models`) is available by name: its file
name, with or without `.jlm`, or an alias. A request naming one that is not
loaded loads it, under its file name as its id, with `-devices` (auto by
default), `-gpu-layers` and `-sessions`; the folder is read at each lookup,
so a container converted into it is there for the next request. A model that
does not fit pages (that is the design); a load that fails answers with the
model, its size and the host budget it was opened against. `-on-demand=false`
turns this off. `GET /v1/models` lists the loaded models and then the
available ones (`"loaded": false`), each alias as an entry of its own.

Aliases and enable/disable live in the folder's `jitllm-models.json`, so they
survive a restart and move with the models:

```sh
jitllmd models -alias Qwen3-0.6B-Q8_0=qwen3,small   # several names, unique across the folder
jitllmd models -unalias Qwen3-0.6B-Q8_0=small
jitllmd models -disable qwen3   # listed, never loaded on demand; `models -load` still loads it
jitllmd models -enable qwen3
jitllmd models                  # loaded, then on disk with their aliases and state
```

The same is `ModelService.SetModelOptions` on the Connect API, and
`ListModels` carries each file's `aliases`, `disabled` and loaded `model_id`.
`jitllmd models -load` and `-unload` work as before.

## The service

`jitllmd service install` runs `jitllmd serve -addr localhost:4242 -models
<the models folder>` in the background, now and at every login, restarted
after a crash and not after a clean stop. Loopback only: nothing is opened to
the network unless `-addr` says so. `jitllmd service install -api-key KEY`
(default `JITLLM_API_KEY`) writes the key into the service's arguments, in a
file only the user can read.

A service installed by an earlier release listens on `127.0.0.1:8080` and keeps
doing so: only `jitllmd service install` rewrites the unit. Clients that find
nothing on `localhost:4242` look at `127.0.0.1:8080` too, and use a jitllmd
that answers there (an open port is not enough: it must return the capability
manifest, or `ServerInfo`). They say once that the service runs on the old
port; `jitllmd service install` moves it to `localhost:4242`. An address given
with `-addr` or in the settings is never swapped for the old one.

| System | What runs it | Log |
|---|---|---|
| Linux | systemd user unit `~/.config/systemd/user/jitllm.service` | `journalctl --user -u jitllm` |
| macOS | launchd agent `~/Library/LaunchAgents/org.jitllm.service.plist` (the app's "start the server at login" writes the same one) | `~/Library/Logs/jitllm/jitllmd.log` |
| Windows | `%LOCALAPPDATA%\jitllm\jitllmd-login.vbs` under the user Run key (the installer's "start the server at login" calls `jitllmd service install`) | `%LOCALAPPDATA%\jitllm\jitllmd.log` |

`install.sh` installs the service on Linux when it installs jitllmd
(`--no-service` or `JITLLM_NO_SERVICE=1` skips it). `jitllmd service status`
shows whether it is installed, the address, the version, the models folder
and the loaded models. On Linux, `loginctl enable-linger` keeps it running
without a login session.

The Connect API is defined in [`proto/jitllm/v1`](../proto/jitllm/v1/).
