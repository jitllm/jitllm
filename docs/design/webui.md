# The web UI: our own client of the ConnectRPC API

**Status: built for chat, models and devices.** `jitllmd -webui` serves a
TypeScript client generated from `proto/jitllm/v1`: devices, loaded models and
model files with Load and Unload, and a multi-turn chat through
`InferenceService.Generate` with Stop, sampling controls, stop strings and
pictures. The placement, residency and telemetry views, decision models and
the RPCs listed below as missing are not built. Nothing here is a rule; AGENTS.md governs.

jitllm does not embed llama.cpp's web UI behind a compatibility layer. It
builds its own, and the UI is a client of the same `jitllm.v1` services the
desktop app and the CLI use, not of `/v1/chat/completions`. The OpenAI and
Anthropic routes stay for third-party tools; the UI needs what they cannot
carry: placement, residency, per-device telemetry, decision models, blocks.

## Look

Matches `ui/` and the website: Tokyo Night (`website/public/themes.css`:
`--bg #1a1b26`, `--surface #1f2230`, `--line-strong #414868`, `--text #c0caf5`),
brand `#9ece6a`, sans for prose (Geist), mono only for data and code (ids,
sizes, token ids). The spike copies the token names so the website's other
themes can be taken later without a rewrite. Square corners, as the website.

## Stack

| choice | why |
|---|---|
| **TypeScript, no UI framework, for now** | The page is a few tables, a log and a stream. A framework is a dependency tree to vet and a build to keep; the spike is 140 lines of TS and a 40 KB gzipped bundle, most of it the protobuf and Connect runtimes. Reconsider (Preact, 4 KB) when the placement view needs real state; a framework is a decision, not a default. |
| **`@bufbuild/protoc-gen-es` from the same `buf.gen.yaml`** | A third plugin entry (`out: server/webui/src/gen`, `target=ts`) beside protoc-gen-go and connect-go. One `buf generate` writes every language, as the repo's rule says. In protobuf-es v2 the service descriptors are in the `_pb.ts` files themselves, so `@connectrpc/connect` takes them directly and there is no separate connect-es plugin. |
| **`@connectrpc/connect-web`** | The Connect transport over `fetch`. No hand-written wire code anywhere. |
| **esbuild, one script (`build.mjs`)** | One binary, no config, bundles and minifies in tens of milliseconds. No Vite, no webpack. |
| **Versions pinned exactly** | `package.json` pins; `package-lock.json` is committed and CI uses `npm ci`. |

The generated TS is checked in like `server/gen`, so the Go build and the
`generated code is current` CI step need no Node; the TS build is the only job
that does.

## Assets: built, embedded, opt-in

- Source in `server/webui/` (inside the `server` module, because `go:embed`
  cannot reach a parent directory). `npm run build` writes `server/webui/dist/`.
- `server/webui.go` embeds `dist` (`//go:embed all:webui/dist`) and
  `WithWebUI(next)` serves it from memory at `/ui/`, passing everything else
  through. `jitllmd -webui` turns it on; the default is off, so a headless
  server exposes no extra surface and the binary grows by the bundle only
  (~180 KB uncompressed at this size).
- The UI is served by the same listener as the API, behind the same `Guard`:
  same-origin by default and the Host check covers it. The `/ui/` files are the
  one exception to the API key (`openRoute`): the page holds no data and has to
  load before it can ask for the key, and every RPC it makes carries one.
- `dist/` is committed while the bundle is small, so `go build ./server/...`
  works from a clean checkout with no Node. When it grows, or churns diffs,
  switch to building it in CI and in the release job before `go build`, with
  a placeholder `dist/index.html` committed so the embed pattern still matches.

## CI

The `check` job: `npm ci`, `npm run typecheck`, `npm test` (node's runner over
`src/chat.test.ts`), `npm run build`, then the same
diff check as `server/gen` (`git diff --exit-code server/webui/dist
server/webui/src/gen`) so a stale bundle or stale generated client fails the
build. The existing buf step gains `npm ci` first, because the plugin is a
devDependency found by path. The release job runs the same build before
cross-compiling `jitllmd`. CI stays model-free; a headless-browser smoke test
against `jitllmd` with the committed stories model is the one UI gate worth
adding (it is how the spike's screenshot was taken).

## Streaming from a browser

`Generate`, `WatchStats`, `Convert` and `TuneSeam` are server-streaming, which
`connect-web` does over `fetch` on HTTP/1.1 or HTTP/2, one response stream per
call, cancelled by `AbortController` (the spike's Stop button; the server sees
the context cancel). The server already mounts the services on a mux wrapped
in h2c, and Connect works over plain HTTP/1.1, so no gRPC-Web proxy is needed.
Browsers cannot do client or bidirectional streaming over fetch; no jitllm RPC
needs it. Tokens arrive as `GenerateToken` events, so the UI renders
incrementally with `textContent` (never `innerHTML`: model output is
untrusted).

## Auth and CORS when not same-origin

`server/guard.go` already does the work: loopback binds answer only loopback
`Host`s, cross-origin requests are refused unless the origin is loopback or
listed in `-allow-origin`, listed origins get CORS headers and preflights are
answered, and `-api-key` is `Authorization: Bearer`. The spike's
`?server=URL&key=KEY` uses exactly that. Open points:

- `?key=` is read once, kept in `sessionStorage` (never `localStorage`) and
  removed from the address bar; an `Unauthenticated` answer shows a key box.
  A key typed there never enters a URL.
- `Access-Control-Expose-Headers` is not set. Connect's unary and streaming
  error details ride in the body, so nothing breaks today; the Connect
  protocol's trailer headers (`Connect-` prefixed) would need exposing for a
  strict client, so add them when a UI feature reads one.
- A UI hosted elsewhere (the website) talking to a user's local `jitllmd` is
  a decision for the user: it needs `-allow-origin` and, on public pages,
  Private Network Access preflight headers.

## What the UI shows, and the RPC behind each part

| view | RPCs | state |
|---|---|---|
| Chat with streaming, stop, sampling controls, stop strings | `Generate` (`ChatPrompt`, `SamplingParams`), `Cancel` | built, except that the history is in memory and lost on reload |
| Pictures in a turn | `ChatMessage.images` (`ChatImage`) | built: a file picker; a model with no vision tower answers with the server's refusal, shown in the turn |
| Tool calls, structured output | `GenerateToolCalls`, `grammar`, JSON schema | in the protos |
| Models: files on disk, loaded, load and unload with options | `ListModels`, `LoadModel`, `UnloadModel`, `SetModelOptions`, `Convert` (stream) | in the protos; load progress is not streamed |
| Devices and memory | `ListDevices`, `GetMemoryTopology` | in the protos; spike lists devices |
| Placement: which block runs where, forcing it, relocation | `GetPlacement`, `SetPlacement`, `RelocateBlocks`, `SetRelocation`, `TuneSeam` (stream) | in the protos |
| Paging and residency: pages in, evicted, budget | `GetResidency`, `SetPageBudget` | in the protos; unary only, polled |
| Telemetry: tokens/s, queue, per-device memory | `WatchStats` (stream) | in the protos |
| Sessions: KV, prefix cache, reset | `CreateSession`, `ListSessions`, `ResetSession`, `GetDeviceQueue` | in the protos |
| Decision models (Laya, Clef): questions and options, scored answers, confidence | none | **missing**: only the HTTP route `compat_systemone.go` serves it |
| Server capabilities (what this build converts, runs, serves) | none | **missing**: `/v1/capabilities` is an HTTP route; `jitllm doctor` reads it |

## RPCs the protos lack

1. **`InferenceService.Decide`** (or a `DecisionService`): the decision-model
   request and answer as messages, so the UI is not forced onto the compat
   route for the one feature that is jitllm's alone.
2. **`ModelService.WatchLoad`** (or `LoadModel` as a stream): load and page-in
   progress, so a 30 GB load is a progress bar and not a hung request.
3. **`PlacementService.WatchResidency`**: a pushed residency and placement
   stream. Polling `GetResidency` works for a spike and is wrong for a view
   that animates pages faulting in; `WatchStats` is the model for it.
4. **`TelemetryService.GetCapabilities`**: the manifest as a message.
5. **A message history store is not an RPC.** Chats live in the browser
   (IndexedDB) for now; a server-side store is a product decision, not a gap.

## Size and startup

Spike: `app.js` 138 KB minified, 41 KB gzipped, almost all `@bufbuild/protobuf`
and `connect-web`, plus 2 KB of CSS and a 0.7 KB page. Generated descriptors
for all seven services are pulled in only for the services the page imports,
so tree-shaking should keep growth modest (unmeasured); the bar is a
bundle under 300 KB gzipped and first paint of the shell before the first RPC
returns. No web fonts are fetched: Geist and JetBrains Mono are named in the
font stack with system fallbacks, so the UI works offline on an isolated box.
Startup cost inside `jitllmd` is zero when `-webui` is off, and a map of the
embedded files when on.

## Open questions

- Whether the UI is also the website's live demo (a hosted page pointed at
  a local server) or only ever served by the server it controls.
- Whether to take the website's seven themes or Tokyo Night only.
- When a framework earns its place; the trigger is the placement and
  residency views, not chat.
