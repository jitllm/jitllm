# Capabilities: what this build converts, runs and serves

`docs/capabilities.json` is the machine-readable answer to "does this build
support my workload". It lists:

| field | what | from |
|---|---|---|
| `container` | the `.jlm` version the engine reads (only that one) | `jlm.Version` |
| `input_formats` | what `jitllm convert` takes, each with an example and the gate behind it | `internal/capabilities/maintained.json` |
| `architectures` | every GGUF architecture name and the graph it converts to | `convert.Architectures` |
| `safetensors_classes` | every Hugging Face class and its graph | `convert.HFClasses` |
| `projectors` | every vision projector | `convert.Projectors` |
| `quantization` | the tensor types that decode, and those with packed generated kernels | `quant.Dequantable`, `quant.PackedTypes` |
| `backends` | each host and device backend: implemented, and verified on which hardware, with the gates that are the evidence | `maintained.json` |
| `api.endpoints` | every HTTP route, its methods, and each request field with what it does and its limits | the server's route table and request structs, by reflection |
| `api.refusals` | combinations the server refuses rather than ignores, with the message it returns | the server's refusal table, held to the handlers |
| `api.control` | the ConnectRPC services and methods | the generated descriptors |

`implemented` with an empty `verified` list is a claim about code, not about a
card: the HIP backend is built and its kernels compile for five AMD targets,
and it has not run on an AMD card (`.notes/design/amd-hip.md`).

## Reading it at run time

The file describes the source tree. The running binary answers for itself,
with its version and revision added:

    jitllm capabilities          # a summary
    jitllm capabilities -json    # the manifest
    curl localhost:4242/v1/capabilities   # jitllmd, the routes it is serving

## Keeping it true

Everything derivable is derived; regenerate after changing a list, a route or
a request field:

    cd server && go run ./cmd/gencapabilities

It writes `internal/capabilities/api.json` (the server's half, which the
engine module embeds because it cannot import the server) and
`docs/capabilities.json`. The gates:

- `srcgate.TestCapabilitiesDocIsGenerated`: the file is what the code renders.
- `server.TestAPIManifestIsGenerated`: `api.json` is what the route table
  and request structs give.
- `server.TestAPIManifestCoversEveryField`: every request field has a note in
  `fieldNotes`, and every note a field.
- `server.TestRefusalsAreRefused`: each refusal with a probe draws a 400
  carrying its message from the route; each engine-level one is a message
  the server's source returns.
- `capabilities.TestMaintainedEvidenceExists`: every gate `maintained.json`
  names as evidence is a test in the tree, and a backend verified nowhere says
  so in its note.

Changing what hardware a backend is verified on is an edit to
`maintained.json` naming the gate that ran there, then the generator.

## `jitllm doctor`

`jitllm doctor [-json] [model.jlm]` inspects; it never downloads, opens a
model's pages or runs a kernel. It reports the build's version and revision,
the host's CPU and code-generator tier, memory (total, and the budget the
engine sizes its page frames from), every device a backend opened, which of
CUDA, HIP, Vulkan and Metal is usable and why not, and, given a container,
its version against the one this build reads and whether this build knows its
architecture.

| exit | `status` | meaning |
|---|---|---|
| 0 | `ok` | everything a run needs is here |
| 2 | `missing` | inspected, and something is missing: `problems` names each with a `next` action |
| 1 | `error` | the inspection itself failed (bad arguments, an unreadable file); with `-json`, `error` carries it |

No GPU is a `note`, not a problem: a CPU machine is a supported configuration.
The problem codes are `container-missing`, `not-a-container`,
`container-version`, `container-unreadable`, `architecture-unknown`,
`no-code-generator` and `cpu-below-baseline`. Logs go to stderr; with
`-json`, stdout is only the report.
