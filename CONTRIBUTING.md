# Contributing

[AGENTS.md](AGENTS.md) is the rules file, for people and coding agents alike:
the five principles, how a performance claim is measured, what a gate must
prove. Everything under [docs/](docs/) is evidence behind those rules; where a
sentence there reads like an instruction, AGENTS.md governs. Read the evidence
file for the subsystem you are about to touch first (AGENTS.md's routing table
names it): many ideas there were already tried, with the measurement.

## Build

Go 1.26 or newer, no C toolchain: there is no cgo anywhere, and CUDA, Vulkan
and Metal are opened at run time. The repository holds five modules: the
engine at the root, the daemon in `server/`, the desktop app in `ui/`, the
terminal app in `tui/`, and `common/`, the logic the two front ends share:

    go build ./cmd/jitllm
    (cd server && go build ./cmd/jitllmd)
    (cd ui && CGO_ENABLED=0 go build .)
    (cd tui && go build .)

## Heavy jobs run under `scripts/cap`

Every build of the whole tree, test run and benchmark goes through
`scripts/cap`, which runs it in a memory-capped cgroup:

    ./scripts/cap 8G -- go test ./tok/... -count=1

Without it a test that converts or opens a large model can push the machine
into memory pressure, and the out-of-memory daemon then kills whatever is
under the most pressure, which on a desktop is the session rather than the
test. Use `24G` for a model over ~16 GB and for `-race`. A measurement is also
pinned to the performance cores (AGENTS.md RULE 3 and RULE 5).

## Tests without models

`.github/scripts/test-model-free.sh` (the program is `scripts/modelfree`) is
what CI runs on Linux, macOS and Windows: vet aside, every package of the
five modules but `./dev/bench`, `./ui/stage` (CI's ui job runs it) and, on
macOS, `./jit_old/gpu/backend`, with no GPU and no model but the committed
stories260K, and a summary that lists every skip and the reason it printed.
It sets `JITLLM_MODEL_FREE=1`, under which a test that reports a missing model
through `testmodels.Missing` skips; without it that test fails
(docs/testing.md, "A missing model is loud").

    ./scripts/cap 8G -- .github/scripts/test-model-free.sh
    ./scripts/vet && ./scripts/vet server && ./scripts/vet ui
    (cd ui && ../scripts/cap 8G -- go test ./mock ./stage ./cmd/shots -count=1)

`GOOS=windows ./scripts/vet` checks the Windows build, test files included,
from a Linux box.

A skip is not a pass. Read the summary's skip list as what the run did not
prove.

## Tests with models

Model files are too big to commit. `internal/testmodels/fetch.sh` downloads
the ones the tests read, into `$JITLLM_MODELS` (or `models/` at the
repository root):

    internal/testmodels/fetch.sh --list      # what the tests read, and what is here
    internal/testmodels/fetch.sh             # the default set
    JITLLM_MODELS=/path/to/models ./scripts/cap 16G -- go test ./engine/model -run TestGreedyMatchesLlamaCpp -count=1

[docs/testing.md](docs/testing.md) has every variable that selects a test's
model and the gates that need a GPU. `engine/model` as a whole package is too
much for one run; slice it with `-run`.

## Commits

One change per commit. The subject is a full sentence stating what is now
true, not what was done ("A batched row honours ignore_eos", not "fix batch
EOS"); the body says why, and for a performance change carries the
measurement, its baseline and its backend (AGENTS.md RULE 1 and RULE 2).
Every change runs the gates of what it touches, and a new gate is run against
a deliberate violation once before it is trusted (RULE 10). Repository prose
carries no dates; a negative result reads "tried X, this happened".

## A new architecture

The list of architectures is a decision, not an accident. A new entry meets
AGENTS.md RULE 7's bar before it counts: a converter entry, a fixture from the
family's own reference class with every parameter and buffer randomised and a
golden from that class, a gate that fails when each of its graph features is
removed, every tier (the host on amd64 and arm64, CUDA, Vulkan and Metal with
every block placed), the five principles' gates with the model added to each
list, and a real checkpoint against llama.cpp wherever one fits.
[docs/design/model-coverage.md](docs/design/model-coverage.md) is the survey an
addition is chosen from.

## Reporting results

Try a model and [share what you saw](https://github.com/jitllm/jitllm/issues)
with `jitllm hardware`, the model and quantization, and the command. A rate is
worth more beside the same machine's llama.cpp rate:
[`scripts/vs-llamacpp.sh`](scripts/vs-llamacpp.sh) measures both.
