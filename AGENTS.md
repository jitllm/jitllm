# jitllm — project rules

**This file is the only place a binding requirement lives.** Everything under
`docs/` is EVIDENCE: what was measured, what was tried, what was wrong. If a
sentence there reads like an instruction, this file governs.

That separation exists because a review of the single 415 KB file this
was split out of named its real failure exactly: *"historical
findings, current instructions and withdrawn conclusions all use the same
imperative voice. Several corrections are good; their surrounding text still
permits the opposite decision."* Five rules were being quoted as current while the
code had moved underneath them.

**When you correct a rule, EDIT IT.** Do not append another emphatic paragraph
beneath the old one — that is how this file reached 415 KB and how a withdrawn
conclusion kept the appearance of a binding rule. Move the narrative to the
matching evidence file and leave a pointer.

## Routing — read the evidence file before working on its subsystem

| working on | read first |
|---|---|
| the scoreboard, any ratio against llama.cpp | `docs/perf/current.md` |
| amd64/arm64 kernels, packing, attention on the host | `docs/engineering-history/cpu-kernels.md` |
| PTX/SPIR-V/MSL, the IR, device prefill, tensor cores, Metal decode | `docs/engineering-history/gpu-kernels.md` |
| a new architecture, tokenizer, MoE, the vision tower, the chat template | `docs/engineering-history/model-correctness.md` |
| the pool, dispatch, the JIT's structure, benchmark harnesses, the measurement method | `docs/engineering-history/scheduling-and-measurement.md` |
| CPU/GPU seam, VRAM budgets, what goes on the card, paging, the prefix cache | `docs/engineering-history/placement.md` |
| what is open, and the levers shelved for being small | `docs/open-questions.md` |
| the designs: block paging, device sessions, KV paging, vision as blocks, the coverage survey; instances and sessions | `docs/design/` |
| which models a test reads, and every model-selecting test variable | `docs/testing.md` |
| every architecture, class and projector that converts, and the models each covers | `docs/models.md` (generated from the converter's lists) |
| what this build converts, runs and serves, by machine; `jitllm doctor` | `docs/capabilities.md` (`docs/capabilities.json` is generated) |
| which checks a change needs, and running them | `docs/checks.md` (`go run ./scripts/checks <paths or range>`) |
| an architecture's or projector's gates and evidence; a recorded llama.cpp divergence | `docs/architectures.md`; `docs/divergences.md` |

Those files carry the reasoning and the failed attempts. Skipping them is how an
idea gets proposed for the fourth time; several have a section that exists
solely to say "this was tried, and here is the measurement". Text that stood in
this file before it was cut back is in them under headings that say so ("as it
stood in AGENTS.md").

---

## The five principles. Every new architecture, kernel and path keeps all five.

Stated by the user as what jitllm is. A feature that breaks one is not done,
however good its numbers; each has a gate, and a new architecture runs every
gate below before it counts.

| principle | what it means | the gate |
|---|---|---|
| **JIT** | every kernel is emitted at run time for the device in front of it; no pregenerated or copied kernels (RULE 8) | `cpu.TestNoAssemblyBesidesTheTrampolines`; device code only from the composer's forms (no hand-written kernels) |
| **No Go compute** | no arithmetic a token runs is a Go loop; a missing kernel is a BASIC generated kernel, never a fallback (RULE 8) | `TestNoProductionCodeImportsTheOracle`, `engine.TestEveryModelRunsGenerated` |
| **Relocation** | any block, its KV pages and its recurrent state can move between any two devices, the CPU among them, mid-sequence, with the same answer; no device is a default where blocks live, and every move names its destination | `TestLayerRelocatesToAnotherDevice`, `TestBlocksMoveBetweenDevices`, `TestRelocatedBlocksComeBack`, `TestKVPagesRelocateWithTheLayer`, `TestBatchSeamMovesCarryEveryRow(Hybrid)`, `TestMLASeamMovesWithItsHistory`, `TestShrinkingTheSeamMovesTheSummaryToTheCPU`; decision models: `TestDecisionRelocatesMidPrompt`, `TestDecisionMovesBetweenCards`, `TestDecisionDistributes` |
| **Paging** | paging is the design, never a mode: a model that does not fit pages, a budget below the model evicts and re-reads with the same answer, every entry point faults its blocks in, and a windowed layer holds its window and not the context (placement.md 15z) | `TestPagingDoesNotChangeTheAnswer`, `TestOpenHonoursAPageBudget`, `TestForwardBatchFaultsItsBlocksIn`, `TestTwoPagedModels*`, `TestWindowedLayersHoldTheirWindow`, `TestPrefixCacheRestoresAWindowedLayer`, `TestKVIsHeldOnce` (a position's history in one place per device: no page a session holds beside the prefix cache's on its device, no host mirror of a card's history), `TestSharedPrefixMovesToTheCard`, `TestBlocksPageOnTheListedCard` (cards listed without the CPU, `WithCPU(false)`, page the model on themselves and nothing runs on the CPU); decision models: `TestDecisionPagesWithTheSameAnswer` |
| **Low to no Go allocation** | a warm decode token and a warm step make zero engine heap allocations on the host and every device backend; weights and frames live off the heap (RULE 2f) | `engine.TestDecodeDoesNotAllocate` (JITLLM_ALLOC_MODELS adds models), `TestTwoPagedModels*`; a decision model's warm question: `TestDecisionDoesNotAllocate` |

A new architecture is added to each gate's model list, not only to the
correctness gates -- a gate that never runs a model does not cover it
(RULE 11d).

## RULE 1: every perf claim is a ratio against a baseline measured in the same pass.

- **Never carry a baseline across sessions.** Re-measure the arm you are
  comparing against, in the same pass, with the same tool. A 2.7-point session
  bias once survived an ABBA interleave, a PSI gate, a clamp gate and an IQR of
  0.013, and it was the difference between "2.5% from a win" and "at parity".
- **Quote the backend with the number, always.** A jitllm CUDA
  figure against a llama.cpp Vulkan figure is not a comparison; that mistake
  once produced a commit message claiming a win the CUDA number did not support.
- **Say the generation length.** Short samples flatter jitllm, because
  `llama-bench` discards a warm-up pass while `jitllm run` measures every token
  from process start.
- **Report tok/s and bytes/token and effective GB/s as a fraction of a wall
  measured in the same process.** "85 tok/s = 45.3 GB/s = 81% of the 55.7 GB/s
  wall" is checkable with a calculator; "85 tok/s" is not. Above the roofline is
  a harness bug, not a win.
- **Take Linux rows on a Xeon, not on the laptop.** The laptop is the developer
  machine: it reaches 100 C under the six-core decode a board is made of, and a
  docs-only commit read 0.8866 on it against its byte-identical predecessor's
  0.9853. The Xeon holds its clock (the D-2123IT gated at IQR/median 0.0057 on
  an idle box); the E5-2680 v4 rows are taken on one socket's twelve cores
  under `numactl --membind=0` (RULE 5).
- **When a row indicts a commit, run that commit before theorising about the
  room it was measured in.** One `git worktree add`, one build, two passes. A
  "regression" was hunted through four experiments before the binary it named
  was built and read 1.035.
- **A docs-only commit is a bisect's null control.** Validate every bisect step
  against `git show --stat`, and probe a comment-only commit first: if it reads
  differently from its neighbour, the harness cannot resolve the effect and every
  other row in the run is noise.
- **When the stopwatch cannot resolve, count.** At `JITLLM_CORES=1` there are no
  workers and no spin, so the instruction count is the work and the rate is the
  engine. Token ids, regions per token, bytes read per token and container size
  are deterministic. `perf stat` instructions above one thread are NOT a work
  measure: the pool spins, so a spinning worker's count tracks how long the
  region took, which on a clamped box is temperature.
- **A row runs outside any memory cgroup** (RULE 3). The one exception is an
  over-committed row, where the ceiling IS the experiment: **it gives both arms
  the same ceiling by shrinking the cgroup, stated with the row, never by
  `-maxmem` on one arm.** llama.cpp mmaps and reaches the whole
  cgroup through the page cache, while jitllm self-limits to `sched.MemBudget()`;
  a row where one arm holds 14 GiB of weights and the other 26 is not an engine
  comparison.
- **This file carries no scoreboard.** The current board is
  `docs/perf/current.md`.

Evidence: `docs/perf/current.md` ("Boards that stood in AGENTS.md RULE 1");
`docs/engineering-history/scheduling-and-measurement.md` ("The 0.90 and 0.70
readings, and the harness that produced them").

## RULE 2: a perf claim is paired, interleaved, and gated.

A/B/A/B **within one process**, n >= 20, never A-then-B. Report the **median of
per-round ratios** with its IQR; **reject if IQR/median > 0.10**.

- **Soak to thermal steady state first.** Cooling between samples is the
  intuitive fix and the wrong one: it puts every sample at a different point on
  the frequency ramp. On the laptop a row needs nothing else running,
  `JITLLM_SOAK=10` and 16 rounds.
- **Run the whole comparison twice.** A dispersion gate cannot see a bias common
  to both arms, and that is the failure that moves a row across parity.
- **Run the A/A self-control FIRST, on the harness you are about to quote.** One
  M4 A/B had a 3.5% slot bias only a same-binary-both-arms run exposed, and a
  self-control does not transfer between harnesses: on one Linux box the
  in-process probe controls at 0.9948 and the process-per-sample A/B beside it at
  1.09. Reproducing a number proves the harness CONSISTENT; only the
  self-control proves it DISCRIMINATES.
- **Gate on the cores you are about to pin to, not only on the machine.** PSI
  (`/proc/pressure/cpu` some avg10 < 25%) is a gate and the load average is not
  one (it reads 4.42 on an idle box here) -- but PSI measures stall, and passed a
  box at 69% foreign load with a 913% tenant on the first pinned core.
  `scripts/corebusy.py` samples `/proc/stat` for exactly the taskset CPUs, and
  `vs-llamacpp.sh` refuses above 20% (`JITLLM_VS_BUSY`). Contention is a cliff:
  one contended core of six costs as much as six, because the barrier makes the
  slowest worker the whole region.
- **Temperature is reported, not a refusal.** A warm package is the state to
  measure in; it is printed beside the busy figure so a clamped row can be read
  against it.
- **Discard power-clamped samples, never average them.** When frequencies are
  low and two passes disagree, stop measuring and let the box rest -- do not
  raise the round count.
- **The coherence gate stays on** (`JITLLM_COHERE`, 1.05, on clocks sampled from
  the taskset cores only): a too-aggressive gate refuses rows, a too-permissive
  one publishes them. Whether 1.05 is too tight for a mixture is open. It is a CPU-row gate: a GPU row with every block on the cards leaves
  the host idle, its clocks are not the row's, and `vs-llamacpp.sh` turns the
  clock gates off for an unpinned GPU row.
- **A clock-trend correlation is a suspicion, not a verdict.** A real trend
  tightens the residual when removed; `scripts/clockadj.py` prints that
  comparison, refuses below |r| = 0.5 and refuses to extrapolate.
- **`scripts/vs-llamacpp.sh` is the only source of a jitllm-vs-llama.cpp row**,
  on both hosts. A scoreboard whose rows come from two tools is not a scoreboard.
- **One run measures a cold engine.** jitllm tunes itself while it runs; take the
  last of several, or warm the cache and say which.
- **Check the baseline arm's absolute rate against the figure recorded for that
  host before reading the ratio.** A mis-configured baseline moves the ratio as
  far as a real win: llama.cpp told `-t 6` and confined to four cores gated at
  IQR/median 0.004 and read 1.2413 where the truth was 0.9637. No dispersion
  gate, A/A control or second pass can see that. The harness takes llama.cpp's
  thread count from `$CORES`.

Evidence: `docs/engineering-history/scheduling-and-measurement.md` ("RULE 2 as
it stood in AGENTS.md", "The 0.90 and 0.70 readings").

## RULE 2f: the Go collector is part of the engine's budget.

Page frames and the dense region are anonymous mappings outside the Go heap
(`offheap.MapOffHeap`), outside the collector's goal, its memory-limit
arithmetic and its marking. Page frames are permanently live, so on the heap a
budget close to the memory limit leaves the collector a goal it can never reach:
on Qwen3-Next-80B it was 42.5% of the CPU, and the cliff is sharp (8.34 GiB of
budget 6 GC cycles, 8.99 GiB 648).

- The Go memory limit is only a backstop above the frames, set when it leaves
  the real heap a gigabyte; a limit sitting just above the heap spirals the same
  way.
- `sched.GCBudgetCap`'s quarter-of-the-heap reserve applies only with the
  weights ON the heap -- `JITLLM_OFFHEAP=0`, the control arm, or a platform with
  no anonymous mmap. `GOGC` is not the fix: under a memory limit the goal is
  `min(live*(1+p/100), limit-nonheap)`, and no p moves it once live is at the
  limit.
- `weak.Pointer` is not a page cache: it has no LRU and no pressure order (64
  weakly held buffers, one `runtime.GC()`, 0 alive).
- The cost is the collector's safety net: a span used after its model's Close
  faults.
- A gate that measures RSS is measuring the allocator. A leak gate reads the live
  heap and the tier's device ledger.

Evidence: `docs/engineering-history/scheduling-and-measurement.md` ("THE GO
COLLECTOR WAS 42% OF THE ENGINE'S CPU", "RULE 2f as it stood in AGENTS.md").

## RULE 3: on the developer's machine, every heavy job runs under `scripts/cap`. Agents included.

    ./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go test ./... -count=1

`scripts/cap` protects the machine the developer is working on from OOM:
systemd-oomd has killed the desktop session three times here, and MemoryHigh
throttling keeps the pressure that triggers it inside the job's cgroup.
`MemoryMax=8G MemoryHigh=7G MemorySwapMax=0`. Use `24G` for a model over ~16 GB
(cgroup v2 charges a mmap'd file's page cache to the cgroup) and for `-race`:
the detector's shadow memory pushes a suite that converts a GGUF past
`MemoryHigh`, and the kernel throttles the cgroup into reclaim without failing or
reporting anything. A slow `-race` run is a cap question before it is a code
question.

**It is not a measurement tool.** A benchmark host (a server, a rented runner)
runs both engines bare, with only the measurement lock; a comparison row never
runs inside a memory cgroup, which charges llama.cpp's page cache and throttles
it while jitllm's O_DIRECT reads are not charged that way.
`scripts/vs-llamacpp.sh` takes the lock itself and runs no cgroup; on the
laptop the hook still wraps it in `scripts/cap`, and the script reuses that
lock. A container with no writable cgroup runs uncapped and says so, never
under `ulimit -v`, which aborts the CUDA runtime.

There is no bypass variable in `scripts/cap`, on purpose. `JITLLM_NO_CAP=1`
belongs to the HOOK, where it lets a command run without this script and is
visible in the command line; a bypass that makes a capped-looking command
uncapped is a trap.

1. **Every prompt dispatching an agent that may build or measure on the
   developer's machine must contain the cap command verbatim.** An agent that
   has not been told will not do it.
2. **Never fan out parallel measurement agents.** Measure serially, in the main
   loop.
3. The lock (`$TMPDIR/jitllm-cap.lock`, `flock`): measurements EXCLUSIVE,
   builds SHARED, taken by `scripts/cap` and by the comparison scripts. It
   protects this repo only — another project's `go test` is invisible to every
   gate here and will land mid-round.
4. **Kill background jobs when the work that spawned them finishes.** Work
   started before a context compaction keeps running afterwards and neither end
   knows.

Evidence: `docs/engineering-history/scheduling-and-measurement.md` ("RULE 3 as
it stood in AGENTS.md", "The cap is the developer machine's, not a
measurement's").

## RULE 4: no perf floors inside correctness tests.

A throughput assertion beside an NMSE assertion makes the whole test flaky on a
thermally noisy box, and a flaky test gets skipped — taking the NMSE and
disassembly checks with it. Perf floors live in `dev/bench/`.

## RULE 5: six physical P-cores: `taskset -c 0,2,4,6,8,10`.

More cores is worse for decode. E-cores add no read bandwidth and become the
straggler behind every barrier; hyperthreads are neutral to negative. **Never
all 14**: leaving the OS nothing to run on costs more than the cores buy (the
coreset A/B: pe 0.864x, smt 0.965x, all 0.807x against p).

- **Prefill's core set is per host.** On the container path the laptop's
  E-cores are neutral in prefill; the M4's are worth +14-54% only through
  prefill's own P+E pool; the Xeon E5-2680 v4 has no E-cores and its rows are
  taken on one socket's twelve cores under `numactl --membind=0`. The old
  "prefill wants 6P+6E (+34%)" was measured on the pre-container engine and does
  not hold.
- **The process mask is the pin; workers are not bound to a CPU.** Binding each
  worker with `sched_setaffinity` cost 3.8% of decode and leaked single-CPU masks
  into the Go runtime's thread pool, silently confining whatever ran on them
  next. `cmd/jitllm` aligns `GOMAXPROCS` to the engine's width,
  `PCores`/`ECores`/`SMTSiblings` intersect `sched_getaffinity` so a narrower
  `taskset` yields fewer workers, and `LockOSThread` survives only where an API
  binds state to a thread (CUDA contexts, `backend/cuda`).
- **The shipped per-region dispatch cost** is `engine.TestFusionCeiling`'s
  2.70 us on Linux and 3.008 us on the M4.
- **Measure at the thread count that SHIPS.** A win measured on one core has
  evaporated or reversed at the real thread count eight times here — the single
  most repeated mistake in the evidence files.

Evidence: `docs/engineering-history/cpu-kernels.md` ("RULE 5 as it stood in
AGENTS.md", "On the M4, prefill wants the E-cores and decode does not",
"Prefill and the E-cores on the container path").

## RULE 6: nothing is refused for being SMALL. Only a clear REGRESSION refuses a change.

There is no size threshold a lever must clear to be built. Levers that each remove
a fraction **of the token** are additive in time saved, so stacking 4% levers is
how the engine passes llama.cpp — and it already did that once, 16.4 -> 27.4
tok/s in six steps, several of them in the 3-5% band.

- **A clear regression is a refusal.** That is the only one.
- **A number decides ORDER, never admission.** 8% goes before 0.4%. "Not now" is
  a schedule; "no" needs a regression behind it.
- **Do not put a perf claim on a sub-noise change.** Build it if it is cheap and
  correct; do not quote it.
- A change too small to measure is also too small to be shown NOT to regress, so
  the correctness gate does the work the perf gate no longer does. **The bar came
  down; the gate did not.**

**`cpu.Spec` has five fields** and is exactly where a tensor compiler would start
growing, so a sixth requires a paired-interleaved measurement and a statement of
what it buys — admitted unless it regresses.

### The regression checklist — "neutral" is a claim about every axis

> Before calling a change neutral, name the workload and check:
>
> - **Time** — whole-token latency/throughput, prefill, the relevant depth range,
>   the shipping thread count, cold preparation.
> - **Memory** — resident/peak bytes, allocation count and bytes, scratch,
>   packing expansion, KV capacity, device placement.
> - **Traffic/locality** — bytes actually MOVED, re-reads, cache-line ownership,
>   prefetch streams, coalescing. Not only unique input bytes.
> - **Execution resources** — registers/spills/occupancy, code size and competing
>   L1i footprint, dependency chains, execution-pipe pressure.
> - **Coordination** — pool regions, barriers, launches, synchronisation,
>   transfers, graph captures.
> - **Correctness** — the changed path actually EXECUTED; equality/NMSE/margin as
>   appropriate; deliberate violations fail.
>
> Report each relevant axis as **improved, regressed, unchanged by construction,
> or unresolved**. An unresolved timing result is not evidence of neutrality.

Source inspection is a valid check — "unchanged by construction" is a real answer
and needs no benchmark. And **"there is always a regression somewhere" is not a
licence to refuse every resource increase**: packing spends bytes to save
operations, and that is a trade to state, not a defect.

Each axis has a case in the evidence files where the cost landed where the
headline could not see it: registers (25% fewer instructions, 25% slower), a
31.7 KB kernel against a 32 KB shared L1i, +75% resident bytes, 48 allocations
per token, 906 dispatch regions, prefetch streams, a VRAM tombstone, and pipe
balance (fewer instructions, 13.5% slower).

## RULE 7: the architecture list is a LIST, and adding one is a decision.

`convert.archOf` (`convert/config.go`) is the list of GGUF architectures,
`hfArchTable` (`convert/safetensors.go`) the list of safetensors classes, and
`convert.projectorOf` (`convert/config.go`, a case each in `visionOf`) the list
of projectors. An architecture not in them does not convert (`configOf`:
"architecture %q is not implemented"); an unknown projector is refused at
conversion with `ErrNotImplemented`. Adding one is the user's decision.

`docs/models.md` is generated from those lists, never written by hand: an entry
adds its models' line to `internal/modelsdoc/describe.go` (the generator refuses
a name without one) and reruns `go run ./scripts/genmodels docs/models.md`;
`srcgate.TestModelsDocIsGenerated` fails until it does. The same lists feed
`docs/capabilities.json` (`cd server && go run ./cmd/gencapabilities`;
`srcgate.TestCapabilitiesDocIsGenerated`).

**Scope is jitllm's only defence.** llama.cpp has 176 `LLM_ARCH_` cases and 50
graph builders; Ollama had a funded team, covered ~21 architectures, and deleted
430,004 lines citing exactly this. `docs/design/model-coverage.md` is the survey
an addition is chosen from and priced against.

### The bar a new entry meets before it counts

1. **A converter entry** that reads the file and states in the container what
   the reference hardcodes against the name (a container field, or an arch code
   of its own so an older reader refuses the container rather than running it
   wrong). An absent key takes the CLASS's default (`hfClassDefaults`), never the
   tree's; a key spelled two ways that disagree is refused, because there is no
   way to know which the weights were trained under.
2. **A fixture from the family's own transformers class, with every parameter
   AND every buffer randomised**, written to GGUF by llama.cpp's own
   `convert_hf_to_gguf.py` where llama.cpp has the family, and a golden from
   that class (`scripts/*gold.py`). A buffer left at its initial zeros is often
   the identity -- DeepSeek-V3's selection bias was, and every gate that claimed
   to cover it compared two arms computing the same function. A new safetensors
   class needs a fixture and a golden in `engine.TestSafetensorsMatchTransformers`
   or it is a claim with nothing behind it.
3. **A features-are-load-bearing gate**: each graph difference, removed, fails
   it. A position-0 comparison cannot see a rotary bug (a rotation by 0 is the
   identity), and a greedy gate cannot see a positive logit divisor -- compare
   values, past position 0.
4. **Every tier**: the CPU device on amd64 and arm64, and CUDA, Vulkan and
   Metal with every block placed against the CPU (RULE 11c), plus a partial
   seam that leaves blocks on the CPU -- a gate asserting every block placed
   on a card runs no CPU attention and cannot see a CPU-side bug.
5. **The five principles' gates**, with the model added to each gate's list.
   The census's list is checked: `engine.TestTheCensusCoversEveryGraph` fails
   until a model it runs reads the new graph or carries the new projector.
6. **A real checkpoint against llama.cpp**, teacher-forced
   (`TestGreedyMatchesLlamaCpp`; `JITLLM_GREEDY_MODEL` names one over its size
   limit), wherever one fits a box.

Further rules for every entry:

- **A router's gating function is read per CLASS, never from an absent key**
  (`hardcodesSigmoid`, `hardcodesSoftmax`); a file whose `scoring_func`
  contradicts its class is refused (`hfGateAgrees`). The same holds for
  renormalisation and every literal a reference builder hardcodes.
- **Names matching is not the graph matching.** A class that maps onto an
  existing entry tensor for tensor still needs its own fixture and golden.
- **The expensive part of a new architecture is never the arithmetic** -- it is
  the tokenizer, and MoE. An unknown `tokenizer.ggml.pre` is REFUSED at load
  (`tok/spm.go`, "pre-tokenizer %q is not implemented"); conversion stores it as
  an empty pipeline so the reader can tell "no stages" from "stages nobody
  verified".
- **Read the artefact, not the description of it.** Transcribe a graph from
  llama.cpp's running output (`llama-eval-callback`) and read the TENSOR table,
  not the key summary. Find a divergence by bisecting the graph against the
  reference layer by layer and tensor by tensor.
- **Quant types are a list too.** `quant.Dequantable` is {F32, F16, BF16, Q4_0,
  Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, MXFP4, TQ1_0, TQ2_0, TQ2_F32}
  (TQ1_0 and bitnet.cpp's I2_S are re-packed into TQ2_0 or TQ2_F32 at
  conversion, exactly). Adding one is two pieces of
  work and the reference dequantizer in `internal/oracle` comes FIRST, because
  every kernel is gated against the oracle. `quant.PackedTypes`
  (`format/quant/types.go`) is the canonical list of packed types a gate sweeps;
  a hand-written copy anywhere else is a bug waiting.
- **The architecture is the converter's alone** (`convert.Arch`,
  `convert/arch.go`): it names which reference a source is read against, and
  nothing past the converter -- not the container, not `model/`, `engine/`, the
  devices or the composer -- holds one (`srcgate.TestNoFamilyCode`). A container
  written before v31 carries its old code as an opaque number for the
  migration's check (`jlm.Stating`, `convert.LegacyStated`). Codes are allocated
  by group: the dense llama family from the top of the code space down, the
  flagships counting up from 44, the state-space families from 100, the vision
  families from the top down, the encoders after BERT from 150, the
  vision-language families' text models up from 200; they must not move, since a
  pre-v31 container's code is read by number.

### The architectures and projectors

The list is `docs/architectures.md`: every architecture and projector, what is
special about its graph, its gates and its evidence. A new architecture or
projector adds one row there and its narrative in
`docs/engineering-history/model-correctness.md` (MC in the evidence lines
below; GK is `gpu-kernels.md`, PL `placement.md`).

### The chat template

- **Completion is the default and `-chat` is opt-in, on purpose.** Every board
  row, `TestGreedyMatchesLlamaCpp` and the teacher-forced comparison are
  completions of a bare prompt; defaulting the template on would silently make
  each a different measurement.
- **A rendered template is a complete prompt**: it is encoded with
  addSpecial=false, and the tokenizer must not add a special the template did
  not render (`TestChatTemplateIsAppliedAndSpecialsAreNotDoubled`).
- **A flag after the file is refused** (`misplacedFlag`, matching the flags the
  set defines): Go's `flag` stops at the first positional, so it would otherwise
  become prompt text.
- `jitllm convert -chat-template` (`convert.WithChatTemplate`) stores the
  publisher's template in place of the GGUF's (RULE 7m).

Evidence: MC "RULE 7's list, scope and refusals as they stood in AGENTS.md",
"The chat template, applied", and each architecture's section.

## RULE 7m: llama.cpp is a good-enough oracle, not the specification.

It is runnable, it is what most people run a GGUF through, and it catches gross
errors instantly -- which is what an oracle is for. It is not the authority on
what a model's tokenizer or graph DOES: the weights were trained against the
model's own reference, and llama.cpp carries a re-implementation of it.

**There are three authorities and they disagree**: the model's own artefact
(`tokenizer.json`, the transformers class -- what the weights were trained
against), llama.cpp's re-implementation (what most people run), and Go's
`unicode` package (whatever the toolchain ships, which is nobody's intent and
changes under a `go` upgrade). The tokenizer does not use the third: its tables
are generated and pinned to Unicode 15.1.0 (`tok/unicodedata.go`,
`scripts/genunicode`), so token ids do not depend on the Go toolchain.

**The rule is not "always match llama.cpp" and not "always be
standards-correct". It is: KNOW which one you are matching, choose per
divergence, and make the list of divergences a reviewed artefact rather than a
surprise.** Being exact where llama.cpp approximates is the better engine, but
it is a decision, because bug-compatibility with what users run has value too.

**A fourth authority outranks all three: the caller.** `tokenizer.ggml.pre` is
a label the converter wrote, an assertion about a converter rather than
evidence. The GGUF stays the default authority, and `tok.WithTokenizer(r)` lets
the caller override the pre-tokenizer with the model's own `tokenizer.json`,
parsed at run time by `pretok.ParsePreTokenizer`. `tok/pretok/table.go` is a
cache of that same parse (`go run ./scripts/gentok tok/pretok/table.go`,
`TestGeneratedTableMatchesArtefact`), and `scripts/gentok`'s `handWritten` map
holds only names with no `tokenizer.json` to parse at all, each with the
artefact that replaces it. The scope is conformance to an identified artefact,
not provenance.

The recorded divergences are `docs/divergences.md`, each with its measurement
in the evidence file. A new divergence adds one row there.

## RULE 8: every op a token runs is generated code. There is no interpreted tier.

Stated by the user: *"a missing kernel means we jit a basic version
of it, why would falling back work and not jitting a basic kernel?"* -- and,
of the device, *"the naive fallback is still jitted on the chosen device"*.

- **A shape with no optimized kernel gets a BASIC generated kernel**, on the
  host and on whichever device runs the block. A width, head dimension, k, row
  count or format is never a reason to finish in Go. On a device the only
  placement decline is *does not fit*; a missing kernel there is a kernel to
  write, not a block to hand back to the host.
- **Basic kernels are the floor, not the goal.** Every optimized kernel is
  still owed; where a basic one runs is the work list, not a design.
- **A block's generated code may differ per device and per model** (stated by
  the user). The JIT emits for the device in front of it: kernel form, tiling,
  reduction order and fusion follow that device's capabilities (sm_70 against
  sm_75 and up, subgroup width, shared memory) and the executed row width. The
  bar is the gated error margin (RULE 11c's coherence band) and speed, never
  bit-identity across devices or forms; "one kernel everywhere" is not a
  constraint (RULE 11e). A lever that pays on one device and does nothing on
  another is guarded by that capability, not removed.
- **The Go arithmetic lives in `internal/oracle` and only tests may import
  it.** `TestNoProductionCodeImportsTheOracle` walks the module and fails on a
  production import, so an engine that needs Go compute does not get past the
  gate. The oracle is what kernels are gated against; llama.cpp and
  transformers goldens are what the ARCHITECTURE is gated against (RULE 7m).
- **No bool from a kernel entry point means "I declined".** The `nn` entry
  points panic on a contract violation and otherwise run; a host with no
  packed kernel for a format makes `model` return an error naming the tensor,
  an x86 below the SSE tier (SSE4.1+SSSE3) refuses to open a model naming the
  missing extensions (`nn.HostRefusal`), and a host with no code generator at
  all refuses too (`nn.Available`).
- **No pregenerated assembly.** The only `.s` files in the project are the
  Go-to-generated-code call boundary, `engine/device/cpu/trampoline_{amd64,arm64}.s`
  (stated by the user). Every kernel is emitted at run time for the
  host in front of it. `cpu.TestNoAssemblyBesidesTheTrampolines` enforces it.
- **Gate the tail.** Every kernel gate sweeps ragged widths (1 through a few
  vectors) with a guard after the output, and is run against a violation --
  a tail kernel that is right at whole vectors and wrong at 7 is the shape
  every fallback in this tree hid behind.

The history -- the arm64 24.5x that returned TRUE, the counters that could not
see an op that never asked, the audits -- is in
`docs/engineering-history/cpu-kernels.md`, "The reference tier, before it was
removed".

## RULE 8b: f32 is the engine's precision. The oracle is f32 too.

There is no float64 on any load or inference path. `quant.Dequant32` decodes
straight to f32 through the same decoders, gated at zero ULP against the f64
signature. The oracle's matvec (`internal/oracle.MatVec`,
test-only) is f32 with Neumaier compensation, so the reference stays an
independent METHOD (compensated scalar) rather than the kernel's arithmetic
twice. What float64 is left is measurement -- the tuners' statistics
(`engine/seamtune.go`, `nn/tune.go`), the expert-use accounting
(`engine/hotness.go`) and setup scalars before a kernel call.

Why: it makes the elementwise ops emittable with the assembler that already
exists. It is not a perf win and must not be quoted as one (decode is bound by
weight traffic, and the conversions were not weights). The gate is
`TestGreedyMatchesLlamaCpp` plus the teacher-forced diff on the clean models.

Evidence: `docs/engineering-history/cpu-kernels.md` ("f32 is the engine's
precision").

## RULE 8a: a BLOCK is the unit, and it must stay orchestrable.

The device never learns what a transformer is beyond one repeating unit:
`model/` states the graph, `engine/` runs it, and nothing in `engine/device/gpu`
knows about embedding lookup, sampling or tied output weights. That is what makes `PrepLayer` able to decline
per block, and it is why a model too big for the card runs its first N blocks on
the device and the rest on the host.

**So a new architecture feature must reach BOTH tiers or the block stops being a
unit.** A graph change landed on the CPU alone silently removes that
architecture from the device — not as an error, as an absence. gemma2/gemma3's
post-norms sat in that state for months; they are `nn.LayerWeights.PostAttnNorm`/
`PostFFNNorm` now and the tier uploads and applies them. gemma2's attention
softcap, gpt-oss's sinks and mixture biases, and every activation kind followed;
a feature the device lacks is declined BY NAME until its
kernel exists (RULE 8), and gated end to end once it does (RULE 11c).

## RULE 9: fuzz with an exec budget, never a time budget.

    ./scripts/cap 4G -- go test ./convert/gguf -run=XXX -fuzz=FuzzParse -fuzztime=1000000x -parallel=2

A gate that looks identical whether it ran 8 thousand or 1 million cases is not a
gate. **Size nothing from untrusted input.**

## RULE 10: a test that proves nothing must not be a green line.

This has fired seven times and is the most expensive recurring failure here.

- **A skipping test is worse than no test.** Check the subtest RAN. A model
  path that misses (wrong `JITLLM_MODELS`, a name not in it) gives `ok` in
  0.001 s. A file behind a build tag that excludes the host prints "no tests to
  run" and exits 0.
- **Run every gate against a VIOLATION before believing it.** A gate that has
  never fired is not a gate, and a mutation that changes nothing looks exactly
  like a passing one.
- **Check the configuration under test was actually SELECTED.** "The two layouts
  agree" is worth nothing if both arms ran the same layout; a counter that counts
  fallbacks certifies nothing, and "it made no difference" and "it never ran"
  are the same number without a counter. Assert the count (blocks placed, tiles
  taken), not that the call succeeded.
- **A degenerate oracle is a failure, not a pass.** `NaN > 0` is false, so a
  non-finite reference once scored 13 subtests at "NMSE 0.00e+00"; check for
  non-finite values first.
- **A counter that counts the acquisition certifies nothing about the release.**
  A fake buffer whose `Free` does nothing hid four device leaks.
- **Diff token IDs, not just NMSE**, whenever a register allocation moves — a
  broken kernel runs at exactly the same speed. A perf row is not a correctness
  signal either: a collapsed model does the same arithmetic at the same speed.

## RULE 11: a missing artefact is a TASK, not a constraint.

If a gate cannot run because a model, file or tool "is not on this box", **go get
it in the background and carry on.** Do not record the absence as a reason.

Attention biases shipped gated only by "models without biases are unchanged",
with the written excuse *"it cannot be gated without a qwen2 GGUF, which this box
does not have"* — a 941 MB download. A real bug lived behind that sentence for
weeks. The same reflex covers tools: `spirv-val` logged "not installed" while
sitting in a snap path, and `llama-mtmd-debug` — which solved a VLM bug in one
command — sat unused in the same bin directory for a day. The Vulkan validation
layer, fetched without root, named in one run a fault that six eliminations had
not: a backend with a conformance checker gets it pointed at a wrong model
first.

## RULE 11b: "pre-existing" is not a status. It is a bug you just found.

A red gate, a stale golden, a wrong number or a broken invariant that was already
there when you arrived is **yours now**. "Pre-existing" describes when it
started, never whether it is your problem -- and writing it down as a reason is
the same move RULE 11 refuses for a missing artefact: it converts work into a
constraint and leaves the next session to find it again.

- **A gate that is always red is a gate nobody reads**, which is RULE 10 from the
  other direction.
- **Blessing a golden is not fixing it.** `-update` makes a golden agree with
  whatever the code now emits, which is exactly what a wrong change would also
  produce. A golden may only be updated once the behaviour behind it has been
  re-derived from something that is NOT the golden.
- **A gate that has been red since a refactor is that refactor's unfinished
  half**, and the commonest shape is a test that worked by accident.

Evidence: `docs/engineering-history/scheduling-and-measurement.md` ("RULE 11b's
cases").

## RULE 11d: a gate that never runs on an architecture does not cover it.

**Run the MODEL gates on both hosts before believing an arm64 change**, not only
the kernel packages. The ones that catch the class of bug that lives above a
kernel and below a model compare the engine against something that is not
itself -- the batched path (`TestBatchMatchesForward`) and llama.cpp's goldens
(`TestGreedyMatchesLlamaCpp`):

    GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 ./scripts/cap 8G -- go test -c -o model.test ./engine
    # copy model.test and the package's testdata/ to the arm64 host, sign it
    # with .github/release/entitlements.plist (codesign -s - --entitlements), then:
    ./model.test -test.run 'TestGreedyMatchesLlamaCpp|TestBatchMatchesForward|TestEveryModelRunsGenerated'

(the M4 runs were driven by a local helper, `scripts/local-mac-arm.sh`, not in
the repo; a shipped binary has no `go.mod` above it, so set `JITLLM_MODELS` and
`JITLLM_SHIPPED_BINARY=1` in its environment there). The two hosts
cover different halves -- Linux pins what the bytes decode to, the M4 what they
do when executed -- and a gate that skips on both is the one to hunt: diff the
skip lists rather than reading either one.

The SSE tier binds the same way on amd64:

- **A probe wired to nothing is worse than no probe.** The tier is decided once
  (`cpu.HostTier`) and every emitter is reached through that tier's table.
- **Mapping proves nothing about executing.** An SSE kernel gate CALLS the
  kernel and runs the VEX-leak gate (`requireSSEKernel`) on every kernel at every
  shape.
- **Run the WHOLE suite on the SSE host, not a filtered slice**, natively on the
  Atom:

      model.test -test.run 'TestEveryModelRunsSSE|TestGreedyMatchesLlamaCpp|TestBatchMatchesForward|TestPrefillMatchesForward|TestEveryModelRunsGenerated|TestHybridRuns|TestTowerEncodes'

- **The SSE and AVX2 tiers are two transcriptions, not one**: a comparison
  between them pins the KV width and holds the SSE arm to a same-process control
  (`TestEveryModelRunsSSE`). A lower tier is a CPU device opened as one:
  `WithDevices("cpu:sse")` for a model (`cpu:novnni` and `cpu:nodotprod`
  likewise), a `cpuArm` at `cpu.AtTier(cpu.TierSSE)` for a composer gate.
  Nothing forces a tier process-wide.

Evidence: `docs/engineering-history/cpu-kernels.md` ("RULE 11d's cases", "THE
SSE TIER").

## RULE 11c: the kernels are gated. The tier's wiring of them is not.

**A new graph feature is not on the device until an end-to-end gate runs a model
that HAS it, every block placed, against the host.** A gate on a composed form
proves the kernel and says nothing about whether the tier asked for the right
one; a kernel gate and a host gate can both be green while the tier wires them
wrongly. Declining a block is not the answer to a packing choice of the file:
that dies at conversion.

- **CPU-vs-device token equality is not the bar; coherence is.** Device logits
  differ from the host's in the f32 reduction-order band (0.2-0.94 of a logit),
  and the split tuner picks a reduction order per process: pin it
  (`JITLLM_GPU_TUNE=0`) before comparing anything. An equality gate pins every
  knob its equality depends on.
- **The band grows with the blocks below the seam.** An absolute logit
  difference across a deep seam is not evidence of a device bug; the falsifiable
  form is to move the seam and see whether the error tracks the depth below it
  (`engine.TestHybridDeviceErrorIsAmplification`,
  `engine.TestPostNormDeviceErrorIsAmplification`).
- **A gate that pins a knob does not cover that knob's default.** Run the
  shipping default (the f16 KV cache on amd64) on a partial seam too.

Evidence: `docs/engineering-history/gpu-kernels.md` ("RULE 11c's cases");
`docs/engineering-history/placement.md` 15i.

## RULE 11e: do not invent constraints. Name a limit's source before obeying it.

Before code waits, locks, declines, serialises or falls back "because X",
name what X is: a hardware or driver limit, cited from its documentation or
measured, or a choice this code made. Only the first is a constraint; the
second is a design, and the work is to find the way past it that gives the
best rate.

- **A lock is a claim about what it protects.** Name the state, then ask
  whether that state needs to be shared at all.
- **Inherited limits rot fastest.** A layer that copies the limit of the
  layer below upward keeps it after the layer below is fixed, and widens it.
- **"The hardware does X" needs the hardware's documentation or a
  measurement**, not the way our backend happens to drive it.
- **A real limit is where the search starts, not where it ends**: batch,
  stage, queue or reorder around it before accepting its cost.

Evidence: `docs/engineering-history/gpu-kernels.md` ("RULE 11e's cases").

## RULE 11f: before building anything, find what already does it.

Find the machinery that already does the job, or can be adapted to it, and use
it. Then check that what you reuse or build keeps the five principles (JIT, no
Go compute, relocation, paging, low to no allocation) and prove no regression
with RULE 6's checklist. A PR reports all three, each under its own heading:
**what existed and was reused**, **the principles kept**, **the regression
evidence**. A PR that cannot name what it searched for first skipped the first
step.

Evidence: `docs/engineering-history/scheduling-and-measurement.md` ("RULE 11f's
cases: a cache that existed, a store that existed").

## RULE 12: a comment explaining why something is NOT done is a claim about the code, and it rots.

Before inheriting a "we do not do X because Y", check Y at the source. It costs
one grep and has paid six times: a comment claiming the IR was "not getting"
shared memory, one claiming folding the projection in "saves nothing", one
claiming qwen2 biases were not loaded (they were), one claiming they were (they
were not), a 5% threshold living on in a source comment after the rule that set
it was retired, and one claiming the vision tower has no block pages. The same
holds for a number cited as measured: check that the measurement it came from
still stands.

Evidence: `docs/engineering-history/placement.md` ("RULE 12's tower case").

## RULE 13: device kernels — three invariants that do not bend.

- **No kernel may read a buffer it writes within one phase.** The IR has no
  conditional execution outside a loop, so surplus threads are CLAMPED to the
  last work item; that is benign only when the result is a pure function of
  read-only inputs. A grid barrier (`ir.OpGridBarrier`) ends a phase: every
  write before it is visible to every read after it. `ir.Validate` refuses a
  buffer read and written in one phase.
- **A grid barrier runs only on a grid the device holds resident at once.** A
  workgroup waiting for one that cannot start until it finishes never returns.
  `backend.GridLimit` states the limit from a documented source (the CUDA
  occupancy calculator, a vendor extension's compute-unit count, the
  IORegistry's core count), every launch refuses a wider grid, and a device
  that states none cannot run one.
- **Poison the buffer, do not trust the allocator.** A fresh device buffer is
  usually zero, so reading memory nothing wrote is invisible until it is not.
  `0 * NaN` is NaN.

## The shape: model, instance, session, scheduler

Stated by the user, and the shape every entry point takes:

1. **A model** is the weights, loaded once.
2. **Composing makes an instance**: every kernel for one placement (which
   device each block runs on, how heavily it pages). A model has as many
   instances as the caller makes, each its own placement.
3. **Composition happens only when an instance is made, relocated or tuned
   at runtime** -- never in a prefill, a token or a draw, and never because a
   session opened (`engine.TestComposesOnlyWithAnInstance`,
   `Instance.Account(dev).Asks`, the CPU's as a card's; a card's
   `UnformedAtStep`). Each device composes only inside a placement call, and
   a form asked at a step is declined unformed
   (`engine.TestTheCPUDeviceComposesNothingAtAStep`,
   `gpu.TestTheCardComposesNothingAtAStep`).
4. **A session is a conversation's history** -- its own KV pages, recurrent
   state, positions, logits and device view (`engine.Session`). Opening one
   composes nothing.
5. **The scheduler leases a session an instance per phase** (prefill,
   generate), the lease held for the whole phase and handed back
   (`engine.Scheduler`, swappable by `WithScheduler`; `engine.Pool` is the
   default). Prefill and generation may lease different instances; a session
   moving between them moves only the blocks placed differently, the source
   freed once the copy has landed.
6. **Batching is many sessions leasing one instance**: each step carries every
   decoding session's next token and prompts' chunks beside them, one pass
   over the weights per instance (`engine.StepRuns`, `engine.Batcher`), rows
   reassigned every step (`engine.TestSessionsShareAnInstance`).
7. **Relocating an instance moves the history of every session leasing it**;
   parking a session moves its history alone and leaves the instance where it
   is (`Instance.Park`, `Instance.Settle`).

Evidence: `docs/design/instances-and-sessions.md`.

## Scope

**jitllm is an inference OS, and orchestration is its core.** It should use every
CPU core and every accelerator on the machine, and move layers and pages between
them to deliver the best rate it can -- and every one of those decisions must be
forcible by the caller through a `With...()` option. That is the standing
direction, stated by the user, and it is what "done" means below.

**The analogy is conceptual, not literal: jitllm is ONE process.** What Linux
does for processes and threads, jitllm does for models and blocks:

| Linux kernel | jitllm |
|---|---|
| process | a loaded model |
| thread | a layer / block |
| CPU cores | CPU cores AND every accelerator |
| the scheduler | which block runs where |
| a physical page | a packed block in a slot |
| swap / backing store | the container, through the pager |
| a page fault | a page-in |

**"Scheduler" is already taken in this tree for a different axis.**
`engine.Scheduler` puts SEQUENCES onto batch rows (admission and retirement,
driven rather than self-running). Block placement puts BLOCKS onto compute
resources with memory paged; it is the core. You cannot schedule what you cannot
relocate, which is why block paging is primed whether or not the model fits
(`docs/design/block-paging.md`).

**Standing goal, not current work: a kernel composer.** Stated by the user:
the JIT should eventually *compose* kernels, choosing and fusing from the
kernels it already knows, the operation graph and the hardware in front of it
(ISA, subgroup width, cache and shared-memory sizes, measured per-launch and
per-byte costs), rather than hand-writing each fusion. Keep new kernels and
tuners expressible as data a composer could select from (a shape, a cost
measured on this host, what it fuses), and prefer that to a special case baked
into a call site.

**What is missing is the PLURAL**: several devices at once rather than one,
layer migration across devices rather than only to and from the host, and
device-side paging. Not done, named rather than implied: two instances putting
one block on different cards of one tier (the tier keeps one route a block),
and the sessions-per-device budget.

- **The libraries read no environment.** Every knob is a `With...()` option;
  `cmd/jitllm` reads the `JITLLM_*` names and passes them through. The one file
  that reads an environment variable is device fault injection under the
  `jitllmfault` build tag (`tier/fault.go`), not in a release build. Count a
  package's knobs by what it READS, not by a literal grep for `os.Getenv`. A
  `t.Setenv` of a migrated name reaches nothing and leaves a gate green on a
  configuration never selected.
- **A session owns its history; the instance owns the placement; the tier owns
  the weights.** Each session's device history is its own view of the tier
  (`gpu.GPU.Attach()`), which satisfies `nn.LayerDevice` unchanged. Refusing a
  second session is not an option, stated by the user: *"we can't refuse,
  jitllm is an inference os multimodel multi devices multi instances"*. A
  session's view letting a block go leaves it on the card for every other view
  holding it (`engine.TestABatchLeavesAnotherSessionAlone`).
- **KV is held once per device** (stated by the user). Each session reads each
  position's history from one place. On one device, sessions and the prefix
  cache share one page for a prefix they agree on (`SharedStore` on the host,
  the card's named pages on a card), never a copy per session. Devices may each
  hold a copy for sessions placed on them independently: a sharing session's
  block moves to a card like any other, taking the card's page of its prefix
  or naming the one it brings (`kvCache.nameMoved`). A session keeps no mirror
  of a history it moved, and a host page the card now holds goes at its last
  holder's release.
- **A device that shares the host's memory is not a second pool.** An integrated
  GPU's heap and Apple Silicon's are system RAM: `UnifiedMemory()` reports it on
  every backend with one contract -- SUBTRACT, DO NOT ADD. Vulkan decides this by
  device TYPE rather than heap flags (an rBAR discrete card's host-visible heap is
  still VRAM).
- **One physical device under two backends is one device.** Identity is the
  device UUID (`VkPhysicalDeviceIDProperties.deviceUUID`, `cuDeviceGetUuid_v2`),
  NEVER the device name -- two identical cards share a name, and a name match
  deletes half a machine's VRAM. A backend that cannot supply an identity leaves
  its devices distinct. For one card under several backends the DEFAULT set
  takes CUDA > Metal > Vulkan; dedupe governs the default and the arithmetic,
  never what a caller may ask for, and the duplicate is still listed
  (`same_device_as`).
- **More device blocks is not always better.** A device whose memory is host
  memory adds no bandwidth and can make a placement slower (`seamtune` is off by
  default; placement.md has the measurement).
- **The engine runs one weight format, and GGUF is not it.** `model.Open` refuses
  anything but a `.jlm` container with a `NotConvertedError` carrying the convert
  command; `gguf.Open` survives for the converter (`convert.FromGGUF`) and nothing
  else on the weight path. Two formats would mean two loaders, two layouts and
  two answers to every residency question; both tiers read the device layout.
- **Paging is always on.** A page is either allocated or it is not; "the model
  fits" means enough pages were allocated to hold every block, and nothing is
  ever evicted. There is no paging-off mode, and a measurement that reports one
  measures a path that no longer exists.
- **The test suite converts** (`engine/jlm_helper_test.go`) because it has
  to run what ships; an open error is a failure, never a `t.Skipf`.
- **`composer/` generates code and never runs a model** (stated by the user).
  The passes, the lowerings and every device's emitters live there; what runs
  a model is the devices' runtimes under `engine/device/`, the CPU's beside
  the cards' because the CPU is a device.
- **Three GPU backends, one interface.** CUDA, Vulkan and Metal are reached
  through goffi (`backend/ffi`) with no cgo, one `backend.Device` interface over
  three implementations. **Do not add a fourth abstraction in advance of a second
  implementation of anything.**
- **Models live on a dedicated model disk** (`JITLLM_MODELS`) — never `~`, never `/tmp`.
  Tests find them through `JITLLM_MODELS` (default: the repo's `models/`);
  `docs/testing.md` has every model-selecting test variable.

Evidence: `docs/engineering-history/placement.md` ("Scope as it stood in
AGENTS.md", 15u); `docs/design/device-sessions.md`.

## There is one model. Vision is not a second one.

**A vision tower is a transformer over patches.** Every way the engine once
treated it as a separate thing (a second GGUF, loader, container, page budget,
JIT and state) was llama.cpp's technical debt, not a property of the model.

- **GGUF is the converter's input, so the two-file contract dies at
  conversion**: `jitllm convert model.gguf mmproj.gguf out.jlm`. `model.Open`
  builds the tower when the container carries one; there is no `OpenTower`
  and no `-mmproj`. A projector width that does not match the text model is
  refused at conversion, where there is a file in front of a person.
- **The converter states the tower; the engine knows no family.** What a
  family's reference does -- its preprocessing steps, its entry and head as
  programs of ops, its block's attributes, its picture's markers -- is written
  into the container at conversion (`jlm.Tower`), and `model.Open` lowers that
  record into graphs, never switching on `Projector`. A family the converter
  does not state yet is refused at Open by name.
- **Two page arrays, one pager.** Vision block i is at
  `VisDataOff + i*VisPageSize`, because one page size is wrong in both
  directions; vision blocks occupy `NBlocks..NBlocks+NVisBlocks-1` of one block
  space, so residency, eviction and the budget (in bytes, `jlm.File.SetBudget`)
  are written once. The pager does not know a tower exists.
- **The tower is a segment of the model, run by the one runner**: its blocks are
  `Model.layers[NBlocks:]`, run through the text runner's block body on the text
  State's JIT and pool. Two pools on six P-cores is the worst configuration this
  engine has. `docs/design/vision-as-blocks.md` is the design.
- **A picture is a span of the prompt, not a stage before it**: the prefill
  encodes it through the tower's instance wherever its rows are not restored
  (`encodePictures`). It is named by a `RowsKey` (sha256 over what the tower
  reads, known before a block runs), so the prefix cache names the rows after
  it as it names text, and a restore past the picture runs no tower block
  (`TestTwoTurnImageChatRestoresPastThePicture`).
- **A picture of a size of its own** (`jlm.PreSmartResize`) is encoded in the
  smallest of the squares the entry is formed for at placement, its rows the
  call's live rows (`composer.TestComposedCPUFoldFollowsTheCall`).
- **The tower runs on the device** through the one placement path
  (`Instance.offerRange`); a block the device cannot express is declined by the
  device, never withheld by the model (RULE 8a). A vision block is one
  submission: bidirectional attention needs every row's k and v first.
- Gates: `jlm.TestTowerRefusesWhatItCannotRead`, each stater's
  `convert.Test*StatesItsTower` (SmolVLM, Gemma 3, llava, InternVL, Qwen2-VL,
  Qwen2.5-VL, Qwen3.5-VL) and `convert.TestQwen3VLStatesItsTaps`,
  `TestStatedTowerComposesOnTheCPU`, `TestStatedTowerBlocksArePagedBlocks`,
  `TestStatedTowerEncodesAPicture`, `TestWarmImageEncodeDoesNotAllocate`,
  `TestStatedTowerSeesThePicture` and each family's `Test*TowerSeesThePicture`
  (a blank picture is the violation), `TestStatedTowerRunsOnTheCard` (every
  block and a partial seam, against the CPU), `TestQwen2VLTowerMatchesTransformers`
  (Qwen2-VL's, Qwen2.5-VL's, Qwen3.5-VL's and Qwen3-VL's: the processor's
  samples, every one, the tower's rows and Qwen3-VL's taps'),
  `TestQwen2VLTowerRunsOnTheCard` (their grid rotary from the card's own table,
  Qwen2.5-VL's windows and Qwen3-VL's taps, every block and a half seam),
  `TestQwen2VLMatchesTransformers`, the M-RoPE gates (`TestMRope*`, the text
  side on transformers' own tower rows), `TestChatSpansFollowTheTemplate`,
  `TestPictureFormatsEncodeAsTheirPixels`, `composer.TestComposedCPUPicture*`,
  `composer.TestComposedCPURegroup`, `composer.TestComposedCPUPatchifyTiles`
  and `composer.TestComposedCPUUntile` (the windows' order, against
  get_window_index), `composer.TestComposedCPUPosResampleFollowsTheReference`
  and `composer.TestComposedCPUPosResampleCubic` (a learned table at the
  picture's grid, fast_pos_embed_interpolate and grid_sample's bicubic);
  through the front ends
  `server.TestImageRequestsMatchTheEngine`, `cmd/jitllm.TestImageRunsThroughTheTower`
  (a run restored past the picture answers alike) and
  `common/engine.TestAnAttachedPictureReachesTheModel` (a follow-up turn sees
  the picture in its history).

**The bar for a new projector family**, on top of RULE 7's:

- **Its preprocessing is the reference processor's, to the bit**, stated as
  steps (`jlm.PreStep`), held to transformers' pixel values on real pictures
  (PNG -- a JPEG's decoders disagree first), or a divergence recorded in
  `docs/divergences.md`. Every pixel and patch passes through generated code
  (`graph.OpPicture`, `graph.OpPatchify`): the decoded rows, the resize, the
  normalisation, the patch gather and the position tables built per grid;
  what is Go is geometry from the sizes and the tower's constant tables
  (`TestImagePathRunsGeneratedCode`).
- **A gate against transformers runs the PROJECTOR ALONE too**
  (`TestProjectorMatchesTransformers`): the tower's int8 activations put
  end-to-end features wider than what the projector's own errors move.
- **The class token goes where the reference puts it** (InternVL first, as
  transformers does and llama.cpp does not; Llama 4 last).
- **A bidirectional run in a causal prompt is a per-row key count**
  (`Span.Bidir`, `nn.KeyRunDevice`, `engine/bidir.go`): a prefill chunk
  never cuts one, and a device that cannot honour it does not run the chunk.

Evidence: `docs/engineering-history/model-correctness.md` ("Vision as blocks:
the tower on the one runner" and each family's section).

## No mmap of files. Anonymous memory is decided by measurement.

Stated by the user, about FILES: the engine does not let the OS manage the model
file's pages. A file mapping hands residency to the kernel, measured slower even
with madvise (ReadAt x16 at 4.60-4.73 GiB/s against an mmap fault's 1.28-2.22),
and mmap does not remove the copy, it moves it into the page cache. `jlm` reads
pages with **O_DIRECT**: one copy, ours, no cache.

Anonymous mmap for memory whose lifetime the engine manages -- frames, expert and
KV pages, buffers -- is allowed, and taken where it measures better (it keeps
those bytes off the Go heap, RULE 2f). Reading LESS into a page (`EnsureRange`)
is legitimate; fragmenting a page's memory is what this rule rules out.

**Pages are static and fixed at conversion, and every expert is its own page**
(container v26, on the user's decision): a mixture's banks sit in a third page
array, one page per (mixture block, expert), evicted least recently used; block
pages keep the MRU rule the cyclic scan makes optimal. Count reads by syscalls,
not by `/proc/<pid>/io` `read_bytes`, which misses whatever the buffered handle
serves from cache.

Evidence: `docs/engineering-history/placement.md` ("No mmap of files: the
measurements, and expert pages"); `format/jlm/read.go`.

## Open questions and shelved levers

They live in `docs/open-questions.md`, a work list rather than rules. RULE 6
governs every entry: a number decides ORDER, never admission. A closed entry
moves to the matching evidence file.
