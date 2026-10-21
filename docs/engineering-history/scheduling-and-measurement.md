# Scheduling, dispatch and measurement discipline — evidence

> **EVIDENCE, NOT INSTRUCTIONS.** This file records what was measured, what was
> tried, and what was wrong. It is the reason a rule exists, not the rule.
> **Binding requirements live only in `AGENTS.md`** -- if something here reads as
> an instruction, `AGENTS.md` is what governs. Numbers here are dated by the
> commit that added them; treat any of them as provisional until re-measured
> (RULE 1a: a baseline pinned to a version rots silently).
>
> Split out of a single 415 KB AGENTS.md, after a review
> identified the file's real failure as *unresolved authority* -- historical
> findings, current instructions and withdrawn conclusions all speaking in the
> same imperative voice.

## ★ RULE 2: a perf claim is paired, interleaved, and gated.

A/B/A/B **within one process**, n >= 20, never A-then-B: the GPU clock ramps 840->1755 MHz and the
CPU drifts 25% thermally, and one sequential pass "measured" a fusion win at -21.7% where
interleaved rounds showed +5.4%. Report the **median of per-round ratios** with the IQR; reject if
IQR/median > 0.10. Gate on `/proc/loadavg` 1-min < 2.0 and thermal zones < **90 C** (three zones
read 99-101 C here during ordinary use — a gate set at 78 never opens and gets disabled on day one).

**★ 2c. SOAK TO THERMAL STEADY STATE BEFORE MEASURING. COOLING BETWEEN SAMPLES IS THE INTUITIVE FIX
AND IT IS THE WRONG ONE.** This laptop saturates within seconds of six-core AVX2 and settles at 100 C
with the clocks pinned. On the way there it passes through every frequency in between, so WHERE on
that ramp a sample lands is a bigger effect than anything being measured. The same A/B, three ways:

    cold start, two passes back to back     ratios 0.9733 and 0.8530     14% apart, both "passed" IQR
    6 s cooldown between every sample       IQR/median 0.154             REJECTED
    soaked to 100 C first, then measured    IQR/median 0.067             usable, and reproducible
    the same soaked A/B at ONE core         IQR/median 0.004             one core does not heat the box

**The first row is the dangerous one**: within a pass both engines drift TOGETHER, so the IQR gate
sees a tight distribution and passes, while the ratio itself moves 14% between passes. A dispersion
gate cannot see a bias that is common to both arms. Run the whole comparison twice before believing
it; that check costs one repeat and it is the only thing that catches this.

Cooling between samples puts every sample at a different point on the ramp. Saturating first puts
them all at the same point -- the slowest one, which is also the stable one, and stability is what a
ratio needs. Absolute numbers from a soaked run are ~20% under this box's peak and must not be
compared with a cold single run; **the ratio is the measurement**.

`scripts/vs-llamacpp.sh` does the soak, and is where any jitllm-versus-llama.cpp claim should come from.
It is the one comparison jitllm cannot do in one process -- the other engine is another program -- so it
interleaves ABBA, reports the median of per-round ratios with its IQR, gates on PSI, and prints every
raw sample so a bad run is visible rather than averaged away.

**★ 2d. AND THE BOX POWER-CLAMPS TO 900 MHz AT RANDOM, WHICH NEITHER GATE COULD SEE.** A decode run
would read 25 tok/s where its neighbours read 65 -- uniformly across every op, with byte-identical
output and the same profile split. That is the same code running slower, and the giveaway is that
the die gets COOLER while the rate collapses: 84 C on a clamped run against 99 C on the fast ones,
because a package at 900 MHz stops producing heat. It is the sustained-power limit, not temperature.
Rate against the median frequency sampled during that same run:

    3599 MHz -> 74.68      3100 MHz -> 64.56      1500 MHz -> 37.95
    3500 MHz -> 73.00      3100 MHz -> 63.54       900 MHz -> 25.18

**THE IQR GATE IS STRUCTURALLY BLIND TO IT.** The clamp lands on whichever arm happens to be
running, so it does not cancel in an ABBA the way drift does; and one clamped sample in twelve moves
the MEDIAN while leaving the DISPERSION inside 0.10. Two consecutive passes of one comparison read
0.9607 (IQR 0.045, PASSED) and 0.9446 (IQR 0.526, REJECTED) -- which is RULE 2c's "run the whole
comparison twice" earning its keep, and the only reason it was noticed at all.

`scripts/vs-llamacpp.sh` now records the median CPU frequency observed during EVERY sample and
discards any sample below `JITLLM_CLAMP_FRAC` (0.75) of the session median. The first gated run caught
a round where the clamp had hit **llama.cpp** instead (844 and 894 MHz, 26.2 and 31.1 tok/s) -- which
without the gate would have flattered jitllm by the same mechanism.

**A clamped sample is DISCARDED, never averaged**, because it measures the power budget of the box
and not the engine. And the gate only catches the hard clamp: after a long session this laptop
drifts into a partial band around 1300-1700 MHz that 0.75 keeps, and in that state no gated number
is obtainable at all. **When two passes disagree and the frequencies are low, the answer is to stop
measuring and let the box rest, not to raise the round count.**

**★ 2b. THE QUIET-BOX GATE IS PSI, NOT THE LOAD AVERAGE, BECAUSE THE LOAD AVERAGE IS WRONG HERE BY
4x.** Measured on a genuinely idle machine -- ONE runnable thread, which was the `ps` doing the
looking, and no process above 6% CPU:

    /proc/loadavg               4.42     the old gate: "something else is running"
    /proc/pressure/cpu avg10    3.53%    nothing was waiting for a CPU at all
    /proc/pressure/io  avg10    0.00%

And it did not decay: it sat above 4.0 through eight minutes of idle. Linux counts uninterruptible
tasks in the load average, and this desktop runs dockerd with **1226 threads**, which keeps a steady
trickle of them. The number simply never returns to a benchmark's idea of quiet.

So the gate that RULE 2 has always specified could not open on the machine RULE 2 was written for.
That is the failure mode the thermal limit's own comment names three lines above it -- "a gate that
never opens is a gate that gets commented out on the first day, taking the load check with it" -- and
it had already happened to the load check without anyone noticing, because the way you notice is by
being blocked, and the way it actually presents is a `t.Skip` that looks like the test passing.

`bench.Guard` now reads `/proc/pressure/cpu`'s `some avg10` and refuses above 25%. That is the
quantity the gate was always reaching for: the share of the last ten seconds in which anything was
kept waiting for a CPU. It responds in ten seconds instead of sixty, and it is zero on an idle box.
The load average survives only as the fallback where PSI is absent, at a threshold loose enough to
catch a genuinely loaded machine and no tighter.

Every benchmark declares `bytesPerIter` and the harness **panics** if the implied GB/s exceeds the
host ceiling. This mechanically catches the two ways a microbenchmark lies: a working set that fits
in the 24 MB L3 and reports cache as DRAM, and identical accumulator chains that the compiler CSEs
into one (which reported 231 GFLOP/s/core — impossible at 2 FMA ports).

Always report tok/s **and** bytes/token **and** effective GB/s as a fraction of a wall measured in
the same process. "85 tok/s" is unfalsifiable; "85 tok/s = 45.3 GB/s = 81% of the 55.7 GB/s wall" is
checkable with a calculator. Ceilings: tinyllama CPU ~105 tok/s, gemma ~33. Above the roofline is a
harness bug, not a win.

**★ 2a. THE ENGINE CONVERGES, SO ONE RUN MEASURES A COLD ENGINE.** jitllm tunes itself while it runs --
`engine/nn/tune.go` times consecutive real decode tokens and climbs the interleave width, `tier/split.go`
sweeps the split per shape at load, `tier/choose.go` probes the GPU APIs -- and some of that is
cached to `~/.cache/jitllm` while some is not. **A first run therefore measures the search, not the
answer.** Before quoting a rate: run the same prompt several times and take the LAST, or warm the
cache and say which. A number from a cold cache is a lower bound on what the engine reaches, and
comparing a cold jitllm against a warm llama.cpp is not a comparison.

This is also why `jitllm verify -ab` exists and why RULE 6 insists the A/B live inside ONE process: both
arms then see the same converged state, so the comparison is of the change and not of how far each
side had got.

## ★ RULE 3: every heavy job runs under `scripts/cap`. NO EXCEPTIONS, INCLUDING SUBAGENTS.

    ./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go test ./... -count=1

`MemoryMax=8G MemoryHigh=7G MemorySwapMax=0`. `MemoryHigh` is what stops systemd-oomd killing GNOME
Shell instead of the test. `MemorySwapMax=0` matters because this box has 31 GB of swap, and
swapping is how an OOM becomes a 20-minute freeze instead of a clean kill. Uncapped runs require an
explicit `JITLLM_NO_CAP=1`.

This rule is inherited from warpjs-itr2, where running a suite uncapped OOM-rebooted this machine
mid-session. A crashed process is recoverable; a rebooted laptop is not.

**★ 3a. THE RULE BINDS SUBAGENTS AND WORKFLOWS, NOT JUST THE MAIN LOOP.**

Killed the user's GNOME session once. Four recon agents were dispatched in parallel, each
told to compile C benchmarks with "100+ MB working sets", mmap RWX pages and hammer `runtime.GC()`.
Every command the main loop ran that day was capped; not one of the agents' was, because the prompt
never said to. Four uncapped benchmark harnesses at once is precisely the scenario this rule exists
to prevent, and the 15-minute load average peaked at 18.4.

So:

1. **Every prompt that dispatches an agent which may build or measure must contain the cap command
   verbatim**, not a reference to this file. An agent that has not been told will not do it.
2. **Never fan out parallel measurement agents.** Benchmarks need a quiet box to be valid anyway
   (RULE 2 gates on load < 2.0), so N agents measuring at once are both dangerous *and* producing
   numbers that must be thrown away. Measure serially, in the main loop, under cap.
3. **Prefer doing measurement work in the main loop.** Delegate reading and design, not load.

**★ 3b. THE RULE IS ENFORCED BY A HOOK, BECAUSE WRITING IT DOWN DID NOT WORK.**

It was violated twice on the same day, killing the desktop both times. The second time the offending
command was `gofmt -w jit_old/ && go test ./jit_old/ -run='TestEncodings|TestObjdump'` — two trivial tests,
which is exactly why it felt safe. It is not the test that is heavy, it is `go test`: the package
gets compiled first, at whatever parallelism the toolchain feels like.

`scripts/require-cap.sh` is a `PreToolUse` hook (wired in `.claude/settings.json`) that refuses any
`go test|build|run|vet|generate`, `gcc`/`clang`/`g++`/`nvcc`, `make`, or direct run of a `.test` or
`dev/probe/` binary unless the command contains `scripts/cap` or an explicit `JITLLM_NO_CAP=1`. It fails
open on malformed input — a broken hook must never wedge a session — and it ships with a block/allow
test matrix.

It matches only at **shell command positions**, with quoted strings and heredoc bodies stripped
first. The first version regexed the raw string and refused a `git commit` whose message contained
the word "make". A safety check that blocks real work is a safety check that earns itself a bypass,
so false positives matter almost as much as false negatives.

The general lesson, which is worth more than the specific rule: **a safety rule that depends on
remembering it is not a safety rule.** If a constraint has been broken once, the fix is a mechanism,
not a firmer promise.

**★ 3d. AND "MEASURE SERIALLY" WAS STILL ONLY A PROMISE. IT IS A MUTEX NOW.**

RULE 3a says "never fan out parallel measurement agents" and RULE 2 gates on a quiet box, and
nothing enforced either: a script checks PSI once at the top and then measures for half an hour,
so an interloper that starts at minute twelve is invisible to every gate this project has.

Once, four wait-loops left over from an earlier part of the session were still sleeping on
conditions that no longer existed -- their task files were gone with the context that made them --
and one of them would have run `go build` plus sixteen prefill runs the moment its condition fired.
It fired inside a gated A/B, and the round it landed on read 1.005 where its five neighbours read
1.033-1.048. **The dispersion gate passed anyway**, because one bad round in six moves the median
less than the IQR band; it was caught only because the ratio's own neighbours disagreed with it.

`scripts/cap` now takes a `flock` over every heavy job:

    MEASUREMENT payloads   jitllm run/bench/verify, llama-bench, the *bench probes,
                           a compiled .test, `go test -bench`        -> EXCLUSIVE
    everything else        builds, plain `go test`                   -> SHARED

so two measurements exclude each other, a build excludes a measurement, and two builds still run
concurrently. A waiter prints the holder's pid and command line, because the point is to make the
stray VISIBLE rather than to serialise silently -- the failure above presents as a number, not as a
conflict. `JITLLM_NO_LOCK=1` opts out; `JITLLM_LOCK_WAIT` sets the timeout; `scripts/cap --selftest` runs
the classifier over a twelve-case block/allow matrix.

**It went in `scripts/cap` because RULE 3 already forces every heavy job through it and
`scripts/require-cap.sh` already enforces that.** A new script would have been a new thing to
remember, and RULE 3b is the standing verdict on those: *a safety rule that depends on remembering
it is not a safety rule.* This is the third time that has been settled by a mechanism rather than a
firmer promise, and the second time the mechanism was bolted to the cap.

★ THE GENERAL HAZARD, WHICH OUTLIVES THIS FIX: **background work started before a context
compaction keeps running afterwards, and neither end knows.** The loops were waiting on files whose
names only made sense in the earlier context, so nothing would ever have collected them. Kill
background jobs when the work that spawned them is finished, not when they are noticed.

**★ 3c. THE THING THAT WAS ACTUALLY KILLING THE DESKTOP WAS `systemd-oomd`, AND IT WAS NOT DISABLED.**

The third kill happened with the job *properly capped*, which is what finally made it diagnosable.
`journalctl` named the killer:

    systemd-oomd[...]: org.gnome.Shell@x11.service: systemd-oomd killed some process(es) in this unit

`user@1000.service` carries `ManagedOOMMemoryPressure=kill`, so oomd kills on **PSI pressure, not on
usage**. Every capped scope is a descendant of it. The moment a job crosses `MemoryHigh` the kernel
throttles it, and *throttling is itself the pressure signal oomd acts on* — so the cap contains the
memory and oomd kills GNOME Shell anyway. The offending command was `cap 4G -- go build ./...`, and
`scripts/cap`'s own comments already document that 4G throttles a Go build "to 3.6% CPU for 21
minutes". Too small a cap is not the safe direction.

**oomd had been disabled and was running regardless**, because it is socket-activated:

    Loaded: ...; disabled; preset: enabled
    Active: active (running)
    TriggeredBy: ● systemd-oomd.socket

`systemctl disable` stops it starting at boot; the socket relaunches it on demand anyway. The fix is
to mask both, which the user has done:

    sudo systemctl mask --now systemd-oomd.socket systemd-oomd.service

Two things follow. First, `scripts/cap` now bounds the compiler as well as its memory
(`GOFLAGS=-p=4`, `GOMAXPROCS=6`): `go build` defaults to one compile process per CPU, which is 20
here, and holding it to 4 keeps peak RSS so far below the cap that the throttle band is never
entered. Attacking the cause is cheaper than surviving the symptom, and it still holds on a machine
where oomd is live. Second, warpjs-itr2's cap script says `MemoryHigh` is "the missing half" that
prevents oomd killing the desktop. On a box with `ManagedOOMMemoryPressure=kill`, that is backwards:
`MemoryHigh` is what *generates* the pressure. Keep it — a runaway still needs the throttle — but do
not rely on it as the defence.

## ★ RULE 4: no perf floors inside correctness tests.

A throughput assertion next to an NMSE assertion makes the whole test flaky on a thermally-noisy
box, and a flaky test gets skipped — taking the NMSE and disassembly checks with it. Perf floors
live in `dev/bench/`. Correctness lives in `_test.go`. They never share a file.

## ★ RULE 9: fuzz with an exec budget, never a time budget.

    ./scripts/cap 4G -- go test ./convert/gguf -run=XXX -fuzz=FuzzParse -fuzztime=1000000x -parallel=2

With `-fuzztime=60s` the coordinator prints `execs: 8371 (0/sec)` from the third
second onward while the workers sit at 120% CPU and 1.2 GB resident, and then
reports **PASS**. It is not a hang — 1,000,000 executions of the same target
finish in 11 s — the counter is just reported lazily. But a gate that looks
identical whether it ran 8 thousand or 1 million cases is not a gate. `Nx` either
completes the count or does not.

While establishing that, the fuzzer found three real bugs, all of the same shape
— trusting a number in the header before checking the file can back it:
allocations sized from unvalidated counts (it OOM-killed its own cgroup),
`maxTensors`/`maxKV` set 500x past any real model, and 0-dimension tensors
accepted with a nonsense byte size. Size nothing from untrusted input.

## ★ RULE 10: the JIT is EAGER. Lazy emission and a code cache would both be slower.

**★ AND THE TYPE IS `nn.JIT` NOW, NOT `nn.Fast`.** "Fast" named a PROPERTY -- fast compared to
what? -- where the thing itself is the generated-code tier, and generating its own kernels is what
this engine IS. `NewJIT` builds them; `nn.Device` had already taken "accelerator" for the GPU tier,
so `NewAccel` would have named the wrong thing. `JITLLM_DISABLE_FAST` still works alongside
`JITLLM_DISABLE_JIT`: an env var in scripts and muscle memory is a public interface, and renaming it to
match an internal rename breaks other people's commands for nothing.

**★ THE WORD "AOT" USED TO HEAD THIS RULE AND IT WAS WRONG, IN THE WAY THAT MATTERS.** jitllm emits
machine code at RUNTIME, in-process, from shapes it does not know until `model.Open` returns. That is
a JIT by any definition; jitllm is an inference engine that generates its own kernels, not a compiler
that ships an artifact. What this rule actually establishes is that emission is EAGER rather than
lazy, and eager-versus-lazy is a scheduling choice inside a JIT, not the JIT/AOT axis. The mislabel
survived long enough to be quoted back as "jitllm is AOT" in this session.

Measured on gemma-2b, not asserted:

    model.Open (mmap + parse + norms):  29.445 ms
    NewState (allocs + ALL codegen):       316 us
      codegen alone, 8 kernels, 4617 B:     18 us
      mmap+mprotect for those kernels:      36 us

Generating every kernel a model needs costs **0.06% of load time**. Lazy
generation would save at most 18 us while adding a branch to every matvec and a
first-use stall inside a kernel call. An on-disk cache would be worse than
useless: `mmap`+`mprotect` alone costs twice what regenerating from scratch does,
and that cost is unavoidable either way.

Kernel shapes come from the GGUF header, so nothing is speculative, so there is
no deopt, no tier-up and no cache invalidation.

**And codegen does NOT grow with model size**, because kernels are keyed by
(type, blocks-per-row) rather than by tensor or by layer:

    gemma-2b     164 tensors, 127 matvec,  3 distinct shapes -> 6 kernels, 12 us
    tinyllama    201 tensors,  67 matvec,  5 distinct shapes -> 5 kernels, 16 us
    qwen3-1.7B   310 tensors, 197 matvec,  4 distinct shapes -> 4 kernels, 10 us

qwen3 has nearly twice gemma's tensors and generates LESS code in LESS time. The
count is bounded by the distinct roles in a transformer block (q/k/v/o/gate/up/
down times quant type), and those repeat identically across every layer. The same
argument covers mixture-of-experts: all experts in a layer share their shapes, so
128 experts cost what one does. Unverified on a real MoE -- worth checking before
relying on it.

## ★ RULE 11: the bottleneck was DISPATCH, not the kernel. Measure the harness before the code it runs.

jitllm sat at 43% of the memory wall on tinyllama while llama.cpp reached 65%. Three separate analyses
concluded the k-quant kernels were the gap and proposed rewriting them. All three were wrong, and
the measurement that showed it took twenty minutes:

    cores   1      2      4      6
    tok/s  16.9   29.8   46.4   44.6     <- SIX CORES WAS SLOWER THAN FOUR

Fitting `T(n) = W/n + B*R` to 1/2/4 gives ~51 ms of parallel work and **7.9 ms of fixed per-token
overhead** that does not shrink with cores. That is 37% of a 21 ms token. `sched.Pool.Do` was N
channel sends to parked goroutines plus N receives, and one empty round trip measured:

    workers        1        2        4        6
    channel     4.8us    6.6us   15.5us   19.9us     <- x ~300 regions/token = 6 ms
    seq + spin  0.07us   0.58us   1.6us    2.1us

Decode is dispatch-DENSE: 7 matvecs x 22 layers, each with a quantize region and a kernel region,
plus lm_head and the elementwise ops, is ~300 regions per token. **The design document (ARCHITECTURE.md
§6, since retired) quoted 620 ns for a spin barrier, but that was a C probe of a barrier that
was never built** -- the shipped dispatch had never been measured at all.

Two changes, both in `sched/pool_linux.go`:

1. **A job is published by bumping a sequence number and workers spin on it.** The channel survives
   only as the parking fallback. The park handshake is Dekker's -- worker stores `parked` then
   re-reads `seq`, `Do` bumps `seq` then reads `parked` -- which is safe because Go's `sync/atomic`
   is sequentially consistent.
2. **The caller is worker 0.** `len(cpus)` participants means `len(cpus)-1` goroutines. Spawning one
   worker per CPU and having the caller block makes N+1 threads compete for N cores, and rule 5
   already records what that costs.

**Paired, interleaved, alternating binaries:** tinyllama **1.3675x at IQR/median 0.036**, n=12,
48.73 -> 66.55 tok/s. gemma **1.1043x at IQR 0.056**, n=10.

**★ 11a. SPINNING LONGER IS NOT MONOTONICALLY BETTER.** The budget sweep, end to end:

    never spin  39.5      1 ms   63.4
      50 us     57.1      2 ms   64.4
     250 us     71.7      5 ms   58.5
                         20 ms   46.7

Regions arrive every ~50 us, yet a 50 us budget captures only half the win -- the gaps that matter
are the long ones, across a layer boundary and between tokens where sampling over 32000 logits runs.
A budget must cover the TAIL of the gap distribution, not its mean. But past ~1 ms it gets steadily
worse, back to roughly the parked number at 20 ms: a worker that never parks is still spinning
through prefill and through those same serial stretches, competing with the caller for the Go
scheduler instead of idling out of its way. **Parking between tokens is not a cost to minimise; past
a point it is what makes the serial path fast.** Default 250 us; `JITLLM_SPIN_US` overrides.

**★ 11b. THE PROFILER HAD A BLIND SPOT THAT HID ALL OF THIS.** `JIT.MatVec` has three successful
exits and only one recorded the region split. The pack path and the `code4 == nil` path both
returned early -- and those are the paths that ship: gemma takes pack, and tinyllama is 100%
k-quant, gets `BestPack=1`, has no interleaved kernel, and so takes `code4 == nil`. **`JITLLM_PROFILE=1`
had been reporting a blank breakdown for both models since pack landed.** Fixed by routing all three
through one `done()` helper. It immediately said `matvec 93.4%`, which is what made the core-scaling
curve the obvious next measurement.

**★ 11c. THE PHYSICS GUARD CANNOT WORK UNDER `-race` AND IS NOW DISABLED THERE.** The race detector
instruments Go and not generated machine code, so `MemWall` reads ~a third of normal while the
kernel runs at full speed; the guard then compares an uninstrumented result against an instrumented
ceiling and panics on a correct number. This predates the dispatch work (HEAD fails identically at
33.1 vs 16.1 GB/s) and it made `go test -race ./...` impossible to run clean -- which is exactly what
a lock-free dispatch handshake needs. See `dev/bench/guard_race.go`.

**The general lesson: the harness is code too.** Three analyses modelled instructions per weight byte
and none measured the thing between the kernels. When a number sits far from every ceiling you can
name, the missing time is usually not in the code you have been reading.


## ★ A PROCESS-GLOBAL MUTEX ON THE NORM DISPATCH PATH, FOUND BY READING SOMEONE ELSE'S BUG

Aman Chadha's llama.cpp contribution (issue #27078, PR #27891) traced
an Apple Silicon token-generation regression to a KleidiAI *identity check* on
the dispatch path:

    op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type()

which looks like a pointer compare and is not -- the right-hand call reaches
`init_kleidiai_context()`, which enters a global critical section EVERY time, not
only the first. Answering "is this buffer mine?" took a mutex, per op, per token.
The fix was two lines (`buft->context == this`) and the reported effect on the
author's M5 was ~53 -> ~220 tok/s with `__psynch_mutexwait` samples going
5,703 -> 0.

**★ jitllm HAD THE SAME SHAPE, AND `engine/model/prefill.go` PUT IT UNDER THE POOL.**
`RMSNorm32JIT` and `LayerNorm32JIT` took `norms.mu` / `lns.mu` -- process-global
-- on EVERY call, to answer "do I have a kernel for width n?". The struct's own
comment says the answer stops changing almost immediately: *"this is two or three
kernels for the life of the process"*. And prefill calls it from inside
`Parallel(ntok, ...)`, so every pool worker serialised on one lock once per
token, per norm, per layer.

    the same two map lookups, this box, six P-cores
      threads     1      4      6     10
      mutex    16.6   35.6   48.7   65.6 ns/op    <- ANTI-SCALING
      atomic   10.4    2.7    4.3    6.2 ns/op

**★ AND IT IS WORTH 0.09%, WHICH IS WHY IT IS RECORDED HERE AND NOT QUOTED
ANYWHERE.** A tinyllama prefill at n=507 makes 22,815 of these calls (22 layers x
2 norms x 507, plus the output norm): 1.11 ms serialised against 0.10 ms
snapshotted, so ~1.0 ms of an ~1100 ms prefill. Decode is worse per call and
far fewer of them -- qwen3's per-head q/k norms are 953 calls a token, 0.016 ms,
0.06%. **Both are below what `scripts/vs-llamacpp.sh` resolves**, so RULE 6
applies exactly as written: build it if it is cheap and correct, do not put a
perf claim on it.

What makes it worth building anyway is the SHAPE rather than the size: a cost
that GROWS with the thread count, on a path whose call count GROWS with the
prompt. Both curves point the wrong way, and the fix is a copy-on-write snapshot
behind `atomic.Pointer` -- readers take no lock, writers copy two three-entry
maps a handful of times per process.

★ THE FAILURE MODE MOVED, SO THE GATE DID TOO. A reader without the lock turns a
"stale map entry" into a CONCURRENT MAP ACCESS, which is a crash rather than a
wrong number. `TestNormDispatchIsConcurrentSafe` drives both dispatchers from 16
goroutines over five widths and two eps values so the snapshot is genuinely
rewritten while others read it; under `-race`, mutating the live map instead of
copying fails it with `WARNING: DATA RACE`.

★ WHERE ELSE TO LOOK, since the pattern is what generalises: any per-call lookup
that answers a question fixed at warm-up. `engine/nn/jit.go`'s `packMu` (the generated
packer cache) and the `gtile`/`gtune` tuner maps are the same shape, but they are
consulted per MATMUL -- 2002 calls in the same prefill against 22,815 -- so they
are two orders of magnitude cheaper and have not been changed.

## ★ THE GO COLLECTOR WAS 42% OF THE ENGINE'S CPU AND NOTHING REPORTED IT.

`top` showed `jitllm` at over 400% CPU on a run that was mostly waiting for
disk, which reads as work. It was not. A CPU profile of Qwen3-Next-80B under
`scripts/cap 12G`, `-devices gpu:0`:

    runtime.gcBgMarkWorker          42.5%
    sched.(*Pool).await             22.0%   -- workers SPINNING
    os.(*File).ReadAt -> Syscall6   13.2%   -- the pager
    runtime._ExternalCode            5.0%   -- the generated kernels

**Nine tenths of the CPU was not inference.** `GODEBUG=gctrace=1` names it:

    gc 512 @54.445s 12%: ... 9956->9956->9867 MB, 9956 MB goal
    gc 513 @54.450s 12%: ... 9867->9867->9867 MB, 9867 MB goal
    ...

**The live heap IS the goal**, so every allocation triggers a cycle that can
free nothing: 841 cycles in a two-token run.

### Why the budget put it there

Page frames are permanently live. `sched.MemBudget` takes eight tenths of the
cgroup and `cmd/jitllm` raises that to **nine tenths** when the container opens
O_DIRECT -- correctly, for the reason it states: direct reads leave no page-cache
shadow, so the conservative fraction throws away real headroom. That reasoning
is about RSS and says nothing about the collector's goal, which under a memory
limit is `limit - non-heap`. Nine tenths of a 10.79 GiB ceiling is 9.71 GiB of
live frames against a ~9.86 GiB goal.

### The dose-response, and the cliff is sharp

Qwen3-Next-80B, two tokens, one run per point, the same binary:

| weight budget | GC cycles | decode |
|---|---|---|
| 6.10 GiB | 5 | 0.56 tok/s |
| 7.04 | 6 | 0.61 |
| 7.97 | 6 | 0.60 |
| **8.34** | **6** | 0.49 |
| **8.99** | **648** | 0.42 |
| 9.30 (old default) | 458 | 0.27 |
| 9.55 | 991 | 0.57 |

**8.34 GiB is quiet and 8.99 GiB is 648 cycles.** There is no gradient to climb,
only a line to stay under -- which is why the fix is a RESERVE and not a tuning
curve. `sched.GCBudgetCap` leaves the collector a quarter of its usable heap
(capped at 8 GiB so a margin sized for a laptop is not charged to a server), and
**every** path that sizes the weight budget passes through it, including the
O_DIRECT raise that was the actual culprit.

    before   budget 9.71 GiB   860 cycles   sys 116.71s   312% CPU   80.4s wall
    after    budget 7.34 GiB     6 cycles   sys   5.10s   164% CPU   28.9s wall

System time falls 23x. `sched.SetGCReserve` and `sched.SetGCPercent` are the
knobs (`JITLLM_GC_RESERVE`, `JITLLM_GOGC`), and the `pages` line now prints what
is left for the Go heap, because a live heap sitting on its goal is invisible
from outside and looks exactly like work.

★ **AND GOGC IS NOT THE FIX, WHICH IS WORTH SAYING BEFORE SOMEONE REACHES FOR
IT.** Under a memory limit the goal is `min(live*(1+p/100), limit-nonheap)`, so
once live is at the limit no value of p moves it. The reserve keeps live off the
limit; GOGC tunes the collector once it is.

### What was left: the pool spinning through a blocking read

After the GC fix the profile is 68% `Pool.await`. A model that FITS wants the
spin -- a parked worker pays a wakeup at the head of every region, which is what
RULE 5's rows are taken with. A model that is OVER BUDGET reads a block page off
disk on the critical path of nearly every layer, and six workers spinning
through it is six cores of heat.

**The rule is per token, not per model.** `model.State.forward` parks the pool
for the next token when this one faulted a page, and spins when it read
nothing (`sched.Pool.SetSpinning`). The first rule parked whenever
`WeightBytes() > maxmem` -- but an over-budget mixture reads nothing on most
tokens once its working set is resident, and parked workers then paid a wakeup
at every region for no read at all. Qwen3-30B-A3B Q4_K_M, Xeon E5-2680 v4 CPU,
one socket's fourteen cores, n=64, alternated one process per sample, eight
rounds, A/A first (0.997 prompt / 0.998 decode, IQR/median <= 0.016), two
passes, size rule (A) against the per-token rule (B):

    cap     prompt B/A        decode B/A        bytes read, both arms
    16 GiB  1.416 / 1.414     1.260 / 1.260     11.00 GiB (warm-up only)
    12 GiB  1.463 / 1.459     1.128 / 1.130     12.12 GiB (tokens page)

Every arm IQR/median <= 0.016. At 12 GiB tokens still page, so fewer of them
spin and the decode gain is smaller. `model.TestThePoolParksOnlyWhileTokensPage`
holds the rule: park after a faulting token, spin after a quiet one.

`State.forward`'s comment carried the size rule's cost on one run: "Deciding
by the model's size against the budget parked Qwen3-30B in a 16 GiB cgroup
for a run that read nothing after warm-up: 7.8 tok/s against 20.4 spinning,
on a Xeon."

The size rule's own row, kept because it is what the first version rested on.
Alternated 250us/park, three rounds, on a box at 7-16% busy:

    round   decode 250us / park     our user CPU 250us / park
      1      6.693s / 6.723s           7.62s /  5.07s
      2     18.973s / 9.371s          33.66s / 11.88s
      3      6.904s / 6.443s           9.19s /  5.16s

**The rate is UNRESOLVED and is not claimed**: per-round ratios 1.00, 2.02, 1.07
is a median of 1.07 with a dispersion RULE 2 rejects. Our own user CPU falls
1.5x, 2.8x and 1.8x, same sign every round, and that is a counter rather than a
stopwatch -- it measures this process and a foreign tenant cannot add to it. It
lands on RULE 6's terms: CPU improved, rate unresolved, nothing regressed.

### ★ AND weak.Pointer IS THE RIGHT SHAPE AND THE WRONG MECHANISM, MEASURED

The suggestion is a good one -- a resident page is reconstructible from the
container, so losing one costs a re-read rather than correctness, which is
exactly what a weak reference is for. Go's weak pointers do not have the
semantics a cache needs. 64 weakly-held 4 MiB buffers, one `runtime.GC()`:

    before GC      1 of 64 alive
    after 1 GC     0 of 64 alive

**All of them, on one cycle, with no ordering and no pressure model.** There is
no LRU and no partial eviction: a page cache built on this loses its whole
working set every collection, so the thing that made the spiral expensive --
cycle frequency -- would become catastrophic instead of merely wasteful. The
pager's own policy is Belady-optimal over a cyclic scan (`sched.NextUse`) and it
cannot be expressed to the collector. That is what trying it measured.

## ★ AND THE LEAK GATE'S 21 MiB PER CYCLE WAS RSS, NOT A LEAK.

`model.TestConcurrentMultiDeviceDoesNotLeak` asserted on RSS after a scavenge
and flipped run to run -- fail, fail, pass on one binary. Its own comment had
already swept the round count (4: +85,796 KiB, 8: +180,084, 16: +242,080), read
it as linear, concluded *"it is device BUFFERS"*, refused to move the bar, and
named the fix without making it: *"What this gate needs is an EXACT counter...
the tier already keeps an integer ledger of device bytes. Asserting on that
instead of on RSS is the change, and it is not made here because it is a gate
rewrite rather than a bar."*

Both counters exist and both are flat:

    live heap after runtime.GC   504 -> 536 KiB over 4 cycles   (+32 KiB)
    device bytes after Close     back to the placement figure, every cycle
    RSS after FreeOSMemory       106,812 -> 215,624 KiB         (+108 MiB)

So there is no leak, in Go or on the card, and RSS is the scavenger's opinion
about returning pages to the kernel. `HeapAlloc` taken immediately after
`runtime.GC` is what a leak IS; the tier's integer is what a device leak is.
Neither needs a sleep and neither moves under a foreign tenant. Three greens
where there were two reds, and against a violation -- one State left open --
the ledger reads the bytes back by name.

★ **THE STANDING LESSON.** A gate that measures RSS is measuring the allocator.
Both of this session's memory findings are the same shape from opposite sides:
one was a real cost that no counter reported, the other a reported cost that was
not real.

## A llama.cpp build with no GPU backend accepts `-ngl` and ignores it

Moved from ARCHITECTURE.md §5a-bis-bis when that file was retired. The build paths
themselves are listed in `docs/design/model-coverage.md` ("The oracles that exist on this box").

Benchmarking the 70B against "llama.cpp" picked the CPU-only build in the
llama.cpp builds directory. Its banner prints `loaded CPU backend from
libggml-cpu-alderlake.so` and nothing else, so `-ngl 10` changed nothing and the row would have been
reported as a GPU number that never touched the GPU. Only the `load_backend:` lines tell the builds
apart. A second wrong turn followed: llama.cpp publishes no Linux CUDA binary (checked b10825 and
b10946-b10948: ubuntu gets vulkan, rocm, sycl and openvino; CUDA is Windows-only), which is true and
was irrelevant, because a CUDA build of llama.cpp was already on the box and
`scripts/vs-llamacpp.sh` takes it from `$JITLLM_LCPP`.
The harness already knew; the filesystem search did not.

## ggml falls back to a static split, which is what jitllm's pool never does

From the design survey, moved from ARCHITECTURE.md §6 and re-checked against
`ggml/src/ggml-cpu/ggml-cpu.c` in a llama.cpp source checkout at the
same time. When `nchunk0 * nchunk1 < nth * 4` (or on NUMA), ggml's matmul abandons its chunk queue and gives
each thread one equal slice of rows. tinyllama's GQA `attn_k`/`attn_v` (256 rows, four chunks) hit that
fallback at every thread count. On a hybrid CPU an equal slice is wrong by a kernel-dependent factor
-- the design probes measured the P:E throughput ratio at 5.1x for VNNI int8, 2.6x for f32 FMA and
1.76x for the memory-bound decode matvec -- which is why `sched.Pool.drain` claims chunks dynamically
and has no static path (its comment quotes the first and last).

## The decode row's two boundaries: `jitllm run` against llama-bench's tg

A CUDA board read `jitllm speed`'s warm decode faster than llama-bench (A100,
Llama-3.2-1B Q4_K_M: 1,615 against 1,854 us a token) while the harness's
decode row, which times `jitllm run`, read 0.98-1.00. What each arm times:

- **llama-bench tg** creates the context untimed, runs one untimed warm-up
  generation, then times its n tokens from a fresh KV at position 0.
- **`jitllm run`** (vs-llamacpp.sh's decode arm) opens, places, prefills the
  prompt, and times every token from the first decode: the decode timer
  starts after the prompt, so open, placement, upload and the prompt are
  outside it, and the first decode token of the session is inside it.

Measured per token from `run` (a timer around each step of its decode loop,
a local build; it is not in the tree), Llama-3.2-1B Q4_K_M unless said:

    V100, cuda      token 1: 3.32-3.52 ms; tokens 2-7: 1.83-1.90; median after 8: 1.842-1.862
                    jitllm speed tg128 on the same card: 1.83-1.86 ms (538-546 tok/s)
    T4, cuda        token 1: 5.54-5.86 ms; tokens 2-7: 4.57-4.94; median after 8: 4.16-4.39
                    jitllm speed tg128: 4.05-4.35 ms
    T4, 8B Q4_K_M   token 1: 22.2-22.9 ms; tokens 2-7: 20.6-20.9; median 20.86-20.89
                    jitllm speed tg128: 20.9-22.5 ms

So on both cards `run` pays one thing per session that `speed` does not: its
first decode token is 1.5 ms (V100) to 1.7 ms (T4) longer, which is the
session's decode graph being captured (a recording is keyed by the session,
graphKey.sid, and speed's untimed warm-up captured it). Over 128 tokens that
is 0.5-0.7% of the row. After it, `run` and `speed` decode at the same rate
within 1%. Nothing else is paid per token: the printing goroutine, the
sampler and the greedy step are off the device's path at these rates.

Two start-up costs found on the way, neither per token:

- **A cold tune cache tunes only part of the decode in a `run` process.** The
  background split tuner (tier.WithLazyTune) times a trial only after 50 ms of
  device quiet, and a `run` has that only before its prompt: on a V100 with an
  empty cache one `run` tuned 3 of the 1B's 6 decode shapes and a `speed`
  process all 6. The rest run the table's split until a later process tunes
  them. On the V100 the table's split decoded at the tuned rate (1,846 us a
  token both ways); a card the table was not fitted on can differ.
- **The first prompt compiles one kernel after the model is ready**
  (`jitllm speed -ttft` refuses the row and names it): a staged prefill
  attention variant (pagedscoresmma) at the first prompt's span, compiled when
  a chunk first needs it rather than at placement (V100, -p 16 and -p 514).

The A100 gap (run about 0.3 ms a token slower than speed) is not reproduced on
a V100 or a T4, where the gap after the first token is under 0.1 ms. It is
open: what settles it is the same per-token timer from `run` on the A100, one
short job.

## The 0.90 and 0.70 readings, and the harness that produced them

★★★ **Llama-3.2-1B ON LINUX CPU IS A GATED WIN AT 1.0425 / 1.0436, TWO PASSES
-- AND THE "0.90 REGRESSION" WAS THE MEASUREMENT.** The row this
entry spent a day failing to take:

| | ratio | IQR/median | n | | clamped | incoherent |
|---|---|---|---|---|---|---|
| pass 1 | **1.0425** | **0.012** | 13 | **WINS** | 2 of 32 | 17 pairs |
| pass 2 | **1.0436** | 0.020 | 15 | **WINS** | 3 of 32 | 14 pairs |

Two passes agreeing to **0.11 points**, both far inside the gate, neither
showing a clock trend (r = -0.04 and +0.20 over 1771-1938 and 1750-2012 MHz), so
no adjustment applies and the plain median is the statistic. Llama-3.2-1B-Instruct
Q4_K_M, CPU decode, n=320, six P-cores, 16 rounds, `JITLLM_SOAK=10`.

**1.0425 sits just ABOVE container v6's own 1.0037, so there was never a
regression to attribute** -- which is what the nine deterministic counters below
already said and what two days of timing could not.

★★ **AND olmoe IS 0.92, NOT 0.70, ON THE SAME RECIPE.** Two passes,
olmoe-1b-7b-0924-instruct Q4_K_M, n=320, six P-cores, 16 rounds:

| | ratio | IQR/median | n | |
|---|---|---|---|---|
| pass 1 | **0.9165** | 0.013 | 13 | loses |
| pass 2 | **0.9322** | 0.036 | 8 | loses |

Agreeing to 1.6 points, both gating. **So both numbers this goal was set on were
the measurement**: 0.90 was really 1.043 and 0.70 was really 0.92. olmoe does
genuinely lose -- consistent with its 59% of the read wall against a dense
model's 85% -- but by 7 points, not 30.

★ **AND PASS 1's CLOCK-TREND FLAG WAS REFUTED BY ITS OWN ADJUSTMENT, WHICH IS
WHY THE FLAG IS NOT A VERDICT.** It fired at r=+0.54 over 13 points; removing
that line took IQR/median from 0.013 to **0.064**, five times WORSE. A trend
that is real tightens the residual -- subtracting one that is not injects
variance. Pass 2 says the same from the other side at r=+0.01. `scripts/clockadj.py`
prints that comparison, and `vs-llamacpp.sh` now points at it rather than
prescribing the frequency pin: a correlation is a suspicion, not a verdict.

★★ **AND WHAT FIXED THE MEASUREMENT WAS MUNDANE: A QUIET BOX AND A LONGER SOAK.
IT DID NOT NEED THE FREQUENCY PIN, AND SAYING IT DID WAS WRONG TWICE.** The
winning runs held 1700-2100 MHz for their whole length -- the sustained band
this file already records -- where every earlier attempt wandered 800-2338. Two
causes, and the first is self-inflicted:

    a SUBAGENT COMPILING GO IN THE SAME TREE during the rows taken earlier that
    day. RULE 3 item 3 names it exactly: another job is invisible to every gate here
    and lands mid-round. `scripts/cap` serialises measurements against each
    other and does NOT stop a build from heating the package.

    STARTING COOL. A run begun at 61 C spends itself climbing the ramp and
    samples every frequency on the way; one begun already soaked at 68 C with
    ten soak iterations never leaves steady state. RULE 2 says soak rather than
    cool, and the failure mode when it is skimped is not noise -- it is a
    reading of the ramp.

**So the sequence that produces a row on this box is: nothing else running,
JITLLM_SOAK=10, 16 rounds.** At six rounds the coherence gate leaves too few
pairs; at sixteen it leaves 13-15 of 32, which is n enough.

★★★ **AND THE COHERENCE GATE IS UNDER SUSPICION BY ITS OWN DATA, TAKEN
THE DAY IT WAS ADDED. IT MAY BE DISCARDING AN ENGINE DIFFERENCE RATHER THAN A
CLOCK ACCIDENT.** Run over the 32 pairs of one real pass:

    no gates at all                ratio 1.0644  IQR/med 0.037  n=32
    clamp gate only                ratio 1.0593  IQR/med 0.044  n=27
    clamp + coherence (shipping)   ratio 1.0536  IQR/med 0.042  n= 9

**It discards 23 of 32 pairs and moves the ratio by 0.6%.** That is only a good
trade if what it removes is NOISE, and the same data says it is not:

    jitllm's sampled MHz exceeds llama.cpp's in **27 of 32 pairs**
    median delta +159 MHz, median ratio f/g **1.080**
    the gate's threshold is **1.05** -- BELOW the systematic median

A threshold sitting under the systematic separation of the two arms does not
remove outliers; it removes the majority and keeps the minority in which the
pattern happened not to hold (5 of 9 survivors show it, against 27 of 32
overall). The discarded ratios are 0.96-1.09 -- clustered, not wild.

★ **THE MECHANISM IS NOT ESTABLISHED, AND THE PROBE I REACHED FOR WAS
CONFOUNDED.** The hypothesis is that `mhz_now()` averages over ALL CPUs while
jitllm's pool SPINS for 250 us (this file records 60e9 instructions of workers
WAITING), so jitllm holds its cores' clocks up where llama.cpp's threads park. A
quick probe read jitllm 3018 MHz against llama.cpp 1274 on the same taskset --
but it sampled two seconds in, which for `llama-bench` is during MODEL LOAD with
the cores idle. It measures when I looked, not how they run.

★★★ **THE SAMPLER WAS FIXED AND THE HYPOTHESIS IS REFUTED AS THE MAIN CAUSE.**
Sampling only the taskset cores is built (`mhz_now`, and the same selection out
of /proc/cpuinfo where a host has no cpufreq sysfs -- the Xeon has none, and a
per-core sysfs read there silently fell back to the package average, defeating
the fix). The package average really is a different number from the measured
cores', in OPPOSITE directions on the two boxes: laptop 1183 MHz package against
1452 on its six, Xeon 1475 against 1350 on its four.

**It improved the measurement and did NOT explain the discarding**, which is the
half worth keeping. olmoe on the Xeon, two passes, only the sampler changed:

    before   0.8706  IQR 0.019  n=8     0.8694  IQR 0.008  n=5  REJECTED
    after    0.8743  IQR 0.004  n=9     0.8746  IQR 0.009  n=7  both GATE

Median held to 0.4 points (so the old gate was not deleting an effect),
dispersion fell 0.019 -> 0.004, the passes went from agreeing at 0.14 points to
0.03 -- and **n barely moved: 11 and 12 pairs still discarded, against 12 and
15.** So idle-core contamination was not what made the arms' clocks disagree.

★ **WHAT IS LEFT IS A CANDIDATE, NAMED RATHER THAN ASSUMED.** The two engines
may genuinely drive the clock differently ON A MIXTURE: `ensureExperts` does
synchronous preads on the critical path -- 6.1% of samples by the profiler --
where llama.cpp mmaps and faults at 4 KiB. That would make 1.05 too tight for
MoE models specifically rather than wrong in general, and it predicts the dense
model should discard far fewer, which it does (4 and 10 against 11 and 12).
Unmeasured. **The gate stays on meanwhile**: a too-aggressive gate refuses rows,
a too-permissive one publishes them.

★ **THE EARLIER "THIS BOX CANNOT PRODUCE A ROW" CONCLUSION DID NOT HOLD.** It
was stated twice, each time from runs that had one of the two faults above. The
frequency pin (`min_perf_pct = max_perf_pct`) would make this cheaper and more
repeatable and is still worth having; it is not required, and
`scripts/clockadj.py` -- written for the case where the ratio tracks the clock
-- turned out not to be needed here. It is kept as insurance: it refuses to
adjust below |r| = 0.5 and refuses to extrapolate outside the measured range.

★★★ **THE SUPERSEDED READING, KEPT BECAUSE THE WAY IT WENT WRONG IS THE
LESSON.** The rows below are kept in full because the WAY this went wrong is the
lesson: every quoted arm GATED, both endpoints REPRODUCED, and the conclusion was
still an artefact of a box that cannot resolve five points. Measured with
each binary against llama.cpp in its OWN interleaved pass:

| arm | ratio | IQR/median | |
|---|---|---|---|
| HEAD | 0.9482 | 0.168 | REJECTED |
| **container v6 (0406c41)** | **1.0067** | 0.049 | **WIN** |
| **HEAD** | **0.9170** | 0.026 | **loses** |
| **container v6** | **1.0147** | 0.018 | **WIN** |

Both v6 arms GATE and both WIN, agreeing to 0.8 points; the gated HEAD arm loses
by nine, and v6's 1.0067/1.0147 reproduces its own earlier 1.0037. Four
gates and a reproduction, and the answer is still wrong.

★ **THE IN-PROCESS INTERLEAVED HARNESS THIS ENTRY ASKED FOR WAS BUILT, AND IT
REVERSES THE SIGN.** `model.TestFusionCeiling` already decodes 64 warm tokens in
process and reports a rate, and BOTH trees carry it unchanged -- so it is one
test binary per tree, ABBA, A/A self-control FIRST (RULE 2), Llama-3.2-1B
Q4_K_M, six P-cores, each binary on its own container:

    A/A self-control (HEAD in BOTH arms)   0.9948   IQR/median 0.097
    pass 1   HEAD / v6                     1.0481   IQR/median 0.087   GATES
    pass 2   HEAD / v6                     1.0435   IQR/median 0.141   REJECTED
    pooled median-of-samples, n=24/arm     1.059 raw, 1.048 bias-corrected

**HEAD is ~5% FASTER than v6.** The two passes agree to 0.5 points and the A/A
bias is 1.1%, and the number is RULE 5's pinning removal (+3.8%, the only
decode-relevant pool change in the range) arriving from an independent harness.

★ **AND ON THE SAME DAY BOTH PROCESS-PER-SAMPLE HARNESSES REFUSED TO PRODUCE A
ROW AT ALL**, which is the measurement that says the 0.9170/1.0147 pair was never
resolvable:

    scripts/vs-llamacpp.sh, HEAD    0.9383  IQR/median 0.148  REJECTED,
                                    4 samples power-clamped, cores 794-2196 MHz
    direct A/B, A/A SELF-CONTROL    1.09    samples from 16.4 to 56.3 tok/s

**A self-control of 1.09 IS the effect being hunted.** What separates the harness
that gates from the two that do not is process startup and the container read:
the in-process probe warms eight tokens and measures sixty-four more, so it
prices the DECODE LOOP and nothing else.

★★ **AND THE COUNTER I SWITCHED TO WHEN TIMING FAILED IS A TRAP: `perf stat`
INSTRUCTIONS ARE NOT A WORK MEASURE ABOVE ONE THREAD, BECAUSE THE POOL SPINS.**
At n=320 on six cores HEAD read 229.3e9 and 285.6e9 instructions against v6's
186.6e9 and 211.8e9 -- which reads as a deterministic, thermally invariant,
depth-dependent 23-35% more work, and is the 250 us spin budget:

    six cores, n=320    spin=250us (default)      spin=0
    HEAD                189.1e9 instructions      127.7e9
    v6                  186.0e9                   160.7e9

Sixty billion of HEAD's instructions are workers WAITING, and a spinning worker's
count scales with how long the region took -- so on a clamped box the counter
tracks TEMPERATURE, which is the one thing it was adopted to be immune to. HEAD's
own count swinging 229e9 -> 286e9 between two runs was the tell, and I read it as
"variable work per token" instead.

★ **ONE THREAD IS THE INVARIANT MEASUREMENT AND IT IS THE STRONGEST EVIDENCE
HERE.** At `JITLLM_CORES=1` there are no workers and no spin, so the count is the
work:

    JITLLM_CORES=1, n=320, two runs each
    HEAD   110,849,406,673 and 110,866,108,794 instr   25.91, 25.72 tok/s
    v6     110,744,329,381 and 110,765,973,656         25.73, 25.64 tok/s

**0.09% apart in instructions and inside noise on rate.** Whatever the 44 commits
did, they did not change what a token costs.

★ **THE f16 KV CACHE IS NOT THE CAUSE, WHICH IS THE FIRST THING TO CHECK BECAUSE
IT IS A DEFAULT THAT FLIPPED IN THAT RANGE AND ITS amd64 ROW WAS NEVER TAKEN.**
ABBA, same session: f16 ON 0.8498 / 0.9160, f16 OFF 0.9021 / 0.8667. Identical
means, and the WITHIN-ARM spread (7 points) is as large as the effect being
hunted -- so this is a refutation of the hypothesis AND a statement that the box
can no longer resolve one.

★★★ **AND THE ATTRIBUTION WAS ATTEMPTED TWICE AND FAILED TWICE, EACH TIME
NAMING A COMMIT THAT ADDS NO EXECUTABLE CODE. THE TECHNIQUE THAT CAUGHT BOTH IS
THE THING TO KEEP: A DOCS-ONLY COMMIT IS THE HARNESS'S NULL CONTROL.**

    method              named            what that commit actually changes
    single probe        229729c  0.8866  AGENTS.md + one _test.go. NOTHING ELSE.
                                         Its predecessor 0287c0e -- byte-identical
                                         shipping code -- read 0.9853.
    ALTERNATED A/B      68d951e  0.926   comments in pack.go + AGENTS.md. Zero
                                         executable lines. Reference arms BOTH
                                         gated, 0.9799 and 0.9883.

Both readings GATED or near it, and both are impossible. **A bisect step on this
box must be validated against `git show --stat`, and any range should be probed
at a comment-only commit FIRST**: if that reads different from its neighbour, the
harness cannot resolve the effect and every other row in the run is noise.

★ **WHY INTERLEAVING IS NOT ENOUGH, WHICH IS THE PART THAT SURPRISED ME.**
`vs-llamacpp.sh` alternates the two ENGINES inside one run, so a drifting box
should cancel -- except this file already records that **the two engines do not
shed rate to a clamp equally** (the M4 prefill row: jitllm -7.4%, llama.cpp
-4.2%). The RATIO is therefore itself a function of temperature, and two runs
taken twenty minutes apart are not comparable however tightly each one gates.
The same binary bears it out: f35b1fd measured **1.0099, 1.0095, 0.9799, 0.9883**
across four sessions. Only a gap measured WITHIN one alternating session means
anything, and on this box not even that is reliable below ~6 points.

★ **"WHAT SURVIVES IS THE ENDPOINT PAIR" IS THE SENTENCE THAT WAS WRONG, AND IT
SURVIVED BECAUSE REPRODUCING A NUMBER IS NOT THE SAME AS RESOLVING ONE.** v6 won
twice in one session (1.0067, 1.0147) against a gated HEAD at 0.9170 and
reproduced at 1.0095 in a second -- all true, all gated, and all taken with a
harness whose A/A self-control reads 1.09. **A reproduction confirms the harness
is CONSISTENT; only a self-control says it is DISCRIMINATING**, and the
self-control had never been run on that harness when those rows were taken.

**NINE DETERMINISTIC CHECKS NOW SAY THE TWO BINARIES ARE THE SAME ENGINE, AND
THE THREE ADDED LAST CLOSE THE DISPATCH QUESTION THE OTHERS LEFT OPEN:**

    container size       858,316,800 B both              identical
    bytes read / token   0.76 GiB both                   identical
    token ids at n=48    identical, all 48
    per-op profile       matvec 96.5% / 96.6%            identical
    one-thread instrs    110.849e9 / 110.744e9           0.09% apart
    one-thread rate      25.91 / 25.73 tok/s             inside noise
    REGIONS PER TOKEN    218.0 / 218.0 parallel, 0 inline  identical
    region round trip    2.519 us / 2.777 us             same order
    the tuner's cache    ~/.cache/jitllm/packwidth UNCHANGED by a run of
                         EITHER binary -- so neither arm poisons the other

Dispatch was the last standing hypothesis and **218.0 regions/token in both trees
retires it**: the coordination is identical, the per-token work is identical, and
the only gated paired rate says HEAD is faster.

★★★ **RE-DERIVED FROM SCRATCH ON A BOX AT 69% FOREIGN LOAD, WHICH IS
THE POINT: NONE OF THESE IS A STOPWATCH THAT NEEDS A RESTED MACHINE.** Both
binaries rebuilt from source (v6 = 0406c41 in /tmp/v6b, HEAD = 0b13c91), each
converting its OWN container from the same GGUF:

    container size       858,316,800 B both                 identical
    token ids at n=48    identical, all 48
    REGIONS PER TOKEN    218.0 / 218.0 parallel, 0.0 inline  identical
    one-thread instrs    v6   111,704,440,088  111,036,211,080   spread 0.60%
                         HEAD 111,185,422,324  111,073,082,209   spread 0.10%
                         between arms 0.22% -- SMALLER than v6's own spread
    one-thread rate      v6 24.95 / 24.69   HEAD 24.59 / 25.00   0.10% apart

★ **AND THAT IS WHAT RETIRES THE 0.90 AS A CODE REGRESSION, BY ARITHMETIC
RATHER THAN BY A ROW.** A ratio of 0.90 requires HEAD to be ~10% slower. At
`JITLLM_CORES=1` there are no workers and no pool spin, so the instruction count
IS the work and the rate IS the engine -- and the two binaries are 0.22% and
0.10% apart. A 10% regression cannot be invisible at one thread and appear only
at six unless it lives in the COORDINATION, and 218.0 regions/token in both
trees closes that door. **The 0.90 was the thermometer.**

★ **THE CONTAINERS ARE THE SAME SIZE AND NOT THE SAME BYTES, WHICH IS EXPECTED
AND WORTH STATING SO NOBODY READS THE SIZE CHECK AS MORE THAN IT IS.** 592.3 MB
of 858.3 MB differ -- the v6 -> v15 d-plane row-pairing changed the LAYOUT at
identical size. Identical token ids out of two different layouts is the stronger
statement: the bytes moved, the decoded weights did not.

★ **WHAT THIS DOES NOT ESTABLISH.** It says the 0.90 is not attributable to a
change in jitllm between v6 and HEAD. It says nothing about where the ratio
against llama.cpp actually sits -- that still needs a rested paired row, and the
in-process harness reading HEAD 61.2 against v6 52.8 tok/s in the same session
is a single unpaired sample on a loaded box and is NOT a result. It is recorded
only because its direction is the opposite of a regression.

★★★ **AND olmoe's 0.70 GETS THE SAME TREATMENT AND A STRONGER ANSWER: HEAD DOES
STRICTLY LESS WORK THAN v6 FOR BYTE-IDENTICAL OUTPUT.** Same method, same box,
olmoe-1b-7b-0924-instruct Q4_K_M:

    container size       4,588,560,384 B both               identical
    token ids at n=48    identical, all 48
    REGIONS PER TOKEN    v6 906.0  ->  HEAD 234.0           **3.87x FEWER**
    one-thread instrs    v6   62,143,057,191  60,888,411,103   spread 2.04%
                         HEAD 57,204,573,995  57,097,439,273   spread 0.19%
                         between arms **7.1% FEWER on HEAD**, well outside both

**A regression executes MORE work, not less.** HEAD removes 672 dispatch regions
and 7.1% of the instructions per token and emits the same 48 tokens, so 0.70
cannot be attributed to a change in this engine either. The instruction drop is
the batched MoE dispatch (`nn.MatVecPackedMulti`/`MatVecPackedGather`) deleting
the k redundant quantizations of the shared activation, and it lands within a
point of the ~10% kernel-CPU fall that change measured directly.

★ **AND THE RATE NOT MOVING IS THE ALREADY-KNOWN olmoe FINDING, NOT A PUZZLE.**
The in-process harness reads HEAD 39.8 against v6 40.5 tok/s -- level, single
unpaired samples on a loaded box, not a result. 7.1% less work and 3.87x less
coordination buying nothing is exactly what the MoE profile entry below records:
olmoe is bound by the SERIAL STRETCH between regions (the router softmax, the
top-k, the sort, `ensureExperts`' synchronous reads), which neither batching nor
kernel work touches. **So olmoe's deficit against a dense model is a standing
property -- 59% of the read wall against 85% -- and not something that changed
between v6 and HEAD.**

★ **AND THAT PREDICTION WAS TESTED AND HELD, WHICH IS WORTH MORE
THAN THE ROW IT PRODUCED.** "Kernel work does not touch it" was written before
anyone generated the top-k. Generating it -- the selection, the ascending sort
and the renormalising divide, on all three tiers -- moved olmoe's decode by
**1.0021 / 1.0064 against a 0.9979 A/A self-control**, all three gating. The
file said the rate would not move and the rate did not move. A prediction that
survives its own experiment is the cheapest evidence in this project, and it is
why the levers here are named with a measurement rather than a story.

★ **THE ONE-THREAD RATES ARE NOT QUOTED FOR olmoe AND THE REASON IS VISIBLE IN
THEM.** v6 read 25.65/25.67 and HEAD 22.34/17.68 -- a 26% within-arm spread on
HEAD against 0.08% on v6, which is the foreign tenant landing on one arm and not
the other. The INSTRUCTION counts from the same four runs are 0.19-2.04% tight,
which is the whole reason this section uses them: on a contended box the counter
is invariant and the stopwatch is not.

★ **AND THE Q4_K-KERNEL CHECK THAT USED TO HEAD THIS LIST IS DELETED, NOT
DEMOTED: IT WAS MEASURING A KERNEL NEITHER BINARY RUNS.** It read *"`jitllm asm
Q4_K` byte-identical -- the only diff in 138 lines is objdump's own temp
filename"*, and `asm` dumps `cpu.Emit(...)`, the GGUF ROW-MAJOR matvec. A
container model decodes through `cpu.EmitPackedMatVecFused`, which that
subcommand never prints. RULE 10's shape in a diagnostic: a confident equality
over a configuration nobody selected.

★★ **BECAUSE SIX CONSECUTIVE IDENTICAL RUNS DRIFT 17% ON HEAT ALONE.** Same
binary, same model, same prompt, n=320, back to back:

    53.73  48.62  47.58  48.28  45.49  44.69  tok/s

Monotonic, and it is the die temperature climbing through the sequence. **Any
A/B whose arms run SEQUENTIALLY is measuring the ramp**, which is why
`scripts/vs-llamacpp.sh` interleaves ABBA -- and interleaving was still not
enough, because a process-per-sample harness spends most of a sample outside the
decode loop. **The prescription in this paragraph was right and it worked: the
in-process interleaved harness is what finally gated**, and it did not need more
rounds (RULE 2).

★★ **AND THE MACHINE IS THERMALLY SICK, WHICH IS A FINDING RATHER THAN AN
EXCUSE.** 95°C against a 100°C limit under six loaded P-cores, with the fans at
4130/3912 RPM of a 5900/5600 maximum; nine minutes of complete idle (PSI 0.00)
only brought it to 76°C, and it re-soaks to 94°C inside ONE comparison however
long it is cooled first. Two full passes of the seven-model board,
with a cooldown between every model:

| model | pass 1 | pass 2 | gated |
|---|---|---|---|
| Qwen3-MoE-4x0.6B | 1.1197 | 1.0584 | 0 of 2 |
| Qwen3-1.7B | 0.9678 | 1.0354 | 1 of 2 |
| Qwen2-1.5B | 0.9617 | 0.9541 | 0 of 2 |
| tinyllama q3_K_M | 0.9031 | 0.8564 | 0 of 2 |
| Llama-3.2-1B | 0.8997 | 0.8961 | 1 of 2 |
| gemma-2b | 0.8161 | 0.7943 | **2 of 2** |
| olmoe-1b-7b | 0.6999 | 0.7028 | 1 of 2 |

**The MEDIANS are reproducible and the DISPERSION is not** -- olmoe agrees to
0.4% across passes and Llama-3.2-1B to 0.4 points, while five of seven rows blew
the IQR gate at least once. So the ordering above is usable as a LEAD and no row
is a standing. **olmoe at 0.70 is the worst row on either host and is next after
the Llama-3.2-1B regression.**

★★★ **AND THE XEON D-2123IT IS THE MEASUREMENT HOST THIS PROJECT HAS BEEN
MISSING, WHICH IS A BIGGER RESULT THAN THE ROW IT PRODUCED.**
Llama-3.2-1B-Instruct Q4_K_M, CPU decode, n=128, 4 threads, ABBA, 6 rounds, on
an idle box:

| arm | median | spread | ISA |
|---|---|---|---|
| jitllm | 31.04 tok/s | **0.55%** | AVX2, pre-VNNI, 256-bit |
| llama.cpp b11011 | 31.99 tok/s | **0.63%** | **AVX-512**, skylakex |

    ratio 0.9712   IQR/median 0.0057   n=12   GATES
    jitllm    25.3 GB/s = 69% of the 36.63 GB/s wall
    llama.cpp 26.1 GB/s = 71% of wall

**IQR/median 0.0057 is twenty times tighter than anything the laptop produced
all day** (0.042 on its one gated pass, 0.254 on the one that failed). That box
hits 100 C, clamps to 934 MHz mid-round and discarded 18 of 32 pairs on
coherence; this one holds its clock and threw away nothing. **Take Linux rows
here.** The laptop is the developer machine and has been the measurement
instrument by accident.

★★★★★ **AND THE REMAINING ~10 POINTS WERE ALREADY IN THIS FILE, MEASURED, AS A
TECHNIQUE LESSON RATHER THAN AS A MAGNITUDE. THE ACCOUNT IS COMPLETE.**

    229729c  **0.8866**  AGENTS.md + one _test.go. NOTHING ELSE.
    0287c0e  **0.9853**  its predecessor -- BYTE-IDENTICAL shipping code

**A 9.9-point swing between two binaries with no executable difference**, on
that box, in that window. A second one says the same: 68d951e -- comments in
pack.go, zero executable lines -- read 0.926 while its reference arms gated at
0.9799 and 0.9883.

**So the harness produced 0.8866 for a commit that changed nothing, and 0.8997
for HEAD. Those are the same number.** The board rows the goal was written on
are indistinguishable from the null control taken beside them.

★ **THE TWO TERMS ADD UP.** 0.90 against the later 1.03 is ~13 points:

    ~3 points   sustained thermal clamp, from this file's own asymmetry
                (jitllm -7.4% against llama.cpp -4.2% -> a 0.967 ratio shift)
                on a box at 95 C with its fans at 70%
    ~10 points  the harness's own irresolution on that box, DEMONSTRATED by a
                docs-only commit reading 0.8866 against a byte-identical
                predecessor's 0.9853

Nothing is left over, and no term is a story: each is a measurement, one taken
later and one taken then.

★ **WHY IT TOOK THIS LONG, WHICH IS THE PART WORTH KEEPING.** The 9.9-point
number was recorded earlier as a lesson about BISECTION -- "probe a
comment-only commit FIRST" -- and filed under how to attribute a change. It was
never read as an answer to "how big can this harness's error be", which is the
same number viewed from the other side. **A measurement filed as a technique
note is still a measurement.** Four experiments and a rebuild went into hunting
a magnitude that was already written down one screen away.

★★★★ **AND THE MACHINE ITSELF IS THE DIFFERENCE, WHICH IS AS CLOSE TO A
MECHANISM AS THIS WILL GET. THE FANS WERE AT 70% THEN AND ARE AT MAXIMUM NOW.**

    then         95 C under six loaded P-cores, fans **4130 / 3912**
                 of a 5900 / 5600 maximum -- ~70%. Nine minutes of
                 complete idle only reached 76 C.
    later        **<= 74 C** across every board run that day, idle 61 C,
                 fans **5904 / 5569** -- AT MAXIMUM.
    same day     **100 C**, fans **4246-4274 / 3983-4024** -- 72%.
    (later)      Six P-cores, Llama-3.2-1B, n=320, one ordinary decode.

★★★ **AND THE THIRD LINE REFUTES THE SECOND, TAKEN THE SAME DAY. THE "IT IS NOT
A STATE TODAY'S BOX CAN BE PUT INTO" SENTENCE WAS WRONG: it takes one
`jitllm run`.** Sampled every two seconds through a plain 320-token decode on
`taskset -c 0,2,4,6,8,10`, the package goes 60 C -> 100 C in under four seconds
and STAYS at 99-100 C for the whole run -- against a crit of 100 -- with the
fans never leaving ~72% of maximum. That is HOTTER than the 95 C the standing
goal is written on, not cooler, and the rate shows it: **48.27 tok/s** where
this file records 57.67-59.93 for the same model, coreset and length.

So the "physical difference in the machine" reading was a difference between two
MEASUREMENTS, not between two machines: the <= 74 C line was taken across board
runs whose load profile is not a sustained six-core decode, and the fans hitting
5904/5569 in one sample does not mean they hold there. **The laptop reaches its
critical temperature under the exact workload the board is made of**, which is
the whole reason Linux rows are taken on the Xeon -- where the same measurement
holds 2600 MHz at 77 C with the cores 0.0% busy.

What the older entry got RIGHT and keeps: contention is a cliff, not a slope, so
it cannot produce a median of 0.90; and the thermal state explains the SIGN of a
depressed ratio rather than its exact size.

★ **AND A SUSTAINED CLAMP BIASES THE RATIO DOWN, WHICH IS THE RIGHT DIRECTION
AND ABOUT A THIRD OF THE MAGNITUDE.** This file's own asymmetry -- the M4
prefill row, jitllm -7.4% against llama.cpp -4.2% -- works out to a ratio shift
of 0.967, roughly **3 points of the ~13** between 0.90 and 1.03. So the thermal
state explains the SIGN and part of the SIZE, and does not explain all of it.
Recorded as a partial account rather than a closed one.

★ **WHY THE TRANSIENT-CLAMP COUNT ABOVE DOES NOT CONTRADICT THIS.** That count
found the clamp landing on llama.cpp 5 times to jitllm's 2 and biasing UP -- but
those were TRANSIENT clamps on a box at 59-74 C, where one sample in a round
collapses. A SUSTAINED clamp is a different regime: it is present in every
sample of every round, so nothing is discarded and the asymmetry accumulates
instead of showing up as dispersion. The two findings are about different
things and both stand.

★★★★ **AND THE QUESTION IS CLOSED BY MEASURING THE BINARY THE 0.90 WAS TAKEN
ON. IT READS 1.035.** Every test before this one compared THE THEN-CURRENT
HEAD, which carries 48 later commits -- so none of them could tell a measurement
artefact from a real regression that was later fixed. `52890bb` is the commit
whose own message reads *"Llama-3.2-1B on Linux CPU regressed ~9 points since
container v6 -- it is the engine"*. Built from source and run on the board that
now gates at 0.015-0.038:

    52890bb   **1.0366**  IQR 0.014  n=13  GATES
    52890bb   **1.0347**  IQR 0.029  n=13  GATES

**The binary that produced 0.8997 and 0.8961 reads 1.035.** Not a regression
that was later fixed; not slow at all. The engine at that commit was never the
cause, and the sentence "it is the engine" is refuted by running the engine it
names.

★ **SO THE SUBJECT OF THE QUESTION IS ANSWERED EVEN THOUGH THE MECHANISM IS
NOT.** "Why does the container path read 0.90" presumes the path reads 0.90; it
does not, on any binary, on any box, under any of the four conditions tested. The
four eliminations above stand, and the specific cause of the two original
readings is unknown and now unknowable without that day's machine state --
which nothing recorded, because the busy figure that would have captured it was
only added afterwards.

★ **THE STANDING LESSON IS THE ORDER OF OPERATIONS.** Four experiments went into
explaining a number as an artefact of the HARNESS before anyone measured the
BINARY the number was about. The binary took twenty minutes: one `git worktree
add`, one build, two passes. **When a row indicts a commit, run that commit
before theorising about the room it was measured in.**

★★★ **FOUR CANDIDATES FOR THE 0.90 ARE NOW ELIMINATED BY MEASUREMENT, AND THE
READING IS NOT REPRODUCIBLE ON THIS BOX UNDER ANY OF THEM.**

    candidate                what it actually does              verdict
    clamp asymmetry          biases the ratio UP (2.373 on a     REFUTED
                             clamped llama.cpp pair; the clamp
                             landed on llama.cpp 5 times to
                             jitllm's 2)
    cold start / no soak     biases UP ~1 point (cold 1.0445 /   REFUTED
                             1.0326 vs soaked 1.0339 / 1.0271)
    CPU contention           a CLIFF: 1.0276 clean -> 0.2595     REFUTED as the
                             with ONE of six cores contended     cause of 0.90
                             -> 0.3217 with six. Two states,
                             nothing between, so no dose gives
                             a MEDIAN of 0.90
    cold pack-width cache    nothing. Alternated warm/cold/      REFUTED
                             warm/cold, 12 rounds each, ALL
                             FOUR GATING: 1.0325 / 1.0313 /
                             1.0322 / 1.0229 -- warm mean
                             1.0324 against cold 1.0271

**And the laptop produced about ten independent passes that day, every one between
1.027 and 1.054.** A reproducible 0.8997 / 0.8961 is not reachable here under
any condition constructed. That is not a mechanism and must not be written up as
one: what it establishes is that the reading does not survive contact with a
gated harness, not why it happened.

★ **WHAT IS SETTLED WITHOUT IT.** The engine did not regress -- the gated
v6/HEAD row above says HEAD is 2.6-3.4% FASTER, behind a 0.17% self-control --
and contention, the one candidate that moves a row far enough to matter, is now
REFUSED by the busy gate rather than averaged in. The operative purpose of the
question is met even though the question is not answered.

★ **AND THE LAPTOP IS A USABLE INSTRUMENT AGAIN, WHICH IT WAS NOT HOURS EARLIER.**
Four alternated arms gating at IQR 0.015-0.038 and agreeing to one point is a
different machine from the one that produced 0.042 and 0.254 twelve hours
earlier. What changed is the harness, not the hardware: the busy gate, the
per-core clock sampler, the container-version check and the failed-frequency-read
fix. The box was never the whole problem.

★★★ **CONTENTION IS A CLIFF, NOT A SLOPE, SO IT DOES NOT EXPLAIN 0.90 -- AND
THE ENTRY THAT SAID IT DID WAS REFUTED THE SAME DAY IT WAS WRITTEN.** The
dose-response, measured after that claim was committed:

    0 of 6 cores contended   **1.0276**  IQR 0.012  n=12
    1 of 6 cores contended   **0.2595**  IQR 0.014  n= 5  (thin, but a tight median)
    6 of 6 cores contended   **0.3217**  IQR 0.045  n= 9

**ONE contended core is as bad as SIX.** That is the straggler: jitllm runs six
workers and every region ends at a barrier, so one slow worker holds the whole
region, while llama.cpp's threads degrade one at a time. The curve has two
states, ~1.03 and ~0.26, and nothing in between -- so a MEDIAN of 0.90 cannot be
produced by it at any dose. Intermittent contention does not rescue the story
either: a median over rounds lands on whichever state held for more than half of
them, which is ~1.03 or ~0.26, not 0.90.

**So all three named conditions are now eliminated** and the mechanism behind a
REPRODUCIBLE 0.8997 / 0.8961 is unknown. What is NOT in doubt: it was not a
regression (the gated v6/HEAD row above says HEAD is 2.6-3.4% faster), and
contention of any amount is now REFUSED by the busy gate rather than averaged
in. What is owed is an explanation for a 12% uniform deficit that three
candidates do not produce.

★ **AND THE CLIFF IS ITSELF THE MORE USEFUL FINDING.** "Do not measure on a busy
box" was already the rule; this prices breaking it. A SINGLE foreign process on
ONE of the six pinned cores costs 75% of the ratio -- not the ~17% a
proportional model predicts -- because the barrier makes the slowest worker the
whole region. That is why the busy gate's threshold is 20% and not 50%, and it
is why PSI missed it: one busy core out of twenty stalls almost nothing.

★★ **THE SUPERSEDED CLAIM, KEPT BECAUSE OVERCLAIMING IT IS THE LESSON.** The
entry below read "THE MECHANISM IS CPU CONTENTION ... AND IT IS THE ONLY
CANDIDATE OF THREE THAT MOVES THE RATIO THE RIGHT WAY", on the strength of one
dose (six cores) and a direction. One more dose refuted it. A mechanism that
moves a number the right way is not the same as a mechanism that produces it,
and the difference is a dose-response -- which was listed as "owed" in that very
entry and should have been taken before the claim, not after.

★★★ **THE MECHANISM IS CPU CONTENTION ON THE MEASURED CORES, AND IT IS THE ONLY
CANDIDATE OF THREE THAT MOVES THE RATIO THE RIGHT WAY.**
Llama-3.2-1B, `vs-llamacpp.sh`, 8 rounds, same box, same session:

    CLEAN                              **1.0276**  IQR 0.012  n=12  GATES
    a `go build` looping on all six    **0.3217**  IQR 0.045  n= 9  GATES

**A 3.2x collapse, and it is asymmetric by construction.** jitllm's pool SPINS
for 250 us, so a worker that loses its core to the compiler burns its whole spin
budget waiting; llama.cpp's threads simply deschedule. That is the same spin this
file already records as 60e9 instructions of workers WAITING, seen from the
other side.

★ **AND IT IS THE ONLY ONE OF THE THREE NAMED CONDITIONS THAT GOES DOWN**, which
is what makes it the account rather than a third candidate:

    clamp asymmetry   REFUTED -- the clamp landed on llama.cpp 5 times to
                      jitllm's 2 and drove pairs to 2.373 / 2.323 / 1.815, so it
                      biases the ratio UP. Ungated 1.0644 against gated 1.0536.
    cold start        REFUTED -- cold 1.0445 / 1.0326 against soaked 1.0339 /
                      1.0271. The soak is worth about a point and moves it UP.
                      (Worth keeping: this file PRESCRIBES JITLLM_SOAK=10 and the
                      reasoning is sound -- jitllm self-tunes, llama.cpp does not
                      -- but four passes say it is hygiene, not the difference
                      between a 0.90 row and a 1.04 one.)
    contention        CONFIRMED, above.

★ **AND THE GATE THAT CATCHES IT WAS ADDED THE SAME DAY, WHICH IS WHY THE 0.90
WAS INVISIBLE BEFORE.** The contended run had to be forced: `JITLLM_VS_BUSY=100`
disables the busy gate, because at its default of 20% it REFUSES -- the cores
read **98.5%** busy. PSI, the only gate that existed when the 0.90 rows were
taken, saw nothing, for the reason recorded under RULE 2: it measures STALL, and
on a 20-CPU box a job that fits in the spare cores stalls almost nothing.

★ **WHAT IS NOT ESTABLISHED: THE DOSE-RESPONSE.** 0.90 lies between 1.0276 and
0.3217 and has not been placed on that curve. The attempt to place it was
BOTCHED -- the hog loop spawned two processes per iteration and never waited, so
it fork-bombed the box rather than occupying two cores, and both rows were
rejected at n=1 and n=3. The gates refusing them is the system working; the
experiment was still wrong. A clean dose-response is owed if anyone wants to
predict a contaminated row's value rather than merely refuse it.

★★★ **AND "WHAT MECHANISM PRODUCED THE 0.90?" IS A SEPARATE QUESTION FROM "WAS
IT A REGRESSION?", AND THE OBVIOUS ANSWER IS REFUTED ON THIS BOX.** The row
above settles the second: HEAD is 2.6-3.4% FASTER than v6, so there is no
regression to attribute. The first is not settled, and it is worth saying so
rather than letting "it was the thermometer" stand as if it named a cause.

THE OBVIOUS MECHANISM, AND WHY IT DOES NOT HOLD HERE. This file records that
"the two engines do not shed rate to a clamp equally (the M4 prefill row: jitllm
-7.4%, llama.cpp -4.2%)", which predicts a clamped box biases the ratio DOWN --
the 0.90 direction. Counted over 32 real pairs from one pass, with the clamp
threshold at 0.75 of the session median (1511 of 2015 MHz):

    jitllm    arm   2 of 32 samples clamped
    llama.cpp arm   5 of 32 samples clamped
    a clamped llama.cpp sample drives its pair to 2.373, 2.323, 1.815

**The clamp landed on llama.cpp more than twice as often, and it biases the
ratio UP.** Ungated, that pass reads 1.0644 against the gated 1.0536. So on this
box, on this day, clamping flatters jitllm rather than punishing it, and the
"jitllm sheds more" mechanism is not what produced a 0.90.

WHAT IS ESTABLISHED, AND WHAT IS NOT. Established: the 0.90/0.70 readings were
taken under two conditions this file already names -- a SUBAGENT COMPILING GO IN
THE SAME TREE (RULE 3 item 3: invisible to every gate here) and a run STARTED COOL
rather than soaked, so it sampled the frequency ramp instead of steady state.
Not established: that either condition yields 0.90 SPECIFICALLY, or by what
mechanism. Both 0.90 rows reproduced across two passes (0.8997 / 0.8961), which
is what makes this unsatisfying -- a reproducible number usually has a cause, and
naming the conditions is not the same as naming the cause. **Recorded as open.**

★★★ **AND THE v6-vs-HEAD ROW ITSELF, GATED, WITH A SELF-CONTROL IN FRONT OF IT.
HEAD IS 2.6-3.4% FASTER THAN v6 AND THERE IS NO REGRESSION.**
Llama-3.2-1B-Instruct Q4_K_M, `model.TestFusionCeiling` -- one test binary per
tree, each converting with its OWN container code -- six P-cores, ABBA, on a
laptop that was 3.7% busy at 59 C:

    A/A self-control (HEAD in BOTH arms), run FIRST   0.9983   IQR 0.013  n=12  GATES
    HEAD / v6   pass 1                                1.0339   IQR 0.019  n=12  GATES
    HEAD / v6   pass 2                                1.0258   IQR 0.066  n=12  GATES

**The self-control is the part that matters.** This file already records why:
"Reproducing a number proves the harness is CONSISTENT; only the self-control
proves it DISCRIMINATES", and the self-control was never run on the harness that
produced the original 0.90. Here it reads 0.17% bias against an effect of 2.6-3.4%.
The two passes agree to 0.8 points and both gate.

★ **IT REPRODUCES THE EARLIER IN-PROCESS FINDING FROM A COLD START.** That
entry read 1.0481 / 1.0435 with a 1.1% A/A bias; this is 1.0339 / 1.0258 with a
0.17% bias, taken with different binaries built on a different day. Same sign,
same order, tighter control.

## RULE 2 as it stood in AGENTS.md: PSI, the busy-core gate and the baseline check

### ★ RULE 2: a perf claim is paired, interleaved, and gated.

A/B/A/B **within one process**, n >= 20, never A-then-B. Report the **median of
per-round ratios** with its IQR; **reject if IQR/median > 0.10**.

- **Soak to thermal steady state first.** Cooling between samples is the
  intuitive fix and it is the wrong one — it puts every sample at a different
  point on the frequency ramp.
- **Run the whole comparison twice.** A dispersion gate cannot see a bias common
  to both arms, and that is the failure that moves a row across parity.
- **Gate on PSI, not the load average** (`/proc/pressure/cpu` some avg10 < 25%).
  The load average reads 4.42 on a genuinely idle box here.

  ★★★ **AND PSI ALONE IS NOT ENOUGH, AS MEASURED. IT IS BLIND TO THE
  TENANT THAT ACTUALLY RUINS A ROW, AND THIS GATE PASSED A BOX AT 69% FOREIGN
  LOAD.** PSI "some" measures time tasks spend STALLED waiting for a CPU. This
  laptop has 20 logical CPUs, so a foreign job on nine of them stalls almost
  nothing:

      test262.test (another project)   913% CPU, on psr 0
      a Go compile (another session)   386% CPU, on psr 1
      total foreign                    1372% of 2000%  -- 69% of the machine
      package                          90 C
      PSI some avg10                   7.52%   -> the gate PASSED

  **psr 0 is the FIRST core in RULE 5's `taskset -c 0,2,4,6,8,10`**, so the row
  would have been pinned directly on top of a 913% tenant with the guard
  reporting a quiet box. That is RULE 3 item 3 -- "another project's `go test` is
  invisible to every gate here" -- and the gate written to catch it could not.

  **The question is not "is the machine stalled" but "are the cores I am about
  to pin to already busy".** `scripts/corebusy.py` samples `/proc/stat` for
  exactly those CPUs over one second; `vs-llamacpp.sh` refuses above 20%
  (`JITLLM_VS_BUSY`). It sees a tenant whether pinned or unpinned, another
  project's test or another session's compile, and it needs no process
  attribution. Live on the box above it reads **52.6%** and refuses by name.

  ★ **TEMPERATURE IS REPORTED AND IS NOT A REFUSAL**, because RULE 2 soaks to
  steady state on purpose -- a warm package is the correct state to measure in,
  and a gate on it would refuse every legitimate row. It is printed beside the
  busy figure so a clamped row can be read against it afterwards.

  ★ **AND HOW MANY PAST ROWS THIS CONTAMINATED IS UNKNOWN.** The board threw out
  3 of 7 rows and a third of its samples as "power-clamped" on a box where this
  gate was passing silently. That is a candidate cause and NOT an established
  one: no row carries a record of what else was running, which is exactly why
  the busy figure is printed at the head of every run now.
- **Discard power-clamped samples, never average them.** The package drops to
  ~900 MHz at random and the die gets *cooler* as the rate collapses. When
  frequencies are low and two passes disagree, **stop measuring and let the box
  rest** — do not raise the round count.
- **Run the A/A self-control FIRST, ON THE HARNESS YOU ARE ABOUT TO QUOTE.** One
  M4 A/B had a 3.5% slot bias that only a same-binary-both-arms run exposed. And
  a self-control is not transferable between harnesses: on the Linux box the
  in-process probe controls at 0.9948 while the process-per-sample A/B beside it
  controls at **1.09**, which is larger than most effects worth measuring.
  **Reproducing a number proves the harness is CONSISTENT; only the self-control
  proves it DISCRIMINATES**, and a nine-point "regression" that did not survive is what that
  distinction cost -- see the Llama-3.2-1B entry above.
- **`scripts/vs-llamacpp.sh` is the only source of a jitllm-vs-llama.cpp row**, on
  both hosts. A scoreboard whose rows come from two tools is not a scoreboard.
- **One run measures a cold engine.** jitllm tunes itself while it runs; take the
  last of several, or warm the cache and say which.
- **★ CHECK THE BASELINE ARM'S ABSOLUTE RATE AGAINST WHAT THIS FILE ALREADY
  RECORDS FOR THAT HOST, BEFORE READING THE RATIO.** A ratio is two numbers and
  a mis-configured baseline moves it exactly as far as a real win does.
  Measured: `LCPP_THREADS=6` was HARDCODED -- this laptop's six
  P-cores -- while jitllm sizes its pool from the taskset it is handed. On the
  Xeon, coreset 0,1,2,3, llama.cpp was told `-t 6` and then CONFINED to four
  cores:

      inside the harness, -t 6 on 4 cores      24.7 tok/s   ratio 1.2413 "WINS"
      standalone,         -t 4 on 4 cores      32.08        ratio 0.9637 loses

  **Both rows GATE at IQR/median 0.004 over twenty pairs, and one of them is
  nonsense.** No dispersion gate, no A/A control and no second pass can see
  this -- the arms are tight, reproducible and wrong -- and it biases the ratio
  UP, which is the direction that flatters this engine. What caught it was that
  llama.cpp's absolute rate disagreed with the 31.99 already on record for that
  host. The thread count follows `$CORES` now, because a constant cannot follow
  the box; the LESSON is that the standing figures in this file are a gate the
  statistics cannot replace.

## RULE 2f as it stood in AGENTS.md: the collector and the off-heap frames

### ★ RULE 2f: THE GO COLLECTOR IS PART OF THE ENGINE'S BUDGET, AND A LIVE HEAP ON ITS GOAL COSTS 42% OF THE CPU.

Page frames are permanently live, so a weight budget close to the memory limit
leaves the collector a goal it can never reach: it runs back to back and frees
nothing. Measured on Qwen3-Next-80B under `scripts/cap 12G`, and the
cliff is sharp rather than gradual -- **8.34 GiB of budget is 6 GC cycles and
8.99 GiB is 648**:

    profile, old default   gcBgMarkWorker 42.5%   Pool.await 22.0%
                           pager reads 13.2%      generated kernels 5.0%
    before   budget 9.71 GiB   860 cycles   sys 116.71s   312% CPU   80.4s wall
    after    budget 7.34 GiB     6 cycles   sys   5.10s   164% CPU   28.9s wall

**THE FIX IS THAT THE FRAMES ARE NOT HEAP. Page frames and the dense region
are anonymous mappings (`kernels.MapOffHeap`, the device arena's allocator),
outside the collector's goal, its memory-limit arithmetic and its marking.**
olmoe in a 5 GiB cgroup, at the same 3.30 GiB budget: frames on the heap 6,540
cycles and 28.6 s of GC CPU, off it 7 cycles and 0.02 s, prompt 1.51x. With no
collector reserve the budget holds the whole model there, and it reads 3.58 GiB
instead of 23.8. The Go memory limit is only a backstop above the frames, set
when it leaves the real heap a gigabyte; a limit sitting just above the heap
spirals the same way (0.21 GiB live against 0.06, 6,686 cycles). The cost is the
collector's safety net: a span used after its model's Close faults.

`sched.GCBudgetCap`'s quarter-of-the-heap reserve applies only with the weights
ON the heap -- `JITLLM_OFFHEAP=0`, the control arm, or a platform with no
anonymous mmap. `GOGC` was never the fix there: under a memory limit the goal is
`min(live*(1+p/100), limit-nonheap)`, so once live is at the limit no value of p
moves it.

**And `weak.Pointer` was tried and measured**: 64 weakly held buffers, one
`runtime.GC()`, **0 of 64 alive**. Go's weak pointers have no LRU and no
pressure order, so a page cache built on them loses its whole working set every
cycle.

**A gate that measures RSS is measuring the allocator.** The leak gate's "21 MiB
per cycle" was the scavenger: live heap +32 KiB over four cycles and the tier's
device ledger back to its placement figure, against +108 MiB of RSS.
`docs/engineering-history/scheduling-and-measurement.md` has both.

## RULE 3 as it stood in AGENTS.md: the cap, and why -race needs 24G

### ★ RULE 3: every heavy job runs under `scripts/cap`. NO EXCEPTIONS, INCLUDING SUBAGENTS.

    ./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go test ./... -count=1

`MemoryMax=8G MemoryHigh=7G MemorySwapMax=0`. Use `24G` for a model over ~16 GB:
cgroup v2 charges a mmap'd file's page cache to the cgroup.

**★ AND `-race` NEEDS `24G` TOO, WHICH COST AN HOUR OF MISATTRIBUTION.** The
detector's shadow memory is several times the live heap, so a suite that
converts a GGUF crosses `MemoryHigh=7G` and the kernel throttles the cgroup into
reclaim. It does not fail and it does not report anything -- it just runs at a
fraction of the rate, and the test that happens to be running when reclaim bites
looks like the culprit. Same tree, same command, one flag apart:

    ./scripts/cap  8G -- go test ./format/jlm/ -count=1 -race    > 380 s, still going
    ./scripts/cap 24G -- go test ./format/jlm/ -count=1 -race      114.8 s

**A slow `-race` run is a CAP question before it is a code question.** I read
that 4x first as a regression in the code I had just written, then as foreign
load -- citing a 460% process that was my own test -- and both were wrong. The
tell was available the whole time: the per-test breakdown put every nonzero
second in GGUF conversion tests and 0.00s in every test of the code under
suspicion.

**There is no bypass variable in `scripts/cap`, on purpose.** `JITLLM_NO_CAP=1`
belongs to the HOOK, where it lets a command run *without* this script and is
visible in the command line. A bypass visible in the command is a decision; a
bypass that makes a capped-looking command uncapped is a trap. systemd-oomd has
killed the desktop three times here.

1. **Every prompt dispatching an agent that may build or measure must contain the
   cap command verbatim.** An agent that has not been told will not do it.
2. **Never fan out parallel measurement agents.** Measure serially, in the main
   loop.
3. `scripts/cap` takes a `flock`: measurements EXCLUSIVE, builds SHARED. It
   protects this repo only — another project's `go test` is invisible to every
   gate here and will land mid-round.
4. **Kill background jobs when the work that spawned them finishes.** Work
   started before a context compaction keeps running afterwards and neither end
   knows.

## The cap is the developer machine's, not a measurement's

RULE 3 used to say "no exceptions", and `scripts/vs-llamacpp.sh` took it
literally: to hold the measurement lock for a whole comparison it re-exec'd
itself under `scripts/cap` (8G by default) on every host, so the Xeon rows and
the rented-runner rows ran both engines inside a memory cgroup. That cgroup
charges llama.cpp's mmap'd model to the cgroup as page cache and throttles it
at MemoryHigh; jitllm reads with O_DIRECT into anonymous frames, which the
cgroup charges differently. A large model's row could then measure the cap as
well as the engines. The user's decision: the cap protects the developer's own
machine from OOM, and a benchmark host runs bare with only the lock. The script
now takes the lock file itself, and on the laptop reuses the descriptor the
outer `scripts/cap` already locked (a second open would be a second lock,
waiting on its own parent). `vs-llamacpp-paged.sh` capped each sample
separately and was changed the same way.

The container fallback went with it: with no systemd and no writable cgroup,
`scripts/cap` used `ulimit -v`, which limits address space rather than memory.
The CUDA runtime reserves address space in proportion to the card, so
llama-bench aborted at a 12 GiB limit and ran at 256 GiB. A container is not the
developer's machine either; it now runs uncapped and prints that it did.

## RULE 11b's cases: a red golden, and tests that worked by accident

**It cost a red gate that nobody looked at.**
`lowertest.TestDenseKernelsUnchanged` had been failing on four
`matvec-moe/*` hashes since the expert-major packing rewrote expert indexing
(`wrow`/`wrowD`/`wrowSC` from `PackedWords`). The change was correct and the
golden was simply never updated, so the gate said FAIL for something real that
was already fixed -- and a gate that is always red is a gate nobody reads, which
is RULE 10 from the other direction.

**And blessing it is not fixing it.** `-update` makes a golden agree with
whatever the code now emits, which is exactly what a wrong change would also
produce. A golden may only be updated once the behaviour behind it has been
re-derived from something that is NOT the golden -- here
`backend.TestIndexedMatVec{MatchesDense,PerSlotActivations}`, which were red for
the SAME change and turned out to be the evidence: they packed the bank as one
tall sheet and indexed it at `e*rows+r`, the layout from before expert-major
packing, where packing a bank as one sheet scatters expert e at stride
`experts*rows`. They pack per-sheet now and their oracle runs the dense kernel
on that expert's OWN sheet. The kernels were right all along.

★ **AND THE COMMONEST SHAPE IT FINDS IS A TEST THAT WORKED BY ACCIDENT.** Three
of the ones collected in one sweep were green only because `build()` used to
fault every block at load, and went red when `Open` became O(bytes read) rather
than O(model): `TestMoEFaultsAreVisible` called `s.moe()` without paging the
block in, and `tensorBytes` read `len(t.data)` -- zero for a weight whose page is
out, which priced every block at nothing and made a PLACEMENT think the whole
model was free. A gate that has been red since a refactor is not noise; it is
that refactor's unfinished half.

## Fused runs: a decode token's resident host blocks as one pool region

`TestFusionCeiling` priced whole-layer fusion as regions removed times a round
trip, and the round trip is mostly the region's door, not its barrier: a
sequence bump, a gate every worker CASes to join and the caller closes, and a
wait for every joined worker to leave. `sched.Pool.Fuse` keeps the door open.
The workers join once; every region the caller issues until `Unfuse` is a
PHASE: one store of a word carrying the phase's tag, its chunk count and the
next chunk, claimed by CAS on the whole word, and finished when the chunks done
reach the count. Completion is counted in chunks rather than workers, so a
worker the OS keeps off a CPU holds nothing up unless it claimed a chunk -- the
property the gate exists for -- and a worker reads the job only after a claim
succeeded, so a straggler can never run a chunk of the next phase with the last
phase's body.

What the model fuses is a RUN, not a block: consecutive resident host blocks
and the host head are one region (`engine/model/fuse.go`). A device run, a block
whose page has to be read, or the end of the token closes it. Nothing is
compiled, so a relocation, an eviction or a seam move needs no re-link: the next
token's runs follow its placement as it is.

    Llama-3.2-1B Q4_K_M, 16 blocks   per op 218 regions/token   fused 1 region + 218 phases
    olmoe-1b-7b Q4_K_M,  16 blocks   per op 234                 fused 1 + 234
    Qwen3.5-0.8B Q4_K_M, 24 blocks   per op 354                 fused 1 + 354
    gemma-3-1b Q4_K_M,   26 blocks   per op 353                 fused 1 + 353
    the same with a two-frame page budget: one run per block read (15-25 a token)

The kernels, chunks and order are the per-op path's, so the logits are the same
bits (`TestFusedBlocksMatchPerOp`, every principle fixture, resident and paged).
Run against violations: a phase published before its predecessor finished
(`TestFusedPhaseOrder` fails; the model gate crashes), and workers that drop a
third of their chunks (the model gate fails at token 0). The first version also
caught its own bug: a worker that joined late read `p.j` after the first phase
had replaced the fused job and drained the phase as an ordinary region, running
chunks twice. Whether a worker is fused is now asked of the region, not the job.

Not fused, by choice: a crew another session or crew shares the turn with (a
fused caller holding the cores while it waits on a lock another session's
region holds would deadlock), a NUMA pool (a phase drains flat, and DoNodes'
placement is what that pool is for), and a run with pprof region labels on (a
phase carries no label, so a profile is the per-op run). Prefill and the batched
step are not fused; their regions carry far more work each.

On the laptop an empty phase cost 489 ns against an empty region's 1.06 us
(`JITLLM_FUSE_BENCH=1 go test ./engine/sched -run Cheaper`, wide IQR: indicative).
The Xeon rows are in docs/perf/current.md's evidence trail and the PR.

## The serving split: the E-cores prefill while the P-cores decode

Evidence for `model.WithServingSplit` (on by default where the host has E-cores
a thread can be kept on: macOS on Apple Silicon). The rule that governs it is
AGENTS.md's; this is what was measured.

With the default pools, one session decoding while another prefills drops the
decoder from ~52 tok/s (alone) to ~11: both sessions' P-core pools and the
prefill's P+E pool overlap, so every region of either waits on one lock, and the
decoder, never alone on its crew, never fuses. With the split, while another
State of the model ran a token in the last 250 ms a prefill goes (chunk by
chunk) to the E-cores' own pool and lock, its threads at the E-core QoS class;
the decoder keeps the P pool and fuses. A session alone, or two sessions both
prefilling, are on today's pools exactly.

Two things the first build got wrong, both found by the A/B showing 0 fused
runs: an E-core crew over cores the P+E prefill crew overlaps took that crew's
lock (and so was a user of the decoder's), and a session's idle decode view was
a user of the P-cores while it prefilled elsewhere. The E crew now has a lock of
its own, the P+E crew takes it after its own, and a view whose caller prefills
on the E-cores is parked (`sched.Pool.Park`) so the decoder is alone.

Rows (M4, Llama-3.2-1B Q4_K_M; decoder at depth 64 timing 64 tokens while a
second session prefills 32-token prompts in a loop; one process, base and split
interleaved, n=20, A/A first, two passes; decode ratios are base/split time):

| E-core QoS class | load | A/A | pass 1 | pass 2 | decode base -> split | prefill beside it, base -> split |
|---|---|---|---|---|---|---|
| UTILITY (0x11) | none | 1.0025 (1.1%) | 4.391 (3.2%) | 4.361 (3.9%) | 9.6 -> 42 tok/s | 154 -> 106 tok/s |
| USER_INITIATED (0x19) | none | 1.0051 (2.4%) | 2.467 (3.9%) | 2.462 (2.1%) | 10.9 -> 27 tok/s | 174 -> 256 tok/s |
| UTILITY | 6 foreign busy loops | 1.0060 (1.8%) | 4.520 (5.6%) | 4.642 (6.8%) | 6.7 -> 30.5 tok/s | 107 -> 15 tok/s |
| USER_INITIATED | 6 foreign busy loops | 0.9959 (1.5%) | 2.527 (7.2%) | 2.621 (5.0%) | 6.7 -> 17 tok/s | 107 -> 236 tok/s |

(Fused runs: none in the base arm, 1260 in every split arm.) USER_INITIATED's
prefill beside the decoder is faster than the base arm's: its threads are not
held on the efficiency cluster, they take P-cores from the decoder, and the
win is half. BACKGROUND (0x09), measured before, held the E-cores but starved to
0 tok/s under a foreign E-core load. UTILITY is the class that holds the
decode win in both cases and still finishes prompts under load (15 tok/s against
107, not 0); `TestServingSplitAB` fails on a split round that finishes none.

A single session decoding and prefilling in turn, split on against off: decode
plus second prefill 1.0028x and 1.0000x (IQR/median 2.1%, 1.8%; A/A 0.987), first
prefill 1.0084x, 1.0085x; no prefill was placed on the E-cores
(`TestServingSplitSingleSessionUnchanged`). Linux is unchanged by construction:
`sched.CanSplitE` is false off macOS, so no prefill is placed on an E pool, no
crew takes an extra lock, and the thread-class calls are empty.

The cost is the prefill beside a decoder: it loses about 30% against the
shared pools (154 -> 106). It is the trade, and `WithServingSplit(false)` /
`JITLLM_SERVING_SPLIT=0` is the other arm.

## RULE 11f's cases: a cache that existed, a store that existed

Both cases are work proposed or built beside machinery that already did it.

- **A device expert cache.** It was proposed as new work. `streamBank` in
  `jit_old/gpu/tier/layer.go` already is one: the host side of a block whose routed
  experts are assembled per token, with its own residency. The task was to read
  it and extend it, not to write a second cache beside it.
- **Decision prefixes.** The decision branch reused the prefix cache through
  `MemStore`, which copies pages in and out, so every cached prefix was a second
  host copy of the KV; and sealing copied the device's KV home on every prompt,
  which made warm decisions slower than cold on a V100 (260 against 121 ms).
  PR #148 (closed unmerged) instead kept its own trunk snapshot and recurrent
  copy per Decider. `engine/model/kvshare.go`'s `SharedStore` already shared
  prefix pages between sessions without copying them, which is also what "KV is
  never held twice" requires.

A PR's "what existed and was reused" section could not have been written for
either at the time, which is how the duplication was found late instead of at
the proposal.

## Four concurrent requests served a third of one stream: the memory cache sealed every prompt on the step loop

**The reading.** The deployment search (docs/perf/http-bench.md) found every
setting it swept, the unchanged default included, at 12-17 tok/s of aggregate
throughput at concurrency 4 and a TTFT p99 near 18 s, on Llama-3.1-8B Q4_K_M on
one V100, where one stream on the same card decodes 138 tok/s and the loop's
decode steps cost 8-12 ms. So the decode was not the cost, and no setting
moved it.

**Where a request's time went** (jitllmd, one request at a time, a 129-token
prompt, 64 tokens, the step loop timed per step; V100 under the shared box's
ordinary load):

| | prompt step | TTFT p50 | the same with `-no-mem-cache` |
|---|---|---|---|
| main | 470-630 ms | 480-950 ms | 64 ms prefill, TTFT 62 ms |

`State.SealPrompt` (the memory cache's offer of a finished prompt) runs inside
the step that finishes the prompt, and a profile and per-phase timers put the
time there:

1. **The store half, 290-360 ms.** `sealTail` offers the partly filled last page
   at every sixteen-position boundary and its exact end, each as a page of its
   own, so a 129-token prompt copied 704 positions of K and V per layer (5.5
   times the prompt), through `io.ReadAll`'s grown and cleared slice and then a
   second copy into the mapping, with the collector scanning the garbage. The
   fills are byte prefixes of one another.
2. **The card half, 90-140 ms on a quiet box, 200-570 ms under the box's other
   jobs, outliers of 2 and 7 s.** `syncKVFromDevice` reads each layer's pages
   home (a gather, a read, two device allocations, a host transpose) so the store
   has something to hold. Device allocation and free are 1-2 ms of it, the gather
   3-4 ms, the reads 17-55 ms; the rest is host work that slows with the host.
3. **The step budget collapsed.** `stepBudget` fits a step's time against the
   prompt tokens it carried, and the seal was inside that time. A step that
   completed a prompt read 250 ms for a few tokens; the fit went to its floor
   and stayed there (`rows=9 prompt=8` step after step), so every later prompt
   beside a decoding row was fed 8 tokens a step and its first token waited
   16 steps, each carrying the seal of the prompt before. This is the failure
   the budget's own comment describes for an outlier step; the outlier was the
   seal, every time.
4. **A latent panic the fix exposed.** With rows no longer in lockstep, a State
   that had led a host step across sessions ran its next token alone and sliced
   a nil `s.qgate` (`takeHostBatch` had lent the model the leader's scratch):
   an architecture with a gated attention output (Qwen3.5 reproduces it)
   panicked in `projectQ`. It reproduces on main with
   two States and no server (`TestAStateThatLedAHostStepDecodesAlone`).

**What changed.** `MemStore.SetFills` holds the fills of a page over one
off-heap copy, charged once; `Set` reads a reader that knows its length
straight into the mapping. The loop seals after the step is timed, so the
budget and the fairness quantum are fitted to the step alone. `forward()` makes
the gate rows a State needs when it no longer holds them. Gates run against
violations: a corrupted copy fails `TestMemStoreFillsServeASharedPrefix`; the
parent commit panics `TestAStateThatLedAHostStepDecodesAlone`. The store half
fell from 290-360 ms to 35-45 ms a request.

**Paired, interleaved, A/A in the same pass** (`dev/deploysearch`, a fresh
server per sample, warm-up at the measured concurrency, ratios are fix/main per
round, median [IQR/median]; a ratio is quotable at 8 rounds or more and
IQR/median up to 0.10). The fix is the commit after the one the main arm is
built from.

Laptop, Llama-3.2-1B Q4_K_M, GeForce RTX 3050 Ti Laptop GPU (CUDA), six P-core
threads, 8 rounds, host load 2-6:

| | main | fix | fix/main | A/A (main twice) |
|---|---|---|---|---|
| c=1 tok/s | 142.4 | 149.9 | 1.053 [0.006] | 0.998 [0.006] |
| c=1 TTFT p50 | 84 ms | 62 ms | 0.743 [0.018] | 1.003 [0.006] |
| c=1 memory | 2651 MiB | 2479 MiB | 0.936 [0.003] | 1.000 |
| c=4 tok/s | 238.0 | 271.5 | 1.133 [0.263] refused; 1.153 [0.087] in a second pass | 0.990 [0.149] refused; 1.008 [0.197] refused |
| c=4 TTFT p99 | 490 ms | 367 ms | 0.740 [0.171] refused | 1.060 [0.213] refused |

c=4 against c=1: 1.67 times one stream before, 1.81 after. On this 1B the seal
is 11-16 ms, so the gain at c=4 is the budget, not the copy.

V100 (Tesla V100-SXM2-16GB, CUDA, one card under its lock), Llama-3.1-8B
Q4_K_M, the box carrying other projects' jobs (load average 5-57 while these
ran, which moves TTFT between 0.5 and 3.7 s at one stream and is why the
numbers are not gated):

| pass | c | main tok/s | fix tok/s | fix/main | A/A (main twice) |
|---|---|---|---|---|---|
| 8 rounds | 4 | 23.0 | 53.6 | 2.142 [0.461] refused | 0.994 [0.441] refused |
| 8 rounds | 4 | 59.0 | 80.6 | 1.563 [0.173] refused | 0.904 [0.835] refused |
| 12 rounds | 4 | 19.9 | 25.5 | 1.294 [0.243] refused | 0.920 [0.204], 0.949 [0.115] refused |
| 8 rounds | 1 | 18.2 | 36.0 | 1.564 [0.377] refused | 1.070 [0.366] refused |
| 8 rounds | 1 | 17.1 | 28.0 | 1.610 [0.222] refused | 0.932 [0.786] refused |
| 12 rounds | 1 | 15.0 | 23.7 | 1.501 [0.206] refused | 1.004 [0.100] |

Every V100 ratio is on the side of the fix (three passes at c=4, 1.29-2.14, TTFT
p50 0.23-0.70; three at c=1, 1.50-1.61; host plus device memory 0.81-0.82,
because the store's fills are held once), and no throughput ratio passes the
gate: the A/A copy of main moves as far as the fix does. The last c=1 pass, on
a quieter box, did resolve one: TTFT p99 0.846 [0.054] with the A/A at 1.002
[0.017]. That is the measurement, not a claim of a gain of any size. The laptop
pair is the one that resolves throughput.

On the V100 the fix did not make four streams scale: in the 12-round passes c=4
read 1.33 times c=1 on main (19.9 against 15.0 tok/s) and 1.08 times after
(25.5 against 23.7). The seal's card half is what is left.

**What is left, and what was tried.**

- The seal's card half still runs on the loop: 90-140 ms a request at 8B on a
  quiet box, and under load seconds, during which no row steps. At c=4 on the
  V100 TTFT is still the seal and the prompt step in turn. Reading the history home without a host copy of it per layer, or
  reading it home only for the pages the store does not hold, is the next
  lever.
- Sealing on a goroutine beside the other rows' steps (the row waits for its
  own seal; decode and other prompts proceed) was built, with a gate that holds
  one seal and demands another request finish, and passed the batch gates and
  `-race`. On the laptop it left tok/s at 1.08 [0.14] and made TTFT p50 1.5 times
  worse (the seal's device calls wait behind the other rows' steps); on the
  V100 it read 1.63 [1.2] against 1.56 [0.17] for sealing after the step, TTFT
  p99 7.7 s against 3.9 s. Not kept.
- A cold server is not a warm one: `hb`'s warm-up ran at concurrency 1, so a
  server measured at c=4 compiled and tuned each joint width inside the
  measurement. `httpbench -warmup-c` and `deploysearch` now warm at the mix's
  concurrency.
- The 64-request memory gate (`TestAServerUnderLoadStaysUnderItsBound`) took
  a cache that held 33 MB of each prompt; now 6 MB, so it reaches its bound with
  a host reporting 512 MiB available.
