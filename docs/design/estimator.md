# The composer's estimator

The estimator prices a kernel from its final IR, before anything is lowered
or run: the tool a tuner, a placement search and an auto-tuner ask "which
form, where, at what cost". It runs no model code. What it knows of a
device comes from `engine/probe`, which initialises the device and measures
it with generated kernels of its own; whether the estimate is right is held
by tests that run the composed step and time it, never by the estimator.

- `jit/estimate_model.go` -- the model.
- `jit/estimate.go` -- `Estimate`, `Suggest`, `EstimateCodes` (each call's
  estimate beside the values, reaches and storage it assumed).
- `engine/probe/host_amd64.go`, `engine/probe/gpu.go` -- the probes.
- `paritytest/board/estimate_test.go` (`TestEstimateMatchesTheHost`),
  `paritytest/board/estimate_gpu_test.go` (`TestEstimateMatchesTheGPU`) --
  the validation.

## What a kernel costs

Counts come from the IR at the step, an instruction the device's compiler
folds (a constant, or pure arithmetic on constants) never issued: each
instruction's runs (element count
times the trips of its loops, trips evaluated from the step's descriptors,
page table and rotary positions, per element where a count reads its
coordinates), each buffer's distinct bytes (bounded by its shape, the bytes
touched, its loads' footprints and the interval its index takes) at the
element's real width.

A kernel's time is the largest of:

- **memory**: each buffer's bytes at the rate of the levels serving it, plus
  the longest stream start among its buffers (a few lines read one at a
  time where that is shorter). On a GPU the levels are one pipeline -- every
  byte from farther out passes through each nearer level -- so it is the
  busiest level's time, and a load whose lanes sit a sector or more apart
  moves a sector a lane through the first cache a launch does not start
  empty;
- **compute**: the busiest instruction class at its rate, or all of them at
  the rate a mix issues; a GPU kernel at the share of the multiprocessors
  its workgroups fill;
- **waits**: the longest chain of dependent reads; on a GPU each load's runs
  a thread times the mean latency of its buffer's reads over the loads a
  thread keeps in flight (a subgroup issues in order);
- **chains**: the longest fold's trips times its step's critical path, or
  the critical path through the code outside any loop; on a GPU, one
  thread's schedule with its subgroup issuing in order, each instruction a
  slot after the last and no sooner than its operands are done -- a load in
  a loop whose place reads nothing the loop carries issued ahead, its
  latency over the loads a thread keeps in flight -- a loop its per-thread
  trips times a trip;

plus the wait for its code's first line where the code left the nearest
cache, and the call (host) or the device's per-kernel floor, or the host's
launch where launches issue slower (GPU).

Where a buffer is served from is a reuse-distance stack across the step's
kernels. A cache holds a buffer whose bytes, with everything touched since
its last use, fit it; past that it keeps the fraction its replacement keeps,
`Retain` to the power of how far over the level those bytes are, and no more
of a buffer larger than itself than its `Cycle` share. A GPU's L1 starts
each launch empty and serves only the kernel's own re-reads. A part's
boundary value is the storage of the value the part before it wrote.

## What the probes measure

Host (amd64): each cache's size from the OS, read rate on one P-core and all,
latency (a chase on huge pages within the TLB's reach), stream start,
retention after twice its size is read, and the share a buffer twice its
size cycling through it keeps; memory's rate, latency and stream start; each
class's issue rate and the delay of a dependent chain, a mix's issue rate,
an empty call's cost.

GPU, on any backend with a per-kernel device clock (CUDA's events,
Metal's command buffers' stamps): the host's launch rate and a kernel's
floor, both after reads of the whole memory have woken the device; memory
rate and latency; L1 and L2 found from one thread's chase over growing sets
(read rates cannot find them: the multiprocessors' L1s together can hold
more than the L2), L2's rate from every workgroup reading one set; loads in
flight a thread; each class's rate and delay, the group sum built as the
composer's lowering builds it, each class's instruction one the device's
compiler cannot fold (the integer class a shift and an exclusive or, since
an integer multiply is slow on some devices); the multiprocessor count
(CUDA's, the IORegistry's GPU cores on Metal). Instruction-class
measurements run behind a read of the whole memory, so a one-thread kernel
is timed at a loaded device's clock. A device whose runtime states no
resident threads takes the device tier's 16384.

## How well it holds

synth-ernie45, one decode step, `scripts/cap`:

| host | regime | step estimate / measured | calls within 2x | median error |
|---|---|---|---|---|
| laptop, one P-core (i9-12900HK, AVX2) | caches swept between steps | 1.00-1.03 | 53-54 of 64 | 1.28-1.42x |
| laptop, one P-core | steady (step after step) | 0.92-1.20 | 40-57 of 64 | 1.26-1.83x |
| V100-SXM2-16GB (cato, CUDA, sm_70) | steady | 0.88 | 65 of 65 | 1.07x |
| Apple M4 (Metal) | steady | 1.18-1.20 | 59 of 65 | 1.06-1.08x |

The device clocks were checked against the wall (`-estimate-clock`): on the
M4, Metal's per-dispatch stamps and 200 back-to-back launches in one
command buffer agree within 10-25% kernel by kernel, so the stamps measure
what the kernels take.

The laptop's steady regime moves with the probe: this fixture's weights are
about the size of its L3, so the answer turns on the replacement numbers,
which a shared desktop measures noisily. The cold regime is the one a model
larger than the caches lives in.

## Open

- **One-thread kernels on the GPU.** The composed argmax runs 103,424 logits
  on one thread (2.6 ms a step on the V100, 1.2 ms on the M4); the estimate
  reads 0.86 and 1.21 of it. The kernel is owed a reduction across the
  device.
- **One warp of straight-line code.** A quantize kernel of about 2,300 PTX
  instructions on 32 threads, with float and integer divisions by run-time
  values, reads 0.59: the device expands those divisions into sequences the
  IR's instruction count cannot see.
- **Barriers** are not priced; the attention stage, with eight, reads 0.83.
- **Apple's compiler removes work the IR counts.** The M4's six outliers
  are one kernel, the activation quantize (one workgroup of 1024 threads,
  848 integer instructions a thread, all of which NVIDIA's compiler keeps):
  2.8x over. One 1024-thread workgroup runs 53 G integer operations a second
  on the M4, a uniform chain no more than 1.5x that, so the kernel's 2.7 us
  of work leaves room for a fifth of the instructions. Seeing which needs
  Apple's ISA (a binary archive through `applegpu`'s disassembler).
- **The M4's launch cost** reads 1.5 us on most runs and 3.3 us on some,
  the same within a run: where macOS places the encoding thread.
- **Arm64 and the SSE tier** have no host probe yet.
- **Per-kernel times on the host** carry the clock's cost (taken off) and
  rdtsc ordering; a call shorter than about 20 ns is within its noise.
