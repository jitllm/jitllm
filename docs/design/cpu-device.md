# The CPU as a device

What "the CPU is a device" (AGENTS.md, the five principles' Relocation row
and the Scope's "every accelerator on the machine") means for the code.
This file is the design; the rule is AGENTS.md's.

## What it replaces

The composed CPU runs a block through methods of the instance
(`engine/model/cpuserve.go`: `cpuStep`, `cpuRows`, `cpuRecRows`,
`cpuSessions`), keeps its forms, scratch and counters in the instance's
`cpuState`, and reports through eight instance accessors
(`CPUBlockSteps`, `CPUEmbedSteps`, `CPUHeadSteps`, ...). The card is a
device: a tier behind `nn.LayerDevice` and its optional interfaces, with
its own `Stats`. Placement is `onDev []bool` beside one card route, and
the decode, prefill and batch loops each branch "a run on the card, else a
CPU step". So the CPU is the code's default: what runs a block no card
holds, counted in a vocabulary of its own.

## The shape

- **One interface.** The CPU device implements the interface the card
  implements: `PrepLayer` composes and binds block li's CPU form at
  placement, `Layers` runs a range of blocks over a call's rows,
  `MigrateKV` and `MigrateRec` move a block's history into or out of the
  pages it reads, `ReserveKV` says whether those pages can grow. The
  optional interfaces a batch or a session step needs (`RowsDevice`,
  `SeqKVDevice`, ...) are the card's, implemented by the CPU device where
  it serves them.
- **A device per block.** Placement is the device each block runs on, the
  CPU device among them; the loops run each run of consecutive blocks
  sharing a device as one call on it. A device that declines a block
  names why, and placing the block is choosing another device for it,
  never falling to the CPU.
- **One kind of counter.** Every device reports what it served and what
  it declined, by part (blocks, input rows, head, pool, streams,
  prediction blocks, sampler) and by `composer.Decline`, in one shape,
  read through the instance's devices. No instance-level CPU counter.

- **Two halves, as every card has.** The CPU device's code generation is
  the composer's, beside each card's emitters: `composer/x86` and
  `composer/a64` hold the assemblers, the ISA bits the composer forms for
  and the calling convention generated code keeps. Its runtime is two
  packages, as a card's is `backend` and `engine/device/gpu`:
  `engine/device/cpu` is the call boundary (the trampolines), the
  executable memory, the probe of the running CPU and the worker pool;
  `engine/device/cpudev`, beside `engine/device/gpu`, is the CPU device
  itself -- its forms, scratch, history and counters. The composer
  imports nothing under `engine/` (`composer.LowersCPU` takes the ISA the
  probe states); the device imports the composer and the call boundary.
  The call boundary cannot be the device's package: the composer's own
  gates execute what it forms through it, and a device that composes would
  make each of them an import cycle.
- **Nothing of the CPU in the instance.** `engine/model` holds no CPU
  form, scratch, counter or call: renaming the instance's CPU state in
  place is the same design under another name.

## The device, as the card's

    // engine/device/cpudev
    type Device struct { ... }           // the pool, the registry, the placed blocks, Stats
    func Open(opts ...Option) (*Device, error)
    func (d *Device) Attach() *View      // a session's view: its pages, as gpu.GPU.Attach
    func (v *View) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool
    func (v *View) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool
    func (v *View) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool
    func (d *Device) Stats() Stats

`PrepLayer` composes the plan's graph for the CPU and resolves every role
the forms bind to an address in the weights it is handed, once: a call
binds the rows, the rotary tables and the session's pages, never looks a
weight up. The mapped kernels and their call words (today's `cpuCode`,
`cpuArg`, `cpuBlockForm`) are the device's.

A call word names where its value comes from as data: the call's own
buffers, its rows and residual streams, the session's pages and recurrent
state, the rotary tables, or a weight by role. The call record the device
binds them from holds addresses and words only -- no layer, page pool or
session of the model's -- and a weight is a slot the form states once and
a block's page-in fills (`Model.pageIn` re-points a block's tensors at the
frame it landed in), never a role looked up at a call.

A block's addresses hold while it is placed, as a card's upload does. The
host frames the pager fills are the CPU device's memory as VRAM is the
card's: the page budget is the CPU device's budget, a block paged in is a
block the CPU device places, and one paged out is released from it (the
card reads a block it uploads from those frames and then holds its own). The
matrices reach the device by role, as the vectors do
(`nn.LayerWeights.Vectors`), so neither device reads a named weight field.

## Steps, each gated on its own

0. The codegen half into the composer: the arm64 assembler, the ISA
   description and the call's argument layout move out of
   `engine/device/cpu` (main's emitters there are deleted, not moved:
   #123); the probe of the running CPU stays the runtime's and hands the
   composer a device description, as a card's runtime does. Gates: the
   composer's CPU suite on the Xeon and on the M4, and
   `cpu.TestNoAssemblyBesidesTheTrampolines`.
1. The CPU device package (`engine/device/cpudev`): built from what the card is handed -- a
   block's graph and tables (`nn.LayerPlan`), its weights by role
   (`nn.LayerWeights`, the vector and matrix tables), the input rows',
   head's and sampler's graphs (`nn.Head`) -- composing its own forms for
   the CPU with the composer, its forms, scratch, worker pool and
   counters its own, its `Stats` the card's shape. The runtime moves out
   of `engine/model` (`cpuserve.go`, `cpuembed.go`, `cpuhead.go`,
   `cpukvconv.go`, `cpumtp.go`, `cpupool.go`, `cpustreams.go`,
   `cputables.go`) into it; nothing is kept beside it. Gates: the composed
   CPU suite (blocks, rows, features, embeddings, experts, census) and
   `TestDecodeDoesNotAllocate`.
2. Its history: the pages its blocks read are the CPU device's own, as
   the card's are its own, a session's view of them as `gpu.GPU.Attach`
   gives a session its view of the card; `MigrateKV` and `MigrateRec`
   move a block's history between it and any other device. Gates: the
   relocation and paging gates of the five principles.
3. The instance holds devices: a device per block, `onDev` gone, each
   run of consecutive blocks one call on its device, a decline placing
   the block on another device. Gates: as 2, plus the seam gates
   (`TestBatchSeamMovesCarryEveryRow*`, `TestDecisionRelocatesMidPrompt`).
