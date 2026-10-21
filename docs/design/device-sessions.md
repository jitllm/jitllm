# Device sessions: a tier holds one MODEL, a session holds one SEQUENCE

**Status: built through step 3, with the capacity and the release made the
session's too, and sessions stepping on one device at once (last section,
"Sessions at once").** A step ACROSS sessions (`model.StepRuns`, below) still
runs as one submission: the server's step loop (`server/batch.go`) makes every
generate on a session wholly on one device a row of its model's shared decode
steps.

The user's framing, which is the requirement and not a preference: *"we can't
refuse, jitllm is an inference os multimodel multi devices multi intances"*.
Refusing a second State would make the wrong answer honest; it would not make the
engine an inference OS. AGENTS.md's Scope section already names the missing piece
-- *"What is missing is the PLURAL"* -- and this is what the plural costs.

## The bug, measured

tinyllama-1.1b-q3_K_M, two `model.State`s attached to one `tier.GPU`, 22 blocks
placed each, six tokens interleaved A,B,A,B... against the same two sequences run
alone:

    interleaved on ONE tier:  NMSE  A = 2.110e-01   B = 3.587e-04
    greedy:                   A got 505, want 6231      B got 388, want 388

**A's token is wrong and nothing reports anything.** Not drift -- 2.1e-01 is a
different answer. B looks nearly right only because it wrote last.

Two more, from the same afternoon:

    concurrent SetDevice   state 0 -> 1 block, state 1 -> 1, state 2 -> 22,
                           state 3 -> 1     (serially: 22 each)
    placement DURING a forward
                           "blk.1.attn_q.weight is in the device layout and no
                            host reader accepted it"

The first is silent: correct answers, no acceleration, no signal. RULE 10's
shape. The second at least fails loudly.

## Why: the tier holds one sequence's state and nothing says so

    jit_old/gpu/tier/layer.go:39    kc, vc backend.Buf  // on the LAYER, keyed by
                                                    // block index
    engine/nn/device.go:331            Layers(lo, hi, pos, n, x, cs, head) bool

`Layers` takes a POSITION and carries no state identity, and nothing anywhere in
`jit_old/gpu/tier` keys device state by State or session. So two sequences write the
same cache at the same positions. The scratch has the same shape: `g.bs`,
`g.vbs` and `g.bbs[w]` are one set per DEVICE, and a concurrent forward clobbers
another's intermediates.

★ **AND THE CPU TIER ALREADY HAS THE RIGHT SPLIT, WHICH IS WHY THIS IS A GAP AND
NOT A DESIGN.** `model.State` owns its KV arrays and its scratch; `model.Model`
owns the weights. Every State on the host is independent for free. The device
tier grew from the other end -- one device, one model, one caller -- and the
split was never made.

## The split

    PER MODEL, shared, correct today     PER SEQUENCE, must not be shared
    ---------------------------------   --------------------------------
    packed weights (g.res)               the KV cache (kc, vc)
    norms, biases, router                blockScratch: x, x2, h, q, k, v,
    compiled kernels (g.kerns)           att, prob, g, u, act, logits, ...
    the unpack staging                   the capture graph
    the packed arena                     the position cursor

## The shape: a session-scoped VIEW, not a wider interface

`nn.LayerDevice` does NOT grow a session parameter. `model.State` obtains its
device by a type assertion (`ld, ok := d.(nn.LayerDevice)`, engine/model/forward.go),
so a tier can hand back a view that implements the same interface and closes
over its own per-sequence state:

    tier.GPU.Attach() nn.LayerDevice    // one session; Close releases it
    State.SetDeviceLayers               // uses Attach when the device offers it

Everything model-resident is reached through the shared tier; everything
per-sequence lives on the view. A single-State caller gets exactly today's
behaviour with one session, which is what keeps every existing gate meaningful.

★ **THE INTERFACE STAYING STILL IS THE POINT.** Widening `Layers` would touch
`nn`, `model`, both fake devices and every test that builds one, for a change
whose entire content is "who owns this buffer". The view costs one method on
`tier.GPU` and one line in `SetDeviceLayers`.

## What it costs in VRAM, which is the part that decides the budget

KV per block is `(MaxSeq+1) * NKVHead * HeadDim * 4 * 2` bytes, charged per
SESSION rather than per block. tinyllama at 22 blocks on the 3050 Ti is the row
to take before building: the card has 3.81 GiB with ~2.40 free, so the number of
concurrent sessions a device can hold is `(free - weights) / per-session KV`, and
that is a NEW budget dimension `Slot.Bytes` does not express today.

**Unmeasured, and it is the first thing to measure**: if two sessions do not fit
on this card at this MaxSeq, the multi-instance answer on one GPU is "fewer
blocks each", not "more sessions", and the placement policy has to know that
before the feature is worth having.

## Order of work, each landable and gated on its own

1. **KV per session.** Fixes the wrong answer. `TestTwoStatesOnOneTierShareAKVCache`
   goes green. Scratch stays shared, so concurrent forwards are still wrong --
   which means step 1 must land with the concurrency gate still skipped, and say
   so rather than implying it is fixed.
2. **Scratch per session.** Fixes concurrent forwards.
   `TestConcurrentMultiDeviceDoesNotLeak` goes green. Built as lanes, held by a
   call rather than owned by a session ("Sessions at once" below).
3. **Placement under a lock.** Fixes `SetDevice` racing itself -- the 1-block-vs-22
   result. A per-`GPU` placement mutex held across a State's whole offer
   sequence; today each `PrepLayer` takes and releases `g.mu`, so the sequence is
   not atomic.
4. **The budget dimension.** Sessions per device, and what the placement policy
   does when they do not fit. Measured: one lane per session stepping beside
   another ("Sessions at once"); the policy is not built -- a lane that does
   not fit makes the sessions take turns.

## Gates

    model.TestTwoStatesOnOneTierShareAKVCache     RED today; step 1 turns it green.
                                                  Interleaved A,B,A,B against each
                                                  sequence alone, NMSE bound 1e-3.
                                                  Against a violation (KV shared
                                                  again) it reads 2.1e-01.
    model.TestConcurrentMultiDeviceDoesNotLeak    step 2. Four States, every
                                                  device, RSS/fds/threads/goroutines
                                                  flat across cycles.
    model.TestConcurrentPlacementSerialForward    step 3. Must assert the block
                                                  COUNT is the same as serial
                                                  placement -- it passes today
                                                  while placing 1 of 22, which is
                                                  exactly the gate that proves
                                                  nothing until it checks the
                                                  number.

★ **AND GATE 3 IS THE ONE TO WRITE FIRST, BECAUSE IT CURRENTLY PASSES WHILE
BROKEN.** It asserts that forwards succeed; it does not assert that placement
achieved anything. A test that goes green on 1 block where 22 were possible is
RULE 10's "check the configuration under test was actually SELECTED", and it is
already in the tree in that state.

## What this does NOT change

The single-State path, which is everything the engine ships and every board row
quoted in AGENTS.md. One session is today's behaviour, and the view exists so
that stays true by construction rather than by care.

## A session's capacity and its release are its own

Keying the KV cache by session was not enough, because two things around it
were still the device's:

    the CAPACITY   kvCap, one number per device, baked into the scratch's
                   kernels (K's transposed stride, the score row, the
                   accumulate). A growth regrew EVERY session's caches, so a
                   batch's nseq*maxSeq rows -- or one long conversation --
                   tripled everyone's history on the card.
    the RELEASE    ReleaseLayers freed the block and every session's history on
                   it, and the route with it. One session that could not grow
                   and brought a block home took that block off the card for
                   every other session of the model; the next one to run read a
                   route that was gone.

Both are the session's now. The capacity is gone rather than moved: the
history is paged (docs/design/device-kv-paging.md), so no kernel bakes a
length, a session's growth takes pages for its own sequences alone, and a trim
gives only its own back. Which session's pages a call reads is an argument of
the call (its session id), and the captured graphs are keyed by it; every
session shares one scratch, whose bound only grows to
the longest context any session asked for. (Before paging, each capacity had
its own scratch, parked while nobody was current at it.)

A session's release (`GPU.ReleaseLayers` through its view) drops its own history
and leaves the block, and its route, while another session has history there
(`devTier.leaveLayers`); only the last one out frees the weights. For the same
reason a block another session uses does not move between cards: the spill and
`PrepLayerOn` refuse, and the router's `PrepLayer` offers an already-placed block
only its own card -- a session that card cannot hold runs the block on the host.

The gates:

    model.TestABatchLeavesAnotherSessionAlone   a single sequence A and a
                   three-row batch B on one tier. With room, B's rows leave A's
                   history byte for byte (HeldBytes) and A's logits identical to
                   A alone; with room for half of B, B takes 8 of 16 blocks home
                   for itself and A keeps all 16. Against a growth over every
                   session: A's history 16.8 -> 50.4 MB. Against a release that
                   takes the whole block: A's placement 16 -> 15 blocks and its
                   next token fails.
    tier.TestSessionCapacitiesAreTheirOwn       two sessions at two capacities
                   switching and detaching: each switch selects its scratch, a
                   detached capacity's scratch goes, and every fake buffer comes
                   back at Close, none freed twice. Against a parked scratch never
                   freed: "1 parked scratch(es) are still held".

The scratch is charged to the budget the blocks use, and the prompt chunk's
and the ragged step's are reserved when the first block is placed
(`jit_old/gpu/tier/scratch.go`; `docs/engineering-history/placement.md` 15x). What
is not done: a growth commits the caches before it rebuilds the scratch, so a
scratch that fails to build after them leaves a session's caches and kernels at
different capacities.

## A hybrid's recurrent state across sessions

A step across sessions (`model.Step` -> `gpuSession.LayersSessions`) runs one
row per session in one submission, and a linear block's kernels index ONE state
buffer by row. Each session used to keep its recurrent pair in buffers of its
own, so the step refused a hybrid outright ("its recurrent state is per
session"). Copying each row's state into a batch buffer and back is ~2 MB per
row per block per step, which is the whole of what the step saves.

So a block's states are one POOL (`tier/recpool.go`): a pair of buffers whose
SLOTS are sequences' states. A session holds a SEAT -- a run of slots, one per
row of its batch -- at the same slots in every linear block's pool on the
device, so one per-call descriptor (`kernels.RecDescWords`, staged in
`bs.dRows` beside the real row count) tells every recurrent kernel where each
row's state is, in the step's own block range, without a write per block. The
solo decode, a prompt chunk, a session's own batch and a step across sessions
all read the descriptor; only what is in it differs.

    the delta rule    read at the row's IN slot, written at its OUT slot
    the window        read and written at the IN slot, both halves
    the shared form   OUT is the row's place in recNext; CopySlots takes it
                      home to the IN slot

The pool stays double-buffered, because RULE 13 does not bend: a kernel reads
one half and writes the other. Which half is current is the SESSION's, per block
(`recPair.cur`), since sessions step apart. A step across sessions reads one
half for every row, so `alignRec` first copies the minority's states to the
half most of the rows read -- once, when the company of the step changes, not
while the same sessions keep stepping together. The graph key carries the
parity of EVERY linear block in the range (`parityMask`), since a session that
stepped apart can read a half its neighbours do not.

The pools grow when a session takes a seat past the last slot and shrink when
the last-seated session leaves (`recResize`, carrying every other seat's state
across); a session leaving from the middle frees its slots for the next one.
`recTidy` gives a pool no session holds, the seats of sessions that hold no
block, and the shared form's scratch back to the budget, so the last session
out returns every byte.

Nothing on an arrival or a departure goes through the host. The carry, and
`alignRec`'s move between halves, are `backend.Device.Copy` -- cuMemcpyDtoD,
vkCmdCopyBuffer on Vulkan's one-shot command buffer, a Metal blit -- one copy
per run of adjacent seats. It was a read of every pool to the host and a write
back, because `backend.Buf` had no device-to-device copy: on Qwen3.5-0.8B,
two arrivals and two departures read 242 MB home and wrote it back (doubled
form), where they now read nothing. A resized pool zeroes from the host only
the slots no carry fills, rather than the whole pool before the carry
overwrote most of it. And `MigrateRec` brings home only its seat: a buffer
reads from its start, so a seat above slot 0 used to bring every other
session's state below it home too; it is copied to a buffer of its own first.

Why a copy in the backend and not the generated `kernels.CopySlots`: the
carry is a contiguous run of bytes between two buffers outside any session,
which every runtime has a primitive for. CopySlots would compile a kernel per
seat count and stage a descriptor from the host for every carry, and a
whole-pool launch of 128-thread groups passes the Iris Xe's
maxComputeWorkGroupCount of 65535 at 32 one-megabyte slots. CopySlots stays where a device copy cannot go: inside the step's own
submission, bringing the shared form's next state home.

The gates:

    model.TestStepAcrossSessionsHybridMatchesEachAlone   Qwen3.5-0.8B, three
                   sessions, eight teacher-forced joint steps against each alone,
                   both forms, cuda and vulkan: logit NMSE 7.3e-04 / 7.0e-04 on
                   the laptop. The first session leads by a token, so 18 states
                   (one a linear block) are aligned at the first joint step and
                   none after. Stats.SessionLinear is the selection check.
    model.TestStepAcrossSessionsHybridGateDiscriminates   (jitllmfault) every
                   row on the first row's seat reads 1.23; a row brought to the
                   other half without its state, 0.16.
    model.TestRecurrentPoolKeepsASessionAcrossOthers    a session seated at
                   slot 1 decoding through two arrivals and three departures
                   (four rebuilds of every pool, the last carrying a run that
                   starts above slot 0) is bit-identical to it alone, with
                   Stats.RecCarryBytes moved and backend.HostReads not; with
                   the rebuild carrying nothing (jitllmfault "resize") it parts
                   at token 0, with the carry landing at slot 0 it parts too,
                   and with the old read-and-write-back it reads 288 times.
    tier.TestRecurrentPoolReturnsEveryByte              the ledger, the live
                   buffers and the seats through arrivals, departures from the
                   top and the middle, a reused seat, a batch's growth and two
                   block releases; every byte and buffer back at the end; the
                   device copies carry exactly the seats held, the host reads
                   nothing and uploads only the slots no carry fills.
    tier.TestMigrateRecBringsOnlyItsSeatHome            three sessions on
                   every real device, both forms: each reads back exactly its
                   own state and exactly its seat's bytes.
    backend.TestDeviceCopyMovesExactlyTheRange          ragged offsets and
                   lengths, copies inside one buffer, a source written by an
                   unwaited session; poison outside every range survives, and
                   overlap, overrun and part-word copies are refused.
    backend.TestIndependentRowsStepEachSequenceAlone and the kernel gates
                   beside it run every recurrent kernel at scattered slots of a
                   larger pool, poison in the slots no row names.

## A sequence's rows in one step are a run

A step's rows need not be one a sequence: a prompt chunk rides beside the
decoding rows (`model.Scheduler` within a batch, `model.StepRuns` across
sessions), several consecutive positions of one sequence in one step. Its
attention needs nothing new -- every row's K and V are written before any row
attends. Its recurrence does: the k-th row of the chunk must read the state
the (k-1)-th left, not the state from before the step.

So a ragged step's rows are grouped into RUNS, one a sequence
(`devTier.groupRuns`: a session and a base, `(slot-pos)/seqLen` naming the
sequence's place in its session's batch seat), each run's rows in position
order, and the recurrent descriptor of a ragged step is the run descriptor
(`kernels.RunDescWords`): each run's slots, its first entry in an order list
and its length, the order list itself, and each row's run and place in it.
The run forms of the kernels read it:

    GatedDeltaFused (Runs)  one lane group a run and a state row, looping
                            over the run's rows in position order -- a
                            single thread a row where the device guarantees
                            no 32-lane subgroup (Stats.LinearRowsScalar)
    Conv1dRowsRuns          row r's window is its run's state followed by the
                            run's rows up to r
    Conv1dShiftRuns         a run's next window is the last taps-1 columns
                            of [its state ; its rows]

A decoding sequence is a run of one row, which is the old per-row form. The
rows may stand in any order -- the rows that want logits lead -- since the
order list, not the row index, is what a run walks. A step of S runs compiled
for N > S rows repeats the last run in the surplus entries, whose threads
recompute it and store what it stores. Positions that are not consecutive
within a sequence are refused rather than chained (a gap would step the
state past a token no row ran). The pools stay double-buffered: a run reads
one half and writes the other, as a row did.

A batch's seat grows to the highest sequence a step names, keeping the states
it had (`growSeat` reads them home and writes them back wherever the seat
lands), because a scheduler admits sequences into rows as they free and its
first steps need not name them all.

    model.TestSchedulerMixesPromptsIntoDecodeSteps/hybrid   chunks of four
                   beside decoding rows; Stats.RecChainRows the selection
                   check (558 row-steps chained). Violation (jitllmfault
                   "chain", every row its own run): parts at 10 logits.
    model.TestStepRunsAdmitsAPromptBesideDecoding/hybrid    a third session's
                   prompt in chunks of three across sessions, 6.8e-4 to
                   8.1e-4 against each alone; "chain" reads 0.39-0.40.
    backend.TestRunsStepEachSequenceAlone   the run forms against each
                   sequence's chunk kernels, rows shuffled and runs fewer than
                   rows: bit for bit on ptx; rows that do not chain read 0.07
                   to 0.69.

Kimi-Linear's KDA kernels take the descriptor too, and its full blocks --
latent attention -- run in the ragged step (device-kv-paging.md, slice 5), so
a Kimi-Linear step across sessions reaches both kinds end to end:
`model.TestStepAcrossSessionsLatentMatchesEachAlone` on synth-kimilinear runs
the hybrid checks above beside the latent ones.

## Sessions at once

Two `model.State`s attached to one tier -- the app's chat and an API request
-- step on one device at the same time, each with exactly the answers it gives
alone. The device's tier is three parts (`jit_old/gpu/tier/devsess.go`):

    devShared   everything every session reads: blocks and their residency,
                kernels and their caches, the paged KV pool and its tables, the
                recurrent pools, the budget. Under g.mu.
    devSess     one session's per-call state on the device: the submission's
                arguments and staging, the ragged step, the embedding promise,
                key runs, per-layer inputs, its queue.
    lane        a scratch: every geometry set, the batched widths, the
                partials, the recordings made over them. A call holds one for
                its whole run.

lane0 is the scratch placement builds. A call takes the lane its session ran
in last, lane0, a free clone, or builds a clone of lane0 (charged as
scratch); with no lane free and no room for another it waits for one, so a
budget too small for two scratches serialises the sessions rather than
sending one to the host. A clone borrows lane0's head and owns its logits; a
change of lane0's shape retires every clone, and a session that detaches
gives the free ones back. A vision tower's set is not cloned: a tower's range
runs in lane0, borrowed. Measured (`model.TestSessionsStepAtOnce` logs it), a
second session stepping at once costs one lane: stories15M 17.1 MB on CUDA,
21.3 MB on Vulkan; Llama-3.2-1B 139 MB on CUDA, 155 MB on Vulkan (the prompt
widths' batched scratches are most of it). That is the sessions-per-device
budget dimension: per session, its KV pages plus one lane while it steps
beside another.

A step takes the device's call lock shared (`gpuSession.step`); placement,
releases, migrations and the head take it exclusively. A submission runs on
the session's own queue (`backend.Queued`) with g.mu let go; what it builds on
the way takes g.mu for the build (`subLock`), and its counters are its own
until it completes. What changes something every submission may read -- a
buffer a recording names, a block's weights, a KV page another sequence's
table names, a recurrent pool -- first waits for every submission in flight
(`quiesce`, `jit_old/gpu/tier/inflight.go`), and every wait comes before a
submission's prologue, never between the prologue and the submission: a wait
there let another session's prologue send this session's history home under
descriptors already built (it read 0.45 NMSE). A block a submission is about
to read is held, no victim to another session's page-in (`holdPages`); a KV
page released while submissions are in flight is fenced until they complete.
A submission over what is the device's -- a tower, a linear block's recurrent
pool, a streamed block, a DeepSeek V4 block, a step over several sessions'
rows, history streamed from home -- runs alone.

Gates, each run against a violation:

    model.TestSessionsStepAtOnce     stories15M on cuda:0, vulkan:0 (NVIDIA)
                   and vulkan:1 (Iris Xe): two States decode 48 tokens from two
                   goroutines, ids and logits bit-identical to each alone;
                   Stats.SubsBeside > 0 (submissions started while another was
                   in flight). A lock across every submission: SubsBeside 0,
                   fails.
    model.TestSessionsOverlapOnCUDA  the wall: two at once against one alone,
                   median of ten interleaved rounds 1.27-1.29x; with the lock,
                   2.03x, fails at 1.6.
    model.TestSessionsPageAtOnce     every block streamed through two slots,
                   ~170 page-ins under the other session's submissions, ids and
                   logits bit-identical to each alone with every block resident.
                   Without the hold, a block about to be read is a victim: the
                   step falls back and the logits differ.
    model.TestSessionsEvictAtOnce    KV pages sent home while both step, held
                   within the eviction gate's band. The wait moved past the
                   prologue: 1.3e-2 to 2.1e-1, fails.

NVIDIA's Vulkan driver runs one queue's dispatches at a time
(`backend.TestQueuesRunSessionsAtOnce`), so there the gates hold the answers
and the submissions' overlap, not the wall. Metal implements the queues and
is not gated here: the four gates above are to be run on the M4.
