# Stated blocks: the converter states what a family adds

A vision tower is a program the converter writes (`jlm.Tower`), lowered by
`model` with no switch on the family. The text model is not there yet: the
front end (`model/front.go`) builds each block's graph from config fields,
and where a family adds a feature -- a bank of residual checkpoints, gated
hyper-connections, a block selection -- the front end, the plan and the
devices carry code named for it (`model/k3.go`, `model/q4e.go`,
`nn.K3Plan`, `nn.LayerPlan.MSA()`). A feature with one family behind it is
that family's code whatever it is named (AGENTS.md, the five principles; the
user's ruling on renames). This is the design that takes it out.

## What is stated

The converter states, per block, everything a family adds, in the op set the
composer already reads. Nothing new is a "feature language": a stated piece
is a program of ordinary ops -- a norm, a projection, an expression, a
reduction, a slice, a concatenation, a multiply-add -- over named inputs,
and its weights are bindings by role.

1. **Programs.** `jlm.Program` (the tower's) grows the text ops it lacks
   (`TOpReduce`, `TOpConcat`, `TOpSlice`, `TOpMap`, `TOpHeadNorm`), and
   `model.LowerProgram` lowers them one to one into `graph` ops.
2. **Streams.** A residual of several streams (Gemma 3n's AltUp, Q4E's
   hyper-connections, Kimi-K3's checkpoint bank) is stated as its count and,
   per block, what each sublayer reads (a program over the streams) and what
   the block writes back (a stream set to a value: K3's checkpoint). The
   front end lowers the reads and writes; it does not know which family
   chose them.
3. **Bindings.** Each program names its weights by role with the shape it
   wants; the loader reads every stated binding generically, replacing the
   family loaders (`loadK3`, `loadQ4E`, the two indexer loaders) and their
   shape checks.
4. **Resources.** What a device sizes -- the stream count, a latent width, a
   cache row's extra head, a gathered key count -- is derived from the
   stated programs where it can be and stated beside them where it cannot.
   `nn.LayerPlan` carries the derived numbers, never a family's plan.

## Lowering and gates

`model` lowers a stated piece where the front end built the family's code
before, and the two must agree before the old code goes: a gate states each
fixture both ways and compares the graphs op for op (`graph` canonical
order), then the family's end-to-end gates (matches its reference, features
load-bearing, every device, relocation, paging, allocation) run on the
stated path alone. A violation is a rewrite of the stated program, never a
fault knob in `model`.

## Containers already converted

The statement is a new optional tail of the config section. A container
without it is upgraded in place by `jitllm upgrade`, which states the
programs from the config and tensors the container already holds -- the
family interpretation lives in the converter and the upgrade, nowhere else.
A container that should state a piece and does not is refused with the
upgrade command, as one written before activations were stated is.

## Order

Each step lands working, its family path deleted once covered:

1. The program ops and their lowering, exercised by the first slice.
2. Kimi-K3's checkpoint bank: the streams statement, the bank mixes as
   programs, the checkpoint writes; then its latent mixture and output gate
   as bindings; `model/k3.go`, `engine/k3run.go` and `nn.K3Plan` go.
3. Q4E's hyper-connections and n-gram streams, the prediction's collapse:
   `model/q4e.go`, `engine/q4e*.go` go.
4. The block selections (MiniMax-M3's, Q4E's): scorer and selection as
   programs over the indexer's stated bindings, the history layout and the
   gathered capacity derived; `MSA()`, `QSA()` and the card's selection
   branches go.
5. What config still reads off the architecture code (#151), stated the
   same way; the srcgate family gate (`TestNoFamilyCode`) lands green.
