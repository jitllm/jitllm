# Stated blocks: the converter states what a family adds

A vision tower is a program the converter writes (`jlm.Tower`), lowered by
`model` with no switch on the family, and so is every text block: the
converter states each block's graph (`jlm.Config.Blocks`) from its library
file, and the front end (`model/front.go`) lowers the stated graph and builds
none from config fields. Where a family adds a feature -- a bank of residual
checkpoints, gated hyper-connections, a block selection -- it is a statement
in the family's library file, not code named for it. A feature with one
family behind it is that family's code whatever it is named (AGENTS.md, the
five principles; the user's ruling on renames). This is the design that took
it out.

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

`model` lowers each stated graph; the family's end-to-end gates (matches its
reference, features load-bearing, every device, relocation, paging,
allocation) hold it. While the front end still built blocks from the config,
a gate stated each fixture both ways and compared the graphs op for op; that
builder and its comparison are gone. A violation is a rewrite of the stated
program, or the model restated at conversion: its source converted again
with the violated config or tensors (`convert.WithStatedSource`), the
library stating the violated model's graphs -- never a fault knob in
`model`.

## Containers already converted

The statement is an optional tail of the config section. A container
without it states no block's graph, and Open refuses it naming the
conversion: only conversion states a graph, from the source file, and a v31
container records no architecture an upgrade could state one by.

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
