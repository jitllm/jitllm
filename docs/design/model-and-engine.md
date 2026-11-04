# The model outside the engine

Stated by the user: *"model should live outside of engine ... engine manages
model instance not model itself"*. A model is the weights, loaded once; the
engine makes instances of it, runs sessions on them and schedules them
(AGENTS.md, "The shape"). Today both live in `engine/model`, one package of
~32,000 lines in which `Model` holds the engine's state beside its own and
the instance reaches into the model's fields.

## What each side owns

**The model** (`model/`, a sibling of `engine/`, importing nothing under
`engine/`):

- the container and its pager (`jlm.File`, block and expert pages, the
  bindings a page-in re-points), the config (`Config`), the vocabulary
  and tokenizer, the weights by role (`layer`, `tensor`, the vectors);
- the front end: every graph a block, the input rows, the head and the
  sampler state (`front*.go`, `Composing`, the inventory), and the
  rotaries' tables the graphs turn by;
- the text formats a prompt is written in: the chat template
  (`chat.go`) and the tool syntaxes, grammars and streams (`tool*.go`);
- what an input is decoded from (`wav.go`, the image preprocessing the
  tower reads).

**The engine** (`engine/instance`, beside `engine/device/*`):

- `Instance`, `Session`, the `Scheduler` and `Pool`, `StepRuns`, the
  `Batcher`, prefill and decode, speculation, sampling;
- every session's history and its stores (`kv*.go`, `memstore.go`, the
  prefix cache), placement, relocation, paging policy and parking
  (`placeset.go`, `moveblocks.go`, `park.go`, `residency.go`,
  `seamtune.go`);
- the state `Model` holds for its instances today: the instance pool
  (`insts`), the history pools and arenas (`kvPool`, `kvArenas`), the CPU
  device's forms (`cpuForms`), the step and prompt buffers, the devices'
  owners and forgetters, the expert I/O counters. One engine record a
  loaded model, made when the engine takes it.

The CPU device's runners (`cpu*.go`) go to `engine/device/cpudev`
(docs/design/cpu-device.md), not to the instance.

## The boundary

The instance reads the model through methods, never its fields: a
block's graph and its weights by role (what `nn.LayerPlan` and
`nn.LayerWeights` already carry to a card), a page-in of a block and of a
block's selected experts, the config's facts, the front's graphs, the
vocabulary. Each is read-only: two engines may hold one model. Options
split by who reads them: an option that shapes the weights or the front
(the tokenizer, the chat template, the page budget's container side) is
the model's; one that shapes an instance (placement, history format,
speculation, chunks, devices) is the engine's.

## Steps, each gated on its own

1. The engine's state off `Model`, still in one package: a per-model
   engine record holding everything listed above, `Model` keeping the
   model's alone. Gates: the model suite slices, `TestTwoPagedModels*`,
   the server's host batch.
2. The model's surface: every read the instance makes of `Model`,
   `layer` or `tensor` goes through a method of the model's; the field
   accesses in the instance's files go to zero (a source gate counts
   them).
3. The move: the model's files to `model/`, the instance's to
   `engine/instance`, their tests with them; `server` and `cmd/jitllm`
   import both. A source gate holds that `model/` imports nothing under
   `engine/`, as the composer's does.
4. The options split, and `model.Open` returning a model no instance
   has been made of; `engine/instance.New(m, opts...)` makes one.
