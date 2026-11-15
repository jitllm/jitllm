# model.json: the converter's family tables as data

A design, not a rule. AGENTS.md RULE 7 governs until this lands.

Stated by the user: *"we will turn all that into json files, so even
converter ain't hard coded"*, and *"each model will have model.json just
like tokenizer.json"*.

## What it replaces

Past the converter nothing holds a family (`srcgate.TestNoFamilyCode`). The
converter still reads each source through Go written per family:

| table | where | entries |
|---|---|---|
| GGUF architecture names | `convert.archOf` | 70 names onto 48 `configOf` cases |
| safetensors classes | `hfArchTable` | 32 classes, each a name map, a config reader and hooks |
| projectors | `convert.projectorOf` and each `visionOf` case | 19 |
| name ambiguities | `retarget` | a role per architecture |

Each safetensors entry is the same five things: a tensor-name-to-role map,
a config reader, and up to three hooks (`split`, `gather`, `fixup`) that
reshape bytes on the way in. The hooks are a small vocabulary used many
times over (`kimiSqueezeConv1d` six times, `deepseekSplitKVB` three, a
`(1 + w)` norm fold in two families), and the config readers mostly copy a
key into a field with a default.

`model.json` states those five things as data. The vocabulary of
transforms stays Go, closed and gated, the way the composer's ops are: a
model.json selects and parameterises transforms; it does not carry code.

## Where a model.json comes from

In order of authority, as RULE 7m orders a tokenizer's:

1. The caller: `jitllm convert -model-json FILE`.
2. The model's own directory, beside `config.json` and `tokenizer.json`.
3. The converter's library, embedded (`convert/models/*.json`) and keyed by
   the class in `config.json`'s `architectures` or the GGUF's
   `general.architecture`. The library is RULE 7's list, as data; adding a
   file is still the user's decision and still owes its fixture and golden.

## The schema

`convert/models/qwen3_5.json` is the first file and the reference for the
schema; `convert/modeljson.go` reads it. Its sections:

- `classes`: the `architectures` names it converts.
- `config.root`: where paths read from (`text_config` for a model with a
  tower); `$.key` reads the file's top.
- `config.arch` and `config.keys`: the graph `configOf` builds, by its GGUF
  name, and each GGUF key it reads (without the architecture's prefix) as a
  typed expression, `{"uint": ...}`, `{"float": ...}`, `{"int32s": ...}` or `{"bool": ...}`,
  over config.json. `configOf` stays the one reader of what a key means, as it
  is for Kimi-K3's safetensors entry.
- `config.vocab`, `config.head` (`tensor`, `tied`) and `config.layers` (a
  list key and what each of its values is, `recurrent` or `attention`): what
  `configOf` reads off a GGUF's tensor table, stated.
- `config.refuse`: `{"if": expr, "why": text}`, each a graph this engine does
  not run, refused with `ErrNotImplemented`.
- `tensors.model`, `tensors.block_prefix`, `tensors.block`: each source name
  onto a role by the role's GGUF tensor name (`roleOf`, one vocabulary for
  both formats), with `do`, the transforms it takes in order.
- `tensors.gather`: a role built from several sources a block by a transform.
- `tensors.experts`: a mixture's per-expert matrices, the prefix before an
  expert's index and each matrix's bank by its GGUF name, stacked one bank a role.
- `tensors.ignore`, `tensors.ignore_prefix`: what is named and not carried.

The expressions are arithmetic, comparisons, `in`, `&&`, `||`, `!` and
`a ?? b` (b where a is absent) over paths. An absent key is an error naming
it unless the file writes the class's default with `??`; the reader's own
default is another class's (RULE 7). There are no loops and no calls, and the
nesting is bounded (`FuzzModelExpr`).

A GGUF source is to use the same schema (step 3 below): its tensor names are
already llama.cpp's, which `roles.go` maps once for every family, so a GGUF
model.json is mostly the config section and the `retarget` it needs.

## What stays Go

- The readers (GGUF, safetensors, by range), the container writer, quantisation.
- The role list and each role's shape rule (`hfCheckShape`), driven by roles,
  not by families.
- The transform vocabulary (`convert/mjtransforms.go`), each its own named
  transform with its own gate (`TestModelJSONTransforms`). Built: `plus_one`,
  `neg_exp`, `squeeze` and the gather `interleave_groups` (qwen3next's beta
  and alpha). The Go table's hooks are a handful more shapes, to move with
  their classes: `repeat_per_head` (KDA's decay), `concat_rows`
  (`fuseOrder`), `split_rows` (`fused`), `split_kv_b` (MLA),
  `permute_head_pairs` (llama's rotary order), `stack_experts`,
  `retype_ternary` (BitNet), `gather_product` (Kimi-K3's residual attention).
  A new family that needs a new shape adds a transform, gated against its
  reference, not a family file.
- The expression evaluator (`convert/modelexpr.go`).

## How it lands

1. The interpreter, and Qwen3.5 as its first file, gated by transformers'
   own class on a fixture (`synth-hf-qwen35`, scripts/hfgold.py) and on
   Qwen3.8-27B against the Go entry it replaced, which never landed.
2. Each safetensors class moves to a file. Gate per entry: the container the
   file converts is byte-identical to the one the Go entry converted, on that
   entry's fixture (`engine.TestSafetensorsMatchTransformers`' fixtures); then
   the Go entry is deleted.
3. The GGUF names, then the projectors' staters, the same way.
4. `srcgate.TestNoFamilyCode` extends to `convert/`: no architecture constant
   or family name in its Go. The `Arch` codes of pre-v31 containers stay, as
   the opaque numbers `jlm.Stating` already treats them as.

## Defaults phase 1 took

Each is a choice for the user and easy to reverse:

- The library is embedded in the binary (`go:embed`), so a release converts
  what it lists with no file beside it.
- Derived keys and refusals use the expression language above.
- A model.json in the model's directory or repository is read without a flag:
  it is data over a closed vocabulary, so the risk is a wrong model, not code.
