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
| projectors | the tower staters (`convert.towerStater`); the list and `visionOf`'s cases are `convert/projectors.json` | 18 staters |
| name ambiguities | `retarget` | a role per architecture |

Each safetensors entry is the same five things: a tensor-name-to-role map,
a config reader, and up to three hooks (`split`, `gather`, `fixup`) that
reshape bytes on the way in. The hooks are a small vocabulary used many
times over (`kimiSqueezeConv1d` six times, `deepseekSplitKVB` three, a
`(1 + w)` norm fold in two families), and the config readers mostly copy a
key into a field with a default.

`model.json` states those five things as data. What stays Go is generic:
the tensor primitives (`convert/mjtransforms.go`) -- map, reshape,
transpose, slice, concat, stack, repeat and a storage type, each a general
operation that knows no model -- and the readers of the config statements.
A model's recipe (llama's rotary row order, qwen3next's beta/alpha
interleave, a `(1 + w)` norm) is a composition of primitives written in its
model.json, every size an expression over config.json. A Go function that
encodes one model's recipe is family code whatever it is called; a class
that needs an operation the primitives cannot compose gets a new primitive
only if it is a general tensor operation another model could use unchanged,
and otherwise stays in the Go table.

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
  `configOf` reads off a GGUF's tensor table, stated. Every other fact it reads
  there (which blocks route, a shared expert's width) it reads off the
  checkpoint's own block tensors, each under the name the file gives its role,
  an expert's under its bank's, with its shape.
- `config.gating`: the class's router by name (`softmax`, `sigmoid`), written
  as `expert_gating_func`; a config.json `scoring_func` naming another gate is
  refused, since the class computes its own whatever the file says.
- A mixture's blocks carry experts and a router exactly where its dense lead
  and step (`leading_dense_block_count`, `interleave_moe_layer_step`) place
  one; a checkpoint whose weights stand elsewhere is refused.
- The statements `configOf` does not read, each read by generic code after
  it, so the statement decides:
  - `config.activation`: the feed-forward's activation by its transformers
    name, onto the container's activation flags (`mjActivations`: silu,
    swish, gelu_pytorch_tanh, relu2, xielu). Required; another name is refused.
    An activation whose parameters a block carries names their role
    (`mjActivationParams`: xielu, `ffn_xielu`): every block then carries
    them, and a file stating another activation is refused when a block
    does, since the engine runs a block's stored parameters as its
    activation (`mjActivationCover`).
  - `config.rope_scaling`: the rope scaling object (`rope_parameters ??
    rope_scaling`). Null and `default` state nothing; a published method is
    implemented in code once (`mjRopeScaling`): llama3 states its per-pair
    factors as `rope_freqs.weight`, as convert_hf_to_gguf.py writes them
    (gated by `synth-hf-llama3` against transformers' own class). One that is
    not implemented is refused by name.
  - `config.rope_methods`: the methods the class's attention is gated
    against; one outside the list is refused by name. Absent, every
    implemented method is accepted.
  - `config.yarn_log_multiplier`: a number expression, the coefficient of
    ln(factor) the softmax scale takes where a yarn scaling applies (the
    GGUF key `rope.scaling.yarn_log_multiplier`); DeepSeek's is
    `0.1 * mscale_all_dim`.
  - `config.window`: `{"size": expr, "used": expr}`. In use, it is stated as
    `attention.sliding_window`, and a graph whose config does not carry it
    is refused.
  - `config.flags`: named graph features (`mjFlags`:
    `attention_output_gate`, `non_causal`, `qk_norm`), each a boolean onto
    the flag the engine runs it by; a feature the engine does not run is
    named too (`mjFlagsNotImplemented`: `post_norm`), and stated true is
    refused.
  - `config.rotary_layers`: `{"each": expr, "every": expr}`, which layers
    rotate as transformers states it (`no_rope_layers`, 1 rotating, and
    where it is absent every `no_rope_layer_interval`-th layer skipping the
    rotary). The container states the skipping layers as a period, so a list
    no period describes is refused (`mjRotaryLayersOf`).
  - A partial rotary is `rope.dimension_count` among the keys, stated only
    by a file whose class rotates part of each head (Phi-3, GLM-4-MoE,
    Qwen3.5); every other class rotates the whole head whatever
    `partial_rotary_factor` says, so its file leaves the key unread.
- `config.window.layers`: which layers slide, deciding after configOf --
  `list`, each layer's type (`layer_types`), a layer sliding where its type
  is `slides`; where the list is null, `period` (every layer but each
  period'th) or `from` (each layer from it on); with none of them, every
  layer. The container states it as a period (`windowPeriod`), so a pattern
  no period describes is refused.
- `config.defaults`: the class's config defaults, each a key of the root
  that a config.json lacking it takes, as the class's dataclass does. A key
  present as null keeps its null, and the expression maps it where the
  class's `__post_init__` does (`num_key_value_heads ?? (num_attention_heads)`),
  so an explicit null and an absent key read as their class reads them.
- There are no refusals in a file. A refusal is missing support, and it
  lives in converter code, one rule for every model: an activation, a rope
  scaling method, a window or a flag the code cannot express, a tensor no
  role names (an `mlp_bias` checkpoint's biases, a shared expert), a
  mixture some block of which carries no experts, and a config key that
  changes the graph (`mjGraphKeys`: a partial rotary, a window, `clip_qkv`,
  `use_bidirectional_attention`) present with
  another value than its neutral one and not read by the file. A file
  states what its class computes (a router's gate, a whole-head rotary);
  the one rule that refuses a key contradicting a statement is Go's.
- `container`: what the container states for each GGUF architecture the
  file names (`gguf`), read by `configOfFile` for a GGUF and for a
  config.json the file has mapped onto the keys alike (`mjStates`,
  `convert/mjstates.go`). Each statement is a Jinja expression over the
  GGUF's keys, as a recipe reads them (`src['attention.layer_norm_epsilon']
  | default(1e-5)`): `flags` (the `mjFlags` names, each true or false),
  `activation` (an `mjActivations` name, or an `mjNamedActivations` one no
  flag carries: `gelu`, with erf), `revision` (what `StateRevision` states once the geometry is
  read: `act_quant` by name, `rope_scale_swa` a number, and
  `swa_global_first` and `picture_bidir_sliding`, each true or false and
  applied on a windowed model), `pre_tokenizer` (a pre-tokenizer by name,
  the one a file naming none of its own, `default`, splits with; its
  `add_bos` and `ignore_merges` are `convert/pretokenizers.json`'s),
  `rope_layout` (`interleaved` pair by pair or `xd` over the elements, by
  name; the rotary table `stateRopeTable` states lays the sections one
  after another where unstated),
  `norm_eps` and `embd_scale`
  (numbers; `src['embedding_length'] ** 0.5` is the square root exactly),
  `attn_softcap` and `final_softcap`, `rope_base` (the rotary base of the
  layers no window states one for, refused unless positive), and `window`: `size` (0 states no
  window, and nothing else of it applies), a fixed `period` or a `pattern`,
  the period where the file's `attention.sliding_window_pattern` does not
  state one, or `from`, the first layer that slides (every layer from it
  on; none at or past the last, and a first layer between them follows no
  period and is refused), and `rope_base`, the windowed layers' rotary
  base. A fixed period over no window, and a chunked window over none, are
  refused. `logit_div` and `clamp_kqv` (the bound on q, k and v) are
  numbers, 0 for none; an expression over a key the file lacks, with no
  default, is refused, which is how a class whose every file states a key
  holds a file to it: deepseek32's indexer counts are read so, since its
  reference always has one, while deepseek4's default to 0. `latent` states latent attention's
  ranks, `kv` and `q` (0 where the query is projected in one step; a file
  whose keys and values are no latent states `q` alone), and `indexer` the
  indexer selecting the positions attention reads: `heads`, `key_width` and
  `top_k`, each a count, all three 0 for none. A stated `kv` of 0, and an
  indexer missing a count, with a key that is not whole groups of eight or
  narrower than the rotary, or over a key-value latent with no query latent
  are refused (`indexerFits`). `recurrent` is the recurrent layers'
  geometry, each a count: `conv_kernel` (two or more, always stated),
  `groups` (the key heads), `inner`, `state_size` and `heads_v`; a field
  left out is zero, one stated is at least one, and parts that do not divide
  (value heads into the inner width, key heads into the value heads and the
  inner width) are refused. `layers` is each layer's kind off the tensors it
  carries: a list of `{"tensors": [names past "blk.N."], "kind": ...}`
  (`mjLayerKinds`: `attention`, `linear_attention`, `ssd`,
  `ssd_attention`, `short_conv`, `scan`), a layer taking the last entry
  whose tensors it all carries and attention where none; a statement no
  layer meets is refused, and one a reader has already set the kinds for
  does not apply. `rope_sections` is the M-RoPE split of a head's rotary
  pairs between a position's axes, a list of counts (`src['rope.dimension_sections']
  | default([])`; empty states none); one that is not a list of counts, or
  does not sum to the head's rotary pairs, is refused. Besides `src`, an expression may read `layer_tensors`, the
  set of names the layers' tensors take past their "blk.N." prefix
  (`'attn_q_norm.weight' in layer_tensors`).
  `residual_scale` and `moe_step` (the mixture blocks' interleave, at least 1) are
  numbers; `mrope`, true, reads the rotary's sections off
  `rope.dimension_sections`; `whole_rotary: true` states that the class
  turns the whole head, and a file whose rotary is narrower contradicts it.
  `router` is what the class's router does whatever the keys say: `gate`
  (softmax, sigmoid, or sparsemixer, which reads no gating key, ranks
  the raw logits and weighs each pick by sparsemixer),
  `renormalise` (always or never), `unscaled`, `selects` (the experts it
  picks); a key contradicting it is refused. Every file is held to its
  tensors alike: a router over no `expert_count` is refused
  (`statedTensorsAgree`), as is `rope.scaling.finetuned` false where the
  graph states a rotary, and a tensor the stated block graph binds and the
  block does not store (a stated q/k norm with no norm). An expression may read the tensors' names too
  (`'blk.0.attn_q_norm.weight' in tensors`). It replaces `configOfFile`'s
  case for the architecture; one statement per architecture, and an
  expression reading anything but `src`, `layer_tensors` and `tensors` is refused.
- `tensors.model`, `tensors.block_prefix`, `tensors.block`: each source name
  onto a role by the role's GGUF tensor name (`roleOf`, one vocabulary for
  both formats), with `do`, the primitives it takes in order. A transform
  that changes a tensor's shape or type states its result before the tensor
  is laid out (a dry run of the same primitives), so a matrix is still read
  when the writer reaches it. A source may name a list of two or more
  targets instead, each a role and the primitives that cut it from the
  source (Phi-3's fused qkv_proj, `reshape`d to its fused shape and
  `slice`d); the parts are laid out in its place, in the list's order.
- `tensors.gather`: a role built a block from several sources (`from`), each
  entering as the vector of its values, by primitives (`do`) that leave one
  tensor: `stack` or `concat` them, or `map` over them, then shape the result.
  An operand may itself be composed, `{"from": [...], "do": [...]}`, its own
  primitives over its own sources, so each operand is rounded once (Apertus's
  xIELU: softplus of one number, beta plus softplus of another, then the
  four joined). The result is named as the container names the role
  (`blk.N.ffn_xielu`), as the GGUF path names the same tensor.
- `tensors.experts`: a mixture's per-expert matrices, the prefix before an
  expert's index and each matrix's bank by its GGUF name, stacked one bank a role.
- `tensors.ignore`, `tensors.ignore_prefix`: what is named and not carried.

The expressions are arithmetic, comparisons, `in`, `&&`, `||`, `!` and
`a ?? b` (b where a is absent) over paths. An absent key is an error naming
it unless the file writes the class's default with `??`; the reader's own
default is another class's (RULE 7). `a ?? b` with both sides bare paths
says they are one key spelled two ways, and both present and disagreeing is
an error; a parenthesised side states precedence instead (`a ?? (b)` is a,
and b only where a is absent), which is how a default read from another key
is written (`num_key_value_heads ?? (num_attention_heads ?? 32)`). There are
no loops; calls (`exp`, `log`, `sqrt`, `tanh`, `softplus`) are a tensor map's alone; the
nesting is bounded (`FuzzModelExpr`).

The primitives, each a step of a `do` list with exactly one field:

| step | does |
|---|---|
| `{"map": "x + 1"}` | an elementwise expression over the tensors in hand (`x`, `y`, `z`), float32 out |
| `{"reshape": [a, b, -1]}` | a new shape, slowest axis first; one size may be -1; no byte moves |
| `{"transpose": [1, 0, 2]}` | an axis permutation: the result's axis i is the input's `[i]` |
| `{"slice": [axis, start, length]}` | a range of one axis |
| `{"concat": axis}`, `{"stack": axis}` | the tensors in hand joined along an existing axis, or a new one |
| `{"repeat": [axis, times]}` | each index of an axis, times times over in place |
| `{"type": "f32"}` | the storage type, f32 or f16 |

Every size and axis is a number or an expression over config.json. Each
primitive has its definition and its violations in
`TestModelJSONPrimitives`. As composed in the library: a `(1 + w)` norm is
`{"map": "x + 1"}`; a log decay as a rate `{"map": "-exp(x)"}`; torch's
`[channels, 1, taps]` convolution `{"reshape": [-1, "linear_conv_kernel_dim"]}`;
Qwen3.5's beta and alpha, per key head its value heads' rows from each,
`stack` 0, `reshape` `[2, key heads, -1]`, `transpose` `[1, 0, 2]`, `reshape`
`[2 * value heads, -1]`; llama's rotary row order on Mixtral's q and k,
`reshape` `[heads, 2, head_dim / 2, hidden]`, `transpose` `[0, 2, 1, 3]`,
`reshape` back.

A GGUF source is to use the same schema (step 3 below): its tensor names are
already llama.cpp's, which `roles.go` maps once for every family, so a GGUF
model.json is mostly the config section and the `retarget` it needs.

## What stays Go

- The readers (GGUF, safetensors, by range), the container writer, quantisation.
- The role list and each role's shape rule (`hfCheckShape`), driven by roles,
  not by families.
- The tensor primitives (`convert/mjtransforms.go`), generic, each with its
  gate (`TestModelJSONPrimitives`). The Go table's hooks move with their
  classes as compositions: KDA's per-head decay is a repeat, `fuseOrder` a
  concat, `fused` slices, MLA's `kv_b` split slices of a reshape, llama's
  rotary order a reshape and a transpose by the head counts. What the
  primitives cannot compose (BitNet's ternary retype, a quantised source)
  stays in the Go table until a general primitive does it.
- The readers of the statements: the activation names, the published rope
  scaling methods, the named flags, the window.
- The expression evaluator (`convert/modelexpr.go`).

## How it lands

1. The interpreter, and Qwen3.5 as its first file, gated by transformers'
   own class on a fixture (`synth-hf-qwen35`, scripts/hfgold.py) and on
   Qwen3.8-27B against the Go entry it replaced, which never landed.
2. Each safetensors class moves to a file. Gate per entry: the container the
   file converts is byte-identical to the one the Go entry converted, on that
   entry's fixture (`engine.TestSafetensorsMatchTransformers`' fixtures); then
   the Go entry is deleted.
3. The GGUF names, then the projectors' staters, the same way. The
   projector list and what `visionOf` read per projector are
   `convert/projectors.json`: per `clip.projector_type`, the container's
   projector code, the output matrix (tensor and dimension), the tower's
   features (`vision_rope`, `gelu`), the tensors it requires, and its
   geometry -- `patch_size`, `scale`, `merge_of` (the matrix taking
   Scale*Scale patches), `tiles`, `pixels`, `tile`, `window`, `resize`
   -- each an expression over the keys (`src`), the tensors' shapes
   (`shape`) and the tower's patch, picture and merge (`tower`); `mean`,
   `std` and `activation` replace the file's; `when` is the condition an
   entry holds under (`states` what it covers), outside which the file is
   refused, and `as` takes another entry's statements; `transforms` names,
   per phase of conversion (`joined`, `source`, `prepare`, `finish`), the
   tensor transforms the tower's tensors take (`convert.towerTransforms`);
   `taps` says the projector reads taps off the tower's blocks
   (`v.deepstack.N.*`), `blocks` "staged" that the file names its blocks
   by a stage and a place in it (`v.blk.S.I.*`, numbered stage by stage),
   and `hard_tokens` the first id of a token table the tower embeds itself
   (read by the `hard_token_embeddings` transform). The transforms are
   named primitives (`stage_places`, `conv_layouts`, `fold_channel_scales`,
   `point_conv_projection`, ...), each usable by any entry.
   The file also lists the audio towers that convert (`audio`); a tower
   converted alone carries its audio half as a joined one does. Each check is one
   rule for every entry (a merge of 0, a tile range or a pixel range that is
   empty, a window with no pattern).
4. `srcgate.TestNoFamilyCode` extends to `convert/`: no architecture constant
   or family name in its Go. The `Arch` codes of pre-v31 containers stay, as
   the opaque numbers `jlm.Stating` already treats them as.

## Defaults phase 1 took

Each is a choice for the user and easy to reverse:

- The library is embedded in the binary (`go:embed`), so a release converts
  what it lists with no file beside it.
- Derived keys use the expression language above; refusals are code.
- A model.json in the model's directory or repository is read without a flag:
  it is data over closed primitives, so the risk is a wrong model, not code.
