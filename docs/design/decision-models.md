# Decision models

The user's term for a family of models that answer typed questions about a
piece of text (or an image) in one forward pass and generate nothing. Hugging
Face tags them `decision-model`, `system-one`, `systemone`, `jev`,
`calibrated-decisions`, `rlcd`; their pipelines read zero-shot-classification
or text-classification, and several say image-text-to-text because their
backbone is a vision-language model.

This file is the research the build stands on. It was written before any code,
from the model cards, the reference code each publisher ships, TypeSafe's SDK
and llama.cpp's implementation (PR 29818, PR 29831, master at the time of
writing, `tools/server/server-decision.cpp`, `src/models/modern-bert.cpp`,
`src/models/clef.cpp`, `src/models/lfm2.cpp`, `conversion/{bert,lev,lfm2,clef}.py`).
The user decided the family is to be added (recorded in
`docs/design/model-coverage.md`).

## 1. What JEV, System One and `/v1/systemone` are

**Jev is TypeSafe AI's hosted "System One model"** (closed weights, `jev-latest`
on `api.typesafe.ai`). It takes a **state** (any string or JSON value) and a map
of named **questions**, and returns a typed **answer** per question with
probabilities, from a forward pass rather than generated text. "System One" is
the Kahneman framing: the fast, non-deliberative decision, as opposed to an
LLM's System Two.

The wire contract is TypeSafe's OpenAPI (`https://api.typesafe.ai/openapi.json`,
vendored as `typesafe_sdk/_schemas/models.py` in the PyPI `typesafe-sdk` 0.7.2):

    POST /v1/systemone
    request  {"model": str, "state": str|object|array,
              "questions": {id: Question}}                      (>= 1 question)
    Question  {"type": "noul",   "instructions"?: J, "criteria"?: {"true"?: J, "false"?: J}}
            | {"type": "choice", "instructions"?: J, "criteria": {name: J|null}}
            | {"type": "score",  "instructions"?: J, "criteria": [J, ...]}   (level i = index i)
    response {"model": str, "answers": {id: Answer},
              "usage": {"input_tokens": int, "output_tokens": int}}
    Answer    {"type": "noul",   "noul": P(yes)}
            | {"type": "choice", "choice": name, "confidence": c, "probabilities": {name: p}}
            | {"type": "score",  "score": E[level], "confidence": c,
               "legend": {"i": criteria[i]}, "probabilities": {"i": p}}
    422      {"detail": [{"loc": [...], "msg": str, "type": str}]}
    GET /v1/models -> {"models": [{"name", "description", "release_date"}]}

`J` is a string, an object or an array. `noul` is TypeSafe's word for a yes/no
(it reads as "null-or-yes/no"; the answer is one probability). Every open model
below says it "speaks Jev": it implements this endpoint, or its server does.

**ggmlc** is not a model: it is `monatis/ggmlc`, a compiler that lowers PyTorch
models to GGML graphs; its GGUFs (`mys/laya-GGUF`, `mys/kev-*-GGUF`) carry
`general.architecture = ggmlc` and are explicitly not llama.cpp GGUFs. jitllm's
converter does not read them (it reads llama.cpp's own conversions,
`ggml-org/*-GGUF`, and the safetensors releases).

**`ggml-org/convert`** is the automated converter that produced
`ggml-org/Laya-GGUF`, `ggml-org/Clef-Flash-GGUF`, `ggml-org/lev-GGUF`.

## 2. How a decision is read out -- the four mechanisms

Every model maps (state, question) to one score per option, then
`softmax(score / T)` per question with a fitted temperature `T`, then the
answer arithmetic of section 4. They differ in where the scores come from:

| readout | models | the score of option i |
|---|---|---|
| **marker head** (encoder) | Laya, Julia-1, Von | a [MASK] token precedes each option's text; after the encoder, a type embedding is added to every row, a small head (2 pre-norm transformer blocks) runs, and a scorer (LayerNorm, Linear, GELU, Linear to 1) reads each marker row |
| **label-token logits** (decoder) | Lev, d1, OpenJev, Nimble, Decider, SemIf, Rizzo | each option is shown with a one-token code (A, B, ...; yes/no; digits); the next-token logit of that code at the prompt's last position |
| **pointer head** (decoder) | Kev | the hidden state at each option's end token dotted with the question's last-token state, both through a learned projection |
| **joint span head** (decoder) | Clef, Clef-Flash | mean-pooled spans of every question and option over the backbone's last hidden states, a small transformer head over those summaries, all questions in one pass |

Only the first and last need weights beyond the backbone. The label-token
readout is the backbone's own LM head read at a handful of ids, which is why
it covers most of the family with no new architecture.

## 3. The models

### Laya (convaiinnovations/laya; GGUF ggml-org/Laya-GGUF BF16, Q8_0)

- **Backbone**: ModernBERT-large, 28 blocks, d=1024, 16 heads, GeGLU FFN 2624,
  LayerNorm without bias (eps 1e-5), no biases anywhere, rotary everywhere,
  every third layer global (theta 160000) and the rest local (theta 10000,
  symmetric window 128: a token sees 64 either side), layer 0's attention norm
  the identity, a final norm. Tokenizer: GPT-2 byte-level BPE (`modern-bert`
  pre-tokenizer, which jitllm already has), [CLS]=50281, [SEP]=50282,
  [MASK]=50284. Multilingual variant: mmBERT-base (22 blocks, 256k vocab), same
  graph. ModernBERT is a NEW encoder architecture for jitllm: the bert and
  nomic-bert encoders are post-norm with learned or plain rotary.
- **Head** (`rl_common.DecisionModel`): `h = encoder(x) + type_emb[qtype]` on
  every row; two `nn.TransformerEncoderLayer(d, 16 heads, 4d, norm_first=True)`
  (LayerNorm with bias, biased fused qkv and out, ReLU FFN, no positions,
  bidirectional); scorer `Linear(1,d) . GELU(erf) . Linear(d,d) . LayerNorm(d)`
  on the marker rows. An `act_head` exists and is unused for the answer (the
  card says it carries no signal; llama.cpp drops it).
- **Prompt**: `[CLS] "<type> question: <instructions>" [SEP] ([MASK] " <option>")* [SEP] <state> [SEP]`,
  option texts `key: description` (choice), `level i: description` (score),
  `false: ...` / `true: ...` (noul, with defaults "no, the statement does not
  hold" / "yes, the statement holds"); each option cut to 48 tokens, all
  options shrunk evenly when they leave the question under 16 of
  `head_max_len` (192), the question cut to the rest, the state cut to
  `max_len` (512). One sequence per question.
- **Readout**: the scorer at the markers; one forward per question.
- **Calibration**: `temperature` per type (choice 1.637, score 1.251, noul
  1.983) and `temperature_by_options` per (type, option-count band 2 / 3-5 /
  6-10 / 11+), which wins when present.
- **Confidence**: Laya's own code returns `1 - H(p)/log k`; llama.cpp returns
  TypeSafe's formula (section 4). A divergence (RULE 7m), recorded below.
- **llama.cpp**: `modern-bert` arch with `decision.block_count` trailing head
  blocks, `token_types` (3 rows), `cls`/`cls.norm`/`cls.output` the scorer;
  `decision.type = laya`, `decision.max_head_tokens`, temperatures as
  `decision.temperature.<type>[.<band>]`; the prompt is the GGUF's
  `tokenizer.chat_template.systemone`. The graph runs the head three times (one
  per type) because the type is not a graph input; only the asked type is
  needed.

### Clef-Flash (Cloudflare/clef-flash; GGUF ggml-org/Clef-Flash-GGUF Q8_0)

- **Backbone**: Qwen3.5-9B (jitllm's `qwen35`, a hybrid of gated delta rule and
  attention), with its vision tower.
- **Head** (`joint_schema_model.py`, `joint_head.safetensors`): the backbone's
  final hidden states, mean-pooled over each question's instruction span and
  each option's span (the prompt marks them; llama.cpp passes the spans as a
  per-token "decision order": 0 text, 1/2/3 a noul/choice/score question, 4 an
  option), then "routing" blocks and head blocks (`decision.routing_block_count`,
  `decision.block_count`), projections (memory, question, option-question,
  global, option-context, option-lexical), norms and a scorer -- one logit per
  option of every question, jointly, in one pass.
- **Prompt**: a system prompt ("Read the complete state and schema. Decide
  every field jointly...") and the whole schema with JSON keys sorted, choice
  options sorted by key, noul shown true first.
- **llama.cpp**: arch `clef` (PR 29831), text only, one sequence per batch.
- **jitllm**: the arch `clef` converts into qwen35's container
  (`ArchQwen3Next`) with `jlm.DecisionClef`; `decision.*` and `dec.blk.N.*`
  are a role group of their own (`format/jlm/decisionroles.go`) at
  `DenseBlock` with the head block in `Index`, and the container states the
  routing and joint block counts, the head count and the head's LayerNorm
  epsilon (`Config.DecisionRouting`, `DecisionBlocks`, `DecisionHeads`,
  `DecisionEps`). The prompt is built as `encode_record` builds it, piece by
  piece, and one prefill keeps every row's final residual (`State.rowTap`
  under an embedding mode with no pooling, so no output projection runs);
  the head (`engine/model/decideclef.go`) runs on the host's generated
  kernels, or on the device holding the output projection
  (`engine/model/decideclefdev.go`, `jit/gpu/tier/dechead.go`).

### d1-3B (LiquidAI/d1-3B; GGUF LiquidAI/d1-3B-GGUF Q4_K_M/Q8_0/BF16 + mmproj)

- **Backbone**: LFM2.5-VL-3B: LFM2 (short convolutions beside attention; jitllm
  has `lfm2`) and a SigLIP2 NaFlex tower (no projector entry in jitllm yet).
- **Readout**: label-token logits (`decision.type = lfm2-d1`). Codes: the
  option keys themselves when every key is one letter, else A, B, ... (<= 26)
  or 00..99, falling back through a pool, each code required to be one token;
  a group per option of that token and the token of " "+code, scored by the
  max logit in the group. noul reads {yes, Yes, YES} against {no, No, NO};
  score reads the digit keys. noul is shown true first.
- **Prompt**: `<|startoftext|><|im_start|>user\n<state>\n\n\nQUESTION:\n<instructions>`
  + the options block + `<|im_end|>\n<|im_start|>assistant\n`; state may be
  null when images are the state.
- "d1-3B-AD" in the user's table is a file name, not a different release found
  on the Hub; LiquidAI's GGUFs are the publisher's.

### Lev 4B (interfaze-ai/lev; GGUF ggml-org/lev-GGUF)

- **Backbone**: Qwen3.5-4B + a LoRA (r=32, alpha=64, q/k/v/o and MLP), PEFT
  safetensors. Merged at conversion (`scale * B @ A` added to the base weight),
  which is what llama.cpp's converter does and what the reference computes:
  the adapter is static, nothing switches it at run time, so a runtime adapter
  would spend a matmul per projection per token on a constant (house rule:
  precompute at conversion).
- **Readout**: label-token logits ("mode A"). Codes A..Z then AA..ZZ, only
  those that are one token, at most 255. A choice is rendered twice, options in
  forward and in reverse order, and the two softmaxes averaged (cancels
  position bias). A noul is a 9-level scale "0 = certainly no ... 8 = certainly
  yes", read at the first nine codes, `noul = sum_i p_i * i/8`.
- **Mode B** (`mode_b_head.pt`, a candidate-path head for large option sets) is
  not converted by llama.cpp; lev's server routes choice sets over the tokenizer
  limit to it. Not in scope until mode A is done.
- **Prompt**: Qwen chat format, system prompt "You are a System One decision
  model...", `# Evidence`, `# Criterion`, `# Options` / `# Scale`, an empty
  `<think>` block, JSON keys sorted.
- **Calibration**: `calibration.json`, per (type, mode, band) with bands small
  <= 8, mid <= 26, large.
- **Reference server**: `lev serve` (FastAPI): `POST /v1/systemone`,
  `GET /health`, 422 on a malformed question.

### Coordinator additions (named, not all built)

| model | backbone | readout | notes |
|---|---|---|---|
| Kev (jaredpalmer/kev-0.8b/4b/9b) | Qwen3.5 + LoRA r16 | pointer head | state reused across questions |
| Nimble (bespokelabs/Bespoke-Nimble-9B) | Qwen3.5-9B + LoRA r16 | label-token logits; the prompt lists every question | |
| OpenJev (ggml-org/OpenJev-GGUF) | 27B VL | label-token logits, letters A..z | |
| SemIf | any instruct LLM, e.g. Qwen3.5-4B | label-token logits; no checkpoint | a readout mode, not a model |
| Rizzo Flow (Spark-X2.5 1.7B-4B) | via llama.cpp | label-token logits A/B/C | KV reuse across questions |
| Von (wfzyx/von-1.0) | ModernBERT 395M | marker head (OptionMarker) | Laya's graph |
| NanoJev (C-Tianyu/NanoJev) | Qwen3-0.6B | per-type heads (set attention, sigmoid, ordinal) | |

## 4. The answer arithmetic (shared)

Per question, per variant: `p = softmax((s - max s) / T)`; variants (Lev's
reversed order) averaged in the original order. Then:

- noul: `P(true)`; Lev: the expected rating `sum p_i i/8`.
- choice: argmax key, `probabilities`, `confidence`.
- score: `score = sum i p_i`, `legend`, `probabilities`, `confidence`.

Confidence, as llama.cpp implements it ("the ones published by TypeSafe"):
choice `max(0, (p_max - 1/k) / (1 - 1/k))`; score `max(0, 1 - E|i - mode| /
E_uniform|i - (k-1)/2|)`. Laya's own code uses `1 - H(p)/log k` for both.

Temperature lookup: `<type>.<band>` then `<type>` then 1. Bands are 2 / 3-5 /
6-10 / 11+ except Lev (small <= 8, mid <= 26, large).

## 5. What "per word" means in the user's benchmark

The user's table (per-word p50 latency, words processed in 32 s, accuracy,
"centipede names caught", "wrong picks") was not found published. Its shape is
a stream of words, each one a `/v1/systemone` request whose state is the word
(or the word in context) and whose question is a noul ("is this a centipede
name?") or a choice. So:

- **per-word p50**: the median wall time of one request, request to answer.
- **words in 32 s**: requests completed in a 32-second window, serially.
- **accuracy**: argmax answer equal to the label, over the labelled words.
- **caught**: positives answered positive (recall's numerator).
- **wrong picks**: negatives answered positive (false positives).

`dev/decisionbench` reproduces it from a labelled file (one word and its label
per line) against any server speaking `/v1/systemone` -- jitllm's, llama.cpp's,
`lev serve` -- so the same harness measures every engine in the table. Its
correctness (the counting, not the speed) is gated on a small labelled file
against a scripted server.

## 6. llama.cpp's coverage (b11514, the build fetched here)

`llama-server` serves `POST /v1/systemone` when the GGUF carries
`<arch>.decision.type` (`laya`, `julia-1`, `lev`, `openjev`, `kev`, `nimble`,
`pplx-decider`, `clef`, `lfm2-d1`, `lfm2-d1-omni`), with the prompt from the
GGUF's `tokenizer.chat_template.systemone` rendered by its jinja engine and the
temperatures from the metadata. One task per question (one per variant for
Lev), the shared prompt prefix of a request's tasks evaluated once on causal
models. Requires `instructions` (TypeSafe makes it optional: a divergence of
llama.cpp from the protocol, and jitllm follows the protocol, rendering the
question id when it is absent as Clef's code does).

## 7. Where jitllm stands, and the order

What exists: `qwen35` (Lev's and Clef's backbone), `lfm2` (d1's), the GGUF
chat templates carried by name into the container (so `systemone` arrives
already), `tok/jinja` to render it, the prefix cache to reuse a state's rows
across questions, the bert/nomic-bert encoder (on the host when this was
written; section 10 puts every encoder on the device).

Order, by what each unlocks (coordinator's priorities):

1. **The protocol and the label-token readout**: the decision metadata in the
   container (kind, temperatures, head budget), the systemone template rendered
   with `tok/jinja`, the logits read at label ids at the last row, the answer
   arithmetic, `POST /v1/systemone` on the server, `jitllm decide` on the CLI,
   and `dev/decisionbench`. Covers Lev and d1 (text) on every tier the backbone
   already runs on, and SemIf/Rizzo/Nimble/OpenJev as readout modes.
2. **Laya**: ModernBERT as a new encoder architecture to RULE 7's bar, plus the
   marker head and scorer as generated kernels; host first, then the device
   (section 10).
3. **Clef-Flash**: the joint head on qwen35.
4. Kev's pointer head, NanoJev, d1's vision tower.

## 8. Divergences to record (RULE 7m)

| where | llama.cpp | the reference | jitllm |
|---|---|---|---|
| `instructions` absent | refused (400) | TypeSafe: optional; Clef: the question id | the protocol: the id is rendered |
| Laya confidence | TypeSafe's formula | `1 - H(p)/log k` | the reference |
| Laya head per type | runs all three types | runs the asked type | the asked type (same answer, a third of the head's work) |
| Laya head's last block | every row | every row | the marker rows only (keys and values from every row) |
| Laya head heads | the encoder's count | `max(1, d // 64)` | the reference's, stated in the container (`Config.DecisionHeads`); they agree on every published checkpoint |
| Laya sequence | template rendered whole, then re-cut | each piece tokenized alone | the reference; token for token equal to both on the gates' requests |
| Laya GELU | tanh (GeGLU), erf (scorer) | erf in both | the reference: `ActGELUErf` on every tier (host and device); worst logit 1e-6 on the f32 fixture, where the tanh form left 3e-4 |
| Laya state cut | the context | `max_len` (512) | the context: the GGUF does not carry `max_len` |
| Clef confidence | TypeSafe's formula | `systemone_answer`: the chosen option's probability, a score's largest | the reference |
| Clef state cut | the context | `max_length` (16384) less the schema | the Decider's sequence less the schema, as the reference cuts it |

## 9. What was built, and the gates

| model | readout | gates | against the reference | against llama.cpp b11514 |
|---|---|---|---|---|
| d1-3B (Q8_0) | label logits, lfm2 | `TestDecisionMatchesLlamaCpp`, `TestDecisionReadoutIsLoadBearing` (noul shown true first) | not run: the reference is transformers remote code (>=5.14) and was not run here | worst probability 0.021 (f16 kv, llama.cpp's own), 0.020 (f32 kv); prompts token for token |
| Lev (Q8_0 of the merged adapter) | label logits, qwen35 | same, plus two option orders, sorted keys and calibration load-bearing | the graph only: Qwen3.5-4B with the adapter merged, f32 in transformers, on jitllm's prompt ids: raw label logits within 0.3 on a scale of 20 (Q8_0 against BF16); `lev serve`'s own prompt code was not run | worst 0.029 (f16 kv), 0.090 (f32 kv, lev's tone near-tie; see decide_test.go) |
| Clef-Flash (Q8_0) | joint span head, qwen35 | `TestClefMatchesReference`, `TestClefFeaturesAreLoadBearing` (noul true first, sorted choices, the hidden norm, the lexical projection, the type row, the self-attention, the prior, the cosine, the span mean, erf GELU) and every decision gate | the random fixture: prompts, spans and logits within 1.7e-6 of Clef's own code (f32); the release (`clefgold.py real`, f32 from the BF16 safetensors, run on the V100 server's CPU): prompts and spans token for token, logits within 0.076 and 0.156 on a range of -4.8 to 1.3 -- Q8_0 rounding: llama.cpp on the same file sits 0.017 and 0.038 from the reference in probability | the release: worst probability 0.0095 and 0.0079 (f32 kv), 0.017 and 0.010 (f16 kv) on the two requests, prompts token for token (627 and 535); the fixture 2e-5 |
| Laya (BF16) | marker head, ModernBERT | `TestLayaMatchesReference`, `TestLayaFeaturesAreLoadBearing` (window, local base, GeGLU's gated half, type row, head count, erf GELU), `TestLayaPagesWithTheSameAnswer`, `TestDecisionMatchesLlamaCpp` | logits within 1e-6 (random fixture, f32) and 0.027 (checkpoint) of Laya's own code | worst probability 0.0039 |

Features the gates could not show load-bearing, kept as the reference has them:
d1's second code form (" A" beside "A", at most 0.004 on either request), and
lev's sorted keys moved the answers 0.019 against a 0.02 tolerance on the
object state (the gate keeps it at five questions).

Not done, named: Kev's pointer head, NanoJev, Nimble;
lev's mode-B head (option sets past the 255 single-token codes) and its
safetensors adapter merge (the GGUF path reads ggml-org's merged file); the
label pick on the device (section 10: the head's logits come home whole and
the readout's gather, group max and answer arithmetic run on the host's
generated kernels); a decision head of another geometry than its encoder on
the same device (declined by name, section 10); prefix reuse of a request's
state across its questions (a hybrid's recurrent state is sealed only at a
prompt's end, so the qwen35 and lfm2 backbones restore nothing shorter); the
Connect RPC beside the JSON endpoint; images and video in a decision request.

## 10. On the device

Every readout runs on CUDA, Vulkan and Metal, through the one placement path
(`Decider.SetDevice`; the server places a model's Decider and Embedders where
its sessions go, `jitllm decide` and `jitllm embed` take `-devices`).

- **d1 and lev**: their backbones' blocks and head are placed as any State's
  (`State.SetDeviceLayers`); a question's prompt prefills there. The device
  gate found that `State.Reset` left a placed linear block's recurrent state
  on the card, so each question after the first started from the last one's
  summary (d1's second question 0.27 of a logit off on CUDA and Vulkan, its
  first 0.02); Reset now zeroes it there as Retire zeroes a row's
  (`TestResetZeroesTheDeviceRecurrentState`).
- **Encoders** (ModernBERT, Laya's head, BERT, nomic-bert): an encoder's
  blocks are non-causal blocks -- the sequence goes in whole and the keys die
  with the block -- which is what a vision tower's are, so they reach a device
  the way a tower's do: a segment State over the encoder's blocks
  (`engine/model/encdev.go`) offers each one (`offerRange`) with the plan and
  weights read off the encoder's own structs, and the Embedder runs each run
  of placed blocks as one `Layers` call over the sequence's rows and every
  other block on its host kernels. What the tier gained for them, each a
  plan field rather than a model name: a window a row
  (`nn.RowWindowDevice`, ModernBERT's symmetric local window, staged once a
  sequence into the existing window mask), the local rotary table chosen by
  `SWALocal` (`nn.LayerPlan.LocalRope`: ModernBERT's local blocks are all but
  every third from the FIRST, which the period's own expression does not
  say), a block with no attention norm beside an FFN norm (ModernBERT's
  block 0), the norms after the residual adds (`PostResidNorm`, BERT's
  order), and per block on a non-causal set its own MLP width, gating,
  activation and rotary (Laya's ungated ReLU head beside a gated GELU-erf
  encoder, the set grown to the wider MLP). No kernel is new: each is a
  launch of an existing generated one (the window mask, the LayerNorm, the
  activations, the row copy).
- **Clef**: its backbone's blocks are placed as lev's are, and its joint
  head follows the output projection: with the projection on a device the
  prompt's final residual rows go up whole, with the request's pooling rows
  (a span's rows weighted 1/len, one row picking the last position), the
  options' questions, the questions' type rows, the per-question mask and the
  options' lexical means (the output embedding's rows for their tokens, a
  gather the host's generated row decoders do, as the prompt's own token
  embedding is), and one logit per option comes back. The head runs as one
  submission of `kernels.DH*` launches (a batched matrix product either way
  round, LayerNorm and RMSNorm rows, a gather-add, L2 rows, a gathered row
  dot, the scorer's features and the final combination) -- plain f32 kernels,
  one thread an output, the basic form RULE 8 asks for -- over the head's
  matrices uploaded once in f32 (`jit/gpu/tier/dechead.go`). The program is
  compiled and kept per request shape (rows, questions, options; the last
  eight), so a warm request allocates nothing. Measured on the V100 (cards 6
  and 7): the device's head against the host's over the same rows 0.0014
  (CUDA) and 0.0009 (Vulkan) of a probability; every block and the head on
  the card against the host 0.0197 (CUDA) and 0.0060 (Vulkan); the
  violation, the device's copy of the head with its joint score gated off,
  moves the answers 0.60. A seam leaves the head home with the projection.
- **Declined by name**: a non-causal block whose heads or width differ from
  the set it would join (`tier.nonCausalConflict`), which before ran on the
  first block's kernels unasked. Laya's random fixture is the one case (its
  head: 2 heads of 64 beside an encoder's 4 of 32); its head runs on the
  host. EmbeddingGemma's bidirectional decoder blocks stay home too (the
  decoder path plans them causal on a device).
- **The readout**: with the head on the device the label logits come from
  the device's projection; the gather of a question's label ids, a group's
  max (the generated argmax) and the answer arithmetic run on the host's
  generated kernels over those few numbers. Reading back only the label
  logits (an `nn.Head` pick beside `SampleK`) is the lever left.

| gate | what it holds |
|---|---|
| `TestDecisionOnEveryDevice` | each readout's answers with every block on each device, and with half, against the host; a feature removed from the device's plan must move them past the band (d1, lev: NORM pairs for NEOX; laya: every block global) |
| `TestDecisionRelocatesMidPrompt`, `TestDecisionRelocatesBetweenDevices` | blocks onto a device a third into a prompt and home at two thirds (history and recurrent state with them), and a Decider moved host, each device, host; the violation is the recurrent state lost on the way |
| `TestDecisionPagesWithTheSameAnswer`, `TestLayaPagesWithTheSameAnswer` | half the block pages (d1, lev) or one frame (Laya, host and with half the blocks placed): the resident answer bit for bit |
| `TestDecisionDoesNotAllocate` | a warm question's prompt run and readout (Laya's encoder, head and scorer; a decoder's prefill and label pick) at zero engine heap allocations, on the host and with a seam and every block on each device; writing the prompt (the template, the tokenizer) is the request's text work and sits outside the window |
| `TestDecisionMovesBetweenCards` | every block moved card to card (`State.MoveBlocks`, the history and recurrent summaries copied peer to peer) a third into a decoder's prompt and back at two thirds, and every model's between two questions: no prompt row computed again, bit for bit the unmoved run on two cards of one backend; the violation is the recurrent state lost on the way |
| `TestDecisionDistributes` | blocks split across two cards at once, and across the host and two cards, every block where the placement names it, against the host; the violation is the device feature cut |
| `TestDecisionBatchMatchesAlone`, `server.TestSystemOneBatchesConcurrentRequests` | several requests in one `Decider.DecideBatch` (a decoder's prompts as rows of one batch session, more than one in a step) and concurrent `/v1/systemone` requests answered as one batch, each against itself alone; a request that fails its checks fails alone; the violation is prompt chunks at reversed positions |
| `TestLayaMatchesReference`, `TestLayaFeaturesAreLoadBearing` | a device arm each: every block placed against Laya's own code, and each feature removed from what the device is handed |
| `TestEmbeddingsOnEveryDevice` | every BERT-family embedding model on each device against the host and llama.cpp; the violation is BERT's output norm without its weight |
| `TestEveryModelRunsGenerated` | answers a request on every decision model it finds (`auditRun`) |

## 11. The five principles on the decision models

Each principle has a gate that runs Laya, d1, lev and Clef (the release and
its random fixture) (section 10's table):
JIT and no Go compute through `TestEveryModelRunsGenerated`; relocation
host to device mid-prompt and between questions (`TestDecisionRelocatesMidPrompt`,
`TestDecisionRelocatesBetweenDevices`), card to card
(`TestDecisionMovesBetweenCards`) and distribution over two cards and the host
(`TestDecisionDistributes`); paging (`TestDecisionPagesWithTheSameAnswer`,
`TestLayaPagesWithTheSameAnswer`); allocation (`TestDecisionDoesNotAllocate`);
batching (`TestDecisionBatchMatchesAlone`). Laya has no history, so its
relocation is between questions; its mid-prompt arm would move nothing.

What the allocation gate found, each fixed: a host prompt made a closure a
token (`prefillInto`), one a chunk for the embedding rows and one a block for
the attention rows (now pool regions on the State, as decode's are); a reset
State handed its KV pages to the collector and made them again (`kvPages.keepOnReset`);
the encoder's page-in listed its matrices and spans every block (`encoder.pages`),
its float GEMM, norms, bias rows and attention made a closure a call (`hotRegions.floatMM`,
`encRegions`); on a device, a reset uploaded and freed its zeros every prompt
(`devTier.recZeros`), a CUDA copy made a closure (`cudaCopy`), the embedding
gather's promise and ids were made each chunk, and Laya's non-causal set was
rebuilt -- its kernels compiled -- whenever a question's length moved it past
twice or half its rows, which the gates' two requests do at every question:
it now shrinks only after `visShrinkCalls` calls in a row need half of it
(`tier.TestATowerIsBuiltForItsPicture`).

Batching is per readout: a decoder's prompts are rows of one batch session; an
encoder's sequences run one encoder pass each, inside the same batch call,
because its blocks attend over every row of a call and the device's
non-causal set has one window list, not a sequence partition beside the local
windows. Packing several Laya sequences into one pass is open. Clef answers a request in one pass of its own, in a batch too: its head reads
every row of the prompt, and a batch session gives back each row's last.

