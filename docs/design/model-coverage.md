# Model coverage: what jitllm runs, what it does not, and what each gap costs

Read-only survey. Nothing here was built or measured; every
claim points at a file, a string table or a URL.

**This is EVIDENCE, not instruction.** AGENTS.md RULE 7 governs: the
architecture list is a LIST and adding to it is a decision. What follows is the
material for that decision, grouped by ENGINE CAPABILITY rather than by model
name, because one capability usually unlocks several families and that ratio is
the only thing that makes this tractable.

The question behind it: the user wants "models like deepseek, kimi, etc.
especially new versions".

---

## 0. The headline, before the detail

Six things came out of the survey that change the shape of the answer:

1. **`engine/model/forward.go` never mentions `Arch`.** `grep -c Arch engine/model/forward.go`
   is **0**; the only `jlm.Arch` reference in the whole `model` package is a
   validity check (`engine/model/model.go:201`). The graph is driven entirely by
   `jlm.Config` fields, `jlm.Flags` bits and which `jlm.Role` tensors exist. The
   architecture switch lives in ONE function, `convert.configOf`
   (`convert/config.go:67`). So jitllm does not pay llama.cpp's "50 graph
   builders" cost per name — it pays per CAPABILITY, and a name that reuses
   existing capabilities is close to free. That is the single most important
   structural fact for this plan.

2. **The tokenizers for most of these families are already generated.**
   `tok/pretok/table.go` carries 66 pre-tokenizer pipelines recovered from each
   model's own `tokenizer.json` by `scripts/gentok`, and the list already
   includes `deepseek-v3`, `deepseek-llm`, `deepseek-coder`, `deepseek-r1-qwen`,
   `glm4`, `minimax-m2`, `hunyuan`, `hunyuan-dense`, `bailingmoe`, `bailingmoe2`,
   `cohere2moe`, `exaone4`, `exaone-moe`, `falcon-h1`, `lfm2`, `afmoe`, `qwen35`,
   `laguna`, `seed-coder`. RULE 7 says "the expensive part of a new architecture
   is never the arithmetic — it is the tokenizer, and MoE". For **these**
   families the tokenizer half is already paid.

3. **`kimi-k2` is not an architecture in llama.cpp at all.** It does not appear
   in the arch-name table of the installed `libllama.so` (section 1 below has the
   full dump). Kimi-K2 ships as `DeepseekV3ForCausalLM` — corroborated
   independently by the HF fixture repo `tiny-random/kimi-k2`, which the search
   result describes as "contains a DeepseekV3ForCausalLM model structure". So
   **DeepSeek-V3 and Kimi-K2 are one capability, not two**, which confirms the
   premise this survey started from. And `kimi-linear` — which IS its own arch —
   turns out to need MLA as well (`kv_lora_rank: 512` in its own config.json,
   section 2F), so **one mechanism covers all four product lines the user
   named**.

4. **The converter already refuses DeepSeek by name, in one place.**
   `convert/config.go:411` — `rope.scaling.yarn_log_multiplier` is rejected as
   *"YaRN with a log multiplier (DeepSeek's mscale_all_dim) is not
   implemented"*. That is the only DeepSeek-shaped thing in the tree today, and
   it is one float field away from being implemented.

5. **This box cannot RUN a 671B model, and it can fully VERIFY one.** Those are
   different claims and separating them is what makes the plan tractable.
   The model disk has **68 GB free of 916 G** and the machine has **31 GB
   of RAM**, so DeepSeek-V3 at the smallest circulating quantization does not fit
   on the disk, never mind in memory. But `scripts/hfgold.py` already builds
   SYNTHETIC fixtures from transformers' own config classes — two layers, hidden
   size 64, random norms — and gates jitllm's graph against the reference's
   logits with nothing downloaded (section 6). The V3 graph is gateable in
   megabytes. What is additionally available is
   **DeepSeek-V2-Lite-Chat Q4_K_M, 10,364,416,672 bytes**
   (`Jianping746/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF`), a real 15.7B MLA model that
   fits, against a `llama-eval-callback` that dispatches `deepseek2` on this
   machine.

6. **The I-quant premise is REFUTED, and the real quant gap is one type.** A
   census of the actual published files — parsed out of the GGUF tensor tables,
   not read off model cards — finds that DeepSeek-R1 and Kimi-K2 ship straight
   `Q4_K_M` (404.4 GB and 620.8 GB), `Q6_K`, `Q8_0` and `BF16`, and that
   Unsloth's `UD-Q4_K_XL` and `UD-Q2_K_XL` contain **zero I-quant tensors** —
   they are Q5_K/Q4_K and Q3_K/Q2_K respectively. **jitllm's existing
   `Dequantable` set already covers every one of those files.** What is locked
   out is the whole 2-bit tier, and it is locked out for want of exactly one
   type: `Q2_K`. I-quants buy the sub-2-bit tier specifically, which is real but
   is neither the critical path nor the cheapest unlock. Section 2G has the
   per-shard counts.

---

## 1. The three lists, and what a name means in each

The three engines key on three different things and a table that mixes them is
unreadable. Stated once:

| engine | keys on | example |
|---|---|---|
| jitllm | the GGUF `general.architecture` string, mapped in `convert.archOf` | `qwen3moe` |
| llama.cpp | the same GGUF architecture string, its `LLM_ARCH_*` table | `deepseek2` |
| vLLM | the HuggingFace `config.json` `architectures[0]` CLASS name | `DeepseekV3ForCausalLM` |

jitllm has a second list keyed the vLLM way — `hfArchTable` in
`convert/safetensors.go:595` — for the safetensors input path.

### jitllm: 13 architecture strings

`convert/config.go:25-39`:

    llama (llama2, llama3)   qwen2      qwen3      qwen3moe    olmoe
    gemma                    gemma2     gemma3     phi3 (phi4)
    qwen3next                qwen2vl    gpt-oss    clip (tower, not an LM)

**Second list, safetensors** — `convert/safetensors.go:595`, 8 HF classes:
`LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`,
`Qwen3MoeForCausalLM`, `OlmoeForCausalLM`, `MixtralForCausalLM`,
`GemmaForCausalLM`, `Phi3ForCausalLM`.

**Third list, vision projectors** — `format/jlm/config.go`, `Projector`: `idefics3`,
`mlp` (llava), `qwen2vl_merger`.

### llama.cpp: 130 architecture strings

Extracted from a CUDA build of llama.cpp (`$JITLLM_LCPP`)'s `libllama.so`'s own
`.rodata` — the contiguous `LLM_ARCH_NAMES` table, so this is what that build
really dispatches on rather than a guess:

    clip llama llama4 deci falcon grok gpt2 gptj gptneox baichuan starcoder
    refact modern-bert nomic-bert nomic-bert-moe neo-bert jina-bert-v2
    jina-bert-v3 eurobert bloom stablelm qwen2moe qwen2vl qwen3 qwen3moe
    qwen3next qwen3vl qwen3vlmoe qwen35 qwen35moe phi2 phi3 phimoe plamo plamo2
    plamo3 codeshell orion internlm2 minicpm minicpm3 gemma gemma2 gemma3
    gemma3n gemma4 gemma4-assistant gemma-embedding starcoder2 mamba mamba2
    jamba falcon-h1 xverse command-r cohere2 cohere2moe dbrx olmo olmo2 olmoe
    openelm arctic deepseek deepseek2 deepseek2-ocr deepseek32 deepseek4
    chatglm glm4moe glm-dsa bitnet t5encoder jais jais2 nemotron nemotron_h
    nemotron_h_moe exaone exaone4 exaone-moe rwkv6 rwkv6qwen2 arwkv7 granite
    granitemoe granitehybrid chameleon wavtokenizer-dec bailingmoe bailingmoe2
    dots1 arcee afmoe laguna ernie4_5 ernie4_5-moe hunyuan-moe hunyuan-dense
    hunyuan_vl hy_v3 smollm3 gpt-oss lfm2 lfm2moe dream smallthinker llada
    llada-moe seed_oss grovemoe apertus minimax-m2 minimax-m3 cogvlm rnd1
    pangu-embedded mistral3 eagle3 dflash mistral4 paddleocr mimo2 step35
    llama-embed maincoder kimi-linear talkie mellum nanbeige

**Note what is NOT there: `kimi-k2`.** And note the five DeepSeek entries —
`deepseek`, `deepseek2`, `deepseek2-ocr`, `deepseek32`, `deepseek4` — which is
V1, V2/V3/R1, the OCR vision model, V3.2 (sparse attention) and V4.

The same `.rodata` dump gives the hyper-parameter key table, which is what
prices the converter work exactly. The keys relevant here:

    leading_dense_block_count        moe_every_n_layers   interleave_moe_layer_step
    expert_count  expert_used_count  expert_shared_count
    expert_group_count  expert_group_used_count  experts_per_group
    expert_weights_scale  expert_weights_norm   expert_gating_func
    attention.key_length  attention.value_length
    attention.q_lora_rank  attention.kv_lora_rank
    attention.key_length_mla  attention.value_length_mla
    attention.indexer.top_k  attention.indexer.types
    attention.decay_lora_rank  attention.iclr_lora_rank  attention.gate_lora_rank
    kda.head_dim   full_attention_interval   nextn_predict_layers
    ssm.conv_kernel ssm.inner_size ssm.state_size ssm.time_step_rank
    ssm.group_count ssm.dt_b_c_rms

**`attention.value_length` exists in every GGUF, and it is exactly the field MLA
needs.** When this survey was written the converter read `attention.key_length`
only and used that one number as `Config.HeadDim` for both K and V, which was
benign for the 13 architectures supported then, where the two are equal.
`convert/config.go` now reads it into `Config.HeadDimV` (defaulting to the key
width), and on an MLA model takes the natural widths from
`attention.key_length_mla` / `attention.value_length_mla`.
`jitllm info <file> -kv` prints every header key verbatim.

### vLLM: 125 text-generation classes, 121 multimodal, 62 speculative

From `vllm/model_executor/models/registry.py` on `main` (latest
release v0.30.0, 2026-09-22). Section 8 has the complete
lists. Three structural facts matter more than the count:

1. **vLLM has an escape hatch and the other two do not.**
   `TransformersForCausalLM` / `TransformersMoEForCausalLM` run *any* HF model
   whose implementation follows vLLM's attention interface, with no registry
   entry at all. So "absent from vLLM's registry" does not mean "cannot run".
   llama.cpp's 153 `llm_arch` entries are a hard ceiling and so is jitllm's 13.
2. **vLLM DELETES architectures**, and keeps a `_PREVIOUSLY_SUPPORTED_MODELS`
   table of 50+ entries naming the version that dropped each one. See section 5
   — this is the strongest external evidence for RULE 7's position that exists.
3. **vLLM no longer reads GGUF in-tree.** There is no
   `layers/quantization/gguf.py` on `main`; the docs say *"GGUF support has
   migrated to OOT vllm-gguf-plugin"*, and the RFC that moved it
   (vllm-project/vllm#39583) gives the reason: *"GGUF ... see very low usage
   relative to the maintenance burden ... users often citing running GGUF models
   with llamacpp to be faster."* The plugin does carry all nine I-quant CUDA
   kernels, ported from llama.cpp.

Roughly a third of vLLM's multimodal list is **audio/speech** (Whisper,
Ultravox, Voxtral, Qwen2-Audio, Qwen3-ASR, FunASR, GraniteSpeech,
AudioFlamingo3, Kimi-Audio, GlmAsr, CohereAsr, FireRedASR2), which is the single
biggest structural difference from llama.cpp and is entirely outside anything
jitllm contemplates.

---

## 2. The capability map

The missing names collapse into a small number of engine capabilities. Ordered
by how many families each one unlocks.

### A — Per-layer dense/MoE mix (`leading_dense_block_count`)

**What it is.** The first N blocks are a dense FFN and the rest are a mixture.
DeepSeek-V2-Lite has `first_k_dense_replace: 1`, DeepSeek-V3 has 3.
`interleave_moe_layer_step` and `moe_every_n_layers` are the same axis for other
families.

**What jitllm has.** Nothing. `engine/model/model.go:151` is
`func (c *Config) MoE() bool { return c.NExpert > 0 }` — a WHOLE-MODEL property,
consulted at seven sites in `engine/model/forward.go` (lines 464, 1057, 1095, 1123,
1156, 1212, 2077). `jlm.LayerKinds` already exists as the per-layer vocabulary
(`format/jlm/config.go`, `LayerFullAttn` / `LayerLinearAttn`) and the container already
writes one byte per layer, so the storage and the precedent are both there.

**What it costs.** One `jlm.Config` field or one more `LayerKind` bit, a
`jlm.Version` bump (22 -> 23), a converter line, and turning seven `c.MoE()`
call sites into `c.MoELayer(li)`. No kernel. No tokenizer.

**Families unlocked (as a precondition, not alone):** deepseek2, deepseek32,
deepseek4, **kimi-linear** (`first_k_dense_replace: 1` in its own config),
glm4moe, glm-dsa, hunyuan-moe, ernie4_5-moe, bailingmoe, bailingmoe2, dots1,
minimax-m2, minimax-m3, nemotron_h_moe, lfm2moe, exaone-moe, grovemoe, llada-moe,
smallthinker. **Nineteen.**

### B — MoE routing variants (sigmoid, grouped top-k, selection bias, weight scale)

**What it is.** DeepSeek-V3's router is not jitllm's. The reference chain is
sigmoid over all experts, then ADD a per-expert bias (`e_score_correction_bias`,
the aux-loss-free load balancer) **for selection only**, then a grouped top-k
(`n_group: 8`, `topk_group: 4` — pick the best 4 of 8 expert groups by their top
scores, then the best 8 experts within those), then gather the UNBIASED sigmoid
scores of the winners, normalise them, and multiply by
`routed_scaling_factor: 2.5`.

All four are confirmed from `deepseek-ai/DeepSeek-V3/config.json`:
`"scoring_func": "sigmoid"`, `"n_group": 8`, `"topk_group": 4`,
`"topk_method": "noaux_tc"`, `"norm_topk_prob": true`,
`"routed_scaling_factor": 2.5`, `"n_routed_experts": 256`,
`"num_experts_per_tok": 8`, `"n_shared_experts": 1`. The bias tensor is real:
`e_score_correction_bias` appears **59 times** in
`model.safetensors.index.json` — 58 MoE layers (61 minus
`first_k_dense_replace: 3`) plus one MTP layer.

**★ AND THE GROUPING IS DEGENERATE ON TWO OF THE THREE TARGETS, WHICH SPLITS
THIS CAPABILITY IN TWO.** Kimi-K2-Instruct ships `n_group: 1, topk_group: 1`
and Kimi-Linear ships `num_expert_group: 1, topk_group: 1` — plain top-k in both
cases. So the sub-capability breakdown is:

    sigmoid scoring + selection-only bias + routed_scaling_factor
        -> needed by DeepSeek-V3, R1, Kimi-K2 AND Kimi-Linear
    grouped top-k (n_group 8, topk_group 4)
        -> needed by DeepSeek-V3 and R1 ONLY

The first is the smaller change and covers more of what the user asked for.
Build it first and the grouped top-k second; a `n_group == 1` fast path is the
same kernel jitllm emits today.

**What jitllm has, and it is more than it looks.** The chain in `engine/model/moe.go:72`
is matvec -> `routerBias` -> softmax -> `nn.MoETopK32JIT(probs, !NoExpertNorm,
route)`. So:

- **router bias already exists** — `jlm.RoleRouterBias` (role 44), added for
  gpt-oss, applied at `engine/model/moe.go:72` and `:459`. gpt-oss's semantics differ
  (its bias feeds both selection and weights) but the tensor, the role, the
  upload and the device path are all built.
- **"do not renormalise" already exists** — `jlm.FlagNoExpertNorm` (bit 4),
  added for olmoe.
- **shared experts already exist** — `Config.NFFNShExp` plus
  `RoleShExpGate/Up/Down` and `RoleShRouter`. DeepSeek-V2-Lite's
  `n_shared_experts: 2` is llama.cpp's `expert_shared_count`, which it folds into
  one shared FFN of `moe_intermediate_size * n_shared_experts` — the shape jitllm
  already runs for qwen3next.

**So DeepSeek-V2-Lite's router is ALREADY EXPRESSIBLE.** Its config says
`scoring_func: softmax`, `topk_method: greedy`, `n_group: 1`, `topk_group: 1`,
`norm_topk_prob: false`, `routed_scaling_factor: 1.0` — that is exactly
softmax + plain top-k + `FlagNoExpertNorm`. Only **V3** needs the new router.

**What V3's router costs.** `nn.MoETopK32JIT` has no Go fallback by design
(`engine/nn/moetopk.go:19`) and the kernel bakes `(n, k, norm)`
(`cpu.EmittersFor(t).MoETopK(n, k, norm)`). Adding sigmoid + grouping + a
split between selection score and weight score means:

- a wider key and a wider emitter signature in **three host tiers** — AVX2
  (`jit_old/cpu/topk*.go`), SSE (`jit_old/cpu/sse_complete.go:146` lists `MoETopK` as a
  required SSE kernel) and arm64 (`jit_old/cpu/topk_a64.go`);
- **one** IR kernel change on the device side — `kernels.ExpertRank` /
  `kernels.ExpertWeights` in `jit_old/gpu/kernels/moe.go` lower to PTX, SPIR-V and
  MSL from one definition, so the device is one edit, not three;
- 4-5 `jlm.Config` fields (`ExpertGroups`, `ExpertGroupsUsed`,
  `ExpertWeightsScale`, a gating-function enum) and a version bump.

`ExpertRank`'s rank-by-counting design (`rank(e) = #{f: l_f > l_e} + #{f<e:
l_f == l_e}`) generalises to a grouped top-k without introducing mutable state,
which is what `ir.Validate` and RULE 13 require — the group's score is a max over
its members, computable the same branch-free way. That is an opinion from reading
the kernel, not a measurement.

**Families unlocked:** deepseek2 (V3/R1/V3.1/Kimi-K2), glm4moe, bailingmoe2,
dots1, hunyuan-moe, minimax-m2/m3, ernie4_5-moe. **Eight**, and it is the
difference between "DeepSeek-V2-Lite runs" and "DeepSeek-V3 runs".

### C — MLA (Multi-head Latent Attention)

**What it is.** The KV cache is a single low-rank latent per position instead of
a per-head K and V. DeepSeek-V2-Lite's numbers, from its own `config.json`:

    hidden_size 2048   num_attention_heads 16   num_hidden_layers 27
    kv_lora_rank 512   q_lora_rank null
    qk_nope_head_dim 128   qk_rope_head_dim 64   v_head_dim 128
    -> the query/key head is 128+64 = 192 wide, and the VALUE head is 128

DeepSeek-V3 adds `q_lora_rank: 1536` (the query is also low-rank) and keeps the
same 192/128 split.

**Why it does not fit today, precisely.**

1. **`jlm.Config` has ONE `HeadDim`** (`format/jlm/config.go:137`) and no field that can
   hold `kv_lora_rank`, `q_lora_rank`, or a rope/no-rope split of a head.
2. **`model.kvLayout` has ONE `headDim`** (`engine/model/kvlayout.go:31`) and computes
   `kvDim() = nKV * headDim` for both K and V. Under MLA those differ.
3. **`nn.JIT.AddAttn(hd, kvStride, f16)`** (`engine/nn/attn.go:75`) maps SIX kernels
   from one `hd`: `AttnScores`, `AttnAcc`, `AttnAccInto`, `AttnScores2`,
   `AttnAcc2`, `AttnAcc2Into`. The `*2` variants are the PAIRED kernels that walk
   K and V in one pass — they assume K and V share a layout. The underlying
   emitters already take `hd` as a parameter, so the arithmetic generalises; what
   does not generalise is the single `hd` threaded through `AddAttn`, the cache
   and the device tier's KV buffer.
4. **No roles exist for MLA's tensors.** `format/jlm/schema.go` has
   `RoleAttnQ/K/V/Out/QKV/QNorm/KNorm` and nothing for `kv_a_proj_with_mqa`,
   `kv_b_proj`, `q_a_proj`, `q_b_proj`, `kv_a_layernorm`, `q_a_layernorm`. About
   six to seven new role codes.

**Two implementation shapes, and the choice matters.**

- *Naive / decompressed*: materialise per-head K (192) and V (128) each step and
  cache them. Simple, and the cache is enormous — that is why nobody ships it.
- *Absorbed*: fold `W_UK` into `W_UQ` at load and cache only the latent
  `[c_KV(512) ; k_rope(64)] = 576` once per position, MQA-style, with V being the
  512-wide PREFIX of the same row. The cache shrinks dramatically; the cost is a
  load-time matrix product the container would have to store (a converter
  change), and an attention kernel where the V read is a prefix slice of the K
  row rather than a separate buffer.

Absorbed is the right target and it interacts with `jlm`'s "the converter owns
the layout" design rather well: llama.cpp ships `attention.key_length_mla` /
`attention.value_length_mla` precisely because it does the same split.

**What it costs.** The biggest single item here. Roughly: 4 Config fields +
~7 roles + a container version bump; `kvLayout` and `AddAttn` taking two head
dims; one new attention-kernel shape (V as a prefix of K) on three host tiers
and one IR definition; a converter case; and the YaRN `mscale_all_dim` term
(section D below), which is one float.

**Families unlocked:** deepseek2 — which is DeepSeek-V2, V2.5, V3, V3.1, R1
**and Kimi-K2** — plus **kimi-linear**, whose seven full-attention layers are
MLA (section 2F), plus the text half of deepseek2-ocr, and it is the
prerequisite for deepseek32/deepseek4/glm-dsa.

**★ MLA IS THE ONE CAPABILITY THAT COVERS EVERY MODEL THE USER NAMED.**
DeepSeek-V2/V3/R1, Kimi-K2 and Kimi-Linear are four product lines and one
attention mechanism. Nothing else in this document has that ratio.

### D — YaRN with a log multiplier (`mscale_all_dim`)

**What it is.** DeepSeek's rotary scaling multiplies the cos/sin magnitude by
`0.1 * mscale_all_dim * ln(factor) + 1` rather than `0.1 * ln(factor) + 1`.

**What jitllm has.** The whole YaRN path, added for gpt-oss: `Config.YarnFactor`,
`YarnOrigCtx`, `YarnBetaFast`, `YarnBetaSlow`, `FlagYarnExact`, and
`ropeScalingOf` folding the magnitude into `AttnFactor`
(`convert/config.go:386-418`). It already refuses this exact key by name:

    convert/config.go:411
      if _, ok := f.Key("rope.scaling.yarn_log_multiplier"); ok {
        return fmt.Errorf("YaRN with a log multiplier (DeepSeek's mscale_all_dim) "+
          "is not implemented: %w", ErrNotImplemented)
      }

**What it costs.** One multiply, and the decision whether it is a new Config
field or folded into `AttnFactor` at conversion (it can be folded — `AttnFactor`
already scales cos and sin, and `mscale_all_dim` is constant per model). Call it
an afternoon including its gate.

**★ AND IT IS ONLY LOAD-BEARING FOR V2, WHICH IS WORTH KNOWING BEFORE ANYONE
SCHEDULES IT AS A BLOCKER.** DeepSeek-V2-Lite ships `mscale: 0.707,
mscale_all_dim: 0.707` — a real correction. DeepSeek-V3, R1, V3.1 and V3.2 all
ship `mscale: 1.0, mscale_all_dim: 1.0`, where the term is the identity and the
existing formula is already correct. Kimi-Linear ships `rope_scaling: null`
entirely. So this is needed to run **DeepSeek-V2/V2-Lite** — which is the
verification fixture, so it is still on the critical path — and is a no-op for
everything after it.

### E — Mamba-2 / selective state space

**What it is.** A different recurrence from qwen3next's gated delta rule: a
diagonal decay `A`, per-group `B`/`C`, a softplus'd `dt`, and an optional
`D` skip. `ssm.dt_b_c_rms` and `ssm.group_count` are its extra GGUF keys.

**What jitllm has, and it is most of the scaffolding.** This is the asset a plan
that counts names would miss:

- `jlm.LayerKind.LayerLinearAttn` and per-layer kinds written by the converter
  from the TENSORS, not from a rule (`convert/config.go`, the `ssm_conv1d`
  probe) — so an irregular hybrid pattern costs nothing extra.
- `jlm.SSMConfig{ConvKernel, Groups, Inner, StateSize, NHeadV}` — five fields,
  and `Groups` is already there for exactly Mamba-2's group axis.
- Eight SSM roles already defined: `RoleSSMInProj`, `RoleSSMBA`,
  `RoleSSMConv1d`, `RoleSSMA`, `RoleSSMDtBias`, `RoleSSMNorm`, `RoleSSMOut`,
  `RoleSSMOutGate` (`format/jlm/schema.go:245-253`).
- The device kernels: `kernels.Conv1dRows`, `Conv1dShift`, `GatedDeltaStep`,
  `DeltaGate` (`jit_old/gpu/kernels/recurrent.go`), all out-of-place so
  `ir.Validate`'s no-read-your-own-write rule holds.
- The state machinery: the double-buffered `recPair` and its swap, `MigrateRec`
  / `syncRecFromDevice` / `pushRecToDevice`, a hybrid's single KV restore point,
  and a device tier that places recurrent blocks.
- `DeltaGate` already computes `exp(A * softplus(alpha + dtBias))` with a
  branch-free softplus — which IS Mamba-2's decay term.

**What it costs.** One new recurrence kernel (the SSD step) on three host tiers
plus one IR definition, one or two `SSMConfig` fields, and a converter case. The
hard parts — two state buffers, migration, per-layer kinds, hybrid KV caching,
device placement of a recurrent block — are built and gated.

**Families unlocked:** mamba2, jamba, falcon-h1, nemotron_h, nemotron_h_moe,
granitehybrid, plamo2, plamo3. **Eight**, and `mamba` (v1) is a near neighbour.

### F — Linear/delta attention variants (Kimi Delta Attention)

**What it is.** `kimi-linear` is its own llama.cpp architecture and brings its
own keys: `kda.head_dim`, `attention.decay_lora_rank`,
`attention.iclr_lora_rank`, `attention.gate_lora_rank`.

**What jitllm has.** `kernels.GatedDeltaStep(vHeads, vDim, kDim, rep)`
(`jit_old/gpu/kernels/recurrent.go:146`) — per-head SCALAR `decay` and `beta`, a
rank-one state update `S <- S*decay + beta*(v - S*decay*k)*k^T`, out of place,
with the state-buffer swap owned by the caller. That is Gated DeltaNet.

**The hypothesis is CONFIRMED and the citation is unambiguous.** The Kimi Linear
paper (arXiv:2510.26692) describes KDA as *"a hardware-efficient linear attention
module that extends Gated DeltaNet with a finer-grained gating mechanism... a
channel-wise variant in which each feature dimension maintains an independent
forgetting rate"*, and a third-party survey (arXiv:2607.07953) puts it flatly:
*"Kimi Delta Attention keeps the gated delta-rule structure but replaces scalar
forgetting with channel-wise forgetting."* NVIDIA's follow-up (arXiv:2605.22791,
Gated DeltaNet-2) generalises both.

In `GatedDeltaStep` the decay is `decay := b.Load(ir.F32, pDecay, vh, 0)`, one
load hoisted above the k-loop. Channel-wise means loading `pDecay[kBase+i]`
INSIDE the loop. That really is a small edit to one IR kernel and its host twins
— plus the low-rank projections that produce the gate (`decay_lora_rank`,
`iclr_lora_rank`, `gate_lora_rank`), which are ordinary matvecs and new roles.
**The cost is still a LEAD and not a costing** until someone reads Moonshot's
kernel: a per-channel decay changes the numerics of the `sk` reduction and the
chunked form, and the paper's efficiency claims rest on a chunkwise algorithm
this note has not read.

**★ AND Kimi-Linear IS NOT A SHORTCUT AROUND MLA — IT NEEDS IT TOO.** From
`moonshotai/Kimi-Linear-48B-A3B-Instruct/config.json`:

    "architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear"
    kv_lora_rank 512   q_lora_rank null   mla_use_nope true
    qk_nope_head_dim 128   qk_rope_head_dim 64   v_head_dim 128
    linear_attn_config: { kda_layers: [20 of them], full_attn_layers: [4,8,12,
                          16,20,24,27], head_dim 128, num_heads 32,
                          short_conv_kernel_size 4 }
    first_k_dense_replace 1   num_experts 256   num_experts_per_token 8
    moe_router_activation_func "sigmoid"   moe_renormalize true
    routed_scaling_factor 2.446   num_shared_experts 1
    use_grouped_topk true   num_expert_group 1   topk_group 1
    num_nextn_predict_layers 0

So its seven full-attention layers are **MLA**, its twenty linear layers are
KDA over a causal convolution of kernel 4 (jitllm's `SSMConfig.ConvKernel`
exactly), and its FFN is a sigmoid-routed 256-expert mixture with a shared
expert, a weight scale, and a dense first layer. Kimi-Linear therefore needs
capabilities A, B (sigmoid + `routed_scaling_factor`; grouping is degenerate at
`num_expert_group: 1`), C and F — a strict superset of the DeepSeek-V3 work.

**Families unlocked by F alone:** kimi-linear, and it deepens qwen3next.
RWKV6/ARWKV7 are a different recurrence (token shift + WKV) and are NOT in this
family.

### G — Quantization formats

**What jitllm decodes.** `quant.Dequantable` (`format/quant/dequant.go:216`):
`F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4` — eleven.
**What it PACKS into a container** is `quant.PackedTypes`: eight
(`Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4`). The container cannot even
NAME anything else — `jlm.SourceType`/`TypeOf` (`format/jlm/schema.go:366-420`) is an
eleven-arm switch.

**What it does not decode, though `format/quant/types.go` knows their block geometry**
(`blockInfo`, lines 44-70 — the sizes are there so `Open` can compute a tensor's
length and refuse cleanly): `Q2_K`, `Q4_1`, `Q5_1`, `Q8_1`, `Q8_K`, `IQ1_S`,
`IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`,
`TQ1_0`, `TQ2_0`.

**There is exactly ONE quantizer in the tree and it is Q8_0 only.** Package
`quant` has `dequant.go` and `types.go` and nothing that produces quantized
blocks. But `convert.quantizeTower` (`convert/vision.go:114`) rewrites every
tower matrix to Q8_0 via `quantizeQ8Rows` — *"the only lossy step in the
converter, and it says so"* — so the TRANSCODE pattern (dequantize the source
block, re-quantize into a type the engine packs) exists, is gated, and has a
precedent.

**That matters for the I-quant question, and it does not rescue it.** A decoder
alone — one case arm in `format/quant/dequant.go` — plus the transcode path would let
jitllm OPEN an I-quant file without any new kernel. It would also inflate it:
IQ2_XXS is about 2.06 bits per weight, Q8_0 is 8.5, and even a hypothetical
Q4_K re-quantizer roughly doubles it. For a 671B model, where the whole point of
the I-quant is that the file fits at all, transcoding defeats the purpose.
**Native decode is the only route that preserves the size**, and the transcode
path is worth knowing about only for small models and for making a refusal into
a degraded success.

### ★ AND THE I-QUANT PREMISE IS REFUTED BY A CENSUS OF THE ACTUAL FILES

The claim this survey was asked to test — *"nobody ships a 671B at Q4_K_M; the
files people download are I-quants, so an engine without I-quant decode cannot
open them at all"* — is **false on both halves**. Verified by parsing the GGUF
tensor-info section (name, dims, `ggml_type`, offset) over HTTP range requests
rather than from model cards.

**Every major publisher ships straight `Q4_K_M`, `Q8_0` and `BF16`:**

| model | Q4_K_M | Q8_0 | BF16 |
|---|---|---|---|
| DeepSeek-R1 | **404.4 GB** (9 shards) | 713.3 | 1342.3 |
| DeepSeek-V3 / V3-0324 | 404.4 GB | 713.3 | 1342.3 |
| DeepSeek-V3.1 / V3.2 | 405.4 GB | 713.3 | 1342.3 |
| Kimi-K2-Instruct | **620.8 GB** | 1091.1 | 2053.2 |
| Kimi-K2-Thinking | 621.2 GB (13 shards) | 1091.1 | 2053.2 |

`unsloth/DeepSeek-V3-GGUF` is **100% K-quant with no UD variants at all**, and
bartowski independently ships Q4_K_M / Q6_K / Q8_0 for V3-0324, V3.1 and
K2-Thinking.

**And Unsloth's UD-Q\* "Dynamic" quants contain ZERO I-quant tensors.** A census
over every shard:

    unsloth/DeepSeek-R1-0528-GGUF/UD-Q4_K_XL   1086 tensors
        Q4_K 483  F32 361  Q8_0 122  Q5_K 97  Q6_K 23     IQ present: FALSE
    unsloth/DeepSeek-R1-0528-GGUF/UD-Q2_K_XL   1086 tensors
        Q4_K 397  F32 361  Q8_0 122  Q2_K 116  Q3_K 51    IQ present: FALSE
    unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL
        F32 361  Q4_K 326  Q6_K 184  Q2_K 171  Q3_K 3     IQ present: FALSE
    DeepSeek-R1 Q4_K_M, all 9 shards
        Q4_K 606  F32 361  Q6_K 58                        IQ present: FALSE

Unsloth's own naming key explains it: the "Details" column is a bit-width PAIR,
so `UD-Q4_K_XL` is "5.5/4.5 bit" = Q5_K over Q4_K and `UD-Q2_K_XL` is "3.5/2.5
bit" = Q3_K over Q2_K. Only the `UD-IQ*` names carry IQ blocks.

**★ SO jitllm's EXISTING `Dequantable` SET ALREADY OPENS R1 AND KIMI-K2** — at
Q4_K_M, Q5_K_M, Q6_K, Q8_0, Q3_K_M, BF16 **and UD-Q4_K_XL**. What it cannot open
is the entire 2-bit tier, and that is locked out **for want of exactly one type,
`Q2_K` (ggml type 10)**, which is the format Unsloth itself recommends as the
size/accuracy sweet spot.

**One naming trap worth recording**: `unsloth/Kimi-K2-Instruct-GGUF/UD-TQ1_0`
contains **no TQ1_0** — it reads IQ4_XS 88, IQ3_S 69, IQ1_S 57, IQ2_XXS 29,
IQ3_XXS 28 plus k-quants. Never infer block types from a directory name; parse
the header.

**And a second trap, for anyone benchmarking against the community:**
`ubergarm/Kimi-K2-Thinking-GGUF` ships `IQ2_KL`, `IQ3_K`, `smol-IQ1_KT` — these
are **ik_llama.cpp fork-exclusive types** (the header parse returns unknown ggml
types 138/139/141). Mainline llama.cpp cannot read them either, so they are not
a coverage gap.

**What I-quants actually buy, stated exactly:** the sub-2-bit tier.
`UD-IQ1_S` at **140.2 GB** is the only DeepSeek-R1 file that fits a 128 GB-class
box, and bartowski's entire sub-4-bit lineup is I-quant. That is real and
popular — and it is not required to open R1 or K2 at all.

**Native weights, for completeness.** DeepSeek-V3/R1/V3.1/V3.2 and
Kimi-K2-Instruct ship **FP8 e4m3, block-scaled**
(`weight_block_size: [128,128]`, V3.1/V3.2 adding `scale_fmt: ue8m0`). GGUF has
no FP8-e4m3 type with a 2-D scale grid, so every converter upcasts FP8 -> BF16
first — which is why every Unsloth repo carries a 1342 GB `BF16/` directory and
`UD-Q2_K_XL` shard 1's metadata reads `general.name = "DeepSeek R1 BF16"`.
**Kimi-K2-Thinking is the exception: native INT4 QAT**, compressed-tensors
pack-quantized, group 32, symmetric, MoE only. **Nothing in the DeepSeek or Kimi
line uses MXFP4** — that is gpt-oss's alone, and MXFP4 blocks survive into every
gpt-oss GGUF whatever its name says (`unsloth/gpt-oss-120b-GGUF/Q4_K_M` reads
`F32 345, MXFP4 85, Q5_0 76, Q4_K 29, Q8_0 13`), which is why every variant
weighs ~63 GB.

**Not all of the missing types cost the same, and that is the useful
distinction:**

| format | shape | how far from something jitllm has |
|---|---|---|
| `Q2_K` | 2-bit primary plane + 4-bit scale/min, k-quant super-block | **closest.** It needs the 2-bit primary emitter shape that AGENTS.md already records as the blocker for native Q3_K packing, so the two share the work |
| `IQ4_NL`, `IQ4_XS` | 4-bit INDEX into a fixed 16-entry non-linear value table, plus scales | **close.** A 16-entry int8 table lookup is exactly one `VPSHUFB` per lane on AVX2, one `TBL` on NEON, a constant array on the device. The looked-up values are SIGNED where the existing nibble path feeds `VPMADDUBSW` unsigned bytes, so it needs the same offset-and-correct trick the packed kernels already carry for centering. Structurally Q4_K plus a lookup, not a new family |
| `Q4_1`, `Q5_1` | Q4_0/Q5_0 with a per-block minimum | close; the k-quants already carry a min plane |
| `IQ1_*`, `IQ2_*`, `IQ3_*` | codebook GRIDS — an index selects 8 packed values from a 256- or 512-entry table, plus sign bits | **alien.** A gather per 8 weights against a plane-and-scale design. This is a different kernel family, not a variant |
| `TQ1_0`, `TQ2_0` | ternary, for BitNet | only useful with the `bitnet` architecture |

**What one format costs, measured by the last one added.** MXFP4 (for gpt-oss)
touches twenty non-test files:

    internal/oracle/packed.go   the reference decoder the kernels are gated against
    format/quant/dequant.go, types.go
    format/jlm/schema.go, format/jlm/write.go  a container type code -> a Version bump
    jit_old/gpu/kernels/{matvec,mma,pack,quantof}.go     the device side
    jit_old/cpu/{codes,matvec,packact,packed_a64,packed_other,prevnni}.go
    jit_old/cpu/{sse_budget,sse_helpers,sse_norm,sse_packed,sse_quantact}.go
    convert/library/models.go, ui/catalog/quant.go

Note `mma.go` in that list: AGENTS.md records three separate incidents of a
plane-layout change reaching `matvec.go` and not `mma.go`, the most recent of
which threw away Q5_K's fifth bit on the CUDA prefill path. Any new format pays
that attention tax.

**Where the leverage actually is.** A quant format unlocks the FILES for every
architecture at once, which no single architecture does. But it is not on the
critical path to DeepSeek running at all — see section 4.

### H — Sparse attention indexer (DSA)

`deepseek32` (V3.2), `deepseek4` and `glm-dsa` add a "lightning indexer" that
scores all past keys cheaply and attends to only the top-k
(`attention.indexer.top_k`, `attention.indexer.types`). This is a new kernel
family with a new data dependency (a per-query key selection feeding the score
kernel) and it sits ON TOP of MLA. Nothing in jitllm is close to it.

### I — Multi-token prediction (`nextn_predict_layers`)

DeepSeek-V3 and GLM-4.5 ship extra MTP layers in the checkpoint. llama.cpp loads
the model without them. jitllm must do the same — but **explicitly**, because
`convert.identify` refuses an unrecognised tensor by design ("a tensor dropped at
conversion is a model that loads and is wrong, so this is a refusal",
`convert/safetensors.go:665`). So MTP tensors need an explicit ignore entry with
a comment, not a silent skip.

### J — Vision towers and projectors

`deepseek2-ocr` (DeepEncoder: a SAM encoder, a CLIP encoder and a convolutional
compressor), `kimi-vl` (MoonViT), `qwen3vl`/`qwen3vlmoe` (deepstack —
`n_deepstack_layers` and `deepstack_mapping` are GGUF keys), `glm4v`,
`hunyuan_vl`, `cogvlm`, `paddleocr`, `llama4`'s vision half. Each is its own
tower graph plus its own projector. jitllm has three projectors and one CLIP-ish
tower.

### J1 — The families that need none of the in-flight vision work, surveyed

Read off llama.cpp b10825's `tools/mtmd` (the oracle binaries' commit,
9e0e22059), transformers' processors and models, and vLLM. Two other pieces of
vision work were in flight when this was written -- Gemma 3 vision with InternVL
(since landed; the PIL families below share its resampler at Pillow's
precision), and M-RoPE with 2-D rotary and Qwen2.5/3-VL -- so each family is
classed by whether it needs either. "Tower" is what the tower needs beyond the CLIP/SigLIP
block jitllm already runs on every tier (LayerNorm with biases, biased q/k/v/o,
an ungated MLP, bidirectional attention, a learned table for one square grid).

| family | llama.cpp projector | tower needs | projector | preprocessing | text arch | depends on in-flight? | cost |
|---|---|---|---|---|---|---|---|
| **MiniCPM-V 2.6 / 4.0 / 4.5** | `resampler` (minicpmv_version 3-6) | any grid: a 70x70 table read by NaViT bucketing; SigLIP-400M, FFN 4304 (not a whole Q8_0 block) | perceiver: 64 learned queries, one cross-attention (128-wide heads, a constant), 2-D sin/cos on the keys, LN, a final matrix | llava-uhd: an overview and up to 9 slices, Pillow BICUBIC, 64 tokens each, `<image>`/`<slice>`/`\n` markers (+ `<image_id>` in the processor, not in mtmd) | qwen2 / llama (+longrope factors) / qwen3 | **no** | M -- **built first** |
| MiniCPM-V 4.6 | `minicpmv4_6` | any grid, plus a windowed-attention 2x2 merger inserted after layer 6 (the tower's row count halves mid-stack) | 2x2 merge, LN, erf-GELU MLP | the same llava-uhd, aligned to 56 | qwen35 (IMROPE, but every image row gets one sequential position) | no | L: the mid-stack merger is a second block shape and a sequence that shrinks between device runs |
| Phi-4-reasoning-vision-15B | `phi4` (NOT Phi-4-multimodal) | SigLIP2 NaFlex: any grid, the table bilinear-antialias-resized per grid; layer -2 (the converter drops the last block) | mm.0, GELU, mm.2 | dynamic size between min and max patches, no tiles, no markers | phi3 (15B) | no | M -- **built** on the vision runner (below) |
| Phi-4-multimodal-instruct | none -- no converter, no mmproj anywhere | SigLIP 448, layer -2, NaViT 32x32 buckets over the valid patches of a padded crop | 2x2 AvgPool, sub/glb separator embeddings, MLP | dynamic-HD crops padded white, global crop bicubic | phi3 **plus a vision LoRA merged into the text weights** | no | XL: a safetensors-only path and a LoRA merge |
| **Janus-Pro** | `janus_pro` | none: SigLIP-L, a 576-row table, one 384 square | mm.0, GELU, mm.1 (`mm.1`, not `mm.2`) | 384 square, BICUBIC, letterboxed in (127,127,127); HF wraps `<begin_of_image>`/`<end_of_image>`, mtmd wraps nothing | llama | no | **S** -- **built** |
| GLM-Edge-V | `adapter` | none (fixed square, all layers, post-LN) | 2x2 conv merge, LN, GELU, gated MLP, BOI/EOI rows that are NOT text tokens | fixed square, bicubic letterbox | chatglm -- a text arch jitllm does not have | no | M (the text arch) |
| LFM2-VL | `lfm2` | any grid: SigLIP2 NaFlex table bilinear-antialias-resized | pixel unshuffle 2x2, LN, MLP | 512 tiles + thumbnail, or one variable image | lfm2 (short-conv hybrid) -- not in jitllm | no | M + a new text arch |
| CogVLM | `cogvlm` | post-sublayer LN (not pre-LN), class token after the patches in llama.cpp | fc, LN, GELU, gated MLP, BOI/EOI rows | fixed square letterbox | cogvlm: llama with a visual expert per layer selected by the batch | no | L (the text side) |
| granite4_vision | `granite4_vision` | intermediate-layer taps, no post-LN | a windowed Q-Former per tapped layer | llava-next anyres | granite with deepstack injection into the text layers | no | XL |
| **Qwen3-VL** (2B-235B, dense and mixture) | `qwen3vl_merger` | qwen2vl's 2-D rope, LayerNorm, fused qkv, GELU-tanh, a learned 48x48 table resampled bilinear to the grid | 2x2 merge; deepstack mergers on three taps, their rows added into text blocks 0-2 | dynamic size aligned to 32, floor 256x256 | qwen3 / qwen3moe with **interleaved** M-RoPE | -- | **built** (`jlm.ArchQwen3VL`, `ProjQwen3VL`) |
| GLM-4.1V / 4.5V / 4.6V | `glm4v` | qwen2vl's 2-D vision rope, RMSNorm, fused qkv, SwiGLU, a learned table resampled bicubic, an RMSNorm after the patch embedding | 2x2 conv merge, LN, GELU, SwiGLU | dynamic size aligned to 28, bounds on the two temporal frames | glm4 / glm4moe with **M-RoPE** | -- | **built** (`jlm.ArchGLM4`, `ArchGLM4VMoE`, `ProjGLM4V`) |
| Kimi-VL | `kimivl` | MoonViT: 2-D vision rope in complex pairs (column, row alternating), LayerNorm, GELU-tanh, a learned table resampled bicubic | LN per patch, 2x2 merge, MLP | unresized under 4096 patches, padded black to 28 | deepseek2 | -- | **built** (`ProjKimiVL`) |
| youtuvl, exaone4_5, step3vl, dotsocr, mimovl, minimax_m3, ling3vl | -- | 2-D vision rope (and window attention for several) | -- | -- | -- | **yes** | queued |
| HunyuanVL (HunyuanOCR) | `hunyuanvl` | a LayerNorm ViT, no rotary, the patches in raster order, a learned table resampled bilinear | RMSNorm, 2x2 conv, GELU, 1x1 conv, a newline row per merged row, linear, begin and end rows, RMSNorm | dynamic size aligned to 32 between the file's bounds, PIL bicubic | hunyuan_vl: hunyuan's dense block under XD-RoPE (four coordinates, assigned per element) | -- | **built** (`jlm.ArchHunyuanVL`, `ProjHunyuanVL`) |

What the table says, in order: **MiniCPM-V is the most valuable** -- three
product generations on one projector, two of them on text archs jitllm already
runs -- and what it costs is the two capabilities every newer family needs
anyway: a tower that reads ANY grid, and a picture cut into an overview and
slices. **Janus-Pro is nearly free** once the projector's second matrix is
named for what it is. **Phi-4** in llama.cpp is the 15B reasoning-vision model,
not Phi-4-multimodal, which no GGUF exists for at all.

Three divergences between llama.cpp and the reference, each a choice
(RULE 7m), each measured where the family is built
(`docs/engineering-history/model-correctness.md`): mtmd drops MiniCPM-V's
`<image_id>`; it squashes a small MiniCPM-V picture to a square where the
processor keeps its aspect, and slices a long thin one the processor does not;
and its rational position buckets part from torch's float32 ones by one bucket
at 1078 coordinates of the axis lengths below 1500.

**Phi-4-reasoning-vision is built on the vision runner** (towers as ordinary
blocks through the text model's runner, cache, placement and scheduler). It was
built and measured on the old path first, which is how what it needed was
known before the runner existed:

| piece | what it needs |
|---|---|
| a picture at its own size | the processor's `get_image_size_for_max_num_patches`: the picture's own patch count clamped to the file's `clip.vision.image_min_pixels`/`image_max_pixels` (256..3600 patches of 16), then a binary search for the largest scale whose ceil()ed, patch-aligned size holds it. The two bounds go in the container: an optional vision-section tail AFTER InternVL's MinTiles/MaxTiles, so a tower with them writes zero tiles first |
| the position table | 16x16 rows resized to each grid as Siglip2VisionEmbeddings does: F.interpolate, bilinear, align_corners=False, antialias=True, in float. These are `imageproc.go`'s `aaWeights` taps before the fixed point (factor them out), applied across then down as axpys over whole channel rows, cached per grid |
| the resize | Pillow BILINEAR: `imageproc.go`'s resampler at Pillow's fixed 22-bit precision, as MiniCPM-V's BICUBIC (`resizePIL`) |
| rows | a grid of up to 3600 patches, so the scratch and the device's vision attention take a grid's rows, not a square's |
| projector | mm.0, GELU, mm.2 over every patch: llava's two-matrix MLP with a per-picture token count |
| markers | none: the processor splits the prompt at `<image>` and puts the rows there |
| the text model | phi3, 40 layers, 16k context, and its GGUF states `attention.sliding_window` = 0 (the config's null). A PRESENT zero must mean no window; today only an absent key is meant to fall to phi3's defaults, and the converter refuses this file |

Measured on the old path, before the runner:
the processor's pixels and resample sizes exact over 14 picture sizes; the
table resize within 3e-07 of F.interpolate over 7 grids; the tower 1.7e-03 /
3.8e-03 and the projector 4.5e-04 / 8.7e-04 NMSE against transformers'
hidden_states[-2]; 0.002 to 0.033 relative RMS against llama.cpp's own run,
node by node; every block on cuda / vulkan (V100) / metal (M4) 3.8e-04 /
1.0e-04 / 1.2e-04 against the host; 12 of 12 teacher-forced positions of
llama.cpp's 2003-position prompt on the laptop, 11 of 12 on the M4 (a 0.004
tie); "a front page of the new york times newspaper reporting on the moon
landing." Its 2003-position prompt is what found the two long-context decode
allocations fixed beside this (the host's page walk past four KV pages, the
device's streamed eviction path).

**Built on the vision runner since this survey** (the evidence is in
`docs/engineering-history/model-correctness.md`, one section each):

- **Pixtral / Mistral 3** (`pixtral`: Mistral Small 3.x, Ministral 3,
  Pixtral-12B) -- Qwen2.5-VL's kit without biases, Pixtral's resize, the
  even/odd 2-D rotary on adjacent pairs, Mistral 3's merger permuted at
  conversion, and the [IMG_BREAK] rows the head writes.
- **Llama 4** (`llama4`: Scout, Maverick) -- the CLIP kit over 336 tiles, the
  class token last, a 2-D rotary beside the table from position one, the
  bfloat16 processor, the adapter's two GELUs.
- **Gemma 4** (`gemma4v`: E2B, E4B, 26B-A4B, 31B) -- the gemma kit on patches
  with two position tables, a half-head rotary made whole-head NEOX at
  conversion, every matrix clamped on the E-models, the pooling and
  standardising head, and the 26B's and 31B's image rows bidirectional on
  the sliding layers only.
- **Phi-4-reasoning-vision** (`phi4`) -- SigLIP2 NaFlex at the picture's own
  size between the file's pixel bounds, its 16x16 table resized to each grid,
  hidden_states[-2], llava's two matrices over every patch, and phi3 text with
  a stated window of zero.
- **Gemma 3n** (`gemma3nv`: E2B, E4B) -- timm's MobileNet-V5, the first
  tower that is not a transformer: 84 convolutional blocks (edge residuals,
  universal inverted residuals, multi-query attention with a downsampled key
  and value) under TF SAME padding, the fusion adapter over two stages, the
  embedder; on a device each block is a program of generated kernels
  (`jit_old/gpu/tier/conv.go`), the activation changing shape from block to
  block. The text side gained the vision embedder's hard tokens (the
  end-of-image marker is one) and the per-layer table's zero padding trimmed.

### K — Names that need no new capability at all

This is the cheap tail and it should be named, because RULE 7's scope argument is
about CAPABILITIES and these are not capabilities:

- **`qwen2moe`** (Qwen1.5-MoE) = qwen2's biased-QKV attention + a routed mixture
  + a GATED shared expert. jitllm has all three today (`FlagRopeNeox`,
  `NFFNShExp`, `RoleShRouter` — the shared-expert gate was built for qwen3next).
  Plausibly a converter entry and a gate, nothing else. **Unverified.**
- **`qwen35` / `qwen35moe`** — successors to qwen3/qwen3moe; the pre-tokenizer
  `qwen35` is already in `tok/pretok/table.go` and in `qwen2Family`'s digit rule.
- **`olmo2`, `granite`, `granitemoe`, `smollm3`, `arcee`, `seed_oss`, `apertus`,
  `exaone4`, `cohere2`, `cohere2moe`, `minicpm`, `internlm2`, `stablelm`,
  `phimoe`, `mistral3`** — llama/qwen-shaped with one of: a norm moved, a
  `logit_scale`, a `residual_scale`, an activation swap, a QK-norm variant.
  Several are probably one `Flags` bit each. **Each needs its own reading; do
  not batch-claim them.**

---

## 3. What jitllm already has, with the symbol that proves it

A plan that ignores this over-estimates the work badly. Every row is a real
asset for the families above.

| asset | where | what it covers |
|---|---|---|
| the architecture switch is ONE function | `convert.configOf`, `convert/config.go:67` | a new name is a case arm, not a graph builder |
| the graph is config-driven | `grep -c Arch engine/model/forward.go` = **0** | no per-architecture runtime code |
| per-layer kinds in the container | `jlm.LayerKinds`, one byte per layer | hybrids, irregular patterns, and the natural home for per-layer dense/MoE |
| gated delta rule, host + all 3 device backends | `kernels.GatedDeltaStep`, `Conv1dRows`, `Conv1dShift`, `DeltaGate` | qwen3next; the base for KDA and most of Mamba-2 |
| recurrent state migration | `MigrateRec`, `syncRecFromDevice`, `pushRecToDevice`, `recPair` | any recurrent family, on the device |
| shared expert + its gate | `Config.NFFNShExp`, `RoleShRouter/ShExpGate/ShExpUp/ShExpDown` | DeepSeek's `n_shared_experts`, qwen2moe, glm4moe, hunyuan |
| router bias | `jlm.RoleRouterBias` (44), `engine/model/moe.go:72` | gpt-oss today; the tensor DeepSeek-V3's `e_score_correction_bias` needs |
| "do not renormalise the mixture" | `jlm.FlagNoExpertNorm` (bit 4) | olmoe today; DeepSeek-V2-Lite's `norm_topk_prob: false` |
| the router as a generated kernel, no Go fallback | `nn.MoETopK32JIT`, `cpu.MoETopK(n,k,norm)`, `kernels.ExpertRank` | the thing a new routing rule extends |
| YaRN, exact or truncated | `Config.Yarn*`, `FlagYarnExact`, `ropeScalingOf` | DeepSeek's rope scaling minus one multiplier |
| partial rotary | `Config.NRot < HeadDim`, `RoPERows` writes only what it rotates | DeepSeek's 64-of-192 rope split |
| sliding window with an arbitrary period | `Config.SWAWindow`/`SWAPeriod`, `allLocal()` | gemma2/3, gpt-oss, phi3; llama4's chunked local attention is adjacent |
| two logit softcaps | `AttnSoftcap`, `FinalSoftcap` | gemma2; `router_logit_softcapping` is the same shape |
| attention sinks | `RoleAttnSinks` (30), one slot on the score row | gpt-oss; several newer models reuse the idea |
| fused-tensor unfusing at conversion | `convert.unfuse`, gated on `ArchPhi3` | any model shipping a fused qkv or gate/up |
| 66 generated pre-tokenizer pipelines | `tok/pretok/table.go`, `scripts/gentok` | `deepseek-v3`, `glm4`, `minimax-m2`, `hunyuan`, `bailingmoe2`, `exaone4`, `cohere2moe`, `falcon-h1`, `lfm2`, `afmoe`, `qwen35` |
| a tokenizer override path | `tok.WithTokenizer(r)` + `pretok.ParsePreTokenizer` | rescues a `pre` name the table does not carry |
| the safetensors input path | `convert/safetensors.go`, `hfArchTable` | a model with no GGUF can still be converted |
| the transformers oracle | `scripts/hfgold.py`, `model.TestSafetensorsMatchTransformers` | a tiny-random fixture gates a graph without the real weights |
| the llama.cpp oracle, with deepseek2 in it | all four `libllama.so` on this box | per-node intermediates for every family in section 1 |

---

## 4. The order, and the reasoning

RULE 6: a number decides ORDER, never admission. Nothing below is refused for
being small; the sequence is about breadth per unit of work and about what
unblocks what.

### Step 0 — YaRN's log multiplier (Capability D)

One multiply. It is a hard prerequisite for every DeepSeek model, it has a
standing refusal in the tree naming it, and it is gateable against
`transformers._compute_yarn_parameters` the same way
`nn.TestYarnFreqsMatchTransformers` already gates the gpt-oss path. Do it first
because it is the cheapest thing on the critical path, not because it is
valuable alone.

### Step 1 — Per-layer dense/MoE (Capability A)

A Config field, a version bump, seven call sites. **Eighteen architectures need
it and none of them can be started without it.** Highest breadth-per-unit-work
of anything in this document. It ships no model by itself, which is the only
argument against putting it first — and Step 0 does not ship one either.

### Step 2 — MLA, targeting DeepSeek-V2-Lite (Capability C)

The big one, and V2-Lite is the right first target for three reasons that are
about verification rather than about ambition:

- its router is **already expressible** (softmax + plain top-k + no renorm +
  shared experts), so MLA is the only new thing in the run and a wrong answer
  has one possible cause;
- it has `q_lora_rank: null`, so the query projection is ordinary and only the
  KV side is low-rank — half the new machinery;
- it fits: **10.36 GB** on a box with 68 GB free and 31 GB of RAM, and
  `llama-eval-callback` on this machine dispatches `deepseek2`, so the node-by-
  node gate that verified gpt-oss (`model.TestGPTOSSMatchesLlamaCppIntermediates`)
  transfers directly.

### Step 3 — The DeepSeek-V3 router (Capability B)

sigmoid + grouped top-k + selection bias + weight scale. With Steps 1 and 2 done,
this is what turns "V2-Lite runs" into "V3, R1 and Kimi-K2 run", and it
simultaneously unlocks glm4moe, dots1, bailingmoe2, hunyuan-moe and minimax-m2's
routers. Gate it on a `tiny-random` DeepseekV3 fixture against transformers —
`trl-internal-testing/tiny-DeepseekV3ForCausalLM` is 16.9 MB and transformers has
native `deepseek_v3` support, so `scripts/hfgold.py` runs it with no remote code.

**★ AND KIMI-K2 NEEDS ONE MORE THING, WHICH IS ITS TOKENIZER AND NOTHING ELSE.**
llama.cpp's converter carries an explicit `if tokpre == "kimi-k2":` branch that
rebuilds the merge list from tiktoken's `_mergeable_ranks`, and
`tok/pretok/table.go`'s footer records `kimi-k2 (moonshotai/Kimi-K2-Base): http 404`
— a dead repo pointer in llama.cpp's own directory, not a missing capability.
`Kimi-K2-Instruct` exists. Re-running `go run ./scripts/gentok` against a live
pointer is the whole of that work, and RULE 11 says the 404 is a task.

**At the end of Step 3, DeepSeek-V3/R1 and Kimi-K2 are architecturally supported
and cannot be run on this hardware.** That is the honest state, and it is the
same state gpt-oss-120b was in.

### Step 4 — Q2_K, and ONLY Q2_K (Capability G)

**This step shrank when the file census came back.** The premise this plan
started on — that I-quants are what people actually distribute for 671B models,
and therefore the highest-leverage item of all — is refuted (section 2G).
DeepSeek-R1 and Kimi-K2 ship straight Q4_K_M, Q6_K and Q8_0, and Unsloth's
UD-Q4_K_XL contains zero I-quant tensors. **jitllm's existing `Dequantable` set
already covers every one of those files.**

What is actually locked out is the whole 2-bit tier, for want of **one ggml type,
`Q2_K` (type 10)**. That opens `UD-Q2_K_XL` — the variant Unsloth itself
recommends as the sweet spot — plus straight `Q2_K`/`Q2_K_L`/`Q2_K_XS` across
every DeepSeek and Kimi repo. And it is the cheapest of the missing types anyway,
because its 2-bit primary plane is the same emitter shape AGENTS.md already lists
as owed for native Q3_K packing: **one piece of work, two formats.**

**Explicitly NOT the I-quants, and now for a measured reason rather than a
taste.** They buy the sub-2-bit tier specifically — `UD-IQ1_S` at 140.2 GB is the
only DeepSeek-R1 file that fits a 128 GB box — which is a real audience and not
this project's critical path. When it is reopened, reopen it at **IQ4_XS/IQ4_NL
first** (a 16-entry table lookup on top of Q4_K) and leave the 1-3 bit codebook
grids until someone prices the gather.

### Step 5 — Mamba-2 (Capability E)

Eight hybrid families on scaffolding that already exists and is gated. This is
the best remaining breadth-per-unit-work after Step 1, and it is independent of
everything above, so it can be interleaved or handed to someone else.

### Step 6 — Kimi Delta Attention (Capability F)

**This is what finishes Kimi-Linear**, and it is sequenced after Steps 1-3 not by
preference but by dependency: Kimi-Linear's full-attention layers are MLA and its
FFN is a sigmoid-routed mixture with a dense first layer, so A, B and C all
precede it. The KDA half itself is a scalar-to-channel-wise change in one IR
kernel and its host twins, plus the low-rank gate projections — small, and a
LEAD rather than a costing until Moonshot's chunkwise kernel has been read.

### Not scheduled

DSA (Capability H), vision towers (J), and the cheap-tail names (K) — see
section 5 and section 6.

---

## 4b. Kimi-Linear, priced against the engine as it stands

Written after deepseek2 shipped, because "kimi" now means two different graphs
and only one of them is covered. Kimi-K2 declares `DeepseekV3ForCausalLM` and
RUNS (Moonlight-16B-A3B, its sibling, verifies at 19 of 20 teacher-forced
positions against llama.cpp). **Kimi-Linear is a separate architecture** --
`KimiLinearForCausalLM`, `model_type: kimi_linear` -- and this section is the
material for the RULE 7 decision on it, not an argument for taking it.

### What it is

27 blocks: **7 full-attention and 20 KDA**, named explicitly in
`linear_attn_config.full_attn_layers` / `kda_layers` rather than as a prefix.
The full-attention blocks are MLA (`kv_lora_rank` 512, `qk_nope_head_dim` 128,
`qk_rope_head_dim` 64, `v_head_dim` 128) -- **the graph jitllm already runs**.

### The one sentence that decides the cost

transformers' own class docstring:

> Kimi Linear Attention: this is essentialy the same a gated delta net (GDN)
> but **decay is per-channel instead of per-token**.

`kernels.GatedDeltaStep` declares `pDecay` as `[vHeads]` and reads it once per
head (`b.Load(ir.F32, pDecay, vh, 0)`). KDA's decay is `[heads][head_dim]` and
has to be read per CHANNEL inside the inner loop. **That is a kernel change on
every backend, not a parameter.**

★ **AND THE "SECOND SHAPE RATHER THAN AN EDIT" HALF OF THAT TURNED OUT WRONG
ONCE THE REFERENCE WAS READ INSTEAD OF THE SHAPE. IT IS ONE OPERAND
WIDER, AND THE HOST KERNEL IS BUILT ON ALL THREE TIERS.** The pricing committed
in `856119e` said KDA needed a second kernel beside the per-head one because
the decay is `[heads][head_dim]`. What that did not ask is WHICH AXIS the
head_dim is. `recurrent_kimi_delta_attention` keeps the state as
`[k_head_dim, v_head_dim]` and applies `g_i = g[:, i][..., None].exp()` -- a
decay of k_head_dim broadcast over v_head_dim -- and reduces over `dim=-2`,
the k axis. **This tree holds the transpose**: `state[j]` is a row indexed by
the VALUE dim and `dot(state[j], k)` reduces along the row, so the decay varies
along the WITHIN-ROW index and is IDENTICAL for every row j.

That is the axis the emitters already walk in vectors, so the change is an
indexed load where the scalar form broadcasts once:

    scalar        VBROADCASTSS(Y10, At(R10, 0))  once, then VMULPS(Y8, Y8, Y10)
    per-channel   VMULPSMem(Y8, Y8, At(R11, i*32))

`EmitGatedDeltaChan` is that on AVX2, SSE and NEON (`AScale2` carries the n
floats; `Scr[0]` goes unread), gated against `oracle.GatedDeltaChan` at 21
ragged widths over four tokens with guards on every buffer -- the decay buffer
included, because it is the one argument read as a VECTOR and a tail running
the vector body walks off exactly its end.

★ **THE GATE CARRIES ITS OWN DISCRIMINATION CHECK, AND THE REASON IS THAT A
BROADCAST BUG IS A DIFFERENT ARCHITECTURE RATHER THAN A WRONG NUMBER.** A
per-channel kernel that broadcasts `decay[0]` computes qwen3next's rule and
stays fluent. So the fixture runs the SCALAR kernel at `decay[0]` on the same
inputs and REQUIRES it to disagree with the per-channel reference: without
that, a decay vector whose entries happened to be close would score the bug as
a pass. It also means the arm64 arm needs no separate violation run -- the
per-channel kernel matching the per-channel oracle WHILE the scalar one misses
it is proof the NEON code indexes rather than broadcasts. Verified on the M4.

**The DEVICE half was built the same day**: `kernels.GatedDeltaStepChan` moves
the load inside both k loops, indexed by the VALUE head -- k and q are shared
by the rep value heads of one key head, but the forget gate produces one vector
per value head, so indexing the decay by `kh` is right at rep=1 and wrong above
it.

★ **AND WHAT IS LEFT IS THE GRAPH, WITH ONE KERNEL STILL OWED THAT THE FIRST
SURVEY MISSED: THE OUTPUT GATE IS A SIGMOID, NOT A SiLU.**
`KimiLinearRMSNormGated` sets `self.activation = "sigmoid"` and applies
`hidden_states * ACT2FN["sigmoid"](gate)`, where qwen3next's linear block
multiplies by SiLU. `cpu.ActKind` is {SiLU, GELU, QuickGELU, SwiGLUOAI} -- no
sigmoid -- so this is a generated kernel on three host tiers and in the device
IR, not a branch (RULE 8). The exp machinery it needs already exists: the gate
kernel computes `beta[i] = sigmoid(b[i])` on every tier today.

**Status: SHIPPED and verified.**
`model.TestSafetensorsMatchTransformers/synth-kimilinear` reads **8 of 8
positions, worst NMSE 5.08e-11 against a 1e-9 bound**.

    the container            DONE  arch, interleaved LayerKinds, the SSM
                                   geometry, the fused q|k|v and conv1d, the
                                   per-channel rate. The fuse ORDER is gated
                                   against the sources.
    the delta kernel         DONE  AVX2, SSE, NEON and the device IR on
                                   ptx/spirv/msl, with a discrimination arm
                                   against the per-head rule.
    the four gate roles      DONE  RoleSSMFA/FB/GA/GB.
    the sigmoid output gate  DONE  and it needed NO new kernel:
                                   nn.SigmoidMul32JIT already existed on all
                                   three host tiers for qwen3next's FULL
                                   blocks. The survey that called it owed had
                                   read cpu.ActKind's exported set instead of
                                   grepping for the capability -- RULE 12 in a
                                   comment written the same day.
    linearAttn's KDA path    DONE  the low-rank forget gate feeding
                                   DeltaGate32JIT at vHeads*kDim, beta alone
                                   from b_proj, GatedDeltaChan in delta(),
                                   and the q/k L2 norm at the reference's own
                                   hardcoded 1e-6 rather than rms_norm_eps.
    NoPE attention           DONE  FlagNoPosEnc. See below -- it is the bug
                                   that is EXACT at position 0.
    the device, MLA half     DONE  model.TestNoPEMLAOnEveryDevice: the
                                   full-attention block on cuda and vulkan at
                                   worst logit NMSE 1.0e-12 / 9.4e-13 against
                                   the host, with the linear blocks declined
                                   BY NAME and that name asserted.
    the device, KDA half     DONE on CUDA. All 3 blocks placed, 2 of them
                                   linear, worst logit NMSE 4.993e-12 against
                                   the host. emitLinear branches on
                                   RecurrentPlan.ChanDecay: the two low-rank
                                   gates chain two matvecs each through a
                                   quantized rank-r activation, DeltaGate runs
                                   twice at two widths, GatedDeltaStepChan
                                   replaces GatedDeltaStep, and the output
                                   gate is SigmoidMul. NO new kernel was
                                   needed -- SigmoidMul already existed for
                                   qwen3next's attention out-gate.
    the device, SPIR-V       DONE  3 of 3 blocks on vulkan at worst logit
                                   NMSE 7.073e-12. It was blocked by a bug
                                   older than KDA that broke EVERY hybrid on
                                   that backend: Ctx held ONE command buffer
                                   shared by the recorded Batch and by the
                                   one-shot copyBuf/Kernel.Launch, so a
                                   Session.Write from inside emitLinear reset
                                   the buffer the batch was recording into and
                                   Vulkan silently dropped every dispatch
                                   after it. Ctx.one is a second buffer for
                                   the one-shots. Found by the Khronos
                                   validation layer after every other
                                   candidate had been eliminated by
                                   measurement.
    a rank not a multiple of 32     DECLINED BY NAME. KDA's low-rank gates
                                   chain through a quantized activation of
                                   rank width and kernels.Quantize works in
                                   32-element blocks. Every shipped
                                   Kimi-Linear uses a multiple of 32; the
                                   fixture does now too, and the first version
                                   of it did not, which is how the decline got
                                   written.
    arm64 end to end         DONE  8 of 8 positions on the M4, worst NMSE
                                   5.60e-11. The arm64 test run ships the
                                   HuggingFace fixtures now -- see below,
                                   because the gap was never Kimi-Linear's.

### What already exists, with the symbol

| KDA needs | jitllm has |
|---|---|
| a per-layer kind map | `jlm.LayerKinds` -- built for exactly this |
| the causal conv over a shifting history | `kernels.Conv1dRows`, `Conv1dShift` |
| two state buffers and a swap | `tier.recPair`, and `State.rconv`/`rstate` |
| per-head log decay, delta-t bias | `RoleSSMA`, `RoleSSMDtBias` |
| a gated RMSNorm over one head | `RoleSSMNorm` |
| MLA on the 7 full blocks | shipped, gated on three backends |
| the `kimi-k2` pre-tokenizer | shipped |

### What it costs, and none of it is a flag

1. **`GatedDeltaStep` with a per-channel decay.** The device IR plus the host
   path. It is the same kernel with the decay moved inside the k loop, and it
   is a second shape rather than an edit -- qwen3next's models still want the
   per-head one.
2. **A forget-gate kernel.** `KimiLinearForgetGate` is
   `f_b_proj(f_a_proj(x)) + dt_bias`, then `-exp(A_log) * softplus(g)` with a
   `g > 20` cutoff. The low-rank pair is two matvecs jitllm has; **the softplus
   with that cutoff is new**, and RULE 8 says it is generated, not a Go loop.
3. **A LOW-RANK OUTPUT GATE.** `g_a_proj`/`g_b_proj` where `RoleSSMOutGate` is
   one matrix. Two roles, or one role that carries a rank.
4. **SPLIT q/k/v.** Kimi projects them separately where qwen3next fuses into
   `RoleSSMInProj`. This is the `convert.unfuse` move in reverse -- JOIN them
   while the bytes are still row-major -- so it costs conversion code and no
   new container shape, exactly as phi3's split did.
5. **Config plumbing.** `linear_attn_config` into `LayerKinds`, plus
   `linear_num_heads`, `linear_head_dim`, `short_conv_kernel_size`.
6. **A fixture and a golden** (RULE 7). `KimiLinearForCausalLM` is in
   transformers 5.17, so `scripts/hfgold.py`'s `SYNTH` map can build one the
   way `synth-glm4moelite` was built.

### The comparison that matters

This is **qwen3next's shape of cost, not deepseek2's**. deepseek2 was admitted
because one entry bought four product lines; Kimi-Linear is one model and needs
two new kernels. qwen3next cost "a BF16 dequantizer, a per-layer kind map, an
SSM geometry, a shared expert, a double-width query, partial rotary, and three
new kernels on two architectures" -- and was added on the user's explicit
decision, as deepseek2 was.

**So it is priced, and it is a decision. Nothing here recommends it.**

---

## 4c. The state-space hybrids and the short convolutions, priced

Every price below is read off llama.cpp's builder (`src/models/<arch>.cpp`,
`mamba-base.cpp`), its converter (`conversion/<family>.py`) and transformers'
own forward (`models/<family>/modeling_*.py`), against the engine as it stands.
The four mixtures of the same request (bailingmoe2, dots1, phimoe, apertus) are
priced in their own subsection.

### The one sentence that decides most of it

**A Mamba-2 head IS a gated linear-attention head over the state layout this
tree already keeps.** `ggml_compute_forward_ssm_scan_f32` holds the state
`{d_state, head_dim, n_head}` -- per head, `head_dim` rows of `d_state` -- which
is `deltaGeom`'s `[vHeads][vDim][kDim]` with B as the key (`ngroups` key heads
of `d_state`), C as the query and x as the value, value head h reading group
`h / (nheads/ngroups)` exactly as qwen3next's grouped pairing does
(`repeat_interleave`, both references). Per token and row j:

    dt   = softplus(dt_raw + dt_bias)           DeltaGate's softplus, unchanged
    S[j] = S[j]*exp(A*dt) + (x[j]*dt)*B         the delta rule with no "- S k"
    y[j] = dot(S[j], C) + D*x[j]                and beta replaced by dt

So the recurrence is the delta rule's kernel with one pass removed (no key
dot, no subtraction) and one fused skip added: a VARIANT of `EmitGatedDelta`,
`gatedDeltaStep` and `gatedDeltaScan` (with its runs and one-thread forms),
not a new family. Everything around it is built and gated: `recPair`,
`MigrateRec`, the prefix cache's one restore point, the ragged step's run
chaining, `ResetRecurrent`, the spec snapshots -- all keyed on
`LayerLinearAttn`, which a Mamba-2 layer is.

What is new for Mamba-2 itself:

| piece | why | cost |
|---|---|---|
| the SSD variant of the rule | no key dot, beta := dt, y += D*x | one flag through 3 host emitters, 3 IR forms, the oracle |
| a Mamba gate | dt itself is an operand, not only exp(A*dt) | `DeltaGate` with a third output, or formed inside the fused scan |
| conv bias, D | two per-block vectors | two roles; the bias rides the fused scan's SiLU |
| the GATED norm in the other order | Mamba normalises `y*silu(z)`, qwen3next `norm(y)*silu(z)` | an order switch, and a GROUPED norm (`d_inner/ngroups`) with a full-width weight: `HeadNorm` with a per-group weight |
| in_proj is z \| xBC \| dt fused | one matrix where qwen3next ships three | split at conversion (a row range is a byte range in GGUF), and xBC's rows and conv1d's filters REORDERED to C \| B \| x so `deltaGeom` slices it as q \| k \| v unchanged |
| a block with no FFN | plain mamba/mamba2 and every Nemotron-H mixer layer | the FFN half skipped on every host entry point and on the device |

Two references disagree and the choice is written down (RULE 7m):

- **The gated norm's groups.** mamba_ssm (the authors' code), llama.cpp and
  transformers' Nemotron-H group it by `ngroups`; transformers' `Mamba2` and
  `GraniteMoeHybrid` normalise all of `d_inner` at once. They agree at
  `ngroups = 1`, which is every Granite 4.0-H and every state-spaces Mamba-2;
  Mamba-Codestral (8 groups) is where they part. Grouped, as the authors' code.
- **The dt clamp.** transformers' prompt path clamps `dt` to
  `time_step_limit`, its decode path does not, llama.cpp never does, and a GGUF
  does not carry the limit. Not clamped; fixtures keep `dt` off the limit.

### Per family

| family | arch | what it needs beyond Mamba-2 | cost |
|---|---|---|---|
| **Mamba-2** | `mamba2` | nothing but the no-FFN block; tied or untied head | the cheapest: one config case |
| **Granite 4.0-H** | `granitehybrid` | Granite's four scales (built, `graniteConfig`), NoPE attention (`rope.scaling.finetuned` false -> `FlagNoPosEnc`, built), per-layer `head_count_kv` with zeros on the recurrent layers, a softmax mixture beside an ungated shared MLP (built) | a config case |
| **Nemotron-H / Nano 2** | `nemotron_h` | each HF layer is ONE mixer (M, \*, -) with its own pre-norm; a mixer followed by an MLP is merged at conversion into one block (the MLP's norm becomes `ffn_norm`), a mixer followed by a mixer is a block with no FFN. Ungated relu^2 MLP (built, `FlagReLU2`) | the merge in the converter, the no-FFN block |
| **Nemotron-H MoE** (Nemotron 3 Nano) | `nemotron_h_moe` | the 'E' layer: a sigmoid top-k with a selection bias and a routed scale (DeepSeek-V3's, built) over UNGATED relu^2 experts beside an ungated shared expert; latent MoE projections (Super/Ultra only) refused by name | a mixture whose experts have no gate bank |
| **Falcon-H1** | `falcon-h1` | attention AND Mamba-2 in PARALLEL on the same normed input, summed, then the FFN; every muP multiplier folded into the weights by llama.cpp's converter (`ssm_multipliers` per channel of in_proj) | a block of a new shape: two mixers, one residual, on both tiers |
| **Jamba** | `jamba` | Mamba-ONE layers (below), NoPE attention, a softmax top-2 mixture every other layer | needs Mamba-1 |
| **Mamba** (v1) | `mamba` | a per-CHANNEL A (`[d_inner][d_state]`), `x_proj` -> (dt_rank, B, C), `dt_proj`, no groups, no norm, no FFN; Jamba adds RMS norms on dt, B and C | the v1 scan is the per-channel-decay form (`GatedDeltaChan`'s shape) at head width 1, plus a decay kernel `exp(dt[c]*A[c][:])` |
| **LFM2** | `lfm2` | a SHORT gated convolution layer: in_proj -> B, C, x; conv1d(B*x) with no bias and no activation; C*out; out_proj. Its state is the convolution window alone (`l_cache-1` planes). Attention layers carry q/k RMSNorm | a third layer kind: conv state without an SSM state; the conv kernels exist |
| **LFM2-MoE** | `lfm2moe` | LFM2 plus a sigmoid top-k with a selection bias after a dense lead | composition |

The order follows the dependency, not the size: the SSD variant and the no-FFN
block first (Mamba-2), then the families that are configuration over them
(Granite 4.0-H, Nemotron-H), then the two that need a new block shape
(Falcon-H1's parallel mixers) or a second recurrence (Mamba-1, Jamba), with
LFM2's short convolution beside them.
## 4c. Ling 2.0, dots.llm1, Phi-3.5-MoE and Apertus, priced against the engine as it stands

Four text families the user admitted together ("support all modern models").
Each is read off llama.cpp's builder (`src/models/<arch>.cpp`), its converter
(`conversion/<family>.py`) and the reference class, and priced against the
symbols that already run. Two are compositions; two need one kernel each on
every tier. Arch codes 39..42 (`format/jlm/archmoe2.go`).

### Ling / Ring 2.0 (`bailingmoe2`, `BailingMoeV2ForCausalLM`, code 42)

The reference is inclusionAI's remote code (`modeling_bailing_moe_v2.py`
shipped with Ling-mini-2.0); transformers has no class for it.

| feature | the reference | already runs as |
|---|---|---|
| fused `query_key_value` | one matrix, q|k|v by head | `convert.unfuse` (`fusesQKV` gains the arch) |
| per-head q/k RMSNorm before the rotary | `query_layernorm`/`key_layernorm` | `jlm.FlagQKNorm` |
| NEOX rotary on half the head | `partial_rotary_factor` 0.5 | `partialRotary`, `FlagRopeNeox` |
| sigmoid router, selection bias | `expert_bias` buffer added for selection only | `FlagExpertSigmoid`, `RoleExpProbsB` |
| group-limited top-k | `n_group` 8, `topk_group` 4, group score = top-2 sum | `NExpertGroup`, `ExpertGroupScore` |
| renormalised `+1e-20`, routed scale 2.5 | `BailingMoeV2Gate.forward` | `MoESumFloor`, `ExpertScale` |
| shared expert, dense lead | `num_shared_experts`, `first_k_dense_replace` | `NFFNShExp`, `NDenseLead` |
| MTP layers | `num_nextn_predict_layers` (0 on Ling-mini) | `nextnOf`, set aside |

**Cost: none of it is new.** The arch code, `fusesQKV`, the router gate held
to sigmoid (the class hardcodes `torch.sigmoid`) and a fixture whose
`expert_bias` buffer is randomised (it is a buffer, zero-initialised, the
identity otherwise).

### dots.llm1 (`dots1`, `Dots1ForCausalLM`, code 41)

DeepSeek-V3's router (`Dots1TopkRouter`: `router_logits.sigmoid()` hardcoded,
`e_score_correction_bias`, grouped top-k, `+1e-20`, `routed_scaling_factor`)
on GQA attention with qwen3's per-head q/k norm and NEOX rotary over the whole
head, a dense lead and two shared experts. **Every piece runs** (glm4moe's
router, qwen3's attention). `sliding_window` exists in the class; the
published config has `use_sliding_window: false` and llama.cpp applies none,
so a file stating one is refused by name. Real checkpoint is 142B (~88 GB at
Q4_K_M): the V100 box's RAM holds it, the laptop does not.

### Phi-3.5-MoE (`phimoe`, `PhiMoEForCausalLM`, code 40)

| feature | transformers (eval) | llama.cpp | here |
|---|---|---|---|
| block norm | `nn.LayerNorm` with bias | RMSNorm plus a bias (`LLM_NORM_RMS`) | the C6 kit's `FlagLayerNorm` |
| q/k/v/o biases, head bias | `attention_bias`, `lm_head_bias` | same | tensors |
| router | **sparsemixer**, top-2 | softmax top-2, renormalised | a new generated kernel |
| longrope | short factors, `short_mscale` on cos/sin | short factors, `attn_factor` derived from the context ratio | phi3's short-factor path |

**Two of llama.cpp's choices differ from the model's own class** (RULE 7m),
and the engine matches the class: sparsemixer's eval path is

    m1 = max(s); i1 = argmax(s)
    keep_j  =  (m1 - s_j) / max(|s_j|, m1)  <=  2 * router_jitter_noise
    w1 = 1 / SUM_{kept j} exp(s_j - m1)
    i2 = argmax(s with i1 removed); m2 = s_i2
    keep2_j =  (m2 - s_j) / max(|s_j|, m2)  <=  2 * router_jitter_noise,  j != i1
    w2 = 1 / SUM_{kept2 j} exp(s_j - m2)

with no renormalisation -- each weight is near one, where llama.cpp's
renormalised softmax makes them sum to one. `router_jitter_noise` (0.01) is
the one constant, a config value the GGUF does not carry (llama.cpp ignores
it); the converter takes Phi-3.5-MoE's 0.01 and the safetensors path reads
it. The LayerNorm is a real one: llama.cpp's RMSNorm-with-bias skips the mean.
`short_mscale` (1.243 on Phi-3.5-MoE) is not in a GGUF either -- llama.cpp
writes its own `attn_factor` (1.190) -- so the GGUF path carries the file's
factor and the safetensors path the class's; the two inputs differ by that
number on the real model, stated rather than hidden.

**Cost: the sparsemixer**, one kernel on four tiers -- the host router
(AVX2, SSE, arm64, beside `cpu.MoERoute`) and the device weights kernel
(beside `kernels.ExpertWeights`; the selection is `ExpertRank`'s plain logit
rank, which is sparsemixer's argmax order). Real checkpoint 42B (~25 GB at
Q4_K_M): V100 box.

### Apertus (`apertus`, `ApertusForCausalLM`, code 39)

llama's block with qwen3's per-head q/k RMSNorm, llama3 rotary
(`rope_freqs`), and an UNGATED MLP (up, xIELU, down) whose activation carries
four numbers per layer:

    alpha_p = softplus(a_p)        alpha_n = beta + softplus(a_n)
    x > 0:   alpha_p x^2 + beta x
    x <= 0:  alpha_n (expm1(min(x, eps)) - x) + beta x         eps = -1e-6

`beta` and `eps` are buffers. llama.cpp's converter writes the raw `a_p`,
`a_n`, `beta`, `eps` per layer and its graph applies the softplus at build
time; here the converter applies it once (precompute at conversion) and the
block carries the four effective values as a per-block vector, so the device
uploads them with the block and relocation moves them. `expm1(m) - x` is
written as `g(m) + (m - x)` with `g(x) = expm1(x) - x` a polynomial near zero:
`exp(x) - 1 - x` in f32 loses every digit of the x^2/2 it is made of.

**Cost: xIELU**, one ungated activation with a parameter vector on every
tier (AVX2, SSE, arm64; the device IR lowered to PTX, SPIR-V and MSL), and a
role for the vector. Real checkpoint Apertus-8B (~5 GB at Q4_K_M): laptop.

---

## 5. What NOT to chase

RULE 7: *"llama.cpp has 176 `LLM_ARCH_` cases and 50 graph builders. Ollama had a
funded team, covered ~21 architectures, and deleted 430,004 lines citing exactly
this. Scope is jitllm's only defence."* The installed build's table is 130 names;
llama.cpp `master` is at **153**. Matching it is not a goal and should not become
one by drift.

**★ AND vLLM IS A SECOND, LIVE INSTANCE OF THE SAME LESSON, WHICH IS WORTH MORE
THAN THE OLLAMA ANECDOTE BECAUSE IT IS ONGOING.** `registry.py` keeps a
`_PREVIOUSLY_SUPPORTED_MODELS` table — architectures vLLM removed, each tagged
with the version that dropped it. It is 50+ entries and growing, and it contains
models that were current news when they landed:

    MllamaForConditionalGeneration  0.10.2   (Llama-3.2 Vision)
    Dots1ForCausalLM                0.23.0
    MiniMaxText01 / MiniMaxM1       0.23.0
    BambaForCausalLM                0.23.0
    XverseForCausalLM               0.23.0
    Grok1ForCausalLM                0.24.0
    Plamo2ForCausalLM               0.26.0
    ArcticForCausalLM               0.28.0
    ChameleonForConditionalGeneration 0.28.0
    MPTForCausalLM                  0.28.0

The best-resourced open serving engine in this space **prunes**, and it prunes
things it shipped two releases ago. A project with one maintainer that adds an
architecture per week is choosing a future deletion. The second-order lesson is
also worth taking: vLLM can prune because its `TransformersForCausalLM` fallback
catches what it drops. jitllm has no fallback (RULE 8 forbids an interpreted
tier by design), so **every name jitllm adds is a permanent obligation**, which
makes the bar higher here than it is there.

**Refuse, with the reason recorded:**

- **Encoder-only and embedding models** — `modern-bert`, `nomic-bert`,
  `nomic-bert-moe`, `neo-bert`, `jina-bert-v2`, `jina-bert-v3`, `eurobert`,
  `gemma-embedding`, `llama-embed`, `t5encoder`, `roberta`. `State.Forward`
  returns logits over a vocabulary; a pooling head is a different return type and
  a different product. ROADMAP item 2 already contemplates BERT and calls it "the
  first NON-generative model"; that is a decision to take deliberately, not a
  coverage gap to close.
- **Non-text output** — `wavtokenizer-dec`, `chameleon`, `paddleocr`,
  `deepseek2-ocr`'s vision half. Different modalities, different everything.
- **Draft/speculation scaffolding** — `eagle3`, `dflash`, `nextn_predict_layers`.
  These are serving features wearing an architecture's clothes; they belong with
  ROADMAP item 4, not here.
- **The 1-3 bit I-quants** — `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
  `IQ3_XXS`, `IQ3_S`. Codebook grids against a plane-and-scale design. Say no by
  measurement when someone prices the gather, not by assertion — but do not start
  here.
- **Audio, ASR and omni** — about a third of vLLM's multimodal registry. Nothing
  in jitllm points this way and nothing should.
- **RWKV / ARWKV** — `rwkv6`, `rwkv6qwen2`, `arwkv7`. A genuinely different
  recurrence (token shift, WKV, per-channel time-mix) with its own tensor
  vocabulary, and no overlap with the delta-rule machinery despite both being
  "linear attention". **vLLM has never supported it at all** — zero hits in
  `registry.py` including the removed list — which is a useful second opinion.
- **The long tail of one-model architectures** — `deci`, `grok`, `arctic`,
  `xverse`, `openelm`, `orion`, `codeshell`, `jais`, `jais2`, `bitnet`,
  `nanbeige`, `maincoder`, `rnd1`, `step35`, `mimo2`, `pangu-embedded`,
  `talkie`, `mellum`, `dream`, `llada`, `llada-moe`, `apertus`, `grovemoe`,
  `laguna`, `afmoe`. Each is a decision with no leverage behind it. Four of them
  (`grok`, `arctic`, `xverse`, plus `dots1` and `plamo2` from elsewhere in this
  document) are on vLLM's DELETED list, and `grovemoe`, `bitnet`, `openelm`,
  `llada`, `codeshell` and `rnd1` it never had.
- **DSA** (`deepseek32`, `deepseek4`, `glm-dsa`) — not "never", but not until MLA
  ships and there is a model of it small enough to verify. Building the sparse
  indexer before the dense MLA path is correct would mean debugging two new
  things at once, which is the failure mode AGENTS.md spends pages on.

**The rule to hold the line with:** admit a capability that unlocks several
families; admit a name that needs no capability at all and costs a converter case
plus a gate; refuse a name that needs a capability of its own and unlocks nothing
else. That is a rule about the ratio, and it is the only one that survives 130
names.

---

## 6. Verification cost per family

This project treats an ungated entry as "a claim with nothing behind it"
(RULE 7). The verification machinery is the constraint, so it is worth stating
per family which oracle exists.

### The oracles that exist on this box

1. **llama.cpp per-node intermediates.** All four builds
   (CUDA and Vulkan builds under `$JITLLM_LCPP`,
   a CPU and a Vulkan release build of b10825) carry `deepseek2`,
   `glm4moe`, `kimi-linear`, `llama4` and `mamba2` in their arch table — checked
   by `strings`. So `llama-eval-callback` is a node-by-node oracle for every
   family in this document **provided a GGUF small enough to run exists**.
2. **transformers, via `scripts/hfgold.py`** into
   the reference venv (`$JITLLM_HF_PY`; torch 2.14 CPU, transformers 5.17), gated by
   `model.TestSafetensorsMatchTransformers` against teacher-forced logits over
   fixed ids. **This one does not need a real model at all** — the existing
   goldens are `tiny-random-*` fixtures of a few MB, and random weights make
   every wiring error an uncorrelated logit vector rather than a near-miss.

   ★ **AND IT DOES NOT NEED A DOWNLOAD EITHER, WHICH IS THE STRONGEST RESULT IN
   THIS SECTION.** `hfgold.py`'s `SYNTH` table builds a fixture from
   transformers' OWN config class — `AutoConfig.for_model(model_type, **kw)`
   then `from_config` — at `hidden_size 64, layers 2, head_dim 16`, with
   **random norm weights so every norm wiring error shows** and a donor
   tokenizer copied in. `synth-gemma`, `synth-mixtral`, `synth-qwen2` and
   `synth-phi3` are made that way today, and `synth-phi3` carries
   `sliding_window=4` precisely so the window BITES inside the eight golden
   positions.

   A `synth-deepseek-v3` entry with `kv_lora_rank`, `q_lora_rank`,
   `n_routed_experts=8`, `num_experts_per_tok=2`, `n_group=2`, `topk_group=1`
   and `first_k_dense_replace=1` would gate **MLA, the sigmoid grouped router,
   the per-expert selection bias, the routed scaling factor, the shared expert
   AND the per-layer dense/MoE split** — capabilities A, B and C together — in a
   fixture of a few megabytes, against the reference implementation, with
   nothing downloaded. That is the same trick `synth-phi3` uses to gate a
   sliding window without a 4k prompt.

   The one thing to check first: that transformers 5.17 has `deepseek_v3`
   natively (it appears at `huggingface.co/docs/transformers/model_doc/
   deepseek_v3`, so almost certainly yes) — `hfgold.py` passes no
   `trust_remote_code` anywhere, so a remote-code-only class cannot be used.
3. **`TestGreedyMatchesLlamaCpp`** — token ids from `llama-cli`, for a converted
   GGUF.
4. **`llama-tokenize --ids`** for the tokenizer half.

### Per family

| family | GGUF small enough? | llama.cpp oracle | transformers oracle | verdict |
|---|---|---|---|---|
| DeepSeek-V2-Lite (MLA, softmax router) | **yes, 10.36 GB** | yes (`deepseek2`) | yes, `DeepseekV2ForCausalLM` — but it uses `auto_map`/remote code, so `hfgold.py` may need `trust_remote_code`. **Unverified.** | **fully gateable here** |
| DeepSeek-V3 / R1 / Kimi-K2 | **no.** Smallest real file is `UD-IQ1_S` at 140.2 GB; Q4_K_M is 404.4 GB (R1) / 620.8 GB (K2) against 68 GB free | yes in principle, no file | **yes** — `trl-internal-testing/tiny-DeepseekV3ForCausalLM` (16.9 MB) or a `synth-deepseek-v3` built in-process, native `deepseek_v3` in transformers | **graph fully gateable, model not runnable here** |
| Kimi-Linear 48B-A3B | marginal — 48B total at Q4_K_M is ~29 GB against 68 GB free; it would fit on disk and page from it | yes (`kimi-linear`) | yes with `trust_remote_code` (its config has `auto_map` -> `modeling_kimi.py`), or a tiny-random fixture | gateable, and the only one of these that might actually RUN here |
| Mamba-2 / Jamba / Falcon-H1 | yes — several small members (Mamba2-370M/1.3B, Falcon-H1-0.5B) | yes (`mamba2`, `jamba`, `falcon-h1`) | yes, native in transformers | **fully gateable here** |
| GLM-4.x MoE | **no** for GLM-4.5/4.6; GLM-4-9B (`chatglm`) is a different arch | yes (`glm4moe`) | tiny-random fixtures exist | graph gateable |
| Llama 4 | Scout is 109B total — **no** at Q4, marginal at low bits | yes (`llama4`) | gated repo; needs an HF token | graph gateable with effort |
| qwen2moe (Qwen1.5-MoE-A2.7B) | **yes**, ~10 GB at Q4_K_M | yes (`qwen2moe`) | yes | fully gateable here |

**The tokenizer half is separately cheap for most of these.** `deepseek-v3`,
`glm4`, `minimax-m2`, `hunyuan`, `bailingmoe2`, `exaone4`, `cohere2moe`,
`falcon-h1`, `lfm2`, `qwen35` are already generated into `tok/pretok/table.go`. Three
names that matter here are NOT, and `tok/pretok/table.go`'s own footer says exactly
why:

    kimi-k2 (moonshotai/Kimi-K2-Base): http 404
    llama4 (meta-llama/Llama-4-Scout-17B-16E-Instruct): gated repo: needs HF_TOKEN

RULE 11 covers both: *"a missing artefact is a TASK, not a constraint"*. The
Kimi-K2 404 is a dead repo pointer in llama.cpp's directory, not a missing
capability — `Kimi-K2-Instruct` exists — and `llama4` needs a token. Neither is a
reason to record an absence.

**The disk is a real constraint and belongs in the plan.** 68 GB free, 132
entries already in the model directory, 31 GB of RAM. Adding
DeepSeek-V2-Lite (10.4 GB GGUF plus its ~10 GB container) plus a Mamba-2 and a
qwen2moe fixture is roughly 45 GB of the 68. Something has to be pruned, or the
containers converted and the GGUFs deleted — which `jlm-is-self-sufficient`
already says is the intended lifecycle.

**And a version bump re-converts everything.** `jlm.Version` is 22
(`format/jlm/format.go:168`); 25 containers on disk are already v21 and unreadable by
this build. Capabilities A, B, C and G each require a bump. Batching them into
ONE bump would save a full re-conversion pass over a 132-entry model directory —
that is a scheduling argument for doing A+B+C as one container revision rather
than three.

---

## 7. Where I am uncertain, stated rather than buried

- **KDA's exact form is confirmed; its COST is not.** Three papers agree that it
  is the gated delta rule with scalar forgetting replaced by channel-wise
  forgetting. What is not established is whether a per-channel decay survives
  `GatedDeltaStep`'s current numerics unchanged, and whether the chunkwise
  algorithm the paper's efficiency rests on is needed at jitllm's decode width of
  one token. The "one kernel edit" figure is a LEAD.
- **The transformers oracle may need remote code for two of these.**
  `Kimi-Linear`'s config carries `auto_map` pointing at `modeling_kimi.py`, and
  DeepSeek-V2's at `modeling_deepseek.py`. DeepSeek-V3 is native in transformers
  and needs none. Running downloaded model code inside
  the reference venv is a decision, not a detail.
- **Whether the absorbed MLA form needs a stored derived tensor.** llama.cpp
  ships `attention.key_length_mla` / `attention.value_length_mla` and splits
  `kv_b_proj` into `k_b` and `v_b`, which suggests it does the absorption at
  load rather than at conversion. Which side of the container boundary jitllm
  should do it on is a design fork, not a fact.
- **`qwen2moe` being nearly free.** That is a reading of the tensor shapes, not a
  transcription of the running graph. RULE 7: *"Read the artefact, not the
  description of it"* — transcribe it from `llama-eval-callback` before costing
  it.
- **The cheap-tail names in section 2K.** Each was classified from its llama.cpp
  arch name and general knowledge, not from its graph. Do not quote the list as a
  work estimate.
- **`DeepseekV2ForCausalLM` under transformers 5.17.** V3 is native; V2 may still
  be remote-code-only, which would make `hfgold.py` need `trust_remote_code=True`
  and therefore a decision about running downloaded code in the oracle venv.
- **Whether `attention.value_length` being dropped is latent anywhere today.**
  Every supported architecture has K and V head dims equal, so it should be
  benign — but "should be" is the sentence AGENTS.md warns about, and the check
  is one grep over the GGUFs on disk that nobody has run.
- **Per-quant download share is unverifiable, not merely unverified.** HuggingFace
  publishes repo-level counters only; `/api/models/<repo>/downloads` 404s and the
  tree API carries no per-file counts. Repo-level 30-day figures
  (`unsloth/Kimi-K2-Instruct-GGUF` 44,452, `unsloth/DeepSeek-R1-GGUF` 28,308,
  `bartowski/...V3.1` 5,277) count shard fetches, not users, and a flat-layout
  repo inflates badly. **The block-type census in section 2G does not depend on
  any of this** — it is a parse of the files themselves, which is why it is the
  evidence quoted.
- **Whether any 671B repo has been re-cut under Unsloth "Dynamic 3.0".** The
  Dynamic 2.0 page now 404s and the 3.0 page names only Qwen3.8-27B. The census
  was taken against the files that exist today.
- **Three llama.cpp/vLLM name pairs were not mapped**: llama.cpp's `mistral4`,
  `step35` and `mimo2` against vLLM's `MistralLarge3ForCausalLM`,
  `Step3p5ForCausalLM` and `MiMoV2ForCausalLM`. They look like two names for the
  same models; nobody confirmed it.

---

## 8. Appendix: vLLM's registry, verbatim

Source: `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/
model_executor/models/registry.py`, `main` at the time of the survey.

### `_TEXT_GENERATION_MODELS` — 125 classes

AfmoeForCausalLM · ApertusForCausalLM · ArceeForCausalLM · AXK1ForCausalLM ·
BailingMoeForCausalLM · BailingMoeV2ForCausalLM · BailingMoeV2_5ForCausalLM ·
BailingMoeV3ForCausalLM · BloomForCausalLM · ChatGLMModel ·
ChatGLMForConditionalGeneration · CohereForCausalLM · Cohere2ForCausalLM ·
Cohere2MoeForCausalLM · CwmForCausalLM · DbrxForCausalLM · DeciLMForCausalLM ·
DeepseekForCausalLM · **DeepseekV2ForCausalLM** · **DeepseekV3ForCausalLM** ·
**DeepseekV32ForCausalLM** · **DeepseekV4ForCausalLM** · Ernie4_5ForCausalLM ·
Ernie4_5_MoeForCausalLM · ExaoneForCausalLM · Exaone4ForCausalLM ·
ExaoneMoeForCausalLM · FalconForCausalLM · FalconMambaForCausalLM ·
FalconH1ForCausalLM · GemmaForCausalLM · Gemma2ForCausalLM · Gemma3ForCausalLM ·
Rnj1ForCausalLM · Gemma3nForCausalLM · Gemma4ForCausalLM · Qwen3NextForCausalLM ·
Qwen4ExpForCausalLM · GlmForCausalLM · Glm4ForCausalLM · Glm4MoeForCausalLM ·
Glm4MoeLiteForCausalLM · GlmMoeDsaForCausalLM · Glm5NextForCausalLM ·
GptOssForCausalLM · GPT2LMHeadModel · GPTJForCausalLM · GPTNeoXForCausalLM ·
GraniteForCausalLM · GraniteMoeForCausalLM · GraniteMoeHybridForCausalLM ·
GraniteMoeSharedForCausalLM · GraniteMoeSWAForCausalLM · GraniteSWAForCausalLM ·
HrmTextForCausalLM · HYV3ForCausalLM · HYV4ForCausalLM · HCXVisionV2ForCausalLM ·
HyperCLOVAXForCausalLM · InternLM2ForCausalLM · InternLM3ForCausalLM ·
IQuestCoderForCausalLM · IQuestLoopCoderForCausalLM · Jais2ForCausalLM ·
JambaForCausalLM · K2HorizonForCausalLM · **KimiLinearForCausalLM** ·
Lfm2ForCausalLM · Lfm2MoeForCausalLM · LagunaForCausalLM · LlamaForCausalLM ·
Llama4ForCausalLM · LLaMAForCausalLM · LongcatFlashForCausalLM ·
LongcatFlashNgramForCausalLM · MambaForCausalLM · Mamba2ForCausalLM ·
MellumForCausalLM · MiniCPMForCausalLM · MiniCPM3ForCausalLM ·
MiniMaxM2ForCausalLM · MiniMaxM3SparseForCausalLM · InklingForCausalLM ·
InklingForConditionalGeneration · Ministral3ForCausalLM · MistralForCausalLM ·
MistralLarge3ForCausalLM · MixtralForCausalLM · MiMoForCausalLM ·
MiMoV2FlashForCausalLM · MiMoV2ForCausalLM · NemotronForCausalLM ·
NemotronHForCausalLM · NemotronHPuzzleForCausalLM · MuseGlimmerForCausalLM ·
OlmoHybridForCausalLM · OlmoeForCausalLM · OPTForCausalLM · OrionForCausalLM ·
PanguEmbeddedForCausalLM · PanguProMoEV2ForCausalLM · PanguUltraMoEForCausalLM ·
Param2MoEForCausalLM · PhiForCausalLM · Phi3ForCausalLM · PhiMoEForCausalLM ·
Plamo3ForCausalLM · Qwen2ForCausalLM · Qwen2MoeForCausalLM · Qwen3ForCausalLM ·
Qwen3MoeForCausalLM · Qwen3_5ForCausalLM · Qwen3_5MoeForCausalLM ·
SarvamMoEForCausalLM · SarvamMLAForCausalLM · SeedOssForCausalLM ·
Step1ForCausalLM · Step3TextForCausalLM · Step3p5ForCausalLM ·
StableLmForCausalLM · SolarForCausalLM · TeleChat2ForCausalLM ·
TeleChat3ForCausalLM · TeleFLMForCausalLM · Zamba2ForCausalLM

**Transformers-backend only** (real support, no native implementation):
FlexOlmoForCausalLM · GPTBigCodeForCausalLM · HunYuanDenseV1ForCausalLM ·
HunYuanMoEV1ForCausalLM · NanbeigeForCausalLM · OlmoForCausalLM ·
Olmo2ForCausalLM · Olmo3ForCausalLM · SmolLM3ForCausalLM ·
Starcoder2ForCausalLM · VaultGemmaForCausalLM · Emu3ForConditionalGeneration ·
HunYuanVLForConditionalGeneration · VibeVoiceAsrForConditionalGeneration

### `_MULTIMODAL_MODELS` — 121 classes

AriaForConditionalGeneration · AudioFlamingo3ForConditionalGeneration ·
BagelForConditionalGeneration · BailingMoeV3VLForConditionalGeneration ·
BeeForConditionalGeneration · Blip2ForConditionalGeneration ·
Cohere2VisionForConditionalGeneration · CohereCompassForConditionalGeneration ·
Cosmos3ForConditionalGeneration · Cosmos3EdgeForConditionalGeneration ·
**DeepseekVLV2ForCausalLM** · **DeepseekOCRForCausalLM** ·
**DeepseekOCR2ForCausalLM** · DeepseekV4ForConditionalGeneration ·
DeepseekV41ForCausalLM · Dots3NoteForCausalLM · UnlimitedOCRForCausalLM ·
DotsOCRForCausalLM · Eagle2_5_VLForConditionalGeneration ·
Ernie4_5_VLMoeForConditionalGeneration · Exaone4_5_ForConditionalGeneration ·
FireRedASR2ForConditionalGeneration · FunASRForConditionalGeneration ·
FunAudioChatForConditionalGeneration · Gemma3ForConditionalGeneration ·
Gemma3nForConditionalGeneration · DiffusionGemmaForBlockDiffusion ·
Gemma4ForConditionalGeneration · Gemma4UnifiedForConditionalGeneration ·
GlmAsrForConditionalGeneration · GLM4VForCausalLM ·
Glm4vForConditionalGeneration · Glm4vMoeForConditionalGeneration ·
GlmOcrForConditionalGeneration · Glm5NextForConditionalGeneration ·
GraniteSpeechForConditionalGeneration ·
GraniteSpeechPlusForConditionalGeneration ·
Granite4VisionForConditionalGeneration · H2OVLChatModel · InternVLChatModel ·
InternS1ForConditionalGeneration · InternVLForConditionalGeneration ·
InternS1ProForConditionalGeneration · InternS2MobiusForConditionalGeneration ·
InternS2PreviewForConditionalGeneration · Idefics3ForConditionalGeneration ·
IsaacForConditionalGeneration · KananaVForConditionalGeneration ·
KeyeForConditionalGeneration · KeyeVL1_5ForConditionalGeneration ·
**KimiVLForConditionalGeneration** · **KimiK25ForConditionalGeneration** ·
**KimiK3ForConditionalGeneration** · MoonshotKimiaForCausalLM ·
MossTranscribeDiarizeForConditionalGeneration ·
LightOnOCRForConditionalGeneration · Lfm2VlForConditionalGeneration ·
Llama4ForConditionalGeneration · Llama_Nemotron_Nano_VL ·
LlavaForConditionalGeneration · LlavaNextForConditionalGeneration ·
LlavaNextVideoForConditionalGeneration ·
LlavaOnevisionForConditionalGeneration ·
LlavaOnevision2ForConditionalGeneration · MiDashengLMModel ·
MiMoV2OmniForCausalLM · MiniMaxM3SparseForConditionalGeneration · MiniCPMO ·
MiniCPMV · MiniCPMV4_6ForConditionalGeneration ·
Mistral3ForConditionalGeneration · MolmoForCausalLM ·
Molmo2ForConditionalGeneration · Moondream3ForCausalLM · MossAudioModel ·
HfMoondream · NemotronH_Nano_VL_V2 · NemotronH_Nano_Omni_Reasoning_V3 ·
NemotronH_Super_Omni_Reasoning_V3 · NemotronH_Omni_Reasoning_V3 · NVLM_D ·
MuseGlimmerForConditionalGeneration · OpenCUAForConditionalGeneration ·
OpenPanguVLForConditionalGeneration · OpenVLAForActionPrediction · Ovis ·
Ovis2_5 · Ovis2_6ForCausalLM · Ovis2_6_MoeForCausalLM ·
PaddleOCRVLForConditionalGeneration · PaliGemmaForConditionalGeneration ·
Phi3VForCausalLM · Phi4ForCausalLMV · Phi4MMForCausalLM ·
PixtralForConditionalGeneration · QianfanOCRForConditionalGeneration ·
Qwen2VLForConditionalGeneration · Qwen2_5_VLForConditionalGeneration ·
Qwen2AudioForConditionalGeneration · Qwen2_5OmniModel ·
Qwen2_5OmniForConditionalGeneration · Qwen3OmniMoeForConditionalGeneration ·
Qwen3ASRForConditionalGeneration · Qwen3ASRRealtimeGeneration ·
Qwen3VLForConditionalGeneration · Qwen3VLMoeForConditionalGeneration ·
Qwen3_5ForConditionalGeneration · Qwen3_5MoeForConditionalGeneration ·
Qwen4ExpForConditionalGeneration · RForConditionalGeneration ·
SkyworkR1VChatModel · SmolVLMForConditionalGeneration ·
StepVLForConditionalGeneration · Step3VLForConditionalGeneration ·
Step3p7ForConditionalGeneration · UltravoxModel ·
VoxtralForConditionalGeneration · VoxtralRealtimeGeneration ·
CohereAsrForConditionalGeneration · NemotronParseForConditionalGeneration ·
WhisperForConditionalGeneration

(The last three are the only encoder-decoder models left; T5, BART, Donut and
Florence-2 were deleted with the V0 engine and BART/Florence-2 now live in an
out-of-tree plugin.)

### The families this document is about

| family | vLLM class | note |
|---|---|---|
| DeepSeek V2 | `DeepseekV2ForCausalLM` | |
| DeepSeek V3 / R1 / V3.1 | `DeepseekV3ForCausalLM` | R1 is architecturally identical to V3; V3.1 differs only in FP8 `scale_fmt` |
| DeepSeek V3.2 | `DeepseekV32ForCausalLM` | adds DSA: `index_n_heads 64`, `index_head_dim 128`, `index_topk 2048` |
| DeepSeek V4 / V4.1 | `DeepseekV4ForCausalLM`, `DeepseekV41ForCausalLM` | |
| **Kimi-K2** | **none — it IS `DeepseekV3ForCausalLM`** | its `config.json` says `"architectures": ["DeepseekV3ForCausalLM"], "model_type": "kimi_k2"` |
| Kimi-K2.5 / K3 | `KimiK25ForConditionalGeneration`, `KimiK3ForConditionalGeneration` | multimodal |
| Kimi-Linear | `KimiLinearForCausalLM` | |
| Kimi-VL | `KimiVLForConditionalGeneration` | |
| MiniMax M1 | — | **REMOVED at 0.23.0** |
| MiniMax M2 / M3 | `MiniMaxM2ForCausalLM`, `MiniMaxM3SparseForCausalLM` | |
| GLM-4.5/4.6/4.7 | `Glm4MoeForCausalLM` | plus `Glm4MoeLiteForCausalLM`, `GlmMoeDsaForCausalLM`, `Glm5NextForCausalLM` |
| Llama 4 | `Llama4ForCausalLM` / `Llama4ForConditionalGeneration` | Llama-3.2 Vision (`Mllama`) was removed at 0.10.2 |
| Qwen3-Next / Qwen3-VL / Qwen 3.5 / Qwen4-exp | all present, with MTP heads | |
| Gemma 3n / Gemma 4 | `Gemma3nForCausalLM`, `Gemma4ForCausalLM` | |
| Mamba2 / Jamba / Falcon-H1 / Nemotron-H / granitehybrid / Zamba2 | all present | `Plamo2ForCausalLM` **removed at 0.26.0**; `Plamo3ForCausalLM` present |
| Hunyuan | `HYV3ForCausalLM`, `HYV4ForCausalLM` native; `HunYuanMoEV1`/`Dense`/`VL` **Transformers-backend only** | |
| dots1 | — | **REMOVED at 0.23.0** |
| Ernie 4.5, bailing/Ling, Seed-OSS, ExaOne4, Cohere2, LFM2 | all present | |
| SmolLM3 | `SmolLM3ForCausalLM` | Transformers backend only |
| RWKV / ARWKV, GroveMoE | — | **never supported, zero hits** |

### Kimi-K2's config, for the record

`moonshotai/Kimi-K2-Instruct/config.json`:

    "architectures": ["DeepseekV3ForCausalLM"],  "model_type": "kimi_k2",
    "auto_map": {"AutoConfig": "configuration_deepseek.DeepseekV3Config", ...}

Identical to DeepSeek-V3 in `num_hidden_layers` 61, `hidden_size` 7168,
`moe_intermediate_size` 2048, `kv_lora_rank` 512, `q_lora_rank` 1536,
`qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128,
`scoring_func` sigmoid, `topk_method` noaux_tc, `num_experts_per_tok` 8,
`n_shared_experts` 1. Differs in `n_routed_experts` **384** (vs 256),
`num_attention_heads` **64** (vs 128), `n_group`/`topk_group` **1/1** (vs 8/4),
`first_k_dense_replace` **1** (vs 3), `rope_theta` **50000**,
`routed_scaling_factor` **2.827**, `vocab_size` **163840**,
`num_nextn_predict_layers` **0**.

And llama.cpp's own converter registers `DeepseekV2Model`
(`model_arch = MODEL_ARCH.DEEPSEEK2`) for `DeepseekV2ForCausalLM`,
`DeepseekV3ForCausalLM`, `DeepseekOCRForCausalLM`, `UnlimitedOCRForCausalLM`,
`KimiVLForConditionalGeneration`, `KimiK25ForConditionalGeneration`,
`YoutuForCausalLM` and `YoutuVLForConditionalGeneration` — with an explicit
`if tokpre == "kimi-k2":` branch that rebuilds merges from tiktoken's
`_mergeable_ranks`. **So a Kimi-K2 GGUF carries
`general.architecture = deepseek2`, and the only K2-specific work is the
tokenizer** — which is precisely the one `tok/pretok/table.go` records as a 404.

### Sources

- vLLM registry: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py
- vLLM GGUF migration RFC: https://github.com/vllm-project/vllm/issues/39583
- llama.cpp arch enum: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.h
- DeepSeek-V3 config: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
- DeepSeek-V2-Lite config: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json
- Kimi-K2 config: https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json
- Kimi-Linear config: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json
- Kimi Linear paper: https://arxiv.org/abs/2510.26692
- KDA as a gated-delta-rule variant: https://arxiv.org/pdf/2607.07953, https://arxiv.org/abs/2605.22791
- DeepSeek-V2-Lite Q4_K_M GGUF (10,364,416,672 B): https://huggingface.co/Jianping746/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF
- tiny DeepSeek-V3 fixture (16.9 MB): https://huggingface.co/trl-internal-testing/tiny-DeepseekV3ForCausalLM
- Unsloth quant naming: https://unsloth.ai/docs/models/tutorials/deepseek-v3-0324-how-to-run-locally.md

---

## 9. The flagship survey: Gemma 4, Gemma 3n, DeepSeek V3.2 and V4, Kimi-K3, MiniMax-M2/M3

Read off llama.cpp's builders (`src/models/<arch>.cpp` at 85ca3b5), its
converter (`conversion/<family>.py`), transformers' forwards (5.17.0:
`models/{gemma4,gemma3n,deepseek_v32,deepseek_v4,minimax_m2,minimax_m3_vl}`)
and the published config.json of each model, before anything was written. Kimi-K3
has no transformers class (remote code only); its column is llama.cpp's
`kimi-k3.cpp` and `conversion/kimi_k3.py` alone. The question per model is the
one RULE 7 asks: which pieces the engine already runs (by the symbol that runs
them), which are new, and what the new ones cost -- a kernel on four targets, a
graph feature on both tiers, or a change to a shape the whole engine assumes is
uniform.

### The table

| order | model (llama.cpp arch) | what is already here | what is new | cost | real checkpoint |
|---|---|---|---|---|---|
| **1** | **MiniMax-M2/M2.1/M2.5** (`minimax-m2`) | whole-projection q/k RMSNorm (`FlagQKNormWide`, OLMoE/OLMo 2), NEOX partial rotary (`NRot`), sigmoid router + selection bias + renormalise (DeepSeek-V3/glm4moe), no shared expert | nothing | **S**: converter + fixture | 230B-A10B; Q3_K_M 109 GB, fits the V100 box's RAM, not the laptop |
| **2** | **Gemma 4** (`gemma4`): E2B, E4B, 26B-A4B, 31B | gemma3's block (post-norms, sandwich norms, per-head q/k norm, 5:1 sliding window, final softcap, sqrt(d) embedding), GELU-tanh, rope freq factors (`RoleRopeFreqs`), softmax top-k | **per-layer attention geometry** (head_dim 256 sliding / 512 global; kv heads 16 / 4 on the 31B) -- every tier assumes one HeadDim; weightless V RMSNorm; K = V on global layers (`attention_k_eq_v`); attention scale 1; "proportional" rope on global layers (NEOX over the whole head, frequencies base^(-2i/head_dim) for the first quarter and zero after); `layer_scalar`; the 26B's FFN = dense MLP + a mixture in parallel, each with its own pre/post norms, a router on an unweighted RMSNorm x scale x d^-1/2 and a per-expert output scale; E2B/E4B add **per-layer embeddings** (a second, 35x256-wide embedding table and a per-block gate/projection) and **KV sharing** (the last 20 of 35 blocks read an earlier block's cache), and a double-width FFN on the shared blocks | **L**: one engine-wide shape change (per-layer head geometry: Config, KV pages, host attention kernels keyed by shape, device plan) plus five graph features on both tiers; then **M** more for PLE + KV sharing | E2B 3.1 GB, E4B 5 GB, 26B-A4B 16 GB, 31B 18 GB at Q4_K_M: all on the laptop's disk; the two small ones in its RAM |
| 3 | **DeepSeek V3.2** (`deepseek32`) | all of deepseek2: MLA, V3 router, grouped top-k, dense lead, MTP set aside, YaRN log multiplier | the **lightning indexer (DSA)**: a LayerNorm-with-bias k, partial rotary on q/k of the indexer, a Hadamard rotation, score = sum_h w_h relu(q_h . k), a top-2048 over every cached position, and attention restricted to it; a second per-layer key cache (the indexer's), which must page and relocate with the block | **M**: three kernels on four targets (indexer score, top-k over positions, a masked/gathered attention), one more paged stream. Below 2048 positions the top-k keeps every key and the graph is exactly deepseek2's -- which is also why a short fixture cannot see the indexer and the gate has to be longer than top_k | 685B-A37B; Q2_K-class only on the V100 box (needs Q2_K, which `Dequantable` lacks) |
| 4 | **MiniMax-M3** (`minimax-m3`) | MiniMax-M2's attention (per-head q/k norm this time), DeepSeek-V3's dense lead + shared experts + routed scale, `FlagSwiGLUOAI` (gpt-oss), Gemma-style (1+w) norms baked by the converter | **MiniMax Sparse Attention**: a per-GQA-group indexer (q/k projections + norms) scoring 128-token blocks, top-k blocks plus a local window, block-sparse attention; llama.cpp itself falls back to dense attention without flash attention | **M-L**: the block indexer and block-sparse attention on four targets, its key cache paged | 428B-A23B; V100 box only |
| 5 | **Gemma 3n** (`gemma3n`, text) | gemma3's attention, per-layer embeddings and KV sharing (once Gemma 4 has them) | **AltUp** (4 parallel residual streams with a learned predict/correct mixing per block and a router), **LAuReL** (a low-rank residual branch), **gaussian top-k activation sparsity** on the first 10 FFNs (mean + 1.645 std threshold) | **L**: three graph features with no counterpart, each on both tiers; the streams quadruple the residual state the step and the relocation carry | E2B/E4B, small; superseded by Gemma 4 E2B/E4B |
| 6 | **DeepSeek V4** (`deepseek4`) | MLA-like low-rank q, sinks (gpt-oss), MTP | **manifold hyper-connections** (4 residual streams mixed by Sinkhorn-normalised matrices per block), **compressed attention** (per-layer compression ratios 4/128 with a gated compressor and its own rope base), a grouped low-rank output projection, **sqrt-softplus** gating, hash-routed layers, a SwiGLU clamp, an indexer | **XL**: a new residual structure and a new attention family; the builder is 1500 lines | V100 box only, Q2-class |
| 7 | **Kimi-K3** (`kimi-k3`) | kimi-linear's KDA + MLA hybrid, V3 router | cross-layer **residual attention** over block checkpoints, **latent MoE** (experts at a narrower width behind down/up projections), the **situ** activation, a sigmoid MLA output gate, a full-rank KDA gate | **XL**, and no transformers class to gate against: llama.cpp is the only reference | 2.8T-A50B: no box here holds even the smallest quant on disk with room to spare |

### Why that order

Value is "how many people run it, on hardware this engine targets"; cost is the
new machinery, priced above. MiniMax-M2 goes first because it costs a fixture --
the engine already ran all three of its pieces -- and it is a frontier
agentic/coding model. Gemma 4 is the highest-value entry on the list (four
sizes, two of which run on a laptop, Apache-2.0) and the costliest of the
buildable ones, because its global layers are a different SHAPE: every tier,
the KV pager and the device plan assume one head width per model, and Gemma 4
is the first architecture here where that is false. DeepSeek V3.2 is next:
one capability (the indexer) over a graph that is already verified, and its
real checkpoint is the most-run open 600B-class model; it needs Q2_K for any
quant the V100 box can hold with room. MiniMax-M3 shares M2's attention and
DeepSeek's mixture, so its cost is MSA alone. Gemma 3n's three features have no
counterpart anywhere else and its niche is taken by Gemma 4's E-models. DeepSeek
V4 and Kimi-K3 are new residual structures (hyper-connections, residual
attention) on models no machine here can hold.

### Where the references part, recorded before building

- **Gemma 4's rotary.** transformers' "proportional" rope computes
  `inv_freq = base^(-2i/head_dim)` for `i < partial*head_dim/2` and **zero**
  after, then pairs NEOX over the whole head; llama.cpp gets the same function
  from `rope_freqs` (freq factors, infinite past the rotated quarter). It is NOT
  partial rotary in this engine's sense (`NRot`), whose exponent is
  `2i/NRot`: a 512-wide global head rotating 128 dimensions at `base^(-2i/128)`
  would be fluent and wrong.
- **Gemma 4's V norm** has no weight (`with_scale=False`), and on the global
  layers of the 31B and 26B there is no `v_proj` at all: V is K's projection,
  normed without K's weight and not rotated.
- **Gemma 4's attention scale is 1**, not 1/sqrt(head_dim): the q and k norms
  carry the scale (`self.scaling = 1.0`; llama.cpp `f_attention_scale = 1`).
- **MiniMax-M2's q/k norm** is over the whole projection
  (`MiniMaxM2RMSNorm(num_heads*head_dim)`), where MiniMax-M3's is per head
  (`{n_embd_head_k}` in llama.cpp's tensor table): same family, the two
  readings of the same tensor names.
- **DeepSeek V3.2's indexer cache** is FP8 in the reference and F32 in
  llama.cpp's converter (`tensor_force_quant`); the top-k is exact over scores,
  so the indexer's precision decides which keys a long context keeps.

### What was built

Section 9 is kept current with what landed; the evidence for each is in
`docs/engineering-history/model-correctness.md`.

- **minimax-m2: built.** `jlm.ArchMiniMaxM2` (44), `convert/flagship.go`,
  `synth-minimaxm2`; host, CUDA, Vulkan and Metal at 6.5e-13..7.9e-13 against
  transformers and the host, every principle gate. The real MiniMax-M2
  Q3_K_M (230B, on the V100 box's host tier) agrees with llama.cpp 20 of 20
  teacher-forced and for 20 free-running tokens.
- **gemma4 31B and 26B-A4B: built.** `jlm.ArchGemma4` (45), per-layer
  attention geometry on every tier (host attention kernel sets, per-layer KV
  pages, a device scratch set per geometry with runs cut where it changes),
  the proportional rotary, the weightless v norm, v from k, layer_scalar, the
  SentencePiece-style BPE tokenizer, and the 26B's dense-and-mixture block
  (`jlm.Flag2DenseMoE`, a per-expert weight factor in the router kernel on
  every tier); `synth-gemma4`, `synth-gemma4-kv` and `synth-gemma4-moe` at
  4.1e-12..7.3e-11 against transformers and the host on host, CUDA, Vulkan
  and Metal, every principle gate. The real 31B and 26B-A4B Q4_K_M agree with
  llama.cpp at every teacher-forced position (20 of 20, 12 of 12). E2B/E4B
  run too: per-layer embeddings, KV sharing and the double-wide FFN on both
  tiers, `synth-gemma4-e` at 5.4e-12 against transformers and 3.3e-12..9.5e-12
  on CUDA, Vulkan and Metal; the real E2B and E4B agree with llama.cpp at
  every teacher-forced position. `synth-gemma4-wide` runs the real head widths
  (256 and 512) on every tier. On the device the real checkpoints sit in a band
  the model sets (llama.cpp's own CUDA-vs-CPU reads 2.8-3.0 on them); the 31B's
  last block alone reads 0.64 against other models' 0.25, unresolved
  (`docs/engineering-history/model-correctness.md`).
- **deepseek32: built.** `jlm.ArchDeepseek32` (46), the indexer's three
  matrices and its key norm as roles 312-316, the indexer's key as the tail of
  the MLA cache row (so the "second per-layer key cache" is the same pages,
  and moves with them), host masking through a generated top-k and a ReLU
  kernel, and on the device four basic kernels (score, select, gather,
  descriptors) in front of the unchanged MLA paged attention.
  `synth-deepseek32` at 6.5e-12 against transformers and 2.2e-12..5.0e-12
  device-vs-host. The Hadamard rotation is not built and need not be: it is
  orthogonal and applied to both q and k, so it preserves every score
  (transformers omits it for the same reason); the reference's FP8 indexer
  cache is F32 here. No real checkpoint: at Q4_K_M the 685B is 405 GB (the publishers' table
  in section 2), past every box here, and the smaller GGUFs are Q2_K-class,
  which `Dequantable` lacks.
- **gemma3n: built.** `jlm.ArchGemma3n` (47), roles 317-328 and a sixth
  config tail (AltUp, NSparse, SparseStd). The four streams are carried
  STREAM-MAJOR in one residual (stream k of row r at (k*rows + r)*n_embd), so
  every n_embd kernel of gemma3's block reads stream 0 untouched and a device
  is handed the whole residual as one slice (`Config.ResidW`,
  `LayerPlan.ResidW`); the prediction, LAuReL, the gaussian top-k and the
  correction are host kernels (two new row statistics, `cpu.EmitGaussTopK`
  and `cpu.EmitMagMatch`, and `ActIdentity`) and seven basic IR kernels
  (`kernels/altup.go`). The head reads the streams' mean, which the host
  takes, so an AltUp head is never folded into a device submission.
  `synth-gemma3n` at 2.8e-11 against transformers and 1.2e-11..3.3e-11
  device-vs-host; the real Gemma 3n E2B Q4_K_M agrees with llama.cpp at 18 of
  20 teacher-forced positions, the other two ties.
- **minimax-m3: built.** `jlm.ArchMiniMaxM3` (48), the indexer's query and its
  norm as roles 329-330 (the key reuses deepseek32's `RoleIdxK` and
  `RoleIdxKNorm`), block size and forced blocks as a seventh config tail. The
  indexer's key is one more kv head of the cached row (`Config.KVRowAt`,
  `LayerPlan.KVRow`), so it pages and relocates with K and V; the selecting
  blocks run in a device scratch set of their own (`geoMSA`). The host ranks
  blocks by walking the positions best-first with the sampler's generated
  ordering (a block's first position seen is its maximum) and masks each kv
  group's scores; the device runs seven basic kernels (scores, block max,
  select, compact, gather K, gather V, descriptors) in front of the unchanged
  paged attention. `synth-minimaxm3` at 4.5e-12 against transformers and
  3.6e-12/3.0e-13 device-vs-host on CUDA. The published model is 428B (a
  Q3_K_M is 197 GB in five parts, which only the V100 box holds); its
  comparison against llama.cpp (which runs MSA only with flash attention on)
  is not yet taken.
- **deepseek4: built.** `jlm.ArchDeepseek4` (49), roles 331-349 (the two
  hyper-connections' and the head's fn/base/scale, the grouped output's first
  half, the compressor's and the indexer compressor's kv/gate/position
  bias/norm, the hash blocks' token table) and an eighth config tail (the
  streams, Sinkhorn rounds and eps, the output groups, both rates, the hash
  block count and a compression kind per block). The four streams are one
  stream-major residual as gemma3n's; the mixer and the compressor's pool are
  generated on AVX2, SSE and arm64 (`cpu.EmitHCMix`, `cpu.EmitColPool`) and the
  device runs twelve basic kernels (`kernels/ds4.go`) in front of the
  unchanged paged attention. A position's cached row is its key and the
  compressor's inputs, so the window holds what the compressor still folds;
  the entries are a second paged history per block (`kvCache.ent`, the
  device's `entKey` pool) that relocation, the step and the prefix cache carry,
  and that decide a prefix restore. `synth-deepseek4` at 3.2e-10 against
  transformers and 1.5e-10/8.0e-12 device-vs-host on CUDA. No published V4 is
  held by any box here, so no comparison against llama.cpp is taken.
- **kimi-k3: built (GGUF).** `jlm.ArchKimiK3` (50), roles 350-355 (the
  residual scores of each sublayer and the head's, routed_down/up and the
  routed norm) and a ninth config tail (the checkpoint interval, the latent
  width, the situ bounds, the KDA decay bound, the MLA latent norms' epsilon).
  The checkpoints are streams of one stream-major residual as DeepSeek V4's;
  situ and the bounded decay are generated on AVX2, SSE and arm64, and the
  device runs three basic kernels (`kernels/k3.go`) for the mixes beside the
  generic block with its mixture moved to the latent. `synth-kimik3`, from
  Moonshot's own KimiLinearForCausalLM, at 3.0e-10 against it and
  9.8e-11/4.5e-12 device-vs-host on CUDA. The safetensors path is not built
  (every published K3 ships compressed-tensors MXFP4 experts), and no real
  checkpoint is held here (2.8T; the pruned75 derivative is 442.5 GiB).

## 10. Decision models: added on the user's decision

The user decided to add the decision-model family (models that answer typed
questions in one forward pass and generate nothing: Laya, Clef-Flash, d1, Lev,
and the open models that speak TypeSafe's `/v1/systemone`). The research, the
four readout mechanisms and the build order are in
`docs/design/decision-models.md`. Most of the family is a readout over a
backbone jitllm already runs (qwen35, lfm2); Laya brings a new encoder
architecture (ModernBERT) and Clef a joint head.
