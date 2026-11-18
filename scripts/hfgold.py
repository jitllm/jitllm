#!/usr/bin/env python3
"""Dump teacher-forced logits from HuggingFace transformers, once.
-> testdata/golden/hf/<dir>.json

  $JITLLM_HF_PY scripts/hfgold.py [dir ...]

THE ORACLE FOR THE SAFETENSORS CONVERTER. convert/hfvsgguf_test.go can see a
wrong name mapping only where a GGUF of the same weights exists, which is one
model. The tiny-random fixtures have no GGUF; what they have is the reference
implementation itself, and random weights make every wiring error an
uncorrelated logit vector rather than a near-miss.

The venv ($JITLLM_HF_PY is its python) is CPU torch:
  uv venv /path/to/hfref-venv
  uv pip install --python ... torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python ... transformers safetensors numpy sentencepiece protobuf

Fixed ids, not a prompt, so the gate compares the GRAPH and the tokenizer cannot
be the variable. The tokenizer half is separate and reads tokenizer.json through
the `tokenizers` library, NOT through AutoTokenizer -- because the converter's
authority is that artefact (RULE 7m), and transformers 5's LlamaTokenizer class
DISCARDS it: it rebuilds a non-legacy Metaspace(prepend_scheme=first) whatever
the file says. On tiny-random-Llama's " leading space" that drops the leading
"\u2581" piece (29871) which the file's Prepend+Replace, sentencepiece itself
and llama.cpp all emit. A known divergence from one class, recorded, not matched.

And where the directory ships tokenizer.model, sentencepiece itself is recorded
too ("sp_ids"): it is what the weights were TRAINED against, and tokenizer.json
is HuggingFace's conversion of it, which can depart. Mixtral's does -- its
Metaspace(prepend_scheme=first) skips the dummy prefix when the text already
starts with a space, where sentencepiece adds it unconditionally. The gate
matches sentencepiece where it exists and names every place the file departs. Everything runs in float32 whatever the checkpoint's dtype: the
converter widens bf16 exactly, so both arms see the same weights.
"""
import json, os, sys, math
import torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("JITLLM_MODELS", os.path.join(HERE, "..", "models"))
OUT = os.path.join(HERE, "..", "testdata", "golden", "hf")
DIRS = [
    "qwen3", "qwen2.5-tiny-random", "tiny-random-LlamaForCausalLM",
    "tiny-qwen3moe-hf", "tiny-random-OlmoeForCausalLM",
    "tiny-random-GemmaForCausalLM", "tiny-random-phi3", "tiny-random-Phi3ForCausalLM",
    # The mixture families' fixtures as scripts/moegold.py saves them with each
    # family's own class (links to its $CV_HF/<name>), so the safetensors path is
    # gated on the same weights as the GGUF one.
    "synth-glm4moe-hf", "synth-qwen2moe-hf", "synth-ernie45moe-hf", "synth-ernie45-hf",
    "synth-hunyuanmoe-hf", "synth-hunyuan-hf",
    # The modern text group (scripts/moegold.py's fixtures, the same way).
    "synth-dots1-hf", "synth-phimoe-hf", "synth-apertus-hf",
    # YaRN's stated and derived magnitudes, as scripts/c6gold.py saves them.
    "synth-olmo3-af-hf", "synth-mistral3-af-hf", "synth-deepseek-yarn-hf", "synth-deepseek-yarn-af-hf",
    # A class with a tower, read by a model.json (convert/models).
    "synth-hf-qwen35",
]
NPOS, NLOGIT = 8, 256
# Texts for the tokenizer half: a leading space, a double space (sentencepiece
# has "\u2581\u2581" pieces a splitting pre-tokenizer can never form), a
# newline and characters that reach byte fallback.
TEXTS = [
    "The capital of France is",
    " leading space and  double  spaces",
    "line one\nline two\ttab",
    "caf\u00e9 na\u00efve \u00fcber 12345 \U0001F600 \u4e2d\u6587",
]


# The 32000-token SentencePiece donor's own unk/bos/eos: the classes' defaults
# (SmolLM3's pad 128004, OLMo 2's eos 50279) lie outside its vocabulary, and
# nn.Embedding refuses a padding_idx there.
SPM_IDS = dict(pad_token_id=0, bos_token_id=1, eos_token_id=2)


# SYNTHETIC FIXTURES, made by the reference's own classes. The downloaded
# tiny-random checkpoints are real files and stay, but two of their geometries
# are blind: head_dim 2 makes NEOX and adjacent-pair rotary the SAME single pair
# (gemma's NORM-rope violation passed), and all-one norm weights make every norm
# wiring invisible. These have head_dim 16, GQA, random norms and weights large
# enough that the FFN moves the residual, and they borrow a donor's tokenizer.
#   name: (model_type, config overrides, tokenizer donor)
SYNTH = {
    "synth-gemma": ("gemma", dict(hidden_act="gelu_pytorch_tanh"), "tiny-random-GemmaForCausalLM"),
    "synth-mixtral": ("mixtral", dict(num_local_experts=4, num_experts_per_tok=2, hidden_act="silu"),
                      "tiny-random-MixtralForCausalLM"),
    "synth-qwen2": ("qwen2", dict(), "qwen2.5-tiny-random"),
    # Qwen3 under YaRN, a factor of 4 over an original context of 16, so the
    # correction range and the magnitude bite inside the golden positions.
    "synth-hf-qwen3-yarn": ("qwen3", dict(rope_parameters={"rope_type": "yarn", "factor": 4.0,
                                                          "original_max_position_embeddings": 16,
                                                          "rope_theta": 10000.0}),
                            "qwen2.5-tiny-random"),
    # Mixtral with a window of 4: MixtralModel masks every layer to it.
    "synth-hf-mixtral-swa": ("mixtral", dict(num_local_experts=4, num_experts_per_tok=2, hidden_act="silu",
                                             sliding_window=4), "tiny-random-MixtralForCausalLM"),
    # A window of 4 so it bites inside the 8 golden positions: Phi3 masks every
    # layer to it, and the tiny-random fixtures' 2047 never reach it.
    "synth-phi3": ("phi3", dict(sliding_window=4), "tiny-random-Phi3ForCausalLM"),
    # Phi3 under LongRoPE over an original context of 16, every factor its
    # own: a golden of 40 positions runs past it, so transformers turns the
    # long factors and the magnitude of a fourfold context, and an instance of
    # 41 positions turns them too (model.RopeFor).
    "synth-phi3-longrope": ("phi3", dict(
        original_max_position_embeddings=16, max_position_embeddings=64,
        rope_parameters={"rope_type": "longrope", "rope_theta": 10000.0, "original_max_position_embeddings": 16,
                         "short_factor": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75],
                         "long_factor": [3.0, 4.5, 2.5, 6.0, 3.5, 5.0, 7.5, 4.0]}),
                            "tiny-random-Phi3ForCausalLM"),
    # MLA and the V3 router, in megabytes. Every knob here is switched ON
    # because the point of a synthetic fixture is that a wiring error shows as
    # an uncorrelated logit vector: q_lora_rank exercises the two-step query,
    # first_k_dense_replace=1 puts a DENSE block in front of two mixture ones,
    # n_group=2/topk_group=1 makes the grouped selection actually discard a
    # group, and sigmoid + routed_scaling_factor + norm_topk_prob are the three
    # router knobs that are independent of each other.
    #
    # The head dims are deliberately UNEQUAL and none is a multiple of another:
    # qk_nope 16 + qk_rope 8 = 24 against v_head_dim 16 and kv_lora_rank 16.
    # Equal ones would hide exactly the confusion this architecture invites --
    # a key head that is not the value head, and neither of them the latent.
    "synth-deepseek": ("deepseek_v3", dict(
        q_lora_rank=32, kv_lora_rank=16,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        # ★ head_dim MUST BE REPEATED HERE AND IT IS NOT REDUNDANT.
        # DeepseekV3Config derives self.head_dim = qk_rope_head_dim, but the
        # base kwargs above pass head_dim=16 for every fixture and an explicit
        # kwarg wins -- which builds the rotary over 16 dimensions instead of
        # the 8 that are rotated, and the reference then fails in its own
        # apply_rotary_pos_emb_interleave rather than producing a wrong golden.
        # Loud, luckily. Keep it equal to qk_rope_head_dim.
        head_dim=8,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, first_k_dense_replace=1,
        n_group=2, topk_group=1, routed_scaling_factor=2.5,
        norm_topk_prob=True, scoring_func="sigmoid",
        num_hidden_layers=3, hidden_act="silu",
        # ★ AND MLA HAS NO GQA, so this must equal num_attention_heads. The
        # base kwargs use 2 for every other fixture; here expand_kv has
        # ALREADY produced one key per attention head, and a
        # num_key_value_groups of 2 then repeats them to 8 against a 4-head
        # query. It is not a tuning choice -- the latent is shared by every
        # head, which is what the "MQA" in kv_a_proj_with_mqa means.
        num_key_value_heads=4,
    ), "qwen2.5-tiny-random"),
    # ★ THE SAME MODEL WITH EVERY BLOCK DENSE, WHICH SEPARATES MLA FROM THE
    # ROUTER. first_k_dense_replace == num_hidden_layers means no block ever
    # reaches the mixture, so a mismatch here is the attention and a mismatch
    # only in synth-deepseek is the router. Two fixtures that differ in ONE
    # field are worth more than one that tests both at once -- the first run of
    # synth-deepseek read NMSE 2.5e-01 and could not say which half owned it.
    #
    # It also exercises the dense-lead path at its limit: NDenseLead == NLayer
    # is the boundary where an off-by-one puts a router on the last block.
    "synth-deepseek-dense": ("deepseek_v3", dict(
        q_lora_rank=32, kv_lora_rank=16,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, head_dim=8,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, first_k_dense_replace=3,
        n_group=2, topk_group=1, routed_scaling_factor=2.5,
        norm_topk_prob=True, scoring_func="sigmoid",
        num_hidden_layers=3, hidden_act="silu", num_key_value_heads=4,
    ), "qwen2.5-tiny-random"),
    # ★ DeepSeek-V2-Lite's SHAPE, WHICH IS NOT V3's ON FIVE AXES AT ONCE, and
    # the only fixture that reaches the ONE-STEP QUERY. q_lora_rank null means
    # there is no q_a_proj and no q_a_layernorm at all -- the query projects
    # straight to NHead*qk_head_dim -- and both fixtures above set it to 32, so
    # that branch of model.mlaProject had no gate until this existed.
    #
    # It also differs in every router axis: softmax gating rather than sigmoid,
    # NO correction bias, no grouping, no renormalisation and a scale of 1. So
    # between the three the router kernel is exercised with its bias plane, its
    # group phase and its scale multiply each present AND absent, which is what
    # the emitter keys on.
    #
    # Two shared experts rather than one, because NFFNShExp is
    # n_shared_experts * moe_intermediate_size and a fixture with one cannot
    # tell that product from either factor.
    # ★ deepseek_v2 AND NOT deepseek_v3 WITH scoring_func="softmax", BECAUSE
    # THE V3 CLASS IGNORES THAT FIELD. DeepseekV3TopkRouter.forward hardcodes
    # `router_logits.sigmoid()` and never reads config.scoring_func, so a V3
    # fixture asking for softmax records a SIGMOID golden -- and the converter,
    # which does read the field, then disagrees with its own oracle. The first
    # version of this entry did exactly that and read NMSE 8.9e-02 against a
    # correct engine. The V2 class implements `router_logits.softmax(...)` for
    # real, and V2's config carries no scoring_func at all: softmax is what
    # that architecture IS. RULE 7m from the other side -- a reference that
    # silently ignores a key is not an oracle for that key.
    "synth-deepseek-lite": ("deepseek_v2", dict(
        q_lora_rank=None, kv_lora_rank=16,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, head_dim=8,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=2,
        moe_intermediate_size=32, first_k_dense_replace=1,
        norm_topk_prob=False, routed_scaling_factor=1.0,
        num_hidden_layers=3, hidden_act="silu", num_key_value_heads=4,
    ), "qwen2.5-tiny-random"),
    # ★ GLM-4.7-Flash's SHAPE, AND THE POINT IS THAT IT IS DeepSeek's GRAPH
    # UNDER A DIFFERENT CLASS NAME. Glm4MoeLiteForCausalLM declares
    # kv_lora_rank 512 and q_lora_rank 768, and its published safetensors index
    # carries kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, q_a_proj,
    # q_a_layernorm, q_b_proj, mlp.gate.e_score_correction_bias and
    # mlp.shared_experts.* -- the SAME names deepseek2HF already maps, tensor
    # for tensor.
    #
    # ★ SO THE ENTRY IS A CLAIM THAT NEEDS AN ORACLE, WHICH IS WHY IT IS HERE
    # RATHER THAN JUST IN hfArchTable. "The names match" is not "the graph
    # matches": a reference is free to norm in a different place, rotate a
    # different half or scale the router differently under the same weights.
    # RULE 7 asks for a fixture and a golden, and this is what makes adding the
    # architecture a measurement instead of an assertion.
    #
    # Its sibling Glm4MoeForCausalLM (GLM-4.5-Air) is NOT this graph -- it has
    # no kv_lora_rank at all -- and is deliberately absent.
    "synth-glm4moelite": ("glm4_moe_lite", dict(
        q_lora_rank=32, kv_lora_rank=16,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, head_dim=8,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, first_k_dense_replace=1,
        n_group=2, topk_group=1, routed_scaling_factor=2.5,
        norm_topk_prob=True, num_hidden_layers=3, hidden_act="silu",
        num_key_value_heads=4,
    ), "qwen2.5-tiny-random"),
    # ★ KIMI-LINEAR: MLA ON THE FULL-ATTENTION BLOCKS AND KIMI DELTA ATTENTION
    # ON THE REST, WHICH IS qwen3next's GATED DELTA RULE WITH A PER-CHANNEL
    # DECAY. Its own class docstring is the specification: "essentialy the same
    # a gated delta net (GDN) but decay is per-channel instead of per-token".
    #
    # ★ THE LAYER TYPES ARE NAMED RATHER THAN A PREFIX, and the fixture makes
    # them INTERLEAVED on purpose: [kda, full, kda] puts a full-attention block
    # BETWEEN two recurrent ones, so a reader that treats the kinds as a prefix
    # (NDenseLead's shape) cannot pass. That is the one thing jlm.LayerKinds
    # exists for and the fixture should exercise it rather than the easy case.
    #
    # linear_num_heads == num_attention_heads and linear_head_dim == head_dim
    # here so the two halves share a geometry; the real model has 32 of each.
    "synth-kimilinear": ("kimi_linear", dict(
        kv_lora_rank=16, q_lora_rank=None,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, head_dim=8,
        num_attention_heads=4, num_key_value_heads=4,
        # ★ linear_head_dim IS A MULTIPLE OF 32 ON PURPOSE. It is the rank of
        # KDA's two low-rank gates, and the device chains two matvecs through
        # a quantized activation of that width -- kernels.Quantize works in
        # 32-element blocks, so a smaller rank is declined by name and the
        # device half of this architecture would go untested by this fixture.
        linear_num_heads=4, linear_head_dim=32, linear_conv_kernel_dim=4,
        num_local_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, n_group=1, topk_group=1,
        routed_scaling_factor=2.446, norm_topk_prob=True,
        num_hidden_layers=3, hidden_act="silu",
        # A context past one device prefill chunk (512 rows), which the
        # hybrid batch gate's prompt crosses: KimiLinearConfig's context key
        # is model_max_length.
        model_max_length=1024,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        # The donor vocabulary is far smaller than this config's default
        # pad/bos/eos, and nn.Embedding refuses a padding_idx outside it.
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    ), "qwen2.5-tiny-random"),
    # THE DENSE LLAMA FAMILY, one per class, each at a value that BITES; the
    # 32000-token donor keeps them a few MB. The same graphs from a GGUF are
    # scripts/c6gold.py's synth-smollm3 .. synth-mistral3.
    #
    # SmolLM3 at an interval of 3 over six layers, so layers 2 and 5 are NoPE
    # and a converter that hardcodes llama.cpp's 4 cannot pass; tied, as the
    # class defaults.
    "synth-hf-smollm3": ("smollm3", dict(**SPM_IDS, num_hidden_layers=6, no_rope_layer_interval=3,
                                         tie_word_embeddings=True), "tiny-random-MixtralForCausalLM"),
    # Arcee: up -> relu^2 -> down, no gate.
    "synth-hf-arcee": ("arcee", dict(**SPM_IDS, num_hidden_layers=3, hidden_act="relu2"), "tiny-random-MixtralForCausalLM"),
    # Seed-OSS: q/k/v biased, a head of 32 against hidden/heads 16.
    "synth-hf-seedoss": ("seed_oss", dict(**SPM_IDS, num_hidden_layers=3, head_dim=32, attention_bias=True,
                                          attention_out_bias=False, attention_dropout=0.0,
                                          residual_dropout=0.0), "tiny-random-MixtralForCausalLM"),
    # OLMo 2: post-norms only, the q/k norm over the whole projection. MHA,
    # since a GQA OLMo 2 is refused by name.
    "synth-hf-olmo2": ("olmo2", dict(**SPM_IDS, num_hidden_layers=3, num_key_value_heads=4), "tiny-random-MixtralForCausalLM"),
    # OLMo 3: OLMo 2's block with a window of 4 on three layers in four.
    "synth-hf-olmo3": ("olmo3", dict(**SPM_IDS, num_hidden_layers=4, num_key_value_heads=4, sliding_window=4),
                       "tiny-random-MixtralForCausalLM"),
    # EXAONE 4: post-norms, a per-head q/k norm, a window of 4 on three layers
    # in four and no rotary on the fourth.
    "synth-hf-exaone4": ("exaone4", dict(**SPM_IDS, num_hidden_layers=4, sliding_window=4, sliding_window_pattern=4),
                         "tiny-random-MixtralForCausalLM"),
    # Ministral 3: YaRN factor 16 over 4 original positions, so the temperature
    # (llama_4_scaling_beta 0.5) moves at every 4th position; mscale_all_dim 0.5
    # against mscale 1 puts the magnitude at 1.12 rather than exactly one.
    # Llama 3's rope scaling over 16 original positions: every pair's
    # wavelength is past 4, so all but the first are divided by the factor and
    # the first is blended, which bites at every one of the 8 positions.
    "synth-hf-llama3": ("llama", dict(**SPM_IDS, rope_parameters={
        "rope_type": "llama3", "rope_theta": 10000.0, "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 16}), "tiny-random-LlamaForCausalLM"),
    # Two classes added by their library files alone (convert/models/gptneox.json
    # and opt.json): GPT-NeoX's parallel block, its per-head fused q|k|v and its
    # rotary over half of each head; OPT's learned positions read past an offset
    # of 2 and its ReLU MLP.
    "synth-hf-gptneox": ("gpt_neox", dict(**SPM_IDS, num_hidden_layers=3, hidden_act="gelu", use_parallel_residual=True,
                                          rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                           "partial_rotary_factor": 0.5}),
                         "tiny-random-MixtralForCausalLM"),
    "synth-hf-opt": ("opt", dict(**SPM_IDS, num_hidden_layers=3, ffn_dim=96, word_embed_proj_dim=64,
                                 activation_function="relu", do_layer_norm_before=True),
                     "tiny-random-MixtralForCausalLM"),
    "synth-hf-ministral3": ("ministral3", dict(**SPM_IDS, 
        num_hidden_layers=3,
        rope_parameters={"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
                         "original_max_position_embeddings": 4, "beta_fast": 32.0, "beta_slow": 1.0,
                         "mscale": 1.0, "mscale_all_dim": 0.5, "llama_4_scaling_beta": 0.5},
    ), "tiny-random-MixtralForCausalLM"),
}
TOKFILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]


def synth(name):
    import shutil
    from transformers import AutoTokenizer
    mt, over, donor = SYNTH[name]
    dst = os.path.realpath(os.path.join(MODELS, name))
    src = os.path.realpath(os.path.join(MODELS, donor))
    vocab = Tokenizer.from_file(os.path.join(src, "tokenizer.json")).get_vocab_size()
    kw = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, vocab_size=vocab, max_position_embeddings=512,
              initializer_range=0.1, tie_word_embeddings=(mt == "gemma"))
    kw.update(over)
    cfg = AutoConfig.for_model(mt, **kw)
    torch.manual_seed(1234)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1:  # norms (and biases): random, so their wiring shows
                p.copy_(torch.randn_like(p) * 0.5 + (0 if mt == "gemma" else 1))
        # ★ AND THE SELECTION BIAS BY NAME, BECAUSE IT IS A BUFFER AND NOT A
        # PARAMETER. DeepseekV3TopkRouter declares
        # `self.e_score_correction_bias = nn.Buffer(torch.zeros(num_experts))`,
        # so the walk above never reached it and it stayed at ZERO -- and a zero
        # correction is the IDENTITY. Every gate that claimed to cover
        # e_score_correction_bias was therefore comparing two arms that compute
        # the same function, on the host and on all three device backends.
        #
        # Found by running internal/oracle's MoEFaultBiasedWeights against the
        # DEVICE: taking the mixture weights from the BIASED scores instead of
        # the unbiased ones passed, because the two were equal. RULE 10's
        # "check the configuration under test was actually SELECTED", arriving
        # as a fixture VALUE rather than as a missing flag -- the tensor was
        # present, converted, uploaded and read, and it did nothing.
        #
        # It is NOT centred on 1 like the norms: a correction is a signed
        # nudge, and centring it would move every expert's score the same way
        # and change no selection.
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.5)
    m.save_pretrained(dst)
    for f in TOKFILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), dst)
    print("made", dst)


# Qwen3.5 as Qwen3_5ForConditionalGeneration, the class convert/models/qwen3_5.json
# reads: the text model under model.language_model., hyperparameters under
# text_config, and a one-block tower under model.visual. that the converter
# skips. Four value heads over two key heads, so transformers' grouped head
# order and the beta/alpha interleave bite; a full block between recurrent
# ones; the (1 + w) norms random around 0 and the gated norm, a plain weight,
# around 1. Its own builder: AutoModelForCausalLM would make the flat text class.
QWEN35_TEXT = dict(
    hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=32, hidden_act="silu", attn_output_gate=True,
    linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_num_key_heads=2, linear_num_value_heads=4,
    layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    max_position_embeddings=512, rms_norm_eps=1e-6, initializer_range=0.1,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)


def synth_qwen35(name):
    import shutil
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    donor = "qwen2.5-tiny-random"
    dst = os.path.realpath(os.path.join(MODELS, name))
    src = os.path.realpath(os.path.join(MODELS, donor))
    vocab = Tokenizer.from_file(os.path.join(src, "tokenizer.json")).get_vocab_size()
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  patch_size=16, spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                  deepstack_visual_indexes=[])
    cfg = Qwen3_5Config(text_config=dict(QWEN35_TEXT, vocab_size=vocab), vision_config=vision,
                        tie_word_embeddings=False)
    torch.manual_seed(1234)
    m = Qwen3_5ForConditionalGeneration(cfg).to(torch.float32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.copy_(torch.randn_like(p) * 0.5 + (1 if n.endswith("linear_attn.norm.weight") else 0))
    m.save_pretrained(dst)
    for f in TOKFILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), dst)
    print("made", dst)


def qwen35Stale(name):
    """True when name has no fixture, or one whose text_config disagrees with QWEN35_TEXT."""
    p = os.path.join(MODELS, name, "config.json")
    if not os.path.exists(p):
        return True
    have = json.load(open(p)).get("text_config", {})
    return any(k in have and have[k] != v for k, v in QWEN35_TEXT.items() if k != "rope_parameters")


BUILT = {"synth-hf-qwen35": (synth_qwen35, qwen35Stale)}


# A windowed fixture runs long enough to cross KV page boundaries: a window
# SHORTER than a page (32 positions) is read out of one page until it is not,
# and 8 positions never leave the first.
LONG = {"synth-phi3": 80, "synth-phi3-longrope": 40, "synth-hf-qwen3-yarn": 40, "synth-hf-mixtral-swa": 40, "synth-hf-olmo3": 40, "synth-hf-exaone4": 40, "synth-hf-ministral3": 24, "synth-hf-gptneox": 24, "synth-hf-opt": 24}


def ids_for(vocab, n=NPOS):
    return [(i * 7919 + 13) % min(vocab, 32000) for i in range(n)]


def synthStale(name):
    """True when name has no fixture, or one whose config disagrees with SYNTH."""
    p = os.path.join(MODELS, name, "config.json")
    if not os.path.exists(p):
        return True
    try:
        have = json.load(open(p))
    except Exception:
        return True
    for k, v in SYNTH[name][1].items():
        if k in have and have[k] != v:
            print(f"{name}: {k} is {have[k]} on disk and {v} in the recipe -- rebuilding")
            return True
    return False


def main(dirs):
    os.makedirs(OUT, exist_ok=True)
    for d in dirs:
        # ★ REBUILT WHEN THE RECIPE MOVED, NOT ONLY WHEN THE DIRECTORY IS
        # ABSENT. This was `not os.path.exists(config.json)`, so editing a
        # SYNTH entry and re-running regenerated the GOLDEN from the OLD
        # weights: the fixture silently kept the shape it was first built
        # with, and the golden it produced was correct for a model the recipe
        # no longer describes. Changing linear_head_dim from 16 to 32 and
        # getting a byte-identical argmax list is what caught it.
        if d in SYNTH and synthStale(d):
            synth(d)
        if d in BUILT and BUILT[d][1](d):
            BUILT[d][0](d)
        path = os.path.realpath(os.path.join(MODELS, d))
        cfg = AutoConfig.from_pretrained(path)
        # A class with a tower is loaded as itself, its vocabulary its text model's.
        cls = AutoModelForCausalLM
        if d in BUILT:
            import transformers
            cls = getattr(transformers, cfg.architectures[0])
        vocab = getattr(cfg, "vocab_size", None) or cfg.get_text_config().vocab_size
        note = None
        # transformers 5 reads hidden_act ONLY; hidden_activation is the key 4.x
        # added so gemma's mislabelled "gelu" (trained as the tanh approximation)
        # could be corrected. A converter that honours it is right and this
        # version of the reference is not, so the reference is told.
        ha = getattr(cfg, "hidden_activation", None)
        if ha and ha != cfg.hidden_act:
            note = "hidden_act %s -> %s (hidden_activation)" % (cfg.hidden_act, ha)
            cfg.hidden_act = ha
        m = cls.from_pretrained(path, config=cfg, dtype=torch.float32)
        m.eval()
        ids = ids_for(vocab, LONG.get(d, NPOS))
        with torch.no_grad():
            lg = m(torch.tensor([ids])).logits[0]
        rows = []
        for p in range(len(ids)):
            r = lg[p]
            rows.append({
                "argmax": int(r.argmax()),
                "max": float(r.max()),
                "lse": float(torch.logsumexp(r, 0)),
                "head": [round(float(x), 7) for x in r[:NLOGIT]],
            })
        tk = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        # The specials the class adds: a class that reads add_bos_token or
        # add_eos_token rebuilds tokenizer.json's post-processor from them.
        try:
            from transformers import AutoTokenizer
            ctk = AutoTokenizer.from_pretrained(path)
        except Exception:
            ctk = None
        sp = None
        if os.path.exists(os.path.join(path, "tokenizer.model")):
            import sentencepiece
            sp = sentencepiece.SentencePieceProcessor(model_file=os.path.join(path, "tokenizer.model"))
        toks = []
        for x in TEXTS:
            e = tk.encode(x).ids
            r = {"text": x, "ids": e, "dec": tk.decode(e, skip_special_tokens=True)}
            if ctk is not None:
                c, cp = ctk(x)["input_ids"], ctk(x, add_special_tokens=False)["input_ids"]
                k = next((i for i in range(len(c) - len(cp) + 1) if c[i:i + len(cp)] == cp), None)
                if k is not None:
                    cls = c[:k] + tk.encode(x, add_special_tokens=False).ids + c[k + len(cp):]
                    if cls != e:
                        r["cls_ids"] = cls
            if sp is not None:
                # The post-processor's leading specials (a BOS), then the
                # sentencepiece encoding; sentencepiece adds no specials itself.
                lead = e[:len(e) - len(tk.encode(x, add_special_tokens=False).ids)]
                r["sp_ids"] = lead + sp.encode(x)
                r["sp_dec"] = tk.decode(r["sp_ids"], skip_special_tokens=True)
            toks.append(r)
        # The checkpoint's bytes: the gates sweep one over 2 GiB only when asked.
        size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors"))
        rec = {"dir": d, "ids": ids, "vocab": vocab, "pos": rows, "tok": toks, "bytes": size}
        if note:
            rec["note"] = note
        with open(os.path.join(OUT, d + ".json"), "w") as f:
            json.dump(rec, f)
        print(d, "argmax", [r["argmax"] for r in rows], note or "")


if __name__ == "__main__":
    main(sys.argv[1:] or DIRS + list(SYNTH))
