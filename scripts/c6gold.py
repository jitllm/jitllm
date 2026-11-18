#!/usr/bin/env python3
"""Build the classic-transformer (C6) fixtures and record transformers' logits.

    $JITLLM_HF_PY scripts/c6gold.py [name ...]
    JITLLM_MODELS=/path/to/models ... (where the GGUFs go)

-> $JITLLM_MODELS/<name>.gguf and testdata/golden/c6/<name>.json

THE ORACLE FOR EACH GRAPH IS transformers' OWN CLASS, AND THE GGUF IS WRITTEN BY
llama.cpp's OWN CONVERTER. A hand-written GGUF (llama4gold.py's route) would
encode this script's reading of llama.cpp's tensor names, permutations and key
spellings; running convert_hf_to_gguf.py puts the SAME bytes in front of jitllm
that a real HuggingFace-to-GGUF conversion does -- phi2's fused or split q/k/v,
falcon's reordered query_key_value, starcoder's MQA packing -- so a converter
bug on jitllm's side cannot hide behind a matching bug in the fixture.

Each fixture carries every feature of its graph at a value that BITES: the
LayerNorm weights and biases are perturbed away from 1 and 0 (transformers
initialises them to exactly the identity, which would let a missing norm bias
pass), every projection bias is non-zero, and the rotary is partial wherever
the architecture's is. Every tensor is F32, so what separates the two arms is
reduction order.

The tokenizer is the family's real one, copied from
$CV_TOK/<family> (downloaded from the model's own repo), so the
converter writes the pre-tokenizer name llama.cpp would write for it. Its ids
come from tokenizer.json, or from tokenizer.model for a SentencePiece family.
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from tokenizers import Tokenizer

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "c6")
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
]


def phi2():
    from transformers import PhiConfig, PhiForCausalLM
    # llama.cpp's Phi2Model writes feed_forward_length = 4*n_embd whatever the
    # config says, so the fixture is built at exactly that width.
    cfg = PhiConfig(vocab_size=51200, hidden_size=64, intermediate_size=256,
                    num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                    partial_rotary_factor=0.5, hidden_act="gelu_new", layer_norm_eps=1e-5,
                    max_position_embeddings=2048, rope_theta=10000.0,
                    tie_word_embeddings=False, qk_layernorm=False)
    return PhiForCausalLM, cfg, "phi-2"


def starcoder():
    from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM
    # MQA (llama.cpp's converter writes head_count_kv 1 and nothing else) and
    # n_inner = 4*n_embd (it writes that whatever the config says). n_positions
    # is short so the fixture's table is small; twelve positions use 12 rows.
    cfg = GPTBigCodeConfig(vocab_size=49152, n_embd=64, n_inner=256, n_layer=3, n_head=4,
                           n_positions=64, multi_query=True,
                           activation_function="gelu_pytorch_tanh", layer_norm_epsilon=1e-5,
                           tie_word_embeddings=True)
    return GPTBigCodeForCausalLM, cfg, "starcoder"


def commandr():
    from transformers import CohereConfig, CohereForCausalLM
    # logit_scale 0.0625 is Command-R's own; GQA 4:2 so the grouped KV is
    # exercised; the tokenizer is command-r's (blob 9d110643, via mlx-community).
    cfg = CohereConfig(vocab_size=256000, hidden_size=64, intermediate_size=128,
                       num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                       logit_scale=0.0625, use_qk_norm=False, layer_norm_eps=1e-5,
                       rope_theta=10000.0, max_position_embeddings=2048, tie_word_embeddings=True,
                       bos_token_id=5, eos_token_id=255001, pad_token_id=0)
    return CohereForCausalLM, cfg, "command-r"


def cohere2():
    from transformers import Cohere2Config, Cohere2ForCausalLM
    # Four layers so layer 3 is the one global (unrotated) layer at llama.cpp's
    # default period 4, and a window of 4 so the sliding layers forget inside
    # twelve positions. rotary_pct is a key llama.cpp's converter reads and
    # transformers' class does not define, so it is carried as an extra.
    cfg = Cohere2Config(vocab_size=256000, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        logit_scale=0.25, layer_norm_eps=1e-5, rope_theta=10000.0,
                        sliding_window=4, sliding_window_pattern=4,
                        max_position_embeddings=2048, tie_word_embeddings=True,
                        bos_token_id=5, eos_token_id=255001, pad_token_id=0)
    cfg.rotary_pct = 1.0
    return Cohere2ForCausalLM, cfg, "command-r"


def stablelm(parallel):
    def spec():
        from transformers import StableLmConfig, StableLmForCausalLM
        # StableLM 2's shape: biased q/k/v and an unbiased o, NEOX rotary on a
        # QUARTER of each head (4 of 16), SwiGLU, untied head. The parallel
        # twin drops the FFN norm, which is how llama.cpp reads the residual.
        cfg = StableLmConfig(vocab_size=100352, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                             partial_rotary_factor=0.25, use_qkv_bias=True, qk_layernorm=False,
                             use_parallel_residual=parallel, hidden_act="silu", layer_norm_eps=1e-5,
                             rope_theta=10000.0, max_position_embeddings=2048,
                             tie_word_embeddings=False)
        return StableLmForCausalLM, cfg, "stablelm2"
    return spec


def starcoder2():
    from transformers import Starcoder2Config, Starcoder2ForCausalLM
    # GQA 4:2, every matrix biased (use_bias), full NEOX rotary. The window is
    # far past the golden's twelve positions, as it is past the prompts
    # llama.cpp is compared on (its GGUF carries no window at all).
    cfg = Starcoder2Config(vocab_size=49152, hidden_size=64, intermediate_size=256,
                           num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                           use_bias=True, hidden_act="gelu_pytorch_tanh", norm_epsilon=1e-5,
                           rope_theta=10000.0, max_position_embeddings=2048, sliding_window=4096,
                           tie_word_embeddings=True)
    return Starcoder2ForCausalLM, cfg, "starcoder"


def falcon(two_norms):
    def spec():
        from transformers import FalconConfig, FalconForCausalLM
        # ★ THE ACTIVATION IS GELU-tanh HERE AND FALCON TRAINS WITH erf GELU.
        # llama.cpp runs falcon.cpp's FFN through ggml_gelu, the tanh form,
        # and so does this engine (RULE 7m: the divergence from the model's
        # own function is llama.cpp's and is adopted knowingly); the fixture
        # pins the graph, so it pins the activation both engines run.
        # 7B's shape: MQA, one input norm feeding both branches. 40B's: GQA
        # and two input norms (ln_attn, ln_mlp).
        if two_norms:
            cfg = FalconConfig(vocab_size=65024, hidden_size=64, num_hidden_layers=3,
                               num_attention_heads=4, num_kv_heads=2,
                               new_decoder_architecture=True, num_ln_in_parallel_attn=2,
                               parallel_attn=True, bias=False, alibi=False,
                               activation="gelu_pytorch_tanh", layer_norm_epsilon=1e-5,
                               rope_theta=10000.0, max_position_embeddings=2048,
                               tie_word_embeddings=True)
        else:
            cfg = FalconConfig(vocab_size=65024, hidden_size=64, num_hidden_layers=3,
                               num_attention_heads=4, num_kv_heads=1, multi_query=True,
                               new_decoder_architecture=False, parallel_attn=True, bias=False,
                               alibi=False, activation="gelu_pytorch_tanh",
                               layer_norm_epsilon=1e-5, rope_theta=10000.0,
                               max_position_embeddings=2048, tie_word_embeddings=True)
        return FalconForCausalLM, cfg, "falcon"
    return spec


def nemotron():
    from transformers import NemotronConfig, NemotronForCausalLM
    # Nemotron-Mini's shape: GQA, NEOX rotary on HALF of each head, an
    # ungated relu2 FFN, LayerNorm1p (transformers adds the 1 in forward;
    # llama.cpp's converter adds it to the weight, which is what this pins).
    # mlp_bias on so the optional FFN biases llama.cpp reads are exercised.
    cfg = NemotronConfig(vocab_size=256000, hidden_size=64, intermediate_size=256,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         hidden_act="relu2", norm_eps=1e-5, partial_rotary_factor=0.5,
                         rope_theta=10000.0, max_position_embeddings=2048,
                         attention_bias=False, mlp_bias=True, tie_word_embeddings=False)
    return NemotronForCausalLM, cfg, "nemotron"


def dbrx():
    from transformers import DbrxConfig, DbrxForCausalLM
    # DBRX's shape at toy size: GQA 4:2, a softmax top-2 of 4 SwiGLU experts
    # renormalised over the two, bias-free LayerNorms, NEOX rotary over the
    # whole head, an untied head -- and clip_qkv at 1.5, where q, k and v
    # under these perturbed weights run at a standard deviation near 1.6, so
    # the clamp bites on a third of every projection. (Real DBRX clips at 8.)
    # ★ THE ROTARY BASE IS STATED TWICE ON PURPOSE. transformers 5 reads
    # rope_parameters and llama.cpp's converter reads attn_config.rope_theta;
    # left alone the first is 10000 and the second 500000, and the two engines
    # would be compared on two different models.
    cfg = DbrxConfig(d_model=64, n_heads=4, n_layers=3, max_seq_len=2048, vocab_size=100352,
                     attn_config={"clip_qkv": 1.5, "kv_n_heads": 2, "rope_theta": 500000.0},
                     ffn_config={"ffn_hidden_size": 32, "moe_num_experts": 4, "moe_top_k": 2,
                                 "ffn_act_fn": {"name": "silu"}, "hidden_size": 64},
                     rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
                     tie_word_embeddings=False)
    return DbrxForCausalLM, cfg, "dbrx"


def smollm3():
    from transformers import SmolLM3Config, SmolLM3ForCausalLM
    # Four layers so layer 3 is the NoPE one at the default interval of 4,
    # which is the constant llama.cpp hardcodes; GQA 4:2, tied head.
    cfg = SmolLM3Config(vocab_size=128256, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        rms_norm_eps=1e-6, max_position_embeddings=2048,
                        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                        tie_word_embeddings=True)
    return SmolLM3ForCausalLM, cfg, "smollm3"


def arcee():
    from transformers import ArceeConfig, ArceeForCausalLM
    # AFM's shape: GQA, an ungated relu2 FFN, untied head.
    cfg = ArceeConfig(vocab_size=128256, hidden_size=64, intermediate_size=256,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      hidden_act="relu2", rms_norm_eps=1e-5, max_position_embeddings=2048,
                      rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                      tie_word_embeddings=False)
    return ArceeForCausalLM, cfg, "arcee"


def apertus():
    from transformers import ApertusConfig, ApertusForCausalLM
    # Apertus's shape: GQA 4:2, the per-head q/k RMSNorm, an ungated xIELU
    # MLP, llama 3's rotary scaling at an original context short enough
    # (64) that the eight pairs of a 16-wide head fall in all three bands --
    # factor 1, the smooth blend and the full factor -- and an untied head.
    cfg = ApertusConfig(vocab_size=131072, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        hidden_act="xielu", rms_norm_eps=1e-5, max_position_embeddings=512,
                        rope_parameters={"rope_type": "llama3", "rope_theta": 10000.0, "factor": 8.0,
                                         "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                         "original_max_position_embeddings": 64},
                        tie_word_embeddings=False)
    return ApertusForCausalLM, cfg, "apertus"


def apertus_buffers(m):
    """xIELU's beta and eps are buffers, which perturb() does not reach: beta
    at its 0.5 would hide a beta read as another layer's, and eps at -1e-6
    clamps almost nothing. Each layer gets its own beta and an eps wide enough
    that min(x, eps) bites."""
    with torch.no_grad():
        for layer in m.model.layers:
            a = layer.mlp.act_fn
            a.beta.copy_(0.3 + 0.4 * torch.rand(()))
            a.eps.copy_(-(0.05 + 0.3 * torch.rand(())))


# AFTER runs after perturb and before the checkpoint is saved.
AFTER = {"synth-apertus": apertus_buffers}


def seed_oss():
    from transformers import SeedOssConfig, SeedOssForCausalLM
    # Seed-OSS's shape: q/k/v biased and o not, a head wider than
    # hidden/heads (32 against 16, as 36B's 128 against 64), GQA 4:2.
    cfg = SeedOssConfig(vocab_size=155136, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=32, attention_bias=True, attention_out_bias=False,
                        attention_dropout=0.0, residual_dropout=0.0,
                        rms_norm_eps=1e-6, max_position_embeddings=2048,
                        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                        tie_word_embeddings=False)
    return SeedOssForCausalLM, cfg, "seed-oss"


def olmo2():
    from transformers import Olmo2Config, Olmo2ForCausalLM
    # OLMo 2's shape: no pre-norms, q/k RMSNorm over the whole projection,
    # post-norms. MHA, since the whole-projection k norm is n_kv*head_dim wide
    # and the 1B and 7B are MHA.
    cfg = Olmo2Config(vocab_size=100352, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                      rms_norm_eps=1e-6, max_position_embeddings=2048,
                      rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
                      tie_word_embeddings=False, pad_token_id=100277, eos_token_id=100257)
    return Olmo2ForCausalLM, cfg, "olmo2"


def olmo2_gqa():
    from transformers import Olmo2Config, Olmo2ForCausalLM
    # OLMo-2-32B's shape at toy size: GQA 4:2, so the whole-projection k norm
    # is n_kv*head_dim wide and q's n_head*head_dim.
    cfg = Olmo2Config(vocab_size=100352, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      rms_norm_eps=1e-6, max_position_embeddings=2048,
                      rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
                      tie_word_embeddings=False, pad_token_id=100277, eos_token_id=100257)
    return Olmo2ForCausalLM, cfg, "olmo2"


def olmo3():
    from transformers import Olmo3Config, Olmo3ForCausalLM
    # OLMo 3's shape: a window of 4 on three layers in four, GQA 4:2 (the 32B
    # is GQA), and YaRN on the global layer only -- the sliding layers keep a
    # default rotary, magnitude one. original_max_position_embeddings is 4 and
    # factor 8, so the YaRN ramp and its 1.21 magnitude bite inside twelve
    # positions.
    cfg = Olmo3Config(vocab_size=100352, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      rms_norm_eps=1e-6, max_position_embeddings=2048, sliding_window=4,
                      rope_theta=500000.0,
                      rope_scaling={"rope_type": "yarn", "factor": 8.0,
                                    "original_max_position_embeddings": 4,
                                    "beta_fast": 32.0, "beta_slow": 1.0},
                      tie_word_embeddings=False, pad_token_id=100277, eos_token_id=100257)
    return Olmo3ForCausalLM, cfg, "olmo2"


def olmo():
    from transformers import OlmoConfig, OlmoForCausalLM
    # The first OLMo: LayerNorms with no weight and no bias, q, k and v each
    # clamped to [-clip_qkv, clip_qkv] before the rotary (the 7B's 0424
    # revision clamps at 8), a SwiGLU MLP, untied. A clamp of 0.1 bites at
    # this width: the projections' outputs run to several tenths.
    cfg = OlmoConfig(vocab_size=50304, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                     max_position_embeddings=2048, clip_qkv=0.1,
                     rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                     tie_word_embeddings=False, pad_token_id=1, eos_token_id=50279)
    return OlmoForCausalLM, cfg, "olmo"


def exaone4():
    from transformers import Exaone4Config, Exaone4ForCausalLM
    # The 32B's form at toy size: a window of 4 on three layers in four and
    # no rotary on the fourth, per-head q/k norm, post-norms only, GQA 4:2.
    cfg = Exaone4Config(vocab_size=102400, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        rms_norm_eps=1e-5, max_position_embeddings=2048,
                        sliding_window=4, sliding_window_pattern=4,
                        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
                        tie_word_embeddings=False)
    return Exaone4ForCausalLM, cfg, "exaone4"


def mistral3():
    from transformers import Ministral3Config, Ministral3ForCausalLM
    # Ministral 3's text block: YaRN, and llama_4_scaling_beta as the
    # temperature on every layer. original_max_position_embeddings is 4 so
    # floor(pos/4) moves twice inside twelve positions, and beta is 0.5 so it
    # bites; mscale_all_dim 0.5 against mscale 1 puts the YaRN magnitude at
    # 1.12 where the real model's 1/1 puts it at exactly one.
    cfg = Ministral3Config(vocab_size=131072, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, rms_norm_eps=1e-5, max_position_embeddings=2048,
                           rope_parameters={"rope_type": "yarn", "rope_theta": 1000000.0,
                                            "factor": 16.0, "original_max_position_embeddings": 4,
                                            "beta_fast": 32.0, "beta_slow": 1.0,
                                            "mscale": 1.0, "mscale_all_dim": 0.5,
                                            "llama_4_scaling_beta": 0.5},
                           tie_word_embeddings=False)
    return Ministral3ForCausalLM, cfg, "mistral3"


def olmo3_af():
    # synth-olmo3 with YaRN's magnitude STATED: attention_factor 1.5 where the
    # formula gives 0.1*ln(8)+1 = 1.208. transformers puts the stated value on
    # cos and sin; llama.cpp's converter writes it as
    # rope.scaling.yarn_attn_factor and llama.cpp's runtime never reads it, so
    # transformers is the only oracle here.
    cls, cfg, tok = olmo3()
    cfg.rope_parameters["full_attention"]["attention_factor"] = 1.5
    return cls, cfg, tok


def mistral3_af():
    # synth-mistral3 with attention_factor 0.75, which outranks the
    # mscale/mscale_all_dim ratio (1.12 here) in _compute_yarn_parameters.
    cls, cfg, tok = mistral3()
    cfg.rope_parameters["attention_factor"] = 0.75
    return cls, cfg, tok


def deepseek_yarn(attention_factor=None):
    def spec():
        from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
        # MLA with YaRN, every block dense so the rotary is the subject:
        # factor 8 over 4 original positions bites inside twelve, and mscale ==
        # mscale_all_dim (every shipped DeepSeek) leaves cos and sin at exactly
        # one while the score takes mscale(8, 0.707)^2 = 1.31. The stated
        # attention_factor variant puts 0.8 on cos and sin beside that -- below
        # one, because 1.5 sharpened position 3's softmax enough that Q8_0
        # rounding alone read 7e-2 there (TestSafetensorsQ8MatchTransformers).
        rope = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 8.0,
                "original_max_position_embeddings": 4, "beta_fast": 32.0,
                "beta_slow": 1.0, "mscale": 0.707, "mscale_all_dim": 0.707}
        if attention_factor is not None:
            rope["attention_factor"] = attention_factor
        cfg = DeepseekV3Config(vocab_size=151936, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                               q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16,
                               qk_rope_head_dim=8, v_head_dim=16, head_dim=8,
                               n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
                               moe_intermediate_size=32, first_k_dense_replace=3,
                               rms_norm_eps=1e-6, max_position_embeddings=2048,
                               rope_parameters=rope, tie_word_embeddings=False)
        return DeepseekV3ForCausalLM, cfg, "qwen2moe"
    return spec


def glm4():
    from transformers import Glm4Config, Glm4ForCausalLM
    # GLM-4-0414's shape: q/k/v biased and o not, a norm after the attention
    # and after the MLP beside the pre-norms, a fused gate|up, and the rotary
    # on half of each 32-wide head in NORM pairs (no sections, so llama.cpp's
    # converter leaves q and k as they are). GQA 4:2.
    cfg = Glm4Config(vocab_size=151552, hidden_size=128, intermediate_size=192,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                     attention_bias=True, rms_norm_eps=1e-5, max_position_embeddings=2048,
                     rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                      "partial_rotary_factor": 0.5},
                     tie_word_embeddings=False)
    return Glm4ForCausalLM, cfg, "glm4moe"


SPECS = {
    "synth-glm4": glm4,
    "synth-olmo3-af": olmo3_af,
    "synth-mistral3-af": mistral3_af,
    "synth-deepseek-yarn": deepseek_yarn(),
    "synth-deepseek-yarn-af": deepseek_yarn(0.8),
    "synth-smollm3": smollm3,
    "synth-apertus": apertus,
    "synth-arcee": arcee,
    "synth-seedoss": seed_oss,
    "synth-olmo2": olmo2,
    "synth-olmo2-gqa": olmo2_gqa,
    "synth-olmo3": olmo3,
    "synth-olmo": olmo,
    "synth-exaone4": exaone4,
    "synth-mistral3": mistral3,
    "synth-phi2": phi2,
    "synth-starcoder2": starcoder2,
    "synth-nemotron": nemotron,
    "synth-falcon": falcon(False),
    "synth-falcon40": falcon(True),
    "synth-stablelm": stablelm(False),
    "synth-stablelm-par": stablelm(True),
    "synth-starcoder": starcoder,
    "synth-commandr": commandr,
    "synth-cohere2": cohere2,
    "synth-dbrx": dbrx,
}


# Tokenizer families whose tokenizer_config must be loaded as tokenizer.json
# says rather than as the class it names; see build().
VERBATIM_TOKENIZER = {"dbrx"}


def perturb(m):
    # ★ A NORM IS A PARAMETER OF A NORM MODULE, NOT ONE WITH "norm" SOMEWHERE
    # IN ITS PATH. DBRX wraps its attention in a module called norm_attn_norm,
    # so the test used to be "norm" in the full name and it gave Wqkv and
    # out_proj weights of 1 +- 0.3: every output of the projection was nearly
    # the sum of its inputs, the attention output carried a per-token common
    # mode of 30-90 against a spread of 4, and the next LayerNorm cancelled it
    # away at a precision that put transformers' OWN f32 run 2e-9 from its
    # f64 one. The module's own name is the test; every other fixture's
    # parameters classify exactly as they did.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("weight") and "norm" in (n.split(".")[-2] if "." in n else ""):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


def stateClassEOS(d, dst):
    """The GGUF converter takes add_eos_token from tokenizer.json's template;
    a class may rebuild its own from tokenizer_config's flag and append
    nothing (OLMo's GPTNeoXTokenizer). The GGUF states the class's, which the
    golden is read from; the GGUF oracle's completion tool refuses a GGUF
    that would end every prompt."""
    from gguf import GGUFReader
    r = GGUFReader(dst, "r+")
    f = r.fields.get("tokenizer.ggml.add_eos_token")
    if f is None or not f.parts[f.data[0]][0]:
        return
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(d)
    ids = tk("a")["input_ids"]
    if tk.eos_token_id is not None and ids and ids[-1] == tk.eos_token_id:
        return
    f.parts[f.data[0]][0] = 0
    r.data.flush()
    print("%s: add_eos_token cleared, as %s appends no EOS" % (dst, type(tk).__name__))


def build(name):
    cls, cfg, tokfam = SPECS[name]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20260927)
    m = cls(cfg).float().eval()
    perturb(m)
    if name in AFTER:
        AFTER[name](m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(os.path.join(TOKDIR, tokfam)):
        if f != "config.json":
            shutil.copy(os.path.join(TOKDIR, tokfam, f), d)
    # ★ transformers 5 LOADS A tokenizer_class OF GPT2Tokenizer AS GPT-2's OWN
    # REGEX, DISCARDING tokenizer.json's pre-tokenizer, and llama.cpp's
    # converter hashes AutoTokenizer's output to name the pipeline -- so the
    # DBRX tokenizer (cl100k's regex) came out labelled granite-docling
    # (GPT-2's), and llama.cpp and jitllm then agreed with each other and not
    # with tokenizer.json on "(self):" and a tab. PreTrainedTokenizerFast takes
    # tokenizer.json verbatim, which is the artefact the golden is read from.
    if tokfam in VERBATIM_TOKENIZER:
        tc = os.path.join(d, "tokenizer_config.json")
        j = json.load(open(tc))
        j["tokenizer_class"] = "PreTrainedTokenizerFast"
        json.dump(j, open(tc, "w"), indent=2)
    dst = os.path.join(MODELS, name + ".gguf")
    subprocess.run([PY, os.path.join(LCPP, "convert_hf_to_gguf.py"), d,
                    "--outtype", "f32", "--outfile", dst], check=True)
    stateClassEOS(d, dst)
    ids = [(i * 7919 + 13) % min(cfg.vocab_size, 50000) for i in range(NPOS)]
    with torch.no_grad():
        lg = m(torch.tensor([ids])).logits[0]
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
    # A SentencePiece family's authority is its tokenizer.model, which the
    # weights were trained with; tokenizer.json is a conversion of it, and on
    # Nemotron's they part on a leading space (Metaspace prepend_scheme
    # "first" adds no "\u2581" before a text that starts with one;
    # sentencepiece and llama.cpp both do).
    spm = os.path.join(d, "tokenizer.model")
    if os.path.exists(spm):
        import sentencepiece
        sp = sentencepiece.SentencePieceProcessor(model_file=spm)
        texts = [{"text": t, "ids": sp.encode(t)} for t in TEXTS]
    else:
        tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"ids": ids, "vocab": cfg.vocab_size, "pos": rows, "texts": texts},
              open(os.path.join(OUT, name + ".json"), "w"))
    print("wrote", dst, "argmax", [r["argmax"] for r in rows])


if __name__ == "__main__":
    for n in sys.argv[1:] or list(SPECS):
        build(n)
