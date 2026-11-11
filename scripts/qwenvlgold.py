#!/usr/bin/env python3
"""Build the Qwen2-VL family fixtures and record transformers' answers.

    $JITLLM_HF_PY scripts/qwenvlgold.py [name ...]
    JITLLM_MODELS=/path/to/models  (where the GGUFs go)
    QVL_TOK=<a Qwen2-VL checkpoint dir>          (its tokenizer and processor)

-> $JITLLM_MODELS/<name>.gguf, $JITLLM_MODELS/mmproj-<name>.gguf
   and testdata/golden/qwenvl/<name>.json

THE ORACLE IS transformers' OWN Qwen2VLForConditionalGeneration (and the 2.5
class), AND BOTH GGUFs ARE WRITTEN BY llama.cpp's OWN convert_hf_to_gguf.py --
the text model plainly and the tower with --mmproj -- so jitllm converts the
same bytes a real checkpoint's conversion puts in front of it.

The prompt is text, ONE NON-SQUARE IMAGE and text, then a continuation the
engine decodes token by token. M-RoPE is what the fixture exists to pin: the
image's merged grid is H != W, so a row turned by h where it should be w, by
its sequence index where it should be its grid coordinate, or a continuation
resumed at the cache position rather than at s + max(H, W), each moves the
logits. The golden carries:

    image   the picture (RGB bytes), its pixel_values' grid and the tower's
            output rows (transformers' get_image_features), which the text
            gate splices in directly so the text side is pinned alone
    pos     every row's (t, h, w) from transformers' get_rope_index
    logits  at the last prompt row and every continuation row: argmax and the
            first NLOGIT values
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOK = refenv.under("QVL_TOK", "qvl", "hf", "Qwen2-VL-2B-Instruct")
HFDIR = refenv.under("QVL_HF", "qvl", "fixtures")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "qwenvl")
NLOGIT = 512
# The picture, in pixels, per fixture: a multiple of 28 on both sides, so the
# processor's smart_resize leaves it alone and the merged grid is (H/28) x
# (W/28). Qwen2-VL's is 3 x 5; Qwen2.5-VL's is 6 x 8, so its 4x4-unit windows
# (window_size 112) are four windows of three sizes.
#
# The PREPROCESSING golden is another picture, 160 x 240 and textured, which
# smart_resize brings down to 112 x 168 under a ceiling of 25 merge units
# (the gate sets the tower's to match): the processor's own resize -- PIL's
# BICUBIC, antialiased on the way down -- is what it records.
PREP_H, PREP_W = 160, 240
PREP_MAX = 28 * 28 * 25


def qwen2vl():
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    # head_dim 16, so 8 rotary pairs split [2, 3, 3] between t, h and w; GQA
    # 4:2. The tower is Qwen2-VL's own shape at toy size, three blocks deep:
    # llama.cpp maps merger.mlp.{bid} through the BLOCK table, so a tower
    # shallower than three cannot name mm.2.
    cfg = Qwen2VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         rms_norm_eps=1e-6, max_position_embeddings=4096,
                         rope_parameters={"rope_type": "default", "rope_theta": 1000000.0,
                                          "mrope_section": [2, 3, 3]},
                         tie_word_embeddings=False),
        vision_config=dict(depth=3, embed_dim=64, hidden_size=64, num_heads=4, mlp_ratio=4,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                           hidden_act="quick_gelu"),
        tie_word_embeddings=False)
    return Qwen2VLForConditionalGeneration, cfg, (84, 140)


def qwen25vl():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    # Qwen2.5-VL's tower at toy size: RMSNorm, a gated SiLU MLP whose width
    # (100) is not a whole quantization block -- the 3B's 3420 is not either,
    # and the converter pads it -- and window attention on every block but
    # the full ones, here every second (n_wa_pattern 2).
    cfg = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         rms_norm_eps=1e-6, max_position_embeddings=4096,
                         rope_parameters={"rope_type": "default", "rope_theta": 1000000.0,
                                          "mrope_section": [2, 3, 3]},
                         tie_word_embeddings=False),
        vision_config=dict(depth=4, hidden_size=64, out_hidden_size=64, intermediate_size=100,
                           num_heads=4, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                           window_size=112, fullatt_block_indexes=[1, 3], hidden_act="silu"),
        tie_word_embeddings=False)
    return Qwen2_5_VLForConditionalGeneration, cfg, (168, 224)


def qwen3vl_vision():
    # Qwen3-VL's tower at toy size: LayerNorm with biases, an ungated
    # GELU-tanh MLP whose width (100) is not a whole quantization block, the
    # 2-D rotary, AND a learned 8x8 position table bilinearly resampled to the
    # picture's 6x10 patch grid (up one axis, down the other), and deepstack
    # taps after blocks 0 and 2, each through its own post-shuffle-norm
    # merger into a text layer.
    return dict(depth=4, hidden_size=64, out_hidden_size=64, intermediate_size=100,
                num_heads=4, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                num_position_embeddings=64, deepstack_visual_indexes=[0, 2],
                hidden_act="gelu_pytorch_tanh")


def qwen3vl_text(**kw):
    # head_dim 16, so 8 rotary pairs, INTERLEAVED over (t, h, w): pairs 0, 3
    # and 6 turn by t, 1, 4 and 7 by h, 2 and 5 by w. [3, 3, 2] is a split
    # transformers and ggml read alike; [2, 3, 3] would send pair 6 to
    # ggml's fourth axis where transformers keeps it on t.
    return dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                rms_norm_eps=1e-6, max_position_embeddings=4096,
                rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                 "mrope_section": [3, 3, 2], "mrope_interleaved": True},
                tie_word_embeddings=False, **kw)


def qwen3vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    cfg = Qwen3VLConfig(text_config=qwen3vl_text(), vision_config=qwen3vl_vision(),
                        tie_word_embeddings=False)
    return Qwen3VLForConditionalGeneration, cfg, (96, 160)


def qwen3vlmoe():
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
    # The same tower in front of qwen3moe's block: four experts, two routed.
    cfg = Qwen3VLMoeConfig(
        text_config=qwen3vl_text(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                                 decoder_sparse_step=1),
        vision_config=qwen3vl_vision(), tie_word_embeddings=False)
    return Qwen3VLMoeForConditionalGeneration, cfg, (96, 160)


def glm4v():
    from transformers import Glm4vConfig, Glm4vForConditionalGeneration
    # GLM-4.1V at toy size. The text model is GLM-4-0414's block (biased
    # q/k/v, a norm after the attention and after the MLP, a fused gate|up)
    # with M-RoPE on half of each head: 32-wide heads, 8 rotary pairs split
    # [2, 3, 3] (llama.cpp's converter permutes q and k to NEOX for it). The
    # tower: RMSNorm, a gated SiLU MLP as wide as the text model, the 2-D
    # rotary and an 8x8 learned table resampled BICUBIC to the 6x10 patch
    # grid; a 2x2 convolution merges, then linear, LayerNorm, GELU and a
    # gated MLP (96 wide).
    cfg = Glm4vConfig(
        text_config=dict(vocab_size=151552, hidden_size=96, intermediate_size=192,
                         num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=1, head_dim=32,
                         rms_norm_eps=1e-5, max_position_embeddings=4096, attention_bias=True,
                         rope_parameters={"rope_type": "default", "rope_theta": 500000.0,
                                          "mrope_section": [2, 3, 3], "partial_rotary_factor": 0.5},
                         tie_word_embeddings=False),
        vision_config=dict(depth=3, hidden_size=64, out_hidden_size=96, intermediate_size=96,
                           num_heads=4, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                           image_size=112, hidden_act="silu", rms_norm_eps=1e-5),
        tie_word_embeddings=False)
    return Glm4vForConditionalGeneration, cfg, (84, 140)


def qwen35vl():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    # Qwen3.5: Qwen3-VL's tower with no deepstack taps, in front of the
    # hybrid text model (mtpgold.py's shape, no prediction block): linear
    # and full attention alternating, the full layers' rotary on half a head
    # under the interleaved M-RoPE.
    v = qwen3vl_vision()
    v.pop("deepstack_visual_indexes")
    cfg = Qwen3_5Config(
        text_config=dict(vocab_size=248320, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, tie_word_embeddings=False,
                         layer_types=["linear_attention", "full_attention",
                                      "linear_attention", "full_attention"],
                         linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
                         linear_num_key_heads=2, linear_num_value_heads=4,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                                          "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1],
                                          "mrope_interleaved": True},
                         rms_norm_eps=1e-6, max_position_embeddings=4096),
        vision_config=v, tie_word_embeddings=False)
    return Qwen3_5ForConditionalGeneration, cfg, (96, 160)


def glm4vmoe():
    from transformers import Glm4vMoeConfig, Glm4vMoeForConditionalGeneration
    # GLM-4.5V / 4.6V at toy size: glm4v's tower in front of GLM-4.5's
    # mixture (a dense lead block, then four experts, two routed, a shared
    # one, sigmoid with the selection-only bias, a routed scale) under the
    # same M-RoPE on half of each head.
    v = glm4v()[1].vision_config.to_dict()
    cfg = Glm4vMoeConfig(
        text_config=dict(vocab_size=151552, hidden_size=96, intermediate_size=192,
                         num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=1, head_dim=32,
                         rms_norm_eps=1e-5, max_position_embeddings=4096, attention_bias=True,
                         moe_intermediate_size=32, num_experts_per_tok=2, n_shared_experts=1,
                         n_routed_experts=4, routed_scaling_factor=2.5, first_k_dense_replace=1,
                         rope_parameters={"rope_type": "default", "rope_theta": 500000.0,
                                          "mrope_section": [2, 3, 3], "partial_rotary_factor": 0.5},
                         tie_word_embeddings=False),
        vision_config=v, tie_word_embeddings=False)
    return Glm4vMoeForConditionalGeneration, cfg, (84, 140)


SPECS = {
    "synth-glm4vmoe": glm4vmoe,
    "synth-qwen35vl": qwen35vl,
    "synth-glm4v": glm4v,
    "synth-qwen2vl": qwen2vl,
    "synth-qwen25vl": qwen25vl,
    "synth-qwen3vl": qwen3vl,
    "synth-qwen3vlmoe": qwen3vlmoe,
}

# The Qwen3-VL fixtures take their tokenizer and processor from a Qwen3-VL
# checkpoint (patch 16, mean and std 0.5), and their processor golden is cut
# at 15 merge units of 32 pixels.
QVL3_TOK = refenv.under("QVL3_TOK", "qvl3", "hf", "Qwen3-VL-2B-Instruct")
QVL3_PREP_MAX = 32 * 32 * 15
QVL35_TOK = refenv.under("QVL35_TOK", "qvl3", "hf", "Qwen3.5-0.8B")
# The GLM-4.xV fixtures take theirs from GLM-4.1V.
GLMV_TOK = refenv.under("GLMV_TOK", "glmv", "hf", "GLM-4.1V-9B-Thinking")


def perturb(m, qkv=0.3, vis=0.2, txt=0.2):
    # Norm weights away from one and every bias away from zero, so a missing
    # norm or bias moves the answer (transformers initialises both to the
    # identity). See c6gold.py's perturb for why the module name is the test.
    with torch.no_grad():
        for n, p in m.named_parameters():
            mod = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and ("norm" in mod or mod.startswith("ln")):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif "pos_embed" in n or "position_embedding" in n:
                # Qwen3-VL's position table at the patch embedding's scale
                # (whose 768-wide rows sum to a few units each), so a table
                # read at the wrong place moves the tower.
                p.copy_(torch.randn_like(p) * 4)
            elif n.startswith("model.visual") and ".qkv." in n:
                # Sharper tower attention, within reason. At the 0.2 every
                # other weight takes, dropping the 2-D rotary moves the
                # tower's output by 5% (NMSE); at 0.3 by 16-21% and full
                # attention for windowed by 31%, while the Q8_0 weights and
                # int8 activations the engine runs move it by 2-5e-4; at 0.5
                # the attention is so sharp that the int8 alone moves it by
                # 1e-2 and no tight gate can be held.
                p.copy_(torch.randn_like(p) * qkv)
            elif n.startswith("model.visual"):
                p.copy_(torch.randn_like(p) * vis)
            else:
                p.copy_(torch.randn_like(p) * txt)
        # A mixture's selection-only bias is a buffer, which named_parameters
        # does not reach; at zero it is the identity and nothing could see it.
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.3)


def picture(h, w):
    # Smooth gradients and a disc: not a uniform image, whose patches are all
    # equal and make every attention row the same (model-correctness.md).
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * (0.5 + 0.5 * np.sin(x / 9.0))
    g = 255 * (y / h)
    b = 255 * (((x - 70) ** 2 + (y - 40) ** 2) < 30 ** 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def texture(h, w):
    # Stripes a few pixels wide under a disc: content the filters disagree on
    # when they reduce it, which a smooth gradient is not.
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * (0.5 + 0.5 * np.sin(x / 1.7))
    g = 255 * (0.5 + 0.5 * np.cos((x + y) / 2.3))
    b = 255 * (((x - 120) ** 2 + (y - 80) ** 2) < 50 ** 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def q8rows(w):
    # A tower matrix as the converter stores it: Q8_0 along k, ggml's recipe
    # (d = amax/127 stored as f16, codes rounded from the unrounded 1/d), k
    # zero-padded to a whole block first as convert.reshapePatchEmbd and
    # convert.padTowerFFN pad it. The padding is exact, so the result is cut
    # back to the original k.
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = torch.nn.functional.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    c = torch.round(x * idd)
    y = (c * d.half().float()).reshape(-1, kp)[:, :k]
    return y.reshape(w.shape)


def quantize_tower(m):
    # Every matrix the engine's tower multiplies by, at the converter's Q8_0;
    # the patch embedding's two temporal planes folded into one first (a still
    # image feeds both, convert.foldPatchPlanes), so its quantization is the
    # folded matrix's.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not n.startswith("model.visual") or not n.endswith("weight"):
                continue
            # Qwen3-VL's learned position table is a lookup, kept exact.
            if "pos_embed" in n:
                continue
            if "patch_embed" in n:
                o = p.shape[0]
                folded = p[:, :, 0] + p[:, :, 1]  # (o, c, ph, pw)
                qf = q8rows(folded.reshape(o, -1)).reshape(folded.shape)
                p[:, :, 0] = qf
                p[:, :, 1] = 0
            elif "downsample" in n:
                # GLM's 2x2 merging convolution, which the converter stores as a
                # matrix over the (dy, dx, channel) row and quantizes along it.
                w = p.permute(0, 2, 3, 1)  # (o, ky, kx, c)
                p.copy_(q8rows(w.reshape(w.shape[0], -1)).reshape(w.shape).permute(0, 3, 1, 2))
            elif p.ndim == 2:
                p.copy_(q8rows(p))


def qact(x):
    # An activation as the engine's packed matmul reads it: int8 codes per
    # block of 32 along k, d = amax/127.
    k = x.shape[-1]
    kp = (k + 31) // 32 * 32
    y = torch.nn.functional.pad(x.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = y.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(y * idd) * d).reshape(-1, kp)[:, :k].reshape(x.shape)


def image_features_a8(m, pv, thw):
    # The tower with every Linear's input quantized as qact does: what the
    # engine's int8-activation matmuls add to the Q8_0 weights.
    hs = [mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),))
          for n, mod in m.named_modules()
          if n.startswith("model.visual") and isinstance(mod, torch.nn.Linear)]
    # GLM's merging convolution reads each merge group's (dy, dx, channel) row.
    def qgroup(mod, inp):
        x = inp[0].permute(0, 2, 3, 1)  # (N, dy, dx, c)
        return (qact(x.reshape(x.shape[0], -1)).reshape(x.shape).permute(0, 3, 1, 2),)
    hs += [mod.register_forward_pre_hook(qgroup) for n, mod in m.named_modules()
           if n.startswith("model.visual") and n.endswith("downsample")]
    try:
        return image_features_deep(m, pv, thw)
    finally:
        for h in hs:
            h.remove()


def image_features(m, pv, thw):
    return image_features_deep(m, pv, thw)[0]


def image_features_deep(m, pv, thw):
    # The merged rows, and Qwen3-VL's deepstack rows -- [taps, rows, width],
    # one merger per tapped block, None for a tower without them.
    with torch.no_grad():
        out = m.model.get_image_features(pv, thw)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), 0)
    deep = getattr(out, "deepstack_features", None)
    if deep:
        deep = torch.stack(list(deep), 0)
    return feats, deep


def flat(x):
    return [] if x is None else [round(float(v), 8) for v in x.reshape(-1)]


def b64(img):
    # The picture's bytes, base64: what encoding/json reads into a []byte.
    import base64
    return base64.b64encode(img.reshape(-1).tobytes()).decode()


def unpatch(pv, gh, gw, merge, patch):
    # pixel_values back to [y][x][channel]: the processor lays each patch out
    # (channel, temporal, py, px), the patches merge-block-major (gh/m, gw/m,
    # m, m), and a still image's two temporal planes are one picture.
    p = pv.reshape(gh // merge, gw // merge, merge, merge, 3, 2, patch, patch)[:, :, :, :, :, 0]
    # (bh, bw, mh, mw, c, py, px) -> (bh, mh, py, bw, mw, px, c)
    p = p.permute(0, 2, 5, 1, 3, 6, 4)
    return p.reshape(gh * patch, gw * patch, 3)


def experts_as_published(d):
    # transformers 5 saves a Qwen3-VL mixture's packed experts as [E, out,
    # in]; the published checkpoints -- and so llama.cpp's converter, which
    # transposes them -- are [E, in, out]. The file is rewritten in the
    # published layout so the converter reads what a real checkpoint gives it.
    from safetensors.torch import load_file, save_file
    f = os.path.join(d, "model.safetensors")
    sd = load_file(f)
    for k in list(sd):
        if k.endswith("mlp.experts.down_proj") or k.endswith("mlp.experts.gate_up_proj"):
            sd[k] = sd[k].transpose(1, 2).contiguous()
    save_file(sd, f, metadata={"format": "pt"})


def build(name):
    from transformers import AutoTokenizer, Qwen2VLImageProcessorPil
    cls, cfg, (img_h, img_w) = SPECS[name]()
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    torch.manual_seed(20261003)
    m = cls(cfg).float().eval()
    # GLM's tower and text model amplify the engine's int8 several times
    # more than the Qwen fixtures' at the same weights (each tower block alone
    # matches the int8 arithmetic to 1e-13), so their weights are softer.
    if name.startswith("synth-glm"):
        perturb(m, qkv=0.2, vis=0.1, txt=0.1)
    elif name.startswith("synth-qwen35"):
        # Qwen3.5's hybrid text model likewise turns the tower's int8 into an
        # argmax flip at the Qwen fixtures' text weights.
        perturb(m, txt=0.1)
    else:
        perturb(m)
    d = os.path.join(HFDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    if "vlmoe" in name:
        experts_as_published(d)
    glm = name.startswith("synth-glm")
    tokdir, prep_max = TOK, PREP_MAX
    if name.startswith("synth-qwen35"):
        tokdir, prep_max = QVL35_TOK, QVL3_PREP_MAX
    elif name.startswith("synth-qwen3"):
        tokdir, prep_max = QVL3_TOK, QVL3_PREP_MAX
    elif glm:
        tokdir = GLMV_TOK
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "preprocessor_config.json", "chat_template.json", "chat_template.jinja",
              "generation_config.json"):
        if os.path.exists(os.path.join(tokdir, f)):
            shutil.copy(os.path.join(tokdir, f), d)
    for kind, dst in (("", name + ".gguf"), ("--mmproj", "mmproj-" + name + ".gguf")):
        args = [PY, os.path.join(refenv.lcpp_src(), "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(MODELS, dst)]
        if kind:
            args.append(kind)
        elif name.startswith("synth-qwen35"):
            # Qwen3.5 carries no prediction block here, and llama.cpp's
            # converter asks for one unless told there is none.
            args.append("--no-mtp")
        subprocess.run(args, check=True)

    # The PIL processor: transformers' default backend is torchvision's, which
    # this venv does not carry, and the image is sized so neither resizes.
    if glm:
        from transformers import Glm4vImageProcessorPil as procls
    else:
        procls = Qwen2VLImageProcessorPil
    improc = procls.from_pretrained(d)
    tok = AutoTokenizer.from_pretrained(d)
    img = picture(img_h, img_w)
    pv = improc(images=[img], return_tensors="pt", do_resize=False)
    thw = pv["image_grid_thw"]
    merge = cfg.vision_config.spatial_merge_size
    gt, gh, gw = [int(v) for v in thw[0]]
    nimg = gt * gh * gw // (merge * merge)
    if glm:
        pre = tok.encode("[gMASK]<sop><|user|>\n", add_special_tokens=False) + [cfg.image_start_token_id]
        post = [cfg.image_end_token_id] + tok.encode(
            "What is in the picture?<|assistant|>\n", add_special_tokens=False)
    else:
        pre = tok.encode("<|im_start|>user\n", add_special_tokens=False) + [cfg.vision_start_token_id]
        post = [cfg.vision_end_token_id] + tok.encode(
            "What is in the picture?<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    cont = tok.encode(" A disc on a gradient.", add_special_tokens=False)
    ids = pre + [cfg.image_token_id] * nimg + post + cont
    types = [0] * len(pre) + [1] * nimg + [0] * (len(post) + len(cont))
    t_ids = torch.tensor([ids])
    t_types = torch.tensor([types], dtype=torch.int32)
    feats, deep = image_features_deep(m, pv["pixel_values"], thw)
    with torch.no_grad():
        pos, delta = m.model.get_rope_index(t_ids, t_types, image_grid_thw=thw)
        out = m(input_ids=t_ids, pixel_values=pv["pixel_values"], image_grid_thw=thw,
                mm_token_type_ids=t_types)
    lg = out.logits[0]
    # The tower again with its matrices at the converter's Q8_0, which is
    # what separates the engine's tower from transformers' before any of the
    # engine's own arithmetic: the gate holds the tower to this one.
    import copy
    mq = copy.deepcopy(m)
    quantize_tower(mq)
    feats_q8, deep_q8 = image_features_deep(mq, pv["pixel_values"], thw)
    feats_a8, deep_a8 = image_features_a8(mq, pv["pixel_values"], thw)
    first = len(pre) + nimg + len(post) - 1  # the prompt's last row
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]
    g = {
        "pre": pre, "post": post, "cont": cont,
        "image": {"h": img_h, "w": img_w, "rgb": b64(img),
                  "grid": [gt, gh // merge, gw // merge],
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_q8": [round(float(x), 8) for x in feats_q8.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)],
                  # Qwen3-VL's deepstack rows, [taps][rows][width]: what each
                  # text layer from 0 adds to the image's rows.
                  "deep": flat(deep), "deep_q8": flat(deep_q8), "deep_a8": flat(deep_a8)},
        "pos": pos[:, 0, :].T.tolist(),
        "delta": int(delta.reshape(-1)[0]),
        "logits": rows,
    }
    # The processor's resize, on a picture it must resize: its own smart_resize
    # and PIL resample, then rescale and normalise.
    pimg = texture(PREP_H, PREP_W)
    prep_min = 56 * 56
    if glm:
        # GLM's bounds are on the picture's two temporal frames: 25 merge
        # units, under its own floor.
        prep_min, prep_max = 112 * 112, 2 * 28 * 28 * 25
        ppv = improc(images=[pimg], return_tensors="pt",
                     size={"shortest_edge": prep_min, "longest_edge": prep_max})
    else:
        ppv = improc(images=[pimg], return_tensors="pt", min_pixels=prep_min, max_pixels=prep_max)
    _, ph, pw = [int(v) for v in ppv["image_grid_thw"][0]]
    px = unpatch(ppv["pixel_values"], ph, pw, merge, cfg.vision_config.patch_size)
    # The bounds as the tower takes them: per frame, where GLM's processor
    # bounds a still image's two frames together.
    frames = 2 if glm else 1
    g["prep"] = {"h": PREP_H, "w": PREP_W, "min": prep_min // frames, "max": prep_max // frames,
                 "rgb": b64(pimg),
                 "out_h": ph * cfg.vision_config.patch_size, "out_w": pw * cfg.vision_config.patch_size,
                 "px": [round(float(x), 5) for x in px.reshape(-1)]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{name}: {len(ids)} rows, image {gh // merge}x{gw // merge} merged, "
          f"rope delta {g['delta']}, {len(rows)} logit rows")


if __name__ == "__main__":
    for n in sys.argv[1:] or SPECS:
        build(n)
