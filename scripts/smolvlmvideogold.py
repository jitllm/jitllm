#!/usr/bin/env python3
"""SmolVLM's video, from the reference itself.

    $JITLLM_HF_PY scripts/smolvlmvideogold.py

-> $JITLLM_MODELS/synth-smolvlm2.gguf and mmproj-synth-smolvlm2.gguf, and
   testdata/golden/smolvlm2/synth-smolvlm2.video.json

The fixture is transformers' own SmolVLMForConditionalGeneration at toy
size, every parameter and buffer randomised, written to GGUF by llama.cpp's
own convert_hf_to_gguf.py (the text model as the LlamaForCausalLM it is, the
tower with --mmproj); the tokenizer is SmolVLM2's (SMOLVLM2_TOK, its
tokenizer.json and configs). The processor is SmolVLMProcessor with
SmolVLMVideoProcessor at SmolVLM2's preprocessor_config.json, the sizes
scaled to the toy tower's 64-pixel square: a frame a second, at most 64,
sample_frames' linspace a second in from either end; the prompt its
introduction (the count in words, the duration as H:MM:SS), each frame's
"\nFrame from MM:SS:" and its global image, "\n\n" after the last. The video
is SEVENTY frames of 40 x 24 at a frame a second, so the sampler skips the
first and last second and truncates its linspace.

The frames are resized as the engine resizes them -- torchvision's Lanczos
straight to the square, the one pass of the picture op at torchvision's
precision -- rather than as the video processor does (torchvision's Lanczos
to the longer side, then to the square: docs/divergences.md); the golden
holds everything else to the reference, and records that divergence's own
size. It carries:

    pre_text, post_text  the text around the video, which the video's own
                         text at either end is tokenized with
    bos                  the BOS the tokenizer puts first, -1 for none
    ids, image_token     the whole prompt as the processor writes it
    cont                 the continuation
    video                the frames (RGB bytes), the rate, the frames
                         sampled, the rows a frame, and every frame's rows
                         over the engine's pixels in f32 and at the engine's
                         arithmetic (embd_a8); divergence, the NMSE of the
                         rows over the processor's own pixels from them
    logits               the prompt's last row and each continuation row's
                         over the f32 rows; logits_a8 over the int8 rows
"""
import base64
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refenv  # noqa: E402

FIX = "synth-smolvlm2"
OUT = os.path.join(HERE, "..", "testdata", "golden", "smolvlm2")
NLOGIT = 512
SIDE, PATCH, SCALE = 64, 16, 2
N, H, W, FPS = 70, 24, 40, 1.0


def frames():
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    out = []
    for f in range(N):
        r = 255 * (0.5 + 0.5 * np.sin((x + f) / 5.0))
        g = 255 * (y / H)
        b = 255 * (((x - (f * 40) // N) ** 2 + (y - 12) ** 2) < 8 ** 2)
        out.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    return np.stack(out)


def lanczos(img, n):
    """The engine's resize of an 8-bit frame to n x n: torchvision's
    antialiased Lanczos of a uint8 tensor, one pass (jlm.FilterPILLanczos at
    jlm.PrecTorch)."""
    import torchvision.transforms.v2.functional as TF
    from torchvision.transforms import InterpolationMode
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()
    out = TF.resize(t, [n, n], interpolation=InterpolationMode.LANCZOS, antialias=True)
    return out.permute(1, 2, 0).numpy().astype(np.float32)


def q8rows(w):
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = torch.nn.functional.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(x * idd) * d.half().float()).reshape(-1, kp)[:, :k].reshape(w.shape)


def qact(x):
    k = x.shape[-1]
    kp = (k + 31) // 32 * 32
    y = torch.nn.functional.pad(x.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = y.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(y * idd) * d).reshape(-1, kp)[:, :k].reshape(x.shape)


def qpatches(x, p):
    b, c, h, w = x.shape
    t = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(-1, c * p * p)
    t = qact(t).reshape(b, h // p, w // p, c, p, p).permute(0, 3, 1, 4, 2, 5)
    return t.reshape(b, c, h, w)


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in last:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif "position_embedding" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
        for n, b in m.named_buffers():
            if b.is_floating_point() and "inv_freq" not in n:
                b.copy_(torch.randn_like(b) * 0.3)


def features(m, pv, a8=False):
    """Every frame's rows (get_image_features): in f32, or at the engine's
    arithmetic (every matmul's input at int8 per block of 32, the patch
    conv's per patch)."""
    hs = []
    if a8:
        for n, mod in m.model.named_modules():
            if not n.startswith(("vision_model", "connector")):
                continue
            if isinstance(mod, torch.nn.Conv2d):
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qpatches(inp[0], PATCH),)))
            elif isinstance(mod, torch.nn.Linear):
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))
    try:
        with torch.no_grad():
            f = m.model.get_image_features(pv.unsqueeze(0))
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        return f.reshape(-1, f.shape[-1])
    finally:
        for h in hs:
            h.remove()


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    from PIL import Image  # noqa: F401
    from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM, SmolVLMConfig,
                              SmolVLMForConditionalGeneration, SmolVLMProcessor)
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig
    from transformers.models.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor
    from transformers.models.smolvlm.video_processing_smolvlm import SmolVLMVideoProcessor
    models = refenv.models()
    tok_dir = refenv.need("SMOLVLM2_TOK", "SmolVLM2's tokenizer.json and configs (HuggingFaceTB/SmolVLM2-256M-Video-Instruct)")
    lcpp = refenv.lcpp_src()
    hf_dir = refenv.under("SMOLVLM2_HF", "vlmb", "fixtures")
    torch.manual_seed(20261112)
    tc = LlamaConfig(vocab_size=49280, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-5,
                     rope_theta=100000.0, max_position_embeddings=8192, tie_word_embeddings=False)
    vc = SmolVLMVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                             image_size=SIDE, patch_size=PATCH)
    cfg = SmolVLMConfig(text_config=tc.to_dict(), vision_config=vc.to_dict(), scale_factor=SCALE,
                        image_token_id=49190, tie_word_embeddings=False)
    m = SmolVLMForConditionalGeneration(cfg).float().eval()
    perturb(m)

    pc = json.load(open(os.path.join(tok_dir, "preprocessor_config.json")))
    pc.update({"max_image_size": {"longest_edge": SIDE}, "size": {"longest_edge": 4 * SIDE},
               "video_sampling": {**pc["video_sampling"], "video_size": {"longest_edge": SIDE}}})
    d = os.path.join(hf_dir, FIX)
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        shutil.copy(os.path.join(tok_dir, f), d)
    json.dump(pc, open(os.path.join(d, "preprocessor_config.json"), "w"), indent=1)
    json.dump({"image_seq_len": (SIDE // PATCH // SCALE) ** 2, "processor_class": "SmolVLMProcessor"},
              open(os.path.join(d, "processor_config.json"), "w"))
    # llama.cpp converts SmolVLM's text model as the LlamaForCausalLM it is.
    td = d + "-text"
    shutil.rmtree(td, ignore_errors=True)
    lm = LlamaForCausalLM(m.config.text_config).float().eval()
    lm.model.load_state_dict(m.model.text_model.state_dict())
    lm.lm_head.load_state_dict(m.lm_head.state_dict())
    lm.save_pretrained(td, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        shutil.copy(os.path.join(tok_dir, f), td)
    env = {**os.environ, "PYTHONPATH": os.path.join(lcpp, "gguf-py")}
    for src, kind, dst in ((td, "", FIX + ".gguf"), (d, "--mmproj", "mmproj-" + FIX + ".gguf")):
        args = [sys.executable, os.path.join(lcpp, "convert_hf_to_gguf.py"), src, "--outtype", "f32",
                "--outfile", os.path.join(models, dst)]
        if kind:
            args.append(kind)
        subprocess.run(args, check=True, env=env)

    tok = AutoTokenizer.from_pretrained(d)
    kw = {k: v for k, v in pc.items() if k not in ("image_processor_type", "processor_class")}
    proc = SmolVLMProcessor(image_processor=SmolVLMImageProcessor(**kw), tokenizer=tok,
                            video_processor=SmolVLMVideoProcessor(**kw), image_seq_len=(SIDE // PATCH // SCALE) ** 2)
    vid = frames()
    pre_text, post_text = "<|im_start|>User: ", "Describe the video.<end_of_utterance>\nAssistant:"
    enc = proc(text=[pre_text + "<video>" + post_text], videos=[[vid]],
               video_metadata=[{"fps": FPS, "total_num_frames": N, "duration": N / FPS}], do_sample_frames=True,
               return_metadata=True, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    # The frames the processor sampled: its video processor's metadata.
    meta = proc.video_processor(videos=[vid], video_metadata=[{"fps": FPS, "total_num_frames": N, "duration": N / FPS}],
                                do_sample_frames=True, return_metadata=True, return_tensors="pt")["video_metadata"][0]
    sampled = [int(i) for i in meta.frames_indices]
    pv_ref = enc["pixel_values"][0]
    # The frames over the engine's pixels.
    eng = np.stack([(lanczos(vid[i], SIDE) / 255.0 - 0.5) / 0.5 for i in sampled]).astype(np.float32)
    pv = torch.from_numpy(eng).permute(0, 3, 1, 2).contiguous()
    feats = features(m, pv)
    ref = features(m, pv_ref)
    mq = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in mq.model.named_parameters():
            if not n.startswith(("vision_model.", "connector.")) or not n.endswith("weight"):
                continue
            if p.ndim == 4:
                p.copy_(q8rows(p.reshape(p.shape[0], -1)).reshape(p.shape))
            elif p.ndim == 2 and "position_embedding" not in n:
                p.copy_(q8rows(p))
    feats8 = features(mq, pv, a8=True)
    cont = tok.encode(" A disc moves to the right.", add_special_tokens=False)
    full = torch.tensor([ids + cont])
    slots = full[0] == cfg.image_token_id
    assert int(slots.sum()) == feats.shape[0], (int(slots.sum()), feats.shape)
    first = len(ids) - 1

    def logits(rows):
        with torch.no_grad():
            e = m.get_input_embeddings()(full)
            e[0, slots] = rows
            lg = m(inputs_embeds=e).logits[0]
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                 "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]

    bos = tok.bos_token_id if tok("x").input_ids[0] == tok.bos_token_id else -1
    g = {
        "pre_text": pre_text, "post_text": post_text, "bos": bos, "ids": ids, "image_token": cfg.image_token_id,
        "cont": cont,
        "video": {"n": N, "h": H, "w": W, "fps": FPS, "rgb": base64.b64encode(vid.reshape(-1).tobytes()).decode(),
                  "frames": sampled, "sampled": len(sampled), "soft": feats.shape[0] // len(sampled),
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats8.reshape(-1)],
                  "divergence": nmse(ref, feats)},
        "logits": logits(feats),
        "logits_a8": logits(feats8),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, FIX + ".video.json"), "w") as f:
        json.dump(g, f)
    print(f"{FIX}: {len(ids)} ids, {len(sampled)} frames {sampled[:3]}..{sampled[-2:]}, {g['video']['soft']} rows each; "
          f"bos {bos}; the processor's resize parts the rows by NMSE {g['video']['divergence']:.3e}")
    print(repr(tok.decode([i for i in ids if i != cfg.image_token_id])[:400]))


if __name__ == "__main__":
    main()
