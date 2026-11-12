#!/usr/bin/env python3
"""Janus-Pro's own processor, vision tower and aligner, run on pictures. -> DIR/gold/

    $JITLLM_HF_PY scripts/janusgold.py DIR IMAGE...
    $JITLLM_HF_PY scripts/janusgold.py --fixture
        -> $JITLLM_MODELS/synth-janus.gguf and its mmproj, and
           testdata/golden/janus/synth-janus.json (below, after main)

DIR holds deepseek-community/Janus-Pro-1B's config.json and
preprocessor_config.json and vision.safetensors, the model.vision_model.* and
model.aligner.* tensors cut out of its checkpoint (minicpmv's fetchst.py does
the cutting). transformers' own JanusVisionModel and JanusVisionAlignerMLP run
them in float32 on the CPU, and the processor is the PIL one
(JanusImageProcessorPil), which is Pillow's resize as the GGUF's preprocessing
is.

For each picture: NAME.json, and raw little-endian float32
    NAME.px    the normalised 384x384 picture, [y][x][channel]
    NAME.vit   the tower's last_hidden_state (after its post-LayerNorm), [576][1024]
    NAME.emb   the aligner's output, [576][2048]
"""
import json, os, sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import JanusConfig
from transformers.models.janus.modeling_janus import JanusVisionModel, JanusVisionAlignerMLP
from transformers.models.janus.image_processing_pil_janus import JanusImageProcessorPil


def main():
    d, images = sys.argv[1], sys.argv[2:]
    out = os.path.join(d, "gold")
    os.makedirs(out, exist_ok=True)
    cfg = JanusConfig.from_pretrained(d)
    vis = JanusVisionModel(cfg.vision_config)
    al = JanusVisionAlignerMLP(cfg.vision_config)
    sd = {k: v.float() for k, v in load_file(os.path.join(d, "vision.safetensors")).items()}
    vis.load_state_dict({k[len("model.vision_model."):]: v for k, v in sd.items()
                         if k.startswith("model.vision_model.")}, strict=True)
    al.load_state_dict({k[len("model.aligner."):]: v for k, v in sd.items()
                        if k.startswith("model.aligner.")}, strict=True)
    vis.float().eval()
    al.float().eval()
    pc = json.load(open(os.path.join(d, "preprocessor_config.json")))
    proc = JanusImageProcessorPil(**{k: v for k, v in pc.items() if k not in ("image_processor_type", "processor_class")})
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        pv = proc(img, return_tensors="pt")["pixel_values"]  # [1, 3, 384, 384]
        with torch.no_grad():
            vit = vis(pv).last_hidden_state
            emb = al(vit)
        base = os.path.join(out, name)
        np.ascontiguousarray(pv[0].permute(1, 2, 0).numpy().astype(np.float32)).tofile(base + ".px")
        vit[0].numpy().astype(np.float32).tofile(base + ".vit")
        emb[0].numpy().astype(np.float32).tofile(base + ".emb")
        json.dump({"w": img.size[0], "h": img.size[1], "vit_sum": float(vit.sum()), "emb_sum": float(emb.sum())},
                  open(base + ".json", "w"))
        print(f"{name}: {img.size} -> vit {tuple(vit.shape)} emb {tuple(emb.shape)} sum {float(emb.sum()):.4f}")


# The fixture: transformers' own JanusForConditionalGeneration at toy size,
# every parameter and buffer randomised, written to GGUF by llama.cpp's own
# convert_hf_to_gguf.py (the text model plainly and the tower with
# --mmproj); the tokenizer is Janus-Pro-1B's. Its golden is the end-to-end
# record: the prompt around one picture, the picture, the aligner's rows (f32
# and at the engine's arithmetic) and transformers' logits after the prompt.

FIX = "synth-janus"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata", "golden", "janus")
NLOGIT = 512
SIDE, PATCH = 64, 16
IMG_H, IMG_W = 40, 72


def q8rows(w):
    # A matrix as the converter stores it: Q8_0 along k (d = amax/127 as f16).
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = torch.nn.functional.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(x * idd) * d.half().float()).reshape(-1, kp)[:, :k].reshape(w.shape)


def qact(x):
    # An activation as the engine's packed matmul reads it: int8 per block of
    # 32 along k, d = amax/127.
    k = x.shape[-1]
    kp = (k + 31) // 32 * 32
    y = torch.nn.functional.pad(x.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = y.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(y * idd) * d).reshape(-1, kp)[:, :k].reshape(x.shape)


def qpatches(x, p):
    # The patch conv's input as the engine's patch matmul reads it: each
    # patch flattened channel-planar (c, ky, kx), at int8 per block of 32.
    b, c, h, w = x.shape
    t = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(-1, c * p * p)
    t = qact(t).reshape(b, h // p, w // p, c, p, p).permute(0, 3, 1, 4, 2, 5)
    return t.reshape(b, c, h, w)


def tower_params(m):
    return [(n, p) for n, p in m.model.named_parameters() if n.startswith(("vision_model.", "aligner."))]


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
    """The tower's rows and the aligner's: in float32, or at the engine's
    arithmetic (every matmul's input at int8 per block of 32, the patch
    conv's per patch)."""
    hs = []
    if a8:
        for n, mod in m.model.named_modules():
            if not n.startswith(("vision_model", "aligner")):
                continue
            if isinstance(mod, torch.nn.Conv2d):
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qpatches(inp[0], PATCH),)))
            elif isinstance(mod, torch.nn.Linear):
                hs.append(mod.register_forward_pre_hook(lambda mod, inp: (qact(inp[0]),)))
    try:
        with torch.no_grad():
            vit = m.model.vision_model(pv).last_hidden_state
            emb = m.model.aligner(vit)
    finally:
        for h in hs:
            h.remove()
    return vit[0], emb[0]


def picture(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * (0.5 + 0.5 * np.sin(x / 7.0))
    g = 255 * (y / h)
    b = 255 * (((x - 40) ** 2 + (y - 18) ** 2) < 14 ** 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def fixture():
    import base64
    import copy
    import shutil
    import subprocess
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import refenv
    from transformers import AutoTokenizer, JanusForConditionalGeneration, LlamaConfig
    from transformers.models.janus.configuration_janus import JanusVisionConfig, JanusVQVAEConfig
    models = refenv.models()
    tok_dir = refenv.under("JANUS_TOK", "minicpmv", "hf", "Janus-Pro-1B-tok")
    proc_dir = refenv.under("JANUS_HF", "minicpmv", "hf", "Janus-Pro-1B")
    hf_dir = refenv.under("JANUS_FIX_HF", "vlmb", "fixtures")
    lcpp = refenv.lcpp_src()
    torch.manual_seed(20261111)
    tc = LlamaConfig(vocab_size=102400, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                     rope_theta=10000.0, max_position_embeddings=4096, bos_token_id=100000,
                     eos_token_id=100001, pad_token_id=100002, tie_word_embeddings=False)
    vc = JanusVisionConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=SIDE,
                           patch_size=PATCH, mlp_ratio=2.0, projection_dim=tc.hidden_size, depth=2,
                           num_image_tokens=(SIDE // PATCH) ** 2)
    # The image-generation half, which nothing here runs and llama.cpp's
    # converter drops; its GroupNorms want channels in thirty-twos.
    qc = JanusVQVAEConfig(embed_dim=8, num_embeddings=64, latent_channels=32, base_channels=32,
                          channel_multiplier=[1, 1], num_res_blocks=1, projection_dim=tc.hidden_size,
                          image_token_embed_dim=tc.hidden_size, num_hidden_layers=1)
    cfg = JanusConfig(text_config=tc.to_dict(), vision_config=vc.to_dict(), vq_config=qc.to_dict(),
                      image_token_id=100581, tie_word_embeddings=False)
    m = JanusForConditionalGeneration(cfg).float().eval()
    perturb(m)

    d = os.path.join(hf_dir, FIX)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(tok_dir, f), d)
    pc = json.load(open(os.path.join(proc_dir, "preprocessor_config.json")))
    pc["size"] = {"height": SIDE, "width": SIDE}
    json.dump(pc, open(os.path.join(d, "preprocessor_config.json"), "w"), indent=1)
    for kind, dst in (("", FIX + ".gguf"), ("--mmproj", "mmproj-" + FIX + ".gguf")):
        args = [sys.executable, os.path.join(lcpp, "convert_hf_to_gguf.py"), d, "--outtype", "f32",
                "--outfile", os.path.join(models, dst)]
        if kind:
            args.append(kind)
        env = {**os.environ, "PYTHONPATH": os.path.join(lcpp, "gguf-py")}
        subprocess.run(args, check=True, env=env)

    proc = JanusImageProcessorPil(**{k: v for k, v in pc.items() if k not in ("image_processor_type", "processor_class")})
    img = picture(IMG_H, IMG_W)
    pv = proc(Image.fromarray(img), return_tensors="pt")["pixel_values"]
    _, feats = features(m, pv)
    mq = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in tower_params(mq):
            if n.endswith("weight") and p.ndim == 4:
                p.copy_(q8rows(p.reshape(p.shape[0], -1)).reshape(p.shape))
            elif n.endswith("weight") and p.ndim == 2 and "position_embedding" not in n:
                p.copy_(q8rows(p))
    _, feats8 = features(mq, pv, a8=True)

    tok = AutoTokenizer.from_pretrained(tok_dir)
    pre = tok.encode("<|User|>: <begin_of_image>", add_special_tokens=True)
    post = tok.encode("<end_of_image>\nWhat is in the picture?\n\n<|Assistant|>:", add_special_tokens=False)
    cont = tok.encode(" A disc on stripes.", add_special_tokens=False)
    emb = m.get_input_embeddings()
    first = len(pre) + feats.shape[0] + len(post) - 1

    def logits(rows):
        with torch.no_grad():
            e = torch.cat([emb(torch.tensor(pre)), rows, emb(torch.tensor(post + cont))], 0)
            lg = m(inputs_embeds=e.unsqueeze(0)).logits[0]
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                 "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg[first:]]

    g = {
        "pre": pre, "post": post, "cont": cont,
        "image": {"h": IMG_H, "w": IMG_W, "rgb": base64.b64encode(img.reshape(-1).tobytes()).decode(),
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats8.reshape(-1)]},
        "logits": logits(feats),
        "logits_a8": logits(feats8),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, FIX + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{FIX}: {IMG_W}x{IMG_H} -> {feats.shape[0]} rows, {len(pre) + feats.shape[0] + len(post) + len(cont)} positions; "
          f"pre {pre}, post {post[:3]}...")


if __name__ == "__main__":
    if sys.argv[1:] == ["--fixture"]:
        fixture()
    else:
        main()
