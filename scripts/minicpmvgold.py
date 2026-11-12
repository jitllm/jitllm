#!/usr/bin/env python3
"""MiniCPM-V's own processor, tower and resampler, run on pictures. -> DIR/gold/

    $JITLLM_HF_PY scripts/minicpmvgold.py DIR IMAGE...

DIR holds the checkpoint's remote code and configs (config.json,
preprocessor_config.json, image_processing_minicpmv.py, modeling_navit_siglip.py,
resampler.py, configuration_minicpm.py) and vision.safetensors, the vpm.* and
resampler.* tensors cut out of the checkpoint. Nothing here is llama.cpp's: this
is the reference the weights were trained against (RULE 7m), run in float32 on
the CPU.

For each picture it writes NAME.json (the layout: every piece's pixel size,
patch grid and position ids) and, per piece i, raw little-endian float32:

    NAME.pN.px     the normalised piece, [y][x][channel]
    NAME.pN.vit    the tower's output after its post-LayerNorm, [patches][1152]
    NAME.pN.emb    the resampler's output, [queries][d]

The test reads them with JITLLM_MINICPMV_GOLD pointing at DIR/gold.
"""
import json, os, sys, types

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def load(d):
    sys.path.insert(0, d)
    # The remote code imports itself as a package; give it one.
    pkg = types.ModuleType("mcpv")
    pkg.__path__ = [d]
    sys.modules["mcpv"] = pkg
    import importlib
    conf = importlib.import_module("mcpv.configuration_minicpm")
    siglip = importlib.import_module("mcpv.modeling_navit_siglip")
    rs = importlib.import_module("mcpv.resampler")
    cfg = conf.MiniCPMVConfig.from_pretrained(d)
    cfg.vision_config._attn_implementation = "eager"
    vpm = siglip.SiglipVisionTransformer(cfg.vision_config)
    if cfg.drop_vision_last_layer:
        vpm.encoder.layers = vpm.encoder.layers[:-1]
    res = rs.Resampler(num_queries=cfg.query_num, embed_dim=cfg.hidden_size,
                       num_heads=cfg.hidden_size // 128, kv_dim=cfg.vision_config.hidden_size,
                       adaptive=True)
    sd = load_file(os.path.join(d, "vision.safetensors"))
    sd = {k: v.float() for k, v in sd.items()}
    vpm.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vpm.")}, strict=True)
    res.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("resampler.")}, strict=False)
    vpm.float().eval()
    res.float().eval()
    return cfg, vpm, res, processor(d)


def processor(d):
    sys.path.insert(0, d)
    pkg = types.ModuleType("mcpv")
    pkg.__path__ = [d]
    sys.modules.setdefault("mcpv", pkg)
    import importlib
    ip = importlib.import_module("mcpv.image_processing_minicpmv")
    pc = json.load(open(os.path.join(d, "preprocessor_config.json")))
    skip = ("auto_map", "image_processor_type", "processor_class")
    return ip.MiniCPMVImageProcessor(**{k: v for k, v in pc.items() if k not in skip})


def layouts(d):
    """The processor's cut of many picture sizes, from its own methods, into
    testdata/golden/minicpmv-layout.json: no pixels and no weights, so the
    test that reads it needs no model."""
    proc = processor(d)
    sizes = [(384, 384), (448, 448), (500, 300), (300, 500), (640, 488), (896, 448), (448, 896),
             (1000, 700), (1024, 768), (1920, 1080), (1080, 1920), (3000, 400), (400, 3000),
             (14, 6000), (6000, 14), (1344, 1344), (2000, 2000), (700, 699), (449, 448),
             (100, 100), (28, 2000), (1500, 900), (4000, 3000), (813, 1217)]
    rows = []
    for w, h in sizes:
        grid = proc.get_sliced_grid((w, h), proc.max_slice_nums)
        if grid is None:
            ov = proc.find_best_resize((w, h), proc.scale_resolution, proc.patch_size, allow_upscale=True)
            ref = None
        else:
            ov = proc.find_best_resize((w, h), proc.scale_resolution, proc.patch_size)
            ref = proc.get_refine_size((w, h), grid, proc.scale_resolution, proc.patch_size, allow_upscale=True)
        rows.append({"w": w, "h": h, "overview": list(ov), "grid": grid, "refine": list(ref) if ref else None})
    # The tower's position buckets on one axis of n patches, as
    # modeling_navit_siglip.py computes them in float32 (B = 70).
    B = 70
    bounds = torch.arange(1 / B, 1.0, 1 / B)
    # Stored as the places where float32 disagrees with the rational
    # floor(B*i/n) -- llama.cpp's formula -- which is what makes them worth a
    # golden; every other index is that floor.
    buckets = {}
    for n in range(1, 1500):
        frac = torch.arange(0, 1 - 1e-6, 1 / torch.tensor(n))
        got = torch.bucketize(frac, bounds, right=True).tolist()
        assert len(got) == n
        buckets[n] = [[i, b] for i, b in enumerate(got) if b != (B * i) // n]
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "..", "testdata", "golden", "minicpmv-layout.json")
    json.dump({"layouts": rows, "buckets": buckets}, open(dst, "w"))
    print("wrote", dst, len(rows), "sizes")


PIL_SIZES = [(448, 448), (200, 150), (300, 620), (523, 523), (384, 97)]


def pil(src):
    """Pillow's BICUBIC resize of one picture at a few sizes, up and down and
    unequal per axis, as PNGs under testdata/golden/minicpmv-pil/: 8-bit
    samples, which is what Pillow stores and what the Go resampler must match."""
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "..", "testdata", "golden", "minicpmv-pil")
    os.makedirs(dst, exist_ok=True)
    img = Image.open(src).convert("RGB")
    for w, h in PIL_SIZES:
        img.resize((w, h), resample=Image.Resampling.BICUBIC).save(os.path.join(dst, f"{w}x{h}.png"))
    print("wrote", len(PIL_SIZES), "resizes of", src, "to", dst)


# The fixture: MiniCPM-V 4's own remote code (MiniCPMV: a Llama text model
# under LongRoPE, the SigLIP-NaViT tower, the perceiver resampler) at toy
# size, every parameter randomised and rounded to bfloat16 -- llama.cpp's
# minicpmv-surgery.py loads the checkpoint in bfloat16, so the GGUFs its
# converters write hold the reference's weights exactly. The LongRoPE
# factors are resized to the toy head and kept far from one, so a reader
# that drops them parts from the reference. The golden is the end-to-end
# record: the prompt around one unsliced picture, the resampler's rows (f32
# and at the engine's arithmetic) and the reference's logits after it.

FIX = "synth-minicpmv"
NLOGIT = 512
IMG_W, IMG_H = 168, 126
SHORT = [1.0, 1.3, 1.7, 2.2, 2.9, 3.8, 5.0, 6.5]


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


def q8rows(w):
    # A matrix as the converter stores it: Q8_0 along k (d = amax/127 as f16).
    k = w.shape[-1]
    kp = (k + 31) // 32 * 32
    x = torch.nn.functional.pad(w.reshape(-1, k), (0, kp - k)).reshape(-1, 32)
    d = x.abs().amax(1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round(x * idd) * d.half().float()).reshape(-1, kp)[:, :k].reshape(w.shape)


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            last = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and ("norm" in last or last.startswith("ln")):
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            elif n.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.3)
            elif n.endswith("query") or "position_embedding" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
            p.copy_(p.to(torch.bfloat16).float())


def fixture():
    import base64, copy, importlib, shutil, subprocess
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import refenv
    models = refenv.models()
    src = refenv.under("MINICPMV_HF", "minicpmv", "hf", "MiniCPM-V-4")
    hf_dir = refenv.under("MINICPMV_FIX_HF", "vlmb", "fixtures")
    lcpp = refenv.lcpp_src()
    d = os.path.join(hf_dir, FIX)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for f in os.listdir(src):
        if f.endswith(".py") or f in ("config.json", "preprocessor_config.json", "generation_config.json",
                                      "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"):
            shutil.copy(os.path.join(src, f), d)
    sys.path.insert(0, d)
    pkg = types.ModuleType("mcpv")
    pkg.__path__ = [d]
    sys.modules["mcpv"] = pkg
    conf = importlib.import_module("mcpv.configuration_minicpm")
    mod = importlib.import_module("mcpv.modeling_minicpmv")
    cfg = conf.MiniCPMVConfig.from_pretrained(d)
    # The text model 256 wide (the resampler's heads are its width over
    # 128), 16 heads of 16 under the toy factors; the tower two blocks of
    # 64; 16 queries.
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 256, 384, 2
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 16, 2, 16
    cfg.query_num = 16
    rs = dict(cfg.rope_scaling)
    rs["short_factor"], rs["long_factor"] = list(SHORT), list(SHORT)
    cfg.rope_scaling = rs
    vc = cfg.vision_config
    vc.hidden_size, vc.intermediate_size, vc.num_hidden_layers, vc.num_attention_heads = 64, 96, 2, 4
    cfg._attn_implementation = "eager"
    vc._attn_implementation = "eager"
    torch.manual_seed(20261111)
    m = mod.MiniCPMV(cfg).float().eval()
    perturb(m)
    m.save_pretrained(d, safe_serialization=True)
    pc = json.load(open(os.path.join(src, "preprocessor_config.json")))
    pc["image_feature_size"] = cfg.query_num
    json.dump(pc, open(os.path.join(d, "preprocessor_config.json"), "w"), indent=1)

    env = {**os.environ, "PYTHONPATH": os.path.join(lcpp, "gguf-py")}
    legacy = os.path.join(lcpp, "tools", "mtmd", "legacy-models")
    # minicpmv-surgery.py's split, done here: its AutoModel load of the remote
    # class fails on this transformers (no all_tied_weights_keys), and the
    # split is all it does -- the resampler's tensors, the tower's without
    # their "vpm." and the text model saved on its own with the tokenizer.
    sd = m.state_dict()
    torch.save({k: v.float() for k, v in sd.items() if k.startswith("resampler")}, os.path.join(d, "minicpmv.projector"))
    torch.save({k.replace("vpm.", ""): v.float() for k, v in sd.items() if k.startswith("vpm")}, os.path.join(d, "minicpmv.clip"))
    m.llm.save_pretrained(os.path.join(d, "model"), safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(d, trust_remote_code=True).save_pretrained(os.path.join(d, "model"))
    subprocess.run([sys.executable, os.path.join(legacy, "minicpmv-convert-image-encoder-to-gguf.py"), "-m", d,
                    "--minicpmv-projector", os.path.join(d, "minicpmv.projector"), "--output-dir", d,
                    "--minicpmv_version", "5", "--use-f32"], check=True, env=env)
    mm = [f for f in os.listdir(d) if f.startswith("mmproj") and f.endswith(".gguf")]
    if len(mm) != 1:
        sys.exit(f"the image encoder converter wrote {mm}")
    shutil.copy(os.path.join(d, mm[0]), os.path.join(models, "mmproj-" + FIX + ".gguf"))
    subprocess.run([sys.executable, os.path.join(lcpp, "convert_hf_to_gguf.py"), os.path.join(d, "model"),
                    "--outtype", "f32", "--outfile", os.path.join(models, FIX + ".gguf")], check=True, env=env)

    proc_mod = importlib.import_module("mcpv.processing_minicpmv")
    ip = processor(d)
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    # This transformers loads the tokenizer without the picture's markers as
    # single tokens, so the processor finds no picture run; they are in the
    # vocabulary at their ids (as the GGUF holds them), registered here.
    tok.add_special_tokens({"additional_special_tokens": ["<image>", "</image>", "<image_id>", "</image_id>", "<slice>", "</slice>"]})
    proc = proc_mod.MiniCPMVProcessor(image_processor=ip, tokenizer=tok)
    y, x = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    rgb = np.stack([255 * (0.5 + 0.5 * np.sin(x / 9.0)), 255 * (y / IMG_H),
                    255 * (((x - 70) ** 2 + (y - 50) ** 2) < 30 ** 2)], -1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    if ip.get_sliced_grid(img.size, ip.max_slice_nums) is not None:
        sys.exit("the fixture's picture is sliced")
    prompt = "<|im_start|>user\n(<image>./</image>)\nWhat is in the picture?<|im_end|>\n<|im_start|>assistant\n"
    answer = " A disc on stripes."
    batch = proc([prompt + answer], [[img]], max_slice_nums=ip.max_slice_nums, use_image_id=True, return_tensors="pt")
    ids = batch["input_ids"][0]
    bound = batch["image_bound"][0]
    if len(bound) != 1:
        sys.exit(f"the prompt holds {len(bound)} picture runs")
    lo, hi = int(bound[0][0]), int(bound[0][1])
    cont = tok.encode(answer, add_special_tokens=False)
    if ids[-len(cont):].tolist() != cont:
        sys.exit("the answer does not tokenize alone as it does after the prompt")
    data = {"input_ids": batch["input_ids"], "pixel_values": batch["pixel_values"],
            "tgt_sizes": batch["tgt_sizes"], "image_bound": batch["image_bound"]}

    def rows(model, a8):
        hs = []
        if a8:
            for n, mod_ in model.named_modules():
                if not n.startswith(("vpm", "resampler")):
                    continue
                if isinstance(mod_, torch.nn.Conv2d):
                    hs.append(mod_.register_forward_pre_hook(lambda m_, inp: (qpatches(inp[0], ip.patch_size),)))
                elif isinstance(mod_, torch.nn.Linear):
                    hs.append(mod_.register_forward_pre_hook(lambda m_, inp: (qact(inp[0]),)))
        try:
            with torch.no_grad():
                emb, _ = model.get_vllm_embedding(data)
        finally:
            for h in hs:
                h.remove()
        return emb

    with torch.no_grad():
        emb = rows(m, False)
        lg = m.llm(inputs_embeds=emb).logits[0]
        mq = copy.deepcopy(m)
        for n, p in mq.named_parameters():
            if n.startswith(("vpm", "resampler")) and n.endswith("weight") and p.ndim == 2:
                p.copy_(q8rows(p))
        emb8 = rows(mq, True)
    first = len(ids) - len(cont) - 1
    g = {
        "pre": ids[:lo].tolist(), "post": ids[hi:len(ids) - len(cont)].tolist(), "cont": cont,
        "image": {"h": IMG_H, "w": IMG_W, "rgb": base64.b64encode(rgb.reshape(-1).tobytes()).decode(),
                  "embd": [round(float(v), 8) for v in emb[0, lo:hi].reshape(-1)],
                  "embd_a8": [round(float(v), 8) for v in emb8[0, lo:hi].reshape(-1)]},
        "logits": [{"argmax": int(r.argmax()), "head": [round(float(v), 7) for v in r[:NLOGIT]]}
                   for r in lg[first:]],
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata", "golden", "minicpmv")
    os.makedirs(out, exist_ok=True)
    json.dump(g, open(os.path.join(out, FIX + ".json"), "w"))
    print(f"{FIX}: {len(ids)} ids, the picture {hi - lo} rows at [{lo}, {hi}), {len(g['logits'])} logit rows")


def main():
    if sys.argv[1:] == ["--fixture"]:
        return fixture()
    if sys.argv[2:] == ["--layouts"]:
        return layouts(sys.argv[1])
    if sys.argv[1] == "--pil":
        return pil(sys.argv[2])
    d, images = sys.argv[1], sys.argv[2:]
    out = os.path.join(d, "gold")
    os.makedirs(out, exist_ok=True)
    cfg, vpm, res, proc = load(d)
    P = proc.patch_size
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        pieces = proc.get_sliced_images(img, proc.max_slice_nums)
        grid = proc.get_sliced_grid(img.size, proc.max_slice_nums)
        man = {"w": img.size[0], "h": img.size[1], "grid": grid, "pieces": []}
        for i, piece in enumerate(pieces):
            arr = np.array(piece).astype(np.float32) / 255
            arr = proc.normalize(arr, mean=proc.mean, std=proc.std, input_data_format="channels_last")
            arr = np.ascontiguousarray(arr.astype(np.float32))
            h, w = arr.shape[0], arr.shape[1]
            gh, gw = h // P, w // P
            chw = np.transpose(arr, (2, 0, 1))
            pv = torch.from_numpy(proc.reshape_by_patch(chw)).unsqueeze(0)  # [1, 3, P, n*P]
            tgt = torch.tensor([[gh, gw]], dtype=torch.int32)
            mask = torch.ones((1, 1, gh * gw), dtype=torch.bool)
            with torch.no_grad():
                vit = vpm(pv, patch_attention_mask=mask, tgt_sizes=tgt).last_hidden_state
                rso = res(vit, tgt)
            # The tower's position ids, as its embeddings computed them.
            B = vpm.embeddings.num_patches_per_side
            bounds = torch.arange(1 / B, 1.0, 1 / B)
            fh = torch.arange(0, 1 - 1e-6, 1 / tgt[0][0])
            fw = torch.arange(0, 1 - 1e-6, 1 / tgt[0][1])
            bh = torch.bucketize(fh, bounds, right=True)
            bw = torch.bucketize(fw, bounds, right=True)
            pos = (bh[:, None] * B + bw).flatten().tolist()
            base = os.path.join(out, f"{name}.p{i}")
            arr.tofile(base + ".px")
            vit[0].numpy().astype(np.float32).tofile(base + ".vit")
            rso[0].numpy().astype(np.float32).tofile(base + ".emb")
            man["pieces"].append({"w": w, "h": h, "gh": gh, "gw": gw, "pos": pos,
                                  "vit_sum": float(vit.sum()), "emb_sum": float(rso.sum())})
            print(f"{name} piece {i}: {w}x{h} ({gw}x{gh} patches) emb {tuple(rso.shape)} sum {float(rso.sum()):.4f}")
        json.dump(man, open(os.path.join(out, name + ".json"), "w"))


if __name__ == "__main__":
    main()
