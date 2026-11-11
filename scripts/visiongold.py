"""Transformers goldens for the gemma3, internvl and llava vision towers.

For each family this writes, under $JITLLM_MODELS/vlm/goldens/<family>/:

  <image>.pixels.f32    the reference processor's pixel_values for a real
                        picture, float32 [tile][channel][y][x]
  <image>.json          {"tiles": n, "size": ImageSz}
  <image>.features.f32  the tower + projector's embeddings for those pixels,
                        float32 [token][text width]
  rainbow.features.f32  the same for llama-mtmd-debug's raw "rainbow" picture,
                        which has no preprocessing in it

The tower and projector are transformers' own classes with the checkpoint's
weights (only the vision tensors are fetched: vlm/fetchst.py), run in float32
on the CPU. The pictures are PNG so the Go test decodes the same pixels PIL
does; a JPEG's decoders disagree before any of this runs.

Usage: visiongold.py gemma3|internvl|llava
"""
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

import refenv  # where things are: environment variables only (docs/testing.md)

MODELS = refenv.models()
VLM = os.path.join(MODELS, "vlm")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def rainbow(n):
    # llama-mtmd-debug's, in float32 as it computes it.
    img = np.zeros((3, n, n), np.float32)
    cx = cy = np.float32(n / 2.0)
    max_dist = np.float32(math.sqrt(cx * cx + cy * cy))
    for y in range(n):
        for x in range(n):
            dx, dy = np.float32(x) - cx, np.float32(y) - cy
            hue = np.float32(math.atan2(dy, dx)) / np.float32(2 * 3.14159265)
            if hue < 0:
                hue += np.float32(1)
            sat = np.float32(math.sqrt(dx * dx + dy * dy)) / max_dist
            sat = min(sat, np.float32(1))
            h6 = hue * np.float32(6)
            i6 = int(h6)
            f = h6 - np.float32(i6)
            p, q, t = 1 - sat, 1 - sat * f, 1 - sat * (1 - f)
            rgb = [(1, t, p), (q, 1, p), (p, 1, t), (p, q, 1), (t, p, 1), (1, p, q)][i6 % 6]
            img[:, y, x] = rgb
    return torch.from_numpy(img)[None]


def grid(n, patch):
    # Four colours in every 2x2 block of patches (red, green / blue, white),
    # dimmed by a diagonal ramp: a picture on which the order of a pixel
    # shuffle's group is the whole difference. Raw floats in [0, 1]; the Go
    # gates build the same (the engine's tower gates).
    pal = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    img = np.zeros((3, n, n), np.float32)
    for y in range(n):
        for x in range(n):
            k = ((y // patch) % 2) * 2 + (x // patch) % 2
            ramp = np.float32(0.25) + np.float32(0.75) * np.float32(x + y) / np.float32(2 * n)
            for ch in range(3):
                img[ch, y, x] = np.float32(pal[k][ch]) * ramp
    return torch.from_numpy(img)[None]


def patch_rows(n, width):
    # A tower's last hidden state that is no picture at all: every row and
    # channel distinct, so the projector alone is held -- a pool or shuffle in
    # the wrong order cannot hide behind neighbouring patches looking alike.
    # The Go gate builds the same (the engine's projector gate).
    p = np.arange(n, dtype=np.float64)[:, None]
    c = np.arange(width, dtype=np.float64)[None, :]
    return torch.from_numpy((np.sin(0.37 * p + 0.11 * c) + 0.5 * np.cos(0.013 * p * c)).astype(np.float32))


def projector(fam, m, size):
    side = size // 14
    n = side * side
    width = m.config.vision_config.hidden_size
    x = patch_rows(n, width)[None]
    with torch.no_grad():
        if fam in ("gemma3", "llava"):
            return m.multi_modal_projector(x)
        f = m.pixel_shuffle(x.reshape(1, side, side, width), scale_factor=m.config.downsample_ratio)
        return m.multi_modal_projector(f.reshape(1, -1, f.shape[-1]))


def pictures(out):
    """The real pictures, as PNG: the repo's synthetic one and a photograph."""
    os.makedirs(out, exist_ok=True)
    pics = {"quad-and-disc": os.path.join(REPO, "engine/testdata/quad-and-disc.png")}
    photo = os.path.join(out, "..", "photo.png")
    if not os.path.exists(photo):
        src = os.path.join(refenv.lcpp_src(), "tools", "mtmd", "test-1.jpeg")
        Image.open(src).convert("RGB").save(photo)
    pics["photo"] = photo
    # The photograph already at each tower's size, for the llama-mtmd-cli
    # comparison (scripts/vlmgold.py): a picture neither engine resizes.
    for n in (448, 896):
        sq = os.path.join(out, "..", "photo-%d.png" % n)
        if not os.path.exists(sq):
            Image.open(photo).convert("RGB").resize((n, n), Image.BICUBIC).save(sq)
    return pics


def save(path, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(path)


def gemma3():
    from transformers import AutoProcessor, Gemma3Config, Gemma3Model
    d = os.path.join(VLM, "hf/gemma-3-4b-it")
    cfg = Gemma3Config.from_pretrained(d)
    # The text model is not under test: one layer and a token table of a few
    # rows keep it from costing 2.7 GB of float32 nobody reads.
    cfg.text_config.num_hidden_layers = 1
    cfg.text_config.vocab_size = 512
    m = Gemma3Model(cfg).float().eval()  # the config says bfloat16; the reference is float32
    w = load_file(os.path.join(d, "vision.safetensors"))
    # The checkpoint nests the tower one module deeper than this transformers'
    # Gemma3Model does (from_pretrained renames it the same way).
    w = {k.removeprefix("model.").replace("vision_tower.vision_model.", "vision_tower."): v.float()
         for k, v in w.items()}
    missing, unexpected = m.load_state_dict(w, strict=False)
    bad = [k for k in missing if k.startswith(("vision_tower", "multi_modal_projector"))]
    assert not bad and not unexpected, (bad, unexpected)
    proc = AutoProcessor.from_pretrained(d)
    return m, proc, cfg.vision_config.image_size


def internvl():
    from transformers import AutoProcessor, InternVLConfig, InternVLModel
    d = os.path.join(VLM, "hf/InternVL3-1B-hf")
    cfg = InternVLConfig.from_pretrained(d)
    cfg.text_config.num_hidden_layers = 1
    m = InternVLModel(cfg).float().eval()
    w = load_file(os.path.join(d, "vision.safetensors"))
    w = {k.removeprefix("model."): v.float() for k, v in w.items()}
    missing, unexpected = m.load_state_dict(w, strict=False)
    bad = [k for k in missing if k.startswith(("vision_tower", "multi_modal_projector"))]
    assert not bad and not unexpected, (bad, unexpected)
    proc = AutoProcessor.from_pretrained(d)
    size = cfg.vision_config.image_size
    return m, proc, size[0] if isinstance(size, (list, tuple)) else size


def llava():
    from transformers import AutoProcessor, LlavaConfig, LlavaModel
    d = os.path.join(VLM, "hf/llava-phi-3-mini-hf")
    cfg = LlavaConfig.from_pretrained(d)
    cfg.text_config.num_hidden_layers = 1
    m = LlavaModel(cfg).float().eval()
    w = load_file(os.path.join(d, "vision.safetensors"))
    # The checkpoint nests the tower one module deeper, as Gemma 3's does.
    w = {k.removeprefix("model.").replace("vision_tower.vision_model.", "vision_tower."): v.float()
         for k, v in w.items()}
    missing, unexpected = m.load_state_dict(w, strict=False)
    bad = [k for k in missing if k.startswith(("vision_tower", "multi_modal_projector"))]
    assert not bad and not unexpected, (bad, unexpected)
    proc = AutoProcessor.from_pretrained(d)
    # The repository has no processor_config.json: what it would state, from
    # the model's own config (the class row is the one extra token).
    proc.patch_size = cfg.vision_config.patch_size
    proc.vision_feature_select_strategy = cfg.vision_feature_select_strategy
    proc.num_additional_image_tokens = 1
    return m, proc, cfg.vision_config.image_size


def features(m, px):
    with torch.no_grad():
        f = m.get_image_features(pixel_values=px)
    f = getattr(f, "pooler_output", f)
    # Llava's is a tensor per picture.
    if isinstance(f, (list, tuple)):
        f = torch.cat(list(f), 0)
    return f.reshape(-1, f.shape[-1])


def main():
    fam = sys.argv[1]
    m, proc, size = {"gemma3": gemma3, "internvl": internvl, "llava": llava}[fam]()
    out = os.path.join(VLM, "goldens", fam)
    os.makedirs(out, exist_ok=True)
    save(os.path.join(out, "rainbow.features.f32"), features(m, rainbow(size)))
    save(os.path.join(out, "grid.features.f32"), features(m, grid(size, 14)))
    save(os.path.join(out, "projector.features.f32"), projector(fam, m, size).reshape(-1, m.config.text_config.hidden_size))
    print("rainbow and grid", size)
    for name, path in pictures(out).items():
        img = Image.open(path)
        # Through the processor, not its image processor alone: the processor
        # carries InternVL's crop_to_patches default, which the image
        # processor's own config turns off.
        enc = proc(text=proc.image_token, images=[img], return_tensors="pt")
        px = enc["pixel_values"].reshape(-1, 3, size, size)
        save(os.path.join(out, name + ".pixels.f32"), px)
        save(os.path.join(out, name + ".features.f32"), features(m, px))
        # The whole chat prompt as the processor tokenizes it, for the marker
        # gate: an explicit system turn, since InternVL's GGUF carries Qwen's
        # template (whose default system turn the HF one has not).
        # Tokenized by apply_chat_template itself, which is the processor's
        # documented path: rendering to text and calling the processor on it
        # adds a second BOS to a template that writes its own (Gemma's).
        # llava-phi-3-mini-hf publishes no processor chat template: its
        # golden is the tower's alone.
        msgs = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "Describe this image in one sentence."}]}]
        chat, chat_ids = None, None
        if getattr(proc, "chat_template", None):
            chat = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            chat_ids = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump({"tiles": px.shape[0], "size": size, "w": img.size[0], "h": img.size[1],
                       "processor": type(proc.image_processor).__name__,
                       "image_token": m.config.image_token_id,
                       "chat": chat, "chat_ids": chat_ids}, f)
        print(name, tuple(px.shape), type(proc.image_processor).__name__)


if __name__ == "__main__":
    main()
