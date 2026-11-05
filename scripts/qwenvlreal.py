#!/usr/bin/env python3
"""Record transformers' answer for a real Qwen-VL checkpoint on a real picture.

    $JITLLM_HF_PY scripts/qwenvlreal.py <hf dir> <name> [image]

-> testdata/golden/qwenvl/real-<name>.json, and with QVL_GGUF=1 also
   $JITLLM_MODELS/<name>-f16.gguf and mmproj-<name>-f16.gguf (llama.cpp's own
   converter, F16), the files the engine is compared on.

The prompt is the model's own chat template around one picture
(engine/testdata/quad-and-disc.png by default) and a question; the
continuation is transformers' own greedy answer, which the engine is then
teacher-forced through. The golden carries:

    ids      the prompt, with the image's <|image_pad|> run
    image    the image's merged grid and transformers' tower output for it,
             which the text gate splices in so the text model is pinned alone
    pos      every row's (t, h, w) from get_rope_index
    logits   argmax and the first NLOGIT values at the prompt's last row and
             after each continuation token

Run on the CPU in float32: the reference is the arithmetic, not a fast path.
QVL_DTYPE=bfloat16 runs it in bfloat16 instead, for a checkpoint whose float32
copy does not fit the box (the 3B's is 15 GB); the golden records which.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
OUT = os.path.join(HERE, "..", "testdata", "golden", "qwenvl")
NLOGIT = 512
NCONT = 12


def main(hfdir, name, image):
    from PIL import Image
    from transformers import AutoConfig, AutoTokenizer, Qwen2VLImageProcessorPil
    cfg = AutoConfig.from_pretrained(hfdir)
    if cfg.model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as cls
    elif cfg.model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as cls
    elif cfg.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as cls
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    torch.set_num_threads(12)
    dtype = os.environ.get("QVL_DTYPE", "float32")
    m = cls.from_pretrained(hfdir, torch_dtype=getattr(torch, dtype), attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(hfdir)
    improc = Qwen2VLImageProcessorPil.from_pretrained(hfdir)
    img = np.array(Image.open(image).convert("RGB"))
    pv = improc(images=[img], return_tensors="pt")
    thw = pv["image_grid_thw"]
    merge = cfg.vision_config.spatial_merge_size
    gt, gh, gw = [int(v) for v in thw[0]]
    nimg = gt * gh * gw // (merge * merge)
    # The template's own rendering of one user turn with a picture.
    pre = tok.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                     "<|im_start|>user\n", add_special_tokens=False) + [cfg.vision_start_token_id]
    post = [cfg.vision_end_token_id] + tok.encode(
        "Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    prompt = pre + [cfg.image_token_id] * nimg + post
    types = [0] * len(pre) + [1] * nimg + [0] * len(post)
    with torch.no_grad():
        out = m.model.get_image_features(pv["pixel_values"], thw)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        if isinstance(feats, (list, tuple)):
            feats = torch.cat(list(feats), 0)
        # Qwen3-VL's deepstack rows, [taps][rows][width].
        deep = getattr(out, "deepstack_features", None)
        deep = torch.stack(list(deep), 0) if deep else None
        # The greedy continuation, one token at a time over the whole
        # sequence: no cache, so every position is the model from scratch.
        ids, tt = list(prompt), list(types)
        cont = []
        for _ in range(NCONT):
            t_ids = torch.tensor([ids])
            out = m(input_ids=t_ids, pixel_values=pv["pixel_values"], image_grid_thw=thw,
                    mm_token_type_ids=torch.tensor([tt], dtype=torch.int32))
            nxt = int(out.logits[0, -1].argmax())
            cont.append(nxt)
            ids.append(nxt)
            tt.append(0)
        pos, delta = m.model.get_rope_index(torch.tensor([ids]), torch.tensor([tt], dtype=torch.int32),
                                            image_grid_thw=thw)
    lg = out.logits[0]
    first = len(prompt) - 1
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 6) for x in r[:NLOGIT].float()]} for r in lg[first:]]
    g = {
        "dtype": dtype,
        "pre": pre, "post": post, "cont": cont,
        "reply": tok.decode(cont),
        "image": {"grid": [gt, gh // merge, gw // merge],
                  "embd": [round(float(x), 6) for x in feats.float().reshape(-1)],
                  "deep": [] if deep is None else [round(float(x), 6) for x in deep.float().reshape(-1)]},
        "pos": pos[:, 0, :].T.tolist(),
        "delta": int(delta.reshape(-1)[0]),
        "logits": rows[:len(cont)],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "real-" + name + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{name}: prompt {len(prompt)} rows, image {gh // merge}x{gw // merge} merged, "
          f"rope delta {g['delta']}: {g['reply']!r}")
    if os.environ.get("QVL_GGUF") == "1":
        for kind, dst in (("", name + "-f16.gguf"), ("--mmproj", "mmproj-" + name + "-f16.gguf")):
            args = [sys.executable, os.path.join(LCPP, "convert_hf_to_gguf.py"), hfdir,
                    "--outtype", "f16", "--outfile", os.path.join(MODELS, dst)]
            if kind:
                args.append(kind)
            elif cfg.model_type == "qwen3_5":
                # Its prediction layers stay out, as the fixtures' do.
                args.append("--no-mtp")
            subprocess.run(args, check=True)


if __name__ == "__main__":
    img = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "engine", "model", "testdata",
                                                               "quad-and-disc.png")
    main(sys.argv[1], sys.argv[2], img)
