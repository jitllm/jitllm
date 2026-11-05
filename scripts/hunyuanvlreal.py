#!/usr/bin/env python3
"""Record transformers' answer for the real HunyuanOCR on a real picture.

    $JITLLM_HF_PY scripts/hunyuanvlreal.py [hf dir] [image]

-> testdata/golden/qwenvl/real-HunyuanOCR.json, and with HYVL_GGUF=1 also
   $JITLLM_MODELS/HunyuanOCR-f16.gguf and mmproj-HunyuanOCR-f16.gguf
   (llama.cpp's own converter, F16), the files the engine is compared on
   (TestQwenVLRealMatchesTransformers).

The hf dir defaults to $HYVL_SRC, else $JITLLM_MODELS/hunyuanvl/hf/HunyuanOCR
(tencent/HunyuanOCR). The prompt is the checkpoint's own chat template around
one picture (engine/testdata/quad-and-disc.png by default) and a
question, the pixels the PIL processor's (HunyuanOCR's own resizes with
PIL.Image.resize); the continuation is transformers' own greedy answer, which
the engine is then teacher-forced through. The golden carries the tokens
around the picture's rows, transformers' rows for it, every row's four
XD-RoPE coordinates (pos4) and the logits at the prompt's last row and after
each continuation token. Run on the CPU in float32.
"""
import json
import os
import subprocess
import sys

import torch

import refenv  # where things are: environment variables only (docs/testing.md)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
LCPP = refenv.need("LCPP_SRC", "a llama.cpp source checkout")
OUT = os.path.join(HERE, "..", "testdata", "golden", "qwenvl")
NAME = "HunyuanOCR"
NLOGIT = 512
NCONT = 12


def main(hfdir, image):
    from PIL import Image
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
    from transformers.models.hunyuan_vl.image_processing_pil_hunyuan_vl import HunYuanVLImageProcessorPil
    torch.set_num_threads(12)
    m = HunYuanVLForConditionalGeneration.from_pretrained(hfdir, dtype=torch.float32,
                                                          attn_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained(hfdir)
    proc.image_processor = HunYuanVLImageProcessorPil.from_pretrained(hfdir)
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "Describe this image in one sentence."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[Image.open(image).convert("RGB")], return_tensors="pt")
    prompt = inp["input_ids"][0].tolist()
    pv, thw = inp["pixel_values"], inp["image_grid_thw"]
    _, gh, gw = [int(v) for v in thw[0]]
    tt = inp["mm_token_type_ids"][0].tolist()
    first, last = tt.index(1), len(tt) - 1 - tt[::-1].index(1)
    pre, post = prompt[:first], prompt[last + 1:]
    with torch.no_grad():
        feats = m.model.get_image_features(pixel_values=pv, image_grid_thw=thw).pooler_output.reshape(
            -1, m.config.text_config.hidden_size)
        if feats.shape[0] != last - first + 1:
            sys.exit(f"the merger emits {feats.shape[0]} rows and the prompt holds {last - first + 1}")
        # The greedy continuation, one token at a time over the whole
        # sequence: no cache, so every position is the model from scratch.
        ids, types, cont = list(prompt), list(tt), []
        for _ in range(NCONT):
            out = m(input_ids=torch.tensor([ids]), pixel_values=pv, image_grid_thw=thw,
                    mm_token_type_ids=torch.tensor([types]), use_cache=False)
            nxt = int(out.logits[0, -1].argmax())
            cont.append(nxt)
            ids.append(nxt)
            types.append(0)
        pos, _ = m.model.get_rope_index(torch.tensor([ids]), torch.tensor([types]), thw)
    lg = out.logits[0]
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 6) for x in r[:NLOGIT]]} for r in lg[len(prompt) - 1:]]
    g = {
        "dtype": "float32",
        "pre": pre, "post": post, "cont": cont,
        "reply": proc.tokenizer.decode(cont),
        "image": {"grid": [1, gh // 2, gw // 2], "embd": [round(float(x), 6) for x in feats.reshape(-1)]},
        "pos4": pos[:, 0, :].T.tolist(),
        "logits": rows[:len(cont)],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "real-" + NAME + ".json"), "w") as f:
        json.dump(g, f)
    print(f"{NAME}: prompt {len(prompt)} rows, image {gh // 2}x{gw // 2} merged: {g['reply']!r}")
    if os.environ.get("HYVL_GGUF") == "1":
        for kind, dst in (("", NAME + "-f16.gguf"), ("--mmproj", "mmproj-" + NAME + "-f16.gguf")):
            args = [sys.executable, os.path.join(LCPP, "convert_hf_to_gguf.py"), hfdir,
                    "--outtype", "f16", "--outfile", os.path.join(MODELS, dst)]
            if kind:
                args.append(kind)
            subprocess.run(args, check=True)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else refenv.under("HYVL_SRC", "hunyuanvl", "hf", "HunyuanOCR")
    img = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "engine", "model", "testdata",
                                                               "quad-and-disc.png")
    main(src, img)
