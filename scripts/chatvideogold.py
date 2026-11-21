#!/usr/bin/env python3
"""Record the prompt transformers' processor makes of a chat turn holding a
picture, a video and text, on the Qwen2-VL fixture.

    $JITLLM_HF_PY scripts/chatvideogold.py

-> testdata/golden/qwenvl/synth-qwen2vl.chat.json

The fixture is qwenvlgold.py's HF directory, read and not rebuilt: its
tokenizer and chat template are Qwen2-VL-2B-Instruct's. The turn is a user
message whose content is a picture, then a video, then text -- the order a
message carrying both puts them in (model.ChatMessage: its pictures, then
its videos) -- rendered by the tokenizer's chat template with the generation
prompt, then expanded by Qwen2VLProcessor: the picture's <|image_pad|> to its
merged rows, the video's <|video_pad|> to every pair's. The video processor
is Qwen2VLVideoProcessor at its class defaults, every frame read, which is
what the converter states (convert.qwen2vlVideo). The golden carries:

    text     the rendered template
    picture  its RGB bytes and size
    video    its frames' RGB bytes, their count and size
    ids      the processor's input_ids
"""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwenvlgold as qg  # noqa: E402

NAME = "synth-qwen2vl"
PIC_H, PIC_W = 56, 84
FRAMES, FRAME_H, FRAME_W = 4, 84, 112


def picture():
    y, x = np.mgrid[0:PIC_H, 0:PIC_W].astype(np.float64)
    rgb = np.stack([255 * x / PIC_W, 255 * y / PIC_H, 128 + 0 * x], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def frames():
    y, x = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float64)
    out = []
    for f in range(FRAMES):
        r = 255 * (0.5 + 0.5 * np.sin((x + 5 * f) / 7.0))
        g = 255 * (y / FRAME_H)
        b = 255 * (((x - 20 - 15 * f) ** 2 + (y - 40) ** 2) < 15 ** 2)
        out.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    return out


def main():
    from PIL import Image
    from transformers import AutoTokenizer, Qwen2VLImageProcessorPil, Qwen2VLProcessor, Qwen2VLVideoProcessor
    d = os.path.join(qg.HFDIR, NAME)
    tok = AutoTokenizer.from_pretrained(d)
    proc = Qwen2VLProcessor(image_processor=Qwen2VLImageProcessorPil.from_pretrained(d), tokenizer=tok,
                            video_processor=Qwen2VLVideoProcessor(cap_pixels_per_frame=False),
                            chat_template=tok.chat_template)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "video"},
                                         {"type": "text", "text": "What happens in the video?"}]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pic, vid = picture(), frames()
    out = proc(text=[text], images=[Image.fromarray(pic)], videos=[np.stack(vid)], do_sample_frames=False,
               return_tensors="np")
    ids = [int(i) for i in out["input_ids"][0]]
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    g = {
        "text": text,
        "picture": {"h": PIC_H, "w": PIC_W, "rgb": b64(pic)},
        "video": {"n": FRAMES, "h": FRAME_H, "w": FRAME_W, "rgb": b64(np.stack(vid))},
        "ids": ids,
    }
    path = os.path.join(qg.OUT, NAME + ".chat.json")
    with open(path, "w") as f:
        json.dump(g, f)
    print(f"{NAME}: {len(ids)} ids, {ids.count(proc.image_token_id if hasattr(proc, 'image_token_id') else -1)} picture rows,"
          f" grid {[int(v) for v in out['video_grid_thw'][0]]}")


if __name__ == "__main__":
    main()
