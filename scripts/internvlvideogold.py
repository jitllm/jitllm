"""Transformers' golden for InternVL3's video.

    $JITLLM_HF_PY -E -s scripts/internvlvideogold.py

-> $JITLLM_MODELS/vlm/goldens/internvl/video.json, video.features.f32 and
   video-thumbs.features.f32

The video is three frames of 640 x 360 with a disc moving across them: a
size InternVL's picture tiling cuts into tiles and a thumbnail, so a frame's
last square is not its first. The processor is the checkpoint's own
(InternVLProcessor and InternVLVideoProcessor, which reads every frame at
448 x 448 with the picture's normalisation), on the chat template's
rendering of a video between a user turn's header and its text. The golden
carries:

    video.json                 the frames (RGB bytes) and their size, the
                               prompt's ids as the processor writes them, and
                               the ids of the text before and after the video
    video.features.f32         each frame's rows, get_image_features over the
                               video processor's pixels
    video-thumbs.features.f32  each frame read as a picture by the image
                               processor, its last square's rows: the
                               thumbnail the engine reads a frame as

The tower and projector are visiongold.py's: transformers' own classes with
the checkpoint's vision weights, in float32 on the CPU.
"""
import base64
import json
import os

import numpy as np
import torch
from PIL import Image

import visiongold as vg

N, H, W = 3, 360, 640
PRE = "<|im_start|>user\n"
POST = "\nDescribe this video.<|im_end|>\n<|im_start|>assistant\n"


def frames():
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    out = []
    for f in range(N):
        r = 255 * (0.5 + 0.5 * np.sin((x + 7 * f) / 9.0))
        g = 255 * (y / H)
        b = 255 * (((x - 120 - 180 * f) ** 2 + (y - 180) ** 2) < 70 ** 2)
        out.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    return out


def main():
    m, proc, size = vg.internvl()
    out = os.path.join(vg.VLM, "goldens", "internvl")
    os.makedirs(out, exist_ok=True)
    fs = frames()
    vid = np.stack(fs)
    enc = proc(text=PRE + "<video>" + POST, videos=[vid], return_tensors="pt")
    px = enc["pixel_values"].reshape(-1, 3, size, size)
    assert px.shape[0] == N, px.shape
    vg.save(os.path.join(out, "video.features.f32"), vg.features(m, px))
    thumbs = []
    for f in fs:
        e = proc(text=proc.image_token, images=[Image.fromarray(f)], return_tensors="pt")
        tiles = e["pixel_values"].reshape(-1, 3, size, size)
        assert tiles.shape[0] > 1, "a frame that does not tile has no thumbnail to tell from its first square"
        thumbs.append(tiles[-1:])
    vg.save(os.path.join(out, "video-thumbs.features.f32"), vg.features(m, torch.cat(thumbs)))
    tok = proc.tokenizer
    with open(os.path.join(out, "video.json"), "w") as f:
        json.dump({"n": N, "h": H, "w": W, "rgb": base64.b64encode(vid.reshape(-1).tobytes()).decode(),
                   "ids": enc["input_ids"][0].tolist(),
                   "pre": tok(PRE, add_special_tokens=False)["input_ids"],
                   "post": tok(POST, add_special_tokens=False)["input_ids"],
                   "image_token": m.config.image_token_id,
                   "video_processor": type(proc.video_processor).__name__}, f)
    print("frames", tuple(px.shape), "ids", enc["input_ids"].shape[1], "tiles", tiles.shape[0])


if __name__ == "__main__":
    main()
