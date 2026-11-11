#!/usr/bin/env python3
"""Record transformers' answers for a video on the Qwen2-VL family fixtures.

    $JITLLM_HF_PY scripts/qwenvlvideogold.py [name ...]

-> testdata/golden/qwenvl/<name>.video.json

The fixtures are qwenvlgold.py's -- its HF directories and the GGUFs written
from them -- read here and not rebuilt. The video is FIVE frames of 300 x 340
with a disc moving across them: an odd count, so the processor repeats the
last; a size smart_resize changes (to 308 x 336, the frame's area alone
within the bounds); three pairs, so the temporal axis holds three positions.
The processor is Qwen2VLVideoProcessor at its class defaults, which is what
the converter states (convert.qwen2vlVideo), every frame read.

On Qwen2.5-VL the video comes at half a frame a second, so its pairs stand
tokens_per_second * int(2 / 0.5) = 8 positions apart on the temporal axis,
and the prompt ends two tokens after the video: the temporal axis reaches
past the text, so the first decoded row takes one past the prompt's highest
position (get_rope_index's rope delta) rather than the next text position.
The continuation is decoded a token at a time through the cache, as generate
runs it, not teacher-forced in one pass. The golden carries:

    pre, post  the text before <|vision_start|> and after <|vision_end|>
    video      the frames (RGB bytes), the rate, the grid (pairs, merged rows,
               merged columns), and every pair's tower rows
               (get_video_features): in f32 (embd) and with the tower's
               matrices at the converter's Q8_0 and their inputs at int8
               (embd_a8) -- the patch embedding's two planes quantized as the
               container holds them, their sum and the second
    pos        every prompt row's (t, h, w), and the rope delta
    logits     the prompt's last row and each continuation row's
"""
import copy
import json
import os
import sys

import numpy as np
import torch

import qwenvlgold as qg

NAMES = {
    # name: (class, the rate, the text after the video)
    "synth-qwen2vl": ("Qwen2VLForConditionalGeneration", 0,
                      "What happens in the video?<|im_end|>\n<|im_start|>assistant\n"),
    "synth-qwen25vl": ("Qwen2_5_VLForConditionalGeneration", 0.5, "Go"),
}
FRAMES, FRAME_H, FRAME_W = 5, 300, 340


def frames():
    # The picture's gradients with a disc that moves, so each pair's frames
    # differ and the entry's second plane (the frames' difference) is read.
    y, x = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float64)
    out = []
    for f in range(FRAMES):
        r = 255 * (0.5 + 0.5 * np.sin((x + 7 * f) / 9.0))
        g = 255 * (y / FRAME_H)
        b = 255 * (((x - 60 - 40 * f) ** 2 + (y - 150) ** 2) < 50 ** 2)
        out.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    return out


def quantize_video_tower(m):
    # The tower's matrices as quantize_tower holds them, but for the patch
    # embedding, which a video reads as the container holds it: the planes'
    # sum and the second plane, each at Q8_0, against the first frame and the
    # frames' difference -- so the first plane is their difference.
    with torch.no_grad():
        planes = None
        for n, p in m.named_parameters():
            if n.startswith("model.visual") and "patch_embed" in n and n.endswith("weight"):
                planes = (n, p[:, :, 0].clone(), p[:, :, 1].clone())
        qg.quantize_tower(m)
        n, w0, w1 = planes
        p = dict(m.named_parameters())[n]
        o = p.shape[0]
        qs = qg.q8rows((w0 + w1).reshape(o, -1)).reshape(w0.shape)
        q1 = qg.q8rows(w1.reshape(o, -1)).reshape(w1.shape)
        p[:, :, 0] = qs - q1
        p[:, :, 1] = q1


def video_features(m, pv, thw):
    with torch.no_grad():
        out = m.model.get_video_features(pv, thw)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), 0)
    return feats


def video_features_a8(m, pv, thw):
    hs = [mod.register_forward_pre_hook(lambda mod, inp: (qg.qact(inp[0]),))
          for n, mod in m.named_modules()
          if n.startswith("model.visual") and isinstance(mod, torch.nn.Linear)]
    try:
        return video_features(m, pv, thw)
    finally:
        for h in hs:
            h.remove()


def build(name):
    import transformers
    from transformers import AutoTokenizer, Qwen2VLVideoProcessor
    clsname, fps, after = NAMES[name]
    d = os.path.join(qg.HFDIR, name)
    m = getattr(transformers, clsname).from_pretrained(d, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = m.config
    tok = AutoTokenizer.from_pretrained(d)
    vp = Qwen2VLVideoProcessor(cap_pixels_per_frame=False)
    vids = frames()
    out = vp(videos=[np.stack(vids)], return_tensors="pt", do_sample_frames=False)
    pv, thw = out["pixel_values_videos"], out["video_grid_thw"]
    merge = cfg.vision_config.spatial_merge_size
    gt, gh, gw = [int(v) for v in thw[0]]
    nvid = gt * gh * gw // (merge * merge)
    pre = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    post = tok.encode(after, add_special_tokens=False)
    cont = tok.encode(" A disc moves.", add_special_tokens=False)
    ids = pre + [cfg.vision_start_token_id] + [cfg.video_token_id] * nvid + [cfg.vision_end_token_id] + post
    types = [0] * (len(pre) + 1) + [2] * nvid + [0] * (1 + len(post))
    t_ids = torch.tensor([ids])
    t_types = torch.tensor([types], dtype=torch.int32)
    kw = {}
    if clsname.startswith("Qwen2_5"):
        # The published checkpoints' tokens_per_second, which the converter
        # states (an mmproj carries none); the fixture was built at the
        # class's 4.
        cfg.vision_config.tokens_per_second = 2
        rate = fps or 24
        kw["second_per_grid_ts"] = torch.tensor([vp.temporal_patch_size / rate])
    feats = video_features(m, pv, thw)
    mq = copy.deepcopy(m)
    quantize_video_tower(mq)
    feats_a8 = video_features_a8(mq, pv, thw)
    with torch.no_grad():
        pos, delta = m.model.get_rope_index(t_ids, t_types, video_grid_thw=thw, **kw)
        m.model.rope_deltas = None
        o = m(input_ids=t_ids, pixel_values_videos=pv, video_grid_thw=thw, mm_token_type_ids=t_types,
              use_cache=True, **kw)
        lg = [o.logits[0, -1]]
        pkv = o.past_key_values
        for t in cont[:-1]:
            o = m(input_ids=torch.tensor([[t]]), past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            lg.append(o.logits[0, -1])
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:qg.NLOGIT]]} for r in lg]
    g = {
        "pre": pre, "post": post, "cont": cont,
        "video": {"n": FRAMES, "h": FRAME_H, "w": FRAME_W, "fps": fps,
                  "rgb": qg.b64(np.stack(vids)),
                  "grid": [gt, gh // merge, gw // merge],
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
        "pos": pos[:, 0, :].T.tolist(),
        "delta": int(delta.reshape(-1)[0]),
        "logits": rows,
    }
    os.makedirs(qg.OUT, exist_ok=True)
    with open(os.path.join(qg.OUT, name + ".video.json"), "w") as f:
        json.dump(g, f)
    top = int(pos.max())
    print(f"{name}: {len(ids)} rows, {gt} pairs of {gh // merge}x{gw // merge} merged, "
          f"positions to {top}, the last prompt row's t {int(pos[0, 0, -1])}, rope delta {g['delta']}, "
          f"{len(rows)} logit rows")


if __name__ == "__main__":
    for n in sys.argv[1:] or NAMES:
        build(n)
