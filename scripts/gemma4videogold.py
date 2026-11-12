#!/usr/bin/env python3
"""Record transformers' answers for a video on synth-gemma4v.

    $JITLLM_HF_PY scripts/gemma4videogold.py

-> testdata/golden/gemma4v/synth-gemma4v.video.json

The fixture is gemma4visiongold.py's -- its HF directory and the GGUFs written
from it -- read here and not rebuilt. The processor is Gemma4Processor with
Gemma4VideoProcessor at the class defaults the checkpoints publish (their
processor_config.json): 32 frames sampled uniformly, each resized at its
aspect to 70 soft tokens; in the prompt each frame's time as MM:SS, a space,
and the frame between the image markers, the frames a space apart. The video
is FORTY frames of 96 x 128 at 8 frames a second with a disc moving across
them, so the sampler skips frames and the times reach four seconds. The
golden carries:

    pre, post, cont  the text around the video and the continuation
    ids              the whole prompt as the processor writes it
    video            the frames (RGB bytes), the rate, the frames sampled,
                     each frame's grid and soft tokens, and every frame's
                     rows (get_video_features) in f32 and at the engine's
                     arithmetic (embd_a8: gemma4visiongold.py's int8)
    logits           the prompt's last row and each continuation row's, with
                     each frame's rows bidirectional on the sliding layers;
                     logits_a8 over the int8 rows; logits_causal with the
                     frames' rows causal on every layer, the violation
"""
import copy
import json
import os

import numpy as np
import torch

import gemma4visiongold as gv

N, H, W, FPS = 40, 96, 128, 8.0


def frames():
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    out = []
    for f in range(N):
        r = 255 * (0.5 + 0.5 * np.sin((x + 3 * f) / 7.0))
        g = 255 * (y / H)
        b = 255 * (((x - 10 - 3 * f) ** 2 + (y - 48) ** 2) < 20 ** 2)
        out.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    return np.stack(out)


def main():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, Gemma4Processor
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor
    d = os.path.join(gv.HFDIR, gv.NAME)
    m = Gemma4ForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32).eval()
    m.config._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(d)
    proc = Gemma4Processor(feature_extractor=Gemma4AudioFeatureExtractor(),
                           image_processor=Gemma4ImageProcessorPil(max_soft_tokens=280),
                           tokenizer=tok, video_processor=Gemma4VideoProcessor())
    vid = frames()
    pre_text, post_text = "<|turn>user\n", "What happens in the video?<turn|>\n<|turn>model\n"
    enc = proc(text=pre_text + proc.video_token + post_text, videos=[vid],
               video_metadata=[{"fps": FPS, "total_num_frames": N}], return_metadata=True, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    pv, pos = enc["pixel_values_videos"], enc["video_position_ids"]
    meta = enc.get("video_metadata")
    sampled = [int(i) for i in meta[0].frames_indices] if meta is not None else None
    nf = pv.shape[1]
    vt = proc.video_token_id
    pre = tok.encode(pre_text, add_special_tokens=False)
    post = tok.encode(post_text, add_special_tokens=False)
    cont = tok.encode("A disc moves to the right.", add_special_tokens=False)
    # The processor writes no BOS; the prompt runs after one, as
    # gemma4visiongold.py's does.
    assert ids[:len(pre)] == pre and ids[len(ids) - len(post):] == post, ids[:8]
    full = [2] + ids + cont
    t_ids = torch.tensor([full])
    types = torch.tensor([[2 if i == vt else 0 for i in full]])
    with torch.no_grad():
        lg = m(input_ids=t_ids, pixel_values_videos=pv, video_position_ids=pos, mm_token_type_ids=types).logits[0]
        causal = m(input_ids=t_ids, pixel_values_videos=pv, video_position_ids=pos).logits[0]
        vf = m.model.get_video_features(pv, pos, return_dict=True).pooler_output
        vf = torch.cat(list(vf), 0) if isinstance(vf, (list, tuple)) else vf.reshape(-1, vf.shape[-1])
    # The frames as a batch of pictures: the rows get_video_features writes.
    fpv, fpos = pv.flatten(0, 1), pos.flatten(0, 1)
    feats = gv.features(m, fpv, fpos)
    assert torch.equal(feats.reshape(vf.shape), vf), "a frame's rows are not its picture's"
    mq = copy.deepcopy(m)
    gv.quantize_vision(mq)
    feats_a8 = gv.features(mq, fpv, fpos, a8=True)
    with torch.no_grad():
        emb = m.get_input_embeddings()(t_ids)
        sel = (t_ids[0] == vt).nonzero().reshape(-1)
        emb[0, sel] = feats_a8.reshape(-1, emb.shape[-1]).to(emb.dtype)
        lg8 = m(inputs_embeds=emb, mm_token_type_ids=types).logits[0]
    first = len(ids)

    def rows(lg):
        return [{"argmax": int(r.argmax()), "max": float(r.max()),
                 "margin": float(r.topk(2).values[0] - r.topk(2).values[1]),
                 "head": [round(float(x), 7) for x in r[:gv.NLOGIT]]} for r in lg[first:]]

    valid = (pos[0, 0] >= 0).all(-1)
    xs, ys = pos[0, 0][valid][:, 0], pos[0, 0][valid][:, 1]
    soft = enc["num_soft_tokens_per_video"][0] if "num_soft_tokens_per_video" in enc else int(valid.sum()) // 9
    g = {
        "pre": pre, "post": post, "cont": cont, "ids": ids, "video_token": vt,
        "video": {"n": N, "h": H, "w": W, "fps": FPS, "rgb": gv.b64(vid), "frames": sampled, "sampled": nf,
                  "grid": [int(ys.max()) + 1, int(xs.max()) + 1], "soft": int(soft),
                  "embd": [round(float(x), 8) for x in feats.reshape(-1)],
                  "embd_a8": [round(float(x), 8) for x in feats_a8.reshape(-1)]},
        "logits": rows(lg),
        "logits_a8": rows(lg8),
        "logits_causal": rows(causal),
    }
    with open(os.path.join(gv.OUT, gv.NAME + ".video.json"), "w") as f:
        json.dump(g, f)
    print(f"{gv.NAME}: {len(ids)} ids, {nf} frames {sampled}, {g['video']['grid']} patches, {soft} soft tokens each")
    print(" ".join(s for s in tok.decode([i for i in ids if i != vt]).split() if ":" in s)[:400])


if __name__ == "__main__":
    main()
