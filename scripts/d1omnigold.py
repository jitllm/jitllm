#!/usr/bin/env python3
"""d1-omni's goldens, from its own code (LiquidAI/d1-omni-600M's modeling_d1.py,
encoder.py, prompt.py), for engine/model's decision gates.

    $JITLLM_HF_PY scripts/d1omnigold.py DIR REQUESTS_DIR OUT_DIR

DIR is the checkpoint (its .py files, config.json, tokenizer and
model.safetensors), read in float32 on the CPU. For every /v1/systemone body in
REQUESTS_DIR (*.json with "state" and "questions"; pictures in "images" as
data URLs, PIL-decoded) it writes OUT_DIR/d1-omni.<request>.reference.json:
each question's ids and option-marker positions (prompt.encode), the trunk's
output rows at the markers, the head's raw logits (before the temperature),
and system_one's answer. A voice clip in "files" (a data:audio/wav URL, 16 kHz
mono PCM16) is the audio prefix; its golden carries the clip's log-mel
features (the first frames) and the prefix rows' first values too.

With D1OMNI_F16=1 every matrix the GGUF stores in F16 (each 2-D weight but the
question-type table) is rounded to f16 first, so the golden is the F16 GGUF's
function and what is left between the engine and it is f32 order.
"""
import base64
import importlib
import importlib.util
import io
import json
import os
import sys

import torch


def load(d):
    spec = importlib.util.spec_from_file_location("d1omni", os.path.join(d, "__init__.py"),
                                                  submodule_search_locations=[d])
    sys.modules["d1omni"] = importlib.util.module_from_spec(spec)
    md = importlib.import_module("d1omni.modeling_d1")
    pr = importlib.import_module("d1omni.prompt")
    model = md.D1OmniModel.from_pretrained(d, dtype=torch.float32).eval()
    if os.environ.get("D1OMNI_F16") == "1":
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() == 2 and "type_emb" not in name:
                    p.copy_(p.half().float())
    return md, pr, model


def pictures(body):
    from PIL import Image
    out = []
    for u in body.get("images") or []:
        out.append(Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB"))
    return out


def clip(body):
    import wave
    import numpy as np
    for u in body.get("files") or []:
        if u.startswith("data:audio/"):
            w = wave.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1])))
            if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise SystemExit("a voice clip must be 16 kHz mono PCM16")
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    return None


def main(d, reqs, out):
    if not os.path.exists(os.path.join(d, "__init__.py")):
        open(os.path.join(d, "__init__.py"), "w").close()
    md, pr, model = load(d)
    tok = model.tokenizer
    for name in sorted(os.listdir(reqs)):
        if not name.endswith(".json"):
            continue
        body = json.load(open(os.path.join(reqs, name)))
        if not isinstance(body, dict) or "questions" not in body or "state" not in body:
            continue
        if body.get("videos"):
            continue
        audio = clip(body)
        if body.get("files") and audio is None:
            continue
        imgs = pictures(body) or None
        qs = {n: pr.as_question(q) for n, q in body["questions"].items()}
        state = body["state"]
        with torch.no_grad():
            resp = model.system_one(state, body["questions"], images=imgs, audio=audio)
            # The rows the head reads: the same encode system_one runs.
            if imgs:
                prefix = model.vision(imgs)
            elif audio is not None:
                prefix = model.audio(audio)
            else:
                prefix = None
            p = 0 if prefix is None else prefix.shape[1]
            max_len = (model.config.image_text_length if imgs else
                       model.config.audio_text_length if audio is not None else model.config.max_length)
            max_len = min(max_len, model.config.max_length - p)
            noul = md.YES_NO if imgs or audio is not None else None
            spoken = audio is not None
            st = state
            if spoken and st is None:
                st = {}
            st = "" if st is None else st
            golden = {"answers": resp["answers"], "usage": resp["usage"], "questions": []}
            if spoken:
                am = importlib.import_module("d1omni.audio")
                mel, frames = model.audio.frontend(am.waveform(audio))
                golden["mel_frames"] = int(frames[0])
                golden["mel"] = [[round(float(v), 6) for v in mel[0, :, t]] for t in range(min(4, mel.shape[2]))]
                golden["prefix_rows"] = [[round(float(v), 6) for v in prefix[0, r, :16]] for r in range(p)]
            for qid, q in qs.items():
                ids, markers = pr.encode(tok, st, q, max_len, noul, spoken)
                row = (prefix, ids, markers, q, prefix is None)
                text = model.encoder.embed_tokens(torch.tensor(ids))
                seq = text if prefix is None else torch.cat([prefix[0], text])
                h = model.encoder(seq[None], torch.ones(1, len(seq), dtype=torch.bool), torch.tensor([p]))
                ht = h[0, p:]
                qtype = torch.tensor([pr.QTYPES[q.type]])
                logits = model.head(ht[None], torch.ones(1, len(ids), dtype=torch.bool),
                                    torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool), qtype)[0]
                golden["questions"].append({
                    "id": qid, "ids": ids, "markers": markers, "prefix": p,
                    "trunk_at_markers": [[round(float(v), 7) for v in ht[m]] for m in markers],
                    "logits": [float(v) for v in logits[:q.options]],
                    "probabilities": model._run([row])[0],
                })
        path = os.path.join(out, "d1-omni.%s.reference.json" % name[:-5])
        json.dump(golden, open(path, "w"))
        print(path, json.dumps(resp["answers"])[:160])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
