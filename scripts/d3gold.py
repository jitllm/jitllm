#!/usr/bin/env python3
"""vllm-d3's goldens, from its own runtime (vllm-sr/d3-edge's d3_runtime.py),
for the engine's decision gates.

    $JITLLM_HF_PY -I scripts/d3gold.py DIR REQUEST OUT

DIR is the checkpoint (vllm-sr/d3-edge: its .py files, config.json, tokenizer,
processor configs, model and readout safetensors), run in float32 on the CPU.
REQUEST is a /v1/systemone body ("state", "questions", pictures in "images" and
video in "videos" as data URLs); OUT receives the runtime's answers and usage
for it, as the reference golden engine.TestDecisionOverPicturesMatchesItsGoldens
and TestDecisionOverVideoMatchesItsGolden read.
"""
import base64
import io
import json
import sys


def frames(video):
    """A video stated as its frames' data URLs is its frames as PIL pictures,
    which the runtime reads as frames; any other form passes as it is."""
    if not isinstance(video, list):
        return video
    from PIL import Image
    return [Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB") for u in video]


def main():
    d, req, out = sys.argv[1:4]
    sys.path.insert(0, d)
    import torch
    import d3_runtime

    torch.set_grad_enabled(False)
    model = d3_runtime.D3.from_pretrained(d, local_files_only=True, device="cpu", readout_dtype="float32")
    # The runtime loads its backbone in bfloat16; widened, the weights are the
    # same and the arithmetic is the engine's f32.
    model.backbone.float()
    body = json.load(open(req))
    res = model.system_one(state=body.get("state"), questions=body["questions"],
                           images=body.get("images") or None,
                           videos=[frames(v) for v in body.get("videos") or []] or None)
    with open(out, "w") as f:
        json.dump({"answers": res["answers"], "usage": res.get("usage", {})}, f)
    print(json.dumps(res["answers"])[:400])


if __name__ == "__main__":
    main()
