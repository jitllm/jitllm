"""Laya's goldens, from Laya's own code (rl_common.py / rl_agent_api.py in
convaiinnovations/laya), for the engine's decision gates.

    python -I scripts/layagold.py real    <laya_dir> <requests_dir> <out_dir>
    python -I scripts/layagold.py fixture <laya_dir> <requests_dir> <out_dir> <llama.cpp_dir>

real: the published checkpoint, the answer of RLAgent.system_one and the raw
scorer logits at the markers (before the temperature) for every question of
every request (*.json in requests_dir that is a /v1/systemone body). Written
as laya.<request>.reference.json.

fixture: a small DecisionModel (ModernBERT encoder, two head blocks, scorer)
built from Laya's own classes with EVERY parameter and buffer drawn at random,
saved in the checkpoint's layout and converted to GGUF by llama.cpp's own
convert_hf_to_gguf.py, with the same goldens from the same code. Written as
laya-fixture.gguf and laya-fixture.<request>.reference.json. The window is cut
to 8 so a request's state crosses it, and the rotary bases differ by layer
kind, so neither the local layers nor their base can be dropped silently.

The reference runs in float32 on the CPU (rl_agent_api enables autocast only
on CUDA).
"""
import json
import os
import shutil
import subprocess
import sys

import torch


def load_reference(laya_dir):
    sys.path.insert(0, laya_dir)
    import rl_agent_api
    import rl_common
    return rl_agent_api, rl_common


def answers(agent, rl_common, body):
    """RLAgent.system_one, plus the raw logits and the token ids per question."""
    state, questions = body["state"], body["questions"]
    out = {"answers": agent.system_one(state, questions)["answers"], "logits": {}, "ids": {}, "markers": {}}
    for qid, qdef in questions.items():
        q = agent._to_internal(qdef)
        seq, markers = rl_common.build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
        b = rl_common.collate_items([[{"ids": seq, "markers": markers, "qtype": rl_common.QTYPES[q["t"]],
                                        "target": [0.0] * len(markers), "label": -1, "episode": 0, "ep_step": 0,
                                        "ep_len": 1, "src": "gold"}]], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, _ = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        out["logits"][qid] = logits[0, :len(markers)].tolist()
        out["ids"][qid] = seq
        out["markers"][qid] = markers
    return out


def requests(requests_dir):
    for name in sorted(os.listdir(requests_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(requests_dir, name)) as f:
            body = json.load(f)
        if isinstance(body, dict) and "questions" in body and "state" in body:
            yield name[:-5], body


def write(out_dir, prefix, agent, rl_common, requests_dir):
    for name, body in requests(requests_dir):
        g = answers(agent, rl_common, body)
        path = os.path.join(out_dir, "%s.%s.reference.json" % (prefix, name))
        with open(path, "w") as f:
            json.dump(g, f, indent=1)
        print(path)


def real(laya_dir, requests_dir, out_dir):
    api, rl_common = load_reference(laya_dir)
    agent = api.RLAgent(laya_dir, device="cpu")
    write(out_dir, "laya", agent, rl_common, requests_dir)


def fixture(laya_dir, requests_dir, out_dir, llama_dir):
    api, rl_common = load_reference(laya_dir)
    work = os.path.join(out_dir, "laya-fixture.src")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "encoder"))
    shutil.copytree(os.path.join(laya_dir, "tokenizer"), os.path.join(work, "tokenizer"))
    with open(os.path.join(laya_dir, "encoder", "config.json")) as f:
        ecfg = json.load(f)
    # Small, with every structural feature of the real encoder kept: global
    # every third layer, a local window, two rotary bases, GeGLU, no biases.
    ecfg.update(hidden_size=128, num_attention_heads=4, num_hidden_layers=6, intermediate_size=96,
                local_attention=8, global_attn_every_n_layers=3, max_position_embeddings=1024,
                layer_types=["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(6)])
    ecfg["rope_parameters"] = {"full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                               "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}
    with open(os.path.join(work, "encoder", "config.json"), "w") as f:
        json.dump(ecfg, f, indent=1)
    with open(os.path.join(laya_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["max_len"] = 160
    cfg["head_max_len"] = 64
    with open(os.path.join(work, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    torch.manual_seed(20261008)
    model = rl_common.build_model(cfg, encoder_dir=os.path.join(work, "encoder"))
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if not p.is_floating_point():
                continue
            if p.dim() == 1 and name.endswith("weight"):
                p.copy_(1 + 0.3 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(work, "model.safetensors"))
    for f in ("rl_agent_api.py", "rl_common.py", "config.json"):
        shutil.copy(os.path.join(laya_dir, f), os.path.join(work, f))
    gguf = os.path.join(out_dir, "laya-fixture.gguf")
    subprocess.run([sys.executable, os.path.join(llama_dir, "convert_hf_to_gguf.py"), work, "--outtype", "f32",
                    "--outfile", gguf], check=True)
    agent = api.RLAgent(work, device="cpu")
    write(out_dir, "laya-fixture", agent, rl_common, requests_dir)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "real":
        real(*sys.argv[2:5])
    elif mode == "fixture":
        fixture(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
