"""Clef's goldens, from Clef's own code (joint_schema_model.py in
Cloudflare/clef-flash), for engine/model's decision gates.

    python -I scripts/clefgold.py fixture <clef_dir> <requests_dir> <out_dir>
    python -I scripts/clefgold.py real    <clef_dir> <requests_dir> <out_dir>

clef_dir holds the release's joint_schema_model.py, config.json,
joint_head_config.json and its tokenizer (tokenizer.json,
tokenizer_config.json, chat_template.jinja); the weights are not read.

fixture: a small Qwen3.5 backbone (transformers' Qwen3_5ForConditionalGeneration,
the release's class, linear and full attention alternating, a toy vision
tower) and a JointSchemaHead of two routing and two joint blocks, built from
those classes with EVERY parameter and buffer drawn at random -- the three
learned scalars too, which the release initialises at zero -- saved in the
release's layout and converted to GGUF by llama.cpp's own
convert_hf_to_gguf.py (its conversion/clef.py), with the goldens from the
same code: for every question of every request (*.json in requests_dir that
is a /v1/systemone body) the prompt's ids, each question's and option's span,
the head's logits in the prompt's option order, and systemone's answer.
Written as <out_dir>/clef-fixture.gguf and
<out_dir>/clef-fixture.<request>.reference.json.

real: the release itself (clef_dir the whole Cloudflare/clef-flash
snapshot), read by the release's classes in float32, with the same goldens,
written as <out_dir>/clef.<request>.reference.json. It needs some 40 GB of
memory.

The reference runs in float32 on the CPU. LCPP_SRC names the llama.cpp
checkout whose converter knows the clef arch.
"""
import json
import os
import shutil
import subprocess
import sys
import types

import torch



def load_reference(clef_dir):
    sys.path.insert(0, clef_dir)
    import joint_schema_model
    return joint_schema_model


def requests(requests_dir):
    for name in sorted(os.listdir(requests_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(requests_dir, name)) as f:
            body = json.load(f)
        if isinstance(body, dict) and "questions" in body and "state" in body:
            yield name[:-5], body


def goldens(jsm, model, tok, body):
    req = dict(body, model="clef")
    proc = types.SimpleNamespace(tokenizer=tok)
    with torch.no_grad():
        answer = jsm.systemone(model, proc, req)
        rec = jsm.encode_record(tok, req)
        logits = model(jsm.collate_records([rec], tok.pad_token_id or 0, torch.device("cpu")))[0]
    return {
        "answers": answer["answers"],
        "usage": answer["usage"],
        "ids": list(rec.input_ids),
        "questions": [{"id": q.question_id, "type": q.question_type, "span": list(q.question_span),
                       "options": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
                      for q in rec.questions],
        "logits": {q.question_id: l.tolist() for q, l in zip(rec.questions, logits)},
    }


def randomise(module, seed):
    # Every tensor the checkpoint carries: parameters and persistent buffers.
    # A non-persistent buffer (the rotary's inv_freq) is derived from the
    # config at load and is not the model's to randomise -- a random one is
    # a model no checkpoint can hold.
    torch.manual_seed(seed)
    saved = set(module.state_dict().keys())
    with torch.no_grad():
        for name, p in list(module.named_parameters()) + list(module.named_buffers()):
            if name not in saved or not p.is_floating_point():
                continue
            if p.dim() == 0:
                p.copy_(0.7 * torch.randn_like(p))
            elif p.dim() == 1 and name.endswith("weight"):
                p.copy_(1 + 0.3 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))


def fixture(clef_dir, requests_dir, out_dir):
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    jsm = load_reference(clef_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import refenv
    lcpp = refenv.lcpp_src()
    work = os.path.join(out_dir, "clef-fixture.src")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    # The release's shape at toy size: Qwen3.5's hybrid text model (three
    # linear blocks then a full one), the interleaved M-RoPE on a quarter of
    # each head, untied output, and a small tower the text path never reads.
    cfg = Qwen3_5Config(
        text_config=dict(vocab_size=248320, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=32, tie_word_embeddings=False, attn_output_gate=True,
                         layer_types=["linear_attention", "linear_attention",
                                      "linear_attention", "full_attention"],
                         linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
                         linear_num_key_heads=2, linear_num_value_heads=4,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                                          "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                                          "mrope_interleaved": True},
                         rms_norm_eps=1e-6, max_position_embeddings=4096),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                           num_position_embeddings=64, deepstack_visual_indexes=[]),
        tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    torch.manual_seed(20261013)
    backbone = Qwen3_5ForConditionalGeneration(cfg).float().eval()
    randomise(backbone, 20261013)
    # The hybrid's recurrence amplifies a large random weight into a
    # saturated state; the backbone's matrices are softer than the head's.
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if p.dim() >= 2:
                p.mul_(0.5)
    backbone.save_pretrained(work, safe_serialization=True)
    head_cfg = dict(hidden_size=64, width=32, routing_layers=2, layers=2, heads=4, feedforward=48)
    head = jsm.JointSchemaHead(**head_cfg).float().eval()
    randomise(head, 20261014)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, os.path.join(work, "joint_head.safetensors"))
    with open(os.path.join(work, "joint_head_config.json"), "w") as f:
        json.dump(head_cfg, f, indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "joint_schema_model.py"):
        shutil.copy(os.path.join(clef_dir, f), os.path.join(work, f))
    gguf = os.path.join(out_dir, "clef-fixture.gguf")
    subprocess.run([sys.executable, os.path.join(lcpp, "convert_hf_to_gguf.py"), work, "--outtype", "f32",
                    "--outfile", gguf], check=True)
    tok = AutoTokenizer.from_pretrained(work)
    # The goldens come from the checkpoint as written, read back as the
    # release is read (load_release_model's classes), so they are the
    # function of the files the converter read.
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(work, dtype=torch.float32).eval()
    head = jsm.JointSchemaHead(**head_cfg).float().eval()
    from safetensors.torch import load_file
    head.load_state_dict(load_file(os.path.join(work, "joint_head.safetensors")), strict=True)
    model = jsm.ClefModel(backbone, head).eval()
    for name, body in requests(requests_dir):
        g = goldens(jsm, model, tok, body)
        path = os.path.join(out_dir, "clef-fixture.%s.reference.json" % name)
        with open(path, "w") as f:
            json.dump(g, f, indent=1)
        print(path)


def real(clef_dir, requests_dir, out_dir):
    """The release (load_release_model's classes) in float32 on the CPU, with
    the same goldens as the fixture's, as clef.<request>.reference.json."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    jsm = load_reference(clef_dir)
    tok = AutoTokenizer.from_pretrained(clef_dir)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(clef_dir, dtype=torch.float32).eval()
    with open(os.path.join(clef_dir, "joint_head_config.json")) as f:
        head = jsm.JointSchemaHead(**json.load(f)).float().eval()
    head.load_state_dict(load_file(os.path.join(clef_dir, "joint_head.safetensors")), strict=True)
    head = head.float()
    model = jsm.ClefModel(backbone, head).eval()
    for name, body in requests(requests_dir):
        g = goldens(jsm, model, tok, body)
        path = os.path.join(out_dir, "clef.%s.reference.json" % name)
        with open(path, "w") as f:
            json.dump(g, f, indent=1)
        print(path)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "fixture":
        fixture(*sys.argv[2:5])
    elif len(sys.argv) == 5 and sys.argv[1] == "real":
        real(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
