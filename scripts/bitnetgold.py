#!/usr/bin/env python3
"""Build the BitNet b1.58 fixtures and record transformers' logits.

    $JITLLM_HF_PY scripts/bitnetgold.py
    JITLLM_MODELS=/path/to/models BITNETCPP_SRC=/path/to/bitnet.cpp

-> $CV_HF/synth-bitnet-hf          the release's form: uint8 codes and a bf16
                                   weight_scale each projection is multiplied by
                                   (autobitlinear, offline)
   $CV_HF/synth-bitnet-online-hf   the latent f32 weights, ternarized every call
                                   (autobitlinear, online)
   $JITLLM_MODELS/synth-bitnet-{f16,i2s}.gguf
                                   bitnet.cpp's convert-hf-to-gguf-bitnet.py over
                                   the online checkpoint ("bitnet-b1.58")
   testdata/golden/bitnet/synth-bitnet{,-online,-online-f16,-online-i2s}.json

THE ORACLE IS transformers' OWN BitNetForCausalLM, RUN THROUGH ITS OWN "bitnet"
QUANTIZER (integrations/bitnet.py): every projection's input quantized to int8
per token by its absolute maximum, the weight ternary. A fixture built on
nn.Linear would be a llama with two extra norms and gate nothing that makes
BitNet BitNet.

Every parameter and buffer is drawn: the latent weights, every norm (the two
inner ones included) away from one, and each packed weight_scale its own
bf16 number, so a scale dropped or inverted moves the logits. The rotary is
rotate_half over the rows as shipped (llama.cpp's NORM pairs would read them
wrong), and the gate is relu2. Every float is F32; the release form's scales
are bf16 values, so the container holds them exactly.

The online form's GGUFs are what bitnet.cpp's converter writes: its own
ternarization (the same ops as WeightQuant, in f32), stored as F16 or as I2_S.
llama.cpp's converter cannot write this class (it maps only 1bitLLM's names
for the two inner norms). I2_S carries each magnitude as the f32 the class
computes and writes the embedding as F16, so that container is held to the
class run on an f16-rounded embedding (-online-i2s); F16 rounds each
magnitude to an f16 too, so that one is held to the class run on those
rounded magnitudes (-online-f16). Every container holds its file's weights exactly, so every
golden is met at the activations' rounding alone.
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from tokenizers import Tokenizer

import refenv

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = refenv.models()
TOKDIR = refenv.under("CV_TOK", "tok")
HFDIR = refenv.under("CV_HF", "hf")
BNCPP = refenv.need("BITNETCPP_SRC", "a bitnet.cpp source checkout, its submodule included")
PY = sys.executable
OUT = os.path.join(HERE, "..", "testdata", "golden", "bitnet")
NPOS, NLOGIT = 12, 512
TEXTS = [
    "The capital of France is",
    "def getUserName(self):\n    return 12345",
    " leading space and  double  spaces\nline two\ttab",
]
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def config():
    from transformers import BitNetConfig
    # The 2B-4T's shape at toy size: GQA 4:2, 64-wide heads, rows a multiple
    # of a 256-element ternary block, tied embeddings, relu2.
    return BitNetConfig(vocab_size=128256, hidden_size=256, intermediate_size=512,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        hidden_act="relu2", rms_norm_eps=1e-5, max_position_embeddings=2048,
                        rope_theta=500000.0, tie_word_embeddings=True,
                        bos_token_id=128000, eos_token_id=128001)


def perturb(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            mod = n.split(".")[-2] if "." in n else ""
            if n.endswith("weight") and "norm" in mod:
                p.copy_(torch.randn_like(p) * 0.3 + 1)
            else:
                p.copy_(torch.randn_like(p) * 0.2)


def ternary(w):
    # WeightQuant, in f32: the codes and the magnitude 1/s.
    s = 1.0 / w.abs().mean().clamp(min=1e-5)
    q = (w * s).round().clamp(-1, 1)
    return q, s


def copy_tok(d):
    for f in os.listdir(os.path.join(TOKDIR, "bitnet")):
        shutil.copy(os.path.join(TOKDIR, "bitnet", f), d)


def half_magnitudes():
    # WeightQuant with each ternarized weight stored as an f16: what
    # bitnet.cpp's converter writes at --outtype f16, so its GGUF is held to
    # the class run on the file's own weights.
    from transformers.integrations import bitnet as B

    def fwd(ctx, weight):
        dtype = weight.dtype
        weight = weight.float()
        scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        w = (weight * scale).round().clamp(-1, 1) / scale
        return w.half().float().to(dtype)

    was = B.WeightQuant.forward
    B.WeightQuant.forward = staticmethod(fwd)
    return lambda: setattr(B.WeightQuant, "forward", was)


def golden(name, d, ids, cfg, f16=False, embf16=False):
    from transformers import AutoModelForCausalLM
    undo = half_magnitudes() if f16 else (lambda: None)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
    if embf16:
        # bitnet.cpp's I2_S writes the embedding (and so the tied head) as F16.
        with torch.no_grad():
            e = m.model.embed_tokens.weight
            e.copy_(e.half().float())
    m.config._attn_implementation = "eager"
    m.eval()
    with torch.no_grad():
        lg = m(torch.tensor([ids])).logits[0]
    undo()
    rows = [{"argmax": int(r.argmax()), "max": float(r.max()),
             "head": [round(float(x), 7) for x in r[:NLOGIT]]} for r in lg]
    tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    texts = [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"ids": ids, "vocab": cfg.vocab_size, "pos": rows, "texts": texts},
              open(os.path.join(OUT, name + ".json"), "w"))
    print("wrote", name, "argmax", [r["argmax"] for r in rows])


def build():
    from safetensors.torch import save_file
    from transformers import BitNetForCausalLM
    from transformers.integrations.bitnet import pack_weights
    cfg = config()
    cfg._attn_implementation = "eager"
    torch.manual_seed(20261013)
    m = BitNetForCausalLM(cfg).float().eval()
    perturb(m)
    sd = {k: v.detach().clone().contiguous() for k, v in m.state_dict().items() if k != "lm_head.weight"}
    ids = [(i * 7919 + 13) % 50000 for i in range(NPOS)]

    # The latent checkpoint, quantized online.
    on = os.path.join(HFDIR, "synth-bitnet-online-hf")
    shutil.rmtree(on, ignore_errors=True)
    os.makedirs(on)
    c = cfg.to_dict()
    c["architectures"] = ["BitNetForCausalLM"]
    # The release states its base as rope_theta, which every converter reads;
    # transformers 5 writes rope_parameters, which bitnet.cpp's does not.
    c.pop("rope_parameters", None)
    c["rope_theta"] = 500000.0
    c["torch_dtype"] = "float32"
    c["quantization_config"] = {"quant_method": "bitnet", "linear_class": "autobitlinear",
                                "quantization_mode": "online"}
    json.dump(c, open(os.path.join(on, "config.json"), "w"), indent=2)
    save_file(sd, os.path.join(on, "model.safetensors"), metadata={"format": "pt"})
    copy_tok(on)
    golden("synth-bitnet-online", on, ids, cfg)
    golden("synth-bitnet-online-f16", on, ids, cfg, f16=True)
    golden("synth-bitnet-online-i2s", on, ids, cfg, embf16=True)

    # The release's form: the same codes, packed, and a bf16 multiplier per
    # projection that is not the latent magnitude, so the scale is the
    # file's and nothing else's.
    pk = os.path.join(HFDIR, "synth-bitnet-hf")
    shutil.rmtree(pk, ignore_errors=True)
    os.makedirs(pk)
    psd = {}
    for k, v in sd.items():
        if any(k.endswith(p + ".weight") for p in PROJ):
            q, s = ternary(v)
            psd[k] = pack_weights(q.to(torch.int8).clone())
            ws = (1.0 / s) * (0.75 + 0.5 * torch.rand(()))
            psd[k[: -len(".weight")] + ".weight_scale"] = ws.reshape(1).to(torch.bfloat16)
        else:
            psd[k] = v
    c["quantization_config"] = {"quant_method": "bitnet", "linear_class": "autobitlinear",
                                "quantization_mode": "offline"}
    c["torch_dtype"] = "float32"
    json.dump(c, open(os.path.join(pk, "config.json"), "w"), indent=2)
    save_file(psd, os.path.join(pk, "model.safetensors"), metadata={"format": "pt"})
    copy_tok(pk)
    golden("synth-bitnet", pk, ids, cfg)

    # bitnet.cpp's converter: llama.cpp's maps none of transformers' inner-norm
    # names (it knows 1bitLLM's inner_attn_ln and ffn_layernorm only).
    env = dict(os.environ, PYTHONPATH=os.path.join(BNCPP, "3rdparty", "llama.cpp", "gguf-py"))
    for ot in ("f16", "i2_s"):
        dst = os.path.join(MODELS, "synth-bitnet-%s.gguf" % ot.replace("_", ""))
        subprocess.run([PY, os.path.join(BNCPP, "utils", "convert-hf-to-gguf-bitnet.py"), on,
                        "--outtype", ot, "--outfile", dst], check=True, env=env,
                       cwd=os.path.join(BNCPP, "utils"))
        print("wrote", dst)


if __name__ == "__main__":
    build()
