#!/usr/bin/env python3
"""synth-llama4-q8.gguf as its transformers class, for scripts/rowsgold.py.

    L4_TOK=<dir> $JITLLM_HF_PY scripts/llama4hf.py

-> $JITLLM_MODELS/synth-llama4-q8-hf: config.json, model.safetensors and
   Llama 4's tokenizer.json

The fixture has no safetensors source: scripts/llama4gold.py builds
Llama4ForCausalLM in memory and writes the GGUF. This reads every tensor back
from that GGUF, Q8_0 dequantized, and puts each where llama4gold.py took it
from, under the config llama4gold.py states; the keys the GGUF carries are
checked against that config, its rope_freqs against the class's own llama3
rotary, and its token list against the tokenizer.json. Nothing is written
until the model's logits match the fixture's golden
(testdata/golden/llama4/synth-llama4-q8.json), taken from the model the GGUF
was written from: the weights are the container's.
"""
import json
import os
import shutil

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import Llama4ForCausalLM, Llama4TextConfig

import refenv

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "synth-llama4-q8"
SRC = os.path.join(refenv.models(), NAME + ".gguf")
DST = os.path.join(refenv.models(), NAME + "-hf")
GOLD = os.path.join(HERE, "..", "testdata", "golden", "llama4", NAME + ".json")
TOK = os.path.join(refenv.need("L4_TOK", "the directory holding Llama 4's tokenizer.json"), "tokenizer.json")

r = GGUFReader(SRC)


def kv(key):
    return r.fields["llama4." + key].contents()


T = {t.name: torch.from_numpy(np.array(dequantize(t.data, t.tensor_type), dtype=np.float32))
     for t in r.tensors}
tokens = r.fields["tokenizer.ggml.tokens"].contents()
V = len(tokens)

# llama4gold.py's config; what the GGUF states is checked against it below.
cfg = Llama4TextConfig(
    vocab_size=V, hidden_size=64, intermediate_size=64, intermediate_size_mlp=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_local_experts=4, num_experts_per_tok=1, interleave_moe_layer_step=2,
    attention_chunk_size=4, floor_scale=4, attn_scale=0.1,
    attn_temperature_tuning=True, use_qk_norm=True, rms_norm_eps=1e-5,
    max_position_embeddings=4096, tie_word_embeddings=False,
    bos_token_id=r.fields["tokenizer.ggml.bos_token_id"].contents(),
    eos_token_id=r.fields["tokenizer.ggml.eos_token_id"].contents(),
    rope_parameters=dict(rope_type="llama3", rope_theta=500000.0, factor=8.0, low_freq_factor=1.0,
                         high_freq_factor=8.0, original_max_position_embeddings=256),
)
stated = {
    "block_count": cfg.num_hidden_layers, "embedding_length": cfg.hidden_size,
    "feed_forward_length": cfg.intermediate_size_mlp, "expert_feed_forward_length": cfg.intermediate_size,
    "attention.head_count": cfg.num_attention_heads, "attention.head_count_kv": cfg.num_key_value_heads,
    "attention.key_length": cfg.head_dim, "expert_count": cfg.num_local_experts,
    "expert_used_count": cfg.num_experts_per_tok, "interleave_moe_layer_step": cfg.interleave_moe_layer_step,
    "attention.chunk_size": cfg.attention_chunk_size, "attention.temperature_length": cfg.floor_scale,
    "context_length": cfg.max_position_embeddings,
}
for k, want in stated.items():
    if kv(k) != want:
        raise SystemExit("%s: the GGUF states %r, the config %r" % (k, kv(k), want))
for k, want in {"rope.freq_base": 500000.0, "attention.layer_norm_rms_epsilon": cfg.rms_norm_eps,
                "attention.temperature_scale": cfg.attn_scale}.items():
    if not np.isclose(kv(k), want, rtol=1e-6):
        raise SystemExit("%s: the GGUF states %r, the config %r" % (k, kv(k), want))

cfg._attn_implementation = "eager"
m = Llama4ForCausalLM(cfg).float().eval()
P = dict(m.named_parameters())
F = cfg.intermediate_size
with torch.no_grad():
    P["model.embed_tokens.weight"].copy_(T.pop("token_embd.weight"))
    P["lm_head.weight"].copy_(T.pop("output.weight"))
    P["model.norm.weight"].copy_(T.pop("output_norm.weight"))
    for i in range(cfg.num_hidden_layers):
        b, g = "blk.%d." % i, "model.layers.%d." % i
        put = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
               "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj"}
        if i in cfg.moe_layers:
            put |= {"ffn_gate_inp": "feed_forward.router", "ffn_gate_shexp": "feed_forward.shared_expert.gate_proj",
                    "ffn_up_shexp": "feed_forward.shared_expert.up_proj",
                    "ffn_down_shexp": "feed_forward.shared_expert.down_proj"}
            gu = P[g + "feed_forward.experts.gate_up_proj"]  # (E, H, 2F)
            gu[:, :, :F] = T.pop(b + "ffn_gate_exps.weight").transpose(1, 2)
            gu[:, :, F:] = T.pop(b + "ffn_up_exps.weight").transpose(1, 2)
            P[g + "feed_forward.experts.down_proj"].copy_(T.pop(b + "ffn_down_exps.weight").transpose(1, 2))
        else:
            put |= {"ffn_gate": "feed_forward.gate_proj", "ffn_up": "feed_forward.up_proj",
                    "ffn_down": "feed_forward.down_proj"}
        for s, d in put.items():
            P[g + d + ".weight"].copy_(T.pop(b + s + ".weight"))
rf = T.pop("rope_freqs.weight")
if T:
    raise SystemExit("tensors placed nowhere: %s" % sorted(T))
base = 1.0 / (cfg.rope_parameters["rope_theta"] ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
if not torch.allclose(m.model.rotary_emb.inv_freq, base / rf, rtol=1e-6):
    raise SystemExit("rope_freqs.weight is not the class's llama3 rotary")

tj = json.load(open(TOK))
named = [""] * V
for s, i in tj["model"]["vocab"].items():
    named[i] = s
for t in tj["added_tokens"]:
    named[t["id"]] = t["content"]
if named != list(tokens):
    raise SystemExit("%s does not name the GGUF's %d tokens" % (TOK, V))

g = json.load(open(GOLD))
with torch.no_grad():
    lg = m(torch.tensor([g["ids"]])).logits[0]
worst = 0.0
for p, row in enumerate(g["pos"]):
    head = lg[p][:len(row["head"])]
    worst = max(worst, float((head - torch.tensor(row["head"])).abs().max()))
    if int(lg[p].argmax()) != row["argmax"]:
        raise SystemExit("position %d: argmax %d where the golden's is %d" % (p, int(lg[p].argmax()), row["argmax"]))
if worst > 1e-5:
    raise SystemExit("the logits are %.3g from the fixture's golden: these are not its weights" % worst)

os.makedirs(DST, exist_ok=True)
m.save_pretrained(DST, safe_serialization=True)
shutil.copy(TOK, os.path.join(DST, "tokenizer.json"))
print("wrote", DST, "from", SRC, "; %d positions within %.3g of the fixture's golden" % (len(g["pos"]), worst))
