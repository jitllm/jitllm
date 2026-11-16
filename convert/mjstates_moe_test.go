package convert

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestTheLibraryStatesTheMixtures holds each mixture and flagship
// architecture whose container the library states to what its reader in
// configOfFile set before the files did: the flags, the router's gate, the
// scales, the interleave and the rotary base, and each refusal it made,
// with the reader's own case beside it.
func TestTheLibraryStatesTheMixtures(t *testing.T) {
	neox, qk := jlm.FlagRopeNeox, jlm.FlagRopeNeox|jlm.FlagQKNorm
	sig := jlm.FlagExpertSigmoid
	norms := []string{"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"}
	u, fl := meta.MakeUint, meta.MakeFloat
	type keys = map[string]meta.Value
	for _, tc := range []struct {
		name     string
		arch     string
		keys     keys
		tensors  []string
		nrot     uint32
		flags    jlm.Flags
		eps      float32
		logit    float32
		scale    float32
		residual float32
		step     uint32
		rope     float32
		win, per uint32
		temp     [3]float32
		refused  string
	}{
		{name: "qwen2moe", arch: "qwen2moe", keys: keys{"expert_count": u(4)},
			tensors: []string{"blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight"}, flags: neox | jlm.FlagNoExpertNorm},
		{name: "qwen2moe renormalising", arch: "qwen2moe", keys: keys{"expert_count": u(4), "expert_weights_norm": meta.MakeBool(true)},
			tensors: []string{"blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight"}, flags: neox},
		{name: "qwen2moe, a block with no router", arch: "qwen2moe", keys: keys{"expert_count": u(4)},
			tensors: []string{"blk.0.ffn_gate_inp.weight"}, refused: "block 1 has no ffn_gate_inp.weight"},
		{name: "qwen2moe, no experts", arch: "qwen2moe",
			tensors: []string{"blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight"}, refused: "no expert_count"},
		{name: "qwen2moe, a sigmoid", arch: "qwen2moe", keys: keys{"expert_count": u(4), "expert_gating_func": u(2)},
			tensors: []string{"blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight"}, refused: "router is softmax"},
		{name: "ernie", arch: "ernie4_5-moe", keys: keys{"interleave_moe_layer_step": u(2)}, step: 2},
		{name: "ernie, no step", arch: "ernie4_5-moe", refused: "no interleave_moe_layer_step"},
		{name: "ernie, not renormalised", arch: "ernie4_5-moe",
			keys: keys{"interleave_moe_layer_step": u(1), "expert_weights_norm": meta.MakeBool(false)}, refused: "always renormalises"},
		{name: "dots1", arch: "dots1", keys: keys{"expert_count": u(4)}, tensors: norms, flags: qk | sig},
		{name: "dots1, a window", arch: "dots1", keys: keys{"expert_count": u(4), "attention.sliding_window": u(8)},
			tensors: norms, refused: "max_window_layers"},
		{name: "dots1, a partial rotary", arch: "dots1", keys: keys{"expert_count": u(4)}, tensors: norms, nrot: 4,
			refused: "turns the whole head"},
		{name: "dots1, no k norm", arch: "dots1", keys: keys{"expert_count": u(4)}, tensors: norms[:1],
			refused: "no blk.0.attn_k_norm.weight"},
		{name: "bailingmoe2", arch: "bailingmoe2", keys: keys{"expert_count": u(4)}, tensors: norms, flags: qk | sig},
		{name: "bailingmoe2, a softmax", arch: "bailingmoe2", keys: keys{"expert_count": u(4), "expert_gating_func": u(1)},
			tensors: norms, refused: "router is sigmoid"},
		{name: "glm4moe-air", arch: "glm4moe", flags: neox | sig},
		{name: "glm4moe", arch: "glm4moe", tensors: norms, flags: qk | sig},
		{name: "glm4", arch: "glm4"},
		{name: "glm4, prediction layers", arch: "glm4", keys: keys{"nextn_predict_layers": u(1)}, refused: "nextn_predict_layers"},
		{name: "dbrx", arch: "dbrx", keys: keys{"attention.clamp_kqv": fl(8), "expert_count": u(4), "attention.layer_norm_epsilon": fl(1e-6)},
			flags: jlm.FlagLayerNorm | neox, eps: 1e-6},
		{name: "dbrx, no clamp", arch: "dbrx", keys: keys{"expert_count": u(4)}, refused: "no attention.clamp_kqv"},
		{name: "granite", arch: "granite", keys: keys{"logit_scale": fl(8), "embedding_scale": fl(12), "residual_scale": fl(0.25)},
			logit: 8, scale: 12, residual: 0.25},
		{name: "granitemoe", arch: "granitemoe", keys: keys{"logit_scale": fl(6)}, logit: 6, scale: 1},
		{name: "granite, no logit scale", arch: "granite", refused: "logit_scale"},
		{name: "granite, no rotary", arch: "granite", keys: keys{"logit_scale": fl(8), "rope.scaling.finetuned": meta.MakeBool(false)},
			refused: "neither a rotary"},
		{name: "minimax-m2", arch: "minimax-m2", keys: keys{"expert_count": u(4)},
			tensors: []string{"blk.0.exp_probs_b.bias", "blk.1.exp_probs_b.bias"}, flags: qk | jlm.FlagQKNormWide | sig},
		{name: "minimax-m2, a shared expert", arch: "minimax-m2", keys: keys{"expert_count": u(4), "expert_shared_count": u(1)},
			tensors: []string{"blk.0.exp_probs_b.bias", "blk.1.exp_probs_b.bias"}, refused: "expert_shared_count"},
		{name: "hunyuan-moe", arch: "hunyuan-moe", flags: qk | jlm.FlagQKNormPostRope, rope: 10000},
		{name: "hunyuan-dense", arch: "hunyuan-dense", flags: qk | jlm.FlagQKNormPostRope, rope: 10000},
		{name: "hunyuan-moe, a routed scale", arch: "hunyuan-moe", keys: keys{"expert_weights_scale": fl(2.5)},
			refused: "scales nothing"},
		{name: "hunyuan_vl", arch: "hunyuan_vl", keys: keys{"rope.freq_base": fl(10000), "rope.scaling.alpha": fl(1000),
			"rope.dimension_count": u(8), "rope.dimension_sections": meta.MakeInt32s([]int32{1, 1, 1, 1})},
			flags: qk | jlm.FlagQKNormPostRope, rope: float32(float64(float32(10000)) * math.Pow(1000, 8.0/6))},
		{name: "hunyuan_vl, the rotary's width off the head's", arch: "hunyuan_vl", keys: keys{"rope.scaling.alpha": fl(1000),
			"attention.key_length": u(8), "rope.dimension_sections": meta.MakeInt32s([]int32{1, 1, 1, 1})},
			flags: qk | jlm.FlagQKNormPostRope, rope: float32(10000 * math.Pow(1000, 8.0/6))},
		{name: "apertus", arch: "apertus", flags: qk},
		{name: "llama4 scout", arch: "llama4", keys: keys{"expert_count": u(16)},
			flags: jlm.FlagSWAChunked | jlm.FlagNoPEGlobal | jlm.FlagExpertWeightIn | jlm.FlagQKL2Norm | sig | jlm.FlagNoExpertNorm,
			win:   8192, per: 4, temp: [3]float32{0.1, 8192, 1}, step: 1},
		{name: "llama4 maverick", arch: "llama4", keys: keys{"expert_count": u(128), "interleave_moe_layer_step": u(2),
			"attention.chunk_size": u(4096), "attention.sliding_window_pattern": u(3), "attention.temperature_scale": fl(0.5),
			"attention.temperature_length": u(1024)},
			flags: jlm.FlagSWAChunked | jlm.FlagNoPEGlobal | jlm.FlagExpertWeightIn | sig | jlm.FlagNoExpertNorm,
			win:   4096, per: 3, temp: [3]float32{0.5, 1024, 1}, step: 2},
		{name: "llama4, every layer global", arch: "llama4", keys: keys{"attention.sliding_window": u(0)}, refused: "every layer attending globally"},
		{name: "llama4, no chunk", arch: "llama4", keys: keys{"attention.chunk_size": u(0)}, refused: "over no chunk"},
		{name: "llama4, renormalised", arch: "llama4", keys: keys{"expert_weights_norm": meta.MakeBool(true)}, refused: "does not renormalise"},
		{name: "phimoe", arch: "phimoe", keys: keys{"expert_count": u(4), "expert_used_count": u(2), "block_count": u(2),
			"attention.sliding_window": u(4096), "context_length": u(8192)},
			flags: jlm.FlagLayerNorm | neox | jlm.FlagNoExpertNorm, win: 4096, per: 3},
		{name: "phimoe, a window no shorter than the context", arch: "phimoe", keys: keys{"expert_count": u(4), "expert_used_count": u(2),
			"block_count": u(2), "attention.sliding_window": u(8192), "context_length": u(8192)},
			flags: jlm.FlagLayerNorm | neox | jlm.FlagNoExpertNorm},
		{name: "phimoe, three experts", arch: "phimoe", keys: keys{"expert_count": u(4), "expert_used_count": u(3), "block_count": u(2)},
			refused: "selects 2"},
		{name: "phimoe, a gate", arch: "phimoe", keys: keys{"expert_count": u(4), "expert_used_count": u(2), "block_count": u(2),
			"expert_gating_func": u(1)}, refused: "router is sparsemixer"},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(tc.arch)}
		for k, v := range tc.keys {
			kv[tc.arch+"."+k] = v
		}
		var ts []meta.Tensor
		for _, n := range tc.tensors {
			ts = append(ts, meta.Tensor{Name: n})
		}
		nrot := tc.nrot
		if nrot == 0 {
			nrot = 8
		}
		c := &Config{Config: &jlm.Config{NLayer: 2, HeadDim: 8, NRot: nrot, RopeBase: 10000}}
		err := stateContainer(meta.New(tc.arch+".gguf", kv, ts), c)
		if tc.refused != "" {
			if err == nil || !strings.Contains(err.Error(), tc.refused) {
				t.Errorf("%s: %v, want a refusal naming %q", tc.name, err, tc.refused)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		rope := tc.rope
		if rope == 0 {
			rope = 10000
		}
		if c.Flags != tc.flags || c.RMSEps != tc.eps || c.LogitScale != tc.logit || c.EmbdScale != tc.scale ||
			c.ResidualScale != tc.residual || c.MoEStep != tc.step || c.RopeBase != rope ||
			c.SWAWindow != tc.win || c.SWAPeriod != tc.per || [3]float32{c.AttnTempScale, float32(c.AttnTempFloor), c.AttnTempOffset} != tc.temp {
			t.Errorf("%s: flags %v eps %g logits /%g scale %g residual %g step %d base %g window %d every %d temp %v; "+
				"want %v %g %g %g %g %d %g %d %d %v", tc.name, c.Flags, c.RMSEps, c.LogitScale, c.EmbdScale, c.ResidualScale,
				c.MoEStep, c.RopeBase, c.SWAWindow, c.SWAPeriod, [3]float32{c.AttnTempScale, float32(c.AttnTempFloor), c.AttnTempOffset},
				tc.flags, tc.eps, tc.logit, tc.scale, tc.residual, tc.step, rope, tc.win, tc.per, tc.temp)
		}
	}
}

// TestRouterStatementsAreChecked refuses a router gate and a
// renormalisation the container does not state, and an empty name among
// the keys and tensors a class has or lacks.
func TestRouterStatementsAreChecked(t *testing.T) {
	for _, tc := range []struct{ states, want string }{
		{`"x": {"router": {"gate": "relu"}}`, "router gate"},
		{`"x": {"router": {"renormalise": "sometimes"}}`, "not always or never"},
		{`"x": {"requires": [""]}`, "empty key or tensor name"},
		{`"x": {"lacks": [""]}`, "empty key or tensor name"},
		{`"x": {"mrope": "sections"}`, "is not the keys"},
		{`"x": {"attn_temp": {"scale": "0.1"}}`, "scale, floor and offset"},
	} {
		_, err := parseModelJSON([]byte(`{"gguf": ["x"], "container": {`+tc.states+`}}`), "test")
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.states, err, tc.want)
		}
	}
}
