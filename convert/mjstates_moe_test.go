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
// scales, the interleave and the rotary base. A refusal the reader made is
// held where it is still one, made by the generic rule that now makes it
// (an expression over a key the file lacks, a stated q/k norm or router
// the tensors contradict, a key contradicting the stated router or
// rotary); one the engine runs for another model converts (a dense block
// in a mixture, shared experts, a router with no selection bias).
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
		{name: "qwen2moe, a dense block", arch: "qwen2moe", keys: keys{"expert_count": u(4)},
			tensors: []string{"blk.0.ffn_gate_inp.weight"}, flags: neox | jlm.FlagNoExpertNorm},
		{name: "qwen2moe, no experts", arch: "qwen2moe",
			tensors: []string{"blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight"}, refused: "no expert_count"},
		{name: "qwen2moe, a sigmoid", arch: "qwen2moe", keys: keys{"expert_count": u(4), "expert_gating_func": u(2)},
			tensors: []string{"blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight"}, refused: "router is softmax"},
		{name: "ernie", arch: "ernie4_5-moe", keys: keys{"interleave_moe_layer_step": u(2)}, step: 2},
		{name: "ernie, no step", arch: "ernie4_5-moe", refused: "interleave_moe_layer_step']\" is undefined"},
		{name: "ernie, a step of 0", arch: "ernie4_5-moe", keys: keys{"interleave_moe_layer_step": u(0)}, refused: "moe_step is 0, not a step"},
		{name: "ernie, not renormalised", arch: "ernie4_5-moe",
			keys: keys{"interleave_moe_layer_step": u(1), "expert_weights_norm": meta.MakeBool(false)}, refused: "always renormalises"},
		{name: "dots1", arch: "dots1", keys: keys{"expert_count": u(4)}, tensors: norms, flags: qk | sig},
		{name: "dots1, a window", arch: "dots1", keys: keys{"expert_count": u(4), "attention.sliding_window": u(8)},
			tensors: norms, refused: "max_window_layers"},
		{name: "dots1, a window past the last layer", arch: "dots1", keys: keys{"expert_count": u(4),
			"attention.sliding_window": u(8), "max_window_layers": u(2)}, tensors: norms, flags: qk | sig},
		{name: "dots1, a window from the first layer", arch: "dots1", keys: keys{"expert_count": u(4),
			"attention.sliding_window": u(8), "max_window_layers": u(0)}, tensors: norms, flags: qk | sig, win: 8},
		{name: "dots1, a window from a middle layer", arch: "dots1", keys: keys{"expert_count": u(4),
			"attention.sliding_window": u(8), "max_window_layers": u(1)}, tensors: norms, refused: "follows no period"},
		{name: "dots1, a router over no experts", arch: "dots1", tensors: append([]string{"blk.1.ffn_gate_inp.weight"}, norms...),
			refused: "block 1 stores a router, and the file states no expert_count"},
		{name: "dots1, a partial rotary", arch: "dots1", keys: keys{"expert_count": u(4)}, tensors: norms, nrot: 4,
			refused: "turns the whole head"},
		{name: "bailingmoe2", arch: "bailingmoe2", keys: keys{"expert_count": u(4)}, tensors: norms, flags: qk | sig},
		{name: "bailingmoe2, a softmax", arch: "bailingmoe2", keys: keys{"expert_count": u(4), "expert_gating_func": u(1)},
			tensors: norms, refused: "router is sigmoid"},
		{name: "glm4moe-air", arch: "glm4moe", flags: neox | sig},
		{name: "glm4moe", arch: "glm4moe", tensors: norms, flags: qk | sig},
		{name: "glm4", arch: "glm4"},
		{name: "dbrx", arch: "dbrx", keys: keys{"attention.clamp_kqv": fl(8), "expert_count": u(4), "attention.layer_norm_epsilon": fl(1e-6)},
			flags: jlm.FlagLayerNorm | neox, eps: 1e-6},
		{name: "dbrx, no clamp", arch: "dbrx", keys: keys{"expert_count": u(4)}, refused: "attention.clamp_kqv']\" is undefined"},
		{name: "dbrx, a negative clamp", arch: "dbrx", keys: keys{"expert_count": u(4), "attention.clamp_kqv": fl(-8)},
			refused: "clamp_kqv is -8"},
		{name: "granite", arch: "granite", keys: keys{"logit_scale": fl(8), "embedding_scale": fl(12), "residual_scale": fl(0.25)},
			logit: 8, scale: 12, residual: 0.25},
		{name: "granitemoe", arch: "granitemoe", keys: keys{"logit_scale": fl(6)}, logit: 6, scale: 1},
		{name: "granite, no logit scale", arch: "granite", refused: "logit_scale']\" is undefined"},
		{name: "granite, a negative embedding scale", arch: "granite", keys: keys{"logit_scale": fl(8), "embedding_scale": fl(-1)},
			refused: "embd_scale is -1"},
		{name: "granite, an infinite embedding scale", arch: "granite", keys: keys{"logit_scale": fl(8),
			"embedding_scale": fl(float32(math.Inf(1)))}, refused: "embd_scale is +Inf"},
		{name: "minimax-m2", arch: "minimax-m2", keys: keys{"expert_count": u(4)},
			tensors: append([]string{"blk.0.exp_probs_b.bias", "blk.1.exp_probs_b.bias"}, norms...), flags: qk | jlm.FlagQKNormWide | sig},
		{name: "minimax-m2, a shared expert", arch: "minimax-m2", keys: keys{"expert_count": u(4), "expert_shared_count": u(1)},
			tensors: append([]string{"blk.0.exp_probs_b.bias", "blk.1.exp_probs_b.bias"}, norms...), flags: qk | jlm.FlagQKNormWide | sig},
		{name: "minimax-m2, no selection bias", arch: "minimax-m2", keys: keys{"expert_count": u(4)},
			tensors: norms, flags: qk | jlm.FlagQKNormWide | sig},
		{name: "minimax-m2, a softmax", arch: "minimax-m2", keys: keys{"expert_count": u(4), "expert_gating_func": u(1)},
			tensors: norms, refused: "router is sigmoid"},
		{name: "hunyuan-moe", arch: "hunyuan-moe", tensors: norms, flags: qk | jlm.FlagQKNormPostRope, rope: 10000},
		{name: "hunyuan-dense", arch: "hunyuan-dense", tensors: norms, flags: qk | jlm.FlagQKNormPostRope, rope: 10000},
		{name: "hunyuan-moe, a routed scale", arch: "hunyuan-moe", keys: keys{"expert_weights_scale": fl(2.5)},
			refused: "scales nothing"},
		{name: "hunyuan_vl", arch: "hunyuan_vl", keys: keys{"rope.freq_base": fl(10000), "rope.scaling.alpha": fl(1000),
			"rope.dimension_count": u(8), "rope.dimension_sections": meta.MakeInt32s([]int32{1, 1, 1, 1})}, tensors: norms,
			flags: qk | jlm.FlagQKNormPostRope, rope: float32(float64(float32(10000)) * math.Pow(1000, 8.0/6))},
		{name: "hunyuan_vl, the rotary's width off the head's", arch: "hunyuan_vl", keys: keys{"rope.scaling.alpha": fl(1000),
			"attention.key_length": u(8), "rope.dimension_sections": meta.MakeInt32s([]int32{1, 1, 1, 1})}, tensors: norms,
			flags: qk | jlm.FlagQKNormPostRope, rope: float32(10000 * math.Pow(1000, 8.0/6))},
		{name: "hunyuan_vl, a partial rotary", arch: "hunyuan_vl", keys: keys{"rope.dimension_sections": meta.MakeInt32s([]int32{1, 1, 0, 0})},
			tensors: norms, nrot: 4, refused: "turns the whole head"},
		{name: "apertus", arch: "apertus", tensors: norms, flags: qk},
		{name: "llama4 scout", arch: "llama4", keys: keys{"expert_count": u(16)},
			flags: jlm.FlagSWAChunked | jlm.FlagNoPEGlobal | jlm.FlagExpertWeightIn | jlm.FlagQKL2Norm | sig | jlm.FlagNoExpertNorm,
			win:   8192, per: 4, temp: [3]float32{0.1, 8192, 1}, step: 1},
		{name: "llama4 maverick", arch: "llama4", keys: keys{"expert_count": u(128), "interleave_moe_layer_step": u(2),
			"attention.chunk_size": u(4096), "attention.sliding_window_pattern": u(3), "attention.temperature_scale": fl(0.5),
			"attention.temperature_length": u(1024)},
			flags: jlm.FlagSWAChunked | jlm.FlagNoPEGlobal | jlm.FlagExpertWeightIn | sig | jlm.FlagNoExpertNorm,
			win:   4096, per: 3, temp: [3]float32{0.5, 1024, 1}, step: 2},
		{name: "llama4, every layer global", arch: "llama4", keys: keys{"attention.sliding_window": u(0)}, refused: "every layer attending globally"},
		{name: "llama4, a sliding_window beside the chunk", arch: "llama4", keys: keys{"expert_count": u(16), "attention.sliding_window": u(512)},
			flags: jlm.FlagSWAChunked | jlm.FlagNoPEGlobal | jlm.FlagExpertWeightIn | jlm.FlagQKL2Norm | sig | jlm.FlagNoExpertNorm,
			win:   8192, per: 4, temp: [3]float32{0.1, 8192, 1}, step: 1},
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
// renormalisation the container does not state, a refusal written in a
// file (requires), and a window stating two layouts.
func TestRouterStatementsAreChecked(t *testing.T) {
	for _, tc := range []struct{ states, want string }{
		{`"x": {"router": {"gate": "relu"}}`, "router gate"},
		{`"x": {"router": {"renormalise": "sometimes"}}`, "not always or never"},
		{`"x": {"requires": ["expert_count"]}`, "requires"},
		{`"x": {"window": {"size": "8", "period": "2", "from": "1"}}`, "one of a period, a pattern or a first layer"},
		{`"x": {"rope_sections": "sections"}`, "is not the keys"},
		{`"x": {"attn_temp": {"scale": "0.1"}}`, "its scale and its floor"},
	} {
		_, err := parseModelJSON([]byte(`{"gguf": ["x"], "container": {`+tc.states+`}}`), "test", testNames(t))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.states, err, tc.want)
		}
	}
}

// TestTheGenericRulesRefuseWhatTheReadersDid holds the refusals the deleted
// readers made that configOfFile now makes for every model: a rotary the
// file turns off where the graph states one, a router selecting more
// experts than the file has, and prediction layers with no block behind
// them.
func TestTheGenericRulesRefuseWhatTheReadersDid(t *testing.T) {
	for _, tc := range []struct {
		name, arch string
		keys       map[string]meta.Value
		refused    string
	}{
		{"granite, no rotary", "granite", map[string]meta.Value{"logit_scale": meta.MakeFloat(8),
			"rope.scaling.finetuned": meta.MakeBool(false)}, "rope.scaling.finetuned false"},
		{"llama, no rotary", "llama", map[string]meta.Value{"rope.scaling.finetuned": meta.MakeBool(false)},
			"rope.scaling.finetuned false"},
		{"phimoe, one expert", "phimoe", map[string]meta.Value{"expert_count": meta.MakeUint(1),
			"expert_used_count": meta.MakeUint(2)}, "expert_used_count=2"},
		{"glm4, prediction layers", "glm4", map[string]meta.Value{"nextn_predict_layers": meta.MakeUint(1)},
			"nextn_predict_layers is 1 and the file has no blk.2.nextn.eh_proj.weight"},
	} {
		k := baseKeys(3)
		for n, v := range tc.keys {
			k[n] = v
		}
		if _, err := configOf(headerOnly(tc.arch, k)); err == nil || !strings.Contains(err.Error(), tc.refused) {
			t.Errorf("%s: %v, want a refusal naming %q", tc.name, err, tc.refused)
		}
	}
	k := baseKeys(3)
	k["logit_scale"] = meta.MakeFloat(8)
	k["rope.scaling.finetuned"] = meta.MakeBool(true)
	if _, err := configOf(headerOnly("granite", k)); err != nil {
		t.Errorf("granite with its rotary: %v", err)
	}
}
