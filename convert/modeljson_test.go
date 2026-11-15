package convert

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

func mjEnv(m map[string]any) exprEnv {
	return func(p string) any {
		var cur any = m
		for _, k := range strings.Split(strings.TrimPrefix(p, "$."), ".") {
			mm, ok := cur.(map[string]any)
			if !ok {
				return nil
			}
			cur = mm[k]
		}
		return cur
	}
}

// TestModelExprEvaluates holds the expression language to its definition: the
// precedence, ?? on an absent key, short-circuit logic, in, and the errors
// that name the key a default was not written for.
func TestModelExprEvaluates(t *testing.T) {
	env := mjEnv(map[string]any{
		"head_dim": 256.0, "act": "silu", "gate": true, "secs": []any{11.0, 11.0, 10.0},
		"rope": map[string]any{"theta": 1e7, "partial": 0.25},
	})
	for _, c := range []struct {
		src  string
		want any
	}{
		{"head_dim * rope.partial", 64.0},
		{"1 + 2 * 3", 7.0},
		{"(1 + 2) * 3", 9.0},
		{"-head_dim / 4", -64.0},
		{"8 - 2 - 1", 5.0},
		{"missing ?? 1e-6", 1e-6},
		{"rope.theta ?? 10000", 1e7},
		{"rope.missing ?? missing ?? 0.25", 0.25},
		{"act != 'silu'", false},
		{"!(act in ['silu', 'swish'])", false},
		{"'gelu' in ['silu', 'swish']", false},
		{"gate == true && head_dim >= 256", true},
		{"missing != null && missing > 3", false},
		{"missing == null || missing > 3", true},
		{"secs ?? [1]", []any{11.0, 11.0, 10.0}},
		{"missing ?? [11, 11, 10]", []any{11.0, 11.0, 10.0}},
		{"$.head_dim", 256.0},
		{"secs == [11, 11, 10]", true},
		{"secs != [11, 11]", true},
		{"(missing ?? []) != []", false},
		{"null", nil},
		// One key spelled two ways that agree, and a default read from
		// another key, which is precedence.
		{"head_dim ?? $.head_dim", 256.0},
		{"head_dim ?? (rope.theta)", 256.0},
		{"missing ?? rope.theta", 1e7},
		{"(rope ?? $.rope) == rope", true},
	} {
		n, err := parseExpr(c.src)
		if err != nil {
			t.Fatalf("%q: %v", c.src, err)
		}
		got, err := evalMJ(n, env)
		if err != nil {
			t.Fatalf("%q: %v", c.src, err)
		}
		if l, ok := got.([]any); ok {
			if !slices.Equal(l, c.want.([]any)) {
				t.Errorf("%q = %v, want %v", c.src, got, c.want)
			}
			continue
		}
		if got != c.want {
			t.Errorf("%q = %v, want %v", c.src, got, c.want)
		}
	}
	for _, c := range []struct{ src, err string }{
		{"missing * 2", "missing is absent"},
		{"act + 1", "act is a string"},
		{"1 / 0", "division by zero"},
		{"gate > 1", "gate is a boolean"},
		{"act in head_dim", "wants a list"},
		{"head_dim ?? rope.theta", "spelled two ways"},
		{"rope.partial ?? head_dim ?? 1", "spelled two ways"},
		{"exp(head_dim)", "calls are a tensor map's"},
	} {
		n, err := parseExpr(c.src)
		if err != nil {
			t.Fatalf("%q: %v", c.src, err)
		}
		if _, err := evalMJ(n, env); err == nil || !strings.Contains(err.Error(), c.err) {
			t.Errorf("%q: error %v, want one saying %q", c.src, err, c.err)
		}
	}
	for _, src := range []string{"", "1 +", "(1", "[1, 2", "'open", "a ^ b", "a..b", "1 2", "in", strings.Repeat("(", 100) + "1" + strings.Repeat(")", 100)} {
		if _, err := parseExpr(src); err == nil {
			t.Errorf("%q parsed", src)
		}
	}
}

// FuzzModelExpr: no expression panics or runs away, whatever its text.
// Run with an exec budget (RULE 9):
//
//	go test ./convert -run=XXX -fuzz=FuzzModelExpr -fuzztime=1000000x -parallel=2
func FuzzModelExpr(f *testing.F) {
	for _, s := range []string{"head_dim * (a.b ?? 0.25)", "!(x in ['a', 'b']) && y != null", "[1, [2, 3]] ?? -4e-2"} {
		f.Add(s)
	}
	env := mjEnv(map[string]any{"head_dim": 2.0, "x": "a", "a": map[string]any{"b": 0.5}})
	f.Fuzz(func(t *testing.T, s string) {
		if n, err := parseExpr(s); err == nil {
			evalMJ(n, env)
		}
	})
}

// TestModelLibraryParses: every file in convert/models parses, names classes
// no other file or the Go table names, lowers to an entry, and is in the list
// the docs are generated from.
func TestModelLibraryParses(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	if len(lib) == 0 {
		t.Fatal("the library is empty -- this gate proved nothing")
	}
	classes := HFClasses()
	for c, mj := range lib {
		if _, err := mj.hfArch(); err != nil {
			t.Errorf("%s: %v", c, err)
		}
		if !slices.Contains(classes, c) {
			t.Errorf("%s is in the library and not in HFClasses", c)
		}
		if g, ok := HFClassGraph(c); !ok || g == ArchNone {
			t.Errorf("%s: HFClassGraph %v %v", c, g, ok)
		}
	}
	if g, _ := HFClassGraph("Qwen3_5ForConditionalGeneration"); g != ArchQwen3Next {
		t.Errorf("Qwen3_5ForConditionalGeneration converts to %v", g)
	}
}

// TestModelJSONRefuses: a file that names what the converter does not know
// is refused at parse or lowering, never read as something else.
func TestModelJSONRefuses(t *testing.T) {
	const ok = `{"classes": ["X"], "config": {"arch": "qwen3next", "vocab": "vocab_size",
		"head": {"tensor": "lm_head.weight"}, "activation": "'silu'"}, "tensors": {"block_prefix": "model.layers.",
		"block": {"a.weight": "attn_q.weight"}}}`
	if mj, err := parseModelJSON([]byte(ok), "ok"); err != nil {
		t.Fatalf("the control does not parse: %v", err)
	} else if _, err := mj.hfArch(); err != nil {
		t.Fatalf("the control does not lower: %v", err)
	}
	for _, c := range []struct{ name, from, to, err string }{
		{"unknown field", `"classes"`, `"klasses": [], "classes"`, "unknown field"},
		{"no class", `["X"]`, `[]`, "names no class"},
		{"unknown graph", `"qwen3next"`, `"nonesuch"`, "not one configOf builds"},
		{"unknown gate", `"activation": "'silu'"`, `"activation": "'silu'", "gating": "sparsemixer"`, "gating \"sparsemixer\""},
		{"no vocab", `"vocab": "vocab_size",`, ``, "no vocab"},
		{"unknown role", `"attn_q.weight"`, `"attn_quux.weight"`, "not a tensor name"},
		{"unknown primitive", `"attn_q.weight"`, `{"role": "attn_q.weight", "do": [{"frobnicate": 1}]}`, "unknown field"},
		{"a recipe by name", `"attn_q.weight"`, `{"role": "attn_q.weight", "do": ["plus_one"]}`, "cannot unmarshal"},
		{"two primitives in a step", `"attn_q.weight"`, `{"role": "attn_q.weight", "do": [{"reshape": [-1], "map": "x"}]}`, "one primitive a step"},
		{"a map that is no arithmetic", `"attn_q.weight"`, `{"role": "attn_q.weight", "do": [{"map": "x == 1"}]}`, "not arithmetic"},
		{"no activation", `, "activation": "'silu'"`, ``, "states no activation"},
		{"an unknown flag", `"activation": "'silu'"`, `"activation": "'silu'", "flags": {"gated": "true"}`, "not a graph feature"},
		{"unknown expert bank", `"block": {"a.weight": "attn_q.weight"}`,
			`"block": {"a.weight": "attn_q.weight"}, "experts": {"prefix": "e.", "names": {"w.weight": "ffn_quux_exps.weight"}}`,
			"bank \"ffn_quux_exps.weight\""},
	} {
		src := strings.Replace(ok, c.from, c.to, 1)
		if src == ok {
			t.Fatalf("%s: the mutation changed nothing", c.name)
		}
		mj, err := parseModelJSON([]byte(src), c.name)
		if err == nil {
			_, err = mj.hfArch()
		}
		if err == nil || !strings.Contains(err.Error(), c.err) {
			t.Errorf("%s: %v, want an error saying %q", c.name, err, c.err)
		}
	}
}

// TestLibraryIgnoresTheRotaryBuffers: the files that replaced llama-named Go
// entries ignore the rotary inverse frequencies a checkpoint may carry, the
// model's and each layer's (older transformers saved them), as those entries
// did: a derived buffer is named and not carried, never refused.
func TestLibraryIgnoresTheRotaryBuffers(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []string{"Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "GemmaForCausalLM",
		"Ernie4_5ForCausalLM", "Qwen2MoeForCausalLM", "Ernie4_5_MoeForCausalLM", "Dots1ForCausalLM",
		"Glm4MoeForCausalLM", "LlamaForCausalLM"} {
		mj := lib[c]
		if mj == nil {
			t.Fatalf("no %s in the library", c)
		}
		a, err := mj.hfArch()
		if err != nil {
			t.Fatal(err)
		}
		for _, n := range []string{"model.rotary_emb.inv_freq", "model.layers.3.self_attn.rotary_emb.inv_freq"} {
			if _, _, _, ignored, err := a.identify(n, nil); err != nil || !ignored {
				t.Errorf("%s: %s is not ignored (%v)", c, n, err)
			}
		}
	}
}

// TestModelJSONHandsConfigOfTheTensorTable: configOf reads which blocks route
// off the tensor table, and a file's config hands it the checkpoint's.
// Qwen1.5-MoE states its shared expert by key and every block routes: a block
// without its router is refused as its GGUF is, an expert's matrix stands in
// for its bank with its shape, and the stated softmax router refuses a
// config.json naming another gate.
func TestModelJSONHandsConfigOfTheTensorTable(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["Qwen2MoeForCausalLM"]
	if mj == nil {
		t.Fatal("no Qwen2MoeForCausalLM in the library")
	}
	js := `{"architectures": ["Qwen2MoeForCausalLM"], "vocab_size": 100, "hidden_size": 64,
	  "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
	  "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
	  "shared_expert_intermediate_size": 48}`
	c := hfConfigOf(t, js)
	names := []string{"lm_head.weight", "model.layers.0.mlp.experts.3.gate_proj.weight",
		"model.layers.1.mlp.experts.0.gate_proj.weight", "model.layers.0.mlp.gate.weight",
		"model.layers.1.mlp.gate.weight"}
	cfg, err := mj.config(c, names)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Arch != ArchQwen2MoE || cfg.NExpert != 8 || cfg.NExpertUsed != 2 || cfg.NFFNExp != 32 ||
		cfg.NFFNShExp != 48 || !cfg.Flags.Has(jlm.FlagNoExpertNorm) || !cfg.Flags.Has(jlm.FlagRopeNeox) ||
		cfg.Flags.Has(jlm.FlagExpertSigmoid) {
		t.Errorf("mixture %d of %d experts %d wide, shared %d, flags %v", cfg.NExpertUsed, cfg.NExpert,
			cfg.NFFNExp, cfg.NFFNShExp, cfg.Flags)
	}
	if _, err := mj.config(c, names[:4]); err == nil || !strings.Contains(err.Error(), "block 1 has no router") {
		t.Errorf("a block without its router: %v", err)
	}
	_, err = mj.config(hfConfigOf(t, strings.Replace(js, `"num_experts"`, `"scoring_func": "sigmoid", "num_experts"`, 1)), names)
	if err == nil || !strings.Contains(err.Error(), "scoring_func sigmoid") || !errors.Is(err, ErrNotImplemented) {
		t.Errorf("a sigmoid scoring_func on a softmax router: %v", err)
	}
	// The width read off a bank, where the file states none: an expert's
	// [24, 64] matrix is the bank's {64, 24}.
	mj2 := *mj
	mj2.Config.Keys = maps.Clone(mj.Config.Keys)
	delete(mj2.Config.Keys, "expert_feed_forward_length")
	c.shapes = map[string][]uint64{"model.layers.0.mlp.experts.3.gate_proj.weight": {24, 64}}
	if cfg, err := mj2.config(c, names); err != nil || cfg.NFFNExp != 24 {
		t.Errorf("the width off the bank: %v, %v", cfg, err)
	}
}

// TestModelJSONHoldsMixturesToTheStep: a mixture's blocks carry experts and a
// router past the dense lead and on the step its keys state, and a checkpoint
// whose weights stand elsewhere is refused, for every file. ERNIE 4.5's lead
// of one and step of two route blocks 1 and 3 of four. Its shared experts'
// width is the shared expert's tensor's, as its GGUF's is.
func TestModelJSONHoldsMixturesToTheStep(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["Ernie4_5_MoeForCausalLM"]
	if mj == nil {
		t.Fatal("no Ernie4_5_MoeForCausalLM in the library")
	}
	c := hfConfigOf(t, `{"architectures": ["Ernie4_5_MoeForCausalLM"], "vocab_size": 100, "hidden_size": 64,
	  "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
	  "head_dim": 16, "moe_num_experts": 8, "moe_k": 2, "moe_intermediate_size": 32, "moe_num_shared_experts": 2,
	  "moe_layer_start_index": 1, "moe_layer_interval": 2, "moe_layer_end_index": 3}`)
	sh := "model.layers.1.mlp.shared_experts.gate_proj.weight"
	c.shapes = map[string][]uint64{sh: {64, 64}}
	at := func(b int) []string {
		p := "model.layers." + itoa(b) + ".mlp."
		return []string{p + "gate.weight", p + "experts.0.gate_proj.weight"}
	}
	names := slices.Concat([]string{sh}, at(1), at(3))
	cfg, err := mj.config(c, names)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.NDenseLead != 1 || cfg.MoEStep != 2 || cfg.NExpert != 8 || cfg.NExpertUsed != 2 || cfg.NFFNExp != 32 ||
		cfg.NFFNShExp != 64 || !cfg.Flags.Has(jlm.FlagTiedEmbd) || cfg.Flags.Has(jlm.FlagNoExpertNorm) {
		t.Errorf("lead %d step %d, %d of %d experts %d wide, shared %d, flags %v", cfg.NDenseLead, cfg.MoEStep,
			cfg.NExpertUsed, cfg.NExpert, cfg.NFFNExp, cfg.NFFNShExp, cfg.Flags)
	}
	for _, r := range []struct {
		name  string
		names []string
		err   string
	}{
		{"a mixture off the step", slices.Concat(names, at(2)), "block 2 carries experts, where"},
		{"a router alone", append(slices.Clone(names), "model.layers.2.mlp.gate.weight"), "block 2 carries a router, where"},
		{"a mixture missing", slices.Concat([]string{sh}, at(1)), "block 3 carries no experts, where"},
	} {
		if _, err := mj.config(c, r.names); err == nil || !strings.Contains(err.Error(), r.err) || !errors.Is(err, ErrNotImplemented) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
}

// TestDots1ThroughTheLibrary: dots.llm1's DeepSeek-V3 router through
// convert/models/dots1.json -- a sigmoid gate, a routed scale, groups, a dense
// lead and shared experts as wide as their tensor -- and its refusals: a
// softmax scoring_func, groups the experts do not divide into, a block 0
// without its q/k norm, and a router keys without a default leave absent.
func TestDots1ThroughTheLibrary(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["Dots1ForCausalLM"]
	if mj == nil {
		t.Fatal("no Dots1ForCausalLM in the library")
	}
	js := `{"architectures": ["Dots1ForCausalLM"], "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
	  "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 16,
	  "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "n_shared_experts": 2,
	  "first_k_dense_replace": 1, "norm_topk_prob": true, "routed_scaling_factor": 2.5, "n_group": 1,
	  "topk_group": 1, "scoring_func": "sigmoid",
	  "sliding_window": null, "layer_types": ["full_attention", "full_attention", "full_attention"]}`
	sh := "model.layers.1.mlp.shared_experts.gate_proj.weight"
	names := []string{"lm_head.weight", "model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight", sh}
	for _, b := range []string{"1", "2"} {
		p := "model.layers." + b + ".mlp."
		names = append(names, p+"gate.weight", p+"experts.0.gate_proj.weight")
	}
	noQK := append([]string{names[0]}, names[3:]...)
	c := hfConfigOf(t, js)
	c.shapes = map[string][]uint64{sh: {64, 64}}
	cfg, err := mj.config(c, names)
	if err != nil {
		t.Fatal(err)
	}
	want := jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagExpertSigmoid
	if cfg.Arch != ArchDots1 || cfg.Flags&want != want || cfg.Flags.Has(jlm.FlagNoExpertNorm) || cfg.ExpertScale != 2.5 ||
		cfg.NDenseLead != 1 || cfg.NExpert != 8 || cfg.NExpertUsed != 2 || cfg.NFFNExp != 32 || cfg.NFFNShExp != 64 ||
		cfg.NExpertGroup != 1 || cfg.NExpertGroupUsed != 1 {
		t.Errorf("flags %v scale %v lead %d, %d of %d experts %d wide, shared %d, groups %d/%d", cfg.Flags,
			cfg.ExpertScale, cfg.NDenseLead, cfg.NExpertUsed, cfg.NExpert, cfg.NFFNExp, cfg.NFFNShExp,
			cfg.NExpertGroup, cfg.NExpertGroupUsed)
	}
	for _, r := range []struct {
		name, from, to string
		names          []string
		err            string
	}{
		{"softmax", `"scoring_func": "sigmoid"`, `"scoring_func": "softmax"`, names, "scoring_func softmax"},
		{"uneven groups", `"n_group": 1`, `"n_group": 3`, names, "8 experts in 3 groups"},
		{"no q/k norm", `"n_group": 1`, `"n_group": 1`, noQK, "attn_q_norm"},
		{"no expert count", `"n_routed_experts": 8,`, ``, names, "n_routed_experts"},
		{"a window in use", `"sliding_window": null, "layer_types": ["full_attention"`,
			`"sliding_window": 16, "layer_types": ["sliding_attention"`, names, "window"},
	} {
		c := hfConfigOf(t, strings.Replace(js, r.from, r.to, 1))
		c.shapes = map[string][]uint64{sh: {64, 64}}
		if _, err := mj.config(c, r.names); err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
}

// TestGlm4MoeThroughTheLibrary: GLM-4.5 through convert/models/glm4moe.json --
// half of each head rotated, the q/k norm present where block 0 ships it, and
// its prediction block counted where the checkpoint ships it and not where it
// does not -- and a softmax scoring_func refused.
func TestGlm4MoeThroughTheLibrary(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["Glm4MoeForCausalLM"]
	if mj == nil {
		t.Fatal("no Glm4MoeForCausalLM in the library")
	}
	js := `{"architectures": ["Glm4MoeForCausalLM"], "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
	  "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
	  "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "n_shared_experts": 1,
	  "first_k_dense_replace": 1, "routed_scaling_factor": 2.5, "num_nextn_predict_layers": 1,
	  "rope_parameters": {"rope_type": "default", "rope_theta": 1000000, "partial_rotary_factor": 0.5}}`
	trunk := []string{"lm_head.weight"}
	for _, b := range []string{"1", "2", "3"} {
		p := "model.layers." + b + ".mlp."
		trunk = append(trunk, p+"gate.weight", p+"experts.0.gate_proj.weight")
	}
	qk := []string{"model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"}
	mtp := []string{"model.layers.3.eh_proj.weight", "model.layers.3.enorm.weight", "model.layers.3.hnorm.weight"}
	for _, r := range []struct {
		name  string
		names []string
		mtp   uint32
		qk    bool
	}{
		{"whole", slices.Concat(trunk, qk, mtp), 1, true},
		{"no prediction block", slices.Concat(trunk, qk), 0, true},
		{"no q/k norm", slices.Concat(trunk, mtp), 1, false},
	} {
		cfg, err := mj.config(hfConfigOf(t, js), r.names)
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		want := jlm.FlagRopeNeox | jlm.FlagExpertSigmoid
		if cfg.Arch != ArchGLM4MoE || cfg.NLayer != 3 || cfg.NMTP != r.mtp || cfg.NRot != 8 || cfg.HeadDim != 16 ||
			cfg.Flags&want != want || cfg.Flags.Has(jlm.FlagQKNorm) != r.qk || cfg.NDenseLead != 1 ||
			cfg.ExpertScale != 2.5 || cfg.NFFNShExp != 32 || cfg.Flags.Has(jlm.FlagNoExpertNorm) {
			t.Errorf("%s: %d blocks and %d predicting, rotary %d of %d, lead %d, scale %v, shared %d, flags %v",
				r.name, cfg.NLayer, cfg.NMTP, cfg.NRot, cfg.HeadDim, cfg.NDenseLead, cfg.ExpertScale, cfg.NFFNShExp, cfg.Flags)
		}
	}
	_, err = mj.config(hfConfigOf(t, strings.Replace(js, `"n_shared_experts"`, `"scoring_func": "softmax", "n_shared_experts"`, 1)), trunk)
	if err == nil || !strings.Contains(err.Error(), "scoring_func softmax") {
		t.Errorf("a softmax scoring_func on a sigmoid router: %v", err)
	}
}

// qwen35TestConfig is a Qwen3_5ForConditionalGeneration config.json with a
// head of 32, a quarter of it rotated in sections [2, 1, 1], and four value
// heads over two key heads; edit replaces text in it.
func qwen35TestConfig(t *testing.T, edit ...string) *hfConfig {
	t.Helper()
	js := `{"architectures": ["Qwen3_5ForConditionalGeneration"], "tie_word_embeddings": false,
	  "text_config": {"vocab_size": 100, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
	    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "hidden_act": "silu",
	    "attn_output_gate": true, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32,
	    "linear_value_head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
	    "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
	    "rms_norm_eps": 1e-6, "max_position_embeddings": 512,
	    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
	      "mrope_section": [2, 1, 1], "mrope_interleaved": true}}}`
	for i := 0; i+1 < len(edit); i += 2 {
		if !strings.Contains(js, edit[i]) {
			t.Fatalf("the edit %q matches nothing", edit[i])
		}
		js = strings.Replace(js, edit[i], edit[i+1], 1)
	}
	return hfConfigOf(t, js)
}

// TestQwen35ConfigThroughTheLibrary: Qwen3.5's config.json, read through
// convert/models/qwen3_5.json, states what qwen35Of and configOf state from
// llama.cpp's GGUF of the same model, and each refusal fires.
func TestQwen35ConfigThroughTheLibrary(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["Qwen3_5ForConditionalGeneration"]
	if mj == nil {
		t.Fatal("no Qwen3_5ForConditionalGeneration in the library")
	}
	names := []string{"lm_head.weight"}
	c, err := mj.config(qwen35TestConfig(t), names)
	if err != nil {
		t.Fatal(err)
	}
	want := jlm.SSMConfig{ConvKernel: 4, Groups: 2, Inner: 128, StateSize: 32, NHeadV: 4}
	kinds := []jlm.LayerKind{jlm.LayerLinearAttn, jlm.LayerLinearAttn, jlm.LayerFullAttn, jlm.LayerLinearAttn}
	switch {
	case c.Arch != ArchQwen3Next:
		t.Errorf("arch %v", c.Arch)
	case c.NLayer != 4 || c.NEmbd != 64 || c.NFFN != 96 || c.NHead != 4 || c.NKVHead != 2 || c.HeadDim != 32:
		t.Errorf("geometry %+v", c.Config)
	case c.NRot != 8 || c.RopeSections != [4]uint32{2, 1, 1, 0} || c.RopeBase != 1e7 || c.NCtx != 512:
		t.Errorf("rotary: %d pairs %v base %v ctx %d", c.NRot, c.RopeSections, c.RopeBase, c.NCtx)
	case c.RMSEps != float32(1e-6):
		t.Errorf("eps %v", c.RMSEps)
	case c.SSM != want:
		t.Errorf("recurrence %+v, want %+v", c.SSM, want)
	case !slices.Equal(c.LayerKinds, kinds):
		t.Errorf("kinds %v, want %v", c.LayerKinds, kinds)
	case !c.Flags.Has(jlm.FlagRopeNeox) || !c.Flags.Has(jlm.FlagQKNorm) || !c.Flags.Has(jlm.FlagAttnOutGate):
		t.Errorf("flags %v", c.Flags)
	case c.Flags.Has(jlm.FlagDeltaKeyTiled):
		t.Error("the value heads are transformers' grouped order, and the container says tiled")
	case c.Flags.Has(jlm.FlagTiedEmbd):
		t.Error("tied, with lm_head.weight present")
	}
	// The class defaults: an absent eps, context and sections are Qwen3.5's.
	c, err = mj.config(qwen35TestConfig(t, `"rms_norm_eps": 1e-6, "max_position_embeddings": 512,`, ``,
		`"mrope_section": [2, 1, 1], `, ``, `"head_dim": 32`, `"head_dim": 256`), names)
	if err != nil {
		t.Fatal(err)
	}
	if c.RMSEps != float32(1e-6) || c.NCtx != 32768 || c.RopeSections != [4]uint32{11, 11, 10, 0} || c.NRot != 64 {
		t.Errorf("defaults: eps %v ctx %d sections %v rot %d", c.RMSEps, c.NCtx, c.RopeSections, c.NRot)
	}
	for _, r := range []struct {
		name  string
		edit  []string
		names []string
		err   string
	}{
		{"gelu", []string{`"hidden_act": "silu"`, `"hidden_act": "gelu"`}, names, "activation gelu is not implemented"},
		{"yarn", []string{`"rope_type": "default"`, `"rope_type": "yarn"`}, names, "rope scaling yarn is not implemented"},
		{"two rotary fractions", []string{`"hidden_act": "silu"`, `"hidden_act": "silu", "partial_rotary_factor": 0.5`}, names, "spelled two ways"},
		{"a window the file does not read", []string{`"hidden_act": "silu"`, `"hidden_act": "silu", "sliding_window": 4`}, names, "does not read it"},
		{"two thetas", []string{`"hidden_act": "silu"`, `"hidden_act": "silu", "rope_theta": 5`}, names, "disagree"},
		{"short kinds", []string{`"full_attention", "linear_attention"]`, `"full_attention"]`}, names, "3 entries in a 4-block"},
		{"unknown kind", []string{`"full_attention"`, `"sliding_attention"`}, names, "does not name"},
		{"absent heads", []string{`"num_attention_heads": 4,`, ``}, names, "is absent"},
		{"no head", nil, nil, "missing, not tied"},
		{"tied with a head", []string{`"tie_word_embeddings": false`, `"tie_word_embeddings": true`}, names, "guess"},
		{"sections over the rotary", []string{`[2, 1, 1]`, `[2, 2, 1]`}, names, "rotary pairs"},
	} {
		_, err := mj.config(qwen35TestConfig(t, r.edit...), r.names)
		if err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
	if _, err := mj.config(qwen35TestConfig(t, `"hidden_act": "silu"`, `"hidden_act": "gelu"`), names); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("a refused graph is %v, not ErrNotImplemented", err)
	}
}

// mjF32 is a float32 tensor of shape s holding vs.
func mjF32(s []uint64, vs ...float32) mjVal {
	return mjVal{shape: s, typ: jlm.TypeF32, data: f32AsBytes(vs)}
}

// mjRun runs a "do" list, as a file writes it, on vs.
func mjRun(t *testing.T, env exprEnv, do string, vs ...mjVal) (mjVal, error) {
	t.Helper()
	var p mjPipe
	if err := json.Unmarshal([]byte(do), &p); err != nil {
		t.Fatalf("%s: %v", do, err)
	}
	if err := p.check(); err != nil {
		return mjVal{}, err
	}
	return p.run(env, vs)
}

// TestModelJSONPrimitives holds each tensor primitive to its definition, and
// to a violation: a shape it cannot take is refused, never read as another.
func TestModelJSONPrimitives(t *testing.T) {
	env := mjEnv(map[string]any{"heads": 2.0, "kernel": 3.0})
	f32 := func(v mjVal) []float32 {
		t.Helper()
		if v.typ != jlm.TypeF32 {
			t.Fatalf("a %v result", v.typ)
		}
		return mjFloats(v)
	}
	for _, c := range []struct {
		name, do string
		in       []mjVal
		shape    []uint64
		want     []float32
	}{
		{"map x + 1", `[{"map": "x + 1"}]`, []mjVal{mjF32([]uint64{3}, 0, 0.5, -1)}, []uint64{3}, []float32{1, 1.5, 0}},
		{"map -exp(x)", `[{"map": "-exp(x)"}]`, []mjVal{mjF32([]uint64{2}, 0, float32(math.Log(2)))}, []uint64{2}, []float32{-1, -2}},
		{"map over two", `[{"map": "x * y + heads"}]`, []mjVal{mjF32([]uint64{2}, 1, 2), mjF32([]uint64{2}, 3, 4)}, []uint64{2}, []float32{5, 10}},
		{"reshape", `[{"reshape": [-1, "kernel"]}]`, []mjVal{mjF32([]uint64{2, 1, 3}, 0, 1, 2, 3, 4, 5)}, []uint64{2, 3}, []float32{0, 1, 2, 3, 4, 5}},
		{"transpose", `[{"transpose": [1, 0]}]`, []mjVal{mjF32([]uint64{2, 3}, 0, 1, 2, 3, 4, 5)}, []uint64{3, 2}, []float32{0, 3, 1, 4, 2, 5}},
		{"slice", `[{"slice": [1, 1, 2]}]`, []mjVal{mjF32([]uint64{2, 3}, 0, 1, 2, 3, 4, 5)}, []uint64{2, 2}, []float32{1, 2, 4, 5}},
		{"concat", `[{"concat": 1}]`, []mjVal{mjF32([]uint64{2, 1}, 0, 1), mjF32([]uint64{2, 2}, 2, 3, 4, 5)}, []uint64{2, 3}, []float32{0, 2, 3, 1, 4, 5}},
		{"stack", `[{"stack": 0}]`, []mjVal{mjF32([]uint64{2}, 0, 1), mjF32([]uint64{2}, 2, 3)}, []uint64{2, 2}, []float32{0, 1, 2, 3}},
		{"stack inner", `[{"stack": 1}]`, []mjVal{mjF32([]uint64{2}, 0, 1), mjF32([]uint64{2}, 2, 3)}, []uint64{2, 2}, []float32{0, 2, 1, 3}},
		{"repeat", `[{"repeat": [0, "heads"]}]`, []mjVal{mjF32([]uint64{2, 1}, 7, 8)}, []uint64{4, 1}, []float32{7, 7, 8, 8}},
		// Four value heads over two key heads, rows of two (Qwen3.5's beta
		// and alpha): per key head, its two heads' first rows, then their
		// second's.
		{"an interleave by group", `[{"stack": 0}, {"reshape": [2, "heads", -1]}, {"transpose": [1, 0, 2]}, {"reshape": ["2 * 4", -1]}]`,
			[]mjVal{mjF32([]uint64{8}, 0, 0, 1, 1, 2, 2, 3, 3), mjF32([]uint64{8}, 10, 10, 11, 11, 12, 12, 13, 13)},
			[]uint64{8, 2}, []float32{0, 0, 1, 1, 10, 10, 11, 11, 2, 2, 3, 3, 12, 12, 13, 13}},
		// Two heads of four rows (llama's rotary order): each head's rows i
		// and i + 2 become adjacent.
		{"a pair order by head", `[{"reshape": ["heads", 2, -1, 1]}, {"transpose": [0, 2, 1, 3]}, {"reshape": [-1, 1]}]`,
			[]mjVal{mjF32([]uint64{8, 1}, 0, 1, 2, 3, 4, 5, 6, 7)}, []uint64{8, 1}, []float32{0, 2, 1, 3, 4, 6, 5, 7}},
	} {
		got, err := mjRun(t, env, c.do, c.in...)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if !slices.Equal(got.shape, c.shape) || !slices.Equal(f32(got), c.want) {
			t.Errorf("%s: %v %v, want %v %v", c.name, got.shape, f32(got), c.shape, c.want)
		}
		// A dry run states the same shape without bytes.
		dry := make([]mjVal, len(c.in))
		for i, v := range c.in {
			dry[i] = mjVal{shape: v.shape, typ: v.typ}
		}
		if d, err := mjRun(t, env, c.do, dry...); err != nil || !slices.Equal(d.shape, c.shape) || d.data != nil {
			t.Errorf("%s: dry %v %v", c.name, d.shape, err)
		}
	}
	// Moves are by element: an f16 tensor transposes as two-byte elements,
	// and a map or a type reads it as its values.
	h := mjVal{shape: []uint64{2, 2}, typ: jlm.TypeF16}
	for _, f := range []float32{1, 2, 3, 4} {
		h.data = binary.LittleEndian.AppendUint16(h.data, quant.EncodeHalf(f))
	}
	if got, err := mjRun(t, env, `[{"transpose": [1, 0]}, {"type": "f32"}]`, h); err != nil || !slices.Equal(f32(got), []float32{1, 3, 2, 4}) {
		t.Errorf("f16 transpose: %v %v", got, err)
	}
	if got, err := mjRun(t, env, `[{"map": "x + 1"}]`, h); err != nil || !slices.Equal(f32(got), []float32{2, 3, 4, 5}) {
		t.Errorf("f16 map: %v %v", got, err)
	}
	if got, err := mjRun(t, env, `[{"type": "f16"}]`, mjF32([]uint64{2}, 1, 2)); err != nil || got.typ != jlm.TypeF16 || !slices.Equal(got.data, h.data[:4]) {
		t.Errorf("type f16: %v %v", got, err)
	}
	two := []mjVal{mjF32([]uint64{2}, 0, 1), mjF32([]uint64{3}, 2, 3, 4)}
	for _, c := range []struct {
		name, do string
		in       []mjVal
		err      string
	}{
		{"a reshape that loses elements", `[{"reshape": [4]}]`, two[:1], "2 elements into 4"},
		{"a reshape of two unknowns", `[{"reshape": [-1, -1]}]`, two[:1], "a size of -1"},
		{"a reshape that does not divide", `[{"reshape": [-1, "kernel"]}]`, two[:1], "do not divide"},
		{"a transpose that is no permutation", `[{"transpose": [0, 0]}]`, []mjVal{mjF32([]uint64{1, 2}, 0, 1)}, "not a permutation"},
		{"a transpose of the wrong rank", `[{"transpose": [1, 0]}]`, two[:1], "1-axis"},
		{"a slice past the end", `[{"slice": [0, 1, 2]}]`, two[:1], "slice [1, 3)"},
		{"a slice of a missing axis", `[{"slice": [1, 0, 1]}]`, two[:1], "axis 1 of a 1-axis"},
		{"a concat of unequal rows", `[{"concat": 0}]`, []mjVal{mjF32([]uint64{1, 2}, 0, 1), mjF32([]uint64{1, 3}, 0, 1, 2)}, "concat on axis 0"},
		{"a stack of unequal shapes", `[{"stack": 0}]`, two, "stack of"},
		{"a map of unequal sizes", `[{"map": "x + y"}]`, two, "over"},
		{"a map of a missing tensor", `[{"map": "x + y"}]`, two[:1], "tensor 2 of 1"},
		{"a map of an unknown function", `[{"map": "erf(x)"}]`, two[:1], "not a function"},
		{"a map of a comparison", `[{"map": "x > 1"}]`, two[:1], "not arithmetic"},
		{"a repeat of zero", `[{"repeat": [0, 0]}]`, two[:1], "repeat 0 times"},
		{"one tensor's step on two", `[{"reshape": [-1]}]`, two, "2 are in hand"},
		{"two tensors left", `[]`, two, "2 tensors are left"},
		{"two primitives in a step", `[{"reshape": [-1], "transpose": [0]}]`, two[:1], "one primitive a step"},
		{"no primitive in a step", `[{}]`, two[:1], "one primitive a step"},
		{"an unknown type", `[{"type": "bf16"}]`, two[:1], "not f32 or f16"},
		{"a slice of two values", `[{"slice": [0, 1]}]`, two[:1], "takes 3 values"},
		{"a size that is no count", `[{"reshape": ["kernel / 2", -1]}]`, two[:1], "not a whole number"},
		{"a size of an absent key", `[{"reshape": ["missing", -1]}]`, two[:1], "is absent"},
	} {
		if _, err := mjRun(t, env, c.do, c.in...); err == nil || !strings.Contains(err.Error(), c.err) {
			t.Errorf("%s: %v, want an error saying %q", c.name, err, c.err)
		}
	}
	var p mjPipe
	if err := json.Unmarshal([]byte(`["plus_one"]`), &p); err == nil {
		t.Error("a transform named by a string, the recipe vocabulary, parsed")
	}
}

// TestModelJSONStatements: activation, flags, window and rope scaling are
// read by the generic code and decide after configOf; what it cannot express
// it refuses by name.
func TestModelJSONStatements(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["Qwen3_5ForConditionalGeneration"]
	names := []string{"lm_head.weight"}
	c, err := mj.config(qwen35TestConfig(t, `"attn_output_gate": true`, `"attn_output_gate": false`), names)
	if err != nil || c.Flags.Has(jlm.FlagAttnOutGate) {
		t.Errorf("attn_output_gate false: %v, gate flag %v", err, err == nil && c.Flags.Has(jlm.FlagAttnOutGate))
	}
	// A mixture some block of which carries no experts is refused; every block
	// carrying them is the control.
	moe := lib["Qwen3MoeForCausalLM"]
	moeCfg := hfConfigOf(t, `{"architectures": ["Qwen3MoeForCausalLM"], "vocab_size": 100, "hidden_size": 64,
	  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "num_experts": 4,
	  "num_experts_per_tok": 2, "moe_intermediate_size": 32, "intermediate_size": 96}`)
	exp := func(b string) string { return "model.layers." + b + ".mlp.experts.0.up_proj.weight" }
	if _, err := moe.config(moeCfg, []string{"lm_head.weight", exp("0"), exp("1")}); err != nil {
		t.Errorf("a mixture in every block: %v", err)
	}
	if _, err := moe.config(moeCfg, []string{"lm_head.weight", exp("0")}); err == nil || !strings.Contains(err.Error(), "block 1 carries no experts") {
		t.Errorf("a dense block in a mixture: %v", err)
	}
	g := lib["GemmaForCausalLM"]
	gemma := func(extra string) (*Config, error) {
		js := `{"architectures": ["GemmaForCausalLM"], "vocab_size": 100, "hidden_size": 64, "intermediate_size": 96,
		  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16` + extra + `}`
		return g.config(hfConfigOf(t, js), nil)
	}
	for _, r := range []struct {
		name, extra string
		flags, not  jlm.Flags
		err         string
	}{
		{"the class default", ``, jlm.FlagGELU, jlm.FlagNonCausal, ""},
		{"silu stated", `, "hidden_act": "silu"`, 0, jlm.FlagGELU, ""},
		{"hidden_activation over hidden_act", `, "hidden_act": "gelu", "hidden_activation": "gelu_pytorch_tanh"`, jlm.FlagGELU, 0, ""},
		{"bidirectional", `, "use_bidirectional_attention": true`, jlm.FlagNonCausal, 0, ""},
		{"erf gelu", `, "hidden_act": "gelu"`, 0, 0, "activation gelu is not implemented"},
		{"a window in use", `, "sliding_window": 4096`, 0, 0, "a sliding window of 4096"},
		{"a window not in use", `, "sliding_window": 4096, "use_sliding_window": false`, jlm.FlagGELU, 0, ""},
		{"llama3 scaling", `, "rope_scaling": {"rope_type": "llama3", "factor": 8}`, 0, 0, "rope scaling llama3 is not implemented"},
		{"default scaling", `, "rope_parameters": {"rope_type": "default", "rope_theta": 10000}`, jlm.FlagGELU, 0, ""},
		{"two thetas", `, "rope_parameters": {"rope_theta": 10000}, "rope_theta": 500000`, 0, 0, "spelled two ways"},
		{"a partial rotary", `, "partial_rotary_factor": 0.5`, jlm.FlagGELU, 0, ""},
	} {
		c, err := gemma(r.extra)
		switch {
		case r.err != "":
			if err == nil || !strings.Contains(err.Error(), r.err) {
				t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
			}
		case err != nil:
			t.Errorf("%s: %v", r.name, err)
		case c.Flags&r.flags != r.flags || c.Flags&r.not != 0:
			t.Errorf("%s: flags %v, want %v set and %v clear", r.name, c.Flags, r.flags, r.not)
		case r.name == "a partial rotary" && c.NRot != 8:
			t.Errorf("a partial rotary of a half: %d rotated of 16", c.NRot)
		}
	}
	for _, r := range []struct{ name, js, err string }{
		{"no activation", `"activation": "hidden_act",`, "states no activation"},
		{"an unknown flag", `"flags": {"attention_output_gate"`, "not a graph feature"},
		{"a window without its use", `"window": {"size": "sliding_window", "used": "use_sliding_window ?? true"}`, "whether it is used"},
	} {
		lib := map[string]string{"no activation": "qwen3.json", "an unknown flag": "qwen3_5.json", "a window without its use": "gemma.json"}
		b, err := modelLibraryFS.ReadFile("models/" + lib[r.name])
		if err != nil {
			t.Fatal(err)
		}
		src := string(b)
		switch r.name {
		case "no activation":
			src = strings.Replace(src, `"activation": "hidden_act ?? 'silu'",`, ``, 1)
		case "an unknown flag":
			src = strings.Replace(src, `"flags": {"attention_output_gate"`, `"flags": {"attention_output_gat"`, 1)
		default:
			src = strings.Replace(src, `"used": "use_sliding_window ?? true"`, `"used": ""`, 1)
		}
		if src == string(b) {
			t.Fatalf("%s: the mutation changed nothing", r.name)
		}
		if _, err := parseModelJSON([]byte(src), r.name); err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
}
