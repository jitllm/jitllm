package convert

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
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
		"head": {"tensor": "lm_head.weight"}}, "tensors": {"block_prefix": "model.layers.",
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
		{"no vocab", `"vocab": "vocab_size",`, ``, "no vocab"},
		{"unknown role", `"attn_q.weight"`, `"attn_quux.weight"`, "not a tensor name"},
		{"unknown transform", `"attn_q.weight"`, `{"role": "attn_q.weight", "do": ["frobnicate"]}`, "not in the vocabulary"},
		{"two shape transforms", `"attn_q.weight"`, `{"role": "attn_q.weight", "do": ["squeeze", "squeeze"]}`, "one is the limit"},
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
		{"gelu", []string{`"hidden_act": "silu"`, `"hidden_act": "gelu"`}, names, "silu-gated"},
		{"no gate", []string{`"attn_output_gate": true`, `"attn_output_gate": false`}, names, "gate their output"},
		{"yarn", []string{`"rope_type": "default"`, `"rope_type": "yarn"`}, names, "rope_type"},
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

// TestModelJSONTransforms holds each transform to its definition.
func TestModelJSONTransforms(t *testing.T) {
	e := jlm.Tensor{Name: "n", Type: jlm.TypeF32, NDim: 1, Dims: [4]uint64{3}}
	got, err := mjElems()["plus_one"](&e, f32AsBytes([]float32{0, 0.5, -1}))
	if err != nil || !slices.Equal(got, f32AsBytes([]float32{1, 1.5, 0})) {
		t.Errorf("plus_one: %v %v", got, err)
	}
	e = jlm.Tensor{Name: "n", Type: jlm.TypeF32, NDim: 1, Dims: [4]uint64{2}}
	got, err = mjElems()["neg_exp"](&e, f32AsBytes([]float32{0, float32(math.Log(2))}))
	if err != nil || !slices.Equal(got, f32AsBytes([]float32{-1, -2})) {
		t.Errorf("neg_exp: %v %v", got, err)
	}
	e.Type = jlm.TypeF16
	if _, err := mjElems()["neg_exp"](&e, nil); err == nil {
		t.Error("neg_exp ran on f16 bytes")
	}
	conv := jlm.Tensor{Name: "conv", Type: jlm.TypeF32, NDim: 3, Dims: [4]uint64{4, 1, 10}}
	parts, err := mjShapes()["squeeze"](conv, nil, nil)
	if err != nil || parts[0].NDim != 2 || parts[0].Dims != [4]uint64{4, 10} {
		t.Errorf("squeeze: %+v %v", parts, err)
	}
	// Four value heads over two key heads, rows of two: per key head, its two
	// heads' beta rows, then their alpha rows.
	c := &Config{Config: &jlm.Config{SSM: jlm.SSMConfig{Groups: 2, NHeadV: 4}}}
	beta := []float32{0, 0, 1, 1, 2, 2, 3, 3}
	alpha := []float32{10, 10, 11, 11, 12, 12, 13, 13}
	v, dims, err := mjGathers()["interleave_groups"](c, [][]float32{beta, alpha})
	want := []float32{0, 0, 1, 1, 10, 10, 11, 11, 2, 2, 3, 3, 12, 12, 13, 13}
	if err != nil || !slices.Equal(v, want) || dims != [2]uint64{2, 8} {
		t.Errorf("interleave_groups: %v %v %v", v, dims, err)
	}
	if _, _, err := mjGathers()["interleave_groups"](c, [][]float32{beta, alpha[:6]}); err == nil {
		t.Error("interleave_groups joined sources of different lengths")
	}
}
