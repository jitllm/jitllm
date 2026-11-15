package convert

import (
	"errors"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestPhi3ThroughTheLibrary: phi3.json cuts qkv_proj into q, k and v and
// gate_up_proj into gate then up, by row ranges, as Phi3Attention and Phi3MLP
// read them, each part in its source's place; a fused tensor of another row
// count is refused, not cut. Every layer slides when the class states a
// window, and a null one is none (not llama.cpp's defaults for a GGUF without
// the key).
func TestPhi3ThroughTheLibrary(t *testing.T) {
	mj := libraryFile(t, "Phi3ForCausalLM")
	a, err := mj.hfArch()
	if err != nil {
		t.Fatal(err)
	}
	js := `{"architectures": ["Phi3ForCausalLM"], "vocab_size": 100, "hidden_size": 2, "intermediate_size": 3,
	  "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2, "hidden_act": "silu",
	  "sliding_window": 4, "rope_scaling": null, "tie_word_embeddings": false}`
	names := []string{"lm_head.weight"}
	c, err := a.config(hfConfigOf(t, js), names)
	if err != nil {
		t.Fatal(err)
	}
	if c.Arch != ArchPhi3 || c.SWAWindow != 4 || c.SWAPeriod != allLocal(2) || !c.Flags.Has(jlm.FlagRopeNeox) {
		t.Errorf("arch %v, window %d every %d, flags %v", c.Arch, c.SWAWindow, c.SWAPeriod, c.Flags)
	}
	qkv, gu := iota32(16), iota32(12)
	for _, tc := range []struct {
		name  string
		src   []float32
		roles []jlm.Role
		rows  []int
	}{
		{"self_attn.qkv_proj.weight", qkv, []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV}, []int{4, 2, 2}},
		{"mlp.gate_up_proj.weight", gu, []jlm.Role{jlm.RoleFFNGate, jlm.RoleFFNUp}, []int{3, 3}},
	} {
		cut := a.split[tc.name]
		if cut == nil {
			t.Fatalf("%s is not cut", tc.name)
		}
		e := jlm.Tensor{Name: "model.layers.0." + tc.name, Type: jlm.TypeF32, NDim: 2,
			Dims: [4]uint64{2, uint64(len(tc.src) / 2)}}
		parts, err := cut(e, f32AsBytes(tc.src), c)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if len(parts) != len(tc.roles) {
			t.Fatalf("%s: %d parts, want %d", tc.name, len(parts), len(tc.roles))
		}
		at := 0
		for i, p := range parts {
			want := tc.src[at*2 : (at+tc.rows[i])*2]
			at += tc.rows[i]
			if p.Role != tc.roles[i] || p.Dims != [4]uint64{2, uint64(tc.rows[i])} || string(p.Data) != string(f32AsBytes(want)) {
				t.Errorf("%s part %d: %v %v, want %v of %d rows", tc.name, i, p.Role, p.Dims, tc.roles[i], tc.rows[i])
			}
		}
		e.Dims[1]--
		if _, err := cut(e, f32AsBytes(tc.src[:len(tc.src)-2]), c); err == nil {
			t.Errorf("%s: a fused tensor a row short was cut", tc.name)
		}
	}

	if c, err := a.config(hfConfigOf(t, strings.Replace(js, `"sliding_window": 4`, `"sliding_window": null`, 1)), names); err != nil || c.SWAWindow != 0 {
		t.Errorf("a null window: %v, %v", c, err)
	}
	scaled := func(scaling string) string {
		return strings.Replace(js, `"rope_scaling": null`, `"rope_scaling": `+scaling, 1)
	}
	if _, err := a.config(hfConfigOf(t, scaled(`{"type": "longrope"}`)), names); err == nil || !strings.Contains(err.Error(), "LongRoPE factors") {
		t.Errorf("longrope with no factors: %v, want a refusal naming them", err)
	}
	if _, err := a.config(hfConfigOf(t, scaled(`{"type": "su2"}`)), names); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("a scaling no file reads: %v, want ErrNotImplemented", err)
	}
	hc := hfConfigOf(t, scaled(`{"type": "longrope", "short_factor": [1.5], "long_factor": [2.5], "original_max_position_embeddings": 8}`))
	if c, err := a.config(hc, names); err != nil || c.YarnOrigCtx != 8 || len(hc.extra) != 2 {
		t.Errorf("longrope with its factors: %v, %v, %d tensors", c, err, len(hc.extra))
	}
}

// TestModelJSONTargetsRefuse: a source of several targets names two or more,
// each with primitives to cut it by and a role the container knows.
func TestModelJSONTargetsRefuse(t *testing.T) {
	base := `{"classes": ["X"], "config": {"arch": "phi3", "vocab": "1", "head": {"tensor": "lm_head.weight"},
	  "activation": "'silu'"}, "tensors": {"block_prefix": "model.layers.", "block": {"a.weight": %s}}}`
	cut := `{"role": "attn_k.weight", "do": [{"slice": [0, 0, 1]}]}`
	for _, tc := range []struct{ name, targets, err string }{
		{"one target in a list", `[` + cut + `]`, "two or more"},
		{"a target with no primitives", `["attn_q.weight", ` + cut + `]`, "no primitives"},
		{"an unknown role", `[{"role": "nope.weight", "do": [{"slice": [0, 0, 1]}]}, ` + cut + `]`, "not a tensor name"},
	} {
		mj, err := parseModelJSON([]byte(strings.Replace(base, "%s", tc.targets, 1)), "test")
		if err == nil {
			_, err = mj.hfArch()
		}
		if err == nil || !strings.Contains(err.Error(), tc.err) {
			t.Errorf("%s: %v, want an error saying %q", tc.name, err, tc.err)
		}
	}
}
