package convert

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestMixtralThroughTheLibrary: mixtral.json's q and k recipes (a reshape, an
// axis swap and a reshape back) move each head's rows from rotate_half's order
// to the adjacent pairs llama's graph turns, the permutation a
// GGUF's q and k carry (permuteHeadPairs); the router renormalises,
// and a window masks every layer, as MixtralModel's does.
func TestMixtralThroughTheLibrary(t *testing.T) {
	mj := libraryFile(t, "MixtralForCausalLM")
	env := mjEnv(map[string]any{"hidden_size": 3.0, "num_attention_heads": 4.0, "num_key_value_heads": 2.0, "head_dim": 8.0})
	for _, tc := range []struct {
		name  string
		heads int
	}{{"self_attn.q_proj.weight", 4}, {"self_attn.k_proj.weight", 2}} {
		rows := tc.heads * 8
		v := iota32(rows * 3)
		got, err := mj.Tensors.Block[tc.name][0].Do.run(env, []mjVal{mjF32([]uint64{uint64(rows), 3}, v...)})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		want := permuteHeadPairs(f32AsBytes(v), tc.heads, 8, 3*4)
		if string(want) == string(f32AsBytes(v)) {
			t.Fatal("the permutation is the identity on this input; the gate proves nothing")
		}
		if !slices.Equal(got.shape, []uint64{uint64(rows), 3}) || string(got.data) != string(want) {
			t.Errorf("%s: shape %v, and the rows are not llama's rotary order", tc.name, got.shape)
		}
	}

	js := `{"architectures": ["MixtralForCausalLM"], "vocab_size": 100, "hidden_size": 32, "intermediate_size": 16,
	  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "num_local_experts": 4,
	  "num_experts_per_tok": 2, "hidden_act": "silu", "sliding_window": null, "tie_word_embeddings": false}`
	names := []string{"lm_head.weight"}
	c, err := mj.config(hfConfigOf(t, js), names)
	if err != nil {
		t.Fatal(err)
	}
	if c.Name != "llama" || c.NExpert != 4 || c.NExpertUsed != 2 || c.NFFNExp != 16 || c.Flags.Has(jlm.FlagNoExpertNorm) {
		t.Errorf("arch %v, %d experts, %d used, %d wide, flags %v", c.Name, c.NExpert, c.NExpertUsed, c.NFFNExp, c.Flags)
	}
	if c.SWAWindow != 0 {
		t.Errorf("a null window states one of %d", c.SWAWindow)
	}
	c, err = mj.config(hfConfigOf(t, strings.Replace(js, `"sliding_window": null`, `"sliding_window": 4096`, 1)), names)
	if err != nil || c.SWAWindow != 4096 || !c.Sliding(0) || !c.Sliding(1) {
		t.Errorf("a window of 4096: %v, window %d, layers sliding %v %v", err, c.SWAWindow, c.Sliding(0), c.Sliding(1))
	}
	for _, r := range [][2]string{
		{`"hidden_act": "silu"`, `"hidden_act": "gelu"`},
		{`"hidden_act": "silu"`, `"hidden_act": "silu", "rope_scaling": {"rope_type": "dynamic", "factor": 4}`},
	} {
		if _, err := mj.config(hfConfigOf(t, strings.Replace(js, r[0], r[1], 1)), names); !errors.Is(err, ErrNotImplemented) {
			t.Errorf("%s: %v, want ErrNotImplemented", r[1], err)
		}
	}
}
