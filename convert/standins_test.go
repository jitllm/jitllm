package convert

import (
	"slices"
	"testing"
)

// TestStandInsAreEachRolesShape: a source feeding several roles stands in for
// configOf under each, at the shape its primitives cut it to -- Phi-3's fused
// qkv_proj as q, k and v, its gate_up_proj as gate and up -- not under its
// first role at the source's shape.
func TestStandInsAreEachRolesShape(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["Phi3ForCausalLM"]
	hc := hfConfigOf(t, `{"architectures": ["Phi3ForCausalLM"], "vocab_size": 100, "hidden_size": 64,
	  "intermediate_size": 96, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
	  "head_dim": 16}`)
	env, err := mj.env(hc)
	if err != nil {
		t.Fatal(err)
	}
	qkv, gu := "model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.mlp.gate_up_proj.weight"
	shapes := map[string][]uint64{qkv: {128, 64}, gu: {192, 64}}
	ts, err := mj.blockStandIns(env, shapes, []string{qkv, gu}, map[string]bool{})
	if err != nil {
		t.Fatal(err)
	}
	got := map[string][]uint64{}
	for _, s := range ts {
		got[s.Name] = s.Dims
	}
	for name, want := range map[string][]uint64{
		"blk.0.attn_q.weight": {64, 64}, "blk.0.attn_k.weight": {64, 32}, "blk.0.attn_v.weight": {64, 32},
		"blk.0.ffn_gate.weight": {64, 96}, "blk.0.ffn_up.weight": {64, 96},
	} {
		if !slices.Equal(got[name], want) {
			t.Errorf("%s stands in as %v, want %v", name, got[name], want)
		}
	}
}
