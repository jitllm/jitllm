package convert

import (
	"strings"
	"testing"
)

// TestBitNetThroughTheLibrary: BitNetForCausalLM's config.json, read through
// convert/models/bitnet.json, states the squared-ReLU graph, and every
// quantization_config the file does not store is refused by what it says.
func TestBitNetThroughTheLibrary(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["BitNetForCausalLM"]
	if mj == nil || mj.Tensors.Ternary == nil {
		t.Fatal("no BitNetForCausalLM with ternary projections in the library")
	}
	js := `{"architectures": ["BitNetForCausalLM"], "vocab_size": 100, "hidden_size": 256, "intermediate_size": 512,
	  "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "hidden_act": "relu2",
	  "tie_word_embeddings": true, "rope_theta": 500000.0,
	  "quantization_config": {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "offline"}}`
	cfg, err := mj.config(hfConfigOf(t, js), nil)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Arch != ArchBitNet || cfg.NKVHead != 2 || cfg.RopeBase != 500000 {
		t.Errorf("arch %v, %d kv heads, rope base %v", cfg.Arch, cfg.NKVHead, cfg.RopeBase)
	}
	for _, r := range []struct{ name, from, to, err string }{
		{"no quantization", `"quantization_config": {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "offline"}`, `"x": 1`, "quantization_config"},
		{"another method", `"quant_method": "bitnet"`, `"quant_method": "gptq"`, "quantization_config"},
		{"an RMSNorm in each projection", `"quantization_mode": "offline"`, `"quantization_mode": "offline", "use_rms_norm": true`, "use_rms_norm"},
		{"a projection left full precision", `"quantization_mode": "offline"`, `"quantization_mode": "offline", "modules_to_not_convert": ["q_proj"]`, "modules_to_not_convert"},
		{"online bitlinear", `"linear_class": "autobitlinear", "quantization_mode": "offline"`, `"linear_class": "bitlinear", "quantization_mode": "online"`, "not a stored form"},
		{"another linear", `"autobitlinear"`, `"other"`, "linear_class"},
		{"SiLU", `"hidden_act": "relu2"`, `"hidden_act": "silu"`, "hidden_act"},
		{"a feed-forward bias", `"tie_word_embeddings": true`, `"tie_word_embeddings": true, "mlp_bias": true`, "mlp_bias"},
	} {
		if !strings.Contains(js, r.from) {
			t.Fatalf("%s: the edit matches nothing", r.name)
		}
		_, err := mj.config(hfConfigOf(t, strings.Replace(js, r.from, r.to, 1)), nil)
		if err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
}

// TestTernaryScalesBelongToTheirWeights: a scale tensor beside no weight is
// refused, and one beside its weight is consumed.
func TestTernaryScalesBelongToTheirWeights(t *testing.T) {
	used, err := ternaryGather("_scale", []string{"a.weight", "a.weight_scale", "b.weight"})
	if err != nil || len(used) != 1 || !used["a.weight_scale"] {
		t.Errorf("a scale beside its weight: %v, %v", used, err)
	}
	if _, err := ternaryGather("_scale", []string{"a.weight_scale"}); err == nil || !strings.Contains(err.Error(), "scales no weight") {
		t.Errorf("a scale with no weight: %v", err)
	}
}
