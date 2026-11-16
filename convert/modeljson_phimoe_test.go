package convert

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestPhiMoEThroughTheLibrary: PhiMoEForCausalLM's config.json, read through
// convert/models/phimoe.json, states LayerNorm and the NEOX rotary, the
// sparsemixer router without a renormalisation, and LongRoPE's two factor
// vectors with its original context; each statement the class's reference
// depends on and the container does not vary is refused when it differs.
func TestPhiMoEThroughTheLibrary(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["PhimoeForCausalLM"]
	if mj == nil || lib["PhiMoEForCausalLM"] != mj {
		t.Fatal("PhimoeForCausalLM and PhiMoEForCausalLM are not one file of the library")
	}
	js := `{"architectures": ["PhiMoEForCausalLM"], "vocab_size": 100, "hidden_size": 64, "intermediate_size": 32,
	  "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "num_local_experts": 8,
	  "num_experts_per_tok": 2, "max_position_embeddings": 4096, "router_jitter_noise": 0.01,
	  "sliding_window": null, "original_max_position_embeddings": 512,
	  "rope_parameters": {"rope_type": "longrope", "rope_theta": 10000.0, "short_mscale": 1.25, "long_mscale": 1.25,
	    "short_factor": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], "long_factor": [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]}}`
	names := []string{"lm_head.weight"}
	c := hfConfigOf(t, js)
	cfg, err := mj.config(c, names)
	if err != nil {
		t.Fatal(err)
	}
	want := jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagNoExpertNorm
	if cfg.Arch != ArchPhiMoE || cfg.Flags&want != want || cfg.NExpert != 8 || cfg.NExpertUsed != 2 ||
		cfg.YarnOrigCtx != 512 || cfg.AttnFactor != 1.25 || cfg.SWAWindow != 0 || len(c.extra) != 2 {
		t.Errorf("arch %v flags %v, %d of %d experts, original context %d, magnitude %v, window %d, %d tensors",
			cfg.Arch, cfg.Flags, cfg.NExpertUsed, cfg.NExpert, cfg.YarnOrigCtx, cfg.AttnFactor, cfg.SWAWindow, len(c.extra))
	}
	if cfg, err = mj.config(hfConfigOf(t, strings.Replace(js, `"sliding_window": null`, `"sliding_window": 1024`, 1)), names); err != nil ||
		cfg.SWAWindow != 1024 || cfg.SWAPeriod == 0 {
		t.Errorf("a window shorter than the context: %v, window %d period %d", err, cfg.SWAWindow, cfg.SWAPeriod)
	}
	if cfg, err = mj.config(hfConfigOf(t, strings.Replace(js, `"sliding_window": null`, `"sliding_window": 4096`, 1)), names); err != nil ||
		cfg.SWAWindow != 0 {
		t.Errorf("a window as long as the context masks nothing: %v, window %d", err, cfg.SWAWindow)
	}
	for _, r := range []struct{ name, from, to, err string }{
		{"another jitter", `"router_jitter_noise": 0.01`, `"router_jitter_noise": 0.02`, "router_jitter_noise"},
		{"three experts a token", `"num_experts_per_tok": 2`, `"num_experts_per_tok": 3`, "expert_used_count"},
		{"factors for another rotary width", `"short_factor": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]`, `"short_factor": [1, 1.5]`, "LongRoPE factors"},
		{"a magnitude past the original context", `"long_mscale": 1.25`, `"long_mscale": 1.5`, "long_mscale"},
		{"a partial rotary", `"max_position_embeddings": 4096`, `"max_position_embeddings": 4096, "partial_rotary_factor": 0.5`, "partial_rotary_factor"},
	} {
		if !strings.Contains(js, r.from) {
			t.Fatalf("%s: the edit matches nothing", r.name)
		}
		_, err := mj.config(hfConfigOf(t, strings.Replace(js, r.from, r.to, 1)), names)
		if err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
}

// TestPhiMoEJitterIsTheStatements: the threshold phimoe.json holds a config
// to is the one phimoe's router statement runs at, a config that states
// another is refused, an absent router_jitter_noise reads as the class's
// default, and a null one is refused: PhimoeConfig keeps it None, and
// sparsemixer's threshold 2 * None does not compute.
func TestPhiMoEJitterIsTheStatements(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	st, _, err := ggufStates("phimoe")
	if err != nil || st == nil || st.Router == nil || st.Router.Jitter == 0 {
		t.Fatalf("phimoe's statement states no router jitter: %v", err)
	}
	jitter := st.Router.Jitter
	mj := lib["PhimoeForCausalLM"]
	js := `{"architectures": ["PhiMoEForCausalLM"], "vocab_size": 100, "hidden_size": 64, "intermediate_size": 32,
	  "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "num_local_experts": 8,
	  "num_experts_per_tok": 2, "max_position_embeddings": 4096, "sliding_window": null %s}`
	for _, tc := range []struct {
		name, key string
		ok        bool
	}{
		{"the statement's", fmt.Sprintf(`, "router_jitter_noise": %v`, jitter), true},
		{"another", fmt.Sprintf(`, "router_jitter_noise": %v`, 2*jitter), false},
		{"zero", `, "router_jitter_noise": 0`, false},
		{"absent", ``, true},
		{"null", `, "router_jitter_noise": null`, false},
	} {
		_, err := mj.config(hfConfigOf(t, fmt.Sprintf(js, tc.key)), []string{"lm_head.weight"})
		if (err == nil) != tc.ok {
			t.Errorf("%s: %v, accepted %v", tc.name, err, tc.ok)
		}
	}
}
