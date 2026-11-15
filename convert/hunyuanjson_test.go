package convert

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestHunyuanThroughTheLibrary: both Hunyuan classes through
// convert/models/hunyuan.json -- the NTK-alpha base, the mixture read off
// num_experts with a per-layer list of one value, and what the class refuses.
func TestHunyuanThroughTheLibrary(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	const base = `"vocab_size": 100, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3,
		"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "tie_word_embeddings": true`
	const alpha = `, "rope_parameters": {"rope_type": "dynamic", "alpha": 1000.0, "factor": 1.0, "rope_theta": 10000.0}`
	moeNames := []string{"model.layers.0.mlp.shared_mlp.gate_proj.weight"}
	for b := range 3 {
		p := fmt.Sprintf("model.layers.%d.mlp.", b)
		moeNames = append(moeNames, p+"gate.wg.weight", p+"experts.0.gate_proj.weight")
	}
	for b := 1; b < 3; b++ {
		moeNames = append(moeNames, fmt.Sprintf("model.layers.%d.mlp.shared_mlp.gate_proj.weight", b))
	}
	run := func(class, extra string, names []string) (*Config, error) {
		c := hfConfigOf(t, fmt.Sprintf(`{"architectures": [%q], %s%s}`, class, base, extra))
		c.shapes = map[string][]uint64{}
		for _, n := range names {
			if strings.HasSuffix(n, "shared_mlp.gate_proj.weight") {
				c.shapes[n] = []uint64{128, 64}
			}
		}
		return lib[class].config(c, names)
	}
	const dense, moe = "HunYuanDenseV1ForCausalLM", "HunYuanMoEV1ForCausalLM"

	wantBase := float32(10000 * math.Pow(1000, 16.0/14.0))
	for _, tc := range []struct {
		name, extra string
		base        float32
	}{
		{"the alpha scales the base", alpha, wantBase},
		{"a top-level rope_theta is the base", `, "rope_theta": 50000.0`, 50000},
		{"no rotary statement is the class's 10000", ``, 10000},
	} {
		c, err := run(dense, tc.extra, nil)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if c.RopeBase != tc.base || c.NExpert != 0 || c.NFFNShExp != 0 ||
			!c.Flags.Has(jlm.FlagRopeNeox|jlm.FlagQKNorm|jlm.FlagQKNormPostRope) {
			t.Errorf("%s: base %v (want %v), %d experts, shared %d, flags %v", tc.name, c.RopeBase, tc.base,
				c.NExpert, c.NFFNShExp, c.Flags)
		}
	}

	c, err := run(moe, alpha+`, "num_experts": 8, "moe_topk": [2, 2, 2], "num_shared_expert": [1, 1, 1]`, moeNames)
	if err != nil {
		t.Fatalf("a mixture: %v", err)
	}
	if c.NExpert != 8 || c.NExpertUsed != 2 || c.NFFNExp != 128 || c.NFFNShExp != 128 || c.RopeBase != wantBase {
		t.Errorf("a mixture: %d experts of %d used, %d wide, shared %d, base %v", c.NExpert, c.NExpertUsed,
			c.NFFNExp, c.NFFNShExp, c.RopeBase)
	}
	if c, err = run(moe, alpha+`, "num_experts": 1`, nil); err != nil || c.NExpert != 0 {
		t.Errorf("one expert is a dense model: %v, %v", c, err)
	}

	for _, tc := range []struct{ class, extra, want string }{
		{dense, `, "use_cla": true`, "use_cla"},
		{dense, `, "rope_parameters": {"type": "dynamic", "factor": 2.0}`, "dynamic"},
		{dense, `, "rope_parameters": {"type": "linear", "factor": 2.0}`, "linear"},
		{dense, `, "rope_scaling": "dynamic"`, "is not an object"},
		{moe, alpha + `, "num_experts": 8, "moe_topk": [2, 3, 2]`, "entries differ"},
		{moe, alpha + `, "num_experts": 8, "moe_topk": 2, "moe_intermediate_size": 64`, "moe_intermediate_size"},
		{moe, alpha + `, "num_experts": 8, "moe_topk": 2, "num_shared_expert": 2`, "num_shared_expert"},
		{moe, alpha + `, "num_experts": 8, "moe_topk": 2, "scoring_func": "sigmoid"`, "softmax"},
	} {
		names := moeNames
		if tc.class == dense {
			names = nil
		}
		if _, err := run(tc.class, tc.extra, names); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s%s: %v, want a refusal naming %q", tc.class, tc.extra, err, tc.want)
		}
	}
}
