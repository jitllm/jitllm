package convert

import (
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestStatedExpertNormDecides holds expert_weights_norm to one reading for
// every graph: a stated key decides, and an absent one leaves the graph's
// default -- olmoe's no renormalisation, qwen3moe's renormalisation. The
// safetensors path states the key from norm_topk_prob, so an OLMoE checkpoint
// that asks for renormalisation gets it.
func TestStatedExpertNormDecides(t *testing.T) {
	yes, no := true, false
	for _, tc := range []struct {
		arch   string
		stated *bool
		renorm bool
	}{
		{"olmoe", nil, false},
		{"olmoe", &yes, true},
		{"olmoe", &no, false},
		{"qwen3moe", nil, true},
		{"qwen3moe", &yes, true},
		{"qwen3moe", &no, false},
	} {
		k := baseKeys(2)
		k["expert_count"] = meta.MakeUint(4)
		k["expert_used_count"] = meta.MakeUint(2)
		k["expert_feed_forward_length"] = meta.MakeUint(32)
		if tc.stated != nil {
			k["expert_weights_norm"] = meta.MakeBool(*tc.stated)
		}
		c, err := configOf(headerOnly(tc.arch, k))
		if err != nil {
			t.Fatalf("%s: %v", tc.arch, err)
		}
		if got := !c.Flags.Has(jlm.FlagNoExpertNorm); got != tc.renorm {
			stated := "absent"
			if tc.stated != nil {
				stated = map[bool]string{true: "true", false: "false"}[*tc.stated]
			}
			t.Errorf("%s with expert_weights_norm %s: renormalises %v, want %v", tc.arch, stated, got, tc.renorm)
		}
	}
}
