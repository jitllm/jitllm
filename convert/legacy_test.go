package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestLegacyStatedRefusesWhatV31CannotRecover holds the v30 migration's
// check: a ModernBERT container stating no activations and a Phi-3.5-MoE
// one stating no routed weight are refused, the same containers stating
// them and another architecture stating neither are not.
func TestLegacyStatedRefusesWhatV31CannotRecover(t *testing.T) {
	routed := &Config{Arch: ArchPhiMoE, Config: &jlm.Config{NExpert: 16}}
	routed.RouterWeight = RouterWeight(routed)
	for _, c := range []struct {
		name string
		arch Arch
		cfg  *jlm.Config
		bad  bool
	}{
		{"modernbert without activations", ArchModernBERT, &jlm.Config{}, true},
		{"modernbert with them", ArchModernBERT, &jlm.Config{Acts: []jlm.Expr{{}}}, false},
		{"phimoe without a routed weight", ArchPhiMoE, &jlm.Config{NExpert: 16}, true},
		{"phimoe with one", ArchPhiMoE, routed.Config, false},
		{"llama with neither", ArchLlama, &jlm.Config{}, false},
	} {
		if err := LegacyStated(uint16(c.arch), c.cfg); (err != nil) != c.bad {
			t.Errorf("%s: %v, want refused %v", c.name, err, c.bad)
		}
	}
}
