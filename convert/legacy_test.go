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
	routed := jlm.Config{Arch: jlm.ArchPhiMoE, NExpert: 16}
	routed.RouterWeight = RouterWeight(&routed)
	for _, c := range []struct {
		name string
		cfg  jlm.Config
		bad  bool
	}{
		{"modernbert without activations", jlm.Config{Arch: jlm.ArchModernBERT}, true},
		{"modernbert with them", jlm.Config{Arch: jlm.ArchModernBERT, Acts: []jlm.Expr{{}}}, false},
		{"phimoe without a routed weight", jlm.Config{Arch: jlm.ArchPhiMoE, NExpert: 16}, true},
		{"phimoe with one", routed, false},
		{"llama with neither", jlm.Config{Arch: jlm.ArchLlama}, false},
	} {
		if err := LegacyStated(&c.cfg); (err != nil) != c.bad {
			t.Errorf("%s: %v, want refused %v", c.name, err, c.bad)
		}
	}
}
