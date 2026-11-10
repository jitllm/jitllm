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
	// What a revision states from the architecture is stated by the
	// migration, the last place that reads it: Qwen3-VL's split interleaved,
	// and nothing for an architecture whose split is contiguous.
	for _, c := range []struct {
		arch        Arch
		interleaved bool
	}{{ArchQwen3VL, true}, {ArchQwen2VL, false}} {
		cfg := &jlm.Config{Revision: 2, RopeSections: [4]uint32{8, 12, 12, 0}}
		if err := LegacyStated(uint16(c.arch), cfg); err != nil {
			t.Fatal(err)
		}
		if cfg.Revision != jlm.StatementRevision || cfg.Flags.Has(jlm.FlagRopeInterleaved) != c.interleaved {
			t.Errorf("%v at revision 2: revision %d, interleaved %v, want %d and %v", c.arch, cfg.Revision,
				cfg.Flags.Has(jlm.FlagRopeInterleaved), jlm.StatementRevision, c.interleaved)
		}
	}
}
