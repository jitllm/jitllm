package convert

import (
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestLegacyStatedRefusesWhatV31CannotRecover holds the v30 migration's
// check: a ModernBERT container stating no activations and a Phi-3.5-MoE
// one stating no routed weight are refused, the same containers stating
// them and another architecture stating neither are not.
func TestLegacyStatedRefusesWhatV31CannotRecover(t *testing.T) {
	routed := statedBy(t, "phimoe", &jlm.Config{NExpert: 16})
	var err error
	if routed.RouterWeight, err = RouterWeight(routed); err != nil {
		t.Fatal(err)
	}
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

// TestLegacyCodesStateWhatTheRevisionStated holds the migration's table of
// opaque codes to what a revision before jlm.SourceRevision left to each
// architecture's code, on a windowed model and an unwindowed one.
func TestLegacyCodesStateWhatTheRevisionStated(t *testing.T) {
	sliding := []bool{true, false, true, false}
	for _, c := range []struct {
		arch        Arch
		win         uint32
		flags       jlm.Flags
		quant       jlm.ActQuant
		scale       float32
		globalFirst bool
		bidir       []bool
	}{
		{arch: ArchQwen3VL, flags: jlm.FlagRopeInterleaved},
		{arch: ArchQwen3VLMoE, flags: jlm.FlagRopeInterleaved},
		{arch: ArchQwen3Next, flags: jlm.FlagRopeInterleaved},
		{arch: ArchQwen4Exp, flags: jlm.FlagRopeInterleaved},
		{arch: ArchHunyuanVL, flags: jlm.FlagRopeXD},
		{arch: ArchBitNet, quant: jlm.ActQuantRowInt8},
		{arch: ArchOLMo3, win: 512, scale: 1},
		{arch: ArchModernBERT, win: 512, globalFirst: true},
		{arch: ArchModernBERT},
		{arch: ArchGemma4, win: 512, bidir: sliding},
		{arch: ArchGemma4},
		{arch: ArchLlama, win: 512},
		{arch: ArchQwen2VL},
	} {
		cfg := &jlm.Config{NLayer: 4, SWAWindow: c.win, SWAPeriod: 2, Acts: []jlm.Expr{{}}}
		if err := LegacyStated(uint16(c.arch), cfg); err != nil {
			t.Fatalf("%v: %v", c.arch, err)
		}
		if cfg.Revision != jlm.StatementRevision || cfg.Flags != c.flags || cfg.ActQuant != c.quant || cfg.RopeScaleSWA != c.scale ||
			cfg.Flags2.Has(jlm.Flag2SWAGlobalFirst) != c.globalFirst || !reflect.DeepEqual(cfg.BidirLayers, c.bidir) {
			t.Errorf("%v (window %d): revision %d, flags %v, quant %d, scale %v, global first %v, bidir %v", c.arch, c.win,
				cfg.Revision, cfg.Flags, cfg.ActQuant, cfg.RopeScaleSWA, cfg.Flags2.Has(jlm.Flag2SWAGlobalFirst), cfg.BidirLayers)
		}
	}
}
