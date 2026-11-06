package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestDeltaStatementRoundTrips holds the delta rule's stated decay and gate
// to the config's contract: they round-trip; a container an older writer
// made states them from its bound and its sigmoid bit (and the per-channel
// decay's architectures' own gate); and a statement no delta rule reads is
// refused.
func TestDeltaStatementRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{Arch: ArchQwen3Next, NLayer: 4, NEmbd: 64, NHead: 4, NKVHead: 2, HeadDim: 16, NRot: 16,
			NFFN: 128, NVocab: 1000, RMSEps: 1e-6, RopeBase: 1e6,
			LayerKinds: []LayerKind{LayerLinearAttn, LayerFullAttn, LayerLinearAttn, LayerFullAttn}}
	}
	c := base()
	c.DeltaDecay, c.DeltaGate = DeltaDecayBounded(-5), DeltaGateSigmoid()
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.DeltaDecay, c.DeltaDecay) || !reflect.DeepEqual(got.DeltaGate, c.DeltaGate) {
		t.Fatalf("round trip: decay %+v, gate %+v", got.DeltaDecay, got.DeltaGate)
	}

	for _, tc := range []struct {
		name         string
		edit         func(c *Config)
		decay, gated *Program
	}{
		{"the defaults", func(c *Config) {}, DeltaDecaySoftplus(), DeltaGateSiLU()},
		{"a bound", func(c *Config) { c.KDALowerBound = -5 }, DeltaDecayBounded(-5), DeltaGateSiLU()},
		{"a sigmoid gate", func(c *Config) { c.Flags2 |= Flag2DeltaSigmoidGate }, DeltaDecaySoftplus(), DeltaGateSigmoid()},
		{"a per-channel decay's architecture", func(c *Config) { c.Arch = ArchKimiLinear }, DeltaDecaySoftplus(), DeltaGateSigmoid()},
	} {
		old := base()
		tc.edit(old)
		if err := old.olderDelta(); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !reflect.DeepEqual(old.DeltaDecay, tc.decay) || !reflect.DeepEqual(old.DeltaGate, tc.gated) {
			t.Errorf("%s: stated decay %+v and gate %+v", tc.name, old.DeltaDecay, old.DeltaGate)
		}
	}
	plain := base()
	plain.LayerKinds = nil
	if err := plain.olderDelta(); err != nil || plain.DeltaDecay != nil {
		t.Errorf("a model with no delta rule stated %+v (%v)", plain.DeltaDecay, err)
	}

	for _, tc := range []struct {
		name string
		edit func(c *Config)
		want string
	}{
		{"a decay alone", func(c *Config) { c.DeltaDecay = DeltaDecaySoftplus() }, "without its gate"},
		{"no delta rule", func(c *Config) {
			c.LayerKinds = nil
			c.DeltaDecay, c.DeltaGate = DeltaDecaySoftplus(), DeltaGateSiLU()
		}, "no delta rule"},
		{"a gate of one input", func(c *Config) {
			c.DeltaDecay, c.DeltaGate = DeltaDecaySoftplus(), DeltaDecaySoftplus()
		}, "gate over 1"},
		{"a positive bound", func(c *Config) { c.KDALowerBound = 2 }, "a bound is negative"},
	} {
		bad := base()
		tc.edit(bad)
		if _, err := decodeConfig(encodeConfig(bad)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.name, err, tc.want)
		}
	}
}
