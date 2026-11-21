package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestDeltaStatementRoundTrips holds the delta rule's stated decay and gate
// to the config's contract: they round-trip, and a statement no delta rule
// reads is refused. A container an older writer made states neither; the
// model states them from the file (model.olderDelta).
func TestDeltaStatementRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 4, NEmbd: 64, NHead: 4, NKVHead: 2, HeadDim: 16, NRot: 16,
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
	} {
		bad := base()
		tc.edit(bad)
		if _, err := decodeConfig(encodeConfig(bad)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.name, err, tc.want)
		}
	}
}
