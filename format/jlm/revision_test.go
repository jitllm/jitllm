package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestRevisionTailRoundTrips holds the seventeenth tail to the config's
// contract: the revision and what it states round-trip, a config without
// them reads as revision 0, and a per-layer statement of the wrong length is
// refused.
func TestRevisionTailRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 4, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 16, NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 1e4, SWAWindow: 64, SWAPeriod: 2}
	}
	c := base()
	c.Revision, c.RopeScaleSWA = StatementRevision, 1
	for li := range int(c.NLayer) {
		c.BidirLayers = append(c.BidirLayers, c.Sliding(li))
	}
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if got.Revision != c.Revision || got.RopeScaleSWA != 1 || !reflect.DeepEqual(got.BidirLayers, []bool{true, false, true, false}) {
		t.Fatalf("round trip: revision %d, scale %v, layers %v", got.Revision, got.RopeScaleSWA, got.BidirLayers)
	}
	if plain, err := decodeConfig(encodeConfig(base())); err != nil || plain.Revision != 0 || plain.BidirLayers != nil {
		t.Fatalf("a config without the tail read as revision %d, layers %v (%v)", plain.Revision, plain.BidirLayers, err)
	}
	c.BidirLayers = c.BidirLayers[:3]
	if _, err := decodeConfig(encodeConfig(c)); err == nil || !strings.Contains(err.Error(), "3 layers' bidirectional statement for 4") {
		t.Fatalf("a short per-layer statement: %v", err)
	}
}
