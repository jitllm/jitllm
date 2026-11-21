package jlm

import (
	"math"
	"reflect"
	"testing"
)

// TestClampTailRoundTrips holds DBRX's clip_qkv to the optional-tail contract
// the Llama 4 and Granite groups already keep: it round-trips, it implies both
// groups ahead of it (so a reader never has to guess which group a lone tail
// is), a model without it writes the record it always wrote, and a negative,
// NaN or infinite clamp -- which would clamp everything to NaN or nothing --
// is refused.
func TestClampTailRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 3, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 16, NFFN: 32, NVocab: 100352, NExpert: 4, NExpertUsed: 2,
			NFFNExp: 32, RMSEps: 1e-5, RopeBase: 5e5, EmbdScale: 1, Flags: FlagLayerNorm | FlagRopeNeox}
	}
	plain := encodeConfig(base())
	d := base()
	d.ClampKQV = 8
	rec := encodeConfig(d)
	if len(rec) != len(plain)+16+12+4 {
		t.Fatalf("the tail is %d bytes, want 32 (Llama 4's group, Granite's, then the clamp)", len(rec)-len(plain))
	}
	got, err := decodeConfig(rec)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, d) {
		t.Errorf("round trip:\n got %+v\nwant %+v", got, d)
	}
	if got, err = decodeConfig(plain); err != nil || got.ClampKQV != 0 {
		t.Errorf("a tailless record read as clamp %v (%v)", got.ClampKQV, err)
	}
	if _, err := decodeConfig(rec[:len(rec)-4]); err != nil {
		t.Errorf("the record without its clamp is a valid Granite-tailed one: %v", err)
	}
	if _, err := decodeConfig(rec[:len(rec)-2]); err == nil {
		t.Error("a truncated clamp decoded")
	}
	for _, bad := range []float32{-8, float32(math.NaN()), float32(math.Inf(1))} {
		b := base()
		b.ClampKQV = bad
		if _, err := decodeConfig(encodeConfig(b)); err == nil {
			t.Errorf("a clamp of %g decoded", bad)
		}
	}
}
