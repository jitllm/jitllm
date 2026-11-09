package jlm

import (
	"math"
	"reflect"
	"testing"
)

// TestScalesTailRoundTrips holds Granite's three scales to the same contract as
// Llama 4's tail: they round-trip, they imply the Llama 4 group ahead of them,
// a model without them writes the record it always wrote, and a bad value or a
// truncated tail is refused rather than read as a default.
func TestScalesTailRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 4, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 16, NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 1e7, EmbdScale: 12}
	}
	plain := encodeConfig(base())
	g := base()
	g.AttnScale, g.ResidualScale, g.LogitScale = 0.015625, 0.22, 8
	rec := encodeConfig(g)
	if len(rec) != len(plain)+16+12 {
		t.Fatalf("the tail is %d bytes, want 28 (the Llama 4 group, then the scales)", len(rec)-len(plain))
	}
	got, err := decodeConfig(rec)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, g) {
		t.Errorf("round trip:\n got %+v\nwant %+v", got, g)
	}
	// Llama 4 and the scales together.
	g.MoEStep, g.AttnTempScale, g.AttnTempFloor, g.AttnTempOffset = 2, 0.1, 8192, 1
	if got, err = decodeConfig(encodeConfig(g)); err != nil || !reflect.DeepEqual(got, g) {
		t.Errorf("both tails: %v\n got %+v\nwant %+v", err, got, g)
	}
	if got, err = decodeConfig(plain); err != nil || got.AttnScale != 0 || got.ResidualScale != 0 || got.LogitScale != 0 {
		t.Errorf("a tailless record read as %+v (%v)", got, err)
	}
	if _, err := decodeConfig(rec[:len(rec)-4]); err == nil {
		t.Error("a truncated scale tail decoded")
	}
	for _, bad := range []float32{-8, float32(math.NaN()), float32(math.Inf(1))} {
		b := base()
		b.LogitScale = bad
		if _, err := decodeConfig(encodeConfig(b)); err == nil {
			t.Errorf("a logit scale of %g decoded", bad)
		}
	}
}
