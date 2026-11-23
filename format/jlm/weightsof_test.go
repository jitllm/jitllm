package jlm

import (
	"slices"
	"testing"
)

// TestWeightsOfRoundTrips holds the pages blocks read to the record: an
// unrolled loop's statement survives encoding, and a statement of the wrong
// length, or one reading a page before the page ahead of it is taken, is
// refused.
func TestWeightsOfRoundTrips(t *testing.T) {
	base := Config{NLayer: 6, NEmbd: 8, NHead: 2, NKVHead: 2, HeadDim: 4, NRot: 4, NFFN: 16, NVocab: 32, NCtx: 64, RMSEps: 1e-6}
	loop := base
	loop.WeightsOf = []uint32{0, 1, 2, 1, 2, 3}
	got, err := decodeConfig(encodeConfig(&loop))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.WeightsOf, loop.WeightsOf) {
		t.Fatalf("decoded %v, encoded %v", got.WeightsOf, loop.WeightsOf)
	}
	for li, want := range []int{0, 1, 2, 1, 2, 3} {
		if p := got.PageOf(li); p != want {
			t.Fatalf("block %d reads page %d, want %d", li, p, want)
		}
	}
	if got, err := decodeConfig(encodeConfig(&base)); err != nil || got.WeightsOf != nil || got.PageOf(4) != 4 {
		t.Fatalf("no statement: %v, %v, page %d", err, got.WeightsOf, got.PageOf(4))
	}
	for _, bad := range [][]uint32{{0, 1, 2, 1, 2}, {0, 2, 1, 1, 2, 3}, {1, 0, 1, 0, 1, 0}} {
		c := base
		c.WeightsOf = bad
		if _, err := decodeConfig(encodeConfig(&c)); err == nil {
			t.Errorf("%v decoded", bad)
		}
	}
}
