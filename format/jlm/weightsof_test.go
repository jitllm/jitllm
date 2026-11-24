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

// TestWindowsRoundTrip holds each block's own window to the record: a
// pattern no period describes survives encoding and is what Sliding and
// WindowAt read, with shared pages beside it or none; a statement of the
// wrong length, one beside a period and one past the model's window are
// refused.
func TestWindowsRoundTrip(t *testing.T) {
	base := Config{NLayer: 4, NEmbd: 8, NHead: 2, NKVHead: 2, HeadDim: 4, NRot: 4, NFFN: 16, NVocab: 32, NCtx: 64, RMSEps: 1e-6, SWAWindow: 8}
	for _, shared := range [][]uint32{nil, {0, 1, 0, 1}} {
		c := base
		c.Windows, c.WeightsOf = []uint32{8, 0, 0, 8}, shared
		got, err := decodeConfig(encodeConfig(&c))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got.Windows, c.Windows) || !slices.Equal(got.WeightsOf, shared) {
			t.Fatalf("decoded windows %v pages %v, encoded %v %v", got.Windows, got.WeightsOf, c.Windows, shared)
		}
		for li, want := range []bool{true, false, false, true} {
			if got.Sliding(li) != want || got.WindowAt(li) != c.Windows[li] {
				t.Fatalf("block %d: sliding %v window %d, want %v %d", li, got.Sliding(li), got.WindowAt(li), want, c.Windows[li])
			}
		}
	}
	for _, bad := range []func(*Config){
		func(c *Config) { c.Windows = []uint32{8, 0, 8} },
		func(c *Config) { c.Windows, c.SWAPeriod = []uint32{8, 0, 0, 8}, 2 },
		func(c *Config) { c.Windows = []uint32{8, 0, 0, 16} },
	} {
		c := base
		bad(&c)
		if _, err := decodeConfig(encodeConfig(&c)); err == nil {
			t.Errorf("windows %v (period %d) decoded", c.Windows, c.SWAPeriod)
		}
	}
}
