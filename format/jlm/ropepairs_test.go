package jlm

import (
	"slices"
	"testing"
)

// TestRopePairsReadsTheStatedLayouts holds each layout's pair map against one
// written out by hand, and refuses sections that cannot be one.
func TestRopePairsReadsTheStatedLayouts(t *testing.T) {
	for _, tc := range []struct {
		name       string
		nrot       uint32
		sec        [4]uint32
		flags      Flags
		axes, freq []int
	}{
		{"contiguous", 12, [4]uint32{1, 2, 3, 0}, 0, []int{0, 1, 1, 2, 2, 2}, []int{0, 1, 2, 3, 4, 5}},
		{"interleaved", 12, [4]uint32{2, 2, 2, 0}, FlagRopeInterleaved, []int{0, 1, 2, 0, 1, 2}, []int{0, 1, 2, 3, 4, 5}},
		{"interleaved short", 12, [4]uint32{2, 1, 1, 0}, FlagRopeInterleaved, []int{0, 1, 2, 0, 0, 0}, []int{0, 1, 2, 3, 4, 5}},
		{"over the elements", 8, [4]uint32{1, 1, 1, 1}, FlagRopeXD, []int{0, 0, 1, 1, 2, 2, 3, 3}, []int{0, 1, 2, 3, 0, 1, 2, 3}},
	} {
		c := Config{NRot: tc.nrot, RopeSections: tc.sec, Flags: tc.flags}
		pairs, err := c.RopePairs()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		var axes, freq []int
		for _, p := range pairs {
			axes, freq = append(axes, p.Axis), append(freq, p.Freq)
		}
		if !slices.Equal(axes, tc.axes) || !slices.Equal(freq, tc.freq) {
			t.Errorf("%s: axes %v frequencies %v, want %v %v", tc.name, axes, freq, tc.axes, tc.freq)
		}
	}
	for _, tc := range []struct {
		name  string
		nrot  uint32
		sec   [4]uint32
		flags Flags
	}{
		{"short of the pairs", 12, [4]uint32{1, 1, 0, 0}, 0},
		{"past the pairs", 12, [4]uint32{3, 3, 3, 0}, 0},
		{"elements short", 8, [4]uint32{1, 1, 1, 0}, FlagRopeXD},
		{"an interleaved fourth axis", 12, [4]uint32{2, 2, 1, 1}, FlagRopeInterleaved},
		{"both layouts", 12, [4]uint32{2, 2, 2, 0}, FlagRopeInterleaved | FlagRopeXD},
		{"no rotary", 0, [4]uint32{1, 0, 0, 0}, 0},
		{"an odd rotary", 3, [4]uint32{1, 0, 0, 0}, 0},
	} {
		c := Config{NRot: tc.nrot, RopeSections: tc.sec, Flags: tc.flags}
		if pairs, err := c.RopePairs(); err == nil {
			t.Errorf("%s: sections %v over %d read as %v", tc.name, tc.sec, tc.nrot, pairs)
		}
	}
	if pairs, err := (&Config{NRot: 64}).RopePairs(); pairs != nil || err != nil {
		t.Fatalf("a rotary of one position reads %v, %v", pairs, err)
	}
}
