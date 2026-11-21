package jlm

import (
	"slices"
	"testing"
)

// TestKVSourceIsTheRuntimesRule holds the container's KV-sharing rule to
// what the runtime asks of it: only an attending trunk block of the last
// NKVShared shares, and its source is the last attending block before the
// run of its own kind, sliding or global, under either window period.
func TestKVSourceIsTheRuntimesRule(t *testing.T) {
	cases := []struct {
		name string
		c    Config
		want []int
	}{
		{"last of each period global", Config{NLayer: 6, NKVShared: 2, SWAWindow: 8, SWAPeriod: 3},
			[]int{0, 1, 2, 3, 3, 2}},
		{"first of each period global", Config{NLayer: 6, NKVShared: 2, SWAWindow: 8, SWAPeriod: 3, Flags2: Flag2SWAGlobalFirst},
			[]int{0, 1, 2, 3, 2, 2}},
		{"a recurrent block keeps no history", Config{NLayer: 6, NKVShared: 3,
			LayerKinds: []LayerKind{LayerFullAttn, LayerFullAttn, LayerSSD, LayerFullAttn, LayerSSD, LayerFullAttn}},
			[]int{0, 1, 2, 1, 4, 1}},
		{"a prediction block keeps its own", Config{NLayer: 4, NKVShared: 2},
			[]int{0, 1, 1, 1, 4}},
	}
	for _, tc := range cases {
		var got []int
		for b := range len(tc.want) {
			got = append(got, tc.c.KVSource(b))
		}
		if !slices.Equal(got, tc.want) {
			t.Errorf("%s: sources %v, want %v", tc.name, got, tc.want)
		}
	}
}
