package convert

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestConverterStatesTheRotaryTable holds each stated layout's pair map, as
// the converter states it, against one written out by hand from the
// references (transformers' mrope_section split, Qwen3-VL's
// apply_interleaved_mrope, HunyuanVL's XD-RoPE), each picture layout's
// family against get_rope_index's, and states nothing for sections the
// rotary cannot hold.
func TestConverterStatesTheRotaryTable(t *testing.T) {
	for _, tc := range []struct {
		name       string
		layout     string
		nrot       uint32
		sec        [4]uint32
		axes, freq []int
		side       bool
	}{
		{"contiguous", "", 12, [4]uint32{1, 2, 3, 0}, []int{0, 1, 1, 2, 2, 2}, []int{0, 1, 2, 3, 4, 5}, true},
		{"interleaved", "interleaved", 12, [4]uint32{2, 2, 2, 0}, []int{0, 1, 2, 0, 1, 2}, []int{0, 1, 2, 3, 4, 5}, true},
		{"interleaved short", "interleaved", 12, [4]uint32{2, 1, 1, 0}, []int{0, 1, 2, 0, 0, 0}, []int{0, 1, 2, 3, 4, 5}, true},
		{"over the elements", "xd", 8, [4]uint32{1, 1, 1, 1}, []int{0, 0, 1, 1, 2, 2, 3, 3}, []int{0, 1, 2, 3, 0, 1, 2, 3}, false},
	} {
		c := &Config{Stated: &Statement{ropeLayout: tc.layout}, Config: &jlm.Config{NRot: tc.nrot, RopeSections: tc.sec}}
		stateRopeTable(c)
		var axes, freq []int
		for _, p := range c.RopeTable {
			axes, freq = append(axes, p.Axis), append(freq, p.Freq)
		}
		if !slices.Equal(axes, tc.axes) || !slices.Equal(freq, tc.freq) {
			t.Errorf("%s: axes %v frequencies %v, want %v %v", tc.name, axes, freq, tc.axes, tc.freq)
		}
		if c.PictureRope.Side != tc.side || c.PictureRope.Axes[0].Pos == tc.side {
			t.Errorf("%s: pictures laid out %+v", tc.name, c.PictureRope)
		}
	}
	for _, tc := range []struct {
		name   string
		layout string
		nrot   uint32
		sec    [4]uint32
	}{
		{"short of the pairs", "", 12, [4]uint32{1, 1, 0, 0}},
		{"past the pairs", "", 12, [4]uint32{3, 3, 3, 0}},
		{"elements short", "xd", 8, [4]uint32{1, 1, 1, 0}},
		{"an interleaved fourth axis", "interleaved", 12, [4]uint32{2, 2, 1, 1}},
		{"no rotary", "", 0, [4]uint32{1, 0, 0, 0}},
		{"an odd rotary", "", 3, [4]uint32{1, 0, 0, 0}},
	} {
		c := &Config{Stated: &Statement{ropeLayout: tc.layout}, Config: &jlm.Config{NRot: tc.nrot, RopeSections: tc.sec}}
		stateRopeTable(c)
		if c.RopeTable != nil || c.PictureRope != (jlm.PictureRope{}) {
			t.Errorf("%s: sections %v over %d stated %v, %+v", tc.name, tc.sec, tc.nrot, c.RopeTable, c.PictureRope)
		}
	}
	c := &Config{Stated: &Statement{}, Config: &jlm.Config{NRot: 64}}
	if stateRopeTable(c); c.RopeTable != nil {
		t.Fatalf("a rotary of one position states %v", c.RopeTable)
	}
}
