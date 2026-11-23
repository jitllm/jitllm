package convert

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestUnrollStatesALoop: a loop over layers 1..2 run twice gives each block
// its layer's page and per-layer statements, the source's tensors their
// page, and refuses a statement that leaves a layer unrun, names a layer the
// source lacks or a per-layer list of the wrong length.
func TestUnrollStatesALoop(t *testing.T) {
	src := func() *jlm.Source {
		c := &jlm.Config{NLayer: 4, LayerKinds: []jlm.LayerKind{0, 1, 0, 1}, ActLayers: []uint8{0, 1, 2, 3}}
		s := &jlm.Source{Config: c}
		for b := int32(-1); b < 4; b++ {
			s.Tensors = append(s.Tensors, jlm.Tensor{Block: b})
		}
		return s
	}
	s := src()
	if err := UnrollSource(s, []int{0, 1, 2, 1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	c := s.Config
	if c.NLayer != 6 || !slices.Equal(c.WeightsOf, []uint32{0, 1, 2, 1, 2, 3}) ||
		!slices.Equal(c.ActLayers, []uint8{0, 1, 2, 1, 2, 3}) || !slices.Equal(c.LayerKinds, []jlm.LayerKind{0, 1, 0, 1, 0, 1}) {
		t.Fatalf("unrolled to %d blocks, pages %v, acts %v, kinds %v", c.NLayer, c.WeightsOf, c.ActLayers, c.LayerKinds)
	}
	var blocks []int32
	for _, tn := range s.Tensors {
		blocks = append(blocks, tn.Block)
	}
	if !slices.Equal(blocks, []int32{-1, 0, 1, 2, 3}) {
		t.Fatalf("tensors at %v", blocks)
	}

	// A loop whose first pass runs its layers out of order: the pages follow
	// the blocks' first reads, and the tensors follow their pages.
	s = src()
	if err := UnrollSource(s, []int{0, 2, 1, 2, 1, 3}); err != nil {
		t.Fatal(err)
	}
	blocks = blocks[:0]
	for _, tn := range s.Tensors {
		blocks = append(blocks, tn.Block)
	}
	if !slices.Equal(s.Config.WeightsOf, []uint32{0, 1, 2, 1, 2, 3}) || !slices.Equal(blocks, []int32{-1, 0, 2, 1, 3}) ||
		!slices.Equal(s.Config.ActLayers, []uint8{0, 2, 1, 2, 1, 3}) {
		t.Fatalf("pages %v, tensors %v, acts %v", s.Config.WeightsOf, blocks, s.Config.ActLayers)
	}

	for _, bad := range []struct {
		order []int
		edit  func(*jlm.Config)
		why   string
	}{
		{[]int{0, 1, 1, 3}, nil, "runs 3 of"},
		{[]int{0, 1, 2, 3, 4}, nil, "runs layer 4"},
		{[]int{0, 1, 2, 3, 1}, func(c *jlm.Config) { c.ActLayers = c.ActLayers[:3] }, "activation layers has 3"},
	} {
		s := src()
		if bad.edit != nil {
			bad.edit(s.Config)
		}
		err := UnrollSource(s, bad.order)
		if err == nil || !strings.Contains(err.Error(), bad.why) {
			t.Errorf("%v: %v, want a refusal naming %q", bad.order, err, bad.why)
		}
	}
	if v, err := exprInts("unroll", nil); v != nil || err != nil {
		t.Errorf("a null unroll: %v, %v", v, err)
	}
	if _, err := exprInts("unroll", []any{1.0, 0.5}); err == nil {
		t.Error("an unroll naming layer 0.5 was taken")
	}
}
