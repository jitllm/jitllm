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
		// Scalars read by a block's index: refused where the index would
		// say otherwise than the layer's.
		{[]int{0, 1, 2, 3, 1, 2}, func(c *jlm.Config) { c.SWAWindow, c.SWAPeriod = 8, 2 }, "sliding window's period"},
		{[]int{0, 1, 2, 1, 2, 3}, func(c *jlm.Config) {
			c.SWAWindow, c.SWAPeriod, c.Flags2 = 8, 3, c.Flags2|jlm.Flag2SWAGlobalFirst
		}, "sliding window's period"},
		{[]int{0, 1, 2, 1, 2, 3}, func(c *jlm.Config) { c.NExpert, c.NDenseLead = 4, 2 }, "leading dense blocks"},
		{[]int{0, 1, 2, 3, 1, 2}, func(c *jlm.Config) { c.NExpert, c.MoEStep = 4, 2 }, "mixture's step"},
		{[]int{0, 1, 2, 1, 2, 3}, func(c *jlm.Config) { c.NHashLayers = 2 }, "hash-routed"},
		{[]int{0, 1, 2, 3, 1, 2, 3}, func(c *jlm.Config) { c.NKVShared = 1 }, "block count"},
		{[]int{0, 1, 2, 3, 1, 2, 3}, func(c *jlm.Config) { c.AttnResBlock = 2 }, "residual checkpoint"},
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
	// A period the loop's passes keep: blocks 4 and 5 run layers 2 and 3,
	// sliding as they do, and a leading dense block stays the only dense one.
	s = src()
	s.Config.SWAWindow, s.Config.SWAPeriod, s.Config.NExpert, s.Config.NDenseLead = 8, 2, 4, 1
	if err := UnrollSource(s, []int{0, 1, 2, 3, 2, 3}); err != nil {
		t.Errorf("a loop keeping the period's answer: %v", err)
	}
	if v, err := exprInts("unroll", nil); v != nil || err != nil {
		t.Errorf("a null unroll: %v, %v", v, err)
	}
	if _, err := exprInts("unroll", []any{1.0, 0.5}); err == nil {
		t.Error("an unroll naming layer 0.5 was taken")
	}
}

// TestLoopEvaluates holds loop() to the order a looped model's blocks run its
// layers in, and refuses a loop outside the model or of no pass.
func TestLoopEvaluates(t *testing.T) {
	env := mjEnv(map[string]any{"n": 8.0})
	x, err := parseExpr("loop(n, 2, 6, 2)")
	if err != nil {
		t.Fatal(err)
	}
	v, err := evalMJ(x, env)
	if err != nil {
		t.Fatal(err)
	}
	got, err := exprInts("loop", v)
	if err != nil || !slices.Equal(got, []int{0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7}) {
		t.Fatalf("loop(8, 2, 6, 2) is %v (%v)", got, err)
	}
	for _, bad := range []string{"loop(n, 6, 2, 2)", "loop(n, 2, 9, 2)", "loop(n, 2, 6, 0)"} {
		x, err := parseExpr(bad)
		if err != nil {
			t.Fatal(err)
		}
		if v, err := evalMJ(x, env); err == nil {
			t.Errorf("%s evaluated to %v", bad, v)
		}
	}
}

// TestFlipReversesAnAxis holds flip to an axis's indices last first, and to
// the taps the dynamic convolution's flip makes of a reference's newest-first
// (head, tap) rows: the other axes stay where they are.
func TestFlipReversesAnAxis(t *testing.T) {
	env := mjEnv(map[string]any{"heads": 2.0, "k": 3.0})
	v, err := mjRun(t, env, `[{"flip": 1}]`, mjF32([]uint64{2, 3}, 1, 2, 3, 4, 5, 6))
	if err != nil {
		t.Fatal(err)
	}
	if got := mjFloats(v); !slices.Equal(got, []float32{3, 2, 1, 6, 5, 4}) || !slices.Equal(v.shape, []uint64{2, 3}) {
		t.Fatalf("flipped to %v %v", v.shape, got)
	}
	v, err = mjRun(t, env, `[{"reshape": ["heads", "k", -1]}, {"flip": 1}, {"reshape": ["heads * k", -1]}]`,
		mjF32([]uint64{6, 2}, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61))
	if err != nil {
		t.Fatal(err)
	}
	if got := mjFloats(v); !slices.Equal(got, []float32{30, 31, 20, 21, 10, 11, 60, 61, 50, 51, 40, 41}) || !slices.Equal(v.shape, []uint64{6, 2}) {
		t.Fatalf("rows flipped within each head to %v %v", v.shape, got)
	}
	if _, err := mjRun(t, env, `[{"flip": 2}]`, mjF32([]uint64{2, 3}, 1, 2, 3, 4, 5, 6)); err == nil {
		t.Fatal("a flip of an axis the tensor lacks was taken")
	}
}

// TestEveryEvaluates holds every() to a class's default layer list, one
// kind every period layers, and refuses a position outside the period.
func TestEveryEvaluates(t *testing.T) {
	env := mjEnv(map[string]any{"n": 8.0})
	x, err := parseExpr("every(n, 3, 2, 'full', 'sliding')")
	if err != nil {
		t.Fatal(err)
	}
	v, err := evalMJ(x, env)
	if err != nil {
		t.Fatal(err)
	}
	want := []any{"sliding", "sliding", "full", "sliding", "sliding", "full", "sliding", "sliding"}
	if !slices.Equal(v.([]any), want) {
		t.Fatalf("every: %v, want %v", v, want)
	}
	for _, bad := range []string{"every(n, 3, 3, 1, 0)", "every(n, 0, 0, 1, 0)"} {
		x, err := parseExpr(bad)
		if err == nil {
			_, err = evalMJ(x, env)
		}
		if err == nil {
			t.Errorf("%s evaluated", bad)
		}
	}
}
