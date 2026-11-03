//go:build jitllmbench

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// JITLLM_* variables select its parameters.

package model

import (
	"os"
	"testing"
	"time"
)

// TestTowerFusionCeiling is TestFusionCeiling for the VISION TOWER, which that
// measurement does not reach: it drives Forward on a text model and a tower runs
// through Encode. It prices the region count with a per-region cost measured in
// the same pass, and prints it per layer as well as per encode, since a tower's
// region count can scale with the patch count.
func TestTowerFusionCeiling(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates and measures")
	}
	if _, err := os.Stat(towerPath()); err != nil {
		t.Skip("no mmproj present")
	}
	// A dispatch measurement needs a quiet box.
	if p, err := cpuPressure(); err == nil && p > 25 {
		t.Skipf("CPU pressure is %.1f%% over the last 10s", p)
	}
	vlm, tw := openTower(t)
	defer vlm.Close()
	s := tw.testState()
	defer s.Close()

	img := towerCb(tw.Cfg.ImageSz)
	// Warm: codegen, tuning and first-touch faults are not per-encode costs.
	if _, err := s.Encode(img); err != nil {
		t.Fatal(err)
	}

	const encodes = 3
	s.ResetRegions()
	start := time.Now()
	for i := 0; i < encodes; i++ {
		if _, err := s.Encode(img); err != nil {
			t.Fatal(err)
		}
	}
	elapsed := time.Since(start)
	par, ser := s.Regions()
	per := elapsed / encodes
	parPer := float64(par) / encodes
	nl := tw.Cfg.NLayer
	t.Logf("%d tower layers, %d patches: %.0f parallel regions/encode (%.0f per layer), "+
		"%.0f inline, %v/encode", nl, tw.Cfg.Patches(), parPer, parPer/float64(nl),
		float64(ser)/encodes, per)

	// The idle price (dispatchCost, empty regions) is an upper bound: in situ,
	// dispatch overlaps computation (see splitCost in
	// fusion_test.go). So report a band, not one number.
	const inSitu = 0.592
	idle := dispatchCost(t)
	cost := idle
	measured := false
	if split, iqr := splitCost(t, tw.Cfg.NEmbd*8, 8); split > 0 && iqr <= 0.10 {
		t.Logf("one region of real work (dispatch + straggler): %v  IQR/med %.1f%%", split, 100*iqr)
		cost, measured = split, true
	}
	if !measured {
		cost = time.Duration(float64(idle) * inSitu)
		t.Logf("split-cost measurement rejected; using the tree's in-situ factor %.3f", inSitu)
	}
	t.Logf("one region round trip: %v idle (UPPER BOUND), %v in situ (best estimate)", idle, cost)

	// Fusion cannot recover more of the encode than dispatch costs.
	total := time.Duration(parPer * float64(cost))
	hi := time.Duration(parPer * float64(idle))
	t.Logf("dispatch is %v..%v of a %v encode = %.0f..%.0f%% -- the CEILING on any "+
		"fusion, in situ to idle. NOT \"most of the encode\": the majority is arithmetic.",
		total, hi, per, 100*float64(total)/float64(per), 100*float64(hi)/float64(per))

	// A ViT layer's own ops issue about 12 regions; a matmul that falls to a
	// per-row loop issues ~2 per row. Ops are constant per layer and rows scale
	// with patches, so regions/layer tracking the patch count means the
	// dispatch is per row and fusing ops cannot touch it.
	perLayer := parPer / float64(nl)
	mm := 6 // q, k, v, o, up, down -- a ViT block's matrices
	perRow := perLayer / float64(mm*tw.Cfg.Patches())
	t.Logf("%.1f regions/layer against %d matmuls x %d patches = %.2f dispatches per matmul ROW",
		perLayer, mm, tw.Cfg.Patches(), perRow)
	if perRow < 0.5 {
		t.Logf("  regions do NOT scale with rows; the layer's op count dominates and " +
			"whole-layer fusion is the lever")
	} else {
		t.Logf("  ★ REGIONS SCALE WITH ROWS, NOT OPS. Fusing the layer's dozen ops "+
			"cannot reach %.0f%% of an encode. The lever is a BATCHED PACKED GEMM "+
			"so a projection is one dispatch instead of %d.",
			100*float64(hi)/float64(per), tw.Cfg.Patches())
	}

	// What the textbook fusion levers are worth here.
	for _, d := range []struct {
		saved float64
		what  string
	}{
		{2, "siblings only (q/k/v into one region)"},
		{6, "siblings + elementwise epilogues (2 residuals, GELU, the score scale)"},
	} {
		pct := 100 * d.saved * float64(nl) * float64(cost) / float64(per)
		t.Logf("  save %4.1f regions/layer -> %5.2f%% of an encode   (%s)", d.saved, pct, d.what)
	}
}
