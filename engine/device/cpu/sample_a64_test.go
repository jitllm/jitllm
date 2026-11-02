//go:build arm64

package cpu

import (
	"math"
	"testing"
)

// The sampler's kernels on NEON, held to the same oracle as the two x86 tiers
// (samplemodel_test.go). The NEON masked select, folds (FMAXV, SMINV, ADDV)
// and compares are different instructions, so the x86 gates say nothing about
// them. Run on arm64 hardware: cross-compile with GOOS=darwin GOARCH=arm64
// go test -c ./jit/cpu and run the binary there with -test.run '^TestSample'
// -test.v (every TestSample* gate).

func sampleA64Kernels(t *testing.T) (firstC, elig, sweep, draw, pen *Code) {
	t.Helper()
	for _, e := range []struct {
		dst           **Code
		first, idsMem bool
	}{{&firstC, true, false}, {&elig, false, false}, {&sweep, true, true}} {
		b, err := EmitSampleSegMax(e.first, e.idsMem)
		if err != nil {
			t.Fatalf("EmitSampleSegMax(%v,%v): %v", e.first, e.idsMem, err)
		}
		c, err := MapNamed(b, "sample_segmax_a64")
		if err != nil {
			t.Fatalf("mapping: %v", err)
		}
		t.Cleanup(func() { c.Close() })
		*e.dst = c
	}
	b, err := EmitSampleDraw()
	if err != nil {
		t.Fatalf("EmitSampleDraw: %v", err)
	}
	if draw, err = MapNamed(b, "sample_draw_a64"); err != nil {
		t.Fatalf("mapping the draw: %v", err)
	}
	t.Cleanup(func() { draw.Close() })
	b, err = EmitSamplePenalty()
	if err != nil {
		t.Fatalf("EmitSamplePenalty: %v", err)
	}
	if pen, err = MapNamed(b, "sample_penalty_a64"); err != nil {
		t.Fatalf("mapping the penalty: %v", err)
	}
	t.Cleanup(func() { pen.Close() })
	return
}

// TestSampleSegMaxMatchesTheModel is the AVX2 gate's arm64 twin: every row, at
// several segment lengths, in each of the three configurations the ordering
// uses.
func TestSampleSegMaxMatchesTheModel(t *testing.T) {
	for _, first := range []bool{true, false} {
		for _, idsMem := range []bool{false, true} {
			b, err := EmitSampleSegMax(first, idsMem)
			if err != nil {
				t.Fatalf("EmitSampleSegMax(%v,%v): %v", first, idsMem, err)
			}
			c, err := MapNamed(b, "sample_segmax_a64_gate")
			if err != nil {
				t.Fatalf("mapping: %v", err)
			}
			for _, row := range sampleRows() {
				n := len(row.v)
				for _, segLen := range []int{1, 2, 3, 4, 5, 7, 8, 9, 16, n} {
					if segLen > n {
						continue
					}
					segs := (n + segLen - 1) / segLen
					if segs*segLen != n {
						continue
					}
					for _, base := range []int32{0, 1, 1000} {
						ids := make([]int32, n)
						for i := range ids {
							ids[i] = base + int32(i)
							if idsMem {
								ids[i] = base + int32(i*i+i)
							}
						}
						for _, prev := range []struct {
							pv float32
							pi int32
						}{{float32(math.Inf(1)), sampleEmpty}, {5, 0}, {2, base + 3},
							{0, base}, {float32(math.Inf(-1)), base}, {-2.5, base + 1}} {
							wantV, wantI := segMaxGo(row.v, ids, segLen, segs, first, prev.pv, prev.pi)
							gotV, gotI := segMaxCall(c, row.v, ids, segLen, segs,
								int(base), first, idsMem, prev.pv, prev.pi)
							for s := range wantV {
								if math.Float32bits(gotV[s]) != math.Float32bits(wantV[s]) || gotI[s] != wantI[s] {
									t.Fatalf("neon first=%v idsMem=%v %s segLen=%d base=%d prev=(%v,%d) seg %d: "+
										"got (%v, %d), want (%v, %d)",
										first, idsMem, row.name, segLen, base,
										prev.pv, prev.pi, s, gotV[s], gotI[s], wantV[s], wantI[s])
								}
							}
						}
					}
				}
			}
			c.Close()
		}
	}
}

// TestSampleSegMaxWalksAWholeRowInOrder is the property the sampler depends on:
// driving the three kernels as nn.SampleOrder does takes every element out
// exactly once, in the strict total order.
func TestSampleSegMaxWalksAWholeRowInOrder(t *testing.T) {
	firstC, elig, sweep, _, _ := sampleA64Kernels(t)
	for _, row := range sampleRows() {
		for _, segLen := range []int{1, 2, 4, 8, 16} {
			gotV, gotI := sampleWalk(t, firstC, elig, sweep, row.v, segLen)
			sampleCheckWalk(t, "neon", row.name, segLen, row.v, gotV, gotI)
		}
	}
}

// TestSampleDrawMatchesTheGoLoops runs the NEON draw against the Go loops it
// replaced, over the cut boundaries and both disabled forms.
func TestSampleDrawMatchesTheGoLoops(t *testing.T) {
	_, _, _, draw, _ := sampleA64Kernels(t)
	inf := float32(math.Inf(1))
	rows := [][]float32{
		{1},
		{0.5, 0.5},
		{0.25, 0.25, 0.25, 0.25},
		{0.4, 0.3, 0.2, 0.05, 0.03, 0.02},
		{0.9, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002},
		{1, 0, 0, 0, 0},
		{0.34, 0.33, 0.33},
	}
	long := make([]float32, 257)
	for i := range long {
		long[i] = float32(1) / float32(len(long))
	}
	rows = append(rows, long)
	for _, p := range rows {
		for _, minp := range []float32{0, 0.01, 0.1, 0.5, 1} {
			for _, topp := range []float32{inf, 0.25, 0.5, p[0], 0.9, 1} {
				for _, u := range []float32{0, 0.001, 0.25, 0.5, 0.75, 0.999} {
					for _, sumAll := range []float32{-1, 1, 2} {
						want := drawGo(p, minp, topp, u, sumAll)
						if got := drawCall(draw, p, minp, topp, u, sumAll); got != want {
							t.Fatalf("neon draw n=%d minp=%v topp=%v u=%v sumAll=%v: got %+v, want %+v",
								len(p), minp, topp, u, sumAll, got, want)
						}
					}
				}
			}
		}
	}
}

// TestSamplePenaltyMatchesTheSignRule runs the NEON scatter against the rule,
// including both zeros.
func TestSamplePenaltyMatchesTheSignRule(t *testing.T) {
	_, _, _, _, pk := sampleA64Kernels(t)
	vals := []float32{2, -3, 0, float32(math.Copysign(0, -1)), 4, -8, 1.5, 7, 0.25, -0.25, 100, -100}
	offs := [][]int64{
		{0},
		{4 * 7},
		{4 * 2, 4 * 3},
		{4 * 0, 4 * 1, 4 * 2, 4 * 3, 4 * 4, 4 * 5, 4 * 6, 4 * 7, 4 * 8, 4 * 9, 4 * 10, 4 * 11},
		{4 * 11, 4 * 0, 4 * 5},
	}
	for _, off := range offs {
		for _, pen := range []float32{1.1, 1.5, 2, 4, 100} {
			want := penaltyGo(vals, off, pen)
			got := penaltyCall(pk, vals, off, pen)
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("neon penalty off=%v pen=%v at %d: got %v, want %v",
						off, pen, i, got[i], want[i])
				}
			}
		}
	}
}
