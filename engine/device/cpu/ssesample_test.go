//go:build amd64 && jitllmtest

package cpu

import (
	"math"
	"testing"
)

// The sampler's kernels on the forced SSE tier. sample_test.go already
// calls the SSE kernels; this arm runs them through the table under the force
// and asserts the SSE tier mapped kernels and no AVX2 kernel was mapped.
func TestSampleKernelsOnTheForcedSSETier(t *testing.T) {
	withTier(t, TierSSE)
	before := MappedByTier()
	em := EmittersFor(HostTier())

	row := []float32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	ids := make([]int32, len(row))
	for i := range ids {
		ids[i] = int32(i)
	}
	const segLen = 11

	for _, e := range []struct{ first, idsMem bool }{{true, false}, {false, false}, {true, true}} {
		b, err := em.SampleSegMax(e.first, e.idsMem)
		if err != nil {
			t.Fatalf("SampleSegMax(%v,%v): %v", e.first, e.idsMem, err)
		}
		if got := KernelTier(b); got != TierSSE {
			t.Fatalf("the forced tier handed a segment maximum declaring %v", got)
		}
		requireSSEKernel(t, "sse_sample_segmax", b)
		c, err := MapNamed(b, "sample_segmax_forced_sse")
		if err != nil {
			t.Fatalf("mapping: %v", err)
		}
		pv, pi := float32(math.Inf(1)), sampleEmpty
		if !e.first {
			pv, pi = 5, 4
		}
		wantV, wantI := segMaxGo(row, ids, segLen, 1, e.first, pv, pi)
		gotV, gotI := segMaxCall(c, row, ids, segLen, 1, 0, e.first, e.idsMem, pv, pi)
		if math.Float32bits(gotV[0]) != math.Float32bits(wantV[0]) || gotI[0] != wantI[0] {
			t.Errorf("forced SSE segmax(%v,%v): got (%v, %d), want (%v, %d)",
				e.first, e.idsMem, gotV[0], gotI[0], wantV[0], wantI[0])
		}
		c.Close()
	}

	p := []float32{0.4, 0.3, 0.2, 0.06, 0.04}
	b, err := em.SampleDraw()
	if err != nil {
		t.Fatalf("SampleDraw: %v", err)
	}
	requireSSEKernel(t, "sse_sample_draw", b)
	c, err := MapNamed(b, "sample_draw_forced_sse")
	if err != nil {
		t.Fatalf("mapping: %v", err)
	}
	for _, u := range []float32{0, 0.25, 0.5, 0.9} {
		if got, want := drawCall(c, p, 0.1, 0.9, u, -1), drawGo(p, 0.1, 0.9, u, -1); got != want {
			t.Errorf("forced SSE draw u=%v: got %+v, want %+v", u, got, want)
		}
	}
	c.Close()

	b, err = em.SamplePenalty()
	if err != nil {
		t.Fatalf("SamplePenalty: %v", err)
	}
	requireSSEKernel(t, "sse_sample_penalty", b)
	c, err = MapNamed(b, "sample_penalty_forced_sse")
	if err != nil {
		t.Fatalf("mapping: %v", err)
	}
	pv := []float32{2, -3, 0, 4, -8}
	off := []int64{0, 4, 8, 16}
	got := penaltyCall(c, pv, off, 2)
	want := penaltyGo(pv, off, 2)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Errorf("forced SSE penalty at %d: got %v, want %v", i, got[i], want[i])
		}
	}
	c.Close()

	after := MappedByTier()
	if d := after[TierAVX2] - before[TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped under the forced SSE tier", d)
	}
	if after[TierSSE] == before[TierSSE] {
		t.Error("no SSE-tier kernel was mapped -- this arm did not run the tier it names")
	}
}
