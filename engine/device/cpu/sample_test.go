//go:build amd64

package cpu

import (
	"fmt"
	"math"
	"testing"
)

// The sampler's kernel gate, for both x86 tiers.
//
// The bar is bit equality with an independent Go model: the segment maximum
// returns one of its inputs, the draw returns an index, and the cumulative is
// a sequential float32 chain the model reproduces term for term (a last-bit
// difference is a different top-p cut). Both tiers run against the same
// oracle; their different lane grouping does not matter since the ordering
// does no arithmetic and the draw is sequential in lane 0.

// sampleMapPair maps both x86 tiers' bytes for one op, and checks the SSE ones
// declare the SSE tier and carry no VEX prefix before either executes. On a
// host without AVX2 only the SSE arm is mapped (tierArmRuns).
func sampleMapPair(t *testing.T, name string, avx, sse []byte) []struct {
	name string
	c    *Code
} {
	t.Helper()
	if got := KernelTier(sse); got != TierSSE {
		t.Fatalf("%s: the SSE kernel was generated for tier %v -- a VEX instruction "+
			"would fault on the host it exists for", name, got)
	}
	requireSSEKernel(t, "sse_"+name, sse)
	var out []struct {
		name string
		c    *Code
	}
	for _, e := range []struct {
		n string
		b []byte
	}{{"avx2", avx}, {"sse", sse}} {
		if !tierArmRuns(t, e.n+" "+name, e.b) {
			continue
		}
		c, err := MapNamed(e.b, name+"_"+e.n)
		if err != nil {
			t.Fatalf("mapping the %s %s: %v", e.n, name, err)
		}
		t.Cleanup(func() { c.Close() })
		out = append(out, struct {
			name string
			c    *Code
		}{e.n, c})
	}
	return out
}

// ---------------------------------------------------------------- the gates

// TestSampleSegMaxMatchesTheModel runs both tiers' segmented maximum over every
// row, at several segment lengths, in each of the three configurations the
// ordering uses: the initial all-eligible contiguous pass, the eligible rescan
// of one segment, and the all-eligible sweep of the summary, whose ids come
// from memory.
func TestSampleSegMaxMatchesTheModel(t *testing.T) {
	for _, first := range []bool{true, false} {
		for _, idsMem := range []bool{false, true} {
			avx, err := EmitSampleSegMax(first, idsMem)
			if err != nil {
				t.Fatalf("EmitSampleSegMax(%v,%v): %v", first, idsMem, err)
			}
			sse, err := EmitSampleSegMaxSSE(first, idsMem)
			if err != nil {
				t.Fatalf("EmitSampleSegMaxSSE(%v,%v): %v", first, idsMem, err)
			}
			arms := sampleMapPair(t, fmt.Sprintf("sample_segmax_%v_%v", first, idsMem), avx, sse)

			for _, row := range sampleRows() {
				n := len(row.v)
				for _, segLen := range []int{1, 2, 3, 4, 5, 7, 8, 9, 16, n} {
					if segLen > n {
						continue
					}
					segs := (n + segLen - 1) / segLen
					if segs*segLen != n {
						continue // the kernel takes ONE segment length
					}
					// Ids are contiguous from base in both modes; the summary
					// sweep's are not contiguous in the engine, so a permuted
					// set is used for the memory mode.
					for _, base := range []int32{0, 1, 1000} {
						ids := make([]int32, n)
						for i := range ids {
							ids[i] = base + int32(i)
						}
						if idsMem {
							// A scattered, strictly increasing id set: the
							// summary's ids are segment bests and are never
							// consecutive.
							for i := range ids {
								ids[i] = base + int32(i*i+i)
							}
						}
						for _, prev := range []struct {
							pv float32
							pi int32
						}{{float32(math.Inf(1)), sampleEmpty}, {5, 0}, {2, base + 3},
							{0, base}, {float32(math.Inf(-1)), base}, {-2.5, base + 1}} {
							wantV, wantI := segMaxGo(row.v, ids, segLen, segs, first, prev.pv, prev.pi)
							for _, arm := range arms {
								gotV, gotI := segMaxCall(arm.c, row.v, ids, segLen, segs,
									int(base), first, idsMem, prev.pv, prev.pi)
								for s := range wantV {
									if math.Float32bits(gotV[s]) != math.Float32bits(wantV[s]) || gotI[s] != wantI[s] {
										t.Fatalf("%s first=%v idsMem=%v %s segLen=%d base=%d prev=(%v,%d) seg %d: "+
											"got (%v, %d), want (%v, %d)",
											arm.name, first, idsMem, row.name, segLen, base,
											prev.pv, prev.pi, s, gotV[s], gotI[s], wantV[s], wantI[s])
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestSampleSegMaxWalksAWholeRowInOrder is the property the sampler actually
// depends on: driving the kernel the way nn.SampleOrder does takes every
// element out exactly once, in the strict total order -- higher value first,
// lower id on a tie -- and then reports the row exhausted. Each call can be
// right about its segment while the calls together duplicate or lose a token,
// which only a whole-row walk sees.
func TestSampleSegMaxWalksAWholeRowInOrder(t *testing.T) {
	type kern struct {
		name                string
		firstC, elig, sweep *Code
	}
	var arms []kern
	for _, tier := range []struct {
		name string
		f    func(first, idsMem bool) ([]byte, error)
	}{{"avx2", EmitSampleSegMax}, {"sse", EmitSampleSegMaxSSE}} {
		k := kern{name: tier.name}
		runs := true
		for _, e := range []struct {
			dst           **Code
			first, idsMem bool
		}{{&k.firstC, true, false}, {&k.elig, false, false}, {&k.sweep, true, true}} {
			b, err := tier.f(e.first, e.idsMem)
			if err != nil {
				t.Fatalf("%s(%v,%v): %v", tier.name, e.first, e.idsMem, err)
			}
			if !tierArmRuns(t, tier.name+" sample_walk", b) {
				runs = false
				break
			}
			c, err := MapNamed(b, "sample_walk")
			if err != nil {
				t.Fatalf("mapping: %v", err)
			}
			t.Cleanup(func() { c.Close() })
			*e.dst = c
		}
		if runs {
			arms = append(arms, k)
		}
	}
	for _, row := range sampleRows() {
		for _, segLen := range []int{1, 2, 4, 8, 16} {
			for _, arm := range arms {
				gotV, gotI := sampleWalk(t, arm.firstC, arm.elig, arm.sweep, row.v, segLen)
				sampleCheckWalk(t, arm.name, row.name, segLen, row.v, gotV, gotI)
			}
		}
	}
}

// TestSampleDrawMatchesTheGoLoops runs both tiers' draw kernel against the Go
// loops it replaced, over the cut boundaries and both disabled forms.
func TestSampleDrawMatchesTheGoLoops(t *testing.T) {
	avx, err := EmitSampleDraw()
	if err != nil {
		t.Fatalf("EmitSampleDraw: %v", err)
	}
	sse, err := EmitSampleDrawSSE()
	if err != nil {
		t.Fatalf("EmitSampleDrawSSE: %v", err)
	}
	arms := sampleMapPair(t, "sample_draw", avx, sse)

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
	// A long one, to run several vectors of the two loops.
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
						for _, arm := range arms {
							got := drawCall(arm.c, p, minp, topp, u, sumAll)
							if got != want {
								t.Fatalf("%s draw n=%d minp=%v topp=%v u=%v sumAll=%v: got %+v, want %+v",
									arm.name, len(p), minp, topp, u, sumAll, got, want)
							}
						}
					}
				}
			}
		}
	}
}

// TestSamplePenaltyMatchesTheSignRule runs both tiers' scatter against the
// rule, including both zeros and an id repeated in the history.
func TestSamplePenaltyMatchesTheSignRule(t *testing.T) {
	avx, err := EmitSamplePenalty()
	if err != nil {
		t.Fatalf("EmitSamplePenalty: %v", err)
	}
	sse, err := EmitSamplePenaltySSE()
	if err != nil {
		t.Fatalf("EmitSamplePenaltySSE: %v", err)
	}
	arms := sampleMapPair(t, "sample_penalty", avx, sse)

	vals := []float32{2, -3, 0, float32(math.Copysign(0, -1)), 4, -8, 1.5, 7, 0.25, -0.25, 100, -100}
	offs := [][]int64{
		{0},
		{4 * 7},
		{4 * 2, 4 * 3},
		{4 * 0, 4 * 1, 4 * 2, 4 * 3, 4 * 4, 4 * 5, 4 * 6, 4 * 7, 4 * 8, 4 * 9, 4 * 10, 4 * 11},
		{4 * 7, 4 * 7, 4 * 7}, // a repeated id COMPOUNDS, and the caller must not repeat
		{4 * 11, 4 * 0, 4 * 5},
	}
	for _, off := range offs {
		for _, pen := range []float32{1.1, 1.5, 2, 4, 100} {
			want := penaltyGo(vals, off, pen)
			for _, arm := range arms {
				got := penaltyCall(arm.c, vals, off, pen)
				for i := range want {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("%s penalty off=%v pen=%v at %d: got %v, want %v",
							arm.name, off, pen, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestSampleKernelsRefuseTheirViolations: each mutation below is one a real
// edit could make, and each must be visible to the gates above.
func TestSampleKernelsRefuseTheirViolations(t *testing.T) {
	row := []float32{5, 5, 3, 3, 1, 9, 9, 2, 7}
	ids := make([]int32, len(row))
	for i := range ids {
		ids[i] = int32(i)
	}
	const segLen, segs = 3, 3

	// The ordering's own model, mutated.
	viol := []struct {
		name string
		f    func() ([]float32, []int32)
	}{
		{"a tie goes to the HIGHER id", func() ([]float32, []int32) {
			outV := make([]float32, segs)
			outI := make([]int32, segs)
			for s := 0; s < segs; s++ {
				best, bi := float32(math.Inf(-1)), sampleEmpty
				for j := s * segLen; j < (s+1)*segLen; j++ {
					if bi == sampleEmpty || row[j] >= best {
						best, bi = row[j], ids[j]
					}
				}
				outV[s], outI[s] = best, bi
			}
			return outV, outI
		}},
		{"an exhausted segment reports id 0", func() ([]float32, []int32) {
			v, i := segMaxGo(row, ids, segLen, segs, false, float32(math.Inf(-1)), 0)
			for s := range i {
				if i[s] == sampleEmpty {
					i[s] = 0
				}
			}
			return v, i
		}},
		{"the segment boundary is off by one", func() ([]float32, []int32) {
			return segMaxGo(row, ids, segLen+1, segs, true, 0, 0)
		}},
	}
	// The violations run against this host's tier: the SSE kernels on an
	// SSE-only host, the AVX2 ones above it.
	tab := EmittersFor(HostTier())
	b, err := tab.SampleSegMax(true, false)
	if err != nil {
		t.Fatal(err)
	}
	c, err := MapNamed(b, "sample_segmax_viol")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	be, err := tab.SampleSegMax(false, false)
	if err != nil {
		t.Fatal(err)
	}
	ce, err := MapNamed(be, "sample_segmax_viol_e")
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()

	gotV, gotI := segMaxCall(c, row, ids, segLen, segs, 0, true, false, 0, 0)
	eV, eI := segMaxCall(ce, row, ids, segLen, segs, 0, false, false, float32(math.Inf(-1)), 0)
	for _, v := range viol {
		wV, wI := v.f()
		same := true
		for s := range wV {
			g, gi := gotV[s], gotI[s]
			if v.name == "an exhausted segment reports id 0" {
				g, gi = eV[s], eI[s]
			}
			if math.Float32bits(g) != math.Float32bits(wV[s]) || gi != wI[s] {
				same = false
			}
		}
		if same {
			t.Errorf("violation %q is invisible to the segment-maximum gate", v.name)
		}
	}

	// The draw's, mutated.
	//
	// The rows are chosen so the cumulative lands exactly on zero: `r <= 0`
	// and `r < 0` differ only there ({0.5, 0.5} at u = 0.5).
	dp := [][]float32{
		{0.4, 0.3, 0.2, 0.06, 0.04},
		{0.5, 0.5},
		{0.25, 0.25, 0.25, 0.25},
	}
	bd, err := tab.SampleDraw()
	if err != nil {
		t.Fatal(err)
	}
	cd, err := MapNamed(bd, "sample_draw_viol")
	if err != nil {
		t.Fatal(err)
	}
	defer cd.Close()
	walk := func(p []float32, o drawGoOut, u float32, strict bool) drawGoOut {
		r := u * o.sum
		o.res, o.resolved = o.m-1, false
		for i := 0; i < o.m; i++ {
			r -= p[i]
			if (strict && r < 0) || (!strict && r <= 0) {
				o.res, o.resolved = i, true
				break
			}
		}
		return o
	}
	cuts := func(p []float32, minp, topp float32, minpLE, toppGT bool) drawGoOut {
		var o drawGoOut
		cut := minp * p[0]
		acc := float32(0)
		for i, v := range p {
			hit := v < cut
			if minpLE {
				hit = v <= cut
			}
			if !o.cut && hit {
				o.sum, o.m, o.cut = acc, i, true
			}
			acc += v
			hit = acc >= topp
			if toppGT {
				hit = acc > topp
			}
			if !o.cut && hit {
				o.sum, o.m, o.cut = acc, i+1, true
			}
		}
		if !o.cut {
			o.m, o.sum = len(p), acc
		}
		return o
	}
	dviol := []struct {
		name string
		f    func(p []float32, minp, topp, u float32) drawGoOut
	}{
		{"min-p cuts at <= instead of <", func(p []float32, minp, topp, u float32) drawGoOut {
			return walk(p, cuts(p, minp, topp, true, false), u, false)
		}},
		{"top-p cuts at > instead of >=", func(p []float32, minp, topp, u float32) drawGoOut {
			return walk(p, cuts(p, minp, topp, false, true), u, false)
		}},
		{"the walk stops at r < 0 instead of r <= 0", func(p []float32, minp, topp, u float32) drawGoOut {
			return walk(p, cuts(p, minp, topp, false, false), u, true)
		}},
		{"the mass is renormalized to 1", func(p []float32, minp, topp, u float32) drawGoOut {
			o := cuts(p, minp, topp, false, false)
			o.sum = 1
			return walk(p, o, u, false)
		}},
	}
	for _, v := range dviol {
		fired := false
		for _, p := range dp {
			for _, minp := range []float32{0, 0.15, 0.5} {
				for _, topp := range []float32{float32(math.Inf(1)), 0.4, 0.5, 0.7, 0.9} {
					for _, u := range []float32{0, 0.25, 0.5, 0.75, 0.9999} {
						if drawCall(cd, p, minp, topp, u, -1) != v.f(p, minp, topp, u) {
							fired = true
						}
					}
				}
			}
		}
		if !fired {
			t.Errorf("violation %q is invisible to the draw gate", v.name)
		}
	}

	// And the penalty's.
	bp, err := tab.SamplePenalty()
	if err != nil {
		t.Fatal(err)
	}
	cp, err := MapNamed(bp, "sample_penalty_viol")
	if err != nil {
		t.Fatal(err)
	}
	defer cp.Close()
	pv := []float32{2, -3, 0, 4, -8}
	off := []int64{0, 4, 8, 12, 16}
	got := penaltyCall(cp, pv, off, 2)
	for _, v := range []struct {
		name string
		f    func() []float32
	}{
		{"the sign rule is flipped", func() []float32 {
			out := append([]float32(nil), pv...)
			for i := range out {
				if out[i] > 0 {
					out[i] *= 2
				} else {
					out[i] /= 2
				}
			}
			return out
		}},
		{"zero is divided rather than multiplied", func() []float32 {
			out := penaltyGo(pv, off, 2)
			out[2] = 0 / float32(2)
			out[2] = float32(math.Float32frombits(0x80000000)) // a -0.0 where +0.0 is right
			return out
		}},
	} {
		want := v.f()
		same := true
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				same = false
			}
		}
		if same {
			t.Errorf("violation %q is invisible to the penalty gate", v.name)
		}
	}
}
