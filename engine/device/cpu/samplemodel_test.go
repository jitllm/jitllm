//go:build amd64 || arm64

package cpu

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// The sampler kernels' oracle and the calls into them, shared by
// sample_test.go (x86) and sample_a64_test.go (NEON) so all three tiers
// are held to one model of the contract.

// ---------------------------------------------------------------- the model

const sampleEmpty = int32(math.MaxInt32)

// segMaxGo is EmitSampleSegMax's contract written out: per segment, the largest
// eligible element, with the lowest id winning a tie, and (-Inf, INT_MAX) for a
// segment with nothing eligible left.
func segMaxGo(vals []float32, ids []int32, segLen, segs int, first bool, pv float32, pi int32) ([]float32, []int32) {
	outV := make([]float32, segs)
	outI := make([]int32, segs)
	for s := 0; s < segs; s++ {
		best, bestID := float32(math.Inf(-1)), sampleEmpty
		for j := s * segLen; j < (s+1)*segLen && j < len(vals); j++ {
			v, id := vals[j], ids[j]
			ok := false
			if first {
				ok = v == v // ordered: every element but a NaN
			} else {
				ok = v < pv || (v == pv && id > pi)
			}
			if !ok {
				continue
			}
			if bestID == sampleEmpty || v > best {
				best, bestID = v, id
			}
		}
		outV[s], outI[s] = best, bestID
	}
	return outV, outI
}

type drawGoOut struct {
	res, m        int
	cut, resolved bool
	sum           float32
}

// drawGo is the Go loops engine/model/sample.go used to run, written as one function:
// the min-p break, the top-p cumulative and its cut, the surviving mass, and
// the inverse-CDF walk with its fallthrough onto the last survivor.
func drawGo(p []float32, minp, topp, u, sumAll float32) drawGoOut {
	var o drawGoOut
	cut := minp * p[0]
	acc := float32(0)
	for i, v := range p {
		if !o.cut && v < cut {
			o.sum, o.m, o.cut = acc, i, true
		}
		acc += v
		if !o.cut && acc >= topp {
			o.sum, o.m, o.cut = acc, i+1, true
		}
	}
	if !o.cut {
		o.m = len(p)
		o.sum = acc
		if sumAll >= 0 {
			o.sum = sumAll
		}
	}
	r := u * o.sum
	o.res = o.m - 1
	for i := 0; i < o.m; i++ {
		r -= p[i]
		if r <= 0 {
			o.res, o.resolved = i, true
			break
		}
	}
	return o
}

// penaltyGo is llama.cpp's repeat-penalty rule: a positive value divided, and
// everything else -- including both zeros -- multiplied.
func penaltyGo(vals []float32, off []int64, pen float32) []float32 {
	out := append([]float32(nil), vals...)
	for _, b := range off {
		i := b / 4
		if out[i] > 0 {
			out[i] /= pen
		} else {
			out[i] *= pen
		}
	}
	return out
}

// ---------------------------------------------------------------- the calls

func segMaxCall(c *Code, vals []float32, ids []int32, segLen, segs, base int,
	first, idsMem bool, pv float32, pi int32) ([]float32, []int32) {
	outV := make([]float32, segs)
	outI := make([]int32, segs)
	prev := [2]float32{pv, math.Float32frombits(uint32(pi))}
	args := Args{
		Q32:      &vals[0],
		Out:      &outV[0],
		AHalfSum: &outI[0],
		K:        int64(segLen),
		Rows:     int64(segs),
		Cols:     int64(base),
		Scr:      (*byte)(unsafe.Pointer(&sampleTestConsts[0])),
	}
	if idsMem {
		args.ASum = &ids[0]
	}
	if !first {
		args.AScale = &prev[0]
	}
	c.Call(&args)
	return outV, outI
}

var sampleTestConsts = MoETopKConsts()

func drawCall(c *Code, p []float32, minp, topp, u, sumAll float32) drawGoOut {
	par := [SampleParN]float32{minp, topp, u, sumAll}
	var out [SampleOutN]int32
	var sum [1]float32
	args := Args{
		Q32:    &p[0],
		K:      int64(len(p)),
		AScale: &par[0],
		ASum:   &out[0],
		Out:    &sum[0],
		Scr:    (*byte)(unsafe.Pointer(&sampleTestConsts[0])),
	}
	c.Call(&args)
	return drawGoOut{
		res:      int(out[SampleOutRes]),
		m:        int(out[SampleOutM]),
		cut:      out[SampleOutCut] != 0,
		resolved: out[SampleOutResolved] != 0,
		sum:      sum[0],
	}
}

// segMaxRaw is the segment maximum with every argument spelled out, for the
// walk gate: it drives the kernel exactly as nn.SampleOrder does, writing into
// slices the caller already owns rather than allocating a result.
func segMaxRaw(c *Code, vals []float32, ids []int32, segLen, segs, base int, prev *[2]float32,
	outV []float32, outI []int32) {
	args := Args{
		Q32: &vals[0], Out: &outV[0], AHalfSum: &outI[0],
		K: int64(segLen), Rows: int64(segs), Cols: int64(base),
		Scr: (*byte)(unsafe.Pointer(&sampleTestConsts[0])),
	}
	if ids != nil {
		args.ASum = &ids[0]
	}
	if prev != nil {
		args.AScale = &prev[0]
	}
	c.Call(&args)
}

// sampleWalk drives the three kernels the way nn.SampleOrder does and returns
// every element of the row in the order they came out. It is the gate's model
// of the ordering as a whole, which no single call can be.
func sampleWalk(t *testing.T, firstC, elig, sweep *Code, row []float32, segLen int) ([]float32, []int32) {
	t.Helper()
	n := len(row)
	segs := (n + segLen - 1) / segLen
	full := n / segLen
	segV := make([]float32, segs)
	segI := make([]int32, segs)
	if full > 0 {
		segMaxRaw(firstC, row, nil, segLen, full, 0, nil, segV, segI)
	}
	if rem := n - full*segLen; rem > 0 {
		segMaxRaw(firstC, row[full*segLen:], nil, rem, 1, full*segLen, nil, segV[full:], segI[full:])
	}
	var gotV []float32
	var gotI []int32
	for step := 0; step < n+2; step++ {
		oneV := make([]float32, 1)
		oneI := make([]int32, 1)
		segMaxRaw(sweep, segV, segI, segs, 1, 0, nil, oneV, oneI)
		if oneI[0] == sampleEmpty {
			break
		}
		gotV = append(gotV, oneV[0])
		gotI = append(gotI, oneI[0])
		prev := [2]float32{oneV[0], math.Float32frombits(uint32(oneI[0]))}
		s := int(oneI[0]) / segLen
		base := s * segLen
		span := segLen
		if base+span > n {
			span = n - base
		}
		segMaxRaw(elig, row[base:], nil, span, 1, base, &prev, segV[s:], segI[s:])
	}
	return gotV, gotI
}

// sampleCheckWalk holds a walk to the strict total order: every finite element
// once, in descending value with the lower id first, and each carrying the
// row's own bits.
func sampleCheckWalk(t *testing.T, arm, name string, segLen int, row []float32, gotV []float32, gotI []int32) {
	t.Helper()
	finite := 0
	for _, v := range row {
		if v == v {
			finite++
		}
	}
	if len(gotI) != finite {
		t.Fatalf("%s %s segLen=%d: the walk produced %d elements, want %d",
			arm, name, segLen, len(gotI), finite)
	}
	seen := map[int32]bool{}
	for i := range gotI {
		if seen[gotI[i]] {
			t.Fatalf("%s %s segLen=%d: id %d came out twice", arm, name, segLen, gotI[i])
		}
		seen[gotI[i]] = true
		if math.Float32bits(gotV[i]) != math.Float32bits(row[gotI[i]]) {
			t.Fatalf("%s %s segLen=%d: id %d came out with %v, the row holds %v",
				arm, name, segLen, gotI[i], gotV[i], row[gotI[i]])
		}
		if i > 0 {
			a, b := gotV[i-1], gotV[i]
			ai, bi := gotI[i-1], gotI[i]
			if !(a > b || (a == b && ai < bi)) {
				t.Fatalf("%s %s segLen=%d: (%v,%d) then (%v,%d) is not the strict total order",
					arm, name, segLen, a, ai, b, bi)
			}
		}
	}
}

func penaltyCall(c *Code, vals []float32, off []int64, pen float32) []float32 {
	out := append([]float32(nil), vals...)
	args := Args{
		Q32:    &out[0],
		W:      (*byte)(unsafe.Pointer(&off[0])),
		K:      int64(len(off)),
		AScale: &pen,
	}
	c.Call(&args)
	return out
}

// ---------------------------------------------------------------- the inputs

// sampleRows are the value rows both ordering arms run: ragged widths around
// the two vector lengths, ties, both zeros, both infinities and a NaN.
func sampleRows() []struct {
	name string
	v    []float32
} {
	mk := func(name string, v ...float32) struct {
		name string
		v    []float32
	} {
		return struct {
			name string
			v    []float32
		}{name, v}
	}
	var rows []struct {
		name string
		v    []float32
	}
	// Ragged widths: 1..20 covers every n%8 and n%4 with room for several
	// whole vectors on both tiers.
	for n := 1; n <= 20; n++ {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*37)%n) - float32(n)/3
		}
		rows = append(rows, mk(fmt.Sprintf("ramp%d", n), v...))
	}
	rows = append(rows,
		mk("allequal", 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
		mk("tiedpairs", 5, 5, 3, 3, 1, 1, -1, -1, -3, -3, -5),
		mk("bothzeros", float32(math.Copysign(0, -1)), 0, 0, float32(math.Copysign(0, -1)), 1, -1, 0),
		mk("descending", 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3),
		mk("ascending", -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
		mk("infinities", float32(math.Inf(1)), 1, float32(math.Inf(-1)), 0,
			float32(math.Inf(1)), -1, float32(math.Inf(-1)), 2, 3),
		mk("onenan", 1, float32(math.NaN()), 3, 2, 5, 4, 7, 6, 9),
		mk("allnan", float32(math.NaN()), float32(math.NaN()), float32(math.NaN())),
		mk("neginfs", float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)),
			float32(math.Inf(-1)), float32(math.Inf(-1))),
	)
	return rows
}
