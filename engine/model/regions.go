package model

import (
	"github.com/jitllm/jitllm/engine/nn"
)

// regions is the per-instance form of the pool regions a decode token runs every
// layer, with the function each hands the pool built once -- engine/nn's
// hotRegions, for the graph's own regions. A closure literal over a call's
// locals escapes into the pool's job and is a heap object per region, which is
// how a warm token came to allocate per region
// (TestDecodeDoesNotAllocate). An instance runs one region at a time, so
// one set of fields serves every call; each is cleared of its slices
// afterwards so the instance pins no caller's buffer.
type regions struct {
	elem elemRegion
	rows rowRegion
}

// elemOp is which elementwise op an elemRegion runs.
type elemOp uint8

const (
	elemActMul elemOp = iota // a = act(a) * b: the gated FFN's activation
	elemAct                  // a = act(a): an ungated FFN's
	elemAxpy                 // a += alpha*b: the residual add
	elemReduce               // a += sum_j w[j]*moeDown[j]: a mixture's routed sum
	elemXIELU                // a = xielu(a; b): Apertus's ungated FFN, b its four numbers
)

// elemRegion is an elementwise op split by vector across the pool. Every op
// but the residual add carries the ragged remainder with the last chunk, as
// the closures it replaces did.
type elemRegion struct {
	op    elemOp
	a, b  []float32 // the operands; b is the routed weights for elemReduce
	act   nn.ActKind
	alpha float32
	n, v  int // the logical length (elemReduce's row width) and its whole vectors
	fn    func(lo, hi int)
}

// elemRun dispatches the elementwise region described by s.rg.elem over its v
// vectors, and clears its slices afterwards.
func (s *Instance) elemRun() {
	j := &s.rg.elem
	if j.fn == nil {
		j.fn = s.elemChunk
	}
	s.jit.Parallel(j.v, 64, j.fn)
	j.a, j.b = nil, nil
}

// elemChunk is one chunk of the elementwise region.
func (s *Instance) elemChunk(lo, hi int) {
	j := &s.rg.elem
	lanes := nn.ElemLanes
	a, b := lo*lanes, hi*lanes
	if hi == j.v && j.op != elemAxpy {
		b = j.n // the ragged remainder rides with the last chunk
	}
	switch j.op {
	case elemActMul:
		s.actmul(j.a[a:b], j.b[a:b], j.act)
	case elemAct:
		nn.Act32JIT(j.a[a:b], j.act)
	case elemXIELU:
		nn.XIELU32JIT(j.a[a:b], j.b)
	case elemAxpy:
		s.axpy(j.a[a:b], j.b[a:b], j.alpha)
	case elemReduce:
		n := j.n
		for e, w := range j.b {
			nn.Axpy32JIT(j.a[a:b], s.moeDown[e*n+a:e*n+b], w)
		}
	}
}

// rowRegion is a region over rows whose arguments the caller fills before the
// dispatch (rowRun), without a closure per call: dst row += alpha * b, or the
// table row idx[r] of b.
type rowRegion struct {
	dst, b []float32
	idx    []int32
	dim    int
	alpha  float32
	fn     func(lo, hi int)
}

// rowRun dispatches the row region described by s.rg.rows over n items in
// chunks of chunk, and clears its slices afterwards.
func (s *Instance) rowRun(n, chunk int) {
	j := &s.rg.rows
	if j.fn == nil {
		j.fn = s.rowItems
	}
	s.jit.Parallel(n, chunk, j.fn)
	j.dst, j.b, j.idx = nil, nil, nil
}

// rowItems is one chunk of the row region.
func (s *Instance) rowItems(lo, hi int) {
	j := &s.rg.rows
	d := j.dim
	for i := lo; i < hi; i++ {
		b := j.b
		if j.idx != nil {
			r := int(j.idx[i])
			b = j.b[r*d : (r+1)*d]
		}
		nn.Axpy32JIT(j.dst[i*d:(i+1)*d], b, j.alpha)
	}
}

// elemChunk is the chunk a pool region of items elements runs in: four
// chunks per worker, so a chunk's execution dominates the claim, callback and
// kernel entry.
func elemChunk(items, workers int) int {
	if workers < 1 {
		workers = 1
	}
	if c := items / (4 * workers); c > 1 {
		return c
	}
	return 1
}

// rowChunk is elemChunk for a loop whose unit is a row of width elements:
// 1024 rows at chunk 1 is 1024 atomic claims around one kernel call each.
func (s *Instance) rowChunk(rows, width int) int {
	if !s.worthPooling(rows * width) {
		return rows // one chunk: the pool runs it on the caller
	}
	return elemChunk(rows, s.jit.Workers())
}

// worthPooling is the crossover: parallelise only when the serial work
// exceeds the dispatch cost by enough to pay for it, Tserial > D/(1 - 1/S).
// With a region dispatch of ~3 us and a speedup of ~3 that is ~4.5 us of
// serial work, expressed in elements at a conservative streaming rate.
func (s *Instance) worthPooling(elems int) bool {
	if !s.m.opt.elemPool {
		return false
	}
	const elemsPerUs = 2000 // conservative f32 streaming rate, one core
	const crossoverUs = 10  // ~3x the region dispatch, with margin
	return elems >= crossoverUs*elemsPerUs
}
