//go:build amd64 || arm64

package nn

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The sampler's generated code. Like the greedy argmax and the mixture
// router's top-k these are package-level kernels rather than a JIT's: sampling
// is not a layer and takes only a logits row.
//
// There is no Go fallback. Every tier (AVX2, SSE, NEON) emits all three
// kernels, and a host below the SSE floor is refused by model.Open.

// sampleNone is the id a segment reports when it has nothing eligible left. It
// is INT_MAX, which no vocabulary produces and which loses the lowest-id
// tie-break to every real id.
const sampleNone = int32(math.MaxInt32)

type sampleKey struct {
	t             cpu.Tier
	op            int
	first, idsMem bool
}

const (
	opSegMax = iota
	opDraw
	opPenalty
)

// The kernels, keyed by tier and by the two baked flags. A process asks for at
// most five keys per tier for its whole life, so the map never grows.
var sampleKernels struct {
	mu   sync.Mutex
	snap atomic.Pointer[map[sampleKey]*cpu.Code]
}

var sampleConsts = cpu.MoETopKConsts()

func init() {
	m := map[sampleKey]*cpu.Code{}
	sampleKernels.snap.Store(&m)
}

func sampleFor(op int, first, idsMem bool) *cpu.Code {
	key := sampleKey{cpu.HostTier(), op, first, idsMem}
	if c, ok := (*sampleKernels.snap.Load())[key]; ok {
		return c
	}
	return sampleFill(key)
}

// sampleFill emits and maps one kernel under the lock and publishes a new
// snapshot. A failure is cached as a nil entry rather than retried: an emitter
// that refuses on this tier refuses every token.
func sampleFill(key sampleKey) *cpu.Code {
	sampleKernels.mu.Lock()
	defer sampleKernels.mu.Unlock()
	old := *sampleKernels.snap.Load()
	if c, ok := old[key]; ok {
		return c
	}
	em := cpu.EmittersFor(key.t)
	var (
		b    []byte
		err  error
		name string
	)
	switch key.op {
	case opSegMax:
		b, err = em.SampleSegMax(key.first, key.idsMem)
		name = "sample_segmax"
	case opDraw:
		b, err = em.SampleDraw()
		name = "sample_draw"
	default:
		b, err = em.SamplePenalty()
		name = "sample_penalty"
	}
	var c *cpu.Code
	if err == nil {
		if mapped, mErr := cpu.MapNamed(b, name); mErr == nil {
			c = mapped
		}
	}
	next := make(map[sampleKey]*cpu.Code, len(old)+1)
	for kk, vv := range old {
		next[kk] = vv
	}
	next[key] = c
	sampleKernels.snap.Store(&next)
	return c
}

func sampleMust(op int, first, idsMem bool) *cpu.Code {
	c := sampleFor(op, first, idsMem)
	if c == nil {
		panic(fmt.Sprintf("jit: no sampler kernel (op %d) for tier %v -- sampling is "+
			"generated code and has no Go path", op, cpu.HostTier()))
	}
	return c
}

// SampleOrder walks a logits row in strict descending order -- higher value
// first, lower id on a tie -- one element at a time, and is reused across
// tokens so a decode allocates nothing here.
//
// It is a segmented selection: Begin records the best of each segment in one
// pass; Next sweeps that summary for the overall best and rescans only the
// winner's segment, since every other segment's best is unchanged. One
// extraction costs cpu.SampleSegLen(n) plus the segment count comparisons.
// Taking every element out costs n x (L + n/L), so callers stop early
// (engine/model/sample.go's growth loop).
type SampleOrder struct {
	vals     []float32
	segLen   int
	segShift uint
	segs     int
	segVals  []float32
	segIds   []int32
	prev     [2]float32 // {prevVal, bitcast(prevIdx)}
	oneVal   [1]float32
	oneId    [1]int32
}

// Reserve sizes the segment summary for any row of up to n elements, so
// Begin on such a row allocates nothing.
func (o *SampleOrder) Reserve(n int) {
	if cap(o.segVals) < n {
		o.segVals = make([]float32, 0, n)
		o.segIds = make([]int32, 0, n)
	}
}

// Begin prepares an ordering of vals. It runs one pass over the whole row.
func (o *SampleOrder) Begin(vals []float32) {
	n := len(vals)
	if n == 0 {
		panic("jit: SampleOrder.Begin on an empty logits row")
	}
	o.vals = vals
	if o.segLen == 0 || o.segs*o.segLen < n || cpu.SampleSegLen(n) != o.segLen {
		o.segLen = cpu.SampleSegLen(n)
		for o.segShift = 0; 1<<o.segShift != uint(o.segLen); o.segShift++ {
		}
		o.segs = (n + o.segLen - 1) / o.segLen
		// A row whose length moves (the lightning indexer's, one position
		// longer each token) re-segments without allocating once Reserve has
		// sized the summary.
		if cap(o.segVals) < o.segs {
			o.segVals = make([]float32, o.segs)
			o.segIds = make([]int32, o.segs)
		}
		o.segVals, o.segIds = o.segVals[:o.segs], o.segIds[:o.segs]
	}
	o.segs = (n + o.segLen - 1) / o.segLen

	c := sampleMust(opSegMax, true, false)
	full := n / o.segLen
	if full > 0 {
		args := cpu.Args{
			Q32:      &vals[0],
			Out:      &o.segVals[0],
			AHalfSum: &o.segIds[0],
			K:        int64(o.segLen),
			Rows:     int64(full),
			Cols:     0,
			Scr:      (*byte)(unsafe.Pointer(&sampleConsts[0])),
		}
		c.Call(&args)
	}
	// The ragged last segment is its own call: the kernel takes one segment
	// length, and the full length would read past the row.
	if rem := n - full*o.segLen; rem > 0 {
		args := cpu.Args{
			Q32:      &vals[full*o.segLen],
			Out:      &o.segVals[full],
			AHalfSum: &o.segIds[full],
			K:        int64(rem),
			Rows:     1,
			Cols:     int64(full * o.segLen),
			Scr:      (*byte)(unsafe.Pointer(&sampleConsts[0])),
		}
		c.Call(&args)
	}
}

// Next returns the next element in descending order, and false once every
// element has been taken.
func (o *SampleOrder) Next() (float32, int32, bool) {
	sweep := sampleMust(opSegMax, true, true)
	args := cpu.Args{
		Q32:      &o.segVals[0],
		ASum:     &o.segIds[0],
		Out:      &o.oneVal[0],
		AHalfSum: &o.oneId[0],
		K:        int64(o.segs),
		Rows:     1,
		Scr:      (*byte)(unsafe.Pointer(&sampleConsts[0])),
	}
	sweep.Call(&args)
	v, id := o.oneVal[0], o.oneId[0]
	if id == sampleNone {
		return 0, 0, false
	}

	// Only the winner's own segment can have changed, so only it is rescanned.
	o.prev[0] = v
	o.prev[1] = math.Float32frombits(uint32(id))
	s := int(id) >> o.segShift
	base := s << o.segShift
	span := o.segLen
	if base+span > len(o.vals) {
		span = len(o.vals) - base
	}
	rescan := sampleMust(opSegMax, false, false)
	re := cpu.Args{
		Q32:      &o.vals[base],
		Out:      &o.segVals[s],
		AHalfSum: &o.segIds[s],
		K:        int64(span),
		Rows:     1,
		Cols:     int64(base),
		AScale:   &o.prev[0],
		Scr:      (*byte)(unsafe.Pointer(&sampleConsts[0])),
	}
	rescan.Call(&re)
	return v, id, true
}

// SampleDrawResult is what one call of the draw kernel reports.
type SampleDrawResult struct {
	// Res is the index into the candidate array that the inverse-CDF walk
	// landed on.
	Res int
	// M is how many candidates survived min-p and top-p.
	M int
	// Cut is whether either filter fired. When it did not, M is the whole
	// array and the caller may need more candidates than it supplied.
	Cut bool
	// Resolved is whether the walk terminated inside the array rather than
	// running off its end onto the last survivor.
	Resolved bool
	// Sum is the surviving mass, the divisor the walk used.
	Sum float32
}

// SampleDraw is the parameter and counter block one caller reuses, so a decode
// token allocates nothing here. The kernel writes through it, so it must not be
// shared between concurrent draws.
type SampleDraw struct {
	par [cpu.SampleParN]float32
	out [cpu.SampleOutN]int32
	sum [1]float32
}

// Run applies min-p and top-p to p -- the candidates' probabilities,
// descending -- and draws from what survives with the uniform u.
//
// minp of 0 disables min-p and topp of +Inf disables top-p, by arithmetic
// rather than by a branch. sumAll is the whole distribution's mass when p is
// only a prefix of it and a negative value when p is the whole of it.
func (d *SampleDraw) Run(p []float32, minp, topp, u, sumAll float32) SampleDrawResult {
	if len(p) == 0 {
		panic("jit: a sampler draw over no candidates")
	}
	par, out, sum := &d.par, &d.out, &d.sum
	par[cpu.SampleParMinP] = minp
	par[cpu.SampleParTopP] = topp
	par[cpu.SampleParU] = u
	par[cpu.SampleParSum] = sumAll
	args := cpu.Args{
		Q32:    &p[0],
		K:      int64(len(p)),
		AScale: &par[0],
		ASum:   &out[0],
		Out:    &sum[0],
		Scr:    (*byte)(unsafe.Pointer(&sampleConsts[0])),
	}
	sampleMust(opDraw, false, false).Call(&args)
	return SampleDrawResult{
		Res:      int(out[cpu.SampleOutRes]),
		M:        int(out[cpu.SampleOutM]),
		Cut:      out[cpu.SampleOutCut] != 0,
		Resolved: out[cpu.SampleOutResolved] != 0,
		Sum:      sum[0],
	}
}

// Mass is the total of p, measured by the same kernel with both cuts disabled
// and its walk skipped (it runs over the whole vocabulary, where the walk would
// be a second full pass with a meaningless answer). It is the divisor a caller
// needs when its candidates are only a prefix of the distribution.
func (d *SampleDraw) Mass(p []float32) float32 {
	if len(p) == 0 {
		return 0
	}
	par, out, sum := &d.par, &d.out, &d.sum
	par[cpu.SampleParMinP] = 0
	par[cpu.SampleParTopP] = float32(math.Inf(1))
	par[cpu.SampleParU] = 0
	par[cpu.SampleParSum] = -1
	args := cpu.Args{
		Q32:    &p[0],
		K:      int64(len(p)),
		RowStr: 1, // the mass alone: no walk
		AScale: &par[0],
		ASum:   &out[0],
		Out:    &sum[0],
		Scr:    (*byte)(unsafe.Pointer(&sampleConsts[0])),
	}
	sampleMust(opDraw, false, false).Call(&args)
	return sum[0]
}

// SamplePenalty32JIT applies llama.cpp's repeat penalty in place: a positive
// value is divided by pen and everything else multiplied by it.
//
// off is the history as byte offsets into vals, not token ids: addressing with
// an id needs a sign-extending 32-bit load into a general register that none of
// the three assemblers has, and the caller scales once per observed token.
func SamplePenalty32JIT(vals []float32, off []int64, pen float32) {
	if len(off) == 0 || len(vals) == 0 {
		return
	}
	args := cpu.Args{
		Q32:    &vals[0],
		W:      (*byte)(unsafe.Pointer(&off[0])),
		K:      int64(len(off)),
		AScale: &pen,
	}
	sampleMust(opPenalty, false, false).Call(&args)
}
