//go:build amd64 || arm64

package nn

import (
	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// MaxGroup is how many matvecs one MatVecPackedGroup call runs.
const MaxGroup = 3

// MVGroup is a set of packed matvecs that read one activation: a block's q, k
// and v, its gate and up, or a single projection. Out[i] gets Rows[i] rows of
// P[i] (type T[i]) times X[:K].
type MVGroup struct {
	N    int
	Out  [MaxGroup][]float32
	T    [MaxGroup]quant.Type
	P    [MaxGroup]*Packed
	Rows [MaxGroup]int
	X    []float32
	K    int
	// ActUp, when set, makes the activation act(X)*ActUp (Act's gated kernel)
	// rather than X: each worker forms it in its own buffer before quantizing,
	// so the gated FFN's activation needs no region of its own and X is left
	// as it was.
	ActUp []float32
	Act   ActKind
	// Resid, when set, takes matrix 0's output rows into the residual as
	// they are made: Resid[r] += Alpha*Out[0][r] (the generated axpy), by the
	// worker that made them, so the residual add needs no pass of its own.
	Resid []float32
	Alpha float32
}

// MatVecPackedGroup runs every matvec of g as ONE pool region with no
// activation-quantization region in front of it: each worker quantizes X into
// its own buffers the first time it claims a chunk, so no worker waits for
// another between the producer of X and the products. A worker's chunks of
// every matrix follow one another with no wait between matrices either.
//
// Each matrix runs the kernel MatVecPacked would run on it -- the fused
// kernel, at the distance its chooser settled on, over the same row chunks and
// k pieces -- so every output row is the same bits (TestGroupMatchesMatVec).
// It returns false, having run nothing, whenever any matrix would take another
// path there (a chooser still timing, a k-split, NUMA placement, the wide or
// tiled kernel); the caller then runs them one at a time.
func (f *JIT) MatVecPackedGroup(g *MVGroup) bool {
	if f == nil || g.N < 1 || g.N > MaxGroup || len(g.X) < g.K {
		return false
	}
	k := g.K
	j := &f.hot.group
	j.nch = 0
	maxRows := 0
	for i := 0; i < g.N; i++ {
		t, p, nrows := g.T[i], g.P[i], g.Rows[i]
		if p == nil || len(p.QS) == 0 || len(g.Out[i]) < nrows || nrows <= 0 {
			return false
		}
		q, ok := kernels.QuantOf(t)
		if !ok {
			return false
		}
		fused := f.packedFused[t]
		step := cpu.PackedOuterElems(t)
		if fused == nil || step == 0 || k%step != 0 || nrows%grpOf(t) != 0 {
			return false
		}
		stride := p.Stride
		if stride == 0 {
			stride = nrows
		}
		// MatVecPacked's chooser: only a settled distance is the same kernel.
		if pk := f.pickFor(t, nrows, k, stride); pk != nil {
			if !pk.dist || pk.best < 0 {
				return false
			}
			if !pk.applied {
				f.settleDist(t, nrows*k, pk.best)
				pk.applied = true
			}
			if c := f.fusedKernels(pk.best)[t]; c != nil {
				fused = c
			}
		}
		groups := nrows / grpOf(t)
		per := f.fusedChunk(groups, t, k)
		if block, _ := f.numaRows(p, p.Row, stride, grpOf(t)); block > 0 {
			return false
		}
		if f.fusedSlices(t, k, nrows, per) > 1 {
			return false
		}
		rowsPer := per * grpOf(t)
		maxRows = max(maxRows, rowsPer)
		m := &j.m[i]
		m.fc = f.newFusedCall(fused, t, q, k, rowsPer)
		m.p, m.out, m.stride, m.groups, m.per = p, g.Out[i], stride, groups, per
		m.first = j.nch
		m.slot = -1
		for s := 0; s < i; s++ {
			if g.T[s] == t {
				m.slot = j.m[s].slot
				break
			}
		}
		if m.slot < 0 {
			m.slot = j.nslot
			j.slotT[j.nslot] = t
			// Resolved here, on the caller: quantKernel fills a map on
			// first use, which workers must not race on.
			j.slotQK[j.nslot] = f.quantKernel(t)
			j.nslot++
		}
		j.nch += (groups + per - 1) / per
	}
	f.ensureAct(k)
	f.growPackedScratch(f.pool.Max() * maxRows)
	f.growGroupAct(k)
	j.n, j.x, j.k, j.maxRows = g.N, g.X[:k], k, maxRows
	j.actUp, j.act = nil, g.Act
	j.resid, j.alpha = nil, g.Alpha
	if g.Resid != nil {
		if len(g.Resid) < g.Rows[0] {
			return false
		}
		j.resid = g.Resid
	}
	if g.ActUp != nil {
		if len(g.ActUp) < k || gatedIndex(g.Act) < 0 {
			return false
		}
		j.actUp = g.ActUp[:k]
	}
	f.groupGen++
	j.gen = f.groupGen
	for i := 0; i < g.N; i++ {
		clear(g.Out[i][:g.Rows[i]])
	}
	f.pool.DoLabeled(regionLabel(g.T[0], "/group"), j.nch, 1, f.hot.groupFn(f))
	for i := 0; i < g.N; i++ {
		j.m[i].p, j.m[i].out = nil, nil
	}
	j.x, j.actUp, j.resid, j.nslot = nil, nil, nil, 0
	return true
}

// growGroupAct sizes each worker's private quantized activation, one per
// distinct activation format a group can hold.
func (f *JIT) growGroupAct(k int) {
	w := f.pool.Max()
	nb := k/cpu.Q8Block + 1
	if len(f.grq) < w*MaxGroup || len(f.grq[0]) < k || len(f.grpairs[0]) < 2*nb {
		f.grq = make([][]int8, w*MaxGroup)
		f.grpairs = make([][]float32, w*MaxGroup)
		f.grhalfs = make([][]float32, w*MaxGroup)
		f.grseen = make([]uint64, w*MaxGroup)
		f.gract = make([][]float32, w)
		for i := range f.gract {
			f.gract[i] = make([]float32, k)
		}
		for i := range f.grq {
			f.grq[i] = make([]int8, k)
			f.grpairs[i] = make([]float32, 2*nb)
			f.grhalfs[i] = make([]float32, k/16+2)
		}
	}
}

// groupMat is one matrix of a running group.
type groupMat struct {
	fc                  fusedCall
	p                   *Packed
	out                 []float32
	stride, groups, per int
	first, slot         int
}

// groupFn is the group's chunk: chunk c is matrix i's row chunk c-first[i].
func (h *hotRegions) groupFn(f *JIT) func(worker, lo, hi int) {
	if h.group.fn == nil {
		h.group.fn = func(worker, lo, hi int) {
			j := &h.group
			for c := lo; c < hi; c++ {
				i := j.n - 1
				for i > 0 && c < j.m[i].first {
					i--
				}
				m := &j.m[i]
				// This worker's quantized activation, made once per group call.
				b := worker*MaxGroup + m.slot
				t := j.slotT[m.slot]
				var half []float32
				if cpu.NeedsHalfSums(t) {
					half = f.grhalfs[b][:j.k/16]
				}
				if f.grseen[b] != j.gen {
					nb := j.k / cpu.Q8Block
					x := j.x
					if j.actUp != nil {
						x = f.gract[worker][:j.k]
						copy(x, j.x)
						ActMul32JIT(x, j.actUp, j.act)
					}
					j.slotQK[m.slot].Run(t, f.grq[b][:j.k], f.grpairs[b][:2*nb], half,
						f.quantScratch(worker), x, j.k, 0, nb, f.actWindow)
					f.grseen[b] = j.gen
				}
				g0 := (c - m.first) * m.per
				g1 := min(m.groups, g0+m.per)
				m.fc.run(m.p, m.out, m.p.Row, m.stride, g0, g1,
					f.grq[b], f.grpairs[b], half, worker*j.maxRows)
				if i == 0 && j.resid != nil {
					r0, r1 := g0*m.fc.grp, g1*m.fc.grp
					Axpy32JIT(j.resid[r0:r1], m.out[r0:r1], j.alpha)
				}
			}
		}
	}
	return h.group.fn
}
