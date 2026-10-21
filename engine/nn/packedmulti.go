//go:build amd64 || arm64

package nn

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// MatVecPackedMulti computes several packed matvecs that share one activation
// (a mixture's experts' gate and up), in one pool region instead of one region
// each. The arithmetic is the same fused kernel on the same bytes; the pool
// splits one group space spanning every matrix, so a worker that finishes one
// expert's rows walks into the next instead of waiting at a barrier.
//
// Every matrix must share the type, shape and the fused kernel's row-group
// alignment -- a mixture's bank does by construction. Anything else returns
// false and the caller keeps its loop, which is what makes this additive.
func (f *JIT) MatVecPackedMulti(outs [][]float32, t quant.Type, ps []*Packed, x []float32, nrows, k int) bool {
	m := len(ps)
	if f == nil || m == 0 || len(outs) != m || len(x) < k || nrows <= 0 {
		return false
	}
	x = x[:k]
	// One matrix is not a batch; MatVecPacked serves it.
	if m == 1 {
		return false
	}
	fused := f.packedFused[t]
	step := cpu.PackedOuterElems(t)
	if !f.MixtureBatches(t) || step == 0 || k%step != 0 || k > len(f.q) ||
		nrows%grpOf(t) != 0 {
		return false
	}
	nb := k / cpu.Q8Block
	if 2*nb > len(f.pairs) {
		return false
	}
	for i, p := range ps {
		if p == nil || len(p.QS) == 0 || len(outs[i]) < nrows {
			return false
		}
	}
	f.prepAct(t, x, k, nb)

	groups := nrows / grpOf(t)
	total := m * groups
	per := f.fusedChunk(total, t, k)
	rowsPer := per * grpOf(t)
	f.growPackedScratch(f.pool.Max() * rowsPer)
	q, _ := kernels.QuantOf(t)
	j := &f.hot.multi
	j.fc = f.newFusedCall(fused, t, q, k, min(rowsPer, nrows))
	j.ps, j.outs, j.groups, j.nrows, j.rowsPer = ps, outs, groups, nrows, rowsPer
	for _, o := range outs {
		clear(o[:nrows])
	}
	if first, ok := f.nodeOrder(ps); ok {
		// Each matrix where its page lives: the order alternates the nodes,
		// so block b of one matrix's groups is on node (first+b).
		j.ps, j.outs = f.permuted(ps, outs)
		f.pool.DoNodes(regionLabel(t, "/multi"), total, per, groups, first, f.hot.multiFn(f))
		j.ps, j.outs = nil, nil
		f.mixAffine++
		return true
	}
	f.pool.DoLabeled(regionLabel(t, "/multi"), total, per, f.hot.multiFn(f))
	j.ps, j.outs = nil, nil
	return true
}

// MatVecPackedGather computes m packed matvecs where each matrix has its own
// activation (a mixture's down projections), in two pool regions: one that
// quantizes every activation and one that does all the arithmetic. Activation
// j is x[j*xstride : j*xstride+k].
//
// It shares MatMulPacked's activation scratch (growPackedMM) and leaves f.q and
// f.pairs alone, so MatVecPacked's cached activation survives the call.
func (f *JIT) MatVecPackedGather(outs [][]float32, t quant.Type, ps []*Packed, x []float32, xstride, nrows, k int) bool {
	return f.matVecPackedGather(outs, t, ps, x, nil, 0, xstride, nrows, k)
}

// MatVecPackedGatherAct is MatVecPackedGather over act(x)*up, each activation
// j's gated product taken in place in x (up at the same stride) inside the
// region that quantizes it: a mixture's down projections after their SwiGLU,
// one pool region fewer than the activation and the gather apart, with the
// same generated activation kernel and so the same bits.
func (f *JIT) MatVecPackedGatherAct(outs [][]float32, t quant.Type, ps []*Packed, x, up []float32, act ActKind, xstride, nrows, k int) bool {
	if gatedIndex(act) < 0 || len(up) < len(ps)*xstride-xstride+k {
		return false
	}
	return f.matVecPackedGather(outs, t, ps, x, up, act, xstride, nrows, k)
}

func (f *JIT) matVecPackedGather(outs [][]float32, t quant.Type, ps []*Packed, x, up []float32, act ActKind, xstride, nrows, k int) bool {
	m := len(ps)
	if f == nil || m < 2 || len(outs) != m || nrows <= 0 || k <= 0 ||
		xstride < k || len(x) < (m-1)*xstride+k {
		return false
	}
	fused := f.packedFused[t]
	step := cpu.PackedOuterElems(t)
	if !f.MixtureBatches(t) || step == 0 || k%step != 0 || nrows%grpOf(t) != 0 {
		return false
	}
	nb := k / cpu.Q8Block
	if nb == 0 {
		return false
	}
	for i, p := range ps {
		if p == nil || len(p.QS) == 0 || len(outs[i]) < nrows {
			return false
		}
	}
	half := 0
	if cpu.NeedsHalfSums(t) {
		half = k / 16
	}
	f.growPackedMM(m, k, 2*nb, half)

	// Each matrix where its page lives (nodeOrder): the activations are
	// quantized into the permuted order's slots, so slot i is matrix ord[i]'s.
	first, affine := f.nodeOrder(ps)
	gperm := []int(nil)
	if affine {
		gperm = f.mord
	}
	tq := f.quantStart()
	// One region for every activation's quantize, where there were m.
	gq := &f.hot.gatherQuant
	gq.perm = gperm
	gq.qk, gq.t, gq.x, gq.up, gq.act = f.quantKernel(t), t, x, up, act
	gq.xstride, gq.k, gq.nb, gq.half, gq.win = xstride, k, nb, half, f.actWindow
	f.pool.Do(m, max(1, m/(4*f.pool.N())), f.hot.gatherQuantFn(f))
	gq.x, gq.up = nil, nil
	f.quantSince(tq)

	groups := nrows / grpOf(t)
	total := m * groups
	per := f.fusedChunk(total, t, k)
	rowsPer := per * grpOf(t)
	f.growPackedScratch(f.pool.Max() * rowsPer)
	q, _ := kernels.QuantOf(t)
	j := &f.hot.gather
	j.fc = f.newFusedCall(fused, t, q, k, min(rowsPer, nrows))
	j.ps, j.outs, j.groups, j.nrows, j.rowsPer = ps, outs, groups, nrows, rowsPer
	j.half, j.nb, j.k = half, nb, k
	// The kernel accumulates into Out, so the caller owns the zero.
	for _, o := range outs {
		clear(o[:nrows])
	}
	if affine {
		j.ps, j.outs = f.permuted(ps, outs)
		f.pool.DoNodes(regionLabel(t, "/gather"), total, per, groups, first, f.hot.gatherFn(f))
		f.mixAffine++
	} else {
		f.pool.DoLabeled(regionLabel(t, "/gather"), total, per, f.hot.gatherFn(f))
	}
	j.ps, j.outs = nil, nil
	return true
}

// MixtureBatches reports whether MatVecPackedMulti and MatVecPackedGather
// serve t here, rather than declining to the caller's per-expert loop.
//
// Not on arm64 by default: there the fused kernel loses to the tiled one at a
// mixture's shapes, so the per-expert loop goes through
// MatVecPacked's per-shape choice instead. See
// docs/engineering-history/cpu-kernels.md.
func (f *JIT) MixtureBatches(t quant.Type) bool {
	if f == nil || f.packedFused[t] == nil || f.cfg.MixtureBatch < 0 {
		return false
	}
	return f.cfg.MixtureBatch > 0 || f.tier != cpu.TierNEON
}

// grpOf is the fused kernel's row group for t. It varies by format (Q5_K's is
// eight), and Args.Rows counts groups, so a hardcoded group reads past the
// payload.
func grpOf(t quant.Type) int { return cpu.PackedFusedGroupOf(t) }

// multiRegion is MatVecPackedMulti's or MatVecPackedGather's arithmetic region,
// kept on the JIT with its arguments in fields (see hotRegions): a mixture runs
// them every layer of every token, and a closure over the call's locals was two
// heap objects each time.
type multiRegion struct {
	fc                     fusedCall
	ps                     []*Packed
	outs                   [][]float32
	groups, nrows, rowsPer int
	half, nb, k            int // gather only: each matrix's own activation
	fn                     func(worker, lo, hi int)
}

func (h *hotRegions) multiFn(f *JIT) func(worker, lo, hi int) {
	if h.multi.fn == nil {
		h.multi.fn = func(worker, lo, hi int) {
			j := &h.multi
			// A worker's range may span matrices, so the call is per matrix and
			// the loop walks the boundaries.
			for lo < hi {
				mi := lo / j.groups
				g0 := lo - mi*j.groups
				g1 := min(hi-mi*j.groups, j.groups)
				p, out := j.ps[mi], j.outs[mi]
				// The stride is the buffer's row count, not this weight's, so
				// both come from the Packed being read, never from the batch.
				stride, row := p.Stride, p.Row
				if stride == 0 {
					stride = j.nrows
				}
				j.fc.run(p, out, row, stride, g0, g1, f.q, f.pairs, f.half, worker*j.rowsPer)
				lo = mi*j.groups + g1
			}
		}
	}
	return h.multi.fn
}

func (h *hotRegions) gatherFn(f *JIT) func(worker, lo, hi int) {
	if h.gather.fn == nil {
		h.gather.fn = func(worker, lo, hi int) {
			j := &h.gather
			// A worker's range may span matrices, which is the point.
			for lo < hi {
				mi := lo / j.groups
				g0 := lo - mi*j.groups
				g1 := min(hi-mi*j.groups, j.groups)
				p, out := j.ps[mi], j.outs[mi]
				stride, row := p.Stride, p.Row
				if stride == 0 {
					stride = j.nrows
				}
				var hs []float32
				if j.half > 0 {
					hs = f.mhalf[mi*j.half : (mi+1)*j.half]
				}
				k, nb := j.k, j.nb
				j.fc.run(p, out, row, stride, g0, g1, f.mq[mi*k:(mi+1)*k], f.mpairs[mi*2*nb:(mi+1)*2*nb], hs, worker*j.rowsPer)
				lo = mi*j.groups + g1
			}
		}
	}
	return h.gather.fn
}

func (h *hotRegions) gatherQuantFn(f *JIT) func(worker, lo, hi int) {
	if h.gatherQuant.fn == nil {
		h.gatherQuant.fn = func(wk, lo, hi int) {
			j := &h.gatherQuant
			k, nb := j.k, j.nb
			for i := lo; i < hi; i++ {
				var hs []float32
				if j.half > 0 {
					hs = f.mhalf[i*j.half : (i+1)*j.half]
				}
				src := i
				if j.perm != nil {
					src = j.perm[i]
				}
				xi := j.x[src*j.xstride : src*j.xstride+k]
				if j.up != nil {
					// The gated activation in place, in the quantize's own
					// region (MatVecPackedGatherAct).
					ActMul32JIT(xi, j.up[src*j.xstride:src*j.xstride+k], j.act)
				}
				j.qk.Run(j.t, f.mq[i*k:(i+1)*k], f.mpairs[i*2*nb:(i+1)*2*nb], hs, f.quantScratch(wk),
					xi, k, 0, nb, j.win)
			}
		}
	}
	return h.gatherQuant.fn
}

// nodeOrder fills f.mord with an order of ps that alternates the NUMA nodes
// their pages are on, round robin from the node holding the most, and
// returns the first matrix's node index. It reports false on a pool of one
// node, or where the pages sit on one node: there is nothing to place.
// Matrices whose node the kernel will not say go last.
func (f *JIT) nodeOrder(ps []*Packed) (first int, ok bool) {
	n := f.pool.Nodes()
	if n < 2 || len(ps) < 2 {
		return 0, false
	}
	if cap(f.mbyNode) < n+1 {
		f.mbyNode = make([][]int, n+1)
	}
	by := f.mbyNode[:n+1]
	for i := range by {
		by[i] = by[i][:0]
	}
	for i, p := range ps {
		ni := f.pool.NodeIndex(sched.NodeOf(unsafe.Pointer(&p.QS[0])))
		if ni < 0 {
			ni = n
		}
		by[ni] = append(by[ni], i)
	}
	nodesUsed, big := 0, 0
	for i := 0; i < n; i++ {
		if len(by[i]) > 0 {
			nodesUsed++
		}
		if len(by[i]) > len(by[big]) {
			big = i
		}
	}
	if nodesUsed < 2 {
		return 0, false
	}
	f.mord = f.mord[:0]
	if cap(f.mcur) < n {
		f.mcur = make([]int, n)
	}
	cur := f.mcur[:n]
	clear(cur)
	for left := len(ps) - len(by[n]); left > 0; {
		for s := 0; s < n; s++ {
			ni := (big + s) % n
			if cur[ni] < len(by[ni]) {
				f.mord = append(f.mord, by[ni][cur[ni]])
				cur[ni]++
				left--
			}
		}
	}
	f.mord = append(f.mord, by[n]...)
	return big, true
}

// permuted is ps and outs in f.mord's order, in the JIT's own slices.
func (f *JIT) permuted(ps []*Packed, outs [][]float32) ([]*Packed, [][]float32) {
	f.mps, f.mouts = f.mps[:0], f.mouts[:0]
	for _, i := range f.mord {
		f.mps = append(f.mps, ps[i])
		f.mouts = append(f.mouts, outs[i])
	}
	return f.mps, f.mouts
}

// MixtureAffine is how many mixture regions ran node-affine (nodeOrder): the
// check that the NUMA path ran.
func (f *JIT) MixtureAffine() int64 { return f.mixAffine }
