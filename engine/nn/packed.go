//go:build amd64 || arm64

package nn

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// MatVecPacked computes out[r] = dot(row r, x) from a weight already in the
// device layout -- what a jlm container stores and what every accelerator
// uploads untouched, so host and device share one page format.
//
// The column-major layout suits the host too: one 256-bit load is one payload
// word for eight rows, each 32-bit lane accumulates its own row, and the kernel
// ends with no horizontal reduction.
func (f *JIT) MatVecPacked(out []float32, t quant.Type, p *Packed, x []float32, nrows, k int) bool {
	// x may be longer than k and its first k are the activation (a hybrid's
	// attention output projection reads a model-width scratch).
	if f == nil || p == nil || len(p.QS) == 0 || len(out) < nrows || len(x) < k {
		return false
	}
	x = x[:k]
	q, ok := kernels.QuantOf(t)
	if !ok {
		return false
	}
	// The stride is the buffer's row count, not this weight's. They differ
	// for one expert of a mixture's bank, where rows [Row, Row+nrows) are
	// interleaved through a Stride-row buffer.
	stride, row := p.Stride, p.Row
	if stride == 0 {
		stride = nrows
	}
	// The d plane has its own row and super-block strides, and on a narrow
	// format neither is the payload's (two rows share a word). Both come from
	// cpu so the emitters and this caller cannot drift apart.
	db, dstr := cpu.DRowBytes(t), int64(cpu.DSuperBytes(t, stride))
	code := f.packed[t]
	step := cpu.PackedOuterElems(t)
	// false means this host has no packed kernel for t, and the caller reports
	// it; there is no interpreted tier behind this.
	if code == nil || step == 0 || k%step != 0 {
		return false
	}
	nb := k / cpu.Q8Block
	f.ensureAct(k)
	f.prepAct(t, x, k, nb)

	// The shape's chooser (mvpick.go): which kernel this call runs, and the
	// clock it is timed by while the choice is open.
	var pk *mvPick
	arm, dist := -1, -1
	if f.packedFused[t] != nil && code != nil && nrows%grpOf(t) == 0 {
		if pk = f.pickFor(t, nrows, k, stride); pk != nil {
			if pk.dist {
				dist = pk.arm()
			} else {
				arm = pk.arm()
			}
		}
	}
	if pk != nil && pk.dist && pk.best >= 0 && !pk.applied {
		f.settleDist(t, nrows*k, pk.best)
		pk.applied = true
	}
	t0 := time.Now()

	// The fused kernel is preferred: it runs every word of a sub-block in one
	// pass and keeps the accumulator in a register, where the wide kernel
	// spills it between per-word passes.
	if fused := f.packedFused[t]; fused != nil && nrows%grpOf(t) == 0 && arm != 0 {
		if c := f.fusedKernels(max(dist, 0))[t]; dist >= 0 && c != nil {
			fused = c
		}
		groups := nrows / grpOf(t)
		// The row chunk is the balanced share and the GC budget is paid in k
		// instead (fusedPiece): the kernel reads each payload word as one run
		// of rows*4 bytes, so short row chunks mean short runs. Walking k in
		// pieces is the same sums in the same order, bit-identical.
		per := f.fusedChunk(groups, t, k)
		block, first := f.numaRows(p, row, stride, grpOf(t))
		if block > 0 {
			// Two chunks per worker: one each makes the region wait on the
			// slowest worker, and smaller chunks lose more to per-call cost
			// than they recover (measured on a two-socket host).
			per = max(1, (groups+2*f.pool.N()-1)/(2*f.pool.N()))
		}
		if block == 0 {
			s := f.fusedSlices(t, k, nrows, per)
			if arm > 0 {
				s = arm
			}
			if s > 1 {
				f.fusedSliced(fused, t, q, p, out, row, stride, nrows, k, s)
				pk.observe(time.Since(t0))
				return true
			}
		}
		rowsPer := per * grpOf(t)
		f.growPackedScratch(f.pool.Max() * rowsPer)
		j := &f.hot.fused
		j.fc, j.p, j.out, j.row, j.stride, j.rowsPer = f.newFusedCall(fused, t, q, k, rowsPer), p, out, row, stride, rowsPer
		clear(out[:nrows])
		f.pool.DoNodes(regionLabel(t, "/fused"), groups, per, block, first, f.hot.fusedFn(f))
		j.p, j.out = nil, nil
		pk.observe(time.Since(t0))
		return true
	}

	// Next the wide kernel: accumulators in memory let one call serve the
	// worker's whole row range, so it reads long contiguous runs where the
	// 64-row tile reads 256 bytes and then jumps a row stride.
	if wide := f.packedWide[t]; wide != nil && nrows%cpu.PackedWideGroup == 0 {
		groups := nrows / cpu.PackedWideGroup
		per := packedChunk(f, groups, t, k, cpu.PackedWideGroup)
		if need := f.pool.Max() * per * cpu.PackedWideGroup; len(f.piacc) < need {
			f.piacc = make([]int32, need)
		}
		// The kernel ACCUMULATES into Out, so the caller owns the zero. Doing
		// it in the kernel would cost a pass per sub-block instead of one.
		clear(out[:nrows])
		// scratchRows is how far apart each worker's private accumulators sit;
		// RowStr below is the weight's row stride. Do not name this stride:
		// confusing the two is silent and wrong.
		scratchRows := per * cpu.PackedWideGroup
		f.pool.DoLabeled(regionLabel(t, "/wide"), groups, per, func(worker, lo, hi int) {
			r0 := lo * cpu.PackedWideGroup
			args := cpu.Args{
				Out: &out[r0], W: &p.QS[(row+r0)*4], A: &f.q[0], AScale: &f.pairs[0],
				Rows: int64(hi - lo), RowStr: int64(stride * 4),
				K: int64(k / step), Scr: &f.pkonst[0], AHalf: &f.half[0],
				PD:      &p.D[(row+r0)*db],
				DStr:    dstr,
				Scratch: (*byte)(unsafe.Pointer(&f.piacc[worker*scratchRows])),
			}
			if len(p.SC) > 0 {
				args.PSC = &p.SC[(row+r0)*4]
			}
			wide.Call(&args)
		})
		return true
	}

	// The tiled kernel. RowStr is the tensor's full row stride, not the
	// tile's: every index in the layout is `... * nrows + r`.
	//
	// The caller owns the zero. The amd64 tiled kernel clears Out at each
	// tile head, but EmitA64PackedMatVec nests super-blocks outside tiles
	// (the scale planes advance per super-block), so it load-accumulates and
	// requires a zeroed Out. A gate needs a second call into the same buffer
	// to see a missing clear, since a fresh make() is already zero.
	clear(out[:nrows])
	tiles := nrows / cpu.PackedRows
	if tiles > 0 {
		chunk := packedChunk(f, tiles, t, k, cpu.PackedRows)
		j := &f.hot.tiled
		j.code, j.p, j.out, j.row, j.stride, j.ksteps, j.db, j.dstr = code, p, out, row, stride, k/step, db, dstr
		f.pool.DoLabeled(regionLabel(t, "/packed"), tiles, chunk, f.hot.tiledFn(f))
		j.p, j.out = nil, nil
	}
	// The tail kernel runs the rows PackedRows does not divide.
	r := tiles * cpu.PackedRows
	tc := f.packedTail[t]
	if g := (nrows - r) / cpu.PackedTail; g > 0 {
		args := cpu.Args{
			Out: &out[r], W: &p.QS[(row+r)*4], A: &f.q[0], AScale: &f.pairs[0],
			Rows: int64(g), RowStr: int64(stride * 4),
			K: int64(k / step), Scr: &f.pkonst[0],
			AHalf: &f.half[0], PD: &p.D[(row+r)*db], DStr: dstr,
		}
		if len(p.SC) > 0 {
			args.PSC = &p.SC[(row+r)*4]
		}
		tc.Call(&args)
		r += g * cpu.PackedTail
	}
	// Fewer than PackedTail rows cannot be a tail group in place (its loads are
	// a whole group wide and would run past the tensor), so they are copied
	// into a zero-padded window of one group and the same kernel runs there.
	if n := nrows - r; n > 0 {
		if err := f.packedWindow(out[r:nrows], q, p, t, x, stride, k, row+r, n); err != nil {
			panic(err)
		}
	}
	pk.observe(time.Since(t0))
	return true
}

// packedWindow computes rows [lo, lo+n) by copying them into a tensor of
// exactly PackedTail rows and running THAT through MatVecPacked, so the window
// gets whatever kernel a real tensor of that size gets -- which is what makes
// its rows bit-identical to the same rows computed in place.
func (f *JIT) packedWindow(out []float32, q kernels.Quant, p *Packed, t quant.Type, x []float32, stride, k, lo, n int) error {
	to := cpu.PackedTail
	nq, nd, nsc, err := kernels.PackedWords(q, to, k)
	if err != nil {
		return err
	}
	grow := func(b *[]byte, n int) []byte {
		if len(*b) < n {
			*b = make([]byte, n)
		}
		return (*b)[:n]
	}
	// A window of exactly PackedTail rows takes no window of its own, so the
	// call below cannot reach here again while f.winP is in use.
	w := &f.winP
	*w = Packed{QS: grow(&f.winQS, nq*4), D: grow(&f.winD, nd*4)}
	if nsc > 0 {
		w.SC = grow(&f.winSC, nsc*4)
	}
	if err := kernels.RowWindow(q, p.QS, p.D, p.SC, stride, k, lo, n, to, w.QS, w.D, w.SC); err != nil {
		return err
	}
	res := f.winOut[:]
	if !f.MatVecPacked(res, t, w, x, to, k) {
		return fmt.Errorf("nn: a %d-row window of %s declined", to, t)
	}
	copy(out[:n], res[:n])
	return nil
}

// ensureAct grows the quantized-activation buffers to hold k elements. They
// are sized at NewJIT from the widest k the caller named, and a matvec wider
// than that grows them here -- on the caller's goroutine, before any pool
// region reads them -- rather than declining.
func (f *JIT) ensureAct(k int) {
	if k > len(f.q) {
		f.q = make([]int8, k)
		f.cacheOK = false
	}
	if n := 2 * (k/cpu.Q8Block + 1); n > len(f.pairs) {
		f.pairs = make([]float32, n)
	}
	if n := k/16 + 2; n > len(f.half) {
		f.half = make([]float32, n)
	}
}

// quantKernel is the activation quantizer for t: the generated kernels and
// their constant block. It is the one lookup for prepAct, MatMulPacked and
// MatVecPackedGather, so they cannot disagree about which kernel a format gets.
func (f *JIT) quantKernel(t quant.Type) cpu.QuantActKernels {
	i := 0
	if cpu.NeedsHalfSums(t) {
		i = 1
	}
	konst := f.qkonst[t]
	if konst == nil {
		konst = cpu.QuantActConsts(t)
		f.qkonst[t] = konst
	}
	return cpu.QuantActKernels{Wide: f.qact[i], Narrow: f.qactNarrow[i], Konst: konst}
}

// quantScratch is worker w's slice of the narrow kernel's spill space. The
// kernel writes eight scales through it, so two workers sharing one slice
// would each quantize with the other's.
func (f *JIT) quantScratch(w int) []float32 {
	return f.qscr[w*cpu.QuantActNarrowScratch:]
}

// prepAct quantizes the activation for a packed matvec, once per (pointer,
// length, type, generation). It is the same prep the GGUF kernels use, and
// MatVecPackedMulti shares it so the cache check exists once.
func (f *JIT) prepAct(t quant.Type, x []float32, k, nb int) {
	if f.cacheOK && f.cachePtr == &x[0] && f.cacheLen == k &&
		f.cacheTyp == t && f.cacheGen == f.gen {
		return
	}
	tq := f.quantStart()
	// The sixteen-wide formats correct their bias from per-16 sums: their scale
	// changes every sixteen elements, so the per-32 pair would fold two
	// different scales into one correction.
	qk := f.quantKernel(t)
	var half []float32
	if cpu.NeedsHalfSums(t) {
		half = f.half[:k/16]
	}
	qs, ps, win := f.q[:k], f.pairs[:2*nb], f.actWindow
	if nb >= 4*f.pool.N() {
		j := &f.hot.quant
		j.qk, j.t, j.qs, j.ps, j.half, j.x, j.k, j.win = qk, t, qs, ps, half, x, k, win
		f.pool.Do(nb, max(1, nb/(4*f.pool.N())), f.hot.quantFn(f))
		j.x = nil
	} else {
		qk.Run(t, qs, ps, half, f.quantScratch(0), x, k, 0, nb, win)
	}
	f.quantSince(tq)
	f.cachePtr, f.cacheLen, f.cacheTyp, f.cacheGen, f.cacheOK = &x[0], k, t, f.gen, true
}

// numaRows is sched.DoNodes' layout for a packed weight's rows, or 0 when it
// has none. A word-line is stride*4 contiguous bytes; weight memory is
// interleaved one 4 KiB page at a time, so a 1024-row block of a line sits on
// one node -- and on the SAME node for every word when a line is a whole
// number of node rounds (stride a multiple of 1024 x nodes), which
// jlm.TestInterleaveFollowsThePageAddress measured as (addr>>12) mod nodes on
// a two-socket host. The node of block 0 is ASKED, not computed; any
// other shape runs unaffine, which is correct and only slower.
func (f *JIT) numaRows(p *Packed, row, stride, grp int) (block, first int) {
	n := f.pool.Nodes()
	const rowsPerPage = 4096 / 4
	if n < 2 || rowsPerPage%grp != 0 || row%rowsPerPage != 0 || stride%(rowsPerPage*n) != 0 ||
		uintptr(unsafe.Pointer(&p.QS[0]))%4096 != 0 {
		return 0, 0
	}
	first = f.pool.NodeIndex(sched.NodeOf(unsafe.Pointer(&p.QS[row*4])))
	if first < 0 {
		return 0, 0
	}
	return rowsPerPage / grp, first
}

// fusedSliceElems and fusedSliceRows decide when a matrix is split over k
// across workers as well as over rows: at least fusedSliceElems weights, and a
// worker's balanced share of rows under fusedSliceRows. See fusedSlices.
const (
	fusedSliceElems = 32 << 20
	fusedSliceRows  = 4096
)

// fusedSlices is how many k-slices MatVecPacked splits a matrix into.
//
// The kernel reads each payload word as one run of rows*4 bytes, so a worker's
// balanced share of rows sets the run length. Two k-slices over N/2 row chunks
// give every worker twice the rows over half of k -- the same bytes in runs
// twice as long -- and the partial outputs are added afterwards. Two, on a
// pool of even width (the count must divide the pool), for a large matrix
// whose runs are short; more slices shorten k past what the rows buy, and a
// matrix whose rows are already long (the output head) stays whole. See
// docs/engineering-history/cpu-kernels.md.
//
// The sum is a reassociation, so this path is bounded against the whole-k one
// rather than bit-identical (TestFusedKSlicesMatchWholeK); FusedKSlices -1
// turns it off, and so does GEMMExact unless a count is pinned. The decision
// depends on the pool's width, so the same model splits on one host and not
// on another.
func (f *JIT) fusedSlices(t quant.Type, k, nrows, per int) int {
	if f.cfg.FusedKSplit < 0 || f.cfg.FusedKSlices < 0 {
		return 1
	}
	nsup := k / cpu.PackedOuterElems(t)
	if f.cfg.FusedKSlices > 0 {
		return max(1, min(f.cfg.FusedKSlices, nsup, f.pool.N()))
	}
	// GEMMExact is the one summation order: every batched matmul sums a row
	// over the whole of k, so the matvec does too. Left to decide, a pool wide
	// enough to make the balanced chunk short (a 32000-row head on a
	// 14-worker two-socket pool) splits it, and decode then parts from the batched
	// path by the reassociation alone -- a tiny NMSE there, 0 on a narrower pool.
	if f.cfg.GEMMExact {
		return 1
	}
	if nrows*k < fusedSliceElems || per*grpOf(t) >= fusedSliceRows || nsup < 2 || f.pool.N()%2 != 0 {
		return 1
	}
	return 2
}

// fusedSliced runs the fused kernel over s k-slices x pool/s row chunks, slice
// 0 into out and the rest into partials that are added in afterwards.
func (f *JIT) fusedSliced(fused *cpu.Code, t quant.Type, q kernels.Quant, p *Packed, out []float32,
	row, stride, nrows, k, s int) {
	grp := grpOf(t)
	groups := nrows / grp
	nsup := k / cpu.PackedOuterElems(t)
	ss := (nsup + s - 1) / s // super-blocks per slice
	s = (nsup + ss - 1) / ss
	c := max(1, f.pool.N()/s)
	cg := (groups + c - 1) / c // row groups per chunk
	c = (groups + cg - 1) / cg
	rowsPer := cg * grp
	f.growPackedScratch(f.pool.Max() * rowsPer)
	if need := (s - 1) * nrows; len(f.kpart) < need {
		f.kpart = make([]float32, need)
	}
	f.ksliced++
	j := &f.hot.sliced
	j.fc = f.newFusedCall(fused, t, q, ss*cpu.PackedOuterElems(t), rowsPer)
	j.fc.nsup = nsup
	j.p, j.out, j.row, j.stride, j.nrows, j.groups = p, out, row, stride, nrows, groups
	j.s, j.ss, j.cg, j.rowsPer = s, ss, cg, rowsPer
	clear(out[:nrows])
	clear(f.kpart[:(s-1)*nrows])
	f.pool.DoLabeled(regionLabel(t, "/fused-k"), s*c, 1, f.hot.slicedFn(f))
	j.p, j.out = nil, nil
	for si := 1; si < s; si++ {
		Add32JIT(out[:nrows], f.kpart[(si-1)*nrows:si*nrows])
	}
}

// fusedChunk is the row-group chunk the fused kernel's callers hand the pool:
// the balanced share, one per worker, with the per-call budget paid in k
// (fusedCall) -- or, under FusedKSplit -1, the old budget-capped chunk.
func (f *JIT) fusedChunk(groups int, t quant.Type, k int) int {
	if f.cfg.FusedKSplit < 0 {
		return packedChunk(f, groups, t, k, grpOf(t))
	}
	return max(1, (groups+f.pool.N()-1)/f.pool.N())
}

// fusedCall is one fused kernel's walk of k in pieces (fusedPiece), shared by
// MatVecPacked and the batched mixture paths, which chunk rows the same way.
type fusedCall struct {
	f             *JIT
	code          *cpu.Code
	t             quant.Type
	step, nsup    int
	piece         int
	qsAdv, scAdv  int // words from one super-block to the next, per row
	grp, dRowByte int
}

func (f *JIT) newFusedCall(code *cpu.Code, t quant.Type, q kernels.Quant, k, rows int) fusedCall {
	step := cpu.PackedOuterElems(t)
	fc := fusedCall{f: f, code: code, t: t, step: step, nsup: k / step,
		piece: f.fusedPiece(t, k, rows), grp: grpOf(t), dRowByte: cpu.DRowBytes(t)}
	fc.qsAdv, fc.scAdv = superWords(q, step)
	return fc
}

// run computes row groups [lo, hi) of p (rows row..row+stride interleaved, as
// Packed.Row/Stride say) into out, from the quantized activation a, its block
// scales pairs and its per-16 sums half (nil when the format has none). scr is
// the worker's offset into the packed scratch.
func (fc *fusedCall) run(p *Packed, out []float32, row, stride, lo, hi int,
	a []int8, pairs, half []float32, scr int) {
	fc.runSupers(p, out, row, stride, lo, hi, 0, fc.nsup, a, pairs, half, scr)
}

// runSupers is run over super-blocks [sb, se) of k only.
func (fc *fusedCall) runSupers(p *Packed, out []float32, row, stride, lo, hi, sb, se int,
	a []int8, pairs, half []float32, scr int) {
	f := fc.f
	r0 := lo * fc.grp
	rowStr := stride * 4
	dstr := int64(cpu.DSuperBytes(fc.t, stride))
	for s0 := sb; s0 < se; s0 += fc.piece {
		e := s0 * fc.step
		args := cpu.Args{
			Out: &out[r0], W: &p.QS[(row+r0)*4+s0*fc.qsAdv*rowStr], A: &a[e],
			AScale: &pairs[e/cpu.Q8Block*2],
			Rows:   int64(hi - lo), RowStr: int64(rowStr),
			K: int64(min(fc.piece, se-s0)), Scr: &f.pkonst[0], AHalf: halfAt(half, e/16),
			PD:      &p.D[(row+r0)*fc.dRowByte+s0*int(dstr)],
			DStr:    dstr,
			Scratch: (*byte)(unsafe.Pointer(&f.pdscr[scr])),
			Q32:     &f.pmscr[scr],
		}
		if len(p.SC) > 0 {
			args.PSC = &p.SC[(row+r0)*4+s0*fc.scAdv*rowStr]
		}
		fc.code.Call(&args)
	}
}

// fusedPiece is how many super-blocks one call of the fused kernel may walk
// over rows rows and stay inside invariant I3's budget (cpu.RowsPerCallFor):
// the whole of k when the rows fit, otherwise the largest piece that does.
// FusedKSplit pins it (a positive count), and -1 restores the old contract --
// the whole of k per call with the row chunk capped instead.
func (f *JIT) fusedPiece(t quant.Type, k, rows int) int {
	step := cpu.PackedOuterElems(t)
	nsup := k / step
	switch {
	case f.cfg.FusedKSplit < 0:
		return nsup
	case f.cfg.FusedKSplit > 0:
		return min(nsup, f.cfg.FusedKSplit)
	}
	for ps := nsup; ps > 1; ps-- {
		if cpu.RowsPerCallFor(f.tier, t, ps*step) >= rows {
			return ps
		}
	}
	return 1
}

// superWords is how many 32-bit words one super-block of one row occupies in
// the payload plane and in the SC plane: the row-strided distance, in words,
// from one super-block to the next. Derived from the packer's own arithmetic.
func superWords(q kernels.Quant, step int) (qs, sc int) {
	nq, _, nsc, err := kernels.PackedWords(q, 64, step)
	if err != nil {
		panic(fmt.Sprintf("nn: superWords: %v", err))
	}
	return nq / 64, nsc / 64
}

// packedChunk is how many row groups one kernel call serves. Two limits pull
// opposite ways: a goroutine inside generated code cannot be async-preempted,
// so a long call stalls every GC (cpu.RowsPerCallFor prices that per tier),
// and one chunk per worker makes the region as slow as its unluckiest core.
func packedChunk(f *JIT, groups int, t quant.Type, k, rowsPerGroup int) int {
	n := cpu.RowsPerCallFor(f.tier, t, k) / rowsPerGroup
	if n < 1 {
		n = 1
	}
	if bal := (groups + f.pool.N() - 1) / f.pool.N(); bal > 0 && bal < n {
		n = bal
	}
	return n
}

// hotRegions is the per-JIT state of the three regions every decode token runs
// many times -- the fused matvec, the tiled one and the activation quantize --
// with the closure each hands the pool built once. A closure literal at the
// call site escapes and allocates per matvec, and on macOS the scavenger's
// madvise calls made that a large share of decode. A JIT runs one region at a
// time (sched.Pool is not reentrant), so one set of fields serves every call;
// each is cleared of its slices afterwards.
type hotRegions struct {
	// group is MatVecPackedGroup's (packedgroup.go).
	group struct {
		m                         [MaxGroup]groupMat
		n, nch, nslot, k, maxRows int
		slotT                     [MaxGroup]quant.Type
		slotQK                    [MaxGroup]cpu.QuantActKernels
		x, actUp                  []float32
		act                       ActKind
		resid                     []float32
		alpha                     float32
		gen                       uint64
		fn                        func(worker, lo, hi int)
	}
	fused struct {
		fc                   fusedCall
		p                    *Packed
		out                  []float32
		row, stride, rowsPer int
		fn                   func(worker, lo, hi int)
	}
	tiled struct {
		code                    *cpu.Code
		p                       *Packed
		out                     []float32
		row, stride, ksteps, db int
		dstr                    int64
		fn                      func(worker, lo, hi int)
	}
	// sliced is fusedSliced's region: a short, long-k matrix split over k as
	// well as rows. It runs once a token wherever the pool is wide enough to
	// make a balanced row chunk short (a vocabulary head on a 14-worker
	// socket), so a closure over its locals was two heap objects a token
	// there and nothing on a six-worker host, which never splits that shape.
	sliced struct {
		fc                         fusedCall
		p                          *Packed
		out                        []float32
		row, stride, nrows, groups int
		s, ss, cg, rowsPer         int
		fn                         func(worker, lo, hi int)
	}
	multi, gather multiRegion
	quant         struct {
		qk          cpu.QuantActKernels
		t           quant.Type
		qs          []int8
		ps, half, x []float32
		k, win      int
		fn          func(worker, lo, hi int)
	}
	// gatherQuant is MatVecPackedGather's quantize of every activation.
	gatherQuant struct {
		qk                        cpu.QuantActKernels
		t                         quant.Type
		x                         []float32
		xstride, k, nb, half, win int
		fn                        func(worker, lo, hi int)
	}
	// float is the F32/F16/BF16 matvec: a mixture's router, every token.
	float struct {
		code  *cpu.Code
		out   []float32
		w     []byte
		x     []float32
		k, es int
		fn    func(lo, hi int)
	}
	// floatMM is MatMulFloat's tile of tokens over rows: an encoder's every
	// projection, every block of every sequence.
	floatMM struct {
		code               *cpu.Code
		out, x             []float32
		w                  []byte
		k, rowBytes, nrows int
		fn                 func(lo, hi int)
	}
}

func (h *hotRegions) fusedFn(f *JIT) func(worker, lo, hi int) {
	if h.fused.fn == nil {
		h.fused.fn = func(worker, lo, hi int) {
			j := &h.fused
			j.fc.run(j.p, j.out, j.row, j.stride, lo, hi, f.q, f.pairs, f.half, worker*j.rowsPer)
		}
	}
	return h.fused.fn
}

func (h *hotRegions) slicedFn(f *JIT) func(worker, lo, hi int) {
	if h.sliced.fn == nil {
		h.sliced.fn = func(worker, lo, hi int) {
			j := &h.sliced
			for i := lo; i < hi; i++ {
				si, ci := i%j.s, i/j.s
				g0 := ci * j.cg
				dst := j.out
				if si > 0 {
					dst = f.kpart[(si-1)*j.nrows : si*j.nrows]
				}
				j.fc.runSupers(j.p, dst, j.row, j.stride, g0, min(j.groups, g0+j.cg), si*j.ss,
					min(j.fc.nsup, (si+1)*j.ss), f.q, f.pairs, f.half, worker*j.rowsPer)
			}
		}
	}
	return h.sliced.fn
}

func (h *hotRegions) tiledFn(f *JIT) func(worker, lo, hi int) {
	if h.tiled.fn == nil {
		h.tiled.fn = func(_, lo, hi int) {
			j := &h.tiled
			r0 := lo * cpu.PackedRows
			args := cpu.Args{
				Out: &j.out[r0], W: &j.p.QS[(j.row+r0)*4], A: &f.q[0], AScale: &f.pairs[0],
				Rows: int64(hi - lo), RowStr: int64(j.stride * 4),
				K: int64(j.ksteps), Scr: &f.pkonst[0],
				AHalf: &f.half[0], PD: &j.p.D[(j.row+r0)*j.db], DStr: j.dstr,
			}
			if len(j.p.SC) > 0 {
				args.PSC = &j.p.SC[(j.row+r0)*4]
			}
			j.code.Call(&args)
		}
	}
	return h.tiled.fn
}

func (h *hotRegions) floatFn() func(lo, hi int) {
	if h.float.fn == nil {
		h.float.fn = func(lo, hi int) {
			j := &h.float
			args := cpu.Args{
				Out: &j.out[lo], W: &j.w[lo*j.k*j.es],
				// Args.A means "the activations"; its type is *int8 because
				// every other kernel quantizes them. Still a real Go pointer,
				// so the GC scans it (see the Args doc).
				A:      (*int8)(unsafe.Pointer(&j.x[0])),
				Rows:   int64(hi - lo),
				K:      int64(j.k),
				RowStr: int64(j.k * j.es),
			}
			j.code.Call(&args)
		}
	}
	return h.float.fn
}

func (h *hotRegions) floatMMFn() func(lo, hi int) {
	if h.floatMM.fn == nil {
		h.floatMM.fn = func(lo, hi int) {
			j := &h.floatMM
			j.code.Call(&cpu.Args{
				Out: &j.out[lo], W: &j.w[lo*j.rowBytes], A: (*int8)(unsafe.Pointer(&j.x[0])),
				Rows: int64(hi - lo), K: int64(j.k), RowStr: int64(j.rowBytes), OutStr: int64(4 * j.nrows),
			})
		}
	}
	return h.floatMM.fn
}

func (h *hotRegions) quantFn(f *JIT) func(worker, lo, hi int) {
	if h.quant.fn == nil {
		h.quant.fn = func(w, blo, bhi int) {
			j := &h.quant
			j.qk.Run(j.t, j.qs, j.ps, j.half, f.quantScratch(w), j.x, j.k, blo, bhi, j.win)
		}
	}
	return h.quant.fn
}

// regionLabel is a region's pprof label, or "" while labelling is off -- the
// concatenation allocated on every call either way.
func regionLabel(t quant.Type, what string) string {
	if !sched.RegionLabels() {
		return ""
	}
	return t.String() + what
}
