//go:build amd64 || arm64

package nn

import (
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// MatMulPacked is MatVecPacked for many activation rows at once: out is
// [ntok][nrows] and x is [ntok][k], against one weight in the device layout.
//
// It runs one pool dispatch over row groups with the token loop inside it,
// rather than two dispatches (quantize, kernel) per token, using the
// weight-stationary GEMM, the token-tiled kernel or the per-token fused kernel
// as available.
//
// It declines rather than approximating: a caller that gets false loops
// MatVecPacked and gets exactly what it had.
func (f *JIT) MatMulPacked(out []float32, t quant.Type, p *Packed, x []float32, nrows, k, ntok int) bool {
	if f == nil || p == nil || ntok < 2 || len(p.QS) == 0 {
		return false
	}
	if len(out) < nrows*ntok || len(x) < ntok*k {
		return false
	}
	// x is ntok rows of kx; a ragged kx reads a weight packed at k
	// (ragged.go), each row's tail quantized as zeros.
	kx := k
	k = raggedK(t, k)
	// Either the fused kernel or the tile will do: without a fused kernel the
	// tile serves the remainder itself (tok=1 is a legal width).
	fused := f.packedFused[t]
	if nrows%grpOf(t) != 0 {
		return false
	}
	step := cpu.PackedOuterElems(t)
	if step == 0 || k%step != 0 {
		return false
	}
	// The d plane has its own row and super-block strides; on a narrow format
	// neither is the payload's. Both come from cpu, so this caller and the
	// emitters cannot drift apart.
	db, dstr := cpu.DRowBytes(t), int64(cpu.DSuperBytes(t, nrows))
	nb := k / cpu.Q8Block
	if nb == 0 {
		return false
	}
	stride, row := p.Stride, p.Row
	if stride == 0 {
		stride = nrows
	}

	// The activation scratch is per token: a worker walking its row range
	// visits every token, so all ntok quantized activations must be live.
	half := 0
	if cpu.NeedsHalfSums(t) {
		half = k / 16
	}
	// The weight-stationary GEMM reads its tokens at a padded stride
	// (cpu.GEMMPad), so whether it will run decides the layout the quantizer
	// writes. Every other path below reads them dense.
	// NoGEMM (the prefill gates' exact arm) turns this GEMM off with the GGUF
	// one: its integer-accumulating form is not bit-identical to the matvec.
	var ws *cpu.Code
	if f.cfg.GEMMTok >= 0 && !f.cfg.NoGEMM && nrows%8 == 0 && f.gemmRows != gemmOff && kx == k {
		ws = f.gemmFor(t, k, nrows)
	}
	qs, ps, hsz := k, 2*nb, half
	if ws != nil {
		qs, ps, hsz = k+cpu.GEMMPad, 2*nb+cpu.GEMMPad/4, half+cpu.GEMMPad/4
	}
	f.growPackedMM(ntok, qs, ps, hsz)
	if kx != k {
		f.growRagStage(k)
	}

	tq := f.quantStart()
	// One dispatch for every token's quantize, where there were ntok. The
	// region's state is the JIT's (mmJob) and its function a method value
	// built once, so a warm call allocates nothing.
	j := &f.mmj
	*j = mmJob{out: out, t: t, p: p, x: x, nrows: nrows, k: k, kx: kx, ntok: ntok, row: row, stride: stride,
		db: db, dstr: dstr, half: half, qs: qs, ps: ps, hsz: hsz, nb: nb, win: f.actWindow,
		qk: f.quantKernel(t), step: step}
	defer func() { f.mmj = mmJob{} }()
	f.mmFns()
	f.pool.Do(ntok, max(1, ntok/(4*f.pool.N())), f.mmQuantFn)
	f.quantSince(tq)

	if ws != nil {
		f.packedGEMM(ws)
		return true
	}

	groups := nrows / grpOf(t)
	per := f.batchedChunk(groups, t, k, grpOf(t), ntok, p, stride)
	rowsPer := per * grpOf(t)
	f.mmChunkRows.Store(int64(rowsPer))
	f.growPackedScratch(f.pool.Max() * rowsPer)
	// The kernel ACCUMULATES into Out, so the caller owns the zero.
	clear(out[:nrows*ntok])

	// One dispatch for every token's arithmetic, where there were ntok. A
	// worker owns a row range for the WHOLE batch, so its slice of the weight
	// stays hot across the token loop instead of being re-reached per dispatch.
	f.mmCalls.Add(1)
	// The token-tiled kernel for the tokens it covers, the per-token one for
	// the remainder, in the same dispatch: ntok is rarely a multiple of the
	// tile, and a separate region for the remainder would cost a dispatch.
	tiled, ttok := f.tiledFor(t, k, nrows, 0)
	nt := 0
	if tiled != nil && ttok >= 2 {
		nt = ntok / ttok * ttok
	}
	// The remainder: the fused per-token kernel where it exists, a one-token
	// tile where it does not.
	var rem1 *cpu.Code
	if fused == nil && nt < ntok {
		rem1, _ = f.tiledFor(t, k, nrows, 1)
	}
	if fused == nil && tiled == nil {
		return false
	}
	if fused == nil && nt < ntok && rem1 == nil {
		return false
	}
	// Count that it ran, not that it exists: ntok below one tile leaves every
	// token on the per-token path.
	if nt > 0 {
		f.tiledCalls.Add(1)
	}
	j.fused, j.tiled, j.rem1, j.nt, j.ttok, j.rowsPer = fused, tiled, rem1, nt, ttok, rowsPer
	f.pool.DoLabeled(regionLabel(t, "/fused-mm"), groups, per, f.mmRunFn)
	return true
}

// mmRun is one worker's row range of MatMulPacked's arithmetic: the tiled
// kernel for whole tiles of tokens, the per-token one for the remainder.
func (f *JIT) mmRun(worker, lo, hi int) {
	j := &f.mmj
	out, p, k, nrows, row, stride, db, dstr, half := j.out, j.p, j.k, j.nrows, j.row, j.stride, j.db, j.dstr, j.half
	nb, step, fused, tiled, rem1, nt, ttok, ntok, rowsPer := j.nb, j.step, j.fused, j.tiled, j.rem1, j.nt, j.ttok, j.ntok, j.rowsPer
	t := j.t
	r0 := lo * grpOf(t)
	dscr := (*byte)(unsafe.Pointer(&f.pdscr[worker*rowsPer]))
	mscr := &f.pmscr[worker*rowsPer]
	for i := 0; i < nt; i += ttok {
		args := cpu.Args{
			Out: &out[i*nrows+r0], W: &p.QS[(row+r0)*4],
			A: &f.mq[i*k], AScale: &f.mpairs[i*2*nb],
			Rows: int64(hi - lo), RowStr: int64(stride * 4),
			K: int64(k / step), Scr: &f.pkonst[0],
			PD:      &p.D[(row+r0)*db],
			DStr:    dstr,
			Scratch: dscr,
			// R10 is the minimum scratch for a format with a dmin array and
			// the per-16 activation sums for a 16-wide one. They are
			// mutually exclusive by construction -- a format with a minimum
			// has 32-element sub-blocks -- which is what lets one register
			// serve both.
			Q32: mscr,
			// Offset to the tile, not the batch: the kernel indexes AHalf
			// at i*ahTok for i in [0,tok), so a base pointer would make
			// every tile after the first read token 0's half-sums.
			AHalf: halfAt(f.mhalf, i*half),
		}
		if len(p.SC) > 0 {
			args.PSC = &p.SC[(row+r0)*4]
		}
		tiled.Call(&args)
	}
	for i := nt; i < ntok && fused == nil; i++ {
		args := cpu.Args{
			Out: &out[i*nrows+r0], W: &p.QS[(row+r0)*4],
			A: &f.mq[i*k], AScale: &f.mpairs[i*2*nb],
			Rows: int64(hi - lo), RowStr: int64(stride * 4),
			K: int64(k / step), Scr: &f.pkonst[0],
			PD:      &p.D[(row+r0)*db],
			DStr:    dstr,
			Scratch: dscr,
			Q32:     mscr,
			AHalf:   halfAt(f.mhalf, i*half),
		}
		if len(p.SC) > 0 {
			args.PSC = &p.SC[(row+r0)*4]
		}
		rem1.Call(&args)
	}
	for i := nt; i < ntok && fused != nil; i++ {
		args := cpu.Args{
			Out: &out[i*nrows+r0], W: &p.QS[(row+r0)*4],
			A: &f.mq[i*k], AScale: &f.mpairs[i*2*nb],
			Rows: int64(hi - lo), RowStr: int64(stride * 4),
			K: int64(k / step), Scr: &f.pkonst[0],
			PD:      &p.D[(row+r0)*db],
			DStr:    dstr,
			Scratch: dscr,
			Q32:     mscr,
		}
		if half > 0 {
			args.AHalf = &f.mhalf[i*half]
		}
		if len(p.SC) > 0 {
			args.PSC = &p.SC[(row+r0)*4]
		}
		fused.Call(&args)
	}
}

// mmQuant is one worker's tokens of MatMulPacked's activation quantize.
func (f *JIT) mmQuant(wk, lo, hi int) {
	j := &f.mmj
	for i := lo; i < hi; i++ {
		q := f.mq[i*j.qs : i*j.qs+j.k]
		var hs []float32
		if j.half > 0 {
			hs = f.mhalf[i*j.hsz : i*j.hsz+j.half]
		}
		f.quantRow(j.qk, j.t, q, f.mpairs[i*j.ps:i*j.ps+2*j.nb], hs, f.quantScratch(wk), wk,
			j.x[i*j.kx:(i+1)*j.kx], j.kx, j.k, 0, j.nb, j.win)
	}
}

// mmJob is MatMulPacked's state for one call. It lives on the JIT, beside the
// scratch the call already reuses, so the pool regions read it through method
// values built once (mmFns) and a warm call allocates nothing.
type mmJob struct {
	out, x                                    []float32
	t                                         quant.Type
	p                                         *Packed
	nrows, k, kx, ntok, row, stride, db, half int
	dstr                                      int64
	qs, ps, hsz, nb, win, step                int
	qk                                        cpu.QuantActKernels
	fused, tiled, rem1                        *cpu.Code
	nt, ttok, rowsPer                         int
	// The weight-stationary GEMM's (packedGEMM).
	code                              *cpu.Code
	mout                              []float32
	os, gT, ntt, per, groups, sstride int
}

// mmFns builds the regions' method values on first use.
func (f *JIT) mmFns() {
	if f.mmQuantFn == nil {
		f.mmQuantFn, f.mmRunFn, f.mmGemmFn, f.mmCopyFn = f.mmQuant, f.mmRun, f.mmGemm, f.mmCopy
	}
}

// batchedChunk is packedChunk with a locality cap (Config.ChunkBytes) on top.
// The token loop lives inside the per-chunk function, so a worker re-reads its
// row slice once per token tile and the cache level holding the slice serves
// every re-read. Batched path only: a decode matvec reads each weight once.
// Whether the cap pays is host-dependent; see newRowChunkDuel.
//
// The per-row size comes from the buffer: QS, D and SC cover bufRows rows
// between them. bufRows is the whole buffer's row count (Packed.Stride), not
// this weight's, since a weight may be a row range of a shared buffer.
func (f *JIT) batchedChunk(groups int, t quant.Type, k, rowsPerGroup, ntok int, p *Packed, bufRows int) int {
	per := packedChunk(f, groups, t, k, rowsPerGroup)
	if f.chunkBytes <= 0 || ntok < 2 || p == nil || bufRows <= 0 {
		return per
	}
	perRow := (len(p.QS) + len(p.D) + len(p.SC)) / bufRows
	if perRow <= 0 {
		return per
	}
	// At least one group, always: refusing would make the shape unrunnable
	// rather than slow, which is RowsPerCall's own rule one level up.
	if n := f.chunkBytes / (perRow * rowsPerGroup); n >= 1 && n < per {
		per = n
	}
	return per
}

// BatchedChunkRows reports how many rows the last batched matmul gave one pool
// chunk, so a gate can assert a locality cap narrowed something.
func (f *JIT) BatchedChunkRows() int {
	if f == nil {
		return 0
	}
	return int(f.mmChunkRows.Load())
}

// growPackedScratch sizes the per-worker scale scratch the packed kernels share.
// The two buffers grow independently because not every kernel uses both (the
// weight-stationary GEMM grows pdscr only).
func (f *JIT) growPackedScratch(need int) {
	if len(f.pdscr) < need {
		f.pdscr = make([]float32, need)
	}
	if len(f.pmscr) < need {
		f.pmscr = make([]float32, need)
	}
}

// growPackedMM sizes the batched activation staging. It grows and never
// shrinks. The three arguments are the per-token strides in elements, which
// the weight-stationary GEMM pads (cpu.GEMMPad).
func (f *JIT) growPackedMM(ntok, qs, ps, hs int) {
	if len(f.mq) < ntok*qs {
		f.mq = make([]int8, ntok*qs)
	}
	if len(f.mpairs) < ntok*ps {
		f.mpairs = make([]float32, ntok*ps)
	}
	if hs > 0 && len(f.mhalf) < ntok*hs {
		f.mhalf = make([]float32, ntok*hs)
	}
}

// PackedBatchCalls reports how many MatMulPacked calls ran the batched kernel.
// A decline produces the same answer more slowly, so only a counter can tell a
// working batch from one that never ran.
func (f *JIT) PackedBatchCalls() int64 {
	if f == nil {
		return 0
	}
	return f.mmCalls.Load()
}

// tiledFor returns this shape's token-tiled kernel, emitting it on first use.
// It is lazy because k and nrows are baked as immediates (keeping them off the
// register budget), so there is one kernel per (type, k, nrows).
func (f *JIT) tiledFor(t quant.Type, k, nrows, want int) (*cpu.Code, int) {
	tok := f.em.MaxTiledTokens(t)
	// A pin narrows, it does not widen: a width beyond the register file would
	// emit nothing and silently disable the kernel.
	if f.tilePin > 0 && f.tilePin < tok {
		tok = f.tilePin
	}
	if want > 0 {
		tok = want
	}
	if tok < 1 || nrows%grpOf(t) != 0 || f.pkonst == nil {
		return nil, 0
	}
	key := tiledKey{t, k, nrows, tok}
	f.tiledMu.Lock()
	defer f.tiledMu.Unlock()
	if c, ok := f.tiled[key]; ok {
		return c, f.tiledTok[key] // nil is a cached REFUSAL, not a miss
	}
	if f.tiled == nil {
		f.tiled = make(map[tiledKey]*cpu.Code)
		f.tiledTok = make(map[tiledKey]int)
	}
	// The tile is bounded by the L1i code budget as well as by registers, and
	// only the latter is known before emitting (Q6_K fits three tokens in
	// registers and then assembles over budget). Narrow until it fits rather
	// than refusing the format: two tokens still halves the payload traffic.
	var code *cpu.Code
	floor := 2
	if want > 0 {
		floor = want
	}
	for ; tok >= floor; tok-- {
		// The tier's table, because this kernel is about to run. A pre-VNNI
		// host's tiled emitter refuses and the SSE tier has none
		// (MaxTiledTokens is 0); either refusal is cached below, and
		// MatMulPacked falls to the per-token fused kernel.
		b, err := f.em.PackedTiled(t, k, nrows, tok)
		if err != nil {
			continue
		}
		if c, err := cpu.MapNamed(b, t.String()+"_packed_tiled"); err == nil {
			code = c
			break
		}
	}
	if code == nil {
		tok = 0
	}
	f.tiled[key] = code
	f.tiledTok[key] = tok
	return code, tok
}

// TiledWidth reports the token tile this shape would run at, emitting the
// kernel if it has not been built yet, and 0 when there is none. It is how a
// gate checks a pin took effect; TiledMatMulCalls cannot say how wide a tile was.
func (f *JIT) TiledWidth(t quant.Type, k, nrows int) int {
	if f == nil {
		return 0
	}
	_, tok := f.tiledFor(t, k, nrows, 0)
	return tok
}

// TiledMatMulCalls reports how many MatMulPacked calls ran the token-tiled
// kernel, so a gate can assert it was selected rather than quietly declined
// into the per-token path.
func (f *JIT) TiledMatMulCalls() int64 {
	if f == nil {
		return 0
	}
	return f.tiledCalls.Load()
}

type tiledKey struct {
	t            quant.Type
	k, rows, tok int
}

// halfAt is &s[i] for a slice that may be empty, which it is for every format
// with no per-16 activation sums.
func halfAt(s []float32, i int) *float32 {
	if len(s) == 0 {
		return nil
	}
	return &s[i]
}

// gemmRows is how many rows one call of the weight-stationary GEMM covers when
// nothing pins it (measured on a 12-core host; the gemmrows duel re-measures
// per host). Narrow rows win because a call is a work item, so a short
// projection still spreads over every worker. Below 16 rows a block is not a
// whole 64-byte output line (cpu.OutLine) and neighbouring workers trade the
// line on every store.
const gemmRows = 16

// gemmCallMACs bounds one call against the GC-preemption budget
// (cpu.RowsPerCall's reason): at the ~17 MAC/cycle/core this kernel sustains,
// 4 Mi of them is on the order of 100 us. It decides the tokens
const gemmCallMACs = 4 << 20

// packedGEMM runs MatMulPacked's arithmetic through the weight-stationary GEMM
// and reports whether it did. The activations are already quantized into
// f.mq/f.mpairs/f.mhalf.
//
// A work item is (row block, token block); splitting tokens as well as rows
// keeps a short projection from running on a fraction of the workers. It
// writes a padded scratch and copies out, for the reason cpu.GEMMPad gives:
// nrows*4 is a multiple of 4 KiB on every real projection.
func (f *JIT) packedGEMM(code *cpu.Code) {
	j := &f.mmj
	t, k, ntok, nrows, half := j.t, j.k, j.ntok, j.nrows, j.half
	T := f.cfg.GEMMTok
	if T == 0 {
		T = min(64, max(8, gemmCallMACs/(gemmRows*k)/8*8))
	}
	os := nrows + cpu.GEMMPad/4
	if len(f.mout) < ntok*os {
		f.mout = make([]float32, ntok*os)
	}
	mout := f.mout[:ntok*os]
	groups := nrows / 8
	T = min(T, ntok)
	per := gemmRows / 8
	switch {
	case f.cfg.GEMMRows > 0:
		per = f.cfg.GEMMRows / 8
	case f.gemmRows > gemmOff:
		per = f.gemmRows / 8 // the gemmrows duel's choice
	}
	per = max(1, min(per, groups))
	nrc := (groups + per - 1) / per
	ntt := (ntok + T - 1) / T
	rowsPer := per * 8
	f.mmChunkRows.Store(int64(rowsPer))
	// Each worker's Scratch is the kernel's row scales (16 f32) and, on the
	// integer-accumulating form, 16 f32 of accumulators per token.
	sstride := max(rowsPer, 16+16*T)
	if need := f.pool.Max() * sstride; len(f.pdscr) < need {
		f.pdscr = make([]float32, need)
	}
	clear(mout)
	f.mmCalls.Add(1)
	f.gemmCalls.Add(1)
	j.code, j.mout, j.os, j.gT, j.ntt, j.per, j.groups, j.sstride = code, mout, os, T, ntt, per, groups, sstride
	j.qs, j.ps, j.hsz = k+cpu.GEMMPad, 2*(k/cpu.Q8Block)+cpu.GEMMPad/4, half+cpu.GEMMPad/4
	f.pool.DoLabeled(regionLabel(t, "/gemm"), nrc*ntt, 1, f.mmGemmFn)
	f.pool.Do(ntok, max(1, ntok/(2*f.pool.N())), f.mmCopyFn)
}

// mmGemm is one worker's (row block, token block) items of packedGEMM.
func (f *JIT) mmGemm(worker, lo, hi int) {
	j := &f.mmj
	p := j.p
	dscr := (*byte)(unsafe.Pointer(&f.pdscr[worker*j.sstride]))
	for it := lo; it < hi; it++ {
		g0 := (it / j.ntt) * j.per
		g1 := min(g0+j.per, j.groups)
		t0 := (it % j.ntt) * j.gT
		r0 := g0 * 8
		args := cpu.Args{
			Out: &j.mout[t0*j.os+r0], W: &p.QS[(j.row+r0)*4],
			A: &f.mq[t0*j.qs], AScale: &f.mpairs[t0*j.ps],
			Rows: int64(g1 - g0), RowStr: int64(j.stride * 4),
			K: int64(j.k / j.step), Scr: &f.pkonst[0],
			PD:      &p.D[(j.row+r0)*j.db],
			DStr:    j.dstr,
			Scratch: dscr,
			AHalf:   halfAt(f.mhalf, t0*j.hsz),
			Cols:    int64(min(j.gT, j.ntok-t0)),
		}
		if len(p.SC) > 0 {
			args.PSC = &p.SC[(j.row+r0)*4]
		}
		j.code.Call(&args)
	}
}

// mmCopy is one worker's tokens of packedGEMM's copy out of its padded
// scratch.
func (f *JIT) mmCopy(_, lo, hi int) {
	j := &f.mmj
	for i := lo; i < hi; i++ {
		copy(j.out[i*j.nrows:(i+1)*j.nrows], j.mout[i*j.os:i*j.os+j.nrows])
	}
}

type wsKey struct {
	t        quant.Type
	k, nrows int
}

// gemmFor returns the weight-stationary GEMM for this shape, emitting it on
// first use; nil when the tier or the format has none (a cached refusal).
func (f *JIT) gemmFor(t quant.Type, k, nrows int) *cpu.Code {
	if f.pkonst == nil || f.em.PackedGEMM == nil {
		return nil
	}
	key := wsKey{t, k, nrows}
	f.tiledMu.Lock()
	defer f.tiledMu.Unlock()
	if c, ok := f.wsGEMM[key]; ok {
		return c
	}
	if f.wsGEMM == nil {
		f.wsGEMM = make(map[wsKey]*cpu.Code)
	}
	var code *cpu.Code
	win := f.actWindow
	if f.cfg.GEMMExact {
		win = 0 // no window covers a super-block, so no integer accumulation
	}
	if b, err := f.em.PackedGEMM(t, k, nrows, win); err == nil {
		if c, err := cpu.MapNamed(b, t.String()+"_packed_gemm"); err == nil {
			code = c
		}
	}
	f.wsGEMM[key] = code
	return code
}

// PackedGEMMCalls reports how many MatMulPacked calls ran the weight-stationary
// GEMM, so a gate can assert the kernel under test was selected.
func (f *JIT) PackedGEMMCalls() int64 {
	if f == nil {
		return 0
	}
	return f.gemmCalls.Load()
}
