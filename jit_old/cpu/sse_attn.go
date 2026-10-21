package cpu

import "fmt"

// The attention kernels on the SSE tier: attn.go's seven kernels in legacy
// 128-bit code, for a host with no AVX at all. The ABI is attn.go's field for
// field (hd and kvStride baked in elements, f16 selecting the cache type), so
// nn calls these exactly as it calls the AVX2 ones.
//
// What changes:
//
//   - Four lanes, not eight. A head's tail is hd%4, done a dimension at a time
//     in lane 0.
//   - No FMA: each multiply-accumulate is MULPS then ADDPS and rounds twice,
//     so these are not bit-identical to the AVX2 kernels.
//   - The query is not a memory operand: a legacy packed op with an m128
//     operand faults on an address that is not 16-byte aligned, and a Go
//     []float32 is 4-aligned. It is MOVUPS'd into a register first.
//
// The bit-identities kept within the tier, which nn and model rely on:
//
//   - A paired kernel is two single-head calls, bit for bit: attnScoreRowSSE
//     is one body for the single, paired and tiled score kernels.
//   - An Into kernel split across KV pages is one contiguous call: every
//     accumulate nests the output dimension outside the position walk, so the
//     additions happen in position order whatever the block size.
//   - An f16 cache on values that are exact halves is the f32 cache, since
//     the widening (halfToFloatSSE) is exact.
//
// Without F16C the f16 kernels are much larger (PMOVZXWD plus the software
// widening per four halves), so nn.JIT.KVWidthPaysOff keeps the cache f32 on
// this tier unless the caller forces f16 (model.WithKVF16).
//
// Every kernel starts with DeclareISA(ISATierSSE) and ends in a plain RET.
// RDI is *Args throughout, R12 and R13 are saved by the trampoline, and R14,
// RBP and RSP are never written.

// attnKVSSE is how an SSE attention kernel reads its KV cache: four elements
// into a vector, or one element into lane 0. An f32 cache is MOVUPS and MOVSS;
// an f16 cache is PMOVZXWD (or PINSRW for one element) and halfToFloatSSE,
// whose four registers this carries; a q8 cache is PMOVSXBD, CVTDQ2PS and a
// multiply by the block's scale, broadcast into t0. For an f32 cache the four
// are unused and may be anything; a q8 cache uses t0 alone.
type attnKVSSE struct {
	a                  kvAddr
	src, t0, t1, magic Reg
}

// setup loads the widening constant into k.magic, once, clobbering gp. It
// emits nothing for an f32 or q8 cache.
func (k attnKVSSE) setup(a *Buf, gp Reg) {
	if k.a.f == KVF16 {
		loadHalfMagicSSE(a, k.magic, gp)
	}
}

// load4 brings four cache elements from element e into dst as float32. For
// f16, PMOVZXWD from memory reads exactly eight bytes and zero-extends four
// u16 into the dwords halfToFloatSSE takes. For q8, PMOVSXBD reads four bytes;
// e is a multiple of four, so the four share one 32-element block's scale.
func (k attnKVSSE) load4(a *Buf, dst, base Reg, e int) {
	switch k.a.f {
	case KVF16:
		a.PMOVZXWDLoad(k.src, At(base, k.a.off(e)))
		halfToFloatSSE(a, dst, k.src, k.t0, k.t1, k.magic)
	case KVQ8:
		a.PMOVSXBDLoad(dst, At(base, k.a.off(e)))
		a.CVTDQ2PS(dst, dst)
		bcastSS(a, k.t0, At(base, k.a.scale(e)))
		a.MULPS(dst, dst, k.t0)
	default:
		a.MOVUPSLoad(dst, At(base, k.a.off(e)))
	}
}

// load1 brings element e into lane 0 of dst -- the tail of a head that is not
// a whole vector. f32 and f16 read exactly the element's bytes (MOVSS four,
// PINSRW two, so the last element of the last row is never read past) and
// zero lanes 1..3. q8 is load4 from e: lanes 1..3 hold the next elements,
// which every tail ignores (it runs MULSS and ADDSS on lane 0 alone), and the
// four-byte read stays inside the row, whose scale pairs follow its int8 run.
func (k attnKVSSE) load1(a *Buf, dst, base Reg, e int) {
	switch k.a.f {
	case KVF16:
		a.PXOR(k.src, k.src, k.src)
		a.PINSRWLoad(k.src, k.src, At(base, k.a.off(e)), 0)
		halfToFloatSSE(a, dst, k.src, k.t0, k.t1, k.magic) // a zero half widens to +0
	case KVQ8:
		k.load4(a, dst, base, e)
	default:
		a.MOVSSLoad(dst, At(base, k.a.off(e)))
	}
}

// attnScoreHeadSSE is one query of a score kernel: its accumulator chains
// start at acc (acc, acc+1, ... for as many chains as the kernel runs), its
// query is at q+qOff, and its score for the current position goes to
// out+outOff.
type attnScoreHeadSSE struct {
	acc    Reg
	q      Reg
	qOff   int32
	out    Reg
	outOff int32
}

// attnScoreRowSSE emits one position of a score kernel for every head in hs:
// score = dot(query, K row at RDX), loading each K vector ONCE for all heads.
//
// chains is 4 (EmitAttnScores' latency argument: four independent ADDPS chains
// per head, vector i into chain i%4) or 1 (the tiled kernel: qt independent
// queries are the independent chains). The tail's hd%4 dimensions go into lane
// 0 of chain 0. The reduction is ((c0 + c1) + (c2 + c3)) lane-wise and then
// the two HADDPS folds, so with one chain a head's score is
// (l0 + l1) + (l2 + l3) over its four lanes -- oracle.Dot32's order exactly.
//
// k holds the K vector and t the product; with an f16 cache t may be kv.t0,
// because the widening is finished before the first product.
func attnScoreRowSSE(a *Buf, kv attnKVSSE, hd, chains int, hs []attnScoreHeadSSE, k, t Reg) {
	for _, h := range hs {
		for c := 0; c < chains; c++ {
			r := h.acc + Reg(c)
			a.PXOR(r, r, r)
		}
	}
	nv := hd / 4
	for i := 0; i < nv; i++ {
		kv.load4(a, k, RDX, i*4)
		for _, h := range hs {
			// The query is L1-hot (every position re-reads it) and possibly
			// misaligned, so it is MOVUPS'd rather than used as an m128 operand.
			a.MOVUPSLoad(t, At(h.q, h.qOff+int32(i)*16))
			a.MULPS(t, t, k)
			acc := h.acc + Reg(i%chains)
			a.ADDPS(acc, acc, t)
		}
	}
	for d := nv * 4; d < hd; d++ {
		kv.load1(a, k, RDX, d)
		for _, h := range hs {
			a.MOVSSLoad(t, At(h.q, h.qOff+int32(4*d)))
			a.MULSS(t, t, k)
			a.ADDSS(h.acc, h.acc, t)
		}
	}
	for _, h := range hs {
		c0 := h.acc
		if chains == 4 {
			a.ADDPS(c0, c0, c0+1)
			a.ADDPS(c0+2, c0+2, c0+3)
			a.ADDPS(c0, c0, c0+2)
		}
		hsum4SSE(a, c0)
		a.MOVSSStore(At(h.out, h.outOff), c0)
	}
}

// EmitAttnScoresSSE is EmitAttnScores on the SSE tier: scores[t] = dot(q, K[t])
// for t in [0, Rows). It reads Out, W, Rows and Q32.
//
// Registers: XMM0-3 the four chains, XMM4 the K vector, XMM5 the product, and
// for an f16 cache XMM6-9 the widening.
func EmitAttnScoresSSE(hd, kvStride int, fm KVFmt) ([]byte, error) {
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScoresSSE: hd=%d must be positive", hd)
	}
	ka, err := newKVAddr(fm, hd, kvStride)
	if err != nil {
		return nil, err
	}
	kv := attnKVSSE{a: ka, src: XMM6, t0: XMM7, t1: XMM8, magic: XMM9}
	var a Buf
	a.DeclareISA(ISATierSSE)
	kv.setup(&a, RAX)
	a.MOVLoad(RCX, At(RDI, 0))   // Out: scores
	a.MOVLoad(RDX, At(RDI, 8))   // W:   K cache row 0
	a.MOVLoad(RAX, At(RDI, 32))  // Rows: positions
	a.MOVLoad(RSI, At(RDI, 112)) // Q32: query

	done, row := a.Label(), a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)
	a.Bind(row)
	attnScoreRowSSE(&a, kv, hd, 4, []attnScoreHeadSSE{{acc: XMM0, q: RSI, out: RCX}}, XMM4, XMM5)
	a.ADDimm(RCX, 4)
	a.ADDimm(RDX, kv.a.stride(kvStride))
	a.DEC(RAX)
	a.JNZ(row)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}

// EmitAttnScores2SSE is EmitAttnScores2 on the SSE tier: two query heads
// sharing one walk of K, bit-identical to two EmitAttnScoresSSE calls. It
// reads Out, Out2, W, Rows, Q32 and Q2.
//
// Registers: XMM0-3 head 0's chains, XMM4-7 head 1's, XMM8 the shared K
// vector, XMM9 the product, and for an f16 cache XMM10-13 the widening --
// fourteen of sixteen.
func EmitAttnScores2SSE(hd, kvStride int, fm KVFmt) ([]byte, error) {
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScores2SSE: hd=%d must be positive", hd)
	}
	ka, err := newKVAddr(fm, hd, kvStride)
	if err != nil {
		return nil, err
	}
	kv := attnKVSSE{a: ka, src: XMM10, t0: XMM11, t1: XMM12, magic: XMM13}
	var a Buf
	a.DeclareISA(ISATierSSE)
	kv.setup(&a, RAX)
	a.MOVLoad(RCX, At(RDI, 0))   // Out:  scores, head 0
	a.MOVLoad(RDX, At(RDI, 8))   // W:    K cache row 0
	a.MOVLoad(RAX, At(RDI, 32))  // Rows: positions
	a.MOVLoad(RSI, At(RDI, 112)) // Q32:  query 0
	a.MOVLoad(R8, At(RDI, 120))  // Out2: scores, head 1
	a.MOVLoad(R9, At(RDI, 128))  // Q2:   query 1

	done, row := a.Label(), a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)
	a.Bind(row)
	// One K load, two products, through the single-head kernel's own body.
	attnScoreRowSSE(&a, kv, hd, 4, []attnScoreHeadSSE{
		{acc: XMM0, q: RSI, out: RCX},
		{acc: XMM4, q: R9, out: R8},
	}, XMM8, XMM9)
	a.ADDimm(RCX, 4)
	a.ADDimm(R8, 4)
	a.ADDimm(RDX, kv.a.stride(kvStride))
	a.DEC(RAX)
	a.JNZ(row)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}

// EmitAttnScoresTiledSSE is EmitAttnScoresTiled on the SSE tier: qt queries
// share one pass over K, one accumulator chain each. Out row j is at
// j*scoreStride and query j at j*qStride (elements, baked). It reads Out, W,
// Rows and Q32.
//
// One chain per query makes each score oracle.Dot32 exactly: four lanes
// striding the head, the tail into lane 0, and (l0+l1)+(l2+l3) at the end.
// The qt independent chains cover the ADDPS latency.
//
// Registers: XMM0..qt-1 the chains, then K and the product (f32: qt+2 <= 16,
// so qt <= 14 as on AVX2), and for an f16 cache three more for the widening
// with the product doubling as its first temporary (qt+5 <= 16, qt <= 11).
func EmitAttnScoresTiledSSE(hd, kvStride, qStride, scoreStride, qt int, fm KVFmt) ([]byte, error) {
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScoresTiledSSE: hd=%d must be positive", hd)
	}
	ka, err := newKVAddr(fm, hd, kvStride)
	if err != nil {
		return nil, err
	}
	maxQt := 14
	if fm != KVF32 {
		maxQt = 11
	}
	if qt < 1 || qt > maxQt {
		need := "qt chains, K and a product"
		if fm != KVF32 {
			need += ", and three registers for the f16 widening"
		}
		return nil, fmt.Errorf("jit: EmitAttnScoresTiledSSE: qt=%d (%v) does not fit the "+
			"16 XMM registers (%s): at most %d", qt, fm, need, maxQt)
	}
	k, t := Reg(qt), Reg(qt+1)
	kv := attnKVSSE{a: ka}
	if fm != KVF32 {
		kv.t0, kv.src, kv.t1, kv.magic = t, Reg(qt+2), Reg(qt+3), Reg(qt+4)
	}
	hs := make([]attnScoreHeadSSE, qt)
	for j := range hs {
		hs[j] = attnScoreHeadSSE{acc: Reg(j), q: RSI, qOff: int32(j*qStride) * 4,
			out: RCX, outOff: int32(j*scoreStride) * 4}
	}

	var a Buf
	a.DeclareISA(ISATierSSE)
	kv.setup(&a, RAX)
	a.MOVLoad(RCX, At(RDI, 0))   // Out:  scores, row j at j*scoreStride
	a.MOVLoad(RDX, At(RDI, 8))   // W:    K cache row 0
	a.MOVLoad(RAX, At(RDI, 32))  // Rows: positions
	a.MOVLoad(RSI, At(RDI, 112)) // Q32:  query 0, row j at j*qStride

	done, row := a.Label(), a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)
	a.Bind(row)
	attnScoreRowSSE(&a, kv, hd, 1, hs, k, t)
	a.ADDimm(RCX, 4)
	a.ADDimm(RDX, kv.a.stride(kvStride))
	a.DEC(RAX)
	a.JNZ(row)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}

// attnWalkSSE emits the position walk of an accumulate kernel around body:
// RSI walks the V cache from RDX by stride bytes, each weight cursor w[i][0]
// walks its row from w[i][1] by 4 bytes, and R11 counts Rows (RAX) down. An
// empty window runs body zero times, so a kernel called with Rows == 0 stores
// its seed (zero, or Out itself for Into) rather than looping on a wrapped
// counter.
func attnWalkSSE(a *Buf, stride int32, w [][2]Reg, body func()) {
	a.MOVQ(RSI, RDX)
	for _, c := range w {
		a.MOVQ(c[0], c[1])
	}
	a.MOVQ(R11, RAX)
	skip, pos := a.Label(), a.Label()
	a.TESTQ(R11, R11)
	a.JZ(skip)
	a.Bind(pos)
	body()
	a.ADDimm(RSI, stride)
	for _, c := range w {
		a.ADDimm(c[0], 4)
	}
	a.DEC(R11)
	a.JNZ(pos)
	a.Bind(skip)
}

// EmitAttnAccSSE is EmitAttnAcc on the SSE tier: out[i] = sum over t of
// att[t] * V[t][i]. It reads Out, W, AScale and Rows.
func EmitAttnAccSSE(hd, kvStride int, fm KVFmt) ([]byte, error) {
	return emitAttnAccSSE(hd, kvStride, fm, false)
}

// EmitAttnAccIntoSSE is EmitAttnAccInto on the SSE tier: it ADDS INTO Out, so
// a window split across KV pages sums to exactly what one call gives (the
// file comment's second bit-identity).
func EmitAttnAccIntoSSE(hd, kvStride int, fm KVFmt) ([]byte, error) {
	return emitAttnAccSSE(hd, kvStride, fm, true)
}

// emitAttnAccSSE holds the output in registers across the position walk, in
// column blocks, as attn.go does -- the alternative is a dependent round trip
// to memory on every accumulation.
//
// Fourteen vectors a block rather than AVX2's eight: with no FMA a step needs
// only the V vector (the product, in place) and the weight broadcast, so
// fourteen accumulators fit, 56 floats. An f16 cache spends four more on the
// widening: ten a block.
//
// For every output dimension the per-position step is out = out + (V * w),
// MULPS then ADDPS -- oracle.AxpyF32's arithmetic, in its order.
func emitAttnAccSSE(hd, kvStride int, fm KVFmt, into bool) ([]byte, error) {
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnAccSSE: hd=%d must be positive", hd)
	}
	ka, err := newKVAddr(fm, hd, kvStride)
	if err != nil {
		return nil, err
	}
	perBlock := 14
	vreg, wreg := XMM14, XMM15
	kv := attnKVSSE{a: ka}
	if fm != KVF32 {
		perBlock = 10
		wreg, vreg = XMM10, XMM11
		kv.src, kv.t0, kv.t1, kv.magic = XMM12, XMM13, XMM14, XMM15
	}
	stride := kv.a.stride(kvStride)
	cursors := [][2]Reg{{R9, R8}}

	var a Buf
	a.DeclareISA(ISATierSSE)
	kv.setup(&a, RAX)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W: V cache
	a.MOVLoad(R8, At(RDI, 24))  // AScale: attention weights
	a.MOVLoad(RAX, At(RDI, 32)) // Rows

	nv := hd / 4
	for base := 0; base < nv; base += perBlock {
		n := min(perBlock, nv-base)
		for v := 0; v < n; v++ {
			r := Reg(v)
			if into {
				a.MOVUPSLoad(r, At(RCX, int32(base+v)*16))
			} else {
				a.PXOR(r, r, r)
			}
		}
		attnWalkSSE(&a, stride, cursors, func() {
			bcastSS(&a, wreg, At(R9, 0))
			for v := 0; v < n; v++ {
				kv.load4(&a, vreg, RSI, (base+v)*4)
				a.MULPS(vreg, vreg, wreg)
				a.ADDPS(Reg(v), Reg(v), vreg)
			}
		})
		for v := 0; v < n; v++ {
			a.MOVUPSStore(At(RCX, int32(base+v)*16), Reg(v))
		}
	}
	// The last hd%4 dimensions: one lane-0 accumulator each, walking the
	// positions in the same order -- so a paged window still sums to one
	// contiguous call's bits.
	if tail := hd % 4; tail > 0 {
		d0 := int32(nv * 4)
		for j := 0; j < tail; j++ {
			r := Reg(j)
			if into {
				a.MOVSSLoad(r, At(RCX, 4*(d0+int32(j))))
			} else {
				a.PXOR(r, r, r)
			}
		}
		attnWalkSSE(&a, stride, cursors, func() {
			a.MOVSSLoad(wreg, At(R9, 0))
			for j := 0; j < tail; j++ {
				kv.load1(&a, vreg, RSI, int(d0)+j)
				a.MULSS(vreg, vreg, wreg)
				a.ADDSS(Reg(j), Reg(j), vreg)
			}
		})
		for j := 0; j < tail; j++ {
			a.MOVSSStore(At(RCX, 4*(d0+int32(j))), Reg(j))
		}
	}
	a.RET()
	return a.Bytes(), nil
}

// EmitAttnAcc2SSE is EmitAttnAcc2 on the SSE tier: two heads sharing one kv
// head, each V vector loaded once and weighted into both outputs,
// bit-identical to two EmitAttnAccSSE calls. It reads Out, Out2, W, AScale,
// AScale2 and Rows.
func EmitAttnAcc2SSE(hd, kvStride int, fm KVFmt) ([]byte, error) {
	return emitAttnAcc2SSE(hd, kvStride, fm, false)
}

// EmitAttnAcc2IntoSSE is EmitAttnAcc2SSE that ADDS INTO both outputs, so a
// paired window split across KV pages sums to what one call would have given.
func EmitAttnAcc2IntoSSE(hd, kvStride int, fm KVFmt) ([]byte, error) {
	return emitAttnAcc2SSE(hd, kvStride, fm, true)
}

// emitAttnAcc2SSE is emitAttnAccSSE for two heads.
//
// Six vectors per head a block: V is shared, so head 0's product needs its
// own temporary, leaving twelve accumulators. An f16 cache spends four more on
// the widening: four per head.
//
// Head 0's product is (V copied) * w0 and head 1's is V * w1, the single
// kernel's V * w for each, so the two outputs are two single calls' bits.
func emitAttnAcc2SSE(hd, kvStride int, fm KVFmt, into bool) ([]byte, error) {
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnAcc2SSE: hd=%d must be positive", hd)
	}
	ka, err := newKVAddr(fm, hd, kvStride)
	if err != nil {
		return nil, err
	}
	per := 6
	w0, w1, vreg, prod := XMM12, XMM13, XMM14, XMM15
	kv := attnKVSSE{a: ka}
	if fm != KVF32 {
		per = 4
		w0, w1, vreg, prod = XMM8, XMM9, XMM10, XMM15
		kv.src, kv.t0, kv.t1, kv.magic = XMM11, XMM12, XMM13, XMM14
	}
	h0 := func(i int) Reg { return Reg(i) }       // head 0: XMM0..per-1
	h1 := func(i int) Reg { return Reg(per + i) } // head 1: XMMper..2per-1
	stride := kv.a.stride(kvStride)
	cursors := [][2]Reg{{R9, R8}, {R13, R12}}

	var a Buf
	a.DeclareISA(ISATierSSE)
	kv.setup(&a, RAX)
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RDX, At(RDI, 8))   // W: V cache
	a.MOVLoad(R8, At(RDI, 24))   // AScale
	a.MOVLoad(RAX, At(RDI, 32))  // Rows
	a.MOVLoad(R10, At(RDI, 120)) // Out2
	a.MOVLoad(R12, At(RDI, 136)) // AScale2

	nv := hd / 4
	for base := 0; base < nv; base += per {
		n := min(per, nv-base)
		for v := 0; v < n; v++ {
			off := int32(base+v) * 16
			if into {
				a.MOVUPSLoad(h0(v), At(RCX, off))
				a.MOVUPSLoad(h1(v), At(R10, off))
			} else {
				a.PXOR(h0(v), h0(v), h0(v))
				a.PXOR(h1(v), h1(v), h1(v))
			}
		}
		attnWalkSSE(&a, stride, cursors, func() {
			bcastSS(&a, w0, At(R9, 0))
			bcastSS(&a, w1, At(R13, 0))
			for v := 0; v < n; v++ {
				// One load, two products, as in the paired scores.
				kv.load4(&a, vreg, RSI, (base+v)*4)
				a.MULPS(prod, vreg, w0)
				a.ADDPS(h0(v), h0(v), prod)
				a.MULPS(vreg, vreg, w1)
				a.ADDPS(h1(v), h1(v), vreg)
			}
		})
		for v := 0; v < n; v++ {
			off := int32(base+v) * 16
			a.MOVUPSStore(At(RCX, off), h0(v))
			a.MOVUPSStore(At(R10, off), h1(v))
		}
	}
	// The last hd%4 dimensions (at most three, so both heads' lane-0
	// accumulators fit in one walk), in the single kernel's order.
	if tail := hd % 4; tail > 0 {
		d0 := int32(nv * 4)
		for j := 0; j < tail; j++ {
			off := 4 * (d0 + int32(j))
			if into {
				a.MOVSSLoad(h0(j), At(RCX, off))
				a.MOVSSLoad(h1(j), At(R10, off))
			} else {
				a.PXOR(h0(j), h0(j), h0(j))
				a.PXOR(h1(j), h1(j), h1(j))
			}
		}
		attnWalkSSE(&a, stride, cursors, func() {
			a.MOVSSLoad(w0, At(R9, 0))
			a.MOVSSLoad(w1, At(R13, 0))
			for j := 0; j < tail; j++ {
				kv.load1(&a, vreg, RSI, int(d0)+j)
				a.MULSS(prod, vreg, w0)
				a.ADDSS(h0(j), h0(j), prod)
				a.MULSS(vreg, vreg, w1)
				a.ADDSS(h1(j), h1(j), vreg)
			}
		})
		for j := 0; j < tail; j++ {
			off := 4 * (d0 + int32(j))
			a.MOVSSStore(At(RCX, off), h0(j))
			a.MOVSSStore(At(R10, off), h1(j))
		}
	}
	a.RET()
	return a.Bytes(), nil
}

// EmitKVWidenSSE is EmitKVWiden on the SSE tier: Rows q8_0 head rows widened
// to float32 through attnKVSSE's own load. It reads Out, W and Rows.
func EmitKVWidenSSE(hd int) ([]byte, error) {
	ka, err := newKVAddr(KVQ8, hd, hd)
	if err != nil {
		return nil, err
	}
	kv := attnKVSSE{a: ka, t0: XMM1}
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out: f32 rows
	a.MOVLoad(RDX, At(RDI, 8))  // W:   q8 rows
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	done, row := a.Label(), a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)
	a.Bind(row)
	for i := 0; i < hd/4; i++ {
		kv.load4(&a, XMM0, RDX, i*4)
		a.MOVUPSStore(At(RCX, int32(i*16)), XMM0)
	}
	for d := hd / 4 * 4; d < hd; d++ {
		kv.load1(&a, XMM0, RDX, d)
		a.MOVSSStore(At(RCX, int32(4*d)), XMM0)
	}
	a.ADDimm(RCX, int32(4*hd))
	a.ADDimm(RDX, ka.stride(hd))
	a.DEC(RAX)
	a.JNZ(row)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
