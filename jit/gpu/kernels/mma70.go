package kernels

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// voltaLane is where lane l sits in ir.MMAVolta: its quad-pair q, the A row /
// B column r it holds, and the D row and column of accumulator i (see
// ir.MMAVolta); every kernel on this shape derives its indices here.
type voltaLane struct {
	b    *ir.Builder
	lane ir.Value
}

// quad is the lane's quad-pair, 0..3.
func (v voltaLane) quad() ir.Value {
	b := v.b
	return b.And(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 2)), b.Const(ir.U32, 3))
}

// row is the A row (and B column) the lane holds, 0..7.
func (v voltaLane) row() ir.Value {
	b := v.b
	return b.Add(ir.U32, b.And(ir.U32, v.lane, b.Const(ir.U32, 3)),
		b.Shl(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 4)), b.Const(ir.U32, 2)))
}

// dRow is the D row of accumulator i: (l&1) + 2*((i>>1)&1) + 4*(l>>4).
func (v voltaLane) dRow(i int) ir.Value {
	b := v.b
	r := b.Add(ir.U32, b.And(ir.U32, v.lane, b.Const(ir.U32, 1)),
		b.Shl(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 4)), b.Const(ir.U32, 2)))
	if o := 2 * ((i >> 1) & 1); o != 0 {
		r = b.Add(ir.U32, r, b.Const(ir.U32, int64(o)))
	}
	return r
}

// dCol is the D column of accumulator i: (i&1) + 2*((l>>1)&1) + 4*(i>>2).
func (v voltaLane) dCol(i int) ir.Value {
	b := v.b
	c := b.Shl(ir.U32, b.And(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 1)), b.Const(ir.U32, 1)),
		b.Const(ir.U32, 1))
	if o := (i & 1) + 4*(i>>2); o != 0 {
		c = b.Add(ir.U32, c, b.Const(ir.U32, int64(o)))
	}
	return c
}

// MMAVoltaProbe multiplies, per quad-pair q, A rows 8q..8q+7 (row-major, 32x4
// float) by B columns 8q..8q+7 (B is 4x32, row-major float) and stores the
// 8x8 product at out[(8q+row)*8 + col]. It exists to hold the lowering and the
// lane layout in voltaLane to a CPU reference.
func MMAVoltaProbe() *ir.Kernel {
	b := ir.New("mmavoltaprobe", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	v := voltaLane{b, b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))}
	q8 := b.Shl(ir.U32, v.quad(), b.Const(ir.U32, 3))
	r := b.Add(ir.U32, q8, v.row())
	aIdx := b.Shl(ir.U32, r, b.Const(ir.U32, 2))
	a := []ir.Value{
		b.PackF16(b.Load(ir.F32, pA, aIdx, 0), b.Load(ir.F32, pA, aIdx, 1)),
		b.PackF16(b.Load(ir.F32, pA, aIdx, 2), b.Load(ir.F32, pA, aIdx, 3)),
	}
	bb := []ir.Value{
		b.PackF16(b.Load(ir.F32, pB, r, 0), b.Load(ir.F32, pB, r, 32)),
		b.PackF16(b.Load(ir.F32, pB, r, 64), b.Load(ir.F32, pB, r, 96)),
	}
	zero := b.ConstF32(0)
	c := make([]ir.Value, 8)
	for i := range c {
		c[i] = zero
	}
	d := b.MMA(ir.MMAVolta(), a, bb, c)
	for i, x := range d {
		row := b.Add(ir.U32, q8, v.dRow(i))
		b.Store(pOut, b.Add(ir.U32, b.Shl(ir.U32, row, b.Const(ir.U32, 3)), v.dCol(i)), x, 0)
	}
	return b.Done()
}

// Volta70OK reports whether MatVecMMA70 serves a format: 4-bit payloads (with or
// without a secondary plane), 8-bit Q8_0 and MXFP4's e2m1 code, no float.
//
// MXFP4 needs no table on the f16 path: an e2m1 code placed at bits 9-11 of
// a binary16 (sign at 15) is its value times 2^-14, subnormals included, so
// the dequant is a mask, two shifts and two f16 multiplies (voltaDeq.frags).
//
// A format whose arithmetic is over int8 activations (qinfo.intAct: TQ2) is
// declined: these kernels multiply binary16 activations, which for a ternary
// model is a different function.
func Volta70OK(q Quant) bool {
	return volta70Packed(q) && !qtab[q].intAct
}

// GemmVoltaOK reports whether GemmVolta (and the ActF16T layout it reads)
// takes q: every format Volta70OK takes, and the two-byte float weights,
// binary16 and bfloat16. A float weight is staged as binary16 -- a bfloat16
// rounded to binary16's ten-bit mantissa from its seven, so exactly unless
// it lies outside binary16's range (above 65504 it is infinite, below 2^-14
// it loses bits as a subnormal), which no projection of a trained model
// reaches -- and its sub-block takes the 8-bit formats' step order.
func GemmVoltaOK(q Quant) bool {
	return Volta70OK(q) || int(q) < len(qtab) && qtab[q].float == 2
}

// byteOrder says a sub-block's m8n8k4 steps take consecutive elements
// (voltaSteps): an 8-bit format's, and a two-byte float's.
func (qi qinfo) byteOrder() bool { return qi.bits == 8 || qi.float == 2 }

func volta70Packed(q Quant) bool {
	if int(q) >= len(qtab) {
		return false
	}
	qi := qtab[q]
	return qi.blockE != 0 && qi.float == 0 && (qi.codes == nil || qi.e8m0) &&
		(qi.bits == 4 || (qi.bits == 8 && qi.hi == 0)) && DeviceWhyNot(q) == ""
}

// voltaSteps is how many m8n8k4 steps one sub-block takes, and the element each
// lane-slot of a step holds -- the permutation ActF16 writes the activations
// in so the B fragment matches the A fragment's unpack order.
//
// A 4-bit word w of a sub-block holds elements 4w+j in byte j's low nibble and
// h+4w+j in its high one (h = sub/2, packSub). Or-ing a nibble pair into the
// bits of 1024.0 gives two f16 at once, and the pairs a word yields are bytes
// (0,2) and (1,3) of each nibble half -- so step 2w holds elements
// {4w, 4w+2, 4w+1, 4w+3} and step 2w+1 the same h further on. An 8-bit word is
// one step: {4w, 4w+2, 4w+1, 4w+3}.
func voltaSteps(qi qinfo) (steps int, elem func(step, slot int) int) {
	order := [4]int{0, 2, 1, 3}
	if qi.byteOrder() {
		return qi.sub / 4, func(s, j int) int { return 4*s + order[j] }
	}
	h := qi.sub / 2
	return qi.sub / 4, func(s, j int) int { return (s/2)*4 + order[j] + (s%2)*h }
}

// ActF16 writes the float activations of ntok tokens of width k (pX, token-
// major f32) as binary16, in the order MatVecMMA70 reads them: for global step
// s and token t, two u32 (four halves) at ((s*ntok + t)*2). One thread per
// (step, token).
func ActF16(q Quant, ntok, k int) (*ir.Kernel, error) {
	if !Volta70OK(q) {
		return nil, fmt.Errorf("kernels: ActF16: %v has no Volta layout", q)
	}
	qi := qtab[q]
	if k%qi.sub != 0 {
		return nil, fmt.Errorf("kernels: ActF16: k=%d is not a multiple of %d", k, qi.sub)
	}
	steps, elem := voltaSteps(qi)
	nsteps := k / qi.sub * steps
	b := ir.New("actf16", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pB := b.Param("pB", ir.U32)
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	tid = b.Min(ir.U32, tid, b.Const(ir.U32, int64(nsteps*ntok-1)))
	s := b.Div(ir.U32, tid, b.Const(ir.U32, int64(ntok)))
	t := b.Rem(ir.U32, tid, b.Const(ir.U32, int64(ntok)))
	sub := b.Div(ir.U32, s, b.Const(ir.U32, int64(steps)))
	// The element of slot j is sub*qi.sub + elem(ls, j), and ls is a runtime
	// value, so the slots come out of a select chain over a sub-block's steps.
	ls := b.Rem(ir.U32, s, b.Const(ir.U32, int64(steps)))
	base := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(k))),
		b.Mul(ir.U32, sub, b.Const(ir.U32, int64(qi.sub))))
	val := func(j int) ir.Value {
		e := b.Const(ir.U32, int64(elem(0, j)))
		for x := 1; x < steps; x++ {
			hit := b.Lt(ir.U32, b.Const(ir.U32, int64(x-1)), ls) // ls >= x
			e = b.Select(ir.U32, hit, b.Const(ir.U32, int64(elem(x, j))), e)
		}
		return b.Load(ir.F32, pX, b.Add(ir.U32, base, e), 0)
	}
	out := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, s, b.Const(ir.U32, int64(ntok))), t), b.Const(ir.U32, 2))
	b.Store(pB, out, b.PackF16(val(0), val(1)), 0)
	b.Store(pB, out, b.PackF16(val(2), val(3)), 1)
	return b.Done(), nil
}

// ActF16T is ActF16 laid out TOKEN-major, for GemmVolta: token t's steps in
// order, two u32 each, at t*k/2 + 2*s. One thread per (token, sub-block),
// which reads the sub-block's floats in 16-byte vectors and writes its
// permuted halves the same way, so the permutation is fixed at emit time and
// needs none of ActF16's select chain.
func ActF16T(q Quant, ntok, k int) (*ir.Kernel, error) {
	return actF16T(q, ntok, k)
}

// ActMulF16T is ActMul followed by ActF16T in one launch: each element is
// act(g) times u as ActMul computes it, packed straight into ActF16T's
// token-major layout for q, so the gated FFN's float row is neither written
// nor read again when the down projection is a binary16 GEMM. The same bits
// as the two kernels. Parameters pG, pU, pB; the same launch as ActF16T.
//
// One thread an output word: two elements of one (token, sub-block), so a
// warp reads and writes contiguous runs and every element is one thread's,
// where ActF16T's thread a sub-block left an activation's exponentials 16 or
// 32 deep on one thread. Launch ActMulF16TThreads(ntok, k) threads.
func ActMulF16T(q Quant, ntok, k int, kind ActKind) (*ir.Kernel, error) {
	if err := actMulKind(kind); err != nil {
		return nil, err
	}
	if !GemmVoltaOK(q) {
		return nil, fmt.Errorf("kernels: ActMulF16T: %v has no Volta layout", q)
	}
	qi := qtab[q]
	if k%qi.sub != 0 || qi.sub%8 != 0 {
		return nil, fmt.Errorf("kernels: ActMulF16T: k=%d is not a multiple of %d", k, qi.sub)
	}
	nw := k / 2
	b := ir.New("actmulf16t", [3]int{128, 1, 1})
	pG, pU := b.Param("pG", ir.F32), b.Param("pU", ir.F32)
	pB := b.Param("pB", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	wd := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), c(int64(ntok*nw-1)))
	t, wt := b.Div(ir.U32, wd, c(int64(nw))), b.Rem(ir.U32, wd, c(int64(nw)))
	e0, e1 := f16WordElems(b, qi, wt)
	base := b.Mul(ir.U32, t, c(int64(k)))
	val := func(e ir.Value) ir.Value {
		at := b.Add(ir.U32, base, e)
		return actMulValue(b, kind, b.Load(ir.F32, pG, at, 0), func() ir.Value { return b.Load(ir.F32, pU, at, 0) })
	}
	b.Store(pB, wd, b.PackF16(val(e0), val(e1)), 0)
	return b.Done(), nil
}

// ActMulF16TThreads is ActMulF16T's launch: one thread an output word.
func ActMulF16TThreads(ntok, k int) int { return ntok * k / 2 }

// f16WordElems is which two elements of a row, in natural order, output word
// wt (a row's word index) of ActF16T's layout packs: word 2s+h of a
// sub-block is step s's slots 2h and 2h+1 (voltaSteps), its sub-block's
// elements base+h and base+h+2 with base the step's first. Arithmetic on a
// runtime word, where voltaSteps' elem is a table over compile-time steps.
func f16WordElems(b *ir.Builder, qi qinfo, wt ir.Value) (e0, e1 ir.Value) {
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	half := qi.sub / 2
	sb := b.Div(ir.U32, wt, c(int64(half)))
	w := b.Rem(ir.U32, wt, c(int64(half)))
	s, h := b.Shr(ir.U32, w, c(1)), b.And(ir.U32, w, c(1))
	var base ir.Value
	if qi.byteOrder() {
		base = b.Shl(ir.U32, s, c(2))
	} else {
		// (s/2)*4 + (s%2)*sub/2
		base = b.Add(ir.U32, b.Shl(ir.U32, b.Shr(ir.U32, s, c(1)), c(2)), b.Mul(ir.U32, b.And(ir.U32, s, c(1)), c(int64(qi.sub/2))))
	}
	e0 = b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, sb, c(int64(qi.sub))), base), h)
	return e0, b.Add(ir.U32, e0, c(2))
}

func actF16T(q Quant, ntok, k int) (*ir.Kernel, error) {
	if !GemmVoltaOK(q) {
		return nil, fmt.Errorf("kernels: ActF16T: %v has no Volta layout", q)
	}
	qi := qtab[q]
	if k%qi.sub != 0 || qi.sub%8 != 0 {
		return nil, fmt.Errorf("kernels: ActF16T: k=%d is not a multiple of %d", k, qi.sub)
	}
	nsub := k / qi.sub
	b := ir.New("actf16t", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pB := b.Param("pB", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	tid = b.Min(ir.U32, tid, c(int64(nsub*ntok-1)))
	// tid is t*nsub + sub, so the float index of the sub-block is tid*sub and
	// its f16 output index tid*sub/2: consecutive threads, consecutive bytes.
	x := make([]ir.Value, 0, qi.sub)
	in := b.Mul(ir.U32, tid, c(int64(qi.sub)))
	for o := 0; o < qi.sub; o += 4 {
		x = append(x, b.LoadV(ir.F32, pX, in, int64(o), 4)...)
	}
	storeF16T(b, qi, pB, tid, x)
	return b.Done(), nil
}

// storeF16T writes one (token, sub-block)'s values x, in natural order, as
// ActF16T's permuted binary16 words at index tid (t*nsub + sub).
func storeF16T(b *ir.Builder, qi qinfo, pB, tid ir.Value, x []ir.Value) {
	steps, elem := voltaSteps(qi)
	out := b.Mul(ir.U32, tid, b.Const(ir.U32, int64(qi.sub/2)))
	w := make([]ir.Value, 0, 2*steps)
	for s := 0; s < steps; s++ {
		w = append(w, b.PackF16(x[elem(s, 0)], x[elem(s, 1)]), b.PackF16(x[elem(s, 2)], x[elem(s, 3)]))
	}
	for o := 0; o < len(w); o += 4 {
		b.StoreV(pB, out, int64(o), w[o:o+4]...)
	}
}

// ActF16TGather is ActF16T of GatherRows: token t of the output reads source
// row perm[t] of nsrc rows of width k, and a zero row where perm[t] >= nsrc --
// the padding column of a grouped mixture's sorted run. Parameters pX (the
// nsrc source rows), pPerm (ntok u32), pB; one thread per (token, sub-block).
//
// The gather and the conversion are one pass, so the grouped mixture no
// longer writes and re-reads an np x k float32 copy of the gathered rows.
func ActF16TGather(q Quant, ntok, k, nsrc int) (*ir.Kernel, error) {
	if !Volta70OK(q) {
		return nil, fmt.Errorf("kernels: ActF16TGather: %v has no Volta layout", q)
	}
	qi := qtab[q]
	if k%qi.sub != 0 || qi.sub%8 != 0 || nsrc < 1 {
		return nil, fmt.Errorf("kernels: ActF16TGather: k=%d (a multiple of %d), nsrc=%d", k, qi.sub, nsrc)
	}
	steps, elem := voltaSteps(qi)
	nsub := k / qi.sub
	b := ir.New("actf16tgather", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pPerm := b.Param("pPerm", ir.U32)
	pB := b.Param("pB", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	tid = b.Min(ir.U32, tid, c(int64(nsub*ntok-1)))
	t, sb := b.Div(ir.U32, tid, c(int64(nsub))), b.Rem(ir.U32, tid, c(int64(nsub)))
	src := b.Load(ir.U32, pPerm, t, 0)
	last := c(int64(nsrc - 1))
	pad := b.Lt(ir.U32, last, src)
	in := b.Add(ir.U32, b.Mul(ir.U32, b.Min(ir.U32, src, last), c(int64(k))), b.Mul(ir.U32, sb, c(int64(qi.sub))))
	zero := b.ConstF32(0)
	x := make([]ir.Value, 0, qi.sub)
	for o := 0; o < qi.sub; o += 4 {
		for _, v := range b.LoadV(ir.F32, pX, in, int64(o), 4) {
			x = append(x, b.Select(ir.F32, pad, zero, v))
		}
	}
	out := b.Mul(ir.U32, tid, c(int64(qi.sub/2)))
	w := make([]ir.Value, 0, 2*steps)
	for s := 0; s < steps; s++ {
		w = append(w, b.PackF16(x[elem(s, 0)], x[elem(s, 1)]), b.PackF16(x[elem(s, 2)], x[elem(s, 3)]))
	}
	for o := 0; o < len(w); o += 4 {
		b.StoreV(pB, out, int64(o), w[o:o+4]...)
	}
	return b.Done(), nil
}

// MatVecMMA70 is the batched quantized matvec on sm_70's f16 tensor cores
// (ir.MMAVolta), for a card with no integer matrix instruction.
//
// The weight is dequantized in registers, two halves at a time: a nibble
// pair Or-ed into the bits of 1024.0 is 1024+q in two integer ops, less 1024
// it is q exactly, and one fused f16 multiply-add applies the sub-block's
// scale and minimum. The tensor core then multiplies f16 weights by f16
// activations (ActF16) into f32, and the only epilogue is the store. An
// exact-integer form was issue-bound on per-sub-block scale epilogues;
// llama.cpp's Volta prefill also multiplies f16-dequantized weights.
//
// Tile: a warp is four quad-pairs; quad-pair p holds rows 32*i + 8*p .. +7 of
// m-tile i and tokens 8*j .. +7 of n-tile j, so a warp covers 32*MT rows by
// 8*NT tokens. Parameters: pQS, pD, pSC, pB (ActF16's output), pOut, then pBias.
func MatVecMMA70(s MatVecShape) (*ir.Kernel, error) {
	if !Volta70OK(s.T) {
		return nil, fmt.Errorf("kernels: MatVecMMA70: %v has no Volta layout", s.T)
	}
	qi := qtab[s.T]
	if s.Experts > 1 {
		return nil, fmt.Errorf("kernels: MatVecMMA70: experts are not supported")
	}
	split := max(s.Split, 1)
	mt, nt := max(s.MT, 1), max(s.NT, 1)
	if s.Rows%(32*mt) != 0 || s.NTok%(8*nt) != 0 {
		return nil, fmt.Errorf("kernels: MatVecMMA70: %dx%d tile does not divide rows=%d ntok=%d",
			32*mt, 8*nt, s.Rows, s.NTok)
	}
	if s.K%qi.blockE != 0 || s.K <= 0 {
		return nil, fmt.Errorf("kernels: MatVecMMA70: K=%d is not a multiple of %d", s.K, qi.blockE)
	}
	pw, hw := qi.sub*qi.bits/32, qi.sub*qi.hi/32
	words := pw + hw
	steps, _ := voltaSteps(qi)
	nsub := s.K / qi.sub
	if nsub%split != 0 {
		return nil, fmt.Errorf("kernels: MatVecMMA70: split %d does not divide %d sub-blocks", split, nsub)
	}
	per := nsub / split // sub-blocks a warp walks

	b := ir.New(fmt.Sprintf("matvecmma70_%s_%dx%d_t%d_m%dn%d_s%d", strings.ToLower(s.T.String()),
		s.Rows, s.K, s.NTok, mt, nt, split), [3]int{128, 1, 1})
	pQS := b.Param("pQS", ir.U32)
	pD := b.Param("pD", ir.U32)
	pSC := b.Param("pSC", ir.U32)
	pB := b.Param("pB", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	var pBias ir.Value
	if s.Bias {
		pBias = b.Param("pBias", ir.F32)
	}
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	add := func(v ir.Value, n int64) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, c(n))
	}
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	v := voltaLane{b, b.And(ir.U32, tid, c(31))}
	mtiles, ntiles := s.Rows/(32*mt), s.NTok/(8*nt)
	// Split over k where the tile leaves the grid thin (a small matvec leaves
	// too few warps). Segment seg walks sub-blocks [seg*per, seg*per+per) and
	// writes its partial at seg*NTok*Rows; a Reduce sums them. The bias rides
	// segment 0 alone.
	warp := b.Min(ir.U32, b.Shr(ir.U32, tid, c(5)), c(int64(mtiles*ntiles*split-1)))
	seg := b.Div(ir.U32, warp, c(int64(mtiles*ntiles)))
	warp = b.Rem(ir.U32, warp, c(int64(mtiles*ntiles)))
	wm := b.Rem(ir.U32, warp, c(int64(mtiles)))
	wn := b.Div(ir.U32, warp, c(int64(mtiles)))
	q8 := b.Shl(ir.U32, v.quad(), c(3))
	rowBase := b.Add(ir.U32, b.Mul(ir.U32, wm, c(int64(32*mt))), q8)
	tokBase := b.Mul(ir.U32, wn, c(int64(8*nt)))
	nrows := c(int64(s.Rows))
	aRow := b.Add(ir.U32, rowBase, v.row())
	bTok := b.Add(ir.U32, tokBase, v.row())

	dq := newVoltaDeq(b, s.T, pQS, pD, pSC, s.Rows)

	zeroF := b.ConstF32(0)
	one := c(1)
	bStepStride := int64(s.NTok * 2)

	sub0 := b.Mul(ir.U32, seg, c(int64(per)))
	w0 := b.Mul(ir.U32, sub0, c(int64(words)*int64(s.Rows)))
	b0 := b.Mul(ir.U32, sub0, c(int64(steps)*bStepStride))
	b.Loop(int64(per))
	accs := make([][][]ir.Value, mt)
	for i := range accs {
		accs[i] = make([][]ir.Value, nt)
		for j := range accs[i] {
			z := make([]ir.Value, 8)
			for x := range z {
				z[x] = b.Phi(ir.F32, zeroF)
			}
			accs[i][j] = z
		}
	}
	subIdx := b.Phi(ir.U32, sub0)
	W := b.Phi(ir.U32, w0)
	Bc := b.Phi(ir.U32, b0)

	af := make([][][]ir.Value, mt)
	for i := 0; i < mt; i++ {
		s2, m2 := dq.scales(subIdx, add(aRow, int64(32*i)))
		af[i] = dq.frags(dq.words(b.Add(ir.U32, W, aRow), int64(32*i)), s2, m2)
	}
	bIdx := b.Add(ir.U32, Bc, b.Shl(ir.U32, bTok, one))
	d := make([][][]ir.Value, mt)
	for i := range d {
		d[i] = make([][]ir.Value, nt)
		for j := range d[i] {
			d[i][j] = accs[i][j]
		}
	}
	for st := 0; st < steps; st++ {
		bf := make([][]ir.Value, nt)
		for j := 0; j < nt; j++ {
			o := int64(st)*bStepStride + int64(j*16)
			bf[j] = []ir.Value{b.Load(ir.U32, pB, bIdx, o), b.Load(ir.U32, pB, bIdx, o+1)}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				d[i][j] = b.MMA(ir.MMAVolta(), af[i][st], bf[j], d[i][j])
			}
		}
	}
	for i := 0; i < mt; i++ {
		for j := 0; j < nt; j++ {
			for x := 0; x < 8; x++ {
				b.SetPhi(accs[i][j][x], d[i][j][x])
			}
		}
	}
	b.SetPhi(subIdx, b.Add(ir.U32, subIdx, one))
	b.SetPhi(W, b.Add(ir.U32, W, c(int64(words)*int64(s.Rows))))
	b.SetPhi(Bc, b.Add(ir.U32, Bc, c(int64(steps)*bStepStride)))
	b.EndLoop()

	outSeg := b.Mul(ir.U32, seg, c(int64(s.NTok*s.Rows)))
	var first ir.Value
	if s.Bias && split > 1 {
		first = b.Lt(ir.U32, seg, one)
	}
	dRow := []ir.Value{b.Add(ir.U32, rowBase, v.dRow(0)), b.Add(ir.U32, rowBase, v.dRow(2))}
	dCol := []ir.Value{b.Add(ir.U32, tokBase, v.dCol(0)), b.Add(ir.U32, tokBase, v.dCol(1)),
		b.Add(ir.U32, tokBase, v.dCol(4)), b.Add(ir.U32, tokBase, v.dCol(5))}
	for i := 0; i < mt; i++ {
		for j := 0; j < nt; j++ {
			for comp := 0; comp < 8; comp++ {
				rr := add(dRow[(comp>>1)&1], int64(32*i))
				tok := add(dCol[(comp&1)+2*(comp>>2)], int64(8*j))
				val := d[i][j][comp]
				if s.Bias {
					bv := b.Load(ir.F32, pBias, rr, 0)
					if first != 0 {
						bv = b.Select(ir.F32, first, bv, zeroF)
					}
					val = b.Add(ir.F32, val, bv)
				}
				b.Store(pOut, b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, tok, nrows), rr), outSeg), val, 0)
			}
		}
	}
	return b.Done(), nil
}

// voltaDeq is the dequantization MatVecMMA70 and GemmVolta share: one row's
// packed words for one sub-block, and that row's scale and minimum, turned into
// the f16 A-fragment words of every m8n8k4 step of the sub-block.
//
// One transcription for both kernels on purpose: a second copy of this
// arithmetic is how a plane change reaches one reader of the packed layout
// and not the other (see gpu-kernels.md).
type voltaDeq struct {
	b                   *ir.Builder
	qi                  qinfo
	t                   Quant
	pQS, pD, pSC        ir.Value
	rows                int
	wstride             int64 // words between a sub-block's consecutive words
	pw, hw, steps       int
	magic, off2, m4, m8 ir.Value
	zeroF               ir.Value
	// carry selects the SC stream's carried form (SetCarry): scaleWords loads
	// the word after the pair's first and the caller supplies the first.
	carry  bool
	scLast int // the row's last stream word, for carry's clamp
	// qBase, dBase and scBase are an expert sheet's word offsets in the three
	// planes (0 on a dense matrix): a bank is n_expert independent sheets back
	// to back, and the three planes hold different word counts per sheet, so
	// one shared base would read another expert's scales (see MatVec).
	qBase, dBase, scBase ir.Value
}

// expert points the dequant at sheet e of a bank of rows x k sheets.
func (v *voltaDeq) expert(e ir.Value, k int) error {
	nq, nd, nsc, err := PackedWords(v.t, v.rows, k)
	if err != nil {
		return err
	}
	b := v.b
	base := func(n int) ir.Value { return b.Mul(ir.U32, e, b.Const(ir.U32, int64(n))) }
	v.qBase, v.dBase = base(nq), base(nd)
	if nsc > 0 {
		v.scBase = base(nsc)
	}
	return nil
}

// plus adds a sheet base, emitting nothing on a dense matrix so a dense
// kernel's IR is the one it always was.
func (v *voltaDeq) plus(x, base ir.Value) ir.Value {
	if base == 0 {
		return x
	}
	return v.b.Add(ir.U32, x, base)
}

func newVoltaDeq(b *ir.Builder, t Quant, pQS, pD, pSC ir.Value, rows int) *voltaDeq {
	qi := qtab[t]
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	steps, _ := voltaSteps(qi)
	offBits := int64(0x6400) // 1024.0
	if qi.bits == 8 {
		offBits = 0x6480 // 1152.0: the signed byte flipped to unsigned
	}
	pw, wstride := qi.sub*qi.bits/32, int64(rows)
	if qi.float > 0 {
		// A float row is row-major: a sub-block's words are consecutive.
		pw, wstride = qi.sub*qi.float/4, 1
	}
	return &voltaDeq{b: b, qi: qi, t: t, pQS: pQS, pD: pD, pSC: pSC, rows: rows, wstride: wstride,
		pw: pw, hw: qi.sub * qi.hi / 32, steps: steps,
		magic: c(0x64006400), off2: c(offBits | offBits<<16), m4: c(0x000F000F), m8: c(0x00FF00FF),
		zeroF: b.ConstF32(0)}
}

// SetCarry selects the carried form of the SC stream for a k-long row, and
// reports whether it applies: it needs consecutive staged sub-blocks at most
// 32 bits apart in the stream (kb <= 2), so the pair's first word is always
// the previous trip's first or the one after it. See GemmVolta.
func (v *voltaDeq) SetCarry(k, kb int) bool {
	v.carry = ScStream(v.t) && kb*12 <= 32
	v.scLast = ScStreamWords*k/v.qi.blockE - 1
	return v.carry
}

// scWord loads stream word wi of row.
func (v *voltaDeq) scWord(wi, row ir.Value) ir.Value {
	b := v.b
	return b.Load(ir.U32, v.pSC, v.plus(b.Add(ir.U32, b.Mul(ir.U32, wi, b.Const(ir.U32, int64(v.rows))), row), v.scBase), 0)
}

// scFirst is the stream word the pair of subIdx starts in, loaded: the carried
// form's seed.
func (v *voltaDeq) scFirst(subIdx, row ir.Value) ir.Value {
	b := v.b
	return v.scWord(b.Shr(ir.U32, b.Mul(ir.U32, subIdx, b.Const(ir.U32, 12)), b.Const(ir.U32, 5)), row)
}

// scNext is the carried word for the trip kb sub-blocks after cur: w1 (the
// word after cur's first) where that pair starts one word on -- cur's bit is
// at least 32-12*kb -- and x otherwise. The final trip's clamp (GemmVolta's
// advance) repeats cur and may step anyway; that trip is staged into the
// buffer no MMA reads, so its scales do not matter.
func (v *voltaDeq) scNext(x, w1, cur ir.Value, kb int) ir.Value {
	b := v.b
	bit := b.And(ir.U32, b.Mul(ir.U32, cur, b.Const(ir.U32, 12)), b.Const(ir.U32, 31))
	return b.Select(ir.U32, b.Lt(ir.U32, b.Const(ir.U32, int64(31-12*kb)), bit), w1, x)
}

// words loads one row's packed words for one sub-block: the primary plane's
// pw, then the secondary's hw. idx is the sub-block's word base plus the row;
// im is a constant row offset on top of it.
func (v *voltaDeq) words(idx ir.Value, im int64) []ir.Value {
	return v.wordsN(idx, im, v.pw)
}

// wordsN is words for the first n primary words of a sub-block (and every
// secondary word), so a thread can stage a part of a sub-block. idx may
// already be offset to the part's first word.
func (v *voltaDeq) wordsN(idx ir.Value, im int64, n int) []ir.Value {
	out := make([]ir.Value, n+v.hw)
	idx = v.plus(idx, v.qBase)
	for w := range out {
		out[w] = v.b.Load(ir.U32, v.pQS, idx, im+int64(w)*v.wstride)
	}
	return out
}

// scales is the row's scale and negated minimum, each packed twice as f16x2.
func (v *voltaDeq) scales(subIdx, row ir.Value) (s2, m2 ir.Value) {
	return v.scalesOf(v.scaleWords(subIdx, row), subIdx, row)
}

// scaleWords loads the words scalesOf reads for one row and sub-block: the d
// word, then the sub-block scale word where the format has one. It is apart
// from the arithmetic so a pipelined kernel can issue it a trip early.
func (v *voltaDeq) scaleWords(subIdx, row ir.Value) []ir.Value {
	b, qi := v.b, v.qi
	if qi.float > 0 {
		return nil // a float weight carries no scale
	}
	c := func(x int64) ir.Value { return b.Const(ir.U32, x) }
	super := subIdx
	if qi.perSuper > 1 {
		super = b.Shr(ir.U32, subIdx, c(int64(log2(qi.perSuper))))
	}
	var out []ir.Value
	if dSlots := DSlots(v.t); dSlots > 1 {
		idx := b.Add(ir.U32, b.Mul(ir.U32, super, c(int64((v.rows+dSlots-1)/dSlots))),
			b.Shr(ir.U32, row, c(int64(log2(dSlots)))))
		out = append(out, b.Load(ir.U32, v.pD, v.plus(idx, v.dBase), 0))
	} else {
		out = append(out, b.Load(ir.U32, v.pD, v.plus(b.Add(ir.U32, b.Mul(ir.U32, super, c(int64(v.rows))), row), v.dBase), 0))
	}
	if ScStream(v.t) && v.carry {
		// The carried form (voltaDeq.carry): the word after the pair's first,
		// clamped to the row's last. The first is the caller's.
		lo := b.Shr(ir.U32, b.Mul(ir.U32, subIdx, c(12)), c(5))
		out = append(out, v.scWord(b.Min(ir.U32, b.Add(ir.U32, lo, c(1)), c(int64(v.scLast))), row))
	} else if ScStream(v.t) {
		// Container v27's stream (scstream.go): only the pair's first word is
		// loaded a trip early; scalesOf reads the next itself, because a third
		// pipelined register per item measured slower.
		lo, _, _ := scStreamAt(b, subIdx)
		out = append(out, b.Load(ir.U32, v.pSC, v.plus(b.Add(ir.U32, b.Mul(ir.U32, lo, c(int64(v.rows))), row), v.scBase), 0))
	} else if qi.perSuper > 1 {
		perWord := 4
		if qi.biasArray {
			perWord = 2
		}
		wi := b.Shr(ir.U32, subIdx, c(int64(log2(perWord))))
		out = append(out, b.Load(ir.U32, v.pSC, v.plus(b.Add(ir.U32, b.Mul(ir.U32, wi, c(int64(v.rows))), row), v.scBase), 0))
	}
	return out
}

// scalesOf is scales from words scaleWords loaded for the same subIdx and row.
func (v *voltaDeq) scalesOf(w []ir.Value, subIdx, row ir.Value) (s2, m2 ir.Value) {
	return v.scalesOfX(w, 0, subIdx, row)
}

// scalesOfX is scalesOf in the carried form: x is the stream word the pair
// starts in, which the caller carried from the previous trip (voltaDeq.carry);
// w[1] is the word after it.
func (v *voltaDeq) scalesOfX(w []ir.Value, x, subIdx, row ir.Value) (s2, m2 ir.Value) {
	b, qi := v.b, v.qi
	sc, mn := v.scalesF32X(w, x, subIdx, row)
	if qi.e8m0 && DSlots(v.t) > 1 {
		return b.PackF16(sc, sc), b.PackF16(v.zeroF, v.zeroF)
	}
	if !qi.biasArray && qi.biasK != 0 {
		mn = b.Mul(ir.F32, b.ConstF32(qi.biasK), sc)
	}
	neg := b.Sub(ir.F32, v.zeroF, mn)
	return b.PackF16(sc, sc), b.PackF16(neg, neg)
}

// scalesF32X is the row's sub-block scale and stored minimum in float32, the
// weight being sc*q - mn: scalesOfX's arithmetic before it packs to binary16,
// and without the minimum a centred format's bias would add (biasK*sc), which
// the integer GEMM takes off the weights instead (GemmInt8). mn is 0 where
// the format stores no minimum.
func (v *voltaDeq) scalesF32X(w []ir.Value, x, subIdx, row ir.Value) (sc, mn ir.Value) {
	b, qi := v.b, v.qi
	c := func(x int64) ir.Value { return b.Const(ir.U32, x) }
	// The row's super-scale and minimum, following matvec.go's layout.
	var d, dm ir.Value
	if dSlots := DSlots(v.t); dSlots > 1 {
		sh := b.Mul(ir.U32, b.And(ir.U32, row, c(int64(dSlots-1))), c(int64(32/dSlots)))
		if qi.e8m0 {
			// The stored byte shifted left 23 is the f32 scale (e8m0Store),
			// and the code carries no bias: frags decodes it to its value.
			d = b.Bitcast(ir.F32, b.Shl(ir.U32, b.And(ir.U32, b.Shr(ir.U32, w[0], sh), c(0xFF)), c(23)))
			return d, v.zeroF
		}
		d = b.CvtF16H(b.Shr(ir.U32, w[0], sh))
	} else if qi.d32 {
		d = b.Bitcast(ir.F32, w[0]) // TQ2F: the word is the scale
	} else {
		d = b.CvtF16H(w[0])
		if qi.biasArray {
			dm = b.CvtF16H(b.Shr(ir.U32, w[0], c(16)))
		}
	}
	sc, mn = d, v.zeroF
	if ScStream(v.t) && v.carry {
		bit := b.And(ir.U32, b.Mul(ir.U32, subIdx, c(12)), c(31))
		scI, mI := scStreamPair(b, x, w[1], bit)
		sc, mn = b.Mul(ir.F32, d, b.CvtF32(scI)), b.Mul(ir.F32, dm, b.CvtF32(mI))
	} else if ScStream(v.t) {
		_, hi, bit := scStreamAt(b, subIdx)
		hw := b.Load(ir.U32, v.pSC, v.plus(b.Add(ir.U32, b.Mul(ir.U32, hi, c(int64(v.rows))), row), v.scBase), 0)
		scI, mI := scStreamPair(b, w[1], hw, bit)
		sc, mn = b.Mul(ir.F32, d, b.CvtF32(scI)), b.Mul(ir.F32, dm, b.CvtF32(mI))
	} else if qi.perSuper > 1 {
		perWord, stride := 4, 8
		if qi.biasArray {
			perWord, stride = 2, 16
		}
		base := b.Mul(ir.U32, b.And(ir.U32, subIdx, c(int64(perWord-1))), c(int64(stride)))
		sw := w[1]
		scI := b.And(ir.U32, b.Shr(ir.U32, sw, base), c(0xFF))
		if qi.scOff != 0 {
			scI = b.Sub(ir.I32, scI, b.Const(ir.I32, int64(-qi.scOff)))
		}
		sc = b.Mul(ir.F32, d, b.CvtF32(scI))
		if qi.biasArray {
			mI := b.And(ir.U32, b.Shr(ir.U32, sw, b.Add(ir.U32, base, c(8))), c(0xFF))
			mn = b.Mul(ir.F32, dm, b.CvtF32(mI))
		}
	} else if qi.biasArray {
		mn = dm // Q5_1: the minimum alone, no sc plane (MinInD)
	}
	return sc, mn
}

// frags turns one row's raw words (words' order) into the A-fragment words of
// every step of the sub-block: steps entries of two u32, four f16 each, in the
// element order voltaSteps gives and ActF16 writes the activations in.
//
// A nibble pair Or-ed into the bits of 1024.0 is 1024+q in two integer ops;
// less 1024 it is q, exactly; one fused f16 multiply-add applies the scale and
// minimum.
func (v *voltaDeq) frags(raw []ir.Value, s2, m2 ir.Value) [][]ir.Value {
	return v.fragsN(raw, s2, m2, v.pw)
}

// fragsN is frags over n primary words: a part of a sub-block, which on a
// format with no secondary plane decodes exactly as the sub-block's first n
// words do (the scale is the sub-block's, the position does not enter).
func (v *voltaDeq) fragsN(raw []ir.Value, s2, m2 ir.Value, n int) [][]ir.Value {
	b, qi := v.b, v.qi
	if n != v.pw && v.hw > 0 {
		panic("kernels: voltaDeq.fragsN: a part of a sub-block with a secondary plane")
	}
	c := func(x int64) ir.Value { return b.Const(ir.U32, x) }
	hm := int64(1)<<uint(qi.hi) - 1
	// pair is two exact integer weights q as f16: the nibble pair at bit sh
	// (with a secondary plane's bits from sw at hsh) Or-ed into 1024.0, less it.
	pair := func(w ir.Value, sh int, sw ir.Value, hsh int) ir.Value {
		x := w
		if sh != 0 {
			x = b.Shr(ir.U32, w, c(int64(sh)))
		}
		if qi.bits == 8 {
			return b.SubF16x2(b.Xor(ir.U32, b.And(ir.U32, x, v.m8), v.magic), v.off2)
		}
		p := b.And(ir.U32, x, v.m4)
		if sw != 0 {
			y := sw
			if hsh != 0 {
				y = b.Shr(ir.U32, sw, c(int64(hsh)))
			}
			p = b.Xor(ir.U32, p, b.Shl(ir.U32, b.And(ir.U32, y, c(hm|hm<<16)), c(4)))
		}
		return b.SubF16x2(b.Xor(ir.U32, p, v.magic), v.off2)
	}
	deq := func(p ir.Value) ir.Value { return b.FmaF16x2(p, s2, m2) }
	out := make([][]ir.Value, v.steps*n/v.pw)
	if qi.float == 2 {
		// Words 2s and 2s+1 hold elements 4s..4s+3, low half first; step s
		// is the pairs (4s, 4s+2) and (4s+1, 4s+3) as binary16.
		lo := func(w ir.Value) ir.Value {
			if qi.bf16 {
				return b.Bitcast(ir.F32, b.Shl(ir.U32, w, c(16)))
			}
			return b.CvtF16H(w)
		}
		hi := func(w ir.Value) ir.Value {
			if qi.bf16 {
				return b.Bitcast(ir.F32, b.And(ir.U32, w, c(0xFFFF0000)))
			}
			return b.CvtF16H(b.Shr(ir.U32, w, c(16)))
		}
		for s := range out {
			w0, w1 := raw[2*s], raw[2*s+1]
			out[s] = []ir.Value{b.PackF16(lo(w0), lo(w1)), b.PackF16(hi(w0), hi(w1))}
		}
		return out
	}
	if qi.e8m0 {
		// An e2m1 code at bits 9-11 of a binary16, sign at 15, is its value
		// times 2^-14. 2^15 makes it the doubled value mxfp4Codes holds
		// (exact, at most 12). The E8M0 scale is a second multiply rather than
		// folded into the first, because scale*2^15 overflows f16.
		k15 := c(0x78007800) // 2^15, twice
		zero2 := c(0)
		e2m1 := func(w ir.Value, sh int) ir.Value {
			x := w
			if sh != 0 {
				x = b.Shr(ir.U32, w, c(int64(sh)))
			}
			p := b.And(ir.U32, x, v.m4)
			h := b.Xor(ir.U32, b.Shl(ir.U32, b.And(ir.U32, p, c(0x00070007)), c(9)),
				b.Shl(ir.U32, b.And(ir.U32, p, c(0x00080008)), c(12)))
			return deq(b.FmaF16x2(h, k15, zero2))
		}
		for w := 0; w < n; w++ {
			for half := 0; half < 2; half++ {
				out[2*w+half] = []ir.Value{e2m1(raw[w], 4*half), e2m1(raw[w], 4*half+8)}
			}
		}
		return out
	}
	if qi.bits == 8 {
		for w := 0; w < n; w++ {
			x := b.Xor(ir.U32, raw[w], c(0x80808080))
			out[w] = []ir.Value{deq(pair(x, 0, 0, 0)), deq(pair(x, 8, 0, 0))}
		}
		return out
	}
	lanes := 0
	if qi.hi > 0 {
		lanes = 8 / qi.hi
	}
	sec := raw[n:]
	h := qi.sub / 2
	for w := 0; w < n; w++ {
		wd := raw[w]
		for half := 0; half < 2; half++ {
			var sw ir.Value
			hsh := 0
			if v.hw > 0 {
				g := w + half*(h/4)
				sw, hsh = sec[g/lanes], qi.hi*(g%lanes)
			}
			out[2*w+half] = []ir.Value{deq(pair(wd, 4*half, sw, hsh)), deq(pair(wd, 4*half+8, sw, hsh+8))}
		}
	}
	return out
}
