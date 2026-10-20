package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/kernels"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// MMAProbe is a bare int8 GEMM through one warp-level matrix instruction, used
// to pin down the fragment layout on real hardware before any real kernel
// depends on it.
//
// The layout is the part that fails silently: a wrong lane mapping transposes
// a block or swaps rows without faulting. So the mapping is written once here,
// checked against a float64 reference on the device, and every real MMA kernel
// reuses these index expressions. The operands are in natural order in memory
// (A row-major [M][K], B column-major [K][N], as .row.col means) and the kernel
// does the gather; a probe that pre-arranged them would only test itself.
func MMAProbe(sh ir.MMAShape) (*ir.Kernel, error) {
	if sh.Kind == ir.MMAF16 {
		return mmaProbeF16(sh)
	}
	na, nb, nc := sh.Frags()
	if na == 0 || nb == 0 || nc == 0 {
		return nil, fmt.Errorf("kernels: MMAProbe: bad shape %dx%dx%d", sh.M, sh.N, sh.K)
	}
	b := ir.New("mmaprobe", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.U32)
	pB := b.Param("pB", ir.U32)
	pOut := b.Param("pOut", ir.I32)

	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))
	g := b.Shr(ir.U32, lane, b.Const(ir.U32, 2)) // 0..7
	t := b.And(ir.U32, lane, b.Const(ir.U32, 3)) // 0..3
	kw := sh.K / 4                               // u32 per row of A / column of B

	// A[m][k], row-major: lane holds rows g and g+8, four consecutive k at
	// 4t, and (for K=32) another four at 16+4t.
	af := make([]ir.Value, na)
	for i := 0; i < na; i++ {
		row, half := g, 0
		if i%2 == 1 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		if i >= 2 {
			half = 4 // 16 elements on, in u32 units
		}
		af[i] = b.Load(ir.U32, pA, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(kw))), t),
			int64(half))
	}
	// B[k][n], column-major: lane holds column g, the same k slices.
	bf := make([]ir.Value, nb)
	for i := 0; i < nb; i++ {
		half := 0
		if i >= 1 {
			half = 4
		}
		bf[i] = b.Load(ir.U32, pB, b.Add(ir.U32, b.Mul(ir.U32, g, b.Const(ir.U32, int64(kw))), t),
			int64(half))
	}
	zero := b.Const(ir.I32, 0)
	cf := make([]ir.Value, nc)
	for i := range cf {
		cf[i] = zero
	}
	d := b.MMA(sh, af, bf, cf)

	// D[m][n], row-major out: components 0,1 are row g at columns 2t and 2t+1;
	// components 2,3 are row g+8.
	for i, v := range d {
		row := g
		if i >= 2 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		col := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, 2)), b.Const(ir.U32, int64(i%2)))
		b.Store(pOut, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(sh.N))), col), v, 0)
	}
	return b.Done(), nil
}

// MatVecMMA is the prefill GEMM on the warp-level matrix instruction (int8
// tensor cores, as llama.cpp's prefill uses, in place of dp4a).
//
// It needs no shared-memory staging: the JIT chooses the packed layout, and it
// already is the fragment layout. PackWeights writes
// qs[(si*words + w)*nrows + r], and for a 4-bit format word w holds elements
// {4w..4w+3} in the low nibbles and {sub/2+4w ..} in the high ones; mma's A
// fragment wants lane (g,t) to hold A[g][4t..4t+3] and A[g][16+4t..], which are
// the two nibble halves of one packed word (w = t, row = g). Across the warp
// those loads are fully coalesced.
//
// The tile's K is the format's sub-block, not a free parameter: each MMA must
// produce one complete integer dot for the per-sub-block scale to multiply, so
// a 16-element sub-block takes m16n8k16 and a 32-element one m16n8k32.
func MatVecMMA(s MatVecShape) (*ir.Kernel, error) {
	if int(s.T) >= kernels.Quants() {
		return nil, fmt.Errorf("kernels: MatVecMMA: no layout for %s", s.T)
	}
	qi := infoOf(s.T)
	if qi.blockE == 0 {
		return nil, fmt.Errorf("kernels: MatVecMMA: no layout for %s", s.T)
	}
	if why := DeviceWhyNot(s.T); why != "" {
		return nil, fmt.Errorf("kernels: MatVecMMA: %s", why)
	}
	if qi.float > 0 {
		return nil, fmt.Errorf("kernels: MatVecMMA: %v is a float format; its weights are not integers", s.T)
	}
	if qi.codes != nil && (qi.bits != 4 || qi.sub != 32) {
		// Only the 4-bit, 32-wide arm of afrag decodes a code table.
		return nil, fmt.Errorf("kernels: MatVecMMA: %v's code table needs the 4-bit 32-wide arm", s.T)
	}
	if s.Experts > 1 || s.Split > 1 {
		return nil, fmt.Errorf("kernels: MatVecMMA: experts and split are not supported")
	}
	if s.NTok < 8 {
		return nil, fmt.Errorf("kernels: MatVecMMA: NTok=%d needs at least 8 token columns", s.NTok)
	}
	mt, nt := max(s.MT, 1), max(s.NT, 1)
	mrows, ncols := 16*mt, 8*nt
	if s.Rows%mrows != 0 || s.NTok%ncols != 0 {
		return nil, fmt.Errorf("kernels: MatVecMMA: %dx%d tile does not divide rows=%d ntok=%d",
			mrows, ncols, s.Rows, s.NTok)
	}
	if s.K%qi.blockE != 0 || s.K <= 0 {
		return nil, fmt.Errorf("kernels: MatVecMMA: K=%d is not a multiple of %d", s.K, qi.blockE)
	}
	sh := ir.MMAShape{M: 16, N: 8, K: qi.sub}
	if !sh.Valid() {
		return nil, fmt.Errorf("kernels: MatVecMMA: sub-block %d has no matrix tile", qi.sub)
	}
	// No integer fold here: summing sub-blocks in integer before one float
	// multiply per super-block needs a second accumulator array, which raised
	// register use from 127 to 173 and cost about 25% in occupancy. On this
	// kernel occupancy beats instruction count. See
	// docs/engineering-history/gpu-kernels.md.
	na, nb, nc := sh.Frags()
	nsub := s.K / qi.sub
	// pw primary words then hw secondary ones; see qinfo.hi and packSub.
	pw, hw := qi.sub*qi.bits/32, qi.sub*qi.hi/32
	words := pw + hw     // packed u32 per sub-block per row
	awords := qi.sub / 4 // activation u32 per sub-block per token
	nbq := s.K / 32      // 32-element activation blocks per token

	const group = 128
	b := ir.New("matvecmma", [3]int{group, 1, 1})
	pQS := b.Param("pQS", ir.U32)
	pD := b.Param("pD", ir.U32)
	pSC := b.Param("pSC", ir.U32)
	pA := b.Param("pA", ir.U32)
	pAX := b.Param("pAX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	var pBias ir.Value
	if s.Bias {
		pBias = b.Param("pBias", ir.F32)
	}

	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	warp := b.Div(ir.U32, tid, b.Const(ir.U32, 32))
	lane := b.And(ir.U32, tid, b.Const(ir.U32, 31))
	g := b.Shr(ir.U32, lane, b.Const(ir.U32, 2)) // 0..7
	t := b.And(ir.U32, lane, b.Const(ir.U32, 3)) // 0..3

	mtiles, ntiles := s.Rows/mrows, s.NTok/ncols
	// Row varies fastest, so the warps of a CTA read different row blocks. Making
	// the token slice fastest (sharing rows across a CTA) measured slower: the
	// redundant reads were served by L2, and the row-fastest coalescing is real.
	warp = b.Min(ir.U32, warp, b.Const(ir.U32, int64(mtiles*ntiles-1)))
	wm := b.Rem(ir.U32, warp, b.Const(ir.U32, int64(mtiles)))
	wn := b.Div(ir.U32, warp, b.Const(ir.U32, int64(mtiles)))
	rowBase := b.Add(ir.U32, b.Mul(ir.U32, wm, b.Const(ir.U32, int64(mrows))), g)
	tokBase := b.Mul(ir.U32, wn, b.Const(ir.U32, int64(ncols)))
	nrows := b.Const(ir.U32, int64(s.Rows))
	add := func(v ir.Value, n int64) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, b.Const(ir.U32, n))
	}
	// The weight row this lane holds for m-tile i, half h (the +8 rows).
	wrow := make([]ir.Value, 2*mt)
	for i := 0; i < mt; i++ {
		wrow[2*i] = add(rowBase, int64(i*16))
		wrow[2*i+1] = add(rowBase, int64(i*16+8))
	}
	// The token column this lane holds for n-tile j (the B fragment).
	btok := make([]ir.Value, nt)
	for j := 0; j < nt; j++ {
		btok[j] = add(b.Add(ir.U32, tokBase, g), int64(j*8))
	}

	zeroF := b.ConstF32(0)
	one := b.Const(ir.U32, 1)
	mask := b.Const(ir.U32, 0x0F0F0F0F)
	// The zero point moves onto the weights: a packed quant is unsigned and the
	// true weight is q - biasK. Adding (0x80 - biasK) to each byte keeps every
	// byte below 256 for every eligible format (so nothing carries between
	// lanes) and the XOR with 0x80 then maps it to the signed q - biasK, which
	// mma's s8 operand takes as is. Two ops per fragment replace an Fma per
	// output element per sub-block, and the sumA registers are freed.
	centered := !s.Center.NoCenter && qi.biasK != 0 && !qi.biasArray
	var cAdd, cXor ir.Value
	if centered {
		bk := int64(qi.biasK)
		c := (0x80 - bk) & 0xFF
		cAdd = b.Const(ir.U32, c|c<<8|c<<16|c<<24)
		cXor = b.Const(ir.U32, int64(0x80808080))
	}
	center := func(v ir.Value) ir.Value {
		if !centered {
			return v
		}
		return b.Xor(ir.U32, b.Add(ir.U32, v, cAdd), cXor)
	}
	four := b.Const(ir.U32, 4)
	negBiasK := b.ConstF32(-qi.biasK)

	// Every address that does not change with the sub-block is computed once,
	// and the two that do are induction variables. wBase0 and aBase0 hold the
	// loop-invariant part of a fragment address; the loop carries only the
	// offset into the row.
	wStride := int64(words) * int64(s.Rows)
	tn := b.Mul(ir.U32, t, nrows)
	if qi.bits == 4 && qi.sub == 16 || qi.bits == 2 {
		// These formats index by word t&1, not t.
		tn = b.Mul(ir.U32, b.And(ir.U32, t, one), nrows)
	}
	// One base register and compile-time immediates: the rows a lane holds
	// differ by i*16 + h*8, a constant, so all its loads share one base and
	// take immediate displacements rather than a 64-bit address computation
	// each. rowImm is that displacement, in elements, and it is the same for the
	// payload, the super-scale and the sub-block scale (all indexed by row at
	// stride nrows).
	rowImm := func(x int) int64 { return int64((x/2)*16 + (x%2)*8) }
	wBase0 := b.Add(ir.U32, tn, wrow[0])
	// The nibble-half selector for the 16-element 4-bit formats, hoisted: it
	// depends on the lane and not on the sub-block.
	var nibShift ir.Value
	if qi.bits == 4 && qi.sub == 16 {
		nibShift = b.Mul(ir.U32, b.Shr(ir.U32, t, one), four)
	}
	// TQ2's 2-bit plane at sub 32: k=4t.. is field t>>1 of word t&1 and
	// k=16+4t.. field 2+(t>>1) of the same word (packSub), so a lane's two
	// halves are one load and two shifts.
	var fieldShift ir.Value
	if qi.bits == 2 {
		if qi.sub != 32 {
			return nil, fmt.Errorf("kernels: MatVecMMA: %v's 2-bit plane at sub %d has no arm", s.T, qi.sub)
		}
		fieldShift = b.Mul(ir.U32, b.Shr(ir.U32, t, one), b.Const(ir.U32, 2))
	}
	// The secondary plane's base is not the primary's: wBase0 carries the
	// word-within-sub-block selector, and the hi plane's base is the row alone
	// plus pw words.
	//
	// A group gi sits in secondary word gi/lanes at bit hi*(gi%lanes), as in
	// matvec.go's hiOf. A lane's low-nibble group is t and its high-nibble group
	// t+pw, with t a runtime value, so the word index must be constant over t in
	// [0,4): either the group fits in one word (c+3 < lanes) or the offset is a
	// whole number of words (c%lanes == 0). Anything else is refused. Deriving
	// this from the format's arithmetic, rather than from which arm a format
	// falls in, is what keeps Q5_K (4+1 at sub 32) correct here; plane changes
	// have more than once reached matvec.go and not this file.
	var wBaseHi, hiMask ir.Value
	hiShift := make([]ir.Value, 2) // [low nibbles, high nibbles]
	hiWordImm := make([]int64, 2)
	if hw > 0 {
		// A secondary plane is combined only in the two 4-bit arms of afrag, so
		// an 8-bit primary with one would silently read the primary alone.
		// Refuse it here.
		if qi.bits != 4 {
			return nil, fmt.Errorf("kernels: MatVecMMA: %v has a secondary plane and a "+
				"%d-bit primary; only the 4-bit arms combine one", s.T, qi.bits)
		}
		lanes := 8 / qi.hi
		hiMask = b.Const(ir.U32, int64(rep4(uint32(1<<uint(qi.hi)-1))))
		wBaseHi = b.Add(ir.U32, wrow[0], b.Mul(ir.U32, b.Const(ir.U32, int64(pw)), nrows))
		// Group offsets: 0 for the low nibbles, pw for the high ones. A
		// 16-wide format has only the first -- its k=16 tile is one
		// sub-block and one nibble half per lane.
		cs := []int{0, pw}
		if qi.sub == 16 {
			cs = []int{0}
		}
		for gi, c := range cs {
			switch {
			case c+3 < lanes:
				hiWordImm[gi] = 0
				hiShift[gi] = b.Mul(ir.U32, add(t, int64(c)), b.Const(ir.U32, int64(qi.hi)))
			case c%lanes == 0:
				hiWordImm[gi] = int64(c / lanes)
				hiShift[gi] = b.Mul(ir.U32, t, b.Const(ir.U32, int64(qi.hi)))
			default:
				return nil, fmt.Errorf("kernels: MatVecMMA: %v's secondary plane "+
					"spans words for group offset %d (hi=%d lanes=%d pw=%d)",
					s.T, c, qi.hi, lanes, pw)
			}
		}
	}
	// Same for the activation side: n-tile j is 8 token columns further on, so
	// its offset is j*8*(K/4) elements from tile 0's.
	aBase0 := b.Add(ir.U32, b.Mul(ir.U32, btok[0], b.Const(ir.U32, int64(s.K/4))), t)
	aImm := func(j int) int64 { return int64(j) * 8 * int64(s.K/4) }
	// afrag emits this lane's A fragments for m-tile i, with W the streaming
	// weight cursor (words*Rows further along each sub-block).
	afrag := func(W ir.Value, i int) []ir.Value {
		f := make([]ir.Value, na)
		idx := b.Add(ir.U32, W, wBase0)
		for h := 0; h < 2; h++ {
			im := rowImm(2*i + h)
			switch {
			case qi.bits == 4 && qi.sub == 32:
				// One word carries both halves: low nibbles are k=4t.., high
				// nibbles k=16+4t.. -- exactly a0/a2 (h=0) or a1/a3 (h=1).
				w := b.Load(ir.U32, pQS, idx, im)
				lo := b.And(ir.U32, w, mask)
				hi := b.And(ir.U32, b.Shr(ir.U32, w, four), mask)
				if qi.codes != nil {
					lo, hi = decodeE2M1(b, lo), decodeE2M1(b, hi)
				}
				if hw > 0 {
					// Combined before center, which is affine per byte (centering the
					// planes apart subtracts the offset twice). Xor because the planes
					// occupy disjoint bits and the IR has no Or. Group t for the low
					// nibbles, t+pw for the high, the pairing packSub writes.
					hv := b.Load(ir.U32, pQS, b.Add(ir.U32, W, wBaseHi), im)
					hvHi := hv
					if hiWordImm[1] != hiWordImm[0] {
						hvHi = b.Load(ir.U32, pQS, b.Add(ir.U32, W, wBaseHi),
							im+(hiWordImm[1]-hiWordImm[0])*int64(s.Rows))
					}
					shl := b.Const(ir.U32, int64(qi.bits))
					lo = b.Xor(ir.U32, lo, b.Shl(ir.U32,
						b.And(ir.U32, b.Shr(ir.U32, hv, hiShift[0]), hiMask), shl))
					hi = b.Xor(ir.U32, hi, b.Shl(ir.U32,
						b.And(ir.U32, b.Shr(ir.U32, hvHi, hiShift[1]), hiMask), shl))
				}
				f[h] = center(lo)
				f[2+h] = center(hi)
			case qi.bits == 4 && qi.sub == 16:
				// Two words per sub-block, four k per nibble half: lane t reads
				// word t&1 and takes the low half for t<2, the high for t>=2.
				w := b.Load(ir.U32, pQS, idx, im)
				q := b.And(ir.U32, b.Shr(ir.U32, w, nibShift), mask)
				if hw > 0 {
					// Combined before center; see the 32-wide arm.
					hv := b.Load(ir.U32, pQS, b.Add(ir.U32, W, wBaseHi), im)
					code := b.And(ir.U32, b.Shr(ir.U32, hv, hiShift[0]), hiMask)
					q = b.Xor(ir.U32, q, b.Shl(ir.U32, code, b.Const(ir.U32, int64(qi.bits))))
				}
				f[h] = center(q)
			case qi.bits == 2:
				w := b.Load(ir.U32, pQS, idx, im)
				m3 := b.Const(ir.U32, 0x03030303)
				lo := b.And(ir.U32, b.Shr(ir.U32, w, fieldShift), m3)
				hi := b.And(ir.U32, b.Shr(ir.U32, w, b.Add(ir.U32, fieldShift, four)), m3)
				f[h] = center(lo)
				f[2+h] = center(hi)
			default:
				// 8-bit: one word is four quants, so a fragment register is a
				// load and, for a biased format, its centering. A secondary plane
				// cannot reach here (refused beside wBaseHi).
				for x := 0; x < na/2; x++ {
					f[x*2+h] = center(b.Load(ir.U32, pQS, idx, im+int64(x*4)*int64(s.Rows)))
				}
			}
		}
		return f
	}
	// bfrag emits this lane's B fragment for n-tile j, with A the streaming
	// activation cursor.
	bfrag := func(idx ir.Value, j int) []ir.Value {
		f := make([]ir.Value, nb)
		for x := 0; x < nb; x++ {
			f[x] = b.Load(ir.U32, pA, idx, aImm(j)+int64(x*4))
		}
		return f
	}
	// rowScaleInt reads the per-sub-block integer scale (and the min index for a
	// format that carries one) for the row this lane holds in slot x. scRow is
	// the pSC word offset shared by every row, computed once per sub-block.
	scRow := func(subIdx ir.Value) (off, base ir.Value) {
		perWord, stride := 4, 8
		if qi.biasArray {
			perWord, stride = 2, 16
		}
		wi := b.Shr(ir.U32, subIdx, b.Const(ir.U32, int64(log2(perWord))))
		return b.Mul(ir.U32, wi, nrows),
			b.Mul(ir.U32, b.And(ir.U32, subIdx, b.Const(ir.U32, int64(perWord-1))),
				b.Const(ir.U32, int64(stride)))
	}
	rowScaleInt := func(idx, base ir.Value, x int) (scI, mI ir.Value) {
		sw := b.Load(ir.U32, pSC, idx, rowImm(x))
		sc := b.And(ir.U32, b.Shr(ir.U32, sw, base), b.Const(ir.U32, 0xFF))
		scI = sc
		if qi.scOff != 0 {
			scI = b.Sub(ir.I32, sc, b.Const(ir.I32, int64(-qi.scOff)))
		}
		if qi.biasArray {
			mI = b.And(ir.U32, b.Shr(ir.U32, sw, b.Add(ir.U32, base, b.Const(ir.U32, 8))),
				b.Const(ir.U32, 0xFF))
		}
		return
	}
	// superScale reads the super-block scale (and its minimum) for slot x. This
	// is the device's second reader of the d plane beside matvec.go; a layout
	// change must reach both, or decode is right and prefill is fluent garbage.
	//
	// On the narrow formats DIndex packs DSlots rows into one word, so the slot
	// is the row's residue. Every rowImm is a multiple of eight, so the slot is
	// rowBase's alone (loop-invariant) and the per-slot displacement stays an
	// immediate.
	dSlots := DSlots(s.T)
	narrowD := dSlots > 1
	dRow := func(super ir.Value) (off, shift ir.Value) {
		if narrowD {
			return b.Mul(ir.U32, super, b.Const(ir.U32, int64((s.Rows+dSlots-1)/dSlots))),
				b.Mul(ir.U32, b.And(ir.U32, rowBase, b.Const(ir.U32, int64(dSlots-1))),
					b.Const(ir.U32, int64(32/dSlots)))
		}
		return b.Mul(ir.U32, super, nrows), 0
	}
	superScale := func(idx, shift ir.Value, x int) (d, dmin ir.Value) {
		im := rowImm(x)
		if narrowD {
			im /= int64(dSlots)
		}
		dw := b.Load(ir.U32, pD, idx, im)
		if narrowD {
			// No minimum array on a narrow format, so the bits above this
			// row's are the next rows' scales and must be shifted away.
			v := b.Shr(ir.U32, dw, shift)
			if qi.e8m0 {
				// One byte, and the byte shifted left 23 is the f32; no convert.
				return b.Bitcast(ir.F32, b.Shl(ir.U32,
					b.And(ir.U32, v, b.Const(ir.U32, 0xFF)), b.Const(ir.U32, 23))), 0
			}
			return b.CvtF16H(v), 0
		}
		if qi.biasArray {
			return b.CvtF16H(dw), b.CvtF16H(b.Shr(ir.U32, dw, b.Const(ir.U32, 16)))
		}
		if qi.d32 {
			return b.Bitcast(ir.F32, dw), 0 // TQ2F: the word is the scale
		}
		return b.CvtF16H(dw), 0
	}
	// sumAt is the per-16 activation sum for the output column this lane owns in
	// n-tile j, pair member c, at sub-block subIdx.
	sumOff := int64(s.NTok * nbq)
	// The two per-output-column bases, both loop-invariant.
	tk0 := outTokIdx(b, tokBase, t, 0, 0)
	sumBase0 := b.Mul(ir.U32, tk0, b.Const(ir.U32, int64(s.K/16)))
	daBase0 := b.Mul(ir.U32, tk0, b.Const(ir.U32, int64(nbq)))
	// Output column y is (y/2)*8 + y%2 token columns on from column 0.
	colImm := func(y, stride int) int64 { return int64((y/2)*8+y%2) * int64(stride) }
	// The caller builds the cursor; for a 32-element sub-block it advances by
	// two because the per-16 sums array has two entries per sub-block.
	sumStep := 1
	if qi.sub != 16 {
		sumStep = 2
	}
	sumAt := func(idx ir.Value, y int) ir.Value {
		o := sumOff + colImm(y, s.K/16)
		if qi.sub == 16 {
			return b.Load(ir.F32, pAX, idx, o)
		}
		return b.Add(ir.F32, b.Load(ir.F32, pAX, idx, o), b.Load(ir.F32, pAX, idx, o+1))
	}
	daAt := func(idx ir.Value, y int) ir.Value {
		return b.Load(ir.F32, pAX, idx, colImm(y, nbq))
	}

	zeroU := b.Const(ir.U32, 0)
	zeroI := b.Const(ir.I32, 0)
	cf := make([]ir.Value, nc)
	for i := range cf {
		cf[i] = zeroI
	}
	var accs []ir.Value

	{
		b.Loop(int64(nsub))
		accs = make([]ir.Value, mt*nt*nc)
		for i := range accs {
			accs[i] = b.Phi(ir.F32, zeroF)
		}
		subIdx := b.Phi(ir.U32, zeroU)
		W := b.Phi(ir.U32, zeroU)
		A := b.Phi(ir.U32, zeroU)

		af := make([][]ir.Value, mt)
		for i := 0; i < mt; i++ {
			af[i] = afrag(W, i)
		}
		aIdx := b.Add(ir.U32, A, aBase0)
		bfr := make([][]ir.Value, nt)
		for j := 0; j < nt; j++ {
			bfr[j] = bfrag(aIdx, j)
		}
		super := subIdx
		if qi.perSuper > 1 {
			super = b.Shr(ir.U32, subIdx, b.Const(ir.U32, int64(log2(qi.perSuper))))
		}
		dOff, dShift := dRow(super)
		dBase := wrow[0]
		if narrowD {
			// DSlots rows to a word, so the row enters the index divided.
			dBase = b.Shr(ir.U32, wrow[0], b.Const(ir.U32, int64(log2(dSlots))))
		}
		dIdx := b.Add(ir.U32, dBase, dOff)
		var scIdx, scBase, scHi ir.Value
		if ScStream(s.T) {
			// Container v27's stream: two words and a bit, once per sub-block.
			lo, hi, bit := scStreamAt(b, subIdx)
			scIdx = b.Add(ir.U32, wrow[0], b.Mul(ir.U32, lo, nrows))
			scHi = b.Add(ir.U32, wrow[0], b.Mul(ir.U32, hi, nrows))
			scBase = bit
		} else if qi.perSuper > 1 {
			var scOff ir.Value
			scOff, scBase = scRow(subIdx)
			scIdx = b.Add(ir.U32, wrow[0], scOff)
		}
		scale := make([]ir.Value, 2*mt)
		dmin := make([]ir.Value, 2*mt)
		mInt := make([]ir.Value, 2*mt)
		for x := 0; x < 2*mt; x++ {
			d, dm := superScale(dIdx, dShift, x)
			scale[x], dmin[x] = d, dm
			if ScStream(s.T) {
				scI, mI := scStreamPair(b, b.Load(ir.U32, pSC, scIdx, rowImm(x)), b.Load(ir.U32, pSC, scHi, rowImm(x)), scBase)
				scale[x] = b.Mul(ir.F32, d, b.CvtF32(scI))
				mInt[x] = mI
			} else if qi.perSuper > 1 {
				scI, mI := rowScaleInt(scIdx, scBase, x)
				scale[x] = b.Mul(ir.F32, d, b.CvtF32(scI))
				mInt[x] = mI
			}
		}
		aBlk := subIdx
		if qi.sub == 16 {
			aBlk = b.Shr(ir.U32, subIdx, one)
		}
		daIdx := b.Add(ir.U32, daBase0, aBlk)
		da := make([]ir.Value, 2*nt)
		sumA := make([]ir.Value, 2*nt)
		for y := range da {
			da[y] = daAt(daIdx, y)
		}
		// The sums are not loaded at all when the weights carry the bias:
		// Builder.Done does no dead-code elimination.
		if !centered {
			sumIdx := b.Add(ir.U32, sumBase0, b.Mul(ir.U32, subIdx, b.Const(ir.U32, int64(sumStep))))
			for y := range sumA {
				sumA[y] = sumAt(sumIdx, y)
			}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				d4 := b.MMA(sh, af[i], bfr[j], cf)
				for c := 0; c < nc; c++ {
					x, y := 2*i+c/2, j*2+c%2
					var v ir.Value
					switch {
					case qi.biasArray:
						bias := dmin[x] // Q5_1: no sc plane and m is one (MinInD)
						if qi.perSuper > 1 {
							bias = b.Mul(ir.F32, dmin[x], b.CvtF32(mInt[x]))
						}
						v = b.Sub(ir.F32, b.Mul(ir.F32, b.CvtF32(d4[c]), scale[x]),
							b.Mul(ir.F32, sumA[y], bias))
					case qi.biasK == 0 || centered:
						// centered: the A fragment already holds q - biasK, so the
						// integer dot is the corrected dot.
						v = b.Mul(ir.F32, b.CvtF32(d4[c]), scale[x])
					default:
						v = b.Mul(ir.F32, scale[x], b.Fma(negBiasK, sumA[y], b.CvtF32(d4[c])))
					}
					n := (i*nt+j)*nc + c
					b.SetPhi(accs[n], b.Fma(v, da[y], accs[n]))
				}
			}
		}
		b.SetPhi(subIdx, b.Add(ir.U32, subIdx, one))
		b.SetPhi(W, b.Add(ir.U32, W, b.Const(ir.U32, wStride)))
		b.SetPhi(A, b.Add(ir.U32, A, b.Const(ir.U32, int64(awords))))
		b.EndLoop()
	}

	// The epilogue is one base and immediates too: out[token][row] at stride
	// Rows, and token and row differ from tile 0's by compile-time amounts.
	outBase := b.Add(ir.U32, b.Mul(ir.U32, outTokIdx(b, tokBase, t, 0, 0), nrows), wrow[0])
	for i := 0; i < mt; i++ {
		for j := 0; j < nt; j++ {
			for c := 0; c < nc; c++ {
				val := accs[(i*nt+j)*nc+c]
				if s.Bias {
					val = b.Add(ir.F32, val, b.Load(ir.F32, pBias, wrow[0], rowImm(2*i+c/2)))
				}
				off := int64(j*8+c%2)*int64(s.Rows) + rowImm(2*i+c/2)
				b.Store(pOut, outBase, val, off)
			}
		}
	}
	return b.Done(), nil
}

// outTokIdx is the output column this lane owns: n-tile j, pair member c.
func outTokIdx(b *ir.Builder, tokBase, t ir.Value, j, c int) ir.Value {
	v := b.Add(ir.U32, tokBase, b.Mul(ir.U32, t, b.Const(ir.U32, 2)))
	if off := int64(j*8 + c); off != 0 {
		v = b.Add(ir.U32, v, b.Const(ir.U32, off))
	}
	return v
}

// mmaProbeF16 is MMAProbe for the binary16 shape. A and B arrive as float32 in
// natural order and the kernel packs them, as the real kernels do: only the
// fragment is half precision.
//
// The fragment layout is not the int8 one with different arithmetic: a lane
// holds two halves per register rather than four bytes, so it gets two k at
// 2t and two more at 2t+8. Reading the s8 table instead transposes a block.
func mmaProbeF16(sh ir.MMAShape) (*ir.Kernel, error) {
	na, nb, nc := sh.Frags()
	b := ir.New("mmaprobef16", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))
	g := b.Shr(ir.U32, lane, b.Const(ir.U32, 2)) // 0..7
	t := b.And(ir.U32, lane, b.Const(ir.U32, 3)) // 0..3
	two := b.Mul(ir.U32, t, b.Const(ir.U32, 2))

	// A[m][k], row-major: register i covers row g (+8 for odd i) at k = 2t,
	// plus 8 more for the second pair of registers.
	af := make([]ir.Value, na)
	for i := 0; i < na; i++ {
		row := g
		if i%2 == 1 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		off := int64(0)
		if i >= 2 {
			off = 8
		}
		idx := b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(sh.K))), two)
		af[i] = b.PackF16(b.Load(ir.F32, pA, idx, off), b.Load(ir.F32, pA, idx, off+1))
	}
	// B[k][n], column-major: column g, the same k pairs.
	bf := make([]ir.Value, nb)
	for i := 0; i < nb; i++ {
		off := int64(0)
		if i >= 1 {
			off = 8
		}
		idx := b.Add(ir.U32, b.Mul(ir.U32, g, b.Const(ir.U32, int64(sh.K))), two)
		bf[i] = b.PackF16(b.Load(ir.F32, pB, idx, off), b.Load(ir.F32, pB, idx, off+1))
	}
	zero := b.ConstF32(0)
	cf := make([]ir.Value, nc)
	for i := range cf {
		cf[i] = zero
	}
	d := b.MMA(sh, af, bf, cf)
	for i, v := range d {
		row := g
		if i >= 2 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		col := b.Add(ir.U32, two, b.Const(ir.U32, int64(i%2)))
		b.Store(pOut, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(sh.N))), col), v, 0)
	}
	return b.Done(), nil
}
