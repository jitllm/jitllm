package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Gemm is a chunk matvec's group on the binary16 matrix instruction: Rows
// outputs of a weight in format T over K, against NTok tokens' binary16
// operand (ActF16T's layout), each output passed through Epi. The format's
// dequantization to binary16 fragments is its op fragment
// (kernels.VoltaDeq); Gemm is the loop nest around it, by its schedule
// (Tile: the warp tiles, the workgroup's warps, the sub-blocks staged a trip,
// the instruction shape and single or double staging; Split segments of k,
// each written to its plane for SplitSum).
//
// Parameters: pQS, pD, pSC, pB (the operand), pOut, then the epilogue's.
type Gemm struct {
	T             kernels.Quant
	K, Rows, NTok int
	Tile          kernels.VoltaTile
	Split         int
	Epi           *Epi
}

// Encode is the group's structure and schedule, for the kernel cache's key.
func (m *Gemm) Encode() []uint32 {
	t := m.Tile
	e := []uint32{5, uint32(m.T), uint32(m.K), uint32(m.Rows), uint32(m.NTok), uint32(m.Split),
		uint32(t.MT), uint32(t.NT), uint32(t.WM), uint32(t.WN), uint32(t.KB), uint32(t.F16K), b2u(t.Single)}
	if m.Epi != nil {
		e = append(e, m.Epi.Encode()...)
	}
	return e
}

// Width is the workgroup; Groups the grid.
func (m *Gemm) Width() int { return m.Tile.Threads() }

// Groups is the workgroups the group launches.
func (m *Gemm) Groups() int {
	return (m.Rows / m.Tile.Rows()) * (m.NTok / m.Tile.Toks()) * max(m.Split, 1)
}

// Emit forms the kernel.
func (m *Gemm) Emit() (*ir.Kernel, error) {
	t := m.Tile
	if !kernels.GemmVoltaOK(m.T) {
		return nil, fmt.Errorf("compose: Gemm: %v has no binary16 dequantization", m.T)
	}
	sub, bits, hi, fbytes, blockE := kernels.GemmFormat(m.T)
	fl := fbytes > 0
	if t.MT < 1 || t.NT < 1 || t.WM < 1 || t.WN < 1 || t.KB < 1 || t.F16K != 0 && t.F16K != 8 && t.F16K != 16 {
		return nil, fmt.Errorf("compose: Gemm: tile %+v", t)
	}
	split := max(m.Split, 1)
	if split > 1 && m.Epi != nil {
		return nil, fmt.Errorf("compose: Gemm: a split of %d writes partials; its epilogue is the reduction's", split)
	}
	BM, BN, nth := t.Rows(), t.Toks(), t.Threads()
	if m.Rows%BM != 0 || m.NTok%BN != 0 {
		return nil, fmt.Errorf("compose: Gemm: a %dx%d block over rows=%d ntok=%d", BM, BN, m.Rows, m.NTok)
	}
	if m.K%blockE != 0 || m.K <= 0 {
		return nil, fmt.Errorf("compose: Gemm: K=%d is not a multiple of %d", m.K, blockE)
	}
	nsub := m.K / sub
	if nsub%(split*t.KB) != 0 {
		return nil, fmt.Errorf("compose: Gemm: split %d x %d sub-blocks over %d", split, t.KB, nsub)
	}
	trips := nsub / split / t.KB
	words := sub*bits/32 + sub*hi/32
	subStride, rowStride := int64(words*m.Rows), int64(1)
	if fl {
		words = sub * fbytes / 4
		subStride, rowStride = int64(words), int64(m.K*fbytes/4)
	}
	kHalf := t.KB * sub / 2
	C := kHalf / 4
	subC := sub / 8
	switch {
	case C&(C-1) != 0 || C < 2 || subC < 1:
		return nil, fmt.Errorf("compose: Gemm: %d chunks a trip", C)
	case t.F16K != 0 && C != 4 && C%8 != 0:
		return nil, fmt.Errorf("compose: Gemm: an m16n8 tile stages 32 or a multiple of 64 elements a trip, not %d", 8*C)
	case t.Single && t.F16K == 0:
		return nil, fmt.Errorf("compose: Gemm: a single-buffered stage is an m16n8 tile's")
	}
	nbuf := 2
	if t.Single {
		nbuf = 1
	}
	if sh := nbuf * (BM + BN) * kHalf * 4; sh > 48<<10 {
		return nil, fmt.Errorf("compose: Gemm: %d bytes of shared memory", sh)
	}
	if (BM*t.KB)%nth != 0 && BM*t.KB > nth || (BN*C)%nth != 0 || nth%C != 0 {
		return nil, fmt.Errorf("compose: Gemm: %d threads do not tile %d A items and %d B chunks", nth, BM*t.KB, BN*C)
	}
	if m.Epi != nil {
		if err := m.Epi.Check(); err != nil {
			return nil, err
		}
	}

	b := ir.New("composed_gemm", [3]int{nth, 1, 1})
	pQS := b.Param("pQS", ir.U32)
	pD := b.Param("pD", ir.U32)
	pSC := b.Param("pSC", ir.U32)
	pB := b.Param("pB", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	if m.Epi != nil {
		m.Epi.Params(b)
	}
	shA := b.Shared("gemmA", ir.U32, nbuf*BM*kHalf)
	shB := b.Shared("gemmB", ir.U32, nbuf*BN*kHalf)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	add := func(v ir.Value, n int64) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, c(n))
	}
	wide := t.F16K != 0 && C >= 8
	swzOf := func(r ir.Value, weights bool) ir.Value {
		if wide {
			return kernels.F16Swizzle(b, r)
		}
		return kernels.VoltaSwizzle(b, r, C, weights)
	}
	chunkAt := func(swz ir.Value) []ir.Value {
		out := make([]ir.Value, C)
		for j := range out {
			out[j] = b.Shl(ir.U32, b.Xor(ir.U32, swz, c(int64(j))), c(2))
		}
		return out
	}

	// The workgroup's block, token blocks fastest so the workgroups reading
	// one weight block run together; a split's segment slowest.
	mB, nB := m.Rows/BM, m.NTok/BN
	grp := b.CTAID()
	bn := b.Rem(ir.U32, grp, c(int64(nB)))
	bm := b.Rem(ir.U32, b.Div(ir.U32, grp, c(int64(nB))), c(int64(mB)))
	seg := b.Div(ir.U32, grp, c(int64(mB*nB)))
	rowBlk := b.Mul(ir.U32, bm, c(int64(BM)))
	tokBlk := b.Mul(ir.U32, bn, c(int64(BN)))
	tid := b.TID()

	dq := kernels.NewVoltaDeq(b, m.T, pQS, pD, pSC, m.Rows)
	carry := dq.SetCarry(m.K, t.KB)

	// A staging: item it = tid + nth*i is (row it%BM, sub-block it/BM).
	nA := max(BM*t.KB/nth, 1)
	aC := make([]struct{ g, s, row, sb int64 }, nA)
	var aRow0, aSb0 ir.Value
	switch {
	case BM%nth == 0:
		per := BM / nth
		aRow0, aSb0 = tid, c(0)
		for i := range aC {
			r, sb := int64(nth*(i%per)), int64(i/per)
			aC[i].g, aC[i].s, aC[i].row, aC[i].sb = sb*subStride+r*rowStride, r*int64(kHalf), r, sb
		}
	case nth%BM == 0:
		aRow0, aSb0 = b.Rem(ir.U32, tid, c(int64(BM))), b.Div(ir.U32, tid, c(int64(BM)))
		if BM*t.KB < nth {
			aSb0 = b.Rem(ir.U32, aSb0, c(int64(t.KB)))
		}
		for i := range aC {
			sb := int64(i * (nth / BM))
			aC[i].g, aC[i].sb = sb*subStride, sb
		}
	case BM*t.KB < nth:
		item := b.Rem(ir.U32, tid, c(int64(BM*t.KB)))
		aRow0, aSb0 = b.Rem(ir.U32, item, c(int64(BM))), b.Div(ir.U32, item, c(int64(BM)))
	default:
		return nil, fmt.Errorf("compose: Gemm: %d rows and %d threads do not nest", BM, nth)
	}
	aRowG := b.Add(ir.U32, rowBlk, aRow0)
	aGBase := b.Add(ir.U32, b.Mul(ir.U32, aSb0, c(subStride)), b.Mul(ir.U32, aRowG, c(rowStride)))
	aSBase := b.Mul(ir.U32, aRow0, c(int64(kHalf)))
	aSw := swzOf(aRow0, true)
	aAt := make([][]ir.Value, nA)
	for i := range aAt {
		aAt[i] = make([]ir.Value, subC)
		for q := range aAt[i] {
			ch := add(b.Mul(ir.U32, aSb0, c(int64(subC))), aC[i].sb*int64(subC)+int64(q))
			aAt[i][q] = b.Add(ir.U32, aSBase, b.Shl(ir.U32, b.Xor(ir.U32, aSw, ch), c(2)))
		}
	}

	// B staging: chunk e = tid + nth*i of the trip's slice is (token e/C,
	// chunk e%C).
	nBc := BN * C / nth
	bTok0 := b.Div(ir.U32, tid, c(int64(C)))
	bCh := b.Rem(ir.U32, tid, c(int64(C)))
	bGBase := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, tokBlk, bTok0), c(int64(m.K/2))), b.Shl(ir.U32, bCh, c(2)))
	bStep := int64(nth / C)
	bS := make([]ir.Value, nBc)
	bSOff := make([]int64, nBc)
	for i := range bS {
		tk := bTok0
		if bStep%4 != 0 {
			tk = add(bTok0, int64(i)*bStep)
		} else {
			bSOff[i] = int64(i) * bStep * int64(kHalf)
		}
		if i == 0 || bStep%4 != 0 {
			bS[i] = b.Add(ir.U32, b.Mul(ir.U32, tk, c(int64(kHalf))),
				b.Shl(ir.U32, b.Xor(ir.U32, bCh, swzOf(tk, false)), c(2)))
		} else {
			bS[i] = bS[0]
		}
	}

	// The fragment bases: warp (wm, wn), the lane's A row and B token.
	warp := b.Shr(ir.U32, tid, c(5))
	wm := b.Rem(ir.U32, warp, c(int64(t.WM)))
	wn := b.Div(ir.U32, warp, c(int64(t.WM)))
	v := kernels.NewVoltaLane(b, b.And(ir.U32, tid, c(31)))
	var rowW, tokW ir.Value
	var fragA, fragB []ir.Value
	var fragA16 [2][2]ir.Value
	var fragB16 [2]ir.Value
	var lq, lg ir.Value
	if t.F16K == 0 {
		q8 := b.Shl(ir.U32, v.Quad(), c(3))
		rowW = b.Add(ir.U32, b.Mul(ir.U32, wm, c(int64(32*t.MT))), q8)
		tokW = b.Mul(ir.U32, wn, c(int64(8*t.NT)))
		rowL := b.Add(ir.U32, rowW, v.Row())
		tokL := b.Add(ir.U32, tokW, v.Row())
		fragA, fragB = chunkAt(kernels.VoltaSwizzle(b, rowL, C, true)), chunkAt(kernels.VoltaSwizzle(b, tokL, C, false))
		baseA := b.Mul(ir.U32, rowL, c(int64(kHalf)))
		baseB := b.Mul(ir.U32, tokL, c(int64(kHalf)))
		for j := range fragA {
			fragA[j] = b.Add(ir.U32, baseA, fragA[j])
			fragB[j] = b.Add(ir.U32, baseB, fragB[j])
		}
	} else {
		lg = b.Shr(ir.U32, b.And(ir.U32, tid, c(31)), c(2))
		lq = b.And(ir.U32, b.And(ir.U32, tid, c(31)), c(3))
		rowW = b.Mul(ir.U32, wm, c(int64(16*t.MT)))
		tokW = b.Mul(ir.U32, wn, c(int64(8*t.NT)))
		frag := func(r ir.Value, weights bool) (out [2]ir.Value) {
			ch := b.Xor(ir.U32, lq, swzOf(r, weights))
			base := b.Mul(ir.U32, r, c(int64(kHalf)))
			out[0] = b.Add(ir.U32, base, b.Shl(ir.U32, ch, c(2)))
			if wide {
				out[1] = b.Add(ir.U32, base, b.Shl(ir.U32, b.Xor(ir.U32, ch, c(4)), c(2)))
			}
			return
		}
		for h := 0; h < 2; h++ {
			fragA16[h] = frag(add(b.Add(ir.U32, rowW, lg), int64(8*h)), true)
		}
		fragB16 = frag(b.Add(ir.U32, tokW, lg), false)
	}
	nAcc := 8
	if t.F16K != 0 {
		nAcc = 4
	}

	// One trip's global words into registers; staged after the dequant.
	load := func(W, Bc, sb ir.Value) (ra [][]ir.Value, rb [][]ir.Value) {
		ra = make([][]ir.Value, nA)
		wa := b.Add(ir.U32, W, aGBase)
		sbBase := b.Add(ir.U32, sb, aSb0)
		for i := range ra {
			ra[i] = append(dq.Words(wa, aC[i].g), dq.ScaleWords(add(sbBase, aC[i].sb), add(aRowG, aC[i].row))...)
		}
		rb = make([][]ir.Value, nBc)
		wb := b.Add(ir.U32, Bc, bGBase)
		for i := range rb {
			rb[i] = b.LoadV(ir.U32, pB, wb, int64(i)*bStep*int64(m.K/2), 4)
		}
		return
	}
	stage := func(ra, rb [][]ir.Value, sb, offA, offB ir.Value, x []ir.Value) {
		sbBase := b.Add(ir.U32, sb, aSb0)
		vals := make([][][]ir.Value, nA)
		for i := 0; i < nA; i++ {
			var xi ir.Value
			if carry {
				xi = x[i]
			}
			var s2, m2 ir.Value
			if !fl {
				s2, m2 = dq.ScalesOfX(ra[i][words:], xi, add(sbBase, aC[i].sb), add(aRowG, aC[i].row))
			}
			vals[i] = dq.Frags(ra[i][:words], s2, m2)
		}
		for i := 0; i < nA; i++ {
			f := vals[i]
			for q := 0; q < subC; q++ {
				b.StoreV(shA, b.Add(ir.U32, aAt[i][q], offA), aC[i].s, f[2*q][0], f[2*q][1], f[2*q+1][0], f[2*q+1][1])
			}
		}
		for i := 0; i < nBc; i++ {
			b.StoreV(shB, b.Add(ir.U32, bS[i], offB), bSOff[i], rb[i]...)
		}
	}
	last := c(0)
	advance := func(sb, W, Bc ir.Value) (ir.Value, ir.Value, ir.Value) {
		n := b.Min(ir.U32, add(sb, int64(t.KB)), last)
		delta := b.Sub(ir.U32, n, sb)
		return n, b.Add(ir.U32, W, b.Mul(ir.U32, delta, c(subStride))),
			b.Add(ir.U32, Bc, b.Mul(ir.U32, delta, c(int64(sub/2))))
	}
	sub0 := b.Mul(ir.U32, seg, c(int64(trips*t.KB)))
	last = b.Add(ir.U32, sub0, c(int64((trips-1)*t.KB)))
	w0 := b.Mul(ir.U32, sub0, c(subStride))
	b0 := b.Mul(ir.U32, sub0, c(int64(sub/2)))
	sizeA, sizeB := c(int64(BM*kHalf)), c(int64(BN*kHalf))
	nextX := func(x []ir.Value, ra [][]ir.Value, sb ir.Value) []ir.Value {
		if !carry {
			return nil
		}
		out := make([]ir.Value, nA)
		for i := range out {
			out[i] = dq.ScNext(x[i], ra[i][words+1], add(b.Add(ir.U32, sb, aSb0), aC[i].sb), t.KB)
		}
		return out
	}
	var x0 []ir.Value
	if carry {
		for i := 0; i < nA; i++ {
			x0 = append(x0, dq.ScFirst(add(b.Add(ir.U32, sub0, aSb0), aC[i].sb), add(aRowG, aC[i].row)))
		}
	}
	ra0, rb0 := load(w0, b0, sub0)
	stage(ra0, rb0, sub0, c(0), c(0), x0)
	sub1, w1, b1 := advance(sub0, w0, b0)
	x1 := nextX(x0, ra0, sub0)
	ra1, rb1 := load(w1, b1, sub1)
	b.Barrier()

	zeroF := b.ConstF32(0)
	zero := c(0)
	b.Loop(int64(trips))
	accs := make([][][]ir.Value, t.MT)
	for i := range accs {
		accs[i] = make([][]ir.Value, t.NT)
		for j := range accs[i] {
			z := make([]ir.Value, nAcc)
			for x := range z {
				z[x] = b.Phi(ir.F32, zeroF)
			}
			accs[i][j] = z
		}
	}
	curA := b.Phi(ir.U32, zero)
	curB := b.Phi(ir.U32, zero)
	subR := b.Phi(ir.U32, sub1)
	WR := b.Phi(ir.U32, w1)
	BR := b.Phi(ir.U32, b1)
	phis := func(init [][]ir.Value) [][]ir.Value {
		out := make([][]ir.Value, len(init))
		for i := range init {
			out[i] = make([]ir.Value, len(init[i]))
			for w := range init[i] {
				out[i][w] = b.Phi(ir.U32, init[i][w])
			}
		}
		return out
	}
	ra, rb := phis(ra1), phis(rb1)
	xR := make([]ir.Value, len(x1))
	for i := range x1 {
		xR[i] = b.Phi(ir.U32, x1[i])
	}
	d := make([][][]ir.Value, t.MT)
	for i := range d {
		d[i] = make([][]ir.Value, t.NT)
		copy(d[i], accs[i])
	}
	for ks := 0; ks < C/4 && t.F16K != 0; ks++ {
		p, off := ks&1, int64(32*(ks>>1))
		pa0, pa1 := b.Add(ir.U32, fragA16[0][p], curA), b.Add(ir.U32, fragA16[1][p], curA)
		pb := b.Add(ir.U32, fragB16[p], curB)
		a0 := make([][]ir.Value, t.MT)
		a1 := make([][]ir.Value, t.MT)
		for i := range a0 {
			a0[i] = b.LoadV(ir.U32, shA, pa0, int64(16*i*kHalf)+off, 4)
			a1[i] = b.LoadV(ir.U32, shA, pa1, int64(16*i*kHalf)+off, 4)
		}
		bf := make([][]ir.Value, t.NT)
		for j := range bf {
			bf[j] = b.LoadV(ir.U32, shB, pb, int64(8*j*kHalf)+off, 4)
		}
		for st := 0; st < 2; st++ {
			e0, e1 := 2*st, 2*st+1
			for i := 0; i < t.MT; i++ {
				for j := 0; j < t.NT; j++ {
					if t.F16K == 16 {
						d[i][j] = b.MMA(ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16},
							[]ir.Value{a0[i][e0], a1[i][e0], a0[i][e1], a1[i][e1]},
							[]ir.Value{bf[j][e0], bf[j][e1]}, d[i][j])
						continue
					}
					for _, w := range []int{e0, e1} {
						d[i][j] = b.MMA(ir.MMAShape{M: 16, N: 8, K: 8, Kind: ir.MMAF16},
							[]ir.Value{a0[i][w], a1[i][w]}, []ir.Value{bf[j][w]}, d[i][j])
					}
				}
			}
		}
	}
	for ch := 0; ch < C && t.F16K == 0; ch++ {
		pa, pb := b.Add(ir.U32, fragA[ch], curA), b.Add(ir.U32, fragB[ch], curB)
		af := make([][]ir.Value, t.MT)
		for i := range af {
			af[i] = b.LoadV(ir.U32, shA, pa, int64(32*i*kHalf), 4)
		}
		bf := make([][]ir.Value, t.NT)
		for j := range bf {
			bf[j] = b.LoadV(ir.U32, shB, pb, int64(8*j*kHalf), 4)
		}
		for x := 0; x < 2; x++ {
			for i := 0; i < t.MT; i++ {
				for j := 0; j < t.NT; j++ {
					d[i][j] = b.MMA(ir.MMAVolta(), af[i][2*x:2*x+2], bf[j][2*x:2*x+2], d[i][j])
				}
			}
		}
	}
	nxA, nxB := b.Sub(ir.U32, sizeA, curA), b.Sub(ir.U32, sizeB, curB)
	if t.Single {
		nxA, nxB = zero, zero
		b.Barrier()
	}
	stage(ra, rb, subR, nxA, nxB, xR)
	subN, WN, BN2 := advance(subR, WR, BR)
	na, nb := load(WN, BN2, subN)
	b.Barrier()
	for i := 0; i < t.MT; i++ {
		for j := 0; j < t.NT; j++ {
			for x := 0; x < nAcc; x++ {
				b.SetPhi(accs[i][j][x], d[i][j][x])
			}
		}
	}
	b.SetPhi(curA, nxA)
	b.SetPhi(curB, nxB)
	xN := nextX(xR, ra, subR)
	b.SetPhi(subR, subN)
	for i := range xR {
		b.SetPhi(xR[i], xN[i])
	}
	b.SetPhi(WR, WN)
	b.SetPhi(BR, BN2)
	for i := range ra {
		for w := range ra[i] {
			b.SetPhi(ra[i][w], na[i][w])
		}
	}
	for i := range rb {
		for w := range rb[i] {
			b.SetPhi(rb[i][w], nb[i][w])
		}
	}
	b.EndLoop()

	// The store: out[token][row], a split's segment at its plane.
	outSeg := b.Mul(ir.U32, seg, c(int64(m.NTok*m.Rows)))
	rowG := b.Add(ir.U32, rowBlk, rowW)
	tokG := b.Add(ir.U32, tokBlk, tokW)
	var rowOf, tokOf func(comp int) ir.Value
	if t.F16K == 0 {
		dRow := []ir.Value{b.Add(ir.U32, rowG, v.DRow(0)), b.Add(ir.U32, rowG, v.DRow(2))}
		dCol := []ir.Value{b.Add(ir.U32, tokG, v.DCol(0)), b.Add(ir.U32, tokG, v.DCol(1)),
			b.Add(ir.U32, tokG, v.DCol(4)), b.Add(ir.U32, tokG, v.DCol(5))}
		rowOf = func(comp int) ir.Value { return dRow[(comp>>1)&1] }
		tokOf = func(comp int) ir.Value { return dCol[(comp&1)+2*(comp>>2)] }
	} else {
		r0 := b.Add(ir.U32, rowG, lg)
		dRow := []ir.Value{r0, add(r0, 8)}
		t0 := b.Add(ir.U32, tokG, b.Shl(ir.U32, lq, c(1)))
		dCol := []ir.Value{t0, add(t0, 1)}
		rowOf = func(comp int) ir.Value { return dRow[comp>>1] }
		tokOf = func(comp int) ir.Value { return dCol[comp&1] }
	}
	mRows := 32
	if t.F16K != 0 {
		mRows = 16
	}
	nrows := c(int64(m.Rows))
	for i := 0; i < t.MT; i++ {
		for j := 0; j < t.NT; j++ {
			for comp := 0; comp < nAcc; comp++ {
				rr := add(rowOf(comp), int64(mRows*i))
				tok := add(tokOf(comp), int64(8*j))
				val := d[i][j][comp]
				at := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, tok, nrows), rr), outSeg)
				if m.Epi != nil {
					val = m.Epi.Apply(b, val, at, rr)
				}
				b.Store(pOut, at, val, 0)
			}
		}
	}
	return b.Done(), nil
}

// Operand is the row group that writes this GEMM's binary16 operand from the
// chunk's float rows (pX, NTok rows of K; pB, its words): a row a token, a
// column a 32-bit word packing two elements in the order the staged
// sub-blocks are read -- word 2s+h of a sub-block holds elements base+h and
// base+h+2, base the step's first (four apart in a byte-ordered format, else
// alternating halves of the sub-block). Two indexed loads, a pack and a store.
func (m *Gemm) Operand() *Row {
	sub, bits, _, fbytes, _ := kernels.GemmFormat(m.T)
	g := &Row{Width: m.K / 2, Rows: m.NTok}
	src, out := g.Buf(BufRow), g.Buf(BufOutU32)
	u := func(n int) int { return g.UConst(n) }
	w := g.Column()
	sb, wr := g.UBin(OpDiv, w, u(sub/2)), g.UBin(OpRem, w, u(sub/2))
	s, h := g.UBin(OpDiv, wr, u(2)), g.UBin(OpRem, wr, u(2))
	var base int
	if bits == 8 || fbytes == 2 {
		base = g.UBin(OpMul, s, u(4))
	} else {
		base = g.UBin(OpAdd, g.UBin(OpMul, g.UBin(OpDiv, s, u(2)), u(4)), g.UBin(OpMul, g.UBin(OpRem, s, u(2)), u(sub/2)))
	}
	e0 := g.UBin(OpAdd, g.UBin(OpAdd, g.UBin(OpMul, sb, u(sub)), base), h)
	row := g.UBin(OpMul, g.RowIdx(), u(m.K))
	x0 := g.LoadAt(src, g.UBin(OpAdd, row, e0))
	x1 := g.LoadAt(src, g.UBin(OpAdd, row, g.UBin(OpAdd, e0, u(2))))
	g.Store(out, g.Bin(OpPackF16, x0, x1))
	return g
}
