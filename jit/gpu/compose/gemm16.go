package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Gemm16 is a batched matvec of a packed weight on the binary16 m16n8
// matrix instruction with both operands staged through workgroup memory: a
// workgroup of WM*WN subgroups covers WM*MT*16 rows by WN*NT*8 tokens, each
// subgroup MT m-tiles by NT n-tiles; a trip stages KB sub-blocks of k, the
// weights dequantized once to binary16 and the tokens' binary16 operand as
// it is. The next trip's global words are loaded a trip ahead; with Single
// the stage takes one buffer, else two. Split cuts k into that many
// segments, each its own plane of the output.
type Gemm16 struct {
	T                  kernels.Quant
	Rows, K, NTok      int
	MT, NT, WM, WN, KB int
	K8, Single         bool
	Split              int
}

func (s Gemm16) rows() int    { return s.WM * s.MT * 16 }
func (s Gemm16) toks() int    { return s.WN * s.NT * 8 }
func (s Gemm16) Threads() int { return 32 * s.WM * s.WN }
func (s Gemm16) Groups() int  { return s.Rows / s.rows() * (s.NTok / s.toks()) * max(s.Split, 1) }

// Gemm16 forms s over the group (Threads() elements, Groups() rows, a row a
// workgroup): qs, d and sc the weight's planes (sc -1 where the format has
// none), b the tokens' binary16 operand (token-major, K/2 words a token),
// out the output, [segment][token][row].
func (g *Row) Gemm16(s Gemm16, qs, d, sc, b, out int) error {
	info := s.T.Info()
	if info.Bits != 8 || info.Hi != 0 || info.PerSuper != 1 || info.BiasK != 0 || info.BiasArray || info.Codes != nil ||
		info.E8M0 || info.D32 || info.Float > 0 || kernels.DSlots(s.T) != 2 {
		return fmt.Errorf("compose: a binary16 GEMM over %v has no node form yet", s.T)
	}
	u, ub := g.UConst, g.UBin
	add := func(x, y int) int { return ub(OpAdd, x, y) }
	addc := func(x, n int) int {
		if n == 0 {
			return x
		}
		return add(x, u(n))
	}
	mul := func(x, n int) int { return ub(OpMul, x, u(n)) }
	split := max(s.Split, 1)
	BM, BN, nth := s.rows(), s.toks(), s.Threads()
	sub := info.Sub
	nsub := s.K / sub
	if s.Rows%BM != 0 || s.NTok%BN != 0 || s.K%sub != 0 || nsub%(split*s.KB) != 0 {
		return fmt.Errorf("compose: a binary16 GEMM's %dx%d block over rows %d, tokens %d, k %d", BM, BN, s.Rows, s.NTok, s.K)
	}
	trips := nsub / split / s.KB
	words := sub * info.Bits / 32
	subStride := words * s.Rows
	kHalf := s.KB * sub / 2
	C, subC := kHalf/4, sub/8
	wide := C >= 8
	if C&(C-1) != 0 || C < 2 || C != 4 && C%8 != 0 || (BN*C)%nth != 0 || nth%C != 0 ||
		(BM*s.KB)%nth != 0 && BM*s.KB > nth {
		return fmt.Errorf("compose: a binary16 GEMM of %d chunks a trip over %d threads", C, nth)
	}
	nbuf := 2
	if s.Single {
		nbuf = 1
	}
	shA, shB := g.Shared(nbuf*BM*kHalf, true), g.Shared(nbuf*BN*kHalf, true)
	bit := func(r, i int) int { return ub(OpAnd, ub(OpShr, r, u(i)), u(1)) }
	swz := func(r int, weights bool) int {
		if wide {
			return add(ub(OpShl, ub(OpAnd, r, u(1)), u(2)), ub(OpAnd, ub(OpShr, r, u(1)), u(3)))
		}
		if !weights {
			if C == 4 {
				return bit(r, 1)
			}
			return u(0)
		}
		b23 := ub(OpXor, bit(r, 2), bit(r, 3))
		if C == 4 {
			return add(bit(r, 1), ub(OpShl, b23, u(1)))
		}
		return b23
	}

	mB, nB := s.Rows/BM, s.NTok/BN
	grp := g.RowIdx()
	bn := ub(OpRem, grp, u(nB))
	bm := ub(OpRem, ub(OpDiv, grp, u(nB)), u(mB))
	seg := ub(OpDiv, grp, u(mB*nB))
	rowBlk, tokBlk := mul(bm, BM), mul(bn, BN)
	tid := g.Column()

	nA := max(BM*s.KB/nth, 1)
	type item struct{ g, s, row, sb int }
	aC := make([]item, nA)
	var aRow0, aSb0 int
	switch {
	case BM%nth == 0:
		per := BM / nth
		aRow0, aSb0 = tid, u(0)
		for i := range aC {
			r, sb := nth*(i%per), i/per
			aC[i] = item{g: sb*subStride + r, s: r * kHalf, row: r, sb: sb}
		}
	case nth%BM == 0:
		aRow0, aSb0 = ub(OpRem, tid, u(BM)), ub(OpDiv, tid, u(BM))
		if BM*s.KB < nth {
			aSb0 = ub(OpRem, aSb0, u(s.KB))
		}
		for i := range aC {
			sb := i * (nth / BM)
			aC[i] = item{g: sb * subStride, sb: sb}
		}
	default:
		it := ub(OpRem, tid, u(BM*s.KB))
		aRow0, aSb0 = ub(OpRem, it, u(BM)), ub(OpDiv, it, u(BM))
	}
	aRowG := add(rowBlk, aRow0)
	aGBase := add(ub(OpMul, aSb0, u(subStride)), aRowG)
	aSBase := mul(aRow0, kHalf)
	aSw := swz(aRow0, true)
	aAt := make([][]int, nA)
	for i := range aAt {
		aAt[i] = make([]int, subC)
		for q := range aAt[i] {
			ch := addc(ub(OpMul, aSb0, u(subC)), aC[i].sb*subC+q)
			aAt[i][q] = add(aSBase, ub(OpShl, ub(OpXor, aSw, ch), u(2)))
		}
	}

	nBc := BN * C / nth
	bTok0, bCh := ub(OpDiv, tid, u(C)), ub(OpRem, tid, u(C))
	bGBase := add(mul(add(tokBlk, bTok0), s.K/2), ub(OpShl, bCh, u(2)))
	bStep := nth / C
	bS := make([]int, nBc)
	bSOff := make([]int, nBc)
	for i := range bS {
		tk := bTok0
		if bStep%4 != 0 {
			tk = addc(bTok0, i*bStep)
		} else {
			bSOff[i] = i * bStep * kHalf
		}
		if i == 0 || bStep%4 != 0 {
			bS[i] = add(mul(tk, kHalf), ub(OpShl, ub(OpXor, bCh, swz(tk, false)), u(2)))
		} else {
			bS[i] = bS[0]
		}
	}

	warp := ub(OpShr, tid, u(5))
	wm, wn := ub(OpRem, warp, u(s.WM)), ub(OpDiv, warp, u(s.WM))
	lane := ub(OpAnd, tid, u(31))
	lg, lq := ub(OpShr, lane, u(2)), ub(OpAnd, lane, u(3))
	rowW, tokW := mul(wm, 16*s.MT), mul(wn, 8*s.NT)
	frag := func(r int, weights bool) (o [2]int) {
		ch := ub(OpXor, lq, swz(r, weights))
		base := mul(r, kHalf)
		o[0] = add(base, ub(OpShl, ch, u(2)))
		if wide {
			o[1] = add(base, ub(OpShl, ub(OpXor, ch, u(4)), u(2)))
		}
		return
	}
	var fragA [2][2]int
	for h := 0; h < 2; h++ {
		fragA[h] = frag(addc(add(rowW, lg), 8*h), true)
	}
	fragB := frag(add(tokW, lg), false)

	// The weight's binary16 pairs: an 8-bit code flipped unsigned, two bytes
	// at a time into the mantissas of 1152.0, less it, scaled.
	magic, off2, m8 := u(0x64006400), u(0x64806480), u(0x00FF00FF)
	zero := g.Const(0)
	load := func(W, Bc, sb int) (ra, rb [][]int) {
		wa := add(W, aGBase)
		sbBase := add(sb, aSb0)
		for _, a := range aC {
			var ws []int
			for w := 0; w < words; w++ {
				ws = append(ws, g.LoadAt(qs, addc(wa, a.g+w*s.Rows)))
			}
			dRow := addc(aRowG, a.row)
			di := add(mul(addc(sbBase, a.sb), (s.Rows+1)/2), ub(OpShr, dRow, u(1)))
			ra = append(ra, append(ws, g.LoadAt(d, di)))
		}
		wb := add(Bc, bGBase)
		for i := 0; i < nBc; i++ {
			at := addc(wb, i*bStep*(s.K/2))
			rb = append(rb, []int{g.LoadVec(b, at, 4, 0), g.LoadVec(b, at, 4, 1), g.LoadVec(b, at, 4, 2), g.LoadVec(b, at, 4, 3)})
		}
		return
	}
	stage := func(ra, rb [][]int, offA, offB, after, in int) int {
		vals := make([][][2]int, nA)
		for i, a := range aC {
			dRow := addc(aRowG, a.row)
			sh := ub(OpMul, ub(OpAnd, dRow, u(1)), u(16))
			dv := g.Add(Node{Kind: NUnary, Op: OpHalf, A: ub(OpShr, ra[i][words], sh)})
			z := zero
			if in >= 0 {
				z = g.After(zero, in)
			}
			neg := g.Bin(OpSub, z, z)
			s2, m2 := g.Bin(OpPackF16, dv, dv), g.Bin(OpPackF16, neg, neg)
			for w := 0; w < words; w++ {
				x := ub(OpXor, ra[i][w], u(0x80808080))
				pair := func(sh int) int {
					y := x
					if sh != 0 {
						y = ub(OpShr, x, u(sh))
					}
					return g.Bin(OpSubH2, ub(OpXor, ub(OpAnd, y, m8), magic), off2)
				}
				deq := func(p int) int { return g.Add(Node{Kind: NBinary, Op: OpFmaH2, A: p, B: s2, C: m2}) }
				vals[i] = append(vals[i], [2]int{deq(pair(0)), deq(pair(8))})
			}
		}
		tok := after
		for i, a := range aC {
			f := vals[i]
			for q := 0; q < subC; q++ {
				tok = g.WriteVec(shA, addc(add(aAt[i][q], offA), a.s), tok, f[2*q][0], f[2*q][1], f[2*q+1][0], f[2*q+1][1])
			}
		}
		for i := 0; i < nBc; i++ {
			tok = g.WriteVec(shB, addc(add(bS[i], offB), bSOff[i]), tok, rb[i]...)
		}
		return tok
	}
	advance := func(sb, last, W, Bc int) (int, int, int) {
		n := ub(OpMin, addc(sb, s.KB), last)
		delta := ub(OpSub, n, sb)
		return n, add(W, mul(delta, subStride)), add(Bc, mul(delta, sub/2))
	}
	sub0 := mul(seg, trips*s.KB)
	last := addc(sub0, (trips-1)*s.KB)
	w0, b0 := mul(sub0, subStride), mul(sub0, sub/2)
	ra0, rb0 := load(w0, b0, sub0)
	tok := g.Barrier(stage(ra0, rb0, u(0), u(0), -1, -1))
	sub1, w1, b1 := advance(sub0, last, w0, b0)
	ra1, rb1 := load(w1, b1, sub1)

	c := g.Loop(trips)
	accs := make([][][]int, s.MT)
	for i := range accs {
		accs[i] = make([][]int, s.NT)
		for j := range accs[i] {
			z := make([]int, 4)
			for x := range z {
				z[x] = g.Fold(c, zero)
			}
			accs[i][j] = z
		}
	}
	curA, curB := g.Fold(c, u(0)), g.Fold(c, u(0))
	order := g.Fold(c, tok)
	subR, WR, BR := g.Fold(c, sub1), g.Fold(c, w1), g.Fold(c, b1)
	folds := func(init [][]int) [][]int {
		out := make([][]int, len(init))
		for i := range init {
			for _, v := range init[i] {
				out[i] = append(out[i], g.Fold(c, v))
			}
		}
		return out
	}
	ra, rb := folds(ra1), folds(rb1)

	dd := make([][][]int, s.MT)
	for i := range dd {
		dd[i] = append([][]int(nil), accs[i]...)
	}
	sh16 := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	sh8 := ir.MMAShape{M: 16, N: 8, K: 8, Kind: ir.MMAF16}
	ld := func(buf, at, off int) []int {
		at = g.After(addc(at, off), order)
		return []int{g.LoadVec(buf, at, 4, 0), g.LoadVec(buf, at, 4, 1), g.LoadVec(buf, at, 4, 2), g.LoadVec(buf, at, 4, 3)}
	}
	for ks := 0; ks < C/4; ks++ {
		p, off := ks&1, 32*(ks>>1)
		pa0, pa1, pb := add(fragA[0][p], curA), add(fragA[1][p], curA), add(fragB[p], curB)
		a0, a1 := make([][]int, s.MT), make([][]int, s.MT)
		for i := range a0 {
			a0[i], a1[i] = ld(shA, pa0, 16*i*kHalf+off), ld(shA, pa1, 16*i*kHalf+off)
		}
		bf := make([][]int, s.NT)
		for j := range bf {
			bf[j] = ld(shB, pb, 8*j*kHalf+off)
		}
		for st := 0; st < 2; st++ {
			x0, x1 := 2*st, 2*st+1
			for i := 0; i < s.MT; i++ {
				for j := 0; j < s.NT; j++ {
					if !s.K8 {
						dd[i][j] = g.MMA(sh16, []int{a0[i][x0], a1[i][x0], a0[i][x1], a1[i][x1]}, []int{bf[j][x0], bf[j][x1]}, dd[i][j])
						continue
					}
					for _, x := range []int{x0, x1} {
						dd[i][j] = g.MMA(sh8, []int{a0[i][x], a1[i][x]}, []int{bf[j][x]}, dd[i][j])
					}
				}
			}
		}
	}
	nxA, nxB := ub(OpSub, u(BM*kHalf), curA), ub(OpSub, u(BN*kHalf), curB)
	after := g.After(curA, order)
	if s.Single {
		nxA, nxB = u(0), u(0)
		after = g.Barrier(after)
	}
	bt := stage(ra, rb, nxA, nxB, after, curA)
	subN, WN, BN2 := advance(subR, last, WR, BR)
	na, nb := load(WN, BN2, subN)
	bt = g.Barrier(bt)
	for i := range accs {
		for j := range accs[i] {
			for x := range accs[i][j] {
				g.SetFold(accs[i][j][x], dd[i][j][x])
			}
		}
	}
	g.SetFold(order, bt)
	g.SetFold(curA, nxA)
	g.SetFold(curB, nxB)
	g.SetFold(subR, subN)
	g.SetFold(WR, WN)
	g.SetFold(BR, BN2)
	for i := range ra {
		for w := range ra[i] {
			g.SetFold(ra[i][w], na[i][w])
		}
	}
	for i := range rb {
		for w := range rb[i] {
			g.SetFold(rb[i][w], nb[i][w])
		}
	}
	outSeg := mul(seg, s.NTok*s.Rows)
	rowG, tokG := add(rowBlk, rowW), add(tokBlk, tokW)
	r0 := add(rowG, lg)
	t0 := add(tokG, ub(OpShl, lq, u(1)))
	dRow, dCol := [2]int{r0, addc(r0, 8)}, [2]int{t0, addc(t0, 1)}
	for i := 0; i < s.MT; i++ {
		for j := 0; j < s.NT; j++ {
			for x := 0; x < 4; x++ {
				rr := addc(dRow[x>>1], 16*i)
				tk := addc(dCol[x&1], 8*j)
				g.StoreAt(out, g.Last(accs[i][j][x]), add(add(mul(tk, s.Rows), rr), outSeg))
			}
		}
	}
	return nil
}

// ActF16T is rows of k floats at x as the binary16 operand Gemm16 reads at
// out, token-major: a (token, sub-block) of the format's an element of the
// group (Width tokens*k/sub, Rows 1), its values packed two to a word in the
// order the instruction's steps take them -- an 8-bit or binary16 format's
// in byte order, a narrower one's halves interleaved.
func (g *Row) ActF16T(t kernels.Quant, k, x, out int) error {
	info := t.Info()
	sub := info.Sub
	if k%sub != 0 || sub%8 != 0 {
		return fmt.Errorf("compose: a binary16 operand of k=%d over %v", k, t)
	}
	u, ub := g.UConst, g.UBin
	order := [4]int{0, 2, 1, 3}
	elem := func(s, j int) int { return (s/2)*4 + order[j] + (s%2)*(sub/2) }
	if info.Bits == 8 || info.Float == 2 {
		elem = func(s, j int) int { return 4*s + order[j] }
	}
	e := g.Column()
	in := ub(OpMul, e, u(sub))
	var v []int
	for o := 0; o < sub; o += 4 {
		at := ub(OpAdd, in, u(o))
		for i := 0; i < 4; i++ {
			v = append(v, g.LoadVec(x, at, 4, i))
		}
	}
	var w []int
	for s := 0; s < sub/4; s++ {
		w = append(w, g.Bin(OpPackF16, v[elem(s, 0)], v[elem(s, 1)]), g.Bin(OpPackF16, v[elem(s, 2)], v[elem(s, 3)]))
	}
	base := ub(OpMul, e, u(sub/2))
	for o := 0; o < len(w); o += 4 {
		g.Effect(g.WriteVec(out, ub(OpAdd, base, u(o)), -1, w[o:o+4]...))
	}
	return nil
}
