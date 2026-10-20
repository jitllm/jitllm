package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// GemmTile is a batched matvec of a packed weight on the collective 8x8
// binary16 tiles (ir.OpTileMMA) with both operands staged through workgroup
// memory: a workgroup of WM*WN subgroups covers WM*MT*8 rows by WN*NT*8
// tokens, each subgroup MT 8-row tiles (16-row where Tall) by NT 8-token
// tiles; a trip stages KB sub-blocks of k, the weights dequantized to
// binary16 and the tokens' binary16 operand (ActF16T) as it is, a trip
// ahead in a second buffer. With Experts above one it is a grouped mixture:
// a token block's expert at Sel, the token block varying slowest. TokOff is
// a dense launch's first token, a tail's after a wider block's lead.
type GemmTile struct {
	T                  kernels.Quant
	Rows, K, NTok      int
	MT, NT, WM, WN, KB int
	Tall               bool
	Experts            int
	TokOff             int
}

// Toks is the tokens a workgroup covers.
func (s GemmTile) Toks() int { return s.toks() }

// BlockRows is the rows a workgroup covers.
func (s GemmTile) BlockRows() int { return s.rows() }

func (s GemmTile) rows() int { return s.WM * s.MT * 8 }
func (s GemmTile) toks() int { return s.WN * s.NT * 8 }

// Threads is the workgroup's threads.
func (s GemmTile) Threads() int { return ir.SubgroupLanes * s.WM * s.WN }

// Groups is the workgroups over the whole shape.
func (s GemmTile) Groups() int { return s.Rows / s.rows() * (s.NTok / s.toks()) }

// tilePad is the half-words after every staged row: the eight rows one tile
// load reads land four banks apart, every vector access 16-byte aligned.
const tilePad = 8

// GemmTile forms s over the group (Threads() elements, Groups() rows, a
// row a workgroup): qs, d and sc the weight's planes, b the tokens'
// binary16 operand (token-major, K/2 words a token), out [token][row] as
// f32 tiles, bias (-1 for none) [row] or [expert][row], sel (-1 dense) a
// token block's expert.
func (g *Row) GemmTile(s GemmTile, qs, d, sc, b, out, bias, sel int) error {
	info := s.T.Info()
	if info.BlockE == 0 || info.Float > 0 || info.IntAct || info.D32 || info.Codes != nil && !info.E8M0 ||
		info.Bits != 4 && (info.Bits != 8 || info.Hi != 0) || kernels.DeviceWhyNot(s.T) != "" {
		return fmt.Errorf("compose: a tile GEMM over %v has no node form yet", s.T)
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
	grouped := s.Experts > 1
	BM, BN, nth := s.rows(), s.toks(), s.Threads()
	sub := info.Sub
	if s.MT < 1 || s.NT < 1 || s.WM < 1 || s.WN < 1 || s.KB < 1 || s.Tall && s.MT%2 != 0 ||
		s.Rows%BM != 0 || s.NTok%BN != 0 || s.K%info.BlockE != 0 || (s.K/sub)%s.KB != 0 {
		return fmt.Errorf("compose: a tile GEMM's %dx%d block over rows %d, tokens %d, k %d", BM, BN, s.Rows, s.NTok, s.K)
	}
	nsub := s.K / sub
	trips := nsub / s.KB
	pw, hw := sub*info.Bits/32, sub*info.Hi/32
	words := pw + hw
	KT := s.KB * sub
	C, subC := KT/8, sub/8
	RS := (KT + tilePad) / 2
	if KT%8 != 0 || subC < 1 || 2*(BM+BN)*RS*4 > 32<<10 {
		return fmt.Errorf("compose: a tile GEMM staging %d halves a trip", KT)
	}
	nA := max(BM*s.KB/nth, 1)
	dup := BM*s.KB < nth
	if dup && nth%BM != 0 || !dup && ((BM*s.KB)%nth != 0 || nth%BM != 0 && BM%nth != 0) || (BN*C)%nth != 0 || nth%C != 0 {
		return fmt.Errorf("compose: a tile GEMM of %d threads over %d rows by %d sub-blocks", nth, BM, s.KB)
	}
	parts := 1
	if dup && info.Hi == 0 {
		if p := nth / (BM * s.KB); nth%(BM*s.KB) == 0 && pw%p == 0 && subC%p == 0 {
			parts = p
		}
	}
	pwP, subCP := pw/parts, subC/parts
	lw := pwP + hw
	shA, shB := g.Shared(2*BM*RS, true), g.Shared(2*BN*RS, true)

	mB, nB := s.Rows/BM, s.NTok/BN
	grp := g.RowIdx()
	var bn, bm int
	if grouped {
		bm = ub(OpRem, grp, u(mB))
		bn = ub(OpMin, ub(OpDiv, grp, u(mB)), u(nB-1))
	} else {
		bn = ub(OpRem, grp, u(nB))
		bm = ub(OpRem, ub(OpDiv, grp, u(nB)), u(mB))
	}
	if grouped && s.TokOff != 0 {
		return fmt.Errorf("compose: a grouped tile GEMM at token %d", s.TokOff)
	}
	rowBlk, tokBlk := mul(bm, BM), addc(mul(bn, BN), s.TokOff)
	tid := g.Column()
	var qBase, dBase, scBase int = -1, -1, -1
	if grouped {
		nq, nd, nsc, err := kernels.PackedWords(s.T, s.Rows, s.K)
		if err != nil {
			return err
		}
		e := g.LoadAt(sel, bn)
		qBase, dBase = mul(e, nq), mul(e, nd)
		if nsc > 0 {
			scBase = mul(e, nsc)
		}
	}
	plus := func(x, base int) int {
		if base < 0 {
			return x
		}
		return add(x, base)
	}

	type aItem struct{ g, s, row, sb int }
	aC := make([]aItem, nA)
	var aRow0, aSb0 int
	aPart := u(0)
	if BM%nth == 0 {
		per := BM / nth
		aRow0, aSb0 = tid, u(0)
		for i := range aC {
			r, sb := nth*(i%per), i/per
			aC[i] = aItem{g: sb*words*s.Rows + r, s: r * RS, row: r, sb: sb}
		}
	} else {
		aRow0, aSb0 = ub(OpRem, tid, u(BM)), ub(OpDiv, tid, u(BM))
		if dup {
			aSb0 = ub(OpRem, aSb0, u(s.KB))
		}
		if parts > 1 {
			aPart = ub(OpDiv, tid, u(BM*s.KB))
		}
		for i := range aC {
			sb := i * (nth / BM)
			aC[i] = aItem{g: sb * words * s.Rows, sb: sb}
		}
	}
	aRowG := add(rowBlk, aRow0)
	aGBase := add(mul(aSb0, words*s.Rows), aRowG)
	if parts > 1 {
		aGBase = add(aGBase, mul(aPart, pwP*s.Rows))
	}
	aSBase := add(mul(aRow0, RS), mul(aSb0, 4*subC))
	if parts > 1 {
		aSBase = add(aSBase, mul(aPart, 4*subCP))
	}

	nBc := BN * C / nth
	bTok0, bCh := ub(OpDiv, tid, u(C)), ub(OpRem, tid, u(C))
	bGBase := add(mul(add(tokBlk, bTok0), s.K/2), ub(OpShl, bCh, u(2)))
	bStep := nth / C
	bSBase := add(mul(bTok0, RS), ub(OpShl, bCh, u(2)))

	slots := kernels.DSlots(s.T)
	carry := kernels.ScStream(s.T) && s.KB*12 <= 32
	scLast := kernels.ScStreamWords*s.K/info.BlockE - 1
	scAt := func(wi, row int) int { return g.LoadAt(sc, plus(add(mul(wi, s.Rows), row), scBase)) }
	scaleWords := func(subIdx, row int) []int {
		super := subIdx
		if info.PerSuper > 1 {
			super = ub(OpShr, subIdx, u(log2(info.PerSuper)))
		}
		var o []int
		if slots > 1 {
			idx := add(mul(super, (s.Rows+slots-1)/slots), ub(OpShr, row, u(log2(slots))))
			o = append(o, g.LoadAt(d, plus(idx, dBase)))
		} else {
			o = append(o, g.LoadAt(d, plus(add(mul(super, s.Rows), row), dBase)))
		}
		switch {
		case carry:
			o = append(o, scAt(ub(OpMin, addc(ub(OpShr, mul(subIdx, 12), u(5)), 1), u(scLast)), row))
		case kernels.ScStream(s.T):
			o = append(o, scAt(ub(OpShr, mul(subIdx, 12), u(5)), row))
		case info.PerSuper > 1:
			per := 4
			if info.BiasArray {
				per = 2
			}
			o = append(o, scAt(ub(OpShr, subIdx, u(log2(per))), row))
		}
		return o
	}
	zero := g.Const(0)
	float := func(n int) int { return g.Add(Node{Kind: NUnary, Op: OpFloat, A: n}) }
	half := func(n int) int { return g.Add(Node{Kind: NUnary, Op: OpHalf, A: n}) }
	pack := func(a, b int) int { return g.Bin(OpPackF16, a, b) }
	scales := func(w []int, x, subIdx, row, z int) (int, int) {
		var dv, dm int
		if slots > 1 {
			sh := ub(OpMul, ub(OpAnd, row, u(slots-1)), u(32/slots))
			if info.E8M0 {
				dv = g.Add(Node{Kind: NUnary, Op: OpBits, A: ub(OpShl, ub(OpAnd, ub(OpShr, w[0], sh), u(0xFF)), u(23))})
				return pack(dv, dv), pack(z, z)
			}
			dv = half(ub(OpShr, w[0], sh))
		} else {
			dv = half(w[0])
			if info.BiasArray {
				dm = half(ub(OpShr, w[0], u(16)))
			}
		}
		scv, mn := dv, z
		switch {
		case carry:
			bit := ub(OpAnd, mul(subIdx, 12), u(31))
			pr := g.Add(Node{Kind: NUBinary, Op: OpFunnel, A: x, B: w[1], C: bit})
			scv = g.Bin(OpMul, dv, float(ub(OpAnd, pr, u(63))))
			mn = g.Bin(OpMul, dm, float(ub(OpAnd, ub(OpShr, pr, u(6)), u(63))))
		case kernels.ScStream(s.T):
			pos := mul(subIdx, 12)
			bit := ub(OpAnd, pos, u(31))
			hi := add(ub(OpShr, pos, u(5)), g.Select(g.Less(u(20), bit), u(1), u(0)))
			hwv := scAt(hi, row)
			pr := g.Add(Node{Kind: NUBinary, Op: OpFunnel, A: w[1], B: hwv, C: bit})
			scv = g.Bin(OpMul, dv, float(ub(OpAnd, pr, u(63))))
			mn = g.Bin(OpMul, dm, float(ub(OpAnd, ub(OpShr, pr, u(6)), u(63))))
		case info.PerSuper > 1:
			per, stride := 4, 8
			if info.BiasArray {
				per, stride = 2, 16
			}
			base := ub(OpMul, ub(OpAnd, subIdx, u(per-1)), u(stride))
			scI := ub(OpAnd, ub(OpShr, w[1], base), u(0xFF))
			if info.ScOff != 0 {
				scI = ub(OpSub, scI, g.Add(Node{Kind: NIConst, At: -info.ScOff}))
			}
			scv = g.Bin(OpMul, dv, float(scI))
			if info.BiasArray {
				mn = g.Bin(OpMul, dm, float(ub(OpAnd, ub(OpShr, w[1], add(base, u(8))), u(0xFF))))
			}
		case info.BiasArray:
			mn = dm
		}
		if !info.BiasArray && info.BiasK != 0 {
			mn = g.Bin(OpMul, g.Const(info.BiasK), scv)
		}
		neg := g.Bin(OpSub, z, mn)
		return pack(scv, scv), pack(neg, neg)
	}
	magic, m4, m8 := u(0x64006400), u(0x000F000F), u(0x00FF00FF)
	off := 0x6400
	if info.Bits == 8 {
		off = 0x6480
	}
	off2 := u(off | off<<16)
	frags := func(raw []int, s2, m2 int) [][2]int {
		deq := func(p int) int { return g.Add(Node{Kind: NBinary, Op: OpFmaH2, A: p, B: s2, C: m2}) }
		pair := func(w, sh, sw, hsh int) int {
			x := w
			if sh != 0 {
				x = ub(OpShr, w, u(sh))
			}
			if info.Bits == 8 {
				return g.Bin(OpSubH2, ub(OpXor, ub(OpAnd, x, m8), magic), off2)
			}
			p := ub(OpAnd, x, m4)
			if sw >= 0 {
				y := sw
				if hsh != 0 {
					y = ub(OpShr, sw, u(hsh))
				}
				hm := 1<<info.Hi - 1
				p = ub(OpXor, p, ub(OpShl, ub(OpAnd, y, u(hm|hm<<16)), u(4)))
			}
			return g.Bin(OpSubH2, ub(OpXor, p, magic), off2)
		}
		var o [][2]int
		switch {
		case info.E8M0:
			k15 := u(0x78007800)
			e2m1 := func(w, sh int) int {
				x := w
				if sh != 0 {
					x = ub(OpShr, w, u(sh))
				}
				p := ub(OpAnd, x, m4)
				h := ub(OpXor, ub(OpShl, ub(OpAnd, p, u(0x00070007)), u(9)), ub(OpShl, ub(OpAnd, p, u(0x00080008)), u(12)))
				return deq(g.Add(Node{Kind: NBinary, Op: OpFmaH2, A: h, B: k15, C: u(0)}))
			}
			for w := 0; w < pwP; w++ {
				for hf := 0; hf < 2; hf++ {
					o = append(o, [2]int{e2m1(raw[w], 4*hf), e2m1(raw[w], 4*hf+8)})
				}
			}
		case info.Bits == 8:
			for w := 0; w < pwP; w++ {
				x := ub(OpXor, raw[w], u(0x80808080))
				o = append(o, [2]int{deq(pair(x, 0, -1, 0)), deq(pair(x, 8, -1, 0))})
			}
		default:
			lanes := 0
			if info.Hi > 0 {
				lanes = 8 / info.Hi
			}
			h := sub / 2
			for w := 0; w < pwP; w++ {
				for hf := 0; hf < 2; hf++ {
					sw, hsh := -1, 0
					if hw > 0 {
						gi := w + hf*(h/4)
						sw, hsh = raw[pwP+gi/lanes], info.Hi*(gi%lanes)
					}
					o = append(o, [2]int{deq(pair(raw[w], 4*hf, sw, hsh)), deq(pair(raw[w], 4*hf+8, sw, hsh+8))})
				}
			}
		}
		return o
	}
	load := func(W, Bc, sb int) (ra, rb [][]int) {
		wa := add(W, aGBase)
		sbBase := add(sb, aSb0)
		for _, a := range aC {
			var ws []int
			for w := 0; w < lw; w++ {
				ws = append(ws, g.LoadAt(qs, plus(addc(wa, a.g+w*s.Rows), qBase)))
			}
			ra = append(ra, append(ws, scaleWords(addc(sbBase, a.sb), addc(aRowG, a.row))...))
		}
		wb := add(Bc, bGBase)
		for i := 0; i < nBc; i++ {
			at := addc(wb, i*bStep*(s.K/2))
			rb = append(rb, []int{g.LoadVec(b, at, 4, 0), g.LoadVec(b, at, 4, 1), g.LoadVec(b, at, 4, 2), g.LoadVec(b, at, 4, 3)})
		}
		return
	}
	stage := func(ra, rb [][]int, sb, offA, offB, after, in int, x []int) int {
		sbBase := add(sb, aSb0)
		z := zero
		if in >= 0 {
			z = g.After(zero, in)
		}
		tok := after
		vals := make([][][2]int, nA)
		for i, a := range aC {
			xi := -1
			if carry {
				xi = x[i]
			}
			s2, m2 := scales(ra[i][lw:], xi, addc(sbBase, a.sb), addc(aRowG, a.row), z)
			vals[i] = frags(ra[i][:lw], s2, m2)
		}
		baseA := add(aSBase, offA)
		for i, a := range aC {
			f := vals[i]
			for q := 0; q < subCP; q++ {
				tok = g.WriteVec(shA, addc(baseA, a.s+(a.sb*subC+q)*4), tok, f[2*q][0], f[2*q][1], f[2*q+1][0], f[2*q+1][1])
			}
		}
		baseB := add(bSBase, offB)
		for i := 0; i < nBc; i++ {
			tok = g.WriteVec(shB, addc(baseB, i*bStep*RS), tok, rb[i]...)
		}
		return tok
	}
	last := u((trips - 1) * s.KB)
	advance := func(sb, W, Bc int) (int, int, int) {
		n := ub(OpMin, addc(sb, s.KB), last)
		delta := ub(OpSub, n, sb)
		return n, add(W, mul(delta, words*s.Rows)), add(Bc, mul(delta, sub/2))
	}
	nextX := func(x []int, ra [][]int, sb int) []int {
		if !carry {
			return nil
		}
		o := make([]int, nA)
		for i, a := range aC {
			bit := ub(OpAnd, mul(addc(add(sb, aSb0), a.sb), 12), u(31))
			o[i] = g.Select(g.Less(u(31-12*s.KB), bit), ra[i][lw+1], x[i])
		}
		return o
	}
	sub0, w0, b0 := u(0), u(0), u(0)
	var x0 []int
	if carry {
		for _, a := range aC {
			x0 = append(x0, scAt(ub(OpShr, mul(addc(add(sub0, aSb0), a.sb), 12), u(5)), addc(aRowG, a.row)))
		}
	}
	ra0, rb0 := load(w0, b0, sub0)
	tok := stage(ra0, rb0, sub0, u(0), u(0), -1, -1, x0)
	sub1, w1, b1 := advance(sub0, w0, b0)
	x1 := nextX(x0, ra0, sub0)
	ra1, rb1 := load(w1, b1, sub1)
	tok = g.Barrier(tok)

	sg := g.Subgroup()
	wm, wn := ub(OpRem, sg, u(s.WM)), ub(OpDiv, sg, u(s.WM))
	rowW, tokW := mul(wm, 8*s.MT), mul(wn, 8*s.NT)
	TM, MA := 8, s.MT
	if s.Tall {
		TM, MA = 16, s.MT/2
	}
	aT := ir.TileType{Rows: TM, Cols: 8, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF16, Use: ir.TileB}
	cT := ir.TileType{Rows: TM, Cols: 8, Elem: ir.TileF32, Use: ir.TileAcc}
	rowG, tokG := add(rowBlk, rowW), add(tokBlk, tokW)
	aH, bH, strideH := mul(rowW, 2*RS), mul(tokW, 2*RS), u(2*RS)
	init := make([][]int, MA)
	for i := range init {
		init[i] = make([]int, s.NT)
		for j := range init[i] {
			if bias >= 0 {
				bi := addc(rowG, TM*i)
				if grouped {
					bi = add(mul(g.LoadAt(sel, bn), s.Rows), bi)
				}
				init[i][j] = g.TileLoad(cT, bias, bi, u(0), true, -1)
			} else {
				init[i][j] = g.TileSplat(cT, zero)
			}
		}
	}

	c := g.Loop(trips)
	accs := make([][]int, MA)
	for i := range accs {
		accs[i] = make([]int, s.NT)
		for j := range accs[i] {
			accs[i][j] = g.Fold(c, init[i][j])
		}
	}
	curA, curB := g.Fold(c, u(0)), g.Fold(c, u(0))
	order := g.Fold(c, tok)
	subR, WR, BR := g.Fold(c, sub1), g.Fold(c, w1), g.Fold(c, b1)
	folds := func(init [][]int) [][]int {
		o := make([][]int, len(init))
		for i := range init {
			for _, v := range init[i] {
				o[i] = append(o[i], g.Fold(c, v))
			}
		}
		return o
	}
	ra, rb := folds(ra1), folds(rb1)
	xR := make([]int, len(x1))
	for i := range x1 {
		xR[i] = g.Fold(c, x1[i])
	}
	dd := make([][]int, MA)
	for i := range dd {
		dd[i] = append([]int(nil), accs[i]...)
	}
	baseAH, baseBH := add(ub(OpShl, curA, u(1)), aH), add(ub(OpShl, curB, u(1)), bH)
	for kk := 0; kk < KT/8; kk++ {
		av := make([]int, MA)
		for i := range av {
			av[i] = g.TileLoad(aT, shA, addc(baseAH, TM*i*2*RS+8*kk), strideH, false, order)
		}
		bv := make([]int, s.NT)
		for j := range bv {
			bv[j] = g.TileLoad(bT, shB, addc(baseBH, 8*j*2*RS+8*kk), strideH, true, order)
		}
		for i := 0; i < MA; i++ {
			for j := 0; j < s.NT; j++ {
				dd[i][j] = g.TileMMA(av[i], bv[j], dd[i][j])
			}
		}
	}
	nxA, nxB := ub(OpSub, u(BM*RS), curA), ub(OpSub, u(BN*RS), curB)
	bt := stage(ra, rb, subR, nxA, nxB, g.After(curA, order), curA, xR)
	xN := nextX(xR, ra, subR)
	subN, WN, BN2 := advance(subR, WR, BR)
	na, nb := load(WN, BN2, subN)
	bt = g.Barrier(bt)
	for i := range accs {
		for j := range accs[i] {
			g.SetFold(accs[i][j], dd[i][j])
		}
	}
	g.SetFold(order, bt)
	g.SetFold(curA, nxA)
	g.SetFold(curB, nxB)
	g.SetFold(subR, subN)
	g.SetFold(WR, WN)
	g.SetFold(BR, BN2)
	for i := range xR {
		g.SetFold(xR[i], xN[i])
	}
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
	for i := 0; i < MA; i++ {
		for j := 0; j < s.NT; j++ {
			at := add(mul(addc(tokG, 8*j), s.Rows), addc(rowG, TM*i))
			g.Effect(g.WriteTile(out, at, u(s.Rows), accs[i][j], cT, true, -1))
		}
	}
	return nil
}
