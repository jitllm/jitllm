package mainkernels

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// TileGemm is GemmTile's blocking: a workgroup of WM*WN subgroups, each MT
// 8-row tiles by NT 8-token tiles of the output, and KB sub-blocks of k staged
// per trip. The workgroup covers WM*MT*8 rows by WN*NT*8 tokens.
type TileGemm struct {
	MT, NT, WM, WN, KB int
	// NoSplit keeps the duplicate staging where a sub-block could split
	// between threads: the A/B arm and the bisection switch.
	NoSplit bool
	// Tall multiplies 16-row A tiles (and 16-row accumulators), two of the
	// MT 8-row slots each, where the device lowers a 16x8x8 product
	// (backend.LowersTile): mma.sync's own m16 shape, where 8-row tiles run it
	// with half its rows zero. The staging and the block are unchanged; MT
	// must be even. Metal's simdgroup_matrix is 8x8 only.
	Tall bool
	// TokOff is the first token this launch covers: a prompt whose token
	// count the widest block does not divide runs the widest block over the
	// leading tokens and a narrower one over the rest at an offset, both
	// reading and writing the whole buffers (s.NTok is the launch's own count).
	TokOff int
}

// Rows and Toks are the workgroup's block of the output.
func (t TileGemm) Rows() int { return t.WM * t.MT * 8 }
func (t TileGemm) Toks() int { return t.WN * t.NT * 8 }

// Threads is the workgroup width.
func (t TileGemm) Threads() int { return ir.SubgroupLanes * t.WM * t.WN }

// GemmTileGroups is how many workgroups GemmTile launches for a shape. A
// grouped shape launches GemmTileGrouped(s, t, used) instead.
func GemmTileGroups(s MatVecShape, t TileGemm) int {
	return (s.Rows / t.Rows()) * (s.NTok / t.Toks())
}

// GemmTileGrouped is how many workgroups a grouped GemmTile launches when the
// first used token blocks hold columns: every row block of each, the row block
// varying fastest, so a launch that stops early drops whole token blocks --
// the padding runs at the end of the sorted order -- as GemmVoltaGrouped.
func GemmTileGrouped(s MatVecShape, t TileGemm, used int) int {
	return (s.Rows / t.Rows()) * used
}

// gemmTilePad is the half-words of padding after every staged row. A row of
// KT halves at a stride of KT would put the eight rows one 8x8 tile load
// reads at the same bank offset; eight more halves (one 16-byte chunk) shift
// each row by four banks and keep every StoreV/LoadV 16-byte aligned. MLX's
// steel GEMM pads its threadgroup tiles for the same reason (tgp_padding).
const gemmTilePad = 8

// GemmTile is the batched quantized matvec on a collective matrix unit (the ir
// tile ops, which Metal lowers to simdgroup_matrix) with both operands staged
// through workgroup memory. It reads ActF16T's token-major binary16
// activations and writes pOut[tok*Rows + row], as GemmVolta does, so the tier
// launches it through the same path.
//
// It is llama.cpp's Metal kernel_mul_mm written in the IR: dequantize a tile
// of weights to half into threadgroup memory, stage the activations beside
// it, and let simdgroups multiply 8x8 half tiles into float accumulators.
// Apple has no integer dot product, so the dp4a batched path was slower on a
// prompt than decoding token by token.
//
// The dequant is GemmVolta's (voltaDeq), in the permuted element order
// voltaSteps gives and ActF16T writes; a product over k does not care about
// the order, so every format with a Volta layout has this one.
//
// Unlike GemmVolta: no swizzle (an 8x8 tile load names eight whole rows at one
// stride, so rows are padded instead) and no k-split, each workgroup owning
// its output block. A bias seeds the accumulator: a tile loaded from pBias at
// stride 0 is bias[row] in every column.
//
// With Experts > 1 it is a grouped mixture with GemmVolta's contract (sorted
// (token, slot) columns, one expert per token block in pSel, a bias bank
// [expert][row]), as llama.cpp's kernel_mul_mm_id.
//
// Parameters: pQS, pD, pSC, pB (ActF16T's output), pOut, pBias when s.Bias,
// and pSel when grouped. Launch GemmTileGroups(s, t) workgroups of
// t.Threads(), or GemmTileGrouped(s, t, used).
func GemmTile(s MatVecShape, t TileGemm) (*ir.Kernel, error) {
	if !Volta70OK(s.T) {
		return nil, fmt.Errorf("kernels: GemmTile: %v has no binary16 dequant", s.T)
	}
	grouped := s.Experts > 1
	if max(s.Split, 1) > 1 {
		return nil, fmt.Errorf("kernels: GemmTile: an unsplit shape only (split %d)", s.Split)
	}
	qi := qtab[s.T]
	if t.MT < 1 || t.NT < 1 || t.WM < 1 || t.WN < 1 || t.KB < 1 || t.Tall && t.MT%2 != 0 {
		return nil, fmt.Errorf("kernels: GemmTile: tile %+v", t)
	}
	BM, BN, nth := t.Rows(), t.Toks(), t.Threads()
	if s.Rows%BM != 0 || s.NTok%BN != 0 {
		return nil, fmt.Errorf("kernels: GemmTile: %dx%d block does not divide rows=%d ntok=%d", BM, BN, s.Rows, s.NTok)
	}
	if s.K%qi.blockE != 0 || s.K <= 0 {
		return nil, fmt.Errorf("kernels: GemmTile: K=%d is not a multiple of %d", s.K, qi.blockE)
	}
	nsub := s.K / qi.sub
	if nsub%t.KB != 0 {
		return nil, fmt.Errorf("kernels: GemmTile: %d sub-blocks a trip does not divide %d", t.KB, nsub)
	}
	trips := nsub / t.KB
	words := qi.sub*qi.bits/32 + qi.sub*qi.hi/32
	KT := t.KB * qi.sub          // halves per row per trip
	C := KT / 8                  // 16-byte chunks per row per trip
	subC := qi.sub / 8           // chunks per sub-block
	RS := (KT + gemmTilePad) / 2 // words per staged row
	if KT%8 != 0 || subC < 1 {
		return nil, fmt.Errorf("kernels: GemmTile: %d halves a trip is not whole chunks", KT)
	}
	if sh := 2 * (BM + BN) * RS * 4; sh > 32<<10 {
		return nil, fmt.Errorf("kernels: GemmTile: %d bytes of workgroup memory, over the 32 KiB Metal allows", sh)
	}
	// A items (row, sub-block) per thread. Fewer items than threads is allowed
	// where the rows nest in the threads: the surplus stage a duplicate item to
	// the same shared address, as in GemmVolta.
	nA := max(BM*t.KB/nth, 1)
	dup := BM*t.KB < nth
	if dup && nth%BM != 0 || !dup && ((BM*t.KB)%nth != 0 || nth%BM != 0 && BM%nth != 0) {
		return nil, fmt.Errorf("kernels: GemmTile: %d threads do not tile %d rows x %d sub-blocks", nth, BM, t.KB)
	}
	if (BN*C)%nth != 0 || nth%C != 0 {
		return nil, fmt.Errorf("kernels: GemmTile: %d threads do not tile %d B chunks", nth, BN*C)
	}
	// Where the items are fewer than the threads, the surplus threads would
	// dequantize a duplicate. On a format with no secondary plane a sub-block
	// splits by its primary words instead: parts threads each decode and stage
	// 1/parts of it, so no thread repeats another's work. Apple's matrix
	// multiply runs on the same ALUs as the dequant, so a duplicate is not free.
	pw := qi.sub * qi.bits / 32
	parts := 1
	if dup && qi.hi == 0 && !t.NoSplit {
		if p := nth / (BM * t.KB); nth%(BM*t.KB) == 0 && pw%p == 0 && subC%p == 0 {
			parts = p
		}
	}
	pwP, subCP := pw/parts, subC/parts
	lw := pwP + qi.sub*qi.hi/32 // words a thread loads per item

	name := fmt.Sprintf("gemmtile_%s_%dx%d_t%d_m%dn%dw%dx%dk%d", strings.ToLower(s.T.String()),
		s.Rows, s.K, s.NTok, t.MT, t.NT, t.WM, t.WN, t.KB)
	if parts > 1 {
		name += fmt.Sprintf("_p%d", parts)
	}
	if t.Tall {
		name += "_tall"
	}
	if grouped {
		name += fmt.Sprintf("_e%d", s.Experts)
	}
	if t.TokOff != 0 {
		if grouped || t.TokOff < 0 {
			return nil, fmt.Errorf("kernels: GemmTile: token offset %d on a grouped or negative launch", t.TokOff)
		}
		name += fmt.Sprintf("_o%d", t.TokOff)
	}
	b := ir.New(name, [3]int{nth, 1, 1})
	pQS := b.Param("pQS", ir.U32)
	pD := b.Param("pD", ir.U32)
	pSC := b.Param("pSC", ir.U32)
	pB := b.Param("pB", ir.U32)
	pOut := b.ParamTile("pOut", ir.TileF32)
	var pBias, pSel ir.Value
	if s.Bias {
		pBias = b.ParamTile("pBias", ir.TileF32)
	}
	if grouped {
		pSel = b.Param("pSel", ir.U32)
	}
	shA := b.Shared("gemmA", ir.U32, 2*BM*RS)
	shB := b.Shared("gemmB", ir.U32, 2*BN*RS)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	add := func(v ir.Value, n int64) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, c(n))
	}

	// The workgroup's block: token blocks vary fastest, so the workgroups
	// that dequantize one weight block run together -- except grouped, where
	// the row block varies fastest so a launch of GemmTileGrouped workgroups
	// covers the first `used` token blocks whole.
	mB, nB := s.Rows/BM, s.NTok/BN
	grp := b.CTAID()
	var bn, bm ir.Value
	if grouped {
		bm = b.Rem(ir.U32, grp, c(int64(mB)))
		bn = b.Min(ir.U32, b.Div(ir.U32, grp, c(int64(mB))), c(int64(nB-1)))
	} else {
		bn = b.Rem(ir.U32, grp, c(int64(nB)))
		bm = b.Rem(ir.U32, b.Div(ir.U32, grp, c(int64(nB))), c(int64(mB)))
	}
	rowBlk := b.Mul(ir.U32, bm, c(int64(BM)))
	tokBlk := add(b.Mul(ir.U32, bn, c(int64(BN))), int64(t.TokOff))
	tid := b.TID()

	dq := newVoltaDeq(b, s.T, pQS, pD, pSC, s.Rows)
	// The SC stream's first word carried across trips, as GemmVolta does.
	carry := dq.SetCarry(s.K, t.KB)
	var eSel ir.Value
	if grouped {
		eSel = b.Load(ir.U32, pSel, bn, 0)
		if err := dq.expert(eSel, s.K); err != nil {
			return nil, err
		}
	}

	// A staging: item it = tid + nth*i is (row it%BM, sub-block it/BM).
	type aItem struct{ g, s, row, sb int64 }
	aC := make([]aItem, nA)
	var aRow0, aSb0 ir.Value
	aPart := c(0)
	if BM%nth == 0 {
		per := BM / nth
		aRow0, aSb0 = tid, c(0)
		for i := range aC {
			r, sb := int64(nth*(i%per)), int64(i/per)
			aC[i] = aItem{g: sb*int64(words*s.Rows) + r, s: r * int64(RS), row: r, sb: sb}
		}
	} else {
		aRow0, aSb0 = b.Rem(ir.U32, tid, c(int64(BM))), b.Div(ir.U32, tid, c(int64(BM)))
		if dup {
			aSb0 = b.Rem(ir.U32, aSb0, c(int64(t.KB)))
		}
		if parts > 1 {
			aPart = b.Div(ir.U32, tid, c(int64(BM*t.KB)))
		}
		for i := range aC {
			sb := int64(i * (nth / BM))
			aC[i] = aItem{g: sb * int64(words*s.Rows), sb: sb}
		}
	}
	aRowG := b.Add(ir.U32, rowBlk, aRow0)
	aGBase := b.Add(ir.U32, b.Mul(ir.U32, aSb0, c(int64(words*s.Rows))), aRowG)
	if parts > 1 {
		aGBase = b.Add(ir.U32, aGBase, b.Mul(ir.U32, aPart, c(int64(pwP*s.Rows))))
	}
	// aAt[i][q] is where item i's chunk q goes: row*RS + (sub-block*subC+q)*4.
	aSBase := b.Add(ir.U32, b.Mul(ir.U32, aRow0, c(int64(RS))), b.Mul(ir.U32, aSb0, c(int64(4*subC))))
	if parts > 1 {
		aSBase = b.Add(ir.U32, aSBase, b.Mul(ir.U32, aPart, c(int64(4*subCP))))
	}
	aAt := make([][]int64, nA)
	for i := range aAt {
		aAt[i] = make([]int64, subCP)
		for q := range aAt[i] {
			aAt[i][q] = aC[i].s + (aC[i].sb*int64(subC)+int64(q))*4
		}
	}

	// B staging: chunk e = tid + nth*i of the trip's slice is (token e/C,
	// chunk e%C); e%C is tid%C for every i.
	nBc := BN * C / nth
	bTok0 := b.Div(ir.U32, tid, c(int64(C)))
	bCh := b.Rem(ir.U32, tid, c(int64(C)))
	bGBase := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, tokBlk, bTok0), c(int64(s.K/2))), b.Shl(ir.U32, bCh, c(2)))
	bStep := int64(nth / C) // tokens between one item and the next
	bSBase := b.Add(ir.U32, b.Mul(ir.U32, bTok0, c(int64(RS))), b.Shl(ir.U32, bCh, c(2)))

	load := func(W, Bc, sub ir.Value) (ra [][]ir.Value, rb [][]ir.Value) {
		ra = make([][]ir.Value, nA)
		wa := b.Add(ir.U32, W, aGBase)
		sbBase := b.Add(ir.U32, sub, aSb0)
		for i := range ra {
			ra[i] = append(dq.wordsN(wa, aC[i].g, pwP), dq.scaleWords(add(sbBase, aC[i].sb), add(aRowG, aC[i].row))...)
		}
		rb = make([][]ir.Value, nBc)
		wb := b.Add(ir.U32, Bc, bGBase)
		for i := range rb {
			rb[i] = b.LoadV(ir.U32, pB, wb, int64(i)*bStep*int64(s.K/2), 4)
		}
		return
	}
	stage := func(ra, rb [][]ir.Value, sub, offA, offB ir.Value, x []ir.Value) {
		sbBase := b.Add(ir.U32, sub, aSb0)
		vals := make([][][]ir.Value, nA)
		for i := 0; i < nA; i++ {
			var xi ir.Value
			if carry {
				xi = x[i]
			}
			s2, m2 := dq.scalesOfX(ra[i][lw:], xi, add(sbBase, aC[i].sb), add(aRowG, aC[i].row))
			vals[i] = dq.fragsN(ra[i][:lw], s2, m2, pwP)
		}
		baseA := b.Add(ir.U32, aSBase, offA)
		for i := 0; i < nA; i++ {
			f := vals[i]
			for q := 0; q < subCP; q++ {
				b.StoreV(shA, baseA, aAt[i][q], f[2*q][0], f[2*q][1], f[2*q+1][0], f[2*q+1][1])
			}
		}
		baseB := b.Add(ir.U32, bSBase, offB)
		for i := 0; i < nBc; i++ {
			b.StoreV(shB, baseB, int64(i)*bStep*int64(RS), rb[i]...)
		}
	}
	last := c(0)
	advance := func(sub, W, Bc ir.Value) (ir.Value, ir.Value, ir.Value) {
		n := b.Min(ir.U32, add(sub, int64(t.KB)), last)
		delta := b.Sub(ir.U32, n, sub) // 0 or KB
		return n, b.Add(ir.U32, W, b.Mul(ir.U32, delta, c(int64(words*s.Rows)))),
			b.Add(ir.U32, Bc, b.Mul(ir.U32, delta, c(int64(qi.sub/2))))
	}
	sub0 := c(0)
	last = c(int64((trips - 1) * t.KB))
	w0, b0 := c(0), c(0)
	// Double-buffered, as GemmVolta: trip t's MMAs read buffer t%2 while the
	// same straight-line block dequantizes trip t+1 into the other, and one
	// barrier a trip covers both hazards.
	sizeA, sizeB := c(int64(BM*RS)), c(int64(BN*RS))
	nextX := func(x []ir.Value, ra [][]ir.Value, sub ir.Value) []ir.Value {
		if !carry {
			return nil
		}
		out := make([]ir.Value, nA)
		for i := range out {
			out[i] = dq.scNext(x[i], ra[i][lw+1], add(b.Add(ir.U32, sub, aSb0), aC[i].sb), t.KB)
		}
		return out
	}
	var x0 []ir.Value
	if carry {
		for i := 0; i < nA; i++ {
			x0 = append(x0, dq.scFirst(add(b.Add(ir.U32, sub0, aSb0), aC[i].sb), add(aRowG, aC[i].row)))
		}
	}
	ra0, rb0 := load(w0, b0, sub0)
	stage(ra0, rb0, sub0, c(0), c(0), x0)
	sub1, w1, b1 := advance(sub0, w0, b0)
	x1 := nextX(x0, ra0, sub0)
	ra1, rb1 := load(w1, b1, sub1)
	b.Barrier()

	// The subgroup's tiles: subgroup (wm, wn) owns rows wm*MT*8.. and tokens
	// wn*NT*8.. of the block. Every tile address is built from the subgroup
	// index and the CTA, which is what a collective access requires.
	sg := b.SubgroupIndex()
	wm := b.Rem(ir.U32, sg, c(int64(t.WM)))
	wn := b.Div(ir.U32, sg, c(int64(t.WM)))
	rowW := b.Mul(ir.U32, wm, c(int64(8*t.MT)))
	tokW := b.Mul(ir.U32, wn, c(int64(8*t.NT)))
	// TM rows a tile, MA tiles down the subgroup's rows (see Tall).
	TM, MA := 8, t.MT
	if t.Tall {
		TM, MA = 16, t.MT/2
	}
	aT := ir.TileType{Rows: TM, Cols: 8, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF16, Use: ir.TileB}
	cT := ir.TileType{Rows: TM, Cols: 8, Elem: ir.TileF32, Use: ir.TileAcc}
	rowG := b.Add(ir.U32, rowBlk, rowW)
	tokG := b.Add(ir.U32, tokBlk, tokW)
	// Half-word offsets of the subgroup's first A row and first B token.
	aH := b.Mul(ir.U32, rowW, c(int64(2*RS)))
	bH := b.Mul(ir.U32, tokW, c(int64(2*RS)))
	strideH := c(int64(2 * RS))
	zero := c(0)
	init := make([][]ir.Value, MA)
	for i := range init {
		init[i] = make([]ir.Value, t.NT)
		for j := range init[i] {
			if s.Bias {
				// Column-major at stride 0: element (r, c) is pBias[row+r],
				// and a grouped bias is a bank, [expert][row].
				bi := add(rowG, int64(TM*i))
				if grouped {
					bi = b.Add(ir.U32, b.Mul(ir.U32, eSel, c(int64(s.Rows))), bi)
				}
				init[i][j] = b.TileLoad(cT, pBias, bi, zero, true)
			} else {
				init[i][j] = b.TileSplat(cT, b.ConstF32(0))
			}
		}
	}

	b.Loop(int64(trips))
	accs := make([][]ir.Value, MA)
	for i := range accs {
		accs[i] = make([]ir.Value, t.NT)
		for j := range accs[i] {
			accs[i][j] = b.TilePhi(init[i][j])
		}
	}
	curA := b.Phi(ir.U32, zero) // the buffer this trip's MMAs read, in words
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

	d := make([][]ir.Value, MA)
	for i := range d {
		d[i] = append([]ir.Value{}, accs[i]...)
	}
	baseAH := b.Add(ir.U32, b.Shl(ir.U32, curA, c(1)), aH)
	baseBH := b.Add(ir.U32, b.Shl(ir.U32, curB, c(1)), bH)
	for kk := 0; kk < KT/8; kk++ {
		av := make([]ir.Value, MA)
		for i := range av {
			av[i] = b.TileLoad(aT, shA, add(baseAH, int64(TM*i*2*RS+8*kk)), strideH, false)
		}
		bv := make([]ir.Value, t.NT)
		for j := range bv {
			bv[j] = b.TileLoad(bT, shB, add(baseBH, int64(8*j*2*RS+8*kk)), strideH, true)
		}
		for i := 0; i < MA; i++ {
			for j := 0; j < t.NT; j++ {
				d[i][j] = b.TileMMA(av[i], bv[j], d[i][j])
			}
		}
	}
	nxA, nxB := b.Sub(ir.U32, sizeA, curA), b.Sub(ir.U32, sizeB, curB)
	stage(ra, rb, subR, nxA, nxB, xR)
	subN, WN, BN2 := advance(subR, WR, BR)
	na, nb := load(WN, BN2, subN)
	b.Barrier()
	for i := 0; i < MA; i++ {
		for j := 0; j < t.NT; j++ {
			b.SetPhi(accs[i][j], d[i][j])
		}
	}
	b.SetPhi(curA, nxA)
	b.SetPhi(curB, nxB)
	xN := nextX(xR, ra, subR) // last: see GemmVolta
	for i := range xR {
		b.SetPhi(xR[i], xN[i])
	}
	b.SetPhi(subR, subN)
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

	// C is rows x tokens and pOut is [token][row], so each tile is stored
	// column-major at a stride of Rows.
	//
	// Store the phi, not the last MMA: after EndLoop a phi holds its final
	// value on every target, while on MSL the MMA result is scoped inside the
	// loop body.
	nrows := c(int64(s.Rows))
	for i := 0; i < MA; i++ {
		for j := 0; j < t.NT; j++ {
			off := b.Add(ir.U32, b.Mul(ir.U32, add(tokG, int64(8*j)), nrows), add(rowG, int64(TM*i)))
			b.TileStore(pOut, off, nrows, accs[i][j], true)
		}
	}
	return b.Done(), nil
}
