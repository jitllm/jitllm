package mainkernels

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// VoltaTile is GemmVolta's blocking: a workgroup of WM*WN warps, each warp MT
// m-tiles of 32 rows (16 with F16K) by NT n-tiles of 8 tokens, and KB
// sub-blocks of k staged per trip. The workgroup covers WM*MT*32 rows by
// WN*NT*8 tokens.
//
// F16K picks the warp's matrix instruction: 0 is sm_70's m8n8k4
// (ir.MMAVolta()), 16 is m16n8k16 (sm_80 on) and 8 is m16n8k8 (sm_75's
// binary16 shape). The staging is the same for all three; only the fragment
// reads, the instruction and the epilogue's lane map differ.
//
// Single stages one buffer instead of two: a trip is MMAs, a barrier, the
// next trip's stage, a barrier. It halves the shared memory, so an m16n8
// tile can stage 128 elements of k a trip (two barriers a trip, four a 256
// of k -- llama.cpp's MMQ count) where double-buffering fits 64. The next
// trip's global words are still loaded into registers a trip ahead.
type VoltaTile struct {
	MT, NT, WM, WN, KB int
	F16K               int
	Single             bool
}

// mRows is one m-tile's rows.
func (t VoltaTile) mRows() int {
	if t.F16K != 0 {
		return 16
	}
	return 32
}

// Rows and Toks are the workgroup's block of the output.
func (t VoltaTile) Rows() int { return t.WM * t.MT * t.mRows() }
func (t VoltaTile) Toks() int { return t.WN * t.NT * 8 }

// Threads is the workgroup width.
func (t VoltaTile) Threads() int { return 32 * t.WM * t.WN }

// GemmVoltaGroups is how many workgroups GemmVolta launches for a shape. A
// grouped shape launches GemmVoltaGrouped(s, t, used) instead.
func GemmVoltaGroups(s MatVecShape, t VoltaTile) int {
	return (s.Rows / t.Rows()) * (s.NTok / t.Toks()) * max(s.Split, 1)
}

// GemmVoltaGrouped is how many workgroups a grouped GemmVolta launches when
// the first used token blocks hold columns: every row block of each. The row
// block varies fastest, so a launch that stops early drops whole token blocks
// -- the padding runs at the end of the sorted order -- and nothing else.
func GemmVoltaGrouped(s MatVecShape, t VoltaTile, used int) int {
	return (s.Rows / t.Rows()) * used
}

// f16Swizzle is an m16n8 tile's chunk permutation for a row of 8 or more
// chunks (64 elements of k a trip and up). A fragment load's phase is lanes
// 0-7: rows (or tokens) g and g+1, g even, chunk q = 0..3 each, so g+1's
// chunks must land in the other half of the 8 bank chunks; a staging store's
// phase is eight consecutive rows at one chunk (A) or one token's eight
// chunks (B). swz(r) = 4(r&1) + (r>>1)&3 is a permutation of 0..7 over any
// eight consecutive rows and puts rows g and g+1 four chunks apart.
func f16Swizzle(b *ir.Builder, r ir.Value) ir.Value {
	k := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	return b.Add(ir.U32, b.Shl(ir.U32, b.And(ir.U32, r, k(1)), k(2)), b.And(ir.U32, b.Shr(ir.U32, r, k(1)), k(3)))
}

// voltaSwizzle is the 16-byte chunk permutation of one shared-memory row of c
// chunks: row r's chunk j lives at chunk j^swz(r). See GemmVolta.
func voltaSwizzle(b *ir.Builder, r ir.Value, c int, weights bool) ir.Value {
	k := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	bit := func(i int64) ir.Value { return b.And(ir.U32, b.Shr(ir.U32, r, k(i)), k(1)) }
	if !weights {
		// Tokens: a fragment load names four tokens a phase, t..t+3.
		switch {
		case c >= 8:
			return b.And(ir.U32, r, k(3))
		case c == 4:
			return bit(1)
		}
		return k(0)
	}
	// Rows: a fragment load's phase is rows {0-3, 8-11} (+4, +16), and a
	// staging store's is eight consecutive rows; b2^b3 separates both.
	b23 := b.Xor(ir.U32, bit(2), bit(3))
	switch {
	case c >= 8:
		return b.Add(ir.U32, b.And(ir.U32, r, k(3)), b.Shl(ir.U32, b23, k(2)))
	case c == 4:
		return b.Add(ir.U32, bit(1), b.Shl(ir.U32, b23, k(1)))
	}
	return b23
}

// GemmVolta is the batched quantized matvec on sm_70's f16 tensor cores
// (ir.MMAVolta()) with both operands staged through shared memory: MatVecMMA70's
// arithmetic, blocked like a GEMM. It reads ActF16T's token-major activations.
//
// MatVecMMA70 gives every warp its own rows and its own dequant and reads the
// activations from global memory inside the MMA loop, so a large chunk
// dequantizes every weight many times. Here a workgroup stages KB sub-blocks
// of its rows' weights, dequantized once to f16, and the matching slice of
// its tokens' activations, and every warp reads its fragments from there. The
// next trip's global words are loaded into registers before this trip's MMAs.
// See docs/engineering-history/gpu-kernels.md.
//
// Every shared access is 16 bytes (two m8n8k4 steps) and conflict-free by a
// swizzle rather than padding. A 16-byte access is served eight lanes a
// phase; the eight A rows a phase names are {0-3, 8-11} (or +4, +16) under the
// Volta lane layout and a staging store's are eight consecutive rows, so
// chunk j of row r is stored at j^swz(r) with swz built from row bits that
// separate both sets (voltaSwizzle). The tokens are the same argument with
// four tokens a phase.
//
// Parameters are MatVecMMA70's with pB from ActF16T: pQS, pD, pSC, pB, pOut,
// and pBias. The grid is exact (GemmVoltaGroups workgroups of t.Threads()), so
// no thread is surplus.
//
// With Experts > 1 it is a grouped mixture (llama.cpp's MUL_MAT_ID on the
// tensor cores): the NTok columns are (token, slot) pairs sorted by expert,
// each expert's run padded to a whole token block, and pSel (after pOut and
// pBias) holds one expert per token block. A workgroup dequantizes its block's
// expert sheet, so one staged tile serves every token routed there. Launch
// GemmVoltaGrouped(s, t, used) workgroups; the row block varies fastest, so
// token blocks past `used` are never launched.
func GemmVolta(s MatVecShape, t VoltaTile) (*ir.Kernel, error) { return gemmVolta(s, t, nil) }

// gemmCtx is where gemmVolta emits when it is one SEGMENT of a
// GemmVoltaSegments kernel: the shared builder, a suffix that keeps its
// parameters apart from the other segments', the operand every segment
// reads, the workgroup index relative to the segment's first, and the
// staging buffers declared once before any segment's loop.
type gemmCtx struct {
	b        *ir.Builder
	suffix   string
	pB       ir.Value
	cta      func() ir.Value
	shA, shB ir.Value
}

func gemmVolta(s MatVecShape, t VoltaTile, into *gemmCtx) (*ir.Kernel, error) {
	if !GemmVoltaOK(s.T) {
		return nil, fmt.Errorf("kernels: GemmVolta: %v has no Volta layout", s.T)
	}
	qi := qtab[s.T]
	grouped := s.Experts > 1
	fl := qi.float > 0
	if fl && grouped {
		return nil, fmt.Errorf("kernels: GemmVolta: a grouped %v bank has no Volta layout", s.T)
	}
	if grouped && (s.NTok < 2 || max(s.Split, 1) > 1) {
		return nil, fmt.Errorf("kernels: GemmVolta: a grouped shape takes no split (ntok %d split %d)",
			s.NTok, s.Split)
	}
	if t.MT < 1 || t.NT < 1 || t.WM < 1 || t.WN < 1 || t.KB < 1 || t.F16K != 0 && t.F16K != 8 && t.F16K != 16 {
		return nil, fmt.Errorf("kernels: GemmVolta: tile %+v", t)
	}
	split := max(s.Split, 1)
	BM, BN, nth := t.Rows(), t.Toks(), t.Threads()
	if s.Rows%BM != 0 || s.NTok%BN != 0 {
		return nil, fmt.Errorf("kernels: GemmVolta: %dx%d block does not divide rows=%d ntok=%d", BM, BN, s.Rows, s.NTok)
	}
	if s.K%qi.blockE != 0 || s.K <= 0 {
		return nil, fmt.Errorf("kernels: GemmVolta: K=%d is not a multiple of %d", s.K, qi.blockE)
	}
	nsub := s.K / qi.sub
	if nsub%(split*t.KB) != 0 {
		return nil, fmt.Errorf("kernels: GemmVolta: split %d x %d sub-blocks does not divide %d", split, t.KB, nsub)
	}
	trips := nsub / split / t.KB
	words := qi.sub*qi.bits/32 + qi.sub*qi.hi/32
	// subStride and rowStride place a row's sub-block in the weight's words: a
	// packed format's planes are sub-block-major with the rows interleaved by
	// word, a float weight row-major (MatVec's layout; GemmVoltaOK).
	subStride, rowStride := int64(words*s.Rows), int64(1)
	if fl {
		words = qi.sub * qi.float / 4
		subStride, rowStride = int64(words), int64(s.K*qi.float/4)
	}
	kHalf := t.KB * qi.sub / 2 // u32 (two f16) per row per trip
	C := kHalf / 4             // 16-byte chunks per row per trip
	subC := qi.sub / 8         // chunks per sub-block
	if C&(C-1) != 0 || C < 2 || subC < 1 {
		return nil, fmt.Errorf("kernels: GemmVolta: %d chunks a trip; KB*sub must be 16, 32, 64 or more by powers of two", C)
	}
	if t.F16K != 0 && C != 4 && C%8 != 0 {
		// Lane q of an m16n8 fragment reads chunk 4s+q of its row in step s:
		// a trip is whole 32-element steps, and a wide one is 8 chunks or more
		// so f16Swizzle's permutation of eight has whole rows to work in.
		return nil, fmt.Errorf("kernels: GemmVolta: an m16n8 tile stages 32 or a multiple of 64 elements a trip, not %d", 8*C)
	}
	if t.Single && t.F16K == 0 {
		return nil, fmt.Errorf("kernels: GemmVolta: a single-buffered stage is an m16n8 tile's")
	}
	nbuf := 2
	if t.Single {
		nbuf = 1
	}
	if sh := nbuf * (BM + BN) * kHalf * 4; sh > 48<<10 {
		return nil, fmt.Errorf("kernels: GemmVolta: %d bytes of shared memory, over the 48 KiB a kernel may declare", sh)
	}
	if (BM*t.KB)%nth != 0 && BM*t.KB > nth || (BN*C)%nth != 0 || nth%C != 0 {
		return nil, fmt.Errorf("kernels: GemmVolta: %d threads do not tile %d A items and %d B chunks", nth, BM*t.KB, BN*C)
	}

	name := fmt.Sprintf("gemmvolta_%s_%dx%d_t%d_m%dn%dw%dx%dk%d_s%d", strings.ToLower(s.T.String()),
		s.Rows, s.K, s.NTok, t.MT, t.NT, t.WM, t.WN, t.KB, split)
	if t.F16K != 0 {
		name += fmt.Sprintf("_f%d", t.F16K)
	}
	if t.Single {
		name += "_1b"
	}
	if grouped {
		name += fmt.Sprintf("_e%d", s.Experts)
	}
	var b *ir.Builder
	var pQS, pD, pSC, pB, pOut, shA, shB ir.Value
	sfx := ""
	cta := func() ir.Value { return b.CTAID() }
	if into == nil {
		b = ir.New(name, [3]int{nth, 1, 1})
		pQS = b.Param("pQS", ir.U32)
		pD = b.Param("pD", ir.U32)
		pSC = b.Param("pSC", ir.U32)
		pB = b.Param("pB", ir.U32)
		pOut = b.Param("pOut", ir.F32)
	} else {
		if grouped || split > 1 {
			return nil, fmt.Errorf("kernels: GemmVoltaSegments: a segment is dense and unsplit")
		}
		b, sfx, cta, pB = into.b, into.suffix, into.cta, into.pB
		pQS = b.Param("pQS"+sfx, ir.U32)
		pD = b.Param("pD"+sfx, ir.U32)
		pSC = b.Param("pSC"+sfx, ir.U32)
		pOut = b.Param("pOut"+sfx, ir.F32)
	}
	var pBias, pSel ir.Value
	if s.Bias {
		pBias = b.Param("pBias"+sfx, ir.F32)
	}
	if grouped {
		pSel = b.Param("pSel", ir.U32)
	}
	if into == nil {
		shA = b.Shared("gemmA", ir.U32, nbuf*BM*kHalf)
		shB = b.Shared("gemmB", ir.U32, nbuf*BN*kHalf)
	} else {
		shA, shB = into.shA, into.shB
	}
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	add := func(v ir.Value, n int64) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, c(n))
	}
	// wide is an m16n8 trip of 8 chunks or more, swizzled by f16Swizzle.
	wide := t.F16K != 0 && C >= 8
	swzOf := func(r ir.Value, weights bool) ir.Value {
		if wide {
			return f16Swizzle(b, r)
		}
		return voltaSwizzle(b, r, C, weights)
	}
	// chunkAt is the shared index of chunk j of the row whose swizzle is swz:
	// one Value per j, built once before the loop.
	chunkAt := func(swz ir.Value) []ir.Value {
		out := make([]ir.Value, C)
		for j := range out {
			out[j] = b.Shl(ir.U32, b.Xor(ir.U32, swz, c(int64(j))), c(2))
		}
		return out
	}

	// The workgroup's block: token blocks vary fastest, so the workgroups that
	// read one weight block run together and it comes from DRAM about once.
	mB, nB := s.Rows/BM, s.NTok/BN
	grp := cta()
	var bn, bm, seg ir.Value
	if grouped {
		// The row block fastest: a launch of GemmVoltaGrouped workgroups
		// covers the first `used` token blocks whole, and consecutive
		// workgroups share the token block's staged activations.
		bm = b.Rem(ir.U32, grp, c(int64(mB)))
		bn = b.Min(ir.U32, b.Div(ir.U32, grp, c(int64(mB))), c(int64(nB-1)))
		seg = c(0)
	} else {
		bn = b.Rem(ir.U32, grp, c(int64(nB)))
		bm = b.Rem(ir.U32, b.Div(ir.U32, grp, c(int64(nB))), c(int64(mB)))
		seg = b.Div(ir.U32, grp, c(int64(mB*nB)))
	}
	rowBlk := b.Mul(ir.U32, bm, c(int64(BM)))
	tokBlk := b.Mul(ir.U32, bn, c(int64(BN)))
	tid := b.TID()

	dq := newVoltaDeq(b, s.T, pQS, pD, pSC, s.Rows)
	// The SC stream's first word is carried, not loaded: container v27 packs a
	// row's scale/min pairs 12 bits a sub-block, so a pair can straddle two
	// words. A thread stages the same rows every trip and each trip moves at most
	// 24 bits on, so the pair's first word is the previous trip's first or the
	// one after it: one load a trip instead of two.
	carry := dq.SetCarry(s.K, t.KB)
	var eSel ir.Value
	if grouped {
		eSel = b.Load(ir.U32, pSel, bn, 0)
		if err := dq.expert(eSel, s.K); err != nil {
			return nil, err
		}
	}

	// A staging: item it = tid + nth*i is (row it%BM, sub-block it/BM). Its
	// indices are a per-thread base plus a per-item constant, and every item
	// of a thread has the same low row bits, hence one swizzle.
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
			// Fewer items than threads: the surplus stage a duplicate item,
			// the same words to the same shared address. (A 64-row block
			// exists for 2880 rows, which no 128-row block divides.)
			aSb0 = b.Rem(ir.U32, aSb0, c(int64(t.KB)))
		}
		for i := range aC {
			sb := int64(i * (nth / BM))
			aC[i].g, aC[i].sb = sb*subStride, sb
		}
	case BM*t.KB < nth:
		// Fewer items than threads with rows that do not nest in them (a
		// 96-row block): item tid mod BM*KB, the surplus duplicating the first
		// items exactly as above.
		item := b.Rem(ir.U32, tid, c(int64(BM*t.KB)))
		aRow0, aSb0 = b.Rem(ir.U32, item, c(int64(BM))), b.Div(ir.U32, item, c(int64(BM)))
	default:
		return nil, fmt.Errorf("kernels: GemmVolta: %d rows and %d threads do not nest", BM, nth)
	}
	aRowG := b.Add(ir.U32, rowBlk, aRow0)
	aGBase := b.Add(ir.U32, b.Mul(ir.U32, aSb0, c(subStride)), b.Mul(ir.U32, aRowG, c(rowStride)))
	aSBase := b.Mul(ir.U32, aRow0, c(int64(kHalf)))
	// aAt[i][q] is where item i's chunk q goes in its row: chunk
	// (sub-block)*subC+q, swizzled. The item's sub-block is a constant when a
	// thread's items differ only in row, and tid/BM plus one otherwise.
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
	// chunk e%C); e%C is tid%C for every i. The token's swizzle reads its two
	// low bits, so it is one Value for all of a thread's items when they are a
	// multiple of four tokens apart and one per item otherwise.
	nBc := BN * C / nth
	bTok0 := b.Div(ir.U32, tid, c(int64(C)))
	bCh := b.Rem(ir.U32, tid, c(int64(C)))
	bGBase := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, tokBlk, bTok0), c(int64(s.K/2))), b.Shl(ir.U32, bCh, c(2)))
	bStep := int64(nth / C) // tokens between one item and the next
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

	// The fragment bases: warp (wm, wn) of the workgroup, the lane's A row and
	// B token within the warp's tile, and one index per chunk of each.
	warp := b.Shr(ir.U32, tid, c(5))
	wm := b.Rem(ir.U32, warp, c(int64(t.WM)))
	wn := b.Div(ir.U32, warp, c(int64(t.WM)))
	v := voltaLane{b, b.And(ir.U32, tid, c(31))}
	var rowW, tokW ir.Value
	var fragA, fragB []ir.Value
	// m16n8's lane (g, q): g = lane/4 is the A row (and g+8) and the B token,
	// q = lane%4 the k pair. fragA16 holds rows g and g+8, chunk q each.
	// A wide trip's step s reads chunk (4s+q)^swz = (q^swz)^4(s&1) + 8(s>>1):
	// index [p] is the even (p = 0) and odd step's base, the rest a constant.
	var fragA16 [2][2]ir.Value
	var fragB16 [2]ir.Value
	var lq, lg ir.Value
	if t.F16K == 0 {
		q8 := b.Shl(ir.U32, v.quad(), c(3))
		rowW = b.Add(ir.U32, b.Mul(ir.U32, wm, c(int64(32*t.MT))), q8)
		tokW = b.Mul(ir.U32, wn, c(int64(8*t.NT)))
		rowL := b.Add(ir.U32, rowW, v.row())
		tokL := b.Add(ir.U32, tokW, v.row())
		fragA, fragB = chunkAt(voltaSwizzle(b, rowL, C, true)), chunkAt(voltaSwizzle(b, tokL, C, false))
		baseA := b.Mul(ir.U32, rowL, c(int64(kHalf)))
		baseB := b.Mul(ir.U32, tokL, c(int64(kHalf)))
		for j := range fragA {
			fragA[j] = b.Add(ir.U32, baseA, fragA[j])
			fragB[j] = b.Add(ir.U32, baseB, fragB[j])
		}
	} else {
		// Lane q reads chunk q of its row and of its token: chunk q holds the
		// trip's words 4q..4q+3, and word 4q+2s+h feeds step s's k pair q
		// (h = 0) or q+4 (h = 1) in both operands alike, so every word of A
		// meets the same word of B whatever order the staging wrote k in.
		// The swizzle depends on row bits 1-3, so rows g and g+8 take one
		// each and the m-tiles (16 rows on) share them; tokens use bit 1 only.
		lg = b.Shr(ir.U32, v.lane, c(2))
		lq = b.And(ir.U32, v.lane, c(3))
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

	// One trip's global words into registers: each A item's packed words then
	// its scale words, and each B chunk as one 16-byte load.
	load := func(W, Bc, sub ir.Value) (ra [][]ir.Value, rb [][]ir.Value) {
		ra = make([][]ir.Value, nA)
		wa := b.Add(ir.U32, W, aGBase)
		sbBase := b.Add(ir.U32, sub, aSb0)
		for i := range ra {
			ra[i] = append(dq.words(wa, aC[i].g), dq.scaleWords(add(sbBase, aC[i].sb), add(aRowG, aC[i].row))...)
		}
		rb = make([][]ir.Value, nBc)
		wb := b.Add(ir.U32, Bc, bGBase)
		for i := range rb {
			rb[i] = b.LoadV(ir.U32, pB, wb, int64(i)*bStep*int64(s.K/2), 4)
		}
		return
	}
	// stage dequantizes one trip's words and stores them, with its B chunks,
	// into the buffer at shared offsets (offA, offB). The ALU work is emitted
	// first and the stores last, so the stores cannot pin anything above them.
	// x is each item's carried SC word where carry holds (see above).
	stage := func(ra, rb [][]ir.Value, sub, offA, offB ir.Value, x []ir.Value) {
		sbBase := b.Add(ir.U32, sub, aSb0)
		vals := make([][][]ir.Value, nA)
		for i := 0; i < nA; i++ {
			var xi ir.Value
			if carry {
				xi = x[i]
			}
			var s2, m2 ir.Value
			if !fl {
				s2, m2 = dq.scalesOfX(ra[i][words:], xi, add(sbBase, aC[i].sb), add(aRowG, aC[i].row))
			}
			vals[i] = dq.frags(ra[i][:words], s2, m2)
		}
		for i := 0; i < nA; i++ {
			f := vals[i]
			for q := 0; q < subC; q++ {
				// Chunk q of the sub-block holds its steps 2q and 2q+1.
				b.StoreV(shA, b.Add(ir.U32, aAt[i][q], offA), aC[i].s, f[2*q][0], f[2*q][1], f[2*q+1][0], f[2*q+1][1])
			}
		}
		for i := 0; i < nBc; i++ {
			b.StoreV(shB, b.Add(ir.U32, bS[i], offB), bSOff[i], rb[i]...)
		}
	}
	// advance is the trip after sub, clamped to the last: the words past the
	// end are re-read, which is in bounds, and their stage lands in the buffer
	// no MMA reads again.
	last := c(0)
	advance := func(sub, W, Bc ir.Value) (ir.Value, ir.Value, ir.Value) {
		n := b.Min(ir.U32, add(sub, int64(t.KB)), last)
		delta := b.Sub(ir.U32, n, sub) // 0 or KB
		return n, b.Add(ir.U32, W, b.Mul(ir.U32, delta, c(subStride))),
			b.Add(ir.U32, Bc, b.Mul(ir.U32, delta, c(int64(qi.sub/2))))
	}
	sub0 := b.Mul(ir.U32, seg, c(int64(trips*t.KB)))
	last = b.Add(ir.U32, sub0, c(int64((trips-1)*t.KB)))
	w0 := b.Mul(ir.U32, sub0, c(subStride))
	b0 := b.Mul(ir.U32, sub0, c(int64(qi.sub/2)))
	// Double-buffered, so a trip's dequant runs under the previous trip's MMAs:
	// trip t's MMAs read buffer t%2 while the same straight-line block
	// dequantizes trip t+1 into the other, which ptxas can interleave to fill
	// the tensor pipe's issue stalls. One barrier a trip covers both hazards.
	sizeA, sizeB := c(int64(BM*kHalf)), c(int64(BN*kHalf))
	// nextX is each item's carried word for the trip after sub.
	nextX := func(x []ir.Value, ra [][]ir.Value, sub ir.Value) []ir.Value {
		if !carry {
			return nil
		}
		out := make([]ir.Value, nA)
		for i := range out {
			out[i] = dq.scNext(x[i], ra[i][words+1], add(b.Add(ir.U32, sub, aSb0), aC[i].sb), t.KB)
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
	curA := b.Phi(ir.U32, zero) // the buffer this trip's MMAs read
	curB := b.Phi(ir.U32, zero)
	subR := b.Phi(ir.U32, sub1) // the trip the registers hold
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
		// One 32-element step of the trip: chunk 4ks+q of each row.
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
			w0, w1 := 2*st, 2*st+1
			for i := 0; i < t.MT; i++ {
				for j := 0; j < t.NT; j++ {
					if t.F16K == 16 {
						d[i][j] = b.MMA(ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16},
							[]ir.Value{a0[i][w0], a1[i][w0], a0[i][w1], a1[i][w1]},
							[]ir.Value{bf[j][w0], bf[j][w1]}, d[i][j])
						continue
					}
					// m16n8k8 takes one word of each operand: the pair q.
					for _, w := range []int{w0, w1} {
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
		// One buffer: every warp's fragment loads are done before it is
		// overwritten.
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
	// The carried word's update is emitted last: it is only needed by the
	// next trip, and emitting it beside stage stretched its live range over
	// the loads and the barrier.
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

	outSeg := b.Mul(ir.U32, seg, c(int64(s.NTok*s.Rows)))
	var first ir.Value
	if s.Bias && split > 1 {
		first = b.Lt(ir.U32, seg, c(1))
	}
	rowG := b.Add(ir.U32, rowBlk, rowW)
	tokG := b.Add(ir.U32, tokBlk, tokW)
	// rowOf and tokOf are accumulator comp's output row and token, before the
	// tile offsets.
	var rowOf, tokOf func(comp int) ir.Value
	if t.F16K == 0 {
		dRow := []ir.Value{b.Add(ir.U32, rowG, v.dRow(0)), b.Add(ir.U32, rowG, v.dRow(2))}
		dCol := []ir.Value{b.Add(ir.U32, tokG, v.dCol(0)), b.Add(ir.U32, tokG, v.dCol(1)),
			b.Add(ir.U32, tokG, v.dCol(4)), b.Add(ir.U32, tokG, v.dCol(5))}
		rowOf = func(comp int) ir.Value { return dRow[(comp>>1)&1] }
		tokOf = func(comp int) ir.Value { return dCol[(comp&1)+2*(comp>>2)] }
	} else {
		// m16n8: c0 and c1 are row g at tokens 2q and 2q+1, c2 and c3 row g+8.
		r0 := b.Add(ir.U32, rowG, lg)
		dRow := []ir.Value{r0, add(r0, 8)}
		t0 := b.Add(ir.U32, tokG, b.Shl(ir.U32, lq, c(1)))
		dCol := []ir.Value{t0, add(t0, 1)}
		rowOf = func(comp int) ir.Value { return dRow[comp>>1] }
		tokOf = func(comp int) ir.Value { return dCol[comp&1] }
	}
	nrows := c(int64(s.Rows))
	for i := 0; i < t.MT; i++ {
		for j := 0; j < t.NT; j++ {
			for comp := 0; comp < nAcc; comp++ {
				rr := add(rowOf(comp), int64(t.mRows()*i))
				tok := add(tokOf(comp), int64(8*j))
				val := d[i][j][comp]
				if s.Bias {
					// A grouped bias is a bank, [expert][row]; adding it here
					// saves an elementwise pass over the sorted output.
					bi := rr
					if grouped {
						bi = b.Add(ir.U32, b.Mul(ir.U32, eSel, c(int64(s.Rows))), rr)
					}
					bv := b.Load(ir.F32, pBias, bi, 0)
					if first != 0 {
						bv = b.Select(ir.F32, first, bv, zeroF)
					}
					val = b.Add(ir.F32, val, bv)
				}
				b.Store(pOut, b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, tok, nrows), rr), outSeg), val, 0)
			}
		}
	}
	if into != nil {
		return nil, nil
	}
	return b.Done(), nil
}

// GemmVoltaSegments is several GemmVolta GEMMs over one binary16 operand in
// one launch: a block's q, k and v, whose weights may be different formats
// and blockings as long as their workgroups are one width. Segment i owns a
// contiguous run of workgroups (GemmVoltaSegmentsGrid) and runs GemmVolta's
// own body for its shape there, guarded by a loop that runs once in its
// workgroups and never in the others, as MatVecSegments does. Every segment
// is unsplit: a small k or v projection that alone would split k to fill the
// card (and reduce its partials) rides q's grid instead.
//
// Parameters: pB (the shared operand), then per segment pQS, pD, pSC, pOut
// and, when the segment has one, pBias. The staging buffers are declared
// once, at the largest segment's size.
func GemmVoltaSegments(segs []MatVecShape, tiles []VoltaTile) (*ir.Kernel, error) {
	blocks, nth, err := GemmVoltaSegmentsGrid(segs, tiles)
	if err != nil {
		return nil, err
	}
	name := "gemmvoltaseg"
	shA, shB := 0, 0
	for i, sg := range segs {
		t := tiles[i]
		qi := qtab[sg.T]
		kHalf := t.KB * qi.sub / 2
		nbuf := 2
		if t.Single {
			nbuf = 1
		}
		shA, shB = max(shA, nbuf*t.Rows()*kHalf), max(shB, nbuf*t.Toks()*kHalf)
		name += fmt.Sprintf("_%s_%d_m%dn%dk%d", strings.ToLower(sg.T.String()), sg.Rows, t.MT, t.NT, t.KB)
	}
	name += fmt.Sprintf("x%d_t%d", segs[0].K, segs[0].NTok)
	if sh := 4 * (shA + shB); sh > 48<<10 {
		return nil, fmt.Errorf("kernels: GemmVoltaSegments: %d bytes of shared memory, over the 48 KiB a kernel may declare", sh)
	}
	b := ir.New(name, [3]int{nth, 1, 1})
	ctx := gemmCtx{b: b, pB: b.Param("pB", ir.U32)}
	ctx.shA = b.Shared("gemmA", ir.U32, shA)
	ctx.shB = b.Shared("gemmB", ir.U32, shB)
	start := 0
	for i, sg := range segs {
		lo, hi := b.Const(ir.U32, int64(start)), b.Const(ir.U32, int64(start+blocks[i]))
		c := b.CTAID()
		zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
		b.LoopN(b.Select(ir.U32, b.Lt(ir.U32, c, hi), b.Select(ir.U32, b.Lt(ir.U32, c, lo), zero, one), zero))
		seg := ctx
		seg.suffix = fmt.Sprintf("%d", i)
		seg.cta = func() ir.Value { return b.Sub(ir.U32, b.CTAID(), lo) }
		if _, err := gemmVolta(sg, tiles[i], &seg); err != nil {
			return nil, fmt.Errorf("kernels: GemmVoltaSegments: segment %d: %w", i, err)
		}
		b.EndLoop()
		start += blocks[i]
	}
	return b.Done(), nil
}

// GemmVoltaSegmentsGrid is GemmVoltaSegments' launch: the workgroups each
// segment owns, in order, and the workgroup width. It refuses what the kernel
// cannot express: one segment, a split or grouped one, segments over
// different operands (K, NTok, or the activation layout a format reads), or
// tiles of different widths.
func GemmVoltaSegmentsGrid(segs []MatVecShape, tiles []VoltaTile) (blocks []int, nth int, err error) {
	if len(segs) < 2 || len(tiles) != len(segs) {
		return nil, 0, fmt.Errorf("kernels: GemmVoltaSegments: %d segment(s), %d tile(s)", len(segs), len(tiles))
	}
	nth = tiles[0].Threads()
	for i, sg := range segs {
		t := tiles[i]
		switch {
		case !GemmVoltaOK(sg.T) || qtab[sg.T].float > 0:
			return nil, 0, fmt.Errorf("kernels: GemmVoltaSegments: segment %d: %v", i, sg.T)
		case sg.Experts > 1 || max(sg.Split, 1) > 1:
			return nil, 0, fmt.Errorf("kernels: GemmVoltaSegments: segment %d is split or grouped", i)
		case sg.K != segs[0].K || sg.NTok != segs[0].NTok || qtab[sg.T].sub != qtab[segs[0].T].sub:
			return nil, 0, fmt.Errorf("kernels: GemmVoltaSegments: segment %d reads another operand", i)
		case t.Threads() != nth || t.F16K != tiles[0].F16K:
			return nil, 0, fmt.Errorf("kernels: GemmVoltaSegments: segment %d's tile is another width", i)
		case sg.Rows%t.Rows() != 0 || sg.NTok%t.Toks() != 0:
			return nil, 0, fmt.Errorf("kernels: GemmVoltaSegments: segment %d: tile does not divide it", i)
		}
		blocks = append(blocks, GemmVoltaGroups(sg, t))
	}
	return blocks, nth, nil
}
