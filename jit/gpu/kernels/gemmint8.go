package kernels

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Int8Tile is GemmInt8's blocking: a workgroup of WM*WN warps, each warp MT
// m-tiles of 16 rows by NT n-tiles of 8 tokens. The workgroup covers
// WM*MT*16 rows by WN*NT*8 tokens, 32*D elements of k a trip (D 0 is 1).
//
// D > 1 is llama.cpp MMQ's deep stage: one barrier pair amortised over
// several 32-element steps, the staged rows padded by eight words so a
// fragment load's four rows meet four bank groups. Single stages one buffer
// (MMAs, a barrier, the next trip's stage, a barrier) where two would not fit.
// M8 multiplies on m8n8k16, the int8 instruction sm_75 has (MMAS8Turing):
// the m16n8 fragment's words are that shape's registers side by side, so a
// tile is two (16-element sub-blocks) or four m8n8k16 a step.
type Int8Tile struct {
	MT, NT, WM, WN int
	D              int
	Single, M8     bool
	// LdM stages the weights in natural k order, rows padded to an odd
	// count of 16-byte chunks, and reads each m-tile's A fragments with one
	// ldmatrix.x4 a step (ir.Builder.LdMatrix) instead of four 8-byte loads.
	LdM bool
}

func (t Int8Tile) depth() int { return max(t.D, 1) }

// Rows and Toks are the workgroup's block of the output.
func (t Int8Tile) Rows() int { return t.WM * t.MT * 16 }
func (t Int8Tile) Toks() int { return t.WN * t.NT * 8 }

// Threads is the workgroup width.
func (t Int8Tile) Threads() int { return 32 * t.WM * t.WN }

// GemmInt8Groups is how many workgroups GemmInt8 launches.
func GemmInt8Groups(s MatVecShape, t Int8Tile) int {
	return (s.Rows / t.Rows()) * (s.NTok / t.Toks()) * max(s.Split, 1)
}

// Int8OK reports whether GemmInt8 has a decode for q: the formats GemmVolta
// dequantizes, less those whose code is not an integer (MXFP4), plus those
// GemmVolta declines only for their int8 activations (TQ2), which are this
// kernel's.
func Int8OK(q Quant) bool {
	return volta70Packed(q) && qtab[q].codes == nil
}

// int8Pos is where natural word w (k 4w..4w+3) of a row's 32-element trip
// sits in shared memory: words w and w+4 side by side, so lane q of a
// fragment reads both of its words -- k 4q.. and 16+4q.. -- as one 8-byte
// load. For a 16-element sub-block those are the same lane's words of the
// trip's two sub-blocks.
func int8Pos(w int) int { return (w%4)*2 + w/4 }

// GemmInt8 is the batched quantized matvec on the int8 matrix instruction
// (m16n8k32, or m16n8k16 per 16-element sub-block; sm_80 on) with both
// operands staged through shared memory, as llama.cpp's MMQ does: MatVecMMA's
// arithmetic, blocked like GemmVolta.
//
// MatVecMMA gives every warp its own rows and reads the packed weights and
// the activations from global memory inside the MMA loop, so a 512-row chunk
// decodes every weight sixteen times. Here a workgroup decodes a trip's 32
// elements of its rows once -- to signed bytes, a centred format's zero point
// taken off the weights, a stored minimum kept apart -- with each row's
// float32 scale and minimum per sub-block beside them, stages the matching
// int8 activations with their block scale and sums, and every warp reads its
// fragments from there. The next trip's global words are loaded into
// registers before this trip's MMAs, double-buffered as GemmVolta is.
//
// Each instruction covers one sub-block, so its integer dot is complete and
// the per-sub-block float epilogue is MatVecMMA's: scale*dot (less the
// minimum times the activations' sum), times the activation block's scale.
//
// Parameters are MatVecMMA's: pQS, pD, pSC, pA, pAX, pOut and pBias. With
// Split > 1 each segment writes its own slice of pOut (Reduce sums them) and
// only the first adds the bias. The grid is exact (GemmInt8Groups workgroups
// of t.Threads()).
func GemmInt8(s MatVecShape, t Int8Tile) (*ir.Kernel, error) {
	if int(s.T) >= len(qtab) || !Int8OK(s.T) {
		return nil, fmt.Errorf("kernels: GemmInt8: %v has no int8 decode", s.T)
	}
	qi := qtab[s.T]
	if t.MT < 1 || t.NT < 1 || t.WM < 1 || t.WN < 1 {
		return nil, fmt.Errorf("kernels: GemmInt8: tile %+v", t)
	}
	if s.Experts > 1 {
		return nil, fmt.Errorf("kernels: GemmInt8: a grouped shape is not supported")
	}
	split := max(s.Split, 1)
	BM, BN, nth := t.Rows(), t.Toks(), t.Threads()
	if s.Rows%BM != 0 || s.NTok%BN != 0 {
		return nil, fmt.Errorf("kernels: GemmInt8: %dx%d block does not divide rows=%d ntok=%d", BM, BN, s.Rows, s.NTok)
	}
	if s.K%qi.blockE != 0 || s.K <= 0 || s.K%32 != 0 {
		return nil, fmt.Errorf("kernels: GemmInt8: K=%d is not a multiple of %d", s.K, qi.blockE)
	}
	if qi.sub != 32 && qi.sub != 16 {
		return nil, fmt.Errorf("kernels: GemmInt8: sub-block %d", qi.sub)
	}
	D := t.depth()    // 32-element steps a trip
	SB := 32 / qi.sub // sub-blocks a step
	KB := D * SB      // sub-blocks a trip
	nsub := s.K / qi.sub
	if nsub%(split*KB) != 0 {
		return nil, fmt.Errorf("kernels: GemmInt8: split %d does not divide %d trips", split, nsub/KB)
	}
	trips := nsub / split / KB
	words := qi.sub*qi.bits/32 + qi.sub*qi.hi/32
	// RS is a staged row's words: 8 a step, padded by 8 past one step so the
	// rows g..g+3 a fragment load's phase names start in four bank groups.
	RS := 8 * D
	if D > 1 {
		RS += 8
	}
	// RSA is a staged weight row's words: RS, or with LdM 8D+4, an odd
	// number of 16-byte chunks so ldmatrix's eight row reads a matrix meet
	// eight bank groups.
	RSA := RS
	if t.LdM {
		RSA = 8*D + 4
	}
	itemsA, itemsB := BM*KB, BN*2*D // (row, sub-block) decodes; 16-byte activation chunks
	if itemsA%nth != 0 && nth%itemsA != 0 || itemsB%nth != 0 && nth%itemsB != 0 {
		return nil, fmt.Errorf("kernels: GemmInt8: %d threads do not tile %d A items and %d B chunks", nth, itemsA, itemsB)
	}
	nA, nB := max(itemsA/nth, 1), max(itemsB/nth, 1)
	centered := qi.biasK != 0 && !qi.biasArray
	mins := qi.biasArray
	nbq := s.K / 32 // activation blocks a token
	nbuf := 2
	if t.Single {
		nbuf = 1
	}
	shWords := nbuf * (BM*RSA + BN*RS + KB*BM + D*BN)
	if mins {
		shWords += nbuf * (KB*BM + KB*BN)
	}
	if shWords*4 > 48<<10 {
		return nil, fmt.Errorf("kernels: GemmInt8: %d bytes of shared memory, over the 48 KiB a kernel may declare", shWords*4)
	}

	name := fmt.Sprintf("gemmint8_%s_%dx%d_t%d_m%dn%dw%dx%d_s%d", strings.ToLower(s.T.String()),
		s.Rows, s.K, s.NTok, t.MT, t.NT, t.WM, t.WN, split)
	if D > 1 {
		name += fmt.Sprintf("_d%d", D)
	}
	if t.Single {
		name += "_1b"
	}
	if t.M8 {
		name += "_m8"
	}
	if t.LdM {
		name += "_ldm"
	}
	b := ir.New(name, [3]int{nth, 1, 1})
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
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	add := func(v ir.Value, n int64) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, c(n))
	}
	// Shared memory, two buffers each: the weights (8 words a row), their
	// scales and minimums ([sub-block][row]), the activations (8 words a
	// token), their block scale and their sums ([sub-block][token]).
	shA := b.Shared("i8A", ir.U32, nbuf*BM*RSA)
	shB := b.Shared("i8B", ir.U32, nbuf*BN*RS)
	shSc := b.Shared("i8Sc", ir.F32, nbuf*KB*BM)
	shDa := b.Shared("i8Da", ir.F32, nbuf*D*BN)
	var shMn, shSum ir.Value
	if mins {
		shMn = b.Shared("i8Mn", ir.F32, nbuf*KB*BM)
		shSum = b.Shared("i8Sum", ir.F32, nbuf*KB*BN)
	}

	mB, nBk := s.Rows/BM, s.NTok/BN
	grp := b.CTAID()
	bn := b.Rem(ir.U32, grp, c(int64(nBk)))
	bm := b.Rem(ir.U32, b.Div(ir.U32, grp, c(int64(nBk))), c(int64(mB)))
	seg := b.Div(ir.U32, grp, c(int64(mB*nBk)))
	rowBlk := b.Mul(ir.U32, bm, c(int64(BM)))
	tokBlk := b.Mul(ir.U32, bn, c(int64(BN)))
	tid := b.TID()

	dq := newVoltaDeq(b, s.T, pQS, pD, pSC, s.Rows)

	// A staging: item it = tid + nth*i (mod itemsA when the items are
	// fewer, the surplus threads restaging an item to the same addresses) is
	// row it%BM, sub-block it/BM of the trip.
	itemA := tid
	if itemsA < nth {
		itemA = b.Rem(ir.U32, tid, c(int64(itemsA)))
	}
	aRow := make([]ir.Value, nA) // the row within the block
	aSb := make([]ir.Value, nA)  // the sub-block within the trip
	for i := range aRow {
		it := add(itemA, int64(i*nth))
		aRow[i], aSb[i] = b.Rem(ir.U32, it, c(int64(BM))), b.Div(ir.U32, it, c(int64(BM)))
	}
	// B staging: chunk e = tid + nth*i is token e/(2D), part e%(2D): step
	// part/2, words 4(part%2)..+3 of it.
	itemB := tid
	if itemsB < nth {
		itemB = b.Rem(ir.U32, tid, c(int64(itemsB)))
	}
	bTok := make([]ir.Value, nB)
	bHalf := make([]ir.Value, nB)
	bStep := make([]ir.Value, nB)
	bPart := make([]ir.Value, nB)
	for i := range bTok {
		e := add(itemB, int64(i*nth))
		bTok[i], bPart[i] = b.Div(ir.U32, e, c(int64(2*D))), b.Rem(ir.U32, e, c(int64(2*D)))
		bHalf[i], bStep[i] = b.And(ir.U32, bPart[i], c(1)), b.Shr(ir.U32, bPart[i], c(1))
	}

	// load issues one trip's global loads into registers: each A item's
	// packed words then its scale words, each B chunk as one 16-byte load,
	// and the chunk's token's block scale and sums.
	type tripRegs struct {
		a  [][]ir.Value // packed words then scale words, per A item
		bw [][]ir.Value // four activation words per B chunk
		bd []ir.Value   // block scale per B chunk
		bs [][]ir.Value // sums per B chunk, KB of them
	}
	load := func(sub ir.Value) tripRegs {
		var r tripRegs
		for i := 0; i < nA; i++ {
			sb := b.Add(ir.U32, sub, aSb[i])
			row := b.Add(ir.U32, rowBlk, aRow[i])
			idx := b.Add(ir.U32, b.Mul(ir.U32, sb, c(int64(words*s.Rows))), row)
			r.a = append(r.a, append(dq.words(idx, 0), dq.scaleWords(sb, row)...))
		}
		// blk is the 32-element activation block a chunk's step reads.
		blk0 := b.Shr(ir.U32, sub, c(int64(log2(SB))))
		for i := 0; i < nB; i++ {
			tok := b.Add(ir.U32, tokBlk, bTok[i])
			blk := b.Add(ir.U32, blk0, bStep[i])
			wi := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, tok, c(int64(s.K/4))), b.Shl(ir.U32, blk0, c(3))),
				b.Shl(ir.U32, bPart[i], c(2)))
			r.bw = append(r.bw, b.LoadV(ir.U32, pA, wi, 0, 4))
			r.bd = append(r.bd, b.Load(ir.F32, pAX, b.Add(ir.U32, b.Mul(ir.U32, tok, c(int64(nbq))), blk), 0))
			if mins {
				// Per-16 sums: [token][K/16] after the block scales.
				si := b.Add(ir.U32, b.Mul(ir.U32, tok, c(int64(s.K/16))), b.Shl(ir.U32, blk, c(1)))
				s0 := b.Load(ir.F32, pAX, si, int64(s.NTok*nbq))
				s1 := b.Load(ir.F32, pAX, si, int64(s.NTok*nbq+1))
				if SB == 1 {
					r.bs = append(r.bs, []ir.Value{b.Add(ir.F32, s0, s1)})
				} else {
					r.bs = append(r.bs, []ir.Value{s0, s1})
				}
			}
		}
		return r
	}
	mask := c(0x0F0F0F0F)
	var cAdd, cXor ir.Value
	if centered {
		bk := int64(qi.biasK)
		v := (0x80 - bk) & 0xFF
		cAdd, cXor = c(v|v<<8|v<<16|v<<24), c(0x80808080)
	}
	center := func(v ir.Value) ir.Value {
		if !centered {
			return v
		}
		return b.Xor(ir.U32, b.Add(ir.U32, v, cAdd), cXor)
	}
	// decode turns one item's packed words into its sub-block's int8 words
	// in natural order (word w is elements 4w..4w+3): sub words of 32, a
	// half trip of 16.
	decode := func(raw []ir.Value) []ir.Value {
		n := qi.sub / 4
		out := make([]ir.Value, n)
		if qi.bits == 8 {
			for w := range out {
				out[w] = center(raw[w])
			}
			return out
		}
		pw := qi.sub * qi.bits / 32
		sec := raw[pw:]
		lanes := 0
		if qi.hi > 0 {
			lanes = 8 / qi.hi
		}
		hm := c(int64(rep4(uint32(1<<uint(qi.hi) - 1))))
		half := qi.sub / 8 // words from one primary word's low (high) nibbles: 4 at sub 32, 2 at 16
		for w := 0; w < pw; w++ {
			for h := 0; h < 2; h++ {
				v := raw[w]
				if h == 1 {
					v = b.Shr(ir.U32, v, c(4))
				}
				v = b.And(ir.U32, v, mask)
				g := w + h*half // the natural word, which is the secondary group
				if qi.hi > 0 {
					sh := sec[g/lanes]
					if s := qi.hi * (g % lanes); s != 0 {
						sh = b.Shr(ir.U32, sh, c(int64(s)))
					}
					v = b.Xor(ir.U32, v, b.Shl(ir.U32, b.And(ir.U32, sh, hm), c(int64(qi.bits))))
				}
				out[g] = center(v)
			}
		}
		return out
	}
	// stage decodes one trip's registers and stores them into buffer buf
	// (0 or 1): the ALU work first, the stores last.
	stage := func(r tripRegs, sub, buf ir.Value) {
		type aOut struct {
			w      []ir.Value
			sc, mn ir.Value
		}
		outs := make([]aOut, nA)
		for i := 0; i < nA; i++ {
			sb := b.Add(ir.U32, sub, aSb[i])
			row := b.Add(ir.U32, rowBlk, aRow[i])
			sc, mn := dq.scalesF32X(r.a[i][words:], 0, sb, row)
			outs[i] = aOut{decode(r.a[i][:words]), sc, mn}
		}
		baseA := b.Mul(ir.U32, buf, c(int64(BM*RSA)))
		baseS := b.Mul(ir.U32, buf, c(int64(KB*BM)))
		for i, o := range outs {
			rw := b.Add(ir.U32, baseA, b.Mul(ir.U32, aRow[i], c(int64(RSA))))
			switch {
			case t.LdM:
				// Natural order, sub-block aSb's words from word aSb*len:
				// 8 a step, a 16-element sub-block l of step s at 8s + 4l.
				p := b.Add(ir.U32, rw, b.Shl(ir.U32, aSb[i], c(int64(log2(len(o.w))))))
				for x := 0; x < len(o.w); x += 4 {
					b.StoreV(shA, p, int64(x), o.w[x], o.w[x+1], o.w[x+2], o.w[x+3])
				}
			case SB == 1:
				// Step aSb's natural words 0..7 at int8Pos: (w0 w4 w1 w5) (w2 w6 w3 w7).
				if D > 1 {
					rw = b.Add(ir.U32, rw, b.Shl(ir.U32, aSb[i], c(3)))
				}
				b.StoreV(shA, rw, 0, o.w[0], o.w[4], o.w[1], o.w[5])
				b.StoreV(shA, rw, 4, o.w[2], o.w[6], o.w[3], o.w[7])
			default:
				// Sub-block sb is half sb&1 of step sb>>1: its natural words
				// at positions 2x + (sb&1) of the step.
				p := b.Add(ir.U32, rw, b.And(ir.U32, aSb[i], c(1)))
				if D > 1 {
					p = b.Add(ir.U32, p, b.Shl(ir.U32, b.Shr(ir.U32, aSb[i], c(1)), c(3)))
				}
				for x, v := range o.w {
					b.Store(shA, p, v, int64(2*x))
				}
			}
			si := b.Add(ir.U32, b.Add(ir.U32, baseS, b.Mul(ir.U32, aSb[i], c(int64(BM)))), aRow[i])
			b.Store(shSc, si, o.sc, 0)
			if mins {
				b.Store(shMn, si, o.mn, 0)
			}
		}
		baseB := b.Mul(ir.U32, buf, c(int64(BN*RS)))
		for i := 0; i < nB; i++ {
			// Chunk h's natural words 4h+x go to position 2x+h of its step.
			p := b.Add(ir.U32, b.Add(ir.U32, baseB, b.Mul(ir.U32, bTok[i], c(int64(RS)))), bHalf[i])
			if D > 1 {
				p = b.Add(ir.U32, p, b.Shl(ir.U32, bStep[i], c(3)))
			}
			for x, v := range r.bw[i] {
				b.Store(shB, p, v, int64(2*x))
			}
			// The step's scale and sums for the token, written by both its
			// chunks alike: [buf][step][token] and [buf][sub-block][token].
			db := b.Add(ir.U32, b.Mul(ir.U32, buf, c(int64(D*BN))), bTok[i])
			if D > 1 {
				db = b.Add(ir.U32, db, b.Mul(ir.U32, bStep[i], c(int64(BN))))
			}
			b.Store(shDa, db, r.bd[i], 0)
			if mins {
				sbB := b.Add(ir.U32, b.Mul(ir.U32, buf, c(int64(KB*BN))), bTok[i])
				if D > 1 {
					sbB = b.Add(ir.U32, sbB, b.Mul(ir.U32, bStep[i], c(int64(SB*BN))))
				}
				for sb := 0; sb < SB; sb++ {
					b.Store(shSum, sbB, r.bs[i][sb], int64(sb*BN))
				}
			}
		}
	}

	// The fragment bases: warp (wm, wn), lane (g, q).
	warp := b.Shr(ir.U32, tid, c(5))
	wm := b.Rem(ir.U32, warp, c(int64(t.WM)))
	wn := b.Div(ir.U32, warp, c(int64(t.WM)))
	lane := b.And(ir.U32, tid, c(31))
	lg := b.Shr(ir.U32, lane, c(2))
	lq := b.And(ir.U32, lane, c(3))
	rowW := b.Mul(ir.U32, wm, c(int64(16*t.MT)))
	tokW := b.Mul(ir.U32, wn, c(int64(8*t.NT)))
	rowL := b.Add(ir.U32, rowW, lg) // + 8h + 16i
	tokL := b.Add(ir.U32, tokW, lg) // + 8j
	fragA := b.Add(ir.U32, b.Mul(ir.U32, rowL, c(int64(RSA))), b.Shl(ir.U32, lq, c(1)))
	if t.LdM {
		// Lane l names row l%8 of matrix l/8 = h + 2kh: row 8h + l%8 of the
		// m-tile, words 4kh..4kh+3 of the step.
		mat := b.Shr(ir.U32, lane, c(3))
		r8 := b.Add(ir.U32, b.Add(ir.U32, rowW, b.And(ir.U32, lane, c(7))), b.Shl(ir.U32, b.And(ir.U32, mat, c(1)), c(3)))
		fragA = b.Add(ir.U32, b.Mul(ir.U32, r8, c(int64(RSA))), b.Shl(ir.U32, b.Shr(ir.U32, mat, c(1)), c(2)))
	}
	fragB := b.Add(ir.U32, b.Mul(ir.U32, tokL, c(int64(RS))), b.Shl(ir.U32, lq, c(1)))
	tokE := b.Add(ir.U32, tokW, b.Shl(ir.U32, lq, c(1))) // the D fragment's first token, + 8j

	sub0 := b.Mul(ir.U32, seg, c(int64(trips*KB)))
	last := add(sub0, int64((trips-1)*KB))
	advance := func(sub ir.Value) ir.Value { return b.Min(ir.U32, add(sub, int64(KB)), last) }
	r0 := load(sub0)
	stage(r0, sub0, c(0))
	sub1 := advance(sub0)
	r1 := load(sub1)
	b.Barrier()

	zeroF := b.ConstF32(0)
	zeroI := b.Const(ir.I32, 0)
	zeroU := c(0)
	b.Loop(int64(trips))
	acc := make([][][]ir.Value, t.MT)
	for i := range acc {
		acc[i] = make([][]ir.Value, t.NT)
		for j := range acc[i] {
			acc[i][j] = []ir.Value{b.Phi(ir.F32, zeroF), b.Phi(ir.F32, zeroF), b.Phi(ir.F32, zeroF), b.Phi(ir.F32, zeroF)}
		}
	}
	cur := b.Phi(ir.U32, zeroU) // the buffer this trip's MMAs read
	subR := b.Phi(ir.U32, sub1)
	phiRegs := func(r tripRegs) tripRegs {
		var o tripRegs
		ph := func(vs []ir.Value, t ir.Type) []ir.Value {
			out := make([]ir.Value, len(vs))
			for i, v := range vs {
				out[i] = b.Phi(t, v)
			}
			return out
		}
		for _, a := range r.a {
			o.a = append(o.a, ph(a, ir.U32))
		}
		for _, w := range r.bw {
			o.bw = append(o.bw, ph(w, ir.U32))
		}
		o.bd = ph(r.bd, ir.F32)
		for _, s := range r.bs {
			o.bs = append(o.bs, ph(s, ir.F32))
		}
		return o
	}
	rr := phiRegs(r1)

	// d is the trip's running sums, a copy of the phis: the epilogue writes
	// it element by element, and an aliased slice would overwrite the phis
	// SetPhi needs below.
	d := make([][][]ir.Value, t.MT)
	for i := range d {
		d[i] = make([][]ir.Value, t.NT)
		for j := range d[i] {
			d[i][j] = append([]ir.Value(nil), acc[i][j]...)
		}
	}
	pa := b.Add(ir.U32, fragA, b.Mul(ir.U32, cur, c(int64(BM*RSA))))
	pb := b.Add(ir.U32, fragB, b.Mul(ir.U32, cur, c(int64(BN*RS))))
	// The scales this lane's D fragments need: rows g and g+8 of each m-tile
	// per sub-block, tokens 2q and 2q+1 of each n-tile.
	scBase := b.Add(ir.U32, b.Mul(ir.U32, cur, c(int64(KB*BM))), rowL)
	daBase := b.Add(ir.U32, b.Mul(ir.U32, cur, c(int64(D*BN))), tokE)
	sumBase := b.Add(ir.U32, b.Mul(ir.U32, cur, c(int64(KB*BN))), tokE)
	m8 := func(a, bw ir.Value, c2 []ir.Value) []ir.Value {
		return b.MMA(ir.MMAS8Turing, []ir.Value{a}, []ir.Value{bw}, c2)
	}
	for st := 0; st < D; st++ {
		af := make([][2][]ir.Value, t.MT) // [m-tile][row g, g+8] -> (word q, word q+4)
		for i := range af {
			if t.LdM {
				// Components h + 2kh: af's (word q, word q+4) per row half.
				m := b.LdMatrix(shA, pa, int64(16*i*RSA+8*st), 4)
				af[i][0] = []ir.Value{m[0], m[2]}
				af[i][1] = []ir.Value{m[1], m[3]}
				continue
			}
			for h := 0; h < 2; h++ {
				af[i][h] = b.LoadV(ir.U32, shA, pa, int64((16*i+8*h)*RSA+8*st), 2)
			}
		}
		bf := make([][]ir.Value, t.NT)
		for j := range bf {
			bf[j] = b.LoadV(ir.U32, shB, pb, int64(8*j*RS+8*st), 2)
		}
		da := make([][2]ir.Value, t.NT)
		for j := range da {
			for x := 0; x < 2; x++ {
				da[j][x] = b.Load(ir.F32, shDa, daBase, int64(st*BN+8*j+x))
			}
		}
		for l := 0; l < SB; l++ {
			sb := st*SB + l
			sc := make([][2]ir.Value, t.MT)
			mn := make([][2]ir.Value, t.MT)
			for i := range sc {
				for h := 0; h < 2; h++ {
					sc[i][h] = b.Load(ir.F32, shSc, scBase, int64(sb*BM+16*i+8*h))
					if mins {
						mn[i][h] = b.Load(ir.F32, shMn, scBase, int64(sb*BM+16*i+8*h))
					}
				}
			}
			var sum [][2]ir.Value
			if mins {
				sum = make([][2]ir.Value, t.NT)
				for j := range sum {
					for x := 0; x < 2; x++ {
						sum[j][x] = b.Load(ir.F32, shSum, sumBase, int64(sb*BN+8*j+x))
					}
				}
			}
			for i := 0; i < t.MT; i++ {
				for j := 0; j < t.NT; j++ {
					var d4 []ir.Value
					z4 := []ir.Value{zeroI, zeroI, zeroI, zeroI}
					switch {
					case t.M8:
						// Row half h on m8n8k16, one per 16 elements of k: the
						// whole step (both words) for a 32-element sub-block,
						// word l for half l of a 16-element pair.
						d4 = make([]ir.Value, 0, 4)
						for h := 0; h < 2; h++ {
							acc2 := []ir.Value{zeroI, zeroI}
							if SB == 1 {
								acc2 = m8(af[i][h][0], bf[j][0], acc2)
								acc2 = m8(af[i][h][1], bf[j][1], acc2)
							} else {
								acc2 = m8(af[i][h][l], bf[j][l], acc2)
							}
							d4 = append(d4, acc2...)
						}
					case SB == 1:
						d4 = b.MMA(ir.MMAShape{M: 16, N: 8, K: 32},
							[]ir.Value{af[i][0][0], af[i][1][0], af[i][0][1], af[i][1][1]},
							[]ir.Value{bf[j][0], bf[j][1]}, z4)
					default:
						d4 = b.MMA(ir.MMAShape{M: 16, N: 8, K: 16},
							[]ir.Value{af[i][0][l], af[i][1][l]},
							[]ir.Value{bf[j][l]}, z4)
					}
					for x := 0; x < 4; x++ {
						h, tk := x>>1, x&1
						v := b.Mul(ir.F32, b.CvtF32(d4[x]), sc[i][h])
						if mins {
							v = b.Sub(ir.F32, v, b.Mul(ir.F32, mn[i][h], sum[j][tk]))
						}
						d[i][j][x] = b.Fma(v, da[j][tk], d[i][j][x])
					}
				}
			}
		}
	}
	nx := b.Sub(ir.U32, c(1), cur)
	if t.Single {
		// One buffer: every warp's fragment loads are done before it is
		// overwritten.
		nx = zeroU
		b.Barrier()
	}
	stage(rr, subR, nx)
	subN := advance(subR)
	rn := load(subN)
	b.Barrier()
	for i := range acc {
		for j := range acc[i] {
			for x := range acc[i][j] {
				b.SetPhi(acc[i][j][x], d[i][j][x])
			}
		}
	}
	b.SetPhi(cur, nx)
	b.SetPhi(subR, subN)
	setRegs := func(ph, v tripRegs) {
		for i := range ph.a {
			for w := range ph.a[i] {
				b.SetPhi(ph.a[i][w], v.a[i][w])
			}
		}
		for i := range ph.bw {
			for w := range ph.bw[i] {
				b.SetPhi(ph.bw[i][w], v.bw[i][w])
			}
		}
		for i := range ph.bd {
			b.SetPhi(ph.bd[i], v.bd[i])
		}
		for i := range ph.bs {
			for w := range ph.bs[i] {
				b.SetPhi(ph.bs[i][w], v.bs[i][w])
			}
		}
	}
	setRegs(rr, rn)
	b.EndLoop()

	// D fragment x: row g (+8 when x >= 2), token 2q + x%2.
	outSeg := b.Mul(ir.U32, seg, c(int64(s.NTok*s.Rows)))
	var first ir.Value
	if s.Bias && split > 1 {
		first = b.Lt(ir.U32, seg, c(1))
	}
	rowG := b.Add(ir.U32, rowBlk, rowL)
	tokG := b.Add(ir.U32, tokBlk, tokE)
	nrows := c(int64(s.Rows))
	for i := 0; i < t.MT; i++ {
		for j := 0; j < t.NT; j++ {
			for x := 0; x < 4; x++ {
				rr := add(rowG, int64(16*i+8*(x>>1)))
				tok := add(tokG, int64(8*j+x&1))
				val := d[i][j][x]
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
