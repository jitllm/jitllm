package ptx

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/composer/ir"
)

// The collective tile ops (ir/tile.go) on NVIDIA's mma.sync.
//
// A collective tile leaves the lane layout to the lowering, and mma.sync has a
// documented one, so the lowering chooses it: every tile is a grid of 8x8
// atoms, and a lane holds, of each atom, the two elements mma.sync's fragments
// give it. With g = lane>>2 and t = lane&3:
//
//	A (M x K, f16)   atom (i, j): row 8i+g, columns 8j+2t and 8j+2t+1, one f16x2
//	B (K x N, f16)   atom (i, j): rows 8i+2t and 8i+2t+1, column 8j+g, one f16x2
//	C (M x N, f32)   atom (i, j): row 8i+g, columns 8j+2t and 8j+2t+1, two f32
//
// These are exactly the registers of mma.sync.m16n8k16 (A's four are atoms
// (0,0), (1,0), (0,1), (1,1); B's two are (0,0), (1,0); C's four are atom
// (0,0)'s pair then (1,0)'s), so a TileMMA of any shape whose sides are
// multiples of 8 is a sequence of m16n8k16 (or m16n8k8) instructions over
// atoms, with no data movement between them. A side of 8 rows -- Metal's 8x8
// tile -- runs the m16 instruction with the second row of atoms zero and its
// half of the result dropped: correct, and half the matrix unit's work wasted.
//
// Loads and stores go element by element from the view's address, so a stride
// of zero (a bias broadcast down a column) and any alignment are just
// addresses. That is the simple form and its ceiling is the load count; the
// upgrade is ldmatrix for a shared view whose rows the kernel asserts are
// 16-byte aligned.
//
// The matrix instructions are sm_75 (m16n8k8) and sm_80 (m16n8k16). Only f16
// operands with an f32 accumulator are lowered below sm_80: s8 tiles, and f32
// operands there, are refused by name.
//
// sm_70 and sm_72 have one instruction, m8n8k4 (ir.MMAVolta()): each of the
// warp's four quad-pairs (lanes 4q..4q+3 and 16+4q..16+4q+3) multiplies its
// own 8x4 by 4x8. Its fragments are not m16n8's, so there the tile is laid out
// by k: quad-pair q holds k-chunks 4s+q (four elements of k each) of every
// atom, s counting 16 of k, and lane p of the pair (p = (lane&3) + 4(lane>>4))
// holds an A atom's row p and a B atom's column p:
//
//	A (M x K, f16)   row 8i+p, k 16s+4q .. 16s+4q+3, two f16x2
//	B (K x N, f16)   k 16s+4q .. 16s+4q+3, column 8j+p, two f16x2
//	C (M x N, f32)   atom (i, j)'s eight at m8n8k4's D places, the quad-pair's
//	                 partial sum: the tile is the sum over the four pairs
//
// so a TileMMA is one m8n8k4 an atom and 16 of k, every quad-pair adding its
// own chunks' products, and nothing moves between them until a store sums
// the four (two butterfly shuffles) and pair q writes registers 2q and 2q+1.
// A loaded or splat accumulator is held the same way, pair q holding those two
// and zeros. A k of 8 past the last 16 leaves pairs 2 and 3 zeros.
//
// Where every tile of the kernel has its rows and columns (an operand's
// outer side) in sixteens, the pairs split the output instead: block (i, j)
// of 16 x 16 is four atoms, pair q taking atom (q>>1, q&1), so with
// qr = 8(q>>1) and qc = 8(q&1)
//
//	A (M x K, f16)   row 16i+qr+p, k 4s .. 4s+3, two f16x2 a block and 4 of k
//	B (K x N, f16)   k 4s .. 4s+3, column 16j+qc+p, two f16x2
//	C (M x N, f32)   block (i, j)'s pair atom, eight at m8n8k4's D places
//
// and a TileMMA is one m8n8k4 a block and 4 of k. A pair's operands are
// another's too (the pairs of one row of atoms read one A), but the
// accumulator is a quarter of the k-split's and is the tile itself: a store
// writes it as it is.

// tileLower holds the per-kernel state the tile ops share.
type tileLower struct {
	b      *Buf
	sm     int
	g, t2  Reg // lane>>2 and 2*(lane&3), computed once at entry
	t1     Reg // lane&3, the f32 operands' column
	zeroH  Reg // an f16x2 zero, the missing half of an 8-row A
	zeroF  Reg // an f32 zero, the missing half of an 8-row accumulator
	regs   map[ir.Value][]Reg
	shared []bool
	// volta is sm_70's quad-pair layout: q the lane's pair, p its row (A) or
	// column (B), q4 4q, dr and dc accumulator 0's row and column, qIs[n]
	// q == n and q2 q < 2.
	volta  bool
	q, p   Reg
	q4     Reg
	dr, dc Reg
	qIs    [4]Reg
	q2     Reg
	// pairs is the output-split layout: pA and pB the A row and B column a
	// lane holds of a block, aR and aC its accumulator 0's, zero a zero
	// lane part.
	pairs          bool
	pA, pB, aR, aC Reg
	zero           Reg
}

// pairsFit reports whether every tile of k has its outer sides in sixteens,
// the output-split layout's blocks.
func pairsFit(k *ir.Kernel) bool {
	for _, o := range k.Ops {
		t := o.Tile
		if t == nil {
			continue
		}
		switch {
		case t.Use == ir.TileA && t.Rows%16 != 0, t.Use == ir.TileB && t.Cols%16 != 0,
			t.Use == ir.TileAcc && (t.Rows%16 != 0 || t.Cols%16 != 0):
			return false
		}
	}
	return true
}

// voltaSM reports whether sm's matrix instruction is m8n8k4 alone.
func voltaSM(sm int) bool { return sm >= 70 && sm < 75 }

// QuadPairTiles reports whether target lowers the collective tiles in the
// quad-pair layout.
func QuadPairTiles(target string) bool { return voltaSM(smOf(target)) }

// smOf parses "sm_86" as 86, or 0.
func smOf(target string) int {
	n, _ := strconv.Atoi(strings.TrimPrefix(target, "sm_"))
	return n
}

// tileSupported is the lowering's refusal for a tile type, nil if it lowers.
func tileSupported(t ir.TileType, sm int) error {
	if sm < 70 {
		return fmt.Errorf("ptx: no collective tile below sm_70 (this is sm_%d): mma.sync starts there", sm)
	}
	switch {
	case t.Use == ir.TileAcc && t.Elem == ir.TileF32:
		return nil
	case t.Use != ir.TileAcc && t.Elem == ir.TileF16:
		return nil
	case t.Use != ir.TileAcc && t.Elem == ir.TileF32 && sm >= 80:
		return nil
	}
	return fmt.Errorf("ptx: no %v on sm_%d: binary16 operands (sm_75+) and float32 operands "+
		"(sm_80+, as three tf32 products) into a float32 accumulator are lowered; an int8 tile "+
		"is a different instruction and is not built", t, sm)
}

// prologue computes the lane coordinates every tile access uses.
func (tl *tileLower) prologue() {
	b := tl.b
	lane := b.Reg(B32)
	b.Op("mov.u32 %v,%%laneid;", lane)
	tl.g = b.Reg(B32)
	b.Op("shr.u32 %v,%v,2;", tl.g, lane)
	tl.t1 = b.Reg(B32)
	b.Op("and.b32 %v,%v,3;", tl.t1, lane)
	tl.t2 = b.Reg(B32)
	b.Op("shl.b32 %v,%v,1;", tl.t2, tl.t1)
	tl.zeroH = b.Reg(B32)
	b.Op("mov.b32 %v,0;", tl.zeroH)
	tl.zeroF = b.Reg(F32)
	b.Op("mov.f32 %v,0f00000000;", tl.zeroF)
	if !tl.volta {
		return
	}
	tl.q = b.Reg(B32)
	b.Op("bfe.u32 %v,%v,2,2;", tl.q, lane)
	tl.q4 = b.Reg(B32)
	b.Op("shl.b32 %v,%v,2;", tl.q4, tl.q)
	hi := b.Reg(B32)
	b.Op("shr.u32 %v,%v,4;", hi, lane)
	b.Op("shl.b32 %v,%v,2;", hi, hi)
	tl.p = b.Reg(B32)
	b.Op("add.u32 %v,%v,%v;", tl.p, tl.t1, hi)
	tl.dr = b.Reg(B32)
	b.Op("and.b32 %v,%v,1;", tl.dr, lane)
	b.Op("add.u32 %v,%v,%v;", tl.dr, tl.dr, hi)
	tl.dc = b.Reg(B32)
	b.Op("and.b32 %v,%v,2;", tl.dc, lane)
	for n := range tl.qIs {
		tl.qIs[n] = b.Reg(Pred)
		b.Op("setp.eq.u32 %v,%v,%d;", tl.qIs[n], tl.q, n)
	}
	tl.q2 = b.Reg(Pred)
	b.Op("setp.lt.u32 %v,%v,2;", tl.q2, tl.q)
	if !tl.pairs {
		return
	}
	qr, qc := b.Reg(B32), b.Reg(B32)
	b.Op("and.b32 %v,%v,2;", qr, tl.q)
	b.Op("shl.b32 %v,%v,2;", qr, qr)
	b.Op("and.b32 %v,%v,1;", qc, tl.q)
	b.Op("shl.b32 %v,%v,3;", qc, qc)
	tl.pA, tl.pB, tl.aR, tl.aC, tl.zero = b.Reg(B32), b.Reg(B32), b.Reg(B32), b.Reg(B32), b.Reg(B32)
	b.Op("add.u32 %v,%v,%v;", tl.pA, tl.p, qr)
	b.Op("add.u32 %v,%v,%v;", tl.pB, tl.p, qc)
	b.Op("add.u32 %v,%v,%v;", tl.aR, tl.dr, qr)
	b.Op("add.u32 %v,%v,%v;", tl.aC, tl.dc, qc)
	b.Op("mov.u32 %v,0;", tl.zero)
}

// slots is how many 16s of k a Volta operand of k elements holds.
func slots(k int) int { return (k + 15) / 16 }

// count is how many registers a lane holds of a tile.
func (tl *tileLower) count(t ir.TileType) int {
	n := (t.Rows / 8) * (t.Cols / 8)
	if tl.pairs {
		switch t.Use {
		case ir.TileA:
			return 2 * (t.Rows / 16) * (t.Cols / 4)
		case ir.TileB:
			return 2 * (t.Rows / 4) * (t.Cols / 16)
		}
		return 8 * (t.Rows / 16) * (t.Cols / 16)
	}
	if tl.volta {
		switch t.Use {
		case ir.TileA:
			return 2 * (t.Rows / 8) * slots(t.Cols)
		case ir.TileB:
			return 2 * slots(t.Rows) * (t.Cols / 8)
		}
		return 8 * n
	}
	if t.Elem == ir.TileF32 {
		return 2 * n
	}
	return n
}

// fresh allocates a tile's registers.
func (tl *tileLower) fresh(t ir.TileType) []Reg {
	rs := make([]Reg, tl.count(t))
	for i := range rs {
		if t.Elem == ir.TileF32 {
			rs[i] = tl.b.Reg(F32)
		} else {
			rs[i] = tl.b.Reg(B32)
		}
	}
	return rs
}

// move copies one tile's registers into another's.
func (tl *tileLower) move(dst, src []Reg, t ir.TileType) {
	for i := range dst {
		if t.Elem == ir.TileF32 {
			tl.b.Op("mov.f32 %v,%v;", dst[i], src[i])
		} else {
			tl.b.Op("mov.b32 %v,%v;", dst[i], src[i])
		}
	}
}

// element is where a lane's element e (0 or 1) of atom (i, j) sits: its row
// and column as a lane part (a register) plus a constant.
func (tl *tileLower) element(t ir.TileType, i, j, e int) (rowL Reg, rowC int, colL Reg, colC int) {
	if t.Use != ir.TileAcc && t.Elem == ir.TileF32 {
		// The tf32 fragments of m16n8k8: A's lane holds (g, t) and (g, t+4)
		// of an atom, B's (t, g) and (t+4, g).
		if t.Use == ir.TileB {
			return tl.t1, 8*i + 4*e, tl.g, 8 * j
		}
		return tl.g, 8 * i, tl.t1, 8*j + 4*e
	}
	if t.Use == ir.TileB {
		return tl.t2, 8*i + e, tl.g, 8 * j
	}
	return tl.g, 8 * i, tl.t2, 8*j + e
}

// access loads (store == nil) or stores a tile through a view: element (r, c)
// is at offset + r*stride + c, or offset + c*stride + r column-major, counted
// in the tile's components.
func (tl *tileLower) access(o ir.Op, base Reg, offset, stride Reg, store []Reg) []Reg {
	if tl.volta {
		return tl.voltaAccess(o, base, offset, stride, store)
	}
	b := tl.b
	t := *o.Tile
	esz := t.Elem.Bytes()
	col := ir.ColMajor(o)
	sp := space(tl.shared[o.Args[0]])

	// The lane's part of the index, once: offset + majorLane*stride + minorLane.
	rowL, _, colL, _ := tl.element(t, 0, 0, 0)
	majL, minL := rowL, colL
	if col {
		majL, minL = colL, rowL
	}
	lanePart := b.Reg(B32)
	b.Op("mad.lo.u32 %v,%v,%v,%v;", lanePart, majL, stride, offset)
	b.Op("add.u32 %v,%v,%v;", lanePart, lanePart, minL)

	// One address per distinct constant major coordinate; the minor constant
	// rides in the instruction's immediate.
	addrs := map[int]Reg{}
	addrOf := func(maj int) Reg {
		if a, ok := addrs[maj]; ok {
			return a
		}
		idx := b.Reg(B32)
		b.Op("mad.lo.u32 %v,%v,%d,%v;", idx, stride, maj, lanePart)
		a := b.Reg(B64)
		b.Op("mad.wide.u32 %v,%v,%d,%v;", a, idx, esz, base)
		addrs[maj] = a
		return a
	}
	at := func(i, j, e int) (Reg, int) {
		_, rc, _, cc := tl.element(t, i, j, e)
		maj, mn := rc, cc
		if col {
			maj, mn = cc, rc
		}
		return addrOf(maj), mn * esz
	}

	var out []Reg
	if store == nil {
		out = tl.fresh(t)
	}
	n := 0
	for i := 0; i < t.Rows/8; i++ {
		for j := 0; j < t.Cols/8; j++ {
			if t.Elem == ir.TileF32 {
				for e := 0; e < 2; e++ {
					a, off := at(i, j, e)
					if store != nil {
						b.Op("st.%s.f32 [%v+%d],%v;", sp, a, off, store[n])
					} else {
						b.Op("ld.%s.f32 %v,[%v+%d];", sp, out[n], a, off)
					}
					n++
				}
				continue
			}
			a0, off0 := at(i, j, 0)
			a1, off1 := at(i, j, 1)
			if store == nil && ir.Aligned(o) && a0 == a1 && off1 == off0+2 && t.Elem == ir.TileF16 {
				b.Op("ld.%s.b32 %v,[%v+%d];", sp, out[n], a0, off0)
				n++
				continue
			}
			h0, h1 := b.Reg(B16), b.Reg(B16)
			if store != nil {
				b.Op("mov.b32 {%v,%v},%v;", h0, h1, store[n])
				b.Op("st.%s.b16 [%v+%d],%v;", sp, a0, off0, h0)
				b.Op("st.%s.b16 [%v+%d],%v;", sp, a1, off1, h1)
			} else {
				b.Op("ld.%s.b16 %v,[%v+%d];", sp, h0, a0, off0)
				b.Op("ld.%s.b16 %v,[%v+%d];", sp, h1, a1, off1)
				b.Op("mov.b32 %v,{%v,%v};", out[n], h0, h1)
			}
			n++
		}
	}
	return out
}

// voltaElem is a Volta tile's register n: the constant parts of its elements'
// row and column (two elements a register of an f16 tile, the column's next
// for the second), and the predicate a lane holds it under, nil for every lane.
type voltaElem struct {
	row, col int
	pred     *Reg
}

// voltaElems lists a Volta tile's registers in the order count allocates them.
func (tl *tileLower) voltaElems(t ir.TileType) []voltaElem {
	var out []voltaElem
	if tl.pairs {
		switch t.Use {
		case ir.TileA:
			for i := 0; i < t.Rows/16; i++ {
				for s := 0; s < t.Cols/4; s++ {
					for h := 0; h < 2; h++ {
						out = append(out, voltaElem{16 * i, 4*s + 2*h, nil})
					}
				}
			}
		case ir.TileB:
			for s := 0; s < t.Rows/4; s++ {
				for j := 0; j < t.Cols/16; j++ {
					for h := 0; h < 2; h++ {
						out = append(out, voltaElem{4*s + 2*h, 16 * j, nil})
					}
				}
			}
		default:
			for i := 0; i < t.Rows/16; i++ {
				for j := 0; j < t.Cols/16; j++ {
					for x := 0; x < 8; x++ {
						out = append(out, voltaElem{16*i + 2*((x>>1)&1), 16*j + (x & 1) + 4*(x>>2), nil})
					}
				}
			}
		}
		return out
	}
	part := func(k, s int) *Reg {
		if 16*s+16 > k {
			return &tl.q2
		}
		return nil
	}
	switch t.Use {
	case ir.TileA:
		for i := 0; i < t.Rows/8; i++ {
			for s := 0; s < slots(t.Cols); s++ {
				for h := 0; h < 2; h++ {
					out = append(out, voltaElem{8 * i, 16*s + 2*h, part(t.Cols, s)})
				}
			}
		}
	case ir.TileB:
		for s := 0; s < slots(t.Rows); s++ {
			for j := 0; j < t.Cols/8; j++ {
				for h := 0; h < 2; h++ {
					out = append(out, voltaElem{16*s + 2*h, 8 * j, part(t.Rows, s)})
				}
			}
		}
	default:
		for i := 0; i < t.Rows/8; i++ {
			for j := 0; j < t.Cols/8; j++ {
				for x := 0; x < 8; x++ {
					out = append(out, voltaElem{8*i + 2*((x>>1)&1), 8*j + (x & 1) + 4*(x>>2), &tl.qIs[x>>1]})
				}
			}
		}
	}
	return out
}

// voltaAccess is access in the quad-pair layout. A register some lanes do not
// hold is loaded under its predicate (zero elsewhere) and stored under it; an
// accumulator is summed over the four pairs before it is stored.
func (tl *tileLower) voltaAccess(o ir.Op, base Reg, offset, stride Reg, store []Reg) []Reg {
	b := tl.b
	t := *o.Tile
	esz := t.Elem.Bytes()
	col := ir.ColMajor(o)
	sp := space(tl.shared[o.Args[0]])

	// An f16 element's second half is the next k: A's next column, B's next row.
	rowL, colL := tl.dr, tl.dc
	switch {
	case tl.pairs && t.Use == ir.TileA:
		rowL, colL = tl.pA, tl.zero
	case tl.pairs && t.Use == ir.TileB:
		rowL, colL = tl.zero, tl.pB
	case tl.pairs:
		rowL, colL = tl.aR, tl.aC
	case t.Use == ir.TileA:
		rowL, colL = tl.p, tl.q4
	case t.Use == ir.TileB:
		rowL, colL = tl.q4, tl.p
	}
	majL, minL := rowL, colL
	if col {
		majL, minL = colL, rowL
	}
	lanePart := b.Reg(B32)
	b.Op("mad.lo.u32 %v,%v,%v,%v;", lanePart, majL, stride, offset)
	b.Op("add.u32 %v,%v,%v;", lanePart, lanePart, minL)
	addrs := map[int]Reg{}
	at := func(r, c int) (Reg, int) {
		maj, mn := r, c
		if col {
			maj, mn = c, r
		}
		a, ok := addrs[maj]
		if !ok {
			idx := b.Reg(B32)
			b.Op("mad.lo.u32 %v,%v,%d,%v;", idx, stride, maj, lanePart)
			a = b.Reg(B64)
			b.Op("mad.wide.u32 %v,%v,%d,%v;", a, idx, esz, base)
			addrs[maj] = a
		}
		return a, mn * esz
	}
	guard := func(p *Reg) string {
		if p == nil {
			return ""
		}
		return "@" + p.String() + " "
	}

	if store != nil && t.Use == ir.TileAcc && !tl.pairs {
		sum := make([]Reg, len(store))
		for n, r := range store {
			s, x := r, b.Reg(F32)
			for _, m := range []int{4, 8} {
				b.Op("shfl.sync.bfly.b32 %v,%v,%d,31,-1;", x, s, m)
				y := b.Reg(F32)
				b.Op("add.rn.f32 %v,%v,%v;", y, s, x)
				s = y
			}
			sum[n] = s
		}
		store = sum
	}
	var out []Reg
	if store == nil {
		out = tl.fresh(t)
	}
	// A run of k is contiguous where k is the view's minor axis: A row-major,
	// B column-major. A lane's two registers of one block and four of k are
	// then four consecutive halves, read in one access of an aligned view.
	vec := store == nil && ir.Aligned(o) && t.Elem == ir.TileF16 && (t.Use == ir.TileA) != col
	els := tl.voltaElems(t)
	// In the quad-pair layout a lane's run of four k and the next four are
	// eight consecutive halves: over a view aligned to eight, one access.
	wide := map[int]int{}
	if vec && tl.pairs && ir.Aligned8(o) {
		key := func(el voltaElem) [2]int {
			if col {
				return [2]int{el.col, el.row}
			}
			return [2]int{el.row, el.col}
		}
		of := map[[2]int]int{}
		for n := 0; n+1 < len(els); n += 2 {
			of[key(els[n])] = n
		}
		for n := 0; n+1 < len(els); n += 2 {
			if k := key(els[n]); k[1]%8 == 0 {
				if m, ok := of[[2]int{k[0], k[1] + 4}]; ok {
					wide[n], wide[m] = m, -1
				}
			}
		}
	}
	for n := 0; n < len(els); n++ {
		el := els[n]
		g := guard(el.pred)
		if m, ok := wide[n]; ok {
			if m >= 0 {
				a, off := at(el.row, el.col)
				b.Op("ld.%s.v4.b32 {%v,%v,%v,%v},[%v+%d];", sp, out[n], out[n+1], out[m], out[m+1], a, off)
			}
			n++
			continue
		}
		if vec && n%2 == 0 && n+1 < len(els) {
			a, off := at(el.row, el.col)
			if el.pred != nil {
				b.Op("mov.b32 %v,0;", out[n])
				b.Op("mov.b32 %v,0;", out[n+1])
			}
			b.Op("%sld.%s.v2.b32 {%v,%v},[%v+%d];", g, sp, out[n], out[n+1], a, off)
			n++
			continue
		}
		if t.Elem == ir.TileF32 {
			a, off := at(el.row, el.col)
			if store != nil {
				b.Op("%sst.%s.f32 [%v+%d],%v;", g, sp, a, off, store[n])
				continue
			}
			if el.pred != nil {
				b.Op("mov.f32 %v,0f00000000;", out[n])
			}
			b.Op("%sld.%s.f32 %v,[%v+%d];", g, sp, out[n], a, off)
			continue
		}
		r2, c2 := el.row, el.col+1
		if t.Use == ir.TileB {
			r2, c2 = el.row+1, el.col
		}
		a0, off0 := at(el.row, el.col)
		a1, off1 := at(r2, c2)
		h0, h1 := b.Reg(B16), b.Reg(B16)
		if store != nil {
			b.Op("mov.b32 {%v,%v},%v;", h0, h1, store[n])
			b.Op("%sst.%s.b16 [%v+%d],%v;", g, sp, a0, off0, h0)
			b.Op("%sst.%s.b16 [%v+%d],%v;", g, sp, a1, off1, h1)
			continue
		}
		if el.pred != nil {
			b.Op("mov.b32 %v,0;", out[n])
		}
		b.Op("%sld.%s.b16 %v,[%v+%d];", g, sp, h0, a0, off0)
		b.Op("%sld.%s.b16 %v,[%v+%d];", g, sp, h1, a1, off1)
		b.Op("%smov.b32 %v,{%v,%v};", g, out[n], h0, h1)
	}
	return out
}

// scale is accumulator rs of t with each element times the f32 word of
// shared array base at offset plus the element's row.
func (tl *tileLower) scale(t ir.TileType, rs []Reg, base, offset Reg) []Reg {
	b := tl.b
	out := tl.fresh(t)
	type rowAt struct {
		l Reg
		c int
	}
	factor := map[rowAt]Reg{}
	mul := func(n int, l Reg, c int) {
		f, ok := factor[rowAt{l, c}]
		if !ok {
			idx := b.Reg(B32)
			b.Op("add.u32 %v,%v,%v;", idx, offset, l)
			a := b.Reg(B64)
			b.Op("mad.wide.u32 %v,%v,4,%v;", a, idx, base)
			f = b.Reg(F32)
			b.Op("ld.shared.f32 %v,[%v+%d];", f, a, 4*c)
			factor[rowAt{l, c}] = f
		}
		b.Op("mul.rn.f32 %v,%v,%v;", out[n], rs[n], f)
	}
	if tl.volta {
		l := tl.dr
		if tl.pairs {
			l = tl.aR
		}
		for n, el := range tl.voltaElems(t) {
			mul(n, l, el.row)
		}
		return out
	}
	n := 0
	for i := 0; i < t.Rows/8; i++ {
		for j := 0; j < t.Cols/8; j++ {
			for e := 0; e < 2; e++ {
				l, c, _, _ := tl.element(t, i, j, e)
				mul(n, l, c)
				n++
			}
		}
	}
	return out
}

// splat is a tile every element of which is the f32 value v.
func (tl *tileLower) splat(t ir.TileType, v Reg) []Reg {
	b := tl.b
	if tl.volta {
		out := tl.fresh(t)
		h := b.Reg(B16)
		w := b.Reg(B32)
		if t.Elem != ir.TileF32 {
			b.Op("cvt.rn.f16.f32 %v,%v;", h, v)
			b.Op("mov.b32 %v,{%v,%v};", w, h, h)
		}
		for n, el := range tl.voltaElems(t) {
			switch {
			case t.Elem == ir.TileF32 && el.pred == nil:
				b.Op("mov.f32 %v,%v;", out[n], v)
			case t.Elem == ir.TileF32:
				b.Op("selp.f32 %v,%v,0f00000000,%v;", out[n], v, *el.pred)
			case el.pred != nil:
				b.Op("selp.b32 %v,%v,0,%v;", out[n], w, *el.pred)
			default:
				b.Op("mov.b32 %v,%v;", out[n], w)
			}
		}
		return out
	}
	out := tl.fresh(t)
	if t.Elem == ir.TileF32 {
		for _, r := range out {
			b.Op("mov.f32 %v,%v;", r, v)
		}
		return out
	}
	h := b.Reg(B16)
	b.Op("cvt.rn.f16.f32 %v,%v;", h, v)
	for _, r := range out {
		b.Op("mov.b32 %v,{%v,%v};", r, h, h)
	}
	return out
}

// mma is D = A*B + C over atoms. A is M x K, B K x N, C M x N.
func (tl *tileLower) mma(ta, tb, tc ir.TileType, a, bb, c []Reg) []Reg {
	if ta.Elem == ir.TileF32 {
		return tl.mma32(ta, tb, a, bb, c)
	}
	if tl.volta {
		return tl.voltaMMA(ta, tb, a, bb, c)
	}
	b := tl.b
	mA, kA, nB := ta.Rows/8, ta.Cols/8, tb.Cols/8
	aAt := func(i, k int) Reg {
		if i >= mA {
			return tl.zeroH
		}
		return a[i*kA+k]
	}
	bAt := func(k, j int) Reg { return bb[k*nB+j] }
	d := make([]Reg, len(c))
	copy(d, c)
	for i := 0; i < mA; i += 2 {
		for j := 0; j < nB; j++ {
			// The running accumulator: atom (i, j)'s pair and (i+1, j)'s, or
			// zeros where an 8-row tile has no second row of atoms.
			acc := [4]Reg{d[(i*nB+j)*2], d[(i*nB+j)*2+1], tl.zeroF, tl.zeroF}
			if i+1 < mA {
				acc[2], acc[3] = d[((i+1)*nB+j)*2], d[((i+1)*nB+j)*2+1]
			}
			for k := 0; k < kA; {
				out := [4]Reg{b.Reg(F32), b.Reg(F32), b.Reg(F32), b.Reg(F32)}
				if k+1 < kA && tl.sm >= 80 {
					b.Op("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%v,%v,%v,%v},{%v,%v,%v,%v},{%v,%v},{%v,%v,%v,%v};",
						out[0], out[1], out[2], out[3],
						aAt(i, k), aAt(i+1, k), aAt(i, k+1), aAt(i+1, k+1),
						bAt(k, j), bAt(k+1, j),
						acc[0], acc[1], acc[2], acc[3])
					k += 2
				} else {
					b.Op("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%v,%v,%v,%v},{%v,%v},{%v},{%v,%v,%v,%v};",
						out[0], out[1], out[2], out[3],
						aAt(i, k), aAt(i+1, k), bAt(k, j),
						acc[0], acc[1], acc[2], acc[3])
					k++
				}
				acc = out
			}
			d[(i*nB+j)*2], d[(i*nB+j)*2+1] = acc[0], acc[1]
			if i+1 < mA {
				d[((i+1)*nB+j)*2], d[((i+1)*nB+j)*2+1] = acc[2], acc[3]
			}
		}
	}
	return d
}

// voltaMMA is mma in the quad-pair layout: an m8n8k4 an atom of D and 16 of
// k, each pair adding its own chunk's products into its partial sums.
func (tl *tileLower) voltaMMA(ta, tb ir.TileType, a, bb, c []Reg) []Reg {
	b := tl.b
	if tl.pairs {
		mi, nj, ks := ta.Rows/16, tb.Cols/16, ta.Cols/4
		d := make([]Reg, len(c))
		copy(d, c)
		for i := 0; i < mi; i++ {
			for j := 0; j < nj; j++ {
				acc := d[(i*nj+j)*8 : (i*nj+j)*8+8]
				for s := 0; s < ks; s++ {
					out := make([]Reg, 8)
					for x := range out {
						out[x] = b.Reg(F32)
					}
					ai, bi := (i*ks+s)*2, (s*nj+j)*2
					b.Op("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 %s,{%v,%v},{%v,%v},%s;",
						regList(out), a[ai], a[ai+1], bb[bi], bb[bi+1], regList(acc))
					acc = out
				}
				copy(d[(i*nj+j)*8:], acc)
			}
		}
		return d
	}
	mA, nB, ks := ta.Rows/8, tb.Cols/8, slots(ta.Cols)
	d := make([]Reg, len(c))
	copy(d, c)
	for i := 0; i < mA; i++ {
		for j := 0; j < nB; j++ {
			acc := d[(i*nB+j)*8 : (i*nB+j)*8+8]
			for s := 0; s < ks; s++ {
				out := make([]Reg, 8)
				for x := range out {
					out[x] = b.Reg(F32)
				}
				ai, bi := (i*ks+s)*2, (s*nB+j)*2
				b.Op("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 %s,{%v,%v},{%v,%v},%s;",
					regList(out), a[ai], a[ai+1], bb[bi], bb[bi+1], regList(acc))
				acc = out
			}
			copy(d[(i*nB+j)*8:], acc)
		}
	}
	return d
}

// regList is registers as an operand vector, {r0,r1,...}.
func regList(rs []Reg) string {
	var s strings.Builder
	s.WriteByte('{')
	for i, r := range rs {
		if i > 0 {
			s.WriteByte(',')
		}
		s.WriteString(r.String())
	}
	s.WriteByte('}')
	return s.String()
}

// mma32 is mma over float32 operands: each product is three tf32 ones
// (a_hi*b_lo + a_lo*b_hi + a_hi*b_hi, the small terms first), with x_hi the
// tf32 rounding of x and x_lo that of the remainder. That carries about 21 of
// float32's 24 mantissa bits through the matrix unit, where one tf32 product
// carries 11; Metal's float simdgroup_matrix is the arm this stands in for.
func (tl *tileLower) mma32(ta, tb ir.TileType, a, bb, c []Reg) []Reg {
	b := tl.b
	mA, kA, nB := ta.Rows/8, ta.Cols/8, tb.Cols/8
	split := map[Reg][2]Reg{}
	parts := func(x Reg) [2]Reg {
		if p, ok := split[x]; ok {
			return p
		}
		hi, lo := b.Reg(B32), b.Reg(B32)
		hf, rem := b.Reg(F32), b.Reg(F32)
		b.Op("cvt.rna.tf32.f32 %v,%v;", hi, x)
		b.Op("mov.b32 %v,%v;", hf, hi)
		b.Op("sub.f32 %v,%v,%v;", rem, x, hf)
		b.Op("cvt.rna.tf32.f32 %v,%v;", lo, rem)
		split[x] = [2]Reg{hi, lo}
		return split[x]
	}
	zero := [2]Reg{tl.zeroH, tl.zeroH}
	aAt := func(i, k, e, h int) Reg {
		if i >= mA {
			return zero[h]
		}
		return parts(a[(i*kA+k)*2+e])[h]
	}
	bAt := func(k, j, e, h int) Reg { return parts(bb[(k*nB+j)*2+e])[h] }
	d := make([]Reg, len(c))
	copy(d, c)
	for i := 0; i < mA; i += 2 {
		for j := 0; j < nB; j++ {
			acc := [4]Reg{d[(i*nB+j)*2], d[(i*nB+j)*2+1], tl.zeroF, tl.zeroF}
			if i+1 < mA {
				acc[2], acc[3] = d[((i+1)*nB+j)*2], d[((i+1)*nB+j)*2+1]
			}
			for k := 0; k < kA; k++ {
				for _, hl := range [][2]int{{0, 1}, {1, 0}, {0, 0}} {
					ha, hb := hl[0], hl[1]
					out := [4]Reg{b.Reg(F32), b.Reg(F32), b.Reg(F32), b.Reg(F32)}
					b.Op("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%v,%v,%v,%v},{%v,%v,%v,%v},{%v,%v},{%v,%v,%v,%v};",
						out[0], out[1], out[2], out[3],
						aAt(i, k, 0, ha), aAt(i+1, k, 0, ha), aAt(i, k, 1, ha), aAt(i+1, k, 1, ha),
						bAt(k, j, 0, hb), bAt(k, j, 1, hb),
						acc[0], acc[1], acc[2], acc[3])
					acc = out
				}
			}
			d[(i*nB+j)*2], d[(i*nB+j)*2+1] = acc[0], acc[1]
			if i+1 < mA {
				d[((i+1)*nB+j)*2], d[((i+1)*nB+j)*2+1] = acc[2], acc[3]
			}
		}
	}
	return d
}
