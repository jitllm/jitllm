package ptx

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/jit/ir"
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
// The matrix instructions are sm_75 (m16n8k8) and sm_80 (m16n8k16). sm_70's
// only one is the quad-pair m8n8k4 (ir.MMAVolta()), whose fragments are not
// these, so a tile kernel is refused there. Only f16 operands with an f32
// accumulator are lowered: s8 tiles and f32 operands are refused by name.

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
}

// smOf parses "sm_86" as 86, or 0.
func smOf(target string) int {
	n, _ := strconv.Atoi(strings.TrimPrefix(target, "sm_"))
	return n
}

// tileSupported is the lowering's refusal for a tile type, nil if it lowers.
func tileSupported(t ir.TileType, sm int) error {
	if sm < 75 {
		return fmt.Errorf("ptx: no collective tile below sm_75 (this is sm_%d): mma.sync's "+
			"m16n8 fragments start there, and sm_70's m8n8k4 is the quad-pair layout ir.OpMMA carries", sm)
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
}

// count is how many registers a lane holds of a tile.
func count(t ir.TileType) int {
	n := (t.Rows / 8) * (t.Cols / 8)
	if t.Elem == ir.TileF32 {
		return 2 * n
	}
	return n
}

// fresh allocates a tile's registers.
func (tl *tileLower) fresh(t ir.TileType) []Reg {
	rs := make([]Reg, count(t))
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

// splat is a tile every element of which is the f32 value v.
func (tl *tileLower) splat(t ir.TileType, v Reg) []Reg {
	b := tl.b
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
