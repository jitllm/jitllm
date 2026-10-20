package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// GatherPacked is rows of a packed matrix (w, NRows rows of K) as floats, a
// thread a (token, sub-block): token t's row is the id at ids, each element
// scale*q - bias times emb, stored at out[t*K + element] -- zero for an id
// past the matrix (a padding row). Group: Rows tokens by K/Sub
// sub-blocks.
func (g *Row) GatherPacked(w MatVecWeight, ids, out int, emb float32) error {
	info := w.T.Info()
	switch {
	case info.Float > 0 || info.Codes != nil || kernels.DeviceWhyNot(w.T) != "":
		return fmt.Errorf("compose: a gather of %v rows", w.T)
	case w.K%info.BlockE != 0 || info.Bits != 2 && info.Bits != 4 && info.Bits != 8:
		return fmt.Errorf("compose: a gather of %v at K=%d", w.T, w.K)
	}
	u, ub := g.UConst, g.UBin
	pw := info.Sub * info.Bits / 32
	words := pw + info.Sub*info.Hi/32
	si, tok := g.Column(), g.RowIdx()
	id := g.LoadAt(ids, tok)
	valid := g.Less(id, u(w.Rows))
	r := ub(OpMin, id, u(w.Rows-1))
	super := ub(OpDiv, si, u(info.PerSuper))
	slots := kernels.DSlots(w.T)
	var dw int
	if slots > 1 {
		dw = ub(OpShr, g.LoadAt(w.D, ub(OpAdd, ub(OpMul, super, u((w.Rows+slots-1)/slots)), ub(OpDiv, r, u(slots)))),
			ub(OpMul, ub(OpRem, r, u(slots)), u(32/slots)))
	} else {
		dw = g.LoadAt(w.D, ub(OpAdd, ub(OpMul, super, u(w.Rows)), r))
	}
	float := func(n int) int { return g.Add(Node{Kind: NUnary, Op: OpFloat, A: n}) }
	half := func(n int) int { return g.Add(Node{Kind: NUnary, Op: OpHalf, A: n}) }
	var d int
	switch {
	case info.E8M0:
		d = g.Add(Node{Kind: NUnary, Op: OpBits, A: ub(OpShl, ub(OpAnd, dw, u(0xFF)), u(23))})
	case info.D32:
		d = g.Add(Node{Kind: NUnary, Op: OpBits, A: dw})
	default:
		d = half(dw)
	}
	sc, bias := d, -1
	scAt := func(wi int) int { return g.LoadAt(w.SC, ub(OpAdd, ub(OpMul, wi, u(w.Rows)), r)) }
	switch {
	case kernels.ScStream(w.T):
		pos := ub(OpMul, si, u(12))
		lo, bit := ub(OpShr, pos, u(5)), ub(OpAnd, pos, u(31))
		hi := ub(OpAdd, lo, g.Select(g.Less(u(20), bit), u(1), u(0)))
		pair := g.Add(Node{Kind: NUBinary, Op: OpFunnel, A: scAt(lo), B: scAt(hi), C: bit})
		sc = g.Bin(OpMul, d, g.Bin(OpAdd, float(ub(OpAnd, pair, u(63))), g.Const(float32(info.ScOff))))
		bias = g.Bin(OpMul, half(ub(OpShr, dw, u(16))), float(ub(OpAnd, ub(OpShr, pair, u(6)), u(63))))
	case info.PerSuper > 1:
		per, stride := 4, 8
		if info.BiasArray {
			per, stride = 2, 16
		}
		sw := scAt(ub(OpShr, si, u(log2(per))))
		sh := ub(OpMul, ub(OpAnd, si, u(per-1)), u(stride))
		sc = g.Bin(OpMul, d, g.Bin(OpAdd, float(ub(OpAnd, ub(OpShr, sw, sh), u(0xFF))), g.Const(float32(info.ScOff))))
		if info.BiasArray {
			bias = g.Bin(OpMul, half(ub(OpShr, dw, u(16))), float(ub(OpAnd, ub(OpShr, sw, ub(OpAdd, sh, u(8))), u(0xFF))))
		}
	case info.BiasArray:
		bias = half(ub(OpShr, dw, u(16)))
	}
	if !info.BiasArray {
		bias = g.Bin(OpMul, sc, g.Const(info.BiasK))
	}
	base := ub(OpAdd, ub(OpMul, si, u(words*w.Rows)), r)
	e0 := ub(OpAdd, ub(OpMul, tok, u(w.K)), ub(OpMul, si, u(info.Sub)))
	word := func(k int) int { return g.LoadAt(w.QS, ub(OpAdd, base, u(k*w.Rows))) }
	put := func(l, q int) {
		v := g.Bin(OpSub, g.Bin(OpMul, sc, q), bias)
		if emb != 1 {
			v = g.Bin(OpMul, v, g.Const(emb))
		}
		g.StoreAt(out, g.Select(valid, v, g.Const(0)), ub(OpAdd, e0, u(l)))
	}
	switch info.Bits {
	case 8:
		for k := 0; k < pw; k++ {
			v := word(k)
			for b := 0; b < 4; b++ {
				by := ub(OpAnd, ub(OpShr, v, u(8*b)), u(0xFF))
				q := float(by)
				if info.SignedQ {
					q = g.Bin(OpSub, q, g.Bin(OpMul, g.Const(256), float(ub(OpShr, by, u(7)))))
				}
				put(k*4+b, q)
			}
		}
	case 2:
		run := info.Sub / 4
		for k := 0; k < pw; k++ {
			v := word(k)
			for b := 0; b < 4; b++ {
				for f := 0; f < 4; f++ {
					put(k*4+b+f*run, float(ub(OpAnd, ub(OpShr, v, u(8*b+2*f)), u(3))))
				}
			}
		}
	default:
		lanes := 1
		if info.Hi > 0 {
			lanes = 8 / info.Hi
		}
		his := map[int]int{}
		hiOf := func(l int) int {
			gi := l / 4
			wi := pw + gi/lanes
			hv, ok := his[wi]
			if !ok {
				hv = word(wi)
				his[wi] = hv
			}
			return ub(OpAnd, ub(OpShr, hv, u(8*(l%4)+info.Hi*(gi%lanes))), u(1<<info.Hi-1))
		}
		h := info.Sub / 2
		for k := 0; k < pw; k++ {
			v := word(k)
			for b := 0; b < 4; b++ {
				by := ub(OpShr, v, u(8*b))
				l := k*4 + b
				q0, q1 := ub(OpAnd, by, u(0xF)), ub(OpAnd, ub(OpShr, by, u(4)), u(0xF))
				if info.Hi > 0 {
					q0 = ub(OpAdd, q0, ub(OpShl, hiOf(l), u(4)))
					q1 = ub(OpAdd, q1, ub(OpShl, hiOf(l+h), u(4)))
				}
				put(l, float(q0))
				put(l+h, float(q1))
			}
		}
	}
	return nil
}
