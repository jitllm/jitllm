package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// convTaps is the most taps one map of a convolution sums: each tap holds
// its offsets' bounds and its weight's vector, and a square's row of five
// over a vector loop holds more than the registers.
const convTaps = 3

// decomposeConv is a depthwise convolution over the call's rows
// (graph.OpConv) as a grid GridW wide, a square of taps at SAME padding,
// at a stride where Stride is set (the output a grid of its own rows,
// padded as TensorFlow pads, the smaller half first). Each output element
// is its channel's taps times the rows they reach, a row past the grid
// zero, summed in tap order a few taps of a row of the square a map
// (convTaps), each map adding its taps to the one before's sums; a tap
// that lands on the grid from no output row is no term at all. The taps
// are the weight's rows (tap-major), decoded from its format. A line of
// rows (no GridW) is refused: no gate holds it.
func decomposeConv(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 {
		return nil, fmt.Errorf("a convolution of %d operands", len(o.In))
	}
	U, F := ir.U32, ir.F32
	sh, osh := c.rows(o.In[0]), c.rows(o.Out[0])
	R, C, T, s := sh[0], sh[1], o.Attr.Taps, max(1, o.Attr.Stride)
	if o.Attr.GridW < 1 {
		return nil, fmt.Errorf("a convolution over a line of rows: only a grid's is formed")
	}
	W := o.Attr.GridW
	H, side := R/W, T*T
	Ho, Wo := (H+s-1)/s, (W+s-1)/s
	h := T / 2
	// The padding before the grid's first row and column: h at stride one.
	padY, padX := max((Ho-1)*s+T-H, 0)/2, max((Wo-1)*s+T-W, 0)/2
	if p := o.Attr.Offset; p > 0 {
		padY, padX = p, p
	}
	switch {
	case T < 1 || T%2 == 0:
		return nil, fmt.Errorf("a convolution of %d taps: a centred one is odd", T)
	case o.Attr.Offset > T/2:
		return nil, fmt.Errorf("a convolution of %d taps padded by %d: the output grid would grow", T, o.Attr.Offset)
	case H*W != R:
		return nil, fmt.Errorf("a convolution over %d rows, not a whole grid %d wide", R, W)
	case osh[0] != Ho*Wo || osh[1] != C:
		return nil, fmt.Errorf("a convolution of a %dx%d grid at stride %d into %d rows of %d", H, W, s, osh[0], osh[1])
	case o.Weight[0].K != C || o.Weight[0].Rows != side:
		return nil, fmt.Errorf("a convolution's %dx%d taps over %d channels of %d taps", o.Weight[0].Rows, o.Weight[0].K, C, side)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	out := c.value(o.Out[0])
	// A tap's offsets from the output coordinate times the stride.
	type tap struct{ at, dy, dx int }
	// lands reports whether offset d from some output coordinate of no
	// (times the stride) falls inside the side n.
	lands := func(d, n, no int) bool {
		for p := range no {
			if q := p*s + d; q >= 0 && q < n {
				return true
			}
		}
		return false
	}
	var taps []tap
	for ky := range T {
		dy := ky - padY
		for kx := range T {
			if dx := kx - padX; lands(dy, H, Ho) && lands(dx, W, Wo) {
				taps = append(taps, tap{ky*T + kx, dy, dx})
			}
		}
	}
	// reach is a coordinate plus a tap's offset d (at least -h), clamped
	// into the side n (one past an edge wraps above it, unsigned, and
	// clamps to the last), and whether it lies inside; ok is -1 where it
	// always does.
	reach := func(x *exprs, at, d, n int) (v, ok int) {
		if d == 0 {
			return at, -1
		}
		p := x.op(OpSub, U, x.op(OpAdd, U, at, x.u(uint32(d+h))), x.u(uint32(h)))
		return x.op(OpMin, U, p, x.u(uint32(n-1))), x.op(OpLess, ir.Pred, p, x.u(uint32(n)))
	}
	// The maps' taps: a row of the square at most convTaps at a time, so a
	// map tests one row's bounds.
	var groups [][]tap
	for i, t := range taps {
		if g := len(groups) - 1; i == 0 || len(groups[g]) == convTaps || groups[g][0].dy != t.dy {
			groups = append(groups, nil)
		}
		groups[len(groups)-1] = append(groups[len(groups)-1], t)
	}
	var prims []Prim
	prev := Operand{}
	for g, group := range groups {
		// Row r of the output is its grid's row y = r/Wo (a division by a
		// constant), column r - y*Wo; a side of one is no coordinate at
		// all. The input's are those times the stride.
		var x exprs
		r, ch := x.coord(0), x.coord(1)
		y, xc := r, r
		switch {
		case Ho > 1 && Wo > 1:
			y = x.op(OpDiv, U, r, x.u(uint32(Wo)))
			xc = x.op(OpSub, U, r, x.op(OpMul, U, y, x.u(uint32(Wo))))
		case Ho == 1 && Wo == 1:
			y, xc = x.u(0), x.u(0)
		case Wo == 1:
			xc = x.u(0)
		case Ho == 1:
			y = x.u(0)
		}
		if s > 1 {
			y, xc = x.op(OpMul, U, y, x.u(uint32(s))), x.op(OpMul, U, xc, x.u(uint32(s)))
		}
		in := []Operand{c.value(o.In[0])}
		facts := Facts{Extents: map[int]int{0: R * C}}
		acc := -1
		if g > 0 {
			in = append(in, prev)
			facts.Extents[1] = Ho * Wo * C
			acc = x.at(1, osh, r, ch)
		}
		for _, t := range group {
			w, err := weightElement(&x, wo, side, C, &in, &facts, x.u(uint32(t.at)), ch)
			if err != nil {
				return nil, err
			}
			row, oks := r, [2]int{-1, -1}
			switch {
			case W == 1:
				row, oks[0] = reach(&x, y, t.dy, H)
			case H == 1:
				row, oks[1] = reach(&x, xc, t.dx, W)
			case s == 1 && t.dy == 0 && t.dx == 0:
			default:
				var yy, xx int
				yy, oks[0] = reach(&x, y, t.dy, H)
				xx, oks[1] = reach(&x, xc, t.dx, W)
				row = x.op(OpAdd, U, x.op(OpMul, U, yy, x.u(uint32(W))), xx)
			}
			// Read by its index, the centre too: a map over the input reads
			// rows other trips write, so its output never takes the input's
			// storage (InPlace takes only an input read at its own element).
			term := x.op(OpMul, F, x.load(0, x.op(OpAdd, U, x.op(OpMul, U, row, x.u(uint32(C))), ch)), w)
			for _, ok := range oks {
				if ok >= 0 {
					term = x.sel(F, ok, term, x.f(0))
				}
			}
			if acc < 0 {
				acc = term
				continue
			}
			acc = x.op(OpAdd, F, acc, term)
		}
		to := out
		if g < len(groups)-1 {
			to = Operand{Kind: OperandScratch, Name: fmt.Sprintf("%s.taps%d", out.Name, g), Elem: F, Temp: true}
		}
		prims = append(prims, Prim{Kind: PrimMap, Shape: osh, In: in, Out: to, Expr: x.e, Facts: facts})
		prev = to
	}
	return prims, nil
}

// decomposeUnfold is each output row of the input's grid as the window of
// rows around it (graph.OpUnfold), one map over the output's rows and
// width: element j of row r is channel j%C of tap j/C's row, the row that
// tap reaches from r's grid position times the stride, less the padding
// before the grid (TensorFlow's SAME, as decomposeConv pads); zero where
// that row lies past the grid or j past the taps; an Offset is PyTorch's
// padding, as decomposeConv takes it. An offset below zero wraps above the
// side, unsigned, so one comparison tests both edges.
func decomposeUnfold(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 0 {
		return nil, fmt.Errorf("an unfold of %d operands", len(o.In))
	}
	U := ir.U32
	sh, osh := c.rows(o.In[0]), c.rows(o.Out[0])
	R, C, T, s, W := sh[0], sh[1], o.Attr.Taps, max(1, o.Attr.Stride), o.Attr.GridW
	if W < 1 {
		return nil, fmt.Errorf("an unfold over a line of rows: only a grid's is formed")
	}
	H := R / W
	Ho, Wo := (H+s-1)/s, (W+s-1)/s
	padY, padX := max((Ho-1)*s+T-H, 0)/2, max((Wo-1)*s+T-W, 0)/2
	if p := o.Attr.Offset; p > 0 {
		padY, padX = p, p
	}
	switch {
	case o.Attr.Offset > T/2:
		return nil, fmt.Errorf("an unfold of %d taps padded by %d: the output grid would grow", T, o.Attr.Offset)
	case T < 1 || T%2 == 0:
		return nil, fmt.Errorf("an unfold of %d taps: a centred one is odd", T)
	case H*W != R:
		return nil, fmt.Errorf("an unfold over %d rows, not a whole grid %d wide", R, W)
	case osh[0] != Ho*Wo || osh[1] < T*T*C:
		return nil, fmt.Errorf("an unfold of a %dx%d grid by %dx%d at stride %d into %d rows of %d", H, W, T, T, s, osh[0], osh[1])
	}
	var x exprs
	r, j := x.coord(0), x.coord(1)
	y, xc := r, r
	switch {
	case Ho > 1 && Wo > 1:
		y = x.op(OpDiv, U, r, x.u(uint32(Wo)))
		xc = x.op(OpSub, U, r, x.op(OpMul, U, y, x.u(uint32(Wo))))
	case Ho == 1 && Wo == 1:
		y, xc = x.u(0), x.u(0)
	case Wo == 1:
		xc = x.u(0)
	case Ho == 1:
		y = x.u(0)
	}
	tap, ch := j, x.u(0)
	if C > 1 {
		tap = x.op(OpDiv, U, j, x.u(uint32(C)))
		ch = x.op(OpSub, U, j, x.op(OpMul, U, tap, x.u(uint32(C))))
	}
	ky := x.op(OpDiv, U, tap, x.u(uint32(T)))
	kx := x.op(OpSub, U, tap, x.op(OpMul, U, ky, x.u(uint32(T))))
	// at is a coordinate times the stride plus a tap less the padding.
	at := func(p, k, pad int) int {
		if s > 1 {
			p = x.op(OpMul, U, p, x.u(uint32(s)))
		}
		p = x.op(OpAdd, U, p, k)
		if pad > 0 {
			p = x.op(OpSub, U, p, x.u(uint32(pad)))
		}
		return p
	}
	iy, ix := at(y, ky, padY), at(xc, kx, padX)
	row := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpMin, U, iy, x.u(uint32(H-1))), x.u(uint32(W))), x.op(OpMin, U, ix, x.u(uint32(W-1))))
	v := x.load(0, x.op(OpAdd, U, x.op(OpMul, U, row, x.u(uint32(C))), ch))
	v = x.sel(ir.F32, x.op(OpLess, ir.Pred, ix, x.u(uint32(W))), v, x.f(0))
	v = x.sel(ir.F32, x.op(OpLess, ir.Pred, iy, x.u(uint32(H))), v, x.f(0))
	if osh[1] > T*T*C {
		v = x.sel(ir.F32, x.op(OpLess, ir.Pred, j, x.u(uint32(T*T*C))), v, x.f(0))
	}
	return []Prim{{Kind: PrimMap, Shape: osh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{0: R * C}}}}, nil
}
