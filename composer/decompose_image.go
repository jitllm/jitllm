package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// RequestGrid is a picture's step's request word: its grid's width in
// patches.
const RequestGrid = 0

// A picture's own request words (graph.OpPicture): its width and height in
// pixels, and its rows' stride in bytes; a second plane's stride in bytes,
// the byte offsets of its second and third planes in the step's picture
// (YCbCr's chroma, or a palette's entries), and its chroma subsampling: the
// horizontal and vertical shifts from a pixel to its chroma sample, then a
// sub-picture's phase on each axis, a byte each; then the width and height
// it is resampled to, where the op's size is the request's (PicLive).
const (
	RequestPicW = RequestRows + iota
	RequestPicH
	RequestPicStride
	RequestPicCStride
	RequestPicP2
	RequestPicP3
	RequestPicSub
	RequestPicOW
	RequestPicOH
	// RequestPicTaps is where a picture resized by Pillow's filters
	// (graph.ChainPIL) has its taps: the output's columns' (PILTaps), then
	// its rows', each entry Taps+1 words.
	RequestPicTaps
)

// PictureWords is the most words a step's picture holds: a picture 4096
// pixels square, four bytes a pixel.
const PictureWords = 4096 * 4096

// decomposePicture is each output pixel's channel of the step's picture:
// on each axis the two source samples around its centre
// ((i+0.5)*n/size - 0.5, clamped to the picture), weighted 1-t and t,
// horizontal first, then the sample read as [0, 1] and normalised by its
// channel's mean and std; zero past the picture.
func decomposePicture(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	pil := a.Chain == graph.ChainPIL && (a.Filter == graph.FilterBilinear || a.Filter == graph.FilterCubic) && a.Taps > 0
	if len(o.In) != 0 || len(o.Out) != 1 || a.PicW < 1 || a.PicH < 1 ||
		!pil && (a.Chain != graph.ChainHalfPixel || a.Filter != graph.FilterLinear) ||
		a.Pixels < graph.PixRGBA8 || a.Pixels >= graph.PixFormats {
		return nil, fmt.Errorf("a picture of format %d resampled to %dx%d by chain %d and filter %d",
			a.Pixels, a.PicW, a.PicH, a.Chain, a.Filter)
	}
	if pil {
		return decomposePicturePIL(o, c)
	}
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(o.Out[0])
	W := sh[1]
	var x exprs
	u := func(v int) int { return x.u(uint32(v)) }
	flat := x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), u(W)), x.coord(1))
	pix, ch := x.op(OpDiv, U, flat, u(3)), x.op(OpRem, U, flat, u(3))
	ow, oh := u(a.PicW), u(a.PicH)
	if a.PicLive {
		ow = x.op(OpMin, U, x.op(OpMax, U, x.load(1, u(RequestPicOW)), u(1)), u(a.PicW))
		oh = x.op(OpMin, U, x.op(OpMax, U, x.load(1, u(RequestPicOH)), u(1)), u(a.PicH))
	}
	oy, ox := x.op(OpDiv, U, pix, ow), x.op(OpRem, U, pix, ow)
	sw := x.op(OpMax, U, x.load(1, u(RequestPicW)), u(1))
	sh2 := x.op(OpMax, U, x.load(1, u(RequestPicH)), u(1))
	stride := x.load(1, u(RequestPicStride))
	// axis is cell's two taps over a source n samples wide into size: the
	// floor of its centre in integers ((2cell+1)n + size) / 2size - 1, its
	// fraction in floats, both indices clamped to the source.
	axis := func(cell, n, size int) (i0, i1, t int) {
		q := x.op(OpDiv, U, x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, cell, u(2)), u(1)), n), size),
			x.op(OpMul, U, size, u(2)))
		fl := x.op(OpSub, I, x.un(OpBitcast, I, q), x.i(1))
		at := x.op(OpSub, F, x.op(OpDiv, F, x.op(OpMul, F, x.op(OpAdd, F, x.un(OpCvtF32, F, cell), x.f(0.5)), x.un(OpCvtF32, F, n)),
			x.un(OpCvtF32, F, size)), x.f(0.5))
		t = x.op(OpSub, F, at, x.un(OpCvtF32, F, fl))
		last := x.op(OpSub, I, x.un(OpBitcast, I, n), x.i(1))
		clamp := func(v int) int { return x.un(OpBitcast, U, x.op(OpMin, I, x.op(OpMax, I, v, x.i(0)), last)) }
		return clamp(fl), clamp(x.op(OpAdd, I, fl, x.i(1))), t
	}
	y0, y1, ty := axis(oy, sh2, oh)
	x0, x1, tx := axis(ox, sw, ow)
	// sample is the picture's channel for (y, x), as a float.
	sample := func(y, xx int) int {
		return x.un(OpCvtF32, F, pictureSample(&x, a.Pixels, stride, y, xx, ch))
	}
	lerp := func(p, q, t int) int {
		return x.op(OpAdd, F, x.op(OpMul, F, p, x.op(OpSub, F, x.f(1), t)), x.op(OpMul, F, q, t))
	}
	v := lerp(lerp(sample(y0, x0), sample(y0, x1), tx), lerp(sample(y1, x0), sample(y1, x1), tx), ty)
	byChannel := func(m [3]float64) int {
		return x.sel(F, x.op(OpLess, ir.Pred, ch, u(1)), x.f(float32(m[0])),
			x.sel(F, x.op(OpLess, ir.Pred, ch, u(2)), x.f(float32(m[1])), x.f(float32(m[2]))))
	}
	v = x.op(OpDiv, F, x.op(OpSub, F, x.op(OpDiv, F, v, x.f(255)), byChannel(a.Mean)), byChannel(a.Std))
	x.sel(F, x.op(OpLess, ir.Pred, pix, x.op(OpMul, U, ow, oh)), v, x.f(0))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{TablePicture.op(), TableRequest.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{0: PictureWords, 1: RequestPicOH + 1}}}}, nil
}

// pictureSample is channel ch of the picture's pixel (y, xx), the byte its
// format holds or makes (U32): the picture in operand 0, the request in 1,
// stride its rows' bytes.
func pictureSample(x *exprs, px uint8, stride, y, xx, ch int) int {
	U := ir.U32
	u := func(v int) int { return x.u(uint32(v)) }
	byteAt := func(off int) int {
		w := x.load(0, x.op(OpShr, U, off, u(2)))
		return x.op(OpAnd, U, x.op(OpShr, U, w, x.op(OpShl, U, x.op(OpAnd, U, off, u(3)), u(3))), u(255))
	}
	at := x.op(OpAdd, U, x.op(OpMul, U, y, stride), xx)
	switch px {
	case graph.PixRGBA8:
		at = x.op(OpAdd, U, x.op(OpAdd, U, x.op(OpMul, U, y, stride), x.op(OpMul, U, xx, u(4))), ch)
	case graph.PixPaletted8:
		at = x.op(OpAdd, U, x.op(OpAdd, U, x.load(1, u(RequestPicP2)), x.op(OpMul, U, byteAt(at), u(4))), ch)
	case graph.PixYCbCr8:
		return ycbcr(x, byteAt, at, y, xx, ch)
	}
	return byteAt(at)
}

// decomposePicturePIL is a picture resized by Pillow's antialiased filter
// (graph.ChainPIL), its taps in the request (PILTaps): for each source row
// (PicSide of them) and output column's channel the sum over the tap axis
// (Taps long, its weights zero past the column's window) of each weight times
// its sample, in integers; then for each output pixel's channel the same over
// its rows, each term the horizontal sum made a byte -- from half Pillow's
// unit, clamped at zero, shifted down by its fixed point and clamped to 255;
// then that sum made a byte, read as [0, 1] and normalised by its channel's
// mean and std; zero past the picture. A source row or column past the
// picture's reads its last, at a weight of zero.
func decomposePicturePIL(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(o.Out[0])
	W, K := sh[1], a.Taps
	if W%3 != 0 || W >= 1<<15 {
		return nil, fmt.Errorf("a picture's rows of %d samples, not whole pixels", W)
	}
	tapsV := RequestPicTaps + a.PicW*(K+1)
	reqWords := tapsV + a.PicH*(K+1)
	name := c.Graph.Values[o.Out[0]].Name
	hsh := []int{a.PicSide, a.PicW, 3}
	hsum := Operand{Kind: OperandScratch, Name: name + ".rows", Elem: I}
	vsum := Operand{Kind: OperandScratch, Name: name + ".cols", Elem: I}
	// finish is a sum of weights times bytes as Pillow's byte (U32).
	finish := func(x *exprs, acc int) int {
		v := x.un(OpBitcast, U, x.op(OpMax, I, x.op(OpAdd, I, acc, x.i(1<<(pilPrecision-1))), x.i(0)))
		return x.op(OpMin, U, x.op(OpShr, U, v, x.u(pilPrecision)), x.u(255))
	}
	// weight is tap t's weight at entry base of the request's taps.
	weight := func(x *exprs, base, t int) int {
		return x.un(OpBitcast, I, x.load(1, x.op(OpAdd, U, x.op(OpAdd, U, base, x.u(1)), t)))
	}
	facts := Facts{Extents: map[int]int{0: PictureWords, 1: reqWords}}

	// The horizontal pass, a reduction a channel: the channel a constant.
	var prims []Prim
	for ch := range 3 {
		var h exprs
		r, xo, t := h.coord(0), h.coord(1), h.coord(2)
		hu := func(v int) int { return h.u(uint32(v)) }
		sw := h.op(OpMax, U, h.load(1, hu(RequestPicW)), hu(1))
		srcH := h.op(OpMax, U, h.load(1, hu(RequestPicH)), hu(1))
		y := h.op(OpMin, U, r, h.op(OpSub, U, srcH, hu(1)))
		base := h.op(OpAdd, U, hu(RequestPicTaps), h.op(OpMul, U, xo, hu(K+1)))
		xi := h.op(OpMin, U, h.op(OpAdd, U, h.load(1, base), t), h.op(OpSub, U, sw, hu(1)))
		stride := h.load(1, hu(RequestPicStride))
		h.op(OpMul, I, weight(&h, base, t), h.un(OpBitcast, I, pictureSample(&h, a.Pixels, stride, y, xi, hu(ch))))
		var hi exprs
		hi.op(OpAdd, U, hi.op(OpMul, U, hi.op(OpAdd, U, hi.op(OpMul, U, hi.coord(0), hi.u(uint32(a.PicW))), hi.coord(1)), hi.u(3)),
			hi.u(uint32(ch)))
		hf := Facts{Extents: facts.Extents, OutExtent: a.PicSide * a.PicW * 3}
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{a.PicSide, a.PicW, K}, In: []Operand{TablePicture.op(), TableRequest.op()},
			Out: hsum, Combine: OpAdd, Type: I, Expr: h.e, Index: hi.e, Facts: hf})
	}

	var v exprs
	oy, ox, vch, vt := v.coord(0), v.coord(1), v.coord(2), v.coord(3)
	vu := func(n int) int { return v.u(uint32(n)) }
	vsrcH := v.op(OpMax, U, v.load(1, vu(RequestPicH)), vu(1))
	vbase := v.op(OpAdd, U, vu(tapsV), v.op(OpMul, U, oy, vu(K+1)))
	yi := v.op(OpMin, U, v.op(OpAdd, U, v.load(1, vbase), vt), v.op(OpSub, U, vsrcH, vu(1)))
	v.op(OpMul, I, weight(&v, vbase, vt), v.un(OpBitcast, I, finish(&v, v.at(2, hsh, yi, ox, vch))))

	var m exprs
	mu := func(n int) int { return m.u(uint32(n)) }
	n := a.PicW * a.PicH * 3
	flat := m.op(OpAdd, U, m.op(OpMul, U, m.coord(0), mu(W)), m.coord(1))
	col := m.coord(1)
	mch := m.op(OpSub, U, col, m.op(OpMul, U, m.op(OpShr, U, m.op(OpMul, U, col, mu(0xAAAB)), mu(17)), mu(3)))
	byChannel := func(c [3]float64) int {
		return m.sel(F, m.op(OpLess, ir.Pred, mch, mu(1)), m.f(float32(c[0])),
			m.sel(F, m.op(OpLess, ir.Pred, mch, mu(2)), m.f(float32(c[1])), m.f(float32(c[2]))))
	}
	// A live picture is dense in the output and laid at the form's width in
	// the sums: its pixel (y, x) is sum (y, x) of the reductions above.
	at, live := flat, mu(n)
	in := []Operand{vsum}
	ext := map[int]int{0: n}
	if a.PicLive {
		in = append(in, TableRequest.op())
		ext[1] = reqWords
		ow := m.op(OpMin, U, m.op(OpMax, U, m.load(1, mu(RequestPicOW)), mu(1)), mu(a.PicW))
		oh := m.op(OpMin, U, m.op(OpMax, U, m.load(1, mu(RequestPicOH)), mu(1)), mu(a.PicH))
		pix, ch := m.op(OpDiv, U, flat, mu(3)), m.op(OpRem, U, flat, mu(3))
		at = m.op(OpAdd, U, m.op(OpMul, U, m.op(OpAdd, U, m.op(OpMul, U, m.op(OpDiv, U, pix, ow), mu(a.PicW)), m.op(OpRem, U, pix, ow)), mu(3)), ch)
		live = m.op(OpMul, U, m.op(OpMul, U, ow, oh), mu(3))
	}
	b := m.un(OpCvtF32, F, finish(&m, m.load(0, m.op(OpMin, U, at, mu(n-1)))))
	val := m.op(OpDiv, F, m.op(OpSub, F, m.op(OpDiv, F, b, m.f(255)), byChannel(a.Mean)), byChannel(a.Std))
	m.sel(F, m.op(OpLess, ir.Pred, flat, live), val, m.f(0))
	return append(prims,
		Prim{Kind: PrimReduce, Shape: []int{a.PicH, a.PicW, 3, K}, In: []Operand{TablePicture.op(), TableRequest.op(), hsum},
			Out: vsum, Combine: OpAdd, Type: I, Expr: v.e, Facts: facts},
		Prim{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: m.e,
			Facts: Facts{Extents: ext}}), nil
}

// ycbcr is channel ch of the YCbCr pixel (y, xx), its luma at byte at: its
// chroma sample the pixel's coordinates, each plus its axis's phase, shifted
// down by its axis's subsampling (RequestPicSub), in the second and third
// planes; converted as image/color.YCbCrToRGB does, in integers -- luma
// times 0x10101 plus the channel's chroma terms, clamped to 24 bits, the top
// byte.
func ycbcr(x *exprs, byteAt func(int) int, at, y, xx, ch int) int {
	U, I := ir.U32, ir.I32
	u := func(v int) int { return x.u(uint32(v)) }
	s := x.load(1, u(RequestPicSub))
	field := func(k int) int { return x.op(OpAnd, U, x.op(OpShr, U, s, u(8*k)), u(255)) }
	cy := x.op(OpShr, U, x.op(OpAdd, U, y, field(3)), field(1))
	cx := x.op(OpShr, U, x.op(OpAdd, U, xx, field(2)), field(0))
	coff := x.op(OpAdd, U, x.op(OpMul, U, cy, x.load(1, u(RequestPicCStride))), cx)
	yy := x.un(OpBitcast, I, x.op(OpMul, U, byteAt(at), u(0x10101)))
	chroma := func(plane int) int {
		b := byteAt(x.op(OpAdd, U, x.load(1, u(plane)), coff))
		return x.op(OpSub, I, x.un(OpBitcast, I, b), x.i(128))
	}
	cb, cr := chroma(RequestPicP2), chroma(RequestPicP3)
	// The channel's two chroma weights: red 0 and 91881, green -22554 and
	// -46802, blue 116130 and 0 -- each channel YCbCrToRGB's one sum.
	byChannel := func(r, g, b int32) int {
		return x.sel(I, x.op(OpLess, ir.Pred, ch, u(1)), x.i(r), x.sel(I, x.op(OpLess, ir.Pred, ch, u(2)), x.i(g), x.i(b)))
	}
	v := x.op(OpAdd, I, yy, x.op(OpAdd, I, x.op(OpMul, I, byChannel(0, -22554, 116130), cb),
		x.op(OpMul, I, byChannel(91881, -46802, 0), cr)))
	return x.op(OpShr, U, x.un(OpBitcast, U, x.op(OpMin, I, x.op(OpMax, I, v, x.i(0)), x.i(0xffffff))), u(16))
}

// decomposePatchify is each row's patch of the picture, gathered: a row
// past the Offset before it the patch its position less Offset names, of a
// grid RequestGrid patches wide, its element c channel c/(P*P), row (c/P)%P
// and column c%P of the patch; zero past 3*P*P and in the rows before
// Offset.
func decomposePatchify(o graph.Op, c *Context) ([]Prim, error) {
	P := o.Attr.Span
	if len(o.In) != 1 || len(o.Out) != 1 || P < 1 || c.width(o.In[0]) != 3*P*P || c.width(o.Out[0]) < 3*P*P {
		return nil, fmt.Errorf("a patch gather of %d operands, patches of %d", len(o.In), P)
	}
	U, F := ir.U32, ir.F32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	R, PW := sh[0], uint32(3*P*P)
	var x exprs
	row, col := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	lead := o.Attr.Offset
	r := x.op(OpSub, U, x.op(OpMax, U, x.load(2, x.op(OpAdd, U, row, u(1))), u(lead)), u(lead))
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), u(1))
	py, pxi := x.op(OpDiv, U, r, gw), x.op(OpRem, U, r, gw)
	pp := uint32(P * P)
	ch, ky, kx := x.op(OpDiv, U, col, x.u(pp)), x.op(OpRem, U, x.op(OpDiv, U, col, u(P)), u(P)), x.op(OpRem, U, col, u(P))
	y := x.op(OpAdd, U, x.op(OpMul, U, py, u(P)), ky)
	xx := x.op(OpAdd, U, x.op(OpMul, U, pxi, u(P)), kx)
	pix := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, y, x.op(OpMul, U, gw, u(P))), xx), u(3)), ch)
	at := x.op(OpMin, U, x.op(OpDiv, U, pix, x.u(PW)), u(R-1))
	v := x.at(0, ish, at, x.op(OpRem, U, pix, x.u(PW)))
	v = x.sel(F, x.op(OpLess, ir.Pred, col, x.u(PW)), v, x.f(0))
	if lead > 0 {
		x.sel(F, rowBefore(&x, row, lead), x.f(0), v)
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), TableRequest.op(), TablePos.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{1: RequestRows + R, 2: R + 1}}}}, nil
}

// rowBefore is the mask of row r lying before row n.
func rowBefore(x *exprs, r, n int) int {
	return x.op(OpLess, ir.Pred, r, x.u(uint32(n)))
}

// decomposeFold is each Span consecutive rows as one: output row g is
// input rows g*Span to g*Span+Span-1, each a run of the input's width.
func decomposeFold(o graph.Op, c *Context) ([]Prim, error) {
	k := o.Attr.Span
	if len(o.In) != 1 || len(o.Out) != 1 || k < 2 || c.width(o.Out[0]) != k*c.width(o.In[0]) {
		return nil, fmt.Errorf("a fold of %d operands by %d", len(o.In), k)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	E := ish[1]
	var x exprs
	g, j := x.coord(0), x.coord(1)
	row := x.op(OpAdd, U, x.op(OpMul, U, g, x.u(uint32(k))), x.op(OpDiv, U, j, x.u(uint32(E))))
	x.at(0, ish, x.op(OpMin, U, row, x.u(uint32(ish[0]-1))), x.op(OpRem, U, j, x.u(uint32(E))))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeRegroup is space-to-depth over the grid of rows RequestGrid
// wide: output row g is the S x S block of rows at its place, slot (dy, dx)
// the run dy*S+dx of the input's width.
func decomposeRegroup(o graph.Op, c *Context) ([]Prim, error) {
	S := o.Attr.Span
	if len(o.In) != 1 || len(o.Out) != 1 || S < 2 || c.width(o.Out[0]) != S*S*c.width(o.In[0]) {
		return nil, fmt.Errorf("a regroup of %d operands by %d", len(o.In), S)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	R, E := ish[0], ish[1]
	var x exprs
	g, j := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), u(S))
	gsw := x.op(OpDiv, U, gw, u(S))
	slot, e := x.op(OpDiv, U, j, u(E)), x.op(OpRem, U, j, u(E))
	row := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, g, gsw), u(S)), x.op(OpDiv, U, slot, u(S)))
	colm := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpRem, U, g, gsw), u(S)), x.op(OpRem, U, slot, u(S)))
	p := x.op(OpAdd, U, x.op(OpMul, U, row, gw), colm)
	x.at(0, ish, x.op(OpMin, U, p, u(R-1)), e)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), TableRequest.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{1: RequestRows + sh[0]}}}}, nil
}

// decomposeLeadRow is the input with its first Offset rows each the
// weight's row.
func decomposeLeadRow(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 || o.Attr.Offset < 1 || o.Weight[0].K != c.width(o.In[0]) {
		return nil, fmt.Errorf("a lead row of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	sh := c.rows(o.Out[0])
	w := o.Weight[0]
	var x exprs
	r, i := x.coord(0), x.coord(1)
	in, lead := x.at(0, sh, r, i), x.at(1, []int{sh[1]}, i)
	x.sel(ir.F32, rowBefore(&x, r, o.Attr.Offset), lead, in)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), {Kind: OperandVector, Role: w.Role, Name: w.Role.String()}},
		Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeTailRow is the input with its last live row the weight's row:
// row r takes it where r+1 is not below the step's live rows.
func decomposeTailRow(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 || o.Weight[0].K != c.width(o.In[0]) {
		return nil, fmt.Errorf("a tail row of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	U := ir.U32
	sh := c.rows(o.Out[0])
	w := o.Weight[0]
	var x exprs
	r, i := x.coord(0), x.coord(1)
	in, tail := x.at(0, sh, r, i), x.at(1, []int{sh[1]}, i)
	live := x.load(2, x.u(0))
	x.sel(ir.F32, x.op(OpLess, ir.Pred, x.op(OpAdd, U, r, x.u(1)), live), in, tail)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), {Kind: OperandVector, Role: w.Role, Name: w.Role.String()}, TablePos.op()},
		Out: c.value(o.Out[0]), Expr: x.e, Facts: Facts{Extents: map[int]int{2: sh[0] + 1}}}}, nil
}

// decomposeDropRows is the input from row Offset on.
func decomposeDropRows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || o.Attr.Offset < 1 || c.width(o.In[0]) != c.width(o.Out[0]) {
		return nil, fmt.Errorf("a row drop of %d operands", len(o.In))
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	var x exprs
	at := x.op(OpMin, U, x.op(OpAdd, U, x.coord(0), x.u(uint32(o.Attr.Offset))), x.u(uint32(ish[0]-1)))
	x.at(0, ish, at, x.coord(1))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposePosResample is each row's position-table row resampled to the
// grid: the row's cell, on each axis the source coordinate by the op's
// chain and its filter's taps' weights and clamped indices, then the sum of
// every tap pair's weight product times its table row, in the taps' order.
func decomposePosResample(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 0 || len(o.Out) != 1 || len(o.Weight) != 1 || a.Span < 2 || o.Weight[0].Rows != a.Span*a.Span ||
		a.Chain < graph.ChainAligned || a.Chain > graph.ChainHalfPixel || a.Filter < graph.FilterBilinear || a.Filter > graph.FilterCubic {
		return nil, fmt.Errorf("a position resample of %d weights, chain %d, filter %d", len(o.Weight), a.Chain, a.Filter)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(o.Out[0])
	R, E, S := sh[0], sh[1], a.Span
	var x exprs
	r, e := x.coord(0), x.coord(1)
	in := []Operand{TablePos.op(), TableRequest.op()}
	facts := Facts{Extents: map[int]int{0: R + 1, 1: RequestRows + R}}
	lead := uint32(a.Offset)
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), x.u(1))
	live := x.op(OpMax, U, x.load(0, x.u(0)), x.u(lead+1))
	gh := x.op(OpMax, U, x.op(OpDiv, U, x.op(OpSub, U, live, x.u(lead)), gw), x.u(1))
	q := x.op(OpSub, U, x.op(OpMax, U, x.load(0, x.op(OpAdd, U, r, x.u(1))), x.u(lead)), x.u(lead))
	f := func(v float32) int { return x.f(v) }
	side := f(float32(S))
	type tap struct{ w, i int }
	axis := func(cell, size int) []tap {
		i, n := x.un(OpCvtF32, F, cell), x.un(OpCvtF32, F, size)
		var v int
		switch a.Chain {
		case graph.ChainAligned:
			v = x.op(OpDiv, F, x.op(OpMul, F, i, f(float32(S-1))), x.op(OpMax, F, x.op(OpSub, F, n, f(1)), f(1)))
		case graph.ChainHalfPixel:
			v = x.op(OpAdd, F, x.op(OpMul, F, x.op(OpAdd, F, i, f(0.5)), x.op(OpDiv, F, side, n)), f(-0.5))
		default:
			v = x.op(OpAdd, F, x.op(OpAdd, F, x.op(OpMul, F, x.op(OpDiv, F, x.op(OpAdd, F, i, f(0.5)), n), f(2)), f(-1)), f(1))
			v = x.op(OpDiv, F, x.op(OpAdd, F, x.op(OpMul, F, v, side), f(-1)), f(2))
		}
		if a.Filter == graph.FilterLinear {
			v = x.op(OpMax, F, v, f(0))
		}
		// floor(v), in integers where the chain is a ratio of them, so the
		// taps' indices are a general register's words and not vectors held
		// across the row; a float within a rounding of an integer reads the
		// neighbouring tap at a weight of one, the same sum.
		var fl int
		switch a.Chain {
		case graph.ChainAligned:
			fl = x.un(OpBitcast, I, x.op(OpDiv, U, x.op(OpMul, U, cell, x.u(uint32(S-1))),
				x.op(OpMax, U, x.op(OpSub, U, size, x.u(1)), x.u(1))))
		case graph.ChainHalfPixel:
			num := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, cell, x.u(2)), x.u(1)), x.u(uint32(S))), size)
			q := x.op(OpDiv, U, num, x.op(OpMul, U, size, x.u(2)))
			fl = x.op(OpSub, I, x.un(OpBitcast, I, q), x.un(OpBitcast, I, x.u(1)))
			if a.Filter == graph.FilterLinear {
				fl = x.op(OpMax, I, fl, x.un(OpBitcast, I, x.u(0)))
			}
		default:
			tr := x.un(OpCvtI32, I, v)
			above := x.op(OpLess, ir.Pred, v, x.un(OpCvtF32, F, tr))
			fl = x.sel(I, above, x.op(OpSub, I, tr, x.un(OpBitcast, I, x.u(1))), tr)
		}
		t := x.op(OpSub, F, v, x.un(OpCvtF32, F, fl))
		abs := func(v int) int { return x.op(OpMax, F, v, x.op(OpSub, F, f(0), v)) }
		conv1 := func(v int) int {
			return x.op(OpAdd, F, x.op(OpMul, F, x.op(OpMul, F, x.op(OpSub, F, x.op(OpMul, F, f(1.25), v), f(2.25)), v), v), f(1))
		}
		conv2 := func(v int) int {
			b := x.op(OpAdd, F, x.op(OpMul, F, f(-0.75), v), f(3.75))
			return x.op(OpAdd, F, x.op(OpMul, F, x.op(OpSub, F, x.op(OpMul, F, b, v), f(6)), v), f(3))
		}
		var ws []int
		off := 0
		switch a.Filter {
		case graph.FilterBilinear:
			ws = []int{x.op(OpMax, F, x.op(OpSub, F, f(1), abs(t)), f(0)), x.op(OpMax, F, x.op(OpSub, F, f(1), abs(x.op(OpSub, F, t, f(1)))), f(0))}
		case graph.FilterLinear:
			ws = []int{x.op(OpSub, F, f(1), t), t}
		default:
			off = -1
			ws = []int{conv2(x.op(OpAdd, F, t, f(1))), conv1(t), conv1(x.op(OpSub, F, f(1), t)), conv2(x.op(OpSub, F, f(2), t))}
		}
		out := make([]tap, len(ws))
		for k, w := range ws {
			j := x.op(OpAdd, I, fl, x.un(OpBitcast, I, x.u(uint32(int32(k+off)))))
			j = x.op(OpMin, I, x.op(OpMax, I, j, x.un(OpBitcast, I, x.u(0))), x.un(OpBitcast, I, x.u(uint32(S-1))))
			out[k] = tap{w, x.un(OpBitcast, U, j)}
		}
		return out
	}
	ys := axis(x.op(OpDiv, U, q, gw), gh)
	xs := axis(x.op(OpRem, U, q, gw), gw)
	// Separable: each row tap's sum over the column taps, weighted by the
	// row tap's weight, so the floats held across the row are the axes'
	// weights, not their products.
	acc := -1
	for _, ty := range ys {
		in1 := -1
		for _, tx := range xs {
			row := x.op(OpAdd, U, x.op(OpMul, U, ty.i, x.u(uint32(S))), tx.i)
			v, err := weightElement(&x, wo, S*S, E, &in, &facts, row, e)
			if err != nil {
				return nil, fmt.Errorf("a position table of %s: %w", wo.Format, err)
			}
			term := x.op(OpMul, F, tx.w, v)
			if in1 < 0 {
				in1 = term
			} else {
				in1 = x.op(OpAdd, F, in1, term)
			}
		}
		term := x.op(OpMul, F, ty.w, in1)
		if acc < 0 {
			acc = term
		} else {
			acc = x.op(OpAdd, F, acc, term)
		}
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}

// decomposePosAxes is each row's position rows of a table of two axes: the
// row at the cell's column plus the row at Span plus the cell's row.
func decomposePosAxes(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 0 || len(o.Out) != 1 || len(o.Weight) != 1 || a.Span < 1 || o.Weight[0].Rows != 2*a.Span {
		return nil, fmt.Errorf("a two-axis position table of %d weights over %d rows an axis", len(o.Weight), a.Span)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	U := ir.U32
	sh := c.rows(o.Out[0])
	R, E, A := sh[0], sh[1], uint32(a.Span)
	var x exprs
	r, e := x.coord(0), x.coord(1)
	in := []Operand{TablePos.op(), TableRequest.op()}
	facts := Facts{Extents: map[int]int{0: R + 1, 1: RequestRows + R}}
	lead := uint32(a.Offset)
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), x.u(1))
	q := x.op(OpSub, U, x.op(OpMax, U, x.load(0, x.op(OpAdd, U, r, x.u(1))), x.u(lead)), x.u(lead))
	col := x.op(OpMin, U, x.op(OpRem, U, q, gw), x.u(A-1))
	row := x.op(OpAdd, U, x.op(OpMin, U, x.op(OpDiv, U, q, gw), x.u(A-1)), x.u(A))
	cv, err := weightElement(&x, wo, 2*a.Span, E, &in, &facts, col, e)
	if err != nil {
		return nil, fmt.Errorf("a position table of %s: %w", wo.Format, err)
	}
	rv, err := weightElement(&x, wo, 2*a.Span, E, &in, &facts, row, e)
	if err != nil {
		return nil, fmt.Errorf("a position table of %s: %w", wo.Format, err)
	}
	x.op(OpAdd, ir.F32, cv, rv)
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}

// SpliceNone is a step row's request word naming no picture row.
const SpliceNone = ^uint32(0)

// decomposeSplice is the step's input rows with a picture's rows in place:
// row r takes row RequestRows+r of the request of the projector's rows (a
// picture's step's, Step.Picture of them), where the word is not SpliceNone.
func decomposeSplice(o graph.Op, c *Context) ([]Prim, error) {
	P := c.Step.Picture
	if len(o.In) != 2 || len(o.Out) != 1 || P < 1 || c.width(o.In[1]) < c.width(o.In[0]) {
		return nil, fmt.Errorf("a splice of %d operands over pictures of %d rows", len(o.In), P)
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.Out[0])
	// A picture's row is its projected row first, Wp wide with its taps'.
	R, Wp := sh[0], c.width(o.In[1])
	var x exprs
	r, e := x.coord(0), x.coord(1)
	at := x.load(2, x.op(OpAdd, U, x.u(RequestRows), r))
	none := x.op(OpLess, ir.Pred, x.u(SpliceNone-1), at)
	pic := x.at(1, []int{P, Wp}, x.op(OpMin, U, at, x.u(uint32(P-1))), e)
	x.sel(F, none, x.at(0, sh, r, e), pic)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), c.value(o.In[1]), TableRequest.op()}, Out: c.value(o.Out[0]),
		Expr: x.e, Facts: Facts{Extents: map[int]int{1: P * Wp, 2: RequestRows + R}}}}, nil
}
