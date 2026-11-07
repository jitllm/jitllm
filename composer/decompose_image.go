package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// RequestGrid and RequestGridH are a picture's step's request words: its
// grid's width and height in patches.
const (
	RequestGrid  = 0
	RequestGridH = 1
)

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
// channel's mean and std; zero past the picture, and where the picture is
// padded (PicPad) laid out over its padded grid, a pad cell a zero sample
// normalised.
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
	wd, hd := padded(&x, ow, a.PicPad), padded(&x, oh, a.PicPad)
	oy, ox := x.op(OpDiv, U, pix, wd), x.op(OpRem, U, pix, wd)
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
	norm := func(v int) int {
		return x.op(OpDiv, F, x.op(OpSub, F, x.op(OpDiv, F, v, x.f(255)), byChannel(a.Mean)), byChannel(a.Std))
	}
	v = norm(v)
	if a.PicPad > 0 {
		pad := norm(x.f(0))
		v = x.sel(F, x.op(OpLess, ir.Pred, ox, ow), x.sel(F, x.op(OpLess, ir.Pred, oy, oh), v, pad), pad)
	}
	x.sel(F, x.op(OpLess, ir.Pred, pix, x.op(OpMul, U, wd, hd)), v, x.f(0))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{TablePicture.op(), TableRequest.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{0: PictureWords, 1: RequestPicOH + 1}}}}, nil
}

// padded is n rounded up to a multiple of unit (U32), n itself where unit is
// zero.
func padded(x *exprs, n, unit int) int {
	if unit <= 0 {
		return n
	}
	U := ir.U32
	u := x.u(uint32(unit))
	return x.op(OpMul, U, x.op(OpDiv, U, x.op(OpAdd, U, n, x.u(uint32(unit-1))), u), u)
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
// mean and std; zero past the picture, and where the picture is padded
// (PicPad) laid out over its padded grid, a pad cell a zero sample
// normalised. A source row or column past the picture's reads its last, at
// a weight of zero.
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
	var ow, oh, py, px int
	if a.PicLive {
		in = append(in, TableRequest.op())
		ext[1] = reqWords
		ow = m.op(OpMin, U, m.op(OpMax, U, m.load(1, mu(RequestPicOW)), mu(1)), mu(a.PicW))
		oh = m.op(OpMin, U, m.op(OpMax, U, m.load(1, mu(RequestPicOH)), mu(1)), mu(a.PicH))
		wd, hd := padded(&m, ow, a.PicPad), padded(&m, oh, a.PicPad)
		pix, ch := m.op(OpDiv, U, flat, mu(3)), m.op(OpRem, U, flat, mu(3))
		py, px = m.op(OpDiv, U, pix, wd), m.op(OpRem, U, pix, wd)
		at = m.op(OpAdd, U, m.op(OpMul, U, m.op(OpAdd, U, m.op(OpMul, U, py, mu(a.PicW)), px), mu(3)), ch)
		live = m.op(OpMul, U, m.op(OpMul, U, wd, hd), mu(3))
	}
	norm := func(b int) int {
		return m.op(OpDiv, F, m.op(OpSub, F, m.op(OpDiv, F, b, m.f(255)), byChannel(a.Mean)), byChannel(a.Std))
	}
	val := norm(m.un(OpCvtF32, F, finish(&m, m.load(0, m.op(OpMin, U, at, mu(n-1))))))
	if a.PicLive && a.PicPad > 0 {
		pad := norm(m.f(0))
		val = m.sel(F, m.op(OpLess, ir.Pred, px, ow), m.sel(F, m.op(OpLess, ir.Pred, py, oh), val, pad), pad)
	}
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
	gh := x.op(OpMax, U, x.op(OpDiv, U, x.op(OpSub, U, x.op(OpMax, U, x.load(2, x.u(0)), u(lead+1)), u(lead)), gw), u(1))
	py, pxi := tileCell(&x, r, gw, gh, o.Attr.Width, o.Attr.Window)
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

// decomposeUntile is each output row the input row of the cell at its
// raster index, of the grid of cells the request's patch grid over Width
// is, taken from the input's nested order (tileIndex).
func decomposeUntile(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.In[0]) != c.width(o.Out[0]) || o.Attr.Window < 1 {
		return nil, fmt.Errorf("an untile of %d operands over windows of %d", len(o.In), o.Attr.Window)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	m := max(o.Attr.Width, 1)
	var x exprs
	r := x.coord(0)
	u := func(v int) int { return x.u(uint32(v)) }
	gw := x.op(OpMax, U, x.op(OpDiv, U, x.load(1, x.u(RequestGrid)), u(m)), u(1))
	gh := x.op(OpMax, U, x.op(OpDiv, U, x.op(OpDiv, U, x.load(2, x.u(0)), u(m*m)), gw), u(1))
	at := tileIndex(&x, x.op(OpDiv, U, r, gw), x.op(OpRem, U, r, gw), gw, gh, o.Attr.Window)
	x.at(0, ish, x.op(OpMin, U, at, u(ish[0]-1)), x.coord(1))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), TableRequest.op(), TablePos.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{1: RequestRows + sh[0], 2: c.Step.Rows + 1}}}}, nil
}

// tileIndex is the index of cell (cy, cx) of a grid gw cells wide and gh tall
// in its nested order of win x win windows: tileCell's inverse at one level.
func tileIndex(x *exprs, cy, cx, gw, gh, win int) int {
	U := ir.U32
	u := func(v int) int { return x.u(uint32(v)) }
	wy, wx := x.op(OpDiv, U, cy, u(win)), x.op(OpDiv, U, cx, u(win))
	top, left := x.op(OpMul, U, wy, u(win)), x.op(OpMul, U, wx, u(win))
	wh := x.op(OpMax, U, x.op(OpMin, U, u(win), x.op(OpSub, U, x.op(OpMax, U, gh, top), top)), u(1))
	ww := x.op(OpMax, U, x.op(OpMin, U, u(win), x.op(OpSub, U, x.op(OpMax, U, gw, left), left)), u(1))
	at := x.op(OpAdd, U, x.op(OpMul, U, top, gw), x.op(OpMul, U, left, wh))
	return x.op(OpAdd, U, at, x.op(OpAdd, U, x.op(OpMul, U, x.op(OpSub, U, cy, top), ww), x.op(OpSub, U, cx, left)))
}

// tileCell is the row and column of the patch at index r of a grid gw
// patches wide and gh tall in nested order (graph.OpPatchify): merge x merge
// merge windows of patches (merge 0 or 1: none), the patches of one in
// raster order and consecutive; win x win windows of merge windows (win 0:
// one window, the grid), each window's merge windows in raster order and
// consecutive; the windows in raster order, those at the grid's right and
// lower edges cut to it. The grid's sides are whole merge windows.
func tileCell(x *exprs, r, gw, gh, merge, win int) (y, c int) {
	U := ir.U32
	u := func(v int) int { return x.u(uint32(v)) }
	m := max(merge, 1)
	unit, in := r, 0
	if m > 1 {
		unit, in = x.op(OpDiv, U, r, u(m*m)), x.op(OpRem, U, r, u(m*m))
		gw, gh = x.op(OpMax, U, x.op(OpDiv, U, gw, u(m)), u(1)), x.op(OpMax, U, x.op(OpDiv, U, gh, u(m)), u(1))
	}
	var uy, ux int
	if win < 1 {
		uy, ux = x.op(OpDiv, U, unit, gw), x.op(OpRem, U, unit, gw)
	} else {
		// A row of windows is win unit rows of the grid's width, but for the
		// last; a window of it is win units wide, but for the last.
		band := x.op(OpMul, U, gw, u(win))
		wy, rest := x.op(OpDiv, U, unit, band), x.op(OpRem, U, unit, band)
		top := x.op(OpMul, U, wy, u(win))
		wh := x.op(OpMax, U, x.op(OpMin, U, u(win), x.op(OpSub, U, x.op(OpMax, U, gh, top), top)), u(1))
		block := x.op(OpMul, U, wh, u(win))
		wx, at := x.op(OpDiv, U, rest, block), x.op(OpRem, U, rest, block)
		left := x.op(OpMul, U, wx, u(win))
		ww := x.op(OpMax, U, x.op(OpMin, U, u(win), x.op(OpSub, U, x.op(OpMax, U, gw, left), left)), u(1))
		uy = x.op(OpAdd, U, top, x.op(OpDiv, U, at, ww))
		ux = x.op(OpAdd, U, left, x.op(OpRem, U, at, ww))
	}
	if m == 1 {
		return uy, ux
	}
	y = x.op(OpAdd, U, x.op(OpMul, U, uy, u(m)), x.op(OpDiv, U, in, u(m)))
	c = x.op(OpAdd, U, x.op(OpMul, U, ux, u(m)), x.op(OpRem, U, in, u(m)))
	return y, c
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
// grid: the row's cell (tileCell, the patch gather's order), on each axis the source coordinate by the op's
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
	cy, cx := tileCell(&x, q, gw, gh, a.Width, a.Window)
	ys := axis(cy, gh)
	xs := axis(cx, gw)
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

// decomposeRowBreak is the input's rows, a grid W wide (the request's
// patch grid over Span), with the weight's row after each of the grid's
// rows: output row r is input row (r/(W+1))*W + r%(W+1), or the weight's
// where r%(W+1) is W.
func decomposeRowBreak(o graph.Op, c *Context) ([]Prim, error) {
	S := o.Attr.Span
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 || S < 1 || o.Weight[0].K != c.width(o.In[0]) ||
		c.width(o.Out[0]) != c.width(o.In[0]) {
		return nil, fmt.Errorf("a row break of %d operands, %d weights, by %d", len(o.In), len(o.Weight), S)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	w := o.Weight[0]
	var x exprs
	r, i := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	W := x.op(OpDiv, U, x.op(OpMax, U, x.load(2, u(RequestGrid)), u(S)), u(S))
	line := x.op(OpAdd, U, W, u(1))
	k := x.op(OpRem, U, r, line)
	src := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, r, line), W), k)
	in := x.at(0, ish, x.op(OpMin, U, src, u(ish[0]-1)), i)
	nl := x.at(1, []int{sh[1]}, i)
	x.sel(ir.F32, x.op(OpLess, ir.Pred, k, W), in, nl)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), {Kind: OperandVector, Role: w.Role, Name: w.Role.String()},
		TableRequest.op()}, Out: c.value(o.Out[0]), Expr: x.e, Facts: Facts{Extents: map[int]int{2: RequestRows + sh[0]}}}}, nil
}

// decomposeWrapRows is the first weight's row, the input's rows and the
// second weight's row: the input a grid of the step's patches (its rows
// less Offset lead rows, the request's width) over Span, each of its rows
// Width rows longer, so its rows are H*(W+Width) and the second weight's
// row output row H*(W+Width)+1.
func decomposeWrapRows(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	S := a.Span
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 2 || S < 1 || a.Width < 0 ||
		o.Weight[0].K != c.width(o.In[0]) || o.Weight[1].K != c.width(o.In[0]) || c.width(o.Out[0]) != c.width(o.In[0]) {
		return nil, fmt.Errorf("a row wrap of %d operands, %d weights, by %d", len(o.In), len(o.Weight), S)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	var x exprs
	r, i := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	// The framed rows: H rows of W windows and Width more, H and W the
	// grid's height and width over S, then the end row.
	framed := func(x *exprs) int {
		u := func(v int) int { return x.u(uint32(v)) }
		gw := x.op(OpMax, U, x.load(3, u(RequestGrid)), u(S))
		gh := x.op(OpMax, U, x.load(3, u(RequestGridH)), u(S))
		w := x.op(OpAdd, U, x.op(OpDiv, U, gw, u(S)), u(a.Width))
		return x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, gh, u(S)), w), u(1))
	}
	// A row past the end row is clamped to it, recomputing it: no row
	// past the picture's is written (the rows after it are the prompt's).
	end := framed(&x)
	rr := x.op(OpMin, U, r, x.op(OpMin, U, end, u(sh[0]-1)))
	src := x.op(OpMin, U, x.op(OpSub, U, x.op(OpMax, U, rr, u(1)), u(1)), u(ish[0]-1))
	in := x.at(0, ish, src, i)
	bv, ev := x.at(1, []int{sh[1]}, i), x.at(2, []int{sh[1]}, i)
	v := x.sel(ir.F32, x.op(OpLess, ir.Pred, rr, u(1)), bv, x.sel(ir.F32, x.op(OpLess, ir.Pred, rr, end), in, ev))
	vec := func(w graph.Binding) Operand {
		return Operand{Kind: OperandVector, Role: w.Role, Name: w.Role.String()}
	}
	out := c.value(o.Out[0])
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), vec(o.Weight[0]), vec(o.Weight[1]),
		TableRequest.op()}, Out: out, Expr: x.e,
		Facts:  Facts{Extents: map[int]int{3: RequestRows + sh[0]}},
		Stores: []PrimStore{{Out: out, Ax: []int{rr, i}, Ext: sh, Value: v}}}}, nil
}
