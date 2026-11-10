package composer

import (
	"fmt"
	"math"

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
	// RequestPicFrame is the byte a video's second frame starts at in the
	// step's picture (graph.Attrs.PicFrame).
	RequestPicFrame
	// RequestFrameStride through RequestFrameSub are the second frame's own
	// layout, as RequestPicStride through RequestPicSub are the first's: a
	// pair's frames may decode to different layouts.
	RequestFrameStride
	RequestFrameCStride
	RequestFrameP2
	RequestFrameP3
	RequestFrameSub
	// RequestPicBand is the first source row of a band a Pillow resize's
	// band pass runs over (graph.PicBand), and RequestPicBandRows the source
	// rows a band holds (PicBandRows).
	RequestPicBand
	RequestPicBandRows
	// RequestPicCol is the first source column of a band of columns a
	// Pillow resize's column pass runs over (graph.PicCols): 0 replaces the
	// horizontal accumulator, any other adds into it.
	RequestPicCol
	// RequestMark is a square's marker row in the head's table plus one, 0
	// for a square with none (graph.OpMarkRow), and RequestTrail the one
	// after its rows.
	RequestMark
	RequestTrail
	// RequestMarkN and RequestTrailN are the runs' lengths, a mark's none
	// where it is 0.
	RequestMarkN
	RequestTrailN
	// RequestPicTaps is where a picture resized by Pillow's filters
	// (graph.ChainPIL) has its taps: the output's columns' (PILTaps), then
	// its rows', each entry PILEntryWords(Taps) words.
	RequestPicTaps
)

// PictureWords is the most words a step's picture holds: its samples are
// read at U32 byte offsets (pictureSample).
const PictureWords = 1<<30 - 1

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
	wd, hd, fx, fy := frame(&x, a, ow, oh)
	oy := x.op(OpSub, U, x.op(OpDiv, U, pix, wd), fy)
	ox := x.op(OpSub, U, x.op(OpRem, U, pix, wd), fx)
	sw := x.op(OpMax, U, x.load(1, u(RequestPicW)), u(1))
	sh2 := x.op(OpMax, U, x.load(1, u(RequestPicH)), u(1))
	stride := x.load(1, u(picWordsOf(a).stride))
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
		return x.un(OpCvtF32, F, pictureSample(&x, a, stride, y, xx, ch))
	}
	lerp := func(p, q, t int) int {
		return x.op(OpAdd, F, x.op(OpMul, F, p, x.op(OpSub, F, x.f(1), t)), x.op(OpMul, F, q, t))
	}
	v := lerp(lerp(sample(y0, x0), sample(y0, x1), tx), lerp(sample(y1, x0), sample(y1, x1), tx), ty)
	norm := func(v int) int { return normalized(&x, a, ch, v) }
	v = norm(v)
	if a.PicPad > 0 || a.PicBox {
		pad := norm(fillOf(&x, a, ch))
		v = x.sel(F, x.op(OpLess, ir.Pred, ox, ow), x.sel(F, x.op(OpLess, ir.Pred, oy, oh), v, pad), pad)
	}
	x.sel(F, x.op(OpLess, ir.Pred, pix, x.op(OpMul, U, wd, hd)), v, x.f(0))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{TablePicture.op(), TableRequest.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{0: PictureWords, 1: RequestPicOH + 1}}}}, nil
}

// frame is a live picture's frame and its place in it: the output w x h
// rounded up to the pad's units, or the op's PicW x PicH with the picture
// centred (PicBox), or the output itself; the picture's corner (U32s).
func frame(x *exprs, a graph.Attrs, w, h int) (fw, fh, x0, y0 int) {
	if !a.PicBox {
		return padded(x, w, a.PicPad), padded(x, h, a.PicPad), x.u(0), x.u(0)
	}
	U := ir.U32
	fw, fh = x.u(uint32(a.PicW)), x.u(uint32(a.PicH))
	return fw, fh, x.op(OpShr, U, x.op(OpSub, U, fw, w), x.u(1)), x.op(OpShr, U, x.op(OpSub, U, fh, h), x.u(1))
}

// fillOf is channel ch's fill sample (F32): the box's colour, zero for a
// pad.
func fillOf(x *exprs, a graph.Attrs, ch int) int {
	F := ir.F32
	f := func(c int) int { return x.f(float32(a.PicFill[c])) }
	return x.sel(F, x.op(OpLess, ir.Pred, ch, x.u(1)), f(0), x.sel(F, x.op(OpLess, ir.Pred, ch, x.u(2)), f(1), f(2)))
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

// accOf is the accumulator instance a picture op's bands add into: a video
// pair's second frame its own (graph.Attrs.PicFrame).
func accOf(a graph.Attrs) int {
	if a.PicFrame {
		return 1
	}
	return 0
}

// picWords is the request words a picture op's frame reads its layout
// from: a video pair's second frame its own (graph.Attrs.PicFrame).
type picWords struct{ stride, cstride, p2, p3, sub int }

func picWordsOf(a graph.Attrs) picWords {
	if a.PicFrame {
		return picWords{RequestFrameStride, RequestFrameCStride, RequestFrameP2, RequestFrameP3, RequestFrameSub}
	}
	return picWords{RequestPicStride, RequestPicCStride, RequestPicP2, RequestPicP3, RequestPicSub}
}

// pictureSample is channel ch of the picture's pixel (y, xx), the byte its
// format holds or makes (U32): the picture in operand 0, the request in 1,
// stride its rows' bytes.
func pictureSample(x *exprs, a graph.Attrs, stride, y, xx, ch int) int {
	U := ir.U32
	px, pw := a.Pixels, picWordsOf(a)
	u := func(v int) int { return x.u(uint32(v)) }
	byteAt := func(off int) int {
		if a.PicFrame {
			off = x.op(OpAdd, U, off, x.load(1, u(RequestPicFrame)))
		}
		w := x.load(0, x.op(OpShr, U, off, u(2)))
		return x.op(OpAnd, U, x.op(OpShr, U, w, x.op(OpShl, U, x.op(OpAnd, U, off, u(3)), u(3))), u(255))
	}
	at := x.op(OpAdd, U, x.op(OpMul, U, y, stride), xx)
	switch px {
	case graph.PixRGBA8:
		at = x.op(OpAdd, U, x.op(OpAdd, U, x.op(OpMul, U, y, stride), x.op(OpMul, U, xx, u(4))), ch)
	case graph.PixPaletted8:
		at = x.op(OpAdd, U, x.op(OpAdd, U, x.load(1, u(pw.p2)), x.op(OpMul, U, byteAt(at), u(4))), ch)
	case graph.PixYCbCr8:
		return ycbcr(x, pw, byteAt, at, y, xx, ch)
	}
	return byteAt(at)
}

// decomposePicturePIL is a picture resized by Pillow's antialiased filter
// (graph.ChainPIL), its taps in the request (PILTaps): for each source row
// of a band and output column's channel the sum over the tap axis (Taps at
// most, its trips the column's window's count) of each weight times its
// sample, in integers; then for each output pixel's channel the same over
// its rows, each term the horizontal sum made a byte -- from half Pillow's
// unit, clamped at zero, shifted down by its fixed point and clamped to 255;
// then that sum made a byte, read as [0, 1] and normalised by its channel's
// mean and std; zero past the picture, and where the picture is padded
// (PicPad) laid out over its padded grid, a pad cell a zero sample
// normalised. A source row or column past the picture's reads its last, at
// a weight of zero. The sums are laid out at the output's live width, the
// vertical ones holding the op's area (graph.Attrs.Area) and the horizontal
// ones PicHCap pixels, so a band holds PicBandRows source rows.
func decomposePicturePIL(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(o.Out[0])
	W, K := sh[1], a.Taps
	if W%3 != 0 || W >= 1<<15 {
		return nil, fmt.Errorf("a picture's rows of %d samples, not whole pixels", W)
	}
	E := PILEntryWords(K)
	A, HC := a.Area(), PicHCap(a)
	reqWords := RequestPicTaps + PicEntries(a)*E
	name := c.Graph.Values[o.Out[0]].Name
	hsum := Operand{Kind: OperandScratch, Name: name + ".rows", Elem: I}
	vsum := Operand{Kind: OperandScratch, Name: name + ".cols", Elem: I}
	// liveW is the output's width (U32): the request's, at most PicW, for a
	// picture of its own size, and PicW for one of the op's.
	liveW := func(x *exprs) int {
		if !a.PicLive {
			return x.u(uint32(a.PicW))
		}
		return x.op(OpMin, U, x.op(OpMax, U, x.load(1, x.u(RequestPicOW)), x.u(1)), x.u(uint32(a.PicW)))
	}
	// liveH is the output's height (U32), as liveW its width.
	liveH := func(x *exprs) int {
		if !a.PicLive {
			return x.u(uint32(a.PicH))
		}
		return x.op(OpMin, U, x.op(OpMax, U, x.load(1, x.u(RequestPicOH)), x.u(1)), x.u(uint32(a.PicH)))
	}
	// rowEntry is output row oy's tap entry, after the columns' (U32), oy at
	// most the last live row.
	rowEntry := func(x *exprs, oy int) int {
		oy = x.op(OpMin, U, oy, x.op(OpSub, U, liveH(x), x.u(1)))
		return x.op(OpAdd, U, x.u(RequestPicTaps), x.op(OpMul, U, x.op(OpAdd, U, liveW(x), oy), x.u(uint32(E))))
	}
	// finish is a sum of weights times bytes as Pillow's byte (U32).
	finish := func(x *exprs, acc int) int {
		v := x.un(OpBitcast, U, x.op(OpMax, I, x.op(OpAdd, I, acc, x.i(1<<(pilPrecision-1))), x.i(0)))
		return x.op(OpMin, U, x.op(OpShr, U, v, x.u(pilPrecision)), x.u(255))
	}
	// weight is tap t's weight at entry base of the request's taps.
	weight := func(x *exprs, base, t int) int {
		return x.un(OpBitcast, I, x.load(1, x.op(OpAdd, U, x.op(OpAdd, U, base, x.u(2)), t)))
	}
	// trips is a tap axis as a range: the entry's count of taps, at entry
	// base's word after its first sample, the loop's trips.
	trips := func(entry func(x *exprs) int) *Range {
		var lo, hi exprs
		lo.u(0)
		hi.load(1, hi.op(OpAdd, U, entry(&hi), hi.u(1)))
		return &Range{First: lo.e, End: hi.e}
	}
	facts := Facts{Extents: map[int]int{0: PictureWords, 1: reqWords}}

	// The horizontal pass, a reduction a channel: the channel a constant,
	// sum i the band's source row i / w and output column i % w.
	var prims []Prim
	for ch := range 3 {
		if a.PicColumns {
			// The row band's horizontal sums are its column bands'.
			break
		}
		var h exprs
		i, t := h.coord(0), h.coord(1)
		hu := func(v int) int { return h.u(uint32(v)) }
		w := liveW(&h)
		r, xo := h.op(OpDiv, U, i, w), h.op(OpRem, U, i, w)
		sw := h.op(OpMax, U, h.load(1, hu(RequestPicW)), hu(1))
		srcH := h.op(OpMax, U, h.load(1, hu(RequestPicH)), hu(1))
		src := r
		if a.PicPass == graph.PicBand || a.PicPass == graph.PicCols {
			src = h.op(OpAdd, U, r, h.load(1, hu(RequestPicBand)))
		}
		y := h.op(OpMin, U, src, h.op(OpSub, U, srcH, hu(1)))
		base := h.op(OpAdd, U, hu(RequestPicTaps), h.op(OpMul, U, xo, hu(E)))
		xi := h.op(OpMin, U, h.op(OpAdd, U, h.load(1, base), t), h.op(OpSub, U, sw, hu(1)))
		stride := h.load(1, hu(picWordsOf(a).stride))
		h.op(OpMul, I, weight(&h, base, t), h.un(OpBitcast, I, pictureSample(&h, a, stride, y, xi, hu(ch))))
		var hi exprs
		hi.op(OpAdd, U, hi.op(OpMul, U, hi.coord(0), hi.u(3)), hi.u(uint32(ch)))
		hr := trips(func(x *exprs) int {
			return x.op(OpAdd, U, x.u(RequestPicTaps), x.op(OpMul, U, x.op(OpRem, U, x.coord(0), liveW(x)), x.u(uint32(E))))
		})
		hr.Axis = 1
		hf := Facts{Extents: facts.Extents, OutExtent: HC * 3, Range: hr}
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{HC, K}, In: []Operand{TablePicture.op(), TableRequest.op()},
			Out: hsum, Combine: OpAdd, Type: I, Expr: h.e, Index: hi.e, Facts: hf})
	}

	if a.PicPass == graph.PicCols {
		// The column band's sums added into the horizontal accumulator, the
		// first column band's replacing what it held; the op's own value is
		// zeros.
		n := HC * 3
		var ac exprs
		i := ac.coord(0)
		first := ac.op(OpLess, ir.Pred, ac.load(1, ac.u(RequestPicCol)), ac.u(1))
		held := ac.sel(I, first, ac.i(0), ac.un(OpBitcast, I, ac.load(2, i)))
		ac.un(OpBitcast, U, ac.op(OpAdd, I, held, ac.load(0, i)))
		acc := Prim{Kind: PrimMap, Shape: []int{n}, In: []Operand{hsum, TableRequest.op(), TablePicHAcc.of(accOf(a))}, Expr: ac.e,
			Facts:  Facts{Extents: map[int]int{0: n, 1: reqWords, 2: n}, Whole: true},
			Stores: []PrimStore{{Out: TablePicHAccOut.of(accOf(a)), Index: i, Value: len(ac.e) - 1, Extent: n}}}
		var z exprs
		z.f(0)
		return append(prims, acc, Prim{Kind: PrimMap, Shape: sh, Out: c.value(o.Out[0]), Expr: z.e}), nil
	}
	vr := trips(func(x *exprs) int { return rowEntry(x, x.op(OpDiv, U, x.coord(0), liveW(x))) })
	if a.PicPass == graph.PicBand {
		// A band's pass reads only the taps whose source rows are its own:
		// from max(band, first) to min(band + rows, first + count), each
		// less the window's first row.
		var lo, hi exprs
		bound := func(x *exprs, last bool) {
			e := rowEntry(x, x.op(OpDiv, U, x.coord(0), liveW(x)))
			first, band := x.load(1, e), x.load(1, x.u(RequestPicBand))
			from := x.op(OpSub, U, x.op(OpMax, U, band, first), first)
			if last {
				to := x.op(OpMin, U, x.op(OpAdd, U, band, x.load(1, x.u(RequestPicBandRows))),
					x.op(OpAdd, U, first, x.load(1, x.op(OpAdd, U, e, x.u(1)))))
				x.op(OpSub, U, x.op(OpMax, U, to, x.op(OpAdd, U, first, from)), first)
			}
		}
		bound(&lo, false)
		bound(&hi, true)
		vr = &Range{First: lo.e, End: hi.e}
	}
	vr.Axis = 2
	var v exprs
	j, vch, vt := v.coord(0), v.coord(1), v.coord(2)
	if a.PicPass == graph.PicBand {
		vt = v.pos()
	}
	vu := func(n int) int { return v.u(uint32(n)) }
	vw := liveW(&v)
	oy, ox := v.op(OpDiv, U, j, vw), v.op(OpRem, U, j, vw)
	vsrcH := v.op(OpMax, U, v.load(1, vu(RequestPicH)), vu(1))
	vbase := rowEntry(&v, oy)
	yi := v.op(OpMin, U, v.op(OpAdd, U, v.load(1, vbase), vt), v.op(OpSub, U, vsrcH, vu(1)))
	if a.PicPass == graph.PicBand {
		// The tap's source row in the band's sums.
		yi = v.op(OpSub, U, yi, v.load(1, vu(RequestPicBand)))
	}
	hat := v.op(OpMin, U, v.op(OpAdd, U, v.op(OpMul, U, v.op(OpAdd, U, v.op(OpMul, U, yi, vw), ox), vu(3)), vch), vu(HC*3-1))
	hs, hv := hsum, v.load(2, hat)
	if a.PicColumns {
		hs, hv = TablePicHAcc.of(accOf(a)), v.un(OpBitcast, I, hv)
	}
	v.op(OpMul, I, weight(&v, vbase, vt), v.un(OpBitcast, I, finish(&v, hv)))

	n := A * 3
	vext := map[int]int{0: PictureWords, 1: reqWords, 2: HC * 3}
	vred := Prim{Kind: PrimReduce, Shape: []int{A, 3, K}, In: []Operand{TablePicture.op(), TableRequest.op(), hs},
		Out: vsum, Combine: OpAdd, Type: I, Expr: v.e, Facts: Facts{Extents: vext, Range: vr}}
	if a.PicPass == graph.PicBand {
		// The band's sums added into the accumulator, the first band's
		// replacing what it held; the op's own value is zeros.
		var ac exprs
		i := ac.coord(0)
		first := ac.op(OpLess, ir.Pred, ac.load(1, ac.u(RequestPicBand)), ac.u(1))
		held := ac.sel(I, first, ac.i(0), ac.un(OpBitcast, I, ac.load(2, i)))
		ac.un(OpBitcast, U, ac.op(OpAdd, I, held, ac.load(0, i)))
		acc := Prim{Kind: PrimMap, Shape: []int{n}, In: []Operand{vsum, TableRequest.op(), TablePicAcc.of(accOf(a))}, Expr: ac.e,
			Facts:  Facts{Extents: map[int]int{0: n, 1: reqWords, 2: n}, Whole: true},
			Stores: []PrimStore{{Out: TablePicAccOut.of(accOf(a)), Index: i, Value: len(ac.e) - 1, Extent: n}}}
		var z exprs
		z.f(0)
		return append(prims, vred, acc, Prim{Kind: PrimMap, Shape: sh, Out: c.value(o.Out[0]), Expr: z.e}), nil
	}
	if a.PicPass == graph.PicFinish {
		prims = nil
	}

	var m exprs
	mu := func(n int) int { return m.u(uint32(n)) }
	flat := m.op(OpAdd, U, m.op(OpMul, U, m.coord(0), mu(W)), m.coord(1))
	col := m.coord(1)
	mch := m.op(OpSub, U, col, m.op(OpMul, U, m.op(OpShr, U, m.op(OpMul, U, col, mu(0xAAAB)), mu(17)), mu(3)))
	// A live picture is dense in the output and laid at its live width in
	// the sums: its pixel (y, x) is sum y * w + x of the reductions above.
	at, live := flat, mu(n)
	in := []Operand{vsum}
	if a.PicPass == graph.PicFinish {
		in[0] = TablePicAcc.of(accOf(a))
	}
	ext := map[int]int{0: n}
	var ow, oh, py, px, fch int
	if a.PicLive {
		in = append(in, TableRequest.op())
		ext[1] = reqWords
		ow, oh = liveW(&m), liveH(&m)
		wd, hd, fx, fy := frame(&m, a, ow, oh)
		pix, ch := m.op(OpDiv, U, flat, mu(3)), m.op(OpRem, U, flat, mu(3))
		py, px = m.op(OpSub, U, m.op(OpDiv, U, pix, wd), fy), m.op(OpSub, U, m.op(OpRem, U, pix, wd), fx)
		fch = ch
		at = m.op(OpAdd, U, m.op(OpMul, U, m.op(OpAdd, U, m.op(OpMul, U, py, ow), px), mu(3)), ch)
		live = m.op(OpMul, U, m.op(OpMul, U, wd, hd), mu(3))
	}
	norm := func(b int) int { return normalized(&m, a, mch, b) }
	sum := m.load(0, m.op(OpMin, U, at, mu(n-1)))
	if a.PicPass == graph.PicFinish {
		sum = m.un(OpBitcast, I, sum)
	}
	val := norm(m.un(OpCvtF32, F, finish(&m, sum)))
	if a.PicLive && (a.PicPad > 0 || a.PicBox) {
		pad := norm(fillOf(&m, a, fch))
		val = m.sel(F, m.op(OpLess, ir.Pred, px, ow), m.sel(F, m.op(OpLess, ir.Pred, py, oh), val, pad), pad)
	}
	m.sel(F, m.op(OpLess, ir.Pred, flat, live), val, m.f(0))
	if a.PicPass == graph.PicWhole {
		prims = append(prims, vred)
	}
	return append(prims, Prim{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: m.e,
		Facts: Facts{Extents: ext}}), nil
}

// ycbcr is channel ch of the YCbCr pixel (y, xx), its luma at byte at: its
// chroma sample the pixel's coordinates, each plus its axis's phase, shifted
// down by its axis's subsampling (RequestPicSub), in the second and third
// planes; converted as image/color.YCbCrToRGB does, in integers -- luma
// times 0x10101 plus the channel's chroma terms, clamped to 24 bits, the top
// byte.
func ycbcr(x *exprs, pw picWords, byteAt func(int) int, at, y, xx, ch int) int {
	U, I := ir.U32, ir.I32
	u := func(v int) int { return x.u(uint32(v)) }
	s := x.load(1, u(pw.sub))
	field := func(k int) int { return x.op(OpAnd, U, x.op(OpShr, U, s, u(8*k)), u(255)) }
	cy := x.op(OpShr, U, x.op(OpAdd, U, y, field(3)), field(1))
	cx := x.op(OpShr, U, x.op(OpAdd, U, xx, field(2)), field(0))
	coff := x.op(OpAdd, U, x.op(OpMul, U, cy, x.load(1, u(pw.cstride))), cx)
	yy := x.un(OpBitcast, I, x.op(OpMul, U, byteAt(at), u(0x10101)))
	chroma := func(plane int) int {
		b := byteAt(x.op(OpAdd, U, x.load(1, u(plane)), coff))
		return x.op(OpSub, I, x.un(OpBitcast, I, b), x.i(128))
	}
	cb, cr := chroma(pw.p2), chroma(pw.p3)
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
	in := []Operand{c.value(o.In[0])}
	var gw, gsw int
	if W := o.Attr.GridW; W > 0 {
		// A stated grid's sides fold here: the map lowers no op over two
		// constants.
		gw, gsw = u(W), u(W/S)
	} else {
		in = append(in, TableRequest.op())
		gw = x.op(OpMax, U, x.load(1, x.u(RequestGrid)), u(S))
		gsw = x.op(OpDiv, U, gw, u(S))
	}
	slot, e := x.op(OpDiv, U, j, u(E)), x.op(OpRem, U, j, u(E))
	row := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, g, gsw), u(S)), x.op(OpDiv, U, slot, u(S)))
	colm := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpRem, U, g, gsw), u(S)), x.op(OpRem, U, slot, u(S)))
	p := x.op(OpAdd, U, x.op(OpMul, U, row, gw), colm)
	x.at(0, ish, x.op(OpMin, U, p, u(R-1)), e)
	if len(in) == 1 {
		return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e}}, nil
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e,
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
		a.Chain < graph.ChainAligned || a.Chain > graph.ChainHalfPixel || a.Filter < graph.FilterBilinear || a.Filter > graph.FilterAntialias ||
		a.Filter == graph.FilterAntialias && a.Chain != graph.ChainHalfPixel {
		return nil, fmt.Errorf("a position resample of %d weights, chain %d, filter %d", len(o.Weight), a.Chain, a.Filter)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	if a.Filter == graph.FilterAntialias {
		return decomposePosAntialias(o, c, wo)
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

// decomposePosAntialias is a position resample by torch's antialiased
// bilinear (graph.FilterAntialias): each row's cell's weights over the
// table's S samples an axis, then the row the sum of the table's S*S rows,
// each weighted by its sample's weight on both axes. On an axis of n cells
// the scale is S/n and sample j's weight for cell i max(0, 1 - |j+0.5 -
// (i+0.5)*scale| / max(scale, 1)) over the S weights' sum -- a sample past
// the support weighs nothing, so the sum is torch's window.
func decomposePosAntialias(o graph.Op, c *Context, wo Operand) ([]Prim, error) {
	a := o.Attr
	U, F := ir.U32, ir.F32
	sh := c.rows(o.Out[0])
	R, E, S := sh[0], sh[1], a.Span
	name := c.Graph.Values[o.Out[0]].Name
	lead := uint32(a.Offset)
	var prims []Prim
	// weights is each row's cell's triangle weights on one axis (down for
	// the rows, else across), each over their sum: a map of the row by the
	// samples.
	weights := func(down bool, tag string) Operand {
		ws := Operand{Kind: OperandScratch, Name: name + ".aaw" + tag}
		var w exprs
		f := func(v float32) int { return w.f(v) }
		r, j := w.coord(0), w.coord(1)
		gw := w.op(OpMax, U, w.load(1, w.u(RequestGrid)), w.u(1))
		live := w.op(OpMax, U, w.load(0, w.u(0)), w.u(lead+1))
		q := w.op(OpSub, U, w.op(OpMax, U, w.load(0, w.op(OpAdd, U, r, w.u(1))), w.u(lead)), w.u(lead))
		var gh, cy, cx int
		if a.Width <= 1 && a.Window == 0 {
			// A raster grid's cell by a float quotient, which the CPU
			// divides in lanes: exact below 2^24 patches, where no
			// quotient of two whole numbers rounds across a whole one.
			quo := func(v int) int {
				return w.un(OpBitcast, U, w.un(OpCvtI32, ir.I32, w.op(OpDiv, F, w.un(OpCvtF32, F, v), w.un(OpCvtF32, F, gw))))
			}
			gh = w.op(OpMax, U, quo(w.op(OpSub, U, live, w.u(lead))), w.u(1))
			cy = quo(q)
			cx = w.op(OpSub, U, q, w.op(OpMul, U, cy, gw))
		} else {
			gh = w.op(OpMax, U, w.op(OpDiv, U, w.op(OpSub, U, live, w.u(lead)), gw), w.u(1))
			cy, cx = tileCell(&w, q, gw, gh, a.Width, a.Window)
		}
		cell, size := cx, gw
		if down {
			cell, size = cy, gh
		}
		i, n := w.un(OpCvtF32, F, cell), w.un(OpCvtF32, F, size)
		centre := w.op(OpMul, F, w.op(OpAdd, F, i, f(0.5)), w.op(OpDiv, F, f(float32(S)), n))
		inv := w.op(OpMin, F, w.op(OpDiv, F, n, f(float32(S))), f(1))
		tri := func(at int) int {
			d := w.op(OpSub, F, w.op(OpAdd, F, at, f(0.5)), centre)
			d = w.op(OpMax, F, d, w.op(OpSub, F, f(0), d))
			return w.op(OpMax, F, w.op(OpSub, F, f(1), w.op(OpMul, F, d, inv)), f(0))
		}
		sum := tri(f(0))
		for k := 1; k < S; k++ {
			sum = w.op(OpAdd, F, sum, tri(f(float32(k))))
		}
		w.op(OpDiv, F, tri(w.un(OpCvtF32, F, j)), sum)
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, S}, In: []Operand{TablePos.op(), TableRequest.op()}, Out: ws,
			Expr: w.e, Facts: Facts{Extents: map[int]int{0: R + 1, 1: RequestRows + R}}})
		return ws
	}
	wyo, wxo := weights(true, ".y"), weights(false, ".x")

	// The row: the table's rows by both axes' weights.
	var x exprs
	rr, e, k := x.coord(0), x.coord(1), x.coord(2)
	wy := x.at(0, []int{R, S}, rr, x.op(OpDiv, U, k, x.u(uint32(S))))
	wx := x.at(1, []int{R, S}, rr, x.op(OpRem, U, k, x.u(uint32(S))))
	in := []Operand{wyo, wxo}
	facts := Facts{Extents: map[int]int{0: R * S, 1: R * S}}
	v, err := weightElement(&x, wo, S*S, E, &in, &facts, k, e)
	if err != nil {
		return nil, fmt.Errorf("a position table of %s: %w", wo.Format, err)
	}
	x.op(OpMul, F, x.op(OpMul, F, wy, wx), v)
	return append(prims, Prim{Kind: PrimReduce, Shape: []int{R, E, S * S}, In: in, Out: c.value(o.Out[0]),
		Combine: OpAdd, Expr: x.e, Facts: facts}), nil
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
	H := x.op(OpDiv, U, x.op(OpMax, U, x.load(2, u(RequestGridH)), u(S)), u(S))
	line := x.op(OpAdd, U, W, u(1))
	// A row past the frame's last is clamped to it, recomputing it: the
	// break may be the head's last op, and no row past the picture's is
	// written.
	last := x.op(OpSub, U, x.op(OpMul, U, H, line), u(1+o.Attr.Offset))
	rr := x.op(OpMin, U, r, x.op(OpMin, U, last, u(sh[0]-1)))
	k := x.op(OpRem, U, rr, line)
	src := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, rr, line), W), k)
	in := x.at(0, ish, x.op(OpMin, U, src, u(ish[0]-1)), i)
	nl := x.at(1, []int{sh[1]}, i)
	v := x.sel(ir.F32, x.op(OpLess, ir.Pred, k, W), in, nl)
	out := c.value(o.Out[0])
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), {Kind: OperandVector, Role: w.Role, Name: w.Role.String()},
		TableRequest.op()}, Out: out, Expr: x.e, Facts: Facts{Extents: map[int]int{2: RequestRows + sh[0]}},
		Stores: []PrimStore{{Out: out, Ax: []int{rr, i}, Ext: sh, Value: v}}}}, nil
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

// decomposeMarkRow is the input's rows -- a square's merged grid, the
// request's grid over Width, or where Width is 0 all of them -- after the run of the weight's rows the
// request names (RequestMark, its first row plus one, none where it is 0;
// RequestMarkN long, at most Span) and before the run it names after them
// (RequestTrail and RequestTrailN, at most Offset). A row past the square's
// last is clamped to it, recomputing it: no row past the square's is
// written, the next square's or the prompt's.
func decomposeMarkRow(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	S := a.Width
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 || S < 0 || o.Weight[0].Rows < 1 || a.Span < 1 || a.Offset < 0 ||
		o.Weight[0].K != c.width(o.In[0]) || c.width(o.Out[0]) != c.width(o.In[0]) {
		return nil, fmt.Errorf("a marker row of %d operands, %d weights, by %d", len(o.In), len(o.Weight), S)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	T := o.Weight[0].Rows
	var x exprs
	r, i := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	// body is the rows between the runs: the merged grid's, or where Width
	// is 0 every row of the input's.
	body := u(ish[0])
	if S > 0 {
		W := x.op(OpDiv, U, x.op(OpMax, U, x.load(1, u(RequestGrid)), u(S)), u(S))
		H := x.op(OpDiv, U, x.op(OpMax, U, x.load(1, u(RequestGridH)), u(S)), u(S))
		body = x.op(OpMul, U, H, W)
	}
	// run is a mark's first row and length: none where its word is 0, the
	// length at most the op's.
	run := func(at, n, most int) (first, length int) {
		m := x.load(1, u(at))
		first = x.op(OpSub, U, x.op(OpMax, U, m, u(1)), u(1))
		length = x.op(OpMin, U, x.op(OpMul, U, x.op(OpMin, U, m, u(1)), x.load(1, u(n))), u(most))
		return first, length
	}
	mark, lead := run(RequestMark, RequestMarkN, a.Span)
	trail, tail := u(0), u(0)
	if a.Offset > 0 {
		trail, tail = run(RequestTrail, RequestTrailN, a.Offset)
	}
	end := x.op(OpAdd, U, body, lead)
	last := x.op(OpSub, U, x.op(OpAdd, U, end, tail), u(1))
	rr := x.op(OpMin, U, r, x.op(OpMin, U, last, u(sh[0]-1)))
	src := x.op(OpMin, U, x.op(OpSub, U, x.op(OpMax, U, rr, lead), lead), u(ish[0]-1))
	in := []Operand{c.value(o.In[0]), TableRequest.op()}
	facts := Facts{Extents: map[int]int{1: RequestRows + sh[0]}}
	before := x.op(OpLess, ir.Pred, rr, lead)
	after := x.op(OpLess, ir.Pred, x.op(OpSub, U, end, u(1)), rr)
	named := x.sel(U, before, x.op(OpAdd, U, mark, rr), x.op(OpAdd, U, trail, x.op(OpSub, U, x.op(OpMax, U, rr, end), end)))
	row := x.op(OpMin, U, named, u(T-1))
	mv, err := weightElement(&x, wo, T, sh[1], &in, &facts, row, i)
	if err != nil {
		return nil, fmt.Errorf("a marker table of %s: %w", wo.Format, err)
	}
	v := x.sel(ir.F32, before, mv, x.sel(ir.F32, after, mv, x.at(0, ish, src, i)))
	out := c.value(o.Out[0])
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: out, Expr: x.e, Facts: facts,
		Stores: []PrimStore{{Out: out, Ax: []int{rr, i}, Ext: sh, Value: v}}}}, nil
}

// normalized is sample v (an F32 of the byte) of channel ch (a U32) read as
// [0, 1] and normalised by the channel's mean and std (operand order the
// op's): in float32, or where the op says (PicBF16) in bfloat16 -- the
// sample times 1/255, less the mean, over the std, each result rounded to
// nearest even and the mean and std rounded too, as torch's arithmetic on a
// bfloat16 tensor rounds them.
func normalized(x *exprs, a graph.Attrs, ch, v int) int {
	F, U := ir.F32, ir.U32
	byChannel := func(c [3]float64) int {
		k := func(i int) int {
			if a.PicBF16 {
				return x.f(bf16Of(float32(c[i])))
			}
			return x.f(float32(c[i]))
		}
		return x.sel(F, x.op(OpLess, ir.Pred, ch, x.u(1)), k(0), x.sel(F, x.op(OpLess, ir.Pred, ch, x.u(2)), k(1), k(2)))
	}
	if !a.PicBF16 {
		return x.op(OpDiv, F, x.op(OpSub, F, x.op(OpDiv, F, v, x.f(255)), byChannel(a.Mean)), byChannel(a.Std))
	}
	bf := func(v int) int {
		b := x.un(OpBitcast, U, v)
		odd := x.op(OpAnd, U, x.op(OpShr, U, b, x.u(16)), x.u(1))
		b = x.op(OpAdd, U, b, x.op(OpAdd, U, x.u(0x7fff), odd))
		return x.un(OpBitcast, F, x.op(OpAnd, U, b, x.u(0xffff0000)))
	}
	s := bf(x.op(OpMul, F, v, x.f(1.0/255)))
	return bf(x.op(OpDiv, F, bf(x.op(OpSub, F, s, byChannel(a.Mean))), byChannel(a.Std)))
}

// bf16Of is v rounded to bfloat16, to nearest even.
func bf16Of(v float32) float32 {
	b := math.Float32bits(v)
	b += 0x7fff + (b>>16)&1
	return math.Float32frombits(b &^ 0xffff)
}

// decomposePosBuckets is each row's row of a Span x Span position table at
// its cell of the request's grid bucketed onto it: row (Span*y)/gh, column
// (Span*x)/gw, in integers.
func decomposePosBuckets(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 0 || len(o.Out) != 1 || len(o.Weight) != 1 || a.Span < 1 || o.Weight[0].Rows != a.Span*a.Span {
		return nil, fmt.Errorf("a bucketed position table of %d weights over %d a side", len(o.Weight), a.Span)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	U := ir.U32
	sh := c.rows(o.Out[0])
	R, E, S := sh[0], sh[1], uint32(a.Span)
	var x exprs
	r, e := x.coord(0), x.coord(1)
	in := []Operand{TablePos.op(), TableRequest.op()}
	facts := Facts{Extents: map[int]int{0: R + 1, 1: RequestRows + R}}
	lead := uint32(a.Offset)
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), x.u(1))
	gh := x.op(OpMax, U, x.load(1, x.u(RequestGridH)), x.u(1))
	q := x.op(OpSub, U, x.op(OpMax, U, x.load(0, x.op(OpAdd, U, r, x.u(1))), x.u(lead)), x.u(lead))
	col := x.op(OpDiv, U, x.op(OpMul, U, x.op(OpRem, U, q, gw), x.u(S)), gw)
	row := x.op(OpDiv, U, x.op(OpMul, U, x.op(OpDiv, U, q, gw), x.u(S)), gh)
	at := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpMin, U, row, x.u(S-1)), x.u(S)), x.op(OpMin, U, col, x.u(S-1)))
	if _, err := weightElement(&x, wo, a.Span*a.Span, E, &in, &facts, at, e); err != nil {
		return nil, fmt.Errorf("a position table of %s: %w", wo.Format, err)
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}

// decomposeTableRows is the weight's rows, each a row of the output: a
// table the reader copies out, read as the block's vector.
func decomposeTableRows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 0 || len(o.Out) != 1 || len(o.Weight) != 1 || c.width(o.Out[0]) != o.Weight[0].K ||
		c.rows(o.Out[0])[0] != o.Weight[0].Rows {
		return nil, fmt.Errorf("a table's rows of %d weights", len(o.Weight))
	}
	sh := c.rows(o.Out[0])
	w := o.Weight[0]
	var x exprs
	x.at(0, []int{sh[0] * sh[1]}, x.op(OpAdd, ir.U32, x.op(OpMul, ir.U32, x.coord(0), x.u(uint32(sh[1]))), x.coord(1)))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{{Kind: OperandVector, Role: w.Role, Name: w.Role.String()}},
		Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeRowsAttend is each query row (operand 0) attending over the
// step's live rows of the keys and values (1 and 2): per head, the scores
// q.k times the scale -- none past the live rows -- their maximum, the
// weights exp(score - maximum) and their sum, the weighted values summed
// over the rows, and each element that sum over the weights'.
func decomposeRowsAttend(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	H, hd, G := a.NHead, a.HeadDim, a.NKVHead
	if G == 0 {
		G = H
	}
	rel := len(o.In) == 4 && len(o.Weight) == 1
	if !rel && len(o.In) != 3 || len(o.Out) != 1 || H < 1 || hd < 1 || G < 1 || H%G != 0 || c.width(o.In[0]) != H*hd ||
		c.width(o.In[1]) != G*hd || c.width(o.In[2]) != G*hd || c.width(o.Out[0]) != H*hd ||
		rel && (G != H || c.width(o.In[3]) != H*hd) {
		return nil, fmt.Errorf("a rows attention of %d operands, %d heads of %d over %d", len(o.In), H, hd, G)
	}
	F, U := ir.F32, ir.U32
	qs, ks, vs := c.rows(o.In[0]), c.rows(o.In[1]), c.rows(o.In[2])
	Q, R, D := qs[0], ks[0], H*hd
	if vs[0] != R {
		return nil, fmt.Errorf("a rows attention over %d keys and %d values", R, vs[0])
	}
	// kvAt is element e of query head h's key or value head: the one its
	// group of H/G heads reads.
	kvAt := func(x *exprs, h, e int) int {
		return x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, h, x.u(uint32(H/G))), x.u(uint32(hd))), e)
	}
	minus := math.Float32frombits(0xff7fffff)
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + "." + name, Temp: true}
	}
	dots, scores, maxes, weights, sums, num := tmp("dots"), tmp("scores"), tmp("max"), tmp("weights"), tmp("sum"), tmp("num")
	sh, qh := []int{Q, H, R}, []int{Q, H}

	var dx exprs
	{
		q, h, j, e := dx.coord(0), dx.coord(1), dx.coord(2), dx.coord(3)
		el := dx.op(OpAdd, U, dx.op(OpMul, U, h, dx.u(uint32(hd))), e)
		kel := el
		if G != H {
			kel = kvAt(&dx, h, e)
		}
		dx.op(OpMul, F, dx.at(0, qs, q, el), dx.at(1, ks, j, kel))
	}
	var relPrims []Prim
	var rels Operand
	if rel {
		// Each query row's dots with the positions its keys' offsets name:
		// position q-j, the table's row top-1-q+j.
		tb := o.Weight[0]
		to, err := c.weight(tb)
		if err != nil {
			return nil, err
		}
		if tb.K != D || tb.Rows < 2*R-1 || tb.Rows%2 == 0 {
			return nil, fmt.Errorf("a relative-position attention over %d rows by a %dx%d table over %d", R, tb.Rows, tb.K, D)
		}
		top := (tb.Rows + 1) / 2
		rels = tmp("rel")
		var rx exprs
		rin := []Operand{c.value(o.In[3])}
		rf := Facts{Extents: map[int]int{}}
		{
			q, h, j, e := rx.coord(0), rx.coord(1), rx.coord(2), rx.coord(3)
			el := rx.op(OpAdd, U, rx.op(OpMul, U, h, rx.u(uint32(hd))), e)
			row := rx.op(OpSub, U, rx.op(OpAdd, U, rx.u(uint32(top-1)), j), q)
			pv, err := weightElement(&rx, to, tb.Rows, D, &rin, &rf, row, el)
			if err != nil {
				return nil, err
			}
			rx.op(OpMul, F, rx.at(0, c.rows(o.In[3]), q, el), pv)
		}
		relPrims = []Prim{{Kind: PrimReduce, Shape: []int{Q, H, R, hd}, In: rin, Out: rels, Combine: OpAdd, Expr: rx.e, Facts: rf}}
	}
	var sx exprs
	scoreIn := []Operand{dots, TablePos.op()}
	scoreFacts := Facts{Extents: map[int]int{1: R + 1}}
	{
		j := sx.coord(2)
		live := sx.load(1, sx.u(0))
		d := sx.at(0, sh, sx.coord(0), sx.coord(1), j)
		if rel {
			scoreIn = append(scoreIn, rels)
			d = sx.op(OpAdd, F, d, sx.at(len(scoreIn)-1, sh, sx.coord(0), sx.coord(1), j))
		}
		keep := sx.op(OpMul, F, d, sx.f(float32(a.Scale)))
		if s := uint32(a.Offset); s > 0 {
			// Query row q reads the keys of its own range.
			scoreIn = append(scoreIn, TableRequest.op())
			scoreFacts.Extents[2] = RequestRows + int(s+2)*max(Q, R)
			lo, hi := requestSlot(&sx, 2, 1, s, sx.coord(0)), requestSlot(&sx, 2, 1, s+1, sx.coord(0))
			keep = sx.sel(F, sx.op(OpLess, ir.Pred, j, lo), sx.f(minus), sx.sel(F, sx.op(OpLess, ir.Pred, j, hi), keep, sx.f(minus)))
		}
		sx.sel(F, sx.op(OpLess, ir.Pred, j, live), keep, sx.f(minus))
	}
	var mx exprs
	mx.at(0, sh, mx.coord(0), mx.coord(1), mx.coord(2))
	var wx exprs
	wx.un(OpExp, F, wx.op(OpSub, F, wx.at(0, sh, wx.coord(0), wx.coord(1), wx.coord(2)), wx.at(1, qh, wx.coord(0), wx.coord(1))))
	var lx exprs
	lx.at(0, sh, lx.coord(0), lx.coord(1), lx.coord(2))
	var vx exprs
	{
		q, e, j := vx.coord(0), vx.coord(1), vx.coord(2)
		h := vx.op(OpDiv, U, e, vx.u(uint32(hd)))
		ve := e
		if G != H {
			ve = kvAt(&vx, h, vx.op(OpRem, U, e, vx.u(uint32(hd))))
		}
		vx.op(OpMul, F, vx.at(0, sh, q, h, j), vx.at(1, vs, j, ve))
	}
	var ox exprs
	{
		q, e := ox.coord(0), ox.coord(1)
		ox.op(OpDiv, F, ox.at(0, []int{Q, D}, q, e), ox.at(1, qh, q, ox.op(OpDiv, U, e, ox.u(uint32(hd)))))
	}
	return append(relPrims, []Prim{
		{Kind: PrimReduce, Shape: []int{Q, H, R, hd}, In: []Operand{c.value(o.In[0]), c.value(o.In[1])}, Out: dots, Combine: OpAdd, Expr: dx.e},
		{Kind: PrimMap, Shape: sh, In: scoreIn, Out: scores, Expr: sx.e, Facts: scoreFacts},
		{Kind: PrimReduce, Shape: sh, In: []Operand{scores}, Out: maxes, Combine: OpMax, Init: int64(math.Float32bits(minus)), Expr: mx.e},
		{Kind: PrimMap, Shape: sh, In: []Operand{scores, maxes}, Out: weights, Expr: wx.e},
		{Kind: PrimReduce, Shape: sh, In: []Operand{weights}, Out: sums, Combine: OpAdd, Expr: lx.e},
		{Kind: PrimReduce, Shape: []int{Q, D, R}, In: []Operand{weights, c.value(o.In[2])}, Out: num, Combine: OpAdd, Expr: vx.e},
		{Kind: PrimMap, Shape: []int{Q, D}, In: []Operand{num, sums}, Out: c.value(o.Out[0]), Expr: ox.e},
	}...), nil
}

// decomposeFrames is each frame's samples of the step's sound
// (graph.OpFrames): sample k = t*Stride + i - Width/2 of the clip, less
// Const times the one before it, zero past either end.
func decomposeFrames(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	sh := c.rows(o.Out[0])
	R, N := sh[0], sh[1]
	if N != a.Width || a.Stride < 1 {
		return nil, fmt.Errorf("frames of %d samples every %d into rows of %d", a.Width, a.Stride, N)
	}
	U, F := ir.U32, ir.F32
	var x exprs
	t, i := x.coord(0), x.coord(1)
	n := x.load(1, x.u(RequestPicW))
	// k+Width/2, the frame's sample on the clip, unsigned: below Width/2 is
	// before the clip.
	k := x.op(OpAdd, U, x.op(OpMul, U, t, x.u(uint32(a.Stride))), i)
	half := x.u(uint32(N / 2))
	at := x.op(OpSub, U, k, half)
	in := x.op(OpLess, ir.Pred, x.op(OpSub, U, k, half), n)
	inside := x.sel(ir.U32, x.op(OpLess, ir.Pred, k, half), x.u(0), x.sel(ir.U32, in, x.u(1), x.u(0)))
	safe := x.op(OpMin, U, at, x.op(OpMax, U, x.op(OpSub, U, n, x.u(1)), x.u(0)))
	// The samples are a sound op's where the frames read one (f32), the
	// step's picture table's words where not.
	src, words := TablePicture.op(), PictureWords
	if len(o.In) == 1 {
		src, words = c.value(o.In[0]), c.width(o.In[0])
	}
	sample := func(i int) int {
		if len(o.In) == 1 {
			return x.load(0, i)
		}
		return x.un(OpBitcast, F, x.load(0, i))
	}
	cur := sample(safe)
	prev := sample(x.op(OpSub, U, safe, x.u(1)))
	prev = x.sel(F, x.op(OpLess, ir.Pred, safe, x.u(1)), x.f(0), prev)
	v := x.op(OpSub, F, cur, x.op(OpMul, F, prev, x.f(float32(a.Const))))
	x.sel(F, x.op(OpLess, ir.Pred, x.u(0), inside), v, x.f(0))
	return []Prim{{Kind: PrimMap, Shape: []int{R, N}, In: []Operand{src, TableRequest.op()}, Out: c.value(o.Out[0]),
		Expr: x.e, Facts: Facts{Extents: map[int]int{0: words, 1: RequestPicW + 1}}}}, nil
}

// decomposeFeatureNorm is each column normalised over the step's first
// live-Offset rows (graph.OpFeatureNorm): their sum, the squares of their
// distances from its mean, then each row's value over the std, the rows
// after zero.
func decomposeFeatureNorm(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.In[0]) != c.width(o.Out[0]) {
		return nil, fmt.Errorf("a feature norm of %d operands", len(o.In))
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.In[0])
	R, W := sh[0], sh[1]
	drop := uint32(o.Attr.Offset)
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + "." + name, Temp: true}
	}
	sum, sq := tmp("sum"), tmp("sq")
	facts := Facts{Extents: map[int]int{1: R + 1}}
	// n is the rows the statistics read, at least one.
	n := func(x *exprs, pos int) int {
		live := x.load(pos, x.u(0))
		return x.op(OpMax, U, x.sel(U, x.op(OpLess, ir.Pred, live, x.u(drop+1)), x.u(0), x.op(OpSub, U, live, x.u(drop))), x.u(1))
	}
	flt := func(x *exprs, v int) int { return x.un(OpCvtF32, F, x.un(OpBitcast, ir.I32, v)) }
	var sx exprs
	{
		col, r := sx.coord(0), sx.coord(1)
		v := sx.at(0, sh, r, col)
		sx.sel(F, sx.op(OpLess, ir.Pred, r, n(&sx, 1)), v, sx.f(0))
	}
	var qx exprs
	{
		col, r := qx.coord(0), qx.coord(1)
		mean := qx.op(OpDiv, F, qx.at(2, []int{W}, col), flt(&qx, n(&qx, 1)))
		d := qx.op(OpSub, F, qx.at(0, sh, r, col), mean)
		qx.sel(F, qx.op(OpLess, ir.Pred, r, n(&qx, 1)), qx.op(OpMul, F, d, d), qx.f(0))
	}
	var ox exprs
	{
		r, col := ox.coord(0), ox.coord(1)
		cnt := n(&ox, 1)
		mean := ox.op(OpDiv, F, ox.at(2, []int{W}, col), flt(&ox, cnt))
		den := flt(&ox, ox.op(OpMax, U, ox.op(OpSub, U, cnt, ox.u(1)), ox.u(1)))
		std := ox.un(OpSqrt, F, ox.op(OpDiv, F, ox.at(3, []int{W}, col), den))
		v := ox.op(OpDiv, F, ox.op(OpSub, F, ox.at(0, sh, r, col), mean), ox.op(OpAdd, F, std, ox.f(float32(o.Attr.Eps))))
		ox.sel(F, ox.op(OpLess, ir.Pred, r, cnt), v, ox.f(0))
	}
	return []Prim{
		{Kind: PrimReduce, Shape: []int{W, R}, In: []Operand{c.value(o.In[0]), TablePos.op()}, Out: sum, Combine: OpAdd, Expr: sx.e, Facts: facts},
		{Kind: PrimReduce, Shape: []int{W, R}, In: []Operand{c.value(o.In[0]), TablePos.op(), sum}, Out: sq, Combine: OpAdd, Expr: qx.e, Facts: facts},
		{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), TablePos.op(), sum, sq}, Out: c.value(o.Out[0]), Expr: ox.e, Facts: facts},
	}, nil
}

// decomposeSplitRows is each input row's runs of Width elements as rows
// of their own (graph.OpSplitRows).
func decomposeSplitRows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a row split of %d operands", len(o.In))
	}
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	W := sh[1]
	if W < 1 || ish[1]%W != 0 || sh[0] != ish[0]*(ish[1]/W) {
		return nil, fmt.Errorf("rows of %d split into %d rows of %d", ish[1], sh[0], W)
	}
	U := ir.U32
	k := uint32(ish[1] / W)
	var x exprs
	r, e := x.coord(0), x.coord(1)
	src := x.op(OpDiv, U, r, x.u(k))
	at := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpRem, U, r, x.u(k)), x.u(uint32(W))), e)
	x.at(0, ish, src, at)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeClipMask is each row kept where its grid row is below the
// clip's valid length (graph.OpClipMask), zero after.
func decomposeClipMask(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 1 || len(o.Out) != 1 || a.Stride < 1 || c.width(o.In[0]) != c.width(o.Out[0]) {
		return nil, fmt.Errorf("a clip mask of %d operands every %d samples", len(o.In), a.Stride)
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.In[0])
	var x exprs
	r := x.coord(0)
	row := r
	if a.GridW > 0 {
		row = x.op(OpDiv, U, r, x.u(uint32(a.GridW)))
	}
	// Each halving is L-1 halved, plus one, so row < L after Span of them
	// is row*2^Span < L0 = n/Stride, that is (row*2^Span + 1)*Stride <= n:
	// no division of the sample count, which a vector of lanes has none of.
	end := x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, row, x.u(uint32(1)<<a.Span)), x.u(1)), x.u(uint32(a.Stride)))
	x.sel(F, x.op(OpLess, ir.Pred, x.load(1, x.u(RequestPicW)), end), x.f(0), x.at(0, sh, r, x.coord(1)))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), TableRequest.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{1: RequestPicW + 1}}}}, nil
}

// decomposeUpsample is a grid of rows GridW wide Span times as large each
// way (graph.OpUpsample), one map over the output's rows and width: row r
// of the grid Span*GridW wide reads the input row at (y/Span, x/Span).
func decomposeUpsample(o graph.Op, c *Context) ([]Prim, error) {
	S, W := o.Attr.Span, o.Attr.GridW
	if len(o.In) != 1 || len(o.Out) != 1 || S < 2 || W < 1 || c.width(o.Out[0]) != c.width(o.In[0]) {
		return nil, fmt.Errorf("an upsample of %d operands by %d over a grid %d wide", len(o.In), S, W)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	if ish[0]%W != 0 || sh[0] != ish[0]*S*S {
		return nil, fmt.Errorf("an upsample of %d rows %d wide into %d", ish[0], W, sh[0])
	}
	var x exprs
	r, j := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	y, col := x.op(OpDiv, U, r, u(W*S)), x.op(OpRem, U, r, u(W*S))
	p := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, y, u(S)), u(W)), x.op(OpDiv, U, col, u(S)))
	x.at(0, ish, p, j)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeAppendRow is the input's rows, then row Offset of the weight
// (graph.OpAppendRow): one map over the output's rows.
func decomposeAppendRow(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 || o.Weight[0].K != c.width(o.In[0]) ||
		o.Attr.Offset < 0 || o.Attr.Offset >= max(o.Weight[0].Rows, 1) {
		return nil, fmt.Errorf("an appended row of %d operands and %d weights", len(o.In), len(o.Weight))
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	w := o.Weight[0]
	R := ish[0]
	if sh[0] != R+1 {
		return nil, fmt.Errorf("an appended row after %d rows writes %d", R, sh[0])
	}
	var x exprs
	r, i := x.coord(0), x.coord(1)
	in := x.at(0, ish, x.op(OpMin, U, r, x.u(uint32(R-1))), i)
	row := x.at(1, []int{max(w.Rows, 1), sh[1]}, x.u(uint32(o.Attr.Offset)), i)
	x.sel(ir.F32, x.op(OpLess, ir.Pred, r, x.u(uint32(R))), in, row)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), {Kind: OperandVector, Role: w.Role, Name: w.Role.String()}},
		Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
