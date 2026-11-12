package convert

import (
	"fmt"
	"github.com/jitllm/jitllm/format/jlm"
)

// GeometryOf is the geometry of tower t's picture, from the steps its
// stater wrote: the resize each states, read in the entry's patches, a
// fixed square's grid side x side, the markers where the head writes them.
func GeometryOf(t *jlm.Tower, v *jlm.Vision) (*jlm.PictureGeometry, error) {
	ops := t.Entry.Ops
	if len(ops) == 0 || ops[0].Code != jlm.TOpPatchify || ops[0].Span < 1 || v.PatchSize == 0 {
		return nil, fmt.Errorf("convert: a picture's geometry of an entry that does not begin with its patches")
	}
	p := int(ops[0].Span)
	side := int(v.ImageSize / v.PatchSize)
	if len(t.Chain) > 0 {
		side = int(v.ImageSize) / p
	}
	marks := false
	for _, op := range t.Head.Ops {
		marks = marks || op.Code == jlm.TOpMarkRow
	}
	var fixedW, fixedH, short, minT, maxT, longest, doubled, grid int
	thumb, canvas := false, false
	for _, s := range t.Pre {
		in := func(i int) int { return int(s.Ints[i]) }
		switch s.Kind {
		case jlm.PreResize:
			fixedW, fixedH = in(0), in(1)
		case PreResizeShort:
			short = in(0)
		case PreCenterCrop:
			fixedW, fixedH = in(0), in(1)
		case PreTile:
			minT, maxT, thumb = in(0), in(1), in(2) == 1
		case PreBestFit:
			minT, maxT, thumb, canvas = 1, in(0), true, true
		case PreLongTiles:
			longest, doubled, grid = in(0), in(1), in(2)
		case PreSmartResize:
			return smartGeometry(in(0), in(1), in(2), len(s.Ints) == 6 && in(5) == 1, p), nil
		case PreNaFlex:
			return naflexGeometry(in(0), in(1), in(2)), nil
		case PreLimitPad:
			return limitGeometry(in(0), in(1), in(2), p), nil
		case PreLongBox:
			return boxGeometry(in(0), in(1), side), nil
		case PreCapUnit:
			return capGeometry(in(0), in(1), p), nil
		case PreAspectBudget:
			return budgetGeometry(in(0), in(1), in(2), p), nil
		case PreTileSmart:
			return smartTileGeometry(in(0), in(1), in(2), in(3), in(4), in(5), in(6), s.Floats[0], p), nil
		case PreSliceUHD:
			return sliceGeometry(in(0), in(1), in(2), p, marks), nil
		}
	}
	switch {
	case longest > 0 && fixedW > 0:
		if (longest+fixedW-1)/fixedW > grid {
			return nil, fmt.Errorf("convert: a longer side of %d pixels is more tiles of %d than the %d a side the processor names",
				longest, fixedW, grid)
		}
		return longTileGeometry(longest, fixedW, side, doubled, grid, marks), nil
	case canvas:
		return canvasGeometry(maxT, fixedW, side, marks), nil
	case maxT > 0:
		return tileGeometry(minT, maxT, thumb, fixedW, side), nil
	case short > 0:
		return shortCropGeometry(short, fixedW, fixedH, side), nil
	case fixedW > 0:
		return fixedGeometry(fixedW, fixedH, side), nil
	}
	return nil, fmt.Errorf("convert: a picture never resized to the tower's size")
}

// geom builds a picture's geometry (jlm.PictureGeometry): bindings in
// order, each an expression of the picture's sides and the bindings before
// it, or a choice from a table; then the squares' count and fields. Every
// constant is exact: the host evaluates them in float64, as the processors'
// Python does.
type geom struct {
	g jlm.PictureGeometry
	n uint32
}

func newGeom() *geom { return &geom{n: jlm.PicBound} }

// The picture's operands.
func pw() jlm.Arg { return jlm.Operand(jlm.PicW) }
func ph() jlm.Arg { return jlm.Operand(jlm.PicH) }
func pi() jlm.Arg { return jlm.Operand(jlm.PicSquare) }
func pn() jlm.Arg { return jlm.Operand(jlm.PicCount) }
func pf() jlm.Arg { return jlm.Operand(jlm.PicFrames) }

// k is the exact constant v.
func k(v float64) jlm.Arg { return jlm.ExactNum(v) }

// built is the expression f states, its value its last term.
func built(f func(e *sizes) jlm.Arg) jlm.Expr {
	var e sizes
	a := f(&e)
	if a.Kind != jlm.ArgTerm || int(a.Index) != len(e.T)-1 {
		e.add(a, k(0))
	}
	return e.expr()
}

// value binds the value f states and is its operand.
func (b *geom) value(f func(e *sizes) jlm.Arg) jlm.Arg {
	b.g.Bind = append(b.g.Bind, jlm.PictureBind{Value: built(f)})
	b.n++
	return jlm.Operand(b.n - 1)
}

// choose binds a row of rows by the choice's rule, its expressions given
// the row's operands; the row's operands are returned, the chosen row's
// values once bound.
func (b *geom) choose(rows [][]float64, eligible func(e *sizes, row []jlm.Arg) jlm.Arg,
	costs []func(e *sizes, row []jlm.Arg) jlm.Arg, tie func(e *sizes, row []jlm.Arg) jlm.Arg) []jlm.Arg {
	cols := len(rows[0])
	row := make([]jlm.Arg, cols)
	for j := range row {
		row[j] = jlm.Operand(b.n + uint32(j))
	}
	c := &jlm.PictureChoice{Cols: int32(cols)}
	for _, r := range rows {
		c.Rows = append(c.Rows, r...)
	}
	of := func(f func(e *sizes, row []jlm.Arg) jlm.Arg) jlm.Expr {
		if f == nil {
			return jlm.Expr{}
		}
		return built(func(e *sizes) jlm.Arg { return f(e, row) })
	}
	c.Eligible, c.Tie = of(eligible), of(tie)
	for _, f := range costs {
		c.Cost = append(c.Cost, of(f))
	}
	b.g.Bind = append(b.g.Bind, jlm.PictureBind{Choice: c})
	b.n += uint32(cols)
	return row
}

// field states square field f.
func (b *geom) field(f int, fn func(e *sizes) jlm.Arg) { b.g.Square[f] = built(fn) }

// count states the squares' count.
func (b *geom) count(fn func(e *sizes) jlm.Arg) { b.g.Count = built(fn) }

// one states a picture of one square: the resize size, the window the whole
// of it, the grid, the canvas the resize.
func (b *geom) one(ow, oh, gw, gh jlm.Arg) {
	b.count(func(e *sizes) jlm.Arg { return k(1) })
	b.whole(ow, oh, gw, gh)
}

// whole states every field of a square that is the whole resize: ow x oh,
// read as a gw x gh grid.
func (b *geom) whole(ow, oh, gw, gh jlm.Arg) {
	for f, a := range map[int]jlm.Arg{jlm.SqOw: ow, jlm.SqOh: oh, jlm.SqPw: ow, jlm.SqPh: oh, jlm.SqGw: gw, jlm.SqGh: gh,
		jlm.SqCw: ow, jlm.SqCh: oh, jlm.SqX0: k(0), jlm.SqY0: k(0), jlm.SqCx: k(0), jlm.SqCy: k(0),
		jlm.SqMark: k(0), jlm.SqMarkN: k(0), jlm.SqTrail: k(0), jlm.SqTrailN: k(0)} {
		b.field(f, func(e *sizes) jlm.Arg { return a })
	}
}

func (b *geom) done(unit, maxPix, maxPatches int) *jlm.PictureGeometry {
	b.g.Bounds = jlm.PictureBounds{Unit: int32(unit), MaxPix: int32(maxPix), MaxPatches: int32(maxPatches)}
	return &b.g
}

// Arithmetic of the builder.
func (e *sizes) floor(a jlm.Arg) jlm.Arg     { return e.fn(jlm.FnFloor, a) }
func (e *sizes) ceil(a jlm.Arg) jlm.Arg      { return e.fn(jlm.FnCeil, a) }
func (e *sizes) trunc(a jlm.Arg) jlm.Arg     { return e.fn(jlm.FnTrunc, a) }
func (e *sizes) roundEven(a jlm.Arg) jlm.Arg { return e.fn(jlm.FnRoundEven, a) }
func (e *sizes) abs(a jlm.Arg) jlm.Arg       { return e.fn(jlm.FnAbs, a) }
func (e *sizes) sqrt(a jlm.Arg) jlm.Arg      { return e.fn(jlm.FnSqrt, a) }
func (e *sizes) log(a jlm.Arg) jlm.Arg       { return e.fn(jlm.FnLog, a) }
func (e *sizes) f32(a jlm.Arg) jlm.Arg       { return e.fn(jlm.FnF32, a) }
func (e *sizes) equal(a, b jlm.Arg) jlm.Arg  { return e.Fn(jlm.FnEqual, a, b) }

// mod is a - b*floor(a/b), a whole and b positive.
func (e *sizes) mod(a, b jlm.Arg) jlm.Arg { return e.sub(a, e.mul(b, e.floor(e.div(a, b)))) }

// idiv is floor(a/b), a whole and b positive.
func (e *sizes) idiv(a, b jlm.Arg) jlm.Arg { return e.floor(e.div(a, b)) }

// fixedGeometry is a picture resized whole to pw x ph, read as a side x
// side grid.
func fixedGeometry(w, h, side int) *jlm.PictureGeometry {
	b := newGeom()
	b.one(k(float64(w)), k(float64(h)), k(float64(side)), k(float64(side)))
	return b.done(0, w*h, side*side)
}

// shortCropGeometry is a picture resized so its shorter side is short and
// its longer int(short * long / short), then its centre w x h window, read
// as a side x side grid (transformers' resize and center_crop).
func shortCropGeometry(short, w, h, side int) *jlm.PictureGeometry {
	b := newGeom()
	s := k(float64(short))
	ow := b.value(func(e *sizes) jlm.Arg { return e.where(e.less(ph(), pw()), e.trunc(e.div(e.mul(s, pw()), ph())), s) })
	oh := b.value(func(e *sizes) jlm.Arg { return e.where(e.less(ph(), pw()), s, e.trunc(e.div(e.mul(s, ph()), pw()))) })
	b.one(ow, oh, k(float64(side)), k(float64(side)))
	b.field(jlm.SqPw, func(e *sizes) jlm.Arg { return k(float64(w)) })
	b.field(jlm.SqPh, func(e *sizes) jlm.Arg { return k(float64(h)) })
	b.field(jlm.SqX0, func(e *sizes) jlm.Arg { return e.idiv(e.sub(ow, k(float64(w))), k(2)) })
	b.field(jlm.SqY0, func(e *sizes) jlm.Arg { return e.idiv(e.sub(oh, k(float64(h))), k(2)) })
	return b.done(0, w*h, side*side)
}

// smartSize binds a picture's own size under Qwen2-VL's smart_resize: where
// lift, a side shorter than factor scaled up first so that side is factor
// and each truncated; each side rounded half to even to a multiple of
// factor, at least one; then past maxPix both scaled down by sqrt(area /
// maxPix) and floored to factors, at least one, or short of minPix scaled up
// by sqrt(minPix / area) and ceiled.
func (b *geom) smartSize(w, h jlm.Arg, factor, minPix, maxPix int, lift bool) (ow, oh jlm.Arg) {
	fa := k(float64(factor))
	if lift {
		small := func(e *sizes) jlm.Arg { return e.add(e.less(w, fa), e.less(h, fa)) }
		scale := func(e *sizes) jlm.Arg { return e.max(e.div(fa, h), e.div(fa, w)) }
		lw := b.value(func(e *sizes) jlm.Arg { return e.where(small(e), e.trunc(e.mul(w, scale(e))), w) })
		lh := b.value(func(e *sizes) jlm.Arg { return e.where(small(e), e.trunc(e.mul(h, scale(e))), h) })
		w, h = lw, lh
	}
	round := func(e *sizes, v jlm.Arg) jlm.Arg { return e.mul(e.max(e.roundEven(e.div(v, fa)), k(1)), fa) }
	rw := b.value(func(e *sizes) jlm.Arg { return round(e, w) })
	rh := b.value(func(e *sizes) jlm.Arg { return round(e, h) })
	// Side v of the picture, r its rounding.
	side := func(v, r jlm.Arg) func(e *sizes) jlm.Arg {
		return func(e *sizes) jlm.Arg {
			area := e.mul(w, h)
			pix := e.mul(rw, rh)
			down := e.sqrt(e.div(area, k(float64(maxPix))))
			up := e.sqrt(e.div(k(float64(minPix)), area))
			shrunk := e.max(fa, e.mul(e.floor(e.div(e.div(v, down), fa)), fa))
			grown := e.mul(e.ceil(e.div(e.mul(v, up), fa)), fa)
			return e.where(e.less(k(float64(maxPix)), pix), shrunk, e.where(e.less(pix, k(float64(minPix))), grown, r))
		}
	}
	return b.value(side(w, rw)), b.value(side(h, rh))
}

// FramesOf is the geometry of video v's frames where it reads them through
// an entry of its own: one square, the size its resize (PreVideoResize)
// gives frames of w x h sized together in PicFrames, read in the entry's
// patches; none where the processor refuses them.
func FramesOf(v *jlm.Video) (*jlm.PictureGeometry, error) {
	if len(v.Entry.Ops) == 0 || v.Entry.Ops[0].Code != jlm.TOpPatchify || v.Entry.Ops[0].Span < 1 {
		return nil, fmt.Errorf("convert: a video's frames' geometry of an entry that does not begin with their patches")
	}
	for _, s := range v.Pre {
		if s.Kind != PreVideoResize {
			continue
		}
		if len(s.Ints) != 6 && len(s.Ints) != 7 || s.Ints[0] <= 0 || s.Ints[5] < 1 {
			return nil, fmt.Errorf("convert: a video resized with %v", s.Ints)
		}
		in := func(i int) int { return int(s.Ints[i]) }
		alone := len(s.Ints) == 7 && s.Ints[6] == 1
		return videoGeometry(in(0), in(1), in(2), in(5), alone, int(v.Entry.Ops[0].Span)), nil
	}
	return nil, fmt.Errorf("convert: a video's frames never resized")
}

// videoGeometry is Qwen's video smart_resize (PreVideoResize) of frames
// sized together in PicFrames, or each alone, in multiples of factor; a
// frame past an aspect of 200, or fewer frames than per sized together,
// becomes no square.
func videoGeometry(factor, minPix, maxPix, per int, alone bool, patch int) *jlm.PictureGeometry {
	b := newGeom()
	fa := k(float64(factor))
	w, h, n, t := pw(), ph(), pf(), k(float64(per))
	if alone {
		n, t = k(1), k(1)
	} else {
		small := func(e *sizes) jlm.Arg { return e.add(e.less(w, fa), e.less(h, fa)) }
		scale := func(e *sizes) jlm.Arg { return e.max(e.div(fa, h), e.div(fa, w)) }
		lw := b.value(func(e *sizes) jlm.Arg { return e.where(small(e), e.trunc(e.mul(w, scale(e))), w) })
		lh := b.value(func(e *sizes) jlm.Arg { return e.where(small(e), e.trunc(e.mul(h, scale(e))), h) })
		w, h = lw, lh
	}
	wb := b.value(func(e *sizes) jlm.Arg { return e.mul(e.roundEven(e.div(w, fa)), fa) })
	hb := b.value(func(e *sizes) jlm.Arg { return e.mul(e.roundEven(e.div(h, fa)), fa) })
	tb := b.value(func(e *sizes) jlm.Arg { return e.mul(e.roundEven(e.div(n, t)), t) })
	side := func(v, r jlm.Arg) func(e *sizes) jlm.Arg {
		return func(e *sizes) jlm.Arg {
			area := e.mul(n, e.mul(w, h))
			pix := e.mul(tb, e.mul(wb, hb))
			down := e.sqrt(e.div(area, k(float64(maxPix))))
			up := e.sqrt(e.div(k(float64(minPix)), area))
			shrunk := e.max(fa, e.mul(e.floor(e.div(e.div(v, down), fa)), fa))
			grown := e.mul(e.ceil(e.div(e.mul(v, up), fa)), fa)
			return e.where(e.less(k(float64(maxPix)), pix), shrunk, e.where(e.less(pix, k(float64(minPix))), grown, r))
		}
	}
	ow, oh := b.value(side(w, wb)), b.value(side(h, hb))
	b.count(func(e *sizes) jlm.Arg {
		long := e.less(k(200), e.div(e.max(w, h), e.min(w, h)))
		few := k(0)
		if !alone {
			few = e.less(pf(), t)
		}
		return e.where(e.add(long, few), k(0), k(1))
	})
	p := k(float64(patch))
	gw := b.value(func(e *sizes) jlm.Arg { return e.idiv(ow, p) })
	gh := b.value(func(e *sizes) jlm.Arg { return e.idiv(oh, p) })
	b.whole(ow, oh, gw, gh)
	most := maxPix
	if !alone {
		most = maxPix / per
	}
	return b.done(factor, most, most/(patch*patch))
}

// smartGeometry is a picture at its own size under smart_resize (smartSize),
// read in patches of patch pixels.
func smartGeometry(factor, minPix, maxPix int, lift bool, patch int) *jlm.PictureGeometry {
	b := newGeom()
	ow, oh := b.smartSize(pw(), ph(), factor, minPix, maxPix, lift)
	b.ownSize(ow, oh, patch)
	return b.done(factor, maxPix, maxPix/(patch*patch))
}

// ownSize states a picture of one square at its own size ow x oh, read in
// patches of patch pixels.
func (b *geom) ownSize(ow, oh jlm.Arg, patch int) {
	p := k(float64(patch))
	gw := b.value(func(e *sizes) jlm.Arg { return e.idiv(ow, p) })
	gh := b.value(func(e *sizes) jlm.Arg { return e.idiv(oh, p) })
	b.one(ow, oh, gw, gh)
}

// naflexGeometry is a picture at SigLIP2's NaFlex size: its own whole
// patches' count clamped to [lo, hi], then get_image_size_for_max_num_patches'
// bisection for the largest scale whose sides, ceiled to whole patches and
// at least one, hold that many -- its halvings unrolled, each a pair of
// bounds.
func naflexGeometry(patch, lo, hi int) *jlm.PictureGeometry {
	b := newGeom()
	p := k(float64(patch))
	target := b.value(func(e *sizes) jlm.Arg {
		n := e.max(e.mul(e.idiv(pw(), p), e.idiv(ph(), p)), k(1))
		return e.min(e.max(n, k(float64(lo))), k(float64(hi)))
	})
	scaled := func(e *sizes, scale, size jlm.Arg) jlm.Arg {
		return e.max(p, e.mul(e.ceil(e.div(e.mul(size, scale), p)), p))
	}
	const eps = 1e-5
	smin, smax := k(eps/10), k(100)
	for l := 100 - eps/10; l >= eps; l /= 2 {
		lo, hi := smin, smax
		over := func(e *sizes) jlm.Arg {
			sc := e.div(e.add(lo, hi), k(2))
			th, tw := scaled(e, sc, ph()), scaled(e, sc, pw())
			return e.less(target, e.mul(e.div(th, p), e.div(tw, p)))
		}
		mid := func(e *sizes) jlm.Arg { return e.div(e.add(lo, hi), k(2)) }
		smin = b.value(func(e *sizes) jlm.Arg { return e.where(over(e), lo, mid(e)) })
		smax = b.value(func(e *sizes) jlm.Arg { return e.where(over(e), mid(e), hi) })
	}
	ow := b.value(func(e *sizes) jlm.Arg { return scaled(e, smin, pw()) })
	oh := b.value(func(e *sizes) jlm.Arg { return scaled(e, smin, ph()) })
	b.ownSize(ow, oh, patch)
	return b.done(patch, hi*patch*patch, hi)
}

// limitGeometry is a picture kept at its size unless its whole limitPatch
// patches pass limit, where both sides are scaled by sqrt(limit / patches)
// and truncated; read over its grid padded to multiples of pad pixels
// (Kimi-VL's rescale with pad_input). Its squares reach the most patches a
// picture pads to: a side of a whole patches pads to at most a over the
// pad's patches plus one of them, and its other side has at most limit/a
// (any up to the widest picture where a is zero).
func limitGeometry(limitPatch, limit, pad, patch int) *jlm.PictureGeometry {
	b := newGeom()
	lp := k(float64(limitPatch))
	scale := func(e *sizes) jlm.Arg {
		n := e.mul(e.idiv(pw(), lp), e.idiv(ph(), lp))
		return e.where(e.less(k(float64(limit)), n), e.sqrt(e.div(k(float64(limit)), n)), k(1))
	}
	ow := b.value(func(e *sizes) jlm.Arg { return e.trunc(e.mul(pw(), scale(e))) })
	oh := b.value(func(e *sizes) jlm.Arg { return e.trunc(e.mul(ph(), scale(e))) })
	padded := func(v jlm.Arg) func(e *sizes) jlm.Arg {
		return func(e *sizes) jlm.Arg {
			return e.idiv(e.mul(e.ceil(e.div(v, k(float64(pad)))), k(float64(pad))), k(float64(patch)))
		}
	}
	gw, gh := b.value(padded(ow)), b.value(padded(oh))
	b.one(ow, oh, gw, gh)
	b.field(jlm.SqCw, func(e *sizes) jlm.Arg { return e.mul(gw, k(float64(patch))) })
	b.field(jlm.SqCh, func(e *sizes) jlm.Arg { return e.mul(gh, k(float64(patch))) })
	u, most, maxPatches := pad/patch, widestPicture/patch, 0
	for a := 0; a <= most; a++ {
		other := most
		if a > 0 {
			other = min(most, limit/a)
		}
		maxPatches = max(maxPatches, (a/u+1)*u*((other/u+1)*u))
	}
	return b.done(pad, limit*limitPatch*limitPatch, maxPatches)
}

// boxGeometry is a picture resized so its longer side is box pixels and its
// shorter round-half-even(side * box / longer), at least boxMin, laid on the
// box (Janus's resize and pad_to_square), read as a side x side grid.
func boxGeometry(box, boxMin, side int) *jlm.PictureGeometry {
	b := newGeom()
	bx := k(float64(box))
	d := func(e *sizes) jlm.Arg { return e.div(bx, e.max(pw(), ph())) }
	ow := b.value(func(e *sizes) jlm.Arg { return e.max(e.roundEven(e.mul(pw(), d(e))), k(float64(boxMin))) })
	oh := b.value(func(e *sizes) jlm.Arg { return e.max(e.roundEven(e.mul(ph(), d(e))), k(float64(boxMin))) })
	b.one(ow, oh, k(float64(side)), k(float64(side)))
	b.field(jlm.SqCw, func(e *sizes) jlm.Arg { return bx })
	b.field(jlm.SqCh, func(e *sizes) jlm.Arg { return bx })
	b.field(jlm.SqCx, func(e *sizes) jlm.Arg { return e.idiv(e.sub(bx, ow), k(2)) })
	b.field(jlm.SqCy, func(e *sizes) jlm.Arg { return e.idiv(e.sub(bx, oh), k(2)) })
	return b.done(0, box*box, side*side)
}

// capGeometry is a picture whose longer side, past capSide, is brought to it
// -- both sides divided by longer / capSide and floored -- then each side
// rounded up to a whole unit (Pixtral's resize), read in patch-pixel patches.
func capGeometry(capSide, unit, patch int) *jlm.PictureGeometry {
	b := newGeom()
	r := func(e *sizes) jlm.Arg { return e.div(e.max(pw(), ph()), k(float64(capSide))) }
	side := func(v jlm.Arg) func(e *sizes) jlm.Arg {
		return func(e *sizes) jlm.Arg {
			s := e.where(e.less(k(1), r(e)), e.floor(e.div(v, r(e))), v)
			return e.mul(e.add(e.idiv(e.sub(e.max(s, k(1)), k(1)), k(float64(unit))), k(1)), k(float64(unit)))
		}
	}
	ow, oh := b.value(side(pw())), b.value(side(ph()))
	b.ownSize(ow, oh, patch)
	top := (capSide + unit - 1) / unit * unit
	return b.done(unit, top*top, top*top/(patch*patch))
}

// budgetGeometry is the largest size of the picture's aspect within budget
// patches of budgetPatch pixels, each side whole units: both scaled by
// sqrt(budget area / area) and floored to the unit; a side that floors to
// nothing one unit, the other the aspect's whole multiple of it, at most the
// budget's units a row (Gemma 4's get_aspect_ratio_preserving_size).
func budgetGeometry(budget, budgetPatch, unit, patch int) *jlm.PictureGeometry {
	b := newGeom()
	u := k(float64(unit))
	s := b.value(func(e *sizes) jlm.Arg {
		return e.sqrt(e.div(k(float64(budget*budgetPatch*budgetPatch)), e.mul(ph(), pw())))
	})
	th := b.value(func(e *sizes) jlm.Arg { return e.mul(e.floor(e.div(e.mul(s, ph()), u)), u) })
	tw := b.value(func(e *sizes) jlm.Arg { return e.mul(e.floor(e.div(e.mul(s, pw()), u)), u) })
	g := unit / budgetPatch
	most := k(float64((budget / (g * g)) * unit))
	zero := func(e *sizes, v jlm.Arg) jlm.Arg { return e.equal(v, k(0)) }
	ow := b.value(func(e *sizes) jlm.Arg {
		both := e.mul(zero(e, th), zero(e, tw))
		onlyH := e.mul(zero(e, th), e.sub(k(1), zero(e, tw)))
		onlyW := e.mul(zero(e, tw), e.sub(k(1), zero(e, th)))
		fromH := e.min(e.mul(e.floor(e.div(pw(), ph())), u), most)
		return e.where(both, u, e.where(onlyH, fromH, e.where(onlyW, u, tw)))
	})
	oh := b.value(func(e *sizes) jlm.Arg {
		both := e.mul(zero(e, th), zero(e, tw))
		onlyH := e.mul(zero(e, th), e.sub(k(1), zero(e, tw)))
		onlyW := e.mul(zero(e, tw), e.sub(k(1), zero(e, th)))
		fromW := e.min(e.mul(e.floor(e.div(ph(), pw())), u), most)
		return e.where(both, u, e.where(onlyH, u, e.where(onlyW, fromW, th)))
	})
	b.ownSize(ow, oh, patch)
	return b.done(unit, budget*budgetPatch*budgetPatch, budget*budgetPatch*budgetPatch/(patch*patch))
}

// grids is every columns x rows grid of minT to maxT tiles in the order the
// tilers walk them: by tile count, columns ascending (InternVL's
// target_ratios, stably sorted).
func grids(minT, maxT int) [][]float64 {
	var out [][]float64
	for n := minT; n <= maxT; n++ {
		for c := 1; c <= maxT; c++ {
			if n%c == 0 {
				out = append(out, []float64{float64(c), float64(n / c)})
			}
		}
	}
	return out
}

// closestGrid binds the grid of grids whose aspect is closest to the
// picture's, a tie to the later grid while the picture's area is more than
// half of it in s-pixel tiles (find_closest_aspect_ratio): its columns and
// rows.
func (b *geom) closestGrid(minT, maxT, s int) (cols, rows jlm.Arg) {
	g := b.choose(grids(minT, maxT), nil,
		[]func(e *sizes, row []jlm.Arg) jlm.Arg{func(e *sizes, row []jlm.Arg) jlm.Arg {
			return e.abs(e.sub(e.div(pw(), ph()), e.div(row[0], row[1])))
		}},
		func(e *sizes, row []jlm.Arg) jlm.Arg {
			return e.sub(e.mul(pw(), ph()), e.mul(k(0.5*float64(s*s)), e.mul(row[0], row[1])))
		})
	return g[0], g[1]
}

// tileGeometry is a picture cut into the grid of minT to maxT s-pixel tiles
// closest to its aspect (closestGrid), resized onto it and cut row-major,
// then, where thumb and the grid is more than one tile, the whole resized to
// one tile; each read as a side x side grid (InternVL's dynamic_preprocess).
func tileGeometry(minT, maxT int, thumb bool, s, side int) *jlm.PictureGeometry {
	b := newGeom()
	cols, rows := b.closestGrid(minT, maxT, s)
	b.tiles(cols, rows, thumb, s, side, func(e *sizes) jlm.Arg { return e.mul(cols, k(float64(s))) },
		func(e *sizes) jlm.Arg { return e.mul(rows, k(float64(s))) })
	return b.done(0, s*s, side*side)
}

// longTileGeometry is Idefics3's cut (SmolVLM's): the picture's longer side
// resized to longest pixels and its shorter at the aspect, truncated, a
// whole even number, at least one (_resize_output_size_rescale_to_max_len);
// then the longer side up to whole s-pixel tiles and the shorter at that
// resize's aspect, truncated and up to whole tiles
// (resize_for_vision_encoder); cut row-major, then for more than one tile
// the whole resized to one tile (split_images); each read as a side x side
// grid. With marks, the marker runs PreLongTiles states, the double
// newline doubled rows of the table and the tiles' pairs a grid-square
// grid's.
func longTileGeometry(longest, s, side, doubled, grid int, marks bool) *jlm.PictureGeometry {
	b := newGeom()
	l, sz := k(float64(longest)), k(float64(s))
	atLeast := func(e *sizes, a, c jlm.Arg) jlm.Arg { return e.sub(k(1), e.less(a, c)) }
	even := func(e *sizes, v jlm.Arg) jlm.Arg { return e.max(e.add(v, e.mod(v, k(2))), k(1)) }
	wide := b.value(func(e *sizes) jlm.Arg { return atLeast(e, pw(), ph()) })
	w1 := b.value(func(e *sizes) jlm.Arg { return e.where(wide, l, even(e, e.trunc(e.mul(l, e.div(pw(), ph()))))) })
	h1 := b.value(func(e *sizes) jlm.Arg { return e.where(wide, even(e, e.trunc(e.div(l, e.div(pw(), ph())))), l) })
	up := func(e *sizes, v jlm.Arg) jlm.Arg { return e.mul(e.ceil(e.div(v, sz)), sz) }
	wide1 := b.value(func(e *sizes) jlm.Arg { return atLeast(e, w1, h1) })
	w2 := b.value(func(e *sizes) jlm.Arg {
		return e.where(wide1, up(e, w1), up(e, e.trunc(e.mul(up(e, h1), e.div(w1, h1)))))
	})
	h2 := b.value(func(e *sizes) jlm.Arg {
		return e.where(wide1, up(e, e.trunc(e.div(up(e, w1), e.div(w1, h1)))), up(e, h1))
	})
	cols := b.value(func(e *sizes) jlm.Arg { return e.div(w2, sz) })
	rows := b.value(func(e *sizes) jlm.Arg { return e.div(h2, sz) })
	n := b.value(func(e *sizes) jlm.Arg { return e.mul(cols, rows) })
	b.tiles(cols, rows, true, s, side, func(e *sizes) jlm.Arg { return w2 }, func(e *sizes) jlm.Arg { return h2 })
	if marks {
		span := k(float64(grid))
		fake, pairs := k(float64(2+doubled)), float64(4+doubled)
		one := func(e *sizes) jlm.Arg { return e.equal(n, k(1)) }
		tile := func(e *sizes) jlm.Arg { return e.mul(e.less(k(1), n), e.less(pi(), n)) }
		global := func(e *sizes) jlm.Arg { return e.mul(e.less(k(1), n), e.equal(pi(), n)) }
		b.field(jlm.SqMark, func(e *sizes) jlm.Arg {
			place := e.add(e.mul(e.idiv(pi(), cols), span), e.mod(pi(), cols))
			return e.where(tile(e), e.add(k(pairs), e.mul(k(2), place)), e.where(global(e), k(2), fake))
		})
		b.field(jlm.SqMarkN, func(e *sizes) jlm.Arg {
			return e.where(one(e), k(2), e.where(global(e), k(float64(doubled+2)), k(2)))
		})
		rowEnd := func(e *sizes) jlm.Arg {
			return e.mul(e.equal(e.mod(pi(), cols), e.sub(cols, k(1))), e.less(e.idiv(pi(), cols), e.sub(rows, k(1))))
		}
		b.field(jlm.SqTrail, func(e *sizes) jlm.Arg { return e.where(tile(e), rowEnd(e), fake) })
		b.field(jlm.SqTrailN, func(e *sizes) jlm.Arg { return e.where(tile(e), rowEnd(e), k(1)) })
	}
	return b.done(0, s*s, side*side)
}

// tiles states cols x rows tiles of s pixels cut from the picture resized to
// ow x oh, then a thumbnail of the whole where thumb and there is more than
// one, each read as a side x side grid.
func (b *geom) tiles(cols, rows jlm.Arg, thumb bool, s, side int, ow, oh func(e *sizes) jlm.Arg) {
	sz := k(float64(s))
	n := b.value(func(e *sizes) jlm.Arg { return e.mul(cols, rows) })
	b.count(func(e *sizes) jlm.Arg {
		if !thumb {
			return n
		}
		return e.add(n, e.less(k(1), n))
	})
	tile := func(e *sizes) jlm.Arg { return e.less(pi(), n) }
	for f, v := range map[int]func(e *sizes) jlm.Arg{
		jlm.SqOw: func(e *sizes) jlm.Arg { return e.where(tile(e), ow(e), sz) },
		jlm.SqOh: func(e *sizes) jlm.Arg { return e.where(tile(e), oh(e), sz) },
		jlm.SqX0: func(e *sizes) jlm.Arg { return e.where(tile(e), e.mul(e.mod(pi(), cols), sz), k(0)) },
		jlm.SqY0: func(e *sizes) jlm.Arg { return e.where(tile(e), e.mul(e.idiv(pi(), cols), sz), k(0)) },
		jlm.SqPw: func(e *sizes) jlm.Arg { return sz }, jlm.SqPh: func(e *sizes) jlm.Arg { return sz },
		jlm.SqGw: func(e *sizes) jlm.Arg { return k(float64(side)) }, jlm.SqGh: func(e *sizes) jlm.Arg { return k(float64(side)) },
		jlm.SqCw: func(e *sizes) jlm.Arg { return sz }, jlm.SqCh: func(e *sizes) jlm.Arg { return sz },
		jlm.SqCx: func(e *sizes) jlm.Arg { return k(0) }, jlm.SqCy: func(e *sizes) jlm.Arg { return k(0) },
		jlm.SqMark: func(e *sizes) jlm.Arg { return k(0) }, jlm.SqMarkN: func(e *sizes) jlm.Arg { return k(0) },
		jlm.SqTrail: func(e *sizes) jlm.Arg { return k(0) }, jlm.SqTrailN: func(e *sizes) jlm.Arg { return k(0) },
	} {
		b.field(f, v)
	}
}

// canvases is every rows x columns canvas of at most most tiles in the
// order Llama 4's processor walks them: tile counts from most down, each
// count's factor pairs by rows ascending, grouped by aspect in the order
// each first appears.
func canvases(most int) [][]float64 {
	var ratios []float64
	byRatio := map[float64][][]float64{}
	for n := most; n >= 1; n-- {
		for r := 1; r <= n; r++ {
			if n%r != 0 {
				continue
			}
			q := float64(r) / float64(n/r)
			if _, ok := byRatio[q]; !ok {
				ratios = append(ratios, q)
			}
			byRatio[q] = append(byRatio[q], []float64{float64(r), float64(n / r)})
		}
	}
	var out [][]float64
	for _, q := range ratios {
		out = append(out, byRatio[q]...)
	}
	return out
}

// canvasGeometry is a picture on the canvas of at most most s-pixel tiles
// whose limiting scale is the smallest upscale, or failing any the largest
// downscale, the smallest area breaking a tie (the scales float32, as the
// reference divides two integer tensors): resized onto it without
// distortion at its top left and cut row-major, then, for more than one
// tile, the whole resized to one tile; each read as a side x side grid; with
// marks, the canvas's marker runs (Llama 4's).
func canvasGeometry(most, s, side int, marks bool) *jlm.PictureGeometry {
	b := newGeom()
	sz := k(float64(s))
	scale := func(e *sizes, row []jlm.Arg) jlm.Arg {
		sw := e.f32(e.div(e.mul(row[1], sz), pw()))
		sh := e.f32(e.div(e.mul(row[0], sz), ph()))
		return e.min(sh, sw)
	}
	g := b.choose(canvases(most), nil, []func(e *sizes, row []jlm.Arg) jlm.Arg{
		func(e *sizes, row []jlm.Arg) jlm.Arg { return e.less(scale(e, row), k(1)) },
		func(e *sizes, row []jlm.Arg) jlm.Arg {
			v := scale(e, row)
			return e.where(e.less(v, k(1)), e.sub(k(0), v), v)
		},
		func(e *sizes, row []jlm.Arg) jlm.Arg { return e.mul(row[0], row[1]) },
	}, nil)
	rows, cols := g[0], g[1]
	// canvasSize: the largest size of the picture's aspect within the canvas
	// and its own size raised to one tile, the shorter scale's side floored.
	th := b.value(func(e *sizes) jlm.Arg { return e.min(e.max(ph(), sz), e.mul(rows, sz)) })
	tw := b.value(func(e *sizes) jlm.Arg { return e.min(e.max(pw(), sz), e.mul(cols, sz)) })
	wide := func(e *sizes) jlm.Arg { return e.less(e.div(tw, pw()), e.div(th, ph())) }
	ow := b.value(func(e *sizes) jlm.Arg {
		return e.max(e.where(wide(e), tw, e.min(e.floor(e.mul(pw(), e.div(th, ph()))), tw)), k(1))
	})
	oh := b.value(func(e *sizes) jlm.Arg {
		return e.max(e.where(wide(e), e.min(e.floor(e.mul(ph(), e.div(tw, pw()))), th), th), k(1))
	})
	b.tiles(cols, rows, true, s, side, func(e *sizes) jlm.Arg { return ow }, func(e *sizes) jlm.Arg { return oh })
	if marks {
		last := func(e *sizes) jlm.Arg { return e.equal(pi(), e.sub(pn(), k(1))) }
		inRow := func(e *sizes) jlm.Arg { return e.less(e.mod(pi(), cols), e.sub(cols, k(1))) }
		b.field(jlm.SqMark, func(e *sizes) jlm.Arg { return e.mul(last(e), k(3)) })
		b.field(jlm.SqMarkN, func(e *sizes) jlm.Arg { return last(e) })
		b.field(jlm.SqTrail, func(e *sizes) jlm.Arg {
			return e.mul(e.sub(k(1), last(e)), e.where(inRow(e), k(1), k(2)))
		})
		b.field(jlm.SqTrailN, func(e *sizes) jlm.Arg { return e.sub(k(1), last(e)) })
	}
	return b.done(0, s*s, side*side)
}

// smartTileGeometry is LFM2-VL's: the picture at its own size (smartSize
// over unit, minPix and maxPix, no lift); where its sides, rounded half to
// even to units and at least a patch each, hold more than tolerance times
// maxPix pixels, cut into the grid of minT to maxT s-pixel tiles closest to
// its aspect, resized onto it and cut row-major, each tile marked by its
// place (rows by span), then the whole at its own size as a thumbnail
// marked after them; else the one piece at its own size.
func smartTileGeometry(s, minT, maxT, smartPatch, unit, minPix, maxPix int, tolerance float32, patch int) *jlm.PictureGeometry {
	b := newGeom()
	ow, oh := b.smartSize(pw(), ph(), unit, minPix, maxPix, false)
	u := k(float64(unit))
	round := func(e *sizes, v jlm.Arg) jlm.Arg {
		return e.max(k(float64(smartPatch)), e.mul(e.roundEven(e.div(v, u)), u))
	}
	tiled := b.value(func(e *sizes) jlm.Arg {
		return e.less(e.mul(k(float64(maxPix)), k(float64(tolerance))), e.mul(round(e, ph()), round(e, pw())))
	})
	gc, gr := b.closestGrid(minT, maxT, s)
	cols := b.value(func(e *sizes) jlm.Arg { return e.where(tiled, gc, k(1)) })
	rows := b.value(func(e *sizes) jlm.Arg { return e.where(tiled, gr, k(1)) })
	n := b.value(func(e *sizes) jlm.Arg { return e.where(tiled, e.add(e.mul(cols, rows), k(1)), k(1)) })
	b.count(func(e *sizes) jlm.Arg { return n })
	sz, p := k(float64(s)), k(float64(patch))
	tile := func(e *sizes) jlm.Arg { return e.less(pi(), e.sub(n, k(1))) }
	span := k(float64(maxT))
	for f, v := range map[int]func(e *sizes) jlm.Arg{
		jlm.SqOw: func(e *sizes) jlm.Arg { return e.where(tile(e), e.mul(cols, sz), ow) },
		jlm.SqOh: func(e *sizes) jlm.Arg { return e.where(tile(e), e.mul(rows, sz), oh) },
		jlm.SqX0: func(e *sizes) jlm.Arg { return e.where(tile(e), e.mul(e.mod(pi(), cols), sz), k(0)) },
		jlm.SqY0: func(e *sizes) jlm.Arg { return e.where(tile(e), e.mul(e.idiv(pi(), cols), sz), k(0)) },
		jlm.SqPw: func(e *sizes) jlm.Arg { return e.where(tile(e), sz, ow) },
		jlm.SqPh: func(e *sizes) jlm.Arg { return e.where(tile(e), sz, oh) },
		jlm.SqGw: func(e *sizes) jlm.Arg { return e.idiv(e.where(tile(e), sz, ow), p) },
		jlm.SqGh: func(e *sizes) jlm.Arg { return e.idiv(e.where(tile(e), sz, oh), p) },
		jlm.SqCw: func(e *sizes) jlm.Arg { return e.where(tile(e), sz, ow) },
		jlm.SqCh: func(e *sizes) jlm.Arg { return e.where(tile(e), sz, oh) },
		jlm.SqCx: func(e *sizes) jlm.Arg { return k(0) }, jlm.SqCy: func(e *sizes) jlm.Arg { return k(0) },
		jlm.SqMark: func(e *sizes) jlm.Arg {
			place := e.add(e.add(e.mul(e.idiv(pi(), cols), span), e.mod(pi(), cols)), k(1))
			return e.where(tile(e), place, e.mul(tiled, e.add(e.mul(span, span), k(1))))
		},
		jlm.SqMarkN:  func(e *sizes) jlm.Arg { return e.where(tile(e), k(1), tiled) },
		jlm.SqTrail:  func(e *sizes) jlm.Arg { return k(0) },
		jlm.SqTrailN: func(e *sizes) jlm.Arg { return k(0) },
	} {
		b.field(f, v)
	}
	most := max(maxPix, s*s)
	return b.done(unit, most, most/(patch*patch))
}

// sliceGeometry is llava-uhd's cut (MiniCPM-V's): an overview at its own
// size -- slice^2 pixels at the picture's aspect, each side whole
// slicePatch-pixel patches rounded half to even, at least one -- then,
// where the picture's area over slice^2 rounds up past one, the slices of
// the grid of at most most whose aspect is closest to the picture's (the
// counts one either side of that multiple, each count's factor pairs in
// order, the first of the least error): the picture resized to whole
// best-sized slices and cut row-major, the overview upscaled only where
// nothing is cut. With marks, <image> and </image> about the overview,
// <slice> and </slice> about a slice, a slice row's first after the first
// taking the last two before it.
func sliceGeometry(slice, slicePatch, most, patch int, marks bool) *jlm.PictureGeometry {
	b := newGeom()
	sp := k(float64(slicePatch))
	ss := k(float64(slice * slice))
	multiple := b.value(func(e *sizes) jlm.Arg { return e.min(e.ceil(e.div(e.mul(pw(), ph()), ss)), k(float64(most))) })
	cut := b.value(func(e *sizes) jlm.Arg { return e.less(k(1), multiple) })
	var table [][]float64
	for n := 2; n <= most; n++ {
		for m := 1; m <= n; m++ {
			if n%m == 0 {
				table = append(table, []float64{float64(m), float64(n / m), float64(n)})
			}
		}
	}
	g := b.choose(table, func(e *sizes, row []jlm.Arg) jlm.Arg {
		return e.less(e.abs(e.sub(row[2], multiple)), k(1.5))
	}, []func(e *sizes, row []jlm.Arg) jlm.Arg{func(e *sizes, row []jlm.Arg) jlm.Arg {
		return e.abs(e.sub(e.log(e.div(pw(), ph())), e.log(e.div(row[0], row[1]))))
	}}, nil)
	cols := b.value(func(e *sizes) jlm.Arg { return e.where(cut, g[0], k(1)) })
	rows := b.value(func(e *sizes) jlm.Arg { return e.where(cut, g[1], k(1)) })
	units := func(e *sizes, v, u jlm.Arg) jlm.Arg { return e.max(e.mul(e.roundEven(e.div(v, u)), u), u) }
	// sliceSize of w x h: past slice^2, or where up, slice^2 at its aspect
	// (the height truncated, the width the height times the aspect,
	// truncated), then each side whole patches.
	best := func(w, h jlm.Arg, up func(e *sizes) jlm.Arg) (jlm.Arg, jlm.Arg) {
		big := func(e *sizes) jlm.Arg { return e.add(e.less(ss, e.mul(w, h)), up(e)) }
		nh := func(e *sizes) jlm.Arg { return e.trunc(e.div(k(float64(slice)), e.sqrt(e.div(w, h)))) }
		bh := b.value(func(e *sizes) jlm.Arg { return units(e, e.where(big(e), nh(e), h), sp) })
		bw := b.value(func(e *sizes) jlm.Arg { return units(e, e.where(big(e), e.trunc(e.mul(nh(e), e.div(w, h))), w), sp) })
		return bw, bh
	}
	ow, oh := best(pw(), ph(), func(e *sizes) jlm.Arg { return e.sub(k(1), cut) })
	rw0 := b.value(func(e *sizes) jlm.Arg { return units(e, pw(), cols) })
	rh0 := b.value(func(e *sizes) jlm.Arg { return units(e, ph(), rows) })
	pw0 := b.value(func(e *sizes) jlm.Arg { return e.div(rw0, cols) })
	ph0 := b.value(func(e *sizes) jlm.Arg { return e.div(rh0, rows) })
	sw, sh := best(pw0, ph0, func(e *sizes) jlm.Arg { return k(1) })
	b.count(func(e *sizes) jlm.Arg { return e.add(k(1), e.mul(cut, e.mul(cols, rows))) })
	over := func(e *sizes) jlm.Arg { return e.equal(pi(), k(0)) }
	kk := func(e *sizes) jlm.Arg { return e.sub(pi(), k(1)) }
	p := k(float64(patch))
	for f, v := range map[int]func(e *sizes) jlm.Arg{
		jlm.SqOw: func(e *sizes) jlm.Arg { return e.where(over(e), ow, e.mul(sw, cols)) },
		jlm.SqOh: func(e *sizes) jlm.Arg { return e.where(over(e), oh, e.mul(sh, rows)) },
		jlm.SqX0: func(e *sizes) jlm.Arg { return e.where(over(e), k(0), e.mul(e.mod(kk(e), cols), sw)) },
		jlm.SqY0: func(e *sizes) jlm.Arg { return e.where(over(e), k(0), e.mul(e.idiv(kk(e), cols), sh)) },
		jlm.SqPw: func(e *sizes) jlm.Arg { return e.where(over(e), ow, sw) },
		jlm.SqPh: func(e *sizes) jlm.Arg { return e.where(over(e), oh, sh) },
		jlm.SqGw: func(e *sizes) jlm.Arg { return e.idiv(e.where(over(e), ow, sw), p) },
		jlm.SqGh: func(e *sizes) jlm.Arg { return e.idiv(e.where(over(e), oh, sh), p) },
		jlm.SqCw: func(e *sizes) jlm.Arg { return e.where(over(e), ow, sw) },
		jlm.SqCh: func(e *sizes) jlm.Arg { return e.where(over(e), oh, sh) },
		jlm.SqCx: func(e *sizes) jlm.Arg { return k(0) }, jlm.SqCy: func(e *sizes) jlm.Arg { return k(0) },
		jlm.SqMark: func(e *sizes) jlm.Arg { return k(0) }, jlm.SqMarkN: func(e *sizes) jlm.Arg { return k(0) },
		jlm.SqTrail: func(e *sizes) jlm.Arg { return k(0) }, jlm.SqTrailN: func(e *sizes) jlm.Arg { return k(0) },
	} {
		b.field(f, v)
	}
	if marks {
		rowStart := func(e *sizes) jlm.Arg {
			return e.mul(e.less(k(0), kk(e)), e.equal(e.mod(kk(e), cols), k(0)))
		}
		b.field(jlm.SqMark, func(e *sizes) jlm.Arg { return e.where(over(e), k(1), e.where(rowStart(e), k(5), k(3))) })
		b.field(jlm.SqMarkN, func(e *sizes) jlm.Arg { return e.where(over(e), k(1), e.where(rowStart(e), k(2), k(1))) })
		b.field(jlm.SqTrail, func(e *sizes) jlm.Arg { return e.where(over(e), k(2), k(4)) })
		b.field(jlm.SqTrailN, func(e *sizes) jlm.Arg { return k(1) })
	}
	top := (slice + slicePatch) * (slice + slicePatch)
	return b.done(slicePatch, top, top/(patch*patch))
}
