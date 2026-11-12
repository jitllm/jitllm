package jlm

import (
	"encoding/binary"
	"fmt"
	"math"
)

// PictureGeometry is how a picture of w x h pixels becomes the tower's
// squares, stated as expressions the host evaluates in float64, as Python
// computes a processor's sizes (docs/design/picture-statement.md): bindings
// in order, each a value or a choice from a table; the squares' count; each
// square's sizes; and the bounds every square stays within, which size the
// classes formed at placement.
type PictureGeometry struct {
	Bind   []PictureBind
	Count  Expr
	Square [SquareFields]Expr
	Bounds PictureBounds
}

// The operands a picture's expressions read: its width and height, the
// square's number and the squares' count (Square's expressions), then the
// bindings in order from PicBound.
const (
	PicW = iota
	PicH
	PicSquare
	PicCount
	PicBound
)

// A square's fields, each an expression of the operands: the size the
// picture is resized to (Ow, Oh); the window of it the square is (X0, Y0,
// Pw, Ph); the patch grid the tower reads (Gw, Gh); the canvas the window is
// laid on (Cw, Ch) and its corner there (Cx, Cy); and the marker runs before
// and after its rows (Mark, MarkN, Trail, TrailN; 0 for none).
const (
	SqOw = iota
	SqOh
	SqX0
	SqY0
	SqPw
	SqPh
	SqGw
	SqGh
	SqCw
	SqCh
	SqCx
	SqCy
	SqMark
	SqMarkN
	SqTrail
	SqTrailN
	SquareFields
)

// PictureBind is one binding: Value, or, where Choice is set, a row of its
// table, which binds as many values as the table has columns.
type PictureBind struct {
	Value  Expr
	Choice *PictureChoice
}

// PictureChoice binds a row of Rows, Cols values a row: scanning the rows
// in order, the first eligible one, then any eligible row whose Cost is
// lexicographically smaller, or equal where Tie is positive. Its
// expressions read the operands before it and the row's values in the
// operands it binds. An empty Eligible makes every row eligible, an empty
// Tie keeps the first of equals.
type PictureChoice struct {
	Cols     int32
	Rows     []float64
	Eligible Expr
	Cost     []Expr
	Tie      Expr
}

// PictureBounds is what every square stays within: its grid's sides whole
// Units of pixels, its grid at most MaxPix pixels and MaxPatches patches.
// Unit 0 says every square's grid is the entry's own, one size.
type PictureBounds struct {
	Unit, MaxPix, MaxPatches int32
}

// Limits on a statement's size: a reader allocates nothing past them.
const (
	maxPictureBinds  = 512
	maxPictureCols   = 8
	maxPictureRows   = 4096
	maxPictureCosts  = 4
	maxExprTermsRead = 4096
)

// Reads is how many operands the statement's expressions may read: the
// picture's, then every binding.
func (g *PictureGeometry) Reads() int {
	n := PicBound
	for i := range g.Bind {
		n += g.Bind[i].Width()
	}
	return n
}

// Width is how many values b binds.
func (b *PictureBind) Width() int {
	if b.Choice != nil {
		return int(b.Choice.Cols)
	}
	return 1
}

// Valid reports why g is not a statement this build evaluates.
func (g *PictureGeometry) Valid() error {
	if len(g.Bind) > maxPictureBinds {
		return fmt.Errorf("jlm: a picture's geometry of %d bindings", len(g.Bind))
	}
	in := PicBound
	for k := range g.Bind {
		b := &g.Bind[k]
		if c := b.Choice; c != nil {
			if len(b.Value.Terms) != 0 {
				return fmt.Errorf("jlm: picture binding %d is a value and a choice", k)
			}
			if c.Cols < 1 || c.Cols > maxPictureCols || len(c.Rows) == 0 || len(c.Rows)%int(c.Cols) != 0 ||
				len(c.Rows)/int(c.Cols) > maxPictureRows || len(c.Cost) > maxPictureCosts {
				return fmt.Errorf("jlm: picture binding %d chooses from %d values in rows of %d by %d costs",
					k, len(c.Rows), c.Cols, len(c.Cost))
			}
			for _, v := range c.Rows {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return fmt.Errorf("jlm: picture binding %d's table holds %v", k, v)
				}
			}
			reads := in + int(c.Cols)
			for j, e := range append([]*Expr{&c.Eligible, &c.Tie}, costs(c)...) {
				if j < 2 && len(e.Terms) == 0 {
					continue
				}
				if err := hostExprValid(e, reads); err != nil {
					return fmt.Errorf("jlm: picture binding %d's choice: %w", k, err)
				}
			}
		} else if err := hostExprValid(&b.Value, in); err != nil {
			return fmt.Errorf("jlm: picture binding %d: %w", k, err)
		}
		in += b.Width()
	}
	if err := hostExprValid(&g.Count, in); err != nil {
		return fmt.Errorf("jlm: a picture's square count: %w", err)
	}
	for f := range g.Square {
		if err := hostExprValid(&g.Square[f], in); err != nil {
			return fmt.Errorf("jlm: a picture square's field %d: %w", f, err)
		}
	}
	if b := g.Bounds; b.Unit < 0 || b.MaxPix < max(b.Unit*b.Unit, 1) || b.MaxPatches < 1 {
		return fmt.Errorf("jlm: a picture's squares bounded by %+v", b)
	}
	return nil
}

func costs(c *PictureChoice) []*Expr {
	out := make([]*Expr, len(c.Cost))
	for i := range c.Cost {
		out[i] = &c.Cost[i]
	}
	return out
}

// hostExprValid is e as an expression of in operands the host evaluates: a
// number of the picture, not of a row.
func hostExprValid(e *Expr, in int) error {
	if len(e.Terms) > maxExprTermsRead {
		return fmt.Errorf("jlm: an expression of %d terms", len(e.Terms))
	}
	if err := e.ValidOver(in); err != nil {
		return err
	}
	return scalar(e)
}

// geometryTag opens a picture's geometry in the vision section's tail: a
// quiet NaN of a payload the video record's is not, so a build that reads
// no geometry refuses the tail rather than read it as a video.
const geometryTag = 0x7fc00002

// pictureRecord is the geometry record's revision.
const pictureRecord = 1

func encodeGeometry(w *wbuf, g *PictureGeometry) {
	w.u32(geometryTag)
	w.u32(pictureRecord)
	w.u32(uint32(len(g.Bind)))
	for _, b := range g.Bind {
		if c := b.Choice; c != nil {
			w.u8(1)
			w.i32(c.Cols)
			w.u32(uint32(len(c.Rows)))
			for _, v := range c.Rows {
				w.u64(math.Float64bits(v))
			}
			encodeExpr(w, &c.Eligible)
			w.u32(uint32(len(c.Cost)))
			for i := range c.Cost {
				encodeExpr(w, &c.Cost[i])
			}
			encodeExpr(w, &c.Tie)
			continue
		}
		w.u8(0)
		encodeExpr(w, &b.Value)
	}
	encodeExpr(w, &g.Count)
	for f := range g.Square {
		encodeExpr(w, &g.Square[f])
	}
	w.i32(g.Bounds.Unit)
	w.i32(g.Bounds.MaxPix)
	w.i32(g.Bounds.MaxPatches)
}

func decodeGeometry(r *rbuf) *PictureGeometry {
	if tag := r.u32(); r.err == nil && tag != geometryTag {
		r.err = fmt.Errorf("jlm: a picture geometry tagged %#x", tag)
		return nil
	}
	if rev := r.u32(); r.err == nil && rev != pictureRecord {
		r.err = fmt.Errorf("jlm: a picture geometry of revision %d, where this build reads %d", rev, pictureRecord)
		return nil
	}
	g := &PictureGeometry{}
	n := r.count(5)
	if n > maxPictureBinds {
		r.err = fmt.Errorf("jlm: a picture's geometry of %d bindings", n)
		return nil
	}
	g.Bind = make([]PictureBind, n)
	for k := range g.Bind {
		b := &g.Bind[k]
		switch kind := r.u8(); kind {
		case 0:
			decodeExpr(r, &b.Value)
		case 1:
			c := &PictureChoice{Cols: r.i32()}
			if m := r.count(8); m > 0 {
				c.Rows = make([]float64, m)
				for i := range c.Rows {
					c.Rows[i] = math.Float64frombits(r.u64())
				}
			}
			decodeExpr(r, &c.Eligible)
			if m := r.count(4); m > 0 && m <= maxPictureCosts {
				c.Cost = make([]Expr, m)
				for i := range c.Cost {
					decodeExpr(r, &c.Cost[i])
				}
			} else if m > maxPictureCosts {
				r.err = fmt.Errorf("jlm: a picture choice of %d costs", m)
			}
			decodeExpr(r, &c.Tie)
			b.Choice = c
		default:
			if r.err == nil {
				r.err = fmt.Errorf("jlm: a picture binding of kind %d", kind)
			}
		}
		if r.err != nil {
			return nil
		}
	}
	decodeExpr(r, &g.Count)
	for f := range g.Square {
		decodeExpr(r, &g.Square[f])
	}
	g.Bounds = PictureBounds{Unit: r.i32(), MaxPix: r.i32(), MaxPatches: r.i32()}
	return g
}

// tailTag is the word that opens the next record of the vision section's
// tail, 0 where fewer than four bytes are left.
func tailTag(r *rbuf) uint32 {
	if len(r.b) < 4 {
		return 0
	}
	return binary.LittleEndian.Uint32(r.b)
}
