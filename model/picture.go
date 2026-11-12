package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// geometry evaluates a picture's stated geometry (jlm.PictureGeometry): the
// operands its expressions read and the terms' scratch, reused from one
// picture to the next.
type geometry struct {
	g    *jlm.PictureGeometry
	in   []float64
	cand []float64
	vals []float64
}

// bind evaluates the statement's bindings for a picture of w x h pixels:
// the operands its count and squares read.
func (e *geometry) bind(w, h int) error {
	e.in = append(e.in[:0], float64(w), float64(h), 0, 0)
	for k := range e.g.Bind {
		b := &e.g.Bind[k]
		if b.Choice == nil {
			v, err := e.eval(&b.Value, e.in)
			if err != nil {
				return fmt.Errorf("model: picture binding %d: %w", k, err)
			}
			e.in = append(e.in, v)
			continue
		}
		row, err := e.choose(b.Choice)
		if err != nil {
			return fmt.Errorf("model: picture binding %d: %w", k, err)
		}
		e.in = append(e.in, row...)
	}
	return nil
}

func (e *geometry) eval(x *jlm.Expr, in []float64) (float64, error) {
	v, vals, err := evalInto(x, in, e.vals)
	e.vals = vals
	return v, err
}

// choose is the row of c's table its rule picks over the operands so far:
// the first eligible row, then each eligible row of a lexicographically
// smaller cost, or of an equal one where the tie says so.
func (e *geometry) choose(c *jlm.PictureChoice) ([]float64, error) {
	cols := int(c.Cols)
	best := -1
	var bestCost, cost [4]float64
	e.cand = append(e.cand[:0], e.in...)
	e.cand = append(e.cand, make([]float64, cols)...)
	row := e.cand[len(e.in):]
	for r := 0; r < len(c.Rows)/cols; r++ {
		copy(row, c.Rows[r*cols:(r+1)*cols])
		if len(c.Eligible.Terms) > 0 {
			ok, err := e.eval(&c.Eligible, e.cand)
			if err != nil {
				return nil, err
			}
			if !(ok > 0) {
				continue
			}
		}
		for i := range c.Cost {
			v, err := e.eval(&c.Cost[i], e.cand)
			if err != nil {
				return nil, err
			}
			cost[i] = v
		}
		switch order := compareCosts(cost[:len(c.Cost)], bestCost[:len(c.Cost)]); {
		case best < 0 || order < 0:
			best, bestCost = r, cost
		case order == 0 && len(c.Tie.Terms) > 0:
			tie, err := e.eval(&c.Tie, e.cand)
			if err != nil {
				return nil, err
			}
			if tie > 0 {
				best, bestCost = r, cost
			}
		}
	}
	if best < 0 {
		return nil, fmt.Errorf("no row of %d is eligible", len(c.Rows)/cols)
	}
	return c.Rows[best*cols : (best+1)*cols], nil
}

// compareCosts orders two costs lexicographically: -1 where a comes first,
// 0 where they are equal, 1 otherwise. A cost that is not a number orders
// after every number.
func compareCosts(a, b []float64) int {
	for i := range a {
		switch {
		case a[i] < b[i] || !math.IsNaN(a[i]) && math.IsNaN(b[i]):
			return -1
		case a[i] > b[i] || math.IsNaN(a[i]) && !math.IsNaN(b[i]):
			return 1
		}
	}
	return 0
}

// count is how many squares the bound picture becomes.
func (e *geometry) count() (int, error) {
	v, err := e.eval(&e.g.Count, e.in)
	if err != nil {
		return 0, fmt.Errorf("model: a picture's square count: %w", err)
	}
	n, ok := whole(v)
	if !ok || n < 1 {
		return 0, fmt.Errorf("model: a picture of %v squares", v)
	}
	return n, nil
}

// square is square i of n of the bound picture, every field of it.
func (e *geometry) square(i, n int) ([jlm.SquareFields]int, error) {
	var out [jlm.SquareFields]int
	e.in[jlm.PicSquare], e.in[jlm.PicCount] = float64(i), float64(n)
	for f := range e.g.Square {
		v, err := e.eval(&e.g.Square[f], e.in)
		if err != nil {
			return out, fmt.Errorf("model: a picture square's field %d: %w", f, err)
		}
		k, ok := whole(v)
		if !ok || k < 0 {
			return out, fmt.Errorf("model: square %d's field %d is %v", i, f, v)
		}
		out[f] = k
	}
	return out, nil
}

// whole is v as an int where it is a whole number an int holds.
func whole(v float64) (int, bool) {
	if v != math.Trunc(v) || math.Abs(v) > 1<<40 {
		return 0, false
	}
	return int(v), true
}

// GeometrySquare is square i of a w x h picture as g states it, and how many
// squares the picture becomes; its class the caller's to choose.
func GeometrySquare(g *jlm.PictureGeometry, w, h, i int) (Square, int, error) {
	e := geometry{g: g}
	if err := e.bind(w, h); err != nil {
		return Square{}, 0, err
	}
	n, err := e.count()
	if err != nil {
		return Square{}, 0, err
	}
	if i < 0 || i >= n {
		return Square{}, n, fmt.Errorf("model: square %d of a picture of %d", i, n)
	}
	f, err := e.square(i, n)
	if err != nil {
		return Square{}, n, err
	}
	return Square{Ow: f[jlm.SqOw], Oh: f[jlm.SqOh], X0: f[jlm.SqX0], Y0: f[jlm.SqY0], Pw: f[jlm.SqPw], Ph: f[jlm.SqPh],
		Gw: f[jlm.SqGw], Gh: f[jlm.SqGh], Mark: f[jlm.SqMark], MarkN: f[jlm.SqMarkN], Trail: f[jlm.SqTrail],
		TrailN: f[jlm.SqTrailN]}, n, nil
}
