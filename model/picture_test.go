package model

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// gridChoice is a geometry binding the grid of its table its rule picks --
// the row's columns c and r, the cost |w/h - c/r|, the tie where tie is
// stated -- the count c*r, every square's Ow the chosen c and Oh r.
func gridChoice(rows []float64, tie bool, eligible bool) *jlm.PictureGeometry {
	var cost, el, t, cnt jlm.ExprBuilder
	cost.Fn(jlm.FnAbs, cost.Fn(jlm.FnSub, cost.Fn(jlm.FnDiv, jlm.Operand(jlm.PicW), jlm.Operand(jlm.PicH)),
		cost.Fn(jlm.FnDiv, jlm.Operand(jlm.PicBound), jlm.Operand(jlm.PicBound+1))))
	el.Fn(jlm.FnLess, jlm.Operand(jlm.PicBound), jlm.ExactNum(3))
	t.Fn(jlm.FnSub, jlm.Operand(jlm.PicW), jlm.ExactNum(100))
	cnt.Fn(jlm.FnMul, jlm.Operand(jlm.PicBound), jlm.Operand(jlm.PicBound+1))
	c := &jlm.PictureChoice{Cols: 2, Rows: rows, Cost: []jlm.Expr{{Terms: cost.T}}}
	if tie {
		c.Tie = jlm.Expr{Terms: t.T}
	}
	if eligible {
		c.Eligible = jlm.Expr{Terms: el.T}
	}
	g := &jlm.PictureGeometry{Bind: []jlm.PictureBind{{Choice: c}}, Count: jlm.Expr{Terms: cnt.T},
		Bounds: jlm.PictureBounds{Unit: 1, MaxPix: 1, MaxPatches: 1}}
	for f := range g.Square {
		var e jlm.ExprBuilder
		e.Fn(jlm.FnAdd, jlm.ExactNum(0), jlm.ExactNum(0))
		g.Square[f] = jlm.Expr{Terms: e.T}
	}
	var ow, oh jlm.ExprBuilder
	ow.Fn(jlm.FnAdd, jlm.Operand(jlm.PicBound), jlm.ExactNum(0))
	oh.Fn(jlm.FnAdd, jlm.Operand(jlm.PicBound+1), jlm.ExactNum(0))
	g.Square[jlm.SqOw], g.Square[jlm.SqOh] = jlm.Expr{Terms: ow.T}, jlm.Expr{Terms: oh.T}
	return g
}

// TestAStatedChoiceFollowsItsRule holds a geometry's choice to its rule:
// the least cost wins, the first of equals unless the tie says the later,
// an ineligible row is never chosen, and no eligible row is refused; a
// square's field that is no whole number, and a count under one, are
// refused.
func TestAStatedChoiceFollowsItsRule(t *testing.T) {
	// (1,1) (2,1) (1,2) (4,2): a 200x100 picture is 2, equal to (4,2)'s.
	rows := []float64{1, 1, 2, 1, 1, 2, 4, 2}
	for _, c := range []struct {
		name    string
		w, h    int
		tie, el bool
		ow, oh  int
	}{
		{"least cost", 300, 100, false, false, 2, 1},
		{"the first of equals", 200, 100, false, false, 2, 1},
		{"the tie's later", 200, 100, true, false, 4, 2},
		{"the tie's first where it is not positive", 100, 50, true, false, 2, 1},
		{"an ineligible row passed", 200, 100, true, true, 2, 1},
		{"a square picture", 64, 64, false, false, 1, 1},
	} {
		g := gridChoice(rows, c.tie, c.el)
		sq, n, err := GeometrySquare(g, c.w, c.h, 0)
		if err != nil || sq.Ow != c.ow || sq.Oh != c.oh || n != c.ow*c.oh {
			t.Errorf("%s: %dx%d chose %dx%d (%d squares, %v), want %dx%d", c.name, c.w, c.h, sq.Ow, sq.Oh, n, err, c.ow, c.oh)
		}
	}
	g := gridChoice([]float64{4, 2}, false, true)
	if _, _, err := GeometrySquare(g, 100, 100, 0); err == nil {
		t.Error("a choice of no eligible row chose one")
	}
	g = gridChoice(rows, false, false)
	var half jlm.ExprBuilder
	half.Fn(jlm.FnMul, jlm.Operand(jlm.PicW), jlm.ExactNum(0.5))
	g.Square[jlm.SqX0] = jlm.Expr{Terms: half.T}
	if _, _, err := GeometrySquare(g, 101, 100, 0); err == nil {
		t.Error("a square at x 50.5 was read")
	}
	g = gridChoice(rows, false, false)
	var zero jlm.ExprBuilder
	zero.Fn(jlm.FnSub, jlm.Operand(jlm.PicW), jlm.Operand(jlm.PicW))
	g.Count = jlm.Expr{Terms: zero.T}
	if _, _, err := GeometrySquare(g, 100, 100, 0); err == nil {
		t.Error("a picture of no squares was read")
	}
}
