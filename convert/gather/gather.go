// Package gather states row gathers as a container holds them (jlm.TOpGather):
// a reference's reordering of a value's rows, written once at conversion as
// an index expression the engine lowers without knowing its geometry.
package gather

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// op is the gather of x's index with r its source row, rows output rows an
// input row (0: the program's).
func op(x *graph.Index, r graph.Arg, rows int32) jlm.TowerOp {
	e, bound := graph.GatherIndex(x.Attrs(r))
	return jlm.TowerOp{Code: jlm.TOpGather, Expr: e, Span: bound, Width: rows}
}

// Untile is the input's rows -- the cells of a grid in nested order, window x
// window windows of cells, each window's cells in raster order and
// consecutive, the windows in raster order and those at the grid's right and
// lower edges cut to it -- in the cells' raster order: the grid the request's
// patch grid over merge wide, its live rows over merge*merge over that tall.
func Untile(merge, window int) (jlm.TowerOp, error) {
	if merge < 1 || window < 1 {
		return jlm.TowerOp{}, fmt.Errorf("gather: an untile of %d-cell merges in windows of %d cells: both must be at least one", merge, window)
	}
	var x graph.Index
	m := merge
	gw := x.Max(x.Div(x.In(graph.GatherGridW), x.K(m)), x.K(1))
	gh := x.Max(x.Div(x.Div(x.In(graph.GatherLive), x.K(m*m)), gw), x.K(1))
	r := x.In(graph.GatherRow)
	cy := x.Div(r, gw)
	cx := x.Rem(r, gw, cy)
	win := x.K(window)
	wy, wx := x.Div(cy, win), x.Div(cx, win)
	top, left := x.Mul(wy, win), x.Mul(wx, win)
	wh := x.Max(x.Min(win, x.Sub(x.Max(gh, top), top)), x.K(1))
	ww := x.Max(x.Min(win, x.Sub(x.Max(gw, left), left)), x.K(1))
	at := x.Add(x.Mul(top, gw), x.Mul(left, wh))
	at = x.Add(at, x.Add(x.Mul(x.Sub(cy, top), ww), x.Sub(cx, left)))
	return op(&x, x.Min(at, x.InLast()), 1), nil
}

// ClipMask is the input with the rows past a clip's valid length zero: the
// rows a grid w wide (0: one row a grid row), a grid row kept below L, L the
// request's samples over stride halved halvings times as a stride-two
// convolution of three taps padded by one halves it (L = (L-1)/2+1): row*2^h
// < L0 is (row*2^h + 1)*stride <= the samples, no division of the count.
func ClipMask(w, stride, halvings int) (jlm.TowerOp, error) {
	if w < 0 || stride < 1 || halvings < 0 || halvings > 16 {
		return jlm.TowerOp{}, fmt.Errorf("gather: a clip mask on a grid %d wide, %d samples a frame, %d halvings: "+
			"the width must not be negative, the stride at least one and the halvings 0 to 16", w, stride, halvings)
	}
	var x graph.Index
	r := x.In(graph.GatherRow)
	row := r
	if w > 0 {
		row = x.Div(r, x.K(w))
	}
	end := x.Mul(x.Add(x.Mul(row, x.K(1<<halvings)), x.K(1)), x.K(stride))
	return op(&x, x.Where(x.Less(x.In(graph.GatherSamples), end), x.K(graph.GatherNone), r), 0), nil
}
