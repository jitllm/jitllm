package graph

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// row is a value of g named name, n wide, written by op o.
func (g *Graph) row(name string, n int, o Op) ValueID {
	v := g.value(name, n)
	o.Out = []ValueID{v}
	g.op(o)
	return v
}

// mapc is f over the inputs, Const the second operand where f takes one more.
func (g *Graph) mapc(name string, n int, f MapFn, c float64, in ...ValueID) ValueID {
	return g.row(name, n, Op{Kind: OpMap, In: in, Attr: Attrs{Fn: f, Const: c}})
}

// rms is sqrt(mean(x*x)) of the n elements of x, at least eps under the
// root where eps is not 0.
func (g *Graph) rms(name string, x ValueID, n int, eps float64) ValueID {
	sq := g.mapc(name+".sq", n, MapMul, 0, x, x)
	ss := g.row(name+".ss", 1, Op{Kind: OpReduce, In: []ValueID{sq}, Attr: Attrs{Fn: MapAdd, Order: OrderSeq}})
	ms := g.mapc(name+".mean", 1, MapDiv, float64(n), ss)
	if eps != 0 {
		ms = g.mapc(name+".floor", 1, MapMax, eps, ms)
	}
	return g.mapc(name+".rms", 1, MapSqrt, 0, ms)
}

// magMatch is AltUp's magnitude match of x to ref, both n wide:
// x * sqrt(mean(ref^2)) / sqrt(max(mean(x^2), eps)).
func (g *Graph) magMatch(name string, x, ref ValueID, n int, eps float64) ValueID {
	f := g.mapc(name+".ratio", 1, MapDiv, 0, g.rms(name+".ref", ref, n, 0), g.rms(name+".x", x, n, eps))
	return g.mapc(name, n, MapMul, 0, x, f)
}

// MagEps is AltUp's magnitude match's floor.
const MagEps = 1e-5

// altExpand is AltUp's residual of n streams from the row x, d wide:
// stream 0 x, stream k its projection's k-th row matched to x's magnitude.
func (g *Graph) altExpand(x ValueID, n, d int) ValueID {
	e := g.row("alt.proj", (n-1)*d, Op{Kind: OpMatVec, In: []ValueID{x},
		Weight: []Binding{{Role: jlm.RoleAltUpProj, K: d, Rows: (n - 1) * d}}})
	parts := []ValueID{x}
	for k := 1; k < n; k++ {
		s := g.row("alt.stream", d, Op{Kind: OpSlice, In: []ValueID{e}, Attr: Attrs{Offset: (k - 1) * d, Width: d}})
		parts = append(parts, g.magMatch("alt.matched", s, x, d, MagEps))
	}
	return g.row("x_streams", n*d, Op{Kind: OpConcat, In: parts})
}

// altCollapse is the head's row from AltUp's n streams of x: the mean of
// stream 0 and each other stream's unembedding matched to stream 0's
// magnitude, added in order.
func (g *Graph) altCollapse(x ValueID, n, d int) ValueID {
	x0 := g.row("alt.stream0", d, Op{Kind: OpSlice, In: []ValueID{x}, Attr: Attrs{Offset: 0, Width: d}})
	acc := x0
	for k := 1; k < n; k++ {
		s := g.row("alt.stream", d, Op{Kind: OpSlice, In: []ValueID{x}, Attr: Attrs{Offset: k * d, Width: d}})
		u := g.row("alt.unembd", d, Op{Kind: OpMatVec, In: []ValueID{s},
			Weight: []Binding{{Role: jlm.RoleAltUpUnembd1 + jlm.Role(k-1), K: d, Rows: d}}})
		acc = g.row("alt.sum", d, Op{Kind: OpAdd, In: []ValueID{acc, g.magMatch("alt.matched", u, x0, d, MagEps)}})
	}
	return g.mapc("collapsed", d, MapMul, float64(float32(1/float64(n))), acc)
}

// PLE is a model's per-layer inputs (Gemma 3n's and Gemma 4's E-models):
// Layers inputs of Dim a row, from the token's row of a second table of
// Rows rows and the embedding row's projection.
type PLE struct {
	Layers, Dim, Rows int
}

// inputs is each layer's input from the embedding row x, d wide: the
// token's table row times sqrt(Dim), plus the projection of x over
// sqrt(d), each layer's slice of it RMS-normed against RolePLEProjNorm,
// the sum over sqrt(2); one value a layer.
func (p *PLE) inputs(g *Graph, x ValueID, d int, eps float64) []ValueID {
	w := p.Layers * p.Dim
	t := g.row("ple.tok", w, Op{Kind: OpEmbed, Weight: []Binding{{Role: jlm.RolePLETokEmbd, K: w, Rows: p.Rows}}})
	t = g.mapc("ple.tok_s", w, MapMul, float64(float32(math.Sqrt(float64(p.Dim)))), t)
	u := g.row("ple.proj", w, Op{Kind: OpMatVec, In: []ValueID{x}, Weight: []Binding{{Role: jlm.RolePLEModelProj, K: d, Rows: w}}})
	u = g.mapc("ple.proj_s", w, MapMul, float64(float32(1/math.Sqrt(float64(d)))), u)
	u = g.row("ple.normed", w, Op{Kind: OpHeadNorm, In: []ValueID{u}, Weight: []Binding{{Role: jlm.RolePLEProjNorm, K: p.Dim}},
		Attr: Attrs{Eps: eps, HeadDim: p.Dim}})
	s := g.row("ple.sum", w, Op{Kind: OpAdd, In: []ValueID{u, t}})
	s = g.mapc("ple.all", w, MapMul, float64(float32(1/math.Sqrt2)), s)
	out := make([]ValueID, p.Layers)
	for li := range out {
		out[li] = g.row("ple_in", p.Dim, Op{Kind: OpSlice, In: []ValueID{s}, Attr: Attrs{Offset: li * p.Dim, Width: p.Dim}})
	}
	return out
}
