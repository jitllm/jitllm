// Package graphref is the block graph's reference evaluator: each op by its
// definition (docs/design/composer.md, "Decomposition"), over one token, in
// float32 with every reduction in its stated order. It is the golden a
// composed kernel is compared against (model.TestComposerMatchesTheGolden).
// Test-only, as all of internal/oracle is.
package graphref

import (
	"fmt"
	"math"
	"math/big"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// failer records the first failure an evaluation meets.
type failer struct{ err error }

func (f *failer) Fatalf(format string, a ...any) {
	if f.err == nil {
		f.err = fmt.Errorf("graphref: "+format, a...)
	}
	panic(f.err)
}

// EvalOp is op o of g over the values in (by id) and the weights w: its
// outputs by id, or why it is not evaluated.
func EvalOp(g *graph.Graph, o graph.Op, in map[graph.ValueID][]float32, w map[jlm.Role][]float32) (map[graph.ValueID][]float32, error) {
	one := &graph.Graph{Version: g.Version, Values: g.Values, Ops: []graph.Op{o}}
	return EvalGraph(one, in, w)
}

func EvalGraph(g *graph.Graph, in map[graph.ValueID][]float32, w map[jlm.Role][]float32) (v map[graph.ValueID][]float32, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("graphref: %v", r)
		}
	}()
	t := &failer{}
	v = map[graph.ValueID][]float32{}
	for id, x := range in {
		v[id] = x
	}
	for _, o := range g.Ops {
		width := g.Values[o.Out[0]].Width
		out := make([]float32, width)
		switch o.Kind {
		case graph.OpMatVec:
			x, m := v[o.In[0]], w[o.Weight[0].Role]
			k := len(x)
			for r := range out {
				var s float32
				for c := 0; c < k; c++ {
					s += m[r*k+c] * x[c]
				}
				out[r] = s
			}
		case graph.OpNorm:
			// An RMSNorm in float32 with float64 statistics: the oracle's.
			oracle.RMSNorm32(out, v[o.In[0]], w[o.Weight[0].Role], o.Attr.Eps)
		case graph.OpAdd:
			// In[0] + Alpha*In[1] (Alpha 0 is one), the sum times Scale where
			// the op states one.
			a, b := v[o.In[0]], v[o.In[1]]
			for i := range out {
				y := b[i]
				if al := o.Attr.Alpha; al != 0 && al != 1 {
					y *= float32(al)
				}
				out[i] = a[i] + y
				if sc := o.Attr.Scale; sc != 0 {
					out[i] *= float32(sc)
				}
			}
		case graph.OpMap:
			var ops [][]float32
			for _, id := range o.In {
				ops = append(ops, v[id])
			}
			if len(o.Weight) > 0 {
				ops = append(ops, w[o.Weight[0].Role])
			}
			if len(ops) < o.Attr.Fn.Arity() {
				ops = append(ops, []float32{float32(o.Attr.Const)})
			}
			at := func(op []float32, i int) float32 { return op[i/(width/len(op))] }
			for i := range out {
				a := at(ops[0], i)
				switch o.Attr.Fn {
				case graph.MapAdd:
					out[i] = a + at(ops[1], i)
				case graph.MapSub:
					out[i] = a - at(ops[1], i)
				case graph.MapMul:
					out[i] = a * at(ops[1], i)
				case graph.MapDiv:
					out[i] = a / at(ops[1], i)
				case graph.MapMax:
					out[i] = max(a, at(ops[1], i))
				case graph.MapMin:
					out[i] = min(a, at(ops[1], i))
				case graph.MapExp:
					out[i] = float32(math.Exp(float64(a)))
				case graph.MapSigmoid:
					out[i] = float32(1 / (1 + math.Exp(-float64(a))))
				case graph.MapSoftplus:
					out[i] = float32(math.Log1p(math.Exp(float64(a))))
				case graph.MapSqrt:
					out[i] = float32(math.Sqrt(float64(a)))
				case graph.MapWhere:
					out[i] = at(ops[2], i)
					if a > 0 {
						out[i] = at(ops[1], i)
					}
				case graph.MapLess:
					if a < at(ops[1], i) {
						out[i] = 1
					}
				case graph.MapFma:
					out[i] = float32(math.FMA(float64(a), float64(at(ops[1], i)), float64(at(ops[2], i))))
				default:
					t.Fatalf("map %s", o.Attr.Fn)
				}
			}
		case graph.OpReduce:
			x := v[o.In[0]]
			span := o.Attr.Span
			if span == 0 {
				span = len(x)
			}
			for gi := range out {
				grp := x[gi*span : (gi+1)*span]
				var acc float32
				switch o.Attr.Fn {
				case graph.MapAdd:
				case graph.MapMax, graph.MapMin:
					acc = grp[0]
				}
				if o.Attr.Fn == graph.MapAdd && o.Attr.Order == graph.OrderTree {
					// Pairwise: the first half with the second, down to one.
					t := append([]float32(nil), grp...)
					for n := len(t); n > 1; {
						h := (n + 1) / 2
						for i := 0; i < n-h; i++ {
							t[i] += t[i+h]
						}
						n = h
					}
					out[gi] = t[0]
					continue
				}
				for _, e := range grp {
					switch o.Attr.Fn {
					case graph.MapAdd:
						acc += e
					case graph.MapMax:
						acc = max(acc, e)
					case graph.MapMin:
						acc = min(acc, e)
					}
				}
				out[gi] = acc
			}
		case graph.OpTopK, graph.OpRank:
			x := v[o.In[0]]
			span := o.Attr.Span
			if span == 0 {
				span = len(x)
			}
			rank := func(grp []float32, e int) int {
				n := 0
				for f, y := range grp {
					if y > grp[e] || y == grp[e] && f < e {
						n++
					}
				}
				return n
			}
			if o.Kind == graph.OpRank {
				for e := range out {
					gi := e / span
					out[e] = float32(rank(x[gi*span:(gi+1)*span], e%span))
				}
				break
			}
			k := o.Attr.TopK
			vals := make([]float32, width)
			for gi := 0; gi < len(x)/span; gi++ {
				grp := x[gi*span : (gi+1)*span]
				for e := range grp {
					if r := rank(grp, e); r < k {
						out[gi*k+r], vals[gi*k+r] = float32(e), grp[e]
					}
				}
			}
			v[o.Out[1]] = vals
		case graph.OpTake:
			x, ix := v[o.In[0]], v[o.In[1]]
			for j := range out {
				out[j] = x[int(ix[j])]
			}
		case graph.OpTile:
			x := v[o.In[0]]
			span := o.Attr.Span
			if span == 0 {
				span = len(x)
			}
			for i := range out {
				out[i] = x[i%span]
			}
		case graph.OpSlice:
			// Of In[0], or of the bound vector where there is no input.
			var x []float32
			if len(o.In) > 0 {
				x = v[o.In[0]]
			} else {
				x = w[o.Weight[0].Role]
			}
			a := o.Attr
			if a.HeadDim == 0 {
				copy(out, x[a.Offset:a.Offset+a.Width])
				break
			}
			for h := 0; h < len(x)/a.HeadDim; h++ {
				copy(out[h*a.Width:], x[h*a.HeadDim+a.Offset:h*a.HeadDim+a.Offset+a.Width])
			}
		case graph.OpConcat:
			off := 0
			for _, id := range o.In {
				off += copy(out[off:], v[id])
			}
		case graph.OpHeadNorm:
			// A norm of each head: x * (1/sqrt(mean(x^2) + eps)), the mean
			// accumulated in f64 as the oracle's does, times the bound scale.
			x, hd := v[o.In[0]], o.Attr.HeadDim
			for h := 0; h < len(x)/hd; h++ {
				var sum float64
				for _, e := range x[h*hd : (h+1)*hd] {
					sum += float64(e) * float64(e)
				}
				scale := 1 / math.Sqrt(sum/float64(hd)+o.Attr.Eps)
				for i := 0; i < hd; i++ {
					out[h*hd+i] = float32(float64(x[h*hd+i]) * scale)
					if len(o.Weight) > 0 {
						// The bound scale: one of the head's width shared by
						// every head, or one of the row's width.
						ws := w[o.Weight[0].Role]
						if len(ws) == hd {
							out[h*hd+i] *= ws[i]
						} else if len(ws) == len(x) {
							out[h*hd+i] *= ws[h*hd+i]
						}
					}
				}
			}
		case graph.OpAct:
			x := v[o.In[0]]
			for i := range out {
				out[i] = float32(refAct(t, o.Attr.Act, float64(x[i])))
			}
		case graph.OpActMul:
			g, up := v[o.In[0]], v[o.In[1]]
			for i := range out {
				if o.Attr.Act == graph.ActSigmoid {
					out[i] = float32(1/(1+math.Exp(-float64(g[i])))) * up[i]
					continue
				}
				out[i] = float32(refAct(t, o.Attr.Act, float64(g[i]))) * up[i]
			}
		case graph.OpCombine:
			continue
		default:
			t.Fatalf("op %s is not a primitive", o.Kind)
		}
		v[o.Out[0]] = out
	}
	if t.err != nil {
		return nil, t.err
	}
	return v, nil
}

func refAct(t *failer, a graph.Act, x float64) float64 {
	switch a {
	case graph.ActSiLU:
		return oracle.SiLU(x)
	case graph.ActGELUTanh:
		return oracle.GELUTanh(x)
	case graph.ActGELUErf:
		return oracle.GELUErf(x)
	}
	t.Fatalf("act %d", a)
	return 0
}

func fma32(a, b, c float32) float32 {
	if isNonFinite(a) || isNonFinite(b) || isNonFinite(c) {
		return a*b + c
	}
	z := new(big.Float).SetPrec(300).SetFloat64(float64(a))
	z.Mul(z, new(big.Float).SetPrec(300).SetFloat64(float64(b)))
	z.Add(z, new(big.Float).SetPrec(300).SetFloat64(float64(c)))
	f, _ := z.Float32()
	return f
}

func isNonFinite(x float32) bool { return math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) }
