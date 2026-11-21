package graph

import (
	"fmt"
	"math"
	"slices"
)

// IterateElems is the element type of each of an OpIterate's terms. Value
// j's is its output's (Out[j], or Out[0] where one output holds every
// value), and so is term j's, which starts it; a take's is its table's;
// any other term's is a float where one of its operands is, else a word. A
// word read by a float term is its integer's float. A float is never read
// as a word, a take's index is a word, a term of constants alone has no
// type, a sum over a run is a trip's read of a float term before it, and a
// value is carried out by a term of its own type.
func IterateElems(g *Graph, o Op) ([]Elem, error) {
	n := o.Attr.Taps
	if n < 1 || len(o.In) < 1 || len(o.Out) != 1 && len(o.Out) != n || len(o.Attr.Expr) < 2*n {
		return nil, fmt.Errorf("%d values from %d terms into %d outputs", n, len(o.Attr.Expr), len(o.Out))
	}
	val := func(j int) Elem {
		if len(o.Out) == n {
			return g.Values[o.Out[j]].Elem
		}
		return g.Values[o.Out[0]].Elem
	}
	es := make([]Elem, len(o.Attr.Expr))
	for k, tm := range o.Attr.Expr {
		ar := tm.Fn.Arity()
		if ar == 0 {
			return nil, fmt.Errorf("term %d has no function", k)
		}
		var ops []Elem
		var consts []float64
		for j, a := range tm.Args {
			switch {
			case j >= ar:
				if a.Kind != ArgNone {
					return nil, fmt.Errorf("term %d's operand %d is past its function's", k, j)
				}
				continue
			case tm.Fn == MapTake && j == 0:
				if a.Kind != ArgIn || a.Index < 1 || a.Index >= len(o.In) {
					return nil, fmt.Errorf("term %d takes from no table", k)
				}
				continue
			}
			switch a.Kind {
			case ArgIn:
				if a.Index != 0 {
					return nil, fmt.Errorf("term %d reads table %d outside a take", k, a.Index)
				}
				ops = append(ops, g.Values[o.In[0]].Elem)
			case ArgTerm:
				if a.Index < 0 || a.Index >= k {
					return nil, fmt.Errorf("term %d reads term %d", k, a.Index)
				}
				ops = append(ops, es[a.Index])
			case ArgState:
				if k < n || a.Index < 0 || a.Index >= n {
					return nil, fmt.Errorf("term %d reads carried value %d", k, a.Index)
				}
				ops = append(ops, val(a.Index))
			case ArgSum:
				if k < n || o.Attr.Width < 1 || a.Index < 0 || a.Index >= k {
					return nil, fmt.Errorf("term %d sums term %d over runs of %d", k, a.Index, o.Attr.Width)
				}
				if es[a.Index] != ElemF32 {
					return nil, fmt.Errorf("term %d sums term %d, which is no float", k, a.Index)
				}
				ops = append(ops, ElemF32)
			case ArgConst:
				consts = append(consts, a.Const)
			default:
				return nil, fmt.Errorf("term %d's operand %d is of kind %d", k, j, a.Kind)
			}
		}
		e := ElemU32
		switch {
		case k < n:
			e = val(k)
		case tm.Fn == MapTake:
			e = g.Values[o.In[tm.Args[0].Index]].Elem
		case len(ops) == 0:
			return nil, fmt.Errorf("term %d reads constants alone, which have no type", k)
		case slices.Contains(ops, ElemF32):
			e = ElemF32
		}
		if e != ElemU32 && e != ElemF32 {
			return nil, fmt.Errorf("term %d is of type %d, neither a word nor a float", k, e)
		}
		if tm.Fn == MapTake && (len(ops) != 1 || ops[0] != ElemU32) {
			return nil, fmt.Errorf("term %d takes at an index that is no word", k)
		}
		if e == ElemU32 && tm.Fn != MapTake {
			for _, op := range ops {
				if op != ElemU32 {
					return nil, fmt.Errorf("term %d reads a float into a word", k)
				}
			}
			for _, c := range consts {
				if c < 0 || c >= 1<<32 || c != math.Trunc(c) {
					return nil, fmt.Errorf("term %d reads %v, which is no word", k, c)
				}
			}
		}
		es[k] = e
	}
	for j := range n {
		next := len(es) - n + j
		if len(o.Attr.Words) == n {
			next = int(o.Attr.Words[j])
		}
		if next < n || next >= len(es) {
			return nil, fmt.Errorf("value %d is carried out by term %d of %d", j, next, len(es))
		}
		if es[next] != val(j) {
			return nil, fmt.Errorf("value %d, of type %d, is carried out by term %d, of type %d", j, val(j), next, es[next])
		}
	}
	return es, nil
}
