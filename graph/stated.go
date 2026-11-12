package graph

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// Stated is the function the container states (e) over the operands in,
// into a row of width, as ops of this language: each reduction an OpReduce
// over the row, the terms between them OpExprs. A stored vector a term reads
// is bound at its shape in f. Every operand must be read: an expression that
// leaves one unread does not take the operands it is given.
func (s *Builder) Stated(name string, e *jlm.Expr, in []ValueID, width int, f Facts) (ValueID, error) {
	if e == nil {
		return 0, fmt.Errorf("no expression stated for %s", name)
	}
	if err := e.Valid(); err != nil {
		return 0, err
	}
	if n := e.Reads(); n != len(in) {
		return 0, fmt.Errorf("an expression of %d operands over %d", n, len(in))
	}
	val := make([]ValueID, len(e.Terms))
	have := make([]bool, len(e.Terms))
	// materialize is term k as a value: the OpExpr of every term it depends
	// on back to the operands and the reductions already values.
	materialize := func(k int) (ValueID, error) {
		if have[k] {
			return val[k], nil
		}
		need := make([]bool, k+1)
		need[k] = true
		for j := k; j >= 0; j-- {
			if !need[j] || have[j] {
				continue
			}
			for _, a := range e.Terms[j].Args {
				if a.Kind == jlm.ArgTerm {
					need[a.Index] = true
				}
			}
		}
		var ops []ValueID
		var w []Binding
		inOf := map[ValueID]int{}
		bind := func(v ValueID) int {
			i, ok := inOf[v]
			if !ok {
				i = len(ops)
				inOf[v], ops = i, append(ops, v)
			}
			return i
		}
		roleOf := map[jlm.Role]int{}
		at := map[int]int{}
		var terms []Term
		for j := 0; j <= k; j++ {
			if !need[j] || have[j] {
				continue
			}
			t := e.Terms[j]
			out := Term{Fn: t.Fn}
			for i := 0; i < t.Fn.Arity(); i++ {
				a := t.Args[i]
				switch a.Kind {
				case jlm.ArgIn:
					out.Args[i] = Arg{Kind: ArgIn, Index: bind(in[a.Index])}
				case jlm.ArgTerm:
					if have[a.Index] {
						out.Args[i] = Arg{Kind: ArgIn, Index: bind(val[a.Index])}
					} else {
						out.Args[i] = Arg{Kind: ArgTerm, Index: at[int(a.Index)]}
					}
				case jlm.ArgConst:
					out.Args[i] = Arg{Kind: ArgConst, Const: float64(a.Value)}
				case jlm.ArgWidth:
					out.Args[i] = Arg{Kind: ArgConst, Const: float64(width)}
				case jlm.ArgExact:
					return 0, fmt.Errorf("the expression holds an exact constant, which only the host evaluates")
				case jlm.ArgStored:
					sh, ok := f.Inventory[a.Role]
					if !ok {
						return 0, fmt.Errorf("the expression reads %v, which the block does not store", a.Role)
					}
					r, ok := roleOf[a.Role]
					if !ok {
						r = len(w)
						roleOf[a.Role], w = r, append(w, Binding{Role: a.Role, K: sh.K, Rows: sh.Rows, Sheets: sh.Sheets})
					}
					out.Args[i] = Arg{Kind: ArgWeight, Index: r, Elem: int(a.Index)}
				}
			}
			at[j] = len(terms)
			terms = append(terms, out)
		}
		v := s.Expr(name, width, terms, w, ops...)
		val[k], have[k] = v, true
		return v, nil
	}
	for k, t := range e.Terms {
		if !t.Reduce {
			continue
		}
		a := t.Args[0]
		var x ValueID
		if a.Kind == jlm.ArgIn {
			x = in[a.Index]
		} else {
			var err error
			if x, err = materialize(int(a.Index)); err != nil {
				return 0, err
			}
		}
		order := OrderSeq
		if t.Tree {
			order = OrderTree
		}
		val[k], have[k] = s.Emit(OpReduce, []ValueID{x}, []Result{{Name: name + ".reduce", Width: 1}}, nil,
			Attrs{Fn: t.Fn, Order: order}, nil, nil)[0], true
	}
	return materialize(len(e.Terms) - 1)
}
