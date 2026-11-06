package engine

import (
	"slices"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

func argIn(i int) graph.Arg    { return graph.Arg{Kind: graph.ArgIn, Index: i} }
func argTerm(i int) graph.Arg  { return graph.Arg{Kind: graph.ArgTerm, Index: i} }
func argK(v float64) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: v} }
func term(f graph.MapFn, a ...graph.Arg) graph.Term {
	t := graph.Term{Fn: f}
	copy(t.Args[:], a)
	return t
}

// outGateTerms is sigmoid(gate) * attention, the output gate as the front
// end states it.
func outGateTerms() []graph.Term {
	return []graph.Term{term(graph.MapSigmoid, argIn(0)), term(graph.MapMul, argTerm(0), argIn(1))}
}

// isOutGate reports whether o is the attention's output gate.
func isOutGate(o graph.Op) bool {
	return o.Kind == graph.OpExpr && slices.Equal(o.Attr.Expr, outGateTerms())
}

// sigmoidGate is sigmoid(g) * v as an op of b.
func sigmoidGate(b *graph.Builder, g, v graph.ValueID, width int) graph.ValueID {
	return b.Expr("gated", width, outGateTerms(), nil, g, v)
}

// reluSquaredTerms and siluTerms are two activations a fault swaps one for
// the other.

// statedViolations are m's stated activations taken apart where the composer
// reads them, the host computing the clean model: a constant moved, a
// select's branches swapped. A model the front end states in part gets none,
// since the blocks it leaves out never read them.
func statedViolations(m *model.Model) []c6Violation {
	if coveredBlocks(m) != m.Cfg.NLayer {
		return nil
	}
	var consts, wheres bool
	for _, e := range m.Container.Config().Acts {
		for _, t := range e.Terms {
			wheres = wheres || t.Fn == jlm.FnWhere
			for _, a := range t.Args {
				consts = consts || a.Kind == jlm.ArgConst
			}
		}
	}
	var vs []c6Violation
	if consts {
		// A bound (a min's or a max's constant) moves nothing a fixture's
		// values never reach -- a clamped SwiGLU's 7 -- so the constant moved
		// is the first one no bound holds, the first of all where every one is
		// a bound.
		vs = append(vs, c6Violation{"the stated activation's first constant moved", statedExprMut(func(ts []jlm.Term) {
			at := func(bounds bool) bool {
				for k := range ts {
					if !bounds && (ts[k].Fn == jlm.FnMin || ts[k].Fn == jlm.FnMax) {
						continue
					}
					for j := range ts[k].Args {
						if ts[k].Args[j].Kind == jlm.ArgConst {
							ts[k].Args[j].Value++
							return true
						}
					}
				}
				return false
			}
			if !at(false) {
				at(true)
			}
		})})
	}
	if wheres {
		vs = append(vs, c6Violation{"the stated activation's branches swapped", statedMut(func(t *jlm.Term) bool {
			if t.Fn != jlm.FnWhere {
				return false
			}
			t.Args[1], t.Args[2] = t.Args[2], t.Args[1]
			return true
		})})
	}
	return vs
}

// actSwap runs every block under the activation the converter states for
// flag k (0: the gated SiLU), as a container of that activation states it.
func actSwap(k jlm.Flags) func(*model.Model) func() {
	return func(m *model.Model) func() {
		c := m.Container.Config()
		was, wasL := c.Acts, c.ActLayers
		s := *c
		s.Arch = jlm.ArchLlama
		s.Flags &^= jlm.FlagGELU | jlm.FlagReLU2 | jlm.FlagSwiGLUOAI
		s.Flags2 &^= jlm.Flag2SwiGLUClamp
		s.SituBeta, s.SituLinearBeta, s.SparseStd = 0, 0, 0
		s.Flags |= k
		gated := make([]bool, c.NLayer)
		for li := range gated {
			// As the converter reads it: the dense FFN's roles or the
			// experts' banks.
			gate := m.Stores(li, jlm.RoleFFNGate) || m.Stores(li, jlm.RoleExpGateBank)
			gated[li] = gate || !m.Stores(li, jlm.RoleFFNUp) && !m.Stores(li, jlm.RoleExpUpBank)
		}
		if err := convert.StateGated(&s, gated); err != nil {
			panic(err)
		}
		c.Acts, c.ActLayers = s.Acts, s.ActLayers
		undo := restate(m, func(map[jlm.Role]graph.Shape) {})
		return func() {
			c.Acts, c.ActLayers = was, wasL
			undo()
		}
	}
}

// statedMut runs f over each stated activation's terms, a copy, until it
// changes one, and has the front end state the blocks again.
func statedMut(f func(*jlm.Term) bool) func(*model.Model) func() {
	return statedExprMut(func(ts []jlm.Term) {
		for k := range ts {
			if f(&ts[k]) {
				break
			}
		}
	})
}

// statedExprMut is statedMut with f given each stated expression's terms.
func statedExprMut(f func([]jlm.Term)) func(*model.Model) func() {
	return func(m *model.Model) func() {
		c := m.Container.Config()
		was := c.Acts
		acts := make([]jlm.Expr, len(was))
		for i, e := range was {
			acts[i].Terms = slices.Clone(e.Terms)
			f(acts[i].Terms)
		}
		c.Acts = acts
		undo := restate(m, func(map[jlm.Role]graph.Shape) {})
		return func() {
			c.Acts = was
			undo()
		}
	}
}
