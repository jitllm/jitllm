package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
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
var outGateTerms = []graph.Term{term(graph.MapSigmoid, argIn(0)), term(graph.MapMul, argTerm(0), argIn(1))}

// isOutGate reports whether o is the attention's output gate.
func isOutGate(o graph.Op) bool {
	return o.Kind == graph.OpExpr && slices.Equal(o.Attr.Expr, outGateTerms)
}

// sigmoidGate is sigmoid(g) * v as an op of b.
func sigmoidGate(b *graph.Builder, g, v graph.ValueID, width int) graph.ValueID {
	return b.Expr("gated", width, outGateTerms, nil, g, v)
}

// reluSquaredTerms and siluTerms are two activations a fault swaps one for
// the other.
var (
	reluSquaredTerms = []graph.Term{term(graph.MapMax, argIn(0), argK(0)), term(graph.MapMul, argTerm(0), argTerm(0))}
	siluTerms        = []graph.Term{term(graph.MapSub, argK(0), argIn(0)), term(graph.MapExp, argTerm(0)),
		term(graph.MapAdd, argK(1), argTerm(1)), term(graph.MapDiv, argIn(0), argTerm(2))}
)

// statedViolations are m's stated activations taken apart where the composer
// reads them, the host computing the clean model: a constant moved, a
// select's branches swapped. A model the front end states in part gets none,
// since the blocks it leaves out never read them.
func statedViolations(m *Model) []c6Violation {
	if coveredBlocks(m) != m.Cfg.NLayer {
		return nil
	}
	var consts, wheres bool
	for _, e := range m.container.Config().Acts {
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

// actSwap runs every block under activation k: in the model's config, which
// main's kernels read, and stated by the converter, as a container of that
// activation states it, for the front end and the composer.
func actSwap(k nn.ActKind) func(*Model) func() {
	return func(m *Model) func() {
		undoC := cfgMut(m, func(c *Config) { c.Act = k })
		c := m.container.Config()
		was, wasL := c.Acts, c.ActLayers
		s := *c
		s.Arch = jlm.ArchLlama
		s.Flags &^= jlm.FlagGELU | jlm.FlagReLU2 | jlm.FlagSwiGLUOAI
		s.Flags2 &^= jlm.Flag2SwiGLUClamp
		s.SituBeta, s.SituLinearBeta, s.SparseStd = 0, 0, 0
		switch k {
		case nn.ActGELU:
			s.Flags |= jlm.FlagGELU
		case nn.ActReLU2:
			s.Flags |= jlm.FlagReLU2
		case nn.ActSiLU:
		default:
			panic(fmt.Sprintf("actSwap: no statement for %v", k))
		}
		gated := make([]bool, c.NLayer)
		for li := range gated {
			gated[li] = m.stores(li, jlm.RoleFFNGate) || !m.stores(li, jlm.RoleFFNUp)
		}
		if err := convert.StateGated(&s, gated); err != nil {
			panic(err)
		}
		c.Acts, c.ActLayers = s.Acts, s.ActLayers
		undo := m.restate(func(map[jlm.Role]graph.Shape) {})
		return func() {
			c.Acts, c.ActLayers = was, wasL
			undo()
			undoC()
		}
	}
}

// statedMut runs f over each stated activation's terms, a copy, until it
// changes one, and has the front end state the blocks again.
func statedMut(f func(*jlm.Term) bool) func(*Model) func() {
	return statedExprMut(func(ts []jlm.Term) {
		for k := range ts {
			if f(&ts[k]) {
				break
			}
		}
	})
}

// statedExprMut is statedMut with f given each stated expression's terms.
func statedExprMut(f func([]jlm.Term)) func(*Model) func() {
	return func(m *Model) func() {
		c := m.container.Config()
		was := c.Acts
		acts := make([]jlm.Expr, len(was))
		for i, e := range was {
			acts[i].Terms = slices.Clone(e.Terms)
			f(acts[i].Terms)
		}
		c.Acts = acts
		undo := m.restate(func(map[jlm.Role]graph.Shape) {})
		return func() {
			c.Acts = was
			undo()
		}
	}
}
