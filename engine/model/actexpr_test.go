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

// gatedBy is the terms of act(in0) * in1 where gated, act(in0) where not.
func gatedBy(act []graph.Term, gated bool) []graph.Term {
	if !gated {
		return act
	}
	return append(slices.Clone(act), term(graph.MapMul, argTerm(len(act)-1), argIn(1)))
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
		vs = append(vs, c6Violation{"the stated activation's first constant moved", statedMut(func(t *jlm.Term) bool {
			for j := range t.Args {
				if t.Args[j].Kind == jlm.ArgConst {
					t.Args[j].Value++
					return true
				}
			}
			return false
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
	return func(m *Model) func() {
		c := m.container.Config()
		was := c.Acts
		acts := make([]jlm.Expr, len(was))
		for i, e := range was {
			acts[i].Terms = slices.Clone(e.Terms)
			for k := range acts[i].Terms {
				if f(&acts[i].Terms[k]) {
					break
				}
			}
		}
		c.Acts = acts
		undo := m.restate(func(map[jlm.Role]graph.Shape) {})
		return func() {
			c.Acts = was
			undo()
		}
	}
}
