package convert

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// The FFN activations, stated in the op language (jlm.Config.Acts): the
// formula each family's reference computes, written once here, where the
// family is known, so the engine reads an expression and no activation's
// name. A gated block's expression is of its gate (operand 0) and up
// (operand 1); an ungated one's of its up alone.

// expr builds an expression term by term; each method is a term's value.
type expr struct{ t []jlm.Term }

func (e *expr) fn(f jlm.Fn, args ...jlm.Arg) jlm.Arg {
	var t jlm.Term
	t.Fn = f
	copy(t.Args[:], args)
	e.t = append(e.t, t)
	return jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(e.t) - 1)}
}

// sum is a over the whole row, one value a row.
func (e *expr) sum(a jlm.Arg) jlm.Arg {
	e.t = append(e.t, jlm.Term{Fn: jlm.FnAdd, Reduce: true, Tree: true, Args: [3]jlm.Arg{a}})
	return jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(e.t) - 1)}
}

func operand(i uint32) jlm.Arg { return jlm.Arg{Kind: jlm.ArgIn, Index: i} }

func num(v float32) jlm.Arg { return jlm.Arg{Kind: jlm.ArgConst, Value: v} }

func storedAt(r jlm.Role, i uint32) jlm.Arg { return jlm.Arg{Kind: jlm.ArgStored, Role: r, Index: i} }

func (e *expr) silu(a jlm.Arg) jlm.Arg {
	return e.fn(jlm.FnDiv, a, e.fn(jlm.FnAdd, num(1), e.fn(jlm.FnExp, e.fn(jlm.FnSub, num(0), a))))
}

// geluTanh is 0.5 a (1 + tanh(sqrt(2/pi) (a + 0.044715 a^3))).
func (e *expr) geluTanh(a jlm.Arg) jlm.Arg {
	cube := e.fn(jlm.FnMul, num(0.044715), e.fn(jlm.FnMul, a, e.fn(jlm.FnMul, a, a)))
	th := e.fn(jlm.FnTanh, e.fn(jlm.FnMul, num(0.7978845608028654), e.fn(jlm.FnAdd, a, cube)))
	return e.fn(jlm.FnMul, e.fn(jlm.FnMul, num(0.5), a), e.fn(jlm.FnAdd, num(1), th))
}

// geluErf is 0.5 a (1 + erf(a / sqrt(2))).
func (e *expr) geluErf(a jlm.Arg) jlm.Arg {
	erf := e.fn(jlm.FnErf, e.fn(jlm.FnMul, a, num(0.70710678118654752)))
	return e.fn(jlm.FnMul, e.fn(jlm.FnMul, num(0.5), a), e.fn(jlm.FnAdd, num(1), erf))
}

func (e *expr) relu2(a jlm.Arg) jlm.Arg {
	r := e.fn(jlm.FnMax, a, num(0))
	return e.fn(jlm.FnMul, r, r)
}

// quickGELU is a sigma(1.702 a).
func (e *expr) quickGELU(a jlm.Arg) jlm.Arg {
	return e.fn(jlm.FnDiv, a, e.fn(jlm.FnAdd, num(1), e.fn(jlm.FnExp, e.fn(jlm.FnSub, num(0), e.fn(jlm.FnMul, num(1.702), a)))))
}

// xielu is Apertus's xIELU from the four numbers each block stores
// (jlm.RoleXIELU: alpha_p, alpha_n, beta, eps, folded at conversion):
//
//	a > 0:   alpha_p*a*a + beta*a
//	a <= 0:  (expm1(min(a, eps)) - a)*alpha_n + beta*a
func (e *expr) xielu(a jlm.Arg) jlm.Arg {
	alphaP, alphaN, beta, eps := storedAt(jlm.RoleXIELU, 0), storedAt(jlm.RoleXIELU, 1), storedAt(jlm.RoleXIELU, 2), storedAt(jlm.RoleXIELU, 3)
	neg := e.fn(jlm.FnMul, e.fn(jlm.FnSub, e.fn(jlm.FnExpm1, e.fn(jlm.FnMin, a, eps)), a), alphaN)
	pos := e.fn(jlm.FnMul, e.fn(jlm.FnMul, a, alphaP), a)
	return e.fn(jlm.FnAdd, e.fn(jlm.FnWhere, a, pos, neg), e.fn(jlm.FnMul, a, beta))
}

// gaussTopK is Gemma 3n's sparse gate: what lies std standard deviations
// above the row's mean, less that cutoff, the deviation over the row's
// width (torch's std, unbiased=False).
func (e *expr) gaussTopK(a jlm.Arg, std float32) jlm.Arg {
	width := jlm.Arg{Kind: jlm.ArgWidth}
	mean := e.fn(jlm.FnDiv, e.sum(a), width)
	d := e.fn(jlm.FnSub, a, mean)
	sd := e.fn(jlm.FnSqrt, e.fn(jlm.FnDiv, e.sum(e.fn(jlm.FnMul, d, d)), width))
	return e.fn(jlm.FnMax, e.fn(jlm.FnSub, a, e.fn(jlm.FnFma, sd, num(std), mean)), num(0))
}

func (e *expr) clamp(a jlm.Arg, lim float32) jlm.Arg {
	return e.fn(jlm.FnMax, e.fn(jlm.FnMin, a, num(lim)), num(-lim))
}

// bounded is c tanh(a/c), Kimi-K3's situ bound.
func (e *expr) bounded(a jlm.Arg, c float32) jlm.Arg {
	return e.fn(jlm.FnMul, num(c), e.fn(jlm.FnTanh, e.fn(jlm.FnMul, num(1/c), a)))
}

// Activation is the FFN activation the converter states for block li of a
// model of config c, gated or not.
func Activation(c *jlm.Config, li int, gated bool) (jlm.Expr, error) {
	var e expr
	g, u := operand(0), operand(1)
	var unary func(jlm.Arg) jlm.Arg
	switch {
	case c.Flags2.Has(jlm.Flag2SwiGLUClamp):
		// DeepSeek V4: silu(min(gate, 10)) * clamp(up, -10, 10).
		e.fn(jlm.FnMul, e.silu(e.fn(jlm.FnMin, g, num(10))), e.clamp(u, 10))
	case c.Flags.Has(jlm.FlagSwiGLUOAI):
		// gpt-oss: quickGELU(min(gate, 7)) * (clamp(up, -7, 7) + 1).
		e.fn(jlm.FnMul, e.quickGELU(e.fn(jlm.FnMin, g, num(7))), e.fn(jlm.FnAdd, e.clamp(u, 7), num(1)))
	case c.SituBeta != 0 || c.SituLinearBeta != 0:
		e.fn(jlm.FnMul, e.fn(jlm.FnMul, e.bounded(g, c.SituBeta), e.fn(jlm.FnSigmoid, g)), e.bounded(u, c.SituLinearBeta))
	case c.Flags.Has(jlm.FlagReLU2):
		unary = e.relu2
	case c.Arch == jlm.ArchModernBERT:
		unary = e.geluErf
	case c.Flags.Has(jlm.FlagGELU) && c.SparseStd != 0 && li < int(c.NSparse):
		unary = func(a jlm.Arg) jlm.Arg { return e.geluTanh(e.gaussTopK(a, c.SparseStd)) }
	case c.Flags.Has(jlm.FlagGELU):
		unary = e.geluTanh
	case c.Arch == jlm.ArchApertus:
		unary = e.xielu
	default:
		unary = e.silu
	}
	switch {
	case unary == nil && !gated:
		return jlm.Expr{}, fmt.Errorf("convert: block %d's activation is a gated one, and the block stores no gate", li)
	case unary != nil && gated:
		e.fn(jlm.FnMul, unary(g), u)
	case unary != nil:
		unary(g)
	}
	x := jlm.Expr{Terms: e.t}
	return x, x.Valid()
}

// StateActivations states every layer's FFN activation in s's config: one
// expression a distinct activation, and each layer's index into them where
// the layers differ. A layer is gated unless it stores an up and no gate.
func StateActivations(s *jlm.Source) error {
	c := s.Config
	if c == nil || c.NLayer == 0 || c.Arch == jlm.ArchCLIP {
		return nil
	}
	n := int(c.NLayer)
	up, gate := make([]bool, n), make([]bool, n)
	for _, t := range s.Tensors {
		if t.Block < 0 || int(t.Block) >= n {
			continue
		}
		switch t.Role {
		case jlm.RoleFFNUp:
			up[t.Block] = true
		case jlm.RoleFFNGate:
			gate[t.Block] = true
		}
	}
	return StateGated(c, gating(n, func(li int) bool { return gate[li] }, func(li int) bool { return up[li] }))
}

// StateStored states the FFN activations of a container written before
// they were stated, from its config and the tensors it stores, as
// StateActivations states a source's; false for one that states them or has
// no layers (jlm.Restate).
func StateStored(f *jlm.File, c *jlm.Config) (bool, error) {
	if len(c.Acts) != 0 && len(c.RouterScore.Terms) == len(RouterScore(c).Terms) || c.NLayer == 0 || c.Arch == jlm.ArchCLIP {
		return false, nil
	}
	stores := func(r jlm.Role) func(int) bool {
		return func(li int) bool {
			_, ok := f.Find(r, int32(li), -1)
			return ok
		}
	}
	return true, StateGated(c, gating(int(c.NLayer), stores(jlm.RoleFFNGate), stores(jlm.RoleFFNUp)))
}

// gating is each of n layers' gating: gated unless it stores an up and no
// gate.
func gating(n int, gate, up func(li int) bool) []bool {
	g := make([]bool, n)
	for li := range n {
		g[li] = gate(li) || !up(li)
	}
	return g
}

// StateGated states every layer's FFN activation in c, layer li gated where
// gated[li] is.
func StateGated(c *jlm.Config, gated []bool) error {
	n := int(c.NLayer)
	if len(gated) != n {
		return fmt.Errorf("convert: the gating of %d layers stated for %d", len(gated), n)
	}
	c.Acts, c.ActLayers = nil, make([]uint8, n)
	for li := range n {
		e, err := Activation(c, li, gated[li])
		if err != nil {
			return err
		}
		i := slices.IndexFunc(c.Acts, func(a jlm.Expr) bool { return slices.Equal(a.Terms, e.Terms) })
		if i < 0 {
			if len(c.Acts) == 255 {
				return fmt.Errorf("convert: more than 255 activations")
			}
			i, c.Acts = len(c.Acts), append(c.Acts, e)
		}
		c.ActLayers[li] = uint8(i)
	}
	if len(c.Acts) == 1 {
		c.ActLayers = nil
	}
	if c.RouterScore = RouterScore(c); len(c.RouterScore.Terms) != 0 {
		return c.RouterScore.Valid()
	}
	return nil
}

// RouterScore is the score a mixture of config c's router gives each expert
// from its logits (operand 0), the formula its family's reference computes:
// what the selection ranks and the routed weights are read from. None for a
// model with no mixture, or one whose routed weights are not the selected
// scores (Phi-3.5-MoE's sparsemixer weighs each pick by a softmax over the
// logits near it).
func RouterScore(c *jlm.Config) jlm.Expr {
	if c.NExpert == 0 || c.Arch == jlm.ArchPhiMoE {
		return jlm.Expr{}
	}
	var e expr
	x := operand(0)
	switch {
	case c.Flags2.Has(jlm.Flag2ExpertSqrtSoftplus):
		e.fn(jlm.FnSqrt, e.fn(jlm.FnSoftplus, x))
	case c.Flags.Has(jlm.FlagExpertSigmoid):
		e.fn(jlm.FnSigmoid, x)
	default:
		// softmax over the experts: exp(x - max x), over its sum.
		m := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(e.t))}
		e.t = append(e.t, jlm.Term{Fn: jlm.FnMax, Reduce: true, Args: [3]jlm.Arg{x}})
		ex := e.fn(jlm.FnExp, e.fn(jlm.FnSub, x, m))
		s := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(e.t))}
		e.t = append(e.t, jlm.Term{Fn: jlm.FnAdd, Reduce: true, Args: [3]jlm.Arg{ex}})
		e.fn(jlm.FnDiv, ex, s)
	}
	return jlm.Expr{Terms: e.t}
}
