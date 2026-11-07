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
type expr struct{ jlm.ExprBuilder }

func operand(i uint32) jlm.Arg { return jlm.Operand(i) }

func num(v float32) jlm.Arg { return jlm.Num(v) }

func storedAt(r jlm.Role, i uint32) jlm.Arg { return jlm.Arg{Kind: jlm.ArgStored, Role: r, Index: i} }

func (e *expr) relu2(a jlm.Arg) jlm.Arg {
	r := e.Fn(jlm.FnMax, a, num(0))
	return e.Fn(jlm.FnMul, r, r)
}

// xielu is Apertus's xIELU from the four numbers each block stores
// (jlm.RoleXIELU: alpha_p, alpha_n, beta, eps, folded at conversion):
//
//	a > 0:   alpha_p*a*a + beta*a
//	a <= 0:  (expm1(min(a, eps)) - a)*alpha_n + beta*a
func (e *expr) xielu(a jlm.Arg) jlm.Arg {
	alphaP, alphaN, beta, eps := storedAt(jlm.RoleXIELU, 0), storedAt(jlm.RoleXIELU, 1), storedAt(jlm.RoleXIELU, 2), storedAt(jlm.RoleXIELU, 3)
	neg := e.Fn(jlm.FnMul, e.Fn(jlm.FnSub, e.Fn(jlm.FnExpm1, e.Fn(jlm.FnMin, a, eps)), a), alphaN)
	pos := e.Fn(jlm.FnMul, e.Fn(jlm.FnMul, a, alphaP), a)
	return e.Fn(jlm.FnAdd, e.Fn(jlm.FnWhere, a, pos, neg), e.Fn(jlm.FnMul, a, beta))
}

// gaussTopK is Gemma 3n's sparse gate: what lies std standard deviations
// above the row's mean, less that cutoff, the deviation over the row's
// width (torch's std, unbiased=False).
func (e *expr) gaussTopK(a jlm.Arg, std float32) jlm.Arg {
	width := jlm.Arg{Kind: jlm.ArgWidth}
	mean := e.Fn(jlm.FnDiv, e.Sum(a), width)
	d := e.Fn(jlm.FnSub, a, mean)
	sd := e.Fn(jlm.FnSqrt, e.Fn(jlm.FnDiv, e.Sum(e.Fn(jlm.FnMul, d, d)), width))
	return e.Fn(jlm.FnMax, e.Fn(jlm.FnSub, a, e.Fn(jlm.FnFma, sd, num(std), mean)), num(0))
}

func (e *expr) clamp(a jlm.Arg, lim float32) jlm.Arg {
	return e.Fn(jlm.FnMax, e.Fn(jlm.FnMin, a, num(lim)), num(-lim))
}

// bounded is c tanh(a/c), Kimi-K3's situ bound.
func (e *expr) bounded(a jlm.Arg, c float32) jlm.Arg {
	return e.Fn(jlm.FnMul, num(c), e.Fn(jlm.FnTanh, e.Fn(jlm.FnMul, num(1/c), a)))
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
		e.Fn(jlm.FnMul, e.SiLU(e.Fn(jlm.FnMin, g, num(10))), e.clamp(u, 10))
	case c.Flags.Has(jlm.FlagSwiGLUOAI):
		// gpt-oss: quickGELU(min(gate, 7)) * (clamp(up, -7, 7) + 1).
		e.Fn(jlm.FnMul, e.QuickGELU(e.Fn(jlm.FnMin, g, num(7))), e.Fn(jlm.FnAdd, e.clamp(u, 7), num(1)))
	case c.SituBeta != 0 || c.SituLinearBeta != 0:
		e.Fn(jlm.FnMul, e.Fn(jlm.FnMul, e.bounded(g, c.SituBeta), e.Fn(jlm.FnSigmoid, g)), e.bounded(u, c.SituLinearBeta))
	case c.Flags.Has(jlm.FlagReLU2):
		unary = e.relu2
	case c.Arch == jlm.ArchModernBERT:
		unary = e.GELUErf
	case c.Flags.Has(jlm.FlagGELU) && c.SparseStd != 0 && li < int(c.NSparse):
		unary = func(a jlm.Arg) jlm.Arg { return e.GELUTanh(e.gaussTopK(a, c.SparseStd)) }
	case c.Flags.Has(jlm.FlagGELU):
		unary = e.GELUTanh
	case c.Arch == jlm.ArchApertus:
		unary = e.xielu
	default:
		unary = e.SiLU
	}
	switch {
	case unary == nil && !gated:
		return jlm.Expr{}, fmt.Errorf("convert: block %d's activation is a gated one, and the block stores no gate", li)
	case unary != nil && gated:
		e.Fn(jlm.FnMul, unary(g), u)
	case unary != nil:
		unary(g)
	}
	x := jlm.Expr{Terms: e.T}
	return x, x.Valid()
}

// StateActivations states every layer's FFN activation in s's config: one
// expression a distinct activation, and each layer's index into them where
// the layers differ. A layer is gated unless it stores an up and no gate
// (a dense one or an expert bank).
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
		case jlm.RoleFFNUp, jlm.RoleExpUpBank:
			up[t.Block] = true
		case jlm.RoleFFNGate, jlm.RoleExpGateBank:
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
	streams := (c.AttnResBlock == 0 && c.HCLowRank == 0 || c.Streams.N > 1) && (c.NGramBlock == 0 || StatesPre(c))
	rev := c.Revision >= jlm.StatementRevision
	sel := c.IdxHeads == 0 || c.Selection != nil
	if len(c.Acts) != 0 && len(c.RouterScore.Terms) == len(RouterScore(c).Terms) && len(c.RouterWeight.Terms) == len(RouterWeight(c).Terms) && streams && rev && sel ||
		c.NLayer == 0 || c.Arch == jlm.ArchCLIP {
		return false, nil
	}
	if !streams {
		StateStreams(c, func(r jlm.Role, li int32) bool {
			_, ok := f.Find(r, li, -1)
			return ok
		})
	}
	if !rev {
		StateRevision(c)
	}
	if !sel {
		if err := StateSelection(c, func(r jlm.Role, li int32) bool {
			_, ok := f.Find(r, li, -1)
			return ok
		}); err != nil {
			return false, err
		}
	}
	stores := func(rs ...jlm.Role) func(int) bool {
		return func(li int) bool {
			for _, r := range rs {
				if _, ok := f.Find(r, int32(li), -1); ok {
					return true
				}
			}
			return false
		}
	}
	return true, StateGated(c, gating(int(c.NLayer), stores(jlm.RoleFFNGate, jlm.RoleExpGateBank), stores(jlm.RoleFFNUp, jlm.RoleExpUpBank)))
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
	c.RouterScore, c.RouterWeight = RouterScore(c), RouterWeight(c)
	for _, e := range []*jlm.Expr{&c.RouterScore, &c.RouterWeight} {
		if len(e.Terms) != 0 {
			if err := e.Valid(); err != nil {
				return err
			}
		}
	}
	return nil
}

// RouterScore is the score a mixture of config c's router gives each expert
// from its logits (operand 0), the formula its family's reference computes:
// what the selection ranks, and the routed weights are read from where the
// config states no RouterWeight. None for a model with no mixture.
func RouterScore(c *jlm.Config) jlm.Expr {
	if c.NExpert == 0 {
		return jlm.Expr{}
	}
	var e expr
	x := operand(0)
	switch {
	case c.Arch == jlm.ArchPhiMoE:
		// The raw logits: sparsemixer picks by them (RouterWeight).
		e.Fn(jlm.FnMul, x, num(1))
	case c.Flags2.Has(jlm.Flag2ExpertSqrtSoftplus):
		e.Fn(jlm.FnSqrt, e.Fn(jlm.FnSoftplus, x))
	case c.Flags.Has(jlm.FlagExpertSigmoid):
		e.Fn(jlm.FnSigmoid, x)
	default:
		// softmax over the experts: exp(x - max x), over its sum.
		m := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(e.T))}
		e.T = append(e.T, jlm.Term{Fn: jlm.FnMax, Reduce: true, Args: [3]jlm.Arg{x}})
		ex := e.Fn(jlm.FnExp, e.Fn(jlm.FnSub, x, m))
		s := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(e.T))}
		e.T = append(e.T, jlm.Term{Fn: jlm.FnAdd, Reduce: true, Args: [3]jlm.Arg{ex}})
		e.Fn(jlm.FnDiv, ex, s)
	}
	return jlm.Expr{Terms: e.T}
}

// RouterWeight is the routed weight of a mixture of config c whose weights
// are not its selected scores, over operand 0, the logits with the earlier
// picks removed, and operand 1, the pick's logit m: Phi-3.5-MoE's
// sparsemixer (transformers' PhimoeTopKRouter in eval), the softmax over the
// logits s_j kept where (m - s_j)/max(|s_j|, m) is not past twice its
// jitter, read at the pick. None for every other model.
func RouterWeight(c *jlm.Config) jlm.Expr {
	if c.NExpert == 0 || c.Arch != jlm.ArchPhiMoE {
		return jlm.Expr{}
	}
	return SparseMixerWeight(jlm.PhiMoEJitter)
}

// SparseMixerWeight is sparsemixer's routed weight at jitter eps
// (RouterWeight).
func SparseMixerWeight(eps float32) jlm.Expr {
	var e expr
	s, m := operand(0), operand(1)
	ratio := e.Fn(jlm.FnDiv, e.Fn(jlm.FnSub, m, s), e.Fn(jlm.FnMax, e.Fn(jlm.FnAbs, s), m))
	out := e.Fn(jlm.FnLess, num(float32(2*float64(eps))), ratio)
	ex := e.Fn(jlm.FnWhere, out, num(0), e.Fn(jlm.FnExp, e.Fn(jlm.FnSub, s, m)))
	e.Fn(jlm.FnDiv, ex, e.Sum(ex))
	return jlm.Expr{Terms: e.T}
}

// StatesPre reports that c's streams state an addition before some block's
// attention (an n-gram block's).
func StatesPre(c *jlm.Config) bool {
	for i := range c.Streams.Blocks {
		if c.Streams.Blocks[i].Pre.Read != nil {
			return true
		}
	}
	return false
}
