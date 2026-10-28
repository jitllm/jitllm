package jlm

import "errors"

// errVisionActs is a tower whose flags ask for two activations at once.
var errVisionActs = errors.New("jlm: the vision tower asks for GELU and quick-GELU at once")

// ExprBuilder builds an expression term by term; each method is a term's
// value. The converter states a text block's activation with it
// (convert.Activation), and a vision tower's is read through it from the
// tower's flags (Vision.Activation).
type ExprBuilder struct{ T []Term }

// Fn is the term f of args.
func (e *ExprBuilder) Fn(f Fn, args ...Arg) Arg {
	var t Term
	t.Fn = f
	copy(t.Args[:], args)
	e.T = append(e.T, t)
	return Arg{Kind: ArgTerm, Index: uint32(len(e.T) - 1)}
}

// Sum is a over the whole row, one value a row.
func (e *ExprBuilder) Sum(a Arg) Arg {
	e.T = append(e.T, Term{Fn: FnAdd, Reduce: true, Tree: true, Args: [3]Arg{a}})
	return Arg{Kind: ArgTerm, Index: uint32(len(e.T) - 1)}
}

// Operand is the expression's operand i.
func Operand(i uint32) Arg { return Arg{Kind: ArgIn, Index: i} }

// Num is the constant v.
func Num(v float32) Arg { return Arg{Kind: ArgConst, Value: v} }

// SiLU is a sigma(a).
func (e *ExprBuilder) SiLU(a Arg) Arg {
	return e.Fn(FnDiv, a, e.Fn(FnAdd, Num(1), e.Fn(FnExp, e.Fn(FnSub, Num(0), a))))
}

// GELUTanh is 0.5 a (1 + tanh(sqrt(2/pi) (a + 0.044715 a^3))).
func (e *ExprBuilder) GELUTanh(a Arg) Arg {
	cube := e.Fn(FnMul, Num(0.044715), e.Fn(FnMul, a, e.Fn(FnMul, a, a)))
	th := e.Fn(FnTanh, e.Fn(FnMul, Num(0.7978845608028654), e.Fn(FnAdd, a, cube)))
	return e.Fn(FnMul, e.Fn(FnMul, Num(0.5), a), e.Fn(FnAdd, Num(1), th))
}

// GELUErf is 0.5 a (1 + erf(a / sqrt(2))).
func (e *ExprBuilder) GELUErf(a Arg) Arg {
	erf := e.Fn(FnErf, e.Fn(FnMul, a, Num(0.70710678118654752)))
	return e.Fn(FnMul, e.Fn(FnMul, Num(0.5), a), e.Fn(FnAdd, Num(1), erf))
}

// QuickGELU is a sigma(1.702 a).
func (e *ExprBuilder) QuickGELU(a Arg) Arg {
	return e.Fn(FnDiv, a, e.Fn(FnAdd, Num(1), e.Fn(FnExp, e.Fn(FnSub, Num(0), e.Fn(FnMul, Num(1.702), a)))))
}

// Activation is the tower's MLP activation its flags state: GELU's tanh
// form, quick-GELU, or SiLU where it states neither; of its gate (operand 0)
// and up (operand 1) where gated, of its up alone where not.
func (v *Vision) Activation(gated bool) (Expr, error) {
	var e ExprBuilder
	unary := e.SiLU
	switch {
	case v.Flags.Has(FlagGELU) && v.Flags.Has(FlagQuickGELU):
		return Expr{}, errVisionActs
	case v.Flags.Has(FlagGELU):
		unary = e.GELUTanh
	case v.Flags.Has(FlagQuickGELU):
		unary = e.QuickGELU
	}
	if gated {
		e.Fn(FnMul, unary(Operand(0)), Operand(1))
	} else {
		unary(Operand(0))
	}
	x := Expr{Terms: e.T}
	return x, x.Valid()
}
