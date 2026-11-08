package jlm

import "fmt"

// Fn is a function of the op language: an elementwise map of an expression's
// term, or a reduction's combiner. The codes are this format's, and the block
// graph reads them as its own (graph.MapFn).
type Fn uint8

const (
	FnAdd      Fn = iota + 1 // a + b
	FnSub                    // a - b
	FnMul                    // a * b
	FnDiv                    // a / b
	FnMax                    // max(a, b)
	FnMin                    // min(a, b)
	FnExp                    // exp(a)
	FnSigmoid                // 1 / (1 + exp(-a))
	FnSoftplus               // log(1 + exp(a))
	FnSqrt                   // sqrt(a)
	FnWhere                  // b where a > 0, else c
	FnLess                   // 1 where a < b, else 0
	FnFma                    // a*b + c, the product not rounded before the sum
	FnExpm1                  // exp(a) - 1, exact near zero where the difference cancels
	FnAbs                    // |a|
	FnRound                  // a to the nearest integer, halves away from zero
	FnRecip                  // 1 / a
	FnTanh                   // tanh(a)
	FnErf                    // erf(a)
	FnEqual                  // 1 where a == b, else 0
	FnLog                    // the natural logarithm of a
	numFns
)

// Arity is how many operands f takes.
func (f Fn) Arity() int {
	switch f {
	case FnExp, FnSigmoid, FnSoftplus, FnSqrt, FnExpm1, FnAbs, FnRound, FnRecip, FnTanh, FnErf, FnLog:
		return 1
	case FnWhere, FnFma:
		return 3
	case FnAdd, FnSub, FnMul, FnDiv, FnMax, FnMin, FnLess, FnEqual:
		return 2
	}
	return 0
}

func (f Fn) String() string {
	if f == 0 {
		return "?"
	}
	switch f {
	case FnAdd:
		return "add"
	case FnSub:
		return "sub"
	case FnMul:
		return "mul"
	case FnDiv:
		return "div"
	case FnMax:
		return "max"
	case FnMin:
		return "min"
	case FnExp:
		return "exp"
	case FnSigmoid:
		return "sigmoid"
	case FnSoftplus:
		return "softplus"
	case FnSqrt:
		return "sqrt"
	case FnWhere:
		return "where"
	case FnLess:
		return "less"
	case FnFma:
		return "fma"
	case FnExpm1:
		return "expm1"
	case FnAbs:
		return "abs"
	case FnRound:
		return "round"
	case FnRecip:
		return "recip"
	case FnTanh:
		return "tanh"
	case FnErf:
		return "erf"
	case FnEqual:
		return "equal"
	case FnLog:
		return "log"
	}
	return fmt.Sprintf("fn%d", f)
}

// Expr is a function the container states in the op language, a block's
// activation: Terms in order, each a function of the expression's operands,
// constants, a stored vector's elements and earlier terms. Its value is its
// last term's.
type Expr struct {
	Terms []Term
}

// Term is one step of an Expr. A map applies Fn to Fn.Arity() Args, element
// by element; a reduction (Reduce) combines Args[0] over the whole row by Fn
// (FnAdd, FnMax or FnMin), pairwise where Tree, from the first element
// otherwise, one value a row that the terms after it read at every element.
type Term struct {
	Fn     Fn
	Reduce bool
	Tree   bool
	Args   [3]Arg
}

// ArgKind is what a term's operand is.
type ArgKind uint8

const (
	ArgNone ArgKind = iota
	// ArgIn is the expression's operand Index: a gated activation's gate is
	// 0 and its up 1, an ungated one's input 0.
	ArgIn
	// ArgTerm is the value of the earlier term Index.
	ArgTerm
	// ArgConst is Value.
	ArgConst
	// ArgStored is element Index of the block's stored vector Role.
	ArgStored
	// ArgWidth is the row's width, as a constant: a mean's divisor.
	ArgWidth
)

// Arg is a term's operand.
type Arg struct {
	Kind  ArgKind
	Index uint32
	Role  Role
	Value float32
}

// MaxExprIn is the operands an expression reads at most.
const MaxExprIn = 2

func (a Arg) String() string {
	switch a.Kind {
	case ArgIn:
		return fmt.Sprintf("in%d", a.Index)
	case ArgTerm:
		return fmt.Sprintf("t%d", a.Index)
	case ArgConst:
		return fmt.Sprintf("%g", a.Value)
	case ArgStored:
		return fmt.Sprintf("%v[%d]", a.Role, a.Index)
	case ArgWidth:
		return "width"
	}
	return "?"
}

// Valid reports why e is not an expression this format defines, nil when it
// is one.
func (e *Expr) Valid() error {
	if len(e.Terms) == 0 {
		return fmt.Errorf("jlm: an expression of no terms")
	}
	for k, t := range e.Terms {
		n := t.Fn.Arity()
		if t.Reduce {
			if t.Fn != FnAdd && t.Fn != FnMax && t.Fn != FnMin {
				return fmt.Errorf("jlm: term %d reduces by %v", k, t.Fn)
			}
			if a := t.Args[0].Kind; a != ArgIn && a != ArgTerm {
				return fmt.Errorf("jlm: term %d reduces a %d operand", k, a)
			}
			n = 1
		} else if t.Tree {
			return fmt.Errorf("jlm: term %d is a map in a reduction's order", k)
		}
		if n == 0 {
			return fmt.Errorf("jlm: term %d has function %d, which this format does not define", k, t.Fn)
		}
		for j, a := range t.Args {
			switch {
			case j >= n && a.Kind != ArgNone:
				return fmt.Errorf("jlm: term %d's %v takes %d operands, not %d", k, t.Fn, n, j+1)
			case j >= n:
			case a.Kind == ArgIn && a.Index >= MaxExprIn:
				return fmt.Errorf("jlm: term %d reads operand %d", k, a.Index)
			case a.Kind == ArgTerm && int(a.Index) >= k:
				return fmt.Errorf("jlm: term %d reads term %d, not an earlier one", k, a.Index)
			case a.Kind == ArgNone || a.Kind > ArgWidth:
				return fmt.Errorf("jlm: term %d's operand %d is of kind %d", k, j, a.Kind)
			}
		}
	}
	if e.Terms[len(e.Terms)-1].Reduce {
		return fmt.Errorf("jlm: an expression whose value is a reduction")
	}
	return nil
}

// Reads reports how many operands e reads: one past the highest ArgIn.
func (e *Expr) Reads() int {
	n := 0
	for _, t := range e.Terms {
		for _, a := range t.Args {
			if a.Kind == ArgIn && int(a.Index) >= n {
				n = int(a.Index) + 1
			}
		}
	}
	return n
}

// ActAt is the activation the container states for block li's FFN, nil
// where it states none.
func (c *Config) ActAt(li int) *Expr {
	switch {
	case len(c.Acts) == 0:
		return nil
	case len(c.ActLayers) == 0:
		return &c.Acts[0]
	case li < 0 || li >= len(c.ActLayers):
		return nil
	}
	return &c.Acts[c.ActLayers[li]]
}

func encodeExprs(w *wbuf, c *Config) {
	w.u32(uint32(len(c.Acts)))
	for i := range c.Acts {
		encodeExpr(w, &c.Acts[i])
	}
	w.u32(uint32(len(c.ActLayers)))
	for _, i := range c.ActLayers {
		w.u8(i)
	}
}

func encodeExpr(w *wbuf, e *Expr) {
	w.u32(uint32(len(e.Terms)))
	for _, t := range e.Terms {
		w.u8(uint8(t.Fn))
		var fl uint8
		if t.Reduce {
			fl |= 1
		}
		if t.Tree {
			fl |= 2
		}
		w.u8(fl)
		for _, a := range t.Args {
			w.u8(uint8(a.Kind))
			switch a.Kind {
			case ArgIn, ArgTerm:
				w.u32(a.Index)
			case ArgConst:
				w.f32(a.Value)
			case ArgStored:
				w.u16(uint16(a.Role))
				w.u32(a.Index)
			}
		}
	}
}

func decodeExprs(r *rbuf, c *Config) {
	if n := r.count(4); n > 0 {
		c.Acts = make([]Expr, n)
		for i := range c.Acts {
			decodeExpr(r, &c.Acts[i])
		}
	}
	if m := r.count(1); m > 0 {
		c.ActLayers = make([]uint8, m)
		for i := range c.ActLayers {
			c.ActLayers[i] = r.u8()
		}
	}
}

func decodeExpr(r *rbuf, e *Expr) {
	if n := r.count(5); n > 0 {
		e.Terms = make([]Term, n)
	}
	for k := range e.Terms {
		t := &e.Terms[k]
		t.Fn = Fn(r.u8())
		fl := r.u8()
		t.Reduce, t.Tree = fl&1 != 0, fl&2 != 0
		for j := range t.Args {
			a := &t.Args[j]
			a.Kind = ArgKind(r.u8())
			switch a.Kind {
			case ArgIn, ArgTerm:
				a.Index = r.u32()
			case ArgConst:
				a.Value = r.f32()
			case ArgStored:
				a.Role = Role(r.u16())
				a.Index = r.u32()
			}
		}
	}
}

// exprsValid reports why c's stated activations are not ones this format
// defines.
func exprsValid(c *Config) error {
	for i := range c.Acts {
		if err := c.Acts[i].Valid(); err != nil {
			return fmt.Errorf("activation %d: %w", i, err)
		}
	}
	if n := uint32(len(c.ActLayers)); n != 0 && n != c.NLayer && n != c.NLayer+c.DecisionBlocks {
		return fmt.Errorf("jlm: %d activation indices for %d layers and %d head blocks", n, c.NLayer, c.DecisionBlocks)
	}
	for li, i := range c.ActLayers {
		if int(i) >= len(c.Acts) {
			return fmt.Errorf("jlm: layer %d states activation %d of %d", li, i, len(c.Acts))
		}
	}
	if len(c.RouterScore.Terms) != 0 {
		if err := c.RouterScore.Valid(); err != nil {
			return fmt.Errorf("the router's score: %w", err)
		}
		if n := c.RouterScore.Reads(); n != 1 {
			return fmt.Errorf("jlm: the router's score reads %d operands, not its logits alone", n)
		}
	}
	if len(c.RouterWeight.Terms) != 0 {
		if len(c.RouterScore.Terms) == 0 {
			return fmt.Errorf("jlm: a router's weight stated with no score to select by")
		}
		if err := c.RouterWeight.Valid(); err != nil {
			return fmt.Errorf("the router's weight: %w", err)
		}
		if n := c.RouterWeight.Reads(); n != 2 {
			return fmt.Errorf("jlm: the router's weight reads %d operands, not the logits and the pick's", n)
		}
	}
	return nil
}
