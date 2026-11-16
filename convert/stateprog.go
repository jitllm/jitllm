package convert

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/gather"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// stateProg builds a stated program -- ops of the tower op language
// (towerOps) by name -- in its statement's environment: the expressions'
// values (value, num), the facts an each binds its variable in, the role a
// weight's name binds, and the op lists a call names (defs). where names the
// statement in messages.
type stateProg struct {
	where string
	value func(string) (jinja.Value, error)
	num   func(string) (float64, error)
	facts map[string]jinja.Value
	role  func(string) (jlm.Role, bool)
	defs  map[string][]mjTowerOp
}

// prog is ops as a program over nin inputs, held saying which optional
// roles the statement's tensors carry.
func (sp *stateProg) prog(ops []mjTowerOp, nin int, held func(string) bool) (jlm.Program, error) {
	p, _, err := sp.progNamed(ops, nin, held)
	return p, err
}

// progNamed is prog and the values its ops name: in, in1 ... for the
// inputs, and each op's Out.
func (sp *stateProg) progNamed(ops []mjTowerOp, nin int, held func(string) bool) (jlm.Program, map[string]uint16, error) {
	p := jlm.Program{NIn: uint16(nin)}
	named := map[string]uint16{}
	for i := range max(nin, 2) {
		named[inName(i)] = uint16(i)
	}
	last := uint16(0)
	if err := sp.run(&p, ops, held, named, &last); err != nil {
		return p, nil, err
	}
	p.Out = last
	return p, named, nil
}

// run appends ops to p, last the value the next op of no In reads.
func (sp *stateProg) run(p *jlm.Program, ops []mjTowerOp, held func(string) bool, named map[string]uint16, last *uint16) error {
	nin := int(p.NIn)
	for _, o := range ops {
		if o.If != "" {
			on, err := sp.value(o.If)
			if err != nil {
				return err
			}
			if !on.IsTruthy() {
				continue
			}
		}
		// A let names a value and writes no op; a call runs a def's ops over
		// its operands, named in, in1 ... in the def, its value the last op's.
		if o.Op == "let" || o.Op == "call" {
			ins, err := sp.operands(p, o, named, *last)
			if err != nil {
				return err
			}
			if o.Op == "let" {
				if len(ins) != 1 {
					return fmt.Errorf("convert: %s: a let names one value, not %d", sp.where, len(ins))
				}
				*last = ins[0]
			} else {
				def, ok := sp.defs[o.Def]
				if !ok {
					return fmt.Errorf("convert: %s: a call of %q, which no def states", sp.where, o.Def)
				}
				local := map[string]uint16{}
				for i, v := range ins {
					local[inName(i)] = v
				}
				if err := sp.run(p, def, held, local, last); err != nil {
					return err
				}
			}
			if o.Out != "" {
				named[o.Out] = *last
			}
			continue
		}
		if o.Op == "each" {
			from, err := sp.num(o.Start)
			if err != nil {
				return err
			}
			n, err := sp.num(o.Count)
			if err != nil {
				return err
			}
			if n > 1<<16 {
				return fmt.Errorf("convert: %s: an each of %v runs", sp.where, n)
			}
			outer, had := sp.facts[o.Var]
			for i := int64(from); i < int64(n); i++ {
				sp.facts[o.Var] = jinja.NewInt(i)
				if err := sp.run(p, o.Ops, held, named, last); err != nil {
					return err
				}
			}
			if had {
				sp.facts[o.Var] = outer
			} else {
				delete(sp.facts, o.Var)
			}
			continue
		}
		op := jlm.TowerOp{Code: towerOps()[o.Op]}
		if o.Op == "untile" {
			merge, err := sp.num(o.Span)
			if err != nil {
				return err
			}
			window, err := sp.num(o.Width)
			if err != nil {
				return err
			}
			if op, err = gather.Untile(int(merge), int(window)); err != nil {
				return fmt.Errorf("convert: %s: %w", sp.where, err)
			}
		}
		// A clip mask's grid width, samples a frame and halvings.
		if o.Op == "clip_mask" {
			var w, st, h float64
			for _, f := range []struct {
				src string
				to  *float64
			}{{o.GridW, &w}, {o.Stride, &st}, {o.Span, &h}} {
				x, err := sp.num(f.src)
				if err != nil {
					return err
				}
				*f.to = x
			}
			var err error
			if op, err = gather.ClipMask(int(w), int(st), int(h)); err != nil {
				return fmt.Errorf("convert: %s: %w", sp.where, err)
			}
		}
		ins, err := sp.operands(p, o, named, *last)
		if err != nil {
			return err
		}
		op.In = ins
		for _, w := range o.W {
			n, optional := strings.CutSuffix(w, "?")
			if at, ok := strings.CutPrefix(n, "@"); ok {
				v, err := sp.value(at)
				if err != nil {
					return err
				}
				if !v.IsString() {
					return fmt.Errorf("convert: %s: op %s's weight %q is %v, not a name", sp.where, o.Op, at, v)
				}
				n = v.AsString()
			}
			if optional && !held(n) {
				continue
			}
			r, ok := sp.role(n)
			if !ok {
				return fmt.Errorf("convert: %s: op %s binds %q, which is no tensor the container names", sp.where, o.Op, n)
			}
			op.Bind = append(op.Bind, r)
		}
		for _, f := range []struct {
			src string
			to  *int32
		}{{o.Span, &op.Span}, {o.Offset, &op.Offset}, {o.Width, &op.Width}, {o.GridW, &op.GridW}, {o.Stride, &op.Stride}} {
			if o.Op == "untile" || o.Op == "clip_mask" {
				break
			}
			x, err := sp.num(f.src)
			if err != nil {
				return err
			}
			*f.to = int32(x)
		}
		eps, err := sp.num(o.Eps)
		if err != nil {
			return err
		}
		op.Eps = float32(eps)
		op.Chain, op.Filter = towerChains()[o.Chain], towerTaps()[o.Taps]
		if o.Fn != "" {
			c, err := sp.num(o.Const)
			if err != nil {
				return err
			}
			var e jlm.ExprBuilder
			towerFns()[o.Fn](&e, float32(c))
			op.Expr = &jlm.Expr{Terms: e.T}
		}
		if o.Expr != "" {
			x, err := statedExpr(o.Expr, map[string]uint32{"x": 0, "y": 1, "z": 2}, sp.num)
			if err != nil {
				return fmt.Errorf("convert: %s: %w", sp.where, err)
			}
			op.Expr = &x
		}
		p.Ops = append(p.Ops, op)
		*last = uint16(nin + len(p.Ops) - 1)
		if o.Out != "" {
			named[o.Out] = *last
		}
	}
	return nil
}

// progsCheck checks programs' ops: each an op of the tower op language, extra
// or an each, a call naming one of defs, a function or a resample the
// container knows, and each weight not an @ and an expression a name role
// accepts. It returns the expressions the ops read and the variables their
// eaches bind.
func progsCheck(progs [][]mjTowerOp, extra []string, defs map[string][]mjTowerOp, role func(string) error) (srcs, vars []string, err error) {
	for i := 0; i < len(progs); i++ {
		for _, o := range progs[i] {
			if o.Op == "each" {
				if o.Var == "" || o.Count == "" || len(o.Ops) == 0 {
					return nil, nil, fmt.Errorf("an each states its variable, its count and its ops")
				}
				progs = append(progs, o.Ops)
				srcs = append(srcs, o.If, o.Start, o.Count)
				vars = append(vars, o.Var)
				continue
			}
			if _, ok := defs[o.Def]; o.Op == "call" && !ok {
				return nil, nil, fmt.Errorf("a call of %q, which no def states", o.Def)
			}
			if o.Expr != "" {
				x, err := parseExpr(o.Expr)
				if err != nil {
					return nil, nil, err
				}
				srcs = append(srcs, exprPaths(x, "x", "y", "z")...)
			}
			for _, n := range o.In {
				if at, ok := strings.CutPrefix(n, "@"); ok {
					srcs = append(srcs, at)
				}
			}
			if _, ok := towerOps()[o.Op]; !ok && !slices.Contains(extra, o.Op) {
				return nil, nil, fmt.Errorf("op %q is not one the program runs", o.Op)
			}
			if _, ok := towerFns()[o.Fn]; o.Fn != "" && !ok {
				return nil, nil, fmt.Errorf("function %q is not one a stated op runs", o.Fn)
			}
			if o.Fn != "" && o.Expr != "" {
				return nil, nil, fmt.Errorf("a stated op states its function or its expression, not both")
			}
			if _, ok := towerChains()[o.Chain]; o.Chain != "" && !ok {
				return nil, nil, fmt.Errorf("resample chain %q is not aligned or half_pixel", o.Chain)
			}
			if _, ok := towerTaps()[o.Taps]; o.Taps != "" && !ok {
				return nil, nil, fmt.Errorf("resample taps %q are not ones the container names", o.Taps)
			}
			for _, w := range o.W {
				if at, ok := strings.CutPrefix(strings.TrimSuffix(w, "?"), "@"); ok {
					srcs = append(srcs, at)
					continue
				}
				if err := role(w); err != nil {
					return nil, nil, err
				}
			}
			srcs = append(srcs, o.If, o.Span, o.Offset, o.Width, o.Eps, o.Const, o.GridW, o.Stride)
		}
	}
	return srcs, vars, nil
}

// operands are o's inputs: values by name, or by number where a name is an
// @ and an expression; the last value where o names none.
func (sp *stateProg) operands(p *jlm.Program, o mjTowerOp, named map[string]uint16, last uint16) ([]uint16, error) {
	if o.In == nil {
		return []uint16{last}, nil
	}
	var out []uint16
	for _, n := range o.In {
		if at, ok := strings.CutPrefix(n, "@"); ok {
			x, err := sp.num(at)
			if err != nil {
				return nil, err
			}
			if x < 0 || int(x) >= int(p.NIn)+len(p.Ops) {
				return nil, fmt.Errorf("convert: %s: op %s reads value %v of %d", sp.where, o.Op, x, int(p.NIn)+len(p.Ops))
			}
			out = append(out, uint16(x))
			continue
		}
		i, ok := named[n]
		if !ok {
			return nil, fmt.Errorf("convert: %s: op %s reads %q, which no earlier op names", sp.where, o.Op, n)
		}
		out = append(out, i)
	}
	return out, nil
}

// inName is input i's name in a program or a def: in, in1, in2 ...
func inName(i int) string {
	if i == 0 {
		return "in"
	}
	return "in" + itoa(i)
}

// exprFns are the functions a stated op's expression calls, the
// container's.
func exprFns() map[string]jlm.Fn {
	return map[string]jlm.Fn{"exp": jlm.FnExp, "log": jlm.FnLog, "sqrt": jlm.FnSqrt, "tanh": jlm.FnTanh,
		"sigmoid": jlm.FnSigmoid, "max": jlm.FnMax, "min": jlm.FnMin, "abs": jlm.FnAbs}
}

// exprPaths are the names of x's paths that are not among operands: the
// numbers it reads.
func exprPaths(x *exprNode, operands ...string) []string {
	var out []string
	var walk func(n *exprNode)
	walk = func(n *exprNode) {
		if n.op == "path" {
			for _, o := range operands {
				if n.str == o {
					return
				}
			}
			out = append(out, n.str)
			return
		}
		for _, a := range n.args {
			walk(a)
		}
	}
	walk(x)
	return out
}

// statedExpr lowers src to the container's terms over operands, each term
// in the order a post-order walk reaches it, as an ExprBuilder writes it
// called inside out; a path not an operand is a number num reads, and a
// subexpression of no operand is folded to its number in float64 and written
// once as a float32, as a Go builder computes a constant before it.
// gelu_tanh, gelu_erf and silu are the builder's own forms.
func statedExpr(src string, operands map[string]uint32, num func(string) (float64, error)) (jlm.Expr, error) {
	x, err := parseExpr(src)
	if err != nil {
		return jlm.Expr{}, err
	}
	var e jlm.ExprBuilder
	bin := map[string]jlm.Fn{"+": jlm.FnAdd, "-": jlm.FnSub, "*": jlm.FnMul, "/": jlm.FnDiv}
	type val struct {
		arg   jlm.Arg
		konst bool
		f     float64
	}
	var lower func(n *exprNode) (val, error)
	lowerAll := func(ns []*exprNode) ([]val, bool, error) {
		vs := make([]val, len(ns))
		all := true
		for i, a := range ns {
			v, err := lower(a)
			if err != nil {
				return nil, false, err
			}
			vs[i], all = v, all && v.konst
		}
		return vs, all, nil
	}
	args := func(vs []val) []jlm.Arg {
		out := make([]jlm.Arg, len(vs))
		for i, v := range vs {
			out[i] = v.arg
			if v.konst {
				out[i] = jlm.Num(float32(v.f))
			}
		}
		return out
	}
	lower = func(n *exprNode) (val, error) {
		switch n.op {
		case "num":
			return val{konst: true, f: n.num}, nil
		case "path":
			if i, ok := operands[n.str]; ok {
				return val{arg: jlm.Operand(i)}, nil
			}
			f, err := num(n.str)
			return val{konst: true, f: f}, err
		case "neg":
			vs, all, err := lowerAll(n.args)
			if err != nil {
				return val{}, err
			}
			if all {
				return val{konst: true, f: -vs[0].f}, nil
			}
			return val{arg: e.Fn(jlm.FnSub, jlm.Num(0), args(vs)[0])}, nil
		case "call":
			vs, all, err := lowerAll(n.args)
			if err != nil {
				return val{}, err
			}
			switch n.str {
			case "gelu_tanh", "gelu_erf", "silu":
				if len(vs) != 1 {
					return val{}, fmt.Errorf("%s takes one operand", n.str)
				}
				a := args(vs)[0]
				switch n.str {
				case "gelu_tanh":
					return val{arg: e.GELUTanh(a)}, nil
				case "gelu_erf":
					return val{arg: e.GELUErf(a)}, nil
				}
				return val{arg: e.SiLU(a)}, nil
			}
			f, ok := exprFns()[n.str]
			if !ok {
				return val{}, fmt.Errorf("%s is no function a stated op calls", n.str)
			}
			if all {
				fs := make([]float64, len(vs))
				for i, v := range vs {
					fs[i] = v.f
				}
				if fold, ok := exprFolds()[n.str]; ok && len(fs) > 0 {
					return val{konst: true, f: fold(fs)}, nil
				}
			}
			return val{arg: e.Fn(f, args(vs)...)}, nil
		}
		f, ok := bin[n.op]
		if !ok || len(n.args) != 2 {
			return val{}, fmt.Errorf("%q is no operation a stated op writes", n.op)
		}
		vs, all, err := lowerAll(n.args)
		if err != nil {
			return val{}, err
		}
		if all {
			l, r := vs[0].f, vs[1].f
			c := map[string]float64{"+": l + r, "-": l - r, "*": l * r, "/": l / r}[n.op]
			return val{konst: true, f: c}, nil
		}
		return val{arg: e.Fn(f, args(vs)...)}, nil
	}
	v, err := lower(x)
	if err != nil {
		return jlm.Expr{}, err
	}
	if v.konst {
		return jlm.Expr{}, fmt.Errorf("%q reads no operand", src)
	}
	return jlm.Expr{Terms: e.T}, nil
}

// exprFolds are exprFns of constants, in float64.
func exprFolds() map[string]func([]float64) float64 {
	return map[string]func([]float64) float64{
		"exp": func(a []float64) float64 { return math.Exp(a[0]) }, "log": func(a []float64) float64 { return math.Log(a[0]) },
		"sqrt": func(a []float64) float64 { return math.Sqrt(a[0]) }, "tanh": func(a []float64) float64 { return math.Tanh(a[0]) },
		"abs": func(a []float64) float64 { return math.Abs(a[0]) },
		"max": func(a []float64) float64 { return math.Max(a[0], a[len(a)-1]) },
		"min": func(a []float64) float64 { return math.Min(a[0], a[len(a)-1]) },
	}
}
