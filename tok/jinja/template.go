package jinja

import (
	"fmt"
	"strings"
)

// Template is a compiled Jinja template ready for rendering.
type Template struct {
	nodes    []node
	src      string
	builtins *scope
}

// Compile parses a Jinja template source string and returns a compiled
// template that can be rendered multiple times with different data.
func Compile(source string) (*Template, error) {
	segs, err := scan(source)
	if err != nil {
		return nil, fmt.Errorf("jinja scan: %w", err)
	}

	nodes, err := parseTemplate(segs)
	if err != nil {
		return nil, fmt.Errorf("jinja parse: %w", err)
	}

	return &Template{nodes: nodes, src: source, builtins: newBuiltins()}, nil
}

// Render executes the template with the provided data and returns the
// rendered string. Each call creates an isolated scope so templates are
// safe to render concurrently after compilation.
func (t *Template) Render(data map[string]any) (string, error) {
	// Convert Go data to Value types.
	vals := make(map[string]Value, len(data))
	for k, v := range data {
		vals[k] = FromGoValue(v)
	}

	return render(t.nodes, t.builtins, vals)
}

// Expr is a compiled Jinja expression: what a model's statement computes, a
// value rather than text.
type Expr struct {
	ex       expr
	src      string
	builtins *scope
}

// CompileExpr parses one Jinja expression.
func CompileExpr(source string) (*Expr, error) {
	ex, err := parseExpr(source)
	if err != nil {
		return nil, fmt.Errorf("jinja expression %q: %w", source, err)
	}
	return &Expr{ex: ex, src: source, builtins: newBuiltins()}, nil
}

// Names are the variables the expression reads, each once: a caller that
// knows its variables refuses a name it does not hold rather than reading it
// as undefined.
func (x *Expr) Names() []string {
	var out []string
	seen := map[string]bool{}
	var walk func(ex expr)
	walk = func(ex expr) {
		switch n := ex.(type) {
		case *nameExpr:
			if !seen[n.name] {
				seen[n.name] = true
				out = append(out, n.name)
			}
		case *attrExpr:
			walk(n.obj)
		case *itemExpr:
			walk(n.obj)
			walk(n.key)
		case *sliceExpr:
			for _, e := range []expr{n.obj, n.start, n.stop, n.step} {
				if e != nil {
					walk(e)
				}
			}
		case *unaryExpr:
			walk(n.operand)
		case *binExpr:
			walk(n.left)
			walk(n.right)
		case *condExpr:
			walk(n.trueExpr)
			walk(n.condition)
			if n.falseExpr != nil {
				walk(n.falseExpr)
			}
		case *callExpr:
			walk(n.fn)
			for _, a := range n.args {
				walk(a)
			}
			for _, k := range n.kwargs {
				walk(k.value)
			}
		case *filterExpr:
			walk(n.expr)
			for _, a := range n.args {
				walk(a)
			}
			for _, k := range n.kwargs {
				walk(k.value)
			}
		case *testExpr:
			walk(n.expr)
			for _, a := range n.args {
				walk(a)
			}
		case *listExpr:
			for _, a := range n.items {
				walk(a)
			}
		case *dictExpr:
			for _, p := range n.pairs {
				walk(p.key)
				walk(p.value)
			}
		}
	}
	walk(x.ex)
	return out
}

// Eval is the expression's value over vars, in at most steps steps.
func (x *Expr) Eval(vars map[string]Value, steps int) (Value, error) {
	e := evaluator{builtins: x.builtins, steps: &steps}
	e.scope = newScope(e.builtins)
	for k, v := range vars {
		e.scope.set(k, v)
	}
	v, err := e.evalExpr(x.ex)
	if err != nil {
		return Undefined(), fmt.Errorf("jinja expression %q: %w", x.src, err)
	}
	return v, nil
}

// =============================================================================

// render executes the AST with the given data and returns the result.
func render(nodes []node, builtins *scope, data map[string]Value) (string, error) {
	steps := MaxSteps
	e := evaluator{
		output: strings.Builder{},
		steps:  &steps,
	}

	// The template's builtins scope is only read after Compile (lookupFilter,
	// chained get() through parent links, passed to nested evaluators), so
	// sharing it across renders is safe and avoids cloning the entries per
	// render.
	e.builtins = builtins

	// Create user data scope on top of builtins.
	e.scope = newScope(e.builtins)
	for k, v := range data {
		e.scope.set(k, v)
	}

	if err := e.execNodes(nodes); err != nil {
		return "", err
	}

	return e.output.String(), nil
}
