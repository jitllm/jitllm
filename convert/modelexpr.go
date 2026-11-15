package convert

import (
	"fmt"
	"strconv"
	"strings"
)

// A model.json expression reads a config.json: numbers, strings, booleans,
// lists, null, paths into the config (a.b.c from the file's root, $.a.b from
// its top), arithmetic (+ - * /), comparisons (== != < <= > >=, in), logic
// (&& || !) and a ?? b, which is b where a is absent. There are no loops and
// no calls, so an expression sizes nothing (RULE 9), and its depth is bounded.
//
// An absent key is null; null in arithmetic or an ordering is an error that
// names the key, so a default is written, never assumed.

const exprMaxDepth = 64

type exprNode struct {
	op   string // "num", "str", "bool", "null", "path", "list", or an operator
	num  float64
	str  string
	b    bool
	args []*exprNode
}

type exprParser struct {
	src   string
	toks  []string
	i     int
	depth int
}

func parseExpr(src string) (*exprNode, error) {
	toks, err := lexExpr(src)
	if err != nil {
		return nil, err
	}
	p := &exprParser{src: src, toks: toks}
	n, err := p.coalesce()
	if err != nil {
		return nil, err
	}
	if p.i != len(p.toks) {
		return nil, fmt.Errorf("expression %q: %q after its end", src, p.toks[p.i])
	}
	return n, nil
}

func lexExpr(src string) ([]string, error) {
	var toks []string
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n':
			i++
		case c == '\'':
			j := strings.IndexByte(src[i+1:], '\'')
			if j < 0 {
				return nil, fmt.Errorf("expression %q: a string is not closed", src)
			}
			toks = append(toks, src[i:i+j+2])
			i += j + 2
		case c >= '0' && c <= '9' || c == '.' && i+1 < len(src) && src[i+1] >= '0' && src[i+1] <= '9':
			j := i
			for j < len(src) && (src[j] >= '0' && src[j] <= '9' || src[j] == '.' || src[j] == 'e' || src[j] == 'E' ||
				(src[j] == '-' || src[j] == '+') && j > i && (src[j-1] == 'e' || src[j-1] == 'E')) {
				j++
			}
			toks = append(toks, src[i:j])
			i = j
		case c == '_' || c == '$' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z':
			j := i
			for j < len(src) && (src[j] == '_' || src[j] == '$' || src[j] == '.' || src[j] >= 'a' && src[j] <= 'z' ||
				src[j] >= 'A' && src[j] <= 'Z' || src[j] >= '0' && src[j] <= '9') {
				j++
			}
			toks = append(toks, src[i:j])
			i = j
		default:
			if i+1 < len(src) {
				switch two := src[i : i+2]; two {
				case "??", "||", "&&", "==", "!=", "<=", ">=":
					toks = append(toks, two)
					i += 2
					continue
				}
			}
			if !strings.ContainsRune("+-*/<>!()[],", rune(c)) {
				return nil, fmt.Errorf("expression %q: %q is not in the language", src, c)
			}
			toks = append(toks, string(c))
			i++
		}
	}
	return toks, nil
}

func (p *exprParser) peek() string {
	if p.i < len(p.toks) {
		return p.toks[p.i]
	}
	return ""
}

func (p *exprParser) enter() error {
	if p.depth++; p.depth > exprMaxDepth {
		return fmt.Errorf("expression %q nests deeper than %d", p.src, exprMaxDepth)
	}
	return nil
}

// binary parses next (op next)* for the operators in ops, left to right.
func (p *exprParser) binary(next func() (*exprNode, error), ops ...string) (*exprNode, error) {
	l, err := next()
	if err != nil {
		return nil, err
	}
	for {
		op := p.peek()
		found := false
		for _, o := range ops {
			found = found || op == o
		}
		if !found {
			return l, nil
		}
		p.i++
		r, err := next()
		if err != nil {
			return nil, err
		}
		l = &exprNode{op: op, args: []*exprNode{l, r}}
	}
}

func (p *exprParser) coalesce() (*exprNode, error) {
	if err := p.enter(); err != nil {
		return nil, err
	}
	defer func() { p.depth-- }()
	return p.binary(p.or, "??")
}

func (p *exprParser) or() (*exprNode, error)  { return p.binary(p.and, "||") }
func (p *exprParser) and() (*exprNode, error) { return p.binary(p.not, "&&") }

func (p *exprParser) not() (*exprNode, error) {
	if p.peek() != "!" {
		return p.cmp()
	}
	p.i++
	if err := p.enter(); err != nil {
		return nil, err
	}
	defer func() { p.depth-- }()
	x, err := p.not()
	if err != nil {
		return nil, err
	}
	return &exprNode{op: "!", args: []*exprNode{x}}, nil
}

func (p *exprParser) cmp() (*exprNode, error) {
	l, err := p.sum()
	if err != nil {
		return nil, err
	}
	switch op := p.peek(); op {
	case "==", "!=", "<", "<=", ">", ">=", "in":
		p.i++
		r, err := p.sum()
		if err != nil {
			return nil, err
		}
		return &exprNode{op: op, args: []*exprNode{l, r}}, nil
	}
	return l, nil
}

func (p *exprParser) sum() (*exprNode, error)  { return p.binary(p.prod, "+", "-") }
func (p *exprParser) prod() (*exprNode, error) { return p.binary(p.unary, "*", "/") }

func (p *exprParser) unary() (*exprNode, error) {
	if p.peek() != "-" {
		return p.atom()
	}
	p.i++
	if err := p.enter(); err != nil {
		return nil, err
	}
	defer func() { p.depth-- }()
	x, err := p.unary()
	if err != nil {
		return nil, err
	}
	return &exprNode{op: "neg", args: []*exprNode{x}}, nil
}

func (p *exprParser) atom() (*exprNode, error) {
	t := p.peek()
	if t == "" {
		return nil, fmt.Errorf("expression %q ends where a value belongs", p.src)
	}
	p.i++
	switch {
	case t == "(":
		x, err := p.coalesce()
		if err != nil {
			return nil, err
		}
		if p.peek() != ")" {
			return nil, fmt.Errorf("expression %q: a parenthesis is not closed", p.src)
		}
		p.i++
		return x, nil
	case t == "[":
		n := &exprNode{op: "list"}
		for p.peek() != "]" {
			if len(n.args) > 0 {
				if p.peek() != "," {
					return nil, fmt.Errorf("expression %q: a list wants a comma, not %q", p.src, p.peek())
				}
				p.i++
			}
			x, err := p.coalesce()
			if err != nil {
				return nil, err
			}
			n.args = append(n.args, x)
		}
		p.i++
		return n, nil
	case t[0] == '\'':
		return &exprNode{op: "str", str: t[1 : len(t)-1]}, nil
	case t == "true" || t == "false":
		return &exprNode{op: "bool", b: t == "true"}, nil
	case t == "null":
		return &exprNode{op: "null"}, nil
	case t[0] >= '0' && t[0] <= '9' || t[0] == '.':
		v, err := strconv.ParseFloat(t, 64)
		if err != nil {
			return nil, fmt.Errorf("expression %q: %q is not a number", p.src, t)
		}
		return &exprNode{op: "num", num: v}, nil
	case t == "in":
		return nil, fmt.Errorf("expression %q: \"in\" where a value belongs", p.src)
	default:
		if strings.HasPrefix(t, ".") || strings.HasSuffix(t, ".") || strings.Contains(t, "..") {
			return nil, fmt.Errorf("expression %q: %q is not a path", p.src, t)
		}
		return &exprNode{op: "path", str: t}, nil
	}
}

// exprEnv resolves a path to its value: float64, string, bool, []any, or nil
// for an absent key.
type exprEnv func(path string) any

func evalMJ(n *exprNode, env exprEnv) (any, error) {
	switch n.op {
	case "num":
		return n.num, nil
	case "str":
		return n.str, nil
	case "bool":
		return n.b, nil
	case "null":
		return nil, nil
	case "path":
		return env(n.str), nil
	case "list":
		out := make([]any, len(n.args))
		for i, a := range n.args {
			v, err := evalMJ(a, env)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case "??":
		l, err := evalMJ(n.args[0], env)
		if err != nil || l != nil {
			return l, err
		}
		return evalMJ(n.args[1], env)
	case "&&", "||":
		l, err := evalBool(n.args[0], env)
		if err != nil || l == (n.op == "||") {
			return l, err
		}
		return evalBool(n.args[1], env)
	case "!":
		v, err := evalBool(n.args[0], env)
		return !v, err
	case "neg":
		v, err := evalNum(n.args[0], env)
		return -v, err
	case "==", "!=":
		l, err := evalMJ(n.args[0], env)
		if err != nil {
			return nil, err
		}
		r, err := evalMJ(n.args[1], env)
		if err != nil {
			return nil, err
		}
		eq, err := exprEqual(l, r)
		return eq == (n.op == "=="), err
	case "in":
		l, err := evalMJ(n.args[0], env)
		if err != nil {
			return nil, err
		}
		r, err := evalMJ(n.args[1], env)
		if err != nil {
			return nil, err
		}
		list, ok := r.([]any)
		if !ok {
			return nil, fmt.Errorf("%s: \"in\" wants a list on its right", exprText(n.args[1]))
		}
		for _, x := range list {
			if eq, err := exprEqual(l, x); err != nil || eq {
				return eq, err
			}
		}
		return false, nil
	}
	l, err := evalNum(n.args[0], env)
	if err != nil {
		return nil, err
	}
	r, err := evalNum(n.args[1], env)
	if err != nil {
		return nil, err
	}
	switch n.op {
	case "+":
		return l + r, nil
	case "-":
		return l - r, nil
	case "*":
		return l * r, nil
	case "/":
		if r == 0 {
			return nil, fmt.Errorf("%s: division by zero", exprText(n))
		}
		return l / r, nil
	case "<":
		return l < r, nil
	case "<=":
		return l <= r, nil
	case ">":
		return l > r, nil
	case ">=":
		return l >= r, nil
	}
	return nil, fmt.Errorf("operator %q", n.op)
}

func evalNum(n *exprNode, env exprEnv) (float64, error) {
	v, err := evalMJ(n, env)
	if err != nil {
		return 0, err
	}
	f, ok := v.(float64)
	if !ok {
		return 0, fmt.Errorf("%s is %s, not a number", exprText(n), exprKind(v))
	}
	return f, nil
}

func evalBool(n *exprNode, env exprEnv) (bool, error) {
	v, err := evalMJ(n, env)
	if err != nil {
		return false, err
	}
	b, ok := v.(bool)
	if !ok {
		return false, fmt.Errorf("%s is %s, not a boolean", exprText(n), exprKind(v))
	}
	return b, nil
}

// exprEqual compares scalars; null equals only null.
func exprEqual(a, b any) (bool, error) {
	if a == nil || b == nil {
		return a == nil && b == nil, nil
	}
	switch a.(type) {
	case float64, string, bool:
	default:
		return false, fmt.Errorf("%s compared with ==", exprKind(a))
	}
	return a == b, nil
}

func exprKind(v any) string {
	switch v.(type) {
	case nil:
		return "absent"
	case float64:
		return "a number"
	case string:
		return "a string"
	case bool:
		return "a boolean"
	case []any:
		return "a list"
	}
	return fmt.Sprintf("%T", v)
}

// exprText names a node for an error: the path where it is one.
func exprText(n *exprNode) string {
	if n.op == "path" {
		return n.str
	}
	return "the " + n.op + " expression"
}
