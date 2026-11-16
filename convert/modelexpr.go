package convert

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// A model.json expression reads a config.json: numbers, strings, booleans,
// lists, null, paths into the config (a.b.c from the file's root, $.a.b from
// its top), arithmetic (+ - * /), comparisons (== != < <= > >=, in), logic
// (&& || !), a ?? b, which is b where a is absent or null, c ? a : b, a ** b,
// and a key of a parenthesised value, (a ?? b).key. There are no loops, and
// calls (exp, log, sqrt, tanh) are a tensor map's alone, bar uniform(x) and
// the layer lists' each_layer(n, members, first, inside, outside) and
// partitions(n, first, a, b), whose counts are bounded, so an expression
// sizes nothing unchecked (RULE 9), and its depth is bounded.
//
// An absent key is null; null in arithmetic or an ordering is an error that
// names the key, so a default is written, never assumed.
//
// a ?? b with both sides bare config paths says they are one key spelled two
// ways: both present and disagreeing is an error, as there is no knowing which
// the weights were trained under. A parenthesised side states precedence
// instead -- a ?? (b) is a, and b only where a is absent -- which is how a
// default read from another key is written.

const exprMaxDepth = 64

type exprNode struct {
	op   string // "num", "str", "bool", "null", "path", "list", or an operator
	num  float64
	str  string
	b    bool
	args []*exprNode
	// paren marks a parenthesised node: a ?? (b) states precedence, not a
	// second spelling.
	paren bool
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
	n, err := p.ternary()
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
		case c == '.' && len(toks) > 0 && (toks[len(toks)-1] == ")" || toks[len(toks)-1] == "]"):
			j := i + 1
			for j < len(src) && (src[j] == '_' || src[j] == '.' || src[j] >= 'a' && src[j] <= 'z' ||
				src[j] >= 'A' && src[j] <= 'Z' || src[j] >= '0' && src[j] <= '9') {
				j++
			}
			if j == i+1 {
				return nil, fmt.Errorf("expression %q: a . wants a key", src)
			}
			toks = append(toks, src[i:j])
			i = j
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
				case "??", "||", "&&", "==", "!=", "<=", ">=", "**":
					toks = append(toks, two)
					i += 2
					continue
				}
			}
			if !strings.ContainsRune("+-*/<>!()[],?:", rune(c)) {
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

// ternary is c ? a : b, the loosest operator.
func (p *exprParser) ternary() (*exprNode, error) {
	c, err := p.coalesce()
	if err != nil || p.peek() != "?" {
		return c, err
	}
	p.i++
	if err := p.enter(); err != nil {
		return nil, err
	}
	defer func() { p.depth-- }()
	a, err := p.ternary()
	if err != nil {
		return nil, err
	}
	if p.peek() != ":" {
		return nil, fmt.Errorf("expression %q: a ? wants a : , not %q", p.src, p.peek())
	}
	p.i++
	b, err := p.ternary()
	if err != nil {
		return nil, err
	}
	return &exprNode{op: "?:", args: []*exprNode{c, a, b}}, nil
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
		return p.pow()
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

// pow is a ** b, right-associative and tighter than a unary minus on its left.
func (p *exprParser) pow() (*exprNode, error) {
	l, err := p.atom()
	if err != nil || p.peek() != "**" {
		return l, err
	}
	p.i++
	if err := p.enter(); err != nil {
		return nil, err
	}
	defer func() { p.depth-- }()
	r, err := p.unary()
	if err != nil {
		return nil, err
	}
	return &exprNode{op: "**", args: []*exprNode{l, r}}, nil
}

// member follows a parenthesised value or a list with the keys it is read by.
func (p *exprParser) member(n *exprNode) *exprNode {
	for strings.HasPrefix(p.peek(), ".") {
		n = &exprNode{op: "member", str: p.peek()[1:], args: []*exprNode{n}}
		p.i++
	}
	return n
}

func (p *exprParser) atom() (*exprNode, error) {
	t := p.peek()
	if t == "" {
		return nil, fmt.Errorf("expression %q ends where a value belongs", p.src)
	}
	p.i++
	switch {
	case t == "(":
		x, err := p.ternary()
		if err != nil {
			return nil, err
		}
		if p.peek() != ")" {
			return nil, fmt.Errorf("expression %q: a parenthesis is not closed", p.src)
		}
		p.i++
		x.paren = true
		if strings.HasPrefix(p.peek(), ".") {
			return p.member(x), nil
		}
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
			x, err := p.ternary()
			if err != nil {
				return nil, err
			}
			n.args = append(n.args, x)
		}
		p.i++
		return p.member(n), nil
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
		if p.peek() == "(" && !strings.ContainsAny(t, ".$") {
			p.i++
			if err := p.enter(); err != nil {
				return nil, err
			}
			defer func() { p.depth-- }()
			n := &exprNode{op: "call", str: t}
			for p.peek() != ")" {
				if p.peek() == "" {
					return nil, fmt.Errorf("expression %q: a call is not closed", p.src)
				}
				if len(n.args) > 0 {
					if p.peek() != "," {
						return nil, fmt.Errorf("expression %q: a call wants a comma, not %q", p.src, p.peek())
					}
					p.i++
				}
				x, err := p.ternary()
				if err != nil {
					return nil, err
				}
				n.args = append(n.args, x)
			}
			p.i++
			return n, nil
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
		if err != nil {
			return nil, err
		}
		if l == nil {
			return evalMJ(n.args[1], env)
		}
		if a, b := n.args[0], n.args[1]; a.op == "path" && b.op == "path" && !a.paren && !b.paren {
			if r := env(b.str); r != nil {
				if eq, err := exprEqual(l, r); err != nil || !eq {
					return nil, fmt.Errorf("%s and %s are one key spelled two ways, and they disagree (%v and %v)",
						a.str, b.str, l, r)
				}
			}
		}
		return l, nil
	case "?:":
		c, err := evalBool(n.args[0], env)
		if err != nil {
			return nil, err
		}
		if c {
			return evalMJ(n.args[1], env)
		}
		return evalMJ(n.args[2], env)
	case "member":
		v, err := evalMJ(n.args[0], env)
		if err != nil {
			return nil, err
		}
		for _, k := range strings.Split(n.str, ".") {
			if v == nil {
				return nil, nil
			}
			m, ok := v.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("%v has no key %q: it is not an object", v, k)
			}
			v = m[k]
		}
		return v, nil
	case "call":
		switch {
		case n.str == "uniform" && len(n.args) == 1:
			return evalUniform(n.args[0], env)
		case n.str == "each_layer" && len(n.args) == 5:
			return evalEachLayer(n.args, env)
		case n.str == "partitions" && len(n.args) == 4:
			return evalPartitions(n.args, env)
		}
		return nil, fmt.Errorf("%s(): calls are a tensor map's, not a config's", n.str)
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
	case "**":
		v := math.Pow(l, r)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("%s: %g ** %g is not a number", exprText(n), l, r)
		}
		return v, nil
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

// evalLayerList is a list of layer positions the config names, an absent one
// none, each a whole number.
func evalLayerList(n *exprNode, env exprEnv) ([]float64, error) {
	v, err := evalMJ(n, env)
	if err != nil || v == nil {
		return nil, err
	}
	list, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("%s is %s, not a list of layers", exprText(n), exprKind(v))
	}
	out := make([]float64, len(list))
	for i, x := range list {
		f, ok := x.(float64)
		if !ok || f != math.Trunc(f) {
			return nil, fmt.Errorf("%s[%d] is %v, not a layer", exprText(n), i, x)
		}
		out[i] = f
	}
	return out, nil
}

// evalLayerCount is a count of layers, bounded so nothing is sized from a
// file unchecked (RULE 9).
func evalLayerCount(n *exprNode, env exprEnv) (int, error) {
	f, err := evalNum(n, env)
	if err != nil {
		return 0, err
	}
	if f < 0 || f > 1<<16 || f != math.Trunc(f) {
		return 0, fmt.Errorf("%s is %v, not a count of layers", exprText(n), f)
	}
	return int(f), nil
}

// evalEachLayer is each_layer(n, members, first, inside, outside): n values,
// the i-th inside where i + first is among members (a list of positions
// counted from first), outside elsewhere.
func evalEachLayer(args []*exprNode, env exprEnv) (any, error) {
	n, err := evalLayerCount(args[0], env)
	if err != nil {
		return nil, err
	}
	members, err := evalLayerList(args[1], env)
	if err != nil {
		return nil, err
	}
	first, err := evalNum(args[2], env)
	if err != nil {
		return nil, err
	}
	in, err := evalMJ(args[3], env)
	if err != nil {
		return nil, err
	}
	out, err := evalMJ(args[4], env)
	if err != nil {
		return nil, err
	}
	at := map[float64]bool{}
	for _, m := range members {
		at[m-first] = true
	}
	res := make([]any, n)
	for i := range res {
		res[i] = out
		if at[float64(i)] {
			res[i] = in
		}
	}
	return res, nil
}

// evalPartitions is partitions(n, first, a, b): whether lists a and b name
// each of n positions, counted from first, exactly once between them.
func evalPartitions(args []*exprNode, env exprEnv) (any, error) {
	n, err := evalLayerCount(args[0], env)
	if err != nil {
		return nil, err
	}
	first, err := evalNum(args[1], env)
	if err != nil {
		return nil, err
	}
	seen := make([]bool, n)
	for _, l := range args[2:] {
		list, err := evalLayerList(l, env)
		if err != nil {
			return nil, err
		}
		for _, m := range list {
			i := m - first
			if i < 0 || i >= float64(n) || seen[int(i)] {
				return false, nil
			}
			seen[int(i)] = true
		}
	}
	return !slices.Contains(seen, false), nil
}

// evalUniform is uniform(x): a value, or the one value every entry of a
// per-layer list shares. A list whose entries differ is refused, as the
// container carries one value for the model.
func evalUniform(n *exprNode, env exprEnv) (any, error) {
	v, err := evalMJ(n, env)
	if err != nil {
		return nil, err
	}
	list, ok := v.([]any)
	if !ok {
		return v, nil
	}
	if len(list) == 0 {
		return nil, fmt.Errorf("%s: uniform of an empty list", exprText(n))
	}
	for _, x := range list[1:] {
		if eq, err := exprEqual(x, list[0]); err != nil || !eq {
			return nil, fmt.Errorf("%s: a per-layer list %v whose entries differ, and the container "+
				"carries one value: %w", exprText(n), list, ErrNotImplemented)
		}
	}
	return list[0], nil
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

// exprEqual compares scalars, and lists element by element; null equals
// only null.
func exprEqual(a, b any) (bool, error) {
	if a == nil || b == nil {
		return a == nil && b == nil, nil
	}
	switch av := a.(type) {
	case float64, string, bool:
		return a == b, nil
	case []any:
		bv, ok := b.([]any)
		if !ok || len(bv) != len(av) {
			return false, nil
		}
		for i := range av {
			if eq, err := exprEqual(av[i], bv[i]); err != nil || !eq {
				return eq, err
			}
		}
		return true, nil
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok || len(bv) != len(av) {
			return false, nil
		}
		for k, x := range av {
			if eq, err := exprEqual(x, bv[k]); err != nil || !eq {
				return eq, err
			}
		}
		return true, nil
	}
	return false, fmt.Errorf("%s compared with ==", exprKind(a))
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
	case map[string]any:
		return "an object"
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
