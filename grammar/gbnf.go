// Package grammar constrains generation to a context-free grammar: GBNF text
// (llama.cpp's grammar format) or a JSON Schema compiled to it, matched one
// code point at a time by a pushdown automaton and turned, at each step, into
// the set of vocabulary tokens that keep the output inside the grammar.
//
// What runs per step, and which side of RULE 8 it is on, is in
// docs/engineering-history/model-correctness.md, "grammar".
package grammar

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// elemKind is what one position of a flattened rule holds.
type elemKind uint8

const (
	// eEnd closes a rule's last alternative.
	eEnd elemKind = iota
	// eAlt closes an alternative that another follows.
	eAlt
	// eRef refers to rule arg.
	eRef
	// eChar matches one code point of class arg.
	eChar
)

type elem struct {
	kind elemKind
	arg  int32
}

// class is a set of code points: sorted, merged inclusive ranges, or their
// complement.
type class struct {
	neg bool
	r   [][2]rune
}

func (c *class) has(x rune) bool {
	_, found := slices.BinarySearchFunc(c.r, x, func(r [2]rune, x rune) int {
		switch {
		case r[1] < x:
			return -1
		case r[0] > x:
			return 1
		}
		return 0
	})
	return found != c.neg
}

// meets is whether any code point in [lo, hi] is in the class.
func (c *class) meets(lo, hi rune) bool {
	if !c.neg {
		for _, r := range c.r {
			if r[0] <= hi && r[1] >= lo {
				return true
			}
		}
		return false
	}
	// A complement meets the range unless its ranges cover all of it.
	at := lo
	for _, r := range c.r {
		if r[1] < at {
			continue
		}
		if r[0] > at {
			return true
		}
		at = r[1] + 1
		if at > hi {
			return false
		}
	}
	return at <= hi
}

// Grammar is a compiled grammar: every rule's alternatives flattened into one
// array of positions, as llama.cpp lays them out.
type Grammar struct {
	el   []elem
	cls  []class
	alts [][]int32 // each rule's alternatives, by their first position
	root int32
	// names are the rules' names, for errors.
	names []string
	src   string
}

// Source is the GBNF text the grammar was compiled from.
func (g *Grammar) Source() string { return g.src }

// item is one element of a rule before flattening.
type item struct {
	ref bool
	arg int32
}

type parser struct {
	s     string
	i     int
	rules [][][]item // per rule, alternatives
	names []string
	ids   map[string]int32
	cls   []class
	clsID map[string]int32
	// defined says which rules have a body; a name only referred to is an
	// error at the end.
	defined []bool
	anon    int
}

// Parse compiles GBNF text. The root rule is "root". Left recursion is
// refused: a pushdown automaton stepping by code point cannot run it.
func Parse(src string) (*Grammar, error) {
	p := &parser{s: src, ids: map[string]int32{}, clsID: map[string]int32{}}
	if err := p.parse(); err != nil {
		return nil, err
	}
	root, ok := p.ids["root"]
	if !ok {
		return nil, fmt.Errorf("grammar: no rule named root")
	}
	for i, d := range p.defined {
		if !d {
			return nil, fmt.Errorf("grammar: rule %q is used but never defined", p.names[i])
		}
	}
	g := &Grammar{cls: p.cls, root: root, names: p.names, src: src}
	g.alts = make([][]int32, len(p.rules))
	for r, alts := range p.rules {
		for ai, seq := range alts {
			g.alts[r] = append(g.alts[r], int32(len(g.el)))
			for _, it := range seq {
				k := eChar
				if it.ref {
					k = eRef
				}
				g.el = append(g.el, elem{k, it.arg})
			}
			if ai == len(alts)-1 {
				g.el = append(g.el, elem{kind: eEnd})
			} else {
				g.el = append(g.el, elem{kind: eAlt})
			}
		}
	}
	if err := g.refuseLeftRecursion(); err != nil {
		return nil, err
	}
	return g, nil
}

func (p *parser) errf(format string, a ...any) error {
	line := 1 + strings.Count(p.s[:p.i], "\n")
	return fmt.Errorf("grammar: line %d: %s", line, fmt.Sprintf(format, a...))
}

func (p *parser) rule(name string) int32 {
	if id, ok := p.ids[name]; ok {
		return id
	}
	id := int32(len(p.rules))
	p.ids[name] = id
	p.names = append(p.names, name)
	p.rules = append(p.rules, nil)
	p.defined = append(p.defined, false)
	return id
}

func (p *parser) newRule(alts [][]item) int32 {
	p.anon++
	id := p.rule(fmt.Sprintf("_%d", p.anon))
	p.rules[id], p.defined[id] = alts, true
	return id
}

func (p *parser) class(c class) int32 {
	slices.SortFunc(c.r, func(a, b [2]rune) int { return int(a[0] - b[0]) })
	var m [][2]rune
	for _, r := range c.r {
		if n := len(m); n > 0 && r[0] <= m[n-1][1]+1 {
			m[n-1][1] = max(m[n-1][1], r[1])
			continue
		}
		m = append(m, r)
	}
	c.r = m
	key := fmt.Sprint(c.neg, c.r)
	if id, ok := p.clsID[key]; ok {
		return id
	}
	id := int32(len(p.cls))
	p.cls = append(p.cls, c)
	p.clsID[key] = id
	return id
}

// space skips blanks and comments; newlines too where nl is set (between
// rules, and inside parentheses).
func (p *parser) space(nl bool) {
	for p.i < len(p.s) {
		switch c := p.s[p.i]; {
		case c == ' ' || c == '\t' || c == '\r':
			p.i++
		case c == '\n' && nl:
			p.i++
		case c == '#':
			for p.i < len(p.s) && p.s[p.i] != '\n' {
				p.i++
			}
		default:
			return
		}
	}
}

func isNameByte(c byte) bool {
	return c == '-' || c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

func (p *parser) name() string {
	j := p.i
	for p.i < len(p.s) && isNameByte(p.s[p.i]) {
		p.i++
	}
	return p.s[j:p.i]
}

func (p *parser) parse() error {
	for {
		p.space(true)
		if p.i >= len(p.s) {
			return nil
		}
		n := p.name()
		if n == "" {
			return p.errf("expected a rule name at %q", clip(p.s[p.i:]))
		}
		p.space(false)
		if !strings.HasPrefix(p.s[p.i:], "::=") {
			return p.errf("expected ::= after %q", n)
		}
		p.i += 3
		id := p.rule(n)
		if p.defined[id] {
			return p.errf("rule %q is defined twice", n)
		}
		alts, err := p.alternatives(false)
		if err != nil {
			return err
		}
		p.rules[id], p.defined[id] = alts, true
	}
}

// alternatives reads seq ( "|" seq )*, to the end of the line (a rule) or to
// the closing parenthesis (a group, nested).
func (p *parser) alternatives(nested bool) ([][]item, error) {
	var alts [][]item
	for {
		seq, err := p.sequence(nested)
		if err != nil {
			return nil, err
		}
		alts = append(alts, seq)
		p.space(nested)
		if p.i < len(p.s) && p.s[p.i] == '|' {
			p.i++
			p.space(true)
			continue
		}
		return alts, nil
	}
}

func (p *parser) sequence(nested bool) ([]item, error) {
	var seq []item
	last := -1 // where the last atom starts in seq, for a postfix operator
	for {
		p.space(nested)
		if p.i >= len(p.s) {
			return seq, nil
		}
		c := p.s[p.i]
		switch {
		case c == '"':
			p.i++
			last = len(seq)
			for {
				if p.i >= len(p.s) {
					return nil, p.errf("unterminated string")
				}
				if p.s[p.i] == '"' {
					p.i++
					break
				}
				r, err := p.char()
				if err != nil {
					return nil, err
				}
				seq = append(seq, item{arg: p.class(class{r: [][2]rune{{r, r}}})})
			}
			if last == len(seq) {
				last = -1 // "" is nothing
			}
		case c == '[':
			p.i++
			cl := class{}
			if p.i < len(p.s) && p.s[p.i] == '^' {
				cl.neg = true
				p.i++
			}
			for {
				if p.i >= len(p.s) {
					return nil, p.errf("unterminated character class")
				}
				if p.s[p.i] == ']' {
					p.i++
					break
				}
				lo, err := p.char()
				if err != nil {
					return nil, err
				}
				hi := lo
				if p.i+1 < len(p.s) && p.s[p.i] == '-' && p.s[p.i+1] != ']' {
					p.i++
					if hi, err = p.char(); err != nil {
						return nil, err
					}
				}
				if hi < lo {
					return nil, p.errf("character range %q-%q runs backwards", lo, hi)
				}
				cl.r = append(cl.r, [2]rune{lo, hi})
			}
			last = len(seq)
			seq = append(seq, item{arg: p.class(cl)})
		case c == '.':
			p.i++
			last = len(seq)
			seq = append(seq, item{arg: p.class(class{neg: true})})
		case c == '(':
			p.i++
			alts, err := p.alternatives(true)
			if err != nil {
				return nil, err
			}
			p.space(true)
			if p.i >= len(p.s) || p.s[p.i] != ')' {
				return nil, p.errf("expected )")
			}
			p.i++
			last = len(seq)
			seq = append(seq, item{ref: true, arg: p.newRule(alts)})
		case isNameByte(c):
			// A name followed by ::= starts the next rule: this sequence ended
			// at the line break before it.
			j := p.i
			n := p.name()
			k := p.i
			p.space(false)
			if strings.HasPrefix(p.s[p.i:], "::=") {
				p.i = j
				return seq, nil
			}
			p.i = k
			last = len(seq)
			seq = append(seq, item{ref: true, arg: p.rule(n)})
		case c == '*' || c == '+' || c == '?' || c == '{':
			if last < 0 {
				return nil, p.errf("%q follows nothing", c)
			}
			lo, hi, err := p.repeat()
			if err != nil {
				return nil, err
			}
			atom := slices.Clone(seq[last:])
			seq = append(seq[:last], p.repetition(atom, lo, hi)...)
			last = -1
		default:
			if c == '\n' || c == ')' || c == '|' {
				return seq, nil
			}
			return nil, p.errf("unexpected %q", clip(p.s[p.i:]))
		}
	}
}

// repeat reads a postfix operator as bounds; hi < 0 is unbounded.
func (p *parser) repeat() (lo, hi int, err error) {
	c := p.s[p.i]
	p.i++
	switch c {
	case '*':
		return 0, -1, nil
	case '+':
		return 1, -1, nil
	case '?':
		return 0, 1, nil
	}
	end := strings.IndexByte(p.s[p.i:], '}')
	if end < 0 {
		return 0, 0, p.errf("unterminated {")
	}
	body := strings.TrimSpace(p.s[p.i : p.i+end])
	p.i += end + 1
	a, b, comma := strings.Cut(body, ",")
	if lo, err = strconv.Atoi(strings.TrimSpace(a)); err != nil || lo < 0 {
		return 0, 0, p.errf("bad repetition {%s}", body)
	}
	hi = lo
	if comma {
		hi = -1
		if b = strings.TrimSpace(b); b != "" {
			if hi, err = strconv.Atoi(b); err != nil || hi < lo {
				return 0, 0, p.errf("bad repetition {%s}", body)
			}
		}
	}
	if lo > maxRepeat || hi > maxRepeat {
		return 0, 0, p.errf("repetition {%s} is past %d", body, maxRepeat)
	}
	return lo, hi, nil
}

// maxRepeat bounds a counted repetition, which is expanded.
const maxRepeat = 4096

// repetition is atom{lo,hi} as plain items: lo copies, then the optional
// ones nested -- (x (x (x)?)?)? -- or a right-recursive star.
func (p *parser) repetition(atom []item, lo, hi int) []item {
	var out []item
	for range lo {
		out = append(out, atom...)
	}
	switch {
	case hi < 0:
		// R ::= atom R | (empty): right recursion, which the automaton runs
		// as a tail call, so a long run does not deepen the stack.
		r := p.newRule(nil)
		p.rules[r] = [][]item{append(slices.Clone(atom), item{ref: true, arg: r}), nil}
		out = append(out, item{ref: true, arg: r})
	case hi > lo:
		var tail []item
		for range hi - lo {
			seq := append(slices.Clone(atom), tail...)
			tail = []item{{ref: true, arg: p.newRule([][]item{seq, nil})}}
		}
		out = append(out, tail...)
	}
	return out
}

// char reads one code point of a literal or a class, escapes decoded.
func (p *parser) char() (rune, error) {
	if p.s[p.i] != '\\' {
		r, n := utf8.DecodeRuneInString(p.s[p.i:])
		if r == utf8.RuneError && n <= 1 {
			return 0, p.errf("invalid UTF-8")
		}
		p.i += n
		return r, nil
	}
	if p.i+1 >= len(p.s) {
		return 0, p.errf("a trailing backslash")
	}
	c := p.s[p.i+1]
	p.i += 2
	switch c {
	case 'n':
		return '\n', nil
	case 'r':
		return '\r', nil
	case 't':
		return '\t', nil
	case '\\', '"', '[', ']', '-', '^', '\'':
		return rune(c), nil
	case 'x', 'u', 'U':
		n := map[byte]int{'x': 2, 'u': 4, 'U': 8}[c]
		if p.i+n > len(p.s) {
			return 0, p.errf("a short \\%c escape", c)
		}
		v, err := strconv.ParseUint(p.s[p.i:p.i+n], 16, 32)
		if err != nil {
			return 0, p.errf("a bad \\%c escape", c)
		}
		p.i += n
		return rune(v), nil
	}
	return 0, p.errf("unknown escape \\%c", c)
}

func clip(s string) string {
	if len(s) > 16 {
		return s[:16] + "..."
	}
	return s
}

// refuseLeftRecursion finds a rule that can reach itself before consuming a
// code point: the automaton would expand it forever.
func (g *Grammar) refuseLeftRecursion() error {
	n := len(g.alts)
	nullable := make([]bool, n)
	for changed := true; changed; {
		changed = false
		for r := range n {
			if nullable[r] {
				continue
			}
			for _, a := range g.alts[r] {
				if g.seqNullable(a, nullable) {
					nullable[r], changed = true, true
					break
				}
			}
		}
	}
	// left[r] is the rules r can begin with.
	state := make([]uint8, n) // 0 unseen, 1 on the path, 2 done
	var visit func(r int32) error
	visit = func(r int32) error {
		switch state[r] {
		case 1:
			return fmt.Errorf("grammar: rule %q is left-recursive", g.names[r])
		case 2:
			return nil
		}
		state[r] = 1
		for _, a := range g.alts[r] {
			for p := a; g.el[p].kind == eRef || g.el[p].kind == eChar; p++ {
				if g.el[p].kind == eChar {
					break
				}
				if err := visit(g.el[p].arg); err != nil {
					return err
				}
				if !nullable[g.el[p].arg] {
					break
				}
			}
		}
		state[r] = 2
		return nil
	}
	for r := range n {
		if err := visit(int32(r)); err != nil {
			return err
		}
	}
	return nil
}

func (g *Grammar) seqNullable(p int32, nullable []bool) bool {
	for ; ; p++ {
		switch e := g.el[p]; e.kind {
		case eEnd, eAlt:
			return true
		case eChar:
			return false
		case eRef:
			if !nullable[e.arg] {
				return false
			}
		}
	}
}
