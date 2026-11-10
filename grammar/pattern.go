package grammar

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// span is the code points lo through hi.
type span struct{ lo, hi rune }

// runeSet is a set of code points as sorted, disjoint, non-adjacent spans.
type runeSet []span

// setOf is the set the spans cover.
func setOf(spans ...span) runeSet {
	s := slices.Clone(spans)
	slices.SortFunc(s, func(a, b span) int { return int(a.lo - b.lo) })
	var out runeSet
	for _, x := range s {
		if n := len(out); n > 0 && x.lo <= out[n-1].hi+1 {
			out[n-1].hi = max(out[n-1].hi, x.hi)
			continue
		}
		out = append(out, x)
	}
	return out
}

// complement is every code point outside s, the surrogates excluded: a string
// holds none of them.
func (s runeSet) complement() runeSet {
	var out runeSet
	next := rune(0)
	for _, x := range setOf(append(slices.Clone(s), span{0xD800, 0xDFFF})...) {
		if x.lo > next {
			out = append(out, span{next, x.lo - 1})
		}
		next = x.hi + 1
	}
	if next <= 0x10FFFF {
		out = append(out, span{next, 0x10FFFF})
	}
	return out
}

// minus is s without t's code points.
func (s runeSet) minus(t runeSet) runeSet {
	var out runeSet
	for _, x := range s {
		lo := x.lo
		for _, y := range t {
			if y.hi < lo || y.lo > x.hi {
				continue
			}
			if y.lo > lo {
				out = append(out, span{lo, y.lo - 1})
			}
			lo = y.hi + 1
		}
		if lo <= x.hi {
			out = append(out, span{lo, x.hi})
		}
	}
	return out
}

func (s runeSet) has(r rune) bool {
	for _, x := range s {
		if x.lo <= r && r <= x.hi {
			return true
		}
	}
	return false
}

// The ECMA-262 classes a pattern names.
func digits() runeSet { return setOf(span{'0', '9'}) }
func word() runeSet   { return setOf(span{'0', '9'}, span{'A', 'Z'}, span{'_', '_'}, span{'a', 'z'}) }
func spaces() runeSet {
	return setOf(span{'\t', '\r'}, span{' ', ' '}, span{0xA0, 0xA0}, span{0x1680, 0x1680}, span{0x2000, 0x200A},
		span{0x2028, 0x2029}, span{0x202F, 0x202F}, span{0x205F, 0x205F}, span{0x3000, 0x3000}, span{0xFEFF, 0xFEFF})
}
func lineEnds() runeSet { return setOf(span{'\n', '\n'}, span{'\r', '\r'}, span{0x2028, 0x2029}) }

// jsonChar is GBNF for one character of a JSON string whose decoded value is
// in s: a literal where JSON writes it bare, its escape where JSON must
// escape it -- a quote, a backslash, a control character -- as the string
// primitive writes them.
func (c *converter) jsonChar(s runeSet) (string, error) {
	var alts []string
	if lit := s.minus(setOf(span{0, 0x1F}, span{'"', '"'}, span{'\\', '\\'}, span{0x7F, 0x7F})); len(lit) > 0 {
		var b strings.Builder
		b.WriteByte('[')
		for _, x := range lit {
			b.WriteString(classRune(x.lo))
			if x.hi > x.lo {
				b.WriteString("-" + classRune(x.hi))
			}
		}
		b.WriteByte(']')
		alts = append(alts, b.String())
	}
	if s.has('"') {
		alts = append(alts, literal(`\"`))
	}
	if s.has('\\') {
		alts = append(alts, literal(`\\`))
	}
	if ctl := setOf(span{0, 0x1F}); len(ctl.minus(s)) == 0 {
		alts = append(alts, literal(`\`)+` ( [bfnrt] | "u00" [01] [0-9a-fA-F] )`)
	} else {
		for r := rune(0); r <= 0x1F; r++ {
			if s.has(r) {
				alts = append(alts, literal(jsonEscape(r)))
			}
		}
	}
	if s.has(0x7F) {
		alts = append(alts, literal(`\u007f`))
	}
	switch len(alts) {
	case 0:
		return "", ErrUnsupported{"a character class nothing matches, in a pattern"}
	case 1:
		return alts[0], nil
	}
	return "( " + strings.Join(alts, " | ") + " )", nil
}

// jsonEscape is how JSON writes control character r.
func jsonEscape(r rune) string {
	switch r {
	case '\b':
		return `\b`
	case '\t':
		return `\t`
	case '\n':
		return `\n`
	case '\f':
		return `\f`
	case '\r':
		return `\r`
	}
	return fmt.Sprintf(`\u%04x`, r)
}

// pattern is GBNF for the content of a JSON string that holds a match of
// src, a JSON Schema pattern (an ECMA-262 regular expression, unanchored: a
// branch with no ^ may start anywhere, one with no $ end anywhere).
// Lookarounds, backreferences, word boundaries and Unicode property escapes
// are refused by name.
func (c *converter) pattern(src string) (string, error) {
	any := c.prim("char") + "*"
	var alts []string
	for _, b := range topBranches([]rune(src)) {
		start := len(b) > 0 && b[0] == '^'
		if start {
			b = b[1:]
		}
		end := len(b) > 0 && b[len(b)-1] == '$' && !escaped(b, len(b)-1)
		if end {
			b = b[:len(b)-1]
		}
		p := &rxParser{c: c, s: b}
		body, err := p.alt()
		if err != nil {
			return "", fmt.Errorf("%w (pattern %q)", err, src)
		}
		if p.i < len(p.s) {
			return "", fmt.Errorf("grammar: pattern %q: a ) with no ( before it", src)
		}
		var parts []string
		if !start {
			parts = append(parts, any)
		}
		if body != "" {
			parts = append(parts, "( "+body+" )")
		}
		if !end {
			parts = append(parts, any)
		}
		if len(parts) == 0 {
			parts = []string{`""`}
		}
		alts = append(alts, strings.Join(parts, " "))
	}
	return strings.Join(alts, " | "), nil
}

// topBranches is src cut at its top-level |, outside groups and classes.
func topBranches(src []rune) [][]rune {
	var out [][]rune
	depth, from := 0, 0
	for i := 0; i < len(src); i++ {
		switch src[i] {
		case '\\':
			i++
		case '[':
			i = classEnd(src, i)
		case '(':
			depth++
		case ')':
			depth--
		case '|':
			if depth == 0 {
				out = append(out, src[from:i])
				from = i + 1
			}
		}
	}
	return append(out, src[from:])
}

// classEnd is the index of the ] that closes the class opened at i.
func classEnd(src []rune, i int) int {
	i++
	if i < len(src) && src[i] == '^' {
		i++
	}
	for ; i < len(src); i++ {
		switch src[i] {
		case '\\':
			i++
		case ']':
			return i
		}
	}
	return i
}

// escaped reports whether src[i] follows an odd run of backslashes.
func escaped(src []rune, i int) bool {
	n := 0
	for j := i - 1; j >= 0 && src[j] == '\\'; j-- {
		n++
	}
	return n%2 == 1
}

// rxParser reads one branch of a pattern into GBNF.
type rxParser struct {
	c *converter
	s []rune
	i int
}

func (p *rxParser) at(lit string) bool {
	return strings.HasPrefix(string(p.s[p.i:]), lit)
}

func (p *rxParser) alt() (string, error) {
	var alts []string
	for {
		s, err := p.seq()
		if err != nil {
			return "", err
		}
		if s == "" {
			s = `""`
		}
		alts = append(alts, s)
		if p.i < len(p.s) && p.s[p.i] == '|' {
			p.i++
			continue
		}
		break
	}
	if len(alts) == 1 {
		if alts[0] == `""` {
			return "", nil
		}
		return alts[0], nil
	}
	return "( " + strings.Join(alts, " | ") + " )", nil
}

func (p *rxParser) seq() (string, error) {
	var items []string
	for p.i < len(p.s) && p.s[p.i] != '|' && p.s[p.i] != ')' {
		a, err := p.atom()
		if err != nil {
			return "", err
		}
		lo, hi, ok, err := p.quantifier()
		if err != nil {
			return "", err
		}
		if ok {
			a = "( " + a + " )" + rep(lo, hi)
		}
		items = append(items, a)
	}
	return strings.Join(items, " "), nil
}

// quantifier reads a quantifier after an atom, ok false where none follows;
// a lazy one matches the same strings.
func (p *rxParser) quantifier() (lo, hi int, ok bool, err error) {
	if p.i >= len(p.s) {
		return 0, 0, false, nil
	}
	switch p.s[p.i] {
	case '*':
		lo, hi, ok = 0, -1, true
		p.i++
	case '+':
		lo, hi, ok = 1, -1, true
		p.i++
	case '?':
		lo, hi, ok = 0, 1, true
		p.i++
	case '{':
		n, l, h, valid := braces(p.s[p.i:])
		if !valid {
			return 0, 0, false, nil
		}
		if l > maxRepeat || h > maxRepeat {
			return 0, 0, false, fmt.Errorf("grammar: a pattern repeats past %d", maxRepeat)
		}
		lo, hi, ok = l, h, true
		p.i += n
	default:
		return 0, 0, false, nil
	}
	if p.i < len(p.s) && p.s[p.i] == '?' {
		p.i++
	}
	return lo, hi, ok, nil
}

// braces reads {n}, {n,} or {n,m} at the start of s: its length and bounds,
// valid false where s starts no such quantifier (a { that is a literal).
func braces(s []rune) (n, lo, hi int, valid bool) {
	end := slices.Index(s, '}')
	if end < 0 {
		return 0, 0, 0, false
	}
	body := string(s[1:end])
	a, b, comma := strings.Cut(body, ",")
	l, err := strconv.Atoi(a)
	if err != nil || l < 0 {
		return 0, 0, 0, false
	}
	h := l
	if comma {
		h = -1
		if b != "" {
			if h, err = strconv.Atoi(b); err != nil || h < l {
				return 0, 0, 0, false
			}
		}
	}
	return end + 1, l, h, true
}

func (p *rxParser) atom() (string, error) {
	r := p.s[p.i]
	switch r {
	case '(':
		p.i++
		switch {
		case p.at("?:"):
			p.i += 2
		case p.at("?<") && !p.at("?<=") && !p.at("?<!"):
			end := slices.Index(p.s[p.i:], '>')
			if end < 0 {
				return "", fmt.Errorf("grammar: a pattern's named group has no >")
			}
			p.i += end + 1
		case p.at("?"):
			return "", ErrUnsupported{"a lookaround or group modifier in a pattern"}
		}
		body, err := p.alt()
		if err != nil {
			return "", err
		}
		if p.i >= len(p.s) || p.s[p.i] != ')' {
			return "", fmt.Errorf("grammar: a pattern's group has no )")
		}
		p.i++
		if body == "" {
			body = `""`
		}
		return "( " + body + " )", nil
	case '[':
		set, err := p.class()
		if err != nil {
			return "", err
		}
		return p.c.jsonChar(set)
	case '.':
		p.i++
		return p.c.jsonChar(lineEnds().complement())
	case '\\':
		lit, set, err := p.escape(false)
		if err != nil {
			return "", err
		}
		if set == nil {
			set = setOf(span{lit, lit})
		}
		return p.c.jsonChar(set)
	case '^', '$':
		return "", ErrUnsupported{"an anchor inside a pattern"}
	case '*', '+', '?':
		return "", fmt.Errorf("grammar: a pattern's %c repeats nothing", r)
	case '{':
		if _, _, _, valid := braces(p.s[p.i:]); valid {
			return "", fmt.Errorf("grammar: a pattern's {} repeats nothing")
		}
	}
	p.i++
	return p.c.jsonChar(setOf(span{r, r}))
}

// escape reads the escape at p.i: a character, or a class (set non-nil).
// Inside a class \b is a backspace, outside it a word boundary.
func (p *rxParser) escape(inClass bool) (rune, runeSet, error) {
	if p.i+1 >= len(p.s) {
		return 0, nil, fmt.Errorf("grammar: a pattern ends in a backslash")
	}
	c := p.s[p.i+1]
	p.i += 2
	hex := func(n int) (rune, error) {
		if p.i+n > len(p.s) {
			return 0, fmt.Errorf("grammar: a pattern's \\%c escape is short", c)
		}
		v, err := strconv.ParseUint(string(p.s[p.i:p.i+n]), 16, 32)
		if err != nil {
			return 0, fmt.Errorf("grammar: a pattern's \\%c escape is not hex", c)
		}
		p.i += n
		return rune(v), nil
	}
	switch c {
	case 'd':
		return 0, digits(), nil
	case 'D':
		return 0, digits().complement(), nil
	case 'w':
		return 0, word(), nil
	case 'W':
		return 0, word().complement(), nil
	case 's':
		return 0, spaces(), nil
	case 'S':
		return 0, spaces().complement(), nil
	case 'n':
		return '\n', nil, nil
	case 'r':
		return '\r', nil, nil
	case 't':
		return '\t', nil, nil
	case 'f':
		return '\f', nil, nil
	case 'v':
		return '\v', nil, nil
	case 'x':
		r, err := hex(2)
		return r, nil, err
	case 'u':
		r, err := hex(4)
		return r, nil, err
	case 'c':
		if p.i < len(p.s) && (p.s[p.i]|0x20) >= 'a' && (p.s[p.i]|0x20) <= 'z' {
			r := p.s[p.i] % 32
			p.i++
			return r, nil, nil
		}
		return '\\', nil, nil
	case 'b':
		if inClass {
			return '\b', nil, nil
		}
		return 0, nil, ErrUnsupported{"a word boundary (\\b) in a pattern"}
	case 'B':
		return 0, nil, ErrUnsupported{"a word boundary (\\B) in a pattern"}
	case 'p', 'P':
		return 0, nil, ErrUnsupported{"a Unicode property escape in a pattern"}
	case 'k':
		return 0, nil, ErrUnsupported{"a backreference in a pattern"}
	case '0':
		if p.i < len(p.s) && p.s[p.i] >= '0' && p.s[p.i] <= '9' {
			return 0, nil, ErrUnsupported{"an octal escape in a pattern"}
		}
		return 0, nil, nil
	}
	if c >= '1' && c <= '9' {
		if inClass {
			return 0, nil, ErrUnsupported{"an octal escape in a pattern"}
		}
		return 0, nil, ErrUnsupported{"a backreference in a pattern"}
	}
	return c, nil, nil
}

// class reads a character class at p.i.
func (p *rxParser) class() (runeSet, error) {
	p.i++
	neg := false
	if p.i < len(p.s) && p.s[p.i] == '^' {
		neg = true
		p.i++
	}
	var spans []span
	for {
		if p.i >= len(p.s) {
			return nil, fmt.Errorf("grammar: a pattern's character class has no ]")
		}
		if p.s[p.i] == ']' {
			p.i++
			break
		}
		lo, set, err := p.classAtom()
		if err != nil {
			return nil, err
		}
		if set != nil {
			spans = append(spans, set...)
			continue
		}
		if p.i+1 < len(p.s) && p.s[p.i] == '-' && p.s[p.i+1] != ']' {
			p.i++
			hi, hset, err := p.classAtom()
			if err != nil {
				return nil, err
			}
			if hset != nil {
				// A range to a class is the three of them (ECMA-262 B.1.2).
				spans = append(append(spans, span{lo, lo}, span{'-', '-'}), hset...)
				continue
			}
			if hi < lo {
				return nil, fmt.Errorf("grammar: a pattern's range %c-%c runs backwards", lo, hi)
			}
			spans = append(spans, span{lo, hi})
			continue
		}
		spans = append(spans, span{lo, lo})
	}
	s := setOf(spans...)
	if neg {
		s = s.complement()
	}
	return s, nil
}

func (p *rxParser) classAtom() (rune, runeSet, error) {
	if p.s[p.i] == '\\' {
		return p.escape(true)
	}
	r := p.s[p.i]
	p.i++
	return r, nil, nil
}
