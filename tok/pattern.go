package tok

import (
	"fmt"
	"slices"
	"strconv"

	"github.com/jitllm/jitllm/internal/uniprop"
)

// A split pattern is a tokenizer.json Split stage's regex, compiled once at
// load and run as the model's own tokenizer runs it: the tokenizers library
// hands the pattern to Oniguruma (Ruby syntax, UTF-8), so alternation is
// leftmost-first, quantifiers are greedy and backtrack, lookahead is atomic,
// `.` is any code point but \n, \s is White_Space, \d is \p{Nd}, \w is the
// library's word class (patternParser.word), and (?i) folds case. Every class
// is resolved at compile time against the pinned Unicode tables
// (internal/uniprop, the case mappings in unicodedata.go), so what a pattern
// matches does not depend on the Go toolchain. Those tables are Unicode
// 15.1.0 and the library's regex reads a later Unicode, so a code point
// assigned after 15.1.0 belongs to no class here; every code point 15.1.0
// assigns is classed as the library classes it (TestClassesAreTheLibrarys).
//
// ^ and $ are Ruby's line anchors, \A \z \Z the text's, \G where the
// search started, \b and \B the boundaries of \w; possessive quantifiers
// and (?>...) are atomic. Case folding is the simple lowercase
// mapping plus U+017F (ucFold), which is every fold the contraction
// alternatives the patterns carry can meet; the multi-character folds (U+00DF
// against "ss") are not taken. A construct outside what the patterns in the
// wild use -- lookbehind, backreferences, POSIX brackets --
// is refused at compile, naming it, rather than read as something nearby.

// spanSet is a set of code points as sorted, disjoint, non-adjacent spans.
type spanSet []uniprop.Span

func normSpans(s spanSet) spanSet {
	s = slices.Clone(s)
	slices.SortFunc(s, func(a, b uniprop.Span) int { return int(a.Lo - b.Lo) })
	var out spanSet
	for _, x := range s {
		if n := len(out); n > 0 && x.Lo <= out[n-1].Hi+1 {
			out[n-1].Hi = max(out[n-1].Hi, x.Hi)
			continue
		}
		out = append(out, x)
	}
	return out
}

func (s spanSet) union(t spanSet) spanSet { return normSpans(append(slices.Clone(s), t...)) }

func (s spanSet) complement() spanSet {
	var out spanSet
	next := rune(0)
	for _, x := range normSpans(s) {
		if x.Lo > next {
			out = append(out, uniprop.Span{Lo: next, Hi: x.Lo - 1})
		}
		next = x.Hi + 1
	}
	if next <= 0x10FFFF {
		out = append(out, uniprop.Span{Lo: next, Hi: 0x10FFFF})
	}
	return out
}

func (s spanSet) intersect(t spanSet) spanSet {
	return s.complement().union(t.complement()).complement()
}

func (s spanSet) has(r rune) bool {
	lo, hi := 0, len(s)
	for lo < hi {
		m := int(uint(lo+hi) >> 1)
		switch {
		case r < s[m].Lo:
			hi = m
		case r > s[m].Hi:
			lo = m + 1
		default:
			return true
		}
	}
	return false
}

func runeSpan(r rune) spanSet { return spanSet{{Lo: r, Hi: r}} }

// foldClosure adds to s every code point that folds alike with one it holds:
// what a class or a literal matches under (?i).
func foldClosure(s spanSet) spanSet {
	targets := map[rune]bool{}
	visit := func(r rune) {
		if t := ucFold(r); t != r && (s.has(r) || s.has(t)) {
			targets[t] = true
		}
	}
	visit(0x017F)
	for c := rune('A'); c <= 'Z'; c++ {
		visit(c)
	}
	for o := 0; o+6 <= len(ucLower); o += 6 {
		visit(rune(ucWord(ucLower, o)))
	}
	if len(targets) == 0 {
		return s
	}
	add := slices.Clone(s)
	member := func(r rune) {
		if targets[ucFold(r)] {
			add = append(add, uniprop.Span{Lo: r, Hi: r})
		}
	}
	member(0x017F)
	for c := rune('A'); c <= 'Z'; c++ {
		member(c)
	}
	for o := 0; o+6 <= len(ucLower); o += 6 {
		member(rune(ucWord(ucLower, o)))
	}
	for t := range targets {
		add = append(add, uniprop.Span{Lo: t, Hi: t})
	}
	return normSpans(add)
}

// runeClass is a compiled set: a bitmap for ASCII, spans above it.
type runeClass struct {
	ascii [2]uint64
	high  spanSet
}

func classOf(s spanSet) runeClass {
	var c runeClass
	for _, x := range s {
		for r := x.Lo; r <= x.Hi && r < 0x80; r++ {
			c.ascii[r>>6] |= 1 << (uint(r) & 63)
		}
		if x.Hi >= 0x80 {
			c.high = append(c.high, uniprop.Span{Lo: max(x.Lo, 0x80), Hi: x.Hi})
		}
	}
	return c
}

func (c *runeClass) has(r rune) bool {
	if r < 0x80 {
		return r >= 0 && c.ascii[r>>6]&(1<<(uint(r)&63)) != 0
	}
	return c.high.has(r)
}

type nodeKind uint8

const (
	nodeEmpty nodeKind = iota
	nodeSet
	nodeConcat
	nodeAlt
	nodeRepeat
	nodeLook
	nodeAtomic
	nodeAssert
)

// The zero-width assertions, as a nodeAssert's min.
const (
	assertLineStart = iota
	assertLineEnd
	assertTextStart
	assertTextEnd
	assertTextEndNewline
	assertSearchStart
	assertWordBoundary
	assertNotWordBoundary
)

type node struct {
	kind     nodeKind
	set      spanSet
	subs     []*node
	min, max int // nodeRepeat; max -1 is unbounded
	lazy     bool
	possess  bool // nodeRepeat: possessive, which gives nothing back
	neg      bool // nodeLook: (?!...)
}

func (n *node) canEmpty() bool {
	switch n.kind {
	case nodeEmpty, nodeLook, nodeAssert:
		return true
	case nodeAtomic:
		return n.subs[0].canEmpty()
	case nodeSet:
		return false
	case nodeConcat:
		for _, s := range n.subs {
			if !s.canEmpty() {
				return false
			}
		}
		return true
	case nodeAlt:
		for _, s := range n.subs {
			if s.canEmpty() {
				return true
			}
		}
		return false
	case nodeRepeat:
		return n.min == 0 || n.subs[0].canEmpty()
	}
	return true
}

// maxRepeat bounds a counted repetition, which compiles to that many copies.
const maxRepeat = 1000

// maxProgram bounds a compiled pattern's instructions. Counted repetitions
// multiply when nested, so a short pattern can ask for billions; the largest
// a shipped stage compiles to is under a hundred.
const maxProgram = 1 << 16

// size is how many instructions emit writes for n, saturating past maxProgram
// so that nothing is allocated before a pattern is refused.
func (n *node) size() int {
	sat := func(v int) int { return min(v, maxProgram+1) }
	switch n.kind {
	case nodeEmpty:
		return 0
	case nodeSet, nodeAssert:
		return 1
	case nodeConcat, nodeAlt:
		t := 0
		for _, s := range n.subs {
			t = sat(t + s.size())
		}
		if n.kind == nodeAlt {
			t = sat(t + 2*(len(n.subs)-1))
		}
		return t
	case nodeLook, nodeAtomic:
		return sat(n.subs[0].size() + 2)
	case nodeRepeat:
		sub := n.subs[0]
		if sub.kind == nodeSet && !n.lazy {
			return 1
		}
		b := sub.size()
		t := sat(n.min * b)
		if n.max < 0 {
			t = sat(t + b + 2)
			if sub.canEmpty() {
				t = sat(t + 2)
			}
		} else {
			t = sat(t + (n.max-n.min)*(b+1))
		}
		if n.possess {
			t = sat(t + 2)
		}
		return t
	}
	return 0
}

type patternParser struct {
	src    []rune
	i      int
	fold   bool
	dotAll bool
	props  map[string]spanSet
}

func (p *patternParser) errf(format string, a ...any) error {
	return fmt.Errorf("tok: split pattern %q at %d: %s", string(p.src), p.i, fmt.Sprintf(format, a...))
}

func (p *patternParser) more() bool { return p.i < len(p.src) }

func (p *patternParser) peek(k int) rune {
	if p.i+k < len(p.src) {
		return p.src[p.i+k]
	}
	return -1
}

func (p *patternParser) alt() (*node, error) {
	var subs []*node
	for {
		s, err := p.seq()
		if err != nil {
			return nil, err
		}
		subs = append(subs, s)
		if p.peek(0) != '|' {
			break
		}
		p.i++
	}
	if len(subs) == 1 {
		return subs[0], nil
	}
	return &node{kind: nodeAlt, subs: subs}, nil
}

func (p *patternParser) seq() (*node, error) {
	var subs []*node
	for p.more() && p.peek(0) != '|' && p.peek(0) != ')' {
		a, err := p.atom()
		if err != nil {
			return nil, err
		}
		if a == nil {
			continue // an inline flag group, which matches nothing
		}
		if a, err = p.quantified(a); err != nil {
			return nil, err
		}
		subs = append(subs, a)
	}
	switch len(subs) {
	case 0:
		return &node{kind: nodeEmpty}, nil
	case 1:
		return subs[0], nil
	}
	return &node{kind: nodeConcat, subs: subs}, nil
}

// interval reads {n}, {n,}, {n,m} or {,m} at p.i; ok is false where the brace
// is a literal, as Oniguruma reads one that forms no interval.
func (p *patternParser) interval() (lo, hi, end int, ok bool) {
	j := p.i + 1
	num := func() (int, bool) {
		s := j
		for j < len(p.src) && p.src[j] >= '0' && p.src[j] <= '9' {
			j++
		}
		if j == s {
			return 0, false
		}
		v, err := strconv.Atoi(string(p.src[s:j]))
		return v, err == nil
	}
	lo, hasLo := num()
	hi = lo
	if j < len(p.src) && p.src[j] == ',' {
		j++
		var hasHi bool
		if hi, hasHi = num(); !hasHi {
			hi = -1
		}
		if !hasLo && !hasHi {
			return 0, 0, 0, false
		}
	} else if !hasLo {
		return 0, 0, 0, false
	}
	if j >= len(p.src) || p.src[j] != '}' {
		return 0, 0, 0, false
	}
	return lo, hi, j + 1, true
}

func (p *patternParser) quantified(a *node) (*node, error) {
	for p.more() {
		var lo, hi int
		simple := true
		switch p.peek(0) {
		case '*':
			lo, hi = 0, -1
			p.i++
		case '+':
			lo, hi = 1, -1
			p.i++
		case '?':
			lo, hi = 0, 1
			p.i++
		case '{':
			l, h, end, ok := p.interval()
			if !ok {
				return a, nil
			}
			if l > maxRepeat || h > maxRepeat || (h >= 0 && h < l) {
				return nil, p.errf("repetition {%d,%d}", l, h)
			}
			lo, hi, simple = l, h, false
			p.i = end
		default:
			return a, nil
		}
		lazy, possess := false, false
		switch p.peek(0) {
		case '?':
			lazy = true
			p.i++
		case '+':
			// Ruby's syntax reads + after * + ? as possessive; after an
			// interval it is a repetition of the repetition.
			if simple {
				possess = true
				p.i++
			}
		}
		a = &node{kind: nodeRepeat, subs: []*node{a}, min: lo, max: hi, lazy: lazy, possess: possess}
	}
	return a, nil
}

func (p *patternParser) literal(r rune) *node {
	s := runeSpan(r)
	if p.fold {
		s = foldClosure(s)
	}
	return &node{kind: nodeSet, set: s}
}

func (p *patternParser) atom() (*node, error) {
	c := p.src[p.i]
	switch c {
	case '(':
		return p.group()
	case '[':
		p.i++
		s, err := p.class()
		if err != nil {
			return nil, err
		}
		return &node{kind: nodeSet, set: s}, nil
	case '.':
		p.i++
		if p.dotAll {
			return &node{kind: nodeSet, set: spanSet{{Lo: 0, Hi: 0x10FFFF}}}, nil
		}
		return &node{kind: nodeSet, set: runeSpan('\n').complement()}, nil
	case '\\':
		switch p.peek(1) {
		case 'A':
			p.i += 2
			return &node{kind: nodeAssert, min: assertTextStart}, nil
		case 'z':
			p.i += 2
			return &node{kind: nodeAssert, min: assertTextEnd}, nil
		case 'Z':
			p.i += 2
			return &node{kind: nodeAssert, min: assertTextEndNewline}, nil
		case 'G':
			p.i += 2
			return &node{kind: nodeAssert, min: assertSearchStart}, nil
		case 'b', 'B':
			kind := assertWordBoundary
			if p.peek(1) == 'B' {
				kind = assertNotWordBoundary
			}
			p.i += 2
			w, err := p.word()
			if err != nil {
				return nil, err
			}
			return &node{kind: nodeAssert, min: kind, set: w}, nil
		}
		p.i++
		r, s, err := p.escape(false)
		if err != nil {
			return nil, err
		}
		if s != nil {
			if p.fold {
				s = foldClosure(s)
			}
			return &node{kind: nodeSet, set: s}, nil
		}
		return p.literal(r), nil
	case '^':
		p.i++
		return &node{kind: nodeAssert, min: assertLineStart}, nil
	case '$':
		p.i++
		return &node{kind: nodeAssert, min: assertLineEnd}, nil
	case '*', '+', '?':
		return nil, p.errf("%q with nothing to repeat", c)
	}
	p.i++
	return p.literal(c), nil
}

func (p *patternParser) group() (*node, error) {
	p.i++ // (
	fold, dotAll := p.fold, p.dotAll
	defer func() { p.fold, p.dotAll = fold, dotAll }()
	look, neg, atomic := false, false, false
	if p.peek(0) == '?' {
		p.i++
		switch p.peek(0) {
		case ':':
			p.i++
		case '=':
			look = true
			p.i++
		case '!':
			look, neg = true, true
			p.i++
		case '<':
			if p.peek(1) == '=' || p.peek(1) == '!' {
				return nil, p.errf("lookbehind")
			}
			p.i++
			for p.more() && p.peek(0) != '>' {
				p.i++
			}
			if !p.more() {
				return nil, p.errf("an unterminated group name")
			}
			p.i++
		case '>':
			atomic = true
			p.i++
		default:
			on := true
			for {
				switch p.peek(0) {
				case 'i':
					p.fold = on
				case 'm':
					p.dotAll = on
				case '-':
					on = false
				case ':':
					p.i++
					goto body
				case ')':
					// A flag group with no body sets the flags to the end of the
					// enclosing group: the parser keeps them, the caller's restore
					// is skipped.
					p.i++
					fold, dotAll = p.fold, p.dotAll
					return nil, nil
				default:
					return nil, p.errf("the group flag %q", p.peek(0))
				}
				p.i++
			}
		}
	}
body:
	a, err := p.alt()
	if err != nil {
		return nil, err
	}
	if p.peek(0) != ')' {
		return nil, p.errf("an unclosed group")
	}
	p.i++
	if look {
		return &node{kind: nodeLook, subs: []*node{a}, neg: neg}, nil
	}
	if atomic {
		return &node{kind: nodeAtomic, subs: []*node{a}}, nil
	}
	return a, nil
}

func (p *patternParser) property(name string) (spanSet, error) {
	if s, ok := p.props[name]; ok {
		return s, nil
	}
	key := uniprop.Key("", name)
	if key == "" {
		key = uniprop.Key("sc", name)
	}
	if key == "" {
		return nil, p.errf("\\p{%s} names no Unicode %s property", name, uniprop.Version)
	}
	spans, err := uniprop.Spans(key)
	if err != nil {
		return nil, err
	}
	if p.props == nil {
		p.props = map[string]spanSet{}
	}
	p.props[name] = spans
	return spans, nil
}

func (p *patternParser) hex(n int) (rune, error) {
	if p.i+n > len(p.src) {
		return 0, p.errf("a short hex escape")
	}
	v, err := strconv.ParseUint(string(p.src[p.i:p.i+n]), 16, 31)
	if err != nil {
		return 0, p.errf("the hex escape %q", string(p.src[p.i:p.i+n]))
	}
	p.i += n
	return rune(v), nil
}

// escape reads the escape after a backslash: a set, or a rune when s is nil.
func (p *patternParser) escape(inClass bool) (r rune, s spanSet, err error) {
	if !p.more() {
		return 0, nil, p.errf("a trailing backslash")
	}
	c := p.src[p.i]
	p.i++
	switch c {
	case 's', 'S':
		s, err = p.property("White_Space")
	case 'd', 'D':
		s, err = p.property("Nd")
	case 'w', 'W':
		s, err = p.word()
	case 'h', 'H':
		s = spanSet{{Lo: '0', Hi: '9'}, {Lo: 'A', Hi: 'F'}, {Lo: 'a', Hi: 'f'}}
	case 'p', 'P':
		if p.peek(0) != '{' {
			return 0, nil, p.errf("\\%c without a brace", c)
		}
		end := slices.Index(p.src[p.i:], '}')
		if end < 0 {
			return 0, nil, p.errf("an unterminated \\%c{", c)
		}
		name := string(p.src[p.i+1 : p.i+end])
		p.i += end + 1
		neg := c == 'P'
		if len(name) > 0 && name[0] == '^' {
			neg, name = !neg, name[1:]
		}
		if s, err = p.property(name); err != nil {
			return 0, nil, err
		}
		if neg {
			s = s.complement()
		}
		return 0, s, nil
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
	case 'a':
		return 0x07, nil, nil
	case 'e':
		return 0x1B, nil, nil
	case '0':
		return 0, nil, nil
	case 'x':
		if p.peek(0) == '{' {
			end := slices.Index(p.src[p.i:], '}')
			if end < 2 {
				return 0, nil, p.errf("an empty \\x{}")
			}
			p.i++
			r, err := p.hex(end - 1)
			p.i++
			return r, nil, err
		}
		n := 0
		for n < 2 && isHexDigit(p.peek(n)) {
			n++
		}
		if n == 0 {
			return 0, nil, p.errf("\\x with no digits")
		}
		r, err := p.hex(n)
		return r, nil, err
	case 'u':
		r, err := p.hex(4)
		return r, nil, err
	case 'b':
		if inClass {
			return 0x08, nil, nil
		}
		return 0, nil, p.errf("the word boundary \\b")
	default:
		if c < 0x80 && (c >= '0' && c <= '9' || c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z') {
			return 0, nil, p.errf("the escape \\%c", c)
		}
		return c, nil, nil
	}
	if err != nil {
		return 0, nil, err
	}
	if c >= 'A' && c <= 'Z' {
		s = s.complement()
	}
	return 0, s, nil
}

// word is \w as the tokenizers library's regex reads it: Alphabetic, \p{M},
// \p{Nd} and \p{Pc}, and below U+0100 every \p{N}, the superscripts and
// vulgar fractions of Latin-1 among them; a number above it that is not a
// decimal digit (U+2460) is not a word character (TestClassesAreTheLibrarys).
func (p *patternParser) word() (spanSet, error) {
	n, err := p.property("N")
	if err != nil {
		return nil, err
	}
	w := n.intersect(spanSet{{Lo: 0, Hi: 0xFF}})
	for _, name := range []string{"Alphabetic", "M", "Nd", "Pc"} {
		s, err := p.property(name)
		if err != nil {
			return nil, err
		}
		w = w.union(s)
	}
	return w, nil
}

func isHexDigit(r rune) bool {
	return r >= '0' && r <= '9' || r >= 'a' && r <= 'f' || r >= 'A' && r <= 'F'
}

// class reads a bracket expression after its '[' through its ']': items and
// ranges, nested classes as unions, && as intersection, and a leading ^.
func (p *patternParser) class() (spanSet, error) {
	neg := false
	if p.peek(0) == '^' {
		neg = true
		p.i++
	}
	var acc spanSet
	first := true
	for {
		if !p.more() {
			return nil, p.errf("an unterminated class")
		}
		c := p.src[p.i]
		if c == ']' && !first {
			p.i++
			break
		}
		first = false
		if c == '&' && p.peek(1) == '&' {
			p.i += 2
			// The right operand runs to this class's ']', which it consumes.
			right, err := p.classRest()
			if err != nil {
				return nil, err
			}
			acc = normSpans(acc).intersect(right)
			break
		}
		if c == '[' {
			if p.peek(1) == ':' {
				return nil, p.errf("a POSIX bracket")
			}
			p.i++
			sub, err := p.class()
			if err != nil {
				return nil, err
			}
			acc = append(acc, sub...)
			continue
		}
		lo, set, err := p.classAtom()
		if err != nil {
			return nil, err
		}
		if set != nil {
			acc = append(acc, set...)
			continue
		}
		hi := lo
		if p.peek(0) == '-' && p.peek(1) != ']' && p.peek(1) >= 0 {
			save := p.i
			p.i++
			h, hs, err := p.classAtom()
			if err != nil {
				return nil, err
			}
			if hs != nil {
				// "a-\s" is a literal '-' beside the set, as Oniguruma reads it.
				p.i = save
			} else {
				if h < lo {
					return nil, p.errf("the range %q-%q runs backwards", lo, h)
				}
				hi = h
			}
		}
		acc = append(acc, uniprop.Span{Lo: lo, Hi: hi})
	}
	s := normSpans(acc)
	if p.fold {
		s = foldClosure(s)
	}
	if neg {
		s = s.complement()
	}
	return s, nil
}

// classRest is the right operand of &&: the items to the enclosing ']',
// itself possibly another intersection.
func (p *patternParser) classRest() (spanSet, error) {
	// Parsed as a class body with no leading '^' of its own reading: a '^'
	// here is a literal, as the operand is not a bracket expression.
	var acc spanSet
	for {
		if !p.more() {
			return nil, p.errf("an unterminated class")
		}
		c := p.src[p.i]
		if c == ']' {
			p.i++
			return normSpans(acc), nil
		}
		if c == '&' && p.peek(1) == '&' {
			p.i += 2
			right, err := p.classRest()
			if err != nil {
				return nil, err
			}
			return normSpans(acc).intersect(right), nil
		}
		if c == '[' {
			p.i++
			sub, err := p.class()
			if err != nil {
				return nil, err
			}
			acc = append(acc, sub...)
			continue
		}
		lo, set, err := p.classAtom()
		if err != nil {
			return nil, err
		}
		if set != nil {
			acc = append(acc, set...)
			continue
		}
		hi := lo
		if p.peek(0) == '-' && p.peek(1) != ']' && p.peek(1) >= 0 {
			p.i++
			if hi, _, err = p.classAtom(); err != nil {
				return nil, err
			}
		}
		acc = append(acc, uniprop.Span{Lo: lo, Hi: hi})
	}
}

func (p *patternParser) classAtom() (rune, spanSet, error) {
	c := p.src[p.i]
	p.i++
	if c == '\\' {
		return p.escape(true)
	}
	return c, nil, nil
}

type instOp uint8

const (
	instSet instOp = iota
	instRep
	instSplit
	instLoop
	instJmp
	instLook
	instAtomic
	instLookEnd
	instAssert
	instMark
	instCheck
	instMatch
)

type inst struct {
	op       instOp
	neg      bool
	set      int32
	x, y     int32
	min, max int32
}

// splitPattern is a compiled pattern and how its Split stage cuts the text.
type splitPattern struct {
	src   string
	prog  []inst
	sets  []runeClass
	slots int
	reps  int // the class repetitions, each its index in its inst's x
	loops int // the loop heads, each its index in its instLoop's set
	// searchReps are the class repetitions whose continuation can reach \G,
	// so what failed after them holds for one search, not for the text.
	searchReps []int32
	behavior   splitBehavior
	invert     bool
}

// splitBehavior is what a Split does with a match: the tokenizers library's
// SplitDelimiterBehavior, by its own names.
type splitBehavior uint8

const (
	splitIsolated splitBehavior = iota
	splitRemoved
	splitMergedWithPrevious
	splitMergedWithNext
	splitContiguous
)

func splitBehaviorOf(name string) (splitBehavior, error) {
	switch name {
	case "Isolated":
		return splitIsolated, nil
	case "Removed":
		return splitRemoved, nil
	case "MergedWithPrevious":
		return splitMergedWithPrevious, nil
	case "MergedWithNext":
		return splitMergedWithNext, nil
	case "Contiguous":
		return splitContiguous, nil
	}
	return 0, fmt.Errorf("tok: split behavior %q is not one the tokenizers library defines", name)
}

// compileSplit compiles a Split stage: its pattern, its behavior's name and
// whether it is inverted.
func compileSplit(pattern, behavior string, invert bool) (*splitPattern, error) {
	b, err := splitBehaviorOf(behavior)
	if err != nil {
		return nil, err
	}
	p := &patternParser{src: []rune(pattern)}
	root, err := p.alt()
	if err != nil {
		return nil, err
	}
	if p.more() {
		return nil, p.errf("an unbalanced ')'")
	}
	if root.size() >= maxProgram {
		return nil, fmt.Errorf("tok: split pattern %q compiles to more than %d instructions", pattern, maxProgram)
	}
	sp := &splitPattern{src: pattern, behavior: b, invert: invert}
	sets := map[string]int32{}
	sp.emit(root, sets)
	sp.prog = append(sp.prog, inst{op: instMatch})
	sp.markSearchStart()
	return sp, nil
}

// markSearchStart finds every instruction from which a \G can be reached
// before its run ends, and takes those states out of the memos that outlive
// a search: \G is the one thing a continuation reads besides the position.
func (sp *splitPattern) markSearchStart() {
	preds := make([][]int32, len(sp.prog))
	var work []int32
	for pc, in := range sp.prog {
		var succ [2]int32
		k := 0
		switch in.op {
		case instLookEnd, instMatch:
		case instJmp:
			succ[0], k = in.x, 1
		case instSplit, instLoop:
			succ[0], succ[1], k = in.x, in.y, 2
		case instCheck, instLook, instAtomic:
			succ[0], succ[1], k = int32(pc+1), in.x, 2
		default:
			succ[0], k = int32(pc+1), 1
		}
		for _, t := range succ[:k] {
			preds[t] = append(preds[t], int32(pc))
		}
		if in.op == instAssert && in.min == assertSearchStart {
			work = append(work, int32(pc))
		}
	}
	sees := make([]bool, len(sp.prog))
	for _, pc := range work {
		sees[pc] = true
	}
	for len(work) > 0 {
		pc := work[len(work)-1]
		work = work[:len(work)-1]
		for _, q := range preds[pc] {
			if !sees[q] {
				sees[q] = true
				work = append(work, q)
			}
		}
	}
	for pc := range sp.prog {
		in := &sp.prog[pc]
		if !sees[pc] {
			continue
		}
		switch in.op {
		case instLoop:
			in.set = -1
		case instRep:
			sp.searchReps = append(sp.searchReps, in.x)
		}
	}
}

func (sp *splitPattern) setIndex(s spanSet, sets map[string]int32) int32 {
	key := fmt.Sprint(s)
	if i, ok := sets[key]; ok {
		return i
	}
	i := int32(len(sp.sets))
	sp.sets = append(sp.sets, classOf(s))
	sets[key] = i
	return i
}

func (sp *splitPattern) pc() int32 { return int32(len(sp.prog)) }

func (sp *splitPattern) emit(n *node, sets map[string]int32) {
	switch n.kind {
	case nodeEmpty:
	case nodeSet:
		sp.prog = append(sp.prog, inst{op: instSet, set: sp.setIndex(n.set, sets)})
	case nodeConcat:
		for _, s := range n.subs {
			sp.emit(s, sets)
		}
	case nodeAlt:
		var jumps []int32
		for k, s := range n.subs {
			if k == len(n.subs)-1 {
				sp.emit(s, sets)
				break
			}
			split := sp.pc()
			sp.prog = append(sp.prog, inst{op: instSplit, x: split + 1})
			sp.emit(s, sets)
			jumps = append(jumps, sp.pc())
			sp.prog = append(sp.prog, inst{op: instJmp})
			sp.prog[split].y = sp.pc()
		}
		for _, j := range jumps {
			sp.prog[j].x = sp.pc()
		}
	case nodeLook, nodeAtomic:
		at := sp.pc()
		op := instLook
		if n.kind == nodeAtomic {
			op = instAtomic
		}
		sp.prog = append(sp.prog, inst{op: op, neg: n.neg})
		sp.emit(n.subs[0], sets)
		sp.prog = append(sp.prog, inst{op: instLookEnd})
		sp.prog[at].x = sp.pc()
	case nodeAssert:
		in := inst{op: instAssert, min: int32(n.min)}
		if n.set != nil {
			in.set = sp.setIndex(n.set, sets)
		}
		sp.prog = append(sp.prog, in)
	case nodeRepeat:
		sub := n.subs[0]
		if sub.kind == nodeSet && !n.lazy {
			sp.prog = append(sp.prog, inst{op: instRep, set: sp.setIndex(sub.set, sets), min: int32(n.min), max: int32(n.max), neg: n.possess, x: int32(sp.reps)})
			sp.reps++
			return
		}
		if n.possess {
			greedy := *n
			greedy.possess = false
			sp.emit(&node{kind: nodeAtomic, subs: []*node{&greedy}}, sets)
			return
		}
		for range n.min {
			sp.emit(sub, sets)
		}
		if n.max < 0 {
			loop := sp.pc()
			sp.prog = append(sp.prog, inst{op: instLoop, set: int32(sp.loops)})
			sp.loops++
			slot := int32(-1)
			if sub.canEmpty() {
				slot = int32(sp.slots)
				sp.slots++
				sp.prog = append(sp.prog, inst{op: instMark, set: slot})
			}
			sp.prog[loop].x = loop + 1
			sp.emit(sub, sets)
			check := int32(-1)
			if slot >= 0 {
				check = sp.pc()
				sp.prog = append(sp.prog, inst{op: instCheck, set: slot})
			}
			sp.prog = append(sp.prog, inst{op: instJmp, x: loop})
			out := sp.pc()
			sp.prog[loop].y = out
			if check >= 0 {
				sp.prog[check].x = out
			}
			if n.lazy {
				sp.prog[loop].x, sp.prog[loop].y = sp.prog[loop].y, sp.prog[loop].x
			}
			return
		}
		var splits []int32
		for range n.max - n.min {
			splits = append(splits, sp.pc())
			sp.prog = append(sp.prog, inst{op: instSplit, x: sp.pc() + 1})
			sp.emit(sub, sets)
		}
		out := sp.pc()
		for _, s := range splits {
			sp.prog[s].y = out
			if n.lazy {
				sp.prog[s].x, sp.prog[s].y = sp.prog[s].y, sp.prog[s].x
			}
		}
	}
}

// matchFrame is one backtrack point: an alternative to resume (frameAlt), a
// greedy class run to give one code point back (frameRep), a loop mark to
// restore (frameSlot), or a loop head whose every continuation has failed
// once it is popped (frameLoop, its memo index in n).
type matchFrame struct {
	kind    uint8
	pc, pos int32
	n       int32
}

const (
	frameAlt uint8 = iota
	frameRep
	frameSlot
	frameLoop
)

// A loop head's memo, per position: unknown, every continuation failed, or
// (in a lookahead's body) the lookahead matched.
const (
	loopUnknown uint8 = iota
	loopFailed
	loopMatched
)

// loopMemoBytes bounds the loop-head memo, one byte per loop head and
// position; loop heads past it run unmemoized on a text that long.
const loopMemoBytes = 1 << 24

// matcher is a split's per-stage scratch, reused across calls.
//
// Its memos rest on the program holding no state but the position: there are
// no captures, and the memos are off where an empty-loop mark would be state.
// A continuation that failed from a position fails from it again, and one
// that reached a lookahead's end reaches it again, so a text whose runs are
// retried from every start (\s+$ before a digit, a lookahead counting digit
// triples to the end of the run) is split in linear time. Every memo holds
// for the whole text except what can reach \G, which is reset per search.
type matcher struct {
	stack []matchFrame
	slots []int32
	start int // where the search began, for \G
	memo  bool
	// fail is, per class repetition, a run of end positions its continuation
	// is known to fail from.
	fail []failRun
	// runs is, per class, the last maximal run of it found.
	runs []classRun
	// loop is the loop heads' memo, width positions per head; heads from
	// loopsOn on are not memoized.
	loop    []uint8
	width   int
	loopsOn int
	steps   int // instructions run, code points scanned and frames popped, ever
}

// failRun is the positions lo through hi, empty when lo > hi.
type failRun struct{ lo, hi int32 }

// classRun is the positions from up to to, all in the class; ended when the
// class does not continue at to. The zero value is the empty run at 0.
type classRun struct {
	from, to int32
	ended    bool
}

func (f *failRun) learn(p int32) {
	switch {
	case f.lo <= p && p <= f.hi:
	case p == f.lo-1:
		f.lo = p
	case p == f.hi+1:
		f.hi = p
	default:
		f.lo, f.hi = p, p
	}
}

// reset readies m for a text of n code points.
func (sp *splitPattern) reset(m *matcher, n int) {
	if len(m.slots) < sp.slots {
		m.slots = make([]int32, sp.slots)
	}
	m.memo = sp.slots == 0
	m.fail = slices.Grow(m.fail[:0], sp.reps)[:sp.reps]
	for i := range m.fail {
		m.fail[i] = failRun{lo: 1}
	}
	m.runs = slices.Grow(m.runs[:0], len(sp.sets))[:len(sp.sets)]
	clear(m.runs)
	m.loopsOn = 0
	if m.memo && sp.loops > 0 {
		m.width = n + 1
		m.loopsOn = min(sp.loops, loopMemoBytes/m.width)
		m.loop = slices.Grow(m.loop[:0], m.loopsOn*m.width)[:m.loopsOn*m.width]
		clear(m.loop)
	}
}

// scan is how many code points from pos the class repetition in takes, at
// most its max; a run the class is known to hold is crossed without reading
// it. A class run is the text's, whatever the program holds.
func (sp *splitPattern) scan(m *matcher, rs []rune, in *inst, pos int) int {
	set := &sp.sets[in.set]
	r := &m.runs[in.set]
	c, ended := 0, false
	for {
		if in.max >= 0 && c >= int(in.max) {
			c = int(in.max)
			break
		}
		p := int32(pos + c)
		if r.from <= p && p < r.to {
			c = int(r.to) - pos
			continue
		}
		if p == r.to && r.ended {
			ended = true
			break
		}
		if int(p) >= len(rs) || !set.has(rs[p]) {
			ended = true
			break
		}
		m.steps++
		c++
	}
	e := int32(pos + c)
	if int32(pos) <= r.to && e >= r.from {
		r.from = min(r.from, int32(pos))
		if e > r.to {
			r.to, r.ended = e, ended
		} else if e == r.to {
			r.ended = r.ended || ended
		}
	} else {
		*r = classRun{from: int32(pos), to: e, ended: ended}
	}
	return c
}

// run executes prog from pc at pos and reports where the match ends. Frames
// below the stack depth it starts at belong to its caller, which is how a
// lookahead runs as a sub-match that keeps none of its own alternatives.
func (sp *splitPattern) run(m *matcher, rs []rune, pc, pos int) (int, bool) {
	base := len(m.stack)
	n := len(rs)
	for {
		in := &sp.prog[pc]
		m.steps++
		ok := true
		switch in.op {
		case instSet:
			if pos < n && sp.sets[in.set].has(rs[pos]) {
				pos++
				pc++
			} else {
				ok = false
			}
		case instRep:
			c := sp.scan(m, rs, in, pos)
			if c < int(in.min) {
				ok = false
				break
			}
			if m.memo {
				f := &m.fail[in.x]
				if f.lo <= int32(pos+c) && int32(pos+c) <= f.hi {
					if in.neg || f.lo-int32(pos)-1 < in.min {
						ok = false
						break
					}
					c = int(f.lo) - pos - 1
				}
			}
			if c > int(in.min) && !in.neg {
				m.stack = append(m.stack, matchFrame{kind: frameRep, pc: int32(pc), pos: int32(pos), n: int32(c - 1)})
			}
			pos += c
			pc++
		case instLoop:
			if in.set >= 0 && int(in.set) < m.loopsOn {
				switch m.loop[int(in.set)*m.width+pos] {
				case loopFailed:
					ok = false
				case loopMatched:
					return pos, true
				default:
					m.stack = append(m.stack, matchFrame{kind: frameLoop, pc: int32(pc), pos: int32(pos), n: in.set})
				}
			}
			if ok {
				m.stack = append(m.stack, matchFrame{kind: frameAlt, pc: in.y, pos: int32(pos)})
				pc = int(in.x)
			}
		case instSplit:
			m.stack = append(m.stack, matchFrame{kind: frameAlt, pc: in.y, pos: int32(pos)})
			pc = int(in.x)
		case instJmp:
			pc = int(in.x)
		case instLook:
			depth := len(m.stack)
			_, found := sp.run(m, rs, pc+1, pos)
			if found {
				for _, f := range m.stack[depth:] {
					if f.kind == frameLoop {
						m.loop[int(f.n)*m.width+int(f.pos)] = loopMatched
					}
				}
			}
			m.stack = m.stack[:depth]
			if found != in.neg {
				pc = int(in.x)
			} else {
				ok = false
			}
		case instAtomic:
			depth := len(m.stack)
			end, found := sp.run(m, rs, pc+1, pos)
			m.stack = m.stack[:depth]
			if found {
				pos, pc = end, int(in.x)
			} else {
				ok = false
			}
		case instAssert:
			switch in.min {
			case assertLineStart:
				ok = pos == 0 || rs[pos-1] == '\n'
			case assertLineEnd:
				ok = pos == n || rs[pos] == '\n'
			case assertTextStart:
				ok = pos == 0
			case assertTextEnd:
				ok = pos == n
			case assertTextEndNewline:
				ok = pos == n || pos == n-1 && rs[pos] == '\n'
			case assertSearchStart:
				ok = pos == m.start
			case assertWordBoundary, assertNotWordBoundary:
				w := &sp.sets[in.set]
				before := pos > 0 && w.has(rs[pos-1])
				after := pos < n && w.has(rs[pos])
				ok = (before != after) == (in.min == assertWordBoundary)
			}
			if ok {
				pc++
			}
		case instLookEnd, instMatch:
			return pos, true
		case instMark:
			m.stack = append(m.stack, matchFrame{kind: frameSlot, pc: in.set, pos: m.slots[in.set]})
			m.slots[in.set] = int32(pos)
			pc++
		case instCheck:
			if m.slots[in.set] == int32(pos) {
				pc = int(in.x)
			} else {
				pc++
			}
		}
		if ok {
			continue
		}
		for resumed := false; !resumed; {
			if len(m.stack) == base {
				return 0, false
			}
			f := m.stack[len(m.stack)-1]
			m.stack = m.stack[:len(m.stack)-1]
			m.steps++
			switch f.kind {
			case frameAlt:
				pc, pos, resumed = int(f.pc), int(f.pos), true
			case frameRep:
				in := &sp.prog[f.pc]
				k := f.n
				if m.memo {
					// Every path from the count above this one has failed.
					iv := &m.fail[in.x]
					iv.learn(f.pos + k + 1)
					if iv.lo <= f.pos+k && f.pos+k <= iv.hi {
						k = iv.lo - f.pos - 1
					}
				}
				if k < in.min {
					break
				}
				if k > in.min {
					m.stack = append(m.stack, matchFrame{kind: frameRep, pc: f.pc, pos: f.pos, n: k - 1})
				}
				pc, pos, resumed = int(f.pc)+1, int(f.pos+k), true
			case frameSlot:
				m.slots[f.pc] = f.pos
			case frameLoop:
				m.loop[int(f.n)*m.width+int(f.pos)] = loopFailed
			}
		}
	}
}

// find is the leftmost match starting at or after from.
func (sp *splitPattern) find(m *matcher, rs []rune, from int) (int, int, bool) {
	for _, i := range sp.searchReps {
		m.fail[i] = failRun{lo: 1}
	}
	m.start = from
	for s := from; s <= len(rs); s++ {
		m.stack = m.stack[:0]
		if e, ok := sp.run(m, rs, 0, s); ok {
			return s, e, true
		}
	}
	return 0, 0, false
}

// each splits s, handing every non-empty piece to emit in order: the matches
// as the tokenizers library finds them (an empty match right after the last
// match is skipped, one code point on), the text between them, and the
// behavior applied to the two.
func (sp *splitPattern) each(s string, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	n := len(rs)
	m := &sc.m
	sp.reset(m, n)
	out := func(a, b int) {
		if a < b {
			emit(sc.piece(s, a, b))
		}
	}
	var pa, pb int
	var pm, have, prevMatch bool
	seg := func(a, b int, match bool) {
		if sp.invert {
			match = !match
		}
		switch sp.behavior {
		case splitIsolated:
			out(a, b)
		case splitRemoved:
			if !match {
				out(a, b)
			}
		case splitMergedWithPrevious:
			if match && !prevMatch && have {
				pb = b
			} else {
				if have {
					out(pa, pb)
				}
				pa, pb, have = a, b, true
			}
			prevMatch = match
		case splitMergedWithNext:
			if have && pm && !match {
				out(pa, b)
				have = false
			} else {
				if have {
					out(pa, pb)
				}
				pa, pb, pm, have = a, b, match, true
			}
		case splitContiguous:
			if have && match == pm {
				pb = b
			} else {
				if have {
					out(pa, pb)
				}
				pa, pb, pm, have = a, b, match, true
			}
		}
	}
	prev, lastEnd, lastMatchEnd := 0, 0, -1
	for lastEnd <= n {
		a, b, ok := sp.find(m, rs, lastEnd)
		if !ok {
			break
		}
		if a == b && lastMatchEnd == b {
			lastEnd = b + 1
			continue
		}
		lastEnd, lastMatchEnd = b, b
		if prev != a {
			seg(prev, a, false)
		}
		seg(a, b, true)
		prev = b
	}
	if prev != n {
		seg(prev, n, false)
	}
	if have {
		out(pa, pb)
	}
}

// split is each, gathered.
func (sp *splitPattern) split(s string) []string {
	var sc runes
	return collect(func(emit func(string)) { sp.each(s, &sc, emit) })
}
