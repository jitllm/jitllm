package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"github.com/jitllm/jitllm/format/jlm"
)

// The one reader of every stated tool form (jlm.ToolForm): a reply matched
// against the form's templates, holes read as the form says. It knows text,
// holes and a value notation; no form is its own code.

type tkind uint8

const (
	tLit tkind = iota
	tHole
	tOpt
)

// tnode is a template's part: text, a hole (text is its name, NAME for
// $NAME), or an optional run of parts.
type tnode struct {
	kind tkind
	text string
	sub  []tnode
}

// parseTemplate is t's parts.
func parseTemplate(t string) []tnode {
	var out []tnode
	var lit strings.Builder
	flush := func() {
		if lit.Len() > 0 {
			out = append(out, tnode{kind: tLit, text: lit.String()})
			lit.Reset()
		}
	}
	for i := 0; i < len(t); {
		switch {
		case strings.HasPrefix(t[i:], "(?"):
			end := strings.Index(t[i+2:], "?)")
			if end < 0 {
				lit.WriteString(t[i:])
				i = len(t)
				continue
			}
			flush()
			out = append(out, tnode{kind: tOpt, sub: parseTemplate(t[i+2 : i+2+end])})
			i += end + 4
		case t[i] == '$' && i+1 < len(t) && t[i+1] >= 'A' && t[i+1] <= 'Z':
			j := i + 1
			for j < len(t) && t[j] >= 'A' && t[j] <= 'Z' {
				j++
			}
			flush()
			out = append(out, tnode{kind: tHole, text: t[i+1 : j]})
			i = j
		default:
			lit.WriteByte(t[i])
			i++
		}
	}
	flush()
	return out
}

// mstate is a match's outcome: ok with where it ended, partial where the text
// ended inside it (a stream may yet finish it), or neither.
type mstate uint8

const (
	mFail mstate = iota
	mOK
	mPartial
)

// param is one argument a $PARAMS read: its key, its value as text (a text
// value) or as JSON, and the type word its $TYPE gave.
type param struct {
	key, val, typ string
	text          bool
}

// caps is what a call's holes read. It is copied as a match backtracks, its
// slices never appended in place.
type caps struct {
	name, args string
	hasArgs    bool
	calls      []ToolCall
	params     []param
	key, typ   string
	val        string
	valText    bool
	// tail is the whitespace the last text part ended with, which a text
	// value next to it does not hold.
	tail string
	// whole says every hole of the call has read: what is left is marks.
	whole bool
}

// hasHole is whether the parts ns hold a hole.
func hasHole(ns []tnode) bool {
	for _, n := range ns {
		if n.kind == tHole || n.kind == tOpt && hasHole(n.sub) {
			return true
		}
	}
	return false
}

// matcher matches one reply against a form.
type matcher struct {
	f    *jlm.ToolForm
	ts   ToolSet
	text string
}

// res is a match's result.
type res struct {
	st  mstate
	pos int
	c   caps
}

// chunks are a text part's marks: its text split at whitespace, which
// matches any run of whitespace or none.
func chunks(s string) []string { return strings.Fields(s) }

// skipWS is pos past the whitespace at it.
func (m *matcher) skipWS(pos int) int {
	for pos < len(m.text) && unicode.IsSpace(rune(m.text[pos])) {
		pos++
	}
	return pos
}

// lit matches text part s at pos.
func (m *matcher) lit(pos int, s string) (int, mstate) {
	for _, ch := range chunks(s) {
		pos = m.skipWS(pos)
		rest := m.text[pos:]
		switch {
		case strings.HasPrefix(rest, ch):
			pos += len(ch)
		case len(rest) < len(ch) && strings.HasPrefix(ch, rest):
			return pos, mPartial
		default:
			return pos, mFail
		}
	}
	return pos, mOK
}

// nextMark is the first mark of the parts ns, or after when they hold none
// that must be written.
func nextMark(ns []tnode, after string) string {
	for _, n := range ns {
		if n.kind == tLit {
			if cs := chunks(n.text); len(cs) > 0 {
				return cs[0]
			}
		}
		if n.kind == tHole {
			return ""
		}
	}
	return after
}

// lead is the whitespace s opens with.
func lead(s string) string { return s[:len(s)-len(strings.TrimLeftFunc(s, unicode.IsSpace))] }

// trail is the whitespace s ends with.
func trail(s string) string { return s[len(strings.TrimRightFunc(s, unicode.IsSpace)):] }

// seq matches the parts ns at pos, then k; after is the mark that follows
// ns where ns end on a hole that reads up to one.
func (m *matcher) seq(ns []tnode, pos int, c caps, after string, k func(int, caps) res) res {
	if len(ns) == 0 {
		return k(pos, c)
	}
	n, rest := ns[0], ns[1:]
	switch n.kind {
	case tLit:
		p, st := m.lit(pos, n.text)
		if st != mOK {
			return res{st: st, pos: p, c: c}
		}
		c.tail = trail(n.text)
		return m.seq(rest, p, c, after, k)
	case tOpt:
		with := m.seq(n.sub, pos, c, nextMark(rest, after), func(p int, c2 caps) res { return m.seq(rest, p, c2, after, k) })
		if with.st == mOK {
			return with
		}
		without := m.seq(rest, pos, c, after, k)
		if without.st == mOK || with.st != mPartial {
			return without
		}
		return with
	}
	mark := nextMark(rest, after)
	next := ""
	if len(rest) > 0 && rest[0].kind == tLit {
		next = lead(rest[0].text)
	}
	p, c2, st := m.hole(n.text, pos, c, mark, next)
	if st != mOK {
		return res{st: st, pos: p, c: c}
	}
	c2.tail = ""
	c2.whole = !hasHole(rest)
	return m.seq(rest, p, c2, after, k)
}

// upTo is the text from pos to mark, and where mark begins: partial where
// the text ends before it.
func (m *matcher) upTo(pos int, mark string) (string, int, mstate) {
	if mark == "" {
		return "", pos, mFail
	}
	i := strings.Index(m.text[pos:], mark)
	if i < 0 {
		return "", len(m.text), mPartial
	}
	return m.text[pos : pos+i], pos + i, mOK
}

func nameByte(c byte) bool {
	return c == '_' || c == '.' || c == '-' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

// hole reads hole h at pos; mark is the template's next mark, next the
// whitespace its next text part opens with.
func (m *matcher) hole(h string, pos int, c caps, mark, next string) (int, caps, mstate) {
	switch h {
	case "NAME", "ID":
		p := m.skipWS(pos)
		st := p
		for p < len(m.text) && nameByte(m.text[p]) && (h == "NAME" || m.text[p] != '.' && m.text[p] != '-' && m.text[p] != '_') {
			p++
		}
		if p == len(m.text) {
			return p, c, mPartial
		}
		if p == st {
			return p, c, mFail
		}
		if h == "NAME" {
			c.name = m.text[st:p]
		}
		return p, c, mOK
	case "QNAME":
		v, p, s := m.value(pos)
		if s != mOK {
			return p, c, s
		}
		var name string
		if json.Unmarshal([]byte(v), &name) != nil {
			return p, c, mFail
		}
		c.name = name
		return p, c, mOK
	case "XNAME", "KEY", "TYPE":
		v, p, s := m.upTo(pos, mark)
		if s != mOK {
			return p, c, s
		}
		v = strings.TrimSpace(v)
		switch h {
		case "XNAME":
			c.name = strings.ReplaceAll(strings.ReplaceAll(v, "&quot;", `"`), "&amp;", "&")
		case "KEY":
			c.key = v
		default:
			c.typ = v
		}
		return p, c, mOK
	case "VALUE":
		if m.f.TextValues {
			v, p, s := m.upTo(pos, mark)
			if s != mOK {
				return p, c, s
			}
			v = strings.TrimSuffix(strings.TrimPrefix(v, c.tail), next)
			c.val, c.valText = v, true
			return p, c, mOK
		}
		v, p, s := m.value(pos)
		if s != mOK {
			return p, c, s
		}
		c.val, c.valText = v, false
		return p, c, mOK
	case "ARGS":
		v, p, s := m.value(pos)
		if s != mOK {
			return p, c, s
		}
		a, ok := compactObject([]byte(v))
		if !ok {
			return p, c, mFail
		}
		c.args, c.hasArgs = a, true
		return p, c, mOK
	case "CALL", "CALLS":
		p := m.skipWS(pos)
		doubled := strings.HasPrefix(m.text[p:], "{{")
		if doubled {
			// A template's example may write a call {{"name": ...}}, which
			// transformers renders verbatim and a model copies.
			p++
		}
		v, p, s := m.value(p)
		if s != mOK {
			return p, c, s
		}
		if doubled {
			if q := m.skipWS(p); strings.HasPrefix(m.text[q:], "}") {
				p = q + 1
			}
		}
		calls, ok := m.jsonCalls(json.RawMessage(v), h == "CALLS")
		if !ok {
			return p, c, mFail
		}
		c.calls = append(slices.Clip(c.calls), calls...)
		return p, c, mOK
	case "PARAMS":
		return m.params(pos, c, mark)
	}
	return pos, c, mFail
}

// params reads the arguments of $PARAMS: each in one of the form's Params
// templates, ParamSep between two, until none matches.
func (m *matcher) params(pos int, c caps, mark string) (int, caps, mstate) {
	var ps []param
	tmpls := make([][]tnode, len(m.f.Params))
	for i, t := range m.f.Params {
		tmpls[i] = parseTemplate(t)
	}
	p := pos
	for {
		q := p
		if len(ps) > 0 && m.f.ParamSep != "" {
			var st mstate
			if q, st = m.lit(q, m.f.ParamSep); st != mOK {
				break
			}
		}
		found := false
		for _, t := range tmpls {
			r := m.seq(t, q, caps{}, mark, func(e int, cc caps) res { return res{st: mOK, pos: e, c: cc} })
			if r.st == mPartial {
				return r.pos, c, mPartial
			}
			if r.st == mOK {
				ps = append(ps, param{key: r.c.key, val: r.c.val, typ: r.c.typ, text: r.c.valText})
				p, found = r.pos, true
				break
			}
		}
		if !found {
			break
		}
	}
	c.params = append(slices.Clip(c.params), ps...)
	return p, c, mOK
}

// jsonCalls is the calls of one value: an object, or (many) a list of them,
// each its name and arguments by the form's keys.
func (m *matcher) jsonCalls(raw json.RawMessage, many bool) ([]ToolCall, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return nil, false
	}
	list := []json.RawMessage{raw}
	if raw[0] == '[' {
		if !many || json.Unmarshal(raw, &list) != nil || len(list) == 0 {
			return nil, false
		}
	}
	calls := make([]ToolCall, 0, len(list))
	for _, r := range list {
		c, ok := m.jsonCall(r)
		if !ok {
			return nil, false
		}
		calls = append(calls, c)
	}
	return calls, true
}

// jsonCall is one call written as an object: {NameKey: ..., ArgsKey: {...}}
// (AltArgsKey where ArgsKey is absent), or {NAME: {...}} when Keyed.
func (m *matcher) jsonCall(r []byte) (ToolCall, bool) {
	var o map[string]json.RawMessage
	if json.Unmarshal(r, &o) != nil {
		return ToolCall{}, false
	}
	f := m.f
	if f.Keyed {
		if len(o) != 1 {
			return ToolCall{}, false
		}
		for n, a := range o {
			args, ok := compactObject(a)
			return ToolCall{Name: n, Arguments: args}, ok
		}
	}
	var name string
	if json.Unmarshal(o[f.NameKey], &name) != nil || name == "" {
		return ToolCall{}, false
	}
	a := o[f.ArgsKey]
	if len(a) == 0 && f.AltArgsKey != "" {
		a = o[f.AltArgsKey]
	}
	args, ok := compactObject(a)
	return ToolCall{Name: name, Arguments: args}, ok
}

// jsonNotation is whether the form writes values as JSON does.
func (m *matcher) jsonNotation() bool {
	n := m.f.Notation
	return len(n.Quotes) == 1 && n.Quotes[0] == `"` && n.Escapes && !n.BareKeys
}

// value reads one value at pos in the form's notation, as JSON text.
func (m *matcher) value(pos int) (string, int, mstate) {
	pos = m.skipWS(pos)
	if pos >= len(m.text) {
		return "", pos, mPartial
	}
	if m.jsonNotation() {
		d := json.NewDecoder(strings.NewReader(m.text[pos:]))
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
				return "", len(m.text), mPartial
			}
			return "", pos, mFail
		}
		return string(raw), pos + int(d.InputOffset()), mOK
	}
	var b strings.Builder
	p, st := m.nvalue(pos, &b)
	return b.String(), p, st
}

// nvalue writes the value at pos, in the form's notation, to b as JSON.
func (m *matcher) nvalue(pos int, b *strings.Builder) (int, mstate) {
	n := m.f.Notation
	t := m.text
	pos = m.skipWS(pos)
	if pos >= len(t) {
		return pos, mPartial
	}
	if s, p, st := m.nstring(pos); st != mFail {
		if st == mOK {
			j, _ := json.Marshal(s)
			b.Write(j)
		}
		return p, st
	}
	switch t[pos] {
	case '{':
		b.WriteByte('{')
		p := pos + 1
		for i := 0; ; i++ {
			p = m.skipWS(p)
			if p >= len(t) {
				return p, mPartial
			}
			if t[p] == '}' {
				b.WriteByte('}')
				return p + 1, mOK
			}
			if i > 0 {
				if t[p] != ',' {
					return p, mFail
				}
				b.WriteByte(',')
				p = m.skipWS(p + 1)
			}
			key, q, st := m.nstring(p)
			if st == mFail && n.BareKeys {
				c := strings.IndexByte(t[p:], ':')
				if c < 0 {
					return len(t), mPartial
				}
				key, q, st = strings.TrimSpace(t[p:p+c]), p+c, mOK
			}
			if st != mOK {
				return q, st
			}
			kj, _ := json.Marshal(key)
			b.Write(kj)
			q = m.skipWS(q)
			if q >= len(t) {
				return q, mPartial
			}
			if t[q] != ':' {
				return q, mFail
			}
			b.WriteByte(':')
			var vst mstate
			if p, vst = m.nvalue(q+1, b); vst != mOK {
				return p, vst
			}
		}
	case '[':
		b.WriteByte('[')
		p := pos + 1
		for i := 0; ; i++ {
			p = m.skipWS(p)
			if p >= len(t) {
				return p, mPartial
			}
			if t[p] == ']' {
				b.WriteByte(']')
				return p + 1, mOK
			}
			if i > 0 {
				if t[p] != ',' {
					return p, mFail
				}
				b.WriteByte(',')
				p = m.skipWS(p + 1)
				if p < len(t) && t[p] == ']' {
					b.WriteByte(']')
					return p + 1, mOK
				}
			}
			var vst mstate
			if p, vst = m.nvalue(p, b); vst != mOK {
				return p, vst
			}
		}
	}
	for _, w := range [3]struct {
		words []string
		json  string
	}{{n.True, "true"}, {n.False, "false"}, {n.None, "null"}} {
		for _, x := range w.words {
			if strings.HasPrefix(t[pos:], x) {
				b.WriteString(w.json)
				return pos + len(x), mOK
			}
		}
	}
	e := pos
	for e < len(t) && strings.IndexByte("+-0123456789.eE", t[e]) >= 0 {
		e++
	}
	if e == pos {
		return pos, mFail
	}
	if e == len(t) {
		return e, mPartial
	}
	f, err := strconv.ParseFloat(t[pos:e], 64)
	if err != nil {
		return pos, mFail
	}
	if json.Valid([]byte(t[pos:e])) {
		b.WriteString(t[pos:e])
	} else {
		b.WriteString(strconv.FormatFloat(f, 'g', -1, 64))
	}
	return e, mOK
}

// nstring reads a string at pos between one of the notation's quotes.
func (m *matcher) nstring(pos int) (string, int, mstate) {
	n := m.f.Notation
	t := m.text
	for _, q := range n.Quotes {
		if !strings.HasPrefix(t[pos:], q) {
			continue
		}
		var b strings.Builder
		for p := pos + len(q); p < len(t); {
			if strings.HasPrefix(t[p:], q) {
				return b.String(), p + len(q), mOK
			}
			c := t[p]
			if n.Escapes && c == '\\' {
				if p+1 >= len(t) {
					return "", len(t), mPartial
				}
				switch e := t[p+1]; e {
				case 'n':
					b.WriteByte('\n')
				case 't':
					b.WriteByte('\t')
				case 'r':
					b.WriteByte('\r')
				default:
					b.WriteByte(e)
				}
				p += 2
				continue
			}
			b.WriteByte(c)
			p++
		}
		return "", len(t), mPartial
	}
	return "", pos, mFail
}

// call matches one call at pos: in its Section, in the first of the form's
// templates that reads it.
func (m *matcher) call(pos int) res {
	var tmpl string
	for _, t := range m.f.Calls {
		tmpl += m.f.Section[0] + t + m.f.Section[1] + "\x00"
	}
	partial := res{st: mFail}
	for _, t := range strings.Split(strings.TrimSuffix(tmpl, "\x00"), "\x00") {
		r := m.seq(parseTemplate(t), pos, caps{}, "", func(e int, c caps) res { return res{st: mOK, pos: e, c: c} })
		if r.st == mOK {
			return r
		}
		if r.st == mPartial && partial.st != mPartial {
			partial = r
		}
	}
	return partial
}

// callsOf is the calls a matched call's holes read.
func (m *matcher) callsOf(c caps) []ToolCall {
	if len(c.calls) > 0 {
		return c.calls
	}
	args := c.args
	if !c.hasArgs {
		var ob objectBuilder
		for _, p := range c.params {
			typ := p.typ
			if w, ok := m.f.Types[typ]; ok {
				typ = w
			}
			if !p.text {
				v := p.val
				var s string
				if t := m.ts.propType(c.name, p.key); t != "" && t != "string" && json.Unmarshal([]byte(v), &s) == nil {
					// A notation that quotes every value; the schema says what
					// it is.
					var x any
					if json.Unmarshal([]byte(s), &x) == nil {
						v = s
					}
				}
				ob.add(p.key, v, "json", false)
				continue
			}
			guess := m.f.GuessJSON && typ == ""
			if typ == "" {
				typ = m.ts.propType(c.name, p.key)
			}
			ob.add(p.key, p.val, typ, guess)
		}
		args = ob.String()
	}
	return []ToolCall{{Name: c.name, Arguments: args}}
}
