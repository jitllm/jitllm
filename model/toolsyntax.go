package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode"

	"github.com/jitllm/jitllm/format/jlm"
)

// ToolSyntax is how a model writes a tool call: the form its container states
// (jlm.ToolForm), stated by the converter from the chat template the weights
// were trained to emit. Every form is read by one matcher (toolmatch.go)
// over the form's templates; no form has code of its own.
//
// A form is read from the reply's RAW text: the control tokens it is written
// in spelled out as their literal text, since the text a person reads drops
// them (ToolStream builds that text from the ids).
type ToolSyntax struct {
	f *jlm.ToolForm
	// envelope is the generic form: the reply is one JSON object holding the
	// calls or the answer (toolgeneric.go), read by its own scanner.
	envelope bool
}

// ToolSyntaxFor is the syntax of a stated form.
func ToolSyntaxFor(f jlm.ToolForm) ToolSyntax { return ToolSyntax{f: &f} }

// String is the form's name.
func (s ToolSyntax) String() string {
	if s.f == nil {
		return "none"
	}
	return s.f.Name
}

// ToolSyntax is the syntax this model calls tools in: the form its container
// states, or the generic form where its template reads no tools. A container
// with a chat template and no form was written before forms were stated, and is
// refused rather than read in a form it never stated; one with no template
// states no calls.
func (m *Model) ToolSyntax() (ToolSyntax, error) {
	var v *jlm.Vocab
	if m != nil && m.Container != nil {
		v = m.Container.Vocab()
	}
	if v == nil || len(v.Templates) == 0 {
		return ToolSyntax{}, nil
	}
	if src, err := m.ChatTemplateFor(true, false); err == nil && !readsTools(src) {
		return genericSyntax(), nil
	}
	if len(v.Tools.Calls) == 0 {
		return ToolSyntax{}, fmt.Errorf("model: this container states no tool-call form (it was written before forms were stated): convert it again")
	}
	return ToolSyntaxFor(v.Tools), nil
}

// ToolSet is a request's declared tools: their names, which are the only
// calls a reply can make, and their parameter schemas, which type the
// arguments of a form that writes them as text.
type ToolSet struct {
	names   []string
	schemas map[string]json.RawMessage
	// any accepts every name.
	any bool
}

// ToolSetOf is a set of the named tools with no schemas; nil names any.
func ToolSetOf(names []string) ToolSet { return ToolSet{names: names, any: names == nil} }

// ParseToolSet reads an OpenAI-shaped tool list ([{"type": "function",
// "function": {name, parameters}}]).
func ParseToolSet(tools []byte) ToolSet {
	var ts []struct {
		Function struct {
			Name       string          `json:"name"`
			Parameters json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	t := ToolSet{schemas: map[string]json.RawMessage{}}
	if json.Unmarshal(tools, &ts) != nil {
		return t
	}
	for _, x := range ts {
		t.names = append(t.names, x.Function.Name)
		t.schemas[x.Function.Name] = x.Function.Parameters
	}
	return t
}

// Names are the declared tools' names.
func (t ToolSet) Names() []string { return t.names }

func (t ToolSet) known(n string) bool { return n != "" && (t.any || slices.Contains(t.names, n)) }

// propType is the declared type of tool's argument key: "" when the schema
// names none.
func (t ToolSet) propType(tool, key string) string {
	var s struct {
		Properties map[string]struct {
			Type json.RawMessage `json:"type"`
		} `json:"properties"`
	}
	if json.Unmarshal(t.schemas[tool], &s) != nil {
		return ""
	}
	p, ok := s.Properties[key]
	if !ok {
		return ""
	}
	var one string
	if json.Unmarshal(p.Type, &one) == nil {
		return one
	}
	var many []string
	if json.Unmarshal(p.Type, &many) == nil {
		for _, x := range many {
			if x != "null" {
				return x
			}
		}
	}
	return ""
}

// toolSpan is one region of a reply that is tool-call markup: [start, end) of
// the raw text, whether its closing mark has arrived, and its calls. ok is
// false for markup that turned out not to be a declared, well-formed call,
// which stays text.
type toolSpan struct {
	start, end int
	closed, ok bool
	calls      []ToolCall
}

// ToolCallSpan is a call found in a reply: where its markup is in the raw
// text, and whether the markup has closed (a stream can send the call).
type ToolCallSpan struct {
	Start, End int
	Closed     bool
	Calls      []ToolCall
}

// Find reads every tool call in text. Markup that is not a well-formed call
// to a declared tool is not a call and stays text.
func (s ToolSyntax) Find(text string, tools ToolSet) []ToolCallSpan {
	if s.envelope {
		if r := readGeneric(text, tools, true); len(r.calls) > 0 {
			return []ToolCallSpan{{Start: r.start, End: len(text), Closed: true, Calls: r.calls}}
		}
		return nil
	}
	var out []ToolCallSpan
	for _, sp := range s.find(text, tools) {
		if sp.ok {
			out = append(out, ToolCallSpan{Start: sp.start, End: sp.end, Closed: sp.closed, Calls: sp.calls})
		}
	}
	return out
}

// Parse is the reply's text with the calls' markup cut out, and the calls.
func (s ToolSyntax) Parse(text string, tools ToolSet) (string, []ToolCall) {
	if s.envelope {
		r := readGeneric(text, tools, true)
		return r.text(text), r.calls
	}
	spans := s.Find(text, tools)
	if len(spans) == 0 {
		return text, nil
	}
	var b strings.Builder
	var calls []ToolCall
	at := 0
	for _, sp := range spans {
		b.WriteString(text[at:sp.Start])
		at = sp.End
		calls = append(calls, sp.Calls...)
	}
	b.WriteString(text[at:])
	return strings.TrimSpace(b.String()), calls
}

// opens are the marks a reply's calls may begin at anywhere in it: Open's
// first, else the Section's, else each template's that opens with text.
func (s ToolSyntax) opens() []string {
	f := s.f
	first := func(t string) string {
		ns := parseTemplate(t)
		if len(ns) > 0 && ns[0].kind == tLit {
			if cs := chunks(ns[0].text); len(cs) > 0 {
				return cs[0]
			}
		}
		return ""
	}
	if o := first(f.Open); o != "" {
		return []string{o}
	}
	if o := first(f.Section[0]); o != "" {
		return []string{o}
	}
	var out []string
	for _, t := range f.Calls {
		if o := first(t); o != "" && !slices.Contains(out, o) {
			out = append(out, o)
		}
	}
	return out
}

// starts is where the calls may begin from at on: the reply's start, where
// the form may open a reply with its calls, and each mark of opens.
func (s ToolSyntax) starts(text string, at int) []int {
	f := s.f
	start := len(text) - len(strings.TrimLeftFunc(text, unicode.IsSpace))
	var out []int
	if at <= start && (f.Bare || f.Leads) {
		out = append(out, start)
	}
	if f.Leads {
		return out
	}
	for _, o := range s.opens() {
		if i := strings.Index(text[at:], o); i >= 0 {
			out = append(out, at+i)
		}
	}
	slices.Sort(out)
	return slices.Compact(out)
}

// reply matches a reply's calls at pos: Open, one call or more with Sep
// between, and Close. A text that ends inside them is partial, holding the
// calls read whole before it.
func (s ToolSyntax) reply(m *matcher, pos int) (toolSpan, mstate) {
	f := s.f
	sp := toolSpan{start: pos}
	p := pos
	if f.Open != "" {
		r := m.seq(parseTemplate(f.Open), p, caps{}, "", func(e int, c caps) res { return res{st: mOK, pos: e} })
		switch {
		case r.st == mOK:
			p = r.pos
		case r.st == mPartial:
			return sp, mPartial
		case !f.Bare:
			return sp, mFail
		}
	}
	for {
		r := m.call(p)
		if r.st == mPartial {
			// A call the text ends in its closing marks has said what it
			// calls: it counts, its span still open.
			if r.c.whole {
				sp.calls = append(sp.calls, m.callsOf(r.c)...)
			}
			return sp, mPartial
		}
		if r.st != mOK {
			if len(sp.calls) == 0 {
				return sp, mFail
			}
			break
		}
		sp.calls = append(sp.calls, m.callsOf(r.c)...)
		p = r.pos
		q := p
		// Where the form writes no Close, the reply's calls may end here:
		// text that has not reached a next one does not hold this one.
		open := f.Close != ""
		if f.Sep != "" {
			var st mstate
			if q, st = m.lit(q, f.Sep); st == mPartial && open {
				return sp, mPartial
			} else if st != mOK {
				break
			}
		}
		next := m.call(q)
		if next.st == mFail || next.st == mPartial && !open {
			break
		}
		if next.st == mPartial {
			return sp, mPartial
		}
		p = q
	}
	if f.Close != "" {
		q, st := m.lit(p, f.Close)
		switch st {
		case mOK:
			p = q
		case mPartial:
			sp.end = len(m.text)
			return sp, mPartial
		}
	}
	sp.end, sp.closed = p, true
	return sp, mOK
}

// find reads every call in text: at each place the calls may begin, the
// form's reply matched, a reply the text ends inside running to the end.
func (s ToolSyntax) find(text string, ts ToolSet) []toolSpan {
	if s.f == nil {
		return nil
	}
	m := &matcher{f: s.f, ts: ts, text: text}
	var out []toolSpan
	at := 0
	for at <= len(text) {
		var found bool
		for _, st := range s.starts(text, at) {
			sp, ms := s.reply(m, st)
			if ms == mFail {
				continue
			}
			sp.ok = len(sp.calls) > 0
			for _, c := range sp.calls {
				sp.ok = sp.ok && ts.known(c.Name)
			}
			if ms == mPartial {
				sp.end, sp.closed = len(text), false
			}
			out = append(out, sp)
			at, found = sp.end, true
			break
		}
		if !found || at >= len(text) {
			break
		}
	}
	return out
}

// Hold is where a stream must stop sending text because a call has begun, or
// may be beginning, at that byte of text: -1 when nothing needs holding. A
// call whose markup has closed is resolved, and holds nothing after it.
func (s ToolSyntax) Hold(text string) int {
	return s.hold(text, s.find(text, ToolSet{any: true}))
}

// hold is Hold over the spans find read from text.
func (s ToolSyntax) hold(text string, spans []toolSpan) int {
	from := 0
	for _, sp := range spans {
		if !sp.closed {
			return sp.start
		}
		from = sp.end
	}
	if h := s.holdTail(text[from:], from == 0); h >= 0 {
		return from + h
	}
	return -1
}

// holdTail is where text, holding no call, may be beginning one: at a mark a
// call opens with, at such a mark's first bytes at the very end, or -- for a
// form whose reply may simply BE its calls -- at a reply with nothing in it
// yet.
func (s ToolSyntax) holdTail(text string, replyStart bool) int {
	if s.f == nil {
		return -1
	}
	h := -1
	at := func(i int) {
		if i >= 0 && (h < 0 || i < h) {
			h = i
		}
	}
	for _, o := range s.opens() {
		at(strings.Index(text, o))
		for k := min(len(o)-1, len(text)); k > 0; k-- {
			if strings.HasSuffix(text, o[:k]) {
				at(len(text) - k)
				break
			}
		}
	}
	if replyStart && (s.f.Bare || s.f.Leads) && strings.TrimSpace(text) == "" {
		at(len(text) - len(strings.TrimLeftFunc(text, unicode.IsSpace)))
	}
	return h
}

// objectBuilder writes a JSON object from arguments written one by one as
// text, in their order.
type objectBuilder struct {
	b strings.Builder
	n int
}

// add writes key with v: as a string where typ is "string" (or the value is
// not JSON), as JSON otherwise. guess reads a value of no declared type as
// JSON when it is a number, boolean, null, object or array, rather than as
// text.
func (o *objectBuilder) add(key, v, typ string, guess bool) {
	if o.n == 0 {
		o.b.WriteByte('{')
	} else {
		o.b.WriteByte(',')
	}
	o.n++
	k, _ := json.Marshal(key)
	o.b.Write(k)
	o.b.WriteByte(':')
	if typ != "string" && (typ != "" || guess) {
		var buf bytes.Buffer
		if json.Valid([]byte(v)) && json.Compact(&buf, []byte(strings.TrimSpace(v))) == nil {
			o.b.Write(buf.Bytes())
			return
		}
	}
	s, _ := json.Marshal(v)
	o.b.Write(s)
}

func (o *objectBuilder) String() string {
	if o.n == 0 {
		return "{}"
	}
	return o.b.String() + "}"
}

// compactObject is b, a JSON object (or a string holding one), compact.
func compactObject(b []byte) (string, bool) {
	b = bytes.TrimSpace(b)
	var s string
	if json.Unmarshal(b, &s) == nil {
		b = bytes.TrimSpace([]byte(s))
	}
	if len(b) == 0 {
		return "{}", true
	}
	var buf bytes.Buffer
	if json.Compact(&buf, b) != nil || buf.Bytes()[0] != '{' {
		return "", false
	}
	return buf.String(), true
}
