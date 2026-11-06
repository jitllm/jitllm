package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/grammar"
)

// ToolChoiceMode is what a request lets the model do with its tools.
type ToolChoiceMode string

const (
	// ToolChoiceAuto lets the model call or answer ("" is the same).
	ToolChoiceAuto ToolChoiceMode = "auto"
	// ToolChoiceNone forbids calls.
	ToolChoiceNone ToolChoiceMode = "none"
	// ToolChoiceRequired requires at least one call to a declared tool.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceFunction requires a call to the tool Name.
	ToolChoiceFunction ToolChoiceMode = "function"
)

// ToolChoice is OpenAI's tool_choice (and Anthropic's): Mode, the named
// tool, and Single for a reply limited to one call (parallel_tool_calls
// false, disable_parallel_tool_use).
type ToolChoice struct {
	Mode   ToolChoiceMode
	Name   string
	Single bool
}

// Forces says whether the reply must be a call.
func (c ToolChoice) Forces() bool {
	return c.Mode == ToolChoiceRequired || c.Mode == ToolChoiceFunction
}

// templateValue is tool_choice as a template reads it: the mode's string,
// or OpenAI's object for a named function. nil for none given.
func (c ToolChoice) templateValue() any {
	switch c.Mode {
	case "":
		return nil
	case ToolChoiceFunction:
		return map[string]any{"type": "function", "function": map[string]any{"name": c.Name}}
	}
	return string(c.Mode)
}

// ToolGrammar is the grammar a reply forced to call (choice.Forces) is held
// to: this model's tool-call syntax around arguments that match the called
// tool's parameter schema -- the named tool's, or any declared tool's for
// "required" -- one call, or one or more where the syntax writes several and
// choice is not Single. prompt is the rendered prompt's end, as raw text
// (control tokens spelled out): a model whose prompt left it reasoning, or
// whose template reasons, finishes or writes its reasoning block before the
// call (Reasoning).
//
// A parameter schema the converter does not build (grammar.ErrUnsupported)
// constrains that tool's arguments to any JSON object rather than refusing
// the request: the call itself is still forced.
func (m *Model) ToolGrammar(tools ToolSet, choice ToolChoice, prompt string) (string, error) {
	src, _ := m.ChatTemplateFor(true)
	syn, err := m.ToolSyntax()
	if err != nil {
		return "", err
	}
	return ToolGrammarOf(syn, syn.Reasoning(src, prompt), tools, choice)
}

// Reasoning is where a forced reply stands with respect to a reasoning block.
type Reasoning uint8

const (
	// ReasonNone: the reply is the call.
	ReasonNone Reasoning = iota
	// ReasonInside: the prompt opened the block (DeepSeek-R1's ends in
	// "<think>\n"); the reply finishes it, then calls.
	ReasonInside
	// ReasonMay: the reply may write a whole block, then calls.
	ReasonMay
)

// Reasoning reads, from a model's template src and the end of its rendered
// prompt, whether a reply in this syntax reasons first: inside a block the
// prompt opened, never when the prompt wrote the block already closed (a
// template's thinking switched off), and otherwise when the template
// closes such blocks at all.
func (s ToolSyntax) Reasoning(src, prompt string) Reasoning {
	open, close := s.thinkMarks()
	if s.f != nil && s.f.ThinkAlways {
		return ReasonMay
	}
	if o := strings.LastIndex(prompt, open); o >= 0 && o > strings.LastIndex(prompt, close) {
		return ReasonInside
	}
	if strings.HasSuffix(strings.TrimRight(prompt, " \t\n"), close) {
		return ReasonNone
	}
	mark := close
	if s.f != nil && s.f.ThinkSeen != "" {
		mark = s.f.ThinkSeen
	}
	if strings.Contains(src, mark) {
		return ReasonMay
	}
	return ReasonNone
}

// thinkMarks are the marks a reasoning block opens and closes with: the
// form's, else <think> and </think>.
func (s ToolSyntax) thinkMarks() (open, close string) {
	if s.f != nil && s.f.ThinkOpen != "" {
		return s.f.ThinkOpen, s.f.ThinkClose
	}
	return "<think>", "</think>"
}

// ToolGrammarOf is ToolGrammar for a syntax.
func ToolGrammarOf(syn ToolSyntax, reason Reasoning, tools ToolSet, choice ToolChoice) (string, error) {
	if syn.f == nil {
		return "", fmt.Errorf("model: this model states no tool-call form to force a call in")
	}
	names := tools.names
	if choice.Mode == ToolChoiceFunction {
		if !slices.Contains(names, choice.Name) {
			return "", fmt.Errorf("tool_choice names %q, which is not a declared tool", choice.Name)
		}
		names = []string{choice.Name}
	}
	if len(names) == 0 {
		return "", fmt.Errorf("tool_choice %q with no tools declared", choice.Mode)
	}
	g := &toolGrammar{b: grammar.NewBuilder(), tools: tools, w: syn.f.Write, f: syn.f}
	many := !choice.Single && choice.Mode != ToolChoiceFunction && syn.f.Several
	calls, err := g.calls(names, many)
	if err != nil {
		return "", err
	}
	// A reply may open with a newline or two, as several templates write
	// one before the call.
	ws := g.b.Rule("lead", `[ \t\n]{0,4}`)
	open, close := syn.thinkMarks()
	body := g.b.Until("thought", close) + " " + lit(close)
	if syn.f.ThinkThen != "" {
		body += " " + lit(syn.f.ThinkThen)
	}
	root := ws + " " + calls
	switch reason {
	case ReasonInside:
		root = body + " " + ws + " " + calls
	case ReasonMay:
		root = ws + " ( " + lit(open) + " " + body + " " + ws + " )? " + calls
	}
	return g.b.Text(root), nil
}

// ToolMarkers are the texts this syntax's grammar writes that a vocabulary
// may hold as control tokens: the mask must offer those tokens by their
// literal text, which Piece does not give.
func (s ToolSyntax) ToolMarkers() []string {
	if s.f == nil {
		return nil
	}
	return s.f.Markers
}

type toolGrammar struct {
	b     *grammar.Builder
	tools ToolSet
	w     jlm.ToolWrite
	f     *jlm.ToolForm
}

// args is the rule for name's arguments as one JSON object.
func (g *toolGrammar) args(name string) (string, error) {
	r, err := g.b.Schema(g.tools.schemas[name], "args-"+name)
	var unsupported grammar.ErrUnsupported
	if errors.As(err, &unsupported) {
		return g.b.Object(), nil
	}
	return r, err
}

// props is name's arguments one by one, for a form that writes them so; a
// schema it cannot build is refused, a form writing arguments one by one
// having no way to say "any object".
func (g *toolGrammar) props(name string) ([]grammar.Prop, error) {
	ps, err := g.b.Properties(g.tools.schemas[name], "arg-"+name)
	var unsupported grammar.ErrUnsupported
	if errors.As(err, &unsupported) {
		return nil, fmt.Errorf("tool %q: a forced call writes its arguments one by one in this model's "+
			"syntax, and its schema is not built: %v", name, err)
	}
	return ps, err
}

// list joins items as the reply writes them: in their order, each present
// unless optional, sep between those present. Two chains of rules from the
// end: first_i is the items from i on with none written yet, more_i with one
// already written (so each present item takes sep before it).
func (g *toolGrammar) list(name string, items []string, req []bool, sep string) string {
	first, more := `""`, `""`
	for i := len(items) - 1; i >= 0; i-- {
		f := items[i] + " " + more
		m := sep + " " + items[i] + " " + more
		if !req[i] {
			f += " | " + first
			m = "( " + m + " ) | " + more
		}
		first = g.b.Rule(fmt.Sprintf("%s-first-%d", name, i), f)
		more = g.b.Rule(fmt.Sprintf("%s-more-%d", name, i), m)
	}
	return first
}

// seq is each argument in turn with no separator: required ones always,
// optional ones present or not.
func seq(items []string, req []bool) string {
	var b []string
	for i, it := range items {
		if req[i] {
			b = append(b, it)
		} else {
			b = append(b, "( "+it+" )?")
		}
	}
	if len(b) == 0 {
		return `""`
	}
	return strings.Join(b, " ")
}

func lit(s string) string { return grammar.Literal(s) }

func jsonName(n string) string {
	b, _ := json.Marshal(n)
	return string(b)
}

// attrEscape is a name as a template writes it in a tag's attribute.
func attrEscape(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "&", "&amp;"), `"`, "&quot;")
}

// fill is t, one of the form's templates, as a grammar: its text, and its
// holes -- $NAME, $QNAME, $XNAME and $KEY as literals, the others as the
// rules holes holds -- (?...?) the text inside written or not.
func (g *toolGrammar) fill(t string, holes map[string]string) string {
	var out []string
	for _, n := range parseTemplate(t) {
		switch n.kind {
		case tLit:
			out = append(out, lit(n.text))
		case tOpt:
			var sub strings.Builder
			for _, s := range n.sub {
				switch s.kind {
				case tLit:
					sub.WriteString(s.text)
				case tHole:
					sub.WriteString("$" + s.text)
				}
			}
			out = append(out, "( "+g.fill(sub.String(), holes)+" )?")
		case tHole:
			switch n.text {
			case "NAME":
				out = append(out, lit(holes["name"]))
			case "QNAME":
				out = append(out, lit(jsonName(holes["name"])))
			case "XNAME":
				out = append(out, lit(attrEscape(holes["name"])))
			case "KEY":
				out = append(out, lit(holes["key"]))
			default:
				out = append(out, holes[strings.ToLower(n.text)])
			}
		}
	}
	return strings.Join(out, " ")
}

// id is the rule a call's $ID is written by.
func (g *toolGrammar) id() string {
	f := g.f
	switch {
	case f.IDAlnum:
		return fmt.Sprintf("[a-zA-Z0-9]{%d}", f.IDDigits)
	case f.IDMin > 0:
		return fmt.Sprintf("[%d-9] [0-9]{0,%d}", f.IDMin, f.IDDigits-1)
	}
	return fmt.Sprintf("[0-9]{1,%d}", f.IDDigits)
}

// one is a single call to name in the form's text.
func (g *toolGrammar) one(name string) (string, error) {
	holes := map[string]string{"name": name}
	if strings.Contains(g.w.Call, "$ARGS") {
		a, err := g.args(name)
		if err != nil {
			return "", err
		}
		holes["args"] = a
	}
	if strings.Contains(g.w.Call, "$PARAMS") {
		p, err := g.params(name)
		if err != nil {
			return "", err
		}
		holes["params"] = p
	}
	if strings.Contains(g.w.Call, "$ID") {
		holes["id"] = g.id()
	}
	return g.fill(g.w.Call, holes), nil
}

// jsonValues is whether the form writes a value as JSON does, strings
// quoted with ".
func (g *toolGrammar) jsonValues() bool {
	n := g.f.Notation
	return len(n.Quotes) > 0 && n.Quotes[0] == `"`
}

// valueEnd is the text a parameter template writes after its $VALUE, up to
// its next hole: where a text value ends.
func valueEnd(t string) string {
	_, after, _ := strings.Cut(t, "$VALUE")
	if i := strings.IndexByte(after, '$'); i >= 0 {
		after = after[:i]
	}
	return after
}

// params is name's arguments written one by one: a string in Params[0], as
// text up to that template's next mark where the form's values are text,
// any other in the last of Params, as the form's notation writes it;
// ParamSep between them, or none.
func (g *toolGrammar) params(name string) (string, error) {
	if len(g.w.Params) == 0 {
		return "", fmt.Errorf("model: the %s tool form writes $PARAMS and states no parameter", g.f.Name)
	}
	ps, err := g.props(name)
	if err != nil {
		return "", err
	}
	if g.w.Sorted {
		slices.SortFunc(ps, func(a, b grammar.Prop) int { return strings.Compare(a.Key, b.Key) })
	}
	str, other := g.w.Params[0], g.w.Params[len(g.w.Params)-1]
	var items []string
	var req []bool
	for _, p := range ps {
		t, v := other, p.Rule
		switch {
		case !g.jsonValues():
			v = g.quotedValue(p)
		case p.Type == "string" && g.f.TextValues:
			t, v = str, g.b.Until("text-"+name, valueEnd(str))
		}
		items = append(items, g.fill(t, map[string]string{"key": p.Key, "value": v}))
		req = append(req, p.Required)
	}
	if g.w.ParamSep != "" {
		return g.list("p-"+name, items, req, lit(g.w.ParamSep)), nil
	}
	return seq(items, req), nil
}

// quotedValue is an argument's value in the form's notation: a string between
// its quote marks, a number or boolean as JSON writes it, anything else as
// that notation's objects and lists, keys bare where it writes them so.
func (g *toolGrammar) quotedValue(p grammar.Prop) string {
	quote := g.f.Notation.Quotes[0]
	q := lit(quote)
	switch p.Type {
	case "string":
		return q + " " + g.b.Until("q-text", quote) + " " + q
	case "integer":
		return g.b.Rule("q-int", `"-"? ([0] | [1-9] [0-9]{0,15})`)
	case "number":
		return g.b.Rule("q-num", `"-"? ([0] | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})?`)
	case "boolean":
		return g.b.Rule("q-bool", `"true" | "false"`)
	}
	str := q + " " + g.b.Until("q-text", quote) + " " + q
	key := str
	if g.f.Notation.BareKeys {
		key = g.b.Rule("q-key", "[a-zA-Z_] [a-zA-Z0-9_]*")
	}
	val := g.b.Rule("q-value", str+` | "-"? [0-9]+ ("." [0-9]+)? | "true" | "false" | "null" | q-obj | q-arr`)
	g.b.Rule("q-obj", fmt.Sprintf(`"{" ( %s ":" %s ( "," %s ":" %s )* )? "}"`, key, val, key, val))
	g.b.Rule("q-arr", fmt.Sprintf(`"[" ( %s ( "," %s )* )? "]"`, val, val))
	return val
}

// calls is one call to any of names, or one or more when many, inside the
// form's Open and Close, each in its Section.
func (g *toolGrammar) calls(names []string, many bool) (string, error) {
	alts := make([]string, len(names))
	for i, n := range names {
		c, err := g.one(n)
		if err != nil {
			return "", err
		}
		alts[i] = c
	}
	call := g.b.Rule("call", strings.Join(alts, " | "))
	if g.w.Section[0] != "" || g.w.Section[1] != "" {
		call = g.b.Rule("section", g.fill(g.w.Section[0], nil)+" "+call+" "+g.fill(g.w.Section[1], nil))
	}
	open, sep, close := g.fill(g.w.Open, nil), g.fill(g.w.Sep, nil), g.fill(g.w.Close, nil)
	body := call
	if many {
		body = call + " ( " + sep + " " + call + " )*"
		if sep == "" {
			body = call + "+"
		}
	}
	return strings.TrimSpace(open + " " + body + " " + close), nil
}
