package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/grammar"
)

// The generic tool form is what a model whose chat template reads no tools is
// served: it was not trained to call a tool, so the request tells it, in the
// system message, what tools it may call and how to answer, and its reply is
// held to the answer's shape. It is llama.cpp's (common_chat_params_init_generic
// and minja's tool polyfills, which the request's prompt is built through):
//
//   - the system message gets the sentence genericSystem, then the tools as
//     the request wrote them and an example call, which minja reads off the
//     template itself (genericExample);
//   - a call an assistant turn made is written into its content as
//     {"tool_calls": [...]}, and a tool's result is a user turn holding
//     {"tool_response": {...}}, since the template renders neither;
//   - the reply is held to one JSON object, {"tool_call": {"name",
//     "arguments"}} or {"response": text}, and read back as a call or as
//     content. The template cannot render several calls in one turn, so the
//     form states one.
//
// Nothing in it is the model's own syntax, so the reply is read from its text
// by one scanner (readGeneric) and forced by one schema (genericGrammar).
// docs/divergences.md says where this part from llama.cpp.

const genericSystem = "Respond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request"

// readsTools is whether a chat template renders tool definitions or calls: one
// that does was trained to call tools in the form it writes, one that does not
// is served the generic form.
func readsTools(src string) bool {
	return strings.Contains(src, "tools") || strings.Contains(src, "tool_calls")
}

// genericSyntax is the syntax of a template that reads no tools.
func genericSyntax() ToolSyntax {
	return ToolSyntax{f: &jlm.ToolForm{Name: "generic"}, envelope: true}
}

// Generic is whether this syntax is the generic form: the reply is one JSON
// object, forced from the first token whenever the request has tools.
func (s ToolSyntax) Generic() bool { return s.envelope }

// ConstrainsReply is whether a request with tools under choice has its reply
// held to a grammar: a forced call always is, and the generic form's reply is
// whenever the request lets it call.
func (s ToolSyntax) ConstrainsReply(choice ToolChoice) bool {
	return choice.Forces() || s.envelope && choice.Mode != ToolChoiceNone
}

// carriesToolTurns is whether a conversation holds a call or a result.
func carriesToolTurns(msgs []ChatMessage) bool {
	return slices.ContainsFunc(msgs, func(x ChatMessage) bool { return x.Role == "tool" || len(x.ToolCalls) > 0 })
}

// polyCall and polyCalls are a turn's calls as the conversation writes them
// into its content; polyResult is a tool's result.
type polyCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	ID        string          `json:"id,omitempty"`
}

type polyCalls struct {
	Calls   []polyCall `json:"tool_calls"`
	Content string     `json:"content,omitempty"`
}

type polyResult struct {
	Result struct {
		Content string `json:"content"`
		Name    string `json:"name,omitempty"`
		ID      string `json:"tool_call_id,omitempty"`
	} `json:"tool_response"`
}

// dump2 is v as JSON written with a two-space indent.
func dump2(v any) string {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	e.SetIndent("", "  ")
	e.Encode(v)
	return strings.TrimSuffix(b.String(), "\n")
}

// genericMessages is the conversation as the template is handed it: each
// call and result written into the turn's content, and, where the request
// has tools, the system message holding the sentence, the tools and the
// example (the sentence goes first, as llama.cpp adds it before minja adds
// the tools).
func genericMessages(msgs []ChatMessage, tools []byte, example string) []ChatMessage {
	out := slices.Clone(msgs)
	for i, x := range out {
		switch {
		case x.Role == "assistant" && len(x.ToolCalls) > 0:
			p := polyCalls{Content: x.Content}
			for _, c := range x.ToolCalls {
				pc := polyCall{Name: c.Name, ID: c.ID, Arguments: json.RawMessage(c.Arguments)}
				if !json.Valid(pc.Arguments) {
					pc.Arguments, _ = json.Marshal(c.Arguments)
				}
				p.Calls = append(p.Calls, pc)
			}
			out[i].Content, out[i].ToolCalls = dump2(p), nil
		case x.Role == "tool":
			var r polyResult
			r.Result.Content, r.Result.Name, r.Result.ID = x.Content, x.Name, x.ToolCallID
			out[i] = ChatMessage{Role: "user", Content: dump2(r)}
		}
	}
	if len(tools) == 0 {
		return out
	}
	list := string(tools)
	var b bytes.Buffer
	if json.Indent(&b, tools, "", "  ") == nil {
		list = b.String()
	}
	system := genericSystem + "\n\nYou can call any of the following tools to satisfy the user's requests: " + list
	if example != "" {
		system += "\n\nExample tool call syntax:\n\n" + example + "\n\n"
	}
	if len(out) > 0 && out[0].Role == "system" {
		out[0].Content += "\n\n" + system
		return out
	}
	return append([]ChatMessage{{Role: "system", Content: system}}, out...)
}

// genericExample is the call the template shows the model: what it writes for
// a turn that made one, after the text it writes for a plain user turn. render
// is the template over a conversation (add_generation_prompt as its second
// argument) and eos its end-of-sequence token. llama.cpp's own derivation
// (minja's chat_template), the markup of the template's turn included.
func genericExample(render func([]ChatMessage, bool) (string, error), eos string) string {
	user := ChatMessage{Role: "user", Content: "Hey"}
	call := polyCalls{Calls: []polyCall{{Name: "tool_name", ID: "call_1___", Arguments: json.RawMessage(`{"arg1":"some_value"}`)}}}
	prefix, err := render([]ChatMessage{user}, true)
	if err != nil {
		return ""
	}
	full, err := render([]ChatMessage{user, {Role: "assistant", Content: dump2(call)}}, false)
	if err != nil {
		return ""
	}
	if i := strings.LastIndex(full, eos); eos != "" && i >= 0 &&
		(i == len(prefix)-len(eos) || strings.HasSuffix(full, "\n") && i == len(full)-len(eos)-1) {
		full = full[:i]
	}
	n := 0
	for i := 0; i < len(prefix) && i < len(full) && prefix[i] == full[i]; i++ {
		// A '<' is never the end of what the two share: a template that
		// writes a tag only on one side (DeepSeek's <think>) diverges there.
		if prefix[i] != '<' {
			n = i + 1
		}
	}
	ex := full[n:]
	if !strings.Contains(ex, "tool_name") && !strings.Contains(ex, "some_value") {
		return ""
	}
	return ex
}

// genericGrammar is the grammar a generic reply is held to: llama.cpp's
// schema for it, through the same schema converter as response_format. A tool
// whose parameter schema the converter does not build takes any JSON object
// for its arguments, as a forced call in any other form does.
func genericGrammar(tools ToolSet, choice ToolChoice, names []string) (string, error) {
	build := func(plain bool) (string, error) {
		calls := make([]string, len(names))
		for i, n := range names {
			params := string(bytes.TrimSpace(tools.schemas[n]))
			if plain || params == "" || params == "null" {
				params = `{"type":"object"}`
			}
			calls[i] = `{"type":"object","properties":{"name":{"type":"string","const":` + jsonName(n) +
				`},"arguments":` + params + `},"required":["name","arguments"]}`
		}
		call := calls[0]
		if len(calls) > 1 {
			call = `{"anyOf":[` + strings.Join(calls, ",") + `]}`
		}
		schema := `{"type":"object","properties":{"tool_call":` + call + `},"required":["tool_call"]}`
		if !choice.Forces() {
			schema = `{"anyOf":[` + schema + `,{"type":"object","properties":{"response":{"type":"string"}},` +
				`"required":["response"]}]}`
		}
		return grammar.FromJSONSchema([]byte(schema))
	}
	g, err := build(false)
	var unsupported grammar.ErrUnsupported
	if errors.As(err, &unsupported) {
		return build(true)
	}
	return g, err
}

// genericReply is what a generic reply holds so far, read from its text.
type genericReply struct {
	// start is where the object opens; -1 while the text is only space, and
	// -2 when it opens with anything else: such a reply is plain text.
	start int
	// response is the decoded text of a string "response" so far, done once
	// its closing quote has come; raw is a response that is any other value.
	response string
	done     bool
	raw      string
	has      bool
	// calls are the calls whose objects have closed, in the order written.
	calls []ToolCall
}

// readGeneric reads a reply's text from its start. final says the text is all
// there will be: a literal at its end is whole.
func readGeneric(s string, tools ToolSet, final bool) genericReply {
	g := genericScan{s: s, final: final}
	r := genericReply{start: -1}
	i := g.space(0)
	if i >= len(s) {
		return r
	}
	if s[i] != '{' {
		r.start = -2
		return r
	}
	r.start = i
	i++
	for {
		i = g.space(i)
		if i >= len(s) || s[i] == '}' {
			return r
		}
		if s[i] == ',' {
			i++
			continue
		}
		if s[i] != '"' {
			return r
		}
		key, e, ok := g.str(i)
		if !ok {
			return r
		}
		i = g.space(e)
		if i >= len(s) || s[i] != ':' {
			return r
		}
		i = g.space(i + 1)
		if i >= len(s) {
			return r
		}
		switch {
		case key == "response" && s[i] == '"':
			r.response, e, r.done = g.str(i)
			r.has = true
			if !r.done {
				return r
			}
			i = e
		case key == "response":
			e, ok := g.skip(i)
			if !ok {
				return r
			}
			r.has, r.raw, i = true, s[i:e], e
		case key == "tool_call":
			e, ok := g.skip(i)
			if !ok {
				return r
			}
			r.call(s[i:e], tools)
			i = e
		case key == "tool_calls" && s[i] == '[':
			i++
			for {
				i = g.space(i)
				if i >= len(s) {
					return r
				}
				if s[i] == ']' {
					i++
					break
				}
				if s[i] == ',' {
					i++
					continue
				}
				e, ok := g.skip(i)
				if !ok {
					return r
				}
				r.call(s[i:e], tools)
				i = e
			}
		default:
			e, ok := g.skip(i)
			if !ok {
				return r
			}
			i = e
		}
	}
}

// call adds the call one object writes, when it names a declared tool.
func (r *genericReply) call(obj string, tools ToolSet) {
	var c struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		ID        string          `json:"id"`
	}
	if json.Unmarshal([]byte(obj), &c) != nil || !tools.known(c.Name) {
		return
	}
	args := bytes.TrimSpace(c.Arguments)
	var inner string
	if len(args) > 0 && args[0] == '"' && json.Unmarshal(args, &inner) == nil && json.Valid([]byte(inner)) {
		args = []byte(inner)
	}
	var b bytes.Buffer
	if len(args) == 0 || string(args) == "null" || json.Compact(&b, args) != nil {
		b.Reset()
		b.WriteString("{}")
	}
	r.calls = append(r.calls, ToolCall{ID: c.ID, Name: c.Name, Arguments: b.String()})
}

// text is the reply's content: the response's text, a response that is not
// one written out indented, or, where the object holds no response and no
// call, the reply as it was written.
func (r genericReply) text(s string) string {
	switch {
	case r.start == -1:
		return ""
	case r.start == -2 || !r.has && len(r.calls) == 0:
		return strings.TrimSpace(s)
	case r.raw != "":
		var b bytes.Buffer
		if json.Indent(&b, []byte(r.raw), "", "  ") == nil {
			return b.String()
		}
		return r.raw
	}
	return r.response
}

// genericScan reads a generic reply's text.
type genericScan struct {
	s     string
	final bool
}

func (g genericScan) space(i int) int {
	for i < len(g.s) && (g.s[i] == ' ' || g.s[i] == '\t' || g.s[i] == '\n' || g.s[i] == '\r') {
		i++
	}
	return i
}

// str decodes the string whose opening quote is at i: its text so far, where
// it ends (after the closing quote), and whether the quote has come. An
// escape the text ends inside is left unread.
func (g genericScan) str(i int) (string, int, bool) {
	var b strings.Builder
	i++
	for i < len(g.s) {
		c := g.s[i]
		switch {
		case c == '"':
			return b.String(), i + 1, true
		case c != '\\':
			b.WriteByte(c)
			i++
			continue
		case i+1 >= len(g.s):
			return b.String(), i, false
		}
		e := g.s[i+1]
		if e != 'u' {
			switch e {
			case 'n':
				e = '\n'
			case 't':
				e = '\t'
			case 'r':
				e = '\r'
			case 'b':
				e = '\b'
			case 'f':
				e = '\f'
			}
			b.WriteByte(e)
			i += 2
			continue
		}
		if i+6 > len(g.s) {
			return b.String(), i, false
		}
		r, ok := hex4(g.s[i+2 : i+6])
		switch {
		case !ok:
			b.WriteRune(utf8.RuneError)
			i += 6
		case !utf16.IsSurrogate(r) || r >= 0xDC00:
			b.WriteRune(r)
			i += 6
		case i+12 > len(g.s) && !g.final:
			return b.String(), i, false
		default:
			lo, ok := rune(0), false
			if i+12 <= len(g.s) && g.s[i+6] == '\\' && g.s[i+7] == 'u' {
				lo, ok = hex4(g.s[i+8 : i+12])
			}
			if d := utf16.DecodeRune(r, lo); ok && d != utf8.RuneError {
				b.WriteRune(d)
				i += 12
			} else {
				b.WriteRune(utf8.RuneError)
				i += 6
			}
		}
	}
	return b.String(), i, false
}

func hex4(s string) (rune, bool) {
	var r rune
	for i := 0; i < 4; i++ {
		c := s[i]
		switch {
		case c >= '0' && c <= '9':
			r = r<<4 | rune(c-'0')
		case c >= 'a' && c <= 'f':
			r = r<<4 | rune(c-'a'+10)
		case c >= 'A' && c <= 'F':
			r = r<<4 | rune(c-'A'+10)
		default:
			return 0, false
		}
	}
	return r, true
}

// skip is where the value at i ends and whether it has: a string to its
// closing quote, an object or array to its closing bracket, any other to the
// next delimiter.
func (g genericScan) skip(i int) (int, bool) {
	if i >= len(g.s) {
		return i, false
	}
	if c := g.s[i]; c != '{' && c != '[' && c != '"' {
		j := i
		for j < len(g.s) && !strings.ContainsRune(",}] \t\r\n", rune(g.s[j])) {
			j++
		}
		return j, j > i && (j < len(g.s) || g.final)
	}
	depth := 0
	for j := i; j < len(g.s); j++ {
		switch g.s[j] {
		case '"':
			for j++; j < len(g.s) && g.s[j] != '"'; j++ {
				if g.s[j] == '\\' {
					j++
				}
			}
			if j >= len(g.s) {
				return len(g.s), false
			}
			if depth == 0 {
				return j + 1, true
			}
		case '{', '[':
			depth++
		case '}', ']':
			if depth--; depth == 0 {
				return j + 1, true
			}
		}
	}
	return len(g.s), false
}

// wholeRunes is s without a trailing character cut short.
func wholeRunes(s string) string {
	for k := 1; k < utf8.UTFMax && k <= len(s); k++ {
		if utf8.RuneStart(s[len(s)-k]) {
			if !utf8.FullRuneInString(s[len(s)-k:]) {
				return s[:len(s)-k]
			}
			break
		}
	}
	return s
}

// genericRead is a ToolStream's state reading a generic reply: how much of
// the response has been shown, how many calls returned, and the text of the
// reply as a person reads it, for a reply that is not an object.
type genericRead struct {
	shown, sent int
	held, chat  strings.Builder
}

// add is ToolStream.add for the generic form: text is the reply so far.
func (g *genericRead) add(text string, tools ToolSet, chat string) (string, []ToolCall) {
	g.chat.WriteString(chat)
	r := readGeneric(text, tools, false)
	switch r.start {
	case -1:
		g.held.WriteString(chat)
		return "", nil
	case -2:
		out := g.held.String() + chat
		g.held.Reset()
		return out, nil
	}
	var out string
	if r.raw == "" {
		out = wholeRunes(r.response)
		out = out[min(g.shown, len(out)):]
		g.shown += len(out)
	}
	calls := r.calls[min(g.sent, len(r.calls)):]
	g.sent += len(calls)
	return out, calls
}

// finish is ToolStream.Finish for the generic form.
func (g *genericRead) finish(text string, tools ToolSet) (string, []ToolCall, string) {
	r := readGeneric(text, tools, true)
	content := r.text(text)
	switch r.start {
	case -1:
		return "", nil, ""
	case -2:
		return "", nil, strings.TrimSpace(g.chat.String())
	}
	tail := ""
	switch {
	case r.raw == "" && r.has:
		tail = content[min(g.shown, len(content)):]
	case r.raw != "" || !r.has && len(r.calls) == 0:
		tail = content
	}
	calls := r.calls[min(g.sent, len(r.calls)):]
	return tail, calls, content
}
