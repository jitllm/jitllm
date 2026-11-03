package model

import (
	"bytes"
	"encoding/json"
	"slices"
	"strconv"
	"strings"
)

// ToolSyntax is how a model family writes a tool call: the markup around the
// call, and how the name and arguments sit inside it. It is a property of the
// model, read from its chat template (ToolSyntaxOf) -- the template is what
// the weights were trained to emit -- rather than found by trying every
// parser on the reply.
//
// Every syntax is read from the reply's RAW text: the control tokens it is
// written in ([TOOL_CALLS], <｜tool▁sep｜>, <|call|>) spelled out as their
// literal text, since the text a person reads drops them (ToolStream builds
// that text from the ids).
type ToolSyntax uint8

const (
	// <tool_call>{"name": ..., "arguments": {...}}</tool_call>, any number,
	// anywhere; a reply that opens with a bare JSON call is read too. Qwen2.5,
	// Qwen3, Qwen3-Next, InternVL3, Jamba, Granite 4, EXAONE 4, Falcon-H1,
	// SmolLM3, Hermes, and the default for a template that names no syntax.
	ToolSyntaxHermes ToolSyntax = iota
	// A reply that IS a JSON call, {"name": ..., "parameters": {...}}, several
	// separated by ';': Llama 3.1, 3.2, 3.3.
	ToolSyntaxJSON
	// <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function>
	// </tool_call>: Qwen3-Coder, Qwen3.5, Nemotron 3.
	ToolSyntaxQwenXML
	// Qwen's parameters inside <seed:tool_call>: Seed-OSS.
	ToolSyntaxSeedXML
	// <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>:
	// GLM-4.5, 4.6, 4.7, Ling 3.
	ToolSyntaxGLM
	// <minimax:tool_call><invoke name="N"><parameter name="K">V</parameter>
	// </invoke></minimax:tool_call>: MiniMax-M2.
	ToolSyntaxMiniMax
	// MiniMax's invoke under the ｜DSML｜ prefix, each parameter saying
	// whether it is a string: DeepSeek V4.
	ToolSyntaxDSML
	// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
	// ```json{...}```<｜tool▁call▁end｜><｜tool▁calls▁end｜>: DeepSeek V3, R1
	// and their distills.
	ToolSyntaxDeepSeekV3
	// The same tokens with NAME<｜tool▁sep｜>{...}: DeepSeek V3.1, V3.2.
	ToolSyntaxDeepSeekV31
	// <|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX
	// <|tool_call_argument_begin|>{...}<|tool_call_end|>
	// <|tool_calls_section_end|>: Kimi-K2.
	ToolSyntaxKimi
	// <|open|>tools<|sep|><|open|>call tool="NAME" index="1"<|sep|>...
	// <|close|>call<|sep|><|close|>tools<|sep|>: Kimi-K3's XTML.
	ToolSyntaxKimiXTML
	// gpt-oss's harmony: a commentary message addressed to=functions.NAME
	// whose body is the arguments, ended by <|call|>.
	ToolSyntaxHarmony
	// [TOOL_CALLS][{"name": ..., "arguments": {...}}]: Mistral v0.3,
	// Mixtral, Nemo.
	ToolSyntaxMistral
	// [TOOL_CALLS]NAME[ARGS]{...}: Mistral Small 3.2, Ministral 3, Devstral,
	// Magistral.
	ToolSyntaxMistralArgs
	// <|tool_call|>[{"name": ..., "arguments": {...}}]: Granite 3.x.
	ToolSyntaxGranite
	// <TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>: Nemotron-H.
	ToolSyntaxNemotron
	// <|START_ACTION|>[{"tool_call_id": ..., "tool_name": ...,
	// "parameters": {...}}]<|END_ACTION|>: Command R7B, Command A.
	ToolSyntaxCommandR7B
	// Action: ```json [{"tool_name": ..., "parameters": {...}}] ```:
	// Command-R and R+.
	ToolSyntaxCommandR
	// <|tools_prefix|>[{"NAME": {...}}]<|tools_suffix|>: Apertus.
	ToolSyntaxApertus
	// functools[{"name": ..., "arguments": {...}}]: Phi-4-mini.
	ToolSyntaxPhi4Mini
	// [NAME(key=value, ...), ...] as Python: Llama 4.
	ToolSyntaxPythonic
	// The Python list inside <|tool_call_start|> <|tool_call_end|>: LFM2.
	ToolSyntaxLFM2
	// <tool_calls><tool_call>NAME ```json{...}```</tool_call></tool_calls>:
	// Hunyuan.
	ToolSyntaxHunyuan
	// <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>: Gemma 4.
	ToolSyntaxGemma4

	numToolSyntaxes
)

// String is the syntax's name.
func (s ToolSyntax) String() string {
	switch s {
	case ToolSyntaxHermes:
		return "hermes"
	case ToolSyntaxJSON:
		return "json"
	case ToolSyntaxQwenXML:
		return "qwen-xml"
	case ToolSyntaxSeedXML:
		return "seed-xml"
	case ToolSyntaxGLM:
		return "glm"
	case ToolSyntaxMiniMax:
		return "minimax"
	case ToolSyntaxDSML:
		return "dsml"
	case ToolSyntaxDeepSeekV3:
		return "deepseek-v3"
	case ToolSyntaxDeepSeekV31:
		return "deepseek-v3.1"
	case ToolSyntaxKimi:
		return "kimi"
	case ToolSyntaxKimiXTML:
		return "kimi-xtml"
	case ToolSyntaxHarmony:
		return "harmony"
	case ToolSyntaxMistral:
		return "mistral"
	case ToolSyntaxMistralArgs:
		return "mistral-args"
	case ToolSyntaxGranite:
		return "granite"
	case ToolSyntaxNemotron:
		return "nemotron"
	case ToolSyntaxCommandR7B:
		return "command-r7b"
	case ToolSyntaxCommandR:
		return "command-r"
	case ToolSyntaxApertus:
		return "apertus"
	case ToolSyntaxPhi4Mini:
		return "phi4-mini"
	case ToolSyntaxPythonic:
		return "pythonic"
	case ToolSyntaxLFM2:
		return "lfm2"
	case ToolSyntaxHunyuan:
		return "hunyuan"
	case ToolSyntaxGemma4:
		return "gemma4"
	}

	return "tool-syntax(" + strconv.Itoa(int(s)) + ")"
}

// ToolSyntaxOf reads the syntax a chat template trains its model to call
// tools in, from the markup the template itself writes or describes. The
// checks are ordered: a template naming several markers is the family whose
// marker is most specific (Qwen3.5's writes <tool_call> around <function=,
// GLM's around <arg_key>).
func ToolSyntaxOf(template string) ToolSyntax {
	has := func(s ...string) bool {
		for _, x := range s {
			if !strings.Contains(template, x) {
				return false
			}
		}
		return true
	}
	switch {
	case has("<|channel|>", "to="):
		return ToolSyntaxHarmony
	case has("｜DSML｜"):
		return ToolSyntaxDSML
	case has("<｜tool▁calls▁begin｜>", "```json"):
		return ToolSyntaxDeepSeekV3
	case has("<｜tool▁calls▁begin｜>"):
		return ToolSyntaxDeepSeekV31
	case has("<|tool_calls_section_begin|>"):
		return ToolSyntaxKimi
	case has("<|open|>", "'tools'"):
		return ToolSyntaxKimiXTML
	case has("[ARGS]"):
		return ToolSyntaxMistralArgs
	case has("[TOOL_CALLS]"):
		return ToolSyntaxMistral
	case has("<minimax:tool_call>"):
		return ToolSyntaxMiniMax
	case has("<seed:tool_call>"):
		return ToolSyntaxSeedXML
	case has("<function="):
		return ToolSyntaxQwenXML
	case has("<arg_key>"):
		return ToolSyntaxGLM
	case has("<|tool_call>"):
		return ToolSyntaxGemma4
	case has("<|START_ACTION|>"):
		return ToolSyntaxCommandR7B
	case has("Action:", "```json", "tool_name"):
		return ToolSyntaxCommandR
	case has("<TOOLCALL>"):
		return ToolSyntaxNemotron
	case has("<|tool_call|>"):
		return ToolSyntaxGranite
	case has("<|tools_prefix|>"):
		return ToolSyntaxApertus
	case has("<|tool_list_start|>") || has("<|tool_call_start|>") || has("List of tools:"):
		return ToolSyntaxLFM2
	case has("<|tool|>", "<|end|>") && !has("<tool_call>"):
		return ToolSyntaxPhi4Mini
	case has("<tool_calls>", "```json"):
		return ToolSyntaxHunyuan
	case has("<tool_call>"):
		return ToolSyntaxHermes
	case has("<|header_start|>", "tool_calls"):
		return ToolSyntaxPythonic
	case has("<|start_header_id|>", "ipython"):
		return ToolSyntaxJSON
	}
	return ToolSyntaxHermes
}

// ToolSyntax is the syntax this model calls tools in: its tool-use template's
// (ToolSyntaxOf), harmony whenever the vocabulary frames messages that way.
func (m *Model) ToolSyntax() ToolSyntax {
	if m != nil && m.Vocab != nil && m.Vocab.Harmony() {
		return ToolSyntaxHarmony
	}
	src, err := m.chatTemplateFor(true)
	if err != nil {
		return ToolSyntaxHermes
	}
	return ToolSyntaxOf(src)
}

// ToolSet is a request's declared tools: their names, which are the only
// calls a reply can make, and their parameter schemas, which type the
// arguments of a syntax that writes them as text.
type ToolSet struct {
	names   []string
	schemas map[string]json.RawMessage
	// any accepts every name: ParseToolCalls with nil names.
	any bool
}

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
// the raw text, whether its closing marker has arrived, and its calls. ok is
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

func (s ToolSyntax) find(text string, ts ToolSet) []toolSpan {
	switch s {
	case ToolSyntaxHermes:
		if strings.Contains(text, "<tool_call>") {
			return sections(text, "<tool_call>", "</tool_call>", func(body string) ([]ToolCall, bool) {
				c, ok := hermesCall(body)
				return []ToolCall{c}, ok && ts.known(c.Name)
			})
		}
		if strings.HasPrefix(strings.TrimLeft(text, " \t\r\n"), "[TOOL_CALLS]") {
			// Mistral v0.3's opener, spelled out: the raw text keeps the
			// control token that decoded text drops.
			return mistralCalls(text, ts)
		}
		return bareJSON(text, ts)
	case ToolSyntaxJSON:
		t := text
		off := 0
		if i := strings.Index(t, "<|python_tag|>"); i >= 0 && strings.TrimSpace(t[:i]) == "" {
			off = i + len("<|python_tag|>")
		}
		sps := bareJSON(t[off:], ts)
		for i := range sps {
			sps[i].start += off
			sps[i].end += off
			if off > 0 && i == 0 {
				sps[i].start = 0
			}
		}
		return sps
	case ToolSyntaxQwenXML:
		return sections(text, "<tool_call>", "</tool_call>", func(b string) ([]ToolCall, bool) {
			return xmlFunctions(b, ts, "<function=", ">", "</function>", "<parameter=", ">", "</parameter>", true)
		})
	case ToolSyntaxSeedXML:
		return sections(text, "<seed:tool_call>", "</seed:tool_call>", func(b string) ([]ToolCall, bool) {
			return xmlFunctions(b, ts, "<function=", ">", "</function>", "<parameter=", ">", "</parameter>", true)
		})
	case ToolSyntaxMiniMax:
		return sections(text, "<minimax:tool_call>", "</minimax:tool_call>", func(b string) ([]ToolCall, bool) {
			return xmlFunctions(b, ts, `<invoke name="`, `">`, "</invoke>", `<parameter name="`, `">`, "</parameter>", false)
		})
	case ToolSyntaxDSML:
		return sections(text, "<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>", func(b string) ([]ToolCall, bool) {
			return dsmlCalls(b, ts)
		})
	case ToolSyntaxGLM:
		return sections(text, "<tool_call>", "</tool_call>", func(b string) ([]ToolCall, bool) {
			c, ok := glmCall(b, ts)
			return []ToolCall{c}, ok
		})
	case ToolSyntaxDeepSeekV3, ToolSyntaxDeepSeekV31:
		return sections(text, "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>", func(b string) ([]ToolCall, bool) {
			return deepseekCalls(b, ts)
		})
	case ToolSyntaxKimi:
		return sections(text, "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", func(b string) ([]ToolCall, bool) {
			return kimiCalls(b, ts)
		})
	case ToolSyntaxKimiXTML:
		return sections(text, "<|open|>tools<|sep|>", "<|close|>tools<|sep|>", func(b string) ([]ToolCall, bool) {
			return xtmlCalls(b, ts)
		})
	case ToolSyntaxHarmony:
		return harmonyCalls(text, ts)
	case ToolSyntaxMistral, ToolSyntaxMistralArgs:
		if !strings.Contains(text, "[TOOL_CALLS]") {
			// v0.3's [TOOL_CALLS] is a control token: text decoded
			// without it opens with the array.
			return bareJSON(text, ts)
		}
		return mistralCalls(text, ts)
	case ToolSyntaxGranite:
		return jsonLists(text, ts, "<|tool_call|>", "", false, nameArgs)
	case ToolSyntaxNemotron:
		return jsonLists(text, ts, "<TOOLCALL>", "</TOOLCALL>", false, nameArgs)
	case ToolSyntaxCommandR7B:
		return jsonLists(text, ts, "<|START_ACTION|>", "<|END_ACTION|>", false, cohereArgs)
	case ToolSyntaxCommandR:
		return jsonLists(text, ts, "Action:", "", true, cohereArgs)
	case ToolSyntaxApertus:
		return jsonLists(text, ts, "<|tools_prefix|>", "<|tools_suffix|>", false, keyedArgs)
	case ToolSyntaxPhi4Mini:
		return jsonLists(text, ts, "functools", "", false, nameArgs)
	case ToolSyntaxPythonic:
		t := strings.TrimLeft(text, " \t\r\n")
		at := len(text) - len(t)
		if strings.HasPrefix(t, "<|python_start|>") {
			t = strings.TrimLeft(t[len("<|python_start|>"):], " \t\r\n")
		}
		if !strings.HasPrefix(t, "[") {
			return nil
		}
		st := len(text) - len(t)
		calls, n, ok := pyCalls(t, ts)
		end := st + n
		if rest := text[end:]; strings.HasPrefix(strings.TrimLeft(rest, " \t\r\n"), "<|python_end|>") {
			end += strings.Index(rest, "<|python_end|>") + len("<|python_end|>")
		}
		return []toolSpan{{start: at, end: end, closed: ok, ok: ok, calls: calls}}
	case ToolSyntaxLFM2:
		return sections(text, "<|tool_call_start|>", "<|tool_call_end|>", func(b string) ([]ToolCall, bool) {
			calls, n, ok := pyCalls(strings.TrimSpace(b), ts)
			return calls, ok && n == len(strings.TrimSpace(b))
		})
	case ToolSyntaxHunyuan:
		return sections(text, "<tool_calls>", "</tool_calls>", func(b string) ([]ToolCall, bool) {
			return hunyuanCalls(b, ts)
		})
	case ToolSyntaxGemma4:
		return sections(text, "<|tool_call>", "<tool_call|>", func(b string) ([]ToolCall, bool) {
			c, ok := gemma4Call(b, ts)
			return []ToolCall{c}, ok
		})
	}
	return nil
}

// sections finds every open...close region and reads it with read; a region
// whose close has not arrived runs to the end of text and is read as it
// stands (a reply that ended at EOS before writing it).
func sections(text, open, close string, read func(body string) ([]ToolCall, bool)) []toolSpan {
	var out []toolSpan
	at := 0
	for {
		i := strings.Index(text[at:], open)
		if i < 0 {
			return out
		}
		st := at + i
		body := text[st+len(open):]
		sp := toolSpan{start: st, end: len(text)}
		if j := strings.Index(body, close); j >= 0 {
			body = body[:j]
			sp.end, sp.closed = st+len(open)+j+len(close), true
		}
		sp.calls, sp.ok = read(body)
		if sp.ok {
			for _, c := range sp.calls {
				if c.Name == "" {
					sp.ok = false
				}
			}
		}
		out = append(out, sp)
		at = sp.end
		if at >= len(text) {
			return out
		}
	}
}

// hermesCall reads one Hermes body.
func hermesCall(body string) (ToolCall, bool) {
	body = strings.TrimSpace(body)
	c, ok := decodeCall([]byte(body))
	// Qwen2.5's own template shows the call as
	// {{"name": <function-name>, "arguments": ...}} -- a doubled brace
	// transformers renders verbatim -- and the model copies it.
	// Qwen2.5-1.5B doubles only the opener, so both are tried.
	if !ok && strings.HasPrefix(body, "{{") {
		if c, ok = decodeCall([]byte(body[1:])); !ok && strings.HasSuffix(body, "}}") {
			c, ok = decodeCall([]byte(body[1 : len(body)-1]))
		}
	}
	return c, ok
}

// bareJSON reads a reply that opens with JSON calls: objects or arrays of
// them, Llama 3.1's ';' between. What follows the last call is text.
func bareJSON(text string, ts ToolSet) []toolSpan {
	t := strings.TrimLeft(text, " \t\r\n")
	at := len(text) - len(t)
	if t == "" || (t[0] != '{' && t[0] != '[') {
		return nil
	}
	sp := toolSpan{start: at, end: at}
	for {
		rest := text[sp.end:]
		lead := len(rest) - len(strings.TrimLeft(rest, " \t\r\n;"))
		r := rest[lead:]
		if r == "" || (r[0] != '{' && r[0] != '[') {
			break
		}
		d := json.NewDecoder(strings.NewReader(r))
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			// Unfinished, or not JSON: a stream holds it; at the end it is
			// text.
			if len(sp.calls) == 0 {
				return []toolSpan{{start: at, end: len(text)}}
			}
			break
		}
		calls, ok := listCalls(raw, nameArgs)
		for _, c := range calls {
			ok = ok && ts.known(c.Name)
		}
		if !ok {
			if len(sp.calls) == 0 {
				return []toolSpan{{start: at, end: at + int(d.InputOffset()) + lead, closed: true}}
			}
			break
		}
		sp.calls = append(sp.calls, calls...)
		sp.end += lead + int(d.InputOffset())
	}
	sp.closed, sp.ok = true, true
	return []toolSpan{sp}
}

// callShape reads one element of a JSON list of calls.
type callShape func(json.RawMessage) (ToolCall, bool)

// nameArgs is {"name", "arguments"|"parameters"}.
func nameArgs(r json.RawMessage) (ToolCall, bool) { return decodeCall(r) }

// cohereArgs is {"tool_name", "parameters"}.
func cohereArgs(r json.RawMessage) (ToolCall, bool) {
	var c struct {
		Name string          `json:"tool_name"`
		Args json.RawMessage `json:"parameters"`
	}
	if json.Unmarshal(r, &c) != nil || c.Name == "" {
		return ToolCall{}, false
	}
	a, ok := compactObject(c.Args)
	return ToolCall{Name: c.Name, Arguments: a}, ok
}

// keyedArgs is {"NAME": {...}}.
func keyedArgs(r json.RawMessage) (ToolCall, bool) {
	var m map[string]json.RawMessage
	if json.Unmarshal(r, &m) != nil || len(m) != 1 {
		return ToolCall{}, false
	}
	for n, a := range m {
		args, ok := compactObject(a)
		return ToolCall{Name: n, Arguments: args}, ok
	}
	return ToolCall{}, false
}

// listCalls is the calls of one JSON value: an object, or an array of them.
func listCalls(raw json.RawMessage, shape callShape) ([]ToolCall, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return nil, false
	}
	list := []json.RawMessage{raw}
	if raw[0] == '[' {
		if json.Unmarshal(raw, &list) != nil || len(list) == 0 {
			return nil, false
		}
	}
	calls := make([]ToolCall, 0, len(list))
	for _, r := range list {
		c, ok := shape(r)
		if !ok {
			return nil, false
		}
		calls = append(calls, c)
	}
	return calls, true
}

// jsonLists finds open, a JSON list of calls (fenced in ```json when fence),
// and close when the family writes one.
func jsonLists(text string, ts ToolSet, open, close string, fence bool, shape callShape) []toolSpan {
	var out []toolSpan
	at := 0
	for {
		i := strings.Index(text[at:], open)
		if i < 0 {
			return out
		}
		st := at + i
		p := st + len(open)
		skip := func() { p += len(text[p:]) - len(strings.TrimLeft(text[p:], " \t\r\n")) }
		skip()
		if fence && strings.HasPrefix(text[p:], "```") {
			p += 3
			if strings.HasPrefix(text[p:], "json") {
				p += 4
			}
			skip()
		}
		sp := toolSpan{start: st, end: len(text)}
		d := json.NewDecoder(strings.NewReader(text[p:]))
		var raw json.RawMessage
		if d.Decode(&raw) == nil {
			p += int(d.InputOffset())
			sp.calls, sp.ok = listCalls(raw, shape)
			for _, c := range sp.calls {
				sp.ok = sp.ok && ts.known(c.Name)
			}
			end := p
			skip()
			if fence && strings.HasPrefix(text[p:], "```") {
				p += 3
				end = p
				skip()
			}
			if close != "" && strings.HasPrefix(text[p:], close) {
				end = p + len(close)
			}
			sp.end = end
			sp.closed = close == "" || strings.HasSuffix(text[:end], close)
		}
		out = append(out, sp)
		if sp.end >= len(text) {
			return out
		}
		at = sp.end
	}
}

// mistralCalls reads [TOOL_CALLS] followed by a JSON list (v0.3) or by
// NAME[ARGS]{...} (the tokenizers from v11 on), several of either.
func mistralCalls(text string, ts ToolSet) []toolSpan {
	const open, args = "[TOOL_CALLS]", "[ARGS]"
	var out []toolSpan
	at := 0
	for {
		i := strings.Index(text[at:], open)
		if i < 0 {
			return out
		}
		st := at + i
		p := st + len(open)
		r := strings.TrimLeft(text[p:], " \t\r\n")
		if r != "" && (r[0] == '[' || r[0] == '{') {
			sps := jsonLists(text[st:], ts, open, "", false, nameArgs)
			sp := sps[0]
			sp.start, sp.end = sp.start+st, sp.end+st
			out = append(out, sp)
			at = sp.end
			if at >= len(text) {
				return out
			}
			continue
		}
		sp := toolSpan{start: st, end: len(text)}
		j := strings.Index(text[p:], args)
		if j >= 0 {
			// v13's tokenizer puts the call's id between: NAME[CALL_ID]ID[ARGS].
			name, _, _ := strings.Cut(text[p:p+j], "[CALL_ID]")
			name = strings.TrimSpace(name)
			q := p + j + len(args)
			d := json.NewDecoder(strings.NewReader(text[q:]))
			var raw json.RawMessage
			if d.Decode(&raw) == nil {
				a, ok := compactObject(raw)
				sp.calls = []ToolCall{{Name: name, Arguments: a}}
				sp.ok = ok && ts.known(name)
				sp.end, sp.closed = q+int(d.InputOffset()), true
			}
		}
		out = append(out, sp)
		at = sp.end
		if at >= len(text) {
			return out
		}
	}
}

// deepseekCalls reads the calls inside <｜tool▁calls▁begin｜>: each
// <｜tool▁call▁begin｜>HEAD<｜tool▁sep｜>REST<｜tool▁call▁end｜>, HEAD being
// "function" and REST NAME then fenced JSON (V3, R1), or HEAD the name and
// REST the JSON (V3.1).
func deepseekCalls(body string, ts ToolSet) ([]ToolCall, bool) {
	const begin, sep, end = "<｜tool▁call▁begin｜>", "<｜tool▁sep｜>", "<｜tool▁call▁end｜>"
	var calls []ToolCall
	for _, part := range strings.Split(body, begin)[1:] {
		part, _, _ = strings.Cut(part, end)
		head, rest, ok := strings.Cut(part, sep)
		if !ok {
			return nil, false
		}
		name := strings.TrimSpace(head)
		if name == "function" {
			n, r, _ := strings.Cut(rest, "\n")
			name, rest = strings.TrimSpace(n), r
		}
		a, ok := compactObject([]byte(unfence(rest)))
		if !ok || !ts.known(name) {
			return nil, false
		}
		calls = append(calls, ToolCall{Name: name, Arguments: a})
	}
	return calls, len(calls) > 0
}

// unfence is s without a ```json ... ``` fence around it.
func unfence(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(strings.TrimPrefix(s, "```"), "json")
		s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	}
	return strings.TrimSpace(s)
}

// kimiCalls reads Kimi-K2's calls: <|tool_call_begin|>functions.NAME:IDX
// <|tool_call_argument_begin|>{...}<|tool_call_end|>.
func kimiCalls(body string, ts ToolSet) ([]ToolCall, bool) {
	const begin, argb, end = "<|tool_call_begin|>", "<|tool_call_argument_begin|>", "<|tool_call_end|>"
	var calls []ToolCall
	for _, part := range strings.Split(body, begin)[1:] {
		part, _, _ = strings.Cut(part, end)
		id, args, ok := strings.Cut(part, argb)
		if !ok {
			return nil, false
		}
		id = strings.TrimSpace(id)
		if k := strings.LastIndexByte(id, ':'); k >= 0 {
			id = id[:k]
		}
		name := strings.TrimPrefix(id, "functions.")
		a, ok := compactObject([]byte(args))
		if !ok || !ts.known(name) {
			return nil, false
		}
		calls = append(calls, ToolCall{Name: name, Arguments: a})
	}
	return calls, len(calls) > 0
}

// xtmlCalls reads Kimi-K3's calls: <|open|>call tool="NAME" ...<|sep|>
// holding <|open|>argument key="K" type="T"<|sep|>V<|close|>argument<|sep|>
// each, or one <|open|>json ...<|sep|>{...}<|close|>json<|sep|>.
func xtmlCalls(body string, ts ToolSet) ([]ToolCall, bool) {
	const open, sep, closeCall = "<|open|>call", "<|sep|>", "<|close|>call<|sep|>"
	var calls []ToolCall
	for _, part := range strings.Split(body, open)[1:] {
		part, _, _ = strings.Cut(part, closeCall)
		attrs, inner, ok := strings.Cut(part, sep)
		if !ok {
			return nil, false
		}
		name := xtmlAttr(attrs, "tool")
		var args string
		if strings.HasPrefix(strings.TrimSpace(inner), "<|open|>json") {
			_, j, _ := strings.Cut(inner, sep)
			j, _, _ = strings.Cut(j, "<|close|>json")
			if args, ok = compactObject([]byte(j)); !ok {
				return nil, false
			}
		} else {
			var ob objectBuilder
			for _, a := range strings.Split(inner, "<|open|>argument")[1:] {
				at, v, ok := strings.Cut(a, sep)
				if !ok {
					return nil, false
				}
				v, _, _ = strings.Cut(v, "<|close|>argument")
				typ := xtmlAttr(at, "type")
				if typ == "" {
					typ = ts.propType(name, xtmlAttr(at, "key"))
				}
				ob.add(xtmlAttr(at, "key"), v, typ, false)
			}
			args = ob.String()
		}
		if !ts.known(name) {
			return nil, false
		}
		calls = append(calls, ToolCall{Name: name, Arguments: args})
	}
	return calls, len(calls) > 0
}

// xtmlAttr is attribute k of an XTML tag, unescaped as the template escapes
// it (& and ").
func xtmlAttr(attrs, k string) string {
	i := strings.Index(attrs, " "+k+`="`)
	if i < 0 {
		return ""
	}
	v := attrs[i+len(k)+3:]
	v, _, _ = strings.Cut(v, `"`)
	return strings.ReplaceAll(strings.ReplaceAll(v, "&quot;", `"`), "&amp;", "&")
}

// harmonyCalls reads gpt-oss's calls: a message whose header carries
// to=functions.NAME -- in the role (<|start|>assistant to=functions.NAME
// <|channel|>commentary) or the channel (<|channel|>commentary
// to=functions.NAME) -- and whose body, up to <|call|>, is the arguments.
func harmonyCalls(text string, ts ToolSet) []toolSpan {
	const to, msg = "to=functions.", "<|message|>"
	var out []toolSpan
	at := 0
	for {
		i := strings.Index(text[at:], to)
		if i < 0 {
			return out
		}
		i += at
		// The header begins at the message's <|start|>, or at its
		// <|channel|> when the reply opened inside the assistant's header.
		st := max(strings.LastIndex(text[:i], "<|start|>"), 0)
		if c := strings.LastIndex(text[:i], "<|channel|>"); c > st && !strings.Contains(text[st:c], msg) {
			// the channel is this message's own
		} else if c > st {
			st = c
		}
		if m := strings.LastIndex(text[:i], msg); m >= st {
			// "to=" sits in a body, not a header: text.
			at = i + len(to)
			continue
		}
		name := text[i+len(to):]
		if k := strings.IndexAny(name, " <\n"); k >= 0 {
			name = name[:k]
		}
		sp := toolSpan{start: st, end: len(text), calls: []ToolCall{{Name: name}}}
		if j := strings.Index(text[i:], msg); j >= 0 {
			body := text[i+j+len(msg):]
			end := len(text)
			for _, e := range []string{"<|call|>", "<|end|>"} {
				if k := strings.Index(body, e); k >= 0 && i+j+len(msg)+k+len(e) < end {
					end = i + j + len(msg) + k + len(e)
					body, sp.closed = body[:k], true
				}
			}
			a, ok := compactObject([]byte(body))
			sp.calls[0].Arguments, sp.ok = a, ok && ts.known(name)
			sp.end = end
		}
		out = append(out, sp)
		if sp.end >= len(text) {
			return out
		}
		at = sp.end
	}
}

// xmlFunctions reads the functions of an XML section: fnOpen NAME fnMid ...
// fnClose, each holding pOpen KEY pMid VALUE pClose parameters. trimNL drops
// the newline Qwen's template writes either side of a value.
func xmlFunctions(body string, ts ToolSet, fnOpen, fnMid, fnClose, pOpen, pMid, pClose string, trimNL bool) ([]ToolCall, bool) {
	var calls []ToolCall
	for _, part := range strings.Split(body, fnOpen)[1:] {
		part, _, _ = strings.Cut(part, fnClose)
		name, inner, ok := strings.Cut(part, fnMid)
		if !ok {
			return nil, false
		}
		name = strings.TrimSpace(name)
		var ob objectBuilder
		for _, p := range strings.Split(inner, pOpen)[1:] {
			k, v, ok := strings.Cut(p, pMid)
			if !ok {
				return nil, false
			}
			v, _, _ = strings.Cut(v, pClose)
			if trimNL {
				v = strings.TrimSuffix(strings.TrimPrefix(v, "\n"), "\n")
			}
			ob.add(strings.TrimSpace(k), v, ts.propType(name, strings.TrimSpace(k)), true)
		}
		if !ts.known(name) {
			return nil, false
		}
		calls = append(calls, ToolCall{Name: name, Arguments: ob.String()})
	}
	return calls, len(calls) > 0
}

// dsmlCalls reads DeepSeek V4's invokes: each parameter says string="true"
// (the text as it stands) or "false" (JSON).
func dsmlCalls(body string, ts ToolSet) ([]ToolCall, bool) {
	const fnOpen, fnClose, pOpen, pClose = `<｜DSML｜invoke name="`, "</｜DSML｜invoke>", `<｜DSML｜parameter name="`, "</｜DSML｜parameter>"
	var calls []ToolCall
	for _, part := range strings.Split(body, fnOpen)[1:] {
		part, _, _ = strings.Cut(part, fnClose)
		name, inner, ok := strings.Cut(part, `">`)
		if !ok {
			return nil, false
		}
		var ob objectBuilder
		for _, p := range strings.Split(inner, pOpen)[1:] {
			head, v, ok := strings.Cut(p, ">")
			if !ok {
				return nil, false
			}
			v, _, _ = strings.Cut(v, pClose)
			k, attrs, _ := strings.Cut(head, `"`)
			typ := "string"
			if strings.Contains(attrs, `string="false"`) {
				typ = "json"
			}
			ob.add(k, v, typ, false)
		}
		if !ts.known(name) {
			return nil, false
		}
		calls = append(calls, ToolCall{Name: name, Arguments: ob.String()})
	}
	return calls, len(calls) > 0
}

// glmCall reads GLM's NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
func glmCall(body string, ts ToolSet) (ToolCall, bool) {
	name, rest, _ := strings.Cut(body, "<arg_key>")
	name = strings.TrimSpace(name)
	var ob objectBuilder
	if rest != "" {
		for _, p := range strings.Split("<arg_key>"+rest, "<arg_key>")[1:] {
			k, v, ok := strings.Cut(p, "</arg_key>")
			if !ok {
				return ToolCall{}, false
			}
			_, v, ok = strings.Cut(v, "<arg_value>")
			if !ok {
				return ToolCall{}, false
			}
			v, _, _ = strings.Cut(v, "</arg_value>")
			ob.add(strings.TrimSpace(k), v, ts.propType(name, strings.TrimSpace(k)), true)
		}
	}
	return ToolCall{Name: name, Arguments: ob.String()}, ts.known(name)
}

// hunyuanCalls reads <tool_call>NAME ```json{...}```</tool_call> each.
func hunyuanCalls(body string, ts ToolSet) ([]ToolCall, bool) {
	var calls []ToolCall
	for _, part := range strings.Split(body, "<tool_call>")[1:] {
		part, _, _ = strings.Cut(part, "</tool_call>")
		name, args, _ := strings.Cut(part, "\n")
		name = strings.TrimSpace(name)
		a, ok := compactObject([]byte(unfence(args)))
		if !ok || !ts.known(name) {
			return nil, false
		}
		calls = append(calls, ToolCall{Name: name, Arguments: a})
	}
	return calls, len(calls) > 0
}

// objectBuilder writes a JSON object from arguments written one by one as
// text, in their order.
type objectBuilder struct {
	b strings.Builder
	n int
}

// add writes key with v: as a string where typ is "string" (or the value is
// not JSON), as JSON otherwise. guess reads a value of no declared type as
// JSON when it is a number, boolean, null, object or array -- what vLLM's
// XML parsers do -- rather than as text.
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

// holdTail is where text, holding no call, may be beginning one: at a
// marker, at a marker's first bytes at the very end, or -- for a family
// whose reply may simply BE a call -- at a reply that opens like one.
func (s ToolSyntax) holdTail(text string, replyStart bool) int {
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
	if replyStart && s.leadingJSON() {
		t := strings.TrimLeft(text, " \t\r\n")
		if t == "" || t[0] == '{' || t[0] == '[' {
			at(len(text) - len(t))
		}
	}
	if s == ToolSyntaxHarmony {
		// A header still open at the end may yet name a function.
		st := max(strings.LastIndex(text, "<|start|>"), strings.LastIndex(text, "<|channel|>"))
		if st >= 0 && !strings.Contains(text[st:], "<|message|>") {
			at(st)
		}
	}
	return h
}

// opens are the markers that begin a call.
func (s ToolSyntax) opens() []string {
	switch s {
	case ToolSyntaxHermes, ToolSyntaxQwenXML, ToolSyntaxGLM:
		return []string{"<tool_call>"}
	case ToolSyntaxSeedXML:
		return []string{"<seed:tool_call>"}
	case ToolSyntaxMiniMax:
		return []string{"<minimax:tool_call>"}
	case ToolSyntaxDSML:
		return []string{"<｜DSML｜tool_calls>"}
	case ToolSyntaxDeepSeekV3, ToolSyntaxDeepSeekV31:
		return []string{"<｜tool▁calls▁begin｜>"}
	case ToolSyntaxKimi:
		return []string{"<|tool_calls_section_begin|>"}
	case ToolSyntaxKimiXTML:
		return []string{"<|open|>tools<|sep|>"}
	case ToolSyntaxHarmony:
		return []string{"to=functions."}
	case ToolSyntaxMistral, ToolSyntaxMistralArgs:
		return []string{"[TOOL_CALLS]"}
	case ToolSyntaxGranite:
		return []string{"<|tool_call|>"}
	case ToolSyntaxNemotron:
		return []string{"<TOOLCALL>"}
	case ToolSyntaxCommandR7B:
		return []string{"<|START_ACTION|>"}
	case ToolSyntaxCommandR:
		return []string{"Action:"}
	case ToolSyntaxApertus:
		return []string{"<|tools_prefix|>"}
	case ToolSyntaxPhi4Mini:
		return []string{"functools"}
	case ToolSyntaxPythonic:
		return []string{"<|python_start|>"}
	case ToolSyntaxLFM2:
		return []string{"<|tool_call_start|>"}
	case ToolSyntaxHunyuan:
		return []string{"<tool_calls>"}
	case ToolSyntaxGemma4:
		return []string{"<|tool_call>"}
	}
	return nil
}

// leadingJSON is whether a reply that opens with JSON (or, for the pythonic
// families, a list) may be a call.
func (s ToolSyntax) leadingJSON() bool {
	switch s {
	case ToolSyntaxHermes, ToolSyntaxJSON, ToolSyntaxMistral, ToolSyntaxMistralArgs, ToolSyntaxPythonic:
		return true
	}
	return false
}

// pyCalls reads a Python list of calls, [f(a=1, b="x"), g()], from the start
// of s: the calls, how many bytes they took, and whether they read whole.
func pyCalls(s string, ts ToolSet) ([]ToolCall, int, bool) {
	p := &pyParser{s: s}
	if !p.eat('[') {
		return nil, 0, false
	}
	var calls []ToolCall
	for {
		p.ws()
		if p.eat(']') {
			break
		}
		if len(calls) > 0 && !p.eat(',') {
			return nil, len(s), false
		}
		p.ws()
		name := p.ident()
		if name == "" || !p.eat('(') {
			return nil, len(s), false
		}
		var ob objectBuilder
		for {
			p.ws()
			if p.eat(')') {
				break
			}
			if ob.n > 0 && !p.eat(',') {
				return nil, len(s), false
			}
			p.ws()
			k := p.ident()
			p.ws()
			if k == "" || !p.eat('=') {
				return nil, len(s), false
			}
			v, ok := p.value()
			if !ok {
				return nil, len(s), false
			}
			typ := ts.propType(name, k)
			if s, isStr := v.(string); isStr && typ != "" && typ != "string" {
				// Llama 4's template quotes every value; the schema says
				// what it is.
				var x any
				if json.Unmarshal([]byte(s), &x) == nil {
					v = x
				}
			}
			j, err := json.Marshal(v)
			if err != nil {
				return nil, len(s), false
			}
			ob.add(k, string(j), "json", false)
		}
		if !ts.known(name) {
			return nil, len(s), false
		}
		calls = append(calls, ToolCall{Name: name, Arguments: ob.String()})
	}
	return calls, p.i, len(calls) > 0
}

// pyParser reads the Python literals a call's arguments are written in.
type pyParser struct {
	s string
	i int
}

func (p *pyParser) ws() {
	for p.i < len(p.s) && strings.IndexByte(" \t\r\n", p.s[p.i]) >= 0 {
		p.i++
	}
}

func (p *pyParser) eat(c byte) bool {
	if p.i < len(p.s) && p.s[p.i] == c {
		p.i++
		return true
	}
	return false
}

func (p *pyParser) ident() string {
	st := p.i
	for p.i < len(p.s) {
		c := p.s[p.i]
		if c == '_' || c == '.' || c == '-' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' {
			p.i++
			continue
		}
		break
	}
	return p.s[st:p.i]
}

func (p *pyParser) value() (any, bool) {
	p.ws()
	if p.i >= len(p.s) {
		return nil, false
	}
	switch c := p.s[p.i]; {
	case c == '"' || c == '\'':
		return p.str(c)
	case c == '[':
		p.i++
		var out []any
		for {
			p.ws()
			if p.eat(']') {
				return out, true
			}
			if len(out) > 0 && !p.eat(',') {
				return nil, false
			}
			p.ws()
			if p.eat(']') {
				return out, true
			}
			v, ok := p.value()
			if !ok {
				return nil, false
			}
			out = append(out, v)
		}
	case c == '{':
		p.i++
		out := map[string]any{}
		for {
			p.ws()
			if p.eat('}') {
				return out, true
			}
			if len(out) > 0 && !p.eat(',') {
				return nil, false
			}
			p.ws()
			k, ok := p.value()
			ks, isStr := k.(string)
			p.ws()
			if !ok || !isStr || !p.eat(':') {
				return nil, false
			}
			v, ok := p.value()
			if !ok {
				return nil, false
			}
			out[ks] = v
		}
	}
	w := p.ident()
	switch w {
	case "True", "true":
		return true, true
	case "False", "false":
		return false, true
	case "None", "null":
		return nil, true
	}
	if p.i < len(p.s) && (p.s[p.i] == '-' || p.s[p.i] == '+') && w == "" {
		p.i++
		w = p.s[p.i-1:p.i] + p.ident()
	}
	if f, err := strconv.ParseFloat(w, 64); err == nil && w != "" {
		return json.Number(strconv.FormatFloat(f, 'g', -1, 64)), true
	}
	return nil, false
}

func (p *pyParser) str(q byte) (any, bool) {
	p.i++
	var b strings.Builder
	for p.i < len(p.s) {
		c := p.s[p.i]
		p.i++
		switch c {
		case q:
			return b.String(), true
		case '\\':
			if p.i >= len(p.s) {
				return nil, false
			}
			e := p.s[p.i]
			p.i++
			switch e {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			default:
				b.WriteByte(e)
			}
		default:
			b.WriteByte(c)
		}
	}
	return nil, false
}

// gemma4Call reads call:NAME{key:value,...}, strings between <|"|>.
func gemma4Call(body string, ts ToolSet) (ToolCall, bool) {
	body = strings.TrimSpace(body)
	if !strings.HasPrefix(body, "call:") {
		return ToolCall{}, false
	}
	name, rest, ok := strings.Cut(body[len("call:"):], "{")
	if !ok {
		return ToolCall{}, false
	}
	g := &gemmaParser{s: "{" + rest}
	v, ok := g.value()
	obj, isObj := v.(*orderedObject)
	if !ok || !isObj {
		return ToolCall{}, false
	}
	name = strings.TrimSpace(name)
	return ToolCall{Name: name, Arguments: obj.json()}, ts.known(name)
}

// orderedObject is a JSON object keeping its keys' order.
type orderedObject struct {
	keys []string
	vals []any
}

func (o *orderedObject) json() string {
	var ob objectBuilder
	for i, k := range o.keys {
		ob.add(k, jsonOf(o.vals[i]), "json", false)
	}
	return ob.String()
}

func jsonOf(v any) string {
	switch x := v.(type) {
	case *orderedObject:
		return x.json()
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = jsonOf(e)
		}
		return "[" + strings.Join(parts, ",") + "]"
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// gemmaParser reads Gemma 4's argument notation: JSON with bare keys and
// strings between <|"|> marks.
type gemmaParser struct {
	s string
	i int
}

const gemmaQuote = `<|"|>`

func (g *gemmaParser) value() (any, bool) {
	s := g.s[g.i:]
	switch {
	case strings.HasPrefix(s, gemmaQuote):
		j := strings.Index(s[len(gemmaQuote):], gemmaQuote)
		if j < 0 {
			return nil, false
		}
		g.i += len(gemmaQuote) + j + len(gemmaQuote)
		return s[len(gemmaQuote) : len(gemmaQuote)+j], true
	case strings.HasPrefix(s, "{"):
		g.i++
		o := &orderedObject{}
		for {
			if g.eat("}") {
				return o, true
			}
			if len(o.keys) > 0 && !g.eat(",") {
				return nil, false
			}
			k := g.s[g.i:]
			c := strings.IndexByte(k, ':')
			if c <= 0 {
				return nil, false
			}
			g.i += c + 1
			v, ok := g.value()
			if !ok {
				return nil, false
			}
			o.keys, o.vals = append(o.keys, strings.TrimSpace(k[:c])), append(o.vals, v)
		}
	case strings.HasPrefix(s, "["):
		g.i++
		var out []any
		for {
			if g.eat("]") {
				return out, true
			}
			if len(out) > 0 && !g.eat(",") {
				return nil, false
			}
			v, ok := g.value()
			if !ok {
				return nil, false
			}
			out = append(out, v)
		}
	}
	end := strings.IndexAny(s, ",}]")
	if end < 0 {
		end = len(s)
	}
	w := strings.TrimSpace(s[:end])
	g.i += end
	switch w {
	case "true":
		return true, true
	case "false":
		return false, true
	case "null":
		return nil, true
	}
	if _, err := strconv.ParseFloat(w, 64); err == nil {
		return json.Number(w), true
	}
	return nil, false
}

func (g *gemmaParser) eat(t string) bool {
	if strings.HasPrefix(g.s[g.i:], t) {
		g.i += len(t)
		return true
	}
	return false
}
