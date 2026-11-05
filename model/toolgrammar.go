package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/engine/grammar"
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
	syn := m.ToolSyntax()
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
	if s == ToolSyntaxHarmony {
		return ReasonMay
	}
	if o := strings.LastIndex(prompt, open); o >= 0 && o > strings.LastIndex(prompt, close) {
		return ReasonInside
	}
	if strings.HasSuffix(strings.TrimRight(prompt, " \t\n"), close) {
		return ReasonNone
	}
	mark := close
	if s == ToolSyntaxKimiXTML {
		mark = "'think'" // the template writes its tags through a macro
	}
	if strings.Contains(src, mark) {
		return ReasonMay
	}
	return ReasonNone
}

// thinkMarks are the markers a reasoning block opens and closes with; the
// close of harmony's analysis message is its <|end|>, after which the reply
// opens its next message.
func (s ToolSyntax) thinkMarks() (open, close string) {
	switch s {
	case ToolSyntaxHarmony:
		return "<|channel|>analysis<|message|>", "<|end|>"
	case ToolSyntaxKimiXTML:
		return "<|open|>think<|sep|>", "<|close|>think<|sep|>"
	case ToolSyntaxSeedXML:
		return "<seed:think>", "</seed:think>"
	case ToolSyntaxMistral, ToolSyntaxMistralArgs:
		return "[THINK]", "[/THINK]"
	case ToolSyntaxCommandR7B:
		return "<|START_THINKING|>", "<|END_THINKING|>"
	}
	return "<think>", "</think>"
}

// ToolGrammarOf is ToolGrammar for a syntax.
func ToolGrammarOf(syn ToolSyntax, reason Reasoning, tools ToolSet, choice ToolChoice) (string, error) {
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
	g := &toolGrammar{b: grammar.NewBuilder(), tools: tools, syn: syn}
	many := !choice.Single && choice.Mode != ToolChoiceFunction && syn.writesSeveral()
	calls, err := g.calls(names, many)
	if err != nil {
		return "", err
	}
	// A reply may open with a newline or two, as several templates write
	// one before the call.
	ws := g.b.Rule("lead", `[ \t\n]{0,4}`)
	open, close := syn.thinkMarks()
	body := g.b.Until("thought", close) + " " + lit(close)
	if syn == ToolSyntaxHarmony {
		body += " " + lit("<|start|>assistant")
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

// writesSeveral says whether one reply in this syntax can carry several
// calls (harmony ends its message at <|call|>, an end of generation).
func (s ToolSyntax) writesSeveral() bool { return s != ToolSyntaxHarmony }

// ToolMarkers are the texts this syntax's grammar writes that a vocabulary
// may hold as control tokens: the mask must offer those tokens by their
// literal text, which Piece does not give.
func (s ToolSyntax) ToolMarkers() []string {
	switch s {
	case ToolSyntaxHermes, ToolSyntaxQwenXML, ToolSyntaxGLM:
		return []string{"<tool_call>", "</tool_call>", "<think>", "</think>"}
	case ToolSyntaxSeedXML:
		return []string{"<seed:tool_call>", "</seed:tool_call>", "<seed:think>", "</seed:think>"}
	case ToolSyntaxMiniMax:
		return []string{"<minimax:tool_call>", "</minimax:tool_call>", "<think>", "</think>"}
	case ToolSyntaxDeepSeekV3, ToolSyntaxDeepSeekV31:
		return []string{"<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>", "<｜tool▁call▁begin｜>",
			"<｜tool▁call▁end｜>", "<｜tool▁sep｜>", "<think>", "</think>"}
	case ToolSyntaxKimi:
		return []string{"<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>",
			"<|tool_call_argument_begin|>", "<|tool_call_end|>", "<think>", "</think>"}
	case ToolSyntaxKimiXTML:
		return []string{"<|open|>", "<|close|>", "<|sep|>"}
	case ToolSyntaxHarmony:
		return []string{"<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|constrain|>"}
	case ToolSyntaxMistral, ToolSyntaxMistralArgs:
		return []string{"[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "[THINK]", "[/THINK]"}
	case ToolSyntaxGranite:
		return []string{"<|tool_call|>"}
	case ToolSyntaxNemotron:
		return []string{"<TOOLCALL>", "</TOOLCALL>", "<think>", "</think>"}
	case ToolSyntaxCommandR7B:
		return []string{"<|START_ACTION|>", "<|END_ACTION|>", "<|START_THINKING|>", "<|END_THINKING|>"}
	case ToolSyntaxApertus:
		return []string{"<|tools_prefix|>", "<|tools_suffix|>"}
	case ToolSyntaxPythonic:
		return []string{"<|python_start|>", "<|python_end|>"}
	case ToolSyntaxLFM2:
		return []string{"<|tool_call_start|>", "<|tool_call_end|>", "<think>", "</think>"}
	case ToolSyntaxHunyuan:
		return []string{"<tool_calls>", "</tool_calls>", "<tool_call>", "</tool_call>", "<think>", "</think>"}
	case ToolSyntaxGemma4:
		return []string{"<|tool_call>", "<tool_call|>", `<|"|>`}
	case ToolSyntaxDSML:
		return []string{"<think>", "</think>"}
	}
	return nil
}

type toolGrammar struct {
	b     *grammar.Builder
	tools ToolSet
	syn   ToolSyntax
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

// props is name's arguments one by one, for a syntax that writes them so: nil
// with ok false where the schema is not built, and the arguments are then any
// JSON object's members (a syntax that cannot say that is refused).
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

// xmlArgs is name's parameters written each as open KEY mid VALUE close, a
// string's value as text up to close and any other as JSON.
func (g *toolGrammar) xmlArgs(name string, item func(key, value string) string, strClose string) (string, error) {
	ps, err := g.props(name)
	if err != nil {
		return "", err
	}
	var items []string
	var req []bool
	for _, p := range ps {
		v := p.Rule
		if p.Type == "string" {
			v = g.b.Until("text-"+name, strClose)
		}
		items = append(items, item(p.Key, v))
		req = append(req, p.Required)
	}
	return seq(items, req), nil
}

func lit(s string) string { return grammar.Literal(s) }

func jsonName(n string) string {
	b, _ := json.Marshal(n)
	return string(b)
}

// one is a single call to name in the syntax.
func (g *toolGrammar) one(name string) (string, error) {
	ln := lit(name)
	switch g.syn {
	case ToolSyntaxHermes:
		a, err := g.args(name)
		return lit("<tool_call>\n{\"name\": "+jsonName(name)+", \"arguments\": ") + " " + a + " " + lit("}\n</tool_call>"), err
	case ToolSyntaxJSON:
		a, err := g.args(name)
		return lit("{\"name\": "+jsonName(name)+", \"parameters\": ") + " " + a + " " + lit("}"), err
	case ToolSyntaxMistral, ToolSyntaxGranite, ToolSyntaxNemotron, ToolSyntaxPhi4Mini:
		a, err := g.args(name)
		return lit("{\"name\": "+jsonName(name)+", \"arguments\": ") + " " + a + " " + lit("}"), err
	case ToolSyntaxCommandR7B:
		a, err := g.args(name)
		return lit("{\"tool_call_id\": \"") + " [0-9]{1,3} " + lit("\", \"tool_name\": "+jsonName(name)+", \"parameters\": ") +
			" " + a + " " + lit("}"), err
	case ToolSyntaxCommandR:
		a, err := g.args(name)
		return lit("{\"tool_name\": "+jsonName(name)+", \"parameters\": ") + " " + a + " " + lit("}"), err
	case ToolSyntaxApertus:
		a, err := g.args(name)
		return lit("{"+jsonName(name)+": ") + " " + a + " " + lit("}"), err
	case ToolSyntaxMistralArgs:
		a, err := g.args(name)
		// From the v13 tokenizer on the model writes the call's id too.
		return lit("[TOOL_CALLS]") + " " + ln + " ( " + lit("[CALL_ID]") + " [a-zA-Z0-9]{9} )? " + lit("[ARGS]") + " " + a, err
	case ToolSyntaxDeepSeekV31:
		a, err := g.args(name)
		return lit("<｜tool▁call▁begin｜>") + " " + ln + " " + lit("<｜tool▁sep｜>") + " " + a + " " + lit("<｜tool▁call▁end｜>"), err
	case ToolSyntaxDeepSeekV3:
		a, err := g.args(name)
		return lit("<｜tool▁call▁begin｜>function<｜tool▁sep｜>"+name+"\n```json\n") + " " + a + " " +
			lit("\n```<｜tool▁call▁end｜>"), err
	case ToolSyntaxKimi:
		a, err := g.args(name)
		return lit("<|tool_call_begin|>functions."+name+":") + " [0-9]{1,3} " + lit("<|tool_call_argument_begin|>") +
			" " + a + " " + lit("<|tool_call_end|>"), err
	case ToolSyntaxKimiXTML:
		a, err := g.args(name)
		return lit(`<|open|>call tool="`+xtmlEscape(name)+`" index="`) + " [1-9] [0-9]{0,2} " +
			lit(`"<|sep|><|open|>json type="object"<|sep|>`) + " " + a + " " +
			lit("<|close|>json<|sep|><|close|>call<|sep|>"), err
	case ToolSyntaxHarmony:
		a, err := g.args(name)
		return lit("<|channel|>commentary to=functions."+name+" <|constrain|>json<|message|>") + " " + a, err
	case ToolSyntaxHunyuan:
		a, err := g.args(name)
		return lit("<tool_call>"+name+"\n```json\n") + " " + a + " " + lit("\n```</tool_call>"), err
	case ToolSyntaxQwenXML:
		a, err := g.xmlArgs(name, func(k, v string) string {
			return lit("<parameter="+k+">\n") + " " + v + " " + lit("\n</parameter>\n")
		}, "\n</parameter>")
		return lit("<function="+name+">\n") + " " + a + " " + lit("</function>\n"), err
	case ToolSyntaxSeedXML:
		a, err := g.xmlArgs(name, func(k, v string) string {
			return lit("<parameter="+k+">") + " " + v + " " + lit("</parameter>\n")
		}, "</parameter>")
		return lit("<function="+name+">\n") + " " + a + " " + lit("</function>\n"), err
	case ToolSyntaxMiniMax:
		a, err := g.xmlArgs(name, func(k, v string) string {
			return lit(`<parameter name="`+k+`">`) + " " + v + " " + lit("</parameter>\n")
		}, "</parameter>")
		return lit(`<invoke name="`+name+"\">\n") + " " + a + " " + lit("</invoke>\n"), err
	case ToolSyntaxDSML:
		ps, err := g.props(name)
		if err != nil {
			return "", err
		}
		var items []string
		var req []bool
		for _, p := range ps {
			it := lit(`<｜DSML｜parameter name="`+p.Key+`" string="false">`) + " " + p.Rule + " " + lit("</｜DSML｜parameter>\n")
			if p.Type == "string" {
				it = lit(`<｜DSML｜parameter name="`+p.Key+`" string="true">`) + " " +
					g.b.Until("text-"+name, "</｜DSML｜parameter>") + " " + lit("</｜DSML｜parameter>\n")
			}
			items, req = append(items, it), append(req, p.Required)
		}
		return lit(`<｜DSML｜invoke name="`+name+"\">\n") + " " + seq(items, req) + " " + lit("</｜DSML｜invoke>\n"), nil
	case ToolSyntaxGLM:
		a, err := g.xmlArgs(name, func(k, v string) string {
			return lit("<arg_key>"+k+"</arg_key>\n<arg_value>") + " " + v + " " + lit("</arg_value>\n")
		}, "</arg_value>")
		return lit("<tool_call>"+name+"\n") + " " + a + " " + lit("</tool_call>"), err
	case ToolSyntaxPythonic, ToolSyntaxLFM2:
		ps, err := g.props(name)
		if err != nil {
			return "", err
		}
		var items []string
		var req []bool
		for _, p := range ps {
			items, req = append(items, lit(p.Key+"=")+" "+p.Rule), append(req, p.Required)
		}
		return ln + " " + lit("(") + " " + g.list("py-"+name, items, req, lit(", ")) + " " + lit(")"), nil
	case ToolSyntaxGemma4:
		ps, err := g.props(name)
		if err != nil {
			return "", err
		}
		slices.SortFunc(ps, func(a, b grammar.Prop) int { return strings.Compare(a.Key, b.Key) })
		var items []string
		var req []bool
		for _, p := range ps {
			items, req = append(items, lit(p.Key+":")+" "+g.gemmaValue(p)), append(req, p.Required)
		}
		return lit("<|tool_call>call:"+name+"{") + " " + g.list("gm-"+name, items, req, lit(",")) + " " + lit("}<tool_call|>"), nil
	}
	return "", fmt.Errorf("tool syntax %s has no grammar", g.syn)
}

// gemmaValue is a Gemma 4 argument's value: a string between <|"|> marks,
// a number or boolean as JSON writes it, anything else in Gemma's notation.
func (g *toolGrammar) gemmaValue(p grammar.Prop) string {
	q := lit(gemmaQuote)
	switch p.Type {
	case "string":
		return q + " " + g.b.Until("gm-text", gemmaQuote) + " " + q
	case "integer":
		return g.b.Rule("gm-int", `"-"? ([0] | [1-9] [0-9]{0,15})`)
	case "number":
		return g.b.Rule("gm-num", `"-"? ([0] | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})?`)
	case "boolean":
		return g.b.Rule("gm-bool", `"true" | "false"`)
	}
	str := q + " " + g.b.Until("gm-text", gemmaQuote) + " " + q
	key := g.b.Rule("gm-key", "[a-zA-Z_] [a-zA-Z0-9_]*")
	val := g.b.Rule("gm-value", str+` | "-"? [0-9]+ ("." [0-9]+)? | "true" | "false" | "null" | gm-obj | gm-arr`)
	g.b.Rule("gm-obj", fmt.Sprintf(`"{" ( %s ":" %s ( "," %s ":" %s )* )? "}"`, key, val, key, val))
	g.b.Rule("gm-arr", fmt.Sprintf(`"[" ( %s ( "," %s )* )? "]"`, val, val))
	return val
}

// calls is one call to any of names, or one or more when many.
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
	sep, open, close := "", "", ""
	switch g.syn {
	case ToolSyntaxHermes, ToolSyntaxGLM, ToolSyntaxQwenXML:
		sep = lit("\n")
	case ToolSyntaxJSON:
		sep = lit("; ")
	case ToolSyntaxMistral:
		open, sep, close = lit("[TOOL_CALLS]["), lit(", "), lit("]")
	case ToolSyntaxGranite:
		open, sep, close = lit("<|tool_call|>["), lit(", "), lit("]")
	case ToolSyntaxNemotron:
		open, sep, close = lit("<TOOLCALL>["), lit(", "), lit("]</TOOLCALL>")
	case ToolSyntaxPhi4Mini:
		open, sep, close = lit("functools["), lit(", "), lit("]")
	case ToolSyntaxApertus:
		open, sep, close = lit("<|tools_prefix|>["), lit(", "), lit("]<|tools_suffix|>")
	case ToolSyntaxCommandR7B:
		open, sep, close = lit("<|START_ACTION|>[\n    "), lit(",\n    "), lit("\n]<|END_ACTION|>")
	case ToolSyntaxCommandR:
		open, sep, close = lit("Action: ```json\n[\n    "), lit(",\n    "), lit("\n]\n```")
	case ToolSyntaxDeepSeekV3, ToolSyntaxDeepSeekV31:
		open, sep, close = lit("<｜tool▁calls▁begin｜>"), lit("\n"), lit("<｜tool▁calls▁end｜>")
	case ToolSyntaxKimi:
		open, close = lit("<|tool_calls_section_begin|>"), lit("<|tool_calls_section_end|>")
	case ToolSyntaxKimiXTML:
		open, close = lit("<|open|>tools<|sep|>"), lit("<|close|>tools<|sep|>")
	case ToolSyntaxPythonic:
		open, sep, close = lit("["), lit(", "), lit("]")
	case ToolSyntaxLFM2:
		open, sep, close = lit("<|tool_call_start|>["), lit(", "), lit("]<|tool_call_end|>")
	case ToolSyntaxHunyuan:
		open, close = lit("<tool_calls>"), lit("</tool_calls>")
	}
	switch g.syn {
	case ToolSyntaxQwenXML:
		call = g.b.Rule("section", lit("<tool_call>\n")+" "+call+" "+lit("</tool_call>"))
	case ToolSyntaxSeedXML:
		call = g.b.Rule("section", lit("<seed:tool_call>\n")+" "+call+" "+lit("</seed:tool_call>"))
		sep = lit("\n")
	case ToolSyntaxMiniMax:
		open, close = lit("<minimax:tool_call>\n"), lit("</minimax:tool_call>")
	case ToolSyntaxDSML:
		open, close = "( "+lit("\n\n")+" )? "+lit("<｜DSML｜tool_calls>\n"), lit("</｜DSML｜tool_calls>")
	}
	body := call
	if many {
		body = call + " ( " + sep + " " + call + " )*"
		if sep == "" {
			body = call + "+"
		}
	}
	return strings.TrimSpace(open + " " + body + " " + close), nil
}

// xtmlEscape is an XTML attribute value as Kimi-K3's template escapes it.
func xtmlEscape(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "&", "&amp;"), `"`, "&quot;")
}
