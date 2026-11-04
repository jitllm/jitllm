package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/grammar"
)

// The tool-call syntaxes are held to each family's own template: the
// template renders an assistant turn that calls a tool, the turn's text after
// the generation prompt is what the model was trained to write, and that text
// must parse back to the call -- and must be a reply the forced-call grammar
// accepts.

const syntaxTools = `[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}, "days": {"type": "integer", "description": "Forecast length"}, "radius_km": {"type": "number", "description": "Search radius"}}, "required": ["location"]}}}, {"type": "function", "function": {"name": "get_time", "description": "Get the time in a zone", "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}}}}]`

const syntaxArgs = `{"location": "Zürich", "unit": "celsius", "days": 3, "radius_km": 10.5}`

// syntaxCase is one family's template. sample, when set, is the reply to
// read instead of the template's rendering: a template that does not render
// an assistant's calls (it only describes them), or one whose rendering is
// not what the model writes (harmony's recipient sits in the channel when
// gpt-oss writes it). noGrammar marks a rendering the grammar writes
// differently on purpose.
type syntaxCase struct {
	file      string
	syn       ToolSyntax
	sample    string
	noGrammar string
}

func syntaxCases() []syntaxCase {
	return []syntaxCase{
		{file: "qwen2.5.jinja", syn: ToolSyntaxHermes},
		{file: "qwen3.jinja", syn: ToolSyntaxHermes},
		{file: "qwen3-next.jinja", syn: ToolSyntaxHermes},
		{file: "internvl3.jinja", syn: ToolSyntaxHermes},
		{file: "jamba.jinja", syn: ToolSyntaxHermes,
			sample: "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}\n</tool_call>"},
		{file: "granite4.jinja", syn: ToolSyntaxHermes},
		{file: "exaone4.jinja", syn: ToolSyntaxHermes, noGrammar: "EXAONE writes no newlines inside <tool_call>"},
		{file: "smollm3.jinja", syn: ToolSyntaxHermes,
			sample: "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}\n</tool_call>"},
		{file: "falcon-h1.jinja", syn: ToolSyntaxHermes,
			sample: "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}\n</tool_call>"},
		{file: "qwen3.5.jinja", syn: ToolSyntaxQwenXML},
		{file: "nemotron3.jinja", syn: ToolSyntaxQwenXML},
		{file: "Qwen3-Coder.jinja", syn: ToolSyntaxQwenXML},
		{file: "seed-oss.jinja", syn: ToolSyntaxSeedXML},
		{file: "glm4.5.jinja", syn: ToolSyntaxGLM},
		{file: "GLM-4.6.jinja", syn: ToolSyntaxGLM},
		{file: "GLM-4.7-Flash.jinja", syn: ToolSyntaxGLM, noGrammar: "GLM-4.7 writes no newlines between its tags"},
		{file: "inclusionai-ling-3.0-flash.jinja", syn: ToolSyntaxGLM, noGrammar: "Ling writes no newlines between its tags"},
		{file: "minimax-m2.jinja", syn: ToolSyntaxMiniMax},
		{file: "deepseek-ai-DeepSeek-V4.jinja", syn: ToolSyntaxDSML},
		{file: "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja", syn: ToolSyntaxDeepSeekV3},
		{file: "deepseek-v3.2.jinja", syn: ToolSyntaxDeepSeekV31},
		{file: "deepseek-ai-DeepSeek-V3.1.jinja", syn: ToolSyntaxDeepSeekV31},
		{file: "Kimi-K2-Instruct.jinja", syn: ToolSyntaxKimi},
		{file: "Kimi-K3.jinja", syn: ToolSyntaxKimiXTML, noGrammar: "the grammar writes the arguments as one JSON block, the template's other form"},
		{file: "gpt-oss.jinja", syn: ToolSyntaxHarmony,
			sample: "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>" + syntaxArgs},
		{file: "mistral-v0.3.jinja", syn: ToolSyntaxHermes,
			sample:    "[{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}]",
			noGrammar: "this template declares no tools: the default syntax's grammar writes Hermes"},
		{file: "ministral3.jinja", syn: ToolSyntaxMistralArgs},
		{file: "Mistral-Small-3.2-24B-Instruct-2506.jinja", syn: ToolSyntaxMistralArgs},
		{file: "granite3.3.jinja", syn: ToolSyntaxGranite,
			sample: "<|tool_call|>[{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}]"},
		{file: "ibm-granite-granite-3.3-2B-Instruct.jinja", syn: ToolSyntaxGranite,
			sample: "<|tool_call|>[{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}]"},
		{file: "nemotron-h.jinja", syn: ToolSyntaxNemotron,
			sample: "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}]</TOOLCALL>"},
		{file: "command-r7b.tool_use.jinja", syn: ToolSyntaxCommandR7B},
		{file: "command-r.tool_use.jinja", syn: ToolSyntaxCommandR,
			sample:    "Action: ```json\n[\n    {\n        \"tool_name\": \"get_weather\",\n        \"parameters\": " + syntaxArgs + "\n    }\n]\n```",
			noGrammar: "the sample is indented as the template's example is; the grammar writes one line per call"},
		{file: "Apertus-8B-Instruct.jinja", syn: ToolSyntaxApertus},
		{file: "phi4-mini.jinja", syn: ToolSyntaxPhi4Mini,
			sample: "functools[{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs + "}]"},
		{file: "llama3.jinja", syn: ToolSyntaxJSON},
		{file: "meta-llama-Llama-3.2-3B-Instruct.jinja", syn: ToolSyntaxJSON},
		{file: "llama4.jinja", syn: ToolSyntaxPythonic, noGrammar: "Llama 4's template quotes every value; the grammar writes each as its type"},
		{file: "lfm2-moe.jinja", syn: ToolSyntaxLFM2,
			sample: "<|tool_call_start|>[get_weather(location=\"Zürich\", unit=\"celsius\", days=3, radius_km=10.5)]<|tool_call_end|>"},
		{file: "LFM2.5-Instruct.jinja", syn: ToolSyntaxLFM2,
			sample: "<|tool_call_start|>[get_weather(location=\"Zürich\", unit=\"celsius\", days=3, radius_km=10.5)]<|tool_call_end|>"},
		{file: "hunyuan.jinja", syn: ToolSyntaxHunyuan},
		{file: "gemma4.jinja", syn: ToolSyntaxGemma4},
	}
}

// readSyntaxTemplate finds a template in tok/jinja's set or this package's:
// testdata/tooltemplates holds, byte for byte, llama.cpp's models/templates
// copies of the families tok/jinja's set does not carry.
func readSyntaxTemplate(t *testing.T, file string) string {
	for _, dir := range []string{filepath.Join("..", "..", "tok", "jinja", "testdata", "templates"),
		filepath.Join("testdata", "tooltemplates")} {
		if b, err := os.ReadFile(filepath.Join(dir, file)); err == nil {
			return string(b)
		}
	}
	t.Fatalf("%s: not in either template set", file)
	return ""
}

// renderedCall is the template's assistant turn that calls get_weather: the
// conversation with the call, less the same conversation's generation prompt.
func renderedCall(t *testing.T, src string) (reply, prompt string) {
	user := ChatMessage{Role: "user", Content: "What is the weather in Zürich for the next 3 days?"}
	call := ChatMessage{Role: "assistant", ToolCalls: []ToolCall{{ID: "a1b2c3d4e", Name: "get_weather", Arguments: syntaxArgs}}}
	p, err := renderChatTemplate(src, "<s>", "</s>", nil, []ChatMessage{user}, []byte(syntaxTools), true)
	if err != nil {
		t.Fatalf("generation prompt: %v", err)
	}
	f, err := renderChatTemplate(src, "<s>", "</s>", nil, []ChatMessage{user, call}, []byte(syntaxTools), false)
	if err != nil {
		t.Fatalf("the call: %v", err)
	}
	i := 0
	for i < len(p) && i < len(f) && p[i] == f[i] {
		i++
	}
	// A marker the prompt and the call both open with ("<" of "<think>" and
	// of "<｜tool▁calls▁begin｜>") belongs to the call.
	if i > 0 && (f[i-1] == '<' || f[i-1] == '[') {
		i--
	}
	return f[i:], p[:i]
}

func sameArgs(t *testing.T, got, want string) bool {
	var g, w any
	if err := json.Unmarshal([]byte(got), &g); err != nil {
		t.Errorf("arguments %q are not JSON: %v", got, err)
		return false
	}
	json.Unmarshal([]byte(want), &w)
	return reflect.DeepEqual(g, w)
}

func TestToolSyntaxReadsItsTemplatesCalls(t *testing.T) {
	tools := ParseToolSet([]byte(syntaxTools))
	for _, c := range syntaxCases() {
		t.Run(c.file, func(t *testing.T) {
			src := readSyntaxTemplate(t, c.file)
			if got := ToolSyntaxOf(src); got != c.syn {
				t.Fatalf("syntax %s, want %s", got, c.syn)
			}
			reply, prompt := c.sample, ""
			if reply == "" {
				reply, prompt = renderedCall(t, src)
			}
			content, calls := c.syn.Parse(reply, tools)
			if len(calls) != 1 || calls[0].Name != "get_weather" || !sameArgs(t, calls[0].Arguments, syntaxArgs) {
				t.Fatalf("reply %q read as %+v", reply, calls)
			}
			if strings.Contains(content, "get_weather") || strings.Contains(content, "Zürich") {
				t.Errorf("the call's markup stayed in the text: %q", content)
			}
			// The declared-name check: the same call to a tool the request
			// did not declare is text.
			only := ParseToolSet([]byte(`[{"type": "function", "function": {"name": "get_time"}}]`))
			if _, calls := c.syn.Parse(reply, only); len(calls) != 0 {
				t.Errorf("a call to an undeclared tool was read: %+v", calls)
			}
			if c.noGrammar != "" {
				t.Logf("grammar not held to this rendering: %s", c.noGrammar)
				return
			}
			g, err := ToolGrammarOf(c.syn, c.syn.Reasoning(src, prompt), tools, ToolChoice{Mode: ToolChoiceFunction, Name: "get_weather"})
			if err != nil {
				t.Fatal(err)
			}
			if !grammarAccepts(t, g, strings.TrimRight(reply, " \n")) {
				t.Errorf("the forced-call grammar refuses the family's own call %q:\n%s", reply, g)
			}
		})
	}
}

// grammarAccepts reports whether src accepts some prefix of text that ends
// where the grammar may end, the rest being the template's end-of-turn
// markup.
func grammarAccepts(t *testing.T, src, text string) bool {
	gr, err := grammar.Parse(src)
	if err != nil {
		t.Fatalf("grammar: %v\n%s", err, src)
	}
	m := grammar.NewMatcher(gr, grammar.IndexTokens(charVocab{}))
	st := m.Start()
	for i := 0; i < len(text); i++ {
		if m.Done(st) && i > 0 {
			return true
		}
		next, ok := m.AcceptText(st, text[i:i+1])
		if !ok {
			t.Logf("refused at byte %d: %q|%q", i, text[:i], text[i:])
			return false
		}
		st = next
	}
	return m.Done(st)
}

// charVocab is a vocabulary of nothing: the matcher is driven by text.
type charVocab struct{}

func (charVocab) Size() int                     { return 1 }
func (charVocab) Piece(id int32) (string, bool) { return "x", true }
func (charVocab) IsEOG(id int32) bool           { return false }

// TestToolSyntaxStreamsCallsAsTheyClose: a reply streamed piece by piece
// shows the text around the calls, never the markup, and hands each call out
// when its markup closes rather than at the end.
func TestToolSyntaxStreamsCallsAsTheyClose(t *testing.T) {
	tools := ParseToolSet([]byte(syntaxTools))
	reply := "Let me check. <tool_call>\n{\"name\": \"get_weather\", \"arguments\": " + syntaxArgs +
		"}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>"
	ts := NewTextToolStream(ToolSyntaxHermes, tools)
	var shown strings.Builder
	var at []int
	var calls []ToolCall
	for i := 0; i < len(reply); i += 3 {
		s, cs := ts.PushText(reply[i:min(i+3, len(reply))])
		shown.WriteString(s)
		for range cs {
			at = append(at, i)
		}
		calls = append(calls, cs...)
	}
	tail, rest, content := ts.Finish()
	shown.WriteString(tail)
	calls = append(calls, rest...)
	if strings.TrimSpace(shown.String()) != "Let me check." || content != "Let me check." {
		t.Fatalf("shown %q, content %q", shown.String(), content)
	}
	if len(calls) != 2 || calls[0].Name != "get_weather" || calls[1].Name != "get_time" ||
		calls[1].Arguments != `{"zone":"CET"}` {
		t.Fatalf("calls %+v", calls)
	}
	if len(at) != 2 || at[0] >= strings.Index(reply, "get_time") {
		t.Fatalf("the first call came out at byte %v, not when its markup closed", at)
	}
}

// TestToolChoiceReachesTheTemplate: a template that reads tool_choice
// (Kimi-K3's) is told "required" and "none"; one that does not renders the
// same either way, and no choice renders as before.
func TestToolChoiceReachesTheTemplate(t *testing.T) {
	user := []ChatMessage{{Role: "user", Content: "Weather in Zürich?"}}
	render := func(file string, c ToolChoice) string {
		src := readSyntaxTemplate(t, file)
		tpl, err := (&Model{}).compiledTemplate(src)
		if err != nil {
			t.Fatal(err)
		}
		out, err := renderCompiled(tpl, src, "", "", nil, user, []byte(syntaxTools), c, true, false, "")
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		return out
	}
	auto := render("Kimi-K3.jinja", ToolChoice{})
	req := render("Kimi-K3.jinja", ToolChoice{Mode: ToolChoiceRequired})
	none := render("Kimi-K3.jinja", ToolChoice{Mode: ToolChoiceNone})
	if strings.Contains(auto, "MUST") || !strings.Contains(req, "You MUST call tools") ||
		!strings.Contains(none, "You MUST NOT call any tools") {
		t.Fatalf("Kimi-K3 did not read tool_choice:\nauto %q\nrequired %q\nnone %q", tail(auto), tail(req), tail(none))
	}
	if a, r := render("qwen3.jinja", ToolChoice{}), render("qwen3.jinja", ToolChoice{Mode: ToolChoiceRequired}); a != r {
		t.Fatalf("a template that does not read tool_choice rendered it")
	}
}

func tail(s string) string { return s[max(0, len(s)-160):] }
