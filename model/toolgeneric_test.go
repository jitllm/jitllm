package model

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/jitllm/jitllm/grammar"
	"github.com/jitllm/jitllm/tok/jinja"
)

const genericTools = `[{"type":"function","function":{"name":"get_weather","description":"Get the weather",` +
	`"parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string",` +
	`"enum":["celsius","fahrenheit"]}},"required":["location"]}}},{"type":"function","function":{"name":"get_time",` +
	`"parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}}}]`

// noToolsTemplate is a chat template that never names a tool, as most
// instruct models' do.
const noToolsTemplate = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>\n{% endfor %}" +
	"{% if add_generation_prompt %}<|assistant|>{% endif %}"

// noSystemTemplate renders no system turn, as Gemma 2's does not.
const noSystemTemplate = "{% for m in messages %}{% if m.role == 'user' %}<u>{{ m.content }}</u>" +
	"{% elif m.role == 'assistant' %}<a>{{ m.content }}</a>{% else %}{{ raise_exception('role ' + m.role) }}" +
	"{% endif %}{% endfor %}{% if add_generation_prompt %}<a>{% endif %}"

func TestGenericReplyIsRead(t *testing.T) {
	tools := ParseToolSet([]byte(genericTools))
	syn := genericSyntax()
	for _, c := range []struct {
		name, reply, content string
		calls                []ToolCall
	}{
		{"answer", `{"response": "Hello \"there\"\nThat is \u00e9 and \ud83d\ude00."}`,
			"Hello \"there\"\nThat is é and 😀.", nil},
		{"an answer in bytes", "{\"response\": \"café 😀 ok\"}", "café 😀 ok", nil},
		{"one call", `{"tool_call": {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}}`, "",
			[]ToolCall{{Name: "get_weather", Arguments: `{"location":"Paris","unit":"celsius"}`}}},
		{"two calls", `{"tool_calls": [{"name": "get_weather", "arguments": {"location": "Rome"}, "id": "call-a"}, ` +
			`{"name": "get_time", "arguments": {"zone": "CET"}, "id": "call-b"}]}`, "",
			[]ToolCall{{ID: "call-a", Name: "get_weather", Arguments: `{"location":"Rome"}`},
				{ID: "call-b", Name: "get_time", Arguments: `{"zone":"CET"}`}}},
		{"an answer that is not text", `{"response": {"a": [1, 2]}}`, "{\n  \"a\": [\n    1,\n    2\n  ]\n}", nil},
		{"not an object", "I cannot say.", "I cannot say.", nil},
		{"a tool nobody declared", `{"tool_call": {"name": "launch", "arguments": {}}}`,
			`{"tool_call": {"name": "launch", "arguments": {}}}`, nil},
		{"cut off in the answer", `{"response": "It is warm and sun`, "It is warm and sun", nil},
		{"arguments as text", `{"tool_call": {"name": "get_time", "arguments": "{\"zone\": \"UTC\"}"}}`, "",
			[]ToolCall{{Name: "get_time", Arguments: `{"zone":"UTC"}`}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if content, calls := syn.Parse(c.reply, tools); content != c.content || !slices.Equal(calls, c.calls) {
				t.Errorf("Parse: content %q calls %+v, want %q %+v", content, calls, c.content, c.calls)
			}
			// The same reply a byte at a time (a character split across pushes
			// included): what streams, then the tail, is the content, and each
			// call is returned once.
			ts := NewTextToolStream(syn, tools)
			var shown strings.Builder
			var calls []ToolCall
			for i := 0; i < len(c.reply); i++ {
				s, cs := ts.PushText(c.reply[i : i+1])
				if !utf8.ValidString(s) {
					t.Fatalf("a push showed %q, not whole characters", s)
				}
				shown.WriteString(s)
				calls = append(calls, cs...)
			}
			tail, rest, content := ts.Finish()
			shown.WriteString(tail)
			calls = append(calls, rest...)
			if content != c.content || !slices.Equal(calls, c.calls) {
				t.Errorf("stream: content %q calls %+v, want %q %+v", content, calls, c.content, c.calls)
			}
			if got := shown.String(); got != c.content && !(c.calls != nil && got == "") {
				t.Errorf("stream showed %q, want %q", got, c.content)
			}
		})
	}
}

// TestGenericCallsStreamAsTheyClose: the first call of a reply is returned
// when its object closes, before the second is written.
func TestGenericCallsStreamAsTheyClose(t *testing.T) {
	tools := ParseToolSet([]byte(genericTools))
	ts := NewTextToolStream(genericSyntax(), tools)
	_, first := ts.PushText(`{"tool_calls": [{"name": "get_weather", "arguments": {"location": "Rome"}, "id": "aaaa"}, {"name": "get_ti`)
	if len(first) != 1 || first[0].Name != "get_weather" {
		t.Fatalf("calls after the first object: %+v", first)
	}
	_, second := ts.PushText(`me", "arguments": {"zone": "CET"}, "id": "bbbb"}]}`)
	if len(second) != 1 || second[0].Name != "get_time" {
		t.Fatalf("calls after the second: %+v", second)
	}
	if _, rest, _ := ts.Finish(); len(rest) != 0 {
		t.Fatalf("a call returned twice: %+v", rest)
	}
}

// toyTokens is one token per byte.
type toyTokens struct{}

func (toyTokens) Size() int { return 257 }
func (toyTokens) Piece(id int32) (string, bool) {
	if id >= 256 {
		return "", false
	}
	return string([]byte{byte(id)}), true
}
func (toyTokens) IsEOG(id int32) bool { return id == 256 }

func TestGenericGrammarHoldsTheReply(t *testing.T) {
	tools := ParseToolSet([]byte(genericTools))
	names := tools.Names()
	for _, c := range []struct {
		name   string
		choice ToolChoice
		names  []string
		yes    []string
		no     []string
	}{
		{"a call or an answer", ToolChoice{}, names,
			[]string{`{"tool_call": {"name": "get_weather", "arguments": {"location": "Paris"}}}`,
				`{"tool_call": {"name": "get_time", "arguments": {"zone": "CET"}}}`,
				`{"response": "Fine."}`},
			[]string{`Fine.`,
				`{"tool_call": {"name": "launch", "arguments": {}}}`,
				`{"tool_call": {"name": "get_weather", "arguments": {"city": "Paris"}}}`,
				`{"tool_calls": [{"name": "get_time", "arguments": {"zone": "CET"}, "id": "abcd"}]}`}},
		{"a call is required", ToolChoice{Mode: ToolChoiceRequired}, names,
			[]string{`{"tool_call": {"name": "get_time", "arguments": {"zone": "CET"}}}`},
			[]string{`{"response": "Fine."}`}},
		{"the named call", ToolChoice{Mode: ToolChoiceFunction, Name: "get_time"}, []string{"get_time"},
			[]string{`{"tool_call": {"name": "get_time", "arguments": {"zone": "CET"}}}`},
			[]string{`{"tool_call": {"name": "get_weather", "arguments": {"location": "Paris"}}}`, `{"response": "Fine."}`}},
	} {
		t.Run(c.name, func(t *testing.T) {
			src, err := genericGrammar(tools, c.choice, c.names)
			if err != nil {
				t.Fatal(err)
			}
			gr, err := grammar.Parse(src)
			if err != nil {
				t.Fatalf("%v\n%s", err, src)
			}
			m := grammar.NewMatcher(gr, grammar.IndexTokens(toyTokens{}))
			accepts := func(s string) bool {
				st, ok := m.AcceptText(m.Start(), s)
				return ok && m.Done(st)
			}
			for _, s := range c.yes {
				if !accepts(s) {
					t.Errorf("refused %s", s)
				}
			}
			for _, s := range c.no {
				if accepts(s) {
					t.Errorf("accepted %s", s)
				}
			}
		})
	}
}

// genericCallText is the example call a template that writes its turns as
// <|role|>text<|end|> shows the model, as llama.cpp reads it off the template.
const genericCallText = "{\n  \"tool_calls\": [\n    {\n      \"name\": \"tool_name\",\n      \"arguments\": {\n" +
	"        \"arg1\": \"some_value\"\n      },\n      \"id\": \"call_1___\"\n    }\n  ]\n}"

// TestGenericPromptTellsTheModel: a request with tools to a template that
// reads none renders with the tools, the example call and the sentence that
// tells the model how to answer in the system message, folded into the first
// turn where the template has no system role; a request that forbids calls
// renders as a plain conversation; a call and its result are written into
// their turns; and a template that does read tools is rendered as before.
func TestGenericPromptTellsTheModel(t *testing.T) {
	user := []ChatMessage{{Role: "user", Content: "What is the weather in Paris?"}}
	render := func(src string, msgs []ChatMessage, tools string, choice ToolChoice) string {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatal(err)
		}
		var tb []byte
		if tools != "" {
			tb = []byte(tools)
		}
		out, err := RenderCompiled(tpl, src, "", "", nil, msgs, tb, choice, true, false, "")
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	var list bytes.Buffer
	if err := json.Indent(&list, []byte(genericTools), "", "  "); err != nil {
		t.Fatal(err)
	}
	system := func(example string) string {
		return genericSystem + "\n\nYou can call any of the following tools to satisfy the user's requests: " +
			list.String() + "\n\nExample tool call syntax:\n\n" + example + "\n\n"
	}
	plain := render(noToolsTemplate, user, "", ToolChoice{})
	if strings.Contains(plain, genericSystem) {
		t.Fatalf("a request with no tools was told to answer in JSON: %q", plain)
	}
	want := "<|system|>" + system(genericCallText+"<|end|>\n") + "<|end|>\n" + plain
	if got := render(noToolsTemplate, user, genericTools, ToolChoice{}); got != want {
		t.Errorf("tools to a template that reads none:\n got %q\nwant %q", got, want)
	}
	sys := append([]ChatMessage{{Role: "system", Content: "Be brief."}}, user...)
	want = "<|system|>Be brief.\n\n" + system(genericCallText+"<|end|>\n") +
		"<|end|>\n<|user|>What is the weather in Paris?<|end|>\n<|assistant|>"
	if got := render(noToolsTemplate, sys, genericTools, ToolChoice{Mode: ToolChoiceRequired}); got != want {
		t.Errorf("a system turn:\n got %q\nwant %q", got, want)
	}
	if got := render(noToolsTemplate, user, genericTools, ToolChoice{Mode: ToolChoiceNone}); got != plain {
		t.Errorf("tool_choice none:\n got %q\nwant %q", got, plain)
	}
	want = "<u>" + system(genericCallText+"</a>") + "\nWhat is the weather in Paris?</u><a>"
	if got := render(noSystemTemplate, user, genericTools, ToolChoice{}); got != want {
		t.Errorf("a template with no system role:\n got %q\nwant %q", got, want)
	}
	// The conversation after a call, with and without the tools declared: the
	// call is the text of its turn and the result a user turn.
	turns := append(slices.Clone(user),
		ChatMessage{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "get_weather", Arguments: `{"location":"Paris"}`}}},
		ChatMessage{Role: "tool", ToolCallID: "call_1", Name: "get_weather", Content: "21 C"})
	for _, tools := range []string{genericTools, ""} {
		got := render(noSystemTemplate, turns, tools, ToolChoice{})
		for _, w := range []string{
			"<a>{\n  \"tool_calls\": [\n    {\n      \"name\": \"get_weather\",\n      \"arguments\": {\n" +
				"        \"location\": \"Paris\"\n      },\n      \"id\": \"call_1\"\n    }\n  ]\n}</a>",
			"<u>{\n  \"tool_response\": {\n    \"content\": \"21 C\",\n    \"name\": \"get_weather\",\n" +
				"    \"tool_call_id\": \"call_1\"\n  }\n}</u>"} {
			if !strings.Contains(got, w) {
				t.Errorf("tools %v: the conversation after a call lacks %s:\n%s", tools != "", w, got)
			}
		}
		if told := strings.Contains(got, genericSystem); told != (tools != "") {
			t.Errorf("tools %v: told how to answer %v", tools != "", told)
		}
	}
	// A template that reads tools keeps its own form.
	own := "{% if tools %}TOOLS:{{ tools | length }} {% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"
	if got := render(own, user, genericTools, ToolChoice{}); strings.Contains(got, genericSystem) || !strings.HasPrefix(got, "TOOLS:2 ") {
		t.Errorf("a template that reads tools was served the generic form: %q", got)
	}
}
