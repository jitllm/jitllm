package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// A request picks between named templates as transformers' get_chat_template
// does: tools take "tool_use" when there is one, anything else takes "default",
// a single unnamed template serves every request, and several with no default
// are refused. A conversation carrying a video takes "video" where the model
// has one, as a processor that keeps its own template for one does.
func TestARequestPicksItsTemplateAsTransformersDoes(t *testing.T) {
	one := []jlm.ChatTemplate{{Body: "only"}}
	named := []jlm.ChatTemplate{{Name: "default", Body: "D"}, {Name: "rag", Body: "R"}, {Name: "tool_use", Body: "T"}}
	noTool := []jlm.ChatTemplate{{Name: "default", Body: "D"}, {Name: "rag", Body: "R"}}
	noDefault := []jlm.ChatTemplate{{Name: "rag", Body: "R"}, {Name: "tool_use", Body: "T"}}
	video := []jlm.ChatTemplate{{Name: "default", Body: "D"}, {Name: "video", Body: "V"}}
	for _, c := range []struct {
		name         string
		ts           []jlm.ChatTemplate
		tools, video bool
		want         string // "" is a refusal
	}{
		{"one, no tools", one, false, false, "only"},
		{"one, tools", one, true, false, "only"},
		{"named, no tools", named, false, false, "D"},
		{"named, tools", named, true, false, "T"},
		{"no tool_use, tools", noTool, true, false, "D"},
		{"no default, tools", noDefault, true, false, "T"},
		{"no default, no tools", noDefault, false, false, ""},
		{"none", nil, false, false, ""},
		{"video, a video", video, false, true, "V"},
		{"video, none", video, false, false, "D"},
		{"no video template, a video", named, false, true, "D"},
	} {
		got, err := model.PickChatTemplate(c.ts, c.tools, c.video)
		switch {
		case c.want == "" && err == nil:
			t.Errorf("%s: picked %q, want a refusal", c.name, got)
		case c.want != "" && (err != nil || got != c.want):
			t.Errorf("%s: picked %q (%v), want %q", c.name, got, err, c.want)
		}
	}
}

// synth-commandr was written by llama.cpp's converter from Command-R's named
// list -- default, rag and tool_use -- and a request with tools renders with
// tool_use, one without with default. Command-R's default template never reads
// `tools`, so rendering a tools request with it is refused outright.
func TestAToolsRequestRendersWithToolUse(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("synth-commandr.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var names []string
	for _, x := range m.Container.Vocab().Templates {
		names = append(names, x.Name)
	}
	if strings.Join(names, ",") != "default,rag,tool_use" {
		t.Fatalf("synth-commandr carries %v, not default, rag and tool_use: this gate proves nothing", names)
	}
	def, _ := m.ChatTemplate("default")
	tu, _ := m.ChatTemplate("tool_use")
	if got, err := m.ChatTemplateFor(true, false); err != nil || got != tu {
		t.Errorf("with tools: picked the wrong template (%v)", err)
	}
	if got, err := m.ChatTemplateFor(false, false); err != nil || got != def {
		t.Errorf("without tools: picked the wrong template (%v)", err)
	}
	msgs := []model.ChatMessage{{Role: "user", Content: "weather in Paris?"}}
	// Command-R's tool_use template reads Cohere's own tool shape.
	tools := []byte(`[{"name": "get_weather", "description": "the weather in a city",
		"parameter_definitions": {"city": {"description": "the city", "type": "str", "required": true}}}]`)
	with, err := m.ChatPromptTools(msgs, tools, true)
	if err != nil {
		t.Fatalf("with tools: %v", err)
	}
	if !strings.Contains(with, "get_weather") {
		t.Errorf("the tools request does not name its tool:\n%s", with)
	}
	without, err := m.ChatPrompt(msgs, true)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(without, "get_weather") || without == with {
		t.Errorf("the plain request rendered the tools prompt:\n%s", without)
	}
	t.Logf("with tools %d bytes, without %d", len(with), len(without))
}
