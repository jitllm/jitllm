package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/convert"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestParseToolCalls covers the two call shapes and the cases that must stay
// text: an undeclared name (a model asked to answer IN JSON), prose, and a
// malformed call.
func TestParseToolCalls(t *testing.T) {
	names := []string{"get_weather", "get_time"}
	w := model.ToolCall{Name: "get_weather", Arguments: `{"location":"Paris","format":"celsius"}`}
	tm := model.ToolCall{Name: "get_time", Arguments: `{}`}
	for _, c := range []struct {
		name, in, content string
		calls             []model.ToolCall
	}{
		{"hermes", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\", \"format\": \"celsius\"}}\n</tool_call>", "", []model.ToolCall{w}},
		{"hermes two, prose first", "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\", \"format\": \"celsius\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>", "Let me check.", []model.ToolCall{w, tm}},
		{"hermes doubled braces (Qwen2.5 copies its template)", "<tool_call>\n{{\"name\": \"get_time\", \"arguments\": {}}}\n</tool_call>", "", []model.ToolCall{tm}},
		{"hermes doubled opener only (Qwen2.5-1.5B, measured)", "<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\", \"format\": \"celsius\"}}\n</tool_call>", "", []model.ToolCall{w}},
		{"hermes unclosed at EOS", "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}", "", []model.ToolCall{tm}},
		{"hermes arguments as a string", `<tool_call>{"name": "get_weather", "arguments": "{\"location\": \"Paris\", \"format\": \"celsius\"}"}</tool_call>`, "", []model.ToolCall{w}},
		{"llama3 parameters", `{"name": "get_weather", "parameters": {"location": "Paris", "format": "celsius"}}`, "", []model.ToolCall{w}},
		{"llama3 parallel with ;", `{"name": "get_time", "parameters": {}}; {"name": "get_weather", "parameters": {"location": "Paris", "format": "celsius"}}`, "", []model.ToolCall{tm, w}},
		{"mistral array", `[{"name": "get_weather", "arguments": {"location": "Paris", "format": "celsius"}}]`, "", []model.ToolCall{w}},
		{"undeclared name stays text", `{"name": "Paris", "population": 2100000}`, `{"name": "Paris", "population": 2100000}`, nil},
		{"prose", "The weather in Paris is mild.", "The weather in Paris is mild.", nil},
		{"broken hermes stays text", "<tool_call>{\"name\": \"get_time\", </tool_call>", "<tool_call>{\"name\": \"get_time\", </tool_call>", nil},
	} {
		content, calls := hermesForm().Parse(c.in, model.ToolSetOf(names))
		if content != c.content || !reflect.DeepEqual(calls, c.calls) {
			t.Errorf("%s: got %q %+v, want %q %+v", c.name, content, calls, c.content, c.calls)
		}
	}
}

// TestToolCallHold: text before a call streams, the call is held, and a
// partial opener at the end is held until it resolves.
func TestToolCallHold(t *testing.T) {
	for in, want := range map[string]int{
		"The weather":            -1,
		"Sure. <tool_call>{":     6,
		"Sure. <tool_":           6,
		"  {\"name\"":            2,
		"":                       0,
		"Answer: {\"a\": 1}":     -1,
		"Checking <tool_call>\n": 9,
	} {
		if got := hermesForm().Hold(in); got != want {
			t.Errorf("%q: hold at %d, want %d", in, got, want)
		}
	}
}

// toolModels are the containers whose templates the render gate holds to
// HuggingFace's own renderer. Each is one family's tool format.
func toolModels() []string {
	return []string{
		"qwen2.5-1.5b-instruct-q4_k_m",    // Hermes <tool_call>, tools in the system turn
		"Qwen3-0.6B-Q8_0",                 // Hermes, `arguments is string` branch
		"Llama-3.2-1B-Instruct-Q4_K_M",    // JSON in the first user turn, ipython role
		"Mistral-7B-Instruct-v0.3-Q4_K_M", // this GGUF's template has NO tools: refused
	}
}

// toolFixture is the conversation every template renders: a tool list whose
// property order is NOT alphabetical (location before format), a call, its
// result, and the generation prompt.
const toolFixtureTools = `[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}, "format": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}]`

func toolFixtureMessages() []model.ChatMessage {
	return []model.ChatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the weather in Paris?"},
		{Role: "assistant", ToolCalls: []model.ToolCall{{ID: "a1b2c3d4e", Name: "get_weather",
			Arguments: `{"location": "Paris", "format": "celsius"}`}}},
		{Role: "tool", ToolCallID: "a1b2c3d4e", Name: "get_weather", Content: `{"temperature": 18, "sky": "cloudy"}`},
	}
}

// TestToolPromptMatchesTransformers renders the fixture through each model's
// own template and holds it to transformers' renderer byte for byte
// (scripts/toolgold.py writes the want files from the templates this dumps
// under JITLLM_TOOLGOLD=dump). A near-miss serialisation still gets calls back,
// just worse ones, so the bar is byte equality.
// toolGoldDate is the date the tool-prompt goldens are rendered at, and
// scripts/toolgold.py's FIXED: Llama 3's own default when no clock is given.
func toolGoldDate() time.Time { return time.Date(2024, time.July, 26, 0, 0, 0, 0, time.UTC) }

func TestToolPromptMatchesTransformers(t *testing.T) {
	dir := filepath.Join("..", "testdata", "golden", "tools")
	dump := os.Getenv("JITLLM_TOOLGOLD") == "dump"
	ran := 0
	for _, name := range toolModels() {
		path, ok := existingModel(testmodels.Path(name + ".gguf"))
		if !ok {
			t.Logf("%s: not on this box", name)
			continue
		}
		// The date the templates print is pinned on both sides (toolGoldDate,
		// scripts/toolgold.py): a golden rendered on one day must match on the
		// next.
		m, err := model.Open(jlmOf(t, path), WithChatClock(func() time.Time { return toolGoldDate() }))
		if err != nil {
			t.Fatal(err)
		}
		if dump {
			src, _ := m.ChatTemplate("")
			bos, eos := m.SpecialText()
			b, err := json.MarshalIndent(map[string]string{"template": src, "bos": bos, "eos": eos}, "", " ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, name+".tpl.json"), b, 0o644); err != nil {
				t.Fatal(err)
			}
			m.Close()
			continue
		}
		if src, _ := m.ChatTemplate(""); !strings.Contains(src, "tools") && !strings.Contains(src, "tool_calls") {
			// A template trained without tools must refuse them, not render the
			// conversation as if none had been declared.
			_, err := m.ChatPromptTools(toolFixtureMessages(), []byte(toolFixtureTools), true)
			m.Close()
			if err == nil || !strings.Contains(err.Error(), "never reads `tools`") {
				t.Errorf("%s: a template without tools rendered a tool request (err %v)", name, err)
			}
			ran++
			continue
		}
		want, err := os.ReadFile(filepath.Join(dir, name+".want.txt"))
		if err != nil {
			m.Close()
			t.Fatalf("%s: %v -- run JITLLM_TOOLGOLD=dump, then scripts/toolgold.py (RULE 11)", name, err)
		}
		got, err := m.ChatPromptTools(toolFixtureMessages(), []byte(toolFixtureTools), true)
		m.Close()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got != string(want) {
			i := 0
			for i < len(got) && i < len(want) && got[i] == want[i] {
				i++
			}
			t.Errorf("%s: parts from transformers at byte %d:\n got  %q\n want %q", name, i,
				clip(got[i:]), clip(string(want[i:])))
			continue
		}
		if !strings.Contains(got, "get_weather") || !strings.Contains(got, "cloudy") {
			t.Errorf("%s: the prompt carries neither the tool nor its result -- the fixture did not reach the template", name)
		}
		ran++
	}
	if !dump && ran == 0 {
		testmodels.Missing(t, "%s", "no tool-calling model on this box (set JITLLM_MODELS)")
	}
}

func clip(s string) string {
	if len(s) > 120 {
		return s[:120]
	}
	return s
}

// hermesForm is the default form a reply with no model of its own is read in.
func hermesForm() model.ToolSyntax { return model.ToolSyntaxFor(convert.DefaultToolForm()) }
