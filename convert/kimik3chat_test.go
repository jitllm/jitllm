package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/tok/jinja"
)

type kimiK3ChatCase struct {
	Name                string  `json:"name"`
	Messages            []any   `json:"messages"`
	AddGenerationPrompt bool    `json:"add_generation_prompt"`
	Thinking            *bool   `json:"thinking"`
	ThinkingEffort      *string `json:"thinking_effort"`
	Text                string  `json:"text"`
}

func kimiK3ChatCases(t *testing.T) []kimiK3ChatCase {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "kimik3_chat.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cs []kimiK3ChatCase
	if err := json.Unmarshal(b, &cs); err != nil {
		t.Fatal(err)
	}
	if len(cs) == 0 {
		t.Fatal("no cases in the golden")
	}
	return cs
}

// renderKimiK3 renders c through tpl in the context model/ binds a chat with,
// thinking and thinking_effort set where the case names them, as
// apply_chat_template takes them.
func renderKimiK3(tpl *jinja.Template, c kimiK3ChatCase) (string, error) {
	ctx := map[string]any{
		"messages":              c.Messages,
		"tools":                 nil,
		"documents":             nil,
		"add_generation_prompt": c.AddGenerationPrompt,
		"bos_token":             "",
		"eos_token":             "",
	}
	if c.Thinking != nil {
		ctx["thinking"] = *c.Thinking
	}
	if c.ThinkingEffort != nil {
		ctx["thinking_effort"] = *c.ThinkingEffort
	}
	return tpl.Render(ctx)
}

// TestKimiK3ChatTemplateMatchesItsRenderer renders every case of
// testdata/golden/kimik3_chat.json -- the release's own encoding_k3.py
// rendering them (scripts/kimik3chatgold.py) -- through the class's
// transcription and demands the renderer's text byte for byte. A case of
// plain string turns and the defaults goes through model.RenderChatTemplate
// too, the path a chat request takes. The transcription with one character
// changed must part from the golden (RULE 10).
func TestKimiK3ChatTemplateMatchesItsRenderer(t *testing.T) {
	tpl, err := jinja.Compile(kimiK3ChatTemplate)
	if err != nil {
		t.Fatal(err)
	}
	cases := kimiK3ChatCases(t)
	engine := 0
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			got, err := renderKimiK3(tpl, c)
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Text {
				t.Fatalf("the transcription renders\n%q\nthe release's renderer\n%q", got, c.Text)
			}
			msgs, plain := []model.ChatMessage{}, c.Thinking == nil && c.ThinkingEffort == nil
			for _, m := range c.Messages {
				mm := m.(map[string]any)
				content, ok := mm["content"].(string)
				plain = plain && ok && len(mm) == 2
				msgs = append(msgs, model.ChatMessage{Role: mm["role"].(string), Content: content})
			}
			if !plain {
				return
			}
			engine++
			out, err := model.RenderChatTemplate(kimiK3ChatTemplate, "", "", nil, msgs, nil, c.AddGenerationPrompt)
			if err != nil {
				t.Fatal(err)
			}
			if out != c.Text {
				t.Fatalf("through model.RenderChatTemplate\n%q\nthe release's renderer\n%q", out, c.Text)
			}
		})
	}
	if engine == 0 {
		t.Fatal("no case went through model.RenderChatTemplate")
	}
	t.Logf("%d cases match the release's renderer, %d of them through model.RenderChatTemplate", len(cases), engine)

	bad, err := jinja.Compile(strings.Replace(kimiK3ChatTemplate, "<|close|>think<|sep|>", "<|close|>think <|sep|>", 1))
	if err != nil {
		t.Fatal(err)
	}
	parted := 0
	for _, c := range cases {
		if got, err := renderKimiK3(bad, c); err != nil || got != c.Text {
			parted++
		}
	}
	if parted == 0 {
		t.Fatal("a transcription with a stray space in the think channel's close matches every case: the gate cannot see it")
	}
}

// TestKimiK3ConversionStatesItsChatTemplate converts Kimi-K3 releases, which
// carry no chat_template, and demands the class's transcription in the
// container's vocabulary: without it the converter calls the model a base
// model and a chat request has nothing to render.
func TestKimiK3ConversionStatesItsChatTemplate(t *testing.T) {
	for _, dir := range []string{filepath.Join(c6HFDir(), "synth-kimik3"), testmodels.Path("kimik3/Kimi-K3-0.40B-hf")} {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (docs/testing.md names where the Kimi-K3 checkpoints "+
					"come from) -- this gate proved nothing", err)
			}
			s := hfSource(t, dir)
			if len(s.Vocab.Templates) != 1 || s.Vocab.Templates[0].Body != kimiK3ChatTemplate {
				t.Fatalf("the vocabulary states %d chat templates, not the class's transcription", len(s.Vocab.Templates))
			}
		})
	}
}
