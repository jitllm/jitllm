package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/tok/jinja"
)

// chatGold is tok/jinja's testdata: one chat template per supported family,
// the conversations they are rendered over, and transformers' renderings.
func chatGold() string { return filepath.Join("..", "tok", "jinja", "testdata") }

// TestChatPromptMatchesTransformers is tok/jinja's gate one level up: the
// same templates and conversations, rendered through renderChatTemplate from
// ChatMessages -- the context this package builds from a request, not a
// context handed to it -- and held byte for byte to apply_chat_template's
// rendering (render_jinja_template, recorded by scripts/chatgold.py).
//
// That covers what tok/jinja's gate cannot: tools and documents bound None
// when a request carries none, as apply_chat_template binds them; a tool
// call's arguments handed as a mapping, or as the request's JSON text to a
// template that only reads text (DeepSeek-V3.2); and the request's floats.
// Two of this package's choices part from transformers on purpose and are
// counted rather than compared: a system turn folded into the next turn where
// the template has no system role (foldSystem, llama.cpp's choice), and tools
// refused for a template that never reads them. The image case is ChatSpans'
// and the documents case has no field in a request, so neither runs here.
func TestChatPromptMatchesTransformers(t *testing.T) {
	read := func(name string, v any) {
		b, err := os.ReadFile(filepath.Join(chatGold(), name))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(b, v); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	var manifest []struct {
		File      string `json:"file"`
		Tools     string `json:"tools"`
		Arguments string `json:"arguments"`
		Content   string `json:"content"`
		BOS       string `json:"bos"`
		EOS       string `json:"eos"`
	}
	read(filepath.Join("templates", "manifest.json"), &manifest)
	var conv struct {
		Arguments json.RawMessage            `json:"arguments"`
		Tools     map[string]json.RawMessage `json:"tools"`
		Cases     []struct {
			Name     string          `json:"name"`
			Gen      bool            `json:"add_generation_prompt"`
			Tools    bool            `json:"tools"`
			Image    bool            `json:"image"`
			Docs     bool            `json:"documents"`
			Messages json.RawMessage `json:"messages"`
		} `json:"cases"`
	}
	read("conversations.json", &conv)
	clock := func() time.Time { return time.Date(2024, 7, 26, 0, 0, 0, 0, time.UTC) }

	compared, raised, folded, refused, stringArgs := 0, 0, 0, 0, 0
	for _, e := range manifest {
		srcB, err := os.ReadFile(filepath.Join(chatGold(), "templates", e.File))
		if err != nil {
			t.Fatal(err)
		}
		src := string(srcB)
		var golden map[string]struct {
			Out   string `json:"out"`
			Raise string `json:"raise"`
		}
		read(filepath.Join("golden", strings.TrimSuffix(e.File, ".jinja")+".json"), &golden)
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%s: %v", e.File, err)
		}
		// The goldens give a parts-only template its text as typed parts,
		// which is what the probe must choose for it, and only for it.
		parts := model.WantsContentParts(tpl)
		if parts != (e.Content == "parts") {
			t.Errorf("%s: wantsContentParts says %v; the manifest says content %q", e.File, parts, e.Content)
		}
		for _, c := range conv.Cases {
			if c.Image || c.Docs {
				continue
			}
			want, ok := golden[c.Name]
			if !ok {
				t.Fatalf("%s/%s: no golden", e.File, c.Name)
			}
			msgs := chatMessagesOf(t, strings.ReplaceAll(string(c.Messages), `"$ARGS"`, string(conv.Arguments)))
			var tools []byte
			if c.Tools {
				shape := "openai"
				if e.Tools != "" {
					shape = e.Tools
				}
				tools = conv.Tools[shape]
			}
			got, err := model.RenderChatTemplate(src, e.BOS, e.EOS, clock, msgs, tools, c.Gen)
			name := e.File + "/" + c.Name
			probe := map[string]any{"tools": nil, "documents": nil, "bos_token": e.BOS, "eos_token": e.EOS}
			if tools != nil {
				tv, terr := jinja.FromJSON(tools)
				if terr != nil {
					t.Fatal(terr)
				}
				probe["tools"] = tv
			}
			fold := msgs[0].Role == "system" && !model.SystemRendered(tpl, probe, parts)
			switch {
			case tools != nil && !strings.Contains(src, "tools") && !strings.Contains(src, "tool_calls"):
				refused++
				if err == nil || !strings.Contains(err.Error(), "never reads `tools`") {
					t.Errorf("%s: a template that never reads tools rendered (%v)", name, err)
				}
			case want.Raise != "" && err != nil:
				raised++
			case fold:
				// transformers renders the system turn or refuses it; this
				// package folds it into the next turn and must render.
				folded++
				if err != nil || got == "" {
					t.Errorf("%s: the system turn should fold into the next one: %v", name, err)
				}
			case want.Raise != "":
				t.Errorf("%s: transformers raises %q and this rendered %d bytes", name, want.Raise, len(got))
			case err != nil:
				t.Errorf("%s: %v", name, err)
			case got != want.Out:
				i := 0
				for i < len(got) && i < len(want.Out) && got[i] == want.Out[i] {
					i++
				}
				t.Errorf("%s: diverges from transformers at byte %d\n  got:  %q\n  want: %q", name, i,
					got[max(i-60, 0):min(i+60, len(got))], want.Out[max(i-60, 0):min(i+60, len(want.Out))])
			default:
				compared++
				if e.Arguments == "string" && model.CarriesToolCalls(msgs) {
					stringArgs++
				}
			}
		}
	}
	// The DeepSeek-V3.2 row is what proves the string fallback ran.
	if compared < 250 || folded == 0 || refused == 0 || stringArgs == 0 {
		t.Fatalf("%d renderings matched, %d folded, %d refused, %d with string arguments: the gate is not reaching what it covers",
			compared, folded, refused, stringArgs)
	}
	t.Logf("%d renderings byte-identical to transformers, %d raising in both, %d with the system folded, "+
		"%d with tools refused, %d with string arguments", compared, raised, folded, refused, stringArgs)
}

// chatMessagesOf reads OpenAI-shaped messages into ChatMessages, a tool
// call's arguments kept as the JSON text the request carried.
func chatMessagesOf(t *testing.T, raw string) []model.ChatMessage {
	var in []struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Function struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
		ToolCallID string `json:"tool_call_id"`
		Name       string `json:"name"`
	}
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		t.Fatal(err)
	}
	out := make([]model.ChatMessage, len(in))
	for i, m := range in {
		out[i] = model.ChatMessage{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID, Name: m.Name}
		for _, c := range m.ToolCalls {
			out[i].ToolCalls = append(out[i].ToolCalls, model.ToolCall{ID: c.ID, Name: c.Function.Name,
				Arguments: string(c.Function.Arguments)})
		}
	}
	return out
}
