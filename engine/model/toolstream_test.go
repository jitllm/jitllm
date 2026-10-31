package model

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestToolStreamReadsControlTokens: a call written in control tokens, fed as
// the model's own ids, is read from the raw text -- which the decoded text a
// person reads cannot show -- and the text around it streams without its
// markup. DeepSeek's tool tokens, and harmony, whose call sits in a message
// header the chat text drops.
func TestToolStreamReadsControlTokens(t *testing.T) {
	tools := ParseToolSet([]byte(`[{"type":"function","function":{"name":"get_weather","parameters":` +
		`{"type":"object","properties":{"location":{"type":"string"}}}}}]`))
	for _, c := range []struct {
		file, reply, marker, shown string
		syn                        ToolSyntax
	}{
		{"DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.jlm",
			"<think>\nThe user wants Paris.\n</think>\n\nChecking.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function" +
				"<｜tool▁sep｜>get_weather\n```json\n{\"location\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			"", "<think>\nThe user wants Paris.\n</think>\n\nChecking.", ToolSyntaxDeepSeekV3},
		{"gpt-oss-20b-Q4_K_M.jlm",
			"<|channel|>analysis<|message|>The user wants Paris.<|end|><|start|>assistant<|channel|>commentary " +
				"to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"Paris\"}",
			"<|channel|>", "<think>The user wants Paris.</think>", ToolSyntaxHarmony},
		// Mistral v0.3's [TOOL_CALLS] is a control token. (This GGUF's
		// template declares no tools, so the model's syntax is the
		// default, which reads the opener spelled out.)
		{"Mistral-7B-Instruct-v0.3-Q4_K_M.jlm",
			"[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]",
			"[TOOL_CALLS]", "", ToolSyntaxHermes},
	} {
		t.Run(c.file, func(t *testing.T) {
			path := testmodels.Path(c.file)
			if _, err := os.Stat(path); err != nil {
				// The suite converts what ships: the source beside it.
				src := strings.TrimSuffix(path, ".jlm") + ".gguf"
				if _, serr := os.Stat(src); serr != nil {
					testmodels.Missing(t, "%s: %v (set JITLLM_MODELS)", c.file, err)
				}
				path = jlmOf(t, src)
			}
			m, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if got := m.ToolSyntax(); got != c.syn {
				t.Fatalf("syntax %s, want %s", got, c.syn)
			}
			ids := m.Vocab.EncodeSpecial(c.reply, false)
			if c.marker != "" {
				// The case exercises a control token: one id the decoded
				// text drops.
				id, ok := m.Vocab.ID(c.marker)
				if !ok || !slices.Contains(ids, id) {
					t.Fatalf("%q is not one token of the reply", c.marker)
				}
				if p, plain := m.Vocab.Piece(id); plain {
					t.Fatalf("%q decodes to %q: the case does not exercise a control token", c.marker, p)
				}
			}
			ts := m.NewToolStream(tools)
			var shown strings.Builder
			var calls []ToolCall
			for _, id := range ids {
				s, cs := ts.Push(id)
				shown.WriteString(s)
				calls = append(calls, cs...)
			}
			tail, rest, content := ts.Finish()
			shown.WriteString(tail)
			calls = append(calls, rest...)
			if len(calls) != 1 || calls[0].Name != "get_weather" || calls[0].Arguments != `{"location":"Paris"}` {
				t.Fatalf("calls %+v from %q", calls, c.reply)
			}
			if shown.String() != c.shown || content != strings.TrimSpace(c.shown) {
				t.Fatalf("shown %q content %q, want %q", shown.String(), content, c.shown)
			}
		})
	}
}
