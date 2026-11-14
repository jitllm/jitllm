package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
)

// Models whose chat template reads no tools are served llama.cpp's generic
// tool format: the system message names the tools and an example call, the
// reply is held to {"tool_call": ...} or {"response": ...}, and the reply is
// read back as OpenAI tool calls or content. testdata/golden/generic holds
// each gated request and what llama.cpp (b4689, the last build with the
// format; scripts/genericgold.py) rendered and answered with for it on each
// model.
func genericModels() []string {
	return []string{"gemma-2-2b-it-Q4_K_M", "Qwen2-1.5B-Instruct-Q4_K_M", "Mistral-7B-Instruct-v0.3-Q4_K_M"}
}

type genericGolden struct {
	Content   string `json:"content"`
	ToolCalls []struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"tool_calls"`
}

// compact is JSON without its whitespace.
func compact(t *testing.T, s string) string {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("%s: %v", s, err)
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// TestGenericToolsMatchLlamaCpp: on a model whose template reads no tools,
// each request renders the prompt llama.cpp renders for it and is answered as
// llama.cpp answers -- the same tool call, arguments and all, or the same
// content -- through the server's own chat completions, on the CPU and on a
// card. The violation is the system message left out of the prompt, or the
// reply left unconstrained.
func TestGenericToolsMatchLlamaCpp(t *testing.T) {
	dir := filepath.Join("testdata", "golden", "generic")
	reqs, err := filepath.Glob(filepath.Join(dir, "*.request.json"))
	if err != nil || len(reqs) < 5 {
		t.Fatalf("%d requests in %s (%v): the gate covers nothing", len(reqs), dir, err)
	}
	for _, stem := range genericModels() {
		for _, arm := range []string{"cpu", "card"} {
			t.Run(stem+"/"+arm, func(t *testing.T) {
				devs := []string{"cpu"}
				if arm == "card" {
					if !testdevices.Selected() {
						t.Skip("JITLLM_TEST_DEVICES selects no card: the card arm runs where one is given")
					}
					devs = testdevices.Take(t, 1, "cuda:0")
				}
				_, lm, c := loadedEngine(t, stem+".jlm", "m", LoadOptions{DeviceIDs: devs})
				syn, err := lm.m.ToolSyntax()
				if err != nil || !syn.Generic() {
					t.Fatalf("%s: tool syntax %v (%v), want the generic form: the gate proves nothing", stem, syn, err)
				}
				bos, eos := lm.m.SpecialText()
				before := lm.grammars.toolForced.Load()
				ran, skipped := 0, 0
				for _, rf := range reqs {
					name := strings.TrimSuffix(filepath.Base(rf), ".request.json")
					t.Run(name, func(t *testing.T) {
						body, err := os.ReadFile(rf)
						if err != nil {
							t.Fatal(err)
						}
						wantPrompt, err := os.ReadFile(filepath.Join(dir, name+"."+stem+".prompt.txt"))
						if err != nil {
							t.Fatalf("%s: no golden (%v): record it with scripts/genericgold.py", name, err)
						}
						var want genericGolden
						gb, err := os.ReadFile(filepath.Join(dir, name+"."+stem+".response.json"))
						if err != nil || json.Unmarshal(gb, &want) != nil {
							t.Fatalf("%s: no golden answer (%v)", name, err)
						}

						// The prompt: the request the server read, rendered by the model.
						f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
						r := post(t, serve(t, f), "/v1/chat/completions", string(body))
						r.Body.Close()
						ch := f.opts().Prompt.Chat
						prompt, err := lm.m.ChatPromptToolChoice(ch.Messages, ch.Tools, ch.ToolChoice, ch.AddGenerationPrompt)
						if err != nil {
							t.Fatalf("%s: %v", name, err)
						}
						// llama.cpp hands its template an empty bos_token (the tokenizer adds it)
						// and an empty eos_token, where the chat path renders both as the template
						// writes them: the prompts are equal past those two.
						got := strings.TrimPrefix(prompt, bos)
						if strings.ReplaceAll(got, eos, "") != strings.ReplaceAll(string(wantPrompt), eos, "") {
							i := 0
							for i < len(got) && i < len(wantPrompt) && got[i] == wantPrompt[i] {
								i++
							}
							t.Fatalf("%s: prompt parts from llama.cpp's at byte %d:\n got  %q\n want %q", name, i,
								got[i:min(len(got), i+80)], string(wantPrompt[i:min(len(wantPrompt), i+80)]))
						}
						// The answer, from the same tokens: a conversation that ends a turn with
						// the eos token is not the one llama.cpp answered.
						if got != string(wantPrompt) {
							t.Logf("%s: the prompt writes %q where llama.cpp's does not: the answer is not compared", name, eos)
							skipped++
							return
						}
						status, b := postRaw(t, c.url+"/v1/chat/completions", string(body))
						var out oaChatResponse
						if status != 200 || json.Unmarshal([]byte(b), &out) != nil || len(out.Choices) != 1 {
							t.Fatalf("%s: %d %s", name, status, b)
						}
						msg := out.Choices[0].Message
						if len(msg.ToolCalls) != len(want.ToolCalls) {
							t.Fatalf("%s: calls %+v content %q, llama.cpp answered %+v content %q", name, msg.ToolCalls,
								msg.Content, want.ToolCalls, want.Content)
						}
						for i, tc := range msg.ToolCalls {
							w := want.ToolCalls[i]
							if tc.Function.Name != w.Name || compact(t, tc.Function.Arguments) != compact(t, string(w.Arguments)) {
								t.Fatalf("%s: call %s(%s), llama.cpp answered %s(%s)", name, tc.Function.Name,
									tc.Function.Arguments, w.Name, w.Arguments)
							}
						}
						if len(want.ToolCalls) > 0 && *out.Choices[0].FinishReason != "tool_calls" {
							t.Fatalf("%s: finish %q for a call", name, *out.Choices[0].FinishReason)
						}
						// Words are the CPU's to match; a card's rounding may part them at
						// a near tie (RULE 11c), so there the answer must be an answer.
						if arm == "cpu" && strings.TrimSpace(msg.Content) != strings.TrimSpace(want.Content) ||
							(strings.TrimSpace(msg.Content) == "") != (strings.TrimSpace(want.Content) == "") {
							t.Fatalf("%s: content %q, llama.cpp answered %q", name, msg.Content, want.Content)
						}
						t.Logf("%s: %d calls, content %q", name, len(msg.ToolCalls), msg.Content)
						ran++
					})
				}
				if ran+skipped != len(reqs) || skipped > 1 || lm.grammars.toolForced.Load() == before {
					t.Fatalf("ran %d and skipped %d of %d requests with %d constrained replies", ran, skipped, len(reqs),
						lm.grammars.toolForced.Load()-before)
				}
			})
		}
	}
}

// TestGenericToolsStreamAndForbid: the same replies stream (a call as it
// closes, an answer as it is written), the Anthropic route returns the call as
// a tool_use block, and tool_choice none is a plain conversation that is told
// nothing of its tools.
func TestGenericToolsStreamAndForbid(t *testing.T) {
	dir := filepath.Join("testdata", "golden", "generic")
	_, lm, c := loadedEngine(t, genericModels()[1]+".jlm", "m", LoadOptions{DeviceIDs: []string{"cpu"}})
	read := func(name string) string {
		b, err := os.ReadFile(filepath.Join(dir, name+".request.json"))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	streamed := func(body string) (string, []oaToolCall, string) {
		t.Helper()
		body = strings.Replace(body, `"temperature"`, `"stream":true,"temperature"`, 1)
		resp, err := http.Post(c.url+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		fr := newFrameReader(resp.Body)
		var text strings.Builder
		var calls []oaToolCall
		finish := ""
		for {
			x, ok := fr.next()
			if !ok || x.data == "[DONE]" {
				break
			}
			var ch oaChatResponse
			if err := json.Unmarshal([]byte(x.data), &ch); err != nil {
				t.Fatalf("%s: %v", x.data, err)
			}
			if d := ch.Choices[0].Delta; d != nil {
				text.WriteString(d.Content)
				calls = append(calls, d.ToolCalls...)
			}
			if ch.Choices[0].FinishReason != nil {
				finish = *ch.Choices[0].FinishReason
			}
		}
		return text.String(), calls, finish
	}
	plain := func(body string) oaChatResponse {
		t.Helper()
		status, b := postRaw(t, c.url+"/v1/chat/completions", body)
		var out oaChatResponse
		if status != 200 || json.Unmarshal([]byte(b), &out) != nil {
			t.Fatalf("%d %s", status, b)
		}
		return out
	}
	for _, name := range []string{"weather", "result"} {
		body := read(name)
		whole := plain(body).Choices[0].Message
		text, calls, finish := streamed(body)
		if len(calls) != len(whole.ToolCalls) || strings.TrimSpace(text) != strings.TrimSpace(whole.Content) {
			t.Fatalf("%s: streamed %q %+v, whole %q %+v", name, text, calls, whole.Content, whole.ToolCalls)
		}
		for i, tc := range calls {
			if tc.Function.Name != whole.ToolCalls[i].Function.Name || tc.Index == nil || *tc.Index != i ||
				compact(t, tc.Function.Arguments) != compact(t, whole.ToolCalls[i].Function.Arguments) {
				t.Fatalf("%s: streamed call %d %+v, whole %+v", name, i, tc, whole.ToolCalls[i])
			}
		}
		if (len(calls) > 0) != (finish == "tool_calls") {
			t.Fatalf("%s: finish %q with %d calls", name, finish, len(calls))
		}
	}

	// Anthropic: the same tools and questions, as a tool_use block and as text.
	var oa struct {
		Tools []struct {
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal([]byte(read("weather")), &oa); err != nil {
		t.Fatal(err)
	}
	type anTool struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		InputSchema json.RawMessage `json:"input_schema"`
	}
	var ats []anTool
	for _, x := range oa.Tools {
		ats = append(ats, anTool{x.Function.Name, x.Function.Description, x.Function.Parameters})
	}
	tb, _ := json.Marshal(ats)
	an := func(msgs string) anResponse {
		t.Helper()
		body := fmt.Sprintf(`{"model":"m","max_tokens":200,"temperature":0,"messages":%s,"tools":%s}`, msgs, tb)
		status, b := postRaw(t, c.url+"/v1/messages", body)
		var r anResponse
		if status != 200 || json.Unmarshal([]byte(b), &r) != nil || r.StopReason == nil {
			t.Fatalf("anthropic: %d %s", status, b)
		}
		return r
	}
	ask := `{"role":"user","content":"What is the weather in Paris right now?"}`
	ar := an(`[` + ask + `]`)
	used := 0
	for _, blk := range ar.Content {
		if blk.Type == "tool_use" && blk.Name == "get_weather" {
			used++
			validArgs(t, "anthropic", blk.Name, string(blk.Input))
		}
	}
	if *ar.StopReason != "tool_use" || used != 1 {
		t.Fatalf("anthropic: %d get_weather blocks, stop %s", used, *ar.StopReason)
	}
	// A reply that answers is the answer's text, not the JSON it came in.
	ans := an(`[` + ask + `,{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather",` +
		`"input":{"location":"Paris"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1",` +
		`"content":"21 C and sunny"}]}]`)
	if *ans.StopReason != "end_turn" || len(ans.Content) != 1 || ans.Content[0].Type != "text" || ans.Content[0].Text == nil ||
		strings.TrimSpace(*ans.Content[0].Text) == "" || strings.Contains(*ans.Content[0].Text, `"response"`) {
		t.Fatalf("anthropic answer: stop %s, %d blocks", *ans.StopReason, len(ans.Content))
	}
	t.Logf("anthropic: %s(%s), then %q", ar.Content[0].Name, ar.Content[0].Input, *ans.Content[0].Text)

	// Forbidden: the server drops the tools, so the model is told nothing of
	// them and answers in text.
	before := lm.grammars.toolForced.Load()
	none := plain(strings.Replace(read("weather"), `"parallel_tool_calls"`, `"tool_choice":"none","parallel_tool_calls"`, 1))
	if m := none.Choices[0].Message; len(m.ToolCalls) != 0 || strings.TrimSpace(m.Content) == "" ||
		strings.Contains(m.Content, `"tool_call"`) || lm.grammars.toolForced.Load() != before {
		t.Fatalf("tool_choice none: %+v", m)
	}
}
