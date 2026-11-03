package server

import (
	"encoding/json"
	"strings"
	"testing"
)

// A tool call through both shims, both ways: the tokens a Qwen-style model
// emits for a call, split mid-opener so the holdback has to work, then what a
// client of each API receives.

func toolScript() []string {
	return []string{"Let me check. <tool", "_call>\n{\"name\": \"get_weather\", ",
		"\"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"}
}

const oaToolReq = `{"model":"m-1","messages":[` +
	`{"role":"user","content":"weather?"},` +
	`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Rome\"}"}}]},` +
	`{"role":"tool","tool_call_id":"call_1","content":"sunny"}],` +
	`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]%s}`

const anToolReq = `{"model":"m-1","max_tokens":64,"messages":[` +
	`{"role":"user","content":"weather?"},` +
	`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"Rome"}}]},` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}]}],` +
	`"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"location":{"type":"string"}}}}]%s}`

// TestToolRequestsReachTheTemplate: both shims hand the engine the tool list
// in the OpenAI shape, the earlier call and its result as chat turns.
func TestToolRequestsReachTheTemplate(t *testing.T) {
	for _, c := range []struct{ path, body string }{
		{"/v1/chat/completions", strings.Replace(oaToolReq, "%s", "", 1)},
		{"/v1/messages", strings.Replace(anToolReq, "%s", "", 1)},
	} {
		f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
		r := post(t, serve(t, f), c.path, c.body)
		r.Body.Close()
		ch := f.opts().Prompt.Chat
		if ch == nil || !strings.Contains(string(ch.Tools), `"name":"get_weather"`) ||
			!strings.Contains(string(ch.Tools), `"type":"function"`) {
			t.Fatalf("%s: tools did not reach the engine: %+v", c.path, ch)
		}
		ms := ch.Messages
		if len(ms) != 3 || len(ms[1].ToolCalls) != 1 || ms[1].ToolCalls[0].Name != "get_weather" ||
			!strings.Contains(ms[1].ToolCalls[0].Arguments, "Rome") ||
			ms[2].Role != "tool" || ms[2].Content != "sunny" || ms[2].ToolCallID == "" {
			t.Fatalf("%s: turns %+v", c.path, ms)
		}
	}
	// tool_choice none withholds the list on both.
	for _, c := range []struct{ path, body string }{
		{"/v1/chat/completions", strings.Replace(oaToolReq, "%s", `,"tool_choice":"none"`, 1)},
		{"/v1/messages", strings.Replace(anToolReq, "%s", `,"tool_choice":{"type":"none"}`, 1)},
	} {
		f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
		r := post(t, serve(t, f), c.path, c.body)
		r.Body.Close()
		if ch := f.opts().Prompt.Chat; ch == nil || ch.Tools != nil {
			t.Fatalf("%s: tool_choice none still rendered tools", c.path)
		}
	}
}

func TestOpenAIToolCallNonStreaming(t *testing.T) {
	f := &fakeBackend{tokens: toolScript(), reason: FinishEOS}
	r := post(t, serve(t, f), "/v1/chat/completions", strings.Replace(oaToolReq, "%s", "", 1))
	defer r.Body.Close()
	var oa oaChatResponse
	if err := json.NewDecoder(r.Body).Decode(&oa); err != nil {
		t.Fatal(err)
	}
	ch := oa.Choices[0]
	if ch.FinishReason == nil || *ch.FinishReason != "tool_calls" {
		t.Fatalf("finish_reason %v, want tool_calls", ch.FinishReason)
	}
	tc := ch.Message.ToolCalls
	if len(tc) != 1 || tc[0].Function.Name != "get_weather" || tc[0].Function.Arguments != `{"location":"Paris"}` ||
		tc[0].ID == "" || tc[0].Type != "function" || tc[0].Index != nil {
		t.Fatalf("tool_calls %+v", tc)
	}
	if ch.Message.Content != "Let me check." {
		t.Fatalf("content %q", ch.Message.Content)
	}
}

func TestOpenAIToolCallStreams(t *testing.T) {
	f := &fakeBackend{tokens: toolScript(), reason: FinishEOS}
	r := post(t, serve(t, f), "/v1/chat/completions",
		strings.Replace(oaToolReq, "%s", `,"stream":true`, 1))
	defer r.Body.Close()
	fr := newFrameReader(r.Body)
	var text strings.Builder
	var calls []oaToolCall
	var finish string
	for {
		x, ok := fr.next()
		if !ok || x.data == "[DONE]" {
			break
		}
		var c oaChatResponse
		if err := json.Unmarshal([]byte(x.data), &c); err != nil {
			t.Fatal(err)
		}
		d := c.Choices[0].Delta
		if strings.Contains(d.Content, "<tool") || strings.Contains(d.Content, "get_weather") {
			t.Fatalf("the call leaked into content: %q", d.Content)
		}
		text.WriteString(d.Content)
		calls = append(calls, d.ToolCalls...)
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if strings.TrimSpace(text.String()) != "Let me check." || finish != "tool_calls" ||
		len(calls) != 1 || calls[0].Index == nil || *calls[0].Index != 0 ||
		calls[0].Function.Arguments != `{"location":"Paris"}` {
		t.Fatalf("streamed %q, finish %q, calls %+v", text.String(), finish, calls)
	}
}

func TestAnthropicToolUse(t *testing.T) {
	f := &fakeBackend{tokens: toolScript(), reason: FinishEOS}
	r := post(t, serve(t, f), "/v1/messages", strings.Replace(anToolReq, "%s", "", 1))
	defer r.Body.Close()
	var an anResponse
	if err := json.NewDecoder(r.Body).Decode(&an); err != nil {
		t.Fatal(err)
	}
	if an.StopReason == nil || *an.StopReason != "tool_use" || len(an.Content) != 2 ||
		an.Content[0].Type != "text" || *an.Content[0].Text != "Let me check." ||
		an.Content[1].Type != "tool_use" || an.Content[1].Name != "get_weather" ||
		string(an.Content[1].Input) != `{"location":"Paris"}` || an.Content[1].ID == "" {
		t.Fatalf("response %+v", an)
	}

	// Streamed: the text block, then a tool_use block assembled from one
	// input_json_delta.
	f = &fakeBackend{tokens: toolScript(), reason: FinishEOS}
	r2 := post(t, serve(t, f), "/v1/messages", strings.Replace(anToolReq, "%s", `,"stream":true`, 1))
	defer r2.Body.Close()
	fr := newFrameReader(r2.Body)
	var names []string
	var text, js, stop string
	for {
		x, ok := fr.next()
		if !ok {
			break
		}
		names = append(names, x.event)
		var raw map[string]json.RawMessage
		json.Unmarshal([]byte(x.data), &raw)
		var delta map[string]any
		json.Unmarshal(raw["delta"], &delta)
		switch delta["type"] {
		case "text_delta":
			text += delta["text"].(string)
		case "input_json_delta":
			js += delta["partial_json"].(string)
		}
		if s, ok := delta["stop_reason"].(string); ok {
			stop = s
		}
	}
	want := "message_start,content_block_start,content_block_delta,content_block_stop," +
		"content_block_start,content_block_delta,content_block_stop,message_delta,message_stop"
	if strings.Join(names, ",") != want || strings.TrimSpace(text) != "Let me check." ||
		js != `{"location":"Paris"}` || stop != "tool_use" {
		t.Fatalf("events %v text %q json %q stop %q", names, text, js, stop)
	}
}

// Without tools a JSON answer is an answer: the holdback never engages.
func TestNoToolsMeansNoToolParsing(t *testing.T) {
	f := &fakeBackend{tokens: []string{`{"name": "get_weather"}`}, reason: FinishEOS}
	r := post(t, serve(t, f), "/v1/chat/completions",
		`{"model":"m-1","messages":[{"role":"user","content":"json please"}]}`)
	defer r.Body.Close()
	var oa oaChatResponse
	json.NewDecoder(r.Body).Decode(&oa)
	if oa.Choices[0].Message.Content != `{"name": "get_weather"}` || len(oa.Choices[0].Message.ToolCalls) != 0 {
		t.Fatalf("%+v", oa.Choices[0].Message)
	}
}
