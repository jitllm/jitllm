package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// The compat shims' refusal and failure paths. A client library dispatches on
// the status and the error body's type, so each has to be the one that API
// documents -- and a refusal must happen BEFORE the engine is asked for
// anything.

// errBody reads either API's error shape: OpenAI's {"error":{type,message}}
// or Anthropic's {"type":"error","error":{type,message}}.
type errBody struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func readErr(t *testing.T, resp *http.Response) errBody {
	t.Helper()
	defer resp.Body.Close()
	var e errBody
	b, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatalf("the error body is not JSON: %q", b)
	}
	return e
}

func do(t *testing.T, method, url, body string) *http.Response {
	t.Helper()
	r, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

const (
	oaChat = "/v1/chat/completions"
	oaText = "/v1/completions"
	anMsgs = "/v1/messages"
)

// TestTheShimsRefuseWhatTheyCannotParse: every request here is malformed, so
// it is answered with the API's own error shape and the engine never runs.
// With oaContent accepting anything (returning "", nil) the content case
// reaches the engine and this fails.
func TestTheShimsRefuseWhatTheyCannotParse(t *testing.T) {
	for _, tc := range []struct {
		what, method, path, body string
		status                   int
		typ                      string
		says                     string
	}{
		{"GET on chat", "GET", oaChat, "", 405, "invalid_request_error", "POST"},
		{"GET on completions", "GET", oaText, "", 405, "invalid_request_error", "POST"},
		{"GET on messages", "GET", anMsgs, "", 405, "invalid_request_error", "POST"},

		{"chat: not JSON", "POST", oaChat, `{`, 400, "invalid_request_error", ""},
		{"chat: content is a number", "POST", oaChat,
			`{"model":"m-1","messages":[{"role":"user","content":42}]}`, 400, "invalid_request_error", "content must be"},
		{"chat: tools is not a list", "POST", oaChat,
			`{"model":"m-1","tools":{"x":1},"messages":[{"role":"user","content":"hi"}]}`, 400, "invalid_request_error", "tools must be"},
		{"chat: a tool with no name", "POST", oaChat,
			`{"model":"m-1","tools":[{"type":"function","function":{}}],"messages":[{"role":"user","content":"hi"}]}`,
			400, "invalid_request_error", "tools[0]"},
		{"chat: a tool that is not a function", "POST", oaChat,
			`{"model":"m-1","tools":[{"type":"retrieval","function":{"name":"f"}}],"messages":[{"role":"user","content":"hi"}]}`,
			400, "invalid_request_error", "tools[0]"},
		{"chat: no model", "POST", oaChat, `{"messages":[{"role":"user","content":"hi"}]}`, 404, "invalid_request_error", "model"},

		{"completions: not JSON", "POST", oaText, `[`, 400, "invalid_request_error", ""},
		{"completions: prompt is an object", "POST", oaText, `{"model":"m-1","prompt":{"a":1}}`,
			400, "invalid_request_error", "prompt must be"},
		{"completions: no model", "POST", oaText, `{"prompt":"hi"}`, 404, "invalid_request_error", "model"},

		{"messages: not JSON", "POST", anMsgs, `nope`, 400, "invalid_request_error", ""},
		{"messages: system is a number", "POST", anMsgs,
			`{"model":"m-1","max_tokens":8,"system":7,"messages":[{"role":"user","content":"hi"}]}`,
			400, "invalid_request_error", "content must be"},
		{"messages: content is a number", "POST", anMsgs,
			`{"model":"m-1","max_tokens":8,"messages":[{"role":"user","content":7}]}`, 400, "invalid_request_error", "content must be"},
		{"messages: a tool_result whose content is a number", "POST", anMsgs,
			`{"model":"m-1","max_tokens":8,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":7}]}]}`,
			400, "invalid_request_error", "content must be"},
		{"messages: a tool with no name", "POST", anMsgs,
			`{"model":"m-1","max_tokens":8,"tools":[{"input_schema":{}}],"messages":[{"role":"user","content":"hi"}]}`,
			400, "invalid_request_error", "tools[0]"},
		{"messages: negative max_tokens", "POST", anMsgs,
			`{"model":"m-1","max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}`, 400, "invalid_request_error", "max_tokens"},
		{"messages: no model", "POST", anMsgs,
			`{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, 404, "not_found_error", "model"},
	} {
		t.Run(tc.what, func(t *testing.T) {
			f := &fakeBackend{tokens: []string{"x"}, reason: FinishEOS}
			s := serve(t, f)
			resp := do(t, tc.method, s.URL+tc.path, tc.body)
			if resp.StatusCode != tc.status {
				b, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				t.Fatalf("status %d, want %d: %s", resp.StatusCode, tc.status, b)
			}
			e := readErr(t, resp)
			if e.Error.Type != tc.typ || !strings.Contains(e.Error.Message, tc.says) {
				t.Fatalf("error %+v, want type %q saying %q", e.Error, tc.typ, tc.says)
			}
			// The two APIs' bodies differ: only Anthropic's has a top-level type.
			if anthropic := tc.path == anMsgs; anthropic != (e.Type == "error") {
				t.Fatalf("the %s error body has top-level type %q", tc.path, e.Type)
			}
			if k := f.opts().Prompt.Kind; k != PromptNone {
				t.Fatalf("a refused request still reached the engine (prompt kind %d)", k)
			}
		})
	}
}

// TestABackendErrorBecomesTheStatusAClientActsOn: an unknown id is 404, a
// queue timeout is 429 (retrying is correct), a malformed request is 400 and
// only a genuine fault is 500. With the ErrInvalid arm removed from oaFailErr
// the 400 comes back 500, which a client retries forever.
func TestABackendErrorBecomesTheStatusAClientActsOn(t *testing.T) {
	bodies := map[string]string{
		oaChat: `{"model":"m-1","messages":[{"role":"user","content":"hi"}]}`,
		oaText: `{"model":"m-1","prompt":"hi"}`,
		anMsgs: `{"model":"m-1","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`,
	}
	for _, tc := range []struct {
		err          error
		status       int
		oaTyp, anTyp string
	}{
		{fmt.Errorf("%w: session %q", ErrNotFound, "s"), 404, "invalid_request_error", "not_found_error"},
		{ErrQueueTimeout, 429, "rate_limit_error", "overloaded_error"},
		{fmt.Errorf("%w: no prompt", ErrInvalid), 400, "invalid_request_error", "invalid_request_error"},
		{chatErr(&model.SystemRoleError{}), 400, "invalid_request_error", "invalid_request_error"},
		{errors.New("the card fell off the bus"), 500, "server_error", "api_error"},
	} {
		for _, path := range []string{oaChat, oaText, anMsgs} {
			s := serve(t, &fakeBackend{failWith: tc.err})
			resp := post(t, s, path, bodies[path])
			if resp.StatusCode != tc.status {
				resp.Body.Close()
				t.Errorf("%s with %q: status %d, want %d", path, tc.err, resp.StatusCode, tc.status)
				continue
			}
			want := tc.oaTyp
			if path == anMsgs {
				want = tc.anTyp
			}
			if e := readErr(t, resp); e.Error.Type != want || e.Error.Message != tc.err.Error() {
				t.Errorf("%s with %q: error %+v, want type %q and the engine's message", path, tc.err, e.Error, want)
			}
		}
	}
}

// collectFrames reads a whole SSE body.
func collectFrames(t *testing.T, resp *http.Response) []sseFrame {
	t.Helper()
	defer resp.Body.Close()
	fr := newFrameReader(resp.Body)
	var out []sseFrame
	for {
		x, ok := fr.next()
		if !ok {
			return out
		}
		out = append(out, x)
	}
}

// TestAStreamThatFailsEndsInAnErrorEvent: once the status line is 200 an
// error cannot become a status, so it is an `event: error` frame. OpenAI's
// stream still ends with [DONE] -- its clients wait for it -- and Anthropic's
// ends at the error, with no message_stop claiming success.
func TestAStreamThatFailsEndsInAnErrorEvent(t *testing.T) {
	for _, tc := range []struct {
		path, body string
		done       bool
	}{
		{oaChat, `{"model":"m-1","stream":true,"messages":[{"role":"user","content":"hi"}]}`, true},
		{oaText, `{"model":"m-1","stream":true,"prompt":"hi"}`, true},
		{anMsgs, `{"model":"m-1","stream":true,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, false},
	} {
		for _, b := range []Backend{
			&fakeBackend{failWith: errors.New("the card fell off the bus")},
			// A backend that returns nil without a finished event must not be
			// read as a success either.
			silentBackend{&fakeBackend{}},
		} {
			s := serve(t, b)
			resp := post(t, s, tc.path, tc.body)
			if resp.StatusCode != 200 {
				t.Fatalf("%s: status %d before the stream opened", tc.path, resp.StatusCode)
			}
			frames := collectFrames(t, resp)
			var names []string
			sawErr := false
			for _, f := range frames {
				names = append(names, f.event+"|"+f.data)
				if f.event == "error" {
					var e errBody
					if err := json.Unmarshal([]byte(f.data), &e); err != nil || e.Error.Type != "server_error" ||
						e.Error.Message == "" {
						t.Fatalf("%s: the error frame %q does not carry a typed message", tc.path, f.data)
					}
					sawErr = true
				}
				if f.event == "message_stop" {
					t.Fatalf("%s: a failed stream ended with message_stop, which reads as success", tc.path)
				}
			}
			if !sawErr {
				t.Fatalf("%s: a failed stream carried no error event: %v", tc.path, names)
			}
			last := frames[len(frames)-1]
			if gotDone := last.data == "[DONE]"; gotDone != tc.done {
				t.Fatalf("%s: last frame %q; [DONE] expected=%v", tc.path, last.data, tc.done)
			}
		}
	}
}

// silentBackend finishes without ever emitting: no started, no finished.
type silentBackend struct{ *fakeBackend }

func (silentBackend) Generate(context.Context, GenerateOptions, func(Event) error) error { return nil }

// TestAGenerateWithNoFinishedEventIsAFaultNotAPanic: the non-streaming arms
// read three fields off the finished event, so its absence must be a 500 and
// not a nil dereference that takes the process down.
func TestAGenerateWithNoFinishedEventIsAFaultNotAPanic(t *testing.T) {
	s := serve(t, silentBackend{&fakeBackend{}})
	for path, body := range map[string]string{
		oaChat: `{"model":"m-1","messages":[{"role":"user","content":"hi"}]}`,
		oaText: `{"model":"m-1","prompt":"hi"}`,
		anMsgs: `{"model":"m-1","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`,
	} {
		resp := post(t, s, path, body)
		if resp.StatusCode != 500 {
			resp.Body.Close()
			t.Fatalf("%s: status %d, want 500", path, resp.StatusCode)
		}
		if e := readErr(t, resp); !strings.Contains(e.Error.Message, "finished event") {
			t.Fatalf("%s: %+v does not say what was missing", path, e.Error)
		}
	}
}

// TestFinishReasonsSpeakEachAPIsVocabulary: one engine reason, two spellings,
// and Anthropic names the stop sequence that matched.
func TestFinishReasonsSpeakEachAPIsVocabulary(t *testing.T) {
	for _, tc := range []struct {
		reason      FinishReason
		stopAt      string
		oa, an      string
		wantStopSeq bool
	}{
		{FinishMaxTokens, "", "length", "max_tokens", false},
		{FinishStop, "END", "stop", "stop_sequence", true},
		{FinishEOS, "", "stop", "end_turn", false},
		{FinishCancelled, "", "stop", "end_turn", false},
	} {
		f := &fakeBackend{tokens: []string{"a", "b"}, reason: tc.reason, stopAt: tc.stopAt}
		s := serve(t, f)

		ro := post(t, s, oaChat, `{"model":"m-1","messages":[{"role":"user","content":"hi"}]}`)
		var oa oaChatResponse
		json.NewDecoder(ro.Body).Decode(&oa)
		ro.Body.Close()
		if len(oa.Choices) != 1 || oa.Choices[0].FinishReason == nil || *oa.Choices[0].FinishReason != tc.oa {
			t.Errorf("reason %d: openai finish_reason %+v, want %q", tc.reason, oa.Choices, tc.oa)
		}

		rt := post(t, s, oaText, `{"model":"m-1","prompt":"hi"}`)
		var legacy struct {
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		json.NewDecoder(rt.Body).Decode(&legacy)
		rt.Body.Close()
		if len(legacy.Choices) != 1 || legacy.Choices[0].FinishReason == nil || *legacy.Choices[0].FinishReason != tc.oa {
			t.Errorf("reason %d: completions finish_reason wrong, want %q", tc.reason, tc.oa)
		}

		ra := post(t, s, anMsgs, `{"model":"m-1","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
		var an anResponse
		json.NewDecoder(ra.Body).Decode(&an)
		ra.Body.Close()
		if an.StopReason == nil || *an.StopReason != tc.an {
			t.Errorf("reason %d: anthropic stop_reason %q, want %q", tc.reason, deref(an.StopReason), tc.an)
		}
		if got := an.StopSequence != nil && *an.StopSequence == tc.stopAt; got != tc.wantStopSeq {
			t.Errorf("reason %d: anthropic stop_sequence %q, want %q", tc.reason, deref(an.StopSequence), tc.stopAt)
		}

		// The streamed message_delta carries the same pair.
		rs := post(t, s, anMsgs, `{"model":"m-1","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
		for _, fr := range collectFrames(t, rs) {
			if fr.event != "message_delta" {
				continue
			}
			var d struct {
				Delta struct {
					StopReason   string  `json:"stop_reason"`
					StopSequence *string `json:"stop_sequence"`
				} `json:"delta"`
				Usage anUsage `json:"usage"`
			}
			json.Unmarshal([]byte(fr.data), &d)
			if d.Delta.StopReason != tc.an || (d.Delta.StopSequence != nil) != tc.wantStopSeq || d.Usage.OutputTokens != 2 {
				t.Errorf("reason %d: streamed message_delta %s", tc.reason, fr.data)
			}
		}
	}
}

// TestRequestFieldsReachTheEngine: sampling, the session extension, echo and
// every prompt shape /v1/completions accepts arrive in the one GenerateOptions.
func TestRequestFieldsReachTheEngine(t *testing.T) {
	f := &fakeBackend{tokens: []string{"x"}, reason: FinishEOS}
	s := serve(t, f)
	send := func(path, body string) GenerateOptions {
		t.Helper()
		resp := post(t, s, path, body)
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s %s: %d %s", path, body, resp.StatusCode, b)
		}
		return f.opts()
	}

	o := send(oaChat, `{"model":"m-1","temperature":0.7,"top_p":0.9,"seed":42,"messages":[{"role":"user","content":"hi"}]}`)
	if o.Sampling == nil || o.Sampling.Temp != 0.7 || o.Sampling.TopP != 0.9 || o.Sampling.Seed != 42 {
		t.Fatalf("openai sampling reached the engine as %+v", o.Sampling)
	}
	o = send(oaChat, `{"model":"m-1","top_k":40,"min_p":0.05,"repetition_penalty":1.1,"presence_penalty":0,"n":1,"logprobs":false,"messages":[{"role":"user","content":"hi"}]}`)
	if o.Sampling == nil || o.Sampling.TopK != 40 || o.Sampling.MinP != 0.05 || o.Sampling.RepeatPen != 1.1 || o.Sampling.Temp != 1 {
		t.Fatalf("top_k, min_p and repetition_penalty reached the engine as %+v", o.Sampling)
	}
	o = send(anMsgs, `{"model":"m-1","max_tokens":8,"temperature":0.5,"top_p":0.8,"top_k":40,"messages":[{"role":"user","content":"hi"}]}`)
	if o.Sampling == nil || o.Sampling.Temp != 0.5 || o.Sampling.TopP != 0.8 || o.Sampling.TopK != 40 {
		t.Fatalf("anthropic sampling reached the engine as %+v", o.Sampling)
	}

	for _, path := range []string{oaChat, anMsgs} {
		body := `{"jitllm_session":"sess-7","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`
		if o := send(path, body); o.SessionID != "sess-7" || o.ModelID != "" {
			t.Fatalf("%s: jitllm_session reached the engine as session %q model %q", path, o.SessionID, o.ModelID)
		}
	}

	o = send(oaText, `{"model":"m-1","prompt":["first","second"],"echo":true,"stop":["a","b"]}`)
	if o.Prompt.Kind != PromptText || o.Prompt.Text != "second" || !o.Echo || strings.Join(o.Stop, ",") != "a,b" {
		t.Fatalf("a batched prompt's last run reached the engine as %+v echo=%v stop=%v", o.Prompt, o.Echo, o.Stop)
	}
	o = send(oaText, `{"model":"m-1","prompt":[1,2,3],"max_tokens":5}`)
	if o.Prompt.Kind != PromptIDs || len(o.Prompt.IDs) != 3 || o.Prompt.IDs[2] != 3 || o.MaxTokens != 5 {
		t.Fatalf("a token-id prompt reached the engine as %+v (max %d)", o.Prompt, o.MaxTokens)
	}

	// An Anthropic conversation with a tool round trip: the assistant's
	// tool_use becomes its call, and the tool_result a "tool" turn of its own.
	o = send(anMsgs, `{"model":"m-1","max_tokens":8,"tool_choice":{"type":"none"},
		"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],
		"messages":[
		{"role":"user","content":"weather?"},
		{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"get_weather"}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"sunny"},{"type":"text","text":"and?"}]}]}`)
	m := o.Prompt.Chat.Messages
	if len(m) != 4 || len(m[1].ToolCalls) != 1 || m[1].ToolCalls[0].Arguments != "{}" ||
		m[2].Role != "tool" || m[2].ToolCallID != "t1" || m[2].Content != "sunny" || m[3].Content != "and?" {
		t.Fatalf("the tool round trip reached the template as %+v", m)
	}
	// tool_choice none withholds the tools from the template.
	if o.Prompt.Chat.Tools != nil {
		t.Fatalf("tool_choice none still rendered tools %s", o.Prompt.Chat.Tools)
	}
}

// TestLegacyCompletionsStreamTextChunksAndEndWithDone: /v1/completions
// streams `text` chunks, not deltas, and its last chunk carries the finish
// reason and usage.
func TestLegacyCompletionsStreamTextChunksAndEndWithDone(t *testing.T) {
	s := serve(t, &fakeBackend{tokens: []string{" Par", "is", "", "."}, reason: FinishMaxTokens})
	frames := collectFrames(t, post(t, s, oaText, `{"model":"m-1","stream":true,"prompt":"The capital is"}`))
	if len(frames) == 0 || frames[len(frames)-1].data != "[DONE]" {
		t.Fatalf("the stream did not end with [DONE]: %v", frames)
	}
	var text strings.Builder
	var finish string
	var usage *oaUsage
	for _, f := range frames[:len(frames)-1] {
		var c struct {
			Object  string `json:"object"`
			Choices []struct {
				Text         string  `json:"text"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *oaUsage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(f.data), &c); err != nil {
			t.Fatalf("frame %q: %v", f.data, err)
		}
		if c.Object != "text_completion" || len(c.Choices) != 1 {
			t.Fatalf("frame %q is not a text_completion chunk", f.data)
		}
		text.WriteString(c.Choices[0].Text)
		if c.Choices[0].FinishReason != nil {
			finish, usage = *c.Choices[0].FinishReason, c.Usage
		}
	}
	// The empty token sends no frame of its own.
	if text.String() != " Paris." || len(frames) != 5 {
		t.Fatalf("streamed %q over %d frames, want \" Paris.\" over 3 text frames, a finish and [DONE]",
			text.String(), len(frames))
	}
	if finish != "length" || usage == nil || usage.CompletionTokens != 4 || usage.TotalTokens != 11 {
		t.Fatalf("the last chunk carries finish %q usage %+v", finish, usage)
	}
}

// TestTheShimsOverTheRealEngine: BindTarget resolves a model by its id or its
// file name and a session by the jitllm extension, and a chat request to a
// model with no template is a 400 from both shims, since a 500 would be
// retried by a client.
func TestTheShimsOverTheRealEngine(t *testing.T) {
	e, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})

	resp := do(t, "GET", c.url+"/v1/models", "")
	var ml struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&ml)
	resp.Body.Close()
	var listed []string
	for _, d := range ml.Data {
		listed = append(listed, d.ID)
	}
	if strings.Join(listed, ",") != "small,stories260K" {
		t.Fatalf("/v1/models lists %v, want the id and the file name", listed)
	}

	for _, name := range []string{"small", "stories260K"} {
		resp := do(t, "POST", c.url+oaText, `{"model":"`+name+`","prompt":"`+story+`","max_tokens":3}`)
		var out struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
		if resp.StatusCode != 200 || len(out.Choices) != 1 || out.Choices[0].Text == "" {
			t.Fatalf("model %q: status %d, choices %+v", name, resp.StatusCode, out.Choices)
		}
	}
	if resp := do(t, "POST", c.url+oaText, `{"model":"nope","prompt":"x"}`); resp.StatusCode != 404 {
		t.Fatalf("an unknown model: status %d, want 404", resp.StatusCode)
	} else {
		resp.Body.Close()
	}
	if resp := do(t, "POST", c.url+oaText, `{"jitllm_session":"nope","prompt":"x"}`); resp.StatusCode != 404 {
		t.Fatalf("an unknown session: status %d, want 404", resp.StatusCode)
	} else {
		resp.Body.Close()
	}

	if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "s", MaxSeq: 64}); err != nil {
		t.Fatal(err)
	}
	resp = do(t, "POST", c.url+oaText, `{"jitllm_session":"s","prompt":"`+story+`","max_tokens":3}`)
	resp.Body.Close()
	if s, _ := e.Session("s"); resp.StatusCode != 200 || s.snapPos.Load() == 0 {
		t.Fatalf("a generate on jitllm_session: status %d, the session did not advance", resp.StatusCode)
	}

	for path, body := range map[string]string{
		oaChat: `{"model":"small","messages":[{"role":"user","content":"hi"}]}`,
		anMsgs: `{"model":"small","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`,
	} {
		resp := do(t, "POST", c.url+path, body)
		if resp.StatusCode != 400 {
			resp.Body.Close()
			t.Fatalf("%s to a model with no template: status %d, want 400", path, resp.StatusCode)
		}
		if e := readErr(t, resp); e.Error.Type != "invalid_request_error" || !strings.Contains(e.Error.Message, "RAW COMPLETION") {
			t.Fatalf("%s: %+v", path, e.Error)
		}
	}

	var o GenerateOptions
	if err := e.BindTarget(&o, "", ""); err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("BindTarget with neither a model nor a session: %v", err)
	}
}

// TestAnErrorNamesTheModelAsTheRequestDid: a model loaded under an id and
// addressed by its file name is named in an error as the request named it --
// a client that sent "stories260K" cannot act on an id it never saw.
func TestAnErrorNamesTheModelAsTheRequestDid(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	for _, name := range []string{"stories260K", "small"} {
		for path, body := range map[string]string{
			oaChat: `{"model":"` + name + `","messages":[{"role":"user","content":"hi"}]}`,
			anMsgs: `{"model":"` + name + `","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`,
		} {
			resp := do(t, "POST", c.url+path, body)
			e := readErr(t, resp)
			if !strings.Contains(e.Error.Message, `model "`+name+`"`) {
				t.Errorf("%s addressed as %q: the error names something else: %s", path, name, e.Error.Message)
			}
		}
	}
}

// deref prints an optional string field.
func deref(p *string) string {
	if p == nil {
		return "<absent>"
	}
	return *p
}
