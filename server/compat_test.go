package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---------------------------------------------------------------- helpers

type sseFrame struct {
	event string
	data  string
}

// readFrames reads SSE frames one at a time from a live response body. It is
// deliberately incremental: a helper that consumed the whole body first would
// make every streaming gate below pass against a server that buffers.
type frameReader struct {
	sc *bufio.Scanner
}

func newFrameReader(r io.Reader) *frameReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	return &frameReader{sc: sc}
}

func (f *frameReader) next() (sseFrame, bool) {
	var fr sseFrame
	for f.sc.Scan() {
		line := f.sc.Text()
		switch {
		case line == "":
			if fr.data != "" || fr.event != "" {
				return fr, true
			}
		case strings.HasPrefix(line, "event: "):
			fr.event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			fr.data = strings.TrimPrefix(line, "data: ")
		}
	}
	if fr.data != "" || fr.event != "" {
		return fr, true
	}
	return sseFrame{}, false
}

func serve(t *testing.T, b Backend) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(CompatHandler(b))
	t.Cleanup(s.Close)
	return s
}

// post issues the request under a context cancelled by cleanup, so a stream
// gate that fails mid-response does not leave httptest.Server.Close blocked
// until the package timeout.
func post(t *testing.T, s *httptest.Server, path, body string) *http.Response {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("building POST %s: %v", path, err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	return resp
}

// ---------------------------------------------------------------- streaming

// TestOpenAIChatStreamArrivesTokenByToken proves the flush: the fake backend
// will not produce token N+1 until the test has read token N off the wire, so
// a buffering handler times out. Without `s.rc.Flush()` in sseWriter.send it
// fails with "the stream is buffered, not flushed".
func TestOpenAIChatStreamArrivesTokenByToken(t *testing.T) {
	// The release is a close, not N sends, so a failing run unblocks the
	// handler in beforeToken and httptest.Server.Close does not hang.
	release := make(chan struct{})
	var once sync.Once
	letAllThrough := func() { once.Do(func() { close(release) }) }
	// A defer, not t.Cleanup: cleanups run LIFO, so serve's later
	// t.Cleanup(s.Close) would run first and block on this very request.
	defer letAllThrough()

	f := &fakeBackend{
		tokens: []string{"Hello", " there", "!"},
		reason: FinishEOS,
		beforeToken: func(i int) {
			if i > 0 {
				<-release
			}
		},
	}
	s := serve(t, f)
	resp := post(t, s, "/v1/chat/completions",
		`{"model":"m-1","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("Content-Type %q, want text/event-stream", ct)
	}

	fr := newFrameReader(resp.Body)
	type got struct {
		fr sseFrame
		ok bool
	}
	nextWithTimeout := func(which int) sseFrame {
		ch := make(chan got, 1)
		go func() { x, ok := fr.next(); ch <- got{x, ok} }()
		select {
		case g := <-ch:
			if !g.ok {
				t.Fatalf("stream ended early at frame %d", which)
			}
			return g.fr
		case <-time.After(3 * time.Second):
			t.Fatalf("timed out waiting for streamed frame %d: the stream is buffered, not flushed", which)
			return sseFrame{}
		}
	}

	// Frame 0: the role-only delta. Several clients key their state machine on
	// it, so it is asserted rather than tolerated.
	first := nextWithTimeout(0)
	var chunk oaChatResponse
	if err := json.Unmarshal([]byte(first.data), &chunk); err != nil {
		t.Fatalf("frame 0 is not JSON: %q", first.data)
	}
	if chunk.Object != "chat.completion.chunk" {
		t.Fatalf("frame 0 object %q, want chat.completion.chunk", chunk.Object)
	}
	if len(chunk.Choices) != 1 || chunk.Choices[0].Delta == nil || chunk.Choices[0].Delta.Role != "assistant" {
		t.Fatalf("frame 0 is not the role delta: %q", first.data)
	}

	// Frame 1 is the first token, and it must arrive while the backend is
	// still blocked on `release` -- i.e. before the generate has finished.
	one := nextWithTimeout(1)
	json.Unmarshal([]byte(one.data), &chunk)
	if chunk.Choices[0].Delta.Content != "Hello" {
		t.Fatalf("frame 1 content %q, want %q", chunk.Choices[0].Delta.Content, "Hello")
	}

	// Now let the rest through and collect everything.
	letAllThrough()

	var text strings.Builder
	text.WriteString("Hello")
	sawDone, finish := false, ""
	for {
		x, ok := fr.next()
		if !ok {
			break
		}
		if x.data == "[DONE]" {
			sawDone = true
			break
		}
		var c oaChatResponse
		if err := json.Unmarshal([]byte(x.data), &c); err != nil {
			t.Fatalf("frame is not JSON: %q", x.data)
		}
		if len(c.Choices) == 0 {
			continue
		}
		if d := c.Choices[0].Delta; d != nil {
			text.WriteString(d.Content)
		}
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if got := text.String(); got != "Hello there!" {
		t.Fatalf("streamed text %q, want %q", got, "Hello there!")
	}
	if finish != "stop" {
		t.Fatalf("finish_reason %q, want %q", finish, "stop")
	}
	// [DONE] is not part of SSE but is part of this API; OpenAI clients
	// block waiting for it.
	if !sawDone {
		t.Fatal("the stream never sent `data: [DONE]`; an OpenAI client waits for that sentinel forever")
	}
}

// TestAnthropicStreamIsNamedEventsAndEndsWithMessageStop: Anthropic clients
// dispatch on the SSE event name and get no [DONE]. With sse.send in place of
// sse.sendNamed it fails with "frame 0 has no event name".
func TestAnthropicStreamIsNamedEventsAndEndsWithMessageStop(t *testing.T) {
	f := &fakeBackend{tokens: []string{"Par", "is"}, reason: FinishEOS}
	s := serve(t, f)
	resp := post(t, s, "/v1/messages",
		`{"model":"m-1","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"capital?"}]}`)
	defer resp.Body.Close()

	fr := newFrameReader(resp.Body)
	var names []string
	var text strings.Builder
	for {
		x, ok := fr.next()
		if !ok {
			break
		}
		if x.data == "[DONE]" {
			t.Fatal("the Anthropic stream sent OpenAI's [DONE] sentinel, which no Anthropic client has a case for")
		}
		if x.event == "" {
			t.Fatalf("frame %d has no event name; an Anthropic client dispatches on it (data %q)",
				len(names), x.data)
		}
		names = append(names, x.event)
		if x.event == "content_block_delta" {
			var d struct {
				Delta struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"delta"`
			}
			json.Unmarshal([]byte(x.data), &d)
			if d.Delta.Type != "text_delta" {
				t.Fatalf("delta type %q, want text_delta", d.Delta.Type)
			}
			text.WriteString(d.Delta.Text)
		}
	}
	want := []string{"message_start", "content_block_start", "content_block_delta",
		"content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence\n got %v\nwant %v", names, want)
	}
	if got := text.String(); got != "Paris" {
		t.Fatalf("streamed text %q, want %q", got, "Paris")
	}
}

// ---------------------------------------------------------------- request shapes

// TestBothShimsAcceptRealClientRequestBodies uses the bodies the official
// SDKs send: content-part arrays, max_completion_tokens, a top-level system.
// With oaContent (or anText) reduced to a plain string unmarshal it fails
// with status 400.
func TestBothShimsAcceptRealClientRequestBodies(t *testing.T) {
	cases := []struct {
		name    string
		path    string
		body    string
		wantSys string
		wantMax int
		wantMsg string
	}{{
		name: "openai/content-parts and max_completion_tokens",
		path: "/v1/chat/completions",
		body: `{"model":"m-1","max_completion_tokens":128,"messages":[
		         {"role":"system","content":"be terse"},
		         {"role":"user","content":[{"type":"text","text":"what is 2+2?"}]}]}`,
		wantMax: 128,
		wantMsg: "what is 2+2?",
	}, {
		name:    "openai/plain string content and stop as a string",
		path:    "/v1/chat/completions",
		body:    `{"model":"m-1","max_tokens":16,"stop":"\n\n","messages":[{"role":"user","content":"hi"}]}`,
		wantMax: 16,
		wantMsg: "hi",
	}, {
		name: "anthropic/top-level system and content blocks",
		path: "/v1/messages",
		body: `{"model":"m-1","max_tokens":1024,"system":"You are terse.","messages":[
		         {"role":"user","content":[{"type":"text","text":"what is 2+2?"}]}]}`,
		wantSys: "You are terse.",
		wantMax: 1024,
		wantMsg: "what is 2+2?",
	}, {
		name:    "anthropic/system as a block list",
		path:    "/v1/messages",
		body:    `{"model":"m-1","max_tokens":8,"system":[{"type":"text","text":"terse"}],"messages":[{"role":"user","content":"hi"}]}`,
		wantSys: "terse",
		wantMax: 8,
		wantMsg: "hi",
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := &fakeBackend{tokens: []string{"4"}, reason: FinishEOS}
			s := serve(t, f)
			resp := post(t, s, c.path, c.body)
			defer resp.Body.Close()
			b, _ := io.ReadAll(resp.Body)
			if resp.StatusCode != 200 {
				t.Fatalf("status %d: %s", resp.StatusCode, bytes.TrimSpace(b))
			}
			o := f.opts()
			if o.Prompt.Kind != PromptChat || o.Prompt.Chat == nil {
				t.Fatalf("the request did not reach the engine as a chat prompt: %+v", o.Prompt)
			}
			if o.MaxTokens != c.wantMax {
				t.Fatalf("max tokens reached the engine as %d, want %d", o.MaxTokens, c.wantMax)
			}
			last := o.Prompt.Chat.Messages[len(o.Prompt.Chat.Messages)-1]
			if last.Content != c.wantMsg {
				t.Fatalf("the user message reached the engine as %q, want %q", last.Content, c.wantMsg)
			}
			// Anthropic's system prompt is top-level and sets HasSystem;
			// OpenAI's arrives as a message and must not.
			if c.wantSys != "" {
				if !o.Prompt.Chat.HasSystem || o.Prompt.Chat.System != c.wantSys {
					t.Fatalf("top-level system reached the engine as (%v, %q), want (true, %q)",
						o.Prompt.Chat.HasSystem, o.Prompt.Chat.System, c.wantSys)
				}
			} else if o.Prompt.Chat.HasSystem {
				t.Fatalf("a system MESSAGE was promoted to the top-level system field; "+
					"OpenAI's system is a message and flattening it loses which one the caller meant "+
					"(got %q)", o.Prompt.Chat.System)
			}
		})
	}
}

// TestStopSequencesReachTheEngineFromBothShims. The two APIs spell it
// differently -- `stop` and `stop_sequences` -- and both must arrive.
func TestStopSequencesReachTheEngineFromBothShims(t *testing.T) {
	for _, c := range []struct{ name, path, body string }{
		{"openai/array", "/v1/chat/completions",
			`{"model":"m-1","stop":["END","###"],"messages":[{"role":"user","content":"hi"}]}`},
		{"anthropic/stop_sequences", "/v1/messages",
			`{"model":"m-1","max_tokens":8,"stop_sequences":["END","###"],"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := &fakeBackend{tokens: []string{"x"}, reason: FinishEOS}
			s := serve(t, f)
			resp := post(t, s, c.path, c.body)
			resp.Body.Close()
			got := strings.Join(f.opts().Stop, ",")
			if got != "END,###" {
				t.Fatalf("stop reached the engine as %q, want %q", got, "END,###")
			}
		})
	}
}

// TestAnthropicRequiresMaxTokens: required by that API, optional in OpenAI's.
func TestAnthropicRequiresMaxTokens(t *testing.T) {
	f := &fakeBackend{tokens: []string{"x"}}
	s := serve(t, f)
	resp := post(t, s, "/v1/messages", `{"model":"m-1","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d for a request with no max_tokens, want 400", resp.StatusCode)
	}
	var e anError
	json.NewDecoder(resp.Body).Decode(&e)
	if e.Type != "error" || e.Error.Type != "invalid_request_error" {
		t.Fatalf("error body %+v does not match the Anthropic error shape", e)
	}
}

// ---------------------------------------------------------------- one path

// TestBothShimsProduceTheSameTextFromOneBackend asserts the shims are thin
// adapters: one token script through two wire formats must reconstruct the
// same text.
func TestBothShimsProduceTheSameTextFromOneBackend(t *testing.T) {
	script := []string{"The ", "capital", " is ", "Paris", "."}

	fo := &fakeBackend{tokens: script, reason: FinishEOS}
	so := serve(t, fo)
	ro := post(t, so, "/v1/chat/completions",
		`{"model":"m-1","messages":[{"role":"user","content":"capital?"}]}`)
	defer ro.Body.Close()
	var oa oaChatResponse
	json.NewDecoder(ro.Body).Decode(&oa)

	fa := &fakeBackend{tokens: script, reason: FinishEOS}
	sa := serve(t, fa)
	ra := post(t, sa, "/v1/messages",
		`{"model":"m-1","max_tokens":64,"messages":[{"role":"user","content":"capital?"}]}`)
	defer ra.Body.Close()
	var an anResponse
	json.NewDecoder(ra.Body).Decode(&an)

	want := strings.Join(script, "")
	if len(oa.Choices) != 1 || oa.Choices[0].Message == nil {
		t.Fatalf("openai response has no message: %+v", oa)
	}
	if got := oa.Choices[0].Message.Content; got != want {
		t.Fatalf("openai text %q, want %q", got, want)
	}
	if len(an.Content) != 1 {
		t.Fatalf("anthropic response has %d content blocks, want 1", len(an.Content))
	}
	if got := an.Content[0].Text; got == nil || *got != want {
		t.Fatalf("anthropic text %v, want %q", got, want)
	}

	// The usage fields are spelled differently and must both be right.
	if oa.Usage.CompletionTokens != len(script) || an.Usage.OutputTokens != len(script) {
		t.Fatalf("usage disagrees: openai completion_tokens=%d, anthropic output_tokens=%d, want %d",
			oa.Usage.CompletionTokens, an.Usage.OutputTokens, len(script))
	}
	// And the stop vocabularies are different: "stop" against "end_turn".
	if oa.Choices[0].FinishReason == nil || *oa.Choices[0].FinishReason != "stop" {
		t.Fatalf("openai finish_reason %v, want \"stop\"", oa.Choices[0].FinishReason)
	}
	if an.StopReason == nil || *an.StopReason != "end_turn" {
		t.Fatalf("anthropic stop_reason %v, want \"end_turn\"", an.StopReason)
	}
}

// TestTheQueueWaitIsReportedThroughBothShims: neither spec has a field for
// the queue wait or placement, so the `jitllm` extension carries them.
func TestTheQueueWaitIsReportedThroughBothShims(t *testing.T) {
	f := &fakeBackend{tokens: []string{"x"}, reason: FinishEOS}
	s := serve(t, f)

	ro := post(t, s, "/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":"hi"}]}`)
	defer ro.Body.Close()
	var oa oaChatResponse
	json.NewDecoder(ro.Body).Decode(&oa)
	if oa.Jitllm == nil {
		t.Fatal("the openai response carries no jitllm extension: a caller cannot learn it queued")
	}
	if oa.Jitllm.QueuedMillis != 1500 || oa.Jitllm.DeviceBlocks != 22 || oa.Jitllm.CPUBlocks != 10 {
		t.Fatalf("jitllm extension %+v does not carry the queue wait and the split", oa.Jitllm)
	}
	if oa.Jitllm.BytesPerToken == 0 {
		t.Fatal("bytes_per_token is 0: a tok/s figure with no byte count behind it is unfalsifiable")
	}

	ra := post(t, s, "/v1/messages", `{"model":"m-1","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	defer ra.Body.Close()
	var an anResponse
	json.NewDecoder(ra.Body).Decode(&an)
	if an.Jitllm == nil || an.Jitllm.QueuedMillis != 1500 {
		t.Fatalf("the anthropic response carries no queue wait: %+v", an.Jitllm)
	}
}

// TestOpenAIModelsListsLoadedModelsAndTheirFileNames.
func TestOpenAIModelsListsLoadedModelsAndTheirFileNames(t *testing.T) {
	s := serve(t, &fakeBackend{})
	resp, err := http.Get(s.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Object string `json:"object"`
		Data   []struct {
			ID     string `json:"id"`
			Object string `json:"object"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if out.Object != "list" {
		t.Fatalf("object %q, want list", out.Object)
	}
	var ids []string
	for _, d := range out.Data {
		ids = append(ids, d.ID)
	}
	// Both the generated id and the file name resolve, because a client
	// configured from `jitllm run` sends the file name.
	if strings.Join(ids, ",") != "m-1,tinyllama" {
		t.Fatalf("ids %v, want [m-1 tinyllama]", ids)
	}
}

// TestLegacyCompletionsCarryTextNotAMessage: /v1/completions has its own body
// shape, with `text` rather than a message.
func TestLegacyCompletionsCarryTextNotAMessage(t *testing.T) {
	f := &fakeBackend{tokens: []string{" Paris."}, reason: FinishEOS}
	s := serve(t, f)
	resp := post(t, s, "/v1/completions",
		`{"model":"m-1","prompt":"The capital of France is","max_tokens":8}`)
	defer resp.Body.Close()
	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Text         string  `json:"text"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if out.Object != "text_completion" {
		t.Fatalf("object %q, want text_completion", out.Object)
	}
	if len(out.Choices) != 1 || out.Choices[0].Text != " Paris." {
		t.Fatalf("choices %+v, want one with text %q", out.Choices, " Paris.")
	}
	if o := f.opts(); o.Prompt.Kind != PromptText || o.Prompt.Text != "The capital of France is" {
		t.Fatalf("the prompt reached the engine as %+v, want a text prompt", o.Prompt)
	}
}
