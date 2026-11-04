package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// The native API against the OpenAI routes on real models: the same request
// through either door must give the same text, finish reason, usage, tool
// calls and log-probabilities. stories260K carries no chat template (the
// completion paths); Qwen3-0.6B does (chat, tools, response_format).

const parityChatModel = "Qwen3-0.6B-Q8_0.jlm"

type pcall struct{ Name, Args string }

type ptop struct {
	Tok string
	LP  float32
}

type plp struct {
	Tok string
	LP  float32
	Top []ptop
}

type pchoice struct {
	Text, Finish string
	Calls        []pcall
	LPs          []plp
}

// preply is a reply in the terms both surfaces share.
type preply struct {
	Choices                          []pchoice
	Prompt, Completion, Total, Cache int
}

// pbFinishName is a native finish reason in the OpenAI surface's words.
func pbFinishName(r v1.FinishReason) string {
	switch r {
	case v1.FinishReason_FINISH_REASON_MAX_TOKENS:
		return "length"
	case v1.FinishReason_FINISH_REASON_TOOL_CALLS:
		return "tool_calls"
	case v1.FinishReason_FINISH_REASON_STOP, v1.FinishReason_FINISH_REASON_EOS:
		return "stop"
	}
	return r.String()
}

func chatLPs(l *oaChatLogprobs) []plp {
	if l == nil {
		return nil
	}
	var out []plp
	for _, c := range l.Content {
		e := plp{Tok: c.Token, LP: c.Logprob}
		for _, a := range c.TopLogprobs {
			e.Top = append(e.Top, ptop{a.Token, a.Logprob})
		}
		out = append(out, e)
	}
	return out
}

// legacyLPs reads /v1/completions' parallel arrays; its alternatives are a
// map, so they are ordered here.
func legacyLPs(l *oaLegacyLogprobs) []plp {
	if l == nil {
		return nil
	}
	var out []plp
	for i, tok := range l.Tokens {
		e := plp{Tok: tok, LP: l.TokenLogprobs[i]}
		for k, v := range l.TopLogprobs[i] {
			e.Top = append(e.Top, ptop{k, v})
		}
		sortTop(e.Top)
		out = append(out, e)
	}
	return out
}

func sortTop(t []ptop) {
	slices.SortFunc(t, func(a, b ptop) int {
		switch {
		case a.LP > b.LP:
			return -1
		case a.LP < b.LP:
			return 1
		}
		return strings.Compare(a.Tok, b.Tok)
	})
}

func httpPost(t *testing.T, url, body string) []byte {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%s: %d %s", url, resp.StatusCode, b)
	}
	return b
}

func usageOf(u *oaUsage, r *preply) {
	if u == nil {
		return
	}
	r.Prompt, r.Completion, r.Total = u.PromptTokens, u.CompletionTokens, u.TotalTokens
	if u.PromptTokensDetails != nil {
		r.Cache = u.PromptTokensDetails.CachedTokens
	}
}

// httpChatReply posts a chat body; stream reads the SSE instead.
func httpChatReply(t *testing.T, base, body string, stream bool) preply {
	t.Helper()
	var r preply
	if !stream {
		var out oaChatResponse
		if err := json.Unmarshal(httpPost(t, base+"/v1/chat/completions", body), &out); err != nil {
			t.Fatal(err)
		}
		for _, c := range out.Choices {
			pc := pchoice{Text: c.Message.Content, Finish: *c.FinishReason, LPs: chatLPs(c.Logprobs)}
			for _, tc := range c.Message.ToolCalls {
				pc.Calls = append(pc.Calls, pcall{tc.Function.Name, tc.Function.Arguments})
			}
			r.Choices = append(r.Choices, pc)
		}
		usageOf(out.Usage, &r)
		return r
	}
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	fr := newFrameReader(resp.Body)
	grow := func(i int) {
		for len(r.Choices) <= i {
			r.Choices = append(r.Choices, pchoice{})
		}
	}
	for {
		x, ok := fr.next()
		if !ok || x.data == "[DONE]" {
			break
		}
		var ch oaChatResponse
		if err := json.Unmarshal([]byte(x.data), &ch); err != nil {
			t.Fatalf("%s: %v", x.data, err)
		}
		usageOf(ch.Usage, &r)
		for _, c := range ch.Choices {
			grow(c.Index)
			pc := &r.Choices[c.Index]
			if c.Delta != nil {
				pc.Text += c.Delta.Content
				for _, tc := range c.Delta.ToolCalls {
					pc.Calls = append(pc.Calls, pcall{tc.Function.Name, tc.Function.Arguments})
				}
			}
			pc.LPs = append(pc.LPs, chatLPs(c.Logprobs)...)
			if c.FinishReason != nil {
				pc.Finish = *c.FinishReason
			}
		}
	}
	return r
}

// httpCompletionReply posts a /v1/completions body.
func httpCompletionReply(t *testing.T, base, body string) preply {
	t.Helper()
	var out legacyResponse
	if err := json.Unmarshal(httpPost(t, base+"/v1/completions", body), &out); err != nil {
		t.Fatal(err)
	}
	var r preply
	for _, c := range out.Choices {
		r.Choices = append(r.Choices, pchoice{Text: c.Text, Finish: *c.FinishReason, LPs: legacyLPs(c.Logprobs)})
	}
	usageOf(out.Usage, &r)
	return r
}

func pbLPs(in []*v1.TokenLogprob) []plp {
	var out []plp
	for _, l := range in {
		e := plp{Tok: l.Token, LP: l.Logprob}
		for _, a := range l.Top {
			e.Top = append(e.Top, ptop{a.Token, a.Logprob})
		}
		out = append(out, e)
	}
	return out
}

func pbUsage(u *v1.Usage, r *preply) {
	if u != nil {
		r.Prompt, r.Completion, r.Total, r.Cache = int(u.PromptTokens), int(u.CompletionTokens),
			int(u.TotalTokens), int(u.CachedTokens)
	}
}

// connectReply sends m to Complete, or to Generate when stream is set.
func connectReply(t *testing.T, cl jitllmv1connect.InferenceServiceClient, m *v1.GenerateRequest, stream bool) preply {
	t.Helper()
	ctx := context.Background()
	var r preply
	if !stream {
		resp, err := cl.Complete(ctx, connect.NewRequest(m))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		for _, c := range resp.Msg.Choices {
			pc := pchoice{Text: c.Text, Finish: pbFinishName(c.Finished.Reason), LPs: pbLPs(c.Logprobs)}
			for _, tc := range c.ToolCalls {
				pc.Calls = append(pc.Calls, pcall{tc.Name, tc.Arguments})
			}
			r.Choices = append(r.Choices, pc)
		}
		pbUsage(resp.Msg.Usage, &r)
		// The first choice is also the response's own fields.
		if f := resp.Msg.Choices[0]; resp.Msg.Text != f.Text || resp.Msg.Finished.GetReason() != f.Finished.GetReason() ||
			resp.Msg.Finished.GetCompletionTokens() != f.Finished.GetCompletionTokens() {
			t.Fatalf("the response's own fields are not the first choice's")
		}
		return r
	}
	m.IncludeUsage = true
	st, err := cl.Generate(ctx, connect.NewRequest(m))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	grow := func(i int) *pchoice {
		for len(r.Choices) <= i {
			r.Choices = append(r.Choices, pchoice{})
		}
		return &r.Choices[i]
	}
	for st.Receive() {
		ev := st.Msg()
		pc := grow(int(ev.Choice))
		switch e := ev.Event.(type) {
		case *v1.GenerateResponse_Token:
			pc.Text += e.Token.Text
			if e.Token.Logprob != nil {
				pc.LPs = append(pc.LPs, pbLPs([]*v1.TokenLogprob{e.Token.Logprob})...)
			}
		case *v1.GenerateResponse_ToolCalls:
			for _, tc := range e.ToolCalls.Calls {
				if int(e.ToolCalls.FirstIndex)+0 != len(pc.Calls) {
					t.Fatalf("calls arrived at index %d with %d sent", e.ToolCalls.FirstIndex, len(pc.Calls))
				}
				pc.Calls = append(pc.Calls, pcall{tc.Name, tc.Arguments})
			}
		case *v1.GenerateResponse_Finished:
			pc.Finish = pbFinishName(e.Finished.Reason)
		case *v1.GenerateResponse_Usage:
			pbUsage(e.Usage.Usage, &r)
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return r
}

// both runs a request through the native API once to warm the memory cache,
// then through each surface once. Both then restore the prompt, so
// cached_tokens is a number the surfaces must agree on rather than a
// difference of order.
func both(warm, openai, native func() preply) (preply, preply) {
	warm()
	return openai(), native()
}

func sameReply(t *testing.T, what string, native, openai preply) {
	t.Helper()
	if !reflect.DeepEqual(native, openai) {
		t.Fatalf("%s: Connect and OpenAI differ\n  Connect %+v\n  OpenAI  %+v", what, native, openai)
	}
}

func parityEngine(t *testing.T) (clients, *Engine) {
	t.Helper()
	small, chatPath := modelPath(t, smallModel), modelPath(t, parityChatModel)
	e := New(admitConfig(Config{ModelDir: filepath.Dir(small), Probe: noProbe, Version: "test", NoOnDemand: true}))
	t.Cleanup(e.Close)
	for id, p := range map[string]string{"s": small, "q": chatPath} {
		if _, err := e.LoadModel(LoadOptions{Path: p, ModelID: id}); err != nil {
			t.Fatalf("LoadModel(%s): %v", p, err)
		}
	}
	return serveEngine(t, e), e
}

func TestNativeMatchesOpenAIOnCompletions(t *testing.T) {
	c, _ := parityEngine(t)
	text := func(s string) *v1.PromptInput { return &v1.PromptInput{Input: &v1.PromptInput_Text{Text: s}} }
	samp := func(temp, topP float32, seed uint64) *v1.SamplingParams {
		return &v1.SamplingParams{Temperature: &temp, TopP: &topP, Seed: &seed}
	}
	compare := func(name, body string, m *v1.GenerateRequest) preply {
		t.Helper()
		run := func() preply { return connectReply(t, c.inference, m, false) }
		openai, native := both(run, func() preply { return httpCompletionReply(t, c.url, body) }, run)
		sameReply(t, name, native, openai)
		if len(native.Choices) == 0 || native.Choices[0].Text == "" || native.Completion == 0 {
			t.Fatalf("%s: nothing was generated: %+v", name, native)
		}
		t.Logf("%s: %q finish %s usage %d+%d cached %d", name, native.Choices[0].Text, native.Choices[0].Finish,
			native.Prompt, native.Completion, native.Cache)
		return native
	}

	greedy := compare("greedy with logprobs",
		`{"model":"s","prompt":"Once upon a time","max_tokens":16,"temperature":0,"logprobs":3}`,
		&v1.GenerateRequest{ModelId: "s", Prompt: text("Once upon a time"), MaxTokens: 16,
			Sampling: samp(0, 1, 0), Logprobs: true, TopLogprobs: ptr(int32(3))})
	if n := len(greedy.Choices[0].LPs); n != greedy.Completion || len(greedy.Choices[0].LPs[0].Top) != 3 {
		t.Fatalf("logprobs: %d entries for %d tokens, top %d", n, greedy.Completion, len(greedy.Choices[0].LPs[0].Top))
	}
	if greedy.Choices[0].Finish != "length" {
		t.Fatalf("finish %q, want length", greedy.Choices[0].Finish)
	}

	compare("sampled, three choices",
		`{"model":"s","prompt":"Once upon a time","max_tokens":12,"n":3,"temperature":0.75,"top_p":0.875,`+
			`"top_k":40,"min_p":0.0625,"repetition_penalty":1.125,"seed":11}`,
		&v1.GenerateRequest{ModelId: "s", Prompt: text("Once upon a time"), MaxTokens: 12, N: 3,
			Sampling: &v1.SamplingParams{Temperature: ptr(float32(0.75)), TopP: ptr(float32(0.875)),
				TopK: ptr(int32(40)), MinP: ptr(float32(0.0625)), RepeatPenalty: ptr(float32(1.125)),
				Seed: ptr(uint64(11))}})

	// A stop string taken from the greedy text ends it there.
	stop := greedy.Choices[0].Text[5:8]
	stopped := compare("stop sequence",
		fmt.Sprintf(`{"model":"s","prompt":"Once upon a time","max_tokens":16,"temperature":0,"stop":[%q]}`, stop),
		&v1.GenerateRequest{ModelId: "s", Prompt: text("Once upon a time"), MaxTokens: 16,
			Sampling: samp(0, 1, 0), Stop: []string{stop}})
	if stopped.Choices[0].Finish != "stop" || strings.Contains(stopped.Choices[0].Text, stop) {
		t.Fatalf("stop %q: finish %q text %q", stop, stopped.Choices[0].Finish, stopped.Choices[0].Text)
	}

	compare("echo",
		`{"model":"s","prompt":"Once upon a time","max_tokens":6,"temperature":0,"echo":true}`,
		&v1.GenerateRequest{ModelId: "s", Prompt: text("Once upon a time"), MaxTokens: 6,
			Sampling: samp(0, 1, 0), EchoPrompt: true})

	compare("token ids",
		`{"model":"s","prompt":[1,2,3],"max_tokens":8,"temperature":0}`,
		&v1.GenerateRequest{ModelId: "s", MaxTokens: 8, Sampling: samp(0, 1, 0),
			Prompt: &v1.PromptInput{Input: &v1.PromptInput_TokenIds{TokenIds: &v1.TokenIDs{Ids: []int32{1, 2, 3}}}}})

	// The streamed forms: the same text and the same usage.
	stream := `{"model":"s","prompt":"Once upon a time","max_tokens":10,"temperature":0,"stream":true}`
	var oa preply
	{
		resp, err := http.Post(c.url+"/v1/completions", "application/json", strings.NewReader(stream))
		if err != nil {
			t.Fatal(err)
		}
		fr := newFrameReader(resp.Body)
		for {
			x, ok := fr.next()
			if !ok || x.data == "[DONE]" {
				break
			}
			var ch legacyResponse
			if err := json.Unmarshal([]byte(x.data), &ch); err != nil {
				t.Fatal(err)
			}
			usageOf(ch.Usage, &oa)
			for _, cc := range ch.Choices {
				for len(oa.Choices) <= cc.Index {
					oa.Choices = append(oa.Choices, pchoice{})
				}
				oa.Choices[cc.Index].Text += cc.Text
				if cc.FinishReason != nil {
					oa.Choices[cc.Index].Finish = *cc.FinishReason
				}
			}
		}
		resp.Body.Close()
	}
	nat := connectReply(t, c.inference, &v1.GenerateRequest{ModelId: "s", Prompt: text("Once upon a time"),
		MaxTokens: 10, Sampling: samp(0, 1, 0)}, true)
	sameReply(t, "streamed", nat, oa)
}

// forcedNative is forcedTools and forcedPrompt as a native chat request.
func forcedNative(t *testing.T, choice *v1.ToolChoice) *v1.GenerateRequest {
	t.Helper()
	var tools []struct {
		Function struct {
			Name, Description string
			Parameters        json.RawMessage
		}
	}
	if err := json.Unmarshal([]byte(forcedTools), &tools); err != nil {
		t.Fatal(err)
	}
	var msgs []struct{ Role, Content string }
	if err := json.Unmarshal([]byte(forcedPrompt), &msgs); err != nil {
		t.Fatal(err)
	}
	cp := &v1.ChatPrompt{AddGenerationPrompt: true, ToolChoice: choice}
	for _, m := range msgs {
		cp.Messages = append(cp.Messages, &v1.ChatMessage{Role: m.Role, Content: m.Content})
	}
	for _, x := range tools {
		cp.Tools = append(cp.Tools, &v1.Tool{Name: x.Function.Name, Description: x.Function.Description,
			ParametersJson: string(x.Function.Parameters)})
	}
	z := float32(0)
	return &v1.GenerateRequest{ModelId: "q", MaxTokens: 1024,
		Prompt:   &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: cp}},
		Sampling: &v1.SamplingParams{Temperature: &z, TopP: ptr(float32(1)), Seed: ptr(uint64(0))}}
}

func TestNativeMatchesOpenAIOnChat(t *testing.T) {
	c, _ := parityEngine(t)
	msgs := []*v1.ChatMessage{{Role: "system", Content: "You answer in one short sentence."},
		{Role: "user", Content: "Name a colour."}}
	const msgsJSON = `[{"role":"system","content":"You answer in one short sentence."},{"role":"user","content":"Name a colour."}]`
	greedy := func() *v1.SamplingParams {
		return &v1.SamplingParams{Temperature: ptr(float32(0)), TopP: ptr(float32(1)), Seed: ptr(uint64(0))}
	}
	chatReq := func(mod func(*v1.GenerateRequest)) *v1.GenerateRequest {
		m := &v1.GenerateRequest{ModelId: "q", MaxTokens: 24, Sampling: greedy(),
			Prompt: &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{Messages: msgs,
				AddGenerationPrompt: true}}}}
		if mod != nil {
			mod(m)
		}
		return m
	}
	compare := func(name, body string, m *v1.GenerateRequest, stream bool) preply {
		t.Helper()
		run := func() preply { return connectReply(t, c.inference, m, stream) }
		openai, native := both(run, func() preply { return httpChatReply(t, c.url, body, stream) }, run)
		sameReply(t, name, native, openai)
		t.Logf("%s: %+v", name, native)
		return native
	}
	base := `{"model":"q","messages":` + msgsJSON + `,"max_tokens":24,"temperature":0,"top_p":1,"seed":0`

	plain := compare("greedy", base+`}`, chatReq(nil), false)
	if plain.Choices[0].Text == "" || plain.Completion == 0 || plain.Cache == 0 {
		t.Fatalf("greedy chat: %+v (a settled request restores its prompt)", plain)
	}
	compare("greedy, logprobs", base+`,"logprobs":true,"top_logprobs":2}`,
		chatReq(func(m *v1.GenerateRequest) { m.Logprobs, m.TopLogprobs = true, ptr(int32(2)) }), false)
	compare("streamed, logprobs and usage", base+`,"logprobs":true,"top_logprobs":2,"stream":true}`,
		chatReq(func(m *v1.GenerateRequest) { m.Logprobs, m.TopLogprobs = true, ptr(int32(2)) }), true)

	sampled := compare("sampled, two choices", `{"model":"q","messages":`+msgsJSON+`,"max_tokens":16,"n":2,`+
		`"temperature":0.875,"top_p":0.9375,"top_k":50,"seed":21}`,
		chatReq(func(m *v1.GenerateRequest) {
			m.N, m.MaxTokens = 2, 16
			m.Sampling = &v1.SamplingParams{Temperature: ptr(float32(0.875)), TopP: ptr(float32(0.9375)),
				TopK: ptr(int32(50)), Seed: ptr(uint64(21))}
		}), false)
	if len(sampled.Choices) != 2 {
		t.Fatalf("%d choices", len(sampled.Choices))
	}
	compare("sampled, streamed two choices", `{"model":"q","messages":`+msgsJSON+`,"max_tokens":16,"n":2,`+
		`"temperature":0.875,"top_p":0.9375,"top_k":50,"seed":21,"stream":true}`,
		chatReq(func(m *v1.GenerateRequest) {
			m.N, m.MaxTokens = 2, 16
			m.Sampling = &v1.SamplingParams{Temperature: ptr(float32(0.875)), TopP: ptr(float32(0.9375)),
				TopK: ptr(int32(50)), Seed: ptr(uint64(21))}
		}), true)

	const schema = `{"type":"object","properties":{"colour":{"enum":["red","green","blue"]},"ok":{"type":"boolean"}},` +
		`"required":["colour","ok"]}`
	js := compare("response_format json_schema",
		`{"model":"q","messages":`+msgsJSON+`,"max_tokens":64,"temperature":0,"top_p":1,"seed":0,`+
			`"response_format":{"type":"json_schema","json_schema":{"name":"c","strict":true,"schema":`+schema+`}}}`,
		chatReq(func(m *v1.GenerateRequest) {
			m.MaxTokens = 64
			m.ResponseFormat = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_JSON_SCHEMA, Name: "c",
				Strict: ptr(true), JsonSchema: schema}
		}), false)
	var doc struct {
		Colour string
		Ok     *bool
	}
	if err := json.Unmarshal([]byte(js.Choices[0].Text), &doc); err != nil || doc.Colour == "" || doc.Ok == nil {
		t.Fatalf("the constrained reply %q is not the schema's: %v", js.Choices[0].Text, err)
	}
	compare("response_format json_object",
		base+`,"response_format":{"type":"json_object"}}`,
		chatReq(func(m *v1.GenerateRequest) {
			m.ResponseFormat = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_JSON_OBJECT}
		}), false)

	// Tools: a prompt no model answers with a call of its own accord, forced
	// into one, by name and by "required", whole and streamed.
	oaTool := func(extra string, stream bool) string {
		return fmt.Sprintf(`{"model":"q","messages":%s,"tools":%s,"max_tokens":1024,"temperature":0,"top_p":1,"seed":0%s,"stream":%v}`,
			forcedPrompt, forcedTools, extra, stream)
	}
	for _, arm := range []struct {
		name, extra string
		choice      *v1.ToolChoice
		want        string
		stream      bool
	}{
		{"required", `,"tool_choice":"required"`,
			&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_REQUIRED}, "", false},
		{"named get_time", `,"tool_choice":{"type":"function","function":{"name":"get_time"}}`,
			&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_FUNCTION, Name: "get_time"}, "get_time", true},
		{"named get_weather, one call", `,"tool_choice":{"type":"function","function":{"name":"get_weather"}},"parallel_tool_calls":false`,
			&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_FUNCTION, Name: "get_weather", SingleCall: true},
			"get_weather", false},
	} {
		r := compare(fmt.Sprintf("%s, streamed %v", arm.name, arm.stream), oaTool(arm.extra, arm.stream),
			forcedNative(t, arm.choice), arm.stream)
		ch := r.Choices[0]
		if ch.Finish != "tool_calls" || len(ch.Calls) == 0 {
			t.Fatalf("%s: finish %q calls %+v", arm.name, ch.Finish, ch.Calls)
		}
		for _, tc := range ch.Calls {
			if arm.want != "" && tc.Name != arm.want {
				t.Fatalf("%s called %q", arm.name, tc.Name)
			}
			validArgs(t, arm.name, tc.Name, tc.Args)
		}
	}

	// A violation: the same request with its tool list dropped must not make
	// a call, or the comparison above compares replies that never saw the
	// tools.
	bare := forcedNative(t, &v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_REQUIRED})
	bare.Prompt.GetChat().Tools = nil
	bare.MaxTokens = 16
	if r := connectReply(t, c.inference, bare, false); len(r.Choices[0].Calls) != 0 ||
		r.Choices[0].Finish == "tool_calls" {
		t.Fatalf("a request without tools made calls: %+v", r.Choices[0])
	}
}

// TestNativeListsWhatTheOpenAIRouteLists: the model listing carries the same
// ids, aliases and context, with the capabilities the OpenAI listing has no
// field for.
func TestNativeListsWhatTheOpenAIRouteLists(t *testing.T) {
	c, e := parityEngine(t)
	resp, err := http.Get(c.url + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var oa struct {
		Data []struct {
			ID          string   `json:"id"`
			MaxModelLen int      `json:"max_model_len"`
			Loaded      bool     `json:"loaded"`
			Aliases     []string `json:"aliases"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&oa); err != nil {
		t.Fatal(err)
	}
	lr, err := c.model.ListModels(context.Background(), connect.NewRequest(&v1.ListModelsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(lr.Msg.Loaded) != 2 {
		t.Fatalf("%d loaded models", len(lr.Msg.Loaded))
	}
	caps := map[string][]string{}
	for _, m := range lr.Msg.Loaded {
		found := false
		for _, o := range oa.Data {
			if o.ID == m.ModelId {
				found = true
				if o.MaxModelLen != int(m.DefaultMaxSeq) || o.MaxModelLen == 0 || !o.Loaded ||
					!slices.Equal(o.Aliases, m.Aliases) {
					t.Fatalf("%s: OpenAI %+v, native max seq %d aliases %v", m.ModelId, o, m.DefaultMaxSeq, m.Aliases)
				}
			}
		}
		if !found {
			t.Fatalf("%s is not in the OpenAI listing", m.ModelId)
		}
		if m.ContextLength == 0 {
			t.Fatalf("%s: no context length", m.ModelId)
		}
		caps[m.ModelId] = m.Capabilities
	}
	for _, w := range []string{"completion", "json_schema", "logprobs"} {
		if !slices.Contains(caps["s"], w) || !slices.Contains(caps["q"], w) {
			t.Fatalf("capabilities %v / %v lack %s", caps["s"], caps["q"], w)
		}
	}
	if slices.Contains(caps["s"], "chat") || slices.Contains(caps["s"], "tools") {
		t.Fatalf("stories260K has no template, yet is %v", caps["s"])
	}
	if !slices.Contains(caps["q"], "chat") || !slices.Contains(caps["q"], "tools") {
		t.Fatalf("Qwen3 is %v", caps["q"])
	}

	// An unloaded container in the folder is listed with what its header
	// states, without loading it.
	if _, err := e.UnloadModel("s", true); err != nil {
		t.Fatal(err)
	}
	lr, err = c.model.ListModels(context.Background(), connect.NewRequest(&v1.ListModelsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	var file *v1.ModelFile
	for _, f := range lr.Msg.Files {
		if f.Name == smallModel || strings.TrimSuffix(f.Name, ".jlm") == strings.TrimSuffix(smallModel, ".jlm") {
			file = f
		}
	}
	if file == nil || file.ContextLength == 0 || !slices.Contains(file.Capabilities, "completion") ||
		slices.Contains(file.Capabilities, "chat") {
		t.Fatalf("the unloaded file is listed as %+v", file)
	}
}

// TestNativeEmbedMatchesOpenAI: the same vectors and token counts through
// either door.
func TestNativeEmbedMatchesOpenAI(t *testing.T) {
	_, lm, url := embedServer(t, embedGateModels()[0], Config{})
	cl := jitllmv1connect.NewInferenceServiceClient(http.DefaultClient, url)
	texts := []string{embedTexts()[0], embedTexts()[2]}
	code, oa := postEmbeddings(t, url, map[string]any{"model": "emb", "input": texts})
	if code != http.StatusOK {
		t.Fatalf("OpenAI embeddings: %d", code)
	}
	req := &v1.EmbedRequest{ModelId: "emb"}
	for _, s := range texts {
		req.Inputs = append(req.Inputs, &v1.EmbedInput{Input: &v1.EmbedInput_Text{Text: s}})
	}
	nat, err := cl.Embed(context.Background(), connect.NewRequest(req))
	if err != nil {
		t.Fatal(err)
	}
	if len(nat.Msg.Embeddings) != len(oa.Data) || len(oa.Data) != 2 {
		t.Fatalf("%d native vectors, %d OpenAI", len(nat.Msg.Embeddings), len(oa.Data))
	}
	for i, e := range nat.Msg.Embeddings {
		var want []float32
		if err := json.Unmarshal(oa.Data[i].Embedding, &want); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(e.Vector, want) || len(e.Vector) != lm.m.Cfg.NEmbd {
			t.Fatalf("vector %d differs between the surfaces", i)
		}
	}
	if int(nat.Msg.TotalTokens) != oa.Usage.TotalTokens || oa.Usage.TotalTokens == 0 {
		t.Fatalf("tokens: native %d, OpenAI %d", nat.Msg.TotalTokens, oa.Usage.TotalTokens)
	}
}
