package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// The native Connect API and the OpenAI-compatible routes are two doors on
// one generate path. These gates hold that: the same request through either
// builds the same GenerateOptions (the first half, on a scripted backend, so
// every field's mapping is seen), and on a real engine gives the same text,
// finish reason, usage, logprobs and tool calls (parity_model_test.go). A
// value one surface refuses, the other refuses in the same words.

// pair is both surfaces on one scripted backend.
type pair struct {
	f   *fakeBackend
	url string
	cl  jitllmv1connect.InferenceServiceClient
}

func newPair(t *testing.T) *pair {
	t.Helper()
	f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
	mux := http.NewServeMux()
	path, h := jitllmv1connect.NewInferenceServiceHandler(&InferenceService{B: f})
	mux.Handle(path, h)
	mux.Handle("/v1/", CompatHandler(f))
	s := httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return &pair{f: f, url: s.URL, cl: jitllmv1connect.NewInferenceServiceClient(http.DefaultClient, s.URL)}
}

// viaHTTP posts body and returns the options the backend received, or the
// refusal's status and message.
func (p *pair) viaHTTP(t *testing.T, route, body string) (GenerateOptions, int, string) {
	t.Helper()
	p.f.mu.Lock()
	p.f.last = GenerateOptions{}
	p.f.mu.Unlock()
	resp, err := http.Post(p.url+route, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		var e oaError
		if json.Unmarshal(b, &e) != nil {
			t.Fatalf("%s: %d %s", route, resp.StatusCode, b)
		}
		return GenerateOptions{}, resp.StatusCode, e.Error.Message
	}
	return p.f.opts(), resp.StatusCode, ""
}

// viaConnect sends m to Complete and returns the options the backend received,
// or the refusal.
func (p *pair) viaConnect(t *testing.T, m *v1.GenerateRequest) (GenerateOptions, error) {
	t.Helper()
	p.f.mu.Lock()
	p.f.last = GenerateOptions{}
	p.f.mu.Unlock()
	if _, err := p.cl.Complete(context.Background(), connect.NewRequest(m)); err != nil {
		return GenerateOptions{}, err
	}
	return p.f.opts(), nil
}

// chat is a native chat request for model m-1 with the sampling every OpenAI
// body below also states, so neither side draws a random seed.
func chat(msgs []*v1.ChatMessage, mod func(*v1.GenerateRequest, *v1.ChatPrompt)) *v1.GenerateRequest {
	cp := &v1.ChatPrompt{Messages: msgs, AddGenerationPrompt: true}
	m := &v1.GenerateRequest{
		ModelId: "m-1",
		Prompt:  &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: cp}},
		Sampling: &v1.SamplingParams{Temperature: ptr(float32(0.5)), TopP: ptr(float32(1)),
			Seed: ptr(uint64(3))},
	}
	if mod != nil {
		mod(m, cp)
	}
	return m
}

func ptr[T any](v T) *T { return &v }

const (
	parityMsgs = `[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]`
	// every OpenAI body states this, as chat() does for Connect.
	paritySampling = `"temperature":0.5,"top_p":1,"seed":3`
)

func parityMessages() []*v1.ChatMessage {
	return []*v1.ChatMessage{{Role: "system", Content: "be brief"}, {Role: "user", Content: "hi"}}
}

// pngDataURL is a small picture as an OpenAI client sends it.
func pngBytes(t *testing.T) []byte {
	t.Helper()
	im := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for i := 0; i < 8; i++ {
		im.Set(i, i, color.RGBA{R: 200, A: 255})
	}
	var b bytes.Buffer
	if err := png.Encode(&b, im); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// TestNativeRequestsBuildTheOptionsTheOpenAIRoutesDo: each request pair, one
// OpenAI body and the Connect request that states the same thing, reaches the
// engine as the same GenerateOptions. A field the Connect mapping dropped or
// misread shows as the difference.
func TestNativeRequestsBuildTheOptionsTheOpenAIRoutesDo(t *testing.T) {
	const weather = `{"type":"function","function":{"name":"get_weather","description":"Get it",` +
		`"parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}`
	weatherTool := &v1.Tool{Name: "get_weather", Description: "Get it",
		ParametersJson: `{"type":"object","properties":{"location":{"type":"string"}}}`}
	withTools := func(c *v1.ToolChoice) func(*v1.GenerateRequest, *v1.ChatPrompt) {
		return func(_ *v1.GenerateRequest, cp *v1.ChatPrompt) {
			cp.Tools, cp.ToolChoice = []*v1.Tool{weatherTool}, c
		}
	}
	toolBody := func(extra string) string {
		return `{"model":"m-1","messages":` + parityMsgs + `,` + paritySampling + `,"tools":[` + weather + `]` + extra + `}`
	}
	imgBytes := pngBytes(t)
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgBytes)
	oneImage := func(img *v1.ChatImage) *v1.GenerateRequest {
		return chat([]*v1.ChatMessage{{Role: "user", Content: "what is this", Images: []*v1.ChatImage{img}}}, nil)
	}
	imageBody := `{"model":"m-1",` + paritySampling + `,"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"` + dataURL + `"}}]}]}`

	cases := []struct {
		name, route, body string
		native            *v1.GenerateRequest
	}{
		{"messages and sampling", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,"temperature":0.5,"top_p":0.75,"top_k":40,"min_p":0.25,` +
				`"repetition_penalty":1.5,"seed":7,"max_tokens":12,"stop":["\n","END"],"ignore_eos":true}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.Sampling = &v1.SamplingParams{Temperature: ptr(float32(0.5)), TopP: ptr(float32(0.75)),
					TopK: ptr(int32(40)), MinP: ptr(float32(0.25)), RepeatPenalty: ptr(float32(1.5)),
					Seed: ptr(uint64(7))}
				m.MaxTokens, m.Stop, m.IgnoreEos = 12, []string{"\n", "END"}, true
			})},
		{"top_k -1 disables", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,"temperature":0.5,"top_p":1,"top_k":-1,"seed":3}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.Sampling.TopK = ptr(int32(-1))
			})},
		{"tools, no choice", "/v1/chat/completions", toolBody(""), chat(parityMessages(), withTools(nil))},
		{"tool_choice auto", "/v1/chat/completions", toolBody(`,"tool_choice":"auto"`),
			chat(parityMessages(), withTools(&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_AUTO}))},
		{"tool_choice none withholds the tools", "/v1/chat/completions", toolBody(`,"tool_choice":"none"`),
			chat(parityMessages(), withTools(&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_NONE}))},
		{"tool_choice required", "/v1/chat/completions", toolBody(`,"tool_choice":"required"`),
			chat(parityMessages(), withTools(&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_REQUIRED}))},
		{"tool_choice named, one call", "/v1/chat/completions",
			toolBody(`,"tool_choice":{"type":"function","function":{"name":"get_weather"}},"parallel_tool_calls":false`),
			chat(parityMessages(), withTools(&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_FUNCTION,
				Name: "get_weather", SingleCall: true}))},
		{"parallel_tool_calls false alone", "/v1/chat/completions", toolBody(`,"parallel_tool_calls":false`),
			chat(parityMessages(), withTools(&v1.ToolChoice{SingleCall: true}))},
		{"tool turns", "/v1/chat/completions",
			`{"model":"m-1",` + paritySampling + `,"messages":[{"role":"user","content":"weather?"},` +
				`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":` +
				`{"name":"get_weather","arguments":"{\"location\":\"Rome\"}"}}]},` +
				`{"role":"tool","tool_call_id":"call_1","name":"get_weather","content":"sunny"}]}`,
			chat([]*v1.ChatMessage{{Role: "user", Content: "weather?"},
				{Role: "assistant", ToolCalls: []*v1.ToolCall{{Id: "call_1", Name: "get_weather",
					Arguments: `{"location":"Rome"}`}}},
				{Role: "tool", ToolCallId: "call_1", Name: "get_weather", Content: "sunny"}}, nil)},
		{"response_format json_object", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,` + paritySampling + `,"response_format":{"type":"json_object"}}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.ResponseFormat = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_JSON_OBJECT}
			})},
		{"response_format json_schema", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,` + paritySampling + `,"response_format":{"type":"json_schema",` +
				`"json_schema":{"name":"x","strict":true,"schema":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}}}}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.ResponseFormat = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_JSON_SCHEMA, Name: "x",
					Strict:     ptr(true),
					JsonSchema: `{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`}
			})},
		{"response_format text", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,` + paritySampling + `,"response_format":{"type":"text"}}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.ResponseFormat = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_TEXT}
			})},
		{"n, logprobs, top_logprobs", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,"temperature":0.5,"top_p":1,"seed":5,"n":3,` +
				`"logprobs":true,"top_logprobs":3}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.Sampling.Seed = ptr(uint64(5))
				m.N, m.Logprobs, m.TopLogprobs = 3, true, ptr(int32(3))
			})},
		{"logprobs without alternatives", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,` + paritySampling + `,"logprobs":true}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) { m.Logprobs = true })},
		{"jitllm extensions", "/v1/chat/completions",
			`{"model":"m-1","messages":` + parityMsgs + `,` + paritySampling + `,"jitllm_priority":"high","jitllm_speculate":true}`,
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.Priority, m.Speculation = "high", &v1.SpeculationParams{Enabled: true}
			})},
		{"a picture as a data URL, by url", "/v1/chat/completions", imageBody,
			oneImage(&v1.ChatImage{Url: dataURL})},
		{"a picture as a data URL, by bytes", "/v1/chat/completions", imageBody,
			oneImage(&v1.ChatImage{Data: imgBytes, MediaType: "image/png"})},
		{"completion, text", "/v1/completions",
			`{"model":"m-1","prompt":"Once","max_tokens":5,"echo":true,` + paritySampling + `}`,
			&v1.GenerateRequest{ModelId: "m-1", MaxTokens: 5, EchoPrompt: true,
				Prompt: &v1.PromptInput{Input: &v1.PromptInput_Text{Text: "Once"}},
				Sampling: &v1.SamplingParams{Temperature: ptr(float32(0.5)), TopP: ptr(float32(1)),
					Seed: ptr(uint64(3))}}},
		{"completion, token ids, legacy logprobs", "/v1/completions",
			`{"model":"m-1","prompt":[1,2,3],"logprobs":2,` + paritySampling + `}`,
			&v1.GenerateRequest{ModelId: "m-1", Logprobs: true, TopLogprobs: ptr(int32(2)),
				Prompt: &v1.PromptInput{Input: &v1.PromptInput_TokenIds{TokenIds: &v1.TokenIDs{Ids: []int32{1, 2, 3}}}},
				Sampling: &v1.SamplingParams{Temperature: ptr(float32(0.5)), TopP: ptr(float32(1)),
					Seed: ptr(uint64(3))}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := newPair(t)
			want, status, msg := p.viaHTTP(t, c.route, c.body)
			if status != http.StatusOK {
				t.Fatalf("the OpenAI body was refused: %d %s", status, msg)
			}
			got, err := p.viaConnect(t, c.native)
			if err != nil {
				t.Fatalf("the Connect request was refused: %v", err)
			}
			// The chat route always asks for a generation prompt; the model is
			// named the same.
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Connect built\n  %+v\nthe OpenAI route built\n  %+v", got, want)
			}
			// The comparison must be of something: a request that built no
			// prompt would equal itself.
			if got.Prompt.Kind == PromptChat && got.Prompt.Chat == nil || got.Prompt.Kind == 0 && got.Prompt.Text == "" &&
				got.Prompt.Chat == nil && len(got.Prompt.IDs) == 0 {
				t.Fatalf("no prompt reached the backend: %+v", got.Prompt)
			}
		})
	}
}

// TestParityOptionsAreSensitive runs the gate above against a violation: one
// field changed on the Connect side must fail the comparison, or the
// comparison compares nothing.
func TestParityOptionsAreSensitive(t *testing.T) {
	p := newPair(t)
	want, _, _ := p.viaHTTP(t, "/v1/chat/completions",
		`{"model":"m-1","messages":`+parityMsgs+`,`+paritySampling+`,"max_tokens":9}`)
	got, err := p.viaConnect(t, chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) { m.MaxTokens = 10 }))
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(got, want) {
		t.Fatal("max_tokens 9 and 10 compared equal")
	}
	got, _ = p.viaConnect(t, chat(parityMessages(), func(m *v1.GenerateRequest, cp *v1.ChatPrompt) {
		m.MaxTokens = 9
		cp.Tools = []*v1.Tool{{Name: "f"}}
	}))
	if reflect.DeepEqual(got, want) {
		t.Fatal("a request with tools compared equal to one without")
	}
}

// TestNativeRefusesWhatTheOpenAIRoutesRefuse: a value the engine does not
// take is refused by name on both surfaces, in the same words, as a 400 and
// as InvalidArgument.
func TestNativeRefusesWhatTheOpenAIRoutesRefuse(t *testing.T) {
	const weather = `{"type":"function","function":{"name":"get_weather"}}`
	chatBody := func(extra string) string {
		return `{"model":"m-1","messages":` + parityMsgs + `,` + extra + `}`
	}
	toolsAnd := func(choice *v1.ToolChoice) func(*v1.GenerateRequest, *v1.ChatPrompt) {
		return func(_ *v1.GenerateRequest, cp *v1.ChatPrompt) {
			cp.Tools, cp.ToolChoice = []*v1.Tool{{Name: "get_weather"}}, choice
		}
	}
	samp := func(f func(*v1.SamplingParams)) func(*v1.GenerateRequest, *v1.ChatPrompt) {
		return func(m *v1.GenerateRequest, _ *v1.ChatPrompt) { m.Sampling = &v1.SamplingParams{}; f(m.Sampling) }
	}
	cases := []struct {
		name, route, body string
		native            *v1.GenerateRequest
		contains          string
	}{
		{"presence_penalty above 2", "/v1/chat/completions", chatBody(`"presence_penalty":3`),
			chat(parityMessages(), samp(func(s *v1.SamplingParams) { s.PresencePenalty = ptr(float32(3)) })),
			"outside [-2, 2]"},
		{"frequency_penalty below -2", "/v1/chat/completions", chatBody(`"frequency_penalty":-3`),
			chat(parityMessages(), samp(func(s *v1.SamplingParams) { s.FrequencyPenalty = ptr(float32(-3)) })),
			"outside [-2, 2]"},
		{"negative temperature", "/v1/chat/completions", chatBody(`"temperature":-1`),
			chat(parityMessages(), samp(func(s *v1.SamplingParams) { s.Temperature = ptr(float32(-1)) })), "is negative"},
		{"top_p above 1", "/v1/chat/completions", chatBody(`"top_p":2`),
			chat(parityMessages(), samp(func(s *v1.SamplingParams) { s.TopP = ptr(float32(2)) })), "outside [0, 1]"},
		{"top_k below -1", "/v1/chat/completions", chatBody(`"top_k":-2`),
			chat(parityMessages(), samp(func(s *v1.SamplingParams) { s.TopK = ptr(int32(-2)) })), "below -1"},
		{"min_p above 1", "/v1/chat/completions", chatBody(`"min_p":2`),
			chat(parityMessages(), samp(func(s *v1.SamplingParams) { s.MinP = ptr(float32(2)) })), "outside [0, 1]"},
		{"repetition penalty negative", "/v1/chat/completions", chatBody(`"repetition_penalty":-1`),
			chat(parityMessages(), samp(func(s *v1.SamplingParams) { s.RepeatPenalty = ptr(float32(-1)) })),
			"is not positive"},
		{"n too large", "/v1/chat/completions", chatBody(`"n":129`),
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) { m.N = 129 }), "n must be between"},
		{"n negative", "/v1/chat/completions", chatBody(`"n":-1`),
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) { m.N = -1 }), "n must be between"},
		{"top_logprobs without logprobs", "/v1/chat/completions", chatBody(`"top_logprobs":2`),
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) { m.TopLogprobs = ptr(int32(2)) }),
			"top_logprobs needs logprobs"},
		{"top_logprobs too large", "/v1/chat/completions", chatBody(`"logprobs":true,"top_logprobs":21`),
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.Logprobs, m.TopLogprobs = true, ptr(int32(21))
			}), "top_logprobs must be between"},
		{"tool_choice names an undeclared tool", "/v1/chat/completions",
			chatBody(`"tools":[` + weather + `],"tool_choice":{"type":"function","function":{"name":"get_time"}}`),
			chat(parityMessages(), toolsAnd(&v1.ToolChoice{Mode: v1.ToolChoiceMode_TOOL_CHOICE_MODE_FUNCTION,
				Name: "get_time"})),
			`names "get_time"`},
		{"a tool with no name", "/v1/chat/completions",
			chatBody(`"tools":[{"type":"function","function":{"name":""}}]`),
			chat(parityMessages(), func(_ *v1.GenerateRequest, cp *v1.ChatPrompt) { cp.Tools = []*v1.Tool{{}} }),
			"only"},
		{"an unsupported schema", "/v1/chat/completions",
			chatBody(`"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","not":{"type":"string"}}}}`),
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.ResponseFormat = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_JSON_SCHEMA,
					JsonSchema: `{"type":"object","not":{"type":"string"}}`}
			}), "response_format"},
		{"a schema format without a schema", "/v1/chat/completions",
			chatBody(`"response_format":{"type":"json_schema"}`),
			chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
				m.ResponseFormat = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_JSON_SCHEMA}
			}), "needs json_schema.schema"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := newPair(t)
			_, status, msg := p.viaHTTP(t, c.route, c.body)
			if status != http.StatusBadRequest {
				t.Fatalf("the OpenAI route answered %d %q, want 400", status, msg)
			}
			if !strings.Contains(msg, c.contains) {
				t.Fatalf("the OpenAI route refused with %q, which lacks %q", msg, c.contains)
			}
			_, err := p.viaConnect(t, c.native)
			ce := wantCode(t, "Complete", err, connect.CodeInvalidArgument)
			if ce.Message() != msg {
				t.Fatalf("Connect refused with\n  %q\nthe OpenAI route with\n  %q", ce.Message(), msg)
			}
		})
	}
}

// TestNativeRefusesAPictureTheSameWay: a picture named by a remote URL is
// refused unless the host is allowed, an unknown scheme is refused, and a
// request giving both bytes and a URL is refused, with the cause the OpenAI
// route gives.
func TestNativeRefusesAPictureTheSameWay(t *testing.T) {
	p := newPair(t)
	for _, u := range []string{"http://example.invalid/a.png", "https://example.invalid/a.png", "file:///etc/passwd"} {
		_, status, msg := p.viaHTTP(t, "/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[`+
			`{"type":"image_url","image_url":{"url":"`+u+`"}}]}]}`)
		if status != http.StatusBadRequest {
			t.Fatalf("%s: OpenAI route %d %q", u, status, msg)
		}
		_, err := p.viaConnect(t, chat([]*v1.ChatMessage{{Role: "user", Images: []*v1.ChatImage{{Url: u}}}}, nil))
		ce := wantCode(t, u, err, connect.CodeInvalidArgument)
		// Each names its own part of the request; the cause that follows is
		// the same words.
		cause := func(s string) string {
			_, rest, _ := strings.Cut(strings.TrimPrefix(s, "generate: "), " ")
			return rest
		}
		if cause(ce.Message()) != cause(msg) || cause(msg) == "" {
			t.Fatalf("%s: Connect %q, OpenAI %q", u, ce.Message(), msg)
		}
	}
	_, err := p.viaConnect(t, chat([]*v1.ChatMessage{{Role: "user",
		Images: []*v1.ChatImage{{Url: "data:image/png;base64,AA==", Data: []byte{1}}}}}, nil))
	ce := wantCode(t, "data and url", err, connect.CodeInvalidArgument)
	if !strings.Contains(ce.Message(), "both data and url") {
		t.Fatalf("message %q", ce.Message())
	}
}

// TestNativeConstraintsAreOneEach: a response_format, a grammar and a
// json_schema are one constraint each, and a forced tool call is a fourth.
func TestNativeConstraintsAreOneEach(t *testing.T) {
	p := newPair(t)
	rf := &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_JSON_OBJECT}
	for name, mod := range map[string]func(*v1.GenerateRequest, *v1.ChatPrompt){
		"response_format and grammar": func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
			m.ResponseFormat, m.Grammar = rf, `root ::= "a"`
		},
		"response_format and json_schema": func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
			m.ResponseFormat, m.JsonSchema = rf, `{"type":"object"}`
		},
	} {
		_, err := p.viaConnect(t, chat(parityMessages(), mod))
		ce := wantCode(t, name, err, connect.CodeInvalidArgument)
		if !strings.Contains(ce.Message(), "one constraint each") {
			t.Fatalf("%s: %q", name, ce.Message())
		}
	}
	// text constrains nothing, so it sits beside a grammar.
	if _, err := p.viaConnect(t, chat(parityMessages(), func(m *v1.GenerateRequest, _ *v1.ChatPrompt) {
		m.ResponseFormat, m.Grammar = &v1.ResponseFormat{Type: v1.ResponseFormat_TYPE_TEXT}, `root ::= "a"`
	})); err != nil {
		t.Fatalf("text beside a grammar: %v", err)
	}
}

// TestEmbedRefusesAsTheOpenAIRouteDoes: empty inputs and a negative width are
// refused in the same words before any model is asked.
func TestEmbedRefusesAsTheOpenAIRouteDoes(t *testing.T) {
	p := newPair(t)
	post := func(body string) string {
		resp, err := http.Post(p.url+"/v1/embeddings", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var e oaError
		b, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusBadRequest || json.Unmarshal(b, &e) != nil {
			t.Fatalf("%d %s", resp.StatusCode, b)
		}
		return e.Error.Message
	}
	text := func(s string) *v1.EmbedInput { return &v1.EmbedInput{Input: &v1.EmbedInput_Text{Text: s}} }
	for _, c := range []struct {
		body string
		req  *v1.EmbedRequest
	}{
		{`{"model":"e","input":[]}`, &v1.EmbedRequest{ModelId: "e"}},
		{`{"model":"e","input":["a",""]}`, &v1.EmbedRequest{ModelId: "e", Inputs: []*v1.EmbedInput{text("a"), text("")}}},
		{`{"model":"e","input":["a"],"dimensions":-1}`,
			&v1.EmbedRequest{ModelId: "e", Inputs: []*v1.EmbedInput{text("a")}, Dimensions: -1}},
	} {
		want := post(c.body)
		_, err := p.cl.Embed(context.Background(), connect.NewRequest(c.req))
		ce := wantCode(t, c.body, err, connect.CodeInvalidArgument)
		if ce.Message() != want {
			t.Fatalf("%s: Connect %q, OpenAI %q", c.body, ce.Message(), want)
		}
	}
}
