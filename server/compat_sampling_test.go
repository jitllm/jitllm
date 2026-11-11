package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestNoSamplingFieldMeansTheAPIsDefaults: a request naming no sampling field
// samples at temperature 1 with top_p 1 on both shims, as the APIs document
// and every other engine does, so one request body gets one sampler wherever
// it is sent. Greedy is temperature 0. A named session with no sampling field
// keeps its own sampler.
func TestNoSamplingFieldMeansTheAPIsDefaults(t *testing.T) {
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
	msgs := `"messages":[{"role":"user","content":"hi"}]`
	for _, c := range []struct{ path, body string }{
		{oaChat, `{"model":"m-1",` + msgs + `}`},
		{oaText, `{"model":"m-1","prompt":"hi"}`},
		{anMsgs, `{"model":"m-1","max_tokens":8,` + msgs + `}`},
	} {
		o := send(c.path, c.body)
		if o.Sampling == nil || o.Sampling.Temp != 1 || o.Sampling.TopP != 1 || o.Sampling.TopK != 0 {
			t.Fatalf("%s with no sampling field ran %+v, want temperature 1, top_p 1", c.path, o.Sampling)
		}
	}
	// Two unseeded requests do not draw from one seed.
	a, b := send(oaChat, `{"model":"m-1",`+msgs+`}`), send(oaChat, `{"model":"m-1",`+msgs+`}`)
	if a.Sampling.Seed == b.Sampling.Seed {
		t.Fatalf("two unseeded requests share seed %d", a.Sampling.Seed)
	}
	if o := send(oaChat, `{"model":"m-1","temperature":0,`+msgs+`}`); o.Sampling == nil || o.Sampling.Temp != 0 {
		t.Fatalf("temperature 0 ran %+v, want greedy", o.Sampling)
	}
	if o := send(oaChat, `{"model":"m-1","top_k":-1,`+msgs+`}`); o.Sampling.TopK != 0 {
		t.Fatalf("top_k -1 ran top_k %d, want disabled", o.Sampling.TopK)
	}
	for _, path := range []string{oaChat, anMsgs} {
		if o := send(path, `{"jitllm_session":"sess-7","max_tokens":8,`+msgs+`}`); o.Sampling != nil {
			t.Fatalf("%s: a named session with no sampling field was overridden with %+v", path, o.Sampling)
		}
	}
}

// TestUnsupportedSamplingIsRefusedNotIgnored: a field the engine cannot honour
// is a 400 naming it. Ignoring it would send the client an answer to a request
// it did not make, and a benchmark a number for a sampler it did not ask for.
func TestUnsupportedSamplingIsRefusedNotIgnored(t *testing.T) {
	s := serve(t, &fakeBackend{tokens: []string{"x"}, reason: FinishEOS})
	msgs := `"messages":[{"role":"user","content":"hi"}]`
	for _, c := range []struct{ path, body, want string }{
		// logprobs and n are supported (compat_choices.go); what stays refused
		// is outside their range or their shape.
		{oaChat, `{"model":"m-1","n":0,` + msgs + `}`, "n must be"},
		{oaChat, `{"model":"m-1","n":129,` + msgs + `}`, "n must be"},
		{oaChat, `{"model":"m-1","logprobs":5,` + msgs + `}`, "logprobs"},
		{oaChat, `{"model":"m-1","top_logprobs":3,` + msgs + `}`, "top_logprobs"},
		{oaChat, `{"model":"m-1","logprobs":true,"top_logprobs":21,` + msgs + `}`, "top_logprobs"},
		{oaText, `{"model":"m-1","logprobs":21,"prompt":"hi"}`, "logprobs"},
		{oaText, `{"model":"m-1","logprobs":true,"prompt":"hi"}`, "logprobs"},
		{oaText, `{"model":"m-1","n":2,"prompt":["a","b"]}`, "list of prompts"},
		{oaChat, `{"model":"m-1","presence_penalty":0.5,` + msgs + `}`, "presence_penalty"},
		{oaText, `{"model":"m-1","frequency_penalty":0.5,"prompt":"hi"}`, "frequency_penalty"},
		{oaChat, `{"model":"m-1","temperature":-1,` + msgs + `}`, "temperature"},
		{oaChat, `{"model":"m-1","top_p":1.5,` + msgs + `}`, "top_p"},
		{oaChat, `{"model":"m-1","min_p":2,` + msgs + `}`, "min_p"},
		{oaChat, `{"model":"m-1","repetition_penalty":0,` + msgs + `}`, "repetition_penalty"},
		{oaText, `{"model":"m-1","stream":true,"prompt":["a","b"]}`, "list of prompts"},
		{anMsgs, `{"model":"m-1","max_tokens":8,"temperature":-0.5,` + msgs + `}`, "temperature"},
	} {
		resp := post(t, s, c.path, c.body)
		e := readErr(t, resp)
		if resp.StatusCode != http.StatusBadRequest || !strings.Contains(e.Error.Message, c.want) {
			t.Errorf("%s %s: %d %q, want a 400 naming %q", c.path, c.body, resp.StatusCode, e.Error.Message, c.want)
		}
	}
}

// TestAListOfPromptsIsAChoiceEach: /v1/completions with several prompts
// answers every one, in order, never only the first.
func TestAListOfPromptsIsAChoiceEach(t *testing.T) {
	f := &fakeBackend{tokens: []string{"a", "b"}, reason: FinishMaxTokens}
	s := serve(t, f)
	for _, prompt := range []string{`["one","two","three"]`, `[[1,2],[3],[4,5,6]]`} {
		resp := post(t, s, oaText, `{"model":"m-1","max_tokens":2,"prompt":`+prompt+`}`)
		var out struct {
			Choices []struct {
				Index        int    `json:"index"`
				Text         string `json:"text"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage oaUsage `json:"usage"`
		}
		json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
		if len(out.Choices) != 3 {
			t.Fatalf("%s: %d choices, want 3", prompt, len(out.Choices))
		}
		for i, c := range out.Choices {
			if c.Index != i || c.Text != "ab" || c.FinishReason != "length" {
				t.Fatalf("%s: choice %d is %+v", prompt, i, c)
			}
		}
		if out.Usage.PromptTokens != 21 || out.Usage.CompletionTokens != 6 || out.Usage.TotalTokens != 27 {
			t.Fatalf("%s: usage %+v, want the three generates summed", prompt, out.Usage)
		}
	}
}

// TestTheStreamsLastChunkAlwaysCarriesUsage: a harness computes its token
// count from the final chunk's usage, so it is there whether or not the
// client sent stream_options, on both OpenAI routes, and counts every token.
func TestTheStreamsLastChunkAlwaysCarriesUsage(t *testing.T) {
	s := serve(t, &fakeBackend{tokens: []string{"a", "b", "c"}, reason: FinishMaxTokens})
	for _, c := range []struct{ path, body string }{
		{oaChat, `{"model":"m-1","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		{oaChat, `{"model":"m-1","stream":true,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"hi"}]}`},
		{oaText, `{"model":"m-1","stream":true,"prompt":"hi"}`},
	} {
		frames := collectFrames(t, post(t, s, c.path, c.body))
		if len(frames) < 2 || frames[len(frames)-1].data != "[DONE]" {
			t.Fatalf("%s: stream did not end with [DONE]: %+v", c.body, frames)
		}
		var last struct {
			Usage *oaUsage `json:"usage"`
		}
		json.Unmarshal([]byte(frames[len(frames)-2].data), &last)
		if last.Usage == nil || last.Usage.CompletionTokens != 3 || last.Usage.PromptTokens != 7 || last.Usage.TotalTokens != 10 {
			t.Fatalf("%s: last chunk's usage %+v", c.body, last.Usage)
		}
	}
}

// TestHealthAndModelLength: /health answers beside /healthz, and /v1/models
// carries each model's context as max_model_len.
func TestHealthAndModelLength(t *testing.T) {
	for _, p := range []string{"/health", "/healthz"} {
		rec := httptest.NewRecorder()
		(&Engine{}).Handler().ServeHTTP(rec, httptest.NewRequest("GET", p, nil))
		if rec.Code != 200 {
			t.Fatalf("%s: %d", p, rec.Code)
		}
	}
	s := serve(t, &fakeBackend{})
	resp, err := http.Get(s.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var ml struct {
		Data []struct {
			ID          string `json:"id"`
			MaxModelLen int    `json:"max_model_len"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&ml)
	resp.Body.Close()
	if len(ml.Data) == 0 {
		t.Fatal("no models listed")
	}
	for _, d := range ml.Data {
		if d.MaxModelLen != 2048 {
			t.Fatalf("%s: max_model_len %d, want 2048", d.ID, d.MaxModelLen)
		}
	}
}
