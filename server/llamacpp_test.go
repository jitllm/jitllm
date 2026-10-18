package server

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
)

// llamaFake is the scripted backend with a byte tokenizer (id 1 is the BOS,
// every byte b is id 256+b), so the llama.cpp routes run without a model.
type llamaFake struct {
	*fakeBackend
	fim bool
}

func (f *llamaFake) LlamaModel(name string) (LlamaModelInfo, error) {
	info := LlamaModelInfo{ID: "m-1", Name: "tinyllama", Path: "/models/tiny.jlm", NCtx: 2048,
		ChatTemplate: "{{ messages }}", Version: "test"}
	if f.fim {
		info.FIM = &FIMTokens{Pre: 10, Suf: 11, Mid: 12, Repo: -1, Sep: -1}
	}
	return info, nil
}

func (f *llamaFake) Tokenize(_, text string, addSpecial, _ bool) ([]int32, error) {
	var ids []int32
	if addSpecial {
		ids = append(ids, 1)
	}
	for _, b := range []byte(text) {
		ids = append(ids, 256+int32(b))
	}
	return ids, nil
}

func (f *llamaFake) Piece(_ string, id int32) ([]byte, error) { return []byte{byte(id - 256)}, nil }

func (f *llamaFake) Detokenize(_ string, ids []int32) (string, error) {
	var b []byte
	for _, id := range ids {
		if id >= 256 {
			b = append(b, byte(id-256))
		}
	}
	return string(b), nil
}

func (f *llamaFake) ApplyTemplate(_ string, msgs []model.ChatMessage, _ []byte) (string, error) {
	var sb strings.Builder
	for _, m := range msgs {
		sb.WriteString("<" + m.Role + ">" + m.Content + "\n")
	}
	return sb.String() + "<assistant>", nil
}

func (f *llamaFake) Slots(string) []LlamaSlot { return []LlamaSlot{{ID: "s1", NCtx: 2048}} }

func llamaServe(t *testing.T, f *llamaFake) *httptest.Server {
	t.Helper()
	return serve(t, f)
}

func llamaPost(t *testing.T, s *httptest.Server, path, body string) (int, map[string]any) {
	t.Helper()
	resp := post(t, s, path, body)
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var out map[string]any
	json.Unmarshal(b, &out)
	if out == nil {
		out = map[string]any{"raw": string(b)}
	}
	return resp.StatusCode, out
}

func TestLlamaCompletionShape(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"Hel", "lo", "\n"}, reason: FinishMaxTokens}}
	s := llamaServe(t, f)
	code, out := llamaPost(t, s, "/completion", `{"prompt":"Hi","n_predict":3,"return_tokens":true}`)
	if code != 200 {
		t.Fatalf("%d %v", code, out)
	}
	if out["content"] != "Hello\n" || out["stop"] != true || out["stop_type"] != "limit" {
		t.Fatalf("content/stop/stop_type: %v", out)
	}
	if out["tokens_predicted"] != float64(3) || out["tokens_evaluated"] != float64(7) {
		t.Fatalf("counts: %v", out)
	}
	if toks, _ := out["tokens"].([]any); len(toks) != 3 {
		t.Fatalf("return_tokens: %v", out["tokens"])
	}
	for _, k := range []string{"timings", "generation_settings", "model", "prompt", "id_slot", "stopping_word", "tokens_cached", "truncated", "has_new_line", "index"} {
		if _, ok := out[k]; !ok {
			t.Errorf("missing %s: %v", k, out)
		}
	}
	tm := out["timings"].(map[string]any)
	for _, k := range []string{"cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second", "prompt_per_second"} {
		if _, ok := tm[k]; !ok {
			t.Errorf("timings.%s missing", k)
		}
	}
	if gs := out["generation_settings"].(map[string]any); gs["n_predict"] != float64(3) || gs["n_ctx"] != float64(2048) {
		t.Fatalf("generation_settings: %v", gs)
	}
	// Prompt "Hi" is tokenized with the BOS, as llama-server does.
	if got := f.opts().Prompt.IDs; len(got) != 3 || got[0] != 1 {
		t.Fatalf("prompt ids: %v", got)
	}
	if f.opts().MaxTokens != 3 {
		t.Fatalf("n_predict not mapped: %d", f.opts().MaxTokens)
	}
	// Without return_tokens the array is empty.
	_, out = llamaPost(t, s, "/completion", `{"prompt":"Hi"}`)
	if toks, _ := out["tokens"].([]any); len(toks) != 0 {
		t.Fatalf("tokens without return_tokens: %v", out["tokens"])
	}
}

func TestLlamaStopTypes(t *testing.T) {
	for _, c := range []struct {
		reason FinishReason
		want   string
		word   string
	}{{FinishEOS, "eos", ""}, {FinishStop, "word", "END"}, {FinishMaxTokens, "limit", ""}} {
		f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: c.reason, stopAt: c.word}}
		_, out := llamaPost(t, llamaServe(t, f), "/completion", `{"prompt":"x","stop":["END"]}`)
		if out["stop_type"] != c.want || out["stopping_word"] != c.word {
			t.Errorf("%v: stop_type %v word %v", c.reason, out["stop_type"], out["stopping_word"])
		}
		if c.word != "" && len(f.opts().Stop) != 1 {
			t.Errorf("stop not passed: %v", f.opts().Stop)
		}
	}
}

func TestLlamaPromptForms(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	cases := []struct {
		body string
		want []int32
	}{
		{`{"prompt":"A"}`, []int32{1, 256 + 'A'}},
		{`{"prompt":[5,6,7]}`, []int32{5, 6, 7}},
		{`{"prompt":[5,"A",6,"B"]}`, []int32{5, 1 + 0*0, 6}}, // replaced below
	}
	cases[2].want = []int32{5, 256 + 'A', 6, 256 + 'B'}
	for _, c := range cases {
		if code, out := llamaPost(t, s, "/completion", c.body); code != 200 {
			t.Fatalf("%s: %d %v", c.body, code, out)
		}
		got := f.opts().Prompt.IDs
		if len(got) != len(c.want) {
			t.Fatalf("%s: ids %v want %v", c.body, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("%s: ids %v want %v", c.body, got, c.want)
			}
		}
	}
	for _, bad := range []string{`{"prompt":[["a"],["b"]]}`, `{"prompt":{"prompt_string":"x","multimodal_data":["AAA"]}}`, `{}`} {
		if code, _ := llamaPost(t, s, "/completion", bad); code != 400 {
			t.Errorf("%s answered %d", bad, code)
		}
	}
}

func TestLlamaSamplingDefaultsAndMapping(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	llamaPost(t, s, "/completion", `{"prompt":"x"}`)
	sm := f.opts().Sampling
	if sm.Temp != 0.8 || sm.TopK != 40 || sm.TopP != float64(float32(0.95)) && sm.TopP != 0.95 || sm.MinP != 0.05 {
		t.Fatalf("llama-server defaults not applied: %+v", sm)
	}
	llamaPost(t, s, "/completion", `{"prompt":"x","temperature":0,"top_k":5,"top_p":0.5,"min_p":0.2,"seed":42,"repeat_penalty":1.3}`)
	sm = f.opts().Sampling
	if sm.Temp != 0 || sm.TopK != 5 || sm.TopP != 0.5 || sm.MinP != 0.2 || sm.Seed != 42 || sm.RepeatPen != 1.3 {
		t.Fatalf("fields not mapped: %+v", sm)
	}
	llamaPost(t, s, "/completion", `{"prompt":"x","json_schema":{"type":"object"}}`)
	if f.opts().Grammar == "" {
		t.Fatal("json_schema produced no grammar")
	}
	llamaPost(t, s, "/completion", `{"prompt":"x","grammar":"root ::= \"a\""}`)
	if f.opts().Grammar != `root ::= "a"` {
		t.Fatalf("grammar: %q", f.opts().Grammar)
	}
}

// TestLlamaRefusals: every parameter the engine cannot honour is refused
// naming itself, and its neutral value is accepted.
func TestLlamaRefusals(t *testing.T) {
	s := llamaServe(t, &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}})
	refuse := map[string]string{
		"mirostat": `2`, "dry_multiplier": `0.8`, "xtc_probability": `0.5`, "typical_p": `0.9`,
		"presence_penalty": `0.3`, "frequency_penalty": `0.3`, "dynatemp_range": `0.5`,
		"logit_bias": `[[15,1.0]]`, "samplers": `["dry","top_k"]`, "n_cmpl": `2`, "id_slot": `3`,
		"n_predict": `0`, "n_keep": `5`, "lora": `[{"id":0,"scale":1}]`, "post_sampling_probs": `true`,
		"timings_per_token": `true`, "t_max_predict_ms": `100`, "n_probs": `99`, "repeat_last_n": `128`,
		"response_fields": `["content"]`, "mystery_field": `1`,
	}
	for k, v := range refuse {
		code, out := llamaPost(t, s, "/completion", `{"prompt":"x","`+k+`":`+v+`}`)
		msg, _ := json.Marshal(out)
		if code != 400 || !strings.Contains(string(msg), k) {
			t.Errorf("%s=%s: %d %s", k, v, code, msg)
		}
	}
	accept := `{"prompt":"x","mirostat":0,"dry_multiplier":0,"xtc_probability":0,"typical_p":1,"presence_penalty":0,
		"logit_bias":[],"samplers":["top_k","top_p","min_p","temperature"],"n_cmpl":1,"id_slot":-1,"cache_prompt":true,
		"n_keep":0,"timings_per_token":false,"top_n_sigma":-1,"repeat_last_n":64}`
	if code, out := llamaPost(t, s, "/completion", accept); code != 200 {
		t.Fatalf("neutral values refused: %d %v", code, out)
	}
	// The error body is llama-server's.
	_, out := llamaPost(t, s, "/completion", `{"prompt":"x","mirostat":1}`)
	e := out["error"].(map[string]any)
	if e["code"] != float64(400) || e["type"] != "invalid_request_error" || e["message"] == "" {
		t.Fatalf("error shape: %v", out)
	}
}

func TestLlamaStream(t *testing.T) {
	gate := make(chan struct{})
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a", "b", "c"}, reason: FinishEOS}}
	f.beforeToken = func(i int) {
		if i == 1 {
			<-gate
		}
	}
	s := llamaServe(t, f)
	resp := post(t, s, "/completion", `{"prompt":"x","stream":true}`)
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type %q", ct)
	}
	rd := bufio.NewReader(resp.Body)
	read := func() map[string]any {
		for {
			line, err := rd.ReadString('\n')
			if err != nil {
				t.Fatalf("stream ended: %v", err)
			}
			if rest, ok := strings.CutPrefix(line, "data: "); ok {
				var m map[string]any
				if err := json.Unmarshal([]byte(rest), &m); err != nil {
					t.Fatalf("frame %q: %v", rest, err)
				}
				return m
			}
		}
	}
	first := read() // arrives while the generate is held before token 2: frames are flushed
	if first["content"] != "a" || first["stop"] != false {
		t.Fatalf("first frame: %v", first)
	}
	if toks, _ := first["tokens"].([]any); len(toks) != 1 {
		t.Fatalf("stream frames carry token ids: %v", first)
	}
	close(gate)
	var text = "a"
	var last map[string]any
	for {
		m := read()
		if m["stop"] == true {
			last = m
			break
		}
		text += m["content"].(string)
	}
	if text != "abc" {
		t.Fatalf("streamed %q", text)
	}
	if last["stop_type"] != "eos" || last["tokens_predicted"] != float64(3) || last["timings"] == nil || last["generation_settings"] == nil {
		t.Fatalf("final frame: %v", last)
	}
	rest, _ := io.ReadAll(rd)
	if strings.Contains(string(rest), "[DONE]") {
		t.Fatalf("llama-server sends no [DONE]: %q", rest)
	}
}

func TestLlamaTokenizeRoundTrip(t *testing.T) {
	s := llamaServe(t, &llamaFake{fakeBackend: &fakeBackend{}})
	code, out := llamaPost(t, s, "/tokenize", `{"content":"héllo"}`)
	if code != 200 {
		t.Fatalf("%d %v", code, out)
	}
	toks := out["tokens"].([]any)
	if len(toks) != 6 { // h, 2 bytes of é, l, l, o; no BOS: add_special defaults false
		t.Fatalf("tokens %v", toks)
	}
	b, _ := json.Marshal(map[string]any{"tokens": toks})
	_, back := llamaPost(t, s, "/detokenize", string(b))
	if back["content"] != "héllo" {
		t.Fatalf("round trip: %v", back)
	}
	_, out = llamaPost(t, s, "/tokenize", `{"content":"a","add_special":true}`)
	if toks := out["tokens"].([]any); len(toks) != 2 || toks[0] != float64(1) {
		t.Fatalf("add_special: %v", toks)
	}
	_, out = llamaPost(t, s, "/tokenize", `{"content":"é","with_pieces":true}`)
	ps := out["tokens"].([]any)
	if p := ps[0].(map[string]any)["piece"]; len(ps) != 2 || p == nil {
		t.Fatalf("with_pieces: %v", ps)
	} else if _, isBytes := p.([]any); !isBytes {
		t.Fatalf("an invalid-utf8 piece is a byte list: %v", p)
	}
	if code, _ := llamaPost(t, s, "/tokenize", `{}`); code != 400 {
		t.Fatalf("content is required: %d", code)
	}
	if code, _ := llamaPost(t, s, "/tokenize", `{"content":"x","bogus":1}`); code != 400 {
		t.Fatalf("unknown field accepted: %d", code)
	}
}

func TestLlamaApplyTemplate(t *testing.T) {
	s := llamaServe(t, &llamaFake{fakeBackend: &fakeBackend{}})
	code, out := llamaPost(t, s, "/apply-template", `{"messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 || out["prompt"] != "<user>hi\n<assistant>" {
		t.Fatalf("%d %v", code, out)
	}
	if code, _ := llamaPost(t, s, "/apply-template", `{}`); code != 400 {
		t.Fatalf("messages required: %d", code)
	}
}

func TestLlamaInfill(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"x"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	if code, out := llamaPost(t, s, "/infill", `{"input_prefix":"a","input_suffix":"b"}`); code != 400 {
		t.Fatalf("a model without FIM tokens must be refused: %d %v", code, out)
	}
	f.fim = true
	code, out := llamaPost(t, s, "/infill", `{"input_prefix":"a","input_suffix":"b","prompt":"c"}`)
	if code != 200 {
		t.Fatalf("%d %v", code, out)
	}
	want := []int32{1, 10, 256 + 'a', 11, 256 + 'b', 12, 256 + 'c'}
	got := f.opts().Prompt.IDs
	if len(got) != len(want) {
		t.Fatalf("ids %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ids %v want %v", got, want)
		}
	}
	if code, _ := llamaPost(t, s, "/infill", `{"input_prefix":"a"}`); code != 400 {
		t.Fatalf("input_suffix required: %d", code)
	}
	if code, _ := llamaPost(t, s, "/completion", `{"prompt":"x","input_prefix":"a"}`); code != 400 {
		t.Fatalf("input_prefix belongs to /infill: %d", code)
	}
}

func TestLlamaPropsSlotsHealthEmbedding(t *testing.T) {
	s := llamaServe(t, &llamaFake{fakeBackend: &fakeBackend{}})
	get := func(path string) (int, map[string]any) {
		resp, err := http.Get(s.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var m map[string]any
		json.NewDecoder(resp.Body).Decode(&m)
		return resp.StatusCode, m
	}
	code, p := get("/props")
	if code != 200 || p["total_slots"] != float64(1) || p["model_path"] != "/models/tiny.jlm" || p["chat_template"] != "{{ messages }}" {
		t.Fatalf("props: %d %v", code, p)
	}
	d := p["default_generation_settings"].(map[string]any)
	if d["n_ctx"] != float64(2048) || d["params"].(map[string]any)["temperature"] == nil {
		t.Fatalf("default_generation_settings: %v", d)
	}
	if code, _ := llamaPost(t, s, "/props", `{}`); code != 501 {
		t.Fatalf("POST /props: %d", code)
	}
	resp, err := http.Get(s.URL + "/slots")
	if err != nil {
		t.Fatal(err)
	}
	var slots []map[string]any
	json.NewDecoder(resp.Body).Decode(&slots)
	resp.Body.Close()
	if len(slots) != 1 || slots[0]["id"] != float64(0) || slots[0]["n_ctx"] != float64(2048) || slots[0]["is_processing"] != false {
		t.Fatalf("slots: %v", slots)
	}
	// The fake embeds nothing: the refusal is llama-server's error body.
	code, out := llamaPost(t, s, "/embedding", `{"content":"hi"}`)
	if code != 400 || out["error"] == nil {
		t.Fatalf("embedding: %d %v", code, out)
	}
	if code, _ := llamaPost(t, s, "/embedding", `{"content":"hi","embd_normalize":0}`); code != 400 {
		t.Fatalf("embd_normalize 0 accepted: %d", code)
	}
}

func TestLlamaEmbeddingShape(t *testing.T) {
	s := serve(t, &llamaEmbedFake{llamaFake: &llamaFake{fakeBackend: &fakeBackend{}}})
	resp := post(t, s, "/embedding", `{"content":["a","b"]}`)
	defer resp.Body.Close()
	var out []struct {
		Index     int         `json:"index"`
		Embedding [][]float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[1].Index != 1 || len(out[0].Embedding) != 1 || len(out[0].Embedding[0]) != 3 {
		t.Fatalf("shape: %+v", out)
	}
}

type llamaEmbedFake struct{ *llamaFake }

func (f *llamaEmbedFake) Embed(_ context.Context, o EmbedOptions) (*EmbedResult, error) {
	r := &EmbedResult{}
	for range o.Inputs {
		r.Vectors = append(r.Vectors, []float32{1, 0, 0})
		r.Tokens = append(r.Tokens, 2)
	}
	return r, nil
}

// TestLlamaAuthApplies: the key guards the llama.cpp routes; /health and
// /v1/health stay open, as llama-server's are.
func TestLlamaAuthApplies(t *testing.T) {
	h := routeHealth(CompatHandler(&llamaFake{fakeBackend: &fakeBackend{}}))
	do := func(method, path, key string) int {
		req := httptest.NewRequest(method, path, strings.NewReader(`{"content":"x"}`))
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code
	}
	for _, p := range []string{"/tokenize", "/detokenize", "/completion", "/infill", "/apply-template", "/embedding"} {
		if c := do("POST", p, ""); c != 401 {
			t.Errorf("%s without a key: %d", p, c)
		}
		if c := do("POST", p, "k"); c == 401 {
			t.Errorf("%s with the key: %d", p, c)
		}
	}
	for _, p := range []string{"/props", "/slots"} {
		if c := do("GET", p, ""); c != 401 {
			t.Errorf("%s without a key: %d", p, c)
		}
	}
	for _, p := range []string{"/health", "/v1/health"} {
		if c := do("GET", p, ""); c != 200 {
			t.Errorf("%s needs no key: %d", p, c)
		}
	}
}

// routeHealth serves the engine's health routes beside the compat handler, as
// Engine.Handler mounts them.
func routeHealth(compat http.Handler) http.Handler {
	e := &Engine{}
	mux := http.NewServeMux()
	mux.Handle("/", compat)
	for _, r := range engineRoutes() {
		h := r.handle
		mux.HandleFunc(r.path, func(w http.ResponseWriter, q *http.Request) { h(e, w, q) })
	}
	return Guard{APIKey: "k"}.Wrap(mux)
}

func TestLlamaHealthShape(t *testing.T) {
	h := (&Engine{}).Handler()
	for _, p := range []string{"/health", "/v1/health"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", p, nil))
		if rec.Code != 200 || strings.TrimSpace(rec.Body.String()) != `{"status":"ok"}` {
			t.Errorf("%s: %d %q", p, rec.Code, rec.Body.String())
		}
	}
}
