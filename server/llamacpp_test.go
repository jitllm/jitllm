package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image/color"
	"io"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"
)

// llamaFake is the scripted backend with a byte tokenizer (id 1 is the BOS,
// every byte b is id 256+b), so the llama.cpp routes run without a model.
type llamaFake struct {
	*fakeBackend
	fim bool
	// slots is how many slots Slots lists, one when 0.
	slots int
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

func (f *llamaFake) Slots(string) []LlamaSlot {
	out := make([]LlamaSlot, max(f.slots, 1))
	for i := range out {
		out[i] = LlamaSlot{ID: "s" + strconv.Itoa(i+1), NCtx: 2048}
	}
	return out
}

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
	for _, bad := range []string{`{"prompt":[["a"],["b"]]}`, `{"prompt":[[5,["x"]]]}`, `{"prompt":[]}`,
		`{"prompt":{"prompt_string":"x","multimodal_data":["AAA"]}}`, `{"prompt":{"prompt_string":"<__media__>","multimodal_data":["AAA"]}}`, `{}`} {
		if code, _ := llamaPost(t, s, "/completion", bad); code != 400 {
			t.Errorf("%s answered %d", bad, code)
		}
	}
}

func llamaPostAll(t *testing.T, s *httptest.Server, body string) []map[string]any {
	t.Helper()
	resp := post(t, s, "/completion", body)
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var out []map[string]any
	if resp.StatusCode != 200 || json.Unmarshal(b, &out) != nil {
		t.Fatalf("%s: %d %s", body, resp.StatusCode, b)
	}
	return out
}

// TestLlamaPromptsAndCompletions: a prompt array holding no token id is a list
// of prompts, each answered with n_cmpl completions (n its alias), the results
// in llama-server's order -- each prompt, then its completions -- one result
// alone the object itself; a seed gives completion i seed+i, whole or
// streamed.
func TestLlamaPromptsAndCompletions(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"Hel", "lo"}, reason: FinishMaxTokens}}
	s := llamaServe(t, f)
	out := llamaPostAll(t, s, `{"prompt":["Hi",[5,6],{"prompt_string":"Yo"}],"n_predict":2}`)
	all := f.took()
	wantPrompts := []string{"Hi", "", "Yo"}
	if len(out) != 3 || len(all) != 3 {
		t.Fatalf("%d results from %d generates, want 3 of 3: %v", len(out), len(all), out)
	}
	for i, r := range out {
		if r["index"] != float64(i) || r["prompt"] != wantPrompts[i] || r["content"] != "Hello" || r["stop"] != true {
			t.Errorf("result %d: %v", i, r)
		}
	}
	wantIDs := [][]int32{{1, 256 + 'H', 256 + 'i'}, {5, 6}, {1, 256 + 'Y', 256 + 'o'}}
	for i, o := range all {
		if !slices.Equal(o.Prompt.IDs, wantIDs[i]) || len(o.Seeds) != 0 {
			t.Errorf("generate %d: ids %v seeds %v, want %v and none", i, o.Prompt.IDs, o.Seeds, wantIDs[i])
		}
	}

	out = llamaPostAll(t, s, `{"prompt":["Hi","Yo"],"n_cmpl":2,"seed":42}`)
	all = f.took()
	if len(out) != 4 || len(all) != 2 {
		t.Fatalf("%d results from %d generates, want 4 of 2", len(out), len(all))
	}
	for i, r := range out {
		if r["index"] != float64(i) || r["prompt"] != []string{"Hi", "Hi", "Yo", "Yo"}[i] {
			t.Errorf("result %d: index %v prompt %v", i, r["index"], r["prompt"])
		}
	}
	for i, o := range all {
		if !slices.Equal(o.Seeds, []int64{42, 43}) {
			t.Errorf("generate %d seeds %v, want [42 43]", i, o.Seeds)
		}
	}

	// n is n_cmpl's alias; with no seed the completions still differ.
	out = llamaPostAll(t, s, `{"prompt":"Hi","n":3}`)
	all = f.took()
	if len(out) != 3 || len(all) != 1 {
		t.Fatalf("n=3: %d results from %d generates, want 3 of 1", len(out), len(all))
	}
	if sd := all[0].Seeds; len(out) != 3 || len(sd) != 3 || sd[1] != sd[0]+1 || sd[2] != sd[0]+2 {
		t.Fatalf("n=3: %d results, seeds %v", len(out), sd)
	}

	// One prompt in a list is one result, answered as the object.
	if code, one := llamaPost(t, s, "/completion", `{"prompt":["Hi"]}`); code != 200 || one["content"] != "Hello" {
		t.Fatalf("one prompt in a list: %d %v", code, one)
	}
	for _, bad := range []string{`{"prompt":"Hi","n_cmpl":0}`, `{"prompt":"Hi","n_cmpl":129}`} {
		code, msg := llamaPost(t, s, "/completion", bad)
		if b, _ := json.Marshal(msg); code != 400 || !strings.Contains(string(b), "n_cmpl") {
			t.Errorf("%s: %d %s", bad, code, b)
		}
	}

	resp := post(t, s, "/completion", `{"prompt":["Hi","Yo"],"n_cmpl":2,"stream":true}`)
	defer resp.Body.Close()
	var finals []float64
	tokens := map[float64]int{}
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		var fr map[string]any
		if err := json.Unmarshal([]byte(line), &fr); err != nil {
			t.Fatalf("frame %q: %v", line, err)
		}
		ix, _ := fr["index"].(float64)
		if fr["stop"] == true {
			finals = append(finals, ix)
		} else {
			tokens[ix]++
		}
	}
	if !slices.Equal(finals, []float64{0, 1, 2, 3}) {
		t.Fatalf("streamed results ended %v, want each of 0..3 once in order", finals)
	}
	for ix := range 4 {
		if tokens[float64(ix)] != 2 {
			t.Errorf("result %d streamed %d token frames, want 2", ix, tokens[float64(ix)])
		}
	}
}

// TestLlamaNPredictZeroAndTimingsPerToken: n_predict 0 asks for the one token
// llama-server answers it with, and reports 0; timings_per_token puts the
// timings so far on every token frame, and without it a token frame carries
// none.
func TestLlamaNPredictZeroAndTimingsPerToken(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"Hel", "lo", "!"}, reason: FinishMaxTokens}}
	s := llamaServe(t, f)
	code, out := llamaPost(t, s, "/completion", `{"prompt":"Hi","n_predict":0}`)
	if code != 200 {
		t.Fatalf("n_predict 0: %d %v", code, out)
	}
	if f.opts().MaxTokens != 1 {
		t.Fatalf("n_predict 0 asked for %d tokens, want 1", f.opts().MaxTokens)
	}
	if gs := out["generation_settings"].(map[string]any); gs["n_predict"] != float64(0) {
		t.Fatalf("generation_settings.n_predict %v, want 0", gs["n_predict"])
	}
	for _, perToken := range []bool{true, false} {
		body := `{"prompt":"Hi","stream":true,"timings_per_token":false}`
		if perToken {
			body = `{"prompt":"Hi","stream":true,"timings_per_token":true}`
		}
		resp := post(t, s, "/completion", body)
		sc := bufio.NewScanner(resp.Body)
		n := 0
		for sc.Scan() {
			line, ok := strings.CutPrefix(sc.Text(), "data: ")
			if !ok {
				continue
			}
			var fr map[string]any
			if err := json.Unmarshal([]byte(line), &fr); err != nil {
				t.Fatalf("frame %q: %v", line, err)
			}
			if fr["stop"] == true {
				continue
			}
			n++
			tm, has := fr["timings"].(map[string]any)
			if has != perToken {
				t.Fatalf("timings_per_token=%v: token frame %d timings %v", perToken, n, fr["timings"])
			}
			if perToken && (tm["predicted_n"] != fr["tokens_predicted"] || tm["prompt_n"] != float64(7) || tm["prompt_ms"] != float64(40)) {
				t.Fatalf("token frame %d timings %v, tokens_predicted %v", n, tm, fr["tokens_predicted"])
			}
		}
		resp.Body.Close()
		if n != 3 {
			t.Fatalf("timings_per_token=%v: %d token frames, want 3", perToken, n)
		}
	}
}

// TestLlamaPredictTimeReachesTheEngine: t_max_predict_ms is the generate's
// PredictTime, and 0 sets none.
func TestLlamaPredictTimeReachesTheEngine(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	for body, want := range map[string]time.Duration{
		`{"prompt":"x","t_max_predict_ms":250}`: 250 * time.Millisecond,
		`{"prompt":"x","t_max_predict_ms":0}`:   0,
		`{"prompt":"x"}`:                        0,
	} {
		if code, out := llamaPost(t, s, "/completion", body); code != 200 {
			t.Fatalf("%s: %d %v", body, code, out)
		}
		if got := f.opts().PredictTime; got != want {
			t.Errorf("%s: PredictTime %v, want %v", body, got, want)
		}
	}
}

// TestLlamaSSEPings: a stream silent for sse_ping_interval seconds is sent an
// SSE comment, once for each such silence; -1 sends none, and the 30 second
// default none in a stream that never falls silent that long.
// TestLlamaStreamsPromptProgress: return_progress on a stream asks the
// generate for its prompt's progress and sends each as a frame of its own
// before the reply's, as llama-server does: no text, no token, stop false,
// and prompt_progress's total, cache, processed and time_ms. Without a stream,
// or without the field, the generate is not asked.
func TestLlamaStreamsPromptProgress(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"Hel", "lo"}, reason: FinishMaxTokens}}
	s := llamaServe(t, f)
	frames := func(body string) []map[string]any {
		resp := post(t, s, "/completion", body)
		defer resp.Body.Close()
		var out []map[string]any
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			line, ok := strings.CutPrefix(sc.Text(), "data: ")
			if !ok {
				continue
			}
			var fr map[string]any
			if err := json.Unmarshal([]byte(line), &fr); err != nil {
				t.Fatalf("frame %q: %v", line, err)
			}
			out = append(out, fr)
		}
		return out
	}
	got := frames(`{"prompt":"Hi","stream":true,"return_progress":true}`)
	if !f.opts().Progress {
		t.Fatal("return_progress on a stream did not ask the generate for the prompt's progress")
	}
	if len(got) != 6 {
		t.Fatalf("%d frames, want three of progress, two of text and the last: %v", len(got), got)
	}
	for i, done := range []float64{2, 5, 7} {
		fr := got[i]
		pp, _ := fr["prompt_progress"].(map[string]any)
		want := map[string]any{"total": float64(7), "cache": float64(2), "processed": done, "time_ms": done}
		if pp == nil || !maps.Equal(pp, want) {
			t.Fatalf("frame %d: prompt_progress %v, want %v", i, fr["prompt_progress"], want)
		}
		toks, _ := fr["tokens"].([]any)
		if fr["content"] != "" || toks == nil || len(toks) != 0 || fr["stop"] != false || fr["index"] != float64(0) ||
			fr["tokens_predicted"] != float64(0) || fr["tokens_evaluated"] != float64(7) {
			t.Fatalf("frame %d is not a progress frame: %v", i, fr)
		}
	}
	if _, ok := got[3]["prompt_progress"]; ok || got[3]["content"] != "Hel" {
		t.Fatalf("the reply's first frame: %v", got[3])
	}
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"Hi","return_progress":true}`); code != 200 || f.opts().Progress {
		t.Fatalf("return_progress without a stream: %d, progress asked %v: %v", code, f.opts().Progress, out)
	}
	for _, fr := range frames(`{"prompt":"Hi","stream":true}`) {
		if _, ok := fr["prompt_progress"]; ok || f.opts().Progress {
			t.Fatalf("a stream without return_progress: %v", fr)
		}
	}
}

// TestLlamaIndentAndPromptTime: n_indent reaches the generate as MinIndent
// and the answer is cut at the line the indentation ended it on, its leading
// spaces kept; t_max_prompt_ms is taken, to no effect, as llama-server takes
// it.
func TestLlamaIndentAndPromptTime(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a", "\n", "  b", "\n c", "d"}, reason: FinishMaxTokens}}
	s := llamaServe(t, f)
	code, out := llamaPost(t, s, "/completion", `{"prompt":"x","n_indent":2,"t_max_prompt_ms":500}`)
	if code != 200 {
		t.Fatalf("%d %v", code, out)
	}
	if f.opts().MinIndent != 2 {
		t.Fatalf("n_indent reached the generate as %d", f.opts().MinIndent)
	}
	if out["content"] != "a\n  b\n " {
		t.Fatalf("content %q, want the short line cut at its first character", out["content"])
	}
	for in, want := range map[string]string{"a\n  b": "a\n  b", "a\n b": "a\n ", "a\n  ": "a\n  ", "ab": "ab", "a\n\tb": "a\n\t"} {
		if got := cutShortLine(in, 2); got != want {
			t.Errorf("cutShortLine(%q, 2) = %q, want %q", in, got, want)
		}
	}
	if code, _ := llamaPost(t, s, "/completion", `{"prompt":"x"}`); code != 200 || f.opts().MinIndent != 0 {
		t.Fatalf("no n_indent: %d, MinIndent %d", code, f.opts().MinIndent)
	}
}

func TestLlamaSSEPings(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a", "b"}, reason: FinishEOS, beforeToken: func(i int) {
		if i == 1 {
			time.Sleep(1500 * time.Millisecond)
		}
	}}}
	s := llamaServe(t, f)
	pings := func(body string) (n, frames int) {
		resp := post(t, s, "/completion", body)
		defer resp.Body.Close()
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			switch l := sc.Text(); {
			case l == ":":
				n++
			case strings.HasPrefix(l, "data: "):
				frames++
			}
		}
		return n, frames
	}
	if n, fr := pings(`{"prompt":"x","stream":true,"sse_ping_interval":1}`); n != 1 || fr != 3 {
		t.Fatalf("a 1.5 s silence under a 1 s interval: %d pings and %d frames, want 1 and 3", n, fr)
	}
	for _, body := range []string{`{"prompt":"x","stream":true,"sse_ping_interval":-1}`, `{"prompt":"x","stream":true}`} {
		if n, fr := pings(body); n != 0 || fr != 3 {
			t.Errorf("%s: %d pings and %d frames, want none and 3", body, n, fr)
		}
	}
}

// TestLlamaAnySlot: any id_slot is taken, as llama-server takes it, and
// reported wrapped by the slot count (the fake's one).
func TestLlamaAnySlot(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a", "b"}, reason: FinishEOS}, slots: 4}
	s := llamaServe(t, f)
	for slot, want := range map[string]float64{"-1": 0, "0": 0, "3": 3, "6": 2} {
		code, out := llamaPost(t, s, "/completion", `{"prompt":"x","id_slot":`+slot+`}`)
		if code != 200 || out["id_slot"] != want {
			t.Errorf("id_slot %s: %d, reported %v, want %v", slot, code, out["id_slot"], want)
		}
		// A stream reports it on every frame, its replies' and its last.
		resp := post(t, s, "/completion", `{"prompt":"x","stream":true,"id_slot":`+slot+`}`)
		sc := bufio.NewScanner(resp.Body)
		frames := 0
		for sc.Scan() {
			line, ok := strings.CutPrefix(sc.Text(), "data: ")
			if !ok {
				continue
			}
			var fr map[string]any
			if err := json.Unmarshal([]byte(line), &fr); err != nil {
				t.Fatalf("frame %q: %v", line, err)
			}
			frames++
			if fr["id_slot"] != want {
				t.Errorf("id_slot %s: frame %d reported %v, want %v", slot, frames, fr["id_slot"], want)
			}
		}
		resp.Body.Close()
		if frames != 3 {
			t.Errorf("id_slot %s: %d frames, want two of text and the last", slot, frames)
		}
	}
}

// TestLlamaPicturesReachTheGenerate: a prompt's multimodal_data reaches the
// generate as its text and pictures, each where the text holds <__media__>,
// in order; a list of such prompts gives each its own.
func TestLlamaPicturesReachTheGenerate(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	red := base64.StdEncoding.EncodeToString(testPNG(t, 4, 3, color.NRGBA{R: 200, A: 255}))
	blue := base64.StdEncoding.EncodeToString(testPNG(t, 2, 5, color.NRGBA{B: 200, A: 255}))
	one := `{"prompt_string":"a <__media__> b <__media__>","multimodal_data":["` + red + `","` + blue + `"]}`
	if code, out := llamaPost(t, s, "/completion", `{"prompt":`+one+`}`); code != 200 {
		t.Fatalf("%d %v", code, out)
	}
	p := f.opts().Prompt
	if p.Kind != PromptText || p.Text != "a <__media__> b <__media__>" || p.Marker != "<__media__>" || len(p.Images) != 2 ||
		p.Images[0].Bounds().Dx() != 4 || p.Images[1].Bounds().Dx() != 2 {
		t.Fatalf("the generate's prompt: kind %v text %q marker %q, %d pictures", p.Kind, p.Text, p.Marker, len(p.Images))
	}
	f.took()
	two := `{"prompt_string":"<__media__>c","multimodal_data":["` + blue + `"]}`
	if code, out := llamaPost(t, s, "/completion", `{"prompt":[`+one+`,`+two+`,"plain"]}`); code != 200 {
		t.Fatalf("a list: %d", code)
	} else if _, many := out["raw"]; !many {
		t.Fatalf("three prompts answered one object: %v", out)
	}
	all := f.took()
	if len(all) != 3 || len(all[0].Prompt.Images) != 2 || len(all[1].Prompt.Images) != 1 || all[1].Prompt.Images[0].Bounds().Dx() != 2 ||
		all[2].Prompt.Kind != PromptIDs {
		t.Fatalf("a list of prompts with pictures reached %d generates", len(all))
	}
}

// TestLlamaSamplersChooseTheStages: samplers is the stages that run, in
// order, as llama-server reads it: a stage it leaves out is off whatever its
// field says (no temperature draws at 1), its names take llama.cpp's
// aliases and letters, and an order the draw does not run, an unknown name or
// a stage not built is refused naming samplers.
func TestLlamaSamplersChooseTheStages(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	const fields = `"top_k":5,"top_p":0.5,"min_p":0.2,"temperature":0.3,"repeat_penalty":1.3,"typical_p":0.7`
	for _, samplers := range []string{`["top_k","top_p"]`, `"kp"`, `["Top-K","nucleus"]`, `["topk","TOP_P"]`} {
		code, out := llamaPost(t, s, "/completion", `{"prompt":"x",`+fields+`,"samplers":`+samplers+`}`)
		if code != 200 {
			t.Fatalf("%s: %d %v", samplers, code, out)
		}
		sm := f.opts().Sampling
		if sm.TopK != 5 || sm.TopP != 0.5 || sm.MinP != 0 || sm.Temp != 1 || sm.RepeatPen != 1 || sm.TypicalP != 1 {
			t.Errorf("%s ran %+v, want top-k 5 and top-p 0.5 alone at temperature 1", samplers, sm)
		}
		gs := out["generation_settings"].(map[string]any)
		if got, _ := json.Marshal(gs["samplers"]); string(got) != `["top_k","top_p"]` || gs["temperature"] != 0.3 {
			t.Errorf("%s: generation_settings samplers %s, temperature %v", samplers, got, gs["temperature"])
		}
	}
	if code, _ := llamaPost(t, s, "/completion", `{"prompt":"x",`+fields+`}`); code != 200 {
		t.Fatalf("no samplers: %d", code)
	}
	if sm := f.opts().Sampling; sm.TopK != 5 || sm.MinP != 0.2 || sm.Temp != 0.3 || sm.RepeatPen != 1.3 {
		t.Errorf("the default chain ran %+v, want every stage", sm)
	}
	for _, bad := range []string{`["top_p","top_k"]`, `["top_k","foo"]`, `["top_k","infill","top_p"]`, `"kq"`, `7`} {
		code, out := llamaPost(t, s, "/completion", `{"prompt":"x","samplers":`+bad+`}`)
		if b, _ := json.Marshal(out); code != 400 || !strings.Contains(string(b), "samplers") {
			t.Errorf("samplers %s: %d %s", bad, code, b)
		}
	}
}

// TestLlamaInfillSamplerReachesTheDraw: infill as samplers' last stage, by
// name or letter, asks the generate for the infill sampler, which
// generation_settings lists; no samplers, or a list without it, does not.
func TestLlamaInfillSamplerReachesTheDraw(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	for _, samplers := range []string{`["top_k","top_p","infill"]`, `"kpi"`, `["Infill"]`} {
		code, out := llamaPost(t, s, "/completion", `{"prompt":"x","top_k":40,"samplers":`+samplers+`}`)
		if code != 200 {
			t.Fatalf("%s: %d %v", samplers, code, out)
		}
		if sm := f.opts().Sampling; sm == nil || !sm.Infill {
			t.Fatalf("%s: the draw has no infill sampler", samplers)
		}
		gs := out["generation_settings"].(map[string]any)
		if got, _ := json.Marshal(gs["samplers"]); !strings.HasSuffix(string(got), `"infill"]`) {
			t.Errorf("%s: generation_settings samplers %s", samplers, got)
		}
	}
	for _, body := range []string{`{"prompt":"x"}`, `{"prompt":"x","samplers":["top_k","top_p"]}`} {
		if code, _ := llamaPost(t, s, "/completion", body); code != 200 || f.opts().Sampling.Infill {
			t.Fatalf("%s: %d, infill %v", body, code, f.opts().Sampling.Infill)
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
	_, out := llamaPost(t, s, "/completion", `{"prompt":"x","n_probs":3,"post_sampling_probs":true}`)
	if o := f.opts(); !o.Logprobs || o.TopLogprobs != 3 || !o.PostSampling {
		t.Fatalf("post_sampling_probs reached the generate as logprobs %v, %d alternatives, after sampling %v", o.Logprobs, o.TopLogprobs, o.PostSampling)
	}
	if gs := out["generation_settings"].(map[string]any); gs["post_sampling_probs"] != true {
		t.Fatalf("generation_settings: %v", gs)
	}
	// adaptive-p, named anywhere in samplers, as llama.cpp runs it last.
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","samplers":["adaptive_p","top_k","temperature"],"adaptive_target":0.3,"adaptive_decay":0.8}`); code != 200 {
		t.Fatalf("adaptive-p: %d %v", code, out)
	}
	if sm := f.opts().Sampling; !sm.Adaptive || sm.AdaptiveTarget != 0.3 || sm.AdaptiveDecay != 0.8 || sm.TopK != 40 {
		t.Fatalf("adaptive-p reached the sampler as %v at %v, decay %v, top-k %d", sm.Adaptive, sm.AdaptiveTarget, sm.AdaptiveDecay, sm.TopK)
	}
	if code, _ := llamaPost(t, s, "/completion", `{"prompt":"x","adaptive_target":0.3}`); code != 200 || f.opts().Sampling.Adaptive {
		t.Fatalf("adaptive_target without adaptive_p in samplers: %d, adaptive %v", code, f.opts().Sampling.Adaptive)
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
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"a"}, reason: FinishEOS}}
	s := llamaServe(t, f)
	refuse := map[string]string{
		"mirostat": `3`, "dry_multiplier": `0.8`,
		"presence_penalty": `3`, "frequency_penalty": `-3`,
		"logit_bias": `[[15]]`, "samplers": `["top_k","dry"]`, "n_cmpl": `0`, "id_slot": `-2`,
		"lora":    `[{"id":0,"scale":1}]`,
		"n_probs": `99`, "repeat_last_n": `-2`,
		"response_fields": `"content"`, "mystery_field": `1`,
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
		"n_keep":5,"n_discard":2,"n_cache_reuse":256,"timings_per_token":false,"top_n_sigma":-1,"repeat_last_n":64}`
	if code, out := llamaPost(t, s, "/completion", accept); code != 200 {
		t.Fatalf("neutral values refused: %d %v", code, out)
	}
	// repeat_last_n is the sampler's window, -1 the context (the fake's 2048).
	for v, want := range map[string]int{`128`: 128, `-1`: 2048} {
		if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","repeat_penalty":1.1,"repeat_last_n":`+v+`}`); code != 200 {
			t.Fatalf("repeat_last_n=%s: %d %v", v, code, out)
		}
		if got := f.opts().Sampling.RepeatLastN; got != want {
			t.Errorf("repeat_last_n=%s reached the sampler as %d, want %d", v, got, want)
		}
	}
	// presence_penalty and frequency_penalty reach the sampler.
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","presence_penalty":0.4,"frequency_penalty":-0.25}`); code != 200 {
		t.Fatalf("the penalties: %d %v", code, out)
	}
	if smp := f.opts().Sampling; smp.PresencePen != 0.4 || smp.FreqPen != -0.25 {
		t.Errorf("the penalties reached the sampler as %v and %v", smp.PresencePen, smp.FreqPen)
	}
	// logit_bias in both forms reaches the generate: ids, texts and false.
	for body, want := range map[string][]TokenBias{
		`[[15,1.5],["ab",false]]`: {{ID: 15, Bias: 1.5}, {Text: "ab", Bias: float32(math.Inf(-1))}},
		`{"7":-2,"cd":false}`:     {{ID: 7, Bias: -2}, {Text: "cd", Bias: float32(math.Inf(-1))}},
	} {
		if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","logit_bias":`+body+`}`); code != 200 {
			t.Fatalf("logit_bias %s: %d %v", body, code, out)
		}
		if got := f.opts().LogitBias; !slices.Equal(got, want) {
			t.Errorf("logit_bias %s reached the generate as %v, want %v", body, got, want)
		}
	}
	// typical_p reaches the sampler.
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","typical_p":0.7,"samplers":["typ_p","temperature"]}`); code != 200 {
		t.Fatalf("typical_p: %d %v", code, out)
	}
	if got := f.opts().Sampling.TypicalP; got != 0.7 {
		t.Errorf("typical_p reached the sampler as %v", got)
	}
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","xtc_probability":0.5,"xtc_threshold":0.2}`); code != 200 {
		t.Fatalf("xtc: %d %v", code, out)
	}
	if smp := f.opts().Sampling; smp.XTCProbability != 0.5 || smp.XTCThreshold != 0.2 {
		t.Errorf("XTC reached the sampler as %v and %v", smp.XTCProbability, smp.XTCThreshold)
	}
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","top_n_sigma":1.5}`); code != 200 {
		t.Fatalf("top_n_sigma: %d %v", code, out)
	}
	if got := f.opts().Sampling.TopNSigma; got != 1.5 {
		t.Errorf("top_n_sigma reached the sampler as %v", got)
	}
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","dynatemp_range":0.5,"dynatemp_exponent":2}`); code != 200 {
		t.Fatalf("dynatemp: %d %v", code, out)
	}
	if smp := f.opts().Sampling; smp.DynatempRange != 0.5 || smp.DynatempExponent != 2 {
		t.Errorf("the dynamic temperature reached the sampler as %v and %v", smp.DynatempRange, smp.DynatempExponent)
	}
	if code, out := llamaPost(t, s, "/completion", `{"prompt":"x","mirostat":2,"mirostat_tau":4,"mirostat_eta":0.2}`); code != 200 {
		t.Fatalf("mirostat: %d %v", code, out)
	}
	if smp := f.opts().Sampling; smp.Mirostat != 2 || smp.MirostatTau != 4 || smp.MirostatEta != 0.2 {
		t.Errorf("Mirostat reached the sampler as %d, %v and %v", smp.Mirostat, smp.MirostatTau, smp.MirostatEta)
	}
	// The error body is llama-server's.
	_, out := llamaPost(t, s, "/completion", `{"prompt":"x","mirostat":3}`)
	e := out["error"].(map[string]any)
	if e["code"] != float64(400) || e["type"] != "invalid_request_error" || e["message"] == "" {
		t.Fatalf("error shape: %v", out)
	}
}

// TestLlamaResponseFields: response_fields keeps the paths asked for, a
// nested one lifted to the root under its path, and leaves out a path that
// names nothing; the stream's last frame is picked the same way, its token
// frames whole.
func TestLlamaResponseFields(t *testing.T) {
	f := &llamaFake{fakeBackend: &fakeBackend{tokens: []string{"Hel", "lo"}, reason: FinishMaxTokens}}
	s := llamaServe(t, f)
	code, out := llamaPost(t, s, "/completion", `{"prompt":"Hi","n_predict":2,
		"response_fields":["content","generation_settings/n_predict","timings/predicted_n","nowhere/at/all","content/deeper"]}`)
	if code != 200 {
		t.Fatalf("%d %v", code, out)
	}
	want := map[string]any{"content": "Hello", "generation_settings/n_predict": float64(2), "timings/predicted_n": float64(2)}
	if len(out) != len(want) {
		t.Fatalf("kept %v, want exactly %v", out, want)
	}
	for k, v := range want {
		if out[k] != v {
			t.Errorf("%s = %v, want %v", k, out[k], v)
		}
	}
	// An empty list keeps the whole object.
	if _, out := llamaPost(t, s, "/completion", `{"prompt":"Hi","response_fields":[]}`); out["timings"] == nil || out["tokens_predicted"] == nil {
		t.Fatalf("an empty response_fields dropped fields: %v", out)
	}
	resp := post(t, s, "/completion", `{"prompt":"Hi","stream":true,"response_fields":["stop","tokens_predicted"]}`)
	defer resp.Body.Close()
	var frames []map[string]any
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		var fr map[string]any
		if err := json.Unmarshal([]byte(line), &fr); err != nil {
			t.Fatalf("frame %q: %v", line, err)
		}
		frames = append(frames, fr)
	}
	if len(frames) < 2 {
		t.Fatalf("%d frames", len(frames))
	}
	last := frames[len(frames)-1]
	if len(last) != 2 || last["stop"] != true || last["tokens_predicted"] != float64(2) {
		t.Fatalf("the last frame: %v", last)
	}
	if frames[0]["content"] != "Hel" {
		t.Fatalf("a token frame was picked too: %v", frames[0])
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
	if code, _ := llamaPost(t, s, "/embedding", `{"content":"hi","embd_normalize":"x"}`); code != 400 {
		t.Fatalf("embd_normalize \"x\" accepted: %d", code)
	}
}

func TestLlamaEmbeddingShape(t *testing.T) {
	s := serve(t, &llamaEmbedFake{llamaFake: &llamaFake{fakeBackend: &fakeBackend{}}})
	for _, path := range []string{"/embedding", "/embeddings"} {
		t.Run(path, func(t *testing.T) { llamaEmbeddingShape(t, s, path) })
	}
}

// TestLlamaModelsIsV1Models: llama-server's /models is its /v1/models.
func TestLlamaModelsIsV1Models(t *testing.T) {
	s := serve(t, &fakeBackend{})
	read := func(path string) []byte {
		resp, err := http.Get(s.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d", path, resp.StatusCode)
		}
		b, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	var m struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	b := read("/models")
	if err := json.Unmarshal(b, &m); err != nil || len(m.Data) == 0 || m.Data[0].ID == "" {
		t.Fatalf("/models: %s (%v)", b, err)
	}
	if v := read("/v1/models"); !bytes.Equal(b, v) {
		t.Fatalf("/models %s, /v1/models %s", b, v)
	}
}

func llamaEmbeddingShape(t *testing.T, s *httptest.Server, path string) {
	resp := post(t, s, path, `{"content":["a","b"]}`)
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

// TestLlamaEmbeddingNormalize: embd_normalize reaches the embedder as asked,
// every kind llama-server reads, and none where the request names none.
func TestLlamaEmbeddingNormalize(t *testing.T) {
	f := &llamaEmbedFake{llamaFake: &llamaFake{fakeBackend: &fakeBackend{}}}
	s := serve(t, f)
	for _, c := range []struct {
		body string
		want *int
	}{
		{`{"content":"a"}`, nil}, {`{"content":"a","embd_normalize":-1}`, ptr(-1)}, {`{"content":"a","embd_normalize":0}`, ptr(0)},
		{`{"content":"a","embd_normalize":2}`, ptr(2)}, {`{"content":"a","embd_normalize":5}`, ptr(5)},
	} {
		resp := post(t, s, "/embedding", c.body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d", c.body, resp.StatusCode)
		}
		if got := f.norm; (got == nil) != (c.want == nil) || got != nil && *got != *c.want {
			t.Fatalf("%s: the embedder was asked %v", c.body, got)
		}
	}
}

type llamaEmbedFake struct {
	*llamaFake
	// norm is the normalisation the last request asked.
	norm *int
}

func (f *llamaEmbedFake) Embed(_ context.Context, o EmbedOptions) (*EmbedResult, error) {
	f.norm = o.Normalize
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
