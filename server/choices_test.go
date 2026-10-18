package server

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestGenerateLogprobsAreTheModelsRawDistribution: the logprobs a generate
// reports at every sampled token are the oracle's log-softmax of the logits
// the model produced there, before temperature -- replayed on a fresh State of
// the same model from the token ids the generate returned.
func TestGenerateLogprobsAreTheModelsRawDistribution(t *testing.T) {
	e, lm, _ := loadedEngine(t, smallModel, "small", LoadOptions{})
	type step struct {
		id int32
		lp *TokenLogprob
	}
	var steps []step
	var prompt int
	err := e.Generate(context.Background(), GenerateOptions{
		ModelID: "small", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 12,
		Sampling: &model.Sampler{Temp: 0.9, Seed: 5}, Logprobs: true, TopLogprobs: 4,
	}, func(ev Event) error {
		switch ev.Kind {
		case EventStarted:
			prompt = ev.Started.PromptTokens
		case EventToken:
			if ev.Token.ID >= 0 {
				steps = append(steps, step{ev.Token.ID, ev.Token.Logprob})
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) < 4 {
		t.Fatalf("%d tokens, too few to compare", len(steps))
	}
	ids := lm.m.Vocab.Encode(story, true)
	if len(ids) != prompt {
		t.Fatalf("the prompt encodes to %d tokens here and %d in the generate", len(ids), prompt)
	}
	st := lm.m.NewState(len(ids) + len(steps) + 1)
	defer st.Close()
	logits, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range steps {
		if s.lp == nil || len(s.lp.Top) != 4 {
			t.Fatalf("token %d carries %+v, want a logprob and 4 alternatives", i, s.lp)
		}
		want := oracle.LogSoftmax32(logits)
		if d := math.Abs(float64(s.lp.Logprob) - want[s.id]); !(d < 1e-4) {
			t.Fatalf("token %d (id %d): logprob %v, the oracle's raw %v", i, s.id, s.lp.Logprob, want[s.id])
		}
		for j, a := range s.lp.Top {
			if d := math.Abs(float64(a.Logprob) - want[a.ID]); !(d < 1e-4) {
				t.Fatalf("token %d alternative %d (id %d): %v, the oracle %v", i, j, a.ID, a.Logprob, want[a.ID])
			}
			if j > 0 && a.Logprob > s.lp.Top[j-1].Logprob {
				t.Fatalf("token %d: alternatives out of order: %+v", i, s.lp.Top)
			}
		}
		if logits, err = st.Forward(s.id); err != nil {
			t.Fatal(err)
		}
	}
}

// legacyOut is the part of a /v1/completions body these gates read.
type legacyOut struct {
	Choices []struct {
		Index    int               `json:"index"`
		Text     string            `json:"text"`
		Logprobs *oaLegacyLogprobs `json:"logprobs"`
	} `json:"choices"`
	Usage  *oaUsage       `json:"usage"`
	Jitllm *oaJitllmExtra `json:"jitllm"`
}

func postJSON(t *testing.T, url, path, body string, out any) int {
	t.Helper()
	resp, err := http.Post(url+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode == 200 && out != nil {
		if err := json.Unmarshal(b, out); err != nil {
			t.Fatalf("POST %s: %v in %s", path, err, b)
		}
	}
	return resp.StatusCode
}

// streamLegacy posts a streamed /v1/completions and folds its chunks back
// into one body per choice.
func streamLegacy(t *testing.T, url, body string) legacyOut {
	t.Helper()
	resp, err := http.Post(url+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out legacyOut
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || line == "[DONE]" {
			continue
		}
		var c legacyOut
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("chunk %s: %v", line, err)
		}
		for _, ch := range c.Choices {
			for len(out.Choices) <= ch.Index {
				out.Choices = append(out.Choices, struct {
					Index    int               `json:"index"`
					Text     string            `json:"text"`
					Logprobs *oaLegacyLogprobs `json:"logprobs"`
				}{Index: len(out.Choices), Logprobs: &oaLegacyLogprobs{}})
			}
			o := &out.Choices[ch.Index]
			o.Text += ch.Text
			if ch.Logprobs != nil {
				o.Logprobs.Tokens = append(o.Logprobs.Tokens, ch.Logprobs.Tokens...)
				o.Logprobs.TokenLogprobs = append(o.Logprobs.TokenLogprobs, ch.Logprobs.TokenLogprobs...)
				o.Logprobs.TopLogprobs = append(o.Logprobs.TopLogprobs, ch.Logprobs.TopLogprobs...)
			}
		}
		if c.Usage != nil {
			out.Usage, out.Jitllm = c.Usage, c.Jitllm
		}
	}
	return out
}

// TestCompletionsLogprobsShape: /v1/completions' logprobs, whole and
// streamed, carry one entry per completion token with the asked number of
// alternatives, the chosen token's value never above the best alternative's,
// and the stream the same values as the whole response.
func TestCompletionsLogprobsShape(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	body := `{"model":"small","prompt":"Once upon a time","max_tokens":10,"temperature":0.8,"seed":3,"logprobs":3%s}`
	var whole legacyOut
	if code := postJSON(t, c.url, "/v1/completions", fmt.Sprintf(body, ""), &whole); code != 200 {
		t.Fatalf("status %d", code)
	}
	lp := whole.Choices[0].Logprobs
	if lp == nil || len(lp.Tokens) != whole.Usage.CompletionTokens || len(lp.TopLogprobs) != len(lp.Tokens) ||
		len(lp.TextOffset) != len(lp.Tokens) {
		t.Fatalf("logprobs %+v for %d completion tokens", lp, whole.Usage.CompletionTokens)
	}
	for i, v := range lp.TokenLogprobs {
		if len(lp.TopLogprobs[i]) == 0 || len(lp.TopLogprobs[i]) > 3 || v > 0 {
			t.Fatalf("entry %d: %v with alternatives %v", i, v, lp.TopLogprobs[i])
		}
		best := float32(math.Inf(-1))
		for _, a := range lp.TopLogprobs[i] {
			best = max(best, a)
		}
		if v > best+1e-6 {
			t.Fatalf("entry %d: the chosen token's %v is above the best alternative's %v", i, v, best)
		}
	}
	if strings.Join(lp.Tokens, "") != whole.Choices[0].Text {
		t.Fatalf("the tokens %q do not spell the text %q", lp.Tokens, whole.Choices[0].Text)
	}
	streamed := streamLegacy(t, c.url, fmt.Sprintf(body, `,"stream":true`))
	if len(streamed.Choices) != 1 || streamed.Choices[0].Text != whole.Choices[0].Text {
		t.Fatalf("streamed %+v, whole %q", streamed.Choices, whole.Choices[0].Text)
	}
	sl := streamed.Choices[0].Logprobs
	if fmt.Sprint(sl.TokenLogprobs) != fmt.Sprint(lp.TokenLogprobs) || fmt.Sprint(sl.Tokens) != fmt.Sprint(lp.Tokens) {
		t.Fatalf("streamed logprobs %v %q, whole %v %q", sl.TokenLogprobs, sl.Tokens, lp.TokenLogprobs, lp.Tokens)
	}
	// Refused as the API refuses them.
	for _, bad := range []string{
		`{"model":"small","prompt":"x","logprobs":21}`,
		`{"model":"small","prompt":"x","logprobs":2,"echo":true}`,
		`{"model":"small","prompt":"x","n":0}`,
	} {
		if code := postJSON(t, c.url, "/v1/completions", bad, nil); code != 400 {
			t.Fatalf("%s: status %d, want 400", bad, code)
		}
	}
	if code := postJSON(t, c.url, "/v1/chat/completions",
		`{"model":"small","messages":[{"role":"user","content":"x"}],"top_logprobs":2}`, nil); code != 400 {
		t.Fatalf("top_logprobs without logprobs: status %d, want 400", code)
	}
}

// TestChatLogprobsShape: the chat shape, whole and streamed, on a model with
// a template.
func TestChatLogprobsShape(t *testing.T) {
	_, _, c := loadedEngine(t, chatModel, "chat", LoadOptions{})
	type chatOut struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			Logprobs *oaChatLogprobs `json:"logprobs"`
		} `json:"choices"`
		Usage *oaUsage `json:"usage"`
	}
	body := `{"model":"chat","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0.7,"seed":1,"logprobs":true,"top_logprobs":2%s}`
	var whole chatOut
	if code := postJSON(t, c.url, "/v1/chat/completions", fmt.Sprintf(body, ""), &whole); code != 200 {
		t.Fatalf("status %d", code)
	}
	content := whole.Choices[0].Logprobs.Content
	if len(content) != whole.Usage.CompletionTokens {
		t.Fatalf("%d logprob entries for %d completion tokens", len(content), whole.Usage.CompletionTokens)
	}
	for i, x := range content {
		// A token that is part of a UTF-8 sequence shows as U+FFFD, and its bytes are
		// the raw ones, so the two lengths agree only for a whole character.
		if len(x.TopLogprobs) != 2 || (len(x.Bytes) != len(x.Token) && !strings.ContainsRune(x.Token, '\uFFFD')) || x.Logprob > 0 {
			t.Fatalf("entry %d: %+v", i, x)
		}
	}
	resp, err := http.Post(c.url+"/v1/chat/completions", "application/json",
		strings.NewReader(fmt.Sprintf(body, `,"stream":true`)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got []oaLogprobContent
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || line == "[DONE]" {
			continue
		}
		var ch chatOut
		if err := json.Unmarshal([]byte(line), &ch); err != nil {
			t.Fatal(err)
		}
		for _, x := range ch.Choices {
			if x.Logprobs != nil {
				got = append(got, x.Logprobs.Content...)
			}
		}
	}
	if len(got) != len(content) {
		t.Fatalf("the stream carried %d entries, the whole response %d", len(got), len(content))
	}
	for i := range got {
		if got[i].Token != content[i].Token || got[i].Logprob != content[i].Logprob {
			t.Fatalf("entry %d: streamed %+v, whole %+v", i, got[i], content[i])
		}
	}
}

// TestNChoicesEachMatchTheRequestAlone: n continuations of one prompt, seeded,
// are each the same request run alone with seed+i -- and the prompt ran once:
// the first choice computed every prompt position and every other restored all
// of them from the request's prompt store. Unseeded, the seeds are distinct.
func TestNChoicesEachMatchTheRequestAlone(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	const n, seed = 3, 11
	for _, stream := range []bool{false, true} {
		body := fmt.Sprintf(`{"model":"small","prompt":"Once upon a time","max_tokens":16,"temperature":1.0,"seed":%d,"n":%d,"logprobs":1,"stream":%v}`,
			seed, n, stream)
		var many legacyOut
		if stream {
			many = streamLegacy(t, c.url, body)
		} else if code := postJSON(t, c.url, "/v1/completions", body, &many); code != 200 {
			t.Fatalf("status %d", code)
		}
		if len(many.Choices) != n || many.Usage == nil || many.Jitllm == nil {
			t.Fatalf("stream=%v: %d choices, usage %v, jitllm %v", stream, len(many.Choices), many.Usage, many.Jitllm)
		}
		var alone legacyOut
		completion := 0
		distinct := map[string]bool{}
		for i := 0; i < n; i++ {
			if code := postJSON(t, c.url, "/v1/completions", fmt.Sprintf(
				`{"model":"small","prompt":"Once upon a time","max_tokens":16,"temperature":1.0,"seed":%d}`, seed+i), &alone); code != 200 {
				t.Fatalf("status %d", code)
			}
			if many.Choices[i].Text != alone.Choices[0].Text {
				t.Fatalf("stream=%v choice %d: %q, the request alone with seed %d: %q",
					stream, i, many.Choices[i].Text, seed+i, alone.Choices[0].Text)
			}
			completion += alone.Usage.CompletionTokens
			distinct[alone.Choices[0].Text] = true
		}
		if len(distinct) < 2 {
			t.Fatalf("every choice is %q: the seeds did not reach the sampler", many.Choices[0].Text)
		}
		if many.Usage.PromptTokens != alone.Usage.PromptTokens || many.Usage.CompletionTokens != completion {
			t.Fatalf("usage %+v, want the prompt once (%d) and %d completion tokens",
				many.Usage, alone.Usage.PromptTokens, completion)
		}
		// The prompt prefilled at most once: every choice after the first
		// restores all of it. The first may restore too, since the choices
		// run through the model's memory cache and an earlier request (the
		// streamed pass, or the request alone) left the prompt there.
		r := many.Jitllm.PromptRestored
		computed := 0
		for _, x := range r {
			computed += many.Usage.PromptTokens - x
		}
		later := len(r) == n
		for _, x := range r[min(1, len(r)):] {
			later = later && x == many.Usage.PromptTokens
		}
		if !later || computed > many.Usage.PromptTokens {
			t.Fatalf("restored per choice %v for a %d-token prompt: computed %d positions, want one prompt",
				r, many.Usage.PromptTokens, computed)
		}
		if fmt.Sprint(many.Jitllm.Seeds) != fmt.Sprint([]int64{seed, seed + 1, seed + 2}) {
			t.Fatalf("seeds %v", many.Jitllm.Seeds)
		}
	}

	var un legacyOut
	if code := postJSON(t, c.url, "/v1/completions",
		`{"model":"small","prompt":"Once upon a time","max_tokens":4,"temperature":1.0,"n":4}`, &un); code != 200 {
		t.Fatalf("status %d", code)
	}
	seen := map[int64]bool{}
	for _, s := range un.Jitllm.Seeds {
		seen[s] = true
	}
	if len(un.Choices) != 4 || len(seen) != 4 {
		t.Fatalf("unseeded n=4: %d choices, seeds %v", len(un.Choices), un.Jitllm.Seeds)
	}
}
