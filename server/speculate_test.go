package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestSpeculationMatchesPlainGreedy: a greedy generate with speculation on
// (prompt lookup: the instruct model has no prediction block) returns plain
// greedy's ids and finish reason, on a reply that ends on its end-of-turn
// token, on one cut by max_tokens inside a round, and on a copying prompt
// where the drafts are accepted; through Connect and through the OpenAI
// extension.
func TestSpeculationMatchesPlainGreedy(t *testing.T) {
	_, lm, c := loadedEngine(t, instructModel, "inst", LoadOptions{})
	questions := []string{
		ignoreEOSQuestion,
		"Repeat this sentence twice, word for word: the quick brown fox jumps over the lazy dog " +
			"while the farmer watches from the old wooden fence.",
	}
	for _, q := range questions {
		conv, err := lm.m.ChatIDsTools([]model.ChatMessage{{Role: "user", Content: q}}, nil, true)
		if err != nil {
			t.Fatal(err)
		}
		for _, max := range []int32{96, 13} {
			gen := func(spec *v1.SpeculationParams) completion {
				r := complete(c, &v1.GenerateRequest{ModelId: "inst", Prompt: ids(conv...), MaxTokens: max,
					Speculation: spec})
				if r.err != nil {
					t.Fatal(r.err)
				}
				return r
			}
			want := gen(nil)
			for _, spec := range []*v1.SpeculationParams{{Enabled: true}, {Enabled: true, DraftTokens: 2}} {
				got := gen(spec)
				if !slices.Equal(got.ids, want.ids) || got.finished.GetReason() != want.finished.GetReason() {
					t.Fatalf("%q max %d draft %d: speculation gave %v (%v), plain greedy %v (%v)", q, max,
						spec.DraftTokens, got.ids, got.finished.GetReason(), want.ids, want.finished.GetReason())
				}
				if got.started.GetBatched() {
					t.Fatal("a speculative generate ran as a row of the step loop")
				}
			}
			t.Logf("%q max %d: %d tokens, %v, equal with speculation", q[:24], max, len(want.ids),
				want.finished.GetReason())
		}
	}
	plain := postOAText(t, c.url, fmt.Sprintf(
		`{"model":"inst","messages":[{"role":"user","content":%q}],"max_tokens":64,"temperature":0}`, questions[1]))
	spec := postOAText(t, c.url, fmt.Sprintf(
		`{"model":"inst","messages":[{"role":"user","content":%q}],"max_tokens":64,"temperature":0,`+
			`"jitllm_speculate":true}`, questions[1]))
	if plain != spec || plain == "" {
		t.Fatalf("/v1/chat/completions: %q with jitllm_speculate, %q without", spec, plain)
	}
}

// TestSpeculationKeepsLogprobs: a greedy generate with speculation and
// logprobs gives plain greedy's tokens, each with the log-probability plain
// decode gives it -- a verification row's against decode's -- on a copying
// prompt whose drafts the trunk accepts.
func TestSpeculationKeepsLogprobs(t *testing.T) {
	e, lm, _ := loadedEngine(t, instructModel, "inst", LoadOptions{})
	conv, err := lm.m.ChatIDsTools([]model.ChatMessage{{Role: "user", Content: "Repeat this sentence twice, word for " +
		"word: the quick brown fox jumps over the lazy dog while the farmer watches from the old wooden fence."}}, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	run := func(spec *Speculation) ([]int32, []float32) {
		var ids []int32
		var lps []float32
		err := e.Generate(context.Background(), GenerateOptions{ModelID: "inst", Prompt: Prompt{Kind: PromptIDs, IDs: conv},
			MaxTokens: 48, Logprobs: true, TopLogprobs: 2, Speculation: spec}, func(ev Event) error {
			if ev.Kind == EventToken && ev.Token.ID >= 0 {
				if ev.Token.Logprob == nil {
					t.Fatalf("token %d came with no logprob", len(ids))
				}
				ids = append(ids, ev.Token.ID)
				lps = append(lps, ev.Token.Logprob.Logprob)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids, lps
	}
	want, wlp := run(nil)
	got, glp := run(&Speculation{Enabled: true})
	if !slices.Equal(got, want) || len(want) < 16 {
		t.Fatalf("speculation gave %v, plain greedy %v", got, want)
	}
	for i := range want {
		if d := math.Abs(float64(glp[i] - wlp[i])); d > 2e-2*(1+math.Abs(float64(wlp[i]))) {
			t.Fatalf("token %d: logprob %.5f with speculation, %.5f without", i, glp[i], wlp[i])
		}
	}
	t.Logf("%d tokens, each with plain decode's logprob", len(want))
}

// TestSpeculationRefusesAContinuePastItsRound: a session whose speculative
// generate stopped inside a round has rows in its model the reply did not
// keep; a continue_session on it is refused rather than run on them, and a
// fresh generate clears it.
func TestSpeculationRefusesAContinuePastItsRound(t *testing.T) {
	e, lm, c := loadedEngine(t, instructModel, "inst", LoadOptions{})
	if _, err := e.CreateSession(SessionOptions{ModelID: "inst", SessionID: "s", MaxSeq: 512,
		Speculation: Speculation{Enabled: true, Draft: 6}}); err != nil {
		t.Fatal(err)
	}
	conv, err := lm.m.ChatIDsTools([]model.ChatMessage{{Role: "user", Content: "Repeat this sentence twice: " +
		"one two three four five six seven eight nine ten."}}, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	// A stop string ends a generate wherever it matches; try each word until
	// one ends inside a round. (max_tokens never does: the Speculator is
	// limited to what the reply has room for.)
	cut := false
	for _, stop := range []string{"two", "three", "four", "five", "six", "seven", "eight", "nine"} {
		if cut {
			break
		}
		r := complete(c, &v1.GenerateRequest{SessionId: "s", Prompt: ids(conv...), MaxTokens: 64,
			Stop: []string{stop}})
		if r.err != nil {
			t.Fatal(r.err)
		}
		s, err := e.Session("s")
		if err != nil {
			t.Fatal(err)
		}
		cut = s.specRan
	}
	if !cut {
		t.Fatal("no cut length ended inside a speculative round: the refusal was never exercised")
	}
	r := complete(c, &v1.GenerateRequest{SessionId: "s", Prompt: ids(conv[len(conv)-1]), MaxTokens: 4,
		ContinueSession: true})
	if r.err == nil {
		t.Fatal("a continue past a cut speculative round ran")
	}
	if r := complete(c, &v1.GenerateRequest{SessionId: "s", Prompt: ids(conv...), MaxTokens: 4}); r.err != nil {
		t.Fatalf("a fresh generate after the refusal: %v", r.err)
	}
}

// postOAText is a chat completion's text.
func postOAText(t *testing.T, base, body string) string {
	t.Helper()
	text, _ := postOAChat(t, base, body)
	return text
}

// postOAChat is a chat completion's text and its finish_reason.
func postOAChat(t *testing.T, base, body string) (text, finish string) {
	t.Helper()
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("%s: %d %s", body, resp.StatusCode, b)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(b, &out); err != nil || len(out.Choices) != 1 {
		t.Fatalf("%s: %v in %s", body, err, b)
	}
	return out.Choices[0].Message.Content, out.Choices[0].FinishReason
}
