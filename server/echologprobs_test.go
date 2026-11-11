package server

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

// TestEchoLogprobsArePromptTokens: /v1/completions with echo and logprobs
// sends a log-probability for every prompt token, the first null; and a
// prompt that ends in a greedy reply's own tokens gets, at each of them, the
// log-probability decode gave it -- the prompt's rows against decode's, on
// stories15M.
func TestEchoLogprobsArePromptTokens(t *testing.T) {
	sm := structuredModels()[0]
	e, lm, c := loadedEngine(t, sm.file, sm.id, LoadOptions{})
	ids := lm.m.Vocab.Encode(sm.prompt, true)
	var reply []int32
	var lps []float32
	err := e.Generate(context.Background(), GenerateOptions{ModelID: sm.id, Prompt: Prompt{Kind: PromptIDs, IDs: ids},
		MaxTokens: 4, Logprobs: true}, func(ev Event) error {
		if ev.Kind == EventToken && ev.Token.ID >= 0 {
			reply = append(reply, ev.Token.ID)
			lps = append(lps, ev.Token.Logprob.Logprob)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(reply) != 4 {
		t.Fatalf("the greedy reply is %v", reply)
	}
	prompt := append(append([]int32(nil), ids...), reply[:3]...)
	var nums []string
	for _, id := range prompt {
		nums = append(nums, fmt.Sprint(id))
	}
	var out struct {
		Choices []struct {
			Text     string `json:"text"`
			Logprobs struct {
				Tokens        []string             `json:"tokens"`
				TokenLogprobs []*float32           `json:"token_logprobs"`
				TopLogprobs   []map[string]float32 `json:"top_logprobs"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	body := fmt.Sprintf(`{"model":%q,"prompt":[%s],"max_tokens":1,"temperature":0,"echo":true,"logprobs":2}`, sm.id, strings.Join(nums, ","))
	if code := postJSON(t, c.url, "/v1/completions", body, &out); code != 200 || len(out.Choices) != 1 {
		t.Fatalf("echo with logprobs: %d %+v", code, out)
	}
	lp := out.Choices[0].Logprobs
	if len(lp.TokenLogprobs) != len(prompt)+1 || len(lp.Tokens) != len(prompt)+1 || len(lp.TopLogprobs) != len(prompt)+1 {
		t.Fatalf("%d logprobs, %d tokens, %d alternatives for %d prompt tokens and one reply", len(lp.TokenLogprobs),
			len(lp.Tokens), len(lp.TopLogprobs), len(prompt))
	}
	if lp.TokenLogprobs[0] != nil || lp.TopLogprobs[0] != nil {
		t.Fatalf("the first prompt token's logprob is %v, its alternatives %v: want null", lp.TokenLogprobs[0], lp.TopLogprobs[0])
	}
	for i, v := range lp.TokenLogprobs[1:] {
		if v == nil || *v > 0 || math.IsInf(float64(*v), 0) || len(lp.TopLogprobs[i+1]) == 0 {
			t.Fatalf("token %d: logprob %v, alternatives %v", i+1, v, lp.TopLogprobs[i+1])
		}
	}
	for i := range 4 {
		got := *lp.TokenLogprobs[len(ids)+i]
		if d := math.Abs(float64(got - lps[i])); d > 2e-2*(1+math.Abs(float64(lps[i]))) {
			t.Fatalf("reply token %d: %.5f as a prompt row, %.5f decoded", i, got, lps[i])
		}
	}
	raw, _ := json.Marshal(lp.TokenLogprobs[:3])
	t.Logf("prompt logprobs start %s; the reply's rows match decode's", raw)
}
