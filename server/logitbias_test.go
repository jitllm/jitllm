package server

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

// TestLogitBiasMovesTheDraw: on greedy decode, a ban of the first token
// plain decode draws keeps it out of every step, +100 on one token draws it
// at every step, and a bias of 0 leaves the decode as it was. Over HTTP the
// OpenAI route's bias and llama-server's give the engine's own text, and a
// bias out of range or on no token is refused.
func TestLogitBiasMovesTheDraw(t *testing.T) {
	sm := structuredModels()[0]
	e, _, c := loadedEngine(t, sm.file, sm.id, LoadOptions{})
	o := GenerateOptions{ModelID: sm.id, Prompt: Prompt{Kind: PromptText, Text: sm.prompt}, MaxTokens: 16}
	_, plain, _ := genText(t, e, o)
	if len(plain) < 8 {
		t.Fatalf("%d tokens: too short to show anything", len(plain))
	}
	o.LogitBias = []TokenBias{{ID: plain[0], Bias: 0}, {ID: plain[1], Bias: 0}}
	if _, got, _ := genText(t, e, o); !slices.Equal(got, plain) {
		t.Fatalf("a zero bias decodes %v, plain %v", got, plain)
	}
	o.LogitBias = []TokenBias{{ID: plain[0], Bias: float32(math.Inf(-1))}}
	_, banned, _ := genText(t, e, o)
	if len(banned) == 0 || slices.Contains(banned, plain[0]) {
		t.Fatalf("token %d banned, and the decode drew %v", plain[0], banned)
	}
	held := plain[1]
	o.LogitBias = []TokenBias{{ID: held, Bias: 100}}
	want, got, _ := genText(t, e, o)
	if len(got) != o.MaxTokens || slices.ContainsFunc(got, func(t int32) bool { return t != held }) {
		t.Fatalf("token %d at +100, and the decode drew %v", held, got)
	}

	var oa struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if code := postJSON(t, c.url, "/v1/completions", fmt.Sprintf(`{"model":%q,"prompt":%q,"max_tokens":16,`+
		`"temperature":0,"logit_bias":{"%d":100}}`, sm.id, sm.prompt, held), &oa); code != 200 || len(oa.Choices) != 1 ||
		oa.Choices[0].Text != want {
		t.Fatalf("OpenAI's logit_bias: %d %+v, the engine %q", code, oa, want)
	}
	var ll struct {
		Content string `json:"content"`
	}
	if code := postJSON(t, c.url, "/completion", fmt.Sprintf(`{"model":%q,"prompt":%q,"n_predict":16,"temperature":0,`+
		`"logit_bias":[[%d,100]]}`, sm.id, sm.prompt, held), &ll); code != 200 || ll.Content != want {
		t.Fatalf("llama-server's logit_bias: %d %q, the engine %q", code, ll.Content, want)
	}
	for _, bad := range []string{`{"x":1}`, `{"5":101}`, `{"99999999":1}`} {
		code, body := postRaw(t, c.url+"/v1/completions", fmt.Sprintf(`{"model":%q,"prompt":"x","max_tokens":4,`+
			`"logit_bias":%s}`, sm.id, bad))
		if code != 400 || !strings.Contains(body, "logit_bias") {
			t.Errorf("logit_bias %s: %d %s", bad, code, body)
		}
	}
}
