package server

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"

	"github.com/jitllm/jitllm/model"
)

// OpenAI's logprobs and n.
//
// The log-probabilities are the model's raw distribution -- before
// temperature, the repeat penalty and top-k/top-p/min-p -- which is what vLLM
// returns by default (logprobs_mode "raw_logprobs"); model.Logprobs computes
// them with generated kernels.

// maxChoices bounds n: each choice is a session's KV history.
const maxChoices = 128

// oaChoiceSeeds is n's seeds: none for n of 1 (the request runs as it always
// did), seed+i for a seeded request, and n distinct seeds from a random base
// for an unseeded one.
func oaChoiceSeeds(n *int, seed *int64) ([]int64, error) {
	if n == nil || *n == 1 {
		return nil, nil
	}
	if *n < 1 || *n > maxChoices {
		return nil, fmt.Errorf("n must be between 1 and %d", maxChoices)
	}
	base := rand.Int64N(1 << 62)
	if seed != nil {
		base = *seed
	}
	out := make([]int64, *n)
	for i := range out {
		out[i] = base + int64(i)
	}
	return out, nil
}

// oaTopLogprob is one alternative in the chat shape.
type oaTopLogprob struct {
	Token   string  `json:"token"`
	Logprob float32 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

// oaLogprobContent is one sampled token in the chat shape.
type oaLogprobContent struct {
	Token       string         `json:"token"`
	Logprob     float32        `json:"logprob"`
	Bytes       []int          `json:"bytes"`
	TopLogprobs []oaTopLogprob `json:"top_logprobs"`
}

// oaChatLogprobs is a chat choice's (or delta's) logprobs.
type oaChatLogprobs struct {
	Content []oaLogprobContent `json:"content"`
}

// oaLegacyLogprobs is /v1/completions' logprobs: parallel arrays, the
// alternatives as a token -> logprob map.
type oaLegacyLogprobs struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float32            `json:"token_logprobs"`
	TopLogprobs   []map[string]float32 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

func oaBytes(s string) []int {
	out := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = int(s[i])
	}
	return out
}

func oaChatEntry(t *TokenLogprob) oaLogprobContent {
	c := oaLogprobContent{Token: t.Text, Logprob: t.Logprob, Bytes: oaBytes(t.Text),
		TopLogprobs: make([]oaTopLogprob, len(t.Top))}
	for i, a := range t.Top {
		c.TopLogprobs[i] = oaTopLogprob{Token: a.Text, Logprob: a.Logprob, Bytes: oaBytes(a.Text)}
	}
	return c
}

// add appends one token, its offset the completion text's length so far.
func (l *oaLegacyLogprobs) add(t *TokenLogprob, offset int) {
	l.Tokens = append(l.Tokens, t.Text)
	l.TokenLogprobs = append(l.TokenLogprobs, t.Logprob)
	m := make(map[string]float32, len(t.Top))
	for _, a := range t.Top {
		if _, ok := m[a.Text]; !ok { // two ids can decode alike; the likelier one stands
			m[a.Text] = a.Logprob
		}
	}
	l.TopLogprobs = append(l.TopLogprobs, m)
	l.TextOffset = append(l.TextOffset, offset)
}

// oaLogprobRequest is what both routes' logprobs fields come down to.
type oaLogprobRequest struct {
	on  bool
	top int
}

func (r oaLogprobRequest) apply(o *GenerateOptions) {
	o.Logprobs, o.TopLogprobs = r.on, r.top
}

// oaChatLogprobsOf reads chat's `logprobs` (a bool) and `top_logprobs`
// (0..20, only with logprobs), as the API refuses them.
func oaChatLogprobsOf(raw json.RawMessage, top *int) (oaLogprobRequest, error) {
	var on bool
	if len(raw) > 0 && string(raw) != "null" {
		if err := json.Unmarshal(raw, &on); err != nil {
			return oaLogprobRequest{}, fmt.Errorf("logprobs must be true or false on chat completions")
		}
	}
	return logprobRequestOf(on, top)
}

// logprobRequestOf is the check on a logprobs switch and its alternative
// count that both surfaces share.
func logprobRequestOf(on bool, top *int) (oaLogprobRequest, error) {
	r := oaLogprobRequest{on: on}
	if top != nil {
		if !r.on {
			return r, fmt.Errorf("top_logprobs needs logprobs set to true")
		}
		if *top < 0 || *top > model.MaxTopLogprobs {
			return r, fmt.Errorf("top_logprobs must be between 0 and %d", model.MaxTopLogprobs)
		}
		r.top = *top
	}
	return r, nil
}

// errLogprobsWithEcho is the refusal of logprobs on a generate that echoes its
// prompt, on both surfaces.
const errLogprobsWithEcho = "logprobs with echo would need the prompt's own " +
	"log-probabilities, which are not built"

// oaLegacyLogprobsOf reads /v1/completions' `logprobs`, an alternative count;
// top_logprobs is chat's field and is refused here rather than ignored.
func oaLegacyLogprobsOf(raw json.RawMessage, top *int) (oaLogprobRequest, error) {
	if top != nil {
		return oaLogprobRequest{}, fmt.Errorf("top_logprobs is a chat field; /v1/completions takes logprobs as a count")
	}
	if len(raw) == 0 || string(raw) == "null" {
		return oaLogprobRequest{}, nil
	}
	var n int
	if err := json.Unmarshal(raw, &n); err != nil {
		return oaLogprobRequest{}, fmt.Errorf("logprobs must be a count on /v1/completions")
	}
	if n < 0 || n > model.MaxTopLogprobs {
		return oaLogprobRequest{}, fmt.Errorf("logprobs must be between 0 and %d", model.MaxTopLogprobs)
	}
	return oaLogprobRequest{on: true, top: n}, nil
}
