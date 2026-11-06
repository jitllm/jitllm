package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"
)

// TopLogprob is one candidate token and its log-probability.
type TopLogprob struct {
	ID      int32
	Logprob float32
}

// Logprobs is a step's log-probabilities as Instance.Logprobs reports them:
// N alternatives per token, and the last call's in Top, most likely first,
// valid until the next call.
//
// The distribution is the model's own -- the log-softmax of the logits the
// head returned, final softcap and head bias included -- before temperature,
// the repeat penalty and every cut. That is what vLLM returns by default
// (logprobs_mode "raw_logprobs") and what the OpenAI API reports: a
// log-probability that does not move with the request's temperature.
type Logprobs struct {
	N   int          // alternatives per token, 0..MaxTopLogprobs
	Top []TopLogprob // the last call's alternatives, most likely first
}

// MaskRow writes logits plus mask into dst, rows of this instance's
// vocabulary -- mask 0 where a constraint allows a token and -inf where it
// does not -- through the CPU's composed form (graph.MaskRow).
func (s *Instance) MaskRow(dst, logits, mask []float32) error {
	n := s.c.NVocab
	if len(dst) != n || len(logits) != n || len(mask) != n {
		return fmt.Errorf("model: a mask over %d logits into %d by %d, and the vocabulary is %d", len(logits), len(dst), len(mask), n)
	}
	if s.heads == nil || !s.heads.Mask(dst, logits, mask) {
		return s.sampleDeclined("the mask")
	}
	return nil
}

// Logprobs returns the log-probability of chosen under logits, a row of
// this instance's vocabulary, and fills l.Top with the l.N most likely
// tokens, from the CPU's composed log-softmax and selection
// (graph.Logprobs: the larger logit first, the lower id on a tie). It does
// not modify logits, and a warm call allocates nothing.
func (s *Instance) Logprobs(l *Logprobs, logits []float32, chosen int32) (float32, error) {
	n := len(logits)
	if n != s.c.NVocab {
		return 0, fmt.Errorf("model: logprobs over %d logits, and the vocabulary is %d", n, s.c.NVocab)
	}
	if s.heads == nil {
		return 0, s.sampleDeclined("the logprobs")
	}
	row, ids, ok := s.heads.Logprobs(logits)
	if !ok {
		return 0, s.sampleDeclined("the logprobs")
	}
	k := min(max(l.N, 0), len(ids))
	if cap(l.Top) < model.MaxTopLogprobs {
		l.Top = make([]TopLogprob, 0, model.MaxTopLogprobs)
	}
	l.Top = l.Top[:0]
	for _, id := range ids[:k] {
		l.Top = append(l.Top, TopLogprob{ID: int32(id), Logprob: row[id]})
	}
	if chosen < 0 || int(chosen) >= n {
		return 0, nil
	}
	return row[chosen], nil
}
