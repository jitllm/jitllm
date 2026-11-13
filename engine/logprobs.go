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
	N   int          // alternatives per token, 0 to the vocabulary
	Top []TopLogprob // the last call's alternatives, most likely first
}

// SampledProbs is Logprobs after every sampler, as llama.cpp's
// post_sampling_probs reports it: chosen's probability under the
// distribution smp's last draw over logits drew from (Sample), and in l.Top
// the l.N likeliest tokens of what survived, the likeliest first, each
// Logprob a probability. It reads the draw as Sample left it in smp, so it
// follows Sample over the same logits, before the token is observed. A
// greedy smp's distribution is its token alone.
func (s *Instance) SampledProbs(l *Logprobs, smp *Sampler, logits []float32, chosen int32) (float32, error) {
	n := len(logits)
	if n != s.c.NVocab {
		return 0, fmt.Errorf("model: sampled probabilities over %d logits, and the vocabulary is %d", n, s.c.NVocab)
	}
	k := min(max(l.N, 0), n)
	l.Top = topOf(l.Top, k)
	if smp == nil || smp.Temp <= 0 {
		if k > 0 {
			l.Top = append(l.Top, TopLogprob{ID: chosen, Logprob: 1})
		}
		return 1, nil
	}
	if s.heads == nil {
		return 0, s.sampleDeclined("the sampled distribution")
	}
	s.post.pall = grow32(s.post.pall, n)
	a := smp.drew
	if smp.staged() {
		// The chain's stages again, XTC's at its chance as the draw's fell.
		row, ok := s.staged(logits, smp, a)
		if !ok {
			return 0, s.sampleDeclined("the sampled distribution")
		}
		logits, a = row, plainArgs(a.Infill)
	}
	dense, ids, vals, ok := false, []uint32(nil), []float32(nil), false
	if smp.adaptive() {
		// adaptive-p's transform at the words its draw read, a whole row.
		if ok = s.adaptive(logits, a, smp.adDrew); ok {
			dense = true
			copy(s.post.pall, s.ad.q)
		}
	} else {
		dense, ids, vals, ok = s.heads.Dist(logits, a, s.post.pall)
	}
	if !ok {
		return 0, s.sampleDeclined("the sampled distribution")
	}
	if dense {
		// The likeliest first, by the composed selection the log-softmax's
		// alternatives are read by.
		_, top, ok := s.heads.Logprobs(s.post.pall, k)
		if !ok {
			return 0, s.sampleDeclined("the sampled distribution")
		}
		for _, id := range top {
			if p := s.post.pall[id]; p > 0 && len(l.Top) < k {
				l.Top = append(l.Top, TopLogprob{ID: int32(id), Logprob: p})
			}
		}
		if chosen < 0 || int(chosen) >= n {
			return 0, nil
		}
		return s.post.pall[chosen], nil
	}
	// A draw's candidates are in their logits' order, and its survivors'
	// shares fall with it: the survivors in that order are the likeliest
	// first.
	var pc float32
	for i, id := range ids {
		p := vals[i]
		if int32(id) == chosen {
			pc = p
		}
		if p > 0 && len(l.Top) < k {
			l.Top = append(l.Top, TopLogprob{ID: int32(id), Logprob: p})
		}
	}
	return pc, nil
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
	row, ids, ok := s.heads.Logprobs(logits, l.N)
	if !ok {
		return 0, s.sampleDeclined("the logprobs")
	}
	k := min(max(l.N, 0), len(ids))
	l.Top = topOf(l.Top, k)
	for _, id := range ids[:k] {
		l.Top = append(l.Top, TopLogprob{ID: int32(id), Logprob: row[id]})
	}
	if chosen < 0 || int(chosen) >= n {
		return 0, nil
	}
	return row[chosen], nil
}

// topOf is top emptied, holding k alternatives and MaxTopLogprobs at least
// without growing: a warm call allocates nothing.
func topOf(top []TopLogprob, k int) []TopLogprob {
	if c := max(k, model.MaxTopLogprobs); cap(top) < c {
		return make([]TopLogprob, 0, c)
	}
	return top[:0]
}
