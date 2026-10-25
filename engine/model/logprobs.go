package model

import "github.com/jitllm/jitllm/engine/nn"

// MaxTopLogprobs is the most alternatives a Logprobs reports per token, the
// OpenAI API's bound on top_logprobs.
const MaxTopLogprobs = 20

// TopLogprob is one candidate token and its log-probability.
type TopLogprob struct {
	ID      int32
	Logprob float32
}

// Logprobs reports a step's log-probabilities: the chosen token's and the N
// most likely tokens'.
//
// The distribution is the model's own -- the log-softmax of the logits the
// head returned, final softcap and head bias included -- before temperature,
// the repeat penalty and every cut. That is what vLLM returns by default
// (logprobs_mode "raw_logprobs") and what the OpenAI API reports: a
// log-probability that does not move with the request's temperature.
//
// Both the log-softmax and the selection are generated code
// (nn.LogSoftmax32JIT, then nn.SampleOrder's strict total order over the
// logits: higher value first, lower id on a tie). The values carry the
// engine's exp, ~5e-5 relative at worst, so a log-probability is good to
// ~1e-4 absolute. The logits are on the host wherever a token is
// sampled -- Forward reads them back; only a greedy ForwardGreedy keeps them
// on a device, and a request for logprobs does not take that path. A warm
// Take allocates nothing: the row, the ordering and Top are reused, so Top is
// valid until the next call.
type Logprobs struct {
	N   int          // alternatives per token, 0..MaxTopLogprobs
	Top []TopLogprob // the last Take's alternatives, most likely first

	row []float32
	ord nn.SampleOrder
}

// Take returns the log-probability of chosen under logits, and fills Top with
// the N most likely tokens. It does not modify logits.
func (l *Logprobs) Take(logits []float32, chosen int32) float32 {
	n := len(logits)
	if n == 0 {
		l.Top = l.Top[:0]
		return 0
	}
	l.row = grow32(l.row, n)
	copy(l.row, logits)
	nn.LogSoftmax32JIT(l.row, n)
	k := min(max(l.N, 0), MaxTopLogprobs, n)
	if cap(l.Top) < k {
		l.Top = make([]TopLogprob, 0, MaxTopLogprobs)
	}
	l.Top = l.Top[:0]
	if k > 0 {
		// Ordered on the logits rather than the log row: the same order, and
		// x - c in f32 can round two close logits onto one value.
		l.ord.Begin(logits)
		for len(l.Top) < k {
			_, id, ok := l.ord.Next()
			if !ok {
				break
			}
			l.Top = append(l.Top, TopLogprob{ID: id, Logprob: l.row[id]})
		}
	}
	if chosen < 0 || int(chosen) >= n {
		return 0
	}
	return l.row[chosen]
}
