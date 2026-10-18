package model

import (
	"math"
	"math/rand"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
)

// Sampler turns logits into a token. The zero value is greedy.
//
// Greedy must stay the default and bit-exact: the correctness gates against
// llama.cpp and between tiers compare argmax, so Temp == 0 short-circuits to
// Greedy before any other field is read.
//
// The filters run in llama.cpp's order (penalty, top-k, min-p, top-p), so a
// prompt behaves comparably across engines.
//
// Every vocabulary-sized operation is generated code: the ordering is a
// segmented selection that takes only as many candidates as the cuts consume
// (nn.SampleOrder), the penalty is a scatter over the history, and the cuts and
// the inverse-CDF walk are one kernel over the survivors. A warm Sample
// allocates nothing (TestSamplerAllocatesNothingWarm).
//
// Equal logits are ordered higher logit first, lower id on a tie, the same
// strict total order the router's top-k uses, so a seed draws the same token
// on every toolchain.
type Sampler struct {
	Temp        float64 // 0 = greedy; llama.cpp's --temp
	TopK        int     // 0 = disabled
	TopP        float64 // 0 or 1 = disabled; nucleus
	MinP        float64 // 0 = disabled; keep tokens above MinP * p(best)
	RepeatPen   float64 // 1 = disabled
	RepeatLastN int     // how far back RepeatPen looks; 0 = 64
	Seed        int64

	rng  *rand.Rand
	hist []int32
	// hoff is hist as byte offsets into a logits row, which is what the penalty
	// kernel addresses with. It is built here rather than in the kernel because
	// none of the three tiers' assemblers has a sign-extending 32-bit load into
	// a general register.
	hoff []int64

	// The per-token working set, allocated on the first call and reused.
	ord  nn.SampleOrder
	draw nn.SampleDraw
	ids  []int32   // the candidates, descending
	p    []float32 // their probabilities, descending
	pen  []float32 // the penalized logits, only when RepeatPen > 1
	pall []float32 // the whole row's softmax, only when top-k is disabled
}

// Observe records a token so RepeatPen can see it. Harmless when disabled.
func (s *Sampler) Observe(t int32) {
	if s == nil || s.RepeatPen == 1 || s.RepeatPen == 0 {
		return
	}
	n := s.RepeatLastN
	if n <= 0 {
		n = 64
	}
	s.hist = append(s.hist, t)
	s.hoff = append(s.hoff, 4*int64(t))
	if len(s.hist) > n {
		s.hist = s.hist[len(s.hist)-n:]
		s.hoff = s.hoff[len(s.hoff)-n:]
	}
}

// sampleGrowStart is how many candidates the unbounded configuration takes
// before it asks whether that was enough.
//
// It is a starting point, not a limit: with no top-k the data decides how deep
// the ordering goes, and a flat distribution doubles its way down.
const sampleGrowStart = 64

// Sample picks a token. It does not modify logits.
func (s *Sampler) Sample(logits []float32) int32 {
	if s == nil || s.Temp <= 0 {
		return Greedy(logits)
	}
	n := len(logits)
	if n == 0 {
		return 0
	}
	s.seed()

	// The repeat penalty first, on the logits, as llama.cpp does: a positive
	// logit is divided by the penalty and a negative one multiplied. It is a
	// scatter over the history; the copy keeps logits unmodified and is skipped
	// when the penalty is off.
	vals := logits
	if s.RepeatPen > 1 && len(s.hoff) > 0 {
		s.pen = grow32(s.pen, n)
		copy(s.pen, logits)
		nn.SamplePenalty32JIT(s.pen, s.hoff, float32(s.RepeatPen))
		vals = s.pen
	}

	// With a top-k the distribution is the softmax over those k alone. Without
	// one it is the softmax over the whole vocabulary, which a prefix of the
	// candidates cannot see, so that path computes the row's probabilities
	// once and hands the total mass to the draw kernel.
	bounded := s.TopK > 0 && s.TopK < n
	order, minp, topp := vals, float32(s.MinP), float32(math.Inf(1))
	if s.TopP > 0 && s.TopP < 1 {
		topp = float32(s.TopP)
	}
	if !bounded {
		s.pall = grow32(s.pall, n)
		copy(s.pall, vals)
		nn.Scale32JIT(s.pall, float32(1/s.Temp))
		nn.Softmax32JIT(s.pall, n)
		order = s.pall
	}

	s.ord.Begin(order)

	// One uniform draw per call: every round of the growth loop must use the
	// same u, or the stream would advance by a data-dependent amount.
	u := float32(s.rng.Float64())

	// The whole row's mass is only the divisor when no cut fires, so it is
	// computed lazily, after a round comes back uncut.
	sumAll, haveSum := float32(-1), bounded

	want := sampleGrowStart
	if bounded {
		want = s.TopK
	}
	have := 0
	for {
		// The candidates already taken stay: a round grows the list, and a
		// buffer reallocated here once lost them, so the first draw a Sampler
		// made past its first round walked zeros.
		s.ids = slices.Grow(s.ids[:have], want-have)[:want]
		s.p = slices.Grow(s.p[:have], want-have)[:want]
		for have < want {
			v, id, ok := s.ord.Next()
			if !ok {
				break
			}
			s.ids[have], s.p[have] = id, v
			have++
		}
		if have == 0 {
			return 0
		}
		if bounded {
			return s.drawBounded(have, u)
		}
		r := s.draw.Run(s.p[:have], minp, topp, u, sumAll)
		if !r.Cut && !haveSum {
			// The cuts did not decide it, so the divisor is the whole row's
			// mass. Mass skips the kernel's walk.
			sumAll, haveSum = s.draw.Mass(s.pall), true
			continue
		}
		// A cut decides the surviving set on its own; without one, a walk that
		// terminated inside the prefix has already found its token, because
		// every candidate below the prefix is below the one it landed on.
		if r.Cut || r.Resolved || have == n {
			res := r.Res
			if res < 0 {
				res = 0 // MinP > 1 cuts everything; the Go loop indexed c[-1]
			}
			if res >= have {
				res = have - 1
			}
			return s.ids[res]
		}
		// Not enough candidates to decide. Take twice as many; the ordering
		// resumes rather than sweeping the vocabulary again.
		want *= 2
		if want > n {
			want = n
		}
		if have == want {
			return s.ids[have-1]
		}
	}
}

// drawBounded is the end of a top-k draw: the candidates are the k best
// logits in s.ids and s.p, best first, and the distribution is the softmax
// over exactly these k at temperature T, cut by min-p and top-p and walked at
// u. Sample and SampleFrom both end here, so a device's candidates draw what
// the host's would.
func (s *Sampler) drawBounded(have int, u float32) int32 {
	topp := float32(math.Inf(1))
	if s.TopP > 0 && s.TopP < 1 {
		topp = float32(s.TopP)
	}
	nn.Scale32JIT(s.p[:have], float32(1/s.Temp))
	nn.Softmax32JIT(s.p[:have], have)
	r := s.draw.Run(s.p[:have], float32(s.MinP), topp, u, -1)
	res := r.Res
	if res < 0 {
		res = 0 // MinP > 1 cuts everything; the Go loop indexed c[-1]
	}
	if res >= have {
		res = have - 1
	}
	return s.ids[res]
}

// Bounded reports whether a draw over n logits is a top-k draw, the one a
// device can select for (nn.Head.SampleK): sampled, with 0 < TopK < n.
func (s *Sampler) Bounded(n int) bool {
	return s != nil && s.Temp > 0 && s.TopK > 0 && s.TopK < n && s.TopK <= MaxTopK
}

// MaxTopK is above every vocabulary a container states, so a top-k past it is
// no top-k at all. TopK comes from a request; the candidate buffers a bounded
// draw sizes from it are held to this where they are sized.
const MaxTopK = 1 << 20

// DeviceArgs is the device sampler's argument block (kernels.SampleArgsWords):
// the history's length, the penalty's bits and the history, which the device
// penalizes exactly as Sample's scatter does -- once per entry, in order.
// The history is empty when the penalty is off. dst is reused.
func (s *Sampler) DeviceArgs(dst []uint32) []uint32 {
	dst = dst[:0]
	if s.RepeatPen > 1 && len(s.hist) > 0 {
		dst = append(dst, uint32(len(s.hist)), math.Float32bits(float32(s.RepeatPen)))
		for _, t := range s.hist {
			dst = append(dst, uint32(t))
		}
		return dst
	}
	return append(dst, 0, math.Float32bits(1))
}

// SampleFrom is Sample for a caller holding the top-k candidates already --
// the k best (penalized) logits, best first, the lower id on a tie, as a
// device's sampler returns them -- instead of the row. It draws the same
// uniform Sample would, so for the same candidates and seed it returns the
// same token.
func (s *Sampler) SampleFrom(vals []float32, ids []uint32) int32 {
	s.seed()
	u := float32(s.rng.Float64())
	k := len(vals)
	if k == 0 {
		return 0
	}
	s.ids = grow32i(s.ids, k)
	s.p = grow32(s.p, k)
	copy(s.p, vals)
	for i, id := range ids[:k] {
		s.ids[i] = int32(id)
	}
	return s.drawBounded(k, u)
}

// seed starts the sampler's stream on first use.
func (s *Sampler) seed() {
	if s.rng == nil {
		s.rng = rand.New(rand.NewSource(s.Seed))
	}
}

func grow32(b []float32, n int) []float32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]float32, n)
}

func grow32i(b []int32, n int) []int32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]int32, n)
}
