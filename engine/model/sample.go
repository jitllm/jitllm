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
// The filters run penalty, top-k, top-p, min-p, then the temperature: top-p
// and min-p cut the candidates' softmax at temperature 1, and the temperature
// shapes only the draw over what survives them.
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
	q    []float32 // a top-k's softmax at temperature 1, which the cuts read
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
	// The window slides within its backing arrays: a slice cut from the
	// front would leave each append less room, and grow it again every
	// window or so of tokens.
	if len(s.hist) >= n {
		copy(s.hist, s.hist[len(s.hist)-n+1:])
		copy(s.hoff, s.hoff[len(s.hoff)-n+1:])
		s.hist, s.hoff = s.hist[:n-1], s.hoff[:n-1]
	}
	s.hist = append(s.hist, t)
	s.hoff = append(s.hoff, 4*int64(t))
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
	if len(logits) == 0 {
		return 0
	}
	return s.sampleAt(logits, float32(s.uniform()))
}

// sampleAt is Sample at the call's uniform u, drawn by the caller.
func (s *Sampler) sampleAt(logits []float32, u float32) int32 {
	n := len(logits)

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

	// One uniform draw per call (u): every round of a growth loop must use
	// the same u, or the stream would advance by a data-dependent amount.
	topp, minp := s.cuts()
	// With a top-k the distribution is the softmax over those k alone.
	if s.TopK > 0 && s.TopK < n {
		s.ord.Begin(vals)
		s.ids, s.p = grow32i(s.ids, s.TopK), grow32(s.p, s.TopK)
		have := 0
		for have < s.TopK {
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
		return s.drawBounded(have, u)
	}
	// Without one it is the softmax over the whole vocabulary, which a prefix
	// of the candidates cannot see: the cuts read the row's at temperature 1,
	// and a draw no cut decides reads it at the temperature.
	if minp > 0 || topp < 1 {
		if m := s.cutRow(vals, minp, topp); m > 0 {
			for i := range m {
				s.p[i] = vals[s.ids[i]]
			}
			return s.drawTempered(m, u)
		}
	}
	return s.drawRow(vals, u)
}

// cutRow orders the row's softmax at temperature 1 into s.ids and s.p only as
// deep as top-p and min-p reach, and returns how many candidates survive
// them, 0 when neither fires anywhere in the row. The candidates already
// taken stay as a round grows the list: a buffer reallocated there once lost
// them.
func (s *Sampler) cutRow(vals []float32, minp, topp float32) int {
	n := len(vals)
	s.pall = grow32(s.pall, n)
	copy(s.pall, vals)
	nn.Softmax32JIT(s.pall, n)
	s.ord.Begin(s.pall)
	have, want := 0, min(sampleGrowStart, n)
	for {
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
		if r := s.draw.Run(s.p[:have], minp, topp, 0, -1); r.Cut {
			return max(r.M, 1) // MinP > 1 cuts everything; the best is kept
		}
		if have < want || have == n {
			return 0
		}
		want = min(2*want, n)
	}
}

// drawRow is a draw no cut decides: the whole row's softmax at the
// temperature, ordered only as deep as the walk at u reaches, its divisor the
// row's mass.
func (s *Sampler) drawRow(vals []float32, u float32) int32 {
	n := len(vals)
	s.pall = grow32(s.pall, n)
	copy(s.pall, vals)
	nn.Scale32JIT(s.pall, float32(1/s.Temp))
	nn.Softmax32JIT(s.pall, n)
	s.ord.Begin(s.pall)
	sumAll := s.draw.Mass(s.pall)
	have, want := 0, min(sampleGrowStart, n)
	for {
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
		// A walk that terminated inside the prefix has found its token: every
		// candidate below the prefix is below the one it landed on.
		r := s.draw.Run(s.p[:have], 0, float32(math.Inf(1)), u, sumAll)
		if r.Resolved || have < want || have == n {
			return s.ids[min(max(r.Res, 0), have-1)]
		}
		want = min(2*want, n)
	}
}

// drawBounded is the end of a top-k draw: the candidates are the k best
// logits in s.ids and s.p, best first. Top-p and min-p cut their softmax at
// temperature 1, and the draw at u is over the survivors' softmax at the
// temperature. Sample and SampleFrom both end here, so a device's candidates
// draw what the host's would.
func (s *Sampler) drawBounded(have int, u float32) int32 {
	topp := float32(math.Inf(1))
	if s.TopP > 0 && s.TopP < 1 {
		topp = float32(s.TopP)
	}
	s.q = grow32(s.q, have)
	copy(s.q, s.p[:have])
	nn.Softmax32JIT(s.q, have)
	m := have
	if r := s.draw.Run(s.q, float32(s.MinP), topp, 0, -1); r.Cut {
		m = max(r.M, 1) // MinP > 1 cuts everything; the best is kept
	}
	return s.drawTempered(m, u)
}

// drawTempered draws at u from the softmax at the temperature over the m
// candidate logits in s.p, best first.
func (s *Sampler) drawTempered(m int, u float32) int32 {
	nn.Scale32JIT(s.p[:m], float32(1/s.Temp))
	nn.Softmax32JIT(s.p[:m], m)
	r := s.draw.Run(s.p[:m], 0, float32(math.Inf(1)), u, -1)
	return s.ids[min(max(r.Res, 0), m-1)]
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

// cuts is top-p (+Inf where off) and min-p (0 where off) as the draw reads
// them.
func (s *Sampler) cuts() (topp, minp float32) {
	topp = float32(math.Inf(1))
	if s.TopP > 0 && s.TopP < 1 {
		topp = float32(s.TopP)
	}
	return topp, float32(s.MinP)
}

// SampleFrom is Sample for a caller holding the top-k candidates already --
// the k best (penalized) logits, best first, the lower id on a tie, as a
// device's sampler returns them -- instead of the row. It draws the same
// uniform Sample would, so for the same candidates and seed it returns the
// same token.
func (s *Sampler) SampleFrom(vals []float32, ids []uint32) int32 {
	return s.sampleFromAt(vals, ids, float32(s.uniform()))
}

// sampleFromAt is SampleFrom at the call's uniform u, drawn by the caller.
func (s *Sampler) sampleFromAt(vals []float32, ids []uint32, u float32) int32 {
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
