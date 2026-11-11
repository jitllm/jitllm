package engine

import (
	"math"
	"math/rand"
)

// Sampler is how a token is drawn from logits: its configuration, the
// repeat window and the uniform stream. The zero value is greedy. The draw
// itself is the composed forms' (Instance.Sample: graph.Sample, SampleWalk,
// SampleRow and SampleCut), which read the window as a count row.
//
// The filters run penalty, top-k, top-p, min-p, then the temperature: top-p
// and min-p cut the candidates' softmax at temperature 1, and the temperature
// shapes only the draw over what survives them. Equal logits are ordered
// higher logit first, lower id on a tie.
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
	// win is each token's count in hist, as the composed draws read the
	// window (graph.SampleTopK's sample.window): the penalty falls once on a
	// token whose count is above zero.
	win []float32
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
	// The window slides within its backing array, reserved once: a slice cut
	// from the front would leave each append less room, and grow it again
	// every window or so of tokens.
	if cap(s.hist) < n {
		s.hist = append(make([]int32, 0, n), s.hist...)
	}
	if len(s.hist) >= n {
		drop := len(s.hist) - n + 1
		for _, e := range s.hist[:drop] {
			s.win[e]--
		}
		copy(s.hist, s.hist[drop:])
		s.hist = s.hist[:n-1]
	}
	s.hist = append(s.hist, t)
	s.reserve(int(t) + 1)
	s.win[t]++
}

// reserve holds the window's count row at n tokens at least: the vocabulary,
// once the engine draws, so a warm token grows nothing.
func (s *Sampler) reserve(n int) {
	if len(s.win) < n {
		s.win = append(s.win, make([]float32, n-len(s.win))...)
	}
}

// window is the count row over a vocabulary of n tokens.
func (s *Sampler) window(n int) []float32 {
	s.reserve(n)
	return s.win[:n]
}

// penalty is the repeat penalty a draw applies, 1 where it applies none.
func (s *Sampler) penalty() float32 {
	if s.RepeatPen > 1 && len(s.hist) > 0 {
		return float32(s.RepeatPen)
	}
	return 1
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

// cuts is top-p (+Inf where off) and min-p (0 where off) as the draw reads
// them.
func (s *Sampler) cuts() (topp, minp float32) {
	topp = float32(math.Inf(1))
	if s.TopP > 0 && s.TopP < 1 {
		topp = float32(s.TopP)
	}
	return topp, float32(s.MinP)
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
