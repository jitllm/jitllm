package engine

import (
	"math"
	"math/rand"

	"github.com/jitllm/jitllm/nn"
)

// Sampler is how a token is drawn from logits: its configuration, the
// repeat window and the uniform stream. The zero value is greedy. The draw
// itself is the composed forms' (Instance.Sample: graph.Sample, SampleWalk,
// SampleRow and SampleCut), which read the window as a count row.
//
// The filters run penalty, top-n-sigma, top-k, typical-p, top-p, min-p,
// XTC, then the temperature (dynamic where DynatempRange is set),
// llama.cpp's order: typical-p, top-p and min-p cut the
// candidates' softmax at temperature 1, each over what the one before it
// left, and the temperature shapes only the draw over what survives them. Equal logits are ordered
// higher logit first, lower id on a tie.
type Sampler struct {
	Temp     float64 // 0 = greedy; llama.cpp's --temp
	TopK     int     // 0 = disabled
	TopP     float64 // 0 or 1 = disabled; nucleus
	MinP     float64 // 0 = disabled; keep tokens above MinP * p(best)
	TypicalP float64 // 0 or 1 = disabled; llama.cpp's typical_p
	// XTCProbability and XTCThreshold are llama.cpp's XTC: with that
	// probability a draw cuts the survivors at or above the threshold but
	// the least; off at probability 0 or a threshold past 0.5.
	XTCProbability, XTCThreshold float64
	TopNSigma                    float64 // 0 or below = disabled; llama.cpp's top_n_sigma
	// DynatempRange and DynatempExponent are llama.cpp's dynamic
	// temperature: off where the range is not above 0; an exponent of 0 is 1.
	DynatempRange, DynatempExponent float64
	// Mirostat is llama.cpp's Mirostat version, 0 for none, 1 or 2; it
	// replaces every other filter and the penalties, as llama.cpp's chain
	// does. MirostatTau (0 is 5) is its target surprise in bits, MirostatEta
	// (0 is 0.1) its learning rate.
	Mirostat                 int
	MirostatTau, MirostatEta float64
	// mu is Mirostat's running estimate, 2*tau before its first draw.
	mu          float32
	muSet       bool
	RepeatPen   float64 // 1 = disabled
	RepeatLastN int     // how far back the penalties look; 0 = 64
	// PresencePen and FreqPen are llama.cpp's presence and frequency
	// penalties over the same window: 0 = disabled.
	PresencePen, FreqPen float64
	Seed                 int64
	// Infill runs llama.cpp's infill sampler over the survivors before the
	// draw (graph.SampleWalk's Infill), at a top-k of at most
	// nn.InfillMaxK: a draw with no top-k, or one past it, draws as though
	// its top-k were that.
	Infill bool

	rng  *rand.Rand
	hist []int32
	// win is each token's count in hist, as the composed draws read the
	// window (graph.SampleTopK's sample.window): the penalty falls once on a
	// token whose count is above zero.
	win []float32
}

// Observe records a token so the penalties can see it. Harmless when they
// are disabled.
func (s *Sampler) Observe(t int32) {
	if s == nil || (s.RepeatPen == 1 || s.RepeatPen == 0) && s.PresencePen == 0 && s.FreqPen == 0 {
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

// penalties are the penalties a draw applies, none where the window holds
// no token.
func (s *Sampler) penalties() nn.Penalties {
	p := nn.NoPenalties()
	if len(s.hist) == 0 {
		return p
	}
	if s.RepeatPen > 1 {
		p[0] = float32(s.RepeatPen)
	}
	p[1], p[2] = float32(s.PresencePen), float32(s.FreqPen)
	return p
}

// Bounded reports whether a draw over n logits is a top-k draw, the one a
// device can select for (nn.Head.SampleK): sampled, with 0 < TopK < n, no
// top-n-sigma, which reads the whole row, and no infill sampler.
func (s *Sampler) Bounded(n int) bool {
	return s != nil && s.Temp > 0 && s.TopK > 0 && s.TopK < n && s.TopK <= MaxTopK && s.TopNSigma <= 0 && s.Mirostat == 0 &&
		!s.Infill
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

// typical is typical-p as the draw reads it: +Inf where off.
func (s *Sampler) typical() float32 {
	if s.TypicalP > 0 && s.TypicalP < 1 {
		return float32(s.TypicalP)
	}
	return float32(math.Inf(1))
}

// xtc is XTC's threshold as a draw reads it (nn.DrawArgs.XTC, 0 for none)
// and its probability, 0 where off. A threshold of 0 reads as the least
// float above it, which every survivor passes as it passes 0.
func (s *Sampler) xtc() (thr, p float32) {
	if s.XTCProbability <= 0 || s.XTCThreshold > 0.5 {
		return 0, 0
	}
	return max(float32(s.XTCThreshold), math.SmallestNonzeroFloat32), float32(min(s.XTCProbability, 1))
}

// xtcDraw is XTC's threshold for one draw: its chance taken from the
// stream after the draw's uniform, 0 where it misses or is off.
func (s *Sampler) xtcDraw() float32 {
	thr, p := s.xtc()
	if p == 0 || float32(s.uniform()) > p {
		return 0
	}
	return thr
}

// dyn is the dynamic temperature as a draw reads it: its range, 0 where off,
// and exponent.
func (s *Sampler) dyn() (rng, ex float32) {
	if s.DynatempRange <= 0 {
		return 0, 1
	}
	ex = float32(s.DynatempExponent)
	if ex == 0 {
		ex = 1
	}
	return float32(s.DynatempRange), ex
}

// miro is a Mirostat draw's tau, eta and mu, mu at 2*tau before the first.
func (s *Sampler) miro() (tau, eta float32, mu *float32) {
	tau, eta = float32(s.MirostatTau), float32(s.MirostatEta)
	if tau == 0 {
		tau = 5
	}
	if eta == 0 {
		eta = 0.1
	}
	if !s.muSet {
		s.mu, s.muSet = 2*tau, true
	}
	return tau, eta, &s.mu
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
