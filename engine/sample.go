package engine

import (
	"math"
	"math/rand"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// Sampler is how a token is drawn from logits: its configuration, the
// repeat window and the uniform stream. The zero value is greedy. The draw
// itself is the composed forms' (Instance.Sample: graph.Sample, SampleWalk,
// SampleRow and SampleCut), which read the window as a count row.
//
// The filters run penalty, the temperature, top-n-sigma, top-k, typical-p,
// top-p, min-p, then XTC, the reference's order (generate applies its
// temperature warper before top-k, top-p and min-p): typical-p, top-p and
// min-p cut the candidates' softmax at the temperature, each over what the
// one before it left, and a dynamic temperature (DynatempRange) shapes the
// draw over what survives them. Equal logits are ordered higher logit first,
// lower id on a tie.
type Sampler struct {
	Temp     float64 // 0 = greedy; the logits' divisor
	TopK     int     // 0 = disabled
	TopP     float64 // 0 or 1 = disabled; nucleus
	MinP     float64 // 0 = disabled; keep tokens above MinP * p(best)
	TypicalP float64 // 0 or 1 = disabled; locally typical sampling's mass
	// XTCProbability and XTCThreshold are XTC, exclude top choices: with that
	// probability a draw cuts the survivors at or above the threshold but
	// the least; off at probability 0 or a threshold past 0.5.
	XTCProbability, XTCThreshold float64
	TopNSigma                    float64 // 0 or below = disabled; logits n deviations under the largest are out
	// DynatempRange and DynatempExponent are the dynamic temperature, moved
	// with the survivors' entropy (graph.SampleStage): off where the range is
	// not above 0; an exponent of 0 is 1.
	DynatempRange, DynatempExponent float64
	// Mirostat is the Mirostat version, 0 for none, 1 or 2; it replaces
	// every other filter and the penalties (graph.SampleMirostat). MirostatTau (0 is 5) is its target surprise in bits, MirostatEta
	// (0 is 0.1) its learning rate.
	Mirostat                 int
	MirostatTau, MirostatEta float64
	// mu is Mirostat's running estimate, 2*tau before its first draw.
	mu          float32
	muSet       bool
	RepeatPen   float64 // 1 or 0 (unset) = disabled; any other value applies, below 1 too
	RepeatLastN int     // how far back the penalties look; 0 = 64
	// PresencePen and FreqPen are the presence and frequency penalties over
	// the same window: 0 = disabled.
	PresencePen, FreqPen float64
	// DryMultiplier, DryBase, DryAllowedLength and DryPenaltyLastN are
	// DRY, "don't repeat yourself": a token that would extend
	// a run the history repeats, at least DryAllowedLength long, falls by
	// DryMultiplier times DryBase to the run's length past it, over the last
	// DryPenaltyLastN tokens (-1 the instance's context). Off at a
	// multiplier of 0, a base below 1 or a window of 0. DryBreakers are the
	// sequence breakers, nil for "\n", ":", "\"" and "*" (dryBreakers): a run
	// does not reach back across one, and a token that is one alone is not
	// penalized.
	DryMultiplier, DryBase            float64
	DryAllowedLength, DryPenaltyLastN int
	DryBreakers                       []string
	// dhist is DRY's history: the tokens its window holds.
	dhist []int32
	Seed  int64
	// Infill runs the fill-in-the-middle draw over the survivors before the
	// draw (graph.SampleWalk's Infill), at a top-k of at most
	// nn.InfillMaxK: a draw with no top-k, or one past it, draws as though
	// its top-k were that.
	Infill bool
	// MinKeep is the fewest candidates a cut leaves: typical-p, top-p, min-p
	// and XTC each leave at least that many of theirs, at most nn.SampleMaxK
	// past a cut; 0 or 1 asks none past what the cut keeps. A draw asking more runs its
	// stages (the fused draw's order where Stages names none), min_keep after
	// each cut (graph.SampleKeep).
	MinKeep int
	// Adaptive draws through adaptive-p in place of the
	// plain walk (graph.SampleAdaptive): tokens near AdaptiveTarget's
	// probability, the target moved by a moving average of the drawn
	// tokens' probabilities decaying at AdaptiveDecay (held to [0, 0.99]).
	// A target below 0 is the plain walk, and Mirostat replaces it.
	Adaptive                      bool
	AdaptiveTarget, AdaptiveDecay float64
	// Stages runs the samplers in an order of the caller's (a request's
	// samplers list) where it is set: each over the row the one before it
	// leaves (graph.SampleStage), then the row's walk at temperature 1, or
	// adaptive-p or the infill sampler in its place. Nil is the fused draw,
	// the reference's order (Sampler), or TempLast's. A stage the list does not name does not run;
	// at temperature 0 the draw is the greedy one, as with every order.
	Stages []Stage
	// TempLast applies the temperature after every cut where Stages names
	// no order: penalties, DRY, top-n-sigma, top-k, typical-p, top-p, min-p
	// and XTC over the softmax at temperature 1, the temperature (dynamic
	// where DynatempRange is set) shaping only the draw over what survives
	// them -- the order the completion endpoint's samplers list states.
	TempLast bool
	// ad is adaptive-p's words for the next draw (graph.SampleAdaptive's
	// sample.adaptive), adSet once they are; adDrew the last draw's.
	ad, adDrew [6]float32
	adSet      bool

	rng  *rand.Rand
	hist []int32
	// win is each token's count in hist, as the composed draws read the
	// window (graph.SampleTopK's sample.window): the penalty falls once on a
	// token whose count is above zero.
	win []float32
	// drew is the last draw's arguments as its distribution reads them
	// (Instance.SampledProbs): XTC's cut at its whole weight where the draw
	// took it, Mirostat at the mu the draw read.
	drew nn.DrawArgs
}

// Prime makes a prompt's tokens the history the penalties and DRY read:
// they look back over the context, the prompt among it, from the first draw,
// so the window holds the prompt's tail. A negative id,
// a picture's row, is no token. The history is the sampler's own after it,
// never one a copy shares.
func (s *Sampler) Prime(prompt []int32) {
	s.hist, s.win, s.dhist = nil, nil, nil
	s.observeAll(prompt)
}

// PrimeSpans is Prime over a prompt of spans: their tokens, a picture's or a
// clip's rows none.
func (s *Sampler) PrimeSpans(spans []model.Span) {
	s.hist, s.win, s.dhist = nil, nil, nil
	for _, sp := range spans {
		if !sp.Rows() && sp.Clip == nil {
			s.observeAll(sp.Tokens)
		}
	}
}

func (s *Sampler) observeAll(ids []int32) {
	for _, t := range ids {
		if t >= 0 {
			s.Observe(t)
		}
	}
}

// Observe records a token so the penalties can see it. Harmless when they
// are disabled.
func (s *Sampler) Observe(t int32) {
	if s == nil {
		return
	}
	if s.dryOn() {
		s.dryObserve(t)
	}
	if (s.RepeatPen == 1 || s.RepeatPen == 0) && s.PresencePen == 0 && s.FreqPen == 0 {
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
	if s.RepeatPen != 0 {
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
		!s.Infill && !s.adaptive() && !s.staged()
}

// Stage is one sampler of a chain (Sampler.Stages).
type Stage uint8

const (
	StagePenalties Stage = iota
	StageTopNSigma
	StageTopK
	StageTypical
	StageTopP
	StageMinP
	StageXTC
	StageTemperature
	StageDRY
)

// defaultStage is the i-th of the default chain, the fused draw's order, the
// one a draw with DRY or a min_keep runs where it names none: the penalties,
// then the temperature before every cut, as the reference's generate applies
// its temperature warper before top-k, top-p and min-p.
func defaultStage(i int) Stage {
	return [...]Stage{StagePenalties, StageDRY, StageTemperature, StageTopNSigma, StageTopK, StageTypical, StageTopP,
		StageMinP, StageXTC}[i]
}

// lastStage is the i-th of TempLast's chain, the temperature after every
// cut.
func lastStage(i int) Stage {
	return [...]Stage{StagePenalties, StageDRY, StageTopNSigma, StageTopK, StageTypical, StageTopP, StageMinP, StageXTC,
		StageTemperature}[i]
}

// chain is how many stages a staged draw runs and the i-th of them: its
// own, or the fused draw's order where it names none.
func (s *Sampler) chain() (int, func(int) Stage) {
	if s.Stages != nil {
		return len(s.Stages), func(i int) Stage { return s.Stages[i] }
	}
	if s.TempLast {
		return 9, lastStage
	}
	return 9, defaultStage
}

// dryOn reports whether a draw runs DRY.
func (s *Sampler) dryOn() bool {
	return s != nil && s.DryMultiplier != 0 && s.DryBase >= 1 && s.DryPenaltyLastN != 0
}

// greedyMoved reports whether a greedy draw's row is not the logits as they
// are: its penalties or DRY move them, and they run before the temperature,
// whose greedy draw keeps the largest of what they leave.
func (s *Sampler) greedyMoved() bool { return s != nil && (!s.penalties().Off() || s.dryOn()) }

// staged reports whether a draw runs its stages in an order of their own.
func (s *Sampler) staged() bool {
	return s != nil && (s.Stages != nil || s.dryOn() || s.MinKeep > 1) && s.Temp > 0 && s.Mirostat == 0
}

// adaptive reports whether a draw runs adaptive-p.
func (s *Sampler) adaptive() bool {
	return s != nil && s.Adaptive && s.AdaptiveTarget >= 0 && s.Temp > 0 && s.Mirostat == 0
}

// adaptiveWords is adaptive-p's words for the next draw: its average starts
// at target/(1-decay) over a weight of 1/(1-decay), the average a long run
// of draws at the target holds.
func (s *Sampler) adaptiveWords() [6]float32 {
	if !s.adSet {
		d := float32(min(max(s.AdaptiveDecay, 0), 0.99))
		t := float32(s.AdaptiveTarget)
		s.ad, s.adSet = [6]float32{t, d, t / (1 - d), 1 / (1 - d), 0, 0}, true
	}
	return s.ad
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
