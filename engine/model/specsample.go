package model

import (
	"math"

	"github.com/jitllm/jitllm/engine/nn"
)

// What speculative sampling needs of the Sampler beyond Sample: the
// probability its distribution gives one token, a uniform draw from its
// stream, and a history that can be taken back.
//
// A rejection test needs p(x) and q(x) at the drafted token x, for the target
// and the draft: the sampler's own distribution -- the penalty, the
// temperature, and the cuts before it -- not a raw softmax, or the
// output is not distributed as plain sampling's.

// specDist is one row's distribution under a Sampler: either the whole row's
// softmax (no cut configured, so every token survives) or the candidates that
// survive the cuts, descending, with their probabilities normalised over the
// survivors.
type specDist struct {
	// dense says every token survives and pall is the distribution.
	dense bool
	pall  []float32
	ids   []int32
	p     []float32
	// The scratch the ordering and the draw reuse.
	ord  nn.SampleOrder
	draw nn.SampleDraw
	vals []float32
}

// prob is the distribution's probability of token t.
func (d *specDist) prob(t int32) float32 {
	if d.dense {
		if int(t) < 0 || int(t) >= len(d.pall) {
			return 0
		}
		return d.pall[t]
	}
	for i, id := range d.ids {
		if id == t {
			return d.p[i]
		}
	}
	return 0
}

// dist computes the distribution Sample draws from, for logits under the
// sampler's current history. It mirrors Sample step for step -- the penalty,
// then the top-k softmax or the whole row's, then min-p and top-p by the same
// draw kernel -- and differs only in walking the candidates to the cut rather
// than to a uniform draw. Every vocabulary-sized operation is the same
// generated code Sample runs.
func (s *Sampler) dist(logits []float32, d *specDist) {
	n := len(logits)
	vals := logits
	if s.RepeatPen > 1 && len(s.hoff) > 0 {
		d.vals = grow32(d.vals, n)
		copy(d.vals, logits)
		nn.SamplePenalty32JIT(d.vals, s.hoff, float32(s.RepeatPen))
		vals = d.vals
	}
	bounded := s.TopK > 0 && s.TopK < n
	minp, topp := float32(s.MinP), float32(math.Inf(1))
	if s.TopP > 0 && s.TopP < 1 {
		topp = float32(s.TopP)
	}
	cuts := s.MinP > 0 || !math.IsInf(float64(topp), 1)
	d.dense = false
	if !bounded {
		d.pall = grow32(d.pall, n)
		copy(d.pall, vals)
		nn.Scale32JIT(d.pall, float32(1/s.Temp))
		nn.Softmax32JIT(d.pall, n)
		if !cuts {
			d.dense = true
			return
		}
	}
	order := vals
	if !bounded {
		order = d.pall
	}
	d.ord.Begin(order)
	want := sampleGrowStart
	if bounded {
		want = s.TopK
	}
	d.ids, d.p = d.ids[:0], d.p[:0]
	// The divisor as Sample has it: the first round's cut is decided as
	// Sample decides it (sumAll negative), and only an uncut round asks for
	// the row's mass.
	sumAll, haveSum := float32(-1), bounded
	for {
		for len(d.ids) < want {
			v, id, ok := d.ord.Next()
			if !ok {
				break
			}
			d.ids, d.p = append(d.ids, id), append(d.p, v)
		}
		have := len(d.ids)
		if have == 0 {
			return
		}
		if bounded {
			nn.Scale32JIT(d.p, float32(1/s.Temp))
			nn.Softmax32JIT(d.p, have)
		}
		// u 0: the walk is not wanted, only the cut and the surviving mass.
		r := d.draw.Run(d.p, minp, topp, 0, sumAll)
		if !r.Cut && !haveSum {
			sumAll, haveSum = d.draw.Mass(d.pall), true
			continue
		}
		// Sample stops early when its walk lands inside the prefix; a
		// distribution needs the whole surviving set, so it stops at the cut.
		if bounded || r.Cut {
			m := min(max(r.M, 1), have)
			d.ids, d.p = d.ids[:m], d.p[:m]
			if r.Sum > 0 {
				nn.Scale32JIT(d.p, 1/r.Sum)
			}
			return
		}
		// No cut inside the prefix: the survivors reach past it. Take twice as
		// many, as Sample does; uncut over the whole row is the dense case.
		if have == n || want >= n {
			d.dense = true
			return
		}
		want = min(2*want, n)
	}
}

// uniform is the next draw of the sampler's stream, the one Sample uses, so a
// seeded run is reproducible.
func (s *Sampler) uniform() float64 {
	s.seed()
	return s.rng.Float64()
}

// samplerMark is a Sampler's history at one point, which a speculation round
// observes drafted tokens past and then takes back.
type samplerMark struct {
	hist []int32
	hoff []int64
}

func (s *Sampler) mark(m *samplerMark) {
	m.hist = append(m.hist[:0], s.hist...)
	m.hoff = append(m.hoff[:0], s.hoff...)
}

func (s *Sampler) restore(m *samplerMark) {
	s.hist = append(s.hist[:0], m.hist...)
	s.hoff = append(s.hoff[:0], m.hoff...)
}
