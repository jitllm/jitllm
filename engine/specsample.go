package engine

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
}

func (s *Sampler) mark(m *samplerMark) {
	m.hist = append(m.hist[:0], s.hist...)
}

// restore takes the window back to m, its counts with it.
func (s *Sampler) restore(m *samplerMark) {
	for _, t := range s.hist {
		s.win[t]--
	}
	s.hist = append(s.hist[:0], m.hist...)
	for _, t := range s.hist {
		s.win[t]++
	}
}
