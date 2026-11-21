package hb

import (
	"math"
	"sort"
)

// Percentile is the p-th percentile (0..100) of xs by linear interpolation
// between closest ranks, numpy.percentile's default. xs need not be sorted. An
// empty xs is NaN, never zero: a level with no samples must not print a latency
// of 0.
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	if p <= 0 {
		return s[0]
	}
	if p >= 100 {
		return s[len(s)-1]
	}
	r := p / 100 * float64(len(s)-1)
	lo := int(math.Floor(r))
	hi := min(lo+1, len(s)-1)
	return s[lo] + (r-float64(lo))*(s[hi]-s[lo])
}

// Median is the 50th percentile.
func Median(xs []float64) float64 { return Percentile(xs, 50) }

// RatioGate is the RULE 2 verdict on a set of per-round ratios: the median,
// the interquartile range computed as the repo's other harnesses compute it
// (sorted[3n/4] - sorted[n/4], dev/bench and scripts/vs-llamacpp.sh), and
// whether the row may be quoted.
type RatioGate struct {
	Median F   `json:"median"`
	IQR    F   `json:"iqr"`
	N      int `json:"n"`
	// Refused names why the row may not be quoted; empty when it may.
	Refused string `json:"refused,omitempty"`
}

// MinRounds is the fewest per-round ratios a row is quoted from, as in
// scripts/vs-llamacpp.sh: a median of three is not a measurement however
// tight its IQR looks.
const MinRounds = 6

// MaxSpread is the IQR/median above which a row is refused.
const MaxSpread = 0.10

// Gate computes the verdict on ratios.
func Gate(ratios []float64) RatioGate {
	s := make([]float64, 0, len(ratios))
	for _, r := range ratios {
		if !math.IsNaN(r) && !math.IsInf(r, 0) && r > 0 {
			s = append(s, r)
		}
	}
	sort.Float64s(s)
	g := RatioGate{N: len(s)}
	if len(s) == 0 {
		g.Median, g.Refused = F(math.NaN()), "no round produced a ratio"
		return g
	}
	g.Median = F(Median(s))
	g.IQR = F(s[3*len(s)/4] - s[len(s)/4])
	switch {
	case len(s) < len(ratios):
		g.Refused = "a round produced no ratio (an arm failed or generated nothing)"
	case len(s) < MinRounds:
		g.Refused = "fewer than 6 rounds"
	case g.IQR/g.Median > MaxSpread:
		g.Refused = "IQR/median above 0.10"
	}
	return g
}
