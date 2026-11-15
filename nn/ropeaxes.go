package nn

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// A multi-axis rotary: a position with more than one coordinate, each turning
// its own run of a head's rotary pairs. It is one capability with three shapes
// on the models this engine reads:
//
//	Qwen2-VL's text M-RoPE     (t, h, w) over [16, 24, 24] pairs of 64, the
//	                           frequencies one sequence across all of them
//	Qwen2-VL's ViT 2-D rotary  (row, col) over two halves of a head's pairs,
//	                           each half restarting at the first frequency
//	Pixtral's ViT 2-D rotary   (row, col) over two halves, taking the even
//	                           and the odd frequencies of one sequence
//
// All three are a table of {cos, sin} per pair, so the ROTATION is the one
// every tier already runs (RoPE32JIT, the device's RoPERows): only the table
// says which coordinate and which frequency a pair takes (jlm.RopePair). A row whose
// coordinates are all equal and whose runs are one frequency sequence is the
// plain table at that position, to the bit.

// RopeRun is a run of consecutive pairs of a multi-axis table, all turned by
// one coordinate of the position.
type RopeRun struct {
	// Pairs is how many consecutive pairs of the table the run fills.
	Pairs int
	// Axis is the coordinate of the position that turns them.
	Axis int
	// Freq is the frequency index of the run's first pair; pair j of the run
	// turns at frequency index Freq + j*Step, where index i is
	// Base^(-2i/NRot) divided by Freqs[i] when Freqs reaches that far.
	Freq int
	// Step is the frequency stride between consecutive pairs; zero is one.
	Step int
}

func (u RopeRun) step() int {
	if u.Step <= 0 {
		return 1
	}
	return u.Step
}

// PairRuns is a rotary table's pairs (each its axis and frequency) as runs:
// consecutive pairs of one axis whose frequencies follow one another are one
// run. A table whose frequencies are its pairs' indices is one frequency
// sequence, so a row whose coordinates agree is the plain table.
func PairRuns(pairs []jlm.RopePair) []RopeRun {
	var runs []RopeRun
	for _, p := range pairs {
		if n := len(runs); n > 0 && runs[n-1].Axis == p.Axis && runs[n-1].Freq+runs[n-1].Pairs == p.Freq {
			runs[n-1].Pairs++
			continue
		}
		runs = append(runs, RopeRun{Pairs: 1, Axis: p.Axis, Freq: p.Freq})
	}
	return runs
}

// TablePairs is how many pairs a table over r's runs holds: their pairs, or
// NRot/2 for a rotary of one position.
func (r *Rope) TablePairs() int {
	if len(r.Runs) == 0 {
		return r.NRot / 2
	}
	n := 0
	for _, u := range r.Runs {
		n += u.Pairs
	}
	return n
}

// TableAxes is each pair's axis of a table over r's runs, none for a rotary
// of one position.
func (r *Rope) TableAxes() []uint8 {
	var out []uint8
	for _, u := range r.Runs {
		for range u.Pairs {
			out = append(out, uint8(u.Axis))
		}
	}
	return out
}

// runRope is the single-axis rotary whose folded planes are run u's: the same
// configuration with its frequency sequence started at u.Freq and strided by
// u.Step. It is a key into the planes cache, so the planes are folded once per
// run for the model's life.
func (r Rope) runRope(u RopeRun) Rope {
	q := r
	q.Runs = nil
	q.first, q.stride = u.Freq, u.step()
	return q
}

// runFreqs is the frequency of every pair of a run, by the plain table's own
// recurrence (f *= step from index zero), so a run starting at index i folds
// exactly the float64 value the plain table folds at pair i: a row whose
// coordinates agree is the plain table to the bit.
func runFreqs(r Rope, npairs int) []float64 {
	first, stride := r.first, max(r.stride, 1)
	last := first + stride*(npairs-1)
	step := math.Pow(r.Base, -2/float64(r.NRot))
	out := make([]float64, npairs)
	f := 1.0
	for i := 0; i <= last; i++ {
		if i >= first && (i-first)%stride == 0 {
			g := f
			if i < len(r.Freqs) {
				g /= float64(r.Freqs[i])
			}
			out[(i-first)/stride] = g
		}
		f *= step
	}
	return out
}
