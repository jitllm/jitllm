package nn

import (
	"math"
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
// says which coordinate and which frequency a pair takes. A row whose
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

// MRopeRuns is Qwen2-VL's text M-RoPE: sections[i] consecutive pairs turned by
// coordinate i, the frequencies one sequence across all of them (llama.cpp's
// ggml_rope_multi, transformers' mrope_section). Empty sections are skipped; a
// model whose sections are all zero has no runs.
func MRopeRuns(sections []int) []RopeRun {
	var runs []RopeRun
	off := 0
	for ax, n := range sections {
		if n <= 0 {
			continue
		}
		runs = append(runs, RopeRun{Pairs: n, Axis: ax, Freq: off})
		off += n
	}
	return runs
}

// IMRopeRuns is Qwen3-VL's INTERLEAVED M-RoPE over pairs pairs: pair i turns
// by coordinate 1 (h) when i%3 == 1 and i < 3*sections[1], by 2 (w) when
// i%3 == 2 and i < 3*sections[2], and by 0 (t) otherwise, at frequency i --
// transformers' recomposition_frequencies. Consecutive pairs of one axis are
// one run; the frequencies stay one sequence across all of them, so a row
// whose coordinates agree is the plain table.
func IMRopeRuns(sections []int, pairs int) []RopeRun {
	sec := func(i int) int {
		if i < len(sections) {
			return sections[i]
		}
		return 0
	}
	var runs []RopeRun
	for i := 0; i < pairs; i++ {
		ax := 0
		switch {
		case i%3 == 1 && i < 3*sec(1):
			ax = 1
		case i%3 == 2 && i < 3*sec(2):
			ax = 2
		}
		if n := len(runs); n > 0 && runs[n-1].Axis == ax {
			runs[n-1].Pairs++
			continue
		}
		runs = append(runs, RopeRun{Pairs: 1, Axis: ax, Freq: i})
	}
	return runs
}

// XDRopeRuns is HunyuanVL's XD-RoPE over pairs NEOX pairs: transformers'
// recomposition_frequencies takes cat(freq, freq) -- 2*pairs elements -- in
// chunks of 2*sections[i] elements, chunk i turned by coordinate i mod
// len(sections). Pair p's first element is element p and its second element
// p+pairs, so the two halves fall in different chunks and take different
// axes. a is the runs of the first halves, b of the second, the frequency
// index p in both, so a row whose coordinates agree is the plain table twice.
func XDRopeRuns(sections []int, pairs int) (a, b []RopeRun) {
	// A zero section is an empty chunk that still takes its index, as
	// torch's split hands enumerate one.
	axis := func(e int) int {
		for i, at := 0, 0; i < len(sections); i++ {
			if at += 2 * sections[i]; e < at {
				return i % len(sections)
			}
		}
		return 0
	}
	runsOf := func(off int) []RopeRun {
		var runs []RopeRun
		for p := 0; p < pairs; p++ {
			ax := axis(p + off)
			if n := len(runs); n > 0 && runs[n-1].Axis == ax {
				runs[n-1].Pairs++
				continue
			}
			runs = append(runs, RopeRun{Pairs: 1, Axis: ax, Freq: p})
		}
		return runs
	}
	return runsOf(0), runsOf(pairs)
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
