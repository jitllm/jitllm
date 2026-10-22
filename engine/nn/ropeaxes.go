package nn

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
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

// RopeRuns is the pair count the runs fill, which is the table's width in pairs.
func RopeRuns(runs []RopeRun) int {
	n := 0
	for _, u := range runs {
		n += u.Pairs
	}
	return n
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

// TableAt fills a multi-axis position's table: run u's pairs at pos[u.Axis].
// Every run is the generated table kernel over its own folded planes, so the
// arithmetic is Table's, pair for pair.
func (r Rope) TableAt(cs []float32, pos []int) {
	r.tableAt(cs, pos, false)
}

// RopeTableAt is TableAt through this JIT, as RopeTable is Table.
func (f *JIT) RopeTableAt(r Rope, cs []float32, pos []int) {
	r.tableAt(cs, pos, f != nil && f.cfg.RopeGo)
}

func (r Rope) tableAt(cs []float32, pos []int, goArm bool) {
	if len(r.Runs) == 0 {
		panic("jit: a multi-axis rotary table for a rotary with no runs")
	}
	if 2*RopeRuns(r.Runs) > len(cs) {
		panic(fmt.Sprintf("jit: a %d-pair multi-axis table into %d floats", RopeRuns(r.Runs), len(cs)))
	}
	off := 0
	for _, u := range r.Runs {
		if u.Axis < 0 || u.Axis >= len(pos) {
			panic(fmt.Sprintf("jit: a rotary run on axis %d of a %d-coordinate position", u.Axis, len(pos)))
		}
		p := pos[u.Axis]
		dst := cs[2*off : 2*(off+u.Pairs)]
		off += u.Pairs
		if u.Pairs <= 0 {
			continue
		}
		if goArm {
			ropeGoRun(r, dst, p, u)
			continue
		}
		if p < 0 || p >= cpu.RopeTabMaxPos {
			panic(fmt.Sprintf("jit: rotary table at position %d, outside the %d the "+
				"kernel's digit decomposition covers", p, cpu.RopeTabMaxPos))
		}
		code := ropeTabCodeFor(u.Pairs)
		if code == nil {
			panic(fmt.Sprintf("jit: no rotary-table kernel for %d pair(s) on tier %v -- "+
				"every tier generates one, so this is a wiring bug and not a fallback",
				u.Pairs, cpu.HostTier()))
		}
		tab := ropeTabFor(r.runRope(u), u.Pairs)
		ropeTabCalls.Add(1)
		code.Call(&cpu.Args{
			Out:    &dst[0],
			AScale: &tab.block[0],
			Scr:    (*byte)(unsafe.Pointer(&ropeTabConsts[0])),
			K:      int64(p),
		})
	}
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

// Planes is the folded block the device's rotary table reads for the whole
// of r's table: TabPlanes at NRot/2 pairs, or for a multi-axis rotary each
// run's planes at its pairs' places, the scale after them, then each pair's
// axis.
func (r Rope) Planes() []float32 {
	if len(r.Runs) == 0 {
		return r.TabPlanes(r.NRot / 2)
	}
	const planes, lanes = 8, 8
	P := RopeRuns(r.Runs)
	stride := (P + lanes - 1) &^ (lanes - 1)
	out := make([]float32, planes*stride+1+P)
	off := 0
	for _, u := range r.Runs {
		if u.Pairs <= 0 {
			continue
		}
		b := ropeTabFor(r.runRope(u), u.Pairs).block
		rs := (u.Pairs + lanes - 1) &^ (lanes - 1)
		for i := range planes {
			copy(out[i*stride+off:i*stride+off+u.Pairs], b[i*rs:i*rs+u.Pairs])
		}
		out[planes*stride] = b[planes*rs]
		for j := range u.Pairs {
			out[planes*stride+1+off+j] = float32(u.Axis)
		}
		off += u.Pairs
	}
	return out
}
