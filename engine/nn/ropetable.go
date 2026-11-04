package nn

import (
	"math"

	"github.com/jitllm/jitllm/graph"
)

// Rope is a model's rotary configuration: per-model constants that travel
// together, so the pair layout cannot be passed wrong at one call site.
type Rope struct {
	NRot int
	Base float64
	// Neox rotates (i, i+NRot/2) instead of (2i, 2i+1). llama is NORM; gemma,
	// qwen and phi3 are NEOX. See the note on Apply.
	Neox bool
	// Freqs divides theta per pair, from rope_freqs.weight (llama 3.1+) or
	// rope_factors_short.weight (phi3 LongRoPE). nil is no rescaling.
	Freqs []float32
	// Scale multiplies cos and sin -- ggml's mscale, from
	// rope.scaling.attn_factor. Zero means one.
	Scale float64
	// Runs makes the position multi-axis: run u's pairs are turned by
	// coordinate u.Axis (see ropeaxes.go and TableAt). nil is a plain index.
	// Table ignores it, which is exact for a row whose coordinates agree and
	// whose runs are one frequency sequence (M-RoPE's text rows).
	Runs []RopeRun

	// first and stride select a frequency sequence other than 0, 1, 2, ...
	// for one run's folded planes (runRope); zero values are the plain table.
	first, stride int
}

// YarnFreqs is YaRN's rotary scaling as per-pair divisors for Rope.Freqs:
// pair p's angle pos*base^(-2p/nrot) becomes that times
// (1-e)/factor + e, where e is 1 below the correction range, 0 above it and a
// linear ramp between.
//
// The range ends are corr(b) = nrot*ln(origCtx/(2*pi*b)) / (2*ln(base)) for
// b = betaFast and betaSlow. llama.cpp and transformers' default floor the
// first and ceil the second; exact skips that, as gpt-oss's config
// (truncate false) asks. The magnitude correction is not here -- it scales
// cos and sin, which Rope.Scale does.
func YarnFreqs(nrot int, base, factor float64, origCtx int, betaFast, betaSlow float64, exact bool) []float32 {
	corr := func(b float64) float64 {
		return float64(nrot) * math.Log(float64(origCtx)/(2*math.Pi*b)) / (2 * math.Log(base))
	}
	lo, hi := corr(betaFast), corr(betaSlow)
	if !exact {
		lo, hi = math.Floor(lo), math.Ceil(hi)
	}
	lo, hi = math.Max(lo, 0), math.Min(hi, float64(nrot-1))
	if hi == lo {
		hi += 0.001
	}
	out := make([]float32, nrot/2)
	for p := range out {
		ramp := math.Min(math.Max((float64(p)-lo)/(hi-lo), 0), 1)
		e := 1 - ramp
		out[p] = float32(1 / ((1-e)/factor + e))
	}
	return out
}

// TabPlanes is the folded constant block a generated rotary-table kernel reads
// -- the four head planes, the four residual planes and mscale, in
// engine/device/cpu/ropetab_const.go's layout. A multi-axis rotary's pairs fold at
// their runs' frequencies.
//
// It is exported for the device, which uploads the planes once per model like
// a weight, so every tier derives the table from the same fold. Each call
// folds afresh: it runs where a block is placed, never in a step.
func (r Rope) TabPlanes(npairs int) []float32 {
	if npairs <= 0 {
		return nil
	}
	return foldRopeTab(r, npairs)
}

// restartingRuns says runs do not continue one frequency sequence: some run
// starts at an index other than the pairs before it, or strides. Only such
// runs fold differently from the plain table.
func restartingRuns(runs []RopeRun) bool {
	off := 0
	for _, u := range runs {
		if u.Freq != off || u.step() != 1 {
			return true
		}
		off += u.Pairs
	}
	return false
}

// foldRopeTab folds one configuration's frequencies into the kernel's planes.
//
// Each digit's weight is folded progressively and exactly: the next weight is
// this one's times 128, and multiplying by a power of two and taking mod 4 of
// an exact value are exact in float64. The only rounding is the first
// quarter-turn conversion and the split into two float32 words.
//
// The frequency uses the float64 table's own recurrence (f *= step, runFreqs),
// so a comparison between the two is about the kernel, not the frequencies.
func foldRopeTab(r Rope, npairs int) []float32 {
	blk := make([]float32, graph.RopeBlock(npairs))
	freqs := runFreqs(r, npairs)
	// Restarting runs fold each pair at its own run's frequency, as the
	// multi-axis table turns it (tableAt).
	if restartingRuns(r.Runs) {
		off := 0
		for _, u := range r.Runs {
			if off >= npairs {
				break
			}
			n := min(u.Pairs, npairs-off)
			copy(freqs[off:off+n], runFreqs(r.runRope(u), n))
			off += n
		}
	}
	for p := 0; p < npairs; p++ {
		w := math.Mod(freqs[p]*(2/math.Pi), 4)
		for d := 0; d < graph.RopeDigits; d++ {
			if d > 0 {
				w = math.Mod(w*(1<<graph.RopeDigitBits), 4)
			}
			h, l := graph.RopeHeadSplit(w)
			blk[graph.RopePlaneOff(d, npairs)+p] = h
			blk[graph.RopePlaneOff(graph.RopeDigits+d, npairs)+p] = l
		}
	}
	// The pad past npairs is never read -- the ragged tail re-runs the LAST
	// whole vector rather than a padded one -- but a zero there is one less
	// claim to re-derive, and make has already put one there.
	mscale := r.Scale
	if mscale == 0 {
		mscale = 1
	}
	blk[graph.RopeScaleOff(npairs)] = float32(mscale)
	return blk
}
