package composer

import (
	"errors"
	"fmt"
	"math"
)

// SWR is libswresample's resampler (ffmpeg's swr, as `ffmpeg -ar` runs it
// with its default options) from one rate to another, for a clip of float
// samples: its polyphase filter bank, built in float64 as resample.c's
// build_filter builds it and kept in float32 as swr's float path keeps it,
// and the step each output takes through the source. A clip whose rate is
// the output's is not resampled (swr makes no resampler): one tap of one.
//
// A sound op runs it (graph.OpSound): Bank is its table, and each output's
// first sample, phase and fraction are its step's words (SWR.Steps).
type SWR struct {
	In, Out int
	// Taps is each phase's filter length (filter_length), Phases the
	// phases of a sample (phase_count), Bank the Phases+1 filters of Taps
	// weights, the last phase 0 one tap later (resample_init's copies).
	Taps, Phases int
	Bank         []float32
	// SrcIncr, Div and Mod are the step from one output to the next: Div
	// phases and Mod of SrcIncr fractions of one.
	SrcIncr, Div, Mod int
}

// swr's defaults (libswresample/options.c): a filter of 32 taps at unit
// scale, 2^10 phases at most, a Kaiser window of beta 9, a cutoff at 0.97
// of the output's Nyquist, exact rational phases, linear interpolation
// between phases where the ratio has more phases than that.
const (
	swrFilterSize = 32
	swrPhaseShift = 10
	swrBeta       = 9
	swrCutoff     = 0.97
)

// NewSWR is the resampler from in to out samples a second.
func NewSWR(in, out int) (SWR, error) {
	if in <= 0 || out <= 0 {
		return SWR{}, fmt.Errorf("composer: a resample from %d to %d samples a second", in, out)
	}
	if in == out {
		return SWR{In: in, Out: out, Taps: 1, Phases: 1, Bank: []float32{1, 1}, SrcIncr: 1, Div: 1}, nil
	}
	factor := min(float64(out)*swrCutoff/float64(in), 1)
	phases := 1 << swrPhaseShift
	taps := max(int(math.Ceil(swrFilterSize/factor)), 1)
	if taps > 1 {
		taps += taps & 1
	}
	g := gcd(out, in)
	if out/g <= phases {
		phases = out / g
	}
	// av_reduce of out against in*phases, then both doubled until either
	// reaches 2^20.
	num, den := int64(out), int64(in)*int64(phases)
	r := gcd64(num, den)
	src, dst := num/r, den/r
	if src > math.MaxInt32/2 || dst > math.MaxInt32/2 {
		return SWR{}, errors.New("composer: a resample ratio past swr's 31 bits")
	}
	for dst < 1<<20 && src < 1<<20 {
		dst *= 2
		src *= 2
	}
	s := SWR{In: in, Out: out, Taps: taps, Phases: phases, SrcIncr: int(src), Div: int(dst / src), Mod: int(dst % src)}
	s.Bank = swrBank(factor, taps, phases, swrBeta)
	return s, nil
}

func gcd64(a, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// Linear says the outputs interpolate between two phases' filters by their
// fraction: a ratio with more phases than swr keeps.
func (s SWR) Linear() bool { return s.Mod != 0 }

// swrBank is build_filter's Kaiser-windowed sinc, of beta beta, for float samples, scale
// 1: phases+1 rows of taps, a phase past the first half its mirror, and the
// last row phase 0 one tap later, its first weight the row's padding past
// the filter (zero where filter_alloc, the taps rounded up to 8, is wider).
func swrBank(factor float64, taps, phases int, beta float64) []float32 {
	bank := make([]float32, (phases+1)*taps)
	half := phases/2 + 1
	if phases%2 == 1 {
		half = phases
	}
	center := (taps - 1) / 2
	tab := make([]float64, taps)
	norm := 0.0
	for ph := range half {
		for i := range tab {
			x := math.Pi * (float64(i-center) - float64(ph)/float64(phases)) * factor
			y := 1.0
			if x != 0 {
				y = math.Sin(x) / x
			}
			w := 2 * x / (factor * float64(taps) * math.Pi)
			y *= besselI0(beta * math.Sqrt(max(1-w*w, 0)))
			tab[i] = y
			if ph == 0 {
				norm += y
			}
		}
		for i, y := range tab {
			bank[ph*taps+i] = float32(y / norm)
		}
		if phases%2 == 0 {
			for i := range taps {
				bank[(phases-ph)*taps+taps-1-i] = bank[ph*taps+i]
			}
		}
	}
	last := bank[phases*taps:]
	copy(last[1:], bank[:taps-1])
	last[0] = 0
	if taps%8 == 0 {
		last[0] = bank[taps-1]
	}
	return bank
}

// besselI0 is av_bessel_i0 (libavutil/mathematics.c): Blair and Edwards'
// minimax rational approximations of the modified Bessel function of order
// zero, as ffmpeg evaluates them.
func besselI0(x float64) float64 {
	p1 := [...]float64{-2.2335582639474375249e+15, -5.5050369673018427753e+14, -3.2940087627407749166e+13,
		-8.4925101247114157499e+11, -1.1912746104985237192e+10, -1.0313066708737980747e+08,
		-5.9545626019847898221e+05, -2.4125195876041896775e+03, -7.0935347449210549190e+00,
		-1.5453977791786851041e-02, -2.5172644670688975051e-05, -3.0517226450451067446e-08,
		-2.6843448573468483278e-11, -1.5982226675653184646e-14, -5.2487866627945699800e-18}
	q1 := [...]float64{-2.2335582639474375245e+15, 7.8858692566751002988e+12, -1.2207067397808979846e+10,
		1.0377081058062166144e+07, -4.8527560179962773045e+03, 1.0}
	p2 := [...]float64{-2.2210262233306573296e-04, 1.3067392038106924055e-02, -4.4700805721174453923e-01,
		5.5674518371240761397e+00, -2.3517945679239481621e+01, 3.1611322818701131207e+01,
		-9.6090021968656180000e+00}
	q2 := [...]float64{-5.5194330231005480228e-04, 3.2547697594819615062e-02, -1.1151759188741312645e+00,
		1.3982595353892851542e+01, -6.0228002066743340583e+01, 8.5539563258012929600e+01,
		-3.1446690275135491500e+01, 1.0}
	if x == 0 {
		return 1
	}
	x = math.Abs(x)
	if x <= 15 {
		y := x * x
		return poly(p1[:], y) / poly(q1[:], y)
	}
	y := 1/x - 1.0/15
	return math.Exp(x) / math.Sqrt(x) * poly(p2[:], y) / poly(q2[:], y)
}

// poly is eval_poly: c[0] + c[1]x + ... by Horner's rule from the top.
func poly(c []float64, x float64) float64 {
	v := c[len(c)-1]
	for i := len(c) - 2; i >= 0; i-- {
		v = v*x + c[i]
	}
	return v
}

// Clip is how a clip of n source samples is resampled: Len, the source the
// outputs read (the clip and the samples swr's flush reflects onto its
// end), and Outputs, how many it makes. Source sample j of a step reads the
// clip at j mirrored about its first sample below 0 (invert_initial_buffer)
// and about its last past n (resample_flush): SoundAt.
func (s SWR) Clip(n int) (length, outputs int) {
	if s.Taps == 1 && s.Phases == 1 && s.In == s.Out {
		return n, n
	}
	// swr's start waits for Taps+1 samples; a clip shorter is its flush's
	// reflection of half of it, or nothing.
	if n < s.Taps+1 {
		length = n + (n+1)/2
		if length < s.Taps+1 {
			return length, 0
		}
		return length, s.count(length)
	}
	// The flush reflects half of what the clip leaves unconsumed, at most
	// half a filter.
	var w swrWalk
	w.start(s)
	for w.sample+s.Taps <= n {
		w.next(s)
	}
	length = n + (min(n-w.sample, s.Taps)+1)/2
	return length, s.count(length)
}

// count is the outputs whose window ends within a source of length samples.
func (s SWR) count(length int) int {
	var w swrWalk
	w.start(s)
	n := 0
	for w.sample+s.Taps <= length {
		n++
		w.next(s)
	}
	return n
}

// swrWalk is resample_common's position: the window's first sample, the
// phase within it and the fraction of one.
type swrWalk struct{ sample, index, frac int }

func (w *swrWalk) start(s SWR) { *w = swrWalk{sample: -((s.Taps - 1) / 2)} }

func (w *swrWalk) next(s SWR) {
	w.frac += s.Mod
	w.index += s.Div
	if w.frac >= s.SrcIncr {
		w.frac -= s.SrcIncr
		w.index++
	}
	for w.index >= s.Phases {
		w.sample++
		w.index -= s.Phases
	}
}

// SWRStepWords is the words a resample's output takes in its steps: the
// window's first source sample (int32), its phase, its fraction.
const SWRStepWords = 3

// Steps writes outputs from to from+len(dst)/SWRStepWords-1 of a clip's
// steps into dst (SWRStepWords a step).
func (s SWR) Steps(from int, dst []uint32) {
	var w swrWalk
	w.start(s)
	for range from {
		w.next(s)
	}
	for i := 0; i+SWRStepWords <= len(dst); i += SWRStepWords {
		dst[i], dst[i+1], dst[i+2] = uint32(int32(w.sample)), uint32(w.index), uint32(w.frac)
		w.next(s)
	}
}

// SoundAt is where source sample j of a clip of n samples is read: swr's
// mirror below the first sample and its reflection past the last.
func SoundAt(j, n int) int {
	if j < 0 {
		j = -j
	}
	if j >= n {
		j = 2*n - 1 - j
	}
	return j
}
