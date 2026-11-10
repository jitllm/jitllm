package composer

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// A sound op's header (graph.OpSound): the step's picture table starts
// with these words, then each channel's weight in the mono mix (SoundMixAt,
// at most SoundChannelsMax), the clip's PCM (SoundPCMAt), each output's
// step and the resampler's bank, where the header says (SoundTable).
const (
	SoundChannels = iota
	SoundFrames
	SoundTaps
	SoundOutputs
	SoundLinear
	// SoundInvIncr is the f32 bits of one over the resampler's fraction of
	// a phase (SWR.SrcIncr).
	SoundInvIncr
	// SoundLast is twice the clip's frames less one: a frame past the last
	// is read at SoundLast less it (SoundAt).
	SoundLast
	SoundSteps
	SoundHeader
)

// SoundChannelsMax is the most channels a clip has: a WAV's mask names 18
// speakers, swr's matrix 64 channels.
const SoundChannelsMax = 64

// SoundMixAt and SoundPCMAt are where the mix and the PCM start in a sound
// op's table.
const (
	SoundMixAt = SoundHeader
	SoundPCMAt = SoundMixAt + SoundChannelsMax
)

// SoundBytes is the bytes a sample of format f takes in a clip's PCM.
func SoundBytes(f uint8) int {
	switch f {
	case graph.SoundU8:
		return 1
	case graph.SoundS16:
		return 2
	case graph.SoundS24:
		return 3
	case graph.SoundS32, graph.SoundF32:
		return 4
	case graph.SoundF64:
		return 8
	}
	return 0
}

// SoundTable lays a clip out for a sound op in dst, the step's picture
// table, and says how many outputs it makes, at most most: the header, each
// channel's weight in the mono mix (MonoMix), the clip's PCM (frames of
// channels samples, interleaved, in the op's format) as words, each output's
// step -- its window's first source sample plus the filter's taps, where its
// phase's filter starts in the table, its fraction (SWR.Steps) -- and the
// resampler's bank.
func SoundTable(dst []uint32, pcm []byte, channels, frames int, mix []float32, s SWR, most int) ([]uint32, int) {
	_, outputs := s.Clip(frames)
	outputs = min(outputs, most)
	// A word past the PCM: a 24-bit sample is read as two.
	stepsAt := SoundPCMAt + (len(pcm)+3)/4 + 1
	bankAt := stepsAt + outputs*SWRStepWords
	dst = append(dst[:0], make([]uint32, bankAt+len(s.Bank))...)
	lin := uint32(0)
	if s.Linear() {
		lin = 1
	}
	copy(dst, []uint32{uint32(channels), uint32(frames), uint32(s.Taps), uint32(outputs), lin,
		math.Float32bits(float32(1 / float64(s.SrcIncr))), uint32(2*frames - 1), uint32(stepsAt)})
	for i, w := range mix {
		dst[SoundMixAt+i] = math.Float32bits(w)
	}
	if len(pcm) > 0 {
		copy(unsafe.Slice((*byte)(unsafe.Pointer(&dst[SoundPCMAt])), 4*(stepsAt-SoundPCMAt)), pcm)
	}
	steps := dst[stepsAt:bankAt]
	s.Steps(0, steps)
	for i := 0; i < len(steps); i += SWRStepWords {
		steps[i] = uint32(int32(steps[i]) + int32(s.Taps))
		steps[i+1] = uint32(bankAt) + steps[i+1]*uint32(s.Taps)
	}
	for i, w := range s.Bank {
		dst[bankAt+i] = math.Float32bits(w)
	}
	return dst, outputs
}

// soundSample is sample q of the clip's PCM at word pcm as a float, as
// libswresample's audioconvert makes one: an unsigned byte less 128 over
// 128, a 16-bit sample over 2^15, a 24-bit one read as the top of a 32-bit
// one (pcm_s24le) and that over 2^31, a 32-bit one over 2^31, a float as it
// is, a double rounded to the nearest float (one below float's least normal
// read as zero).
func soundSample(x *exprs, format uint8, pcm, q int) int {
	U, I, F := ir.U32, ir.I32, ir.F32
	u := func(v uint32) int { return x.u(v) }
	word := func(w int) int { return x.load(0, x.op(OpAdd, U, pcm, w)) }
	// bytes is the 32 bits from byte offset off of the PCM, unaligned.
	bytes := func(off int) int {
		w := x.op(OpShr, U, off, u(2))
		sh := x.op(OpShl, U, x.op(OpAnd, U, off, u(3)), u(3))
		lo := x.op(OpShr, U, word(w), sh)
		hi := x.op(OpShl, U, word(x.op(OpAdd, U, w, u(1))), x.op(OpSub, U, u(32), sh))
		return x.op(OpOr, U, lo, x.sel(U, x.op(OpLess, ir.Pred, sh, u(1)), u(0), hi))
	}
	scaled := func(i int, scale float32) int { return x.op(OpMul, F, x.un(OpCvtF32, F, i), x.f(scale)) }
	switch format {
	case graph.SoundU8:
		b := x.op(OpAnd, U, bytes(q), u(255))
		return scaled(x.op(OpSub, I, x.un(OpBitcast, I, b), x.i(128)), 1.0/(1<<7))
	case graph.SoundS16:
		v := x.op(OpAnd, U, bytes(x.op(OpShl, U, q, u(1))), u(0xFFFF))
		return scaled(x.op(OpSub, I, x.un(OpBitcast, I, x.op(OpXor, U, v, u(0x8000))), x.i(0x8000)), 1.0/(1<<15))
	case graph.SoundS24:
		v := x.op(OpShl, U, bytes(x.op(OpMul, U, q, u(3))), u(8))
		return scaled(x.un(OpBitcast, I, v), 1.0/(1<<31))
	case graph.SoundS32:
		return scaled(x.un(OpBitcast, I, word(q)), 1.0/(1<<31))
	case graph.SoundF32:
		return x.un(OpBitcast, F, word(q))
	}
	// A double's bits to the nearest float: the exponent rebased, the
	// mantissa's top 23 bits rounded by the 29 below them, ties to even, a
	// carry into the exponent the next binade's; past float's range
	// infinity, a NaN a NaN, below its least normal zero, the sign kept.
	lo := word(x.op(OpShl, U, q, u(1)))
	hi := word(x.op(OpAdd, U, x.op(OpShl, U, q, u(1)), u(1)))
	sign := x.op(OpAnd, U, hi, u(0x80000000))
	e := x.op(OpAnd, U, x.op(OpShr, U, hi, u(20)), u(0x7FF))
	m := x.op(OpOr, U, x.op(OpShl, U, x.op(OpAnd, U, hi, u(0xFFFFF)), u(3)), x.op(OpShr, U, lo, u(29)))
	rest := x.op(OpAnd, U, lo, u(0x1FFFFFFF))
	half := u(0x10000000)
	up := x.sel(U, x.op(OpLess, ir.Pred, half, rest), u(1),
		x.sel(U, x.op(OpLess, ir.Pred, rest, half), u(0), x.op(OpAnd, U, m, u(1))))
	normal := x.op(OpAdd, U, x.op(OpOr, U, x.op(OpShl, U, x.op(OpSub, U, e, u(896)), u(23)), m), up)
	big := x.sel(U, x.op(OpLess, ir.Pred, e, u(0x7FF)), u(0x7F800000),
		x.sel(U, x.op(OpLess, ir.Pred, u(0), x.op(OpOr, U, m, rest)), u(0x7FC00000), u(0x7F800000)))
	mag := x.sel(U, x.op(OpLess, ir.Pred, e, u(897)), u(0),
		x.sel(U, x.op(OpLess, ir.Pred, e, u(1151)), normal, big))
	return x.un(OpBitcast, F, x.op(OpOr, U, sign, mag))
}

// decomposeSound is the step's clip as mono samples at the resampler's rate
// (graph.OpSound): each output the sum, over its window's taps and each
// tap's channels, of the sample (soundSample, its frame mirrored before the
// clip's first and after its last, SoundAt) times its channel's weight in
// the mix times its phase's filter at that tap; where the ratio
// interpolates, the next phase's sum too, the two blended by the output's
// fraction; zero past the clip's outputs. The sums run over the taps and
// channels the header states, at most the op's Taps.
func decomposeSound(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	sh := c.rows(o.Out[0])
	R, N := sh[0], sh[1]
	if R != 1 || a.Taps < 1 || SoundBytes(a.Sound) == 0 {
		return nil, fmt.Errorf("a sound of %d rows of %d, %d taps, sample format %d", R, N, a.Taps, a.Sound)
	}
	U, F := ir.U32, ir.F32
	hdr := func(x *exprs, i int) int { return x.load(0, x.u(uint32(i))) }
	step := func(x *exprs, n, w int) int {
		return x.load(0, x.op(OpAdd, U, hdr(x, SoundSteps), x.op(OpAdd, U, x.op(OpMul, U, n, x.u(SWRStepWords)), x.u(uint32(w)))))
	}
	// term is one tap's channel's product of a window sum, at the phase
	// after the output's where next.
	term := func(next bool) []Expr {
		var x exprs
		n, e := x.coord(0), x.coord(1)
		// An output past the clip's reads the last one's steps; the map
		// below zeroes it.
		n = x.op(OpMin, U, n, x.op(OpSub, U, x.op(OpMax, U, hdr(&x, SoundOutputs), x.u(1)), x.u(1)))
		C, T := hdr(&x, SoundChannels), hdr(&x, SoundTaps)
		tap := x.op(OpDiv, U, e, C)
		ch := x.op(OpSub, U, e, x.op(OpMul, U, tap, C))
		row := step(&x, n, 1)
		if next {
			row = x.op(OpAdd, U, row, T)
		}
		// The source frame plus T: below T is before the clip, mirrored
		// about its first frame; past the last it is reflected about it.
		j := x.op(OpAdd, U, step(&x, n, 0), tap)
		m := x.sel(U, x.op(OpLess, ir.Pred, j, T), x.op(OpSub, U, T, j), x.op(OpSub, U, j, T))
		last := hdr(&x, SoundLast)
		m = x.sel(U, x.op(OpLess, ir.Pred, m, hdr(&x, SoundFrames)), m, x.op(OpSub, U, last, x.op(OpMin, U, m, last)))
		v := soundSample(&x, a.Sound, x.u(SoundPCMAt), x.op(OpAdd, U, x.op(OpMul, U, m, C), ch))
		w := x.un(OpBitcast, F, x.load(0, x.op(OpAdd, U, row, tap)))
		mw := x.un(OpBitcast, F, x.load(0, x.op(OpAdd, U, x.u(SoundMixAt), ch)))
		x.op(OpMul, F, x.op(OpMul, F, v, mw), w)
		return x.e
	}
	// span is the sums' trips: the window's taps times the channels, the
	// same for every output -- a count that differs by row holds an
	// induction of the rows in the vector registers, and the jam's tail
	// does not carry one (Jam.stripTail) -- none for a clip of no outputs,
	// and for the second phase none where the ratio does not interpolate.
	span := func(next bool) *Range {
		var lo, hi exprs
		lo.u(0)
		live := hi.sel(U, hi.op(OpLess, ir.Pred, hi.u(0), hdr(&hi, SoundOutputs)), hi.u(1), hi.u(0))
		if next {
			live = hi.sel(U, hi.op(OpLess, ir.Pred, hi.u(0), hdr(&hi, SoundLinear)), live, hi.u(0))
		}
		hi.sel(U, hi.op(OpLess, ir.Pred, hi.u(0), live), hi.op(OpMul, U, hdr(&hi, SoundTaps), hdr(&hi, SoundChannels)), hi.u(0))
		return &Range{Axis: 1, First: lo.e, End: hi.e}
	}
	name := c.Graph.Values[o.Out[0]].Name
	sums := [2]Operand{
		{Kind: OperandScratch, Name: name + ".phase", Elem: F, Temp: true},
		{Kind: OperandScratch, Name: name + ".next", Elem: F, Temp: true},
	}
	ext := map[int]int{0: PictureWords}
	var prims []Prim
	for i, next := range []bool{false, true} {
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{N, a.Taps}, In: []Operand{TablePicture.op()}, Out: sums[i],
			Combine: OpAdd, Expr: term(next), Facts: Facts{Range: span(next), Extents: ext}})
	}
	var x exprs
	n := x.coord(1)
	s0, s1 := x.load(1, n), x.load(2, n)
	frac := x.un(OpCvtF32, F, x.un(OpBitcast, ir.I32, step(&x, x.op(OpMin, U, n, x.op(OpSub, U, x.op(OpMax, U, hdr(&x, SoundOutputs), x.u(1)), x.u(1))), 2)))
	blend := x.op(OpAdd, F, s0, x.op(OpMul, F, x.op(OpMul, F, x.op(OpSub, F, s1, s0), frac), x.un(OpBitcast, F, hdr(&x, SoundInvIncr))))
	v := x.sel(F, x.op(OpLess, ir.Pred, x.u(0), hdr(&x, SoundLinear)), blend, s0)
	x.sel(F, x.op(OpLess, ir.Pred, n, hdr(&x, SoundOutputs)), v, x.f(0))
	prims = append(prims, Prim{Kind: PrimMap, Shape: []int{1, N}, In: []Operand{TablePicture.op(), sums[0], sums[1]}, Out: c.value(o.Out[0]),
		Expr: x.e, Facts: Facts{Extents: map[int]int{0: PictureWords, 1: N, 2: N}}})
	return prims, nil
}
