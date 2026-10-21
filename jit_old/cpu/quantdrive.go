package cpu

import (
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// The selection counters, so a dose over these kernels can assert which arm
// ran what: "it made no difference" and "it never ran" read the same without
// one.
var qaWide, qaNarrow, qaGo atomic.Int64

// QuantActStats reports how many ranges each path has quantized since the last
// reset: whole windows through the wide kernel, groups of eight blocks through
// the narrow one, and ranges that fell back to the Go loop.
func QuantActStats() (wide, narrow, goLoop int64) {
	return qaWide.Load(), qaNarrow.Load(), qaGo.Load()
}

// ResetQuantActStats zeroes them, for a measurement that brackets one arm.
func ResetQuantActStats() { qaWide.Store(0); qaNarrow.Store(0); qaGo.Store(0) }

// quantType is quant.Type under a local name, so the two quantGoLoop files
// (release and jitllmtest) share one signature without either importing the
// package for a single parameter.
type quantType = quant.Type

// QuantActConsts is the constant block both quantizer kernels read:
// PackActConsts plus the two float multipliers the pair layout folds in,
// -biasC/8 for the block sum and -biasC/4 for each sixteen.
//
// The GEMM's block is unchanged and the two slots are appended: EmitPackAct
// addresses slots 0..5 by byte offset, so one layout serves every kernel.
func QuantActConsts(t quant.Type) []float32 {
	c := PackActConsts(t)
	b := BiasC(t)
	return append(c, float32(-b/8), float32(-b/4))
}

// QuantActKernels is the activation quantizer for one format: the two shapes
// and the constants they share.
//
// Two shapes, because the window decides what can be amortised. Wide is one
// scale per 256 elements, where the fold and two dependent divisions are
// cheap; Narrow is one scale per 32, where eight blocks share one pair of
// divisions instead. Both emit the same per-block payload (emitQuantBlock),
// so neither rounds differently from cpu.QuantizeQ8Window.
type QuantActKernels struct {
	Wide   *Code // one or more whole amax windows per call
	Narrow *Code // eight one-block windows per iteration
	Konst  []float32
}

// Applies reports whether Run will use a kernel at this window or fall back to
// the Go loop, so a caller and a gate ask the same question.
func (q QuantActKernels) Applies(window int) bool {
	if window > Q8Block {
		return q.Wide != nil
	}
	return q.Narrow != nil
}

// Run is QuantizeQ8Window plus QuantizeHalfSums for blocks [blo, bhi), driven
// by whichever kernel serves this window. With neither it is those two
// functions.
//
// scr is per-worker scratch of at least QuantActNarrowScratch float32; the
// narrow kernel spills its eight scales through it, so two workers sharing one
// slice would each quantize with the other's scales.
//
// The wide kernel scans a window's amax once and writes it into all its
// pairs, where the Go loop rescans the window for every block. A worker that
// owns part of a window still scans all of it: the scale belongs to the
// window, and blo/bhi only narrow which blocks are quantized, so the pool may
// split anywhere.
func (q QuantActKernels) Run(t quant.Type, dst []int8, pairs, half, scr []float32,
	x []float32, k, blo, bhi, window int) {
	if blo >= bhi {
		return
	}
	// The Go arm exists only under the jitllmtest tag, as the A/B arm that
	// prices these kernels (nn.WithQuantActGo). Every tier generates both
	// shapes, so in a release build quantgo.go panics instead.
	if !q.Applies(window) {
		quantGoLoop(t, dst, pairs, half, x, blo, bhi, window)
		return
	}
	per := window / Q8Block
	if per < 1 {
		per = 1
	}
	nb := k / Q8Block

	args := func(b, blocks, cols, scanFrom, scanLen int) Args {
		a := Args{
			A:    (*int8)(unsafe.Pointer(&x[scanFrom])),
			K:    int64(scanLen),
			Q32:  &x[b*Q8Block],
			Rows: int64(blocks),
			Cols: int64(cols),
			W:    (*byte)(unsafe.Pointer(&dst[b*Q8Block])),
			Out:  &pairs[2*b],
			Scr:  (*byte)(unsafe.Pointer(&q.Konst[0])),
		}
		if len(half) > 0 {
			// AHalfSum is typed *int32 for the GEMM, which writes integer sums
			// there; these kernels write float32. The field is a pointer either
			// way and the emitter decides what it stores.
			a.AHalfSum = (*int32)(unsafe.Pointer(&half[2*b]))
		}
		return a
	}

	b := blo
	// The narrow kernel takes whole groups of eight and nothing else; the
	// remainder (at most seven blocks) goes to the wide kernel.
	if per == 1 && q.Narrow != nil {
		if groups := (bhi - b) / QuantActNarrowBlocks; groups > 0 {
			a := args(b, 1, groups, b*Q8Block, Q8Block)
			a.Scratch = (*byte)(unsafe.Pointer(&scr[0]))
			qaNarrow.Add(1)
			q.Narrow.Call(&a)
			b += groups * QuantActNarrowBlocks
		}
	}
	if b >= bhi {
		return
	}
	if q.Wide == nil {
		quantGoLoop(t, dst, pairs, half, x, b, bhi, window)
		return
	}
	for b < bhi {
		w := (b / per) * per // this window's first block
		// The run of whole windows goes in one call: a call per 32-element
		// window costs more in the trampoline than the arithmetic. Only a
		// window the range starts inside, or the short one k ends inside,
		// needs a call of its own.
		if b == w && (w+per)*Q8Block <= k {
			cols := (bhi - b) / per
			if avail := (nb - b) / per; cols > avail {
				cols = avail
			}
			if cols > 0 {
				a := args(b, per, cols, b*Q8Block, window)
				qaWide.Add(1)
				q.Wide.Call(&a)
				b += cols * per
				continue
			}
		}
		end := min(w+per, bhi)
		w1 := min((w+per)*Q8Block, k)
		a := args(b, end-b, 1, w*Q8Block, w1-w*Q8Block)
		qaWide.Add(1)
		q.Wide.Call(&a)
		b = end
	}
}
