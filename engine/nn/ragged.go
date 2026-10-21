//go:build amd64 || arm64

package nn

import (
	"math"
	"sync/atomic"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// A ragged activation is kx floats read by a weight packed at k, the next
// whole number of its format's outer blocks past kx (cpu.PackedOuterElems):
// a tower 144 wide whose Q8_0 matrices were packed at 160 with zero columns.
// The quantizer takes the tail itself -- the activation is quantized as k
// values whose last k-kx are zero -- so the products run unchanged over k,
// as the device's kernels.QuantizePadded does. The weight's columns past kx
// are zero and so is the activation's tail: neither adds anything, and the
// tail's zeros keep its window's scale the dense row's.

// raggedK is the packed width of a k-wide activation under t: k itself when
// it is a whole number of t's outer blocks.
func raggedK(t quant.Type, k int) int {
	// The host's outer step and the format's own block, both powers of two.
	step := cpu.PackedOuterElems(t)
	if q, ok := kernels.QuantOf(t); ok {
		step = max(step, q.Elems())
	}
	if step == 0 || k%step == 0 {
		return k
	}
	return (k + step - 1) / step * step
}

// quantRow quantizes one activation row x (kx floats) as k: every window
// wholly inside x straight from x, and the windows that reach past it from a
// worker's stage, x's last floats copied in and the rest zero (generated
// copies). blo and bhi are 32-element blocks of the k-wide row, as for
// QuantActKernels.Run, so a caller can split a row's blocks across workers.
func (f *JIT) quantRow(qk cpu.QuantActKernels, t quant.Type, q []int8, pairs, half, scr []float32,
	worker int, x []float32, kx, k, blo, bhi, win int) {
	if kx == k {
		qk.Run(t, q, pairs, half, scr, x, k, blo, bhi, win)
		return
	}
	w := max(win, cpu.Q8Block)
	full := kx / w * w // the windows x holds whole
	if fb := full / cpu.Q8Block; blo < fb {
		qk.Run(t, q, pairs, half, scr, x, full, blo, min(bhi, fb), win)
		blo = fb
	}
	if blo >= bhi {
		return
	}
	RaggedRows.Add(1)
	n := k - full
	st := f.ragStage[worker*f.ragStride : worker*f.ragStride+n]
	Copy32JIT(st, 0, 1, x[full:kx], 0, 1, 1, kx-full)
	Copy32JIT(st[kx-full:], 0, 1, ragZero[:], 0, 0, 1, n-(kx-full))
	if f.ragViolation {
		// A gate's violation: the tail NaN where it should be zero, which a
		// zero weight column turns into NaN.
		Copy32JIT(st[kx-full:], 0, 1, ragNaN[:], 0, 0, 1, n-(kx-full))
	}
	fb := full / cpu.Q8Block
	var h []float32
	if half != nil {
		h = half[full/16:]
	}
	qk.Run(t, q[full:], pairs[2*fb:], h, scr, st, n, blo-fb, bhi-fb, win)
}

// RaggedRows counts the activation rows whose tail a quantizer staged, on
// every JIT, for the gates that must show a ragged product ran.
var RaggedRows atomic.Int64

// ragZero and ragNaN are one-element sources the stage's tail is filled from
// (a zero element stride repeats them).
var (
	ragZero = [1]float32{0}
	ragNaN  = [1]float32{nanF32()}
)

func nanF32() float32 { return float32(math.NaN()) }

// growRagStage makes a stage of k floats a worker for every worker.
func (f *JIT) growRagStage(k int) {
	if k <= f.ragStride && len(f.ragStage) >= f.pool.Max()*f.ragStride {
		return
	}
	f.ragStride = max(k, f.ragStride)
	f.ragStage = make([]float32, f.pool.Max()*f.ragStride)
}
