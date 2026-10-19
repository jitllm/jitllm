package kernels

import "github.com/jitllm/jitllm/jit/gpu/ir"

// Fragments: the pieces of a kernel an op emits into a loop nest someone
// else forms (jit/gpu/compose), as opposed to the whole kernels the rest of
// this package builds. Each is the arithmetic its template uses, so a formed
// kernel and the template it is gated against compute the same bits.

// ActFragment is activation k of v.
func ActFragment(b *ir.Builder, v ir.Value, k ActKind) ir.Value { return act(b, v, k) }

// ActMulFragment is k's gated form of gate times the value up produces.
func ActMulFragment(b *ir.Builder, k ActKind, gate ir.Value, up func() ir.Value) ir.Value {
	return actMulValue(b, k, gate, up)
}

// ActMulOK refuses an activation with no gated form.
func ActMulOK(k ActKind) error { return actMulKind(k) }

// QuantFragment quantizes row r of a group of g threads (tid) into the int8
// operand Quantize writes: elem is the row's value at a column, k the row's
// width (a multiple of 32), rows the operand's rows, window the scale's
// span. shMax and shSum are threadgroup arrays of g floats. Exactly
// Quantize's bits.
func QuantFragment(b *ir.Builder, k, rows, window, g int, r, tid, pA, pAX, shMax, shSum ir.Value,
	elem func(at ir.Value) ir.Value) {
	quantPhase(b, k, rows, window, g, r, tid, pA, pAX, shMax, shSum, elem)
}

// TreeSum is the sum of v over a group of g threads through the threadgroup
// array sh, every thread holding it after; the array is free again after.
func TreeSum(b *ir.Builder, sh, tid, v ir.Value, g int) ir.Value {
	out := treeSum(b, sh, tid, v, g)
	b.Barrier()
	return out
}

// TreeMax is TreeSum's maximum.
func TreeMax(b *ir.Builder, sh, tid, v ir.Value, g int) ir.Value { return treeMax(b, sh, tid, v, g) }
