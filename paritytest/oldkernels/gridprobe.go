package oldkernels

import "github.com/jitllm/jitllm/jit/gpu/ir"

// GridProbePhases is how many phases GridProbe runs.
const GridProbePhases = 8

// GridProbe is the grid barrier's gate kernel (ir/gridsync.go): groups
// workgroups of width threads run GridProbePhases phases separated by grid
// barriers. Phase 0 writes out0[i] = i for its own thread i; phase p writes
// out_p[i] = out_{p-1}[GridProbeSrc(p, i, groups, width)] + 1, a word a
// DIFFERENT workgroup wrote in the phase before. Before each write a workgroup
// spins for a time that differs by workgroup and phase (the sum goes to pSink
// so it is not removed), so a barrier that lets one workgroup through early has
// it read a word not yet written: the outputs start poisoned, and the probe's
// answer is then the poison plus something.
//
// The parameters are out0..out7, pSink, pSync.
func GridProbe(groups, width int) *ir.Kernel {
	n := groups * width
	b := ir.New("gridprobe", [3]int{width, 1, 1})
	var out [GridProbePhases]ir.Value
	for p := range out {
		out[p] = b.Param("out"+string(rune('0'+p)), ir.U32)
	}
	pSink := b.Param("pSink", ir.F32)
	pSync := b.Param("pSync", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	g := b.CTAID()
	i := b.Add(ir.U32, b.Mul(ir.U32, g, c(int64(width))), b.TID())
	b.Store(out[0], i, i, 0)
	acc0 := b.ConstF32(0)
	for p := 1; p < GridProbePhases; p++ {
		b.GridBarrier(pSync)
		// The read comes straight after the barrier and the write after the
		// skew, so a workgroup through early reads a word its writer is still
		// spinning before.
		v := b.Load(ir.U32, out[p-1], gridProbeSrcIR(b, p, i, groups, width), 0)
		// Skew: workgroup g spins ((g*7 + p*13) mod groups) * 256 steps.
		spin := b.Mul(ir.U32, b.Rem(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, g, c(7)), c(int64(p*13))), c(int64(groups))), c(256))
		b.LoopN(spin)
		acc := b.Phi(ir.F32, acc0)
		b.SetPhi(acc, b.Fma(acc, b.ConstF32(0.999), b.ConstF32(1)))
		b.EndLoop()
		b.Store(pSink, b.Add(ir.U32, i, c(int64((p-1)*n))), acc, 0)
		b.Store(out[p], i, b.Add(ir.U32, v, c(1)), 0)
	}
	return b.Done()
}

// gridProbeShift is how many workgroups away phase p reads: never its own
// when there are two or more.
func gridProbeShift(p, groups int) int {
	if groups > 1 {
		return 1 + p%(groups-1)
	}
	return 1
}

// GridProbeSrc is the thread whose phase p-1 word thread i reads in phase p.
func GridProbeSrc(p, i, groups, width int) int {
	return (i + gridProbeShift(p, groups)*width + p) % (groups * width)
}

func gridProbeSrcIR(b *ir.Builder, p int, i ir.Value, groups, width int) ir.Value {
	return b.Rem(ir.U32, b.Add(ir.U32, i, b.Const(ir.U32, int64(gridProbeShift(p, groups)*width+p))),
		b.Const(ir.U32, int64(groups*width)))
}

// GridProbeWant is what GridProbe leaves in out0..out7.
func GridProbeWant(groups, width int) [][]uint32 {
	n := groups * width
	out := make([][]uint32, GridProbePhases)
	out[0] = make([]uint32, n)
	for i := range out[0] {
		out[0][i] = uint32(i)
	}
	for p := 1; p < GridProbePhases; p++ {
		out[p] = make([]uint32, n)
		for i := 0; i < n; i++ {
			out[p][i] = out[p-1][GridProbeSrc(p, i, groups, width)] + 1
		}
	}
	return out
}
