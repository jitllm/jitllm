package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// GroupNormRows is Mamba-2's gated norm's RMSNorm: rows vectors of
// groups*width, each group of width normalised on its own and scaled by ITS
// slice of a groups*width weight. HeadNorm's weight is one head wide and
// shared by every head; a Mamba-2 norm weight is the whole of Inner, a slice a
// group (mamba_ssm's RMSNormGated with group_size, llama.cpp's reshape of
// ssm_norm into {Inner/Groups, Groups}).
//
// At lanes=32 one subgroup normalises one (row, group) and the launch is
// rows*groups groups of 32 (width a multiple of 32); at lanes=1 one thread
// does, in two serial passes, and the launch is rows*groups threads. The
// arithmetic is HeadNorm's, so a width of one group is HeadNorm exactly.
func GroupNormRows(groups, width int, eps float64, rows, lanes int) (*ir.Kernel, error) {
	if groups <= 0 || width <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: GroupNormRows groups=%d width=%d rows=%d", groups, width, rows)
	}
	n := groups * rows
	switch lanes {
	case 1:
		return groupNormScalar(groups, width, n, eps)
	case 32:
	default:
		return nil, fmt.Errorf("kernels: GroupNormRows: lanes=%d, want 32 or 1", lanes)
	}
	if width%32 != 0 {
		return nil, fmt.Errorf("kernels: GroupNormRows: width=%d must be a multiple of 32 at lanes=32", width)
	}
	per := width / 32
	b := ir.New("groupnorm", [3]int{32, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	h := b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(n-1)))
	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))
	base := b.Add(ir.U32, b.Mul(ir.U32, h, b.Const(ir.U32, int64(width))), lane)
	wBase := b.Add(ir.U32, b.Mul(ir.U32, b.Rem(ir.U32, h, b.Const(ir.U32, int64(groups))),
		b.Const(ir.U32, int64(width))), lane)
	acc := b.ConstF32(0)
	for k := 0; k < per; k++ {
		v := b.Load(ir.F32, pIn, base, int64(k*32))
		acc = b.Fma(v, v, acc)
	}
	for m := 1; m < 32; m <<= 1 {
		acc = b.Add(ir.F32, acc, b.ShuffleXor(ir.F32, acc, int64(m)))
	}
	mean := b.Mul(ir.F32, acc, b.ConstF32(float32(1)/float32(width)))
	inv := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(b.Add(ir.F32, mean, b.ConstF32(float32(eps)))))
	for k := 0; k < per; k++ {
		v := b.Load(ir.F32, pIn, base, int64(k*32))
		w := b.Load(ir.F32, pW, wBase, int64(k*32))
		b.Store(pOut, base, b.Mul(ir.F32, b.Mul(ir.F32, v, inv), w), int64(k*32))
	}
	return b.Done()
}

// groupNormScalar is GroupNormRows' one-thread form, headNormScalar with the
// weight read at the group's slice.
func groupNormScalar(groups, width, n int, eps float64) (*ir.Kernel, error) {
	b := ir.New("groupnormscalar", [3]int{64, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	h := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	base := b.Mul(ir.U32, h, b.Const(ir.U32, int64(width)))
	wBase := b.Mul(ir.U32, b.Rem(ir.U32, h, b.Const(ir.U32, int64(groups))), b.Const(ir.U32, int64(width)))
	one := b.Const(ir.U32, 1)
	zero := b.ConstF32(0)

	b.Loop(int64(width))
	acc := b.Phi(ir.F32, zero)
	si := b.Phi(ir.U32, base)
	v := b.Load(ir.F32, pIn, si, 0)
	b.SetPhi(acc, b.Fma(v, v, acc))
	b.SetPhi(si, b.Add(ir.U32, si, one))
	b.EndLoop()

	mean := b.Mul(ir.F32, acc, b.ConstF32(float32(1)/float32(width)))
	inv := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(b.Add(ir.F32, mean, b.ConstF32(float32(eps)))))

	b.Loop(int64(width))
	oi := b.Phi(ir.U32, base)
	wi := b.Phi(ir.U32, wBase)
	x := b.Load(ir.F32, pIn, oi, 0)
	w := b.Load(ir.F32, pW, wi, 0)
	b.Store(pOut, oi, b.Mul(ir.F32, b.Mul(ir.F32, x, inv), w), 0)
	b.SetPhi(oi, b.Add(ir.U32, oi, one))
	b.SetPhi(wi, b.Add(ir.U32, wi, one))
	b.EndLoop()
	return b.Done()
}
