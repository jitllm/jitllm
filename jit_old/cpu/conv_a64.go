//go:build arm64

package cpu

import "fmt"

// The arm64 causal convolution, the NEON twin of jit_old/cpu/conv.go.
//
// The plane offsets do not fit an arm64 load (LDR q reaches 65,520 bytes and
// planes can be 32 KiB apart), so where amd64 uses a disp32 from one base each
// plane here carries its own cursor register.
func EmitConv1d(taps, chans int) ([]byte, error) {
	if taps < 2 || taps > 8 {
		return nil, fmt.Errorf("jit: EmitConv1d: taps=%d is outside [2,8]", taps)
	}
	if chans <= 0 {
		return nil, fmt.Errorf("jit: EmitConv1d: chans=%d must be positive", chans)
	}
	plane := int32(chans) * 4
	var a A64
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X2, X0, 8)   // W:      the state planes
	a.LDRx(X3, X0, 24)  // AScale: the transposed weights
	a.LDRx(X4, X0, 112) // Q32:    the new column

	// A cursor per state plane (X10..) and per weight plane (X20..).
	st := func(t int) XReg { return XReg(10 + t) }
	wt := func(t int) XReg { return XReg(20 + t) }
	for t := 0; t < taps-1; t++ {
		a.MOVreg(st(t), X2)
		a.addBig(st(t), int32(t)*plane, X9)
	}
	for t := 0; t < taps; t++ {
		a.MOVreg(wt(t), X3)
		a.addBig(wt(t), int32(t)*plane, X9)
	}

	// body is one group of channels, whole vectors or ONE channel at off.
	body := func(ld func(VReg, XReg, int32), sto func(VReg, XReg, int32), off int32) {
		ld(V0, X4, off)
		ld(V1, wt(taps-1), off)
		a.FMUL4s(V1, V1, V0)
		for t := 0; t < taps-1; t++ {
			ld(VReg(2+t), st(t), off)
			ld(V15, wt(t), off)
			a.FMLA4s(V1, VReg(2+t), V15)
		}
		for t := 0; t < taps-2; t++ {
			sto(VReg(3+t), st(t), off)
		}
		sto(V0, st(taps-2), off)
		// Out may alias Q32, so it is written only after the last read of x.
		sto(V1, X1, off)
	}
	if blocks := chans / 4; blocks > 0 {
		a.MOVimm(X5, int64(blocks))
		blk := a.Label()
		a.Bind(blk)
		body(a.LDRq, a.STRq, 0)
		a.ADDimm(X1, X1, 16)
		a.ADDimm(X4, X4, 16)
		for t := 0; t < taps-1; t++ {
			a.ADDimm(st(t), st(t), 16)
		}
		for t := 0; t < taps; t++ {
			a.ADDimm(wt(t), wt(t), 16)
		}
		a.SUBimm(X5, X5, 1)
		a.CBNZ(X5, blk)
	}
	// chans is baked, so the last chans%4 are unrolled one channel at a time.
	for i := 0; i < chans%4; i++ {
		body(a.LDRs, a.STRs, int32(4*i))
	}

	a.RET()
	return a.Bytes(), nil
}
