//go:build amd64

package cpu

import "fmt"

// The causal convolution a linear-attention block runs over its own history.
//
// With the state stored plane-major (taps-1 planes of chans, rather than a
// window per channel as llama.cpp's SSM_CONV has it), it is a reduction over
// planes and fully elementwise, with no gather:
//
//	out[c] = x[c]*w[taps-1][c] + sum over t of state[t][c]*w[t][c]
//
// The weights are transposed once at load for the same reason. The state shift
// (state[t] = state[t+1], state[taps-2] = x) rides along, because the values it
// moves are already in registers; a separate pass would re-read the state.
//
// taps and chans are baked: both are container header fields.

// EmitConv1d generates the convolution and its state shift.
//
//	Out    out, chans floats; may alias Q32
//	W      the state, (taps-1)*chans float32, plane-major, read and written
//	AScale the transposed weights, taps*chans float32, plane-major
//	Q32    x, the new column, chans floats
func EmitConv1d(taps, chans int) ([]byte, error) {
	if taps < 2 || taps > 8 {
		return nil, fmt.Errorf("jit: EmitConv1d: taps=%d is outside [2,8]", taps)
	}
	if chans <= 0 {
		return nil, fmt.Errorf("jit: EmitConv1d: chans=%d must be positive", chans)
	}
	plane := int32(chans) * 4
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W:      the state planes
	a.MOVLoad(RSI, At(RDI, 24)) // AScale: the transposed weights
	a.MOVLoad(R8, At(RDI, 112)) // Q32:    the new column

	// body is one group of channels: whole vectors through ld/st at off 0 of
	// the advancing cursors, or one channel at off in the tail. Y15 holds a
	// weight plane, loaded rather than used as a memory operand so the tail
	// reads four bytes and not thirty-two.
	body := func(ld func(Reg, Mem), st func(Mem, Reg), off int32) {
		// The new column, and its own tap -- the last one, because the new
		// position is the most recent entry of the window.
		ld(Y0, At(R8, off))
		ld(Y15, At(RSI, int32(taps-1)*plane+off))
		a.VMULPS(Y1, Y0, Y15)
		// The state planes, all held at once: the shift below moves them.
		for t := 0; t < taps-1; t++ {
			ld(Reg(2+t), At(RDX, int32(t)*plane+off))
			ld(Y15, At(RSI, int32(t)*plane+off))
			a.VFMADD231PS(Y1, Reg(2+t), Y15)
		}
		// state[t] = state[t+1], and the newest column lands in the last plane.
		for t := 0; t < taps-2; t++ {
			st(At(RDX, int32(t)*plane+off), Reg(3+t))
		}
		st(At(RDX, int32(taps-2)*plane+off), Y0)
		// Out may alias Q32, so it is written only after the last read of x.
		st(At(RCX, off), Y1)
	}
	if blocks := chans / 8; blocks > 0 {
		a.MOVimm(RAX, int64(blocks))
		blk := a.Label()
		a.Bind(blk)
		body(a.VMOVDQULoad, a.VMOVDQUStore, 0)
		a.ADDimm(RCX, 32)
		a.ADDimm(RDX, 32)
		a.ADDimm(RSI, 32)
		a.ADDimm(R8, 32)
		a.DEC(RAX)
		a.JNZ(blk)
	}
	// chans is baked, so the last chans%8 are unrolled one channel at a time.
	for i := 0; i < chans%8; i++ {
		body(a.VMOVSSLoad, a.VMOVSSStore, int32(4*i))
	}

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
