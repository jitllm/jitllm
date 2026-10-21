//go:build amd64

package cpu

// EmitDWConv generates one output row of a depthwise convolution over
// channel-last rows, every number of its shape baked (DWShape):
//
//	out[x][c] = sum over ky, kx < K of in[ky*WP + x*Stride + kx][c] * w[ky*K+kx][c]
//
// for x < WOut and c < Chans. in is the padded input's top-left sample of
// the output row's window -- the caller pads (SAME, as the reference does)
// and points each call at row y*Stride -- and w the filter, tap-major, Chans
// floats a tap.
//
//	Out    out, WOut*Chans floats, written
//	AScale w, K*K*Chans floats
//	Q32    in, the window's first sample; rows WP*Chans floats apart
//
// The accumulator stays in a register across every tap of a vector of
// channels; groups of eight run through a loop, the last Chans%8 one channel
// at a time.
func EmitDWConv(s DWShape) ([]byte, error) {
	if err := s.check("EmitDWConv"); err != nil {
		return nil, err
	}
	plane := int32(s.Chans) * 4
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RSI, At(RDI, 24))  // AScale: the filter
	a.MOVLoad(R8, At(RDI, 112))  // Q32:    the window
	a.MOVimm(RAX, int64(s.WOut)) // positions left
	xl := a.Label()
	a.Bind(xl)
	a.MOVQ(R9, R8)   // the window's cursor
	a.MOVQ(R10, RSI) // the filter's
	a.MOVQ(R11, RCX) // the output's
	// body is one group of channels: whole vectors through ld/st at off 0 of
	// the advancing cursors, or one channel at off in the tail. Y0 is the sum.
	body := func(ld func(Reg, Mem), st func(Mem, Reg), off int32) {
		a.VPXOR(Y0, Y0, Y0)
		for ky := 0; ky < s.K; ky++ {
			for kx := 0; kx < s.K; kx++ {
				t := int32(ky*s.K + kx)
				ld(Y1, At(R9, int32(ky*s.WP+kx)*plane+off))
				ld(Y15, At(R10, t*plane+off))
				a.VFMADD231PS(Y0, Y1, Y15)
			}
		}
		st(At(R11, off), Y0)
	}
	if blocks := s.Chans / 8; blocks > 0 {
		a.MOVimm(RDX, int64(blocks))
		cl := a.Label()
		a.Bind(cl)
		body(a.VMOVDQULoad, a.VMOVDQUStore, 0)
		a.ADDimm(R9, 32)
		a.ADDimm(R10, 32)
		a.ADDimm(R11, 32)
		a.DEC(RDX)
		a.JNZ(cl)
	}
	for i := 0; i < s.Chans%8; i++ {
		body(a.VMOVSSLoad, a.VMOVSSStore, int32(4*i))
	}
	a.ADDimm(R8, int32(s.Stride)*plane)
	a.ADDimm(RCX, plane)
	a.DEC(RAX)
	a.JNZ(xl)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
