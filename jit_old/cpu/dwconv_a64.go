//go:build arm64

package cpu

// EmitDWConv is jit_old/cpu/dwconv.go's depthwise row on NEON. A tap's window
// offset does not fit an arm64 load (LDR q reaches 65,520 bytes), so each tap
// forms its address in X9 from the window cursor; the filter is walked by its
// own cursor, one plane a tap.
func EmitDWConv(s DWShape) ([]byte, error) {
	if err := s.check("EmitDWConv"); err != nil {
		return nil, err
	}
	plane := int32(s.Chans) * 4
	var a A64
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X3, X0, 24)  // AScale: the filter
	a.LDRx(X4, X0, 112) // Q32:    the window
	a.MOVimm(X5, int64(s.WOut))
	xl := a.Label()
	a.Bind(xl)
	a.MOVreg(X10, X4) // the window's cursor
	a.MOVreg(X11, X3) // the filter's
	a.MOVreg(X12, X1) // the output's
	body := func(ld func(VReg, XReg, int32), sto func(VReg, XReg, int32), off int32) {
		a.MOVIzero(V0)
		a.MOVreg(X13, X11)
		for ky := 0; ky < s.K; ky++ {
			for kx := 0; kx < s.K; kx++ {
				a.MOVreg(X9, X10)
				a.addBig(X9, int32(ky*s.WP+kx)*plane, X8)
				ld(V1, X9, off)
				ld(V2, X13, off)
				a.FMLA4s(V0, V1, V2)
				a.addBig(X13, plane, X8)
			}
		}
		sto(V0, X12, off)
	}
	if blocks := s.Chans / 4; blocks > 0 {
		a.MOVimm(X6, int64(blocks))
		cl := a.Label()
		a.Bind(cl)
		body(a.LDRq, a.STRq, 0)
		a.ADDimm(X10, X10, 16)
		a.ADDimm(X11, X11, 16)
		a.ADDimm(X12, X12, 16)
		a.SUBimm(X6, X6, 1)
		a.CBNZ(X6, cl)
	}
	for i := 0; i < s.Chans%4; i++ {
		body(a.LDRs, a.STRs, int32(4*i))
	}
	a.addBig(X4, int32(s.Stride)*plane, X8)
	a.addBig(X1, plane, X8)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, xl)
	a.RET()
	return a.Bytes(), nil
}
