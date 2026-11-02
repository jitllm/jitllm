package cpu

import "fmt"

// DWShape is one depthwise convolution row's shape, every number baked into
// its kernel (EmitDWConv): a K x K filter at Stride over channel-last rows of
// Chans floats, the padded input WP samples wide, WOut outputs a row.
type DWShape struct {
	K, Stride, WP, Chans, WOut int
}

// Key names the shape, for a kernel cache.
func (s DWShape) Key() string {
	return fmt.Sprintf("dwconv_k%d_s%d_wp%d_c%d_w%d", s.K, s.Stride, s.WP, s.Chans, s.WOut)
}

// check refuses a shape no kernel can be baked for: a window past the padded
// row, or offsets past what a 32-bit displacement reaches.
func (s DWShape) check(who string) error {
	if s.K < 1 || s.K > 7 || s.Stride < 1 || s.Chans < 1 || s.WOut < 1 {
		return fmt.Errorf("jit: %s: K=%d stride=%d chans=%d wout=%d", who, s.K, s.Stride, s.Chans, s.WOut)
	}
	if (s.WOut-1)*s.Stride+s.K > s.WP {
		return fmt.Errorf("jit: %s: %d outputs at stride %d of a %d filter read past a %d-sample row",
			who, s.WOut, s.Stride, s.K, s.WP)
	}
	if int64(s.K*s.WP+s.K)*int64(s.Chans)*4 >= 1<<31 {
		return fmt.Errorf("jit: %s: a %dx%d window of %d channels is past a 32-bit offset", who, s.K, s.WP, s.Chans)
	}
	return nil
}

// EmitDWConvSSE is EmitDWConv at four lanes, with a multiply and an add where
// AVX2 has a fused one.
func EmitDWConvSSE(s DWShape) ([]byte, error) {
	if err := s.check("EmitDWConvSSE"); err != nil {
		return nil, err
	}
	plane := int32(s.Chans) * 4
	var a Buf
	a.DeclareISA(ISASSE2)
	a.MOVLoad(RCX, At(RDI, 0))
	a.MOVLoad(RSI, At(RDI, 24))
	a.MOVLoad(R8, At(RDI, 112))
	a.MOVimm(RAX, int64(s.WOut))
	xl := a.Label()
	a.Bind(xl)
	a.MOVQ(R9, R8)
	a.MOVQ(R10, RSI)
	a.MOVQ(R11, RCX)
	body := func(ld func(Reg, Mem), st func(Mem, Reg), off int32) {
		a.XORPS(XMM0, XMM0, XMM0)
		for ky := 0; ky < s.K; ky++ {
			for kx := 0; kx < s.K; kx++ {
				t := int32(ky*s.K + kx)
				ld(XMM1, At(R9, int32(ky*s.WP+kx)*plane+off))
				ld(XMM15, At(R10, t*plane+off))
				a.MULPS(XMM1, XMM1, XMM15)
				a.ADDPS(XMM0, XMM0, XMM1)
			}
		}
		st(At(R11, off), XMM0)
	}
	if groups := s.Chans / 4; groups > 0 {
		a.MOVimm(RDX, int64(groups))
		cl := a.Label()
		a.Bind(cl)
		body(a.MOVUPSLoad, a.MOVUPSStore, 0)
		a.ADDimm(R9, 16)
		a.ADDimm(R10, 16)
		a.ADDimm(R11, 16)
		a.DEC(RDX)
		a.JNZ(cl)
	}
	for i := 0; i < s.Chans%4; i++ {
		body(a.MOVSSLoad, a.MOVSSStore, int32(4*i))
	}
	a.ADDimm(R8, int32(s.Stride)*plane)
	a.ADDimm(RCX, plane)
	a.DEC(RAX)
	a.JNZ(xl)
	a.RET()
	return a.Bytes(), nil
}
