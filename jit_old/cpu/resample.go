//go:build amd64

package cpu

// EmitResample generates one output row of a fixed-point resampling pass: the
// image processors' antialiased resize (Pillow's, torchvision's), whose every
// pass is integer arithmetic on 8-bit samples:
//
//	out[x] = min(max((bias + sum_j taps[j] * in_j[x]) >> prec, 0), 255)
//
// in_j is the j-th source row the output row reads, RowStr bytes after the
// one before it: the vertical pass, which reads image rows (the horizontal
// pass is EmitResampleH, a pixel at a time). The samples are uint8 in and
// out, widened to int32 lanes inside; bias is the rounding term 1<<(prec-1).
// prec is baked, so a kernel is per precision. resample_const.go has the
// constant block.
//
//	Out     uint8 out[n]
//	AScale  uint8 in_0
//	W       int32 taps[Cols]
//	Cols    the number of taps, at least one
//	RowStr  bytes from in_j to in_j+1
//	Scr     ResampleConsts(prec)
//	K, Rows n/8 whole vectors and n%8 single samples
//
// Registers: RCX out, RDX in, R9 taps, R12 taps left, R13 RowStr, R10/R11
// counters; R8 and RAX walk one sample's taps. Y0 bias, Y2 255, Y6 the byte
// gather, Y7 zero.
func EmitResample(prec int) []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> in_0
	a.MOVLoad(R9, At(RDI, 8))   // W -> taps
	a.MOVLoad(R12, At(RDI, 80)) // Cols -> taps
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	a.VPBROADCASTD(Y0, At(RBX, resampleBiasOff))
	a.VPBROADCASTD(Y2, At(RBX, resampleMaxOff))
	a.VMOVDQULoad(Y6, At(RBX, resampleGatherOff))
	a.VPXOR(Y7, Y7, Y7)

	body := func(ld func(dst, p Reg), st func(p, src Reg)) {
		a.VPXOR(Y1, Y1, Y1)
		a.MOVQ(R8, RDX)
		a.MOVQ(RAX, R9)
		a.MOVQ(RSI, R12)
		taps := a.Label()
		a.Bind(taps)
		ld(Y3, R8)
		a.VPBROADCASTD(Y4, At(RAX, 0))
		a.VPMULLD(Y3, Y3, Y4)
		a.VPADDD(Y1, Y1, Y3)
		a.ADDQ(R8, R13)
		a.ADDimm(RAX, 4)
		a.DEC(RSI)
		a.JNZ(taps)
		a.VPADDD(Y1, Y1, Y0)
		a.VPSRAD(Y1, Y1, byte(prec))
		// The clamp at zero without a signed max: the sign mask, cleared.
		a.VPSRAD(Y3, Y1, 31)
		a.VPANDN(Y1, Y3, Y1)
		a.VPMINSD(Y1, Y1, Y2)
		st(RCX, Y1)
	}
	byteLoops(&a, R10, R11, []Reg{RCX, RDX}, 8, func() {
		// Eight samples: zero-extended in, and out as the low byte of each
		// lane gathered within each 128-bit half, the halves joined.
		body(func(dst, p Reg) { a.VPMOVZXBD(dst, At(p, 0)) }, func(p, src Reg) {
			a.VPSHUFB(src, src, Y6)
			a.VEXTRACTI128(Y5, src, 1)
			a.VPUNPCKLDQx(src, src, Y5)
			a.VMOVQStore(At(p, 0), src)
		})
	}, func() {
		// One sample: exactly its byte in and out.
		body(func(dst, p Reg) { a.VPINSRBLoad(dst, Y7, At(p, 0), 0) }, func(p, src Reg) {
			a.VPEXTRBStore(At(p, 0), src, 0)
		})
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
