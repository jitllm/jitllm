//go:build amd64

package cpu

// The AVX2 tier's image kernels: imgproc_sse.go's bodies behind a VZEROUPPER,
// so the legacy instructions meet clean upper halves and leave them clean.

func avx2Basic(body func(a *Buf)) []byte {
	var a Buf
	a.VZEROUPPER()
	body(&a)
	a.RET()
	return a.Bytes()
}

func EmitResampleH(prec int) []byte {
	return avx2Basic(func(a *Buf) { emitResampleHX86(a, prec) })
}
func EmitPixLUT() []byte { return avx2Basic(emitPixLUTX86) }
func EmitCopy32() []byte { return avx2Basic(emitCopy32X86) }

func EmitAxisTaps(f AxisFilter) []byte {
	if f > AxisCubic {
		return nil
	}
	return avx2Basic(func(a *Buf) { emitAxisTapsX86(a, f) })
}
func EmitLerpGrid() []byte  { return avx2Basic(emitLerpGridX86) }
func EmitSinCosTab() []byte { return avx2Basic(emitSinCosTabX86) }

func EmitPixelRow(f PixFmt, sub int, o PixOut) ([]byte, error) {
	if err := pixelShape(f, sub, o); err != nil {
		return nil, err
	}
	return avx2Basic(func(a *Buf) { emitPixelRowX86(a, f, sub, o) }), nil
}
