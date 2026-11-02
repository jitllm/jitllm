package cpu

// arm64 has one tier, so the image kernels' architecture-neutral names are
// the NEON emitters (imgproc_neon.go).

func EmitResampleH(prec int) []byte    { return EmitA64ResampleH(prec) }
func EmitPixLUT() []byte               { return EmitA64PixLUT() }
func EmitCopy32() []byte               { return EmitA64Copy32() }
func EmitAxisTaps(f AxisFilter) []byte { return EmitA64AxisTaps(f) }
func EmitLerpGrid() []byte             { return EmitA64LerpGrid() }
func EmitSinCosTab() []byte            { return EmitA64SinCosTab() }

func EmitPixelRow(f PixFmt, sub int, o PixOut) ([]byte, error) { return EmitA64PixelRow(f, sub, o) }
