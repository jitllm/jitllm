package cpu

import (
	"regexp"
	"testing"
)

// TestImageKernelsKeepTheA64RegisterDiscipline disassembles every NEON image
// kernel on any host (the emitters carry no build tag) and refuses one that
// touches a register the runtime or the ABI owns, as
// TestA64KernelRegisterDiscipline does for the matvec. Executing them is the
// arm64 gates' job (imgproc_test.go and its siblings, run under the arm64
// host or an emulator).
func TestImageKernelsKeepTheA64RegisterDiscipline(t *testing.T) {
	mc := lookAny("llvm-mc-18", "llvm-mc", "llvm-mc-17", "llvm-mc-19")
	if mc == "" {
		t.Skip("no llvm-mc to disassemble aarch64")
	}
	kernels := map[string][]byte{
		"resample":   EmitA64Resample(22),
		"resample_h": EmitA64ResampleH(22),
		"pixlut":     EmitA64PixLUT(),
		"copy32":     EmitA64Copy32(),
		"axistaps":   EmitA64AxisTaps(AxisBilinear),
		"axistapsl":  EmitA64AxisTaps(AxisLinear),
		"axistaps4":  EmitA64AxisTaps(AxisCubic),
		"lerpgrid":   EmitA64LerpGrid(),
		"sincostab":  EmitA64SinCosTab(),
	}
	for _, f := range []PixFmt{PixRGBA, PixNRGBA, PixGray, PixRGBA64, PixYCbCr} {
		for _, o := range []PixOut{PixOver8, PixPlanes} {
			sub := 0
			if f == PixYCbCr {
				sub = 2
			}
			b, err := EmitA64PixelRow(f, sub, o)
			if err != nil {
				t.Fatal(err)
			}
			kernels["pixelrow"+string(rune('0'+f))+string(rune('0'+o))] = b
		}
	}
	for name, code := range kernels {
		text := a64Disasm(t, mc, code)
		for _, reg := range []string{"x18", "x27", "x28", "x29", "x30", "sp", "xzr", "wzr"} {
			if regexp.MustCompile(`\b` + reg + `\b`).MatchString(text) {
				t.Errorf("%s touches %s, which the runtime or the ABI owns:\n%s", name, reg, text)
			}
		}
	}
	t.Logf("%d NEON image kernels disassembled clean", len(kernels))
}
