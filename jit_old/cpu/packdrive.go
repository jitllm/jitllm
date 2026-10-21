package cpu

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// PackActConsts is the constant block EmitPackAct reads, on both architectures:
// [127, 1.0, 0.5, absmask, signmask, bias]. arm64 has FABS and ignores the
// absmask slot; it is kept so one layout serves both emitters.
func PackActConsts(t quant.Type) []float32 {
	return []float32{127, 1, 0.5,
		math.Float32frombits(0x7FFFFFFF), math.Float32frombits(0x80000000),
		math.Float32frombits(uint32(int32(BiasC(t))))}
}

// PackActJIT is packGEMM driven by the generated kernel, one call per
// (token, amax window).
//
// The window is the call unit, which keeps the call overhead small (a k=2048
// matrix at the wide window is eight calls per token). The kernel is handed
// the whole window for the amax scan even when the caller's block range
// covers only part of it, because the scale belongs to the window: otherwise
// two workers sharing a window would quantize with different scales.
func PackActJIT(code *Code, t quant.Type, konst []float32, dst []int8, scale []float32,
	sum, half []int32, x []float32, tok, k, blo, bhi, window int) error {
	if code == nil {
		return packGEMM(t, dst, scale, sum, half, x, tok, k, 0, tok, blo, bhi, window)
	}
	if k%Q8Block != 0 {
		return fmt.Errorf("jit: PackActJIT: k=%d is not a multiple of %d", k, Q8Block)
	}
	nb := k / Q8Block
	per := window / Q8Block
	if per < 1 {
		per = 1
	}
	if len(dst) < tok*k || len(x) < tok*k || len(scale) < nb*tok || len(sum) < nb*tok {
		return fmt.Errorf("jit: PackActJIT: short buffer")
	}
	wantHalf := half != nil && NeedsHalfSums(t)
	for n := 0; n < tok; n++ {
		row := x[n*k : (n+1)*k]
		for b := blo; b < bhi; {
			w := (b / per) * per // this window's first block
			end := min(w+per, bhi)
			w1 := min((w+per)*Q8Block, k)
			args := Args{
				A:    (*int8)(unsafe.Pointer(&row[w*Q8Block])),
				K:    int64(w1 - w*Q8Block),
				Q32:  &row[b*Q8Block],
				Rows: int64(end - b),
				W:    (*byte)(unsafe.Pointer(&dst[((b*Q8Block/4)*tok+n)*4])),
				Out:  &scale[b*tok+n],
				ASum: &sum[b*tok+n],
				Scr:  (*byte)(unsafe.Pointer(&konst[0])),
			}
			if wantHalf {
				args.AHalfSum = &half[2*b*tok+n]
			}
			code.Call(&args)
			b = end
		}
	}
	return nil
}
