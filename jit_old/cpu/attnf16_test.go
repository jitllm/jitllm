package cpu_test

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestAttnF16MatchesF32OnExactHalves holds the f16 attention kernels to the f32
// ones bit for bit, on data exactly representable in binary16: then f16 loses
// nothing, so any difference is the kernel reading halves wrongly (which an
// NMSE against float64 would hide inside the rounding budget).
//
// The shapes are the real head dimensions: tinyllama 64, qwen3 128, gemma 256,
// and 8 as the narrowest the emitters accept.
func TestAttnF16MatchesF32OnExactHalves(t *testing.T) {
	// 3, 12 and 17 take the one-half tail load, which is the only code here
	// that is not a VCVTPH2PS of eight.
	for _, hd := range []int{8, 64, 128, 256, 3, 12, 17} {
		for _, npos := range []int{1, 2, 7, 33} {
			stride := hd * 3 // a kv stride that is not the head dim, as GQA gives
			rng := rand.New(rand.NewSource(int64(hd*9176 + npos)))

			// Values that are exactly halves, so f32 and f16 hold the same
			// number and only the kernel's addressing can differ.
			n := npos * stride
			f32 := make([]float32, n)
			f16 := make([]float32, (n+1)/2) // packed halves, two per slot
			hw := unsafe.Slice((*uint16)(unsafe.Pointer(&f16[0])), n)
			for i := range f32 {
				h := uint16(rng.Intn(1 << 16))
				if (h>>10)&0x1F == 0x1F { // no Inf/NaN
					h &^= 1 << 14
				}
				f32[i] = float32(quant.DecodeHalf(h))
				hw[i] = h
			}
			q := make([]float32, hd)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			w := make([]float32, npos)
			for i := range w {
				w[i] = float32(rng.Float64())
			}

			scores := func(useF16 bool, cache []float32) []float32 {
				c := onHost(t)(hostTable().AttnScores(hd, stride, cpu.KVOf(useF16)))
				defer c.Close()
				out := make([]float32, npos)
				args := cpu.Args{Out: &out[0], W: (*byte)(unsafe.Pointer(&cache[0])),
					Rows: int64(npos), Q32: &q[0]}
				c.Call(&args)
				return out
			}
			a, b := scores(false, f32), scores(true, f16)
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("scores hd=%d npos=%d: position %d f32 %v f16 %v",
						hd, npos, i, a[i], b[i])
				}
			}

			acc := func(useF16 bool, cache []float32) []float32 {
				c := onHost(t)(hostTable().AttnAcc(hd, stride, cpu.KVOf(useF16)))
				defer c.Close()
				out := make([]float32, hd)
				args := cpu.Args{Out: &out[0], W: (*byte)(unsafe.Pointer(&cache[0])),
					AScale: &w[0], Rows: int64(npos)}
				c.Call(&args)
				return out
			}
			c, d := acc(false, f32), acc(true, f16)
			for i := range c {
				if c[i] != d[i] {
					t.Fatalf("acc hd=%d npos=%d: element %d f32 %v f16 %v",
						hd, npos, i, c[i], d[i])
				}
			}
		}
	}
}
