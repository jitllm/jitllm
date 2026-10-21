//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// f16Codes maps NarrowF16 and the f16 Widen on every tier this host runs
// (q8Tables: the host's, and the SSE tier on an AVX2 host).
func f16Codes(t *testing.T) (narrow, widen map[string]*cpu.Code) {
	narrow, widen = map[string]*cpu.Code{}, map[string]*cpu.Code{}
	for name, em := range q8Tables() {
		narrow[name] = mapQ8(t, name+"/narrow_f16")(em.NarrowF16())
		widen[name] = mapQ8(t, name+"/widen_f16")(em.Widen(false))
	}
	return narrow, widen
}

func runNarrow(c *cpu.Code, dst []uint16, src []float32) {
	n := len(src)
	c.Call(&cpu.Args{Out: (*float32)(unsafe.Pointer(&dst[0])), W: (*byte)(unsafe.Pointer(&src[0])),
		K: int64(n / cpu.ElemLanes), Rows: int64(n % cpu.ElemLanes)})
}

// TestNarrowF16IsEncodeHalfOnEveryFloat holds the f16 KV cache's store to
// quant.EncodeHalf bit for bit over every one of the 2^32 float32 patterns:
// every rounding boundary at every exponent, the subnormal halves, overflow,
// both zeros, Inf, and NaN of every payload. -short sweeps every 251st
// pattern instead.
func TestNarrowF16IsEncodeHalfOnEveryFloat(t *testing.T) {
	narrow, _ := f16Codes(t)
	const chunk = 1 << 20
	step := uint64(1)
	if testing.Short() {
		step = 251
	}
	src := make([]float32, chunk)
	want := make([]uint16, chunk)
	got := make([]uint16, chunk)
	for base := uint64(0); base < 1<<32; base += chunk * step {
		for i := range src {
			src[i] = math.Float32frombits(uint32(base + uint64(i)*step))
			want[i] = quant.EncodeHalf(src[i])
		}
		for name, c := range narrow {
			runNarrow(c, got, src)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s: float32 %#08x narrows to %#04x, EncodeHalf says %#04x",
						name, math.Float32bits(src[i]), got[i], want[i])
				}
			}
		}
	}
}

// TestWidenF16IsDecodeHalfOnEveryHalf holds the f16 widening to
// quant.DecodeHalf on all 2^16 halves: bit for bit, but a NaN, which must stay
// a NaN (DecodeHalf's is math.NaN() whatever the half's sign and payload; the
// kernel keeps both, quieted).
func TestWidenF16IsDecodeHalfOnEveryHalf(t *testing.T) {
	_, widen := f16Codes(t)
	src := make([]uint16, 1<<16)
	for i := range src {
		src[i] = uint16(i)
	}
	got := make([]float32, len(src))
	for name, c := range widen {
		c.Call(&cpu.Args{Out: &got[0], W: (*byte)(unsafe.Pointer(&src[0])),
			K: int64(len(src) / cpu.ElemLanes), Rows: int64(len(src) % cpu.ElemLanes)})
		for i, h := range src {
			want := float32(quant.DecodeHalf(h))
			g, w := math.Float32bits(got[i]), math.Float32bits(want)
			if math.IsNaN(float64(want)) {
				if !math.IsNaN(float64(got[i])) {
					t.Fatalf("%s: half %#04x widens to %#08x, want a NaN", name, h, g)
				}
				continue
			}
			if g != w {
				t.Fatalf("%s: half %#04x widens to %#08x, DecodeHalf says %#08x", name, h, g, w)
			}
		}
	}
}

// TestNarrowF16Tail runs every length from 1 to five vectors with a guard
// after the output: the tail is right and nothing past it is written.
func TestNarrowF16Tail(t *testing.T) {
	narrow, _ := f16Codes(t)
	const guard = 0xA5A5
	for name, c := range narrow {
		for n := 1; n <= 5*cpu.ElemLanes+3; n++ {
			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i+1)*0.3711 - 3
			}
			dst := make([]uint16, n+16)
			for i := range dst {
				dst[i] = guard
			}
			runNarrow(c, dst, src)
			for i := 0; i < n; i++ {
				if want := quant.EncodeHalf(src[i]); dst[i] != want {
					t.Fatalf("%s n=%d: element %d is %#04x, want %#04x", name, n, i, dst[i], want)
				}
			}
			for i := n; i < len(dst); i++ {
				if dst[i] != guard {
					t.Fatalf("%s n=%d: wrote %#04x past the end at %d", name, n, dst[i], i)
				}
			}
		}
	}
}
