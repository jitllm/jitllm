package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestQuantizeOnTheRoundingBoundary holds every device's activation quantizer
// to the host's BIT FOR BIT on the inputs where the two can differ: elements
// a few ulps either side of a rounding boundary (n+0.5 after scaling), so an
// inv one ulp off the host's 1/(amax/127) moves them to the next int8. The
// scales are compared as bits too. A smooth input almost never lands there,
// which is how mainkernels.Quantize multiplied by 127/amax -- and Vulkan divided to
// 2.5 ulp -- behind green gates on sines and ramps, while the real
// granite-3.3-2b's first block parted from the host at NMSE 1e-4 on CUDA.
//
// The input is checked to discriminate before the device runs: the old
// arithmetic (inv = 127/amax, d = amax*(1/127)), computed here in Go, must
// move hundreds of its int8s.
func TestQuantizeOnTheRoundingBoundary(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	for _, window := range []int{32, 256} {
		const k = 1 << 16
		x := boundaryActivations(rand.New(rand.NewSource(int64(window))), k, window)
		wantA := make([]uint32, k/4)
		wantD := make([]float32, k/32)
		wantS := make([]float32, k/16)
		if err := kernels.PackActivationsInto(wantA, wantD, wantS, x, window); err != nil {
			t.Fatal(err)
		}
		if moved := oldQuantMoves(x, window, wantA); moved < 200 {
			t.Fatalf("window %d: the old arithmetic moves %d int8s of %d: the input does not sit on the "+
				"boundary, and this gate would prove nothing", window, moved, k)
		} else {
			t.Logf("window %d: %d of %d int8s move under inv = 127/amax", window, moved, k)
		}
		for _, d := range devs {
			t.Run(fmt.Sprintf("%s/w%d", d.API(), window), func(t *testing.T) {
				quantBitsCase(t, d, x, window, wantA, wantD, wantS)
			})
		}
	}
}

// boundaryActivations is k values whose every window holds its amax at its
// first element and, elsewhere, values within two ulps of (n+0.5)*d for the
// host's d = amax/127 and |n| from 1 to 125. n = 0 is left out: the host's
// generated kernel and the device both round as trunc(f + 0.5) in f32, which
// parts from math.Round just below 0.5 on both, so it is not a device
// question.
func boundaryActivations(rng *rand.Rand, k, window int) []float32 {
	x := make([]float32, k)
	for w := 0; w < k; w += window {
		amax := float32(math.Exp(rng.Float64()*16 - 8))
		if rng.Intn(2) == 0 {
			x[w] = amax
		} else {
			x[w] = -amax
		}
		inv := 1 / (amax / 127)
		for i := w + 1; i < w+window; i++ {
			n := float64(1 + rng.Intn(125))
			if rng.Intn(2) == 0 {
				n = -n
			}
			v := float32((n + math.Copysign(0.5, n)) / float64(inv))
			bits := int64(math.Float32bits(v)) + int64(rng.Intn(5)-2)
			x[i] = math.Float32frombits(uint32(bits))
		}
	}
	return x
}

// oldQuantMoves counts the int8s the quantizer's former arithmetic -- inv =
// 127/amax, rounded half away from zero -- gives differently from want.
func oldQuantMoves(x []float32, window int, want []uint32) int {
	per := max(1, window/32)
	moved := 0
	for b := 0; b < len(x)/32; b++ {
		w0 := (b / per) * per * 32
		var amax float32
		for _, v := range x[w0 : w0+per*32] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		inv := 127 / amax
		for i := b * 32; i < b*32+32; i++ {
			f := x[i] * inv
			q := int8(int32(f + float32(math.Copysign(0.5, float64(f)))))
			if q != int8(want[i/4]>>(8*(i%4))) {
				moved++
			}
		}
	}
	return moved
}

func quantBitsCase(t *testing.T, d backend.Device, x []float32, window int, wantA []uint32, wantD, wantS []float32) {
	k := len(x)
	kern, err := mainkernels.Quantize(k, window)
	if err != nil {
		t.Fatal(err)
	}
	kk, err := d.Compile(kern)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kk.Close()
	g := newGPU(t, d)
	defer g.free()
	nan := make([]float32, k/32+k/16)
	for i := range nan {
		nan[i] = float32(math.NaN())
	}
	bx, ba, bax := g.up(f32bytes(x)), g.up(make([]byte, k)), g.up(f32bytes(nan))
	if err := kk.Launch((mainkernels.QuantizeThreads(k/32)+127)/128, 128, bx, ba, bax); err != nil {
		t.Fatalf("launch: %v", err)
	}
	gotQ := make([]byte, k)
	raw := make([]byte, 4*(k/32+k/16))
	if ba.Read(gotQ) != nil || bax.Read(raw) != nil {
		t.Fatal("read")
	}
	f := func(i int) float32 {
		return math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 |
			uint32(raw[4*i+3])<<24)
	}
	moved, first := 0, -1
	for i := 0; i < k; i++ {
		if gotQ[i] != byte(wantA[i/4]>>(8*(i%4))) {
			if moved++; first < 0 {
				first = i
			}
		}
	}
	scales := 0
	for i, w := range wantD {
		if math.Float32bits(f(i)) != math.Float32bits(w) {
			scales++
		}
	}
	sums := 0
	for i, w := range wantS {
		if f(k/32+i) != w {
			sums++
		}
	}
	if moved+scales+sums > 0 {
		t.Fatalf("%s at a %d-element window: %d of %d int8s (first at %d), %d of %d scales and %d of %d sums "+
			"differ from the host's", d.API(), window, moved, k, first, scales, len(wantD), sums, len(wantS))
	}
	t.Logf("%s: %d int8s on the rounding boundary and %d scales identical to the host's", d.API(), k, len(wantD))
}
