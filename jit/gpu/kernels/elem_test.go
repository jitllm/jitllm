package kernels_test

import (
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// dev returns the one device this test binary uses, opened once: a CUDA
// context costs tens of milliseconds and every test here wants the same one.
// It is reclaimed at process exit.
var (
	devOnce sync.Once
	devOne  backend.Device
)

func dev(t *testing.T) backend.Device {
	t.Helper()
	devOnce.Do(func() {
		ds := backend.Open()
		if len(ds) == 0 {
			return
		}
		for _, d := range ds[1:] {
			d.Close()
		}
		devOne = ds[0]
	})
	if devOne == nil {
		t.Skip("no GPU on this host")
	}
	return devOne
}

func f32b(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		u := math.Float32bits(x)
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	return b
}

func b2f(b []byte, v []float32) {
	for i := range v {
		v[i] = math.Float32frombits(uint32(b[4*i]) | uint32(b[4*i+1])<<8 |
			uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
	}
}

func ramp(n int, f func(i int) float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = f(i)
	}
	return v
}

// TestQuantizeMatchesCPU checks the GPU's int8 is byte for byte the int8 the
// CPU kernel produces; otherwise the tiers run different arithmetic and every
// comparison between them is void.
func TestQuantizeMatchesCPU(t *testing.T) {
	d := dev(t)
	const k = 2048
	x := ramp(k, func(i int) float32 {
		return float32(math.Sin(float64(i)*0.37)) * float32(1+i%7)
	})
	x64 := x
	wantA := make([]uint32, k/4)
	wantD := make([]float32, k/32)
	wantS := make([]float32, k/16)
	if err := kernels.PackActivationsInto(wantA, wantD, wantS, x64, 32); err != nil {
		t.Fatal(err)
	}

	kern, err := kernels.Quantize(k, 32)
	if err != nil {
		t.Fatal(err)
	}
	kk, err := d.Compile(kern)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kk.Close()
	bx, _ := d.Alloc(k * 4)
	ba, _ := d.Alloc(k)
	bax, _ := d.Alloc((k/32 + k/16) * 4)
	defer func() { bx.Free(); ba.Free(); bax.Free() }()
	if err := bx.Write(f32b(x)); err != nil {
		t.Fatal(err)
	}
	if err := kk.Launch((kernels.QuantizeThreads(k/32)+127)/128, 128, bx, ba, bax); err != nil {
		t.Fatalf("launch: %v", err)
	}
	gotQ := make([]byte, k)
	if err := ba.Read(gotQ); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, (k/32+k/16)*4)
	if err := bax.Read(raw); err != nil {
		t.Fatal(err)
	}
	gotAX := make([]float32, k/32+k/16)
	b2f(raw, gotAX)

	gotW := make([]uint32, k/4)
	for i := range gotW {
		gotW[i] = uint32(gotQ[4*i]) | uint32(gotQ[4*i+1])<<8 |
			uint32(gotQ[4*i+2])<<16 | uint32(gotQ[4*i+3])<<24
	}
	for i := range wantA {
		if gotW[i] != wantA[i] {
			t.Fatalf("packed[%d] = %08x, host packer says %08x", i, gotW[i], wantA[i])
		}
	}
	for i, w := range wantD {
		if math.Float32bits(gotAX[i]) != math.Float32bits(w) {
			t.Fatalf("scale[%d] = %g, host says %g", i, gotAX[i], w)
		}
	}
	for i, w := range wantS {
		if gotAX[k/32+i] != w {
			t.Fatalf("sum16[%d] = %g, host says %g", i, gotAX[k/32+i], w)
		}
	}
	t.Logf("%s: %d int8 identical to the CPU kernel", d.API(), k)
}

// TestQuantizeWideWindow is the same check at the 256-element amax window an
// all-k-quant model uses.
//
// The CPU widens the amax for a k-quant model (cpu.WideActWindow); a device
// that kept per-32 scales would fail the token-id gate for a reason unrelated
// to the device.
func TestQuantizeWideWindow(t *testing.T) {
	// Widths that leave the last group part-full, and the batched row counts
	// the tier quantizes: 2048 alone is one full group.
	for _, c := range []struct{ k, window int }{{2048, 256}, {5632, 256}, {3 * 2048, 256}, {256, 256}, {5632, 32}, {768, 256},
		// Row windows (BitNet's per-token scale): one row and several, each
		// wider than one group's tile, so Quantize loops over it.
		{2560, 2560}, {6912, 6912}, {3 * 2560, 2560}, {7 * 640, 640}, {2 * 6912, 6912}, {16 * 2560, 2560}} {
		t.Run(fmt.Sprintf("k%d_w%d", c.k, c.window), func(t *testing.T) { quantizeWideCase(t, c.k, c.window) })
	}
}

func quantizeWideCase(t *testing.T, k, window int) {
	d := dev(t)
	x := ramp(k, func(i int) float32 {
		return float32(math.Cos(float64(i)*0.23)) * float32(1+i%11)
	})
	x64 := x
	wantA := make([]uint32, k/4)
	wantD := make([]float32, k/32)
	wantS := make([]float32, k/16)
	if err := kernels.PackActivationsInto(wantA, wantD, wantS, x64, window); err != nil {
		t.Fatal(err)
	}
	kern, err := kernels.Quantize(k, window)
	if err != nil {
		t.Fatal(err)
	}
	kk, err := d.Compile(kern)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kk.Close()
	bx, _ := d.Alloc(k * 4)
	ba, _ := d.Alloc(k)
	bax, _ := d.Alloc((k/32 + k/16) * 4)
	defer func() { bx.Free(); ba.Free(); bax.Free() }()
	bx.Write(f32b(x))
	if err := kk.Launch((kernels.QuantizeThreads(k/32)+127)/128, 128, bx, ba, bax); err != nil {
		t.Fatalf("launch: %v", err)
	}
	gotQ := make([]byte, k)
	ba.Read(gotQ)
	raw := make([]byte, (k/32+k/16)*4)
	bax.Read(raw)
	gotAX := make([]float32, k/32+k/16)
	b2f(raw, gotAX)
	for i := range wantA {
		w := uint32(gotQ[4*i]) | uint32(gotQ[4*i+1])<<8 | uint32(gotQ[4*i+2])<<16 | uint32(gotQ[4*i+3])<<24
		if w != wantA[i] {
			t.Fatalf("packed[%d] = %08x, host packer says %08x", i, w, wantA[i])
		}
	}
	for i, w := range wantD {
		if math.Float32bits(gotAX[i]) != math.Float32bits(w) {
			t.Fatalf("scale[%d] = %g, host says %g", i, gotAX[i], w)
		}
	}
	for i, w := range wantS {
		if gotAX[k/32+i] != w {
			t.Fatalf("sum16[%d] = %g, host says %g", i, gotAX[k/32+i], w)
		}
	}
	// Every block inside a window must have come out with the same scale.
	for i := 1; i < len(wantD); i++ {
		if i%(window/32) != 0 && gotAX[i] != gotAX[i-1] {
			t.Fatalf("scale[%d]=%g and scale[%d]=%g differ inside one window",
				i-1, gotAX[i-1], i, gotAX[i])
		}
	}
	t.Logf("%s: %d int8 identical to the host packer at a %d-element window", d.API(), k, window)
}

func TestNormSiLUAdd(t *testing.T) {
	d := dev(t)
	const k, parts = 2048, 64
	x := ramp(k, func(i int) float32 { return float32(math.Sin(float64(i)*0.11)) * 3 })
	w := ramp(k, func(i int) float32 { return 0.5 + float32(i%5)*0.25 })
	const eps = 1e-6

	np, err := kernels.NormPart(k, parts)
	if err != nil {
		t.Fatal(err)
	}
	na, err := kernels.NormApply(k, parts, eps, false)
	if err != nil {
		t.Fatal(err)
	}
	kp, err := d.Compile(np)
	if err != nil {
		t.Fatalf("compile NormPart: %v", err)
	}
	defer kp.Close()
	ka, err := d.Compile(na)
	if err != nil {
		t.Fatalf("compile NormApply: %v", err)
	}
	defer ka.Close()

	bx, _ := d.Alloc(k * 4)
	bw, _ := d.Alloc(k * 4)
	bp, _ := d.Alloc(parts * 4)
	bo, _ := d.Alloc(k * 4)
	defer func() { bx.Free(); bw.Free(); bp.Free(); bo.Free() }()
	bx.Write(f32b(x))
	bw.Write(f32b(w))
	if err := kp.Launch((parts+127)/128, 128, bx, bp); err != nil {
		t.Fatal(err)
	}
	if err := ka.Launch((k+127)/128, 128, bx, bw, bp, bo); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, k*4)
	bo.Read(raw)
	got := make([]float32, k)
	b2f(raw, got)

	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	scale := 1 / math.Sqrt(ss/float64(k)+eps)
	for i := range got {
		want := float64(x[i]) * scale * float64(w[i])
		if math.Abs(float64(got[i])-want) > 1e-4*math.Abs(want)+1e-6 {
			t.Fatalf("norm[%d] = %g, want %g", i, got[i], want)
		}
	}

	// SiLU * up, then a residual add, in one pass over the same buffers.
	sm, err := kernels.ActMul(k, kernels.ActSiLU)
	if err != nil {
		t.Fatal(err)
	}
	ks, err := d.Compile(sm)
	if err != nil {
		t.Fatalf("compile SiLUMul: %v", err)
	}
	defer ks.Close()
	if err := ks.Launch((k+127)/128, 128, bx, bw, bo); err != nil {
		t.Fatal(err)
	}
	bo.Read(raw)
	b2f(raw, got)
	for i := range got {
		g := float64(x[i])
		want := g / (1 + math.Exp(-g)) * float64(w[i])
		if math.Abs(float64(got[i])-want) > 2e-4*math.Abs(want)+1e-6 {
			t.Fatalf("silu[%d] = %g, want %g", i, got[i], want)
		}
	}

	ad, err := kernels.Add(k)
	if err != nil {
		t.Fatal(err)
	}
	kadd, err := d.Compile(ad)
	if err != nil {
		t.Fatalf("compile Add: %v", err)
	}
	defer kadd.Close()
	badd, _ := d.Alloc(k * 4)
	defer badd.Free()
	if err := kadd.Launch((k+127)/128, 128, bx, bw, badd); err != nil {
		t.Fatal(err)
	}
	badd.Read(raw)
	got2 := make([]float32, k)
	b2f(raw, got2)
	for i := range got2 {
		if want := w[i] + x[i]; math.Abs(float64(got2[i]-want)) > 1e-5 {
			t.Fatalf("add[%d] = %g, want %g", i, got2[i], want)
		}
	}
	t.Logf("%s: norm, silu*mul and residual add all agree with the Go reference", d.API())
}
