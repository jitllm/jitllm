package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestMMAFragmentLayout pins the warp-level matrix instruction's fragment
// mapping against a host reference, on real hardware.
//
// Which element a lane holds is fixed by the hardware; a transposed reading of
// the table assembles, runs and produces a wrong matrix. Every MMA kernel reuses
// MMAProbe's index expressions, so this test is what makes them trustworthy.
//
// The first case is a ramp (A[m][k] = m*K+k mod 127), on which a transposition
// of the output is obvious.
func TestMMAFragmentLayout(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	// Both kinds, intersected: this sweep runs s8 and f16 tiles, so a backend
	// lowering only one would otherwise skip inside the shape loop.
	s8 := mmaDevices(t, devs, ir.MMAShape{M: 16, N: 8, K: 16})
	f16 := mmaDevices(t, devs, ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16})
	both := make([]backend.Device, 0, len(s8))
	for _, a := range s8 {
		for _, b := range f16 {
			if a == b {
				both = append(both, a)
			}
		}
	}
	for _, d := range both {
		t.Run(d.API(), func(t *testing.T) {
			for _, sh := range []ir.MMAShape{
				{M: 16, N: 8, K: 16}, {M: 16, N: 8, K: 32},
				{M: 16, N: 8, K: 8, Kind: ir.MMAF16}, {M: 16, N: 8, K: 16, Kind: ir.MMAF16},
			} {
				name := "s8/m16n8k" + itoa(sh.K)
				if sh.Kind == ir.MMAF16 {
					name = "f16/m16n8k" + itoa(sh.K)
				}
				t.Run(name, func(t *testing.T) {
					k, err := kernels.MMAProbe(sh)
					if err != nil {
						t.Fatal(err)
					}
					kern, err := d.Compile(k)
					if err != nil {
						// A decline is expected on Metal (no integer matrix
						// instruction) and Vulkan (an opaque fragment layout);
						// both refuse to lower OpMMA, and the skip names why.
						t.Skipf("declined: %v", err)
					}
					defer kern.Close()
					for _, ramp := range []bool{true, false} {
						if sh.Kind == ir.MMAF16 {
							mmaCaseF16(t, d, kern, sh, ramp)
							continue
						}
						mmaCase(t, d, kern, sh, ramp)
					}
				})
			}
		})
	}
}

func mmaCase(t *testing.T, d backend.Device, kern backend.Kernel, sh ir.MMAShape, ramp bool) {
	t.Helper()
	rng := rand.New(rand.NewSource(7))
	a := make([]int8, sh.M*sh.K)  // row-major [m][k]
	bm := make([]int8, sh.K*sh.N) // column-major: column n is contiguous
	for i := range a {
		if ramp {
			a[i] = int8(i % 127)
		} else {
			a[i] = int8(rng.Intn(255) - 127)
		}
	}
	for i := range bm {
		if ramp {
			bm[i] = int8((i * 5) % 127)
		} else {
			bm[i] = int8(rng.Intn(255) - 127)
		}
	}
	want := make([]int32, sh.M*sh.N)
	for m := 0; m < sh.M; m++ {
		for n := 0; n < sh.N; n++ {
			var acc int32
			for kk := 0; kk < sh.K; kk++ {
				acc += int32(a[m*sh.K+kk]) * int32(bm[n*sh.K+kk])
			}
			want[m*sh.N+n] = acc
		}
	}

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(p []byte) backend.Buf {
		b, err := d.Alloc(len(p))
		if err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, b)
		if err := b.Write(p); err != nil {
			t.Fatal(err)
		}
		return b
	}
	i8b := func(v []int8) []byte {
		out := make([]byte, len(v))
		for i, x := range v {
			out[i] = byte(x)
		}
		return out
	}
	bA, bB := up(i8b(a)), up(i8b(bm))
	bOut := up(make([]byte, sh.M*sh.N*4))
	var err error
	d.Session(func(s backend.Session) {
		if err = s.Launch(kern, 1, 32, bA, bB, bOut); err != nil {
			return
		}
		err = s.Sync()
	})
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, sh.M*sh.N*4)
	d.Session(func(s backend.Session) { err = s.Read(bOut, raw) })
	if err != nil {
		t.Fatal(err)
	}
	// A degenerate reference proves nothing: an all-zero want would pass
	// against an all-zero device buffer.
	nz := 0
	for _, w := range want {
		if w != 0 {
			nz++
		}
	}
	if nz < len(want)/2 {
		t.Fatalf("ramp=%v: only %d/%d reference elements are non-zero; the oracle is degenerate",
			ramp, nz, len(want))
	}
	bad := 0
	for i := range want {
		got := int32(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 | uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
		if got != want[i] {
			if bad < 6 {
				t.Errorf("ramp=%v D[%d][%d] = %d, want %d", ramp, i/sh.N, i%sh.N, got, want[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("ramp=%v: %d/%d elements wrong -- the fragment mapping is not what the kernel assumes",
			ramp, bad, len(want))
	}
}

// mmaCaseF16 is mmaCase for the binary16 shape. The reference rounds each
// operand to binary16 before multiplying, as the hardware does, so the
// tolerance need not be wide enough to hide a transposition.
func mmaCaseF16(t *testing.T, d backend.Device, kern backend.Kernel, sh ir.MMAShape, ramp bool) {
	t.Helper()
	rng := rand.New(rand.NewSource(11))
	a := make([]float32, sh.M*sh.K)
	bm := make([]float32, sh.K*sh.N)
	for i := range a {
		if ramp {
			a[i] = float32(i%37) - 18
		} else {
			a[i] = float32(rng.NormFloat64())
		}
	}
	for i := range bm {
		if ramp {
			bm[i] = float32((i*5)%29) - 14
		} else {
			bm[i] = float32(rng.NormFloat64())
		}
	}
	// Round to nearest even, as PackF16's cvt.rn.f16.f32 does.
	h := func(v float32) float32 { return float32(quant.DecodeHalf(f32ToF16(v))) }
	want := make([]float32, sh.M*sh.N)
	for m := 0; m < sh.M; m++ {
		for n := 0; n < sh.N; n++ {
			var acc float32
			for kk := 0; kk < sh.K; kk++ {
				acc += h(a[m*sh.K+kk]) * h(bm[n*sh.K+kk])
			}
			want[m*sh.N+n] = acc
		}
	}
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(v []float32) backend.Buf {
		b, err := d.Alloc(len(v) * 4)
		if err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, b)
		if err := b.Write(f32bytes(v)); err != nil {
			t.Fatal(err)
		}
		return b
	}
	bA, bB := up(a), up(bm)
	bOut := up(make([]float32, sh.M*sh.N))
	var err error
	d.Session(func(s backend.Session) {
		if err = s.Launch(kern, 1, 32, bA, bB, bOut); err != nil {
			return
		}
		err = s.Sync()
	})
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, sh.M*sh.N*4)
	d.Session(func(s backend.Session) { err = s.Read(bOut, raw) })
	if err != nil {
		t.Fatal(err)
	}
	var sse, sy2 float64
	nz, bad := 0, 0
	for i := range want {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if want[i] != 0 {
			nz++
		}
		e := float64(got) - float64(want[i])
		sse += e * e
		sy2 += float64(want[i]) * float64(want[i])
		if math.Abs(e) > 1e-2*math.Abs(float64(want[i]))+1e-3 {
			if bad < 5 {
				t.Errorf("ramp=%v D[%d][%d] = %v, want %v", ramp, i/sh.N, i%sh.N, got, want[i])
			}
			bad++
		}
	}
	if nz < len(want)/2 {
		t.Fatalf("ramp=%v: only %d/%d reference values are non-zero; the oracle is degenerate", ramp, nz, len(want))
	}
	// The accumulator is float32 and the operands are exactly representable
	// after rounding, so this is a summation-order tolerance, not a precision
	// one: a transposed fragment lands orders of magnitude outside it.
	if nmse := sse / sy2; nmse > 1e-6 || math.IsNaN(nmse) || bad > 0 {
		t.Fatalf("ramp=%v: NMSE %.3e, %d/%d elements outside tolerance -- the fragment mapping is wrong",
			ramp, nmse, bad, len(want))
	}
}

// f32ToF16 rounds to binary16, nearest-even, the way cvt.rn.f16.f32 does.
func f32ToF16(v float32) uint16 {
	b := math.Float32bits(v)
	sign := uint16(b >> 16 & 0x8000)
	e := int32(b>>23&0xFF) - 127 + 15
	m := b & 0x7FFFFF
	switch {
	case e >= 0x1F:
		return sign | 0x7C00 // overflow to infinity
	case e <= 0:
		if e < -10 {
			return sign
		}
		m |= 0x800000
		sh := uint32(14 - e)
		half := uint32(1) << (sh - 1)
		r := (m + half - 1 + (m>>sh)&1) >> sh
		return sign | uint16(r)
	}
	half := uint32(0x1000)
	r := m + half - 1 + (m>>13)&1
	if r&0x800000 != 0 {
		e++
		r = 0
		if e >= 0x1F {
			return sign | 0x7C00
		}
	}
	return sign | uint16(e)<<10 | uint16(r>>13)
}
