package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestGemmVoltaFloatWeights holds GemmVolta over two-byte float weights
// (bfloat16 and binary16, row-major as MatVec reads them) to a float64
// reference: the tile families and staging nestings of the packed formats'
// gate, a k-split with a bias, and k staged one and two sub-blocks a trip.
// The weights are staged as binary16 and the activations are ActF16T's, so
// the bound is binary16 rounding of both operands, not the reference's
// precision. Every element is written (a NaN-filled output) and every
// format's layout is read where MatVec puts it: a transposed or word-major
// read fails the bound by orders of magnitude.
func TestGemmVoltaFloatWeights(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.BFloat16, kernels.Float16} {
			for _, sh := range []struct {
				rows, k, ntok, split int
				tl                   kernels.VoltaTile
				bias                 bool
			}{
				{128, 512, 128, 1, kernels.VoltaTile{MT: 2, NT: 8, WM: 2, WN: 2, KB: 1}, false},
				{256, 512, 128, 2, kernels.VoltaTile{MT: 2, NT: 8, WM: 2, WN: 2, KB: 1}, true},
				{128, 1024, 64, 1, kernels.VoltaTile{MT: 4, NT: 4, WM: 1, WN: 2, KB: 2}, true},
				{64, 512, 64, 1, kernels.VoltaTile{MT: 1, NT: 2, WM: 2, WN: 2, KB: 2}, false},
				{128, 2624, 64, 1, kernels.VoltaTile{MT: 4, NT: 2, WM: 1, WN: 4, KB: 1}, false},
				{192, 512, 128, 2, kernels.VoltaTile{MT: 3, NT: 4, WM: 1, WN: 4, KB: 1}, true},
			} {
				s := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, Split: sh.split, Bias: sh.bias}
				kk, err := kernels.GemmVolta(s, sh.tl)
				if err != nil {
					t.Fatalf("%v %+v: %v", q, sh, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					if d.API() == "ptx" {
						t.Fatalf("%s %v %+v: %v", d.API(), q, sh, err)
					}
					t.Logf("%s: %v", d.API(), err)
					break
				}
				ran++
				floatGemmCase(t, d, kern, s, kernels.GemmVoltaGroups(s, sh.tl), sh.tl.Threads(), fmt.Sprintf("%+v", sh.tl))
				kern.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

func floatGemmCase(t *testing.T, d backend.Device, kern backend.Kernel, s kernels.MatVecShape, groups, width int, what string) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(s.Rows*7 + s.K + s.NTok)))
	g := newGPU(t, d)
	defer g.free()
	// Row-major weights of a trained projection's scale, rounded to the
	// format, so the reference reads exactly what the device holds.
	w := make([]float64, s.Rows*s.K)
	raw := make([]byte, 2*len(w))
	for i := range w {
		v := float32(rng.NormFloat64() * 0.05)
		var h uint16
		if s.T == kernels.BFloat16 {
			h = uint16(math.Float32bits(v) >> 16)
			w[i] = float64(math.Float32frombits(uint32(h) << 16))
		} else {
			h = f16round(v)
			w[i] = f16value(h)
		}
		binary.LittleEndian.PutUint16(raw[2*i:], h)
	}
	x := make([]float32, s.NTok*s.K)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	bias := make([]float32, s.Rows)
	for i := range bias {
		bias[i] = float32(rng.NormFloat64())
	}
	n := s.Rows * s.NTok
	ref := make([]float64, n)
	for tk := 0; tk < s.NTok; tk++ {
		for r := 0; r < s.Rows; r++ {
			var acc float64
			if s.Bias {
				acc = float64(bias[r])
			}
			for k := 0; k < s.K; k++ {
				acc += w[r*s.K+k] * float64(x[tk*s.K+k])
			}
			ref[tk*s.Rows+r] = acc
		}
	}
	ak, err := kernels.ActF16T(s.T, s.NTok, s.K)
	if err != nil {
		t.Fatal(err)
	}
	ac, err := d.Compile(ak)
	if err != nil {
		t.Fatal(err)
	}
	defer ac.Close()
	bX := g.up(f32bytes(x))
	bB := g.up(make([]byte, s.NTok*s.K*2))
	at := s.NTok * s.K / 32
	if err := ac.Launch((at+127)/128, 128, bX, bB); err != nil {
		t.Fatal(err)
	}
	split := max(s.Split, 1)
	got := g.up(f32bytes(nanFill(split * n)))
	// pD and pSC are not read for a float weight; the launch still names them.
	args := []backend.Buf{g.up(raw), g.up(make([]byte, 4)), g.up(make([]byte, 4)), bB, got}
	if s.Bias {
		args = append(args, g.up(f32bytes(bias)))
	}
	if err := kern.Launch(groups, width, args...); err != nil {
		t.Fatal(err)
	}
	parts := make([]byte, split*n*4)
	if err := got.Read(parts); err != nil {
		t.Fatal(err)
	}
	var se, sy float64
	for i := 0; i < n; i++ {
		var v float64
		for sg := 0; sg < split; sg++ {
			v += float64(math.Float32frombits(binary.LittleEndian.Uint32(parts[4*(sg*n+i):])))
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s %v %s: element %d is %v -- never written", d.API(), s.T, what, i, v)
		}
		se += (v - ref[i]) * (v - ref[i])
		sy += ref[i] * ref[i]
	}
	nm := se / sy
	name := fmt.Sprintf("%s gemmvolta %v %dx%d ntok %d %s split %d bias %v", d.API(), s.T, s.Rows, s.K, s.NTok, what, split, s.Bias)
	t.Logf("%s: NMSE %.3e", name, nm)
	if nm > 1e-5 {
		t.Fatalf("%s: NMSE %.3e against the reference", name, nm)
	}
}

// f16round rounds v to binary16 (round to nearest even; the test's values are
// normal and in range).
func f16round(v float32) uint16 {
	b := math.Float32bits(v)
	sign := uint16(b>>16) & 0x8000
	exp := int((b>>23)&0xff) - 127 + 15
	man := b & 0x7fffff
	if exp <= 0 {
		return sign // the test's weights never reach binary16's subnormals
	}
	h := uint32(exp)<<10 | man>>13
	rem := man & 0x1fff
	if rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++
	}
	return sign | uint16(h)
}

func f16value(h uint16) float64 {
	e := int(h>>10) & 0x1f
	m := float64(h & 0x3ff)
	v := 0.0
	if e != 0 {
		v = math.Ldexp(1+m/1024, e-15)
	}
	if h&0x8000 != 0 {
		v = -v
	}
	return v
}
