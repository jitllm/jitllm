package kernels

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/backend/vulkan"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestDivRNIsTheIEEEQuotient holds divRN to Go's float32 division, which is
// IEEE round-to-nearest-even, BIT FOR BIT on every device: quotients across the
// whole normal range, the quantizer's own two (amax/127 and 1/d), denominators
// whose mantissa is a power of two or all ones, and quotients a few ulps
// either side of a power of two, where the candidates cross a binade. It
// reports how many of the backend's own divisions differ, which is what the
// integer rounding exists for: zero on PTX's div.rn, thousands on a Vulkan
// driver. JITLLM_VK_DEVICE pins the Vulkan device, as in the backend suite.
func TestDivRNIsTheIEEEQuotient(t *testing.T) {
	devs := backend.OpenWith(backend.Opts{Vulkan: vulkan.Config{Device: os.Getenv("JITLLM_VK_DEVICE")}})
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	num, den := divRNCases(rand.New(rand.NewSource(7)), 1<<20)
	n := len(num)
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			got, hw := divRNRun(t, d, num, den), divRNRun(t, d, num, den, true)
			bad, hwBad, first := 0, 0, -1
			for i := range num {
				want := math.Float32bits(num[i] / den[i])
				if got[i] != want {
					if bad++; first < 0 {
						first = i
					}
				}
				if hw[i] != want {
					hwBad++
				}
			}
			if bad > 0 {
				t.Fatalf("%s: %d of %d quotients are not the IEEE one (first %g/%g: got %g, want %g)", d.API(), bad,
					n, num[first], den[first], math.Float32frombits(got[first]), num[first]/den[first])
			}
			t.Logf("%s: %d quotients bit for bit; the backend's own division differs in %d", d.API(), n, hwBad)
		})
	}
}

// divRNCases is n (num, den) pairs, positive and normal, with normal
// quotients.
func divRNCases(rng *rand.Rand, n int) (num, den []float32) {
	logu := func(lo, hi float64) float32 { return float32(math.Exp(lo + rng.Float64()*(hi-lo))) }
	ulps := func(x float32, k int) float32 {
		return math.Float32frombits(uint32(int64(math.Float32bits(x)) + int64(k)))
	}
	for len(num) < n {
		var a, b float32
		switch len(num) % 6 {
		case 0: // anywhere in the normal range
			a, b = logu(-40, 40), logu(-40, 40)
		case 1: // the quantizer's scale
			a, b = logu(-20, 20), 127
		case 2: // and its reciprocal, of a scale the quantizer made
			a, b = 1, logu(-20, 20)/127
		case 3: // a power-of-two denominator mantissa, or an all-ones one
			b = math.Float32frombits(uint32(rng.Intn(200)+20)<<23 | uint32(rng.Intn(2))*0x7FFFFF)
			a = logu(-20, 20)
		case 4: // quotients a few ulps either side of a power of two
			b = logu(-20, 20)
			p := float32(math.Ldexp(1, rng.Intn(40)-20))
			a = ulps(p*b, rng.Intn(9)-4)
		default: // numerator and denominator a few ulps apart
			b = logu(-20, 20)
			a = ulps(b, rng.Intn(9)-4)
		}
		q := a / b
		if e := math.Float32bits(q) >> 23; e < 3 || e > 252 {
			continue
		}
		num, den = append(num, a), append(den, b)
	}
	return num, den
}

// divRNRun runs divRN (or, with hw, the backend's own division) over the
// pairs on d and returns the quotients' bits.
func divRNRun(t *testing.T, d backend.Device, num, den []float32, hw ...bool) []uint32 {
	t.Helper()
	n := len(num)
	b := ir.New("divrn", [3]int{128, 1, 1})
	pN, pD, pO := b.Param("pN", ir.F32), b.Param("pD", ir.F32), b.Param("pO", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(n-1)))
	x, y := b.Load(ir.F32, pN, i, 0), b.Load(ir.F32, pD, i, 0)
	q := divRN(b, x, y)
	if len(hw) > 0 {
		q = b.Div(ir.F32, x, y)
	}
	b.Store(pO, i, q, 0)
	built, err := b.Done()
	if err != nil {
		t.Fatal(err)
	}
	k, err := d.Compile(built)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()
	bytes := func(v []float32) []byte {
		out := make([]byte, 4*len(v))
		for j, f := range v {
			u := math.Float32bits(f)
			out[4*j], out[4*j+1], out[4*j+2], out[4*j+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
		}
		return out
	}
	var bufs []backend.Buf
	for _, p := range [][]byte{bytes(num), bytes(den), make([]byte, 4*n)} {
		buf, err := d.Alloc(len(p))
		if err != nil {
			t.Fatal(err)
		}
		defer buf.Free()
		if err := buf.Write(p); err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, buf)
	}
	if err := k.Launch((n+127)/128, 128, bufs...); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 4*n)
	if err := bufs[2].Read(raw); err != nil {
		t.Fatal(err)
	}
	out := make([]uint32, n)
	for j := range out {
		out[j] = uint32(raw[4*j]) | uint32(raw[4*j+1])<<8 | uint32(raw[4*j+2])<<16 | uint32(raw[4*j+3])<<24
	}
	return out
}

// TestQuantScalesAreTheHosts holds quantScales to the host quantizer's d =
// amax/127 and inv = 1/d, BIT FOR BIT on every device, over every mantissa of
// one binade (div127 is exact or not per mantissa: the exponent only moves the
// result), amaxes spanning the range an activation reaches, and a zero amax,
// whose inv is 0.
func TestQuantScalesAreTheHosts(t *testing.T) {
	devs := backend.OpenWith(backend.Opts{Vulkan: vulkan.Config{Device: os.Getenv("JITLLM_VK_DEVICE")}})
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(11))
	const binade = 1 << 23
	n := binade + 1<<20
	amax := make([]float32, n)
	for i := range amax {
		if i < binade {
			amax[i] = math.Float32frombits(127<<23 | uint32(i))
		} else {
			amax[i] = float32(math.Exp(rng.Float64()*60 - 30))
		}
	}
	amax[n-1] = 0
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			got := quantScalesRun(t, d, amax)
			bad := 0
			for i, a := range amax {
				wd := a / 127
				wi := float32(0)
				if wd != 0 {
					wi = 1 / wd
				}
				if got[2*i] != math.Float32bits(wd) || got[2*i+1] != math.Float32bits(wi) {
					if bad++; bad == 1 {
						t.Errorf("amax %g: d %g inv %g, the host's %g and %g", a, math.Float32frombits(got[2*i]),
							math.Float32frombits(got[2*i+1]), wd, wi)
					}
				}
			}
			if bad > 0 {
				t.Fatalf("%s: %d of %d (d, inv) pairs are not the host's", d.API(), bad, n)
			}
			t.Logf("%s: %d (d, inv) pairs bit for bit", d.API(), n)
		})
	}
}

func quantScalesRun(t *testing.T, d backend.Device, amax []float32) []uint32 {
	t.Helper()
	n := len(amax)
	b := ir.New("quantscales", [3]int{128, 1, 1})
	pA, pO := b.Param("pA", ir.F32), b.Param("pO", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(n-1)))
	dd, inv := quantScales(b, b.Load(ir.F32, pA, i, 0))
	i2 := b.Mul(ir.U32, i, b.Const(ir.U32, 2))
	b.Store(pO, i2, dd, 0)
	b.Store(pO, i2, inv, 1)
	built, err := b.Done()
	if err != nil {
		t.Fatal(err)
	}
	k, err := d.Compile(built)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()
	in := make([]byte, 4*n)
	for j, f := range amax {
		u := math.Float32bits(f)
		in[4*j], in[4*j+1], in[4*j+2], in[4*j+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	ba, err := d.Alloc(4 * n)
	if err != nil {
		t.Fatal(err)
	}
	defer ba.Free()
	bo, err := d.Alloc(8 * n)
	if err != nil {
		t.Fatal(err)
	}
	defer bo.Free()
	if err := ba.Write(in); err != nil {
		t.Fatal(err)
	}
	if err := k.Launch((n+127)/128, 128, ba, bo); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 8*n)
	if err := bo.Read(raw); err != nil {
		t.Fatal(err)
	}
	out := make([]uint32, 2*n)
	for j := range out {
		out[j] = uint32(raw[4*j]) | uint32(raw[4*j+1])<<8 | uint32(raw[4*j+2])<<16 | uint32(raw[4*j+3])<<24
	}
	return out
}
