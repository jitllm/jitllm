//go:build amd64 || arm64

package cpu

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The xIELU kernel (Apertus's ungated activation) on every tier this host can
// run -- AVX2 and SSE on amd64, NEON on arm64 -- held to internal/oracle.XIELU,
// a float64 expm1, at ragged widths with a guard after the output.

// xieluParams are the effective four numbers (alpha_p, alpha_n, beta, eps):
// Apertus's shape, an eps wide enough that min(x, eps) clamps a visible band of
// inputs, and one whose alphas are far apart so a swap is loud.
var xieluParams = [][4]float32{
	{0.8, 0.95, 0.5, -1e-6},
	{1.31, 0.42, 0.37, -0.25},
	{0.05, 2.7, 0.9, -0.0625},
}

var xieluWidths = []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100}

// xieluInput mixes the regions the kernel distinguishes: the polynomial's
// [-1/2, 0], exp below it, values near zero where expm1 - x cancels, the eps
// band, deep negatives where exp underflows, and positives.
func xieluInput(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		switch i % 6 {
		case 0:
			x[i] = float32(rng.NormFloat64() * 3)
		case 1:
			x[i] = -float32(rng.Float64()) * 0.5
		case 2:
			x[i] = float32(rng.NormFloat64() * 1e-4)
		case 3:
			x[i] = -0.5 - float32(rng.Float64())*4
		case 4:
			x[i] = -20 - float32(rng.Float64())*100
		default:
			x[i] = float32(rng.Float64()) * 5
		}
	}
	return x
}

type xieluKernel struct {
	name string
	code *Code
}

func xieluKernels(t *testing.T) ([]xieluKernel, func()) {
	t.Helper()
	tiers := []Tier{HostTier()}
	if runtime.GOARCH == "amd64" && HostTier() != TierSSE {
		tiers = append(tiers, TierSSE)
	}
	var out []xieluKernel
	for _, tr := range tiers {
		b, err := EmittersFor(tr).XIELU()
		if err != nil {
			t.Fatalf("%v XIELU: %v", tr, err)
		}
		if tr == TierSSE && KernelTier(b) != TierSSE {
			t.Fatalf("the SSE xIELU declares tier %v", KernelTier(b))
		}
		xieluVEXGate(t, fmt.Sprint(tr), b)
		c, err := MapNamed(b, "xielu")
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, xieluKernel{fmt.Sprint(tr), c})
	}
	return out, func() {
		for _, k := range out {
			k.code.Close()
		}
	}
}

// xieluRun runs k over x in place, with guard words after it.
func xieluRun(k xieluKernel, x []float32, p [4]float32) ([]float32, bool) {
	const guard = 8
	buf := make([]float32, len(x)+guard)
	copy(buf, x)
	sentinel := math.Float32frombits(0x7FC0DEAD)
	for i := len(x); i < len(buf); i++ {
		buf[i] = sentinel
	}
	consts := XIELUConsts()
	k.code.Call(&Args{Out: &buf[0], AScale: &p[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
		K: int64(len(x) / ElemLanes), Rows: int64(len(x) % ElemLanes)})
	ok := true
	for i := len(x); i < len(buf); i++ {
		ok = ok && math.Float32bits(buf[i]) == 0x7FC0DEAD
	}
	return buf[:len(x)], ok
}

// xieluTol is the error a correctly rounded f32 evaluation of the terms can
// carry: a few ulps of each term the result is a sum of.
func xieluTol(x float64, p [4]float32) float64 {
	ap, an, b, e := float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3])
	m := math.Min(x, e)
	return 4e-7*(math.Abs(ap*x*x)+math.Abs(b*x)+math.Abs(an)*(math.Abs(math.Expm1(m)-m)+math.Abs(m-x)+math.Abs(m))) + 1e-30
}

func TestXIELUMatchesTheOracle(t *testing.T) {
	ks, done := xieluKernels(t)
	defer done()
	ran := 0
	for _, k := range ks {
		for _, n := range xieluWidths {
			for pi, p := range xieluParams {
				x := xieluInput(n, int64(n*31+pi))
				got, guarded := xieluRun(k, x, p)
				if !guarded {
					t.Fatalf("%s n=%d: the kernel wrote past its %d elements", k.name, n, n)
				}
				for i, v := range x {
					want := oracle.XIELU(float64(v), float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3]))
					if d := math.Abs(float64(got[i]) - want); !(d <= xieluTol(float64(v), p)) {
						t.Fatalf("%s n=%d params %d: x[%d]=%g gives %g, want %g (|d| %.3g)",
							k.name, n, pi, i, v, got[i], want, d)
					}
				}
				ran++
			}
		}
	}
	if ran == 0 {
		t.Fatal("no kernel ran -- this gate proved nothing")
	}
	t.Logf("%d tier(s), %d (width, params) cases", len(ks), ran)
}

// TestXIELUViolations runs the gate against the faults it exists to catch: a
// wrong arithmetic form must disagree with the kernel on some input, or the
// inputs above could not tell the two apart.
func TestXIELUViolations(t *testing.T) {
	ks, done := xieluKernels(t)
	defer done()
	faults := []struct {
		name string
		f    func(x float64, p [4]float32) float64
	}{
		{"eps ignored", func(x float64, p [4]float32) float64 {
			return oracle.XIELU(x, float64(p[0]), float64(p[1]), float64(p[2]), math.Inf(1))
		}},
		{"the alphas exchanged", func(x float64, p [4]float32) float64 {
			return oracle.XIELU(x, float64(p[1]), float64(p[0]), float64(p[2]), float64(p[3]))
		}},
		{"no beta*x", func(x float64, p [4]float32) float64 {
			return oracle.XIELU(x, float64(p[0]), float64(p[1]), 0, float64(p[3]))
		}},
		// exp(m) - 1 - m in f32 near zero: what the polynomial exists to avoid.
		{"expm1 by exp - 1 in f32", func(x float64, p [4]float32) float64 {
			if x > 0 {
				return oracle.XIELU(x, float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3]))
			}
			m := float32(math.Min(x, float64(p[3])))
			g := float32(math.Exp(float64(m))) - 1 - m
			return float64((g+(m-float32(x)))*p[1] + p[2]*float32(x))
		}},
	}
	for _, f := range faults {
		hits := 0
		for _, n := range xieluWidths {
			for pi, p := range xieluParams {
				x := xieluInput(n, int64(n*31+pi))
				got, _ := xieluRun(ks[0], x, p)
				for i, v := range x {
					if d := math.Abs(float64(got[i]) - f.f(float64(v), p)); d > xieluTol(float64(v), p) {
						hits++
					}
				}
			}
		}
		if hits == 0 {
			t.Errorf("%s: the gate cannot see it", f.name)
		}
		t.Logf("%s caught on %d elements", f.name, hits)
	}
}
