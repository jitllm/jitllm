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
// carry: a few ulps of each term the result is a sum of. It leaves out the
// truncation error of the exp the kernel takes below -1/2, which grows as the
// sixth power of the reduced argument and so peaks at an edge this gate's
// random inputs do not reach; xieluEdgeTol adds it.
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

// xieluExpBound is the relative error of the exp the kernels take below -1/2
// (emitExpPS on every tier): a degree-5 Taylor sum on the reduced argument
// |r| <= ln2/2, whose remainder is 3.24e-6 of exp(r) at r = -ln2/2, and a few
// units in the last place for the rest. The gate above passes the kernels on
// the inputs it samples, none of which sits at that edge.
const xieluExpBound = 3.5e-6

// xieluEdges are the arguments, for eight periods of exp's range reduction
// below zero, where the reduced argument sits just inside either edge of
// [-ln2/2, ln2/2] and exp's Taylor remainder peaks.
func xieluEdges() []float32 {
	var xs []float32
	for n := -1; n >= -8; n-- {
		for _, r := range []float64{-math.Ln2 / 2 * 0.9999, math.Ln2 / 2 * 0.9999} {
			xs = append(xs, float32(float64(n)*math.Ln2+r))
		}
	}
	return xs
}

// xieluEdgeTol is xieluTol with exp's error added where the kernel takes exp:
// below -1/2 the activation carries exp(m) - 1 - m, so the term alpha_n*exp(m)
// carries xieluExpBound of itself.
func xieluEdgeTol(x float64, p [4]float32) float64 {
	t := xieluTol(x, p)
	if m := math.Min(x, float64(p[3])); m <= -0.5 {
		t += xieluExpBound * math.Abs(float64(p[1])) * math.Exp(m)
	}
	return t
}

// xieluShortExp is the activation below -1/2 with exp a Taylor term short,
// degree 4 where the kernels sum to degree 5.
func xieluShortExp(x float64, p [4]float32) float64 {
	m := math.Min(x, float64(p[3]))
	if m > -0.5 || x > 0 {
		return oracle.XIELU(x, float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3]))
	}
	n := math.Round(m / math.Ln2)
	r := m - n*math.Ln2
	e := math.Ldexp(1+r*(1+r*(0.5+r*(1.0/6+r/24))), int(n))
	return (e-1-m+(m-x))*float64(p[1]) + float64(p[2])*x
}

// TestXIELUAtTheExpEdges runs the kernels over the arguments where exp's
// remainder peaks, at the gate's widths and numbers, under the bound with
// exp's error added. Main's gate bound alone does not hold there; the log has
// how far each tier stands from it. A fault in the arithmetic, or in exp
// itself, must leave the bound.
func TestXIELUAtTheExpEdges(t *testing.T) {
	ks, done := xieluKernels(t)
	defer done()
	edges := xieluEdges()
	tiled := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = edges[i%len(edges)]
		}
		return x
	}
	for _, k := range ks {
		worst, worstOld := 0.0, 0.0
		for _, n := range xieluWidths {
			for pi, p := range xieluParams {
				x := tiled(n)
				got, guarded := xieluRun(k, x, p)
				if !guarded {
					t.Fatalf("%s n=%d: the kernel wrote past its %d elements", k.name, n, n)
				}
				for i, v := range x {
					want := oracle.XIELU(float64(v), float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3]))
					d := math.Abs(float64(got[i]) - want)
					if !(d <= xieluEdgeTol(float64(v), p)) {
						t.Fatalf("%s n=%d params %d: x[%d]=%g gives %g, want %g (|d| %.3g, bound %.3g)",
							k.name, n, pi, i, v, got[i], want, d, xieluEdgeTol(float64(v), p))
					}
					worst = math.Max(worst, d/xieluEdgeTol(float64(v), p))
					worstOld = math.Max(worstOld, d/xieluTol(float64(v), p))
				}
			}
		}
		t.Logf("%s: worst %.3f of the bound with exp's error, %.3f of the gate's bound alone", k.name, worst, worstOld)
	}
	for _, f := range []struct {
		name string
		f    func(x float64, p [4]float32) float64
	}{
		{"the alphas exchanged", func(x float64, p [4]float32) float64 {
			return oracle.XIELU(x, float64(p[1]), float64(p[0]), float64(p[2]), float64(p[3]))
		}},
		{"no beta*x", func(x float64, p [4]float32) float64 {
			return oracle.XIELU(x, float64(p[0]), float64(p[1]), 0, float64(p[3]))
		}},
		{"exp a Taylor term short", xieluShortExp},
	} {
		hits := 0
		for _, n := range xieluWidths {
			for _, p := range xieluParams {
				x := tiled(n)
				got, _ := xieluRun(ks[0], x, p)
				for i, v := range x {
					if math.Abs(float64(got[i])-f.f(float64(v), p)) > xieluEdgeTol(float64(v), p) {
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
