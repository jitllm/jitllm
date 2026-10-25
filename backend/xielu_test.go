package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestXIELUMatchesTheOracle holds kernels.XIELU -- Apertus's ungated
// activation, its four numbers a buffer -- to internal/oracle.XIELU (a float64
// expm1) on every backend present, over the regions it distinguishes: the
// polynomial's [-1/2, 0], exp below it, values near zero where expm1 - x
// cancels, the eps band, deep negatives and positives. The element count is
// ragged against the 128-thread group, and the slot past it must stay
// unwritten. A wrong form of each piece must fail the same bound.
func TestXIELUMatchesTheOracle(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const sampled = 1000
	in := make([]float32, sampled)
	for i := range in {
		f := float64(i)
		switch i % 6 {
		case 0:
			in[i] = float32(3 * math.Sin(f*0.37))
		case 1:
			in[i] = float32(-0.5 * math.Abs(math.Sin(f*0.11)))
		case 2:
			in[i] = float32(1e-4 * math.Cos(f*0.7))
		case 3:
			in[i] = float32(-0.5 - 4*math.Abs(math.Sin(f*0.23)))
		case 4:
			in[i] = float32(-20 - 100*math.Abs(math.Cos(f*0.19)))
		default:
			in[i] = float32(5 * math.Abs(math.Sin(f*0.53)))
		}
	}
	// The arguments, for eight periods of a range reduction by ln2 below zero,
	// where the reduced argument sits just inside either edge of [-ln2/2,
	// ln2/2]: where a polynomial exp's remainder peaks (jit/cpu's gate holds
	// the host kernels there). The device exp is the backend's own and is held
	// to this bound, whatever it is built on.
	for k := -1; k >= -8; k-- {
		for _, r := range []float64{-math.Ln2 / 2 * 0.9999, math.Ln2 / 2 * 0.9999} {
			in = append(in, float32(float64(k)*math.Ln2+r))
		}
	}
	n := len(in)
	params := [][4]float32{{0.8, 0.95, 0.5, -1e-6}, {1.31, 0.42, 0.37, -0.25}}
	tol := func(x float64, p [4]float32) float64 {
		ap, an, b, e := float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3])
		m := math.Min(x, e)
		// The device exp is the backend's own, a few ulps.
		return 2e-6*(math.Abs(ap*x*x)+math.Abs(b*x)+math.Abs(an)*(math.Abs(math.Expm1(m)-m)+math.Abs(m-x)+math.Abs(m))) + 1e-30
	}
	k, err := kernels.XIELU(n)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			kern, err := d.Compile(k)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer kern.Close()
			bIn := g.up(f32bytes(in))
			const guard = 0x7FC0DEAD
			for _, p := range params {
				outInit := make([]byte, (n+1)*4)
				binary.LittleEndian.PutUint32(outInit[n*4:], guard)
				bOut := g.up(outInit)
				bP := g.up(f32bytes(p[:]))
				if err := kern.Launch((n+127)/128, 128, bIn, bP, bOut); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, (n+1)*4)
				if err := bOut.Read(raw); err != nil {
					t.Fatal(err)
				}
				if binary.LittleEndian.Uint32(raw[n*4:]) != guard {
					t.Fatalf("the kernel wrote past its %d elements", n)
				}
				got := func(i int) float64 {
					return float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
				}
				worst, worstEdge := 0.0, 0.0
				for i, v := range in {
					x := float64(v)
					want := oracle.XIELU(x, float64(p[0]), float64(p[1]), float64(p[2]), float64(p[3]))
					d := math.Abs(got(i) - want)
					if !(d <= tol(x, p)) {
						t.Fatalf("params %v: x[%d]=%g gives %g, want %g", p, i, v, got(i), want)
					}
					if i < sampled {
						worst = max(worst, d/tol(x, p))
					} else {
						worstEdge = max(worstEdge, d/tol(x, p))
					}
				}
				// Each wrong form must leave the bound on some input.
				for _, f := range []struct {
					name string
					w    func(x float64) float64
				}{
					{"eps ignored", func(x float64) float64 {
						return oracle.XIELU(x, float64(p[0]), float64(p[1]), float64(p[2]), math.Inf(1))
					}},
					{"the alphas exchanged", func(x float64) float64 {
						return oracle.XIELU(x, float64(p[1]), float64(p[0]), float64(p[2]), float64(p[3]))
					}},
					{"no beta*x", func(x float64) float64 {
						return oracle.XIELU(x, float64(p[0]), float64(p[1]), 0, float64(p[3]))
					}},
				} {
					hits := 0
					for i, v := range in {
						if math.Abs(got(i)-f.w(float64(v))) > tol(float64(v), p) {
							hits++
						}
					}
					if hits == 0 && f.name != "eps ignored" || hits == 0 && p[3] < -1e-3 {
						t.Errorf("params %v: %s: the gate cannot see it", p, f.name)
					}
				}
				t.Logf("params %v: %d elements within bound, worst at %.2f of it on the %d sampled and %.2f on the %d at exp's edges", p, n, worst, sampled, worstEdge, n-sampled)
			}
		})
	}
}
