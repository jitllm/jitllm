//go:build amd64 && jitllmtest

package nn

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/cpu"
)

// TestElementwiseEntryPointsComputeOnTheSSETier runs every package-level
// elementwise entry point -- the ones every token of every model calls:
// Softmax32JIT, ActMul32JIT, Act32JIT, SigmoidMul32JIT, Axpy32JIT, Scale32JIT,
// Softcap32JIT -- under the forced SSE tier, and holds each to three things:
//
//   - the float64 definition (internal/oracle where it has one), at a ragged
//     width with a guard after it;
//   - bit-identity with the SSE table's own kernel called directly with the
//     same Args, which says the entry point ran that kernel;
//   - on an AVX2 host, a difference from what the same entry point computed
//     before the force somewhere across the ops -- the SSE bodies round
//     differently (no FMA), so an entry point that kept a cached AVX2 kernel
//     would reproduce the pre-force bits and fail here.
func TestElementwiseEntryPointsComputeOnTheSSETier(t *testing.T) {
	const n = 1027 // 128 whole units and a tail of three
	const guard = 2 * cpu.ElemLanes
	const guardBits = 0x7fc0beef
	in := func(seed float64) []float32 {
		v := make([]float32, n+guard)
		for i := 0; i < n; i++ {
			v[i] = float32(math.Sin(seed+float64(i)*0.37) * float64(1+i%13))
		}
		for i := n; i < len(v); i++ {
			v[i] = math.Float32frombits(guardBits)
		}
		return v
	}
	x, y := in(1), in(2)
	sigmoid := func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }
	quick := func(v float64) float64 { return v * sigmoid(1.702*v) }
	type op struct {
		name  string
		run   func(dst []float32)    // through the nn entry point
		emit  func() ([]byte, error) // the SSE table's kernel for it
		args  func(dst []float32) cpu.Args
		ref   func(x, y float64) float64
		bound float64
	}
	consts := cpu.ActConsts()
	scr := (*byte)(unsafe.Pointer(&consts[0]))
	alpha := float32(0.37)
	capC := float32(3)
	kc := [2]float32{2 / capC, capC}
	clampC := float32(1.25)
	clampLH := [2]float32{-clampC, clampC}
	sse := cpu.EmittersFor(cpu.TierSSE)
	elem := func(dst []float32, withY bool, s *byte) cpu.Args {
		a := cpu.Args{Out: &dst[0], Scr: s, K: n / cpu.ElemLanes, Rows: n % cpu.ElemLanes}
		if withY {
			a.AScale = &y[0]
		}
		return a
	}
	ops := []op{
		{"softmax", func(d []float32) { Softmax32JIT(d, n) }, sse.Softmax,
			func(d []float32) cpu.Args { return elem(d, false, scr) }, nil, 1e-10},
		{"axpy", func(d []float32) { Axpy32JIT(d[:n], y, alpha) }, sse.Axpy,
			func(d []float32) cpu.Args { return elem(d, true, (*byte)(unsafe.Pointer(&alpha))) },
			func(a, b float64) float64 { return a + float64(alpha)*b }, 1e-12},
		{"scale", func(d []float32) { Scale32JIT(d[:n], alpha) }, sse.Scale,
			func(d []float32) cpu.Args { return elem(d, false, (*byte)(unsafe.Pointer(&alpha))) },
			func(a, _ float64) float64 { return a * float64(alpha) }, 1e-12},
		{"softcap", func(d []float32) { Softcap32JIT(d[:n], capC) }, sse.Softcap,
			func(d []float32) cpu.Args {
				a := elem(d, false, scr)
				a.W = (*byte)(unsafe.Pointer(&kc[0]))
				return a
			},
			func(a, _ float64) float64 { return float64(capC) * math.Tanh(a/float64(capC)) }, 1e-9},
		{"sigmoidmul", func(d []float32) { SigmoidMul32JIT(d[:n], y) }, sse.SigmoidMul,
			func(d []float32) cpu.Args { return elem(d, true, scr) },
			func(a, b float64) float64 { return sigmoid(a) * b }, 1e-9},
		{"clamp", func(d []float32) { Clamp32JIT(d[:n], clampC) }, sse.Clamp,
			func(d []float32) cpu.Args { return elem(d, false, (*byte)(unsafe.Pointer(&clampLH[0]))) },
			func(a, _ float64) float64 { return math.Min(math.Max(a, -float64(clampC)), float64(clampC)) }, 0},
	}
	gatedRef := map[ActKind]func(a, b float64) float64{
		ActSiLU:        func(a, b float64) float64 { return oracle.SiLU(a) * b },
		ActGELU:        func(a, b float64) float64 { return oracle.GELUTanh(a) * b },
		ActGELUErf:     func(a, b float64) float64 { return oracle.GELUErf(a) * b },
		ActSwiGLUOAI:   oracle.SwiGLUOAI,
		ActSwiGLUClamp: oracle.SwiGLUClamp,
		ActSitu:        oracle.Situ,
		ActIdentity:    func(g, u float64) float64 { return g * u },
		ActReLU2:       func(g, u float64) float64 { r := math.Max(g, 0); return r * r * u },
	}
	for _, k := range cpu.Gated {
		ops = append(ops, op{"actmul/" + k.String(), func(d []float32) { ActMul32JIT(d[:n], y, k) },
			func() ([]byte, error) { return sse.ActMul(k) },
			func(d []float32) cpu.Args { return elem(d, true, scr) }, gatedRef[k], 1e-9})
	}
	ungatedRef := map[ActKind]func(a, _ float64) float64{
		ActSiLU:        func(a, _ float64) float64 { return oracle.SiLU(a) },
		ActGELU:        func(a, _ float64) float64 { return oracle.GELUTanh(a) },
		ActGELUErf:     func(a, _ float64) float64 { return oracle.GELUErf(a) },
		ActSqrtSigmoid: func(a, _ float64) float64 { return oracle.SqrtSigmoid(a) },
		ActQuickGELU:   func(a, _ float64) float64 { return quick(a) },
		ActReLU2:       func(a, _ float64) float64 { r := math.Max(a, 0); return r * r },
		ActReLU:        func(a, _ float64) float64 { return math.Max(a, 0) },
		ActSqrtSoftplus: func(a, _ float64) float64 {
			return oracle.SqrtSoftplus(a)
		},
	}
	for _, k := range cpu.Ungated {
		ops = append(ops, op{"act/" + k.String(), func(d []float32) { Act32JIT(d[:n], k) },
			func() ([]byte, error) { return sse.Act(k) },
			func(d []float32) cpu.Args { return elem(d, false, scr) }, ungatedRef[k], 1e-9})
	}

	// The host tier's answers, taken before the force.
	before := make([][]float32, len(ops))
	for i, o := range ops {
		before[i] = append([]float32(nil), x...)
		o.run(before[i])
	}

	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forced the SSE tier and the probe reports %v -- the force reached nothing", cpu.HostTier())
	}
	avx2Before := cpu.MappedByTier()[cpu.TierAVX2]

	differ := 0
	for i, o := range ops {
		got := append([]float32(nil), x...)
		o.run(got)
		for j := n; j < len(got); j++ {
			if math.Float32bits(got[j]) != guardBits {
				t.Fatalf("%s: wrote element %d past the end", o.name, j)
			}
		}

		// The SSE table's kernel, called directly with the entry point's Args.
		code, err := o.emit()
		if err != nil {
			t.Fatalf("%s: the SSE table refused: %v", o.name, err)
		}
		if cpu.KernelTier(code) != cpu.TierSSE {
			t.Fatalf("%s: the SSE table's kernel is not declared SSE-tier", o.name)
		}
		c, err := cpu.Map(code)
		if err != nil {
			t.Fatalf("%s: Map on the forced SSE tier: %v", o.name, err)
		}
		direct := append([]float32(nil), x...)
		a := o.args(direct)
		c.Call(&a)
		c.Close()
		for j := 0; j < n; j++ {
			if math.Float32bits(got[j]) != math.Float32bits(direct[j]) {
				t.Fatalf("%s: element %d is %v through the entry point and %v from the SSE kernel -- "+
					"the entry point did not run the SSE tier's kernel", o.name, j, got[j], direct[j])
			}
		}

		// The float64 definition.
		want := make([]float64, n)
		if o.ref == nil { // softmax
			row := append([]float32(nil), x[:n]...)
			oracle.Softmax32(row)
			for j := range want {
				want[j] = float64(row[j])
			}
		} else {
			for j := range want {
				want[j] = o.ref(float64(x[j]), float64(y[j]))
			}
		}
		var se, sy2 float64
		for j := range want {
			d := float64(got[j]) - want[j]
			se, sy2 = se+d*d, sy2+want[j]*want[j]
		}
		if nmse := se / sy2; !(nmse <= o.bound) {
			t.Errorf("%s: NMSE %.3e against the float64 definition on the forced SSE tier (bound %.0e)", o.name, nmse, o.bound)
		}
		d := 0
		for j := 0; j < n; j++ {
			if math.Float32bits(got[j]) != math.Float32bits(before[i][j]) {
				d++
			}
		}
		differ += d
		t.Logf("%s: NMSE %.3e; %d of %d elements differ in the last bit from the %v tier's", o.name, se/sy2, d, n, old)
	}
	if old == cpu.TierAVX2 && differ == 0 {
		t.Error("every element of every op is bit-identical to the AVX2 tier's answer -- " +
			"the FMA-free SSE bodies cannot produce that, so the entry points did not run them")
	}
	if dd := cpu.MappedByTier()[cpu.TierAVX2] - avx2Before; dd != 0 {
		t.Errorf("%d AVX2 kernels were mapped while the SSE tier was forced", dd)
	}
	if cpu.MappedByTier()[cpu.TierSSE] == 0 {
		t.Error("no SSE-tier kernel was ever mapped in this process")
	}
}
