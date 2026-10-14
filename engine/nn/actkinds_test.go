package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/cpu"
)

// TestAct32JITRunsEveryUngatedKind calls the entry point for every ungated
// kind on the host's own tier and holds it to the float64 definition. It gates
// the kind-to-slot lookup (ungatedIndex), which the kernel gates do not reach,
// and runs without the jitllmtest tag.
func TestAct32JITRunsEveryUngatedKind(t *testing.T) {
	ref := map[ActKind]func(float64) float64{
		ActSiLU:    oracle.SiLU,
		ActGELU:    oracle.GELUTanh,
		ActGELUErf: oracle.GELUErf,
		// Qwen3.8-Flash-Next's n-gram gate.
		ActSqrtSigmoid: oracle.SqrtSigmoid,
		ActQuickGELU:   func(x float64) float64 { return x / (1 + math.Exp(-1.702*x)) },
		ActReLU2:       func(x float64) float64 { r := math.Max(x, 0); return r * r },
		ActReLU:        func(x float64) float64 { return math.Max(x, 0) },
		// DeepSeek V4's router gate.
		ActSqrtSoftplus: oracle.SqrtSoftplus,
	}
	const n = 203 // whole vectors and a tail
	for _, k := range cpu.Ungated {
		f, ok := ref[k]
		if !ok {
			t.Fatalf("%v: no reference; add one here", k)
		}
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.37) * float64(1+i%7))
		}
		got := append([]float32(nil), x...)
		Act32JIT(got, k)
		var se, sy float64
		for i := range x {
			w := f(float64(x[i]))
			d := float64(got[i]) - w
			se, sy = se+d*d, sy+w*w
		}
		if nmse := se / sy; !(nmse <= 1e-9) {
			t.Errorf("%v: NMSE %.3e against the definition", k, nmse)
		}
	}
}
