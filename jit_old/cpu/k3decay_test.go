//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestEmitDeltaDecayBoundMatchesReference gates Kimi-K3's bounded KDA decay,
// exp(lb*sigmoid(-A*(a+dt))), on this host's primary tier and (on amd64) the
// SSE tier, against fla's formula in float64: at ragged lengths with a guard
// after the output, over arguments that reach both ends of the bound (A*z
// past the exp's clamp each way) and a bound of -5 (Kimi-K3's) and -0.5.
//
// The inputs must separate the two forms: Kimi-Linear's softplus decay,
// exp(A*softplus(a+dt)), on the same inputs misses the reference by far more
// than the bound.
func TestEmitDeltaDecayBoundMatchesReference(t *testing.T) {
	type arm struct {
		name string
		code []byte
	}
	arms := []arm{{"primary", cpu.EmitDeltaDecayBound()}}
	if runtime.GOARCH == "amd64" {
		b, err := cpu.EmittersFor(cpu.TierSSE).DeltaDecayBound()
		if err != nil {
			t.Fatal(err)
		}
		arms = append(arms, arm{"sse", b})
	}
	consts := cpu.DeltaGateConsts()
	zs := []float32{-200, -60, -20, -3, -0.5, 0, 0.5, 3, 20, 60, 200, 1e-7, -1e-7, 12.5, -12.5}
	for _, a := range arms {
		t.Run(a.name, func(t *testing.T) {
			code := hybMapRunnable(t, a.code)
			defer code.Close()
			worst, apart := 0.0, 0.0
			for _, lb := range []float32{-5, -0.5} {
				for n := 1; n <= 40; n++ {
					rnd := rand.New(rand.NewSource(int64(31 + n)))
					const guard = 8
					alpha, dt, av := make([]float32, n), make([]float32, n), make([]float32, n)
					for i := 0; i < n; i++ {
						z := float32(rnd.NormFloat64() * 5)
						if i < len(zs) {
							z = zs[(i+n)%len(zs)]
						}
						dt[i] = float32(rnd.NormFloat64() * 0.5)
						alpha[i] = z - dt[i]
						// The container's rate is -exp(A_log): always negative.
						av[i] = -float32(math.Exp(rnd.NormFloat64()))
					}
					decay := make([]float32, n+guard)
					for i := n; i < n+guard; i++ {
						decay[i] = -1
					}
					l := lb
					code.Call(&cpu.Args{
						Out: &decay[0], W: (*byte)(unsafe.Pointer(&alpha[0])),
						AScale: &dt[0], AScale2: &l, Q32: &av[0],
						Scr:  (*byte)(unsafe.Pointer(&consts[0])),
						K:    int64(n / cpu.ElemLanes),
						Rows: int64(n % cpu.ElemLanes),
					})
					for i := n; i < n+guard; i++ {
						if decay[i] != -1 {
							t.Fatalf("lb=%v n=%d: wrote element %d past the end", lb, n, i)
						}
					}
					for i := 0; i < n; i++ {
						z := float64(alpha[i]) + float64(dt[i])
						s := 1 / (1 + math.Exp(float64(av[i])*z))
						want := math.Exp(float64(lb) * s)
						worst = max(worst, relErr(float64(decay[i]), want))
						sp := math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z)))
						apart = max(apart, relErr(math.Exp(float64(av[i])*sp), want))
					}
				}
			}
			if apart < 1e-1 {
				t.Fatalf("the softplus decay is within %.3e of the bounded one: the inputs do not "+
					"separate them", apart)
			}
			if worst > 2e-5 {
				t.Errorf("worst relative error %.3e (bound 2e-5, the delta gate's)", worst)
			}
			t.Logf("worst relative error %.3e", worst)
		})
	}
}
