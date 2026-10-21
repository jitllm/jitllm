//go:build amd64 && jitllmtest

package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestForcedSSETierRunsTheHybridKernels is the hybrid's three entry points --
// (*JIT).GatedDelta, (*JIT).Conv1d and DeltaGate32JIT, the exact calls
// engine/model/delta.go makes -- under the forced SSE tier, computing against the
// float64 oracle, with no AVX2 kernel mapped while the force is on.
//
// It is the caller's half of the kernel gates: jit_old/cpu proves the SSE kernels
// on hand-built Args, and this proves nn selects them and builds their Args
// the way they read them.
//
// The shapes are Qwen3-Next's own (a 128-wide head, 4 taps over 8192
// channels, 32 value heads) and a ragged one of each.
func TestForcedSSETierRunsTheHybridKernels(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forced SSE and the probe reports %v -- the force reached nothing", cpu.HostTier())
	}
	avx2Before := cpu.MappedByTier()[cpu.TierAVX2]

	for _, s := range []struct{ n, taps, chans, heads int }{
		{128, 4, 8192, 32},
		{13, 3, 21, 11},
	} {
		sseBefore := cpu.MappedByTier()[cpu.TierSSE]
		f := NewJIT(256, 64, []quant.Type{quant.Q4_K})
		f.AddDelta(s.n)
		f.AddConv1d(s.taps, s.chans)
		if d := cpu.MappedByTier()[cpu.TierSSE] - sseBefore; d < 2 {
			t.Errorf("n=%d: AddDelta and AddConv1d mapped %d SSE-tier kernels, want at least 2", s.n, d)
		}
		rnd := rand.New(rand.NewSource(int64(s.n)))
		randv := func(n int, scale float64) []float32 {
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(rnd.NormFloat64() * scale)
			}
			return v
		}

		// Two tokens of the delta rule on one head, each against the oracle
		// stepped from the kernel's own state.
		st := randv(s.n*s.n, 1)
		for step := 0; step < 2; step++ {
			k, q, v := randv(s.n, 0.1), randv(s.n, 0.1), randv(s.n, 1)
			wantSt := hybWiden64(st)
			wantO := make([]float64, s.n)
			oracle.GatedDelta(wantO, wantSt, hybWiden64(k), hybWiden64(q), hybWiden64(v), 0.93, 0.41)
			o := make([]float32, s.n)
			f.GatedDelta(o, st, k, q, v, 0.93, 0.41, s.n)
			if e := hybNMSE64(o, wantO); !(e <= 1e-10) {
				t.Errorf("n=%d step %d: delta output NMSE %.3e", s.n, step, e)
			}
			if e := hybNMSE64(st, wantSt); !(e <= 1e-12) {
				t.Errorf("n=%d step %d: delta STATE NMSE %.3e", s.n, step, e)
			}
		}

		// The convolution in place, as engine/model/delta.go calls it (out aliases x).
		cst := randv((s.taps-1)*s.chans, 1)
		wT := randv(s.taps*s.chans, 0.3)
		x := randv(s.chans, 1)
		wantSt := hybWiden64(cst)
		wantOut := make([]float64, s.chans)
		oracle.Conv1d(wantOut, wantSt, hybWiden64(wT), hybWiden64(x), s.taps, s.chans)
		f.Conv1d(x, cst, wT, x, s.taps, s.chans)
		for c := range x {
			if d := math.Abs(float64(x[c]) - wantOut[c]); !(d <= 1e-5) {
				t.Fatalf("taps=%d chans=%d: conv out[%d] %v, oracle %v", s.taps, s.chans, c, x[c], wantOut[c])
			}
		}
		for i := range cst {
			if float64(cst[i]) != wantSt[i] {
				t.Fatalf("taps=%d chans=%d: conv STATE[%d] %v, want %v", s.taps, s.chans, i, cst[i], wantSt[i])
			}
		}

		// Both gates over the value heads.
		alpha, dt, b := randv(s.heads, 3), randv(s.heads, 0.5), randv(s.heads, 3)
		a := randv(s.heads, 1)
		for i := range a {
			a[i] = -float32(math.Exp(float64(a[i])))
		}
		decay, beta := make([]float32, s.heads), make([]float32, s.heads)
		DeltaGate32JIT(decay, beta, alpha, dt, b, a)
		wantD, wantB := make([]float64, s.heads), make([]float64, s.heads)
		oracle.DeltaGate(wantD, wantB, hybWiden64(alpha), hybWiden64(dt), hybWiden64(b), hybWiden64(a))
		for i := range decay {
			if r := math.Abs(float64(decay[i])-wantD[i]) / wantD[i]; !(r <= 2e-5) {
				t.Errorf("heads=%d: decay[%d] %v, oracle %v", s.heads, i, decay[i], wantD[i])
			}
			if r := math.Abs(float64(beta[i])-wantB[i]) / wantB[i]; !(r <= 4e-6) {
				t.Errorf("heads=%d: beta[%d] %v, oracle %v", s.heads, i, beta[i], wantB[i])
			}
		}
		f.Close()
		t.Logf("n=%d conv %dx%d heads=%d: delta, conv and gates ran on the SSE tier and matched the oracle",
			s.n, s.taps, s.chans, s.heads)
	}
	if d := cpu.MappedByTier()[cpu.TierAVX2] - avx2Before; d != 0 {
		t.Errorf("%d AVX2 kernels were mapped while the SSE tier was forced", d)
	}
}

func hybWiden64(v []float32) []float64 {
	d := make([]float64, len(v))
	for i, x := range v {
		d[i] = float64(x)
	}
	return d
}

func hybNMSE64(got []float32, want []float64) float64 {
	var se, sy float64
	for i, w := range want {
		d := float64(got[i]) - w
		se += d * d
		sy += w * w
	}
	return se / math.Max(sy, 1e-30)
}
