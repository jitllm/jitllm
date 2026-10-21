//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// ssdShapes pairs a state width (baked: Mamba-2's d_state) with a row count
// (the head's head_dim, a runtime argument), ragged in both and with the rows
// unequal to the width -- the delta rule's kernel has only ever run square.
var ssdShapes = [][2]int{{1, 1}, {5, 3}, {8, 8}, {12, 7}, {16, 64}, {33, 5}, {64, 16},
	{128, 64}, {128, 1}, {130, 9}, {64, 128}}

// ssdStateSSE is the SSE tier's state update to the bit (EmitGatedSSDSSE):
// x*dt rounded, then each element's decay product and input product rounded
// apart and added, as MULPS, MULPS, ADDPS with no FMA. The float32 conversions
// keep Go from fusing either product into the add.
func ssdStateSSE(st, k, v []float32, decay, dt float32) []float32 {
	n := len(k)
	out := append([]float32(nil), st...)
	for j := range v {
		xdt := float32(v[j] * dt)
		for i := 0; i < n; i++ {
			out[j*n+i] = float32(out[j*n+i]*decay) + float32(k[i]*xdt)
		}
	}
	return out
}

// ssdNMSE is got against want in float64.
func ssdNMSE(got []float32, want []float64) float64 {
	var se, sy float64
	for i, w := range want {
		d := float64(got[i]) - w
		se += d * d
		sy += w * w
	}
	return se / math.Max(sy, 1e-30)
}

func ssdWiden(v []float32) []float64 {
	d := make([]float64, len(v))
	for i := range v {
		d[i] = float64(v[i])
	}
	return d
}

// TestEmitGatedSSDMatchesOracle steps four tokens of Mamba-2's selective
// update through the generated kernel at every shape and holds the output and
// the STATE after each to oracle.GatedSSD stepped from the kernel's own state,
// so f32 rounding does not compound. A kernel that dropped the D skip or wrote
// the unscaled value would be right on the state or the output and wrong on
// the other; the violations below prove the gate sees both.
func TestEmitGatedSSDMatchesOracle(t *testing.T) {
	for _, sh := range ssdShapes {
		n, rows := sh[0], sh[1]
		code := onHost(t)(hostTable().GatedSSD(n))
		rnd := rand.New(rand.NewSource(int64(31*n + rows)))
		st := make([]float32, rows*n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		for tok := 0; tok < 4; tok++ {
			k, q, v := make([]float32, n), make([]float32, n), make([]float32, rows)
			for i := 0; i < n; i++ {
				k[i] = float32(rnd.NormFloat64() * 0.3)
				q[i] = float32(rnd.NormFloat64() * 0.3)
			}
			for j := range v {
				v[j] = float32(rnd.NormFloat64())
			}
			decay, dt, d := float32(0.6+0.3*rnd.Float64()), float32(0.05+rnd.Float64()), float32(rnd.NormFloat64())
			wantSt := ssdWiden(st)
			wantO := make([]float64, rows)
			oracle.GatedSSD(wantO, wantSt, ssdWiden(k), ssdWiden(q), ssdWiden(v), float64(decay), float64(dt), float64(d))
			// The SSE tier has no FMA: its state update is defined to the bit by
			// ssdStateSSE, and is held to it before the float64 bound below.
			var model []float32
			if cpu.HostTier() == cpu.TierSSE {
				model = ssdStateSSE(st, k, v, decay, dt)
			}
			noD := make([]float64, rows)
			oracle.GatedSSD(noD, ssdWiden(st), ssdWiden(k), ssdWiden(q), ssdWiden(v), float64(decay), float64(dt), 0)

			o := make([]float32, rows+1)
			o[rows] = -1234.5
			sc := [3]float32{decay, dt, d}
			code.Call(&cpu.Args{Out: &o[0], W: (*byte)(unsafe.Pointer(&st[0])), AScale: &k[0],
				Rows: int64(rows), Scr: (*byte)(unsafe.Pointer(&sc[0])), Q32: &q[0], Q2: &v[0]})
			if o[rows] != -1234.5 {
				t.Fatalf("n=%d rows=%d: wrote past the output", n, rows)
			}
			if e := ssdNMSE(o[:rows], wantO); e > 1e-10 {
				t.Errorf("n=%d rows=%d tok %d: output NMSE %.3e", n, rows, tok, e)
			}
			stateBound := 1e-12
			if model != nil {
				for i := range st {
					if math.Float32bits(st[i]) != math.Float32bits(model[i]) {
						t.Fatalf("n=%d rows=%d tok %d: state[%d] = %v, the SSE tier's unfused update gives %v",
							n, rows, tok, i, st[i], model[i])
					}
				}
				// A product rounded before each add, three to an element: at n=1
				// the NMSE is one element's relative error, nothing averages it,
				// and a cancelling decay and input reach 5.8e-12 on these inputs.
				// The bit-exact model above is the kernel's bar on this tier.
				stateBound = 1e-10
			}
			if e := ssdNMSE(st, wantSt); e > stateBound {
				t.Errorf("n=%d rows=%d tok %d: STATE NMSE %.3e", n, rows, tok, e)
			}
			// The skip is load-bearing at this bound: without it the oracle
			// itself moves far past it.
			if e := ssdNMSE(o[:rows], noD); !(e > 1e-6) && d != 0 {
				t.Errorf("n=%d rows=%d: the D skip is invisible to this gate (%.3e)", n, rows, e)
			}
		}
		code.Close()
	}
}

// TestEmitSSDGateMatchesOracle holds Mamba-2's dt and decay to oracle.SSDGate
// over every residue of the vector width, through softplus's whole range (a
// large positive argument, where softplus is the identity, and a large
// negative one, where it underflows toward zero).
func TestEmitSSDGateMatchesOracle(t *testing.T) {
	code := onHost(t)(hostTable().SSDGate())
	defer code.Close()
	consts := cpu.DeltaGateConsts()
	for _, n := range []int{1, 3, 7, 8, 9, 16, 24, 31, 64, 128, 130} {
		rnd := rand.New(rand.NewSource(int64(n)))
		alpha, bias, a := make([]float32, n), make([]float32, n), make([]float32, n)
		for i := 0; i < n; i++ {
			alpha[i] = float32(rnd.NormFloat64() * 6)
			bias[i] = float32(rnd.NormFloat64() * 3)
			a[i] = -float32(math.Exp(rnd.NormFloat64()))
		}
		alpha[0] = 40
		if n > 1 {
			alpha[1] = -40
		}
		decay, dt := make([]float32, n+1), make([]float32, n+1)
		decay[n], dt[n] = -7, -7
		code.Call(&cpu.Args{Out: &decay[0], Out2: &dt[0], W: (*byte)(unsafe.Pointer(&alpha[0])),
			AScale: &bias[0], Q32: &a[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
			K: int64(n / cpu.ElemLanes), Rows: int64(n % cpu.ElemLanes)})
		if decay[n] != -7 || dt[n] != -7 {
			t.Fatalf("n=%d: wrote past the end", n)
		}
		wd, wt := make([]float64, n), make([]float64, n)
		oracle.SSDGate(wd, wt, ssdWiden(alpha), ssdWiden(bias), ssdWiden(a))
		for i := 0; i < n; i++ {
			if rel := math.Abs(float64(dt[i])-wt[i]) / math.Max(math.Abs(wt[i]), 1e-30); rel > 1e-5 {
				t.Errorf("n=%d dt[%d] = %g, want %g", n, i, dt[i], wt[i])
			}
			if rel := math.Abs(float64(decay[i])-wd[i]) / math.Max(math.Abs(wd[i]), 1e-30); rel > 2e-5 {
				t.Errorf("n=%d decay[%d] = %g, want %g", n, i, decay[i], wd[i])
			}
		}
	}
}
