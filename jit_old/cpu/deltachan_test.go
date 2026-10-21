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

// TestEmitGatedDeltaChanMatchesReference gates Kimi-Linear's delta rule, whose
// decay varies per channel, against oracle.GatedDeltaChan. A kernel that
// broadcast decay[0] would compute the scalar rule, so the test also runs the
// scalar kernel at decay[0] and requires it to disagree. It checks the state as
// well as the output, since a forgotten update only shows on later tokens.
func TestEmitGatedDeltaChanMatchesReference(t *testing.T) {
	// Ragged widths on both hosts: the tail is baked scalar code, and a tail
	// that reads the wrong decay lane shows up here and nowhere else.
	for _, n := range []int{8, 16, 64, 128, 1, 5, 12, 20, 33, 130} {
		code := onHost(t)(hostTable().GatedDeltaChan(n))
		defer code.Close()

		rnd := rand.New(rand.NewSource(int64(n) + 7))
		st := make([]float32, n*n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		k := make([]float32, n)
		q := make([]float32, n)
		v := make([]float32, n)
		decay := make([]float32, n)
		for i := 0; i < n; i++ {
			k[i] = float32(rnd.NormFloat64() * 0.1)
			q[i] = float32(rnd.NormFloat64() * 0.1)
			v[i] = float32(rnd.NormFloat64())
			// Spread across the band a real forget gate produces, and
			// deliberately not constant: see the violation arm below.
			decay[i] = float32(0.55 + 0.4*rnd.Float64())
		}
		gate := float32(0.41)

		wantSt := make([]float64, n*n)
		for i := range st {
			wantSt[i] = float64(st[i])
		}
		wantO := make([]float64, n)
		k64, q64, v64, d64 := f64s(k), f64s(q), f64s(v), f64s(decay)
		oracle.GatedDeltaChan(wantO, wantSt, k64, q64, v64, d64, float64(gate))

		got := make([]float32, n)
		gotSt := append([]float32(nil), st...)
		sc := [2]float32{0, gate}
		code.Call(&cpu.Args{
			Out:     &got[0],
			W:       (*byte)(unsafe.Pointer(&gotSt[0])),
			AScale:  &k[0],
			Rows:    int64(n),
			Scr:     (*byte)(unsafe.Pointer(&sc[0])),
			Q32:     &q[0],
			Q2:      &v[0],
			AScale2: &decay[0],
		})

		if nmse := nmse64(f64s(got), wantO); nmse > 1e-10 {
			t.Errorf("n=%d: output NMSE %.3e", n, nmse)
		}
		if nmse := nmse64(f64s(gotSt), wantSt); nmse > 1e-12 {
			t.Errorf("n=%d: STATE NMSE %.3e -- the recurrence is wrong, not just the output", n, nmse)
		}

		// The violation: the scalar kernel on the same inputs at decay[0],
		// which a broadcasting kernel would reproduce. n=1 is exempt because
		// there one channel is the head.
		if n == 1 {
			continue
		}
		scode := onHost(t)(hostTable().GatedDelta(n))
		defer scode.Close()
		bcast := make([]float32, n)
		bcastSt := append([]float32(nil), st...)
		bsc := [2]float32{decay[0], gate}
		scode.Call(&cpu.Args{
			Out:    &bcast[0],
			W:      (*byte)(unsafe.Pointer(&bcastSt[0])),
			AScale: &k[0],
			Rows:   int64(n),
			Scr:    (*byte)(unsafe.Pointer(&bsc[0])),
			Q32:    &q[0],
			Q2:     &v[0],
		})
		if nmse := nmse64(f64s(bcast), wantO); nmse < 1e-6 {
			t.Errorf("n=%d: a BROADCAST decay scores NMSE %.3e against the "+
				"per-channel reference -- this fixture cannot tell the two "+
				"rules apart and gates nothing", n, nmse)
		}
	}
}

// TestEmitGatedDeltaChanRefusesEmptyWidths: every positive width is served.
func TestEmitGatedDeltaChanRefusesEmptyWidths(t *testing.T) {
	for _, n := range []int{0, -8} {
		if _, err := cpu.EmitGatedDeltaChan(n); err == nil {
			t.Errorf("n=%d was accepted", n)
		}
	}
}

func f64s(x []float32) []float64 {
	o := make([]float64, len(x))
	for i, v := range x {
		o[i] = float64(v)
	}
	return o
}

func nmse64(got, want []float64) float64 {
	var se, sy float64
	for i := range want {
		d := got[i] - want[i]
		se += d * d
		sy += want[i] * want[i]
	}
	return se / math.Max(sy, 1e-30)
}
