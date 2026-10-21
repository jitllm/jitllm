//go:build amd64

package cpu

import (
	"math/rand"
	"strconv"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
)

func hybCallDeltaChan(c *Code, o, st, k, q, v, decay []float32, gate float32, n int) {
	sc := [2]float32{0, gate}
	c.Call(&Args{
		Out:     &o[0],
		W:       (*byte)(unsafe.Pointer(&st[0])),
		AScale:  &k[0],
		Rows:    int64(n),
		Scr:     (*byte)(unsafe.Pointer(&sc[0])),
		Q32:     &q[0],
		Q2:      &v[0],
		AScale2: &decay[0],
	})
}

// TestEmitGatedDeltaChanSSEMatchesOracle is TestEmitGatedDeltaSSEMatchesOracle
// for the per-channel decay: four tokens at every ragged width, guards on every
// buffer, the state checked as well as the output, and the SSE and AVX2 arms
// compared against each other on the same host.
//
// The decay buffer is guarded too: this kernel reads it as a vector, so a tail
// that runs the vector body would read past the last channel.
func TestEmitGatedDeltaChanSSEMatchesOracle(t *testing.T) {
	em := EmittersFor(TierSSE)
	for _, n := range hybWidths {
		b, err := em.GatedDeltaChan(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		c := hybMapSSE(t, "gated_delta_chan_sse_"+strconv.Itoa(n), b)
		var avx *Code
		if HostTier() == TierAVX2 {
			ab, err := EmitGatedDeltaChan(n)
			if err != nil {
				t.Fatal(err)
			}
			avx = mustMap(t, ab)
		}

		rnd := rand.New(rand.NewSource(int64(2000 + n)))
		st, stAll := hybGuarded(n * n)
		o, oAll := hybGuarded(n)
		k, _ := hybGuarded(n)
		q, _ := hybGuarded(n)
		v, _ := hybGuarded(n)
		decay, decayAll := hybGuarded(n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		for i := 0; i < n; i++ {
			k[i] = float32(rnd.NormFloat64() * 0.1)
			q[i] = float32(rnd.NormFloat64() * 0.1)
			v[i] = float32(rnd.NormFloat64())
			decay[i] = float32(0.55 + 0.4*rnd.Float64())
		}
		gate := float32(0.41)

		var avxSt []float32
		if avx != nil {
			avxSt = append([]float32(nil), st...)
		}
		for step := 0; step < 4; step++ {
			wantSt := hybWiden(st)
			wantO := make([]float64, n)
			oracle.GatedDeltaChan(wantO, wantSt, hybWiden(k), hybWiden(q), hybWiden(v), hybWiden(decay), float64(gate))

			hybCallDeltaChan(c, o, st, k, q, v, decay, gate, n)
			hybCheckGuard(t, "gated_delta_chan o", oAll, n)
			hybCheckGuard(t, "gated_delta_chan state", stAll, n*n)
			hybCheckGuard(t, "gated_delta_chan decay", decayAll, n)
			if e := hybNMSE(o, wantO); !(e <= 1e-10) {
				t.Fatalf("n=%d step %d: SSE output NMSE %.3e", n, step, e)
			}
			if e := hybNMSE(st, wantSt); !(e <= 1e-12) {
				t.Fatalf("n=%d step %d: SSE STATE NMSE %.3e", n, step, e)
			}
			if avx != nil {
				ao := make([]float32, n)
				hybCallDeltaChan(avx, ao, avxSt, k, q, v, decay, gate, n)
				if e := hybNMSE(ao, wantO); !(e <= 1e-10) {
					t.Fatalf("n=%d step %d: AVX2 output NMSE %.3e", n, step, e)
				}
			}
		}
	}
}
