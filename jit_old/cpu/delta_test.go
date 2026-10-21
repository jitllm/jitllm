//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestEmitGatedDeltaMatchesReference gates the kernel against the recurrence
// written out in the plainest Go there is.
//
// It checks the state, not only the output: a kernel that forgets the rank-one
// update is right for this token and wrong for every token after it.
func TestEmitGatedDeltaMatchesReference(t *testing.T) {
	// 1, 5, 12, 20, 33 and 130 are ragged on both hosts: the tail is baked
	// scalar code, and a wrong tail shows in the state check below as well as
	// in the output.
	for _, n := range []int{8, 16, 64, 128, 1, 5, 12, 20, 33, 130} {
		code := onHost(t)(hostTable().GatedDelta(n))
		defer code.Close()

		rnd := rand.New(rand.NewSource(int64(n)))
		st := make([]float32, n*n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		k := make([]float32, n)
		q := make([]float32, n)
		v := make([]float32, n)
		for i := 0; i < n; i++ {
			k[i] = float32(rnd.NormFloat64() * 0.1)
			q[i] = float32(rnd.NormFloat64() * 0.1)
			v[i] = float32(rnd.NormFloat64())
		}
		decay, gate := float32(0.93), float32(0.41)

		// The reference, in float64 so the comparison is against the
		// arithmetic and not against another f32 rounding order.
		wantSt := make([]float64, n*n)
		wantO := make([]float64, n)
		for i := range st {
			wantSt[i] = float64(st[i])
		}
		for j := 0; j < n; j++ {
			row := wantSt[j*n : (j+1)*n]
			var sk float64
			for i := range row {
				row[i] *= float64(decay)
				sk += row[i] * float64(k[i])
			}
			d := (float64(v[j]) - sk) * float64(gate)
			var o float64
			for i := range row {
				row[i] += d * float64(k[i])
				o += row[i] * float64(q[i])
			}
			wantO[j] = o
		}

		got := make([]float32, n)
		sc := [2]float32{decay, gate}
		code.Call(&cpu.Args{
			Out:    &got[0],
			W:      (*byte)(unsafe.Pointer(&st[0])),
			AScale: &k[0],
			Rows:   int64(n),
			Scr:    (*byte)(unsafe.Pointer(&sc[0])),
			Q32:    &q[0],
			Q2:     &v[0],
		})

		var se, sy float64
		for j := 0; j < n; j++ {
			d := float64(got[j]) - wantO[j]
			se += d * d
			sy += wantO[j] * wantO[j]
		}
		if nmse := se / math.Max(sy, 1e-30); nmse > 1e-10 {
			t.Errorf("n=%d: output NMSE %.3e", n, nmse)
		}
		se, sy = 0, 0
		for i := range st {
			d := float64(st[i]) - wantSt[i]
			se += d * d
			sy += wantSt[i] * wantSt[i]
		}
		if nmse := se / math.Max(sy, 1e-30); nmse > 1e-12 {
			t.Errorf("n=%d: STATE NMSE %.3e -- the recurrence is wrong, not just the output", n, nmse)
		}
	}
}

// TestEmitGatedDeltaRefusesEmptyWidths: every positive width is served, and
// nothing else is.
func TestEmitGatedDeltaRefusesEmptyWidths(t *testing.T) {
	for _, n := range []int{0, -8} {
		if _, err := cpu.EmitGatedDelta(n); err == nil {
			t.Errorf("n=%d was accepted", n)
		}
	}
}

// TestEmitConv1dMatchesReference gates the convolution and its state shift; a
// kernel that leaves the state unshifted is right only on the first token.
func TestEmitConv1dMatchesReference(t *testing.T) {
	for _, taps := range []int{2, 3, 4, 6} {
		for _, chans := range []int{8, 64, 512, 1, 5, 12, 20, 33} {
			code := onHost(t)(hostTable().Conv1d(taps, chans))
			rnd := rand.New(rand.NewSource(int64(taps*1000 + chans)))
			st := make([]float32, (taps-1)*chans)
			w := make([]float32, taps*chans)
			x := make([]float32, chans)
			for i := range st {
				st[i] = float32(rnd.NormFloat64())
			}
			for i := range w {
				w[i] = float32(rnd.NormFloat64() * 0.3)
			}
			for i := range x {
				x[i] = float32(rnd.NormFloat64())
			}
			wantOut := make([]float64, chans)
			wantSt := make([]float64, len(st))
			for c := 0; c < chans; c++ {
				acc := float64(x[c]) * float64(w[(taps-1)*chans+c])
				for t := 0; t < taps-1; t++ {
					acc += float64(st[t*chans+c]) * float64(w[t*chans+c])
				}
				wantOut[c] = acc
				for t := 0; t < taps-2; t++ {
					wantSt[t*chans+c] = float64(st[(t+1)*chans+c])
				}
				wantSt[(taps-2)*chans+c] = float64(x[c])
			}
			out := make([]float32, chans)
			code.Call(&cpu.Args{
				Out:    &out[0],
				W:      (*byte)(unsafe.Pointer(&st[0])),
				AScale: &w[0],
				Q32:    &x[0],
			})
			code.Close()
			for c := 0; c < chans; c++ {
				if d := math.Abs(float64(out[c]) - wantOut[c]); d > 1e-5 {
					t.Fatalf("taps=%d chans=%d: out[%d] off by %.3e", taps, chans, c, d)
				}
			}
			for i := range st {
				if d := math.Abs(float64(st[i]) - wantSt[i]); d != 0 {
					t.Fatalf("taps=%d chans=%d: STATE[%d] is %v, want %v -- the shift is wrong",
						taps, chans, i, st[i], wantSt[i])
				}
			}
		}
	}
}

// TestEmitConv1dAliases is the in-place contract: out and x are the same buffer
// in the engine, because the convolution replaces the projection it reads.
func TestEmitConv1dAliases(t *testing.T) {
	const taps, chans = 4, 64
	code := onHost(t)(hostTable().Conv1d(taps, chans))
	defer code.Close()
	rnd := rand.New(rand.NewSource(3))
	st := make([]float32, (taps-1)*chans)
	w := make([]float32, taps*chans)
	x := make([]float32, chans)
	for i := range st {
		st[i] = float32(rnd.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rnd.NormFloat64() * 0.3)
	}
	for i := range x {
		x[i] = float32(rnd.NormFloat64())
	}
	sep := append([]float32(nil), st...)
	xc := append([]float32(nil), x...)
	out := make([]float32, chans)
	code.Call(&cpu.Args{Out: &out[0], W: (*byte)(unsafe.Pointer(&sep[0])),
		AScale: &w[0], Q32: &xc[0]})

	code.Call(&cpu.Args{Out: &x[0], W: (*byte)(unsafe.Pointer(&st[0])),
		AScale: &w[0], Q32: &x[0]})
	for c := 0; c < chans; c++ {
		if x[c] != out[c] {
			t.Fatalf("in place differs at %d: %v vs %v", c, x[c], out[c])
		}
	}
	for i := range st {
		if st[i] != sep[i] {
			t.Fatalf("in place left a different state at %d", i)
		}
	}
}

// TestEmitDeltaGateMatchesReference gates both gates against math's own.
//
// softplus(z) overflows above ~88 as log(1+exp(z)); the kernel uses a
// branch-free |z| identity, and z = +-200 is where a wrong identity becomes Inf
// or 0.
func TestEmitDeltaGateMatchesReference(t *testing.T) {
	code := onHost(t)(hostTable().DeltaGate())
	defer code.Close()
	consts := cpu.DeltaGateConsts()

	zs := []float32{-200, -60, -20, -3, -0.5, 0, 0.5, 3, 20, 60, 200,
		1e-7, -1e-7, 12.5, -12.5, 87, -87}
	// 17 is ragged on both hosts and 3 has no whole vector at all.
	for _, n := range []int{len(zs), 3, 32} {
		deltaGateCase(t, code, consts, zs, n)
	}
}

func deltaGateCase(t *testing.T, code *cpu.Code, consts []float32, zs []float32, n int) {
	t.Helper()
	const guard = 8
	alpha := make([]float32, n)
	dt := make([]float32, n)
	b := make([]float32, n)
	av := make([]float32, n)
	rnd := rand.New(rand.NewSource(11))
	for i := 0; i < n; i++ {
		z := float32(0)
		if i < len(zs) {
			z = zs[i]
		} else {
			z = float32(rnd.NormFloat64() * 5)
		}
		dt[i] = float32(rnd.NormFloat64() * 0.5)
		alpha[i] = z - dt[i]
		b[i] = float32(rnd.NormFloat64() * 4)
		// ssm_a is -exp(A_log) in the file, so it is always negative.
		av[i] = -float32(math.Exp(rnd.NormFloat64()))
	}
	decay := make([]float32, n+guard)
	beta := make([]float32, n+guard)
	for i := n; i < n+guard; i++ {
		decay[i], beta[i] = -1, -1
	}
	code.Call(&cpu.Args{
		Out: &decay[0], Out2: &beta[0],
		W:      (*byte)(unsafe.Pointer(&alpha[0])),
		AScale: &dt[0], AScale2: &b[0], Q32: &av[0],
		Scr:  (*byte)(unsafe.Pointer(&consts[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	})
	for i := n; i < n+guard; i++ {
		if decay[i] != -1 || beta[i] != -1 {
			t.Fatalf("n=%d: wrote element %d past the end", n, i)
		}
	}
	for i := 0; i < n; i++ {
		z := float64(alpha[i]) + float64(dt[i])
		sp := math.Log1p(math.Exp(z))
		if z > 30 {
			sp = z // the scalar guard the kernel's identity removes
		}
		wantD := math.Exp(sp * float64(av[i]))
		wantB := 1 / (1 + math.Exp(-float64(b[i])))
		// The shared exp clamps at -87.3 rather than returning zero. Below
		// that the true value is not representable in f32, so the kernel's
		// 1.2e-38 and the reference's 1e-88 both mean "forgotten".
		switch {
		case wantD < 1e-37:
			if decay[i] > 1e-37 {
				t.Errorf("decay[%d]: z=%v A=%v got %v, want underflow", i, z, av[i], decay[i])
			}
		default:
			if d := relErr(float64(decay[i]), wantD); d > 2e-5 {
				t.Errorf("decay[%d]: z=%v A=%v got %v want %v (rel %.2e)",
					i, z, av[i], decay[i], wantD, d)
			}
		}
		if d := relErr(float64(beta[i]), wantB); d > 2e-6 {
			t.Errorf("beta[%d]: b=%v got %v want %v (rel %.2e)",
				i, b[i], beta[i], wantB, d)
		}
	}
}

func relErr(got, want float64) float64 {
	if want == 0 {
		return math.Abs(got)
	}
	return math.Abs(got-want) / math.Abs(want)
}
