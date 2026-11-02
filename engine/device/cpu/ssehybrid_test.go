//go:build amd64

package cpu

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The SSE tier's hybrid kernels, gated against the float64 oracle
// (internal/oracle/delta.go) at the AVX2 kernels' bounds, through the VEX leak
// gate at every shape, and executed (mapping alone proves nothing about
// whether a CPU can run the bytes).
//
// Every buffer is misaligned and guarded: a legacy packed op with a memory
// operand faults on an address that is not 16-byte aligned, which a fresh Go
// slice would hide. Each buffer starts one float into its allocation, and
// sentinels after it catch a tail that stores a whole vector. A Map refusal of
// an SSE-tier kernel is a failure, not a skip.

// hybWidths is 1 through several vectors, every residue mod 4 and mod 8,
// plus the engine's own state width (128) and one past it.
var hybWidths = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 20, 33, 64, 127, 128, 130}

const hybGuard = 8

// hybSentinel is a finite value no kernel here produces. Not a NaN: x86
// arithmetic on a NaN returns its payload unchanged, so a kernel that
// read-modify-writes a guard slot (decays it, adds 0*k to it) would store the
// sentinel's own bits back and the guard would compare equal.
var hybSentinel = float32(-1234.5678)

// hybGuarded returns n floats that start one float into their allocation (so
// they are not 16-byte aligned) followed by hybGuard sentinels, and the whole
// allocation for checking the guard.
func hybGuarded(n int) (s, all []float32) {
	all = make([]float32, 1+n+hybGuard)
	for i := range all {
		all[i] = hybSentinel
	}
	return all[1 : 1+n], all
}

// hybCheckGuard fails t if anything wrote before s or after its n elements.
func hybCheckGuard(t *testing.T, what string, all []float32, n int) {
	t.Helper()
	if math.Float32bits(all[0]) != math.Float32bits(hybSentinel) {
		t.Errorf("%s: wrote the element BEFORE the buffer", what)
	}
	for i := 1 + n; i < len(all); i++ {
		if math.Float32bits(all[i]) != math.Float32bits(hybSentinel) {
			t.Errorf("%s: wrote element %d PAST the end (n=%d): %v", what, i-1, n, all[i])
			return
		}
	}
}

func hybMapSSE(t *testing.T, name string, code []byte) *Code {
	t.Helper()
	requireSSEKernel(t, name, code)
	if KernelTier(code) != TierSSE {
		t.Fatalf("%s: not declared SSE-tier", name)
	}
	c, err := Map(code)
	if err != nil {
		t.Fatalf("%s: Map refused an SSE-tier kernel on a host of tier %v: %v", name, HostTier(), err)
	}
	return c
}

func hybWiden(src []float32) []float64 {
	d := make([]float64, len(src))
	for i, v := range src {
		d[i] = float64(v)
	}
	return d
}

func hybNMSE(got []float32, want []float64) float64 {
	var se, sy float64
	for i, w := range want {
		d := float64(got[i]) - w
		se += d * d
		sy += w * w
	}
	return se / math.Max(sy, 1e-30)
}

// hybCallDelta runs one head of a gated-delta kernel.
func hybCallDelta(c *Code, o, st, k, q, v []float32, decay, gate float32, n int) {
	sc := [2]float32{decay, gate}
	c.Call(&Args{
		Out:    &o[0],
		W:      (*byte)(unsafe.Pointer(&st[0])),
		AScale: &k[0],
		Rows:   int64(n),
		Scr:    (*byte)(unsafe.Pointer(&sc[0])),
		Q32:    &q[0],
		Q2:     &v[0],
	})
}

// TestEmitGatedDeltaSSEMatchesOracle runs four tokens of the recurrence at
// every ragged width and checks the output and the state after each, against
// the oracle stepped from the kernel's own state (so f32 rounding does not
// compound), and the inputs k, q, v unchanged. The state catches a kernel that
// forgets the rank-one update, which this token's output would not.
func TestEmitGatedDeltaSSEMatchesOracle(t *testing.T) {
	em := EmittersFor(TierSSE)
	var sseO, sseS, avxO, avxS float64
	for _, n := range hybWidths {
		var avx *Code
		b, err := em.GatedDelta(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		c := hybMapSSE(t, "gated_delta_sse_"+strconv.Itoa(n), b)
		if HostTier() == TierAVX2 {
			ab, err := EmitGatedDelta(n)
			if err != nil {
				t.Fatal(err)
			}
			avx = mustMap(t, ab)
		}

		rnd := rand.New(rand.NewSource(int64(1000 + n)))
		st, stAll := hybGuarded(n * n)
		o, oAll := hybGuarded(n)
		k, _ := hybGuarded(n)
		q, _ := hybGuarded(n)
		v, _ := hybGuarded(n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		stAVX := append([]float32(nil), st...)
		for step := 0; step < 4; step++ {
			for i := 0; i < n; i++ {
				k[i] = float32(rnd.NormFloat64() * 0.1)
				q[i] = float32(rnd.NormFloat64() * 0.1)
				v[i] = float32(rnd.NormFloat64())
			}
			decay := float32(0.85 + 0.1*rnd.Float64())
			gate := float32(0.2 + 0.6*rnd.Float64())
			k0, q0, v0 := append([]float32(nil), k...), append([]float32(nil), q...), append([]float32(nil), v...)

			wantSt := hybWiden(st)
			wantO := make([]float64, n)
			oracle.GatedDelta(wantO, wantSt, hybWiden(k), hybWiden(q), hybWiden(v), float64(decay), float64(gate))

			hybCallDelta(c, o, st, k, q, v, decay, gate, n)
			hybCheckGuard(t, "gated_delta o", oAll, n)
			hybCheckGuard(t, "gated_delta state", stAll, n*n)
			for i := 0; i < n; i++ {
				if k[i] != k0[i] || q[i] != q0[i] || v[i] != v0[i] {
					t.Fatalf("n=%d step %d: the kernel wrote an INPUT at %d", n, step, i)
				}
			}
			if e := hybNMSE(o, wantO); !(e <= 1e-10) {
				t.Errorf("n=%d step %d: output NMSE %.3e (bound 1e-10)", n, step, e)
			}
			if e := hybNMSE(st, wantSt); !(e <= 1e-12) {
				t.Errorf("n=%d step %d: STATE NMSE %.3e (bound 1e-12) -- the recurrence is wrong, not just the output", n, step, e)
			}

			sseO = math.Max(sseO, hybNMSE(o, wantO))
			sseS = math.Max(sseS, hybNMSE(st, wantSt))

			// The AVX2 twin, same step from the same state: not bit-identical
			// (it fuses the multiply-adds), so it is scored against the same
			// oracle rather than against the SSE bits.
			if avx != nil {
				oa := make([]float32, n)
				hybCallDelta(avx, oa, stAVX, k, q, v, decay, gate, n)
				avxO = math.Max(avxO, hybNMSE(oa, wantO))
				avxS = math.Max(avxS, hybNMSE(stAVX, wantSt))
				copy(stAVX, st)
			}
		}
		if avx != nil {
			avx.Close()
		}
		c.Close()
	}
	t.Logf("worst NMSE against the oracle over %d widths x 4 tokens: SSE output %.2e state %.2e; AVX2 output %.2e state %.2e",
		len(hybWidths), sseO, sseS, avxO, avxS)
}

// TestEmitGatedDeltaSSERefusesEmptyWidths: every positive width is served,
// and nothing else is -- the AVX2 kernel's contract.
func TestEmitGatedDeltaSSERefusesEmptyWidths(t *testing.T) {
	for _, n := range []int{0, -8} {
		if _, err := EmitGatedDeltaSSE(n); err == nil {
			t.Errorf("n=%d was accepted", n)
		}
	}
}

func hybCallConv(c *Code, out, st, w, x []float32) {
	c.Call(&Args{
		Out:    &out[0],
		W:      (*byte)(unsafe.Pointer(&st[0])),
		AScale: &w[0],
		Q32:    &x[0],
	})
}

// TestEmitConv1dSSEMatchesOracle gates the convolution and its state shift at
// every tap count and ragged channel count, including the engine's own shape
// (4 taps x 8192 channels on Qwen3-Next). The shift is exact -- it moves bits
// and computes nothing -- so the state is compared bit for bit.
func TestEmitConv1dSSEMatchesOracle(t *testing.T) {
	em := EmittersFor(TierSSE)
	chansList := append(append([]int(nil), hybWidths...), 512, 8192)
	for _, taps := range []int{2, 3, 4, 5, 6, 8} {
		for _, chans := range chansList {
			b, err := em.Conv1d(taps, chans)
			if err != nil {
				t.Fatalf("taps=%d chans=%d: %v", taps, chans, err)
			}
			c := hybMapSSE(t, "conv1d_sse_"+strconv.Itoa(taps)+"x"+strconv.Itoa(chans), b)
			rnd := rand.New(rand.NewSource(int64(taps*100000 + chans)))
			st, stAll := hybGuarded((taps - 1) * chans)
			w, _ := hybGuarded(taps * chans)
			x, _ := hybGuarded(chans)
			out, outAll := hybGuarded(chans)
			for i := range st {
				st[i] = float32(rnd.NormFloat64())
			}
			for i := range w {
				w[i] = float32(rnd.NormFloat64() * 0.3)
			}
			for i := range x {
				x[i] = float32(rnd.NormFloat64())
			}
			wantSt := hybWiden(st)
			wantOut := make([]float64, chans)
			oracle.Conv1d(wantOut, wantSt, hybWiden(w), hybWiden(x), taps, chans)
			w0, x0 := append([]float32(nil), w...), append([]float32(nil), x...)

			hybCallConv(c, out, st, w, x)
			c.Close()
			hybCheckGuard(t, "conv1d out", outAll, chans)
			hybCheckGuard(t, "conv1d state", stAll, (taps-1)*chans)
			for i := range w {
				if w[i] != w0[i] {
					t.Fatalf("taps=%d chans=%d: the kernel wrote the WEIGHTS at %d", taps, chans, i)
				}
			}
			for i := range x {
				if x[i] != x0[i] {
					t.Fatalf("taps=%d chans=%d: the kernel wrote x at %d (out does not alias it here)", taps, chans, i)
				}
			}
			for ch := 0; ch < chans; ch++ {
				if d := math.Abs(float64(out[ch]) - wantOut[ch]); !(d <= 1e-5) {
					t.Fatalf("taps=%d chans=%d: out[%d] = %v, oracle %v (|d| %.3e)", taps, chans, ch, out[ch], wantOut[ch], d)
				}
			}
			for i := range st {
				if float64(st[i]) != wantSt[i] {
					t.Fatalf("taps=%d chans=%d: STATE[%d] (plane %d, chan %d) is %v, want %v -- the shift is wrong",
						taps, chans, i, i/chans, i%chans, st[i], wantSt[i])
				}
			}
		}
	}
}

// TestEmitConv1dSSEAliases is the in-place contract: out and x are the same
// buffer in the engine (engine/model/delta.go's conv1d), because the convolution
// replaces the projection it reads.
func TestEmitConv1dSSEAliases(t *testing.T) {
	for _, chans := range []int{64, 13, 3} {
		const taps = 4
		b, err := EmitConv1dSSE(taps, chans)
		if err != nil {
			t.Fatal(err)
		}
		c := hybMapSSE(t, "conv1d_sse_alias", b)
		rnd := rand.New(rand.NewSource(3))
		st, _ := hybGuarded((taps - 1) * chans)
		w, _ := hybGuarded(taps * chans)
		x, xAll := hybGuarded(chans)
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
		hybCallConv(c, out, sep, w, xc)
		hybCallConv(c, x, st, w, x)
		c.Close()
		hybCheckGuard(t, "conv1d in place", xAll, chans)
		for ch := 0; ch < chans; ch++ {
			if x[ch] != out[ch] {
				t.Fatalf("chans=%d: in place differs at %d: %v vs %v", chans, ch, x[ch], out[ch])
			}
		}
		for i := range st {
			if st[i] != sep[i] {
				t.Fatalf("chans=%d: in place left a different state at %d", chans, i)
			}
		}
	}
}

// TestEmitConv1dSSERefusesOutOfRangeShapes: taps in [2,8] and a positive
// channel count, as the AVX2 kernel.
func TestEmitConv1dSSERefusesOutOfRangeShapes(t *testing.T) {
	for _, s := range [][2]int{{1, 64}, {9, 64}, {4, 0}, {4, -4}} {
		if _, err := EmitConv1dSSE(s[0], s[1]); err == nil {
			t.Errorf("taps=%d chans=%d was accepted", s[0], s[1])
		}
	}
}

func hybCallGate(c *Code, decay, beta, alpha, dt, b, a []float32, consts []float32, n int) {
	c.Call(&Args{
		Out: &decay[0], Out2: &beta[0],
		W:      (*byte)(unsafe.Pointer(&alpha[0])),
		AScale: &dt[0], AScale2: &b[0], Q32: &a[0],
		Scr:  (*byte)(unsafe.Pointer(&consts[0])),
		K:    int64(n / ElemLanes),
		Rows: int64(n % ElemLanes),
	})
}

// TestEmitDeltaGateSSEMatchesOracle gates both gates against the oracle over
// every length 1..40, with z spanning the branch the |z| identity removes
// (+-200 is where a wrong identity becomes Inf or 0).
//
// The bounds are exp's accuracy class (sigmoid inherits exp's relative error),
// not the AVX2 gate's constants, which the AVX2 kernel itself exceeds on this
// sweep: decay 2e-5, beta 4e-6, and on an AVX2 host the SSE worst must be
// within 10% of the AVX2 kernel's worst on the same inputs.
func TestEmitDeltaGateSSEMatchesOracle(t *testing.T) {
	b, err := EmittersFor(TierSSE).DeltaGate()
	if err != nil {
		t.Fatal(err)
	}
	c := hybMapSSE(t, "delta_gate_sse", b)
	defer c.Close()
	var avx *Code
	if HostTier() == TierAVX2 {
		avx = mustMap(t, EmitDeltaGate())
		defer avx.Close()
	}
	consts := DeltaGateConsts()
	zs := []float32{-200, -60, -20, -3, -0.5, 0, 0.5, 3, 20, 60, 200,
		1e-7, -1e-7, 12.5, -12.5, 87, -87, -1e-30, 1e-30}
	const boundD, boundB = 2e-5, 4e-6
	var sseD, sseB, avxD, avxB float64
	for n := 1; n <= 40; n++ {
		rnd := rand.New(rand.NewSource(int64(11 + n)))
		alpha, _ := hybGuarded(n)
		dt, _ := hybGuarded(n)
		bb, _ := hybGuarded(n)
		av, _ := hybGuarded(n)
		for i := 0; i < n; i++ {
			z := float32(rnd.NormFloat64() * 5)
			if i < len(zs) {
				z = zs[(i+n)%len(zs)]
			}
			dt[i] = float32(rnd.NormFloat64() * 0.5)
			alpha[i] = z - dt[i]
			bb[i] = float32(rnd.NormFloat64() * 4)
			// ssm_a is -exp(A_log) in the file, so it is always negative.
			av[i] = -float32(math.Exp(rnd.NormFloat64()))
		}
		decay, decayAll := hybGuarded(n)
		beta, betaAll := hybGuarded(n)
		hybCallGate(c, decay, beta, alpha, dt, bb, av, consts, n)
		hybCheckGuard(t, "delta_gate decay", decayAll, n)
		hybCheckGuard(t, "delta_gate beta", betaAll, n)

		wantD, wantB := make([]float64, n), make([]float64, n)
		oracle.DeltaGate(wantD, wantB, hybWiden(alpha), hybWiden(dt), hybWiden(bb), hybWiden(av))
		var da, ba []float32
		if avx != nil {
			da, ba = make([]float32, n), make([]float32, n)
			hybCallGate(avx, da, ba, alpha, dt, bb, av, consts, n)
		}
		for i := 0; i < n; i++ {
			// The shared exp clamps at -87.3 rather than returning zero (the
			// AVX2 gate's note): below it both answers mean "forgotten".
			if wantD[i] < 1e-37 {
				if !(decay[i] <= 1e-37) {
					t.Errorf("n=%d decay[%d]: got %v, want underflow", n, i, decay[i])
				}
			} else {
				d := hybRelErr(decay[i], wantD[i])
				if !(d <= boundD) {
					t.Errorf("n=%d decay[%d]: z=%v A=%v got %v want %v (rel %.2e, bound %.0e)",
						n, i, float64(alpha[i])+float64(dt[i]), av[i], decay[i], wantD[i], d, boundD)
				}
				sseD = math.Max(sseD, d)
				if da != nil {
					avxD = math.Max(avxD, hybRelErr(da[i], wantD[i]))
				}
			}
			d := hybRelErr(beta[i], wantB[i])
			if !(d <= boundB) {
				t.Errorf("n=%d beta[%d]: b=%v got %v want %v (rel %.2e, bound %.0e)", n, i, bb[i], beta[i], wantB[i], d, boundB)
			}
			sseB = math.Max(sseB, d)
			if ba != nil {
				avxB = math.Max(avxB, hybRelErr(ba[i], wantB[i]))
			}
		}
	}
	t.Logf("worst relative error against the oracle: SSE decay %.2e beta %.2e; AVX2 decay %.2e beta %.2e",
		sseD, sseB, avxD, avxB)
	if avx != nil {
		if sseD > 1.1*avxD {
			t.Errorf("decay: SSE worst %.2e is more than 10%% over the AVX2 kernel's %.2e", sseD, avxD)
		}
		if sseB > 1.1*avxB {
			t.Errorf("beta: SSE worst %.2e is more than 10%% over the AVX2 kernel's %.2e", sseB, avxB)
		}
	}
}

func hybRelErr(got float32, want float64) float64 {
	if want == 0 {
		return math.Abs(float64(got))
	}
	return math.Abs(float64(got)-want) / math.Abs(want)
}

// TestEveryHybridOpIsGeneratedSSE is TestEveryBlockOpIsGenerated's inventory
// (plus the deltagate row of the elementwise one) over the SSE table, and it
// calls each kernel, since a host maps VEX bytes it then faults on. The call
// is a smoke run on zeros; correctness is the gates above.
func TestEveryHybridOpIsGeneratedSSE(t *testing.T) {
	em := EmittersFor(TierSSE)
	type op struct {
		name string
		emit func() ([]byte, error)
		call func(c *Code)
	}
	delta := func(n int) op {
		return op{"gated_delta_" + strconv.Itoa(n), func() ([]byte, error) { return em.GatedDelta(n) }, func(c *Code) {
			st, o := make([]float32, n*n), make([]float32, n)
			k, q, v := make([]float32, n), make([]float32, n), make([]float32, n)
			hybCallDelta(c, o, st, k, q, v, 1, 1, n)
		}}
	}
	conv := func(taps, chans int) op {
		return op{"conv1d_" + strconv.Itoa(taps) + "x" + strconv.Itoa(chans), func() ([]byte, error) { return em.Conv1d(taps, chans) }, func(c *Code) {
			st, w := make([]float32, (taps-1)*chans), make([]float32, taps*chans)
			x := make([]float32, chans)
			hybCallConv(c, x, st, w, x)
		}}
	}
	ops := []op{delta(128), delta(8), conv(4, 8192), conv(4, 64),
		{"deltagate", em.DeltaGate, func(c *Code) {
			const n = 32
			d, b, al, dt, bb, a := make([]float32, n), make([]float32, n), make([]float32, n),
				make([]float32, n), make([]float32, n), make([]float32, n)
			hybCallGate(c, d, b, al, dt, bb, a, DeltaGateConsts(), n)
			if d[0] != 1 || b[0] != 0.5 {
				t.Errorf("deltagate on zeros: decay %v beta %v, want 1 and 0.5", d[0], b[0])
			}
		}},
	}
	for _, o := range ops {
		b, err := o.emit()
		if err != nil {
			t.Fatalf("%s: %v", o.name, err)
		}
		c := hybMapSSE(t, o.name, b)
		o.call(c)
		c.Close()
	}
	t.Logf("%d hybrid ops generated, VEX-gated and executed on the SSE tier", len(ops))
}
