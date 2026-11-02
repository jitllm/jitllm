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

// TestEmitGatedSSDSSEMatchesOracle is TestEmitGatedSSDMatchesOracle for the
// SSE tier: through the VEX-leak gate, on misaligned guarded buffers, every
// ragged width with rows unequal to it, four tokens stepped from the kernel's
// own state.
func TestEmitGatedSSDSSEMatchesOracle(t *testing.T) {
	em := EmittersFor(TierSSE)
	for _, n := range hybWidths {
		rows := 1 + n%9
		b, err := em.GatedSSD(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		c := hybMapSSE(t, "gated_ssd_sse_"+strconv.Itoa(n), b)
		rnd := rand.New(rand.NewSource(int64(7000 + n)))
		st, stAll := hybGuarded(rows * n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		for tok := 0; tok < 4; tok++ {
			o, oAll := hybGuarded(rows)
			k, _ := hybGuarded(n)
			q, _ := hybGuarded(n)
			v, _ := hybGuarded(rows)
			for i := 0; i < n; i++ {
				k[i] = float32(rnd.NormFloat64() * 0.3)
				q[i] = float32(rnd.NormFloat64() * 0.3)
			}
			for j := range v {
				v[j] = float32(rnd.NormFloat64())
			}
			decay, dt, d := float32(0.7), float32(0.3+rnd.Float64()), float32(rnd.NormFloat64())
			wantSt := hybWiden(st)
			wantO := make([]float64, rows)
			oracle.GatedSSD(wantO, wantSt, hybWiden(k), hybWiden(q), hybWiden(v), float64(decay), float64(dt), float64(d))
			sc := [3]float32{decay, dt, d}
			c.Call(&Args{Out: &o[0], W: (*byte)(unsafe.Pointer(&st[0])), AScale: &k[0],
				Rows: int64(rows), Scr: (*byte)(unsafe.Pointer(&sc[0])), Q32: &q[0], Q2: &v[0]})
			hybCheckGuard(t, "o", oAll, rows)
			hybCheckGuard(t, "state", stAll, rows*n)
			if e := hybNMSE(o, wantO); e > 1e-10 {
				t.Errorf("n=%d rows=%d tok %d: output NMSE %.3e", n, rows, tok, e)
			}
			if e := hybNMSE(st, wantSt); e > 1e-12 {
				t.Errorf("n=%d rows=%d tok %d: STATE NMSE %.3e", n, rows, tok, e)
			}
		}
		c.Close()
	}
}

// TestEmitSSDGateSSEMatchesOracle is the SSE tier's dt and decay against
// oracle.SSDGate, through the VEX-leak gate.
func TestEmitSSDGateSSEMatchesOracle(t *testing.T) {
	b, err := EmittersFor(TierSSE).SSDGate()
	if err != nil {
		t.Fatal(err)
	}
	c := hybMapSSE(t, "ssd_gate_sse", b)
	defer c.Close()
	consts := DeltaGateConsts()
	for _, n := range hybWidths {
		rnd := rand.New(rand.NewSource(int64(n)))
		alpha, _ := hybGuarded(n)
		bias, _ := hybGuarded(n)
		a, _ := hybGuarded(n)
		for i := 0; i < n; i++ {
			alpha[i] = float32(rnd.NormFloat64() * 6)
			bias[i] = float32(rnd.NormFloat64() * 3)
			a[i] = -float32(math.Exp(rnd.NormFloat64()))
		}
		decay, dAll := hybGuarded(n)
		dt, tAll := hybGuarded(n)
		c.Call(&Args{Out: &decay[0], Out2: &dt[0], W: (*byte)(unsafe.Pointer(&alpha[0])),
			AScale: &bias[0], Q32: &a[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
			K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
		hybCheckGuard(t, "decay", dAll, n)
		hybCheckGuard(t, "dt", tAll, n)
		wd, wt := make([]float64, n), make([]float64, n)
		oracle.SSDGate(wd, wt, hybWiden(alpha), hybWiden(bias), hybWiden(a))
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

// TestEmitSelScanSSEMatchesOracle is TestEmitSelScanMatchesOracle for the SSE
// tier: through the VEX-leak gate, on guarded buffers, every ragged width,
// four tokens stepped from the kernel's own state.
func TestEmitSelScanSSEMatchesOracle(t *testing.T) {
	em := EmittersFor(TierSSE)
	consts := DeltaGateConsts()
	for _, n := range hybWidths {
		rows := 1 + n%9
		b, err := em.SelScan(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		c := hybMapSSE(t, "sel_scan_sse_"+strconv.Itoa(n), b)
		rnd := rand.New(rand.NewSource(int64(9000 + n)))
		st, stAll := hybGuarded(rows * n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		for tok := 0; tok < 4; tok++ {
			o, oAll := hybGuarded(rows)
			bv, _ := hybGuarded(n)
			cv, _ := hybGuarded(n)
			x, _ := hybGuarded(rows)
			dt, _ := hybGuarded(rows)
			d, _ := hybGuarded(rows)
			av, _ := hybGuarded(rows * n)
			for i := 0; i < n; i++ {
				bv[i] = float32(rnd.NormFloat64() * 0.3)
				cv[i] = float32(rnd.NormFloat64() * 0.3)
			}
			for j := 0; j < rows; j++ {
				x[j] = float32(rnd.NormFloat64())
				dt[j] = float32(0.01 + 0.5*rnd.Float64())
				d[j] = float32(rnd.NormFloat64())
			}
			for i := range av {
				av[i] = -float32(0.1 + 10*rnd.Float64())
			}
			wantSt := hybWiden(st)
			wantO := make([]float64, rows)
			oracle.SelScan(wantO, wantSt, hybWiden(bv), hybWiden(cv), hybWiden(x), hybWiden(dt),
				hybWiden(av), hybWiden(d))
			c.Call(&Args{Out: &o[0], W: (*byte)(unsafe.Pointer(&st[0])), AScale: &bv[0],
				Rows: int64(rows), Scr: (*byte)(unsafe.Pointer(&consts[0])), AHalf: &av[0],
				Q32: &cv[0], Q2: &x[0], AScale2: &dt[0], PD: (*byte)(unsafe.Pointer(&d[0]))})
			hybCheckGuard(t, "o", oAll, rows)
			hybCheckGuard(t, "state", stAll, rows*n)
			// 1e-9 where the AVX2 kernel holds 1e-10: no FMA, and a one-channel
			// output is a single sum with nothing to average its rounding.
			if e := hybNMSE(o, wantO); e > 1e-9 {
				t.Errorf("n=%d rows=%d tok %d: output NMSE %.3e", n, rows, tok, e)
			}
			if e := hybNMSE(st, wantSt); e > 1e-10 {
				t.Errorf("n=%d rows=%d tok %d: STATE NMSE %.3e", n, rows, tok, e)
			}
		}
		c.Close()
	}
}
