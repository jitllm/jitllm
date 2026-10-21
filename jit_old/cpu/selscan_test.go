//go:build amd64 || arm64

package cpu_test

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// selScanCase is one token of random inputs for a Mamba-1 scan of rows
// channels and state width n, with the decay rates spread over a decade and
// dt over the range a softplus leaves it in.
type selScanCase struct {
	b, c, x, dt, a, d []float32
}

func newSelScanCase(rnd *rand.Rand, rows, n int) selScanCase {
	k := selScanCase{b: make([]float32, n), c: make([]float32, n), x: make([]float32, rows),
		dt: make([]float32, rows), a: make([]float32, rows*n), d: make([]float32, rows)}
	for i := 0; i < n; i++ {
		k.b[i] = float32(rnd.NormFloat64() * 0.3)
		k.c[i] = float32(rnd.NormFloat64() * 0.3)
	}
	for j := 0; j < rows; j++ {
		k.x[j] = float32(rnd.NormFloat64())
		k.dt[j] = float32(0.01 + 0.5*rnd.Float64())
		k.d[j] = float32(rnd.NormFloat64())
	}
	for i := range k.a {
		k.a[i] = -float32(0.1 + 10*rnd.Float64())
	}
	return k
}

func (k selScanCase) call(code *cpu.Code, o, st []float32, consts []float32) {
	code.Call(&cpu.Args{Out: &o[0], W: (*byte)(unsafe.Pointer(&st[0])), AScale: &k.b[0],
		Rows: int64(len(k.x)), Scr: (*byte)(unsafe.Pointer(&consts[0])),
		AHalf: &k.a[0], Q32: &k.c[0], Q2: &k.x[0], AScale2: &k.dt[0],
		PD: (*byte)(unsafe.Pointer(&k.d[0]))})
}

func (k selScanCase) want(st []float32) (o, state []float64) {
	state = ssdWiden(st)
	o = make([]float64, len(k.x))
	oracle.SelScan(o, state, ssdWiden(k.b), ssdWiden(k.c), ssdWiden(k.x), ssdWiden(k.dt),
		ssdWiden(k.a), ssdWiden(k.d))
	return o, state
}

// TestEmitSelScanMatchesOracle steps four tokens of Mamba-1's selective scan
// through the generated kernel at ragged state widths and channel counts, and
// holds the output and the STATE after each to oracle.SelScan stepped from
// the kernel's own state. A decay read off the wrong channel's A row, the
// skip dropped, or the decay taken as one scalar a channel is each a
// violation the gate must see; the last two are run below.
func TestEmitSelScanMatchesOracle(t *testing.T) {
	consts := cpu.DeltaGateConsts()
	for _, sh := range ssdShapes {
		n, rows := sh[0], sh[1]
		code := onHost(t)(hostTable().SelScan(n))
		rnd := rand.New(rand.NewSource(int64(97*n + rows)))
		st := make([]float32, rows*n)
		for i := range st {
			st[i] = float32(rnd.NormFloat64())
		}
		for tok := 0; tok < 4; tok++ {
			k := newSelScanCase(rnd, rows, n)
			wantO, wantSt := k.want(st)
			o := make([]float32, rows+1)
			o[rows] = -1234.5
			k.call(code, o, st, consts)
			if o[rows] != -1234.5 {
				t.Fatalf("n=%d rows=%d: wrote past the output", n, rows)
			}
			if e := ssdNMSE(o[:rows], wantO); e > 1e-10 {
				t.Errorf("n=%d rows=%d tok %d: output NMSE %.3e", n, rows, tok, e)
			}
			if e := ssdNMSE(st, wantSt); e > 1e-10 {
				t.Errorf("n=%d rows=%d tok %d: STATE NMSE %.3e", n, rows, tok, e)
			}
		}
		code.Close()
	}
}

// TestSelScanGateDiscriminates runs the oracle comparison above against two
// wrong inputs: the skip zeroed, and every channel's decay row replaced by
// its first entry (Mamba-2's one scalar a head). Both must land far past the
// bound the kernel is held to.
func TestSelScanGateDiscriminates(t *testing.T) {
	const n, rows = 16, 9
	rnd := rand.New(rand.NewSource(5))
	k := newSelScanCase(rnd, rows, n)
	st := make([]float32, rows*n)
	for i := range st {
		st[i] = float32(rnd.NormFloat64())
	}
	wantO, _ := k.want(st)
	noD := k
	noD.d = make([]float32, rows)
	o, _ := noD.want(st)
	if e := ssdNMSE(f32s(o), wantO); !(e > 1e-4) {
		t.Errorf("the D skip is invisible to this gate (%.3e)", e)
	}
	scalar := k
	scalar.a = make([]float32, rows*n)
	for j := 0; j < rows; j++ {
		for i := 0; i < n; i++ {
			scalar.a[j*n+i] = k.a[j*n]
		}
	}
	o, _ = scalar.want(st)
	if e := ssdNMSE(f32s(o), wantO); !(e > 1e-4) {
		t.Errorf("a scalar decay a channel is invisible to this gate (%.3e)", e)
	}
}

func f32s(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}
