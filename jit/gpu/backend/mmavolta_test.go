package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestMMAVoltaProbeMatchesTheHost runs sm_70's m8n8k4 through the IR on every
// backend that lowers it (PTX; the others refuse fragment MMA by name) and
// holds each quad-pair's 8x8 product to a CPU one. Small integers, so the f16
// operands and f32 sums are exact and the compare is equality.
func TestMMAVoltaProbeMatchesTheHost(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(3))
	A, B := make([]float32, 32*4), make([]float32, 4*32)
	for i := range A {
		A[i] = float32(rng.Intn(33) - 16)
	}
	for i := range B {
		B[i] = float32(rng.Intn(33) - 16)
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		kern, err := d.Compile(mainkernels.MMAVoltaProbe())
		if err != nil {
			t.Logf("%s: %v", d.API(), err)
			continue
		}
		ran++
		g := newGPU(t, d)
		bA, bB, out := g.up(f32bytes(A)), g.up(f32bytes(B)), g.up(f32bytes(nanFill(32*8)))
		if err := kern.Launch(1, 32, bA, bB, out); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, 32*8*4)
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		for q := 0; q < 4; q++ {
			for r := 0; r < 8; r++ {
				for c := 0; c < 8; c++ {
					var want float32
					for k := 0; k < 4; k++ {
						want += A[(8*q+r)*4+k] * B[k*32+8*q+c]
					}
					got := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*((8*q+r)*8+c):]))
					if got != want {
						t.Fatalf("%s: quad %d row %d col %d: %v, want %v", d.API(), q, r, c, got, want)
					}
				}
			}
		}
		kern.Close()
		g.free()
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

// TestMatVecMMA70MatchesTheReference holds the sm_70 tensor-core matvec to a
// float64 dot of the dequantized GGUF weights with the float activations. Its
// operands are f16 (weights rounded once in the dequantizing fma, activations
// by ActF16), so the bound is f16's, and it must be no worse than the dp4a
// kernel's -- whose int8 activations are the path it replaces -- on the same
// inputs, which the test measures beside it.
func TestMatVecMMA70MatchesTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_0, kernels.Q5_1} {
			for _, sh := range []struct {
				rows, k, ntok, mt, nt, split int
				bias                         bool
			}{
				{64, 256, 8, 1, 1, 1, false}, {128, 512, 16, 2, 1, 1, true}, {256, 512, 32, 2, 2, 1, false},
				{64, 1024, 16, 1, 2, 1, false}, {64, 1024, 16, 2, 2, 4, true}, {128, 512, 32, 2, 4, 2, false},
			} {
				s := mainkernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, MT: sh.mt, NT: sh.nt,
					Split: sh.split, Bias: sh.bias}
				kk, err := mainkernels.MatVecMMA70(s)
				if err != nil {
					t.Fatalf("%v %+v: %v", q, sh, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					// Only PTX lowers the shape; anywhere else a refusal is the
					// expected answer. On PTX it is a broken kernel, not a skip.
					if d.API() == "ptx" {
						t.Fatalf("%s %+v: %v", d.API(), sh, err)
					}
					t.Logf("%s: %v", d.API(), err)
					break
				}
				ran++
				warps := (s.Rows / (32 * max(s.MT, 1))) * (s.NTok / (8 * max(s.NT, 1))) * max(s.Split, 1)
				mma70Case(t, d, kern, s, "mma70", (warps*32+127)/128, 128, false)
				kern.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

// ggufOf is each device format's source type, derived from quant.PackedTypes
// rather than written out: a hand-written copy left a new format mapping to
// the zero Type, which is F32, and Dequant32 then "decoded" packed blocks as
// floats.
var ggufOf = func() map[kernels.Quant]quant.Type {
	m := map[kernels.Quant]quant.Type{}
	for _, t := range quant.PackedTypes {
		if q, ok := kernels.QuantOf(t); ok {
			m[q] = t
		}
	}
	return m
}()

// mma70Case runs kern -- an sm_70 tensor-core matvec reading ActF16's
// activations, launched as groups workgroups of width -- against the float64
// reference and the dp4a kernel on the same inputs.
//
// extra launches run after kern on the same parameters: a tail launch that
// covers the tokens kern's block does not.
func mma70Case(t *testing.T, d backend.Device, kern backend.Kernel, s mainkernels.MatVecShape, what string, groups, width int, tokMajor bool,
	extra ...func(args ...backend.Buf) error) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(s.Rows*7 + s.K + s.NTok)))
	g := newGPU(t, d)
	defer g.free()
	raw := rawWeights(s.T, s.Rows, s.K, rng)
	qs, dw, scw, err := kernels.PackWeights(s.T, raw, s.Rows, s.K)
	if err != nil {
		t.Fatal(err)
	}
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	wf := make([]float32, s.Rows*s.K)
	if err := quant.Dequant32(ggufOf[s.T], raw, wf); err != nil {
		t.Fatal(err)
	}
	x := make([]float32, s.NTok*s.K)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	n := s.Rows * s.NTok
	bias := make([]float32, s.Rows)
	for i := range bias {
		bias[i] = float32(rng.NormFloat64())
	}
	ref := make([]float64, n)
	for tk := 0; tk < s.NTok; tk++ {
		for r := 0; r < s.Rows; r++ {
			var acc float64
			if s.Bias {
				acc = float64(bias[r])
			}
			for k := 0; k < s.K; k++ {
				acc += float64(wf[r*s.K+k]) * float64(x[tk*s.K+k])
			}
			ref[tk*s.Rows+r] = acc
		}
	}
	av, as, asum, err := kernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}
	bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
	bA, bAX := g.up(u32bytes(av)), g.up(f32bytes(append(append([]float32{}, as...), asum...)))
	bX := g.up(f32bytes(x))
	nmseOf := func(buf backend.Buf, what string, split int) float64 {
		parts := make([]byte, split*n*4)
		if err := buf.Read(parts); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, n*4)
		for i := 0; i < n; i++ {
			var v float32
			for sg := 0; sg < split; sg++ {
				v += math.Float32frombits(binary.LittleEndian.Uint32(parts[4*(sg*n+i):]))
			}
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		var se, sy float64
		for i := 0; i < n; i++ {
			v := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("%s %s %v: element %d (token %d row %d) is %v -- never written",
					d.API(), what, s.T, i, i/s.Rows, i%s.Rows, v)
			}
			se += (v - ref[i]) * (v - ref[i])
			sy += ref[i] * ref[i]
		}
		return se / sy
	}
	// The dp4a kernel on int8 activations: the path this replaces.
	rk0, err := mainkernels.MatVec(mainkernels.MatVecShape{T: s.T, K: s.K, Rows: s.Rows, NTok: s.NTok, Tok: 8, Bias: s.Bias})
	if err != nil {
		t.Fatal(err)
	}
	rk, err := d.Compile(rk0)
	if err != nil {
		t.Fatal(err)
	}
	defer rk.Close()
	dp := g.up(f32bytes(nanFill(n)))
	th := s.Rows * s.NTok / 8
	bBias := g.up(f32bytes(bias))
	dpArgs := []backend.Buf{bQS, bD, bSC, bA, bAX, dp}
	if s.Bias {
		dpArgs = append(dpArgs, bBias)
	}
	if err := rk.Launch((th+127)/128, 128, dpArgs...); err != nil {
		t.Fatal(err)
	}
	// GemmVolta reads the token-major layout (ActF16T), MatVecMMA70 the
	// step-major one; a kernel handed the other would read every activation
	// from the wrong place, which is what this comparison would catch.
	ak, err := mainkernels.ActF16(s.T, s.NTok, s.K)
	at := s.NTok * s.K / 4
	if tokMajor {
		sub, _, _, _ := kernels.Layout(s.T)
		ak, err = mainkernels.ActF16T(s.T, s.NTok, s.K)
		at = s.NTok * s.K / sub
	}
	if err != nil {
		t.Fatal(err)
	}
	ac, err := d.Compile(ak)
	if err != nil {
		t.Fatal(err)
	}
	defer ac.Close()
	bB := g.up(make([]byte, s.NTok*s.K*2))
	if err := ac.Launch((at+127)/128, 128, bX, bB); err != nil {
		t.Fatal(err)
	}
	split := max(s.Split, 1)
	got := g.up(f32bytes(nanFill(split * n)))
	args := []backend.Buf{bQS, bD, bSC, bB, got}
	if s.Bias {
		args = append(args, bBias)
	}
	if err := kern.Launch(groups, width, args...); err != nil {
		t.Fatal(err)
	}
	for _, f := range extra {
		if err := f(args...); err != nil {
			t.Fatal(err)
		}
	}
	nmDp, nmMM := nmseOf(dp, "dp4a", 1), nmseOf(got, what, split)
	name := fmt.Sprintf("%s %s %v %dx%d ntok %d mt %d nt %d split %d bias %v", d.API(), what, s.T, s.Rows, s.K,
		s.NTok, s.MT, s.NT, split, s.Bias)
	t.Logf("%s: NMSE %.3e (dp4a on int8 activations %.3e)", name, nmMM, nmDp)
	if nmMM > 1e-5 || nmMM > 2*nmDp {
		t.Fatalf("%s: NMSE %.3e against the reference, dp4a %.3e", name, nmMM, nmDp)
	}
}

// TestMMA70Speed times the sm_70 tensor-core matvec against the dp4a batched
// kernel on one shape (JITLLM_MMA70_BENCH=1). A probe, not a gate.
func TestMMA70Speed(t *testing.T) {
	if os.Getenv("JITLLM_MMA70_BENCH") == "" {
		t.Skip("set JITLLM_MMA70_BENCH=1")
	}
	gpuLock(t)
	devs := backend.Open()
	for _, d := range devs {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		const rows, k, ntok = 14336, 4096, 128
		rng := rand.New(rand.NewSource(1))
		g := newGPU(t, d)
		qs, dw, scw, _ := kernels.PackWeights(kernels.Q4_K, rawWeights(kernels.Q4_K, rows, k, rng), rows, k)
		x := make([]float32, ntok*k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		av, as, asum, _ := kernels.PackActivations(x)
		bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
		bA, bAX := g.up(u32bytes(av)), g.up(f32bytes(append(append([]float32{}, as...), asum...)))
		out := g.up(make([]byte, rows*ntok*4))
		bB := g.up(make([]byte, ntok*k*2))
		sync := func() { out.Read(make([]byte, 4)) }
		time1 := func(name string, f func()) {
			f()
			sync()
			const n = 20
			t0 := time.Now()
			for i := 0; i < n; i++ {
				f()
			}
			sync()
			el := time.Since(t0) / n
			t.Logf("%-22s %8.1f us  %.2f TMAC/s", name, float64(el.Microseconds()),
				float64(rows)*k*ntok/el.Seconds()/1e12)
		}
		for _, tk := range []int{16} {
			ref, _ := mainkernels.MatVec(mainkernels.MatVecShape{T: kernels.Q4_K, K: k, Rows: rows, NTok: ntok, Tok: tk, Rowt: 4})
			rk, err := d.Compile(ref)
			if err != nil {
				t.Fatal(err)
			}
			th := rows / 4 * ntok / tk
			time1("dp4a tok16 rowt4", func() { rk.Launch((th+127)/128, 128, bQS, bD, bSC, bA, bAX, out) })
		}
		ak, _ := mainkernels.ActF16(kernels.Q4_K, ntok, k)
		ac, _ := d.Compile(ak)
		at := ntok * k / 4
		bX := g.up(f32bytes(x))
		time1("actf16", func() { ac.Launch((at+127)/128, 128, bX, bB) })
		for _, tl := range [][2]int{{2, 2}, {2, 4}, {4, 2}, {4, 4}, {2, 8}, {1, 8}, {4, 8}, {8, 2}} {
			kk, err := mainkernels.MatVecMMA70(mainkernels.MatVecShape{T: kernels.Q4_K, K: k, Rows: rows, NTok: ntok, MT: tl[0], NT: tl[1]})
			if err != nil {
				t.Log(err)
				continue
			}
			kern, err := d.Compile(kk)
			if err != nil {
				t.Log(err)
				continue
			}
			warps := rows / (32 * tl[0]) * ntok / (8 * tl[1])
			time1(fmt.Sprintf("mma70 mt%d nt%d", tl[0], tl[1]), func() {
				kern.Launch((warps*32+127)/128, 128, bQS, bD, bSC, bB, out)
			})
		}
		g.free()
	}
}
