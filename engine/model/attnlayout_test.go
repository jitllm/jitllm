package model

import (
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestAttnLayoutProbe prices a head-major KV cache layout against row-major.
// Attention kernels read contiguous head-dimension floats, but whether that
// layout difference matters depends on the access pattern. The test compares
// two already-resident layouts with production features (append, prefill writes)
// excluded.
//
// Opt-in: set JITLLM_LAYOUT_PROBE=1.
func TestAttnLayoutProbe(t *testing.T) {
	if os.Getenv("JITLLM_LAYOUT_PROBE") == "" {
		t.Skip("set JITLLM_LAYOUT_PROBE=1")
	}
	const nHead, nKV, hd = 32, 4, 64 // tinyllama
	const gqa = nHead / nKV
	kvDim := nKV * hd

	for _, npos := range []int{1, 513, 1537} {
		cap := npos + 64 // capacity distinct from the live length, on purpose
		rng := rand.New(rand.NewSource(int64(npos) * 7919))

		// Row-major [position][kvHead][dim] and head-major [kvHead][position][dim],
		// filled with the same values so any difference is the layout.
		rowK := make([]float32, cap*kvDim)
		rowV := make([]float32, cap*kvDim)
		for i := range rowK {
			rowK[i] = float32(rng.NormFloat64())
			rowV[i] = float32(rng.NormFloat64())
		}
		hedK := make([]float32, nKV*cap*hd)
		hedV := make([]float32, nKV*cap*hd)
		for kvh := 0; kvh < nKV; kvh++ {
			for t := 0; t < cap; t++ {
				copy(hedK[kvh*cap*hd+t*hd:], rowK[t*kvDim+kvh*hd:t*kvDim+(kvh+1)*hd])
				copy(hedV[kvh*cap*hd+t*hd:], rowV[t*kvDim+kvh*hd:t*kvDim+(kvh+1)*hd])
			}
		}
		q := make([]float32, nHead*hd)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		w := make([]float32, nHead*cap)
		for i := range w {
			w[i] = float32(rng.Float64())
		}

		mk := func(stride int) (sc, ac *cpu.Code) {
			b, err := cpu.EmitAttnScores2(hd, stride, cpu.KVF32)
			if err != nil {
				t.Fatal(err)
			}
			if sc, err = cpu.Map(b); err != nil {
				t.Fatal(err)
			}
			if b, err = cpu.EmitAttnAcc2(hd, stride, cpu.KVF32); err != nil {
				t.Fatal(err)
			}
			if ac, err = cpu.Map(b); err != nil {
				t.Fatal(err)
			}
			return sc, ac
		}
		rowSc, rowAc := mk(kvDim)
		hedSc, hedAc := mk(hd)

		// Through the real pool: one thread walking all heads makes row-major
		// look far worse than six workers reading four kv heads at once.
		pool := nn.NewJIT(2048, 5632, nil)
		defer pool.Close()
		pass := func(sc, ac *cpu.Code, k, v []float32, base func(kvh int) int,
			scores, out []float32) {
			pool.Parallel(nHead/2, 1, func(plo, phi int) {
				for p := plo; p < phi; p++ {
					hh := p * 2
					kvh := hh / gqa
					s0 := scores[hh*cap : hh*cap+npos]
					s1 := scores[(hh+1)*cap : (hh+1)*cap+npos]
					a := cpu.Args{Out: &s0[0], W: f32b(k[base(kvh):]), Rows: int64(npos),
						Q32: &q[hh*hd], Out2: &s1[0], Q2: &q[(hh+1)*hd]}
					sc.Call(&a)
					o0 := out[hh*hd : (hh+1)*hd]
					o1 := out[(hh+1)*hd : (hh+2)*hd]
					b := cpu.Args{Out: &o0[0], W: f32b(v[base(kvh):]), AScale: &w[hh*cap],
						Rows: int64(npos), Out2: &o1[0], AScale2: &w[(hh+1)*cap]}
					ac.Call(&b)
				}
			})
		}
		rowBase := func(kvh int) int { return kvh * hd }
		hedBase := func(kvh int) int { return kvh * cap * hd }

		sRow := make([]float32, nHead*cap)
		sHed := make([]float32, nHead*cap)
		oRow := make([]float32, nHead*hd)
		oHed := make([]float32, nHead*hd)
		pass(rowSc, rowAc, rowK, rowV, rowBase, sRow, oRow)
		pass(hedSc, hedAc, hedK, hedV, hedBase, sHed, oHed)

		// Correctness before timing: the kernel sums positions in the same order
		// in both layouts, so the results must be bit-identical.
		for i := range sRow {
			if math.Float32bits(sRow[i]) != math.Float32bits(sHed[i]) {
				t.Fatalf("npos=%d: score %d differs between layouts: %v vs %v",
					npos, i, sRow[i], sHed[i])
			}
		}
		for i := range oRow {
			if math.Float32bits(oRow[i]) != math.Float32bits(oHed[i]) {
				t.Fatalf("npos=%d: output %d differs between layouts: %v vs %v",
					npos, i, oRow[i], oHed[i])
			}
		}

		// And over a rotating working set, which is the one that decides: real
		// decode walks every layer's cache per token with the weights streaming
		// past, so nothing stays in L3. Both are reported.
		const nLayer = 22
		rotRow := make([][]float32, 0, 2*nLayer)
		rotHed := make([][]float32, 0, 2*nLayer)
		for l := 0; l < nLayer; l++ {
			rk := append([]float32(nil), rowK...)
			rv := append([]float32(nil), rowV...)
			hk := append([]float32(nil), hedK...)
			hv := append([]float32(nil), hedV...)
			rotRow = append(rotRow, rk, rv)
			rotHed = append(rotHed, hk, hv)
		}

		run := func(rows bool, rotate bool, iters int) time.Duration {
			t0 := time.Now()
			for i := 0; i < iters; i++ {
				l := 0
				if rotate {
					l = (i % nLayer) * 2
				}
				if rows {
					k, v := rowK, rowV
					if rotate {
						k, v = rotRow[l], rotRow[l+1]
					}
					pass(rowSc, rowAc, k, v, rowBase, sRow, oRow)
				} else {
					k, v := hedK, hedV
					if rotate {
						k, v = rotHed[l], rotHed[l+1]
					}
					pass(hedSc, hedAc, k, v, hedBase, sHed, oHed)
				}
			}
			return time.Since(t0)
		}

		const rounds, iters = 15, 20
		for _, rot := range []bool{false, true} {
			var rs []float64
			for r := 0; r < rounds; r++ {
				ta := run(true, rot, iters)
				tb := run(false, rot, iters)
				tb2 := run(false, rot, iters)
				ta2 := run(true, rot, iters)
				rs = append(rs, float64(ta+ta2)/float64(tb+tb2))
			}
			med, iqr := medIQR(rs)
			verdict, what := "", "hot single buffer"
			if rot {
				what = "rotating 22 layers"
			}
			if iqr/med > 0.10 {
				verdict = "  UNSTABLE -- do not quote"
			}
			t.Logf("npos=%4d  %-18s  head-major vs row-major: %.4fx  (IQR/median %.1f%%, n=%d)%s",
				npos, what, med, 100*iqr/med, len(rs), verdict)
		}

		// A/A control: the same arm against itself; far from 1.0 means the
		// harness is drifting.
		var aa []float64
		for r := 0; r < rounds; r++ {
			x := run(true, true, iters)
			y := run(true, true, iters)
			aa = append(aa, float64(x)/float64(y))
		}
		am, ai := medIQR(aa)
		t.Logf("npos=%4d  A/A control:        %.4fx  (IQR/median %.1f%%)", npos, am, 100*ai/am)

		rowSc.Close()
		rowAc.Close()
		hedSc.Close()
		hedAc.Close()
	}
}

func medIQR(v []float64) (med, iqr float64) {
	s := append([]float64(nil), v...)
	for i := range s {
		for j := i + 1; j < len(s); j++ {
			if s[j] < s[i] {
				s[i], s[j] = s[j], s[i]
			}
		}
	}
	med = s[len(s)/2]
	return med, s[3*len(s)/4] - s[len(s)/4]
}

func f32b(x []float32) *byte { return (*byte)(unsafe.Pointer(&x[0])) }
