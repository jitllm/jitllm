//go:build (amd64 || arm64) && jitllmtest

package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The matvec family (family 4) at the nn level, on the forced SSE tier: every
// packed path nn has -- the fused kernel, the tile and tail, the window for the
// last few rows, the batched mixture dispatch, the batched prefill -- and the
// float matvec, through the JIT a real host of that tier would build.
//
// The configuration is checked, not assumed: each arm asserts the probe
// reports SSE, the JIT recorded the SSE tier, SSE kernels were mapped and no
// AVX2 kernel was, since a leaked AVX2 kernel would pass every numeric check.

// underSSE runs f with the SSE tier forced, and fails if the force did not
// reach the probe, if f mapped no SSE kernel, or if it mapped an AVX2 one.
func underSSE(t *testing.T, f func()) {
	t.Helper()
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Skipf("the tier could not be forced to SSE on this build (host tier %v)", cpu.HostTier())
	}
	before := cpu.MappedByTier()
	f()
	after := cpu.MappedByTier()
	if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped under the forced SSE tier", d)
	}
	if after[cpu.TierSSE] == before[cpu.TierSSE] {
		t.Errorf("no SSE-tier kernel was mapped -- the arm did not run the tier it names")
	}
}

// TestSSETierRunsTheMatvecGates re-runs nn's own matvec gates, unchanged, on
// the forced SSE tier: the same assertions (bit equality of the batched paths
// against the loop, the second call into the same buffer, the
// longer activation), over the SSE kernels. (The router selection gate
// lives in package nn_test and cannot be called from here; the float matvec
// it rides on is compared tier against tier in TestSSETierTracksTheHostTier.)
func TestSSETierRunsTheMatvecGates(t *testing.T) {
	gates := []struct {
		name string
		fn   func(*testing.T)
	}{
		{"PackedMatVecTailAgrees", TestPackedMatVecTailAgrees},
		{"PackedFamilyIsBuiltOnThisHost", TestPackedFamilyIsBuiltOnThisHost},
		{"MatVecPackedMultiMatchesTheLoopExactly", TestMatVecPackedMultiMatchesTheLoopExactly},
		{"MatVecPackedMultiDeclinesRatherThanGuessing", TestMatVecPackedMultiDeclinesRatherThanGuessing},
		{"MatVecPackedGatherMatchesTheLoopExactly", TestMatVecPackedGatherMatchesTheLoopExactly},
		{"MatVecPackedGatherKeepsTheSingleActivationCache", TestMatVecPackedGatherKeepsTheSingleActivationCache},
		{"MatMulPackedMatchesTheRowLoopExactly", TestMatMulPackedMatchesTheRowLoopExactly},
		{"MatMulPackedDeclinesRatherThanApproximating", TestMatMulPackedDeclinesRatherThanApproximating},
		{"MatVecPackedTakesALongerActivation", TestMatVecPackedTakesALongerActivation},
	}
	underSSE(t, func() {
		for _, g := range gates {
			t.Run(g.name, g.fn)
		}
	})
}

// TestSSETierHasNoTokenTileAndStillBatches: the SSE table refuses the tiled
// kernel by decision, so TiledWidth is 0 there -- and MatMulPacked still
// takes the batch, finishing every token with the fused kernel.
func TestSSETierHasNoTokenTileAndStillBatches(t *testing.T) {
	underSSE(t, func() {
		for _, gt := range quant.PackedTypes {
			if !cpu.PackedFusedSupported(gt) {
				t.Errorf("%s: no fused kernel on the SSE tier, so the batched mixture dispatch declines", gt)
			}
			const k, nrows, ntok = 512, 128, 3
			f := NewJIT(k, nrows, []quant.Type{gt}, WithTune(TuneOff), WithQuietTuner(true))
			if f.tier != cpu.TierSSE {
				t.Fatalf("%s: the JIT recorded tier %v", gt, f.tier)
			}
			if w := f.TiledWidth(gt, k, nrows); w != 0 {
				t.Errorf("%s: TiledWidth %d on the SSE tier, which has no tiled kernel", gt, w)
			}
			pk := packOne(t, gt, nrows, k)
			x := make([]float32, ntok*k)
			copy(x, prevnniAct(k))
			copy(x[k:], prevnniAct(k+1))
			copy(x[2*k:], prevnniAct(k+2))
			out := make([]float32, ntok*nrows)
			if !f.MatMulPacked(out, gt, pk, x, nrows, k, ntok) {
				t.Errorf("%s: MatMulPacked declined on the SSE tier", gt)
			}
			f.Close()
		}
	})
}

// TestSSETierTracksTheHostTier runs MatVecPacked through an AVX2 JIT and a
// forced-SSE JIT on the same weights, at shapes that reach every packed path
// -- the fused kernel (256 rows), the tile + tail (200 rows), the tile + tail +
// window (203 rows) -- and the float matvec, and bounds the difference by
// what the missing FMA costs (see cpu.TestSSEPackedTracksAVX2OnRealScales),
// with both held to the oracle as well.
func TestSSETierTracksTheHostTier(t *testing.T) {
	if cpu.HostTier() == cpu.TierSSE {
		t.Skip("this host IS the SSE tier: there is no second tier to compare against")
	}
	const bound = 1e-6
	shapes := []struct{ rows, k int }{{256, 512}, {200, 512}, {203, 512}, {96, 1024}}
	for _, gt := range quant.PackedTypes {
		for _, sh := range shapes {
			if sh.k%int(gt.BlockElems()) != 0 {
				continue
			}
			pk := packOne(t, gt, sh.rows, sh.k)
			x := prevnniAct(sh.k)
			host := make([]float32, sh.rows)
			fh := NewJIT(sh.k, sh.rows, []quant.Type{gt})
			if !fh.MatVecPacked(host, gt, pk, x, sh.rows, sh.k) {
				t.Fatalf("%s %dx%d: the host tier declined", gt, sh.rows, sh.k)
			}
			fh.Close()
			sse := make([]float32, sh.rows)
			underSSE(t, func() {
				fs := NewJIT(sh.k, sh.rows, []quant.Type{gt})
				defer fs.Close()
				if fs.tier != cpu.TierSSE || fs.packed[gt] == nil || fs.packedFused[gt] == nil {
					t.Fatalf("%s: the forced JIT has tier %v and packed kernels %v/%v",
						gt, fs.tier, fs.packed[gt] != nil, fs.packedFused[gt] != nil)
				}
				if !fs.MatVecPacked(sse, gt, pk, x, sh.rows, sh.k) {
					t.Fatalf("%s %dx%d: the SSE tier declined", gt, sh.rows, sh.k)
				}
			})
			ref := make([]float32, sh.rows)
			q, _ := kernels.QuantOf(gt)
			if err := oracle.MatVecPacked(ref, q, pk.QS, pk.D, pk.SC, x, sh.rows, sh.k); err != nil {
				t.Fatal(err)
			}
			var worst, scale, num, den float64
			for r := range sse {
				if math.IsNaN(float64(sse[r])) || math.IsInf(float64(sse[r]), 0) {
					t.Fatalf("%s %dx%d: row %d is %v", gt, sh.rows, sh.k, r, sse[r])
				}
				worst = math.Max(worst, math.Abs(float64(sse[r])-float64(host[r])))
				scale = math.Max(scale, math.Abs(float64(host[r])))
				d := float64(sse[r]) - float64(ref[r])
				num += d * d
				den += float64(ref[r]) * float64(ref[r])
			}
			if scale == 0 || den == 0 {
				t.Fatalf("%s %dx%d: degenerate outputs -- this gate proved nothing", gt, sh.rows, sh.k)
			}
			if rel := worst / scale; rel > bound {
				t.Errorf("%s %dx%d: max|SSE-host| / max|host| = %.3e, over %.0e", gt, sh.rows, sh.k, rel, bound)
			}
			if nmse := num / den; nmse > 1e-3 {
				t.Errorf("%s %dx%d: SSE NMSE %.3e against the oracle", gt, sh.rows, sh.k, nmse)
			}
		}
	}

	// The float matvec, every float format, at a ragged k.
	for _, ft := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
		const rows, k = 37, 777
		es := int(ft.BlockBytes())
		w := make([]byte, rows*k*es)
		v := prevnniAct(rows * k)
		for i := range v {
			switch ft {
			case quant.F32:
				bits := math.Float32bits(v[i])
				w[4*i], w[4*i+1], w[4*i+2], w[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
			case quant.F16:
				h := quant.EncodeHalf(v[i])
				w[2*i], w[2*i+1] = byte(h), byte(h>>8)
			default:
				b := uint16(math.Float32bits(v[i]) >> 16)
				w[2*i], w[2*i+1] = byte(b), byte(b>>8)
			}
		}
		x := prevnniAct(k + 3)[:k]
		host, sse := make([]float32, rows), make([]float32, rows)
		fh := NewJIT(k, rows, []quant.Type{ft})
		if !fh.MatVec(host, ft, w, x, rows, k) {
			t.Fatalf("%s: the host tier declined the float matvec", ft)
		}
		fh.Close()
		underSSE(t, func() {
			fs := NewJIT(k, rows, []quant.Type{ft})
			defer fs.Close()
			if fs.code[ft] == nil {
				t.Fatalf("%s: the forced JIT built no float matvec", ft)
			}
			if !fs.MatVec(sse, ft, w, x, rows, k) {
				t.Fatalf("%s: the SSE tier declined the float matvec", ft)
			}
		})
		var worst, scale float64
		for r := range sse {
			worst = math.Max(worst, math.Abs(float64(sse[r])-float64(host[r])))
			scale = math.Max(scale, math.Abs(float64(host[r])))
		}
		if scale == 0 || math.IsNaN(worst) {
			t.Fatalf("%s: degenerate float matvec outputs", ft)
		}
		if rel := worst / scale; rel > bound {
			t.Errorf("%s: float matvec max|SSE-host| / max|host| = %.3e, over %.0e", ft, rel, bound)
		}
	}
}
