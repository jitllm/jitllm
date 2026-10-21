//go:build amd64 || arm64

package nn_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestQuantizeKVRowsMatchesTheOracle holds the q8 cache's append to
// oracle.QuantizeQ8Window bit for bit, int8 and scale, at every ragged head
// width: the padding of a partial last block must neither move its amax nor
// leave a stale value behind (each row is quantized after a wider one, so a
// padding that is not cleared shows up).
func TestQuantizeKVRowsMatchesTheOracle(t *testing.T) {
	f := nn.NewJIT(256, 256, []quant.Type{quant.F32})
	defer f.Close()
	rng := rand.New(rand.NewSource(7))
	for _, hd := range []int{1, 5, 8, 31, 32, 33, 64, 80, 100, 128, 256} {
		const n = 3
		nb := (hd + 31) / 32
		rs := cpu.KVRowBytes(cpu.KVQ8, hd) / 4
		src := make([]float32, n*hd)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 3)
		}
		// Leave a wide row's values in the padding first.
		wide := make([]float32, cpu.KVRowBytes(cpu.KVQ8, 256)/4)
		big := make([]float32, 3*256)
		for i := range big {
			big[i] = 1e3
		}
		f.QuantizeKVRows(wide, big, 256, 1)

		dst := make([]float32, n*rs+1)
		dst[n*rs] = -9.5 // guard
		f.QuantizeKVRows(dst, src, hd, n)
		if dst[n*rs] != -9.5 {
			t.Fatalf("hd=%d: the append wrote past its rows", hd)
		}
		for r := 0; r < n; r++ {
			x := make([]float32, nb*32)
			copy(x, src[r*hd:(r+1)*hd])
			wq := make([]int8, nb*32)
			wp := make([]float32, 2*nb)
			oracle.QuantizeQ8Window(wq, wp, x, cpu.BiasC(quant.Q8_0), 0, nb, 32)
			row := dst[r*rs : (r+1)*rs]
			gq := unsafe.Slice((*int8)(unsafe.Pointer(&row[0])), nb*32)
			for i := range wq {
				if gq[i] != wq[i] {
					t.Fatalf("hd=%d row %d: q[%d] = %d, the oracle's %d", hd, r, i, gq[i], wq[i])
				}
			}
			for b := 0; b < nb; b++ {
				if math.Float32bits(row[nb*8+2*b]) != math.Float32bits(wp[2*b]) {
					t.Fatalf("hd=%d row %d: d[%d] = %v, the oracle's %v", hd, r, b, row[nb*8+2*b], wp[2*b])
				}
			}
		}
	}
}

// TestQ8AttentionMatchesTheOracle runs a q8 cache end to end through nn: the
// append quantizes, the AttnSet's q8 kernels read, and the softmaxed output of
// one head is held to the oracle in float64 over the unquantized history. The
// band is the q8_0 rounding's, so the gate is the type's error budget rather
// than the kernels' (TestAttnQ8MatchesF32OnTheDequantizedCache pins those bit
// for bit). The f32 run of the same history is held to a band a hundred times
// tighter, so the gate can tell the two apart (it fails if q8 is not taken).
func TestQ8AttentionMatchesTheOracle(t *testing.T) {
	f := nn.NewJIT(256, 256, []quant.Type{quant.F32})
	defer f.Close()
	for _, hd := range []int{17, 32, 40, 64, 128} {
		for _, npos := range []int{1, 9, 100} {
			rng := rand.New(rand.NewSource(int64(hd*131 + npos)))
			nkv := 2
			stride := nkv * hd
			k := make([]float32, npos*stride)
			v := make([]float32, npos*stride)
			for i := range k {
				k[i] = float32(rng.NormFloat64())
				v[i] = float32(rng.NormFloat64())
			}
			q := make([]float32, hd)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			// The reference: head 1, float64, softmax(q.k/sqrt(hd)) v.
			want := make([]float64, hd)
			sc := make([]float64, npos)
			mx := math.Inf(-1)
			for p := 0; p < npos; p++ {
				s := 0.0
				for i := 0; i < hd; i++ {
					s += float64(q[i]) * float64(k[p*stride+hd+i])
				}
				sc[p] = s / math.Sqrt(float64(hd))
				mx = math.Max(mx, sc[p])
			}
			z := 0.0
			for p := range sc {
				sc[p] = math.Exp(sc[p] - mx)
				z += sc[p]
			}
			for p := 0; p < npos; p++ {
				for i := 0; i < hd; i++ {
					want[i] += sc[p] / z * float64(v[p*stride+hd+i])
				}
			}

			run := func(fm cpu.KVFmt) []float32 {
				set := f.AttnSetFor(hd, hd, stride, fm)
				rb := cpu.KVRowBytes(fm, hd) / 4
				var kc, vc []float32
				switch fm {
				case cpu.KVQ8:
					kc = make([]float32, npos*nkv*rb)
					vc = make([]float32, npos*nkv*rb)
					for p := 0; p < npos; p++ {
						f.QuantizeKVRows(kc[p*nkv*rb:], k[p*stride:], hd, nkv)
						f.QuantizeKVRows(vc[p*nkv*rb:], v[p*stride:], hd, nkv)
					}
				default:
					kc, vc = k, v
				}
				att := make([]float32, npos)
				set.AttnScores(att, kc[rb:], q, npos) // head 1
				m := float32(math.Inf(-1))
				for p := range att {
					att[p] /= float32(math.Sqrt(float64(hd)))
					m = max(m, att[p])
				}
				var s float32
				for p := range att {
					att[p] = float32(math.Exp(float64(att[p] - m)))
					s += att[p]
				}
				for p := range att {
					att[p] /= s
				}
				out := make([]float32, hd)
				set.AttnAcc(out, vc[rb:], att, npos)
				return out
			}
			nmse := func(got []float32) float64 {
				var e, r float64
				for i := range want {
					d := float64(got[i]) - want[i]
					e += d * d
					r += want[i] * want[i]
				}
				return e / r
			}
			g8, g32 := nmse(run(cpu.KVQ8)), nmse(run(cpu.KVF32))
			if math.IsNaN(g8) || g8 > 1e-3 {
				t.Fatalf("hd=%d npos=%d: q8 NMSE %.3g past the q8_0 band", hd, npos, g8)
			}
			if g32 > 1e-9 {
				t.Fatalf("hd=%d npos=%d: f32 NMSE %.3g", hd, npos, g32)
			}
			if npos > 1 && g8 < 100*g32 {
				t.Fatalf("hd=%d npos=%d: q8 NMSE %.3g is not above f32's %.3g: the q8 cache did not run", hd, npos, g8, g32)
			}
			t.Logf("hd=%d npos=%d: NMSE q8 %.3g f32 %.3g", hd, npos, g8, g32)
		}
	}
}
