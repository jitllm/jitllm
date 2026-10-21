//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// q8Cache is one KV run in both forms a q8 gate compares: the q8_0 rows as
// the cache holds them, and the same values dequantized into an f32 run with
// the same element stride. Rows sit at stride elements (q8: stride/hd rows of
// KVRowBytes) and each head row is quantized on its own, as the append does.
type q8Cache struct {
	q8  []float32 // the q8 rows, in float32 slots
	f32 []float32 // d*q per element, the exact value the q8 kernel widens to
}

func newQ8Cache(rng *rand.Rand, hd, stride, npos int) q8Cache {
	rows := stride / hd
	nb := (hd + cpu.KVQ8Block - 1) / cpu.KVQ8Block
	rb := cpu.KVRowBytes(cpu.KVQ8, hd)
	c := q8Cache{
		q8:  make([]float32, npos*rows*rb/4),
		f32: make([]float32, npos*stride),
	}
	// Poison the f32 run between rows: a kernel must never read it.
	for i := range c.f32 {
		c.f32[i] = float32(math.NaN())
	}
	x := make([]float32, nb*cpu.KVQ8Block)
	q := make([]int8, nb*cpu.KVQ8Block)
	pairs := make([]float32, 2*nb)
	for p := 0; p < npos; p++ {
		for r := 0; r < rows; r++ {
			clear(x)
			for i := 0; i < hd; i++ {
				// A spread of magnitudes per block, so each block's d differs.
				x[i] = float32(rng.NormFloat64() * math.Exp(float64(i/cpu.KVQ8Block)))
			}
			oracle.QuantizeQ8Window(q, pairs, x, 0, 0, nb, cpu.KVQ8Block)
			row := unsafe.Slice((*byte)(unsafe.Pointer(&c.q8[(p*rows+r)*rb/4])), rb)
			for i := range q {
				row[i] = byte(q[i])
			}
			ps := unsafe.Slice((*float32)(unsafe.Pointer(&row[nb*cpu.KVQ8Block])), 2*nb)
			copy(ps, pairs)
			for i := 0; i < hd; i++ {
				c.f32[p*stride+r*hd+i] = float32(q[i]) * pairs[2*(i/cpu.KVQ8Block)]
			}
		}
	}
	return c
}

// q8Tables is every tier this host can run: the host's own, and the SSE tier
// on an AVX2 host (the tier an Atom runs, exercised here on its instructions).
func q8Tables() map[string]*cpu.Emitters {
	m := map[string]*cpu.Emitters{cpu.HostTier().String(): hostTable()}
	if runtime.GOARCH == "amd64" && cpu.HostTier() != cpu.TierSSE {
		m[cpu.TierSSE.String()] = cpu.EmittersFor(cpu.TierSSE)
	}
	return m
}

func mapQ8(t *testing.T, name string) func([]byte, error) *cpu.Code {
	return func(b []byte, err error) *cpu.Code {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		c, err := cpu.Map(b)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		return c
	}
}

func bitsEqual(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: element %d is %v on the q8 cache and %v on its dequantized f32 twin", what, i, got[i], want[i])
		}
	}
}

// TestAttnQ8MatchesF32OnTheDequantizedCache holds every q8 attention kernel to
// its f32 twin, bit for bit, run over the q8 cache's dequantized values: the
// q8 kernel widens d*q exactly as the f32 run holds it and then runs the f32
// kernel's arithmetic in its order, so any difference is the q8 kernel's
// addressing (a wrong block's scale, a wrong row, the tail) -- which an NMSE
// against float64 would hide inside the quantization error. The f32 kernels
// are gated against the oracle elsewhere (TestAttn*), which closes the chain.
//
// Ragged head dimensions take every tail: below a vector, between vectors,
// across a partial last 32-element block; stride 3*hd is a row-major GQA walk,
// stride hd a head-major one. Every output carries a guard after it.
func TestAttnQ8MatchesF32OnTheDequantizedCache(t *testing.T) {
	const guard = 4
	for tier, em := range q8Tables() {
		for _, hd := range []int{1, 3, 4, 7, 8, 9, 17, 31, 32, 33, 40, 64, 80, 100, 128, 256} {
			for _, rows := range []int{1, 3} {
				for _, npos := range []int{1, 2, 7, 33} {
					stride := rows * hd
					rng := rand.New(rand.NewSource(int64(hd*7919 + rows*31 + npos)))
					c := newQ8Cache(rng, hd, stride, npos)
					q0 := make([]float32, hd)
					q1 := make([]float32, hd)
					w0 := make([]float32, npos)
					w1 := make([]float32, npos)
					for i := range q0 {
						q0[i], q1[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
					}
					for i := range w0 {
						w0[i], w1[i] = float32(rng.Float64()), float32(rng.Float64())
					}
					seed := make([]float32, hd)
					for i := range seed {
						seed[i] = float32(rng.NormFloat64())
					}
					name := func(k string) string {
						return tier + "/" + k + "/hd" + itoaQ8(hd) + "/s" + itoaQ8(stride) + "/n" + itoaQ8(npos)
					}
					w := func(cache []float32) *byte { return (*byte)(unsafe.Pointer(&cache[0])) }
					out := func(n int, fill []float32) []float32 {
						o := make([]float32, n+guard)
						copy(o, fill)
						for i := n; i < len(o); i++ {
							o[i] = -7.25
						}
						return o
					}
					checkGuard := func(k string, o []float32, n int) {
						for i := n; i < len(o); i++ {
							if o[i] != -7.25 {
								t.Fatalf("%s: wrote past its output at %d", name(k), i)
							}
						}
					}

					type emit func(int, int, cpu.KVFmt) ([]byte, error)
					// Scores and the paired scores.
					run1 := func(e emit, fm cpu.KVFmt, cache []float32) []float32 {
						k := mapQ8(t, name("scores/"+fm.String()))(e(hd, stride, fm))
						defer k.Close()
						o := out(npos, nil)
						k.Call(&cpu.Args{Out: &o[0], W: w(cache), Rows: int64(npos), Q32: &q0[0]})
						checkGuard("scores/"+fm.String(), o, npos)
						return o[:npos]
					}
					bitsEqual(t, name("scores"), run1(em.AttnScores, cpu.KVQ8, c.q8), run1(em.AttnScores, cpu.KVF32, c.f32))

					run2 := func(fm cpu.KVFmt, cache []float32) ([]float32, []float32) {
						k := mapQ8(t, name("scores2/"+fm.String()))(em.AttnScores2(hd, stride, fm))
						defer k.Close()
						a, b := out(npos, nil), out(npos, nil)
						k.Call(&cpu.Args{Out: &a[0], Out2: &b[0], W: w(cache), Rows: int64(npos), Q32: &q0[0], Q2: &q1[0]})
						checkGuard("scores2", a, npos)
						checkGuard("scores2", b, npos)
						return a[:npos], b[:npos]
					}
					a8, b8 := run2(cpu.KVQ8, c.q8)
					a32, b32 := run2(cpu.KVF32, c.f32)
					bitsEqual(t, name("scores2/head0"), a8, a32)
					bitsEqual(t, name("scores2/head1"), b8, b32)

					// The accumulates, plain and Into (seeded).
					for _, k := range []struct {
						tag  string
						e    emit
						into bool
					}{{"acc", em.AttnAcc, false}, {"acc_into", em.AttnAccInto, true}} {
						runA := func(fm cpu.KVFmt, cache []float32) []float32 {
							kc := mapQ8(t, name(k.tag+"/"+fm.String()))(k.e(hd, stride, fm))
							defer kc.Close()
							var fill []float32
							if k.into {
								fill = seed
							}
							o := out(hd, fill)
							kc.Call(&cpu.Args{Out: &o[0], W: w(cache), AScale: &w0[0], Rows: int64(npos)})
							checkGuard(k.tag, o, hd)
							return o[:hd]
						}
						bitsEqual(t, name(k.tag), runA(cpu.KVQ8, c.q8), runA(cpu.KVF32, c.f32))
					}
					for _, k := range []struct {
						tag  string
						e    emit
						into bool
					}{{"acc2", em.AttnAcc2, false}, {"acc2_into", em.AttnAcc2Into, true}} {
						runA := func(fm cpu.KVFmt, cache []float32) ([]float32, []float32) {
							kc := mapQ8(t, name(k.tag+"/"+fm.String()))(k.e(hd, stride, fm))
							defer kc.Close()
							var fill []float32
							if k.into {
								fill = seed
							}
							a, b := out(hd, fill), out(hd, fill)
							kc.Call(&cpu.Args{Out: &a[0], Out2: &b[0], W: w(cache), AScale: &w0[0], AScale2: &w1[0], Rows: int64(npos)})
							checkGuard(k.tag, a, hd)
							checkGuard(k.tag, b, hd)
							return a[:hd], b[:hd]
						}
						a8, b8 := runA(cpu.KVQ8, c.q8)
						a32, b32 := runA(cpu.KVF32, c.f32)
						bitsEqual(t, name(k.tag+"/head0"), a8, a32)
						bitsEqual(t, name(k.tag+"/head1"), b8, b32)
					}

					// The tiled scores, three queries at a query stride of hd.
					const qt = 3
					qs := make([]float32, qt*hd)
					for i := range qs {
						qs[i] = float32(rng.NormFloat64())
					}
					runT := func(fm cpu.KVFmt, cache []float32) []float32 {
						k := mapQ8(t, name("tiled/"+fm.String()))(em.AttnScoresTiled(hd, stride, hd, npos, qt, fm))
						defer k.Close()
						o := out(qt*npos, nil)
						k.Call(&cpu.Args{Out: &o[0], W: w(cache), Rows: int64(npos), Q32: &qs[0]})
						checkGuard("tiled", o, qt*npos)
						return o[:qt*npos]
					}
					if hd%4 == 0 || runtime.GOARCH != "arm64" {
						bitsEqual(t, name("tiled"), runT(cpu.KVQ8, c.q8), runT(cpu.KVF32, c.f32))
					}
				}
			}
		}
	}
}

// TestKVWidenIsTheDequantizedCache holds the q8 cache's widening to d*q bit
// for bit, every element of every row, at every ragged head width, with a
// guard after the output: it is what a q8 history becomes on its way to a
// device that holds float32.
func TestKVWidenIsTheDequantizedCache(t *testing.T) {
	for tier, em := range q8Tables() {
		for _, hd := range []int{1, 3, 4, 7, 8, 9, 17, 31, 32, 33, 40, 64, 80, 100, 128, 256} {
			for _, npos := range []int{0, 1, 5} {
				rng := rand.New(rand.NewSource(int64(hd*17 + npos)))
				c := newQ8Cache(rng, hd, hd, max(npos, 1))
				k := mapQ8(t, tier+"/widen/hd"+itoaQ8(hd))(em.KVWiden(hd))
				out := make([]float32, max(npos, 1)*hd+4)
				for i := range out {
					out[i] = -7.25
				}
				k.Call(&cpu.Args{Out: &out[0], W: (*byte)(unsafe.Pointer(&c.q8[0])), Rows: int64(npos)})
				k.Close()
				bitsEqual(t, tier+"/widen/hd"+itoaQ8(hd)+"/n"+itoaQ8(npos), out[:npos*hd], c.f32[:npos*hd])
				for i := npos * hd; i < len(out); i++ {
					if out[i] != -7.25 {
						t.Fatalf("%s hd=%d npos=%d: the widening wrote past its rows at %d", tier, hd, npos, i)
					}
				}
			}
		}
	}
}

// TestAttnQ8RefusesAPartialRow:a q8 cache is addressed in whole rows, so a
// stride that is not a multiple of hd has no meaning and is refused.
func TestAttnQ8RefusesAPartialRow(t *testing.T) {
	if _, err := hostTable().AttnScores(64, 96, cpu.KVQ8); err == nil {
		t.Fatal("a q8 scores kernel at kvStride 96 over hd 64 was emitted")
	}
}

func itoaQ8(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for ; n > 0; n /= 10 {
		b = append([]byte{byte('0' + n%10)}, b...)
	}
	return string(b)
}
