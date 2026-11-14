package engine

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// blockKVTypes is every KV type held in llama.cpp's blocks, and bfloat16.
func blockKVTypes() []model.KVType {
	return []model.KVType{model.KVQ4_0, model.KVQ4_1, model.KVQ5_0, model.KVQ5_1, model.KVIQ4_NL, model.KVBF16}
}

// TestCPUAttendsABlockHistory is TestCPUAttendsAQ8History for each block
// type: a prompt and its decode on the composed CPU over the type's pages
// and over f32 ones, every block step served composed, the first attending
// block's rows held bit for bit to oracle.QuantizeBlocks of the f32
// history's rows as the container's decoder reads them, and the logits
// finite and not the f32 history's.
func TestCPUAttendsABlockHistory(t *testing.T) {
	const decode = 6
	for _, kt := range blockKVTypes() {
		t.Run(kt.String(), func(t *testing.T) {
			ran := 0
			for _, name := range principleModels() {
				t.Run(name, func(t *testing.T) {
					m, err := model.Open(jlmOf(t, testmodels.Path(name)))
					if err != nil {
						t.Fatal(err)
					}
					defer m.Close()
					c := m.Cfg
					li := -1
					for i := range c.NLayer {
						if c.LayerKind(i).Attends() {
							li = i
							break
						}
					}
					if li < 0 {
						return
					}
					ids := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 6), true)
					n := len(ids) - decode
					type run struct {
						logits [][]float32
						k, v   []float32
						hd     int
					}
					do := func(kv model.KVType) run {
						s := testInstance(t, m, len(ids)+1, WithKVType(kv))
						defer s.Close()
						if s.KVType() != kv {
							t.Fatalf("a %v cache was asked for and %v selected", kv, s.KVType())
						}
						lg, err := s.Prefill(ids[:n])
						if err != nil {
							t.Fatal(err)
						}
						r := run{logits: [][]float32{append([]float32(nil), lg...)}}
						for i := range decode {
							if lg, err = s.Forward(ids[n+i]); err != nil {
								t.Fatal(err)
							}
							r.logits = append(r.logits, append([]float32(nil), lg...))
						}
						cpuServed(t, s, fmt.Sprintf("%v: a prompt of %d", kv, n), (1+decode)*c.NLayer, true)
						h := 16
						if c.SWA(li) && c.SWAWindow > 0 {
							h = min(h, c.SWAWindow)
						}
						hs := testInstance(t, m, h+1, WithKVType(kv))
						defer hs.Close()
						if _, err := hs.Prefill(ids[:h]); err != nil {
							t.Fatal(err)
						}
						l := hs.kvlAt(li)
						_, _, r.k, r.v = kvMigBufs(l, h)
						if l.fmt == nn.KVF32 {
							r.k, r.v = make([]float32, h*l.kvDim()), make([]float32, h*l.kvDim())
						}
						if !hs.kv.MigrateKV(li, r.k, r.v, h, false, hs.kvConv) {
							t.Fatalf("%v: block %d's history was not gathered", kv, li)
						}
						r.hd = l.headDim
						return r
					}
					f, q := do(model.KVF32), do(kt)
					bf := kvFmtOf(kt).Block()
					nb := (f.hd + 31) / 32
					pad, deq, b := make([]float32, nb*32), make([]float32, nb*32), make([]byte, nb*bf.Bytes())
					for _, x := range [][2][]float32{{f.k, q.k}, {f.v, q.v}} {
						for r := 0; r < len(x[0])/f.hd; r++ {
							clear(pad[copy(pad, x[0][r*f.hd:(r+1)*f.hd]):])
							if kt == model.KVBF16 {
								for i, v := range pad {
									deq[i] = math.Float32frombits(uint32(oracle.BF16(v)) << 16)
								}
							} else if err := oracle.QuantizeBlocks(b, pad, bf); err != nil {
								t.Fatal(err)
							} else if err := oracle.DequantBlocks(deq, b, bf); err != nil {
								t.Fatal(err)
							}
							for i := range f.hd {
								// iq4_nl's scale is a quotient of two sums the composed
								// reduction adds in another order: its binary16 may
								// differ by a unit in the last place, 2^-10 of it.
								got := x[1][r*f.hd+i]
								if bf == graph.BlockIQ4NL && math.Abs(float64(got-deq[i])) <= math.Abs(float64(deq[i]))/1000 {
									continue
								}
								if math.Float32bits(got) != math.Float32bits(deq[i]) {
									t.Fatalf("block %d's history row %d element %d is %v, the f32 row quantized is %v", li, r, i, got, deq[i])
								}
							}
						}
					}
					worst, differs := 0.0, false
					for i := range f.logits {
						e, _ := nmseOf(q.logits[i], f.logits[i])
						if math.IsNaN(e) || math.IsInf(e, 0) {
							t.Fatalf("row %d: NMSE %v", i, e)
						}
						worst = max(worst, e)
						differs = differs || e > 0
					}
					if !differs {
						t.Fatalf("the %v history's logits equal the f32 history's: its rows were not quantized", kt)
					}
					t.Logf("block %d's %d history rows quantized exactly; %d prompt rows and %d decoded: worst NMSE %.3e against the f32 history",
						li, len(f.k)/f.hd, n, decode, worst)
					ran++
				})
			}
			if ran == 0 {
				t.Fatalf("no model holds a %v history", kt)
			}
		})
	}
}

// TestPagingDoesNotChangeTheAnswerWithBlockHistories is
// TestPagingDoesNotChangeTheAnswer over each block type's history.
func TestPagingDoesNotChangeTheAnswerWithBlockHistories(t *testing.T) {
	for _, kt := range blockKVTypes() {
		for _, name := range principleModels() {
			t.Run(kt.String()+"/"+name, func(t *testing.T) {
				pagingDoesNotChangeTheAnswer(t, jlmOf(t, testmodels.Path(name)), WithKVType(kt))
			})
		}
	}
}

// TestBlockHistoryFollowsLlamaCpp holds each quantized history type to
// llama.cpp's CPU build run with the same -ctk and -ctv (flash attention on,
// as a quantized V cache needs there), teacher-forced over a document's
// second half on Llama 3.2 1B: what the type moves the log-probabilities by
// against an f32 history (the RMS over llama.cpp's top-16-nat window) is
// held, as a ratio to what it moves llama.cpp's by, to a band: jitllm's own
// reading plus the 0.15 llama.cpp's CUDA build spans against its CPU build
// (0.85-1.02). The directions are not compared: a history bent by 1e-6
// moves jitllm's log-probabilities by 0.091, as much as the two engines'
// f32 arms differ (0.099), and llama.cpp's CUDA and CPU moves agree to a
// cosine of 0.21-0.39 only. The greedy picks are compared: the typed arms
// may disagree on as many of the scored positions as they did when the
// bound was read, plus as many as the f32 arms disagree on (4 of 127). The
// violation reads the scored half's history through scales a fifth too
// large (bendHistory): its move must fall outside the band.
func TestBlockHistoryFollowsLlamaCpp(t *testing.T) {
	dir := os.Getenv("JITLLM_LLAMACPP_CPU")
	if dir == "" {
		dir = os.Getenv("JITLLM_LCPP")
	}
	if dir == "" {
		t.Fatal("JITLLM_LLAMACPP_CPU and JITLLM_LCPP are unset: one names a llama.cpp build holding llama-perplexity, the reference")
	}
	const nctx = 256
	text, err := os.ReadFile("testdata/prompt-28k.txt")
	if err != nil {
		t.Fatal(err)
	}
	gguf := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if !fileExists(gguf) {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", gguf)
	}
	doc := string(text[:8000])
	toks, lcBase := lcppLogProbs(t, dir, gguf, doc, nctx, 0, "-ctk", "f32", "-ctv", "f32", "-fa", "on")
	decode := func(kt model.KVType, bend float32) [][]float32 {
		m, err := model.Open(jlmOf(t, gguf))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		tk := append([]int32(nil), toks...)
		if m.Vocab.AddBOS && m.Vocab.BOS >= 0 {
			tk[0] = int32(m.Vocab.BOS)
		}
		s := testInstance(t, m, nctx, WithKVType(kt))
		defer s.Close()
		if s.KVType() != kt {
			t.Fatalf("a %v cache was asked for and %v selected", kt, s.KVType())
		}
		var out [][]float32
		for i := 0; i < nctx-1; i++ {
			if i == nctx/2 && bend != 1 {
				bendHistory(t, s, bend)
			}
			lg, err := s.Forward(tk[i])
			if err != nil {
				t.Fatal(err)
			}
			if i >= nctx/2 {
				out = append(out, logSoftmax(lg))
			}
		}
		cpuServed(t, s, kt.String(), (nctx-1)*m.Cfg.NLayer, true)
		return out
	}
	base := decode(model.KVF32, 1)
	// gap is the RMS difference of two arms over the reference's window.
	gap := func(a, b [][]float32) float64 {
		var sum float64
		n := 0
		for i := range lcBase {
			top := slices.Max(lcBase[i])
			for j, r := range lcBase[i] {
				if r < top-15.5 || math.IsInf(float64(a[i][j]), -1) || math.IsInf(float64(b[i][j]), -1) {
					continue
				}
				d := float64(a[i][j] - b[i][j])
				sum += d * d
				n++
			}
		}
		return math.Sqrt(sum / float64(n))
	}
	// flips is the scored positions whose greedy picks differ.
	flips := func(a, b [][]float32) int {
		n := 0
		for i := range a {
			if argmax(a[i]) != argmax(b[i]) {
				n++
			}
		}
		return n
	}
	baseFlips := flips(base, lcBase)
	t.Logf("f32: jitllm against llama.cpp %.4f, %d of %d greedy picks differ", gap(base, lcBase), baseFlips, len(base))
	// ratio and picks are each type's readings on this model: its move as a
	// ratio to llama.cpp's and the greedy picks the typed arms disagree on.
	const spread = 0.15
	for _, c := range []struct {
		kt    model.KVType
		ratio float64
		picks int
	}{
		{model.KVQ8_0, 1.067, 6}, {model.KVQ4_0, 1.351, 27}, {model.KVQ4_1, 1.508, 34}, {model.KVQ5_0, 1.283, 17}, {model.KVQ5_1, 1.166, 18}, {model.KVIQ4_NL, 1.341, 28}, {model.KVBF16, 0.973, 5},
	} {
		t.Run(c.kt.String(), func(t *testing.T) {
			_, lcT := lcppLogProbs(t, dir, gguf, doc, nctx, 0, "-ctk", c.kt.String(), "-ctv", c.kt.String(), "-fa", "on")
			lcMove := gap(lcT, lcBase)
			jT := decode(c.kt, 1)
			ratio, fl := gap(jT, base)/lcMove, flips(jT, lcT)
			t.Logf("%v: llama.cpp's move %.4f, jitllm's %.3f of it (band %.3f-%.3f); %d greedy picks differ (at most %d); each arm's own against f32: %d and %d",
				c.kt, lcMove, ratio, c.ratio-spread, c.ratio+spread, fl, c.picks+baseFlips, flips(lcT, lcBase), flips(jT, base))
			if !(math.Abs(ratio-c.ratio) <= spread) {
				t.Errorf("%v moves jitllm's log-probabilities %.3f times as far as llama.cpp's, outside %.3f-%.3f", c.kt, ratio, c.ratio-spread, c.ratio+spread)
			}
			if fl > c.picks+baseFlips {
				t.Errorf("%v: %d greedy picks differ from llama.cpp's, past %d and the f32 arms' %d", c.kt, fl, c.picks, baseFlips)
			}
			br := gap(decode(c.kt, 1.2), base) / lcMove
			t.Logf("%v, its scales a fifth too large: %.3f of llama.cpp's move", c.kt, br)
			if math.Abs(br-c.ratio) <= spread {
				t.Errorf("%v read through scales a fifth too large moves %.3f times as far, inside the band: the gate cannot see it", c.kt, br)
			}
		})
	}
}

// bendHistory widens every attending block's history of s through its
// conversions, scales it by bend and narrows it back: the rows' codes kept
// and every block's scale bend times its own, what a dequantizer reading a
// wrong scale would see.
func bendHistory(t *testing.T, s *Instance, bend float32) {
	pos := s.Pos()
	n := 0
	for li := s.lo; li < s.hi; li++ {
		if s.kv.layers[li].p == 0 {
			continue
		}
		_, _, fk, fv := kvMigBufs(s.kvlAt(li), pos)
		if !s.kv.MigrateKV(li, fk, fv, pos, false, s.kvConv) {
			t.Fatalf("block %d's history was not gathered", li)
		}
		for i := range fk {
			fk[i] *= bend
			fv[i] *= bend
		}
		if !s.kv.MigrateKV(li, fk, fv, pos, true, s.kvConv) {
			t.Fatalf("block %d's history was not scattered", li)
		}
		n++
	}
	if n == 0 {
		t.Fatal("no block's history was bent")
	}
}
