package engine

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
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
						if w, _ := windowAt(m, li); w > 0 {
							h = min(h, w)
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

// TestBlockHistoryFollowsItsReference holds each quantized history type,
// teacher-forced over a document's second half on Llama 3.2 1B, to its
// numerical reference: the model over an f32 history whose every key and
// value row passes through the type's narrowing and widening before the
// block appends it (historyReference), so each row holds the type's values
// from the moment it is written -- a step's own attention reads its newest
// row quantized, as llama.cpp's does -- and an f32 history's attention reads
// them. Two numbers are kept apart.
//
// The implementation's error is a step's: before each scored step the
// reference is given the typed instance's history, and the two steps'
// distributions are compared, KL at the median step. A trajectory cannot
// hold it: a rounding apart in one block's attention moves the next block's
// row across a quantization step, and two arms one unit in the last place
// apart drift as far as the type's loss; a mean is that step's too. The
// steps run on the CPU's float forms, whose history kernels are the typed
// path's own: the int8 activations of the others turn a unit in the last
// place into as much as a violation moves. The bound is four times the
// larger of the reference's own steps given the history a unit in the last
// place up and down: KL is quadratic in a small difference, so four times
// is two units in the last place.
//
// The quantization's loss, the free-running reference against the f32
// history, is held as a ratio to what the type moves llama.cpp's CPU build
// by under the same -ctk and -ctv (flash attention on): a cross-check, its
// band the reference's own reading plus the 0.15 llama.cpp's CUDA build
// spans against its CPU build (0.85-1.02); the typed instance and llama.cpp
// may disagree on as many greedy picks as when that reading was taken plus
// the f32 arms' (4 of 127). The reference's first attending block, whose
// rows no attention feeds, is held to the oracle's quantizer bit for bit,
// and two f32 arms of one process to each other bit for bit.
//
// Two violations, each given the typed history before every step, must
// fall outside the implementation's bound: an f32 history that attends its
// newest row unquantized, and the reference reading the history through
// scales a fifth too large. On each listed card the reference runs with
// every block placed, its loss held to the CPU's within the same 0.15, and
// so does the typed instance, whose card writes its history as the type
// does (deviceGraph): its steps the card's reference's bit for bit. So does
// a typed instance moved to the card mid-sequence beside its reference
// given its history, and both histories come home equal element for element:
// the card's rows narrowed back are the type's values. On a seam of half
// the blocks the typed instance's loss is held to the CPU's within 0.15.
func TestBlockHistoryFollowsItsReference(t *testing.T) {
	dir := os.Getenv("JITLLM_LLAMACPP_CPU")
	if dir == "" {
		dir = os.Getenv("JITLLM_LCPP")
	}
	if dir == "" {
		t.Fatal("JITLLM_LLAMACPP_CPU and JITLLM_LCPP are unset: one names a llama.cpp build holding llama-perplexity, the cross-check")
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
	// open is the model, as kt's reference where ref says, its first token
	// the model's own beginning, and the tokens.
	open := func(t *testing.T, kt model.KVType, ref bool) (*model.Model, []int32) {
		m, err := model.Open(jlmOf(t, gguf))
		if err != nil {
			t.Fatal(err)
		}
		if ref {
			historyReference(t, m, kt)
		}
		tk := append([]int32(nil), toks...)
		if m.Vocab.AddBOS && m.Vocab.BOS >= 0 {
			tk[0] = int32(m.Vocab.BOS)
		}
		return m, tk
	}
	instance := func(t *testing.T, m *model.Model, kt model.KVType, opts ...Option) *Instance {
		s := testInstance(t, m, nctx, append([]Option{WithKVType(kt)}, opts...)...)
		if s.KVType() != kt {
			t.Fatalf("a %v cache was asked for and %v selected", kt, s.KVType())
		}
		return s
	}
	// decode is the scored half's log-probabilities of kt's history, its
	// reference's where ref says, on the CPU device or with every block on
	// card dev, its first blocks where blocks says; on the CPU, every
	// attending block's history at its end too.
	decode := func(t *testing.T, kt model.KVType, ref bool, dev string, blocks ...int) ([][]float32, [][]float32, int) {
		m, tk := open(t, kt, ref)
		defer m.Close()
		if ref {
			kt = model.KVF32
		}
		var g *gpu.GPU
		var err error
		if dev != "" {
			if g, err = openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...); err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
		}
		s := instance(t, m, kt)
		defer s.Close()
		if g != nil {
			s.SetDeviceFallback(false)
			want := m.Cfg.NLayer
			if len(blocks) > 0 {
				want = blocks[0]
				err = s.SetDeviceLayers(g, want)
			} else {
				err = s.SetDevice(g)
			}
			if err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() != want {
				t.Fatalf("%s took %d of %d blocks: %s", dev, s.GPULayers(), want, g.Err())
			}
		}
		var lp [][]float32
		for i := 0; i < nctx-1; i++ {
			lg, err := s.Forward(tk[i])
			if err != nil {
				t.Fatal(err)
			}
			if i >= nctx/2 {
				lp = append(lp, logSoftmax(lg))
			}
		}
		if g != nil {
			return lp, nil, 0
		}
		cpuServed(t, s, fmt.Sprintf("%v (reference %v)", kt, ref), (nctx-1)*m.Cfg.NLayer, true)
		h, hd := histories(t, s)
		return lp, h, hd
	}
	// moved runs kt's typed instance over the first half on the CPU, moves
	// every block with its history to card dev, runs the scored half there
	// and brings the blocks home; beside it the reference, given the typed
	// history before the move, does the same. It returns both scored halves
	// and both histories once home.
	moved := func(t *testing.T, kt model.KVType, dev string) (rows, home [2][][]float32) {
		mt, tk := open(t, kt, false)
		defer mt.Close()
		mr, _ := open(t, kt, true)
		defer mr.Close()
		g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			noDevice(t, dev, err)
		}
		defer g.Close()
		ss := [2]*Instance{instance(t, mt, kt), instance(t, mr, model.KVF32)}
		for _, s := range ss {
			defer s.Close()
			if _, err := s.Prefill(tk[:nctx/2]); err != nil {
				t.Fatal(err)
			}
		}
		h, _ := histories(t, ss[0])
		setHistory(t, ss[1], h, nil)
		for j, s := range ss {
			s.SetDeviceFallback(false)
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() != mt.Cfg.NLayer {
				t.Fatalf("%s took %d of %d blocks mid-sequence: %s", dev, s.GPULayers(), mt.Cfg.NLayer, g.Err())
			}
			for i := nctx / 2; i < nctx-1; i++ {
				lg, err := s.Forward(tk[i])
				if err != nil {
					t.Fatal(err)
				}
				rows[j] = append(rows[j], logSoftmax(lg))
			}
			if s.SetGPULayers(0); s.GPULayers() != 0 {
				t.Fatalf("%d blocks stayed on %s", s.GPULayers(), dev)
			}
			home[j], _ = histories(t, s)
		}
		return rows, home
	}
	// stepped is lockstep's arms' logits over the scored half: the typed
	// instance's, and each f32 instance's from the typed history.
	type stepped struct {
		typed, ref, up, down, late, bent [][]float32
	}
	// lockstep runs kt's typed instance over the scored half and f32
	// instances beside it, each given the typed instance's history before
	// every step, all on the CPU's float forms: the reference, the
	// reference given it a unit in the last place up and down, an f32
	// history that attends its newest row unquantized (late, the plain
	// graph), and the reference reading it through scales a fifth too large
	// (bent).
	lockstep := func(t *testing.T, kt model.KVType) stepped {
		mt, tk := open(t, kt, false)
		defer mt.Close()
		mr, _ := open(t, kt, true)
		defer mr.Close()
		typed := instance(t, mt, kt, withCPUFloat())
		defer typed.Close()
		ref, up, down, bent := instance(t, mr, model.KVF32, withCPUFloat()), instance(t, mr, model.KVF32, withCPUFloat()), instance(t, mr, model.KVF32, withCPUFloat()), instance(t, mr, model.KVF32, withCPUFloat())
		late := instance(t, mt, model.KVF32, withCPUFloat())
		f32s := []*Instance{ref, up, down, late, bent}
		for _, s := range f32s {
			defer s.Close()
		}
		for _, s := range append([]*Instance{typed}, f32s...) {
			if _, err := s.Prefill(tk[:nctx/2]); err != nil {
				t.Fatal(err)
			}
		}
		var r stepped
		rows := []*[][]float32{&r.ref, &r.up, &r.down, &r.late, &r.bent}
		inf := float32(math.Inf(1))
		for i := nctx / 2; i < nctx-1; i++ {
			h, _ := histories(t, typed)
			setHistory(t, ref, h, nil)
			setHistory(t, late, h, nil)
			setHistory(t, up, h, func(x float32) float32 { return math.Nextafter32(x, inf) })
			setHistory(t, down, h, func(x float32) float32 { return math.Nextafter32(x, -inf) })
			setHistory(t, bent, h, func(x float32) float32 { return 1.2 * x })
			lg, err := typed.Forward(tk[i])
			if err != nil {
				t.Fatal(err)
			}
			r.typed = append(r.typed, slices.Clone(lg))
			for j, s := range f32s {
				lg, err := s.Forward(tk[i])
				if err != nil {
					t.Fatal(err)
				}
				*rows[j] = append(*rows[j], slices.Clone(lg))
			}
		}
		cpuServed(t, typed, kt.String(), (nctx-nctx/2)*mt.Cfg.NLayer, true)
		return r
	}
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
	// kl is KL(a || b) at each scored position, the logits a and b
	// normalized at f64 so that the measure does not round a difference
	// below what it measures, its median and its mean: a step whose history
	// crossed a quantization step stands out of a mean, and the median is
	// the step that did not.
	kl := func(a, b [][]float32) (med, mean float64) {
		lse := func(v []float32) float64 {
			top := float64(slices.Max(v))
			var sum float64
			for _, x := range v {
				sum += math.Exp(float64(x) - top)
			}
			return top + math.Log(sum)
		}
		d := make([]float64, len(a))
		for i := range a {
			la, lb := lse(a[i]), lse(b[i])
			for j, x := range a[i] {
				p := float64(x) - la
				d[i] += math.Exp(p) * (p - float64(b[i][j]) + lb)
			}
			mean += d[i] / float64(len(a))
		}
		slices.Sort(d)
		return d[len(d)/2], mean
	}
	// lps is rows of logits as log-probabilities.
	lps := func(rows [][]float32) [][]float32 {
		out := make([][]float32, len(rows))
		for i, r := range rows {
			out[i] = logSoftmax(r)
		}
		return out
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
	base, baseHist, _ := decode(t, model.KVF32, false, "")
	again, _, _ := decode(t, model.KVF32, false, "")
	if e := gap(again, base); e != 0 {
		t.Fatalf("two f32 arms of one process are %.3g apart: the engine is not deterministic here, and no bound below is", e)
	}
	baseFlips := flips(base, lcBase)
	t.Logf("f32: jitllm against llama.cpp %.4f, %d of %d greedy picks differ; a second f32 arm against the first 0", gap(base, lcBase), baseFlips, len(base))
	cardBase, seamBase := map[string][][]float32{}, map[string][][]float32{}
	layers := func() int {
		m, _ := open(t, model.KVF32, false)
		defer m.Close()
		return m.Cfg.NLayer
	}()
	// ratio and picks are each type's readings on this model: its
	// reference's loss as a ratio to llama.cpp's move, and the greedy picks
	// the typed instance and llama.cpp disagree on.
	const spread = 0.15
	for _, c := range []struct {
		kt    model.KVType
		ratio float64
		picks int
	}{
		{model.KVQ8_0, 1.073, 6}, {model.KVQ4_0, 1.455, 27}, {model.KVQ4_1, 1.491, 34}, {model.KVQ5_0, 1.301, 17}, {model.KVQ5_1, 1.320, 18}, {model.KVIQ4_NL, 1.220, 28}, {model.KVBF16, 0.968, 5},
	} {
		t.Run(c.kt.String(), func(t *testing.T) {
			r := lockstep(t, c.kt)
			impl, implMean := kl(r.ref, r.typed)
			up, upMean := kl(r.ref, r.up)
			down, downMean := kl(r.ref, r.down)
			floor := max(up, down)
			t.Logf("%v: a step of the typed instance against its reference's from the same history: KL %.3g at the median, %.3g in the mean, RMS %.3g, %d picks; a unit in the last place up and down: KL %.3g and %.3g, %.3g and %.3g, RMS %.3g and %.3g",
				c.kt, impl, implMean, gap(lps(r.typed), lps(r.ref)), flips(r.typed, r.ref), up, down, upMean, downMean, gap(lps(r.up), lps(r.ref)), gap(lps(r.down), lps(r.ref)))
			if !(impl <= 4*floor) {
				t.Errorf("%v: a step of the typed history is KL %.3g from its reference's at the median, past four times the %.3g a unit in the last place moves it", c.kt, impl, floor)
			}
			for _, v := range []struct {
				what string
				rows [][]float32
			}{{"an f32 history attending its newest row unquantized", r.late}, {"the history read through scales a fifth too large", r.bent}} {
				e, mean := kl(r.ref, v.rows)
				t.Logf("%v, %s: a step KL %.3g from the reference's at the median, %.3g in the mean, RMS %.3g, %d picks", c.kt, v.what, e, mean, gap(lps(v.rows), lps(r.ref)), flips(v.rows, r.ref))
				if e <= 4*floor {
					t.Errorf("%v, %s: a step is KL %.3g from the reference's at the median, inside the bound %.3g: the gate cannot see it", c.kt, v.what, e, 4*floor)
				}
			}
			ref, refHist, hd := decode(t, c.kt, true, "")
			heldToOracle(t, c.kt, refHist[:2], baseHist[:2], hd)
			_, lcT := lcppLogProbs(t, dir, gguf, doc, nctx, 0, "-ctk", c.kt.String(), "-ctv", c.kt.String(), "-fa", "on")
			lcMove, loss := gap(lcT, lcBase), gap(ref, base)
			typed, _, _ := decode(t, c.kt, false, "")
			ratio, fl := loss/lcMove, flips(typed, lcT)
			t.Logf("%v: the quantization's loss %.4f, %.3f of llama.cpp's move %.4f (band %.3f-%.3f); %d greedy picks differ from llama.cpp's (at most %d); each engine's own against f32: %d and %d",
				c.kt, loss, ratio, lcMove, c.ratio-spread, c.ratio+spread, fl, c.picks+baseFlips, flips(typed, base), flips(lcT, lcBase))
			if !(math.Abs(ratio-c.ratio) <= spread) {
				t.Errorf("%v costs jitllm's log-probabilities %.3f times what it costs llama.cpp's, outside %.3f-%.3f", c.kt, ratio, c.ratio-spread, c.ratio+spread)
			}
			if fl > c.picks+baseFlips {
				t.Errorf("%v: %d greedy picks differ from llama.cpp's, past %d and the f32 arms' %d", c.kt, fl, c.picks, baseFlips)
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					cb, ok := cardBase[dev]
					if !ok {
						cb, _, _ = decode(t, model.KVF32, false, dev)
						cardBase[dev] = cb
					}
					rc, _, _ := decode(t, c.kt, true, dev)
					tc, _, _ := decode(t, c.kt, false, dev)
					cl, tl := gap(rc, cb)/loss, gap(tc, cb)/loss
					t.Logf("%v on %s: the reference's loss %.4f and the typed instance's %.4f, %.3f and %.3f of the CPU's; the typed instance against the card's reference %.3g",
						c.kt, dev, gap(rc, cb), gap(tc, cb), cl, tl, gap(tc, rc))
					if !(math.Abs(cl-1) <= spread) || !(math.Abs(tl-1) <= spread) {
						t.Errorf("%v on %s: the reference's loss is %.3f of the CPU's and the typed instance's %.3f, outside %.2f-%.2f", c.kt, dev, cl, tl, 1-spread, 1+spread)
					}
					if e := gap(tc, rc); e != 0 {
						t.Errorf("%v on %s: the typed instance is %.3g from the card's reference: the card does not write its history as the type does", c.kt, dev, e)
					}
					rows, home := moved(t, c.kt, dev)
					exact, total, first := sameHistory(home[0], home[1])
					t.Logf("%v moved to %s mid-sequence: %.3g from its reference moved alike; home again, %d of %d history elements equal%s",
						c.kt, dev, gap(rows[0], rows[1]), exact, total, first)
					if e := gap(rows[0], rows[1]); e != 0 {
						t.Errorf("%v moved to %s is %.3g from its reference moved alike", c.kt, dev, e)
					}
					if exact != total {
						t.Errorf("%v: %d of %d history elements came home from %s other than the reference's%s", c.kt, total-exact, total, dev, first)
					}
					half := layers / 2
					sb, ok := seamBase[dev]
					if !ok {
						sb, _, _ = decode(t, model.KVF32, false, dev, half)
						seamBase[dev] = sb
					}
					st, _, _ := decode(t, c.kt, false, dev, half)
					sl := gap(st, sb) / loss
					t.Logf("%v on a seam of %d blocks on %s: its loss %.4f, %.3f of the CPU's", c.kt, half, dev, gap(st, sb), sl)
					if !(math.Abs(sl-1) <= spread) {
						t.Errorf("%v on a seam of %d blocks on %s: its loss is %.3f of the CPU's, outside %.2f-%.2f", c.kt, half, dev, sl, 1-spread, 1+spread)
					}
				})
			}
		})
	}
}

// historyReference has every block of m that appends to its own history
// write each key and value row as kt holds it (typedAppends, the graph a
// card is offered under kt): narrowed and widened by the ops the instance's
// own conversions take for kt before the append, so an f32 history holds
// kt's values from the moment each row is written. The quantizer is the
// composed one kt's store runs, which TestCPUAttendsABlockHistory holds to
// oracle.QuantizeBlocks; what reads the history is the f32 history's
// attention, not kt's.
func historyReference(t *testing.T, m *model.Model, kt model.KVType) {
	t.Helper()
	f := kvFmtOf(kt)
	fs := m.FrontGraphs()
	n := 0
	for li, g := range fs.Blocks {
		if g == nil {
			continue
		}
		if r := typedAppends(g, f, m.Cfg.HeadDimAt(li)); r != g {
			fs.Blocks[li], n = r, n+1
		}
	}
	if n == 0 {
		t.Fatalf("no block appends to its own history: there is no %v history to reference", kt)
	}
}

// typedRow is row's head rows of hd each as the oracle writes kt's history:
// each padded to whole blocks of 32, quantized and read back.
func typedRow(t *testing.T, kt model.KVType, row []float32, hd int) []float32 {
	t.Helper()
	nb := (hd + 31) / 32
	pad, deq := make([]float32, nb*32), make([]float32, nb*32)
	out := make([]float32, len(row))
	for h := 0; h < len(row)/hd; h++ {
		clear(pad[copy(pad, row[h*hd:(h+1)*hd]):])
		switch kt {
		case model.KVBF16:
			for i, v := range pad {
				deq[i] = math.Float32frombits(uint32(oracle.BF16(v)) << 16)
			}
		case model.KVQ8_0:
			codes, pairs := make([]int8, len(pad)), make([]float32, 2*nb)
			if err := oracle.QuantizeQ8(codes, pairs, pad, 128); err != nil {
				t.Fatal(err)
			}
			for i := range deq {
				deq[i] = pairs[2*(i/nn.KVQ8Block)] * float32(codes[i])
			}
		default:
			bf := kvFmtOf(kt).Block()
			b := make([]byte, nb*bf.Bytes())
			if err := oracle.QuantizeBlocks(b, pad, bf); err != nil {
				t.Fatal(err)
			}
			if err := oracle.DequantBlocks(deq, b, bf); err != nil {
				t.Fatal(err)
			}
		}
		copy(out[h*hd:(h+1)*hd], deq)
	}
	return out
}

// setHistory writes h, every attending block's history as histories
// gathers it, into s's history at its position, each element through f
// where f is set.
func setHistory(t *testing.T, s *Instance, h [][]float32, f func(float32) float32) {
	t.Helper()
	pos, i := s.Pos(), 0
	for li := s.lo; li < s.hi; li++ {
		if s.kv.layers[li].p == 0 {
			continue
		}
		k, v := h[i], h[i+1]
		if f != nil {
			k, v = slices.Clone(k), slices.Clone(v)
			for j := range k {
				k[j], v[j] = f(k[j]), f(v[j])
			}
		}
		if !s.kv.MigrateKV(li, k, v, pos, true, s.kvConv) {
			t.Fatalf("block %d's history was not scattered", li)
		}
		i += 2
	}
}

// histories is every attending block's history of s as floats, its keys
// then its values, and the first one's head width.
func histories(t *testing.T, s *Instance) ([][]float32, int) {
	pos := s.Pos()
	var h [][]float32
	hd := 0
	for li := s.lo; li < s.hi; li++ {
		if s.kv.layers[li].p == 0 {
			continue
		}
		l := s.kvlAt(li)
		_, _, fk, fv := kvMigBufs(l, pos)
		if l.fmt == nn.KVF32 {
			fk, fv = make([]float32, pos*l.kvDim()), make([]float32, pos*l.kvDim())
		}
		if !s.kv.MigrateKV(li, fk, fv, pos, false, s.kvConv) {
			t.Fatalf("block %d's history was not gathered", li)
		}
		if hd == 0 {
			hd = l.headDim
		}
		h = append(h, fk, fv)
	}
	if len(h) == 0 {
		t.Fatal("no block keeps a history")
	}
	return h, hd
}

// sameHistory is how many of a's elements are b's bit for bit, of how
// many, and where the first that is not lies.
func sameHistory(a, b [][]float32) (exact, total int, first string) {
	for i := range min(len(a), len(b)) {
		for j := range min(len(a[i]), len(b[i])) {
			total++
			if math.Float32bits(a[i][j]) == math.Float32bits(b[i][j]) {
				exact++
			} else if first == "" {
				first = fmt.Sprintf("; the first apart: attending block %d's %s, element %d, %v against %v",
					i/2, [2]string{"keys", "values"}[i%2], j, a[i][j], b[i][j])
			}
		}
	}
	return exact, total, first
}

// heldToOracle holds a reference's first attending block's keys and values
// (ref, its two planes) to the oracle's quantizer over the f32 history's
// (f32): no attention feeds that block's rows, so the reference's are the
// f32 rows written as kt holds them. iq4_nl's scale is a quotient of two
// sums the composed reduction adds in another order: its binary16 may
// differ by a unit in the last place, 2^-10 of it.
func heldToOracle(t *testing.T, kt model.KVType, ref, f32 [][]float32, hd int) {
	t.Helper()
	w := 0
	for i := range ref {
		want := typedRow(t, kt, f32[i], hd)
		for j, got := range ref[i] {
			if math.Float32bits(got) == math.Float32bits(want[j]) {
				continue
			}
			if kt == model.KVIQ4_NL && math.Abs(float64(got-want[j])) <= math.Abs(float64(want[j]))/1000 {
				w++
				continue
			}
			t.Fatalf("%v: the reference's first attending block's %s element %d is %v, the oracle's %v",
				kt, [2]string{"keys", "values"}[i], j, got, want[j])
		}
	}
	t.Logf("%v: the reference's first attending block is the oracle's quantizer's, %d elements within a unit of their scale", kt, w)
}
