package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// gemma4Composed are the Gemma 4 fixtures the front end states whole: the
// global layers at another head width, kv head count and rotary width than
// the sliding ones, v k's own on the global layers of two, normed with no
// weight, and every block's output scaled by its stored scalar; and the
// E-models' blocks, each reading its slice of the row's per-layer inputs,
// the last two reading an earlier block's history.
var gemma4Composed = []string{"synth-gemma4", "synth-gemma4-kv", "synth-gemma4-wide", "synth-gemma4-e", "synth-gemma4-e-ragged", "synth-gemma4-moe"}

// TestGemma4ServesComposed holds the composer to each of gemma4Composed's
// blocks and head on every device present, as TestC6ServesComposed holds
// the classic fixtures: every block step and head served, the logits within
// the device band of the host's, and each of Gemma 4's features taken out
// (gemma4Violations) moving them. The rest of a step is not served on a
// card: each geometry's scratch set holds one rotary table where the plan
// states two, so the step's tables are declined by name (DeclineNoTable) at
// every step, counted here, and the input rows with them (open).
func TestGemma4ServesComposed(t *testing.T) {
	const bound = 1e-5
	ran := 0
	for _, name := range gemma4Composed {
		t.Run(name, func(t *testing.T) {
			m, g := openGemma4(t, name)
			defer m.Close()
			if !gemma4Selects(t, m, name) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			ran += servesComposed(t, m, g.IDs, bound, gemma4ComposedViolations(name), false)
			for _, dev := range stepDevices() {
				t.Run(dev+"/step", func(t *testing.T) {
					prompt, toks := g.IDs[:3], g.IDs[3:]
					_, st, _, _ := composedDecode(t, m, dev, prompt, toks, tier.ComposedStep, true, false)
					served, _, _ := st.ServedPart(tier.ComposedTable)
					declined := st.StepDeclines[jit.DeclineNoTable]
					t.Logf("%s: the step's tables served %d times, declined %d times (no table graph)", dev, served, declined)
					nb, db, _ := st.ServedPart(tier.ComposedBlock)
					np, dp, up := st.ServedPart(tier.ComposedPrompt)
					t.Logf("%s: block steps served %d, declined %d; the prompt's %d served, %d declined, %d unasked (%s)", dev, nb, db, np, dp, up, declinesOf(st))
					if served != 0 || declined == 0 {
						t.Fatalf("the step's tables served %d times and declined %d times by name: the gap moved", served, declined)
					}
					// The per-layer inputs' tables are the host's: the device
					// declines the input rows that read them, by name.
					if m.Cfg.PLEDim != 0 {
						ne, de, _ := st.ServedPart(tier.ComposedEmbed)
						dw := st.StepDeclines[jit.DeclineEmbedWeight]
						t.Logf("%s: the input rows served %d times, declined %d (%d for the per-layer tables)", dev, ne, de, dw)
						if ne != 0 || dw == 0 {
							t.Fatalf("the input rows served %d times and declined %d times for the per-layer tables: the gap moved", ne, dw)
						}
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host")
	}
}

// TestGemma4ServesComposedOnCPU runs each of gemma4Composed on the CPU
// device, every block composed for it, teacher-forced on the fixture's own
// ids against main's kernels: every block step and head served (cpuLogits,
// cpuPrompt), within 1e-5 of main's logits (float weights) on decode and on
// a prompt with its decode, and each of Gemma 4's features taken out of the
// composed arm past 100 times its worst and past 1e-4. The ids are the
// golden's, not a greedy run's: a random fixture's greedy tokens repeat, and
// over rows of one token every value is the same, so no change to the
// attention's weights could move the logits.
func TestGemma4ServesComposedOnCPU(t *testing.T) {
	const page, bound = 32, 1e-5
	for _, name := range gemma4Composed {
		t.Run(name, func(t *testing.T) {
			gm, g := openGemma4(t, name)
			gm.Close()
			toks := g.IDs
			// The history at the CPU's own binary16 (jit.Halves) in both arms.
			path := jlmOf(t, testmodels.Path(name+".gguf"))
			ref, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			ref.setKVPageForTest(page)
			want := cpuLogits(t, ref, toks, cpuFaultNone, false)
			m, err := Open(path, noTune, WithComposedCPU(true))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			m.setKVPageForTest(page)
			if c := coveredBlocks(m); c != m.Cfg.NLayer {
				t.Fatalf("the front end states %d of %d blocks: %s", c, m.Cfg.NLayer, m.Uncovered(0))
			}
			worst := worstFrom(want, cpuLogits(t, m, toks, cpuFaultNone, true))
			n := len(toks) - 4
			pworst := 0.0
			pw := cpuPrompt(t, ref, toks, n, 4, cpuFaultNone, false)
			pg := cpuPrompt(t, m, toks, n, 4, cpuFaultNone, true)
			for i := range pw {
				pworst = max(pworst, logitNMSE(pg[i], pw[i]))
			}
			t.Logf("the composed CPU from main: decode %.3g over %d steps, a prompt of %d and its decode %.3g", worst, len(toks), n, pworst)
			if !(worst < bound) || !(pworst < bound) {
				t.Fatalf("the composed CPU is %.3g and %.3g from main, bound %g", worst, pworst, bound)
			}
			for _, v := range gemma4ComposedViolations(name) {
				undo := v.mut(m)
				bad := worstFrom(want, cpuLogits(t, m, toks, cpuFaultNone, true))
				undo()
				t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
				if !(bad > 100*worst) || !(bad > 10*bound) {
					t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
				}
			}
		})
	}
}

// gemma4ComposedViolations is gemma4Violations[name] as the composed gates
// hold them, past 10 times their bound: the final softcap of 30 over a
// random fixture's small logits moves them 1.8e-05 when taken out, so the
// cap is moved to 1 instead, where it bends every logit. A model read
// without its per-layer embeddings leaves their tensors unbound, so the
// front end states no block of it (UncoveredUnbound) and no composed arm
// runs: the composed gates leave the per-layer inputs' norm unweighted
// instead, which every block reads through its second input. Each block
// given the next block's gate places no block on CUDA (the tier took 0 of
// 6), so it is no violation the card arms can run.
func gemma4ComposedViolations(name string) []c6Violation {
	var vs []c6Violation
	for _, v := range gemma4Violations[name] {
		switch v.name {
		case "no final softcap":
			v = c6Violation{"a final softcap of 1", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.FinalSoftcap = 1 }) }}
		case "no per-layer embeddings":
			v = c6Violation{"the per-layer inputs' norm unweighted", func(m *Model) func() {
				was := m.pleNorm
				ones := make([]float32, len(was))
				for i := range ones {
					ones[i] = 1
				}
				m.pleNorm = ones
				return func() { m.pleNorm = was }
			}}
		}
		vs = append(vs, v)
	}
	return vs
}

// gemma4Selects is gemma4Fixtures' selection for name.
func gemma4Selects(t *testing.T, m *Model, name string) bool {
	t.Helper()
	for _, fx := range gemma4Fixtures {
		if fx.name == name {
			return fx.sel(m)
		}
	}
	t.Fatalf("%s is not a Gemma 4 fixture", name)
	return false
}

// TestGemma4PictureServesComposed runs Gemma 4's picture prompt (the
// synth-gemma4v fixture: the soft tokens one run, bidirectional on the
// sliding layers and causal on the global ones) with every text block the
// composer's, on the CPU device and on every card present, the prompt's
// chunks and the decode after it: every block step served and none
// declined, the logits within each device's band of the host's, and the
// mask's two violations -- the run causal everywhere, and bidirectional on
// every layer -- far from them. The run reaches the composed attention as
// each row's key count in the descriptors it reads, never as a flag. The
// run is 266 rows: a card takes it in one call of its reserved width, the
// CPU in calls of 64 after a first pass writes every row's keys.
func TestGemma4PictureServesComposed(t *testing.T) {
	host, g := openGemma4V(t)
	defer host.Close()
	if !host.Cfg.BidirSWA || !host.Cfg.GeomSplit() {
		t.Fatalf("the fixture does not select the shape: BidirSWA %v, two geometries %v", host.Cfg.BidirSWA, host.Cfg.GeomSplit())
	}
	if c := coveredBlocks(host); c != host.Cfg.NLayer {
		t.Fatalf("the front end states %d of %d blocks: %s %v", c, host.Cfg.NLayer, host.Uncovered(0), host.UncoveredErr(0))
	}
	want := g4vRun(t, host, g, true)
	violations := func(t *testing.T, m *Model, run func(bidir bool) [][]float32, want [][]float32, worst, bound float64) {
		t.Helper()
		causal := worstFrom(want, run(false))
		m.Cfg.BidirSWA = false
		every := worstFrom(want, run(true))
		m.Cfg.BidirSWA = true
		t.Logf("violations: causal everywhere %.3e, bidirectional on every layer %.3e, clean %.3e", causal, every, worst)
		if !(causal > 100*worst) || !(causal > 10*bound) || !(every > 100*worst) || !(every > 10*bound) {
			t.Fatalf("the mask's violations read %.3e and %.3e against a clean %.3e: the composed arm cannot see them", causal, every, worst)
		}
	}

	t.Run("cpu", func(t *testing.T) {
		const bound = 1e-5
		// The history at the CPU's own binary16 (jit.Halves) in both arms.
		ref, _ := openGemma4V(t, WithKVF16(true))
		defer ref.Close()
		want := g4vRun(t, ref, g, true)
		m, _ := openGemma4V(t, WithKVF16(true), WithComposedCPU(true))
		defer m.Close()
		run := func(bidir bool) [][]float32 { return g4vRun(t, m, g, bidir) }
		got := run(true)
		served, declined := cpuSteps(t, m, g)
		worst := worstFrom(want, got)
		t.Logf("the composed CPU: %d block steps served, worst logit NMSE %.3e from main", served, worst)
		for i, n := range declined {
			if n != 0 {
				t.Fatalf("%d block step(s) declined: %s", n, jit.DeclineName(i))
			}
		}
		if served == 0 || !(worst < bound) {
			t.Fatalf("%d block steps served, worst NMSE %.3e from main, bound %g", served, worst, bound)
		}
		violations(t, m, run, want, worst, bound)
	})

	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			const bound = 1e-3
			var st tier.Stats
			run := func(bidir bool) [][]float32 {
				gpu, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
					tier.WithConfig(func(c *tier.Config) {
						c.Composed, c.Census, c.NoGraph = tier.ComposedBlock|tier.ComposedPrompt|tier.ComposedHead, true, true
					})}, testTierOpts(t)...)...)
				if err != nil || gpu == nil {
					noDevice(t, dev, err)
				}
				defer gpu.Close()
				out := g4vRunOn(t, host, g, bidir, gpu)
				st = gpu.Stats()
				return out
			}
			worst := worstFrom(want, run(true))
			blocks, bd, _ := st.ServedPart(tier.ComposedBlock)
			prompt, pd, _ := st.ServedPart(tier.ComposedPrompt)
			t.Logf("%s: %d decode and %d prompt block steps served (%d and %d declined); worst logit NMSE %.3e from the host",
				dev, blocks, prompt, bd, pd, worst)
			for i, k := range st.BlockDeclines {
				if k > 0 {
					t.Fatalf("%d block step(s) declined because %s", k, tier.DeclineName(i))
				}
			}
			if want := (len(g.Cont) - 1) * host.Cfg.NLayer; blocks != want || bd != 0 || prompt == 0 || pd != 0 {
				t.Fatalf("%d decode block steps served (want %d), %d prompt block steps, %d and %d declined", blocks, want, prompt, bd, pd)
			}
			if !(worst < bound) {
				t.Fatalf("worst logit NMSE %.3e from the host, bound %g", worst, bound)
			}
			violations(t, host, run, want, worst, bound)
		})
	}
}

// cpuSteps is the composed CPU's block steps over g's picture prompt and its
// decode, in a State of its own.
func cpuSteps(t *testing.T, m *Model, g *g4vGolden) (int, []int) {
	t.Helper()
	st := m.NewState(8)
	pic, err := st.Picture(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	st.Close()
	if err != nil {
		t.Fatal(err)
	}
	spans := []Span{{Tokens: append(append([]int32(nil), g.Pre...), g.BOI)},
		{Picture: pic, Bidir: true}, {Tokens: append([]int32{g.EOI}, g.Post...)}}
	s := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + len(g.Cont) + 1)
	defer s.Close()
	if _, err := s.PrefillMixed(spans...); err != nil {
		t.Fatal(err)
	}
	for _, id := range g.Cont[:len(g.Cont)-1] {
		if _, err := s.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	served, declined := s.CPUBlockSteps()
	return served, append([]int(nil), declined[:]...)
}

// worstFrom is the worst logit NMSE of got's rows from want's, infinite for
// a row that is not finite.
func worstFrom(want, got [][]float32) float64 {
	w := 0.0
	for i := range want {
		e := logitNMSE(got[i], want[i])
		if math.IsNaN(e) {
			return math.Inf(1)
		}
		w = max(w, e)
	}
	return w
}
