package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGemma3nServesComposedOnCPU runs each Gemma 3n fixture on the CPU
// device, every block composed for it over its four residual streams, each
// bound at its offset in main's stream-major residual, teacher-forced on the
// fixture's own ids against main's kernels: every block step and head
// served, within 1e-5 of main's logits on decode and on a prompt with its
// decode, and each piece of the block taken out (gemma3nComposedViolations:
// the prediction, the correction, the side branch, the gaussian top-k on no
// block and on every block, the correction's scale) moving the composed
// arm past 100 times its worst and past 1e-4. The input rows' expansion of
// stream 0 and the head's collapse of the streams are composed too, every
// row of them served (cpuLogits), and streams 1 and 2 swapped where they
// are bound moves the arm as far.
func TestGemma3nServesComposedOnCPU(t *testing.T) {
	const page, bound = 32, 1e-5
	for _, fx := range gemma3nFixtures {
		t.Run(fx.name, func(t *testing.T) {
			gm, g := openGemma3n(t, fx.name)
			ok := fx.sel(gm)
			gm.Close()
			if !ok {
				t.Fatal("the container does not carry the fixture's graph")
			}
			toks := g.IDs
			path := jlmOf(t, testmodels.Path(fx.name+".gguf"))
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
				t.Fatalf("the front end states %d of %d blocks: %s (%v)", c, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
			}
			if g := m.Composing().BlockGraph(0); len(g.Out) != m.Cfg.PredStreams {
				t.Fatalf("block 0 writes %d streams, want %d", len(g.Out), m.Cfg.PredStreams)
			}
			worst := worstFrom(want, cpuLogits(t, m, toks, cpuFaultNone, true))
			n := len(toks) - 4
			pw := cpuPrompt(t, ref, toks, n, 4, cpuFaultNone, false)
			pg := cpuPrompt(t, m, toks, n, 4, cpuFaultNone, true)
			pworst := 0.0
			for i := range pw {
				pworst = max(pworst, logitNMSE(pg[i], pw[i]))
			}
			t.Logf("the composed CPU from main: decode %.3g over %d steps, a prompt of %d and its decode %.3g", worst, len(toks), n, pworst)
			if !(worst < bound) || !(pworst < bound) {
				t.Fatalf("the composed CPU is %.3g and %.3g from main, bound %g", worst, pworst, bound)
			}
			for _, v := range gemma3nComposedViolations() {
				undo := v.mut(m)
				bad := worstFrom(want, cpuLogits(t, m, toks, cpuFaultNone, true))
				undo()
				t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
				if !(bad > 100*worst) || !(bad > 10*bound) {
					t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
				}
			}
			// Streams 1 and 2 swapped where the expansion writes them and the
			// collapse reads them.
			bad := worstFrom(want, cpuLogits(t, m, toks, cpuFaultStreams, true))
			t.Logf("violation \"the streams swapped at the expansion and the collapse\": worst NMSE %.3g", bad)
			if !(bad > 100*worst) || !(bad > 10*bound) {
				t.Errorf("the streams swapped: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
		})
	}
}

// gemma3nComposedViolations is gemma3nViolations as the composed arm reads
// them: the gaussian top-k is the activation the container states for the
// first NSparse blocks, which main's arm runs from NSparse, so its two
// violations restate the blocks' activations instead.
func gemma3nComposedViolations() []c6Violation {
	acts := func(name string, to func(c *Config, acts []uint8) func(li int) uint8) c6Violation {
		return c6Violation{name, func(m *Model) func() {
			jc := m.container.Config()
			was := slices.Clone(jc.ActLayers)
			pick := to(m.Cfg, was)
			for li := range jc.ActLayers {
				jc.ActLayers[li] = pick(li)
			}
			undo := cfgMut(m, func(*Config) {})
			return func() {
				copy(jc.ActLayers, was)
				undo()
			}
		}}
	}
	var vs []c6Violation
	for _, v := range gemma3nViolations {
		switch v.name {
		case "no gaussian top-k":
			v = acts(v.name, func(c *Config, a []uint8) func(int) uint8 {
				return func(int) uint8 { return a[c.NSparse] }
			})
		case "the gaussian on every block":
			v = acts(v.name, func(c *Config, a []uint8) func(int) uint8 {
				return func(int) uint8 { return a[0] }
			})
		}
		vs = append(vs, v)
	}
	return vs
}

// TestGemma3nServesComposed holds the composer to each Gemma 3n fixture's
// blocks on every device present: the residual's streams bound as views at
// their offsets in the stream-major residual on the card, every block step
// served, decode and a prompt's rows (tier.ComposedPrompt), within the
// device band of the host's logits, and each piece of the block taken out
// (gemma3nComposedViolations) moving them past 100 times the clean worst and
// past 1e-4. The streams' expansion at the input rows and their collapse
// before the head are the host's (the CPU device's composed forms,
// TestGemma3nServesComposedOnCPU): the input rows and a head over every
// stream are refused by name (UncoveredStreams), and the card's head reads
// the collapsed row.
func TestGemma3nServesComposed(t *testing.T) {
	const bound, promptLen = 1e-5, 3
	ran := 0
	for _, fx := range gemma3nFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openGemma3n(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatal("the container does not carry the fixture's graph")
			}
			if c := coveredBlocks(m); c != m.Cfg.NLayer {
				t.Fatalf("the front end states %d of %d blocks: %s (%v)", c, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
			}
			if s := m.frontGraphs(); s.head != nil || s.headWhy != UncoveredStreams || s.embed != nil || s.embedWhy != UncoveredStreams {
				t.Fatalf("the head and the input rows over several streams are %s and %s, want both refused by name", s.headWhy, s.embedWhy)
			}
			ids := g.IDs
			host := m.NewState(len(ids) + 1)
			var want [][]float32
			for _, id := range ids {
				lg, err := host.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, append([]float32(nil), lg...))
			}
			host.Close()
			prompt, toks := ids[:promptLen], ids[promptLen:]
			run := func(t *testing.T, dev string) float64 {
				t.Helper()
				parts := tier.ComposedBlock | tier.ComposedPrompt
				got, st, _, _ := composedDecode(t, m, dev, prompt, toks, parts, true, false)
				served, declined, _ := st.ServedPart(tier.ComposedBlock)
				if n := len(toks) * m.Cfg.NLayer; served != n || declined != 0 {
					t.Fatalf("%d block steps served and %d declined, want %d served (%s)", served, declined, n, declinesOf(st))
				}
				np, dp, _ := st.ServedPart(tier.ComposedPrompt)
				if np == 0 || dp != 0 {
					t.Fatalf("the prompt's blocks served %d times and declined %d (%s)", np, dp, declinesOf(st))
				}
				if st.FormedAtStep != 0 {
					t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
				}
				worst := 0.0
				for i := range got {
					e := logitNMSE(got[i], want[promptLen-1+i])
					if math.IsNaN(e) {
						t.Fatalf("step %d: the logits are not finite", i)
					}
					worst = max(worst, e)
				}
				t.Logf("%s: %d block steps and %d prompt blocks served, worst NMSE %.3g", dev, served, np, worst)
				return worst
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					ran++
					worst := run(t, dev)
					if !(worst < bound) {
						t.Fatalf("worst logit NMSE %.3g against the host, bound %g", worst, bound)
					}
					for _, v := range gemma3nComposedViolations() {
						undo := v.mut(m)
						bad := run(t, dev)
						undo()
						t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
						if !(bad > 100*worst) || !(bad > 10*bound) {
							t.Errorf("%s: the composed arm cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
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
