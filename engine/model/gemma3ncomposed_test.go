package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGemma3nServesComposedOnCPU runs each Gemma 3n fixture on the CPU
// device, every block composed for it over its four residual streams, each
// bound at its offset in the stream-major residual, teacher-forced on the
// fixture's own ids against the CPU's float forms over the same history
// (withCPUFloat): every block step and head served, within 1e-5 of their
// logits on decode and on a prompt with its
// decode, and each piece of the block taken out (gemma3nViolations:
// the prediction, the correction, the side branch, the gaussian top-k on no
// block and on every block, the correction's scale) moving the composed
// arm past 100 times its worst and past 1e-4. The input rows' expansion of
// stream 0 and the head's collapse of the streams are composed too, every
// row of them served (cpuLogits), and streams 1 and 2 swapped where the
// expansion writes them and stream 1 read for stream 2 where the collapse
// reads them moves the arm as far.
func TestGemma3nServesComposedOnCPU(t *testing.T) {
	const page, bound = 32, 1e-5
	for _, fx := range gemma3nFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			gm, g := openGemma3n(t, fx.name)
			ok := fx.sel(gm)
			gm.Close()
			if !ok {
				t.Fatal("the container does not carry the fixture's graph")
			}
			toks := g.IDs
			path := jlmOf(t, testmodels.Path(fx.name+".gguf"))
			ref, err := Open(path, withCPUFloat())
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			ref.setKVPageForTest(page)
			want := cpuLogits(t, ref, toks, cpuFaultNone, true)
			m, err := Open(path)
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
			pw := cpuPrompt(t, ref, toks, n, 4, cpuFaultNone, true)
			pg := cpuPrompt(t, m, toks, n, 4, cpuFaultNone, true)
			pworst := 0.0
			for i := range pw {
				pworst = max(pworst, logitNMSE(pg[i], pw[i]))
			}
			t.Logf("the composed CPU from the float forms: decode %.3g over %d steps, a prompt of %d and its decode %.3g", worst, len(toks), n, pworst)
			if !(worst < bound) || !(pworst < bound) {
				t.Fatalf("the composed CPU is %.3g and %.3g from the float forms, bound %g", worst, pworst, bound)
			}
			for _, v := range gemma3nViolations() {
				undo := v.mut(m)
				bad := worstFrom(want, cpuLogits(t, m, toks, cpuFaultNone, true))
				undo()
				t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
				if !(bad > 100*worst) || !(bad > 10*bound) {
					t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
				}
			}
			// Streams 1 and 2 swapped where the expansion writes them, and
			// stream 1 read in stream 2's place where the collapse reads them.
			bad := worstFrom(want, cpuLogits(t, m, toks, cpuFaultStreams, true))
			t.Logf("violation \"the streams swapped at the expansion, one read twice at the collapse\": worst NMSE %.3g", bad)
			if !(bad > 100*worst) || !(bad > 10*bound) {
				t.Errorf("the streams swapped: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
		})
	}
}

// TestGemma3nServesComposed holds the composer to each Gemma 3n fixture's
// blocks on every device present: the residual's streams bound as views at
// their offsets in the stream-major residual on the card, every block step
// served, decode and a prompt's rows (gpu.ComposedPrompt), within the
// device band of the host's logits, and each piece of the block taken out
// (gemma3nViolations) moving them past 100 times the clean worst and
// past 1e-4. The input rows over every stream and the per-layer inputs, and
// the streams' collapse before the head, are stated (a head over every
// stream is refused by name, UncoveredStreams) and served on the card with
// the blocks by TestStreamRowsServeOnTheCard.
func TestGemma3nServesComposed(t *testing.T) {
	const bound, promptLen = 1e-5, 3
	ran := 0
	for _, fx := range gemma3nFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openGemma3n(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatal("the container does not carry the fixture's graph")
			}
			if c := coveredBlocks(m); c != m.Cfg.NLayer {
				t.Fatalf("the front end states %d of %d blocks: %s (%v)", c, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
			}
			if s := m.frontGraphs(); s.head != nil || s.headWhy != UncoveredStreams || s.embed == nil || len(s.embed.Out) != m.Cfg.Streams()+1 {
				t.Fatalf("the head over several streams is %s, want refused by name, and the input rows %s, want stated over every stream and the per-layer inputs", s.headWhy, s.embedWhy)
			}
			ids := g.IDs
			host := m.NewInstance(len(ids) + 1)
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
				parts := gpu.ComposedBlock | gpu.ComposedPrompt
				got, st, _ := composedDecode(t, m, dev, prompt, toks, parts, true, false)
				served, declined := st.ServedPart(gpu.ComposedBlock)
				if n := len(toks) * m.Cfg.NLayer; served != n || declined != 0 {
					t.Fatalf("%d block steps served and %d declined, want %d served (%s)", served, declined, n, declinesOf(st))
				}
				np, dp := st.ServedPart(gpu.ComposedPrompt)
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
					for _, v := range gemma3nViolations() {
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
