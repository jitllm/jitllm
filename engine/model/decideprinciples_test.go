package model

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The five principles on the decision models (AGENTS.md). JIT and no Go
// compute: TestEveryModelRunsGenerated answers a request on every decision
// model (auditRun). Paging: TestLayaPagesWithTheSameAnswer on the host and
// with blocks placed, and the host arm here on d1 and lev. Relocation and
// allocation: here. Their backbones' decode keeps the principles through
// their fixtures (synth-lfm2 in principleFixtures; Qwen3.5 in
// JITLLM_ALLOC_MODELS' default).

// decisionDevices opens each device present, tuning off; they close with t.
func decisionDevices(t *testing.T) []*tier.GPU {
	t.Helper()
	var out []*tier.GPU
	for _, spec := range stepDevices() {
		g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
			tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			continue
		}
		t.Cleanup(func() { g.Close() })
		out = append(out, g)
	}
	return out
}

// TestDecisionRelocatesMidPrompt moves a decoder readout's blocks onto a
// device a third of the way into its prompt and home again at two thirds --
// their KV pages and their recurrent summaries with them -- and holds the label
// logits at the prompt's end to the host's, which never moved: the Relocation
// principle on d1 (lfm2's short convolutions) and lev (qwen35's gated delta
// rule). The violation is the history left behind: the blocks placed with no
// migration, so the device attends over nothing it wrote.
func TestDecisionRelocatesMidPrompt(t *testing.T) {
	for _, c := range decisionCases {
		if c.kind == jlm.DecisionLaya || c.kind == jlm.DecisionD1Omni {
			continue // an encoder has no history: TestDecisionRelocatesBetweenDevices
		}
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind)
			defer m.Close()
			m.SetKVF16(false)
			gpus := decisionDevices(t)
			if len(gpus) == 0 {
				noDevice(t, "device", nil)
			}
			d, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "ambiguous.json"))
			rp := readoutOf(t, d, state, qs)
			ids, labels := rp.ids, rp.finish
			rp.start()
			lg, err := d.st.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			host := labels(lg)
			a, b := len(ids)/3, 2*len(ids)/3
			// moved runs the prompt with the blocks onto g at a and home at b;
			// lose clears the recurrent summaries just before they would go,
			// the violation: a migration that carried nothing.
			headHome := false
			moved := func(t *testing.T, g *tier.GPU, lose bool) ([]float32, int) {
				rp.start()
				rows0 := d.st.PrefillRows()
				if _, err := d.st.Prefill(ids[:a]); err != nil {
					t.Fatal(err)
				}
				if lose {
					for li := range d.st.rconv {
						clear(d.st.rconv[li])
						clear(d.st.rstate[li])
					}
				}
				if err := d.st.SetDeviceLayers(g, -1); err != nil {
					t.Fatal(err)
				}
				placed := d.st.GPULayers()
				if placed == 0 {
					t.Fatalf("%s took no block: %v", g.Name(), g.Err())
				}
				headHome = d.st.headReady == nil
				if _, err := d.st.Prefill(ids[a:b]); err != nil {
					t.Fatal(err)
				}
				if err := d.st.SetDeviceLayers(g, 0); err != nil {
					t.Fatal(err)
				}
				if n := d.st.GPULayers(); n != 0 {
					t.Fatalf("%d blocks stayed on %s", n, g.Name())
				}
				lg, err := d.st.Prefill(ids[b:])
				if err != nil {
					t.Fatal(err)
				}
				// Each row ran once: the moves carried the history rather
				// than computing it again.
				if got := d.st.PrefillRows() - rows0; got != len(ids) {
					t.Fatalf("%d prompt rows run for a %d-token prompt", got, len(ids))
				}
				return labels(lg), placed
			}
			worst := func(got []float32) float64 {
				w := 0.0
				for i := range host {
					w = math.Max(w, math.Abs(float64(got[i]-host[i])))
				}
				return w
			}
			// Under a budget that holds a few blocks and not the head as
			// well, placeHead gives blocks back for the head's room after
			// their history went up: each must take it home first, and a
			// block restored must take it up again. Before that fix a block
			// so given back lost its recurrent summary (lev: a label logit
			// 8.1 off on a V100 at 1200 MiB, worse than losing every summary).
			// Selected (RULE 10): at least one budget must leave the head home
			// with blocks placed, the case that walks the release.
			budgeted := 0
			for _, mb := range []uint64{500, 700, 900, 1200, 1600, 2400} {
				g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(stepDevices()[0]),
					tier.WithDeviceTune(tier.TuneOff), tier.WithBudget(mb << 20)}, testTierOpts(t)...)...)
				if err != nil {
					t.Fatal(err)
				}
				got, placed := moved(t, g, false)
				w := worst(got)
				if headHome && placed > 0 {
					budgeted++
				}
				g.Close()
				t.Logf("budget %d MiB: %d blocks onto the device at token %d and home at %d (worst %.4f)", mb, placed, a, b, w)
				if tol := decisionRelocTol[c.name]; !(w <= tol) {
					t.Errorf("relocated mid-prompt under a %d MiB budget, a label logit moved %.4f from the host's (bound %.2f)",
						mb, w, tol)
				}
			}
			if budgeted == 0 {
				t.Errorf("no budget left the head at home with blocks placed: the release for the head's room never ran")
			}
			for _, g := range gpus {
				t.Run(g.Name(), func(t *testing.T) {
					got, placed := moved(t, g, false)
					w := worst(got)
					t.Logf("%d of %d blocks onto %s at token %d of %d and home at %d: label logits %v, the host's %v (worst %.4f)",
						placed, m.Cfg.NLayer, g.Name(), a, len(ids), b, got, host, w)
					tol := decisionRelocTol[c.name]
					if !(w <= tol) {
						t.Errorf("relocated mid-prompt, a label logit moved %.4f from the host's (bound %.2f)", w, tol)
					}
					bad, _ := moved(t, g, true)
					t.Logf("the recurrent summaries lost on the way: worst %.4f", worst(bad))
					if worst(bad) <= tol {
						t.Errorf("a migration that carried no recurrent state moved a label logit %.4f: the gate does not see it",
							worst(bad))
					}
				})
			}
		})
	}
}

// decisionRelocTol is how far a label logit may move when a third of the
// prompt ran on a device: the f32 reduction-order band of RULE 11c. Measured
// on the V100 (CUDA, Vulkan): d1 0.029-0.037, with its short convolutions'
// state lost 0.082-0.095 (three taps of history: a thin but clean margin);
// lev above 0.3 on CUDA, with its delta-rule state lost 3.2.
var decisionRelocTol = map[string]float64{"d1-3B": 0.06, "lev": 1.0, "clef": 1.0, "clef-fixture": 0.01}

// TestDecisionRelocatesBetweenDevices moves one Decider's blocks from the host
// onto each device in turn and home, answering the same request at every
// stop, against the host's answers: every block of an encoder (Laya), and of
// a decoder whatever each card holds.
func TestDecisionRelocatesBetweenDevices(t *testing.T) {
	for _, c := range append([]struct {
		name, gguf string
		kind       jlm.DecisionKind
	}{{"laya-fixture", "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya},
		{"d1-omni", d1OmniModel, jlm.DecisionD1Omni}}, decisionNames()...) {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind, noTune)
			defer m.Close()
			if !m.IsEncoder() {
				m.SetKVF16(false)
			}
			gpus := decisionDevices(t)
			if len(gpus) == 0 {
				noDevice(t, "device", nil)
			}
			d, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
			answer := func() [][]DecisionAnswer {
				ans, err := d.Decide(state, qs, nil)
				if err != nil {
					t.Fatal(err)
				}
				return [][]DecisionAnswer{ans}
			}
			host := answer()
			tol := decisionDevTol[c.name]
			if c.kind == jlm.DecisionLaya {
				tol = decisionDevTol["laya"]
			}
			nb := decisionBlocks(m)
			for _, g := range gpus {
				if err := d.SetDevice(g, -1); err != nil {
					t.Fatal(err)
				}
				placed := d.DeviceBlocks()
				if placed == 0 || m.IsEncoder() && placed != layaExpressible(m, "") {
					t.Fatalf("%s took %d of %d blocks: %v", g.Name(), placed, nb, g.Err())
				}
				w := decisionDiffRow(t, host, answer())
				t.Logf("%d of %d blocks on %s: worst difference from the host %.5f", placed, nb, g.Name(), w)
				if w > tol {
					t.Errorf("on %s the answers moved %.5f from the host's, past %.3f", g.Name(), w, tol)
				}
			}
			if err := d.SetDevice(nil, 0); err != nil {
				t.Fatal(err)
			}
			if n := d.DeviceBlocks(); n != 0 {
				t.Fatalf("%d blocks stayed on a device", n)
			}
			// Bit for bit: the host's kernels are untuned (noTune), so the
			// arm64 host does not pick another reduction order on a re-run.
			if w := decisionDiffRow(t, host, answer()); w != 0 {
				t.Errorf("home again, the answers moved %.6f from the host's first", w)
			}
		})
	}
}

// decisionNames is decisionCases' decoders, named as the device gate names them.
func decisionNames() []struct {
	name, gguf string
	kind       jlm.DecisionKind
} {
	var out []struct {
		name, gguf string
		kind       jlm.DecisionKind
	}
	for _, c := range decisionCases {
		if c.kind != jlm.DecisionLaya && c.kind != jlm.DecisionD1Omni {
			out = append(out, struct {
				name, gguf string
				kind       jlm.DecisionKind
			}{c.name, c.gguf, c.kind})
		}
	}
	return out
}

// decisionDiffRow is decisionDiff over one request's answers.
func decisionDiffRow(t *testing.T, want, got [][]DecisionAnswer) float64 {
	t.Helper()
	worst := 0.0
	for r := range want {
		for i := range want[r] {
			a, b := want[r][i], got[r][i]
			vals := [][2]float64{{a.Noul, b.Noul}}
			if k := len(a.Probs); k > 1 {
				vals = append(vals, [2]float64{a.Score / float64(k-1), b.Score / float64(k-1)})
			}
			for j := range a.Probs {
				vals = append(vals, [2]float64{a.Probs[j], b.Probs[j]})
			}
			for _, v := range vals {
				if math.IsNaN(v[1]) || math.IsInf(v[1], 0) {
					return math.Inf(1)
				}
				worst = math.Max(worst, math.Abs(v[0]-v[1]))
			}
		}
	}
	return worst
}

// TestDecisionPagesWithTheSameAnswer is the Paging principle on the decoder
// readouts: under a budget of half the model's block pages every prompt
// evicts and re-reads blocks, and the label logits are the resident ones bit
// for bit. One question a model: each prompt reads the paged half again.
func TestDecisionPagesWithTheSameAnswer(t *testing.T) {
	for _, c := range decisionNames() {
		t.Run(c.name, func(t *testing.T) {
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "ambiguous.json"))
			run := func(paged bool) ([]float32, int64, int) {
				m := openDecision(t, c.gguf, c.kind)
				defer m.Close()
				m.SetKVF16(false)
				half := m.container.NPages() / 2
				if paged {
					m.SetPageBudget(uint64(half) * m.PageSize())
				}
				d, err := m.NewDecider(4096)
				if err != nil {
					t.Fatal(err)
				}
				defer d.Close()
				var s []float32
				for range 2 {
					if s, err = d.run(state, &qs[0], 0); err != nil {
						t.Fatal(err)
					}
				}
				return s, m.PagerReads(), m.container.NPages()
			}
			whole, _, _ := run(false)
			paged, reads, pages := run(true)
			if reads <= int64(pages) {
				t.Fatalf("a budget of half the %d block pages read %d: nothing was evicted, so nothing was tested", pages, reads)
			}
			for i := range whole {
				if whole[i] != paged[i] {
					t.Errorf("label %d: %v resident, %v paged", i, whole[i], paged[i])
				}
			}
			t.Logf("%d page reads for %d block pages over two prompts: %s", reads, pages, fmt.Sprint(paged))
		})
	}
}
