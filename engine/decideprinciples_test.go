package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/model"
)

// decisionPrompts are every question variant's prompt ids of both requests,
// rendered ahead of the engine: the text half of a question.
func decisionPrompts(t *testing.T, d *Decider) [][]int32 {
	t.Helper()
	var out [][]int32
	for _, req := range []string{"support", "ambiguous"} {
		state, qs := decisionRequest(t, req)
		for i := range qs {
			q := &qs[i]
			shown := d.shown(q)
			labels, _, err := d.groups(q, shown)
			if err != nil {
				t.Fatal(err)
			}
			for v := range d.variants(q) {
				text, err := d.render(state, q, shown, labels, v)
				if err != nil {
					t.Fatal(err)
				}
				ids, err := d.encode(text)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, ids)
			}
		}
	}
	return out
}

// TestDecisionPagesWithTheSameAnswer is the Paging principle on the label
// readouts: under a budget of half the model's block pages every prompt
// evicts and re-reads blocks, and the answers are the resident ones bit for
// bit. A budget that read no page twice tested nothing.
func TestDecisionPagesWithTheSameAnswer(t *testing.T) {
	for _, c := range decisionCases() {
		t.Run(c.name, func(t *testing.T) {
			state, qs := decisionRequest(t, "ambiguous")
			run := func(budget bool) ([]DecisionAnswer, int64, int) {
				m := openDecision(t, c)
				defer m.Close()
				var opts []Option
				blocks := m.Container.NPages()
				if budget {
					opts = append(opts, WithPageBudget(uint64(blocks/2)*m.PageSize()))
				}
				in, err := NewInstance(m, 4096, opts...)
				if err != nil {
					t.Fatal(err)
				}
				defer in.Close()
				d, err := NewDecider(in)
				if err != nil {
					t.Fatal(err)
				}
				ans, err := d.Decide(state, qs)
				if err != nil {
					t.Fatal(err)
				}
				_, reads, _ := m.PageStats(in.fr)
				return ans, reads, blocks
			}
			whole, _, _ := run(false)
			paged, reads, pages := run(true)
			if reads <= int64(pages) {
				t.Fatalf("a budget of half the %d block pages read %d: nothing was evicted, so nothing was tested", pages, reads)
			}
			for i, q := range qs {
				w, p := whole[i], paged[i]
				if w.Noul != p.Noul || w.Choice != p.Choice || w.Score != p.Score || w.Confidence != p.Confidence {
					t.Errorf("%s: %+v resident, %+v paged", q.ID, w, p)
				}
				for j := range w.Probs {
					if w.Probs[j] != p.Probs[j] {
						t.Errorf("%s option %d: %v resident, %v paged", q.ID, j, w.Probs[j], p.Probs[j])
					}
				}
			}
			t.Logf("%d page reads for %d block pages", reads, pages)
		})
	}
}

// TestDecisionDoesNotAllocate is the allocation principle on the label
// readouts: a warm question's engine half -- the prompt's prefill, the label
// pick and the composed readout -- makes zero heap allocations, on the CPU
// and with every block on each card present. Writing the prompt (the
// template, the tokenizer) is the request's text work, outside the window.
func TestDecisionDoesNotAllocate(t *testing.T) {
	for _, c := range decisionCases() {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c)
			defer m.Close()
			t.Run("cpu", func(t *testing.T) { decisionAllocs(t, m, nil) })
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					decisionAllocs(t, m, g)
				})
			}
		})
	}
}

func decisionAllocs(t *testing.T, m *model.Model, g *gpu.GPU) {
	in, err := NewInstance(m, 4096, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	if g != nil {
		if err := in.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if in.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("the card took %d of %d blocks: %s", in.GPULayers(), m.Cfg.NLayer, g.Err())
		}
	}
	d, err := NewDecider(in)
	if err != nil {
		t.Fatal(err)
	}
	state, qs := decisionRequest(t, "support")
	if _, err := d.Decide(state, qs); err != nil {
		t.Fatal(err)
	}
	prompts := decisionPrompts(t, d)
	one := func() {
		for _, ids := range prompts {
			if err := d.ask(ids); err != nil {
				t.Fatal(err)
			}
			if !in.decideScores(d.vals, d.keep, d.params, d.prior, d.probs, d.stats) {
				t.Fatal("the readout declined")
			}
		}
	}
	one()
	one()
	if n := testing.AllocsPerRun(2, one); n != 0 {
		t.Errorf("%v allocations over %d warm prompts", n, len(prompts))
	}
}

// TestDecisionRelocatesMidPrompt is the Relocation principle on the label
// readouts: one request's questions answered with every block moving
// between the CPU and a card from one question to the next (all on the
// card, the CPU again, the upper half on the card), each answer held to
// llama.cpp's as the resident gate holds it.
func TestDecisionRelocatesMidPrompt(t *testing.T) {
	for _, c := range decisionCases() {
		t.Run(c.name, func(t *testing.T) {
			spec := testdevices.Take(t, 1, "cuda:0")[0]
			g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			m := openDecision(t, c)
			defer m.Close()
			in, err := NewInstance(m, 4096, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer in.Close()
			d, err := NewDecider(in)
			if err != nil {
				t.Fatal(err)
			}
			state, qs := decisionRequest(t, "support")
			ans := make([]DecisionAnswer, len(qs))
			moved := 0
			for i := range qs {
				switch i % 3 {
				case 0:
					err = in.SetDeviceLayers(g, -1)
				case 1:
					in.SetGPULayers(0)
				default:
					err = in.SetDeviceLayers(g, m.Cfg.NLayer/2)
				}
				if err != nil {
					t.Fatal(err)
				}
				if i%3 != 1 && in.GPULayers() > 0 {
					moved++
				}
				a, err := d.Decide(state, qs[i:i+1])
				if err != nil {
					t.Fatal(err)
				}
				ans[i] = a[0]
			}
			if moved == 0 {
				t.Fatalf("no question ran with a block on %s: %s", spec, g.Err())
			}
			worst := checkDecisions(t, qs, ans, readDecisionGold(t, c.gold+".support.llamacpp.json"), c.tol)
			t.Logf("%d of %d questions with blocks on %s; worst difference from llama.cpp %.5f", moved, len(qs), spec, worst)
		})
	}
}

// TestDecisionMovesBetweenCards moves a label readout's every block from one
// card to another between a request's questions, each answer held to
// llama.cpp's.
func TestDecisionMovesBetweenCards(t *testing.T) {
	for _, c := range decisionCases() {
		t.Run(c.name, func(t *testing.T) {
			a, b := moveDevices(t)
			g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, a+","+b, err)
			}
			defer g.Close()
			distinctCards(t, g)
			m := openDecision(t, c)
			defer m.Close()
			in, err := NewInstance(m, 4096, WithKVF16(false),
				WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}}))
			if err != nil {
				t.Fatal(err)
			}
			defer in.Close()
			if err := in.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if len(in.placedOn(a)) == 0 {
				t.Fatalf("%s took no block: %s", a, g.Err())
			}
			d, err := NewDecider(in)
			if err != nil {
				t.Fatal(err)
			}
			state, qs := decisionRequest(t, "support")
			ans := make([]DecisionAnswer, len(qs))
			on := a
			moves := 0
			for i := range qs {
				if i > 0 {
					to := b
					if on == b {
						to = a
					}
					pm, err := in.PrepareMove(0, m.Cfg.NLayer, to)
					if err != nil {
						t.Fatalf("%s -> %s: %v", on, to, err)
					}
					if _, err := pm.Commit(); err != nil {
						t.Fatalf("%s -> %s: %v", on, to, err)
					}
					on = to
					moves++
				}
				if at, _ := in.DeviceOf(0); at != on {
					t.Fatalf("block 0 is on %q, sent to %s", at, on)
				}
				a, err := d.Decide(state, qs[i:i+1])
				if err != nil {
					t.Fatal(err)
				}
				ans[i] = a[0]
			}
			worst := checkDecisions(t, qs, ans, readDecisionGold(t, c.gold+".support.llamacpp.json"), c.tol)
			t.Logf("%d moves between %s and %s; worst difference from llama.cpp %.5f", moves, a, b, worst)
		})
	}
}
