package model

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDecisionDoesNotAllocate is the allocation principle on the decision
// models: a warm question's prompt run and readout -- Laya's encoder, head
// and scorer, a decoder readout's prefill and label pick -- make zero engine
// heap allocations, on the host and with every block on each device present.
// Writing a question's prompt (the template, the tokenizer) is text work
// before the engine runs, and is not inside the window: it is the request's,
// as the server's JSON is.
//
// The window runs every prompt of both requests once, after two warm passes
// over the same prompts, so the Decider's scratch, its State's pages and the
// device's buffers are at their size.
func TestDecisionDoesNotAllocate(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind, noTune)
			defer m.Close()
			if !m.IsEncoder() {
				m.SetKVF16(false) // the device's history is f32
			}
			t.Run("host", func(t *testing.T) { decisionAllocs(t, m, nil, 0) })
			for _, spec := range stepDevices() {
				g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				t.Run(spec, func(t *testing.T) {
					defer g.Close()
					// A seam, the rest of the blocks on the host, and every
					// block placed where the card holds them.
					t.Run("seam", func(t *testing.T) { decisionAllocs(t, m, g, decisionBlocks(m)/2) })
					t.Run("every-block", func(t *testing.T) { decisionAllocs(t, m, g, -1) })
				})
			}
		})
	}
}

// decisionPrompt is one question variant's prompt, written ahead of the
// window: its ids and what its readout reads (a decoder's label groups, an
// encoder's marker rows).
type decisionPrompt struct {
	ids     []int32
	groups  [][]int32
	markers []int
	qt      jlm.QuestionType
	v       int
	clef    *clefPrompt // a Clef request's whole prompt
}

// decisionPrompts writes every question variant of both requests.
func decisionPrompts(t *testing.T, d *Decider) []decisionPrompt {
	t.Helper()
	var out []decisionPrompt
	for _, req := range decisionRequests {
		state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
		if d.kind == jlm.DecisionClef {
			p, err := d.clefPrompt(state, qs)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, decisionPrompt{ids: p.ids, clef: p})
			continue
		}
		for i := range qs {
			q := &qs[i]
			for v := range d.variants(q) {
				p := decisionPrompt{qt: q.Type, v: v}
				var err error
				if d.kind == jlm.DecisionLaya {
					p.ids, p.markers, err = d.layaSequence(state, q)
				} else {
					p.ids, p.groups, err = d.prompt(state, q, v)
				}
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, p)
			}
		}
	}
	return out
}

// decisionAllocs places n of a Decider's blocks on g (-1: every one; nil: the
// host) and holds a warm pass over every prompt at zero engine allocations.
func decisionAllocs(t *testing.T, m *Model, g *tier.GPU, n int) {
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	nb := decisionBlocks(m)
	placed := 0
	if g != nil {
		if err := d.SetDevice(g, n); err != nil {
			t.Fatal(err)
		}
		placed = d.DeviceBlocks()
		want := n
		if n < 0 {
			want = nb
			if m.IsEncoder() {
				want = layaExpressible(m, "")
			}
		}
		switch {
		case placed == 0:
			t.Fatalf("%s took no block: %s", g.Name(), g.Err())
		case n < 0 && placed != want:
			t.Skipf("%s held %d of %d blocks: this card is too small for the every-block arm (%s)",
				g.Name(), placed, want, g.Err())
		}
	}
	ps := decisionPrompts(t, d)
	pass := func() {
		for _, p := range ps {
			var err error
			switch {
			case p.clef != nil:
				err = d.clefForward(p.clef, d.variantScores(0, len(p.clef.opts)))
			case m.IsEncoder():
				_, err = d.layaForward(p.ids, p.qt, p.markers, p.v)
			default:
				_, err = d.forward(p.ids, p.groups, p.v)
			}
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	pass()
	// Then until a pass allocates nothing, a few at most: the host JIT's
	// prefill tuners may still be cycling after two short prompts (a small
	// fixture on the Xeon took three), and a tile it settles on is emitted
	// when it does. The window below is held to zero either way.
	for i := 0; i < 6 && countAllocs(pass).engine > 0; i++ {
	}
	var s0 int
	if g != nil {
		s0 = g.Stats().Submits
	}
	r0 := m.PagerReads()
	w := countAllocs(pass)
	if g != nil {
		// The device ran them: a submission a prompt at least, counted by
		// the device itself, and every block still placed.
		if sub := g.Stats().Submits - s0; sub < len(ps) {
			t.Fatalf("%d submission(s) for %d prompts: the placed blocks did not run on %s", sub, len(ps), g.Name())
		}
		if got := d.DeviceBlocks(); got != placed {
			t.Fatalf("the readout fell from %d device blocks to %d while it ran", placed, got)
		}
	}
	allocVerdict(t, fmt.Sprintf("%d prompt(s), %d of %d blocks placed", len(ps), placed, nb), w, len(ps),
		m.PagerReads()-r0, 0)

	// A batch: every request twice in one DecideBatch, its prompts written
	// by the batch and then run again in the window as the batch ran them
	// (runJobs) -- a decoder's as rows of the batch session.
	var reqs []DecisionRequest
	for range 2 {
		for _, req := range decisionRequests {
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
			reqs = append(reqs, DecisionRequest{State: state, Questions: qs})
		}
	}
	for range 2 {
		if _, err := d.DecideBatch(reqs); err != nil {
			t.Fatal(err)
		}
	}
	jobs := d.jobs[:d.batchN]
	if err := d.runJobs(jobs); err != nil {
		t.Fatal(err)
	}
	st0 := d.BatchStats()
	r0 = m.PagerReads()
	w = countAllocs(func() {
		if err := d.runJobs(jobs); err != nil {
			t.Fatal(err)
		}
	})
	if !m.IsEncoder() && d.kind != jlm.DecisionClef {
		st := d.BatchStats()
		if st.Steps == st0.Steps || st.MaxSeqs < 2 {
			t.Fatalf("the batch ran %d steps, at most %d prompts in one: it was not batched", st.Steps-st0.Steps, st.MaxSeqs)
		}
		if got := d.BatchDeviceBlocks(); got != placed {
			t.Fatalf("the batch session holds %d device blocks, the Decider %d", got, placed)
		}
	}
	allocVerdict(t, fmt.Sprintf("a batch of %d prompt(s), %d of %d blocks placed", len(jobs), placed, nb), w, len(jobs),
		m.PagerReads()-r0, 0)
}
