package model

import (
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestCPUComposesOnce holds the CPU device to the census the cards are held
// to (TestPlacementComposesOnce): every block a State keeps on the CPU is
// composed for it once, at its placement, and serves every step after it --
// prompts of every length and their decode, a second session, a scheduler's
// batch of several sequences -- declining none to main's kernels. Each phase
// names what it declined, by reason, and how many block steps it served.
func TestCPUComposesOnce(t *testing.T) {
	for _, tc := range []struct {
		file    string
		lengths []int
	}{
		{"stories15M-q8_0.gguf", []int{1, 5, 33, 100}},
		{"Qwen3-0.6B-Q8_0.gguf", []int{1, 5, 33, 100, 511, 512, 513, 2000}},
		{"synth-qwen35-hybrid-mtp.gguf", []int{1, 5, 33, 100, 513}},
		{"newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", []int{1, 5, 33, 100}},
		{"synth-mamba2.gguf", []int{1, 5, 33, 100}},
		{"synth-mamba1.gguf", []int{1, 5, 33, 100}},
		{"synth-lfm2.gguf", []int{1, 5, 33, 100}},
		{"synth-falconh1.gguf", []int{1, 5, 33, 100}},
		{"synth-nemotronh.gguf", []int{1, 5, 33, 100}},
		{"synth-gemma4.gguf", []int{1, 5, 33, 100}},
	} {
		t.Run(tc.file, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(tc.file)), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			m.setKVPageForTest(32)
			const decode = 2
			total := 0
			for _, n := range tc.lengths {
				total += n + decode
			}
			text := strings.Repeat("The river rose every spring until the bridges were islands, "+
				"and the clerk counted barrels of salt on the upper floor. ", 400)
			ids := m.Vocab.Encode(text, true)
			if len(ids) < total {
				t.Fatalf("%d prompt tokens, %d wanted", len(ids), total)
			}
			nl := m.Cfg.NLayer
			// placed is how many forms the model's CPU cache held after the
			// last placement: a step forms nothing.
			placed := 0
			// held is the census of s after a phase: every block composed for
			// the CPU once, nothing declined, nothing formed since the last
			// placement, and at least want more block steps served than
			// before.
			held := func(what string, s *State, before, want int) int {
				t.Helper()
				if got := m.cpuFormCount(); got != placed {
					t.Errorf("%s: %d CPU form(s) made at a step, after the placement's %d", what, got-placed, placed)
				}
				served, declined := s.CPUBlockSteps()
				for why, n := range declined {
					if n > 0 {
						t.Errorf("%s: %d block step(s) declined to main's kernels: %s (%v)", what, n, jit.DeclineName(why), s.CPUFormErr())
					}
				}
				for li, n := range s.CPUCompositions() {
					if n != 1 {
						t.Errorf("%s: block %d composed for the CPU %d time(s), want 1", what, li, n)
					}
				}
				if served-before < want {
					t.Errorf("%s: %d block steps served, want at least %d", what, served-before, want)
				}
				if t.Failed() {
					t.FailNow()
				}
				t.Logf("%s: %d block steps served", what, served-before)
				return served
			}

			s := m.NewState(total + 128)
			defer s.Close()
			placed = m.cpuFormCount()
			if got := len(s.CPUCompositions()); got != nl {
				t.Fatalf("%d blocks placed on the CPU of %d", got, nl)
			}
			served := held("the placement", s, 0, 0)
			off := 0
			for _, n := range tc.lengths {
				// A model of a short context (stories15M's 128) starts the
				// session over where the prompt and its decode would not fit.
				if s.pos+n+decode > s.maxSeq {
					s.Reset()
				}
				l, err := s.Prefill(ids[off : off+n])
				if err != nil {
					t.Fatalf("a prompt of %d: %v", n, err)
				}
				off += n
				// The first token greedy, the CPU's sampler's against the
				// definition; the second a top-k draw from its candidates.
				tok := argmaxID(l)
				got, err := s.ForwardGreedy(tok)
				if err != nil {
					t.Fatal(err)
				}
				if want := argmaxID(s.logits); got != want {
					t.Fatalf("a prompt of %d: the CPU's greedy token %d, the logits' largest %d", n, got, want)
				}
				smp := &Sampler{Temp: 0.8, TopK: 40, TopP: 1, RepeatPen: 1.1, Seed: int64(n)}
				smp.Observe(tok)
				if _, err := s.ForwardSample(got, smp); err != nil {
					t.Fatal(err)
				}
				off += decode
				served = held("a prompt of "+strconv.Itoa(n)+" tokens and its decode", s, served, (1+decode)*nl)
			}
			// Every prompt's head and both decode steps' on the CPU's
			// composed head, both decode tokens from its samplers.
			for _, c := range []struct {
				what  string
				steps func() (int, [jit.NumDeclines]int)
				want  int
			}{
				{"head", s.CPUHeadSteps, len(tc.lengths) * (1 + decode)},
				{"sampler", s.CPUSampleSteps, len(tc.lengths) * decode},
			} {
				got, declined := c.steps()
				for why, n := range declined {
					if n > 0 {
						t.Errorf("%d %s step(s) declined to main's kernels: %s (%v)", n, c.what, jit.DeclineName(why), s.CPUFormErr())
					}
				}
				if got != c.want {
					t.Errorf("%d %s steps served on the CPU, want %d", got, c.what, c.want)
				}
			}
			if t.Failed() {
				t.FailNow()
			}

			s2 := m.NewState(512)
			defer s2.Close()
			placed = m.cpuFormCount()
			l, err := s2.Prefill(ids[:100])
			if err != nil {
				t.Fatal(err)
			}
			if _, err := s2.Forward(argmaxID(l)); err != nil {
				t.Fatal(err)
			}
			held("a second session's prompt and decode", s2, 0, 2*nl)
			held("the first session beside the second", s, served, 0)

			sc := NewScheduler(m, 4, 256)
			defer sc.Close()
			placed = m.cpuFormCount()
			seqs := []*Seq{{Prompt: ids[:7], MaxTokens: 4}, {Prompt: ids[50:83], MaxTokens: 6},
				{Prompt: ids[200:300], MaxTokens: 5}, {Prompt: ids[1000:1064], MaxTokens: 3}}
			for _, q := range seqs {
				sc.Submit(q)
			}
			steps := 0
			for done := 0; done < len(seqs); steps++ {
				if steps > 200 {
					t.Fatalf("the batch made no progress: %d of %d retired", done, len(seqs))
				}
				d, err := sc.Step()
				if err != nil {
					t.Fatal(err)
				}
				done += len(d)
			}
			for i, q := range seqs {
				if q.Err != nil {
					t.Fatalf("sequence %d: %v", i, q.Err)
				}
			}
			held("a batch of four sequences", sc.st, 0, steps*nl)
			// Every decoding row's head in the batch on the CPU's head.
			bh, bd := sc.st.CPUHeadSteps()
			for why, n := range bd {
				if n > 0 {
					t.Fatalf("the batch: %d head step(s) declined to main's kernels: %s", n, jit.DeclineName(why))
				}
			}
			if bh < steps {
				t.Fatalf("the batch: %d head rows served on the CPU over %d steps", bh, steps)
			}

			// The lower half of a session's CPU blocks to a card and back,
			// mid-sequence: those composed for the CPU again at their return,
			// the rest not again, and nothing composed after it.
			t.Run("relocation", func(t *testing.T) {
				dev := testdevices.Take(t, 1, "cuda:0")[0]
				g, err := tier.OpenWith(tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				r := m.NewState(512)
				defer r.Close()
				if err := r.SetDeviceLayers(g, 0); err != nil {
					t.Fatal(err)
				}
				half := nl / 2
				turn := func(what string, at, n int) {
					t.Helper()
					l, err := r.Prefill(ids[at : at+n])
					if err != nil {
						t.Fatalf("%s: %v", what, err)
					}
					if _, err := r.Forward(argmaxID(l)); err != nil {
						t.Fatalf("%s: %v", what, err)
					}
				}
				move := func(lo, hi int, to string) {
					t.Helper()
					mv, err := r.MoveBlocks(lo, hi, to)
					if err != nil || len(mv.Declined) > 0 {
						t.Fatalf("blocks %d..%d to %s: %v (declined %v; %s)", lo, hi-1, to, err, mv.Declined, g.Err())
					}
					for li := lo; li < hi; li++ {
						if got := r.WhereIs(li); got != nn.DeviceName(to) {
							t.Fatalf("block %d is on %q, sent to %q", li, got, to)
						}
					}
				}
				census := func(what string, before, want int) int {
					t.Helper()
					served, declined := r.CPUBlockSteps()
					for why, n := range declined {
						if n > 0 {
							t.Fatalf("%s: %d block step(s) declined: %s (%v)", what, n, jit.DeclineName(why), r.CPUFormErr())
						}
					}
					for li, n := range r.CPUCompositions() {
						w := 1
						if li < half && what != "the placement" {
							w = 2
						}
						if n != w {
							t.Fatalf("%s: block %d composed for the CPU %d time(s), want %d", what, li, n, w)
						}
					}
					if served-before < want {
						t.Fatalf("%s: %d block steps served, want at least %d", what, served-before, want)
					}
					t.Logf("%s: %d block steps served", what, served-before)
					return served
				}
				turn("the first prompt", 0, 33)
				served := census("the placement", 0, 2*nl)
				move(0, half, dev)
				turn("a prompt with the lower half on "+dev, 40, 20)
				before := served
				served, _ = r.CPUBlockSteps()
				if got := served - before; got < 2*(nl-half) {
					t.Fatalf("with the lower half on %s the CPU served %d block steps, want at least %d", dev, got, 2*(nl-half))
				}
				move(0, half, nn.CPU)
				turn("a prompt with every block back", 70, 33)
				served = census("after the return", served, 2*nl)
				turn("a prompt after the return", 110, 5)
				census("after the return, again", served, 2*nl)
			})
		})
	}
}
