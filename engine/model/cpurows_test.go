package model

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// cpuRowsModels are the models whose prompts and batched steps the
// composer serves on the CPU, and the prompt lengths each is held at.
func cpuRowsModels() []struct {
	file    string
	lengths []int
} {
	return []struct {
		file    string
		lengths []int
	}{
		{"stories15M-q8_0.gguf", []int{5, 33, 100}},
		{"stories15M-q4_0.gguf", []int{5, 33, 100}},
		{"Qwen3-0.6B-Q8_0.gguf", []int{5, 33, 100, 513}},
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", []int{5, 33, 100}},
		// DeepSeek's latent attention and mixture: a prompt's rows through the
		// absorbed query and the one-row history.
		{"synth-deepseek-yarn-af.gguf", []int{5, 33, 100}},
	}
}

// TestComposedCPURowsServe holds the composed CPU's form of several rows --
// a prompt's chunks, a batched step's rows of several sequences -- to main's
// kernels within main's own spread, as TestComposedCPUBlocksServe holds
// decode: three arms teacher-forced on one text (main, main at another
// activation window as the control, and the composed CPU), the band
// bandRatio and bandWorst over every logit row the API gives. A prompt is
// held at several lengths, its last row and the decode after it, every
// row's history read back by the decode; a batch of three sequences, each
// prompted into its own slot and stepped together, is held to main's batch
// and to each sequence run alone on the composed CPU. The composed arms
// serve every block step, decline none and compose each block once; each
// violation -- the first history page pointed at another slot, each row's
// first key moved by one -- leaves the band.
func TestComposedCPURowsServe(t *testing.T) {
	const page, decode = 32, 6
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 60)
	for _, tc := range cpuRowsModels() {
		t.Run(tc.file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(tc.file))
			open := func(opts ...Option) *Model {
				t.Helper()
				m, err := Open(path, append([]Option{noTune()}, opts...)...)
				if err != nil {
					t.Fatal(err)
				}
				m.setKVPageForTest(page)
				return m
			}
			fm, cm := cpuFloat(t, path, page), open()
			defer fm.Close()
			defer cm.Close()
			src, layers := fixtureSource(testmodels.Path(tc.file)), fm.Cfg.NLayer
			ids := cm.Vocab.Encode(text, true)
			var want, got [][]float32
			var band, last []float64
			for _, n := range tc.lengths {
				if len(ids) < n+decode {
					t.Fatalf("%d tokens, %d wanted", len(ids), n+decode)
				}
				w := cpuPrompt(t, fm, ids, n, decode, cpuFaultNone, true)
				b := lcppBand(t, src, layers, w, lcppSeq{ids[:n], ids[n : n+decode], nil})
				g := cpuPrompt(t, cm, ids, n, decode, cpuFaultNone, true)
				med, worst, widest := cpuBandTo(t, w, g, b)
				t.Logf("a prompt of %d: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", n, med, worst, widest)
				want, band, got, last = append(want, w...), append(band, b...), append(got, g...), b
			}
			med, worst, widest := cpuBandTo(t, want, got, band)
			if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
				t.Fatalf("the composed CPU's prompts leave the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
					med, bandRatio, worst, bandWorst, widest)
			}
			// The longest prompt again with its history paged out and back:
			// the pages a step reads faulted into the layer's arena.
			n := tc.lengths[len(tc.lengths)-1]
			pw := want[len(want)-len(last):]
			pg := cpuPromptPaged(t, cm, ids, n, decode, cpuFaultNone, true, true)
			pm, pwst, pwide := cpuBandTo(t, pw, pg, last)
			t.Logf("a prompt of %d paged: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", n, pm, pwst, pwide)
			if !(pm <= bandRatio) || !(pwst <= bandWorst*pwide) {
				t.Fatalf("the composed CPU's paged prompt leaves the band: median ratio %.3f, worst %.3g (bound %.1f x %.3g)", pm, pwst, bandWorst, pwide)
			}
			for _, f := range []cpuFault{cpuFaultPage, cpuFaultRow} {
				var bad [][]float32
				for _, n := range tc.lengths {
					bad = append(bad, cpuPrompt(t, cm, ids, n, decode, f, true)...)
				}
				bm, bw, _ := cpuBandTo(t, want, bad, band)
				t.Logf("violation %d: median ratio %.3f, worst %.3g", f, bm, bw)
				if bm <= bandRatio && bw <= bandWorst*widest {
					t.Errorf("violation %d stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", f, bm, bw)
				}
			}

			// A batch of three sequences, each prompted into its own slot.
			prompts := [][]int32{ids[:7], ids[40:73], ids[100:150]}
			steps := ids[200 : 200+decode]
			bw := cpuBatch(t, fm, prompts, steps, cpuFaultNone, true)
			var seqs, sseqs []lcppSeq
			// A session's first step is three of its rows, its logits the
			// last's (cpuSessionSteps).
			srows := []int{0}
			for i := min(3, len(steps)); i <= len(steps); i++ {
				srows = append(srows, i)
			}
			for _, p := range prompts {
				seqs = append(seqs, lcppSeq{p, steps, nil})
				sseqs = append(sseqs, lcppSeq{p, steps, srows})
			}
			bb := lcppBand(t, src, layers, bw, seqs...)
			bg := cpuBatch(t, cm, prompts, steps, cpuFaultNone, true)
			med, worst, widest = cpuBandTo(t, bw, bg, bb)
			t.Logf("a batch of %d: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", len(prompts), med, worst, widest)
			if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
				t.Fatalf("the composed CPU's batch leaves the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
					med, bandRatio, worst, bandWorst, widest)
			}
			var alone [][]float32
			for _, p := range prompts {
				alone = append(alone, cpuPrompt(t, cm, append(append([]int32(nil), p...), steps...), len(p), decode, cpuFaultNone, true)...)
			}
			// cpuBatch's rows are by sequence, each its prompt's row and its
			// steps', as alone's are.
			worstAlone := 0.0
			for i := range alone {
				worstAlone = max(worstAlone, logitNMSE(bg[i], alone[i]))
			}
			t.Logf("the batch against each sequence alone: worst NMSE %.3g", worstAlone)
			if !(worstAlone <= bandWorst*widest) {
				t.Fatalf("the composed CPU's batch is %.3g from each sequence alone (bound %.1f x %.3g)", worstAlone, bandWorst, widest)
			}
			// The same sequences as sessions of their own, stepped together
			// (StepRuns), the first step three rows of each: each row's
			// history its own session's.
			sw := cpuSessionSteps(t, fm, prompts, steps, true)
			sg := cpuSessionSteps(t, cm, prompts, steps, true)
			sm, swst, swide := cpuBandTo(t, sw, sg, lcppBand(t, src, layers, sw, sseqs...))
			t.Logf("%d sessions stepped together: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", len(prompts), sm, swst, swide)
			if !(sm <= bandRatio) || !(swst <= bandWorst*swide) {
				t.Fatalf("the composed CPU's step across sessions leaves the band: median ratio %.3f, worst %.3g (bound %.1f x %.3g)", sm, swst, bandWorst, swide)
			}
			for _, f := range []cpuFault{cpuFaultPage, cpuFaultRow} {
				bad := cpuBatch(t, cm, prompts, steps, f, true)
				bm, bwst, _ := cpuBandTo(t, bw, bad, bb)
				t.Logf("batch violation %d: median ratio %.3f, worst %.3g", f, bm, bwst)
				if bm <= bandRatio && bwst <= bandWorst*widest {
					t.Errorf("batch violation %d stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", f, bm, bwst)
				}
			}
		})
	}
}

// cpuPrompt is m's logits for the prompt ids[:n] and the decode steps after
// it, teacher-forced on ids[n:], under fault f; on a composed arm every
// block step served (cpuServed).
func cpuPrompt(t *testing.T, m *Model, ids []int32, n, decode int, f cpuFault, composed bool) [][]float32 {
	return cpuPromptPaged(t, m, ids, n, decode, f, composed, false)
}

// cpuPromptPaged is cpuPrompt, and with paged set the session's history
// held to two pages a layer, the rest evicted into a store and faulted back
// as the steps read it (SetKVBudget): every block step still served, and
// pages faulted back.
func cpuPromptPaged(t *testing.T, m *Model, ids []int32, n, decode int, f cpuFault, composed, paged bool) [][]float32 {
	t.Helper()
	s := m.NewInstance(n + decode + 1)
	defer s.Close()
	s.cpu.fault = f
	if paged {
		s.SetKVStore(NewMemStore())
		pp := 0
		for li := range s.kv.layers {
			pp += 2 * 4 * 2 * s.kv.layers[li].pp
		}
		mustBudget(t, s, uint64(pp))
		defer func() {
			if s.kv.fetched == 0 || s.kv.evicted == 0 {
				t.Errorf("a prompt of %d under a budget of two pages a layer evicted %d pages and faulted back %d: the history never paged", n, s.kv.evicted, s.kv.fetched)
			}
			t.Logf("a prompt of %d paged: %d pages evicted, %d faulted back", n, s.kv.evicted, s.kv.fetched)
		}()
	}
	lg, err := s.Prefill(ids[:n])
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for i := range decode {
		if lg, err = s.Forward(ids[n+i]); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	cpuServed(t, s, fmt.Sprintf("a prompt of %d", n), (1+decode)*m.Cfg.NLayer, composed)
	return out
}

// cpuBatch is m's logits for a batch of the prompts, each prefilled into
// its own slot, then a step of every sequence together for each of steps,
// every sequence given its token: by sequence, its prompt's last row then
// its steps'.
func cpuBatch(t *testing.T, m *Model, prompts [][]int32, steps []int32, f cpuFault, composed bool) [][]float32 {
	t.Helper()
	most := 0
	for _, p := range prompts {
		most = max(most, len(p))
	}
	s := newSessBatch(m, len(prompts), most+len(steps)+1)
	defer s.Close()
	s.cpu.fault = f
	rows := make([][][]float32, len(prompts))
	for i, p := range prompts {
		lg, err := s.PrefillSeq(i, p)
		if err != nil {
			t.Fatal(err)
		}
		rows[i] = append(rows[i], append([]float32(nil), lg...))
	}
	step := make([]int32, len(prompts))
	for j := range len(steps) {
		for i := range step {
			step[i] = steps[j]
		}
		if _, err := s.ForwardBatch(step); err != nil {
			t.Fatal(err)
		}
		for i := range prompts {
			rows[i] = append(rows[i], append([]float32(nil), s.BatchLogits(i)...))
		}
	}
	cpuServed(t, s.Instance, "a batch", (len(prompts)+len(steps))*m.Cfg.NLayer, composed)
	var out [][]float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// cpuSessionSteps is m's logits for each prompt prefilled into a session
// of its own, then steps of every session together (StepRuns), the first
// three of steps, each after it one: by session, its prompt's last row then
// each step's last row. On a composed arm the leading session's census
// counts the steps' blocks: every one served.
func cpuSessionSteps(t *testing.T, m *Model, prompts [][]int32, steps []int32, composed bool) [][]float32 {
	t.Helper()
	states := make([]*Instance, len(prompts))
	rows := make([][][]float32, len(prompts))
	for i, p := range prompts {
		states[i] = m.NewInstance(len(p) + len(steps) + 1)
		defer states[i].Close()
		lg, err := states[i].Prefill(p)
		if err != nil {
			t.Fatal(err)
		}
		rows[i] = append(rows[i], append([]float32(nil), lg...))
	}
	before, _ := states[0].CPUBlockSteps()
	// The first step carries three rows of each session, its logit row
	// leading; each step after it one.
	runs, calls := make([]Run, len(states)), 0
	for j := 0; j < len(steps); {
		k := 1
		if j == 0 {
			k = min(3, len(steps))
		}
		for i, st := range states {
			runs[i] = Run{Instance: st, Tokens: steps[j : j+k], Logits: true}
		}
		lg, err := StepRuns(runs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range states {
			rows[i] = append(rows[i], append([]float32(nil), lg[i]...))
		}
		j, calls = j+k, calls+1
	}
	if composed {
		served, declined := states[0].CPUBlockSteps()
		for i, n := range declined {
			if n > 0 {
				t.Fatalf("a step across sessions: %d block steps declined: %s (%v)", n, composer.DeclineName(i), states[0].CPUFormErr())
			}
		}
		if want := calls * m.Cfg.NLayer; served-before < want {
			t.Fatalf("a step across sessions: %d block steps served, want at least %d", served-before, want)
		}
	}
	var out [][]float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// cpuServed holds s's composed CPU census: on a composed arm at least want
// block steps served, none declined, each block composed once; on main's
// arm none served.
func cpuServed(t *testing.T, s *Instance, what string, want int, composed bool) {
	t.Helper()
	served, declined := s.CPUBlockSteps()
	if !composed {
		if served != 0 {
			t.Fatalf("%s: main's arm served %d composed block steps", what, served)
		}
		return
	}
	for i, n := range declined {
		if n > 0 {
			t.Fatalf("%s: %d block steps declined: %s (%v)", what, n, composer.DeclineName(i), s.CPUFormErr())
		}
	}
	if served < want {
		t.Fatalf("%s: %d block steps served, want at least %d", what, served, want)
	}
	for li, n := range s.CPUCompositions() {
		if n != 1 {
			t.Fatalf("%s: block %d composed for the CPU %d times", what, li, n)
		}
	}
}

// TestComposedCPURowsDoNotAllocate measures the composed CPU's form of
// several rows warm: a prompt's chunk of 33 rows and a batched step of one
// row each of four sequences, each run through every block (cpuRows), and
// the whole warm chunk (Prefill) and the whole warm step (ForwardBatch)
// around them, must make no engine heap allocation, by the decode gate's
// attribution (countAllocs): on the dense models and a mixture, whose calls
// hold the experts their rows pick.
func TestComposedCPURowsDoNotAllocate(t *testing.T) {
	files := []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf"}
	for _, tc := range cpuRowsModels() {
		files = append(files, tc.file)
	}
	for _, file := range files {
		t.Run(file, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(file)), noTune())
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands. ", 20), true)
			blocks := func(s *Instance, seq, pos []int) {
				w := s.c.ResidW()
				for li := range m.Cfg.NLayer {
					why, err := s.cpuRows(li, &m.layers[li], s.bx[:len(pos)*w], s.bcs, s.bcsSWA, seq, pos, nil)
					if err != nil || why != composer.DeclineServes {
						t.Fatalf("block %d: %v (%s)", li, err, why)
					}
				}
			}
			const chunk = 33
			s := m.NewInstance(4 * chunk)
			defer s.Close()
			if _, err := s.Prefill(ids[:chunk]); err != nil {
				t.Fatal(err)
			}
			seq, pos := s.cpuRowsOf(0, 0, chunk)
			blocks(s, seq, pos)
			if w := countAllocs(func() { blocks(s, seq, pos) }); w.engine != 0 || w.unattributed > 0 {
				t.Errorf("a warm composed chunk of %d rows through every block allocates: %d engine, %d unattributed%s", chunk, w.engine, w.unattributed, w.where)
			}
			w := countAllocs(func() {
				if _, err := s.Prefill(ids[chunk : 2*chunk]); err != nil {
					t.Fatal(err)
				}
			})
			t.Logf("a warm prompt chunk of %d rows: %d engine allocation(s), %d unattributed", chunk, w.engine, max(w.unattributed, 0))
			if w.engine != 0 || w.unattributed > 0 {
				t.Errorf("a warm prompt chunk allocates%s", w.where)
			}

			const nseq = 4
			b := newSessBatch(m, nseq, 64)
			defer b.Close()
			for i := range nseq {
				if _, err := b.PrefillSeq(i, ids[i*5:i*5+3+i]); err != nil {
					t.Fatal(err)
				}
			}
			step := ids[60 : 60+nseq]
			for range 2 {
				if _, err := b.ForwardBatch(step); err != nil {
					t.Fatal(err)
				}
			}
			w = countAllocs(func() {
				if _, err := b.ForwardBatch(step); err != nil {
					t.Fatal(err)
				}
			})
			t.Logf("a warm batched step of %d sequences: %d engine allocation(s), %d unattributed", nseq, w.engine, max(w.unattributed, 0))
			if w.engine != 0 || w.unattributed > 0 {
				t.Errorf("a warm batched step allocates%s", w.where)
			}
			if served, declined := b.CPUBlockSteps(); served == 0 {
				t.Fatalf("the batch served no composed block step (declined %v)", declined)
			}
		})
	}
}
