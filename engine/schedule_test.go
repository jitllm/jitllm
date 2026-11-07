package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSchedulerMatchesSoloSequences is the gate that makes the scheduler worth
// having: a sequence run through it must be token-for-token identical to the
// same sequence run alone.
//
// The sequences outnumber the rows, so every row serves at least twice and the
// second occupant finds any stale position or unreturned cache. They have
// different lengths, so rows retire at different steps and the batch is
// ragged; equal lengths survive even a shared position counter.
func TestSchedulerMatchesSoloSequences(t *testing.T) {
	// Plus a hybrid: six requests through three rows reuse every row, and a
	// recurrent block has no position to rewind, so a reused row that kept the
	// last sequence's summary parts from its solo run at the first token.
	all := map[string]string{"qwen35": testmodels.Path("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf")}
	for name, path := range models() {
		all[name] = path
	}
	for name, path := range all {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "model not present: %s", path)
			}
			// An f32 cache: a session's rows read a binary16 history through the
			// rows' form and alone through the one-row form, which round apart
			// (qwen3moe, qwen35, qwen2vl); TestSessionsStepWithinTheBandOnF16KV
			// holds the binary16 default to a band.
			instOpts := []Option{WithKVF16(false)}
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()

			const nseq, nreq = 3, 6
			plen := []int{9, 4, 13, 6, 11, 3}
			want := []int{5, 8, 3, 7, 4, 6}
			maxSeq := 32
			id := func(a, b int) int32 { return int32((11 + a*457 + b*31) % m.Cfg.NVocab) }

			prompts := make([][]int32, nreq)
			for r := range prompts {
				prompts[r] = make([]int32, plen[r])
				for j := range prompts[r] {
					prompts[r][j] = id(r, j)
				}
			}

			// The oracle: each sequence alone, greedy, same budget.
			solo := make([][]int32, nreq)
			for r := range prompts {
				s := testInstance(t, m, maxSeq, instOpts...)
				lg, err := s.Prefill(prompts[r])
				if err != nil {
					s.Close()
					t.Fatal(err)
				}
				out := []int32{Greedy(lg)}
				for len(out) < want[r] {
					if lg, err = s.Forward(out[len(out)-1]); err != nil {
						s.Close()
						t.Fatal(err)
					}
					out = append(out, Greedy(lg))
				}
				s.Close()
				solo[r] = out
			}

			// The batch's instance reads the same f32 history as the solo
			// runs: one made with the default reads binary16 and rounds apart.
			sc := newTestBatcherOn(m, instOpts, nseq, maxSeq)
			defer sc.Close()
			seqs := make([]*Seq, nreq)
			for r := range prompts {
				seqs[r] = &Seq{Prompt: prompts[r], MaxTokens: want[r]}
				sc.Submit(seqs[r])
			}
			finished, guard := 0, 0
			for finished < nreq {
				done, err := sc.Step()
				if err != nil {
					t.Fatal(err)
				}
				finished += len(done)
				if guard++; guard > nreq*maxSeq {
					t.Fatalf("scheduler did not drain: %d of %d finished", finished, nreq)
				}
			}
			if sc.Live() != 0 || sc.Pending() != 0 {
				t.Errorf("drained but %d live and %d pending", sc.Live(), sc.Pending())
			}
			for r, s := range seqs {
				if s.Err != nil {
					t.Errorf("seq %d: %v", r, s.Err)
					continue
				}
				if len(s.Out) != want[r] {
					t.Errorf("seq %d: %d tokens, want %d", r, len(s.Out), want[r])
					continue
				}
				// The sequences run free, so the first token they part at ends
				// the comparison: it must be a tie, the two tokens closer in the
				// solo run's logits than the model's forms read apart at that
				// prefix (devFlip, RULE 8).
				for j := range s.Out {
					if s.Out[j] != solo[r][j] {
						gap, pert := devFlip(t, m, nil, prompts[r], solo[r][:j], solo[r][j], s.Out[j], instOpts...)
						if gap > pert {
							t.Errorf("seq %d token %d: scheduled %d, solo %d, %.4f apart against a %.4f reach of the band -- not a tie",
								r, j, s.Out[j], solo[r][j], gap, pert)
						} else {
							t.Logf("seq %d token %d: scheduled %d, solo %d, a tie (%.4f apart, the band reaching %.4f)",
								r, j, s.Out[j], solo[r][j], gap, pert)
						}
						break
					}
				}
			}
		})
	}
}

// TestSchedulerStopsOnStop covers the other retirement path, since MaxTokens
// alone would let a Stop bug ship: every sequence would still end, on budget.
func TestSchedulerStopsOnStop(t *testing.T) {
	path := models()["stories15M"]
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "model not present: %s", path)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	sc := newTestBatcher(m, 2, 32)
	defer sc.Close()
	// Stop on whatever the first token turns out to be, so the sequence ends
	// after exactly one token regardless of what the model says.
	var first int32 = -1
	s := &Seq{Prompt: []int32{1, 2, 3}, MaxTokens: 20,
		Stop: func(id int32) bool {
			if first < 0 {
				first = id
			}
			return id == first
		}}
	sc.Submit(s)
	for i := 0; i < 40 && !s.Done; i++ {
		if _, err := sc.Step(); err != nil {
			t.Fatal(err)
		}
	}
	if !s.Done {
		t.Fatal("Stop never retired the sequence")
	}
	if len(s.Out) != 1 {
		t.Errorf("stopped after %d tokens, want 1", len(s.Out))
	}
	if sc.Live() != 0 {
		t.Errorf("%d rows still live after the only sequence stopped", sc.Live())
	}
}

// TestSchedulerRejectsAnOversizedPrompt checks that a failing admission frees
// its row rather than wedging it: one bad request must not cost a slot.
func TestSchedulerRejectsAnOversizedPrompt(t *testing.T) {
	path := models()["stories15M"]
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "model not present: %s", path)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	sc := newTestBatcher(m, 1, 16)
	defer sc.Close()
	big := make([]int32, 64) // longer than maxSeq
	for i := range big {
		big[i] = 1
	}
	bad := &Seq{Prompt: big, MaxTokens: 4}
	good := &Seq{Prompt: []int32{1, 2, 3}, MaxTokens: 4}
	sc.Submit(bad)
	sc.Submit(good)
	// Both must come back from Step: a caller that waits for its sequence in
	// the returned list would otherwise wait forever on the failed one.
	returned := map[*Seq]bool{}
	for i := 0; i < 40 && !good.Done; i++ {
		done, err := sc.Step()
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range done {
			returned[d] = true
		}
	}
	if bad.Err == nil {
		t.Error("an oversized prompt should have failed admission")
	}
	if !returned[bad] || !returned[good] {
		t.Errorf("Step returned the failed sequence %v and the finished one %v; both retired", returned[bad], returned[good])
	}
	if !good.Done || len(good.Out) != 4 {
		t.Errorf("the row was not freed for the next request: done=%v out=%d", good.Done, len(good.Out))
	}
}
