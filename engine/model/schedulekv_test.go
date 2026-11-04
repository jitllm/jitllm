package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSchedulerUnderAKVBudgetMatchesOneWithout runs the Batcher with small KV
// pages, and each session a store and a budget far under its history, so
// pages are sealed, evicted and faulted back every step. It must give exactly
// the tokens of the same Batcher with no budget.
//
// The sequences outnumber the instance's width, so a retired sequence's place
// is taken while the others are pages ahead, every session paging its own
// history beside theirs in the same steps.
func TestSchedulerUnderAKVBudgetMatchesOneWithout(t *testing.T) {
	p := testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const page, rows, maxSeq = 8, 3, 64
	defer m.setKVPageForTest(page)()

	plen := []int{21, 4, 13, 6, 17, 3}
	gens := []int{9, 24, 5, 14, 8, 11}
	prompts := make([][]int32, len(plen))
	for r := range prompts {
		prompts[r] = make([]int32, plen[r])
		for j := range prompts[r] {
			prompts[r][j] = int32((11 + r*457 + j*31) % m.Cfg.NVocab)
		}
	}
	run := func(budget bool) ([][]int32, BatcherStats) {
		var opts []BatcherOption
		if budget {
			store := NewMemStore()
			opts = append(opts, WithSessionSetup(func(s *Session) error {
				s.SetKVStore(store)
				// Two pages a layer's worth, against histories reaching five.
				return s.SetKVBudget(uint64(2*len(s.kv.layers)) * s.kv.layers[0].pageBytes() * 2)
			}))
		}
		sc := newTestBatcher(m, rows, maxSeq, opts...)
		t.Cleanup(sc.Close)
		seqs := make([]*Seq, len(prompts))
		for r := range prompts {
			seqs[r] = &Seq{Prompt: prompts[r], MaxTokens: gens[r]}
			sc.Submit(seqs[r])
		}
		for done, steps := 0, 0; done < len(seqs); steps++ {
			if steps > 400 {
				t.Fatalf("no progress: %d of %d retired", done, len(seqs))
			}
			d, err := sc.Step()
			if err != nil {
				t.Fatal(err)
			}
			done += len(d)
		}
		out := make([][]int32, len(seqs))
		for r, s := range seqs {
			if s.Err != nil {
				t.Fatalf("seq %d: %v", r, s.Err)
			}
			out[r] = s.Out
		}
		return out, sc.Stats()
	}
	want, _ := run(false)
	got, st := run(true)
	// The configuration under test must be the one that ran.
	stored, fetched, evicted := st.KVStored, st.KVFetched, st.KVEvicted
	if evicted == 0 || fetched == 0 {
		t.Fatalf("%d page(s) stored, %d fetched, %d evicted: the budget never forced a "+
			"page out and back, so this compared two resident runs", stored, fetched, evicted)
	}
	t.Logf("%d page(s) stored, %d fetched, %d evicted", stored, fetched, evicted)
	for r := range want {
		for j := range want[r] {
			if got[r][j] != want[r][j] {
				t.Fatalf("seq %d token %d: %d under the budget, %d without\ngot  %v\nwant %v",
					r, j, got[r][j], want[r][j], got[r], want[r])
			}
		}
	}
}
