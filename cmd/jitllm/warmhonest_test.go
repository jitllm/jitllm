package main

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
)

// TestWarmPrefillIsComputedNotRestored is the gate on speed's warm prompt
// sample (warmHonest): a fresh instance that ran every row of the prompt and
// read the last row's logits back passes, three times over on one model as
// the harness samples it. Each way a warm sample could lie fails: a prompt
// restored from a prefix cache, a second prompt on an instance that kept its
// first (prefix reuse), and logits that were not a row read back.
func TestWarmPrefillIsComputedNotRestored(t *testing.T) {
	m, err := model.Open(modelPath(t, "Llama-3.2-1B-Instruct-Q4_K_M.jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(strings.Repeat("The capital of France is Paris. ", 80), true)
	n, vocab := len(ids), m.Cfg.NVocab
	for i := 0; i < 3; i++ {
		st := m.NewInstance(n + 1)
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		if err := warmHonest(st, n, lg, vocab); err != nil {
			t.Fatalf("sample %d: an honest fresh prompt was refused: %v", i, err)
		}
		st.Close()
	}

	// Prefix reuse: the same instance, a second prompt on top of the first.
	st := m.NewInstance(2*n + 1)
	defer st.Close()
	if _, err := st.Prefill(ids); err != nil {
		t.Fatal(err)
	}
	lg, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if err := warmHonest(st, n, lg, vocab); err == nil {
		t.Fatal("a second prompt on a State holding the first passed as a fresh sample")
	}

	// Restored: the prompt's pages stored, then a fresh instance restoring them.
	store := model.NewMemStore()
	a := m.NewInstance(n + 1)
	a.SetKVStore(store)
	if err := a.SetCacheKey("warmhonest"); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillCached(ids); err != nil {
		t.Fatal(err)
	}
	a.Close()
	b := m.NewInstance(n + 1)
	defer b.Close()
	b.SetKVStore(store)
	if err := b.SetCacheKey("warmhonest"); err != nil {
		t.Fatal(err)
	}
	lg, err = b.PrefillCached(ids)
	if err != nil {
		t.Fatal(err)
	}
	if b.KVRestored() == 0 {
		t.Fatal("the store restored nothing: the restored arm tests no restore")
	}
	if err := warmHonest(b, n, lg, vocab); err == nil {
		t.Fatalf("a prompt that restored %d position(s) passed as computed", b.KVRestored())
	}

	// Not a row read back.
	c := m.NewInstance(n + 1)
	defer c.Close()
	lg, err = c.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if err := warmHonest(c, n, lg[:vocab/2], vocab); err == nil {
		t.Fatal("half a row of logits passed as the row read back")
	}
}
