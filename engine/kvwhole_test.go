package engine

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestARepeatedPromptOfWholePagesRestoresAllButItsLast: a prompt of whole
// pages that a session before it ran, restored by a session that runs its
// last position (RestorePrefix, the step loop's restore), takes every
// position but that one from the shared store, the last page at the fill
// under it, and answers as it does alone. Against a restore that gives up
// the page its bound falls inside, it restores a page short; against one
// that stops at the chunk under the bound, a chunk short. A copying store,
// which names no page by a fill it was not sealed at, restores the whole
// pages under the bound.
func TestARepeatedPromptOfWholePagesRestoresAllButItsLast(t *testing.T) {
	const page, pages, gen = 64, 2, 8
	m, err := model.Open(jlmOf(t, testmodels.Path(shareModels()[0])))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	opts := []Option{sharedCPU(t), withKVPage(page)}
	prompt := make([]int32, page*pages)
	for j := range prompt {
		prompt[j] = int32((300 + j*7) % m.Cfg.NVocab)
	}
	run := func(store KVStore, share bool) (int, []int32) {
		t.Helper()
		s := testInstance(t, m, len(prompt)+gen+16, opts...)
		t.Cleanup(func() { s.Close() })
		n := 0
		if store != nil {
			mustKey(t, s, "whole-pages")
			if share {
				if err := s.ShareKV(store.(*SharedStore)); err != nil {
					t.Fatal(err)
				}
			} else {
				s.SetKVStore(store)
			}
			if n, err = s.RestorePrefix(prompt); err != nil {
				t.Fatal(err)
			}
		}
		lg, err := s.Prefill(prompt[n:])
		if err != nil {
			t.Fatal(err)
		}
		if store != nil {
			if err := s.SealPrompt(lg); err != nil {
				t.Fatal(err)
			}
		}
		var toks []int32
		for range gen {
			id := Greedy(lg)
			toks = append(toks, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return n, toks
	}
	_, solo := run(nil, false)
	st, mem := NewSharedStore(), NewMemStore()
	if n, _ := run(st, true); n != 0 {
		t.Fatalf("the first session restored %d positions from an empty store", n)
	}
	n, toks := run(st, true)
	if n != len(prompt)-1 {
		t.Fatalf("the repeated prompt of %d positions restored %d from the shared store, want %d", len(prompt), n, len(prompt)-1)
	}
	sameTokens(t, "restored all but its last against solo", solo, toks)
	run(mem, false)
	n, toks = run(mem, false)
	if n != len(prompt)-page {
		t.Fatalf("the repeated prompt restored %d positions from a copying store, want the %d of its whole pages under the bound", n, len(prompt)-page)
	}
	sameTokens(t, "copied against solo", solo, toks)
}
