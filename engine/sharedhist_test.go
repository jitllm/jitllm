package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestAPrefixRestoresPastSharedHistories: sessions of Gemma 4 and Gemma 3n
// restore a prompt's prefix a session before them holds, though their
// KV-sharing blocks keep no history of their own: how much restores is
// decided by the blocks that do (kvCache.covers), and a block reading
// another's history reads the pages restored for that one. Each session
// after the first restores the prefix (KVRestored), answers as the same
// session over a copying store does, bit for bit, and as it does alone,
// token for token.
func TestAPrefixRestoresPastSharedHistories(t *testing.T) {
	// A page of the sliding blocks' window's size, and a prefix of whole
	// pages: what a windowed block stores is its pages whole.
	const page, prefix, suffix, gen, n = 32, 4 * 32, 20, 8, 3
	for _, f := range []struct {
		name string
		open func(*testing.T, string) (*model.Model, *c6Golden, []Option)
	}{{"synth-gemma4-e", openGemma4}, {"synth-gemma3n", openGemma3n}} {
		t.Run(f.name, func(t *testing.T) {
			m, _, io := f.open(t, f.name)
			defer m.Close()
			opts := append(append([]Option(nil), io...), sharedCPU(t), withKVPage(page))
			prompt := func(i int) []int32 {
				p := make([]int32, 0, prefix+suffix)
				for j := range prefix {
					p = append(p, int32((300+j*7)%m.Cfg.NVocab))
				}
				for j := range suffix {
					p = append(p, int32((1000+i*97+j*13)%m.Cfg.NVocab))
				}
				return p
			}
			run := func(i int, store KVStore, share bool) shareRun {
				t.Helper()
				s := testInstance(t, m, prefix+suffix+gen+16, opts...)
				var lg []float32
				var err error
				if store == nil {
					lg, err = s.Prefill(prompt(i))
				} else {
					mustKey(t, s, "shared-history")
					if share {
						err = s.ShareKV(store.(*SharedStore))
					} else {
						s.SetKVStore(store)
					}
					if err == nil {
						lg, err = s.PrefillCached(prompt(i))
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				r := shareRun{s: s}
				for range gen {
					r.logits = append(r.logits, append([]float32(nil), lg...))
					id := Greedy(lg)
					r.toks = append(r.toks, id)
					if lg, err = s.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				r.logits = append(r.logits, append([]float32(nil), lg...))
				return r
			}
			mem, st := NewMemStore(), NewSharedStore()
			for i := range n {
				solo := run(i, nil, false)
				solo.s.Close()
				copied := run(i, mem, false)
				defer copied.s.Close()
				shared := run(i, st, true)
				defer shared.s.Close()
				if i > 0 && (shared.s.KVRestored() < prefix || copied.s.KVRestored() < prefix) {
					t.Fatalf("session %d restored %d positions shared and %d copied: the prefix of %d was not restored",
						i, shared.s.KVRestored(), copied.s.KVRestored(), prefix)
				}
				sameTokens(t, "shared against solo", solo.toks, shared.toks)
				for p := range shared.logits {
					for j := range shared.logits[p] {
						if math.Float32bits(shared.logits[p][j]) != math.Float32bits(copied.logits[p][j]) {
							t.Fatalf("session %d step %d logit %d: %v shared, %v copied", i, p, j, shared.logits[p][j], copied.logits[p][j])
						}
					}
				}
			}
		})
	}
}
