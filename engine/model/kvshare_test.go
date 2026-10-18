package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// shareModels are the prefix-sharing gate's models: the smallest real llama
// and a real Llama 3 (GQA), each on an f32 and a q8_0 cache.
var shareModels = []string{"stories15M-q8_0.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"}

// shareRun is one session of the sharing gate: its prompt's generation and
// every logit row it produced.
type shareRun struct {
	toks   []int32
	logits [][]float32
	s      *State
}

// shareScript is the sharing gate's workload: a system prompt of three pages
// and a suffix per session, then gen greedy tokens.
const (
	sharePage   = 16
	sharePrefix = 3 * sharePage
	shareSuffix = 20
	shareGen    = 8
	shareN      = 3
)

func sharePrompt(m *Model, i int) []int32 {
	p := make([]int32, 0, sharePrefix+shareSuffix)
	for j := 0; j < sharePrefix; j++ {
		p = append(p, int32((300+j*7)%m.Cfg.NVocab))
	}
	for j := 0; j < shareSuffix; j++ {
		p = append(p, int32((1000+i*97+j*13)%m.Cfg.NVocab))
	}
	return p
}

// shareSession runs session i: Prefill (store nil) or PrefillCached through
// store, sharing when share is set, then gen greedy tokens. The State stays
// open for the caller.
func shareSession(t *testing.T, m *Model, i int, store KVStore, share bool) shareRun {
	t.Helper()
	s := m.NewState(sharePrefix + shareSuffix + shareGen + 16)
	prompt := sharePrompt(m, i)
	var lg []float32
	var err error
	if store == nil {
		lg, err = s.Prefill(prompt)
	} else {
		mustKey(t, s, "share-gate")
		if share {
			if err := s.ShareKV(store.(*SharedStore)); err != nil {
				t.Fatal(err)
			}
		} else {
			s.SetKVStore(store)
		}
		lg, err = s.PrefillCached(prompt)
	}
	if err != nil {
		t.Fatal(err)
	}
	r := shareRun{s: s}
	for k := 0; k < shareGen; k++ {
		r.logits = append(r.logits, append([]float32(nil), lg...))
		id := int32(argmax(lg))
		r.toks = append(r.toks, id)
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	r.logits = append(r.logits, append([]float32(nil), lg...))
	return r
}

// TestSharedPrefixIsHeldOnce: sessions whose prompts share a system prompt
// hold its KV pages once, and each answers exactly as it would alone.
//
//   - N sessions on a SharedStore, all open at once, against the same N on a
//     copying MemStore: the pages they read are the same bytes, aliased in one
//     arm and copied in the other, so every logit is BIT-IDENTICAL; and each
//     against its solo run (no store), token for token.
//   - Every session after the first restored the prefix (KVRestored), so the
//     gate is not N independent prefills.
//   - Resident bytes: what the store holds plus every session's own pages is
//     one prefix and each session's suffix pages; with sharing off (MemStore)
//     each session holds its own prefix. The prefix page's reference count is
//     N.
//   - Closing every session and pruning the store returns it to zero bytes.
func TestSharedPrefixIsHeldOnce(t *testing.T) {
	for _, name := range shareModels {
		for _, kt := range []KVType{KVF32, KVQ8_0} {
			t.Run(name+"/"+kt.String(), func(t *testing.T) {
				m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithKVType(kt))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				defer m.setKVPageForTest(sharePage)()

				solo := make([]shareRun, shareN)
				for i := range solo {
					solo[i] = shareSession(t, m, i, nil, false)
					solo[i].s.Close()
				}
				mem := NewMemStore()
				copied := make([]shareRun, shareN)
				for i := range copied {
					copied[i] = shareSession(t, m, i, mem, false)
				}
				st := NewSharedStore()
				shared := make([]shareRun, shareN)
				for i := range shared {
					shared[i] = shareSession(t, m, i, st, true)
					if got := shared[i].s.KVType(); got != kt {
						t.Fatalf("session %d holds a %v cache, asked for %v", i, got, kt)
					}
				}
				for i := range shared {
					if i > 0 && shared[i].s.KVRestored() < sharePrefix {
						t.Fatalf("session %d restored %d positions: the prefix of %d was not shared",
							i, shared[i].s.KVRestored(), sharePrefix)
					}
					sameTokens(t, "shared against solo", solo[i].toks, shared[i].toks)
					for p := range shared[i].logits {
						for j := range shared[i].logits[p] {
							if math.Float32bits(shared[i].logits[p][j]) != math.Float32bits(copied[i].logits[p][j]) {
								t.Fatalf("session %d step %d logit %d: %v shared, %v copied", i, p, j,
									shared[i].logits[p][j], copied[i].logits[p][j])
							}
						}
					}
				}

				// The bytes, per attention layer and page: the prefix's pages
				// once, then each session's own.
				s0 := shared[0].s
				var pageBytes, pages uint64
				written := sharePrefix + shareSuffix + shareGen
				for li := 0; li < m.Cfg.NLayer; li++ {
					pg := &s0.kv.layers[li]
					if pg.p == 0 {
						continue
					}
					per := uint64(pg.pp) * 4 * 2
					pageBytes += per
					all := uint64((written + pg.p - 1) / pg.p)
					pre := uint64(sharePrefix / pg.p)
					pages += per * (pre + shareN*(all-pre))
				}
				// Beside its pages the store holds each prompt's final logits
				// (sealLogits), one row of NVocab a session.
				pages += shareN * 4 * uint64(m.Cfg.NVocab)
				var own, alias, copyOwn uint64
				for i := range shared {
					own += shared[i].s.KVBytes()
					alias += shared[i].s.KVSharedBytes()
					copyOwn += copied[i].s.KVBytes()
				}
				if got := st.Bytes() + own; got != pages {
					t.Errorf("store %d + sessions' own %d = %d bytes, want one prefix and each suffix: %d",
						st.Bytes(), own, got, pages)
				}
				if copyOwn <= pages {
					t.Errorf("the copying arm holds %d bytes, not more than the shared arm's %d: "+
						"the gate cannot tell sharing from not", copyOwn, pages)
				}
				key := s0.kv.pageKey(firstAttn(s0), 0)
				if r := st.Refs(key, firstAttn(s0)*2, 0); r != shareN {
					t.Errorf("the prefix's first page has %d references, want %d", r, shareN)
				}
				t.Logf("%d sessions: shared store %d + own %d bytes (sessions read %d through aliases); "+
					"copying store: sessions hold %d; one page of every layer is %d bytes",
					shareN, st.Bytes(), own, alias, copyOwn, pageBytes)

				for i := range shared {
					shared[i].s.Close()
					copied[i].s.Close()
				}
				st.Prune()
				if b := st.Bytes(); b != 0 {
					t.Fatalf("every session closed and the store pruned, and it still holds %d bytes", b)
				}
			})
		}
	}
}

// firstAttn is the first layer of s that keeps pages.
func firstAttn(s *State) int {
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		if s.kv.layers[li].p > 0 {
			return li
		}
	}
	return -1
}

// TestSharedPageIsNeverWrittenInPlace: a write into a page a session shares
// lands in a private copy (copy-on-write), and every other holder reads what
// it read before. Two sessions share the system prompt; session 1 has a row
// of junk written over a position inside the first shared page (the write
// path a rollback would take), and session 2 then generates. The control arm
// writes nothing; the shipping arm must give session 2 the control's logits
// bit for bit and leave session 1's page private; the violation arm
// (kvPages.cowFault: the write lands in place) must move session 2's logits,
// or this gate cannot see the write it is about.
func TestSharedPageIsNeverWrittenInPlace(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	defer m.setKVPageForTest(sharePage)()

	arm := func(write, inPlace bool) [][]float32 {
		st := NewSharedStore()
		a := shareSession(t, m, 0, st, true)
		defer a.s.Close()
		b := shareSession(t, m, 1, st, true)
		defer b.s.Close()
		c := shareSession(t, m, 2, st, true)
		defer c.s.Close()
		li := firstAttn(b.s)
		pg := &b.s.kv.layers[li]
		if !pg.isShared(0) {
			t.Fatal("session 1's first page is not shared: the gate has nothing to write into")
		}
		key := b.s.kv.pageKey(li, 0)
		before := st.Refs(key, li*2, 0)
		if write {
			pg.cowFault = inPlace
			l := b.s.kvlAt(li)
			junk := make([]float32, l.kvDim())
			for i := range junk {
				junk[i] = 9
			}
			pg.write(l, 0, 3, junk, junk)
			pg.cowFault = false
			if !inPlace {
				if pg.isShared(0) {
					t.Fatal("a write into a shared page left it shared")
				}
				if got := st.Refs(key, li*2, 0); got != before-1 {
					t.Fatalf("the written page's references went %d -> %d, want one fewer", before, got)
				}
			}
		}
		var out [][]float32
		lg := c.logits[len(c.logits)-1]
		for k := 0; k < 4; k++ {
			id := int32(argmax(lg))
			var err error
			if lg, err = c.s.Forward(id); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	same := func(a, b [][]float32) bool {
		for p := range a {
			for i := range a[p] {
				if math.Float32bits(a[p][i]) != math.Float32bits(b[p][i]) {
					return false
				}
			}
		}
		return true
	}
	want := arm(false, false)
	if got := arm(true, false); !same(want, got) {
		t.Fatal("a write by another session moved this session's logits: the shared page was written in place")
	}
	if bad := arm(true, true); same(want, bad) {
		t.Fatal("a write in place into the shared page left the other session's logits unchanged: " +
			"this gate cannot see the write it guards against")
	}
}

// TestShareKVRefusesByName: the sessions that cannot share say why.
func TestShareKVRefusesByName(t *testing.T) {
	st := NewSharedStore()
	for _, c := range []struct{ name, why string }{
		{"synth-deepseek4.gguf", "compressed entries"},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(c.name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			s := m.NewState(64)
			defer s.Close()
			mustKey(t, s, "share-gate")
			if err := s.ShareKV(st); err == nil || !strings.Contains(err.Error(), c.why) {
				t.Fatalf("ShareKV: %v, want a refusal naming %q", err, c.why)
			}
		})
	}
	t.Run("batch", func(t *testing.T) {
		m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		b := m.NewBatch(2, 64)
		defer b.Close()
		if err := b.ShareKV(st); err == nil || !strings.Contains(err.Error(), "batch") {
			t.Fatalf("ShareKV on a batch: %v", err)
		}
	})
}
