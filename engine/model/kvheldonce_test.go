package model

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// kvDuplicates counts the pages of st's history held twice: a page the
// session holds privately that the store holds as well (a copy beside the
// session's), and on a block placed on a device any page the host holds of
// it, or the store holds by its key (a host mirror of the card's history).
// The page the session is writing into (the one past its sealed pages) is
// its own in flight and not counted.
func kvDuplicates(t *testing.T, st *State) int {
	t.Helper()
	kc := st.kv
	if _, none := kc.store.(NoStore); none || kc.ns == "" {
		return 0
	}
	dup := 0
	for li := kc.lo; li < len(kc.layers); li++ {
		pg := &kc.layers[li]
		if pg.p == 0 {
			continue
		}
		cur := st.pos / pg.p
		for n := 0; n <= cur && n < max(len(pg.k), cur+1); n++ {
			held := n < len(pg.k) && pg.k[n] != nil
			if st.devAt(li) {
				if held {
					dup++ // the host holds a page of the card's history
				}
				if (n+1)*pg.p <= st.pos && kvStoreHas(kc, kc.pageKey(li, n), li, n, pg.pp) {
					dup++ // the store holds it: it came home
				}
				continue
			}
			if !held || pg.isShared(n) || n == cur {
				continue
			}
			if kvStoreHas(kc, kc.pageKey(li, n), li, n, pg.pp) {
				dup++
			}
		}
	}
	return dup
}

// kvStoreHas says kc's store holds the page key names for layer li's page n.
func kvStoreHas(kc *kvCache, key string, li, n, pp int) bool {
	if key == "" {
		return false
	}
	buf := make([]float32, pp)
	return kc.store.Get(key, li*2, n, newSliceWriter(kvBytesOf(buf))) == nil
}

// TestKVIsHeldOnce is the principle that a position's history lives in one
// place: a session's pages and the prefix cache never hold the same page
// twice, and a block on a device keeps its history on the card, not beside
// it on the host -- for a chat's turns and for decisions, on the host and on
// every device present. The page a session is writing into is the one
// exception, in flight. The violation is the copying store (MemStore), which
// the cache was before: the gate must count its copies.
func TestKVIsHeldOnce(t *testing.T) {
	m := openDecision(t, "d1/d1-3B-Q8_0.gguf", jlm.DecisionLFM2D1, noTune)
	defer m.Close()
	m.SetKVF16(false)
	reqs := boardRequests(t, 3)
	prompt := make([]int32, 600)
	for i := range prompt {
		prompt[i] = int32(1000 + i*7)
	}
	chat := func(t *testing.T, g *tier.GPU, share func(*State) error) int {
		dup := 0
		for turn := range 2 {
			st := m.NewState(4096)
			if g != nil {
				if err := st.SetDeviceLayers(g, -1); err != nil {
					t.Fatal(err)
				}
			}
			if err := st.SetCacheKey("held-once-chat"); err != nil {
				t.Fatal(err)
			}
			if err := share(st); err != nil {
				t.Fatal(err)
			}
			if _, err := st.PrefillCached(prompt[:500+50*turn]); err != nil {
				t.Fatal(err)
			}
			if turn == 1 && st.KVRestored() == 0 {
				t.Fatal("the second turn restored nothing: it measured a cold prompt")
			}
			for range 4 {
				if _, err := st.Forward(42); err != nil {
					t.Fatal(err)
				}
			}
			dup += kvDuplicates(t, st)
			st.Close()
		}
		return dup
	}
	decide := func(t *testing.T, g *tier.GPU) int {
		d, err := m.NewDecider(4096)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		if g != nil {
			if err := d.SetDevice(g, -1); err != nil {
				t.Fatal(err)
			}
		}
		if on, err := d.SetPromptCache(NewSharedStore(), "held-once-decide"); err != nil || !on {
			t.Fatalf("the Decider did not take the store: %v", err)
		}
		dup := 0
		for _, b := range reqs {
			state, qs := parseRequest(t, b)
			if _, err := d.Decide(state, qs, nil); err != nil {
				t.Fatal(err)
			}
			dup += kvDuplicates(t, d.st)
		}
		if d.PromptRestored() == 0 {
			t.Fatal("the decisions restored nothing: they measured cold prompts")
		}
		return dup
	}
	arms := func(t *testing.T, g *tier.GPU) {
		// A store of the arm's own: one the host arm filled would hold the
		// device arm's pages under the same names, and count as a copy.
		shared := NewSharedStore()
		t.Run("chat", func(t *testing.T) {
			if dup := chat(t, g, func(st *State) error { return st.ShareKV(shared) }); dup != 0 {
				t.Fatalf("%d page(s) of a chat's history held twice", dup)
			}
		})
		t.Run("decide", func(t *testing.T) {
			if dup := decide(t, g); dup != 0 {
				t.Fatalf("%d page(s) of the decisions' history held twice", dup)
			}
		})
		// The violation: the copying store keeps a copy of every sealed page
		// beside the session's.
		t.Run("violation", func(t *testing.T) {
			copying := NewMemStore()
			dup := chat(t, g, func(st *State) error { st.SetKVStore(copying); return nil })
			t.Logf("the copying store: %d page(s) held twice", dup)
			if dup == 0 {
				t.Fatal("a copying store held nothing twice: the gate cannot see a copy")
			}
		})
	}
	t.Run("host", func(t *testing.T) { arms(t, nil) })
	for _, spec := range stepDevices() {
		g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
			tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			arms(t, g)
		})
	}
}
