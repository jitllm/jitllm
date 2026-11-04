package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// kvDuplicates counts the pages of st's history held twice: a page the
// session holds privately that the store holds as well (a copy beside the
// session's), and on a block placed on a device any page the host holds of
// it, or the store holds by its key (a host mirror of the card's history).
// The page the session is writing into (the one past its sealed pages) is
// its own in flight and not counted.
func kvDuplicates(t *testing.T, st *Instance) int {
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
// it on the host -- for a chat's turns, on the host and on every device
// present. The page a session is writing into is the one
// exception, in flight. The violation is the copying store (MemStore), which
// the cache was before: the gate must count its copies.
func TestKVIsHeldOnce(t *testing.T) {
	p := testmodels.Path("d1/d1-3B-Q8_0.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS)", p)
	}
	m, err := Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.SetKVF16(false)
	prompt := make([]int32, 600)
	for i := range prompt {
		prompt[i] = int32(1000 + i*7)
	}
	chat := func(t *testing.T, g *gpu.GPU, share func(*Instance) error) int {
		dup := 0
		for turn := range 2 {
			st := m.NewInstance(4096)
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
	arms := func(t *testing.T, g *gpu.GPU) {
		// A store of the arm's own: one the host arm filled would hold the
		// device arm's pages under the same names, and count as a copy.
		shared := NewSharedStore()
		t.Run("chat", func(t *testing.T) {
			if dup := chat(t, g, func(st *Instance) error { return st.ShareKV(shared) }); dup != 0 {
				t.Fatalf("%d page(s) of a chat's history held twice", dup)
			}
		})
		if g != nil {
			s := g.Stats()
			declined := map[string]int{}
			for i, n := range s.BlockDeclines {
				if n > 0 {
					declined[gpu.DeclineName(i)] = n
				}
			}
			t.Logf("the composer served %d block steps on the card; declined %v", composedBlockSteps(g), declined)
			if composedBlockSteps(g) == 0 {
				t.Error("the composer served no block step on the card")
			}
			for _, why := range []composer.Decline{composer.DeclineStream, composer.DeclineNoRoom} {
				if n := s.BlockDeclines[why]; n > 0 {
					t.Errorf("the composer declined %d block step(s) on the card: %s", n, composer.DeclineName(int(why)))
				}
			}
		}
		// The violation: the copying store keeps a copy of every sealed page
		// beside the session's.
		t.Run("violation", func(t *testing.T) {
			copying := NewMemStore()
			dup := chat(t, g, func(st *Instance) error { st.SetKVStore(copying); return nil })
			t.Logf("the copying store: %d page(s) held twice", dup)
			if dup == 0 {
				t.Fatal("a copying store held nothing twice: the gate cannot see a copy")
			}
		})
	}
	t.Run("host", func(t *testing.T) { arms(t, nil) })
	for _, spec := range stepDevices() {
		g, err := gpu.OpenWith(append([]gpu.Option{gpu.WithDevices(spec),
			gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
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
