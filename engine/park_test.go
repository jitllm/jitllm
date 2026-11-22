package engine

import (
	"bytes"
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// stateBytes is everything a session's next token reads of its history on the
// host: every resident KV page, key and value, and the recurrent state.
func stateBytes(s *Instance) []byte {
	var b bytes.Buffer
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		pg := &s.kv.layers[li]
		for n := range pg.mem.Len() {
			b.Write(kvBytesOf(pg.mem.K(n)))
			if !pg.latent {
				b.Write(kvBytesOf(pg.mem.V(n)))
			}
		}
	}
	for li := range s.kv.view.Blocks() {
		b.Write(kvBytesOf(s.kv.view.Conv(li)))
	}
	for li := range s.kv.view.Blocks() {
		b.Write(kvBytesOf(s.kv.view.State(li)))
	}
	return b.Bytes()
}

// decodeIDs is n greedy tokens from wherever st stands, starting at next.
func decodeIDs(t *testing.T, st *Instance, next int32, n int) ([]int32, int32) {
	t.Helper()
	var out []int32
	for range n {
		out = append(out, next)
		var err error
		if next, err = st.ForwardGreedy(next); err != nil {
			t.Fatal(err)
		}
	}
	return out, next
}

// TestParkResumesByteForByte: a session parked into a store mid-generation --
// its sealed KV pages evicted -- while another session runs on the same model,
// then resumed, holds byte for byte the history it held before, decodes the
// tokens its solo run decodes, and counts its pages out and in. Run on the
// host's paged cache with a MemStore, on a dense model and on a hybrid (whose
// recurrent state stays put and must survive the other session).
//
// A session sharing its pages (ShareKV, a server's generate) lets go of its
// sealed pages as it parks and the store holds them for it: while it is
// parked another session runs its prompt on the same store and the store is
// trimmed to nothing, and the parked session still resumes whole. The
// violation: a store that holds a parked session's pages no more than any
// other's loses them to the trim, and the resume finds no page.
func TestParkResumesByteForByte(t *testing.T) {
	for _, arm := range []struct {
		name         string
		store, share bool
	}{{"stories15M", true, false}, {"stories15M", false, false}, {"hybrid", true, false}, {"hybrid", false, false},
		{"stories15M", false, true}, {"hybrid", false, true}} {
		name := arm.name
		t.Run(fmt.Sprintf("%s/store=%v/share=%v", name, arm.store, arm.share), func(t *testing.T) {
			var m *model.Model
			var io []Option
			if name == "hybrid" {
				m, io = hybridModel(t)
			} else {
				var err error
				if m, err = model.Open(jlmOf(t, models()[name])); err != nil {
					t.Fatal(err)
				}
				defer m.Close()
			}
			prompt := []int32{1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 6, 12, 10, 14, 15, 0, 1, 2, 3, 4}
			// Pages of a key tile, so the history seals several before the
			// park and a store has something to hold.
			ctx := 128
			if m.Cfg.NCtx > 0 && m.Cfg.NCtx < ctx {
				ctx = m.Cfg.NCtx
			}
			n := ctx - len(prompt) - 4
			half := n / 2
			solo := func() []int32 {
				st := testInstance(t, m, ctx, append(append([]Option(nil), io...), withKVPage(kvKeyTile))...)
				defer st.Close()
				st.SetKVStore(NewMemStore())
				lg, err := st.Prefill(prompt)
				if err != nil {
					t.Fatal(err)
				}
				out, _ := decodeIDs(t, st, Greedy(lg), n)
				return out
			}
			want := solo()

			shared := NewSharedStore()
			sharing := func() *Instance {
				st := testInstance(t, m, ctx, append(append([]Option(nil), io...), withKVPage(kvKeyTile))...)
				mustKey(t, st, "park/shared/"+name)
				if err := st.ShareKV(shared); err != nil {
					t.Fatal(err)
				}
				return st
			}
			var st *Instance
			if arm.share {
				st = sharing()
			} else {
				st = testInstance(t, m, ctx, withKVPage(kvKeyTile))
			}
			defer st.Close()
			if arm.store {
				st.SetKVStore(NewMemStore())
			}
			var lg []float32
			var err error
			if arm.share {
				lg, err = st.PrefillCached(prompt)
			} else {
				lg, err = st.Prefill(prompt)
			}
			if err != nil {
				t.Fatal(err)
			}
			got, next := decodeIDs(t, st, Greedy(lg), half)
			before := slices.Clone(stateBytes(st))
			ps, err := st.Park()
			if err != nil {
				t.Fatal(err)
			}
			if ps.PagesOut == 0 {
				t.Fatalf("park evicted no page at position %d: the gate would compare a session with itself", st.pos)
			}
			// Another session on the same model while this one is parked.
			if other := solo(); !slices.Equal(other, want) {
				t.Fatalf("the session run while another was parked decoded %v, want %v", other, want)
			}
			if arm.share {
				// One more on the parked session's prompt and store, gone
				// before the store is trimmed to what sessions hold.
				other := sharing()
				olg, err := other.PrefillCached(prompt)
				if err != nil {
					t.Fatal(err)
				}
				decodeIDs(t, other, Greedy(olg), 4)
				other.Close()
				if p := shared.Stats().Pinned; p == 0 {
					t.Fatalf("a parked session's %d pages out are pinned nowhere in its store", ps.PagesOut)
				}
				shared.Trim(0)
			}
			in, err := st.Resume()
			if err != nil {
				t.Fatal(err)
			}
			if in != ps.PagesOut {
				t.Fatalf("resume brought %d pages back, park sent %d out", in, ps.PagesOut)
			}
			if p := shared.Stats().Pinned; p != 0 {
				t.Fatalf("the store still pins %d pages for a session that resumed", p)
			}
			if after := stateBytes(st); !bytes.Equal(before, after) {
				t.Fatalf("the resumed history differs from the parked one (%d against %d bytes)", len(after), len(before))
			}
			rest, _ := decodeIDs(t, st, next, n-half)
			got = append(got, rest...)
			if !slices.Equal(got, want) {
				t.Fatalf("parked and resumed %v\nsolo %v", got, want)
			}
			t.Logf("%s: ended at %d, %d pages out, %d in; %d tokens equal to the solo run",
				name, st.pos, ps.PagesOut, in, n)
		})
	}
}
