package model

import (
	"bytes"
	"fmt"
	"slices"
	"testing"
)

// stateBytes is everything a session's next token reads of its history on the
// host: every resident KV page, key and value, and the recurrent state.
func stateBytes(s *Instance) []byte {
	var b bytes.Buffer
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		pg := &s.kv.layers[li]
		for n := range pg.k {
			b.Write(kvBytesOf(pg.k[n]))
			if !pg.latent {
				b.Write(kvBytesOf(pg.v[n]))
			}
		}
	}
	for _, r := range s.rconv {
		b.Write(kvBytesOf(r))
	}
	for _, r := range s.rstate {
		b.Write(kvBytesOf(r))
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
func TestParkResumesByteForByte(t *testing.T) {
	for _, arm := range []struct {
		name  string
		store bool
	}{{"stories15M", true}, {"stories15M", false}, {"hybrid", true}, {"hybrid", false}} {
		name := arm.name
		t.Run(fmt.Sprintf("%s/store=%v", name, arm.store), func(t *testing.T) {
			var m *Model
			if name == "hybrid" {
				m = hybridModel(t)
			} else {
				var err error
				if m, err = Open(jlmOf(t, models()[name])); err != nil {
					t.Fatal(err)
				}
				defer m.Close()
			}
			prompt := []int32{1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 6, 12, 10, 14, 15, 0, 1, 2, 3, 4}
			// Pages of a key tile, so the history seals several before the
			// park and a store has something to hold.
			defer m.setKVPageForTest(kvKeyTile)()
			ctx := 128
			if m.Cfg.NCtx > 0 && m.Cfg.NCtx < ctx {
				ctx = m.Cfg.NCtx
			}
			n := ctx - len(prompt) - 4
			half := n / 2
			solo := func() []int32 {
				st := m.NewInstance(ctx)
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

			st := m.NewInstance(ctx)
			defer st.Close()
			if arm.store {
				st.SetKVStore(NewMemStore())
			}
			lg, err := st.Prefill(prompt)
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
			in, err := st.Resume()
			if err != nil {
				t.Fatal(err)
			}
			if in != ps.PagesOut {
				t.Fatalf("resume brought %d pages back, park sent %d out", in, ps.PagesOut)
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
