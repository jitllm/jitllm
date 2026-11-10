package engine

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestSharedPrefixMovesToTheCard: a session whose prefix pages it shares with
// another on the host moves every block to a card, as a session that shares
// nothing does.
//
//   - It answers as a session on the card that never shared, within the
//     card's band of the CPU, on its prompt and on forced steps after.
//   - The session it shared with stays on the CPU and reads the prefix it
//     shares as before: its steps are a solo CPU session's bit for bit.
//   - A third session sharing the prefix on the CPU that moves to the card
//     maps the pages the first move named there (KVPagesMapped): the
//     prefix is held once on the card, and its steps read it right.
func TestSharedPrefixMovesToTheCard(t *testing.T) {
	g, err := openGPU(t)
	if err != nil || g == nil {
		t.Fatalf("no card to move the blocks to: %v", err)
	}
	defer g.Close()
	// A page of the host's and of the card's pool alike: the prefix is named
	// by pages, so the two must cut it at the same positions.
	const band, page, seq = 5e-2, 256, 1024
	prompt := func(m *model.Model, i int) []int32 {
		p := make([]int32, 0, 2*page+shareSuffix)
		for j := 0; j < 2*page; j++ {
			p = append(p, int32((300+j*7)%m.Cfg.NVocab))
		}
		for j := 0; j < shareSuffix; j++ {
			p = append(p, int32((1000+i*97+j*13)%m.Cfg.NVocab))
		}
		return p
	}
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf"} {
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			store := NewSharedStore()
			open := func(card bool) *Instance {
				s := testInstance(t, m, seq)
				if card {
					if err := s.SetDeviceLayers(g, -1); err != nil {
						t.Fatal(err)
					}
				}
				mustKey(t, s, "share-move")
				if err := s.ShareKV(store); err != nil {
					t.Fatal(err)
				}
				return s
			}
			onCard := func(s *Instance, who string) {
				t.Helper()
				if n := s.GPULayers(); n != m.Cfg.NLayer {
					t.Fatalf("%s: the card took %d of %d blocks: %v", who, n, m.Cfg.NLayer, s.DeviceDeclines())
				}
			}
			near := func(what string, got, want []float32) {
				t.Helper()
				if e, _ := nmseOf(got, want); !(e < band) {
					t.Fatalf("%s: %.3e NMSE from the card's own run (bound %.0e)", what, e, band)
				}
			}

			a := open(false)
			defer a.Close()
			if _, err := a.PrefillCached(prompt(m, 0)); err != nil {
				t.Fatal(err)
			}
			b := open(false)
			defer b.Close()
			if _, err := b.PrefillCached(prompt(m, 1)); err != nil {
				t.Fatal(err)
			}
			if b.KVRestored() == 0 {
				t.Fatal("the second session restored nothing: it shares no page to move")
			}

			ref := testInstance(t, m, seq)
			defer ref.Close()
			if err := ref.SetDeviceLayers(g, -1); err != nil {
				t.Fatal(err)
			}
			onCard(ref, "the reference")
			want, err := ref.Prefill(prompt(m, 1))
			if err != nil {
				t.Fatal(err)
			}

			if err := b.SetDeviceLayers(g, -1); err != nil {
				t.Fatal(err)
			}
			onCard(b, "the sharing session")
			for k := 0; k < shareGen; k++ {
				tok := int32(argmax(want))
				if want, err = ref.Forward(tok); err != nil {
					t.Fatal(err)
				}
				got, err := b.Forward(tok)
				if err != nil {
					t.Fatal(err)
				}
				near("a step after the move", got, want)
			}

			solo := testInstance(t, m, seq)
			defer solo.Close()
			ws, err := solo.Prefill(prompt(m, 0))
			if err != nil {
				t.Fatal(err)
			}
			for k := 0; k < shareGen; k++ {
				tok := int32(argmax(ws))
				got, err := a.Forward(tok)
				if err != nil {
					t.Fatal(err)
				}
				if ws, err = solo.Forward(tok); err != nil {
					t.Fatal(err)
				}
				for i := range ws {
					if got[i] != ws[i] {
						t.Fatalf("step %d: the session left on the CPU parts from a solo run at logit %d (%v, %v)", k, i, got[i], ws[i])
					}
				}
			}

			ref2 := testInstance(t, m, seq)
			defer ref2.Close()
			if err := ref2.SetDeviceLayers(g, -1); err != nil {
				t.Fatal(err)
			}
			w2, err := ref2.Prefill(prompt(m, 2))
			if err != nil {
				t.Fatal(err)
			}
			c := open(false)
			defer c.Close()
			lc, err := c.PrefillCached(prompt(m, 2))
			if err != nil {
				t.Fatal(err)
			}
			near("the third session's prompt", lc, w2)
			mapped := g.Stats().KVPagesMapped
			if err := c.SetDeviceLayers(g, -1); err != nil {
				t.Fatal(err)
			}
			onCard(c, "the third session")
			if g.Stats().KVPagesMapped == mapped {
				t.Fatal("the third session moved to the card and mapped no page: the prefix the first move named there is held twice")
			}
			for k := 0; k < shareGen; k++ {
				tok := int32(argmax(w2))
				if w2, err = ref2.Forward(tok); err != nil {
					t.Fatal(err)
				}
				got, err := c.Forward(tok)
				if err != nil {
					t.Fatal(err)
				}
				near("the third session's step on the pages it mapped", got, w2)
			}
			t.Logf("moved a sharing session's %d blocks to the card; the third session mapped %d pages there",
				m.Cfg.NLayer, g.Stats().KVPagesMapped-mapped)
		})
	}
}
