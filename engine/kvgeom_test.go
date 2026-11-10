package engine

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestCardNamesOnlyPagesCutAlike: a session on the card shares a prefix by
// the host's page keys, and a key names a page of the card's pool. Where the
// two cut the history differently -- the host's pages of 16 positions, the
// card's pool of 256 -- a key names a card page holding other positions, so
// the card names and maps nothing: a second session restores none of the
// prefix there and answers as a session that never shared.
func TestCardNamesOnlyPagesCutAlike(t *testing.T) {
	g, err := openGPU(t)
	if err != nil || g == nil {
		t.Fatalf("no card: %v", err)
	}
	defer g.Close()
	const seq = 1024
	m, err := model.Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	store := NewSharedStore()
	open := func(share bool) *Instance {
		s := testInstance(t, m, seq, withKVPage(sharePage))
		if err := s.SetDeviceLayers(g, -1); err != nil {
			t.Fatal(err)
		}
		if n := s.GPULayers(); n != m.Cfg.NLayer {
			t.Fatalf("the card took %d of %d blocks: %v", n, m.Cfg.NLayer, s.DeviceDeclines())
		}
		if share {
			mustKey(t, s, "geometry")
			if err := s.ShareKV(store); err != nil {
				t.Fatal(err)
			}
		}
		return s
	}
	a := open(true)
	defer a.Close()
	if _, err := a.PrefillCached(sharePrompt(m, 0)); err != nil {
		t.Fatal(err)
	}
	ref := open(false)
	defer ref.Close()
	want, err := ref.Prefill(sharePrompt(m, 1))
	if err != nil {
		t.Fatal(err)
	}
	mapped := g.Stats().KVPagesMapped
	b := open(true)
	defer b.Close()
	got, err := b.PrefillCached(sharePrompt(m, 1))
	if err != nil {
		t.Fatal(err)
	}
	if n := g.Stats().KVPagesMapped - mapped; n != 0 || b.KVRestored() != 0 {
		t.Fatalf("the card mapped %d pages and the session restored %d positions by keys of %d positions on a pool of other pages",
			n, b.KVRestored(), sharePage)
	}
	for k := 0; ; k++ {
		if e, _ := nmseOf(got, want); !(e < 1e-6) {
			t.Fatalf("step %d: %.3e NMSE from a session that never shared", k, e)
		}
		if k == shareGen {
			break
		}
		tok := int32(argmax(want))
		if want, err = ref.Forward(tok); err != nil {
			t.Fatal(err)
		}
		if got, err = b.Forward(tok); err != nil {
			t.Fatal(err)
		}
	}
}
