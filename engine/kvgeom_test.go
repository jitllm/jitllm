package engine

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestCardSharesPagesCutApartByItsOwnKeys: a session on the card shares a
// prefix by keys, and a key names a page of the card's pool. Where the two
// cut the history differently -- the host's pages of 16 positions, the card's
// pool of its own -- a host page's key would name a card page holding other
// positions, so the card names its pages by the keys of the positions up to
// each one's end at its own page (nameApart, restoreApart): a second session
// maps the first's prefix there, restores exactly the positions the two
// prompts share, and answers as a session that never shared.
//
// VIOLATION SIGNATURE. Let restoreApart report a fill a chunk past the one it
// mapped and the second session takes the first's suffix for its own: "the
// session restored 64 positions of the 48 shared".
func TestCardSharesPagesCutApartByItsOwnKeys(t *testing.T) {
	g, err := openGPU(t)
	if err != nil || g == nil {
		testmodels.Missing(t, "no card: %v", err)
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
	pc := a.ld.(nn.KVShareDevice).SharePage(0)
	if pc <= 0 || pc == sharePage {
		t.Fatalf("the card pages block 0 at %d positions and the host at %d: the gate selected no pool paging apart", pc, sharePage)
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
	if n := g.Stats().KVPagesMapped - mapped; n < m.Cfg.NLayer || b.KVRestored() != sharePrefix {
		t.Fatalf("the card mapped %d pages for %d blocks and the session restored %d positions of the %d shared, the card paging at %d and the host at %d",
			n, m.Cfg.NLayer, b.KVRestored(), sharePrefix, pc, sharePage)
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
