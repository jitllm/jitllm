package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestSelectionHistoryStaysOnTheCard: a block whose attention selects from
// its whole history (QSA's and MiniMax-M3's block selections, the lightning
// indexer) cannot read pages at home through a streamed pass. On a card
// holding five of Qwen3.8-Flash-Next's blocks the tier sent pages home anyway
// and the next call was refused, failing the device mid-prompt. Such a block's
// pool is now never evictable and the context it was asked for is part of
// what admitting it costs. The plain block is the control: its pages may go.
func TestSelectionHistoryStaysOnTheCard(t *testing.T) {
	plain := &nn.LayerPlan{MaxSeq: 4096}
	qsa := &nn.LayerPlan{MaxSeq: 4096, Q4E: &nn.Q4EPlan{QSARatio: 4}}
	msa := &nn.LayerPlan{MaxSeq: 4096, IdxHeads: 4, IdxBlock: 8}
	idx := &nn.LayerPlan{MaxSeq: 4096, IdxHeads: 4}
	for name, c := range map[string]struct {
		p    *nn.LayerPlan
		want bool
	}{"plain": {plain, false}, "qsa": {qsa, true}, "msa": {msa, true}, "indexer": {idx, true}} {
		if got := selectsHistory(c.p); got != c.want {
			t.Fatalf("%s: selectsHistory = %v, want %v", name, got, c.want)
		}
	}

	kp := newKVPool(64, 0)
	if got, want := contextPages(kp, qsa.MaxSeq, 1), 1+4096/64; got != want {
		t.Fatalf("a 4096-position context is %d pages, want %d", got, want)
	}
	if got, want := contextPages(kp, qsa.MaxSeq, 3), 1+3*(4096/64); got != want {
		t.Fatalf("three sequences hold %d pages, want %d", got, want)
	}

	// The pool's own side: a noEvict layer offers no page, a plain one does.
	s := seqID{1, 0}
	kp.written = map[seqID]int{s: 640}
	for _, noEvict := range []bool{false, true} {
		l := &kvLayerPool{owned: map[seqID][]uint32{s: make([]uint32, 10)}, noEvict: noEvict}
		got, want := kp.evictable(l, s), map[bool]int{false: 10, true: 0}[noEvict]
		if got != want {
			t.Fatalf("noEvict %v: %d page(s) evictable, want %d", noEvict, got, want)
		}
	}
}
