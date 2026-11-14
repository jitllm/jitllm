package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// selectingGraph is a block graph selecting its history's blocks, by their
// pooled keys or by their positions'.
func selectingGraph(pooled bool) *graph.Graph {
	return &graph.Graph{Ops: []graph.Op{{Kind: graph.OpSelect, Attr: graph.Attrs{Pooled: pooled, Span: 4, NKVHead: 1}}}}
}

// TestSelectingHistoryPages: a block selecting its history's blocks -- by
// their pooled keys (QSA's and MiniMax-M3's block selections) or by their
// positions' keys (the lightning indexer) -- reads that history in passes
// as any attention does (its block scores or sums folded sweep by sweep), so
// its layer holds no context of its own: placed, it comes down to its
// working set on a compaction, as the plain block's does, and its pages are
// evictable. A pooled selection once held its whole context from placement,
// with no form over a streamed pass.
func TestSelectingHistoryPages(t *testing.T) {
	plain := &nn.LayerPlan{MaxSeq: 4096}
	qsa := &nn.LayerPlan{MaxSeq: 4096, KeyW: 16, Select: &nn.SelectPlan{Block: 4, Keep: 2}, Graph: selectingGraph(true)}
	idx := &nn.LayerPlan{MaxSeq: 4096, Select: &nn.SelectPlan{Block: 1, Keep: 4}, Graph: selectingGraph(false)}
	for name, c := range map[string]struct {
		p    *nn.LayerPlan
		want bool
	}{"plain": {plain, false}, "qsa": {qsa, true}, "indexer": {idx, true}} {
		if got := selectsBlocks(c.p.Graph); got != c.want {
			t.Fatalf("%s: selectsBlocks = %v, want %v", name, got, c.want)
		}
	}

	// Placed on a device, every layer comes down to its working set on a
	// compaction for another layer's room, selecting or not.
	tg, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	d := tg.devs[0]
	d.mu.Lock()
	defer d.mu.Unlock()
	for li, c := range []struct {
		name string
		p    nn.LayerPlan
	}{
		{"plain", nn.LayerPlan{MaxSeq: 4096, NKVHead: 1, HeadDim: 16}},
		{"qsa", nn.LayerPlan{MaxSeq: 4096, NKVHead: 1, HeadDim: 16, KeyW: 16, Select: &nn.SelectPlan{Block: 4, Keep: 2}, Graph: selectingGraph(true)}},
		{"indexer", nn.LayerPlan{MaxSeq: 4096, NKVHead: 1, HeadDim: 16, Select: &nn.SelectPlan{Block: 1, Keep: 4}, Graph: selectingGraph(false)}},
	} {
		if err := d.pagedLayer(1, &layer{}, li, &c.p); err != nil {
			t.Fatal(err)
		}
		kp := d.kvPages()
		l := kp.layers[li]
		held := l.n
		if err := d.shrinkKVLayer(kp, l); err != nil {
			t.Fatal(err)
		}
		if l.n != 2 || l.noEvict || l.ctx != 0 {
			t.Fatalf("%s: placed at %d pages, %d after a compaction (noEvict %v, context %d pages), want the working set of 2",
				c.name, held, l.n, l.noEvict, l.ctx)
		}
		t.Logf("%s: %d pages placed, %d after a compaction", c.name, held, l.n)
	}

	// The pool's own side: a noEvict layer offers no page, a plain one does.
	kp := newKVPool(64, 0)
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
