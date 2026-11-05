//go:build linux

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestTheDeviceCallbackDoesNotRereadTheBase: the device's callback must not read
// a placed block's own weights. Reading them is correct (every logit is
// identical), so only an instrument that counts reads sees the waste. It counts
// chunks, not readAt calls, because a large run fans out into several calls and
// ChunksIn is what a partial page-in is priced in.
func TestTheDeviceCallbackDoesNotRereadTheBase(t *testing.T) {
	const chunk = uint64(4096)
	// The chunk has to be smaller than what is measured: at the 1 MiB default
	// the fixture's base shares chunks with the bank and both arms read the
	// same chunks.
	m, openOpts := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true, chunk: chunk})
	defer m.Close()
	c := m.Container
	if c.ChunkBytes() != chunk {
		t.Fatalf("the container fills at %d bytes, want the %d WithChunk asked for", c.ChunkBytes(), chunk)
	}
	if c.H.NBlocks == 0 {
		t.Skip("the fixture has no block pages, so there is nothing to re-read")
	}
	s := testInstance(t, m, 64, openOpts...)
	defer s.Close()

	sel := []uint32{3, 1, uint32(m.Cfg.NExpert - 1), 2}
	selected, whole := s.ensureSelected(0), s.ensureLayer(0)

	// Each arm starts from an evicted block: a resident page's chunks are
	// already filled and both arms would read nothing.
	cost := func(f func() error) int64 {
		t.Helper()
		c.DropPage(0)
		before := c.ChunksIn()
		if err := f(); err != nil {
			t.Fatal(err)
		}
		return c.ChunksIn() - before
	}

	var w nn.LayerWeights
	narrow := cost(func() error { return selected(&w, sel) })
	var w2 nn.LayerWeights
	full := cost(func() error { return whole(&w2) })

	// The bar is zero: a bank lives in expert pages, so the callback has
	// nothing to read from the block page (the base is on the card, the sheets
	// come through Gate.Packed.Sheet). The whole-block arm must still read, or
	// the page was already filled and neither number means anything.
	t.Logf("block 0: the device's callback reads %d chunk(s) of the block page, "+
		"the host's whole-block read is %d", narrow, full)
	if full == 0 {
		t.Fatal("the whole-block arm read nothing: the page was already filled and " +
			"this gate proved nothing")
	}
	if narrow != 0 {
		t.Errorf("the device's callback read %d chunk(s) of block 0's page; with the "+
			"bank in expert pages every one of them is weight the card already holds", narrow)
	}
	// The sheets it wants come from their own pages, the selected ones and no
	// others.
	if w.Gate.Packed == nil || w.Gate.Packed.Sheet == nil {
		t.Fatal("the callback handed the device no Sheet for the gate bank")
	}
	for _, x := range sel {
		p, release, err := w.Gate.Packed.Sheet(int(x))
		if err != nil {
			t.Fatal(err)
		}
		if len(p.QS) == 0 {
			t.Errorf("expert %d: Sheet returned no bytes", x)
		}
		release()
	}
	for x := 0; x < m.Cfg.NExpert; x++ {
		picked := false
		for _, y := range sel {
			picked = picked || int(y) == x
		}
		if got := c.Resident(c.ExpertPage(m.Layers[0].Gate.E, x)); got != picked {
			t.Errorf("expert %d: page resident=%v, selected=%v", x, got, picked)
		}
	}
}
