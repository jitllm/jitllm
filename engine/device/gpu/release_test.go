package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/nn"
)

// blockWeights builds one block's weights for the fake device.
func blockWeights(p *nn.LayerPlan) *nn.LayerWeights {
	q4 := func(rows, k int) nn.Weight {
		return nn.Weight{T: quant.Q4_0, Data: make([]byte, rows*(k/32)*18), Rows: rows, K: k}
	}
	return denseWeights(p, q4)
}

// denseWeights is a dense block of p's shape by role: its two norms and its
// seven matrices, each made by mk.
func denseWeights(p *nn.LayerPlan, mk func(rows, k int) nn.Weight) *nn.LayerWeights {
	kv := p.NKVHead * p.HeadDim
	return &nn.LayerWeights{
		Vectors: []nn.RoleVector{{Role: jlm.RoleAttnNorm, V: make([]float32, p.NEmbd)}, {Role: jlm.RoleFFNNorm, V: make([]float32, p.NEmbd)}},
		Mats: []nn.RoleWeight{
			{Role: jlm.RoleAttnQ, W: mk(p.NHead*p.HeadDim, p.NEmbd)},
			{Role: jlm.RoleAttnK, W: mk(kv, p.NEmbd)},
			{Role: jlm.RoleAttnV, W: mk(kv, p.NEmbd)},
			{Role: jlm.RoleAttnOut, W: mk(p.NEmbd, p.NHead*p.HeadDim)},
			{Role: jlm.RoleFFNGate, W: mk(p.NFFN, p.NEmbd)},
			{Role: jlm.RoleFFNUp, W: mk(p.NFFN, p.NEmbd)},
			{Role: jlm.RoleFFNDown, W: mk(p.NEmbd, p.NFFN)},
		},
	}
}

// held is every matrix block l holds on the card, in its page's order.
func (l *layer) held() []*resident {
	rs := make([]*resident, len(l.mats))
	for i, m := range l.mats {
		rs[i] = m.r
	}
	return rs
}

// TestReleaseReturnsTheBudget is the leak gate on runtime migration: take
// blocks, give them back, and used must return exactly to what it was. A
// missed refund shrinks the budget each cycle until PrepLayer declines a
// block that would fit, reported as a -vram problem.
func TestReleaseReturnsTheBudget(t *testing.T) {
	g, _, p := fakeTier(t)
	// fakeTier already took block 0. Add two more so a partial release has
	// something to leave behind.
	for li := 1; li < 3; li++ {
		if !g.PrepLayer(li, p, blockWeights(p)) {
			t.Fatalf("PrepLayer(%d) declined: %s", li, g.LastErr)
		}
	}
	full := g.used
	if full == 0 {
		t.Fatal("three blocks cost nothing; the accounting is not running")
	}

	g.ReleaseLayers(0, 3)
	if g.used != 0 {
		t.Errorf("after releasing every block, used = %d, want 0 (leaked %d bytes: scratch %d, composed %d of it the arenas' %d, KV %d, recurrent %d, the next pool %d)",
			g.used, g.used, g.scratch, g.instBytes, g.arenaBytes, g.KVBytes, g.RecBytes, g.recNextBytes)
	}
	if g.KVBytes != 0 {
		t.Errorf("after releasing every block, KVBytes = %d, want 0", g.KVBytes)
	}
	if len(g.layers) != 0 {
		t.Errorf("%d layers still registered after releasing all of them", len(g.layers))
	}
	// The residency map must be cleared too: g.res is keyed on the weight
	// bytes' address, and a stale entry would hand back freed buffers the
	// next time the block is taken.
	if len(g.res) != 0 {
		t.Errorf("%d resident entries survived the release; a re-take would reuse freed buffers", len(g.res))
	}

	// Re-taking must work and must cost the same as the first time.
	for li := 0; li < 3; li++ {
		if !g.PrepLayer(li, p, blockWeights(p)) {
			t.Fatalf("PrepLayer(%d) declined after release: %s", li, g.LastErr)
		}
	}
	if g.used != full {
		t.Errorf("re-taking three blocks costs %d, first time %d", g.used, full)
	}
}

// TestReleasePartialKeepsTheRest: a service giving back part of the card must
// not disturb the blocks it is keeping.
func TestReleasePartialKeepsTheRest(t *testing.T) {
	g, _, p := fakeTier(t)
	for li := 1; li < 4; li++ {
		if !g.PrepLayer(li, p, blockWeights(p)) {
			t.Fatalf("PrepLayer(%d) declined: %s", li, g.LastErr)
		}
	}
	four := g.used
	g.ReleaseLayers(2, 4) // hand back the tail, as SetGPULayers(2) would
	if len(g.layers) != 2 {
		t.Fatalf("kept %d layers, want 2", len(g.layers))
	}
	if g.layers[0] == nil || g.layers[1] == nil {
		t.Fatal("released the wrong blocks")
	}
	if g.used >= four || g.used == 0 {
		t.Errorf("used %d after releasing half of %d: want strictly between", g.used, four)
	}
	// The kept blocks must still run: their kernels are shared with the freed
	// ones by shape, so closing a shared kernel would break them.
	step(t, g, p, 0)
}
