//go:build linux

package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// convBlocks is one convolutional block of each kind over an 8x8 grid of 32
// channels: an edge residual, an inverted residual with both depthwise
// filters, and attention with a downsampled key and value. Their weights are
// distinct bytes (resident() keys on them).
func convBlocks() ([]*nn.LayerPlan, []*nn.LayerWeights) {
	const c, ce = 32, 64
	f := func(n int) []float32 { return make([]float32, n) }
	seed := 0
	q8 := func(rows, k int) nn.Weight {
		seed++
		b := make([]byte, rows*(k/32)*34)
		for j := range b {
			b[j] = byte(seed*31 + j)
		}
		return nn.Weight{T: quant.Q8_0, Data: b, Rows: rows, K: k}
	}
	ps := []*nn.LayerPlan{
		{NonCausal: true, ActWin: 32, Conv: &nn.ConvPlan{Kind: nn.ConvEdge, HIn: 8, WIn: 8, CIn: c, HOut: 8, WOut: 8,
			COut: c, CExp: ce, Stride: 1, Residual: true, Eps: 1e-6, ActWin: 32}},
		{NonCausal: true, ActWin: 32, Conv: &nn.ConvPlan{Kind: nn.ConvInverted, HIn: 8, WIn: 8, CIn: c, HOut: 4, WOut: 4,
			COut: c, CExp: ce, KStart: 3, SStart: 1, HMid: 8, WMid: 8, KMid: 3, SMid: 2, Eps: 1e-6, ActWin: 32}},
		{NonCausal: true, ActWin: 32, Conv: &nn.ConvPlan{Kind: nn.ConvAttention, HIn: 4, WIn: 4, CIn: c, HOut: 4, WOut: 4,
			COut: c, Heads: 2, KD: 32, KVK: 3, HKV: 2, WKV: 2, Residual: true, Eps: 1e-6, ActWin: 32}},
	}
	ws := []*nn.LayerWeights{
		{Up: q8(ce, 9*c), Down: q8(c, ce), Conv: &nn.ConvWeights{Norm1: f(ce), Norm2: f(c)}},
		{Up: q8(ce, c), Down: q8(c, ce), Conv: &nn.ConvWeights{Norm1: f(ce), Norm2: f(c),
			DwStart: f(9 * c), DwStartNorm: f(c), DwMid: f(9 * ce), DwMidNorm: f(ce)}},
		{Wq: q8(64, c), Wk: q8(32, c), Wv: q8(32, c), Wo: q8(c, 64), Conv: &nn.ConvWeights{Norm1: f(c),
			KDown: f(9 * c), KDownNorm: f(c), VDown: f(9 * c), VDownNorm: f(c)}},
	}
	return ps, ws
}

// TestConvTowerReturnsEveryDeviceBuffer is the device-memory ledger for the
// convolutional blocks: their vectors, their matrices and the scratch they
// share, which goes with the last of them. A release gives the budget back
// to the byte, and Close returns every buffer.
func TestConvTowerReturnsEveryDeviceBuffer(t *testing.T) {
	ps, ws := convBlocks()
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	base := d.allocBytes - d.freedBytes
	for li := range ps {
		if !g.PrepLayer(li, ps[li], ws[li]) {
			t.Fatalf("convolutional block %d declined: %s", li, g.Err())
		}
	}
	x := make([]float32, 8*8*32)
	out := make([]float32, 4*4*32)
	tap := make([]float32, 8*8*32)
	if !g.ConvLayers(0, len(ps), x, out, 0, tap) {
		t.Fatalf("ConvLayers: %s", g.Err())
	}
	if n := g.Stats().ConvBlocks; n != int64(len(ps)) {
		t.Fatalf("ran %d of %d blocks", n, len(ps))
	}
	if d.bufs == 0 {
		t.Fatal("no device buffers allocated; this gate proves nothing")
	}
	g.ReleaseLayers(0, len(ps))
	if live := d.allocBytes - d.freedBytes; live != base {
		t.Errorf("after releasing every block %d bytes are live, %d before the first", live, base)
	}
	g.Close()
	if d.doubleFree != 0 {
		t.Errorf("%d buffer(s) freed twice; the ledger cannot be trusted", d.doubleFree)
	}
	if d.freed != d.bufs || d.freedBytes != d.allocBytes {
		t.Fatalf("allocated %d buffers / %d bytes, freed %d / %d", d.bufs, d.allocBytes, d.freed, d.freedBytes)
	}
	t.Logf("%d convolutional blocks: %d buffers / %d bytes allocated, all returned", len(ps), d.bufs, d.allocBytes)
}
