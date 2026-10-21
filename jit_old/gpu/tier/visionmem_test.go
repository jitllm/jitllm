//go:build linux

package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// visionBlocks builds n distinct ViT blocks (resident() keys on the weight
// bytes, so shared weights would balance the ledger for the wrong reason).
func visionBlocks(p *nn.LayerPlan, n int) []*nn.LayerWeights {
	out := make([]*nn.LayerWeights, n)
	for i := range out {
		q4 := func(rows, k int) nn.Weight {
			b := make([]byte, rows*(k/32)*18)
			for j := range b {
				b[j] = byte(i*31 + j)
			}
			return nn.Weight{T: quant.Q4_0, Data: b, Rows: rows, K: k}
		}
		f := func(n int) []float32 { return make([]float32, n) }
		qd := p.NHead * p.HeadDim
		out[i] = &nn.LayerWeights{
			AttnNorm: f(p.NEmbd), AttnNormB: f(p.NEmbd),
			FFNNorm: f(p.NEmbd), FFNNormB: f(p.NEmbd),
			Wq: q4(qd, p.NEmbd), Wk: q4(qd, p.NEmbd), Wv: q4(qd, p.NEmbd),
			Wo: q4(p.NEmbd, qd),
			Bq: f(qd), Bk: f(qd), Bv: f(qd), Bo: f(p.NEmbd),
			// No Gate: a ViT's MLP is ungated.
			Up: q4(p.NFFN, p.NEmbd), Down: q4(p.NEmbd, p.NFFN),
			BUp: f(p.NFFN), BDown: f(p.NEmbd),
		}
	}
	return out
}

// TestVisionTowerReturnsEveryDeviceBuffer is the device-memory ledger for the
// vision path, which the text ledger cannot reach. A tower's blocks run in a
// scratch set of their own (the non-causal geometry's, sized for the whole
// image because the picture is one run) plus LayerNorm and MLP biases a text
// block lacks; that scratch is the tier's largest single allocation and once
// had no free path at all.
func TestVisionTowerReturnsEveryDeviceBuffer(t *testing.T) {
	p := visionPlan()
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const blocks = 4
	ws := visionBlocks(p, blocks)
	for li := 0; li < blocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("vision block %d declined: %s", li, g.Err())
		}
	}
	if d.bufs == 0 {
		t.Fatal("no device buffers allocated; this gate proves nothing")
	}

	// Run the range so the vision scratch is built and used: a ViT goes
	// in one submission over every patch.
	x := make([]float32, p.MaxSeq*p.NEmbd)
	cs := make([]float32, p.NEmbd)
	g.Layers(0, blocks, 0, p.MaxSeq, x, cs, nil, nil)

	g.Close()

	if d.doubleFree != 0 {
		t.Errorf("%d buffer(s) freed twice; the ledger cannot be trusted", d.doubleFree)
	}
	if d.freed != d.bufs || d.freedBytes != d.allocBytes {
		by := map[string][2]int{}
		for _, b := range d.liveBufs {
			e := by[b.site]
			by[b.site] = [2]int{e[0] + 1, e[1] + b.n}
		}
		msg := ""
		for k, v := range by {
			msg += "\n    " + k + ": " + itoaT(v[0]) + " buffers, " + itoaT(v[1]) + " bytes"
		}
		t.Fatalf("the vision path did not return its device memory after Close:\n"+
			"  allocated %d buffers / %d bytes\n"+
			"  freed     %d buffers / %d bytes\n"+
			"  LEAKED    %d buffers / %d bytes, by allocation site:%s",
			d.bufs, d.allocBytes, d.freed, d.freedBytes,
			d.bufs-d.freed, d.allocBytes-d.freedBytes, msg)
	}
	t.Logf("%d vision blocks: %d buffers / %d bytes allocated, all returned",
		blocks, d.bufs, d.allocBytes)
}
