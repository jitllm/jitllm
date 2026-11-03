//go:build linux

package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// mlaPlan is a latent-attention block's geometry. The widths are all different
// and none a multiple of another (key head 96 = 64 nope + 32 rotary, value head
// 128, latent 160), so confusing one for another cannot go unnoticed.
func mlaPlan() *nn.LayerPlan {
	p := &nn.LayerPlan{
		NEmbd: 256, NHead: 4, NKVHead: 4, HeadDim: 96, NRot: 32, NFFN: 512,
		KVLoraRank: 160, QLoraRank: 192, HeadDimV: 128,
		MaxSeq: 64, Act: nn.ActSiLU, RMSEps: 1e-5, AttnScale: 0.1, RopeBase: 10000,
	}
	p.Graph = latentBlockGraph(p)
	withRotary(p)
	return p
}

// latentBlockGraph is p's block as the front end states DeepSeek's latent
// attention: the two-step query, the latent row and its rotary key from
// kv_a, each head's query absorbed through its sheet of kv_b's key half, the
// history one row a position, each head's latent result through its sheet of
// the value half, then the output projection and the FFN.
func latentBlockGraph(p *nn.LayerPlan) *graph.Graph {
	b := graph.NewBlock(0)
	nh, hd, lat, rot, hdv := p.NHead, p.HeadDim, p.KVLoraRank, p.NRot, p.HeadDimV
	nope, row := hd-rot, lat+rot
	slice := func(v graph.ValueID, name string, headDim, off, width, out int) graph.ValueID {
		return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: name, Width: out}}, nil,
			graph.Attrs{HeadDim: headDim, Offset: off, Width: width}, nil, nil)[0]
	}
	sheets := func(v graph.ValueID, name string, r jlm.Role, k, rows int) graph.ValueID {
		return b.Emit(graph.OpExpertMatVec, []graph.ValueID{v}, []graph.Result{{Name: name, Width: nh * rows}},
			[]graph.Binding{{Role: r, K: k, Rows: rows, Sheets: nh}}, graph.Attrs{}, nil, nil)[0]
	}
	x := b.In("x", p.NEmbd)
	n := b.Norm(x, jlm.RoleAttnNorm, p.RMSEps)
	q := b.MatVec(b.Norm(b.MatVec(n, jlm.RoleAttnQA, p.QLoraRank), jlm.RoleAttnQANorm, p.RMSEps), jlm.RoleAttnQB, nh*hd)
	kv := b.MatVec(n, jlm.RoleAttnKVA, row)
	latent := b.Norm(slice(kv, "latent", 0, 0, lat, lat), jlm.RoleAttnKVANorm, p.RMSEps)
	ra := graph.Attrs{HeadDim: rot, NRot: rot, Base: p.RopeBase, Layout: graph.RopeNorm}
	qpe := b.Emit(graph.OpRope, []graph.ValueID{slice(q, "q.pe", hd, nope, rot, nh*rot)},
		[]graph.Result{{Name: "q.pe.rope", Width: nh * rot}}, nil, ra, nil, nil)[0]
	kpe := b.Emit(graph.OpRope, []graph.ValueID{slice(kv, "k.pe", 0, lat, rot, rot)},
		[]graph.Result{{Name: "k.pe.rope", Width: rot}}, nil, ra, nil, nil)[0]
	abs := sheets(slice(q, "q.nope", hd, 0, nope, nh*nope), "q.abs", jlm.RoleAttnKB, nope, lat)
	qrow := b.Emit(graph.OpConcat, []graph.ValueID{abs, qpe}, []graph.Result{{Name: "q.row", Width: nh * row}}, nil,
		graph.Attrs{NHead: nh}, nil, nil)[0]
	krow := b.Emit(graph.OpConcat, []graph.ValueID{latent, kpe}, []graph.Result{{Name: "kv.row", Width: row}}, nil,
		graph.Attrs{}, nil, nil)[0]
	hist := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(krow, krow, 0)}
	att := b.AttendOf(qrow, hist, graph.Attrs{NHead: nh, NKVHead: 1, HeadDim: row, Scale: p.AttnScale})
	o := sheets(slice(att, "att.latent", row, 0, lat, nh*lat), "att.v", jlm.RoleAttnVB, lat, hdv)
	x = b.Add(x, b.MatVec(o, jlm.RoleAttnOut, p.NEmbd))
	b.Out(swiGLU(b, p, x))
	return b.G
}

// mlaBlocks builds n distinct latent-attention blocks (resident() keys on the
// weight bytes).
func mlaBlocks(p *nn.LayerPlan, n int) []*nn.LayerWeights {
	out := make([]*nn.LayerWeights, n)
	nope := p.HeadDim - p.NRot
	for i := range out {
		q4 := func(rows, k int) nn.Weight {
			b := make([]byte, rows*(k/32)*18)
			for j := range b {
				b[j] = byte(i*31 + j)
			}
			return nn.Weight{T: quant.Q4_0, Data: b, Rows: rows, K: k}
		}
		f := func(n int) []float32 { return make([]float32, n) }
		out[i] = &nn.LayerWeights{
			AttnNorm: f(p.NEmbd), FFNNorm: f(p.NEmbd),
			// No Wq, Wk or Wv: the query is two-step (Wqa/Wqb) and kv_a
			// serves both key and value.
			Wo:  q4(p.NEmbd, p.NHead*p.HeadDimV),
			Wqa: q4(p.QLoraRank, p.NEmbd), Wqb: q4(p.NHead*p.HeadDim, p.QLoraRank),
			Wkva: q4(p.KVRow(), p.NEmbd),
			// The two absorb banks: one sheet per HEAD.
			Wkb: q4(p.NHead*p.KVLoraRank, nope),
			Wvb: q4(p.NHead*p.HeadDimV, p.KVLoraRank),
			// The two latent norms, which each free walk must reach.
			QANorm: f(p.QLoraRank), KVANorm: f(p.KVLoraRank),
			Gate: q4(p.NFFN, p.NEmbd), Up: q4(p.NFFN, p.NEmbd), Down: q4(p.NEmbd, p.NFFN),
		}
	}
	return out
}

// TestLatentAttentionReturnsEveryDeviceBuffer is the device-memory ledger for
// the MLA path. A new weight must reach both free walks, and MLA adds five
// matrices and two norms at once. It also runs a submission, so the MLA
// scratch allocated in initScratch is covered too.
func TestLatentAttentionReturnsEveryDeviceBuffer(t *testing.T) {
	p := mlaPlan()
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const blocks = 4
	ws := mlaBlocks(p, blocks)
	for li := 0; li < blocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("latent block %d declined: %s", li, g.Err())
		}
	}
	if d.bufs == 0 {
		t.Fatal("no device buffers allocated; this gate proves nothing")
	}
	// The configuration under test must have been selected: blocks run as
	// ordinary attention would balance the ledger too.
	for li := 0; li < blocks; li++ {
		l := g.devs[0].layers[li]
		if l == nil || !l.mla {
			t.Fatalf("block %d is on the device and is not a latent block", li)
		}
		for _, w := range []struct {
			name string
			r    *resident
		}{{"attn_q_a", l.wqa}, {"attn_q_b", l.wqb}, {"attn_kv_a", l.wkva},
			{"attn_k_b", l.wkb}, {"attn_v_b", l.wvb}} {
			if w.r == nil || !w.r.ok {
				t.Fatalf("block %d: %s is not resident, so this gate cannot see it leak",
					li, w.name)
			}
		}
		if l.nQA == nil || l.nKVA == nil {
			t.Fatalf("block %d: a latent norm was never uploaded", li)
		}
	}

	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	g.Layers(0, blocks, 0, 1, x, cs, nil, nil)

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
		t.Fatalf("the latent attention path did not return its device memory after Close:\n"+
			"  allocated %d buffers / %d bytes\n"+
			"  freed     %d buffers / %d bytes\n"+
			"  LEAKED    %d buffers / %d bytes, by allocation site:%s",
			d.bufs, d.allocBytes, d.freed, d.freedBytes,
			d.bufs-d.freed, d.allocBytes-d.freedBytes, msg)
	}
	t.Logf("%d latent blocks: %d buffers / %d bytes allocated, all returned",
		blocks, d.bufs, d.allocBytes)
}

// TestLatentAttentionReturnsItsMemoryOnRelease is the same ledger across
// ReleaseLayers, the walk a migration takes. The free walks are separate
// copies, so a weight added to one and not the other leaks only on that path.
func TestLatentAttentionReturnsItsMemoryOnRelease(t *testing.T) {
	p := mlaPlan()
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()

	const blocks = 3
	ws := mlaBlocks(p, blocks)
	for li := 0; li < blocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("latent block %d declined: %s", li, g.Err())
		}
	}
	x := make([]float32, p.NEmbd)
	g.Layers(0, blocks, 0, 1, x, make([]float32, p.NRot), nil, nil)

	before := d.freed
	usedBefore := g.devs[0].used
	g.ReleaseLayers(0, blocks)
	if d.freed == before {
		t.Fatal("releasing every latent block freed no buffer at all")
	}
	// The budget must be refunded too; an unrefunded charge makes the device
	// slowly stop taking work (see addSessionKV).
	if used := g.devs[0].used; used >= usedBefore {
		t.Fatalf("the budget went %d -> %d over a release of %d latent block(s): "+
			"a charge is not being refunded", usedBefore, used, blocks)
	}
	t.Logf("%d latent blocks released: %d buffers freed, budget %d -> %d bytes",
		blocks, d.freed-before, usedBefore, g.devs[0].used)
}

// TestLatentAttentionReturnsItsMemoryOnTheLocalWalk covers dropLayerLocal, the
// walk a re-prepared block takes (`-gpu-grow`, seamtune). Close goes through
// ReleaseLayers, so the gates above stay green if the latent norms are missing
// here. It drives the real walk rather than freeing the buffers itself.
func TestLatentAttentionReturnsItsMemoryOnTheLocalWalk(t *testing.T) {
	p := mlaPlan()
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	dt := g.devs[0]

	ws := mlaBlocks(p, 1)
	if !g.PrepLayer(0, p, ws[0]) {
		t.Fatalf("latent block declined: %s", g.Err())
	}
	l := dt.layers[0]
	if l == nil || !l.mla || l.nQA == nil || l.nKVA == nil {
		t.Fatal("the block is not a latent one with both norms uploaded; this gate proves nothing")
	}
	// The per-layer buffers this walk owns: the norms. The weights live in
	// dt.res and are shared, and the latent history lives in the device's
	// page pool under the block's index, where the re-prepared layer finds it
	// again (addKVLayer keeps a layer it already has); neither is freed here,
	// hence a delta.
	freed0, bytes0 := d.freed, d.freedBytes
	dt.dropLayerLocal(l)
	// The count is exact: two block norms and two latent norms. A ">= 3"
	// would accept the violation.
	const perLayer = 4
	if n := d.freed - freed0; n != perLayer {
		t.Fatalf("the local walk freed %d buffer(s), want %d: the two block norms and the "+
			"two LATENT norms. A missing one is a buffer nothing frees on the walk a "+
			"re-prepared block takes", n, perLayer)
	}
	if dt.kvp == nil || dt.kvp.layers[0] == nil {
		t.Fatal("the local walk dropped the block's pages: the re-prepared layer would " +
			"start with no history")
	}
	if d.doubleFree != 0 {
		t.Fatalf("%d double free(s) on the local walk", d.doubleFree)
	}
	// A second pass is not a no-op: dropLayerLocal frees some buffers without
	// nilling them, so running it twice on one layer double-frees. That is
	// unreachable (no layer is handed to both walks), so idempotency is
	// not asserted.
	t.Logf("the local walk returned %d buffer(s) / %d bytes",
		d.freed-freed0, d.freedBytes-bytes0)
}
