package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/nn"
)

// raceAllocs grows the other device's history to its pool's limit at every
// allocation of fa's device, and counts the growths that took a page.
func raceAllocs(fa *fakeDev, db *devTier, kp *kvPool, l *kvLayerPool) *int {
	grew := new(int)
	fa.onAlloc = func() {
		if growBeside(db, kp, l) > 0 {
			*grew++
		}
	}
	return grew
}

// poolHead is a head of vocab rows over p's width, as forget_test.go's.
func poolHead(p *nn.LayerPlan, vocab int) *nn.Head {
	return headGraphs(&nn.Head{
		Vectors: []nn.RoleVector{{Role: jlm.RoleOutputNorm, V: make([]float32, p.NEmbd)}},
		W:       nn.Weight{T: quant.Q4_0, Data: make([]byte, vocab*(p.NEmbd/32)*18), Rows: vocab, K: p.NEmbd},
		Logits:  make([]float32, vocab),
	}, p, jlm.RoleOutput)
}

// TestAHeadHoldsItsRoomInItsPool: the output projection's admission holds
// the room it found, its logits included, before it uploads, so the other
// device of the pool growing at each of its allocations finds the room
// taken. Checked and charged after, both spent the same bytes.
func TestAHeadHoldsItsRoomInItsPool(t *testing.T) {
	p := pagePlan()
	g, pool, fa, _ := pooledPair(t, 1<<30, 1<<30, 1<<30)
	db := g.devs[1]
	if ok, err := g.PrepLayerOn("a", 0, p, pageBlocks(p, 1)[0]); !ok || err != nil {
		t.Fatalf("block 0 declined: %v %s", err, g.Err())
	}
	kpb, lb := historyOn(t, db)
	pool.SetLimit(pool.Used() + 4<<20)
	grew := raceAllocs(fa, db, kpb, lb)
	ok, err := g.PrepHeadOn("a", poolHead(p, 512))
	fa.onAlloc = nil
	if err != nil {
		t.Fatal(err)
	}
	if *grew == 0 {
		t.Fatalf("the other device grew nothing while the head was placed (%v: %s): the gate proves nothing", ok, g.Err())
	}
	poolHeld(t, g, pool)
	t.Logf("head placed %v beside %d growth(s) of the other device's history: %s", ok, *grew, g.Err())
}

// TestAScratchHoldsItsRoomInItsPool: a placement's scratch -- the decode
// scratch and the reserved widths it probes (reserveScratch) -- holds each
// buffer's room in the pool before it is made, so the other device growing
// at each allocation finds it taken. Measured at the window's close and
// charged then, the pool went past its limit (Stats.ScratchOver).
func TestAScratchHoldsItsRoomInItsPool(t *testing.T) {
	p := pagePlan()
	block, sc := blockAndScratch(t, p, noPromptReserve())
	g, pool, fa, _ := pooledPair(t, 1<<30, 1<<30, 1<<30)
	db := g.devs[1]
	kpb, lb := historyOn(t, db)
	pool.SetLimit(pool.Used() + 4*(sc+block))
	grew := raceAllocs(fa, db, kpb, lb)
	ok, err := g.PrepLayerOn("a", 0, p, pageBlocks(p, 1)[0])
	fa.onAlloc = nil
	if err != nil {
		t.Fatal(err)
	}
	if *grew == 0 {
		t.Fatal("the other device grew nothing while the scratch was built: the gate proves nothing")
	}
	poolHeld(t, g, pool)
	t.Logf("block placed %v beside %d growth(s) of the other device's history: %s", ok, *grew, g.Err())
}

// TestMarksAndRowTablesHoldTheirRoom: the rotary marks' arena (growMarks)
// and a row table's rows (rowSlots) hold their room in the pool before they
// are made. Allocated and charged after, the other device grew into the
// bytes between (Stats.PoolOver).
func TestMarksAndRowTablesHoldTheirRoom(t *testing.T) {
	for _, tc := range []struct {
		name string
		grow func(d *devTier) error
	}{
		{"marks", func(d *devTier) error { return d.growMarks(1 << 14) }},
		{"rows", func(d *devTier) error {
			var r rowRange
			if err := d.rowSlots(&r, 64, 1<<10); err != nil {
				return err
			}
			d.freeRowSlots(&r, 1<<10)
			return nil
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g, pool, fa, _ := pooledPair(t, 1<<30, 1<<30, 1<<30)
			da, db := g.devs[0], g.devs[1]
			kpb, lb := historyOn(t, db)
			pool.SetLimit(pool.Used() + 1<<20)
			grew := raceAllocs(fa, db, kpb, lb)
			da.mu.Lock()
			err := tc.grow(da)
			da.mu.Unlock()
			fa.onAlloc = nil
			if *grew == 0 {
				t.Fatal("the other device grew nothing while this one allocated: the gate proves nothing")
			}
			if err != nil {
				t.Fatalf("a growth that held its room lost it to the other device: %v", err)
			}
			poolHeld(t, g, pool)
		})
	}
}

// layersOn runs one decode token over blocks [lo, hi) at pos.
func layersOn(t *testing.T, g *GPU, p *nn.LayerPlan, lo, hi, pos int) bool {
	t.Helper()
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	return g.Layers(lo, hi, pos, 1, x, nn.RopeTables{cs}, nil)
}

// TestAnAdmissionPagesOutTheOtherDevicesBlock: a block placed resident on a
// device whose pool the other device's streamed blocks fill pages out the
// idle ones (fromPeers), never the other's last, rather than declining to
// the host; the other device pages them back through its slot.
func TestAnAdmissionPagesOutTheOtherDevicesBlock(t *testing.T) {
	const n = 4
	p := pagePlan()
	block, _ := blockAndScratch(t, p, noPromptReserve())
	card := cardFor(t, p, 3, noPromptReserve())
	g, pool, _, _ := pooledPair(t, card, card, 1<<40, noPromptReserve())
	da, db := g.devs[0], g.devs[1]
	streamBlocks(g, n)
	ws := pageBlocks(p, n+1)
	for li := 0; li < n; li++ {
		if ok, err := g.PrepLayerOn("b", li, p, ws[li]); !ok || err != nil {
			t.Fatalf("block %d declined: %v %s", li, err, g.Err())
		}
	}
	if !token(t, g, p, n, 0) {
		t.Fatalf("token 0: %s", g.Err())
	}
	if in := inBlocks(db); in < 2 {
		t.Fatalf("%d streamed block(s) in after a token on three slots: the gate proves nothing", in)
	}
	pool.SetLimit(pool.Used() + block/2)
	ok, err := g.PrepLayerOn("a", n, p, ws[n])
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("a block placed resident declined beside the other device's idle streamed blocks: %s", g.Err())
	}
	if da.stats().PeerPageOuts == 0 {
		t.Fatal("the block was placed and no block of the other device went out: the gate proves nothing")
	}
	if inBlocks(db) == 0 {
		t.Fatal("the other device's last block went out: its slot is gone")
	}
	if !token(t, g, p, n+1, 1) {
		t.Fatalf("a token over both devices' blocks: %s", g.Err())
	}
	poolHeld(t, g, pool)
	t.Logf("%d block(s) of the other device paged out for the admission", da.stats().PeerPageOuts)
}

// TestAPageInPagesOutTheOtherDevicesBlock: a device paging a block in whose
// pool the other device's streamed blocks fill, with none of its own to
// page out and no slot kept (a budget lowered under it gives the slot back,
// trim), pages out the other's idle ones (fromPeers) rather than failing
// the token.
func TestAPageInPagesOutTheOtherDevicesBlock(t *testing.T) {
	const n = 4
	p := pagePlan()
	block, _ := blockAndScratch(t, p, noPromptReserve())
	card := cardFor(t, p, 3, noPromptReserve())
	g, pool, _, _ := pooledPair(t, card, card, 1<<40, noPromptReserve())
	da, db := g.devs[0], g.devs[1]
	streamBlocks(g, 2*n)
	ws := pageBlocks(p, 2*n)
	for li := 0; li < 2*n; li++ {
		name := "b"
		if li >= n {
			name = "a"
		}
		if ok, err := g.PrepLayerOn(name, li, p, ws[li]); !ok || err != nil {
			t.Fatalf("block %d declined on %s: %v %s", li, name, err, g.Err())
		}
	}
	if !token(t, g, p, 2*n, 0) {
		t.Fatalf("token 0: %s", g.Err())
	}
	if in := inBlocks(db); in < 2 {
		t.Fatalf("%d streamed block(s) of the other device in: the gate proves nothing", in)
	}
	da.mu.Lock()
	for li := range da.stream {
		if l := da.layers[li]; l != nil && l.pg != nil && l.pg.in {
			da.pageOut(li)
		}
	}
	da.dropSwap()
	da.mu.Unlock()
	if inBlocks(da) != 0 {
		t.Fatal("a streamed block is still in: the gate proves nothing")
	}
	pool.SetLimit(pool.Used() + block/2)
	outs := da.stats().PeerPageOuts
	if !layersOn(t, g, p, n, 2*n, 1) {
		t.Fatalf("a page-in with the other device's idle blocks filling the pool failed: %s", g.Err())
	}
	if da.stats().PeerPageOuts == outs {
		t.Fatal("the page-ins ran and no block of the other device went out: the gate proves nothing")
	}
	if inBlocks(db) == 0 {
		t.Fatal("the other device's last block went out: its slot is gone")
	}
	if !layersOn(t, g, p, 0, n, 1) {
		t.Fatalf("the device whose blocks went out does not run its token: %s", g.Err())
	}
	poolHeld(t, g, pool)
	t.Logf("%d block(s) of the other device paged out for the page-ins", da.stats().PeerPageOuts-outs)
}
