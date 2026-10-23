package tier

import (
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// TestAnUnknownHostTakesTheUnifiedDevicesOwnFigure: on darwin the host budget
// is unknown (sched.MemBudget is 0), and a unified device's share fell back to
// a 2 GiB default -- which, once the prompt's scratch was charged, no longer
// held a small model's blocks and head. The device's own figure for
// what the GPU may hold stands in for the host's, less the headroom a known
// host keeps (hostShare).
func TestAnUnknownHostTakesTheUnifiedDevicesOwnFigure(t *testing.T) {
	igpu := &fakeDev{unified: true, total: 16 << 30, free: 12 << 30}
	slots := planSlots([]backend.Device{igpu}, nil, openOpts{host: 0, hostSet: true})
	if got := slots[0].Pool.Limit(); got != 15<<30 {
		t.Fatalf("an unknown host with a 16 GiB unified device: pool %d, want 16 GiB less 1 GiB of headroom", got)
	}
	// A device that says nothing leaves the default.
	mute := &fakeDev{unified: true}
	if got := planSlots([]backend.Device{mute}, nil, openOpts{host: 0, hostSet: true})[0].Pool.Limit(); got != defaultBudget {
		t.Fatalf("an unknown host and a device that reports nothing: pool %d, want the %d default", got, uint64(defaultBudget))
	}
}

// fullCard places blocks on a fake card of capBytes until it refuses one, the
// card's budget being the card -- what a model that does not fit does -- and
// then gives blocks back from the end until the whole context's history fits,
// as the model's placement does for its pages. It returns the tier and how
// many blocks it holds.
func fullCard(t *testing.T, p *nn.LayerPlan, capBytes int, cfg func(*Config)) (*GPU, *fakeDev, int) {
	t.Helper()
	d := &fakeDev{capBytes: capBytes}
	g, err := New([]Slot{{Dev: d, Bytes: uint64(capBytes)}}, WithDeviceTune(TuneOff), WithConfig(cfg))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	n := 0
	for ; n < 4096 && g.PrepLayer(n, p, blockWeights(p)); n++ {
	}
	for n > 0 && !g.ReserveKV(p.MaxSeq) {
		n--
		g.ReleaseLayers(n, n+1)
	}
	if n == 0 {
		t.Fatalf("the card holds no block with its history: %s", g.Err())
	}
	return g, d, n
}

// scratchAndBlock sizes the reservation, and one block with the whole
// context's history, on an unbounded card.
func scratchAndBlock(t *testing.T, p *nn.LayerPlan, cfg func(*Config)) (scratch, block uint64) {
	t.Helper()
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff), WithConfig(cfg))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	if !g.PrepLayer(0, p, blockWeights(p)) || !g.ReserveKV(p.MaxSeq) {
		t.Fatalf("sizing: %s", g.Err())
	}
	d := g.devs[0]
	return d.scratch, d.used - d.scratch
}

// TestAPromptFindsItsScratchOnAFullCard is the gate on the reservation: a card
// just big enough for the device's scratch and a few blocks, offered more
// blocks than it holds, still runs a full-width prompt chunk as one batched
// submission. Without the reservation the blocks take the whole card and the
// chunk's scratch is refused at the prompt, so the chunk is split -- the
// small card that answered cuMemAlloc with out of memory, so the prompt went a
// row at a time.
func TestAPromptFindsItsScratchOnAFullCard(t *testing.T) {
	p := fakePlan()
	on := func(c *Config) {}
	scratch, block := scratchAndBlock(t, p, on)
	if scratch == 0 {
		t.Fatal("the reservation charged nothing; this gate proves nothing")
	}
	capBytes := int(scratch + 3*block + block/2)
	run := func(reserve bool) (Stats, int, bool) {
		g, _, n := fullCard(t, p, capBytes, func(c *Config) { c.NoScratchReserve = !reserve })
		x := make([]float32, batchWidth*p.NEmbd)
		cs := make([]float32, batchWidth*p.NRot)
		ok := g.Layers(0, n, 0, batchWidth, x, cs, nil, nil)
		return g.Stats(), n, ok
	}
	st, n, ok := run(true)
	t.Logf("reserved: %d blocks, scratch %d bytes for %d rows, chunk ok %v, %d split(s): %s",
		n, st.ScratchBytes, st.ReservedPrompt, ok, st.PromptSplits, st.LastErr)
	if !ok || st.PromptSplits != 0 || st.ReservedPrompt != batchWidth {
		t.Fatalf("the reserved card split or failed the chunk: ok %v, %d split(s), %d rows reserved: %s",
			ok, st.PromptSplits, st.ReservedPrompt, st.LastErr)
	}
	// The violation: no reservation, the same card.
	vst, vn, vok := run(false)
	t.Logf("unreserved: %d blocks, chunk ok %v, %d split(s): %s", vn, vok, vst.PromptSplits, vst.LastErr)
	if vok && vst.PromptSplits == 0 {
		t.Fatalf("without the reservation the chunk still ran whole on %d blocks: the gate does not discriminate", vn)
	}
	if vn <= n {
		t.Fatalf("without the reservation the card took %d blocks against %d: it was not full", vn, n)
	}
}

// TestAStepAtMaxRowsFitsAFullCard is the server's step loop on a card the
// blocks filled: a ragged step of the full width, every row wanting the head,
// needs the step's scratch and its per-row logits (rows x vocabulary). With
// StepRows set the device reserves both at placement; without it the logits
// are asked for at the first step, from a card with nothing left.
func TestAStepAtMaxRowsFitsAFullCard(t *testing.T) {
	const vocab = 4096
	p := fakePlan()
	p.Vocab = vocab
	step := func(c *Config) { c.StepRows = batchWidth }
	scratch, block := scratchAndBlock(t, p, step)
	head := uint64(vocab*(p.NEmbd/32)*18 + 2*vocab*4 + p.NEmbd*4)
	capBytes := int(scratch + 3*block + head + block/2)
	run := func(reserve bool) (Stats, int, bool) {
		cfg := func(c *Config) {}
		if reserve {
			cfg = step
		}
		g, _, n := fullCard(t, p, capBytes, cfg)
		// The head goes where the last block is, and on a full card it needs
		// the room of the blocks after the seam: the model's placeHead.
		h := &nn.Head{
			Norm:   make([]float32, p.NEmbd),
			W:      nn.Weight{T: quant.Q4_0, Data: make([]byte, vocab*(p.NEmbd/32)*18), Rows: vocab, K: p.NEmbd},
			Logits: make([]float32, vocab),
		}
		for !g.PrepHead(h) {
			if n == 1 {
				t.Fatalf("the head does not fit beside one block: %s", g.Err())
			}
			n--
			g.ReleaseLayers(n, n+1)
		}
		pos := make([]int, batchWidth)
		for i := range pos {
			pos[i] = i
		}
		x := make([]float32, batchWidth*p.NEmbd)
		cs := make([]float32, batchWidth*p.NRot)
		ok := g.LayersRows(0, n, pos, pos, p.MaxSeq, x, cs, nil, h)
		return g.Stats(), n, ok
	}
	st, n, ok := run(true)
	t.Logf("reserved: %d blocks, scratch %d bytes, step %d rows, ok %v: %s",
		n, st.ScratchBytes, st.ReservedStep, ok, st.LastErr)
	if !ok || st.ReservedStep != batchWidth {
		t.Fatalf("a %d-row step on the reserved card: ok %v, %d rows reserved: %s",
			batchWidth, ok, st.ReservedStep, st.LastErr)
	}
	vst, vn, vok := run(false)
	t.Logf("unreserved: %d blocks, ok %v: %s", vn, vok, vst.LastErr)
	if vok {
		t.Fatalf("without the step's reservation a %d-row step still ran on %d blocks: the gate does not discriminate",
			batchWidth, vn)
	}
}

// TestTheScratchIsChargedAndReturned is the ledger for the scratch charge:
// what the reservation and a prompt's lazily grown buffers charged comes back
// when the last block leaves, and every buffer comes back at Close.
func TestTheScratchIsChargedAndReturned(t *testing.T) {
	p := fakePlan()
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for li := 0; li < 3; li++ {
		if !g.PrepLayer(li, p, blockWeights(p)) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	dt := g.devs[0]
	// A chunk of another width than the reserved one grows a second scratch.
	x := make([]float32, batchWidth*p.NEmbd)
	cs := make([]float32, batchWidth*p.NRot)
	if !g.Layers(0, 3, 0, 64, x[:64*p.NEmbd], cs[:64*p.NRot], nil, nil) {
		t.Fatalf("a 64-row chunk: %s", g.Err())
	}
	if dt.scratch == 0 || len(dt.bbs) < 2 {
		t.Fatalf("scratch %d bytes over %d batched widths: the charge is not running", dt.scratch, len(dt.bbs))
	}
	live := uint64(d.allocBytes - d.freedBytes)
	if dt.scratch > live {
		t.Fatalf("the scratch charges %d bytes and the card holds %d", dt.scratch, live)
	}
	g.ReleaseLayers(0, 3)
	if dt.used != 0 || dt.scratch != 0 {
		t.Fatalf("after releasing every block, used %d and scratch %d: the scratch was not refunded", dt.used, dt.scratch)
	}
	if live := d.allocBytes - d.freedBytes; live > 4096 {
		t.Fatalf("a card holding nothing still holds %d bytes", live)
	}
	g.Close()
	if d.doubleFree != 0 || d.freedBytes != d.allocBytes {
		t.Fatalf("Close: %d of %d bytes returned, %d double frees", d.freedBytes, d.allocBytes, d.doubleFree)
	}
}

// TestAKVGrowTheCardRefusesKeepsTheHistory: a layer that grows through the
// host (compactViaHost) frees its buffers before asking for the bigger pair,
// and on a card fuller than its budget says the ask fails. The layer then
// takes its old size back with the history in it, rather than staying
// bufferless for the next grow to read a nil buffer -- which is what filling a
// card a block at a time and then reserving the context's pages found.
func TestAKVGrowTheCardRefusesKeepsTheHistory(t *testing.T) {
	p := fakePlan()
	scratch, block := scratchAndBlock(t, p, func(c *Config) {})
	g, d, n := fullCard(t, p, int(scratch+3*block+block/2), func(c *Config) { c.NoScratchReserve = true })
	dt := g.devs[0]
	if dt.kvp == nil || len(dt.kvp.layers) == 0 {
		t.Fatal("no paged history on the card; this gate proves nothing")
	}
	for li, l := range dt.kvp.layers {
		if l.n > 0 && l.k == nil {
			t.Fatalf("block %d's history holds %d pages and no buffer: a refused grow lost it", li, l.n)
		}
	}
	t.Logf("%d blocks on a %d-byte card; every layer's history has its buffers", n, d.capBytes)
}

// TestTheDriversRoundingIsCharged: a driver hands memory out in pages, so a
// buffer costs the card its size rounded up -- 252 MB over a 2.18 GB model
// on CUDA, which the budget did not see and which a prompt's history then
// could not grow into. On a fake that rounds to 64 KiB, what is charged is
// what the card spends, block, scratch and history, and all of it comes back.
func TestTheDriversRoundingIsCharged(t *testing.T) {
	p := fakePlan()
	d := &fakeDev{page: 64 << 10}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	for li := 0; li < 3; li++ {
		if !g.PrepLayer(li, p, blockWeights(p)) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	if !g.ReserveKV(p.MaxSeq) {
		t.Fatalf("ReserveKV: %s", g.Err())
	}
	dt := g.devs[0]
	if d.Footprint() <= d.Allocated() {
		t.Fatalf("the fake rounded nothing (%d against %d): the gate proves nothing", d.Footprint(), d.Allocated())
	}
	// Every byte of rounding is charged, and what is left uncharged is under a
	// page: a few tiny buffers (an argmax word, a table) were never charged.
	if dt.rounding != d.Footprint()-d.Allocated() || dt.used+64<<10 < d.Footprint() {
		t.Fatalf("%d bytes charged and the card spends %d (%d asked for, %d of rounding charged)",
			dt.used, d.Footprint(), d.Allocated(), dt.rounding)
	}
	g.ReleaseLayers(0, 3)
	if dt.used != dt.rounding || dt.rounding != d.Footprint()-d.Allocated() {
		t.Fatalf("after releasing every block: %d charged, %d of it rounding, the card %d over %d asked",
			dt.used, dt.rounding, d.Footprint(), d.Allocated())
	}
}

// TestARefusedGrowKeepsTheOldBuffer: the shared staging grows by freeing the
// old buffer first, and a card that refuses the bigger one used to leave it
// nil -- under every launch already built against it, which the narrower
// fallback of a refused chunk then ran (a Vulkan panic on a full card, the
// prompt gate's violation arm). The old size comes back instead.
func TestARefusedGrowKeepsTheOldBuffer(t *testing.T) {
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	dt := g.devs[0]
	dt.mu.Lock()
	defer dt.mu.Unlock()
	if !dt.sizePart(64) {
		t.Fatal("a 64-float partial buffer was refused on an empty card")
	}
	d.capBytes = d.allocBytes - d.freedBytes + 64
	if dt.sizePart(1 << 20) {
		t.Fatal("a card with 64 bytes left took a 4 MiB partial buffer: the gate proves nothing")
	}
	if dt.partBuf == nil || dt.partCap != 64 {
		t.Fatalf("a refused grow left the partial buffer %v at %d floats, want the old 64", dt.partBuf, dt.partCap)
	}
}
