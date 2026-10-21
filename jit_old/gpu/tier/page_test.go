package tier

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/nn"
)

// Block paging, at the tier's level: a device that cannot hold a model must
// run all of it, and a device that can must never move a byte.
//
// "Every block placed" and "the tokens were right" are also true of a tier
// that paged nothing, so the does-not-fit tests read Stats.PageIns and the
// fits tests read Stats.PageOuts, and each is run against a violation.

// pagePlan is a block whose weights dominate its KV cache, which is the shape
// paging is for. MaxSeq is 64 on purpose: the KV cache is permanent and comes
// off the budget before slots are counted, and a larger cache leaves a small
// card under the slot floor, where it correctly declines.
func pagePlan() *nn.LayerPlan {
	return &nn.LayerPlan{
		NEmbd: 256, NHead: 4, NKVHead: 2, HeadDim: 64, NRot: 64, NFFN: 1024,
		MaxSeq: 64, RMSEps: 1e-5, RopeBase: 10000,
	}
}

// pageBlocks builds n blocks with distinct weights (resident() keys on the
// address).
func pageBlocks(p *nn.LayerPlan, n int) []*nn.LayerWeights {
	out := make([]*nn.LayerWeights, n)
	q4 := func(rows, k int) nn.Weight {
		return nn.Weight{T: quant.Q4_0, Data: make([]byte, rows*(k/32)*18), Rows: rows, K: k}
	}
	kv := p.NKVHead * p.HeadDim
	for i := range out {
		out[i] = &nn.LayerWeights{
			AttnNorm: make([]float32, p.NEmbd), FFNNorm: make([]float32, p.NEmbd),
			Wq:   q4(p.NHead*p.HeadDim, p.NEmbd),
			Wk:   q4(kv, p.NEmbd),
			Wv:   q4(kv, p.NEmbd),
			Wo:   q4(p.NEmbd, p.NHead*p.HeadDim),
			Gate: q4(p.NFFN, p.NEmbd),
			Up:   q4(p.NFFN, p.NEmbd),
			Down: q4(p.NEmbd, p.NFFN),
		}
	}
	return out
}

// cardFor is a budget for fit blocks of p: their cost and the device's own
// scratch, which it pays once beside them (scratch.go).
func cardFor(t *testing.T, p *nn.LayerPlan, fit int) uint64 {
	b, sc := blockAndScratch(t, p)
	return sc + b*uint64(fit)
}

// blockAndScratch measures one block's cost -- weights, KV cache and norms --
// and the device's scratch.
func blockAndScratch(t *testing.T, p *nn.LayerPlan) (block, scratch uint64) {
	t.Helper()
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	if !g.PrepLayer(0, p, pageBlocks(p, 1)[0]) {
		t.Fatalf("sizing run declined: %s", g.Err())
	}
	// And the page a first token's history takes: a block that cannot run a
	// token is not what these budgets are sized for.
	if !g.ReserveKV(1) {
		t.Fatalf("sizing run: no room for a first page: %s", g.Err())
	}
	d := g.devs[0]
	return d.used - d.scratch, d.scratch
}

// streamBlocks places blocks [0, n) to stream through g's devices (GPU.Stream):
// what a streaming placement asks before the blocks are offered.
func streamBlocks(g *GPU, n int) {
	for li := 0; li < n; li++ {
		g.Stream(li, true)
	}
}

// pageTier offers n blocks to one device whose budget holds fit of them.
func pageTier(t *testing.T, n, fit int, paging bool) (*GPU, *nn.LayerPlan) {
	t.Helper()
	p := pagePlan()
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: cardFor(t, p, fit)}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	if paging {
		streamBlocks(g, n)
	}
	ws := pageBlocks(p, n)
	for li := 0; li < n; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			break // offerRange stops at the first refusal; so does this
		}
	}
	return g, p
}

// token runs one decode token over blocks [0, n).
func token(t *testing.T, g *GPU, p *nn.LayerPlan, n, pos int) bool {
	t.Helper()
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	for i := range x {
		x[i] = float32(i%7) * 0.01
	}
	return g.Layers(0, n, pos, 1, x, cs, nil, nil)
}

func placedTotal(g *GPU) int {
	n := 0
	for _, v := range g.Placed() {
		n += v
	}
	return n
}

// TestAModelTooBigForTheCardRunsEveryBlockOnIt is the gate the feature exists
// for: with paging, every block of an oversized model is placed and run. It
// asserts page-ins, not placement, because a tier that accepted every block
// and kept only some resident would place them all too.
func TestAModelTooBigForTheCardRunsEveryBlockOnIt(t *testing.T) {
	const n, fit = 10, 4
	g, p := pageTier(t, n, fit, true)

	if got := placedTotal(g); got != n {
		t.Fatalf("%d of %d blocks placed under a %d-block budget: %s", got, n, fit, g.Err())
	}
	for tok := 0; tok < 3; tok++ {
		if !token(t, g, p, n, tok) {
			t.Fatalf("token %d: %s", tok, g.Err())
		}
	}
	st := g.Stats()
	if st.PageIns == 0 {
		t.Fatalf("10 blocks placed on a 4-block budget and ZERO page-ins: nothing paged, "+
			"so this test proved nothing (slots %d, page-outs %d)", st.Slots, st.PageOuts)
	}
	if st.PageOuts == 0 {
		t.Fatalf("%d page-ins and no page-outs: the card grew instead of swapping", st.PageIns)
	}
	if st.Blocks != n*3 {
		t.Fatalf("%d blocks RUN over 3 tokens, want %d: placement is not execution", st.Blocks, n*3)
	}
	if st.Slots < minSlots {
		t.Fatalf("%d slots, below the %d a swap needs, yet it paged", st.Slots, minSlots)
	}
	// The policy must beat LRU, which on a cyclic scan evicts every block just
	// before it is needed (page-ins = blocks*tokens).
	if worst := n * 3; st.PageIns >= worst {
		t.Fatalf("%d page-ins over 3 tokens of %d blocks: that is every block every token, "+
			"which is what LRU gives on a cyclic scan -- the victim policy is not working",
			st.PageIns, n)
	}

	// Violation: the same model and budget with paging off.
	t.Run("violation/pagingOff", func(t *testing.T) {
		g, p := pageTier(t, n, fit, false)
		got := placedTotal(g)
		if !token(t, g, p, got, 0) && got > 0 {
			t.Fatalf("token: %s", g.Err())
		}
		st := g.Stats()
		msg := fmt.Sprintf("paging off: %d of %d blocks placed, %d page-ins, %d page-outs; last decline: %s",
			got, n, st.PageIns, st.PageOuts, g.Err())
		if got == n || st.PageIns > 0 {
			t.Fatalf("THE GATE IS NOT A GATE -- %s", msg)
		}
		t.Logf("violation seen to fail as required -- %s", msg)
	})
}

// TestAModelThatFitsNeverPages: a model that fits pays for tracking and
// nothing else, so eviction never fires and each token is one submission. A
// pager that split the range per block would page nothing and still cost a
// crossing per block.
func TestAModelThatFitsNeverPages(t *testing.T) {
	const n = 6
	g, p := pageTier(t, n, n+1, true)
	if got := placedTotal(g); got != n {
		t.Fatalf("%d of %d blocks placed under a budget sized for %d: %s", got, n, n+1, g.Err())
	}
	before := g.Stats()
	for tok := 0; tok < 4; tok++ {
		if !token(t, g, p, n, tok) {
			t.Fatalf("token %d: %s", tok, g.Err())
		}
	}
	st := g.Stats()
	if st.PageOuts != 0 || st.PageIns != 0 {
		t.Fatalf("a model that FITS paged: %d in, %d out (slots %d)", st.PageIns, st.PageOuts, st.Slots)
	}
	if got := st.Submits - before.Submits; got != 4 {
		t.Fatalf("%d submissions for 4 tokens of %d resident blocks, want 4: the pager cut a "+
			"range it did not have to, which is a crossing per block for nothing", got, n)
	}
	if st.Slots < n {
		t.Fatalf("%d slots for %d resident blocks: the slot arithmetic disagrees with the card", st.Slots, n)
	}

	// Violation: the same model on a budget it does not fit.
	t.Run("violation/doesNotFit", func(t *testing.T) {
		g, p := pageTier(t, n, 3, true)
		for tok := 0; tok < 2; tok++ {
			if !token(t, g, p, placedTotal(g), tok) {
				t.Fatalf("token %d: %s", tok, g.Err())
			}
		}
		st := g.Stats()
		msg := fmt.Sprintf("a 3-block budget for %d blocks: %d page-ins, %d page-outs, %d submissions",
			n, st.PageIns, st.PageOuts, st.Submits)
		if st.PageIns == 0 || st.PageOuts == 0 {
			t.Fatalf("THE GATE IS NOT A GATE -- %s", msg)
		}
		t.Logf("violation seen to fail as required -- %s", msg)
	})
}

// TestShrinkingTheBudgetDemotesBlocksWithoutMovingThem is the second-model
// case: on a shrink the bytes must really go back, and the placement must not
// move, since a block that changed tier can change a token id.
func TestShrinkingTheBudgetDemotesBlocksWithoutMovingThem(t *testing.T) {
	const n = 8
	g, p := pageTier(t, n, n+1, true)
	if got := placedTotal(g); got != n {
		t.Fatalf("%d of %d blocks placed: %s", got, n, g.Err())
	}
	if !token(t, g, p, n, 0) {
		t.Fatalf("token before the shrink: %s", g.Err())
	}
	if st := g.Stats(); st.PageOuts != 0 {
		t.Fatalf("it evicted %d times before the budget moved", st.PageOuts)
	}
	was := placement(g, n)
	bytesWas := g.Bytes()

	// Hand half the card back, which is what loading a second model does.
	half := cardFor(t, p, n/2)
	left, err := g.SetBudget(half)
	if err != nil {
		t.Fatal(err)
	}

	st := g.Stats()
	if st.PageOuts == 0 {
		t.Fatalf("the budget halved to %d bytes and nothing demoted: %d blocks still resident, "+
			"%d bytes held -- a pool that cannot shrink is not a pool", half, left, g.Bytes())
	}
	if g.Bytes() >= bytesWas {
		t.Fatalf("holding %d bytes after a shrink from %d under a %d budget", g.Bytes(), bytesWas, half)
	}
	if got := placement(g, n); !sameInts(got, was) {
		t.Fatalf("the placement MOVED across a shrink: %v -> %v. A demotion changes where a "+
			"block's weights are, never which device computes it", was, got)
	}
	// It still runs: the demoted blocks page back in on demand.
	for tok := 1; tok < 3; tok++ {
		if !token(t, g, p, n, tok) {
			t.Fatalf("token %d after the shrink: %s", tok, g.Err())
		}
	}
	if st := g.Stats(); st.PageIns == 0 {
		t.Fatalf("nothing paged back in after the shrink: the demoted blocks are not running")
	}
	if got := g.Stats().Blocks; got != n*3 {
		t.Fatalf("%d blocks run over 3 tokens, want %d: a demoted block stopped computing", got, n*3)
	}

	// Violation: shrink below the slot floor; the device must refuse.
	t.Run("violation/shrinkToNothing", func(t *testing.T) {
		g.SetBudget(1)
		msg := fmt.Sprintf("budget 1 byte: %d bytes held, %d slots", g.Bytes(), g.Stats().Slots)
		if token(t, g, p, n, 4) {
			t.Fatalf("THE GATE IS NOT A GATE -- a card handed back still ran blocks (%s)", msg)
		}
		t.Logf("violation seen to fail as required -- %s; Layers refused: %s", msg, g.Err())
	})
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestPagedBlockGetsTheSameBytesBack: a paged-in block must compute from the
// same bytes, which is what a demotion's token-id claim rests on. recDev keeps
// what was written to it, so the comparison is exact.
func TestPagedBlockGetsTheSameBytesBack(t *testing.T) {
	p := pagePlan()
	d := &recDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	streamBlocks(g, 1)
	w := pageBlocks(p, 1)[0]
	if !g.PrepLayer(0, p, w) {
		t.Fatalf("PrepLayer: %s", g.Err())
	}
	one := g.devs[0]
	snap := func() [][]byte {
		var out [][]byte
		// tensors() carries the shared expert's three optional slots, nil on
		// every model without one; an absent slot is not a failed upload. The
		// count below keeps the snapshot from silently covering less.
		present := 0
		for _, rp := range one.layers[0].tensors() {
			r := *rp
			if r == nil {
				continue
			}
			present++
			if !r.ok {
				t.Fatal("a tensor is not resident")
			}
			for _, b := range []*recBuf{r.qs.(*recBuf), r.d.(*recBuf), r.sc.(*recBuf)} {
				out = append(out, append([]byte(nil), b.data...))
			}
		}
		if present != 7 {
			t.Fatalf("%d resident tensors, want 7", present)
		}
		return out
	}
	first := snap()
	firstIDs := bufIDs(one.layers[0])

	one.mu.Lock()
	one.pageOut(0)
	one.mu.Unlock()
	if one.layers[0].pg.in {
		t.Fatal("pageOut left the page marked resident")
	}
	if !one.pageIn(0, 0, 1) {
		t.Fatalf("pageIn: %s", one.LastErr)
	}
	// Different buffers, or nothing paged: byte equality against the same
	// allocation proves nothing.
	if ids := bufIDs(one.layers[0]); sameInts(ids, firstIDs) {
		t.Fatalf("the page-in landed in the SAME device buffers (%v): nothing was freed", ids)
	}
	second := snap()
	for i := range first {
		if len(first[i]) != len(second[i]) {
			t.Fatalf("array %d: %d bytes then %d", i, len(first[i]), len(second[i]))
		}
		for j := range first[i] {
			if first[i][j] != second[i][j] {
				t.Fatalf("array %d byte %d: %#x then %#x -- a paged block computes from "+
					"different bytes, so its tokens can move", i, j, first[i][j], second[i][j])
			}
		}
	}

	// Violation: doctor one byte and the comparison must fire.
	t.Run("violation/oneByte", func(t *testing.T) {
		b := (*one.layers[0].tensors()[0]).qs.(*recBuf)
		if len(b.data) == 0 {
			t.Fatal("no bytes to doctor")
		}
		was := b.data[0]
		b.data[0] ^= 0xFF
		defer func() { b.data[0] = was }()
		if b.data[0] == first[0][0] {
			t.Fatalf("THE GATE IS NOT A GATE -- a doctored byte still compares equal")
		}
		t.Logf("violation seen to fail as required -- byte 0 is %#x, the first upload had %#x",
			b.data[0], first[0][0])
	})
}

func bufIDs(l *layer) []int {
	var out []int
	for _, rp := range l.tensors() {
		// Optional slots are nil on models without them; see snap().
		r := *rp
		if r == nil {
			continue
		}
		out = append(out, r.qs.(*recBuf).id, r.d.(*recBuf).id, r.sc.(*recBuf).id)
	}
	return out
}

// TestOneSlotStillPages: a swap is evict, read in, run, and one slot does all
// three. Declining below two slots would send the whole model to the host
// (Qwen3-Next-80B's page fits a 2 GiB card once, not twice). One slot costs
// overlap, a rate, not admission.
func TestOneSlotStillPages(t *testing.T) {
	const n = 4
	// fit=2, not 1: slots() subtracts the scratch and KV cache first, so one
	// block's worth buys zero slots. Two leaves exactly one, asserted below.
	g, p := pageTier(t, n, 2, true)
	if got := placedTotal(g); got != n {
		t.Fatalf("%d of %d blocks placed on a ONE-slot budget: it declined "+
			"instead of paging: %s", got, n, g.Err())
	}
	for tok := 0; tok < 2; tok++ {
		if !token(t, g, p, n, tok) {
			t.Fatalf("token %d: %s", tok, g.Err())
		}
	}
	st := g.Stats()
	if st.PageIns == 0 {
		t.Fatalf("a one-slot budget over %d blocks and ZERO page-ins: it declined "+
			"instead of paging (slots %d): %s", n, st.Slots, g.Err())
	}
	if st.PageOuts == 0 {
		t.Fatalf("%d page-ins and no page-outs on ONE slot: every page-in must "+
			"evict the block before it", st.PageIns)
	}
	if st.Blocks != n*2 {
		t.Fatalf("%d blocks RUN over 2 tokens, want %d: a one-slot pager must "+
			"still run every block", st.Blocks, n*2)
	}
	if st.Slots != 1 {
		t.Fatalf("%d slots, want 1: this gate is about the floor and is not "+
			"measuring it", st.Slots)
	}
	t.Logf("one slot over %d blocks: %d page-in(s), %d page-out(s), %d block-run(s)",
		n, st.PageIns, st.PageOuts, st.Blocks)
}

// TestASlotIsCountedAtTheFootprint: a page-in finds room for a block's
// buffers as the driver rounds them (pageIn), so the slot count has to divide
// by the same figure. Counted at the bytes asked for, a block of small tensors
// on a card that hands out 64 KiB pages reads as many slots when it has none,
// and a caller that sizes a budget by Stats.Slots -- as
// model.TestDevicePagingSurvivesTheHostPager does -- gets a budget on which
// every page-in is refused and the blocks run on the host. Every budget this
// scans that reports a slot must run a token's every block on the device.
func TestASlotIsCountedAtTheFootprint(t *testing.T) {
	const n = 4
	p := pagePlan()
	d := &fakeDev{page: 64 << 10}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	streamBlocks(g, n)
	ws := pageBlocks(p, n)
	for li := 0; li < n; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	if !token(t, g, p, n, 0) {
		t.Fatalf("token on the open card: %s", g.Err())
	}
	if d.Footprint() <= d.Allocated() {
		t.Fatalf("the fake rounded nothing (%d against %d): the gate proves nothing", d.Footprint(), d.Allocated())
	}
	full := g.Bytes()
	pos, scanned := 1, 0
	for i := 1; i <= 64; i++ {
		b := full * uint64(65-i) / 65
		if _, err := g.SetBudget(b); err != nil {
			t.Fatal(err)
		}
		st := g.Stats()
		if st.Slots < minSlots {
			continue
		}
		scanned++
		if !token(t, g, p, n, pos) {
			t.Fatalf("a budget of %d bytes reports %d slot(s) and the token was refused: %s", b, st.Slots, g.Err())
		}
		pos++
		if got := g.Stats().Blocks - st.Blocks; got != n {
			t.Fatalf("a budget of %d bytes reports %d slot(s) and ran %d of %d blocks: %s",
				b, st.Slots, got, n, g.Err())
		}
	}
	if scanned == 0 || g.Stats().PageIns == 0 {
		t.Fatalf("%d budget(s) with a slot and %d page-in(s): the scan never paged, so it proved nothing",
			scanned, g.Stats().PageIns)
	}
	t.Logf("%d budget(s) reporting a slot, every block run on each: %d page-in(s), %d page-out(s)",
		scanned, g.Stats().PageIns, g.Stats().PageOuts)
}

// TestVictimIsTheBlockJustComputed pins the eviction policy. Blocks run lo..hi-1
// and wrap every token, so the block just behind the cursor is furthest from
// its next use: Belady's optimum is most-recently-used, the opposite of LRU.
// Page-in counts could not tell the two apart on a small model.
func TestVictimIsTheBlockJustComputed(t *testing.T) {
	g := newDevice(&devShared{layers: map[int]*layer{}})
	for li := 0; li < 5; li++ {
		// bytes is what a page-out would reclaim; a block that frees nothing
		// is not a candidate (see below).
		g.layers[li] = &layer{ok: true, pg: &page{in: true, bytes: 1 << 20}}
	}
	if got := g.victim(2, 0, 5, false); got != 1 {
		t.Fatalf("victim at block 2 of a 0..4 scan is %d, want 1 (the block just computed). "+
			"0 is LRU's answer and is the block needed soonest after the wrap", got)
	}
	if got := g.victim(0, 0, 5, false); got != 4 {
		t.Fatalf("victim at block 0 is %d, want 4: the scan wraps, so the block just before "+
			"the cursor is still the furthest from its next use", got)
	}
	// A block outside the scan is never used again in it, so it goes first.
	g.layers[9] = &layer{ok: true, pg: &page{in: true, bytes: 1 << 20}}
	if got := g.victim(2, 0, 5, false); got != 9 {
		t.Fatalf("victim is %d with block 9 resident and outside the scan [0,5): a block the "+
			"token will not visit must go before any block it will", got)
	}
	// A page already out is not a candidate (no double eviction).
	g.layers[9].pg.in = false
	if got := g.victim(2, 0, 5, false); got != 1 {
		t.Fatalf("victim is %d after block 9 was paged out; it must not be a candidate twice", got)
	}
	// A block whose page-out would reclaim nothing is not a candidate either:
	// on a device that imports the packed arena, evicting frees zero bytes,
	// and trim() would otherwise walk the whole resident set.
	g.layers[1].pg.bytes = 0
	if got := g.victim(2, 0, 5, false); got != 0 {
		t.Fatalf("victim is %d with block 1 holding no reclaimable bytes; the choice must "+
			"move on to the next furthest block, which is 0", got)
	}
	for li := range g.layers {
		g.layers[li].pg.bytes = 0
	}
	if got := g.victim(2, 0, 5, false); got != -1 {
		t.Fatalf("victim is %d when no resident block holds a reclaimable byte, want -1: "+
			"there is nothing to page out and saying so is what stops the loop", got)
	}
}

// TestOnlyAStreamedBlockIsAStreamedVictim: making room for a streamed block
// or for history takes streamed blocks only, so a block placed resident keeps
// its place whatever streams beside it; the budget's owner (trim) may still
// take any unpinned one.
func TestOnlyAStreamedBlockIsAStreamedVictim(t *testing.T) {
	g := newDevice(&devShared{layers: map[int]*layer{}, stream: map[int]bool{3: true}})
	for li := 0; li < 5; li++ {
		g.layers[li] = &layer{ok: true, pg: &page{in: true, bytes: 1 << 20}}
	}
	if got := g.victim(2, 0, 5, true); got != 3 {
		t.Fatalf("the streamed victim at block 2 is %d, want 3: the one streamed block, "+
			"though 1 is further from its next use", got)
	}
	if got := g.victim(2, 0, 5, false); got != 1 {
		t.Fatalf("trim's victim at block 2 is %d, want 1: the budget's owner takes any "+
			"unpinned block", got)
	}
	delete(g.stream, 3)
	if got := g.victim(2, 0, 5, true); got != -1 {
		t.Fatalf("with nothing streamed the streamed victim is %d, want -1: no resident "+
			"block is evicted for a streamed one", got)
	}
}
