package tier

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The device unpack, at the tier's level: a page-in of a covered format must
// upload the raw GGUF bytes and extract on the card, without a host pack, a
// moved byte, a slot lost to staging, or memory a budget shrink cannot
// reclaim.
//
// The device unpack and the host packer produce the same bytes by
// construction, so a tier that quietly kept packing on the host passes every
// byte-level check. Only counters tell them apart: Stats.Unpacks and
// Stats.Packs are asserted in every test, each against a violating arm.

// gpuLock is jit/gpu/backend's cross-process test lock (internal/testlock), so
// this package and backend's unpack gate do not allocate on one card at the
// same time under `go test ./...`.
func gpuLock(t *testing.T) { testlock.GPU(t) }

// unpackPlan's every k is a multiple of 256, Q4_K's super-block.
func unpackPlan() *nn.LayerPlan {
	return &nn.LayerPlan{
		NEmbd: 256, NHead: 8, NKVHead: 4, HeadDim: 32, NRot: 32, NFFN: 512,
		MaxSeq: 64, RMSEps: 1e-5, RopeBase: 10000,
	}
}

// q4kBlocks builds n blocks of Q4_K weights with distinct, pseudo-random bytes:
// distinct because resident() keys on the address, random because a zero
// tensor packs to zeros through any layout.
//
// mix makes the gate and up matrices Q6_K, a format the device unpack does not
// cover, so one block exercises both paths.
func q4kBlocks(p *nn.LayerPlan, n int, mix bool) []*nn.LayerWeights {
	seed := uint32(20260913)
	next := func() byte {
		seed = seed*1664525 + 1013904223
		return byte(seed >> 24)
	}
	mk := func(t quant.Type, rows, k int) nn.Weight {
		var blk int
		switch t {
		case quant.Q4_K:
			blk = 144
		case quant.Q6_K:
			blk = 210
		default:
			panic("q4kBlocks: unhandled type")
		}
		b := make([]byte, rows*(k/256)*blk)
		for i := range b {
			b[i] = next()
		}
		return nn.Weight{T: t, Data: b, Rows: rows, K: k}
	}
	out := make([]*nn.LayerWeights, n)
	kv := p.NKVHead * p.HeadDim
	ffn := quant.Q4_K
	if mix {
		ffn = quant.Q6_K
	}
	for i := range out {
		out[i] = &nn.LayerWeights{
			AttnNorm: make([]float32, p.NEmbd), FFNNorm: make([]float32, p.NEmbd),
			Wq:   mk(quant.Q4_K, p.NHead*p.HeadDim, p.NEmbd),
			Wk:   mk(quant.Q4_K, kv, p.NEmbd),
			Wv:   mk(quant.Q4_K, kv, p.NEmbd),
			Wo:   mk(quant.Q4_K, p.NEmbd, p.NHead*p.HeadDim),
			Gate: mk(ffn, p.NFFN, p.NEmbd),
			Up:   mk(ffn, p.NFFN, p.NEmbd),
			Down: mk(quant.Q4_K, p.NEmbd, p.NFFN),
		}
	}
	return out
}

// unpackTier builds a device whose budget holds fit blocks of this plan.
func unpackTier(t *testing.T, fit int, dev backend.Device, mix bool) (*GPU, *nn.LayerPlan) {
	t.Helper()
	p := unpackPlan()
	// Size one block on an unbounded budget, then give the real tier fit of
	// them.
	sz, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !sz.PrepLayer(0, p, q4kBlocks(p, 1, mix)[0]) {
		t.Fatalf("sizing run declined: %s", sz.Err())
	}
	// The device's own scratch is charged once, beside the blocks
	// (scratch.go): fit blocks is the scratch and fit of them.
	sc := sz.devs[0].scratch
	one := sz.devs[0].used - sc
	sz.Close()

	g, err := New([]Slot{{Dev: dev, Bytes: sc + one*uint64(fit)}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	streamBlocks(g, 64) // every block these tests offer
	return g, p
}

// TestAPageInUnpacksOnTheDeviceAndPacksNothingOnTheHost: for a covered format
// no host repack happens on page-in. It is a counter rather than a rate,
// because a repack produces the right bytes and shows only in seconds.
func TestAPageInUnpacksOnTheDeviceAndPacksNothingOnTheHost(t *testing.T) {
	run := func(t *testing.T, unpack bool) Stats {
		t.Helper()
		g, p := unpackTier(t, 2, &fakeDev{}, false)
		g.Config.NoUnpack = !unpack
		ws := q4kBlocks(p, 4, false)
		for i, w := range ws {
			if !g.PrepLayer(i, p, w) {
				t.Fatalf("block %d declined: %s", i, g.Err())
			}
		}
		// Four blocks through two slots: walking the scan pages evicted blocks
		// back in.
		d := g.devs[0]
		for li := 0; li < 4; li++ {
			if !d.pageIn(li, 0, 4) {
				t.Fatalf("page-in of block %d failed: %s", li, d.LastErr)
			}
		}
		return g.Stats()
	}

	st := run(t, true)
	// 1. Something actually paged, or the counters below prove nothing.
	if st.PageIns == 0 || st.PageOuts == 0 {
		t.Fatalf("%d page-ins and %d page-outs: nothing moved, so this gate proves nothing",
			st.PageIns, st.PageOuts)
	}
	// 2. Every tensor upload went through the device unpack...
	if st.Unpacks == 0 {
		t.Fatalf("Unpacks is 0 with %d page-ins: the tier packed on the host and uploaded "+
			"packed, which is the path this change removes", st.PageIns)
	}
	// 3. ...and nothing was packed on the host, by the tier or the arena.
	if st.Packs != 0 {
		t.Fatalf("%d host pack(s) on a Q4_K-only model; the device unpack covers every "+
			"tensor here, so each one is 0.90 GB/s of work the card does at ~47", st.Packs)
	}
	if st.NoUnpackKernel != 0 || st.NoUnpackRoom != 0 {
		t.Fatalf("%d tensor(s) fell back for want of a kernel and %d for want of room; "+
			"neither should be possible on this block: %s",
			st.NoUnpackKernel, st.NoUnpackRoom, st.UnpackWhy)
	}
	if st.StageVRAM == 0 {
		t.Fatal("StageVRAM is 0 while Unpacks is not: the staging buffer is what the raw " +
			"bytes are uploaded into, so it cannot be absent and the path have run")
	}
	t.Logf("%d page-in(s), %d page-out(s): %d tensor(s) unpacked on the device, %d packed "+
		"on the host, staging %d bytes", st.PageIns, st.PageOuts, st.Unpacks, st.Packs,
		st.StageVRAM)

	// Violation: with the device unpack off, every tensor must be host-packed.
	v := run(t, false)
	if v.Unpacks != 0 {
		t.Fatalf("violation arm: Config.NoUnpack is set and %d tensor(s) still unpacked on "+
			"the device, so the A/B arm does not control the path", v.Unpacks)
	}
	if v.Packs == 0 {
		t.Fatalf("violation arm: 0 host packs with the device unpack off, so Stats.Packs is " +
			"not counting what this gate reads it as")
	}
	if v.StageVRAM != 0 {
		t.Fatalf("violation arm: %d bytes of staging held with the unpack off", v.StageVRAM)
	}
	t.Logf("VIOLATION seen: with Config.NoUnpack set the same page-ins report %d unpacks and "+
		"%d host packs, and the gate above reads %q", v.Unpacks, v.Packs,
		fmt.Sprintf("%d host pack(s) on a Q4_K-only model", v.Packs))
}

// TestTheStagingIsChargedAndGivenBack is the leak gate on the staging buffer.
// Bytes the card holds that the budget cannot see get spent twice, so the
// staging must be exactly the widest covered tensor, included in used, and
// refunded when the blocks are released.
func TestTheStagingIsChargedAndGivenBack(t *testing.T) {
	g, p := unpackTier(t, 4, &fakeDev{}, false)
	d := g.devs[0]
	base := d.used
	ws := q4kBlocks(p, 2, false)
	// The widest covered tensor of a block: the staging holds one tensor,
	// never a block.
	var widest uint64
	for _, x := range []nn.Weight{ws[0].Wq, ws[0].Wk, ws[0].Wv, ws[0].Wo,
		ws[0].Gate, ws[0].Up, ws[0].Down} {
		if n := uint64(len(x.Data)); n > widest {
			widest = n
		}
	}
	for i, w := range ws {
		if !g.PrepLayer(i, p, w) {
			t.Fatalf("block %d declined: %s", i, g.Err())
		}
	}
	if d.StageVRAM != widest {
		t.Fatalf("staging is %d bytes and the widest covered tensor is %d; the buffer is "+
			"sized per TENSOR, so anything else means it is sized from the wrong thing",
			d.StageVRAM, widest)
	}
	if d.rawBytes != widest {
		t.Fatalf("StageVRAM says %d and the buffer is %d", d.StageVRAM, d.rawBytes)
	}
	// It is inside `used`, which is what slots() divides.
	var blocks uint64
	for _, l := range d.layers {
		for _, rp := range l.tensors() {
			blocks += (*rp).bytes()
		}
		blocks += l.kvBytes + l.auxBytes
	}
	blocks += d.kvPoolBytes() // paged history is the device's pool, not a block's
	// And the device's own scratch, charged beside them (scratch.go).
	if d.used != base+blocks+widest+d.scratch {
		t.Fatalf("used is %d; two blocks are %d, the staging %d and the scratch %d, so %d is "+
			"unaccounted for -- bytes the card holds that the budget cannot see",
			d.used, blocks, widest, d.scratch, int64(d.used)-int64(base+blocks+widest+d.scratch))
	}
	g.ReleaseLayers(0, 2)
	if d.used != base {
		t.Fatalf("used is %d after releasing every block, was %d before taking any: "+
			"%d bytes leaked", d.used, base, int64(d.used)-int64(base))
	}
	if d.StageVRAM != 0 || d.rawBuf != nil {
		t.Fatalf("the staging survived a release of every block (%d bytes): a migration "+
			"that leaves it behind leaks one widest-tensor allocation per relocation",
			d.StageVRAM)
	}
	t.Logf("staging %d bytes = the widest covered tensor, charged into used and refunded "+
		"in full; used %d -> %d -> %d", widest, base, base+blocks+widest, d.used)

	// Violation: with the unpack off there is no staging, so used after the
	// same two blocks must be short by exactly the buffer.
	g2, p2 := unpackTier(t, 4, &fakeDev{}, false)
	g2.Config.NoUnpack = true
	d2 := g2.devs[0]
	for i, w := range q4kBlocks(p2, 2, false) {
		if !g2.PrepLayer(i, p2, w) {
			t.Fatalf("violation arm: block %d declined: %s", i, g2.Err())
		}
	}
	if d2.StageVRAM != 0 {
		t.Fatalf("violation arm: %d bytes of staging with the unpack off", d2.StageVRAM)
	}
	if d2.used != base+blocks+d2.scratch {
		t.Fatalf("violation arm: used is %d, want %d (the unpack arm's %d less the %d of "+
			"staging, with its scratch of %d)", d2.used, base+blocks+d2.scratch, base+blocks+widest,
			widest, d2.scratch)
	}
	t.Logf("VIOLATION seen: with the unpack off the same two blocks charge %d instead of "+
		"%d, i.e. exactly the %d of staging", d2.used, base+blocks+widest, widest)
}

// TestShrinkingTheBudgetReleasesTheStaging: a budget shrink must give back the
// staging too, and demote blocks rather than release them, so ownership and KV
// history are unchanged and no token id can move.
func TestShrinkingTheBudgetReleasesTheStaging(t *testing.T) {
	g, p := unpackTier(t, 4, &fakeDev{}, false)
	d := g.devs[0]
	ws := q4kBlocks(p, 3, false)
	for i, w := range ws {
		if !g.PrepLayer(i, p, w) {
			t.Fatalf("block %d declined: %s", i, g.Err())
		}
	}
	if d.StageVRAM == 0 {
		t.Fatal("no staging held before the shrink, so this test would prove nothing")
	}
	held, blocks := d.used, len(d.layers)

	// Below the slot floor: everything demotes and nothing will page back in.
	if n, err := g.SetBudget(1 << 12); err != nil || n != 0 {
		t.Fatalf("%d block(s) still have weights resident after a budget of 4 KiB", n)
	}
	if d.StageVRAM != 0 || d.rawBuf != nil {
		t.Fatalf("the staging survived a shrink to 4 KiB (%d bytes): the budget is not the "+
			"control it claims to be", d.StageVRAM)
	}
	if d.used >= held {
		t.Fatalf("used is %d after the shrink and was %d before: the demotion released "+
			"nothing", d.used, held)
	}
	// Demoted, not released: the blocks still belong to this device with their
	// KV history.
	if len(d.layers) != blocks {
		t.Fatalf("%d blocks own this device after the shrink, was %d: a shrink that FORGETS "+
			"blocks is a placement change, and that can move a token id",
			len(d.layers), blocks)
	}
	if d.KVBytes == 0 && d.kvPoolBytes() == 0 {
		t.Fatal("the KV cache went out with the weights; it has no backing store, so a " +
			"shrink that drops it loses this session's own history")
	}
	afterShrink := d.used
	t.Logf("shrink to 4 KiB: %d -> %d bytes held, staging released, %d blocks still owned "+
		"with %d bytes of KV history intact", held, d.used, len(d.layers), d.KVBytes)

	// And it comes back: restoring the budget re-sizes the staging through
	// ensureRaw and the blocks page back in.
	g.SetBudget(held * 2)
	if !d.pageIn(0, 0, len(ws)) {
		t.Fatalf("block 0 would not page back in after the budget was restored: %s", d.LastErr)
	}
	if d.StageVRAM == 0 {
		t.Fatal("the staging did not come back on a page-in, so the path silently fell to " +
			"the host packer after one shrink")
	}
	t.Logf("restored: staging %d bytes, %d unpacks", d.StageVRAM, d.Stats.Unpacks)

	// Violation: without the unpack there is no staging, so the two arms'
	// shrinks must release different amounts, exactly the staging apart.
	g2, p2 := unpackTier(t, 4, &fakeDev{}, false)
	g2.Config.NoUnpack = true
	d2 := g2.devs[0]
	for i, w := range q4kBlocks(p2, 3, false) {
		if !g2.PrepLayer(i, p2, w) {
			t.Fatalf("violation arm: block %d declined: %s", i, g2.Err())
		}
	}
	held2 := d2.used
	g2.SetBudget(1 << 12)
	withStaging, without := held-afterShrink, held2-d2.used
	if withStaging == without {
		t.Fatalf("violation arm: both arms gave back %d bytes, so the staging was never "+
			"part of what a shrink releases", withStaging)
	}
	t.Logf("VIOLATION seen: the unpack arm's shrink released %d bytes and the no-unpack "+
		"arm's %d -- a difference of %d, the staging",
		withStaging, without, int64(withStaging)-int64(without))
}

// TestDeviceUnpackedWeightsAreTheHostPackersBytes is byte equality at the
// tier's seam. backend's TestUnpackMatchesHostPacker proves the kernel; this
// proves the wiring (shape, staging slice, destination order, and a second
// page-in through a reused staging buffer). Those are permutation bugs, hence
// equality rather than NMSE.
func TestDeviceUnpackedWeightsAreTheHostPackersBytes(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs[1:] {
		d.Close()
	}
	// unpackTier sizes the budget from a fake device's scratch, and a real
	// device's is its own: a V100's (sm_70) is 16 MB, which paged the block
	// straight back out of a four-block budget. Room for 64 blocks beside it,
	// which this test's one block never needs paging for.
	g, p := unpackTier(t, 64, devs[0], false)
	ws := q4kBlocks(p, 1, false)

	read := func(li int) [][]byte {
		t.Helper()
		d := g.devs[0]
		d.mu.Lock()
		defer d.mu.Unlock()
		l := d.layers[li]
		if l == nil {
			t.Fatalf("block %d is not resident", li)
		}
		var out [][]byte
		// tensors() carries optional slots (qwen3next's shared expert) that
		// other models leave nil; the count of seven present keeps the
		// check as strong as before.
		present := 0
		for _, rp := range l.tensors() {
			r := *rp
			if r == nil {
				continue
			}
			present++
			if !r.ok {
				t.Fatalf("block %d has a tensor that is not resident", li)
			}
			nq, nd, nsc, err := kernels.PackedWords(r.t, r.nrows, r.k)
			if err != nil {
				t.Fatal(err)
			}
			for i, b := range []backend.Buf{r.qs, r.d, r.sc} {
				n := [3]int{nq, nd, nsc}[i]
				if n == 0 {
					n = 1
				}
				buf := make([]byte, n*4)
				if err := b.Read(buf); err != nil {
					t.Fatalf("read back: %v", err)
				}
				out = append(out, buf)
			}
		}
		if present != 7 {
			t.Fatalf("block %d has %d resident tensors, want 7: a slot went missing "+
				"and the comparison below would silently cover less", li, present)
		}
		return out
	}

	if !g.PrepLayer(0, p, ws[0]) {
		t.Fatalf("PrepLayer declined: %s", g.Err())
	}
	if u := g.Stats().Unpacks; u != 7 {
		t.Fatalf("%d tensors unpacked on the device, want 7: the configuration under test "+
			"is not the one that ran (RULE 10c)", u)
	}
	// Read straight after placement: a block paged out by then (the budget
	// too small for the device's own scratch) has no bytes to compare.
	if n := g.devs[0].PageOuts; n != 0 {
		t.Fatalf("the block was paged out %d time(s) after placement: the budget holds %d bytes, "+
			"the device's scratch %d", n, g.devs[0].limit, g.devs[0].scratch)
	}
	first := read(0)

	// The host packer's answer, computed here.
	var want [][]byte
	for _, x := range []nn.Weight{ws[0].Wq, ws[0].Wk, ws[0].Wv, ws[0].Wo,
		ws[0].Gate, ws[0].Up, ws[0].Down} {
		qs, dw, sc, err := kernels.PackWeights(kernels.Q4_K, x.Data, x.Rows, x.K)
		if err != nil {
			t.Fatalf("host pack: %v", err)
		}
		for _, v := range [][]uint32{qs, dw, sc} {
			if len(v) == 0 {
				v = []uint32{0}
			}
			b := make([]byte, len(v)*4)
			for i, w := range v {
				b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] =
					byte(w), byte(w>>8), byte(w>>16), byte(w>>24)
			}
			want = append(want, b)
		}
	}
	cmp := func(tag string, got [][]byte) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d arrays, want %d", tag, len(got), len(want))
		}
		for i := range want {
			if !bytes.Equal(want[i], got[i]) {
				bad, at := 0, -1
				for j := range want[i] {
					if want[i][j] != got[i][j] {
						bad++
						if at < 0 {
							at = j
						}
					}
				}
				t.Fatalf("%s: array %d (tensor %d, %s) differs in %d of %d bytes, first at "+
					"%d: host %02x, device %02x", tag, i, i/3,
					[3]string{"qs", "d", "sc"}[i%3], bad, len(want[i]), at,
					want[i][at], got[i][at])
			}
		}
	}
	cmp("first upload", first)

	// A second page-in through the same staging buffer catches a staging not
	// fully overwritten or a kernel cached under the wrong shape.
	g.ReleaseLayers(0, 1)
	if !g.PrepLayer(0, p, ws[0]) {
		t.Fatalf("page-in: PrepLayer declined: %s", g.Err())
	}
	cmp("second upload", read(0))
	if u := g.Stats().Unpacks; u != 14 {
		t.Fatalf("%d unpacks after two uploads of seven tensors, want 14", u)
	}
	if pk := g.Stats().Packs; pk != 0 {
		t.Fatalf("%d host packs on %s: the device unpack ran and the host packed anyway",
			pk, devs[0].Name())
	}
	t.Logf("%s: %d arrays byte-identical to PackWeightsInto across two uploads, 0 host packs",
		devs[0].Name(), len(want))

	// Violation: flipping one payload byte of the source (a Q4_K block is 4
	// bytes of d/dmin, 12 of scales, then 128 of nibbles) must change the
	// packed qs array, or the comparison above is not testing the weights.
	x := ws[0].Wq
	mutated := append([]byte(nil), x.Data...)
	at := (len(mutated)/2/144)*144 + 16 // the first payload byte of a whole block
	mutated[at] ^= 0x10
	qs, _, _, err := kernels.PackWeights(kernels.Q4_K, mutated, x.Rows, x.K)
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, len(qs)*4)
	for i, w := range qs {
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(w), byte(w>>8), byte(w>>16), byte(w>>24)
	}
	if bytes.Equal(b, first[0]) {
		t.Fatal("violation: one payload byte of the q projection was flipped and the packed " +
			"array did not change, so byte equality here is not testing the weights")
	}
	diff := 0
	for i := range b {
		if b[i] != first[0][i] {
			diff++
		}
	}
	t.Logf("VIOLATION seen: one flipped source byte moves %d packed byte(s), and the gate "+
		"above compares exactly this array", diff)
}

// TestAModelThatFitsAdmitsTheSameBlocksEitherWay: a charged staging buffer
// takes s bytes out of the budget, which can cost one block exactly when B mod
// b is under s. It sweeps budgets from two blocks to four in steps under the
// staging size, because the margin is not computable up front (the room check
// prices weights; KV and norms are charged after it).
func TestAModelThatFitsAdmitsTheSameBlocksEitherWay(t *testing.T) {
	p := unpackPlan()

	// One block's cost, and the staging the unpack would want.
	sz, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !sz.PrepLayer(0, p, q4kBlocks(p, 1, false)[0]) {
		t.Fatalf("sizing run declined: %s", sz.Err())
	}
	stage, sc := sz.devs[0].StageVRAM, sz.devs[0].scratch
	one := sz.devs[0].used - stage - sc
	sz.Close()
	if stage == 0 {
		t.Fatal("the sizing run held no staging, so the sweep below would compare two " +
			"identical arms and prove nothing")
	}

	admit := func(budget uint64, unpack bool) (int, int, Stats) {
		g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: budget}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		g.Config.NoUnpack = !unpack
		// Paging off (the default): a declined block is lost for the run. It
		// stops at the first decline, because a device's blocks are one
		// contiguous run and the seam is where the first block is refused.
		n := 0
		for i, w := range q4kBlocks(p, 6, false) {
			if !g.PrepLayer(i, p, w) {
				break
			}
			n++
		}
		// n is the seam; pagesIn is how many of those still have weights on
		// the card, since a block can be admitted and then demoted by trim()
		// in the same call.
		return n, g.devs[0].pagesIn, g.Stats()
	}

	step := stage / 8
	if step == 0 {
		step = 1
	}
	lo, hi := sc+2*one, sc+4*one
	checked, boundaries := 0, 0
	prev := -1
	for b := lo; b <= hi; b += step {
		on, onRes, st := admit(b, true)
		off, offRes, _ := admit(b, false)
		// Not fewer, rather than the same: only a regression refuses a
		// change, and a block lost or left demoted to staging is one.
		if on < off || onRes < offRes {
			t.Fatalf("budget %d (%.2f blocks): the device unpack placed %d block(s) with %d "+
				"resident and the host packer %d with %d. A staging buffer cost a block on "+
				"a model that FITS, which is the RULE 6 obligation this change is under. "+
				"%d tensors were unpacked, %d host-packed, %d refused the staging",
				b, float64(b)/float64(one), on, onRes, off, offRes,
				st.Unpacks, st.Packs, st.NoUnpackRoom)
		}
		if prev >= 0 && off != prev {
			boundaries++
		}
		prev = off
		checked++
	}
	// The sweep must cross an admission boundary, or it tested a constant.
	if boundaries < 1 {
		t.Fatalf("%d budgets from %d to %d and the block count never changed; the sweep "+
			"never reached a margin, so it did not test one", checked, lo, hi)
	}
	t.Logf("%d budgets from %d to %d in steps of %d (staging %d, block %d): identical block "+
		"counts in both arms across %d admission boundaries",
		checked, lo, hi, step, stage, one, boundaries)

	// Violation: the paging arm deliberately keeps the staging, so somewhere
	// in this range it must place fewer than the host-packer arm, or the
	// correction in prepLayer is untested.
	dis := 0
	for b := lo; b <= hi; b += step {
		g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: b}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		streamBlocks(g, 6)
		g.Config.NoUnpack = false
		// Every block is offered once, in order, so only the budget differs.
		n := 0
		for i, w := range q4kBlocks(p, 6, false) {
			if !g.PrepLayer(i, p, w) {
				break
			}
			n++
		}
		hold, res := g.devs[0].StageVRAM, g.devs[0].pagesIn
		g.Close()
		off, offRes, _ := admit(b, false)
		if hold > 0 && (n < off || res < offRes) {
			dis++
		}
	}
	if dis == 0 {
		t.Fatalf("the paging arm, which KEEPS the staging, matched the host-packer arm at "+
			"every one of %d budgets -- so a charged staging never costs a block here and "+
			"the correction in prepLayer is untested", checked)
	}
	t.Logf("VIOLATION seen: with the staging KEPT (the paging arm, where it is deliberately "+
		"kept because it is what stops a repack) the placed-or-resident count falls short of "+
		"the host-packer arm at %d of %d budgets; the non-paging arm above falls short at 0",
		dis, checked)
}
