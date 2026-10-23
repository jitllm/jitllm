package tier

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The zero-copy page-in, at the tier's level: on a device whose heap is host
// memory, a block's weights are wrapped rather than uploaded, and cost the
// device and its pool nothing. "Every block placed" and "the bytes are right"
// are also true of a copy, so the assertions read the import count, the byte
// charge, and a Write flag on the fake buffer that exposes a secret copy.

// impDev is a device that can wrap host memory, with whether its memory is the
// host's and what alignment it demands settable separately.
type impDev struct {
	recDev
	unified bool
	align   int
	refuse  bool // the driver says no, which must fall back to a copy

	asked   int       // Import calls, refused ones included
	imports int       // Import calls that returned a wrapper
	regions []uintptr // the host addresses handed to the driver, in order
}

func (d *impDev) Name() string        { return "imp" }
func (d *impDev) UnifiedMemory() bool { return d.unified }
func (d *impDev) ImportAlign() int    { return d.align }
func (d *impDev) Mem() (uint64, uint64, error) {
	return 0, 0, fmt.Errorf("imp: does not report memory")
}

func (d *impDev) Import(p unsafe.Pointer, n int) (backend.Buf, error) {
	d.asked++
	if d.refuse {
		return nil, fmt.Errorf("imp: this driver refuses host pointers")
	}
	if d.align <= 0 {
		return nil, fmt.Errorf("imp: this device cannot import")
	}
	// The same two conditions the real drivers enforce: pointer and length
	// both aligned.
	if uintptr(p)%uintptr(d.align) != 0 {
		return nil, fmt.Errorf("imp: pointer %p is not %d-aligned", p, d.align)
	}
	if n%d.align != 0 {
		return nil, fmt.Errorf("imp: length %d is not a multiple of %d", n, d.align)
	}
	d.imports++
	d.regions = append(d.regions, uintptr(p))
	return &impBuf{p: p, n: n}, nil
}

// impBuf aliases the caller's memory. A Write to it is the failure being
// tested for: the tier copied bytes it was told not to copy.
type impBuf struct {
	p     unsafe.Pointer
	n     int
	wrote bool
}

func (b *impBuf) Write(p []byte) error {
	b.wrote = true
	return nil
}

func (b *impBuf) WriteAt(off int, p []byte) error {
	b.wrote = true
	return nil
}

func (b *impBuf) Read(p []byte) error {
	copy(p, unsafe.Slice((*byte)(b.p), b.n))
	return nil
}

func (b *impBuf) Free() {}

// importedBufs counts a resident block's tensors by how they got there, and
// reports whether any wrapper was written to. It counts tensors, not buffers,
// because a format with no second scale array still gets a one-word
// placeholder buffer.
func importedBufs(t *testing.T, g *devTier, li int) (wrapped, copied int, wrote bool) {
	t.Helper()
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil {
		t.Fatalf("block %d is not resident", li)
	}
	for _, r := range []*resident{l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down} {
		if r == nil || !r.ok {
			t.Fatalf("block %d has a tensor that is not resident", li)
		}
		if r.imported {
			wrapped++
		} else {
			copied++
		}
		for _, b := range []backend.Buf{r.qs, r.d, r.sc} {
			ib, ok := b.(*impBuf)
			if !ok {
				// A copied tensor's buffers, or the one-word placeholder an
				// absent scale array gets.
				continue
			}
			if !r.imported {
				t.Fatalf("block %d has a WRAPPER on a tensor the tier did not mark imported: "+
					"the charge and the buffer disagree, and the charge is what the budget reads", li)
			}
			wrote = wrote || ib.wrote
		}
	}
	return wrapped, copied, wrote
}

// prepOne builds a one-device tier with the given import behaviour and
// prepares block 0.
func prepOne(t *testing.T, d *impDev, budget, arena uint64) (*GPU, *devTier) {
	t.Helper()
	g, err := New([]Slot{{Dev: d, Bytes: budget}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	g.SetArena(arena)
	p := arenaPlan()
	w := blockWeights(p)
	fill(w, 20260913)
	if !g.devs[0].PrepLayer(0, p, w) {
		g.Close()
		t.Fatalf("PrepLayer declined: %s", g.devs[0].Err())
	}
	return g, g.devs[0]
}

// TestUnifiedImportDoesNotCopyAndDoesNotCharge: on a unified device a block's
// weights become descriptors over the arena's bytes, and the device's weight
// charge must be zero. A charged import would make the card decline blocks to
// protect memory nobody spends.
func TestUnifiedImportDoesNotCopyAndDoesNotCharge(t *testing.T) {
	d := &impDev{unified: true, align: int(kernels.HostAlign())}
	g, one := prepOne(t, d, 1<<40, 1<<30)
	defer g.Close()

	wrapped, copied, wrote := importedBufs(t, one, 0)
	if copied != 0 || wrapped == 0 {
		t.Fatalf("%d wrapped and %d copied weight tensors: a unified device is supposed to "+
			"wrap every one of them", wrapped, copied)
	}
	if wrote {
		t.Fatal("a wrapped buffer was WRITTEN to: that is a copy wearing the import's name, " +
			"and the bytes have now been spent twice")
	}
	if d.imports == 0 {
		t.Fatal("the device was never asked to import, so this test is measuring a copy")
	}
	st := one.stats()
	// The tier counts tensors; the driver is asked once per non-empty array,
	// so its count must not be below the tier's.
	if st.Imports != wrapped || d.imports < st.Imports {
		t.Fatalf("the tier counted %d imported tensors over %d wrapped, and the device saw "+
			"%d Import calls", st.Imports, wrapped, d.imports)
	}
	if st.ImportBytes == 0 {
		t.Fatal("ImportBytes is zero, so the saving has no number on it")
	}
	// The weight charge, isolated from the parts that are genuinely resident.
	one.mu.Lock()
	l := one.layers[0]
	weights := one.used - (l.kvBytes + l.auxBytes + one.kvPoolBytes())
	pool := one.pool.Used()
	pgBytes, pageBytes := l.pg.bytes, one.pageBytes
	one.mu.Unlock()
	if weights != 0 {
		t.Fatalf("%d bytes of weights charged to a device that imported all of them "+
			"(%.2f MiB): the arena is already holding those bytes and HostReserved already "+
			"counts them, so this is the same memory spent twice",
			weights, float64(weights)/(1<<20))
	}
	// The page record must agree with the charge: pg.bytes is what a page-in
	// asks room() for, so a nonzero price would evict for nothing.
	if pgBytes != 0 || pageBytes != 0 {
		t.Fatalf("the page record prices this block at %d bytes and the device holds %d "+
			"pageable bytes, on a block whose weights are wrappers over the arena: a "+
			"page-in of it moves nothing and reclaims nothing", pgBytes, pageBytes)
	}
	if pool != l.kvBytes+l.auxBytes+one.kvPoolBytes() {
		t.Fatalf("the pool holds %d bytes, want %d (KV %d + aux %d): an import must not "+
			"reach the pool either -- that is the account an integrated GPU shares with the host",
			pool, l.kvBytes+l.auxBytes+one.kvPoolBytes(), l.kvBytes+one.kvPoolBytes(), l.auxBytes)
	}
	t.Logf("%d wrapped tensors, %.2f MiB of weights imported, %d bytes charged (KV %.2f MiB)",
		wrapped, float64(st.ImportBytes)/(1<<20), weights, float64(l.kvBytes)/(1<<20))
}

// TestImportIsRefusedWhereItWouldCostMore: the guard has several halves, each
// with a plausible wrong answer.
//
//	discrete    the device can import but its memory is not the host's, so an
//	            imported weight would be re-read across PCIe by every matvec.
//	no arena    nothing has been packed to wrap.
//	alignment   the driver wants more than the arena lays out.
//
// Each must fall back to the copy path and charge for it.
func TestImportIsRefusedWhereItWouldCostMore(t *testing.T) {
	for _, c := range []struct {
		name  string
		dev   *impDev
		arena uint64
		// mute says the tier must not even ask the driver, since it can
		// tell every call would be refused.
		mute bool
	}{
		{"discrete", &impDev{unified: false, align: int(kernels.HostAlign())}, 1 << 30, true},
		{"no arena", &impDev{unified: true, align: int(kernels.HostAlign())}, 0, true},
		{"alignment the arena does not provide",
			&impDev{unified: true, align: int(kernels.HostAlign()) * 4}, 1 << 30, true},
		{"the driver refuses",
			&impDev{unified: true, align: int(kernels.HostAlign()), refuse: true}, 1 << 30, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			g, one := prepOne(t, c.dev, 1<<40, c.arena)
			defer g.Close()
			wrapped, copied, _ := importedBufs(t, one, 0)
			if wrapped != 0 {
				t.Fatalf("%d buffers were imported: %s", wrapped, c.name)
			}
			if copied == 0 {
				t.Fatal("no tensors at all, so nothing was tested")
			}
			one.mu.Lock()
			l := one.layers[0]
			weights := one.used - (l.kvBytes + l.auxBytes + one.kvPoolBytes())
			one.mu.Unlock()
			if weights == 0 {
				t.Fatal("a COPIED block was charged nothing: the fallback path lost the charge, " +
					"and a card that does not count what it holds overcommits silently")
			}
			if c.mute && c.dev.asked != 0 {
				t.Fatalf("the driver was asked to import %d times on a device where every "+
					"call is refused before it is made: that is an FFI call per array per "+
					"page-in, spent to learn something the tier already knew", c.dev.asked)
			}
			if !c.mute && c.dev.asked == 0 {
				t.Fatal("the driver was never asked, so the fallback under test is the wrong one")
			}
			t.Logf("%d copied tensors, %.2f MiB charged, %d Import calls",
				copied, float64(weights)/(1<<20), c.dev.asked)
		})
	}
}

// TestImportedBlocksNeverPageOut: a free page-in means eviction stops. "0
// page-outs" is also true of a tier that placed one block, so the copying arm
// runs first on a budget sized to force paging, and must page.
func TestImportedBlocksNeverPageOut(t *testing.T) {
	p := arenaPlan()
	// One weight set per block: resident() keys on the bytes, so shared
	// weights would never run out of room.
	const blocks = 6
	ws := make([]*nn.LayerWeights, blocks)
	for li := range ws {
		ws[li] = blockWeights(p)
		fill(ws[li], uint32(20260913+li))
	}

	// The budget is sized from the block: it must hold two blocks' weights
	// (or the device declines instead of paging) and fewer than six.
	budget := func() uint64 {
		d := &impDev{align: int(kernels.HostAlign())}
		g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		if !g.devs[0].PrepLayer(0, p, ws[0]) {
			t.Fatalf("sizing PrepLayer declined: %s", g.devs[0].Err())
		}
		g.devs[0].mu.Lock()
		defer g.devs[0].mu.Unlock()
		l := g.devs[0].layers[0]
		// Room for every block's permanent parts plus two blocks of weights.
		return blocks*(l.kvBytes+l.auxBytes) + 2*g.devs[0].widest
	}()

	run := func(t *testing.T, unified bool) (pageOuts, placed, imports int) {
		d := &impDev{unified: unified, align: int(kernels.HostAlign())}
		g, err := New([]Slot{{Dev: d, Bytes: budget}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		streamBlocks(g, blocks)
		g.SetArena(1 << 30)
		one := g.devs[0]
		for li := 0; li < blocks; li++ {
			if one.prepLayer(0, li, p, ws[li], true) {
				placed++
			}
		}
		st := one.stats()
		return st.PageOuts, placed, st.Imports
	}

	outs, placed, imports := run(t, false)
	if outs == 0 {
		t.Fatalf("the COPYING arm placed %d blocks without paging out once, so this budget "+
			"does not exercise the pager and the importing arm's zero would prove nothing", placed)
	}
	if imports != 0 {
		t.Fatalf("the copying arm imported %d tensors", imports)
	}
	t.Logf("copying:   %d blocks placed, %d page-outs", placed, outs)

	outs2, placed2, imports2 := run(t, true)
	if imports2 == 0 {
		t.Fatal("the importing arm imported nothing")
	}
	if outs2 != 0 {
		t.Fatalf("the importing arm paged %d blocks OUT: a wrap costs no device bytes, so "+
			"there is nothing to reclaim and nothing to reclaim it for", outs2)
	}
	if placed2 < placed {
		t.Fatalf("the importing arm placed %d blocks and the copying arm placed %d: importing "+
			"is supposed to admit MORE, not fewer", placed2, placed)
	}
	t.Logf("importing: %d blocks placed, %d page-outs, %d tensors wrapped", placed2, outs2, imports2)
}

// TestImportingDeviceIsNotDeclinedForWeightsItNeverHolds: slots() declines
// below two slots because a swap needs somewhere to swap through, but a device
// that wraps host pointers never holds its weights, so the arena's budget is
// the ceiling instead.
func TestImportingDeviceIsNotDeclinedForWeightsItNeverHolds(t *testing.T) {
	p := arenaPlan()
	const blocks = 4
	ws := make([]*nn.LayerWeights, blocks)
	for li := range ws {
		ws[li] = blockWeights(p)
		fill(ws[li], uint32(20260914+li))
	}
	// Room for every block's KV and norms and half of one block's weights:
	// below two slots by construction.
	budget := func() uint64 {
		d := &impDev{align: int(kernels.HostAlign())}
		g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		if !g.devs[0].PrepLayer(0, p, ws[0]) {
			t.Fatalf("sizing PrepLayer declined: %s", g.devs[0].Err())
		}
		g.devs[0].mu.Lock()
		defer g.devs[0].mu.Unlock()
		l := g.devs[0].layers[0]
		return blocks*(l.kvBytes+l.auxBytes) + g.devs[0].kvPoolBytes()*uint64(blocks) + g.devs[0].widest/2
	}()

	run := func(unified bool) (placed, slots int, why string) {
		d := &impDev{unified: unified, align: int(kernels.HostAlign())}
		g, err := New([]Slot{{Dev: d, Bytes: budget}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		streamBlocks(g, blocks)
		g.SetArena(1 << 30)
		for li := 0; li < blocks; li++ {
			if g.devs[0].prepLayer(0, li, p, ws[li], true) {
				placed++
			}
		}
		return placed, g.devs[0].stats().Slots, g.devs[0].Err()
	}

	placed, slots, why := run(false)
	if placed >= blocks {
		t.Fatalf("the COPYING arm placed all %d blocks on a budget below two slots, so the "+
			"budget is not the one this test needs", blocks)
	}
	t.Logf("copying:   %d of %d blocks, %d slot(s) -- %s", placed, blocks, slots, why)

	placed2, slots2, _ := run(true)
	if placed2 != blocks {
		t.Fatalf("the IMPORTING arm placed %d of %d blocks with %d slot(s): its weights cost "+
			"the device nothing, so there is no budget for them to exceed",
			placed2, blocks, slots2)
	}
	t.Logf("importing: %d of %d blocks, %d slot(s)", placed2, blocks, slots2)
}

// TestShrinkingTheBudgetDoesNotDemoteImportedBlocks: on an importing device a
// shrink has nothing to give back, so paging a wrapper out reclaims zero bytes
// and would only demote blocks.
func TestShrinkingTheBudgetDoesNotDemoteImportedBlocks(t *testing.T) {
	p := arenaPlan()
	const blocks = 4
	ws := make([]*nn.LayerWeights, blocks)
	for li := range ws {
		ws[li] = blockWeights(p)
		fill(ws[li], uint32(20260915+li))
	}
	run := func(unified bool) (before, after, outs int) {
		d := &impDev{unified: unified, align: int(kernels.HostAlign())}
		g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		streamBlocks(g, blocks)
		g.SetArena(1 << 30)
		for li := 0; li < blocks; li++ {
			if !g.devs[0].prepLayer(0, li, p, ws[li], true) {
				t.Fatalf("block %d declined: %s", li, g.devs[0].Err())
			}
		}
		before = g.devs[0].SetBudget(1 << 40) // no change: the baseline
		after = g.devs[0].SetBudget(1 << 16)  // far below one block
		return before, after, g.devs[0].stats().PageOuts
	}

	before, after, outs := run(false)
	if after >= before {
		t.Fatalf("the COPYING arm still holds %d of %d blocks' weights after the budget was "+
			"cut to 64 KiB, so this budget does not exercise a demotion at all", after, before)
	}
	t.Logf("copying:   %d -> %d blocks resident, %d page-outs", before, after, outs)

	before2, after2, outs2 := run(true)
	if after2 != before2 || outs2 != 0 {
		t.Fatalf("the IMPORTING arm went from %d to %d blocks with %d page-outs: paging out a "+
			"wrapper reclaims nothing, so the device is exactly as over budget as it was and "+
			"the only result is %d blocks demoted for nothing",
			before2, after2, outs2, before2-after2)
	}
	t.Logf("importing: %d -> %d blocks resident, %d page-outs", before2, after2, outs2)
}

// TestDropArenaUnwrapsBeforeItUnmaps is the lifetime gate: an imported buffer
// is the arena's memory, and Release() unmaps, so a wrapped block must not stay
// resident across DropArena (its violation is a later SIGSEGV in a driver). A
// copied block holds its own bytes and must be left alone.
func TestDropArenaUnwrapsBeforeItUnmaps(t *testing.T) {
	p := arenaPlan()
	w := blockWeights(p)
	fill(w, 20260916)

	for _, unified := range []bool{true, false} {
		name := "wrapped"
		if !unified {
			name = "copied"
		}
		t.Run(name, func(t *testing.T) {
			d := &impDev{unified: unified, align: int(kernels.HostAlign())}
			g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer g.Close()
			g.SetArena(1 << 30)
			if !g.devs[0].PrepLayer(0, p, w) {
				t.Fatalf("PrepLayer declined: %s", g.devs[0].Err())
			}
			if freed := g.DropArena(0, 1); freed == 0 {
				t.Fatal("DropArena released nothing, so there was no entry to be unsafe about")
			}
			one := g.devs[0]
			one.mu.Lock()
			resident := one.layers[0].pg.in
			one.mu.Unlock()
			if unified && resident {
				t.Fatal("the block is still resident after its arena entry was UNMAPPED: its " +
					"weight buffers are that memory, so the next launch reads unmapped pages " +
					"-- a SIGSEGV in the driver, at a point unrelated to this call")
			}
			if !unified && !resident {
				t.Fatal("a COPIED block was paged out by DropArena: it holds its own bytes and " +
					"keeps computing from them, so demoting it costs a page-in for nothing")
			}
			t.Logf("resident after DropArena: %v", resident)
		})
	}
}

// TestHostReservedCountsTheArena: the packed arena is host memory, so
// HostReserved must include it.
func TestHostReservedCountsTheArena(t *testing.T) {
	// Both pool shapes: a discrete-only tier has no host pool and returns
	// early, an integrated one sums the host-heap devices first.
	for _, unified := range []bool{false, true} {
		name := "discrete, no host pool"
		if unified {
			name = "integrated, on the host pool"
		}
		t.Run(name, func(t *testing.T) {
			d := &impDev{unified: unified, align: int(kernels.HostAlign())}
			slot := Slot{Dev: d, Bytes: 1 << 28}
			if unified {
				slot.Pool = NewHostPool(1 << 30)
			}
			g, err := New([]Slot{slot}, WithDeviceTune(TuneOff))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer g.Close()
			before := g.HostReserved()
			const arena = 3 << 30
			g.SetArena(arena)
			after := g.HostReserved()
			if after-before != arena {
				t.Fatalf("HostReserved moved by %d when the arena took a %d-byte budget: "+
					"the arena is host RAM and the host is the one that has to stop counting it",
					after-before, uint64(arena))
			}
			t.Logf("HostReserved %.2f -> %.2f GiB with a %.2f GiB arena",
				float64(before)/(1<<30), float64(after)/(1<<30), float64(arena)/(1<<30))
		})
	}
}
