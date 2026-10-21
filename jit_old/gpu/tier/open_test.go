package tier

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// The spec grammar, the budget defaults, the pool arithmetic and the rule that
// an integrated GPU stays usable. Excluding a device would make every total
// add up, so the budget gate and the placement gate are both here.

func TestDeviceSpecGrammar(t *testing.T) {
	for _, tc := range []struct {
		spec string
		want []Entry
	}{
		{"", []Entry{{API: "auto", Text: "auto"}}},
		{"cpu", []Entry{{API: "cpu", Text: "cpu"}}},
		{"all", []Entry{{API: "all", Text: "all"}}},
		{"gpu:1", []Entry{{API: "gpu", Sel: "1", HasSel: true, Text: "gpu:1"}}},
		{"CUDA:0", []Entry{{API: "ptx", Sel: "0", HasSel: true, Text: "CUDA:0"}}},
		{"ptx", []Entry{{API: "ptx", Text: "ptx"}}},
		{"metal", []Entry{{API: "msl", Text: "metal"}}},
		{"vulkan:iris", []Entry{{API: "spirv", Sel: "iris", HasSel: true, Text: "vulkan:iris"}}},
		// A per-device byte budget.
		{"cuda:0=3G, vulkan:1=8G", []Entry{
			{API: "ptx", Sel: "0", HasSel: true, Bytes: 3 << 30, Text: "cuda:0=3G"},
			{API: "spirv", Sel: "1", HasSel: true, Bytes: 8 << 30, Text: "vulkan:1=8G"},
		}},
		// cpu beside a device is explicit, not a contradiction.
		{"cpu,cuda:0,vulkan:1", []Entry{
			{API: "cpu", Text: "cpu"},
			{API: "ptx", Sel: "0", HasSel: true, Text: "cuda:0"},
			{API: "spirv", Sel: "1", HasSel: true, Text: "vulkan:1"},
		}},
	} {
		got, err := ParseDevices(tc.spec)
		if err != nil {
			t.Errorf("ParseDevices(%q): %v", tc.spec, err)
			continue
		}
		if len(got) != len(tc.want) {
			t.Errorf("ParseDevices(%q) = %+v, want %+v", tc.spec, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("ParseDevices(%q)[%d] = %+v, want %+v", tc.spec, i, got[i], tc.want[i])
			}
		}
	}
}

// TestDeviceSpecRefusalsSayWhatTheyGot: every rejection names what it got and
// what it wanted.
func TestDeviceSpecRefusalsSayWhatTheyGot(t *testing.T) {
	for _, tc := range []struct{ spec, got, want string }{
		{"nvidia", `"nvidia"`, "cuda"},
		{"cuda:abc", `"abc"`, "ORDINAL"},
		{"cuda:", `"cuda:"`, "colon"},
		{"cuda:0,,vulkan:1", "entry 2", "empty"},
		{"auto,cuda:0", `"auto"`, `"cuda:0"`},
		{"all,vulkan:1", `"all"`, `"vulkan:1"`},
		{"metal:1", `"1"`, "no selector"},
		{"cpu=3G", `"cpu=3G"`, "-maxmem"},
		{"cuda:0=banana", `"banana"`, "byte count"},
		{"auto:2", `"2"`, "no device selector"},
	} {
		_, err := ParseDevices(tc.spec)
		if err == nil {
			t.Errorf("ParseDevices(%q) was accepted", tc.spec)
			continue
		}
		if !strings.Contains(err.Error(), tc.got) || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("ParseDevices(%q) said %q; it must name what it got (%s) and what it wanted (%s)",
				tc.spec, err, tc.got, tc.want)
		}
	}
}

func TestParseBytes(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want uint64
	}{
		{"800000000", 800000000},
		{"3G", 3 << 30},
		{"3GiB", 3 << 30},
		{"3gb", 3 << 30},
		{"512M", 512 << 20},
		{"1.5G", 3 << 29},
		{"64K", 64 << 10},
		{"2T", 2 << 40},
	} {
		got, err := ParseBytes(tc.in)
		if err != nil || got != tc.want {
			t.Errorf("ParseBytes(%q) = %d, %v; want %d", tc.in, got, err, tc.want)
		}
	}
	for _, bad := range []string{"", "G", "-3G", "0", "3X", "three"} {
		if n, err := ParseBytes(bad); err == nil {
			t.Errorf("ParseBytes(%q) = %d, want an error", bad, n)
		}
	}
}

// TestBudgetComesFromTheDevice: the default budget comes from the device's
// free memory. It is asserted as a range (under free, close enough to be
// useful), not by recomputing the formula.
func TestBudgetComesFromTheDevice(t *testing.T) {
	for _, tc := range []struct {
		name           string
		free, total    uint64
		lo, hi         uint64
		wantSubstrings []string
	}{
		{"24 GiB card", 24 << 30, 24 << 30, 20 << 30, 24 << 30, []string{"24.00 GiB free", "headroom"}},
		// A 16 GiB card's free memory: an eighth of it (1.94 GiB) held back cost
		// a large mixture two blocks on two cards; the headroom is capped at 1 GiB.
		{"16 GiB card", 15<<30 + 1<<29, 16 << 30, 14<<30 + 1<<29, 15<<30 + 1<<29, []string{"15.50 GiB free"}},
		{"3 GiB card", 3 << 30, 4 << 30, 2 << 30, 3 << 30, []string{"3.00 GiB free"}},
		{"nearly full", 300 << 20, 4 << 30, 64 << 20, 300 << 20, []string{"0.29 GiB free"}},
		{"full", 0, 4 << 30, minBudget, minBudget, []string{"reports 0 of"}},
	} {
		d := &fakeDev{name: tc.name, free: tc.free, total: tc.total}
		got, why := budgetFor(d, 0, "-vram")
		t.Logf("%-12s free %5.2f GiB -> budget %5.2f GiB (%s)",
			tc.name, float64(tc.free)/(1<<30), float64(got)/(1<<30), why)
		if got < tc.lo || got > tc.hi {
			t.Errorf("%s: budget %d, want between %d and %d", tc.name, got, tc.lo, tc.hi)
		}
		if got > tc.free && tc.free > 0 {
			t.Errorf("%s: budget %d exceeds the %d bytes the device says are free", tc.name, got, tc.free)
		}
		for _, w := range tc.wantSubstrings {
			if !strings.Contains(why, w) {
				t.Errorf("%s: provenance %q does not mention %q", tc.name, why, w)
			}
		}
	}
	// A backend that cannot say falls back to the constant and says so.
	got, why := budgetFor(&fakeDev{name: "silent"}, 0, "-vram")
	if got != defaultBudget || !strings.Contains(why, "does not report") {
		t.Errorf("a device with no memory report got %d (%s), want the %d default with a reason",
			got, why, uint64(defaultBudget))
	}
	// An explicit ask wins over both, and says which flag set it.
	if got, why := budgetFor(&fakeDev{free: 24 << 30, total: 24 << 30}, 1<<30, "the spec"); got != 1<<30 || why != "the spec" {
		t.Errorf("an explicit budget gave %d (%s), want 1 GiB from the spec", got, why)
	}
}

// hostBox is a host with 31 GiB of RAM, a host budget of four fifths of it,
// a discrete 4 GiB card and an integrated GPU whose 20.94 GiB "free" is part
// of the same 31.
const (
	boxRAM       = 31 << 30
	boxHost      = boxRAM / 10 * 8        // what gguf.MemBudget would return
	boxIGPUFree  = 20*(1<<30) + 1<<29     // 20.94 GiB
	boxIGPUTotal = 23*(1<<30) + 1<<28     // 23.26 GiB
	boxCardFree  = 2*(1<<30) + 1<<29      // 2.56 GiB
	boxCardTotal = 3*(1<<30) + 1<<30/1024 // ~3.81 GiB; the exact figure is not load-bearing
)

func thisBox() (card, igpu *fakeDev) {
	return &fakeDev{name: "RTX 3050 Ti", api: "ptx", free: boxCardFree, total: boxCardTotal},
		&fakeDev{name: "Iris Xe", api: "spirv", unified: true, free: boxIGPUFree, total: boxIGPUTotal}
}

// TestHostAndIntegratedDeviceDoNotDoubleSpendTheMachine: an integrated GPU
// reports free memory like any other device, and believing it overcommits the
// machine (which thrashes rather than failing). Both halves are asserted: the
// pooled plan fits, and the per-device plan does not, so the gate cannot pass
// for want of a hazard.
func TestHostAndIntegratedDeviceDoNotDoubleSpendTheMachine(t *testing.T) {
	card, igpu := thisBox()
	slots := planSlots([]backend.Device{card, igpu}, nil, openOpts{host: boxHost, hostSet: true})
	g, err := New(slots)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()

	// The worst case: the iGPU filled to its ceiling, and the host losing all
	// of it (HostReserved reports what it holds, and with nothing placed that
	// is nothing; TestHostLosesOnlyWhatAUnifiedDeviceHolds).
	reserved := g.hostPool.Limit()
	host := uint64(0)
	if boxHost > reserved {
		host = boxHost - reserved
	}
	t.Logf("RAM %.2f GiB, host budget %.2f GiB -> host keeps %.2f with the iGPU full at %.2f, card %.2f (its own %.2f GiB)",
		gib(boxRAM), gib(boxHost), gib(host), gib(reserved), gib(g.devs[0].limit), gib(boxCardFree))

	if reserved == 0 {
		t.Fatal("the integrated GPU may hold NOTHING of the host pool: either it was " +
			"excluded (which fails the non-exclusion requirement) or its bytes are being " +
			"spent twice")
	}
	if total := host + reserved; total > boxRAM {
		t.Fatalf("host %d + devices on host memory %d = %d bytes on a %d byte machine",
			host, reserved, total, uint64(boxRAM))
	}
	if total := host + reserved; total > boxHost {
		t.Fatalf("host %d + devices on host memory %d = %d, over the %d budget they share",
			host, reserved, total, uint64(boxHost))
	}

	// The non-vacuity half: budgeted per device, the same plan must exceed
	// the machine.
	naive := uint64(boxHost)
	for _, d := range []*fakeDev{card, igpu} {
		if !unified(d) {
			continue
		}
		b, _ := budgetFor(d, 0, "-vram")
		naive += b
	}
	t.Logf("the same plan with a budget PER DEVICE: %.2f GiB on a %.2f GiB machine", gib(naive), gib(boxRAM))
	if naive <= boxRAM {
		t.Fatalf("a per-device budget totals %d on a %d byte machine, which is not an "+
			"overcommit -- this test is not exercising the hazard it exists for", naive, uint64(boxRAM))
	}
}

// TestIntegratedDeviceStillTakesBlocks: an iGPU's memory is the host's but its
// execution units are real, so it is charged to the host pool, never declined.
// CPU, integrated and discrete GPU must all hold part of the model.
func TestIntegratedDeviceStillTakesBlocks(t *testing.T) {
	p := fakePlan()
	const nblocks = 8
	ws := blocks(p, nblocks)
	two := twoBlockBytes(t, p)

	card, igpu := thisBox()
	// Two blocks each, eight offered: the host takes the rest. Each device's
	// two blocks are sized on its own API: a PTX device's prompt chunk takes
	// the m16n8 GEMM, whose binary16 activation scratch the others lack.
	twoCard := twoBlockBytesOn(t, &fakeDev{api: card.api}, p)
	slots := planSlots([]backend.Device{card, igpu}, []uint64{twoCard, two},
		openOpts{host: boxHost, hostSet: true})
	g, err := New(slots, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	for li := 0; li < nblocks; li++ {
		g.PrepLayer(li, p, ws[li])
	}

	per := g.Placed()
	onHost := nblocks
	for _, n := range per {
		onHost -= n
	}
	t.Logf("placement %v: card %d, integrated %d, host %d", placement(g, nblocks), per[0], per[1], onHost)
	for i, n := range per {
		if n == 0 {
			t.Fatalf("device %d (%s) took NO blocks: %v. An integrated GPU that is budgeted to "+
				"zero passes every arithmetic gate and has failed the requirement -- its memory "+
				"is the host's, its execution units are not",
				i, g.devs[i].Stats.Device, per)
		}
	}
	if onHost <= 0 {
		t.Fatalf("every block landed on a device (%v); this test needs the HOST to be the "+
			"third member of the split", per)
	}
	// The blocks must also run where they were placed.
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	if !g.Layers(0, per[0]+per[1], 0, 1, x, cs, nil, nil) {
		t.Fatalf("Layers over the placed prefix declined: %s", g.Err())
	}
	for i, d := range g.devs {
		if d.stats().Blocks == 0 {
			t.Fatalf("device %d (%s) holds %d block(s) and RAN none: the placement and the "+
				"submission disagree", i, d.Stats.Device, per[i])
		}
	}
}

// TestOneCardTwoAPIsIsOnePool: one card reached through two backends is two
// devices and one heap, so they share a pool. Pairing is by device UUID, not
// name: the names below differ for the same card and match for two distinct
// cards, which a name match gets wrong both ways.
func TestOneCardTwoAPIsIsOnePool(t *testing.T) {
	const card0 = "00112233445566778899aabbccddeeff"
	a := &fakeDev{name: "#0 RTX 3050 Ti Laptop GPU (sm_86)", api: "ptx", uuid: card0,
		free: boxCardFree, total: boxCardTotal}
	b := &fakeDev{name: "RTX 3050 Ti Laptop GPU", api: "spirv", uuid: card0,
		free: boxCardFree, total: boxCardTotal}
	slots := planSlots([]backend.Device{a, b}, nil, openOpts{host: boxHost, hostSet: true})
	if slots[0].Pool == nil || slots[0].Pool != slots[1].Pool {
		t.Fatalf("one card reached through two APIs got pool %q (%.2f GiB) and pool %q "+
			"(%.2f GiB): two budgets, and they are claims on the same 4 GiB",
			slots[0].Pool.Name(), gib(slots[0].Pool.Limit()),
			slots[1].Pool.Name(), gib(slots[1].Pool.Limit()))
	}
	if got := slots[0].Pool.Limit(); got > boxCardFree {
		t.Fatalf("the shared pool allows %d bytes on a card with %d free", got, uint64(boxCardFree))
	}
	// Two identical cards with the same name and different UUIDs stay two
	// pools; merging them would halve the machine's VRAM.
	c := &fakeDev{name: "RTX 3050 Ti Laptop GPU", api: "spirv", uuid: card0,
		free: boxCardFree, total: boxCardTotal}
	d := &fakeDev{name: "RTX 3050 Ti Laptop GPU", api: "spirv",
		uuid: "ffeeddccbbaa99887766554433221100", free: boxCardFree, total: boxCardTotal}
	two := planSlots([]backend.Device{c, d}, nil, openOpts{host: boxHost, hostSet: true})
	if two[0].Pool == two[1].Pool {
		t.Fatal("two cards with one name were merged into one pool: they are two heaps, " +
			"and half the machine's VRAM would go unused")
	}
	// A device without an identity gets its own pool, even beside another
	// unknown one.
	e := &fakeDev{name: "mystery", api: "fake", free: boxCardFree, total: boxCardTotal}
	f := &fakeDev{name: "mystery", api: "other", free: boxCardFree, total: boxCardTotal}
	anon := planSlots([]backend.Device{e, f}, nil, openOpts{host: boxHost, hostSet: true})
	if anon[0].Pool == anon[1].Pool {
		t.Fatal("two devices with NO identity were merged: an unknown identity must match " +
			"nothing, including another unknown one")
	}
}

// TestUnifiedBudgetIsCarvedFromTheHostBudget: the iGPU's share is bounded by
// the host's budget, not by what the device claims is free.
func TestUnifiedBudgetIsCarvedFromTheHostBudget(t *testing.T) {
	_, igpu := thisBox()
	// The device claims more free memory than the whole host budget.
	big := &fakeDev{name: "Iris Xe", api: "spirv", unified: true, free: boxRAM, total: boxRAM}
	slots := planSlots([]backend.Device{big}, nil, openOpts{host: boxHost, hostSet: true})
	if slots[0].Bytes >= boxHost {
		t.Fatalf("the integrated GPU was given %.2f GiB out of a %.2f GiB host budget: "+
			"it reported %.2f GiB free and every one of those bytes is the host's",
			gib(slots[0].Bytes), gib(boxHost), gib(uint64(boxRAM)))
	}
	if !strings.Contains(slots[0].Why, "host") {
		t.Errorf("provenance %q does not say the budget came off the host's", slots[0].Why)
	}
	// An explicit -vram is taken at face value.
	slots = planSlots([]backend.Device{igpu}, nil, openOpts{host: boxHost, hostSet: true, budget: 3 << 30})
	if slots[0].Bytes != 3<<30 || slots[0].Pool.Limit() != 3<<30 {
		t.Errorf("-vram 3G gave the device %d in a pool of %d, want 3 GiB in both",
			slots[0].Bytes, slots[0].Pool.Limit())
	}
}

// TestHostLosesOnlyWhatAUnifiedDeviceHolds: an integrated GPU's pool is a
// ceiling on the device, not a charge on the host. The host loses what the
// device holds -- nothing while it holds no block, each block's bytes as it is
// placed, and they come back as it is released. Charging the ceiling took a
// quarter of the host budget (1.62 GiB, measured) from the pager for an
// iGPU a bare `-devices gpu` opened and placed nothing on; against that
// HostReserved this fails at zero blocks, reading the pool's limit.
func TestHostLosesOnlyWhatAUnifiedDeviceHolds(t *testing.T) {
	p := fakePlan()
	const nblocks = 4
	ws := blocks(p, nblocks)
	noPrompt := WithConfig(func(c *Config) { c.NoScratchReserve = true })

	_, igpu := thisBox()
	slots := planSlots([]backend.Device{igpu}, nil, openOpts{host: boxHost, hostSet: true})
	g, err := New(slots, WithDeviceTune(TuneOff), noPrompt)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	ig := g.devs[0]
	if ig.pool != g.hostPool || g.hostPool == nil {
		t.Fatal("the integrated device is not on the host pool; this gate has nothing to read")
	}
	ceiling := g.hostPool.Limit()

	// Nothing placed: the host keeps its whole budget.
	if got := g.HostReserved(); got != 0 {
		t.Fatalf("HostReserved = %d (%.2f GiB) with no block on the integrated GPU, under a "+
			"%.2f GiB pool: the host is charged for room the device may grow into, not "+
			"for what it holds", got, gib(got), gib(ceiling))
	}

	// Each block is charged to the host as it lands.
	var held []uint64
	for li := 0; li < nblocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d) declined: %s", li, g.Err())
		}
		if per := g.Placed(); per[0] != li+1 {
			t.Fatalf("after block %d the placement is %v", li, per)
		}
		got := g.HostReserved()
		ig.mu.Lock()
		used := ig.used
		ig.mu.Unlock()
		if got != used || got != g.hostPool.Used() {
			t.Fatalf("with %d block(s) placed HostReserved = %d, the device's ledger %d and its "+
				"pool %d: the host must lose exactly what the device holds",
				li+1, got, used, g.hostPool.Used())
		}
		held = append(held, got)
	}
	t.Logf("integrated GPU under a %.2f GiB ceiling: host loses 0 with no block, then %v bytes "+
		"with 1..%d blocks", gib(ceiling), held, nblocks)
	for i := 1; i < len(held); i++ {
		if held[i] <= held[i-1] {
			t.Fatalf("placing block %d did not raise what the host loses (%v): the charge does "+
				"not move with placement", i, held)
		}
	}
	if held[nblocks-1] >= ceiling {
		t.Fatalf("%d small blocks charged %d, at or over the %d ceiling: this gate cannot tell "+
			"what the device holds from what it may hold", nblocks, held[nblocks-1], ceiling)
	}

	// Releasing gives it back: the last two blocks come home and the host
	// regains exactly what they held.
	g.ReleaseLayers(nblocks-2, nblocks)
	if got := g.HostReserved(); got != held[nblocks-3] {
		t.Fatalf("after releasing two of %d blocks HostReserved = %d, want %d (what two blocks "+
			"held): the charge did not follow the blocks home", nblocks, got, held[nblocks-3])
	}

	// No unified device: the host's budget is untouched, blocks or not.
	g2, err := New(planSlots([]backend.Device{&fakeDev{name: "RTX 3050 Ti", api: "ptx",
		free: boxCardFree, total: boxCardTotal}}, nil, openOpts{host: boxHost, hostSet: true}),
		WithDeviceTune(TuneOff), noPrompt)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g2.Close()
	if !g2.PrepLayer(0, p, blocks(p, 1)[0]) {
		t.Fatalf("the discrete card declined a block: %s", g2.Err())
	}
	if got := g2.HostReserved(); got != 0 {
		t.Fatalf("a discrete card holding a block reserved %d bytes of HOST memory", got)
	}
}

// TestOpenWithRefusesTwoAnswers: WithSlots and WithDevices both name the
// devices, so passing both is refused.
func TestOpenWithRefusesTwoAnswers(t *testing.T) {
	d := &fakeDev{}
	_, err := OpenWith(WithSlots(Slot{Dev: d, Bytes: 1 << 30}), WithDevices("cuda:0"))
	if err == nil {
		t.Fatal("OpenWith accepted both WithSlots and WithDevices")
	}
	if !strings.Contains(err.Error(), "cuda:0") {
		t.Errorf("refusal %q does not name what it got", err)
	}
	// OpenWith owns WithSlots devices and closes them on every error path,
	// matching New; see its doc comment.
	if d.closed != 1 {
		t.Errorf("the caller's device was closed %d times, want exactly 1: OpenWith "+
			"owns WithSlots devices and closes them on every error path", d.closed)
	}
}

// TestOpenWithSlotsTakesTheCallersDevices is the embedder's path: its own
// device and budget, no spec string, no environment.
func TestOpenWithSlotsTakesTheCallersDevices(t *testing.T) {
	d := &fakeDev{name: "embedder's device"}
	g, err := OpenWith(WithSlots(Slot{Dev: d, Bytes: 1 << 30, Why: "the embedder said so"}))
	if err != nil {
		t.Fatalf("OpenWith: %v", err)
	}
	defer g.Close()
	b := g.Budgets()
	if len(b) != 1 || b[0].Limit != 1<<30 || b[0].Why != "the embedder said so" {
		t.Fatalf("Budgets = %+v, want one 1 GiB device with the caller's reason", b)
	}
	var _ nn.LayerDevice = g
}

func gib(n uint64) float64 { return float64(n) / (1 << 30) }

// TestRealThreeWayPlacement is the non-exclusion gate on hardware: the host, a
// discrete card and an integrated GPU (for example `-devices
// cuda:0,vulkan:1`) all hold and run part of one model. The fakes prove the
// arithmetic; this proves two drivers open together. It skips only when the
// hardware is absent, and says which half.
func TestRealThreeWayPlacement(t *testing.T) {
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Skipf("no Vulkan on this host (%v), so there is no integrated device to reach", err)
	}
	sel := -1
	for _, in := range infos {
		if in.Unified && in.Compute && !in.Software {
			sel = in.Index
		}
	}
	if sel < 0 {
		t.Skipf("no integrated Vulkan device among %d: this box cannot show CPU + iGPU + "+
			"discrete, which is the configuration this gate is about", len(infos))
	}
	if n, err := backend.CUDACount(); err != nil || n == 0 {
		t.Skipf("no CUDA device (%d, %v), so the discrete half is missing", n, err)
	}

	p := fakePlan()
	const nblocks = 8
	ws := blocks(p, nblocks)
	// The budgets are blocks, counted on the fake device; a prompt's scratch
	// is the backend's own size and this gate places no prompt, so neither
	// arm reserves one (scratch.go).
	noPrompt := WithConfig(func(c *Config) { c.NoScratchReserve = true })
	// Sized on a fake that rounds as both drivers do a buffer this small
	// (backend's alloccount.go): every buffer of these blocks is under 1 MiB.
	two := twoBlockBytesOn(t, &fakeDev{page: 64 << 10}, p, noPrompt)

	spec := fmt.Sprintf("cuda:0=%d,vulkan:%d=%d", two, sel, two)
	g, err := OpenWith(WithDevices(spec), WithHostBudget(boxHost), noPrompt)
	if err != nil {
		t.Fatalf("OpenWith(%q): %v", spec, err)
	}
	defer g.Close()
	if g.Devices() != 2 {
		t.Fatalf("%q opened %d device(s): %s", spec, g.Devices(), g.Name())
	}
	for _, b := range g.Budgets() {
		t.Logf("%-52s %6.2f MiB  pool %q %.2f GiB host=%v",
			b.Device, float64(b.Limit)/(1<<20), b.Pool, gib(b.PoolLimit), b.Host)
	}
	for li := 0; li < nblocks; li++ {
		g.PrepLayer(li, p, ws[li])
	}
	per := g.Placed()
	onHost := nblocks
	for _, n := range per {
		onHost -= n
	}
	t.Logf("placement %v: discrete %d, integrated %d, host %d", placement(g, nblocks),
		per[0], per[1], onHost)
	if per[1] > 0 && g.HostReserved() == 0 {
		t.Fatal("the integrated device holds blocks and reserved nothing from the host's " +
			"budget: its memory IS the host's, so the host has to lose what it holds")
	}
	for i, n := range per {
		if n == 0 {
			t.Fatalf("%s took NO blocks (%v): an integrated GPU's execution units are real "+
				"even though its memory is not extra", g.DevStats()[i].Device, per)
		}
	}
	if onHost <= 0 {
		t.Fatalf("every block went to a device (%v); the host is the third member of this split", per)
	}
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	if !g.Layers(0, per[0]+per[1], 0, 1, x, cs, nil, nil) {
		t.Fatalf("Layers across both devices: %s", g.Err())
	}
	for i, st := range g.DevStats() {
		if st.Blocks == 0 {
			t.Fatalf("%s holds %d block(s) and ran none", st.Device, per[i])
		}
		t.Logf("%-52s %d placed, %d run", st.Device, per[i], st.Blocks)
	}
}

// TestSameDeviceNamedTwiceIsRefused: `-devices cuda:0,cuda:0` would give each
// context the whole card. The pool rule cannot catch it (two entries with the
// same api look like two cards), so the spec parser refuses it.
func TestSameDeviceNamedTwiceIsRefused(t *testing.T) {
	a := &fakeDev{name: "RTX 3050 Ti", api: "ptx", free: boxCardFree, total: boxCardTotal}
	b := &fakeDev{name: "RTX 3050 Ti", api: "ptx", free: boxCardFree, total: boxCardTotal}
	err := noDuplicates([]backend.Device{a, b}, "cuda:0,cuda:0")
	if err == nil {
		t.Fatal("the same device named twice was accepted: each context would be given the " +
			"whole card")
	}
	if !strings.Contains(err.Error(), "twice") || !strings.Contains(err.Error(), "RTX 3050 Ti") {
		t.Errorf("refusal %q does not name the device it got twice", err)
	}
	// The same model through two APIs is not a duplicate: it is one card,
	// which the pool handles, or two cards.
	c := &fakeDev{name: "RTX 3050 Ti", api: "spirv", free: boxCardFree, total: boxCardTotal}
	if err := noDuplicates([]backend.Device{a, c}, "cuda:0,vulkan:0"); err != nil {
		t.Errorf("one card through two APIs was refused as a duplicate: %v", err)
	}
}

// TestUnifiedDeviceGetsTheSharedPool: on unified memory the device and the
// host are one pool, so by default the device may hold the host budget less
// its headroom -- a model that fits the host budget goes on the device whole.
// The figures are a 16 GiB M4's (a 7.68 GiB host budget, Metal's 10.67 GiB
// working set), where the quarter that stood in hostShare gave 1.92 GiB and
// left the last block and the head of Ministral-3-3B Q4_K_M (2.26 GiB) on the
// CPU. A discrete card beside it keeps budgetFor's figure, untouched.
func TestUnifiedDeviceGetsTheSharedPool(t *testing.T) {
	const (
		host    = 7*(1<<30) + 696<<20 // 7.68 GiB
		working = 10*(1<<30) + 686<<20
		model   = 2*(1<<30) + 266<<20 // 2.26 GiB: Ministral-3-3B Q4_K_M with its head
	)
	m4 := &fakeDev{name: "Apple M4", api: "msl", unified: true, free: working, total: working}
	card, _ := thisBox()
	slots := planSlots([]backend.Device{card, m4}, nil, openOpts{host: host, hostSet: true})

	got := slots[1].Bytes
	t.Logf("host budget %.2f GiB: unified device %.2f GiB (%s), host keeps %.2f GiB when it is full",
		gib(host), gib(got), slots[1].Why, gib(host-got))
	if got < model {
		t.Fatalf("the unified device may hold %.2f GiB of a %.2f GiB host budget: a %.2f GiB "+
			"model that fits the shared pool is split onto the CPU", gib(got), gib(host), gib(model))
	}
	if got >= host || host-got < 256<<20 {
		t.Fatalf("the unified device may hold %.2f GiB of a %.2f GiB host budget: the host "+
			"keeps no headroom for its pager and scratch", gib(got), gib(host))
	}
	if slots[1].Pool.Limit() != got || !slots[1].Pool.Host() {
		t.Fatalf("the unified device's pool is %q at %.2f GiB, want the host pool at %.2f",
			slots[1].Pool.Name(), gib(slots[1].Pool.Limit()), gib(got))
	}
	// Discrete: unchanged by construction, budgetFor's figure on its own pool.
	want, _ := budgetFor(card, 0, "-vram")
	if slots[0].Bytes != want || slots[0].Pool.Host() {
		t.Fatalf("the discrete card got %d on pool %q, want budgetFor's %d on its own",
			slots[0].Bytes, slots[0].Pool.Name(), want)
	}
	// An explicit -vram still wins on both.
	slots = planSlots([]backend.Device{card, m4}, nil, openOpts{host: host, hostSet: true, budget: 1 << 30})
	if slots[0].Bytes != 1<<30 || slots[1].Bytes != 1<<30 {
		t.Fatalf("-vram 1G gave %d and %d", slots[0].Bytes, slots[1].Bytes)
	}
}
