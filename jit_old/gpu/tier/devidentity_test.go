package tier

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// One physical device enumerated by two backends is one device. Counting a
// 4 GiB card once through Vulkan and once through CUDA budgets 7.81 GiB of
// VRAM on a machine with 4. The opposite mistake, two different cards
// reporting the same name, cannot occur on a one-card host, so it is
// fabricated here: keying identity on the name would merge them and lose half
// the machine's VRAM. See docs/engineering-history/placement.md 15u.

// card0 and card1 are two distinct pieces of hardware with the same name on
// purpose; only the identity separates them.
const (
	card0 = "5f3c1a2e9b7d4e6f8a0b1c2d3e4f5a6b"
	card1 = "ffeeddccbbaa99887766554433221100"
	model = "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
)

// TestOnePhysicalDeviceUnderTwoBackendsIsOfferedOnce: the two entries sharing a
// UUID collapse to one, on the preferred backend (CUDA), whatever the
// enumeration order, and their budgets share one pool.
func TestOnePhysicalDeviceUnderTwoBackendsIsOfferedOnce(t *testing.T) {
	vk := &fakeDev{name: model, api: "spirv", uuid: card0, free: boxCardFree, total: boxCardTotal}
	cu := &fakeDev{name: "#0 " + model + " (sm_86)", api: "ptx", uuid: card0,
		free: boxCardFree, total: boxCardTotal}

	// Vulkan first, the order server.probeDevices enumerates in, so a
	// first-wins rule would keep the wrong one.
	all := []backend.Device{vk, cu}
	keep, merges := preferOnePerCard(all)
	if len(keep) != 1 {
		apis := make([]string, len(keep))
		for i, j := range keep {
			apis[i] = all[j].API()
		}
		t.Fatalf("%d devices offered for 1 card (%v): the same %.2f GiB counted twice",
			len(keep), apis, gib(boxCardTotal))
	}
	// The merge record, not just the outcome: one device of two is also what
	// dropping an arbitrary device would give.
	if len(merges) != 1 {
		t.Fatalf("merges = %d, want 1: the count is how this gate tells a dedupe "+
			"from a device that went missing", len(merges))
	}
	if merges[0].ID != "uuid:"+card0 {
		t.Errorf("merged on %q, want the device UUID", merges[0].ID)
	}
	if all[keep[0]].API() != "ptx" || merges[0].KeptAPI != "ptx" || merges[0].DropAPI != "spirv" {
		t.Fatalf("kept %s and dropped %s: CUDA is the preferred backend for an NVIDIA card "+
			"(AGENTS.md: cuda:0 + cpu at +10.2%%, the Vulkan arm of the same configuration "+
			"at -14%%)", merges[0].KeptAPI, merges[0].DropAPI)
	}

	// The other enumeration order must give the same answer.
	rev := []backend.Device{cu, vk}
	keep2, merges2 := preferOnePerCard(rev)
	if len(keep2) != 1 || rev[keep2[0]].API() != "ptx" || len(merges2) != 1 {
		t.Fatalf("CUDA-first enumeration kept %d device(s) on %d merge(s): the answer must "+
			"not depend on enumeration order", len(keep2), len(merges2))
	}

	// Both entries remain valid targets for an explicit spec, so planSlots
	// must pool them.
	slots := planSlots([]backend.Device{vk, cu}, nil, openOpts{host: boxHost, hostSet: true})
	if slots[0].Pool != slots[1].Pool {
		t.Fatalf("two pools for one card: %d + %d bytes claimed against %d free",
			slots[0].Pool.Limit(), slots[1].Pool.Limit(), uint64(boxCardFree))
	}
	if got := slots[0].Pool.Limit(); got > boxCardFree {
		t.Fatalf("the shared pool allows %d bytes on a card with %d free", got, uint64(boxCardFree))
	}
}

// TestTwoIdenticalCardsAreNotMerged: two cards of the same model with
// different UUIDs stay two devices and two pools, on one backend or across
// two.
func TestTwoIdenticalCardsAreNotMerged(t *testing.T) {
	a := &fakeDev{name: model, api: "spirv", uuid: card0, free: boxCardFree, total: boxCardTotal}
	b := &fakeDev{name: model, api: "spirv", uuid: card1, free: boxCardFree, total: boxCardTotal}
	keep, merges := preferOnePerCard([]backend.Device{a, b})
	if len(keep) != 2 || len(merges) != 0 {
		t.Fatalf("two cards with one name became %d device(s) after %d merge(s): half the "+
			"machine's VRAM (%.2f GiB) would go unused", len(keep), len(merges), gib(boxCardTotal))
	}
	// Across backends too: identical cards, one on each API, is four entries
	// and two pieces of hardware.
	c := &fakeDev{name: "#0 " + model + " (sm_86)", api: "ptx", uuid: card0,
		free: boxCardFree, total: boxCardTotal}
	d := &fakeDev{name: "#1 " + model + " (sm_86)", api: "ptx", uuid: card1,
		free: boxCardFree, total: boxCardTotal}
	four := []backend.Device{a, b, c, d}
	keep, merges = preferOnePerCard(four)
	if len(keep) != 2 || len(merges) != 2 {
		t.Fatalf("four entries for two cards became %d device(s) after %d merge(s), want 2 and 2",
			len(keep), len(merges))
	}
	var kept []backend.Device
	for _, j := range keep {
		if four[j].API() != "ptx" {
			t.Errorf("kept a %s entry where CUDA holds the same card", four[j].API())
		}
		kept = append(kept, four[j])
	}
	slots := planSlots(kept, nil, openOpts{host: boxHost, hostSet: true})
	if slots[0].Pool == slots[1].Pool {
		t.Fatal("two cards share one pool: each has its own VRAM and the budgets do not " +
			"contend")
	}
	if want := uint64(2 * slots[0].Pool.Limit()); slots[0].Pool.Limit()+slots[1].Pool.Limit() != want {
		t.Fatalf("two cards of the same model are not two equal budgets: %d and %d",
			slots[0].Pool.Limit(), slots[1].Pool.Limit())
	}
}

// TestADeviceWithNoIdentityStaysDistinct: an unknown identity matches nothing,
// including another unknown one; otherwise every unidentifiable device would
// collapse into one.
func TestADeviceWithNoIdentityStaysDistinct(t *testing.T) {
	a := &fakeDev{name: model, api: "spirv", free: boxCardFree, total: boxCardTotal}
	b := &fakeDev{name: model, api: "ptx", free: boxCardFree, total: boxCardTotal}
	keep, merges := preferOnePerCard([]backend.Device{a, b})
	if len(keep) != 2 || len(merges) != 0 {
		t.Fatalf("two devices that report NO identity became %d after %d merge(s): "+
			"an unknown identity must never match", len(keep), len(merges))
	}
	if id := backend.IdentityOf(a); id.Known() || id.Source == "" {
		t.Errorf("identity %+v: an absent identity must be unknown AND say why", id)
	}
	// Only the pair that shares a UUID may merge.
	c := &fakeDev{name: model, api: "spirv", uuid: card0, free: boxCardFree, total: boxCardTotal}
	d := &fakeDev{name: model, api: "ptx", uuid: card0, free: boxCardFree, total: boxCardTotal}
	keep, merges = preferOnePerCard([]backend.Device{a, b, c, d})
	if len(keep) != 3 || len(merges) != 1 {
		t.Fatalf("got %d device(s) after %d merge(s), want 3 and 1", len(keep), len(merges))
	}
}

// TestAnExplicitAPIStillWinsOverThePreference: the dedupe governs the default
// set, never what a caller asks for. On the auto path, choose() sees both
// entries and WithAPI pins one; the preference must not override the pin.
func TestAnExplicitAPIStillWinsOverThePreference(t *testing.T) {
	vk := &fakeDev{name: model, api: "spirv", uuid: card0, free: boxCardFree, total: boxCardTotal}
	cu := &fakeDev{name: "#0 " + model + " (sm_86)", api: "ptx", uuid: card0,
		free: boxCardFree, total: boxCardTotal}

	got := choose([]backend.Device{vk, cu}, knobs{api: "spirv"})
	if got.API() != "spirv" {
		t.Fatalf("explicit spirv request got %s: the preference may decide the DEFAULT set "+
			"and never what the caller named", got.API())
	}
	if cu.closed == 0 {
		t.Error("the device that was not chosen was never closed: a driver context, a " +
			"locked OS thread and an owner goroutine leak with it")
	}

	// With no pin, the same pair yields the preferred backend without
	// probing: TuneOff proves the answer is the preference, not a timing.
	vk2 := &fakeDev{name: model, api: "spirv", uuid: card0, free: boxCardFree, total: boxCardTotal}
	cu2 := &fakeDev{name: "#0 " + model + " (sm_86)", api: "ptx", uuid: card0,
		free: boxCardFree, total: boxCardTotal}
	if got := choose([]backend.Device{vk2, cu2}, knobs{tune: TuneOff}); got.API() != "ptx" {
		t.Fatalf("auto chose %s for one card: with the tuner off the answer is the "+
			"preference, and the preference is CUDA", got.API())
	}
	if vk2.closed == 0 {
		t.Error("the duplicate Vulkan context was left open")
	}
}

// TestTheRealEnumerationCountsEachCardOnce runs the host's real hardware
// through the same rule; only the real enumeration shows whether Vulkan's
// deviceUUID and CUDA's cuDeviceGetUuid agree on this driver.
func TestTheRealEnumerationCountsEachCardOnce(t *testing.T) {
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) < 2 {
		t.Skipf("this host opens %d backend(s); the double-count needs two on one card", len(devs))
	}
	seen := map[string][]string{}
	for _, d := range devs {
		id := backend.IdentityOf(d)
		t.Logf("%-6s %-42s %s", d.API(), d.Name(), id)
		if !id.Known() {
			continue
		}
		seen[id.Key] = append(seen[id.Key], d.API())
	}
	keep, merges := preferOnePerCard(devs)
	dup := 0
	for _, apis := range seen {
		dup += len(apis) - 1
	}
	if len(merges) != dup {
		t.Fatalf("%d device(s) share an identity and %d were merged", dup, len(merges))
	}
	if len(keep) != len(devs)-dup {
		t.Fatalf("kept %d of %d devices with %d duplicate(s)", len(keep), len(devs), dup)
	}
	if dup == 0 {
		t.Log("no card on this host is reachable through two backends; the merge path ran " +
			"on nothing here and the fabricated cases above are what cover it")
	}
}

// TestAllDevicesOffersEachCardOnce runs the default set (`-devices all`)
// through the real enumeration. backend.Open returns one device per backend,
// while allDevices enumerates every ordinal and Vulkan index, so both lists
// are gated.
func TestAllDevicesOffersEachCardOnce(t *testing.T) {
	devs, err := opener{}.allDevices(backend.Opts{})
	if err != nil {
		t.Skipf("no GPU backend on this host: %v", err)
	}
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	seen := map[string]string{}
	for _, d := range devs {
		id := backend.IdentityOf(d)
		t.Logf("%-6s %-42s %s", d.API(), d.Name(), id)
		if !id.Known() {
			continue
		}
		if prev, dup := seen[id.Key]; dup {
			t.Fatalf("-devices all offers %s [%s] AND %s [%s] for one physical device (%s): "+
				"two contexts, two budgets, one heap", prev, d.API(), d.Name(), d.API(), id.Key)
		}
		seen[id.Key] = d.Name()
	}
	// Every survivor is on the preferred backend: a Vulkan entry left standing
	// must be a card CUDA cannot see.
	cu, err := backend.CUDACount()
	if err != nil {
		return
	}
	cuIDs := map[string]bool{}
	for i := 0; i < cu; i++ {
		if u, ok := cudaUUIDForTest(i); ok {
			cuIDs[u] = true
		}
	}
	for _, d := range devs {
		id := backend.IdentityOf(d)
		if d.API() == "spirv" && id.Known() && cuIDs[id.Key] {
			t.Fatalf("%s is offered through Vulkan while CUDA holds the same card (%s): "+
				"CUDA is the preferred backend", d.Name(), id.Key)
		}
	}
}

// TestAnExplicitVulkanSpecStillOpensVulkan runs the real spec path on a host
// where CUDA holds the same card. `-devices vulkan:0` goes straight to
// backend.OpenVulkan, not through choose(), so it is gated separately in case
// a dedupe is ever added one layer up.
func TestAnExplicitVulkanSpecStillOpensVulkan(t *testing.T) {
	infos, err := backend.VulkanDevices()
	if err != nil || len(infos) == 0 {
		t.Skipf("no Vulkan device here, so the CUDA-holds-the-same-card case cannot be run: %v", err)
	}
	idx := -1
	for _, in := range infos {
		if in.Compute && !in.Software {
			idx = in.Index
			break
		}
	}
	if idx < 0 {
		t.Skip("every Vulkan device here is a software rasteriser or has no compute queue")
	}
	spec := fmt.Sprintf("vulkan:%d", idx)
	es, err := ParseDevices(spec)
	if err != nil {
		t.Fatalf("ParseDevices(%q): %v", spec, err)
	}
	devs, _, _, err := openEntries(es, openOpts{})
	if err != nil {
		t.Fatalf("openEntries(%q): %v", spec, err)
	}
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) != 1 || devs[0].API() != "spirv" {
		got := "nothing"
		if len(devs) > 0 {
			got = devs[0].API()
		}
		t.Fatalf("-devices %s opened %d device(s), API %s: an explicit request outranks the "+
			"CUDA-over-Vulkan preference, which governs the DEFAULT set only", spec, len(devs), got)
	}
	t.Logf("-devices %s -> %s [%s] %s", spec, devs[0].Name(), devs[0].API(),
		backend.IdentityOf(devs[0]))
}
