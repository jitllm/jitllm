package server

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
)

// One physical device enumerated by two backends is one budget: a card seen
// as vulkan:0 and cuda:0 must not be counted twice. See
// docs/engineering-history/placement.md 15u.

// oneCard is the identity both backends report for one discrete card
// (VkPhysicalDeviceIDProperties.deviceUUID and cuDeviceGetUuid agree byte for
// byte, as both specifications promise).
const (
	oneCard   = "uuid:5f3c1a2e9b7d4e6f8a0b1c2d3e4f5a6b"
	otherCard = "uuid:ffeeddccbbaa99887766554433221100"
	sameModel = "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
)

// thisBoxDevices is a real-shaped enumeration of a host with one discrete card
// seen by two backends and an integrated GPU, the two discrete entries
// carrying the identity their drivers report. Vulkan is FIRST, as
// probeDevices enumerates it.
func thisBoxDevices() []DeviceInfo {
	return []DeviceInfo{
		{ID: HostGateID, Backend: "cpu", Kind: KindHost, TotalMemory: 31 << 30,
			CountsTowardHostBudget: true, Available: true},
		{ID: "vulkan:0", Backend: "vulkan", Name: sameModel, Kind: KindDiscrete,
			TotalMemory: 4 << 30, PhysicalID: oneCard, Available: true},
		{ID: "vulkan:1", Backend: "vulkan", Name: "Intel Iris Xe", Kind: KindIntegrated,
			TotalMemory: 21 << 30, CountsTowardHostBudget: true, Available: true},
		{ID: "cuda:0", Backend: "cuda", Name: "#0 " + sameModel + " (sm_86)",
			Kind: KindDiscrete, TotalMemory: 4087087104, PhysicalID: oneCard, Available: true},
	}
}

// TestOnePhysicalDeviceIsSpendableOnce.
//
// VIOLATION SIGNATURE. Remove the `d.SameDeviceAs != ""` skip from
// SpendableTotal and this fails with the card's memory counted twice.
func TestOnePhysicalDeviceIsSpendableOnce(t *testing.T) {
	host := HostInfo{WeightBudget: 15 << 30}
	devs := thisBoxDevices()

	// Assert the link count, not only the total, so "never ran" cannot pass.
	if n := markSameDevice(devs); n != 1 {
		t.Fatalf("markSameDevice linked %d entries, want 1: vulkan:0 and cuda:0 are one "+
			"RTX 3050 Ti and they report the same UUID", n)
	}
	// The one left counting is CUDA, though Vulkan was listed first.
	if devs[1].SameDeviceAs != "cuda:0" {
		t.Errorf("vulkan:0.same_device_as = %q, want cuda:0: CUDA is the preferred backend "+
			"for an NVIDIA card, and a first-wins rule would keep the Vulkan entry",
			devs[1].SameDeviceAs)
	}
	if devs[3].SameDeviceAs != "" {
		t.Errorf("cuda:0 was marked as a duplicate of %q: the preferred backend is the one "+
			"that counts", devs[3].SameDeviceAs)
	}
	// The duplicate is still listed: `-devices vulkan:0` is a legitimate
	// request.
	if len(devs) != 4 {
		t.Fatalf("the list lost an entry: %d devices, want 4", len(devs))
	}

	// 15 GiB of host budget plus the CUDA entry, and neither the Vulkan view
	// of the same card nor the integrated GPU, whose memory is the host's.
	want := uint64(15<<30) + 4087087104
	if got := SpendableTotal(host, devs); got != want {
		t.Fatalf("spendable %s (%d B), want %s (%d B): %s of host budget + %s of the one "+
			"card -- vulkan:0 and cuda:0 are the same %s of VRAM",
			humanBytes(got), got, humanBytes(want), want, humanBytes(host.WeightBudget),
			humanBytes(4087087104), humanBytes(4<<30))
	}
}

// TestTwoIdenticalCardsAreBothSpendable is the trap, and the one a host with a
// single card cannot produce: two DIFFERENT cards reporting the SAME name.
//
// VIOLATION SIGNATURE. Key markSameDevice on d.Name instead of d.PhysicalID
// and this fails with
//
//	spendable 23.00 GiB, want 27.00 GiB: two cards became one and 4.00 GiB of
//	real VRAM disappeared
func TestTwoIdenticalCardsAreBothSpendable(t *testing.T) {
	host := HostInfo{WeightBudget: 19 << 30}
	devs := []DeviceInfo{
		{ID: "cuda:0", Backend: "cuda", Name: "#0 " + sameModel + " (sm_86)",
			Kind: KindDiscrete, TotalMemory: 4 << 30, PhysicalID: oneCard, Available: true},
		{ID: "cuda:1", Backend: "cuda", Name: "#1 " + sameModel + " (sm_86)",
			Kind: KindDiscrete, TotalMemory: 4 << 30, PhysicalID: otherCard, Available: true},
		// The same two cards, as Vulkan sees them: one name, twice.
		{ID: "vulkan:0", Backend: "vulkan", Name: sameModel, Kind: KindDiscrete,
			TotalMemory: 4 << 30, PhysicalID: oneCard, Available: true},
		{ID: "vulkan:1", Backend: "vulkan", Name: sameModel, Kind: KindDiscrete,
			TotalMemory: 4 << 30, PhysicalID: otherCard, Available: true},
	}
	if n := markSameDevice(devs); n != 2 {
		t.Fatalf("markSameDevice linked %d entries, want 2: four entries, two cards", n)
	}
	want := uint64(19<<30 + 4<<30 + 4<<30)
	got := SpendableTotal(host, devs)
	if got != want {
		t.Fatalf("spendable %s, want %s: two cards became one and %s of real VRAM disappeared",
			humanBytes(got), humanBytes(want), humanBytes(4<<30))
	}
	for _, d := range devs {
		if d.Backend == "cuda" && d.SameDeviceAs != "" {
			t.Errorf("%s was marked a duplicate of %s: CUDA is the backend that counts",
				d.ID, d.SameDeviceAs)
		}
	}
}

// TestADeviceWithNoIdentityIsNotMerged: an empty identity matches nothing,
// including another empty one. Under-merging over-counts memory; over-merging
// deletes hardware, which is worse.
//
// VIOLATION SIGNATURE. Drop the `if id == "" { continue }` guard in
// markSameDevice and this fails with
//
//	markSameDevice linked 1 entries with NO identity to match on
func TestADeviceWithNoIdentityIsNotMerged(t *testing.T) {
	host := HostInfo{WeightBudget: 8 << 30}
	devs := []DeviceInfo{
		{ID: "metal:0", Backend: "metal", Name: sameModel, Kind: KindDiscrete,
			TotalMemory: 4 << 30, Available: true},
		{ID: "vulkan:0", Backend: "vulkan", Name: sameModel, Kind: KindDiscrete,
			TotalMemory: 4 << 30, Available: true},
	}
	if n := markSameDevice(devs); n != 0 {
		t.Fatalf("markSameDevice linked %d entries with NO identity to match on", n)
	}
	want := uint64(8<<30 + 4<<30 + 4<<30)
	if got := SpendableTotal(host, devs); got != want {
		t.Fatalf("spendable %s, want %s: two unidentifiable devices were merged", humanBytes(got), humanBytes(want))
	}
}

// TestTheRealProbeCountsEachCardOnce runs this host's actual enumeration:
// only the real probe shows whether the backends agree on a UUID and whether
// probeDevices fills PhysicalID at all.
func TestTheRealProbeCountsEachCardOnce(t *testing.T) {
	devs, err := probeDevices(os.Getenv("JITLLM_ROCM"), testdevices.Admit())
	if err != nil {
		t.Fatalf("probeDevices: %v", err)
	}
	var e Engine
	host := e.Host()
	counted := map[string]string{}
	var sum uint64
	identified, marked := 0, 0
	for _, d := range devs {
		t.Logf("%-10s %-42s kind=%d total=%12d id=%s same=%s",
			d.ID, d.Name, d.Kind, d.TotalMemory, d.PhysicalID, d.SameDeviceAs)
		if d.PhysicalID != "" {
			identified++
		}
		if d.SameDeviceAs != "" {
			marked++
			continue
		}
		if d.Kind == KindHost || d.CountsTowardHostBudget || !d.Available {
			continue
		}
		if prev, dup := counted[d.PhysicalID]; dup && d.PhysicalID != "" {
			t.Fatalf("%s and %s are the same physical device (%s) and BOTH are counted: "+
				"%s of VRAM spent twice", prev, d.ID, d.PhysicalID, humanBytes(d.TotalMemory))
		}
		counted[d.PhysicalID] = d.ID
		sum += d.TotalMemory
	}
	want := host.WeightBudget + sum
	got := SpendableTotal(host, devs)
	if got != want {
		t.Fatalf("spendable %d B, want %d B (%d host budget + %d device)",
			got, want, host.WeightBudget, sum)
	}
	t.Logf("%d of %d entries carry an identity, %d marked as a second view of one card",
		identified, len(devs), marked)
	t.Logf("spendable_total = %d B (%s)", got, humanBytes(got))
	gpus := 0
	for _, d := range devs {
		if d.Backend == "cuda" || d.Backend == "vulkan" {
			gpus++
		}
	}
	if gpus == 0 {
		// Metal states no identity (backend.mtlDev.Identity), so a host with
		// no CUDA or Vulkan device has nothing for the dedupe to read.
		t.Skipf("no CUDA or Vulkan device here (%d entries probed): the sum was checked, the identities were not", len(devs))
	}
	if identified == 0 {
		t.Error("no device on this host reported an identity: either there is no GPU here, " +
			"or both backends stopped answering and the dedupe is dead code")
	}
}

// TestThreeViewsOfOneCardAllPointAtTheOneThatCounts: no duplicate may point
// at another duplicate. The total is right either way, so this needs its own
// assertion.
func TestThreeViewsOfOneCardAllPointAtTheOneThatCounts(t *testing.T) {
	host := HostInfo{WeightBudget: 8 << 30}
	devs := []DeviceInfo{
		{ID: "vulkan:0", Backend: "vulkan", Kind: KindDiscrete, TotalMemory: 4 << 30,
			PhysicalID: oneCard, Available: true},
		{ID: "vulkan:3", Backend: "vulkan", Kind: KindDiscrete, TotalMemory: 4 << 30,
			PhysicalID: oneCard, Available: true},
		{ID: "cuda:0", Backend: "cuda", Kind: KindDiscrete, TotalMemory: 4 << 30,
			PhysicalID: oneCard, Available: true},
	}
	if n := markSameDevice(devs); n != 2 {
		t.Fatalf("markSameDevice linked %d entries, want 2: three views, one card", n)
	}
	for _, d := range devs[:2] {
		if d.SameDeviceAs != "cuda:0" {
			t.Errorf("%s points at %q, want cuda:0: a duplicate must name the entry that is "+
				"actually counted, not another duplicate", d.ID, d.SameDeviceAs)
		}
	}
	if want := uint64(8<<30 + 4<<30); SpendableTotal(host, devs) != want {
		t.Fatalf("spendable %s, want %s", humanBytes(SpendableTotal(host, devs)), humanBytes(want))
	}
}
