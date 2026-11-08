package session

import (
	"strings"
	"testing"
)

// An unmeasured wall is omitted, never guessed: MemWall is 0 until the
// hardware probe has measured it.
func TestSessionRateLineOmitsAnUnmeasuredWall(t *testing.T) {
	got := SessionRateLine(120, 31.04, 816010912, 0)

	if strings.Contains(got, "%") {
		t.Fatalf("a percentage was quoted against an unmeasured wall: %q", got)
	}
	if strings.Contains(got, "wall") {
		t.Fatalf("the wall was named without having been measured: %q", got)
	}
	// The two terms that need no wall are still there.
	for _, want := range []string{"31.04 tok/s", "GB/s"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}
}

// With a wall, all three terms are quoted so the line can be checked with a
// calculator: 31.04 tok/s x 816,010,912 B = 25.3 GB/s = 69% of 36.63 GB/s.
func TestSessionRateLineQuotesRateBandwidthAndWall(t *testing.T) {
	got := SessionRateLine(120, 31.04, 816010912, 36.63e9)

	for _, want := range []string{"31.04 tok/s", "25.3 GB/s", "69%", "36.6 GB/s"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %q", want, got)
		}
	}
}

// Before the first generation there is no rate, and a dash is the honest
// rendering: "0.00 tok/s" and "not measured yet" are different facts.
func TestSessionRateLineBeforeAnyGeneration(t *testing.T) {
	got := SessionRateLine(0, 0, 0, 0)
	if strings.Contains(got, "GB/s") || strings.Contains(got, "/token") {
		t.Fatalf("a bandwidth was quoted with no rate behind it: %q", got)
	}
	if !strings.Contains(got, "--") {
		t.Fatalf("want a dash for an unmeasured rate, got %q", got)
	}
}

// A paging model is said to be paging, in words; the counters are on the
// Machine
func TestSessionPagerLineSaysAModelIsPaging(t *testing.T) {
	got := SessionPagerLine(PagerStat{
		Frames: 16, Resident: 6_291_456, PageIns: 528, PageOuts: 512, TurnOuts: 32,
		BytesRead: 5 << 30, Reads: 24791, ChunkBytes: 64 << 10,
	})
	for _, want := range []string{"paging from disk", "16 layer(s) fit in memory", "5.00 GiB"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %q", want, got)
		}
	}
	for _, not := range []string{"request", "chunk", "page(s)"} {
		if strings.Contains(got, not) {
			t.Errorf("the session line still quotes %q: %q", not, got)
		}
	}
}

// A model that fits has nothing to report, and says nothing.
func TestSessionPagerLineIsSilentForAModelThatFits(t *testing.T) {
	if got := SessionPagerLine(PagerStat{PageIns: 16, BytesRead: 598 << 20, Reads: 17}); got != "" {
		t.Fatalf("a model that pages nothing out reads %q", got)
	}
	// Nor one that paged once, long ago: PageOuts only grows.
	if got := SessionPagerLine(PagerStat{Frames: 5, PageOuts: 44, BytesRead: 1 << 30}); got != "" {
		t.Fatalf("a model that evicted nothing this turn reads %q", got)
	}
}

// A device row must report bytes against a capacity, and must never imply that
// two devices sharing a heap are two pools.
func TestADeviceRowIsBytesAgainstACapacity(t *testing.T) {
	got := DeviceUseLine(DeviceUse{
		Name: "0:NVIDIA GeForce RTX 3050 Ti [cuda]", Blocks: 34,
		Used: 1_610_612_736, Limit: 2_147_483_648, Pool: "cuda:0", KV: 134_217_728,
	})
	for _, want := range []string{"34 block(s)", "1.50 GiB", "2.00 GiB", "128.0 MiB kv"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}
	// The row's label is the name; the caption beside it repeated it.
	if strings.Contains(got, "RTX 3050 Ti") {
		t.Errorf("the caption repeats the device name its label shows: %q", got)
	}
	if strings.Contains(got, "frame") {
		t.Errorf("the row still counts frames: %q", got)
	}
}

// A device whose memory is the host's must say so on the row.
func TestASharedDeviceSaysItIsNotASecondPool(t *testing.T) {
	got := DeviceUseLine(DeviceUse{
		Name: "1:Intel Iris Xe [vulkan]", Blocks: 13,
		Used: 8 << 30, Limit: 20 << 30, Pool: "host", Shared: true,
	})
	if !strings.Contains(got, "not a second pool") {
		t.Errorf("a unified-memory device is drawn as its own pool: %q", got)
	}
}

// The host row carries the page bytes against the budget, and names the dense
// weights separately rather than folding them in, since they never page.
func TestTheHostRowSeparatesWhatNeverPages(t *testing.T) {
	got := HostLine(Allocation{
		NBlocks: 48, CPUBlocks: 44, DeviceBlocks: 4,
		HostUsed: 6 << 30, HostBudget: 8 << 30, Dense: 182_100_000,
	})
	for _, want := range []string{"44 block(s)", "6.00 GiB", "8.00 GiB", "dense"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}

	// No budget is "no cap", not a full bar.
	if got := HostLine(Allocation{NBlocks: 48, HostUsed: 1 << 30}); !strings.Contains(got, "no cap") {
		t.Errorf("an uncapped host reads %q", got)
	}
	if got := HostLine(Allocation{}); !strings.Contains(got, "no model") {
		t.Errorf("with nothing loaded the row reads %q", got)
	}
}

func TestPlacementLineCountsEachTierSeparately(t *testing.T) {
	r := byte(PlaceResident)
	blocks := []byte{
		byte(OnDevice(0)), byte(OnDevice(0)), byte(OnDevice(1)) | r,
		byte(PlaceHost) | r, byte(PlaceHost) | r,
		byte(PlaceHost),
	}
	got := PlacementLine(blocks)
	for _, want := range []string{"3 of 6 block(s) on a device", "2 on the host", "1 paged out"} {
		if !strings.Contains(got, want) {
			t.Errorf("PlacementLine = %q, want it to contain %q", got, want)
		}
	}
}

func TestPlacementLineSaysNothingIsLoadedRatherThanZeroOfZero(t *testing.T) {
	got := PlacementLine(nil)
	if strings.Contains(got, "0 of 0") {
		t.Errorf("PlacementLine(nil) = %q, want a sentence and not a row of zeroes", got)
	}
}

func TestPagerLineDistinguishesNoPagingFromNoData(t *testing.T) {
	quiet := PagerLine(PagerStat{})
	if !strings.Contains(quiet, "budget holds every block") {
		t.Errorf("PagerLine(zero) = %q, want the no-eviction sentence", quiet)
	}

	busy := PagerLine(PagerStat{Frames: 16, Resident: 6_291_456, PageIns: 528, PageOuts: 512,
		BytesRead: 6 << 30, Reads: 24791})
	// The request count is beside the bytes: the pager's cost is per request.
	for _, want := range []string{"16 page(s)", "6.00 GiB read", "24791 request(s)"} {
		if !strings.Contains(busy, want) {
			t.Errorf("PagerLine = %q, want it to contain %q", busy, want)
		}
	}
}
