package tier

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestDeclineSizeRefusesOnlyWhatNoDeviceHolds is the size check placement asks
// before reading a block: a block above every device's whole budget is refused
// by name, one any device could hold is not, and a device that streams the
// block (the whole tier, or this block by placement) never refuses here,
// because its bank is not kept resident.
func TestDeclineSizeRefusesOnlyWhatNoDeviceHolds(t *testing.T) {
	const gib = 1 << 30
	mk := func(lims ...uint64) *GPU {
		g := &GPU{Config: &Config{}}
		for _, l := range lims {
			g.devs = append(g.devs, newDevice(&devShared{Config: g.Config, limit: l}))
		}
		return g
	}
	g := mk(14*gib, 14*gib)
	if why := g.DeclineSize(3, 16*gib, 15*gib); !strings.Contains(why, "above every device's whole budget") {
		t.Fatalf("a 16 GiB block on two 14 GiB cards was not refused: %q", why)
	}
	if why := g.DeclineSize(3, 13*gib, 12*gib); why != "" {
		t.Fatalf("a 13 GiB block on a 14 GiB card was refused: %q", why)
	}
	if why := mk(4*gib, 14*gib).DeclineSize(3, 13*gib, 12*gib); why != "" {
		t.Fatalf("the second card holds it, and it was refused: %q", why)
	}
	g.StreamExperts = true
	if why := g.DeclineSize(3, 16*gib, 15*gib); why != "" {
		t.Fatalf("a streamed tier was refused a block it does not keep resident: %q", why)
	}
	g.StreamExperts = false
	g.Stream(3, true)
	if why := g.DeclineSize(3, 16*gib, 15*gib); why != "" {
		t.Fatalf("a block the placement streams was refused: %q", why)
	}
	if why := g.DeclineSize(4, 16*gib, 15*gib); why == "" {
		t.Fatal("marking block 3 streamed let block 4 through")
	}
	// A dense block (no routed bank) is refused for its size alone: the
	// reason names neither experts nor a streaming placement.
	dense := g.DeclineSize(4, 16*gib, 0)
	if !strings.Contains(dense, "above every device's whole budget") ||
		strings.Contains(dense, "expert") || strings.Contains(dense, "stream") {
		t.Fatalf("a dense block's reason: %q", dense)
	}
	if moe := g.DeclineSize(4, 16*gib, 15*gib); !strings.Contains(moe, "routed experts") {
		t.Fatalf("a mixture block's reason lost its experts: %q", moe)
	}
	if why := mk().DeclineSize(0, 16*gib, 0); why != "" {
		t.Fatalf("a tier with no devices refused: %q", why)
	}
}

// TestAutoStreamMarksOnlyWhatABaseFits: a block whose base fits a card is
// marked streamed on every device and then passes the size check; one whose
// base fits none is not; NoAutoStream turns it off.
func TestAutoStreamMarksOnlyWhatABaseFits(t *testing.T) {
	const gib = 1 << 30
	g := &GPU{Config: &Config{}}
	for range 2 {
		g.devs = append(g.devs, newDevice(&devShared{Config: g.Config, limit: 14 * gib}))
	}
	if !g.AutoStream(5, 16*gib, 15*gib, 896, 16, 93, 0) {
		t.Fatal("a 1 GiB base on 14 GiB cards was not streamed")
	}
	for i, d := range g.devs {
		if _, ok := d.autoStream[5]; !ok {
			t.Fatalf("device %d did not mark block 5", i)
		}
	}
	if why := g.DeclineSize(5, 16*gib, 15*gib); why != "" {
		t.Fatalf("an auto-streamed block is still refused by size: %q", why)
	}
	if g.AutoStream(6, 40*gib, 15*gib, 896, 16, 93, 0) {
		t.Fatal("a 25 GiB base was streamed onto 14 GiB cards")
	}
	if g.AutoStream(7, 16*gib, 0, 896, 16, 93, 0) {
		t.Fatal("a block with no bank was streamed")
	}
	g.NoAutoStream = true
	if g.AutoStream(8, 16*gib, 15*gib, 896, 16, 93, 0) {
		t.Fatal("NoAutoStream streamed a block")
	}
}

// TestBiasedExpertsKeepThePlainBank: a mixture whose experts carry biases
// (gpt-oss) gets no expert cache, because the bias kernels index the true ids
// the cache would overwrite with slots; on gpt-oss-20b that parted from the
// host at the thirteenth token.
func TestBiasedExpertsKeepThePlainBank(t *testing.T) {
	g := newDevice(&devShared{Config: &Config{}, limit: 1 << 40})
	if n := g.cacheSlotsFor(&nn.LayerPlan{}, nil, 64, 4, 32, true); n != 4 {
		t.Fatalf("a biased bank got a %d-sheet cache", n)
	}
}
