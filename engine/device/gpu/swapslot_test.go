package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/backend"
)

// slackCounter is an allocation counter whose pieces hold slack back.
type slackCounter struct {
	backend.AllocCounter
	slack uint64
}

func (c slackCounter) Slack() uint64 { return c.slack }

// TestASwapSlotLeavesTheSlack: swapRoom answers for a page-in, which room
// holds the card's slack back from, so a budget holding the widest streamed
// block but not the slack beside it has no slot to swap through. Counted
// without it, history took that room and the next page-in found nothing to
// page out (DeepSeek V4 HCA: 818 KB free for a 424 KB block and 448 KB of
// slack).
func TestASwapSlotLeavesTheSlack(t *testing.T) {
	const widest, slack = 1 << 20, 256 << 10
	g := newDevice(&devShared{Config: &Config{}, layers: map[int]*layer{}})
	g.cnt = slackCounter{AllocCounter: &fakeDev{}, slack: slack}
	g.layers[0] = &layer{ok: true, pg: &page{bytes: widest}}
	g.stream = map[int]bool{0: true}
	g.used = 8 << 20
	g.limit = g.used + widest + slack - 1
	if g.room(widest) {
		t.Fatal("a page-in of the widest block fits a budget a byte short of it and the slack: the gate proves nothing")
	}
	if g.swapRoom(0) {
		t.Fatalf("a slot to swap through with %d bytes free for a %d-byte block and %d of slack", g.limit-g.used, widest, slack)
	}
	g.limit++
	if !g.room(widest) || !g.swapRoom(0) {
		t.Fatalf("room %v and swapRoom %v with the block and the slack free: they must agree", g.room(widest), g.swapRoom(0))
	}
}
