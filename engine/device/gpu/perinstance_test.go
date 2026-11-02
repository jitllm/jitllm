package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestKnobsArePerTier: two tiers in one process, one pinned and one on the
// defaults, opened in both orders. Each must keep its own knobs, batched
// geometry whichever opened last: a package variable written by every open
// fails this.
func TestKnobsArePerTier(t *testing.T) {
	pinned := []Option{WithDeviceTune(TuneOff), WithLanes(ir.SubgroupLanes),
		WithBatchShape(BatchShape{RowT: 2, QTile: 2, AttnNT: 1, Parts: 32}), WithMMATile(2, 1),
		WithSkip("attn"),
		WithPoisonScratch(true)}
	plain := []Option{WithDeviceTune(TuneOff)}
	open := func(opts []Option) *devTier {
		dev := &fakeDev{name: "fake"}
		g, err := New([]Slot{{Dev: dev, Bytes: 1 << 30}}, opts...)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(g.Close)
		return g.devs[0]
	}
	for _, pinnedFirst := range []bool{true, false} {
		var p, q *devTier
		if pinnedFirst {
			p = open(pinned)
			q = open(plain)
		} else {
			q = open(plain)
			p = open(pinned)
		}
		if l := p.pickLanes(); l != ir.SubgroupLanes {
			t.Errorf("pinned first=%v: pinned tier lanes %d, want its WithLanes(32)", pinnedFirst, l)
		}
		if l := q.pickLanes(); l != 1 {
			t.Errorf("pinned first=%v: default tier lanes %d, want the fake's 1", pinnedFirst, l)
		}
		want := tiles{rowt: 2, mmaMT: 2, mmaNT: 1, qtile: 2, ktile: defAttnKTile, atile: defAttnATile,
			attnNT: 1, parts: 32}
		if p.tiles != want {
			t.Errorf("pinned first=%v: pinned tier tiles %+v, want %+v", pinnedFirst, p.tiles, want)
		}
		if q.tiles != tilesFor(knobs{}) {
			t.Errorf("pinned first=%v: default tier tiles %+v, want the defaults", pinnedFirst, q.tiles)
		}
		if !p.kb.skip["attn"] || q.kb.skip["attn"] || !p.kb.poisonScratch || q.kb.poisonScratch {
			t.Errorf("pinned first=%v: a knob leaked between tiers: pinned %+v, default %+v",
				pinnedFirst, p.kb, q.kb)
		}
	}
}
