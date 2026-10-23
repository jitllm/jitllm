package tier

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestKnobsArePerTier: two tiers in one process, one pinned and one on the
// defaults, opened in both orders. Each must keep its own knobs, batched
// geometry and centering whichever opened last: a package variable written
// by every open fails this.
func TestKnobsArePerTier(t *testing.T) {
	pinned := []Option{WithDeviceTune(TuneOff), WithSplit(4), WithLanes(ir.SubgroupLanes),
		WithBatchShape(BatchShape{RowT: 2, QTile: 2, AttnNT: 1, Parts: 32}), WithMMATile(2, 1),
		WithCentering(false, 1), WithMLAFault(MLAFaultSheetOff), WithSkip("attn"),
		WithPoisonScratch(true), WithMatVecAccSplit(3)}
	plain := []Option{WithDeviceTune(TuneOff)}
	open := func(opts []Option) (*devTier, *fakeDev) {
		dev := &fakeDev{name: "fake"}
		g, err := New([]Slot{{Dev: dev, Bytes: 1 << 30}}, opts...)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(g.Close)
		return g.devs[0], dev
	}
	// ops is the size of the decode matvec the tier emits: centering adds
	// two instructions per weight word, so the two tiers must differ.
	ops := func(d *devTier, dev *fakeDev) int {
		if d.kernelMode(kernels.Q4_0, 256, 64, 1, 1, false, false) == nil {
			t.Fatalf("kernelMode: %s", d.LastErr)
		}
		return len(dev.irs[len(dev.irs)-1].Ops)
	}
	for _, pinnedFirst := range []bool{true, false} {
		var p, q *devTier
		var pdev, qdev *fakeDev
		if pinnedFirst {
			p, pdev = open(pinned)
			q, qdev = open(plain)
		} else {
			q, qdev = open(plain)
			p, pdev = open(pinned)
		}
		if s := p.tuneSplit(kernels.Q4_0, 4096, 4096, nil); s != 4 {
			t.Errorf("pinned first=%v: pinned tier split %d, want its WithSplit(4)", pinnedFirst, s)
		}
		if s := q.tuneSplit(kernels.Q4_0, 4096, 4096, nil); s != chooseSplit(4096, 128, 0) {
			t.Errorf("pinned first=%v: default tier split %d, want the table's %d",
				pinnedFirst, s, chooseSplit(4096, 128, 0))
		}
		if l := p.pickLanes(); l != ir.SubgroupLanes {
			t.Errorf("pinned first=%v: pinned tier lanes %d, want its WithLanes(32)", pinnedFirst, l)
		}
		if l := q.pickLanes(); l != 1 {
			t.Errorf("pinned first=%v: default tier lanes %d, want the fake's 1", pinnedFirst, l)
		}
		if mt, nt, _ := p.mmaTile(128, 64); mt != 2 || nt != 1 {
			t.Errorf("pinned first=%v: pinned tier MMA tile %dx%d, want 2x1", pinnedFirst, mt, nt)
		}
		if mt, nt, _ := q.mmaTile(128, 64); mt != defMMAMT || nt != defMMANT {
			t.Errorf("pinned first=%v: default tier MMA tile %dx%d, want %dx%d",
				pinnedFirst, mt, nt, defMMAMT, defMMANT)
		}
		want := tiles{rowt: 2, mmaMT: 2, mmaNT: 1, qtile: 2, ktile: defAttnKTile, atile: defAttnATile,
			attnNT: 1, parts: 32}
		if p.tiles != want {
			t.Errorf("pinned first=%v: pinned tier tiles %+v, want %+v", pinnedFirst, p.tiles, want)
		}
		if q.tiles != tilesFor(knobs{}) {
			t.Errorf("pinned first=%v: default tier tiles %+v, want the defaults", pinnedFirst, q.tiles)
		}
		if a, b := ops(p, pdev), ops(q, qdev); a == b {
			t.Errorf("pinned first=%v: both tiers emit a %d-op decode matvec; centering at "+
				"MinTok 1 must change the pinned one", pinnedFirst, a)
		}
		if !p.kb.skip["attn"] || q.kb.skip["attn"] || p.kb.mlaFault != MLAFaultSheetOff ||
			q.kb.mlaFault != MLAFaultNone || !p.kb.poisonScratch || q.kb.poisonScratch ||
			accSplitFor(64, 0, p.kb.mvAccSplit) != 3 || accSplitFor(64, 0, q.kb.mvAccSplit) == 3 {
			t.Errorf("pinned first=%v: a knob leaked between tiers: pinned %+v, default %+v",
				pinnedFirst, p.kb, q.kb)
		}
	}
}
