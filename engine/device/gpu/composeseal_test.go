package gpu

import "testing"

// TestTheCardComposesNothingAtAStep is the composition rule made a property
// of the card rather than a count: once the block is placed, a part no
// placement formed -- block 0's prompt part at a row count no placement asked
// for -- is declined unformed at a step, the composer does not run, and the
// ask is counted (Stats.UnformedAtStep). Inside a placement the same request
// forms: that is the gate seeing the guard and not a form that cannot be made.
func TestTheCardComposesNothingAtAStep(t *testing.T) {
	g, _, p := fakeTier(t)
	g.mu.Lock()
	defer g.mu.Unlock()
	bb, l := g.bbs[g.promptW], g.layers[0]
	if bb == nil || l == nil || g.composer == nil {
		t.Fatalf("the placement left no prompt scratch (%d rows), block or composer: the gate would ask for nothing", g.promptW)
	}
	const rows = 3
	if rows >= bb.rows {
		t.Fatalf("the prompt scratch holds %d rows: %d is not a row count of its own", bb.rows, rows)
	}
	ran, asked := g.composer.Formed(), g.UnformedAtStep
	if sv, _, _, err := g.servedStep(bb, l, 0, p, rows, true); err == nil || sv != nil {
		t.Fatalf("a part asked at a step after placement formed (%v)", err)
	}
	if n := g.composer.Formed() - ran; n != 0 {
		t.Fatalf("the declined part ran the composer %d time(s)", n)
	}
	if g.UnformedAtStep != asked+1 {
		t.Fatalf("the declined part was counted %d time(s), want once", g.UnformedAtStep-asked)
	}
	t.Logf("after placement: %s", g.LastErr)
	g.prewarming = true
	sv, _, _, err := g.servedStep(bb, l, 0, p, rows, true)
	g.prewarming = false
	if err != nil || sv == nil {
		t.Fatalf("VIOLATION arm: inside a placement the same part did not form: %v", err)
	}
	if g.composer.Formed() == ran {
		t.Fatal("VIOLATION arm: the part formed inside a placement without running the composer")
	}
}
