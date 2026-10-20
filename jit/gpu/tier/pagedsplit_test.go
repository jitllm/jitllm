package tier

import (
	"testing"

	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestAStagedPlanThatDoesNotFitHalvesItsSplits builds a staged plan for a
// 512-row MLA chunk (Kimi-Linear's shape) on a card that refuses the partials
// the default split count wants, and demands a plan anyway, at fewer splits.
// Without the step-down the build refuses and the whole prompt goes a row at a
// time, which is far slower.
func TestAStagedPlanThatDoesNotFitHalvesItsSplits(t *testing.T) {
	g, d, _ := fakeTier(t)
	g.StagedDecode = true
	shape := mainkernels.FlashShape{Heads: 32, KVHeads: 1, Dim: 576, MLA: 512, Rows: 512, Page: 256}
	bs := &blockScratch{pkv: &pagedScratch{shape: shape, vars: map[mainkernels.DecodePlan]*pagedVariant{}}}
	const keys = 544
	want := mainkernels.PagedStagedPlan(shape, keys, g.dev.Slots())
	if want.Splits < 2 {
		t.Fatalf("the default plan has %d split(s): nothing to step down from, so this proves nothing", want.Splits)
	}
	// Room for the partials at half the default splits, not at the default.
	d.allocMax = mainkernels.FlashPartialFloats(mergeShape(mainkernels.PagedStagedAt(shape, keys, want.Splits/2))) * 4
	v, err := g.pagedPlanFor(bs, keys)
	if err != nil {
		t.Fatalf("a %d-split plan did not fit and no narrower one was tried: %v", want.Splits, err)
	}
	if got := v.plan.Shape.Splits; got >= want.Splits || got < 1 {
		t.Fatalf("built at %d split(s); the default is %d and does not fit", got, want.Splits)
	}
	t.Logf("default %d splits refused; built at %d", want.Splits, v.plan.Shape.Splits)
}
