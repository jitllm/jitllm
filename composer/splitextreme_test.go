package composer_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
)

// TestAStepsArgmaxIsCutByTheCallsRows forms a vocabulary's greedy pick at a
// step of 512 rows on a GPU: its segments are cut at launch by the call's
// rows, a call of two rows taking many segments a row, the launch never past
// the capacity's and the planes the capacity's. The violation: a split cut
// for the capacity alone runs a call of two rows in its capacity's few
// segments a row; one cut for a call's row at every count of rows holds
// planes for a row's segments at every row of the capacity.
func TestAStepsArgmaxIsCutByTheCallsRows(t *testing.T) {
	const capacity, vocab = 512, 151936
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 163840, Lanes: 32, Units: 80, Shared: 96 << 10, Regs: 65536}
	g := (&graph.Sample{NVocab: vocab}).Build()
	_, codes, err := composer.Default().Form(composer.Request{Graph: g, Step: composer.Step{Rows: capacity, Page: 64, Keys: 1}, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	var part *composer.Code
	for i, c := range codes {
		t.Logf("%s: passes %v, cut %+v, groups %d", c.Name, c.Passes, c.Cut, c.Groups)
		if slices.Contains(c.Passes, "splitextreme") && c.Cut.Most > 1 {
			part = &codes[i]
		}
	}
	if part == nil {
		t.Fatal("no part of the pick is cut by the call's rows")
	}
	segs := func(rows int) int { return part.Cut.Of(rows) * part.LiveInner }
	for _, rows := range []int{1, 2, 4, 8, 64, capacity} {
		t.Logf("%d rows: %d segments a row, %d workgroups of %d", rows, segs(rows), part.LiveGroups(rows), part.GPU.Group[0])
		if part.LiveGroups(rows) > part.Groups {
			t.Errorf("%d rows launch %d workgroups, past the capacity's %d", rows, part.LiveGroups(rows), part.Groups)
		}
	}
	if s := segs(2); s < 256 {
		t.Errorf("a call of two rows runs a row in %d segments, want at least 256", s)
	}
	if part.LiveGroups(capacity) != part.Groups {
		t.Errorf("the capacity launches %d workgroups, its form %d", part.LiveGroups(capacity), part.Groups)
	}
	planes := 0
	for _, b := range part.Binds {
		if b.Operand.Kind == composer.OperandScratch && strings.HasSuffix(b.Operand.Name, ".planes") {
			planes++
			if most := int64(4 * capacity * part.LiveInner); b.Bytes > most {
				t.Errorf("%s holds %d bytes, past the capacity's %d", b.Operand.Name, b.Bytes, most)
			}
		}
	}
	if planes == 0 {
		t.Fatal("the cut part binds no planes")
	}
}
