package jit

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestPlanStatesEverySource holds a decode step's plan on the GPU to the
// buffer plan the engine fills: every storage has a source, the step's input
// is one storage, a weight names its tensor, every storage the plan sizes
// has a size, and every source a decode reads appears.
func TestPlanStatesEverySource(t *testing.T) {
	f, err := jlm.Open(testmodels.Path("synth-ernie45.jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	d := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	plan, err := Default().Compose(f, Step{Rows: 1, Page: 64, Keys: 5}, func(Part) Device { return d })
	if err != nil {
		t.Fatal(err)
	}
	seen := map[Source]int{}
	for _, st := range plan.Storages {
		seen[st.Source]++
		if !st.Source.Engine() && st.Source != SourceTransfer && st.Bytes == 0 {
			t.Errorf("%s storage %q of part %d states no size", st.Source, st.Key.Name, st.Key.Of)
		}
		if st.Source == SourceWeight {
			if _, ok := f.Find(st.Role, int32(st.Block), -1); !ok {
				t.Errorf("weight %v of block %d is not in the container", st.Role, st.Block)
			}
		}
	}
	if seen[SourceInput] != 1 {
		t.Errorf("%d input storages, want one", seen[SourceInput])
	}
	for _, s := range []Source{SourceStep, SourceWeight, SourceInput, SourceRows, SourcePages, SourceQueryOffset,
		SourceRopePositions, SourceRopePlanes, SourceKeys, SourceValues, SourceWords} {
		if seen[s] == 0 {
			t.Errorf("no %s storage in a decode step's plan", s)
		}
	}
	for _, pp := range plan.Parts {
		for _, c := range pp.Codes {
			for _, b := range c.Binds {
				if b.Storage < 0 || b.Storage >= len(plan.Storages) {
					t.Fatalf("%s binds storage %d of %d", c.Name, b.Storage, len(plan.Storages))
				}
			}
		}
	}
	t.Logf("%d storages: %v", len(plan.Storages), seen)
}
