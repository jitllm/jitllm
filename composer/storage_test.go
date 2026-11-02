package composer_test

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestPlanStatesEverySource holds a decode step's plan on the GPU to the
// buffer plan the engine fills: every storage has a source, the step's input
// is its tokens -- one storage, and no input row for the engine to fill -- a
// weight names its tensor, every storage the plan sizes has a size, and
// every source a slice without attention reads appears.
func TestPlanStatesEverySource(t *testing.T) {
	f := newSynth(t, jlm.TypeQ8)
	d := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	plan, err := composer.Default().Compose(f, composer.Step{Rows: 1, Page: 64, Keys: 5}, func(composer.Part) composer.Device { return d })
	if err != nil {
		t.Fatal(err)
	}
	seen := map[composer.Source]int{}
	for _, st := range plan.Storages {
		seen[st.Source]++
		if !st.Source.Engine() && st.Source != composer.SourceTransfer && st.Bytes == 0 {
			t.Errorf("%s storage %q of part %d states no size", st.Source, st.Key.Name, st.Key.Of)
		}
		if st.Source == composer.SourceWeight {
			if _, ok := f.Find(st.Role, int32(st.Block), -1); !ok {
				t.Errorf("weight %v of block %d is not in the container", st.Role, st.Block)
			}
		}
	}
	if seen[composer.SourceTokens] != 1 || seen[composer.SourceInput] != 0 {
		t.Errorf("%d token storages and %d input rows, want one and none", seen[composer.SourceTokens], seen[composer.SourceInput])
	}
	for _, s := range []composer.Source{composer.SourceStep, composer.SourceWeight, composer.SourceTokens} {
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
