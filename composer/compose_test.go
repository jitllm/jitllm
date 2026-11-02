package composer_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestEveryKernelRecordsItsPasses composes the fixture on each device and
// holds every kernel formed to naming the passes that formed it.
func TestEveryKernelRecordsItsPasses(t *testing.T) {
	f := newSynth(t, jlm.TypeF32)
	r := composer.Default()
	s := composer.Step{Rows: 1, Page: 64, Keys: 5}
	for _, d := range []composer.Device{{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: composer.DeviceCPU, Slots: 6, Lanes: 8, ExactDiv: true}} {
		plan, err := r.Compose(f, s, func(composer.Part) composer.Device { return d })
		if err != nil {
			t.Fatal(err)
		}
		n := 0
		for _, pp := range plan.Parts {
			for _, c := range pp.Codes {
				n++
				if len(c.Passes) == 0 {
					t.Errorf("device %d: %s %s records no pass", d.Kind, pp.Part, c.Name)
				}
			}
		}
		if n == 0 {
			t.Errorf("device %d formed no kernel", d.Kind)
		}
		t.Logf("device %d: %d kernels, each naming its passes", d.Kind, n)
	}
}

// runProbe records, for each call, the parts whose kernels it was given.
type runProbe struct{ calls *[][]int }

func (runProbe) Name() string { return "runprobe" }

func (p runProbe) Apply(ks []*composer.Kernel, d composer.Device) ([]*composer.Kernel, error) {
	seen := map[int]bool{}
	var of []int
	for _, k := range ks {
		if !seen[k.Of] {
			seen[k.Of] = true
			of = append(of, k.Of)
		}
	}
	*p.calls = append(*p.calls, of)
	return ks, nil
}

// TestPassesSeeRuns holds the passes to seeing each run of consecutive
// parts placed on one device in one call, and a part placed elsewhere
// cutting the run in two.
func TestPassesSeeRuns(t *testing.T) {
	f := newSynth(t, jlm.TypeF32)
	gpu := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	cpu := composer.Device{Kind: composer.DeviceCPU, Slots: 6, Lanes: 8, ExactDiv: true}
	for _, tc := range []struct {
		name string
		host func(composer.Part) bool
		want [][]int
	}{
		{"all on the card", func(p composer.Part) bool { return false }, [][]int{{0, 1, 2, 3, 4, 5}}},
		{"block 1 on the host", func(p composer.Part) bool { return p.Kind == composer.PartBlock && p.Block == 1 }, [][]int{{0, 1}, {2}, {3, 4, 5}}},
	} {
		var calls [][]int
		r := composer.Default()
		for kind, prof := range r.Profiles {
			prof.Passes = append([]composer.Pass{runProbe{&calls}}, prof.Passes...)
			r.Profiles[kind] = prof
		}
		place := func(p composer.Part) composer.Device {
			if tc.host(p) {
				return cpu
			}
			return gpu
		}
		if _, err := r.Compose(f, composer.Step{Rows: 1, Page: 64, Keys: 5}, place); err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(calls) != fmt.Sprint(tc.want) {
			t.Errorf("%s: the passes saw parts %v, want %v", tc.name, calls, tc.want)
		}
		t.Logf("%s: runs %v", tc.name, calls)
	}
}
