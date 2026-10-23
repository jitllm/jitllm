package jit_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// TestEveryKernelRecordsItsPasses composes the fixture on each device and
// holds every kernel formed to naming the passes that formed it.
func TestEveryKernelRecordsItsPasses(t *testing.T) {
	f := newSynth(t, jlm.TypeF32)
	r := jit.Default()
	s := jit.Step{Rows: 1, Page: 64, Keys: 5}
	for _, d := range []jit.Device{{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: jit.DeviceCPU, Slots: 6, Lanes: 8}} {
		plan, err := r.Compose(f, s, func(jit.Part) jit.Device { return d })
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

func (p runProbe) Apply(ks []*jit.Kernel, d jit.Device) ([]*jit.Kernel, error) {
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
	gpu := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	cpu := jit.Device{Kind: jit.DeviceCPU, Slots: 6, Lanes: 8}
	for _, tc := range []struct {
		name string
		host func(jit.Part) bool
		want [][]int
	}{
		{"all on the card", func(p jit.Part) bool { return false }, [][]int{{0, 1, 2, 3, 4, 5}}},
		{"block 1 on the host", func(p jit.Part) bool { return p.Kind == jit.PartBlock && p.Block == 1 }, [][]int{{0, 1}, {2}, {3, 4, 5}}},
	} {
		var calls [][]int
		r := jit.Default()
		for kind, prof := range r.Profiles {
			prof.Passes = append([]jit.Pass{runProbe{&calls}}, prof.Passes...)
			r.Profiles[kind] = prof
		}
		place := func(p jit.Part) jit.Device {
			if tc.host(p) {
				return cpu
			}
			return gpu
		}
		if _, err := r.Compose(f, jit.Step{Rows: 1, Page: 64, Keys: 5}, place); err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(calls) != fmt.Sprint(tc.want) {
			t.Errorf("%s: the passes saw parts %v, want %v", tc.name, calls, tc.want)
		}
		t.Logf("%s: runs %v", tc.name, calls)
	}
}
