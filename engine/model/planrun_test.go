package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestPlanDecodesAsTheHost opens synth-ernie45 on the host and WithPlanDevice
// on every device a backend opens, and holds the planned model's Prefill of a
// prompt and Forward of the host's choices after it to the host's logits in
// the coherence band and its tokens; a batch entry point on the planned
// model refuses, and a warm Forward allocates nothing.
func TestPlanDecodesAsTheHost(t *testing.T) {
	path := testmodels.Path("synth-ernie45.jlm")
	host, err := Open(path, noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	ids := host.Vocab.Encode("Hello there", true)
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	gpu := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	for _, dev := range devs {
		t.Run(dev.API(), func(t *testing.T) {
			m, err := Open(path, noTune, WithPlanDevice(dev, gpu))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			const more = 6
			hs, ps := host.NewState(len(ids)+more+8), m.NewState(64)
			defer hs.Close()
			defer ps.Close()
			want, err := hs.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ps.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for i := range more {
				nmse, _ := nmseOf(got, want)
				best := Greedy(want)
				if !(nmse < 1e-5) {
					t.Fatalf("position %d: logits NMSE %.3g against the host", len(ids)+i-1, nmse)
				}
				if g := Greedy(got); g != best {
					t.Fatalf("position %d: token %d, the host's %d", len(ids)+i-1, g, best)
				}
				worst = max(worst, nmse)
				if want, err = hs.Forward(best); err != nil {
					t.Fatal(err)
				}
				if got, err = ps.Forward(best); err != nil {
					t.Fatal(err)
				}
			}
			if ps.Pos() != hs.Pos() {
				t.Errorf("the planned State is at %d, the host's at %d", ps.Pos(), hs.Pos())
			}
			if _, err := ps.PrefillCached([]int32{1, 2}); err == nil || !strings.Contains(err.Error(), "composer's plan") {
				t.Errorf("PrefillCached on the plan: %v, want a refusal", err)
			}
			allocs := testing.AllocsPerRun(8, func() {
				if _, err := ps.Forward(3); err != nil {
					t.Fatal(err)
				}
			})
			// The race detector's instrumentation allocates: the count is the
			// build's without it.
			if allocs != 0 && !raceEnabled {
				t.Errorf("a warm Forward on the plan allocates %.1f objects", allocs)
			}
			t.Logf("%s: %d positions, worst logits NMSE %.3g, %.0f allocations a token", dev.Name(), len(ids)+more, worst, allocs)
		})
	}
}
