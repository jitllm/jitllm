package cost

import (
	"flag"
	"testing"

	"github.com/jitllm/jitllm/jit"
)

var composeModel = flag.String("compose-model", "synth-ernie45", "the fixture the estimates price")

// TestSuggestPlacesFromTheIR estimates placements from final IR and holds
// plan mode's suggestion to costing no more than any of them, and to the
// estimate of the placement it returns.
func TestSuggestPlacesFromTheIR(t *testing.T) {
	f := composing(t, *composeModel+".jlm")
	r := jit.Default()
	gpu := Device{Device: jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, Bandwidth: 800e9, Launch: 4e-6, Link: 12e9}
	cpu := Device{Device: jit.Device{Kind: jit.DeviceCPU, Slots: 6, Lanes: 8}, Bandwidth: 40e9, Launch: 3e-6, Link: 12e9}
	s := jit.Step{Rows: 1, Page: 64, Keys: 5}
	forced := map[string]Placement{
		"card":          func(jit.Part) Device { return gpu },
		"host":          func(jit.Part) Device { return cpu },
		"head on host":  func(p jit.Part) Device { return map[bool]Device{true: cpu, false: gpu}[p.Kind == jit.PartHead] },
		"table on host": func(p jit.Part) Device { return map[bool]Device{true: cpu, false: gpu}[p.Kind == jit.PartTable] },
	}
	p, c, err := Suggest(r, f, s, []Device{gpu, cpu})
	if err != nil {
		t.Fatal(err)
	}
	again, err := Estimate(r, f, s, p)
	if err != nil {
		t.Fatal(err)
	}
	if d := again.Seconds - c.Seconds; d > 1e-12 || d < -1e-12 {
		t.Errorf("suggested %+v, its estimate %+v", c, again)
	}
	for name, q := range forced {
		e, err := Estimate(r, f, s, q)
		if err != nil {
			t.Fatal(err)
		}
		if e.Bytes == 0 || e.Launches == 0 {
			t.Errorf("%s: an empty estimate %+v", name, e)
		}
		if e.Seconds < c.Seconds-1e-12 {
			t.Errorf("%s at %+v beats the suggestion %+v", name, e, c)
		}
		t.Logf("%-13s %+v", name, e)
	}
	parts, _ := jit.Front(f)
	for _, part := range parts {
		t.Logf("suggested: %s on %d", part, p(part).Kind)
	}
}

// TestEngineTakesOrForcesThePlacement composes the fixture on plan mode's
// suggestion and on a placement the engine forces, each part landing where
// its placement put it.
func TestEngineTakesOrForcesThePlacement(t *testing.T) {
	f := composing(t, *composeModel+".jlm")
	r := jit.Default()
	gpu := Device{Device: jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, Bandwidth: 800e9, Launch: 4e-6, Link: 12e9}
	cpu := Device{Device: jit.Device{Kind: jit.DeviceCPU, Slots: 6, Lanes: 8}, Bandwidth: 40e9, Launch: 3e-6, Link: 12e9}
	s := jit.Step{Rows: 1, Page: 64, Keys: 5}
	suggested, _, err := Suggest(r, f, s, []Device{gpu, cpu})
	if err != nil {
		t.Fatal(err)
	}
	forced := func(p jit.Part) Device {
		if p.Kind == jit.PartSample {
			return cpu
		}
		return gpu
	}
	for name, p := range map[string]Placement{"suggested": suggested, "forced": forced} {
		plan, err := r.Compose(f, s, p.Composer())
		if plan == nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i, pp := range plan.Parts {
			if pp.Device != p(pp.Part).Device {
				t.Errorf("%s: %s on %d, placed on %d", name, pp.Part, pp.Device.Kind, p(pp.Part).Kind)
			}
			if i > 0 && plan.Parts[i-1].Device != pp.Device {
				seen := false
				for _, tr := range plan.Transfers {
					seen = seen || tr.From == i-1 && tr.To == i
				}
				if !seen {
					t.Errorf("%s: no transfer between %s and %s", name, plan.Parts[i-1].Part, pp.Part)
				}
			}
		}
		t.Logf("%s: %d parts, %d transfers", name, len(plan.Parts), len(plan.Transfers))
	}
}
