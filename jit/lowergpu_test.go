package jit_test

import (
	"testing"

	"github.com/jitllm/jitllm/jit"
)

// TestGPUEntryNamesAreIdentifiers holds every composed GPU kernel's entry
// to a name a backend accepts: PTX refuses "reduce+map" and "reduce.part".
func TestGPUEntryNamesAreIdentifiers(t *testing.T) {
	f := composing(t, *composeModel+".jlm")
	parts, err := jit.Front(f)
	if err != nil {
		t.Fatal(err)
	}
	d := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	var run []jit.PartPlan
	for _, p := range parts {
		run = append(run, jit.PartPlan{Part: p, Device: d})
	}
	ks, err := jit.Default().Kernels(f, run, 0, jit.Step{Rows: 1, Page: 64, Keys: 9})
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range ks {
		l, err := jit.GPULowering.Lower(k, d)
		if err != nil {
			t.Fatal(err)
		}
		name := l.GPU.Name
		for i, r := range name {
			if !(r == '_' || 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || i > 0 && '0' <= r && r <= '9') {
				t.Errorf("%s lowers to the entry %q", k.Name, name)
				break
			}
		}
	}
	if jit.EntryName("reduce+map.part") != "reduce_map_part" || jit.EntryName("2x") != "k2x" {
		t.Errorf("entryName: %q %q", jit.EntryName("reduce+map.part"), jit.EntryName("2x"))
	}
}
