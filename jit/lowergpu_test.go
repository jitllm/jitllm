package jit

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGPUEntryNamesAreIdentifiers holds every composed GPU kernel's entry
// to a name a backend accepts: PTX refuses "reduce+map" and "reduce.part".
func TestGPUEntryNamesAreIdentifiers(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	parts, err := Front(f)
	if err != nil {
		t.Fatal(err)
	}
	d := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	var run []PartPlan
	for _, p := range parts {
		run = append(run, PartPlan{Part: p, Device: d})
	}
	ks, err := Default().Kernels(f, run, 0, Step{Rows: 1, Page: 64, Keys: 9})
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range ks {
		l, err := gpuLowering{}.Lower(k, d)
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
	if entryName("reduce+map.part") != "reduce_map_part" || entryName("2x") != "k2x" {
		t.Errorf("entryName: %q %q", entryName("reduce+map.part"), entryName("2x"))
	}
}
