package composer_test

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestGPUEntryNamesAreIdentifiers holds every composed GPU kernel's entry
// to a name a backend accepts: PTX refuses "reduce+map" and "reduce.part".
func TestGPUEntryNamesAreIdentifiers(t *testing.T) {
	f := newSynth(t, jlm.TypeQ8)
	parts, err := composer.Front(f, false)
	if err != nil {
		t.Fatal(err)
	}
	d := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	var run []composer.PartPlan
	for _, p := range parts {
		run = append(run, composer.PartPlan{Part: p, Device: d})
	}
	ks, err := composer.Default().Kernels(f, run, 0, composer.Step{Rows: 1, Page: 64, Keys: 9})
	if err != nil {
		t.Fatal(err)
	}
	if len(ks) == 0 {
		t.Fatal("no kernel formed")
	}
	for _, k := range ks {
		l, err := composer.GPULowering().Lower(k, d)
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
	if composer.EntryName("reduce+map.part") != "reduce_map_part" || composer.EntryName("2x") != "k2x" {
		t.Errorf("entryName: %q %q", composer.EntryName("reduce+map.part"), composer.EntryName("2x"))
	}
}
