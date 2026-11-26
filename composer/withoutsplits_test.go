package composer_test

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
)

// TestWithoutSplitsFormsFlashWhole holds the unsplit registry, a gate's
// reference, to no pass that cuts a sum into segments on the GPU, a flash
// form's history among them (Flash.Split 1), and leaves the registry it is
// taken from as it was. The violation, WithoutSplits leaving Flash's Split
// as it was, fails it.
func TestWithoutSplitsFormsFlashWhole(t *testing.T) {
	r := composer.Default()
	check := func(ps []composer.Pass) (cut, flash, whole int) {
		for _, p := range ps {
			switch x := p.(type) {
			case composer.Split, composer.SplitRange, composer.SplitExtreme, composer.GroupExtreme:
				cut++
			case composer.Flash:
				flash++
				if x.Split == 1 {
					whole++
				}
			}
		}
		return
	}
	cut, flash, whole := check(r.WithoutSplits(composer.DeviceGPU).Profiles[composer.DeviceGPU].Passes)
	if flash == 0 {
		t.Fatal("the GPU profile has no flash pass: the gate sees nothing")
	}
	if cut != 0 || whole != flash {
		t.Errorf("unsplit: %d passes cut a sum, %d of %d flash passes split none", cut, whole, flash)
	}
	cut, _, whole = check(r.Profiles[composer.DeviceGPU].Passes)
	if cut == 0 || whole != 0 {
		t.Errorf("the registry it was taken from: %d passes cut a sum, %d flash passes split none", cut, whole)
	}
}
