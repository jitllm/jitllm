package composer_test

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
)

// TestWithRemaindersFormsTwoProducts holds the registry a caller asks for
// flash's remainders by to flash passes that stage them, and the registry it
// is taken from, the default, to passes that do not.
func TestWithRemaindersFormsTwoProducts(t *testing.T) {
	r := composer.Default()
	count := func(ps []composer.Pass) (flash, rest int) {
		for _, p := range ps {
			if f, ok := p.(composer.Flash); ok {
				flash++
				if f.Remainders {
					rest++
				}
			}
		}
		return
	}
	flash, rest := count(r.WithRemainders(composer.DeviceGPU).Profiles[composer.DeviceGPU].Passes)
	if flash == 0 {
		t.Fatal("the GPU profile has no flash pass: the gate sees nothing")
	}
	if rest != flash {
		t.Errorf("with remainders: %d of %d flash passes stage them", rest, flash)
	}
	if _, rest = count(r.Profiles[composer.DeviceGPU].Passes); rest != 0 {
		t.Errorf("the default: %d flash passes stage the remainders", rest)
	}
}
