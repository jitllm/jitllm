package convert

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestLocalAxesAreRefused holds the refusal of rotary sections under the
// sliding layers' own rotary: a second base or a second width with sections
// is refused by name, and sections alone, or a second rotary alone, are not.
func TestLocalAxesAreRefused(t *testing.T) {
	secs := [4]uint32{16, 24, 24, 0}
	for _, tc := range []struct {
		name   string
		c      jlm.Config
		refuse bool
	}{
		{"sections alone", jlm.Config{Arch: jlm.ArchQwen2VL, NRot: 128, RopeSections: secs}, false},
		{"a local base alone", jlm.Config{Arch: jlm.ArchGemma3, NRot: 128, RopeBaseSWA: 10000}, false},
		{"a local width alone", jlm.Config{Arch: jlm.ArchGemma3, NRot: 128, Geoms: widths(128, 64)}, false},
		{"the same local width", jlm.Config{Arch: jlm.ArchQwen2VL, NRot: 128, Geoms: widths(128, 128), RopeSections: secs}, false},
		{"sections under a local base", jlm.Config{Arch: jlm.ArchQwen2VL, NRot: 128, RopeBaseSWA: 10000, RopeSections: secs}, true},
		{"sections under a local width", jlm.Config{Arch: jlm.ArchQwen2VL, NRot: 128, Geoms: widths(128, 64), RopeSections: secs}, true},
	} {
		err := localAxes(&tc.c)
		switch {
		case tc.refuse && !errors.Is(err, ErrNotImplemented):
			t.Errorf("%s: %v, want a refusal by name", tc.name, err)
		case !tc.refuse && err != nil:
			t.Errorf("%s: refused: %v", tc.name, err)
		}
	}
}

// widths is two blocks' geometries turning those rotary widths.
func widths(a, b uint32) []jlm.Geom { return []jlm.Geom{{NRot: a}, {NRot: b}} }
