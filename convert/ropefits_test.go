package convert

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestARotaryItsBlocksCannotTurnIsRefused: rotary sections that do not
// cover the head's rotated pairs, and an encoder rotating a block by a
// rotary of no width, are refused at conversion with ErrNotImplemented;
// sections covering the pairs, an encoder with no positional encoding and
// one rotating by a rotary of its width convert.
func TestARotaryItsBlocksCannotTurnIsRefused(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(c *jlm.Config)
		refuse bool
	}{
		{"no sections", func(*jlm.Config) {}, false},
		{"sections covering the pairs", func(c *jlm.Config) { c.RopeSections = [4]uint32{16, 8, 8} }, false},
		{"sections short of the pairs", func(c *jlm.Config) { c.RopeSections = [4]uint32{16, 8, 4} }, true},
		{"sections past the pairs", func(c *jlm.Config) { c.RopeSections = [4]uint32{16, 16, 8} }, true},
		{"an encoder rotating by its rotary", func(c *jlm.Config) { c.Flags |= jlm.FlagNonCausal }, false},
		{"an encoder with no positional encoding", func(c *jlm.Config) {
			c.Flags |= jlm.FlagNonCausal | jlm.FlagNoPosEnc
			c.NRot = 0
		}, false},
		{"an encoder rotating by a rotary of no width", func(c *jlm.Config) {
			c.Flags |= jlm.FlagNonCausal
			c.NRot = 0
		}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &jlm.Config{NLayer: 2, NHead: 2, NKVHead: 2, HeadDim: 64, NRot: 64}
			tc.change(c)
			err := ropeFits(c)
			switch {
			case tc.refuse && !errors.Is(err, ErrNotImplemented):
				t.Fatalf("refused with %v, want ErrNotImplemented", err)
			case !tc.refuse && err != nil:
				t.Fatal(err)
			}
		})
	}
}
