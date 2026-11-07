package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// geometryFits refuses blocks whose geometry no rotary or history of the
// model states: a block of its own rotary width that is not a sliding block
// turned by the sliding layers' own rotary (built where they state a base or
// a width of their own), and a KV-sharing block whose source is not an
// earlier block of its own head width and kv heads.
func geometryFits(c *jlm.Config) error {
	n := int(c.NLayer)
	var local uint32
	for b := range n {
		if c.Sliding(b) {
			local = max(local, c.NRotAt(b))
		}
	}
	if local == 0 {
		local = c.NRot
	}
	slidingRope := c.RopeBaseSWA != 0 || local != c.NRot
	for b := range n {
		if r := c.NRotAt(b); r != c.NRot && !(slidingRope && c.Sliding(b)) {
			return fmt.Errorf("convert: %v's block %d turns %d of its head by the model's %d-wide rotary: %w",
				c.Arch, b, r, c.NRot, ErrNotImplemented)
		}
		if !c.KVShared(b) {
			continue
		}
		src := c.KVSource(b)
		if src < 0 || src >= b || c.KVShared(src) || c.HeadDimAt(src) != c.HeadDimAt(b) || c.NKVHeadAt(src) != c.NKVHeadAt(b) {
			return fmt.Errorf("convert: %v's block %d shares a history (block %d's) of another shape: %w",
				c.Arch, b, src, ErrNotImplemented)
		}
	}
	return nil
}
