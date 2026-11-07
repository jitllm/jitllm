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
	_, slidingRope := slidingRotary(c)
	for b := range int(c.NLayer) {
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

// slidingRotary is the width the sliding layers' own rotary turns -- the
// widest of theirs -- and whether the model builds one: where they state a
// base or a width of their own.
func slidingRotary(c *jlm.Config) (uint32, bool) {
	var local uint32
	for b := range int(c.NLayer) {
		if c.Sliding(b) {
			local = max(local, c.NRotAt(b))
		}
	}
	if local == 0 {
		local = c.NRot
	}
	return local, c.RopeBaseSWA != 0 || local != c.NRot
}

// ropeFits refuses a rotary the blocks cannot turn as stated: sections that
// do not cover the head's rotated pairs (an XD-RoPE's chunks covering the
// whole head, never interleaved), and an encoder rotating a block by a
// rotary of no width.
func ropeFits(c *jlm.Config) error {
	var pairs uint32
	for _, n := range c.RopeSections {
		pairs += n
	}
	xd, il := c.Arch.XDRope(), c.Arch.InterleavedMRope()
	if pairs != 0 && (xd && (il || 2*pairs != c.NRot) || !xd && pairs != c.NRot/2) {
		return fmt.Errorf("convert: %v's rotary sections %v do not cover its %d rotated dimensions: %w",
			c.Arch, c.RopeSections, c.NRot, ErrNotImplemented)
	}
	if !c.Flags.Has(jlm.FlagNonCausal) || c.Flags.Has(jlm.FlagNoPosEnc) {
		return nil
	}
	local, own := slidingRotary(c)
	for b := range int(c.NLayer) {
		if c.Flags.Has(jlm.FlagNoPEGlobal) && !(c.SWAPeriod > 0 && uint32(b)%c.SWAPeriod < c.SWAPeriod-1) {
			continue
		}
		w := c.NRot
		if own && c.Sliding(b) && !compressing(c, b) {
			w = local
		}
		if w == 0 {
			return fmt.Errorf("convert: %v's encoder block %d rotates by a rotary of no width: %w",
				c.Arch, b, ErrNotImplemented)
		}
	}
	return nil
}

// compressing reports that block b compresses its history, which it turns
// by the model's rotary as its entries do.
func compressing(c *jlm.Config, b int) bool {
	if b >= len(c.CompKinds) {
		return false
	}
	switch c.CompKinds[b] {
	case jlm.CompCSA:
		return c.CompRateCSA != 0
	case jlm.CompHCA:
		return c.CompRateHCA != 0
	}
	return false
}
