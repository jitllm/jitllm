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
func geometryFits(c *Config) error {
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
func slidingRotary(c *Config) (uint32, bool) {
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

// Encoder reports whether a is an embedding encoder (the BERT family): its
// blocks non-causal, with no KV cache and no logits. A decoder that pools
// (FlagPoolMean and its siblings) is not one.
func Encoder(a Arch) bool {
	return a == ArchBERT || a == ArchNomicBERT || a == ArchModernBERT || a == ArchLFM2Enc
}

// xdRope reports whether a's M-RoPE assigns its axes per element of the
// rotary rather than per pair (HunyuanVL).
func xdRope(a Arch) bool { return a == ArchHunyuanVL }

// interleavedMRope reports whether a's M-RoPE sections are interleaved over a
// head's pairs rather than laid out as contiguous runs (Qwen2-VL's). Qwen3.5
// (ArchQwen3Next with sections) is Qwen3-VL's split under its own tower;
// qwen3next itself carries no sections and has no runs.
func interleavedMRope(a Arch) bool {
	return a == ArchQwen3VL || a == ArchQwen3VLMoE || a == ArchQwen3Next || a == ArchQwen4Exp
}

// ropeFits refuses a rotary the blocks cannot turn as stated: sections that
// do not cover the head's rotated pairs (an XD-RoPE's chunks covering the
// whole head, never interleaved), and an encoder rotating a block by a
// rotary of no width.
func ropeFits(c *Config) error {
	var pairs uint32
	for _, n := range c.RopeSections {
		pairs += n
	}
	xd, il := xdRope(c.Arch), interleavedMRope(c.Arch)
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
func compressing(c *Config, b int) bool {
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
