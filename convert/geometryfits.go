package convert

import (
	"fmt"
	"maps"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
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
			return fmt.Errorf("convert: block %d turns %d of its head by the model's %d-wide rotary: %w",
				b, r, c.NRot, ErrNotImplemented)
		}
		if !c.KVShared(b) {
			continue
		}
		src := c.KVSource(b)
		if src < 0 || src >= b || c.KVShared(src) || c.HeadDimAt(src) != c.HeadDimAt(b) || c.NKVHeadAt(src) != c.NKVHeadAt(b) {
			return fmt.Errorf("convert: block %d shares a history (block %d's) of another shape: %w",
				b, src, ErrNotImplemented)
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

// Encoders is every GGUF architecture the library states non-causal
// whatever its file says, sorted: an embedding encoder, its blocks
// non-causal, with no KV cache and no logits. Its non_causal statement is
// read as conversion reads it, over a file stating no keys; one that reads a
// key the file must state (attention.causal) depends on its file and is not
// one, nor is a decoder that pools (FlagPoolMean and its siblings).
func Encoders() ([]string, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	none := map[string]jinja.Value{"src": jinja.NewDict()}
	var out []string
	for _, mj := range files {
		for _, arch := range slices.Sorted(maps.Keys(mj.Container)) {
			src, ok := mj.Container[arch].Flags["non_causal"]
			if !ok {
				continue
			}
			v, err := mj.Container[arch].exprs[src].Eval(none, blockExprSteps)
			switch {
			case err != nil:
				return nil, fmt.Errorf("convert: %s: %s: flag non_causal: %w", mj.name, arch, err)
			case v.IsUndefined():
			case !v.IsBool():
				return nil, fmt.Errorf("convert: %s: %s: flag non_causal is %v, not true or false", mj.name, arch, v)
			case v.AsBool():
				out = append(out, arch)
			}
		}
	}
	slices.Sort(out)
	return out, nil
}

// stateRopeTable states c's rotary sections as the table a row turns by
// and a picture's layout on its axes (jlm.Config.RopeTable, PictureRope):
// sections laid one after another over the NRot/2 pairs; interleaved pair
// by pair (Qwen3-VL's apply_interleaved_mrope: pair p by axis 1 where
// p%3 == 1 and p < 3*sections[1], by axis 2 where p%3 == 2 and
// p < 3*sections[2], by axis 0 otherwise); or over the elements of
// cat(freq, freq) (HunyuanVL's XD-RoPE: a table of NRot entries, section i
// 2*sections[i] of them, element e at frequency e mod NRot/2). A picture is
// a grid from its span's start, the text after it at that start plus its
// longer side (transformers' get_rope_index); under XD-RoPE its rows keep
// their positions and take column, row and the picture's ordinal on the
// other three axes. Sections ropeFits refuses state nothing.
func stateRopeTable(c *Config) {
	stateRopeLayout(c.Config, c.Stated.ropeLayout == "xd", c.Stated.ropeLayout == "interleaved")
}

// stateRopeLayout is stateRopeTable for a rotary laid over the elements
// (xd) or interleaved (il), whichever source says so.
func stateRopeLayout(c *jlm.Config, xd, il bool) {
	c.RopeTable, c.PictureRope = nil, jlm.PictureRope{}
	if c.RopeSections == [4]uint32{} || c.NRot == 0 || c.NRot%2 != 0 {
		return
	}
	half := int(c.NRot) / 2
	sec := func(i int) int { return int(c.RopeSections[i]) }
	sum := 0
	for i := range c.RopeSections {
		sum += sec(i)
	}
	var out []jlm.RopePair
	switch {
	case xd:
		if sum != half {
			return
		}
		out = make([]jlm.RopePair, 2*half)
		for e := range out {
			out[e].Freq = e % half
			for i, at := 0, 0; i < len(c.RopeSections); i++ {
				if at += 2 * sec(i); e < at {
					out[e].Axis = i
					break
				}
			}
		}
		c.PictureRope = jlm.PictureRope{Axes: [4]jlm.PictureAxis{
			{Pos: true},
			{Mul: jlm.PicOne, Mod: jlm.PicCols},
			{Mul: jlm.PicOne, Div: jlm.PicCols},
			{Base: jlm.PicOrdinal},
		}}
	case il:
		if c.RopeSections[3] != 0 {
			return
		}
		out = make([]jlm.RopePair, half)
		for p := range out {
			out[p].Freq = p
			switch {
			case p%3 == 1 && p < 3*sec(1):
				out[p].Axis = 1
			case p%3 == 2 && p < 3*sec(2):
				out[p].Axis = 2
			}
		}
	default:
		if sum != half {
			return
		}
		for ax := range c.RopeSections {
			for range sec(ax) {
				out = append(out, jlm.RopePair{Axis: ax, Freq: len(out)})
			}
		}
	}
	if !xd {
		frame := jlm.PictureAxis{Base: jlm.PicStart, Mul: jlm.PicStride, Div: jlm.PicCells}
		c.PictureRope = jlm.PictureRope{Side: true, Axes: [4]jlm.PictureAxis{
			frame,
			{Base: jlm.PicStart, Mul: jlm.PicOne, Div: jlm.PicCols, Mod: jlm.PicRows},
			{Base: jlm.PicStart, Mul: jlm.PicOne, Mod: jlm.PicCols},
			frame,
		}}
	}
	c.RopeTable = out
}

// ropeFits refuses a rotary the blocks cannot turn as stated: sections that
// do not cover the head's rotated pairs (an XD-RoPE's chunks covering the
// whole head, never interleaved), and an encoder rotating a block by a
// rotary of no width.
func ropeFits(c *Config) error {
	if c.Stated == nil {
		return unstated("the rotary's layout")
	}
	var pairs uint32
	for _, n := range c.RopeSections {
		pairs += n
	}
	xd, il := c.Stated.ropeLayout == "xd", c.Stated.ropeLayout == "interleaved"
	if pairs != 0 && (xd && (il || 2*pairs != c.NRot) || !xd && pairs != c.NRot/2) {
		return fmt.Errorf("convert: rotary sections %v do not cover the %d rotated dimensions: %w",
			c.RopeSections, c.NRot, ErrNotImplemented)
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
			return fmt.Errorf("convert: encoder block %d rotates by a rotary of no width: %w",
				b, ErrNotImplemented)
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
