package convert

import (
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// swigluLimitDefault is transformers' default swiglu_limit, the published
// config's: the clamp of a file that states none, and of a container
// written when no other clamp converted, restated from its flag.
const swigluLimitDefault = 10

// statedFeaturesFit holds a file to what its statement's features need,
// once the keys are read: an attention whose key is its value
// (Flag2KeyValue, DeepSeek V4's) has one kv head and no value width of its
// own, a SwiGLU clamped per block (Flag2SwiGLUClamp) reads its limits
// (swigluClamps), and compressed blocks (compressed_blocks) read their kinds
// (compressionKinds).
func statedFeaturesFit(f *meta.File, c *Config) error {
	// A latent attention's key carries a shared rotary part beside the
	// latent's, which sizes the cached row.
	if c.KVLoraRank != 0 && (c.NRot == 0 || c.NRot >= c.HeadDim) {
		return fmt.Errorf("rope.dimension_count %d on a key of %d: a latent attention's key carries its "+
			"shared rotary part beside the latent's", c.NRot, c.HeadDim)
	}
	if c.Flags2.Has(jlm.Flag2KeyValue) {
		if c.NKVHead != 1 {
			return fmt.Errorf("%d kv heads, and an attention whose key is its value is one shared key "+
				"and value: %w", c.NKVHead, ErrNotImplemented)
		}
		if c.HeadDimV != 0 {
			return fmt.Errorf("value width %d beside key width %d, and the value is the key: %w",
				c.HeadDimV, c.HeadDim, ErrNotImplemented)
		}
	}
	if c.Flags2.Has(jlm.Flag2SwiGLUClamp) {
		if err := swigluClamps(f, c); err != nil {
			return err
		}
	}
	if c.Stated.compressedBlocks {
		return compressionKinds(f, c)
	}
	return nil
}

// swigluClamps reads each block's SwiGLU clamp: llama.cpp's converter writes the
// routed and the shared limit per block from swiglu_limit (one value stands
// for every block), and a file stating neither has transformers' default. A
// block states one activation, so one whose routed and shared limits differ
// is refused.
func swigluClamps(f *meta.File, c *Config) error {
	n := int(c.NLayer)
	read := func(key string) ([]float32, error) {
		kv, ok := f.Key(key)
		if !ok {
			return nil, nil
		}
		vs, err := kv.Float32s()
		if err != nil {
			v, ok := kv.Float()
			if !ok {
				return nil, fmt.Errorf("%s: %w", key, err)
			}
			vs = []float32{float32(v)}
		}
		if len(vs) == 1 {
			vs = slices.Repeat(vs, n)
		}
		if len(vs) != n {
			return nil, fmt.Errorf("%s: %d limits for %d blocks", key, len(vs), n)
		}
		for i, v := range vs {
			if !(v > 0) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("%s[%d] is %g, and a clamp is positive and finite", key, i, v)
			}
		}
		return vs, nil
	}
	routed, err := read("swiglu_clamp_exp")
	if err != nil {
		return err
	}
	shared, err := read("swiglu_clamp_shexp")
	if err != nil {
		return err
	}
	switch {
	case routed == nil && shared == nil:
		routed = slices.Repeat([]float32{swigluLimitDefault}, n)
	case routed == nil:
		routed = shared
	case shared != nil:
		for i := range routed {
			if routed[i] != shared[i] {
				return fmt.Errorf("block %d clamps its routed experts at %g and its shared expert at %g, "+
					"and a block states one activation: %w", i, routed[i], shared[i], ErrNotImplemented)
			}
		}
	}
	c.SwiGLULimits = routed
	return nil
}

// compressionKinds reads each block's compression off compress_ratios and the
// tensors: a block with the indexer is CSA (its compressor two heads wide,
// the overlapping windows), one with a compressor alone is HCA, one with
// neither a sliding window alone, and its ratio must say the same. llama.cpp
// tells the two kinds apart by the ratio (4 against 128); transformers by the
// layer type, each type at one rate -- which is what the container carries.
func compressionKinds(f *meta.File, c *Config) error {
	kv, ok := f.Key("attention.compress_ratios")
	if !ok {
		return fmt.Errorf("no attention.compress_ratios")
	}
	ratios, err := kv.Int32s()
	if err != nil {
		return fmt.Errorf("attention.compress_ratios: %w", err)
	}
	if len(ratios) < int(c.NLayer) {
		return fmt.Errorf("%d compress ratios for %d blocks", len(ratios), c.NLayer)
	}
	c.CompKinds = make([]jlm.CompKind, c.NLayer)
	for i := range c.CompKinds {
		r := ratios[i]
		_, idx := f.Get("blk." + itoa(i) + ".indexer.proj.weight")
		ckv, cmp := f.Get("blk." + itoa(i) + ".attn_compressor_kv.weight")
		rows := uint64(0)
		if cmp && len(ckv.Dims) == 2 {
			rows = ckv.Dims[1]
		}
		switch {
		case r == 0 && !cmp && !idx:
			c.CompKinds[i] = jlm.CompNone
		case r > 0 && cmp && idx && rows == 2*uint64(c.HeadDim):
			c.CompKinds[i] = jlm.CompCSA
			if c.CompRateCSA != 0 && c.CompRateCSA != uint32(r) {
				return fmt.Errorf("block %d compresses every %d where an earlier CSA block does every %d: %w",
					i, r, c.CompRateCSA, ErrNotImplemented)
			}
			c.CompRateCSA = uint32(r)
		case r > 0 && cmp && !idx && rows == uint64(c.HeadDim):
			c.CompKinds[i] = jlm.CompHCA
			if c.CompRateHCA != 0 && c.CompRateHCA != uint32(r) {
				return fmt.Errorf("block %d compresses every %d where an earlier HCA block does every %d: %w",
					i, r, c.CompRateHCA, ErrNotImplemented)
			}
			c.CompRateHCA = uint32(r)
		default:
			return fmt.Errorf("block %d: ratio %d with a compressor %v (%d rows on %d-wide heads) and an "+
				"indexer %v, which is neither CSA, HCA nor a sliding block", i, r, cmp, rows, c.HeadDim, idx)
		}
	}
	if c.CompRateCSA != 0 && c.IdxHeads == 0 {
		return fmt.Errorf("CSA blocks with no indexer")
	}
	if w := c.SWAWindow; w < c.CompRateHCA || w < 2*c.CompRateCSA {
		return fmt.Errorf("a window of %d under HCA every %d and CSA every %d: the engine keeps the "+
			"compressor's pending inputs in the window's rows: %w", w, c.CompRateHCA, c.CompRateCSA,
			ErrNotImplemented)
	}
	return nil
}

// hashExpertsFit holds each hash-routed block's token-to-expert table to what
// the banks it indexes take: one row of NExpertUsed ids per token, every id an
// expert, and none twice for a token -- the engine selects them as the top-k
// of a mask (model.ds4HashBias), which cannot select one expert twice, where
// both references would run it twice.
func hashExpertsFit(c *Config, s *jlm.Source) error {
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleHashExperts {
			continue
		}
		if t.Type != jlm.TypeF32 || t.NDim != 2 || t.Dims[0] != uint64(c.NExpertUsed) || t.Dims[1] != uint64(c.NVocab) {
			return fmt.Errorf("convert: %s is %v %v, want %d ids for each of %d tokens", t.Name, t.Type,
				t.Dims[:t.NDim], c.NExpertUsed, c.NVocab)
		}
		ids, err := vecOf(t)
		if err != nil {
			return err
		}
		k := int(c.NExpertUsed)
		for j, id := range ids {
			tok, at := j/k, j%k
			if id != float32(int32(id)) || id < 0 || uint32(id) >= c.NExpert {
				return fmt.Errorf("convert: %s routes token %d to expert %v of %d", t.Name, tok, id, c.NExpert)
			}
			if slices.Contains(ids[tok*k:tok*k+at], id) {
				return fmt.Errorf("convert: %s routes token %d to expert %v twice: %w", t.Name, tok, id, ErrNotImplemented)
			}
		}
	}
	return nil
}
