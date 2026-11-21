package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// The flagship architectures (MiniMax-M2 and the group around it).
// .notes/design/model-coverage.md section 9 is the survey they were chosen
// from, with what each needs and what it costs.

// Gemma 4 is transformers' Gemma4ForCausalLM,
// text only: gemma3's block with two attention geometries (the global layers'
// HeadDim, NKVHead and NRot, the sliding layers' their own, each block's
// stated in jlm.Config.Geoms), the global layers' "proportional" rotary as
// RoleRopeFreqs, a weightless norm on v (Flag2VNorm), k's projection as v on
// a global layer with no v_proj, an attention scale of one and a per-block
// layer_scalar (RoleLayerOutScale). The 26B runs a dense MLP beside a
// mixture (jlm.Flag2DenseMoE); the E2B/E4B add per-layer embeddings
// (PLEDim), KV sharing (NKVShared) and a double-width FFN on the sharing
// layers, each block's width in its geometry.
//
// The reference-to-container table: .notes/engineering-history/
// model-correctness.md, "convert/flagship.go: gemma4Config"; the gemma4
// statement states all of it but each block's geometry.

// stateKindGeoms states each block's geometry where it differs from the
// model's, as the file states it per layer: the sliding layers' head and
// rotary where the sliding statement states them, the kv heads of each
// attending layer where the count differs by layer, and the feed-forward
// width where it is stated per layer. The model's geometry is its global
// layers', the keys without _swa being theirs.
func stateKindGeoms(f *meta.File, c *Config) error {
	var ffns []int32
	if kv, ok := f.Key("feed_forward_length"); ok && c.NFFN == 0 && kv.Type == meta.Array {
		ws, err := kv.Int32s()
		if err != nil || len(ws) != int(c.NLayer) {
			return fmt.Errorf("feed_forward_length: %d values for %d layers (%v)", len(ws), c.NLayer, err)
		}
		for i, w := range ws {
			if w <= 0 {
				return fmt.Errorf("feed_forward_length %v: layer %d has no feed-forward: %w", ws, i, ErrNotImplemented)
			}
		}
		c.NFFN, ffns = uint32(ws[0]), ws
	}
	_, kvs, err := layerCount(f, "attention.head_count_kv", c.NHead)
	if err != nil {
		return err
	}
	if kvs != nil && len(kvs) != int(c.NLayer) {
		return fmt.Errorf("attention.head_count_kv: %d values for %d layers", len(kvs), c.NLayer)
	}
	sw := c.Stated.sliding
	if sw == nil && ffns == nil && kvs == nil {
		return nil
	}
	if kvs != nil {
		c.NKVHead = 0
		for _, global := range []bool{true, false} {
			for i, v := range kvs {
				if v != 0 && c.NKVHead == 0 && (!global || !sliding(c, i)) {
					c.NKVHead = uint32(v)
				}
			}
		}
	}
	model := jlm.Geom{NKVHead: c.NKVHead, HeadDim: c.HeadDim, NRot: c.NRot, NFFN: c.NFFN}
	gs := make([]jlm.Geom, c.NLayer)
	differ := false
	for i := range gs {
		g := model
		if sw != nil && sliding(c, i) {
			g.HeadDim, g.NRot = sw.HeadDim, sw.NRot
		}
		if kvs != nil && kvs[i] != 0 {
			g.NKVHead = uint32(kvs[i])
		}
		if g.NKVHead == 0 || c.NHead%g.NKVHead != 0 {
			return fmt.Errorf("block %d: %d kv heads for %d query heads", i, g.NKVHead, c.NHead)
		}
		if ffns != nil {
			g.NFFN = uint32(ffns[i])
		}
		gs[i] = g
		differ = differ || g != model
	}
	if differ {
		c.Geoms = gs
	}
	return nil
}

// blockIndexerFits refuses an indexer selecting blocks of positions (IdxBlock;
// MiniMax Sparse Attention) that the engine's
// selection cannot run: other than one indexer head per kv group, a key of
// another width than the head (it is cached as one more kv head), and no
// query's own block forced in inside the top-k. None for an indexer that
// selects positions.
func blockIndexerFits(c *Config) error {
	switch {
	case c.IdxBlock == 0:
		return nil
	case c.IdxHeads == 0 || c.IdxHeadDim == 0 || c.IdxTopK == 0:
		return fmt.Errorf("indexer %d heads of %d, top %d blocks of %d", c.IdxHeads, c.IdxHeadDim, c.IdxTopK, c.IdxBlock)
	case c.IdxHeads != c.NKVHead:
		return fmt.Errorf("%d indexer heads for %d kv groups, and a block selection "+
			"selects once per group: %w", c.IdxHeads, c.NKVHead, ErrNotImplemented)
	case c.IdxHeadDim != c.HeadDim:
		return fmt.Errorf("indexer keys of %d on %d-wide heads: the engine caches the key as "+
			"one more kv head: %w", c.IdxHeadDim, c.HeadDim, ErrNotImplemented)
	case c.IdxLocal == 0 || c.IdxLocal > c.IdxTopK:
		return fmt.Errorf("%d local blocks of %d selected: a query's own block forced in, "+
			"inside the top-k, is implemented: %w", c.IdxLocal, c.IdxTopK, ErrNotImplemented)
	}
	return nil
}

// denseMoEFits reads a dense MLP beside a mixture (jlm.Flag2DenseMoE): the
// dense MLP is carried as the shared expert at the dense width, so a file
// with a shared expert of its own has no place for it. What its router does
// is its statement's router.
func denseMoEFits(f *meta.File, c *Config) error {
	if c.NExpert == 0 || !c.Flags2.Has(jlm.Flag2DenseMoE) {
		return nil
	}
	if f.UintKey("expert_shared_count", 0) != 0 {
		return fmt.Errorf("a shared expert beside a dense MLP carried as the shared expert: %w", ErrNotImplemented)
	}
	c.NFFNShExp = c.NFFN
	return nil
}

// sliding reports that block i of c's pattern slides.
func sliding(c *Config, i int) bool {
	return c.SWAPeriod != 0 && uint32(i)%c.SWAPeriod < c.SWAPeriod-1
}
