package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
)

// stateQSA states QSA (query-sparse attention) on the attention blocks from
// the ratio each block compresses by, rs (transformers'
// Qwen4ExpForCausalLM, the qsa_ratios statement; none stated
// for nil): the ratio on every attention block, zero on every linear one,
// one ratio throughout, and the indexer's heads, width and a budget of
// whole compressed blocks. Any other pattern is refused.
func stateQSA(f *meta.File, c *Config, rs []uint64) error {
	if rs != nil {
		c.QSARatio = 0
		for _, r := range rs {
			if r == 0 {
				continue
			}
			if c.QSARatio != 0 && uint64(c.QSARatio) != r {
				return fmt.Errorf("QSA blocks of %d and %d positions: one ratio is %w",
					c.QSARatio, r, ErrNotImplemented)
			}
			c.QSARatio = uint32(r)
		}
		if uint64(len(rs)) < uint64(c.NLayer) {
			return fmt.Errorf("%d compress ratios for %d blocks", len(rs), c.NLayer)
		}
		// A prediction block that carries the QSA indexer is a QSA block at the
		// model's ratio. unsloth's published heads state 0 for it where other
		// GGUFs write the ratio, and the release's config makes it full
		// attention, which this model runs as QSA; the tensors decide (RULE
		// 7m).
		trunk := int(c.NLayer) - int(f.UintKey("nextn_predict_layers", 0))
		for i := trunk; i < int(c.NLayer) && i < len(rs) && c.QSARatio != 0; i++ {
			if _, idx := f.Get("blk." + itoa(i) + ".indexer.k_proj.weight"); idx && rs[i] == 0 {
				rs[i] = uint64(c.QSARatio)
			}
		}
		for i := 0; i < int(c.NLayer); i++ {
			_, linear := f.Get("blk." + itoa(i) + ".ssm_conv1d.weight")
			if linear == (rs[i] != 0) && (linear || c.QSARatio != 0) {
				return fmt.Errorf("block %d (linear %v) has compress ratio %d: QSA is every "+
					"attention block at one ratio: %w", i, linear, rs[i], ErrNotImplemented)
			}
		}
	}
	if c.QSARatio != 0 {
		if c.IdxHeads == 0 || c.IdxHeadDim == 0 || c.IdxTopK == 0 || c.IdxTopK%c.QSARatio != 0 {
			return fmt.Errorf("QSA indexer: %d heads of %d, a budget of %d at a ratio of %d",
				c.IdxHeads, c.IdxHeadDim, c.IdxTopK, c.QSARatio)
		}
		if c.NRot > c.IdxHeadDim {
			return fmt.Errorf("a rotary of %d on an indexer key of %d", c.NRot, c.IdxHeadDim)
		}
	}
	return nil
}

// ngramConfig reads the n-gram embedding's keys (the ngram statement): one
// linear block's table of hashed n-grams (ple.*), the hash constants as the
// unsigned integers they are. Refused: more than one n-gram block, and
// constants too short for the heads they describe.
func ngramConfig(f *meta.File, c *Config) error {
	kv, ok := f.Key("ple.layers")
	if !ok {
		return nil
	}
	ls, err := kv.Uint64s()
	if err != nil {
		return fmt.Errorf("ple.layers: %w", err)
	}
	switch {
	case len(ls) == 0:
		return nil
	case len(ls) != 1:
		return fmt.Errorf("%d n-gram blocks: one is %w", len(ls), ErrNotImplemented)
	case ls[0] >= uint64(c.NLayer):
		return fmt.Errorf("n-gram block %d of %d", ls[0], c.NLayer)
	}
	c.NGramBlock = uint32(ls[0]) + 1
	c.NGram = uint32(f.UintKey("ple.ngram_size", 0))
	c.NGramHeadsPer = uint32(f.UintKey("ple.heads_per_ngram", 0))
	c.NGramConv = uint32(f.UintKey("ple.conv_kernel", 0))
	c.NGramRowDim = uint32(f.UintKey("embedding_length_per_layer_input", 0))
	eos, ok := f.Key("ple.eos_token_id")
	if !ok {
		return fmt.Errorf("no ple.eos_token_id: the hash resets on it")
	}
	e, _ := eos.Uint()
	c.NGramEOS = uint32(e)
	// A file that predates the key takes the end-of-sequence id; the image id
	// is what transformers hashes at a picture's rows.
	c.NGramImage = uint32(f.UintKey("ple.image_token_id", e))
	for _, k := range []struct {
		key string
		dst *[]uint64
	}{{"ple.layer_multipliers", &c.NGramMul}, {"ple.head_offsets", &c.NGramOff},
		{"ple.head_vocab_sizes", &c.NGramVoc}} {
		v, ok := f.Key(k.key)
		if !ok {
			return fmt.Errorf("no %s", k.key)
		}
		if *k.dst, err = v.Uint64s(); err != nil {
			return fmt.Errorf("%s: %w", k.key, err)
		}
	}
	if c.NGram < 2 || c.NGramHeadsPer == 0 {
		return fmt.Errorf("%d-grams with %d heads each", c.NGram, c.NGramHeadsPer)
	}
	// The arrays are read at their stated length and must hold at
	// least the heads' worth; the container holds exactly that many.
	heads := int(c.NGram-1) * int(c.NGramHeadsPer)
	if len(c.NGramMul) < int(c.NGram) || len(c.NGramOff) < heads || len(c.NGramVoc) < heads {
		return fmt.Errorf("%d multipliers, %d offsets, %d vocabularies for %d-grams over %d heads",
			len(c.NGramMul), len(c.NGramOff), len(c.NGramVoc), c.NGram, heads)
	}
	c.NGramMul, c.NGramOff, c.NGramVoc = c.NGramMul[:c.NGram], c.NGramOff[:heads], c.NGramVoc[:heads]
	t, ok := f.Get("per_layer_token_embd.weight")
	if !ok {
		return fmt.Errorf("an n-gram block with no per_layer_token_embd table")
	}
	if len(t.Dims) != 2 || t.Dims[0] != uint64(c.NGramRowDim) {
		return fmt.Errorf("per_layer_token_embd is %v, rows of %d expected", t.Dims, c.NGramRowDim)
	}
	for h := range c.NGramVoc {
		if c.NGramVoc[h] == 0 || c.NGramOff[h]+c.NGramVoc[h] > t.Dims[1] {
			return fmt.Errorf("n-gram head %d reads rows [%d, %d) of a %d-row table",
				h, c.NGramOff[h], c.NGramOff[h]+c.NGramVoc[h], t.Dims[1])
		}
	}
	return nil
}
