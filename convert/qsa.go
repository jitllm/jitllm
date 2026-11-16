package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

func qwen4expRoleOf(name string) (jlm.Role, bool) {
	switch name {
	// Qwen3.8-Flash-Next's hyper-connections, per block and at the head.
	case "hc_attn_norm.weight":
		return jlm.RoleHCAttnNorm, true
	case "hc_attn_down.weight":
		return jlm.RoleHCAttnDown, true
	case "hc_attn_up.weight":
		return jlm.RoleHCAttnUp, true
	case "hc_attn_inject.weight":
		return jlm.RoleHCAttnInject, true
	case "hc_ffn_norm.weight":
		return jlm.RoleHCFFNNorm, true
	case "hc_ffn_down.weight":
		return jlm.RoleHCFFNDown, true
	case "hc_ffn_up.weight":
		return jlm.RoleHCFFNUp, true
	case "hc_ffn_inject.weight":
		return jlm.RoleHCFFNInject, true
	case "output_hc_norm.weight":
		return jlm.RoleHCHeadNorm, true
	case "output_hc_down.weight":
		return jlm.RoleHCHeadDown, true
	case "output_hc_up.weight":
		return jlm.RoleHCHeadUp, true
	// The prediction block's own head mixer: the trunk's roles, at the
	// block's index rather than the dense one.
	case "nextn.hc_head_norm.weight":
		return jlm.RoleHCHeadNorm, true
	case "nextn.hc_head_down.weight":
		return jlm.RoleHCHeadDown, true
	case "nextn.hc_head_up.weight":
		return jlm.RoleHCHeadUp, true
	// Its n-gram block. The table itself is per_layer_token_embd, Gemma's
	// name, retargeted (retarget) to RoleNGramTable.
	case "ple_key.weight":
		return jlm.RoleNGramKey, true
	case "ple_value.weight":
		return jlm.RoleNGramValue, true
	case "ple_norm_key.weight":
		return jlm.RoleNGramNormKey, true
	case "ple_norm_query.weight":
		return jlm.RoleNGramNormQuery, true
	case "ple_norm_conv.weight":
		return jlm.RoleNGramNormConv, true
	case "ple_conv1d.weight":
		return jlm.RoleNGramConv, true
	}
	return 0, false
}

// stateQSA states QSA (query-sparse attention) on the attention blocks from
// the ratio each block compresses by, rs (llama.cpp's qwen4exp.cpp and
// transformers' Qwen4ExpForCausalLM, the qsa_ratios statement; none stated
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
		// A prediction block that carries the QSA indexer is a QSA block at
		// the model's ratio. unsloth's published heads state 0 for it where
		// llama.cpp's converter writes the ratio (conversion/qwen4exp.py:
		// "the MTP block is a full-attention QSA layer too"), and the
		// release's config makes it full attention, which this model runs
		// as QSA; the tensors decide (RULE 7m).
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
	// llama.cpp's loader takes the end-of-sequence id when a file predates
	// the key; the image id is what transformers hashes at a picture's rows.
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
	// llama.cpp reads the arrays at their stated length and requires at
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
