package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

func init() {
	// Qwen3.8-Flash-Next's hyper-connections, per block and at the head.
	roleOf["hc_attn_norm.weight"] = jlm.RoleHCAttnNorm
	roleOf["hc_attn_down.weight"] = jlm.RoleHCAttnDown
	roleOf["hc_attn_up.weight"] = jlm.RoleHCAttnUp
	roleOf["hc_attn_inject.weight"] = jlm.RoleHCAttnInject
	roleOf["hc_ffn_norm.weight"] = jlm.RoleHCFFNNorm
	roleOf["hc_ffn_down.weight"] = jlm.RoleHCFFNDown
	roleOf["hc_ffn_up.weight"] = jlm.RoleHCFFNUp
	roleOf["hc_ffn_inject.weight"] = jlm.RoleHCFFNInject
	roleOf["output_hc_norm.weight"] = jlm.RoleHCHeadNorm
	roleOf["output_hc_down.weight"] = jlm.RoleHCHeadDown
	roleOf["output_hc_up.weight"] = jlm.RoleHCHeadUp
	// The prediction block's own head mixer: the trunk's roles, at the
	// block's index rather than the dense one.
	roleOf["nextn.hc_head_norm.weight"] = jlm.RoleHCHeadNorm
	roleOf["nextn.hc_head_down.weight"] = jlm.RoleHCHeadDown
	roleOf["nextn.hc_head_up.weight"] = jlm.RoleHCHeadUp
	// Its n-gram block. The table itself is per_layer_token_embd, Gemma's
	// name, retargeted (retarget) to RoleNGramTable.
	roleOf["ple_key.weight"] = jlm.RoleNGramKey
	roleOf["ple_value.weight"] = jlm.RoleNGramValue
	roleOf["ple_norm_key.weight"] = jlm.RoleNGramNormKey
	roleOf["ple_norm_query.weight"] = jlm.RoleNGramNormQuery
	roleOf["ple_norm_conv.weight"] = jlm.RoleNGramNormConv
	roleOf["ple_conv1d.weight"] = jlm.RoleNGramConv
}

// qwen4expConfig is llama.cpp's qwen4exp.cpp and transformers'
// Qwen4ExpForCausalLM, text only, on top of what the qwen3next case reads
// (the delta rule's geometry, the gated query, per-head q/k norms) and
// qwen35Of (the tiled value heads, the M-RoPE sections):
//
//   - HCMult streams mixed through a HCLowRank gate (hyper_connection.*);
//   - the delta rule's output norm gated by sigmoid, not silu: llama.cpp's
//     graph hardcodes it and the GGUF carries no key, transformers reads
//     output_gate_type, which the release states as sigmoid
//     (Flag2DeltaSigmoidGate);
//   - QSA on the attention blocks: the indexer's heads, width and token
//     budget, and one block size, which compress_ratios states per block
//     (the ratio on every attention block, zero on every linear one; any
//     other pattern is refused);
//   - the n-gram embedding on one linear block (ple.*), with the hash's
//     constants carried exactly.
//
// Refused: no hyper-connections or a count of one (every reference refuses
// it), a QSA block whose ratio differs from another's or sits on a linear
// block, more than one n-gram block or one on an attention block, and
// constants too short for the heads they describe.
func qwen4expConfig(f *meta.File, c *jlm.Config) error {
	c.Flags2 |= jlm.Flag2DeltaSigmoidGate
	if err := partialRotary(c); err != nil {
		return err
	}
	c.HCMult = uint32(f.UintKey("hyper_connection.count", 0))
	c.HCLowRank = uint32(f.UintKey("hyper_connection.low_rank", 0))
	if c.HCMult < 2 || c.HCLowRank == 0 {
		return fmt.Errorf("hyper-connections: %d streams at rank %d", c.HCMult, c.HCLowRank)
	}
	c.IdxHeads = uint32(f.UintKey("attention.indexer.head_count", 0))
	c.IdxHeadDim = uint32(f.UintKey("attention.indexer.key_length", 0))
	c.IdxTopK = uint32(f.UintKey("attention.indexer.top_k", 0))
	if kv, ok := f.Key("attention.compress_ratios"); ok {
		rs, err := kv.Uint64s()
		if err != nil {
			return fmt.Errorf("attention.compress_ratios: %w", err)
		}
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
	return ngramConfig(f, c)
}

// ngramConfig reads the n-gram embedding's keys, the hash constants as the
// unsigned integers they are.
func ngramConfig(f *meta.File, c *jlm.Config) error {
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
