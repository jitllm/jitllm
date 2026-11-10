package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// The dense llama-family architectures (ArchSmolLM3 and its siblings).
// Each is llama's block with a few switches the graph already has; this is
// where the switches are read, from llama.cpp's builder for each
// (src/models/<arch>.cpp) checked against transformers' own class.

// denseConfig reads the dense llama-family architectures into c.
func denseConfig(f *meta.File, c *Config) error {
	switch c.Arch {
	case ArchSmolLM3:
		// smollm3.cpp: llama's block, and use_rope = (il+1) % 4 != 0. The step
		// is a constant there (n_no_rope_layer_step), as it is transformers'
		// default no_rope_layer_interval, and no GGUF key carries it. The period
		// with no window names the NoPE layers and slides none.
		c.SWAWindow, c.SWAPeriod = 0, smolLM3NoPEStep
		c.Flags |= jlm.FlagNoPEGlobal
		return attnScaleKey(f, c)
	case ArchArcee:
		// arcee.cpp: llama's block, build_ffn(up, -, down, LLM_FFN_RELU_SQR,
		// LLM_FFN_SEQ) -- no gate tensor, squared ReLU.
		c.Flags |= jlm.FlagReLU2
		if _, ok := f.Get("blk.0.ffn_gate.weight"); ok {
			return fmt.Errorf("an arcee block with a gate tensor; its FFN is ungated: %w", ErrNotImplemented)
		}
		return attnScaleKey(f, c)
	case ArchSeedOSS:
		// seed-oss.cpp: NEOX rotary, optional q/k/v biases (tensors), and the
		// FFN's pre-norm stored as post_attention_norm (retarget).
		c.Flags |= jlm.FlagRopeNeox
		return attnScaleKey(f, c)
	case ArchOLMo2:
		// olmo2.cpp: no pre-norm; q and k RMSNormed over the whole projection
		// ({n_embd} and {n_head_kv*head_dim}), NEOX rotary, a post-norm after
		// attention and after the FFN. OLMo 3 adds a sliding window on three
		// layers in four, on which llama.cpp turns YaRN off.
		// The k norm is the k projection's width, n_head_kv*head_dim, so a GQA
		// OLMo 2 (the 32B) norms k over fewer channels than q.
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagQKNormWide
		if c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0)); c.SWAWindow > 0 {
			if err := swaPatternOf(f, c, 4); err != nil {
				return err
			}
			// OLMo 3: the sliding layers rotate with no scaling at all, at
			// rope.freq_base_swa where the file states one -- a second rotary
			// table whose magnitude is one (ArchOLMo3), YaRN reaching the
			// global layers only, as it already does.
			c.Arch = ArchOLMo3
			c.RopeBaseSWA = float32(f.FloatKey("rope.freq_base_swa", float64(c.RopeBase)))
		}
		return nil
	case ArchExaone4:
		// exaone4.cpp: OLMo 2's post-norm block with a per-head q/k norm
		// ({head_dim}) and NEOX rotary. The 32B (64 layers) slides a 4096 window
		// on three layers in four whether or not the file says so, and rotates
		// only those; every layer of a model with no window rotates.
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm
		win := uint32(0)
		if c.NLayer == 64 {
			win = 4096
		}
		if c.SWAWindow = uint32(f.UintKey("attention.sliding_window", uint64(win))); c.SWAWindow > 0 {
			if err := swaPatternOf(f, c, 4); err != nil {
				return err
			}
			c.Flags |= jlm.FlagNoPEGlobal
			if b := float32(f.FloatKey("rope.freq_base_swa", float64(c.RopeBase))); b != c.RopeBase {
				c.RopeBaseSWA = b
			}
		}
		return nil
	case ArchMistral3:
		// mistral3.cpp: llama's block, and when attention.temperature_scale is
		// set every layer's q is scaled by 1 + s*ln(1 + floor(pos/n)), n the
		// YaRN original context and no offset (llama4's form without the NoPE
		// selection). Its MoE branch is a softmax top-k, read like any other.
		if s := f.FloatKey("attention.temperature_scale", 0); s != 0 {
			n := f.UintKey("rope.scaling.original_context_length", 0)
			if n == 0 {
				return fmt.Errorf("attention.temperature_scale %g with no original context "+
					"to floor positions by", s)
			}
			c.AttnTempScale, c.AttnTempFloor, c.AttnTempOffset = float32(s), uint32(n), 0
		}
		return attnScaleKey(f, c)
	}
	return fmt.Errorf("denseConfig: %v is not a dense llama-family architecture", c.Arch)
}

// mistral3Yarn folds mistral3's YaRN magnitude into AttnFactor.
//
// Its converter writes mscale_all_dim itself into yarn_log_multiplier, not
// DeepSeek's 0.1*mscale_all_dim, and llama.cpp's context setup (taking mscale
// as 1) scales cos and sin by mscale(factor, 1) / mscale(factor, m), where
// mscale(s, k) = 0.1*k*ln(s) + 1 -- transformers' attention_factor for a
// config carrying both mscale and mscale_all_dim. Ministral 3's are 1 and 1, so
// the factor is exactly one. Nothing reaches the softmax: mistral3.cpp scales
// the scores by 1/sqrt(head_dim) or attention.scale, so YarnLogMul stays zero.
func mistral3Yarn(f *meta.File, c *Config, factor float64) {
	ms := func(k float64) float64 { return 0.1*k*math.Log(factor) + 1 }
	// With no key, llama.cpp's factor is mscale(factor, 1) alone.
	div := 1.0
	if kv, ok := f.Key("rope.scaling.yarn_log_multiplier"); ok {
		if m, ok := kv.Float(); ok && m != 0 {
			div = ms(m)
		}
	}
	c.AttnFactor *= float32(ms(1) / div)
}

// smolLM3NoPEStep is llama.cpp's n_no_rope_layer_step for SmolLM3.
const smolLM3NoPEStep = 4

// attnScaleKey reads attention.scale, which these builders take in place of
// 1/sqrt(head_dim) when a file states it.
func attnScaleKey(f *meta.File, c *Config) error {
	s := f.FloatKey("attention.scale", 0)
	if !(s >= 0) || math.IsInf(s, 0) {
		return fmt.Errorf("attention.scale %g", s)
	}
	c.AttnScale = float32(s)
	return nil
}
