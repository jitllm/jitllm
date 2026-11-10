package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// The classic-transformer blocks (C6) have GPT-2's shape rather than llama's.
// Each architecture here is a set of switches over one kit the graph already
// runs: a LayerNorm (FlagLayerNorm) with or without a bias, a parallel
// residual (FlagParallel), an ungated FFN (the absent gate tensor), biases on
// every projection, partial rotary (NRot).

// layerNormEps reads the LayerNorm epsilon, which is a different key from the
// RMSNorm one: these files carry attention.layer_norm_epsilon and no
// layer_norm_rms_epsilon.
func layerNormEps(f *meta.File, c *Config) {
	c.RMSEps = float32(f.FloatKey("attention.layer_norm_epsilon", 1e-5))
}

// partialRotary checks a rotary width that is not the whole head: it must be
// even (the pairs) and fit inside the head. The rotated dimensions lead and
// the tail passes through, which is what every C6 graph's ggml_rope_ext does
// with n_rot < head_dim.
func partialRotary(c *Config) error {
	if c.NRot == 0 || c.NRot%2 != 0 || c.NRot > c.HeadDim {
		return fmt.Errorf("rotary width %d on a %d-wide head", c.NRot, c.HeadDim)
	}
	return nil
}

// phi2Config is Microsoft's phi-2 (and phi-1.5), llama.cpp's phi2.cpp:
//
//	attn_norm (LayerNorm + bias) feeds both branches    -> Parallel, no ffn_norm
//	q, k, v, o biased; NEOX rotary on n_rot of head_dim -> RopeNeox, NRot
//	ffn_up (+bias), GELU, ffn_down (+bias), no gate    -> the absent gate
//	output_norm (+bias), output (+bias)                -> the head's two biases
//
// llama.cpp scales q by 1/sqrt(head_dim) before the product "to avoid
// precision issues" and runs the softmax at scale 1; that is the same function
// as scaling the scores, which is what this engine does.
func phi2Config(f *meta.File, c *Config) error {
	layerNormEps(f, c)
	c.Flags |= jlm.FlagLayerNorm | jlm.FlagParallel | jlm.FlagRopeNeox | jlm.FlagGELU
	if err := partialRotary(c); err != nil {
		return err
	}
	return nil
}

// starcoderConfig is GPTBigCode, llama.cpp's starcoder.cpp:
//
//	inpL = tok_embd[t] + position_embd[pos]       -> the position table
//	attn_norm (LayerNorm + bias), attn_qkv (+bias) -> LayerNorm, split at
//	    conversion into q, k, v with ONE kv head      conversion (MQA)
//	no rotary                                      -> FlagNoPosEnc
//	ffn_norm (+bias), up (+bias), GELU, down (+bias), sequential residual
//	output_norm (+bias); output is tok_embd when absent
//
// The position table's length bounds the context: a position past it has no
// embedding, so NCtx is clamped to the table's rows.
func starcoderConfig(f *meta.File, c *Config) error {
	layerNormEps(f, c)
	c.Flags |= jlm.FlagLayerNorm | jlm.FlagNoPosEnc | jlm.FlagGELU
	t, ok := f.Get("position_embd.weight")
	if !ok || len(t.Dims) < 2 {
		return fmt.Errorf("no position_embd.weight, which a starcoder carries")
	}
	if uint32(t.Dims[0]) != c.NEmbd {
		return fmt.Errorf("position_embd is %d wide, the model %d", t.Dims[0], c.NEmbd)
	}
	if uint32(t.Dims[1]) < c.NCtx {
		c.NCtx = uint32(t.Dims[1])
	}
	return nil
}

// commandRConfig is Cohere's graph, llama.cpp's command-r.cpp and cohere2.cpp:
//
//	attn_norm (LayerNorm, NO bias) feeds both branches  -> Parallel
//	q, k, v, o unbiased; adjacent-pair rotary            -> NORM rope
//	ffn gate/up/down, SwiGLU                             -> the gated FFN
//	output_norm (LayerNorm, no bias); head = tok_embd    -> tied
//	logits *= logit_scale                                -> 1/s in LogitScale
//
// and for cohere2 a sliding window on the layers set_swa_pattern marks (period
// 4 unless the file says), with the rotary only on those: the global layer
// attends with no positional encoding (FlagNoPEGlobal).
//
// The logit scale is stored as its reciprocal because LogitScale divides
// (Granite's convention). Cohere's scales are powers of two, so 1/s is exact.
// A positive scale moves no argmax; the transformers golden gates it.
//
// command-r-plus's per-head q/k LayerNorm ({head_dim, n_head}, a weight per
// head) is refused by name: running without it would be fluent and wrong.
func commandRConfig(f *meta.File, c *Config, cohere2 bool) error {
	layerNormEps(f, c)
	c.Flags |= jlm.FlagLayerNorm | jlm.FlagParallel
	if _, ok := f.Get("blk.0.attn_q_norm.weight"); ok {
		return fmt.Errorf("a per-head q/k LayerNorm (command-r-plus): %w", ErrNotImplemented)
	}
	s := f.FloatKey("logit_scale", 0)
	switch {
	case s < 0 || math.IsInf(s, 0) || math.IsNaN(s):
		return fmt.Errorf("logit_scale %g", s)
	case s != 0 && s != 1:
		c.LogitScale = float32(1 / s)
	}
	if !cohere2 {
		return nil
	}
	if c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0)); c.SWAWindow == 0 {
		return fmt.Errorf("no attention.sliding_window, which a cohere2 carries")
	}
	if err := swaPatternOf(f, c, 4); err != nil {
		return err
	}
	c.Flags |= jlm.FlagNoPEGlobal
	// The sliding layers rotate at the file's local base when it states one;
	// llama.cpp's default is the global base, which one table serves.
	if b := float32(f.FloatKey("rope.freq_base_swa", float64(c.RopeBase))); b != c.RopeBase {
		c.RopeBaseSWA = b
	}
	return nil
}

// stableLMConfig is llama.cpp's stablelm.cpp:
//
//	attn_norm (LayerNorm + bias); q, k, v biased, o not     -> LayerNorm
//	NEOX rotary on rope.dimension_count of head_dim          -> RopeNeox, NRot
//	ffn gate/up/down, SwiGLU                                 -> the gated FFN
//	ffn_norm present: sequential; absent: parallel from the  -> FlagParallel
//	    attention's normed row (StableLM 2 12B)
//	output_norm (+bias), output untied
//
// The residual is read off the tensor, as llama.cpp reads it
// ("if (model.layers[il].ffn_norm)"); use_parallel_residual is not consulted,
// since some files say true while carrying an ffn_norm in every block.
func stableLMConfig(f *meta.File, c *Config) error {
	layerNormEps(f, c)
	c.Flags |= jlm.FlagLayerNorm | jlm.FlagRopeNeox
	if _, ok := f.Get("blk.0.attn_q_norm.weight"); ok {
		return fmt.Errorf("a per-head q/k LayerNorm (StableLM 2 12B): %w", ErrNotImplemented)
	}
	if _, ok := f.Get("blk.0.ffn_norm.weight"); !ok {
		c.Flags |= jlm.FlagParallel
	}
	return partialRotary(c)
}

// dbrxConfig is Databricks' DBRX, llama.cpp's dbrx.cpp:
//
//	attn_norm (LayerNorm, NO bias)                -> LayerNorm
//	attn_qkv, split at conversion; q, k, v each
//	    clamped to [-clamp_kqv, clamp_kqv]        -> ClampKQV
//	NEOX rotary over the whole head               -> RopeNeox
//	attn_output_norm (LayerNorm, no bias) before
//	    the FFN, sequential residual              -> the FFN norm (retarget)
//	a softmax top-k mixture of SwiGLU experts,
//	    renormalised over the chosen k            -> expert_count, as any mixture
//	output_norm (LayerNorm, no bias), output
//
// The clamp must be positive and finite: zero would mean no clamp, which is
// not DBRX, and a negative or NaN bound clamps nothing sensibly.
func dbrxConfig(f *meta.File, c *Config) error {
	layerNormEps(f, c)
	c.Flags |= jlm.FlagLayerNorm | jlm.FlagRopeNeox
	v := f.FloatKey("attention.clamp_kqv", 0)
	if !(v > 0) || math.IsInf(v, 0) {
		return fmt.Errorf("attention.clamp_kqv %g; a dbrx clamps q, k and v", v)
	}
	c.ClampKQV = float32(v)
	if f.UintKey("expert_count", 0) == 0 {
		return fmt.Errorf("no expert_count; every dbrx block is a mixture")
	}
	return nil
}
