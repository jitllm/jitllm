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

// Microsoft's phi-2 (and phi-1.5), llama.cpp's phi2.cpp, whose container
// classic.json states:
//
//	attn_norm (LayerNorm + bias) feeds both branches    -> Parallel, no ffn_norm
//	q, k, v, o biased; NEOX rotary on n_rot of head_dim -> RopeNeox, NRot
//	ffn_up (+bias), GELU, ffn_down (+bias), no gate    -> the absent gate
//	output_norm (+bias), output (+bias)                -> the head's two biases
//
// llama.cpp scales q by 1/sqrt(head_dim) before the product "to avoid
// precision issues" and runs the softmax at scale 1; that is the same function
// as scaling the scores, which is what this engine does.

// Cohere's graph, llama.cpp's command-r.cpp and cohere2.cpp, whose containers
// classic.json states:
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
// head) is refused by name in configOfFile: running without it would be
// fluent and wrong.

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
