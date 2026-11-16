package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
)

// yarnMscaleRatio folds mistral3's YaRN magnitude into AttnFactor.
//
// Its converter writes mscale_all_dim itself into yarn_log_multiplier, not
// DeepSeek's 0.1*mscale_all_dim, and llama.cpp's context setup (taking mscale
// as 1) scales cos and sin by mscale(factor, 1) / mscale(factor, m), where
// mscale(s, k) = 0.1*k*ln(s) + 1 -- transformers' attention_factor for a
// config carrying both mscale and mscale_all_dim. Ministral 3's are 1 and 1, so
// the factor is exactly one. Nothing reaches the softmax: mistral3.cpp scales
// the scores by 1/sqrt(head_dim) or attention.scale, so YarnLogMul stays zero.
func yarnMscaleRatio(f *meta.File, c *Config, factor float64) {
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

// attnScaleKey reads attention.scale, which a graph takes in place of
// 1/sqrt(head_dim) when a file states it.
func attnScaleKey(f *meta.File, c *Config) error {
	s := f.FloatKey("attention.scale", 0)
	if !(s >= 0) || math.IsInf(s, 0) {
		return fmt.Errorf("attention.scale %g", s)
	}
	c.AttnScale = float32(s)
	return nil
}
