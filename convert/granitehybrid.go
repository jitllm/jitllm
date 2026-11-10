package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// graniteHybridConfig is graniteConfig's four scales for Granite 4.0-H, whose
// attention layers carry no rotary at all: llama.cpp's converter writes
// rope.scaling.finetuned false for every hybrid but Bamba, and its builder
// skips the rotation on exactly that switch.
func graniteHybridConfig(f *meta.File, c *Config) error {
	ls, ok := f.Key("logit_scale")
	if !ok {
		return fmt.Errorf("no logit_scale, which every Granite carries")
	}
	v, ok := ls.Float()
	if !ok || !(v > 0) {
		return fmt.Errorf("logit_scale %v", ls.Short())
	}
	c.LogitScale = float32(v)
	c.EmbdScale = float32(f.FloatKey("embedding_scale", 1))
	c.ResidualScale = float32(f.FloatKey("residual_scale", 0))
	c.AttnScale = float32(f.FloatKey("attention.scale", 0))
	for _, s := range []float32{c.EmbdScale, c.ResidualScale, c.AttnScale} {
		if !(s >= 0) || math.IsInf(float64(s), 0) {
			return fmt.Errorf("scale %g", s)
		}
	}
	rope := true
	if kv, ok := f.Key("rope.scaling.finetuned"); ok {
		if b, ok := kv.Uint(); ok && b == 0 {
			rope = false
		}
	}
	if !rope {
		c.Flags |= jlm.FlagNoPosEnc
	}
	return nil
}
