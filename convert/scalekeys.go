package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
)

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
