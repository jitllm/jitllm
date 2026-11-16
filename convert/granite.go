package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
)

// gemmaAttnScale sets the score scale of a 27B gemma2/gemma3.
//
// The 27B is the one gemma whose score is not 1/sqrt(head_dim): its
// query_pre_attn_scalar is n_embd/n_head. The GGUF states neither the scalar
// nor the size, so like llama.cpp this keys it on the layer count (46, 62).
func gemmaAttnScale(c *Config) {
	if (c.Arch == ArchGemma2 && c.NLayer == 46) || (c.Arch == ArchGemma3 && c.NLayer == 62) {
		c.AttnScale = float32(1 / math.Sqrt(float64(c.NEmbd/c.NHead)))
	}
}

// swaPatternOf reads which layers slide, into c.SWAPeriod, from the file's
// attention.sliding_window_pattern or the architecture's period def.
//
// The key is either a period or a per-layer bool array. The container says
// it only as a period (Config.SWA(il) is il%period < period-1), so an array is
// accepted exactly when it is that rule for some period.
func swaPatternOf(f *meta.File, c *Config, def uint32) error {
	c.SWAPeriod = def
	kv, ok := f.Key("attention.sliding_window_pattern")
	if !ok {
		return nil
	}
	if v, ok := kv.Uint(); ok && kv.Type != meta.Array {
		if v < 2 {
			return fmt.Errorf("attention.sliding_window_pattern %d", v)
		}
		c.SWAPeriod = uint32(v)
		return nil
	}
	if kv.Type != meta.Array || kv.Elem != meta.Bool || uint64(len(kv.Raw)) < kv.N {
		return fmt.Errorf("attention.sliding_window_pattern is %s: %w", kv.Short(), ErrNotImplemented)
	}
	swa := kv.Raw[:kv.N]
	period := uint32(0)
	for i, b := range swa {
		if b == 0 {
			period = uint32(i) + 1
			break
		}
	}
	if period == 0 {
		period = uint32(len(swa)) + 1 // every layer slides: allLocal
	}
	for i, b := range swa {
		if want := uint32(i)%period < period-1; (b != 0) != want {
			return fmt.Errorf("attention.sliding_window_pattern is not periodic (layer %d): %w",
				i, ErrNotImplemented)
		}
	}
	c.SWAPeriod = period
	return nil
}
