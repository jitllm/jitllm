package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// hfLongRope is phi3's LongRoPE as config.json states it: a factor per rotary
// pair for positions below original_max_position_embeddings and another past
// it, and the cos/sin magnitude.
type hfLongRope struct {
	Type       string    `json:"type"`
	RopeType   string    `json:"rope_type"`
	Short      []float64 `json:"short_factor"`
	Long       []float64 `json:"long_factor"`
	ShortScale *float64  `json:"short_mscale"`
	LongScale  *float64  `json:"long_mscale"`
	AttnFactor *float64  `json:"attention_factor"`
	OrigCtx    uint32    `json:"original_max_position_embeddings"`
}

// hfLongRopeConsume reads a LongRoPE rope_scaling and removes it from the
// config, so llamaHFConfig's refusal of every scaling does not fire on it.
// Any other scaling is left for that refusal.
func hfLongRopeConsume(c *hfConfig) (*hfLongRope, error) {
	if len(c.RopeScaling) == 0 || string(c.RopeScaling) == "null" {
		return nil, nil
	}
	var lr hfLongRope
	if err := json.Unmarshal(c.RopeScaling, &lr); err != nil {
		return nil, fmt.Errorf("convert: %s: rope_scaling: %w", c.path, err)
	}
	kind := lr.RopeType
	if kind == "" {
		kind = lr.Type
	}
	if kind != "longrope" && kind != "su" {
		return nil, nil
	}
	if lr.OrigCtx == 0 {
		if raw, ok := c.keys["original_max_position_embeddings"]; ok {
			if err := json.Unmarshal(raw, &lr.OrigCtx); err != nil {
				return nil, fmt.Errorf("convert: %s: original_max_position_embeddings: %w", c.path, err)
			}
		}
	}
	c.RopeScaling = nil
	return &lr, nil
}

// apply writes the factors as the rope_factors_short/long tensors the GGUF
// carries (convert_hf_to_gguf.py writes the same two vectors), the original
// context the engine keys the long factors on (model.RopeFor), and the
// magnitude into AttnFactor. The magnitude is the class's: PhimoeRotaryEmbedding multiplies
// cos and sin by short_mscale, and transformers' LongRoPE by attention_factor,
// or sqrt(1 + ln(ctx/orig)/ln(orig)) where neither is stated.
func (lr *hfLongRope) apply(c *hfConfig, cfg *Config) error {
	ts, err := lr.state(c.path, cfg)
	if err != nil {
		return err
	}
	c.extra = append(c.extra, ts...)
	return nil
}

// state is apply's statement for a config read from path: the two factor
// vectors, with the original context and the magnitude set on cfg.
func (lr *hfLongRope) state(path string, cfg *Config) ([]jlm.Tensor, error) {
	pairs := int(cfg.NRot / 2)
	if len(lr.Short) != pairs || len(lr.Long) != pairs {
		return nil, fmt.Errorf("convert: %s: LongRoPE factors %d short and %d long for %d rotary pairs",
			path, len(lr.Short), len(lr.Long), pairs)
	}
	if lr.OrigCtx == 0 {
		return nil, fmt.Errorf("convert: %s: LongRoPE without original_max_position_embeddings", path)
	}
	cfg.YarnOrigCtx = lr.OrigCtx
	switch {
	case lr.ShortScale != nil:
		cfg.AttnFactor = float32(*lr.ShortScale)
	case lr.AttnFactor != nil:
		cfg.AttnFactor = float32(*lr.AttnFactor)
	case cfg.NCtx > lr.OrigCtx:
		s := float64(cfg.NCtx) / float64(lr.OrigCtx)
		cfg.AttnFactor = float32(math.Sqrt(1 + math.Log(s)/math.Log(float64(lr.OrigCtx))))
	}
	// PhimoeRotaryEmbedding takes long_mscale past the original context; the
	// container states one magnitude for every position.
	if lr.LongScale != nil && float32(*lr.LongScale) != cfg.AttnFactor {
		return nil, fmt.Errorf("convert: %s: long_mscale %g past the original context against %g before it: %w",
			path, *lr.LongScale, cfg.AttnFactor, ErrNotImplemented)
	}
	vec := func(role jlm.Role, name string, f []float64) jlm.Tensor {
		b := make([]byte, 4*len(f))
		for i, x := range f {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(x)))
		}
		return jlm.Tensor{Role: role, Block: jlm.DenseBlock, Index: -1, Type: jlm.TypeF32, NDim: 1,
			Dims: [4]uint64{uint64(len(f))}, Data: b, Name: name}
	}
	return []jlm.Tensor{
		vec(jlm.RoleRopeFactorsShort, "rope_factors_short.weight", lr.Short),
		vec(jlm.RoleRopeFactorsLong, "rope_factors_long.weight", lr.Long)}, nil
}
