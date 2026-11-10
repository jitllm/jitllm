package convert

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// hunyuanConfig is llama.cpp's hunyuan-moe.cpp / hunyuan-vl.cpp (which serves
// hunyuan-dense) and transformers' HunYuanMoEV1ForCausalLM and
// HunYuanDenseV1ForCausalLM:
//
//	no biases on a published checkpoint      -> tensors
//	NEOX rotary over the whole head           -> FlagRopeNeox
//	per-head q/k RMSNorm AFTER the rotary     -> FlagQKNorm | FlagQKNormPostRope,
//	                                             the k weight folded into q's
//	                                             (foldPostRopeQKNorm)
//	softmax top-k, renormalised, no scale     -> the defaults, and a file that
//	                                             states otherwise is refused
//	one ungated shared expert in every block  -> its width off the tensor
//
// The gate, the renormalisation and the absent scale are read from no key:
// llama.cpp's builder passes softmax and true as literals and never loads
// expert_weights_scale for this arch, and HunYuanMoEV1Gate hardcodes
// F.softmax and the division (RULE 7m: the two agree, so a file disagreeing
// with both is refused rather than obeyed).
//
// The NTK-alpha rotary base (rope_theta * alpha^(d/(d-2))) is already folded
// into rope.freq_base by llama.cpp's converter; a GGUF says nothing else.
func hunyuanConfig(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagQKNormPostRope
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 1 {
			return fmt.Errorf("expert_gating_func %d, and HunYuanMoEV1Gate is softmax: %w",
				v, ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("expert_weights_norm"); ok {
		if v, _ := kv.Uint(); v == 0 {
			return fmt.Errorf("expert_weights_norm false, and HunYuanMoEV1Gate always "+
				"renormalises: %w", ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("expert_weights_scale"); ok {
		if v, _ := kv.Float(); v != 0 && v != 1 {
			return fmt.Errorf("expert_weights_scale %g, and neither llama.cpp nor "+
				"transformers scales a Hunyuan mixture: %w", v, ErrNotImplemented)
		}
	}
	return nil
}

// hunyuanVLConfig is HunyuanVL's text model: hunyuan-dense's (hunyuanConfig)
// under XD-RoPE's four sections (rope.dimension_sections, per element; see
// ArchHunyuanVL). llama.cpp's converter writes the NTK alpha beside
// rope.freq_base rather than folding it in, and its loader and transformers'
// HunYuanVLRotaryEmbedding both take base = theta * alpha^(d/(d-2)) over the
// rotary's width.
func hunyuanVLConfig(f *meta.File, name string, c *Config) error {
	if err := hunyuanConfig(f, c); err != nil {
		return fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := mropeSectionsOf(f, name, c); err != nil {
		return err
	}
	// transformers' XD-RoPE turns the whole head: a file turning part of it
	// states a rotary its reference never runs.
	if c.NRot != c.HeadDim {
		return fmt.Errorf("convert: %s: XD-RoPE over %d of a head's %d dimensions, where the reference turns the whole head",
			name, c.NRot, c.HeadDim)
	}
	if alpha := f.FloatKey("rope.scaling.alpha", 0); alpha > 0 {
		d := float64(c.NRot)
		c.RopeBase = float32(float64(c.RopeBase) * math.Pow(alpha, d/(d-2)))
	}
	return nil
}

// foldPostRopeQKNorm moves k's q/k-norm weight into q's, for a model whose
// per-head norm follows the rotary (jlm.FlagQKNormPostRope).
//
// The model computes, per head, q' = w_q * R(q)/rms(q) and k' = w_k *
// R(k)/rms(k) -- a rotation keeps each head's RMS, so rms(R(x)) = rms(x) --
// and the score sum_d q'_d k'_d = sum_d (w_q w_k)_d R(q)_d R(k)_d / (rms(q)
// rms(k)). Both weights are per dimension and shared by every head, so a GQA
// pairing changes nothing. Storing w_q*w_k as q's weight and ones as k's gives
// the same scores with an unweighted k, which commutes with the rotary and so
// runs where every other q/k norm runs, before it; only q's norm moves. The
// value path is untouched.
func foldPostRopeQKNorm(s *jlm.Source) error {
	if !s.Config.Flags.Has(jlm.FlagQKNormPostRope) {
		return nil
	}
	type pair struct{ q, k int }
	blocks := map[int32]*pair{}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleAttnQNorm && t.Role != jlm.RoleAttnKNorm {
			continue
		}
		p := blocks[t.Block]
		if p == nil {
			p = &pair{-1, -1}
			blocks[t.Block] = p
		}
		if t.Role == jlm.RoleAttnQNorm {
			p.q = i
		} else {
			p.k = i
		}
	}
	for b, p := range blocks {
		if p.q < 0 || p.k < 0 {
			return fmt.Errorf("convert: block %d has one q/k norm of two", b)
		}
		q, err := normF32(&s.Tensors[p.q])
		if err != nil {
			return err
		}
		k, err := normF32(&s.Tensors[p.k])
		if err != nil {
			return err
		}
		if len(q) != len(k) {
			return fmt.Errorf("convert: block %d: q norm %d wide, k norm %d", b, len(q), len(k))
		}
		ones := make([]float32, len(k))
		for i := range q {
			q[i] *= k[i]
			ones[i] = 1
		}
		setF32(&s.Tensors[p.q], q)
		setF32(&s.Tensors[p.k], ones)
	}
	return nil
}

// normF32 decodes a norm vector to f32, whatever the source stored.
func normF32(t *jlm.Tensor) ([]float32, error) {
	typ, ok := jlm.SourceType(t.Type)
	if !ok {
		return nil, fmt.Errorf("convert: %s is %v, which has no dequantizer", t.Name, t.Type)
	}
	b, err := t.Bytes()
	if err != nil {
		return nil, err
	}
	n := 1
	for d := 0; d < int(t.NDim); d++ {
		n *= int(t.Dims[d])
	}
	out := make([]float32, n)
	if err := quant.Dequant32(typ, b, out); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
	}
	return out, nil
}

// setF32 replaces a tensor's bytes with v as F32.
func setF32(t *jlm.Tensor, v []float32) {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	t.Type, t.Data, t.Load = jlm.TypeF32, b, nil
}
