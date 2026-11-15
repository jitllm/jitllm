package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// The published rope scaling methods the rope_scaling statement implements
// (mjRopeScaling), each once, for every model that states it.

// ropeFreqsTensor is per-pair rotary factors as the container carries them,
// rope_freqs.weight beside the blocks.
func ropeFreqsTensor(f []float32) jlm.Tensor {
	b := make([]byte, 4*len(f))
	for i, x := range f {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return jlm.Tensor{Role: jlm.RoleRopeFreqs, Block: jlm.DenseBlock, Index: -1,
		Type: jlm.TypeF32, NDim: 1, Dims: [4]uint64{uint64(len(f))}, Data: b, Name: "rope_freqs.weight"}
}

// hfLlama3 is llama 3's rope_scaling.
type hfLlama3 struct {
	Type     string   `json:"type"`
	RopeType string   `json:"rope_type"`
	Factor   *float64 `json:"factor"`
	Low      *float64 `json:"low_freq_factor"`
	High     *float64 `json:"high_freq_factor"`
	OrigCtx  *float64 `json:"original_max_position_embeddings"`
}

// llama3RopeOf reads a rope scaling object, transformers' rope_parameters or
// rope_scaling: for llama3's method, the function of the rotated width and the
// base that computes its per-pair factors; nil for any other method. factor,
// low_freq_factor and high_freq_factor are required, as transformers'
// _compute_llama3_parameters reads them. The original context is origTop, a
// top-level original_max_position_embeddings, which standardize_rope_params
// prefers to the object's; then the object's; then maxPos,
// max_position_embeddings.
func llama3RopeOf(obj json.RawMessage, origTop, maxPos *float64) (func(rotated uint32, base float64) []float32, error) {
	var r hfLlama3
	if err := json.Unmarshal(obj, &r); err != nil {
		return nil, err
	}
	if r.RopeType != "llama3" && (r.RopeType != "" || r.Type != "llama3") {
		return nil, nil
	}
	for _, k := range []struct {
		name string
		v    *float64
	}{{"factor", r.Factor}, {"low_freq_factor", r.Low}, {"high_freq_factor", r.High}} {
		if k.v == nil {
			return nil, fmt.Errorf("llama3 states no %s", k.name)
		}
	}
	orig := r.OrigCtx
	if origTop != nil {
		orig = origTop
	}
	if orig == nil {
		orig = maxPos
	}
	if orig == nil {
		return nil, fmt.Errorf("llama3 states no original_max_position_embeddings, and the config no max_position_embeddings")
	}
	factor, low, high, ctx := *r.Factor, *r.Low, *r.High, *orig
	if !(factor > 0) || !(low > 0) || !(high > low) || !(ctx > 0) {
		return nil, fmt.Errorf("llama3 factor %g, low_freq_factor %g, high_freq_factor %g over %g positions", factor, low, high, ctx)
	}
	return func(rotated uint32, base float64) []float32 {
		return llama3Freqs(rotated, base, factor, low, high, ctx)
	}, nil
}

// llama3Freqs is convert_hf_to_gguf.py's LlamaModel.generate_extra_tensors,
// every step in float32 as its torch tensors run it, so the factors are the
// GGUF's to the bit: freq = 1/base^(i/dim) over even i, then 1 where the
// wavelength is under orig/high, factor where it is over orig/low, and the
// smooth blend between.
func llama3Freqs(headDim uint32, base, factor, low, high, orig float64) []float32 {
	dim := float32(headDim)
	lowWave, highWave := orig/low, orig/high
	twoPi := float32(2 * math.Pi)
	var out []float32
	for i := uint32(0); i < headDim; i += 2 {
		e := float32(float32(i) / dim)
		p := float32(math.Pow(float64(float32(base)), float64(e)))
		freq := float32(1 / p)
		wave := float32(twoPi / freq)
		switch {
		case float64(wave) < highWave:
			out = append(out, 1)
		case float64(wave) > lowWave:
			out = append(out, float32(factor))
		default:
			s := float32(float32(float32(float32(orig)/wave)-float32(low)) / float32(high-low))
			out = append(out, float32(1/float32(float32(float32(1-s)/float32(factor))+s)))
		}
	}
	return out
}
