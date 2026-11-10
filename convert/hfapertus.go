package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// apertusHF is Apertus (ApertusForCausalLM) from safetensors, to the container
// its GGUF converts to (convert/apertus.go is that half): llama's names with
// the norms renamed (attention_layernorm, feedforward_layernorm), qwen3's
// per-head q/k norm, an ungated MLP, llama 3's rotary factors written from
// config.json as the rope_freqs tensor convert_hf_to_gguf.py writes, and the
// four xIELU scalars a block gathered into its RoleXIELU vector.
func apertusHF() *hfArch {
	return &hfArch{
		arch:        ArchApertus,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block: map[string]jlm.Role{
			"attention_layernorm.weight":   jlm.RoleAttnNorm,
			"feedforward_layernorm.weight": jlm.RoleFFNNorm,
			"self_attn.q_proj.weight":      jlm.RoleAttnQ,
			"self_attn.k_proj.weight":      jlm.RoleAttnK,
			"self_attn.v_proj.weight":      jlm.RoleAttnV,
			"self_attn.o_proj.weight":      jlm.RoleAttnOut,
			"self_attn.q_proj.bias":        jlm.RoleAttnQBias,
			"self_attn.k_proj.bias":        jlm.RoleAttnKBias,
			"self_attn.v_proj.bias":        jlm.RoleAttnVBias,
			"self_attn.o_proj.bias":        jlm.RoleAttnOutBias,
			"self_attn.q_norm.weight":      jlm.RoleAttnQNorm,
			"self_attn.k_norm.weight":      jlm.RoleAttnKNorm,
			"mlp.up_proj.weight":           jlm.RoleFFNUp,
			"mlp.down_proj.weight":         jlm.RoleFFNDown,
		},
		ignore: llamaHF().ignore,
		config: apertusHFConfig,
		gather: apertusGather,
	}
}

// apertusHFConfig reads ApertusConfig: hidden_act must be xielu, and q/k are
// normed per head (qk_norm) before the rotary, with no post-norm block.
func apertusHFConfig(c *hfConfig, names []string) (*Config, error) {
	for _, k := range []struct {
		key  string
		want bool
	}{{"qk_norm", true}, {"post_norm", false}} {
		if raw, ok := c.keys[k.key]; ok {
			var v bool
			if err := json.Unmarshal(raw, &v); err != nil {
				return nil, fmt.Errorf("convert: %s: %s: %w", c.path, k.key, err)
			}
			if v != k.want {
				return nil, fmt.Errorf("convert: %s: %s %v is not the Apertus graph: %w",
					c.path, k.key, v, ErrNotImplemented)
			}
		}
	}
	freqs, err := hfLlama3Consume(c)
	if err != nil {
		return nil, err
	}
	cfg, err := hfConfigWith(ArchApertus, jlm.FlagQKNorm|jlm.FlagRopeNeox, "xielu")(c, names)
	if err != nil {
		return nil, err
	}
	if freqs != nil {
		f := freqs(cfg.HeadDim, float64(cfg.RopeBase))
		b := make([]byte, 4*len(f))
		for i, x := range f {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		c.extra = append(c.extra, jlm.Tensor{Role: jlm.RoleRopeFreqs, Block: jlm.DenseBlock, Index: -1,
			Type: jlm.TypeF32, NDim: 1, Dims: [4]uint64{uint64(len(f))}, Data: b, Name: "rope_freqs.weight"})
	}
	return cfg, nil
}

// apertusGather folds each block's mlp.act_fn.{alpha_p, alpha_n, beta, eps}
// into its RoleXIELU vector through xieluTensor, the GGUF path's own fold,
// so the two inputs write the same bytes. Every block must carry all four.
func apertusGather(c *Config, names []string, read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
	used := map[string]bool{}
	for _, n := range names {
		if strings.Contains(n, ".mlp.act_fn.") {
			used[n] = true
		}
	}
	var out []jlm.Tensor
	for b := range int(c.NLayer) {
		var v [4]float64
		for i, k := range []string{"alpha_p", "alpha_n", "beta", "eps"} {
			name := llamaHF().blockPrefix + itoa(b) + ".mlp.act_fn." + k
			if !used[name] {
				return nil, nil, fmt.Errorf("convert: block %d has no %s: Apertus's activation is carried there", b, name)
			}
			x, err := read(name)
			if err != nil {
				return nil, nil, err
			}
			if len(x) != 1 {
				return nil, nil, fmt.Errorf("convert: %s has %d values, want 1", name, len(x))
			}
			v[i] = float64(x[0])
			delete(used, name)
		}
		out = append(out, xieluTensor(b, v[0], v[1], v[2], v[3]))
	}
	// A name of this shape outside the blocks' four is a tensor nobody reads.
	for n := range used {
		return nil, nil, fmt.Errorf("convert: %s is not one of a block's four xIELU numbers", n)
	}
	for b := range int(c.NLayer) {
		for _, k := range []string{"alpha_p", "alpha_n", "beta", "eps"} {
			used[llamaHF().blockPrefix+itoa(b)+".mlp.act_fn."+k] = true
		}
	}
	return out, used, nil
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

// hfLlama3Consume reads a llama3 rope_scaling and removes it from the config,
// so llamaHFConfig's refusal of every scaling does not fire on it, returning
// the function that computes the per-pair factors once the head width and the
// base are known. Any other scaling is left for that refusal.
func hfLlama3Consume(c *hfConfig) (func(headDim uint32, base float64) []float32, error) {
	if len(c.RopeScaling) == 0 || string(c.RopeScaling) == "null" {
		return nil, nil
	}
	var r hfLlama3
	if err := json.Unmarshal(c.RopeScaling, &r); err != nil {
		return nil, fmt.Errorf("convert: %s: rope_scaling: %w", c.path, err)
	}
	if r.RopeType != "llama3" && (r.RopeType != "" || r.Type != "llama3") {
		return nil, nil
	}
	c.RopeScaling = nil
	def := func(p *float64, v float64) float64 {
		if p != nil {
			return *p
		}
		return v
	}
	factor, low, high, orig := def(r.Factor, 8), def(r.Low, 1), def(r.High, 4), def(r.OrigCtx, 8192)
	return func(headDim uint32, base float64) []float32 {
		return llama3Freqs(headDim, base, factor, low, high, orig)
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
