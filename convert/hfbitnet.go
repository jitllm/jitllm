package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// bitnetHF is BitNetForCausalLM from safetensors: llama's names with the two
// inner norms, every projection ternary. transformers runs it through its
// "bitnet" quantizer, whose quantization_config says which form the
// projections are stored in (integrations/bitnet.py):
//
//	autobitlinear, offline   uint8, four 2-bit codes a byte (code-1 is the
//	                         weight), and a bf16 weight_scale the output is
//	                         multiplied by -- the published release
//	autobitlinear, online    the latent weights, ternarized every call by
//	                         WeightQuant: s = 1/mean|W|, round(W*s) clamped to
//	                         [-1, 1], over s
//	bitlinear                the uint8 codes with a weight_scale the output is
//	                         DIVIDED by
//
// Each becomes TypeTQ2 or TypeTQ2F (convert/bitnet.go), the weight's
// magnitude every block's scale at the precision it has.
var bitnetHF = &hfArch{
	arch:        jlm.ArchBitNet,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block: blockPlus(hfLlamaBlockNoBias, map[string]jlm.Role{
		"self_attn.attn_sub_norm.weight": jlm.RoleAttnSubNorm,
		"mlp.ffn_sub_norm.weight":        jlm.RoleFFNSubNorm,
	}),
	ignore: llamaHF.ignore,
	config: bitnetHFConfig,
	gather: bitnetGather,
	retype: bitnetRetype,
}

// bitnetForm is how a checkpoint stores its projections.
type bitnetForm int

const (
	bitnetPackedMul bitnetForm = iota + 1 // autobitlinear, offline
	bitnetLatent                          // autobitlinear, online
	bitnetPackedDiv                       // bitlinear
)

// bitnetQuant is the "bitnet" quantization_config.
type bitnetQuant struct {
	QuantMethod string          `json:"quant_method"`
	LinearClass string          `json:"linear_class"`
	Mode        string          `json:"quantization_mode"`
	UseRMSNorm  *bool           `json:"use_rms_norm"`
	NotConvert  json.RawMessage `json:"modules_to_not_convert"`
}

// bitnetFormOf reads the quantization_config. A BitNetForCausalLM without
// one runs full-precision linears in transformers, which is not the ternary
// graph this arch is, so it is refused; so is every option that changes the
// graph (an RMSNorm inside each linear, another module list).
func bitnetFormOf(c *hfConfig) (bitnetForm, error) {
	raw, ok := c.keys["quantization_config"]
	if !ok || empty(raw) {
		return 0, fmt.Errorf("convert: %s: a BitNetForCausalLM with no quantization_config runs "+
			"full-precision projections in transformers: %w", c.path, ErrNotImplemented)
	}
	var q bitnetQuant
	if err := json.Unmarshal(raw, &q); err != nil {
		return 0, fmt.Errorf("convert: %s: quantization_config: %w", c.path, err)
	}
	if q.QuantMethod != "bitnet" {
		return 0, fmt.Errorf("convert: %s: quant_method %q is not BitNet's: %w", c.path, q.QuantMethod, ErrNotImplemented)
	}
	if q.UseRMSNorm != nil && *q.UseRMSNorm {
		return 0, fmt.Errorf("convert: %s: use_rms_norm puts an RMSNorm inside every projection: %w",
			c.path, ErrNotImplemented)
	}
	if !empty(q.NotConvert) {
		var mods []string
		if err := json.Unmarshal(q.NotConvert, &mods); err != nil {
			return 0, fmt.Errorf("convert: %s: modules_to_not_convert: %w", c.path, err)
		}
		for _, m := range mods {
			if m != "lm_head" {
				return 0, fmt.Errorf("convert: %s: modules_to_not_convert names %q, a projection "+
					"left full precision: %w", c.path, m, ErrNotImplemented)
			}
		}
	}
	class := q.LinearClass
	if class == "" {
		class = "bitlinear" // BitNetQuantConfig's default
	}
	mode := q.Mode
	if mode == "" {
		mode = "offline"
	}
	switch {
	case class == "autobitlinear" && mode == "offline":
		return bitnetPackedMul, nil
	case class == "autobitlinear" && mode == "online":
		return bitnetLatent, nil
	case class == "bitlinear" && mode == "offline":
		return bitnetPackedDiv, nil
	}
	return 0, fmt.Errorf("convert: %s: linear_class %q in mode %q: %w", c.path, class, mode, ErrNotImplemented)
}

// bitnetHFConfig reads BitNetConfig with its class defaults, and the
// quantization_config's form.
func bitnetHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	if _, err := bitnetFormOf(c); err != nil {
		return nil, err
	}
	hfClassDefaults{kvHeads: 5, ropeTheta: 500000, rmsEps: 1e-5, tied: false, nCtx: 2048}.apply(c)
	if c.act() == "" {
		c.HiddenAct = "relu2"
	}
	cfg, err := hfConfigWith(jlm.ArchBitNet, 0, "relu2")(c, names)
	if err != nil {
		return nil, err
	}
	bitnetConfig(cfg)
	return cfg, nil
}

// bitnetGather consumes every projection's weight_scale: it is read with its
// weight (bitnetRetype), never stored. A weight_scale with no projection, or
// a form that has none, is refused.
func bitnetGather(c *jlm.Config, names []string, _ func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
	used := map[string]bool{}
	have := map[string]bool{}
	for _, n := range names {
		have[n] = true
	}
	for _, n := range names {
		if !strings.HasSuffix(n, ".weight_scale") {
			continue
		}
		w := strings.TrimSuffix(n, "_scale")
		if !have[w] {
			return nil, nil, fmt.Errorf("convert: %s scales no weight", n)
		}
		used[n] = true
	}
	return nil, used, nil
}

// bitnetRetype gives a projection its ternary type before anything is laid
// out, and the read that produces its blocks. The scale's precision follows
// from the dtypes alone, since the type is fixed before the data is read: a
// bf16 or f16 scale (the release's weight_scale, a bf16 or f16 latent
// checkpoint's magnitude) is an f16 exactly (TypeTQ2), and an f32 one, or one
// that is a reciprocal (bitlinear divides by its weight_scale), is carried
// as an f32 (TypeTQ2F).
func bitnetRetype(hc *hfConfig, role jlm.Role, name string, t *safetensors.Tensor,
	read func(name string) ([]byte, *safetensors.Tensor, error)) (*hfRetyped, error) {
	if !bitnetProjections[role] {
		return nil, nil
	}
	form, err := bitnetFormOf(hc)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("convert: %s has %d dimensions", name, len(t.Shape))
	}
	typed := func(q quant.Type, shape []uint64, f func() ([]byte, quant.Type, error)) (*hfRetyped, error) {
		ty, err := typeOf(q)
		if err != nil {
			return nil, err
		}
		return &hfRetyped{ty: ty, shape: shape, read: func() ([]byte, error) {
			b, got, err := f()
			if err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			if got != q {
				return nil, fmt.Errorf("convert: %s: laid out as %s and packed as %s", name, q, got)
			}
			return b, nil
		}}, nil
	}
	if form == bitnetLatent {
		round, q := latentRounding(t.DType)
		if round == nil {
			return nil, fmt.Errorf("convert: %s is %s, and BitNet's latent weights are ternarized in "+
				"their own dtype (F32, BF16 or F16): %w", name, t.DType, ErrNotImplemented)
		}
		return typed(q, []uint64{t.Shape[0], t.Shape[1]}, func() ([]byte, quant.Type, error) {
			b, _, err := read(name)
			if err != nil {
				return nil, 0, err
			}
			return bitnetTernarize(b, t.DType, round, q)
		})
	}
	if t.DType != safetensors.U8 {
		return nil, fmt.Errorf("convert: %s is %s, and this checkpoint's projections are "+
			"packed codes (U8)", name, t.DType)
	}
	st, err := scaleTensor(read, name+"_scale")
	if err != nil {
		return nil, err
	}
	q := quant.TQ2_F32
	if form == bitnetPackedMul && (st.DType == safetensors.BF16 || st.DType == safetensors.F16) {
		q = quant.TQ2_0
	}
	out, in := 4*t.Shape[0], t.Shape[1]
	return typed(q, []uint64{out, in}, func() ([]byte, quant.Type, error) {
		b, _, err := read(name)
		if err != nil {
			return nil, 0, err
		}
		sb, st, err := read(name + "_scale")
		if err != nil {
			return nil, 0, err
		}
		ws, err := bitnetScalar(st, sb)
		if err != nil {
			return nil, 0, fmt.Errorf("%s_scale: %w", name, err)
		}
		s := ws
		if form == bitnetPackedDiv {
			s = 1 / ws
		}
		return bitnetUnpack(b, int(out), int(in), q, s)
	})
}

// scaleTensor is the header of name, a weight_scale.
func scaleTensor(read func(string) ([]byte, *safetensors.Tensor, error), name string) (*safetensors.Tensor, error) {
	_, st, err := read(name)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	return st, nil
}

// bitnetScalar reads a one-element float tensor.
func bitnetScalar(t *safetensors.Tensor, b []byte) (float32, error) {
	n := uint64(1)
	for _, d := range t.Shape {
		n *= d
	}
	if n != 1 {
		return 0, fmt.Errorf("%d values, want one", n)
	}
	switch t.DType {
	case safetensors.BF16:
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16), nil
	case safetensors.F32:
		return math.Float32frombits(binary.LittleEndian.Uint32(b)), nil
	case safetensors.F16:
		return float32(quant.DecodeHalf(binary.LittleEndian.Uint16(b))), nil
	}
	return 0, fmt.Errorf("dtype %s", t.DType)
}

// bitnetUnpack turns transformers' packed codes into ternary blocks of type
// q: a [out/4, in] uint8 tensor whose byte (r, i) holds row j*out/4 + r's
// code for column i in bits 2j (pack_weights / unpack_weights), the code
// being the weight plus one, every block at scale s.
func bitnetUnpack(b []byte, out, in int, q quant.Type, s float32) ([]byte, quant.Type, error) {
	rows4 := out / 4
	if out%4 != 0 || len(b) != rows4*in || in%256 != 0 {
		return nil, 0, fmt.Errorf("%d bytes for a [%d, %d] ternary weight", len(b), out, in)
	}
	return ternaryPackAs(q, out*in, func(e int) uint8 {
		o, i := e/in, e%in
		return b[(o%rows4)*in+i] >> uint(2*(o/rows4)) & 3
	}, func(int) float32 { return s })
}

// latentRounding is the rounding torch applies after each op on a tensor of
// dtype d (every op computed in f32, the result stored in d, to nearest
// even), and the type the magnitude is carried in: an f32 needs TQ2_F32, a
// bf16 or f16 is an f16 exactly.
func latentRounding(d safetensors.DType) (func(float32) float32, quant.Type) {
	switch d {
	case safetensors.F32:
		return func(x float32) float32 { return x }, quant.TQ2_F32
	case safetensors.BF16:
		return bf16Round, quant.TQ2_0
	case safetensors.F16:
		return func(x float32) float32 { return float32(quant.DecodeHalf(quant.EncodeHalf(x))) }, quant.TQ2_0
	}
	return nil, 0
}

// bitnetTernarize is WeightQuant over an [out, in] latent weight of dtype d,
// step by step as torch runs it on a tensor of that dtype, round being the
// dtype's rounding:
//
//	m = round(sum|w| / n)         the mean, accumulated wide
//	s = round(1 / m)              after clamp(min=1e-5)
//	q = clamp(rne(round(w*s)), -1, 1)
//	weight = round(q / s)         so the magnitude is round(1/s)
func bitnetTernarize(b []byte, d safetensors.DType, round func(float32) float32, q quant.Type) ([]byte, quant.Type, error) {
	var w []float32
	switch d {
	case safetensors.F32:
		w = make([]float32, len(b)/4)
		for i := range w {
			w[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
	default:
		w = make([]float32, len(b)/2)
		for i := range w {
			h := binary.LittleEndian.Uint16(b[2*i:])
			if d == safetensors.BF16 {
				w[i] = math.Float32frombits(uint32(h) << 16)
			} else {
				w[i] = float32(quant.DecodeHalf(h))
			}
		}
	}
	sum := 0.0
	for _, x := range w {
		sum += math.Abs(float64(x))
	}
	m := round(float32(sum / float64(len(w))))
	if lo := round(1e-5); m < lo {
		m = lo
	}
	s := round(1 / m)
	mag := round(1 / s)
	return ternaryPackAs(q, len(w), func(e int) uint8 {
		v := math.RoundToEven(float64(round(w[e] * s)))
		return uint8(max(-1, min(1, v)) + 1)
	}, func(int) float32 { return mag })
}

// bf16Round is x rounded to the nearest bfloat16, ties to even, as torch's
// float-to-bfloat16 conversion rounds (NaN is kept a NaN).
func bf16Round(x float32) float32 {
	u := math.Float32bits(x)
	if x != x {
		return x
	}
	u += 0x7FFF + (u >> 16 & 1)
	return math.Float32frombits(u &^ 0xFFFF)
}
