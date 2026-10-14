package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// BitNet b1.58 (jlm.ArchBitNet). Its seven projections are ternary, and every
// source form of them reaches the container as a ternary type -- TQ2_0's
// 2-bit codes, with the scale an f16 (TypeTQ2) or, where no f16 holds it, an
// f32 (TypeTQ2F) -- through one of the conversions here, each exact:
//
//	TQ2_0 (llama.cpp --outtype tq2_0)          carried
//	TQ1_0 (llama.cpp --outtype tq1_0)          its base-3 digits re-packed as 2-bit codes
//	I2_S (bitnet.cpp)                          its codes re-packed, its f32 scale carried
//	F32/F16/BF16 holding a ternary tensor      the codes and each block's magnitude
//	(bitnet.cpp's converter dequantizes)       read off it
//	the HuggingFace release, uint8 codes       unpacked, weight_scale as the scale
//	and a weight_scale (offline)
//	the latent checkpoint (online)             BitNetForCausalLM's own absmean
//	                                           ternarization, in the checkpoint's dtype
//	                                           as the class runs it
//
// The scale's precision is the type's, as its grouping is (kernels.TQ2F): an
// f16 where every block's scale is one, which TQ2_0 and the release's bf16
// weight_scale always are, and otherwise the f32, rather than a rounding. On
// the 2B-4T a rounded scale is not small: the synth fixture's F16 GGUF, which
// rounds each magnitude to an f16, reads 4e-4..4e-3 from the latent golden,
// and transformers fed the same rounded magnitudes reproduces that to the
// digit.
//
// The graph's other facts -- squared ReLU on the gate, NEOX rotary on the
// rows as shipped, the two inner norms, per-token int8 activations -- are the
// arch's (bitnetConfig, jlm.ArchBitNet).

// bitnetProjections are the roles a BitNet block runs as BitLinear.
var bitnetProjections = map[jlm.Role]bool{
	jlm.RoleAttnQ: true, jlm.RoleAttnK: true, jlm.RoleAttnV: true, jlm.RoleAttnOut: true,
	jlm.RoleFFNGate: true, jlm.RoleFFNUp: true, jlm.RoleFFNDown: true,
}

// bitnetConfig is the arch's graph on the GGUF side. bitnet.cpp's
// "bitnet-b1.58" states no activation; the class it converts,
// BitNetForCausalLM, declares relu2 (transformers' BitNetConfig.hidden_act)
// and rotates with rotate_half over the rows as shipped, which its converter
// does not permute. bitnet.cpp's graph runs SiLU here (RULE 7m).
func bitnetConfig(c *jlm.Config) {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagReLU2
}

// exactHalf is v as an f16, and false when no f16 is exactly v.
func exactHalf(v float32) (uint16, bool) {
	h := quant.EncodeHalf(v)
	return h, float32(quant.DecodeHalf(h)) == v
}

// ternaryPack writes n ternary weights as TQ2_0 blocks when every block's
// scale is an f16, and as TQ2_F32 blocks otherwise: code(e) is weight e plus
// one (0..2), scale(b) block b's magnitude. The codes are TQ2_0's: element
// 128*(j/32) + 32*l + j%32 is bits 2l of byte j (quant's decoder is the
// specification).
func ternaryPack(n int, code func(e int) uint8, scale func(b int) float32) ([]byte, quant.Type, error) {
	return ternaryPackAs(0, n, code, scale)
}

// ternaryPackAs is ternaryPack into a type decided before the scales were
// known (a safetensors tensor is laid out before it is read): TQ2_F32 takes
// any scale, and TQ2_0 refuses one no f16 holds. force 0 chooses.
func ternaryPackAs(force quant.Type, n int, code func(e int) uint8, scale func(b int) float32) ([]byte, quant.Type, error) {
	if n%256 != 0 {
		return nil, 0, fmt.Errorf("%d weights is not a whole number of 256-blocks", n)
	}
	blocks := n / 256
	t, bb := quant.TQ2_0, 66
	if force == quant.TQ2_F32 {
		t, bb = quant.TQ2_F32, 68
	}
	for b := 0; b < blocks && t == quant.TQ2_0; b++ {
		if _, ok := exactHalf(scale(b)); !ok {
			if force == quant.TQ2_0 {
				return nil, 0, fmt.Errorf("block %d's scale %v is not an f16", b, scale(b))
			}
			t, bb = quant.TQ2_F32, 68
		}
	}
	out := make([]byte, blocks*bb)
	for b := 0; b < blocks; b++ {
		dst := out[b*bb:]
		for j := 0; j < 64; j++ {
			base := b*256 + 128*(j/32) + j%32
			var by byte
			for l := 0; l < 4; l++ {
				c := code(base + 32*l)
				if c > 2 {
					return nil, 0, fmt.Errorf("weight %d has code %d: not ternary", base+32*l, c)
				}
				by |= c << (2 * l)
			}
			dst[j] = by
		}
		if s := scale(b); t == quant.TQ2_0 {
			h, _ := exactHalf(s)
			binary.LittleEndian.PutUint16(dst[64:], h)
		} else {
			binary.LittleEndian.PutUint32(dst[64:], math.Float32bits(s))
		}
	}
	return out, t, nil
}

// tq1ToTQ2 re-packs TQ1_0 blocks as TQ2_0: the same codes and the same d, so
// every weight decodes to the same value.
func tq1ToTQ2(src []byte, n uint64) ([]byte, quant.Type, error) {
	blocks := n / 256
	if n%256 != 0 || uint64(len(src)) < blocks*54 {
		return nil, 0, fmt.Errorf("%d TQ1_0 elements in %d bytes", n, len(src))
	}
	// The digits through quant's own decoder at d = 1, so the order is
	// defined once.
	ones := make([]byte, len(src[:blocks*54]))
	copy(ones, src)
	for b := uint64(0); b < blocks; b++ {
		ones[b*54+52], ones[b*54+53] = 0x00, 0x3C
	}
	vals := make([]float32, n)
	if err := quant.Dequant32(quant.TQ1_0, ones, vals); err != nil {
		return nil, 0, err
	}
	return ternaryPack(int(n), func(e int) uint8 { return uint8(int(vals[e]) + 1) },
		func(b int) float32 { return float32(quant.DecodeHalf(binary.LittleEndian.Uint16(src[b*54+52:]))) })
}

// i2sToTQ2 re-packs bitnet.cpp's I2_S, the layout its x86 quantizer writes
// (ACT_PARALLEL, QK_I2_S 128, the layout of Microsoft's published GGUF):
// n/4 code bytes over the whole tensor, element 128i+32g+p in bits 6-2g of
// byte 32i+p, codes 0, 1, 2 for -1, 0, 1; then one f32 scale, every block's.
func i2sToTQ2(src []byte, n uint64) ([]byte, quant.Type, error) {
	if n%256 != 0 || uint64(len(src)) < n/4+4 {
		return nil, 0, fmt.Errorf("%d I2_S elements in %d bytes", n, len(src))
	}
	s := math.Float32frombits(binary.LittleEndian.Uint32(src[n/4:]))
	if !(s > 0) || math.IsInf(float64(s), 0) {
		return nil, 0, fmt.Errorf("its scale is %v", s)
	}
	return ternaryPack(int(n), func(e int) uint8 {
		return src[(e/128)*32+e%32] >> (6 - 2*((e%128)/32)) & 3
	}, func(int) float32 { return s })
}

// ternaryFromFloats packs a tensor whose values are ternary -- every nonzero
// value of a 256-block the same magnitude -- each block's magnitude its
// scale. bitnet.cpp's converter writes BitNet's projections this way at F32
// and F16 (the ternarized weight, dequantized). A value that is not one of
// its block's three is refused: the tensor is not ternary.
func ternaryFromFloats(vals []float32) ([]byte, quant.Type, error) {
	if len(vals)%256 != 0 {
		return nil, 0, fmt.Errorf("%d values is not a whole number of 256-blocks", len(vals))
	}
	mags := make([]float32, len(vals)/256)
	for b := range mags {
		for _, x := range vals[b*256 : b*256+256] {
			if a := float32(math.Abs(float64(x))); a != 0 {
				if mags[b] == 0 {
					mags[b] = a
				} else if a != mags[b] {
					return nil, 0, fmt.Errorf("block %d holds magnitudes %v and %v: not ternary", b, mags[b], a)
				}
			}
		}
	}
	return ternaryPack(len(vals), func(e int) uint8 {
		switch {
		case vals[e] > 0:
			return 2
		case vals[e] < 0:
			return 0
		}
		return 1
	}, func(b int) float32 { return mags[b] })
}

// floatsOf decodes a verbatim float tensor.
func floatsOf(t jlm.Type, data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	switch t {
	case jlm.TypeF32:
		if len(data) < 4*n {
			return nil, fmt.Errorf("%d f32 values in %d bytes", n, len(data))
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
	case jlm.TypeF16, jlm.TypeBF16:
		if len(data) < 2*n {
			return nil, fmt.Errorf("%d 16-bit values in %d bytes", n, len(data))
		}
		for i := range out {
			h := binary.LittleEndian.Uint16(data[2*i:])
			if t == jlm.TypeF16 {
				out[i] = float32(quant.DecodeHalf(h))
			} else {
				out[i] = math.Float32frombits(uint32(h) << 16)
			}
		}
	default:
		return nil, fmt.Errorf("%v is not a float type", t)
	}
	return out, nil
}

// ternaryGGUF re-packs a GGUF tensor of a ternary source type the container
// has no code for, and reports false for any other type.
func ternaryGGUF(t quant.Type, data []byte, n uint64) ([]byte, quant.Type, bool, error) {
	switch t {
	case quant.TQ1_0:
		b, ty, err := tq1ToTQ2(data, n)
		return b, ty, true, err
	case quant.I2_S:
		b, ty, err := i2sToTQ2(data, n)
		return b, ty, true, err
	}
	return nil, 0, false, nil
}

// bitnetScaleOf reports a GGUF's optional per-tensor projection scale
// (llama.cpp's "blk.N.attn_q.scale"): the tensor name it scales.
func bitnetScaleOf(name string) (string, bool) {
	if !strings.HasPrefix(name, "blk.") || !strings.HasSuffix(name, ".scale") {
		return "", false
	}
	return strings.TrimSuffix(name, ".scale") + ".weight", true
}

// bitnetTensors finishes a BitNet source: every projection becomes a ternary
// type, its optional per-tensor scale folded into each block's scale, and
// anything that is not ternary is refused by name. scales maps a weight's
// name to its scale.
func bitnetTensors(s *jlm.Source, scales map[string]float32) error {
	if s.Config == nil || s.Config.Arch != jlm.ArchBitNet {
		if len(scales) > 0 {
			return fmt.Errorf("convert: %d projection scales on a model that is not BitNet", len(scales))
		}
		return nil
	}
	used := 0
	for i := range s.Tensors {
		e := &s.Tensors[i]
		if !bitnetProjections[e.Role] {
			continue
		}
		n := 1
		for d := 0; d < int(e.NDim); d++ {
			n *= int(e.Dims[d])
		}
		var vals []float32
		switch e.Type {
		case jlm.TypeTQ2, jlm.TypeTQ2F:
			if _, ok := scales[e.Name]; ok {
				var err error
				if vals, err = ternaryValues(e, n); err != nil {
					return fmt.Errorf("convert: %s: %w", e.Name, err)
				}
			}
		case jlm.TypeF32, jlm.TypeF16, jlm.TypeBF16:
			var err error
			if vals, err = floatsOf(e.Type, e.Data, n); err != nil {
				return fmt.Errorf("convert: %s: %w", e.Name, err)
			}
		default:
			return fmt.Errorf("convert: %s is %v: a BitNet projection is ternary, and its activations "+
				"are quantized per token for a ternary weight: %w", e.Name, e.Type, ErrNotImplemented)
		}
		if sc, ok := scales[e.Name]; ok {
			used++
			for j := range vals {
				vals[j] *= sc
			}
		}
		if vals == nil {
			continue
		}
		b, ty, err := ternaryFromFloats(vals)
		if err != nil {
			return fmt.Errorf("convert: %s: %w", e.Name, err)
		}
		if e.Type, err = typeOf(ty); err != nil {
			return err
		}
		e.Data = b
	}
	if used != len(scales) {
		return fmt.Errorf("convert: %d projection scales and %d projections they scale", len(scales), used)
	}
	return nil
}

// ternaryValues decodes a ternary tensor to its values, for a fold.
func ternaryValues(e *jlm.Tensor, n int) ([]float32, error) {
	src, ok := jlm.SourceType(e.Type)
	if !ok {
		return nil, fmt.Errorf("%v has no source type", e.Type)
	}
	vals := make([]float32, n)
	return vals, quant.Dequant32(src, e.Data, vals)
}

// elemsOf is a shape's element count.
func elemsOf(dims []uint64) uint64 {
	n := uint64(1)
	for _, d := range dims {
		n *= d
	}
	return n
}
