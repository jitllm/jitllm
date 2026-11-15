package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/quant"
)

// ternaryForm is how a checkpoint stores its ternary projections: latent
// floats the converter ternarizes (an absmean scale, as BitLinear's
// WeightQuant), or packed 2-bit codes with a scale tensor beside each weight,
// the output multiplied by the scale or divided by it.
type ternaryForm struct {
	latent, divides bool
}

// ternaryGather consumes every projection's scale tensor, the name of its
// weight with suffix added: it is read with its weight (ternaryRetype), never
// stored. A scale with no weight is refused.
func ternaryGather(suffix string, names []string) (map[string]bool, error) {
	used := map[string]bool{}
	have := map[string]bool{}
	for _, n := range names {
		have[n] = true
	}
	for _, n := range names {
		w, ok := strings.CutSuffix(n, suffix)
		if !ok || !strings.HasSuffix(w, ".weight") {
			continue
		}
		if !have[w] {
			return nil, fmt.Errorf("convert: %s scales no weight", n)
		}
		used[n] = true
	}
	return used, nil
}

// ternaryRetype gives a projection its ternary type before anything is laid
// out, and the read that produces its blocks. The scale's precision follows
// from the dtypes alone, since the type is fixed before the data is read: a
// bf16 or f16 scale (a packed form's scale tensor, a bf16 or f16 latent
// checkpoint's magnitude) is an f16 exactly (TypeTQ2), and an f32 one, or one
// that is a reciprocal (a form that divides by its scale), is carried as an
// f32 (TypeTQ2F).
func ternaryRetype(form ternaryForm, suffix, name string, t *safetensors.Tensor,
	read func(name string) ([]byte, *safetensors.Tensor, error)) (*hfRetyped, error) {
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
	if form.latent {
		round, q := latentRounding(t.DType)
		if round == nil {
			return nil, fmt.Errorf("convert: %s is %s, and latent weights are ternarized in "+
				"their own dtype (F32, BF16 or F16): %w", name, t.DType, ErrNotImplemented)
		}
		return typed(q, []uint64{t.Shape[0], t.Shape[1]}, func() ([]byte, quant.Type, error) {
			b, _, err := read(name)
			if err != nil {
				return nil, 0, err
			}
			return ternarize(b, t.DType, round, q)
		})
	}
	if t.DType != safetensors.U8 {
		return nil, fmt.Errorf("convert: %s is %s, and this checkpoint's projections are "+
			"packed codes (U8)", name, t.DType)
	}
	st, err := scaleTensor(read, name+suffix)
	if err != nil {
		return nil, err
	}
	q := quant.TQ2_F32
	if !form.divides && (st.DType == safetensors.BF16 || st.DType == safetensors.F16) {
		q = quant.TQ2_0
	}
	out, in := 4*t.Shape[0], t.Shape[1]
	return typed(q, []uint64{out, in}, func() ([]byte, quant.Type, error) {
		b, _, err := read(name)
		if err != nil {
			return nil, 0, err
		}
		sb, st, err := read(name + suffix)
		if err != nil {
			return nil, 0, err
		}
		ws, err := ternaryScalar(st, sb)
		if err != nil {
			return nil, 0, fmt.Errorf("%s%s: %w", name, suffix, err)
		}
		s := ws
		if form.divides {
			s = 1 / ws
		}
		return ternaryUnpack(b, int(out), int(in), q, s)
	})
}

// scaleTensor is the header of name, a ternary weight's scale.
func scaleTensor(read func(string) ([]byte, *safetensors.Tensor, error), name string) (*safetensors.Tensor, error) {
	_, st, err := read(name)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	return st, nil
}

// ternaryScalar reads a one-element float tensor.
func ternaryScalar(t *safetensors.Tensor, b []byte) (float32, error) {
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

// ternaryUnpack turns transformers' packed codes into ternary blocks of type
// q: a [out/4, in] uint8 tensor whose byte (r, i) holds row j*out/4 + r's
// code for column i in bits 2j (pack_weights / unpack_weights), the code
// being the weight plus one, every block at scale s.
func ternaryUnpack(b []byte, out, in int, q quant.Type, s float32) ([]byte, quant.Type, error) {
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

// ternarize is WeightQuant over an [out, in] latent weight of dtype d,
// step by step as torch runs it on a tensor of that dtype, round being the
// dtype's rounding:
//
//	m = round(sum|w| / n)         the mean, accumulated wide
//	s = round(1 / m)              after clamp(min=1e-5)
//	q = clamp(rne(round(w*s)), -1, 1)
//	weight = round(q / s)         so the magnitude is round(1/s)
func ternarize(b []byte, d safetensors.DType, round func(float32) float32, q quant.Type) ([]byte, quant.Type, error) {
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
