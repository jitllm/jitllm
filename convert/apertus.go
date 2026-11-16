package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// xieluTensors writes each block's xIELU numbers, for a file stating any
// (llama.cpp's converter stores the raw parameters as four per-layer keys,
// a scalar standing for every layer), as that block's RoleXIELU vector,
// folded; a file stating some and not all is refused.
func xieluTensors(c *Config, f *meta.File, s *jlm.Source) error {
	keys := []string{"xielu.alpha_p", "xielu.alpha_n", "xielu.beta", "xielu.eps"}
	if c == nil || !slices.ContainsFunc(keys, func(k string) bool {
		_, ok := xieluKey(f, k)
		return ok
	}) {
		return nil
	}
	n := int(c.NLayer)
	var raw [4][]float64
	for i, k := range keys {
		v, ok := xieluKey(f, k)
		if !ok {
			return fmt.Errorf("convert: xielu: no %s", k)
		}
		if v.Len() == 0 {
			x, ok := v.Float()
			if !ok {
				return fmt.Errorf("convert: xielu: %s is %v, not a number", k, v.Type)
			}
			for range n {
				raw[i] = append(raw[i], x)
			}
			continue
		}
		xs, err := v.Float32s()
		if err != nil {
			return fmt.Errorf("convert: xielu: %s: %w", k, err)
		}
		if len(xs) != n {
			return fmt.Errorf("convert: xielu: %s has %d values for %d blocks", k, len(xs), n)
		}
		for _, x := range xs {
			raw[i] = append(raw[i], float64(x))
		}
	}
	for b := range n {
		s.Tensors = append(s.Tensors, xieluTensor(b, raw[0][b], raw[1][b], raw[2][b], raw[3][b]))
	}
	return nil
}

// xieluKey reads one of the xielu.* keys. llama.cpp's writer stores them
// without the architecture's namespace (gguf_writer.add_xielu_*), so the bare
// name is the one a file carries; a namespaced spelling is read too.
func xieluKey(f *meta.File, k string) (meta.Value, bool) {
	if v, ok := f.KV[k]; ok {
		return v, true
	}
	return f.Key(k)
}

// xieluTensor is block b's RoleXIELU vector from the raw parameters, as
// XIELUActivation computes them each call: alpha_p = softplus(a_p) and
// alpha_n = beta + softplus(a_n), torch's softplus (x past 20 is itself).
// Both inputs reach the container through here, so they write the same bytes.
func xieluTensor(b int, aP, aN, beta, eps float64) jlm.Tensor {
	sp := func(x float64) float64 {
		if x > 20 {
			return x
		}
		return math.Log1p(math.Exp(x))
	}
	p := [4]float32{float32(sp(aP)), float32(beta + sp(aN)), float32(beta), float32(eps)}
	data := make([]byte, 16)
	for i, v := range p {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
	}
	t := jlm.Tensor{Role: jlm.RoleXIELU, Block: int32(b), Index: -1, Type: jlm.TypeF32, NDim: 1,
		Data: data, Name: "blk." + itoa(b) + ".ffn_xielu"}
	t.Dims[0] = 4
	return t
}
