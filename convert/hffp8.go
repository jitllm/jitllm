package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
)

// A checkpoint's hf_quant_config.json names, per parameter group, how its
// tensors were quantized. Two transforms are block formats the container
// holds as stored (jlm.TypeNV4, jlm.TypeMX8); "cast" is a plain dtype.
//
//	nvfp4_nearest  X uint8 [.., rows, k/2] e2m1 codes, element 2j low;
//	               X_scale F8_E4M3 [.., rows, k/16]; weight_scale_format
//	               amax_denominator: X_amax f32 a sheet and
//	               X_outer_scale_denominator f32, the sheet's global scale
//	               their quotient.
//	mxfp8_nearest  X F8_E4M3 [.., rows, k]; X_scale uint8 [.., rows, k/32],
//	               E8M0, one a row's 32 columns (the [32, 32] block shape
//	               is the quantizer's grouping; the scales are stored a row).
type hfQuantConfig struct {
	Model hfQuantModel `json:"model"`
	MTP   hfQuantModel `json:"mtp"`
}

type hfQuantModel struct {
	Groups []hfQuantGroup `json:"parameter_groups"`
}

type hfQuantGroup struct {
	Tensors     []string    `json:"tensors"`
	Transform   string      `json:"weight_transform"`
	Block       []int       `json:"weight_block_shape"`
	ScaleFormat string      `json:"weight_scale_format"`
	Act         *hfActQuant `json:"required_activation_quantization"`
}

// hfActQuant is how the publisher quantizes a group's input rows before
// its product. The engine computes the product with f32 activations, so
// a form it knows is converted and runs unquantized activations, and any
// other is refused by name rather than dropped.
type hfActQuant struct {
	Format     string `json:"format"`
	Block      int    `json:"block_size"`
	Amax       string `json:"amax_type"`
	ScaleDtype string `json:"scale_dtype"`
}

// known is an NVFP4 activation of a token's 16-element blocks, each with
// its E4M3 scale: the form a checkpoint's NVFP4 weights are paired with.
func (a *hfActQuant) known() bool {
	return a == nil || a.Format == "nvfp4_e2m1" && a.Block == 16 && a.Amax == "per_token" && a.ScaleDtype == "float8_e4m3fn"
}

const (
	nv4Scale = "_scale"
	nv4Amax  = "_amax"
	nv4Denom = "_outer_scale_denominator"
)

// hfQuantGroupsOf is each block-quantized tensor hf_quant_config.json names,
// with the container type it is held in; nil where the file is absent. A
// transform it does not know is refused by name.
func hfQuantGroupsOf(dir hfDir) (map[string]jlm.Type, error) {
	const f = "hf_quant_config.json"
	if !dir.has(f) {
		return nil, nil
	}
	b, err := dir.read(f)
	if err != nil {
		return nil, err
	}
	var qc hfQuantConfig
	if err := json.Unmarshal(b, &qc); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", dir.path(f), err)
	}
	out := map[string]jlm.Type{}
	for part, m := range map[string]hfQuantModel{"model": qc.Model, "mtp": qc.MTP} {
		for i, g := range m.Groups {
			if !g.Act.known() {
				return nil, fmt.Errorf("convert: %s: %s group %d quantizes its activations as %+v: %w",
					dir.path(f), part, i, *g.Act, ErrNotImplemented)
			}
			var ty jlm.Type
			switch {
			case g.Transform == "cast":
				continue
			case g.Transform == "nvfp4_nearest" && slices.Equal(g.Block, []int{1, 16}) && g.ScaleFormat == "amax_denominator":
				ty = jlm.TypeNV4
			case g.Transform == "mxfp8_nearest" && slices.Equal(g.Block, []int{32, 32}):
				ty = jlm.TypeMX8
			default:
				return nil, fmt.Errorf("convert: %s: %s group %d is %s, block %v, scales %q: %w",
					dir.path(f), part, i, g.Transform, g.Block, g.ScaleFormat, ErrNotImplemented)
			}
			if part != "model" {
				return nil, fmt.Errorf("convert: %s: %s's tensors are named under a prefix this converter does not know: %w",
					dir.path(f), part, ErrNotImplemented)
			}
			for _, n := range g.Tensors {
				out[n] = ty
			}
		}
	}
	return out, nil
}

// mergeFP8 joins each tensor groups names with its scale (and NVFP4's
// amax and denominator) into one entry of all reading as its container
// type, the shapes checked against each other from the headers.
func mergeFP8(all map[string]hfLoc, groups map[string]jlm.Type) error {
	for name, ty := range groups {
		w, ok := all[name]
		sc, sok := all[name+nv4Scale]
		if !ok || !sok {
			return fmt.Errorf("convert: hf_quant_config.json names %s, which has no tensor and %s beside it", name, nv4Scale)
		}
		p, s := w.t, sc.t
		n := len(p.Shape)
		if n < 2 || len(s.Shape) != n || !slices.Equal(p.Shape[:n-1], s.Shape[:n-1]) {
			return fmt.Errorf("convert: %s is %v and its scales %v", name, p.Shape, s.Shape)
		}
		sheets := uint64(1)
		for _, d := range p.Shape[:n-2] {
			sheets *= d
		}
		loc := hfLoc{f: w.f, t: p, sf: sc.f, st: s, q: ty}
		switch ty {
		case jlm.TypeNV4:
			k := 2 * p.Shape[n-1]
			if p.DType != safetensors.U8 || s.DType != safetensors.F8E4M3 || s.Shape[n-1]*16 != k {
				return fmt.Errorf("convert: %s is %s%v and its scales %s%v; NVFP4 is uint8 codes and an E4M3 a 16",
					name, p.DType, p.Shape, s.DType, s.Shape)
			}
			a, aok := all[name+nv4Amax]
			d, dok := all[name+nv4Denom]
			if !aok || !dok {
				return fmt.Errorf("convert: %s has no %s and %s beside it", name, nv4Amax, nv4Denom)
			}
			if a.t.DType != safetensors.F32 || d.t.DType != safetensors.F32 || volume(a.t.Shape) != sheets || volume(d.t.Shape) != 1 {
				return fmt.Errorf("convert: %s's amax is %s%v and its denominator %s%v; NVFP4 wants an f32 a sheet (%d) and one",
					name, a.t.DType, a.t.Shape, d.t.DType, d.t.Shape, sheets)
			}
			loc.af, loc.at, loc.df, loc.dt = a.f, a.t, d.f, d.t
			delete(all, name+nv4Amax)
			delete(all, name+nv4Denom)
		case jlm.TypeMX8:
			if p.DType != safetensors.F8E4M3 || s.DType != safetensors.U8 || s.Shape[n-1]*32 != p.Shape[n-1] {
				return fmt.Errorf("convert: %s is %s%v and its scales %s%v; MXFP8 is E4M3 values and an E8M0 a 32",
					name, p.DType, p.Shape, s.DType, s.Shape)
			}
		}
		all[name] = loc
		delete(all, name+nv4Scale)
	}
	return nil
}

func volume(s []uint64) uint64 {
	n := uint64(1)
	for _, d := range s {
		n *= d
	}
	return n
}

// fp8Shape is the weight's shape in elements: NVFP4's last axis two codes a
// byte.
func (l hfLoc) fp8Shape() []uint64 {
	s := slices.Clone(l.t.Shape)
	if l.q == jlm.TypeNV4 {
		s[len(s)-1] *= 2
	}
	return s
}

// readFP8 reads one weight's planes and lays them out as its container
// type's rows, every code and scale byte moved as stored.
func (l hfLoc) readFP8() ([]byte, error) {
	p, err := l.f.ReadTensor(l.t)
	if err != nil {
		return nil, err
	}
	s, err := l.sf.ReadTensor(l.st)
	if err != nil {
		return nil, err
	}
	shape := l.fp8Shape()
	n := len(shape)
	k, rows := int(shape[n-1]), int(shape[n-2])
	sheets := int(volume(shape[:n-2]))
	if l.q == jlm.TypeMX8 {
		return jlm.MX8Rows(p, s, sheets*rows, k)
	}
	a, err := l.af.ReadTensor(l.at)
	if err != nil {
		return nil, err
	}
	d, err := l.df.ReadTensor(l.dt)
	if err != nil {
		return nil, err
	}
	denom := math.Float32frombits(binary.LittleEndian.Uint32(d))
	global := make([]float32, sheets)
	for i := range global {
		global[i] = math.Float32frombits(binary.LittleEndian.Uint32(a[4*i:])) / denom
	}
	return jlm.NV4Rows(p, s, global, sheets, rows, k)
}
