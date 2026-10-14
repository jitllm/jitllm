package convert

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The two transforms a vision tower needs before it can be written. The
// engine reads one weight layout, the packed plane layout, and a mostly-F16
// mmproj (llava-phi-3's is 99.79% F16) written verbatim would reach a
// row-major path on the host and could not be placed on any device at all
// (tier.quantOf knows only packed quants). So a tower's matrices are
// quantized to Q8_0 here, where the lossy step is stated; jlm.Write packs the
// Q8_0 blocks immediately, so the read side never sees a source layout.

// towerMatrix is the set of roles a tower multiplies by, as opposed to the
// norms, biases, position table and class embedding it reads once. Only these
// are quantized; see quantizeTower. It is a role list rather than "2-D and
// F16", which would sweep in v.position_embd (shaped like a weight, used as a
// lookup, cheap to keep exact).
var towerMatrix = map[jlm.Role]bool{
	jlm.RoleVPatchEmbd: true,
	jlm.RoleVAttnQ:     true, jlm.RoleVAttnK: true, jlm.RoleVAttnV: true, jlm.RoleVAttnOut: true,
	jlm.RoleVFC1: true, jlm.RoleVFC2: true, jlm.RoleVFFNGate: true,
	jlm.RoleVProj: true, jlm.RoleVProj2: true, jlm.RoleVProj3: true, jlm.RoleVMerge: true,
	jlm.RoleVRsKV: true, jlm.RoleVRsQ: true, jlm.RoleVRsK: true, jlm.RoleVRsV: true,
	jlm.RoleVRsOut: true, jlm.RoleVRsProj: true,
	jlm.RoleVDsFC1: true, jlm.RoleVDsFC2: true,
	jlm.RoleVMergeConv: true, jlm.RoleVProjGate: true, jlm.RoleVProjUp: true, jlm.RoleVProjDown: true,
}

// q8Block is the source format's Q8_0 block: an f16 scale and 32 signed bytes.
const (
	q8Elems = 32
	q8Bytes = 34
)

// reshapePatchEmbd turns the 4-D conv kernel into the 2-D matrix it already is,
// and pads its k out to a whole quantization block.
//
// A conv whose stride equals its kernel is a matmul. v.patch_embd.weight
// arrives as [kx, ky, channels, NEmbd], and jlm.spans would read dims[2..] as
// sheets of 14 columns, which kernels.PackedWords refuses. Element
// (kx, ky, c, o) sits at kx + P*ky + P*P*c + k*o, so row o is already one
// contiguous run and the reshape is free.
//
// The padding is exact: CLIP's 588-element patch pads to 608 with zero
// columns, which contribute exactly zero to a dot product.
func reshapePatchEmbd(t *jlm.Tensor, patch, nembd int) error {
	if t.NDim == 2 {
		// Already 2-D (Llama 4's unfold is a linear, and llama.cpp writes it
		// so): k only needs padding to a whole block.
		return padPatchK(t, nembd)
	}
	if t.NDim != 4 {
		return fmt.Errorf("convert: %s is %d-dimensional, want [P P C NEmbd]", t.Name, t.NDim)
	}
	if int(t.Dims[0]) != patch || int(t.Dims[1]) != patch || int(t.Dims[3]) != nembd {
		return fmt.Errorf("convert: %s is %v, want [%d %d C %d]",
			t.Name, t.Dims[:4], patch, patch, nembd)
	}
	k := patch * patch * int(t.Dims[2])
	padded := (k + q8Elems - 1) / q8Elems * q8Elems
	src, ok := sourceType(t.Type)
	if !ok {
		return fmt.Errorf("convert: %s is %v, which has no source type", t.Name, t.Type)
	}
	if src != quant.F16 && src != quant.F32 && src != quant.BF16 {
		return fmt.Errorf("convert: %s is %v; a patch embedding must be a float type "+
			"to be reshaped and padded", t.Name, t.Type)
	}
	w := int(src.BlockBytes()) / int(src.BlockElems())
	if len(t.Data) != k*nembd*w {
		return fmt.Errorf("convert: %s has %d bytes, want %d", t.Name, len(t.Data), k*nembd*w)
	}
	out := make([]byte, padded*nembd*w) // zero, which is the padding
	for r := 0; r < nembd; r++ {
		copy(out[r*padded*w:], t.Data[r*k*w:(r+1)*k*w])
	}
	t.NDim, t.Data = 2, out
	t.Dims = [4]uint64{uint64(padded), uint64(nembd), 0, 0}
	return nil
}

// padPatchK pads a 2-D patch embedding's k to a whole quantization block with
// zero columns, which contribute nothing.
func padPatchK(t *jlm.Tensor, nembd int) error {
	k, rows := int(t.Dims[0]), int(t.Dims[1])
	if rows != nembd {
		return fmt.Errorf("convert: %s is %dx%d, want %d rows", t.Name, k, rows, nembd)
	}
	if k%q8Elems == 0 {
		return nil
	}
	vals, _, _, err := floatRows(t)
	if err != nil {
		return err
	}
	padded := (k + q8Elems - 1) / q8Elems * q8Elems
	out := make([]float32, padded*rows)
	for r := 0; r < rows; r++ {
		copy(out[r*padded:], vals[r*k:(r+1)*k])
	}
	setF32Dims(t, out, uint64(padded), uint64(rows))
	return nil
}

// quantizeTower rewrites every matrix role to Q8_0 unless it is already a
// packed type, so that no tower weight reaches the engine in a verbatim
// layout. It is the only lossy step in the converter and it says so.
func quantizeTower(ts []jlm.Tensor) error {
	for i := range ts {
		t := &ts[i]
		if !towerMatrix[t.Role] && !jlm.AudioMatrix(t.Role) {
			continue
		}
		if jlm.Packed(t.Type) {
			continue // already this container's layout to be
		}
		src, ok := sourceType(t.Type)
		if !ok {
			return fmt.Errorf("convert: %s is %v, which has no source type", t.Name, t.Type)
		}
		if t.NDim != 2 {
			return fmt.Errorf("convert: %s is %d-dimensional; a tower matrix must be "+
				"2-D by the time it is quantized", t.Name, t.NDim)
		}
		k, rows := int(t.Dims[0]), int(t.Dims[1])
		if k%q8Elems != 0 {
			return fmt.Errorf("convert: %s has k=%d, which is not a multiple of %d, so it "+
				"cannot be quantized; a verbatim tower matrix would reach the source "+
				"format's layout at inference", t.Name, k, q8Elems)
		}
		vals := make([]float32, k*rows)
		if err := quant.Dequant32(src, t.Data, vals); err != nil {
			return fmt.Errorf("convert: %s: %w", t.Name, err)
		}
		t.Data = quantizeQ8Rows(vals, k, rows)
		t.Type = jlm.TypeQ8
	}
	return nil
}

// quantizeQ8Rows writes vals as Q8_0 source blocks.
//
// The recipe is ggml's (amax/127 stored as f16, codes rounded from the
// unrounded d), because quant.Dequant32 and kernels.PackWeights already read
// these bytes back as its inverse; another recipe would be a second
// specification of one layout.
func quantizeQ8Rows(vals []float32, k, rows int) []byte {
	nb := k / q8Elems
	out := make([]byte, rows*nb*q8Bytes)
	for r := 0; r < rows; r++ {
		for b := 0; b < nb; b++ {
			x := vals[r*k+b*q8Elems : r*k+(b+1)*q8Elems]
			amax := float32(0)
			for _, v := range x {
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			d := amax / 127
			id := float32(0)
			if d != 0 {
				id = 1 / d
			}
			o := (r*nb + b) * q8Bytes
			binary.LittleEndian.PutUint16(out[o:], f32ToF16(d))
			for j, v := range x {
				out[o+2+j] = byte(int8(math.Round(float64(v * id))))
			}
		}
	}
	return out
}

// f32ToF16 rounds to nearest even, which is what every float16 writer on the
// read side assumes.
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32((b>>23)&0xFF) - 127
	man := b & 0x7FFFFF
	switch {
	case exp == 128: // Inf or NaN
		if man != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case exp > 15:
		return sign | 0x7C00
	case exp < -24:
		return sign
	case exp < -14: // subnormal
		shift := uint32(-exp - 14)
		m := (man | 0x800000) >> (shift + 13)
		if (man|0x800000)>>(shift+12)&1 == 1 {
			m++
		}
		return sign | uint16(m)
	}
	h := uint16(uint32(exp+15)<<10) | uint16(man>>13)
	if man&0x1000 != 0 && (man&0xFFF != 0 || h&1 == 1) {
		h++
	}
	return sign | h
}

// foldPatchPlanes collapses a conv3d patch embedding into the single matrix the
// container stores, and it is exact for an image.
//
// The two planes are added, not concatenated. Qwen2-VL's ViT takes
// temporal_patch_size=2 frames, and llama.cpp runs IM2COL+MUL_MAT once per
// plane and adds the results:
//
//	node_3  = MUL_MAT(im2col(frame), v.patch_embd.weight)    {64, 1280}
//	node_10 = MUL_MAT(im2col(frame), v.patch_embd.weight.1)  {64, 1280}
//	node_14 = ADD(node_3, node_10)
//
// A still image is fed to both planes, so both read the same im2col output and
// x*W0 + x*W1 = x*(W0+W1). This is exact only for a still image: two video
// frames differ, so the second plane stays in the container beside the sum
// (RoleVPatchEmbd, index 1), and a video's pair of frames runs as
// x0*(W0+W1) + (x1-x0)*W1 = x0*W0 + x1*W1.
func foldPatchPlanes(s *jlm.Source) error {
	var base *jlm.Tensor
	for i := range s.Tensors {
		if s.Tensors[i].Role == jlm.RoleVPatchEmbd && s.Tensors[i].Index < 0 {
			base = &s.Tensors[i]
			break
		}
	}
	keep := s.Tensors[:0]
	var folded int
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleVPatchEmbd || t.Index < 0 {
			keep = append(keep, *t)
			continue
		}
		if base == nil {
			return fmt.Errorf("convert: %s is a second patch-embedding plane and "+
				"there is no first one", t.Name)
		}
		if t.Type != base.Type || t.NDim != base.NDim || t.Dims != base.Dims {
			return fmt.Errorf("convert: patch-embedding planes disagree: %s is %v %v, %s is %v %v",
				base.Name, base.Type, base.Dims, t.Name, t.Type, t.Dims)
		}
		sum, err := addPlanes(base.Data, t.Data, base.Type)
		if err != nil {
			return fmt.Errorf("convert: %s + %s: %w", base.Name, t.Name, err)
		}
		base.Data = sum
		keep = append(keep, *t)
		folded++
	}
	s.Tensors = keep
	if folded > 1 {
		return fmt.Errorf("convert: %d extra patch-embedding planes, this fold handles one", folded)
	}
	return nil
}

// addPlanes adds two dequantized-to-f32 planes and returns them in the source
// type. Only the float types a patch embedding ships are handled: re-quantizing
// a sum would add error the reference never incurs.
func addPlanes(a, b []byte, t jlm.Type) ([]byte, error) {
	switch t {
	case jlm.TypeF32:
		if len(a) != len(b) || len(a)%4 != 0 {
			return nil, fmt.Errorf("f32 planes are %d and %d bytes", len(a), len(b))
		}
		out := make([]byte, len(a))
		for i := 0; i < len(a); i += 4 {
			x := math.Float32frombits(binary.LittleEndian.Uint32(a[i:]))
			y := math.Float32frombits(binary.LittleEndian.Uint32(b[i:]))
			binary.LittleEndian.PutUint32(out[i:], math.Float32bits(x+y))
		}
		return out, nil
	case jlm.TypeF16:
		if len(a) != len(b) || len(a)%2 != 0 {
			return nil, fmt.Errorf("f16 planes are %d and %d bytes", len(a), len(b))
		}
		out := make([]byte, len(a))
		for i := 0; i < len(a); i += 2 {
			x := float32(quant.DecodeHalf(binary.LittleEndian.Uint16(a[i:])))
			y := float32(quant.DecodeHalf(binary.LittleEndian.Uint16(b[i:])))
			binary.LittleEndian.PutUint16(out[i:], quant.EncodeHalf(x+y))
		}
		return out, nil
	}
	return nil, fmt.Errorf("a %v patch-embedding plane cannot be folded without requantizing", t)
}
