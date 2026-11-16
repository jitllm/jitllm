package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// A convolutional tower's tensor transforms (towerTransforms). Every
// convolution weight is laid out tap-major (ky, kx, c), which is the order
// the engine's im2col and depthwise rows read: a full or pointwise one as a
// matrix of k = K*K*Cin, a depthwise filter as K*K rows of channels.

// convLayouts lays every convolution weight out tap-major and the stem's
// bias as a flat vector, before quantizeTower rounds the matrices.
func convLayouts(s *jlm.Source) error {
	for i := range s.Tensors {
		t := &s.Tensors[i]
		var err error
		switch t.Role {
		case jlm.RoleVMNStem, jlm.RoleVMNConvExp, jlm.RoleVMNConvPwl, jlm.RoleVMNPwExp, jlm.RoleVMNPwProj,
			jlm.RoleVMNFusionExp, jlm.RoleVMNFusionProj, jlm.RoleVAttnQ, jlm.RoleVAttnK, jlm.RoleVAttnV,
			jlm.RoleVAttnOut:
			err = convMatrix(t)
		case jlm.RoleVMNDwStart, jlm.RoleVMNDwMid, jlm.RoleVMNKDown, jlm.RoleVMNVDown:
			err = depthwiseTaps(t)
		case jlm.RoleVMNStemBias:
			var x []float32
			if x, err = vecOf(t); err == nil {
				setF32Dims(t, x, uint64(len(x)))
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// foldChannelScales folds each block's per-channel layer scale into what it
// multiplies: the projection's norm in an inverted residual, the output
// matrix's rows in an attention block. Both are per output channel, so the
// fold is exact.
func foldChannelScales(s *jlm.Source) error {
	scales := map[int32][]float32{}
	kept := s.Tensors[:0]
	for i := range s.Tensors {
		t := s.Tensors[i]
		if t.Role == jlm.RoleVMNLayerScale {
			x, err := vecOf(&t)
			if err != nil {
				return err
			}
			scales[t.Block] = x
			continue
		}
		kept = append(kept, t)
	}
	s.Tensors = kept
	for b, g := range scales {
		folded := false
		for i := range s.Tensors {
			t := &s.Tensors[i]
			if t.Block != b {
				continue
			}
			switch t.Role {
			case jlm.RoleVMNPwProjNorm:
				x, err := vecOf(t)
				if err != nil {
					return err
				}
				if len(x) != len(g) {
					return fmt.Errorf("convert: block %d's layer scale is %d and its norm %d", b, len(g), len(x))
				}
				for j := range x {
					x[j] *= g[j]
				}
				setF32Dims(t, x, uint64(len(x)))
				folded = true
			case jlm.RoleVAttnOut:
				vals, k, rows, err := floatRows(t)
				if err != nil {
					return err
				}
				if rows != len(g) {
					return fmt.Errorf("convert: block %d's layer scale is %d and its output %d rows", b, len(g), rows)
				}
				for r := 0; r < rows; r++ {
					for j := 0; j < k; j++ {
						vals[r*k+j] *= g[r]
					}
				}
				setF32Dims(t, vals, uint64(k), uint64(rows))
				folded = true
			}
		}
		if !folded {
			return fmt.Errorf("convert: block %d has a layer scale and nothing to fold it into", b)
		}
	}
	return nil
}

// convMatrix turns a [kw kh Cin Cout] convolution (torch's [Cout][Cin][kh][kw])
// into the matrix the engine multiplies by: Cout rows of k = kh*kw*Cin in
// (ky, kx, c) order, padded to a whole quantization block. A 2-D tensor is
// already one (a linear's).
func convMatrix(t *jlm.Tensor) error {
	if t.NDim == 2 {
		return padK(t)
	}
	if t.NDim != 4 {
		return fmt.Errorf("convert: %s is %d-dimensional, want a convolution", t.Name, t.NDim)
	}
	kw, kh, cin, cout := int(t.Dims[0]), int(t.Dims[1]), int(t.Dims[2]), int(t.Dims[3])
	src, err := vecOf(t)
	if err != nil {
		return err
	}
	k := kh * kw * cin
	out := make([]float32, k*cout)
	for o := 0; o < cout; o++ {
		for c := 0; c < cin; c++ {
			for y := 0; y < kh; y++ {
				for x := 0; x < kw; x++ {
					out[o*k+(y*kw+x)*cin+c] = src[((o*cin+c)*kh+y)*kw+x]
				}
			}
		}
	}
	setF32Dims(t, out, uint64(k), uint64(cout))
	return padK(t)
}

// padK pads a matrix's k to a whole quantization block with zero columns.
func padK(t *jlm.Tensor) error {
	k, rows := int(t.Dims[0]), int(t.Dims[1])
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

// depthwiseTaps turns a [kw kh 1 C] depthwise filter (torch's [C][1][kh][kw])
// into K*K rows of C channels, tap-major.
func depthwiseTaps(t *jlm.Tensor) error {
	if t.NDim != 4 || t.Dims[2] != 1 || t.Dims[0] != t.Dims[1] {
		return fmt.Errorf("convert: %s is %v, want a square depthwise filter [K K 1 C]", t.Name, t.Dims[:t.NDim])
	}
	kk, c := int(t.Dims[0]*t.Dims[1]), int(t.Dims[3])
	src, err := vecOf(t)
	if err != nil {
		return err
	}
	out := make([]float32, kk*c)
	for ch := 0; ch < c; ch++ {
		for tap := 0; tap < kk; tap++ {
			out[tap*c+ch] = src[ch*kk+tap]
		}
	}
	setF32Dims(t, out, uint64(c), uint64(kk))
	return nil
}

// hardTokenEmbeddings folds a tower's own token table into the rows its
// tokens embed to: for an id from offset on, post_norm(W · norm(table[id -
// offset])), with W the projection the picture's rows go through and
// post_norm unweighted (transformers' Gemma3nMultimodalEmbedder's hard path).
// The offset is stated beside the rows. Offline arithmetic in float64, so the
// reader does none of it.
func hardTokenEmbeddings(s *jlm.Source, offset float32) error {
	var tab, nrm, proj *jlm.Tensor
	kept := s.Tensors[:0]
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch t.Role {
		case jlm.RoleVMNHardTable:
			x := *t
			tab = &x
			continue
		case jlm.RoleVMNHardNorm:
			x := *t
			nrm = &x
			continue
		case jlm.RoleVProj:
			x := *t
			proj = &x
		}
		kept = append(kept, *t)
	}
	s.Tensors = kept
	if tab == nil && nrm == nil {
		return nil
	}
	if tab == nil || nrm == nil || proj == nil {
		return fmt.Errorf("convert: the tower's token embedder is missing its table, its norm or its projection")
	}
	if offset <= 0 {
		return fmt.Errorf("convert: the tower embeds its own tokens and its entry states no hard_tokens offset")
	}
	tv, err := vecOf(tab)
	if err != nil {
		return err
	}
	nv, err := vecOf(nrm)
	if err != nil {
		return err
	}
	pv, k, rows, err := floatRows(proj)
	if err != nil {
		return err
	}
	if len(nv) != k || len(tv)%k != 0 {
		return fmt.Errorf("convert: the tower's token table is %d floats and its norm %d, for a %d-wide projection",
			len(tv), len(nv), k)
	}
	eps := float64(s.Vision.Eps)
	n := len(tv) / k
	out := make([]float32, n*rows)
	x := make([]float64, k)
	y := make([]float64, rows)
	for i := 0; i < n; i++ {
		row := tv[i*k : (i+1)*k]
		ss := 0.0
		for j, v := range row {
			ss += float64(v) * float64(v)
			x[j] = float64(v)
		}
		inv := 1 / math.Sqrt(ss/float64(k)+eps)
		for j := range x {
			x[j] *= inv * float64(nv[j])
		}
		ss = 0
		for r := 0; r < rows; r++ {
			acc := 0.0
			w := pv[r*k : (r+1)*k]
			for j := range x {
				acc += float64(w[j]) * x[j]
			}
			y[r] = acc
			ss += acc * acc
		}
		inv = 1 / math.Sqrt(ss/float64(rows)+eps)
		for r := range y {
			out[i*rows+r] = float32(y[r] * inv)
		}
	}
	e := jlm.Tensor{Role: jlm.RoleVMNHardEmbd, Block: jlm.DenseBlock, Index: -1, Name: "mm.hard_embd"}
	setF32Dims(&e, out, uint64(rows), uint64(n))
	o := jlm.Tensor{Role: jlm.RoleVMNHardOffset, Block: jlm.DenseBlock, Index: -1, Name: "mm.hard_offset"}
	setF32Dims(&o, []float32{offset}, 1)
	s.Tensors = append(s.Tensors, e, o)
	return nil
}
