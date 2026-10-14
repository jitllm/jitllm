package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// qwen35Of reads what a Qwen3.5 / Qwen3.6 file says beyond qwen3next's keys:
// the value-head order and the M-RoPE sections. Its multi-token-prediction
// blocks are nextnOf's, as every architecture's are. It is a no-op on a qwen3next file (no nextn layers, no sections).
func qwen35Of(f *meta.File, name string, c *jlm.Config) error {
	// The value heads are tiled in every qwen35 GGUF: llama.cpp's converter
	// reorders them from HuggingFace's grouped order so its graph can
	// ggml_repeat the keys. With one value head per key head the permutation is
	// the identity, so the flag is set only where it changes the pairing.
	if (name == "qwen35" || name == "qwen35moe" || name == "qwen4exp" || name == "clef") && c.SSM.Groups > 0 && c.SSM.NHeadV > c.SSM.Groups {
		if c.SSM.NHeadV%c.SSM.Groups != 0 {
			return fmt.Errorf("%d value heads over %d key heads is not a whole number per key head",
				c.SSM.NHeadV, c.SSM.Groups)
		}
		c.Flags |= jlm.FlagDeltaKeyTiled
	}

	// The sections are pairs and must cover the rotary width, as qwen2vl's must.
	// For a text prompt they select nothing; they are stored for image rows.
	if kv, ok := f.Key("rope.dimension_sections"); ok {
		secs, err := kv.Int32s()
		if err != nil {
			return fmt.Errorf("rope.dimension_sections: %w", err)
		}
		if len(secs) > len(c.RopeSections) {
			return fmt.Errorf("%d rope sections, this format defines %d", len(secs), len(c.RopeSections))
		}
		var sum uint32
		for i, v := range secs {
			if v < 0 {
				return fmt.Errorf("rope section %d is %d", i, v)
			}
			c.RopeSections[i] = uint32(v)
			sum += uint32(v)
		}
		if sum != c.NRot/2 {
			return fmt.Errorf("rope sections %v sum to %d, and a head has %d rotary pairs", secs, sum, c.NRot/2)
		}
	}
	return nil
}

// fuseBetaAlpha folds a Qwen3.5 layer's separate beta and alpha projections
// into the ssm_ba tensor qwen3next ships and the whole engine reads.
//
// The layout is qwen3next's row interleave, not a concatenation:
// engine/model/delta.go and the device's SplitDeltaGates read ssm_ba as
// [kHeads][beta: rep rows | alpha: rep rows]. Each file tensor has one row per
// value head in the file's head order, so interleaving by key head gives
// every value head its own pair whatever that order is. Stacking the halves
// also runs, and pairs each beta with another head's decay.
//
// When the two share a type the interleave is byte copies; otherwise both are
// dequantized and re-stored as Q8_0 along the same k (tiny: two rows per value
// head).
func fuseBetaAlpha(f *meta.File, beta *meta.Tensor, c *jlm.Config, block int32) (jlm.Tensor, error) {
	alphaName := strings.TrimSuffix(beta.Name, "ssm_beta.weight") + "ssm_alpha.weight"
	alpha, ok := f.Get(alphaName)
	if !ok {
		return jlm.Tensor{}, fmt.Errorf("convert: %s has no %s to fuse with", beta.Name, alphaName)
	}
	nv := uint64(c.SSM.NHeadV)
	nk := uint64(c.SSM.Groups)
	if nk == 0 || nv%nk != 0 {
		return jlm.Tensor{}, fmt.Errorf("convert: %s: %d value heads over %d key heads", beta.Name, nv, nk)
	}
	rep := nv / nk
	for _, t := range []*meta.Tensor{beta, alpha} {
		if len(t.Dims) != 2 || t.Dims[1] != nv {
			return jlm.Tensor{}, fmt.Errorf("convert: %s is %v, want [%d, %d] (one row per value head)",
				t.Name, t.Dims, c.NEmbd, nv)
		}
	}
	if beta.Dims[0] != alpha.Dims[0] {
		return jlm.Tensor{}, fmt.Errorf("convert: %s and %s disagree on k (%d, %d)",
			beta.Name, alphaName, beta.Dims[0], alpha.Dims[0])
	}
	k := beta.Dims[0]
	interleave := func(b, a []byte, rowBytes uint64) []byte {
		out := make([]byte, 0, 2*nv*rowBytes)
		for kh := uint64(0); kh < nk; kh++ {
			out = append(out, b[kh*rep*rowBytes:(kh+1)*rep*rowBytes]...)
			out = append(out, a[kh*rep*rowBytes:(kh+1)*rep*rowBytes]...)
		}
		return out
	}
	e := jlm.Tensor{Role: jlm.RoleSSMBA, Block: block, Index: -1, NDim: 2,
		Name: strings.TrimSuffix(beta.Name, "ssm_beta.weight") + "ssm_ba(fused)"}
	e.Dims[0], e.Dims[1] = k, 2*nv
	if beta.Type == alpha.Type {
		ty, err := typeOf(beta.Type)
		if err != nil {
			return jlm.Tensor{}, fmt.Errorf("convert: %s: %w", beta.Name, err)
		}
		be, bb := uint64(beta.Type.BlockElems()), uint64(beta.Type.BlockBytes())
		if be == 0 || k%be != 0 {
			return jlm.Tensor{}, fmt.Errorf("convert: %s rows are %d elements, not whole %s blocks", beta.Name, k, beta.Type)
		}
		rb := k / be * bb
		bd, ad := f.Bytes(beta), f.Bytes(alpha)
		if uint64(len(bd)) < nv*rb || uint64(len(ad)) < nv*rb {
			return jlm.Tensor{}, fmt.Errorf("convert: %s/%s hold %d/%d bytes, want %d", beta.Name, alphaName, len(bd), len(ad), nv*rb)
		}
		e.Type, e.Data = ty, interleave(bd, ad, rb)
		return e, nil
	}
	rows := func(t *meta.Tensor) ([]float32, error) {
		out := make([]float32, nv*k)
		if err := quant.Dequant32(t.Type, f.Bytes(t), out); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
		}
		return out, nil
	}
	bf, err := rows(beta)
	if err != nil {
		return jlm.Tensor{}, err
	}
	af, err := rows(alpha)
	if err != nil {
		return jlm.Tensor{}, err
	}
	fused := make([]float32, 0, 2*nv*k)
	for kh := uint64(0); kh < nk; kh++ {
		fused = append(fused, bf[kh*rep*k:(kh+1)*rep*k]...)
		fused = append(fused, af[kh*rep*k:(kh+1)*rep*k]...)
	}
	if k%q8Elems == 0 {
		e.Type, e.Data = jlm.TypeQ8, quantizeQ8Rows(fused, int(k), int(2*nv))
		return e, nil
	}
	e.Type, e.Data = jlm.TypeF32, f32AsBytes(fused)
	return e, nil
}

// splitBetaAlpha reports a Qwen3.5 gate projection that sourceOf fuses into
// ssm_ba before the name table sees it.
func splitBetaAlpha(name string) bool {
	return strings.HasSuffix(name, ".ssm_beta.weight") || strings.HasSuffix(name, ".ssm_alpha.weight")
}

// textBlock is the block index of a "blk.N." tensor name.
func textBlock(name string) (int32, bool) {
	rest, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return 0, false
	}
	i := strings.IndexByte(rest, '.')
	if i <= 0 {
		return 0, false
	}
	n, err := strconv.Atoi(rest[:i])
	if err != nil || n < 0 {
		return 0, false
	}
	return int32(n), true
}
