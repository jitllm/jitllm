package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// GLM-4.xV's tower tensors (jlm.ProjGLM4V) as llama.cpp's converter writes
// them: the patch embedding's RMSNorm, the 2x2 merging convolution and the
// projector's gated MLP have names of their own, and the convolution is a
// 4-D kernel the container stores as the matrix it is.

// glm4vRole is GLM-4.xV's own tower vocabulary.
func glm4vRole(name string) (jlm.Role, bool) {
	switch name {
	case "v.norm_embd.weight":
		return jlm.RoleVEmbdNorm, true
	case "mm.patch_merger.weight":
		return jlm.RoleVMergeConv, true
	case "mm.patch_merger.bias":
		return jlm.RoleVMergeConvBias, true
	case "mm.post_norm.weight":
		return jlm.RoleVProjNorm, true
	case "mm.post_norm.bias":
		return jlm.RoleVProjNormBias, true
	case "mm.gate.weight":
		return jlm.RoleVProjGate, true
	case "mm.gate.bias":
		return jlm.RoleVProjGateBias, true
	case "mm.up.weight":
		return jlm.RoleVProjUp, true
	case "mm.up.bias":
		return jlm.RoleVProjUpBias, true
	case "mm.down.weight":
		return jlm.RoleVProjDown, true
	case "mm.down.bias":
		return jlm.RoleVProjDownBias, true
	}
	return 0, false
}

// kimivlRole is Kimi-VL's: the projector's LayerNorm over each patch.
func kimivlRole(name string) (jlm.Role, bool) {
	switch name {
	case "mm.input_norm.weight":
		return jlm.RoleVProjInNorm, true
	case "mm.input_norm.bias":
		return jlm.RoleVProjInNormBias, true
	}
	return 0, false
}

// reshapeMergeConv lays the merging convolution out as the matrix the
// projector multiplies the shuffled row by. The GGUF kernel is
// [kx, ky, channel, out] (ggml order: kx fastest), and the shuffled row the
// engine builds is (dy, dx, channel) with the channel fastest, so element
// (kx, ky, c) of output o moves to column (ky*S + kx)*NEmbd + c. A
// convolution whose stride equals its kernel over a merge group is that
// matrix exactly; only the column order is the layout.
func reshapeMergeConv(t *jlm.Tensor, nembd int) error {
	if t.NDim == 2 {
		return nil
	}
	if t.NDim != 4 || t.Dims[0] != t.Dims[1] || int(t.Dims[2]) != nembd {
		return fmt.Errorf("convert: %s is %v, want [S S %d out]", t.Name, t.Dims[:t.NDim], nembd)
	}
	vals, err := normF32(t)
	if err != nil {
		return err
	}
	S, out := int(t.Dims[0]), int(t.Dims[3])
	k := S * S * nembd
	if len(vals) != k*out {
		return fmt.Errorf("convert: %s holds %d values, want %d", t.Name, len(vals), k*out)
	}
	m := make([]float32, k*out)
	for o := 0; o < out; o++ {
		for c := 0; c < nembd; c++ {
			for ky := 0; ky < S; ky++ {
				for kx := 0; kx < S; kx++ {
					m[o*k+(ky*S+kx)*nembd+c] = vals[o*k+kx+S*ky+S*S*c]
				}
			}
		}
	}
	setF32Dims(t, m, uint64(k), uint64(out))
	return nil
}
