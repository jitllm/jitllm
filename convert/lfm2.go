package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// lfm2Config reads LFM2's short convolution into c: a window of
// shortconv.l_cache taps over every channel of the residual, in the delta
// rule's terms a convolution with no state beside it (SSM.Inner is NEmbd,
// and there are no heads).
func lfm2Config(f *meta.File, c *jlm.Config) error {
	l := uint32(f.UintKey("shortconv.l_cache", 0))
	if l < 2 {
		return fmt.Errorf("shortconv.l_cache %d: a short convolution needs two taps or more", l)
	}
	c.SSM = jlm.SSMConfig{ConvKernel: l, Inner: c.NEmbd}
	// llama.cpp's rope-type table puts lfm2 with NEOX.
	c.Flags |= jlm.FlagRopeNeox
	// The q/k RMSNorm per head is on every attention block (llama.cpp loads
	// it unconditionally); the file says so.
	for i := uint32(0); i < c.NLayer; i++ {
		if _, ok := f.Get("blk." + itoa(int(i)) + ".attn_q_norm.weight"); ok {
			c.Flags |= jlm.FlagQKNorm
			break
		}
	}
	return nil
}

// unfuseLFM2 splits a short convolution's in_proj ({n_embd, 3*n_embd}) into
// its three outputs, in the order llama.cpp's lfm2 graph views them: B, C,
// then x. Row ranges of a row-major matrix are byte ranges, so the bytes are
// the source's.
func unfuseLFM2(e *jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, bool, error) {
	if c == nil || c.Arch != jlm.ArchLFM2 && c.Arch != jlm.ArchLFM2Enc || e.Role != jlm.RoleSSMInProj {
		return nil, false, nil
	}
	n := uint64(c.NEmbd)
	if e.NDim != 2 || e.Dims[1] != 3*n {
		return nil, false, fmt.Errorf("convert: %s is %v, want %d rows (B, C and x of %d)",
			e.Name, e.Dims[:e.NDim], 3*n, n)
	}
	rb, err := tensorRowBytes(e)
	if err != nil {
		return nil, false, err
	}
	var out []jlm.Tensor
	for i, role := range []jlm.Role{jlm.RoleSCB, jlm.RoleSCC, jlm.RoleSCX} {
		t := jlm.Tensor{Role: role, Block: e.Block, Index: e.Index, Type: e.Type, NDim: 2,
			Data: e.Data[uint64(i)*n*rb : uint64(i+1)*n*rb], Name: e.Name + "/" + role.String()}
		t.Dims[0], t.Dims[1] = e.Dims[0], n
		out = append(out, t)
	}
	return out, true, nil
}
