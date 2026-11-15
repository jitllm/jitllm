package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// unfuseLFM2 splits a short convolution's in_proj ({n_embd, 3*n_embd}) into
// its three outputs, in the order llama.cpp's lfm2 graph views them: B, C,
// then x. Row ranges of a row-major matrix are byte ranges, so the bytes are
// the source's.
func unfuseLFM2(e *jlm.Tensor, c *Config) ([]jlm.Tensor, bool, error) {
	if c == nil || c.Arch != ArchLFM2 && c.Arch != ArchLFM2Enc || e.Role != jlm.RoleSSMInProj {
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
