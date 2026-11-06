package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// Mamba-1's selective scan (jlm.LayerScan): plain Mamba and FalconMamba
// (arch "mamba") and Jamba's recurrent blocks. In the delta rule's terms every
// one of Inner channels is a value head of one row: NHeadV is Inner and the
// state is StateSize wide per channel, B and C shared by every channel. The
// converter does three things to the source's layout, all of them copies:
//
//	in_proj  x | z by rows -> x (RoleAttnQKV, what convolves) and z (RoleAttnGate)
//	x_proj   dt | B | C by rows -> RoleSSMXDt, RoleSSMXB, RoleSSMXC
//	dt/B/C   FalconMamba's weightless RMSNorms written as ones, so the engine
//	         runs them exactly as Jamba's weighted ones
//
// dt_proj (ssm_dt.weight, RoleSSMDtProj) takes the bottleneck back up to the
// channels; its rank is read off that tensor, not stated in the config.

// mamba1Arch reports the architectures whose recurrent layers are Mamba-1's.
func mamba1Arch(a jlm.Arch) bool { return a == jlm.ArchMamba || a == jlm.ArchJamba }

// mamba1Config reads the Mamba-1 geometry into c.
func mamba1Config(f *meta.File, c *jlm.Config) error {
	inner := uint32(f.UintKey("ssm.inner_size", 0))
	c.SSM = jlm.SSMConfig{
		ConvKernel: uint32(f.UintKey("ssm.conv_kernel", 0)),
		Groups:     1,
		Inner:      inner,
		StateSize:  uint32(f.UintKey("ssm.state_size", 0)),
		NHeadV:     inner,
	}
	s := c.SSM
	if s.ConvKernel < 2 || s.Inner == 0 || s.StateSize == 0 || f.UintKey("ssm.time_step_rank", 0) == 0 {
		return fmt.Errorf("incomplete Mamba-1 geometry %+v (time_step_rank %d)", s, f.UintKey("ssm.time_step_rank", 0))
	}
	// Neither family rotates: Jamba's attention carries no position at all,
	// and a plain Mamba has no attention (configOf set a placeholder head).
	c.Flags |= jlm.FlagNoPosEnc
	if c.Arch == jlm.ArchJamba {
		// JambaSparseMoeBlock takes the top-k of the softmax and never
		// renormalises; llama.cpp's builder passes false as a literal. A file
		// stating otherwise describes a model neither runs.
		if kv, ok := f.Key("expert_weights_norm"); ok {
			if v, _ := kv.Uint(); v != 0 {
				return fmt.Errorf("expert_weights_norm true, and Jamba's router does not "+
					"renormalise: %w", ErrNotImplemented)
			}
		}
		c.Flags |= jlm.FlagNoExpertNorm
	}
	return nil
}

// mamba1Kinds is a Mamba-1 model's layer map: a block with a convolution is a
// Mamba-1 mixer, any other attends. llama.cpp's jamba tests a zero
// head_count_kv instead; the tensors say the same thing.
func mamba1Kinds(f *meta.File, n uint32) ([]jlm.LayerKind, error) {
	kinds := make([]jlm.LayerKind, n)
	rec := 0
	for i := range kinds {
		kinds[i] = jlm.LayerFullAttn
		if _, ok := f.Get("blk." + itoa(i) + ".ssm_conv1d.weight"); ok {
			kinds[i], rec = jlm.LayerScan, rec+1
		}
	}
	if rec == 0 {
		return nil, fmt.Errorf("no layer carries a Mamba-1 mixer")
	}
	return kinds, nil
}

// unfuseMamba1 splits in_proj and x_proj into their outputs (see the top of
// this file). Row ranges of a row-major matrix are byte ranges, so the bytes
// are the source's.
func unfuseMamba1(e *jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, bool, error) {
	if c == nil || !mamba1Arch(c.Arch) {
		return nil, false, nil
	}
	inner, n := uint64(c.SSM.Inner), uint64(c.SSM.StateSize)
	var parts []struct {
		role jlm.Role
		rows uint64
	}
	switch e.Role {
	case jlm.RoleSSMInProj:
		if e.NDim != 2 || e.Dims[1] != 2*inner {
			return nil, false, fmt.Errorf("convert: %s is %v, want %d rows (x and z of %d)",
				e.Name, e.Dims[:e.NDim], 2*inner, inner)
		}
		parts = []struct {
			role jlm.Role
			rows uint64
		}{{jlm.RoleAttnQKV, inner}, {jlm.RoleAttnGate, inner}}
	case jlm.RoleSSMXDt:
		if e.NDim != 2 || e.Dims[0] != inner || e.Dims[1] <= 2*n {
			return nil, false, fmt.Errorf("convert: %s is %v, want %d inputs and a dt rank plus %d rows",
				e.Name, e.Dims[:e.NDim], inner, 2*n)
		}
		parts = []struct {
			role jlm.Role
			rows uint64
		}{{jlm.RoleSSMXDt, e.Dims[1] - 2*n}, {jlm.RoleSSMXB, n}, {jlm.RoleSSMXC, n}}
	default:
		return nil, false, nil
	}
	rb, err := tensorRowBytes(e)
	if err != nil {
		return nil, false, err
	}
	var out []jlm.Tensor
	lo := uint64(0)
	for _, p := range parts {
		t := jlm.Tensor{Role: p.role, Block: e.Block, Index: e.Index, Type: e.Type, NDim: 2,
			Data: e.Data[lo*rb : (lo+p.rows)*rb], Name: e.Name + "/" + p.role.String()}
		t.Dims[0], t.Dims[1] = e.Dims[0], p.rows
		out = append(out, t)
		lo += p.rows
	}
	return out, true, nil
}

// mamba1UnitNorms writes FalconMamba's weightless dt, B and C RMSNorms
// (ssm.dt_b_c_rms) as weights of ones, one set per Mamba-1 block, so the
// engine runs them exactly as Jamba's weighted ones. rank is each block's dt
// bottleneck, read off its x_proj.
func mamba1UnitNorms(f *meta.File, s *jlm.Source) error {
	c := s.Config
	if c == nil || c.Arch != jlm.ArchMamba {
		return nil
	}
	if kv, ok := f.Key("ssm.dt_b_c_rms"); !ok {
		return nil
	} else if v, _ := kv.Uint(); v == 0 {
		return nil
	}
	ones := func(n uint64) []byte {
		b := make([]byte, 4*n)
		for i := uint64(0); i < n; i++ {
			u := math.Float32bits(1)
			b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
		}
		return b
	}
	rank := map[int32]uint64{}
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Role == jlm.RoleSSMXDt {
			rank[t.Block] = t.Dims[1]
		}
	}
	n := uint64(c.SSM.StateSize)
	for b, k := range c.LayerKinds {
		if k != jlm.LayerScan {
			continue
		}
		r, ok := rank[int32(b)]
		if !ok {
			return fmt.Errorf("convert: block %d is a Mamba-1 mixer with no x_proj", b)
		}
		for _, v := range []struct {
			role jlm.Role
			n    uint64
		}{{jlm.RoleSSMDtNorm, r}, {jlm.RoleSSMBNorm, n}, {jlm.RoleSSMCNorm, n}} {
			t := jlm.Tensor{Role: v.role, Block: int32(b), Index: -1, Type: jlm.TypeF32, NDim: 1,
				Data: ones(v.n), Name: "blk." + itoa(b) + "." + v.role.String() + ".weight"}
			t.Dims[0] = v.n
			s.Tensors = append(s.Tensors, t)
		}
	}
	return nil
}
