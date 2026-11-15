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
func mamba1Arch(a Arch) bool { return a == ArchMamba || a == ArchJamba }

// neverRenormalises states a router that never renormalises its weights
// (llama.cpp's builder passes false as a literal), refusing a file whose
// expert_weights_norm says otherwise: it describes a model neither
// reference runs, and the generic reading would obey it.
func neverRenormalises(f *meta.File, c *Config, router string) error {
	if kv, ok := f.Key("expert_weights_norm"); ok {
		if v, _ := kv.Uint(); v != 0 {
			return fmt.Errorf("expert_weights_norm true, and %s does not renormalise: %w", router, ErrNotImplemented)
		}
	}
	c.Flags |= jlm.FlagNoExpertNorm
	return nil
}

// unfuseMamba1 splits in_proj and x_proj into their outputs (see the top of
// this file). Row ranges of a row-major matrix are byte ranges, so the bytes
// are the source's.
func unfuseMamba1(e *jlm.Tensor, c *Config) ([]jlm.Tensor, bool, error) {
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
func mamba1UnitNorms(c *Config, f *meta.File, s *jlm.Source) error {
	if c == nil || c.Arch != ArchMamba {
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
