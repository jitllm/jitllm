package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// The state-space hybrids (jlm.LayerSSD): Mamba-2's mixer, alone (mamba2) or
// beside attention (granitehybrid, nemotron_h). The recurrence is the delta
// rule's state with the key dot removed, so the container states it in the
// delta rule's terms: B is the key (Groups heads of StateSize), C the query
// and x the value (NHeadV heads of Inner/NHeadV), and the convolved channels
// are q | k | v. That last is a reordering: Mamba's in_proj and conv1d run
// x | B | C, and unfuseSSD permutes their rows once here so that deltaGeom
// slices a Mamba-2 block exactly as it slices qwen3next's.

// ssdArch reports the architectures whose recurrent layers are Mamba-2's.
func ssdArch(a Arch) bool {
	switch a {
	case ArchMamba2, ArchGraniteHybrid, ArchNemotronH, ArchFalconH1:
		return true
	}
	return false
}

// attnLayerCount is perLayerCount for a hybrid, whose per-layer arrays carry
// zeros on the recurrent layers (llama.cpp's is_recr test reads exactly that
// zero): the attention layers' entries must agree, and an array of zeros --
// a model with no attention at all -- is zero.
func attnLayerCount(f *meta.File, key string, def uint32) (uint32, error) {
	kv, ok := f.Key(key)
	if !ok {
		return def, nil
	}
	if v, ok := kv.Uint(); ok {
		return uint32(v), nil
	}
	vs, err := kv.Int32s()
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	var n int32
	for i, v := range vs {
		switch {
		case v < 0:
			return 0, fmt.Errorf("%s: %d at layer %d", key, v, i)
		case v == 0:
		case n == 0:
			n = v
		case v != n:
			return 0, fmt.Errorf("%s differs between attention layers (%d and %d at layer %d): %w",
				key, n, v, i, ErrNotImplemented)
		}
	}
	return uint32(n), nil
}

// ssdConfig reads the Mamba-2 geometry and each family's switches into c.
func ssdConfig(f *meta.File, c *Config) error {
	c.SSM = jlm.SSMConfig{
		ConvKernel: uint32(f.UintKey("ssm.conv_kernel", 0)),
		Groups:     uint32(f.UintKey("ssm.group_count", 1)),
		Inner:      uint32(f.UintKey("ssm.inner_size", 0)),
		StateSize:  uint32(f.UintKey("ssm.state_size", 0)),
		// llama.cpp writes Mamba-2's head count as the time-step rank: one dt
		// per head, which is the delta rule's one decay per value head.
		NHeadV: uint32(f.UintKey("ssm.time_step_rank", 0)),
	}
	s := c.SSM
	switch {
	case s.ConvKernel < 2 || s.Inner == 0 || s.StateSize == 0 || s.NHeadV == 0 || s.Groups == 0:
		return fmt.Errorf("incomplete Mamba-2 geometry %+v", s)
	case s.Inner%s.NHeadV != 0:
		return fmt.Errorf("Mamba-2 inner size %d is not %d whole heads", s.Inner, s.NHeadV)
	case s.NHeadV%s.Groups != 0:
		return fmt.Errorf("Mamba-2's %d heads do not divide into %d groups", s.NHeadV, s.Groups)
	case s.Inner%s.Groups != 0:
		return fmt.Errorf("Mamba-2 inner size %d does not divide into %d norm groups", s.Inner, s.Groups)
	}
	// FalconMamba's dt/B/C norms are Mamba-1's; a Mamba-2 file has none.
	if kv, ok := f.Key("ssm.dt_b_c_rms"); ok {
		if v, ok := kv.Uint(); ok && v != 0 {
			return fmt.Errorf("ssm.dt_b_c_rms on a Mamba-2 model: %w", ErrNotImplemented)
		}
	}
	switch c.Arch {
	case ArchMamba2:
		// No layer attends; the placeholder head configOf set rotates nothing.
		c.Flags |= jlm.FlagNoPosEnc
	case ArchGraniteHybrid:
		return graniteHybridConfig(f, c)
	case ArchNemotronH:
		return nemotronConfig(f, c)
	case ArchFalconH1:
		// llama.cpp's rope-type table puts falcon-h1 with NEOX; every muP
		// multiplier is already in the weights (its converter folds them).
		c.Flags |= jlm.FlagRopeNeox
	}
	return nil
}

// unfuseSSD splits Mamba-2's in_proj and puts its convolved channels in the
// delta rule's order.
//
// in_proj is z | xBC | dt by rows (Inner, Inner + 2*Groups*StateSize,
// NHeadV): z becomes the output gate (RoleAttnGate), dt the gate projection
// (RoleSSMBA) and xBC the mixed projection (RoleAttnQKV) with its three parts
// in the order C | B | x. The convolution's filters and bias are permuted the
// same way, so the convolution runs over q | k | v. GGUF rows are byte ranges,
// so all of it is copies; nothing is re-quantized.
func unfuseSSD(e *jlm.Tensor, c *Config) ([]jlm.Tensor, bool, error) {
	if c == nil || !ssdArch(c.Arch) {
		return nil, false, nil
	}
	s := c.SSM
	inner, bc, heads := uint64(s.Inner), uint64(s.Groups)*uint64(s.StateSize), uint64(s.NHeadV)
	chans := inner + 2*bc
	// Rows of the xBC block in the container's order: C, then B, then x.
	reorder := [][2]uint64{{inner + bc, bc}, {inner, bc}, {0, inner}}
	switch e.Role {
	case jlm.RoleSSMInProj:
		if e.NDim != 2 || e.Dims[1] != inner+chans+heads {
			return nil, false, fmt.Errorf("convert: %s is %v, want %d rows (z %d + xBC %d + dt %d)",
				e.Name, e.Dims[:e.NDim], inner+chans+heads, inner, chans, heads)
		}
		rb, err := tensorRowBytes(e)
		if err != nil {
			return nil, false, err
		}
		rows := func(lo, n uint64) []byte { return e.Data[lo*rb : (lo+n)*rb] }
		xbc := make([]byte, 0, chans*rb)
		for _, r := range reorder {
			xbc = append(xbc, rows(inner+r[0], r[1])...)
		}
		part := func(role jlm.Role, data []byte, n uint64) jlm.Tensor {
			t := jlm.Tensor{Role: role, Block: e.Block, Index: e.Index, Type: e.Type,
				NDim: 2, Data: data, Name: e.Name + "/" + role.String()}
			t.Dims[0], t.Dims[1] = e.Dims[0], n
			return t
		}
		return []jlm.Tensor{
			part(jlm.RoleAttnGate, rows(0, inner), inner),
			part(jlm.RoleAttnQKV, xbc, chans),
			part(jlm.RoleSSMBA, rows(inner+chans, heads), heads),
		}, true, nil
	case jlm.RoleSSMConv1d, jlm.RoleSSMConvBias:
		// conv1d is [taps, channels] (a row a channel); its bias one value a
		// channel. Either way a channel is one row of the tensor.
		n := e.Dims[0]
		if e.Role == jlm.RoleSSMConv1d {
			if e.NDim != 2 {
				return nil, false, fmt.Errorf("convert: %s is %v, want [taps, channels]", e.Name, e.Dims[:e.NDim])
			}
			n = e.Dims[1]
		}
		if n != chans {
			return nil, false, fmt.Errorf("convert: %s has %d channels, Mamba-2 convolves %d", e.Name, n, chans)
		}
		rb, err := tensorRowBytes(e)
		if err != nil {
			return nil, false, err
		}
		if e.Role == jlm.RoleSSMConvBias {
			rb = uint64(len(e.Data)) / chans
		}
		if rb*chans != uint64(len(e.Data)) {
			return nil, false, fmt.Errorf("convert: %s holds %d bytes for %d channels", e.Name, len(e.Data), chans)
		}
		out := make([]byte, 0, len(e.Data))
		for _, r := range reorder {
			out = append(out, e.Data[r[0]*rb:(r[0]+r[1])*rb]...)
		}
		t := *e
		t.Data = out
		return []jlm.Tensor{t}, true, nil
	}
	return nil, false, nil
}

// tensorRowBytes is how many bytes one row (Dims[0] elements) of e occupies.
func tensorRowBytes(e *jlm.Tensor) (uint64, error) {
	src, ok := sourceType(e.Type)
	if !ok {
		return 0, fmt.Errorf("convert: %s is %v, which has no source type", e.Name, e.Type)
	}
	be, bb := uint64(src.BlockElems()), uint64(src.BlockBytes())
	if be == 0 || e.Dims[0]%be != 0 {
		return 0, fmt.Errorf("convert: %s rows are %d elements, not a multiple of %s's %d",
			e.Name, e.Dims[0], src, be)
	}
	return e.Dims[0] / be * bb, nil
}
