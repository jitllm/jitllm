package convert

import (
	"fmt"

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

// blockIs reports whether block b's stated kind is one of kinds.
func blockIs(c *Config, b int32, kinds ...jlm.LayerKind) bool {
	if c == nil || b < 0 || int(b) >= len(c.LayerKinds) {
		return false
	}
	for _, k := range kinds {
		if c.LayerKinds[b] == k {
			return true
		}
	}
	return false
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
	if !blockIs(c, e.Block, jlm.LayerSSD, jlm.LayerSSDAttn) {
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
