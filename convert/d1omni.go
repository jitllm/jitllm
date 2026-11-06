package convert

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// d1-omni (LiquidAI/d1-omni-600M) as llama.cpp's converter writes it
// (conversion/lfm2.py): an lfm2 file whose attention.causal is false -- the
// LFM2.5-Encoder trunk -- and whose block_count counts the trunk's blocks
// and the decision head's after them (decision.block_count of them, Laya's
// head: LayerNorm blocks with a fused, biased q|k|v and a ReLU MLP). Every
// per-layer key is written for all of them. The trunk is read as the lfm2 it
// is, through a view of the file cut to its blocks, and stated as
// jlm.ArchLFM2Enc; the head's blocks convert after it, as Laya's do.

// d1OmniHead is the head's block count when f is d1-omni.
func d1OmniHead(f *meta.File) (uint32, bool) {
	if f.Arch() != "lfm2" {
		return 0, false
	}
	v, ok := f.Key("decision.type")
	if !ok {
		return 0, false
	}
	if s, _ := v.String(); s != jlm.DecisionD1Omni.String() {
		return 0, false
	}
	return uint32(f.UintKey("decision.block_count", 0)), true
}

// trunkView is f with block_count and every per-layer array cut to the first
// n blocks.
func trunkView(f *meta.File, n uint32) (*meta.File, error) {
	total := f.UintKey("block_count", 0)
	kv := make(map[string]meta.Value, len(f.KV))
	for k, v := range f.KV {
		if v.Type == meta.Array && v.N == total && (v.Elem == meta.Int32 || v.Elem == meta.Uint32 ||
			v.Elem == meta.Float32) {
			v.N, v.Raw = uint64(n), v.Raw[:4*n]
		}
		kv[k] = v
	}
	bc := f.Arch() + ".block_count"
	if v, ok := kv[bc]; !ok || v.Type != meta.Uint32 {
		return nil, fmt.Errorf("block_count is not a uint32")
	}
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, n)
	kv[bc] = meta.Value{Type: meta.Uint32, Raw: raw}
	view := meta.New(f.Path, kv, f.Tensors)
	return view, nil
}

// d1OmniConfig is configOf for d1-omni: the trunk's lfm2 config, made an
// encoder, with the head's blocks after it.
func d1OmniConfig(f *meta.File, head uint32) (*jlm.Config, error) {
	total := uint32(f.UintKey("block_count", 0))
	if head == 0 || head >= total {
		return nil, fmt.Errorf("convert: d1-omni with %d head blocks in %d", head, total)
	}
	if v, ok := f.Key("attention.causal"); !ok {
		return nil, fmt.Errorf("convert: d1-omni's file does not state attention.causal; its trunk is an encoder")
	} else if v.Type != meta.Bool || len(v.Raw) != 1 || v.Raw[0] != 0 {
		return nil, fmt.Errorf("convert: d1-omni with a causal trunk is not the model this graph is")
	}
	view, err := trunkView(f, total-head)
	if err != nil {
		return nil, fmt.Errorf("convert: d1-omni: %w", err)
	}
	c, err := configOfFile(view)
	if err != nil {
		return nil, err
	}
	// The FFN width is per block in the file; the trunk's are one width.
	if ffn, ok := view.Key("feed_forward_length"); ok && c.NFFN == 0 {
		vs, err := ffn.Int32s()
		if err != nil || len(vs) == 0 {
			return nil, fmt.Errorf("convert: d1-omni: feed_forward_length: %v", err)
		}
		for _, v := range vs {
			if v != vs[0] || v <= 0 {
				return nil, fmt.Errorf("convert: d1-omni: feed_forward_length differs between trunk blocks: %w",
					ErrNotImplemented)
			}
		}
		c.NFFN = uint32(vs[0])
	}
	// The trunk's context is the reference's max_length (D1OmniConfig's
	// default, 16384), not the base model's 128000 the file states: a state
	// is cut to it, and a picture prefix comes off it.
	c.NCtx = uint32(f.UintKey("decision.max_length", 16384))
	c.Arch = jlm.ArchLFM2Enc
	c.Flags |= jlm.FlagNonCausal
	return c, nil
}

// decisionHeadBlock reports whether e is in a decision head's block, after
// the trunk's.
func decisionHeadBlock(e *jlm.Tensor, c *jlm.Config) bool {
	return c.Decision != jlm.DecisionNone && c.DecisionBlocks > 0 && e.Block >= int32(c.NLayer) &&
		e.Block < int32(c.NLayer+c.DecisionBlocks)
}

// unfuseThirds splits a head block's fused q|k|v (or its bias) into three
// equal parts of the residual's width, row ranges (or element ranges) of the
// source's bytes.
func unfuseThirds(e *jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, bool, error) {
	n := uint64(c.NEmbd)
	roles := []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV}
	var unit uint64
	if e.Role == jlm.RoleAttnQKVBias {
		roles = []jlm.Role{jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias}
		if e.NDim != 1 || e.Dims[0] != 3*n {
			return nil, false, fmt.Errorf("convert: %s is %v, want %d", e.Name, e.Dims[:e.NDim], 3*n)
		}
		unit = uint64(len(e.Data)) / (3 * n)
	} else {
		if e.NDim != 2 || e.Dims[1] != 3*n {
			return nil, false, fmt.Errorf("convert: %s is %v, want %d rows", e.Name, e.Dims[:e.NDim], 3*n)
		}
		rb, err := tensorRowBytes(e)
		if err != nil {
			return nil, false, err
		}
		unit = rb
	}
	if unit*3*n != uint64(len(e.Data)) {
		return nil, false, fmt.Errorf("convert: %s holds %d bytes, not three parts of %d", e.Name, len(e.Data), n)
	}
	var out []jlm.Tensor
	for i, role := range roles {
		t := jlm.Tensor{Role: role, Block: e.Block, Index: e.Index, Type: e.Type, NDim: e.NDim,
			Data: e.Data[uint64(i)*n*unit : uint64(i+1)*n*unit], Name: e.Name + "/" + role.String()}
		t.Dims = e.Dims
		if e.NDim == 1 {
			t.Dims[0] = n
		} else {
			t.Dims[1] = n
		}
		out = append(out, t)
	}
	return out, true, nil
}

// withoutAudio is an mmproj with its audio encoder's tensors (a.*, mm.a.*)
// set aside, for the vision half's reading; audioOf reads them. A file with
// no audio encoder comes back as it is.
func withoutAudio(f *meta.File) *meta.File {
	if v, ok := f.KV["clip.has_audio_encoder"]; !ok || v.Type != meta.Bool || len(v.Raw) != 1 || v.Raw[0] == 0 {
		return f
	}
	var ts []meta.Tensor
	for _, t := range f.Tensors {
		if !strings.HasPrefix(t.Name, "a.") && !strings.HasPrefix(t.Name, "mm.a.") {
			ts = append(ts, t)
		}
	}
	g := *f
	g.Tensors = ts
	g.Reindex()
	return &g
}
