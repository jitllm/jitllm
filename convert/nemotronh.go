package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// A source of one-mixer layers (merge_mixers; Nemotron-H's) is layers that
// are each ONE mixer under its own RMSNorm -- a recurrent mixer, attention or
// an MLP -- where this engine's block is a mixer and an FFN. Such a source
// runs them as `x = x + mixer(norm(x))` one at a time; a mixer followed by an
// MLP is exactly one block of a mixer and an FFN, the MLP's norm becoming the
// FFN's. So the converter merges each mixer with the MLP that follows it,
// and a mixer followed by another mixer becomes a block with no FFN (nn's
// NoFFN). An MLP that follows an MLP or opens the stack is refused.

// mixerPlan is the merge: source layer i lands in block dst[i], and ffn[i]
// says it is an MLP layer, whose attn_norm is the block's ffn_norm.
type mixerPlan struct {
	dst   []int32
	ffn   []bool
	kinds []jlm.LayerKind
}

// mixerPlanOf reads the source layers' kinds off the tensors -- a layer
// carrying the tensors of a kind the statement's layers entries name is a
// mixer of that kind, one with a query projection attends, one with an up
// projection or a router is an MLP -- and merges them.
func mixerPlanOf(f *meta.File) (*mixerPlan, error) {
	n := int(f.UintKey("block_count", 0))
	if n == 0 {
		return nil, fmt.Errorf("no block_count")
	}
	st, _, err := ggufStates(f.Arch())
	if err != nil {
		return nil, err
	}
	has := func(i int, t string) bool {
		_, ok := f.Get("blk." + itoa(i) + "." + t)
		return ok
	}
	stated := func(i int) (jlm.LayerKind, bool) {
		var k jlm.LayerKind
		found := false
		for _, l := range st.Layers {
			carries := true
			for _, t := range l.Tensors {
				carries = carries && has(i, t)
			}
			if carries {
				k, found = mjLayerKinds()[l.Kind], true
			}
		}
		return k, found
	}
	p := &mixerPlan{dst: make([]int32, n), ffn: make([]bool, n)}
	open := false // the last block has no FFN yet
	for i := 0; i < n; i++ {
		k, mixer := stated(i)
		switch {
		case mixer:
			p.kinds = append(p.kinds, k)
			open = true
		case has(i, "attn_q.weight"):
			p.kinds = append(p.kinds, jlm.LayerFullAttn)
			open = true
		case has(i, "ffn_up.weight") || has(i, "ffn_gate_inp.weight"):
			if !open {
				return nil, fmt.Errorf("layer %d is an MLP that follows no mixer: %w", i, ErrNotImplemented)
			}
			p.ffn[i], open = true, false
			p.dst[i] = int32(len(p.kinds) - 1)
			continue
		default:
			return nil, fmt.Errorf("layer %d is neither a stated mixer, attention nor an MLP", i)
		}
		p.dst[i] = int32(len(p.kinds) - 1)
	}
	return p, nil
}

// remap moves a source tensor to its merged block and role.
func (p *mixerPlan) remap(role jlm.Role, block int32) (jlm.Role, int32, error) {
	if block < 0 {
		return role, block, nil
	}
	if int(block) >= len(p.dst) {
		return 0, 0, fmt.Errorf("block %d of a %d-layer plan", block, len(p.dst))
	}
	if p.ffn[block] && role == jlm.RoleAttnNorm {
		role = jlm.RoleFFNNorm
	}
	return role, p.dst[block], nil
}

// mergeMixers merges a source of one-mixer layers into blocks (the
// merge_mixers statement; Nemotron-H's, see mixerPlan) and reads the MLP's
// width, which the source states per layer with zero on the mixers.
// Refused: a latent mixture (moe_latent_size, Nemotron 3 Super's: the experts
// at a narrower width between two more matrices) and a prediction block,
// neither built for such a source.
func mergeMixers(f *meta.File, c *Config) error {
	p, err := mixerPlanOf(f)
	if err != nil {
		return err
	}
	c.NLayer = uint32(len(p.kinds))
	c.LayerKinds = p.kinds
	ff, err := uniformCount(f, "feed_forward_length", 0)
	if err != nil {
		return err
	}
	c.NFFN = ff
	if f.UintKey("moe_latent_size", 0) > 0 {
		return fmt.Errorf("a latent mixture (moe_latent_size) over merged mixers: %w", ErrNotImplemented)
	}
	if f.UintKey("nextn_predict_layers", 0) > 0 {
		return fmt.Errorf("a prediction block over merged mixers: %w", ErrNotImplemented)
	}
	return nil
}

// statedMixtureFits refuses a mixture whose expert width or top-k is stated
// per layer (NVIDIA's Puzzle models), which the container states as one, and
// reads a router the library states ungrouped (router.ungrouped): one
// routing group, which every published Nemotron 3 config states, is no
// grouping; transformers' NemotronHTopkRouter groups and llama.cpp's builder
// never does, so more is a divergence (RULE 7m) nobody has chosen, refused
// until one is.
func statedMixtureFits(f *meta.File, c *Config) error {
	if c.NExpert == 0 {
		return nil
	}
	for _, k := range []string{"expert_feed_forward_length", "expert_used_count"} {
		if kv, ok := f.Key(k); ok {
			if _, ok := kv.Uint(); !ok {
				return fmt.Errorf("a per-layer %s: %w", k, ErrNotImplemented)
			}
		}
	}
	if c.Stated.routerUngrouped {
		if c.NExpertGroup > 1 {
			return fmt.Errorf("%d expert groups, and the class's router never groups: %w", c.NExpertGroup, ErrNotImplemented)
		}
		c.NExpertGroup, c.NExpertGroupUsed = 0, 0
	}
	return nil
}
