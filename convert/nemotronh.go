package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// Nemotron-H's source layers are each ONE mixer under its own RMSNorm --
// Mamba-2 (M), attention (*) or a squared-ReLU MLP (-) -- where this engine's
// block is a mixer and an FFN. llama.cpp's nemotron-h.cpp runs them as
// `x = x + mixer(norm(x))` one at a time; a mixer followed by an MLP is
// exactly one block of a mixer and an FFN, the MLP's norm becoming the FFN's.
// So the converter merges each mixer with the MLP that follows it, and a
// mixer followed by another mixer becomes a block with no FFN (nn's NoFFN).
// Every published pattern starts with a mixer and puts every MLP after one;
// an MLP that follows an MLP or opens the stack is refused by name.

// nemotronPlan is the merge: source layer i lands in block dst[i], and ffn[i]
// says it is an MLP layer, whose attn_norm is the block's ffn_norm.
type nemotronPlan struct {
	dst   []int32
	ffn   []bool
	kinds []jlm.LayerKind
}

// nemotronPlanOf reads the source layers' kinds off the tensors -- a layer
// with a convolution is Mamba-2, one with a query projection attends, one
// with an up projection or a router is an MLP -- and merges them.
func nemotronPlanOf(f *meta.File) (*nemotronPlan, error) {
	n := int(f.UintKey("block_count", 0))
	if n == 0 {
		return nil, fmt.Errorf("no block_count")
	}
	has := func(i int, t string) bool {
		_, ok := f.Get("blk." + itoa(i) + "." + t)
		return ok
	}
	p := &nemotronPlan{dst: make([]int32, n), ffn: make([]bool, n)}
	open := false // the last block has no FFN yet
	for i := 0; i < n; i++ {
		switch {
		case has(i, "ssm_conv1d.weight"):
			p.kinds = append(p.kinds, jlm.LayerSSD)
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
			return nil, fmt.Errorf("layer %d is neither Mamba-2, attention nor an MLP", i)
		}
		p.dst[i] = int32(len(p.kinds) - 1)
	}
	return p, nil
}

// remap moves a source tensor to its merged block and role.
func (p *nemotronPlan) remap(role jlm.Role, block int32) (jlm.Role, int32, error) {
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
// merge_mixers statement; Nemotron-H's, see nemotronPlan) and reads the MLP's
// width, which the source states per layer with zero on the mixers.
// Refused: a latent mixture (moe_latent_size, Nemotron 3 Super's: the experts
// at a narrower width between two more matrices) and a prediction block,
// neither built for such a source.
func mergeMixers(f *meta.File, c *Config) error {
	p, err := nemotronPlanOf(f)
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
