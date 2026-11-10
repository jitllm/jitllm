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

// nemotronConfig is Nemotron-H's graph: the merged blocks, no rotary on the
// attention, and the MLP's squared ReLU with no gate.
//
// Nemotron 3 (nemotron_h_moe) replaces the MLP with a mixture of the same
// ungated experts -- up and down, no gate bank -- under DeepSeek-V3's router
// (NemotronHTopkRouter: sigmoid scores, a selection-only bias, renormalised,
// routed_scaling_factor) beside one ungated shared expert. The class
// hardcodes the sigmoid and llama.cpp's builder passes it as a literal; the
// GGUF carries no gating key, so the converter sets it.
func nemotronConfig(f *meta.File, c *Config) error {
	p, err := nemotronPlanOf(f)
	if err != nil {
		return err
	}
	c.NLayer = uint32(len(p.kinds))
	c.LayerKinds = p.kinds
	c.Flags |= jlm.FlagNoPosEnc | jlm.FlagReLU2
	// The MLP width is per source layer, zero on the mixers.
	ff, err := attnLayerCount(f, "feed_forward_length", 0)
	if err != nil {
		return err
	}
	c.NFFN = ff
	// The latent projection (Nemotron 3 Super) runs the experts at a narrower
	// width between two more matrices; not built.
	if f.UintKey("moe_latent_size", 0) > 0 {
		return fmt.Errorf("Nemotron-H's latent mixture (moe_latent_size): %w", ErrNotImplemented)
	}
	if f.UintKey("nextn_predict_layers", 0) > 0 {
		return fmt.Errorf("Nemotron-H's prediction block: %w", ErrNotImplemented)
	}
	return nil
}

// nemotronMixture is the mixture half of nemotronConfig, run once configOf
// has read the file's generic mixture keys.
func nemotronMixture(f *meta.File, c *Config) error {
	if c.NExpert > 0 {
		if kv, ok := f.Key("expert_gating_func"); ok {
			if v, _ := kv.Uint(); v != 2 {
				return fmt.Errorf("expert_gating_func %d, and NemotronHTopkRouter is sigmoid: %w",
					v, ErrNotImplemented)
			}
		}
		c.Flags |= jlm.FlagExpertSigmoid
		// One routing group, which every published config states, is no
		// grouping. transformers' router groups and llama.cpp's builder never
		// does, so a file with more is a divergence (RULE 7m) nobody has
		// chosen; it is refused until one is.
		if c.NExpertGroup > 1 {
			return fmt.Errorf("Nemotron-H with %d expert groups: %w", c.NExpertGroup, ErrNotImplemented)
		}
		c.NExpertGroup, c.NExpertGroupUsed = 0, 0
		// A per-layer expert width or top-k (NVIDIA's Puzzle models) is one
		// value here or refused.
		for _, k := range []string{"expert_feed_forward_length", "expert_used_count"} {
			if kv, ok := f.Key(k); ok {
				if _, ok := kv.Uint(); !ok {
					return fmt.Errorf("Nemotron-H with a per-layer %s: %w", k, ErrNotImplemented)
				}
			}
		}
	}
	return nil
}
