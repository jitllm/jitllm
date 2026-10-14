package convert

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// clefOf is Clef's head as llama.cpp's converter states it (conversion/clef.py
// from joint_head_config.json): routing blocks, then joint blocks, the heads
// they attend with, and the head's LayerNorm epsilon -- torch's default, which
// the reference never sets and the converter writes as layer_norm_epsilon. The
// backbone is qwen35 under another name (the arch "clef").
func clefOf(f *meta.File, c *jlm.Config) error {
	if c.Arch != jlm.ArchQwen3Next || f.Arch() != "clef" {
		return fmt.Errorf("a clef readout is the clef arch's qwen35 backbone, and this is %s (%v)", f.Arch(), c.Arch)
	}
	c.DecisionRouting = uint32(f.UintKey("decision.routing_block_count", 0))
	c.DecisionHeads = uint32(f.UintKey("decision.head_count", 0))
	if c.DecisionBlocks == 0 || c.DecisionHeads == 0 {
		return fmt.Errorf("a clef head with %d joint blocks and %d heads", c.DecisionBlocks, c.DecisionHeads)
	}
	if c.DecisionRouting > 64 || c.DecisionBlocks > 64 {
		return fmt.Errorf("a clef head of %d routing and %d joint blocks", c.DecisionRouting, c.DecisionBlocks)
	}
	eps, ok := f.Key("attention.layer_norm_epsilon")
	if !ok {
		return fmt.Errorf("a clef head needs attention.layer_norm_epsilon, its LayerNorms' epsilon")
	}
	e, _ := eps.Float()
	if !(e > 0) {
		return fmt.Errorf("a clef head's LayerNorm epsilon is %v", e)
	}
	c.DecisionEps = float32(e)
	return nil
}

// Clef's head tensors are named as their roles are (format/jlm/decisionroles.go),
// with ".weight" on all but the biases and the scalars.
func init() {
	for r := jlm.RoleDecHiddenNorm; r <= jlm.RoleDecFFNDownBias; r++ {
		if !r.Valid() {
			continue
		}
		name := r.String()
		if !strings.HasSuffix(name, ".bias") && r != jlm.RoleDecScales {
			name += ".weight"
		}
		roleOf[name] = r
	}
}
