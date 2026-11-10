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
func clefOf(f *meta.File, c *Config) error {
	if c.Arch != ArchQwen3Next || f.Arch() != "clef" {
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
	// joint_schema_model.py: QUESTION_TYPES {noul: 0, choice: 1, score: 2},
	// render's sort_keys, question_options' sorted choices and its noul
	// read true then false; systemone_answer's confidence the largest
	// probability.
	c.JointHead = &jlm.JointHead{TypeRows: [3]uint8{1, 2, 0}, SortKeys: true, SortChoices: true, NoulTrueFirst: true,
		IDInstructions: true, Confidence: jlm.ConfidencePMax}
	return nil
}

// StatesMediaPrefix reports whether a decision kind's container states its
// media prefix (jlm.Flag2MediaPrefix).
func StatesMediaPrefix(kind jlm.DecisionKind) bool { return kind == jlm.DecisionD1Omni }

// StatesJointHead reports whether a decision kind's container states a joint
// head (jlm.JointHead).
func StatesJointHead(kind jlm.DecisionKind) bool { return kind == jlm.DecisionClef }

// Clef's head tensors are named as their roles are (format/jlm/decisionroles.go),
// with ".weight" on all but the biases and the scalars.
func clefRoleOf(name string) (jlm.Role, bool) {
	for r := jlm.RoleDecHiddenNorm; r <= jlm.RoleDecFFNDownBias; r++ {
		if !r.Valid() {
			continue
		}
		n := r.String()
		if strings.HasSuffix(n, ".bias") || r == jlm.RoleDecScales {
			if name == n {
				return r, true
			}
			continue
		}
		if len(name) == len(n)+len(".weight") && strings.HasPrefix(name, n) && strings.HasSuffix(name, ".weight") {
			return r, true
		}
	}
	return 0, false
}
