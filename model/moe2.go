package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// The modern text group's engine-side facts (format/jlm/archmoe2.go).

// sparseMixerOf is the router jitter an architecture's code carries: Phi-3.5-MoE
// routes by sparsemixer, and nothing else here does.
func sparseMixerOf(a jlm.Arch) float32 {
	if a == jlm.ArchPhiMoE {
		return jlm.PhiMoEJitter
	}
	return 0
}

// checkSparseMixer refuses a sparsemixer model whose mixture is not the plain
// top-2 sparsemixer runs over: the router's kernels take the raw logits, keep
// two and renormalise nothing, so a config claiming more would be run as
// something it does not say.
func (c *Config) checkSparseMixer() error {
	if c.SparseMixer == 0 {
		return nil
	}
	switch {
	case c.NExpert < 2 || c.NExpertUsed != 2:
		return fmt.Errorf("model: %s: sparsemixer selects 2 experts; this mixture selects %d of %d",
			c.Arch, c.NExpertUsed, c.NExpert)
	case c.ExpertSigmoid || c.NExpertGroup > 1 || !c.NoExpertNorm || (c.ExpertScale != 0 && c.ExpertScale != 1):
		return fmt.Errorf("model: %s: sparsemixer with a sigmoid %v, %d groups, renormalisation %v "+
			"or a routed scale %g", c.Arch, c.ExpertSigmoid, c.NExpertGroup, !c.NoExpertNorm, c.ExpertScale)
	}
	return nil
}
