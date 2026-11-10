package convert

import (
	"errors"

	"github.com/jitllm/jitllm/format/jlm"
)

// LegacyStated is what a container written before v31 must state that only
// its architecture's code says (jlm.Stating): ModernBERT's erf GELU as its
// activations, Phi-3.5-MoE's routed weight for its router.
func LegacyStated(c *jlm.Config) error {
	switch {
	case c.Arch == jlm.ArchModernBERT && len(c.Acts) == 0:
		return errors.New("states no activations")
	case c.Arch == jlm.ArchPhiMoE && c.NExpert != 0 && len(c.RouterWeight.Terms) == 0:
		return errors.New("states no routed weight for its router")
	}
	return nil
}
