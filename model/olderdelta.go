package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// olderDelta is the delta rule's decay and gate for a container an older
// writer made, which carried them as fields and tensors rather than
// statements: the decay under its lower bound (KDALowerBound) where it has
// one, the gate a sigmoid where the file says so (Flag2DeltaSigmoidGate) or a
// block stores a per-channel decay (as convert.StateDelta states a source's),
// silu otherwise. A model with no delta rule has neither.
func olderDelta(c *jlm.Config, stores func(jlm.Role) bool) (decay, gate *jlm.Program, err error) {
	if !slices.Contains(c.LayerKinds, jlm.LayerLinearAttn) {
		return nil, nil, nil
	}
	if !(c.KDALowerBound <= 0) {
		return nil, nil, fmt.Errorf("model: a decay bound of %g; a bound is negative", c.KDALowerBound)
	}
	decay, gate = jlm.DeltaDecaySoftplus(), jlm.DeltaGateSiLU()
	if c.KDALowerBound != 0 {
		decay = jlm.DeltaDecayBounded(c.KDALowerBound)
	}
	if c.Flags2.Has(jlm.Flag2DeltaSigmoidGate) || stores(jlm.RoleSSMFB) {
		gate = jlm.DeltaGateSigmoid()
	}
	return decay, gate, nil
}
