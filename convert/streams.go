package convert

import (
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// storeRowTables stores as row tables the model-level tensors of s that c's
// statements gather a row at a time (jlm.Streams.RowsGathered).
func storeRowTables(c *jlm.Config, s *jlm.Source) {
	rs := c.Streams.RowsGathered()
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && slices.Contains(rs, t.Role) {
			t.RowTable = true
		}
	}
}

// sourceStores says s holds role for block li (-1 the dense tensors).
func sourceStores(s *jlm.Source) func(jlm.Role, int32) bool {
	held := map[[2]int32]bool{}
	for _, t := range s.Tensors {
		held[[2]int32{int32(t.Role), t.Block}] = true
	}
	return func(r jlm.Role, li int32) bool { return held[[2]int32{int32(r), int32(s.Config.PageOf(int(li)))}] }
}

// StateDelta states the delta rule's decay and output gate where the
// architecture's config left them unstated: the decay through softplus, and
// the gate a sigmoid where the decay is a key channel's (the block stores
// the low-rank decay pair, Kimi Delta Attention's), silu elsewhere.
// stores says block li (-1 the dense tensors) stores role.
func StateDelta(c *jlm.Config, stores func(r jlm.Role, li int32) bool) {
	linear, perChannel := false, false
	for li, k := range c.LayerKinds {
		if k == jlm.LayerLinearAttn {
			linear = true
			perChannel = perChannel || stores(jlm.RoleSSMFB, int32(li))
		}
	}
	if !linear {
		return
	}
	if c.DeltaDecay == nil {
		c.DeltaDecay = jlm.DeltaDecaySoftplus()
	}
	if c.DeltaGate == nil {
		c.DeltaGate = jlm.DeltaGateSiLU()
		if perChannel {
			c.DeltaGate = jlm.DeltaGateSigmoid()
		}
	}
}
