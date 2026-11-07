package model

import (
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAnOlderDeltaRuleIsStatedFromTheFile holds olderDelta to what an older
// container holds: the defaults, its decay bound, its sigmoid bit and a
// per-channel decay each state their decay and gate, a model with no delta
// rule states none, and a positive bound is refused.
func TestAnOlderDeltaRuleIsStatedFromTheFile(t *testing.T) {
	base := func() *jlm.Config {
		return &jlm.Config{NLayer: 2, LayerKinds: []jlm.LayerKind{jlm.LayerLinearAttn, jlm.LayerFullAttn}}
	}
	none := func(jlm.Role) bool { return false }
	perChannel := func(r jlm.Role) bool { return r == jlm.RoleSSMFB }
	for _, tc := range []struct {
		name        string
		edit        func(c *jlm.Config)
		stores      func(jlm.Role) bool
		decay, gate *jlm.Program
	}{
		{"the defaults", func(*jlm.Config) {}, none, jlm.DeltaDecaySoftplus(), jlm.DeltaGateSiLU()},
		{"a bound", func(c *jlm.Config) { c.KDALowerBound = -5 }, none, jlm.DeltaDecayBounded(-5), jlm.DeltaGateSiLU()},
		{"a sigmoid gate", func(c *jlm.Config) { c.Flags2 |= jlm.Flag2DeltaSigmoidGate }, none, jlm.DeltaDecaySoftplus(), jlm.DeltaGateSigmoid()},
		{"a per-channel decay", func(*jlm.Config) {}, perChannel, jlm.DeltaDecaySoftplus(), jlm.DeltaGateSigmoid()},
	} {
		c := base()
		tc.edit(c)
		decay, gate, err := olderDelta(c, tc.stores)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !reflect.DeepEqual(decay, tc.decay) || !reflect.DeepEqual(gate, tc.gate) {
			t.Errorf("%s: stated decay %+v and gate %+v", tc.name, decay, gate)
		}
	}
	plain := base()
	plain.LayerKinds = nil
	if decay, gate, err := olderDelta(plain, perChannel); err != nil || decay != nil || gate != nil {
		t.Errorf("a model with no delta rule stated %+v, %+v (%v)", decay, gate, err)
	}
	bad := base()
	bad.KDALowerBound = 2
	if _, _, err := olderDelta(bad, none); err == nil || !strings.Contains(err.Error(), "a bound is negative") {
		t.Errorf("a positive bound: %v", err)
	}
}
