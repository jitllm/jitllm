package model

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// A principle gate's composed arm runs it again with the composer serving
// every block and the head (Config.Composed), on a model the front end states
// whole, and holds the arm to the gate's own bar and to the composer having
// served (composedRan).

// composedParts is what a composed arm asks the composer for.
const composedParts = tier.ComposedBlock | tier.ComposedHead

// composedArms are the arms a gate runs the model at path in: the tier's own
// kernels, and the composer's where the front end states every block and the
// head. A model it does not state gets no composed arm, rather than one that
// passes on the tier's kernels.
func composedArms(t *testing.T, path string) []tier.Parts {
	t.Helper()
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Uncovered(-1) != Covered {
		return []tier.Parts{0}
	}
	for li := range m.Cfg.NLayer {
		if m.Uncovered(li) != Covered {
			return []tier.Parts{0}
		}
	}
	return []tier.Parts{0, composedParts}
}

// composedArm is the subtest name of name's arm: name itself for the tier's
// own kernels, so a gate's existing names select what they always did.
func composedArm(name string, parts tier.Parts) string {
	if parts == 0 {
		return name
	}
	return name + "/composed"
}

// composedOpt is the tier option that asks the composer for parts.
func composedOpt(parts tier.Parts) tier.Option {
	return tier.WithConfig(func(c *tier.Config) { c.Composed = parts })
}

// composedRan fails t unless the composer, asked for parts on the tiers gs,
// served part steps, formed none at a step and declined none for a reason
// outside allow (tier.DeclineName).
func composedRan(t *testing.T, parts tier.Parts, allow []string, gs ...*tier.GPU) {
	t.Helper()
	if parts == 0 {
		return
	}
	var served, formed int
	declined := map[string]int{}
	for _, g := range gs {
		s := g.Stats()
		n, _, _ := s.ServedPart(parts)
		served += n
		formed += s.FormedAtStep
		for i := range s.BlockDeclines {
			if k := s.BlockDeclines[i] + s.HeadDeclines[i]; k > 0 {
				declined[tier.DeclineName(i)] += k
			}
		}
	}
	t.Logf("the composer served %d part steps; declined %v", served, declined)
	if served == 0 {
		t.Fatal("the composer served no part step: the composed arm ran the tier's own kernels")
	}
	if formed != 0 {
		t.Fatalf("%d composed part(s) formed at a step", formed)
	}
	for why, k := range declined {
		if !slices.Contains(allow, why) {
			t.Fatalf("the composer declined %d part step(s): %s", k, why)
		}
	}
}
