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

// composedOf is the parts g's composer serves of m: what g asks it for
// (Config.Composed, every part in a jitllmcomposed build), where the front
// end states every block of m as ops, and nothing where it does not.
func composedOf(m *Model, g *tier.GPU) tier.Parts {
	for li := range m.Cfg.NLayer {
		if m.Uncovered(li) != Covered {
			return 0
		}
	}
	return g.Composed
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

// composedLater are the declines the composer's paths not yet asked of it
// count, by name (tier.DeclineName): a head over several rows -- a prompt
// chunk's folded head, a ragged step's, speculative rows' -- a prompt's
// one-row head, and, where the prompt is asked, a ragged step's blocks and a
// prompt outside the shadow arm. They ran the tier's kernels before they
// were named.
var composedLater = []string{"a prompt chunk's folded head", "a head over several rows", "a ragged step's rows",
	"every row's logits (speculative rows)", "a prompt's one-row head", "a prompt outside the shadow arm"}

// composedNoGraph is the reason the tier declines a block the front end
// states no graph for (tier.DeclineName).
const composedNoGraph = "no graph from the front end"

// composedCovers is a composed tier's check of one fixture's decode, by the
// counter composedBlocksRan uses (composedBlockSteps: the steps the composer
// emitted and those a recording replayed). Of steps one-row steps over blocks
// [0, placed) of m on the fresh tiers gs it ran every block the front end
// states, formed none at a step, and declined no block step but for the front
// end having no graph: exactly as many of those as the blocks it states none
// of take where the run issues every step itself (exact: no recording, which
// replays nothing a decline is counted from), at most that where it does not.
// A sliding layer is served like any other, the tier declining only a graph
// whose window is not the plan's, so a window decline fails here by its name.
// A tier that is not asked for blocks is not checked.
func composedCovers(t *testing.T, m *Model, steps, placed int, exact bool, gs ...*tier.GPU) {
	t.Helper()
	if len(gs) == 0 || gs[0].Composed&tier.ComposedBlock == 0 {
		return
	}
	covered := 0
	refused := map[string]int{}
	for li := range placed {
		if why := m.Uncovered(li); why != Covered {
			refused[why.String()]++
			continue
		}
		covered++
	}
	formed := 0
	declined := map[string]int{}
	for _, g := range gs {
		s := g.Stats()
		formed += s.FormedAtStep
		for i, k := range s.BlockDeclines {
			if k > 0 {
				declined[tier.DeclineName(i)] += k
			}
		}
	}
	for why, k := range declined {
		if why != composedNoGraph {
			t.Errorf("the composer declined %d block step(s): %s", k, why)
		}
	}
	ran := composedBlockSteps(gs...)
	t.Logf("the front end states %d of %d blocks (refused: %v): the composer ran %d block steps in %d steps, declined %d for no graph",
		covered, placed, refused, ran, steps, declined[composedNoGraph])
	if formed != 0 {
		t.Errorf("%d composed part(s) formed at a step", formed)
	}
	if want := steps * covered; ran != want {
		t.Errorf("the composer ran %d block steps, want %d: %d steps of the %d blocks it states", ran, want, steps, covered)
	}
	got, want := declined[composedNoGraph], steps*(placed-covered)
	if exact && got != want || got > want {
		t.Errorf("%d block steps declined for no graph, want %d (at most, with a recording): %d steps of the %d blocks it states none of", got, want, steps, placed-covered)
	}
}

// composedRan fails t unless the composer, asked for parts on the tiers gs,
// served part steps, formed none at a step and declined none for a reason
// outside allow and composedLater (tier.DeclineName).
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
		if !slices.Contains(allow, why) && !slices.Contains(composedLater, why) {
			t.Fatalf("the composer declined %d part step(s): %s", k, why)
		}
	}
}

// composedBlockSteps is the block steps the composer ran on the tiers gs:
// those it emitted and those a recording of them replayed.
func composedBlockSteps(gs ...*tier.GPU) int {
	n := 0
	for _, g := range gs {
		s := g.Stats()
		b, _, _ := s.ServedPart(tier.ComposedBlock)
		n += b + s.ReplayedPart(tier.ComposedBlock)
	}
	return n
}

// composedBlocksRan fails t unless the composer, asked for parts on the
// tiers gs, ran want block steps past the before already counted
// (composedBlockSteps) -- every one-row step of every block a decode placed
// -- formed none at a step and declined none.
func composedBlocksRan(t *testing.T, parts tier.Parts, before, want int, gs ...*tier.GPU) {
	t.Helper()
	if parts&tier.ComposedBlock == 0 {
		return
	}
	declined, formed := map[string]int{}, 0
	for _, g := range gs {
		s := g.Stats()
		formed += s.FormedAtStep
		for i, k := range s.BlockDeclines {
			if k > 0 {
				declined[tier.DeclineName(i)] += k
			}
		}
	}
	if formed != 0 || len(declined) != 0 {
		t.Fatalf("the composer formed %d block(s) at a step and declined block steps %v", formed, declined)
	}
	if n := composedBlockSteps(gs...) - before; n != want {
		t.Fatalf("the composer ran %d block steps, where the decode's one-row steps of its placed blocks are %d", n, want)
	}
	t.Logf("the composer ran every one of the %d block steps", want)
}
