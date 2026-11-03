package model

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// A principle gate's composed arm runs it again with the composer serving
// every block and the head, on a model the front end states
// whole, and holds the arm to the gate's own bar and to the composer having
// served (composedRan).

// composedParts is what a composed arm asks the composer for.
const composedParts = gpu.ComposedStep

// composedOf is the parts g's composer serves of m: every part where the
// front end states every block of m as ops, and nothing where it does not.
func composedOf(m *Model, g *gpu.GPU) gpu.Parts {
	for li := range m.Cfg.NLayer {
		if m.Uncovered(li) != Covered {
			return 0
		}
	}
	return gpu.ComposedStep
}

// composedNoGraph is the reason the tier declines a block the front end
// states no graph for (gpu.DeclineName).
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
// A block gathering a row table's rows is no card's (cardBlocks) and is not
// counted.
func composedCovers(t *testing.T, m *Model, steps, placed int, exact bool, gs ...*gpu.GPU) {
	t.Helper()
	if len(gs) == 0 {
		return
	}
	covered := 0
	refused := map[string]int{}
	for li := range placed {
		if tableRowsOf(m.Composing().BlockGraph(li)) >= 0 {
			placed--
			continue
		}
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
				declined[gpu.DeclineName(i)] += k
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
		var errs []string
		for _, g := range gs {
			errs = append(errs, g.Err())
		}
		t.Errorf("%d composed part(s) formed at a step: %q", formed, errs)
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
// outside allow (gpu.DeclineName).
func composedRan(t *testing.T, parts gpu.Parts, allow []string, gs ...*gpu.GPU) {
	t.Helper()
	if parts == 0 {
		return
	}
	var served, formed int
	declined := map[string]int{}
	var where []string
	for _, g := range gs {
		s := g.Stats()
		n, _ := s.ServedPart(parts)
		served += n
		formed += s.FormedAtStep
		if s.FormedAtStep > 0 {
			where = append(where, fmt.Sprintf("%v (%s)", s.FormedAtStepParts, g.Err()))
		}
		for i := range s.BlockDeclines {
			if k := s.BlockDeclines[i]; k > 0 {
				declined[gpu.DeclineName(i)] += k
			}
		}
	}
	t.Logf("the composer served %d part steps; declined %v", served, declined)
	if served == 0 {
		t.Fatal("the composer served no part step: the composed arm ran the tier's own kernels")
	}
	if formed != 0 {
		t.Fatalf("%d composed part(s) formed at a step, by part: %s", formed, strings.Join(where, "; "))
	}
	for why, k := range declined {
		if !slices.Contains(allow, why) {
			t.Fatalf("the composer declined %d part step(s): %s", k, why)
		}
	}
}

// composedBlockSteps is the block steps the composer ran on the tiers gs:
// those it emitted and those a recording of them replayed.
func composedBlockSteps(gs ...*gpu.GPU) int {
	n := 0
	for _, g := range gs {
		s := g.Stats()
		b, _ := s.ServedPart(gpu.ComposedBlock)
		n += b + s.ReplayedPart(gpu.ComposedBlock)
	}
	return n
}

// composedBlocksRan fails t unless the composer, asked for parts on the
// tiers gs, ran want block steps past the before already counted
// (composedBlockSteps) -- every one-row step of every block a decode placed
// -- formed none at a step and declined none.
func composedBlocksRan(t *testing.T, parts gpu.Parts, before, want int, gs ...*gpu.GPU) {
	t.Helper()
	if parts&gpu.ComposedBlock == 0 {
		return
	}
	declined, formed := map[string]int{}, 0
	for _, g := range gs {
		s := g.Stats()
		formed += s.FormedAtStep
		for i, k := range s.BlockDeclines {
			if k > 0 {
				declined[gpu.DeclineName(i)] += k
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
