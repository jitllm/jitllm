package model

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
)

// rewriteBlocks states every block of m the front end states to the composer
// as rewrite makes of its graph, until the returned undo: a violation that
// reaches the composer as a different graph and that nothing the tier runs of
// its own can see. A rewrite that does not verify is the gate's fault, not a
// violation, and fails t.
func (m *Model) rewriteBlocks(t *testing.T, rewrite func(*graph.Graph) *graph.Graph) func() {
	t.Helper()
	s := m.frontGraphs()
	set := *s
	set.blocks = slices.Clone(s.blocks)
	rewritten := 0
	for li, g := range set.blocks {
		if g == nil {
			continue
		}
		set.blocks[li] = rewrite(g)
		f, _ := m.inventory(int32(li))
		if err := graph.Verify(set.blocks[li], f); err != nil {
			t.Fatalf("the rewrite of block %d does not verify: %v", li, err)
		}
		rewritten++
	}
	if rewritten == 0 {
		t.Fatal("the front end states no block: there is nothing to rewrite")
	}
	m.front.mu.Lock()
	was := m.front.set
	m.front.set = &set
	m.front.mu.Unlock()
	return func() {
		m.front.mu.Lock()
		m.front.set = was
		m.front.mu.Unlock()
	}
}

// rewriteOps is g with f applied to a copy of each op, and fails t where f
// changed none: a violation that reached nothing.
func rewriteOps(t *testing.T, g *graph.Graph, f func(i int, o *graph.Op) bool) *graph.Graph {
	t.Helper()
	out := *g
	out.Ops = slices.Clone(g.Ops)
	hit := 0
	for i := range out.Ops {
		if f(i, &out.Ops[i]) {
			hit++
		}
	}
	if hit == 0 {
		t.Fatalf("block %d: the rewrite found nothing to change", g.Block)
	}
	return &out
}

// withoutTheGate is g with the attention's output left as it is: the sigmoid
// of the gate and the product with it are a copy of the output.
func withoutTheGate(t *testing.T) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
			if o.Kind != graph.OpActMul || o.Attr.Act != graph.ActSigmoid {
				return false
			}
			*o = graph.Op{Kind: graph.OpMap, In: []graph.ValueID{o.In[1]}, Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 1}}
			return true
		})
	}
}

// withTheHalvesSwapped is g with each head's query taken from the half the
// gate is in, and the gate from the half the query is in.
func withTheHalvesSwapped(t *testing.T) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		var halves []int
		for i, o := range g.Ops {
			if o.Kind == graph.OpSlice && o.Attr.HeadDim > 0 {
				halves = append(halves, i)
			}
		}
		if len(halves) != 2 {
			t.Fatalf("block %d: %d per-head slices, want the query's and the gate's", g.Block, len(halves))
		}
		return rewriteOps(t, g, func(i int, o *graph.Op) bool {
			switch i {
			case halves[0]:
				o.Attr.Offset = g.Ops[halves[1]].Attr.Offset
			case halves[1]:
				o.Attr.Offset = g.Ops[halves[0]].Attr.Offset
			default:
				return false
			}
			return true
		})
	}
}

// withTheHalvesWhole is g with the projection read as its first half the
// query and its second the gate, each head's query and gate no longer
// together: the layout of a model that stores the two as separate halves.
func withTheHalvesWhole(t *testing.T) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
			if o.Kind != graph.OpSlice || o.Attr.HeadDim == 0 {
				return false
			}
			width := g.Values[o.Out[0]].Width
			o.Attr = graph.Attrs{Offset: o.Attr.Offset / o.Attr.Width * width, Width: width}
			return true
		})
	}
}

// TestHybridOutGateViolationsOnDevice holds the device gate on the attention
// output gate to seeing it taken apart. The qwen3next-shaped hybrid's
// attention block runs on the device with the composer serving it, once as the
// front end states it and once with the graph the composer is given
// rewritten: the gate left out, the halves of each head's double-width query
// swapped, or the halves read whole rather than per head. The tier's own
// kernels never read that graph, so what moves is the composed block alone,
// and the gate's own verdict (hybridDeviceVerdict: past the block bound
// (hybridDeviceBound) with the shallow control past it too) must be a failure
// on each. Every run is counted: the composer served the block it was handed
// (composedCovers).
func TestHybridOutGateViolationsOnDevice(t *testing.T) {
	for _, geo := range []struct {
		name string
		opt  hyOpt
	}{{"dense", hyOpt{}}, {"bigdense", hyOpt{big: true}}} {
		for _, spec := range stepDevices() {
			t.Run(geo.name+"/"+spec, func(t *testing.T) {
				clean := hybridDeviceVerdict(t, geo.opt, spec, hyDev{noGraph: true, composed: true})
				t.Logf("as stated: logit NMSE %.3e (bound %.0e), failed %v", clean.worst, hybridDeviceBound, clean.failed)
				if clean.failed {
					t.Fatalf("the block as the front end states it fails the gate")
				}
				for _, v := range []struct {
					name    string
					rewrite func(*graph.Graph) *graph.Graph
				}{
					{"the gate left out", withoutTheGate(t)},
					{"the halves swapped", withTheHalvesSwapped(t)},
					{"the halves read whole", withTheHalvesWhole(t)},
				} {
					bad := hybridDeviceVerdict(t, geo.opt, spec, hyDev{noGraph: true, composed: true, rewrite: v.rewrite})
					t.Logf("%s: logit NMSE %.3e at pos %d (%.0fx the bound %.0e), the shallow control %.3e (%.0fx): the gate fails %v",
						v.name, bad.worst, bad.at, bad.worst/hybridDeviceBound, hybridDeviceBound, bad.ctl, bad.ctl/hybridDeviceBound, bad.failed)
					if !bad.failed {
						t.Errorf("%s: the gate passes it (NMSE %.3e, the shallow control %.3e, bound %.0e): it cannot see the gate on the attention's output", v.name, bad.worst, bad.ctl, hybridDeviceBound)
					}
				}
			})
		}
	}
}
