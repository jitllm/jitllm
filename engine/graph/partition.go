package graph

import (
	"errors"
	"fmt"
)

// PartitionSlice is the hidden channels one slice of the MLP partition
// owns: one workgroup's share of the gate and up rows, whose partial of down
// it writes. 256 is the slice the dependency-block analysis priced
// (docs/design/dep-block.md, "k-slice chaining": 32 slices on a 8192-wide
// intermediate).
const PartitionSlice = 256

// MLPPartition is the hidden-channel MLP partition (docs/design/
// external-optimization-ideas.md, "Joint fusion and partitioning"): gate and
// up's rows cut by channel, each part running its gate and up rows, the
// activation and its columns of down, the parts' partial outputs summed. It
// is matched by the FFN's edges wherever they occur.
//
// On one card it cannot pay: it removes no launch and its partials (8*H per
// slice of PartitionSlice channels) outweigh the intermediate it removes
// (8*F) wherever H > PartitionSlice, every residual in the model list. Across
// two cards it is a capacity lever, not a speed lever: each card holds half
// of gate, up and down, and each row crosses twice (the normed row out, the
// partial sum back). So it is a run-time alternative (WhenRun): never taken
// by the speed policy, listed with its cost, and chosen only by a caller
// whose objective is capacity, on a Target that can execute it (PeerCards,
// a split executor reaching a second card). No executor is built yet, so no
// Target offers it.
var MLPPartition = &Rewrite{
	Name: "mlp-partition",
	Match: func(g *Graph) []Region {
		use, producer := users(g)
		var out []Region
		for i, o := range g.Ops {
			if o.Kind != OpActMul || len(o.In) != 2 || len(o.Out) != 1 {
				continue
			}
			gt, up := producer[o.In[0]], producer[o.In[1]]
			if gt < 0 || up < 0 || g.Ops[gt].Kind != OpMatVec || g.Ops[up].Kind != OpMatVec {
				continue
			}
			for _, u := range use[o.Out[0]] {
				if g.Ops[u].Kind == OpMatVec {
					out = append(out, Region{Ops: []int{gt, up, i, u}})
					break
				}
			}
		}
		return out
	},
	Legal: func(g *Graph, r Region) error {
		use, _ := users(g)
		if len(r.Ops) != 4 {
			return fmt.Errorf("the region has %d ops, not gate, up, act_mul and down", len(r.Ops))
		}
		gate, up, am, down := g.Ops[r.Ops[0]], g.Ops[r.Ops[1]], g.Ops[r.Ops[2]], g.Ops[r.Ops[3]]
		switch {
		case gate.In[0] != up.In[0]:
			return fmt.Errorf("gate and up read different rows; a slice reads one completed input")
		case len(gate.Weight) != 1 || len(up.Weight) != 1 || len(down.Weight) != 1:
			return fmt.Errorf("a bias is placed once, after the slices' sum; the region carries one inside a slice")
		case len(use[gate.Out[0]]) != 1 || len(use[up.Out[0]]) != 1 || len(use[am.Out[0]]) != 1:
			return fmt.Errorf("an intermediate has a reader outside the slice")
		}
		return nil
	},
	// The cross-card form: half of gate, up and down's elements off each
	// card, two crossings of an H-wide f32 row (out and back) a row, and the
	// two copies as launches added.
	Bounds: func(g *Graph, r Region) Resources {
		f := int64(g.Values[g.Ops[r.Ops[2]].Out[0]].Width)
		h := int64(g.Values[g.Ops[r.Ops[3]].Out[0]].Width)
		return Resources{Launches: -2, TrafficBytes: -8 * h, WeightElems: 3 * h * f / 2, Crossings: 2}
	},
	Pays: func(_ *Graph, _ Region, t Target) error {
		if t.PeerCards < 1 {
			return errNoSplit
		}
		return nil
	},
	When:      WhenRun,
	Objective: "capacity",
	Chooser:   "a caller (tier.WithRewrite(\"mlp-partition\", graph.ChoiceTake))",
	Evidence: "backend.TestCopyPeerCrossingLatency: a crossing pair over NVLink (cuMemcpyPeer, synchronous) 56-63 us " +
		"against ~18 us of weight read saved per Llama-3.2-1B block; executable only where Target.PeerCards > 0, " +
		"which no tier states until a split executor exists",
	Keys: func(g *Graph, r Region) string { return "" },
}

// errNoSplit is MLPPartition's refusal on a Target with no split executor.
var errNoSplit = errors.New("no split executor reaches a second card from this target")
