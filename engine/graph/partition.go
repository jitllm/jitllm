package graph

import "fmt"

// PartitionSlice is the hidden channels one slice of the MLP partition
// owns: one workgroup's share of the gate and up rows, whose partial of down
// it writes. 256 is the slice the dependency-block analysis priced
// (docs/design/dep-block.md, "k-slice chaining": 32 slices on a 8192-wide
// intermediate).
const PartitionSlice = 256

// MLPPartition is the hidden-channel MLP partition (docs/design/
// external-optimization-ideas.md, "Joint fusion and partitioning"): gate and
// up's rows cut into slices of PartitionSlice channels, each slice running
// its gate and up rows, the activation and its columns of down, and the
// slices' partial outputs summed. It is matched by the FFN's edges wherever
// they occur and refused by its static bound wherever the partials it writes
// outweigh the intermediate it removes: every graph whose output row is
// wider than PartitionSlice (Bounds). It removes no launch (one fused slice
// launch and a reduction where Gate2 and down were two), so only traffic
// could pay for it, at decode and at a chunk's rows alike (both sides scale
// with the rows). No recipe is built for it and no backend has a contract: a
// graph where Bounds passes is where one would be built.
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
	// The partials are H f32 per slice, written and read once; the
	// intermediate they replace is F f32, written and read once. Per row:
	// 8*F removed, 8*H*ceil(F/PartitionSlice) added, so the bound passes only
	// where H < PartitionSlice: narrower than every residual in the model list.
	Bounds: func(g *Graph, r Region) Resources {
		f := int64(g.Values[g.Ops[r.Ops[2]].Out[0]].Width)
		h := int64(g.Values[g.Ops[r.Ops[3]].Out[0]].Width)
		s := (f + PartitionSlice - 1) / PartitionSlice
		return Resources{Launches: 0, TrafficBytes: 8*f - 8*h*s}
	},
	Keys: func(g *Graph, r Region) string { return "" },
}
