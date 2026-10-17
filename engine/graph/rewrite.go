package graph

import (
	"fmt"
	"sort"
)

// A Rewrite is an optimization written against the block graph
// (docs/design/composer.md, "The rewrite interface"): where it applies, found
// by matching operations and never by asking for an architecture; why a near
// match is refused; what it promises numerically on each backend; what it
// costs statically; and what makes a verdict about it stale. The recipes it
// offers are built and timed by the tier that runs the block; a rewrite adds
// candidates, not a tuner.
type Rewrite struct {
	Name string
	// Match returns the candidate regions of g: op indices in g.Ops, in
	// order, of the shape the rewrite looks for.
	Match func(g *Graph) []Region
	// Legal refuses a matched region with the reason, or returns nil: the
	// graph's own guards (dataflow, shapes, state).
	Legal func(g *Graph, r Region) error
	// Pays refuses a legal region on a target with the reason, or returns
	// nil: the guards on the device, the executed row width and the layout
	// under which the recipe can win at all. A recipe inert on a target (a
	// tile shape the ISA lacks, a row class it does not change) is refused
	// here, by name, never by an if at its call site. nil offers it on every
	// target the Contract names.
	Pays func(g *Graph, r Region, t Target) error
	// Contract is the numerical promise against the reference recipe, per
	// backend; a backend absent from it is not offered the rewrite.
	Contract map[string]Contract
	// Bounds is the region's static cost change: it may refuse a candidate,
	// never accept one (only a timed complete region accepts).
	Bounds func(g *Graph, r Region) Resources
	// Keys names what a timed verdict depends on beyond the tier's shape key
	// and Target.Key, which every verdict carries.
	Keys func(g *Graph, r Region) string
}

// Target is what a rewrite's guards and its verdicts are keyed by: the
// device's capability, the executed row-width class and the layout facts of
// the rows. A verdict is evidence for its Target.Key only; a block on another
// card, at another row class or in another layout takes that key's verdict or
// the reference recipe. Generated code may differ per device and per model:
// the bar is the gated error margin and the speed, never bit identity across
// devices.
type Target struct {
	API       string // "ptx", "spirv", "msl", "host"
	ISA       string // the code generator's tier: "sm_70", "sm_75", "avx2", "sse", "neon-dotprod"
	MMA       string // the matrix shape the ISA offers ("m8n8k4", "m16n8k16"), "" for none
	Subgroup  int    // threads in a subgroup, warp or SIMD-group
	SharedMem int    // bytes of shared memory a workgroup may hold
	Rows      RowClass
	Layout    Layout
}

// RowClass is the executed row-width class a region runs at: a lever's
// arithmetic changes form across these, so a verdict at one is no evidence
// at another.
type RowClass uint8

const (
	RowsDecode RowClass = iota + 1 // one row per sequence
	RowsFew                        // a ragged step or a speculative verify: fewer routed pairs than the bank has experts
	RowsChunk                      // a prompt chunk: a staged GEMM's width
)

// Layout is how the rows a region reads are laid out.
type Layout struct {
	// Streams is the residual's stream count where it is stream-major
	// (hyper-connections, AltUp): a row slice must take whole streams. 0 or
	// 1 is one stream.
	Streams int
}

// Key is the target's part of a persisted verdict key.
func (t Target) Key() string {
	return fmt.Sprintf("%s/%s/%s/sg%d/sh%d/r%d/s%d", t.API, t.ISA, t.MMA, t.Subgroup, t.SharedMem, t.Rows, t.Layout.Streams)
}

// Offer is a legal region's decision on target t: refused where the backend
// has no contract or the rewrite's Pays guard says it cannot win there.
func (rw *Rewrite) Offer(g *Graph, r Region, t Target) error {
	if err := rw.Legal(g, r); err != nil {
		return err
	}
	if _, ok := rw.Contract[t.API]; !ok {
		return fmt.Errorf("no numerical contract on %s", t.API)
	}
	if rw.Pays != nil {
		return rw.Pays(g, r, t)
	}
	return nil
}

// Region is a matched set of ops.
type Region struct {
	Ops []int
}

// Contract is a rewrite's numerical promise on one backend, against the
// reference recipe on that same backend; across devices only the gated band
// binds.
type Contract struct {
	Exact bool    // bit-identical to the reference recipe on this backend
	Band  float64 // otherwise the largest relative logit difference its gate allows
	Gate  string  // the gate that holds it
}

// Resources is a candidate's static cost change against the reference.
type Resources struct {
	Launches     int   // launches removed (negative: added)
	ScratchBytes int64 // scratch added per row
}

// Decision is one rewrite's verdict on one region, for a plan explanation.
type Decision struct {
	Rewrite string
	Region  Region
	Err     error // nil where legal
}

func (d Decision) String() string {
	if d.Err == nil {
		return fmt.Sprintf("%s at ops %v: legal", d.Rewrite, d.Region.Ops)
	}
	return fmt.Sprintf("%s at ops %v: refused: %v", d.Rewrite, d.Region.Ops, d.Err)
}

// Apply runs every rewrite over g and returns each match's decision, legal
// or refused with its reason.
func Apply(g *Graph, rws ...*Rewrite) []Decision {
	var out []Decision
	for _, rw := range rws {
		for _, r := range rw.Match(g) {
			out = append(out, Decision{Rewrite: rw.Name, Region: r, Err: rw.Legal(g, r)})
		}
	}
	return out
}

// users is, per value, the ops that read it, and producer the op that writes
// it (-1 for a block input).
func users(g *Graph) (use [][]int, producer []int) {
	use = make([][]int, len(g.Values))
	producer = make([]int, len(g.Values))
	for i := range producer {
		producer[i] = -1
	}
	for i, o := range g.Ops {
		for _, v := range o.In {
			use[v] = append(use[v], i)
		}
		for _, v := range o.Out {
			producer[v] = i
		}
	}
	return use, producer
}

// Gate2In is the FFN chain the device runs as one launch (the gate2in
// recipe, jit/gpu/tier/recipe.go): a norm of the residual whose output only
// two matvecs read, their outputs only an act_mul reads, its output only a
// matvec reads, and that matvec's output only the residual add reads. Matched
// by those edges, in any block graph that has them.
var Gate2In = &Rewrite{
	Name: "gate2in",
	Match: func(g *Graph) []Region {
		use, _ := users(g)
		var out []Region
		for i, o := range g.Ops {
			if o.Kind != OpNorm || len(o.Out) != 1 {
				continue
			}
			// The two matvecs reading the norm, whatever else does.
			var mvs []int
			for _, u := range use[o.Out[0]] {
				if g.Ops[u].Kind == OpMatVec {
					mvs = append(mvs, u)
				}
			}
			if len(mvs) != 2 {
				continue
			}
			am := -1
			for _, u := range use[g.Ops[mvs[0]].Out[0]] {
				if g.Ops[u].Kind == OpActMul {
					am = u
				}
			}
			if am < 0 {
				continue
			}
			r := Region{Ops: []int{i, mvs[0], mvs[1], am}}
			for _, u := range use[g.Ops[am].Out[0]] {
				if g.Ops[u].Kind == OpMatVec {
					r.Ops = append(r.Ops, u)
					d := g.Ops[u].Out[0]
					for _, a := range use[d] {
						r.Ops = append(r.Ops, a)
					}
					break
				}
			}
			sort.Ints(r.Ops)
			out = append(out, r)
		}
		return out
	},
	Legal: func(g *Graph, r Region) error {
		use, producer := users(g)
		if len(r.Ops) != 6 {
			return fmt.Errorf("the chain has %d ops, not norm, gate, up, act_mul, down and the residual add", len(r.Ops))
		}
		norm, gate, up, am, down, add := g.Ops[r.Ops[0]], g.Ops[r.Ops[1]], g.Ops[r.Ops[2]], g.Ops[r.Ops[3]], g.Ops[r.Ops[4]], g.Ops[r.Ops[5]]
		switch {
		case norm.Kind != OpNorm:
			return fmt.Errorf("the chain does not start at a norm")
		case len(use[norm.Out[0]]) != 2:
			return fmt.Errorf("the norm's output has %d readers; the recipe recomputes it inside gate and up and writes it nowhere", len(use[norm.Out[0]]))
		case len(gate.Weight) != 1 || len(up.Weight) != 1:
			return fmt.Errorf("gate or up carries a bias the recipe does not add")
		case am.Kind != OpActMul || am.In[0] != gate.Out[0] || am.In[1] != up.Out[0]:
			return fmt.Errorf("the act_mul does not read act(gate)*up")
		case len(use[gate.Out[0]]) != 1 || len(use[up.Out[0]]) != 1:
			return fmt.Errorf("gate or up has a reader besides the act_mul")
		case down.Kind != OpMatVec || len(use[am.Out[0]]) != 1:
			return fmt.Errorf("the product has a reader besides down")
		case add.Kind != OpAdd:
			return fmt.Errorf("down's output reaches the residual through a %s, not the add", add.Kind)
		case len(use[down.Out[0]]) != 1:
			return fmt.Errorf("down's output has %d readers", len(use[down.Out[0]]))
		}
		// The norm reads the residual an add wrote: the recipe takes the row
		// the attention output's launch already added.
		in := norm.In[0]
		if p := producer[in]; p < 0 || g.Ops[p].Kind != OpAdd {
			return fmt.Errorf("the norm's input is not a residual an add wrote")
		}
		if add.In[0] != in && add.In[1] != in {
			return fmt.Errorf("down adds into a row other than the one the norm read")
		}
		return nil
	},
	Contract: map[string]Contract{
		"ptx": {Exact: true, Gate: "backend.TestGate2InIsTheTwoLaunches, model.TestFFNRecipesMatchTwoLaunches"},
	},
	// It removes the FFN norm's launch on a one-row decode; a chunk runs the
	// norm once for every row, so there it removes nothing.
	Pays: func(g *Graph, r Region, t Target) error {
		if t.Rows != RowsDecode {
			return fmt.Errorf("the recipe is a decode matvec; at row class %d the norm is one launch for every row", t.Rows)
		}
		return nil
	},
	Bounds: func(g *Graph, r Region) Resources { return Resources{Launches: 1} },
	Keys:   func(g *Graph, r Region) string { return "" },
}

// Offered is a rewrite matched on one block's graph, carried to the tier
// that runs the block (nn.LayerPlan.Rewrites): the tier knows its device and
// the row class it is about to run, so it builds the Target and asks Offer
// there, for every recipe it considers. The graph and region travel with it;
// a block that relocates takes them to the next device, which asks again
// with its own Target.
type Offered struct {
	Rewrite *Rewrite
	Graph   *Graph
	Region  Region
}

// Offer is the rewrite's decision on this region at target t: Legal, the
// backend's contract, and the Pays guard, each refusal with its reason.
func (o Offered) Offer(t Target) error { return o.Rewrite.Offer(o.Graph, o.Region, t) }

// Key is what a timed verdict on this region at t is persisted under beyond
// the tier's shape key: the target's key and whatever the rewrite's Keys
// names.
func (o Offered) Key(t Target) string {
	k := t.Key()
	if o.Rewrite.Keys != nil {
		if s := o.Rewrite.Keys(o.Graph, o.Region); s != "" {
			k += "/" + s
		}
	}
	return k
}

// Match is every region each rewrite matches in g, legal or not: a tier asks
// each one's Offer at its own target, and a near miss refuses there with the
// reason Legal gives.
func Match(g *Graph, rws ...*Rewrite) []Offered {
	out := []Offered{}
	for _, rw := range rws {
		for _, r := range rw.Match(g) {
			out = append(out, Offered{Rewrite: rw, Graph: g, Region: r})
		}
	}
	return out
}

// Find is the first of os made by the rewrite named name, or false.
func Find(os []Offered, name string) (Offered, bool) {
	for _, o := range os {
		if o.Rewrite.Name == name {
			return o, true
		}
	}
	return Offered{}, false
}
