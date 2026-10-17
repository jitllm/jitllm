package graph

import "fmt"

// The operand-prologue rewrites (docs/design/composer.md, "How the
// candidates plug in"): a matvec that builds its int8 operand in threadgroup
// memory from the float row instead of reading one a launch before it wrote
// (kernels.ActIn). Three sites, each found by its edges: the norm whose only
// readers are the q, k and v matvecs (the norm runs inside them), the
// attention output's matvec (the quantize runs inside it), and down reading
// act_mul's product (the quantize runs inside it). The tier times each
// matched site against its two launches (tuneActIn); these decide where it
// may be offered at all.

// actInContract is the prologue's promise: the bits of the two launches it
// replaces, on every backend its kernel gate runs.
var actInContract = map[string]Contract{
	"ptx":   {Exact: true, Gate: "backend.TestActInIsTheTwoLaunches"},
	"spirv": {Exact: true, Gate: "backend.TestActInIsTheTwoLaunches"},
	"msl":   {Exact: true, Gate: "backend.TestActInIsTheTwoLaunches"},
}

// actInPays is the prologue's target guard: it removes a launch only on a
// one-row decode (a chunk quantizes its operand once for every row), and its
// window maximum and sums meet by 32-lane subgroup shuffles.
func actInPays(_ *Graph, _ Region, t Target) error {
	switch {
	case t.Rows != RowsDecode:
		return fmt.Errorf("the prologue is a decode matvec's; at row class %d the operand is built once for every row", t.Rows)
	case t.Subgroup != 32:
		return fmt.Errorf("the prologue meets its window by 32-lane shuffles; this device's subgroup is %d", t.Subgroup)
	}
	return nil
}

// ActInQKV is a norm whose output only the q, k and v matvecs read: the norm
// runs as the prologue of each (or of their one segment launch).
var ActInQKV = &Rewrite{
	Name: "actin-qkv",
	Match: func(g *Graph) []Region {
		use, _ := users(g)
		var out []Region
		for i, o := range g.Ops {
			if o.Kind != OpNorm || len(o.Out) != 1 {
				continue
			}
			r := Region{Ops: []int{i}}
			for _, u := range use[o.Out[0]] {
				if g.Ops[u].Kind == OpMatVec {
					r.Ops = append(r.Ops, u)
				}
			}
			if len(r.Ops) == 4 {
				out = append(out, r)
			}
		}
		return out
	},
	Legal: func(g *Graph, r Region) error {
		use, _ := users(g)
		norm := g.Ops[r.Ops[0]]
		// A q/k/v bias is the kernel's (MatVecShape.Bias): the region binds it.
		if n := len(use[norm.Out[0]]); n != 3 {
			return fmt.Errorf("the norm's output has %d readers; the prologue writes it nowhere", n)
		}
		return nil
	},
	Contract: actInContract,
	Pays:     actInPays,
	Bounds:   func(*Graph, Region) Resources { return Resources{Launches: 1} },
}

// singleReader is the region {producer of m's input, m} where m is a matvec
// reading the output of an op of kind from, which only m reads, and whose
// output only an add reads.
func singleReader(g *Graph, from OpKind) []Region {
	_, producer := users(g)
	var out []Region
	for i, o := range g.Ops {
		if o.Kind != OpMatVec || len(o.In) != 1 {
			continue
		}
		p := producer[o.In[0]]
		if p < 0 || g.Ops[p].Kind != from {
			continue
		}
		out = append(out, Region{Ops: []int{p, i}})
	}
	return out
}

// singleLegal checks a singleReader region: the matvec is its input's only
// reader, carries no bias, and adds into the residual.
func singleLegal(g *Graph, r Region) error {
	use, _ := users(g)
	src, m := g.Ops[r.Ops[0]], g.Ops[r.Ops[1]]
	switch {
	case len(use[src.Out[0]]) != 1:
		return fmt.Errorf("the %s output has %d readers; the prologue writes the operand nowhere", src.Kind, len(use[src.Out[0]]))
	case len(m.Weight) != 1:
		return fmt.Errorf("the matvec carries a bias, which the residual epilogue's slot holds")
	case len(use[m.Out[0]]) != 1 || g.Ops[use[m.Out[0]][0]].Kind != OpAdd:
		return fmt.Errorf("the matvec's output does not reach the residual through one add")
	}
	return nil
}

// ActInO is the attention output's matvec quantizing the attention row
// inside its launch.
var ActInO = &Rewrite{
	Name:     "actin-o",
	Match:    func(g *Graph) []Region { return singleReader(g, OpAttend) },
	Legal:    singleLegal,
	Contract: actInContract,
	Pays:     actInPays,
	Bounds:   func(*Graph, Region) Resources { return Resources{Launches: 1} },
}

// ActInDown is down quantizing act_mul's product inside its launch.
var ActInDown = &Rewrite{
	Name:     "actin-down",
	Match:    func(g *Graph) []Region { return singleReader(g, OpActMul) },
	Legal:    singleLegal,
	Contract: actInContract,
	Pays:     actInPays,
	Bounds:   func(*Graph, Region) Resources { return Resources{Launches: 1} },
}

// Rewrites is every rewrite a block graph is matched against, in the order a
// plan explanation lists them.
var Rewrites = []*Rewrite{Gate2In, ActInQKV, ActInO, ActInDown}
