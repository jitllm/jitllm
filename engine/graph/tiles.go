package graph

import "errors"

// errDeepWidth is DeepTile's refusal below a prompt chunk's width class. A
// value, not a formatted error: the guard is asked as a launch is built,
// inside a prompt's warm path, which allocates nothing.
var errDeepWidth = errors.New("the deep tiles' verdict was timed at a prompt chunk's width; " +
	"a narrower launch (a verification, a prompt's tail) was not timed and takes the default blocking")

// DeepTile is a prompt GEMM's deep m16n8 binary16 tiles (eight warps, 64 of k
// a trip) in place of the shipping blocking, for any matvec of a block: the
// tier times the two per shape at the reserved prompt width (tuneGemm), and
// the verdict is evidence for that width class only. The guard that was a
// comparison at the launch site (ntok*2 >= promptW) is this rewrite's Pays.
var DeepTile = &Rewrite{
	Name: "deep-tile",
	Match: func(g *Graph) []Region {
		var out []Region
		for i, o := range g.Ops {
			if o.Kind == OpMatVec {
				out = append(out, Region{Ops: []int{i}})
			}
		}
		return out
	},
	Legal: func(*Graph, Region) error { return nil },
	// The deep tiles reduce k in another order than the shipping ones; the
	// staged GEMM's own gate holds both to the reference at NMSE 1e-5 (and
	// within twice dp4a's), per output, not per logit.
	Contract: map[string]Contract{
		"ptx": {Band: 1e-5, Gate: "backend.TestGemmF16DeepStageMatchesTheReference (NMSE per GEMM output)"},
	},
	Pays: func(_ *Graph, _ Region, t Target) error {
		if t.Rows != RowsChunk {
			return errDeepWidth
		}
		return nil
	},
	Bounds:  func(*Graph, Region) Resources { return Resources{} },
	When:    WhenLoad,
	Chooser: "the tier's GEMM trial (tuneGemm)",
}
