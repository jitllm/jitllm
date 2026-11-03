package graph

// Drift is the comparison the shadow arm runs on the device after a block: how
// far one arm's output is from the other's, relative to what the block added
// to its input, kept as the worst over the steps compared beside the count of
// them. The squared distance of got from want over the squared distance of
// want from the block's input; a ratio that is not a finite number below
// DriftCap counts as DriftCap, so a form that produced a NaN or an infinity
// cannot hide behind a maximum that ignores one.
//
// Its inputs are the block's input, the two arms' outputs and the state: two
// words, the worst so far and the steps. Its output is the state again, and
// the caller binds it over the state's input.
type Drift struct {
	NEmbd int
}

const (
	// DriftCap is the largest drift the comparison reports.
	DriftCap = 1e30
	// driftFloor keeps a block that added nothing from dividing by zero.
	driftFloor = 1e-30
)

func (d *Drift) Build() *Graph {
	b := New()
	n := d.NEmbd
	in, want, got, state := b.In("in", n), b.In("want", n), b.In("got", n), b.In("state", 2)
	mapn := func(name string, w int, fn MapFn, k float64, ins ...ValueID) ValueID {
		return b.Emit(OpMap, ins, []Result{{Name: name, Width: w}}, nil, Attrs{Fn: fn, Const: k}, nil, nil)[0]
	}
	sum := func(name string, v ValueID) ValueID {
		return b.Emit(OpReduce, []ValueID{v}, []Result{{Name: name, Width: 1}}, nil, Attrs{Fn: MapAdd}, nil, nil)[0]
	}
	word := func(name string, off int) ValueID {
		return b.Emit(OpSlice, []ValueID{state}, []Result{{Name: name, Width: 1}}, nil, Attrs{Offset: off, Width: 1}, nil, nil)[0]
	}
	miss := mapn("miss", n, MapSub, 0, want, got)
	added := mapn("added", n, MapSub, 0, want, in)
	ratio := mapn("ratio", 1, MapDiv, 0,
		sum("miss.sq", mapn("miss.mul", n, MapMul, 0, miss, miss)),
		mapn("added.floor", 1, MapAdd, driftFloor, sum("added.sq", mapn("added.mul", n, MapMul, 0, added, added))))
	finite := mapn("finite", 1, MapLess, DriftCap, ratio)
	capped := mapn("capped", 1, MapWhere, DriftCap, finite, ratio)
	worst := mapn("worst", 1, MapMax, 0, word("worst.was", 0), capped)
	steps := mapn("steps", 1, MapAdd, 1, word("steps.was", 1))
	b.Out(b.Emit(OpConcat, []ValueID{worst, steps}, []Result{{Name: "state.next", Width: 2}}, nil, Attrs{}, nil, nil)[0])
	return b.G
}
