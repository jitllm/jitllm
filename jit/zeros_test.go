package jit

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/cpu"
)

// zerosReadAcrossLoops is each read, in kernel k's sequence, of a float zero
// made inside a loop the read is not inside. A zero made in a vector loop's
// body is not made when the row is narrower than a vector and that loop makes
// no trip, and the tail loop, or code past it, compares against whatever the
// register held.
func zerosReadAcrossLoops(k *Kernel) []string {
	var open []int
	madeIn := map[int]int{}
	var bad []string
	check := func(where string, args []*int) {
		for _, a := range args {
			if ml, made := madeIn[*a]; made && k.isZero(*a) && ml >= 0 && !slices.Contains(open, ml) {
				bad = append(bad, fmt.Sprintf("%s reads the zero made in loop %d, which is not open", where, ml))
			}
		}
	}
	for p, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			open = append(open, e.I)
		case EntryEnd:
			open = open[:len(open)-1]
		case EntryNode:
			nd := k.Nodes[e.I]
			check(fmt.Sprintf("entry %d (node %d, %v)", p, e.I, nd.Op), k.Operands(&nd))
			if k.isZero(e.I) {
				ml := -1
				if len(open) > 0 {
					ml = open[len(open)-1]
				}
				madeIn[e.I] = ml
			}
		case EntryStore:
			s := k.Stores[e.I]
			check(fmt.Sprintf("entry %d (store %d)", p, e.I), k.storeArgs(&s))
		}
	}
	return bad
}

// withOneZero is k with every float zero it makes read through the first
// one, the others no longer made: the kernel before a zero was made in each
// loop that reads it. ok is false for a kernel that makes one zero.
func withOneZero(k *Kernel) (m *Kernel, ok bool) {
	m = k.clone()
	var zeros []int
	for _, e := range m.Seq {
		if e.Kind == EntryNode && m.isZero(e.I) {
			zeros = append(zeros, e.I)
		}
	}
	if len(zeros) < 2 {
		return nil, false
	}
	for _, z := range zeros[1:] {
		for i := range m.Nodes {
			for _, a := range m.Operands(&m.Nodes[i]) {
				if *a == z {
					*a = zeros[0]
				}
			}
		}
		for i := range m.Stores {
			for _, a := range m.storeArgs(&m.Stores[i]) {
				if *a == z {
					*a = zeros[0]
				}
			}
		}
		m.Seq = slices.DeleteFunc(m.Seq, func(e Entry) bool { return e.Kind == EntryNode && e.I == z })
	}
	return m, true
}

// TestAZeroIsMadeInEveryLoopThatReadsIt holds the CPU kernels of a compare
// against zero and of a maximum with it, over rows that are a tail alone, a
// vector and a tail, and several vectors and a tail, to making the zero in
// each loop that reads it. The same kernels with their zeros merged into the
// first are the violation, and must be seen.
func TestAZeroIsMadeInEveryLoopThatReadsIt(t *testing.T) {
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	r := Default().WithoutInt8(DeviceCPU)
	prof := r.Profiles[DeviceCPU]
	prof.Off = map[string]bool{"inplace": true}
	graphs := map[string]func(n int) *graph.Graph{
		"where": func(n int) *graph.Graph {
			b := graph.New()
			x := b.In("x", n)
			mp := func(a graph.Attrs, in ...graph.ValueID) graph.ValueID {
				return b.Emit(graph.OpMap, in, []graph.Result{{Name: "v", Width: n}}, nil, a, nil, nil)[0]
			}
			b.Out(mp(graph.Attrs{Fn: graph.MapWhere}, x, mp(graph.Attrs{Fn: graph.MapMul, Const: 2}, x), mp(graph.Attrs{Fn: graph.MapMul, Const: 3}, x)))
			return b.G
		},
		"max": func(n int) *graph.Graph {
			b := graph.New()
			x := b.In("x", n)
			b.Out(b.Emit(graph.OpMap, []graph.ValueID{x}, []graph.Result{{Name: "v", Width: n}}, nil, graph.Attrs{Fn: graph.MapMax}, nil, nil)[0])
			return b.G
		},
	}
	seen := 0
	for name, mk := range graphs {
		for _, n := range []int{5, 7, 9, 23} {
			g := mk(n)
			c := r.context(g, PartBlock, nil, nil, Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, dev)
			prims, err := r.decomposeOps(c, 0)
			if err != nil {
				t.Fatal(err)
			}
			var ks []*Kernel
			for _, p := range prims {
				k, err := generate(p)
				if err != nil {
					t.Fatal(err)
				}
				ks = append(ks, k)
			}
			if ks, err = prof.run(ks, dev); err != nil {
				t.Fatal(err)
			}
			for _, k := range ks {
				if bad := zerosReadAcrossLoops(k); len(bad) != 0 {
					t.Errorf("%s over %d: kernel %s: %v", name, n, k.Name, bad)
				}
				if m, ok := withOneZero(k); ok {
					seen++
					if len(zerosReadAcrossLoops(m)) == 0 {
						t.Errorf("%s over %d: kernel %s with its zeros merged is not seen to read one across loops", name, n, k.Name)
					}
				}
			}
		}
	}
	if seen == 0 {
		t.Fatal("no kernel made a zero in more than one loop: this gate proved nothing")
	}
	t.Logf("%d kernels made a zero in each loop that reads it; merged into one each is seen", seen)
}
