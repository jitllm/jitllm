package composer

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// TestPartsJoinNoCombineAcrossWidths holds Parts to one width for a part
// whose threads combine or shuffle: two plain maps of 1536 and 4608
// elements join, and the narrower one with a workgroup combine or a lane
// shuffle in it must not -- clamped to its last element, the wider launch's
// surplus threads would combine among themselves and store another value
// there than its own threads do (Mamba-1's int8 forms beside its copies).
func TestPartsJoinNoCombineAcrossWidths(t *testing.T) {
	r := Default().WithoutInt8(DeviceGPU)
	b := graph.New()
	mapn := func(name string, v graph.ValueID, w int) graph.ValueID {
		return b.Emit(graph.OpMap, []graph.ValueID{v}, []graph.Result{{Name: name, Width: w}}, nil, graph.Attrs{Fn: graph.MapMul, Const: 2}, nil, nil)[0]
	}
	b.Out(mapn("narrow", b.In("a", 1536), 1536))
	b.Out(mapn("wide", b.In("b", 4608), 4608))
	dev := Device{Kind: DeviceGPU, Lanes: 32}
	c := r.context(b.G, 0, nil, nil, Step{Rows: 1, Page: 64, Keys: 1}, dev)
	prims, err := r.decomposeOps(c, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(prims) != 2 {
		t.Fatalf("%d primitives, want the two maps", len(prims))
	}
	var ks [2]*Kernel
	for i, p := range prims {
		if ks[i], err = generate(p); err != nil {
			t.Fatal(err)
		}
	}
	p := Parts{Kinds: []PrimKind{PrimMap}}
	if !p.joins(ks[:1], ks[1]) {
		t.Fatal("two plain maps of different widths do not join: the gate selects nothing")
	}
	for _, op := range []Op{OpGroupSum, OpGroupMax, OpShuffle, OpExchange} {
		n := ks[0].clone()
		n.Nodes = append(n.Nodes, Node{Op: op, Type: ir.F32, A: 0, Imm: 1})
		if p.joins([]*Kernel{n}, ks[1]) || p.joins([]*Kernel{ks[1]}, n) {
			t.Errorf("a part of 1536 elements with op %d joins one of 4608", op)
		}
	}
}
