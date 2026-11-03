package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// runCPUPrim forms map p alone on the CPU device and runs its calls over
// bufs, g naming its values.
func runCPUPrim(t *testing.T, p Prim, g *graph.Graph, bufs map[string][]uint32) {
	t.Helper()
	k, err := generate(p)
	if err != nil {
		t.Fatal(err)
	}
	k.Apart = true
	dev := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	reg := Default()
	ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{k}, dev)
	if err != nil {
		t.Fatal(err)
	}
	pcs, err := reg.Lower(ks, dev)
	if err != nil {
		t.Fatal(err)
	}
	var codes []Code
	for _, c := range pcs {
		codes = append(codes, c.Code)
	}
	if err := runCPUCalls(codes, g, bufs); err != nil {
		t.Fatal(err)
	}
}

// indexValues is a row map's graph: a word a row, a float a row, and its
// output word a row.
func indexValues() *graph.Graph {
	return &graph.Graph{Values: []graph.Value{{Name: "w", Width: 1, Elem: graph.ElemU32}, {Name: "f", Width: 1}, {Name: "o", Width: 1, Elem: graph.ElemU32}}}
}

// TestCPUVectorAddressReadsItsCoordinate executes a row map whose one mask
// picks both a float's bits and the index the float is read at: the pick
// makes the mask a vector word, and with it the index's select, which reads
// the row's coordinate. Each row is held to its own word or its own float's
// bits, so an index computed from another call's coordinate fails it.
func TestCPUVectorAddressReadsItsCoordinate(t *testing.T) {
	const R, cut = 6, 4
	U, F := ir.U32, ir.F32
	var x exprs
	r := x.coord(0)
	w := x.at(0, []int{R, 1}, r, x.u(0))
	m := x.op(OpLess, ir.Pred, w, x.u(cut))
	s := x.un(OpBitcast, U, x.load(1, x.sel(U, m, r, x.u(0))))
	v := x.sel(U, m, s, w)
	out := Operand{Kind: OperandValue, Value: 2, Name: "o", Elem: U, Out: true}
	p := Prim{Kind: PrimMap, Shape: []int{R, 1}, Expr: x.e, Facts: Facts{Extents: map[int]int{1: R}},
		In:     []Operand{{Kind: OperandValue, Value: 0, Name: "w", Elem: U}, {Kind: OperandValue, Value: 1, Name: "f", Elem: F}},
		Stores: []PrimStore{{Out: out, Ax: []int{r, x.u(0)}, Ext: []int{R, 1}, Value: v}}}
	words := []uint32{1, 9, 2, 7, 3, 8}
	fs := make([]float32, R)
	for i := range fs {
		fs[i] = 0.25 + float32(i)
	}
	got := make([]uint32, R)
	runCPUPrim(t, p, indexValues(), map[string][]uint32{"w": words, "f": f32Words(fs), "o": got})
	picked := 0
	for i, wi := range words {
		want := wi
		if wi < cut {
			want = math.Float32bits(fs[i])
			picked++
		}
		if got[i] != want {
			t.Fatalf("row %d is %#x, want %#x (rows %#x)", i, got[i], want, got)
		}
	}
	if picked == 0 || picked == R {
		t.Fatal("the rows take one arm of the pick: the gate cannot see the index's")
	}
}

// TestCPUIndexKeepsTheConstantItSubtractsFrom executes a row map reading a
// float at 1 - min(word, 1), an index never below zero whose constant a
// buffer's offset could take, leaving -min(word, 1) to the 32-bit index.
// Each row is held to the float its word picks.
func TestCPUIndexKeepsTheConstantItSubtractsFrom(t *testing.T) {
	const R = 6
	U, F := ir.U32, ir.F32
	var x exprs
	r := x.coord(0)
	w := x.at(0, []int{R, 1}, r, x.u(0))
	ix := x.op(OpSub, U, x.u(1), x.op(OpMin, U, w, x.u(1)))
	v := x.un(OpBitcast, U, x.load(1, ix))
	out := Operand{Kind: OperandValue, Value: 2, Name: "o", Elem: U, Out: true}
	p := Prim{Kind: PrimMap, Shape: []int{R, 1}, Expr: x.e, Facts: Facts{Extents: map[int]int{1: 2}},
		In:     []Operand{{Kind: OperandValue, Value: 0, Name: "w", Elem: U}, {Kind: OperandValue, Value: 1, Name: "f", Elem: F}},
		Stores: []PrimStore{{Out: out, Ax: []int{r, x.u(0)}, Ext: []int{R, 1}, Value: v}}}
	words := []uint32{0, 1, 5, 0, 2, 0}
	fs := []float32{0.5, 1.5}
	got := make([]uint32, R)
	runCPUPrim(t, p, indexValues(), map[string][]uint32{"w": words, "f": f32Words(fs), "o": got})
	for i, wi := range words {
		want := math.Float32bits(fs[0])
		if wi == 0 {
			want = math.Float32bits(fs[1])
		}
		if got[i] != want {
			t.Fatalf("row %d is %#x, want %#x (rows %#x)", i, got[i], want, got)
		}
	}
}
