package composer

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestInPlaceGivesAnOutputOnlyTheCallersStorage: a graph's output takes an
// input's storage only where the caller holds it. A sum that zeroes its
// output in one loop and adds into it in the next reads that output, but
// only what it wrote: the value is made, not the caller's. A map adding it
// to the caller's x stores the graph's output over x, never over the sum's
// storage, which no caller binds. Falcon-H1's residual, its attention and
// mixer outputs added to x in one map, put the block's output in the
// attention projection's scratch on the CPU, and every token read another
// text.
func TestInPlaceGivesAnOutputOnlyTheCallersStorage(t *testing.T) {
	F, U := ir.F32, ir.U32
	x := Operand{Kind: OperandValue, Value: 0, Name: "x", Elem: F}
	o := Operand{Kind: OperandValue, Value: 1, Name: "o", Elem: F}
	y := Operand{Kind: OperandValue, Value: 2, Name: "y", Elem: F, Out: true}
	sh := []int{4, 2}
	sum := &Kernel{Name: "sum", Kind: PrimReduce, Space: []int{4},
		Loops: []Loop{{Trip: 2, Parent: -1}, {Trip: 2, Parent: -1}},
		Bufs:  []Buf{{Kind: BufOut, Elem: F, Operand: o, Shape: sh}},
		Nodes: []Node{
			{Op: OpCoord, Type: U, A: 0},
			{Op: OpConst, Type: F},
			{Op: OpIter, Type: U, At: 0},
			{Op: OpIter, Type: U, At: 1},
			{Op: OpLoad, Type: F, Buf: 0, Ax: []int{0, 3}},
			{Op: OpAdd, Type: F, A: 4, B: 4},
		},
		Stores: []Store{{Buf: 0, Ax: []int{0, 2}, Val: 1, At: 0}, {Buf: 0, Ax: []int{0, 3}, Val: 5, At: 1}},
	}
	add := &Kernel{Name: "add", Kind: PrimMap, Space: sh,
		Bufs: []Buf{{Kind: BufIn, Elem: F, Operand: o, Shape: sh}, {Kind: BufIn, Elem: F, Operand: x, Shape: sh},
			{Kind: BufOut, Elem: F, Operand: y, Shape: sh}},
		Nodes: []Node{
			{Op: OpCoord, Type: U, A: 0},
			{Op: OpCoord, Type: U, A: 1},
			{Op: OpLoad, Type: F, Buf: 0, Ax: []int{0, 1}},
			{Op: OpLoad, Type: F, Buf: 1, Ax: []int{0, 1}},
			{Op: OpAdd, Type: F, A: 2, B: 3},
		},
		Stores: []Store{{Buf: 2, Ax: []int{0, 1}, Val: 4, At: -1}},
	}
	ks, err := InPlace{}.Apply([]*Kernel{sum, add}, Device{})
	if err != nil {
		t.Fatal(err)
	}
	k := ks[1]
	if got := k.Bufs[k.Stores[0].Buf].Operand; got != x {
		t.Fatalf("the graph's output is stored over %q, want the caller's x", got.Name)
	}
}
