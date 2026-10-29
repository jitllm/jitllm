package jit

import (
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
)

// TestRemovingALoopMovesItsInnerLoopsToItsParent removes a loop of one trip
// listed before its parent, as a strip-mined tail's copies are: its inner
// loop and its store move to that parent, numbered as it is after the
// removal.
func TestRemovingALoopMovesItsInnerLoopsToItsParent(t *testing.T) {
	k := &Kernel{Name: "loops", Space: []int{1}, Loops: []Loop{
		{Trip: 4, Parent: -1},
		{Trip: 1, Parent: 2},
		{Trip: 8, Parent: 0},
		{Trip: 2, Parent: 1},
	}}
	v := k.add(Node{Op: OpConst, Type: ir.F32, Imm: 1})
	k.Bufs = append(k.Bufs, Buf{Kind: BufOut, Elem: ir.F32, Operand: Operand{Kind: OperandScratch, Name: "out"}, Shape: []int{1}})
	k.Stores = append(k.Stores, Store{Buf: 0, Val: v, At: 1, Ax: []int{k.u32(0)}})
	if !k.dropUnitLoops() {
		t.Fatal("the loop of one trip was not removed")
	}
	want := []int{-1, 0, 1}
	for i, l := range k.Loops {
		if l.Parent != want[i] {
			t.Errorf("loop %d's parent is %d, want %d", i, l.Parent, want[i])
		}
	}
	if k.Stores[0].At != 1 {
		t.Errorf("the removed loop's store is at loop %d, want its parent, 1", k.Stores[0].At)
	}
}
