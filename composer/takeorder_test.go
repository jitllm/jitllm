package composer

import (
	"slices"
	"testing"
)

// TestTakesReadBeforeTheyWrite holds a loop's takes to the binder's reading
// of them as one assignment: a copy listed first that writes the register a
// later take reads its sheet's term from goes after that take, and two takes
// each writing what the other reads are refused rather than emitted in an
// order that reads a clobbered source.
func TestTakesReadBeforeTheyWrite(t *testing.T) {
	k := &Kernel{
		Bufs:  []Buf{{Reg: 1}, {Reg: 2}},
		Nodes: []Node{{Reg: 3}},
		Ptrs:  []Ptr{{Buf: 0, Reg: 3}, {Buf: 1, Reg: 4}},
		Loops: []Loop{{Cursors: []Cursor{
			{Buf: 0, Ptr: 1, Take: true},
			{Buf: 1, Ptr: 2, Take: true, Init: 1},
		}}},
	}
	if order, ok := k.takeOrder(0); !ok || !slices.Equal(order, []int{1, 0}) {
		t.Fatalf("takes ordered %v (%v), want the sheet's take before the copy into its term's register", order, ok)
	}
	k.Ptrs[1].Reg = 1
	if order, ok := k.takeOrder(0); ok {
		t.Fatalf("two takes each writing the other's source ordered %v", order)
	}
	k.Ptrs[1].Reg, k.Nodes[0].Reg = 4, 5
	if order, ok := k.takeOrder(0); !ok || !slices.Equal(order, []int{0, 1}) {
		t.Fatalf("takes with no shared register ordered %v (%v), want their own order", order, ok)
	}
}
