package tier

import (
	"strings"
	"testing"
)

// TestValuesAreColouredByLiveness holds colourVals to its contract: values
// whose lives meet take different slots, an op's input and output never
// share one (RULE 13), values whose lives do not meet share the lowest, a
// slot's need is its widest value, more values live at once than there are
// slots is refused, and so is a value read and never written.
func TestValuesAreColouredByLiveness(t *testing.T) {
	p := &devProgram{}
	f := func(w int) devType { return devType{Elem: elemF32, Layout: layoutRow, Width: w} }
	x := p.val("x", f(8), slotX)
	a := p.val("a", f(4), slotAuto)
	b := p.val("b", f(16), slotAuto)
	c := p.val("c", f(2), slotAuto)
	y := p.val("y", f(8), slotOut)
	p.ops = []devOp{
		{Kind: dkMul, In: []devVal{x, x}, Out: []devVal{a}}, // a lives 0..1
		{Kind: dkMul, In: []devVal{a, a}, Out: []devVal{b}}, // b lives 1..2: meets a at 1
		{Kind: dkMul, In: []devVal{b, b}, Out: []devVal{c}}, // c lives 2..3: meets b at 2, not a
		{Kind: dkMul, In: []devVal{c, x}, Out: []devVal{y}},
	}
	need, err := colourVals(p)
	if err != nil {
		t.Fatal(err)
	}
	if p.slots[a] == p.slots[b] || p.slots[b] == p.slots[c] {
		t.Fatalf("values whose lives meet share a slot: a %s, b %s, c %s", p.slots[a], p.slots[b], p.slots[c])
	}
	if p.slots[a] != slotVal0 || p.slots[c] != slotVal0 || p.slots[b] != slotVal1 {
		t.Fatalf("a %s, b %s, c %s: the lowest free slot is not taken", p.slots[a], p.slots[b], p.slots[c])
	}
	if need[0] != 4 || need[1] != 16 {
		t.Fatalf("needs %v, want 4 for val0 (a and c) and 16 for val1 (b)", need[:2])
	}
	if p.slots[x] != slotX || p.slots[y] != slotOut {
		t.Fatalf("a value with a slot of its own was recoloured")
	}

	// More live at once than there are slots.
	q := &devProgram{}
	src := q.val("x", f(1), slotX)
	var live []devVal
	for i := 0; i <= numVals; i++ {
		v := q.val("v", f(1), slotAuto)
		q.ops = append(q.ops, devOp{Kind: dkCopy, In: []devVal{src}, Out: []devVal{v}})
		live = append(live, v)
	}
	q.ops = append(q.ops, devOp{Kind: dkAdd, In: live, Out: []devVal{q.val("sum", f(1), slotOut)}})
	if _, err := colourVals(q); err == nil || !strings.Contains(err.Error(), "live") {
		t.Fatalf("%d values live at once were coloured: %v", numVals+1, err)
	}

	// A value read and never written.
	r := &devProgram{}
	ghost := r.val("ghost", f(1), slotAuto)
	r.ops = []devOp{{Kind: dkCopy, In: []devVal{ghost}, Out: []devVal{r.val("y", f(1), slotOut)}}}
	if _, err := colourVals(r); err == nil {
		t.Fatalf("a value never written was coloured")
	}
}
