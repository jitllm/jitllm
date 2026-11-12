package composer

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// TestTakesReadBeforeTheyWrite holds a loop's takes to the binder's reading
// of them as one assignment: a copy listed first that writes the register a
// later take reads its sheet's term from goes after that take, two takes
// each writing what the other reads park one in a spill slot the binding
// left free until the other has read its register, and every slot taken is
// the one case it fails. Random loops of takes over a few registers then run
// the steps against the assignment read in one go; the takes in list order,
// the violation, part from it.
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
	want := func(name string, got []takeStep, ok bool, steps ...takeStep) {
		t.Helper()
		if !ok || !slices.Equal(got, steps) {
			t.Fatalf("%s: takes in steps %v (%v), want %v", name, got, ok, steps)
		}
	}
	got, ok := k.takeOrder(0)
	want("a copy into a term's register", got, ok, takeStep{I: 1}, takeStep{I: 0})
	k.Ptrs[1].Reg = 1
	got, ok = k.takeOrder(0)
	want("two takes each writing the other's source", got, ok,
		takeStep{I: 0, Slot: 1, To: takePark}, takeStep{I: 1}, takeStep{I: 0, Slot: 1, To: takeUnpark})
	k.Ptrs = append(k.Ptrs, Ptr{Buf: 1, Spill: spillSlots})
	if got, ok := k.takeOrder(0); ok {
		t.Fatalf("a cycle with every spill slot taken ordered %v", got)
	}
	k.Ptrs = k.Ptrs[:2]
	k.Ptrs[1].Reg, k.Nodes[0].Reg = 4, 5
	got, ok = k.takeOrder(0)
	want("takes with no shared register", got, ok, takeStep{I: 0}, takeStep{I: 1})

	r := rand.New(rand.NewPCG(1, 2))
	cycles, parted := 0, 0
	for range 2000 {
		k := randomTakes(r)
		steps, ok := k.takeOrder(0)
		if !ok {
			t.Fatalf("takes %+v not ordered", k.Loops[0].Cursors)
		}
		wantRegs := runTakes(k, nil)
		if got := runTakes(k, steps); !slices.Equal(got, wantRegs) {
			t.Fatalf("takes %+v in steps %v leave %v, read in one go %v", k.Loops[0].Cursors, steps, got, wantRegs)
		}
		if slices.ContainsFunc(steps, func(s takeStep) bool { return s.To == takePark }) {
			cycles++
		}
		var list []takeStep
		for i := range k.Loops[0].Cursors {
			list = append(list, takeStep{I: i})
		}
		if !slices.Equal(runTakes(k, list), wantRegs) {
			parted++
		}
	}
	t.Logf("2000 random loops of takes: %d parked a take, %d part from the assignment in list order", cycles, parted)
	if cycles == 0 || parted == 0 {
		t.Fatalf("the random loops held %d cycles and %d list orders that part: the gate saw neither case", cycles, parted)
	}
}

// randomTakes is a loop of two to five takes over six registers, each a
// copy of its buffer's register into a register of its own, half of them
// adding a sheet's term.
func randomTakes(r *rand.Rand) *Kernel {
	n := 2 + r.IntN(4)
	k := &Kernel{Loops: []Loop{{}}}
	dests := r.Perm(6)
	for i := range n {
		k.Bufs = append(k.Bufs, Buf{Reg: 1 + r.IntN(6)})
		k.Ptrs = append(k.Ptrs, Ptr{Buf: i, Reg: 1 + dests[i]})
		cu := Cursor{Buf: i, Ptr: i + 1, Take: true}
		if r.IntN(2) == 0 {
			k.Nodes = append(k.Nodes, Node{Reg: 1 + r.IntN(6)})
			cu.Init = len(k.Nodes)
		}
		k.Loops[0].Cursors = append(k.Loops[0].Cursors, cu)
	}
	return k
}

// runTakes is the registers after loop 0's takes from registers holding
// their own numbers: in steps, or with steps nil every source read before
// any take writes.
func runTakes(k *Kernel, steps []takeStep) []int {
	regs := make([]int, 8)
	for i := range regs {
		regs[i] = i
	}
	val := func(cu Cursor, regs []int) int {
		v := 1000 * regs[k.Bufs[cu.Buf].Reg]
		if cu.Init > 0 {
			v += regs[k.Nodes[cu.Init-1].Reg]
		}
		return v
	}
	cus := k.Loops[0].Cursors
	if steps == nil {
		old := slices.Clone(regs)
		for _, cu := range cus {
			regs[k.Ptrs[cu.Ptr-1].Reg] = val(cu, old)
		}
		return regs
	}
	slots := map[int]int{}
	for _, s := range steps {
		cu := cus[s.I]
		switch s.To {
		case takeOwn:
			regs[k.Ptrs[cu.Ptr-1].Reg] = val(cu, regs)
		case takePark:
			slots[s.Slot] = val(cu, regs)
		case takeUnpark:
			regs[k.Ptrs[cu.Ptr-1].Reg] = slots[s.Slot]
		}
	}
	return regs
}
