package jit

import (
	"errors"

	"github.com/jitllm/jitllm/jit/ir"
)

// Regs binds an ordered kernel's values to registers. An argument word
// takes its field's register, or the first spare one free; a count only one
// loop reads is that loop's counter, a vector loop's and its tail's the two
// counters and a scalar loop's the outer one, and a count several loops
// read is kept in a register of its own. A cursor copy takes the first
// spare register free. Floats take vector registers in the order the
// sequence defines them: a result its first operand's register where the
// operand dies there (an fma's the addend's), else the lowest free.
type Regs struct {
	Fields   map[int]int
	Spare    []int
	Counters [2]int
	Outer    int
	Kept     []int
	Vectors  int
}

func (Regs) Name() string { return "regs" }

func (p Regs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Seq == nil {
			return nil, false, nil
		}
		m := k.clone()
		p.gprs(m)
		if p.vectors(m) != nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// gprs binds argument words, loop counters and cursor copies.
func (p Regs) gprs(k *Kernel) {
	used := map[int]bool{}
	take := func(pool []int) int {
		for _, r := range pool {
			if !used[r] {
				used[r] = true
				return r
			}
		}
		return -1
	}
	readers := map[int][]int{}
	for c, l := range k.Loops {
		if l.Count > 0 {
			readers[l.Count-1] = append(readers[l.Count-1], c)
		}
	}
	for _, e := range k.Seq {
		if e.Kind != EntryNode || k.Nodes[e.I].Op != OpParam {
			continue
		}
		i := e.I
		r := -1
		if ls, ok := readers[i]; ok && len(ls) == 1 && !k.isBaseNode(i) {
			c := ls[0]
			switch l := k.Loops[c]; {
			case l.Lanes > 0:
				r = take([]int{p.Counters[0]})
			case k.vectorTail(c):
				r = take([]int{p.Counters[1]})
			default:
				r = take([]int{p.Outer})
			}
			if r >= 0 {
				k.Loops[c].Reg = r + 1
			}
		} else if ok {
			r = take(p.Kept)
		} else if f, ok := p.Fields[int(k.Nodes[i].Imm)]; ok && !used[f] {
			r = take([]int{f})
		}
		if r < 0 {
			r = take(p.Spare)
		}
		k.Nodes[i].Reg = r + 1
		for b := range k.Bufs {
			if k.Bufs[b].Base == i+1 {
				k.Bufs[b].Reg = r + 1
			}
		}
	}
	for c := range k.Loops {
		l := &k.Loops[c]
		for j := range l.Cursors {
			if l.Cursors[j].Copy {
				l.Cursors[j].Reg = take(p.Spare) + 1
			}
		}
		if l.Reg == 0 {
			l.Reg = take([]int{p.Counters[0], p.Counters[1]}) + 1
		}
	}
}

func (k *Kernel) isBaseNode(i int) bool {
	for _, b := range k.Bufs {
		if b.Base == i+1 {
			return true
		}
	}
	return false
}

// vectorTail reports whether loop c takes a vector loop's remaining trips.
func (k *Kernel) vectorTail(c int) bool {
	for v := range k.Loops {
		if k.Loops[v].Lanes > 0 && k.tailOf(v) == c {
			return true
		}
	}
	return false
}

// vectors binds the float values by a scan of the sequence.
func (p Regs) vectors(k *Kernel) error {
	at := map[int]int{}
	loopEnd := map[int]int{}
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryNode:
			at[e.I] = pos
		case EntryEnd:
			loopEnd[e.I] = pos
		}
	}
	last := map[int]int{}
	use := func(v, pos, home int) {
		if _, ok := at[v]; !ok {
			return
		}
		// A value read in a loop it was made outside of lives to the
		// loop's end.
		for c := home; c >= 0; c = k.Loops[c].Parent {
			if h := k.loopOf(v); !k.within(h, c) || h < 0 {
				if e, ok := loopEnd[c]; ok && e > pos {
					pos = e
				}
			}
		}
		last[v] = max(last[v], pos)
	}
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryNode:
			nd := k.Nodes[e.I]
			for _, a := range k.args(&nd) {
				use(*a, pos, k.loopOf(e.I))
			}
		case EntryStore:
			s := k.Stores[e.I]
			for _, a := range k.storeArgs(&s) {
				use(*a, pos, s.At)
			}
		}
	}
	busy := make([]bool, p.Vectors)
	free := func(pos int) {
		for v, l := range last {
			if l == pos && k.Nodes[v].Reg > 0 && k.needsVector(v) {
				busy[k.Nodes[v].Reg-1] = false
			}
		}
	}
	for pos, e := range k.Seq {
		if e.Kind != EntryNode || !k.needsVector(e.I) {
			if e.Kind != EntryNode {
				free(pos)
			}
			continue
		}
		i := e.I
		nd := k.Nodes[i]
		reg := -1
		first := -1
		switch nd.Op {
		case OpAdd, OpSub, OpMul, OpDiv, OpMin, OpMax:
			first = nd.A
		case OpFma:
			first = nd.C
		}
		if first >= 0 && last[first] == pos && k.Nodes[first].Reg > 0 && k.needsVector(first) {
			reg = k.Nodes[first].Reg - 1
		}
		free(pos)
		if reg < 0 {
			for r := range busy {
				if !busy[r] {
					reg = r
					break
				}
			}
		}
		if reg < 0 {
			return errors.New("more floats live at once than the vector registers")
		}
		busy[reg] = true
		k.Nodes[i].Reg = reg + 1
		if last[i] == 0 {
			busy[reg] = false
		}
	}
	return nil
}

// needsVector reports whether node i is a float the code holds in a vector
// register.
func (k *Kernel) needsVector(i int) bool {
	nd := k.Nodes[i]
	return nd.Type == ir.F32 && nd.Op != OpConst
}
