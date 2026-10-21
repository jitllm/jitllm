package jit

import (
	"errors"

	"github.com/jitllm/jitllm/jit/ir"
)

// Regs binds an ordered kernel's values to registers. An argument word
// takes its field's register, or the first spare one free; a count only one
// loop reads is that loop's counter, a vector loop's and its tail's the two
// counters and a scalar loop's the outer one; a count other counts are made
// from is kept in a register of its own, as is each count made. A loop
// whose count is held elsewhere counts in the first counter free, and a
// cursor's copy takes the first spare register free. Values take registers
// in the order the sequence makes them: a result its first operand's where
// the operand dies there (an fma's the addend's, a sum's step the sum's), a
// sum that carries another on that one's, else the lowest free besides its
// operands -- within a trip of a loop of several trips that adds several
// sums, a register one sum's value freed goes to no other sum's; within a trip of an innermost
// loop, a register freed by an operation that wrote its result elsewhere
// goes to nothing else; and a register that held a sum goes to no value of
// a loop that runs after the sum's.
type Regs struct {
	Fields   map[int]int
	Spare    []int
	Counters [2]int
	Outer    int
	Kept     []int
	Vectors  int
	// Scratch is the word a constant is made in before it is splat into
	// a vector, the splats taking the last vector registers in the order
	// they are made.
	Scratch int
}

func (Regs) Name() string { return "regs" }

func (p Regs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Seq == nil {
			return nil, false, nil
		}
		m := k.clone()
		b := binder86{k: m, p: p, used: map[int]bool{}}
		b.lives()
		if b.gprs() != nil || b.vectors() != nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

type binder86 struct {
	k    *Kernel
	p    Regs
	used map[int]bool
	def  map[int]int
	last map[int]int
	end  map[int]int
}

func (b *binder86) take(pool []int) int {
	for _, r := range pool {
		if !b.used[r] {
			b.used[r] = true
			return r
		}
	}
	return -1
}

// lives finds where each value is made and where it is last read: a sum is
// made where its loop starts, and a value read in a loop it was made
// outside of lives to that loop's end.
func (b *binder86) lives() {
	k := b.k
	hs := k.homes()
	b.def, b.last, b.end = map[int]int{}, map[int]int{}, map[int]int{}
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryNode:
			b.def[e.I] = pos
		case EntryLoop:
			for _, f := range k.foldsOf(e.I) {
				b.def[f] = pos
			}
		case EntryEnd:
			b.end[e.I] = pos
		}
	}
	use := func(v, pos, home int) {
		if _, ok := b.def[v]; !ok {
			return
		}
		h := hs[v]
		if k.Nodes[v].Op == OpFold {
			h = k.Nodes[v].At
		}
		for c := home; c >= 0; c = k.Loops[c].Parent {
			outside := h < 0 || !k.within(h, c)
			if outside || k.Nodes[v].Op == OpFold && h == c {
				if e, ok := b.end[c]; ok && e > pos {
					pos = e
				}
			}
		}
		b.last[v] = max(b.last[v], pos)
	}
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryNode:
			nd := k.Nodes[e.I]
			for _, a := range k.args(&nd) {
				use(*a, pos, hs[e.I])
			}
		case EntryStore:
			s := k.Stores[e.I]
			for _, a := range k.storeArgs(&s) {
				use(*a, pos, s.At)
			}
		case EntryLoop:
			l := k.Loops[e.I]
			if l.Count > 0 {
				use(l.Count-1, pos, l.Parent)
			}
			for _, f := range k.foldsOf(e.I) {
				use(k.Nodes[f].B, pos, l.Parent)
			}
		}
	}
}

// gprs binds argument words, counts, loop counters and cursor copies.
func (b *binder86) gprs() error {
	k, p := b.k, b.p
	readers := map[int][]int{}
	for c, l := range k.Loops {
		if l.Count > 0 {
			readers[l.Count-1] = append(readers[l.Count-1], c)
		}
	}
	direct := map[int]bool{}
	words := k.vectorWords()
	for _, e := range k.Seq {
		if e.Kind != EntryNode {
			continue
		}
		i := e.I
		nd := k.Nodes[i]
		if nd.Op != OpParam && (nd.Type == ir.F32 || nd.Op == OpConst || nd.Op == OpLoad || words[i]) {
			continue
		}
		r := -1
		ls, read := readers[i]
		switch {
		case nd.Op == OpParam && read && len(ls) == 1 && !k.isBaseNode(i) && b.uses(i) == 0:
			c := ls[0]
			switch l := k.Loops[c]; {
			case l.Lanes > 0:
				r = b.take([]int{p.Counters[0]})
			case k.vectorTail(c):
				r = b.take([]int{p.Counters[1]})
			default:
				r = b.take([]int{p.Outer})
			}
			if r >= 0 {
				k.Loops[c].Reg = r + 1
				direct[c] = true
			}
		case nd.Op == OpParam && k.isBaseNode(i):
			if f, ok := p.Fields[int(nd.Imm)]; ok && !b.used[f] {
				r = b.take([]int{f})
			}
		case nd.Op == OpParam:
			r = b.take(p.Kept)
		default:
			if x := nd.A; b.last[x] == b.def[i] && k.Nodes[x].Reg > 0 && k.Nodes[x].Type != ir.F32 {
				r = k.Nodes[x].Reg - 1
			} else {
				r = b.take(p.Kept)
			}
		}
		if r < 0 {
			r = b.take(p.Spare)
		}
		if r < 0 {
			return errors.New("more words live at once than registers")
		}
		k.Nodes[i].Reg = r + 1
		for j := range k.Bufs {
			if k.Bufs[j].Base == i+1 {
				k.Bufs[j].Reg = r + 1
			}
		}
	}
	lastLoop := map[int]int{}
	for c, l := range k.Loops {
		for _, cu := range l.Cursors {
			if cu.Ptr > 0 {
				lastLoop[cu.Ptr] = c
			}
		}
	}
	for _, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			for _, cu := range k.Loops[e.I].Cursors {
				if cu.Take && k.Ptrs[cu.Ptr-1].Reg == 0 {
					r := b.take(p.Spare)
					if r < 0 {
						return errors.New("no register for a copy of an address")
					}
					k.Ptrs[cu.Ptr-1].Reg = r + 1
				}
			}
		case EntryEnd:
			for ptr, c := range lastLoop {
				if c == e.I && k.Ptrs[ptr-1].Reg > 0 {
					b.used[k.Ptrs[ptr-1].Reg-1] = false
				}
			}
		}
	}
	held := map[int]bool{}
	for c := range k.Loops {
		if direct[c] {
			held[k.Loops[c].Reg-1] = true
		}
	}
	for c := range k.Loops {
		l := &k.Loops[c]
		if l.Reg != 0 || l.once() {
			continue
		}
		for _, r := range []int{p.Counters[0], p.Counters[1]} {
			if !held[r] && (!b.used[r] || b.counterFree(r, c)) {
				l.Reg = r + 1
				b.used[r] = true
				break
			}
		}
		if l.Reg == 0 {
			return errors.New("no counter free")
		}
	}
	return nil
}

// counterFree reports whether counter r counts no loop that is loop c, runs
// around it or inside it.
func (b *binder86) counterFree(r, c int) bool {
	k := b.k
	for d, l := range k.Loops {
		if d != c && l.Reg == r+1 && (k.within(c, d) || k.within(d, c)) {
			return false
		}
	}
	for _, nd := range k.Nodes {
		if nd.Reg == r+1 && nd.Type != ir.F32 {
			return false
		}
	}
	return true
}

// uses is how many nodes read node i.
func (b *binder86) uses(i int) int {
	n := 0
	for _, nd := range b.k.Nodes {
		for _, a := range b.k.args(&nd) {
			if *a == i {
				n++
			}
		}
	}
	return n
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

// needsVector reports whether node i is a value the code holds in a vector
// register: a float, a weight's word, or a word of halves.
func (k *Kernel) needsVector(i int) bool {
	nd := k.Nodes[i]
	if nd.Mem {
		return false
	}
	if nd.Type == ir.F32 || nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind == OperandWeight {
		return nd.Op != OpConst
	}
	return k.vectorWords()[i]
}

// vectorWords is the words held in vector registers: the halves a float is
// narrowed to, what is made from them, and the constants they are made
// with.
func (k *Kernel) vectorWords() map[int]bool {
	out := map[int]bool{}
	for changed := true; changed; {
		changed = false
		for i, nd := range k.Nodes {
			if out[i] || nd.Type == ir.F32 {
				continue
			}
			v := nd.Op == OpHalf
			if !v && (nd.Op == OpAnd || nd.Op == OpGtW || nd.Op == OpAndNot) {
				for _, a := range k.args(&nd) {
					v = v || out[*a]
				}
			}
			if v {
				out[i] = true
				changed = true
				for _, a := range k.args(&nd) {
					if k.Nodes[*a].Op == OpConst && !out[*a] {
						out[*a] = true
					}
				}
			}
		}
	}
	return out
}

// vectors binds the floats by a scan of the sequence.
func (b *binder86) vectors() error {
	k := b.k
	busy := make([]bool, b.p.Vectors)
	tag := make([]int, b.p.Vectors)
	chain := k.chains()
	stepOf := map[int]int{}
	for i, nd := range k.Nodes {
		if nd.Op == OpFold && nd.A != i {
			stepOf[nd.A] = i
		}
	}
	free := func(pos int, spent bool) {
		for v, l := range b.last {
			if _, step := stepOf[v]; step {
				continue
			}
			if l == pos && k.Nodes[v].Reg > 0 && k.needsVector(v) && busy[k.Nodes[v].Reg-1] {
				r := k.Nodes[v].Reg - 1
				busy[r] = false
				tag[r] = chain[v]
				if spent {
					tag[r] = -1
				}
			}
		}
	}
	innermost := func(c int) bool { return c >= 0 && len(k.innerLoops(c)) == 0 }
	held := make([]int, b.p.Vectors)
	for r := range held {
		held[r] = -1
	}
	lowest := func(ch, home int) int {
		for r := range busy {
			if busy[r] || tag[r] != 0 && tag[r] != ch {
				continue
			}
			if h := held[r]; home >= 0 && h >= 0 && !k.within(h, home) {
				continue
			}
			return r
		}
		return -1
	}
	hs := k.homes()
	give := func(i, r int) error {
		if r < 0 {
			return errors.New("more floats live at once than the vector registers")
		}
		busy[r], tag[r] = true, 0
		k.Nodes[i].Reg = r + 1
		return nil
	}
	var splats []int
	for _, e := range k.Seq {
		if e.Kind == EntryNode && k.Nodes[e.I].Op == OpConst && k.needsVector(e.I) {
			splats = append(splats, e.I)
		}
	}
	for j, i := range splats {
		r := b.p.Vectors - len(splats) + j
		k.Nodes[i].Reg, k.Nodes[i].Tmp = r+1, b.p.Scratch+1
		busy[r] = true
	}
	for pos, e := range k.Seq {
		if e.Kind == EntryNode && k.Nodes[e.I].Op == OpConst {
			continue
		}
		switch e.Kind {
		case EntryLoop:
			for _, f := range k.foldsOf(e.I) {
				init := k.Nodes[f].B
				if k.Nodes[init].Op == OpFold && k.Nodes[init].Reg > 0 {
					k.Nodes[f].Reg = k.Nodes[init].Reg
					continue
				}
				if err := give(f, lowest(0, -1)); err != nil {
					return err
				}
				held[k.Nodes[f].Reg-1] = e.I
			}
			continue
		case EntryEnd:
			free(pos, false)
			for r := range tag {
				tag[r] = 0
			}
			continue
		case EntryStore:
			free(pos, false)
			continue
		}
		i := e.I
		nd := k.Nodes[i]
		if !k.needsVector(i) {
			free(pos, false)
			continue
		}
		reg := -1
		if f, ok := stepOf[i]; ok && k.Nodes[f].Reg > 0 {
			reg = k.Nodes[f].Reg - 1
		}
		first := -1
		switch nd.Op {
		case OpAdd, OpSub, OpMul, OpMin, OpMax, OpHSum, OpBitcast, OpBroadcast, OpSqrt, OpAnd, OpGtW, OpAndNot:
			first = nd.A
		case OpDiv:
			first = nd.B
		case OpFma:
			first = nd.C
		}
		if reg < 0 && first >= 0 && k.Nodes[first].Reg > 0 && k.needsVector(first) && (b.last[first] == pos || nd.Op == OpBitcast) {
			reg = k.Nodes[first].Reg - 1
		}
		if reg >= 0 {
			free(pos, false)
		} else {
			reg = lowest(chain[i], hs[i])
			free(pos, innermost(hs[i]))
		}
		if err := give(i, reg); err != nil {
			return err
		}
		if nd.Op == OpHSum {
			t := lowest(0, hs[i])
			if t < 0 {
				return errors.New("no register for a lane sum")
			}
			k.Nodes[i].Tmp = t + 1
		}
		if _, ok := b.last[i]; !ok {
			if _, step := stepOf[i]; !step {
				busy[reg] = false
			}
		}
	}
	return nil
}

// chains is, for each value in a trip of a loop adding several sums, 1 +
// the sum it is added into; 0 for the rest.
func (k *Kernel) chains() map[int]int {
	out := map[int]int{}
	hs := k.homes()
	for c, l := range k.Loops {
		folds := k.foldsOf(c)
		if l.Copies < 2 || len(folds) < 2 || l.once() {
			continue
		}
		for j, f := range folds {
			var walk func(i int)
			walk = func(i int) {
				nd := k.Nodes[i]
				if nd.Op == OpFold || out[i] != 0 || hs[i] != c {
					return
				}
				out[i] = j + 1
				for _, a := range k.args(&nd) {
					walk(*a)
				}
			}
			walk(k.Nodes[f].A)
		}
	}
	return out
}
