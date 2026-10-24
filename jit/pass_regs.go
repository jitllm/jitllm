package jit

import (
	"cmp"
	"errors"
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/jit/x86"
)

// Regs binds an ordered kernel's values to registers. An argument word
// takes the first of its field's registers free, or the first spare one
// free; a count only one loop reads is that loop's counter -- a tile pass's
// is kept in the outer register and the pass counts a copy of it -- a vector loop's and its tail's the two
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
// a loop that runs after the sum's -- each but where no other register is
// free. A pair's later head takes its first
// head's registers, moved past them. Where the floats do not fit, a constant
// the loops read is loaded where each read is, the earliest read first,
// until they do.
type Regs struct {
	Fields   map[int][]int
	Spare    []int
	Counters [2]int
	Outer    int
	Kept     []int
	Vectors  int
	// Scratch is the word a constant is made in before it is splat into
	// a vector, the splats taking the last vector registers in the order
	// they are made.
	Scratch int
	// Resident is the vector register each word of the device's shared
	// constant block is held in for the whole call, loaded first in word
	// order; a word it does not name is loaded where each read is.
	Resident map[int]int
	// Walks are the registers that copies of an address a kernel walks in
	// several passes take first; in such a kernel a vector loop's tail
	// counts in the second counter or the first of these free, set as the
	// pass starts.
	Walks []int
	// Lanes are the registers a lane sum, maximum or minimum takes for its
	// scratch, the first of them free.
	Lanes []int
	// Hoist, where set, is the registers below which a value made outside
	// every loop and read in one is held, the rest the loops' own; a
	// constant the loops read is loaded where it is read where they do not
	// fit.
	Hoist int
	// Scalars, where set, is 1 + the first of the registers a constant read
	// outside every loop takes, each its own, upward; Keep is 1 + the
	// register of the zero a tail keeps lanes with.
	Scalars, Keep int
	// Reloads are the general registers a spilled value is loaded into at
	// its reads, taken from every pool when a kernel's words do not fit and
	// the binding spills (spillReg).
	Reloads [2]int
}

// spillSlots is the red zone's 8-byte slots below RSP a kernel spills into:
// all sixteen but the last, which holds a vector shift's count
// (lowercpu.go, shiftCount).
const spillSlots = 15

func (Regs) Name() string { return "regs" }

func (p Regs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Seq == nil {
			return nil, false, nil
		}
		w := k.clone()
		moved := map[int]bool{}
		if p.Resident != nil {
			w.residents(p.Resident, moved)
		}
		late := w.lateParams()
		for range len(k.Nodes) {
			m := w.clone()
			b := binder86{k: m, p: p, used: map[int]bool{}, late: late, isa: x86.ISA(d.ISA)}
			b.lives()
			if b.gprs() != nil {
				// The words do not fit: again with the reload registers kept
				// out of every pool, spilling where none is free.
				m = w.clone()
				b = binder86{k: m, p: p, used: map[int]bool{}, late: late, spill: true, isa: x86.ISA(d.ISA)}
				b.lives()
				if b.gprs() != nil {
					return nil, false, nil
				}
			}
			if b.vectors() == nil {
				return []*Kernel{m}, true, nil
			}
			if m.Ones > 0 {
				// Again with the byte dots making their ones each time.
				b.noOnes = true
				if b.vectors() == nil {
					return []*Kernel{m}, true, nil
				}
			}
			if !w.rematerialize(moved) {
				return nil, false, nil
			}
		}
		return nil, false, nil
	})
}

type binder86 struct {
	k    *Kernel
	p    Regs
	used map[int]bool
	// spill binds with the reload registers held back, a word that finds no
	// register taking one from the live value read latest, which spills.
	spill bool
	// isa is the instruction sets the host executes; noOnes keeps no
	// register of ones for the kernel's byte dots.
	isa    x86.ISA
	noOnes bool
	// owner is each slot's holder: 0 none, 1 + a value's node, or -1 for a
	// loop's iteration or an address copy, which hold theirs to the end.
	owner [spillSlots + 1]int
	// counters are registers a spilling binding holds back for the loops
	// past the two the profile's counters count.
	counters []int
	// params are the argument words held in a register that a spill may
	// take: re-read from the argument block at each read instead.
	params map[int]int
	late   map[int]bool
	def    map[int]int
	last   map[int]int
	end    map[int]int
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

// lives finds where each value is made and where it is last read: a value
// read in a loop the sequence makes it outside of lives to that loop's end,
// as a sum does in its own loop.
func (b *binder86) lives() {
	k := b.k
	hs := k.Homes()
	b.def, b.last, b.end = map[int]int{}, map[int]int{}, map[int]int{}
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryNode:
			b.def[e.I] = pos
		case EntryEnd:
			b.end[e.I] = pos
		}
	}
	start := map[int]int{}
	for p, e := range k.Seq {
		if e.Kind == EntryLoop {
			start[e.I] = p
		}
	}
	use := func(v, pos, home int) {
		d, ok := b.def[v]
		if !ok {
			return
		}
		h := hs[v]
		if k.Nodes[v].Op == OpFold {
			h = k.Nodes[v].At
		}
		for c := home; c >= 0; c = k.Loops[c].Parent {
			outside := d < start[c] || d > b.end[c]
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
			if nd.Op == OpFold {
				use(nd.B, pos, k.Loops[nd.At].Parent)
				continue
			}
			for _, a := range k.Operands(&nd) {
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
		}
	}
}

// gprs binds argument words, counts, loop counters and cursor copies.
func (b *binder86) gprs() error {
	k, p := b.k, b.p
	if b.spill {
		// Loops count at once only along one nest: its depth past the two
		// counters is what to hold back.
		depth := 0
		for c := range k.Loops {
			d := 0
			for x := c; x >= 0; x = k.Loops[x].Parent {
				if !k.Loops[x].once() {
					d++
				}
			}
			depth = max(depth, d)
		}
		for range depth - 2 {
			if r := b.take(p.Spare); r >= 0 {
				b.counters = append(b.counters, r)
			}
		}
	}
	readers := map[int][]int{}
	for c, l := range k.Loops {
		if l.Count > 0 {
			readers[l.Count-1] = append(readers[l.Count-1], c)
		}
	}
	direct := map[int]bool{}
	words := k.VectorWords()
	// A value made in a general register frees it after its last read; an
	// argument word, a length, a count and a buffer's address keep theirs,
	// read through what they address rather than as values.
	freed, valued := map[int]int{}, map[int]bool{}
	if b.spill {
		// The reload registers, and the counters, which a spilling
		// binding keeps for its loops rather than its values.
		for _, r := range p.Reloads {
			b.used[r] = true
		}
		for _, r := range p.Counters {
			b.used[r] = true
		}
	}
	foldOf := map[int]int{}
	for f, nd := range k.Nodes {
		if nd.Op == OpFold {
			foldOf[nd.A] = f
		}
	}
	keeps := func(i int) bool {
		nd := k.Nodes[i]
		if nd.Op == OpParam || nd.Op == OpLength || nd.Op == OpSpan || nd.Op == OpFold || k.madeCount(i) >= 0 {
			return true
		}
		if _, ok := readers[i]; ok {
			return true
		}
		for j := range k.Bufs {
			if k.Bufs[j].Base == i+1 {
				return true
			}
		}
		return false
	}
	// A loop whose iteration a value reads holds it in a register of its own
	// from the loop's start to its end.
	iterReg := map[int]int{}
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			if it := k.iterOf(e.I); it >= 0 && k.Nodes[it].Reg == 0 && b.uses(it) > 0 {
				r := b.take(p.Spare)
				if r < 0 {
					r = b.take(p.Kept)
				}
				if r < 0 && b.spill {
					r = b.evictAt(freed, pos)
				}
				if r < 0 && b.spill {
					r = b.rereadParam()
				}
				if r < 0 && b.spill {
					// The iteration in a slot: zeroed on entry, counted
					// there, loaded where it is read.
					if s := b.slot(pos, -1); s > 0 {
						k.Nodes[it].Spill = s
						continue
					}
				}
				if r < 0 {
					return errors.New("no register for a loop's iteration")
				}
				k.Nodes[it].Reg = r + 1
				iterReg[e.I] = r
				valued[r] = true
			}
			continue
		case EntryEnd:
			if r, ok := iterReg[e.I]; ok {
				b.used[r] = false
			}
			continue
		case EntryStore:
			continue
		}
		i := e.I
		nd := k.Nodes[i]
		if nd.Op != OpParam && (nd.Type == ir.F32 || nd.Op == OpConst || nd.Op == OpLoad && k.needsVector(i) || words[i]) {
			continue
		}
		for v, r := range freed {
			if b.last[v] < b.def[i] {
				b.used[r] = false
				delete(freed, v)
			}
		}
		if f, ok := foldOf[i]; ok && k.Nodes[f].Reg > 0 && k.Nodes[f].Type != ir.F32 {
			// A sum's step writes the sum's register.
			k.Nodes[i].Reg = k.Nodes[f].Reg
			continue
		}
		r := -1
		ls, read := readers[i]
		switch {
		case nd.Op == OpParam && read && len(ls) == 1 && k.Loops[ls[0]].Walk:
			r = b.take([]int{p.Outer})
		case nd.Op == OpParam && k.madeCounts(i) != nil:
			r = b.take([]int{p.Counters[0]})
		// A count is its loop's counter only where the loop is entered once:
		// a loop inside another counts its own copy, the count kept for the
		// next entry.
		case nd.Op != OpParam && k.madeCount(i) >= 0 && k.Loops[k.madeCount(i)].Parent < 0:
			c := k.madeCount(i)
			if k.Loops[c].Lanes > 0 && b.last[nd.A] == b.def[i] && k.Nodes[nd.A].Reg == p.Counters[0]+1 {
				r = p.Counters[0]
			} else {
				r = b.take([]int{p.Counters[1]})
			}
			if r >= 0 {
				k.Loops[c].Reg = r + 1
				direct[c] = true
			}
		case nd.Op == OpParam && read && len(ls) == 1 && !k.isBaseNode(i) && b.uses(i) == 0 && k.Loops[ls[0]].Parent < 0:
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
		case b.reloads(i):
			for j := range k.Bufs {
				if k.Bufs[j].Base == i+1 {
					k.Bufs[j].Reload = true
				}
			}
			continue
		case nd.Op == OpParam && k.isBaseNode(i) && !b.late[i]:
			r = b.take(p.Fields[int(nd.Imm)])
		case b.late[i]:
			r = b.take(p.Spare)
		case nd.Op == OpParam:
			r = b.take(p.Kept)
		default:
			// A value takes the register of an operand read last here: the
			// lowering reads both before it writes, whichever it takes.
			dies := func(x int) bool {
				return b.last[x] == b.def[i] && k.Nodes[x].Reg > 0 && k.Nodes[x].Spill == 0 && k.Nodes[x].Type != ir.F32 && !keeps(x) && k.Nodes[x].Op != OpIter
			}
			switch {
			case dies(nd.A):
				r = k.Nodes[nd.A].Reg - 1
				delete(freed, nd.A)
			case b.spill && binaryInt(nd.Op) && dies(nd.B):
				r = k.Nodes[nd.B].Reg - 1
				delete(freed, nd.B)
			default:
				r = b.take(p.Kept)
			}
		}
		if r < 0 {
			r = b.take(p.Spare)
		}
		if r < 0 && b.spill {
			r = b.evict(freed, i)
		}
		if r < 0 && b.spill {
			r = b.rereadParam()
		}
		if r < 0 && b.spill && nd.Op == OpParam && !k.isBaseNode(i) {
			// An argument word with no register: read from the argument
			// block wherever it is read.
			k.Nodes[i].Spill = -1
			continue
		}
		if r < 0 && b.spill && nd.Op != OpParam && nd.Op != OpFold {
			// Made in the reloaded addresses' register and stored to a slot
			// at once: its reads load it from there. Not a sum, whose step
			// writes its register each trip and stores nothing.
			if s := b.slot(b.def[i], i+1); s > 0 {
				k.Nodes[i].Reg, k.Nodes[i].Spill = int(reloadReg)+1, s
				continue
			}
		}
		if r < 0 {
			return errors.New("more words live at once than registers")
		}
		k.Nodes[i].Reg = r + 1
		valued[r] = true
		if b.spill && nd.Op == OpParam && !k.isBaseNode(i) && !b.countsIn(r) {
			if b.params == nil {
				b.params = map[int]int{}
			}
			b.params[i] = r
		}
		if !keeps(i) {
			freed[i] = r
		}
		for j := range k.Bufs {
			if k.Bufs[j].Base == i+1 {
				k.Bufs[j].Reg = r + 1
			}
		}
	}
	// The counters and copies bound below are bound after the values, so a
	// register a value or an iteration held at any point stays taken: freed
	// in the walk, it was free only between that value's reads.
	for r := range valued {
		b.used[r] = true
	}
	for c := range k.Loops {
		l := &k.Loops[c]
		if l.Kept && l.Reg == 0 {
			if r := b.take([]int{p.Outer}); r >= 0 {
				l.Reg = r + 1
			}
		}
		if !l.Walk || l.Reg != 0 || l.once() {
			continue
		}
		if r := b.take([]int{p.Counters[0], p.Counters[1]}); r >= 0 {
			l.Reg = r + 1
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
	passes := k.passes()
	for _, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			for _, cu := range k.Loops[e.I].Cursors {
				if cu.Take && k.Ptrs[cu.Ptr-1].Reg == 0 && k.Ptrs[cu.Ptr-1].Spill == 0 {
					r := -1
					if passes {
						r = b.take(p.Walks)
					}
					if r < 0 {
						r = b.take(p.Spare)
					}
					if r < 0 && b.spill {
						if s := b.slot(0, -1); s > 0 {
							k.Ptrs[cu.Ptr-1].Spill = s
							continue
						}
					}
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
		if passes && k.vectorTail(c) {
			copies := map[int]bool{}
			for _, d := range []int{c, l.Of - 1} {
				for _, cu := range k.Loops[d].Cursors {
					if cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Reg > 0 {
						copies[k.Ptrs[cu.Ptr-1].Reg-1] = true
					}
				}
			}
			for _, r := range append([]int{p.Counters[1]}, p.Walks...) {
				if !held[r] && !copies[r] && (!b.used[r] || b.counterFree(r, c)) {
					l.Reg = r + 1
					b.used[r] = true
					break
				}
			}
			if l.Reg != 0 {
				continue
			}
		}
		// The counters first, then any register no value holds while the
		// loop runs: a third nested loop counts in one of those.
		for i, r := range slices.Concat([]int{p.Counters[0], p.Counters[1]}, b.counters, p.Spare, p.Kept) {
			free := !b.used[r] || b.counterFree(r, c)
			if i >= 2 && i < 2+len(b.counters) {
				free = b.counterFree(r, c)
			} else if i >= 2 {
				free = !b.used[r] && b.counterFree(r, c)
			}
			if !held[r] && free && !b.copyIn(r) {
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
	return !b.copyIn(r)
}

// copyIn reports whether a copy of an address a loop walks is held in
// register r.
func (b *binder86) copyIn(r int) bool {
	for _, p := range b.k.Ptrs {
		if p.Reg == r+1 {
			return true
		}
	}
	return false
}

// uses is how many nodes read node i.
func (b *binder86) uses(i int) int {
	n := 0
	for _, nd := range b.k.Nodes {
		for _, a := range b.k.Operands(&nd) {
			if *a == i {
				n++
			}
		}
	}
	return n
}

// slot is a red-zone slot free from sequence position from on, taken for
// owner (1 + a value's node, or -1 for good); -1 where none is.
func (b *binder86) slot(from, owner int) int {
	for s := 1; s <= spillSlots; s++ {
		if o := b.owner[s]; o == 0 || o > 0 && b.last[o-1] < from {
			b.owner[s] = owner
			return s
		}
	}
	return -1
}

// evict spills the held value read latest past node i and returns its
// register, -1 where nothing can spill: the value is stored where it is made
// and loaded at every read, so its register is free from here on.
func (b *binder86) evict(freed map[int]int, i int) int {
	return b.evictAt(freed, b.def[i])
}

// evictAt is evict at sequence position pos.
func (b *binder86) evictAt(freed map[int]int, pos int) int {
	v, last := -1, -1
	for x := range freed {
		if l := b.last[x]; l > pos && l > last {
			v, last = x, l
		}
	}
	if v < 0 {
		return -1
	}
	s := b.slot(b.def[v], v+1)
	if s < 0 {
		return -1
	}
	b.k.Nodes[v].Spill = s
	r := freed[v]
	delete(freed, v)
	return r
}

// rereadParam takes the register of an argument word held in one, which is
// read from the argument block at each read from here on (Node.Spill -1),
// and returns it; -1 where none is held.
func (b *binder86) rereadParam() int {
	v := -1
	for x := range b.params {
		if v < 0 || x < v {
			v = x
		}
	}
	if v < 0 {
		return -1
	}
	r := b.params[v]
	delete(b.params, v)
	b.k.Nodes[v].Spill = -1
	return r
}

// binaryInt reports whether the CPU lowering's general-register form of op
// writes its result after reading both operands, whatever the result's
// register: a result in the second operand's register is the same answer.
func binaryInt(op Op) bool {
	switch op {
	case OpAdd, OpSub, OpMul, OpAnd, OpOr, OpXor, OpShl, OpShr, OpMin, OpMax:
		return true
	}
	return false
}

// countsIn reports whether a loop counts in register r.
func (b *binder86) countsIn(r int) bool {
	for _, l := range b.k.Loops {
		if l.Reg == r+1 {
			return true
		}
	}
	return false
}

// reloadReg is the register a kernel with no vector loop reloads a buffer's
// address into at each access, for a buffer no loop walks: such an address
// holds no register for the kernel's life.
const reloadReg = x86.R15

// reloads reports whether buffer address i is reloaded at each access rather
// than held: a kernel with no vector loop, a buffer no loop's cursor walks and
// no pointer copies, and an address read as nothing but an address.
func (b *binder86) reloads(i int) bool {
	k := b.k
	if !k.isBaseNode(i) || b.uses(i) > 0 {
		return false
	}
	for j, bf := range k.Bufs {
		if bf.Base != i+1 {
			continue
		}
		// The address itself walked is held; copies a loop takes of it
		// read it once a take, from its argument word.
		for _, l := range k.Loops {
			for _, cu := range l.Cursors {
				if cu.Buf == j && cu.Ptr == 0 {
					return false
				}
			}
		}
	}
	return true
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
	if nd.Type == ir.F32 || nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind == OperandWeight && k.vectorized() {
		return nd.Op != OpConst || nd.Imm == 0 && k.zeroOperand(i)
	}
	return k.VectorWords()[i]
}

// vectorized reports whether any loop of k runs in vector lanes. A kernel
// none of whose loops does keeps its integers in general registers: an
// index, a weight's word and what is made from them.
func (k *Kernel) vectorized() bool {
	for _, l := range k.Loops {
		if l.Lanes > 0 || l.Wide > 0 {
			return true
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLanes {
			return true
		}
	}
	return false
}

// VectorWords is the words held in vector registers: the halves a float is
// narrowed to, the integers a float is rounded or converted to or from and
// a float's bits, masks, what is made from them, what they are made of, and
// the constants they are made with.
func (k *Kernel) VectorWords() map[int]bool {
	out := map[int]bool{}
	data := func(i int) bool {
		nd := k.Nodes[i]
		return nd.Type != ir.F32 && nd.Op != OpParam && nd.Op != OpIter && nd.Op != OpLength && nd.Op != OpSpan && nd.Op != OpCoord
	}
	vec := k.vectorized()
	for i, nd := range k.Nodes {
		switch {
		case nd.Op == OpLanes || nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind == OperandConsts && nd.Type != ir.F32:
			out[i] = true
		case vec && (nd.Op == OpCvtF32 || nd.Op == OpWidenHalf || nd.Op == OpBitcast && nd.Type == ir.F32):
			if data(nd.A) && k.Nodes[nd.A].Op != OpConst {
				out[nd.A] = true
			}
		case nd.Op == OpBitcast || nd.Op == OpCvtI32:
			if k.Nodes[nd.A].Type == ir.F32 {
				out[i] = true
			}
		}
	}
	for grew := true; grew; {
		before := len(out)
		for changed := true; changed; {
			changed = false
			for i := range k.Nodes {
				if !out[i] {
					continue
				}
				nd := k.Nodes[i]
				if nd.Op == OpLoad || nd.Op == OpFold {
					continue
				}
				for x, a := range k.Operands(&nd) {
					// A shift's count shifts every lane alike: it is a vector
					// only where it is made of one.
					if x == 1 && (nd.Op == OpShl || nd.Op == OpShr || nd.Op == OpSar) {
						continue
					}
					if data(*a) && k.Nodes[*a].Op != OpConst && !out[*a] && k.Nodes[*a].Type != ir.Pred {
						out[*a], changed = true, true
					}
				}
			}
		}
		for changed := true; changed; {
			changed = false
			for i, nd := range k.Nodes {
				if out[i] || nd.Type == ir.F32 {
					continue
				}
				v := nd.Op == OpHalf || nd.Op == OpRoundI32 || (nd.Op == OpLess || nd.Op == OpEq) && (k.Nodes[nd.A].Type == ir.F32 || out[nd.A] || out[nd.B])
				if !v && (nd.Op == OpAnd || nd.Op == OpGtW || nd.Op == OpAndNot || nd.Op == OpXor || nd.Op == OpShr || nd.Op == OpShl || nd.Op == OpSar || (nd.Type == ir.I32 || nd.Type == ir.U32) && nd.Op != OpConst && nd.Op != OpLoad) {
					for _, a := range k.Operands(&nd) {
						v = v || out[*a]
					}
				}
				if !v && nd.Type == ir.I32 && nd.Op == OpFold {
					v = true
				}
				if v {
					out[i] = true
					changed = true
				}
			}
		}
		grew = len(out) > before
	}
	for i, nd := range k.Nodes {
		if !out[i] || nd.Op == OpLoad || nd.Op == OpFold {
			continue
		}
		for x, a := range k.Operands(&nd) {
			if k.Nodes[*a].Op == OpConst && k.Nodes[*a].Type != ir.F32 && !(x == 1 && (nd.Op == OpShl || nd.Op == OpShr || nd.Op == OpSar)) {
				out[*a] = true
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
				if spent && !k.isZero(v) {
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
	shared := map[int]int{}
	next := b.p.Scalars - 1
	hs0 := k.Homes()
	kept := map[int]map[int]bool{}
	for i, nd := range k.Nodes {
		if nd.Hint == 0 || i >= len(hs0) {
			continue
		}
		for c := hs0[i]; c >= 0; c = k.Loops[c].Parent {
			if kept[c] == nil {
				kept[c] = map[int]bool{}
			}
			kept[c][nd.Hint-1] = true
		}
	}
	span := map[int][2]int{}
	for p, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			span[e.I] = [2]int{p, len(k.Seq)}
		case EntryEnd:
			span[e.I] = [2]int{span[e.I][0], p}
		}
	}
	depth := make([]int, len(k.Seq))
	for p, d := 0, 0; p < len(k.Seq); p++ {
		switch k.Seq[p].Kind {
		case EntryLoop:
			d++
		case EntryEnd:
			d--
		}
		depth[p] = d
	}
	inner := map[int]bool{}
	for p, e := range k.Seq {
		if e.Kind == EntryNode && depth[p] > 0 && k.Nodes[e.I].Hint > 0 {
			inner[k.Nodes[e.I].Hint-1] = true
		}
	}
	var hinted []int
	for i, nd := range k.Nodes {
		if _, made := b.def[i]; made && nd.Hint > 0 {
			hinted = append(hinted, i)
		}
	}
	reserved := map[int]bool{}
	bar := func(i int) {
		clear(reserved)
		d, l := b.def[i], b.last[i]
		for c, rs := range kept {
			if sp := span[c]; sp[0] <= l && d <= sp[1] {
				for r := range rs {
					reserved[r] = true
				}
			}
		}
		for _, h := range hinted {
			if h != i && d < b.def[h] && b.def[h] <= l {
				reserved[k.Nodes[h].Hint-1] = true
			}
		}
	}
	lowest := func(ch, home int) int {
		for r := range busy {
			if busy[r] || reserved[r] || tag[r] != 0 && tag[r] != ch {
				continue
			}
			if h := held[r]; home >= 0 && h >= 0 && !k.within(h, home) {
				continue
			}
			return r
		}
		for r := range busy {
			if !busy[r] {
				return r
			}
		}
		return -1
	}
	hs := k.Homes()
	give := func(i, r int) error {
		if r < 0 {
			return fmt.Errorf("more floats live at once than the vector registers, at node %d (%s)", i, k.Nodes[i].Op)
		}
		busy[r], tag[r] = true, 0
		k.Nodes[i].Reg = r + 1
		if l := k.Nodes[i].Share; l > 0 && shared[l] == 0 {
			shared[l] = r + 1
		}
		return nil
	}
	var splats []int
	for _, e := range k.Seq {
		if e.Kind == EntryNode && k.Nodes[e.I].Op == OpConst && k.needsVector(e.I) {
			splats = append(splats, e.I)
		}
	}
	if len(splats) > b.p.Vectors {
		return fmt.Errorf("%d constants to splat and %d vector registers", len(splats), b.p.Vectors)
	}
	// The splats are made before every loop, so their scratch is no
	// loop's counter and no parameter's register, which hold across them.
	across := map[int]bool{}
	for _, l := range k.Loops {
		if l.Reg > 0 {
			across[l.Reg-1] = true
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpParam && nd.Reg > 0 {
			across[nd.Reg-1] = true
		}
	}
	scratch := b.p.Scratch
	if b.used[scratch] || across[scratch] {
		scratch = -1
		for _, r := range slices.Concat(b.p.Spare, b.p.Kept) {
			if !b.used[r] && !across[r] {
				scratch = r
				break
			}
		}
	}
	if scratch < 0 {
		// The reloaded addresses' register: it holds one only inside a
		// loop, and the splats are made before them.
		scratch = int(reloadReg)
	}
	for j, i := range splats {
		r := b.p.Vectors - len(splats) + j
		k.Nodes[i].Reg, k.Nodes[i].Tmp = r+1, scratch+1
		busy[r] = true
	}
	k.Ones = 0
	if b.isa&x86.ISAVNNI == 0 && !b.noOnes {
		for i, nd := range k.Nodes {
			if nd.Op.byteDot() && k.needsVector(i) {
				// A byte dot's pair sums by ones, made once for the kernel.
				if r := b.p.Vectors - len(splats) - 1; r >= 0 && !busy[r] {
					busy[r], k.Ones = true, r+1
				}
				break
			}
		}
	}
	cp := k.copies()
	for pos, e := range k.Seq {
		if e.Kind == EntryNode && cp[e.I] > 0 {
			free(pos, false)
			continue
		}
		if e.Kind == EntryNode && k.Nodes[e.I].Op == OpConst && k.Nodes[e.I].Type != ir.F32 {
			continue
		}
		if e.Kind == EntryNode && k.Nodes[e.I].Op == OpFold {
			f := e.I
			init := k.Nodes[f].B
			if k.Nodes[init].Op == OpFold && k.Nodes[init].Reg > 0 {
				k.Nodes[f].Reg = k.Nodes[init].Reg
				continue
			}
			want := lowest(0, -1)
			if h := k.Nodes[f].Hint - 1; h >= 0 && !busy[h] {
				want = h
			}
			if err := give(f, want); err != nil {
				return err
			}
			held[k.Nodes[f].Reg-1] = k.Nodes[f].At
			continue
		}
		switch e.Kind {
		case EntryLoop:
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
		bar(i)
		reg := -1
		if r := b.scalar(i, hs, &next); r >= 0 && !busy[r] {
			free(pos, false)
			if err := give(i, r); err != nil {
				return err
			}
			continue
		}
		if nd.Hint > 0 || nd.Share > 0 && shared[nd.Share] > 0 {
			free(pos, false)
			want := nd.Hint - 1
			if nd.Share > 0 && shared[nd.Share] > 0 {
				want = shared[nd.Share] - 1
			}
			if want < 0 || busy[want] {
				want = lowest(chain[i], hs[i])
			}
			if depth[pos] == 0 && nd.Op == OpLoad && inner[want] && k.readInLoop(i) {
				return errors.New("a constant held across the loops on a register a loop's value is hinted to")
			}
			reg = want
		} else if r, ok := b.resident(i); ok {
			reg = r
			free(pos, false)
			if err := give(i, reg); err != nil {
				return err
			}
			continue
		} else {
			if f, ok := stepOf[i]; ok && k.Nodes[f].Reg > 0 {
				reg = k.Nodes[f].Reg - 1
			}
			first := -1
			switch nd.Op {
			case OpAdd, OpSub, OpMul, OpMin, OpMax, OpHSum, OpHMax, OpHMin, OpBitcast, OpBroadcast, OpSqrt, OpAnd, OpGtW, OpAndNot, OpOr, OpAddSub, OpFirstLane, OpXor, OpShl, OpShr, OpSar:
				first = nd.A
			case OpCvtF32:
				if k.Nodes[nd.A].Op != OpFold {
					first = nd.A
				}
			case OpDiv:
				first = nd.B
			case OpFma:
				first = nd.C
				if b.last[nd.C] != pos || !k.needsVector(nd.C) {
					first = nd.A
				}
			case OpDot4, OpDot4U:
				first = nd.C
			}
			if reg < 0 && first >= 0 && k.Nodes[first].Reg > 0 && k.needsVector(first) && b.last[first] == pos {
				reg = k.Nodes[first].Reg - 1
			}
			if reg < 0 {
				reg = lowest(chain[i], hs[i])
			}
			free(pos, innermost(hs[i]))
			if h := b.p.Hoist; h > 0 && reg >= h && depth[pos] == 0 && k.readInLoop(i) {
				reg = -1
				for r := range h {
					if !busy[r] && !reserved[r] {
						reg = r
						break
					}
				}
				if reg < 0 {
					return errors.New("more values held across the loops than the registers below the body's")
				}
			}
		}
		if err := give(i, reg); err != nil {
			return err
		}
		if nd.Op.byteDot() && k.needsVector(i) {
			// Two registers besides its own and its operands': the
			// signed operand and the pairs' sums, then the ones.
			var held []int
			for _, a := range k.Operands(&nd) {
				if r := k.Nodes[*a].Reg - 1; r >= 0 && k.needsVector(*a) && !busy[r] {
					busy[r] = true
					held = append(held, r)
				}
			}
			t := lowest(0, hs[i])
			if t >= 0 {
				busy[t] = true
			}
			t2 := lowest(0, hs[i])
			if t >= 0 {
				busy[t] = false
			}
			for _, r := range held {
				busy[r] = false
			}
			if t < 0 || t2 < 0 {
				return errors.New("no registers for a byte dot")
			}
			k.Nodes[i].Tmp, k.Nodes[i].Tmp2 = t+1, t2+1
		}
		if nd.Op == OpSelect || nd.Op == OpZip {
			var held []int
			for _, a := range k.Operands(&nd) {
				if r := k.Nodes[*a].Reg - 1; r >= 0 && k.needsVector(*a) && !busy[r] {
					busy[r] = true
					held = append(held, r)
				}
			}
			t := lowest(0, hs[i])
			for _, r := range held {
				busy[r] = false
			}
			if t < 0 {
				return errors.New("no register for a select")
			}
			k.Nodes[i].Tmp = t + 1
		}
		if nd.Op == OpHSum || nd.Op == OpHMax || nd.Op == OpHMin {
			t := lowest(0, hs[i])
			for _, r := range b.p.Lanes {
				if !busy[r] && !reserved[r] {
					t = r
					break
				}
			}
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
	return k.copyRegs(cp, len(splats))
}

// zerosPerLoop makes a float zero read in several loops in each of them,
// where it is first read there.
func (k *Kernel) zerosPerLoop() {
	hs := k.Homes()
	for z := range k.Nodes {
		if !k.isZero(z) || !k.needsVector(z) {
			continue
		}
		made := map[int]int{}
		for p := 0; p < len(k.Seq); p++ {
			e := k.Seq[p]
			if e.Kind != EntryNode || e.I >= len(hs) || hs[e.I] < 0 {
				continue
			}
			for _, a := range k.Operands(&k.Nodes[e.I]) {
				if *a != z {
					continue
				}
				c := hs[e.I]
				if len(made) == 0 {
					made[c] = z
				}
				n, ok := made[c]
				if !ok {
					n = k.add(k.Nodes[z])
					made[c] = n
					k.Seq = append(k.Seq[:p], append([]Entry{{Kind: EntryNode, I: n}}, k.Seq[p:]...)...)
					p++
				}
				*a = n
			}
		}
	}
}

// copiesAside copies, for each value that shares a register with a later
// one and is read after the later one is made, that value where it is
// made, its readers after the later one reading the copy.
func (k *Kernel) copiesAside() {
	for {
		pos := map[int]int{}
		for p, e := range k.Seq {
			if e.Kind == EntryNode {
				pos[e.I] = p
			}
		}
		first := map[int]int{}
		done := true
		for p, e := range k.Seq {
			if e.Kind != EntryNode || k.Nodes[e.I].Share <= 0 {
				continue
			}
			l := k.Nodes[e.I].Share
			t, ok := first[l]
			first[l] = e.I
			if !ok || !k.readFrom(t, p+1) {
				continue
			}
			c := k.add(Node{Op: OpCopy, Type: k.Nodes[t].Type, A: t})
			for q := p + 1; q < len(k.Seq); q++ {
				for _, a := range k.entryArgs(k.Seq[q]) {
					if *a == t {
						*a = c
					}
				}
			}
			at := pos[t]
			k.Seq = append(k.Seq[:at+1], append([]Entry{{Kind: EntryNode, I: c}}, k.Seq[at+1:]...)...)
			done = false
			break
		}
		if done {
			return
		}
	}
}

// entryArgs is what sequence entry e reads.
func (k *Kernel) entryArgs(e Entry) []*int {
	switch e.Kind {
	case EntryNode:
		return k.Operands(&k.Nodes[e.I])
	case EntryStore:
		return k.storeArgs(&k.Stores[e.I])
	}
	return nil
}

// readFrom reports whether an entry from sequence position p on reads
// node t.
func (k *Kernel) readFrom(t, p int) bool {
	for ; p < len(k.Seq); p++ {
		for _, a := range k.entryArgs(k.Seq[p]) {
			if *a == t {
				return true
			}
		}
	}
	return false
}

// resident is the register node i, a load of a resident word of the shared
// constant block, is held in.
func (b *binder86) resident(i int) (int, bool) {
	nd := b.k.Nodes[i]
	if nd.Op != OpLoad || !b.k.Bufs[nd.Buf].Shared || nd.Share < 0 {
		return 0, false
	}
	r, ok := b.p.Resident[int(b.k.Nodes[nd.A].Imm)]
	return r, ok
}

// scalar is the register node i takes as a constant read outside every
// loop, the next of the scalar registers, or as the zero a tail keeps
// lanes with; -1 for the rest.
func (b *binder86) scalar(i int, hs []int, next *int) int {
	k := b.k
	nd := k.Nodes[i]
	if b.p.Keep > 0 && k.isZero(i) {
		for _, r := range k.Nodes {
			if r.Op == OpFirstLane && r.B == i {
				return b.p.Keep - 1
			}
		}
	}
	if *next < 0 || nd.Op != OpLoad || k.Bufs[nd.Buf].Operand.Kind != OperandConsts || nd.Type != ir.F32 || hs[i] >= 0 || nd.Hint > 0 {
		return -1
	}
	if _, ok := b.resident(i); ok {
		return -1
	}
	for j, r := range k.Nodes {
		for _, a := range k.Operands(&r) {
			if *a == i && (r.Op == OpFold || j < len(hs) && hs[j] >= 0) {
				return -1
			}
		}
	}
	r := *next
	*next++
	return r
}

// copyRegs gives each value of a pair's later head the register of its
// first head's twin plus the registers the first head takes, times the
// head; a value with a hint takes its hint, and one whose twin took the
// register of one of its operands takes its own operand's.
func (k *Kernel) copyRegs(cp []int, splats int) error {
	n := 0
	for i := range k.Nodes {
		if cp[i] == 0 && k.Nodes[i].Reg > 0 && k.needsVector(i) && k.Nodes[i].Op != OpConst && k.Nodes[i].Hint == 0 {
			n = max(n, k.Nodes[i].Reg, k.Nodes[i].Tmp)
		}
	}
	tw := k.twins(cp)
	for i := range k.Nodes {
		if cp[i] == 0 || !k.needsVector(i) {
			continue
		}
		if h := k.Nodes[i].Hint; h > 0 {
			k.Nodes[i].Reg = h
		}
	}
	for i := range k.Nodes {
		if cp[i] == 0 || !k.needsVector(i) || k.Nodes[i].Hint > 0 {
			continue
		}
		t, ok := tw[i]
		if !ok || t < 0 || k.Nodes[t].Reg == 0 {
			return errors.New("a pair's later head with no twin")
		}
		if r := k.operandReg(t, i); r > 0 {
			k.Nodes[i].Reg = r
			continue
		}
		k.Nodes[i].Reg = k.Nodes[t].Reg + cp[i]*n
		if k.Nodes[t].Tmp > 0 {
			k.Nodes[i].Tmp = k.Nodes[t].Tmp + cp[i]*n
		}
		if k.Nodes[i].Reg > 16-splats || k.Nodes[i].Tmp > 16-splats {
			return errors.New("a pair's heads take more vector registers than there are")
		}
	}
	return nil
}

// operandReg is the register of node i's operand in the place where twin
// t's operand holds t's own register, if that operand is hinted; 0 where
// there is none.
func (k *Kernel) operandReg(t, i int) int {
	ta, ia := k.Operands(&k.Nodes[t]), k.Operands(&k.Nodes[i])
	for x := range ta {
		if x < len(ia) && k.Nodes[*ta[x]].Reg == k.Nodes[t].Reg && k.Nodes[*ia[x]].Hint > 0 {
			return k.Nodes[*ia[x]].Hint
		}
	}
	return 0
}

// chains is, for each value in a trip of a loop adding several sums, 1 +
// the sum it is added into; 0 for the rest.
func (k *Kernel) chains() map[int]int {
	out := map[int]int{}
	hs := k.Homes()
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
				for _, a := range k.Operands(&nd) {
					walk(*a)
				}
			}
			walk(k.Nodes[f].A)
		}
	}
	return out
}

// rematerialize loads a constant where each of its readers in the loops
// reads it, rather than once before the loops: the constant read first in
// its loops' trips. False where none is left to move.
func (k *Kernel) rematerialize(moved map[int]bool) bool {
	pos := map[int]int{}
	for p, e := range k.Seq {
		if e.Kind == EntryNode {
			pos[e.I] = p
		}
	}
	hs := k.Homes()
	readers := func(i int) []int {
		var out []int
		for j, x := range k.Nodes {
			if _, in := pos[j]; !in {
				continue
			}
			for _, a := range k.Operands(&x) {
				if *a == i {
					out = append(out, j)
				}
			}
		}
		return out
	}
	best, at := -1, -1
	for i, nd := range k.Nodes {
		p, in := pos[i]
		if o := k.Bufs[nd.Buf].Operand.Kind; !in || moved[i] || nd.Op != OpLoad || o != OperandConsts && k.Bufs[nd.Buf].Words == nil || hs[i] >= 0 {
			continue
		}
		rs := readers(i)
		ok := len(rs) > 0
		first := -1
		for _, r := range rs {
			if hs[r] < 0 {
				ok = false
			}
			if rp := pos[r]; first < 0 || rp < first {
				first = rp
			}
		}
		if ok && first > p && (at < 0 || first > at) {
			best, at = i, first
		}
	}
	if best < 0 {
		return false
	}
	k.reload(best, moved)
	return true
}

// wholeVector reports whether load i reads a word of fixed words that the
// words after it repeat for a vector's lanes, so the vector there is the
// word in every lane.
func (k *Kernel) wholeVector(i int) bool {
	nd := k.Nodes[i]
	ws := k.Bufs[nd.Buf].Words
	ix := k.Nodes[nd.A]
	if nd.Op != OpLoad || ws == nil || ix.Op != OpConst || nd.Ptr != 0 {
		return false
	}
	w := int(ix.Imm)
	if w+8 > len(ws) {
		return false
	}
	for j := w; j < w+8; j++ {
		if ws[j] != ws[w] {
			return false
		}
	}
	return true
}

// reload loads constant best where each of its readers reads it, as an
// integer and's or add's memory operand where the table holds a vector of
// it there.
func (k *Kernel) reload(best int, moved map[int]bool) {
	pos := map[int]int{}
	for p, e := range k.Seq {
		if e.Kind == EntryNode {
			pos[e.I] = p
		}
	}
	var rs []int
	for j, x := range k.Nodes {
		if _, in := pos[j]; !in {
			continue
		}
		for _, a := range k.Operands(&x) {
			if *a == best {
				rs = append(rs, j)
				break
			}
		}
	}
	moved[best] = true
	place := map[int]int{}
	words := k.VectorWords()
	for _, r := range rs {
		c := k.add(k.Nodes[best])
		moved[c] = true
		for _, a := range k.Operands(&k.Nodes[r]) {
			if *a == best {
				*a = c
			}
		}
		if x := k.Nodes[r]; (x.Op == OpAnd || x.Op == OpAdd) && x.B == c && words[r] && k.wholeVector(c) {
			k.Nodes[c].Mem = true
			continue
		}
		place[pos[r]] = c
	}
	var out []Entry
	for p, e := range k.Seq {
		if e.Kind == EntryNode && e.I == best {
			continue
		}
		if c, ok := place[p]; ok {
			out = append(out, Entry{Kind: EntryNode, I: c})
		}
		out = append(out, e)
	}
	k.Seq = out
}

// readInLoop reports whether a node homed in a loop reads node i.
func (k *Kernel) readInLoop(i int) bool {
	hs := k.Homes()
	for j := range k.Nodes {
		if j >= len(hs) || hs[j] < 0 {
			continue
		}
		for _, a := range k.Operands(&k.Nodes[j]) {
			if *a == i {
				return true
			}
		}
	}
	return false
}

// madeCount is the loop node i is the made count of, as madeCounts finds
// them, -1 for none.
func (k *Kernel) madeCount(i int) int {
	nd := k.Nodes[i]
	if nd.Op == OpParam || nd.Op == OpConst || k.Nodes[nd.A].Op != OpParam {
		return -1
	}
	if !slices.Contains(k.madeCounts(nd.A), i) {
		return -1
	}
	return k.countedBy(i)
}

// lateWords loads each address word only accesses after every loop read
// where the first of them is, in a kernel with no tile pass.
func (k *Kernel) lateWords() {
	last := -1
	for p, e := range k.Seq {
		switch {
		case e.Kind == EntryLoop && k.Loops[e.I].Walk:
			return
		case e.Kind == EntryEnd:
			last = p
		}
	}
	if last < 0 {
		return
	}
	first := func(i int) int {
		base := func(b int) bool { return k.Bufs[b].Base == i+1 }
		for p, e := range k.Seq {
			switch e.Kind {
			case EntryNode:
				nd := k.Nodes[e.I]
				if nd.Op == OpLoad && base(nd.Buf) {
					return p
				}
				for _, a := range k.Operands(&nd) {
					if *a == i {
						return p
					}
				}
			case EntryStore:
				if base(k.Stores[e.I].Buf) {
					return p
				}
			case EntryLoop:
				for _, cu := range k.Loops[e.I].Cursors {
					if base(cu.Buf) {
						return p
					}
				}
			}
		}
		return -1
	}
	for _, e := range slices.Clone(k.Seq) {
		if e.Kind != EntryNode || k.Nodes[e.I].Op != OpParam || !k.isBaseNode(e.I) {
			continue
		}
		at := first(e.I)
		if at <= last {
			continue
		}
		from := slices.Index(k.Seq, e)
		k.Seq = slices.Delete(k.Seq, from, from+1)
		at = first(e.I)
		k.Seq = slices.Insert(k.Seq, at, e)
	}
}

// lateParams is the address words the sequence loads after every loop.
func (k *Kernel) lateParams() map[int]bool {
	last := -1
	for p, e := range k.Seq {
		if e.Kind == EntryEnd {
			last = p
		}
	}
	late := map[int]bool{}
	for p, e := range k.Seq {
		if last >= 0 && p > last && e.Kind == EntryNode && k.Nodes[e.I].Op == OpParam && k.isBaseNode(e.I) {
			late[e.I] = true
		}
	}
	return late
}

// residents loads the shared constant block's resident words by word as
// the call starts, after its argument words, and each other word, and a
// resident word read outside every loop, where it is read.
func (k *Kernel) residents(res map[int]int, moved map[int]bool) {
	var loads []int
	for _, e := range k.Seq {
		if e.Kind != EntryNode {
			continue
		}
		nd := k.Nodes[e.I]
		if nd.Op != OpLoad || !k.Bufs[nd.Buf].Shared {
			continue
		}
		if _, ok := res[int(k.Nodes[nd.A].Imm)]; ok {
			loads = append(loads, e.I)
			moved[e.I] = true
		}
	}
	slices.SortStableFunc(loads, func(a, b int) int {
		return cmp.Compare(k.Nodes[k.Nodes[a].A].Imm, k.Nodes[k.Nodes[b].A].Imm)
	})
	if len(loads) > 0 {
		var seq []Entry
		start := 0
		for _, e := range k.Seq {
			if e.Kind == EntryNode && moved[e.I] {
				continue
			}
			seq = append(seq, e)
			if e.Kind == EntryNode && k.Nodes[e.I].Op == OpParam && start == len(seq)-1 {
				start = len(seq)
			}
		}
		head := make([]Entry, 0, len(loads))
		for _, l := range loads {
			head = append(head, Entry{Kind: EntryNode, I: l})
		}
		k.Seq = append(append(append([]Entry(nil), seq[:start]...), head...), seq[start:]...)
	}
	for _, e := range slices.Clone(k.Seq) {
		if e.Kind != EntryNode {
			continue
		}
		nd := k.Nodes[e.I]
		if nd.Op == OpLoad && k.Bufs[nd.Buf].Shared && !moved[e.I] {
			k.reload(e.I, moved)
		}
	}
	hs := k.Homes()
	for _, l := range loads {
		var outside []int
		for j, x := range k.Nodes {
			for _, a := range k.Operands(&x) {
				if *a == l && j < len(hs) && hs[j] < 0 && x.Op != OpFold {
					outside = append(outside, j)
					break
				}
			}
		}
		if len(outside) == 0 {
			continue
		}
		c := k.add(k.Nodes[l])
		k.Nodes[c].Share = -1
		first := len(k.Seq)
		for p, e := range k.Seq {
			if e.Kind == EntryNode && slices.Contains(outside, e.I) && p < first {
				first = p
			}
		}
		for _, j := range outside {
			for _, a := range k.Operands(&k.Nodes[j]) {
				if *a == l {
					*a = c
				}
			}
		}
		k.Seq = append(k.Seq[:first], append([]Entry{{Kind: EntryNode, I: c}}, k.Seq[first:]...)...)
	}
}

// zeroOperand reports whether float zero node i is an operand the code
// computes with, not only a sum's start.
func (k *Kernel) zeroOperand(i int) bool {
	for j, nd := range k.Nodes {
		if nd.Op == OpFold {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if *a == i && j != i {
				return true
			}
		}
	}
	return false
}

// passes reports whether the kernel walks an address in several passes:
// copies of one buffer's address taken by more than one loop.
func (k *Kernel) passes() bool {
	taken := map[Operand]int{}
	for _, l := range k.Loops {
		for _, cu := range l.Cursors {
			if cu.Take {
				taken[k.Bufs[cu.Buf].Operand]++
			}
		}
	}
	for _, n := range taken {
		if n > 1 {
			return true
		}
	}
	return false
}
