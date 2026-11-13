package composer

import (
	"cmp"
	"errors"
	"fmt"
	"maps"
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/format/jlm"
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

// spillSlots is the 8-byte slots below the stack pointer a kernel spills
// into (spillOffset): the red zone's sixteen but the last, which holds a
// vector shift's count (lowercpu.go, shiftCount), and sixteen more below the
// vector temporaries.
const spillSlots = 31

// spillOffset is slot s's displacement from the stack pointer. A kernel runs
// on its caller's goroutine stack below the NOSPLIT trampoline's frame
// (engine/device/cpu, callKernel), where the stack guard the caller checked leaves
// 800 bytes to nosplit frames; the slots past the red zone's sit below the
// lowerings' vector temporaries (onesSlot, sseSlot, a64Save) and end 440
// bytes down.
func spillOffset(s int) int32 {
	if s <= 15 {
		return int32(-8 * s)
	}
	return int32(-320 - 8*(s-16))
}

// KernelAt is a kernel of part Of by the weight it reads first, where it
// reads one.
type KernelAt struct {
	Of     int
	Role   jlm.Role
	Weight bool
}

func (k *Kernel) at() KernelAt {
	for _, b := range k.Bufs {
		if b.Operand.Kind == OperandWeight {
			return KernelAt{k.Of, b.Operand.Role, true}
		}
	}
	return KernelAt{Of: k.Of}
}

// inited reports whether a cursor of k takes its copy at an initializer.
func (k *Kernel) inited() bool {
	for _, l := range k.Loops {
		for _, cu := range l.Cursors {
			if cu.Init > 0 {
				return true
			}
		}
	}
	return false
}

// ErrRegisterPressure is a kernel Regs refuses for holding more values live
// at once than the device's registers.
const ErrRegisterPressure registerError = "register pressure"

type registerError string

func (e registerError) Error() string { return string(e) }

// PressureError is Regs' refusal of a run's kernels for register pressure:
// the parts they are of (Kernel.Of), and the first kernel's reason.
type PressureError struct {
	Parts []int
	// Jammed is the parts of Parts a refused kernel ran several rows a
	// call in (RowJam), Paired those it ran two heads a call in
	// (HeadPairs), Cursored those whose loops walked its addresses
	// (Cursors).
	Jammed, Paired, Cursored []int
	// Wide is the refused kernels that jammed a loop's rows across lanes
	// (Jam).
	Wide []KernelAt
	// Split is the parts a refused kernel cut its byte dots' runs in
	// (HalfSplit).
	Split []int
	// Inited is the refused kernels whose cursors took a copy at an
	// initializer (Cursor.Init).
	Inited []KernelAt
	// Streamed is the refused kernels that walked along the rows a
	// sub-block at a time (Streams), Crowded those of them whose walk held
	// every plane in one rows loop.
	Streamed, Crowded []KernelAt
	// Unrolled is the refused kernels whose vector loops ran several
	// vectors a trip (Unroll).
	Unrolled []KernelAt
	Err      error
}

func (e *PressureError) Error() string { return e.Err.Error() }

func (e *PressureError) Unwrap() error { return e.Err }

func (Regs) Name() string { return "regs" }

func (p Regs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	var refused *PressureError
	out, err := eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		ns, changed, err := p.bind(k, d)
		if errors.Is(err, ErrRegisterPressure) {
			if refused == nil {
				refused = &PressureError{Err: err}
			}
			refused.Parts = append(refused.Parts, k.Of)
			if k.Jam > 1 {
				refused.Jammed = append(refused.Jammed, k.Of)
			}
			if k.paired() {
				refused.Paired = append(refused.Paired, k.Of)
			}
			if k.wide() {
				refused.Wide = append(refused.Wide, k.at())
			}
			if k.streamed() && !k.Aparted && !k.Crowded {
				refused.Crowded = append(refused.Crowded, k.at())
			} else if k.streamed() {
				refused.Streamed = append(refused.Streamed, k.at())
			}
			if k.cursored() {
				refused.Cursored = append(refused.Cursored, k.Of)
			}
			if k.Split {
				refused.Split = append(refused.Split, k.Of)
			}
			if k.inited() {
				refused.Inited = append(refused.Inited, k.at())
			}
			if k.unrolled() {
				refused.Unrolled = append(refused.Unrolled, k.at())
			}
			return nil, false, nil
		}
		return ns, changed, err
	})
	if err == nil && refused != nil {
		return nil, refused
	}
	return out, err
}

// bind is k bound to the registers, or the reason it cannot be.
func (p Regs) bind(k *Kernel, d Device) ([]*Kernel, bool, error) {
	if k.Seq == nil {
		return nil, false, nil
	}
	w := k.clone()
	w.BitsGPR = w.Unvector && d.ISA&uint64(x86.ISASSE2) != 0
	moved := map[int]bool{}
	if p.Resident != nil {
		w.residents(p.Resident, moved)
	}
	late := w.lateParams()
	// try binds st's kernel as it stands: the bound kernel, or why the
	// floats do not fit; an error past them is the words not fitting.
	try := func(st *bindState) (*Kernel, error, error) {
		m := st.w.clone()
		m.vwords = st.memoCopy()
		b := binder86{k: m, p: p, used: map[int]bool{}, late: late, isa: x86.ISA(d.ISA)}
		b.lives()
		if b.gprs() != nil {
			// The words do not fit: again with the reload registers kept
			// out of every pool, spilling where none is free.
			m = st.w.clone()
			m.vwords = st.memoCopy()
			b = binder86{k: m, p: p, used: map[int]bool{}, late: late, spill: true, isa: x86.ISA(d.ISA)}
			b.lives()
			if err := b.gprs(); err != nil {
				return nil, nil, fmt.Errorf("%s: the words do not fit the general registers: %w: %w", k.Name, ErrRegisterPressure, err)
			}
		}
		why := b.vectors()
		if why != nil && m.Ones > 0 {
			// Again with the byte dots making their ones each time.
			b.noOnes = true
			why = b.vectors()
		}
		if why != nil {
			return nil, why, nil
		}
		m.vwords = nil
		return m, nil, nil
	}
	// A failed binding rematerializes a load or sinks a value, one a step,
	// which binding reads nothing of: the steps are the kernel's alone, and
	// the binding is the first step count that binds, at most one a node.
	// Binding is not monotone in the steps -- a sink after the count that
	// binds can make the next counts fail again -- so the counts are walked
	// one by one up to walkSteps. Past them, a kernel of hundreds of steps
	// gallops: the steps between tries double from the last count that
	// failed, then halve back from the first that bound, a binding in a score
	// of tries where the walk took hundreds; it takes a count that binds,
	// the first one where binding is monotone from there. Where galloping
	// finds none, the walk goes on from walkSteps, so a kernel the walk binds
	// is never refused. A failure no step left can lower (hopeless) ends
	// the walk where it stands.
	end := len(k.Nodes) - 1
	cur := &bindState{w: w, moved: moved, sunk: map[int]bool{}}
	var why error
	for {
		m, w2, err := try(cur)
		switch {
		case err != nil:
			return nil, false, err
		case m != nil:
			return []*Kernel{m}, true, nil
		}
		why = w2
		if cur.steps == end || cur.hopeless(why) || cur.advance(1) == 0 {
			return nil, false, bindFailed(k.Name, why)
		}
		if cur.steps == walkSteps {
			break
		}
	}
	start := cur.clone()
	m, w2, err := try(cur)
	switch {
	case err != nil:
		return nil, false, err
	case m != nil:
		return []*Kernel{m}, true, nil
	case cur.hopeless(w2):
		return nil, false, bindFailed(k.Name, w2)
	}
	failed, why := cur.clone(), w2
	for gap := 1; cur.steps < end; gap *= 2 {
		if cur.advance(min(gap, end-cur.steps)) == 0 {
			break
		}
		bound, w2, err := try(cur)
		if err != nil {
			return nil, false, err
		}
		if bound == nil {
			if cur.hopeless(w2) {
				return nil, false, bindFailed(k.Name, w2)
			}
			failed, why = cur.clone(), w2
			continue
		}
		m = bound
		for cur.steps-failed.steps > 1 {
			mid := failed.clone()
			mid.advance((cur.steps - failed.steps) / 2)
			bound, w2, err := try(mid)
			switch {
			case err != nil:
				return nil, false, err
			case bound == nil:
				failed, why = mid, w2
			default:
				cur, m = mid, bound
			}
		}
		return []*Kernel{m}, true, nil
	}
	for cur = start; cur.steps < end && cur.advance(1) == 1; {
		m, w2, err := try(cur)
		switch {
		case err != nil:
			return nil, false, err
		case m != nil:
			return []*Kernel{m}, true, nil
		}
		why = w2
		if cur.hopeless(why) {
			break
		}
	}
	return nil, false, bindFailed(k.Name, why)
}

// walkSteps is how many of Regs' step counts are tried one by one before a
// binding gallops.
const walkSteps = 64

// bindFailed is a kernel's floats not fitting the vector registers, why the
// last binding tried failed.
func bindFailed(name string, why error) error {
	if why == nil {
		why = errors.New("no binding within a rematerialization per node")
	}
	return fmt.Errorf("%s: the floats do not fit the vector registers: %w: %w", name, ErrRegisterPressure, why)
}

// pressureAt is a binding's floats not fitting the vector registers at
// one node: the node, its place in the sequence, and the binding.
type pressureAt struct {
	node, pos int
	op        Op
	in        *binder86
}

func (e *pressureAt) Error() string {
	return fmt.Sprintf("more floats live at once than the vector registers, at node %d (%s)", e.node, e.op)
}

// hopeless reports whether why, a binding of st's kernel failing for its
// floats, is one no steps left can bind (lowers).
func (st *bindState) hopeless(why error) bool {
	var pa *pressureAt
	return errors.As(why, &pa) && !st.lowers(pa.in, pa.pos)
}

// lowers reports whether a step left to take could hold fewer floats live
// across sequence position pos, where binding b found more than the vector
// registers: a constant loaded before the loops that b holds across pos, or
// a vector value a sink could move past pos -- made in the straight run pos
// is in, before it, read by a node of the run after it and by no store.
// Every other step shortens no live range crossing pos, and a step's other
// effect is to lengthen one, so where none is left no count of steps binds.
func (st *bindState) lowers(b *binder86, pos int) bool {
	k := b.k
	hs := k.homesNow()
	for i, nd := range k.Nodes {
		d, ok := b.def.get(i)
		if !ok || st.moved[i] || nd.Op != OpLoad || hs[i] >= 0 {
			continue
		}
		if o := k.Bufs[nd.Buf].Operand.Kind; o != OperandConsts && k.Bufs[nd.Buf].Words == nil {
			continue
		}
		if d < pos && b.last.of(i) > pos {
			return true
		}
	}
	for _, c := range k.copies() {
		if c > 0 {
			return false
		}
	}
	// A sink moves a candidate of its own filter, with its group, to just
	// before the candidate's one reader; it holds fewer floats across pos
	// where a vector value of the group is made before pos and the reader
	// comes after it. Sinks move values later and never earlier, so a value
	// a later sink carries across pos is one a candidate's group carries
	// now.
	var sc stepScratch
	at := sc.places(k)
	off, rd := sc.readers(k, at, true)
	words := k.vectorWords()
	for p, e := range k.Seq {
		x := e.I
		if e.Kind != EntryNode || off[x+1]-off[x] != 1 || rd[off[x]] < 0 || !k.sinkMovable(x) || st.sunk[x] || !k.needsVectorOf(x, words) {
			continue
		}
		to := at[rd[off[x]]]
		if to <= pos || to <= p+1 || !k.straightSeq(p, to) {
			continue
		}
		for g := range k.sinkGroup(x, to, at, off, rd) {
			if at[g] < pos && k.needsVectorOf(g, words) {
				return true
			}
		}
	}
	return false
}

// bindState is a kernel some steps of Regs' rematerializations and sinks on,
// with what those took.
type bindState struct {
	w           *Kernel
	moved, sunk map[int]bool
	steps       int
	// memo is w's vector words and homes (Kernel.vwords), kept across
	// advances: each is found again only where the count of nodes it was
	// found at changed, and a binding that walks its steps one by one
	// would otherwise find them again at every step.
	memo *vectorMemo
	// sc is what each step builds and the next builds again over.
	sc stepScratch
}

// stepScratch is what a binding's steps build afresh at every step and keep
// between them: each node's place in the sequence, its readers, and sink's
// prefix counts. A step of hundreds of nodes made a slice a node otherwise,
// and the collector's marking of them came to a third of composing.
type stepScratch struct {
	pos, off, rd, fill, vecs, others []int
	needs                            []bool
}

func sized[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}

// places is each node's place in k's sequence, -1 for none.
func (sc *stepScratch) places(k *Kernel) []int {
	pos := sized(sc.pos, len(k.Nodes))
	sc.pos = pos
	for i := range pos {
		pos[i] = -1
	}
	for p, e := range k.Seq {
		if e.Kind == EntryNode {
			pos[e.I] = p
		}
	}
	return pos
}

// readers lays out the readers in the sequence of each node placed there:
// node a's are rd[off[a]:off[a+1]], in node order, a reader once however
// many of its operands read a; with stores, then a -1 for each store
// operand reading a.
func (sc *stepScratch) readers(k *Kernel, pos []int, stores bool) (off, rd []int) {
	n := len(k.Nodes)
	off = sized(sc.off, n+1)
	clear(off)
	var buf [3]int
	each := func(f func(a, j int)) {
		for j := range k.Nodes {
			if pos[j] < 0 {
				continue
			}
			as := k.args(j, &buf)
			for t, a := range as {
				if a < 0 || a >= n || pos[a] < 0 || slices.Contains(as[:t], a) {
					continue
				}
				f(a, j)
			}
		}
		if stores {
			for i := range k.Stores {
				for _, a := range k.storeArgs(&k.Stores[i]) {
					if *a >= 0 && *a < n {
						f(*a, -1)
					}
				}
			}
		}
	}
	each(func(a, _ int) { off[a+1]++ })
	for a := range n {
		off[a+1] += off[a]
	}
	rd, fill := sized(sc.rd, off[n]), sized(sc.fill, n)
	copy(fill, off[:n])
	each(func(a, j int) {
		rd[fill[a]] = j
		fill[a]++
	})
	sc.off, sc.rd, sc.fill = off, rd, fill
	return off, rd
}

// advance takes up to n steps more, the rematerializations first: the steps
// taken, fewer where none is left. Its steps keep the kernel's vector words
// (Kernel.vwords) while it holds as many nodes: a rematerialization adds a
// node, and a sink moves one in the sequence, which neither the words nor
// the homes read.
func (st *bindState) advance(n int) int {
	if st.memo == nil {
		st.memo = &vectorMemo{n: -1}
	}
	st.w.vwords = st.memo
	defer func() { st.w.vwords = nil }()
	for i := range n {
		if !st.w.rematerialize(st.moved, &st.sc) && !st.w.sink(st.sunk, &st.sc) {
			return i
		}
		st.steps++
	}
	return n
}

// memoCopy is a copy of st's memo for a binding of a clone of its kernel,
// which holds the same nodes: each of its findings is kept while the clone
// holds as many nodes as it was found at, as the state's own are. The copy's
// findings are replaced, never written in place, so the state's stand.
func (st *bindState) memoCopy() *vectorMemo {
	if st.memo == nil {
		return &vectorMemo{n: -1}
	}
	return st.memo.fresh()
}

func (st *bindState) clone() *bindState {
	c := &bindState{w: st.w.clone(), moved: maps.Clone(st.moved), sunk: maps.Clone(st.sunk), steps: st.steps}
	if st.memo != nil {
		c.memo = st.memo.fresh()
	}
	return c
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
	// def and last are each value's place in the sequence and its last
	// read's, end and start each loop's end and beginning (lives).
	def, last  lifeTab
	end, start lifeTab
	// useN is each node's reads (uses), counted at useKey's counts of
	// nodes and stores.
	useKey [2]int
	useN   []int
	spans  map[int][]span
	// reserved is the registers a spilling binding keeps for reloads and
	// counters.
	reserved map[int]bool
}

// span is a register's hold from one point to another, inclusive: point 2p
// is sequence position p's reads and writes, 2p+1 what a loop beginning at p
// writes after reading its count there.
type span struct{ from, to int }

func (s span) meets(o span) bool { return s.from <= o.to && o.from <= s.to }

func (b *binder86) hold(r int, s span) {
	if b.spans == nil {
		b.spans = map[int][]span{}
	}
	b.spans[r] = append(b.spans[r], s)
}

// fits reports whether register r is held at no position of s.
func (b *binder86) fits(r int, s span) bool {
	for _, o := range b.spans[r] {
		if o.meets(s) {
			return false
		}
	}
	return true
}

// takeOver is the first register of pool free over s, held there; -1 for
// none.
func (b *binder86) takeOver(pool []int, s span) int {
	for _, r := range pool {
		if !b.reserved[r] && b.fits(r, s) {
			b.hold(r, s)
			return r
		}
	}
	return -1
}

// loopSpan is the points loop c's counter is held over in register r: from
// its start, or its vector loop's where that loop sets it (x86Lowerer.enter:
// a tail counting in neither the vector loop's register nor its count's), to
// its end.
func (b *binder86) loopSpan(c, r int) span {
	k := b.k
	s := span{2*b.start.of(c) + 1, 2 * b.end.of(c)}
	l := k.Loops[c]
	// The counter is set before the loop's takes read their initializers
	// (x86Lowerer.take): it is held from the loop's start, where they are
	// last read.
	for _, cu := range l.Cursors {
		if cu.Init > 0 {
			s.from = 2 * b.start.of(c)
		}
	}
	if v := c - 1; v >= 0 && l.Of == c && k.Loops[v].Lanes > 0 && l.Count > 0 && k.Nodes[l.Count-1].Op != OpConst && k.passes() &&
		r != k.Loops[v].Reg-1 && r != k.Nodes[l.Count-1].Reg-1 {
		if d, ok := b.def.get(l.Count - 1); ok && d < b.start.of(v) {
			s.from = 2*b.start.of(v) + 1
		}
	}
	return s
}

// ptrSpan is the points copy ptr is held over: every loop that takes or
// steps it.
func (b *binder86) ptrSpan(ptr int) span {
	s := span{-1, -1}
	for c, l := range b.k.Loops {
		for _, cu := range l.Cursors {
			if cu.Ptr != ptr {
				continue
			}
			// A take that adds an initializer reads it as the copy is
			// written: the copy is held from the loop's start, where the
			// initializer is last read, so the two take different registers.
			from := 2*b.start.of(c) + 1
			if cu.Init > 0 {
				from--
			}
			if s.from < 0 || from < s.from {
				s.from = from
			}
			s.to = max(s.to, 2*b.end.of(c))
		}
	}
	return s
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
	hs := k.homesNow()
	b.def.reset(len(k.Nodes))
	b.last.reset(len(k.Nodes))
	b.end.reset(len(k.Loops))
	b.start.reset(len(k.Loops))
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryNode:
			b.def.put(e.I, pos)
		case EntryEnd:
			b.end.put(e.I, pos)
		case EntryLoop:
			b.start.put(e.I, pos)
		}
	}
	use := func(v, pos, home int) {
		d, ok := b.def.get(v)
		if !ok {
			return
		}
		h := hs[v]
		if k.Nodes[v].Op == OpFold {
			h = k.Nodes[v].At
		}
		for c := home; c >= 0; c = k.Loops[c].Parent {
			outside := d < b.start.of(c) || d > b.end.of(c)
			if outside || k.Nodes[v].Op == OpFold && h == c {
				if e, ok := b.end.get(c); ok && e > pos {
					pos = e
				}
			}
		}
		b.last.put(v, max(b.last.of(v), pos))
	}
	// A node runs in the loops its place in the sequence opens, which may
	// be deeper than its home: a reader of a loop's counter (Induction)
	// whose other operands are outside it.
	var open []int
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryEnd:
			if len(open) > 0 {
				open = open[:len(open)-1]
			}
		case EntryNode:
			nd := k.Nodes[e.I]
			if nd.Op == OpFold {
				use(nd.B, pos, k.Loops[nd.At].Parent)
				continue
			}
			at := hs[e.I]
			if len(open) > 0 {
				at = open[len(open)-1]
			}
			var buf, xbuf [3]int
			for _, a := range k.args(e.I, &buf) {
				use(a, pos, at)
				// A load its reader takes as a memory operand reads its
				// index where the reader runs, not where the load sits.
				if x := &k.Nodes[a]; x.Op == OpLoad && x.Mem {
					for _, ix := range k.args(a, &xbuf) {
						use(ix, pos, at)
					}
				}
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
			for _, cu := range l.Cursors {
				if cu.Init > 0 {
					use(cu.Init-1, pos, l.Parent)
				}
			}
			open = append(open, e.I)
		}
	}
}

// gprs binds argument words, counts, loop counters and cursor copies.
func (b *binder86) gprs() error {
	k, p := b.k, b.p
	if b.spill {
		// The reload registers, and the counters, which a spilling
		// binding keeps for its loops rather than its values: taken first,
		// so a register held back for a deeper loop is neither.
		b.reserved = map[int]bool{}
		for _, r := range p.Reloads {
			b.used[r] = true
			b.reserved[r] = true
		}
		for _, r := range p.Counters {
			b.used[r] = true
			b.reserved[r] = true
		}
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
				b.reserved[r] = true
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
	words := k.vectorWords()
	// A value made in a general register frees it after its last read; an
	// argument word, a length, a count and a buffer's address after their
	// last read in that role (keepEnd).
	type kept struct{ r, end int }
	freed, expire := map[int]int{}, map[int]kept{}
	expired := func(at int) {
		for v, h := range expire {
			if k.Nodes[v].Spill != 0 {
				// Re-read or spilled: its register was handed on.
				delete(expire, v)
				continue
			}
			if h.end < at {
				b.used[h.r] = false
				delete(expire, v)
				delete(b.params, v)
			}
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
				for v, r := range freed {
					if b.last.of(v) < pos {
						b.used[r] = false
						delete(freed, v)
					}
				}
				expired(pos)
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
		if nd.Op != OpParam && (nd.Type == ir.F32 || nd.Op == OpConst || nd.Op == OpLoad && (nd.Mem || k.needsVector(i)) || words[i]) {
			continue
		}
		for v, r := range freed {
			if b.last.of(v) < b.def.of(i) {
				b.used[r] = false
				delete(freed, v)
			}
		}
		expired(b.def.of(i))
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
			if k.Loops[c].Lanes > 0 && b.last.of(nd.A) == b.def.of(i) && k.Nodes[nd.A].Reg == p.Counters[0]+1 {
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
				return b.last.of(x) == b.def.of(i) && k.Nodes[x].Reg > 0 && k.Nodes[x].Spill == 0 && k.Nodes[x].Type != ir.F32 && !keeps(x) && k.Nodes[x].Op != OpIter
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
			if s := b.slot(b.def.of(i), i+1); s > 0 {
				k.Nodes[i].Reg, k.Nodes[i].Spill = int(reloadReg)+1, s
				continue
			}
		}
		if r < 0 {
			return errors.New("more words live at once than registers")
		}
		k.Nodes[i].Reg = r + 1
		for v, h := range expire {
			if h.r == r {
				delete(expire, v)
			}
		}
		for v, x := range b.params {
			if x == r {
				delete(b.params, v)
			}
		}
		if b.spill && nd.Op == OpParam && !k.isBaseNode(i) && !b.countsIn(r) {
			if b.params == nil {
				b.params = map[int]int{}
			}
			b.params[i] = r
		}
		for j := range k.Bufs {
			if k.Bufs[j].Base == i+1 {
				k.Bufs[j].Reg = r + 1
			}
		}
		if !keeps(i) {
			freed[i] = r
		} else if e := b.keepEnd(i); e < len(k.Seq) {
			expire[i] = kept{r, e}
		}
	}
	// The counters and copies bound below take a register only where no
	// value, iteration, copy or counter holds it.
	b.valueSpans(keeps)
	for c := range k.Loops {
		l := &k.Loops[c]
		if l.Kept && l.Reg == 0 {
			if r := b.takeCounter([]int{p.Outer}, c); r >= 0 {
				l.Reg = r + 1
			}
		}
		if !l.Walk || l.Reg != 0 || l.once() {
			continue
		}
		if r := b.takeCounter([]int{p.Counters[0], p.Counters[1]}, c); r >= 0 {
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
	groups := k.offsetGroups(lastLoop)
	var reloads map[int]bool
	if b.spill && !k.spills() {
		reloads = k.reloadCopies(groups, lastLoop, len(p.Reloads))
	}
	reload := func(sp span) int {
		for _, s := range p.Reloads {
			if b.fits(s, sp) {
				b.hold(s, sp)
				return s
			}
		}
		return -1
	}
	// A copy's slot, once its last loop ends, holds the next copy's.
	var freeSlots []int
	for _, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			members := map[int]int{}
			for _, cu := range k.Loops[e.I].Cursors {
				if g, ok := groups[cu.Ptr]; ok && cu.Take && b.offBase(cu) {
					members[g.g]++
				}
			}
			offs := map[int]int{}
			hull := map[int]span{}
			for _, cu := range k.Loops[e.I].Cursors {
				if g, ok := groups[cu.Ptr]; ok && cu.Take && b.offBase(cu) {
					s := b.ptrSpan(cu.Ptr)
					if h, ok := hull[g.g]; ok {
						s = span{min(s.from, h.from), max(s.to, h.to)}
					}
					hull[g.g] = s
				}
			}
			for _, cu := range k.Loops[e.I].Cursors {
				if cu.Take && k.Ptrs[cu.Ptr-1].Reg == 0 && k.Ptrs[cu.Ptr-1].Spill == 0 && k.Ptrs[cu.Ptr-1].Off == 0 {
					if og, ok := groups[cu.Ptr]; ok && members[og.g] > 1 && b.offBase(cu) {
						g := og.g
						if offs[g] == 0 {
							r := b.takeOver(p.Walks, hull[g])
							if r < 0 {
								r = b.takeOver(p.Spare, hull[g])
							}
							if r < 0 {
								r = b.takeOver([]int{p.Outer}, hull[g])
							}
							if r < 0 && reloads[cu.Ptr] {
								r = reload(hull[g])
							}
							offs[g] = r + 1
						}
						if offs[g] > 0 {
							k.Ptrs[cu.Ptr-1].Off, k.Ptrs[cu.Ptr-1].From, k.Ptrs[cu.Ptr-1].Scale = offs[g], cu.From, og.scale
							continue
						}
					}
					r := -1
					sp := b.ptrSpan(cu.Ptr)
					if passes {
						r = b.takeOver(p.Walks, sp)
					}
					if r < 0 {
						r = b.takeOver(p.Spare, sp)
					}
					if r < 0 {
						r = b.takeOver([]int{p.Outer}, sp)
					}
					if r < 0 && reloads[cu.Ptr] {
						r = reload(sp)
					}
					if r < 0 && b.spill {
						s := -1
						if n := len(freeSlots); n > 0 {
							s, freeSlots = freeSlots[n-1], freeSlots[:n-1]
						} else {
							s = b.slot(0, -1)
						}
						if s > 0 {
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
				if c != e.I {
					continue
				}
				if sl := k.Ptrs[ptr-1].Spill; sl > 0 {
					freeSlots = append(freeSlots, sl)
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
				if d < 0 {
					continue
				}
				for _, cu := range k.Loops[d].Cursors {
					if cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Reg > 0 {
						copies[k.Ptrs[cu.Ptr-1].Reg-1] = true
					}
					if cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Off > 0 {
						copies[k.Ptrs[cu.Ptr-1].Off-1] = true
					}
				}
			}
			for _, r := range append([]int{p.Counters[1]}, p.Walks...) {
				if !held[r] && !copies[r] && (r == p.Counters[1] || !b.reserved[r]) && b.fits(r, b.loopSpan(c, r)) {
					l.Reg = r + 1
					b.hold(r, b.loopSpan(c, r))
					break
				}
			}
			if l.Reg != 0 {
				continue
			}
		}
		// The counters first, then any register no value holds while the
		// loop runs: a third nested loop counts in one of those.
		for i, r := range slices.Concat([]int{p.Counters[0], p.Counters[1]}, b.counters, p.Spare, p.Kept, []int{p.Outer}) {
			if !held[r] && (i < 2+len(b.counters) || !b.reserved[r]) && b.fits(r, b.loopSpan(c, r)) {
				l.Reg = r + 1
				b.hold(r, b.loopSpan(c, r))
				break
			}
		}
		if l.Reg == 0 {
			return errors.New("no counter free")
		}
	}
	for r := range b.spans {
		b.used[r] = true
	}
	return nil
}

// keepEnd is the last position value i, held for its role, is read at: its
// last read, its loop's end where it is the loop's counter, and for a
// buffer's address the end of every loop its buffer is reached in, from
// the outermost; len(Seq) where that is not known.
func (b *binder86) keepEnd(i int) int {
	k := b.k
	nd := k.Nodes[i]
	if nd.Op == OpFold || nd.Op == OpSpan {
		return len(k.Seq)
	}
	e := max(b.def.of(i), b.last.of(i))
	for c, l := range k.Loops {
		if l.Count == i+1 && l.Reg == nd.Reg {
			e = max(e, b.end.of(c))
		}
	}
	if !k.isBaseNode(i) {
		return e
	}
	top := func(c int) int {
		for k.Loops[c].Parent >= 0 {
			c = k.Loops[c].Parent
		}
		return b.end.of(c)
	}
	hs := k.homesNow()
	mine := map[int]bool{}
	for j, bf := range k.Bufs {
		if bf.Base == i+1 {
			mine[j] = true
		}
	}
	for x, n := range k.Nodes {
		if n.Op != OpLoad || !mine[n.Buf] {
			continue
		}
		e = max(e, b.def.of(x), b.last.of(x))
		if x < len(hs) && hs[x] >= 0 {
			e = max(e, top(hs[x]))
		}
	}
	for pos, en := range k.Seq {
		switch {
		case en.Kind == EntryStore && mine[k.Stores[en.I].Buf]:
			e = max(e, pos)
			if c := k.Stores[en.I].At; c >= 0 {
				e = max(e, top(c))
			}
		case en.Kind == EntryAhead && mine[k.Nodes[k.Lines[en.I].Load].Buf]:
			e = max(e, pos)
		}
	}
	for c, l := range k.Loops {
		for _, cu := range l.Cursors {
			if mine[cu.Buf] {
				e = max(e, top(c))
			}
		}
	}
	return e
}

// takeCounter is the first register of pool free over loop c, held there.
func (b *binder86) takeCounter(pool []int, c int) int {
	for _, r := range pool {
		if s := b.loopSpan(c, r); b.fits(r, s) {
			b.hold(r, s)
			return r
		}
	}
	return -1
}

// valueSpans holds each general register a value or an iteration takes over
// the positions it is live: a value kept, the whole kernel; a value spilled,
// where it is made.
func (b *binder86) valueSpans(keeps func(int) bool) {
	k := b.k
	words := k.vectorWords()
	whole := span{0, 2 * len(k.Seq)}
	for c := range k.Loops {
		if it := k.iterOf(c); it >= 0 && k.Nodes[it].Reg > 0 {
			b.hold(k.Nodes[it].Reg-1, span{2*b.start.of(c) + 1, 2 * b.end.of(c)})
		}
	}
	for i, nd := range k.Nodes {
		if nd.Reg == 0 || nd.Op == OpIter || nd.Spill < 0 || nd.Reg-1 == int(reloadReg) {
			continue
		}
		if nd.Op != OpParam && (nd.Type == ir.F32 || nd.Op == OpConst || nd.Op == OpLoad && (nd.Mem || k.needsVector(i)) || words[i]) {
			continue
		}
		d, ok := b.def.get(i)
		switch {
		case !ok:
			b.hold(nd.Reg-1, whole)
		case keeps(i):
			b.hold(nd.Reg-1, span{2 * d, 2 * max(d, b.keepEnd(i))})
		case nd.Spill > 0:
			b.hold(nd.Reg-1, span{2 * d, 2 * d})
		default:
			b.hold(nd.Reg-1, span{2 * d, 2 * max(d, b.last.of(i))})
		}
	}
}

func (b *binder86) offBase(cu Cursor) bool {
	if cu.From == 0 {
		return b.k.Bufs[cu.Buf].Reg > 0
	}
	f := b.k.Ptrs[cu.From-1]
	return f.Reg > 0 && f.Off == 0
}

// offsetGroups groups the copies one loop takes that live as long and move
// alike, their sources still: each group can share one offset register.
func (k *Kernel) offsetGroups(lastLoop map[int]int) map[int]offGroup {
	type adv struct{ loop, bytes int }
	take, from, buf := map[int]int{}, map[int]int{}, map[int]int{}
	steps := map[int][]adv{}
	sources := map[[2]int]bool{}
	for c, l := range k.Loops {
		for _, cu := range l.Cursors {
			if cu.Take {
				sources[[2]int{cu.Buf, cu.From}] = true
			}
			if cu.Ptr == 0 || cu.Init > 0 {
				continue
			}
			if cu.Take {
				take[cu.Ptr], from[cu.Ptr], buf[cu.Ptr] = c, cu.From, cu.Buf
			}
			if cu.Elems != 0 {
				steps[cu.Ptr] = append(steps[cu.Ptr], adv{c, cu.Elems * ElemBytes(k.Bufs[cu.Buf])})
			}
		}
	}
	order := map[int]int{}
	ends := map[int]int{}
	for pos, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			order[e.I] = pos
		case EntryEnd:
			ends[e.I] = pos
		}
	}
	moves := func(b, ptr, lo, hi int) bool {
		for c, l := range k.Loops {
			if order[c] < lo || order[c] > hi {
				continue
			}
			for _, cu := range l.Cursors {
				if cu.Buf == b && cu.Ptr == ptr && cu.Elems != 0 {
					return true
				}
			}
		}
		return false
	}
	type member struct{ ptr, unit int }
	keys := map[string][]member{}
	for ptr, c := range take {
		if sources[[2]int{buf[ptr], ptr}] || moves(buf[ptr], from[ptr], order[c], ends[lastLoop[ptr]]) {
			continue
		}
		unit := 0
		for _, a := range steps[ptr] {
			unit = gcd(unit, a.bytes)
		}
		if unit < 0 {
			continue
		}
		norm := make([]adv, len(steps[ptr]))
		for i, a := range steps[ptr] {
			norm[i] = adv{a.loop, a.bytes / max(unit, 1)}
		}
		key := fmt.Sprint(c, lastLoop[ptr], norm)
		keys[key] = append(keys[key], member{ptr, unit})
	}
	out := map[int]offGroup{}
	for g, key := range slices.Sorted(maps.Keys(keys)) {
		ms := keys[key]
		least := ms[0].unit
		for _, m := range ms {
			least = min(least, m.unit)
		}
		for _, m := range ms {
			sc := 1
			if least > 0 {
				sc = m.unit / least
			}
			if least > 0 && m.unit%least != 0 || sc != 1 && sc != 2 && sc != 4 && sc != 8 {
				continue
			}
			out[m.ptr] = offGroup{g, sc}
		}
	}
	return out
}

type offGroup struct{ g, scale int }

// uses is how many nodes and stores read node i.
func (b *binder86) uses(i int) int {
	k := b.k
	if key := [2]int{len(k.Nodes) + 1, len(k.Stores)}; b.useKey != key {
		b.useKey = key
		b.useN = slices.Grow(b.useN[:0], len(k.Nodes))[:len(k.Nodes)]
		clear(b.useN)
		var buf [3]int
		for x := range k.Nodes {
			for _, a := range k.args(x, &buf) {
				if a >= 0 && a < len(b.useN) {
					b.useN[a]++
				}
			}
		}
		for x := range k.Stores {
			for _, a := range k.storeArgs(&k.Stores[x]) {
				if *a >= 0 && *a < len(b.useN) {
					b.useN[*a]++
				}
			}
		}
	}
	if i < 0 || i >= len(b.useN) {
		return 0
	}
	return b.useN[i]
}

// slot is a red-zone slot free from sequence position from on, taken for
// owner (1 + a value's node, or -1 for good); -1 where none is.
func (b *binder86) slot(from, owner int) int {
	for s := 1; s <= spillSlots; s++ {
		if o := b.owner[s]; o == 0 || o > 0 && b.last.of(o-1) < from {
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
	return b.evictAt(freed, b.def.of(i))
}

// evictAt is evict at sequence position pos.
func (b *binder86) evictAt(freed map[int]int, pos int) int {
	v, last := -1, -1
	for x := range freed {
		if l := b.last.of(x); l > pos && (l > last || l == last && x < v) {
			v, last = x, l
		}
	}
	if v < 0 {
		return -1
	}
	s := b.slot(b.def.of(v), v+1)
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
	return k.needsVectorOf(i, nil)
}

// needsVectorOf is needsVector with the kernel's vector words, where
// words is set, those already found (VectorWords).
func (k *Kernel) needsVectorOf(i int, words map[int]bool) bool {
	nd := k.Nodes[i]
	if nd.Mem || k.dotZero(i) {
		return false
	}
	if nd.Type == ir.F32 {
		return nd.Op != OpConst || nd.Imm == 0 && k.zeroOperand(i)
	}
	if words == nil {
		words = k.vectorWords()
	}
	if nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind == OperandWeight && k.vectorized() {
		if words[i] {
			return true
		}
		// A word a scalar loop loads is read by the general registers, as
		// a jammed loop's word an address reads is.
		if hs := k.homesNow(); i < len(hs) && (hs[i] < 0 || k.Loops[hs[i]].Lanes == 0 && k.Loops[hs[i]].Wide == 0) {
			return false
		}
		return !k.wide() || !k.addressNodes()[i]
	}
	return words[i]
}

// vectorWords is VectorWords, kept while a register binding runs
// (Kernel.vwords) for as long as the kernel holds as many nodes: the
// binding adds nodes and never rewrites one's operation in place.
func (k *Kernel) vectorWords() map[int]bool {
	if k.vwords == nil {
		return k.VectorWords()
	}
	if k.vwords.n != len(k.Nodes) {
		k.vwords.n, k.vwords.words = len(k.Nodes), k.VectorWords()
	}
	return k.vwords.words
}

// homesNow is Homes, kept while a register binding runs (Kernel.vwords)
// for as long as the kernel holds as many nodes.
func (k *Kernel) homesNow() []int {
	m := k.vwords
	if m == nil {
		return k.Homes()
	}
	if m.homesN != len(k.Nodes)+1 {
		m.homesN, m.homes = len(k.Nodes)+1, k.Homes()
	}
	return m.homes
}

// vectorMemo is a register binding's VectorWords at a count of nodes.
type vectorMemo struct {
	n     int
	words map[int]bool
	// homesN is the count of nodes homes was found at, plus one (0: none).
	homesN int
	homes  []int
	// vecN is the count of nodes vec was found at, plus one (0: none).
	vecN int
	vec  bool
	// usesN is the count of nodes uses was found at, plus one (0: none),
	// usesS the count of stores (operandUses).
	usesN, usesS int
	uses         []bool
	// made is madeCounts' answers, found at madeKey's counts of nodes,
	// stores and loops; a copy of the memo starts its own (fresh).
	madeKey [3]int
	made    map[int][]int
	// addr is addressNodes' answer, found at addrKey's counts of nodes and
	// stores.
	addrKey [2]int
	addr    map[int]bool
	// dotN is the count of nodes dot was found at, plus one (0: none).
	dotN int
	dot  []bool
}

// fresh is m with madeCounts' answers of its own: copies write none into
// each other's.
func (m vectorMemo) fresh() *vectorMemo {
	m.made, m.madeKey = nil, [3]int{}
	return &m
}

// vectorized reports whether any loop of k runs in vector lanes, or any
// value is a byte dot, which only vector lanes make. A kernel of neither
// keeps its integers in general registers: an index, a weight's word and
// what is made from them.
func (k *Kernel) vectorized() bool {
	m := k.vwords
	if m == nil {
		return k.findVectorized()
	}
	if m.vecN != len(k.Nodes)+1 {
		m.vecN, m.vec = len(k.Nodes)+1, k.findVectorized()
	}
	return m.vec
}

// findVectorized is vectorized, found again.
func (k *Kernel) findVectorized() bool {
	for _, l := range k.Loops {
		if l.Lanes > 0 || l.Wide > 0 {
			return true
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLanes || nd.Op.byteDot() {
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
	in := make([]bool, len(k.Nodes))
	held := 0
	set := func(i int) {
		if !in[i] {
			in[i] = true
			held++
		}
	}
	data := func(i int) bool {
		nd := &k.Nodes[i]
		return nd.Type != ir.F32 && nd.Op != OpParam && nd.Op != OpIter && nd.Op != OpLength && nd.Op != OpSpan && nd.Op != OpCoord
	}
	vec := k.vectorized()
	var hs []int
	var addr map[int]bool
	if vec {
		hs = k.Homes()
		// A word a jammed loop loads for an address (a page's index) is
		// the row's own, read by the general registers.
		addr = k.addressNodes()
	}
	for i := range k.Nodes {
		nd := &k.Nodes[i]
		switch {
		case nd.Op == OpLanes || nd.Op == OpBroadcast && nd.Type != ir.F32 || nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind == OperandConsts && nd.Type != ir.F32 && !k.BitsGPR:
			set(i)
		case vec && nd.Op == OpLoad && nd.Type != ir.F32 && !k.Bufs[nd.Buf].Half && i < len(hs) && hs[i] >= 0 && (k.Loops[hs[i]].Lanes > 0 || k.Loops[hs[i]].Wide > 0 && !addr[i]):
			// A word a vector loop loads is a lane's each.
			set(i)
		case vec && (nd.Op == OpCvtF32 || nd.Op == OpWidenHalf || nd.Op == OpBitcast && nd.Type == ir.F32):
			if data(nd.A) && k.Nodes[nd.A].Op != OpConst {
				set(nd.A)
			}
		case (nd.Op == OpBitcast || nd.Op == OpCvtI32 || nd.Op == OpRoundI32) && !k.BitsGPR:
			if k.Nodes[nd.A].Type == ir.F32 {
				set(i)
			}
		}
	}
	var buf [3]int
	for grew := true; grew; {
		before := held
		for changed := true; changed; {
			changed = false
			for i := range k.Nodes {
				if !in[i] {
					continue
				}
				nd := &k.Nodes[i]
				// A broadcast of a computed word reads it from a general
				// register (Splats).
				if nd.Op == OpLoad || nd.Op == OpFold || nd.Op == OpBroadcast && nd.Type != ir.F32 {
					continue
				}
				for x, a := range k.args(i, &buf) {
					// A shift's count shifts every lane alike: it is a vector
					// only where it is made of one.
					if x == 1 && (nd.Op == OpShl || nd.Op == OpShr || nd.Op == OpSar) {
						continue
					}
					// A compare a vector word reads is a vector compare.
					if data(a) && k.Nodes[a].Op != OpConst && !in[a] && (k.Nodes[a].Type != ir.Pred || k.Nodes[a].Op == OpLess) {
						set(a)
						changed = true
					}
				}
			}
		}
		for changed := true; changed; {
			changed = false
			for i := range k.Nodes {
				nd := &k.Nodes[i]
				if in[i] || nd.Type == ir.F32 {
					continue
				}
				v := nd.Op == OpHalf || nd.Op == OpRoundI32 && !k.BitsGPR || (nd.Op == OpLess || nd.Op == OpEq) && (k.Nodes[nd.A].Type == ir.F32 || in[nd.A] || in[nd.B])
				if !v && (nd.Op == OpAnd || nd.Op == OpGtW || nd.Op == OpAndNot || nd.Op == OpXor || nd.Op == OpShr || nd.Op == OpShl || nd.Op == OpSar || (nd.Type == ir.I32 || nd.Type == ir.U32) && nd.Op != OpConst && nd.Op != OpLoad) {
					for _, a := range k.args(i, &buf) {
						v = v || in[a]
					}
				}
				if !v && nd.Type == ir.I32 && nd.Op == OpFold {
					v = true
				}
				if v {
					set(i)
					changed = true
				}
			}
		}
		grew = held > before
	}
	for i := range k.Nodes {
		nd := &k.Nodes[i]
		if !in[i] || nd.Op == OpLoad || nd.Op == OpFold {
			continue
		}
		for x, a := range k.args(i, &buf) {
			if k.Nodes[a].Op == OpConst && k.Nodes[a].Type != ir.F32 && !(x == 1 && (nd.Op == OpShl || nd.Op == OpShr || nd.Op == OpSar)) {
				set(a)
			}
		}
	}
	out := make(map[int]bool, held)
	for i, v := range in {
		if v {
			out[i] = true
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
	// lastAt[lastOff[q]:lastOff[q+1]] is the values entry q of the sequence
	// reads for the last time, a sum's steps apart: free visits those alone.
	lastOff := make([]int, len(k.Seq)+1)
	lastOf := func(v int) int {
		if l, ok := b.last.get(v); ok && l >= 0 && l < len(k.Seq) {
			if _, step := stepOf[v]; !step {
				return l
			}
		}
		return -1
	}
	for v := range k.Nodes {
		if l := lastOf(v); l >= 0 {
			lastOff[l+1]++
		}
	}
	for q := range len(k.Seq) {
		lastOff[q+1] += lastOff[q]
	}
	lastAt, fill := make([]int, lastOff[len(k.Seq)]), append([]int(nil), lastOff[:len(k.Seq)]...)
	for v := range k.Nodes {
		if l := lastOf(v); l >= 0 {
			lastAt[fill[l]] = v
			fill[l]++
		}
	}
	// carry is the registers of sums whose one reader took the sum's
	// register: each stays the sum's to its last read, for its step.
	carry := map[int]int{}
	free := func(pos int, spent bool) {
		for _, v := range lastAt[lastOff[pos]:lastOff[pos+1]] {
			if k.Nodes[v].Reg > 0 && k.needsVector(v) && busy[k.Nodes[v].Reg-1] {
				r := k.Nodes[v].Reg - 1
				if f, ok := carry[r]; ok && f != v {
					continue
				}
				delete(carry, r)
				busy[r] = false
				tag[r] = chain[v]
				if spent && !k.isZero(v) {
					tag[r] = -1
				}
			}
		}
	}
	innermost := func(c int) bool { return c >= 0 && len(k.innerLoops(c)) == 0 }
	// reads is the reads of each sum in its own loop; one after the loop
	// reads its last step, which its register holds.
	reads, hr := map[int]int{}, k.homesNow()
	within := func(at, f int) bool { return at >= 0 && k.Nodes[f].Op == OpFold && k.within(at, k.Nodes[f].At) }
	var buf [3]int
	for j := range k.Nodes {
		if k.Nodes[j].Op == OpFold || j >= len(hr) {
			continue
		}
		for _, a := range k.args(j, &buf) {
			if within(hr[j], a) {
				reads[a]++
			}
		}
	}
	for x := range k.Stores {
		st := &k.Stores[x]
		for _, a := range k.storeArgs(st) {
			if within(st.At, *a) {
				reads[*a] += 2
			}
		}
	}
	held := make([]int, b.p.Vectors)
	for r := range held {
		held[r] = -1
	}
	shared := map[int]int{}
	next := b.p.Scalars - 1
	hs0 := k.homesNow()
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
		if _, made := b.def.get(i); made && nd.Hint > 0 {
			hinted = append(hinted, i)
		}
	}
	reserved := map[int]bool{}
	bar := func(i int) {
		clear(reserved)
		d, l := b.def.of(i), b.last.of(i)
		for c, rs := range kept {
			if sp := span[c]; sp[0] <= l && d <= sp[1] {
				for r := range rs {
					reserved[r] = true
				}
			}
		}
		for _, h := range hinted {
			if h != i && d < b.def.of(h) && b.def.of(h) <= l {
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
	hs := k.homesNow()
	at := 0
	give := func(i, r int) error {
		if r < 0 {
			return &pressureAt{node: i, pos: at, op: k.Nodes[i].Op, in: b}
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
		at = pos
		if e.Kind == EntryNode && cp[e.I] > 0 {
			free(pos, false)
			continue
		}
		if e.Kind == EntryNode && k.Nodes[e.I].Op == OpConst && k.Nodes[e.I].Type != ir.F32 {
			continue
		}
		if e.Kind == EntryNode && k.Nodes[e.I].Op == OpFold && !k.needsVector(e.I) {
			// A sum of words a general register holds (gprs).
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
				return fmt.Errorf("a constant held across the loops on a register a loop's value is hinted to (node %d, register %d, hint %d)", i, want, nd.Hint)
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
				if b.last.of(nd.C) != pos || !k.needsVector(nd.C) {
					first = nd.A
				}
			case OpDot4, OpDot4U:
				first = nd.C
			}
			// A sum's one reader in its loop, the first of a chain of
			// several to its step, takes the sum's register.
			if f := carried(nd); reg < 0 && f >= 0 && k.Nodes[f].Op == OpFold && k.Nodes[f].Reg > 0 && k.needsVector(f) && k.Nodes[f].At == hs[i] && reads[f] == 1 {
				reg = k.Nodes[f].Reg - 1
				carry[reg] = f
			}
			if reg < 0 && first >= 0 && k.Nodes[first].Reg > 0 && k.needsVector(first) && b.last.of(first) == pos {
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
			need := k.dotTemps(i, b.isa)
			var ts [2]int
			for j := range need {
				ts[j] = lowest(0, hs[i])
				if ts[j] >= 0 {
					busy[ts[j]] = true
				}
			}
			for j := range need {
				if ts[j] >= 0 {
					busy[ts[j]] = false
				}
			}
			for _, r := range held {
				busy[r] = false
			}
			if need > 0 && ts[0] < 0 || need > 1 && ts[1] < 0 {
				return errors.New("no registers for a byte dot")
			}
			k.Nodes[i].Tmp, k.Nodes[i].Tmp2 = 0, 0
			if need > 0 {
				k.Nodes[i].Tmp = ts[0] + 1
			}
			if need > 1 {
				k.Nodes[i].Tmp2 = ts[1] + 1
			}
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
		if _, ok := b.last.get(i); !ok {
			if _, step := stepOf[i]; !step {
				busy[reg] = false
			}
		}
	}
	return k.copyRegs(cp, len(splats))
}

// zerosPerLoop makes a float zero read or stored in several loops in each of them,
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
			if e.Kind == EntryStore && k.Stores[e.I].Val == z && k.Stores[e.I].At >= 0 {
				c := k.Stores[e.I].At
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
				k.Stores[e.I].Val = n
				continue
			}
			if e.Kind != EntryNode || e.I >= len(hs) || hs[e.I] < 0 {
				continue
			}
			// An operand is fetched again after a zero is added: the append
			// can move the nodes, and a pointer taken before it writes the old ones.
			for j := range len(k.Operands(&k.Nodes[e.I])) {
				if *k.Operands(&k.Nodes[e.I])[j] != z {
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
				*k.Operands(&k.Nodes[e.I])[j] = n
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
			k.forgetUses()
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

// foldsAside copies each vector fold read in its loop after its step is
// made where the step is made, those readers reading the copy.
func (k *Kernel) foldsAside() {
	for f := range k.Nodes {
		nd := k.Nodes[f]
		if nd.Op != OpFold || nd.A == f || !k.needsVector(f) {
			continue
		}
		pos, end := -1, -1
		for p, e := range k.Seq {
			if e.Kind == EntryNode && e.I == nd.A {
				pos = p
			}
			if e.Kind == EntryEnd && e.I == nd.At && pos >= 0 {
				end = p
				break
			}
		}
		if pos < 0 || end < 0 || !k.readBetween(f, pos+1, end) {
			continue
		}
		c := k.add(Node{Op: OpCopy, Type: nd.Type, A: f})
		for q := pos + 1; q < end; q++ {
			for _, a := range k.entryArgs(k.Seq[q]) {
				if *a == f {
					*a = c
				}
			}
		}
		k.forgetUses()
		k.Seq = slices.Insert(k.Seq, pos, Entry{Kind: EntryNode, I: c})
	}
}

// readBetween reports whether an entry at sequence positions from to end,
// end excluded, reads node t.
func (k *Kernel) readBetween(t, from, end int) bool {
	for p := from; p < end; p++ {
		for _, a := range k.entryArgs(k.Seq[p]) {
			if *a == t {
				return true
			}
		}
	}
	return false
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
	if *next < 0 || *next >= b.p.Vectors || nd.Op != OpLoad || k.Bufs[nd.Buf].Operand.Kind != OperandConsts || nd.Type != ir.F32 || hs[i] >= 0 || nd.Hint > 0 {
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
	var aside []int
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
		if k.Nodes[t].Hint > 0 {
			// The twin's register is its hint, not the first head's run of
			// registers: the copy takes one no value takes.
			aside = append(aside, i)
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
	taken := make([]bool, 16)
	if k.Ones > 0 {
		taken[k.Ones-1] = true
	}
	for i, nd := range k.Nodes {
		if !k.needsVector(i) {
			continue
		}
		for _, r := range []int{nd.Reg, nd.Tmp, nd.Tmp2} {
			if r > 0 && r <= len(taken) {
				taken[r-1] = true
			}
		}
	}
	// From the top down: the low registers are the ones passes take as
	// scratch for an operation's code (Exp's).
	free := func() int {
		for r := 16 - splats - 1; r >= 0; r-- {
			if !taken[r] {
				taken[r] = true
				return r + 1
			}
		}
		return 0
	}
	for _, i := range aside {
		t := tw[i]
		if k.Nodes[i].Reg = free(); k.Nodes[i].Reg == 0 {
			return errors.New("a pair's heads take more vector registers than there are")
		}
		if k.Nodes[t].Tmp > 0 {
			if k.Nodes[i].Tmp = free(); k.Nodes[i].Tmp == 0 {
				return errors.New("a pair's heads take more vector registers than there are")
			}
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
	hs := k.homesNow()
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
func (k *Kernel) rematerialize(moved map[int]bool, sc *stepScratch) bool {
	pos := sc.places(k)
	hs := k.homesNow()
	// Each node's readers in the sequence, made once: asked of every
	// candidate, a scan of the kernel each made a rematerialization
	// quadratic in its nodes.
	off, rd := sc.readers(k, pos, false)
	pick := func(outside bool) int {
		best, at := -1, -1
		for i, nd := range k.Nodes {
			p := pos[i]
			in := p >= 0
			if o := k.Bufs[nd.Buf].Operand.Kind; !in || nd.Op != OpLoad || hs[i] >= 0 || o != OperandConsts && k.Bufs[nd.Buf].Words == nil || moved[i] {
				continue
			}
			rs := rd[off[i]:off[i+1]]
			ok := len(rs) > 0
			first := -1
			for _, r := range rs {
				if hs[r] < 0 && !outside {
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
		return best
	}
	// A constant read outside every loop -- by a tail of one trip run
	// without its loop, or by a kernel of no loops -- is loaded where each
	// reader reads it once none read only in the loops is left.
	best := pick(false)
	if best < 0 {
		best = pick(true)
	}
	if best < 0 {
		return false
	}
	k.reload(best, moved)
	return true
}

// sink moves a vector value made in a run of straight code long before its
// one reader, with what only it reads, to just before the reader: the
// floats made between then hold no register for it. The value with the
// most vector values between it and its reader moves first; false where
// none is left. A paired kernel's statements stand beside their twins and
// stay.
func (k *Kernel) sink(sunk map[int]bool, sc *stepScratch) bool {
	for _, c := range k.copies() {
		if c > 0 {
			return false
		}
	}
	// pos is each node's place in the sequence (-1 for none), and node a's
	// readers rd[off[a]:off[a+1]] the sequenced nodes reading it (-1 a store).
	pos := sc.places(k)
	off, rd := sc.readers(k, pos, true)
	words := k.vectorWords()
	// vecs[q] and others[q] count the vector values and the entries not a
	// node before sequence place q: a candidate's span is straight where
	// none of the second lies in it, and holds the first's difference.
	vecs, others := sized(sc.vecs, len(k.Seq)+1), sized(sc.others, len(k.Seq)+1)
	needs := sized(sc.needs, len(k.Seq))
	sc.vecs, sc.others, sc.needs = vecs, others, needs
	vecs[0], others[0] = 0, 0
	for q, e := range k.Seq {
		vecs[q+1], others[q+1] = vecs[q], others[q]
		if e.Kind != EntryNode {
			others[q+1]++
			continue
		}
		if needs[q] = k.needsVectorOf(e.I, words); needs[q] {
			vecs[q+1]++
		}
	}
	best, gap, to := -1, 0, 0
	for p, e := range k.Seq {
		x := e.I
		if e.Kind != EntryNode || !needs[p] || off[x+1]-off[x] != 1 || rd[off[x]] < 0 || !k.sinkMovable(x) || sunk[x] {
			continue
		}
		rp := pos[rd[off[x]]]
		if rp <= p+1 || others[rp]-others[p+1] != 0 {
			continue
		}
		if n := vecs[rp] - vecs[p+1]; n > gap {
			best, gap, to = x, n, rp
		}
	}
	if best < 0 {
		return false
	}
	sunk[best] = true
	group := k.sinkGroup(best, to, pos, off, rd)
	var out, moving []Entry
	for p, e := range k.Seq {
		if p == to {
			out = append(out, moving...)
		}
		if e.Kind == EntryNode && group[e.I] {
			moving = append(moving, e)
			continue
		}
		out = append(out, e)
	}
	k.Seq = out
	return true
}

// sinkMovable reports whether a sink may move node i.
func (k *Kernel) sinkMovable(i int) bool {
	nd := k.Nodes[i]
	return nd.Op != OpFold && nd.Op != OpConst && nd.Share == 0 && nd.Spill == 0
}

// straightSeq reports whether the sequence between places from and to holds
// nodes alone.
func (k *Kernel) straightSeq(from, to int) bool {
	for q := from + 1; q < to; q++ {
		if k.Seq[q].Kind != EntryNode {
			return false
		}
	}
	return true
}

// sinkGroup is what a sink of best to sequence place to moves: best, and
// each value of its straight run before to that the group alone reads. pos
// and the readers rd[off[a]:off[a+1]] are sink's.
func (k *Kernel) sinkGroup(best, to int, pos, off, rd []int) map[int]bool {
	group := map[int]bool{best: true}
	for grew := true; grew; {
		grew = false
		for i := range group {
			for _, a := range k.Operands(&k.Nodes[i]) {
				x := *a
				if x < 0 || x >= len(pos) || group[x] || !k.sinkMovable(x) || pos[x] >= to {
					continue
				}
				if pos[x] < 0 || !k.straightSeq(pos[x], pos[best]) {
					continue
				}
				only := true
				for _, r := range rd[off[x]:off[x+1]] {
					only = only && r >= 0 && group[r]
				}
				if only {
					group[x], grew = true, true
				}
			}
		}
	}
	return group
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
		k.forgetUses()
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
		k.forgetUses()
		k.Seq = append(k.Seq[:first], append([]Entry{{Kind: EntryNode, I: c}}, k.Seq[first:]...)...)
	}
}

// carried is the operand of nd a result may take the register of: a
// multiply-add's or byte dot's addend, else the first operand of an
// operation the lowering makes in place of its first; -1 for others.
func carried(nd Node) int {
	switch nd.Op {
	case OpFma, OpDot4, OpDot4U:
		return nd.C
	case OpAdd, OpSub, OpMul, OpMin, OpMax:
		return nd.A
	}
	return -1
}

// dotTemps is the scratch vector registers byte dot i takes beyond its
// own: the signed operand and the magnitudes where A's bytes may be
// negative, else the pairs' sums apart from the result where something is
// added to them or VNNI may sum aside; the legacy encodings take two.
func (k *Kernel) dotTemps(i int, isa x86.ISA) int {
	nd := k.Nodes[i]
	switch {
	case sseNarrow(isa) || nd.Op != OpDot4U && !k.unsignedBytes(nd.A):
		return 2
	case isa&x86.ISAVNNI == 0 && k.dotZero(nd.C):
		return 0
	}
	return 1
}

// dotZero reports whether node i is a word zero read only as a sum's start
// or a byte dot's addend: the dot is its products' sums alone, and the zero
// takes no register.
func (k *Kernel) dotZero(i int) bool {
	if nd := k.Nodes[i]; nd.Op != OpConst || nd.Imm != 0 || nd.Type == ir.F32 {
		return false
	}
	return k.dotZeros()[i]
}

// dotZeros is dotZero for every node, found in one walk of the operands:
// kept while a register binding runs as operandUses is, where needsVector
// asked it of every node and each asking walked the kernel.
func (k *Kernel) dotZeros() []bool {
	m := k.vwords
	if m != nil && m.dotN == len(k.Nodes)+1 {
		return m.dot
	}
	n := len(k.Nodes)
	zero := func(i int) bool {
		nd := k.Nodes[i]
		return nd.Op == OpConst && nd.Imm == 0 && nd.Type != ir.F32
	}
	read, other := make([]bool, n), make([]bool, n)
	var buf [3]int
	for j := range k.Nodes {
		nd := &k.Nodes[j]
		for at, i := range k.args(j, &buf) {
			if i < 0 || i >= n || i == j || !zero(i) || nd.Op == OpFold && nd.B == i && nd.A != i {
				continue
			}
			// A byte dot's third operand is its C.
			if !nd.Op.byteDot() || at != 2 || nd.A == i || nd.B == i {
				other[i] = true
			} else {
				read[i] = true
			}
		}
	}
	for i := range read {
		read[i] = read[i] && !other[i]
	}
	if m != nil {
		m.dotN, m.dot = n+1, read
	}
	return read
}

// zeroOperand reports whether float zero node i is an operand the code
// computes with or a store writes, not only a sum's start.
func (k *Kernel) zeroOperand(i int) bool {
	u := k.operandUses()
	return i < len(u) && u[i]
}

// operandUses is, by node, whether another node reads it (a fold aside) or a
// store writes it: kept while a register binding runs for as long as the
// kernel holds as many nodes and stores, and dropped (forgetUses) where the
// binding re-points an operand in place.
func (k *Kernel) operandUses() []bool {
	m := k.vwords
	if m != nil && m.usesN == len(k.Nodes)+1 && m.usesS == len(k.Stores) {
		return m.uses
	}
	uses := make([]bool, len(k.Nodes))
	var buf [3]int
	for j := range k.Nodes {
		if k.Nodes[j].Op == OpFold {
			continue
		}
		for _, a := range k.args(j, &buf) {
			if a != j && a >= 0 && a < len(uses) {
				uses[a] = true
			}
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run && s.Val >= 0 && s.Val < len(uses) {
			uses[s.Val] = true
		}
	}
	if m != nil {
		m.usesN, m.usesS, m.uses = len(k.Nodes)+1, len(k.Stores), uses
	}
	return uses
}

// forgetUses drops the binding's operandUses and homes after an operand is
// re-pointed: a node's home follows what it reads.
func (k *Kernel) forgetUses() {
	if k.vwords != nil {
		k.vwords.usesN, k.vwords.homesN, k.vwords.dotN = 0, 0, 0
	}
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

// reloadCopies picks the copies that take the n reload registers a binding
// holds back and no value needs: the most read a trip of a loop reading it
// first, no more than n of them live at once.
func (k *Kernel) reloadCopies(groups map[int]offGroup, lastLoop map[int]int, n int) map[int]bool {
	weight := func(c int) int {
		w := 1
		for ; c >= 0; c = k.Loops[c].Parent {
			if !k.Loops[c].once() {
				w *= 16
			}
		}
		return w
	}
	hs := k.Homes()
	reads := map[int]int{}
	loops := map[int]map[int]bool{}
	read := func(ptr, c int) {
		reads[ptr] += weight(c)
		if loops[ptr] == nil {
			loops[ptr] = map[int]bool{}
		}
		loops[ptr][c] = true
	}
	for i, nd := range k.Nodes {
		if nd.Ptr > 0 && hs[i] >= 0 {
			read(nd.Ptr, hs[i])
		}
	}
	for _, st := range k.Stores {
		if st.Ptr > 0 && st.At >= 0 {
			read(st.Ptr, st.At)
		}
	}
	pos, end := map[int]int{}, map[int]int{}
	for p, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			pos[e.I] = p
		case EntryEnd:
			end[e.I] = p
		}
	}
	type cand struct {
		ptrs           []int
		weight, lo, hi int
	}
	byKey := map[[2]int]*cand{}
	var cs []*cand
	for c, l := range k.Loops {
		for _, cu := range l.Cursors {
			if !cu.Take || cu.Ptr == 0 {
				continue
			}
			key := [2]int{-1, cu.Ptr}
			if g, ok := groups[cu.Ptr]; ok {
				key = [2]int{g.g, c}
			}
			x := byKey[key]
			if x == nil {
				x = &cand{lo: pos[c], hi: end[lastLoop[cu.Ptr]]}
				byKey[key] = x
				cs = append(cs, x)
			}
			x.ptrs = append(x.ptrs, cu.Ptr)
			x.weight += reads[cu.Ptr]/max(len(loops[cu.Ptr]), 1) + weight(c)
		}
	}
	slices.SortStableFunc(cs, func(a, b *cand) int { return b.weight - a.weight })
	out := map[int]bool{}
	var chosen []*cand
	for _, x := range cs {
		live := 0
		for _, y := range chosen {
			if x.lo <= y.hi && y.lo <= x.hi {
				live++
			}
		}
		if live >= n {
			continue
		}
		chosen = append(chosen, x)
		for _, p := range x.ptrs {
			out[p] = true
		}
	}
	return out
}

func (k *Kernel) spills() bool {
	for _, nd := range k.Nodes {
		if nd.Spill != 0 {
			return true
		}
	}
	return false
}

// lifeTab is a binder's position for each index: a value, 0 where none was
// put, and whether one was.
type lifeTab struct {
	at  []int
	set []bool
}

// reset empties t for n indexes, keeping its storage.
func (t *lifeTab) reset(n int) {
	t.at = slices.Grow(t.at[:0], n)[:n]
	t.set = slices.Grow(t.set[:0], n)[:n]
	clear(t.at)
	clear(t.set)
}

// get is index i's position and whether one was put.
func (t *lifeTab) get(i int) (int, bool) {
	if i < 0 || i >= len(t.at) {
		return 0, false
	}
	return t.at[i], t.set[i]
}

// of is index i's position, 0 where none was put.
func (t *lifeTab) of(i int) int {
	v, _ := t.get(i)
	return v
}

// put sets index i's position to v.
func (t *lifeTab) put(i, v int) {
	if i >= len(t.at) {
		t.at = append(t.at, make([]int, i+1-len(t.at))...)
		t.set = append(t.set, make([]bool, i+1-len(t.set))...)
	}
	t.at[i], t.set[i] = v, true
}
