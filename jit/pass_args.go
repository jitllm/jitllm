package jit

import (
	"fmt"
	"sort"

	"github.com/jitllm/jitllm/jit/ir"
)

// Args binds a kernel to the words its call passes: every count a loop
// takes that no loop of the kernel moves is a word the caller computes, and
// every buffer the code reads or writes an address word, the part of its
// indices no loop moves added by the caller, the kernel indexing from the
// address it is given. Fields is the device's convention for which word
// each takes, Parts its convention for the kernels of a kind of part where
// the caller of that part sets another.
type Args struct {
	Fields ArgFields
	Parts  map[PartKind]ArgFields
}

// ArgFields are the byte offsets of the argument words: the buffers written
// in order, the first weight's or paged history's address (a table read at
// addresses it loads; or the row a kernel narrows into binary16), the activations a weight is read beside, the
// value a sum over the history's width reads beside it, a block
// of constants (or an input of fixed words), the other buffers read in order; a vector loop's count and
// its tail's, and the other counts in order. Index, where set, is the word
// of a buffer of integers written; Word, where set, the word an input read
// at one fixed word alone is passed in, by value.
type ArgFields struct {
	Out    []int
	Index  int
	Word   int
	Weight int
	Act    int
	Query  int
	// Second is, for a field, the field of a pair's second head's buffer.
	Second        map[int]int
	Consts        int
	In            []int
	Vectors, Tail int
	Counts        []int
	// Coords, where set, are the words each axis of a kernel's calls'
	// space passes its coordinate in, the outermost first, and the kernel
	// addresses every buffer from the address it is given with the whole of
	// its indices: no offset is the caller's but a constant one.
	Coords []int
}

func (Args) Name() string { return "args" }

func (p Args) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Work || k.Group > 0 || k.Args != nil {
			return nil, false, nil
		}
		m := k.clone()
		f := p.Fields
		if pf, ok := p.Parts[k.Part]; ok {
			f = pf
		}
		if err := m.bindCoords(f); err != nil {
			return nil, false, err
		}
		m.bindCounts(f)
		m.bindBufs(f)
		if err := m.oneBufferAField(); err != nil {
			return nil, false, err
		}
		return []*Kernel{m}, true, nil
	})
}

// bindCoords makes each coordinate of the calls' space the kernel still
// reads a word the caller passes: a kernel whose space has more axes than
// the device has words for is refused.
func (k *Kernel) bindCoords(f ArgFields) error {
	if f.Coords == nil {
		return nil
	}
	bound := map[int]int{}
	for i := range k.Nodes {
		nd := k.Nodes[i]
		if nd.Op != OpCoord {
			continue
		}
		if nd.A >= len(k.Space) || nd.A >= len(f.Coords) {
			return fmt.Errorf("%s reads coordinate %d of a calls' space of %d axes, where the device passes %d", k.Name, nd.A, len(k.Space), len(f.Coords))
		}
		if k.Space[nd.A] == 1 {
			k.Nodes[i] = Node{Op: OpConst, Type: ir.U32}
			continue
		}
		if _, ok := bound[nd.A]; !ok {
			bound[nd.A] = i
			k.Args = append(k.Args, Arg{Field: f.Coords[nd.A], Buf: -1, Node: i + 1, Coord: nd.A + 1})
		}
		k.Nodes[i] = Node{Op: OpParam, Type: ir.U32, Imm: int64(f.Coords[nd.A])}
	}
	return nil
}

// oneBufferAField refuses a kernel whose argument words give one field to two
// buffers: the second would overwrite the first's address.
func (k *Kernel) oneBufferAField() error {
	by := map[int]Operand{}
	for _, a := range k.Args {
		if a.Buf < 0 {
			continue
		}
		o := k.Bufs[a.Buf].Operand
		if p, ok := by[a.Field]; ok && p != o {
			return fmt.Errorf("jit: %s: argument field %d holds both %s and %s", k.Name, a.Field, p.Name, o.Name)
		}
		by[a.Field] = o
	}
	return nil
}

// fixed reports whether no loop of k moves node i.
func (k *Kernel) fixed(i int) bool {
	for c := range k.Loops {
		if k.Varies(i, c) {
			return false
		}
	}
	return true
}

// bindCounts makes each loop count the call fixes an argument word, the
// loops taken outermost first: the length a count was split from, whole,
// the count derived from it in the kernel. The call states lengths, never
// what a kernel makes of them.
func (k *Kernel) bindCounts(f ArgFields) {
	next := 0
	field := func(c int) int {
		l := k.Loops[c]
		if l.Lanes > 0 && len(k.foldsOf(c)) == 0 {
			return f.Vectors
		}
		if len(k.foldsOf(c)) == 0 && k.vectorTail(c) {
			return f.Tail
		}
		v := f.Counts[min(next, len(f.Counts)-1)]
		next++
		return v
	}
	bound := map[int]bool{}
	for _, c := range k.preorder() {
		n := k.Loops[c].Count - 1
		if n < 0 || bound[n] || k.Nodes[n].Op == OpConst || k.Nodes[n].Op == OpParam || !k.fixed(n) {
			continue
		}
		{
			ln := k.lengthIn(n)
			if ln < 0 && k.computedFrom(n, OpParam) {
				continue
			}
			// Where the kernel addresses whole indices it counts what it
			// loads itself: a count made from a table's words is its own.
			if ln < 0 && f.Coords != nil && k.computedFrom(n, OpLoad) {
				continue
			}
			if ln < 0 {
				ln = k.splitFrom(n)
			}
			n = ln
			if bound[n] {
				continue
			}
		}
		bound[n] = true
		fl := field(c)
		pv := k.add(Node{Op: OpParam, Type: ir.U32, Imm: int64(fl)})
		k.redirectAll(n, pv)
		k.Args = append(k.Args, Arg{Field: fl, Buf: -1, Node: n + 1})
	}
}

// lengthIn is the run-time length node i is computed from, -1 for none.
func (k *Kernel) lengthIn(i int) int {
	nd := k.Nodes[i]
	if nd.Op == OpLength {
		return i
	}
	if nd.Op == OpFold {
		return -1
	}
	for _, a := range k.Operands(&nd) {
		if l := k.lengthIn(*a); l >= 0 {
			return l
		}
	}
	return -1
}

// preorder is the loops, each before the loops inside it, siblings in
// order.
func (k *Kernel) preorder() []int {
	var out []int
	var walk func(p int)
	walk = func(p int) {
		for c, l := range k.Loops {
			if l.Parent == p {
				out = append(out, c)
				walk(c)
			}
		}
	}
	walk(-1)
	return out
}

// tailOf is the loop taking vector loop c's remaining trips, or -1.
func (k *Kernel) tailOf(c int) int {
	for t := c + 1; t < len(k.Loops); t++ {
		if k.Loops[t].Lanes == 0 && k.Loops[t].Parent == k.Loops[c].Parent && len(k.innerLoops(t)) == 0 {
			return t
		}
	}
	return -1
}

// access is one load or store of a buffer: node Node's or store Store's.
type access struct {
	node, store int
}

// bindBufs gives each buffer the code reads or writes an address word.
func (k *Kernel) bindBufs(f ArgFields) {
	live := k.Live()
	byOp := map[Operand][]access{}
	var order []Operand
	note := func(b int, a access) {
		o := k.Bufs[b].Operand
		if _, ok := byOp[o]; !ok {
			order = append(order, o)
		}
		byOp[o] = append(byOp[o], a)
	}
	written := map[Operand]bool{}
	narrows := false
	for i, s := range k.Stores {
		if !s.Barrier && !s.Run && k.Bufs[s.Buf].Kind != BufShared {
			note(s.Buf, access{node: -1, store: i})
			written[k.Bufs[s.Buf].Operand] = true
			narrows = narrows || k.Bufs[s.Buf].Half
		}
	}
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && live[i] && k.Bufs[nd.Buf].Kind != BufShared {
			note(nd.Buf, access{node: i, store: -1})
		}
	}
	sort.SliceStable(order, func(i, j int) bool { return k.firstBuf(order[i]) < k.firstBuf(order[j]) })
	query := map[Operand]bool{}
	for _, o := range order {
		query[o] = k.queries(o)
	}
	// A kernel whose rows Workers dealt over calls moves each call's rows
	// into its buffers' offsets, the caller's to add: its indices are then
	// the one call's.
	var coords map[int]bool
	if k.Calls > 0 {
		coords = map[int]bool{}
		for _, a := range k.Args {
			if a.Coord > 0 {
				coords[a.Node-1] = true
			}
		}
	}
	offsets := map[Operand]int{}
	for _, o := range order {
		offsets[o] = k.split(byOp[o], f.Coords != nil, coords)
	}
	code := k.Live()
	var kept []Operand
	for _, o := range order {
		in := written[o]
		for _, a := range byOp[o] {
			in = in || a.node >= 0 && code[a.node]
		}
		if in {
			kept = append(kept, o)
		}
	}
	outs, ins := 0, 0
	acts, weight, weighted, consts := false, false, false, false
	for _, o := range kept {
		if o.Kind == OperandWeight || o.Kind == OperandVector {
			weight = true
		}
	}
	for _, o := range kept {
		if f.Word > 0 && !written[o] && k.oneWord(byOp[o]) {
			for _, a := range byOp[o] {
				k.Bufs[k.Nodes[a.node].Buf].Word = f.Word + 1
				k.Bufs[k.Nodes[a.node].Buf].Offset = offsets[o]
			}
			k.Args = append(k.Args, Arg{Field: f.Word, Buf: k.Nodes[byOp[o][0].node].Buf})
			continue
		}
		var fl int
		switch {
		case written[o] && f.Index > 0 && k.integers(o):
			fl = f.Index
		case written[o]:
			fl = f.Out[min(outs, len(f.Out)-1)]
			if o.Copy == 0 {
				outs++
			} else {
				fl = f.Out[0]
			}
		case (o.Kind == OperandConsts || k.fixedWords(o)) && !consts:
			fl, consts = f.Consts, true
		case (o.Kind == OperandWeight || o.Kind == OperandTable && k.indirect(o, offsets[o]) || narrows) && !weighted:
			fl, weighted = f.Weight, true
		case weight && f.Act > 0 && !acts && o.Kind != OperandVector:
			fl, acts = f.Act, true
		case f.Query > 0 && query[o] && o.Kind != OperandTable:
			fl = f.Query
		default:
			fl = f.In[min(ins, len(f.In)-1)]
			ins++
		}
		if o.Copy > 0 {
			if f2, ok := f.Second[fl]; ok {
				fl = f2
			}
		}
		base := k.add(Node{Op: OpParam, Type: ir.U32, Imm: int64(fl)})
		bufs := map[int]bool{}
		for _, a := range byOp[o] {
			if a.node >= 0 {
				bufs[k.Nodes[a.node].Buf] = true
			} else {
				bufs[k.Stores[a.store].Buf] = true
			}
		}
		first := -1
		for b := range k.Bufs {
			if bufs[b] {
				k.Bufs[b].Base = base + 1
				k.Bufs[b].Offset = offsets[o]
				if first < 0 {
					first = b
				}
			}
		}
		k.Args = append(k.Args, Arg{Field: fl, Buf: first})
	}
}

// oneWord reports whether accesses as are loads of one word at a fixed
// index.
func (k *Kernel) oneWord(as []access) bool {
	ix := -1
	for _, a := range as {
		if a.node < 0 {
			return false
		}
		i := k.Nodes[a.node].A
		if !k.fixed(i) {
			return false
		}
		if ix >= 0 && !sameExpr(k, i, k, ix) {
			return false
		}
		ix = i
	}
	return ix >= 0
}

// integers reports whether the buffers of operand o hold integers.
func (k *Kernel) integers(o Operand) bool {
	for _, b := range k.Bufs {
		if b.Operand == o {
			return b.Elem == ir.I32 || b.Elem == ir.U32
		}
	}
	return false
}

func (k *Kernel) firstBuf(o Operand) int {
	for b, x := range k.Bufs {
		if x.Operand == o {
			return b
		}
	}
	return len(k.Bufs)
}

// split moves the part of the accesses' indices no loop moves, and the
// least constant they share, into an offset the caller adds, 1 + its node
// (0 for none); each access is left a flat index from the buffer's address.
// Where whole is set only the least constant is moved, and a constant
// times one of coords, call coordinates' words, where every access reads
// elements of one size: the kernel computes the rest itself.
func (k *Kernel) split(as []access, whole bool, coords map[int]bool) int {
	g := newAlg(k)
	ls := make([]lin, len(as))
	for i, a := range as {
		if a.node >= 0 {
			nd := k.Nodes[a.node]
			ls[i] = g.flat(nd.Buf, nd.A, nd.Ax)
		} else {
			s := k.Stores[a.store]
			ls[i] = g.flat(s.Buf, s.Ix, s.Ax)
		}
	}
	sized := len(coords) > 0
	for _, a := range as {
		b := k.Bufs[k.accessBuf(a)]
		sized = sized && ElemBytes(b) == ElemBytes(k.Bufs[k.accessBuf(as[0])])
	}
	fixed := lin{}
	for m, v := range ls[0] {
		if m != "" && (!whole && g.still(lin{m: 1}) || whole && sized && g.coordMono(m, coords)) {
			fixed[m] = v
		}
	}
	for _, l := range ls[1:] {
		for m, v := range fixed {
			if l[m] != v {
				delete(fixed, m)
			}
		}
	}
	least := ls[0][""]
	for _, l := range ls[1:] {
		least = min(least, l[""])
	}
	if least != 0 {
		fixed[""] = least
	}
	for i, a := range as {
		ls[i].add(fixed, -1)
		ix := g.build(ls[i])
		if a.node >= 0 {
			k.Nodes[a.node].A, k.Nodes[a.node].Ax = ix, nil
		} else {
			k.Stores[a.store].Ix, k.Stores[a.store].Ax = ix, nil
		}
	}
	if len(fixed) == 0 {
		return 0
	}
	return g.build(fixed) + 1
}

// accessBuf is the buffer access a reads or writes.
func (k *Kernel) accessBuf(a access) int {
	if a.node >= 0 {
		return k.Nodes[a.node].Buf
	}
	return k.Stores[a.store].Buf
}

// coordMono reports whether monomial m is one of coords' nodes alone.
func (g *alg) coordMono(m string, coords map[int]bool) bool {
	ids := monoIDs(m)
	if len(ids) != 1 {
		return false
	}
	a := g.atoms[ids[0]]
	return coords[a.node] && a == atom{node: a.node}
}

// callOffset is node off as a constant and, for each axis of the calls'
// space, the elements a call adds a unit of its coordinate: false where it
// is anything else.
func (k *Kernel) callOffset(off int) (int, []int, bool) {
	axis := map[int]int{}
	for _, a := range k.Args {
		if a.Coord > 0 {
			axis[a.Node-1] = a.Coord - 1
		}
	}
	g := newAlg(k)
	var c int
	var per []int
	for m, v := range g.of(off) {
		if m == "" {
			c = int(v)
			continue
		}
		ids := monoIDs(m)
		if len(ids) != 1 {
			return 0, nil, false
		}
		a := g.atoms[ids[0]]
		ax, ok := axis[a.node]
		if !ok || a != (atom{node: a.node}) {
			return 0, nil, false
		}
		for len(per) <= ax {
			per = append(per, 0)
		}
		per[ax] += int(v)
	}
	return c, per, true
}

// flat is an access's flat index: index ix, or coordinates ax of buffer
// b's shape, its extents the caller's where the shape's dims say so.
func (g *alg) flat(b, ix int, ax []int) lin {
	if ax == nil {
		return g.of(ix)
	}
	bf := g.k.Bufs[b]
	out := lin{}
	stride := lin{"": 1}
	for j := len(ax) - 1; j >= 0; j-- {
		out.add(g.merged(g.times(g.of(ax[j]), stride)), 1)
		ext := lin{"": int64(bf.Shape[j])}
		if j < len(bf.Dims) && bf.Dims[j] > 0 {
			ext = g.of(bf.Dims[j] - 1)
		}
		stride = g.times(stride, ext)
	}
	return g.merged(out)
}

// computedFrom reports whether node i is computed from a node of op.
func (k *Kernel) computedFrom(i int, op Op) bool {
	return k.computedFromSeen(i, op, make([]bool, len(k.Nodes)))
}

// computedFromSeen is computedFrom past the nodes seen, each node asked
// once: a chain whose nodes each read the one before more than once is
// walked in its length, not in two to its length.
func (k *Kernel) computedFromSeen(i int, op Op, seen []bool) bool {
	if seen[i] {
		return false
	}
	seen[i] = true
	nd := k.Nodes[i]
	if nd.Op == op {
		return true
	}
	if nd.Op == OpFold {
		return false
	}
	for _, a := range k.Operands(&nd) {
		if k.computedFromSeen(*a, op, seen) {
			return true
		}
	}
	return false
}

// queries reports whether operand o is read beside the history by a sum
// over the history's width: its loads and the history's both in the step
// of a sum whose loop runs over the history's last coordinate.
func (k *Kernel) queries(o Operand) bool {
	for f, nd := range k.Nodes {
		if nd.Op != OpFold {
			continue
		}
		var reads, history bool
		var walk func(i int)
		seen := map[int]bool{}
		walk = func(i int) {
			if seen[i] || i == f {
				return
			}
			seen[i] = true
			x := k.Nodes[i]
			if x.Op == OpLoad {
				b := k.Bufs[x.Buf]
				reads = reads || b.Operand == o
				history = history || b.Half && len(x.Ax) > 0 && k.Varies(x.Ax[len(x.Ax)-1], nd.At)
			}
			if x.Op == OpFold {
				return
			}
			for _, a := range k.Operands(&x) {
				walk(*a)
			}
		}
		walk(nd.A)
		if reads && history {
			return true
		}
	}
	return false
}

// splitFrom is the length node i splits into trips: i with its shifts,
// masks and constant multiples taken off.
func (k *Kernel) splitFrom(i int) int {
	for {
		nd := k.Nodes[i]
		switch nd.Op {
		case OpShr, OpShl, OpAnd, OpMul:
			if k.Nodes[nd.B].Op == OpConst {
				i = nd.A
				continue
			}
		}
		return i
	}
}

// indirect reports whether the kernel reads operand o at an address made
// from a value it loads or from the span of a page it is called for, in
// its indices or in off, 1 + the node the caller adds to them: a paged
// history.
func (k *Kernel) indirect(o Operand, off int) bool {
	var fromLoad func(i int) bool
	fromLoad = func(i int) bool {
		nd := k.Nodes[i]
		if nd.Op == OpLoad || nd.Op == OpSpan {
			return true
		}
		if nd.Op == OpFold {
			return false
		}
		for _, a := range k.Operands(&nd) {
			if fromLoad(*a) {
				return true
			}
		}
		return false
	}
	for _, nd := range k.Nodes {
		if nd.Op != OpLoad || k.Bufs[nd.Buf].Operand != o {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if fromLoad(*a) {
				return true
			}
		}
	}
	return off > 0 && fromLoad(off-1)
}

// fixedWords reports whether operand o is an input of fixed words: a block
// of constants.
func (k *Kernel) fixedWords(o Operand) bool {
	for _, b := range k.Bufs {
		if b.Operand == o && b.Words != nil {
			return true
		}
	}
	return false
}
