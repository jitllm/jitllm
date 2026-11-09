package composer

import (
	"maps"
	"slices"
	"sort"

	"github.com/jitllm/jitllm/composer/ir"
)

// Order writes a kernel as the sequence its code runs: the argument words
// first, in the order of their fields but a vector loop's counts last, a
// value no loop moves before the first loop that needs it, and in each loop
// its values in the order they depend on each other, each inner loop where
// a value first needs what it carries, and its stores; the sums a loop
// carries start just before it, as does a zero its tail keeps lanes with,
// its stores before its sums' steps, and a
// choice another sum carries a value
// with steps once the trip's other reads of it have run. A vector's lanes
// rearranged are made just after it is loaded. A float zero is made where
// it is read. A pair's heads run statement by statement
// side by side, each head's statements from where it reads a finished sum
// on run whole, head after head.
type Order struct{}

func (Order) Name() string { return "order" }

func (p Order) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Args == nil || k.Seq != nil {
			return nil, false, nil
		}
		m := k.clone()
		o := orderer{k: m, done: map[int]bool{}, loopDone: map[int]bool{}, home: map[int]int{}, depths: map[int]int{}, words: m.VectorWords()}
		o.run()
		m.Seq = m.interleave(o.seq)
		return []*Kernel{m}, true, nil
	})
}

type orderer struct {
	k        *Kernel
	seq      []Entry
	done     map[int]bool
	loopDone map[int]bool
	home     map[int]int
	depths   map[int]int
	words    map[int]bool
	// after is, for a fold of the loop being written that other values of
	// its trip read, those values: its step runs once they have.
	after   map[int][]int
	scopeOf int
	// memo is evalArgs' depths for one call, cleared at each (args): what is
	// made changes as need recurses, so no depth outlives its call.
	memo map[int]int
	// twins is each two-operand node's key (twinKey) to the nodes sharing
	// it, in node order: need's candidates for a node's twin.
	twins map[twinKey][]int
	// readers is each node's readers, the nodes then the stores reading it
	// (reads), and countsOf each loop's counts (counts): the nodes, loops
	// and homes they come from are fixed while the orderer runs.
	readers  *readerIndex
	countsOf map[int]map[int]bool
}

// readerIndex is, for each node, the nodes and the stores reading it.
type readerIndex struct {
	nodes, stores [][]int
}

// twinKey is a two-operand node's operation and operands.
type twinKey struct {
	op      Op
	a, b, c int
}

// args is evalArgs over nd with o's scratch memo.
func (o *orderer) args(nd *Node) []*int {
	if o.memo == nil {
		o.memo = map[int]int{}
	}
	clear(o.memo)
	return o.k.evalArgs(nd, o.memo, o.made)
}

// twinsOf is the nodes with nd's operation and operands, in node order.
func (o *orderer) twinsOf(nd *Node) []int {
	if o.twins == nil {
		o.twins = map[twinKey][]int{}
		for j, x := range o.k.Nodes {
			if arity(x.Op) == 2 {
				k := twinKey{x.Op, x.A, x.B, x.C}
				o.twins[k] = append(o.twins[k], j)
			}
		}
	}
	return o.twins[twinKey{nd.Op, nd.A, nd.B, nd.C}]
}

func (o *orderer) run() {
	k := o.k
	live := k.Live()
	vecCount := map[int]bool{}
	for _, l := range k.Loops {
		if l.Count > 0 && (l.Lanes > 0 || o.isTail(l)) {
			vecCount[l.Count-1] = true
		}
	}
	var params, late []int
	made := map[int][]int{}
	for i, nd := range k.Nodes {
		if nd.Op != OpParam || !live[i] && !o.isBase(i) {
			continue
		}
		if ms := k.madeCounts(i); ms != nil {
			made[i] = ms
			late = append(late, i)
		} else if vecCount[i] {
			late = append(late, i)
		} else {
			params = append(params, i)
		}
	}
	consts := func(i int) bool {
		for _, b := range k.Bufs {
			if b.Base == i+1 && b.Operand.Kind == OperandConsts {
				return true
			}
		}
		return false
	}
	sort.SliceStable(params, func(a, b int) bool {
		if ca, cb := consts(params[a]), consts(params[b]); ca != cb {
			return cb
		}
		return k.Nodes[params[a]].Imm < k.Nodes[params[b]].Imm
	})
	sort.SliceStable(late, func(a, b int) bool { return o.countLoop(late[a]) < o.countLoop(late[b]) })
	for _, i := range append(params, late...) {
		o.node(i)
		for _, m := range made[i] {
			o.node(m)
		}
	}
	hs := k.Homes()
	for i := range k.Nodes {
		if live[i] {
			o.home[i] = hs[i]
		}
	}
	o.scope(-1)
}

func (o *orderer) isTail(l Loop) bool {
	for _, v := range o.k.Loops {
		if v.Lanes > 0 && v.Parent == l.Parent && v.Range == l.Range && v.Count > 0 && l.Count > 0 {
			return true
		}
	}
	return false
}

func (o *orderer) isBase(i int) bool {
	for _, b := range o.k.Bufs {
		if b.Base == i+1 {
			return true
		}
	}
	return false
}

// madeCounts is, for a word every reader of which is the count of a vector
// loop or of its tail and read by nothing else, those counts, the vector
// loop's last; nil for any other node.
func (k *Kernel) madeCounts(i int) []int {
	var tails, vectors []int
	var buf [3]int
	for j := range k.Nodes {
		direct := false
		for _, a := range k.args(j, &buf) {
			direct = direct || a == i
		}
		if !direct {
			continue
		}
		c := k.countedBy(j)
		if c < 0 || k.readCount(j) > 0 {
			return nil
		}
		switch {
		case k.Loops[c].Lanes > 0:
			vectors = append(vectors, j)
		case k.vectorTail(c):
			tails = append(tails, j)
		default:
			return nil
		}
	}
	if len(vectors) != 1 {
		return nil
	}
	return append(tails, vectors...)
}

// countedBy is the one loop node j is the count of, -1 for none.
func (k *Kernel) countedBy(j int) int {
	c := -1
	for d, l := range k.Loops {
		if l.Count == j+1 {
			if c >= 0 {
				return -1
			}
			c = d
		}
	}
	return c
}

func (o *orderer) countLoop(i int) int {
	for c, l := range o.k.Loops {
		if l.Count == i+1 {
			return c
		}
	}
	return len(o.k.Loops)
}

func (o *orderer) node(i int) {
	if o.done[i] {
		return
	}
	o.done[i] = true
	o.seq = append(o.seq, Entry{Kind: EntryNode, I: i})
	if o.k.Nodes[i].Op == OpLoad {
		for j, nd := range o.k.Nodes {
			if (nd.Op == OpDupEven || nd.Op == OpDupOdd || nd.Op == OpSwapPairs) && nd.A == i && o.home[j] == o.home[i] {
				o.node(j)
			}
		}
	}
	for f, rs := range o.after {
		ready := true
		for _, r := range rs {
			ready = ready && o.done[r]
		}
		if ready {
			delete(o.after, f)
			o.need(o.k.Nodes[f].A, o.scopeOf)
		}
	}
}

// need emits node i and what it reads, within scope c: a value of an inner
// loop emits that loop; an operation's twin, the same operation on the same
// operands with another immediate, right after it.
func (o *orderer) need(i, c int) {
	if o.done[i] {
		return
	}
	k := o.k
	nd := k.Nodes[i]
	if o.unmade(i) {
		return
	}
	if nd.Op == OpFold {
		if nd.At != c {
			o.loop(o.childOf(nd.At, c))
			return
		}
		o.done[i] = true
		return
	}
	if h := o.home[i]; h != c && k.within(h, c) && h >= 0 {
		o.loop(o.childOf(h, c))
		return
	}
	if nd.Op == OpZip || nd.Op == OpUnpack {
		o.stages(nd.A, nd.B, c)
	}
	for _, a := range o.args(&nd) {
		switch k.Nodes[*a].Op {
		case OpAnd, OpAndNot, OpOr, OpXor:
			for _, b := range o.args(&k.Nodes[*a]) {
				if !o.constLoad(*b) {
					o.need(*b, c)
				}
			}
		}
	}
	for _, a := range o.args(&nd) {
		o.need(*a, c)
	}
	if nd.Share > 0 {
		o.beforeShare(i, c)
	}
	o.node(i)
	if arity(nd.Op) == 2 {
		for _, j := range o.twinsOf(&nd) {
			if j != i && !o.done[j] && k.Nodes[j].Imm != nd.Imm && o.home[j] == c {
				o.node(j)
			}
		}
	}
}

// made is what evalArgs orders operands by as leaves: a value already made
// and a load of constants. The operands of an operation's bitwise operands
// are made first, then the operands by what is left to make of them.
func (o *orderer) made(i int) bool { return o.done[i] || o.constLoad(i) }

// unmade reports whether node i is a value no entry makes: a constant the
// code takes as an immediate, or an iteration.
func (o *orderer) unmade(i int) bool {
	nd := o.k.Nodes[i]
	return nd.Op == OpConst && !o.words[i] && !o.k.isZero(i) || nd.Op == OpIter
}

// constLoad reports whether node i loads a constant from the constant
// block.
func (o *orderer) constLoad(i int) bool {
	nd := o.k.Nodes[i]
	return nd.Op == OpLoad && o.k.Bufs[nd.Buf].Operand.Kind == OperandConsts
}

// beforeShare makes, in scope c, the readers of a value made earlier with
// node i's Share label whose operands are all made: i takes that value's
// register.
func (o *orderer) beforeShare(i, c int) {
	k := o.k
	for x, xn := range k.Nodes {
		if x == i || xn.Share != k.Nodes[i].Share || !o.done[x] {
			continue
		}
		for r := range k.Nodes {
			if o.done[r] || r == i || o.home[r] != c {
				continue
			}
			reads, ready := false, true
			for _, a := range k.Operands(&k.Nodes[r]) {
				reads = reads || *a == x
				ready = ready && (o.done[*a] || k.Nodes[*a].Op == OpConst && o.unmade(*a))
			}
			if reads && ready {
				o.node(r)
			}
		}
	}
}

// childOf is the loop directly inside c that holds loop h.
func (o *orderer) childOf(h, c int) int {
	for h >= 0 && o.k.Loops[h].Parent != c {
		h = o.k.Loops[h].Parent
	}
	return h
}

// scope emits the values homed in loop c that its inner loops need, the
// inner loops in order, then c's stores.
func (o *orderer) scope(c int) {
	k := o.k
	var inner []int
	for d, l := range k.Loops {
		if l.Parent == c {
			inner = append(inner, d)
		}
	}
	for _, d := range inner {
		for _, f := range k.foldsOf(d) {
			if b := k.Nodes[f].B; k.Nodes[b].Op != OpFold && !k.startsOnly(b) && !k.isZero(b) {
				o.need(b, c)
			}
		}
	}
	for _, d := range inner {
		if t := o.tailOf(d); t >= 0 {
			o.startsTogether(c, d, t)
		}
		for _, i := range o.byFirstRead(d, func(i int) bool {
			h, ok := o.home[i]
			return ok && h == c && k.Nodes[i].Op != OpFold && !k.isZero(i) && o.reads(d, i) && !o.counts(d)[i]
		}) {
			o.need(i, c)
		}
		o.countsFor(c, d)
		o.loop(d)
	}
	outer, scope := o.after, o.scopeOf
	o.after, o.scopeOf = map[int][]int{}, c
	carried := map[int]bool{}
	for _, f := range k.foldsOf(c) {
		if v := k.carriedBy(f); v >= 0 {
			carried[v] = true
		}
	}
	for _, f := range k.foldsOf(c) {
		// A sum a general register holds is stepped in its register, with
		// no copy aside for a read after the step: its other reads go first.
		if !carried[f] && (k.Nodes[f].Type == ir.F32 || o.words[f]) {
			continue
		}
		step := k.Nodes[f].A
		for i, nd := range k.Nodes {
			if h, ok := o.home[i]; !ok || h != c || i == step || o.done[i] {
				continue
			}
			for _, a := range k.Operands(&nd) {
				if *a == f {
					o.after[f] = append(o.after[f], i)
					break
				}
			}
		}
	}
	for i, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		for _, a := range k.storeArgs(&s) {
			o.need(*a, c)
		}
		if c >= 0 {
			o.seq = append(o.seq, Entry{Kind: EntryStore, I: i})
		}
	}
	if c < 0 {
		for i, s := range k.Stores {
			if s.At == c && !s.Barrier && !s.Run {
				o.seq = append(o.seq, Entry{Kind: EntryStore, I: i})
			}
		}
	}
	var steps []int
	for _, f := range k.foldsOf(c) {
		if _, wait := o.after[f]; !wait {
			steps = append(steps, f)
		}
	}
	o.lockstep(steps, c)
	// A sum still waiting steps once the trip's other reads of it are made.
	for _, f := range slices.Sorted(maps.Keys(o.after)) {
		for _, r := range o.after[f] {
			o.need(r, c)
		}
		o.need(k.Nodes[f].A, c)
	}
	o.after, o.scopeOf = outer, scope
}

// lockstep makes the steps of folds homed in loop c. Where each step is a
// chain of the same length through its fold -- one link where a sum steps
// once a trip, one for each position of a period (Periods) -- and the
// chains read a value they compute alike, the chains advance link by link
// together: what a position
// computes for every sum is made and read by each before the next
// position's, rather than held across one sum's whole chain.
func (o *orderer) lockstep(folds []int, c int) {
	k := o.k
	chains := make([][]int, len(folds))
	ok := len(folds) > 1
	for x, f := range folds {
		if !ok {
			break
		}
		chains[x] = o.chain(f, c)
		ok = len(chains[x]) > 0 && len(chains[x]) == len(chains[0])
	}
	if ok && o.shareComputed(folds, c) {
		xs := make([]int, len(chains))
		for l := range chains[0] {
			for x, ch := range chains {
				xs[x] = ch[l]
			}
			o.together(xs, c)
		}
	}
	for _, f := range folds {
		o.need(k.Nodes[f].A, c)
	}
}

// together makes nodes xs, the same operation at one place of several
// sums' steps, side by side: their operands at each place made together
// first, in the order need makes the first's, so a value they share is read
// by each as soon as it is made. Nodes that are not the same operation are
// made one after another.
func (o *orderer) together(xs []int, c int) {
	k := o.k
	same, alike := true, true
	x0 := k.Nodes[xs[0]]
	for _, x := range xs {
		nd := k.Nodes[x]
		same = same && x == xs[0]
		alike = alike && !o.done[x] && o.inLoop(x, c) && nd.Op == x0.Op && nd.Type == x0.Type && nd.Imm == x0.Imm && nd.At == x0.At && nd.Op != OpZip && nd.Op != OpUnpack && arity(nd.Op) > 0
	}
	if same || !alike {
		for _, x := range xs {
			o.need(x, c)
		}
		return
	}
	nds := make([]Node, len(xs))
	ops := make([][]*int, len(xs))
	for j, x := range xs {
		nds[j] = k.Nodes[x]
		ops[j] = k.Operands(&nds[j])
	}
	place := func(p *int) int {
		for q, a := range ops[0] {
			if a == p {
				return q
			}
		}
		return -1
	}
	at := func(q int) []int {
		ys := make([]int, len(xs))
		for j := range xs {
			if q >= len(ops[j]) {
				return nil
			}
			ys[j] = *ops[j][q]
		}
		return ys
	}
	// A value every sum reads at this place is made after the places
	// where each reads its own, just before they read it. A place whose
	// sums share fewer than two values still to make is made a sum after
	// another, as the sums are read: side by side, each sum would hold
	// its own value where one shared value is all a sum after another
	// holds.
	args := o.args(&nds[0])
	for _, shared := range []bool{false, true} {
		for _, a := range args {
			if ys := at(place(a)); ys != nil && !o.leaves(ys) && one(ys) == shared && (shared || o.sharedToMake(ys, c) > 1) {
				o.together(ys, c)
			}
		}
	}
	for _, x := range xs {
		o.need(x, c)
	}
}

// sharedToMake is the values of loop c not yet made that every one of xs
// computes from and each reads apart, in a value of its own.
func (o *orderer) sharedToMake(xs []int, c int) int {
	k := o.k
	count := map[int]int{}
	for _, x := range xs {
		seen := map[int]bool{}
		var walk func(i int)
		walk = func(i int) {
			if seen[i] || !o.inLoop(i, c) {
				return
			}
			seen[i] = true
			count[i]++
			for _, a := range k.Operands(&k.Nodes[i]) {
				walk(*a)
			}
		}
		walk(x)
	}
	read := map[int]bool{}
	for i, m := range count {
		if m == len(xs) {
			continue
		}
		for _, a := range k.Operands(&k.Nodes[i]) {
			if count[*a] == len(xs) {
				read[*a] = true
			}
		}
	}
	return len(read)
}

// one reports whether xs are all one node.
func one(xs []int) bool {
	for _, x := range xs {
		if x != xs[0] {
			return false
		}
	}
	return true
}

// leaves reports whether nodes xs, not all one node, are each a load or a
// broadcast of one: each is made just before what reads it.
func (o *orderer) leaves(xs []int) bool {
	k := o.k
	differ := false
	for _, x := range xs {
		nd := k.Nodes[x]
		if nd.Op == OpBroadcast {
			nd = k.Nodes[nd.A]
		}
		if nd.Op != OpLoad {
			return false
		}
		differ = differ || x != xs[0]
	}
	return differ
}

// inLoop reports whether node i is a value of loop c the orderer has yet
// to make, other than a fold.
func (o *orderer) inLoop(i, c int) bool {
	h, ok := o.home[i]
	return ok && h == c && o.k.Nodes[i].Op != OpFold && !o.done[i]
}

// chain is fold f's step as the nodes from the one reading f to the step,
// each the one operand of the next that reaches f; nil where a node reads f
// through two operands.
func (o *orderer) chain(f, c int) []int {
	k := o.k
	reach := map[int]bool{}
	var reaches func(i int) bool
	reaches = func(i int) bool {
		if i == f {
			return true
		}
		if v, ok := reach[i]; ok {
			return v
		}
		reach[i] = false
		if !o.inLoop(i, c) {
			return false
		}
		for _, a := range k.Operands(&k.Nodes[i]) {
			if reaches(*a) {
				reach[i] = true
				return true
			}
		}
		return false
	}
	var out []int
	for x := k.Nodes[f].A; x != f; {
		if !reaches(x) {
			return nil
		}
		out = append(out, x)
		next := -1
		for _, a := range k.Operands(&k.Nodes[x]) {
			if reaches(*a) {
				if next >= 0 && next != *a {
					return nil
				}
				next = *a
			}
		}
		x = next
	}
	slices.Reverse(out)
	return out
}

// shareComputed reports whether two of the folds' steps, homed in loop c,
// read one computed value or one word of a weight: not another load, a
// broadcast of one or a constant.
func (o *orderer) shareComputed(folds []int, c int) bool {
	k := o.k
	owner := map[int]int{}
	for x, f := range folds {
		seen := map[int]bool{}
		var walk func(i int) bool
		walk = func(i int) bool {
			if seen[i] || !o.inLoop(i, c) {
				return false
			}
			seen[i] = true
			nd := k.Nodes[i]
			leaf := nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind != OperandWeight || nd.Op == OpConst || nd.Op == OpParam || nd.Op == OpIter || nd.Op == OpBroadcast && k.Nodes[nd.A].Op == OpLoad
			if !leaf {
				if w, ok := owner[i]; ok && w != x {
					return true
				}
				owner[i] = x
			}
			for _, a := range k.Operands(&nd) {
				if walk(*a) {
					return true
				}
			}
			return false
		}
		if walk(k.Nodes[f].A) {
			return true
		}
	}
	return false
}

// reads reports whether loop d's code reads node i: a node homed in d or a
// loop inside it, or one of d's stores, reaching it.
func (o *orderer) reads(d, i int) bool {
	k := o.k
	if o.readers == nil {
		r := &readerIndex{nodes: make([][]int, len(k.Nodes)), stores: make([][]int, len(k.Nodes))}
		for j := range k.Nodes {
			for _, a := range k.Operands(&k.Nodes[j]) {
				r.nodes[*a] = append(r.nodes[*a], j)
			}
		}
		for si := range k.Stores {
			for _, a := range k.storeArgs(&k.Stores[si]) {
				r.stores[*a] = append(r.stores[*a], si)
			}
		}
		o.readers = r
	}
	for _, j := range o.readers.nodes[i] {
		if h, ok := o.home[j]; ok && h >= 0 && k.within(h, d) {
			return true
		}
	}
	for _, si := range o.readers.stores[i] {
		if s := &k.Stores[si]; s.At >= 0 && k.within(s.At, d) && !s.Barrier && !s.Run {
			return true
		}
	}
	return false
}

// counts is the nodes the counts of loop d and the loops in it are made
// from.
func (o *orderer) counts(d int) map[int]bool {
	if out, ok := o.countsOf[d]; ok {
		return out
	}
	if o.countsOf == nil {
		o.countsOf = map[int]map[int]bool{}
	}
	k := o.k
	out := map[int]bool{}
	o.countsOf[d] = out
	var walk func(i int)
	walk = func(i int) {
		if out[i] {
			return
		}
		out[i] = true
		nd := k.Nodes[i]
		for _, a := range k.Operands(&nd) {
			walk(*a)
		}
	}
	for e, l := range k.Loops {
		if l.Count > 0 && k.within(e, d) {
			walk(l.Count - 1)
		}
	}
	return out
}

// countsFor emits, in scope c, the counts of loop d and the loops in it,
// in the order of their loops, a count made by one operation from a value a
// later count still reads after that count.
func (o *orderer) countsFor(c, d int) {
	k := o.k
	var cs []int
	for _, e := range k.preorder() {
		if l := k.Loops[e]; l.Count > 0 && k.within(e, d) {
			if n := l.Count - 1; o.home[n] == c && !o.done[n] {
				cs = append(cs, n)
			}
		}
	}
	var front, last []int
	for j, x := range cs {
		n := k.Nodes[x]
		deferred := false
		if arity(n.Op) == 2 && k.Nodes[n.B].Op == OpConst {
			for _, later := range cs[j+1:] {
				if later != x && o.readsNode(later, n.A) {
					deferred = true
				}
			}
		}
		if deferred {
			last = append(last, x)
		} else {
			front = append(front, x)
		}
	}
	for _, n := range append(front, last...) {
		o.need(n, c)
	}
}

// readsNode reports whether node i is computed from node x.
func (o *orderer) readsNode(i, x int) bool {
	if i == x {
		return true
	}
	nd := o.k.Nodes[i]
	if nd.Op == OpFold {
		return false
	}
	for _, a := range o.k.Operands(&nd) {
		if o.readsNode(*a, x) {
			return true
		}
	}
	return false
}

func (o *orderer) loop(d int) {
	if d < 0 || o.loopDone[d] {
		return
	}
	o.loopDone[d] = true
	for _, f := range o.k.foldsOf(d) {
		if o.done[f] {
			continue
		}
		o.done[f] = true
		o.seq = append(o.seq, Entry{Kind: EntryNode, I: f})
	}
	k := o.k
	for i, nd := range k.Nodes {
		if nd.Op != OpFirstLane || o.done[nd.B] || !k.isZero(nd.B) {
			continue
		}
		if t := o.home[i]; t >= 0 && k.Loops[t].Of == d+1 && k.Loops[d].Lanes > 0 {
			o.node(nd.B)
		}
	}
	o.seq = append(o.seq, Entry{Kind: EntryLoop, I: d})
	o.scope(d)
	o.seq = append(o.seq, Entry{Kind: EntryEnd, I: d})
}

// byFirstRead is the nodes keep admits in the order loop d's code first
// reads them: loop d is ordered aside, with them taken as made, and each
// placed by its first reader there.
func (o *orderer) byFirstRead(d int, keep func(int) bool) []int {
	k := o.k
	var out []int
	for i := range k.Nodes {
		if keep(i) {
			out = append(out, i)
		}
	}
	if len(out) < 2 {
		return out
	}
	dry := orderer{k: k, done: maps.Clone(o.done), loopDone: maps.Clone(o.loopDone), home: o.home, depths: map[int]int{}, words: o.words,
		after: map[int][]int{}, scopeOf: o.scopeOf, twins: o.twins, readers: o.readers, countsOf: o.countsOf}
	for _, i := range out {
		dry.done[i] = true
	}
	dry.loop(d)
	first := map[int]int{}
	for _, i := range out {
		first[i] = len(dry.seq)
	}
	for p, e := range dry.seq {
		if e.Kind != EntryNode {
			continue
		}
		for _, a := range k.Operands(&k.Nodes[e.I]) {
			if f, ok := first[*a]; ok && p < f {
				first[*a] = p
			}
		}
	}
	slices.SortStableFunc(out, func(a, b int) int { return first[a] - first[b] })
	return out
}

// tailOf is the loop that continues vector loop d, -1 for none.
func (o *orderer) tailOf(d int) int {
	if o.k.Loops[d].Lanes == 0 {
		return -1
	}
	return o.k.tailOf(d)
}

// startsTogether makes, in scope c, vector loop d's and its tail t's
// starts before d: the values the loops read that earlier code computes,
// then for each loop its sums in their order, each making its start, and
// the constants it reads; a sum the tail carries on from d's stays where
// it is.
func (o *orderer) startsTogether(c, d, t int) {
	k := o.k
	constant := func(i int) bool {
		nd := k.Nodes[i]
		return nd.Op == OpConst || nd.Op == OpLanes || nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind == OperandConsts || k.startsOnly(i)
	}
	reads := func(e int, consts bool) {
		for i := range k.Nodes {
			if h, ok := o.home[i]; ok && h == c && k.Nodes[i].Op != OpFold && !k.isZero(i) && constant(i) == consts && o.reads(e, i) && !o.counts(e)[i] {
				o.need(i, c)
			}
		}
	}
	reads(d, false)
	reads(t, false)
	for _, e := range []int{d, t} {
		for _, f := range k.foldsOf(e) {
			b := k.Nodes[f].B
			if k.Nodes[b].Op == OpFold || o.done[f] {
				continue
			}
			if !k.isZero(b) && !k.startsOnly(b) {
				o.need(b, c)
			}
			o.done[f] = true
			o.seq = append(o.seq, Entry{Kind: EntryNode, I: f})
		}
		reads(e, true)
	}
}

// orderCursors orders each loop's cursors: a buffer the loop stores to
// through its own address first (but in a loop kept at one trip), then in
// the order the sequence first reaches each pointer. A tile's pass, which
// takes every copy before it reaches any, keeps its argument words' order.
func (k *Kernel) orderCursors() {
	first := map[[2]int]int{}
	for pos, e := range k.Seq {
		var b, ptr int
		switch e.Kind {
		case EntryNode:
			nd := k.Nodes[e.I]
			if nd.Op != OpLoad {
				continue
			}
			b, ptr = nd.Buf, nd.Ptr
		case EntryStore:
			s := k.Stores[e.I]
			b, ptr = s.Buf, s.Ptr
		default:
			continue
		}
		key := [2]int{k.Bufs[b].Base, ptr}
		if _, ok := first[key]; !ok {
			first[key] = pos
		}
	}
	for c := range k.Loops {
		if k.Loops[c].Walk {
			continue
		}
		cs := k.Loops[c].Cursors
		rank := func(cu Cursor) (int, int) {
			stored := 1
			for _, s := range k.Stores {
				if s.At == c && !s.Barrier && !s.Run && s.Ptr == 0 && cu.Ptr == 0 && k.Bufs[s.Buf].Base == k.Bufs[cu.Buf].Base && !k.Loops[c].Kept {
					stored = 0
				}
			}
			at, ok := first[[2]int{k.Bufs[cu.Buf].Base, cu.Ptr}]
			if !ok {
				at = len(k.Seq)
			}
			return stored, at
		}
		sort.SliceStable(cs, func(i, j int) bool {
			si, ai := rank(cs[i])
			sj, aj := rank(cs[j])
			if si != sj {
				return si < sj
			}
			return ai < aj
		})
	}
}

// interleave puts each statement of a pair's later head beside its first
// head's twin, but after the run of finishing statements -- those reading a
// sum after its loop -- a twin belongs to.
func (k *Kernel) interleave(seq []Entry) []Entry {
	cp := k.copies()
	later := false
	for _, c := range cp {
		later = later || c > 0
	}
	if !later {
		return seq
	}
	tw := k.twins(cp)
	of := map[int][]int{}
	for i, t := range tw {
		if t >= 0 {
			of[t] = append(of[t], i)
		}
	}
	for t := range of {
		sort.Slice(of[t], func(a, b int) bool { return cp[of[t][a]] < cp[of[t][b]] })
	}
	storeOf := map[int][]int{}
	for i, s := range k.Stores {
		if s.Barrier || s.Run || k.Bufs[s.Buf].Operand.Copy == 0 {
			continue
		}
		for j, o := range k.Stores {
			if k.Bufs[o.Buf].Operand.Copy == 0 && !o.Barrier && !o.Run && tw[s.Val] == o.Val && o.At == s.At {
				storeOf[j] = append(storeOf[j], i)
				break
			}
		}
	}
	hs := k.Homes()
	finishing := map[int]bool{}
	var fin func(i int) bool
	fin = func(i int) bool {
		if v, ok := finishing[i]; ok {
			return v
		}
		finishing[i] = false
		nd := k.Nodes[i]
		if nd.Op == OpFold {
			return false
		}
		v := false
		for _, a := range k.Operands(&nd) {
			x := k.Nodes[*a]
			if x.Op == OpFold && x.At != hs[i] && !(hs[i] >= 0 && k.within(hs[i], x.At)) || fin(*a) {
				v = true
			}
		}
		finishing[i] = v
		return v
	}
	isLater := func(e Entry) bool {
		switch e.Kind {
		case EntryNode:
			return cp[e.I] > 0
		case EntryStore:
			s := k.Stores[e.I]
			return !s.Barrier && !s.Run && k.Bufs[s.Buf].Operand.Copy > 0
		}
		return false
	}
	isFinish := func(e Entry) bool {
		switch e.Kind {
		case EntryNode:
			return fin(e.I)
		case EntryStore:
			s := k.Stores[e.I]
			return !s.Barrier && !s.Run && fin(s.Val)
		}
		return false
	}
	var out, pending []Entry
	flush := func() {
		out = append(out, pending...)
		pending = nil
	}
	for _, e := range seq {
		if isLater(e) {
			continue
		}
		if len(pending) > 0 && !isFinish(e) {
			flush()
		}
		out = append(out, e)
		var copies []Entry
		switch e.Kind {
		case EntryNode:
			for _, c := range of[e.I] {
				copies = append(copies, Entry{Kind: EntryNode, I: c})
			}
		case EntryStore:
			for _, c := range storeOf[e.I] {
				copies = append(copies, Entry{Kind: EntryStore, I: c})
			}
		}
		if isFinish(e) {
			pending = append(pending, copies...)
		} else {
			out = append(out, copies...)
		}
	}
	flush()
	return out
}

// startsOnly reports whether node i is a load, at the buffer's own address
// or of constants, only sums start from: the sum's start loads it.
func (k *Kernel) startsOnly(i int) bool {
	if nd := k.Nodes[i]; nd.Op != OpLoad || nd.Ptr > 0 && k.Bufs[nd.Buf].Operand.Kind != OperandConsts {
		return false
	}
	for j, x := range k.Nodes {
		for _, a := range k.Operands(&x) {
			if *a == i && !(x.Op == OpFold && x.B == i && j != i) {
				return false
			}
		}
	}
	return true
}

// isZero reports whether node i is a float zero, made where it is read.
func (k *Kernel) isZero(i int) bool {
	nd := k.Nodes[i]
	return nd.Op == OpConst && nd.Type == ir.F32 && nd.Imm == 0
}

// constsBeforeSiblings loads, outside every loop, a constant that one
// operation reads before the values the operation's reader reads beside
// it that were made after the operation's other operands.
func (k *Kernel) constsBeforeSiblings() {
	depth := make([]int, len(k.Seq))
	d := 0
	pos := map[int]int{}
	for p, e := range k.Seq {
		switch e.Kind {
		case EntryLoop:
			d++
		case EntryEnd:
			d--
		case EntryNode:
			pos[e.I] = p
		}
		depth[p] = d
	}
	readers := func(i int) []int {
		var out []int
		for j := range k.Nodes {
			for _, a := range k.Operands(&k.Nodes[j]) {
				if *a == i {
					out = append(out, j)
					break
				}
			}
		}
		return out
	}
	for p := 0; p < len(k.Seq); p++ {
		e := k.Seq[p]
		if e.Kind != EntryNode || depth[p] != 0 {
			continue
		}
		nd := k.Nodes[e.I]
		if nd.Op != OpLoad || k.Bufs[nd.Buf].Operand.Kind != OperandConsts {
			continue
		}
		rs := readers(e.I)
		if len(rs) != 1 {
			continue
		}
		ps := readers(rs[0])
		if len(ps) != 1 {
			continue
		}
		after := -1
		for _, a := range k.Operands(&k.Nodes[rs[0]]) {
			if q, ok := pos[*a]; ok && *a != e.I {
				after = max(after, q)
			}
		}
		first := p
		for _, a := range k.Operands(&k.Nodes[ps[0]]) {
			if q, ok := pos[*a]; ok && *a != rs[0] && q < first && q > after && depth[q] == 0 {
				first = q
			}
		}
		if first < p {
			copy(k.Seq[first+1:p+1], k.Seq[first:p])
			k.Seq[first] = e
			for q, x := range k.Seq {
				if x.Kind == EntryNode {
					pos[x.I] = q
				}
			}
		}
	}
}

// constRuns loads, outside every loop, the constants of a run of
// operations that each read one constant just loaded before the run.
func (k *Kernel) constRuns() {
	isConst := func(e Entry) bool {
		if e.Kind != EntryNode {
			return false
		}
		nd := k.Nodes[e.I]
		return nd.Op == OpLoad && k.Bufs[nd.Buf].Operand.Kind == OperandConsts
	}
	reads := func(e Entry, c int) bool {
		if e.Kind != EntryNode {
			return false
		}
		for _, a := range k.Operands(&k.Nodes[e.I]) {
			if *a == c {
				return true
			}
		}
		return false
	}
	depth := 0
	for p := 0; p+3 < len(k.Seq); p++ {
		switch k.Seq[p].Kind {
		case EntryLoop:
			depth++
			continue
		case EntryEnd:
			depth--
			continue
		}
		if depth > 0 || !isConst(k.Seq[p]) || !reads(k.Seq[p+1], k.Seq[p].I) {
			continue
		}
		run := p
		for q := p + 2; q+1 < len(k.Seq) && isConst(k.Seq[q]) && reads(k.Seq[q+1], k.Seq[q].I) && reads(k.Seq[q+1], k.Seq[q-1].I); q += 2 {
			c := k.Seq[q]
			copy(k.Seq[run+2:q+1], k.Seq[run+1:q])
			k.Seq[run+1] = c
			run++
		}
	}
}

// readCount is how many nodes and stores read node i.
func (k *Kernel) readCount(i int) int {
	n := 0
	var buf [3]int
	for j := range k.Nodes {
		for _, a := range k.args(j, &buf) {
			if a == i {
				n++
			}
		}
	}
	for j := range k.Stores {
		for _, a := range k.storeArgs(&k.Stores[j]) {
			if *a == i {
				n++
			}
		}
	}
	return n
}

// stages makes the two values a zip interleaves stage by stage: from the
// deepest pair of like operations down their paths, below the first
// through values each read once, each pair's values in the order they were made in the
// decomposition.
func (o *orderer) stages(a, b, c int) {
	k := o.k
	var pairs [][2]int
	for !o.done[a] && !o.done[b] && a != b && k.Nodes[a].Op == k.Nodes[b].Op {
		pairs = append(pairs, [2]int{a, b})
		na, nb := k.Nodes[a], k.Nodes[b]
		if len(pairs) > 1 && (k.uses(a) != 1 || k.uses(b) != 1) {
			break
		}
		next := false
		for j, x := range k.Operands(&na) {
			y := *k.Operands(&nb)[j]
			if *x != y && !o.done[*x] && !o.done[y] && k.Nodes[*x].Op == k.Nodes[y].Op && arity(k.Nodes[*x].Op) > 0 {
				a, b, next = *x, y, true
				break
			}
		}
		if !next {
			break
		}
	}
	for p := len(pairs) - 1; p >= 0; p-- {
		x, y := pairs[p][0], pairs[p][1]
		if y < x {
			x, y = y, x
		}
		o.need(x, c)
		o.need(y, c)
	}
}
