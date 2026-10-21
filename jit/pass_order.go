package jit

import "sort"

// Order writes a kernel as the sequence its code runs: the argument words
// first, in the order of their fields but a vector loop's counts last, a
// value no loop moves before the first loop that needs it, and in each loop
// its values in the order they depend on each other, each inner loop where
// a value first needs what it carries, and its stores.
type Order struct{}

func (Order) Name() string { return "order" }

func (p Order) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Args == nil || k.Seq != nil {
			return nil, false, nil
		}
		m := k.clone()
		o := orderer{k: m, done: map[int]bool{}, loopDone: map[int]bool{}, home: map[int]int{}}
		o.run()
		m.Seq = o.seq
		return []*Kernel{m}, true, nil
	})
}

type orderer struct {
	k        *Kernel
	seq      []Entry
	done     map[int]bool
	loopDone map[int]bool
	home     map[int]int
}

func (o *orderer) run() {
	k := o.k
	live := k.live()
	vecCount := map[int]bool{}
	for _, l := range k.Loops {
		if l.Count > 0 && (l.Lanes > 0 || o.isTail(l)) {
			vecCount[l.Count-1] = true
		}
	}
	var params, late []int
	for i, nd := range k.Nodes {
		if nd.Op != OpParam || !live[i] && !o.isBase(i) {
			continue
		}
		if vecCount[i] {
			late = append(late, i)
		} else {
			params = append(params, i)
		}
	}
	sort.SliceStable(params, func(a, b int) bool { return k.Nodes[params[a]].Imm < k.Nodes[params[b]].Imm })
	sort.SliceStable(late, func(a, b int) bool { return o.countLoop(late[a]) < o.countLoop(late[b]) })
	for _, i := range append(params, late...) {
		o.node(i)
	}
	for i := range k.Nodes {
		if live[i] {
			o.home[i] = k.loopOf(i)
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

func (o *orderer) countLoop(i int) int {
	for c, l := range o.k.Loops {
		if l.Count == i+1 {
			return c
		}
	}
	return len(o.k.Loops)
}

func (o *orderer) node(i int) {
	if !o.done[i] {
		o.done[i] = true
		o.seq = append(o.seq, Entry{Kind: EntryNode, I: i})
	}
}

// need emits node i and what it reads, within scope c: a value of an inner
// loop emits that loop.
func (o *orderer) need(i, c int) {
	if o.done[i] {
		return
	}
	k := o.k
	nd := k.Nodes[i]
	if nd.Op == OpConst || nd.Op == OpIter {
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
	for _, a := range k.args(&nd) {
		o.need(*a, c)
	}
	o.node(i)
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
		for i := range k.Nodes {
			if h, ok := o.home[i]; ok && h == c && o.reads(d, i) {
				o.need(i, c)
			}
		}
		o.loop(d)
	}
	for i, s := range k.Stores {
		if s.At != c || s.Barrier || s.Run {
			continue
		}
		for _, a := range k.storeArgs(&s) {
			o.need(*a, c)
		}
		o.seq = append(o.seq, Entry{Kind: EntryStore, I: i})
	}
}

// reads reports whether loop d's code reads node i: a node homed in d or a
// loop inside it, or one of d's stores, reaching it.
func (o *orderer) reads(d, i int) bool {
	k := o.k
	for j, nd := range k.Nodes {
		if h, ok := o.home[j]; ok && h >= 0 && k.within(h, d) {
			for _, a := range k.args(&nd) {
				if *a == i {
					return true
				}
			}
		}
	}
	for _, s := range k.Stores {
		if s.At >= 0 && k.within(s.At, d) && !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				if *a == i {
					return true
				}
			}
		}
	}
	return false
}

func (o *orderer) loop(d int) {
	if d < 0 || o.loopDone[d] {
		return
	}
	o.loopDone[d] = true
	o.seq = append(o.seq, Entry{Kind: EntryLoop, I: d})
	o.scope(d)
	o.seq = append(o.seq, Entry{Kind: EntryEnd, I: d})
}
