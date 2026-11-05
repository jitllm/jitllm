package grammar

import (
	"slices"
	"unsafe"
)

// The automaton. A stack is the positions still to match, innermost on top;
// stacks are interned as a persistent list (node: top position and the node
// below it), so equal stacks are one id and a set of stacks is a sorted list
// of ids. A set is interned too, and a step from a set by a code point is
// cached: a grammar whose output revisits its states -- JSON does, every
// field -- stops computing after the first visits.
//
// After expansion every stack's top is a character element, or the stack is
// empty (node 0), which is the grammar complete.

type node struct{ pos, below int32 }

// automaton is the grammar's states as far as they have been reached. Not
// safe for concurrent use: Matcher locks around it.
type automaton struct {
	g      *Grammar
	nodes  []node
	nodeID map[node]int32
	// sets are interned sets of nodes; set 0 is the empty set, dead.
	sets  [][]int32
	setID map[string]int32
	// ascii[s][c] is set s stepped by code point c < 128, -1 not yet known;
	// other code points are in wide.
	ascii [][]int32
	wide  map[[2]int32]int32

	// Scratch for expansion: a generation-stamped visited mark per node.
	seen    []uint32
	gen     uint32
	scratch []int32
}

func newAutomaton(g *Grammar) *automaton {
	a := &automaton{g: g, nodeID: map[node]int32{}, setID: map[string]int32{}, wide: map[[2]int32]int32{}}
	a.nodes = append(a.nodes, node{-1, -1}) // node 0, the empty stack
	a.intern(nil)                           // set 0, dead
	return a
}

func (a *automaton) push(pos, below int32) int32 {
	k := node{pos, below}
	if id, ok := a.nodeID[k]; ok {
		return id
	}
	id := int32(len(a.nodes))
	a.nodes = append(a.nodes, k)
	a.nodeID[k] = id
	return id
}

// intern is the id of the set of nodes in s, which it sorts.
func (a *automaton) intern(s []int32) int32 {
	slices.Sort(s)
	s = slices.Compact(s)
	key := ""
	if len(s) > 0 {
		key = unsafe.String((*byte)(unsafe.Pointer(&s[0])), 4*len(s))
	}
	if id, ok := a.setID[key]; ok {
		return id
	}
	id := int32(len(a.sets))
	own := slices.Clone(s)
	a.sets = append(a.sets, own)
	if len(own) > 0 {
		key = string(unsafe.String((*byte)(unsafe.Pointer(&own[0])), 4*len(own)))
	}
	a.setID[key] = id
	row := make([]int32, 128)
	for i := range row {
		row[i] = -1
	}
	a.ascii = append(a.ascii, row)
	return id
}

// at is the stack whose top is position p, over below: the stack itself, or
// below when p ends its sequence (a finished rule pops).
func (a *automaton) at(p, below int32) int32 {
	if k := a.g.el[p].kind; k == eEnd || k == eAlt {
		return below
	}
	return a.push(p, below)
}

// expand adds to a.scratch every stack reachable from stack n by entering
// rules and popping finished ones, stopping at a character element on top
// or the empty stack.
func (a *automaton) expand(n int32) {
	for int(n) >= len(a.seen) {
		a.seen = append(a.seen, 0)
	}
	if a.seen[n] == a.gen {
		return
	}
	a.seen[n] = a.gen
	if n == 0 {
		a.scratch = append(a.scratch, 0)
		return
	}
	nd := a.nodes[n]
	e := a.g.el[nd.pos]
	if e.kind == eChar {
		a.scratch = append(a.scratch, n)
		return
	}
	// A reference: the rest of this sequence goes under each alternative.
	rest := a.at(nd.pos+1, nd.below)
	for _, alt := range a.g.alts[e.arg] {
		a.expand(a.at(alt, rest))
	}
}

func (a *automaton) begin() {
	a.gen++
	if a.gen == 0 {
		clear(a.seen)
		a.gen = 1
	}
	a.scratch = a.scratch[:0]
}

// start is the set the grammar begins in.
func (a *automaton) start() int32 {
	a.begin()
	for _, alt := range a.g.alts[a.g.root] {
		a.expand(a.at(alt, 0))
	}
	return a.intern(a.scratch)
}

// step is set s after code point c: 0 where no stack takes it.
func (a *automaton) step(s int32, c rune) int32 {
	if c < 128 {
		if v := a.ascii[s][c]; v >= 0 {
			return v
		}
	} else if v, ok := a.wide[[2]int32{s, c}]; ok {
		return v
	}
	a.begin()
	for _, n := range a.sets[s] {
		if n == 0 {
			continue
		}
		nd := a.nodes[n]
		if !a.g.cls[a.g.el[nd.pos].arg].has(c) {
			continue
		}
		a.expand(a.at(nd.pos+1, nd.below))
	}
	v := a.intern(a.scratch)
	if c < 128 {
		a.ascii[s][c] = v
	} else {
		a.wide[[2]int32{s, c}] = v
	}
	return v
}

// meets is whether some stack of s can take a code point in [lo, hi]: what a
// partial UTF-8 sequence is checked against before its last byte.
func (a *automaton) meets(s int32, lo, hi rune) bool {
	for _, n := range a.sets[s] {
		if n != 0 && a.g.cls[a.g.el[a.nodes[n].pos].arg].meets(lo, hi) {
			return true
		}
	}
	return false
}

// done is whether s holds the empty stack: the output may end here.
func (a *automaton) done(s int32) bool {
	ss := a.sets[s]
	return len(ss) > 0 && ss[0] == 0
}
