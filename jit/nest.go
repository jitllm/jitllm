package jit

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/ir"
)

// Nest is one kernel's computation: an element space, sequential loops in
// it, scalar nodes, and the stores each element makes, in order.
type Nest struct {
	Name string
	// Space is the element space's extents, the last fastest. Under Work an
	// element is a workgroup of Group threads, else a thread.
	Space  []int
	Work   bool
	Loops  []Loop
	Nodes  []Node
	Stores []Store
	Bufs   []Buf
	Group  int
	// Kind is the primitive the nest lowered, Prim its definition while no
	// pass has changed it, Passes what changed it.
	Kind   PrimKind
	Prim   *Prim
	Passes []string
}

// Loop runs Trip times, or where Count is set node Count-1's value times;
// Parent is the loop it sits in, -1 for the element. A Spread loop deals
// its trips over the workgroup's threads, its iteration the thread plus the
// trip times the group.
type Loop struct {
	Trip   int
	Count  int
	Parent int
	Spread bool
	// Elem, where set, is 1 + the node that is the spread loop's element.
	Elem int
}

type BufKind uint8

const (
	BufIn BufKind = iota
	BufOut
	// BufShared is workgroup memory of Size words.
	BufShared
)

// Buf is a kernel parameter, or workgroup memory: what it binds to and how
// its words are typed.
type Buf struct {
	Kind    BufKind
	Elem    ir.Type
	Operand Operand
	Size    int
}

type Op uint8

const (
	OpConst Op = iota
	// OpCoord is the element's coordinate along Space[A].
	OpCoord
	// OpIter is loop At's iteration.
	OpIter
	// OpFold is a value loop At carries: B before the first trip, A after
	// each.
	OpFold
	OpLoad
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpRem
	OpMin
	OpMax
	OpFma
	OpBitcast
	// OpLess is A < B, a predicate; OpSelect is B where A holds, else C.
	OpLess
	OpSelect
	OpSqrt
	OpExp
	OpShl
	OpShr
	OpAnd
	OpXor
	OpCvtF32
	OpCvtI32
	// OpThread is the thread's index in its workgroup.
	OpThread
	// OpGroupSum and OpGroupMax combine A over the workgroup's threads.
	OpGroupSum
	OpGroupMax
	// OpExchange is A to the thread's slot of a workgroup array, a barrier,
	// and the Imm slots from B combined by At, begun at node C (-1 for the
	// first slot).
	OpExchange
	// OpLocal is element A's place in its workgroup.
	OpLocal
	// OpFlat is the thread's index in the launch, unclamped.
	OpFlat
	// OpShuffle is A of the lane whose index is this one's xor Imm.
	OpShuffle
)

// Node is one scalar value. Load reads Bufs[Buf] at index A.
type Node struct {
	Op      Op
	Type    ir.Type
	A, B, C int
	At      int
	Buf     int
	Imm     int64
}

// Store writes node Val at index Ix of Bufs[Buf]: each trip of loop At, or
// at the element's end where At is -1. A Barrier store writes nothing and
// waits for the workgroup; a Run store writes nothing and runs loop At.
type Store struct {
	Buf, Ix, Val int
	At           int
	Barrier      bool
	Run          bool
}

func (n *Nest) add(nd Node) int {
	n.Nodes = append(n.Nodes, nd)
	return len(n.Nodes) - 1
}

func (n *Nest) u32(v int) int { return n.add(Node{Op: OpConst, Type: ir.U32, Imm: int64(v)}) }

func (n *Nest) f32(bits uint32) int { return n.add(Node{Op: OpConst, Type: ir.F32, Imm: int64(bits)}) }

func (n *Nest) bin(op Op, t ir.Type, a, b int) int { return n.add(Node{Op: op, Type: t, A: a, B: b}) }

func (n *Nest) un(op Op, t ir.Type, a int) int { return n.add(Node{Op: op, Type: t, A: a}) }

func (n *Nest) loop(trip, parent int) int {
	n.Loops = append(n.Loops, Loop{Trip: trip, Parent: parent})
	return len(n.Loops) - 1
}

func (n *Nest) buf(b Buf) int {
	n.Bufs = append(n.Bufs, b)
	return len(n.Bufs) - 1
}

func (n *Nest) clone() *Nest {
	c := *n
	c.Space = append([]int(nil), n.Space...)
	c.Loops = append([]Loop(nil), n.Loops...)
	c.Nodes = append([]Node(nil), n.Nodes...)
	c.Stores = append([]Store(nil), n.Stores...)
	c.Bufs = append([]Buf(nil), n.Bufs...)
	c.Passes = append([]string(nil), n.Passes...)
	c.Prim = nil
	return &c
}

func (n *Nest) elements() int {
	e := 1
	for _, s := range n.Space {
		e *= s
	}
	return e
}

func (n *Nest) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s space %v work %v loops %v\n", n.Name, n.Space, n.Work, n.Loops)
	for i, nd := range n.Nodes {
		fmt.Fprintf(&b, "  %d: op %d %s a %d b %d c %d at %d buf %d imm %d\n", i, nd.Op, nd.Type, nd.A, nd.B, nd.C, nd.At, nd.Buf, nd.Imm)
	}
	for _, s := range n.Stores {
		fmt.Fprintf(&b, "  store buf %d ix %d val %d at %d\n", s.Buf, s.Ix, s.Val, s.At)
	}
	return b.String()
}
