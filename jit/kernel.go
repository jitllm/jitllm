package jit

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/ir"
)

// Kernel is one kernel's computation: an element space, sequential loops in
// it, scalar nodes, and the stores each element makes, in order.
type Kernel struct {
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
	// Kind is the primitive the kernel was generated from, Passes what
	// changed it.
	Kind   PrimKind
	Passes []string
	// Positions and Heads are facts the primitive carried, Part the kind of
	// the model's part the kernel belongs to.
	Positions *Positions
	Heads     *Heads
	Part      PartKind
}

// Positions is a primitive's Range on its kernel: the space axis whose
// coordinate is a position, or where Loop is set the loop whose iteration
// is; positions start at the range's First, rounded down to Align where a
// pass aligned them. First and End are the nodes of the range's bounds, Live
// the nodes standing for its live end, Pos the positions' nodes.
type Positions struct {
	Axis, Loop, Align int
	First, End        int
	Live, Pos         []int
	// Span and Part, where a split set them, are 1 + the nodes of the live
	// positions in the partition and of the partition's coordinate.
	Span, Part int
}

// Loop runs Trip times, or where Count is set node Count-1's value times;
// Parent is the loop it sits in, -1 for the element. A Spread loop deals
// its trips over the workgroup's threads, its iteration the thread plus the
// trip times the group.
type Loop struct {
	Trip   int
	Count  int
	Parent int
	// Range is a loop over a range's positions, Span 1 + the node of how
	// many are live, Unit the positions one trip covers where a binding set
	// it.
	Range  bool
	Span   int
	Unit   int
	Bound  Bound
	// End and Align are the range's: 1 + the node of its end, and the
	// alignment its positions start at.
	End, Align int
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
	// Shape, where set, is the extents loads' and stores' coordinates index.
	Shape []int
	// Plane is workgroup memory a staging may fold into its plane.
	Plane bool
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

// Node is one scalar value. Load reads Bufs[Buf] at index A, or where Ax
// is set at the coordinates Ax of the buffer's Shape.
type Node struct {
	Op      Op
	Type    ir.Type
	A, B, C int
	At      int
	Buf     int
	Imm     int64
	Ax      []int
}

// Store writes node Val at index Ix of Bufs[Buf]: each trip of loop At, or
// at the element's end where At is -1. A Barrier store writes nothing and
// waits for the workgroup; a Run store writes nothing and runs loop At.
type Store struct {
	Buf, Ix, Val int
	At           int
	Barrier      bool
	Run          bool
	// Ax, where set, is the coordinates of the buffer's Shape in place of
	// Ix.
	Ax []int
}

func (n *Kernel) add(nd Node) int {
	n.Nodes = append(n.Nodes, nd)
	return len(n.Nodes) - 1
}

func (n *Kernel) u32(v int) int { return n.add(Node{Op: OpConst, Type: ir.U32, Imm: int64(v)}) }

func (n *Kernel) f32(bits uint32) int {
	return n.add(Node{Op: OpConst, Type: ir.F32, Imm: int64(bits)})
}

func (n *Kernel) bin(op Op, t ir.Type, a, b int) int { return n.add(Node{Op: op, Type: t, A: a, B: b}) }

func (n *Kernel) un(op Op, t ir.Type, a int) int { return n.add(Node{Op: op, Type: t, A: a}) }

func (n *Kernel) loop(trip, parent int) int {
	n.Loops = append(n.Loops, Loop{Trip: trip, Parent: parent})
	return len(n.Loops) - 1
}

func (n *Kernel) buf(b Buf) int {
	n.Bufs = append(n.Bufs, b)
	return len(n.Bufs) - 1
}

func (n *Kernel) clone() *Kernel {
	c := *n
	c.Space = append([]int(nil), n.Space...)
	c.Loops = append([]Loop(nil), n.Loops...)
	c.Nodes = append([]Node(nil), n.Nodes...)
	for i := range c.Nodes {
		c.Nodes[i].Ax = append([]int(nil), c.Nodes[i].Ax...)
	}
	c.Stores = append([]Store(nil), n.Stores...)
	for i := range c.Stores {
		c.Stores[i].Ax = append([]int(nil), c.Stores[i].Ax...)
	}
	c.Bufs = append([]Buf(nil), n.Bufs...)
	c.Passes = append([]string(nil), n.Passes...)
	return &c
}

func (n *Kernel) elements() int {
	e := 1
	for _, s := range n.Space {
		e *= s
	}
	return e
}

func (n *Kernel) String() string {
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

// Bound is how a binding dealt a loop's trips to a workgroup's threads.
type Bound uint8

const (
	Unbound Bound = iota
	BoundRuns
	BoundSpread
	BoundWarps
)
