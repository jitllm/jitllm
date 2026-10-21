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
	// Least is the primitive's Facts.Least.
	Least float32
	// Quarter is the primitive's Facts.Quarter, as nodes.
	Quarter   *Quarter
	Positions *Positions
	Heads     *Heads
	Part      PartKind
	// Of is the model's part the kernel belongs to, by its index.
	Of int
	// Rows is 1 + the node of how many of the leading axis's rows are live;
	// Clamp, where a pass set it, 1 + the node of the elements a launch
	// holds its threads to.
	Rows, Clamp int
	// Seq, where a schedule set it, is the kernel as a sequence: each node,
	// store and loop in the order the code runs them.
	Seq []Entry
	// Args, where a binding to the call's arguments set them, are the words
	// a call passes and what the caller computes each from.
	Args []Arg
	// Ptrs are the copies of buffers' addresses cursors advance.
	Ptrs []Ptr
}

// Arg is the call's argument word at byte offset Field: the address of
// buffer Buf, offset by the buffer's Offset, or where Buf is -1 the value
// of node Node-1, a count of the call computed by the caller.
type Arg struct {
	Field, Buf, Node int
}

// Entry is one entry of a kernel's sequence: node I, store I, the start of
// loop I or its end.
type Entry struct {
	Kind EntryKind
	I    int
}

type EntryKind uint8

const (
	EntryNode EntryKind = iota
	EntryStore
	EntryLoop
	EntryEnd
)

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
	Range bool
	Span  int
	Unit  int
	Bound Bound
	// End and Align are the range's: 1 + the node of its end, and the
	// alignment its positions start at.
	End, Align int
	Spread     bool
	// Elem, where set, is 1 + the node that is the spread loop's element.
	Elem int
	// Lanes, where a vectorization set it, is the consecutive iterations a
	// trip covers, one in each lane.
	Lanes int
	// Wide, where a jam set it, is the lanes the loop's values hold along an
	// axis outside it: they are vectors though its trips are not lanes.
	Wide int
	// Walk is set on a tile's pass: it walks copies of its cursors and of
	// its count, so each tile starts where the first did.
	Walk bool
	// Kept is set on a loop that stays a loop at one trip.
	Kept bool
	// Copies, where an unrolling set it, is how many vectors a trip covers,
	// copy j of the body at iteration j*Lanes on.
	Copies int
	// Cursors are the buffers' addresses a trip advances; Reg, where a
	// register binding set it, is 1 + the register counting the trips.
	Cursors []Cursor
	Reg     int
	// Nonempty, where a guard set it, is a loop the caller never runs for
	// no trips, entered without a test.
	Nonempty bool
	// Of, where a split set it, is 1 + the loop this one takes trips of.
	Of int
}

// Cursor is pointer Ptr of buffer Buf advanced Elems elements a trip: 0 the
// buffer's address itself, else 1 + the kernel's copy, taken from the
// buffer's address as the loop starts where Take is set.
type Cursor struct {
	Buf, Elems int
	Ptr        int
	Take       bool
}

// Ptr is a copy of buffer Buf's address a loop advances in its place, in
// register Reg (1 + the register, where a binding set it).
type Ptr struct {
	Buf, Reg int
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
	// Base, where a binding to the call's arguments set it, is 1 + the node
	// that is the buffer's address, and Offset 1 + the node of the elements
	// the caller adds to it for each call; loads and stores index from it.
	Base, Offset int
	// Reg, where a register binding set it, is 1 + the register holding the
	// address.
	Reg int
	// Words is a block of constants' contents, the caller's to supply.
	Words []uint32
	// Dims, where a run-time length set them, are 1 + the node of each
	// extent of Shape the caller passes, 0 for one that is fixed.
	Dims []int
	// Half is a buffer of binary16 elements: a load widens one to a float,
	// a store keeps a word's low half.
	Half bool
	// Shared is the device's constant block, the same words in every
	// kernel that reads it.
	Shared bool
	// Word, where a binding set it, is 1 + the argument word the buffer's
	// one word is passed in, by value.
	Word int
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
	// OpParam is the word at byte offset Imm of the call's arguments.
	OpParam
	// OpBroadcast is scalar A in every lane.
	OpBroadcast
	// OpLength is a length the caller passes at run time, Imm for this
	// model.
	OpLength
	// OpHSum is the sum of A's lanes.
	OpHSum
	// OpHalf is float A as binary16 in a word's low half, rounded to
	// nearest even, a NaN keeping its payload's high bits.
	OpHalf
	// OpGtW is, in each half of the words, 0xFFFF where A's signed half is
	// greater than B's, else 0.
	OpGtW
	// OpAndNot is A with the bits of B cleared.
	OpAndNot
	// OpZip is A's and B's lanes alternating, A first: the low half of the
	// pairs where Imm is 0, the high where 1.
	OpZip
	// OpUnpack is, in each 128-bit lane, A's and B's lanes alternating, A
	// first: of the lane's low halves where Imm is 0, its high where 1.
	OpUnpack
	// OpUpper is A's upper 128 bits in its lower.
	OpUpper
	// OpRoundI32 is float A rounded to the nearest integer, ties to even.
	OpRoundI32
	// OpSpan is, for a call that runs one span of a range's positions
	// within a page, the span's first position where Imm is 0, its length
	// where Imm is 1: values the caller computes.
	OpSpan
	// OpHMax and OpHMin are the largest and the least of A's lanes.
	OpHMax
	OpHMin
	// OpEq is A == B, a predicate.
	OpEq
	// OpOr is A's bits or B's.
	OpOr
	// OpLanes is each lane's index, of Imm lanes.
	OpLanes
	// OpSar is A shifted right by B, its sign kept.
	OpSar
	// OpDupEven and OpDupOdd are A's even and odd lanes, each in its pair
	// of lanes; OpSwapPairs is A with each pair of lanes swapped.
	OpDupEven
	OpDupOdd
	OpSwapPairs
	// OpAddSub is A less B in the even lanes, A plus B in the odd.
	OpAddSub
	// OpFirstLane is A's first lane and B's, a zero, in the rest.
	OpFirstLane
	// OpCopy is A in another register.
	OpCopy
	numOps
)

// Node is one value: a scalar, or inside a loop of Lanes a vector of them.
// A load's At, where a cursor set it, is 1 + the loop whose address it reads
// from. Load reads Bufs[Buf] at index A, or where Ax
// is set at the coordinates Ax of the buffer's Shape.
type Node struct {
	Op      Op
	Type    ir.Type
	A, B, C int
	At      int
	Buf     int
	Imm     int64
	Ax      []int
	// Reg, where a register binding set it, is 1 + the register holding
	// the value; a load's Ptr, where a cursor set it, is the pointer it
	// reads from (see Cursor). Tmp is 1 + a register the value's code
	// needs besides its own.
	Reg, Ptr, Tmp int
	// Mem is a load its one reader takes as a memory operand.
	Mem bool
	// Copy, where a pairing set it, is the head of the pair a copied value
	// is for.
	Copy int
	// Hint, where a pass set it, is 1 + the vector register the value is
	// to take. Values of one Share label take one register, an earlier one
	// copied aside where it is read after a later one is made; a Share of
	// -1 is a load of a resident word made where it is read.
	Hint, Share int
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
	// Ptr, where a cursor set it, is the pointer the store writes through.
	Ptr int
	// Copy, where a pairing set it, is the head of the pair a copied store
	// is for.
	Copy int
	// Narrow stores the value's low 128 bits.
	Narrow bool
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
	for i := range c.Loops {
		c.Loops[i].Cursors = append([]Cursor(nil), c.Loops[i].Cursors...)
	}
	c.Seq = append([]Entry(nil), n.Seq...)
	c.Args = append([]Arg(nil), n.Args...)
	c.Ptrs = append([]Ptr(nil), n.Ptrs...)
	for i := range c.Bufs {
		c.Bufs[i].Dims = append([]int(nil), c.Bufs[i].Dims...)
	}
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
	if n.Quarter != nil {
		q := *n.Quarter
		c.Quarter = &q
	}
	return &c
}

func (n *Kernel) elements() int {
	e := 1
	for _, s := range n.Space {
		e *= s
	}
	return e
}

var opNames = [numOps]string{"const", "coord", "iter", "fold", "load", "add", "sub", "mul", "div", "rem", "min", "max", "fma", "bitcast",
	"less", "select", "sqrt", "exp", "shl", "shr", "and", "xor", "cvtf32", "cvti32", "thread", "groupsum", "groupmax", "exchange",
	"local", "flat", "shuffle", "param", "broadcast", "length", "hsum", "half", "gtw", "andnot", "zip", "unpack", "upper", "round", "span", "hmax", "hmin", "eq", "or", "lanes", "sar", "dupeven", "dupodd", "swappairs", "addsub", "firstlane", "copy"}

func (o Op) String() string {
	if int(o) < len(opNames) {
		return opNames[o]
	}
	return fmt.Sprintf("op%d", o)
}

func (n *Kernel) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s space %v work %v group %d passes %v\n", n.Name, n.Space, n.Work, n.Group, n.Passes)
	for i, l := range n.Loops {
		fmt.Fprintf(&b, "  loop %d %+v\n", i, l)
	}
	for i, bf := range n.Bufs {
		fmt.Fprintf(&b, "  buf %d kind %d %s elem %s shape %v size %d\n", i, bf.Kind, bf.Operand.Name, bf.Elem, bf.Shape, bf.Size)
	}
	for i, nd := range n.Nodes {
		fmt.Fprintf(&b, "  %d: %s.%s a %d b %d c %d at %d buf %d imm %d ax %v reg %d tmp %d mem %v ptr %d hint %d share %d\n", i, nd.Op, nd.Type, nd.A, nd.B, nd.C, nd.At, nd.Buf, nd.Imm, nd.Ax, nd.Reg, nd.Tmp, nd.Mem, nd.Ptr, nd.Hint, nd.Share)
	}
	for _, s := range n.Stores {
		fmt.Fprintf(&b, "  store buf %d ix %d ax %v val %d at %d barrier %v run %v\n", s.Buf, s.Ix, s.Ax, s.Val, s.At, s.Barrier, s.Run)
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

// groups is how many workgroups a launch of k runs.
func (k *Kernel) groups() int {
	if k.Work {
		return k.elements()
	}
	if k.Group <= 0 {
		return 0
	}
	return (k.elements() + k.Group - 1) / k.Group
}

// once reports whether a loop runs one fixed trip: its body, no count.
func (l Loop) once() bool { return l.Count == 0 && l.Trip == 1 && !l.Kept }
