package jit

import "github.com/jitllm/jitllm/jit/ir"

// Nest is one kernel's computation: an element space, sequential loops in
// it, scalar nodes, and the stores each element makes.
type Nest struct {
	Name string
	// Space is the element space's extents, the last fastest.
	Space  []int
	Loops  []Loop
	Nodes  []Node
	Stores []Store
	Bufs   []Buf
	Group  int
}

// Loop runs Trip times; Parent is the loop it sits in, -1 for the element.
type Loop struct {
	Trip   int
	Parent int
}

type BufKind uint8

const (
	BufIn BufKind = iota
	BufOut
)

// Buf is a kernel parameter: what it binds to and how its words are typed.
type Buf struct {
	Kind    BufKind
	Elem    ir.Type
	Operand Operand
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

// Store writes node Val at index Ix of Bufs[Buf], after loop At's last trip
// (-1 for the element's end).
type Store struct {
	Buf, Ix, Val int
	At           int
}

func (n *Nest) add(nd Node) int {
	n.Nodes = append(n.Nodes, nd)
	return len(n.Nodes) - 1
}

func (n *Nest) u32(v int) int { return n.add(Node{Op: OpConst, Type: ir.U32, Imm: int64(v)}) }

func (n *Nest) f32(bits uint32) int { return n.add(Node{Op: OpConst, Type: ir.F32, Imm: int64(bits)}) }

func (n *Nest) bin(op Op, t ir.Type, a, b int) int { return n.add(Node{Op: op, Type: t, A: a, B: b}) }

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
	return &c
}

func (n *Nest) elements() int {
	e := 1
	for _, s := range n.Space {
		e *= s
	}
	return e
}
