// Package compose forms device kernels from the fragments of the ops a fusion
// group holds (jit/gpu/kernels' fragments), under a schedule the target and
// the tuner choose. Nothing here is a kernel for an op: a group is a
// composition, and its cache key is its structure and its schedule.
package compose

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A Row group is ops over rows of Width elements, one workgroup a row: per
// element expressions over its buffers, row reductions whose results every
// element of the row may read, stores of element values, and an int8
// operand of one of them. The former splits it into passes by its reductions
// and recomputes an element's value from the buffers in each pass that needs
// it, so no pass reads what another wrote (RULE 13).
type Row struct {
	Width, Rows int
	Bufs        []Buf
	Nodes       []Node
	Stores      []Store
	// Q8, where set, writes node Q8.Val as the int8 operand a matvec reads:
	// into buffers Q8.A and Q8.X at Q8.Window, the row padded to Q8.Pad.
	Q8 *Q8
}

// BufKind is how a group reads a buffer.
type BufKind uint8

const (
	// BufRow is a float per element of every row (row*Width + column).
	BufRow BufKind = iota
	// BufVec is one float a column, every row reading the same (a weight).
	BufVec
	// BufScalar is a float every element reads (a scale), at Node.At.
	BufScalar
	// BufRowScalar is one float a row (a row's gate logit).
	BufRowScalar
	// BufOut is a float per element of every row, written.
	BufOut
	// BufQ8 and BufQ8X are an int8 operand's values and scales.
	BufQ8
	BufQ8X
	// BufU32 is a table of indices (a row's offset), read at an index.
	BufU32
)

// Buf is one of a group's buffers, a kernel parameter in order.
type Buf struct{ Kind BufKind }

// NodeKind is what a node computes.
type NodeKind uint8

const (
	NLoad    NodeKind = iota // Bufs[Buf] at the element
	NConst                   // F
	NUnary                   // Op of A
	NBinary                  // Op of A and B
	NReduce                  // Red over the row of A: a row value every element reads
	NColumn                  // the element's column in its row (an index)
	NRowIdx                  // the row's index (an index)
	NUConst                  // the index At
	NUBinary                 // Op (Add, Sub, Mul, Div, Rem, Min) of indices A and B
	NLess                    // A < B, of indices: a condition
	NSelect                  // C ? A : B, of floats or of indices
	NInput                   // a former's own output at the element (a matvec's), in an epilogue
)

// Op is a unary or binary map.
type Op uint8

const (
	OpAdd Op = iota
	OpSub
	OpMul
	OpDiv
	OpMax
	OpMin
	OpAct    // activation Act of A
	OpActMul // the gated activation Act of A times B
	OpRsqrt  // 1/sqrt(A)
	OpAbs
	OpFma // A*B + C, one rounding (C in Node.C)
	OpRem // the remainder of indices
)

// Red is a row reduction.
type Red uint8

const (
	RedSum Red = iota
	RedMax
)

// Node is one value of the group's expression graph.
type Node struct {
	Kind    NodeKind
	Op      Op
	Act     kernels.ActKind
	Red     Red
	Buf     int
	A, B, C int
	F       float32
	// At is a BufScalar load's index, or an NUConst's value.
	At int
	// Indexed loads at node Ix (an index), the buffer's own position
	// where it is false: the element's place in its row, or the row's.
	Indexed bool
	Ix      int
}

// Store writes node Val into buffer Buf at the element, or at index node Ix
// where Indexed (an absolute place in the buffer).
type Store struct {
	Buf, Val int
	Indexed  bool
	Ix       int
}

// Q8 is a group's int8 epilogue.
type Q8 struct {
	Val, A, X   int
	Window, Pad int
}

// Schedule is how a group runs: its workgroup's threads, and whether a
// reduction combines through subgroup shuffles (contiguous 32-lane groups,
// the target says where) or a tree in threadgroup memory.
type Schedule struct {
	Group int
	Warp  bool
}

// Key is a group's identity in a cache: its structure and its schedule,
// hashed with the encoding kept to tell a collision from a hit.
type Key struct {
	Hash uint64
}

// Encode is the group's structure and schedule as integers.
func (g *Row) Encode(s Schedule) []uint32 {
	e := []uint32{1, uint32(g.Width), uint32(g.Rows), uint32(s.Group), b2u(s.Warp), uint32(len(g.Bufs))}
	for _, b := range g.Bufs {
		e = append(e, uint32(b.Kind))
	}
	for _, n := range g.Nodes {
		e = append(e, uint32(n.Kind), uint32(n.Op), uint32(n.Act), uint32(n.Red), uint32(n.Buf),
			uint32(n.A), uint32(n.B), uint32(n.C), f2u(n.F), uint32(n.At), b2u(n.Indexed), uint32(n.Ix))
	}
	for _, s := range g.Stores {
		e = append(e, uint32(s.Buf), uint32(s.Val), b2u(s.Indexed), uint32(s.Ix))
	}
	if q := g.Q8; q != nil {
		e = append(e, uint32(q.Val), uint32(q.A), uint32(q.X), uint32(q.Window), uint32(q.Pad))
	}
	return e
}

// KeyOf is Encode hashed.
func KeyOf(enc []uint32) Key {
	h := fnv.New64a()
	var w [4]byte
	for _, x := range enc {
		binary.LittleEndian.PutUint32(w[:], x)
		h.Write(w[:])
	}
	return Key{Hash: h.Sum64()}
}

// depth is a node's pass: the reductions under it, nested.
func (g *Row) depth() []int {
	d := make([]int, len(g.Nodes))
	for i, n := range g.Nodes {
		switch n.Kind {
		case NUnary:
			d[i] = d[n.A]
		case NBinary, NUBinary, NLess:
			d[i] = max(d[n.A], d[n.B])
			if n.Op == OpFma && n.Kind == NBinary {
				d[i] = max(d[i], d[n.C])
			}
		case NSelect:
			d[i] = max(d[n.A], d[n.B], d[n.C])
		case NReduce:
			d[i] = d[n.A] + 1
		case NLoad:
			if n.Indexed {
				d[i] = d[n.Ix]
			}
		}
	}
	return d
}

// Check refuses a group the former cannot emit: a node reading a later one,
// a buffer of the wrong kind, a reduction of a reduction's own pass.
func (g *Row) Check() error {
	if g.Width < 1 || g.Rows < 1 {
		return fmt.Errorf("compose: a row group of %d x %d", g.Rows, g.Width)
	}
	for i, n := range g.Nodes {
		if n.Kind == NUnary && n.Op == OpAct {
			if err := kernels.ActElemOK(n.Act); err != nil {
				return fmt.Errorf("compose: node %d: %w", i, err)
			}
		}
		in := []int{}
		switch n.Kind {
		case NLoad:
			if n.Buf < 0 || n.Buf >= len(g.Bufs) {
				return fmt.Errorf("compose: node %d loads buffer %d of %d", i, n.Buf, len(g.Bufs))
			}
			if k := g.Bufs[n.Buf].Kind; k != BufRow && k != BufVec && k != BufScalar && k != BufRowScalar && k != BufU32 {
				return fmt.Errorf("compose: node %d loads an output buffer", i)
			}
		case NUnary, NReduce:
			in = []int{n.A}
		case NBinary, NUBinary, NLess:
			in = []int{n.A, n.B}
			if n.Op == OpFma && n.Kind == NBinary {
				in = append(in, n.C)
			}
		case NSelect:
			in = []int{n.A, n.B, n.C}
		}
		if n.Kind == NLoad && n.Indexed {
			in = append(in, n.Ix)
		}
		for _, x := range in {
			if x < 0 || x >= i {
				return fmt.Errorf("compose: node %d reads node %d", i, x)
			}
		}
	}
	for _, s := range g.Stores {
		if s.Buf < 0 || s.Buf >= len(g.Bufs) || g.Bufs[s.Buf].Kind != BufOut || s.Val < 0 || s.Val >= len(g.Nodes) ||
			s.Indexed && (s.Ix < 0 || s.Ix >= len(g.Nodes)) {
			return fmt.Errorf("compose: a store of node %d into buffer %d", s.Val, s.Buf)
		}
	}
	if q := g.Q8; q != nil {
		if q.Val < 0 || q.Val >= len(g.Nodes) || g.Bufs[q.A].Kind != BufQ8 || g.Bufs[q.X].Kind != BufQ8X ||
			q.Pad%32 != 0 || q.Pad < g.Width {
			return fmt.Errorf("compose: an int8 epilogue of node %d at %d", q.Val, q.Pad)
		}
	}
	return nil
}

// Emit forms the group's kernel under schedule s.
func (g *Row) Emit(s Schedule) (*ir.Kernel, error) {
	if err := g.Check(); err != nil {
		return nil, err
	}
	G := s.Group
	if G < 32 || G&(G-1) != 0 {
		return nil, fmt.Errorf("compose: a group of %d threads", G)
	}
	b := ir.New("row", [3]int{G, 1, 1})
	params := make([]ir.Value, len(g.Bufs))
	for i, bf := range g.Bufs {
		t := ir.F32
		if bf.Kind == BufQ8 || bf.Kind == BufU32 {
			t = ir.U32
		}
		params[i] = b.Param(fmt.Sprintf("p%d", i), t)
	}
	sh := b.Shared("red", ir.F32, G)
	r := b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(g.Rows-1)))
	rowBase := b.Mul(ir.U32, r, b.Const(ir.U32, int64(g.Width)))
	tid := b.TID()
	width := b.Const(ir.U32, int64(g.Width))
	last := b.Const(ir.U32, int64(g.Width-1))
	depth := g.depth()
	rowVal := make([]ir.Value, len(g.Nodes)) // a reduction's result, once its pass ran
	// elem emits node n at column idx, its loads and maps in this pass.
	var elem func(n int, idx ir.Value, memo map[int]ir.Value) ir.Value
	elem = func(n int, idx ir.Value, memo map[int]ir.Value) ir.Value {
		if v, ok := memo[n]; ok {
			return v
		}
		nd := g.Nodes[n]
		var v ir.Value
		switch nd.Kind {
		case NLoad:
			if nd.Indexed {
				at := elem(nd.Ix, idx, memo)
				t := ir.F32
				if g.Bufs[nd.Buf].Kind == BufU32 {
					t = ir.U32
				}
				v = b.Load(t, params[nd.Buf], at, 0)
				break
			}
			switch g.Bufs[nd.Buf].Kind {
			case BufU32:
				v = b.Load(ir.U32, params[nd.Buf], r, 0)
			case BufRow:
				v = b.Load(ir.F32, params[nd.Buf], b.Add(ir.U32, rowBase, idx), 0)
			case BufVec:
				v = b.Load(ir.F32, params[nd.Buf], idx, 0)
			case BufRowScalar:
				v = b.Load(ir.F32, params[nd.Buf], r, 0)
			default:
				v = b.Load(ir.F32, params[nd.Buf], b.Const(ir.U32, int64(nd.At)), 0)
			}
		case NConst:
			v = b.ConstF32(nd.F)
		case NColumn:
			v = idx
		case NRowIdx:
			v = r
		case NUConst:
			v = b.Const(ir.U32, int64(nd.At))
		case NUBinary:
			x, y := elem(nd.A, idx, memo), elem(nd.B, idx, memo)
			switch nd.Op {
			case OpAdd:
				v = b.Add(ir.U32, x, y)
			case OpSub:
				v = b.Sub(ir.U32, x, y)
			case OpMul:
				v = b.Mul(ir.U32, x, y)
			case OpDiv:
				v = b.Div(ir.U32, x, y)
			case OpRem:
				v = b.Rem(ir.U32, x, y)
			case OpMin:
				v = b.Min(ir.U32, x, y)
			}
		case NLess:
			v = b.Lt(ir.U32, elem(nd.A, idx, memo), elem(nd.B, idx, memo))
		case NSelect:
			t := ir.F32
			if g.isIndex(nd.A) {
				t = ir.U32
			}
			v = b.Select(t, elem(nd.C, idx, memo), elem(nd.A, idx, memo), elem(nd.B, idx, memo))
		case NReduce:
			v = rowVal[n]
		case NUnary:
			a := elem(nd.A, idx, memo)
			switch nd.Op {
			case OpAct:
				v = kernels.ActFragment(b, a, nd.Act)
			case OpRsqrt:
				v = b.Div(ir.F32, b.ConstF32(1), b.Sqrt(a))
			case OpAbs:
				v = b.Max(ir.F32, a, b.Sub(ir.F32, b.ConstF32(0), a))
			}
		case NBinary:
			a := elem(nd.A, idx, memo)
			switch nd.Op {
			case OpActMul:
				v = kernels.ActMulFragment(b, nd.Act, a, func() ir.Value { return elem(nd.B, idx, memo) })
			case OpFma:
				v = b.Fma(a, elem(nd.B, idx, memo), elem(nd.C, idx, memo))
			default:
				c := elem(nd.B, idx, memo)
				switch nd.Op {
				case OpAdd:
					v = b.Add(ir.F32, a, c)
				case OpSub:
					v = b.Sub(ir.F32, a, c)
				case OpMul:
					v = b.Mul(ir.F32, a, c)
				case OpDiv:
					v = b.Div(ir.F32, a, c)
				case OpMax:
					v = b.Max(ir.F32, a, c)
				case OpMin:
					v = b.Min(ir.F32, a, c)
				}
			}
		}
		memo[n] = v
		return v
	}
	passes := 0
	for _, d := range depth {
		passes = max(passes, d)
	}
	for _, st := range g.Stores {
		passes = max(passes, g.storePass(st, depth))
	}
	iters := int64((g.Width + G - 1) / G)
	step := b.Const(ir.U32, int64(G))
	// Pass p: every reduction of depth p+1 accumulates; every store of depth
	// p writes, once its reductions ran.
	for p := 0; p <= passes; p++ {
		var reds []int
		for i, nd := range g.Nodes {
			if nd.Kind == NReduce && depth[i] == p+1 {
				reds = append(reds, i)
			}
		}
		var stores []Store
		for _, st := range g.Stores {
			if g.storePass(st, depth) == p {
				stores = append(stores, st)
			}
		}
		if len(reds) == 0 && len(stores) == 0 {
			continue
		}
		// The accumulators' starts, defined before the loop that carries them.
		inits := make([]ir.Value, len(reds))
		for j, n := range reds {
			inits[j] = b.ConstF32(0)
			if g.Nodes[n].Red == RedMax {
				inits[j] = b.ConstF32(-3.4028235e38)
			}
		}
		b.Loop(iters)
		i := b.Phi(ir.U32, tid)
		accs := make([]ir.Value, len(reds))
		for j := range reds {
			accs[j] = b.Phi(ir.F32, inits[j])
		}
		idx := b.Min(ir.U32, i, last)
		in := b.Lt(ir.U32, i, width)
		memo := map[int]ir.Value{}
		for j, n := range reds {
			x := elem(g.Nodes[n].A, idx, memo)
			if g.Nodes[n].Red == RedMax {
				b.SetPhi(accs[j], b.Select(ir.F32, in, b.Max(ir.F32, accs[j], x), accs[j]))
			} else {
				b.SetPhi(accs[j], b.Add(ir.F32, accs[j], b.Select(ir.F32, in, x, b.ConstF32(0))))
			}
		}
		for _, st := range stores {
			at := b.Add(ir.U32, rowBase, idx)
			if st.Indexed {
				at = elem(st.Ix, idx, memo)
			}
			b.Store(params[st.Buf], at, elem(st.Val, idx, memo), 0)
		}
		b.SetPhi(i, b.Add(ir.U32, i, step))
		b.EndLoop()
		for j, n := range reds {
			if g.Nodes[n].Red == RedMax {
				rowVal[n] = kernels.TreeMax(b, sh, tid, accs[j], G)
			} else {
				rowVal[n] = kernels.TreeSum(b, sh, tid, accs[j], G)
			}
		}
	}
	if q := g.Q8; q != nil {
		shMax := b.Shared("amax", ir.F32, G)
		shSum := b.Shared("qsum", ir.F32, G)
		zero := b.ConstF32(0)
		kernels.QuantFragment(b, q.Pad, g.Rows, q.Window, G, r, tid, params[q.A], params[q.X], shMax, shSum,
			func(at ir.Value) ir.Value {
				// Past the row's width, the zero pad.
				v := elem(q.Val, b.Min(ir.U32, at, last), map[int]ir.Value{})
				return b.Select(ir.F32, b.Lt(ir.U32, at, width), v, zero)
			})
	}
	return b.Done(), nil
}

func b2u(x bool) uint32 {
	if x {
		return 1
	}
	return 0
}

func f2u(f float32) uint32 { return math.Float32bits(f) }

// isIndex reports node n is an index (an integer), not a float.
func (g *Row) isIndex(n int) bool {
	nd := g.Nodes[n]
	switch nd.Kind {
	case NColumn, NRowIdx, NUConst, NUBinary:
		return true
	case NLoad:
		return g.Bufs[nd.Buf].Kind == BufU32
	case NSelect:
		return g.isIndex(nd.A)
	}
	return false
}

// storePass is the pass a store writes in: once its value and its index have
// their reductions.
func (g *Row) storePass(st Store, depth []int) int {
	d := depth[st.Val]
	if st.Indexed {
		d = max(d, depth[st.Ix])
	}
	return d
}
