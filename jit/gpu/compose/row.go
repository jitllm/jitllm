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
	// Loops are the trip counts of the group's folds' loops (Loop): loop c
	// runs Loops[c] times.
	Loops []int
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
	// BufOutU32 is a 32-bit word per element of every row, written (a pair
	// of binary16 values, OpPackF16).
	BufOutU32
	// BufRowF16 is BufRow stored as binary16, two elements a 32-bit word
	// (the low half first): a load decodes the element's half. A buffer of
	// an odd element count is padded to a whole word.
	BufRowF16
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
	// NIter is loop At's iterator (an index, 0 up to its trip): what a
	// fold's body reads its element by.
	NIter
	// NFold is a value carried by loop At: B before the first trip, then
	// A each trip, where A (any node, a later one too: the loop's back
	// edge) reads the loop's iterator and the values its folds carry at
	// the trip's start. Inside its loop's body a fold reads as its value at
	// the trip's start, elsewhere as its value after the last trip. Folds
	// of one loop run in one loop, each reading the others' values: a
	// count, a sum, a maximum or several carried together, in the order
	// the body states. A loop nested in another's body (a fold whose body
	// reads another loop's iterator) runs inside that body.
	NFold
	// NIConst is the signed 32-bit integer At.
	NIConst
	// NDot4 is C plus the dot product of the four signed bytes of word A
	// with the four of word B, a signed 32-bit integer (At 1: no byte of
	// either is -128, which a target may use).
	NDot4
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
	OpExp // e to the A
	// OpSqrt is the square root of A.
	OpSqrt
	// OpFloat is index A as a float (exact below 2^31).
	OpFloat
	// OpPackF16 is A and B as two binary16 halves of one 32-bit word, A low:
	// a value only a BufOutU32 store takes.
	OpPackF16
	// The bitwise ops of 32-bit words (NUBinary): and, xor, and the shifts
	// of A by B. (The IR has no or: disjoint fields are joined by an add.)
	OpAnd
	OpXor
	OpShl
	OpShr
	// OpHalf is the binary16 in word A's low half, as a float.
	OpHalf
	// OpBits is word A's 32 bits as a float, no conversion.
	OpBits
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
	// Flat runs a thread per element of every row, the grid
	// ceil(Rows*Width/Group) workgroups: the element fastest, no element
	// loop. Only a group with no row reduction takes it.
	Flat bool
}

// Key is a group's identity in a cache: its structure and its schedule,
// hashed with the encoding kept to tell a collision from a hit.
type Key struct {
	Hash uint64
}

// Encode is the group's structure and schedule as integers.
func (g *Row) Encode(s Schedule) []uint32 {
	e := []uint32{1, uint32(g.Width), uint32(g.Rows), uint32(s.Group), b2u(s.Warp), uint32(len(g.Bufs)), b2u(s.Flat)}
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
	if len(g.Loops) > 0 {
		e = append(e, 0xF01D, uint32(len(g.Loops)))
		for _, t := range g.Loops {
			e = append(e, uint32(t))
		}
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

// depth is a node's pass: the reductions under it, nested. A fold reads
// later nodes (its loop's back edge), so the passes are swept to their
// fixed point; false where they have none (a reduction a fold's own body
// reads through the fold).
func (g *Row) depth() ([]int, bool) {
	d := make([]int, len(g.Nodes))
	for sweep := 0; sweep <= len(g.Nodes)+1; sweep++ {
		changed := false
		for i, n := range g.Nodes {
			v := 0
			switch n.Kind {
			case NUnary:
				v = d[n.A]
			case NBinary, NUBinary, NLess:
				v = max(d[n.A], d[n.B])
				if n.Op == OpFma && n.Kind == NBinary {
					v = max(v, d[n.C])
				}
			case NSelect, NDot4:
				v = max(d[n.A], d[n.B], d[n.C])
			case NReduce:
				v = d[n.A] + 1
			case NLoad:
				if n.Indexed {
					v = d[n.Ix]
				}
			case NFold:
				v = max(d[n.A], d[n.B])
			}
			if v != d[i] {
				d[i], changed = v, true
			}
		}
		if !changed {
			return d, true
		}
	}
	return d, false
}

// Check refuses a group the former cannot emit: a node reading a later one,
// a buffer of the wrong kind, a reduction of a reduction's own pass.
func (g *Row) Check() error {
	if g.Width < 1 || g.Rows < 1 {
		return fmt.Errorf("compose: a row group of %d x %d", g.Rows, g.Width)
	}
	for i, n := range g.Nodes {
		in := []int{}
		switch n.Kind {
		case NLoad:
			if n.Buf < 0 || n.Buf >= len(g.Bufs) {
				return fmt.Errorf("compose: node %d loads buffer %d of %d", i, n.Buf, len(g.Bufs))
			}
			if k := g.Bufs[n.Buf].Kind; k != BufRow && k != BufVec && k != BufScalar && k != BufRowScalar && k != BufU32 && k != BufRowF16 {
				return fmt.Errorf("compose: node %d loads an output buffer", i)
			}
		case NUnary, NReduce:
			in = []int{n.A}
		case NBinary, NUBinary, NLess:
			in = []int{n.A, n.B}
			if n.Op == OpFma && n.Kind == NBinary {
				in = append(in, n.C)
			}
		case NSelect, NDot4:
			in = []int{n.A, n.B, n.C}
		case NIter:
			if n.At < 0 || n.At >= len(g.Loops) {
				return fmt.Errorf("compose: node %d iterates loop %d of %d", i, n.At, len(g.Loops))
			}
		case NFold:
			if n.At < 0 || n.At >= len(g.Loops) || g.Loops[n.At] < 1 {
				return fmt.Errorf("compose: node %d folds over loop %d of %d", i, n.At, len(g.Loops))
			}
			// Its step may be any node, a later one too: the back edge.
			if n.A < 0 || n.A >= len(g.Nodes) {
				return fmt.Errorf("compose: fold %d steps to node %d", i, n.A)
			}
			if g.typeOf(n.A) != g.typeOf(n.B) {
				return fmt.Errorf("compose: fold %d steps to a value of another type than its start", i)
			}
			in = []int{n.B}
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
		if s.Buf < 0 || s.Buf >= len(g.Bufs) || g.Bufs[s.Buf].Kind != BufOut && g.Bufs[s.Buf].Kind != BufOutU32 || s.Val < 0 || s.Val >= len(g.Nodes) ||
			s.Indexed && (s.Ix < 0 || s.Ix >= len(g.Nodes)) {
			return fmt.Errorf("compose: a store of node %d into buffer %d", s.Val, s.Buf)
		}
	}
	if g.Q8 != nil {
		// The int8 operand (a scale per window, rounded codes packed four
		// to a word, sums per 16) needs primitives the node language does
		// not have yet: windowed reductions, float to integer, bit packing.
		return fmt.Errorf("compose: an int8 operand has no primitive form yet")
	}
	if _, ok := g.depth(); !ok {
		return fmt.Errorf("compose: a reduction read through a fold of its own")
	}
	return nil
}

// env is the values emitted so far in a scope: the pass's element, and
// each fold loop's body inside it (a body sees what its enclosing scope
// emitted, and nothing it emits outlives it but its folds).
type env struct {
	m  map[int]ir.Value
	up *env
}

func (e *env) get(n int) (ir.Value, bool) {
	for x := e; x != nil; x = x.up {
		if v, ok := x.m[n]; ok {
			return v, true
		}
	}
	return 0, false
}

func newEnv(up *env) *env { return &env{m: map[int]ir.Value{}, up: up} }

// operands are the nodes node n reads where it is emitted: not a fold's
// step (its loop's body), and not a reduction's operand (an earlier pass).
func (g *Row) operands(n int) []int {
	nd := g.Nodes[n]
	switch nd.Kind {
	case NUnary:
		return []int{nd.A}
	case NBinary:
		if nd.Op == OpFma {
			return []int{nd.A, nd.B, nd.C}
		}
		return []int{nd.A, nd.B}
	case NUBinary, NLess:
		return []int{nd.A, nd.B}
	case NSelect, NDot4:
		return []int{nd.A, nd.B, nd.C}
	case NLoad:
		if nd.Indexed {
			return []int{nd.Ix}
		}
	case NFold:
		return []int{nd.B}
	}
	return nil
}

// varies is which nodes change from trip to trip of loop c: its iterator,
// its folds, and whatever reads them -- a fold of another loop whose start
// or step does is a loop nested in c's body. A fixed point, since a fold
// reads later nodes.
func (g *Row) varies(c int) []bool {
	v := make([]bool, len(g.Nodes))
	for changed := true; changed; {
		changed = false
		for i, nd := range g.Nodes {
			if v[i] {
				continue
			}
			x := false
			switch nd.Kind {
			case NIter:
				x = nd.At == c
			case NFold:
				x = nd.At == c
				for _, f := range g.Nodes {
					if f.Kind == NFold && f.At == nd.At && (v[f.B] || v[f.A]) {
						x = true
					}
				}
			default:
				for _, o := range g.operands(i) {
					x = x || v[o]
				}
			}
			if x {
				v[i], changed = true, true
			}
		}
	}
	return v
}

// Emit forms the group's kernel under schedule s.
func (g *Row) Emit(s Schedule) (*ir.Kernel, error) {
	if err := g.Check(); err != nil {
		return nil, err
	}
	// Activations are formulas over primitive nodes; the former emits
	// primitives only.
	g, _, err := g.expanded()
	if err != nil {
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
		if bf.Kind == BufQ8 || bf.Kind == BufU32 || bf.Kind == BufOutU32 || bf.Kind == BufRowF16 {
			t = ir.U32
		}
		params[i] = b.Param(fmt.Sprintf("p%d", i), t)
	}
	depth, _ := g.depth()
	// The row and, under the flat schedule, the element: a workgroup a row
	// with the element looping over the group, or a thread an element of
	// every row, the element fastest (the last thread's surplus clamped to
	// the last element, which it recomputes: RULE 13).
	var sh, r, rowBase, tid, width, last, flatIdx ir.Value
	if s.Flat {
		for i, nd := range g.Nodes {
			if nd.Kind == NReduce && depth[i] > 0 {
				return nil, fmt.Errorf("compose: a flat schedule over a row reduction")
			}
		}
		flat := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
		flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(g.Rows*g.Width-1)))
		width = b.Const(ir.U32, int64(g.Width))
		flatIdx = b.Rem(ir.U32, flat, width)
		r = b.Div(ir.U32, flat, width)
		rowBase = b.Mul(ir.U32, r, width)
	} else {
		sh = b.Shared("red", ir.F32, G)
		r = b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(g.Rows-1)))
		rowBase = b.Mul(ir.U32, r, b.Const(ir.U32, int64(g.Width)))
		tid = b.TID()
		width = b.Const(ir.U32, int64(g.Width))
		last = b.Const(ir.U32, int64(g.Width-1))
	}
	rowVal := make([]ir.Value, len(g.Nodes)) // a reduction's result, once its pass ran
	var emitErr error
	// half is the binary16 element at e of a BufRowF16 buffer: its word,
	// and the half of it the element sits in.
	half := func(p, e ir.Value) ir.Value {
		one := b.Const(ir.U32, 1)
		word := b.Load(ir.U32, p, b.Shr(ir.U32, e, one), 0)
		sh := b.Shl(ir.U32, b.And(ir.U32, e, one), b.Const(ir.U32, 4))
		return b.CvtF16H(b.Shr(ir.U32, word, sh))
	}
	// elem emits node n at column idx, its loads and maps, in scope e.
	var elem func(n int, idx ir.Value, e *env) ir.Value
	// loop emits fold loop c in scope e: first what its body reads that
	// does not change from trip to trip, then the loop, its folds' values
	// after it left in e.
	loop := func(c int, idx ir.Value, e *env) {
		var folds []int
		for i, nd := range g.Nodes {
			if nd.Kind == NFold && nd.At == c {
				folds = append(folds, i)
			}
		}
		vary := g.varies(c)
		// closed is a node reading the iterator of a loop not open here: a
		// loop nested in c's body, so the node is too.
		var closed func(n int) bool
		closed = func(n int) bool {
			if g.Nodes[n].Kind == NIter {
				_, ok := e.get(n)
				return !ok
			}
			for _, o := range g.operands(n) {
				if closed(o) {
					return true
				}
			}
			return false
		}
		seen := make([]bool, len(g.Nodes))
		var hoist func(n int)
		hoist = func(n int) {
			if seen[n] {
				return
			}
			seen[n] = true
			if !vary[n] && !closed(n) {
				elem(n, idx, e)
				return
			}
			nd := g.Nodes[n]
			if nd.Kind == NFold && nd.At != c {
				// A loop nested in c's body: what its starts and steps read.
				for _, f := range g.Nodes {
					if f.Kind == NFold && f.At == nd.At {
						hoist(f.B)
						hoist(f.A)
					}
				}
				return
			}
			for _, o := range g.operands(n) {
				hoist(o)
			}
		}
		inits := make([]ir.Value, len(folds))
		for j, f := range folds {
			inits[j] = elem(g.Nodes[f].B, idx, e)
		}
		for _, f := range folds {
			hoist(g.Nodes[f].A)
		}
		zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
		b.Loop(int64(g.Loops[c]))
		in := newEnv(e)
		it := b.Phi(ir.U32, zero)
		for i, nd := range g.Nodes {
			if nd.Kind == NIter && nd.At == c {
				in.m[i] = it
			}
		}
		phis := make([]ir.Value, len(folds))
		for j, f := range folds {
			phis[j] = b.Phi(g.typeOf(f), inits[j])
			in.m[f] = phis[j]
		}
		next := make([]ir.Value, len(folds))
		for j, f := range folds {
			next[j] = elem(g.Nodes[f].A, idx, in)
		}
		for j := range folds {
			b.SetPhi(phis[j], next[j])
		}
		b.SetPhi(it, b.Add(ir.U32, it, one))
		b.EndLoop()
		for j, f := range folds {
			e.m[f] = phis[j]
		}
	}
	elem = func(n int, idx ir.Value, e *env) ir.Value {
		if v, ok := e.get(n); ok {
			return v
		}
		nd := g.Nodes[n]
		var v ir.Value
		switch nd.Kind {
		case NLoad:
			if nd.Indexed {
				at := elem(nd.Ix, idx, e)
				switch g.Bufs[nd.Buf].Kind {
				case BufU32:
					v = b.Load(ir.U32, params[nd.Buf], at, 0)
				case BufRowF16:
					v = half(params[nd.Buf], at)
				default:
					v = b.Load(ir.F32, params[nd.Buf], at, 0)
				}
				break
			}
			switch g.Bufs[nd.Buf].Kind {
			case BufU32:
				v = b.Load(ir.U32, params[nd.Buf], r, 0)
			case BufRow:
				v = b.Load(ir.F32, params[nd.Buf], b.Add(ir.U32, rowBase, idx), 0)
			case BufRowF16:
				v = half(params[nd.Buf], b.Add(ir.U32, rowBase, idx))
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
		case NIter:
			// Its loop's body binds it; read anywhere else it has no value.
			if emitErr == nil {
				emitErr = fmt.Errorf("compose: node %d reads loop %d's iterator outside its body", n, nd.At)
			}
			v = b.Const(ir.U32, 0)
		case NFold:
			loop(nd.At, idx, e)
			v, _ = e.get(n)
			return v
		case NUBinary:
			x, y := elem(nd.A, idx, e), elem(nd.B, idx, e)
			t := g.typeOf(nd.A)
			switch nd.Op {
			case OpAdd:
				v = b.Add(t, x, y)
			case OpSub:
				v = b.Sub(t, x, y)
			case OpMul:
				v = b.Mul(t, x, y)
			case OpDiv:
				v = b.Div(t, x, y)
			case OpRem:
				v = b.Rem(t, x, y)
			case OpMin:
				v = b.Min(t, x, y)
			case OpAnd:
				v = b.And(t, x, y)
			case OpXor:
				v = b.Xor(t, x, y)
			case OpShl:
				v = b.Shl(t, x, y)
			case OpShr:
				v = b.Shr(t, x, y)
			}
		case NIConst:
			v = b.Const(ir.I32, int64(nd.At))
		case NDot4:
			x, y, acc := elem(nd.A, idx, e), elem(nd.B, idx, e), elem(nd.C, idx, e)
			if nd.At == 1 {
				v = b.Dot4Bounded(x, y, acc)
			} else {
				v = b.Dot4(x, y, acc)
			}
		case NLess:
			v = b.Lt(g.typeOf(nd.A), elem(nd.A, idx, e), elem(nd.B, idx, e))
		case NSelect:
			v = b.Select(g.typeOf(nd.A), elem(nd.C, idx, e), elem(nd.A, idx, e), elem(nd.B, idx, e))
		case NReduce:
			v = rowVal[n]
		case NUnary:
			a := elem(nd.A, idx, e)
			switch nd.Op {
			case OpExp:
				v = b.Exp(a)
			case OpSqrt:
				v = b.Sqrt(a)
			case OpFloat:
				if g.typeOf(nd.A) == ir.I32 {
					v = b.CvtF32(a)
				} else {
					v = b.CvtF32(b.Bitcast(ir.I32, a))
				}
			case OpHalf:
				v = b.CvtF16H(a)
			case OpBits:
				v = b.Bitcast(ir.F32, a)
			case OpRsqrt:
				v = b.Div(ir.F32, b.ConstF32(1), b.Sqrt(a))
			case OpAbs:
				v = b.Max(ir.F32, a, b.Sub(ir.F32, b.ConstF32(0), a))
			}
		case NBinary:
			a := elem(nd.A, idx, e)
			switch nd.Op {
			case OpPackF16:
				v = b.PackF16(a, elem(nd.B, idx, e))
			case OpFma:
				v = b.Fma(a, elem(nd.B, idx, e), elem(nd.C, idx, e))
			default:
				c := elem(nd.B, idx, e)
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
		e.m[n] = v
		return v
	}
	if s.Flat {
		// One element a thread: every store at it, in one pass.
		e := newEnv(nil)
		for _, st := range g.Stores {
			at := b.Add(ir.U32, rowBase, flatIdx)
			if st.Indexed {
				at = elem(st.Ix, flatIdx, e)
			}
			b.Store(params[st.Buf], at, elem(st.Val, flatIdx, e), 0)
		}
		if emitErr != nil {
			return nil, emitErr
		}
		return b.Done(), nil
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
		e := newEnv(nil)
		for j, n := range reds {
			x := elem(g.Nodes[n].A, idx, e)
			if g.Nodes[n].Red == RedMax {
				b.SetPhi(accs[j], b.Select(ir.F32, in, b.Max(ir.F32, accs[j], x), accs[j]))
			} else {
				b.SetPhi(accs[j], b.Add(ir.F32, accs[j], b.Select(ir.F32, in, x, b.ConstF32(0))))
			}
		}
		for _, st := range stores {
			at := b.Add(ir.U32, rowBase, idx)
			if st.Indexed {
				at = elem(st.Ix, idx, e)
			}
			b.Store(params[st.Buf], at, elem(st.Val, idx, e), 0)
		}
		b.SetPhi(i, b.Add(ir.U32, i, step))
		b.EndLoop()
		for j, n := range reds {
			if g.Nodes[n].Red == RedMax {
				rowVal[n] = groupReduce(b, sh, tid, accs[j], G, true)
			} else {
				rowVal[n] = groupReduce(b, sh, tid, accs[j], G, false)
			}
		}
	}
	if emitErr != nil {
		return nil, emitErr
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
func (g *Row) isIndex(n int) bool { return g.typeOf(n) != ir.F32 }

// typeOf is the IR type of node n's value: a float, an unsigned index or
// word, or a signed integer (a dot product, a signed constant).
func (g *Row) typeOf(n int) ir.Type {
	nd := g.Nodes[n]
	switch nd.Kind {
	case NColumn, NRowIdx, NUConst, NIter:
		return ir.U32
	case NIConst, NDot4:
		return ir.I32
	case NUBinary:
		return g.typeOf(nd.A)
	case NFold:
		return g.typeOf(nd.B)
	case NSelect:
		return g.typeOf(nd.A)
	case NLoad:
		if g.Bufs[nd.Buf].Kind == BufU32 {
			return ir.U32
		}
	case NBinary:
		if nd.Op == OpPackF16 {
			return ir.U32
		}
	}
	return ir.F32
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
