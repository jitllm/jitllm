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
	"math/bits"

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
	// Frags is each matrix instruction's fragments (NMMA).
	Frags [][]int
	// Q8, where set, writes node Q8.Val as the int8 operand a matvec reads:
	// into buffers Q8.A and Q8.X at Q8.Window, the row padded to Q8.Pad.
	Q8 *Q8
	// Loops are the trip counts of the group's folds' loops (Loop): loop c
	// runs Loops[c] times, or where that is -1 (LoopN) the value of index
	// node Trip[c], known at run time.
	Loops []int
	Trip  []int
	// Stages are further element spaces of the row the workgroup stores
	// after the group's own (a row group, not flat): each of its own width,
	// its element loop unrolled, its nodes' Column its element, and what it
	// reads of the row's own elements read by NAt.
	Stages []Stage
	// Live, where HasLive, is the rows a flat group writes at run time: word
	// LiveAt of buffer LiveBuf (BufU32). The rest keep what they hold.
	HasLive         bool
	LiveBuf, LiveAt int
	// Ranges are the range loops' coordinates, by loop (RangeLoop).
	Ranges []Range
	// SpanGroup is the aligned rows that share a table and whose spans do
	// not decrease (HintTileFirst, HintTileLast).
	SpanGroup int
	// Effects are writes made for their effect alone, each element's
	// after its stores (Effect).
	Effects []int
}

// Stage is one further element space of a row group: its width and the
// stores at each of its elements.
type Stage struct {
	Width  int
	Stores []Store
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
	// BufShared and BufSharedU32 are a workgroup's own array of Buf.N floats
	// or words, no parameter: stored and read by index, a store reaching a
	// read across the barrier between their phases.
	BufShared
	BufSharedU32
)

// Buf is one of a group's buffers, a kernel parameter in order.
type Buf struct {
	Kind BufKind
	// N is a workgroup array's length.
	N int
	// Of, where not 0, is the array (its index plus one) whose storage this
	// one takes over: its stores after every read of that one, behind a
	// barrier, or with Over in the same phase, each thread rewriting only
	// places it read itself.
	Of   int
	Over bool
}

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
	// NSegment is the thread's segment under a split schedule (an index),
	// 0 without one.
	NSegment
	// NWindow is Red of A over the window of At consecutive elements
	// holding this one (under the flat schedule, At consecutive threads, in
	// order from the window's first), every element of the window holding
	// the same value. At must divide the group.
	NWindow
	// NAt is node A's value at element Ix (an index) of the same row: A
	// recomputed there from what it reads (RULE 13: no stage reads what
	// another wrote).
	NAt
	// Under a tile, the column is NColBase, the tile's first, plus NColOff,
	// the env's place in the tile (a constant), and the row NRowBase plus
	// NRowOff: what the tile's envs share apart from what they do not.
	NColBase
	NColOff
	NRowBase
	NRowOff
	// NShuffle is A's value at the lane At away (an xor of the lane) in the
	// subgroup, under lane rows.
	NShuffle
	// NRowTile and NColTile are the thread's tile among the tiles, before a
	// strided tile spaces it (Schedule.RowStride, ColStride).
	NRowTile
	NColTile
	// NHint is A's value, no instruction of its own, carrying a fact for the
	// passes (Hint, and its arguments).
	NHint
	// NTileLoad is a Tile of buffer Buf from index Ix, its rows B apart
	// (Major: its columns), after token C (-1 none); NTileSplat a Tile of
	// every element A; NTileMMA A times B plus the accumulator C.
	NTileLoad
	NTileSplat
	NTileMMA
	// NWrite stores into buffer Buf at index Ix -- A, and with Vec > 1
	// the values H[0..Vec-2] after it, one vector store; a tile A (Tile
	// set), its rows B apart -- after token C (-1 none), and is a token
	// itself: what reads it comes after the store.
	NWrite
	// NBarrier is the workgroup's barrier after token A (-1 none), a token.
	NBarrier
	// NSubgroup is the thread's subgroup in its workgroup.
	NSubgroup
	// NMMA is the subgroup's matrix instruction MMA over the fragments in
	// the group's Frags[At] (A's, B's, then the accumulator's), its first
	// component; NMMAGet component At of the NMMA at A.
	NMMA
	NMMAGet
	// NThread is the thread's number in the grid, unclamped.
	NThread
	// NLocal is the thread's number in its workgroup.
	NLocal
)

// HintKind is what a hint states.
type HintKind uint8

const (
	// HintTileFirst and HintTileLast: A, taken at the first (last) row of
	// any tile of an aligned group of Row.SpanGroup rows, serves every row of
	// that tile -- a value the group's rows share, the start (end) of a span
	// that only widens across them.
	HintTileFirst HintKind = iota + 1
	HintTileLast
	// HintKey: A is key B clamped into [C, Ix), read only for its page
	// (KeyClamp).
	HintKey
	// HintMultiple: A is a multiple of At (data written so, a table's
	// aligned entries).
	HintMultiple
	// HintAfter: A, taken after token B.
	HintAfter
	// HintLast: fold A's step as its loop's last trip made it, read after
	// the loop (the value the fold then holds, as the trip's own value).
	HintLast
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
	// OpWord is float A's 32 bits as a word, no conversion.
	OpWord
	// OpInt is float A truncated toward zero to a signed integer.
	OpInt
	// OpAsWord is signed integer A's bits as a word: a reinterpretation, which
	// a typed target (SPIR-V) states as a bitcast.
	OpAsWord
	// OpDivExact is A/B correctly rounded, whatever the target's division
	// allows (expandDivs writes it in integer arithmetic).
	OpDivExact
	// OpSubH2 and OpFmaH2 are words of two binary16 halves: A - B, and
	// A*B + C (C in Node.C), each half alone.
	OpSubH2
	OpFmaH2
)

// token is whether fold f carries an order rather than a value: it starts
// at a write or a barrier.
func (g *Row) token(f int) bool {
	k := g.Nodes[g.Nodes[f].B].Kind
	return k == NWrite || k == NBarrier
}

// ternary is whether nd reads a third operand at C as a multiply-add does.
func (nd Node) ternary() bool { return nd.Kind == NBinary && (nd.Op == OpFma || nd.Op == OpFmaH2) }

// Red is a row reduction.
type Red uint8

const (
	RedSum Red = iota
	RedMax
	// RedArgmax is the index of the row's greatest element, the lowest
	// where several are: an index.
	RedArgmax
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
	// Vec, 2 or 4, makes an indexed load component Comp of the Vec
	// consecutive elements from Ix, one vector load for all of them.
	Vec, Comp uint8
	// Hint and H are an NHint's fact and its further arguments.
	Hint HintKind
	H    [3]int32
	// Tile is a tile node's shape and role, Major its column-major layout.
	Tile  ir.TileType
	Major bool
	// MMA is an NMMA's instruction.
	MMA ir.MMAShape
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
	// Tile and TileCols, under Flat, are how many consecutive rows and
	// elements one thread computes: a node is computed once for every row
	// it does not read and every element it does not read (a weight word
	// for every token, a token's operand for every weight row), a fold
	// carries a value for each it does. Rows and Width must be multiples;
	// 0 and 1 are one.
	Tile, TileCols int
	// RowStride and ColStride space a tile's rows and elements that far
	// apart (0 and 1 are adjacent): tile t's first row is
	// t/RowStride*Tile*RowStride + t%RowStride, its rows that plus
	// j*RowStride, and the same for its elements.
	RowStride, ColStride int
	// Target is the device the kernel is formed for: a lowering whose best
	// form depends on it reads it here.
	Target Target
	// Split, under Flat, divides loop 0's trips between that many
	// threads, a segment each, the segment the slowest of the grid: a
	// segment's stores write their own plane of Rows*Width further on, for
	// a group that sums the planes (a matvec's k split).
	Split int
	// GroupSplit, under Split, puts a row's segments in one workgroup of
	// Group threads (Group/Split rows a workgroup, the row fastest, the
	// segment the thread over them): the partials meet in a workgroup array
	// (GroupFold) and every row's tokens are one thread's (Tile is Rows).
	GroupSplit bool
	// MMA, under Flat, runs the group a warp a tile of MT*16 elements by
	// NT*8 rows on the target's int8 matrix instruction (ir.OpMMA): a lane
	// holds elements g, g+8 of each 16 and rows 2t, 2t+1 of each 8 (g its
	// lane over 4, t its lane mod 4), and a fold whose step is an NDot4 of
	// a value that reads no row by one that reads no element, over 4 or 8
	// words, is the instruction's K of 16 or 32 bytes: its operands taken at
	// the lane's fragment, element g (+8) and row g, words t (+4).
	MMA    bool
	MT, NT int
	// Lanes, under Flat, runs a row over that many consecutive lanes of one
	// subgroup, an element a lane (Width, a power of two the target's
	// subgroup holds, or 1): the lane is the thread's own and the row clamped
	// to the last, and a reduction is a butterfly over the lanes, every lane
	// holding it -- a value in any scope, a loop's body too.
	Lanes int
	// WorkRow, under Flat, runs a row a workgroup of Group threads, the
	// element the thread itself and the row the workgroup, the grid the rows
	// exactly (a group of fewer elements clamps its own, and stores at its
	// own places): a row whose elements meet in workgroup arrays and lanes.
	WorkRow bool
	// LaneLoops, under Lanes, spreads a row of one element over the lanes,
	// each loop's trips dealt across them (laneLoops).
	LaneLoops bool
	// GroupRows, under Lanes, takes a row from the workgroup and the thread
	// (Group/Lanes rows a workgroup) rather than from the flat index.
	GroupRows bool
	// Block cuts each range loop into aligned blocks of that many places
	// (blocks), 0 and 1 for none.
	Block int
}

// Key is a group's identity in a cache: its structure and its schedule,
// hashed with the encoding kept to tell a collision from a hit.
type Key struct {
	Hash uint64
}

// Encode is the group's structure and schedule as integers.
func (g *Row) Encode(s Schedule) []uint32 {
	e := []uint32{1, uint32(g.Width), uint32(g.Rows), uint32(s.Group), b2u(s.Warp), uint32(len(g.Bufs)), b2u(s.Flat), uint32(s.Tile), uint32(s.TileCols), uint32(s.Split),
		b2u(s.MMA), uint32(s.MT), uint32(s.NT), uint32(s.Target.API), s.Target.ISA.Word(), uint32(s.Target.Lanes), uint32(s.Lanes), b2u(s.WorkRow), uint32(s.RowStride), uint32(s.ColStride), b2u(s.LaneLoops), uint32(s.Block), b2u(s.GroupRows), b2u(s.GroupSplit)}
	for _, b := range g.Bufs {
		e = append(e, uint32(b.Kind), uint32(b.N), uint32(b.Of), b2u(b.Over))
	}
	for _, n := range g.Nodes {
		e = append(e, uint32(n.Kind), uint32(n.Op), uint32(n.Act), uint32(n.Red), uint32(n.Buf),
			uint32(n.A), uint32(n.B), uint32(n.C), f2u(n.F), uint32(n.At), b2u(n.Indexed), uint32(n.Ix), uint32(n.Vec)<<8|uint32(n.Comp))
		if n.Kind == NHint || n.Kind == NWrite {
			e = append(e, uint32(n.Hint), uint32(n.H[0]), uint32(n.H[1]), uint32(n.H[2]))
		}
		if n.Kind == NTileLoad || n.Kind == NTileSplat || n.Kind == NWrite {
			e = append(e, uint32(n.Tile.Rows)<<16|uint32(n.Tile.Cols), uint32(n.Tile.Elem)<<8|uint32(n.Tile.Use), b2u(n.Major))
		}
		if n.Kind == NMMA {
			e = append(e, uint32(n.MMA.Kind)<<24|uint32(n.MMA.M)<<16|uint32(n.MMA.N)<<8|uint32(n.MMA.K), uint32(len(g.Frags[n.At])))
			for _, a := range g.Frags[n.At] {
				e = append(e, uint32(a))
			}
		}
	}
	for _, s := range g.Stores {
		e = append(e, uint32(s.Buf), uint32(s.Val), b2u(s.Indexed), uint32(s.Ix))
	}
	for _, n := range g.Effects {
		e = append(e, 0xEFF, uint32(n))
	}
	if q := g.Q8; q != nil {
		e = append(e, uint32(q.Val), uint32(q.A), uint32(q.X), uint32(q.Window), uint32(q.Pad))
	}
	if g.HasLive {
		e = append(e, 0x11E, uint32(g.LiveBuf), uint32(g.LiveAt))
	}
	for _, st := range g.Stages {
		e = append(e, 0x57A6, uint32(st.Width), uint32(len(st.Stores)))
		for _, x := range st.Stores {
			e = append(e, uint32(x.Buf), uint32(x.Val), b2u(x.Indexed), uint32(x.Ix))
		}
	}
	if len(g.Loops) > 0 {
		e = append(e, 0xF01D, uint32(len(g.Loops)))
		for _, t := range g.Loops {
			e = append(e, uint32(t))
		}
		for c := range g.Loops {
			e = append(e, uint32(g.tripOf(c)))
		}
	}
	if g.SpanGroup > 0 {
		e = append(e, 0x5BA2, uint32(g.SpanGroup))
	}
	for c, rg := range g.Ranges {
		if rg.ok {
			e = append(e, 0x2A6E, uint32(c), uint32(rg.S), uint32(rg.Lo), uint32(rg.Hi), uint32(rg.Origin), uint32(rg.Ext))
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
			case NUnary, NWindow, NShuffle:
				v = d[n.A]
			case NBinary, NUBinary, NLess:
				v = max(d[n.A], d[n.B])
				if n.ternary() {
					v = max(v, d[n.C])
				}
			case NSelect, NDot4:
				v = max(d[n.A], d[n.B], d[n.C])
			case NAt:
				v = max(d[n.A], d[n.Ix])
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
			if k := g.Bufs[n.Buf].Kind; k != BufRow && k != BufVec && k != BufScalar && k != BufRowScalar && k != BufU32 && k != BufRowF16 && k != BufShared && k != BufSharedU32 {
				return fmt.Errorf("compose: node %d loads an output buffer", i)
			}
		case NTileLoad, NWrite:
			if n.Buf < 0 || n.Buf >= len(g.Bufs) {
				return fmt.Errorf("compose: node %d reaches buffer %d of %d", i, n.Buf, len(g.Bufs))
			}
			in = g.operands(i)
		case NTileSplat, NTileMMA, NBarrier, NMMAGet:
			in = g.operands(i)
		case NMMA:
			na, nb, nc := n.MMA.Frags()
			if !n.MMA.Valid() || n.At < 0 || n.At >= len(g.Frags) || len(g.Frags[n.At]) != na+nb+nc {
				return fmt.Errorf("compose: node %d is a %dx%dx%d product without its fragments", i, n.MMA.M, n.MMA.N, n.MMA.K)
			}
			in = g.Frags[n.At]
		case NUnary, NReduce, NWindow, NShuffle:
			in = []int{n.A}
		case NBinary, NUBinary, NLess:
			in = []int{n.A, n.B}
			if n.ternary() {
				in = append(in, n.C)
			}
		case NSelect, NDot4:
			in = []int{n.A, n.B, n.C}
		case NAt:
			in = []int{n.A, n.Ix}
		case NIter:
			if n.At < 0 || n.At >= len(g.Loops) {
				return fmt.Errorf("compose: node %d iterates loop %d of %d", i, n.At, len(g.Loops))
			}
		case NFold:
			if n.At < 0 || n.At >= len(g.Loops) || g.Loops[n.At] < 1 && g.tripOf(n.At) < 0 {
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
		if s.Buf < 0 || s.Buf >= len(g.Bufs) || g.Bufs[s.Buf].Kind != BufOut && g.Bufs[s.Buf].Kind != BufOutU32 && !g.shared(s.Buf) || s.Val < 0 || s.Val >= len(g.Nodes) ||
			s.Indexed && (s.Ix < 0 || s.Ix >= len(g.Nodes)) {
			return fmt.Errorf("compose: a store of node %d into buffer %d", s.Val, s.Buf)
		}
	}
	if g.HasLive && (g.LiveBuf < 0 || g.LiveBuf >= len(g.Bufs) || g.Bufs[g.LiveBuf].Kind != BufU32 || g.LiveAt < 0) {
		return fmt.Errorf("compose: a live row count in buffer %d", g.LiveBuf)
	}
	for _, st := range g.Stages {
		if st.Width < 1 {
			return fmt.Errorf("compose: a stage of %d elements", st.Width)
		}
		for _, x := range st.Stores {
			if x.Buf < 0 || x.Buf >= len(g.Bufs) || g.Bufs[x.Buf].Kind != BufOut && g.Bufs[x.Buf].Kind != BufOutU32 ||
				x.Val < 0 || x.Val >= len(g.Nodes) || x.Indexed && (x.Ix < 0 || x.Ix >= len(g.Nodes)) {
				return fmt.Errorf("compose: a stage's store of node %d into buffer %d", x.Val, x.Buf)
			}
		}
	}
	if g.Q8 != nil {
		// The int8 operand (a scale per window, rounded codes packed four
		// to a word, sums per 16) needs primitives the node language does
		// not have yet: windowed reductions, float to integer, bit packing.
		return fmt.Errorf("compose: an int8 operand has no primitive form yet")
	}
	if _, ok := g.depth(); !ok && !g.loopReduces() {
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
	// Under a tile: the row (and its first element) and the column this
	// env runs for, zero for the kernel's own; sh the envs a node is read
	// from by what it depends on -- sh[0] one reading neither row nor
	// column, sh[1] the row alone, sh[2] the column alone -- nil where this
	// env is that one; cls what this env's nodes depend on (bit 1 the row,
	// bit 2 the column); and t the tile's envs.
	row, base, col ir.Value
	// rowOff and colOff are this env's place in its tile.
	rowOff, colOff int
	// slot is the thread's place in its window space (NWindow): its
	// element less the first of its workgroup's span.
	slot ir.Value
	// rows says a node that reads no element is read from the row env
	// where it was hoisted there.
	rows bool
	sh   [3]*env
	cls  uint8
	t    *tile
	// own is the nodes computed in this env whatever they read (a
	// fragment's: what reads the iterator it binds), nil for none.
	own []bool
	// last is each fold's step as its loop's last trip left it (HintLast).
	last map[int]ir.Value
}

// carried is a fold a loop carries in one env.
type carried struct {
	f   int
	in  *env // the body env it is read in
	out *env // where its value after the loop goes
	phi ir.Value
}

// unrollTrips is the most trips a loop nested in another's body runs
// unrolled: a sub-block's words, a block's elements.
const unrollTrips = 32

// tile is every env of one tile: one a (row, column) pair, one a row, one
// a column and one for neither, each a node is computed in once.
type tile struct {
	all []*env
	// mma is the tile's matrix layout under an MMA schedule, nil otherwise.
	mma *mmaTile
}

// mmaTile is a lane's place in an MMA schedule's warp tile: its envs by
// what they hold, and the lane's word in a fragment (t).
type mmaTile struct {
	mt, nt int
	none   *env
	// cols[x] is element g + (x/2)*16 + (x%2)*8 of the warp's; rows[y] row
	// 2t + (y/2)*8 + y%2; brows[j] row g + j*8, a B fragment's; outs[x][y]
	// each output.
	cols, rows, brows []*env
	outs              [][]*env
	// word is the lane's word in a fragment (t), g its place among a
	// tile's 8, colBase the lane's first element, tokBase the warp's first
	// row and tok0 the lane's.
	word, g, colBase, tokBase, tok0 ir.Value
}

// mmaEnvs builds m's envs -- its elements, its output rows, its B
// fragments' rows and every output, each over what it reads, under one tile
// (the B rows beside it: no output reads them) -- and returns the outputs.
func mmaEnvs(b *ir.Builder, m *mmaTile, width ir.Value) []*env {
	t := &tile{mma: m}
	mk := func(cls uint8) *env {
		x := newEnv(nil)
		x.cls, x.t = cls, t
		t.all = append(t.all, x)
		return x
	}
	off := func(v ir.Value, n int) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, b.Const(ir.U32, int64(n)))
	}
	m.none = mk(0)
	row := func(x *env, at ir.Value) {
		x.row, x.base, x.col = at, b.Mul(ir.U32, at, width), m.colBase
		x.sh[0] = m.none
	}
	for x := 0; x < 2*m.mt; x++ {
		e := mk(2)
		e.col = off(m.colBase, (x/2)*16+(x%2)*8)
		e.row, e.base = m.tokBase, m.tokBase
		e.sh[0] = m.none
		m.cols = append(m.cols, e)
	}
	for y := 0; y < 2*m.nt; y++ {
		e := mk(1)
		row(e, off(m.tok0, (y/2)*8+y%2))
		m.rows = append(m.rows, e)
	}
	for j := 0; j < m.nt; j++ {
		e := newEnv(nil)
		e.cls = 1
		row(e, off(b.Add(ir.U32, m.tokBase, m.g), j*8))
		m.brows = append(m.brows, e)
	}
	var outs []*env
	for x := range m.cols {
		var o []*env
		for y := range m.rows {
			e := mk(3)
			e.row, e.base, e.col = m.rows[y].row, m.rows[y].base, m.cols[x].col
			e.sh = [3]*env{m.none, m.rows[y], m.cols[x]}
			o = append(o, e)
			outs = append(outs, e)
		}
		m.outs = append(m.outs, o)
	}
	return outs
}

// remap is m with every env replaced by its body env in ins.
func (m *mmaTile) remap(ins map[*env]*env) *mmaTile {
	n := &mmaTile{mt: m.mt, nt: m.nt, none: ins[m.none], word: m.word, g: m.g, colBase: m.colBase, tokBase: m.tokBase, tok0: m.tok0}
	for _, x := range m.cols {
		n.cols = append(n.cols, ins[x])
	}
	for _, x := range m.rows {
		n.rows = append(n.rows, ins[x])
	}
	for _, x := range m.brows {
		n.brows = append(n.brows, ins[x])
	}
	for _, o := range m.outs {
		var r []*env
		for _, x := range o {
			r = append(r, ins[x])
		}
		n.outs = append(n.outs, r)
	}
	return n
}

// setLast records fold f's last step in e.
func (e *env) setLast(f int, v ir.Value) {
	if e.last == nil {
		e.last = map[int]ir.Value{}
	}
	e.last[f] = v
}

// lastOf is fold f's last step, walking out to the env that ran its loop.
func (e *env) lastOf(f int) (ir.Value, bool) {
	for x := e; x != nil; x = x.up {
		if v, ok := x.last[f]; ok {
			return v, true
		}
	}
	return 0, false
}

func (e *env) get(n int) (ir.Value, bool) {
	for x := e; x != nil; x = x.up {
		if v, ok := x.m[n]; ok {
			return v, true
		}
	}
	return 0, false
}

// at is e's row, row base and column, walking out to the env that states
// them; zero where none does (the kernel's own).
func (e *env) at() (row, base, col ir.Value) {
	for x := e; x != nil; x = x.up {
		if row == 0 && x.row != 0 {
			row, base = x.row, x.base
		}
		if col == 0 && x.col != 0 {
			col = x.col
		}
	}
	return row, base, col
}

// readsRow reports whether e (or an env it is in) reads hoisted row values.
func (e *env) readsRow() bool {
	for x := e; x != nil; x = x.up {
		if x.rows {
			return true
		}
	}
	return false
}

// slotOf is e's window slot, zero where none is stated.
func (e *env) slotOf() ir.Value {
	for x := e; x != nil; x = x.up {
		if x.slot != 0 {
			return x.slot
		}
	}
	return 0
}

// tileOf is e's tile, nil outside one.
func (e *env) tileOf() *tile {
	for x := e; x != nil; x = x.up {
		if x.t != nil {
			return x.t
		}
	}
	return nil
}

func newEnv(up *env) *env { return &env{m: map[int]ir.Value{}, up: up} }

// operands are the nodes node n reads where it is emitted: not a fold's
// step (its loop's body), and not a reduction's operand (an earlier pass).
func (g *Row) operands(n int) []int {
	nd := g.Nodes[n]
	some := func(xs ...int) []int {
		var o []int
		for _, x := range xs {
			if x >= 0 {
				o = append(o, x)
			}
		}
		return o
	}
	switch nd.Kind {
	case NHint:
		if nd.Hint == HintAfter {
			return []int{nd.A, nd.B}
		}
		return []int{nd.A}
	case NTileLoad:
		return some(nd.Ix, nd.B, nd.C)
	case NTileSplat:
		return []int{nd.A}
	case NTileMMA:
		return []int{nd.A, nd.B, nd.C}
	case NWrite:
		o := some(nd.Ix, nd.A, nd.B, nd.C)
		for i := 0; i < int(nd.Vec)-1; i++ {
			o = append(o, int(nd.H[i]))
		}
		return o
	case NBarrier:
		return some(nd.A)
	case NMMA:
		return g.Frags[nd.At]
	case NMMAGet:
		return []int{nd.A}
	case NUnary, NWindow, NShuffle:
		return []int{nd.A}
	case NBinary:
		if nd.ternary() {
			return []int{nd.A, nd.B, nd.C}
		}
		return []int{nd.A, nd.B}
	case NUBinary, NLess:
		return []int{nd.A, nd.B}
	case NSelect, NDot4:
		return []int{nd.A, nd.B, nd.C}
	case NAt:
		return []int{nd.A, nd.Ix}
	case NLoad:
		if nd.Indexed {
			return []int{nd.Ix}
		}
	case NFold:
		return []int{nd.B}
	}
	return nil
}

// rowDeps is which nodes depend on the group's row: the row index, a load
// at the row's own place, and whatever reads them (a fixed point, since a
// fold reads later nodes).
// loopOperands is what node n reads for a loop's purposes: its operands, and a
// reduction's operand, since a reduction of what a loop changes changes with it.
func (g *Row) loopOperands(n int) []int {
	if nd := g.Nodes[n]; nd.Kind == NReduce {
		return []int{nd.A}
	}
	return g.operands(n)
}

func (g *Row) rowDeps() []bool {
	v := make([]bool, len(g.Nodes))
	for changed := true; changed; {
		changed = false
		for i, nd := range g.Nodes {
			if v[i] {
				continue
			}
			x := false
			switch nd.Kind {
			case NRowIdx, NRowOff:
				x = true
			case NLoad:
				if !nd.Indexed {
					k := g.Bufs[nd.Buf].Kind
					x = k == BufRow || k == BufRowF16 || k == BufRowScalar || k == BufU32
				}
			case NFold:
				x = v[nd.B] || nd.A >= 0 && v[nd.A]
			}
			for _, o := range g.operands(i) {
				x = x || v[o]
			}
			if x {
				v[i], changed = true, true
			}
		}
	}
	return v
}

// colDeps is which nodes depend on the element's column: the column, a
// load at the element's own place, and whatever reads them.
func (g *Row) colDeps() []bool {
	v := make([]bool, len(g.Nodes))
	for changed := true; changed; {
		changed = false
		for i, nd := range g.Nodes {
			if v[i] {
				continue
			}
			x := false
			switch nd.Kind {
			case NColumn, NColOff, NWrite, NBarrier, NSubgroup, NThread, NLocal:
				x = true
			case NLoad:
				if !nd.Indexed {
					k := g.Bufs[nd.Buf].Kind
					x = k == BufRow || k == BufRowF16 || k == BufVec
				}
			case NFold:
				x = v[nd.B] || nd.A >= 0 && v[nd.A]
			}
			for _, o := range g.operands(i) {
				x = x || v[o]
			}
			if x {
				v[i], changed = true, true
			}
		}
	}
	return v
}

// encloses reports whether loop d holds loop c in its body: c's trip count
// or one of its folds' starts reads d's iterator or folds, so c runs anew
// each trip of d, and d's values stay put through c's trips.
func (g *Row) encloses(d, c int) bool {
	seen := map[int]bool{}
	var reads func(n int) bool
	reads = func(n int) bool {
		if n < 0 || seen[n] {
			return false
		}
		seen[n] = true
		nd := g.Nodes[n]
		if (nd.Kind == NIter || nd.Kind == NFold) && nd.At == d {
			return true
		}
		if nd.Kind == NFold {
			return false
		}
		for _, o := range g.operands(n) {
			if reads(o) {
				return true
			}
		}
		return false
	}
	if reads(g.tripOf(c)) {
		return true
	}
	for _, f := range g.Nodes {
		if f.Kind == NFold && f.At == c && reads(f.B) {
			return true
		}
	}
	return false
}

// varies is which nodes change from trip to trip of loop c: its iterator,
// its folds, and whatever reads them -- a fold of another loop whose start
// or step does is a loop nested in c's body. A fixed point, since a fold
// reads later nodes.
func (g *Row) varies(c int) []bool {
	v := make([]bool, len(g.Nodes))
	outer := map[int]bool{}
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
				enc, ok := outer[nd.At]
				if !ok {
					enc = nd.At != c && g.encloses(nd.At, c)
					outer[nd.At] = enc
				}
				for _, f := range g.Nodes {
					if !enc && f.Kind == NFold && f.At == nd.At && (v[f.B] || v[f.A]) {
						x = true
					}
				}
			default:
				for _, o := range g.loopOperands(i) {
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

// Emit forms the group's kernel under schedule s and runs the IR passes over
// it.
func (g *Row) Emit(s Schedule) (*ir.Kernel, error) {
	k, err := g.form(s)
	if err != nil {
		return nil, err
	}
	return simplify(k), nil
}

// form is the group's kernel under schedule s as the former emits it.
func (g *Row) form(s Schedule) (*ir.Kernel, error) {
	g, err := g.prepared(s)
	if err != nil {
		return nil, err
	}
	b := ir.New("row", [3]int{s.Group, 1, 1})
	return g.emitIn(b, g.bind(b, nil), s, b.CTAID, true)
}

// prepared is g as the former emits it under s: its activations expanded
// to primitives and the schedule's rewrites taken.
func (g *Row) prepared(s Schedule) (*Row, error) {
	if err := g.Check(); err != nil {
		return nil, err
	}
	// Activations are formulas over primitive nodes; the former emits
	// primitives only.
	g, _, err := g.expanded()
	if err != nil {
		return nil, err
	}
	if g, err = g.laneLoops(s); err != nil {
		return nil, err
	}
	g = g.blocks(s).pipelined().tiled(s).hulls(s).inductions(s)
	if s.Lanes == 0 && g.loopReduces() {
		return nil, fmt.Errorf("compose: a reduction in a loop's body off lane rows")
	}
	G := s.Group
	if G < 32 || G&(G-1) != 0 && !(s.WorkRow && G%32 == 0) {
		return nil, fmt.Errorf("compose: a group of %d threads", G)
	}
	return g, nil
}

// bind is each of g's buffers as a value of b: a parameter, or a workgroup
// array, those of have taken from it where it names one.
func (g *Row) bind(b *ir.Builder, have func(i int) (ir.Value, bool)) []ir.Value {
	params := make([]ir.Value, len(g.Bufs))
	for i, bf := range g.Bufs {
		if have != nil {
			if v, ok := have(i); ok {
				params[i] = v
				continue
			}
		}
		t := bufType(bf.Kind)
		if g.shared(i) {
			params[i] = b.Shared(fmt.Sprintf("s%d", i), t, bf.N)
			if bf.Of > 0 {
				params[i] = params[bf.Of-1]
			}
			continue
		}
		params[i] = b.Param(fmt.Sprintf("p%d", i), t)
	}
	return params
}

func bufType(k BufKind) ir.Type {
	if k == BufQ8 || k == BufU32 || k == BufOutU32 || k == BufRowF16 || k == BufSharedU32 {
		return ir.U32
	}
	return ir.F32
}

// emitIn forms prepared group g under s into b, its buffers params and cta
// the workgroup's place among the group's; done ends the kernel.
func (g *Row) emitIn(b *ir.Builder, params []ir.Value, s Schedule, cta func() ir.Value, done bool) (*ir.Kernel, error) {
	G := s.Group
	depth, _ := g.depth()
	// The row and, under the flat schedule, the element: a workgroup a row
	// with the element looping over the group, or a thread an element of
	// every row, the element fastest (the last thread's surplus clamped to
	// the last element, which it recomputes: RULE 13).
	var sh, r, rowBase, tid, width, last, flatIdx ir.Value
	rs, cs := 1, 1
	var rawRow, rawCol ir.Value
	// shi is the index plane an argmax's tree reduces beside sh.
	var shi ir.Value
	// seg is the thread's segment of loop 0 under a split, segTrip its
	// trips.
	var seg ir.Value
	segTrip := 0
	// flatRaw is the flat schedule's clamped thread number; windowSh each
	// window node's threadgroup array, declared before any loop.
	var flatRaw ir.Value
	// mma is the warp tile's layout under an MMA schedule.
	var mma *mmaTile
	windowSh := map[int]ir.Value{}
	// vecs is each vector load emitted, by its buffer, place and width;
	// mmaOut each matrix instruction's components, by its first.
	vecs := map[vecKey][]ir.Value{}
	mmaOut := map[ir.Value][]ir.Value{}
	for i, nd := range g.Nodes {
		if nd.Kind == NWindow {
			windowSh[i] = b.Shared(fmt.Sprintf("win%d", i), ir.F32, G)
		}
	}
	if s.Flat {
		for i, nd := range g.Nodes {
			if nd.Kind == NReduce && depth[i] > 0 && s.Lanes == 0 {
				return nil, fmt.Errorf("compose: a flat schedule over a row reduction")
			}
		}
		tr, tc := max(s.Tile, 1), max(s.TileCols, 1)
		if g.Rows%tr != 0 || g.Width%tc != 0 {
			return nil, fmt.Errorf("compose: a tile of %dx%d over %dx%d", tr, tc, g.Rows, g.Width)
		}
		cols := g.Width / tc
		if rs, cs = max(s.RowStride, 1), max(s.ColStride, 1); g.Rows/tr%rs != 0 || cols%cs != 0 {
			return nil, fmt.Errorf("compose: a tile of %dx%d strided %dx%d over %dx%d", tr, tc, rs, cs, g.Rows, g.Width)
		}
		split := max(s.Split, 1)
		if s.MMA && (!s.Target.IntMMA() || tr*tc > 1 || split > 1 || g.Width%(16*max(s.MT, 1)) != 0 || g.Rows%(8*max(s.NT, 1)) != 0) {
			return nil, fmt.Errorf("compose: a %dx%d warp tile over %dx%d, tile %dx%d split %d, on %v",
				s.MT, s.NT, g.Width, g.Rows, tr, tc, split, s.Target.ISA)
		}
		if split > 1 && (len(g.Loops) == 0 || g.Loops[0]%split != 0) {
			return nil, fmt.Errorf("compose: a split of %d over a loop of %v trips", split, g.Loops)
		}
		flat := b.Add(ir.U32, b.Mul(ir.U32, cta(), b.NTID()), b.TID())
		if s.MMA {
			// A warp a tile, the element's tiles fastest; a lane's place in
			// it (g, t), the surplus warps clamped to the last tile.
			mt, nt := max(s.MT, 1), max(s.NT, 1)
			u := func(n int) ir.Value { return b.Const(ir.U32, int64(n)) }
			mtiles, ntiles := g.Width/(16*mt), g.Rows/(8*nt)
			warp := b.Div(ir.U32, flat, u(ir.SubgroupLanes))
			lane := b.And(ir.U32, flat, u(ir.SubgroupLanes-1))
			lg, lt := b.Shr(ir.U32, lane, u(2)), b.And(ir.U32, lane, u(3))
			warp = b.Min(ir.U32, warp, u(mtiles*ntiles-1))
			wm, wn := b.Rem(ir.U32, warp, u(mtiles)), b.Div(ir.U32, warp, u(mtiles))
			colBase := b.Add(ir.U32, b.Mul(ir.U32, wm, u(16*mt)), lg)
			tokBase := b.Mul(ir.U32, wn, u(8*nt))
			width = u(g.Width)
			// The kernel's own row and element are no env's: what reads
			// neither is all that reads them.
			r, rowBase, flatIdx = tokBase, tokBase, colBase
			mma = &mmaTile{mt: mt, nt: nt, word: lt, g: lg, colBase: colBase, tokBase: tokBase,
				tok0: b.Add(ir.U32, tokBase, b.Mul(ir.U32, lt, u(2)))}
		} else if s.WorkRow {
			if tr*tc > 1 || split > 1 || g.HasLive || g.Width > G {
				return nil, fmt.Errorf("compose: a row of %d a workgroup of %d", g.Width, G)
			}
			u := func(n int) ir.Value { return b.Const(ir.U32, int64(n)) }
			flatIdx, r = b.TID(), cta()
			flatRaw = flat
			width = u(g.Width)
			rowBase = b.Mul(ir.U32, r, width)
		} else if s.Lanes > 0 {
			L := s.Lanes
			if L != g.Width || L&(L-1) != 0 || L > max(s.Target.Lanes, 1) || tr*tc > 1 || split > 1 || g.HasLive {
				return nil, fmt.Errorf("compose: a row of %d over %d lanes on a subgroup of %d", g.Width, L, s.Target.Lanes)
			}
			u := func(n int) ir.Value { return b.Const(ir.U32, int64(n)) }
			switch {
			case L == G:
				// A workgroup a row.
				flatIdx, r = b.And(ir.U32, b.TID(), u(L-1)), b.Min(ir.U32, cta(), u(g.Rows-1))
			case L == 1:
				flatIdx, r = u(0), b.Min(ir.U32, flat, u(g.Rows-1))
			case s.GroupRows:
				// G/L rows a workgroup, a row's lanes consecutive threads.
				flatIdx = b.Rem(ir.U32, b.TID(), u(L))
				r = b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, cta(), u(G/L)), b.Div(ir.U32, b.TID(), u(L))), u(g.Rows-1))
			default:
				flatIdx, r = b.And(ir.U32, flat, u(L-1)), b.Min(ir.U32, b.Div(ir.U32, flat, u(L)), u(g.Rows-1))
			}
			flatRaw = flat
			width = u(g.Width)
			rowBase = b.Mul(ir.U32, r, width)
		} else if s.GroupSplit {
			u := func(n int) ir.Value { return b.Const(ir.U32, int64(n)) }
			rpg := G / split
			if split < 2 || G%split != 0 || g.Width%rpg != 0 || tr != g.Rows || tc > 1 || g.HasLive || len(g.Loops) == 0 || g.Loops[0]%split != 0 {
				return nil, fmt.Errorf("compose: a split of %d in a workgroup of %d over %dx%d", split, G, g.Rows, g.Width)
			}
			tid := b.TID()
			local := b.Rem(ir.U32, tid, u(rpg))
			seg = b.Div(ir.U32, tid, u(rpg))
			segTrip = g.Loops[0] / split
			flatIdx = b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, cta(), u(rpg)), local), u(g.Width-1))
			r = u(0)
			flatRaw, rawRow, rawCol = flatIdx, r, flatIdx
			width = u(g.Width)
			rowBase = b.Mul(ir.U32, r, width)
		} else {
			if g.HasLive {
				if tr*tc > 1 || split > 1 {
					return nil, fmt.Errorf("compose: live rows under a tile or a split")
				}
				live := b.Load(ir.U32, params[g.LiveBuf], b.Const(ir.U32, int64(g.LiveAt)), 0)
				w := b.Const(ir.U32, int64(g.Width))
				lim := b.Max(ir.U32, w, b.Min(ir.U32, b.Mul(ir.U32, live, w), b.Const(ir.U32, int64(g.Rows*g.Width))))
				flat = b.Min(ir.U32, flat, b.Sub(ir.U32, lim, b.Const(ir.U32, 1)))
			} else {
				flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(g.Rows/tr*cols*split-1)))
			}
			flatRaw = flat
			width = b.Const(ir.U32, int64(g.Width))
			tcols := width
			if tc > 1 {
				tcols = b.Const(ir.U32, int64(cols))
			}
			// The element fastest, then the row, the segment slowest (a
			// plane of Rows*Width a segment): the column first, the rest
			// cut into row and segment.
			switch {
			case g.Rows/tr == 1 && split == 1:
				flatIdx, r = flat, b.Const(ir.U32, 0)
			default:
				flatIdx = b.Rem(ir.U32, flat, tcols)
				rest := b.Div(ir.U32, flat, tcols)
				r = rest
				if split > 1 {
					rows := b.Const(ir.U32, int64(g.Rows/tr))
					r, seg = b.Rem(ir.U32, rest, rows), b.Div(ir.U32, rest, rows)
					segTrip = g.Loops[0] / split
				}
			}
			rawRow, rawCol = r, flatIdx
			flatIdx = strided(b, flatIdx, tc, cs)
			r = strided(b, r, tr, rs)
			rowBase = b.Mul(ir.U32, r, width)
		}
	} else {
		sh = b.Shared("red", ir.F32, G)
		for _, nd := range g.Nodes {
			if nd.Kind == NReduce && nd.Red == RedArgmax && shi == 0 {
				shi = b.Shared("redi", ir.U32, G)
			}
		}
		r = b.Min(ir.U32, cta(), b.Const(ir.U32, int64(g.Rows-1)))
		rowBase = b.Mul(ir.U32, r, b.Const(ir.U32, int64(g.Width)))
		tid = b.TID()
		width = b.Const(ir.U32, int64(g.Width))
		last = b.Const(ir.U32, int64(g.Width-1))
	}
	overRows := s.Flat && mma == nil && max(s.Split, 1) == 1 && s.Lanes == 0 && !s.WorkRow && max(s.Tile, 1)*max(s.TileCols, 1) == 1 && g.Rows > 1
	col := g.colDeps()
	rowVal := make([]ir.Value, len(g.Nodes)) // a reduction's result, once its pass ran
	// rowEnv holds the row's values that read no element (a norm's scale),
	// computed once where the passes first need them (hoist) and read by
	// every element after.
	rowEnv := newEnv(nil)

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
	// rowWin is a stage's window that covers its whole row, met once before
	// the stage's elements.
	rowWin := map[int]ir.Value{}
	// loop emits fold loop c in scope e: first what its body reads that
	// does not change from trip to trip, then the loop, its folds' values
	// after it left in e.
	rowDep, colDep := g.rowDeps(), g.colDeps()
	clsOf := func(n int) uint8 {
		c := uint8(0)
		if rowDep[n] {
			c |= 1
		}
		if colDep[n] {
			c |= 2
		}
		return c
	}
	// open is how many loops are being emitted around the one emitting now.
	open := 0
	// mmaLoop emits fold loop c as the matrix instruction where it is one
	// (Schedule.MMA): one fold, its step an NDot4 of itself, of a value that
	// reads no row (A) by one that reads no element (B), over 4 or 8 words.
	// A lane's fragments are A at its elements and B at its B rows, the
	// loop's iterator bound to its word (and the word 4 on); each output's
	// fold start is its accumulator, and the instruction's components are
	// the fold's value at the outputs. False, and nothing emitted, where the
	// loop is not one.
	mmaLoop := func(c int, folds []int, m *mmaTile, idx ir.Value, vary []bool) bool {
		if len(folds) != 1 {
			return false
		}
		f := folds[0]
		st := g.Nodes[g.Nodes[f].A]
		words := g.Loops[c]
		if g.Nodes[f].A < 0 || st.Kind != NDot4 || st.C != f || words != 4 && words != 8 || rowDep[st.A] || colDep[st.B] {
			return false
		}
		sh := ir.MMAShape{M: 16, N: 8, K: 4 * words, Kind: ir.MMAS8}
		// frag is node n in a fresh env at the given row and column envs
		// (nil for one n does not read), the iterator at the lane's word
		// and q*4 on: what reads the iterator computed there, the rest where
		// the tile computes it.
		frag := func(n int, rowE, colE *env, q int) ir.Value {
			fe := newEnv(nil)
			fe.cls, fe.own = 3, vary
			fe.sh = [3]*env{m.none, rowE, colE}
			if rowE != nil {
				fe.row, fe.base, _ = rowE.at()
			}
			var col ir.Value
			if colE != nil {
				_, _, col = colE.at()
				fe.col = col
			}
			w := m.word
			if q > 0 {
				w = b.Add(ir.U32, w, b.Const(ir.U32, int64(4*q)))
			}
			for i, nd := range g.Nodes {
				if nd.Kind == NIter && nd.At == c {
					fe.m[i] = w
				}
			}
			return elem(n, col, fe)
		}
		halves := words / 4
		af := make([][]ir.Value, m.mt)
		for i := range af {
			af[i] = make([]ir.Value, 2*halves)
			for q := 0; q < halves; q++ {
				for h := 0; h < 2; h++ {
					af[i][q*2+h] = frag(st.A, nil, m.cols[2*i+h], q)
				}
			}
		}
		bf := make([][]ir.Value, m.nt)
		for j := range bf {
			bf[j] = make([]ir.Value, halves)
			for q := range bf[j] {
				bf[j][q] = frag(st.B, m.brows[j], nil, q)
			}
		}
		for i := 0; i < m.mt; i++ {
			for j := 0; j < m.nt; j++ {
				var cf [4]*env
				cv := make([]ir.Value, len(cf))
				for k := range cf {
					cf[k] = m.outs[2*i+k/2][2*j+k%2]
					cv[k] = elem(g.Nodes[f].B, idx, cf[k])
				}
				d := b.MMA(sh, af[i], bf[j], cv)
				for k, o := range cf {
					o.m[f] = d[k]
				}
			}
		}
		return true
	}
	loop := func(c int, idx ir.Value, e *env) {
		// The envs the loop runs for: e alone, or every env of e's tile, a
		// fold carried in each env of its own class.
		t := e.tileOf()
		envs := []*env{e}
		if t != nil {
			envs = t.all
		}
		var folds []int
		for i, nd := range g.Nodes {
			if nd.Kind == NFold && nd.At == c {
				folds = append(folds, i)
			}
		}
		// A lane exchange, and what reads it, stays where it is read.
		lane := g.readsShuffle()
		vary := g.varies(c)
		if t != nil && t.mma != nil && mmaLoop(c, folds, t.mma, idx, vary) {
			return
		}
		// closed is a node reading the iterator of a loop not open here: a
		// loop nested in c's body, so the node is too.
		var closed func(n int) bool
		closed = func(n int) bool {
			if g.Nodes[n].Kind == NIter {
				_, ok := e.get(n)
				return !ok
			}
			for _, o := range g.loopOperands(n) {
				if closed(o) {
					return true
				}
			}
			return false
		}
		owns := func(x *env, f int) bool { return t == nil || x.cls == clsOf(f) }
		for _, ej := range envs {
			seen := make([]bool, len(g.Nodes))
			var hoist func(n int)
			hoist = func(n int) {
				if seen[n] {
					return
				}
				seen[n] = true
				if !vary[n] && !closed(n) && !lane[n] {
					elem(n, idx, ej)
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
				for _, o := range g.loopOperands(n) {
					hoist(o)
				}
			}
			for _, f := range folds {
				if owns(ej, f) {
					hoist(g.Nodes[f].A)
				}
			}
			if t == nil {
				for _, st := range g.Stores {
					if g.storeLoop(st) == c {
						hoist(st.Val)
						hoist(st.Ix)
					}
				}
			}
		}
		var cs []carried
		for _, f := range folds {
			for _, ej := range envs {
				if owns(ej, f) {
					cs = append(cs, carried{f: f, out: ej})
				}
			}
		}
		inits := make([]ir.Value, len(cs))
		for i, x := range cs {
			inits[i] = elem(g.Nodes[x.f].B, idx, x.out)
		}
		// Under a split, loop 0 runs its segment's trips from the segment's
		// first.
		trip, first := g.Loops[c], ir.Value(0)
		split0 := c == 0 && segTrip > 0
		if split0 {
			trip, first = segTrip, b.Mul(ir.U32, seg, b.Const(ir.U32, int64(segTrip)))
		}
		// body is a trip's envs, one a tile env, redirecting as theirs do,
		// the iterator it at.
		body := func(it ir.Value) map[*env]*env {
			ins := map[*env]*env{}
			for _, ej := range envs {
				ins[ej] = newEnv(ej)
				ins[ej].cls = ej.cls
			}
			if t != nil {
				nt := &tile{}
				for _, ej := range envs {
					x := ins[ej]
					for k, to := range ej.sh {
						if to != nil {
							x.sh[k] = ins[to]
						}
					}
					x.t = nt
					nt.all = append(nt.all, x)
				}
				if t.mma != nil {
					// The B fragments' rows are no output's: their envs
					// follow the tile's.
					for _, y := range t.mma.brows {
						x := newEnv(y)
						x.cls, x.sh[0] = y.cls, ins[y.sh[0]]
						ins[y] = x
					}
					nt.mma = t.mma.remap(ins)
				}
			}
			for i, nd := range g.Nodes {
				if nd.Kind == NIter && nd.At == c {
					for _, in := range ins {
						in.m[i] = it
					}
				}
			}
			return ins
		}
		var runtime ir.Value
		if tn := g.tripOf(c); tn >= 0 {
			runtime = elem(tn, idx, e)
		}
		// bodyStores are the stores of a trip: those its iterator places
		// (storeLoop), in the trip's env of the element's.
		bodyStores := func(ins map[*env]*env) {
			for _, st := range g.Stores {
				if g.storeLoop(st) != c {
					continue
				}
				if t != nil {
					if emitErr == nil {
						emitErr = fmt.Errorf("compose: a store in loop %d's body under a tile", c)
					}
					return
				}
				in := ins[e]
				at := elem(st.Ix, idx, in)
				b.Store(params[st.Buf], at, elem(st.Val, idx, in), 0)
			}
		}
		open++
		defer func() { open-- }()
		replicas := len(cs) == 0 && g.tripOf(c) < 0
		for _, st := range g.Stores {
			if g.storeLoop(st) == c && vary[st.Val] {
				replicas = false
			}
		}
		if (open > 1 || replicas) && trip > 0 && trip <= unrollTrips && !split0 {
			// A short loop in another's body, or one that only places
			// replicas of values made before it, runs unrolled: each trip's
			// body at its iterator, the folds' values chained from trip to
			// trip.
			cur := inits
			for j := 0; j < trip; j++ {
				ins := body(b.Const(ir.U32, int64(j)))
				nx := make([]ir.Value, len(cs))
				for i, x := range cs {
					ins[x.out].m[x.f] = cur[i]
				}
				for i, x := range cs {
					nx[i] = elem(g.Nodes[x.f].A, idx, ins[x.out])
				}
				bodyStores(ins)
				cur = nx
			}
			for i, x := range cs {
				x.out.m[x.f] = cur[i]
				x.out.setLast(x.f, cur[i])
			}
			return
		}
		zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
		if first == 0 {
			first = zero
		}
		if runtime != 0 {
			b.LoopN(runtime)
		} else {
			b.Loop(int64(trip))
		}
		it := b.Phi(ir.U32, first)
		ins := body(it)
		for i := range cs {
			x := &cs[i]
			x.in = ins[x.out]
			if g.token(x.f) {
				// An order carried from trip to trip: no value to carry.
				x.in.m[x.f] = inits[i]
				continue
			}
			x.phi = b.Phi(g.typeOf(x.f), inits[i])
			x.in.m[x.f] = x.phi
		}
		next := make([]ir.Value, len(cs))
		for i, x := range cs {
			next[i] = elem(g.Nodes[x.f].A, idx, x.in)
			x.out.setLast(x.f, next[i])
		}
		bodyStores(ins)
		for i, x := range cs {
			if x.phi != 0 {
				b.SetPhi(x.phi, next[i])
			}
		}
		b.SetPhi(it, b.Add(ir.U32, it, one))
		b.EndLoop()
		for i, x := range cs {
			if x.phi == 0 {
				x.out.m[x.f] = next[i]
				continue
			}
			x.out.m[x.f] = x.phi
		}
	}
	elem = func(n int, idx ir.Value, e *env) ir.Value {
		// Under a tile, a node is computed in the env of what it reads --
		// but where this env owns it.
		switch dc := clsOf(n); {
		case e.own != nil && e.own[n]:
		case dc == 0 && e.sh[0] != nil:
			return elem(n, idx, e.sh[0])
		case dc == 1 && e.sh[1] != nil:
			return elem(n, idx, e.sh[1])
		case dc == 2 && e.sh[2] != nil:
			return elem(n, idx, e.sh[2])
		}
		if v, ok := e.get(n); ok {
			return v
		}
		if e.readsRow() && !colDep[n] {
			if v, ok := rowEnv.m[n]; ok {
				return v
			}
		}
		// The row and column this env runs for: its tile's, or the kernel's.
		r, rowBase := r, rowBase
		er, eb, ec := e.at()
		if er != 0 {
			r, rowBase = er, eb
		}
		if ec != 0 {
			idx = ec
		}
		nd := g.Nodes[n]
		var v ir.Value
		switch nd.Kind {
		case NLoad:
			if nd.Indexed {
				if nd.Vec > 1 {
					// A vector load's constant offset is its immediate.
					ix, off := nd.Ix, int64(0)
					if x := g.Nodes[ix]; x.Kind == NUBinary && x.Op == OpAdd && g.Nodes[x.B].Kind == NUConst {
						ix, off = x.A, int64(g.Nodes[x.B].At)
					}
					at := elem(ix, idx, e)
					k := vecKey{nd.Buf, at, nd.Vec, off}
					vs, ok := vecs[k]
					if !ok {
						vs = b.LoadV(g.typeOf(n), params[nd.Buf], at, off, int(nd.Vec))
						vecs[k] = vs
					}
					v = vs[nd.Comp]
					break
				}
				at := elem(nd.Ix, idx, e)
				switch g.Bufs[nd.Buf].Kind {
				case BufU32, BufSharedU32:
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
		case NColBase:
			v = flatIdx
		case NRowBase:
			v = r
		case NColOff:
			v = b.Const(ir.U32, int64(e.tileOff(false)/cs))
		case NRowOff:
			v = b.Const(ir.U32, int64(e.tileOff(true)/rs))
		case NRowTile:
			v = rawRow
		case NColTile:
			v = rawCol
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
				if overRows && g.Nodes[nd.B].Kind == NColumn && !col[nd.A] {
					// A row's place plus the column is that place less the
					// row's base plus the flat index the column came from.
					v = b.Add(t, b.Sub(t, x, rowBase), flatRaw)
				} else {
					v = b.Add(t, x, y)
				}
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
			case OpMax:
				v = b.Max(t, x, y)
			case OpAnd:
				v = b.And(t, x, y)
			case OpXor:
				v = b.Xor(t, x, y)
			case OpShl:
				v = b.Shl(t, x, y)
			case OpShr:
				v = b.Shr(t, x, y)
			}
		case NWindow:
			if w, ok := rowWin[n]; ok {
				v = w
				break
			}
			// Through threadgroup memory: each thread's value stored at its
			// slot, then every thread folds its window's, in order.
			x := elem(nd.A, idx, e)
			if nd.At < 1 || G%nd.At != 0 || !s.Flat && e.slotOf() == 0 {
				if emitErr == nil {
					emitErr = fmt.Errorf("compose: a window of %d over a group of %d, outside a flat group or a stage", nd.At, G)
				}
				v = x
				break
			}
			wsh, ok := windowSh[n]
			if !ok {
				if emitErr == nil {
					emitErr = fmt.Errorf("compose: window node %d has no threadgroup array", n)
				}
				v = x
				break
			}
			b.Store(wsh, b.TID(), x, 0)
			b.Barrier()
			local := e.slotOf()
			if local == 0 {
				local = b.Sub(ir.U32, flatRaw, b.Mul(ir.U32, cta(), b.NTID()))
			}
			span := b.Const(ir.U32, int64(nd.At))
			wbase := b.Mul(ir.U32, b.Div(ir.U32, local, span), span)
			if nd.Red == RedMax {
				v = b.ConstF32(0)
				for i := 0; i < nd.At; i++ {
					v = b.Max(ir.F32, v, b.Load(ir.F32, wsh, wbase, int64(i)))
				}
			} else {
				v = b.Load(ir.F32, wsh, wbase, 0)
				for i := 1; i < nd.At; i++ {
					v = b.Add(ir.F32, v, b.Load(ir.F32, wsh, wbase, int64(i)))
				}
			}
		case NShuffle:
			if s.Lanes == 0 && !(s.WorkRow && G%ir.SubgroupLanes == 0 && s.Target.Lanes == ir.SubgroupLanes) && emitErr == nil {
				emitErr = fmt.Errorf("compose: a shuffle off lane rows")
			}
			v = b.ShuffleXor(g.typeOf(nd.A), elem(nd.A, idx, e), int64(nd.At))
		case NAt:
			// A at element ix: a fresh env at that column, what reads no
			// column taken from the env that holds the row's values.
			ix := elem(nd.Ix, idx, e)
			at := newEnv(nil)
			at.col = ix
			at.row, at.base = r, rowBase
			at.cls = 3
			at.sh = [3]*env{rowEnv, rowEnv, nil}
			v = elem(nd.A, ix, at)
		case NIConst:
			v = b.Const(ir.I32, int64(nd.At))
		case NSegment:
			if seg != 0 {
				v = seg
			} else {
				v = b.Const(ir.U32, 0)
			}
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
			if s.Lanes > 0 {
				v = elem(nd.A, idx, e)
				for i := 0; i < bits.Len(uint(s.Lanes))-1; i++ {
					m := s.Lanes >> (i + 1)
					if nd.At == 1 {
						m = 1 << i
					}
					o := b.ShuffleXor(ir.F32, v, int64(m))
					if nd.Red == RedMax {
						v = b.Max(ir.F32, v, o)
					} else {
						v = b.Add(ir.F32, v, o)
					}
				}
				break
			}
			v = rowVal[n]
		case NHint:
			if nd.Hint == HintAfter {
				elem(nd.B, idx, e)
			}
			v = elem(nd.A, idx, e)
			if nd.Hint == HintLast {
				w, ok := e.lastOf(nd.A)
				if !ok && emitErr == nil {
					emitErr = fmt.Errorf("compose: node %d reads fold %d's last trip, which no loop made", n, nd.A)
				}
				v = w
			}
		case NSubgroup:
			v = b.SubgroupIndex()
		case NThread:
			v = b.Add(ir.U32, b.Mul(ir.U32, cta(), b.NTID()), b.TID())
		case NLocal:
			v = b.TID()
		case NMMA:
			na, nb, _ := nd.MMA.Frags()
			args := g.Frags[nd.At]
			vs := make([]ir.Value, len(args))
			for i, a := range args {
				vs[i] = elem(a, idx, e)
			}
			out := b.MMA(nd.MMA, vs[:na], vs[na:na+nb], vs[na+nb:])
			mmaOut[out[0]] = out
			v = out[0]
		case NMMAGet:
			v = mmaOut[elem(nd.A, idx, e)][nd.At]
		case NTileLoad:
			if nd.C >= 0 {
				elem(nd.C, idx, e)
			}
			v = b.TileLoad(nd.Tile, params[nd.Buf], elem(nd.Ix, idx, e), elem(nd.B, idx, e), nd.Major)
		case NTileSplat:
			v = b.TileSplat(nd.Tile, elem(nd.A, idx, e))
		case NTileMMA:
			v = b.TileMMA(elem(nd.A, idx, e), elem(nd.B, idx, e), elem(nd.C, idx, e))
		case NWrite:
			if nd.C >= 0 {
				elem(nd.C, idx, e)
			}
			var at ir.Value
			off := int64(0)
			if x := g.Nodes[nd.Ix]; nd.Vec > 1 && x.Kind == NUBinary && x.Op == OpAdd && g.Nodes[x.B].Kind == NUConst {
				// A vector store's constant offset is its immediate.
				at, off = elem(x.A, idx, e), int64(g.Nodes[x.B].At)
			} else {
				at = elem(nd.Ix, idx, e)
			}
			switch {
			case nd.Tile.Rows > 0:
				b.TileStore(params[nd.Buf], at, elem(nd.B, idx, e), elem(nd.A, idx, e), nd.Major)
			case nd.Vec > 1:
				vs := []ir.Value{elem(nd.A, idx, e)}
				for i := 0; i < int(nd.Vec)-1; i++ {
					vs = append(vs, elem(int(nd.H[i]), idx, e))
				}
				b.StoreV(params[nd.Buf], at, off, vs...)
			default:
				b.Store(params[nd.Buf], at, elem(nd.A, idx, e), 0)
			}
			v = at
		case NBarrier:
			if nd.A >= 0 {
				elem(nd.A, idx, e)
			}
			b.Barrier()
			v = b.Const(ir.U32, 0)
		case NUnary:
			a := elem(nd.A, idx, e)
			switch nd.Op {
			case OpExp:
				v = b.Exp(a)
			case OpSqrt:
				v = b.Sqrt(a)
			case OpFloat:
				v = b.CvtF32(a)
			case OpHalf:
				v = b.CvtF16H(a)
			case OpBits:
				v = b.Bitcast(ir.F32, a)
			case OpWord:
				v = b.Bitcast(ir.U32, a)
			case OpAsWord:
				v = b.Bitcast(ir.U32, a)
			case OpInt:
				v = b.CvtI32(a)
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
			case OpSubH2:
				v = b.SubF16x2(a, elem(nd.B, idx, e))
			case OpFmaH2:
				v = b.FmaF16x2(a, elem(nd.B, idx, e), elem(nd.C, idx, e))
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
		// A thread's tile of rows by elements: every store at each, in one
		// pass, with the envs a node shared across rows or elements is
		// computed in.
		tr, tc := max(s.Tile, 1), max(s.TileCols, 1)
		var outs []*env
		if mma != nil {
			outs = mmaEnvs(b, mma, width)
		} else if tr*tc == 1 {
			outs = []*env{newEnv(nil)}
		} else {
			t := &tile{}
			mk := func(cls uint8) *env {
				x := newEnv(nil)
				x.cls = cls
				x.t = t
				t.all = append(t.all, x)
				return x
			}
			none := mk(0)
			rowAt := func(j int) ir.Value {
				if j == 0 {
					return r
				}
				return b.Add(ir.U32, r, b.Const(ir.U32, int64(j*rs)))
			}
			colAt := func(j int) ir.Value {
				if j == 0 {
					return flatIdx
				}
				return b.Add(ir.U32, flatIdx, b.Const(ir.U32, int64(j*cs)))
			}
			rowsE := make([]*env, tr)
			for j := range rowsE {
				x := mk(1)
				x.row = rowAt(j)
				x.rowOff = j * rs
				x.base = b.Mul(ir.U32, x.row, width)
				x.sh[0] = none
				rowsE[j] = x
			}
			colsE := make([]*env, tc)
			for j := range colsE {
				x := mk(2)
				x.col = colAt(j)
				x.colOff = j * cs
				x.row, x.base = r, rowBase
				x.sh[0] = none
				colsE[j] = x
			}
			for jr := 0; jr < tr; jr++ {
				for jc := 0; jc < tc; jc++ {
					x := mk(3)
					x.row, x.base, x.col = rowsE[jr].row, rowsE[jr].base, colsE[jc].col
					x.rowOff, x.colOff = jr*rs, jc*cs
					x.sh = [3]*env{none, rowsE[jr], colsE[jc]}
					outs = append(outs, x)
				}
			}
			for _, x := range rowsE {
				x.col = flatIdx
			}
		}
		// A tile's elements are stored at the tile's first place plus each
		// env's own (a constant).
		var tileAt ir.Value
		emitStore := func(st Store, e *env, base, col ir.Value) {
			var at ir.Value
			off := 0
			switch {
			case st.Indexed:
				at = elem(st.Ix, col, e)
			case e.t != nil && mma == nil:
				if tileAt == 0 {
					tileAt = b.Add(ir.U32, rowBase, flatIdx)
				}
				at, off = tileAt, e.rowOff*g.Width+e.colOff
			default:
				at = b.Add(ir.U32, base, col)
			}
			if seg != 0 && !s.GroupSplit {
				at = b.Add(ir.U32, at, b.Mul(ir.U32, seg, b.Const(ir.U32, int64(g.Rows*g.Width))))
			}
			if off != 0 {
				at = b.Add(ir.U32, at, b.Const(ir.U32, int64(off)))
			}
			b.Store(params[st.Buf], at, elem(st.Val, col, e), 0)
		}
		ndPhase, stPhase := g.phases()
		// later is which nodes a later phase's node or store reads; inLoop which
		// sit in a loop's body (a fold is its value after the loop).
		later, inLoop := make([]bool, len(g.Nodes)), make([]bool, len(g.Nodes))
		for c := range g.Loops {
			for i, v := range g.varies(c) {
				inLoop[i] = inLoop[i] || v && g.Nodes[i].Kind != NFold
			}
		}
		for i, nd := range g.Nodes {
			ops := g.loopOperands(i)
			if nd.Kind == NFold && nd.A >= 0 {
				ops = append(ops, nd.A)
			}
			for _, o := range ops {
				later[o] = later[o] || ndPhase[o] < ndPhase[i]
			}
		}
		for j, st := range g.Stores {
			later[st.Val] = later[st.Val] || ndPhase[st.Val] < stPhase[j]
			if st.Indexed {
				later[st.Ix] = later[st.Ix] || ndPhase[st.Ix] < stPhase[j]
			}
		}
		// The stores phase by phase, every env's, a barrier after a phase
		// that stored into a workgroup array. A loop whose body stores runs
		// where its first store's phase comes, unless what reads its values
		// ran it.
		last := 0
		for _, p := range stPhase {
			last = max(last, p)
		}
		ranAll := make([][]bool, len(outs))
		for i := range ranAll {
			ranAll[i] = make([]bool, len(g.Loops))
		}
		for p := 0; p <= last; p++ {
			for ei, e := range outs {
				base, col := rowBase, flatIdx
				if e.base != 0 {
					base, col = e.base, e.col
				}
				ran := ranAll[ei]
				for j, st := range g.Stores {
					if stPhase[j] != p {
						continue
					}
					c := g.storeLoop(st)
					if c < 0 {
						emitStore(st, e, base, col)
						continue
					}
					if ran[c] {
						continue
					}
					ran[c] = true
					fold := -1
					for i, nd := range g.Nodes {
						if nd.Kind == NFold && nd.At == c && fold < 0 {
							fold = i
						}
					}
					if fold < 0 {
						loop(c, col, e)
					} else {
						elem(fold, col, e)
					}
				}
				if p < last {
					// What a later phase reads of this one is made here, before
					// the barrier a later version of an array it read waits on.
					for i, nd := range g.Nodes {
						if ndPhase[i] == p && later[i] && !inLoop[i] && nd.Kind != NConst && nd.Kind != NUConst && nd.Kind != NIConst {
							elem(i, col, e)
						}
					}
				}
			}
			if p < last {
				b.Barrier()
			}
		}
		for _, e := range outs {
			col := flatIdx
			if e.base != 0 {
				col = e.col
			}
			for _, n := range g.Effects {
				elem(n, col, e)
			}
		}
		if emitErr != nil {
			return nil, emitErr
		}
		return finish(b, done)
	}
	// iterDep is which nodes read a fold loop's iterator or carried value:
	// those stay in their loop.
	iterDep := make([]bool, len(g.Nodes))
	for c := range g.Loops {
		for i, v := range g.varies(c) {
			iterDep[i] = iterDep[i] || v
		}
	}
	// hoist emits, into rowEnv and before the loop that follows, every node
	// roots reach that reads no element and no loop, of depth at most p
	// (its reductions ran in an earlier pass).
	hoist := func(roots []int, p int) {
		seen := make([]bool, len(g.Nodes))
		var walk func(n int)
		walk = func(n int) {
			if seen[n] {
				return
			}
			seen[n] = true
			nd := g.Nodes[n]
			if !colDep[n] && !iterDep[n] && depth[n] <= p && nd.Kind != NReduce && nd.Kind != NWindow {
				// Constants stay where they are read: hoisting them moves
				// nothing a target pays for.
				if nd.Kind != NConst && nd.Kind != NUConst && nd.Kind != NIConst {
					elem(n, 0, rowEnv)
					return
				}
			}
			for _, o := range g.operands(n) {
				walk(o)
			}
		}
		for _, n := range roots {
			walk(n)
		}
	}
	passes := 0
	for _, d := range depth {
		passes = max(passes, d)
	}
	for _, st := range g.Stores {
		passes = max(passes, g.storePass(st, depth))
	}
	// A butterfly needs the target's 32-lane guarantee and a group of whole
	// subgroups, at most 32 of them (the second stage's one slot a lane).
	warp := s.Warp && G%ir.SubgroupLanes == 0 && G/ir.SubgroupLanes <= ir.SubgroupLanes
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
		var roots []int
		for _, n := range reds {
			roots = append(roots, g.Nodes[n].A)
		}
		for _, st := range stores {
			roots = append(roots, st.Val)
			if st.Indexed {
				roots = append(roots, st.Ix)
			}
		}
		hoist(roots, p)
		// An argmax starts at the thread's first element and its index.
		var first ir.Value
		for j, n := range reds {
			if g.Nodes[n].Red == RedArgmax {
				if first == 0 {
					first = b.Min(ir.U32, tid, last)
				}
				fe := newEnv(nil)
				fe.rows = true
				inits[j] = elem(g.Nodes[n].A, first, fe)
			}
		}
		zero := b.ConstF32(0)
		b.Loop(iters)
		i := b.Phi(ir.U32, tid)
		accs := make([]ir.Value, len(reds))
		accI := make([]ir.Value, len(reds))
		for j, n := range reds {
			accs[j] = b.Phi(ir.F32, inits[j])
			if g.Nodes[n].Red == RedArgmax {
				accI[j] = b.Phi(ir.U32, first)
			}
		}
		idx := b.Min(ir.U32, i, last)
		in := b.Lt(ir.U32, i, width)
		e := newEnv(nil)
		e.rows = true
		for j, n := range reds {
			x := elem(g.Nodes[n].A, idx, e)
			switch g.Nodes[n].Red {
			case RedArgmax:
				// Strictly greater: an index only rises in a thread, so a tie
				// keeps the earlier.
				take := b.Lt(ir.F32, accs[j], x)
				nv, ni := b.Select(ir.F32, take, x, accs[j]), b.Select(ir.U32, take, idx, accI[j])
				b.SetPhi(accs[j], b.Select(ir.F32, in, nv, accs[j]))
				b.SetPhi(accI[j], b.Select(ir.U32, in, ni, accI[j]))
			case RedMax:
				b.SetPhi(accs[j], b.Select(ir.F32, in, b.Max(ir.F32, accs[j], x), accs[j]))
			default:
				b.SetPhi(accs[j], b.Add(ir.F32, accs[j], b.Select(ir.F32, in, x, zero)))
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
			switch g.Nodes[n].Red {
			case RedArgmax:
				rowVal[n] = argmaxTree(b, sh, shi, tid, accs[j], accI[j], G)
			case RedMax:
				rowVal[n] = groupReduce(b, sh, tid, accs[j], G, true, warp)
			default:
				rowVal[n] = groupReduce(b, sh, tid, accs[j], G, false, warp)
			}
		}
	}
	// The stages: each element space after the row's own, its loop
	// unrolled, a thread's element tid + j*G clamped to the last.
	for _, st := range g.Stages {
		var roots []int
		for _, x := range st.Stores {
			roots = append(roots, x.Val)
			if x.Indexed {
				roots = append(roots, x.Ix)
			}
		}
		hoist(roots, passes)
		lastS := b.Const(ir.U32, int64(st.Width-1))
		base := b.Mul(ir.U32, r, b.Const(ir.U32, int64(st.Width)))
		windows := false
		for _, n := range g.Nodes {
			windows = windows || n.Kind == NWindow
		}
		for n, nd := range g.Nodes {
			wsh, ok := windowSh[n]
			if nd.Kind != NWindow || nd.Red != RedMax || nd.At < st.Width || !ok || G&(G-1) != 0 {
				continue
			}
			// The row's maximum: a strided scan of its elements, then a
			// halving tree in the window's array.
			acc := b.ConstF32(0)
			for j := 0; j < (st.Width+G-1)/G; j++ {
				off := b.Const(ir.U32, int64(j*G))
				word := b.Min(ir.U32, b.Add(ir.U32, tid, off), lastS)
				se := newEnv(nil)
				se.col, se.slot, se.rows = word, b.Sub(ir.U32, word, off), true
				acc = b.Max(ir.F32, acc, elem(nd.A, word, se))
			}
			b.Store(wsh, tid, acc, 0)
			b.Barrier()
			top := b.Const(ir.U32, int64(G-1))
			for h := G / 2; h >= 1; h /= 2 {
				mine := b.Load(ir.F32, wsh, tid, 0)
				other := b.Load(ir.F32, wsh, b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(h))), top), 0)
				m := b.Max(ir.F32, mine, other)
				b.Barrier()
				b.Store(wsh, tid, m, 0)
				b.Barrier()
			}
			rowWin[n] = b.Load(ir.F32, wsh, b.Const(ir.U32, 0), 0)
			b.Barrier()
		}
		for j := 0; j < (st.Width+G-1)/G; j++ {
			off := b.Const(ir.U32, int64(j*G))
			word := b.Min(ir.U32, b.Add(ir.U32, tid, off), lastS)
			se := newEnv(nil)
			se.col, se.slot, se.rows = word, b.Sub(ir.U32, word, off), true
			for _, x := range st.Stores {
				at := b.Add(ir.U32, base, word)
				if x.Indexed {
					at = elem(x.Ix, word, se)
				}
				b.Store(params[x.Buf], at, elem(x.Val, word, se), 0)
			}
			if windows {
				// The next element's window writes must not overtake this
				// one's reads.
				b.Barrier()
			}
		}
	}
	if emitErr != nil {
		return nil, emitErr
	}
	return finish(b, done)
}

func finish(b *ir.Builder, done bool) (*ir.Kernel, error) {
	if !done {
		return nil, nil
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
	case NColumn, NRowIdx, NUConst, NIter, NSegment, NColBase, NColOff, NRowBase, NRowOff, NRowTile, NColTile,
		NWrite, NBarrier, NSubgroup, NThread, NLocal:
		return ir.U32
	case NMMA:
		return nd.MMA.Elem()
	case NMMAGet:
		return g.typeOf(nd.A)
	case NIConst, NDot4:
		return ir.I32
	case NReduce:
		if nd.Red == RedArgmax {
			return ir.U32
		}
	case NUBinary:
		return g.typeOf(nd.A)
	case NFold:
		return g.typeOf(nd.B)
	case NSelect, NAt, NShuffle, NHint:
		return g.typeOf(nd.A)
	case NLoad:
		if g.Bufs[nd.Buf].Kind == BufU32 || g.Bufs[nd.Buf].Kind == BufSharedU32 {
			return ir.U32
		}
	case NBinary:
		if nd.Op == OpPackF16 || nd.Op == OpSubH2 || nd.Op == OpFmaH2 {
			return ir.U32
		}
	case NUnary:
		switch nd.Op {
		case OpWord, OpAsWord:
			return ir.U32
		case OpInt:
			return ir.I32
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

// tileOff is e's place in its tile: its row's, or its column's.
func (e *env) tileOff(row bool) int {
	for x := e; x != nil; x = x.up {
		if row && x.rowOff != 0 {
			return x.rowOff
		}
		if !row && x.colOff != 0 {
			return x.colOff
		}
	}
	return 0
}

// strided is tile t's first place, the tile n places spaced stride apart:
// t/stride*n*stride + t%stride.
func strided(b *ir.Builder, t ir.Value, n, stride int) ir.Value {
	switch {
	case n == 1:
		return t
	case stride == 1:
		return b.Mul(ir.U32, t, b.Const(ir.U32, int64(n)))
	}
	k := b.Const(ir.U32, int64(stride))
	return b.Add(ir.U32, b.Mul(ir.U32, b.Div(ir.U32, t, k), b.Const(ir.U32, int64(n*stride))), b.Rem(ir.U32, t, k))
}
