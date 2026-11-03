package composer

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/composer/ir"
)

// Kernel is one kernel's computation: an element space, sequential loops in
// it, scalar nodes, and the stores each element makes, in order.
type Kernel struct {
	Name string
	// Ones, where the CPU's binding set it, is 1 + the vector register
	// holding sixteen-bit ones for the whole kernel.
	Ones int
	// Space is the element space's extents, the last fastest. Under Work an
	// element is a workgroup of Group threads, else a thread.
	Space []int
	// Calls, where a pass dealt the space over a device's workers, is the
	// leading axes of Space that stay the calls' space (Workers).
	Calls  int
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
	// Pick is the primitive's Facts.Pick.
	Pick *Pick
	// Quarter is the primitive's Facts.Quarter, as nodes.
	Quarter   *Quarter
	Positions *Positions
	Heads     *Heads
	Part      PartKind
	// Apart is Request.Apart: the graph's inputs and outputs are buffers
	// of their own.
	Apart bool
	// Of is the model's part the kernel belongs to, by its index.
	Of int
	// Unfused is a kernel of a part the registers refused fused, with no
	// expression left to split: Fuse joins it to nothing.
	Unfused bool
	// Folds is a kernel folding a pass into the running partial
	// (foldsAPass): a pass ends at it, so Fuse joins nothing after it into
	// it and no launch it begins or joins runs past it.
	Folds bool
	// Jam is the rows a call of the kernel runs where RowJam set it;
	// Unjam is the times a part's jammed form the registers refused, each
	// halving the rows a call RowJam runs.
	Jam   int
	Unjam int
	// Unpaired is a kernel of a part whose form of a pair of heads a call
	// (HeadPairs) the registers refused, Unwide one whose form jamming a
	// loop's rows across lanes (Jam) they refused, Uncursored one whose
	// form walking its buffers' addresses (Cursors) they refused.
	Unpaired, Unwide, Uncursored bool
	// Unvector is a kernel of a part whose vector form the registers refused
	// with nothing else left to take apart: formed with no loop across lanes.
	// BitsGPR is one whose float bits a bitcast takes go to the general
	// registers, which spill: an Unvector kernel on an x86 device (Regs).
	Unvector, BitsGPR bool
	// Split is a kernel whose byte dots' runs HalfSplit cut, Unsplit one of
	// a part whose cut form the registers refused.
	Split, Unsplit bool
	// Unstreamed is a kernel of a part whose form walking a weight's planes
	// along the rows (Streams) the registers refused. Crowded is one whose
	// walk of every plane in one rows loop they refused: its walk takes the
	// planes apart (Streams.Apart) however short its rows. Aparted is a
	// walk Streams took apart.
	Unstreamed, Crowded, Aparted bool
	// Rows is 1 + the node of how many of the leading axis's rows are live;
	// Clamp, where a pass set it, 1 + the node of the elements a launch
	// holds its threads to.
	Rows, Clamp int
	// Capacity, where set, is the rows of a step the kernel's leading axis holds
	// at the form's capacity, in units of Capacity/Space[0] rows: a call runs
	// the rows its table states (tableCall), at most Capacity, and
	// the kernel's threads and its launch are held to them (CallRows).
	Capacity int
	// RowsHeld is set where a pass held the kernel to the call's rows.
	RowsHeld bool
	// Resident, where a pass set it, is how many of the kernel's
	// workgroups each of the device's units is to hold at once.
	Resident int
	// Tile, where RowTiles set it, is the rows a thread of the kernel sums,
	// its stores one a row; TileBase and TileLast are 1 + the nodes of the
	// tile's first row and of the last row the call states (0 for none), and
	// TileFirst 1 + the node its first store's row is (the first, held to
	// the last); TileRows is the leading axis's rows before tiling.
	Tile, TileBase, TileLast, TileFirst, TileRows int
	// Few, where set, is a kernel launched in place of the kernel before it
	// at a call of at most Few rows, and Many, where set, one launched only
	// at a call of more than Many rows, the kernel after it holding fewer
	// (Split.Few).
	Few, Many int
	// TileMajor is a kernel whose calls over a step's rows and a weight's
	// tiles (Workers) run every row of a tile before the next tile.
	TileMajor bool
	// Matrix is a matrix kernel's shape: its outputs and its reduction's trip.
	Matrix [2]int
	// Share, where set, is the rows a kernel's calls (Workers) hold, a
	// call's first row its coordinate's ShareUnit rows on and its rows a
	// length the caller sizes to the workers it runs, a multiple of them.
	Share, ShareUnit int
	// Running, where SplitRange set it, is a split whose partitions one
	// workgroup runs in order (Running, the pass).
	Running *RunningSplit
	// Own is the operands each point of the generated space writes at its
	// own place alone: a call over points of it writes its own elements.
	Own []Operand
	// Reach, where Reaches set it, is the operands each point reaches only
	// its own leading rows of.
	Reach []Reach
	// Seq, where a schedule set it, is the kernel as a sequence: each node,
	// store and loop in the order the code runs them.
	Seq []Entry
	// Args, where a binding to the call's arguments set them, are the words
	// a call passes and what the caller computes each from.
	Args []Arg
	// Ptrs are the copies of buffers' addresses cursors advance.
	Ptrs []Ptr
	// Lines are the prefetches the sequence's EntryAhead entries name
	// (Ahead).
	Lines []AheadLine
	// vwords, while a register binding runs, keeps its VectorWords.
	vwords *vectorMemo
}

// Arg is the call's argument word at byte offset Field: the address of
// buffer Buf, offset by the buffer's Offset, or where Buf is -1 the value
// of node Node-1, a count of the call computed by the caller. Coord, where
// set, is 1 + the axis of the calls' space whose coordinate the word is.
type Arg struct {
	Field, Buf, Node int
	Coord            int
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
	// EntryAhead is a prefetch of line I of the kernel's Lines.
	EntryAhead
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
	// Uniform, where a pass set it, is a loop whose count is one value for
	// every thread of a workgroup: a barrier in it is met by all of them.
	Uniform bool
	// StepsLast makes its folds' steps after its stores in each trip: a
	// trip that stages workgroup memory and reads it behind a barrier.
	StepsLast bool
	// Of, where a split set it, is 1 + the loop this one takes trips of.
	Of int
}

// Cursor is pointer Ptr of buffer Buf advanced Elems elements a trip: 0 the
// buffer's address itself, else 1 + the kernel's copy, taken from pointer
// From as the loop starts where Take is set.
type Cursor struct {
	Buf, Elems int
	Ptr, From  int
	Take       bool
}

// Ptr is a copy of buffer Buf's address a loop advances in its place, in
// register Reg (1 + the register, where a binding set it).
type Ptr struct {
	Buf, Reg int
	// Spill, where set, is 1 + the red-zone slot the copy lives in.
	Spill int
	// Off, where set, is 1 + a register of offsets the copies a loop takes
	// together share: the copy is pointer From's address plus it times Scale.
	Off, From, Scale int
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
	// Extent, where Shape is not set, is how many elements a flat index
	// reaches.
	Extent int
	// Plane is workgroup memory a staging may fold into its plane.
	Plane bool
	// Base, where a binding to the call's arguments set it, is 1 + the node
	// that is the buffer's address, and Offset 1 + the node of the elements
	// the caller adds to it for each call; loads and stores index from it.
	Base, Offset int
	// Reg, where a register binding set it, is 1 + the register holding the
	// address.
	Reg int
	// Reload, where a register binding set it, says the buffer's address is
	// loaded from its argument word at each access rather than held.
	Reload bool
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
	// model; with A -1, a call's rows of a Share, Imm at Device.Slots.
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
	// OpDot4 is C plus the sum of the products of A's and B's signed bytes,
	// lane by lane of the four in each word. B is an int8 form's codes,
	// rounded within ±127: never -128.
	OpDot4
	// OpWidenHalf is word A's low binary16 half as a float.
	OpWidenHalf
	// OpDot4U is OpDot4 with A's bytes unsigned.
	OpDot4U
	// OpTileLoad is the collective tile Tile of workgroup buffer Buf at
	// element offset A, row stride B, column-major where Imm's bit 0 is set
	// and over a view stated aligned (ir.TileLoadAligned) where bit 1 is, to
	// eight components (ir.TileLoadAligned8) where bit 2 is as well; At
	// is 1 + the loop whose trip it reads in. OpTileSplat is a tile every element
	// of which is A; OpTileMMA is A times B plus C, over tiles. A tile is
	// read only by a tile op, a tile store or a tile fold.
	OpTileLoad
	OpTileSplat
	OpTileMMA
	// OpSubF16x2 is A less B and OpFmaF16x2 A times B plus C, each word two
	// binary16 halves (ir.OpSubF16x2, ir.OpFmaF16x2), formed only for a
	// device that states them (Device.HalfPairs).
	OpSubF16x2
	OpFmaF16x2
	// OpMMA is the device's own matrix instruction (Device.Fragment) on a
	// lane's fragments, its operands Ax: A's words, B's words, then C's
	// floats; OpMMAGet is its result's float Imm.
	OpMMA
	OpMMAGet
	// OpTileScale is tile A with each row r times the float of workgroup
	// buffer Buf at element B+r.
	OpTileScale
	// OpPackHalves is A's and B's low binary16 halves in one word, A's
	// low, or with Imm 1 their high halves (ir.OpPackHalves).
	OpPackHalves
	numOps
)

// byteDot reports whether op is a dot of four bytes a word.
func (op Op) byteDot() bool { return op == OpDot4 || op == OpDot4U }

// byteDots reports whether the kernel holds a byte dot.
func (k *Kernel) byteDots() bool {
	for _, nd := range k.Nodes {
		if nd.Op.byteDot() {
			return true
		}
	}
	return false
}

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
	// reads from (see Cursor). Tmp and Tmp2 are 1 + registers the value's
	// code needs besides its own.
	Reg, Ptr, Tmp, Tmp2 int
	// Spill, where the CPU's register binding set it, is 1 + the red-zone
	// slot a value lives in: stored where it is made, loaded at every read;
	// -1 for an argument word read from the argument block at every read.
	Spill int
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
	// Tile, where set, is the collective tile the value is.
	Tile *ir.TileType
	// Running marks a fold its loop's trips read as it runs -- an
	// iteration carried as a sum (Induction) -- rather than once the loop
	// is done: its readers are the loop's.
	Running bool
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
	// Tile stores the collective tile Val at element offset Ix, row stride
	// node Stride, column-major where Col is set.
	Tile   bool
	Stride int
	Col    bool
	// Step, where more than one, is the elements between a jammed store's
	// lanes, each stored alone.
	Step int
	// Touch writes nothing: it asks the cache for the word at Ix of Buf,
	// which the kernel reads later.
	Touch bool
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
	c.Lines = append([]AheadLine(nil), n.Lines...)
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

func (n *Kernel) Elements() int {
	e := 1
	for _, s := range n.Space {
		e *= s
	}
	return e
}

func (o Op) String() string {
	switch o {
	case OpConst:
		return "const"
	case OpCoord:
		return "coord"
	case OpIter:
		return "iter"
	case OpFold:
		return "fold"
	case OpLoad:
		return "load"
	case OpAdd:
		return "add"
	case OpSub:
		return "sub"
	case OpMul:
		return "mul"
	case OpDiv:
		return "div"
	case OpRem:
		return "rem"
	case OpMin:
		return "min"
	case OpMax:
		return "max"
	case OpFma:
		return "fma"
	case OpBitcast:
		return "bitcast"
	case OpLess:
		return "less"
	case OpSelect:
		return "select"
	case OpSqrt:
		return "sqrt"
	case OpExp:
		return "exp"
	case OpShl:
		return "shl"
	case OpShr:
		return "shr"
	case OpAnd:
		return "and"
	case OpXor:
		return "xor"
	case OpCvtF32:
		return "cvtf32"
	case OpCvtI32:
		return "cvti32"
	case OpThread:
		return "thread"
	case OpGroupSum:
		return "groupsum"
	case OpGroupMax:
		return "groupmax"
	case OpExchange:
		return "exchange"
	case OpLocal:
		return "local"
	case OpFlat:
		return "flat"
	case OpShuffle:
		return "shuffle"
	case OpParam:
		return "param"
	case OpBroadcast:
		return "broadcast"
	case OpLength:
		return "length"
	case OpHSum:
		return "hsum"
	case OpHalf:
		return "half"
	case OpGtW:
		return "gtw"
	case OpAndNot:
		return "andnot"
	case OpZip:
		return "zip"
	case OpUnpack:
		return "unpack"
	case OpUpper:
		return "upper"
	case OpRoundI32:
		return "round"
	case OpSpan:
		return "span"
	case OpHMax:
		return "hmax"
	case OpHMin:
		return "hmin"
	case OpEq:
		return "eq"
	case OpOr:
		return "or"
	case OpLanes:
		return "lanes"
	case OpSar:
		return "sar"
	case OpDupEven:
		return "dupeven"
	case OpDupOdd:
		return "dupodd"
	case OpSwapPairs:
		return "swappairs"
	case OpAddSub:
		return "addsub"
	case OpFirstLane:
		return "firstlane"
	case OpCopy:
		return "copy"
	case OpDot4:
		return "dot4"
	case OpWidenHalf:
		return "widenhalf"
	case OpDot4U:
		return "dot4u"
	case OpTileLoad:
		return "tileload"
	case OpTileSplat:
		return "tilesplat"
	case OpTileMMA:
		return "tilemma"
	case OpSubF16x2:
		return "subf16x2"
	case OpFmaF16x2:
		return "fmaf16x2"
	case OpMMA:
		return "mma"
	case OpMMAGet:
		return "mmaget"
	case OpTileScale:
		return "tilescale"
	case OpPackHalves:
		return "packhalves"
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
		return k.Elements()
	}
	if k.Group <= 0 {
		return 0
	}
	return (k.Elements() + k.Group - 1) / k.Group
}

// once reports whether a loop runs one fixed trip: its body, no count.
func (l Loop) once() bool { return l.Count == 0 && l.Trip == 1 && !l.Kept }
