package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/x86"
)

// gpuProfile is main's passes on a GPU at a step of rows rows, in order.
func gpuProfile() Profile {
	contract := []PrimKind{PrimContract}
	return Profile{Int8: &Int8Form{Window: 32, Word: 4, Sums: 16}, Passes: []Pass{
		Offsets{Table: tableQOff},
		RunLengths{Grain: 128, Align: 32},
		Flash{Rows: 128, Keys: 32, Shared: 48 << 10, Half: true},
		SplitRange{Work: mainWorkgroups},
		WorkFuse{Work: mainWorkgroups},
		HeadGroups{Work: mainWorkgroups},
		Forward{},
		DeadStores{},
		LoopFuse{},
		Bind{},
		Stage{},
		Unroll{Full: 32, Factor: 4, Over: BoundWarps},
		Chains{N: 4},
		Trips{},
		Mask{},
		Runs{},
		// Before Prefetch: a table read carried from trip to trip is a
		// second fold of the reduction's loop, and a reduction of more than
		// its sum's fold joins nothing.
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimReduce, Unrolled: true}}},
		Fuse{Pairs: []Pair{{From: PrimMap, To: PrimMap, AtStores: true}, {From: PrimContract, To: PrimMap, Additive: true}}},
		Matrix{Rows: 128, Cols: 64, Depth: 32, Warps: [2]int{2, 2}, Resident: 2, Tiles: []MatrixTile{
			{Rows: 128, Cols: 128, Depth: 32, Warps: [2]int{4, 2}, Resident: 1},
			{Rows: 64, Cols: 64, Depth: 32, Warps: [2]int{2, 2}, Resident: 4},
			{Rows: 128, Cols: 32, Depth: 32, Warps: [2]int{4, 1}, Resident: 2},
			{Rows: 128, Cols: 128, Depth: 32, Warps: [2]int{2, 2}, Resident: 2},
		}, Latency: 12000, Decode: 16, Launch: 3 << 20, Plane: 384},
		FloatTiles{Depth: 16, Tiles: []FloatTile{{64, 64, [2]int{4, 4}}, {32, 64, [2]int{4, 4}}, {32, 32, [2]int{4, 4}}, {16, 32, [2]int{2, 4}}, {16, 16, [2]int{2, 2}}}},
		ContractPairs{},
		// Before Coarsen and Parts: a gather the matrix unit's halves read
		// runs in them rather than through a buffer of its own.
		Recompute{MapsOnly: true},
		Prefetch{Except: []PrimKind{PrimMatrix}},
		RowTiles{Rows: 16},
		Threads{Group: 128},
		LiveRows{},
		WorkRow{Group: 1024, Heads: 32},
		Coarsen{Unit: 4},
		DeadStores{},
		Parts{Kinds: []PrimKind{PrimMap}},
		Split{Slots: 30720, Max: 32, Block: 32, Group: true, Matrix: true, Few: 4, Kinds: contract},
		Split{Slots: 30720, Max: 32, Block: 32, Few: 4, Kinds: contract},
		Blocks{Size: 32, Kinds: contract},
		Split{Slots: 30720, Fill: 0.75, Max: 32, Block: 1, Group: true, Matrix: true, Few: 4, Kinds: []PrimKind{PrimReduce}},
		Split{Parts: 64, Group: true, Kinds: []PrimKind{PrimReduce}},
		Epilogue{},
		Recompute{},
		Same{},
		SplitExtreme{Parts: 1024, Kinds: []PrimKind{PrimReduce}},
		GroupExtreme{Trips: 4, Kinds: []PrimKind{PrimReduce}},
		SplitExtreme{Parts: 32, Kinds: []PrimKind{PrimReduce}},
		FMA{},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimMap, Unrolled: true}}},
		Siblings{},
		Prologue{Words: 12 << 10, Lead: 8, Trips: 8},
		CallRows{},
		Affine{Weights: []PrimKind{PrimMatrix, PrimReduce}},
		Prefetch{Staged: []PrimKind{PrimMatrix}},
		Running{},
		HalfPairs{Kinds: []PrimKind{PrimMatrix}},
		Phases{Params: 64},
	}}
}

// cpuProfile is main's passes on the host, and on a host without AVX2,
// FMA and F16C its narrow passes.
func cpuProfile() Profile {
	p := cpuWideProfile()
	p.Narrow, p.Wide = cpuNarrow(p.Passes), uint64(x86.ISATierAVX2)
	return p
}

// cpuNarrow is passes four floats a vector: no fused multiply-add to make,
// so a product keeps a register of its own, and a history narrowed in
// software, which keeps no NaN's payload. A State holding an f32 cache, as
// main's SSE tier ships, forms without Halves (Registry.WithoutHalves).
func cpuNarrow(passes []Pass) []Pass {
	var n []Pass
	for _, ps := range passes {
		switch q := ps.(type) {
		case FMA:
			continue
		case Jam:
			q.Lanes = 4
			ps = q
		case Vectorize:
			q.Lanes, q.Quads = 4, false
			ps = q
		case Halves:
			q.KeepsPayload = false
			ps = q
		}
		n = append(n, ps)
	}
	return n
}

func cpuWideProfile() Profile {
	return Profile{Int8: &Int8Form{Window: 32, Word: 4, Sums: 16, UnsignedDot: uint64(x86.ISAVNNI), Slot: 16, SumCodes: true}, ScanLoops: true, Passes: []Pass{
		RowCalls{},
		Quarters{Regs: []int{4, 6, 7, 8, 10, 12, 9, 15}},
		Prescale{},
		Bounds{},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimReduce, Unrolled: true}}},
		Workers{Rows: 16, Tile: 64},
		RowJam{N: 4},
		Serial{},
		Unsigned{},
		Simplify{ConstLast: true},
		RunLengths{},
		Simplify{ConstLast: true},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimMap, Then: true}, {From: PrimMap, To: PrimReduce, Alongside: true, Combine: OpAdd}}},
		RunLengths{Widths: true},
		Simplify{ConstLast: true},
		Spans{Page: 64},
		Simplify{ConstLast: true},
		HeadPairs{N: 2},
		InPlace{},
		Halves{KeepsPayload: true},
		Axpy{},
		Exp{Scratch: [3]int{0, 1, 2}},
		FMA{},
		FirstTerm{},
		HalfWords{},
		// An Intel core's L1 has 10 line fill buffers (Broadwell, Skylake).
		Streams{Dots: 4, Most: 8, Apart: 1, Fill: 10, Line: 64},
		Jam{Lanes: 8, Sums: 8, Temps: map[Op]int{OpDot4: 2, OpDot4U: 2}, Spread: []int{15, 13}, Rows: 14, Positions: 4},
		Periods{Most: 8},
		Vectorize{Lanes: 8, Pairs: [5]int{0, 2, 3, 4, 5}, Carry: []int{9, 10, 8}, Keep: true, Quads: true, QuadRegs: []int{13, 15, 9}},
		StreamWords{Least: 8},
		InPlace{},
		Unroll{Vectors: 4},
		Chains{N: 4},
		Simplify{ConstLast: true},
		XorSwap{Reg: 14},
		BitMask{},
		BitSelect{Regs: []int{5, 6, 7}},
		Induction{Tail: 13},
		Splats{},
		Consts{Shared: mainBlock()},
		Reaches{},
		Args{Fields: mainArgs(), Parts: map[PartKind]ArgFields{PartSample: mainSampleArgs(), PartTable: mainTableArgs()}},
		Simplify{ConstLast: true},
		Cursors{},
		Simplify{ConstLast: true},
		Guards{},
		FoldLoads{},
		ShiftChain{Regs: []int{8, 9, 0, 1, 2, 3}},
		Horner{Acc: []int{11, 14}, Coef: 13, Var: 10},
		Order{},
		ConstRuns{},
		ConstSiblings{},
		CursorOrder{},
		ZerosPerLoop{},
		CopiesAside{},
		FoldsAside{},
		LateWords{},
		Regs{Fields: map[int][]int{0: {int(x86.RCX)}, 8: {int(x86.RDX)}, 16: {int(x86.RSI)}, 24: {int(x86.RDX), int(x86.R8)},
			56: {int(x86.RBX)}, 112: {int(x86.RSI)}, 120: {int(x86.R8), int(x86.R10)}, 128: {int(x86.R9)}, 136: {int(x86.R12)}},
			// The registers a value takes once its role's are taken: main's six,
			// then every general register no role holds -- not RSP, RDI (the
			// argument block), RBP, R14, the counters R10 and RAX, which are
			// bound after the values, nor R15, which a reloaded address takes
			// (reloadReg).
			Spare: []int{int(x86.RSI), int(x86.R11), int(x86.R8), int(x86.R9), int(x86.R12), int(x86.R13),
				int(x86.RBX), int(x86.RCX), int(x86.RDX)},
			Counters: [2]int{int(x86.R10), int(x86.R11)}, Outer: int(x86.RAX),
			Kept: []int{int(x86.R9), int(x86.RDX), int(x86.R12), int(x86.R13)}, Vectors: 16, Scratch: int(x86.RAX),
			Reloads:  [2]int{int(x86.R13), int(x86.RBX)},
			Resident: map[int]int{0: 9, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 8, 9: 7},
			Walks:    []int{int(x86.R11), int(x86.R8)}, Lanes: []int{1, 4}, Hoist: 8, Scalars: 6, Keep: 7},
		// Intel's L2 streamer runs up to 20 lines ahead of a load.
		Ahead{Trips: 2, Min: 2048, Outer: 1, Reach: 20 * 64},
	}}
}

// mainBlock is main's host constant block, its words 0 to 10.
func mainBlock() []uint32 {
	var out []uint32
	for _, v := range []float32{float32(math.Log2E), -0.693359375, 2.12194440e-4, 1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1, -87.3} {
		out = append(out, math.Float32bits(v))
	}
	return append(out, 127, math.Float32bits(88))
}

// mainArgs is main's host argument block: Out, W, A, AScale, Rows, K, Scr,
// Q32, and the paired kernels' Out2, Q2 and AScale2; a call's coordinates
// in Cols and OutStr.
func mainArgs() ArgFields {
	return ArgFields{Out: []int{0, 120}, Weight: 8, Act: 16, Query: 112, Second: map[int]int{0: 120, 112: 128, 24: 136}, Consts: 56, In: []int{24, 136, 104, 64, 48, 72, 144, 152, 160}, Vectors: 40, Tail: 32, Counts: []int{32, 40},
		Coords: []int{80, 88}, Share: 88}
}

// mainSampleArgs is main's convention for a sample part's words: the
// elements in Q32, their count in K, the value in Out and the index in
// ASum.
func mainSampleArgs() ArgFields {
	return ArgFields{Out: []int{0}, Index: 96, Consts: 56, In: []int{112}, Vectors: 40, Tail: 32, Counts: []int{40}}
}

// mainTableArgs is main's convention for a table part's words: the table
// in Out, its input planes in AScale, the one input word in K. A table over
// several position axes reads more buffers, in fields the registers hold
// apart from AScale's: a further input in Q2, and its positions, read at an
// index a load gives, in AScale2.
func mainTableArgs() ArgFields {
	return ArgFields{Out: []int{0}, Weight: 136, Consts: 56, In: []int{24, 128}, Word: 40, Vectors: 40, Tail: 32, Counts: []int{32}}
}

// mainWorkgroups is main's layout of a range's partitions over workgroups.
var mainWorkgroups = Workgroups{MinWarps: 4, MaxWarps: 8, Shared: 32 << 10, GroupDims: 96, Chunk: 2048, SlotThreads: 1280, PerSplit: 8, Pad: 32, Copies: 32}
