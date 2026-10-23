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
		SplitRange{Pad: 32, Copies: 32, Work: mainWorkgroups},
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
		Prefetch{},
		Fuse{Pairs: []Pair{{From: PrimMap, To: PrimMap, AtStores: true}, {From: PrimContract, To: PrimMap, Additive: true}}},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimReduce, Unrolled: true}}},
		Threads{Group: 128},
		LiveRows{},
		WorkRow{Group: 1024, Heads: 32},
		Coarsen{Unit: 4},
		Parts{Kinds: []PrimKind{PrimMap}},
		Split{Slots: 30720, Max: 32, Block: 32, Kinds: contract},
		Blocks{Size: 32, Kinds: contract},
		Split{Slots: 30720, Max: 32, Block: 1, Group: true, Matrix: true, Kinds: []PrimKind{PrimReduce}},
		Split{Parts: 64, Group: true, Kinds: []PrimKind{PrimReduce}},
		Epilogue{},
		SplitExtreme{Parts: 1024, Kinds: []PrimKind{PrimReduce}},
		SplitExtreme{Parts: 32, Kinds: []PrimKind{PrimReduce}},
		FMA{},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimMap, Unrolled: true}}},
		Affine{},
	}}
}

// cpuProfile is main's passes on the host.
func cpuProfile() Profile {
	return Profile{Passes: []Pass{
		Quarters{Regs: []int{4, 6, 7, 8, 10, 12, 9, 15}},
		Prescale{},
		Bounds{},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimReduce, Unrolled: true}}},
		Serial{},
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
		Jam{Lanes: 8, Sums: 8, Temps: map[Op]int{OpDot4: 2, OpDot4U: 2}, Spread: []int{15, 13}, Rows: 14},
		Vectorize{Lanes: 8, Pairs: [5]int{0, 2, 3, 4, 5}, Carry: []int{9, 10, 8}, Keep: true, Quads: true, QuadRegs: []int{13, 15, 9}},
		InPlace{},
		Unroll{Vectors: 4},
		Chains{N: 4},
		Simplify{ConstLast: true},
		XorSwap{Reg: 14},
		BitMask{},
		BitSelect{Regs: []int{5, 6, 7}},
		Induction{Tail: 13},
		Consts{Shared: mainBlock()},
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
// Q32, and the paired kernels' Out2, Q2 and AScale2.
func mainArgs() ArgFields {
	return ArgFields{Out: []int{0, 120}, Weight: 8, Act: 16, Query: 112, Second: map[int]int{0: 120, 112: 128, 24: 136}, Consts: 56, In: []int{24, 136, 104, 64}, Vectors: 40, Tail: 32, Counts: []int{32, 40}}
}

// mainSampleArgs is main's convention for a sample part's words: the
// elements in Q32, their count in K, the value in Out and the index in
// ASum.
func mainSampleArgs() ArgFields {
	return ArgFields{Out: []int{0}, Index: 96, Consts: 56, In: []int{112}, Vectors: 40, Tail: 32, Counts: []int{40}}
}

// mainTableArgs is main's convention for a table part's words: the table
// in Out, its input planes in AScale, the one input word in K.
func mainTableArgs() ArgFields {
	return ArgFields{Out: []int{0}, Consts: 56, In: []int{24}, Word: 40, Vectors: 40, Tail: 32, Counts: []int{32}}
}

// mainWorkgroups is main's layout of a range's partitions over workgroups.
var mainWorkgroups = Workgroups{MinWarps: 4, MaxWarps: 8, Shared: 32 << 10, GroupDims: 96, Chunk: 2048, SlotThreads: 1280, PerSplit: 8}
