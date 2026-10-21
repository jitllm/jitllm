package jit

import "github.com/jitllm/jitllm/jit/x86"

// gpuProfile is main's passes on a GPU at a step of rows rows, in order.
func gpuProfile() Profile {
	contract := []PrimKind{PrimContract}
	return Profile{Int8: &Int8Form{Window: 32, Word: 4, Sums: 16}, Passes: []Pass{
		Layout{},
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
		Threads{Group: 128},
		LiveRows{},
		WorkRow{Group: 1024},
		Coarsen{Unit: 4},
		Parts{Kinds: []PrimKind{PrimMap}},
		Split{Slots: 30720, Max: 32, Block: 32, Kinds: contract},
		Blocks{Size: 32, Kinds: contract},
		Split{Parts: 64, Kinds: []PrimKind{PrimReduce}},
		FMA{},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimMap, Unrolled: true}}},
	}}
}

// cpuProfile is main's passes on the host.
func cpuProfile() Profile {
	return Profile{Passes: []Pass{
		Serial{},
		Simplify{},
		Fuse{Pairs: []Pair{{From: PrimReduce, To: PrimMap, Then: true}}},
		RunLengths{Widths: true},
		Simplify{},
		InPlace{},
		Axpy{},
		FMA{},
		Vectorize{Lanes: 8},
		Unroll{Vectors: 4},
		Chains{N: 4},
		Simplify{},
		Consts{},
		Args{Fields: mainArgs},
		Simplify{},
		Cursors{},
		Simplify{},
		Guards{},
		FoldLoads{},
		Order{},
		Regs{Fields: map[int]int{0: int(x86.RCX), 8: int(x86.RDX), 16: int(x86.RSI), 24: int(x86.RDX), 56: int(x86.RBX),
			112: int(x86.RSI), 120: int(x86.R8), 128: int(x86.R9), 136: int(x86.R12)},
			Spare:    []int{int(x86.RSI), int(x86.R11), int(x86.R8), int(x86.R9), int(x86.R12), int(x86.R13)},
			Counters: [2]int{int(x86.R10), int(x86.R11)}, Outer: int(x86.RAX),
			Kept: []int{int(x86.R9), int(x86.RDX), int(x86.R12), int(x86.R13)}, Vectors: 16},
	}}
}

// mainArgs is main's host argument block: Out, W, A, AScale, Rows, K, Scr,
// and the paired kernels' Out2 and AScale2.
var mainArgs = ArgFields{Out: []int{0, 120}, Weight: 8, Act: 16, Consts: 56, In: []int{24, 136}, Vectors: 40, Tail: 32, Counts: []int{32, 40}}

// mainWorkgroups is main's layout of a range's partitions over workgroups.
var mainWorkgroups = Workgroups{MinWarps: 4, MaxWarps: 8, Shared: 32 << 10, GroupDims: 96, Chunk: 2048, SlotThreads: 1280, PerSplit: 8}
