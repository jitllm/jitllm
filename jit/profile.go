package jit

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
		Threads{Group: 128},
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
	return Profile{}
}

// mainWorkgroups is main's layout of a range's partitions over workgroups.
var mainWorkgroups = Workgroups{MinWarps: 4, MaxWarps: 8, Shared: 32 << 10, GroupDims: 96, Chunk: 2048, SlotThreads: 1280, PerSplit: 8}
