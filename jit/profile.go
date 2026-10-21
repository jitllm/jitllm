package jit

// gpuProfile is main's passes on a GPU at a step of rows rows, in order.
func gpuProfile() Profile {
	contract := []PrimKind{PrimContract}
	return Profile{Int8: &Int8Form{Window: 32, Word: 4, Sums: 16}, Passes: []Pass{
		Fuse{Pairs: []Pair{{From: PrimMap, To: PrimMap, AtStores: true}, {From: PrimContract, To: PrimMap, Additive: true}}},
		Threads{Group: 128},
		WorkRow{Group: 1024},
		Coarsen{Unit: 4},
		Split{Slots: 30720, Max: 32, Block: 32, Kinds: contract},
		Blocks{Size: 32, Kinds: contract},
	}}
}

// cpuProfile is main's passes on the host.
func cpuProfile() Profile {
	return Profile{}
}
