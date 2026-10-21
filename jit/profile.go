package jit

// gpuProfile is main's passes on a GPU at a step of rows rows, in order.
func gpuProfile() Profile {
	contract := []PrimKind{PrimContract}
	return Profile{Passes: []Pass{
		Fuse{Pairs: []Pair{{From: PrimMap, To: PrimMap, AtStores: true}, {From: PrimContract, To: PrimMap, Additive: true}}},
		Threads{Group: 128},
		Split{Max: 32, Block: 32, Kinds: contract},
		Blocks{Size: 32, Kinds: contract},
	}}
}

// cpuProfile is main's passes on the host.
func cpuProfile() Profile {
	return Profile{}
}
