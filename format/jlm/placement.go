package jlm

// Placement is how one File's weight memory is placed. Each loaded model sets
// its own, so two models in one process can be placed differently. The zero
// value asks for transparent huge pages and leaves the node to first touch.
type Placement struct {
	// NoHugePages leaves the frames on the kernel's default page size.
	NoHugePages bool
	// Nodes spreads the memory page by page over these NUMA nodes. Fewer than
	// two leaves placement to first touch.
	Nodes []int
}

// SetPlacement applies p to the dense region now and to every page frame
// allocated after it. Call it before the first page is read (model.Open does);
// frames already in the free list keep the placement they were given.
func (f *Frames) SetPlacement(p Placement) {
	var mask []uint64
	if len(p.Nodes) >= 2 {
		hi := 0
		for _, n := range p.Nodes {
			hi = max(hi, n)
		}
		mask = make([]uint64, 1+hi/64)
		for _, n := range p.Nodes {
			if n >= 0 {
				mask[n/64] |= 1 << uint(n%64)
			}
		}
	}
	f.mu.Lock()
	f.noHuge, f.nodeMask = p.NoHugePages, mask
	f.nodes = nil
	if mask != nil {
		for _, n := range p.Nodes {
			if n >= 0 {
				m := make([]uint64, 1+n/64)
				m[n/64] = 1 << uint(n%64)
				f.nodes = append(f.nodes, m)
			}
		}
	}
	f.mu.Unlock()
	placeMemory(f.dense, p.NoHugePages, mask)
}

// Interleaved reports whether this File spreads its memory over nodes.
func (f *Frames) Interleaved() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.nodeMask != nil
}
