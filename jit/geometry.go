package jit

import "math"

// Workgroups is how a range's partitions are laid over workgroups: warps
// between MinWarps and MaxWarps, enough to cover a head's width a lane a
// dimension; the query heads sharing a key head grouped while Shared bytes
// of workgroup memory hold their planes and GroupDims bounds their width;
// a partition at most Chunk positions; about SlotThreads threads a slot of
// the device; a step of keys split near the square root of keys/PerSplit.
type Workgroups struct {
	MinWarps, MaxWarps int
	Shared, GroupDims  int
	Chunk, SlotThreads int
	PerSplit           int
}

// workGeom is one workgroup a (row, query-head group, partition): its
// warps, their slices of the head (wd of ds each) and groups of key tiles
// (wk), the query heads a workgroup serves and the positions its plane
// holds.
type workGeom struct {
	warps, wd, wk, ds, group, chunk int
}

func (w Workgroups) geometry(h Heads, lanes int) workGeom {
	var g workGeom
	g.warps = min(w.MaxWarps, max(w.MinWarps, (h.Dim+lanes-1)/lanes))
	g.wd = min(g.warps, (h.Dim+lanes-1)/lanes)
	for g.warps%g.wd != 0 {
		g.wd++
	}
	g.wk, g.ds = g.warps/g.wd, (h.Dim+g.wd-1)/g.wd
	gqa := max(1, h.Query/max(1, h.Key))
	words := w.Shared / 4
	g.group = gqa
	// Each head of the group holds its warps' partial values in a plane of
	// workgroup memory, and Bind's per-trip sums a second one beside it.
	for g.group > 1 && (gqa%g.group != 0 || g.group*(2*g.warps*h.Dim+2*lanes*g.warps) > words || g.group*g.ds > w.GroupDims) {
		g.group--
	}
	T := lanes * g.warps
	c := min(w.Chunk, (words-2*g.group*g.warps*lanes)/(g.group*g.wd)) / T * T
	g.chunk = max(c, T, (g.warps*h.Dim/g.wd+T-1)/T*T)
	return g
}

// splits is how many partitions a step of keys keys takes: about the square
// root of keys/PerSplit, at most what keeps SlotThreads threads a slot of
// the device, at least what a partition's chunk holds.
func (w Workgroups) splits(g workGeom, h Heads, keys, slots int) int {
	groups := max(1, h.Rows) * h.Query / g.group
	floor := (keys + g.chunk - 1) / g.chunk
	capped := max(1, (slots+w.SlotThreads*groups-1)/(w.SlotThreads*groups))
	sq := 1 << int(math.Round(math.Log2(math.Sqrt(max(1, float64(keys)/float64(w.PerSplit))))))
	return max(1, floor, min(sq, capped))
}
