package composer

import "math"

// Workgroups is how a range's partitions are laid over workgroups: warps
// between MinWarps and MaxWarps, enough to cover a head's width a lane a
// dimension; the query heads sharing a key head grouped while Shared bytes
// of workgroup memory hold their planes and GroupDims bounds their width;
// a partition at most Chunk positions; about SlotThreads threads a slot of
// the device; a step of keys split near the square root of keys/PerSplit;
// a partition's partial its weighted sums padded to Pad, then Copies
// copies of its maximum and of its sum (SplitRange). On a device stating
// its workgroup memory (Device.Shared), the plane holds what the rest of
// the workgroup's memory leaves.
type Workgroups struct {
	MinWarps, MaxWarps int
	Shared, GroupDims  int
	Chunk, SlotThreads int
	PerSplit           int
	Pad, Copies        int
}

// stride is the words of a partition's partial a row and head of dim.
func (w Workgroups) stride(dim int) int {
	return (dim+w.Pad-1)/w.Pad*w.Pad + 2*w.Copies
}

// workGeom is one workgroup a (row, query-head group, partition): its
// warps, their slices of the head (wd of ds each) and groups of key tiles
// (wk), the query heads a workgroup serves and the positions its plane
// holds.
type workGeom struct {
	warps, wd, wk, ds, group, chunk int
}

func (w Workgroups) geometry(h Heads, d Device) workGeom {
	lanes := d.Lanes
	var g workGeom
	g.warps = min(w.MaxWarps, max(w.MinWarps, (h.Dim+lanes-1)/lanes))
	g.wd = min(g.warps, (h.Dim+lanes-1)/lanes)
	for g.warps%g.wd != 0 {
		g.wd++
	}
	g.wk, g.ds = g.warps/g.wd, (h.Dim+g.wd-1)/g.wd
	gqa := max(1, h.Query/max(1, h.Key))
	words := w.Shared / 4
	// plane is the words the plane may hold at a group of n heads: what is
	// left of the workgroup's memory beside each head's two combines over
	// the workgroup and Bind's per-trip sums (Bind), and its partial and two
	// running ones (Running) -- of the device's where it states it, else of
	// the profile's (Vulkan states none: a stage at 64 rows over one lane
	// declared 51648 bytes, past every card's static 48 KiB).
	plane := func(n int) int {
		budget := words
		if d.Shared > 0 {
			budget = d.Shared / 4
		}
		return min(words, budget-n*(2*lanes*g.warps+g.warps*h.Dim+3*w.stride(h.Dim)))
	}
	g.group = gqa
	// Each head of the group holds its warps' partial values in a plane of
	// workgroup memory, and Bind's per-trip sums a second one beside it.
	for g.group > 1 && (gqa%g.group != 0 || g.group*(2*g.warps*h.Dim+2*lanes*g.warps) > words || g.group*g.ds > w.GroupDims ||
		g.group*g.warps*h.Dim > plane(g.group)) {
		g.group--
	}
	T := lanes * g.warps
	c := min(w.Chunk, (words-2*g.group*g.warps*lanes)/(g.group*g.wd), plane(g.group)/(g.group*g.wd)) / T * T
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

// sequential reports whether a step's partitions are better one
// workgroup's, run in order: its rows' and heads' workgroups alone keep
// SlotThreads threads a slot of the device, so its partitions are there for
// the positions a workgroup's plane holds and not for the device's slots.
func (w Workgroups) sequential(g workGeom, h Heads, slots int) bool {
	return w.SlotThreads*max(1, h.Rows)*h.Query/g.group >= slots
}
