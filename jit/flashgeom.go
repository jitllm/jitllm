package jit

import "math"

// flashGeom is one workgroup a (row, query-head group, key split): its warps,
// their slices of the head (wd of ds each) and groups of key tiles (wk), the
// query heads a workgroup serves and the keys its score plane holds.
type flashGeom struct {
	warps, wd, wk, ds, group, chunk int
}

func flashGeometry(h Heads) flashGeom {
	var g flashGeom
	g.warps = min(8, max(4, (h.Dim+31)/32))
	g.wd = min(g.warps, (h.Dim+31)/32)
	for g.warps%g.wd != 0 {
		g.wd++
	}
	g.wk, g.ds = g.warps/g.wd, (h.Dim+g.wd-1)/g.wd
	gqa := max(1, h.Query/max(1, h.Key))
	g.group = gqa
	for g.group > 1 && (gqa%g.group != 0 || g.group*(g.warps*h.Dim+64*g.warps) > (32<<10)/4 || g.group*g.ds > 96) {
		g.group--
	}
	T := 32 * g.warps
	c := min(2048, ((32<<10)/4-2*g.group*g.warps*32)/(g.group*g.wd)) / T * T
	g.chunk = max(c, T, (g.warps*h.Dim/g.wd+T-1)/T*T)
	return g
}

// splits is how many key partitions a step of keys keys takes: about the
// square root of keys/8, at most what keeps about 1280 threads a slot of the
// device, at least what the chunk holds.
func (g flashGeom) splits(h Heads, keys, slots int) int {
	if slots <= 0 {
		slots = 16384
	}
	groups := max(1, h.Rows) * h.Query / g.group
	floor := (keys + g.chunk - 1) / g.chunk
	capped := max(1, (slots+1280*groups-1)/(1280*groups))
	sq := 1 << int(math.Round(math.Log2(math.Sqrt(max(1, float64(keys)/8)))))
	return max(1, floor, min(sq, capped))
}
