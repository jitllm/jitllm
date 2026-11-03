package graph

import "hash/fnv"

// Structure is a hash of what a graph does and none of how big it is: its ops
// in order, the roles of their weights, the enums and switches of their
// attributes and which numeric attributes are set, never a width, a count or
// a constant. Two blocks of one structure form the same kernels' shapes of
// work at different sizes.
func (g *Graph) Structure() uint64 {
	h := fnv.New64a()
	var b [1]byte
	put := func(v uint64) {
		for ; ; v >>= 7 {
			b[0] = byte(v & 0x7f)
			if v>>7 != 0 {
				b[0] |= 0x80
			}
			h.Write(b[:])
			if v>>7 == 0 {
				return
			}
		}
	}
	flag := func(on bool) {
		if on {
			put(1)
		} else {
			put(0)
		}
	}
	for i := range g.Ops {
		o := &g.Ops[i]
		put(uint64(o.Kind))
		put(uint64(len(o.In)))
		put(uint64(len(o.Out)))
		put(uint64(len(o.Reads)))
		put(uint64(len(o.Writes)))
		for _, w := range o.Weight {
			put(uint64(w.Role))
			flag(w.Sheets > 0)
		}
		a := &o.Attr
		put(uint64(a.Layout))
		put(uint64(a.Pooling))
		put(uint64(a.Chain))
		put(uint64(a.Filter))
		put(uint64(a.Fn))
		put(uint64(a.Order))
		for _, on := range []bool{a.Freqs, a.Interleaved, a.Chunked, a.Bidir, a.Positional, a.Grid, a.Tiled,
			a.Additive, a.RowWindows, a.Pooled, a.Inverse, a.Reset, a.Exact,
			a.Window != 0, a.Softcap != 0, a.Taps != 0, a.Dil > 1, a.GridW != 0, a.Stride > 1, a.Cycle != 0,
			a.Experts != 0, a.TopK != 0, a.Width != 0, a.Offset != 0, a.Local != 0,
			a.Span != 0, a.NRot != 0, a.Alpha != 0, a.Bound != 0,
			a.NHead != a.NKVHead} {
			flag(on)
		}
		put(uint64(len(a.Expr)))
		for _, t := range a.Expr {
			put(uint64(t.Fn))
			for _, arg := range t.Args {
				put(uint64(arg.Kind))
			}
		}
	}
	return h.Sum64()
}
