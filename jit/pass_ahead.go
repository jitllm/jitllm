package jit

import "slices"

// Ahead asks for a weight's lines Trips trips of their loop before the trip
// that reads them, on a device whose code runs a call at a time: each load
// of a weight at a constant offset from a pointer a register or the red zone holds, whose
// loop's cursor moves it at least Min bytes a trip and whose trip's loads
// through that pointer are lines apart -- strides the hardware prefetchers
// do not follow -- is preceded by a prefetch of its address that many trips
// on, once a line. It runs after the registers are bound: a prefetch writes
// nothing and reads only the pointer its load reads at the same place.
type Ahead struct {
	Trips, Min int
	// Outer is the trips of the loop around a stream of rows (Streams) a
	// prefetch asks ahead: each of its trips is a sub-block of every row.
	// A stream whose run is Reach bytes or more is the hardware
	// prefetcher's: it runs that far ahead of a load.
	Outer, Reach int
}

// AheadLine is a sequence entry's prefetch: node Load's address Bytes on.
type AheadLine struct {
	Load  int
	Bytes int32
}

func (Ahead) Name() string { return "ahead" }

func (p Ahead) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if p.Trips <= 0 || k.Seq == nil || !k.readsWeight() {
			return nil, false, nil
		}
		hs := k.Homes()
		// A stream is a pointer's loads in one loop; its lines are the
		// distinct lines a trip reads through it.
		type stream struct{ buf, ptr, at int }
		lines := map[stream][]int{}
		ahead := map[int]int{}
		for _, e := range k.Seq {
			if e.Kind != EntryNode {
				continue
			}
			if b, ok := p.ahead(k, e.I, hs); ok {
				nd := k.Nodes[e.I]
				s := stream{nd.Buf, nd.Ptr, hs[e.I]}
				if l := k.offsetOf(e.I) >> 6; !slices.Contains(lines[s], l) {
					lines[s] = append(lines[s], l)
				}
				ahead[e.I] = b
			}
		}
		// A trip whose lines through a pointer run on from one another is a
		// sequential stream the prefetchers follow.
		strided := map[stream]bool{}
		for s, ls := range lines {
			slices.Sort(ls)
			strided[s] = true
			for i := 1; i < len(ls); i++ {
				if ls[i]-ls[i-1] <= 1 {
					strided[s] = false
				}
			}
		}
		seen := map[stream]map[int]bool{}
		var seq []Entry
		var pf []AheadLine
		for _, e := range k.Seq {
			if b, ok := ahead[e.I]; ok && e.Kind == EntryNode {
				nd := k.Nodes[e.I]
				s := stream{nd.Buf, nd.Ptr, hs[e.I]}
				if seen[s] == nil {
					seen[s] = map[int]bool{}
				}
				if l := k.offsetOf(e.I) >> 6; strided[s] && !seen[s][l] {
					seen[s][l] = true
					pf = append(pf, AheadLine{Load: e.I, Bytes: int32(b)})
					seq = append(seq, Entry{Kind: EntryAhead, I: len(pf) - 1})
				}
			}
			seq = append(seq, e)
		}
		if len(pf) == 0 {
			return nil, false, nil
		}
		m := k.clone()
		m.Seq, m.Lines = seq, pf
		return []*Kernel{m}, true, nil
	})
}

// offsetOf is load i's constant offset from its pointer, in bytes.
func (k *Kernel) offsetOf(i int) int {
	nd := k.Nodes[i]
	return int(k.Nodes[nd.A].Imm) * ElemBytes(k.Bufs[nd.Buf])
}

// ahead is the bytes past node i's address the line Trips trips on is,
// where node i is a load Ahead asks for.
func (p Ahead) ahead(k *Kernel, i int, hs []int) (int, bool) {
	nd := k.Nodes[i]
	if nd.Op != OpLoad || nd.Ax != nil || k.Bufs[nd.Buf].Operand.Kind != OperandWeight || k.Bufs[nd.Buf].Word > 0 || k.Nodes[nd.A].Op != OpConst {
		return 0, false
	}
	switch {
	case nd.Ptr > 0 && (k.Ptrs[nd.Ptr-1].Reg > 0 || k.Ptrs[nd.Ptr-1].Spill > 0), nd.Ptr == 0 && k.Bufs[nd.Buf].Reg > 0:
	default:
		return 0, false
	}
	c := hs[i]
	if c < 0 {
		return 0, false
	}
	for _, cu := range k.Loops[c].Cursors {
		if cu.Buf != nd.Buf || cu.Ptr != nd.Ptr {
			continue
		}
		step := cu.Elems * ElemBytes(k.Bufs[nd.Buf])
		trips := p.Trips
		if step < p.Min && cu.Take && cu.Ptr > 0 {
			// A copy each trip of the loop around takes afresh, moving
			// little a trip: the loop around moves the address (Streams'
			// rows within a sub-block), and the line Outer of its trips on
			// is the one to ask for.
			l := k.Loops[c]
			if p.Reach > 0 && l.Trip/max(1, l.Lanes)*step >= p.Reach {
				return 0, false
			}
			step, trips = k.outerStep(c, nd.Buf, cu.From), p.Outer
		}
		b := trips * step
		if step < p.Min || b > 1<<30 {
			return 0, false
		}
		return b, true
	}
	return 0, false
}

// outerStep is the bytes the loop around loop c moves buffer b's address a
// trip, 0 for none.
func (k *Kernel) outerStep(c, b, ptr int) int {
	o := k.Loops[c].Parent
	if o < 0 {
		return 0
	}
	for _, cu := range k.Loops[o].Cursors {
		if cu.Buf == b && cu.Ptr == ptr {
			return cu.Elems * ElemBytes(k.Bufs[b])
		}
	}
	return 0
}
