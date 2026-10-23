package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// SplitRange splits a normalized exponential reduction over a range of
// positions -- a maximum over the range, the terms' exp less it, their sum
// and their weighted sum, the latter divided by the former, and what
// computes the terms -- into partitions of each row's live span: the
// range's axis or loop becomes a partition and a position in it, the
// partition's share of the live span rounded up to the range's alignment,
// at most a workgroup's chunk; the live end the partition's. The partitions'
// maxima, sums and weighted sums go into a partial, the weighted sums padded
// to Pad then Copies copies of the maximum and of the sum; the division
// becomes the split's combine, each partition weighted by exp of its
// maximum less the greatest.
type SplitRange struct {
	Pad, Copies int
	Work        Workgroups
}

func (SplitRange) Name() string { return "splitrange" }

func (s SplitRange) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	var out []*Kernel
	for i := 0; i < len(ks); i++ {
		if used, ns, ok := s.split(ks[i:], d); ok {
			for _, n := range ns {
				n.Passes = append(n.Passes, s.Name())
			}
			out = append(out, ns...)
			i += used - 1
			continue
		}
		out = append(out, ks[i])
	}
	return out, nil
}

// member is a kernel of the split and what the split does with it.
type member struct {
	k          *Kernel
	axis, loop int
	out        Operand
}

func (s SplitRange) split(ch []*Kernel, d Device) (int, []*Kernel, bool) {
	var mem []member
	for _, k := range ch {
		p := k.Positions
		if p == nil || k.Heads == nil || len(k.Stores) != 1 || p.Align <= 1 {
			break
		}
		mem = append(mem, member{k: k, axis: p.Axis, loop: p.Loop, out: k.Bufs[k.Stores[0].Buf].Operand})
	}
	if len(mem) == 0 || len(mem) == len(ch) {
		return 0, nil, false
	}
	last := ch[len(mem)]
	if last.Kind != PrimMap || len(last.Loops) != 0 || len(last.Stores) != 1 {
		return 0, nil, false
	}
	var mx, wt, sm, vl *member
	for i := range mem {
		m := &mem[i]
		if f, ok := m.k.fold(); ok && m.loop >= 0 && m.k.Nodes[m.k.Nodes[f].A].Op == OpMax {
			mx = m
		}
	}
	if mx == nil {
		return 0, nil, false
	}
	for i := range mem {
		m := &mem[i]
		f, isFold := m.k.fold()
		switch {
		case m.loop < 0 && m.k.reads(mx.out) && hasNode(m.k, OpExp):
			wt = m
		case wt != nil && isFold && m.loop >= 0 && m.k.reads(wt.out) && last.reads(m.out) && m.k.sumFold(f):
			if len(m.k.Space) == len(last.Space) {
				vl = m
			} else {
				sm = m
			}
		}
	}
	if wt == nil || sm == nil || vl == nil {
		return 0, nil, false
	}
	h := *mx.k.Heads
	C := mx.k.Loops[mx.loop].Trip
	g := s.Work.geometry(h, d.Lanes)
	S := s.Work.splits(g, h, C, d.Slots)
	if S < 2 {
		return 0, nil, false
	}
	J := g.chunk
	D := h.Dim
	stride := (D+s.Pad-1)/s.Pad*s.Pad + 2*s.Copies
	no := last.Bufs[last.Stores[0].Buf].Operand
	part := Operand{Kind: OperandScratch, Name: no.Name + ".part", Of: no.Of}
	split := map[Operand]*member{}
	for i := range mem {
		split[mem[i].out] = &mem[i]
	}
	sp := splitter{S: S, J: J, split: split}
	var ns []*Kernel
	for i := range mem {
		m := &mem[i]
		k, ok := sp.partition(m)
		if !ok {
			return 0, nil, false
		}
		if m == vl {
			st := &k.Stores[0]
			coords := k.coordNodes()
			rh := k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, coords[0], k.u32(h.Query)), coords[1])
			ix := k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, rh, k.u32(S)), coords[3]),
				k.u32(stride)), coords[2])
			st.Ax, st.Ix = nil, ix
			st.Buf = k.buf(Buf{Kind: BufOut, Elem: ir.F32, Operand: part, Extent: h.Rows * h.Query * S * stride})
		}
		ns = append(ns, k)
	}
	cp, mg := s.copies(h, S, stride, part, mx.out, sm.out), s.merge(last, h, S, stride, part, vl.out, sm.out)
	cp.Part, cp.Of, mg.Of = last.Part, last.Of, last.Of
	ns = append(ns, cp, mg)
	return len(mem) + 1, ns, true
}

func hasNode(k *Kernel, op Op) bool {
	for _, nd := range k.Nodes {
		if nd.Op == op {
			return true
		}
	}
	return false
}

// coordNodes is a node for each of k's coordinates.
func (k *Kernel) coordNodes() []int {
	c := make([]int, len(k.Space))
	for i := range c {
		c[i] = k.add(Node{Op: OpCoord, A: i, Type: ir.U32})
	}
	return c
}

type splitter struct {
	S, J  int
	split map[Operand]*member
}

// partition is member m over the partitions: the range's coordinate a
// partition and a position in it, a read of a member's output at the
// position read at the partition's, the live end the partition's.
func (sp splitter) partition(m *member) (*Kernel, bool) {
	U := ir.U32
	k := m.k.clone()
	pos := *m.k.Positions
	k.Positions = &pos
	var key []int
	var part, j int
	if m.axis >= 0 {
		a := m.axis
		k.Space = append(append(append([]int(nil), k.Space[:a]...), sp.S, sp.J), k.Space[a+1:]...)
		for i, nd := range k.Nodes {
			if nd.Op == OpCoord && nd.A == a {
				key = append(key, i)
			}
			if nd.Op == OpCoord && nd.A > a {
				k.Nodes[i].A++
			}
		}
		part = k.add(Node{Op: OpCoord, A: a, Type: U})
		j = k.add(Node{Op: OpCoord, A: a + 1, Type: U})
		pos.Axis = a + 1
	} else {
		k.Space = append(append([]int(nil), k.Space...), sp.S)
		part = k.add(Node{Op: OpCoord, A: len(k.Space) - 1, Type: U})
		k.Loops[m.loop].Trip = sp.J
		it := k.iterOf(m.loop)
		key = []int{it}
		j = k.add(Node{Op: OpIter, At: m.loop, Type: U})
	}
	isKey := func(x int) bool {
		for _, y := range key {
			if x == y {
				return true
			}
		}
		return false
	}
	// The chain's own reads and stores follow the partition.
	for i, nd := range k.Nodes {
		if nd.Op != OpLoad || nd.Ax == nil {
			continue
		}
		q, ok := sp.split[k.Bufs[nd.Buf].Operand]
		if !ok {
			continue
		}
		ax, ok := sp.relayout(q, nd.Ax, isKey, part, j)
		if !ok {
			return nil, false
		}
		k.Nodes[i].Ax = ax
	}
	for b := range k.Bufs {
		if q, ok := sp.split[k.Bufs[b].Operand]; ok {
			k.Bufs[b].Shape = sp.shape(q, k.Bufs[b].Shape)
		}
	}
	if st := &k.Stores[0]; st.Ax != nil {
		ax, ok := sp.relayout(m, st.Ax, isKey, part, j)
		if !ok {
			return nil, false
		}
		st.Ax = ax
	}
	lo, hi, base := sp.bounds(k, &pos, part)
	at := k.bin(OpAdd, U, k.bin(OpSub, U, lo, base), j)
	for _, x := range key {
		k.redirect(x, at, at)
	}
	for _, l := range pos.Live {
		k.redirect(l, hi, hi)
	}
	pos.Live = []int{hi}
	pos.Span = k.bin(OpSub, U, hi, lo) + 1
	pos.Part = part + 1
	return k, true
}

// shape is buffer shape sh of member q's output after the split.
func (sp splitter) shape(q *member, sh []int) []int {
	if q.axis >= 0 {
		a := q.axis
		return append(append(append([]int(nil), sh[:a]...), sp.S, sp.J), sh[a+1:]...)
	}
	return append(append([]int(nil), sh...), sp.S)
}

// relayout is coordinates ax of member q's output after the split: the
// position's coordinate the partition's and the position in it, an output
// the range was reduced away from the partition's appended.
func (sp splitter) relayout(q *member, ax []int, isKey func(int) bool, part, j int) ([]int, bool) {
	if q.axis < 0 {
		return append(append([]int(nil), ax...), part), true
	}
	a := q.axis
	if a >= len(ax) || !isKey(ax[a]) {
		return nil, false
	}
	return append(append(append([]int(nil), ax[:a]...), part, j), ax[a+1:]...), true
}

// bounds is a partition's first position, the first past it, and the
// range's aligned start: the live span from the aligned start, a
// partition's share of it rounded up to the alignment.
func (sp splitter) bounds(k *Kernel, pos *Positions, part int) (lo, hi, base int) {
	U, A := ir.U32, pos.Align
	first, end := pos.First, pos.End
	base = k.bin(OpMul, U, k.bin(OpDiv, U, first, k.u32(A)), k.u32(A))
	span := k.add(Node{Op: OpSelect, Type: U, A: k.add(Node{Op: OpLess, Type: ir.Pred, A: first, B: end}),
		B: k.bin(OpSub, U, end, base), C: k.u32(0)})
	share := k.bin(OpDiv, U, k.bin(OpAdd, U, span, k.u32(sp.S-1)), k.u32(sp.S))
	per := k.bin(OpMul, U, k.bin(OpDiv, U, k.bin(OpAdd, U, share, k.u32(A-1)), k.u32(A)), k.u32(A))
	lo = k.bin(OpMin, U, end, k.bin(OpAdd, U, base, k.bin(OpMul, U, part, per)))
	hi = k.bin(OpMin, U, end, k.bin(OpAdd, U, lo, per))
	return lo, hi, base
}

// copies is each partition's maximum and sum into the partial, Copies times
// each.
func (s SplitRange) copies(h Heads, S, stride int, part, maxes, sums Operand) *Kernel {
	U := ir.U32
	k := &Kernel{Name: "map", Kind: PrimMap, Space: []int{h.Rows, h.Query, S, s.Copies}, Heads: &h}
	c := k.coordNodes()
	k.Positions = &Positions{Axis: -1, Loop: -1, Part: c[2] + 1}
	heads := []int{h.Rows, h.Query, S}
	mb := k.buf(Buf{Kind: BufIn, Elem: ir.F32, Operand: maxes, Shape: heads})
	lb := k.buf(Buf{Kind: BufIn, Elem: ir.F32, Operand: sums, Shape: heads})
	pb := k.buf(Buf{Kind: BufOut, Elem: ir.F32, Operand: part, Extent: h.Rows * h.Query * S * stride})
	rh := k.bin(OpAdd, U, k.bin(OpMul, U, c[0], k.u32(h.Query)), c[1])
	at := func(off int) int {
		return k.bin(OpAdd, U, k.bin(OpAdd, U, k.bin(OpMul, U, k.bin(OpAdd, U, k.bin(OpMul, U, rh, k.u32(S)), c[2]), k.u32(stride)),
			k.u32(off)), c[3])
	}
	mv := k.add(Node{Op: OpLoad, Type: ir.F32, Buf: mb, Ax: []int{c[0], c[1], c[2]}})
	lv := k.add(Node{Op: OpLoad, Type: ir.F32, Buf: lb, Ax: []int{c[0], c[1], c[2]}})
	k.Stores = []Store{{Buf: pb, Ix: at(stride - 2*s.Copies), Val: mv, At: -1}, {Buf: pb, Ix: at(stride - s.Copies), Val: lv, At: -1}}
	return k
}

// merge is the normalization last with its reads of the weighted sums and
// the sums the split's combine: the partitions weighted by exp of their
// maximum less the greatest.
func (s SplitRange) merge(last *Kernel, h Heads, S, stride int, part, values, sums Operand) *Kernel {
	U, F := ir.U32, ir.F32
	k := &Kernel{Name: "map", Kind: PrimMap, Space: []int{h.Rows * h.Query * h.Dim}, Heads: &h, Part: last.Part}
	e := k.add(Node{Op: OpCoord, A: 0, Type: U})
	rh, d := k.bin(OpDiv, U, e, k.u32(h.Dim)), k.bin(OpRem, U, e, k.u32(h.Dim))
	pb := k.buf(Buf{Kind: BufIn, Elem: F, Operand: part, Extent: h.Rows * h.Query * S * stride})
	at := func(p, off int) int {
		return k.bin(OpAdd, U, k.bin(OpMul, U, k.bin(OpAdd, U, k.bin(OpMul, U, rh, k.u32(S)), k.u32(p)), k.u32(stride)), k.u32(off))
	}
	big := k.f32(math.Float32bits(math.Float32frombits(0xff7fffff)))
	maxes, sumsv := make([]int, S), make([]int, S)
	for p := 0; p < S; p++ {
		maxes[p] = k.load(pb, at(p, stride-2*s.Copies))
		sumsv[p] = k.load(pb, at(p, stride-s.Copies))
		big = k.bin(OpMax, F, big, maxes[p])
	}
	L, N := k.f32(0), k.f32(0)
	for p := 0; p < S; p++ {
		x := k.un(OpExp, F, k.bin(OpSub, F, maxes[p], big))
		L = k.add(Node{Op: OpFma, Type: F, A: sumsv[p], B: x, C: L})
		num := k.load(pb, k.bin(OpAdd, U, at(p, 0), d))
		N = k.add(Node{Op: OpFma, Type: F, A: num, B: x, C: N})
	}
	coords := k.decode(e, last.Space)
	st := last.Stores[0]
	leaf := func(i int, nd Node) (int, bool) {
		switch {
		case nd.Op == OpCoord:
			return coords[nd.A], true
		case nd.Op == OpLoad && last.Bufs[nd.Buf].Operand == sums:
			return L, true
		case nd.Op == OpLoad && last.Bufs[nd.Buf].Operand == values:
			return N, true
		}
		return 0, false
	}
	v := copyInto(k, last, st.Val, leaf)
	ns := Store{Buf: k.bufLike(last.Bufs[st.Buf]), Val: v, At: -1}
	for _, a := range st.Ax {
		ns.Ax = append(ns.Ax, copyInto(k, last, a, leaf))
	}
	if st.Ax == nil {
		ns.Ix = copyInto(k, last, st.Ix, leaf)
	}
	k.Stores = []Store{ns}
	k.Passes = append([]string(nil), last.Passes...)
	return k
}
