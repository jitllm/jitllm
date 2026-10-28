package jit

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// RunningSplit is a split whose partitions one workgroup runs in order: the
// partial Part, Stride words a row and head -- the weighted sums, then
// Copies copies of the maximum and of the sum -- one slot a row and head.
type RunningSplit struct {
	Part           Operand
	Stride, Copies int
}

// Running runs a split's partitions in order where SplitRange chose so
// (Kernel.Running): the workgroup's partition coordinate a loop over them,
// the partial each partition stores workgroup memory, combined after it
// into the running maximum, sum and weighted sums, each side weighted by
// exp of its maximum less the greater. The running values sit in two slots,
// a trip reading one and writing the other; after the last they are the
// partial the merge reads, one a row and head. So no buffer holds a
// partition's partial: what the attention keeps is its rows' and heads'.
type Running struct{}

func (Running) Name() string { return "running" }

func (p Running) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Running == nil {
			return nil, false, nil
		}
		if !k.Work || len(k.Space) != 3 || k.Heads == nil || k.Space[1] <= 0 || k.Heads.Query%k.Space[1] != 0 {
			return nil, false, fmt.Errorf("jit: %s runs its partitions in order and is not a workgroup a row, head group and partition", k.Name)
		}
		m, err := p.form(k)
		if err != nil {
			return nil, false, err
		}
		return []*Kernel{m}, true, nil
	})
}

func (p Running) form(k *Kernel) (*Kernel, error) {
	U, F := ir.U32, ir.F32
	rs := *k.Running
	m := k.clone()
	m.Running = nil
	S, G, D, H, W := k.Space[2], k.Heads.Query/k.Space[1], k.Heads.Dim, k.Heads.Query, rs.Stride
	mOff, lOff := W-2*rs.Copies, W-rs.Copies
	m.Space[2] = 1
	pb := -1
	for i, b := range m.Bufs {
		if b.Operand == rs.Part && b.Kind == BufOut {
			pb = i
		}
	}
	if pb < 0 {
		return nil, fmt.Errorf("jit: %s runs its partitions in order and stores no partial", k.Name)
	}
	P := m.loop(S, -1)
	m.Loops[P].Uniform = true
	part := m.add(Node{Op: OpIter, At: P, Type: U})
	same := map[int]int{}
	for i, nd := range m.Nodes {
		if nd.Op == OpCoord && nd.A == 2 {
			same[i] = part
		}
	}
	m.redirectMany(same)
	for c := range m.Loops {
		if c != P && m.Loops[c].Parent < 0 {
			m.Loops[c].Parent = P
		}
	}
	// Workgroup memory changes from partition to partition: a read of it
	// is the trip's, never made once before the loop.
	for i, nd := range m.Nodes {
		if nd.Op == OpLoad && nd.At == 0 && m.Bufs[nd.Buf].Kind == BufShared {
			m.Nodes[i].At = P + 1
		}
	}
	pt := m.buf(Buf{Kind: BufShared, Elem: F, Size: G * W})
	run := m.buf(Buf{Kind: BufShared, Elem: F, Size: 2 * G * W})
	row, grp := m.add(Node{Op: OpCoord, A: 0, Type: U}), m.add(Node{Op: OpCoord, A: 1, Type: U})
	first := m.bin(OpMul, U, m.bin(OpAdd, U, m.bin(OpMul, U, row, m.u32(H)), m.bin(OpMul, U, grp, m.u32(G))), m.u32(W))
	// spread is a loop of n trips dealt over the workgroup in loop parent,
	// and its trip, a surplus thread's held to the last: what it stores is
	// what the last trip's thread stores.
	spread := func(n, parent int) (int, int) {
		l := m.loop(n, parent)
		m.Loops[l].Spread, m.Loops[l].Uniform = true, n%m.Group == 0
		it := m.add(Node{Op: OpIter, At: l, Type: U})
		if n%m.Group != 0 {
			it = m.bin(OpMin, U, it, m.u32(n-1))
		}
		return l, it
	}
	load := func(buf, ix int) int { return m.add(Node{Op: OpLoad, Type: F, Buf: buf, A: ix}) }
	at := func(base, j, off int) int {
		return m.bin(OpAdd, U, base, m.bin(OpAdd, U, m.bin(OpMul, U, j, m.u32(W)), off))
	}
	minus := math.Float32bits(math.Float32frombits(0xff7fffff))
	// is is x == y of unsigned x and y: x - y wraps past every small value
	// but 0.
	is := func(x, y int) int {
		return m.add(Node{Op: OpLess, Type: ir.Pred, A: m.bin(OpSub, U, x, m.u32(y)), B: m.u32(1)})
	}

	var stores []Store
	// The running values' first slot: no maximum, no sum, no weighted sums.
	zl, z := spread(G*W, -1)
	isMax := is(m.bin(OpRem, U, z, m.u32(W)), mOff)
	stores = append(stores, Store{Buf: run, Ix: z, Val: m.add(Node{Op: OpSelect, Type: F, A: isMax, B: m.f32(minus), C: m.f32(0)}), At: zl},
		Store{Barrier: true, At: -1})
	for _, s := range m.Stores {
		if s.At < 0 {
			s.At = P
		}
		if !s.Barrier && !s.Run && s.Buf == pb {
			if s.Ax != nil {
				return nil, fmt.Errorf("jit: %s stores its partial at coordinates", k.Name)
			}
			s.Buf, s.Ix = pt, m.bin(OpSub, U, s.Ix, first)
		}
		stores = append(stores, s)
	}
	stores = append(stores, Store{Barrier: true, At: P})
	old := m.bin(OpMul, U, m.bin(OpAnd, U, part, m.u32(1)), m.u32(G*W))
	next := m.bin(OpMul, U, m.bin(OpAnd, U, m.bin(OpAdd, U, part, m.u32(1)), m.u32(1)), m.u32(G*W))
	zero := m.u32(0)
	// weights is head j's running and partial weights: exp of each maximum
	// less the greater, and the greater.
	weights := func(j int) (int, int, int) {
		mo, mp := load(run, at(old, j, m.u32(mOff))), load(pt, at(zero, j, m.u32(mOff)))
		mn := m.bin(OpMax, F, mo, mp)
		return m.un(OpExp, F, m.bin(OpSub, F, mo, mn)), m.un(OpExp, F, m.bin(OpSub, F, mp, mn)), mn
	}
	al, e := spread(G*D, P)
	j, dd := m.bin(OpDiv, U, e, m.u32(D)), m.bin(OpRem, U, e, m.u32(D))
	a, b, _ := weights(j)
	acc := m.add(Node{Op: OpFma, Type: F, A: load(run, at(old, j, dd)), B: a, C: m.bin(OpMul, F, load(pt, at(zero, j, dd)), b)})
	stores = append(stores, Store{Buf: run, Ix: at(next, j, dd), Val: acc, At: al})
	hl, jh := spread(G, P)
	a, b, mn := weights(jh)
	sum := m.add(Node{Op: OpFma, Type: F, A: load(run, at(old, jh, m.u32(lOff))), B: a, C: m.bin(OpMul, F, load(pt, at(zero, jh, m.u32(lOff))), b)})
	stores = append(stores, Store{Buf: run, Ix: at(next, jh, m.u32(lOff)), Val: sum, At: hl},
		Store{Buf: run, Ix: at(next, jh, m.u32(mOff)), Val: mn, At: hl},
		Store{Barrier: true, At: P})
	// After the last partition the running values are the partial: the
	// weighted sums, the maximum and the sum.
	last := m.u32(S % 2 * G * W)
	fl, f := spread(G*(D+2), -1)
	q := m.bin(OpRem, U, f, m.u32(D+2))
	off := m.add(Node{Op: OpSelect, Type: U, A: m.add(Node{Op: OpLess, Type: ir.Pred, A: q, B: m.u32(D)}), B: q,
		C: m.add(Node{Op: OpSelect, Type: U, A: is(q, D), B: m.u32(mOff), C: m.u32(lOff)})})
	jf := m.bin(OpDiv, U, f, m.u32(D+2))
	stores = append(stores, Store{Buf: pb, Ix: at(first, jf, off), Val: load(run, at(last, jf, off)), At: fl})
	m.Stores = stores
	return m, nil
}
