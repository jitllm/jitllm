package jit

import "github.com/jitllm/jitllm/jit/ir"

// Runs finds a position's page once a run of positions in a bound loop over
// a range: a position held into the range, min(max(x, first), last), and
// divided by or reduced to a page of a multiple of the range's alignment,
// is divided at the run's first position held to the range's last aligned
// run. The run is the lanes' positions where a binding runs lanes over
// consecutive positions, else a trip's; where it is the lanes', a
// position's place in its page is its run's place plus its lane.
type Runs struct{}

func (Runs) Name() string { return "runs" }

func (p Runs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for c, l := range m.Loops {
			if !l.Range || l.Bound == Unbound || l.End == 0 || l.Align <= 1 {
				continue
			}
			for _, i := range m.bodyOf(c) {
				nd := m.Nodes[i]
				if nd.Op != OpDiv && nd.Op != OpRem || m.Nodes[nd.B].Op != OpConst {
					continue
				}
				P := int(m.Nodes[nd.B].Imm)
				x, ok := m.held(nd.A)
				if !ok || P%l.Align != 0 {
					continue
				}
				lane := m.laneOf()
				if nd.Op == OpRem && (l.Bound != BoundRuns || lane < 0) {
					continue
				}
				start := m.runStart(x, c, l.Bound)
				last := m.bin(OpSub, ir.U32, m.bin(OpMax, ir.U32, l.End-1, m.u32(1)), m.u32(1))
				aligned := m.bin(OpMul, ir.U32, m.bin(OpDiv, ir.U32, last, m.u32(l.Align)), m.u32(l.Align))
				h := m.bin(OpMin, ir.U32, start, aligned)
				v := m.bin(nd.Op, ir.U32, h, nd.B)
				if nd.Op == OpRem {
					v = m.bin(OpAdd, ir.U32, v, lane)
				}
				m.redirect(i, v, v)
				changed = true
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// bodyOf is the nodes loop c's trips compute.
func (k *Kernel) bodyOf(c int) []int {
	seen := map[int]bool{}
	var out []int
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := k.Nodes[i]
		if nd.Op == OpFold && nd.At != c {
			return
		}
		for _, a := range k.args(&nd) {
			walk(*a)
		}
		out = append(out, i)
	}
	for _, f := range k.foldsOf(c) {
		walk(k.Nodes[f].A)
	}
	for _, s := range k.Stores {
		if s.At == c && !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				walk(*a)
			}
		}
	}
	return out
}

// held is x of a position held into its range, min(max(x, first), last).
func (k *Kernel) held(i int) (int, bool) {
	nd := k.Nodes[i]
	if nd.Op != OpMin || k.Nodes[nd.A].Op != OpMax {
		return 0, false
	}
	return k.Nodes[nd.A].A, true
}

// laneOf is the thread's lane node, -1 for none.
func (k *Kernel) laneOf() int {
	for i, nd := range k.Nodes {
		if nd.Op == OpRem && k.Nodes[nd.A].Op == OpThread && k.Nodes[nd.B].Op == OpConst {
			return i
		}
	}
	return -1
}

// runStart is position x at its run's first: under lanes over consecutive
// positions the lane's offset dropped, else the trip's first position --
// the threads' and the unrolled steps' offsets dropped.
func (k *Kernel) runStart(x, c int, b Bound) int {
	sub := map[int]int{}
	zero := k.u32(0)
	for i, nd := range k.Nodes {
		switch {
		case nd.Op == OpRem && k.Nodes[nd.A].Op == OpThread:
			sub[i] = zero
		case b != BoundRuns && nd.Op == OpDiv && k.Nodes[nd.A].Op == OpThread:
			sub[i] = zero
		case b != BoundRuns && nd.Op == OpThread:
			sub[i] = zero
		case b != BoundRuns && nd.Op == OpAdd && k.Nodes[nd.B].Op == OpConst && k.isTripBase(nd.A, c):
			sub[i] = nd.A
		}
	}
	return k.copyOf(x, sub)
}

// isTripBase reports whether node i is loop c's iteration times a constant.
func (k *Kernel) isTripBase(i, c int) bool {
	nd := k.Nodes[i]
	return nd.Op == OpMul && k.Nodes[nd.A].Op == OpIter && k.Nodes[nd.A].At == c && k.Nodes[nd.B].Op == OpConst
}
