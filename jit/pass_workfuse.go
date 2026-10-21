package jit

import "github.com/jitllm/jitllm/jit/ir"

// WorkFuse fuses consecutive kernels of a split range -- each over its
// rows, heads and partitions and coordinates of its own -- into one
// workgroup a (row, head, partition): each kernel a phase, its own
// coordinates loops of the phase, a barrier before a phase that reads what
// an earlier one wrote. The workgroup's threads are its warps' lanes.
type WorkFuse struct {
	Work Workgroups
}

func (WorkFuse) Name() string { return "workfuse" }

func (w WorkFuse) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	var out []*Kernel
	for i := 0; i < len(ks); i++ {
		j := i
		for j < len(ks) && partAxis(ks[j]) >= 0 && ks[j].Heads != nil && len(ks[j].Space) >= 3 &&
			ks[j].Space[0] == ks[i].Space[0] && ks[j].Space[1] == ks[i].Space[1] &&
			ks[j].Space[partAxis(ks[j])] == ks[i].Space[partAxis(ks[i])] {
			j++
		}
		if j-i < 2 {
			out = append(out, ks[i])
			continue
		}
		k := w.fuse(ks[i:j], d)
		k.Passes = []string{w.Name()}
		out = append(out, k)
		i = j - 1
	}
	return out, nil
}

// partAxis is the axis of k's space that is its partition, -1 for none.
func partAxis(k *Kernel) int {
	if k.Work || k.Positions == nil || k.Positions.Part == 0 {
		return -1
	}
	nd := k.Nodes[k.Positions.Part-1]
	if nd.Op != OpCoord {
		return -1
	}
	return nd.A
}

func (w WorkFuse) fuse(ms []*Kernel, d Device) *Kernel {
	h := *ms[0].Heads
	g := w.Work.geometry(h, d.Lanes)
	pa := partAxis(ms[0])
	k := &Kernel{Name: "stage", Kind: PrimReduce, Work: true, Group: g.warps * d.Lanes, Heads: &h, Part: ms[0].Part,
		Space: []int{ms[0].Space[0], ms[0].Space[1], ms[0].Space[pa]}}
	outer := []int{k.add(Node{Op: OpCoord, A: 0, Type: ir.U32}), k.add(Node{Op: OpCoord, A: 1, Type: ir.U32}),
		k.add(Node{Op: OpCoord, A: 2, Type: ir.U32})}
	written := map[Operand]bool{}
	for _, m := range ms {
		reads := false
		for _, b := range m.Bufs {
			reads = reads || b.Kind == BufIn && written[b.Operand]
		}
		if reads {
			k.Stores = append(k.Stores, Store{Barrier: true, At: -1})
		}
		w.phase(k, m, outer)
		for _, s := range m.Stores {
			if !s.Barrier && !s.Run {
				written[m.Bufs[s.Buf].Operand] = true
			}
		}
	}
	return k
}

// phase is kernel m as a phase of workgroup k: its row, head and partition
// coordinates the workgroup's, its other coordinates loops nested in
// order, its loops inside them.
func (w WorkFuse) phase(k, m *Kernel, outer []int) {
	pa := partAxis(m)
	coord := map[int]int{0: outer[0], 1: outer[1], pa: outer[2]}
	parent := -1
	axisLoop := map[int]int{}
	for a := range m.Space {
		if _, ok := coord[a]; ok {
			continue
		}
		l := k.loop(m.Space[a], parent)
		coord[a] = k.add(Node{Op: OpIter, At: l, Type: ir.U32})
		axisLoop[a] = l
		parent = l
	}
	loopOf := make([]int, len(m.Loops))
	for i, l := range m.Loops {
		p := l.Parent
		if p >= 0 {
			p = loopOf[p]
		} else {
			p = parent
		}
		loopOf[i] = len(k.Loops)
		l.Parent = p
		k.Loops = append(k.Loops, l)
	}
	nodeOf := make([]int, len(m.Nodes))
	for i := range nodeOf {
		nodeOf[i] = -1
	}
	var walk func(i int) int
	walk = func(i int) int {
		if nodeOf[i] >= 0 {
			return nodeOf[i]
		}
		nd := m.Nodes[i]
		if nd.Op == OpCoord {
			nodeOf[i] = coord[nd.A]
			return nodeOf[i]
		}
		if nd.Op == OpIter || nd.Op == OpFold {
			nd.At = loopOf[nd.At]
		}
		if nd.Op == OpLoad {
			nd.Buf = k.bufLike(m.Bufs[nd.Buf])
		}
		nodeOf[i] = k.add(Node{})
		for _, a := range k.args(&nd) {
			*a = walk(*a)
		}
		k.Nodes[nodeOf[i]] = nd
		return nodeOf[i]
	}
	for i := range m.Nodes {
		walk(i)
	}
	for i, l := range m.Loops {
		if l.Count > 0 {
			k.Loops[loopOf[i]].Count = walk(l.Count-1) + 1
		}
	}
	at := parent
	for _, s := range m.Stores {
		ns := Store{Barrier: s.Barrier, Run: s.Run, At: at}
		if s.At >= 0 {
			ns.At = loopOf[s.At]
		}
		if !s.Barrier && !s.Run {
			ns.Buf, ns.Val = k.bufLike(m.Bufs[s.Buf]), walk(s.Val)
			if s.Ax != nil {
				for _, a := range s.Ax {
					ns.Ax = append(ns.Ax, walk(a))
				}
			} else {
				ns.Ix = walk(s.Ix)
			}
		}
		k.Stores = append(k.Stores, ns)
	}
	if p := m.Positions; p != nil && p.Span > 0 {
		l := -1
		if p.Axis >= 0 {
			l = axisLoop[p.Axis]
		} else if p.Loop >= 0 {
			l = loopOf[p.Loop]
		}
		if l >= 0 {
			k.Loops[l].Range, k.Loops[l].Span = true, walk(p.Span-1)+1
		}
	}
}
