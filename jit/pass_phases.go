package jit

import "github.com/jitllm/jitllm/jit/ir"

// Phases runs consecutive kernels that each launch one workgroup of the
// same width as one launch: each kernel a phase of the workgroup, a barrier
// between two, its elements its threads' as its own launch dealt them.
// Params caps the launch's bound buffers (the parameter list a backend
// launches with).
type Phases struct {
	Params int
}

func (Phases) Name() string { return "phases" }

func (p Phases) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	made := map[Operand]bool{}
	for _, k := range ks {
		for _, b := range k.Bufs {
			if b.Kind == BufOut {
				made[b.Operand] = true
			}
		}
	}
	var out []*Kernel
	for i := 0; i < len(ks); i++ {
		j := i
		bound := map[Operand]bool{}
		for j < len(ks) && phaseable(ks[j]) && ks[j].Group == ks[i].Group && ks[j].Part == ks[i].Part &&
			ks[j].Of == ks[i].Of && ks[j].Apart == ks[i].Apart && !atTheEdge(ks[j], made) {
			for _, b := range ks[j].Bufs {
				if b.Kind != BufShared {
					bound[b.Operand] = true
				}
			}
			if 2*len(bound) > p.Params {
				break
			}
			j++
		}
		if j-i < 2 {
			out = append(out, ks[i])
			continue
		}
		k := &Kernel{Name: "phases", Kind: PrimReduce, Space: []int{ks[i].Group}, Group: ks[i].Group, Part: ks[i].Part,
			Of: ks[i].Of, Apart: ks[i].Apart, Passes: []string{p.Name()}}
		for n, m := range ks[i:j] {
			if n > 0 {
				k.Stores = append(k.Stores, Store{Barrier: true, At: -1})
			}
			p.phase(k, m)
		}
		k.compact()
		out = append(out, k)
		i = j - 1
	}
	return out, nil
}

// phaseable says k launches one workgroup and states nothing a launch of
// its own carries beside its nodes, loops and stores.
func phaseable(k *Kernel) bool {
	if k.Group <= 0 || k.groups() != 1 || k.Kind == PrimMatrix || k.Kind == PrimScan || k.Ones != 0 || k.Calls != 0 ||
		k.Pick != nil || k.Quarter != nil || k.Positions != nil || k.Heads != nil || k.Rows != 0 || k.Clamp != 0 ||
		k.RowsHeld || k.Tile != 0 || k.TileRows != 0 || k.Few != 0 || k.Many != 0 || k.Running != nil || k.Seq != nil ||
		k.Args != nil || k.Ptrs != nil || k.Lines != nil || k.Jam != 0 || k.Resident != 0 {
		return false
	}
	for _, nd := range k.Nodes {
		if nd.Tile != nil || k.Work && nd.Op == OpFlat {
			return false
		}
	}
	for _, l := range k.Loops {
		if l.Cursors != nil || l.Reg != 0 {
			return false
		}
	}
	for _, s := range k.Stores {
		if s.Ptr != 0 || s.Copy != 0 || s.Tile {
			return false
		}
	}
	return true
}

// phase appends kernel m to k: a launch of its own's coordinates made from
// the thread, clamped to its last element, or zero for a workgroup's.
func (Phases) phase(k, m *Kernel) {
	coord := make([]int, len(m.Space))
	if m.Work {
		for a := range coord {
			coord[a] = k.u32(0)
		}
	} else {
		rest := k.bin(OpMin, ir.U32, k.add(Node{Op: OpCoord, A: 0, Type: ir.U32}), k.u32(m.Elements()-1))
		for a := len(m.Space) - 1; a >= 0; a-- {
			switch ext := m.Space[a]; {
			case ext == 1:
				coord[a] = k.u32(0)
			case a == 0:
				coord[a] = rest
			default:
				coord[a] = k.bin(OpRem, ir.U32, rest, k.u32(ext))
				rest = k.bin(OpDiv, ir.U32, rest, k.u32(ext))
			}
		}
	}
	bufOf := make([]int, len(m.Bufs))
	for i, b := range m.Bufs {
		if b.Kind == BufShared {
			bufOf[i] = k.buf(b)
		} else {
			bufOf[i] = k.bufLike(b)
		}
	}
	lb := len(k.Loops)
	for _, l := range m.Loops {
		if l.Parent >= 0 {
			l.Parent += lb
		}
		if l.Of > 0 {
			l.Of += lb
		}
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
			nd.At += lb
		}
		if nd.Op == OpLoad {
			nd.Buf = bufOf[nd.Buf]
		}
		nodeOf[i] = k.add(Node{})
		for _, a := range k.Operands(&nd) {
			*a = walk(*a)
		}
		k.Nodes[nodeOf[i]] = nd
		return nodeOf[i]
	}
	for i := range m.Nodes {
		walk(i)
	}
	ref := func(v int) int {
		if v > 0 {
			return walk(v-1) + 1
		}
		return v
	}
	for i, l := range m.Loops {
		at := &k.Loops[lb+i]
		at.Count, at.Span, at.End, at.Elem = ref(l.Count), ref(l.Span), ref(l.End), ref(l.Elem)
	}
	for _, s := range m.Stores {
		ns := s
		if s.At >= 0 {
			ns.At += lb
		}
		if !s.Barrier && !s.Run {
			ns.Buf, ns.Val = bufOf[s.Buf], walk(s.Val)
			if s.Ax != nil {
				ns.Ax = make([]int, len(s.Ax))
				for a, x := range s.Ax {
					ns.Ax[a] = walk(x)
				}
			} else {
				ns.Ix = walk(s.Ix)
			}
		}
		k.Stores = append(k.Stores, ns)
	}
}

// atTheEdge says k reads a value of the part's input or writes one it hands
// back: the tier writes the output over the input only where the input's
// every read runs in a launch before the output's first write.
func atTheEdge(k *Kernel, made map[Operand]bool) bool {
	for _, b := range k.Bufs {
		if b.Kind == BufOut && b.Operand.Out || b.Kind == BufIn && b.Operand.Kind == OperandValue && !made[b.Operand] {
			return true
		}
	}
	return false
}
