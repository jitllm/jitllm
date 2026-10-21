package jit

import "github.com/jitllm/jitllm/jit/ir"

// Stage moves a workgroup's temporaries -- values it alone writes and reads
// -- into one plane of workgroup memory: an element at the coordinates that
// are not the workgroup's own, a head counted from the workgroup's first,
// each other coordinate held to its extent. Workgroup memory a binding made
// for the warps' sums, live after the temporaries, shares the plane.
type Stage struct{}

func (Stage) Name() string { return "stage" }

func (p Stage) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work || len(k.Space) != 3 || k.Heads == nil {
			return nil, false, nil
		}
		others := map[Operand]bool{}
		for _, o := range ks {
			if o == k {
				continue
			}
			for _, b := range o.Bufs {
				others[b.Operand] = true
			}
		}
		G := k.Heads.Query / k.Space[1]
		m := k.clone()
		group := -1
		for i, nd := range m.Nodes {
			if nd.Op == OpCoord && nd.A == 1 {
				group = i
			}
		}
		staged := map[int]bool{}
		size := 0
		for i, b := range m.Bufs {
			if b.Operand.Temp && b.Shape != nil && b.Kind != BufShared && !others[b.Operand] && len(b.Shape) == 4 {
				staged[i] = true
				size = max(size, G*b.Shape[3])
			}
		}
		if len(staged) == 0 || group < 0 {
			return nil, false, nil
		}
		for _, b := range m.Bufs {
			if b.Plane {
				size = max(size, b.Size)
			}
		}
		J := 0
		for i := range staged {
			J = max(J, m.Bufs[i].Shape[3])
		}
		plane := m.buf(Buf{Kind: BufShared, Elem: ir.F32, Size: size, Shape: []int{size / J, J}})
		base := m.bin(OpMul, ir.U32, group, m.u32(G))
		local := func(b int, ax []int) []int {
			sh := m.Bufs[b].Shape
			return []int{m.bin(OpSub, ir.U32, ax[1], base), m.bin(OpMin, ir.U32, ax[3], m.u32(sh[3]-1))}
		}
		for i := range m.Nodes {
			nd := m.Nodes[i]
			if nd.Op != OpLoad {
				continue
			}
			if staged[nd.Buf] {
				m.Nodes[i] = Node{Op: OpLoad, Type: ir.F32, Buf: plane, Ax: local(nd.Buf, nd.Ax)}
			} else if m.Bufs[nd.Buf].Plane {
				m.Nodes[i].Buf = plane
			}
		}
		for i := range m.Stores {
			s := &m.Stores[i]
			if s.Barrier || s.Run {
				continue
			}
			if staged[s.Buf] {
				s.Ax, s.Buf = local(s.Buf, s.Ax), plane
			} else if m.Bufs[s.Buf].Plane {
				s.Buf = plane
			}
		}
		return []*Kernel{m}, true, nil
	})
}
