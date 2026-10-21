package jit

import "github.com/jitllm/jitllm/jit/ir"

// Mask zeroes what a loop over positions reads of workgroup memory past the
// live span, where its binding gives a position to another thread than the
// loop that wrote it: such a place holds what no trip of this step wrote.
type Mask struct{}

func (Mask) Name() string { return "mask" }

func (p Mask) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if !k.Work {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for c, l := range m.Loops {
			if !l.Range || l.Bound == Unbound || l.Span == 0 {
				continue
			}
			for b := range m.Bufs {
				w, ok := m.writer(b, c)
				if !ok || sameThreads(m.Loops[w].Bound, l.Bound) {
					continue
				}
				for _, f := range m.foldsOf(c) {
					for _, ld := range m.loadsFrom(m.Nodes[f].A, m.Bufs[b].Operand) {
						nd := m.Nodes[ld]
						if nd.Buf != b || len(nd.Ax) != 2 || m.Nodes[nd.Ax[1]].Op != OpMin {
							continue
						}
						j := m.Nodes[nd.Ax[1]].A
						live := m.add(Node{Op: OpLess, Type: ir.Pred, A: j, B: l.Span - 1})
						v := m.add(nd)
						m.Nodes[ld] = Node{Op: OpSelect, Type: nd.Type, A: live, B: v, C: m.f32(0)}
						changed = true
					}
				}
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// writer is the last loop before loop c that stores into workgroup buffer
// b.
func (k *Kernel) writer(b, c int) (int, bool) {
	if k.Bufs[b].Kind != BufShared {
		return 0, false
	}
	w := -1
	for _, s := range k.Stores {
		if s.Run && s.At == c || !s.Barrier && !s.Run && s.At == c {
			break
		}
		if !s.Barrier && !s.Run && s.Buf == b && s.At >= 0 {
			w = s.At
		}
	}
	return w, w >= 0
}

// sameThreads reports whether two bindings give a position the same thread.
func sameThreads(a, b Bound) bool {
	thread := func(x Bound) bool { return x == BoundRuns || x == BoundSpread }
	return thread(a) == thread(b)
}
