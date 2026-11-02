package composer

import "github.com/jitllm/jitllm/composer/ir"

// Serial binds a kernel's elements to loops, for a device whose code runs a
// call at a time: where the kernel carries heads its leading axes over rows
// and query heads stay the calls' space, and every other axis is a loop,
// the outermost first, the kernel's own loops inside the innermost.
type Serial struct{}

func (Serial) Name() string { return "serial" }

func (p Serial) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Group > 0 || k.Work {
			return nil, false, nil
		}
		calls := 0
		if h := k.Heads; h != nil && len(k.Space) >= 2 && k.Space[0] == h.Rows && k.Space[1] == h.Query {
			calls = 2
		}
		if k.Calls > 0 {
			calls = k.Calls
		}
		if calls == len(k.Space) {
			return nil, false, nil
		}
		m := k.clone()
		own := len(m.Loops)
		loopOf := map[int]int{}
		parent := -1
		for a := calls; a < len(m.Space); a++ {
			parent = m.loop(m.Space[a], parent)
			loopOf[a] = parent
		}
		for c := 0; c < own; c++ {
			if m.Loops[c].Parent < 0 {
				m.Loops[c].Parent = parent
			}
		}
		for i := range m.Nodes {
			nd := &m.Nodes[i]
			if nd.Op != OpCoord {
				continue
			}
			if c, ok := loopOf[nd.A]; ok {
				*nd = Node{Op: OpIter, At: c, Type: ir.U32}
			}
		}
		for i := range m.Stores {
			if m.Stores[i].At < 0 {
				m.Stores[i].At = parent
			}
		}
		if m.Share > 0 && calls > 0 && len(m.Space) > calls {
			// A call's rows, from its coordinate's unit on, are a length the
			// caller sizes for the workers it runs.
			last := len(m.Space) - 1
			m.Loops[loopOf[last]].Count = m.add(Node{Op: OpLength, A: -1, Type: ir.U32, Imm: int64(m.Space[last])}) + 1
		}
		if pos := m.Positions; pos != nil && pos.Axis >= calls {
			q := *pos
			q.Loop, q.Axis = loopOf[pos.Axis], -1
			m.Loops[q.Loop].Range = true
			m.Positions = &q
		}
		m.Space = append([]int(nil), m.Space[:calls]...)
		return []*Kernel{m}, true, nil
	})
}
