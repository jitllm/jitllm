package composer

// RowCalls holds a kernel of a form's capacity (Kernel.Capacity) to the
// call's rows on a device whose code runs a call at a time: the leading
// axis, the step's rows, stays the calls' space, and the caller runs the
// units of it the call's rows reach (Code.LiveUnitsOf), as a GPU's launch
// is held to them (CallRows). A kernel over heads keeps its rows and query
// heads as the calls' space (Serial). Each call's coordinates move its
// buffers' offsets (Args), so its code addresses the one call's elements.
type RowCalls struct{}

func (RowCalls) Name() string { return "rowcalls" }

func (p RowCalls) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Capacity == 0 || k.RowsHeld || k.Work || k.Group > 0 || len(k.Space) == 0 || k.Space[0] == 0 {
			return nil, false, nil
		}
		m := k.clone()
		m.RowsHeld, m.Calls = true, 1
		if h := m.Heads; h != nil && len(m.Space) >= 2 && m.Space[0] == h.Rows && m.Space[1] == h.Query {
			m.Calls = 2
		}
		return []*Kernel{m}, true, nil
	})
}
