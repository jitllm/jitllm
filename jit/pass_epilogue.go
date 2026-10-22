package jit

import "slices"

// Epilogue runs a map in the workgroup kernel before it where the kernel's
// last step writes each element of the map's space whole -- a store of the
// kernel's own, after every barrier, at the element's coordinates -- and the
// map reads that output only at its own element: the map's value computed
// from the stored value where the store is, its stores the kernel's, and
// the kernel's own store kept only where a later kernel reads it.
type Epilogue struct{}

func (Epilogue) Name() string { return "epilogue" }

func (p Epilogue) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i := 0; i+1 < len(out); i++ {
		w, m := out[i], out[i+1]
		if !w.Work || m.Work || m.Kind != PrimMap || len(m.Loops) != 0 || m.Part != w.Part || len(w.Stores) == 0 {
			continue
		}
		last := w.Stores[len(w.Stores)-1]
		if last.Barrier || last.Run || last.At != -1 || last.Ax == nil {
			continue
		}
		o := w.Bufs[last.Buf].Operand
		if !slices.Equal(w.Bufs[last.Buf].Shape, m.Space) || !readsOwn(m, o) || w.reads(o) {
			continue
		}
		f := w.clone()
		leaf := func(_ int, nd Node) (int, bool) {
			switch {
			case nd.Op == OpCoord:
				return last.Ax[nd.A], true
			case nd.Op == OpLoad && m.Bufs[nd.Buf].Operand == o:
				return last.Val, true
			}
			return 0, false
		}
		var add []Store
		for _, s := range m.Stores {
			ns := Store{Buf: f.bufLike(m.Bufs[s.Buf]), At: -1, Val: copyInto(f, m, s.Val, leaf)}
			if s.Ax != nil {
				for _, a := range s.Ax {
					ns.Ax = append(ns.Ax, copyInto(f, m, a, leaf))
				}
			} else {
				ns.Ix = copyInto(f, m, s.Ix, leaf)
			}
			add = append(add, ns)
		}
		if soleReader(out, m, o) {
			f.Stores = f.Stores[:len(f.Stores)-1]
		}
		f.Stores = append(f.Stores, add...)
		f.Name = w.Name + "+" + m.Name
		f.Passes = append(append([]string(nil), w.Passes...), p.Name())
		f.compact()
		out[i] = f
		out = append(out[:i+1], out[i+2:]...)
	}
	return out, nil
}

// readsOwn reports whether map m reads o, and only at its own coordinates.
func readsOwn(m *Kernel, o Operand) bool {
	any := false
	for _, nd := range m.Nodes {
		if nd.Op != OpLoad || m.Bufs[nd.Buf].Operand != o {
			continue
		}
		if len(nd.Ax) != len(m.Space) {
			return false
		}
		for a, x := range nd.Ax {
			if m.Nodes[x].Op != OpCoord || m.Nodes[x].A != a {
				return false
			}
		}
		any = true
	}
	return any
}
