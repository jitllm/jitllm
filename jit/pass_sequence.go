package jit

import "slices"

// ConstRuns loads, outside every loop, the constants of a run of
// operations that each read one constant just before the run.
type ConstRuns struct{}

func (ConstRuns) Name() string { return "construns" }

func (p ConstRuns) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return refineSeq(p.Name(), ks, (*Kernel).constRuns)
}

// ConstSiblings loads, outside every loop, a constant one operation reads
// before the values that operation's reader reads beside it.
type ConstSiblings struct{}

func (ConstSiblings) Name() string { return "constsiblings" }

func (p ConstSiblings) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return refineSeq(p.Name(), ks, (*Kernel).constsBeforeSiblings)
}

// CursorOrder orders each loop's cursors: a buffer it stores to through its
// own address first, then in the order its code first reaches them.
type CursorOrder struct{}

func (CursorOrder) Name() string { return "cursororder" }

func (p CursorOrder) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return refineSeq(p.Name(), ks, (*Kernel).orderCursors)
}

// refineSeq runs f over each ordered kernel, a kernel it changes the
// sequence or a loop's cursors of formed by pass name.
func refineSeq(name string, ks []*Kernel, f func(*Kernel)) ([]*Kernel, error) {
	return eachKernel(name, ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Seq == nil {
			return nil, false, nil
		}
		m := k.clone()
		f(m)
		changed := !slices.Equal(m.Seq, k.Seq)
		for c := range m.Loops {
			changed = changed || !slices.Equal(m.Loops[c].Cursors, k.Loops[c].Cursors)
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}
