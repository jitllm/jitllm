package composer

import "reflect"

// Same drops a kernel that computes what an earlier one does -- the same
// space, nodes, loops and stores, its one output a scratch of its own where
// the earlier one's is another -- where nothing between them writes what it
// reads: its readers read the earlier one's output instead. Projections of
// one row each stage the row's halves.
type Same struct{}

func (Same) Name() string { return "same" }

func (p Same) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for j := 1; j < len(out); j++ {
		oj, ok := scratchOutput(out[j])
		if !ok {
			continue
		}
		for i := j - 1; i >= 0; i-- {
			if oi, ok := scratchOutput(out[i]); ok && sameAs(out[i], out[j], oi, oj) {
				for k := j + 1; k < len(out); k++ {
					if out[k].reads(oj) {
						out[k] = out[k].renamed(oj, oi)
					}
				}
				out = append(out[:j], out[j+1:]...)
				j--
				break
			}
			if writesWhatReads(out[i], out[j]) {
				break
			}
		}
	}
	return out, nil
}

// scratchOutput is the one operand k writes, where that is a scratch of the
// step's own.
func scratchOutput(k *Kernel) (Operand, bool) {
	var o Operand
	n := 0
	for _, s := range k.Stores {
		if s.Barrier || s.Run {
			continue
		}
		if b := k.Bufs[s.Buf].Operand; n == 0 || b != o {
			o = b
			n++
		}
	}
	return o, n == 1 && o.Kind == OperandScratch && o.Temp
}

// sameAs reports whether b is a with its output oi named ob.
func sameAs(a, b *Kernel, oi, ob Operand) bool {
	x, y := a.clone(), b.renamed(ob, oi)
	y.Name, y.Passes = x.Name, x.Passes
	return reflect.DeepEqual(x, y)
}

// renamed is k with its buffers of operand from those of to.
func (k *Kernel) renamed(from, to Operand) *Kernel {
	m := k.clone()
	m.Own = k.ownRenamed(from, to)
	for i := range m.Bufs {
		if m.Bufs[i].Operand == from {
			m.Bufs[i].Operand = to
		}
	}
	return m
}

// writesWhatReads reports whether w writes anything r reads or writes.
func writesWhatReads(w, r *Kernel) bool {
	for _, b := range r.Bufs {
		if w.writes(b.Operand) {
			return true
		}
	}
	return false
}
