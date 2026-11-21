package cpudev

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// Bindings is where the CPU device's forms read their weights: each form's
// weights resolved once for the frame its block is in (Bind), each block's
// routed banks as one view of its experts in the frames' expert pages, and
// each row table's rows in a view of the frames.
type Bindings struct {
	fr     *jlm.Frames
	banks  [][3][3]unsafe.Pointer
	tables []rowTable
	bound  map[bindKey]*bound
	err    error
}

// rowTable is a row table's view of the frames, by its role, its rows
// rowBytes apart.
type rowTable struct {
	role     jlm.Role
	view     *jlm.RowView
	rowBytes int
}

// table is the row table of role r, nil where none is open.
func (b *Bindings) table(r jlm.Role) *rowTable {
	for i := range b.tables {
		if b.tables[i].role == r {
			return &b.tables[i]
		}
	}
	return nil
}

// NewBindings binds forms to weights read into fr (nil for weights from no
// file).
func NewBindings(fr *jlm.Frames) Bindings { return Bindings{fr: fr} }

// bindKey names a form's bound weights: the form, the block whose weights
// it reads (negative for none) and the tensors they are read in.
type bindKey struct {
	f  *Form
	li int
	t  Tensors
}

// bound is a form's weights where its call reads them, at the frame they
// were resolved at, and the weight its block does not hold where one is
// missing.
type bound struct {
	frame unsafe.Pointer
	addrs []unsafe.Pointer
	err   error
}

// why is the bound weights' decline: a form binding a weight its block does
// not hold is not formed for it (Bindings.Err).
func (w *bound) why() nn.Decline {
	if w.err != nil {
		return nn.DeclineNotFormed
	}
	return nn.DeclineServes
}

// Err is the weight the last form bound without it was missing, nil where
// every form found its weights.
func (b *Bindings) Err() error { return b.err }

// Bind is form f's weights for block li, read in t, where its call reads
// them: resolved and checked once for the block's frame, and again only
// when the frame has moved -- a page-in after an eviction binds the block's
// tensors elsewhere -- never looked up by role on every call. A tensor no
// page holds is resolved once.
func (b *Bindings) Bind(f *Form, li int, t Tensors) ([]unsafe.Pointer, nn.Decline) {
	var frame unsafe.Pointer
	if li >= 0 && b.fr != nil {
		frame = unsafe.Pointer(unsafe.SliceData(b.fr.Page(li)))
	}
	k := bindKey{f, li, t}
	w := b.bound[k]
	if w != nil && w.frame == frame {
		return w.addrs, w.why()
	}
	if w == nil {
		if b.bound == nil {
			b.bound = map[bindKey]*bound{}
		}
		w = &bound{}
		b.bound[k] = w
	}
	w.frame, w.addrs = frame, w.addrs[:0]
	if w.err = b.Check(f, li, t); w.err == nil {
		for _, x := range f.Weights {
			w.addrs = append(w.addrs, b.at(li, t, x))
		}
	} else {
		b.err = w.err
	}
	return w.addrs, w.why()
}

// Check says every weight f binds for block li is one t holds whole, or
// names the one it does not.
func (b *Bindings) Check(f *Form, li int, t Tensors) error {
	for _, x := range f.Weights {
		switch {
		case x.Vector:
			if len(t.Floats(x.Role)) == 0 {
				return fmt.Errorf("block %d: its form binds the vector %v, which the block does not hold", li, x.Role)
			}
		case jlm.ExpertBank(x.Role):
			if li < 0 || li >= len(b.banks) || b.banks[li][x.Role-jlm.RoleExpGateBank][x.Region] == nil {
				return fmt.Errorf("block %d: its form binds the bank %v, which has no view in the expert pages", li, x.Role)
			}
		case b.table(x.Role) != nil:
		default:
			data, p, _ := t.Bytes(x.Role)
			if p == nil && len(data) == 0 {
				if len(t.Floats(x.Role)) > 0 {
					continue
				}
				return fmt.Errorf("block %d: its form binds the matrix %v, which the block does not hold", li, x.Role)
			}
		}
	}
	return nil
}

// at is where block li's weight x is read: a vector's, a plane of a
// matrix's, a bank's in the expert pages, or a table's row in its view.
func (b *Bindings) at(li int, t Tensors, x Weight) unsafe.Pointer {
	switch {
	case x.Vector:
		return unsafe.Pointer(unsafe.SliceData(t.Floats(x.Role)))
	case jlm.ExpertBank(x.Role):
		return b.banks[li][x.Role-jlm.RoleExpGateBank][x.Region]
	}
	if rt := b.table(x.Role); rt != nil {
		return unsafe.Add(rt.view.Base(), x.From*rt.rowBytes)
	}
	data, p, _ := t.Bytes(x.Role)
	if p == nil && len(data) == 0 {
		// A matrix held as a vector (Check): its floats are its rows.
		return unsafe.Pointer(unsafe.SliceData(t.Floats(x.Role)))
	}
	if p == nil {
		return unsafe.Pointer(unsafe.SliceData(data))
	}
	return unsafe.Pointer(unsafe.SliceData([3][]byte{p.QS, p.D, p.SC}[x.Region]))
}

// BankViews makes block li's routed banks (its gate, up and down, nil for
// one it has none of) each one view of its experts in the frames' expert
// pages (jlm.Frames.BankView), n blocks held, or says one is not.
func (b *Bindings) BankViews(li, n int, banks [3]*jlm.Entry) bool {
	if len(b.banks) < n {
		b.banks = make([][3][3]unsafe.Pointer, n)
	}
	for i, e := range banks {
		if e == nil {
			continue
		}
		planes, _, ok := b.fr.BankView(e)
		if !ok {
			return false
		}
		b.banks[li][i] = planes
	}
	return true
}

// OpenTable reads the row table e's rows, rowBytes apart, through a view
// of the frames (jlm.Frames.NewRowView), opened once per role.
func (b *Bindings) OpenTable(e *jlm.Entry, rowBytes int) error {
	if b.table(e.Role) != nil {
		return nil
	}
	v, err := b.fr.NewRowView(e)
	if err != nil {
		return err
	}
	b.tables = append(b.tables, rowTable{role: e.Role, view: v, rowBytes: rowBytes})
	return nil
}

// Rows makes the rows own names of the row table of role r resident in its
// view.
func (b *Bindings) Rows(r jlm.Role, own []uint32) error {
	t := b.table(r)
	if t == nil {
		return fmt.Errorf("the row table %v has no view in the frames", r)
	}
	return t.view.Ensure(own)
}

// DropRows lets every row table's rows go.
func (b *Bindings) DropRows() {
	for i := range b.tables {
		b.tables[i].view.Drop()
	}
}

// Close returns the row tables' views, held off the heap.
func (b *Bindings) Close() {
	for i := range b.tables {
		b.tables[i].view.Close()
	}
	b.tables = nil
}
