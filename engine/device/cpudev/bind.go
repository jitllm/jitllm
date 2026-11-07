package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// Bindings is where the CPU device's forms read their weights: each form's
// weights resolved once for the frame its block is in (Bind), each block's
// routed banks as one view of its experts in the frames' expert pages, and
// the row table's rows in a view of the frames.
type Bindings struct {
	fr       *jlm.Frames
	banks    [][3][3]unsafe.Pointer
	table    *jlm.RowView
	tableRow int
	bound    map[bindKey]*bound
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
// were resolved at, and why not where the CPU cannot read one.
type bound struct {
	frame unsafe.Pointer
	addrs []unsafe.Pointer
	why   nn.Decline
}

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
		return w.addrs, w.why
	}
	if w == nil {
		if b.bound == nil {
			b.bound = map[bindKey]*bound{}
		}
		w = &bound{}
		b.bound[k] = w
	}
	w.frame, w.addrs = frame, w.addrs[:0]
	if w.why = b.Check(f, li, t); w.why == nn.DeclineServes {
		for _, x := range f.Weights {
			w.addrs = append(w.addrs, b.at(li, t, x))
		}
	}
	return w.addrs, w.why
}

// Check says every weight f binds for block li is one t holds whole.
func (b *Bindings) Check(f *Form, li int, t Tensors) nn.Decline {
	for _, x := range f.Weights {
		switch {
		case x.Vector:
			if len(t.Floats(x.Role)) == 0 {
				return nn.DeclineNoWeight
			}
		case jlm.ExpertBank(x.Role):
			if li < 0 || li >= len(b.banks) || b.banks[li][x.Role-jlm.RoleExpGateBank][x.Region] == nil {
				return nn.DeclineNoWeight
			}
		case x.Role == jlm.RoleNGramTable:
			if b.table == nil {
				return nn.DeclineNoWeight
			}
		default:
			data, p, rows := t.Bytes(x.Role)
			if p == nil && len(data) == 0 {
				if len(t.Floats(x.Role)) > 0 {
					continue
				}
				return nn.DeclineNoWeight
			}
			if p != nil && (p.Row != 0 || p.Stride != 0 && p.Stride != rows || p.Sheet != nil) {
				return nn.DeclineCutWeight
			}
		}
	}
	return nn.DeclineServes
}

// at is where block li's weight x is read: a vector's, a plane of a
// matrix's, a bank's in the expert pages, or a table's row in its view.
func (b *Bindings) at(li int, t Tensors, x Weight) unsafe.Pointer {
	switch {
	case x.Vector:
		return unsafe.Pointer(unsafe.SliceData(t.Floats(x.Role)))
	case jlm.ExpertBank(x.Role):
		return b.banks[li][x.Role-jlm.RoleExpGateBank][x.Region]
	case x.Role == jlm.RoleNGramTable:
		return unsafe.Add(b.table.Base(), x.From*b.tableRow)
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
// of the frames (jlm.Frames.NewRowView), opened once.
func (b *Bindings) OpenTable(e *jlm.Entry, rowBytes int) error {
	if b.table != nil {
		return nil
	}
	v, err := b.fr.NewRowView(e)
	if err != nil {
		return err
	}
	b.table, b.tableRow = v, rowBytes
	return nil
}

// Rows makes the row table's rows own resident in its view.
func (b *Bindings) Rows(own []uint32) error { return b.table.Ensure(own) }

// DropRows lets the row table's rows go.
func (b *Bindings) DropRows() { b.table.Drop() }

// Close returns the row table's view, held off the heap.
func (b *Bindings) Close() {
	if b.table != nil {
		b.table.Close()
		b.table = nil
	}
}
