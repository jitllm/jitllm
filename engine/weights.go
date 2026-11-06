package engine

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// weightSource is where an instance's forms read their weights: the model's
// dense tensors, each block's copy bound at its frame, the head the
// instance projects through, the routed banks' planes in the expert pages
// and the row table's view -- and the bindings made of them (bind), each
// resolved once for a block's frame.
type weightSource struct {
	m       *model.Model
	dense   *model.Dense
	fr      *jlm.Frames
	outW    *model.Tensor
	outNorm []float32
	// banks is each block's routed banks' planes in the expert pages, by
	// bank (gate, up, down) and plane (jlm.File.BankView), set at the
	// block's composition.
	banks [][3][3]unsafe.Pointer
	// ngView is the n-gram table's rows where the forms read them.
	ngView *jlm.RowView
	bound  map[boundKey]*boundWeights
}

// boundKey names a form's bound weights: the form, the block whose
// weights it reads (-1 for the model's) and the copy of it they are read in.
type boundKey struct {
	f  *cpudev.Form
	li int
	l  *model.Layer
}

// boundWeights is a form's weights where its call reads them, at the
// frame they were resolved at, and why not where the CPU cannot read one.
type boundWeights struct {
	frame unsafe.Pointer
	addrs []unsafe.Pointer
	why   nn.Decline
}

// bind is form f's weights for block li (l its copy, nil for the
// model's tensors) where its call reads them: resolved and checked once for
// the block's frame, and again only when the frame has moved -- a page-in
// after an eviction binds the block's tensors elsewhere -- never looked up by
// role on every call. A tensor no page holds is resolved once.
func (w *weightSource) bind(f *cpudev.Form, li int, l *model.Layer) ([]unsafe.Pointer, nn.Decline) {
	var frame unsafe.Pointer
	if li >= 0 && w.fr != nil {
		frame = addr(w.fr.Page(li))
	}
	k := boundKey{f, li, l}
	b := w.bound[k]
	if b != nil && b.frame == frame {
		return b.addrs, b.why
	}
	if b == nil {
		if w.bound == nil {
			w.bound = map[boundKey]*boundWeights{}
		}
		b = &boundWeights{}
		w.bound[k] = b
	}
	b.frame, b.addrs = frame, b.addrs[:0]
	if b.why = w.check(f, li, l); b.why == nn.DeclineServes {
		for _, x := range f.Weights {
			b.addrs = append(b.addrs, w.at(li, l, x))
		}
	}
	return b.addrs, b.why
}

// check says every weight f binds is one block l holds whole.
func (w *weightSource) check(f *cpudev.Form, li int, l *model.Layer) nn.Decline {
	for _, x := range f.Weights {
		switch {
		case x.Vector:
			if len(w.vector(li, l, x.Role)) == 0 {
				return nn.DeclineNoWeight
			}
		case jlm.ExpertBank(x.Role):
			// A bank is read in the host's expert pages (cpuBanksPaged).
			if li >= len(w.banks) || w.banks[li][x.Role-jlm.RoleExpGateBank][x.Region] == nil {
				return nn.DeclineNoWeight
			}
		case x.Role == jlm.RoleNGramTable:
			// A row table is read in its view (cpuForm reads the rows in).
			if w.ngView == nil {
				return nn.DeclineNoWeight
			}
		default:
			t := w.matrix(li, l, x.Role)
			if t == nil || t.Packed == nil && len(t.Data) == 0 {
				return nn.DeclineNoWeight
			}
			if p := t.Packed; p != nil && (p.Row != 0 || p.Stride != 0 && p.Stride != t.Rows || p.Sheet != nil) {
				return nn.DeclineCutWeight
			}
		}
	}
	return nn.DeclineServes
}

// at is where block li's weight w is read: a vector's, a plane of
// a weight's, a bank's in the host's expert pages, or a table's row in its
// view.
func (w *weightSource) at(li int, l *model.Layer, x cpudev.Weight) unsafe.Pointer {
	switch {
	case x.Vector:
		return addr(w.vector(li, l, x.Role))
	case jlm.ExpertBank(x.Role):
		return w.banks[li][x.Role-jlm.RoleExpGateBank][x.Region]
	case x.Role == jlm.RoleNGramTable:
		return unsafe.Add(w.ngView.Base(), x.From*w.m.Ngram.RowBytes)
	}
	t := w.matrix(li, l, x.Role)
	if t.Packed == nil {
		return addr(t.Data)
	}
	return addr([3][]byte{t.Packed.QS, t.Packed.D, t.Packed.SC}[x.Region])
}

// matrix is block li's (l's) matrix of role r, nil for one it does
// not hold; with no block, the head's. An encoder segment's blocks hold
// theirs in the encoder (encRoleWeight).
func (w *weightSource) matrix(li int, l *model.Layer, r jlm.Role) *model.Tensor {
	if l == nil {
		switch m := w.dense; {
		case li == cpuInputRows && r == jlm.RoleTokenEmbd:
			return &m.Embd
		case li == cpuInputRows && r == jlm.RolePosEmbd:
			return &m.PosEmbd
		case r == jlm.RoleOutput || r == jlm.RoleTokenEmbd:
			return w.outW
		case r == jlm.RolePLETokEmbd && m.PleTok.E != nil:
			return &m.PleTok
		case r == jlm.RolePLEModelProj && m.PleProj.E != nil:
			return &m.PleProj
		case r == jlm.RoleEmbdDense1:
			return &m.Dense1
		case r == jlm.RoleEmbdDense2:
			return &m.Dense2
		}
		for _, v := range w.m.VecMats {
			if v.Role == r {
				return v.T
			}
		}
		return w.dense.Matrix(r)
	}
	if t := l.Matrix(r); t != nil {
		return t
	}
	// What the loader bound under no role of its own: a prediction block's
	// matrices and the convolutions' taps, cut from vectors.
	switch r {
	case jlm.RoleNextnEHProj:
		if l.Mtp == nil {
			return nil
		}
		return &l.Mtp.EhProj
	case jlm.RoleNextnEmbd:
		if l.Mtp == nil {
			return nil
		}
		return &l.Mtp.Embd
	case jlm.RoleSSMConv1d:
		return &l.SsmConvTaps
	}
	return nil
}

// vector is block li's (l's) vector of role r, nil for one it does
// not hold; with no block, the head's.
func (w *weightSource) vector(li int, l *model.Layer, r jlm.Role) []float32 {
	if l == nil {
		switch r {
		case jlm.RoleOutputNorm:
			return w.outNorm
		case jlm.RoleOutputNormBias:
			return w.m.OutNormB
		case jlm.RoleOutputBias:
			return w.m.OutB
		case jlm.RolePLEProjNorm:
			return w.m.PleNorm
		}
		return nn.VectorOf(w.m.Vecs, r)
	}
	if v, ok := l.NamedVector(r); ok {
		return v
	}
	return nn.VectorOf(l.Vecs, r)
}
