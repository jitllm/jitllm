package cpudev

import (
	"slices"
	"unsafe"
)

// Bound is a block call's record: what its form's codes bind, every one an
// address or a span the caller made ready -- the block's weights bound at its
// frame, its residual rows and their stride, its rotary tables, its history's
// keys, values and page table, its session's recurrent state and window, the
// call's rows, positions and tokens.
type Bound struct {
	Li      int
	Weights []unsafe.Pointer
	Own     [][]uint32
	CS      []float32
	X       []float32
	// Stride is the elements from one residual stream of X to the next.
	Stride int
	Rows   []uint32
	N      int
	// Keys and Values are the call's history's, where it has one; Pages its
	// page table.
	Keys, Values unsafe.Pointer
	Pages        []uint32
	// Entries and EntryPages are its entries plane and that plane's page
	// table (the composer's history-1 tables), where the block keeps one.
	Entries    unsafe.Pointer
	EntryPages []uint32
	// CSSWA is the call's second rotary's table where the model trains one,
	// RopeRows and RopeMarks a rotary table's call's rows and its session's
	// marks.
	CSSWA               []float32
	RopeRows, RopeMarks []uint32
	// RecLi is the block whose recurrent states a linear block's step binds,
	// RecState and RecConv where its session holds them, RecRows one row's
	// descriptor (one row, the slot it reads and the slot it writes).
	RecLi             int
	RecState, RecConv unsafe.Pointer
	RecRows           [3]uint32
	// Rec, where set, is the call's run descriptor in place of RecRows; its
	// windows are written apart (Runner.ConvOut) and copied over the runs'
	// slots after the call.
	Rec []uint32
	// Side is the bound rows' per-layer inputs, a row each.
	Side []float32
	// IO is the buffers of a graph apart from its residual, its inputs' then
	// its outputs'.
	IO []unsafe.Pointer
	// Req is the step's request words, Pos its live rows then a row's
	// position, Tok its tokens.
	Req, Pos, Tok []uint32
	// Planes, where set, is the rotary planes a table's call reads in place
	// of the model's.
	Planes []float32
	// Pic is the step's picture, its decoded samples' words.
	Pic []uint32
	// Acc is a banded resize's accumulator, Acc1 a video pair's second
	// frame's; HAcc and HAcc1 their horizontal sums over column bands.
	Acc, Acc1   []uint32
	HAcc, HAcc1 []uint32
}

// Resources is what a block's call reads that the caller pages in during
// it: the routed experts its selection picks (count says the step counts
// them as used), held until ReleaseExperts, and the row table's rows its
// hash names, held until DropRows.
type Resources interface {
	Experts(li int, sel []int32, count bool) error
	ReleaseExperts()
	Rows(own []uint32) error
	DropRows()
}

// Runner runs a block's forms on the CPU device: a call record bound to a
// form's codes, the experts its selection picks made resident between the
// router's codes and the experts', the row table's rows between the hash's
// and the gathers'. It holds the call's words and scratch.
type Runner struct {
	dev *Device
	res Resources
	// planes is each rotary's planes as its composed table reads them.
	planes  [2][]float32
	nExpert int
	call    [4]uint32
	recRows [3]uint32
	zero    []uint32
	convOut []float32
	bind    Call
	// sel is a call's experts as the pager's fault path takes them, picked
	// marking those already in it, by expert.
	sel    []int32
	picked []bool
	stash  []uint32
	// FaultPicks gives a call's first row another row's picks, FaultExpert
	// makes another expert resident than the first the kernels read, and
	// FaultWeight weighs each row's first pick by zero: a gate's violations.
	FaultPicks, FaultExpert, FaultWeight bool
}

// NewRunner is a runner of forms on dev, paging through res, its rotary
// tables' planes planes, over a mixture of nExpert experts.
func NewRunner(dev *Device, res Resources, planes [2][]float32, nExpert int) *Runner {
	return &Runner{dev: dev, res: res, planes: planes, nExpert: nExpert, zero: make([]uint32, 4)}
}

// SetPlanes makes planes the rotary tables' planes the forms read.
func (r *Runner) SetPlanes(planes [2][]float32) { r.planes = planes }

// GrowZero gives the calls n zero words to bind.
func (r *Runner) GrowZero(n int) {
	if len(r.zero) < n {
		r.zero = make([]uint32, n)
	}
}

// ConvOut is the window rows a call of n window floats writes apart.
func (r *Runner) ConvOut(n int) []float32 {
	if len(r.convOut) < n {
		r.convOut = make([]float32, n)
	}
	return r.convOut
}

// CallOf is form f's codes' binding of b.
func (r *Runner) CallOf(b *Bound, f *Form) *Call {
	c := &r.bind
	c.Own, c.X, c.Stride, c.N, c.IO = b.Own, b.X, b.Stride, b.N, b.IO
	c.Rope = [2]unsafe.Pointer{ptr(b.CS), ptr(b.CSSWA)}
	c.Planes = [2]unsafe.Pointer{ptr(r.planes[0]), ptr(r.planes[1])}
	if b.Planes != nil {
		c.Planes = [2]unsafe.Pointer{ptr(b.Planes), ptr(b.Planes)}
	}
	r.call[0] = uint32(b.N)
	r.recRows = b.RecRows
	a := &c.At
	*a = [NumSrc]unsafe.Pointer{}
	a[SrcSide] = ptr(b.Side)
	a[SrcRopeRows], a[SrcRopeMarks] = ptr(b.RopeRows), ptr(b.RopeMarks)
	a[SrcKeys], a[SrcValues] = b.Keys, b.Values
	a[SrcPages], a[SrcRows], a[SrcCall] = ptr(b.Pages), ptr(b.Rows), unsafe.Pointer(&r.call[0])
	a[SrcEntries], a[SrcEntryPages] = b.Entries, ptr(b.EntryPages)
	if b.RecState != nil {
		a[SrcRecState], a[SrcRecConv] = b.RecState, b.RecConv
		a[SrcRecConvOut] = a[SrcRecConv]
	}
	a[SrcRecRows] = unsafe.Pointer(&r.recRows[0])
	if b.Rec != nil {
		a[SrcRecConvOut], a[SrcRecRows] = ptr(r.convOut), ptr(b.Rec)
	}
	a[SrcZero] = ptr(r.zero)
	a[SrcRequest], a[SrcTokens], a[SrcPos] = ptr(b.Req), ptr(b.Tok), ptr(b.Pos)
	a[SrcPicture] = ptr(b.Pic)
	a[SrcPicAcc] = ptr(b.Acc)
	a[SrcPicAcc1] = ptr(b.Acc1)
	a[SrcPicHAcc] = ptr(b.HAcc)
	a[SrcPicHAcc1] = ptr(b.HAcc1)
	c.Weights = b.Weights
	return c
}

// Run runs form f's codes over b, in order.
func (r *Runner) Run(b *Bound, f *Form) {
	c := r.CallOf(b, f)
	for i := range f.Codes {
		r.dev.Call(c, &f.Codes[i])
	}
}

// Form runs form f of block li over a decode step's b: its codes in order,
// for a form over a row table the table's rows its hash names read in, and
// for a form over banks the selection's experts made resident and counted.
func (r *Runner) Form(b *Bound, f *Form, li int) error {
	c := r.CallOf(b, f)
	lo := 0
	if f.RowSplit > 0 {
		r.dev.Calls(c, f.Codes[:f.RowSplit])
		defer r.res.DropRows()
		if err := r.res.Rows(b.Own[f.RowSel][:b.N*f.RowN]); err != nil {
			return err
		}
		lo = f.RowSplit
	}
	if f.Split <= 0 {
		r.dev.Calls(c, f.Codes[lo:])
		return nil
	}
	r.dev.Calls(c, f.Codes[lo:f.Split])
	err := r.experts(b, f, li, 1, true)
	defer r.res.ReleaseExperts()
	if err != nil {
		return err
	}
	r.dev.Calls(c, f.Codes[f.Split:])
	return nil
}

// Codes runs form f's codes [lo, hi) of block li over b's k rows, one at a
// time, the row table's rows and the selection's experts paged in where the
// form reaches them and handed back after.
func (r *Runner) Codes(b *Bound, f *Form, li, lo, hi, k int) error {
	c := r.CallOf(b, f)
	for i := lo; i < hi; i++ {
		if i == f.RowSplit && f.RowSplit > 0 {
			if err := r.res.Rows(b.Own[f.RowSel][:k*f.RowN]); err != nil {
				r.res.DropRows()
				return err
			}
		}
		if i == f.Split && f.Split > 0 {
			if err := r.experts(b, f, li, k, false); err != nil {
				r.res.ReleaseExperts()
				r.dropRows(f)
				return err
			}
		}
		if r.FaultWeight && containsInt(f.WReads, i) {
			// Each row's first pick weighted by zero where a kernel reads
			// the weights.
			w := f.SelN / f.Rows
			for row := range k {
				b.Own[f.WSel][row*w] = 0
			}
		}
		r.dev.Call(c, &f.Codes[i])
	}
	r.res.ReleaseExperts()
	r.dropRows(f)
	return nil
}

func (r *Runner) dropRows(f *Form) {
	if f.RowSplit > 0 {
		r.res.DropRows()
	}
}

// experts makes resident the experts n rows of b's selection pick, each once
// however many rows pick it -- the pager's fault path takes them so.
func (r *Runner) experts(b *Bound, f *Form, li, n int, count bool) error {
	k := f.SelN / max(f.Rows, 1)
	picks := b.Own[f.Sel][:n*k]
	ne := r.nExpert
	if r.FaultPicks && n > 1 {
		// The first row given another row's picks, with its own weights: the
		// first row whose picks differ, or the fault would change nothing.
		for j := 1; j < n; j++ {
			if !slices.Equal(picks[:k], picks[j*k:(j+1)*k]) {
				copy(picks[:k], picks[j*k:(j+1)*k])
				break
			}
		}
	}
	if len(r.picked) < ne {
		r.picked = make([]bool, ne)
	}
	r.sel = r.sel[:0]
	for _, x := range picks {
		if int(x) >= ne {
			// No page: the pager names it.
			r.sel = append(r.sel, int32(x))
		} else if !r.picked[x] {
			r.picked[x] = true
			r.sel = append(r.sel, int32(x))
		}
	}
	other := int32(-1)
	if r.FaultExpert {
		// Another expert's page made resident and held than the first the
		// kernels read: one no row picks, where there is one.
		e := r.sel[0]
		for j := 1; j < ne && other < 0; j++ {
			if x := (e + int32(j)) % int32(ne); !r.picked[x] {
				other = x
			}
		}
	}
	for _, x := range r.sel {
		if int(x) < ne {
			r.picked[x] = false
		}
	}
	switch {
	case other >= 0:
		r.sel[0] = other
	case r.FaultExpert && len(r.sel) > 1:
		// Every expert is picked: the first is left out of what is held
		// resident, so the kernels read its sheet where the pager holds
		// nothing of it, or holds it evicted.
		r.sel = r.sel[1:]
	}
	return r.res.Experts(li, r.sel, count)
}

// Stash saves (save) or loads the values crossing form f's last history
// write of call c's rows, own the call's, between a bidirectional run's two
// passes.
func (r *Runner) Stash(f *Form, own [][]uint32, c int, save bool) {
	at := 0
	for _, i := range f.Cross {
		at += len(own[i]) * c
	}
	for _, i := range f.Cross {
		n := len(own[i])
		if end := at + n; len(r.stash) < end {
			r.stash = append(r.stash, make([]uint32, end-len(r.stash))...)
		}
		if save {
			copy(r.stash[at:at+n], own[i])
		} else {
			copy(own[i], r.stash[at:at+n])
		}
		at += n
	}
}

func containsInt(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
