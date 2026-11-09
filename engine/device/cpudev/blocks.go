package cpudev

import (
	"github.com/jitllm/jitllm/graph"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
)

// Row is one row of a block's call, as a decode step runs it: the row's
// residual streams (X, Stride apart) and its rotary tables, its per-layer
// inputs from the row's start, its history's page table, keys and values
// read from KeyStart and written at Pos, its token where Toks says there is
// one, its recurrent state and window and the slot it reads and writes,
// and the block's weights. Faults says the history-word violations apply.
type Row struct {
	X      []float32
	Stride int
	// Cut is the row a block cut in two reads in place of the residual.
	Cut             []float32
	CS, CSSWA, Side []float32
	Pages           []uint32
	Keys, Values    unsafe.Pointer
	Entries         unsafe.Pointer
	EntryPages      []uint32
	KeyStart, Pos   int
	// PagePos is the positions a page of the history holds.
	PagePos int
	Tok     int32
	Toks    bool
	// Req is the step's request words (nn.RequestDevice), nil for none.
	Req               []uint32
	RecLi             int
	RecState, RecConv unsafe.Pointer
	Slot              int
	Weights           []unsafe.Pointer
	Faults            bool
}

// Blocks runs a block's rows on the CPU device through its runner, the
// row's words and record its own.
type Blocks struct {
	dev   *Device
	run   *Runner
	bound Bound
	// rows is a row's history descriptor: its live count's word, where its
	// keys start, one past its position, its position.
	rows [4]uint32
	tok1 [1]uint32
	pos1 [2]uint32
	// rowsTab, posTab and tokTab are a call's rows' spans, positions and
	// tokens at its form's rows; recDesc its run descriptor; xpad and
	// sidePad its residual rows and per-layer inputs where its units reach
	// past them; fpages a page table under the page violation.
	rowsTab, posTab, tokTab, recDesc []uint32
	// zeroReq is the request words a call binds where the step gives none
	// (nn.RequestDevice): its type 0 and every marker row 0.
	zeroReq               []uint32
	xpad, sidePad, cutPad []float32
	fpages                []uint32
	// FaultRow reads a row's history from one past where it starts,
	// FaultWrite writes it one before its position, FaultPage reads a call's
	// first page from another, and FaultKeysFirst runs a bidirectional run's
	// calls whole in one pass: a gate's violations.
	FaultRow, FaultWrite, FaultPage, FaultKeysFirst bool
}

// req is a call's request words: the step's, or zeros for rows rows where
// it gives none.
func (bl *Blocks) req(r []uint32, rows int) []uint32 {
	if r != nil {
		return r
	}
	bl.Reserve(rows)
	return bl.zeroReq
}

// NewBlocks is the rows of blocks on dev, run through run.
func NewBlocks(dev *Device, run *Runner) *Blocks { return &Blocks{dev: dev, run: run} }

// Row runs row r of block li's composed forms cb.
func (bl *Blocks) Row(li int, cb *Block, r *Row) error {
	b := &bl.bound
	*b = Bound{Li: li, Own: bl.dev.Own(cb.F), CS: r.CS, CSSWA: r.CSSWA, X: r.X, Stride: r.Stride, Rows: bl.rows[:], N: 1,
		Pages: r.Pages, Keys: r.Keys, Values: r.Values, RecLi: r.RecLi, RecState: r.RecState, RecConv: r.RecConv,
		Weights: r.Weights, Entries: r.Entries, EntryPages: r.EntryPages, Req: bl.req(r.Req, 1)}
	if r.Side != nil {
		b.Side = r.Side[cb.SideOff:]
	}
	if cb.Cut > 0 {
		cutIO(b, cb.Graph, r.Cut, b.Side, r.X)
	}
	bl.pos1 = [2]uint32{1, uint32(r.Pos)}
	b.Pos = bl.pos1[:]
	if r.Toks {
		bl.tok1[0] = uint32(r.Tok)
		b.Tok = bl.tok1[:]
	}
	bl.rows = [4]uint32{0, uint32(r.KeyStart), uint32(r.Pos + 1), uint32(r.Pos)}
	if r.Faults {
		faultRow(bl.rows[:], bl.FaultRow, bl.FaultWrite)
		b.Pages = bl.faultPages(r.Pages, r.PagePos, r.Pos+1)
	}
	b.RecRows = [3]uint32{1, uint32(r.Slot), uint32(r.Slot)}
	return bl.run.Form(b, cb.F, li)
}

// Rows is a call's rows of a block, as a chunk or a step runs them: their
// residual streams (X, Stride apart, Streams of them, each Width wide), their
// per-layer inputs (a row each, SideW wide, the block's slice SideOff into
// it), their rotary tables (NRot and NSWA a row), their history's page table
// (pages PagePos positions each), keys and values, the rows' spans (four
// words a row: its slot's first key row, where its keys start, one past
// where they end, its position), positions, tokens and sequences, a
// recurrent block's state and window and its session's windows (Conv, a
// slot's ConvLen floats), and the block's weights.
type Rows struct {
	X                      []float32
	Stride, Width, Streams int
	Side                   []float32
	SideW, SideOff         int
	CS                     []float32
	NRot                   int
	CSSWA                  []float32
	NSWA                   int
	Pages                  []uint32
	PagePos                int
	Keys, Values           unsafe.Pointer
	Entries                unsafe.Pointer
	EntryPages             []uint32
	Spans                  []uint32
	Pos                    []int
	Tok                    []int32
	Req                    []uint32
	Seq                    []int
	RecLi                  int
	RecState, RecConv      unsafe.Pointer
	Conv                   []float32
	ConvLen                int
	Weights                []unsafe.Pointer
	// Cut is the rows' cut rows, CutW wide, which a block cut in two reads
	// in place of the residual (Block.Cut); Graph is that block's graph.
	Cut   []float32
	CutW  int
	Graph *graph.Graph
}

// cutIO binds a block cut in two's buffers apart (Call.IO): its cut row,
// its side input where its graph reads one, and the residual it writes.
func cutIO(b *Bound, g *graph.Graph, cut, side, x []float32) {
	b.IO[0] = unsafe.Pointer(unsafe.SliceData(cut))
	if len(g.In) > 1 {
		b.IO[1] = unsafe.Pointer(unsafe.SliceData(side))
	}
	b.IO[len(g.In)] = unsafe.Pointer(unsafe.SliceData(x))
}

const (
	stashNone uint8 = iota
	stashSave
	stashLoad
)

// Reserve sizes the rows' descriptors for calls of up to rows rows.
func (bl *Blocks) Reserve(rows int) {
	if len(bl.rowsTab) < rows*4 {
		bl.rowsTab = make([]uint32, rows*4)
	}
	if len(bl.posTab) < 1+rows {
		bl.posTab = make([]uint32, 1+rows)
	}
	if len(bl.tokTab) < rows {
		bl.tokTab = make([]uint32, rows)
	}
	if len(bl.recDesc) < 1+7*rows {
		bl.recDesc = make([]uint32, 1+7*rows)
	}
	if len(bl.zeroReq) < composer.RequestRows+rows {
		bl.zeroReq = make([]uint32, composer.RequestRows+rows)
	}
	bl.run.GrowZero(rows * 4)
}

// Rows runs form f of block li over in's rows, a call of at most the form's
// rows at a time. A bidirectional run (runs) of more rows than a call holds
// runs every call's codes up to its last history write first, then every
// call's rest, the values crossing between them stashed.
func (bl *Blocks) Rows(li int, f *Form, in *Rows, runs bool) error {
	if runs && len(in.Pos) > f.Rows {
		keys := f.KeysEnd
		if bl.FaultKeysFirst {
			keys = 0
		}
		err := bl.calls(li, f, in, 0, keys, stashSave)
		if err == nil {
			err = bl.calls(li, f, in, keys, len(f.Codes), stashLoad)
		}
		return err
	}
	return bl.calls(li, f, in, 0, len(f.Codes), stashNone)
}

func (bl *Blocks) calls(li int, f *Form, in *Rows, lo, hi int, stash uint8) error {
	n := len(in.Pos)
	d, ns, pw := in.Width, in.Streams, in.SideW
	bl.Reserve(f.Rows)
	b := &bl.bound
	end := 0
	for r0 := 0; r0 < n; r0 += f.Rows {
		k := min(f.Rows, n-r0)
		for i := range k {
			t := bl.rowsTab[i*4 : i*4+4]
			copy(t, in.Spans[(r0+i)*4:(r0+i)*4+4])
			end = max(end, in.Pos[r0+i]+1)
			faultRow(t, bl.FaultRow, bl.FaultWrite)
		}
		bx, bstride := in.X[r0*d:], in.Stride
		var bside, bcut []float32
		if in.Side != nil {
			bside = in.Side[r0*pw:]
		}
		if in.Cut != nil {
			bcut = in.Cut[r0*in.CutW:]
		}
		staged := f.Unit > 1 && k%f.Unit != 0
		if staged {
			if len(bl.xpad) < f.Rows*d*ns {
				bl.xpad = make([]float32, f.Rows*d*ns)
			}
			for j := range ns {
				copy(bl.xpad[j*f.Rows*d:], bx[j*in.Stride:j*in.Stride+k*d])
			}
			bx, bstride = bl.xpad, f.Rows*d
			if in.Side != nil {
				if len(bl.sidePad) < f.Rows*pw {
					bl.sidePad = make([]float32, f.Rows*pw)
				}
				copy(bl.sidePad, bside[:k*pw])
				bside = bl.sidePad
			}
			if bcut != nil {
				if len(bl.cutPad) < f.Rows*in.CutW {
					bl.cutPad = make([]float32, f.Rows*in.CutW)
				}
				copy(bl.cutPad, bcut[:k*in.CutW])
				bcut = bl.cutPad
			}
		}
		if bside != nil {
			bside = bside[in.SideOff:]
		}
		*b = Bound{Li: li, Own: bl.dev.Own(f), CS: in.CS[r0*in.NRot:], X: bx, Stride: bstride, Rows: bl.rowsTab, N: k,
			Pages: bl.faultPages(in.Pages, in.PagePos, end), Side: bside, Weights: in.Weights, Keys: in.Keys, Values: in.Values,
			Entries: in.Entries, EntryPages: in.EntryPages, Req: bl.req(in.Req, f.Rows)}
		if in.CSSWA != nil {
			b.CSSWA = in.CSSWA[r0*in.NSWA:]
		}
		if bcut != nil {
			cutIO(b, in.Graph, bcut, bside, bx)
		}
		bl.posTab[0] = uint32(k)
		for i := range f.Rows {
			bl.posTab[1+i] = uint32(in.Pos[r0+min(i, k-1)])
		}
		b.Pos = bl.posTab
		if in.Tok != nil {
			for i := range f.Rows {
				bl.tokTab[i] = uint32(in.Tok[r0+min(i, k-1)])
			}
			b.Tok = bl.tokTab
		}
		if f.Rec {
			b.RecLi, b.Rec = in.RecLi, bl.recRuns(in.Seq[r0:r0+k])
			b.RecState, b.RecConv = in.RecState, in.RecConv
			bl.run.ConvOut(len(in.Conv))
		}
		if stash == stashLoad {
			bl.run.Stash(f, b.Own, r0/f.Rows, false)
		}
		if err := bl.run.Codes(b, f, li, lo, hi, k); err != nil {
			return err
		}
		if f.Rec {
			// The runs' windows over their slots.
			L := in.ConvLen
			seq := in.Seq[r0 : r0+k]
			for i, q := range seq {
				if !containsInt(seq[:i], q) {
					copy(in.Conv[q*L:(q+1)*L], bl.run.ConvOut(0)[q*L:(q+1)*L])
				}
			}
		}
		if stash == stashSave {
			bl.run.Stash(f, b.Own, r0/f.Rows, true)
		}
		if staged {
			for j := range ns {
				copy(in.X[j*in.Stride+r0*d:j*in.Stride+(r0+k)*d], bl.xpad[j*f.Rows*d:j*f.Rows*d+k*d])
			}
		}
	}
	return nil
}

// recRuns is a call's run descriptor over its rows' sequences seq: their
// count, each run's slots read and written, each run's first row and length
// in the run order, and each row's place in it, its run and its own row.
func (bl *Blocks) recRuns(seq []int) []uint32 {
	k := len(seq)
	if len(bl.recDesc) < 1+7*k {
		bl.recDesc = make([]uint32, 1+7*k)
	}
	w := bl.recDesc[:1+7*k]
	w[0] = uint32(k)
	at, runs := 0, 0
	for i, q := range seq {
		if containsInt(seq[:i], q) {
			continue
		}
		start := at
		for r := i; r < k; r++ {
			if seq[r] == q {
				w[1+4*k+at], w[1+5*k+r], w[1+6*k+r] = uint32(r), uint32(at), uint32(runs)
				at++
			}
		}
		w[1+2*runs], w[2+2*runs] = uint32(q), uint32(q)
		w[1+2*k+2*runs], w[2+2*k+2*runs] = uint32(start), uint32(at-start)
		runs++
	}
	for j := runs; j < k; j++ {
		copy(w[1+2*j:3+2*j], w[2*runs-1:2*runs+1])
		copy(w[1+2*k+2*j:3+2*k+2*j], w[2*k+2*runs-1:2*k+2*runs+1])
	}
	return w
}

// faultPages is the page table t of pages p positions each, the rows reaching
// end, under the page violation: its first page another's its rows reach.
func (bl *Blocks) faultPages(t []uint32, p, end int) []uint32 {
	n := len(t)
	last := (end - 1) / max(p, 1)
	if !bl.FaultPage || n < 2 || p == 0 || end <= p+p/4 || last >= n {
		return t
	}
	if len(bl.fpages) != n {
		bl.fpages = make([]uint32, n)
	}
	copy(bl.fpages, t)
	// Not a neighbouring or free slot: under sessions pooled in one arena
	// that is often another session's page of the same positions, or a freed
	// copy of this one's, whose keys move the step too little to see.
	bl.fpages[0] = t[last]
	return bl.fpages
}

// faultRow is a row's span t under the history-word violations: read from
// one past where it starts (row), or written one before its position
// (write).
func faultRow(t []uint32, row, write bool) {
	if t[3] == 0 {
		return
	}
	switch {
	case row:
		t[1]++
		if t[2] > t[1]+1 {
			t[2]--
		}
	case write:
		t[3]--
	}
}
