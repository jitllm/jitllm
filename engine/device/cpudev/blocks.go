package cpudev

import (
	"unsafe"
)

// Row is one row of a block's call, as a decode step runs it: the row's
// residual streams (X, Stride apart) and its rotary tables, its per-layer
// inputs from the row's start, its history's page table, keys and values
// read from KeyStart and written at Pos, its token where Toks says there is
// one, its recurrent state and window and the slot it reads and writes,
// and the block's weights. Faults says the history-word violations apply.
type Row struct {
	X                 []float32
	Stride            int
	CS, CSSWA, Side   []float32
	Pages             []uint32
	Keys, Values      unsafe.Pointer
	KeyStart, Pos     int
	Tok               int32
	Toks              bool
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
	// FaultRow reads a row's history from one past where it starts, and
	// FaultWrite writes it one before its position: a gate's violations.
	FaultRow, FaultWrite bool
}

// NewBlocks is the rows of blocks on dev, run through run.
func NewBlocks(dev *Device, run *Runner) *Blocks { return &Blocks{dev: dev, run: run} }

// Row runs row r of block li's composed forms cb.
func (bl *Blocks) Row(li int, cb *Block, r *Row) error {
	b := &bl.bound
	*b = Bound{Li: li, Own: bl.dev.Own(cb.F), CS: r.CS, CSSWA: r.CSSWA, X: r.X, Stride: r.Stride, Rows: bl.rows[:], N: 1,
		Pages: r.Pages, Keys: r.Keys, Values: r.Values, RecLi: r.RecLi, RecState: r.RecState, RecConv: r.RecConv,
		Weights: r.Weights}
	if r.Side != nil {
		b.Side = r.Side[cb.SideOff:]
	}
	bl.pos1 = [2]uint32{1, uint32(r.Pos)}
	b.Pos = bl.pos1[:]
	if r.Toks {
		bl.tok1[0] = uint32(r.Tok)
		b.Tok = bl.tok1[:]
	}
	bl.rows = [4]uint32{0, uint32(r.KeyStart), uint32(r.Pos + 1), uint32(r.Pos)}
	if r.Faults {
		bl.faultRows()
	}
	b.RecRows = [3]uint32{1, uint32(r.Slot), uint32(r.Slot)}
	return bl.run.Form(b, cb.F, li)
}

func (bl *Blocks) faultRows() {
	t := &bl.rows
	if t[3] == 0 {
		return
	}
	switch {
	case bl.FaultRow:
		t[1]++
		if t[2] > t[1]+1 {
			t[2]--
		}
	case bl.FaultWrite:
		t[3]--
	}
}
