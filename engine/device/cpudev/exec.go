package cpudev

import (
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// RowsCap is the rows the CPU's form of several rows is composed for: a
// step of more runs in calls of this many, one after another.
const RowsCap = 64

// Exec is one instance's execution on the CPU device: the launches over its
// workers, its composer, the forms of each of its blocks, the runner that
// calls them, and what it composed and served.
type Exec struct {
	workers  *cpu.Pool
	dev      *Device
	comp     *Composer
	run      *Runner
	rows     *Blocks
	blocks   []*Block
	composed []int
	steps    nn.Steps
	// fused is whether a fused run is open: a decode token's blocks one pool
	// region from the first to where the caller ends the run (Unfuse) or a
	// block's page has to be read. fuseOff runs a region an op (SetFusion),
	// fusedRuns counts the runs opened, tries and opened the engine's counts
	// across its instances (FuseCounts).
	fused, fuseOff bool
	fusedRuns      int64
	tries, opened  *atomic.Int64
}

// NewExec is an execution over workers, its forms composed by comp.
func NewExec(workers *cpu.Pool, comp *Composer) *Exec { return &Exec{workers: workers, comp: comp} }

// BlockPlan is what the CPU device composes a block from (Exec.Compose):
// the graph the block states, the graph its forms are composed from and
// the request stating their formats, strides, rotaries and history, or why
// the block is not to be asked.
type BlockPlan struct {
	Why     nn.Decline
	Source  *graph.Graph
	Graph   *graph.Graph
	SideOff int
	Req     composer.Request
	F32     bool
	// Bidir says the block attends bidirectionally inside a prompt's runs,
	// Whole that its rows read the whole picture whatever their windows.
	Bidir, Whole bool
}

// Place readies the execution for a placement of n blocks, none of them
// composed: the launches made once (a placement again, for a new history
// format, keeps the forms' own buffers), the runner reading res and the
// rotaries' planes.
func (e *Exec) Place(n int, res Resources, planes [2][]float32, nExpert int, prof bool) {
	e.blocks = make([]*Block, n)
	e.composed = make([]int, n)
	if e.dev == nil {
		e.dev = NewDevice(e.workers)
		e.dev.Prof = prof
	}
	if e.run == nil {
		e.run = NewRunner(e.dev, res, planes, nExpert)
	}
	if e.rows == nil {
		e.rows = NewBlocks(e.dev, e.run)
	}
	e.run.SetPlanes(planes)
}

// Compose composes block li from p: the form of one row and the rows'
// form, through the model's keyed cache, counted (Composed).
func (e *Exec) Compose(li int, p BlockPlan) {
	cb := &Block{Why: p.Why}
	e.blocks[li] = cb
	if cb.Why != nn.DeclineServes {
		return
	}
	cb.Graph, cb.SideOff, cb.Bidir, cb.Whole = p.Source, p.SideOff, p.Bidir, p.Whole
	f, err := e.form(p, 1)
	if err != nil {
		cb.Why = e.comp.Decline(err)
		return
	}
	cb.F = f
	if cb.Rows, err = e.form(p, RowsCap); err != nil {
		cb.RowsWhy = e.comp.Decline(err)
	}
	e.composed[li]++
	if cb.Rows != nil {
		e.rows.Reserve(RowsCap)
	}
}

// form is p's form for a step of rows.
func (e *Exec) form(p BlockPlan, rows int) (*Form, error) {
	gr := *p.Graph
	req := p.Req
	req.Graph, req.Step.Rows, req.Device = &gr, rows, e.comp.Device()
	return e.comp.Form(req, &gr, rows, false, false, p.F32)
}

// Holds says li is one of the placement's blocks.
func (e *Exec) Holds(li int) bool { return li >= 0 && li < len(e.blocks) }

// Drop lets block li's composition go: it is off the CPU.
func (e *Exec) Drop(li int) {
	if e.Holds(li) {
		e.blocks[li] = nil
	}
}

// Block is block li's composition, made at its placement, or why there is
// none.
func (e *Exec) Block(li int) (*Block, nn.Decline) {
	if !e.Holds(li) || e.blocks[li] == nil {
		return nil, nn.DeclineUnplaced
	}
	cb := e.blocks[li]
	return cb, cb.Why
}

// Composed is how many times each block was composed.
func (e *Exec) Composed() []int { return e.composed }

// Steps is what the device served the instance, part by part.
func (e *Exec) Steps() *nn.Steps { return &e.steps }

// Device is the execution's launches, nil before its first placement.
func (e *Exec) Device() *Device { return e.dev }

// Composer is the execution's composer.
func (e *Exec) Composer() *Composer { return e.comp }

// Runner is the runner that calls the execution's forms.
func (e *Exec) Runner() *Runner { return e.run }

// Blocks is the runner of the execution's blocks' rows.
func (e *Exec) Blocks() *Blocks { return e.rows }

// Close returns the launches' own buffers.
func (e *Exec) Close() {
	if e.dev != nil {
		e.dev.Close()
	}
}

// PartPlan is a part -- a graph apart from any block's residual: the head,
// a sampler, the input rows, a rotary table, the streams' mix, a readout --
// as the CPU device composes it: its weights' formats, the step's rows, the
// sequences and rotary marks a table reads and the activation window. None
// is why not, where the front end states no graph.
type PartPlan struct {
	Graph                        *graph.Graph
	Formats                      map[jlm.Role]jlm.Type
	Rows, Seqs, Marks, ActWindow int
	None                         nn.Decline
}

// FormGraph composes p's graph through the model's keyed cache, its inputs
// and outputs buffers of their own.
func (e *Exec) FormGraph(p PartPlan) (*Form, error) {
	req := composer.Request{Graph: p.Graph, Formats: p.Formats, Step: composer.Step{Rows: p.Rows, Page: 1, Keys: 1,
		Seqs: p.Seqs, Marks: p.Marks, ActWindow: p.ActWindow}, Device: e.comp.Device(), Apart: true}
	return e.comp.Form(req, p.Graph, p.Rows, true, false, false)
}

// FormPart is p composed as a part the device serves, or why not: the front
// end states no graph (None), the host's instructions lower no form, or the
// composer declines it.
func (e *Exec) FormPart(p PartPlan) Part {
	part := Part{Composed: true}
	switch {
	case p.Graph == nil:
		part.Why = p.None
	case !e.comp.Lowers():
		part.Why = nn.DeclineISA
	default:
		f, err := e.FormGraph(p)
		if err != nil {
			part.Why, part.Err = e.comp.Decline(err), err
			return part
		}
		part.F = f
	}
	return part
}

// Step is a block's call of one row on the CPU device: the residual row, its
// streams Stride apart, its rotary tables and per-layer inputs, its position
// and token, its history (History, Seqs sequences wide), the tensors its
// weights are read in and the recurrent state it steps (RecLi's).
type Step struct {
	X                 []float32
	Stride            int
	CS, CSSWA, Side   []float32
	Pos               int
	Tok               int32
	History           History
	Seqs              int
	Tensors           Tensors
	RecLi             int
	RecState, RecConv unsafe.Pointer
}

// Run runs block li's form of one row over st, its weights bound by b, or
// says why not: the block is not composed here, its history is not
// resident, or it reads a weight the device cannot. The row's history is
// committed once the row has run.
func (e *Exec) Run(li int, st *Step, b *Bindings) (nn.Decline, error) {
	cb, why := e.Block(li)
	if why != nn.DeclineServes {
		return why, nil
	}
	hb, why, err := st.History.Prepare(li, st.Pos, st.Seqs)
	if why != nn.DeclineServes || err != nil {
		return why, err
	}
	w, why := b.Bind(cb.F, li, st.Tensors)
	if why != nn.DeclineServes {
		return why, nil
	}
	r := Row{X: st.X, Stride: st.Stride, CS: st.CS, CSSWA: st.CSSWA, Side: st.Side, Pages: hb.Pages,
		KeyStart: hb.KeyStart, Pos: st.Pos, PagePos: hb.PagePos, Tok: st.Tok, Toks: true, RecLi: st.RecLi, Weights: w,
		Faults: true, Keys: hb.Keys, Values: hb.Values, RecState: st.RecState, RecConv: st.RecConv}
	if err := e.rows.Row(li, cb, &r); err != nil {
		return nn.DeclineServes, err
	}
	st.History.Commit(li, st.Pos)
	return nn.DeclineServes, nil
}

// SetFusion turns fused runs on or off.
func (e *Exec) SetFusion(on bool) { e.fuseOff = !on }

// FuseCounts counts every fused run tried and opened in tries and opened as
// well, the engine's counts across its instances.
func (e *Exec) FuseCounts(tries, opened *atomic.Int64) { e.tries, e.opened = tries, opened }

// Fuse opens a fused run where none is open and fusion is on.
func (e *Exec) Fuse() {
	if e.fused || e.fuseOff {
		return
	}
	if e.tries != nil {
		e.tries.Add(1)
	}
	if e.workers.Fuse() {
		e.fused = true
		e.fusedRuns++
		if e.opened != nil {
			e.opened.Add(1)
		}
	}
}

// Unfuse closes the open fused run, if there is one.
func (e *Exec) Unfuse() {
	if e.fused {
		e.fused = false
		e.workers.Unfuse()
	}
}

// FusedRuns counts the fused runs opened.
func (e *Exec) FusedRuns() int64 { return e.fusedRuns }
