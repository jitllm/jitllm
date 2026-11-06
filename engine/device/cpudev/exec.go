package cpudev

import (
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
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
	cb.Graph, cb.SideOff = p.Source, p.SideOff
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
