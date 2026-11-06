package cpudev

import (
	"cmp"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/sched"

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
	// StreamFault is a gate's violation of the streams part: its expansion
	// swaps streams 1 and 2 (Streams.Fault), armed before the parts are placed.
	StreamFault bool
	// cpu is the device the executor runs on, seat the instance's decode seat
	// there and quota its page quota (a root instance's; nil otherwise).
	cpu   *CPU
	seat  *Seat
	quota *Quota
	// chunk pins a prompt chunk's rows (SetChunk), 0 where the executor's
	// rows decide.
	chunk int
	// kvconv is the history's conversions where it is binary16 or q8.
	kvconv *KVConv
	// onE and low are a prompt phase on the efficiency cores (BeginPrefill).
	onE bool
	low sched.Lowered
	// b binds the forms' weights and the rows' table.
	b        *Bindings
	bind     Bindings
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

// UseFrames makes the execution's weight bindings over fr, the frames its
// blocks' weights are read into.
func (e *Exec) UseFrames(fr *jlm.Frames) {
	e.bind = NewBindings(fr)
	e.b = &e.bind
}

// Bindings is the execution's weight bindings (UseFrames).
func (e *Exec) Bindings() *Bindings { return e.b }

// ExpertPager is where the experts a block's selection picks are made resident
// and handed back: the instance's lease of its pages.
type ExpertPager interface {
	Experts(li int, sel []int32, count bool) error
	ReleaseExperts()
}

// resources is the runner's: the experts ex pages in, and the row table's
// rows read through the execution's bindings.
type resources struct {
	ExpertPager
	b *Bindings
}

func (r resources) Rows(own []uint32) error { return r.b.Rows(own) }

func (r resources) DropRows() { r.b.DropRows() }

// Place readies the execution for a placement of n blocks, none of them
// composed: the launches made once (a placement again, for a new history
// format, keeps the forms' own buffers), the runner reading the experts ex
// pages in and the rotaries' planes.
func (e *Exec) Place(n int, ex ExpertPager, planes [2][]float32, nExpert int, prof bool) {
	res := resources{ExpertPager: ex, b: e.b}
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
func (e *Exec) Compose(li int, p nn.BlockSpec) {
	cb := &Block{Why: p.Why}
	e.blocks[li] = cb
	if !e.comp.Lowers() {
		cb.Why = nn.DeclineISA
	}
	if cb.Why != nn.DeclineServes {
		return
	}
	cb.Graph, cb.SideOff, cb.Bidir, cb.Whole, cb.Rec = p.Source, p.SideOff, p.Bidir, p.Whole, p.Rec
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
func (e *Exec) form(p nn.BlockSpec, rows int) (*Form, error) {
	gr := *p.Graph
	req := composer.Request{Formats: p.Formats, Strides: p.Strides,
		Step:     composer.Step{Page: p.Page, Keys: p.Keys, ActWindow: p.ActWindow, Seqs: p.Seqs},
		History:  composer.History{ValuesInKeys: p.ValuesInKeys},
		Rotaries: p.Rotaries}
	req.Graph, req.Step.Rows, req.Device = &gr, rows, e.comp.Device()
	return e.comp.Form(req, &gr, rows, false, false, p.F32)
}

// PrepLayer composes block li from the statement plan carries
// (nn.LayerPlan.Spec), as a card prepares a block it is offered, its rows'
// table opened where its rows read one; false where the block is not
// composed here.
func (e *Exec) PrepLayer(li int, p *nn.LayerPlan, _ *nn.LayerWeights) bool {
	if p.Spec == nil || !e.Holds(li) {
		return false
	}
	s := *p.Spec
	if s.Why == nn.DeclineServes && s.Table != nil && TableRowsOf(s.Graph) >= 0 {
		if err := e.b.OpenTable(s.Table, s.TableRow); err != nil {
			s.Why = e.comp.Decline(err)
		}
	}
	e.Compose(li, s)
	cb := e.blocks[li]
	return cb != nil && cb.Why == nn.DeclineServes
}

// Holds says li is one of the placement's blocks.
func (e *Exec) Holds(li int) bool { return li >= 0 && li < len(e.blocks) }

// ReleaseLayers lets the compositions of blocks [lo, hi) go: they are off
// the CPU.
func (e *Exec) ReleaseLayers(lo, hi int) {
	for li := lo; li < hi; li++ {
		if e.Holds(li) {
			e.blocks[li] = nil
		}
	}
}

// Sit seats the instance on c, the CPU device the executor runs on.
func (e *Exec) Sit(c *CPU) { e.cpu, e.seat = c, c.Sit() }

// Leave gives the instance's seat up.
func (e *Exec) Leave() {
	if e.cpu != nil {
		e.cpu.Leave(e.seat)
	}
}

// Touch records that the instance is decoding now.
func (e *Exec) Touch() { e.seat.Touch() }

// OthersDecoding reports whether another instance on the CPU device is
// decoding.
func (e *Exec) OthersDecoding() bool { return e.cpu.OthersDecoding(e.seat) }

// SetQuota makes q the instance's page quota on the device.
func (e *Exec) SetQuota(q *Quota) { e.quota = q }

// Quota is the instance's page quota on the device, nil for one the root's
// quota covers.
func (e *Exec) Quota() *Quota { return e.quota }

// DropQuota lets the instance's page quota go.
func (e *Exec) DropQuota() {
	if e.quota != nil {
		e.quota.Drop()
		e.quota = nil
	}
}

// SetChunk pins a prompt chunk's rows at n, 0 for the executor's own.
func (e *Exec) SetChunk(n int) { e.chunk = n }

// ChunkRows is how many prompt rows the CPU takes in one chunk
// (nn.ChunkLimiter): the pinned width, else the rows its forms are formed
// for, never past the widest chunk.
func (e *Exec) ChunkRows() int { return min(cmp.Or(e.chunk, RowsCap), nn.MaxHostPrefillChunk) }

// DefaultKV is the width the CPU holds a history at where the caller forces
// none: binary16, half the bytes, wherever it narrows in hardware, f32 where
// the composer forms the narrowing from its narrow passes
// (composer.NarrowsCPU).
func (e *Exec) DefaultKV() nn.KVFmt {
	if composer.NarrowsCPU(Target(e.comp.ISA(), 1)) {
		return nn.KVF32
	}
	return nn.KVF16
}

// ISA is the instruction set the CPU's forms are composed for.
func (e *Exec) ISA() cpu.ISA { return e.comp.ISA() }

// RunsFromFrames says the CPU runs a block from the instance's frames.
func (e *Exec) RunsFromFrames() bool { return true }

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

// AccountOn is what the device served the instance and formed for it, by
// its name (nn.Ledger).
func (e *Exec) AccountOn(dev string) (nn.Account, bool) {
	if dev != nn.CPU {
		return nn.Account{}, false
	}
	return nn.Account{Steps: e.steps, Forms: e.Composed(), Asks: e.comp.forms.Asks(), FormErr: e.comp.Err()}, true
}

// BeginPrefill starts a prompt as a phase of the device's workers, on its
// efficiency cores where e asks and it has them (nn.PrefillPhases).
func (e *Exec) BeginPrefill(onE bool) bool {
	if onE && e.workers.BeginPrefillE() {
		e.onE, e.low = true, e.workers.LowerToE()
		return true
	}
	e.workers.BeginPrefill()
	e.onE = false
	return false
}

// EndPrefill ends the prompt's phase (nn.PrefillPhases).
func (e *Exec) EndPrefill() {
	if e.onE {
		e.low.Raise()
		e.workers.EndPrefillE()
	} else {
		e.workers.EndPrefill()
	}
	e.onE = false
}

// Pool is the device's workers.
func (e *Exec) Pool() *cpu.Pool { return e.workers }

// Runner is the runner that calls the execution's forms.
func (e *Exec) Runner() *Runner { return e.run }

// Blocks is the runner of the execution's blocks' rows.
func (e *Exec) Blocks() *Blocks { return e.rows }

// Close returns the launches' own buffers.
func (e *Exec) Close() {
	if e.dev != nil {
		e.dev.Close()
	}
	e.bind.Close()
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
	// Skip is a part the model has none of: not composed, not asked.
	Skip bool
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
	if p.Skip {
		return Part{}
	}
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
// and token (Toks: it has one), its history (History, Seqs sequences wide),
// the tensors its weights are read in and the recurrent state it steps
// (RecLi's, its sequence's Slot). Faults lets a gate's row violations reach
// the call.
type Step struct {
	X                 []float32
	Stride            int
	CS, CSSWA, Side   []float32
	Pos               int
	Tok               int32
	Toks, Faults      bool
	History           History
	Seqs              int
	Tensors           Tensors
	RecLi, Slot       int
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
		KeyStart: hb.KeyStart, Pos: st.Pos, PagePos: hb.PagePos, Tok: st.Tok, Toks: st.Toks, RecLi: st.RecLi, Weights: w,
		Faults: st.Faults, Keys: hb.Keys, Values: hb.Values, RecState: st.RecState, RecConv: st.RecConv, Slot: st.Slot}
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
