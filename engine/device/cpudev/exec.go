package cpudev

import (
	"cmp"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/sched"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
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
	// setup is what the execution was attached with; lanes its run of
	// blocks, made at its first placement.
	setup Setup
	lanes *Lanes
}

// NewExec is an execution over workers, its forms composed by comp.
func NewExec(workers *cpu.Pool, comp *Composer) *Exec { return &Exec{workers: workers, comp: comp} }

// useFrames makes the execution's weight bindings over fr, the frames its
// blocks' weights are read into.
func (e *Exec) useFrames(fr *jlm.Frames) {
	e.bind = NewBindings(fr)
	e.b = &e.bind
}

// Bindings is the execution's weight bindings (UseFrames).
func (e *Exec) Bindings() *Bindings { return e.b }

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
// pages in and the rotaries' planes, and its lanes over the same lease the
// first time.
func (e *Exec) Place(n int, ex Lease, planes [2][]float32, nExpert int, prof bool) {
	res := resources{ExpertPager: ex, b: e.b}
	if e.lanes == nil {
		st := e.setup
		e.lanes = NewLanes(e, e.b, ex, st.Width, st.Seqs, st.Streams, st.ConvLen, st.Trace)
	}
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
	rows := RowsCap
	if p.Rec && p.Bidir {
		// A recurrent block attending both ways reads its neighbours across
		// the run's rows (a centred convolution), which a call of fewer rows
		// than the run cuts: its rows run in one call, as many as a run
		// holds.
		rows = max(rows, p.Run)
	}
	if cb.Rows, err = e.form(p, rows); err != nil {
		cb.RowsWhy = e.comp.Decline(err)
	}
	e.composed[li]++
	if cb.Rows != nil {
		e.rows.Reserve(rows)
	}
}

// form is p's form for a step of rows.
func (e *Exec) form(p nn.BlockSpec, rows int) (*Form, error) {
	gr := *p.Graph
	req := composer.Request{Formats: p.Formats, Strides: p.Strides,
		Step: composer.Step{Page: p.Page, Keys: p.Keys, ActWindow: p.ActWindow, Seqs: p.Seqs},
		History: composer.History{ValuesInKeys: p.ValuesInKeys,
			Keys: composer.Pool{Q8: p.KVQ8}, Values: composer.Pool{Q8: p.KVQ8}},
		Rotaries: p.Rotaries, More: composer.MoreOf(&gr)}
	req.Graph, req.Step.Rows, req.Device = &gr, rows, e.comp.Device()
	return e.comp.Form(req, &gr, rows, false, false, p.F32)
}

// PrepLayer composes block li from the statement plan carries
// (nn.LayerPlan.Spec), as a card prepares a block it is offered, its rows'
// table opened where its rows read one; false where the block is not
// composed here.
func (e *Exec) PrepLayer(li int, p *nn.LayerPlan, _ *nn.LayerWeights) bool {
	defer e.comp.Place()()
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

// Lanes is the execution's run of blocks for the session it is bound to,
// nil before its first placement.
func (e *Exec) Lanes() *Lanes { return e.lanes }

// PrepErr is the first error the execution's composer met (nn.PrepReporter).
func (e *Exec) PrepErr() error { return e.comp.Err() }

// PagePool is the pool of history pages the model's closed sessions on the
// device gave back (nn.HistoryStorage).
func (e *Exec) PagePool() *PagePool { return e.cpu.KVPool(e.setup.Model) }

// ArenaFor is the model's arena for block li's pages of pp floats, made with
// slots slots on the first ask (nn.HistoryStorage).
func (e *Exec) ArenaFor(li, pp int, latent bool, slots int) (*Arena, error) {
	return e.cpu.Arenas(e.setup.Model).For(li, pp, latent, slots)
}

// NextSequence numbers the model's histories on the device, from 1
// (nn.HistoryStorage).
func (e *Exec) NextSequence() int64 { return e.cpu.NextSeq(e.setup.Model) }

// BindSession binds a session's view of the execution's lanes, prev's in
// place where it is one (nn.SessionBinder); false before the first
// placement.
func (e *Exec) BindSession(prev nn.LayerRunner, h nn.HistoryOf, p nn.Prompt) (nn.LayerRunner, bool) {
	if e.lanes == nil {
		return nil, false
	}
	v, _ := prev.(*SessionLanes)
	if v == nil {
		v = new(SessionLanes)
	}
	v.Bind(e.lanes, h, p)
	return v, true
}

// NotePrep records err as the execution's preparation error (nn.PrepNoter).
func (e *Exec) NotePrep(err error) { e.comp.Note(err) }

// SetSpinning sets whether the execution's idle workers spin (nn.Spinner).
func (e *Exec) SetSpinning(on bool) { e.workers.SetSpinning(on) }

// Workers is the execution's worker count (nn.Threaded).
func (e *Exec) Workers() int { return e.workers.Workers() }

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

// Setup is what an instance's execution on the CPU device is made from: the
// model it runs (whose forms on the device it composes through), the frames
// its weights are read into, whether its forms compute in float, whether a
// prefill beside a decode takes the efficiency cores (the serving split),
// whether its runs fuse, the rows a prompt step carries (0 for the
// device's), and where its fused runs are counted.
type Setup struct {
	Model       uint64
	Frames      *jlm.Frames
	Float       bool
	Split       bool
	NoFuse      bool
	Chunk       int
	Tries, Runs *atomic.Int64
	// Quota is the instance's page quota on the device, the execution's to
	// drop; nil for one another instance's covers.
	Quota *Quota
	// Width is the residual's width, Seqs the sequences its history is
	// strided by, Streams the residual streams and ConvLen a linear block's
	// convolution rows a sequence: what the execution's lanes run (Lanes).
	// Trace is called with each block's residual once it ran.
	Width, Seqs, Streams, ConvLen int
	Trace                         func(li int)
}

// Attach makes an instance's execution on c from st: its workers, its
// composer over the model's forms, its bindings over the frames, and its
// seat on c, given up when it closes.
func (c *CPU) Attach(st Setup) *Exec {
	workers := cpu.NewPool(append(c.PoolOptions(), cpu.WithServingSplit(st.Split))...)
	e := NewExec(workers, NewComposer(c.Forms(st.Model), c.Target.ISA(), workers.Workers(), st.Float))
	e.cpu, e.seat = c, c.Sit()
	e.useFrames(st.Frames)
	e.fuseOff, e.tries, e.opened = st.NoFuse, st.Tries, st.Runs
	e.chunk = st.Chunk
	e.quota = st.Quota
	e.setup = st
	return e
}

// leave gives the instance's seat up.
func (e *Exec) leave() {
	if e.cpu != nil {
		e.cpu.Leave(e.seat)
		e.cpu = nil
	}
}

// Decoding records that the instance is decoding now (nn.DecodeSharer).
func (e *Exec) Decoding() { e.seat.Touch() }

// OthersDecoding reports whether another instance on the CPU device is
// decoding (nn.DecodeSharer).
func (e *Exec) OthersDecoding() bool { return e.cpu != nil && e.cpu.OthersDecoding(e.seat) }

// SetPageQuota sets the instance's page quota, of which the device it
// attached holds held (nn.PageQuota); an execution with none has nothing to
// set.
func (e *Exec) SetPageQuota(bytes, held uint64) {
	if e.quota != nil {
		e.quota.Set(bytes, held)
	}
}

// dropQuota lets the instance's page quota go.
func (e *Exec) dropQuota() {
	if e.quota != nil {
		e.quota.Drop()
		e.quota = nil
	}
}

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

// PoolOf is the worker pool of the CPU execution among ds, nil where none
// runs blocks.
func PoolOf(ds []nn.BlockPlacer) *cpu.Pool {
	for _, d := range ds {
		if e, ok := d.(*Exec); ok {
			return e.workers
		}
	}
	return nil
}

// Runner is the runner that calls the execution's forms.
func (e *Exec) Runner() *Runner { return e.run }

// Blocks is the runner of the execution's blocks' rows.
func (e *Exec) Blocks() *Blocks { return e.rows }

// Close returns the launches' own buffers and lets go of everything the
// attachment made: the seat, the quota, the bindings' views and the
// workers. A second Close does nothing more.
func (e *Exec) Close() {
	e.leave()
	e.dropQuota()
	if e.dev != nil {
		e.dev.Close()
	}
	e.bind.Close()
	if e.workers != nil {
		e.workers.Close()
	}
}

// Detach lets the instance's execution go (nn.Session): its close.
func (e *Exec) Detach() { e.Close() }

// HeldBytes is what the execution holds of a session's history
// (nn.Session): none, the history being the session's own.
func (e *Exec) HeldBytes() uint64 { return 0 }

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
	Req               []uint32
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
		Faults: st.Faults, Keys: hb.Keys, Values: hb.Values, RecState: st.RecState, RecConv: st.RecConv, Slot: st.Slot,
		Entries: hb.Entries, EntryPages: hb.EntryPages, Req: st.Req}
	if err := e.rows.Row(li, cb, &r); err != nil {
		return nn.DeclineServes, err
	}
	st.History.Commit(li, st.Pos)
	return nn.DeclineServes, nil
}

// SetFusion turns fused runs on or off.
func (e *Exec) SetFusion(on bool) { e.fuseOff = !on }

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
