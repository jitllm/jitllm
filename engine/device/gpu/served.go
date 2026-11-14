package gpu

import (
	"errors"
	"fmt"
	"maps"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A part of a model the composer serves in the tier's own submission: a
// whole block, its graph of generic ops the model's front end built from the
// container (nn.LayerPlan.Graph), and the head. Each is asked of the
// composer through its one keyed cache, and what it forms launches on the
// tier's own buffers -- the block's weights and vectors by role, its rotary
// table and its paged history.

// composedPart is a part of the model as the composer is asked for it: its
// graph of generic ops, the formats of the weights it binds, and the tier's
// buffers it binds them to -- weights and vectors by role, the step's tables
// by the composer's name.
type composedPart struct {
	kind nn.Part
	// rotaries is the model's rotaries a text block's rope ops pick their
	// tables from (composer.Request.Rotaries), each bound by instance.
	rotaries []graph.Rotary
	// marks is the most rotary marks a sequence holds (composer.Step.Marks).
	marks int
	// part is what the composer is asked to form it as: a block's (the
	// zero), a table's or a sampling's.
	part    composer.PartKind
	graph   *graph.Graph
	formats map[jlm.Role]jlm.Type
	weights map[jlm.Role]*resident
	// ngram is the n-gram table's rows on the card, each range a gather
	// binds the table from its own (ngramRows).
	ngram   *ngramRows
	vectors map[jlm.Role]backend.Buf
	tables  map[composer.TableKey]backend.Buf
	// history, keys and page are a block's history layout and its step's
	// key bound and page; more its other histories' (its entries').
	history    composer.History
	more       []composer.History
	keys, page int
	// context is the positions the whole history holds, for a selecting form
	// over passes (composer.Step.Context), 0 elsewhere.
	context int
	// seqs is the sequences a step's rows are of (composer.Step.Seqs): 1 for a
	// form of one sequence's rows in position order, 0 for a row each.
	seqs int
	// paged is each bank a streamed block holds in pages and the sheets it
	// holds of it (layer.paged), nil for banks held whole.
	paged map[jlm.Role]int
	// apart says the graph's inputs and outputs are buffers of their own
	// (composer.Request.Apart).
	apart bool
}

// partForm is a part's kernels compiled on the device, in launch order, and
// the plan of what each binds: the caller's inputs and outputs, the part's
// weights, vectors and tables, and the form's own buffers -- the values
// between its kernels, its scratch and its constants -- which each lane
// holds its own of (partInst). One per composer Key on a device, every
// lane's.
type partForm struct {
	codes []partCode
	args  [][]partArg
	// own is each of the form's own buffers: its bytes, and the words a
	// constant is written with.
	own []partBuf
	// phys is the device buffers an instance holds, and slot the one each
	// own buffer is laid on (share).
	phys []partBuf
	slot []int
	// arenaAt is the buffer of a scratch's arena each passing buffer of
	// phys is laid on in a prompt instance, -1 for one the instance holds.
	arenaAt []int
	// width is each output's width in floats at the form's rows.
	width []int
	// key is the composer's Key the form is kept under, and refs the served
	// parts, of every lane, that launch it: the form goes with the last.
	key  composer.Key
	refs int
	// A form over banks in pages (composedPart.paged) runs in two halves
	// (pagedRun): its codes before split, up to the selection, then the
	// rest, which read the bank. sel is the own buffer holding the
	// selection, of selN words, and table the own buffer the step's table
	// "experts" is written to.
	split, sel, selN, table int
	// A form gathering the n-gram table's rows runs in two halves too
	// (rowsRun): its codes before rowSplit, up to the hash, then the
	// gathers. rowSel is the own buffer of the hash's rows, rowN of them a
	// row, rowIDs at the form's rows.
	rowSplit, rowSel, rowN, rowIDs int
	// A form over a history read in passes (composer.History.Passes) runs
	// each sweep's codes once a pass (streamRun): a selection's, then its
	// attention's.
	sweeps []passSweep
}

// passSweep is the codes [lo, hi) a form over passes runs once a pass, and
// the running table the last of them folds into (composer.TableRunOut, an
// attention's partial, or TableSelOut, a selection's block scores).
type passSweep struct {
	lo, hi int
	folds  composer.Table
}

// partBuf is one of a form's own buffers: n bytes, holding words where it
// is a constant.
type partBuf struct {
	n     int
	words []uint32
}

// partCode is one of a form's kernels and its launch shape: groups at the
// form's rows, and its geometry at a call of fewer (composer.Code.LiveGroups).
type partCode struct {
	k             backend.Kernel
	groups, width int
	geom          composer.Code
	// matrix is a kernel formed on the device's matrix unit: one the
	// matrix pass formed whose tiles it multiplies, not the kernel that
	// writes its rows' factor for it. attend is an attention formed there.
	matrix, attend bool
	// sweep is a code a form over passes runs once a pass (composer.Code.Sweep).
	sweep bool
}

// partInst is one lane's instance of a form: the form's own buffers and
// what they charge the device's budget.
type partInst struct {
	// bufs is the buffer each of the form's own buffers is, by index;
	// phys the device buffers they are laid on (partForm.share).
	bufs, phys []backend.Buf
	bytes      uint64
}

// partArg is what one kernel parameter binds: one of the form's own
// buffers, the caller's input or output i, or the block's weight region or
// vector of a role, or a step's table.
type partArg struct {
	kind   uint8
	i      int
	role   jlm.Role
	region int
	table  composer.TableKey
	// from is a weight's first row the binding names (the n-gram table's
	// range a gather reads).
	from int
	// read and writes are whether the code reads and writes it
	// (composer.Binding.Read, Writes): a binding is not always both.
	read, writes bool
}

// what names a: its kind, and its index, role or table.
func (a partArg) what() string {
	switch a.kind {
	case partOwn:
		return fmt.Sprintf("own buffer %d", a.i)
	case partIn:
		return fmt.Sprintf("input %d", a.i)
	case partOut:
		return fmt.Sprintf("output %d", a.i)
	case partWeight:
		return fmt.Sprintf("weight %v plane %d", a.role, a.region)
	case partVector:
		return fmt.Sprintf("vector %v", a.role)
	}
	return fmt.Sprintf("table %v", a.table)
}

const (
	partOwn uint8 = iota
	partIn
	partOut
	partWeight
	partVector
	partTable
)

// jlmType is a resident's format as the container names it: a packed type,
// or a float weight the device keeps verbatim, row-major, as the container
// does.
func jlmType(q layout.Quant) (jlm.Type, bool) {
	switch q {
	case layout.Float32:
		return jlm.TypeF32, true
	case layout.Float16:
		return jlm.TypeF16, true
	case layout.BFloat16:
		return jlm.TypeBF16, true
	case layout.IQ4NL:
		return jlm.TypeIQ4NL, true
	}
	for _, t := range quant.PackedTypes() {
		if k, ok := layout.QuantOf(t); ok && k == q {
			return jlm.TypeOf(t)
		}
	}
	return 0, false
}

// launchWords is the words a launch on the card costs, by its profile.
func (g *devTier) launchWords() int {
	if !g.launchSet && g.Profile != nil {
		if p, ok := g.Profile(g.dev); ok {
			g.launch = p.Device.LaunchWords()
		}
	}
	g.launchSet = true
	return g.launch
}

// units is the device's multiprocessors where its backend says, else 0.
func units(d backend.Device) int {
	if m, ok := d.(backend.MultiprocessorDevice); ok {
		return m.Multiprocessors()
	}
	return 0
}

// sharedPerUnit is the workgroup memory a multiprocessor holds where its
// backend says, else 0.
func sharedPerUnit(d backend.Device) int {
	if s, ok := d.(backend.SharedMemoryDevice); ok {
		return s.SharedPerMultiprocessor()
	}
	return 0
}

// threadRegs is a thread's registers at the device's most resident threads
// where its backend says, else 0.
func threadRegs(d backend.Device) int {
	if r, ok := d.(backend.RegisterDevice); ok {
		return r.ThreadRegisters()
	}
	return 0
}

// askPart asks r for sl at a step of rows: the request's Key, and what it
// forms.
func (g *devTier) askPart(r *composer.Registry, sl *composedPart, rows int) (composer.Key, []composer.Code, error) {
	req, err := g.partRequest(r, sl, rows)
	if err != nil {
		return 0, nil, err
	}
	if req.Device.Tile.Rows > 0 {
		g.Stats.MatrixTiled++
	}
	return r.Form(req)
}

// partRequest is what the composer is asked for part sl at rows rows on the
// device.
func (g *devTier) partRequest(r *composer.Registry, sl *composedPart, rows int) (composer.Request, error) {
	lanes, err := g.composedLanes()
	if err != nil {
		return composer.Request{}, err
	}
	g.Stats.ComposedLanes = lanes
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: g.dev.Slots(), Lanes: lanes, Units: units(g.dev),
		Regs: threadRegs(g.dev), ExactDiv: composer.ExactDivOn(g.dev.API()), ExactArith: composer.ExactArithOn(g.dev.API()),
		Matrix: g.composedMatrix(), HalfPairs: composer.HalfPairsOn(g.dev.API()),
		VectorLoads: composer.VectorLoadsOn(g.dev.API()), Shared: sharedPerUnit(g.dev), Launch: g.launchWords(),
		Buffers: backend.Bindings(g.dev)}
	if dev.Matrix[0] > 0 {
		dev.Fragment = backend.Fragment(g.dev)
		dev.ScaleRows = g.composedScale()
	}
	dev.Tile = g.matrixTile(r, dev, rows)
	return composer.Request{
		Graph:    sl.graph,
		Part:     sl.part,
		Formats:  sl.formats,
		Paged:    sl.paged,
		Apart:    sl.apart,
		Step:     composer.Step{Rows: rows, Page: max(sl.page, composedPageMin), Keys: max(sl.keys, 1), ActWindow: int(g.actWin.Load()), Seqs: sl.seqs, Marks: sl.marks, Context: sl.context},
		Rotaries: sl.rotaries,
		Device:   dev,
		History:  sl.history,
		More:     sl.more,
	}, nil
}

// composedMatrix is the binary16 tile the composer forms a step of rows on
// (composer.Device.Matrix), asked of the device once by compiling a probe
// (backend.LowersTile): none where it does not lower one staged in
// workgroup memory and carried across a loop. The float reference
// (Config.ComposedFloat) states none: the tile multiplies binary16
// operands, and the reference is float arithmetic, not the forms it holds
// to it -- a matrix-unit attention the reference shared would move both
// alike, wrong or right.
func (g *devTier) composedMatrix() [3]int {
	if g.ComposedFloat || g.NoMMA {
		return [3]int{}
	}
	s := backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32}
	if g.matrixCap == 0 {
		g.matrixCap = -1
		if ok, _ := backend.LowersTile(g.dev, s); ok {
			g.matrixCap = 1
		}
	}
	if g.matrixCap < 0 {
		return [3]int{}
	}
	return [3]int{s.M, s.N, s.K}
}

// composedScale reports whether the device lowers a row scale of the
// matrix tile's accumulator (composer.Device.ScaleRows), asked once.
func (g *devTier) composedScale() bool {
	if g.scaleCap == 0 {
		g.scaleCap = -1
		// Not on sm_70's quad-pair accumulators until their row scale is gated
		// alone: with it the float32 prompt gate's worst step on Qwen3-1.7B
		// read 0.00522 against its 0.00212 bound, without it 0.000621.
		if backend.TileQuadPairs(g.dev) {
			return false
		}
		if ok, _ := backend.LowersTile(g.dev, backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32, Scale: true}); ok {
			g.scaleCap = 1
		}
	}
	return g.scaleCap > 0
}

// composedLanes is the subgroup the composer forms for: the device's lanes,
// or Config.ComposedLanes where that narrows them to a power of two they
// hold whole.
func (g *devTier) composedLanes() (int, error) {
	own, n := g.pickLanes(), g.Config.ComposedLanes
	switch {
	case n == 0:
		return own, nil
	case n < 0 || n&(n-1) != 0 || n > own:
		return 0, fmt.Errorf("a composed subgroup of %d lanes on a device that guarantees %d: the forms need a power of two it holds whole", n, own)
	}
	return n, nil
}

// compilePart compiles what the composer formed for sl at a step of rows.
func (g *devTier) compilePart(sl *composedPart, codes []composer.Code, rows int) (*partForm, error) {
	gr := sl.graph
	f := &partForm{}
	io := map[graph.ValueID]partArg{}
	for i, v := range gr.In {
		io[v] = partArg{kind: partIn, i: i}
	}
	for i, v := range gr.Out {
		io[v] = partArg{kind: partOut, i: i}
		f.width = append(f.width, rows*gr.Values[v].Width)
	}
	vals := map[graph.ValueID]int{}
	own := map[composer.Operand]int{}
	buf := func(n int, words []uint32) int {
		f.own = append(f.own, partBuf{n: max(n, 4), words: words})
		return len(f.own) - 1
	}
	// A scratch binding no kernel reads or writes -- a sum a pass carried in
	// registers, still a parameter -- is bound to a word: at a prompt's rows
	// the sub-block dots alone are gigabytes. So is a value no kernel reads or
	// writes: one a pass recomputes where it is read (a softcapped head's
	// doubled sigmoid, a row of logits wide at every row).
	live := map[composer.Operand]bool{}
	liveVal := map[graph.ValueID]bool{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			if bd.Read || bd.Writes {
				live[bd.Operand] = true
				if bd.Operand.Kind == composer.OperandValue {
					liveVal[bd.Operand.Value] = true
				}
			}
		}
	}
	for _, c := range codes {
		if c.GPU == nil {
			return nil, fmt.Errorf("%s has no GPU code", c.Name)
		}
	}
	kerns, err := g.compileCodes(codes)
	if err != nil {
		return nil, err
	}
	for ci, c := range codes {
		geom := composer.Code{Groups: c.Groups, Live: c.Live, LiveUnits: c.LiveUnits, LiveInner: c.LiveInner,
			LiveGroup: c.LiveGroup, Work: c.Work, Few: c.Few, Many: c.Many}
		f.codes = append(f.codes, partCode{k: kerns[ci], groups: c.Groups, width: c.GPU.Group[0], geom: geom,
			matrix: slices.Contains(c.Passes, "matrix") && slices.ContainsFunc(c.GPU.Ops, func(o ir.Op) bool { return o.Kind == ir.OpTileMMA || o.Kind == ir.OpMMA }),
			attend: slices.Contains(c.Passes, "flash"), sweep: c.Sweep})
		var args []partArg
		for _, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == composer.OperandValue:
				if a, ok := io[o.Value]; ok {
					args = append(args, a)
					continue
				}
				// Sized by its rows and by the widest binding of it: a code
				// may reach past the rows' width (the CPU sizes it so too).
				i, ok := vals[o.Value]
				if !ok {
					n := 4 * rows * gr.Values[o.Value].Width
					if !liveVal[o.Value] {
						n = 4
					}
					i = buf(n, nil)
					vals[o.Value] = i
				}
				if liveVal[o.Value] {
					f.own[i].n = max(f.own[i].n, int(bd.Bytes))
				}
				args = append(args, partArg{kind: partOwn, i: i})
			case o.Kind == composer.OperandWeight && o.Region >= 0 && o.Region < 3:
				args = append(args, partArg{kind: partWeight, role: o.Role, region: o.Region, from: o.Instance})
			case o.Kind == composer.OperandVector:
				args = append(args, partArg{kind: partVector, role: o.Role})
			case o.Kind == composer.OperandTable && sl.tables[o.Key()] != nil:
				args = append(args, partArg{kind: partTable, table: o.Key()})
			case o.Kind == composer.OperandTable && o.Table == composer.TableExperts && sl.paged != nil:
				// The banks' page table, the form's own: written each step
				// before the codes that read it run (pagedRun).
				i, ok := own[o]
				if !ok {
					i = buf(4, nil)
					own[o] = i
				}
				f.own[i].n = max(f.own[i].n, int(bd.Bytes))
				args = append(args, partArg{kind: partOwn, i: i})
			case bd.Words != nil || o.Kind == composer.OperandScratch:
				// Sized by the widest binding of the operand: a later code
				// may reach further into it than the first.
				n := int(max(bd.Bytes, 4*int64(len(bd.Words))))
				if bd.Words == nil && !live[o] {
					n = 4
				}
				i, ok := own[o]
				if !ok {
					i = buf(n, bd.Words)
					own[o] = i
				}
				f.own[i].n = max(f.own[i].n, n)
				args = append(args, partArg{kind: partOwn, i: i})
			default:
				for _, k := range kerns[ci+1:] {
					k.Close()
				}
				f.close()
				return nil, fmt.Errorf("%s binds %+v, which the part's call site does not provide", c.Name, o)
			}
		}
		for j, bd := range c.Binds {
			args[j].read, args[j].writes = bd.Read, bd.Writes
		}
		f.args = append(f.args, args)
	}
	if sl.paged != nil {
		if err := f.pagedHalves(gr, vals, own, rows); err != nil {
			f.close()
			return nil, err
		}
	}
	if err := f.rowHalves(gr, vals, rows); err != nil {
		f.close()
		return nil, err
	}
	if sl.history.Passes {
		if err := f.passSweeps(); err != nil {
			f.close()
			return nil, err
		}
	}
	f.share(sl.paged != nil)
	return f, nil
}

// passSweeps finds the sweeps a form over a history read in passes runs,
// each its codes once a pass: a run of codes the composer marks a pass's
// (composer.Code.Sweep), reading the pass's keys and folding a running
// partial, an attention's or a selection's block scores. A code outside --
// a front fused with the row's own append, an entry pooled from the newest
// rows, a selection's ranking -- runs once. A pass writes none of the call's
// buffers, no table but the pools and the partial it folds, and no buffer a
// code of the pass reads before.
func (f *partForm) passSweeps() error {
	f.sweeps = f.sweeps[:0]
	for ci := 0; ci < len(f.codes); ci++ {
		if !f.codes[ci].sweep {
			continue
		}
		sw := passSweep{lo: ci}
		for ; ci < len(f.codes) && f.codes[ci].sweep; ci++ {
			for _, a := range f.args[ci] {
				if a.kind == partTable && a.writes && (a.table.Table == composer.TableRunOut || a.table.Table == composer.TableSelOut) {
					sw.folds = a.table.Table
				}
			}
		}
		sw.hi = ci
		if sw.folds == composer.TableNone {
			return fmt.Errorf("a form over a history read in passes with no pass between codes %d and %d", sw.lo, sw.hi)
		}
		f.sweeps = append(f.sweeps, sw)
	}
	if len(f.sweeps) == 0 {
		return fmt.Errorf("a form over a history read in passes with no pass between codes %d and %d", -1, -1)
	}
	for _, sw := range f.sweeps {
		read := map[int]bool{}
		for ci := sw.lo; ci < sw.hi; ci++ {
			for _, a := range f.args[ci] {
				if a.kind == partOwn && a.read {
					read[a.i] = true
				}
			}
			for _, a := range f.args[ci] {
				switch {
				case !a.writes:
				case a.kind == partIn || a.kind == partOut,
					a.kind == partTable && a.table.Table != sw.folds && a.table.Table != composer.TableKeys && a.table.Table != composer.TableValues,
					a.kind == partOwn && read[a.i]:
					return fmt.Errorf("the composer's kernel %d of a pass writes what a pass cannot run again over", ci)
				}
			}
		}
	}
	return nil
}

// share lays the form's own buffers out on fewer device buffers
// (partForm.slot, phys), as the tier's own scratch reuses its buffers
// block after block: a buffer whose first code writes it without reading
// it, live from that code to the last that binds it, takes a buffer one
// whose last code ran before its first gave up. The codes run in order on
// one stream, so the earlier holder is done with it. A constant, a buffer
// first read (it carries what an earlier call left), and a paged form's
// selection and page table, which the host reads and writes between its
// halves, keep buffers of their own. A binding that neither reads nor
// writes a buffer, a parameter its code never touches, is no use of it.
// Callers form f outside any step.
func (f *partForm) share(paged bool) {
	n := len(f.own)
	first, last := make([]int, n), make([]int, n)
	fresh := make([]bool, n)
	for i := range first {
		first[i] = -1
	}
	for ci, args := range f.args {
		for _, a := range args {
			if a.kind != partOwn || !a.read && !a.writes {
				continue
			}
			if first[a.i] < 0 {
				first[a.i], fresh[a.i] = ci, true
			}
			if first[a.i] == ci && (a.read || !a.writes) {
				fresh[a.i] = false
			}
			last[a.i] = ci
		}
	}
	// A pass runs again over what was written before it.
	for i := range last {
		for _, sw := range f.sweeps {
			if first[i] >= 0 && first[i] < sw.lo && last[i] >= sw.lo {
				last[i] = max(last[i], sw.hi-1)
			}
		}
	}
	f.slot = make([]int, n)
	f.phys = f.phys[:0]
	var free []int // the end code of each shared buffer, by its index in phys
	order := make([]int, 0, n)
	for i, b := range f.own {
		pinned := b.words != nil || !fresh[i] || paged && (i == f.sel || i == f.table)
		if pinned {
			f.slot[i] = len(f.phys)
			f.phys = append(f.phys, b)
			free = append(free, len(f.codes))
			continue
		}
		order = append(order, i)
	}
	slices.SortStableFunc(order, func(a, b int) int { return first[a] - first[b] })
	for _, i := range order {
		need, at := f.own[i].n, -1
		for p, end := range free {
			if f.phys[p].words != nil || end >= first[i] || end == len(f.codes) {
				continue
			}
			// The narrowest that holds it, else the widest, grown.
			switch {
			case at < 0:
				at = p
			case f.phys[p].n >= need && (f.phys[at].n < need || f.phys[p].n < f.phys[at].n):
				at = p
			case f.phys[at].n < need && f.phys[p].n > f.phys[at].n:
				at = p
			}
		}
		if at < 0 {
			at = len(f.phys)
			f.phys = append(f.phys, partBuf{})
			free = append(free, 0)
		}
		f.phys[at].n = max(f.phys[at].n, need)
		free[at] = last[i]
		f.slot[i] = at
	}
	// The passing buffers numbered widest first, so a scratch's arena of
	// every form's (partArena) holds each form's widest in one buffer.
	f.arenaAt = make([]int, len(f.phys))
	var pass []int
	for p, end := range free {
		f.arenaAt[p] = -1
		if end != len(f.codes) {
			pass = append(pass, p)
		}
	}
	slices.SortStableFunc(pass, func(a, b int) int { return f.phys[b].n - f.phys[a].n })
	for j, p := range pass {
		f.arenaAt[p] = j
	}
}

// partArena is a scratch's buffers the prompt instances of every form at
// its rows lay their passing buffers on (partForm.arenaAt): buffer j holds
// the widest j-th of any form's. A prompt chunk runs its blocks one after
// another on one stream, and a passing buffer holds nothing from one code
// run to the next call, so the forms share them as one form's buffers
// share them (share).
type partArena struct {
	bufs []backend.Buf
	n    []int
}

// grow widens the arena to hold form f's passing buffers, charging what it
// adds, and binds every instance of bs (in among them) whose buffers moved.
// Callers hold g.mu, outside any submission.
func (g *devTier) growArena(bs *blockScratch, f *partForm, in *partInst) error {
	if bs.arena == nil {
		bs.arena = &partArena{}
	}
	ar, moved := bs.arena, false
	for p, j := range f.arenaAt {
		if j < 0 {
			continue
		}
		for len(ar.bufs) <= j {
			ar.bufs, ar.n = append(ar.bufs, nil), append(ar.n, 0)
		}
		need := f.phys[p].n
		if ar.n[j] >= need {
			continue
		}
		d, err := g.dev.Alloc(need)
		if err != nil {
			return err
		}
		g.poisonBuf(d, need)
		if ar.bufs[j] != nil {
			ar.bufs[j].Free()
		}
		add := uint64(need - ar.n[j])
		ar.bufs[j], ar.n[j] = d, need
		g.charge(add)
		g.instBytes += add
		g.arenaBytes += add
		moved = true
	}
	bind := func(f *partForm, in *partInst) {
		for i, p := range f.slot {
			if j := f.arenaAt[p]; j >= 0 {
				in.bufs[i] = ar.bufs[j]
			}
		}
	}
	bind(f, in)
	if moved {
		for pf, pi := range bs.prompt {
			bind(pf, pi)
		}
	}
	return nil
}

// needs is what the arena must add to hold form f's passing buffers, each
// priced by foot (the card's footprint of a buffer of that many bytes).
func (ar *partArena) needs(f *partForm, foot func(int) uint64) uint64 {
	var n uint64
	for p, j := range f.arenaAt {
		if j < 0 {
			continue
		}
		have := 0
		if ar != nil && j < len(ar.n) {
			have = ar.n[j]
		}
		if want := f.phys[p].n; want > have {
			add := foot(want)
			if have > 0 {
				add -= min(add, foot(have))
			}
			n += add
		}
	}
	return n
}

// free frees the arena's buffers and reports what they charged.
func (ar *partArena) free() uint64 {
	if ar == nil {
		return 0
	}
	var n uint64
	for j, b := range ar.bufs {
		if b != nil {
			b.Free()
			n += uint64(ar.n[j])
		}
	}
	ar.bufs, ar.n = nil, nil
	return n
}

// tableKey is the step's table t, its one instance.
func tableKey(t composer.Table) composer.TableKey { return composer.TableKey{Table: t} }

// pagedHalves finds where a form over banks in pages is cut (partForm.split):
// at its first code that binds a bank, with the selection those codes read
// written by a code before it, and the page table read after it alone.
func (f *partForm) pagedHalves(gr *graph.Graph, vals map[graph.ValueID]int, own map[composer.Operand]int, rows int) error {
	f.split, f.sel, f.table = -1, -1, -1
	for _, o := range gr.Ops {
		if o.Kind == graph.OpExpertMatVec && len(o.In) == 2 && len(o.Weight) > 0 && jlm.ExpertBank(o.Weight[0].Role) {
			i, ok := vals[o.In[1]]
			switch {
			case !ok:
				return fmt.Errorf("the selection of a bank in pages is not one of the form's own values")
			case f.sel >= 0 && f.sel != i:
				return fmt.Errorf("the banks in pages read two selections")
			}
			f.sel, f.selN = i, rows*gr.Values[o.In[1]].Width
		}
	}
	for o, i := range own {
		if o.Kind == composer.OperandTable && o.Table == composer.TableExperts {
			f.table = i
		}
	}
	for ci, args := range f.args {
		for _, a := range args {
			if a.kind == partWeight && jlm.ExpertBank(a.role) || a.kind == partOwn && a.i == f.table {
				if f.split < 0 {
					f.split = ci
				}
			}
		}
	}
	if f.sel < 0 || f.table < 0 || f.split <= 0 {
		return fmt.Errorf("a form over banks in pages with no selection, table or code before its banks (%d, %d, %d)", f.sel, f.table, f.split)
	}
	written := false
	for _, args := range f.args[:f.split] {
		for _, a := range args {
			written = written || a.kind == partOwn && a.i == f.sel && a.writes
		}
	}
	if !written {
		return fmt.Errorf("the selection of a bank in pages is written by no code before the codes that read the bank")
	}
	return nil
}

// instance is a lane's instance of form f: its own buffers, poisoned where
// the tier's scratch is (WithPoisonScratch) so that a kernel reading what no
// kernel wrote shows, its constants written, and charged to the device's
// budget like a block's weights -- one the budget cannot hold is refused,
// a partRoomError. Outside a submission the room is made as for state
// (stateFits): the pools' free pages first, then a streamed block paged out.
// Callers hold g.mu.
func (g *devTier) instance(f *partForm) (*partInst, error) { return g.instanceOn(f, nil) }

// instanceOn is instance for a prompt instance of scratch bs, its passing
// buffers laid on bs's arena (partArena), or with bs nil every buffer its
// own.
func (g *devTier) instanceOn(f *partForm, bs *blockScratch) (*partInst, error) {
	// Admitted at what the card spends, the driver's page rounding included,
	// as a block's weights are: an instance admitted at its bytes put the card
	// past its budget by the rounding of its small buffers.
	var n, foot, grow uint64
	for p, b := range f.phys {
		if bs == nil || f.arenaAt[p] < 0 {
			n += uint64(b.n)
			foot += g.footprint(b.n)
		}
	}
	if bs != nil {
		grow = bs.arena.needs(f, g.footprint)
	}
	// Room alone counts a paged-in block as space; a streaming device keeps
	// the slots its streamed blocks page through (keepsSlot).
	if need := foot + grow; need > 0 && !(g.room(need) && g.keepsSlot(need)) && (g.inSub || g.prologueHeld || !g.stateFits(need, nil)) {
		return nil, partRoomError{need}
	}
	in := &partInst{}
	at := make([]backend.Buf, len(f.phys))
	for p, b := range f.phys {
		if bs != nil && f.arenaAt[p] >= 0 {
			continue
		}
		d, err := g.dev.Alloc(b.n)
		if err != nil {
			// A driver's refusal inside the budget is no room either.
			in.free()
			return nil, partRoomError{foot + grow}
		}
		g.poisonBuf(d, b.n)
		in.phys = append(in.phys, d)
		at[p] = d
		if b.words != nil {
			if err := d.Write(u32b(b.words)); err != nil {
				in.free()
				return nil, err
			}
		}
	}
	in.bufs = make([]backend.Buf, len(f.own))
	for i, p := range f.slot {
		in.bufs[i] = at[p]
	}
	if bs != nil {
		if err := g.growArena(bs, f, in); err != nil {
			in.free()
			return nil, partRoomError{foot + grow}
		}
	}
	in.bytes = n
	g.charge(n)
	g.instBytes += n
	return in, nil
}

// partRoomError is an instance the device's budget does not hold: a decline
// for room, which a move can answer, never a build failure.
type partRoomError struct{ need uint64 }

func (e partRoomError) Error() string {
	return fmt.Sprintf("its buffers need %d bytes, which the device's budget does not hold", e.need)
}

// launchArgs is an argument list for each of f's kernels, bound again every
// step.
func (f *partForm) launchArgs() [][]backend.Buf {
	out := make([][]backend.Buf, len(f.args))
	for i, args := range f.args {
		out[i] = make([]backend.Buf, len(args))
	}
	return out
}

// free frees an instance's buffers; what they charged is the caller's to
// give back.
func (in *partInst) free() {
	for _, b := range in.phys {
		b.Free()
	}
	in.phys, in.bufs = nil, nil
}

// launch binds the form's kernels to the caller's inputs and outputs, the
// lane's instance inst and the part's weights, vectors and tables, and
// launches them in order: how many of them were formed on the matrix unit.
func (sv *servedPart) launch(lc *launcher, inst *partInst, in, out []backend.Buf) (unitLaunches, error) {
	return sv.launchCodes(lc, inst, in, out, 0, len(sv.f.codes))
}

// unitLaunches is the launches of a form's kernels on the matrix unit: of
// its matrices, and of its attentions.
type unitLaunches struct{ matrix, attend int }

// launchCodes is launch of the form's codes [lo, hi).
func (sv *servedPart) launchCodes(lc *launcher, inst *partInst, in, out []backend.Buf, lo, hi int) (unitLaunches, error) {
	f, sl := sv.f, &sv.sl
	var matrix unitLaunches
	for i := lo; i < hi; i++ {
		fc, bufs := &f.codes[i], sv.args[i]
		if !fc.geom.RunsAt(sv.rows) {
			continue
		}
		for j, a := range f.args[i] {
			var b backend.Buf
			switch a.kind {
			case partOwn:
				b = inst.bufs[a.i]
			case partIn:
				b = in[a.i]
			case partOut:
				b = out[a.i]
			case partVector:
				b = sl.vectors[a.role]
			case partTable:
				b = sl.tables[a.table]
			case partWeight:
				r := sl.weights[a.role]
				if a.role == jlm.RoleNGramTable && sl.ngram != nil {
					r = sl.ngram.at(a.from)
				}
				if r != nil {
					b = [3]backend.Buf{r.qs, r.d, r.sc}[a.region]
				}
			}
			if b == nil {
				return matrix, fmt.Errorf("the composer's kernel %d (%s) reads %s, which the call site does not hold", i, fc.geom.Name, a.what())
			}
			bufs[j] = b
		}
		groups := fc.groups
		if sv.rows > 0 {
			groups = fc.geom.LiveGroups(sv.rows)
		}
		if err := lc.issue(fc.k, groups, fc.width, bufs...); err != nil {
			return matrix, err
		}
		if fc.matrix {
			matrix.matrix++
		}
		if fc.attend {
			matrix.attend++
		}
	}
	return matrix, nil
}

// compileCodes compiles each code's GPU kernel: side by side, on as many
// goroutines as the process runs threads, where the device's compiles run
// beside each other (backend.ConcurrentCompiler), else one after another. On
// an error it closes what it compiled.
func (g *devTier) compileCodes(codes []composer.Code) ([]backend.Kernel, error) {
	kerns := make([]backend.Kernel, len(codes))
	errs := make([]error, len(codes))
	workers := 1
	if c, ok := g.dev.(backend.ConcurrentCompiler); ok && c.ConcurrentCompile() {
		workers = min(runtime.GOMAXPROCS(0), len(codes))
	}
	if workers <= 1 {
		for i, c := range codes {
			if kerns[i], errs[i] = g.compile(c.GPU); errs[i] != nil {
				break
			}
		}
	} else {
		var wg sync.WaitGroup
		for w := range workers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := w; i < len(codes); i += workers {
					kerns[i], errs[i] = g.compile(codes[i].GPU)
				}
			}()
		}
		wg.Wait()
	}
	for i, err := range errs {
		if err != nil {
			for _, k := range kerns {
				if k != nil {
					k.Close()
				}
			}
			return nil, fmt.Errorf("compiling %s: %w", codes[i].Name, err)
		}
	}
	return kerns, nil
}

func (f *partForm) close() {
	for _, c := range f.codes {
		c.k.Close()
	}
	f.codes = nil
}

// clone is a copy of the part whose bindings, bound again every step, are
// its own.
func (sl composedPart) clone() composedPart {
	sl.weights, sl.vectors, sl.tables = maps.Clone(sl.weights), maps.Clone(sl.vectors), maps.Clone(sl.tables)
	return sl
}

// dropPart frees a lane's served part sv: its instance, whose bytes go back
// to the budget, and its form where no part launches it any more, with every
// lane's prompt instances of it. Callers hold g.mu, with no submission
// running.
func (g *devTier) dropPart(sv *servedPart) {
	if sv == nil {
		return
	}
	if in := sv.inst; in != nil {
		in.free()
		g.refund(in.bytes)
		g.instBytes -= in.bytes
		in.bytes = 0
	}
	if sv.f.refs--; sv.f.refs == 0 {
		// A form over passes keeps the decode scratch's instance in its
		// prompt map too (streamServed at one row); a split model's sets not
		// in use hold theirs (geoSets).
		g.eachLane(func(l *lane) {
			drop := func(bs *blockScratch, bbs map[int]*blockScratch) {
				if bs != nil {
					g.dropPromptInst(bs, sv.f)
				}
				for _, b := range bbs {
					g.dropPromptInst(b, sv.f)
				}
			}
			drop(l.bs, l.bbs)
			for _, s := range l.geoSets {
				drop(s.bs, s.bbs)
			}
		})
		delete(g.servedForms, sv.f.key)
		sv.f.close()
	}
}

// promptInst is the lane's instance of prompt form f at bs's rows, made
// where the scratch holds none, nil where the budget does not hold it.
// Callers hold g.mu.
func (g *devTier) promptInst(bs *blockScratch, f *partForm) *partInst {
	if in := bs.prompt[f]; in != nil {
		return in
	}
	in, err := g.instanceOn(f, bs)
	if err != nil {
		g.LastErr = fmt.Sprintf("the composed prompt block: %v (an instance at %d rows, over passes %v, at a placement %v; "+
			"the lane's decode scratch %v, %d instance(s) held there, clone lane %v)",
			err, bs.rows, len(f.sweeps) > 0, g.prewarming, bs == g.bs, len(bs.prompt), g.lane.clone)
		if _, room := err.(partRoomError); room {
			bs.instNoRoom = true
		}
		return nil
	}
	if bs.prompt == nil {
		bs.prompt = map[*partForm]*partInst{}
	}
	bs.prompt[f] = in
	g.placeGen.Add(1)
	return in
}

// dropPromptInst frees bs's instance of form f and gives its bytes back.
// Callers hold g.mu, with no submission running.
func (g *devTier) dropPromptInst(bs *blockScratch, f *partForm) {
	if in := bs.prompt[f]; in != nil {
		in.free()
		g.refund(in.bytes)
		g.instBytes -= in.bytes
		delete(bs.prompt, f)
		g.placeGen.Add(1)
	}
}

// partsClose frees every lane's served parts and every form. Callers hold
// g.mu.
func (g *devTier) partsClose() {
	g.eachLane(func(l *lane) { g.dropLaneParts(l) })
	for k, f := range g.servedForms {
		f.close()
		delete(g.servedForms, k)
	}
	if g.everyRows != nil {
		g.everyRows.Free()
		g.everyRows = nil
		g.refund(4)
	}
}

// everyRow is the device's word stating more rows than any form holds: a
// part bound to it runs its form's every row (composer.CallRows).
func (g *devTier) everyRow() (backend.Buf, error) {
	if g.everyRows != nil {
		return g.everyRows, nil
	}
	b, err := g.dev.Alloc(4)
	if err != nil {
		return nil, err
	}
	if err := b.Write([]byte{0xff, 0xff, 0xff, 0xff}); err != nil {
		b.Free()
		return nil, err
	}
	g.everyRows = b
	g.charge(4)
	return b, nil
}

// dropLaneParts frees lane l's served parts. Callers hold g.mu, with no
// submission running.
func (g *devTier) dropLaneParts(l *lane) {
	for sk, sv := range l.served {
		delete(l.served, sk)
		g.dropPart(sv)
	}
	clear(l.formErr)
}

// dropServed forgets the served parts of graph gr in every lane, a failed
// one's included, and frees each form no part launches any more: a block's
// parts go with the block, and a block placed again forms them again.
// Callers hold g.mu and drop the graph, so no recording launches what it
// frees.
func (g *devTier) dropServed(gr *graph.Graph) {
	g.eachLane(func(l *lane) {
		for sk, sv := range l.served {
			if sk.graph == gr {
				delete(l.served, sk)
				delete(l.formErr, sk)
				g.dropPart(sv)
			}
		}
	})
}

// cloneParts gives lane nl an instance of every part lane0 serves, on the
// device's forms: a clone's first step forms nothing. Callers hold g.mu.
func (g *devTier) cloneParts(nl *lane) error { return g.spreadParts(g.lane0, nl, nil) }

// spreadParts gives lane nl an instance of each part lane src serves that it
// does not, of graph gr where gr is not nil, on the device's forms: a prompt
// part's instance is its scratch's (promptInst). Callers hold g.mu.
func (g *devTier) spreadParts(src, nl *lane, gr *graph.Graph) error {
	for sk, sv := range src.served {
		if sv == nil || gr != nil && sk.graph != gr || nl.served[sk] != nil {
			continue
		}
		var in *partInst
		if sk.kind != nn.PartPrompt {
			var err error
			if in, err = g.instance(sv.f); err != nil {
				return fmt.Errorf("the composed %s: %w", sk.kind, err)
			}
		}
		sv.f.refs++
		if nl.served == nil {
			nl.served = map[servedKey]*servedPart{}
		}
		nl.served[sk] = &servedPart{sl: sv.formed.clone(), formed: sv.formed, f: sv.f, inst: in, args: sv.f.launchArgs(),
			in: make([]backend.Buf, sv.n), out: make([]backend.Buf, sv.m), n: sv.n, m: sv.m}
	}
	return nil
}

// partDecline is why a decode step's or a prompt chunk's composed part is
// not asked for: what the serving conditions leave to the tier's own
// kernels, counted by reason (Stats.Steps).
type partDecline = nn.Decline

const (
	partServes      = nn.DeclineServes
	partCaller      = nn.DeclineCaller
	partNoRoom      = nn.DeclineNoRoom
	partNotFormed   = nn.DeclineNotFormed
	numPartDeclines = nn.NumDeclines
)

// DeclineName is the reason a decline count is kept under, by its index in
// Stats.BlockDeclines or StepDeclines (nn.DeclineName).
func DeclineName(i int) string { return nn.DeclineName(i) }

// ServedPart is how many steps the composer served the parts of k in, and
// how many steps where they serve it was not asked for (Stats.Steps).
func (s Stats) ServedPart(k nn.Parts) (served, declined int) {
	served, why := s.Steps.Of(k)
	for _, n := range why {
		declined += n
	}
	return served, declined
}

// BlockDeclines is the block steps declined, a decode step's, a prompt
// chunk's and a picture's, by reason.
func (s Stats) BlockDeclines() [numPartDeclines]int {
	_, why := s.Steps.Of(nn.ComposedBlock | nn.ComposedPrompt | nn.ComposedTower)
	return why
}

// StepDeclines is the table, sample and input-row steps declined, by reason.
func (s Stats) StepDeclines() [numPartDeclines]int {
	_, why := s.Steps.Of(nn.ComposedTable | nn.ComposedSample | nn.ComposedEmbed | nn.PromptEmbed)
	return why
}

// FormedPart is how many of the parts of k formed at a step rather than
// at a placement (Stats.UnformedAtStepParts).
func (s Stats) FormedPart(k nn.Parts) int {
	n := 0
	for i := range s.UnformedAtStepParts {
		if k&(1<<i) != 0 {
			n += s.UnformedAtStepParts[i]
		}
	}
	return n
}

// AttendPart is how many launches of the parts of k ran an attention formed
// on the device's matrix unit (Stats.AttendLaunches).
func (s Stats) AttendPart(k nn.Parts) int {
	n := 0
	for i := range s.AttendLaunches {
		if k&(1<<i) != 0 {
			n += s.AttendLaunches[i]
		}
	}
	return n
}

// MatrixPart is how many launches of the parts of k ran a kernel formed on
// the device's matrix unit (Stats.MatrixLaunches).
func (s Stats) MatrixPart(k nn.Parts) int {
	n := 0
	for i := range s.MatrixLaunches {
		if k&(1<<i) != 0 {
			n += s.MatrixLaunches[i]
		}
	}
	return n
}

// ReplayedPart is how many steps a recording replayed the parts of k in
// (Stats.PartsReplayed).
func (s Stats) ReplayedPart(k nn.Parts) int {
	n := 0
	for i := range s.PartsReplayed {
		if k&(1<<i) != 0 {
			n += s.PartsReplayed[i]
		}
	}
	return n
}

// servedKey is a served part's shape: the part, its graph, the step's rows,
// a block's key bound, page and history precision, and the head weight's
// format. It names the part so a warm step builds no graph to key it; the
// graph is what the part computes, so the key names it, not a block index.
type servedKey struct {
	kind       nn.Part
	graph      *graph.Graph
	rows       int
	keys, page int
	half       bool
	latent     bool
	t          jlm.Type
	// held is the sheets a streamed block holds of its banks in pages, 0
	// where they are whole.
	held   int
	seqs   int
	stream bool
	// context is the whole history a selecting form over passes ranks.
	context int
}

// servedPart is a served part and its form, with the arguments its launch
// binds, reused every step: its own instance, or for a prompt part none,
// its instance its scratch's (promptInst).
type servedPart struct {
	sl composedPart
	// formed is sl as it formed: a lane's holder rebinds sl's tables every
	// step without g.mu, so a new lane's instance is cloned from this, which
	// nothing writes.
	formed  composedPart
	f       *partForm
	inst    *partInst
	args    [][]backend.Buf
	in, out []backend.Buf
	n, m    int
	// rows, where set, is the call's rows, the word its table states: a
	// launch runs their groups, not its form's every row's.
	rows int
}

// errServedCold is a served part asked for inside a capture before it
// formed: the capture is abandoned and the step issued unrecorded.
const errServedCold tierError = "tier: a served part forms outside a capture"

// errFormedEarlier marks a part whose form failed at an earlier step
// (servedFor), its error stated once.
const errFormedEarlier tierError = "did not form earlier"

// errFormedAhead marks a part a walk ahead asked the composer for and holds
// nothing of (formAhead).
const errFormedAhead tierError = "asked for ahead"

// formsAhead is a walk asking the composer for its parts' forms ahead of the
// walk that holds them: each on a goroutine of its own, as many at once as
// the process runs.
type formsAhead struct {
	wg   sync.WaitGroup
	slot chan struct{}
}

// form asks r for req's form on a goroutine of its own, once a slot frees.
// What it forms is r's cache's: the walk after finds it there, or its error.
func (a *formsAhead) form(r *composer.Registry, req composer.Request) {
	a.slot <- struct{}{}
	a.wg.Go(func() {
		defer func() { <-a.slot }()
		r.Form(req)
	})
}

// formAhead runs walk twice: first asking the composer for every part it
// meets side by side and holding nothing, then as itself, each form found
// made. A placement's parts form independently of one another, and forming
// them one at a time inside the walk was most of a card's start. Callers
// hold g.mu.
func (g *devTier) formAhead(walk func()) {
	n0 := g.composerRuns()
	a := &formsAhead{slot: make(chan struct{}, runtime.GOMAXPROCS(0))}
	g.ahead = a
	walk()
	g.ahead = nil
	a.wg.Wait()
	n1 := g.composerRuns()
	walk()
	g.FormedAhead += n1 - n0
	g.FormedInWalk += g.composerRuns() - n1
}

// composerRuns is the forms the device's composer made rather than found.
func (g *devTier) composerRuns() int {
	if g.composer == nil {
		return 0
	}
	return int(g.composer.Formed())
}

// servedFor is the lane's served part of key sk, built by build on its
// first step and formed through the composer's one keyed cache, its form the
// device's and its instance the lane's. A submission beside other sessions'
// takes the device's lock for the build (subLock); callers otherwise hold
// g.mu.
func (g *devTier) servedFor(sk servedKey, build func() (composedPart, error)) (*servedPart, error) {
	if sv, ok := g.served[sk]; ok {
		if sv == nil {
			// Asked again every step: the error stated once, at the failure.
			g.noteFormCause()
			return nil, g.formErr[sk]
		}
		return sv, nil
	}
	if g.capturing {
		return nil, errServedCold
	}
	if !g.prewarming {
		g.UnformedAtStep++
		g.UnformedAtStepParts[sk.kind]++
		g.LastErr = fmt.Sprintf("the composed %s at %d rows was not formed at placement (keys %d, seqs %d, stream %v, held %d, clone lane %v): a step composes nothing",
			sk.kind, sk.rows, sk.keys, sk.seqs, sk.stream, sk.held, g.lane.clone)
		return nil, errors.New(g.LastErr)
	}
	if g.served == nil {
		g.served = map[servedKey]*servedPart{}
	}
	g.served[sk] = nil
	fail := func(err error) error {
		// Room is a decline a later call may not meet: nothing is
		// remembered, and nothing failed to build.
		var room partRoomError
		if errors.As(err, &room) {
			delete(g.served, sk)
			g.LastErr = err.Error()
			return err
		}
		err = g.buildFailed(err)
		if g.formErr == nil {
			g.formErr = map[servedKey]error{}
		}
		g.formErr[sk] = fmt.Errorf("the composed %s %w: %w", sk.kind, errFormedEarlier, err)
		g.LastErr = err.Error()
		return err
	}
	defer g.subUnlock(g.subLock())
	if g.servedForms == nil {
		g.servedForms = map[composer.Key]*partForm{}
	}
	if g.composer == nil {
		g.composer = composer.Default()
		g.rewriteComposer(g.composer)
		// Poisoned scratch poisons the forms' workgroup arrays too: a kernel
		// reading one it never wrote changes the logits.
		if g.kb.poisonScratch {
			g.composer.PoisonWorkgroupMemory()
		}
		if g.kb.dropSegments {
			g.composer.DropSegments(composer.DeviceGPU)
		}
	}
	r := g.composer
	if g.ComposedFloat || g.ComposedUnsplit {
		if g.composerFloat == nil {
			g.composerFloat = g.composer
			if g.ComposedFloat {
				g.composerFloat = g.composerFloat.WithoutInt8(composer.DeviceGPU)
			}
			if g.ComposedUnsplit {
				g.composerFloat = g.composerFloat.WithoutSplits(composer.DeviceGPU)
			}
		}
		r = g.composerFloat
	}
	sl, err := build()
	if err != nil {
		if g.ahead != nil {
			delete(g.served, sk)
			return nil, errFormedAhead
		}
		return nil, fail(err)
	}
	// A pass's attention leaves its maximum, sum and weighted values for the
	// fold (composer.Registry.ForPasses).
	if sl.history.Passes {
		if r != g.composer {
			r = r.ForPasses(composer.DeviceGPU)
		} else {
			if g.composerPasses == nil {
				g.composerPasses = g.composer.ForPasses(composer.DeviceGPU)
			}
			r = g.composerPasses
		}
	}
	if g.ahead != nil {
		delete(g.served, sk)
		if req, err := g.partRequest(r, &sl, sk.rows); err == nil {
			g.ahead.form(r, req)
		}
		return nil, errFormedAhead
	}
	// A form of several rows reads the rows it runs from the call's word: a
	// part whose call states none runs its form's every row.
	if _, ok := sl.tables[tableKey(composer.TableCall)]; !ok {
		every, err := g.everyRow()
		if err != nil {
			return nil, fail(err)
		}
		if sl.tables == nil {
			sl.tables = map[composer.TableKey]backend.Buf{}
		}
		sl.tables[tableKey(composer.TableCall)] = every
	}
	t0 := time.Now()
	key, codes, err := g.askPart(r, &sl, sk.rows)
	if err != nil {
		return nil, fail(fmt.Errorf("the composed %s: %w", sk.kind, err))
	}
	g.FormAsks++
	f, ok := g.servedForms[key]
	if !ok {
		if f, err = g.compilePart(&sl, codes, sk.rows); err != nil {
			return nil, fail(fmt.Errorf("the composed %s: %w", sk.kind, err))
		}
		g.PartsFormed++
		g.PartKernels += len(codes)
		f.key = key
		g.servedForms[key] = f
	}
	g.FormTook += time.Since(t0)
	var in *partInst
	// A form over a history read in passes is laid on its scratch's arena
	// (streamPart).
	if sk.kind != nn.PartPrompt && !sk.stream {
		if in, err = g.instance(f); err != nil {
			if f.refs == 0 {
				delete(g.servedForms, key)
				f.close()
			}
			err = fmt.Errorf("the composed %s: %w", sk.kind, err)
			// Room is not remembered: the part forms at the next ask, once
			// a move or a page given back has made it.
			if errors.As(err, new(partRoomError)) {
				delete(g.served, sk)
				g.LastErr = err.Error()
				return nil, err
			}
			return nil, fail(err)
		}
	}
	f.refs++
	n, m := len(sl.graph.In), len(sl.graph.Out)
	sv := &servedPart{sl: sl, formed: sl.clone(), f: f, inst: in, args: f.launchArgs(), in: make([]backend.Buf, n), out: make([]backend.Buf, m), n: n, m: m}
	g.served[sk] = sv
	if sk.kind == nn.PartBlock && !sk.stream && sk.graph.Block >= 0 {
		if b := sk.graph.Block; b >= len(g.blockForms) {
			g.blockForms = append(g.blockForms, make([]int, b+1-len(g.blockForms))...)
		}
		g.blockForms[sk.graph.Block]++
	}
	return sv, nil
}

// run launches the served part on in and out, the block's own buffers, in
// instance inst.
func (sv *servedPart) run(g *devTier, lc *launcher, inst *partInst, in, out []backend.Buf) error {
	return sv.runAs(g, lc, inst, sv.sl.kind, in, out)
}

// runAs is run, counted as a step of part kind: a prompt's one-row call
// launches a decode step's form and is the prompt's.
func (sv *servedPart) runAs(g *devTier, lc *launcher, inst *partInst, kind nn.Part, in, out []backend.Buf) error {
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	matrix, err := sv.launch(lc, inst, sv.in[:sv.n], sv.out[:sv.m])
	if err != nil {
		return err
	}
	g.Steps.Serve(kind)
	g.MatrixLaunches[kind] += matrix.matrix
	g.AttendLaunches[kind] += matrix.attend
	return nil
}

// pagedRun is runAs for a form over banks in pages (partForm.split): its
// codes up to the selection, the selection read back and its experts made
// resident in block l's pages on the card by the tier's own fill -- each
// faulted in from the host's pager where the card does not hold it, a
// cache's least recently used slot evicted for it -- the page table
// written, each expert's slot, then the codes that read the banks. uses is
// the session's counts of the block's experts (streamBank.count).
func (sv *servedPart) pagedRun(g *devTier, lc *launcher, s backend.Session, inst *partInst, kind nn.Part, l *layer, nExpert int,
	uses []uint32, in, out []backend.Buf) error {
	f, st := sv.f, l.stream
	if st == nil || len(st.sel) == 0 || f.selN%len(st.sel) != 0 {
		return fmt.Errorf("a form over banks in pages selects %d experts where the block's pages are filled for its selection", f.selN)
	}
	// A call of several rows reads every live row's selection, each expert
	// filled once into the cache: pagedCall holds the call to the rows its
	// slots hold.
	k, live := len(st.sel), len(st.sel)
	if f.selN > k {
		live = min(f.selN, max(sv.rows, 1)*k)
	}
	if live > k {
		if !st.cached() || live > len(st.owner) {
			return fmt.Errorf("%d rows select %d experts, more than the block's %d pages hold", live/k, live, max(len(st.owner), k))
		}
		one := st.sel
		st.rsel = growU32(st.rsel, live)
		st.sel = st.rsel
		defer func() { st.sel = one }()
	}
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	m0, err := sv.launchCodes(lc, inst, sv.in[:sv.n], sv.out[:sv.m], 0, f.split)
	if err != nil {
		return err
	}
	tw := time.Now()
	if err := s.Sync(); err != nil {
		return err
	}
	g.TStreamWait += time.Since(tw)
	if err := s.Read(inst.bufs[f.sel], u32b(st.sel)); err != nil {
		return err
	}
	st.count(uses)
	one := g.kb.pageFault == PageFaultSwap && len(st.sel) == 1 && nExpert > 1
	var picked uint32
	if one {
		picked, st.sel[0] = st.sel[0], (st.sel[0]+1)%uint32(nExpert)
	}
	tf := time.Now()
	switch {
	case g.kb.pageFault != PageFaultStale:
		err = st.fill(g, s, l, nExpert)
	case st.cached():
		st.items = st.items[:0]
		st.lookup(g)
	}
	if err != nil && st.cached() {
		st.forget()
	}
	g.fillDone(time.Since(tf))
	if err != nil {
		return err
	}
	g.StreamFills++
	if live > k {
		g.StreamRowFills++
	}
	st.tab = growU32(st.tab, nExpert)
	if st.cached() {
		for e, slot := range st.slotOf {
			st.tab[e] = uint32(slot)
		}
	} else {
		for j, e := range st.sel {
			st.tab[e] = uint32(j)
		}
	}
	switch {
	case g.kb.pageFault == PageFaultSwap && len(st.sel) > 1:
		a, b := st.sel[0], st.sel[1]
		st.tab[a], st.tab[b] = st.tab[b], st.tab[a]
	case one:
		st.tab[picked] = st.tab[st.sel[0]]
	}
	if err := s.WriteAt(inst.bufs[f.table], 0, u32b(st.tab)); err != nil {
		return err
	}
	m1, err := sv.launchCodes(lc, inst, sv.in[:sv.n], sv.out[:sv.m], f.split, len(f.codes))
	if err != nil {
		return err
	}
	g.Steps.Serve(kind)
	g.MatrixLaunches[kind] += m0.matrix + m1.matrix
	g.AttendLaunches[kind] += m0.attend + m1.attend
	return nil
}

// headSpec is what a head graph binds, whatever its ops compute: its one
// matrix, the projection to the vocabulary, and the vectors the tier holds
// for a head -- the norm's scale and bias, the projection's bias. Its input
// is the rows, its outputs the logits and, where it writes them too, the
// rows it read them from (graph.Head.Hidden).
type headSpec struct {
	out                           jlm.Role
	hasNorm, hasNormBias, hasBias bool
}

// headShape is what gr binds, false for a graph the tier holds no buffers
// for: other than one input and one or two outputs, an output that is its
// input, a matrix other than one projection, or a vector other than the
// head's.
func headShape(gr *graph.Graph) (headSpec, bool) {
	var hs headSpec
	if gr == nil || len(gr.In) != 1 || len(gr.Out) < 1 || len(gr.Out) > 2 || len(gr.Ops) < 1 {
		return hs, false
	}
	for _, v := range gr.Out {
		if v == gr.In[0] {
			return hs, false
		}
	}
	matrix := false
	for _, o := range gr.Ops {
		for _, w := range o.Weight {
			switch {
			case w.Rows > 0:
				if matrix && w.Role != hs.out {
					return hs, false
				}
				hs.out, matrix = w.Role, true
			case w.Role == jlm.RoleOutputNorm:
				hs.hasNorm = true
			case w.Role == jlm.RoleOutputNormBias:
				hs.hasNormBias = true
			case w.Role == jlm.RoleOutputBias:
				hs.hasBias = true
			default:
				return hs, false
			}
		}
	}
	return hs, matrix
}

// headServes is nil where the composer serves a step's head as the model's
// head graph states it -- the graph read as what it binds (headShape), the
// hidden rows' graph binding the same, and every vector it names held
// (PrepHead uploads each) -- and why not where it does not.
func (bs *blockScratch) headServes(norm backend.Buf) error {
	hs, ok := headShape(bs.headGraph)
	hh, hok := headShape(bs.headHidden)
	switch {
	case !ok || !hok:
		return errors.New("the head's graph binds a matrix other than its one projection, or a vector the tier holds no head buffer for")
	case hh != hs || len(bs.headHidden.Out) != 2:
		return errors.New("the head's hidden rows graph binds other than its head, or writes no rows")
	case hs.hasNorm && norm == nil, hs.hasNormBias && bs.hNormB == nil, hs.hasBias && bs.hBias == nil:
		return errors.New("the head's graph names a vector the tier did not upload")
	}
	return nil
}

// runHead launches the served head sv on in, its logits into out and its
// normed rows into h.
func (sv *servedPart) runHead(g *devTier, lc *launcher, in, out, h backend.Buf) error {
	return sv.run(g, lc, sv.inst, []backend.Buf{in}, []backend.Buf{out, h}[:sv.m])
}

// servedHead is the model's head graph served at a step of rows: its norm
// bound to the tier's norm (and its bias, for a LayerNorm head), its
// projection to the head's weight and its bias to the tier's, under the roles
// the graph names (the embedding's where the model ties them). A form a row
// count: a decode step's one row, a ragged step's or a chunk's wanted rows.
// The graph is the one that writes the norm's rows too (nn.Head.HiddenGraph),
// into the scratch's normed rows as the tier's own head does (runHead): what
// a call's hidden readback reads (nn.Head.Hidden).
func (g *devTier) servedHead(bs *blockScratch, norm backend.Buf, rows int) (*servedPart, error) {
	gr, w := bs.headHidden, bs.head
	if g.rowsFaulted("nohidden") {
		gr = bs.headGraph
	}
	hs, ok := headShape(gr)
	if !ok {
		return nil, errors.New("the composed head: a graph binding a matrix other than its one projection, or a vector the tier holds no head buffer for")
	}
	t, ok := jlmType(w.t)
	if !ok {
		return nil, fmt.Errorf("the composed head: format %d has no container type", w.t)
	}
	bind := func(sl *composedPart) {
		sl.weights[hs.out] = w
		if hs.hasNorm {
			sl.vectors[jlm.RoleOutputNorm] = norm
		}
		if hs.hasNormBias {
			sl.vectors[jlm.RoleOutputNormBias] = bs.hNormB
		}
		if hs.hasBias {
			sl.vectors[jlm.RoleOutputBias] = bs.hBias
		}
	}
	sk := servedKey{kind: nn.PartHead, graph: gr, rows: rows, t: t}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		sl := composedPart{kind: nn.PartHead, graph: gr, formats: map[jlm.Role]jlm.Type{hs.out: t},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{}}
		if hs.hasBias {
			sl.formats[jlm.RoleOutputBias] = jlm.TypeF32
		}
		bind(&sl)
		return sl, nil
	})
	if err != nil {
		return nil, err
	}
	bind(&sv.sl)
	return sv, nil
}

// servedPickHead is the head projecting the picked rows alone
// (nn.Head.PickHead): the head's norm, its weight and its bias bound as
// servedHead binds them, the ids its second input. One form for the head's
// label bound, the labels a run-time input.
func (g *devTier) servedPickHead(bs *blockScratch, norm backend.Buf) (*servedPart, error) {
	gr, w := bs.pickHead, bs.head
	if gr == nil || len(gr.In) != 2 || len(gr.Out) != 1 || len(gr.Ops) < 2 || len(gr.Ops[0].Weight) < 1 ||
		gr.Ops[1].Kind != graph.OpMatVec || len(gr.Ops[1].In) != 2 || len(gr.Ops[1].Weight) < 1 || len(gr.Ops[1].Weight) > 2 {
		return nil, errors.New("the composed pick head: a graph that is not a norm and a row-selected projection")
	}
	t, ok := jlmType(w.t)
	if !ok {
		return nil, fmt.Errorf("the composed pick head: format %d has no container type", w.t)
	}
	n, mv := gr.Ops[0].Weight, gr.Ops[1].Weight
	out := mv[0].Role
	if len(mv) == 2 && bs.hBias == nil {
		return nil, fmt.Errorf("the composed pick head: its graph adds %v, which the tier does not hold", mv[1].Role)
	}
	bind := func(sl *composedPart) {
		sl.weights[out], sl.vectors[n[0].Role] = w, norm
		if len(n) == 2 {
			sl.vectors[n[1].Role] = bs.hNormB
		}
		if len(mv) == 2 {
			sl.vectors[mv[1].Role] = bs.hBias
		}
	}
	sk := servedKey{kind: nn.PartHead, graph: gr, rows: 1, t: t}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		sl := composedPart{kind: nn.PartHead, graph: gr, formats: map[jlm.Role]jlm.Type{out: t},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{}}
		if len(mv) == 2 {
			sl.formats[mv[1].Role] = jlm.TypeF32
		}
		bind(&sl)
		return sl, nil
	})
	if err != nil {
		return nil, err
	}
	bind(&sv.sl)
	return sv, nil
}

// tableName is the table gr writes: rope.cs, or rope.cs.i past the first.
func ropeIndex(gr *graph.Graph) (int, bool) {
	if gr == nil || len(gr.Out) != 1 {
		return 0, false
	}
	for _, o := range gr.Ops {
		if o.Kind == graph.OpRopeTable {
			return o.Attr.Rotary, true
		}
	}
	return 0, false
}

// tableAxes says gr is a table over several position axes, which reads each
// row's positions on them (rope.pos.axes) rather than one (rope.pos).
func tableAxes(gr *graph.Graph) bool {
	for _, o := range gr.Ops {
		if o.Kind == graph.OpRopeTable && o.Attr.Sections != [4]int{} {
			return true
		}
	}
	return false
}

// servedTable is a call's rotary table gr served at R rows: writing cs, its
// planes those the tier folded for it and its positions pos, the live row
// count and then each row's (four a row over several axes).
func (g *devTier) servedTable(gr *graph.Graph, R, marks int, cs, planes, rows backend.Buf) (*servedPart, error) {
	sk := servedKey{kind: nn.PartTable, graph: gr, rows: R, keys: marks}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		i, ok := ropeIndex(gr)
		if !ok {
			return composedPart{}, errors.New("the composed table is not a rotary table")
		}
		return composedPart{kind: nn.PartTable, part: composer.PartTable, graph: gr, formats: map[jlm.Role]jlm.Type{}, marks: marks,
			tables: map[composer.TableKey]backend.Buf{{Table: composer.TableAngles, Instance: i}: cs, {Table: composer.TablePlanes, Instance: i}: planes,
				tableKey(composer.TableRopeRows): rows, tableKey(composer.TableRopeMarks): g.marks.buf}}, nil
	})
	if err != nil {
		return nil, err
	}
	// Bound again every step, by the keys the build gave them: the arena
	// moves where a sequence's marks outgrow it.
	for k := range sv.sl.tables {
		switch k.Table {
		case composer.TableAngles:
			sv.sl.tables[k] = cs
		case composer.TablePlanes:
			sv.sl.tables[k] = planes
		case composer.TableRopeRows:
			sv.sl.tables[k] = rows
		case composer.TableRopeMarks:
			sv.sl.tables[k] = g.marks.buf
		}
	}
	return sv, nil
}

// tableServes is nil where the composer serves the tables of a call of R
// rows (rag's, a ragged step's) at cache position pos: a graph for each
// table bs holds (scratchTable), as wide as its rows, and the planes and
// positions each reads held. The front end states every table a rotating
// model turns by and the placement holds their buffers, so a table missing
// either is the step's error, named. Callers hold g.mu.
func (g *devTier) tableServes(bs *blockScratch, p *nn.LayerPlan, pos, R int, rag *ragStep) error {
	for i := range bs.tableCount() {
		gr := scratchTable(bs, p, i)
		if gr == nil {
			return fmt.Errorf("the plan states no graph for rotary table %d", i)
		}
		if w := gr.Values[gr.Out[0]].Width; w != bs.p.RopeW() {
			return fmt.Errorf("rotary table %d's graph is %d wide, its rows %d", i, w, bs.p.RopeW())
		}
		if cs, planes, at := g.tableBinds(bs, i); cs == nil || planes == nil || at == nil {
			return fmt.Errorf("rotary table %d's buffers are not held (table %v, planes %v, positions %v)", i, cs != nil, planes != nil, at != nil)
		}
	}
	return nil
}

// tableCount is how many tables bs holds: cs, and csSWA where the plan has
// local layers.
func (bs *blockScratch) tableCount() int {
	if bs.csSWA != nil {
		return 2
	}
	return 1
}

// scratchTable is the graph of bs's table i (cs, then csSWA) among the
// plan's tables, nil where the plan states none for it. A scratch set of one
// geometry of a model of mixed geometry (scratchPlanOf) holds its own layers'
// table in cs: the plan's first, or its second for blocks rotating by it.
func scratchTable(bs *blockScratch, p *nn.LayerPlan, i int) *graph.Graph {
	if bs.p.MixedGeom {
		switch {
		case i > 0 || len(p.Tables) != 2:
			return nil
		case bs.p.LocalRope:
			return p.Tables[1]
		}
		return p.Tables[0]
	}
	if len(p.Tables) != bs.tableCount() || i >= len(p.Tables) {
		return nil
	}
	return p.Tables[i]
}

// tableBinds is what table i of bs writes and reads: the global table cs or
// the local one, its planes, and its positions (rope.pos, or over several
// axes rope.pos.axes).
func (g *devTier) tableBinds(bs *blockScratch, i int) (cs, planes, rows backend.Buf) {
	if i > 0 {
		return bs.csSWA, bs.ropeTabSWA, bs.ropeRows
	}
	if gr := scratchTable(bs, &bs.p, 0); gr != nil && tableAxes(gr) {
		return bs.cs, bs.ropeTabAxes, bs.ropeRows
	}
	return bs.cs, bs.ropeTab, bs.ropeRows
}

// formTables forms bs's tables at its rows where the composer serves them,
// at a placement or where a width's scratch is built: a table met first at
// a step formed inside it. Callers hold g.mu.
func (g *devTier) formTables(bs *blockScratch) {
	if g.NoPromptPrewarm || bs == nil || bs.cs == nil {
		return
	}
	if bs.p.NonCausal && bs.gridGraph != nil {
		if _, err := g.servedGridTable(bs, bs.rows); err != nil && !errors.Is(err, errFormedAhead) {
			g.LastErr = "the composed grid table: " + err.Error()
		}
		return
	}
	if bs.p.NRot <= 0 {
		return
	}
	for i := range bs.tableCount() {
		gr := scratchTable(bs, &bs.p, i)
		cs, planes, at := g.tableBinds(bs, i)
		if gr == nil || cs == nil || planes == nil || at == nil {
			continue
		}
		if _, err := g.servedTable(gr, bs.rows, bs.p.RopeMarks, cs, planes, at); err != nil && !errors.Is(err, errFormedAhead) {
			g.LastErr = "the composed table: " + err.Error()
		}
	}
}

// prewarmComposedTables forms the step's tables where the composer serves
// them, at placement, in the decode scratch and every width's of every
// scratch set. Callers hold g.mu.
func (g *devTier) prewarmComposedTables() {
	was := g.prewarming
	g.prewarming = true
	defer func() { g.prewarming = was }()
	g.formTables(g.bs)
	for _, bb := range g.bbs {
		g.formTables(bb)
	}
	for _, gs := range g.geoSets {
		g.formTables(gs.bs)
		for _, bb := range gs.bbs {
			g.formTables(bb)
		}
	}
}

// embedServes is nil where the composer serves a step's or a prompt chunk's
// input rows: the front end's graph of the input rows, each of whose weights
// is a table the device holds in a format the composer reads -- the token
// embedding (the head where it is the embedding, nn.Head.TokEmbd otherwise),
// a learned position table (nn.Head.PosEmbd) and every other tensor the graph
// names (nn.Head.Streams). Where their room was short it is errEmbedRoom, and
// the host writes the rows; otherwise it names what the card was not handed.
func (bs *blockScratch) embedServes() error {
	switch {
	case bs.embedGraph == nil:
		return errors.New("the input rows: the head was offered with no graph of them")
	case bs.embErr != nil:
		return bs.embErr
	}
	role, vector, ok := bs.embedGap()
	switch {
	case ok:
		return nil
	case vector:
		return fmt.Errorf("the input rows' graph names %v, a vector the head did not carry", role)
	}
	return fmt.Errorf("the input rows' graph names %v, a table the card does not hold", role)
}

// embedServed is embedServes without its reason, allocating nothing: a step
// asks it every token.
func (bs *blockScratch) embedServed() bool {
	if bs.embedGraph == nil || bs.embErr != nil {
		return false
	}
	_, _, ok := bs.embedGap()
	return ok
}

// embedGap is the first weight the input rows' graph names that the card
// was not handed, and whether it is a vector rather than a table; ok where
// there is none.
func (bs *blockScratch) embedGap() (role jlm.Role, vector, ok bool) {
	for _, o := range bs.embedGraph.Ops {
		for _, w := range o.Weight {
			if w.Rows == 0 {
				if bs.embSV[w.Role] == nil {
					return w.Role, true, false
				}
				continue
			}
			if r := bs.embTable(w.Role); r == nil || !r.ok {
				return w.Role, false, false
			}
		}
	}
	return 0, false, true
}

// errEmbedRoom is input rows whose tables found no room beside the head and
// the blocks placed (prepEmbed): the host writes the rows.
const errEmbedRoom tierError = "the input rows' tables have no room on the card"

// embTable is the resident table of role the input rows read, nil for a
// role the device holds none of.
func (bs *blockScratch) embTable(role jlm.Role) *resident {
	switch role {
	case jlm.RoleTokenEmbd:
		return bs.embW
	case jlm.RolePosEmbd:
		return bs.posW
	}
	return bs.embSW[role]
}

// servedEmbed is the input rows served at R rows: the token embedding and a
// position table where the graph reads one, each row's token in embTok and
// its position in embPos (the live row count, then a word a row).
func (g *devTier) servedEmbed(bs *blockScratch, R int) (*servedPart, error) {
	gr, w := bs.embedGraph, bs.embW
	if gr == nil || w == nil {
		return nil, errors.New("the composed input rows: no graph, or no token embedding on the device")
	}
	t, ok := jlmType(w.t)
	if !ok {
		return nil, fmt.Errorf("the composed input rows: format %d has no container type", w.t)
	}
	sk := servedKey{kind: nn.PartEmbed, graph: gr, rows: R, t: t}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		formats := map[jlm.Role]jlm.Type{}
		weights := map[jlm.Role]*resident{}
		vectors := map[jlm.Role]backend.Buf{}
		for _, o := range gr.Ops {
			for _, w := range o.Weight {
				if w.Rows == 0 {
					v := bs.embSV[w.Role]
					if v == nil {
						return composedPart{}, fmt.Errorf("the composed input rows read %v, which the device does not hold", w.Role)
					}
					formats[w.Role], vectors[w.Role] = jlm.TypeF32, v
					continue
				}
				r := bs.embTable(w.Role)
				if r == nil {
					return composedPart{}, fmt.Errorf("the composed input rows read %v, which the device does not hold", w.Role)
				}
				t, ok := jlmType(r.t)
				if !ok {
					return composedPart{}, fmt.Errorf("the composed input rows: format %d has no container type", r.t)
				}
				formats[w.Role], weights[w.Role] = t, r
			}
		}
		cp := composedPart{kind: nn.PartEmbed, part: composer.PartEmbed, graph: g.embedFault(gr), formats: formats, weights: weights,
			vectors: vectors, tables: map[composer.TableKey]backend.Buf{tableKey(composer.TableTokens): bs.embTok}}
		if bs.embPos != nil {
			cp.tables[tableKey(composer.TablePos)] = bs.embPos
		}
		return cp, nil
	})
	if err != nil {
		return nil, err
	}
	// Bound again every step: a rebuilt scratch holds buffers of its own.
	for role := range sv.sl.weights {
		sv.sl.weights[role] = bs.embTable(role)
	}
	for role := range sv.sl.vectors {
		sv.sl.vectors[role] = bs.embSV[role]
	}
	sv.sl.tables[tableKey(composer.TableTokens)] = bs.embTok
	if _, ok := sv.sl.tables[tableKey(composer.TablePos)]; ok {
		sv.sl.tables[tableKey(composer.TablePos)] = bs.embPos
	}
	return sv, nil
}

// embedFault is gr with its scale dropped under ScaleFaultEmbd: the input
// rows formed unscaled, the violation Granite's gate must see.
func (g *devTier) embedFault(gr *graph.Graph) *graph.Graph {
	if g.kb.scaleFault != ScaleFaultEmbd {
		return gr
	}
	f := *gr
	f.Ops = nil
	alias := map[graph.ValueID]graph.ValueID{}
	for _, o := range gr.Ops {
		o.In = slices.Clone(o.In)
		for i, v := range o.In {
			if a, ok := alias[v]; ok {
				o.In[i] = a
			}
		}
		if o.Kind == graph.OpMap && o.Attr.Fn == graph.MapMul && len(o.In) == 1 && len(o.Out) == 1 {
			alias[o.Out[0]] = o.In[0]
			continue
		}
		f.Ops = append(f.Ops, o)
	}
	f.Out = slices.Clone(gr.Out)
	for i, v := range f.Out {
		if a, ok := alias[v]; ok {
			f.Out[i] = a
		}
	}
	return &f
}

// servedSample is the step's greedy token, gr, served over one row's logits.
func (g *devTier) servedSample(gr *graph.Graph) (*servedPart, error) {
	return g.servedSampleAt(gr, 1)
}

// servedSampleAt is servedSample over rows rows of logits, a token each.
func (g *devTier) servedSampleAt(gr *graph.Graph, rows int) (*servedPart, error) {
	if gr == nil || !(len(gr.In) == 1 && len(gr.Out) == 1 || len(gr.In) == 3 && len(gr.Out) == 2 || len(gr.In) == 2 && len(gr.Out) == 1 ||
		len(gr.In) == 4 && len(gr.Out) == 1) {
		return nil, errors.New("the composed sample: a graph that is not logits in and a token out, the sampler's candidates, its walk or the label pick")
	}
	sk := servedKey{kind: nn.PartSample, graph: gr, rows: rows}
	return g.servedFor(sk, func() (composedPart, error) {
		return composedPart{kind: nn.PartSample, part: composer.PartSample, graph: gr, formats: map[jlm.Role]jlm.Type{}}, nil
	})
}

// attendWindow is the window gr's attention states, and whether it is an
// aligned chunk.
func attendWindow(gr *graph.Graph) (int, bool) {
	if gr != nil {
		for _, o := range gr.Ops {
			if o.Kind == graph.OpAttend {
				return o.Attr.Window, o.Attr.Chunked
			}
		}
	}
	return 0, false
}

// inPlace says the form may write output out over input in: every kernel
// that reads in launches before the first that writes out. One kernel that
// reads in at the elements it writes out at is not enough: a launch clamps
// its surplus threads to the last element, and every lane storing an
// element may store before another lane reads it, adding twice (RULE 13).
// apart says a form of n residual streams (one where n < 2) writes some
// stream's output before it last reads that stream's input: its outputs are
// written apart, in bs.h, and copied over x after (servedCopy).
func (f *partForm) apart(n int) bool {
	for k := range max(n, 1) {
		if !f.inPlace(k, k) {
			return true
		}
	}
	return false
}

func (f *partForm) inPlace(in, out int) bool {
	lastRead, firstWrite := -1, len(f.codes)
	for i, args := range f.args {
		for _, a := range args {
			if a.kind == partIn && a.i == in && a.read {
				lastRead = i
			}
			if a.kind == partOut && a.i == out && a.writes {
				firstWrite = min(firstWrite, i)
			}
		}
	}
	return lastRead < firstWrite
}

// roleWeight is the block's matrix its graph binds under role, nil for one
// the tier does not hold.
func (l *layer) roleWeight(r jlm.Role) *resident {
	if r == jlm.RoleNGramTable && l.ngram != nil && len(l.ngram.ranges) > 0 {
		return &l.ngram.ranges[0].rows
	}
	if w := l.namedWeight(r); w != nil {
		return w
	}
	for _, x := range l.extra {
		if x.role == r {
			return x.r
		}
	}
	return nil
}

// namedWeight is the weight of role r the block holds by name, nil for one
// it holds none of or holds only among its extras.
func (l *layer) namedWeight(r jlm.Role) *resident {
	switch r {
	case jlm.RoleAttnQ, jlm.RoleVAttnQ:
		return l.wq
	case jlm.RoleAttnK, jlm.RoleVAttnK:
		return l.wk
	case jlm.RoleAttnV, jlm.RoleVAttnV:
		return l.wv
	case jlm.RoleAttnOut, jlm.RoleVAttnOut:
		return l.wo
	case jlm.RoleFFNGate, jlm.RoleVFFNGate:
		return l.gate
	case jlm.RoleFFNUp, jlm.RoleVFC1:
		return l.up
	case jlm.RoleFFNDown, jlm.RoleVFC2:
		return l.down
	case jlm.RoleAttnQA:
		return l.wqa
	case jlm.RoleAttnQB:
		return l.wqb
	case jlm.RoleAttnKVA:
		return l.wkva
	case jlm.RoleAttnKB:
		return l.wkb
	case jlm.RoleAttnVB:
		return l.wvb
	case jlm.RoleAttnGate:
		// A linear block of a model of latent attention gates its output
		// by its own.
		if l.linear {
			return l.ssmGate
		}
	case jlm.RoleAttnQKV:
		if l.linear && !l.withAttn {
			return l.wq
		}
	case jlm.RoleSCX:
		if l.linear {
			return l.wq
		}
	case jlm.RoleSCB:
		return l.ssmGate
	case jlm.RoleSCC, jlm.RoleSSMBA:
		return l.ssmBA
	case jlm.RoleSSMXDt:
		return l.ssmFA
	case jlm.RoleSSMDtProj:
		return l.ssmFB
	case jlm.RoleSSMXB:
		return l.ssmGA
	case jlm.RoleSSMXC:
		return l.ssmGB
	case jlm.RoleSSMOut:
		return l.ssmOut
	case jlm.RoleSSMFA:
		return l.ssmFA
	case jlm.RoleSSMFB:
		return l.ssmFB
	case jlm.RoleSSMGA:
		return l.ssmGA
	case jlm.RoleSSMGB:
		return l.ssmGB
	case jlm.RoleSSMConv1d:
		if l.ssmTaps.qs != nil {
			return &l.ssmTaps
		}
	case jlm.RoleRouter:
		if l.router != nil {
			return &l.routerW
		}
	case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank:
		// The banks whole on the card, or a streamed block's in pages: the
		// sheets its cache or its compact bank holds (heldSheets).
		if l.router == nil {
			return nil
		}
		return [...]*resident{l.gate, l.up, l.down}[r-jlm.RoleExpGateBank]
	case jlm.RoleShExpGate:
		return l.shGate
	case jlm.RoleShExpUp:
		return l.shUp
	case jlm.RoleShExpDown:
		return l.shDown
	}
	return nil
}

// expertsHeld says block l's routed experts are where its form reads them:
// every bank its graph binds held whole on the card, or a streamed block's
// in pages -- its cache's slots or its compact bank, each expert reached
// through the step's table (pagedRun) -- at a call of no more rows than its
// pages hold the selections of (pagedRows), its experts sent to the card
// rather than run on the host.
func (l *layer) expertsHeld() bool {
	held, banks := true, false
	for _, o := range l.graph.Ops {
		for _, w := range o.Weight {
			if jlm.ExpertBank(w.Role) {
				r := l.roleWeight(w.Role)
				banks, held = true, held && r != nil && w.Rows > 0 && r.nrows%w.Rows == 0
			}
		}
	}
	return !banks || held
}

// pagedCall is the rows a composed call over blocks [lo, hi) takes at a
// time, at most w: no more than every streamed block's pages hold.
func (g *devTier) pagedCall(lo, hi, w int) int {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.stream != nil {
			w = min(w, max(l.pagedRows(), 1))
		}
	}
	return w
}

// pagedRows is the most rows of a call whose selections a streamed block's
// pages hold whatever its rows choose: its sheets over a row's selection.
func (l *layer) pagedRows() int {
	st := l.stream
	if st == nil || len(st.sel) == 0 {
		return 0
	}
	if st.cached() {
		return len(st.owner) / len(st.sel)
	}
	return 1
}

// paged is each bank of block l's graph a streamed block holds in pages,
// and the sheets it holds of it; nil for a block whose banks are whole.
func (l *layer) paged() map[jlm.Role]int {
	if l.stream == nil {
		return nil
	}
	pg := map[jlm.Role]int{}
	for _, o := range l.graph.Ops {
		for _, w := range o.Weight {
			if r := l.roleWeight(w.Role); jlm.ExpertBank(w.Role) && r != nil && w.Rows > 0 {
				pg[w.Role] = r.nrows / w.Rows
			}
		}
	}
	return pg
}

// pagedHeld is the most sheets block l holds of any bank in pages (paged's
// largest), 0 for a block whose banks are whole: per step, with nothing made.
func (l *layer) pagedHeld() int {
	if l.stream == nil {
		return 0
	}
	n := 0
	for _, o := range l.graph.Ops {
		for _, w := range o.Weight {
			if r := l.roleWeight(w.Role); jlm.ExpertBank(w.Role) && r != nil && w.Rows > 0 {
				n = max(n, r.nrows/w.Rows)
			}
		}
	}
	return n
}

// weightsHeld says block l holds every matrix and vector its graph binds,
// as servedBlock binds them.
func (l *layer) weightsHeld() bool {
	_, missing := l.missingWeight()
	return !missing
}

// missingWeight is the first weight or vector the block's graph binds that
// the tier does not hold, and whether there is one.
func (l *layer) missingWeight() (jlm.Role, bool) {
	for _, o := range l.graph.Ops {
		for _, w := range o.Weight {
			if l.roleVector(w.Role) == nil && (w.Rows == 0 || l.roleWeight(w.Role) == nil) {
				return w.Role, true
			}
		}
	}
	return 0, false
}

// roleBuf is one of a block's vectors on the card.
type roleBuf struct {
	role jlm.Role
	buf  backend.Buf
}

// roleVector is the block's vector its graph binds under role, nil for one
// the tier does not hold.
func (l *layer) roleVector(r jlm.Role) backend.Buf {
	if b, ok := l.namedVector(r); ok {
		return b
	}
	for _, v := range l.vecs {
		if v.role == r {
			return v.buf
		}
	}
	return nil
}

// namedVector is the vector a field of the block holds under r, and whether
// a field holds r at all: a role a field holds is read from it, nil or not,
// as the host's is (model's layer.namedVector), and every other role from
// vecs.
func (l *layer) namedVector(r jlm.Role) (backend.Buf, bool) {
	switch r {
	case jlm.RoleAttnNorm:
		return l.nAttn, true
	case jlm.RoleFFNNorm:
		return l.nFFN, true
	case jlm.RoleAttnNormBias:
		return l.nAttnB, true
	case jlm.RoleFFNNormBias:
		return l.nFFNB, true
	case jlm.RoleAttnQNorm:
		return l.nQ, true
	case jlm.RoleAttnKNorm:
		return l.nK, true
	case jlm.RoleVClamp:
		return l.clampB, true
	case jlm.RoleLayerOutScale:
		return l.outScale, true
	}
	return nil, false
}

// blockServes is nil where the composer serves block li of this step whole,
// and what is missing where it does not: a graph from the front end, its
// experts where its form reads them (expertsHeld), every weight and vector
// its graph binds held (weightsHeld), a recurrent block's state where the
// form steps it (recHeld), the residual streams and the rows' per-layer
// inputs it reads, and the paged history -- which a step builds, so that is
// the step's own condition (errStepCond).
func (g *devTier) blockServes(l *layer, li int, p *nn.LayerPlan, bs *blockScratch, prompt bool) error {
	pk := bs.pkv
	switch {
	case l.graph == nil:
		return errors.New("the block has no graph from the front end")
	case !l.expertsHeld():
		return errors.New("a bank its graph binds is neither whole on the card nor in a streamed block's pages")
	case !l.weightsHeld():
		r, _ := l.missingWeight()
		return fmt.Errorf("its graph binds %v, which the block does not hold", r)
	case l.linear && !g.recHeld(l, p, bs):
		return errors.New("its recurrent state is not where the form steps it")
	case len(l.graph.Out) != p.Streams() || len(l.graph.In) > len(l.graph.Out) && (len(l.graph.In) != len(l.graph.Out)+1 || bs.pleIn == nil):
		// A block reads each residual stream and, past them, the rows'
		// per-layer inputs.
		return fmt.Errorf("its graph reads %d inputs and writes %d streams, where the step holds %d streams and per-layer inputs %v",
			len(l.graph.In), len(l.graph.Out), p.Streams(), bs.pleIn != nil)
	case l.linear && !l.withAttn:
		return nil
	case l.nonCausal:
		return g.towerServes(l, li, p, bs)
	case pk == nil || g.kvp == nil || g.kvp.layers[l.kvSrc] == nil:
		// A block sharing another's history reads its source's pool.
		return fmt.Errorf("%w: no paged history on the device", errStepCond)
	}
	return nil
}

// errStepCond marks what a step stages itself -- the paged history it
// builds, a call's windows -- which no placement decides on (unserved).
const errStepCond tierError = "a step's own condition"

// unserved is why placed block li is one the composer does not serve on
// this device, "" where it does: prepLayer declines it, so it runs on the
// CPU device's composed form, never on the tier's own kernels. A step's own
// conditions (the caller's keep, the pool a step builds) are not a
// placement's to decide.
func (g *devTier) unserved(li int) string {
	defer g.useGeom(g.enterGeomOf(li))
	bs, l := g.bs, g.layers[li]
	if bs == nil || l == nil {
		return ""
	}
	if err := g.blockServes(l, li, &bs.p, bs, false); err != nil && !errors.Is(err, errStepCond) {
		return fmt.Sprintf("block %d: the composer does not serve it on this device (%v)", li, err)
	}
	return ""
}

// recHeld says linear block l's recurrent state is where the composer's
// form steps it: in the block's pool of slots of the delta rule's state
// and any tail its graph states, the step's recurrent descriptor naming
// each run's slots -- a decode's one row, a prompt chunk's rows as one run, a ragged step's
// sequences a run each. A pool of two halves is read in one and written in
// the other; a pool of one shared half is stepped in place, through the
// composer's own descriptor (bs.cRows).
func (g *devTier) recHeld(l *layer, p *nn.LayerPlan, bs *blockScratch) bool {
	return l.pool != nil && bs.dRows != nil && (l.pool.s[1] != nil || g.recShared && bs.cRows != nil) &&
		(p.Recurrent.TailLen == 0 || tailStated(l.graph, p.Recurrent.StateLen))
}

// tailStated says graph g's delta rule reads its state at slots of n floats:
// a tail past the rule's matrices that the graph states (graph.OpDeltaRule's
// Span), so the composer's slot is the pool's.
func tailStated(g *graph.Graph, n int) bool {
	if g == nil {
		return false
	}
	for _, o := range g.Ops {
		if o.Kind == graph.OpDeltaRule {
			return o.Attr.Span == n
		}
	}
	return false
}

// bindRec binds linear block l's step in sv to session sid's half of its
// pool, read, and the other half, written, as emitLinear does -- in the
// shared form the one half both, through the composer's descriptor -- or
// says it cannot: the session holds no seat in the pool.
func (g *devTier) bindRec(sv *servedPart, bs *blockScratch, sid uint64, l *layer) bool {
	par, has := g.stepPar(sid, l)
	if !has || l.pool == nil {
		return false
	}
	t := sv.sl.tables
	if g.keeping {
		k := g.kept
		if k == nil || k.s[l] == nil {
			return false
		}
		t[tableKey(composer.TableRecKeep)], t[tableKey(composer.TableRecConvKeep)] = k.s[l], k.c[l]
	}
	t[tableKey(composer.TableRecConv)], t[tableKey(composer.TableRecConvOut)] = l.pool.c[par], l.pool.c[1-par]
	if g.recShared {
		if bs.cRows == nil {
			return false
		}
		t[tableKey(composer.TableRecState)], t[tableKey(composer.TableRecStateOut)], t[tableKey(composer.TableRecRows)] = l.pool.s[0], l.pool.s[0], bs.cRows
		return true
	}
	t[tableKey(composer.TableRecState)], t[tableKey(composer.TableRecStateOut)] = l.pool.s[par], l.pool.s[1-par]
	return true
}

// servedStep is block li's served part at this step of R rows and the
// lane's instance it launches in, nil where the budget does not hold the
// instance of a prompt's form.
// A form that reads x in the kernel writing its output is written apart, in
// bs.h, and copied over x after (servedCopy): cp is that copy, nil where the
// form writes over x.
func (g *devTier) servedStep(bs *blockScratch, l *layer, li int, p *nn.LayerPlan, R int, one bool) (sv *servedPart, inst *partInst, cp *servedPart, err error) {
	if l.nonCausal {
		return g.towerStep(bs, l, li, p, R)
	}
	rcs := bs.cs
	qOff := bs.zero
	if R > 1 {
		qOff = bs.roff
	}
	var run []backend.Buf
	keys := p.MaxSeq
	if bs.inPasses(l) {
		if run, err = g.runBufs(bs, l, li); err != nil {
			return nil, nil, nil, fmt.Errorf("%w: %w", errStreamForm, err)
		}
		keys = g.passKeys(bs, keys)
	}
	sv, err = g.servedBlockAs(l, li, rcs, g.kvp.layers[l.kvSrc], bs.pkv.pRow, g.windowDesc(bs.pkv, l), qOff, bs.calls, bs.dRows, keys, g.kvp.p, R,
		rowSeqs(R, one), run, bs)
	if err != nil {
		if run != nil {
			err = fmt.Errorf("%w: %w", errStreamForm, err)
		}
		return nil, nil, nil, err
	}
	if sv.f.apart(len(l.graph.Out)) {
		if cp, err = g.servedCopy(bs, R); err != nil {
			return nil, nil, nil, err
		}
	}
	if R == 1 && run == nil {
		return sv, sv.inst, cp, nil
	}
	return sv, g.promptInst(bs, sv.f), cp, nil
}

// errStreamForm names a block whose form over passes did not form: its step
// fails with it, as any form's that does not build.
const errStreamForm tierError = "the composed block over passes"

// servedCopy is the copy of a block's output from bs.h over x at a step of
// rows, formed through the composer like any other part, for a form that
// cannot write its output over x (partForm.apart). bs.h holds every step
// row at the residual's width, its streams laid out as x's
// (servedStreamsApart), so one copy of rows rows of that width moves them
// all.
func (g *devTier) servedCopy(bs *blockScratch, rows int) (*servedPart, error) {
	if rows > bs.rows {
		return nil, fmt.Errorf("tier: a block output of %d rows apart from x, where the scratch holds %d", rows, bs.rows)
	}
	return g.servedRowCopy(bs.p.ResidW(), rows)
}

// prewarmComposed forms every placed block's part and the head's at
// placement, as prewarmDecodePlans builds the decode plans: a form met first
// at a token compiles inside it. Callers hold g.mu.
func (g *devTier) prewarmComposed() {
	g.prewarmComposedTables()
	for li := range g.layers {
		g.prewarmComposedBlock(li, false)
	}
	g.prewarmComposedHead(false)
	g.prepRowsReserved()
}

// prewarmComposedBlock forms block li's part where the composer serves it:
// at the placement, and at every placement of the block after it (a move, a
// seam grown back), which a step would otherwise meet first. A form that
// does not build, or does not fit, is the block's reason to be declined,
// named in LastErr; the other blocks still form.
// Callers hold g.mu.
func (g *devTier) prewarmComposedBlock(li int, narrow bool) (roomless string) {
	if l := g.layers[li]; l != nil && l.nonCausal {
		g.prewarming = true
		defer func() { g.prewarming = false }()
		return g.prewarmTower(li)
	}
	// A sliding block of a split model forms over the sliding set's table.
	defer g.useGeom(g.enterGeomOf(li))
	bs, l := g.bs, g.layers[li]
	if g.NoPromptPrewarm || bs == nil || bs.pkv == nil || l == nil {
		return ""
	}
	// A model with no attention block has no pool from its placement
	// (pagedLayer); its first step would make one and form there.
	g.kvPages()
	g.prewarming = true
	defer func() { g.prewarming = false }()
	if g.blockServes(l, li, &bs.p, bs, false) == nil {
		rcs := bs.cs
		serve := func() (*servedPart, error) {
			return g.servedBlock(l, li, rcs, g.kvp.layers[l.kvSrc], bs.pkv.pRow, g.windowDesc(bs.pkv, l), bs.zero, bs.calls, bs.dRows, bs.p.MaxSeq,
				g.kvp.p, 1, 0, bs)
		}
		sv, err := serve()
		// At a placement the prompt's reserved width gives way first, as it
		// does to the prompt part (fitReserved).
		for narrow && err != nil && errors.As(err, new(partRoomError)) && g.narrowReserved(fmt.Sprintf("block %d's composed buffers", li)) {
			sv, err = serve()
		}
		if err != nil {
			g.LastErr = "the composed block: " + err.Error()
			if errors.As(err, new(partRoomError)) {
				return fmt.Sprintf("block %d: its composed part does not fit beside the blocks placed (%s)", li, err)
			}
			return fmt.Sprintf("block %d: its composed part does not build on this device (%s)", li, err)
		}
		if sv.f.apart(len(l.graph.Out)) {
			if _, err := g.servedCopy(bs, 1); err != nil {
				g.LastErr = "the composed block's copy: " + err.Error()
			}
		}
		// A form that failed earlier was named then: prefixed once.
		if was := g.LastErr; (!l.linear || l.withAttn) && !g.streamServed(l, li, bs) && g.LastErr != was {
			g.LastErr = "the composed block over passes: " + g.LastErr
		}
	}
	// Every prompt width the lane holds a scratch for, and a ragged step's:
	// its first chunk or step forms and allocates nothing.
	for _, bb := range g.bbs {
		g.prewarmPromptBlock(bb, li)
	}
	g.fitReserved()
	// Every other lane's instance too: a session's first step after a move
	// forms nothing.
	g.eachLane(func(o *lane) {
		if o == g.lane {
			return
		}
		if err := g.spreadParts(g.lane, o, l.graph); err != nil {
			g.LastErr = "a lane's composed block: " + err.Error()
		}
		for _, gr := range g.copyGraphs {
			if err := g.spreadParts(g.lane, o, gr); err != nil {
				g.LastErr = "a lane's composed copy: " + err.Error()
			}
		}
	})
	return ""
}

// prewarmPromptBlock forms block li's part at the rows of bb, a prompt
// width's scratch, where the composer serves it, and the scratch's instance
// of its form. Callers hold g.mu and say whether this is the placement's
// (g.prewarming).
func (g *devTier) prewarmPromptBlock(bb *blockScratch, li int) {
	l := g.layers[li]
	if l == nil || bb == nil {
		return
	}
	// Its form over passes too, as main's staged prefill holds its planes
	// from the placement: a chunk over evicted history forms nothing.
	if was := g.LastErr; g.promptServed(l, li, bb) && (!l.linear || l.withAttn) && !g.streamServed(l, li, bb) && g.LastErr != was {
		g.LastErr = "the composed prompt block over passes: " + g.LastErr
	}
}

// promptRoomless is why block li, which the composer serves, has no prompt
// part at the reserved width for want of room, or "" where it has one, the
// composer does not serve it, or what failed was not room. Callers hold g.mu.
func (g *devTier) promptRoomless(li int) string {
	defer g.useGeom(g.enterGeomOf(li))
	// A placement's: what promptServed forms here is formed at it.
	was := g.prewarming
	g.prewarming = true
	defer func() { g.prewarming = was }()
	bb, l := g.bbs[g.promptW], g.layers[li]
	if bb == nil || l == nil || g.blockServes(l, li, &bb.p, bb, true) != nil {
		return ""
	}
	bb.instNoRoom = false
	if g.promptServed(l, li, bb) || !bb.instNoRoom {
		return ""
	}
	return fmt.Sprintf("block %d: its composed prompt part does not fit beside the blocks placed (%s)", li, g.LastErr)
}

// promptServed reports whether the composer serves block li of a prompt call
// or a ragged step at bb's rows (rowsForms, blockServes): its form formed and
// the scratch's instance of it held, both made here where they are not,
// outside the submission. Callers hold g.mu.
func (g *devTier) promptServed(l *layer, li int, bb *blockScratch) bool {
	if bb.rows < 2 || bb.pkv == nil || g.kvp == nil ||
		g.blockServes(l, li, &bb.p, bb, true) != nil {
		return false
	}
	rcs := bb.cs
	// A call's rows are one sequence's in position order (a prompt's
	// chunk), or at the ragged step's width any sequences': a form each,
	// both formed here.
	for _, seqs := range [...]int{1, 0} {
		if seqs == 0 && !g.raggedAt(bb) {
			continue
		}
		sv, err := g.servedBlock(l, li, rcs, g.kvp.layers[l.kvSrc], bb.pkv.pRow, g.windowDesc(bb.pkv, l), bb.roff, bb.calls, bb.dRows,
			bb.p.MaxSeq, g.kvp.p, bb.rows, seqs, bb)
		if errors.Is(err, errFormedAhead) {
			continue
		}
		if err != nil {
			// A form that failed earlier was named then (servedFor).
			if !errors.Is(err, errFormedEarlier) {
				g.LastErr = "the composed prompt block: " + err.Error()
			}
			return false
		}
		if sv.f.apart(len(l.graph.Out)) {
			if _, err := g.servedCopy(bb, bb.rows); err != nil {
				g.LastErr = "the composed prompt block's copy: " + err.Error()
				return false
			}
		}
		if g.promptInst(bb, sv.f) == nil {
			return false
		}
	}
	if bb.inPasses(l) && !g.streamServed(l, li, bb) {
		return false
	}
	return true
}

// streamServed reports whether the composer serves block li at bb's rows
// over a history read in passes: its forms over passes (servedStep's where
// the call's history streams) formed, one a sequences form the rows take,
// and the scratch's instance of each held, made here where they are not.
// A block whose pool never evicts (kvLayerPool.noEvict: a pooled selection,
// not yet formed over passes) is never read in passes, and has no form over
// them to make.
// Callers hold g.mu, outside any submission.
func (g *devTier) streamServed(l *layer, li int, bb *blockScratch) bool {
	if pl := g.kvp.layers[l.kvSrc]; pl != nil && pl.noEvict {
		return true
	}
	run, err := g.runBufs(bb, l, li)
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	rcs, qOff := bb.cs, bb.zero
	if bb.rows > 1 {
		qOff = bb.roff
	}
	for _, seqs := range [...]int{1, 0} {
		if seqs == 1 && bb.rows == 1 || seqs == 0 && bb.rows > 1 && !g.raggedAt(bb) {
			continue
		}
		sv, err := g.servedBlockAs(l, li, rcs, g.kvp.layers[l.kvSrc], bb.pkv.pRow, g.windowDesc(bb.pkv, l), qOff, bb.calls, bb.dRows,
			g.passKeys(bb, bb.p.MaxSeq), g.kvp.p, bb.rows, seqs, run, bb)
		if errors.Is(err, errFormedAhead) {
			continue
		}
		if err != nil {
			if !errors.Is(err, errFormedEarlier) {
				g.LastErr = err.Error()
			}
			return false
		}
		if sv.f.apart(len(l.graph.Out)) {
			if _, err := g.servedCopy(bb, bb.rows); err != nil {
				g.LastErr = err.Error()
				return false
			}
		}
		if g.promptInst(bb, sv.f) == nil {
			return false
		}
	}
	return true
}

// rowSeqs is the sequences a call of R rows is of as its form states them
// (composer.Step.Seqs): 1 where its rows are one sequence's in position order,
// else 0, a row each.
func rowSeqs(R int, one bool) int {
	if R > 1 && one {
		return 1
	}
	return 0
}

// rowsKind is the part a call of R rows asks of the composer for its
// blocks: a prompt's chunk, or a call of rows that is not a ragged step, is
// nn.ComposedPrompt's; a ragged decode step's rows and a one-row step are
// nn.ComposedBlock's.
func (g *devTier) rowsKind(R int, rag bool) nn.Part {
	if (R > 1 || g.prompt) && (!rag || g.prompt) {
		return nn.PartPrompt
	}
	return nn.PartBlock
}

// promptForms forms every placed block's part at bb's rows where the
// composer serves it, with bb's instance of its form: a scratch built for a
// chunk (keepExact) allocates here, outside the submission, and a block
// placed after the placement's width loop (a seam grown back) forms here, as
// the walk compiles the tier's own kernels for it. At a placement
// (reserveScratch) that is the placement's forming. Callers hold g.mu.
func (g *devTier) promptForms(bb *blockScratch) {
	if bb == nil || bb.rows < 2 {
		return
	}
	was := g.prewarming
	g.prewarming = was || g.reserving
	defer func() { g.prewarming = was }()
	g.formTables(bb)
	// The input rows at bb's rows, where the composer gathers a chunk's.
	if bs := g.bs; bs != nil && bs.embedServes() == nil && g.embedBufs(bs) {
		if _, err := g.servedEmbed(bs, bb.rows); err != nil && !errors.Is(err, errFormedAhead) {
			g.LastErr = "the composed input rows: " + err.Error()
		}
	}
	for li := range g.layers {
		// A block of another geometry forms at its own set's scratch of
		// these rows: the current set's holds this set's rotary alone, and
		// a sliding block's second table would bind nothing.
		if g.geoUsed && g.geomFor(li, li+1) != g.geoCur {
			prev := g.enterGeomOf(li)
			g.prewarmPromptBlock(g.bbs[bb.rows], li)
			g.useGeom(prev)
			continue
		}
		g.prewarmPromptBlock(bb, li)
	}
}

// raggedAt says the composer serves a ragged step's rows, any sequences'
// mixed, at batched scratch bb's rows: the server's step width, or the
// reserved width a ragged step runs in (reserveScratch).
func (g *devTier) raggedAt(bb *blockScratch) bool {
	return bb != nil && (g.stepW > 0 && bb.rows == g.stepW || bb == g.reservedFor(nn.ComposedBlock))
}

// keyBound is a block form's key bound for a session of maxSeq positions.
func keyBound(maxSeq int) int { return (maxSeq + scoreGrain - 1) / scoreGrain * scoreGrain }

// prewarmComposedHead forms the head's part where the composer serves it,
// at placement and again at PrepHead, which a model calls after placing its
// blocks: a head met first at a token formed inside it. Callers hold g.mu.
func (g *devTier) prewarmComposedHead(narrow bool) (roomless string) {
	bs := g.bs
	if !g.NoPromptPrewarm && bs != nil && bs.sampleGraph != nil && g.argmaxOut != nil {
		g.prewarming = true
		if _, err := g.servedSample(bs.sampleGraph); err != nil {
			g.LastErr = "the composed sample: " + err.Error()
		}
		g.prewarming = false
	}
	// A ragged step's tokens at each width the lane holds a scratch for.
	if !g.NoPromptPrewarm && bs != nil && bs.sampleGraph != nil {
		g.prewarming = true
		for w := range g.bbs {
			if _, err := g.servedSampleAt(bs.sampleGraph, w); err != nil {
				g.LastErr = "the composed rows sample: " + err.Error()
			}
		}
		g.prewarming = false
	}
	// The sampler's walk and the label pick, one form each whatever k, the
	// cuts, the history and the labels.
	if !g.NoPromptPrewarm && bs != nil && g.argmaxOut != nil {
		g.prewarming = true
		for _, gr := range [...]*graph.Graph{bs.sampleWalk, bs.pickGraph} {
			if gr == nil {
				continue
			}
			if _, err := g.servedSample(gr); err != nil {
				g.LastErr = "the composed sampler's walk or the label pick: " + err.Error()
			}
		}
		g.prewarming = false
	}
	// The collapse of every stream a call of the head alone runs, at a
	// decode step's row and at every width a scratch is held at: a
	// verification's rows run it at their scratch's.
	if !g.NoPromptPrewarm && bs != nil && bs.collapse != nil {
		g.prewarming = true
		form := func(rows int) {
			if _, err := g.servedCollapse(bs.collapse, rows); err != nil {
				g.LastErr = "the composed streams' collapse: " + err.Error()
			}
			if _, err := g.servedRowCopy(bs.p.NEmbd, rows); err != nil {
				g.LastErr = "the composed copy of the streams' collapse: " + err.Error()
			}
		}
		form(1)
		for w := range g.bbs {
			form(w)
		}
		if g.promptsReserved() && g.bbs[batchWidth] == nil {
			form(batchWidth)
		}
		g.prewarming = false
	}
	// The input rows at one row, at every prompt width the lane holds a
	// scratch for, and at every exact width a chunk's scratch is cut to
	// (keepExact), as prewarmPromptWidths forms the blocks'.
	if !g.NoPromptPrewarm && bs != nil && bs.embedServes() == nil && g.embedBufs(bs) {
		g.prewarming = true
		form := func(rows int) {
			if _, err := g.servedEmbed(bs, rows); err != nil {
				g.LastErr = "the composed input rows: " + err.Error()
			}
		}
		form(1)
		for w := range g.bbs {
			form(w)
		}
		if g.promptsReserved() && g.bbs[batchWidth] == nil {
			form(batchWidth)
		}
		if grain := g.promptGrain(); !g.NoExactPrompt && grain > 0 && !g.promptsReserved() {
			for w := grain; w < g.promptW; w += grain {
				form(w)
			}
		}
		g.prewarming = false
	}
	if g.NoPromptPrewarm || bs == nil || bs.head == nil ||
		bs.headServes(bs.hNorm) != nil {
		return
	}
	g.prewarming = true
	defer func() { g.prewarming = false }()
	_, err := g.servedHead(bs, bs.hNorm, 1)
	for narrow && err != nil && errors.As(err, new(partRoomError)) && g.narrowReserved("the composed head's buffers") {
		_, err = g.servedHead(bs, bs.hNorm, 1)
	}
	if err != nil {
		g.LastErr = "the composed head: " + err.Error()
		if errors.As(err, new(partRoomError)) {
			return fmt.Sprintf("the composed head does not fit beside the blocks placed (%s)", err)
		}
	}
	if bs.pickHead != nil {
		if _, err := g.servedPickHead(bs, bs.hNorm); err != nil {
			g.LastErr = "the composed pick head: " + err.Error()
		}
	}
	// A head over rows at each width the lane holds a scratch for, its rows
	// a call's (composer.CallRows): a step of any count of wanted rows forms
	// nothing.
	for w := range g.bbs {
		if _, err := g.servedHead(bs, bs.hNorm, w); err != nil {
			g.LastErr = "the composed rows head: " + err.Error()
			return
		}
	}
	return
}

// poolHistory is how a pool layer of geometry k lays out its pages for the
// composer: keys by dimension and values by position, binary16 where the
// layer holds them so; a latent layer's one row a position, its value the
// row itself.
func poolHistory(k kvGeom) composer.History {
	if k.mla {
		return composer.History{ValuesInKeys: true}
	}
	return composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: k.f16}}
}

// servedBlock is block li served whole at a step of R rows: its graph formed
// as one unit on the block's weights and vectors by role, its rotary table
// cs and its history in the tier's pool layer pl, through the rows'
// descriptors, a windowed attention through rowsW
// (pagedScratch.windowDesc), for a session of maxSeq positions.
//
// The form is the session's: its attention runs as many trips as each row's
// live positions need, a count it reads at run time (composer.Trips), so the
// session's capacity is its key bound and no position compiles a form. A
// prompt chunk's form is its width's, as the tier's kernels are.
func (g *devTier) servedBlock(l *layer, li int, cs backend.Buf, pl *kvLayerPool, rows, rowsW, qOff, calls, rec backend.Buf, maxSeq, page, R, seqs int,
	bs *blockScratch) (*servedPart, error) {
	return g.servedBlockAs(l, li, cs, pl, rows, rowsW, qOff, calls, rec, maxSeq, page, R, seqs, nil, bs)
}

// servedBlockAs is servedBlock, its history read in passes where run, the
// running partial's buffers, is set (composer.History.Passes).
func (g *devTier) servedBlockAs(l *layer, li int, cs backend.Buf, pl *kvLayerPool, rows, rowsW, qOff, calls, rec backend.Buf, maxSeq, page, R, seqs int,
	run []backend.Buf, bs *blockScratch) (*servedPart, error) {
	keys := keyBound(maxSeq)
	kind := nn.PartBlock
	if R > 1 {
		kind = nn.PartPrompt
	}
	// A selection over a history read in passes ranks the whole history's
	// blocks: its form is told the positions the history holds, bounded as
	// its keys are.
	context := 0
	if run != nil && bs != nil && plainSelects(l.graph) {
		context = keyBound(bs.p.MaxSeq)
	}
	// A linear block holds no history, unless it attends too: its state is
	// its pool's (bindRec).
	var kk, kv, tab backend.Buf
	var geom kvGeom
	if pl != nil && (!l.linear || l.withAttn) {
		kk, kv, tab, geom = pl.k, pl.v, pl.tab, pl.geom
	}
	// A block keeping entries reads them as its history 1.
	var ek, etab backend.Buf
	var more []composer.History
	if g.kvp != nil {
		if pe := g.kvp.layers[entKey(l.kvSrc)]; pe != nil {
			ek, etab = pe.k, pe.tab
			more = []composer.History{{ValuesInKeys: true, Period: pe.geom.period}}
		}
	}
	// A step that keeps its rows runs a linear block's keep form.
	bg := l.graph
	if k := g.kept; g.keeping && l.linear && k != nil && k.graphs[l] != nil {
		bg = k.graphs[l]
	}
	sk := servedKey{kind: kind, graph: bg, rows: R, keys: keys, page: page, half: geom.f16, latent: geom.mla, seqs: seqs, stream: run != nil,
		held: l.pagedHeld(), context: context}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		// The request names no block: blocks of one structure and formats
		// share a form, each launched on its own bindings, one after
		// another on the device's queue.
		gr := *bg
		gr.Block = 0
		sl := composedPart{kind: kind, graph: &gr, formats: map[jlm.Role]jlm.Type{},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{},
			tables: map[composer.TableKey]backend.Buf{tableKey(composer.TableAngles): cs, tableKey(composer.TableKeys): kk, tableKey(composer.TableValues): kv, tableKey(composer.TablePages): tab, tableKey(composer.TableRows): rows,
				tableKey(composer.TableWindows): rowsW, tableKey(composer.TableQOff): qOff, tableKey(composer.TableCall): calls},
			history: poolHistory(geom), more: more,
			keys: keys, page: page, seqs: seqs, paged: l.paged(), context: context}
		if ek != nil {
			sl.tables[composer.TableKey{Table: composer.TableKeys, Instance: 1}] = ek
			sl.tables[composer.TableKey{Table: composer.TablePages, Instance: 1}] = etab
		}
		if run != nil {
			sl.history.Passes = true
			sl.tables[tableKey(composer.TableRunIn)], sl.tables[tableKey(composer.TableRunOut)] = run[0], run[1]
			if context > 0 {
				sl.tables[tableKey(composer.TableSelIn)], sl.tables[tableKey(composer.TableSelOut)] = bs.pkv.sel[0], bs.pkv.sel[1]
			}
		}
		g.bindRowTables(sl.tables, bs)
		if bs != nil {
			sl.rotaries = bs.p.Rotaries
		}
		g.bindRotaries(sl.tables, bs, cs, readsPlanes(l.graph))
		if l.linear && l.pool != nil {
			t := sl.tables
			// The shared form holds one half, stepped in place (bindRec).
			out := l.pool.s[1]
			if out == nil {
				out = l.pool.s[0]
			}
			t[tableKey(composer.TableRecState)], t[tableKey(composer.TableRecStateOut)], t[tableKey(composer.TableRecConv)], t[tableKey(composer.TableRecConvOut)] = l.pool.s[0], out, l.pool.c[0], l.pool.c[1]
			t[tableKey(composer.TableRecRows)] = rec
		}
		if k := g.kept; g.keeping && l.linear && k != nil {
			sl.tables[tableKey(composer.TableRecKeep)], sl.tables[tableKey(composer.TableRecConvKeep)] = k.s[l], k.c[l]
		}
		for _, o := range l.graph.Ops {
			for _, w := range o.Weight {
				// The n-gram table is read in its stated format, a call's
				// rows at a time (ngramRows).
				if w.Role == jlm.RoleNGramTable && l.ngram != nil {
					sl.formats[w.Role] = l.ngram.typ
					continue
				}
				// A bank's bias is a row an expert, held as one vector.
				if w.Rows > 0 && (l.roleVector(w.Role) == nil || l.roleWeight(w.Role) != nil) {
					r := l.roleWeight(w.Role)
					if r == nil {
						return sl, fmt.Errorf("the composed block %d binds %v, which the tier holds no matrix for", li, w.Role)
					}
					t, ok := jlmType(r.t)
					if !ok {
						return sl, fmt.Errorf("the composed block %d: format %d of %v has no container type", li, r.t, w.Role)
					}
					sl.weights[w.Role], sl.formats[w.Role] = r, t
					continue
				}
				b := l.roleVector(w.Role)
				if b == nil {
					return sl, fmt.Errorf("the composed block %d binds %v, which the tier holds no vector for", li, w.Role)
				}
				// The tier holds every vector as float32, whatever the file.
				sl.vectors[w.Role], sl.formats[w.Role] = b, jlm.TypeF32
			}
		}
		return sl, nil
	})
	if err != nil {
		return nil, err
	}
	// Bound again every step: a page-in or a move replaces a block's buffers.
	for r := range sv.sl.weights {
		sv.sl.weights[r] = l.roleWeight(r)
	}
	sv.sl.ngram = l.ngram
	for r := range sv.sl.vectors {
		sv.sl.vectors[r] = l.roleVector(r)
	}
	// A kernel reading as a vector a role the part bound as a matrix (a map
	// over a 2-D vector) reads the block's vector of it.
	for _, args := range sv.f.args {
		for _, a := range args {
			if a.kind == partVector && sv.sl.vectors[a.role] == nil {
				if v := l.roleVector(a.role); v != nil {
					sv.sl.vectors[a.role] = v
				}
			}
		}
	}
	t := sv.sl.tables
	t[tableKey(composer.TableAngles)], t[tableKey(composer.TableKeys)], t[tableKey(composer.TableValues)], t[tableKey(composer.TablePages)], t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)], t[tableKey(composer.TableQOff)] = cs, kk, kv, tab, rows, rowsW, qOff
	t[tableKey(composer.TableCall)] = calls
	if ek != nil {
		t[composer.TableKey{Table: composer.TableKeys, Instance: 1}] = ek
		t[composer.TableKey{Table: composer.TablePages, Instance: 1}] = etab
	}
	if run != nil {
		t[tableKey(composer.TableRunIn)], t[tableKey(composer.TableRunOut)] = run[0], run[1]
		if context > 0 {
			t[tableKey(composer.TableSelIn)], t[tableKey(composer.TableSelOut)] = bs.pkv.sel[0], bs.pkv.sel[1]
		}
	}
	if l.linear {
		t[tableKey(composer.TableRecRows)] = rec
	}
	g.bindRowTables(t, bs)
	g.bindRotaries(t, bs, cs, readsPlanes(l.graph))
	return sv, nil
}

// bindRotaries binds a block's rotary tables into t by instance, as its rope
// ops pick them (composer.Request.Rotaries): the global table cs and, where the
// model trains a second rotary, the sliding blocks', with their planes where
// the block's graph reads them. Callers hold g.mu.
func (g *devTier) bindRotaries(t map[composer.TableKey]backend.Buf, bs *blockScratch, cs backend.Buf, planes bool) {
	// A set of blocks rotating by the second table, on a model of mixed
	// geometry, holds that table in cs (scratchTable).
	first := 0
	if bs != nil && bs.p.MixedGeom && bs.p.LocalRope {
		first = 1
	}
	t[composer.TableKey{Table: composer.TableAngles, Instance: first}] = cs
	if bs == nil {
		return
	}
	for i := range bs.tableCount() {
		tab, pl, _ := g.tableBinds(bs, i)
		k := first + i
		if k > first {
			t[composer.TableKey{Table: composer.TableAngles, Instance: k}] = tab
		}
		if planes && pl != nil {
			t[composer.TableKey{Table: composer.TablePlanes, Instance: k}] = pl
		}
	}
}

// readsPlanes reports whether a form of g reads the rotary's planes: a
// pooled selection turning its blocks, or a rotary value at windows.
func readsPlanes(g *graph.Graph) bool {
	return slices.ContainsFunc(g.Ops, func(o graph.Op) bool {
		return o.Kind == graph.OpSelect && o.Attr.Pooled && o.Attr.NRot > 0 || o.Kind == graph.OpRopeTable
	})
}
