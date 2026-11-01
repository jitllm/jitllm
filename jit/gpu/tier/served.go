package tier

import (
	"errors"
	"fmt"
	"maps"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// A part of a model the composer serves in the tier's own submission: a
// whole block, its graph of generic ops the model's front end built from the
// container (nn.LayerPlan.Graph), and the head. Each is asked of the
// composer through its one keyed cache, and what it forms launches on the
// tier's own buffers -- the block's weights and vectors by role, its rotary
// table and its paged history.

type partKind uint8

const (
	partHead partKind = iota + 1
	// partBlock is a whole block, its graph the model's front end's.
	partBlock
	// partShadow is the shadow arm's comparison of a block's two outputs
	// (shadow.go).
	partShadow
	// partPrompt is a whole block at a prompt chunk's rows: one sequence's
	// positions, each row its own descriptor.
	partPrompt
	// partRope is a step's rotary table (nn.LayerPlan.Tables), written where
	// every block reads it.
	partRope
	// partSample is a step's greedy token over the head's logits
	// (nn.Head.Sample), written where the device's argmax writes it.
	partSample
	// partEmbed is a decode step's input row (nn.Head.Embed), gathered from
	// the tables the device holds; partEmbedRows a prompt chunk's input
	// rows, its rows at the chunk's.
	partEmbed
	partEmbedRows
	// partCopy is a block's output, written apart from x where its form
	// reads x in the kernel writing it, copied over x (graph.Copy).
	partCopy
	// partTower is a non-causal call's block whole: a picture's rows, one
	// run (towerserve.go).
	partTower
	// partRows is a picture's rows apart from its blocks: a tower's entry
	// or projector (picrows.go).
	partRows
	// partCollapse is the row a head reads from every residual stream
	// (collapse.go).
	partCollapse
	numParts
)

var partNames = [...]string{partHead: "head", partBlock: "block", partShadow: "shadow", partPrompt: "prompt block",
	partRope: "table", partSample: "sample", partEmbed: "input rows", partEmbedRows: "prompt input rows", partCopy: "copy",
	partTower: "tower block", partRows: "picture rows", partCollapse: "streams' collapse"}

// composedPart is a part of the model as the composer is asked for it: its
// graph of generic ops, the formats of the weights it binds, and the tier's
// buffers it binds them to -- weights and vectors by role, the step's tables
// by the composer's name.
type composedPart struct {
	kind partKind
	// part is what the composer is asked to form it as: a block's (the
	// zero), a table's or a sampling's.
	part    jit.PartKind
	graph   *graph.Graph
	formats map[jlm.Role]jlm.Type
	weights map[jlm.Role]*resident
	vectors map[jlm.Role]backend.Buf
	tables  map[string]backend.Buf
	// history, keys and page are a block's history layout and its step's
	// key bound and page.
	history    jit.History
	keys, page int
	// seqs is the sequences a step's rows are of (jit.Step.Seqs): 1 for a
	// form of one sequence's rows in position order, 0 for a row each.
	seqs int
	// paged is each bank a streamed block holds in pages and the sheets it
	// holds of it (layer.paged), nil for banks held whole.
	paged map[jlm.Role]int
	// apart says the graph's inputs and outputs are buffers of their own
	// (jit.Request.Apart).
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
	key  jit.Key
	refs int
	// A form over banks in pages (composedPart.paged) runs in two halves
	// (pagedRun): its codes before split, up to the selection, then the
	// rest, which read the bank. sel is the own buffer holding the
	// selection, of selN words, and table the own buffer the step's table
	// "experts" is written to.
	split, sel, selN, table int
	// A form over a history read in passes (jit.History.Passes) runs its
	// codes [passLo, passHi) once a pass (streamRun).
	passLo, passHi int
}

// partBuf is one of a form's own buffers: n bytes, holding words where it
// is a constant.
type partBuf struct {
	n     int
	words []uint32
}

// partCode is one of a form's kernels and its launch shape: groups at the
// form's rows, and its geometry at a call of fewer (jit.Code.LiveGroups).
type partCode struct {
	k             backend.Kernel
	groups, width int
	geom          jit.Code
	// matrix is a kernel formed on the device's matrix unit: one the
	// matrix pass formed whose tiles it multiplies, not the kernel that
	// writes its rows' factor for it. attend is an attention formed there.
	matrix, attend bool
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
	table  string
	// read and writes are whether the code reads and writes it
	// (jit.Binding.Read, Writes): a binding is not always both.
	read, writes bool
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
func jlmType(q kernels.Quant) (jlm.Type, bool) {
	switch q {
	case kernels.Float32:
		return jlm.TypeF32, true
	case kernels.Float16:
		return jlm.TypeF16, true
	case kernels.BFloat16:
		return jlm.TypeBF16, true
	}
	for _, t := range quant.PackedTypes {
		if k, ok := kernels.QuantOf(t); ok && k == q {
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
func (g *devTier) askPart(r *jit.Registry, sl *composedPart, rows int) (jit.Key, []jit.Code, error) {
	lanes, err := g.composedLanes()
	if err != nil {
		return 0, nil, err
	}
	g.Stats.ComposedLanes = lanes
	dev := jit.Device{Kind: jit.DeviceGPU, Slots: g.dev.Slots(), Lanes: lanes, Units: units(g.dev),
		Regs: threadRegs(g.dev), ExactDiv: kernels.DivExact(g.dev.API()), ExactArith: kernels.Gate2Exact(g.dev.API()),
		Matrix: g.composedMatrix(), HalfPairs: kernels.HalfPairs(g.dev.API()),
		VectorLoads: kernels.VectorLoads(g.dev.API()), Shared: sharedPerUnit(g.dev), Launch: g.launchWords()}
	if dev.Matrix[0] > 0 {
		dev.Fragment = backend.Fragment(g.dev)
		dev.ScaleRows = g.composedScale()
	}
	dev.Tile = g.matrixTile(r, dev, rows)
	if dev.Tile.Rows > 0 {
		g.Stats.MatrixTiled++
	}
	return r.Form(jit.Request{
		Graph:   sl.graph,
		Part:    sl.part,
		Formats: sl.formats,
		Paged:   sl.paged,
		Apart:   sl.apart,
		Step:    jit.Step{Rows: rows, Page: max(sl.page, 64), Keys: max(sl.keys, 1), ActWindow: g.actWin, Seqs: sl.seqs},
		Device:  dev,
		History: sl.history,
	})
}

// composedMatrix is the binary16 tile the composer forms a step of rows on
// (jit.Device.Matrix), asked of the device once by compiling a probe
// (backend.LowersTile): none where it does not lower one staged in
// workgroup memory and carried across a loop. The float reference
// (Config.ComposedFloat) states none: the tile multiplies binary16
// operands, and the reference is float arithmetic, not the forms it holds
// to it -- a matrix-unit attention the reference shared would move both
// alike, wrong or right.
func (g *devTier) composedMatrix() [3]int {
	if g.ComposedFloat {
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
// matrix tile's accumulator (jit.Device.ScaleRows), asked once.
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
func (g *devTier) compilePart(sl *composedPart, codes []jit.Code, rows int) (*partForm, error) {
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
	own := map[jit.Operand]int{}
	buf := func(n int, words []uint32) int {
		f.own = append(f.own, partBuf{n: max(n, 4), words: words})
		return len(f.own) - 1
	}
	// A scratch binding no kernel reads or writes -- a sum a pass carried in
	// registers, still a parameter -- is bound to a word: at a prompt's rows
	// the sub-block dots alone are gigabytes.
	live := map[jit.Operand]bool{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			if bd.Read || bd.Writes {
				live[bd.Operand] = true
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
		geom := jit.Code{Groups: c.Groups, Live: c.Live, LiveUnits: c.LiveUnits, LiveInner: c.LiveInner,
			LiveGroup: c.LiveGroup, Work: c.Work, Few: c.Few, Many: c.Many}
		f.codes = append(f.codes, partCode{k: kerns[ci], groups: c.Groups, width: c.GPU.Group[0], geom: geom,
			matrix: slices.Contains(c.Passes, "matrix") && slices.ContainsFunc(c.GPU.Ops, func(o ir.Op) bool { return o.Kind == ir.OpTileMMA || o.Kind == ir.OpMMA }),
			attend: slices.Contains(c.Passes, "flash")})
		var args []partArg
		for _, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == jit.OperandValue:
				if a, ok := io[o.Value]; ok {
					args = append(args, a)
					continue
				}
				i, ok := vals[o.Value]
				if !ok {
					i = buf(4*rows*gr.Values[o.Value].Width, nil)
					vals[o.Value] = i
				}
				args = append(args, partArg{kind: partOwn, i: i})
			case o.Kind == jit.OperandWeight && o.Region >= 0 && o.Region < 3:
				args = append(args, partArg{kind: partWeight, role: o.Role, region: o.Region})
			case o.Kind == jit.OperandVector:
				args = append(args, partArg{kind: partVector, role: o.Role})
			case o.Kind == jit.OperandTable && sl.tables[o.Name] != nil:
				args = append(args, partArg{kind: partTable, table: o.Name})
			case o.Kind == jit.OperandTable && o.Name == expertTable && sl.paged != nil:
				// The banks' page table, the form's own: written each step
				// before the codes that read it run (pagedRun).
				i, ok := own[o]
				if !ok {
					i = buf(4, nil)
					own[o] = i
				}
				f.own[i].n = max(f.own[i].n, int(bd.Bytes))
				args = append(args, partArg{kind: partOwn, i: i})
			case bd.Words != nil || o.Kind == jit.OperandScratch:
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
	if sl.history.Passes {
		if err := f.passHalves(); err != nil {
			f.close()
			return nil, err
		}
	}
	f.share(sl.paged != nil)
	return f, nil
}

// passHalves finds the codes a form over a history read in passes runs once
// a pass: from its first code reading the history (its keys or values) to
// its last folding the running partial. A code before it -- a front fused
// with the row's own append, which addresses the pages but reads no history
// -- runs once. A pass writes none of the call's buffers, no table but the
// pools and the partial, and no buffer a code of the pass reads before.
func (f *partForm) passHalves() error {
	f.passLo, f.passHi = -1, -1
	for ci, args := range f.args {
		for _, a := range args {
			if a.kind != partTable {
				continue
			}
			switch a.table {
			case "kv.k", "kv.v":
				if a.read && f.passLo < 0 {
					f.passLo = ci
				}
			case "run.out":
				if a.writes {
					f.passHi = ci + 1
				}
			}
		}
	}
	if f.passLo < 0 || f.passHi <= f.passLo {
		return fmt.Errorf("a form over a history read in passes with no pass between codes %d and %d", f.passLo, f.passHi)
	}
	read := map[int]bool{}
	for ci := f.passLo; ci < f.passHi; ci++ {
		for _, a := range f.args[ci] {
			if a.kind == partOwn && a.read {
				read[a.i] = true
			}
		}
		for _, a := range f.args[ci] {
			switch {
			case !a.writes:
			case a.kind == partIn || a.kind == partOut,
				a.kind == partTable && a.table != "run.out" && a.table != "kv.k" && a.table != "kv.v",
				a.kind == partOwn && read[a.i]:
				return fmt.Errorf("the composer's kernel %d of a pass writes what a pass cannot run again over", ci)
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
// halves, keep buffers of their own. Callers form f outside any step.
func (f *partForm) share(paged bool) {
	n := len(f.own)
	first, last := make([]int, n), make([]int, n)
	fresh := make([]bool, n)
	for i := range first {
		first[i] = -1
	}
	for ci, args := range f.args {
		for _, a := range args {
			if a.kind != partOwn {
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
		if f.passHi > 0 && first[i] >= 0 && first[i] < f.passLo && last[i] >= f.passLo {
			last[i] = max(last[i], f.passHi-1)
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

// expertTable is the step's table of a bank held in pages, as the composer
// names it (jit.Request.Paged).
const expertTable = "experts"

// pagedHalves finds where a form over banks in pages is cut (partForm.split):
// at its first code that binds a bank, with the selection those codes read
// written by a code before it, and the page table read after it alone.
func (f *partForm) pagedHalves(gr *graph.Graph, vals map[graph.ValueID]int, own map[jit.Operand]int, rows int) error {
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
		if o.Kind == jit.OperandTable && o.Name == expertTable {
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
				if r := sl.weights[a.role]; r != nil {
					b = [3]backend.Buf{r.qs, r.d, r.sc}[a.region]
				}
			}
			if b == nil {
				return matrix, fmt.Errorf("the composer's kernel %d reads %v, which the call site does not hold", i, a.role)
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
func (g *devTier) compileCodes(codes []jit.Code) ([]backend.Kernel, error) {
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
		// prompt map too (streamServed at one row).
		g.eachLane(func(l *lane) {
			if l.bs != nil {
				g.dropPromptInst(l.bs, sv.f)
			}
			for _, bs := range l.bbs {
				g.dropPromptInst(bs, sv.f)
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
		g.LastErr = "the composed prompt block: " + err.Error()
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
	}
}

// everyRow is the device's word stating more rows than any form holds: a
// part bound to it runs its form's every row (jit.CallRows).
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
		if sk.kind != partPrompt {
			var err error
			if in, err = g.instance(sv.f); err != nil {
				return fmt.Errorf("the composed %s: %w", partNames[sk.kind], err)
			}
		}
		sv.f.refs++
		if nl.served == nil {
			nl.served = map[servedKey]*servedPart{}
		}
		nl.served[sk] = &servedPart{sl: sv.sl.clone(), f: sv.f, inst: in, args: sv.f.launchArgs(), n: sv.n, m: sv.m}
	}
	return nil
}

// Parts names the parts of a model the composer serves, each counted apart
// (Stats.ServedPart).
type Parts uint16

const (
	ComposedHead Parts = 1 << partHead
	// ComposedBlock serves a block whole from its graph (nn.LayerPlan.Graph).
	ComposedBlock Parts = 1 << partBlock
	// PartShadow selects the shadow arm's comparisons in Stats.ServedPart:
	// Config.ComposedShadow is its switch.
	PartShadow Parts = 1 << partShadow
	// ComposedPrompt serves a prompt chunk's blocks whole at the chunk's
	// rows: a form a width, as the tier keeps a scratch a width.
	ComposedPrompt Parts = 1 << partPrompt
	// ComposedTable serves a call's rotary tables from their graphs
	// (nn.LayerPlan.Tables) at its rows' rotary positions, a step's, a
	// ragged step's or a prompt chunk's.
	ComposedTable Parts = 1 << partRope
	// ComposedSample serves a one-row step's greedy token from its graph
	// (nn.Head.Sample).
	ComposedSample Parts = 1 << partSample
	// ComposedEmbed serves the input rows from their graph (nn.Head.Embed),
	// a decode step's and a prompt chunk's, gathered on the device the
	// caller promises them to (nn.EmbedDevice).
	ComposedEmbed Parts = 1 << partEmbed
	// PromptEmbed counts what ComposedEmbed serves of a prompt chunk's input
	// rows (Stats.ServedPart), apart from a decode step's.
	PromptEmbed Parts = 1 << partEmbedRows
	// PartCopy selects in Stats.ServedPart the copies of block outputs
	// written apart from x (servedCopy): a block the composer serves asks
	// for one where it needs it.
	PartCopy Parts = 1 << partCopy
	// ComposedTower serves a non-causal call's blocks whole from their
	// graphs: a vision tower's, over a picture's rows (towerserve.go).
	ComposedTower Parts = 1 << partTower
	// ComposedRows serves a picture's rows apart from its blocks from their
	// graphs: a tower's entry and projector (nn.RowsPart, picrows.go).
	ComposedRows Parts = 1 << partRows
	// CollapsedHead counts the collapse of every stream a head reads, which
	// the composer serves where it serves the head (ComposedHead).
	CollapsedHead Parts = 1 << partCollapse
	// ComposedStep is every part of a step: its input rows, tables, blocks
	// (a prompt chunk's and a picture's among them), head and token.
	ComposedStep = ComposedEmbed | ComposedTable | ComposedBlock | ComposedPrompt | ComposedTower | ComposedRows | ComposedHead | ComposedSample
)

// partDecline is why a decode step's or a prompt chunk's composed part is
// not asked for: what the serving conditions leave to the tier's own
// kernels, counted by reason (Stats.BlockDeclines, HeadDeclines).
type partDecline = jit.Decline

const (
	partServes       = jit.DeclineServes
	partNoGraph      = jit.DeclineNoGraph
	partPersist      = jit.DeclinePersist
	partSkip         = jit.DeclineSkip
	partNoPool       = jit.DeclineNoPool
	partPrefill      = jit.DeclinePrefill
	partStream       = jit.DeclineStream
	partFeeds        = jit.DeclineFeeds
	partWindow       = jit.DeclineWindow
	partHeadShape    = jit.DeclineHeadShape
	partHeadNorm     = jit.DeclineHeadNorm
	partPick         = jit.DeclinePick
	partCaller       = jit.DeclineCaller
	partNoRoom       = jit.DeclineNoRoom
	partHeadBias     = jit.DeclineHeadBias
	partHeadOther    = jit.DeclineHeadOther
	partNoTable      = jit.DeclineNoTable
	partTableHeld    = jit.DeclineTableHeld
	partNoAxes       = jit.DeclineNoAxes
	partNoSample     = jit.DeclineNoSample
	partNoEmbed      = jit.DeclineNoEmbed
	partEmbedFormat  = jit.DeclineEmbedFormat
	partEmbedWeight  = jit.DeclineEmbedWeight
	partEmbedToken   = jit.DeclineEmbedToken
	partExperts      = jit.DeclineExperts
	partExpertsHost  = jit.DeclineExpertsHost
	partExpertsRows  = jit.DeclineExpertsRows
	partEmbedStreams = jit.DeclineEmbedStreams
	partRecurrent    = jit.DeclineRecurrent
	partNoWeight     = jit.DeclineNoWeight
	numPartDeclines  = jit.NumDeclines
)

// DeclineName is the reason a decline count is kept under, by its index in
// Stats.BlockDeclines or HeadDeclines (jit.DeclineName).
func DeclineName(i int) string { return jit.DeclineName(i) }

// ServedPart is how many steps the composer served the parts of k in, how
// many steps where they serve it was not asked for, and how many other block
// steps asked it for nothing (Stats.PartsServed, PartsDeclined,
// PartsUnasked).
func (s Stats) ServedPart(k Parts) (served, declined, unasked int) {
	for i := range s.PartsServed {
		if k&(1<<i) != 0 {
			served += s.PartsServed[i]
			declined += s.PartsDeclined[i]
			unasked += s.PartsUnasked[i]
		}
	}
	return served, declined, unasked
}

// FormedPart is how many of the parts of k formed at a step rather than
// at a placement (Stats.FormedAtStepParts).
func (s Stats) FormedPart(k Parts) int {
	n := 0
	for i := range s.FormedAtStepParts {
		if k&(1<<i) != 0 {
			n += s.FormedAtStepParts[i]
		}
	}
	return n
}

// AttendPart is how many launches of the parts of k ran an attention formed
// on the device's matrix unit (Stats.AttendLaunches).
func (s Stats) AttendPart(k Parts) int {
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
func (s Stats) MatrixPart(k Parts) int {
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
func (s Stats) ReplayedPart(k Parts) int {
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
	kind       partKind
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
}

// partIO is the most inputs, and the most outputs, a served part binds: a
// block of four residual streams reads them and the rows' per-layer inputs.
const partIO = 8

// servedPart is a served part and its form, with the arguments its launch
// binds, reused every step: its own instance, or for a prompt part none,
// its instance its scratch's (promptInst).
type servedPart struct {
	sl      composedPart
	f       *partForm
	inst    *partInst
	args    [][]backend.Buf
	in, out [partIO]backend.Buf
	n, m    int
	// rows, where set, is the call's rows, the word its table states: a
	// launch runs their groups, not its form's every row's.
	rows int
}

// errServedCold is a served part asked for inside a capture before it
// formed: the capture is abandoned and the step issued unrecorded.
var errServedCold = errors.New("tier: a served part forms outside a capture")

// errFormedEarlier marks a part whose form failed at an earlier step
// (servedFor), its error stated once.
var errFormedEarlier = errors.New("did not form earlier")

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
		g.formErr[sk] = fmt.Errorf("the composed %s %w: %w", partNames[sk.kind], errFormedEarlier, err)
		g.LastErr = err.Error()
		return err
	}
	defer g.subUnlock(g.subLock())
	if g.servedForms == nil {
		g.servedForms = map[jit.Key]*partForm{}
	}
	if !g.prewarming {
		g.FormedAtStep++
		g.FormedAtStepParts[sk.kind]++
		g.LastErr = fmt.Sprintf("the composed %s formed at %d rows outside a placement (keys %d, seqs %d, stream %v, held %d, clone lane %v)",
			partNames[sk.kind], sk.rows, sk.keys, sk.seqs, sk.stream, sk.held, g.lane.clone)
	}
	if g.composer == nil {
		g.composer = jit.Default()
		g.rewriteComposer(g.composer)
		// Poisoned scratch poisons the forms' workgroup arrays too: a kernel
		// reading one it never wrote changes the logits.
		if g.kb.poisonScratch {
			g.composer.PoisonWorkgroupMemory()
		}
		if g.kb.dropSegments {
			g.composer.DropSegments(jit.DeviceGPU)
		}
	}
	r := g.composer
	if g.ComposedFloat || g.ComposedUnsplit {
		if g.composerFloat == nil {
			g.composerFloat = g.composer
			if g.ComposedFloat {
				g.composerFloat = g.composerFloat.WithoutInt8(jit.DeviceGPU)
			}
			if g.ComposedUnsplit {
				g.composerFloat = g.composerFloat.WithoutSplits(jit.DeviceGPU)
			}
		}
		r = g.composerFloat
	}
	sl, err := build()
	if err != nil {
		return nil, fail(err)
	}
	// A pass's attention leaves its maximum, sum and weighted values for the
	// fold (jit.Registry.ForPasses).
	if sl.history.Passes {
		if r != g.composer {
			r = r.ForPasses(jit.DeviceGPU)
		} else {
			if g.composerPasses == nil {
				g.composerPasses = g.composer.ForPasses(jit.DeviceGPU)
			}
			r = g.composerPasses
		}
	}
	// A form of several rows reads the rows it runs from the call's word: a
	// part whose call states none runs its form's every row.
	if _, ok := sl.tables["rows.n"]; !ok {
		every, err := g.everyRow()
		if err != nil {
			return nil, fail(err)
		}
		if sl.tables == nil {
			sl.tables = map[string]backend.Buf{}
		}
		sl.tables["rows.n"] = every
	}
	key, codes, err := g.askPart(r, &sl, sk.rows)
	if err != nil {
		return nil, fail(fmt.Errorf("the composed %s: %w", partNames[sk.kind], err))
	}
	f, ok := g.servedForms[key]
	if !ok {
		if f, err = g.compilePart(&sl, codes, sk.rows); err != nil {
			return nil, fail(fmt.Errorf("the composed %s: %w", partNames[sk.kind], err))
		}
		g.PartsFormed++
		g.PartKernels += len(codes)
		f.key = key
		g.servedForms[key] = f
	}
	var in *partInst
	// A form over a history read in passes is laid on its scratch's arena
	// (streamPart).
	if sk.kind != partPrompt && !sk.stream {
		if in, err = g.instance(f); err != nil {
			if f.refs == 0 {
				delete(g.servedForms, key)
				f.close()
			}
			err = fmt.Errorf("the composed %s: %w", partNames[sk.kind], err)
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
	if len(sl.graph.In) > partIO || len(sl.graph.Out) > partIO {
		return nil, fail(fmt.Errorf("a part of %d inputs and %d outputs, past the %d a served part binds", len(sl.graph.In), len(sl.graph.Out), partIO))
	}
	f.refs++
	sv := &servedPart{sl: sl, f: f, inst: in, args: f.launchArgs(), n: len(sl.graph.In), m: len(sl.graph.Out)}
	g.served[sk] = sv
	if sk.kind == partBlock && !sk.stream && sk.graph.Block >= 0 {
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
func (sv *servedPart) runAs(g *devTier, lc *launcher, inst *partInst, kind partKind, in, out []backend.Buf) error {
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	matrix, err := sv.launch(lc, inst, sv.in[:sv.n], sv.out[:sv.m])
	if err != nil {
		return err
	}
	g.PartsServed[kind]++
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
func (sv *servedPart) pagedRun(g *devTier, lc *launcher, s backend.Session, inst *partInst, kind partKind, l *layer, nExpert int,
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
		st.sel, st.slotSel = st.rsel, growU32(st.slotSel, live)
		defer func() { st.sel, st.slotSel = one, st.slotSel[:k] }()
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
	if g.kb.pageFault == PageFaultSwap && len(st.sel) > 1 {
		a, b := st.sel[0], st.sel[1]
		st.tab[a], st.tab[b] = st.tab[b], st.tab[a]
	}
	if err := s.WriteAt(inst.bufs[f.table], 0, u32b(st.tab)); err != nil {
		return err
	}
	m1, err := sv.launchCodes(lc, inst, sv.in[:sv.n], sv.out[:sv.m], f.split, len(f.codes))
	if err != nil {
		return err
	}
	g.PartsServed[kind]++
	g.MatrixLaunches[kind] += m0.matrix + m1.matrix
	g.AttendLaunches[kind] += m0.attend + m1.attend
	return nil
}

// headSpec is a head graph the tier serves: a norm (an RMSNorm, or a
// LayerNorm with its optional bias), the projection to the vocabulary with
// its optional bias, and a chain of elementwise ops on its logits, whatever
// they compute.
type headSpec struct {
	norm, normBias, out, bias       jlm.Role
	layerNorm, hasNormBias, hasBias bool
}

// headShape is gr read as a head the tier serves, false for any other graph.
// Its outputs are the logits, then the norm's rows where it writes them too
// (graph.Head.Hidden).
func headShape(gr *graph.Graph) (headSpec, bool) {
	var hs headSpec
	if gr == nil || len(gr.In) != 1 || len(gr.Out) < 1 || len(gr.Out) > 2 || len(gr.Ops) < 2 {
		return hs, false
	}
	if len(gr.Out) == 2 && (len(gr.Ops[0].Out) != 1 || gr.Out[1] != gr.Ops[0].Out[0]) {
		return hs, false
	}
	n, mv := gr.Ops[0], gr.Ops[1]
	switch {
	case n.Kind == graph.OpNorm && len(n.Weight) == 1:
	case n.Kind == graph.OpLayerNorm && (len(n.Weight) == 1 || len(n.Weight) == 2):
		hs.layerNorm = true
	default:
		return hs, false
	}
	if len(n.In) != 1 || n.In[0] != gr.In[0] || len(n.Out) != 1 ||
		mv.Kind != graph.OpMatVec || len(mv.In) != 1 || mv.In[0] != n.Out[0] || len(mv.Out) != 1 ||
		len(mv.Weight) < 1 || len(mv.Weight) > 2 {
		return hs, false
	}
	hs.norm, hs.out = n.Weight[0].Role, mv.Weight[0].Role
	if len(n.Weight) == 2 {
		hs.normBias, hs.hasNormBias = n.Weight[1].Role, true
	}
	if len(mv.Weight) == 2 {
		hs.bias, hs.hasBias = mv.Weight[1].Role, true
	}
	last := mv.Out[0]
	for _, o := range gr.Ops[2:] {
		if o.Kind != graph.OpMap || len(o.Weight) != 0 || len(o.In) != 1 || o.In[0] != last || len(o.Out) != 1 {
			return hs, false
		}
		last = o.Out[0]
	}
	return hs, gr.Out[0] == last
}

// headServes is why a step's head is not the one the composer is
// asked for, serves where it is: the model's head graph (headShape) -- a
// norm, the vocabulary's projection with its bias, and what the graph does to
// the logits after it, the final softcap and the scale -- with the norm and
// the vectors it binds the tier's, not the label pick, not inside the
// persistent block.
func (bs *blockScratch) headServes(norm backend.Buf, pickOnly, persist bool) partDecline {
	hs, ok := headShape(bs.headGraph)
	hh, hok := headShape(bs.headHidden)
	switch {
	case !ok || !hok || hh != hs || len(bs.headHidden.Out) != 2:
		return partHeadShape
	case norm == nil || hs.layerNorm != (bs.normVar != nil) || hs.hasNormBias && bs.hNormB == nil:
		return partHeadNorm
	case hs.hasBias && bs.hBias == nil:
		return partHeadBias
	case pickOnly:
		return partPick
	case persist:
		return partPersist
	}
	return partServes
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
	if rowsFaulted("nohidden") {
		gr = bs.headGraph
	}
	hs, ok := headShape(gr)
	if !ok {
		return nil, errors.New("the composed head: a graph that is not a norm, a projection and elementwise ops")
	}
	t, ok := jlmType(w.t)
	if !ok {
		return nil, fmt.Errorf("the composed head: format %d has no container type", w.t)
	}
	bind := func(sl *composedPart) {
		sl.weights[hs.out], sl.vectors[hs.norm] = w, norm
		if hs.hasNormBias {
			sl.vectors[hs.normBias] = bs.hNormB
		}
		if hs.hasBias {
			sl.vectors[hs.bias] = bs.hBias
		}
	}
	sk := servedKey{kind: partHead, graph: gr, rows: rows, t: t}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		sl := composedPart{kind: partHead, graph: gr, formats: map[jlm.Role]jlm.Type{hs.out: t},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{}}
		if hs.hasBias {
			sl.formats[hs.bias] = jlm.TypeF32
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
// (nn.Head.PickHead): the head's norm and its weight bound as servedHead
// binds them, the ids its second input. One form for the head's label bound,
// the labels a run-time input.
func (g *devTier) servedPickHead(bs *blockScratch, norm backend.Buf) (*servedPart, error) {
	gr, w := bs.pickHead, bs.head
	if gr == nil || len(gr.In) != 2 || len(gr.Out) != 1 || len(gr.Ops) < 2 || len(gr.Ops[0].Weight) < 1 ||
		gr.Ops[1].Kind != graph.OpMatVec || len(gr.Ops[1].In) != 2 || len(gr.Ops[1].Weight) != 1 {
		return nil, errors.New("the composed pick head: a graph that is not a norm and a row-selected projection")
	}
	t, ok := jlmType(w.t)
	if !ok {
		return nil, fmt.Errorf("the composed pick head: format %d has no container type", w.t)
	}
	n, out := gr.Ops[0].Weight, gr.Ops[1].Weight[0].Role
	bind := func(sl *composedPart) {
		sl.weights[out], sl.vectors[n[0].Role] = w, norm
		if len(n) == 2 {
			sl.vectors[n[1].Role] = bs.hNormB
		}
	}
	sk := servedKey{kind: partHead, graph: gr, rows: 1, t: t}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		sl := composedPart{kind: partHead, graph: gr, formats: map[jlm.Role]jlm.Type{out: t},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{}}
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
func tableName(gr *graph.Graph) string {
	if gr == nil || len(gr.Out) != 1 {
		return ""
	}
	return gr.Values[gr.Out[0]].Name
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
func (g *devTier) servedTable(gr *graph.Graph, R int, cs, planes, pos backend.Buf) (*servedPart, error) {
	sk := servedKey{kind: partRope, graph: gr, rows: R}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		name := tableName(gr)
		if !strings.HasPrefix(name, "rope.cs") {
			return composedPart{}, fmt.Errorf("the composed table writes %q, which is not a rotary table", name)
		}
		pn := "rope.pos"
		if tableAxes(gr) {
			pn = "rope.pos.axes"
		}
		pl := "rope.planes" + strings.TrimPrefix(name, "rope.cs")
		return composedPart{kind: partRope, part: jit.PartTable, graph: gr, formats: map[jlm.Role]jlm.Type{},
			tables: map[string]backend.Buf{name: cs, pl: planes, pn: pos}}, nil
	})
	if err != nil {
		return nil, err
	}
	// Bound again every step, by the names the build gave them.
	for k := range sv.sl.tables {
		switch {
		case strings.HasPrefix(k, "rope.cs"):
			sv.sl.tables[k] = cs
		case strings.HasPrefix(k, "rope.planes"):
			sv.sl.tables[k] = planes
		default:
			sv.sl.tables[k] = pos
		}
	}
	return sv, nil
}

// tableServes is why the tables of a call of R rows (rag's, a ragged step's)
// at cache position pos are not the ones the composer is asked for, serves
// where they are: a graph for each table bs holds (scratchTable), as wide as
// its rows, the planes and positions each reads, and over several axes each
// row's positions on them, which the caller states for the call (callAxes).
// Callers hold g.mu.
func (g *devTier) tableServes(bs *blockScratch, p *nn.LayerPlan, pos, R int, rag *ragStep) partDecline {
	if g.RopeTableHost {
		return partCaller
	}
	for i := range bs.tableCount() {
		gr := scratchTable(bs, p, i)
		if gr == nil || gr.Values[gr.Out[0]].Width != bs.p.RopeW() {
			return partNoTable
		}
		cs, planes, at := g.tableBinds(bs, i)
		switch {
		case cs == nil || planes == nil || at == nil:
			return partTableHeld
		case tableAxes(gr) && g.callAxes(pos, R, rag) == nil:
			return partNoAxes
		}
	}
	return partServes
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
// geometry of a split model (scratchPlanOf) holds its own layers' table in
// cs: the global set the plan's first, the sliding set its second.
func scratchTable(bs *blockScratch, p *nn.LayerPlan, i int) *graph.Graph {
	if bs.p.GeomSplit {
		switch {
		case i > 0 || len(p.Tables) != 2:
			return nil
		case bs.p.Sliding:
			return p.Tables[1]
		}
		return p.Tables[0]
	}
	if len(p.Tables) != bs.tableCount() || i >= len(p.Tables) {
		return nil
	}
	return p.Tables[i]
}

// callAxes is the rotary positions the caller states for a call of R rows at
// cache position pos, four a row (nn.RopeAxesDevice), nil where it states
// none for this call. A ragged step's are stated at its first row's.
func (g *devTier) callAxes(pos, R int, rag *ragStep) []int32 {
	if rag != nil && len(rag.pos) > 0 {
		pos = rag.pos[0]
	}
	// A call the tier cuts into narrower ones takes its rows' positions from
	// where its first row sits in the caller's.
	off := pos - g.ropeAxesAt
	if g.ropeAxes == nil || off < 0 || rag != nil && off != 0 || len(g.ropeAxes) < 4*(off+R) {
		return nil
	}
	return g.ropeAxes[4*off : 4*(off+R)]
}

// spendAxes drops the caller's rotary positions once its call has run their
// last row, at the call's end (layersFor, LayersRows): a later call states
// its own or none. Callers hold g.mu.
func (g *devTier) spendAxes(pos, R int, rag *ragStep) {
	if rag != nil && len(rag.pos) > 0 {
		pos = rag.pos[0]
	}
	if rag != nil || pos+R >= g.ropeAxesAt+len(g.ropeAxes)/4 {
		g.ropeAxes = nil
	}
}

// tableBinds is what table i of bs writes and reads: the global table cs or
// the local one, its planes, and its positions (rope.pos, or over several
// axes rope.pos.axes).
func (g *devTier) tableBinds(bs *blockScratch, i int) (cs, planes, pos backend.Buf) {
	if i > 0 {
		return bs.csSWA, bs.ropeTabSWA, bs.rpos
	}
	if gr := scratchTable(bs, &bs.p, 0); gr != nil && tableAxes(gr) {
		return bs.cs, bs.ropeTabAxes, bs.rposAxes
	}
	return bs.cs, bs.ropeTab, bs.rpos
}

// formTables forms bs's tables at its rows where the composer serves them,
// at a placement or where a width's scratch is built: a table met first at
// a step formed inside it. Callers hold g.mu.
func (g *devTier) formTables(bs *blockScratch) {
	if g.NoPromptPrewarm || bs == nil || bs.cs == nil || bs.p.NRot <= 0 || bs.p.NonCausal {
		return
	}
	for i := range bs.tableCount() {
		gr := scratchTable(bs, &bs.p, i)
		cs, planes, at := g.tableBinds(bs, i)
		if gr == nil || cs == nil || planes == nil || at == nil {
			continue
		}
		if _, err := g.servedTable(gr, bs.rows, cs, planes, at); err != nil {
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

// embedServes is why a step's or a prompt chunk's input rows are not the
// ones the composer is asked for, serves where they are: the front end's
// graph of the input rows, each of whose weights is a table the device holds
// in a format the composer reads -- the token embedding (the head where it is
// the embedding, nn.Head.TokEmbd otherwise) and a learned position table
// (nn.Head.PosEmbd).
func (bs *blockScratch) embedServes() partDecline {
	if bs.embedGraph == nil {
		return partNoEmbed
	}
	if bs.embWhy != partServes {
		return bs.embWhy
	}
	for _, o := range bs.embedGraph.Ops {
		for _, w := range o.Weight {
			if w.Rows == 0 {
				if bs.embSV[w.Role] == nil {
					return partEmbedWeight
				}
				continue
			}
			r := bs.embTable(w.Role)
			if r == nil || !r.ok {
				return partEmbedWeight
			}
			if _, ok := jlmType(r.t); !ok {
				return partEmbedFormat
			}
		}
	}
	return partServes
}

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
	sk := servedKey{kind: partEmbed, graph: gr, rows: R, t: t}
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
		cp := composedPart{kind: partEmbed, part: jit.PartEmbed, graph: g.embedFault(gr), formats: formats, weights: weights,
			vectors: vectors, tables: map[string]backend.Buf{"tokens": bs.embTok}}
		if bs.posW != nil {
			cp.tables["rope.pos"] = bs.embPos
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
	sv.sl.tables["tokens"] = bs.embTok
	if _, ok := sv.sl.tables["rope.pos"]; ok {
		sv.sl.tables["rope.pos"] = bs.embPos
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
	if gr == nil || !(len(gr.In) == 1 && len(gr.Out) == 1 || len(gr.In) == 3 && len(gr.Out) == 2 || len(gr.In) == 2 && len(gr.Out) == 1) {
		return nil, errors.New("the composed sample: a graph that is not logits in and a token out, the sampler's candidates or the label pick")
	}
	sk := servedKey{kind: partSample, graph: gr, rows: rows}
	return g.servedFor(sk, func() (composedPart, error) {
		return composedPart{kind: partSample, part: jit.PartSample, graph: gr, formats: map[jlm.Role]jlm.Type{}}, nil
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
// streamsInPlace says a form of n residual streams writes each stream's
// output over its input (each stream one view of x): a form that does not
// has no copy to run through, and is refused by name.
func (f *partForm) streamsInPlace(n int) error {
	if n < 2 {
		return nil
	}
	for k := range n {
		if !f.inPlace(k, k) {
			return fmt.Errorf("tier: a block of %d residual streams writes stream %d before it last reads it", n, k)
		}
	}
	return nil
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
	switch r {
	case jlm.RoleAttnQ:
		return l.wq
	case jlm.RoleAttnK:
		return l.wk
	case jlm.RoleAttnV:
		return l.wv
	case jlm.RoleAttnOut:
		return l.wo
	case jlm.RoleFFNGate:
		return l.gate
	case jlm.RoleFFNUp:
		return l.up
	case jlm.RoleFFNDown:
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
		if l.mla {
			return l.mlaGate
		}
	case jlm.RoleAttnQKV:
		if l.withAttn {
			return l.ssmIn
		}
		if l.linear {
			return l.wq
		}
	case jlm.RoleSCX:
		if l.linear {
			return l.wq
		}
	case jlm.RoleFFNRoutedDown:
		return l.routedDown
	case jlm.RoleFFNRoutedUp:
		return l.routedUp
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
	case jlm.RolePLEGate:
		return l.pleGate
	case jlm.RolePLEProj:
		return l.pleProj
	case jlm.RoleStreamRouter:
		return l.altRouterW
	case jlm.RoleLowRankIn:
		return l.laurelL
	case jlm.RoleLowRankOut:
		return l.laurelR
	case jlm.RoleHCAttnDown:
		return l.hcAD
	case jlm.RoleHCAttnUp:
		return l.hcAU
	case jlm.RoleHCAttnInject:
		return l.hcAI
	case jlm.RoleHCFFNDown:
		return l.hcFD
	case jlm.RoleHCFFNUp:
		return l.hcFU
	case jlm.RoleHCFFNInject:
		return l.hcFI
	case jlm.RoleIdxQ:
		return l.idxQ
	case jlm.RoleIdxQB:
		return l.idxQB
	case jlm.RoleIdxK:
		return l.idxK
	case jlm.RoleIdxProj:
		return l.idxProj
	}
	return nil
}

// expertsServe is why block l's routed experts are not where its form
// reads them, serves where they are: every bank its graph binds held whole
// on the card, or a streamed block's in pages -- its cache's slots or its
// compact bank, each expert reached through the step's table (pagedRun) --
// at a call of no more rows than its pages hold the selections of
// (pagedRows), its experts sent to the card rather than run on the host.
func (l *layer) expertsServe(rows int) partDecline {
	held, banks := true, false
	for _, o := range l.graph.Ops {
		for _, w := range o.Weight {
			if jlm.ExpertBank(w.Role) {
				r := l.roleWeight(w.Role)
				banks, held = true, held && r != nil && w.Rows > 0 && r.nrows%w.Rows == 0
			}
		}
	}
	switch {
	case !banks:
		return partServes
	case l.stream != nil && l.stream.hybrid:
		return partExpertsHost
	case !held:
		return partExperts
	case l.stream != nil && rows > l.pagedRows():
		return partExpertsRows
	}
	return partServes
}

// pagedCall is the rows a composed call over blocks [lo, hi) takes at a
// time, at most w: no more than every streamed block's pages hold.
func (g *devTier) pagedCall(lo, hi, w int) int {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.stream != nil && !l.stream.hybrid {
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
	for _, o := range l.graph.Ops {
		for _, w := range o.Weight {
			if l.roleVector(w.Role) == nil && (w.Rows == 0 || l.roleWeight(w.Role) == nil) {
				return false
			}
		}
	}
	return true
}

// roleVector is the block's vector its graph binds under role, nil for one
// the tier does not hold.
func (l *layer) roleVector(r jlm.Role) backend.Buf {
	switch r {
	case jlm.RoleAttnNorm:
		return l.nAttn
	case jlm.RoleFFNNorm:
		return l.nFFN
	case jlm.RoleAttnNormBias:
		return l.nAttnB
	case jlm.RoleFFNNormBias:
		return l.nFFNB
	case jlm.RolePostAttnNorm:
		return l.nPostAttn
	case jlm.RolePostFFNNorm:
		return l.nPostFFN
	case jlm.RoleAttnQNorm:
		return l.nQ
	case jlm.RoleAttnKNorm:
		return l.nK
	case jlm.RoleIdxQNorm:
		return l.nIdxQ
	case jlm.RoleIdxKNorm:
		return l.nIdxK
	case jlm.RoleIdxKNormB:
		return l.nIdxKB
	case jlm.RoleAttnQBias:
		return l.mvq.bias
	case jlm.RoleAttnKBias:
		return l.mvk.bias
	case jlm.RoleAttnVBias:
		return l.mvv.bias
	case jlm.RoleAttnOutBias:
		return l.mvo.bias
	case jlm.RoleFFNUpBias:
		return l.mvu.bias
	case jlm.RoleFFNDownBias:
		return l.mvd.bias
	case jlm.RoleVFFNGateBias:
		// A tower's gated MLP's gate bias, which no text role names.
		return l.mvg.bias
	case jlm.RoleVClamp:
		return l.clampB
	case jlm.RoleAttnOutNorm, jlm.RoleAttnOutNormBias:
		// A post-norm block's norm over x + attention is the one the tier
		// holds as its MLP's (nn.LayerPlan.PostResidNorm).
		if l.nResid == nil {
			return nil
		}
		if r == jlm.RoleAttnOutNorm {
			return l.nFFN
		}
		return l.nFFNB
	case jlm.RoleLayerOutNorm:
		return l.nResid
	case jlm.RoleLayerOutNormBias:
		return l.nResidB
	case jlm.RoleAttnSinks:
		return l.sinks
	case jlm.RoleAttnSubNorm:
		return l.nSubAttn
	case jlm.RoleFFNSubNorm:
		return l.nSubFFN
	case jlm.RoleXIELU:
		return l.xielu
	case jlm.RoleLayerOutScale:
		return l.outScale
	case jlm.RoleAttnQANorm:
		return l.nQA
	case jlm.RoleAttnKVANorm:
		return l.nKVA
	case jlm.RoleRouterBias:
		return l.routerB
	case jlm.RoleExpGateBias:
		return l.expGateB
	case jlm.RoleExpUpBias:
		return l.expUpB
	case jlm.RoleExpDownBias:
		return l.expDownB
	case jlm.RoleExpProbsB:
		return l.expSelB
	case jlm.RoleShRouter:
		return l.shRouter
	case jlm.RoleSSMA:
		return l.ssmA
	case jlm.RoleSSMDtBias:
		return l.ssmDt
	case jlm.RoleSSMNorm:
		return l.ssmNorm
	case jlm.RoleSSMD:
		return l.ssmD
	case jlm.RoleSSMConvBias:
		return l.ssmCB
	case jlm.RoleSSMDtNorm:
		return l.m1DtN
	case jlm.RoleSSMBNorm:
		return l.m1BN
	case jlm.RoleSSMCNorm:
		return l.m1CN
	case jlm.RolePLEPostNorm:
		return l.nPLE
	case jlm.RoleStreamRouterNorm:
		return l.altRouterNorm
	case jlm.RoleStreamPredictCoef:
		return l.altPredT
	case jlm.RoleStreamCorrectCoef:
		return l.altCorrT
	case jlm.RoleStreamCorrectScale:
		return l.altCorrScale
	case jlm.RoleLowRankNorm:
		return l.nLaurel
	case jlm.RoleFFNNorm2:
		return l.nFFN2
	case jlm.RolePostFFNNorm1:
		return l.nPost1
	case jlm.RolePostFFNNorm2:
		return l.nPost2
	case jlm.RoleRouterNorm:
		return l.nRouter
	case jlm.RoleExpScale:
		return l.expScale
	case jlm.RoleAttnResScore, jlm.RoleFFNResScore, jlm.RoleFFNRoutedNorm:
		if l.k3 == nil {
			return nil
		}
		switch r {
		case jlm.RoleAttnResScore:
			return l.k3.resA
		case jlm.RoleFFNResScore:
			return l.k3.resF
		}
		return l.k3.rnorm
	case jlm.RoleHCAttnNorm, jlm.RoleHCFFNNorm:
		if l.q4e == nil {
			return nil
		}
		if r == jlm.RoleHCAttnNorm {
			return l.q4e.normA
		}
		return l.q4e.normF
	}
	return nil
}

// blockServes is why block li of this step is not one the composer serves
// whole, serves where it is: a graph from the front end, a block the caller
// leaves to it, the paged decode -- or a prompt chunk's, which the tier's
// own kernels would run through its paged prefill attention -- the window
// the plan holds the block to (windowHeld),
// outside the persistent block and a fault probe, and every weight and
// vector its graph binds held (weightsHeld).
func (g *devTier) blockServes(l *layer, li int, p *nn.LayerPlan, bs *blockScratch, persist, prompt bool) partDecline {
	pk := bs.pkv
	switch {
	case l.graph == nil:
		return partNoGraph
	case l.expertsServe(1) != partServes:
		return l.expertsServe(1)
	case !l.weightsHeld():
		return partNoWeight
	case slices.Contains(g.TierBlocks, li):
		return partCaller
	case persist:
		return partPersist
	case len(g.kb.skip) != 0:
		return partSkip
	case l.linear && !g.recHeld(l, p, bs):
		return partRecurrent
	case len(l.graph.Out) != p.Streams() || len(l.graph.In) > len(l.graph.Out) && (len(l.graph.In) != len(l.graph.Out)+1 || bs.pleIn == nil):
		// A block reads each residual stream and, past them, the rows'
		// per-layer inputs.
		return partFeeds
	case l.linear && !l.withAttn:
		return partServes
	case l.nonCausal:
		return g.towerServes(l, li, p, bs)
	case pk == nil || g.kvp == nil || g.kvp.layers[l.kvSrc] == nil:
		// A block sharing another's history reads its source's pool.
		return partNoPool
	case pk.prefill && !prompt:
		return partPrefill
	case !l.windowHeld(li, p):
		return partWindow
	}
	return partServes
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
		!l.centred && (p.Recurrent.TailLen == 0 || tailStated(l.graph, p.Recurrent.StateLen))
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
	t["rec.conv"], t["rec.conv.out"] = l.pool.c[par], l.pool.c[1-par]
	if g.recShared {
		if bs.cRows == nil {
			return false
		}
		t["rec.state"], t["rec.state.out"], t["rec.rows"] = l.pool.s[0], l.pool.s[0], bs.cRows
		return true
	}
	t["rec.state"], t["rec.state.out"] = l.pool.s[par], l.pool.s[1-par]
	return true
}

// windowHeld says the window block l's graph states is the plan's for block
// li, whose descriptors the tier stages (pagedStage), or one that cuts
// nothing in the plan's session, where the full descriptors start where the
// window would.
func (l *layer) windowHeld(li int, p *nn.LayerPlan) bool {
	if w := p.Window(li); w > 0 {
		return l.win == w && l.chunked == p.SWAChunked
	}
	return l.win == 0 || l.win >= p.MaxSeq
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
	if bs.csSWA != nil && p.LocalRope(li) {
		rcs = bs.csSWA
	}
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
	sv, err = g.servedBlockAs(l, li, rcs, g.kvp.layers[l.kvSrc], bs.pkv.pRow, bs.pkv.windowDesc(p, li), qOff, bs.calls, bs.dRows, keys, g.kvp.p, R,
		rowSeqs(R, one), run)
	if err != nil {
		if run != nil {
			err = fmt.Errorf("%w: %w", errStreamForm, err)
		}
		return nil, nil, nil, err
	}
	if err := sv.f.streamsInPlace(len(l.graph.Out)); err != nil {
		return nil, nil, nil, err
	}
	if !sv.f.inPlace(0, 0) {
		if cp, err = g.servedCopy(bs, R); err != nil {
			return nil, nil, nil, err
		}
	}
	if R == 1 && run == nil {
		return sv, sv.inst, cp, nil
	}
	return sv, g.promptInst(bs, sv.f), cp, nil
}

// errStreamForm is a block whose form over passes did not form: it declines
// as one whose history streams (partStream).
var errStreamForm = errors.New("the composed block over passes")

// servedCopy is the copy of a block's output from bs.h over x at a step of
// rows, formed through the composer like any other part, for a form that
// cannot write its output over x (partForm.inPlace). bs.h holds a row of
// every step's rows: a block of several residual streams writes each over
// its own view of x or is refused (streamsInPlace), and never reaches here.
func (g *devTier) servedCopy(bs *blockScratch, rows int) (*servedPart, error) {
	w := bs.p.ResidW()
	if w != bs.p.NEmbd || rows > bs.rows {
		return nil, fmt.Errorf("tier: a block output of %d rows of %d apart from x, where the scratch holds %d of %d", rows, w, bs.rows, bs.p.NEmbd)
	}
	return g.servedRowCopy(w, rows)
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
// does not build is named in LastErr and leaves the other blocks to form.
// Callers hold g.mu.
func (g *devTier) prewarmComposedBlock(li int, narrow bool) (roomless string) {
	if l := g.layers[li]; l != nil && l.nonCausal {
		return g.prewarmTower(li)
	}
	bs, l := g.bs, g.layers[li]
	if g.NoPromptPrewarm || bs == nil || bs.pkv == nil ||
		g.kvp == nil || l == nil {
		return ""
	}
	g.prewarming = true
	defer func() { g.prewarming = false }()
	if g.blockServes(l, li, &bs.p, bs, false, false) == partServes {
		rcs := bs.cs
		if bs.csSWA != nil && bs.p.LocalRope(li) {
			rcs = bs.csSWA
		}
		serve := func() (*servedPart, error) {
			return g.servedBlock(l, li, rcs, g.kvp.layers[l.kvSrc], bs.pkv.pRow, bs.pkv.windowDesc(&bs.p, li), bs.zero, bs.calls, bs.dRows, bs.p.MaxSeq,
				g.kvp.p, 1, 0)
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
			return ""
		}
		if !sv.f.inPlace(0, 0) {
			if _, err := g.servedCopy(bs, 1); err != nil {
				g.LastErr = "the composed block's copy: " + err.Error()
			}
		}
		if (!l.linear || l.withAttn) && !g.streamServed(l, li, bs) {
			g.LastErr = "the composed block over passes: " + g.LastErr
		}
	}
	// Every prompt width the lane holds a scratch for, and a ragged step's:
	// its first chunk or step forms and allocates nothing.
	for _, bb := range g.bbs {
		g.prewarmPromptBlock(bb, li)
	}
	g.fitReserved()
	// The shadow arm's comparison of the block's two outputs.
	if g.shadowing() {
		if _, err := g.servedShadow(l, bs.p.NEmbd); err != nil {
			g.LastErr = "the composed shadow: " + err.Error()
			return
		}
	}
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
	if g.promptServed(l, li, bb) && (!l.linear || l.withAttn) && !g.streamServed(l, li, bb) {
		g.LastErr = "the composed prompt block over passes: " + g.LastErr
	}
}

// promptRoomless is why block li, which the composer serves, has no prompt
// part at the reserved width for want of room, or "" where it has one, the
// composer does not serve it, or what failed was not room. Callers hold g.mu.
func (g *devTier) promptRoomless(li int) string {
	bb, l := g.bbs[g.promptW], g.layers[li]
	if bb == nil || l == nil || g.blockServes(l, li, &bb.p, bb, false, true) != partServes {
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
		g.blockServes(l, li, &bb.p, bb, false, true) != partServes {
		return false
	}
	rcs := bb.cs
	if bb.csSWA != nil && bb.p.LocalRope(li) {
		rcs = bb.csSWA
	}
	// A call's rows are one sequence's in position order (a prompt's
	// chunk), or at the ragged step's width any sequences': a form each,
	// both formed here.
	for _, seqs := range [...]int{1, 0} {
		if seqs == 0 && !g.raggedAt(bb) {
			continue
		}
		sv, err := g.servedBlock(l, li, rcs, g.kvp.layers[l.kvSrc], bb.pkv.pRow, bb.pkv.windowDesc(&bb.p, li), bb.roff, bb.calls, bb.dRows,
			bb.p.MaxSeq, g.kvp.p, bb.rows, seqs)
		if err != nil {
			// A form that failed earlier was named then (servedFor).
			if !errors.Is(err, errFormedEarlier) {
				g.LastErr = "the composed prompt block: " + err.Error()
			}
			return false
		}
		if !sv.f.inPlace(0, 0) {
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
// Callers hold g.mu, outside any submission.
func (g *devTier) streamServed(l *layer, li int, bb *blockScratch) bool {
	run, err := g.runBufs(bb, l, li)
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	rcs, qOff := bb.cs, bb.zero
	if bb.csSWA != nil && bb.p.LocalRope(li) {
		rcs = bb.csSWA
	}
	if bb.rows > 1 {
		qOff = bb.roff
	}
	for _, seqs := range [...]int{1, 0} {
		if seqs == 1 && bb.rows == 1 || seqs == 0 && bb.rows > 1 && !g.raggedAt(bb) {
			continue
		}
		sv, err := g.servedBlockAs(l, li, rcs, g.kvp.layers[l.kvSrc], bb.pkv.pRow, bb.pkv.windowDesc(&bb.p, li), qOff, bb.calls, bb.dRows,
			g.passKeys(bb, bb.p.MaxSeq), g.kvp.p, bb.rows, seqs, run)
		if err != nil {
			if !errors.Is(err, errFormedEarlier) {
				g.LastErr = err.Error()
			}
			return false
		}
		if !sv.f.inPlace(0, 0) {
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
// (jit.Step.Seqs): 1 where its rows are one sequence's in position order,
// else 0, a row each.
func rowSeqs(R int, one bool) int {
	if R > 1 && one {
		return 1
	}
	return 0
}

// promptHeld reports whether a placement leaves block li's own batched
// kernels at bb's rows uncompiled: the composer serves its prompt or the
// server's ragged step there (promptServed). They are compiled where the
// block's rows decline (declineTwins). The reserved width a ragged step
// alone is composed in keeps them for the prompts the tier runs there.
// Callers hold g.mu.
func (g *devTier) promptHeld(l *layer, li int, bb *blockScratch) bool {
	return g.promptServed(l, li, bb)
}

// promptDeclines readies, before a prompt call's submission at bs's rows,
// every block of [lo, hi) whose prompt the composer does not serve at this
// call: the tier's own kernels the placement left it (declineTwins); where
// the call asks nothing of the composer (rowsKind), every block. It reports
// whether any block of the range runs them. Callers hold g.mu,
// outside any submission.
func (g *devTier) promptDeclines(bs *blockScratch, lo, hi int, asks bool) (bool, error) {
	tier := false
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.ok || asks && g.promptServed(l, li, bs) {
			continue
		}
		tier = true
		why := g.blockServes(l, li, &bs.p, bs, false, true)
		switch {
		case why != partServes:
		case !asks:
			why = partCaller
		default:
			// The composer serves the block and its prompt part did not form
			// or fit: that is room or a bug to name, never main's kernels.
			return true, fmt.Errorf("block %d: the composed prompt part: %s", li, g.LastErr)
		}
		if err := g.declineTwins(l, li, bs, why); err != nil {
			return true, err
		}
	}
	return tier, nil
}

// rowsKind is the part a call of R rows asks of the composer for its
// blocks: a prompt's chunk, or a call of rows that is not a ragged step, is
// ComposedPrompt's; a ragged decode step's rows and a one-row step are
// ComposedBlock's.
func (g *devTier) rowsKind(R int, rag bool) partKind {
	if (R > 1 || g.prompt) && (!rag || g.prompt) {
		return partPrompt
	}
	return partBlock
}

// declineTwins compiles block li's own batched kernels at bs's rows, which
// the placement left to the composer (promptHeld) and its prompt now
// declines (why): the block's matvecs, its fused operands and the staged
// attention's span variants, counted by reason (Stats.DeclineBuilt).
// Callers hold g.mu, outside any submission.
func (g *devTier) declineTwins(l *layer, li int, bs *blockScratch, why partDecline) error {
	if li >= len(bs.left) || !bs.left[li] {
		return nil
	}
	for _, m := range l.batchedMVs() {
		if m.kern == nil || m.slots > 0 {
			continue
		}
		if _, ok := g.batchMV(*m, bs.rows, bs.tok); !ok {
			return fmt.Errorf("tier: block %d's prompt declines (%s) and has no batched twin at %d rows",
				li, why, bs.rows)
		}
	}
	g.fusedTwins(bs, l)
	g.spanVariants(bs)
	bs.left[li] = false
	g.DeclineBuilt[why]++
	return nil
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
	if bs := g.bs; bs != nil && bs.embedServes() == partServes && g.embedBufs(bs) {
		if _, err := g.servedEmbed(bs, bb.rows); err != nil {
			g.LastErr = "the composed input rows: " + err.Error()
		}
	}
	for li := range g.layers {
		g.prewarmPromptBlock(bb, li)
	}
}

// raggedAt says the composer serves a ragged step's rows, any sequences'
// mixed, at batched scratch bb's rows: the server's step width, or the
// reserved width a ragged step runs in (reserveScratch).
func (g *devTier) raggedAt(bb *blockScratch) bool {
	return bb != nil && (g.stepW > 0 && bb.rows == g.stepW || bb == g.reservedFor(ComposedBlock))
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
	// The sampler's candidates and the label pick, one form each whatever
	// k, the history and the labels.
	if !g.NoPromptPrewarm && bs != nil && g.argmaxOut != nil {
		g.prewarming = true
		for _, gr := range [...]*graph.Graph{bs.sampleTopK, bs.pickGraph} {
			if gr == nil {
				continue
			}
			if _, err := g.servedSample(gr); err != nil {
				g.LastErr = "the composed sampler's candidates or label pick: " + err.Error()
			}
		}
		g.prewarming = false
	}
	// The collapse of every stream a call of the head alone runs, at a
	// decode step's row.
	if !g.NoPromptPrewarm && bs != nil && bs.collapse != nil {
		g.prewarming = true
		if _, err := g.servedCollapse(bs.collapse, 1); err != nil {
			g.LastErr = "the composed streams' collapse: " + err.Error()
		}
		if _, err := g.servedRowCopy(bs.p.NEmbd, 1); err != nil {
			g.LastErr = "the composed copy of the streams' collapse: " + err.Error()
		}
		g.prewarming = false
	}
	// The input rows at one row, at every prompt width the lane holds a
	// scratch for, and at every exact width a chunk's scratch is cut to
	// (keepExact), as prewarmPromptWidths forms the blocks'.
	if !g.NoPromptPrewarm && bs != nil && bs.embedServes() == partServes && g.embedBufs(bs) {
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
		bs.headServes(bs.hNorm, false, false) != partServes {
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
	if bs.pickHead != nil && bs.headCap == nil {
		if _, err := g.servedPickHead(bs, bs.hNorm); err != nil {
			g.LastErr = "the composed pick head: " + err.Error()
		}
	}
	// A head over rows at each width the lane holds a scratch for, its rows
	// a call's (jit.CallRows): a step of any count of wanted rows forms
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
func poolHistory(k kvGeom) jit.History {
	if k.mla {
		return jit.History{ValuesInKeys: true}
	}
	return jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: k.f16}}
}

// servedBlock is block li served whole at a step of R rows: its graph formed
// as one unit on the block's weights and vectors by role, its rotary table
// cs and its history in the tier's pool layer pl, through the rows'
// descriptors, a windowed attention through rowsW
// (pagedScratch.windowDesc), for a session of maxSeq positions.
//
// The form is the session's: its attention runs as many trips as each row's
// live positions need, a count it reads at run time (jit.Trips), so the
// session's capacity is its key bound and no position compiles a form. A
// prompt chunk's form is its width's, as the tier's kernels are.
func (g *devTier) servedBlock(l *layer, li int, cs backend.Buf, pl *kvLayerPool, rows, rowsW, qOff, calls, rec backend.Buf, maxSeq, page, R, seqs int) (*servedPart, error) {
	return g.servedBlockAs(l, li, cs, pl, rows, rowsW, qOff, calls, rec, maxSeq, page, R, seqs, nil)
}

// servedBlockAs is servedBlock, its history read in passes where run, the
// running partial's buffers, is set (jit.History.Passes).
func (g *devTier) servedBlockAs(l *layer, li int, cs backend.Buf, pl *kvLayerPool, rows, rowsW, qOff, calls, rec backend.Buf, maxSeq, page, R, seqs int,
	run []backend.Buf) (*servedPart, error) {
	keys := keyBound(maxSeq)
	kind := partBlock
	if R > 1 {
		kind = partPrompt
	}
	// A linear block holds no history, unless it attends too: its state is
	// its pool's (bindRec).
	var kk, kv, tab backend.Buf
	var geom kvGeom
	if pl != nil && (!l.linear || l.withAttn) {
		kk, kv, tab, geom = pl.k, pl.v, pl.tab, pl.geom
	}
	sk := servedKey{kind: kind, graph: l.graph, rows: R, keys: keys, page: page, half: geom.f16, latent: geom.mla, seqs: seqs, stream: run != nil,
		held: l.pagedHeld()}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		// The request names no block: blocks of one structure and formats
		// share a form, each launched on its own bindings, one after
		// another on the device's queue.
		gr := *l.graph
		gr.Block = 0
		sl := composedPart{kind: kind, graph: &gr, formats: map[jlm.Role]jlm.Type{},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{},
			tables: map[string]backend.Buf{"rope.cs": cs, "kv.k": kk, "kv.v": kv, "pages": tab, "rows": rows,
				"rows.win": rowsW, "q.off": qOff, "rows.n": calls},
			history: poolHistory(geom),
			keys:    keys, page: page, seqs: seqs, paged: l.paged()}
		if run != nil {
			sl.history.Passes = true
			sl.tables["run.in"], sl.tables["run.out"] = run[0], run[1]
		}
		if readsPlanes(l.graph) {
			pl := g.blockPlanes()
			if pl == nil {
				return sl, fmt.Errorf("the composed block %d turns by the rotary's planes, which the tier holds none of", li)
			}
			sl.tables["rope.planes"] = pl
		}
		if l.linear && l.pool != nil {
			t := sl.tables
			// The shared form holds one half, stepped in place (bindRec).
			out := l.pool.s[1]
			if out == nil {
				out = l.pool.s[0]
			}
			t["rec.state"], t["rec.state.out"], t["rec.conv"], t["rec.conv.out"] = l.pool.s[0], out, l.pool.c[0], l.pool.c[1]
			t["rec.rows"] = rec
		}
		for _, o := range l.graph.Ops {
			for _, w := range o.Weight {
				// A bank's bias is a row an expert, held as one vector.
				if w.Rows > 0 && l.roleVector(w.Role) == nil {
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
	for r := range sv.sl.vectors {
		sv.sl.vectors[r] = l.roleVector(r)
	}
	t := sv.sl.tables
	t["rope.cs"], t["kv.k"], t["kv.v"], t["pages"], t["rows"], t["rows.win"], t["q.off"] = cs, kk, kv, tab, rows, rowsW, qOff
	t["rows.n"] = calls
	if run != nil {
		t["run.in"], t["run.out"] = run[0], run[1]
	}
	if l.linear {
		t["rec.rows"] = rec
	}
	if _, ok := t["rope.planes"]; ok {
		t["rope.planes"] = g.blockPlanes()
	}
	return sv, nil
}

// readsPlanes reports whether a form of g reads the rotary's planes: a
// pooled selection turning its blocks, or a rotary at positions of its own.
func readsPlanes(g *graph.Graph) bool {
	return slices.ContainsFunc(g.Ops, func(o graph.Op) bool {
		return o.Kind == graph.OpSelect && o.Attr.Pooled && o.Attr.NRot > 0 || o.Kind == graph.OpRope && o.Attr.Shift > 0
	})
}

// blockPlanes is the model's rotary planes a block's form reads ("rope.planes"):
// the global table's, nil where the tier holds none.
func (g *devTier) blockPlanes() backend.Buf {
	if g.bs == nil {
		return nil
	}
	_, pl, _ := g.tableBinds(g.bs, 0)
	return pl
}
