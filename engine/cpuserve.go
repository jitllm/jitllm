package engine

import (
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// The CPU is a device like a card: a block placed on it runs what the
// composer forms for it from the block's graph (Model.Composing), through
// the one keyed cache, on the instance's own buffers -- the residual, the
// block's weights and vectors by role, its rotary table and its history in
// the host's KV pages (cpudev.Arena) -- each kernel mapped once for the model and
// called through the trampoline, the calls of a kernel over heads spread
// over the instance's pool. Every block step on the CPU is counted served or
// declined by name (CPUBlockSteps), as the tier counts a card's.

// cpuRefused counts the composer's decline of block li's step on the CPU and
// is the step's error: the composer is the only path, so no other kernels run
// the block.
func (s *Instance) cpuRefused(li int, why nn.Decline) error {
	s.cpu.x.Steps().Decline(nn.PartBlock, why)
	if why == nn.DeclineNotFormed && s.cpu.x.Composer().Err() != nil {
		return fmt.Errorf("model: the composer does not serve block %d on the CPU: %s (the first: %w)", li, nn.DeclineName(int(why)), s.cpu.x.Composer().Err())
	}
	if why == nn.DeclineNoGraph {
		u, err := s.m.UncoveredAt(li)
		return fmt.Errorf("model: the composer does not serve block %d on the CPU: %s (the front end: %v, %v)", li, nn.DeclineName(int(why)), u, err)
	}
	return fmt.Errorf("model: the composer does not serve block %d on the CPU: %s", li, nn.DeclineName(int(why)))
}

// cpuHeadRefused is the composer's decline of the head on the CPU as the
// step's error, as cpuRefused is a block's.
func cpuHeadRefused(why nn.Decline) error {
	return fmt.Errorf("model: the composer does not serve the head on the CPU: %s", nn.DeclineName(int(why)))
}

// cpuState is a instance's composed CPU blocks: each block's form and its own
// buffers, the history tables, and the census.
type cpuState struct {
	// workers is the CPU device's workers (cpu.Pool), x the instance's
	// execution on the device: its blocks' forms, the runners that call
	// them and what it served.
	workers *cpu.Pool
	x       *cpudev.Exec
	// lanes runs a decode token's blocks on the device for the session the
	// instance runs (cpudev.Lanes).
	lanes     *cpudev.Lanes
	tabsFresh bool
	tabSeq    []int
	// rseq and rpos are a step's rows' sequences and positions; lastPos
	// each sequence's last position a step's rows reach (cpuRecWhole).
	lastPos    []int
	rseq, rpos []int
	// runWindows says each run of the instance's (bidir) is a window its rows
	// attend inside, none of the positions before it: a tower's windowed
	// block's (cpuTowerRows).
	runWindows bool
	// rowWins is each row's own window where a call's rows' windows overlap
	// (an LFM2 encoder's media prefix), nil else: its rows' descriptors
	// carry them as they are.
	rowWins []nn.KeyRun
	// bound is what a run of a form's codes reads its addresses from.
	bound cpudev.Bound
	// fault is a violation a gate sets: the bindings rewritten so a kernel
	// reads the wrong history page or the wrong first row.
	fault cpuFault
}

// cpuFault is a gate's violation of the composed CPU path's bindings.
type cpuFault uint8

const (
	cpuFaultNone cpuFault = iota
	// cpuFaultPage points a layer's first page at the last slot its own
	// arena holds once that page is full (cpuFaultTable): its positions
	// read another page's (or nothing). A table permuted from the first
	// step only renames the pages, and keys carry their rotation, so a swap
	// of written pages only reorders them.
	cpuFaultPage
	// cpuFaultRow moves the first key row a step attends by one, and past
	// its second row the last by one: the step reads neither its first row
	// nor its newest, the one a block attending mostly to itself reads.
	cpuFaultRow
	// cpuFaultRec binds every linear block to the recurrent state and
	// window of the first: each steps a summary that is not its own.
	cpuFaultRec
	// cpuFaultChunk starts each prompt chunk after the first from a summary
	// of zeros: the recurrence not carried across the chunk boundary.
	cpuFaultChunk
	// cpuFaultHead binds the head's input to the instance's norm row in place
	// of the residual.
	cpuFaultHead
	// cpuFaultExpert makes resident and holds another expert's page than
	// the first the selection names: a bank's sheet read where the pager
	// holds nothing of it, or holds it evicted.
	cpuFaultExpert
	// cpuFaultStreams swaps streams 1 and 2 where the input rows' expansion
	// writes them, and reads stream 1 in stream 2's place where the head's
	// collapse reads them (a mix the same over any order of its streams sees
	// a stream read twice).
	cpuFaultStreams
	// cpuFaultEntryRows gathers each patch row of the tower's entry from the
	// patch after its own.
	cpuFaultEntryRows
	// cpuFaultKeysFirst runs a bidirectional run of several calls' first
	// pass over none of its codes: each call's rows read only the keys of
	// the calls before them and their own.
	cpuFaultKeysFirst
	// cpuFaultWrite writes a step's new key and value at the row before its
	// own, its own row left as the history held it.
	cpuFaultWrite
	// cpuFaultPicks gives a call's first row the second's picks, read with
	// its own weights.
	cpuFaultPicks
	// cpuFaultWeight drops each row's first pick's weight (weightsOf).
	cpuFaultWeight
	// cpuFaultFilterT binds a convolutional block's depthwise filters with
	// their taps' rows and columns swapped.
	cpuFaultFilterT
)

// cpuRowsCap is the rows the CPU's form of several rows is composed for
// (cpudev.RowsCap).
const cpuRowsCap = cpudev.RowsCap

// CPUForm is how the composer forms this host's CPU device
// (composer.CPUForm).
func CPUForm() string {
	return composer.CPUForm(cpudev.Target(cpu.Probe().ISA(), 1), runtime.GOARCH == "arm64")
}

// cpuRows runs block li on the CPU the composer's way for the n rows of x,
// row i at position pos[i] of sequence seq[i], its rotary table the i-th of
// cs, or says why not: through the rows' form, a call of at most its
// capacity at a time, each row's history in its own sequence's pages.
func (s *Instance) cpuRows(li int, l *model.Layer, x, cs, csSWA []float32, seq, pos []int, tok []int32) (nn.Decline, error) {
	if err := s.cpuRowTables(seq, pos); err != nil {
		return nn.DeclineServes, err
	}
	ln := s.cpu.lanes
	ln.Attach(s.kv, s.view)
	f, why := ln.RowsForm(li, l)
	if why != nn.DeclineServes {
		return why, nil
	}
	if why := ln.ReadyRows(li, f, pos); why != nn.DeclineServes {
		return why, nil
	}
	side, _ := s.cpuSideRows(li)
	return nn.DeclineServes, s.cpuRowsRun(li, l, f, x, len(pos)*s.c.NEmbd, side, s.bcs, s.c.RopeW(), s.bcsSWA, s.c.LocalNRot(), seq, pos, tok)
}

// swaRows is the step's second rotary table from row a, nil where the model
// has no second rotary (cpuRowTables writes it into s.bcsSWA).
func (s *Instance) swaRows(a int) []float32 {
	if s.bcsSWA == nil {
		return nil
	}
	return s.bcsSWA[a*s.c.LocalNRot():]
}

// rowsOf is rows [a, b) of a step's rows' tokens, nil for a step of none.
func rowsOf(tok []int32, a, b int) []int32 {
	if tok == nil {
		return nil
	}
	return tok[a:b]
}

// cpuRowsOf is the sequences and positions of a chunk's n rows of slot
// from position p0, in the instance's own rows.
func (s *Instance) cpuRowsOf(slot, p0, n int) (seq, pos []int) {
	if cap(s.cpu.rpos) < n {
		s.cpu.rseq, s.cpu.rpos = make([]int, n), make([]int, n)
	}
	seq, pos = s.cpu.rseq[:n], s.cpu.rpos[:n]
	for i := range n {
		seq[i], pos[i] = slot, p0+i
	}
	return seq, pos
}

// addr is x's first element's address, nil for a nil slice.
func addr[T any](x []T) unsafe.Pointer { return unsafe.Pointer(unsafe.SliceData(x)) }

// unmapOwn returns the own buffers and the n-gram table's view held off
// the heap.
func (s *Instance) unmapOwn() {
	s.cpu.x.Close()
	s.ws.b.Close()
}

// cpuPlace composes a new instance's blocks on the CPU: each block of the
// instance's range no device holds, once, at its placement (cpuCompose).
func (s *Instance) cpuPlace() {
	// A segment's blocks are past the text model's in the one block space.
	n := max(s.c.NLayer, s.hi)
	planes := [2][]float32{s.tablePlanes(0), s.tablePlanes(1)}
	s.lease = blockLease{m: s.m, fr: s.fr, mu: &s.bind, hold: &s.expHold, res: s.res, hot: s.hot, layers: s.layers,
		lo: s.lo, ws: &s.ws}
	s.cpu.x.Place(n, &s.lease, planes, s.c.NExpert, s.prof)
	if s.cpu.lanes == nil {
		s.cpu.lanes = cpudev.NewLanes(s.cpu.x, &s.ws.b, &s.lease, s.c.NEmbd, s.nseq, s.c.Streams(), s.dg.ConvStateLen(),
			func(li int) { s.trace(li, "ffn_resid", s.x) })
		s.bindLanes(s.Session)
	}
	for li := s.lo; li < min(s.hi, n); li++ {
		if !s.devAt(li) {
			s.cpu.x.Compose(li, s.blockPlan(li))
		}
	}
	s.placeParts()
}

// sideAtBinding moves the offset of every slice of gr's side input (its
// input past its residual streams) into the input's binding: where each
// slice reads from one offset, the slices read from 0 and the offset is
// returned for the call to bind the input at. A graph whose slices of the
// side differ, or that reads it otherwise, keeps its offsets (0).
func sideAtBinding(gr *graph.Graph) int {
	if len(gr.In) != len(gr.Out)+1 {
		return 0
	}
	side := gr.In[len(gr.Out)]
	off := -1
	for _, o := range gr.Ops {
		if !slices.Contains(o.In, side) {
			continue
		}
		if o.Kind != graph.OpSlice || o.Attr.HeadDim != 0 || off >= 0 && o.Attr.Offset != off {
			return 0
		}
		off = o.Attr.Offset
	}
	if off <= 0 {
		return 0
	}
	gr.Ops = slices.Clone(gr.Ops)
	for i := range gr.Ops {
		if slices.Contains(gr.Ops[i].In, side) {
			gr.Ops[i].Attr.Offset = 0
		}
	}
	return off
}

// cpuSide is block li's side input of decode's row: the n-gram block's
// gathered rows (ngramForCPU), the row's per-layer inputs on a model of
// them, nil where the block has none.
func (s *Instance) cpuSide(li int) []float32 {
	if s.c.SideWidth() == 0 {
		return nil
	}
	return s.ple
}

// cpuSideRows is cpuSide over a call's rows, and its width a row.
func (s *Instance) cpuSideRows(li int) ([]float32, int) {
	return s.bple, s.c.SideWidth()
}

// blockPlan is what the CPU device composes block li from: the graph its
// front end states, the formats its weights are read in, their strides, the
// step's history geometry and the instance's rotaries -- or what keeps the
// block off the composer's CPU path before it is asked (cpuWhy).
func (s *Instance) blockPlan(li int) cpudev.BlockPlan {
	p := cpudev.BlockPlan{Why: s.blockWhy(li), Rec: s.c.LayerKind(li).Recurrent()}
	if p.Why != nn.DeclineServes {
		return p
	}
	g := s.m.Composing().BlockGraph(li)
	gr := *g
	gr.Block = 0
	p.Source, p.Graph, p.SideOff = g, &gr, sideAtBinding(&gr)
	if s.m.Ngram != nil && cpudev.TableRowsOf(&gr) >= 0 {
		if err := s.ws.b.OpenTable(s.m.Ngram.E, s.m.Ngram.RowBytes); err != nil {
			p.Why = s.cpu.x.Composer().Decline(err)
			return p
		}
	}
	formats := map[jlm.Role]jlm.Type{}
	f := s.m.Composing()
	if li >= s.m.TowerBase {
		formats = s.m.TowerFormats(g)
	}
	for _, o := range g.Ops {
		if li >= s.m.TowerBase {
			break
		}
		for _, w := range o.Weight {
			if e, ok := f.Find(w.Role, jlm.DenseBlock, -1); ok && jlm.RowTable(e) {
				formats[w.Role] = e.Type
			} else if e, ok := f.Find(w.Role, int32(li), -1); ok {
				formats[w.Role] = e.Type
				// A table the block lays out as float32 at load is read so.
				if t := s.ws.matrix(li, s.layer(li), w.Role); w.Rows > 0 && t != nil && t.E == nil && t.Typ == quant.F32 {
					formats[w.Role] = jlm.TypeF32
				}
			} else if w.Rows == 0 {
				// A vector a restatement adds is held as every vector is.
				formats[w.Role] = jlm.TypeF32
			}
		}
	}
	pg := &s.kv.layers[li]
	page := max(pg.p, 1)
	p.Req = composer.Request{Formats: formats, Strides: s.m.CpuStrides(s.fr, li, g),
		Step:    composer.Step{Page: page, Keys: max(pg.arenaSlots(), 1) * page, ActWindow: s.m.ActWindow, Seqs: max(s.nseq, 1)},
		History: composer.History{ValuesInKeys: pg.latent},
		// A block turns by the rotary its rope ops state, of the
		// instance's (rotaries): each its own table, bound by instance.
		Rotaries: s.rotaries()}
	p.F32 = s.kvl.fmt == cpu.KVF32
	p.Bidir = s.c.LayerKind(li).Attends() && s.c.BidirAt(li)
	p.Whole = s.wholePicture(li)
	return p
}

// arena is the table of the layer's arena, nil for a layer with none (or no
// layer).
// kvAddrs is where history pg's keys and values are read, nil for none.
func kvAddrs(pg *kvPages) (keys, values unsafe.Pointer) {
	if t := pg.arena(); t != nil {
		return addr(t.Arena().Keys()), addr(t.Arena().Values())
	}
	return nil, nil
}

func (s *kvPages) arena() *cpudev.Table {
	if s == nil || s.mem == nil {
		return nil
	}
	return s.mem.Table()
}

// arenaSlots is how many pages the layer's arena holds, 0 for none.
func (s *kvPages) arenaSlots() int {
	if t := s.arena(); t != nil {
		return t.Slots()
	}
	return 0
}

// cpuRowsRun runs form f of block li (l its copy) over the rows of x, the
// stream-major residual stride apart, at positions pos of sequences seq with
// tokens tok, their per-layer inputs side, rotary tables cs and swa, on the
// CPU device (cpudev.Lanes.Rows): the session's history attached, the
// prompt's runs and the rows' windows handed before the call.
func (s *Instance) cpuRowsRun(li int, l *model.Layer, f *cpudev.Form, x []float32, stride int, side, cs []float32, nrot int, swa []float32, nswa int, seq, pos []int, tok []int32) error {
	ln := s.cpu.lanes
	ln.Attach(s.kv, s.view)
	ln.KeyRuns(s.bidir, s.cpu.runWindows, s.cpu.rowWins)
	_, pw := s.cpuSideRows(li)
	rc := cpudev.RowsCall{X: x, Stride: stride, Side: side, SideW: pw, CS: cs, NRot: nrot, CSSWA: swa, NSWA: nswa,
		Seq: seq, Pos: pos, Tok: tok}
	return ln.Rows(li, f, l, &rc)
}

// blockWhy is what keeps block li off the composer's CPU path before it is
// asked, as the model and its history state it: no graph, an input the
// device is handed none of, a bank with no expert pages (the forms read a
// bank as one view of its experts in the frames' expert pages,
// jlm.Frames.BankView), a recurrent block with no state, a history format
// the forms do not read (the CPU keeps a history binary16 itself,
// composer.Halves, or at f32 without that pass) or pages outside an arena.
func (s *Instance) blockWhy(li int) nn.Decline {
	c := s.c
	g := s.m.Composing().BlockGraph(li)
	// A block sharing another's history reads its source's pages.
	pg := &s.kv.layers[c.KVSource(li)]
	l := &s.m.Layers[li]
	switch {
	case g == nil:
		return nn.DeclineNoGraph
	case len(g.Out) != c.Streams() || len(g.In) > len(g.Out)+1 || len(g.In) > len(g.Out) && s.cpuSide(li) == nil:
		return nn.DeclineFeeds
	case c.MoEAt(li) && (s.m.Container == nil || l.LegacyExperts ||
		!s.ws.b.BankViews(li, len(s.m.Layers), [3]*jlm.Entry{l.Gate.E, l.Up.E, l.Down.E}) || l.Up.E == nil || l.Down.E == nil):
		return nn.DeclineExperts
	case c.LayerKind(li).Recurrent() && (s.view.State(li) == nil || s.view.Conv(li) == nil):
		return nn.DeclineRecurrent
	case s.kvl.fmt != cpu.KVF16 && s.kvl.fmt != cpu.KVF32:
		return nn.DeclineKVFormat
	case pg.p > 0 && pg.arena() == nil:
		return nn.DeclinePageOut
	}
	return nn.DeclineServes
}
