package engine

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	hw "github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/engine/probe"
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
	s.cpu.steps.Decline(nn.PartBlock, why)
	if why == nn.DeclineNotFormed && s.cpu.formErr != nil {
		return fmt.Errorf("model: the composer does not serve block %d on the CPU: %s (the first: %w)", li, nn.DeclineName(int(why)), s.cpu.formErr)
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
	// workers is the CPU device's workers (cpu.Pool).
	workers   *cpu.Pool
	launch    int
	launchSet bool
	blocks    []*cpuBlock
	composed  []int
	// steps is what the CPU served the instance, part by part.
	steps   nn.Steps
	formErr error
	// fpages is a layer's page table under cpuFaultPage (cpuFaultTable), and
	// nopages a block's that keeps no history (cpuPages).
	fpages, nopages []uint32
	rows            [4]uint32
	zero            []uint32
	// tok1 is a decode's token, tokTab a call's rows' (all ones for a row
	// naming none).
	tok1   [1]uint32
	tokTab []uint32
	// oneFresh and tabsFresh say the decode row's and the step's rows'
	// rotary tables are made (cpuOneTable, cpuRowTables); tabSeq is the
	// rows' slots as the tables read them.
	oneFresh, tabsFresh bool
	tabSeq              []int
	// pos1 and posTab are a block's call's live rows then each row's
	// position (cpuArgPos), a decode's and a call's.
	pos1   [2]uint32
	posTab []uint32
	// rowsTab is a call's rows' descriptors at the rows' form, call its
	// "rows.n" word, and rseq and rpos a step's rows' sequences and
	// positions.
	rowsTab []uint32
	call    [4]uint32
	// recRows is a linear block's call's recurrent descriptor: one row, the
	// slot it reads and the slot it writes, its sequence's.
	recRows [3]uint32
	// recDesc is a call of the rows' form's run descriptor (cpuRecDesc),
	// lastPos each sequence's last position a step's rows reach
	// (cpuRecWhole).
	recDesc []uint32
	lastPos []int
	// convOut is a step of several rows' convolution windows, written apart
	// and copied over the runs' slots after the call.
	convOut    []float32
	rseq, rpos []int
	onePos     [1]int
	// sel is a call's experts as the pager's fault path takes them, picked
	// marking those already in it, by expert (cpuExperts).
	sel    []int32
	picked []bool
	// kvconv is a binary16 or q8 history's conversions (cpukvconv.go).
	kvconv cpuKVConvs
	// xpad is a call's residual rows where its kernels' units reach past
	// them (cpudev.Form.unit).
	xpad []float32
	// sidePad is xpad's per-layer inputs.
	sidePad []float32
	// stash is the values crossing a form's last history write, a call's
	// after another, between a bidirectional run's two passes (cpuRowsRun);
	// stashing says which pass is running.
	stash    []uint32
	stashing uint8
	// runWindows says each run of the instance's (bidir) is a window its rows
	// attend inside, none of the positions before it: a tower's windowed
	// block's (cpuTowerRows).
	runWindows bool
	// rowWins is each row's own window where a call's rows' windows overlap
	// (an LFM2 encoder's media prefix), nil else: its rows' descriptors
	// carry them as they are.
	rowWins []nn.KeyRun
	// dev is the device's launches, bind the addresses a run of a form's
	// codes binds (cpuCallOf) and bound what they are read from.
	dev   *cpudev.Device
	bind  cpudev.Call
	bound cpuBound
	// fault is a violation a gate sets: the bindings rewritten so a kernel
	// reads the wrong history page or the wrong first row.
	fault   cpuFault
	head    cpuHeadState
	streams cpuStreamsState
	// tabPlanes is each rotary's planes as its composed table reads them
	// (composer.TablePlanes), folded from the model's rotaries when the
	// instance places its blocks (cpuTablePlanes).
	tabPlanes [2][]float32
	mtp       cpuMTPState
	embed     cpuEmbedState
	tables    cpuTableState
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

// A bidirectional run's passes (cpuState.stashing).
const (
	stashNone uint8 = iota
	stashSave
	stashLoad
)

// cpuBlock is block li's forms on the CPU, its graph the one they were
// asked for: decode's, of one row, and the rows', at the CPU's capacity
// (cpuRowsCap), whose calls run the rows their "rows.n" word states.
type cpuBlock struct {
	graph *graph.Graph
	f     *cpudev.Form
	rows  *cpudev.Form
	// sideOff is where the block's slice of the rows' per-layer inputs
	// starts in each row: the call binds the side input there, so blocks
	// that differ by their slice alone share a form.
	sideOff int
	why     nn.Decline
	// rowsWhy is why the rows' form is not served, where it is not.
	rowsWhy nn.Decline
}

// cpuRowsCap is the rows the CPU's form of several rows is composed for: a
// step of more runs in calls of this many, one after another.
const cpuRowsCap = 64

// cpuBound is a block's step as its words read it: n rows of the residual
// x, their rotary tables cs and their descriptors rows, their history the
// layer's pages kv through the page table pages.
type cpuBound struct {
	li  int
	l   *model.Layer
	own [][]uint32
	cs  []float32
	x   []float32
	// stride is the elements from one residual stream of x to the next.
	stride int
	rows   []uint32
	n      int
	kv     *kvPages
	pages  []uint32
	// csSWA is the call's second rotary's table where the model trains one,
	// ropeRows and ropeMarks a rotary table's call's rows and its session's
	// marks (cpuTables).
	csSWA               []float32
	ropeRows, ropeMarks []uint32
	// recLi is the block whose recurrent states a linear block's step binds
	// and recOf the session that holds them.
	recLi int
	recOf *Session
	// rec, where set, is the call's run descriptor in place of recRows.
	rec []uint32
	// side is the bound rows' per-layer inputs, a row each (cpuArgSide).
	side []float32
	// io is the buffers of a graph apart from its residual (cpuArgIO), its
	// inputs' then its outputs'.
	io [cpudev.IO]unsafe.Pointer
	// req is the step's request words (cpuArgRequest), pos its live rows,
	// then a row's position (cpuArgPos), tok its tokens (cpuArgTokens).
	req, pos, tok []uint32
	// planes, where set, is the rotary planes a table's call reads in place
	// of the model's (the tower's grid planes, towertable.go).
	planes []float32
	// pic is the step's picture, its decoded samples' words (cpudev.SrcPicture).
	pic []uint32
}

// composedCPUDevice is the CPU of extensions isa as the composer forms for
// it.
func composedCPUDevice(isa cpu.ISA, workers int) composer.Device {
	return composer.Device{Kind: composer.DeviceCPU, Slots: max(workers, 1), Lanes: 8, ISA: uint64(isa), ExactDiv: true}
}

// CPUForm is how the composer forms this host's CPU device
// (composer.CPUForm).
func CPUForm() string {
	return composer.CPUForm(composedCPUDevice(cpu.Probe().ISA(), 1), runtime.GOARCH == "arm64")
}

// composedCPU is this instance's CPU as the composer forms for it, its launch
// cost its probe row's.
func (s *Instance) composedCPU() composer.Device {
	d := composedCPUDevice(s.cpuISA(), s.cpu.workers.Workers())
	if !s.cpu.launchSet {
		if name, err := probe.HostName(); err == nil {
			if p, ok := hw.Known(name, runtime.GOARCH); ok {
				s.cpu.launch = p.Device.LaunchWords()
			}
		}
		s.cpu.launchSet = true
	}
	d.Launch = s.cpu.launch
	return d
}

// cpuFaultRowWords applies the instance's key-row fault to one row's
// descriptor (first key, end, write row), the step's first row left whole.
func (s *Instance) cpuFaultRowWords(t []uint32) {
	if t[3] == 0 {
		return
	}
	switch s.cpu.fault {
	case cpuFaultRow:
		t[1]++
		if t[2] > t[1]+1 {
			t[2]--
		}
	case cpuFaultWrite:
		t[3]--
	}
}

// cpuStep runs block li on the CPU the composer's way, or says why not. The
// residual is s.x; the rotary table cs.
func (s *Instance) cpuStep(li int, l *model.Layer, cs []float32) (nn.Decline, error) {
	cb, why := s.cpuBlockOf(li)
	if why != nn.DeclineServes {
		return why, nil
	}
	if err := s.cpuOneTable(); err != nil {
		return nn.DeclineServes, err
	}
	// A block sharing another's history reads its source's pages.
	hl := s.c.KVSource(li)
	pg := &s.kv.layers[hl]
	pos := s.pos
	if pg.p > 0 {
		if why := s.cpuHistory(hl, pos); why != nn.DeclineServes {
			return why, nil
		}
	}
	pages, err := s.cpuPages(hl, pg)
	if err != nil {
		return nn.DeclineServes, err
	}
	b := &s.cpu.bound
	*b = cpuBound{li: li, l: l, own: s.cpu.dev.Own(cb.f), cs: cs, x: s.resid(), stride: s.c.NEmbd, rows: s.cpu.rows[:], n: 1, kv: pg, pages: pages,
		recLi: s.cpuRecLi(li), recOf: s.Session, side: s.cpuSide(li)}
	if b.side != nil {
		b.side = b.side[cb.sideOff:]
	}
	s.cpu.tok1[0] = uint32(s.tok)
	s.cpu.pos1 = [2]uint32{1, uint32(pos)}
	b.tok, b.pos, b.csSWA = s.cpu.tok1[:], s.cpu.pos1[:], s.csSWA
	if _, why := s.ws.bind(cb.f, li, l); why != nn.DeclineServes {
		return why, nil
	}
	s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos)), uint32(pos + 1), uint32(pos)}
	s.cpu.recRows = [3]uint32{1, 0, 0}
	s.cpu.call[0] = 1
	switch s.cpu.fault {
	case cpuFaultPage:
		b.pages = s.cpuFaultTable(b.pages, pg, pos+1)
	case cpuFaultRow, cpuFaultWrite:
		s.cpuFaultRowWords(s.cpu.rows[:])
	}
	if err := s.cpuForm(b, cb.f, li); err != nil {
		return nn.DeclineServes, err
	}
	if pg.p > 0 && pos+1 > pg.written {
		pg.written = pos + 1
	}
	return nn.DeclineServes, nil
}

// cpuRows runs block li on the CPU the composer's way for the n rows of x,
// row i at position pos[i] of sequence seq[i], its rotary table the i-th of
// cs, or says why not: through the rows' form, a call of at most its
// capacity at a time, each row's history in its own sequence's pages.
func (s *Instance) cpuRows(li int, l *model.Layer, x, cs, csSWA []float32, seq, pos []int, tok []int32) (nn.Decline, error) {
	if err := s.cpuRowTables(seq, pos); err != nil {
		return nn.DeclineServes, err
	}
	f, why := s.cpuRowsForm(li, l)
	if why != nn.DeclineServes {
		return why, nil
	}
	if why := s.cpuRowsReady(li, f, pos); why != nn.DeclineServes {
		return why, nil
	}
	side, _ := s.cpuSideRows(li)
	return nn.DeclineServes, s.cpuRowsRun(li, l, f, x, len(pos)*s.c.NEmbd, side, s.bcs, s.c.RopeW(), s.bcsSWA, s.c.NRotSWA, seq, pos, tok)
}

// swaRows is the step's second rotary table from row a, nil where the model
// has no second rotary (cpuRowTables writes it into s.bcsSWA).
func (s *Instance) swaRows(a int) []float32 {
	if s.bcsSWA == nil {
		return nil
	}
	return s.bcsSWA[a*s.c.NRotSWA:]
}

// cpuSessions is cpuRows for a step across sessions (stepHost): row i's
// history is s.rowOwn[i]'s, its one sequence, and the first nlogit rows are
// each a session's last. Each session runs its own rows on its own block's
// form and buffers, its history bound as its form was composed for. Every
// session's pages are made ready before any row runs; the runs of a
// session's consecutive rows run one after another, the rows after the
// logit rows first, so a session's last row reads the keys its earlier
// rows wrote.
func (s *Instance) cpuSessions(li int, l *model.Layer, x, cs, csSWA []float32, seq, pos []int, tok []int32, nlogit int) (nn.Decline, error) {
	if err := s.cpuRowTables(seq, pos); err != nil {
		return nn.DeclineServes, err
	}
	cs, nrot := s.bcs, s.c.RopeW()
	n := len(pos)
	// Every row's page is writable before any session's range is checked: a
	// session's logit row leads its other rows in the step, and the range it
	// reads holds the pages they write.
	if hl := s.c.KVSource(li); s.kv.layers[hl].p > 0 {
		for a := range n {
			oi := s
			if a < len(s.rowInst) && s.rowInst[a] != nil {
				oi = s.rowInst[a]
			}
			lead := oi.Session
			oi.Session = s.rowOwn[a]
			oi.cpuWritable(hl, pos[a])
			oi.Session = lead
		}
	}
	why, err := s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, tok, 0, n, false)
	if why != nn.DeclineServes || err != nil {
		return why, err
	}
	if why, err = s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, tok, nlogit, n, true); err != nil {
		return why, err
	}
	return s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, tok, 0, nlogit, true)
}

// cpuOwnerRuns makes ready (run unset) or runs block li over the rows
// [lo, hi) of a step across sessions, a run of one session's consecutive
// rows at a time, on that session's form.
func (s *Instance) cpuOwnerRuns(li int, l *model.Layer, x, cs []float32, nrot int, seq, pos []int, tok []int32, lo, hi int, run bool) (nn.Decline, error) {
	d := s.c.NEmbd
	rside, pw := s.cpuSideRows(li)
	for a := lo; a < hi; {
		o, b := s.rowOwn[a], a+1
		for b < hi && s.rowOwn[b] == o {
			b++
		}
		if o.model != s.m {
			return nn.DeclineSessions, nil
		}
		// A row of another instance runs on that instance's form, composed
		// for its history; a session leasing s, on s's with its history bound.
		oi := s
		if a < len(s.rowInst) && s.rowInst[a] != nil {
			oi = s.rowInst[a]
		}
		f, why := oi.cpuRowsForm(li, l)
		if why != nn.DeclineServes {
			return why, nil
		}
		lead := oi.Session
		oi.Session = o
		if run {
			var side []float32
			if pw != 0 && rside != nil {
				side = rside[a*pw:]
			}
			if err := oi.cpuRowsRun(li, l, f, x[a*d:], len(pos)*d, side, cs[a*nrot:], nrot, s.swaRows(a), s.c.NRotSWA, seq[a:b], pos[a:b], rowsOf(tok, a, b)); err != nil {
				oi.Session = lead
				return nn.DeclineServes, err
			}
		} else if why := oi.cpuRowsReady(li, f, pos[a:b]); why != nn.DeclineServes {
			oi.Session = lead
			return why, nil
		}
		oi.Session = lead
		a = b
	}
	return nn.DeclineServes, nil
}

// cpuRecRows runs linear block li over the n rows of x, or says why not:
// on its rows' form, the scan stepping each sequence's rows in order, where
// the step's rows of each sequence stand in order and all are this instance's
// (cpuRecWhole); else on its form of one row, a row at a time in order
// (each row's index; 0.. where order is nil), each stepping its own
// sequence's recurrent state -- its owner's, as rowOwner names it. A
// prompt's rows (base its first row in the prompt, p0 its first position;
// base -1 for none) keep each row's summary as main's row loop does: the
// speculative rows' snapshot, which takes the row at a time, and the
// restore points, where the rows' form ends a call.
func (s *Instance) cpuRecRows(li int, l *model.Layer, x, cs []float32, seq, pos []int, tok []int32, order []int, base, p0 int) (nn.Decline, error) {
	if err := s.cpuRowTables(seq, pos); err != nil {
		return nn.DeclineServes, err
	}
	cb, why := s.cpuBlockOf(li)
	if why != nn.DeclineServes {
		return why, nil
	}
	if f := s.cpuRecWhole(li, l, cb, seq, pos, order, base); f != nil {
		why, err := s.cpuRecRun(li, l, f, x, cs, seq, pos, tok, base, p0)
		if why == nn.DeclineServes && err == nil {
			s.cpu.steps.Whole++
		}
		return why, err
	}
	if len(pos) > 1 {
		s.cpu.steps.Stepped++
	}
	if _, why := s.ws.bind(cb.f, li, l); why != nn.DeclineServes {
		return why, nil
	}
	// x is stream-major: row r's stream k at (k*rows + r)*n_embd.
	d, nrot := s.c.NEmbd, s.c.RopeW()
	stride := len(x) / s.c.Streams()
	b := &s.cpu.bound
	// A step across sessions binds each row's owner for its row: its
	// history, which a block that attends as well reads, is its own.
	lead := s.Session
	for i := range pos {
		r := i
		if order != nil {
			r = order[i]
		}
		o, slot := s.rowOwner(r, seq)
		s.Session = o
		pg := &s.kv.layers[li]
		// A block that attends as well writes and reads the row's history.
		if pg.p > 0 {
			if why := s.cpuHistory(li, pos[r]); why != nn.DeclineServes {
				s.Session = lead
				return why, nil
			}
		}
		if s.cpu.fault == cpuFaultChunk && base > 0 && i == 0 {
			clear(o.view.State(li)[slot*s.dg.StateStride():][:s.dg.StateStride()])
			clear(o.view.Conv(li)[slot*s.dg.ConvStateLen():][:s.dg.ConvStateLen()])
		}
		pages, err := s.cpuPages(li, pg)
		if err != nil {
			s.Session = lead
			return nn.DeclineServes, err
		}
		*b = cpuBound{li: li, l: l, own: s.cpu.dev.Own(cb.f), x: x[r*d:], stride: stride, rows: s.cpu.rows[:], n: 1, kv: pg,
			pages: pages, recLi: s.cpuRecLi(li), recOf: o}
		// One live row, at its own position, as a decode step's.
		s.cpu.pos1 = [2]uint32{1, uint32(pos[r])}
		b.pos = s.cpu.pos1[:]
		if tok != nil {
			s.cpu.tok1[0] = uint32(tok[r])
			b.tok = s.cpu.tok1[:]
		} else if cb.f.Toks {
			s.Session = lead
			return nn.DeclineServes, fmt.Errorf("model: block %d reads its rows' tokens, which the step gives none of", li)
		}
		if rside, pw := s.cpuSideRows(li); pw != 0 && rside != nil {
			b.side = rside[r*pw+cb.sideOff:]
		}
		if nrot > 0 {
			b.cs = cs[r*nrot:]
		}
		if s.bcsSWA != nil {
			b.csSWA = s.bcsSWA[r*s.c.NRotSWA:]
		}
		s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos[r])), uint32(pos[r] + 1), uint32(pos[r])}
		s.cpu.recRows = [3]uint32{1, uint32(slot), uint32(slot)}
		s.cpu.call[0] = 1
		if err := s.cpuForm(b, cb.f, li); err != nil {
			s.Session = lead
			return nn.DeclineServes, err
		}
		if pg.p > 0 && pos[r]+1 > pg.written {
			pg.written = pos[r] + 1
		}
		s.Session = lead
		if base >= 0 {
			s.rows.snapRow(s, li, slot, base+r)
			s.recRowPoint(li, slot, p0+r+1)
		}
	}
	return nn.DeclineServes, nil
}

// cpuRecWhole is block li's rows' form where the step's rows run on it
// whole: several rows, all this instance's, in causal order, each sequence's
// rows in order of position, and no speculative snapshot to take after
// each row; nil where not.
func (s *Instance) cpuRecWhole(li int, l *model.Layer, cb *cpuBlock, seq, pos, order []int, base int) *cpudev.Form {
	if len(pos) < 2 || cb.rows == nil || len(s.rowOwn) > 0 || base >= 0 && s.rows != nil {
		return nil
	}
	for i, r := range order {
		if r != i {
			return nil
		}
	}
	if len(s.cpu.lastPos) < max(s.nseq, 1) {
		s.cpu.lastPos = make([]int, max(s.nseq, 1))
	}
	last := s.cpu.lastPos[:max(s.nseq, 1)]
	for i := range last {
		last[i] = -1
	}
	for i, q := range seq {
		if q < 0 || q >= len(last) || pos[i] <= last[q] {
			return nil
		}
		last[q] = pos[i]
	}
	if _, why := s.ws.bind(cb.rows, li, l); why != nn.DeclineServes {
		return nil
	}
	return cb.rows
}

// cpuRecRun runs linear block li's rows' form f over the step's rows, a
// call at most f's rows; a prompt's (base its first row in the prompt, p0
// its first position) end a call at each restore point inside them and seal
// it there.
func (s *Instance) cpuRecRun(li int, l *model.Layer, f *cpudev.Form, x, cs []float32, seq, pos []int, tok []int32, base, p0 int) (nn.Decline, error) {
	if why := s.cpuRowsReady(li, f, pos); why != nn.DeclineServes {
		return why, nil
	}
	n, d, nrot := len(pos), s.c.NEmbd, s.c.RopeW()
	rside, pw := s.cpuSideRows(li)
	if s.cpu.fault == cpuFaultChunk && base > 0 {
		slot := seq[0]
		clear(s.view.State(li)[slot*s.dg.StateStride():][:s.dg.StateStride()])
		clear(s.view.Conv(li)[slot*s.dg.ConvStateLen():][:s.dg.ConvStateLen()])
	}
	points := base >= 0 && s.recPointsOn()
	for a := 0; a < n; {
		b := n
		for e := a + 1; points && e < n; e++ {
			if s.recPoint(p0 + e) {
				b = e
				break
			}
		}
		side := rside
		if pw != 0 && side != nil {
			side = side[a*pw:]
		}
		if err := s.cpuRowsRun(li, l, f, x[a*d:], n*d, side, cs[a*nrot:], nrot, s.swaRows(a), s.c.NRotSWA, seq[a:b], pos[a:b], rowsOf(tok, a, b)); err != nil {
			return nn.DeclineServes, err
		}
		if base >= 0 {
			s.recRowPoint(li, seq[a], p0+b)
		}
		a = b
	}
	return nn.DeclineServes, nil
}

// rowsOf is rows [a, b) of a step's rows' tokens, nil for a step of none.
func rowsOf(tok []int32, a, b int) []int32 {
	if tok == nil {
		return nil
	}
	return tok[a:b]
}

// cpuRecDesc is the run descriptor (composer.RunDescWords' layout) of a
// call's rows of sequences seq: a run a sequence, its rows in the call's
// order, its slot read and written the sequence's; the entries past the
// runs repeat the last.
func (s *Instance) cpuRecDesc(seq []int) []uint32 {
	k := len(seq)
	if len(s.cpu.recDesc) < 1+7*k {
		s.cpu.recDesc = make([]uint32, 1+7*max(k, cpuRowsCap))
	}
	w := s.cpu.recDesc[:1+7*k]
	w[0] = uint32(k)
	at, runs := 0, 0
	for i, q := range seq {
		if slices.Contains(seq[:i], q) {
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

// cpuRecLi is the block whose recurrent state block li's step binds: its
// own, or under cpuFaultRec the first linear block's.
func (s *Instance) cpuRecLi(li int) int {
	if s.cpu.fault != cpuFaultRec {
		return li
	}
	for r := range s.view.Blocks() {
		if s.view.State(r) != nil {
			return r
		}
	}
	return li
}

// cpuRowsForm is block li's rows' form, its weights the block's, or why
// not.
func (s *Instance) cpuRowsForm(li int, l *model.Layer) (*cpudev.Form, nn.Decline) {
	cb, why := s.cpuBlockOf(li)
	if why != nn.DeclineServes {
		return nil, why
	}
	if cb.rows == nil {
		return nil, cb.rowsWhy
	}
	if _, why := s.ws.bind(cb.rows, li, l); why != nn.DeclineServes {
		return nil, why
	}
	return cb.rows, nn.DeclineServes
}

// cpuRowsReady makes the pages of the history the rows at pos write and
// read resident, in block li's arena, or says why f cannot run them: a
// history past f's key bound, a page outside the arena.
func (s *Instance) cpuRowsReady(li int, f *cpudev.Form, pos []int) nn.Decline {
	hl := s.c.KVSource(li)
	pg := &s.kv.layers[hl]
	if pg.p == 0 {
		return nn.DeclineServes
	}
	// Every row's page is written before any is read: a step's rows stand
	// in any order, and a page holds every sequence's positions.
	start, end := pos[0], 0
	for _, p := range pos {
		s.cpuWritable(hl, p)
		start, end = min(start, pg.keyStart(p)), max(end, p+1)
	}
	if end > f.Keys {
		return nn.DeclineSessions
	}
	if why := s.cpuReadable(hl, start, end-1); why != nn.DeclineServes {
		return why
	}
	return nn.DeclineServes
}

// cpuFaultTable is the page table a step of the layer whose pages pg
// holds reads, its rows' history ending before end: under cpuFaultPage,
// once the layer's first page is full, a table whose first page is the
// page holding the history's last position -- its positions read the newest
// keys, another page's tokens at another rotation. Each layer is pointed into its own arena, so a model
// whose arenas differ (a sliding layer's beside a global one's) is reached.
func (s *Instance) cpuFaultTable(t []uint32, pg *kvPages, end int) []uint32 {
	n := len(t)
	last := (end - 1) / max(pg.p, 1)
	if s.cpu.fault != cpuFaultPage || n < 2 || pg.p == 0 || end <= pg.p+pg.p/4 || last >= n {
		return t
	}
	if len(s.cpu.fpages) != n {
		s.cpu.fpages = make([]uint32, n)
	}
	copy(s.cpu.fpages, t)
	// Not a neighbouring or free slot: under sessions pooled in one arena
	// that is often another session's page of the same positions, or a freed
	// copy of this one's, whose keys move the step too little to see.
	s.cpu.fpages[0] = t[last]
	return s.cpu.fpages
}

// cpuRowsRun runs block li's rows' form f over the rows of x, their history
// the instance's, a call of at most f's capacity at a time. A bidirectional
// run (bidir.go) counts its rows' keys to its end; where the step takes
// more calls than one, the form's codes through its last history write
// (keysEnd) first run over every row, each reading only the keys before
// it, so that every call's rows then read all of them. Where those codes
// write the residual too they run over a copy of it.
func (s *Instance) cpuRowsRun(li int, l *model.Layer, f *cpudev.Form, x []float32, stride int, side, cs []float32, nrot int, swa []float32, nswa int, seq, pos []int, tok []int32) error {
	runs := s.runsAt(li) && s.bidirIn(pos[0], len(pos))
	if runs && len(pos) > f.Rows {
		keys := f.KeysEnd
		if s.cpu.fault == cpuFaultKeysFirst {
			keys = 0
		}
		s.cpu.stashing = stashSave
		err := s.cpuRowsCalls(li, l, f, 0, keys, x, stride, side, cs, nrot, swa, nswa, seq, pos, tok, true)
		s.cpu.stashing = stashLoad
		if err == nil {
			err = s.cpuRowsCalls(li, l, f, keys, len(f.Codes), x, stride, side, cs, nrot, swa, nswa, seq, pos, tok, true)
		}
		s.cpu.stashing = stashNone
		return err
	}
	return s.cpuRowsCalls(li, l, f, 0, len(f.Codes), x, stride, side, cs, nrot, swa, nswa, seq, pos, tok, runs)
}

// cpuStash saves (stashSave) or restores (stashLoad) call c's values form
// f's codes before keysEnd write and the rest read (cpudev.Form.cross).
func (s *Instance) cpuStash(f *cpudev.Form, own [][]uint32, c int) {
	at := 0
	for _, i := range f.Cross {
		at += len(own[i]) * c
	}
	for _, i := range f.Cross {
		n := len(own[i])
		if end := at + n; len(s.cpu.stash) < end {
			s.cpu.stash = append(s.cpu.stash, make([]uint32, end-len(s.cpu.stash))...)
		}
		if s.cpu.stashing == stashSave {
			copy(s.cpu.stash[at:at+n], own[i])
		} else {
			copy(own[i], s.cpu.stash[at:at+n])
		}
		at += n
	}
}

// cpuRowsCalls is cpuRowsRun's calls of f's codes [lo, hi), each row's keys
// counted to its run's end where runs is set; x is the rows' residual,
// stream-major, stride elements from one stream to the next; side is the
// rows' per-layer inputs, nil for a model of none. A form over banks
// (cpudev.Form.split) holds the experts each call's rows pick, from its
// selection to the call's end (cpuExperts).
func (s *Instance) cpuRowsCalls(li int, l *model.Layer, f *cpudev.Form, lo, hi int, x []float32, stride int, side, cs []float32, nrot int, swa []float32, nswa int, seq, pos []int, tok []int32, runs bool) error {
	n := len(pos)
	if f.Toks && tok == nil {
		return fmt.Errorf("model: block %d reads its rows' tokens, which the step gives none of", li)
	}
	hl := s.c.KVSource(li)
	pg := &s.kv.layers[hl]
	pages, err := s.cpuPages(hl, pg)
	if err != nil {
		return err
	}
	d, ns := s.c.NEmbd, s.c.Streams()
	_, pw := s.cpuSideRows(li)
	so := s.cpuSideOff(li)
	slots := max(pg.arenaSlots(), 1)
	b := &s.cpu.bound
	end := 0
	for r0 := 0; r0 < n; r0 += f.Rows {
		k := min(f.Rows, n-r0)
		for i := range k {
			p := pos[r0+i]
			e, st := p+1, pg.keyStart(p)
			if runs {
				e = max(e, s.bidirEnd(p))
				if s.cpu.runWindows {
					st = max(st, s.bidirStart(p))
				}
			}
			if w := s.cpu.rowWins; w != nil && !s.wholePicture(li) {
				st, e = max(st, w[p].Lo), w[p].Hi
			}
			end = max(end, p+1)
			t := s.cpu.rowsTab[i*4 : i*4+4]
			t[0], t[1], t[2], t[3] = uint32(seq[r0+i]*slots), uint32(st), uint32(e), uint32(p)
			s.cpuFaultRowWords(t)
		}
		bx, bstride := x[r0*d:], stride
		var bside []float32
		if side != nil {
			bside = side[r0*pw:]
		}
		staged := f.Unit > 1 && k%f.Unit != 0
		if staged {
			if len(s.cpu.xpad) < f.Rows*d*ns {
				s.cpu.xpad = make([]float32, f.Rows*d*ns)
			}
			for j := range ns {
				copy(s.cpu.xpad[j*f.Rows*d:], bx[j*stride:j*stride+k*d])
			}
			bx, bstride = s.cpu.xpad, f.Rows*d
			if side != nil {
				if len(s.cpu.sidePad) < f.Rows*pw {
					s.cpu.sidePad = make([]float32, f.Rows*pw)
				}
				copy(s.cpu.sidePad, bside[:k*pw])
				bside = s.cpu.sidePad
			}
		}
		if bside != nil {
			bside = bside[so:]
		}
		*b = cpuBound{li: li, l: l, own: s.cpu.dev.Own(f), cs: cs[r0*nrot:], x: bx, stride: bstride, rows: s.cpu.rowsTab, n: k,
			kv: pg, pages: s.cpuFaultTable(pages, pg, end), side: bside}
		if swa != nil {
			b.csSWA = swa[r0*nswa:]
		}
		if len(s.cpu.posTab) < 1+f.Rows {
			s.cpu.posTab = make([]uint32, 1+f.Rows)
		}
		s.cpu.posTab[0] = uint32(k)
		for i := range f.Rows {
			s.cpu.posTab[1+i] = uint32(pos[r0+min(i, k-1)])
		}
		b.pos = s.cpu.posTab
		if tok != nil {
			if len(s.cpu.tokTab) < f.Rows {
				s.cpu.tokTab = make([]uint32, f.Rows)
			}
			for i := range f.Rows {
				s.cpu.tokTab[i] = uint32(tok[r0+min(i, k-1)])
			}
			b.tok = s.cpu.tokTab
		}
		if f.Rec {
			b.recLi, b.recOf, b.rec = s.cpuRecLi(li), s.Session, s.cpuRecDesc(seq[r0:r0+k])
			if n := len(s.view.Conv(b.recLi)); len(s.cpu.convOut) < n {
				s.cpu.convOut = make([]float32, n)
			}
		}
		s.cpu.call[0] = uint32(k)
		if s.cpu.stashing == stashLoad {
			s.cpuStash(f, b.own, r0/f.Rows)
		}
		c := s.cpuCallOf(b, f)
		for i := lo; i < hi; i++ {
			if i == f.RowSplit && f.RowSplit > 0 {
				if err := s.ws.ngView.Ensure(b.own[f.RowSel][:k*f.RowN]); err != nil {
					s.ws.ngView.Drop()
					return err
				}
			}
			if i == f.Split && f.Split > 0 {
				if err := s.cpuExperts(b, f, li, k); err != nil {
					s.expHold.Release()
					s.dropRows(f)
					return err
				}
			}
			if s.cpu.fault == cpuFaultWeight && slices.Contains(f.WReads, i) {
				// Each row's first pick weighted by zero where a kernel
				// reads the weights.
				w := f.SelN / f.Rows
				for r := range k {
					b.own[f.WSel][r*w] = 0
				}
			}
			s.cpu.dev.Call(c, &f.Codes[i])
		}
		s.expHold.Release()
		s.dropRows(f)
		if f.Rec {
			// The runs' windows over their slots.
			L := s.dg.ConvStateLen()
			for i, q := range seq[r0 : r0+k] {
				if !slices.Contains(seq[r0:r0+i], q) {
					copy(s.view.Conv(b.recLi)[q*L:(q+1)*L], s.cpu.convOut[q*L:(q+1)*L])
				}
			}
		}
		if s.cpu.stashing == stashSave {
			s.cpuStash(f, b.own, r0/f.Rows)
		}
		if staged {
			for j := range ns {
				copy(x[j*stride+r0*d:j*stride+(r0+k)*d], s.cpu.xpad[j*f.Rows*d:j*f.Rows*d+k*d])
			}
		}
	}
	if pg.p > 0 && end > pg.written {
		pg.written = end
	}
	return nil
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

// cpuHistory makes the pages block li's step reads and writes resident,
// in the layer's arena, or says why they are not.
func (s *Instance) cpuHistory(li, pos int) nn.Decline {
	s.cpuWritable(li, pos)
	return s.cpuReadable(li, s.kv.layers[li].keyStart(pos), pos)
}

// cpuWritable makes the page block li writes pos into resident and its own.
func (s *Instance) cpuWritable(li, pos int) {
	pg := &s.kv.layers[li]
	s.kvWritable(li, pos)
	n, _ := pg.page(pos)
	pg.grow(n)
	pg.cow(n)
}

// cpuReadable makes the pages of block li's positions [start, pos]
// resident, in the layer's arena, or says why they are not.
func (s *Instance) cpuReadable(li, start, pos int) nn.Decline {
	pg := &s.kv.layers[li]
	n, _ := pg.page(pos)
	s.kvEnsureWindow(li, start, pos-start)
	for p := start / pg.p; p <= n; p++ {
		if !pg.mem.ArenaPage(p) {
			return nn.DeclinePageOut
		}
	}
	return nn.DeclineServes
}

// cpuRunForm runs every code of form f, one after another, on b.
func (s *Instance) cpuRunForm(b *cpuBound, f *cpudev.Form) {
	c := s.cpuCallOf(b, f)
	for i := range f.Codes {
		s.cpu.dev.Call(c, &f.Codes[i])
	}
}

// cpuCallOf is b as the CPU device binds it for a run of form f's codes:
// each source's address, and each of the form's weights' (cpudev.Form.Weights).
func (s *Instance) cpuCallOf(b *cpuBound, f *cpudev.Form) *cpudev.Call {
	c := &s.cpu.bind
	c.Own, c.X, c.Stride, c.N, c.IO = b.own, b.x, b.stride, b.n, b.io
	c.Rope = [2]unsafe.Pointer{addr(b.cs), addr(b.csSWA)}
	c.Planes = [2]unsafe.Pointer{addr(s.cpu.tabPlanes[0]), addr(s.cpu.tabPlanes[1])}
	if b.planes != nil {
		c.Planes = [2]unsafe.Pointer{addr(b.planes), addr(b.planes)}
	}
	a := &c.At
	*a = [cpudev.NumSrc]unsafe.Pointer{}
	a[cpudev.SrcSide] = addr(b.side)
	a[cpudev.SrcRopeRows], a[cpudev.SrcRopeMarks] = addr(b.ropeRows), addr(b.ropeMarks)
	if t := b.kv.arena(); t != nil {
		a[cpudev.SrcKeys], a[cpudev.SrcValues] = addr(t.Arena().Keys()), addr(t.Arena().Values())
	}
	a[cpudev.SrcPages], a[cpudev.SrcRows], a[cpudev.SrcCall] = addr(b.pages), addr(b.rows), addr(s.cpu.call[:])
	if r := b.recOf; r != nil && b.recLi >= 0 && b.recLi < r.view.Blocks() {
		a[cpudev.SrcRecState], a[cpudev.SrcRecConv] = addr(r.view.State(b.recLi)), addr(r.view.Conv(b.recLi))
		a[cpudev.SrcRecConvOut] = a[cpudev.SrcRecConv]
	}
	a[cpudev.SrcRecRows] = addr(s.cpu.recRows[:])
	if b.rec != nil {
		a[cpudev.SrcRecConvOut], a[cpudev.SrcRecRows] = addr(s.cpu.convOut), addr(b.rec)
	}
	a[cpudev.SrcZero] = addr(s.cpu.zero)
	a[cpudev.SrcRequest], a[cpudev.SrcTokens], a[cpudev.SrcPos] = addr(b.req), addr(b.tok), addr(b.pos)
	a[cpudev.SrcPicture] = addr(b.pic)
	c.Weights, _ = s.ws.bind(f, b.li, b.l)
	return c
}

// addr is x's first element's address, nil for a nil slice.
func addr[T any](x []T) unsafe.Pointer { return unsafe.Pointer(unsafe.SliceData(x)) }

// unmapOwn returns the own buffers and the n-gram table's view held off
// the heap.
func (s *Instance) unmapOwn() {
	if s.cpu.dev != nil {
		s.cpu.dev.Close()
	}
	if s.ws.ngView != nil {
		s.ws.ngView.Close()
		s.ws.ngView = nil
	}
}

// cpuPlace composes a new instance's blocks on the CPU: each block of the
// instance's range no device holds, once, at its placement (cpuCompose).
func (s *Instance) cpuPlace() {
	for i := range s.cpu.tabPlanes {
		s.cpu.tabPlanes[i] = s.tablePlanes(i)
	}
	// A segment's blocks are past the text model's in the one block space.
	n := max(s.c.NLayer, s.hi)
	s.cpu.blocks = make([]*cpuBlock, n)
	s.cpu.composed = make([]int, n)
	// A word block for each worker of the widest region the pool runs, a
	// prefill's on the P and E cores among them.
	// Made once: a placement again (a new history format) keeps the forms'
	// own buffers.
	if s.cpu.dev == nil {
		s.cpu.dev = cpudev.NewDevice(s.cpu.workers)
		s.cpu.dev.Prof = s.prof
	}
	s.cpu.zero = make([]uint32, 4)
	for li := s.lo; li < min(s.hi, n); li++ {
		if !s.devAt(li) {
			s.cpuCompose(li)
		}
	}
	s.cpuPlaceHead()
	s.cpuPlaceStreams()
	s.cpuPlacePool()
	s.cpuPlaceMTP()
	s.cpuPlaceEmbed()
	s.cpuPlaceTables()
	s.cpuPlaceKV()
}

// cpuRelocated follows block li onto the CPU, composed for it there, or off
// it, its composition released.
func (s *Instance) cpuRelocated(li int, onCPU bool) {
	if li < 0 || li >= len(s.cpu.blocks) {
		return
	}
	if onCPU {
		s.cpuCompose(li)
		return
	}
	s.cpu.blocks[li] = nil
}

// cpuCompose composes block li for the CPU device: its form, through the
// model's keyed cache, and its bindings, counted (CPUCompositions).
func (s *Instance) cpuCompose(li int) {
	cb := &cpuBlock{}
	s.cpu.blocks[li] = cb
	cb.why = s.cpuWhy(li)
	if cb.why != nn.DeclineServes {
		return
	}
	f, err := s.formCPUBlock(li, cb, 1)
	if err != nil {
		cb.why = s.cpuFormDecline(err)
		return
	}
	cb.f = f
	if cb.rows, err = s.formCPUBlock(li, cb, cpuRowsCap); err != nil {
		cb.rowsWhy = s.cpuFormDecline(err)
	}
	s.cpu.composed[li]++
	if cb.rows != nil && len(s.cpu.rowsTab) < cpuRowsCap*4 {
		s.cpu.rowsTab = make([]uint32, cpuRowsCap*4)
		s.cpu.zero = make([]uint32, cpuRowsCap*4)
	}
}

// cpuFormDecline is the reason a form's error declines its block by, the
// first error kept (CPUFormErr).
func (s *Instance) cpuFormDecline(err error) nn.Decline {
	if s.cpu.formErr == nil {
		s.cpu.formErr = err
	}
	var d cpudev.DeclineError
	if errors.As(err, &d) {
		return d.Why
	}
	return nn.DeclineNotFormed
}

// cpuBlockOf is block li's composition on the CPU, made at its placement.
func (s *Instance) cpuBlockOf(li int) (*cpuBlock, nn.Decline) {
	if li >= len(s.cpu.blocks) || s.cpu.blocks[li] == nil {
		return nil, nn.DeclineUnplaced
	}
	cb := s.cpu.blocks[li]
	return cb, cb.why
}

// cpuWhy is what keeps block li off the composer's CPU path before it is
// asked: what its graph and history need that the CPU device does not hold.
func (s *Instance) cpuWhy(li int) nn.Decline {
	c := s.c
	g := s.m.Composing().BlockGraph(li)
	// A block sharing another's history reads its source's pages.
	pg := &s.kv.layers[c.KVSource(li)]
	switch {
	case !composer.LowersCPU(uint64(s.cpuISA())):
		return nn.DeclineISA
	case g == nil:
		return nn.DeclineNoGraph
	case len(g.Out) != c.Streams() || len(g.In) > len(g.Out)+1 || len(g.In) > len(g.Out) && s.cpuSide(li) == nil:
		return nn.DeclineFeeds
	case c.MoEAt(li) && !s.cpuBanksPaged(li):
		// The forms read a bank as one view of its experts in the host's
		// expert pages (jlm.File.BankView); a bank with none is not read.
		return nn.DeclineExperts
	case c.LayerKind(li).Recurrent() && (s.view.State(li) == nil || s.view.Conv(li) == nil):
		return nn.DeclineRecurrent
	case s.kvl.fmt != cpu.KVF16 && s.kvl.fmt != cpu.KVF32:
		// The CPU's profile keeps a history binary16 itself (composer.Halves),
		// or at f32 without that pass.
		return nn.DeclineKVFormat
	case pg.p > 0 && pg.arena() == nil:
		return nn.DeclinePageOut
	}
	return nn.DeclineServes
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

// cpuSideOff is block li's sideOff, 0 for a block not composed.
func (s *Instance) cpuSideOff(li int) int {
	if li < len(s.cpu.blocks) && s.cpu.blocks[li] != nil {
		return s.cpu.blocks[li].sideOff
	}
	return 0
}

// formCPUBlock is block li's form on the CPU at rows rows through the
// model's keyed cache: its history the session's sequences' pages
// (cpuPages), each a history of the instance's context.
func (s *Instance) formCPUBlock(li int, cb *cpuBlock, rows int) (*cpudev.Form, error) {
	g := s.m.Composing().BlockGraph(li)
	cb.graph = g
	gr := *g
	gr.Block = 0
	cb.sideOff = sideAtBinding(&gr)
	if s.ws.ngView == nil && s.m.Ngram != nil && cpudev.TableRowsOf(&gr) >= 0 {
		v, err := s.fr.NewRowView(s.m.Ngram.E)
		if err != nil {
			return nil, err
		}
		s.ws.ngView = v
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
	req := composer.Request{Graph: &gr, Formats: formats, Strides: s.m.CpuStrides(s.fr, li, g),
		Step:    composer.Step{Rows: rows, Page: page, Keys: max(pg.arenaSlots(), 1) * page, ActWindow: s.m.ActWindow, Seqs: max(s.nseq, 1)},
		History: composer.History{ValuesInKeys: pg.latent},
		Device:  s.composedCPU()}
	// A block turns by the rotary its rope ops state, of the instance's
	// (rotaries): each its own table, bound by instance.
	req.Rotaries = s.rotaries()
	return s.cpuFormOf(req, &gr, rows, false, false, s.kvl.fmt == cpu.KVF32)
}

// cpuFormOf is req's form on the CPU, mapped once for the model by its Key;
// whole is cpudev.BuildForm's.
func (s *Instance) cpuFormOf(req composer.Request, gr *graph.Graph, rows int, apart, whole, f32 bool) (*cpudev.Form, error) {
	return s.eng.cpu.Forms(s.m.Id).Of(req, gr, rows, apart, whole, f32, s.opt.cpuFloat)
}

// dropRows hands back the table's rows form f's call read in.
func (s *Instance) dropRows(f *cpudev.Form) {
	if f.RowSplit > 0 {
		s.ws.ngView.Drop()
	}
}

// cpuBanksPaged says block li's routed banks are each one view of its
// experts in the host's expert pages (jlm.File.BankView).
func (s *Instance) cpuBanksPaged(li int) bool {
	l := &s.m.Layers[li]
	if s.m.Container == nil || l.LegacyExperts {
		return false
	}
	if len(s.ws.banks) < len(s.m.Layers) {
		s.ws.banks = make([][3][3]unsafe.Pointer, len(s.m.Layers))
	}
	for i, t := range []*model.Tensor{&l.Gate, &l.Up, &l.Down} {
		if t.E == nil {
			continue
		}
		planes, _, ok := s.fr.BankView(t.E)
		if !ok {
			return false
		}
		s.ws.banks[li][i] = planes
	}
	return l.Up.E != nil && l.Down.E != nil
}

// cpuForm runs form f of block li over the step b: its codes in order, and
// for a form over banks (cpudev.Form.split) the selection's experts made
// resident and held between its halves -- the pager's own fault path
// (ensureExperts), each read from the file where the budget evicted it --
// and counted as the host's own mixture counts them.
func (s *Instance) cpuForm(b *cpuBound, f *cpudev.Form, li int) error {
	c := s.cpuCallOf(b, f)
	lo := 0
	if f.RowSplit > 0 {
		// The table's rows the call's hash names, read in for its gathers and
		// handed back after the call.
		s.cpu.dev.Calls(c, f.Codes[:f.RowSplit])
		defer s.ws.ngView.Drop()
		if err := s.ws.ngView.Ensure(b.own[f.RowSel][:b.n*f.RowN]); err != nil {
			return err
		}
		lo = f.RowSplit
	}
	if f.Split <= 0 {
		s.cpu.dev.Calls(c, f.Codes[lo:])
		return nil
	}
	s.cpu.dev.Calls(c, f.Codes[lo:f.Split])
	err := s.cpuExperts(b, f, li, 1)
	defer s.expHold.Release()
	if err != nil {
		return err
	}
	if !s.rowsOfChunk {
		s.hot.add(li, s.cpu.sel)
	}
	s.cpu.dev.Calls(c, f.Codes[f.Split:])
	return nil
}

// cpuExperts makes the experts the n rows of a call of form f select
// (cpudev.Form.sel, a row's picks after another's) resident and held until
// s.expHold's release, each once however many rows pick it -- the pager's
// own fault path (ensureExperts), each read from the file where the budget
// evicted it -- and counts them resident as main's mixture does. s.cpu.sel
// is the experts held, in the order the rows first pick them.
func (s *Instance) cpuExperts(b *cpuBound, f *cpudev.Form, li, n int) error {
	k := f.SelN / max(f.Rows, 1)
	picks := b.own[f.Sel][:n*k]
	ne := s.c.NExpert
	if s.cpu.fault == cpuFaultPicks && n > 1 {
		// The first row given the second's picks, with its own weights.
		copy(picks[:k], picks[k:2*k])
	}
	if len(s.cpu.picked) < ne {
		s.cpu.picked = make([]bool, ne)
	}
	s.cpu.sel = s.cpu.sel[:0]
	for _, x := range picks {
		if int(x) >= ne {
			// No page: ensureExperts names it.
			s.cpu.sel = append(s.cpu.sel, int32(x))
		} else if !s.cpu.picked[x] {
			s.cpu.picked[x] = true
			s.cpu.sel = append(s.cpu.sel, int32(x))
		}
	}
	other := int32(-1)
	if s.cpu.fault == cpuFaultExpert {
		// Another expert's page made resident and held than the first the
		// kernels read: one no row picks, where there is one.
		e := s.cpu.sel[0]
		for j := 1; j < ne && other < 0; j++ {
			if x := (e + int32(j)) % int32(ne); !s.cpu.picked[x] {
				other = x
			}
		}
	}
	for _, x := range s.cpu.sel {
		if int(x) < ne {
			s.cpu.picked[x] = false
		}
	}
	if other >= 0 {
		s.cpu.sel[0] = other
	}
	s.bind.Lock()
	err := s.m.EnsureExperts(s.fr, li, s.cpu.sel, &s.expHold, s.layer(li))
	s.bind.Unlock()
	if err != nil {
		return err
	}
	s.res.Touch(li, ne, s.cpu.sel)
	return nil
}

// arena is the table of the layer's arena, nil for a layer with none (or no
// layer).
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

// cpuPages is the table block li's forms read its history pg through: its
// session's view's (cpudev.View), or a zero table for a block keeping none.
func (s *Instance) cpuPages(li int, pg *kvPages) ([]uint32, error) {
	nseq := max(s.nseq, 1)
	t := s.Session.view.At(li)
	switch {
	case t != pg.arena():
		return nil, fmt.Errorf("model: block %d's pages are not its session's view's", li)
	case t == nil:
		if len(s.cpu.nopages) < nseq {
			s.cpu.nopages = make([]uint32, nseq)
		}
		return s.cpu.nopages[:nseq], nil
	}
	return t.Pages(nseq), nil
}
