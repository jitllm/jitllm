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
	s.cpu.steps.Decline(nn.PartBlock, why)
	if why == nn.DeclineNotFormed && s.cpu.comp.Err() != nil {
		return fmt.Errorf("model: the composer does not serve block %d on the CPU: %s (the first: %w)", li, nn.DeclineName(int(why)), s.cpu.comp.Err())
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
	workers  *cpu.Pool
	blocks   []*cpudev.Block
	composed []int
	// steps is what the CPU served the instance, part by part.
	steps nn.Steps
	// nopages is a block's page table that keeps no history (cpuPages).
	nopages []uint32
	// oneFresh and tabsFresh say the decode row's and the step's rows'
	// rotary tables are made (cpuOneTable, cpuRowTables); tabSeq is the
	// rows' slots as the tables read them.
	oneFresh, tabsFresh bool
	tabSeq              []int
	// spans is a call's rows' spans as the device reads them (cpuRowsRun),
	// rowsIn the rows' call itself; rseq and rpos a step's rows' sequences
	// and positions; lastPos each sequence's last position a step's rows
	// reach (cpuRecWhole).
	spans      []uint32
	rowsIn     cpudev.Rows
	lastPos    []int
	rseq, rpos []int
	onePos     [1]int
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
	dev  *cpudev.Device
	comp *cpudev.Composer
	run  *cpudev.Runner
	// rowRun runs a block's rows.
	rowRun *cpudev.Blocks
	bound  cpudev.Bound
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

// cpuRowsCap is the rows the CPU's form of several rows is composed for: a
// step of more runs in calls of this many, one after another.
const cpuRowsCap = 64

// CPUForm is how the composer forms this host's CPU device
// (composer.CPUForm).
func CPUForm() string {
	return composer.CPUForm(cpudev.Target(cpu.Probe().ISA(), 1), runtime.GOARCH == "arm64")
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
	w, why := s.ws.bind(cb.F, li, l)
	if why != nn.DeclineServes {
		return why, nil
	}
	r := cpudev.Row{X: s.resid(), Stride: s.c.NEmbd, CS: cs, CSSWA: s.csSWA, Side: s.cpuSide(li), Pages: pages,
		KeyStart: pg.keyStart(pos), Pos: pos, PagePos: pg.p, Tok: s.tok, Toks: true, RecLi: s.cpuRecLi(li), Weights: w, Faults: true}
	r.Keys, r.Values = kvAddrs(pg)
	r.RecState, r.RecConv = recAddrs(s.Session, r.RecLi)
	if err := s.cpu.rowRun.Row(li, cb, &r); err != nil {
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
	if _, why := s.ws.bind(cb.F, li, l); why != nn.DeclineServes {
		return why, nil
	}
	// x is stream-major: row r's stream k at (k*rows + r)*n_embd.
	d, nrot := s.c.NEmbd, s.c.RopeW()
	stride := len(x) / s.c.Streams()
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
		w, _ := s.ws.bind(cb.F, li, l)
		row := cpudev.Row{X: x[r*d:], Stride: stride, Pages: pages, KeyStart: pg.keyStart(pos[r]), Pos: pos[r],
			RecLi: s.cpuRecLi(li), Slot: slot, Weights: w}
		row.Keys, row.Values = kvAddrs(pg)
		row.RecState, row.RecConv = recAddrs(o, row.RecLi)
		if tok != nil {
			row.Tok, row.Toks = tok[r], true
		} else if cb.F.Toks {
			s.Session = lead
			return nn.DeclineServes, fmt.Errorf("model: block %d reads its rows' tokens, which the step gives none of", li)
		}
		if rside, pw := s.cpuSideRows(li); pw != 0 && rside != nil {
			row.Side = rside[r*pw:]
		}
		if nrot > 0 {
			row.CS = cs[r*nrot:]
		}
		if s.bcsSWA != nil {
			row.CSSWA = s.bcsSWA[r*s.c.NRotSWA:]
		}
		if err := s.cpu.rowRun.Row(li, cb, &row); err != nil {
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
func (s *Instance) cpuRecWhole(li int, l *model.Layer, cb *cpudev.Block, seq, pos, order []int, base int) *cpudev.Form {
	if len(pos) < 2 || cb.Rows == nil || len(s.rowOwn) > 0 || base >= 0 && s.rows != nil {
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
	if _, why := s.ws.bind(cb.Rows, li, l); why != nn.DeclineServes {
		return nil
	}
	return cb.Rows
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
	if cb.Rows == nil {
		return nil, cb.RowsWhy
	}
	if _, why := s.ws.bind(cb.Rows, li, l); why != nn.DeclineServes {
		return nil, why
	}
	return cb.Rows, nn.DeclineServes
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
	// A segment's blocks are past the text model's in the one block space.
	n := max(s.c.NLayer, s.hi)
	s.cpu.blocks = make([]*cpudev.Block, n)
	s.cpu.composed = make([]int, n)
	// A word block for each worker of the widest region the pool runs, a
	// prefill's on the P and E cores among them.
	// Made once: a placement again (a new history format) keeps the forms'
	// own buffers.
	if s.cpu.dev == nil {
		s.cpu.dev = cpudev.NewDevice(s.cpu.workers)
		s.cpu.dev.Prof = s.prof
	}
	planes := [2][]float32{s.tablePlanes(0), s.tablePlanes(1)}
	s.lease = blockLease{m: s.m, fr: s.fr, mu: &s.bind, hold: &s.expHold, res: s.res, hot: s.hot, layers: s.layers,
		lo: s.lo, ws: &s.ws}
	if s.cpu.run == nil {
		s.cpu.run = cpudev.NewRunner(s.cpu.dev, &s.lease, planes, s.c.NExpert)
	}
	if s.cpu.rowRun == nil {
		s.cpu.rowRun = cpudev.NewBlocks(s.cpu.dev, s.cpu.run)
	}
	s.cpu.run.SetPlanes(planes)
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
	cb := &cpudev.Block{}
	s.cpu.blocks[li] = cb
	cb.Why = s.cpuWhy(li)
	if cb.Why != nn.DeclineServes {
		return
	}
	f, err := s.formCPUBlock(li, cb, 1)
	if err != nil {
		cb.Why = s.cpu.comp.Decline(err)
		return
	}
	cb.F = f
	if cb.Rows, err = s.formCPUBlock(li, cb, cpuRowsCap); err != nil {
		cb.RowsWhy = s.cpu.comp.Decline(err)
	}
	s.cpu.composed[li]++
	if cb.Rows != nil {
		s.cpu.rowRun.Reserve(cpuRowsCap)
	}
}

// cpuBlockOf is block li's composition on the CPU, made at its placement.
func (s *Instance) cpuBlockOf(li int) (*cpudev.Block, nn.Decline) {
	if li >= len(s.cpu.blocks) || s.cpu.blocks[li] == nil {
		return nil, nn.DeclineUnplaced
	}
	cb := s.cpu.blocks[li]
	return cb, cb.Why
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
		return s.cpu.blocks[li].SideOff
	}
	return 0
}

// formCPUBlock is block li's form on the CPU at rows rows through the
// model's keyed cache: its history the session's sequences' pages
// (cpuPages), each a history of the instance's context.
func (s *Instance) formCPUBlock(li int, cb *cpudev.Block, rows int) (*cpudev.Form, error) {
	g := s.m.Composing().BlockGraph(li)
	cb.Graph = g
	gr := *g
	gr.Block = 0
	cb.SideOff = sideAtBinding(&gr)
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
		Device:  s.cpu.comp.Device()}
	// A block turns by the rotary its rope ops state, of the instance's
	// (rotaries): each its own table, bound by instance.
	req.Rotaries = s.rotaries()
	return s.cpu.comp.Form(req, &gr, rows, false, false, s.kvl.fmt == cpu.KVF32)
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

// arena is the table of the layer's arena, nil for a layer with none (or no
// layer).
// kvAddrs is where history pg's keys and values are read, nil for none.
func kvAddrs(pg *kvPages) (keys, values unsafe.Pointer) {
	if t := pg.arena(); t != nil {
		return addr(t.Arena().Keys()), addr(t.Arena().Values())
	}
	return nil, nil
}

// recAddrs is where session o holds block li's recurrent state and window,
// nil where it holds none.
func recAddrs(o *Session, li int) (state, conv unsafe.Pointer) {
	if o == nil || li < 0 || li >= o.view.Blocks() {
		return nil, nil
	}
	return addr(o.view.State(li)), addr(o.view.Conv(li))
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

// cpuRowsRun runs form f of block li (l its copy) over the rows of x, the
// stream-major residual stride apart, at positions pos of sequences seq with
// tokens tok, their per-layer inputs side, rotary tables cs and swa, on the
// CPU device (cpudev.Blocks.Rows): each row's history read from where it
// starts to where its bidirectional run or its window ends.
func (s *Instance) cpuRowsRun(li int, l *model.Layer, f *cpudev.Form, x []float32, stride int, side, cs []float32, nrot int, swa []float32, nswa int, seq, pos []int, tok []int32) error {
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
	runs := s.runsAt(li) && s.bidirIn(pos[0], n)
	slots := max(pg.arenaSlots(), 1)
	if cap(s.cpu.spans) < 4*n {
		s.cpu.spans = make([]uint32, 4*n)
	}
	spans := s.cpu.spans[:4*n]
	end := 0
	for i, p := range pos {
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
		spans[i*4], spans[i*4+1], spans[i*4+2], spans[i*4+3] = uint32(seq[i]*slots), uint32(st), uint32(e), uint32(p)
	}
	w, _ := s.ws.bind(f, li, l)
	_, pw := s.cpuSideRows(li)
	in := &s.cpu.rowsIn
	*in = cpudev.Rows{X: x, Stride: stride, Width: s.c.NEmbd, Streams: s.c.Streams(), Side: side, SideW: pw,
		SideOff: s.cpuSideOff(li), CS: cs, NRot: nrot, CSSWA: swa, NSWA: nswa, Pages: pages, PagePos: pg.p, Spans: spans,
		Pos: pos, Tok: tok, Seq: seq, Weights: w}
	in.Keys, in.Values = kvAddrs(pg)
	if f.Rec {
		in.RecLi = s.cpuRecLi(li)
		in.RecState, in.RecConv = recAddrs(s.Session, in.RecLi)
		in.Conv, in.ConvLen = s.view.Conv(in.RecLi), s.dg.ConvStateLen()
	}
	if err := s.cpu.rowRun.Rows(li, f, in, runs); err != nil {
		return err
	}
	if pg.p > 0 && end > pg.written {
		pg.written = end
	}
	return nil
}
