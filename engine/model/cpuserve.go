package model

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	hw "github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/offheap"
)

// The CPU is a device like a card: a block placed on it runs what the
// composer forms for it from the block's graph (Model.Composing), through
// the one keyed cache, on the instance's own buffers -- the residual, the
// block's weights and vectors by role, its rotary table and its history in
// the host's KV pages (kvArena) -- each kernel mapped once for the model and
// called through the trampoline, the calls of a kernel over heads spread
// over the instance's pool. Every block step on the CPU is counted served or
// declined by name (CPUBlockSteps), as the tier counts a card's.

// CPUBlockSteps is how many block steps on the CPU the composer served, and
// how many it declined, by reason (composer.DeclineName); a decline is the
// step's error (cpuRefused).
func (s *Instance) CPUBlockSteps() (served int, declined [composer.NumDeclines]int) {
	return s.cpu.served, s.cpu.declined
}

// cpuRefused counts the composer's decline of block li's step on the CPU and
// is the step's error: the composer is the only path, so no other kernels run
// the block.
func (s *Instance) cpuRefused(li int, why composer.Decline) error {
	s.cpu.declined[why]++
	if why == composer.DeclineNotFormed && s.cpu.formErr != nil {
		return fmt.Errorf("model: the composer does not serve block %d on the CPU: %s (the first: %w)", li, composer.DeclineName(int(why)), s.cpu.formErr)
	}
	return fmt.Errorf("model: the composer does not serve block %d on the CPU: %s", li, composer.DeclineName(int(why)))
}

// cpuHeadRefused is the composer's decline of the head on the CPU as the
// step's error, as cpuRefused is a block's.
func cpuHeadRefused(why composer.Decline) error {
	return fmt.Errorf("model: the composer does not serve the head on the CPU: %s", composer.DeclineName(int(why)))
}

// CPURecSteps is how many steps of several rows the CPU's linear blocks
// ran whole on their rows' form, and how many they stepped a row at a time.
func (s *Instance) CPURecSteps() (whole, stepped int) { return s.cpu.recWhole, s.cpu.recStepped }

// CPUFormErr is the first error the composer gave forming a block of this
// instance's model for the CPU, nil for none.
func (s *Instance) CPUFormErr() error { return s.cpu.formErr }

// The argument sources of a composed CPU kernel's call words.
const (
	cpuArgValue uint8 = iota
	cpuArgX
	cpuArgOwn
	cpuArgWeight
	cpuArgVector
	cpuArgRope
	cpuArgKeys
	cpuArgRopeRows
	cpuArgRopeMarks
	cpuArgEntKeys
	cpuArgEntPages
	cpuArgEntRows
	cpuArgValues
	cpuArgPages
	cpuArgRows
	cpuArgZero
	cpuArgCoord
	cpuArgWord
	cpuArgCall
	// cpuArgRecState and cpuArgRecConv are a linear block's recurrent states
	// and its convolution's windows, a slot a sequence, read and written in
	// place; cpuArgRecRows the call's recurrent descriptor, which names the
	// call's sequence's slot. cpuArgRecConvOut is where a call's windows are
	// written: in place for a row, apart for a step of several (its runs
	// past the step's write a run's window again from the window read).
	cpuArgRecState
	cpuArgRecConv
	cpuArgRecConvOut
	// cpuArgShare is the rows a call of a Share code takes (composer.Code.Share).
	cpuArgShare
	cpuArgRecRows
	// cpuArgPoolPos is the positions table a pooled readout reads its live
	// count from (cpupool.go).
	cpuArgPoolPos
	// cpuArgIO is an input or output of a graph whose output is not its
	// input (the head's, the sampler's): the bound buffer of that place,
	// the inputs' then the outputs' (cpuBound.io).
	cpuArgIO
	// cpuArgSide is a block graph's input past its residual streams, the
	// rows' per-layer inputs (cpuBound.side).
	cpuArgSide
	// cpuArgRequest is the step's request words (cpuBound.req), cpuArgPos
	// its positions (cpuBound.pos).
	cpuArgRequest
	cpuArgPos
	// cpuArgPlanes is the block's rotary planes (nn.Rope.TabPlanes), which
	// a kernel turning at positions of its own reads (cpuForms.planes).
	cpuArgPlanes
	// cpuArgTokens is the step's tokens, a word a row (cpuBound.tok).
	cpuArgTokens
)

// cpuArg is one call word: its byte field in cpu.Args and where its value
// comes from, with the elements it is offset by.
type cpuArg struct {
	field  uintptr
	src    uint8
	region uint8
	coord  uint8
	role   jlm.Role
	i      int
	off    int
	elem   int
	val    uint64
	// stream is the residual stream a cpuArgX address is in.
	stream int
	// per is the elements the address moves a unit of each call
	// coordinate (composer.CallArg.Per), moves whether any does.
	per   [2]int
	moves bool
	// inst is a rotary table's instance (cpuArgRope, cpuArgPlanes).
	inst int
}

// cpuCode is one kernel of a form, mapped, and its call words; reads and
// writes are the residual streams whose input it reads and whose output it
// writes, a bit a stream.
type cpuCode struct {
	reads, writes uint64
	code          *cpu.Code
	args          []cpuArg
	calls         []int
	n             int
	// serial is a kernel over calls that write a buffer every call shares.
	serial bool
	// live is the code's hold on a call's rows (composer.Code's Live and
	// LiveUnits): a call of rows rows runs the units of the leading axis
	// they reach, each of inner calls (composer.RowCalls).
	live  composer.Code
	inner int
	// tiles is a kernel whose inner calls tile a weight (composer.Code.TileMajor):
	// its calls run a tile for every live row before the next tile.
	tiles bool
	// name is the composer's name for the kernel; nanos and runs its time
	// and launches under WithProfile.
	name        string
	desc        string
	nanos, runs int64
	// pcalls, pcallNs, pstartNs and pmaxNs are its calls under WithProfile:
	// their count, time, start after the launch's and each launch's last start.
	pcalls, pcallNs, pstartNs, pmaxNs int64
	// rd and wr are the buffers the kernel's calls read and write, as the
	// call site binds them; join is how many kernels after it run in its
	// pool phase, none reading or writing what another of the phase writes,
	// and ends their calls' running totals (cpuGroup).
	rd, wr []cpuBuf
	join   int
	// chain is how many kernels after it, some reading what one before
	// writes, run in its phase unit by unit (chainPhases), of units units;
	// sliced is the rows of each buffer a call reaches alone (cpuCode.slice).
	chain, units int
	sliced       []cpuSlice
	// share is the code's hold on a weight's rows sized at the launch
	// (composer.Code.Share).
	share composer.Code
}

type cpuSlice struct {
	buf cpuBuf
	per int
}

// cpuBuf is a buffer a call word binds, by where the call site finds it.
type cpuBuf struct {
	src    uint8
	region uint8
	role   jlm.Role
	i      int
	stream int
}

// cpuBlockForm is a block's kernels for the CPU at one composer Key, mapped
// once for the model, and the words of each of its own buffers.
type cpuBlockForm struct {
	codes []cpuCode
	// rec says the graph steps a recurrent state: a recurrent kind's block
	// whose convolution is centred over the call keeps none.
	rec bool
	// rows is the rows the form holds: 1 for decode's, else its capacity;
	// keys the positions of a sequence's history it attends at most.
	rows, keys int
	// unit is the most rows a unit of a kernel's leading axis holds
	// (composer.RowJam): a call's rows past a whole number of them are run
	// from a copy of the residual, so a unit's rows past the call's write
	// no other rows.
	unit int
	own  []int
	// consts is an own buffer's constant words, by its index.
	consts map[int][]uint32
	// keysEnd is the codes up to the last that writes the history: a
	// bidirectional run of more rows than a call runs them over every row
	// first (cpuRowsRun). keysX says one of them writes the residual.
	keysEnd int
	keysX   bool
	// cross is the own buffers codes[:keysEnd] write and the rest read:
	// what a bidirectional run of more rows than a call holds for each
	// call between its two passes (cpuRowsRun).
	cross []int
	// A form over banks in the host's expert pages runs in two halves
	// (cpuForm): its codes before split, up to the selection, then the rest,
	// which read the banks once the selected experts' pages are resident
	// and held. sel is the own buffer holding the selection, of selN words,
	// and wsel the picks' weights' (weightsOf), -1 where none holds them,
	// read by the codes wreads.
	split, sel, selN, wsel int
	wreads                 []int
}

// ownUse is a code's use of an own buffer, by their indices.
type ownUse struct{ code, own int }

func (f *cpuBlockForm) close() {
	for i := range f.codes {
		f.codes[i].code.Close()
	}
}

// cpuForms is a model's composed CPU forms by Key, and the registry they
// are asked of.
type cpuForms struct {
	mu  sync.Mutex
	reg *composer.Registry
	// reg32 is reg with histories kept at f32 (composer.Registry.WithoutHalves).
	reg32 *composer.Registry
	m     map[composer.Key]*cpuBlockForm
	err   map[composer.Key]error
}

func (c *cpuForms) close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, f := range c.m {
		f.close()
	}
	c.m = nil
}

// cpuState is a instance's composed CPU blocks: each block's form and its own
// buffers, the history tables, and the census.
type cpuState struct {
	launch    int
	launchSet bool
	blocks    []*cpuBlock
	composed  []int
	served    int
	declined  [composer.NumDeclines]int
	formErr   error
	// fpages is a layer's page table under cpuFaultPage (cpuFaultTable).
	fpages []uint32
	rows   [4]uint32
	zero   []uint32
	// entRow and entTab are a block's entry descriptors (cpuEntBind), a
	// decode's and a call's rows'.
	entRow [4]uint32
	entTab []uint32
	// tok is the step's rows' tokens (cpuTokens), tok1 a decode's.
	tok  []uint32
	tok1 [1]uint32
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
	convOut []float32
	// recWhole counts the steps of several rows a linear block ran on its
	// rows' form, recStepped those it stepped a row at a time.
	recWhole, recStepped int
	rseq, rpos           []int
	onePos               [1]int
	// sel is a call's experts as the pager's fault path takes them, picked
	// marking those already in it, by expert (cpuExperts).
	sel    []int32
	picked []bool
	// banks is each block's routed banks' planes in the host's expert
	// pages, by bank (gate, up, down) and plane (jlm.File.BankView), set at
	// the block's composition.
	banks [][3][3]unsafe.Pointer
	// xpad is a call's residual rows where its kernels' units reach past
	// them (cpuBlockForm.unit).
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
	// own is each form's own buffers, by form: blocks of one form run one
	// after another, so they share them.
	own map[*cpuBlockForm][][]uint32
	// mapped is the own buffers held off the heap (cpuOwnOffHeap), unmapped
	// with the instance.
	mapped [][]byte
	// args is a call's words for each pool worker (cpuArgBlocks).
	args []cpu.Args
	// part is the workers the launch running sizes its calls for, and ends
	// its phase's kernels' calls' running totals (cpuGroup).
	part int
	ends []int
	// t0 and maxStart are the profiled launch's start and its latest call's.
	t0       time.Time
	maxStart int64
	run      cpuRegion
	bound    cpuBound
	// fault is a violation a gate sets: the bindings rewritten so a kernel
	// reads the wrong history page or the wrong first row.
	fault   cpuFault
	head    cpuHeadState
	streams cpuStreamsState
	// tabPlanes is each rotary's planes as its composed table reads them
	// (composer.TablePlanes), folded from the model's rotaries when the
	// instance places its blocks (cpuTablePlanes).
	tabPlanes [2][]float32
	pool      cpuPoolState
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
	f     *cpuBlockForm
	rows  *cpuBlockForm
	// sideOff is where the block's slice of the rows' per-layer inputs
	// starts in each row: the call binds the side input there, so blocks
	// that differ by their slice alone share a form.
	sideOff int
	why     composer.Decline
	// rowsWhy is why the rows' form is not served, where it is not.
	rowsWhy composer.Decline
}

// cpuRowsCap is the rows the CPU's form of several rows is composed for: a
// step of more runs in calls of this many, one after another.
const cpuRowsCap = 64

// cpuRegion is the kernel a pool region calls, its calls the region's range.
type cpuRegion struct {
	s    *Instance
	b    *cpuBound
	c    *cpuCode
	live int
	fn   func(worker, lo, hi int)
	// g is a phase's kernels (cpuGroup), gfn the region that runs them;
	// cfn runs them unit by unit, units units (cpuChain).
	g     []cpuCode
	gfn   func(worker, lo, hi int)
	cfn   func(worker, lo, hi int)
	units int
}

// cpuBound is a block's step as its words read it: n rows of the residual
// x, their rotary tables cs and their descriptors rows, their history the
// layer's pages kv through the page table pages.
type cpuBound struct {
	li  int
	l   *layer
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
	// ent is the block's entries, entPages their table and entRows the
	// rows' entry descriptors (cpuEntBind).
	ent      *kvPages
	entPages []uint32
	entRows  []uint32
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
	io [cpuIO]unsafe.Pointer
	// req is the step's request words (cpuArgRequest), pos its live rows,
	// then a row's position (cpuArgPos), tok its tokens (cpuArgTokens).
	req, pos, tok []uint32
	// planes, where set, is the rotary planes a table's call reads in place
	// of the model's (the tower's grid planes, towertable.go).
	planes []float32
}

// cpuIO is the most inputs and outputs a graph apart from its residual
// binds: the sampler's candidates read three and write two.
const cpuIO = 6

// composedCPUDevice is the CPU as the composer forms for it.
func composedCPUDevice(workers int) composer.Device {
	return composer.Device{Kind: composer.DeviceCPU, Slots: max(workers, 1), Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
}

// composedCPU is this instance's CPU as the composer forms for it, its launch
// cost its probe row's.
func (s *Instance) composedCPU() composer.Device {
	workers := 1
	if s.jit != nil {
		workers = s.jit.Workers()
	}
	d := composedCPUDevice(workers)
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
func (s *Instance) cpuStep(li int, l *layer, cs []float32) (composer.Decline, error) {
	cb, why := s.cpuBlockOf(li)
	if why != composer.DeclineServes {
		return why, nil
	}
	if err := s.cpuOneTable(); err != nil {
		return composer.DeclineServes, err
	}
	// A block sharing another's history reads its source's pages.
	hl := s.c.KVSource(li)
	pg := &s.kv.layers[hl]
	pos := s.pos
	if pg.p > 0 {
		if why := s.cpuHistory(hl, pos); why != composer.DeclineServes {
			return why, nil
		}
	}
	if why := s.cpuEntReady(li, pos); why != composer.DeclineServes {
		return why, nil
	}
	b := &s.cpu.bound
	*b = cpuBound{li: li, l: l, own: s.cpuOwn(cb.f), cs: cs, x: s.resid(), stride: s.c.NEmbd, rows: s.cpu.rows[:], n: 1, kv: pg, pages: pg.table(max(s.nseq, 1)),
		recLi: s.cpuRecLi(li), recOf: s.Session, side: s.cpuSide(li)}
	if b.side != nil {
		b.side = b.side[cb.sideOff:]
	}
	s.cpu.tok1[0] = uint32(s.tok)
	s.cpu.pos1 = [2]uint32{1, uint32(pos)}
	b.tok, b.pos, b.csSWA = s.cpu.tok1[:], s.cpu.pos1[:], s.csSWA
	s.cpu.onePos[0] = pos
	s.cpuEntBind(b, li, s.cpu.entRow[:], nil, s.cpu.onePos[:])
	if why := s.cpuWeights(cb.f, li, l); why != composer.DeclineServes {
		return why, nil
	}
	s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos)), uint32(pos + 1), uint32(pos)}
	s.cpu.recRows = [3]uint32{1, 0, 0}
	s.cpu.call[0] = 1
	switch s.cpu.fault {
	case cpuFaultPage:
		b.pages = s.cpuFaultTable(pg, pos+1)
	case cpuFaultRow, cpuFaultWrite:
		s.cpuFaultRowWords(s.cpu.rows[:])
	}
	if err := s.cpuForm(b, cb.f, li); err != nil {
		return composer.DeclineServes, err
	}
	if pg.p > 0 && pos+1 > pg.written {
		pg.written = pos + 1
	}
	return composer.DeclineServes, nil
}

// cpuRows runs block li on the CPU the composer's way for the n rows of x,
// row i at position pos[i] of sequence seq[i], its rotary table the i-th of
// cs, or says why not: through the rows' form, a call of at most its
// capacity at a time, each row's history in its own sequence's pages.
func (s *Instance) cpuRows(li int, l *layer, x, cs, csSWA []float32, seq, pos []int) (composer.Decline, error) {
	if err := s.cpuRowTables(seq, pos); err != nil {
		return composer.DeclineServes, err
	}
	f, why := s.cpuRowsForm(li, l)
	if why != composer.DeclineServes {
		return why, nil
	}
	if why := s.cpuRowsReady(li, f, pos); why != composer.DeclineServes {
		return why, nil
	}
	side, _ := s.cpuSideRows(li)
	return composer.DeclineServes, s.cpuRowsRun(li, l, f, x, len(pos)*s.c.NEmbd, side, s.bcs, s.c.RopeW(), s.bcsSWA, s.c.NRotSWA, seq, pos)
}

// swaRows is the step's second rotary table from row a, nil where the model
// has no second rotary (cpuRowTables writes it into s.bcsSWA).
func (s *Instance) swaRows(a int) []float32 {
	if s.bcsSWA == nil {
		return nil
	}
	return s.bcsSWA[a*s.c.NRotSWA:]
}

// cpuRunRows runs non-causal block li on the CPU the composer's way over
// the n rows of x, one run attending both ways or, where wins is not nil,
// each window a run its rows attend inside; cs is the rows' rotary table,
// nrot wide, and swa the second rotary's, nswa wide, where the block's graph
// turns by one (nil where it does not). Its history is the segment's arena
// (ncArena), lent to the block for its run.
func (s *Instance) cpuRunRows(li int, l *layer, x []float32, n int, cs []float32, nrot int, swa []float32, nswa int, wins []nn.KeyRun) (composer.Decline, error) {
	f, why := s.cpuRowsForm(li, l)
	if why != composer.DeclineServes {
		return why, nil
	}
	if n > f.keys {
		return composer.DeclineSessions, nil
	}
	a := s.nc()
	a.run[0] = nn.KeyRun{Lo: 0, Hi: n}
	runs := a.run[:]
	if wins != nil {
		runs, s.cpu.runWindows = wins, true
	}
	if !a.causalFault {
		s.bidir = runs
	}
	seq, pos := s.cpuRowsOf(0, 0, n)
	err := s.cpuRowsRun(li, l, f, x, n*s.c.NEmbd, nil, cs, nrot, swa, nswa, seq, pos)
	s.bidir, s.cpu.runWindows = nil, false
	return composer.DeclineServes, err
}

// cpuSessions is cpuRows for a step across sessions (stepHost): row i's
// history is s.rowOwn[i]'s, its one sequence, and the first nlogit rows are
// each a session's last. Each session runs its own rows on its own block's
// form and buffers, its history bound as its form was composed for. Every
// session's pages are made ready before any row runs; the runs of a
// session's consecutive rows run one after another, the rows after the
// logit rows first, so a session's last row reads the keys its earlier
// rows wrote.
func (s *Instance) cpuSessions(li int, l *layer, x, cs, csSWA []float32, seq, pos []int, nlogit int) (composer.Decline, error) {
	if err := s.cpuRowTables(seq, pos); err != nil {
		return composer.DeclineServes, err
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
	why, err := s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, 0, n, false)
	if why != composer.DeclineServes || err != nil {
		return why, err
	}
	if why, err = s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, nlogit, n, true); err != nil {
		return why, err
	}
	return s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, 0, nlogit, true)
}

// cpuOwnerRuns makes ready (run unset) or runs block li over the rows
// [lo, hi) of a step across sessions, a run of one session's consecutive
// rows at a time, on that session's form.
func (s *Instance) cpuOwnerRuns(li int, l *layer, x, cs []float32, nrot int, seq, pos []int, lo, hi int, run bool) (composer.Decline, error) {
	d := s.c.NEmbd
	rside, pw := s.cpuSideRows(li)
	for a := lo; a < hi; {
		o, b := s.rowOwn[a], a+1
		for b < hi && s.rowOwn[b] == o {
			b++
		}
		if o.model != s.m {
			return composer.DeclineSessions, nil
		}
		// A row of another instance runs on that instance's form, composed
		// for its history; a session leasing s, on s's with its history bound.
		oi := s
		if a < len(s.rowInst) && s.rowInst[a] != nil {
			oi = s.rowInst[a]
		}
		f, why := oi.cpuRowsForm(li, l)
		if why != composer.DeclineServes {
			return why, nil
		}
		lead := oi.Session
		oi.Session = o
		if run {
			var side []float32
			if pw != 0 && rside != nil {
				side = rside[a*pw:]
			}
			if err := oi.cpuRowsRun(li, l, f, x[a*d:], len(pos)*d, side, cs[a*nrot:], nrot, s.swaRows(a), s.c.NRotSWA, seq[a:b], pos[a:b]); err != nil {
				oi.Session = lead
				return composer.DeclineServes, err
			}
		} else if why := oi.cpuRowsReady(li, f, pos[a:b]); why != composer.DeclineServes {
			oi.Session = lead
			return why, nil
		}
		oi.Session = lead
		a = b
	}
	return composer.DeclineServes, nil
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
func (s *Instance) cpuRecRows(li int, l *layer, x, cs []float32, seq, pos, order []int, base, p0 int) (composer.Decline, error) {
	if err := s.cpuRowTables(seq, pos); err != nil {
		return composer.DeclineServes, err
	}
	cb, why := s.cpuBlockOf(li)
	if why != composer.DeclineServes {
		return why, nil
	}
	if f := s.cpuRecWhole(li, l, cb, seq, pos, order, base); f != nil {
		why, err := s.cpuRecRun(li, l, f, x, cs, seq, pos, base, p0)
		if why == composer.DeclineServes && err == nil {
			s.cpu.recWhole++
		}
		return why, err
	}
	if len(pos) > 1 {
		s.cpu.recStepped++
	}
	if why := s.cpuWeights(cb.f, li, l); why != composer.DeclineServes {
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
			if why := s.cpuHistory(li, pos[r]); why != composer.DeclineServes {
				s.Session = lead
				return why, nil
			}
		}
		if s.cpu.fault == cpuFaultChunk && base > 0 && i == 0 {
			clear(o.rstate[li][slot*s.dg.stateStride():][:s.dg.stateStride()])
			clear(o.rconv[li][slot*s.dg.convStateLen():][:s.dg.convStateLen()])
		}
		*b = cpuBound{li: li, l: l, own: s.cpuOwn(cb.f), x: x[r*d:], stride: stride, rows: s.cpu.rows[:], n: 1, kv: pg,
			pages: pg.table(max(s.nseq, 1)), recLi: s.cpuRecLi(li), recOf: o}
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
			return composer.DeclineServes, err
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
	return composer.DeclineServes, nil
}

// cpuRecWhole is block li's rows' form where the step's rows run on it
// whole: several rows, all this instance's, in causal order, each sequence's
// rows in order of position, and no speculative snapshot to take after
// each row; nil where not.
func (s *Instance) cpuRecWhole(li int, l *layer, cb *cpuBlock, seq, pos, order []int, base int) *cpuBlockForm {
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
	if s.cpuWeights(cb.rows, li, l) != composer.DeclineServes {
		return nil
	}
	return cb.rows
}

// cpuRecRun runs linear block li's rows' form f over the step's rows, a
// call at most f's rows; a prompt's (base its first row in the prompt, p0
// its first position) end a call at each restore point inside them and seal
// it there.
func (s *Instance) cpuRecRun(li int, l *layer, f *cpuBlockForm, x, cs []float32, seq, pos []int, base, p0 int) (composer.Decline, error) {
	if why := s.cpuRowsReady(li, f, pos); why != composer.DeclineServes {
		return why, nil
	}
	n, d, nrot := len(pos), s.c.NEmbd, s.c.RopeW()
	rside, pw := s.cpuSideRows(li)
	if s.cpu.fault == cpuFaultChunk && base > 0 {
		slot := seq[0]
		clear(s.rstate[li][slot*s.dg.stateStride():][:s.dg.stateStride()])
		clear(s.rconv[li][slot*s.dg.convStateLen():][:s.dg.convStateLen()])
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
		if err := s.cpuRowsRun(li, l, f, x[a*d:], n*d, side, cs[a*nrot:], nrot, s.swaRows(a), s.c.NRotSWA, seq[a:b], pos[a:b]); err != nil {
			return composer.DeclineServes, err
		}
		if base >= 0 {
			s.recRowPoint(li, seq[a], p0+b)
		}
		a = b
	}
	return composer.DeclineServes, nil
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
	for r := range s.rstate {
		if s.rstate[r] != nil {
			return r
		}
	}
	return li
}

// cpuRowsForm is block li's rows' form, its weights the block's, or why
// not.
func (s *Instance) cpuRowsForm(li int, l *layer) (*cpuBlockForm, composer.Decline) {
	cb, why := s.cpuBlockOf(li)
	if why != composer.DeclineServes {
		return nil, why
	}
	if cb.rows == nil {
		return nil, cb.rowsWhy
	}
	if why := s.cpuWeights(cb.rows, li, l); why != composer.DeclineServes {
		return nil, why
	}
	return cb.rows, composer.DeclineServes
}

// cpuRowsReady makes the pages of the history the rows at pos write and
// read resident, in block li's arena, or says why f cannot run them: a
// history past f's key bound, a page outside the arena.
func (s *Instance) cpuRowsReady(li int, f *cpuBlockForm, pos []int) composer.Decline {
	hl := s.c.KVSource(li)
	pg := &s.kv.layers[hl]
	if pg.p == 0 {
		return composer.DeclineServes
	}
	// Every row's page is written before any is read: a step's rows stand
	// in any order, and a page holds every sequence's positions.
	start, end := pos[0], 0
	for _, p := range pos {
		s.cpuWritable(hl, p)
		start, end = min(start, pg.keyStart(p)), max(end, p+1)
	}
	if end > f.keys {
		return composer.DeclineSessions
	}
	if why := s.cpuReadable(hl, start, end-1); why != composer.DeclineServes {
		return why
	}
	return s.cpuEntReady(li, pos...)
}

// cpuFaultTable is the page table a step of the layer whose pages pg
// holds reads, its rows' history ending before end: under cpuFaultPage,
// once the layer's first page is full, a table whose first page is the
// page holding the history's last position -- its positions read the newest
// keys, another page's tokens at another rotation. Each layer is pointed into its own arena, so a model
// whose arenas differ (a sliding layer's beside a global one's) is reached.
func (s *Instance) cpuFaultTable(pg *kvPages, end int) []uint32 {
	nseq := max(s.nseq, 1)
	t := pg.table(nseq)
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
func (s *Instance) cpuRowsRun(li int, l *layer, f *cpuBlockForm, x []float32, stride int, side, cs []float32, nrot int, swa []float32, nswa int, seq, pos []int) error {
	runs := s.runsAt(li) && s.bidirIn(pos[0], len(pos))
	if runs && len(pos) > f.rows {
		keys := f.keysEnd
		if s.cpu.fault == cpuFaultKeysFirst {
			keys = 0
		}
		s.cpu.stashing = stashSave
		err := s.cpuRowsCalls(li, l, f, 0, keys, x, stride, side, cs, nrot, swa, nswa, seq, pos, true)
		s.cpu.stashing = stashLoad
		if err == nil {
			err = s.cpuRowsCalls(li, l, f, keys, len(f.codes), x, stride, side, cs, nrot, swa, nswa, seq, pos, true)
		}
		s.cpu.stashing = stashNone
		return err
	}
	return s.cpuRowsCalls(li, l, f, 0, len(f.codes), x, stride, side, cs, nrot, swa, nswa, seq, pos, runs)
}

// cpuStash saves (stashSave) or restores (stashLoad) call c's values form
// f's codes before keysEnd write and the rest read (cpuBlockForm.cross).
func (s *Instance) cpuStash(f *cpuBlockForm, own [][]uint32, c int) {
	at := 0
	for _, i := range f.cross {
		at += len(own[i]) * c
	}
	for _, i := range f.cross {
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
// (cpuBlockForm.split) holds the experts each call's rows pick, from its
// selection to the call's end (cpuExperts).
func (s *Instance) cpuRowsCalls(li int, l *layer, f *cpuBlockForm, lo, hi int, x []float32, stride int, side, cs []float32, nrot int, swa []float32, nswa int, seq, pos []int, runs bool) error {
	n := len(pos)
	pg := &s.kv.layers[s.c.KVSource(li)]
	d, ns := s.c.NEmbd, s.c.Streams()
	_, pw := s.cpuSideRows(li)
	so := s.cpuSideOff(li)
	slots := max(pg.arenaSlots(), 1)
	b := &s.cpu.bound
	end := 0
	for r0 := 0; r0 < n; r0 += f.rows {
		k := min(f.rows, n-r0)
		for i := range k {
			p := pos[r0+i]
			e, st := p+1, pg.keyStart(p)
			if runs {
				e = max(e, s.bidirEnd(p))
				if s.cpu.runWindows {
					st = max(st, s.bidirStart(p))
				}
			}
			if w := s.cpu.rowWins; w != nil {
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
		staged := f.unit > 1 && k%f.unit != 0
		if staged {
			if len(s.cpu.xpad) < f.rows*d*ns {
				s.cpu.xpad = make([]float32, f.rows*d*ns)
			}
			for j := range ns {
				copy(s.cpu.xpad[j*f.rows*d:], bx[j*stride:j*stride+k*d])
			}
			bx, bstride = s.cpu.xpad, f.rows*d
			if side != nil {
				if len(s.cpu.sidePad) < f.rows*pw {
					s.cpu.sidePad = make([]float32, f.rows*pw)
				}
				copy(s.cpu.sidePad, bside[:k*pw])
				bside = s.cpu.sidePad
			}
		}
		if bside != nil {
			bside = bside[so:]
		}
		*b = cpuBound{li: li, l: l, own: s.cpuOwn(f), cs: cs[r0*nrot:], x: bx, stride: bstride, rows: s.cpu.rowsTab, n: k,
			kv: pg, pages: s.cpuFaultTable(pg, end), side: bside}
		if len(s.cpu.tok) > r0 {
			b.tok = s.cpu.tok[r0:]
		}
		if swa != nil {
			b.csSWA = swa[r0*nswa:]
		}
		if len(s.cpu.posTab) < 1+f.rows {
			s.cpu.posTab = make([]uint32, 1+f.rows)
		}
		s.cpu.posTab[0] = uint32(k)
		for i := range f.rows {
			s.cpu.posTab[1+i] = uint32(pos[r0+min(i, k-1)])
		}
		b.pos = s.cpu.posTab
		if s.kv.entRate(li) > 0 {
			if len(s.cpu.entTab) < len(s.cpu.rowsTab) {
				s.cpu.entTab = make([]uint32, len(s.cpu.rowsTab))
			}
			s.cpuEntBind(b, li, s.cpu.entTab, seq[r0:r0+k], pos[r0:r0+k])
		}
		if f.rec {
			b.recLi, b.recOf, b.rec = s.cpuRecLi(li), s.Session, s.cpuRecDesc(seq[r0:r0+k])
			if n := len(s.rconv[b.recLi]); len(s.cpu.convOut) < n {
				s.cpu.convOut = make([]float32, n)
			}
		}
		s.cpu.call[0] = uint32(k)
		if s.cpu.stashing == stashLoad {
			s.cpuStash(f, b.own, r0/f.rows)
		}
		for i := lo; i < hi; i++ {
			if i == f.split && f.split > 0 {
				if err := s.cpuExperts(b, f, li, k); err != nil {
					s.expHold.release()
					return err
				}
			}
			if s.cpu.fault == cpuFaultWeight && slices.Contains(f.wreads, i) {
				// Each row's first pick weighted by zero where a kernel
				// reads the weights.
				w := f.selN / f.rows
				for r := range k {
					b.own[f.wsel][r*w] = 0
				}
			}
			s.cpuCall(b, &f.codes[i])
		}
		s.expHold.release()
		if f.rec {
			// The runs' windows over their slots.
			L := s.dg.convStateLen()
			for i, q := range seq[r0 : r0+k] {
				if !slices.Contains(seq[r0:r0+i], q) {
					copy(s.rconv[b.recLi][q*L:(q+1)*L], s.cpu.convOut[q*L:(q+1)*L])
				}
			}
		}
		if s.cpu.stashing == stashSave {
			s.cpuStash(f, b.own, r0/f.rows)
		}
		if staged {
			for j := range ns {
				copy(x[j*stride+r0*d:j*stride+(r0+k)*d], s.cpu.xpad[j*f.rows*d:j*f.rows*d+k*d])
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
func (s *Instance) cpuHistory(li, pos int) composer.Decline {
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
func (s *Instance) cpuReadable(li, start, pos int) composer.Decline {
	pg := &s.kv.layers[li]
	n, _ := pg.page(pos)
	s.kvEnsureWindow(li, start, pos-start)
	for p := start / pg.p; p <= n; p++ {
		if !pg.arenaPage(p) {
			return composer.DeclinePageOut
		}
	}
	return composer.DeclineServes
}

// cpuWeights says every weight f binds is one block l holds whole.
func (s *Instance) cpuWeights(f *cpuBlockForm, li int, l *layer) composer.Decline {
	for i := range f.codes {
		for _, a := range f.codes[i].args {
			switch a.src {
			case cpuArgWeight:
				if jlm.ExpertBank(a.role) {
					// A bank is read in the host's expert pages (cpuBanksPaged).
					if li >= len(s.cpu.banks) || s.cpu.banks[li][a.role-jlm.RoleExpGateBank][a.region] == nil {
						return composer.DeclineNoWeight
					}
					continue
				}
				w := s.cpuRoleWeight(li, l, a.role)
				if w == nil || w.packed == nil && len(w.data) == 0 {
					return composer.DeclineNoWeight
				}
				if p := w.packed; p != nil && (p.Row != 0 || p.Stride != 0 && p.Stride != w.rows || p.Sheet != nil) {
					return composer.DeclineCutWeight
				}
			case cpuArgVector:
				if len(s.cpuRoleVector(li, l, a.role)) == 0 {
					return composer.DeclineNoWeight
				}
			}
		}
	}
	return composer.DeclineServes
}

// cpuOwnOffHeap is the words past which an own buffer is anonymous memory
// rather than the heap's: an attention's scores are rows by heads by the
// context's keys, which the kernel commits as a step's rows reach them, and
// which on the heap would be the collector's to account and mark (RULE 2f).
const cpuOwnOffHeap = 1 << 14

// cpuOwn is form f's own buffers for this Instance.
func (s *Instance) cpuOwn(f *cpuBlockForm) [][]uint32 {
	if b, ok := s.cpu.own[f]; ok {
		return b
	}
	if s.cpu.own == nil {
		s.cpu.own = map[*cpuBlockForm][][]uint32{}
	}
	b := make([][]uint32, len(f.own))
	for i, n := range f.own {
		b[i] = s.ownWords(max(n, 1))
		copy(b[i], f.consts[i])
	}
	s.cpu.own[f] = b
	return b
}

// ownWords is n words of an own buffer, zeroed.
func (s *Instance) ownWords(n int) []uint32 {
	if n >= cpuOwnOffHeap {
		if m, mapped, err := offheap.MapOffHeap(4 * n); err == nil && mapped {
			s.cpu.mapped = append(s.cpu.mapped, m)
			return unsafe.Slice((*uint32)(unsafe.Pointer(&m[0])), n)
		}
	}
	return make([]uint32, n)
}

// unmapOwn returns the own buffers held off the heap.
func (s *Instance) unmapOwn() {
	for _, m := range s.cpu.mapped {
		offheap.UnmapOffHeap(m)
	}
	s.cpu.mapped, s.cpu.own = nil, nil
}

// cpuCall runs one kernel of a block: once, or over its calls on the pool.
func (s *Instance) cpuCall(b *cpuBound, c *cpuCode) {
	if s.prof {
		t0 := time.Now()
		s.cpu.t0, s.cpu.maxStart = t0, 0
		s.cpuLaunch(b, c)
		c.nanos += int64(time.Since(t0))
		c.runs++
		c.pmaxNs += s.cpu.maxStart
		return
	}
	s.cpuLaunch(b, c)
}

func (s *Instance) cpuLaunch(b *cpuBound, c *cpuCode) {
	n, live := c.n, 0
	if c.live.Live > 0 {
		live = c.live.LiveUnitsOf(b.n)
		n = live * c.inner
	}
	if !c.tiles {
		live = 0
	}
	s.cpu.part = s.Workers()
	if c.serial {
		s.cpu.part = 1
	}
	if c.share.Share > 0 {
		_, n = c.share.ShareRows(s.cpu.part)
	}
	if n <= 1 || c.serial || s.jit == nil || s.jit.Workers() <= 1 {
		a := &s.cpu.args[0]
		for i := range max(n, 1) {
			s.cpuWords(a, b, c, c.callAt(i, live))
			c.code.Call(a)
		}
		return
	}
	// A prefill's pool may be wider than the decode pool the words were
	// sized for at placement (the P and E cores'): a word block for each of
	// its workers.
	if w := s.jit.Workers(); len(s.cpu.args) < w {
		s.cpu.args = cpuArgBlocks(w)
	}
	r := &s.cpu.run
	r.s, r.b, r.c, r.live = s, b, c, live
	s.jit.ParallelWorkers(n, 1, r.fn)
}

// callAt is the call the i-th of a kernel's calls runs: in order, or for a
// kernel tiling a weight over live units of rows (tiles), a tile's every
// unit before the next tile's.
func (c *cpuCode) callAt(i, live int) int {
	if live <= 0 {
		return i
	}
	return i%live*c.inner + i/live
}

// cpuArgWords is one call's word block: cpu.Args' layout, which the kernel
// reads, held as words. A word is an address and its offset, which past a
// slice's last element is no pointer the collector accepts (a field of
// cpu.Args is), or a count; the addresses are of the instance's own buffers,
// which it holds.
type cpuArgWords [(unsafe.Sizeof(cpu.Args{}) + 7) / 8]uint64

// cpuArgBlocks is n word blocks seen as the cpu.Args a kernel call takes: the
// collector scans an object by the type it was allocated as.
func cpuArgBlocks(n int) []cpu.Args {
	w := make([]cpuArgWords, n)
	return unsafe.Slice((*cpu.Args)(unsafe.Pointer(&w[0])), n)
}

// cpuWords writes call at's words of kernel c into a.
func (s *Instance) cpuWords(a *cpu.Args, b *cpuBound, c *cpuCode, at int) {
	base := unsafe.Pointer(a)
	rows, _ := c.share.ShareRows(s.cpu.part)
	for i := range c.args {
		ar := &c.args[i]
		var p unsafe.Pointer
		switch ar.src {
		case cpuArgValue:
			*(*uint64)(unsafe.Add(base, ar.field)) = ar.val
			continue
		case cpuArgShare:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(min(rows, c.share.Share-at*rows))
			continue
		case cpuArgCoord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(c.coord(at, int(ar.coord), rows))
			continue
		case cpuArgWord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(b.own[ar.i][ar.off])
			continue
		case cpuArgX:
			p = unsafe.Pointer(&b.x[ar.stream*b.stride])
		case cpuArgIO:
			p = b.io[ar.i]
		case cpuArgSide:
			p = unsafe.Pointer(&b.side[0])
		case cpuArgOwn:
			p = unsafe.Pointer(&b.own[ar.i][0])
		case cpuArgWeight:
			if jlm.ExpertBank(ar.role) {
				p = s.cpu.banks[b.li][ar.role-jlm.RoleExpGateBank][ar.region]
				break
			}
			w := s.cpuRoleWeight(b.li, b.l, ar.role)
			if w.packed == nil {
				p = unsafe.Pointer(&w.data[0])
			} else {
				p = unsafe.Pointer(&[3][]byte{w.packed.QS, w.packed.D, w.packed.SC}[ar.region][0])
			}
		case cpuArgVector:
			p = unsafe.Pointer(&s.cpuRoleVector(b.li, b.l, ar.role)[0])
		case cpuArgRope:
			p = unsafe.Pointer(&b.cs[0])
			if ar.inst == 1 {
				p = unsafe.Pointer(&b.csSWA[0])
			}
		case cpuArgPlanes:
			if b.planes != nil {
				p = unsafe.Pointer(&b.planes[0])
				break
			}
			p = unsafe.Pointer(&s.cpu.tabPlanes[ar.inst][0])
		case cpuArgRopeRows:
			p = unsafe.Pointer(&b.ropeRows[0])
		case cpuArgRopeMarks:
			p = unsafe.Pointer(&b.ropeMarks[0])
		case cpuArgKeys:
			p = unsafe.Pointer(&b.kv.arena.k[0])
		case cpuArgEntKeys:
			p = unsafe.Pointer(&b.ent.arena.k[0])
		case cpuArgEntPages:
			p = unsafe.Pointer(&b.entPages[0])
		case cpuArgEntRows:
			p = unsafe.Pointer(&b.entRows[0])
		case cpuArgValues:
			p = unsafe.Pointer(&b.kv.arena.v[0])
		case cpuArgPages:
			p = unsafe.Pointer(&b.pages[0])
		case cpuArgRows:
			p = unsafe.Pointer(&b.rows[0])
		case cpuArgCall:
			p = unsafe.Pointer(&s.cpu.call[0])
		case cpuArgRecState:
			p = unsafe.Pointer(&b.recOf.rstate[b.recLi][0])
		case cpuArgRecConv:
			p = unsafe.Pointer(&b.recOf.rconv[b.recLi][0])
		case cpuArgRecConvOut:
			p = unsafe.Pointer(&b.recOf.rconv[b.recLi][0])
			if b.rec != nil {
				p = unsafe.Pointer(&s.cpu.convOut[0])
			}
		case cpuArgPoolPos:
			p = unsafe.Pointer(&s.cpu.pool.pos[0])
		case cpuArgRecRows:
			p = unsafe.Pointer(&s.cpu.recRows[0])
			if b.rec != nil {
				p = unsafe.Pointer(&b.rec[0])
			}
		case cpuArgZero:
			p = unsafe.Pointer(&s.cpu.zero[0])
		case cpuArgRequest:
			p = unsafe.Pointer(&b.req[0])
		case cpuArgTokens:
			p = unsafe.Pointer(&b.tok[0])
		case cpuArgPos:
			p = unsafe.Pointer(&b.pos[0])
		}
		off := ar.off
		if ar.moves {
			for ax := range c.calls {
				if ax < len(ar.per) && ar.per[ax] != 0 {
					off += c.coord(at, ax, rows) * ar.per[ax]
				}
			}
		}
		*(*uint64)(unsafe.Add(base, ar.field)) = uint64(uintptr(p)) + uint64(off*ar.elem)
	}
}

// coord is call at's coordinate along axis ax: a Share code's first row
// for calls of rows rows, else cpuCoord's.
func (c *cpuCode) coord(at, ax, rows int) int {
	if rows > 0 {
		return at * rows / max(c.share.ShareUnit, 1)
	}
	return cpuCoord(c.calls, at, ax)
}

// cpuCoord is call at's coordinate along axis ax of the calls' space calls,
// the last axis fastest.
func cpuCoord(calls []int, at, ax int) int {
	for a := len(calls) - 1; a > ax; a-- {
		at /= calls[a]
	}
	return at % calls[ax]
}

// region runs the calls [lo, hi) of the kernel the instance's run names, on
// worker w's words.
func (r *cpuRegion) region(w, lo, hi int) {
	a := &r.s.cpu.args[w]
	for i := lo; i < hi; i++ {
		r.s.cpuWords(a, r.b, r.c, r.c.callAt(i, r.live))
		if r.s.prof {
			r.s.cpuProfCall(r.c, a)
			continue
		}
		r.c.code.Call(a)
	}
}

func (s *Instance) cpuProfCall(c *cpuCode, a *cpu.Args) {
	t := time.Now()
	c.code.Call(a)
	off := int64(t.Sub(s.cpu.t0))
	atomic.AddInt64(&c.pcalls, 1)
	atomic.AddInt64(&c.pcallNs, int64(time.Since(t)))
	atomic.AddInt64(&c.pstartNs, off)
	for {
		m := atomic.LoadInt64(&s.cpu.maxStart)
		if off <= m || atomic.CompareAndSwapInt64(&s.cpu.maxStart, m, off) {
			return
		}
	}
}

// groupRegion runs the calls [lo, hi) of the phase's kernels on worker w's
// words.
func (r *cpuRegion) groupRegion(w, lo, hi int) {
	r.s.cpuGroupRun(&r.s.cpu.args[w], r.b, r.g, lo, hi)
}

// cpuPlace composes a new instance's blocks on the CPU: each block of the
// instance's range no device holds, once, at its placement (cpuCompose).
func (s *Instance) cpuPlace() {
	for i := range s.cpu.tabPlanes {
		s.cpu.tabPlanes[i] = s.m.cpuTablePlanes(i)
	}
	// A segment's blocks are past the text model's in the one block space.
	n := max(s.c.NLayer, s.hi)
	s.cpu.blocks = make([]*cpuBlock, n)
	s.cpu.composed = make([]int, n)
	w := 1
	if s.jit != nil {
		w = s.jit.Workers()
	}
	s.cpu.args = cpuArgBlocks(max(w, 1))
	s.cpu.ends = make([]int, 0, 16)
	s.cpu.run.fn = s.cpu.run.region
	s.cpu.run.gfn = s.cpu.run.groupRegion
	s.cpu.run.cfn = s.cpu.run.chainRegion
	s.cpu.zero = make([]uint32, 4)
	for li := s.lo; li < min(s.hi, n); li++ {
		if !s.devAt(li) {
			s.cpuCompose(li)
		}
	}
	if s.nc() != nil {
		return
	}
	s.cpuPlaceHead()
	s.cpuPlaceStreams()
	s.cpuPlacePool()
	s.cpuPlaceMTP()
	s.cpuPlaceEmbed()
	s.cpuPlaceTables()
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
	if cb.why != composer.DeclineServes {
		return
	}
	f, err := s.m.formCPUBlock(s, li, cb, 1)
	if err != nil {
		cb.why = s.cpuFormDecline(err)
		return
	}
	cb.f = f
	if cb.rows, err = s.m.formCPUBlock(s, li, cb, cpuRowsCap); err != nil {
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
func (s *Instance) cpuFormDecline(err error) composer.Decline {
	if s.cpu.formErr == nil {
		s.cpu.formErr = err
	}
	var d cpuDeclineError
	if errors.As(err, &d) {
		return d.why
	}
	return composer.DeclineNotFormed
}

// cpuBlockOf is block li's composition on the CPU, made at its placement.
func (s *Instance) cpuBlockOf(li int) (*cpuBlock, composer.Decline) {
	if li >= len(s.cpu.blocks) || s.cpu.blocks[li] == nil {
		return nil, composer.DeclineUnplaced
	}
	cb := s.cpu.blocks[li]
	return cb, cb.why
}

// CPUCompositions is how many times each block was composed for the CPU in
// this instance: once at its placement there, and once more for each
// relocation back to it.
func (s *Instance) CPUCompositions() []int { return s.cpu.composed }

// cpuWhy is what keeps block li off the composer's CPU path before it is
// asked: what its graph and history need that the CPU device does not hold.
func (s *Instance) cpuWhy(li int) composer.Decline {
	c := s.c
	g := s.m.Composing().BlockGraph(li)
	// A block sharing another's history reads its source's pages.
	pg := &s.kv.layers[c.KVSource(li)]
	if s.nc() != nil {
		return s.cpuTowerWhy(g)
	}
	switch {
	case !composer.HostCPULowers():
		return composer.DeclineISA
	case g == nil:
		return composer.DeclineNoGraph
	case len(g.Out) != c.Streams() || len(g.In) > len(g.Out)+1 || len(g.In) > len(g.Out) && s.cpuSide(li) == nil:
		return composer.DeclineFeeds
	case c.MoEAt(li) && !s.cpuBanksPaged(li):
		// The forms read a bank as one view of its experts in the host's
		// expert pages (jlm.File.BankView); a bank with none is not read.
		return composer.DeclineExperts
	case c.LayerKind(li).Recurrent() && (li >= len(s.rstate) || s.rstate[li] == nil || s.rconv[li] == nil):
		return composer.DeclineRecurrent
	case s.kvl.fmt != cpu.KVF16 && s.kvl.fmt != cpu.KVF32:
		// The CPU's profile keeps a history binary16 itself (composer.Halves),
		// or at f32 without that pass.
		return composer.DeclineKVFormat
	case pg.p > 0 && pg.arena == nil:
		return composer.DeclinePageOut
	}
	return composer.DeclineServes
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
	if s.c.NGramAt(li) && s.q4 != nil && s.q4.ng != nil {
		return s.q4.ng.emb
	}
	if s.c.PLEDim == 0 {
		return nil
	}
	return s.ple
}

// cpuSideRows is cpuSide over a call's rows, and its width a row.
func (s *Instance) cpuSideRows(li int) ([]float32, int) {
	if s.c.NGramAt(li) && s.q4 != nil && s.q4.ng != nil {
		return s.q4.ng.emb, s.c.NGramWidth()
	}
	return s.bple, s.c.pleWidth()
}

// cpuSideOff is block li's sideOff, 0 for a block not composed.
func (s *Instance) cpuSideOff(li int) int {
	if li < len(s.cpu.blocks) && s.cpu.blocks[li] != nil {
		return s.cpu.blocks[li].sideOff
	}
	return 0
}

// cpuTowerWhy is cpuWhy for a block of a non-causal segment (a tower's or an
// encoder's): its history is the block's run's own, written and read inside
// the one call over the segment's rows (lendNC), so no other path reads its
// format, and its page is the instance's arena.
func (s *Instance) cpuTowerWhy(g *graph.Graph) composer.Decline {
	switch {
	case !composer.HostCPULowers():
		return composer.DeclineISA
	case g == nil:
		return composer.DeclineNoGraph
	case len(g.Out) != 1 || len(g.In) != 1:
		return composer.DeclineFeeds
	case s.nc().arena == nil:
		return composer.DeclinePageOut
	}
	return composer.DeclineServes
}

// cpuDeclineError is a form the CPU device refuses by a named reason.
type cpuDeclineError struct {
	why composer.Decline
	err error
}

func (e cpuDeclineError) Error() string {
	return fmt.Sprintf("%s: %v", e.why.String(), e.err)
}

// formCPUBlock is block li's form on the CPU at rows rows through the
// model's keyed cache: its history the session's sequences' pages
// (cpuPages), each a history of the instance's context.
func (m *Model) formCPUBlock(s *Instance, li int, cb *cpuBlock, rows int) (*cpuBlockForm, error) {
	g := m.Composing().BlockGraph(li)
	cb.graph = g
	gr := *g
	gr.Block = 0
	cb.sideOff = sideAtBinding(&gr)
	formats := map[jlm.Role]jlm.Type{}
	f := m.Composing()
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if e, ok := f.Find(w.Role, int32(li), -1); ok {
				formats[w.Role] = e.Type
				// A table the block lays out as float32 at load is read so.
				if t := s.cpuRoleWeight(li, &m.layers[li], w.Role); w.Rows > 0 && t != nil && t.e == nil && t.typ == quant.F32 {
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
	if a := s.nc(); a != nil {
		page = a.page
	}
	req := composer.Request{Graph: &gr, Formats: formats, Strides: m.cpuStrides(li, g),
		Step:    composer.Step{Rows: rows, Page: page, Keys: max(pg.arenaSlots(), 1) * page, ActWindow: s.jit.ActWindow(), Seqs: max(s.nseq, 1)},
		History: composer.History{ValuesInKeys: pg.latent},
		Device:  s.composedCPU()}
	// A block turns by the rotary its rope ops state, of the model's
	// (composing.Rotaries): each its own table, bound by instance.
	req.Rotaries = composing{m.container, m}.Rotaries()
	return m.cpuFormOf(req, &gr, rows, false, false, s.kvl.fmt == cpu.KVF32)
}

// readsPlanes reports whether a form of g reads the rotary's planes: a
// pooled selection turning its blocks, or a rotary at positions of its own.
func readsPlanes(g *graph.Graph) bool {
	return slices.ContainsFunc(g.Ops, func(o graph.Op) bool {
		return o.Kind == graph.OpSelect && o.Attr.Pooled && o.Attr.NRot > 0 || o.Kind == graph.OpRope && o.Attr.Shift > 0
	})
}

// cpuFormOf is req's form on the CPU, mapped once for the model by its Key;
// whole is buildCPUForm's.
func (m *Model) cpuFormOf(req composer.Request, gr *graph.Graph, rows int, apart, whole, f32 bool) (*cpuBlockForm, error) {
	m.cpuComposes.Add(1)
	c := &m.cpuForms
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.reg == nil {
		c.reg = composer.Default()
		c.reg32 = c.reg.WithoutHalves(composer.DeviceCPU)
	}
	if c.m == nil {
		c.m, c.err = map[composer.Key]*cpuBlockForm{}, map[composer.Key]error{}
	}
	reg := c.reg
	if f32 {
		reg = c.reg32
	}
	if m.opt.cpuFloat {
		reg = reg.WithoutInt8(composer.DeviceCPU).WithoutSplits(composer.DeviceCPU)
	}
	key, codes, err := reg.Form(req)
	if err != nil {
		return nil, err
	}
	if fm, ok := c.m[key]; ok {
		return fm, nil
	}
	if err, ok := c.err[key]; ok {
		return nil, err
	}
	fm, err := buildCPUForm(gr, codes, rows, apart, whole)
	if err != nil {
		c.err[key] = err
		return nil, err
	}
	fm.keys = req.Step.Keys
	c.m[key] = fm
	return fm, nil
}

// CPUComposes is how many times the composer was asked for a CPU form, a
// form the model's cache held among them: what an instance's creation,
// relocation or tuning moves, and a prefill or a generated token must not.
func (m *Model) CPUComposes() int64 { return m.cpuComposes.Load() }

// stepsState reports whether gr reads or writes a recurrent state.
func stepsState(gr *graph.Graph) bool {
	for _, o := range gr.Ops {
		for _, r := range slices.Concat(o.Reads, o.Writes) {
			if r.State == graph.StateRecState || r.State == graph.StateRecConv {
				return true
			}
		}
	}
	return false
}

// buildCPUForm maps codes and plans each call word's source: the graph's
// input and output the residual, or with apart each an io place of its own.
// A whole form is run at its rows and no fewer (a convolutional block's,
// over its grid), so its kernels need not be held to a call's rows.
func buildCPUForm(gr *graph.Graph, codes []composer.Code, rows int, apart, whole bool) (*cpuBlockForm, error) {
	if apart && len(gr.In)+len(gr.Out) > cpuIO {
		return nil, fmt.Errorf("a graph of %d inputs and %d outputs, past the %d places the CPU's call site binds", len(gr.In), len(gr.Out), cpuIO)
	}
	f := &cpuBlockForm{consts: map[int][]uint32{}, rows: rows, sel: -1, wsel: -1, rec: stepsState(gr)}
	selV, wsV := selOf(gr), weightsOf(gr)
	wOwn := map[graph.ValueID]int{}
	var wrote, read []ownUse
	// A kernel's constant block is its own: two kernels' blocks may share a
	// name and hold different words.
	type ownKey struct {
		o    composer.Operand
		code int
	}
	own := map[ownKey]int{}
	ownOf := func(o composer.Operand, code, words int, consts []uint32) int {
		key := ownKey{o, -1}
		if consts != nil {
			key.code = code
		}
		if i, ok := own[key]; ok {
			f.own[i] = max(f.own[i], words)
			return i
		}
		f.own = append(f.own, words)
		i := len(f.own) - 1
		own[key] = i
		if consts != nil {
			f.consts[i] = consts
		}
		return i
	}
	for ci, c := range codes {
		if c.CPU == nil {
			f.close()
			return nil, fmt.Errorf("%s has no CPU code", c.Name)
		}
		cc := cpuCode{calls: c.Calls, n: 1, name: c.Name, share: composer.Code{Share: c.Share, ShareUnit: c.ShareUnit}}
		if c.Share > 0 && len(c.Calls) != 1 {
			f.close()
			return nil, fmt.Errorf("%s: a share of rows over calls %v", c.Name, c.Calls)
		}
		for _, n := range c.Calls {
			cc.n *= n
		}
		if rows > 1 && !c.Scan && !whole && (c.Live > 0 || len(c.Reach) > 0) {
			// A form of several rows runs the units of its calls' leading
			// axis the call's rows reach: a kernel not held to them would
			// read and write rows past the call's. A scan's calls are its
			// state's rows, each reading the call's rows its descriptor
			// names; a kernel that holds no rows of a step (a table over the
			// history) runs whole (composer.Code.LiveUnitsOf).
			if c.Live == 0 || len(c.Calls) == 0 || c.Calls[0] != c.LiveUnits {
				f.close()
				var binds []string
				for _, b := range c.Binds {
					n := fmt.Sprintf("%s/v%d/r%d", b.Operand.Name, b.Operand.Value, b.Operand.Role)
					if b.Writes {
						n += "(w)"
					}
					binds = append(binds, n)
				}
				return nil, fmt.Errorf("%s at %d rows: its calls %v hold no rows of a call (live %d of %d units; passes %v; binds %v)",
					c.Name, rows, c.Calls, c.Live, c.LiveUnits, c.Passes, binds)
			}
			cc.live, cc.inner = composer.Code{Live: c.Live, LiveUnits: c.LiveUnits}, cc.n/c.Calls[0]
			cc.tiles = c.TileMajor && cc.inner > 1
			f.unit = max(f.unit, c.Live/c.LiveUnits)
		}
		bytes := map[composer.Operand]int64{}
		reach := map[composer.Operand][]int{}
		for _, r := range c.Reach {
			reach[r.Operand] = r.Per
		}
		// rw is each operand's use: 1 read, 2 written.
		rw := map[composer.Operand]uint8{}
		for _, bd := range c.Binds {
			if o := bd.Operand; bd.Writes && o.Kind == composer.OperandTable && (o.Table == composer.TableKeys || o.Table == composer.TableValues) {
				f.keysEnd = ci + 1
			}
			if o := bd.Operand; o.Kind == composer.OperandValue {
				if k := streamIn(gr, o.Value); k >= 0 && bd.Read {
					cc.reads |= 1 << k
				}
				if k := slices.Index(gr.Out, o.Value); k >= 0 && bd.Writes {
					cc.writes |= 1 << k
				}
			}
			o := bd.Operand
			o.Copy = 0
			bytes[o] = max(bytes[o], bd.Bytes)
			if bd.Read {
				rw[o] |= 1
			}
			if bd.Writes {
				rw[o] |= 2
			}
			if bd.Writes && bd.Operand.Kind == composer.OperandScratch && !bd.Own && !callsOwnScratch(c, bd.Operand, bd.Bytes) {
				cc.serial = true
			}
			if bd.Writes {
				n := o.Name
				if o.Kind == composer.OperandValue && int(o.Value) < len(gr.Values) {
					n = gr.Values[o.Value].Name
				}
				cc.desc += " " + n
			}
		}
		for _, a := range c.Args {
			// A pair's second head's operand is the first's storage at its
			// offset.
			o := a.Operand
			o.Copy = 0
			ar := cpuArg{field: uintptr(a.Field), off: a.Offset, elem: 4}
			if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(cpu.Args{}) {
				f.close()
				return nil, fmt.Errorf("%s: a call word at %d past the argument block", c.Name, a.Field)
			}
			if a.Elem > 0 {
				ar.elem = a.Elem
			}
			if len(a.Per) > len(ar.per) || len(a.Per) > len(c.Calls) {
				f.close()
				return nil, fmt.Errorf("%s: an address moving along %d axes of a calls' space of %d", c.Name, len(a.Per), len(c.Calls))
			}
			for ax, n := range a.Per {
				ar.per[ax] = n
				ar.moves = ar.moves || n != 0
			}
			switch {
			case a.Coord > 0:
				ar.src, ar.coord = cpuArgCoord, uint8(a.Coord-1)
			case a.Share:
				ar.src = cpuArgShare
			case a.Length:
				ar.src, ar.val = cpuArgValue, uint64(a.Extent)
			case a.Value:
				f.close()
				return nil, fmt.Errorf("%s: a count the caller computes", c.Name)
			case a.Moving:
				f.close()
				return nil, fmt.Errorf("%s: an offset of %v the caller computes", c.Name, o)
			case a.Words != nil:
				ar.src, ar.i = cpuArgOwn, ownOf(o, ci, len(a.Words), a.Words)
			case apart && o.Kind == composer.OperandValue && slices.Contains(gr.In, o.Value):
				ar.src, ar.i = cpuArgIO, slices.Index(gr.In, o.Value)
			case apart && o.Kind == composer.OperandValue && slices.Contains(gr.Out, o.Value):
				ar.src, ar.i = cpuArgIO, len(gr.In)+slices.Index(gr.Out, o.Value)
			case o.Kind == composer.OperandValue && streamIn(gr, o.Value) >= 0:
				ar.src, ar.stream = cpuArgX, streamIn(gr, o.Value)
			case o.Kind == composer.OperandValue && slices.Contains(gr.Out, o.Value):
				ar.src, ar.stream = cpuArgX, slices.Index(gr.Out, o.Value)
			case o.Kind == composer.OperandValue && len(gr.In) == len(gr.Out)+1 && o.Value == gr.In[len(gr.Out)]:
				ar.src = cpuArgSide
			case o.Kind == composer.OperandValue:
				ar.src, ar.i = cpuArgOwn, ownOf(o, ci, max(gr.Values[o.Value].Width, int(bytes[o]/4))+a.Offset, nil)
				switch o.Value {
				case selV:
					f.sel = ar.i
				default:
					if slices.Contains(wsV, o.Value) {
						wOwn[o.Value] = ar.i
					}
				}
			case o.Kind == composer.OperandScratch:
				ar.src, ar.i = cpuArgOwn, ownOf(o, ci, int(bytes[o]/4)+a.Offset+1, nil)
			case o.Kind == composer.OperandWeight && o.Region >= 0 && o.Region < 3:
				ar.src, ar.role, ar.region = cpuArgWeight, o.Role, uint8(o.Region)
			case o.Kind == composer.OperandVector:
				ar.src, ar.role = cpuArgVector, o.Role
			case o.Kind == composer.OperandTable:
				switch t := o.Table; {
				case t == composer.TableAngles && o.Instance < 2:
					ar.src, ar.inst = cpuArgRope, o.Instance
				case t == composer.TablePlanes && o.Instance < 2:
					ar.src, ar.inst = cpuArgPlanes, o.Instance
				case t == composer.TableRopeRows:
					ar.src = cpuArgRopeRows
				case t == composer.TableRopeMarks:
					ar.src = cpuArgRopeMarks
				case t == composer.TableKeys:
					ar.src = cpuArgKeys
				case t == composer.TableEntKeys:
					ar.src = cpuArgEntKeys
				case t == composer.TableEntPages:
					ar.src = cpuArgEntPages
				case t == composer.TableEntRows:
					ar.src = cpuArgEntRows
				case t == composer.TableValues:
					ar.src = cpuArgValues
				case t == composer.TablePages:
					ar.src = cpuArgPages
				case t == composer.TableRows, t == composer.TableWindows:
					// A row's descriptor is its own layer's: its range the
					// layer's window where it has one (kvPages.keyStart).
					ar.src = cpuArgRows
				case t == composer.TableCall:
					ar.src = cpuArgCall
				case t == composer.TableRecState, t == composer.TableRecStateOut:
					ar.src = cpuArgRecState
				case t == composer.TableRecConv:
					ar.src = cpuArgRecConv
				case t == composer.TableRecConvOut:
					ar.src = cpuArgRecConvOut
				case t == composer.TableRecRows:
					ar.src = cpuArgRecRows
				case t == composer.TableQOff:
					ar.src = cpuArgZero
				case t == composer.TableRequest:
					ar.src = cpuArgRequest
				case t == composer.TableTokens:
					ar.src = cpuArgTokens
				case t == composer.TablePos:
					// A pooled readout reads its live count (cpuPoolState), any
					// other graph its rows' positions (cpuBound.pos).
					ar.src = cpuArgPos
					if slices.ContainsFunc(gr.Ops, func(o graph.Op) bool { return o.Kind == graph.OpPool }) {
						ar.src = cpuArgPoolPos
					}
				default:
					f.close()
					return nil, cpuDeclineError{composer.DeclineTable, fmt.Errorf("%s reads the table %s", c.Name, o.Name)}
				}
			default:
				f.close()
				return nil, fmt.Errorf("%s binds %+v, which the CPU's call site does not provide", c.Name, o)
			}
			if ar.src == cpuArgOwn && a.Words == nil {
				if rw[o]&2 != 0 {
					wrote = append(wrote, ownUse{ci, ar.i})
				}
				if rw[o] != 2 {
					read = append(read, ownUse{ci, ar.i})
				}
			}
			if a.Word {
				if ar.src != cpuArgOwn {
					f.close()
					return nil, fmt.Errorf("%s passes a word of %+v by value, which the CPU's call site holds no words of", c.Name, o)
				}
				ar.src = cpuArgWord
			}
			cc.args = append(cc.args, ar)
			if ar.src != cpuArgValue && ar.src != cpuArgCoord && ar.src != cpuArgShare {
				buf := cpuBuf{src: ar.src, region: ar.region, role: ar.role, i: ar.i, stream: ar.stream}
				if buf.src == cpuArgWord {
					buf.src = cpuArgOwn
				}
				use := rw[o]
				if use == 0 || a.Word {
					// Unbound, or a word read when its call is set up: a
					// read, and, unbound, a write too.
					use |= 1
					if rw[o] == 0 {
						use |= 2
					}
				}
				if use&1 != 0 {
					cc.rd = append(cc.rd, buf)
				}
				if use&2 != 0 {
					cc.wr = append(cc.wr, buf)
				}
				cc.slice(buf, reach[o])
			}
		}
		code, err := cpu.MapNamed(c.CPU, c.Name)
		if err != nil {
			f.close()
			return nil, fmt.Errorf("mapping %s: %w", c.Name, err)
		}
		cc.code = code
		f.codes = append(f.codes, cc)
	}
	// The weights as the first of their values a kernel reads.
	for _, v := range wsV {
		own, ok := wOwn[v]
		for _, r := range read {
			if ok && r.own == own && !slices.Contains(f.wreads, r.code) {
				f.wsel, f.wreads = own, append(f.wreads, r.code)
			}
		}
		if f.wsel >= 0 {
			break
		}
	}
	for _, cc := range f.codes[:f.keysEnd] {
		f.keysX = f.keysX || cc.writes != 0
	}
	for _, w := range wrote {
		if w.code >= f.keysEnd || slices.Contains(f.cross, w.own) {
			continue
		}
		for _, r := range read {
			if r.own == w.own && r.code >= f.keysEnd {
				f.cross = append(f.cross, w.own)
				break
			}
		}
	}
	if !apart && !f.inPlace() {
		f.close()
		return nil, fmt.Errorf("the block reads its residual after writing its output over it")
	}
	if err := f.bankHalves(gr, rows); err != nil {
		f.close()
		return nil, err
	}
	if rows == 1 {
		f.chainPhases()
		f.joinPhases()
	}
	f.shareOwn(wrote, read)
	return f, nil
}

// shareOwn lays the form's own buffers out on fewer, as the card's forms
// share theirs (the tier's partForm.share): a buffer whose first code writes
// it without reading it, live from that code's pool phase to the last
// phase that binds it, takes one whose last phase ran before its first. The
// phases run in order -- a joined or chained phase's kernels run together,
// so its buffers are live across all of it (joinPhases, chainPhases, which
// ran over the buffers unshared) -- so the earlier holder is done with it.
// A constant, a buffer first read (it
// carries what an earlier call left), the selection and the picks' weights
// (the host reads and writes them between the halves) and what a
// bidirectional run holds between its passes keep their own. Every own
// index the form holds is renumbered.
func (f *cpuBlockForm) shareOwn(wrote, read []ownUse) {
	n := len(f.own)
	// Each code's phase: a joined or chained run's kernels share theirs.
	phase := make([]int, len(f.codes))
	for ci, p := 0, 0; ci < len(f.codes); p++ {
		end := ci + 1 + max(f.codes[ci].join, f.codes[ci].chain)
		for ; ci < end && ci < len(f.codes); ci++ {
			phase[ci] = p
		}
	}
	first, last := make([]int, n), make([]int, n)
	for i := range first {
		first[i] = -1
	}
	for ci := range f.codes {
		for _, ar := range f.codes[ci].args {
			if ar.src != cpuArgOwn && ar.src != cpuArgWord {
				continue
			}
			if first[ar.i] < 0 {
				first[ar.i] = ci
			}
			last[ar.i] = ci
		}
	}
	fresh := make([]bool, n)
	for _, w := range wrote {
		if w.code == first[w.own] {
			fresh[w.own] = true
		}
	}
	for _, r := range read {
		if r.code == first[r.own] {
			fresh[r.own] = false
		}
	}
	pinned := func(i int) bool {
		_, k := f.consts[i]
		return k || !fresh[i] || first[i] < 0 || i == f.sel || i == f.wsel || slices.Contains(f.cross, i)
	}
	slot := make([]int, n)
	var phys []int
	var free []int // each buffer's last code, len(f.codes) where it is never given up
	var order []int
	for i := range n {
		if pinned(i) {
			slot[i] = len(phys)
			phys, free = append(phys, f.own[i]), append(free, len(f.codes))
			continue
		}
		order = append(order, i)
	}
	slices.SortStableFunc(order, func(a, b int) int { return first[a] - first[b] })
	for _, i := range order {
		need, at := f.own[i], -1
		for p, end := range free {
			if end >= phase[first[i]] {
				continue
			}
			// The narrowest that holds it, else the widest, grown.
			switch {
			case at < 0:
				at = p
			case phys[p] >= need && (phys[at] < need || phys[p] < phys[at]):
				at = p
			case phys[at] < need && phys[p] > phys[at]:
				at = p
			}
		}
		if at < 0 {
			at = len(phys)
			phys, free = append(phys, 0), append(free, 0)
		}
		phys[at] = max(phys[at], need)
		free[at] = phase[last[i]]
		slot[i] = at
	}
	consts := map[int][]uint32{}
	for i, c := range f.consts {
		consts[slot[i]] = c
	}
	f.own, f.consts = phys, consts
	re := func(b *cpuBuf) {
		if b.src == cpuArgOwn {
			b.i = slot[b.i]
		}
	}
	for ci := range f.codes {
		c := &f.codes[ci]
		for k := range c.args {
			if c.args[k].src == cpuArgOwn || c.args[k].src == cpuArgWord {
				c.args[k].i = slot[c.args[k].i]
			}
		}
		for k := range c.rd {
			re(&c.rd[k])
		}
		for k := range c.wr {
			re(&c.wr[k])
		}
		for k := range c.sliced {
			re(&c.sliced[k].buf)
		}
	}
	for k, i := range f.cross {
		f.cross[k] = slot[i]
	}
	if f.sel >= 0 {
		f.sel = slot[f.sel]
	}
	if f.wsel >= 0 {
		f.wsel = slot[f.wsel]
	}
}

// callsOwnScratch reports whether code c's calls each write a slice of
// scratch o no other call writes: every address of o the code is passed
// moves along every axis of its calls, each axis's step past the whole of
// the axes inside it, as an operand laid out by its primitive's leading axes
// is, and the code reaching no more of o than its calls' slices; an axis of
// one call moves nothing. A scratch some call's address does not move along
// is every call's.
func callsOwnScratch(c composer.Code, o composer.Operand, bytes int64) bool {
	if len(c.Calls) == 0 {
		return false
	}
	o.Copy = 0
	seen := false
	for _, a := range c.Args {
		x := a.Operand
		x.Copy = 0
		if x != o {
			continue
		}
		if a.Word || a.Coord > 0 || a.Length || a.Value || a.Moving || len(a.Per) != len(c.Calls) {
			return false
		}
		span := 0
		for ax := range c.Calls {
			if c.Calls[ax] == 1 {
				continue
			}
			inner := 1
			if ax+1 < len(c.Calls) {
				inner = a.Per[ax+1] * c.Calls[ax+1]
			}
			if a.Per[ax] <= 0 || a.Per[ax] < inner {
				return false
			}
			span = max(span, a.Per[ax]*c.Calls[ax])
		}
		elem := int64(4)
		if a.Elem > 0 {
			elem = int64(a.Elem)
		}
		if bytes <= 0 || bytes > int64(span)*elem {
			return false
		}
		seen = true
	}
	return seen
}

// joinPhases runs each run of consecutive kernels that share no buffer one
// of them writes in one pool phase: their calls one space, claimed as one
// kernel's. A run does not cross the selection's split.
func (f *cpuBlockForm) joinPhases() {
	for g := 0; g < len(f.codes); {
		if f.codes[g].chain > 0 {
			g += f.codes[g].chain + 1
			continue
		}
		j := g + 1
		if f.codes[g].joinable() {
			for j < len(f.codes) && j != f.split && f.codes[j].joinable() && f.codes[j].chain == 0 && apart(f.codes[g:j], &f.codes[j]) {
				j++
			}
		}
		f.codes[g].join = j - g - 1
		g = j
	}
}

// slice records the rows of buffer b a call reaches alone along the one
// axis of its calls past a single call (composer.Code.Reach), -1 for none.
func (c *cpuCode) slice(b cpuBuf, reach []int) {
	per := -1
	if ax := oneAxis(c.calls); ax >= 0 && ax < len(reach) && reach[ax] > 0 {
		per = reach[ax]
	}
	for i := range c.sliced {
		if c.sliced[i].buf == b {
			if c.sliced[i].per != per {
				c.sliced[i].per = -1
			}
			return
		}
	}
	c.sliced = append(c.sliced, cpuSlice{b, per})
}

// oneAxis is the one axis of calls past a single call, -1 for none or more.
func oneAxis(calls []int) int {
	ax := -1
	for i, n := range calls {
		if n == 1 {
			continue
		}
		if ax >= 0 {
			return -1
		}
		ax = i
	}
	return ax
}

// chainPhases runs each run of kernels that chained admits, one reading
// what another writes, in one pool phase: unit u runs its calls of each
// kernel in turn. A run does not cross the selection's split.
func (f *cpuBlockForm) chainPhases() {
	for g := 0; g < len(f.codes); {
		j := g + 1
		for j < len(f.codes) && j != f.split && f.codes[j].joinable() && chained(f.codes[g:j+1]) {
			j++
		}
		dep := false
		for k := g + 1; k < j; k++ {
			dep = dep || !apart(f.codes[g:k], &f.codes[k])
		}
		if dep {
			f.codes[g].chain, f.codes[g].units = j-g-1, chainUnits(f.codes[g:j])
		}
		g = j
	}
}

// chainUnits is the units of a phase of kernels cs: the fewest calls of
// one, every kernel's calls along one axis and a whole number a unit; 0
// where they are not.
func chainUnits(cs []cpuCode) int {
	u := cs[0].n
	for i := range cs {
		u = min(u, cs[i].n)
	}
	ax := oneAxis(cs[0].calls)
	for i := range cs {
		if !cs[i].joinable() || cs[i].share.Share > 0 || ax < 0 || oneAxis(cs[i].calls) != ax || len(cs[i].calls) != len(cs[0].calls) || cs[i].n%u != 0 {
			return 0
		}
	}
	return u
}

// chained says kernels cs may run in one phase unit by unit: every buffer
// one writes and another reads or writes, every call of both reaches only
// its own slice of, a unit's calls' slices one span in each.
func chained(cs []cpuCode) bool {
	u := chainUnits(cs)
	if u <= 1 {
		return false
	}
	span := func(c *cpuCode, b cpuBuf) int {
		for _, s := range c.sliced {
			if s.buf == b && s.per > 0 {
				return s.per * (c.n / u)
			}
		}
		return -1
	}
	for i := range cs {
		for _, b := range cs[i].wr {
			for j := range cs {
				if j == i || !slices.Contains(cs[j].rd, b) && !slices.Contains(cs[j].wr, b) {
					continue
				}
				if s := span(&cs[i], b); s < 0 || s != span(&cs[j], b) {
					return false
				}
			}
		}
	}
	return true
}

// joinable says a kernel's calls may share a phase with another's: calls
// on the pool, each free of the others and of the step's rows. A kernel of
// one call runs on the caller, in no phase.
func (c *cpuCode) joinable() bool {
	return c.live.Live == 0 && !c.tiles && c.n > 1 && !c.serial
}

// apart says kernel c reads nothing the kernels gs write and writes nothing
// they read or write.
func apart(gs []cpuCode, c *cpuCode) bool {
	for i := range gs {
		g := &gs[i]
		for _, w := range g.wr {
			if slices.Contains(c.rd, w) || slices.Contains(c.wr, w) {
				return false
			}
		}
		for _, w := range c.wr {
			if slices.Contains(g.rd, w) {
				return false
			}
		}
	}
	return true
}

// selOf is the selection gr's banks are read at, -1 for a graph that reads
// none.
func selOf(gr *graph.Graph) graph.ValueID {
	for _, o := range gr.Ops {
		if o.Kind == graph.OpExpertMatVec && len(o.In) == 2 && len(o.Weight) > 0 && jlm.ExpertBank(o.Weight[0].Role) {
			return o.In[1]
		}
	}
	return -1
}

// weightsOf is the picks' weights gr's banks' outputs are multiplied by,
// then the values of their width each is an elementwise map of, back to the
// first that is not: nil for a graph that weights none.
func weightsOf(gr *graph.Graph) []graph.ValueID {
	bank := map[graph.ValueID]bool{}
	made := map[graph.ValueID]int{}
	w := graph.ValueID(-1)
	for i, o := range gr.Ops {
		for _, v := range o.Out {
			made[v] = i
		}
		if o.Kind == graph.OpExpertMatVec && len(o.In) == 2 && len(o.Weight) > 0 && jlm.ExpertBank(o.Weight[0].Role) {
			bank[o.Out[0]] = true
		}
		if w >= 0 || o.Kind != graph.OpMap || o.Attr.Fn != graph.MapMul || len(o.In) != 2 {
			continue
		}
		switch {
		case bank[o.In[0]]:
			w = o.In[1]
		case bank[o.In[1]]:
			w = o.In[0]
		}
	}
	var ws []graph.ValueID
	for w >= 0 && len(ws) < len(gr.Ops) {
		ws = append(ws, w)
		i, ok := made[w]
		if o := gr.Ops[i]; !ok || o.Kind != graph.OpMap || len(o.In) == 0 || gr.Values[o.In[0]].Width != gr.Values[w].Width {
			break
		}
		w = gr.Ops[i].In[0]
	}
	return ws
}

// bankHalves finds where a form over banks is cut (cpuBlockForm.split): at
// its first code that binds a bank, with the selection written by a code
// before it.
func (f *cpuBlockForm) bankHalves(gr *graph.Graph, rows int) error {
	sel := selOf(gr)
	if sel < 0 {
		return nil
	}
	f.split = -1
	for ci, c := range f.codes {
		for _, a := range c.args {
			if a.src == cpuArgWeight && jlm.ExpertBank(a.role) && f.split < 0 {
				f.split = ci
			}
		}
	}
	if f.sel < 0 || f.split <= 0 {
		return fmt.Errorf("a form over banks with no selection of its own (%d) or no code before its banks (%d)", f.sel, f.split)
	}
	f.selN = rows * gr.Values[sel].Width
	return nil
}

// cpuBanksPaged says block li's routed banks are each one view of its
// experts in the host's expert pages (jlm.File.BankView).
func (s *Instance) cpuBanksPaged(li int) bool {
	l := &s.m.layers[li]
	if s.m.container == nil || l.legacyExperts {
		return false
	}
	if len(s.cpu.banks) < len(s.m.layers) {
		s.cpu.banks = make([][3][3]unsafe.Pointer, len(s.m.layers))
	}
	for i, t := range []*tensor{&l.gate, &l.up, &l.down} {
		if t.e == nil {
			continue
		}
		planes, _, ok := s.m.container.BankView(t.e)
		if !ok {
			return false
		}
		s.cpu.banks[li][i] = planes
	}
	return l.up.e != nil && l.down.e != nil
}

// cpuStrides is each bank of block li's graph in the host's expert pages and
// the words from one expert's sheet to the next's there.
func (m *Model) cpuStrides(li int, g *graph.Graph) map[jlm.Role]int {
	var st map[jlm.Role]int
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			t := m.bankTensor(li, w.Role)
			if t == nil || t.e == nil {
				continue
			}
			if _, stride, ok := m.container.BankView(t.e); ok {
				if st == nil {
					st = map[jlm.Role]int{}
				}
				st[w.Role] = int(stride / 4)
			}
		}
	}
	return st
}

// bankTensor is block li's routed bank of role r, nil for any other role.
func (m *Model) bankTensor(li int, r jlm.Role) *tensor {
	if li >= len(m.layers) || !jlm.ExpertBank(r) {
		return nil
	}
	l := &m.layers[li]
	return [...]*tensor{&l.gate, &l.up, &l.down}[r-jlm.RoleExpGateBank]
}

// cpuForm runs form f of block li over the step b: its codes in order, and
// for a form over banks (cpuBlockForm.split) the selection's experts made
// resident and held between its halves -- the pager's own fault path
// (ensureExperts), each read from the file where the budget evicted it --
// and counted as the host's own mixture counts them.
func (s *Instance) cpuForm(b *cpuBound, f *cpuBlockForm, li int) error {
	if f.split <= 0 {
		s.cpuCalls(b, f.codes)
		return nil
	}
	s.cpuCalls(b, f.codes[:f.split])
	err := s.cpuExperts(b, f, li, 1)
	defer s.expHold.release()
	if err != nil {
		return err
	}
	if !s.rowsOfChunk {
		s.hot.add(li, s.cpu.sel)
	}
	s.cpuCalls(b, f.codes[f.split:])
	return nil
}

// cpuExperts makes the experts the n rows of a call of form f select
// (cpuBlockForm.sel, a row's picks after another's) resident and held until
// s.expHold's release, each once however many rows pick it -- the pager's
// own fault path (ensureExperts), each read from the file where the budget
// evicted it -- and counts them resident as main's mixture does. s.cpu.sel
// is the experts held, in the order the rows first pick them.
func (s *Instance) cpuExperts(b *cpuBound, f *cpuBlockForm, li, n int) error {
	k := f.selN / max(f.rows, 1)
	picks := b.own[f.sel][:n*k]
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
	if err := s.m.ensureExperts(li, s.cpu.sel, &s.expHold); err != nil {
		return err
	}
	s.res.touch(li, ne, s.cpu.sel)
	return nil
}

// cpuCalls runs kernels cs in order, each with those its phase joins
// (cpuBlockForm.joinPhases).
func (s *Instance) cpuCalls(b *cpuBound, cs []cpuCode) {
	for i := 0; i < len(cs); i++ {
		c := &cs[i]
		if c.chain > 0 {
			g := cs[i : i+c.chain+1]
			i += c.chain
			if !s.prof {
				s.cpuChain(b, g, c.units)
				continue
			}
			t0 := time.Now()
			s.cpuChain(b, g, c.units)
			c.nanos += int64(time.Since(t0))
			c.runs++
			continue
		}
		if c.join == 0 {
			s.cpuCall(b, c)
			continue
		}
		g := cs[i : i+c.join+1]
		i += c.join
		if !s.prof {
			s.cpuGroup(b, g)
			continue
		}
		t0 := time.Now()
		s.cpu.t0, s.cpu.maxStart = t0, 0
		s.cpuGroup(b, g)
		d := int64(time.Since(t0))
		g[0].pmaxNs += s.cpu.maxStart
		ends := s.cpu.ends
		for j := range g {
			n := ends[j]
			if j > 0 {
				n -= ends[j-1]
			}
			g[j].nanos += d * int64(n) / int64(ends[len(ends)-1])
			g[j].runs++
		}
	}
}

// cpuGroup runs the kernels g as one phase over the pool: their calls one
// space, in order of the kernels.
func (s *Instance) cpuGroup(b *cpuBound, g []cpuCode) {
	s.cpu.part = s.Workers()
	s.cpu.ends = s.cpu.ends[:0]
	n := 0
	for j := range g {
		if g[j].share.Share > 0 {
			_, k := g[j].share.ShareRows(s.cpu.part)
			n += k
		} else {
			n += max(g[j].n, 1)
		}
		s.cpu.ends = append(s.cpu.ends, n)
	}
	if s.jit == nil || s.jit.Workers() <= 1 {
		s.cpuGroupRun(&s.cpu.args[0], b, g, 0, n)
		return
	}
	if w := s.jit.Workers(); len(s.cpu.args) < w {
		s.cpu.args = cpuArgBlocks(w)
	}
	r := &s.cpu.run
	r.s, r.b, r.g = s, b, g
	s.jit.ParallelWorkers(n, 1, r.gfn)
}

// cpuChain runs the kernels g as one phase over the pool, units units:
// a unit its calls of each kernel in turn (chainPhases).
func (s *Instance) cpuChain(b *cpuBound, g []cpuCode, units int) {
	s.cpu.part = s.Workers()
	if s.jit == nil || s.jit.Workers() <= 1 {
		s.cpuChainRun(&s.cpu.args[0], b, g, units, 0, units)
		return
	}
	if w := s.jit.Workers(); len(s.cpu.args) < w {
		s.cpu.args = cpuArgBlocks(w)
	}
	r := &s.cpu.run
	r.s, r.b, r.g, r.units = s, b, g, units
	s.jit.ParallelWorkers(units, 1, r.cfn)
}

// cpuChainRun runs the units [lo, hi) of the kernels g's phase on a.
func (s *Instance) cpuChainRun(a *cpu.Args, b *cpuBound, g []cpuCode, units, lo, hi int) {
	for u := lo; u < hi; u++ {
		for k := range g {
			q := g[k].n / units
			for at := u * q; at < (u+1)*q; at++ {
				s.cpuWords(a, b, &g[k], at)
				g[k].code.Call(a)
			}
		}
	}
}

// chainRegion runs the units [lo, hi) of the phase's kernels on worker w's
// words.
func (r *cpuRegion) chainRegion(w, lo, hi int) {
	r.s.cpuChainRun(&r.s.cpu.args[w], r.b, r.g, r.units, lo, hi)
}

// cpuGroupRun runs the calls [lo, hi) of the kernels g's one space on a.
func (s *Instance) cpuGroupRun(a *cpu.Args, b *cpuBound, g []cpuCode, lo, hi int) {
	ends := s.cpu.ends
	k := 0
	for i := lo; i < hi; i++ {
		for i >= ends[k] {
			k++
		}
		at := i
		if k > 0 {
			at -= ends[k-1]
		}
		s.cpuWords(a, b, &g[k], at)
		if s.prof {
			s.cpuProfCall(&g[k], a)
			continue
		}
		g[k].code.Call(a)
	}
}

// inPlace says the form may write each residual stream's output over its
// input: every kernel that reads a stream's input runs before the first
// that writes its output.
func (f *cpuBlockForm) inPlace() bool {
	for k := range 64 {
		lastRead, firstWrite := -1, len(f.codes)
		for i, c := range f.codes {
			if c.reads&(1<<k) != 0 {
				lastRead = i
			}
			if c.writes&(1<<k) != 0 {
				firstWrite = min(firstWrite, i)
			}
		}
		if lastRead >= firstWrite {
			return false
		}
	}
	return true
}

// streamIn is the residual stream whose input v is in block graph gr, -1
// for a value that is none: the graph's first inputs, one a stream it
// writes.
func streamIn(gr *graph.Graph, v graph.ValueID) int {
	if k := slices.Index(gr.In, v); k < len(gr.Out) {
		return k
	}
	return -1
}

// cpuRoleWeight is block li's (l's) matrix of role r, nil for one it does
// not hold; with no block, the head's. An encoder segment's blocks hold
// theirs in the encoder (encRoleWeight).
func (s *Instance) cpuRoleWeight(li int, l *layer, r jlm.Role) *tensor {
	if l == nil {
		switch m := s.m; {
		case li == cpuInputRows && r == jlm.RoleTokenEmbd:
			return &m.embd
		case li == cpuInputRows && r == jlm.RolePosEmbd:
			return &m.posEmbd
		case r == jlm.RoleOutput || r == jlm.RoleTokenEmbd:
			return s.outW
		case r == jlm.RoleStreamExpand && m.altProj.rows != 0:
			return &m.altProj
		case r == jlm.RolePLETokEmbd && m.pleTok.e != nil:
			return &m.pleTok
		case r == jlm.RolePLEModelProj && m.pleProj.e != nil:
			return &m.pleProj
		case r >= jlm.RoleStreamCollapse1 && int(r-jlm.RoleStreamCollapse1) < len(m.altUnembd):
			return &m.altUnembd[r-jlm.RoleStreamCollapse1]
		case r == jlm.RoleEmbdDense1:
			return &m.dense1
		case r == jlm.RoleEmbdDense2:
			return &m.dense2
		case m.ds4Head != nil && r == jlm.RoleHCHeadFn:
			return &m.ds4Head.fn
		case m.q4eHead != nil && r == jlm.RoleHCHeadDown:
			return &m.q4eHead.hc.down
		case m.q4eHead != nil && r == jlm.RoleHCHeadUp:
			return &m.q4eHead.hc.up
		}
		return nil
	}
	switch r {
	case jlm.RoleNextnEHProj:
		if l.mtp == nil {
			return nil
		}
		return &l.mtp.ehProj
	case jlm.RoleNextnEmbd:
		if l.mtp == nil {
			return nil
		}
		return &l.mtp.embd
	case jlm.RoleAttnQ:
		return &l.wq
	case jlm.RoleAttnK:
		return &l.wk
	case jlm.RoleAttnV:
		return &l.wv
	case jlm.RoleAttnOut:
		return &l.wo
	case jlm.RoleFFNGate:
		return &l.gate
	case jlm.RoleFFNUp:
		return &l.up
	case jlm.RoleFFNDown:
		return &l.down
	case jlm.RoleRouter:
		return &l.router
	case jlm.RoleShExpGate:
		return &l.shGate
	case jlm.RoleShExpUp:
		return &l.shUp
	case jlm.RoleShExpDown:
		return &l.shDown
	case jlm.RoleFFNRoutedDown:
		return &l.routedDown
	case jlm.RoleFFNRoutedUp:
		return &l.routedUp
	case jlm.RoleAttnQKV:
		if l.ssmIn.rows != 0 {
			return &l.ssmIn
		}
		return &l.wq
	case jlm.RoleSCX:
		return &l.wq
	case jlm.RoleAttnGate, jlm.RoleSCB:
		// A full block of a model of latent attention gates its output by
		// its own; a linear block's is its state's gate.
		if l.mlaGate.rows != 0 && !s.c.LayerKind(li).Recurrent() {
			return &l.mlaGate
		}
		return &l.ssmGate
	case jlm.RoleAttnQA:
		return &l.wqa
	case jlm.RoleAttnQB:
		return &l.wqb
	case jlm.RoleAttnKVA:
		return &l.wkva
	case jlm.RoleAttnKB:
		return &l.wkb
	case jlm.RoleAttnVB:
		return &l.wvb
	case jlm.RoleSSMBA, jlm.RoleSCC:
		return &l.ssmBA
	case jlm.RoleSSMXDt:
		return &l.ssmXDt
	case jlm.RoleSSMXB:
		return &l.ssmXB
	case jlm.RoleSSMXC:
		return &l.ssmXC
	case jlm.RoleSSMDtProj:
		return &l.ssmDtProj
	case jlm.RoleSSMOut:
		return &l.ssmOut
	case jlm.RoleSSMFA:
		return &l.ssmFA
	case jlm.RoleSSMFB:
		return &l.ssmFB
	case jlm.RoleSSMGA:
		return &l.ssmGA
	case jlm.RoleSSMGB:
		return &l.ssmGB
	case jlm.RoleSSMConv1d:
		return &l.ssmConvTaps
	case jlm.RolePLEGate:
		return &l.pleGate
	case jlm.RolePLEProj:
		return &l.pleProj
	case jlm.RoleStreamRouter:
		return &l.altRouter
	case jlm.RoleLowRankIn:
		return &l.laurelL
	case jlm.RoleLowRankOut:
		return &l.laurelR
	case jlm.RoleIdxQ:
		return &l.idxQ
	case jlm.RoleIdxQB:
		return &l.idxQB
	case jlm.RoleIdxK:
		return &l.idxK
	case jlm.RoleIdxProj:
		return &l.idxProj
	}
	if h := l.q4e; h != nil {
		switch r {
		case jlm.RoleHCAttnDown:
			return &h.attn.down
		case jlm.RoleHCAttnUp:
			return &h.attn.up
		case jlm.RoleHCAttnInject:
			return &h.attn.inj
		case jlm.RoleHCFFNDown:
			return &h.ffn.down
		case jlm.RoleHCFFNUp:
			return &h.ffn.up
		case jlm.RoleHCFFNInject:
			return &h.ffn.inj
		case jlm.RoleNGramKey:
			return &h.ngKey
		case jlm.RoleNGramValue:
			return &h.ngVal
		case jlm.RoleNGramConv:
			return &h.ngCvTaps
		}
	}
	if d := l.ds4; d != nil {
		switch r {
		case jlm.RoleHCAttnFn:
			return &d.attnFn
		case jlm.RoleHCFFNFn:
			return &d.ffnFn
		case jlm.RoleAttnOutA:
			return &d.woA
		case jlm.RoleCompKV:
			return &d.compKV
		case jlm.RoleCompGate:
			return &d.compGate
		case jlm.RoleIdxCompKV:
			return &d.idxCompKV
		case jlm.RoleIdxCompGate:
			return &d.idxCompGate
		case jlm.RoleCompAPE:
			return &d.compAPET
		case jlm.RoleIdxCompAPE:
			return &d.idxCompAPET
		case jlm.RoleHashExperts:
			return &d.hashT
		}
	}
	return nil
}

// cpuRoleVector is block li's (l's) vector of role r, nil for one it does
// not hold; with no block, the head's.
func (s *Instance) cpuRoleVector(li int, l *layer, r jlm.Role) []float32 {
	if l == nil {
		switch r {
		case jlm.RoleOutputNorm:
			return s.outNorm
		case jlm.RoleOutputNormBias:
			return s.m.outNormB
		case jlm.RoleOutputBias:
			return s.m.outB
		case jlm.RoleOutputResScore:
			return s.m.resOut
		case jlm.RolePLEProjNorm:
			return s.m.pleNorm
		case jlm.RoleHCHeadNorm:
			if s.m.q4eHead != nil {
				return s.m.q4eHead.hc.norm
			}
		case jlm.RoleHCHeadBase, jlm.RoleHCHeadScale:
			if hd := s.m.ds4Head; hd != nil {
				if r == jlm.RoleHCHeadBase {
					return hd.base[:s.c.HCMult]
				}
				return hd.scale[:1]
			}
		}
		return nil
	}
	switch r {
	case jlm.RoleNextnENorm, jlm.RoleNextnHNorm:
		if l.mtp == nil {
			return nil
		}
		if r == jlm.RoleNextnENorm {
			return l.mtp.enorm
		}
		return l.mtp.hnorm
	case jlm.RoleAttnNorm:
		return l.attnNorm
	case jlm.RoleFFNNorm:
		return l.ffnNorm
	case jlm.RoleAttnNormBias:
		return l.attnNormB
	case jlm.RoleFFNNormBias:
		return l.ffnNormB
	case jlm.RolePostAttnNorm:
		return l.postAttnNorm
	case jlm.RolePostFFNNorm:
		return l.postFFNNorm
	case jlm.RoleAttnQNorm:
		return l.qNorm
	case jlm.RoleAttnKNorm:
		return l.kNorm
	case jlm.RoleAttnQANorm:
		return l.qaNorm
	case jlm.RoleAttnKVANorm:
		return l.kvaNorm
	case jlm.RoleAttnQBias:
		return l.bq
	case jlm.RoleAttnKBias:
		return l.bk
	case jlm.RoleAttnVBias:
		return l.bv
	case jlm.RoleAttnOutBias:
		return l.bo
	case jlm.RoleFFNUpBias:
		return l.upB
	case jlm.RoleFFNDownBias:
		return l.downB
	case jlm.RoleVFFNGateBias:
		return l.gateB
	case jlm.RoleVClamp:
		return l.clamp
	case jlm.RoleAttnSinks:
		return l.sinks
	case jlm.RoleRouterBias:
		return l.routerB
	case jlm.RoleExpGateBias:
		return l.expGateB
	case jlm.RoleExpUpBias:
		return l.expUpB
	case jlm.RoleExpDownBias:
		return l.expDownB
	case jlm.RoleExpProbsB:
		return l.expProbsB
	case jlm.RoleShRouter:
		return l.shRouter
	case jlm.RoleAttnSubNorm:
		return l.attnSubNorm
	case jlm.RoleFFNSubNorm:
		return l.ffnSubNorm
	case jlm.RoleXIELU:
		return l.xielu
	case jlm.RoleIdxQNorm:
		return l.idxQNorm
	case jlm.RoleIdxKNorm:
		return l.idxKNorm
	case jlm.RoleIdxKNormB:
		return l.idxKNormB
	case jlm.RoleLayerOutScale:
		return l.outScaleRow
	case jlm.RoleSSMA:
		return l.ssmA
	case jlm.RoleSSMDtBias:
		return l.ssmDtBias
	case jlm.RoleSSMNorm:
		return l.ssmNorm
	case jlm.RoleSSMD:
		return l.ssmD
	case jlm.RoleSSMConvBias:
		return l.ssmConvB
	case jlm.RoleSSMDtNorm:
		return l.ssmDtNorm
	case jlm.RoleSSMBNorm:
		return l.ssmBNorm
	case jlm.RoleSSMCNorm:
		return l.ssmCNorm
	case jlm.RolePLEPostNorm:
		return l.plePost
	case jlm.RoleStreamRouterNorm:
		return l.altRouterNorm
	case jlm.RoleStreamPredictCoef:
		return l.altPredT
	case jlm.RoleStreamCorrectCoef:
		return l.altCorrT
	case jlm.RoleStreamCorrectScale:
		return l.altCorrScale
	case jlm.RoleLowRankNorm:
		return l.laurelPost
	case jlm.RoleFFNNorm2:
		return l.ffnNorm2
	case jlm.RolePostFFNNorm1:
		return l.postFFNNorm1
	case jlm.RolePostFFNNorm2:
		return l.postFFNNorm2
	case jlm.RoleRouterNorm:
		return l.routerNorm
	case jlm.RoleExpScale:
		return l.expScale
	case jlm.RoleAttnResScore:
		return l.resAttn
	case jlm.RoleFFNResScore:
		return l.resFFN
	case jlm.RoleFFNRoutedNorm:
		return l.routedNorm
	}
	if h := l.q4e; h != nil {
		switch r {
		case jlm.RoleHCAttnNorm:
			return h.attn.norm
		case jlm.RoleHCFFNNorm:
			return h.ffn.norm
		case jlm.RoleNGramNormKey:
			return h.ngNormK
		case jlm.RoleNGramNormQuery:
			return h.ngNormQ
		case jlm.RoleNGramNormConv:
			return h.ngNormC
		}
	}
	if d := l.ds4; d != nil {
		switch r {
		case jlm.RoleHCAttnBase:
			return d.attnBase
		case jlm.RoleHCFFNBase:
			return d.ffnBase
		case jlm.RoleHCAttnScale:
			return d.attnScale3
		case jlm.RoleHCFFNScale:
			return d.ffnScale3
		case jlm.RoleCompNorm:
			return d.compNorm
		case jlm.RoleIdxCompNorm:
			return d.idxCompNorm
		}
	}
	return nil
}

// arenaSlots is how many pages the layer's arena holds, 0 for none.
func (s *kvPages) arenaSlots() int {
	switch {
	case s.arena == nil:
		return 0
	case s.reserved > 0:
		return s.reserved
	}
	return s.arena.slots
}
