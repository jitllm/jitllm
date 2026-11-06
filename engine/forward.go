package engine

import (
	"bytes"
	"fmt"
	"os"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/nn"
)

// Per-op timing, armed by WithProfile: each model keeps its own counters.
type opID int

const (
	opMatVec opID = iota
	opRMSNorm
	opRoPE
	opAttn
	opAct // SiLU / GELU elementwise
	opResid
	// opRouter is the mixture-of-experts gate, split out of opMatVec so its
	// cost is visible on its own.
	opRouter
	nOps
)

func opNames() []string {
	return []string{"matvec", "rmsnorm", "rope", "attention", "activation", "residual", "router"}
}

func (s *Instance) tick() int64 {
	if !s.prof {
		return 0
	}
	return time.Now().UnixNano()
}

func (s *Instance) tock(op opID, t0 int64) {
	if s.prof {
		s.eng.opNanos[op].Add(time.Now().UnixNano() - t0)
	}
}

// OpProfile returns the model's per-op nanoseconds and their names, largest first is the
// caller's problem.
func (s *Instance) OpProfile() ([]string, []int64) {
	out := make([]int64, nOps)
	for i := range out {
		out[i] = s.eng.opNanos[i].Load()
	}
	return opNames(), out
}

// ResetProfile zeroes the model's per-op counters.
//
// The counters accumulate across both phases, and a long prompt's prefill
// swamps decode; reset between the two and each phase answers for itself.
func (s *Instance) ResetProfile() {
	for i := range s.eng.opNanos {
		s.eng.opNanos[i].Store(0)
	}
}

// Instance is one generation session: the KV cache plus the scratch buffers the
// forward pass reuses. One Instance is one conversation; it is not safe to share.
type Instance struct {
	*Session
	// own is the session the instance was made with, which its Close
	// closes; Session is the one its calls run for (use).
	own *Session
	// seat is the instance's place on the CPU device, which a prefill there
	// asks whether another instance decodes.
	seat *cpudev.Seat
	// opt is the instance's knobs, and eng what the engine holds beside it
	// for its model's instances (engineState).
	opt modelOpts
	eng *engineState
	// root is that NewInstance made the instance, which closes the CPU
	// device it opened (engineState.ownCPU); the instances it makes itself
	// share it.
	root bool
	// mu is held by a call running on the instance: sessions leasing it run
	// one call at a time, or together as one step's rows (StepRuns).
	mu sync.Mutex
	// fnDevAt, fnRecAt and fnEndsAt are s's methods a session's cache asks
	// through, made once so binding a session (use) allocates nothing.
	fnDevAt  func(li int) nn.KVShareDevice
	fnRecAt  func(pos int) bool
	fnEndsAt func(c int) [kvEndSeats]uint8
	// cpu is the blocks this instance runs on the CPU as the composer forms
	// them (cpuserve.go).
	cpu cpuState
	// readout is the device reading an embedding model's pooled row, nil on a
	// model with no readout.
	readout nn.PoolDevice
	// tower is the instance's run of its model's vision tower (tower.go),
	// nil for a model with none and, with towerErr why, for one whose tower
	// did not compose.
	tower    *towerRun
	towerErr error
	// bidirAll makes a mixed prompt's every row one bidirectional run
	// (bidirRuns): a non-causal decoder's prompt with pictures in it.
	bidirAll bool
	m        *model.Model
	// pending is the move between devices PrepareMove started and no Commit
	// or Cancel has ended (moveblocks.go).
	pending *PendingMove
	// c is the configuration of the segment this instance runs: the text model's
	// (m.Cfg) for every session a caller makes, the vision tower's for the
	// Instance an image is encoded in (vision.go). The blocks are the model's
	// either way; only their geometry differs.
	c *model.Config
	// SampledOnDevice counts the tokens drawn on the device, whose token
	// came home in place of the logits.
	SampledOnDevice int64
	// outNorm and outW are the head this instance projects through: the model's
	// output norm and projection for a trunk, a prediction block's own head
	// norm (and the trunk's projection, or the block's own copy) for a draft.
	// outW points at the tensor rather than copying it: a block's own head
	// is paged, and a page-in rebinds the model's copy.
	outNorm []float32
	outW    *model.Tensor
	// ws is where this instance's forms read their weights, lease what a
	// block's call pages in during it.
	ws    weightSource
	lease blockLease
	// kvConv is the device converting a binary16 or q8 history's rows, nil
	// for a history of floats.
	kvConv nn.KVConvDevice
	// heads is the device running the head and the samplers.
	heads nn.HeadDevice
	// entry is the device writing a prediction block's entry rows, nil on an
	// instance that starts at no prediction block.
	entry nn.EntryDevice
	// tables is the device writing a call's rotary tables, tabDesc the
	// call's rows as the caller describes them.
	tables  nn.TablesDevice
	tabDesc []uint32
	// inputs is the device writing a call's input rows, inScr the caller's
	// side of them.
	inputs nn.InputsDevice
	inScr  inputScratch
	// streams is the device running the residual streams' expansion and
	// collapse, nil for a model of one stream.
	streams nn.StreamsDevice
	// lo and hi are the blocks this instance runs, [lo, hi): the trunk, [0,
	// NLayer), for every session a caller makes, and a multi-token-prediction
	// block for the draft instance speculation drives (newDraftState). Block
	// indices stay the model's -- the KV cache, the device's placement and the
	// pager all address block li as li -- so a draft instance and its trunk share
	// one device without colliding.
	lo, hi int
	// layers is this instance's own copy of blocks [lo, hi) (layer): the
	// spans its page-ins bind and the views cut from them, so no other
	// instance's page-in moves them. bind serialises binding them: a move's
	// staging binds beside the instance's own call.
	layers []model.Layer
	bind   sync.Mutex
	// owner is the device this instance holds the model on (holdOwner), and
	// holdsCPU that a root instance holds it on its CPU device; each goes at
	// Close.
	owner    nn.ModelOwner
	holdsCPU bool
	// pageQuota is this instance's page budget (SetPageBudget), 0 for
	// unlimited, and quota its record on the CPU device; a root's alone.
	pageQuota atomic.Uint64
	quota     *cpudev.Quota
	// fr is the CPU device's frames for the model (cpudev.CPU.Frames): the
	// memory the blocks this instance runs on the CPU are read into. dense
	// is the instance's own copy of the model's dense tensors, bound at fr.
	fr    *jlm.Frames
	dense model.Dense

	// ServedOnE counts the times a prefill was placed on the E-cores' pool beside a
	// decoding session (serving.go): what a gate reads to know the split ran.
	ServedOnE int64
	groupOff  bool
	// residFold lets the o and down groups add their rows into the
	// residual (WithResidualFold).
	residFold bool
	Grouped   int64

	x, h []float32 // residual stream and two same-width scratch vectors
	q    []float32 // n_head * head_dim
	// attStride is a row of scores' stride over the context: maxSeq rounded
	// up to a vector.
	attStride int
	maxSeq    int
	reqSeq    int       // what the caller asked for; > maxSeq when the model is shorter
	kvFmt     cpu.KVFmt // the KV cache's element format

	// autoStreamed counts the blocks the last placement streamed because no
	// card could hold them resident (nn.AutoStreamer), and streamedOn marks
	// them, with those streamed to make room for the head (streamForHead).
	autoStreamed int
	streamedOn   []bool
	// ShOverlaps counts mixture layers whose shared expert ran behind the
	// routed read: the selection check for WithSharedOverlap.
	// ple is this token's per-layer embedding inputs (Config.PLEDim), pleT
	// the table row they are built from, pleG and pleO a block's gate and
	// projection, and bple a batched chunk's inputs, one row each. See ple.go.
	ple, bple []float32
	// xa is the one-row residual's streams (x is its stream 0), on a model of
	// several (altup.go).
	xa  []float32
	tok int32
	// tok1 hands a decode step's token to the device (tokenIDs) without an
	// allocation, and rowx one row's streams in a prefill's row-by-row
	// device retry (rowStreams).
	tok1    [1]int32
	tokFill bool
	// rowsFresh and rowSlots say a step's rows' rotary tables are made for
	// this step, and each row's slot (rowTables); promptSeq and promptPos
	// are a prompt chunk's rows' sequences and positions (chunkRows).
	rowsFresh            bool
	rowSlots             []int
	promptSeq, promptPos []int
	// oneFresh and onePos say the decode row's rotary tables are made for
	// this token, at onePos.
	oneFresh bool
	onePos   [1]int
	// expHold pins a layer's routed expert pages while its FFN reads them;
	// one per instance, reused every layer of every token.
	expHold model.ExpertHold
	// spareHost is a closed session's prompt scratch, taken from the model with
	// its promptBuf, for growBatch and growMoEBatch to grow into.
	spareHost hostBatch
	// traceLayer is the block moe() is running, for traceExperts.
	// Per-row selection and weight for a batched chunk, so moeBatch can visit
	// experts in order rather than rows.
	bsel []int32
	bw   []float32
	// The per-expert gather of a batched mixture FFN (moeBatchExpert): which
	// rows chose the expert, their weights, and the rows' input, gate, up and
	// down scratch.
	bmRow    []int32
	bmW      []float32
	bmX, bmY []float32
	bmG, bmU []float32

	// qgate is the double-width query projection of an architecture that folds
	// its attention output gate into attn_q, and ogate is the gate after the
	// per-head deinterleave. Both are nil when Config.AttnOutGate is false,
	// which is every architecture but the hybrid.
	qgate, ogate []float32

	// declines counts, by reason, the blocks no device took. See DeviceDeclines.
	declines map[string]int

	// The recurrent state of a hybrid's linear layers, which is not a KV
	// cache: rconv holds the last conv-1 columns the convolution needs and
	// rstate one vDim x kDim matrix per value head, both constant in the
	// context length. Indexed [layer][row*len + i], nil for a full-attention
	// layer and for non-hybrids. dg carries the geometry.
	dg model.DeltaGeom
	// The delta rule's scratch, allocated only for a hybrid. dOut is per
	// value head so the heads can run in parallel without sharing.
	// dLowRank is the rank-kDim temporary KDA's two low-rank gates project
	// through, and dGateWaste receives the half of DeltaGate32JIT's pair that
	// the call in question is not for. Both nil on qwen3next.
	// mvCalls counts mv: each one reads its weight once, the count a gate
	// holds a batched path to (TestLinearRowsCounts).
	// dSB and dSC are a Mamba-1 block's B and C, and dZeroA the zero rates
	// its dt goes through SSDGate32JIT against (only the softplus is kept).

	// res keeps the hot experts resident and drops the rest; nil when the model
	// fits its budget, which is every model but a large MoE on a small one.
	res *model.Residency
	hot *Hotness
	// plan and ldCand are kept so the seam can be moved after load: growing
	// the device's prefix needs the same LayerPlan the first offer used.
	plan   nn.LayerPlan
	ldCand nn.LayerDevice
	// memBudget is what SetMemBudget was asked for -- or what newState read from
	// sched.MemBudget when nobody did -- kept because the share of it the host
	// actually gets depends on the device, which may be attached after the budget
	// was set. See hostBudget.
	memBudget uint64
	// memBudgetAsked separates "the caller set a budget" from "we read one at
	// construction", which is the difference between arming the expert
	// residency tracker and merely recording a number. newState does only the
	// second, and re-applying must not start doing the first.
	memBudgetAsked bool
	// hostRes is the attached device when it reports a holding of the host's
	// memory, counted against the model's page budget while attached, and
	// hostSeen the holding both budgets were last divided for (hostheld.go).
	hostRes  nn.HostReserver
	hostSeen uint64
	// gpuTarget is where the seam is heading; gpuLayers is where it is.
	gpuTarget int
	seam      *seamTuner
	// prof is m.opt.profile, copied at construction so tick/tock (every op of
	// every layer) read one field. WithProfile is a load option, so the copy
	// is exact.
	prof bool

	// devFallback finishes a token on the host when the device fails, rather
	// than returning an error. See SetDeviceFallback.
	devFallback bool
	// noHeadFold keeps the output projection out of a prompt's last chunk (see
	// prefillSrc): the tail call runs it, as before the fold existed. It is the
	// other arm of the fold's gate and nothing else sets it.
	noHeadFold bool
	devDemoted int
	// demotedBy counts the demotions by the cause the device gave, and
	// buildRefused and declined the calls a kernel that did not build and a
	// decline ended (refusal): neither moves anything.
	demotedBy    [nFailCauses]int
	buildRefused int
	declined     int
	// tailRowChunks counts few-row prompt chunks run as one ragged step
	// (tailrows.go). See TailRowChunks.
	tailRowChunks int
	// tailPos is a ragged tail step's positions, kept so the step allocates
	// nothing once grown.
	tailPos []int
	rrFlat  []int
	// relocate hands the device's highest block to the host when its history
	// cannot grow, rather than letting the token fail. See SetRelocate.
	relocate     bool
	devRelocated int
	// pfStreamed and pfKept count the prompts that streamed their host blocks
	// through the device and those that measured the host faster
	// (prefillstream.go).
	pfStreamed, pfKept int
	// devSpilled counts blocks relocateFor moved onto a later device.
	devSpilled int
	// moved marks the blocks relocation sent home, which are the ones it takes
	// back when the device reports room (roomSeen is the report last read).
	moved        []bool
	roomSeen     uint64
	devReclaimed int

	// nseq is the sequences an instance's cache is strided by, one: a
	// session holds one sequence, and sessions step together (StepRuns)
	// rather than as slots of one cache.
	nseq int
	// hist is the sessions whose history lives under this instance's
	// placement: each that ran on it and has not left (use, forget). A move
	// of the instance's blocks moves theirs (settleHist).
	hist     []*Session
	settling bool
	// kvl is where this session's KV cache puts things; see kvLayout. It is the
	// global layers' layout; kvlAt is any layer's.
	kvl kvLayout
	// attnChunk is how many query heads one pool task claims in decode
	// attention. With GQA adjacent heads share a kv head, so a larger chunk
	// keeps sharing heads on one worker. A field so both arms of an A/B are
	// live in one process.
	attnChunk int
	// attnPair is the paired-kernel selector; see AttnPair.
	attnPair int
	// attnPaired counts head pairs actually served by the paired kernels, so a
	// gate can tell the paired path ran (it silently does not at odd gqa).
	// Atomic because the pool writes it from every worker.
	attnPaired atomic.Int64

	blogits []float32 // a step's rows' logits, NVocab each
	allRows []int     // 0..nseq-1, the rows Reset zeroes on a device
	// placing is true while SetDeviceLayers offers blocks, when WithPlacement's
	// map applies; placeErr is what it could not do.
	placing  bool
	placeErr error
	// offerOn is the device MoveBlocks is placing CPU blocks on: offerRange
	// offers each to it alone ("" lets the tier choose).
	offerOn string
	// Attention scores per (sequence, head). atth/attf are the nseq==1 case;
	// a batch parallelises over sequences and heads, so every pair needs a row.
	bathf []float32
	// scoreReach is the key reach bathf and battf are sized for (rowsStride).
	scoreReach int

	// Batched prefill scratch, allocated on first use and sized by
	// PrefillChunk rather than by the sequence length.
	bx, bh    []float32
	bqf, bxbf []float32
	battf     []float32 // per-token score rows, at rowsStride

	// device is the device SetDeviceLayers was handed, before Attach made ld a
	// session of it: what a draft instance attaches to (Speculate).
	device    nn.LayerDevice
	gpuLayers int
	// roomToCPU counts batch steps whose device had no room for them, so the
	// blocks moved to the CPU device before the step (relocateRows, forwardRows).
	roomToCPU int
	// The lists a step this instance leads reuses (step.go): Step's runs, the
	// rows' sessions and positions, the wanted rows' logits and the result.
	stepRuns []Run
	stepSess []nn.LayerRunner
	stepPos  []int
	stepTok  []int32
	// chunkTok is a prefill chunk's rows' tokens, -1 for a row naming none.
	chunkTok   []int32
	stepLogits []float32
	stepRes    [][]float32
	// A step across host sessions (stepHost) this instance leads: each row's
	// session, whether the row is its session's furthest (the one that
	// faults its attention window in), and each row's slot, always 0.
	rowOwn []*Session
	// rowInst is each row's run's instance: a row of another instance runs
	// on that instance's form, composed for its history (cpuOwnerRuns).
	rowInst []*Instance
	rowWin  []bool
	rowSeq  []int
	// stepb is the model's step scratch holder while this instance leads a
	// host step (borrowStep).
	stepb *stepBuf
	// rowCausal is a ragged step's rows in position order (rowsHost).
	rowCausal []int
	// onDev[li] says block li runs on a device: the truth for execution, where
	// gpuLayers is the truth for the seam. Placement is a set and migration a
	// boundary: gpuLayers is the contiguous device prefix (what -gpu-layers
	// sizes and seamtune moves), while a hybrid whose kinds alternate can have
	// placed blocks with a prefix of zero.
	onDev []bool
	// cpuHeld says this instance keeps (jlm.File.Keep) the pages of the
	// blocks of [lo, hi) it does not place, from its making until Close.
	cpuHeld bool
	// head is non-nil when the device also runs the output norm and the
	// vocabulary projection, which it is only offered when it took every block.
	// headReady is what head is set back to by SetHeadOnDevice.
	head      *nn.Head
	headReady *nn.Head
	// HostPick asks a prefill whose head runs on the host to project the
	// listed vocabulary rows alone into HostPickVals (pickOnHost), the
	// host's half of the label pick a device head does (nn.Head.Pick);
	// HostPicked says the last prefill did. A decision readout sets them.
	HostPick     []int32
	HostPickVals []float32
	HostPicked   bool
	// cs is the rotary table for the current position: NRot floats, filled once
	// per token and read by every head and by the device blocks.
	cs []float32
	// csSWA is the local layers' rotary table, allocated only for an
	// architecture with two bases. nil everywhere else, which is what keeps
	// choosing the table a pointer compare for every other model.
	csSWA  []float32
	bcsSWA []float32
	// bcs is cs for a whole prefill chunk: one nRot-wide rotary table per row,
	// which is what the device's batched RoPE reads.
	bcs []float32
	// rowsLog and rowsHid are where a speculation step on the device that wants
	// a tail of its rows has the head write every row (stepRowsDevice), the
	// tail then copied out; rowsTok is where a greedy one's tokens land.
	rowsLog, rowsHid []float32
	// rowsHeads counts a hyper-connected model's verifications whose head
	// ran on the device (rowsHead).
	rowsHeads int
	rowsTok   []int32
	// rowsPos is a stream model's speculation step's positions and slots
	// (stepRowsStreams).
	rowsPos []int
	// draftIn is a prediction block's draft rows over a residual of every
	// stream, each row every stream, which a chunk lays out stream-major
	// (draftStreams); nil on every other Instance.
	draftIn []float32

	// emb is set for the length of one Embedder.Embed on a decoder embedding
	// model: prefill then pools the final residual instead of projecting it
	// onto the vocabulary. nil for every generation session.
	emb *embedMode

	// bidir is set for the length of one PrefillMixed whose spans carry a
	// bidirectional run (a Gemma 3 image): the positions whose rows see each
	// other in both directions. nil in every other call.
	bidir []nn.KeyRun
	// bidirCutFault lets a prefill chunk end inside a run: the violation
	// TestBidirRunIsNeverCut runs. false in every real use.
	bidirCutFault bool

	// rows is set for the length of one speculation step (specrows.go): the
	// rows whose logits and hidden state the step wants back, and the
	// recurrent snapshots it takes. nil in every other call.
	rows *rowsOut

	// rowTap, when set, receives every prefill chunk's final residual rows --
	// after the last block, before the output norm -- with the position of the
	// first. Nil in every session but a gate's: it is how a comparison sees
	// every position of a prompt rather than only the last row's logits.
	rowTap func(base int, rows []float32)
	// recEv is the spacing of this hybrid's restore points (recEvery; 0 not
	// yet asked, -1 none), and recBuf and recRd the buffer and reader a point
	// is stored through, kept so a warm prompt makes none.
	recEv  int
	recBuf []float32
	recRd  bytes.Reader
	// endWord is the one word noteEnd and endsAt move a boundary's prompt
	// ends through.
	endWord [1]float32
	// recBatch is sealRecOnDevice's block list, kept.
	recBatch []int32
	// recStage is faultRecurrentAt's staging, a buffer a linear block.
	recStage [][]float32
}

// shareActWindow hands device d model m's activation window where d
// quantizes activations itself: devices quantizing with two windows do not
// agree on token ids.
func shareActWindow(m *model.Model, d nn.LayerDevice) {
	if w, ok := d.(nn.ActWindower); ok {
		w.SetActWindow(m.ActWindow)
	}
}

// kvFmtOf is the CPU's history layout for history type t.
func kvFmtOf(t model.KVType) cpu.KVFmt {
	switch t {
	case model.KVF16:
		return cpu.KVF16
	case model.KVQ8_0:
		return cpu.KVQ8
	}
	return cpu.KVF32
}

// NewInstance composes model m for one placement -- its devices and knobs
// opts' -- every block formed now, its sessions' histories up to maxSeq
// positions. A CPU device opts name (WithCPU) is the caller's; without one
// the instance opens one of its own and closes it with itself. A knob the
// model refuses (a KV type, Config.KVTypeRefusal) or a CPU the composer
// cannot serve is refused here, naming why.
func NewInstance(m *model.Model, maxSeq int, opts ...Option) (*Instance, error) {
	lo := loadOpts{opt: defaultOpts()}
	for _, o := range opts {
		o(&lo)
	}
	_, listed, err := cpuOfDevices(lo.devices)
	if err != nil {
		return nil, err
	}
	lo.opt.noCPU = !listed
	if lo.opt.kvTypeSet {
		if err := m.Cfg.KVTypeRefusal(lo.opt.kvType); err != nil {
			return nil, err
		}
	}
	g := &engineState{cpu: lo.cpu, fuse: lo.opt.fuse}
	if g.fuse == nil {
		g.fuse = &fuseCount{}
	}
	if g.cpu == nil {
		if g.cpu, err = OpenCPU(opts...); err != nil {
			return nil, err
		}
		g.ownCPU = true
	}
	// The CPU device keeps the model's forms, arenas and frames while an
	// instance holds it there: this one's hold goes at its Close.
	g.cpu.HoldModel(m.Id)
	g.host = g.cpu.Attached(m.Id, func() any { return &hostShared{} }).(*hostShared)
	// The page quota is the caller's, or the model's memory plan, recorded
	// before the frames are made: it budgets them from their first read.
	pq := m.Gc.Budget()
	if lo.opt.pageBudgetSet {
		pq = lo.opt.pageBudget
	}
	q := g.cpu.NewQuota(m.Id, pq)
	fr, err := g.cpu.Frames(m.Id, m.NewFrames)
	if err == nil {
		var s *Instance
		if s, err = newInstanceOver(m, g, fr, lo.opt, m.Cfg, 1, maxSeq, 0, m.Cfg.NLayer); err == nil {
			s.root, s.holdsCPU = true, true
			s.pageQuota.Store(pq)
			s.quota = q
			return s, nil
		}
	}
	q.Drop()
	g.cpu.ReleaseModel(m.Id)
	if g.ownCPU {
		g.cpu.Close()
	}
	return nil, err
}

// OpenCPU is the CPU device opts state: WithDevices' cpu spec (cpu:sse,
// cpu:novnni, ...) and WithPool's workers. Every op a token runs is generated
// code, so a CPU with no lowering, or one below the emitter's baseline (an
// x86 below SSE4.1+SSSE3), is refused, naming what it lacks. The caller
// closes it, after every instance on it and every model they ran.
func OpenCPU(opts ...Option) (*cpudev.CPU, error) {
	var lo loadOpts
	for _, o := range opts {
		o(&lo)
	}
	cpuOpts, _, err := cpuOfDevices(lo.devices)
	if err != nil {
		return nil, err
	}
	c, err := cpudev.Open(cpuOpts, lo.pool...)
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	return c, nil
}

// newInstanceOver is an instance over blocks [lo, hi) of a segment whose
// geometry is c's (the text's, or the tower's: towerConfig), sharing g,
// paging what it runs on the CPU into fr and running with knobs opt.
func newInstanceOver(m *model.Model, g *engineState, fr *jlm.Frames, opt modelOpts, c *model.Config, nseq, maxSeq, lo, hi int) (*Instance, error) {
	// Clamp to the model's context, and record the request (reqSeq) so a
	// later "kv cache is full" can name the number the caller passed.
	req := maxSeq
	if maxSeq > c.NCtx {
		maxSeq = c.NCtx
	}
	s := &Instance{Session: &Session{model: m}, m: m, c: c, lo: lo, hi: hi, maxSeq: maxSeq, reqSeq: req, nseq: nseq,
		devFallback: true, relocate: true, prof: opt.profile, outNorm: m.OutNorm, fr: fr,
		groupOff:  opt.noGroups,
		residFold: opt.residFold, opt: opt, eng: g}
	s.own = s.Session
	s.own.on = s
	if err := m.BindDense(fr, &s.dense); err != nil {
		return nil, err
	}
	s.outW = &s.dense.Output
	s.layers = make([]model.Layer, hi-lo)
	for li := lo; li < hi; li++ {
		m.Layers[li].CloneTo(&s.layers[li-lo])
	}
	s.holdCPURuns()
	s.seat = g.cpu.Sit()
	// An instance over a prediction block projects through that block's
	// head: the trunk's, bound with the instance's dense tensors, where the
	// block has none of its own.
	if hi > lo {
		if w := s.layers[0].Mtp; w != nil {
			s.outNorm, s.outW = w.HeadNorm, &w.Head
			if w.Head.E != nil && w.Head.E.Block == jlm.DenseBlock {
				s.outW = &s.dense.Output
			}
		}
	}
	s.ws = newWeightSource(m, &s.dense, fr, s.outW, s.outNorm)
	s.kvl = kvLayout{maxSeq: maxSeq, nKV: c.NKVHead,
		headDim: c.HeadDim}
	// MLA caches one row per position for the whole layer: the compressed
	// latent plus the shared rotary key (KVLoraRank + NRot), which every head
	// attends to. So nKV is genuinely 1.
	if c.KVLoraRank != 0 {
		s.kvl.nKV, s.kvl.headDim = 1, c.KVDim()
	}
	s.x = make([]float32, c.NEmbd)
	s.h = make([]float32, c.NEmbd)
	// The attention output is exactly n_head*head_dim wide, which is not
	// n_embd on every model (Qwen3-30B-A3B, gemma-2); the output projection
	// requires len(x) == k.
	s.q = make([]float32, c.MaxQDim())
	s.initHotness()
	// A sink takes one slot past the last position, so the row grows by one.
	s.attStride = attStride(maxSeq + m.SinkSlot())
	s.growPLE()
	s.allocStreams()
	s.logits = make([]float32, c.NVocab)
	// One rotary table per row: a ragged batch has rows at different
	// positions.
	s.cs = make([]float32, nseq*c.RopeW())
	if m.RopeSWA != nil {
		s.csSWA = make([]float32, nseq*c.LocalNRot())
	}
	s.bpos = make([]int, nseq)
	s.rmarks = model.RopeMarks{}
	s.allocRecurrent(nseq)
	if c.AttnOutGate {
		qd := c.NHead * c.HeadDim
		s.qgate = make([]float32, nseq*2*qd)
		s.ogate = make([]float32, nseq*qd)
	}

	workers := cpu.NewPool(s.poolOptions()...)
	s.cpu.x = cpudev.NewExec(workers, cpudev.NewComposer(g.cpu.Forms(m.Id), g.cpu.Target.ISA(), workers.Workers(), opt.cpuFloat))
	s.cpu.x.SetFusion(!opt.noFuse)
	s.cpu.x.FuseCounts(&g.fuse.tries, &g.fuse.runs)
	// The cache width: WithKVType (or WithKVF16, SetKVType) forces it;
	// otherwise binary16, half the history's bytes, wherever the CPU
	// narrows in hardware -- everywhere but an x86 host the composer forms
	// from its narrow passes (composer.NarrowsCPU). A quantized cache is
	// only ever forced (KVType).
	want := kvFmtOf(opt.kvType)
	if !opt.kvTypeSet {
		want = cpu.KVF16
		if composer.NarrowsCPU(cpudev.Target(s.cpuISA(), 1)) {
			want = cpu.KVF32
		}
	}
	// A selection caches its indexer's key in the row, and the selection over
	// it is discrete: a key rounded to binary16 moves a block's best score by
	// ~1e-3, which flips a near tie, and a flipped block is a different
	// attention, not a nearby one. The references keep the key at f32
	// (llama.cpp's converter and cache force F32 for an indexer), and so
	// does every device here (the paged K pool is f32), so the CPU's
	// default is f32 too; WithKVF16(true) still forces binary16.
	if c.Selection() != nil && !opt.kvTypeSet {
		want = cpu.KVF32
	}
	// A forced q8 cache on a model that refuses one was refused at
	// NewInstance (Config.KVTypeRefusal); this is the backstop for the
	// instances it makes itself.
	if want == cpu.KVQ8 && c.KVTypeRefusal(model.KVQ8_0) != nil {
		want = cpu.KVF32
	}
	s.kvFmt, s.kvl.fmt = want, want
	s.Session.limit, s.Session.kvFormat, s.Session.span = maxSeq, want, [2]int{lo, hi}
	// kvl.fmt sizes every page.
	s.kv = newKVCacheRange(c, s.nextCacheID(), nseq, s.kvl, nil, s.opt.kvPage, s.lo, s.hi, s.Session.cpuView(), g.cpu.KVPool(m.Id))
	s.kv.reserve(s.maxSeq)
	if err := s.kv.useArenas(s.maxSeq, s.kvArenaFor); err != nil {
		s.cpu.x.Composer().Note(err)
	}
	s.cpuPlace()
	// A text instance of a model with a tower runs its pictures through a
	// tower run of its own, composed with it (tower.go); one that does not
	// compose keeps its error for the picture that asks.
	if c == m.Cfg && m.Tower != nil && s.cpu.x.Composer().Err() == nil {
		if s.tower, s.towerErr = s.newTowerRun(); s.towerErr != nil {
			s.tower = nil
		}
	}
	// Read at construction too, because an embedded caller may never call
	// SetMemBudget; it overrides this with the caller's number.
	s.memBudget = s.m.Gc.Budget()
	return s, nil
}

// SetMemBudget tells the session how many bytes of weights may stay resident,
// so it can keep the experts the router keeps asking for and drop the rest. 0
// means track but never evict.
//
// It is callable at any time, which a service needs: tightening the budget
// demotes on the next token, loosening it stops evicting, and the hotness
// counts and resident set are kept either way.
func (s *Instance) SetMemBudget(bytes uint64) {
	s.memBudget, s.memBudgetAsked = bytes, true
	s.applyMemBudget()
}

// HostBudget is the weight budget actually in force on the host: what
// SetMemBudget was given, less the share of it a device on the host's own memory
// is allowed to fill. It is what a report should print, because it is the number
// the residency manager is working to.
func (s *Instance) HostBudget() uint64 {
	b, _ := s.hostBudget()
	return b
}

// hostBudget splits the requested budget into the host's share and the device's.
//
// An integrated GPU spends the host's bytes, so counting them twice would
// thrash rather than fail. The tier reports what it holds through HostReserved
// and this subtracts it. It is an optional interface so model/ need not know
// what a backend is; a discrete card leaves the budget unchanged.
func (s *Instance) hostBudget() (host, device uint64) {
	host = s.memBudget
	d, ok := s.dev().(nn.HostReserver)
	if !ok {
		return host, 0
	}
	device = d.HostReserved()
	if device >= host {
		// The caller gave a device more than the whole host budget. Refusing to
		// go negative is bookkeeping; the number that matters is that the host
		// keeps nothing, which is what it asked for.
		return 0, device
	}
	return host - device, device
}

// applyMemBudget pushes the host's share at the two managers that spend it. It
// is re-run when a device is attached and whenever what it holds moves
// (followHost), because those are what change the answer.
func (s *Instance) applyMemBudget() {
	host, _ := s.hostBudget()
	if s.memBudgetAsked {
		s.initResidency(host) // recomputes the expert share of the new budget
	}
}

// dev is the attached accelerator, or nil. ldCand rather than ld because ld is
// this instance's session view (nn.Attacher) and ldCand the device itself, which is
// what answers HostReserved.
func (s *Instance) dev() nn.LayerDevice {
	if s.ldCand != nil {
		return s.ldCand
	}
	return s.ld
}

// Close releases the generated kernels and detaches the device session. Every
// Instance must be closed before its Model (`defer m.Close()` then
// `defer st.Close()` gives that order).
func (s *Instance) Close() error {
	if s.own != nil {
		s.Session = s.own
	}
	s.tower.close()
	s.tower = nil
	s.eng.cpu.Leave(s.seat)
	if s.root {
		defer func() {
			if s.holdsCPU {
				s.holdsCPU = false
				s.quota.Drop()
				s.eng.cpu.ReleaseModel(s.m.Id)
			}
			if s.eng.ownCPU {
				s.eng.cpu.Close()
			}
		}()
	}
	// The card this instance holds the model on lets it go first.
	defer s.holdOwner(nil)
	// A move prepared and never committed gives its target's copies back.
	if s.pending != nil {
		s.pending.Cancel()
	}
	// The session's history on the device goes back to the tier, which keeps
	// the model's weights for whoever else is using them.
	if d, ok := s.ld.(nn.Session); ok {
		d.Detach()
	}
	// The history the device gave back was host memory on an integrated GPU,
	// and the model's pages get it back once no instance holds the device.
	s.holdHost(nil)
	s.kv.giveBack(s.eng.cpu.KVPool(s.m.Id))
	s.unmapOwn()
	// The blocks this instance ran on the host no longer need their pages for
	// it. Close may be called twice; cpuHeld makes the second a no-op.
	if s.cpuHeld {
		s.cpuHeld = false
		for li := s.lo; li < s.hi; li++ {
			if !s.devAt(li) {
				s.keep(li, li+1, -1)
			}
		}
	}
	s.keepSpare()
	s.cpu.x.Pool().Close()
	return nil
}

// SetDevice attaches an optional accelerator to this instance's matvecs. Nil, the
// default, is the CPU-only engine.
//
// A Device that can run whole blocks is offered every layer (see offerRange),
// and what it takes is decided by the device's own memory rather than by a
// separate placement policy.
func (s *Instance) SetDevice(d nn.LayerDevice) error {
	return s.SetDeviceLayers(d, -1)
}

// SetDeviceLayers is SetDevice with a cap on how many blocks the device is
// offered. -1 offers all of them and lets the device's own memory decide, which
// is the useful default; a positive cap is for pinning the split by hand, and
// zero attaches the device for single matvecs only.
//
// WithPlacement's map is applied here. The error reports a placement that
// named a device not attached, a strict one that could not be met (nothing is
// placed then), and a device this instance cannot move onto at all.
func (s *Instance) SetDeviceLayers(d nn.LayerDevice, max int) error {
	defer s.settleHist()
	// The tower's blocks go where the text's do, every one offered (tower.go).
	if s.tower != nil {
		if err := s.tower.s.SetDeviceLayers(d, -1); err != nil {
			return fmt.Errorf("model: the tower's blocks: %w", err)
		}
	}
	// MigrateKV moves float32: a binary16 or q8_0 history travels converted
	// (Instance.migrateKVAs).
	// An empty binary16 cache steps down to the f32 the device holds; one
	// with history keeps its format and travels converted, widened on its way
	// to the device and narrowed home (migrateKVAs), as a q8_0 one does. A
	// Instance whose CPU blocks run composed keeps the binary16 its CPU device
	// holds (composer.Halves), in the arenas its blocks were bound to, as a cache
	// with history does.
	// The history comes home before the device changes (or detaches), or the
	// next tier would attend over empty buffers. SetGPULayers(0) is that walk
	// in the right order, reused rather than restated.
	if s.ld != nil && s.devCount() > 0 {
		if s.SetGPULayers(0); s.devCount() != 0 {
			// The old tier refused to give the history back, or a block is
			// pinned there. Keeping it is the only correct answer -- moving on
			// would silently discard it.
			return fmt.Errorf("model: %d block(s) stay on the device attached now "+
				"(pinned, or its history would not come home)", s.devCount())
		}
	}
	// An empty binary16 cache steps down to the f32 the device holds, so a
	// history that later moves between devices through the host comes back
	// as it went (TestLayerRelocatesToAnotherDevice,
	// TestMLAStepsTheKVDownAtTheRightWidths).
	if d != nil && s.pos == 0 && s.kvFmt == cpu.KVF16 && !s.histWritten() {
		s.stepDownKVToF32()
	}
	// A borrowed JIT is the text instance's, attached to its device by it.
	if !s.borrowsJIT() {
		shareActWindow(s.m, d)
	}
	s.clearOnDev()
	s.moved = nil
	s.device = d
	// A device on the host's memory is counted against the model's pages from
	// here, for what it holds; once the offers below have placed what they
	// will, both budgets are re-divided for it.
	s.holdHost(d)
	defer s.followHost()
	s.holdOwner(d)
	if d == nil {
		// Detaching gives the host budget back (an integrated GPU's share).
		s.setLD(nil)
		s.ldCand, s.Session.tier = nil, nil
		s.applyMemBudget()
		return nil
	}
	ld, ok := d.(nn.LayerDevice)
	if !ok {
		return nil
	}
	// Mixture blocks are offered too: Gate/Up/Down carry the bank and
	// LayerWeights.Router carries ffn_gate_inp; PrepLayer declines by name if
	// it cannot run one.
	c := s.c
	p := nn.LayerPlan{
		// Whose blocks these are: a device holds one model's at a time.
		Model: s.m.Id, File: s.m.Container, FrameBudget: s.pageQuota.Load(),
		NEmbd: c.NEmbd, NHead: c.NHead, NKVHead: c.NKVHead, HeadDim: c.HeadDim,
		NRot: c.NRot, NFFN: c.NFFN, MaxSeq: s.maxSeq, Vocab: c.NVocab, FinalSoftcap: c.FinalSoftcap,
		RMSEps: c.RMSEps, RopeBase: c.RopeBase, RopeNeox: c.RopeNeox,
		NoPosEnc: c.NoPosEnc,
		QKNorm:   c.QKNorm, QKNormWide: c.QKNormWide, QKNormPost: c.QKNormPost, NoExpertNorm: c.NoExpertNorm,
		// The tier picks the local or global rotary table per block from
		// this; zero means one kind of layer.
		SWAPeriod: swaPeriodFor(c),
		// The folded constants the host's rotary kernel reads, so the device
		// builds its own table and the two tiers agree by construction.
		RopeTab:    model.RopeTabPlanes(&s.m.Rope, c.NRot),
		RopeTabSWA: model.RopeTabPlanes(s.m.RopeSWA, c.LocalNRot()),
		// The tables as the front end states them, and a multi-axis
		// rotary's planes, for a device that serves them (front.go).
		Tables:      s.tableGraphs(),
		RopeTabAxes: s.axesPlanes(),
		GridPlanes:  s.gridPlanes(),
		RopeMarks:   s.marksCap(),
		Rotaries:    s.rotaries(),
		RopeSplit:   c.RopeXD,
		// The sliding window itself (TestSlidingWindowRunsOnTheDevice).
		SWAWindow: windowFor(c, s.maxSeq),
		// Read off the loaded weights, not the architecture name: the
		// post-norms are optional per-layer roles.
		PostNorm:  s.m.HasPostNorm(),
		NoPreNorm: s.m.HasNoPreNorm(),
		// BitNet's inner norms, read off the weights as the post-norms are.
		SubNorm:     s.m.HasSubNorm(),
		AttnSoftcap: c.AttnSoftcap,
		ActWin:      s.m.ActWindow,
		// Zero for a dense model, which is what makes every existing plan
		// unchanged.
		NExpert: c.NExpert, NExpertUsed: c.NExpertUsed, NFFNExp: c.NFFNExp,
		// The hybrid's shared expert and output gate, which a device must see
		// to run or decline.
		NFFNShExp: c.NFFNShExp, AttnOutGate: c.AttnOutGate,
		// Llama 4's, each of which a device either runs or declines by name.
		SWAChunked: c.SWAChunked, NoPEGlobal: c.NoPEGlobal, QKL2Norm: c.QKL2Norm,
		AttnTempScale: float32(c.AttnTempScale), AttnTempFloor: float32(c.AttnTempFloor),
		AttnTempOffset: float32(c.AttnTempOffs), ExpertWeightIn: c.ExpertWeightIn,
		// Granite's: a device scales each block output at its residual add.
		ResidualScale: c.ResidualScale,
		// The classic block's (C6); UngatedFFN is per block, in planFor.
		LayerNorm: c.LayerNorm, Parallel: c.Parallel,
		// The block's own geometry is planFor's.
		MixedGeom: c.MixedGeom(), VNorm: c.VNorm,
		NoAttention: !c.Attends(),
	}
	s.plan, s.ldCand = p, ld
	hi := s.hi
	if max >= 0 && s.lo+max < hi {
		hi = s.lo + max
	}
	// The device stays attached even when it took nothing, so the seam can
	// grow later. One session per instance: a tier offering Attach hands back a
	// view carrying this instance's own KV cache (see
	// docs/design/device-sessions.md); two instances must not share one cache.
	if a, ok := ld.(nn.Attacher); ok {
		ld = a.Attach()
		// The single matvecs are the session's too: the tier's crossing
		// verdict is sampled per session, and offered through the tier itself
		// they land on its zero session, which outlives every instance, so a
		// second instance would inherit the first's sample and run the same
		// token with different kernels
		// (TestHybridSecondSessionMatchesTheFirst).
		if !s.borrowsJIT() {
			shareActWindow(s.m, ld)
		}
	}
	// Size the scratch before offering anything: the largest matvec is known
	// from the header, and reserving first never competes with weights for the
	// card's last megabyte.
	rows, k := s.m.MaxMatVec()
	ld.Reserve(rows, k)
	// Before any block is offered: a device on host memory shrinks the host's
	// budget by what it already holds (another session's blocks, the scratch
	// reserved above); followHost takes off what the offers add.
	s.setLD(ld)
	s.Session.tier, s.Session.kvRoom = s.ldCand, 0
	s.applyMemBudget()
	// The whole offer sequence is one decision: bracketed, so two instances
	// attaching at once do not interleave their offers against one budget.
	if pl, ok := ld.(nn.Placer); ok {
		pl.BeginPlacement()
		// The router spreads a model that fits over its devices instead of
		// filling the first to the brim (gpu.GPU.PlanBlocks).
		if pb, ok := ld.(nn.BlockPlanner); ok {
			pb.PlanBlocks(hi, s.planExtra(hi))
		}
		s.placing, s.placeErr = true, nil
		s.autoStreamed = 0
		clear(s.streamedOn)
		s.offerRange(s.lo, hi)
		if s.placeErr == nil {
			s.placeErr = s.pageOnCards(s.lo, hi)
		}
		s.placing = false
		pl.EndPlacement()
		// Blocks no card could hold went on streamed by default; whether that
		// beats the host is measured, not assumed (initStreamTrial).
		if s.autoStreamed > 0 && s.seam == nil && !s.opt.noStreamTrial {
			s.initStreamTrial()
		}
	} else {
		s.placing, s.placeErr = true, nil
		s.offerRange(s.lo, hi)
		if s.placeErr == nil {
			s.placeErr = s.pageOnCards(s.lo, hi)
		}
		s.placing = false
	}
	if err := s.placeErr; err != nil {
		s.placeErr = nil
		s.SetGPULayers(0)
		return err
	}
	// The history written so far goes up to the newly placed blocks, as
	// SetGPULayers' grow path does for a seam move. It runs before placeHead
	// because a failed migration shrinks the placement.
	if s.pos > 0 && s.ld != nil {
		placed := s.placedFrom(s.lo)
		for i, li := range placed {
			if !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
				// Resident weights with no history would attend over an empty
				// cache. Give the block straight back, and every placed block
				// after it, exactly as the grow path does -- saying why, or a
				// placement that took nothing reads as one never offered.
				why := "the device refused it"
				if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
					why = e.Err()
				}
				if s.kv.layers[li].anyShared() {
					why = "its pages are shared with another sequence"
				}
				for range placed[i:] {
					s.noteDecline(fmt.Sprintf("block %d's history (%d positions) did not move to the device: %s", li, s.pos, why))
				}
				for _, lj := range placed[i:] {
					s.ld.ReleaseLayers(lj, lj+1)
					s.unmarkOnDev(lj)
				}
				break
			}
		}
	}
	// The head is offered whenever the device took any block: it is read in
	// full every token, so moving it saves bytes whether or not the residual
	// travels. PrepHead needs g.bs, which PrepLayer creates.
	s.placeHead()
	if err := s.finishPlacement(); err != nil {
		return err
	}
	if err := s.reserveRows(); err != nil {
		return err
	}
	// Last, so the blocks, the head and the prompt's scratch have taken their
	// room first: the pool grows ahead into what is left, one prompt chunk's
	// worth at most.
	if pw, ok := d.(nn.KVPrewarmer); ok && s.devCount() > 0 && !s.opt.noKVPrewarm {
		pw.PrewarmKV(min(s.maxSeq, nn.MaxDevicePrefillChunk))
	}
	return nil
}

// reserveRows makes the devices hold every row's history before any is
// written. A batch's rows sit at slot*maxSeq+pos in each block's cache, so it
// needs nseq*maxSeq positions where one sequence grows into maxSeq. The
// capacity is this instance's own (a tier session's), so no other session's caches
// grow; what the cards cannot hold comes home for this instance alone -- a
// session's release leaves blocks another instance is using where they are -- and
// with nothing written yet nothing has to move. A strict placement fails
// instead, since that would move a block it named.
func (s *Instance) reserveRows() error {
	// A device that pages per sequence takes each row's pages as it writes
	// them; a whole-context reservation is the contiguous cache's.
	if sd, ok := s.ld.(nn.SeqKVDevice); ok && sd.PerSequenceKV() {
		return nil
	}
	total := s.nseq * s.maxSeq
	for s.nseq > 1 && s.devCount() > 0 && !s.ld.ReserveKV(total) {
		li := s.relocationVictim()
		if p := s.opt.place; li < 0 || p != nil && p.Strict || !s.cpuListed() {
			return fmt.Errorf("model: the devices cannot hold %d rows of %d positions "+
				"for the blocks placed on them, and no other listed device takes one", s.nseq, s.maxSeq)
		}
		s.relocateToCPU(li)
	}
	return nil
}

// placeOf is the placement WithPlacement names for block li.
func (s *Instance) placeOf(li int) (Place, bool) {
	if p := s.opt.place; p != nil {
		if pl, ok := p.Blocks[li]; ok {
			return pl, true
		}
		if p.Rest != nil {
			return *p.Rest, true
		}
	}
	return Place{}, false
}

// pinnedOnDev reports a device block WithPlacement pinned there.
func (s *Instance) pinnedOnDev(li int) bool {
	pl, ok := s.placeOf(li)
	return ok && pl.Pin && !onCPU(pl.On) && s.devAt(li)
}

// finishPlacement checks a strict placement against where every named block
// and the head landed, undoing all of it when one missed, and pins what the
// map pins.
func (s *Instance) finishPlacement() error {
	p := s.opt.place
	if p == nil || s.ld == nil {
		return nil
	}
	if err := s.placeErr; err != nil {
		s.placeErr = nil
		s.SetGPULayers(0)
		return err
	}
	where, _ := s.ld.(nn.NamedDevice)
	on := func(li int) string {
		if !s.devAt(li) {
			return nn.CPU
		}
		if where != nil {
			if n, ok := where.DeviceOf(li); ok {
				return n
			}
		}
		return "a device"
	}
	if p.Strict {
		for li := range p.Blocks {
			if li < 0 || li >= len(s.m.Layers) {
				s.SetGPULayers(0)
				return fmt.Errorf("model: the placement names block %d of %d", li, len(s.m.Layers))
			}
		}
		for li := s.lo; li < s.hi; li++ {
			pl, named := s.placeOf(li)
			if !named {
				continue
			}
			if got := on(li); got != nn.DeviceName(pl.On) {
				s.SetGPULayers(0)
				return fmt.Errorf("model: strict placement: block %d is on %s, not %s", li, got, pl.On)
			}
		}
		if p.Head != nil {
			got := nn.CPU
			if s.head != nil {
				got = "a device"
				if hn, ok := s.ld.(nn.NamedDevice); ok {
					if n, ok := hn.HeadDeviceName(); ok {
						got = n
					}
				}
			}
			if got != nn.DeviceName(p.Head.On) {
				s.SetGPULayers(0)
				return fmt.Errorf("model: strict placement: the head is on %s, not %s", got, p.Head.On)
			}
		}
	}
	// The device's pin is a residency pin -- never a victim -- which a
	// streamed block cannot have; its Pin keeps it on its device through this
	// instance's moves instead (pinnedOnDev).
	pin, _ := s.ld.(nn.NamedDevice)
	for li := s.lo; li < s.hi; li++ {
		if pl, named := s.placeOf(li); named && pl.Pin && !pl.Stream && !onCPU(pl.On) && s.devAt(li) && pin != nil {
			pin.Pin(li, true)
		}
	}
	return nil
}

// seamCrossBytes is one host/device crossing expressed as host-read bytes
// (crossing time times host read rate, over the share of a byte's cost that
// moving it removes), measured on one host. It is a constant, not a probe,
// and only breaks ties between placements of similar value; measure it before
// giving it more work (see docs/engineering-history/placement.md).
const seamCrossBytes = 2 << 20

// placeHead puts the vocabulary projection on the device, giving up blocks to
// make room for it.
//
// It outranks a block by reads per resident byte: the head is read in full
// every token, while a mixture block's expert banks are
// read a fraction of the time. Releasing from the end keeps the placed blocks a
// contiguous prefix, and a trailing block is the cheapest to give up.
func (s *Instance) placeHead() {
	// WithHeadPlacement(false) leaves the projection on the host -- the other
	// arm of the placement comparison, and the way to bisect a divergence to it.
	if s.ld == nil || s.head != nil || !s.opt.headPlace {
		return
	}
	// A prediction block's own head is in the block's page.
	if s.outW.Data == nil && s.hi > s.lo && s.layers[0].Mtp != nil {
		if _, err := s.pageIn(s.lo); err != nil {
			s.placeErr = err
			return
		}
	}
	if s.outW.Data == nil {
		return
	}
	// The placement's head, when it names one.
	if p := s.opt.place; p != nil && p.Head != nil {
		if onCPU(p.Head.On) {
			return
		}
		if s.placeHeadOn(p.Head.On) || p.Strict {
			return
		}
	}
	if s.gpuLayers <= s.lo {
		return
	}
	h := s.newHead()
	// The trade is priced in bytes a token reads: releasing a block returns
	// its per-token read to the host, placing the projection takes the
	// projection's off it, and the added seam costs seamCrossBytes. So blocks
	// are released only while
	//
	//	headRead > sum(blockRead) + seamCrossBytes
	//
	// On a mixture that is several blocks; on a dense model usually none. The
	// budget gates releasing, not trying: a head that already fits is free.
	var budget uint64
	if hb := s.m.HeadReadBytes(); hb > seamCrossBytes {
		budget = hb - seamCrossBytes
	}
	// What is released is restored when the head still does not fit: the
	// budget is in bytes read, while the constraint is VRAM space, and the
	// loop cannot tell them apart. PrepHead is asked rather than the space
	// predicted, because the device's accounting has slack and fragmentation.
	was := s.gpuLayers
	var streamed []int
	// A block given back mid-sequence takes its history home first, and one
	// restored takes it up again, as SetDeviceLayers' placement does:
	// releasing first frees the only copy of a linear block's summary and an
	// attention block's keys (TestDecisionRelocatesMidPrompt under a budget
	// the head does not fit).
	restore := func() {
		if s.gpuLayers >= was {
			return
		}
		from := s.gpuLayers
		s.offerRange(from, was)
		if s.pos == 0 {
			return
		}
		for _, li := range s.placedFrom(from) {
			if !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
				s.shrinkOnDev(li)
				s.ld.ReleaseLayers(li, s.hi)
				return
			}
		}
	}
	for {
		if s.ld.PrepHead(h) {
			s.head, s.headReady = h, h
			return
		}
		// With every block on the card the room is bought by streaming the
		// last resident mixture blocks instead: each stays on the card, its
		// routed experts sent per token.
		if s.gpuLayers >= s.hi {
			if li, ok := s.streamForHead(&budget); ok {
				streamed = append(streamed, li)
				continue
			}
			s.residentAgain(streamed)
			return
		}
		// Only buy room when the device is already partial. With every block
		// resident the head is pure upside and a refusal means the card is
		// simply full; giving up a block there would trade a whole block's
		// bytes for the head's and add a seam that does not exist.
		// A block given up for the head runs on the CPU, which only a caller
		// that listed it lets the engine choose.
		if s.gpuLayers >= s.hi || s.gpuLayers == s.lo+1 || !s.cpuListed() {
			restore()
			return
		}
		cost := s.m.BlockReadBytes(s.gpuLayers - 1)
		if cost > budget {
			restore()
			return // the next block is worth more than what is left to gain
		}
		// A pinned block is not room to buy.
		if s.pinnedOnDev(s.gpuLayers - 1) {
			restore()
			return
		}
		li := s.gpuLayers - 1
		if s.pos > 0 && (!s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false)) {
			restore()
			return
		}
		budget -= cost
		s.shrinkOnDev(li)
		s.ld.ReleaseLayers(s.gpuLayers, s.gpuLayers+1)
	}
}

// streamForHead streams the last resident mixture block on the card for the
// head (nn.AutoStreamer): its base stays, its routed experts are sent per
// token, while what a token reads of them costs less than what is left of
// budget, the head's read at home (placeHead). It reports the block, or false
// where none is left to stream or the device takes it neither way.
func (s *Instance) streamForHead(budget *uint64) (int, bool) {
	// Asked of the shared tier, the block offered through this instance's view,
	// as placement does (offerRange).
	as, ok := s.ldCand.(nn.AutoStreamer)
	if !ok || s.pos > 0 {
		return 0, false
	}
	c := s.c
	for li := s.hi - 1; li >= s.lo; li-- {
		if !s.devAt(li) || !c.MoEAt(li) || s.streamedAt(li) || s.pinnedOnDev(li) {
			continue
		}
		total, bank := s.m.BlockBytes(li)
		cost := s.m.ExpertReadBytes(li)
		if bank == 0 || cost > *budget {
			return 0, false
		}
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		if as.AutoStream(li, total, bank, c.NExpert, c.NExpertUsed, c.NLayer, s.m.MeanBase()) && s.offerAgain(li) {
			s.autoStreamed++
			s.markStreamed(li, true)
			*budget -= cost
			return li, true
		}
		s.unstream(li)
		return 0, false
	}
	return 0, false
}

// residentAgain puts the blocks streamForHead streamed back resident, the
// head not having fitted.
func (s *Instance) residentAgain(streamed []int) {
	for _, li := range slices.Backward(streamed) {
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		s.autoStreamed--
		s.unstream(li)
	}
}

// unstream takes block li's stream mark off and offers it resident.
func (s *Instance) unstream(li int) {
	if ep, ok := s.ldCand.(nn.ExpertPlacer); ok {
		ep.PlaceExperts(li, "")
	}
	s.markStreamed(li, false)
	s.offerAgain(li)
}

// offerAgain offers block li to the device as placement does, and marks it
// placed where the device takes it.
func (s *Instance) offerAgain(li int) bool {
	plan, w, err := s.blockOffer(li)
	if err != nil {
		return false
	}
	took, err := s.prepOffered(s.ld, li, &plan, &w)
	if err != nil || !took {
		return false
	}
	s.markOnDev(li)
	return true
}

// markStreamed records whether placement streamed block li.
func (s *Instance) markStreamed(li int, on bool) {
	if s.streamedOn == nil {
		s.streamedOn = make([]bool, len(s.m.Layers))
	}
	s.streamedOn[li] = on
}

// streamedAt says placement streamed block li.
func (s *Instance) streamedAt(li int) bool { return li < len(s.streamedOn) && s.streamedOn[li] }

// newHead is the output projection as a device takes it.
func (s *Instance) newHead() *nn.Head {
	c := s.c
	f := s.m.FrontGraphs()
	h := &nn.Head{Model: s.m.Id, File: s.m.Container, Graph: f.CpuHead, HiddenGraph: f.CpuHidden, Sample: f.CpuSample, SampleWalk: f.CpuWalk, PickGraph: f.Pick, PickHead: f.PickHead, Embed: f.Embed, Norm: s.outNorm, W: model.Wt(*s.outW), Logits: s.logits,
		Softcap: c.FinalSoftcap, Embeds: c.TiedEmbd, EmbdScale: float32(c.EmbdScale),
		Stages: s.m.HeadStages(), Bias: s.m.OutB, NoCPU: !s.cpuListed()}
	if !c.TiedEmbd && s.dense.Embd.E != nil {
		h.TokEmbd = model.Wt(s.dense.Embd)
	}
	if s.dense.PosEmbd.E != nil {
		h.PosEmbd = model.Wt(s.dense.PosEmbd)
	}
	h.Streams, h.Collapse = s.streamsPart(f.Embed), s.streamsPart(f.Collapse)
	// The final softcap rides the head: a tier that takes it applies the cap on
	// the device (nn.Head.Softcap), which is what lets a gemma2 token end in
	// the device argmax instead of a 256,000-float readback.
	// A LayerNorm head (C6) carries its bias, zeros where the file has none.
	// The head's own bias (phi-2's output.bias) and the logit scale are
	// stages of the head's graph, which the device applies with the rest
	// (nn.Head.Applied).
	if c.LayerNorm {
		h.NormB = s.m.LnBias(s.m.OutNormB)
	}
	return h
}

// placeHeadOn puts the projection on the named device, as WithPlacement asks.
func (s *Instance) placeHeadOn(name string) bool {
	po, ok := s.ld.(nn.NamedDevice)
	if !ok {
		s.placeErr = fmt.Errorf("model: the placement names %q for the head, and this "+
			"device cannot place it by name", name)
		return false
	}
	h := s.newHead()
	took, err := po.PrepHeadOn(name, h)
	if err != nil {
		s.placeErr = err
		return false
	}
	if took {
		s.head, s.headReady = h, h
	}
	return took
}

// layerWeightsAt gathers block li's weights as a device wants them, from l,
// a copy of the block bound at a device's frames, and banks, its routed banks
// over the same frames. It sets no callback -- the caller does, because they
// call back into here. It is shared by the offer and by every device
// page-in, which must agree.
func (s *Instance) layerWeightsAt(li int, l *model.Layer, banks *[3]nn.Weight) nn.LayerWeights {
	qn, kn := s.qkNorms(li)
	w := nn.LayerWeights{
		Vectors:  l.Vecs,
		AttnNorm: l.AttnNorm, FFNNorm: l.FfnNorm,
		QNorm: qn, KNorm: kn,
		Wq: model.Wt(l.Wq), Wk: model.Wt(l.Wk), Wv: model.Wt(l.Wv), Wo: model.Wt(l.Wo),
		// For a MoE file these three are the banks -- l.gate is the 3-D
		// ffn_gate_exps and l.experts[e] are views into it, so
		// nothing is copied and the device indexes the same mapping.
		Gate: banks[0], Up: banks[1], Down: banks[2],
	}
	// A LayerNorm always hands the device a bias -- zeros where the file has
	// none (command-r) -- because the apply kernel takes one.
	if s.c.LayerNorm {
		w.AttnNormB = s.m.LnBias(l.AttnNormB)
		if l.FfnNorm != nil {
			w.FFNNormB = s.m.LnBias(l.FfnNormB)
		}
	}
	if s.c.MoE() {
		w.Router = model.Wt(l.Router)
		if l.ShGate.Rows != 0 || l.ShUp.Rows != 0 {
			w.ShGate, w.ShUp, w.ShDown = model.Wt(l.ShGate), model.Wt(l.ShUp), model.Wt(l.ShDown)
		}
	}
	if s.c.PLEDim != 0 {
	}
	if s.c.LayerKind(li).Recurrent() {
		w.SSM = nn.SSMWeights{
			Gate: model.Wt(l.SsmGate), BA: model.Wt(l.SsmBA), Out: model.Wt(l.SsmOut), Conv1d: l.SsmConv1d,
			FA: model.Wt(l.SsmFA), FB: model.Wt(l.SsmFB), GA: model.Wt(l.SsmGA), GB: model.Wt(l.SsmGB),
		}
		if s.m.Cfg.Scan() {
			w.SSM.FA, w.SSM.FB, w.SSM.GA, w.SSM.GB = model.Wt(l.SsmXDt), model.Wt(l.SsmDtProj), model.Wt(l.SsmXB), model.Wt(l.SsmXC)
		}
	}
	// MLA's five matrices. Wk and Wv stay zero-valued (MLA has no k or v
	// projection; the tier skips a 0x0 weight). The per-head views are not
	// carried: wkb and wvb are the banks, indexed by head on the device as an
	// expert bank is by expert.
	if s.c.MLA() {
		w.Wqa, w.Wqb = model.Wt(l.Wqa), model.Wt(l.Wqb)
		w.Wkva, w.Wkb, w.Wvb = model.Wt(l.Wkva), model.Wt(l.Wkb), model.Wt(l.Wvb)
	}
	w.Extra = extraWeights(l)
	if k := s.c.LayerKind(li); k.Recurrent() && !k.Attends() {
		// A linear block has none of these; leaving the attention four set
		// would offer a device four matrices whose shapes do not match the
		// plan it was handed. (planFor clears AttnOutGate for the same
		// reason: s.plan sets it on every block of a qwen3next.)
		w.Wk, w.Wv, w.Wo = nn.Weight{}, nn.Weight{}, nn.Weight{}
		w.QNorm, w.KNorm = nil, nil
	}
	return w
}

// extraWeights is every matrix l stores beyond the ones a device is handed
// by name, by role, in the order the block was read.
func extraWeights(l *model.Layer) []nn.RoleWeight {
	named := []*model.Tensor{&l.Wq, &l.Wk, &l.Wv, &l.Wo, &l.Gate, &l.Up, &l.Down, &l.Router,
		&l.ShGate, &l.ShUp, &l.ShDown, &l.SsmGate, &l.SsmBA, &l.SsmOut,
		&l.SsmFA, &l.SsmFB, &l.SsmGA, &l.SsmGB, &l.SsmXDt, &l.SsmDtProj, &l.SsmXB, &l.SsmXC,
		&l.Wqa, &l.Wqb, &l.Wkva, &l.Wkb, &l.Wvb}
	var out []nn.RoleWeight
	for _, m := range l.Mats {
		if slices.Contains(named, m.T) {
			continue
		}
		out = append(out, nn.RoleWeight{Role: m.Role, W: model.Wt(*m.T)})
	}
	return out
}

// offer is block li as a card is offered it: a copy of the block of its
// own, bound at the frames of the card it is offered to (Bind) and read into
// them (Ensure), so the card reads the file into its memory, not the CPU
// device's, and no binding the CPU device runs moves. The card serialises
// its calls on one block; the copy is taken from the instance's under bind.
type offer struct {
	s      *Instance
	li     int
	l      model.Layer
	cloned bool
	// banks is the routed banks' Weights over bankFr (model.Model.BankWeight),
	// built at the first gather over those frames and kept.
	banks  [3]nn.Weight
	bankFr *jlm.Frames
}

// offerWeights is block li's weights as a card is offered them: the
// callbacks of an offer of its own. bind false leaves Bind out, for a block
// the card already holds and reads nothing of.
func (s *Instance) offerWeights(li int, bind bool) nn.LayerWeights {
	o := &offer{s: s, li: li}
	w := nn.LayerWeights{Ensure: o.ensure, EnsureExperts: o.ensureSelected,
		PrefetchExperts: o.prefetch, SpeculateExperts: o.speculate, ExpertUses: s.hot.layer(li)}
	if bind {
		w.Bind = o.bind
	}
	return w
}

func (o *offer) layer() *model.Layer {
	if !o.cloned {
		o.s.bind.Lock()
		o.s.layer(o.li).CloneTo(&o.l)
		o.s.bind.Unlock()
		o.cloned = true
	}
	return &o.l
}

// gather points dst at the copy's spans in fr, keeping dst's callbacks: a
// callback missed here is cleared by the first Ensure (a missed
// PrefetchExperts disables the streamed read/upload pipeline).
func (o *offer) gather(dst *nn.LayerWeights, fr *jlm.Frames) {
	if o.bankFr != fr {
		m, l := o.s.m, &o.l
		o.banks = [3]nn.Weight{m.BankWeight(fr, l.Gate), m.BankWeight(fr, l.Up), m.BankWeight(fr, l.Down)}
		o.bankFr = fr
	}
	w := o.s.layerWeightsAt(o.li, &o.l, &o.banks)
	w.Bind, w.Ensure, w.EnsureExperts = dst.Bind, dst.Ensure, dst.EnsureExperts
	w.PrefetchExperts, w.SpeculateExperts = dst.PrefetchExperts, dst.SpeculateExperts
	w.ExpertUses = dst.ExpertUses
	*dst = w
}

// bind is nn.LayerWeights.Bind: the block's frame in fr claimed, the copy
// bound at it, nothing read.
func (o *offer) bind(dst *nn.LayerWeights, fr *jlm.Frames) error {
	if err := o.s.m.BindBlock(fr, o.li, o.layer()); err != nil {
		return err
	}
	o.gather(dst, fr)
	return nil
}

// ensure is nn.LayerWeights.Ensure: the block read into fr, its routed bank
// with it (a card is handed the whole bank and cannot come back for an
// expert), and dst re-pointed at where the bytes are now -- a frame is
// reused, so slices kept from admission would read whichever block landed
// in it last.
func (o *offer) ensure(dst *nn.LayerWeights, fr *jlm.Frames) error {
	s := o.s
	if err := s.m.PageIn(fr, o.li, o.layer()); err != nil {
		return err
	}
	if s.c.MoE() {
		if err := s.m.EnsureBlockExperts(fr, o.li); err != nil {
			return err
		}
	}
	o.gather(dst, fr)
	return nil
}

// ensureSelected is ensure narrowed to the routed sheets sel: a placed
// block's own weights are on the card, so only they are read
// (model.Model.PageInSelected). See nn.LayerWeights.EnsureExperts.
func (o *offer) ensureSelected(dst *nn.LayerWeights, sel []uint32, fr *jlm.Frames) error {
	s := o.s
	var err error
	if s.c.MoE() {
		err = s.m.PageInSelected(fr, o.li, sel, o.layer())
	} else {
		err = s.m.PageIn(fr, o.li, o.layer())
	}
	if err != nil {
		return err
	}
	o.gather(dst, fr)
	return nil
}

// prefetch is ensureSelected's read and nothing else, for a card that wants
// the next group of sheets on their way while it uploads the last. It binds
// nothing, so it is safe beside the card's use of the copy's spans.
func (o *offer) prefetch(sel []uint32, fr *jlm.Frames) error {
	if !o.s.c.MoE() {
		return nil
	}
	return o.s.m.EnsureBlockSelected(fr, o.li, sel, false)
}

// speculate is the guess half of the read service for the block: the sheets
// the previous block predicted, read in the background into fr and never
// waited for (jlm.Frames.Speculate).
func (o *offer) speculate(sel []uint32, fr *jlm.Frames) {
	if o.s.c.MoE() {
		o.s.m.SpeculateExperts(fr, o.li, sel)
	}
}

// offerRange offers blocks [lo, hi) to the device in order. A declined block
// stays on the host and the rest are still offered (placement is a set; see
// onDev); the seam is the prefix of what was accepted. A tier that cannot run a
// block says so by name through DeclinePlan.
func (s *Instance) offerRange(lo, hi int) {
	ld := s.ldCand
	if ld == nil {
		return
	}
	// Offers go through this instance's own view (gpu.GPU.Attach), which makes it
	// the device's current session: an offer is where a block takes the session's
	// history (KV pages, a linear block's recurrent state), filed under whichever
	// session is current. Offered through the shared tier, a block took it for
	// the last session to make any call -- with two instances placing at once, the
	// other one -- and this instance's first token found no recurrent state and
	// demoted every block to the host.
	adm := s.ld
	if adm == nil {
		adm = ld
	}
	// Each placed block gives its host pages back and the next block's land
	// in the same frames; once the range is offered nothing will take the rest,
	// so they go back to the OS (jlm.File.TrimFree).
	defer s.fr.TrimFree()
	// A group the device took part of (a decline mid-group) comes home whole.
	defer s.fixKVGroups()
	c := s.c
	for li := lo; li < hi; li++ {
		// A block already on a device keeps its place: offering it again
		// would read its page for nothing and prepare it a second time.
		if s.devAt(li) {
			continue
		}
		// A block of a KV-sharing group is offered only with the rest of its
		// group: a source and its readers share one history, so they are
		// placed together or not at all (fixKVGroups).
		if !s.kvGroupOffered(li, lo, hi) {
			continue
		}
		// A block the placement keeps on the host is not offered: while the
		// placement is applied, and afterwards too when it is pinned.
		place, named := s.placeOf(li)
		if named && onCPU(place.On) && (s.placing || place.Pin) {
			continue
		}
		// The block is read here, for the device, and its host page released
		// once it is on the card: one block in flight, not the whole model.
		// Where the block's routed experts run, when a placement or the caller
		// says: set before the size check, so a block placed with its experts
		// off the card is not refused for a bank it will not hold.
		if ep, ok := ld.(nn.ExpertPlacer); ok && c.MoEAt(li) {
			where := s.opt.experts
			if named && place.Experts != "" {
				where = place.Experts
			}
			if where != "" {
				ep.PlaceExperts(li, where)
			}
		}
		// The same for a size no device can hold: a mixture block with its
		// bank can be bigger than a card, and reading it to learn that cost
		// minutes a block on a model that size.
		if d, ok := ld.(nn.SizeDecliner); ok && !(named && place.Stream) {
			total, bank := s.m.BlockBytes(li)
			if why := d.DeclineSize(li, total, bank); why != "" {
				// A mixture block too big for any card resident is streamed
				// there instead when the device offers it: the base on the
				// card, the routed experts sent per token. A placement that
				// keeps the block home is honoured above, and the device can
				// be told not to (gpu.Config.NoAutoStream).
				as, ok := ld.(nn.AutoStreamer)
				if !ok || bank == 0 || !as.AutoStream(li, total, bank, c.NExpert, c.NExpertUsed, c.NLayer, s.m.MeanBase()) {
					s.noteDecline(why)
					continue
				}
				s.autoStreamed++
				s.markStreamed(li, true)
			}
		}
		// A block whose experts are separate tensors has no bank to offer; say
		// why rather than let the tier report zero-value weights. See
		// convert.stackGGUFExperts.
		if s.m.Layers[li].LegacyExperts {
			s.noteDecline("the block's experts are separate tensors rather than a bank " +
				"(a container converted before per-expert GGUFs were stacked, or " +
				"experts of mixed types); re-convert it to place this block")
			continue
		}
		// A block another session placed is shared: the device keeps the
		// weights and reads none of these, and the host page may have been
		// given back after the first upload (nn.HostPageReleaser). Reading it
		// again cost every new session the whole model from disk -- most of a
		// 1B model's time to first token on a 4 GB card (placement.md,
		// "Measurements once cited in engine/model's comments").
		if bh, ok := ld.(nn.BlockHolder); ok && !named && bh.HoldsBlock(li) && s.heldOnOffer(adm, li) {
			w := s.offerWeights(li, false)
			plan := *s.planFor(li)
			took, err := s.prepOffered(adm, li, &plan, &w)
			if err != nil {
				s.placeErr = err
				return
			}
			if took {
				s.markOnDev(li)
				continue
			}
		}
		plan, w, err := s.blockOffer(li)
		if err != nil {
			return
		}
		took := false
		if named && !onCPU(place.On) && s.placing {
			po, ok := adm.(nn.NamedDevice)
			if !ok {
				s.placeErr = fmt.Errorf("model: the placement names %q for block %d, and this "+
					"device cannot place a block by name", place.On, li)
				return
			}
			// Marked before the offer: admission is where a streamed block
			// that does not fit pages others out instead of being declined.
			po.Stream(li, place.Stream)
			var err error
			if took, err = po.PrepLayerOn(place.On, li, &plan, &w); err != nil {
				s.placeErr = err
				return
			}
			if !took && !s.opt.place.Strict {
				s.noteDecline(fmt.Sprintf("%s declined block %d, which the placement names; "+
					"the engine placed it instead", place.On, li))
				took = adm.PrepLayer(li, &plan, &w)
			}
		} else if took, err = s.prepOffered(adm, li, &plan, &w); err != nil {
			s.placeErr = err
			return
		}
		// A mixture block refused for room on every card goes on streamed,
		// its base on a card and its experts off it, rather than to the host:
		// gpt-oss-20b on a 4 GB card held 2 of 24 blocks resident and decoded
		// at 4.84 tok/s, and with all 24 streamed and their experts on the
		// host at 9.56 (placement.md 16c-3). The stream trial measures it
		// against the host like any auto-streamed placement.
		if !took && c.MoEAt(li) && !(named && onCPU(place.On)) {
			if as, ok := ld.(nn.AutoStreamer); ok {
				total, bank := s.m.BlockBytes(li)
				if bank > 0 && as.AutoStream(li, total, bank, c.NExpert, c.NExpertUsed, c.NLayer, s.m.MeanBase()) {
					if took, err = s.prepOffered(adm, li, &plan, &w); err != nil {
						s.placeErr = err
						return
					}
					if took {
						s.autoStreamed++
						s.markStreamed(li, true)
					} else if ep, ok := ld.(nn.ExpertPlacer); ok && !(named && place.Experts != "") && s.opt.experts == "" {
						// It cannot stream either (a float bank with no host
						// side, Gemma 4's dense MLP): the mark goes, or the block
						// could never come back resident once there is room.
						ep.PlaceExperts(li, "")
					}
				}
			}
		}
		if !took {
			// Refused after seeing the weights. The device's own reason when it
			// gives one -- a kernel gap reported as the budget sends a caller
			// shopping for memory it does not need -- and the likeliest one
			// when it does not.
			why := "usually the budget (a placement that streams it, Place.Stream or " +
				"-placement N=DEV~, swaps it through the device instead of leaving it " +
				"on the host); a refusal the PLAN can answer is reported by name above"
			if er, ok := ld.(nn.ErrReporter); ok && er.Err() != "" {
				why = er.Err()
			}
			s.noteDecline("the device refused the block after reading its weights: " + why)
			// Continue, not return: a decline says nothing about the next
			// block.
			continue
		}
		s.markOnDev(li)
		s.releasePlacedPage(ld, li)
	}
}

// prepOffered offers block li to the device MoveBlocks names (offerOn), or to
// the tier's choice when it names none.
func (s *Instance) prepOffered(adm nn.LayerDevice, li int, plan *nn.LayerPlan, w *nn.LayerWeights) (bool, error) {
	if s.offerOn == "" {
		return adm.PrepLayer(li, plan, w), nil
	}
	po, ok := adm.(nn.NamedDevice)
	if !ok {
		return false, fmt.Errorf("model: block %d is moved to %q by name, and this device "+
			"cannot place a block by name", li, s.offerOn)
	}
	return po.PrepLayerOn(s.offerOn, li, plan, w)
}

// heldOnOffer reports whether a block another session holds is where offerOn
// asks for it. A held block is attached where it is, so one held elsewhere is
// offered to offerOn whole, and the tier says why it cannot move.
func (s *Instance) heldOnOffer(adm nn.LayerDevice, li int) bool {
	if s.offerOn == "" {
		return true
	}
	nd, ok := adm.(nn.NamedDevice)
	if !ok {
		return false
	}
	at, ok := nd.DeviceOf(li)
	return ok && at == nn.DeviceName(s.offerOn)
}

// releasePlacedPage gives back the host page of block li once a device holds
// it, or tells the devices to drop their lone-matvec copies of a page that
// stays. offerRange and MoveBlocks share it.
func (s *Instance) releasePlacedPage(ld nn.LayerDevice, li int) {
	l := s.layer(li)
	// The block is on the card, so the host gives up its page; pageIn
	// re-reads it if the seam later moves the block back. Asked, not
	// assumed: a unified device's buffers alias host memory.
	// A prediction block keeps its page: its eh_proj runs on the host every
	// draft step (Speculator.draftRows), and re-reading the page for it
	// would cost a block of I/O a step.
	if r, ok := ld.(nn.HostPageReleaser); ok && r.ReleasesHostPage(li) && l.Mtp == nil {
		// Under bind, as pageIn binds: a move's staging may be binding this
		// block's spans while the instance places it. The page itself stays
		// while another instance runs the block from host memory (Keep): it
		// faults the block in once and then reads its spans unlocked through
		// every matvec of the block.
		keep := false
		if eh, ok := ld.(nn.ExpertHolder); ok {
			keep = !eh.HoldsExperts(li)
		}
		s.bind.Lock()
		model.ReleaseLayer(s.fr, li, l, keep)
		s.bind.Unlock()
	}
}

// blockOffer reads block li's page and gathers what a device is offered for
// it: the block's plan and its weights with every callback. offerRange and
// MoveBlocks share it, so a block moved between devices is offered exactly as
// a placement offers it.
func (s *Instance) blockOffer(li int) (nn.LayerPlan, nn.LayerWeights, error) {
	w, err := s.blockWeights(li)
	if err != nil {
		return nn.LayerPlan{}, nn.LayerWeights{}, err
	}
	// The block is offered and the device decides (PrepLayer). Wq carries
	// the fused q|k|v projection for a linear block. planFor, not s.plan:
	// it adds what this block is (its kind, sinks, mixture biases).
	return *s.planFor(li), w, nil
}

// blockWeights is blockOffer's weights: an offer of block li with every
// callback set, the card binding it at its own frames before it decides
// (nn.LayerWeights.Bind) and reading them once it admits. Nothing is read
// here: the bytes are the card's to read, into its memory. It reads the
// model and nothing of the instance's placement, so a move's staging runs it
// beside the instance's own tokens (PendingMove).
func (s *Instance) blockWeights(li int) (nn.LayerWeights, error) {
	if li < s.lo || li >= s.hi {
		return nn.LayerWeights{}, fmt.Errorf("engine: block %d is not one of this instance's [%d, %d)", li, s.lo, s.hi)
	}
	return s.offerWeights(li, true), nil
}

// planExtra is what a placement of blocks [0,hi) must leave room for besides
// their weights (nn.BlockPlanner): the history this instance's context will hold
// on its attention blocks, counted as float32 K and V (MLA's latent row), and
// the output projection. A recurrent block's state is fixed and small.
func (s *Instance) planExtra(hi int) uint64 {
	c := s.c
	hdK, hdV := c.AttnWidths()
	perPos := uint64(c.NKVHead*(hdK+hdV)) * 4
	if c.KVLoraRank != 0 {
		perPos = uint64(c.KVDim()) * 4
	}
	var n uint64
	for li := 0; li < hi; li++ {
		if c.LayerKind(li).Attends() {
			n++
		}
	}
	return n*perPos*uint64(s.maxSeq)*uint64(max(s.nseq, 1)) + s.m.HeadReadBytes()
}

// planFor is the block's plan with no weights in it: s.plan adjusted for what
// this block is. Nothing here reads the block's page, so nn.PlanDecliner can
// answer before the page-in.
func (s *Instance) planFor(li int) *nn.LayerPlan {
	c := s.c
	plan := s.plan
	// The sinks and the mixture biases are resident vectors read at build(),
	// not page bytes, so asking about them here still needs no page-in.
	l := &s.m.Layers[li]
	plan.AttnSinks = l.Vector(jlm.RoleAttnSinks) != nil
	plan.MoEBias = l.Vector(jlm.RoleRouterBias) != nil || l.Vector(jlm.RoleExpGateBias) != nil || l.Vector(jlm.RoleExpUpBias) != nil || l.Vector(jlm.RoleExpDownBias) != nil
	// An ungated FFN has no gate tensor, which build() recorded without a
	// page-in: the entry is bound at load whether or not the block is resident.
	plan.UngatedFFN = l.Gate.E == nil && (l.Router.E == nil || l.UngatedExp)
	plan.NoFFN = l.NoFFN
	// AttnScale is carried for every architecture: it includes YaRN's
	// magnitude correction (DeepSeek), and under MLA HeadDim is the wrong
	// width to derive it from. MLA's geometry is here so a tier can decline.
	plan.AttnScale = c.AttnScale
	// The block's own attention geometry (Gemma 4's sliding layers differ
	// from its global ones), its v (no projection of its own) and its output
	// scalar.
	plan.HeadDim, plan.NKVHead, plan.NRot = c.HeadDimAt(li), c.NKVHeadAt(li), c.NRotAt(li)
	// The block as ops, from the container: what a device serving it whole
	// forms (front.go), nil where the front end states none.
	plan.Graph = s.m.Composing().BlockGraph(li)
	plan.LocalRope = c.MixedGeom() && c.SWA(li)
	plan.OwnGeom = plan.LocalRope || plan.HeadDim != c.HeadDim || plan.NKVHead != c.NKVHead || plan.NRot != c.NRot
	plan.BidirOff = !c.BidirAt(li)
	plan.VFromK, plan.OutScale = l.VFromK, l.OutScale
	plan.PLEDim, plan.PLEWidth = c.PLEDim, c.SideWidth()
	if c.MLA() {
		plan.KeyW = c.KVDim() - c.KVLoraRank - c.NRot
	} else {
		plan.KeyW = c.KVRowAt(li) - c.KVDimAt(li)
	}
	if c.SelectAt(li) {
		s := c.Selection()
		plan.Select = &nn.SelectPlan{Block: int(s.Block), Keep: int(s.Keep)}
	}
	plan.KVShared, plan.KVSource = c.KVShared(li), c.KVSource(li)
	plan.NFFN, plan.FFNWide = c.NFFNAt(li), c.NFFNAt(li) != c.NFFN
	if c.MLA() {
		plan.KVLoraRank, plan.QLoraRank, plan.HeadDimV = c.KVLoraRank, c.QLoraRank, c.HeadDimV
	}
	// The mixture is a property of the block: DeepSeek's dense lead blocks
	// (Config.MoEAt) must not carry the model-wide expert counts.
	if !c.MoEAt(li) {
		plan.NExpert, plan.NExpertUsed, plan.NFFNExp, plan.NFFNShExp = 0, 0, 0, 0
	}
	// The router's gating shape (V3's sigmoid scoring, selection bias, expert
	// groups and routed scale), which a device must see to run or refuse.
	plan.ExpertSigmoid, plan.ExpertScale = c.ExpertSigmoid, c.ExpertScale
	plan.ExpertGroups, plan.ExpertGroupsUsed = c.NExpertGroup, c.NExpertGroupUsed
	plan.ExpertSelBias = s.m.ExpSelBias() != nil
	if c.LayerKind(li).Recurrent() {
		g := c.Delta()
		plan.Recurrent = nn.RecurrentPlan{
			Conv: g.Conv, Chans: g.Chans, KHeads: g.KHeads, VHeads: g.VHeads,
			KDim: g.KDim, VDim: g.VDim,
			StateLen: g.StateStride(), ConvState: g.ConvStateLen(),
			ChanDecay: c.ChanDecay(), KeyTiled: g.Tiled,
		}
		if g.Ssd {
			plan.Recurrent.SSD, plan.Recurrent.NormGroups = true, c.SSMNormGroups
			plan.Recurrent.NormBeforeGate, plan.Recurrent.NoNorm = c.SSMNormBeforeGate, l.Vector(jlm.RoleSSMNorm) == nil
		}
		// A linear block has no attention, so no out-gate.
		plan.AttnOutGate = false
		plan.Recurrent.WithAttn = c.LayerKind(li).Attends()
		plan.Recurrent.ShortConv = c.ShortConv()
		if g.Scan {
			// No gated norm: the block's output is y*silu(z) straight to ssm_out.
			plan.Recurrent.Scan, plan.Recurrent.Rank, plan.Recurrent.NoNorm = true, g.Rank, true
		}
	}
	plan.LatentNormEps = c.LatentNormEps
	if c.LayerKind(li).Recurrent() {
		plan.Recurrent.TailLen = c.RecTailLen()
	}
	plan.NStreams = c.Streams()
	if w := c.ExpWidth(); c.MoEAt(li) && w != c.NEmbd {
		plan.ExpLatent = w
	}
	if c.NonCausal && (s.m.IsEmbedding() || s.m.Tower != nil && li >= s.m.TowerBase) {
		// A bidirectional decoder (EmbeddingGemma) and a tower: their blocks
		// are non-causal blocks, as an encoder's are -- the sequence (a
		// picture's rows) one call, every row over the call's rows, a sliding
		// layer inside each row's symmetric window (nn.RowWindowDevice;
		// embedstate.go's embedTell). The row windows
		// are the window, so the causal one (SWAWindow) is off: with both, a
		// 600-row sequence read NaN past the causal window's 512 rows.
		plan.NonCausal = true
		plan.Windowed = c.SWA(li)
		plan.SWAWindow = 0
	}
	// What the block is composed from, for a device that composes it so.
	if s.kv != nil && li < len(s.kv.layers) {
		spec := s.blockSpec(li)
		plan.Spec = &spec
	}
	return &plan
}

// markOnDev records that a device accepted block li, and recomputes gpuLayers
// as the contiguous placed prefix (with holes, "li + 1" is not a length).
func (s *Instance) markOnDev(li int) {
	if s.onDev == nil {
		s.onDev = make([]bool, len(s.m.Layers))
	}
	if !s.onDev[li] {
		s.cpuRun(li, -1)
	}
	s.onDev[li] = true
	n := s.lo
	for n < len(s.onDev) && s.onDev[n] {
		n++
	}
	s.gpuLayers = n
}

// unmarkOnDev records that block li went back to the host, and is markOnDev's
// twin: the prefix ends at the first block not placed.
func (s *Instance) unmarkOnDev(li int) {
	if s.onDev[li] {
		s.cpuRun(li, 1)
	}
	s.onDev[li] = false
	s.gpuLayers = min(s.gpuLayers, li)
}

// placedFrom lists the placed blocks at index lo and above.
func (s *Instance) placedFrom(lo int) []int {
	var out []int
	for li := lo; li < len(s.onDev); li++ {
		if s.onDev[li] {
			out = append(out, li)
		}
	}
	return out
}

// clearOnDev forgets the whole placement.
func (s *Instance) clearOnDev() {
	for i := range s.onDev {
		if s.onDev[i] {
			s.cpuRun(i, 1)
		}
		s.onDev[i] = false
	}
	s.gpuLayers = s.lo
	// The census belongs to one placement: -gpu-grow and seamtune re-offer, and
	// counts that survived would report a card refusing more blocks than the
	// model has.
	s.declines = nil
}

// shrinkOnDev gives every block from n up back to the host, and is what a
// caller that writes `s.gpuLayers = n` means: the seam moved down, so
// the blocks above it are no longer placed anywhere.
func (s *Instance) shrinkOnDev(n int) {
	for i := n; i < len(s.onDev); i++ {
		if s.onDev[i] {
			s.cpuRun(i, 1)
		}
		s.onDev[i] = false
	}
	if s.gpuLayers > n {
		s.gpuLayers = n
	}
}

// holdCPURuns keeps the page of every block of [lo, hi): an instance with no
// device attached runs them all on the CPU device, so its card prefix is
// empty.
func (s *Instance) holdCPURuns() {
	s.gpuLayers = s.lo
	s.keep(s.lo, s.hi, 1)
	s.cpuHeld = true
}

// cpuRun moves this instance's keep on block li's page by d, as the block
// leaves (-1) or moves to (+1) the CPU device for it.
func (s *Instance) cpuRun(li, d int) {
	// The device composes the block it takes, or lets one that left go.
	if x := s.cpu.x; x.Holds(li) {
		if d > 0 {
			x.PrepLayer(li, s.planFor(li), nil)
		} else {
			x.Drop(li)
		}
	}
	if s.cpuHeld && li >= s.lo && li < s.hi {
		s.keep(li, li+1, d)
	}
}

// keep moves by d this instance's keep (jlm.File.Keep) on the pages of
// blocks [lo, hi), which it runs from host memory.
func (s *Instance) keep(lo, hi, d int) {
	if s.m.Container == nil {
		return
	}
	for li := lo; li < hi; li++ {
		s.fr.Keep(li, d)
	}
}

// layer is this instance's copy of block li, one of [lo, hi).
func (s *Instance) layer(li int) *model.Layer { return &s.layers[li-s.lo] }

// pageIn reads block li's own weights into its page and binds this
// instance's copy of the block at them (model.Model.PageIn).
func (s *Instance) pageIn(li int) (*model.Layer, error) {
	if li < s.lo || li >= s.hi {
		return nil, fmt.Errorf("engine: block %d is not one of this instance's [%d, %d)", li, s.lo, s.hi)
	}
	l := s.layer(li)
	s.bind.Lock()
	defer s.bind.Unlock()
	return l, s.m.PageIn(s.fr, li, l)
}

// pageInSelected is pageIn of block li for the routed sheets sel alone
// (model.Model.PageInSelected).
func (s *Instance) pageInSelected(li int, sel []uint32) error {
	if li < s.lo || li >= s.hi {
		return fmt.Errorf("engine: block %d is not one of this instance's [%d, %d)", li, s.lo, s.hi)
	}
	s.bind.Lock()
	defer s.bind.Unlock()
	return s.m.PageInSelected(s.fr, li, sel, s.layer(li))
}

// devAt reports whether block li runs on a device.
func (s *Instance) devAt(li int) bool { return li < len(s.onDev) && s.onDev[li] }

// devRuns is the placement as maximal contiguous runs, the unit
// nn.LayerDevice.Layers takes. Each run is one seam crossing (the residual goes
// up and comes back), so alternating tiers is priced rather than refused.
func (s *Instance) devRuns() [][2]int {
	var out [][2]int
	for li := 0; li < len(s.onDev); {
		if !s.onDev[li] {
			li++
			continue
		}
		hi := li
		for hi < len(s.onDev) && s.onDev[hi] {
			hi++
		}
		out = append(out, [2]int{li, hi})
		li = hi
	}
	return out
}

// devCount is how many blocks a device took, which is what a caller asking
// "how much of this model is on the card" means.
func (s *Instance) devCount() int {
	n := 0
	for _, v := range s.onDev {
		if v {
			n++
		}
	}
	return n
}

// prepDeviceRange grows the device's prefix to n blocks. A binary16 history
// travels widened, as SetDeviceLayers' does (migrateKVAs). The offers are
// one placement, ended as an attach's is (nn.Placer): the device forms what
// a prompt and a step of rows read over the blocks it now holds, so no step
// after the move composes.
func (s *Instance) prepDeviceRange(lo, n int) {
	if pl, ok := s.ld.(nn.Placer); ok {
		pl.BeginPlacement()
		defer pl.EndPlacement()
	}
	s.offerRange(lo, n)
	s.placeHead()
}

// embed writes the token's embedding into the residual stream: the input
// rows' graph composed for the CPU (cpuembed.go), its scale, its position
// row and its per-layer inputs with it.
func (s *Instance) embed(token int32) error {
	// On the CPU, unlike a prompt's rows: a device gather here measured
	// slower (one row is usually cache-hot, and the launch sits on the
	// token's critical path).
	e := &s.inScr
	e.tok1[0], e.pos1[0] = token, s.pos
	if why := s.embedRows(s.x, s.ple, e.tok1[:], e.pos1[:]); why != nn.DeclineServes {
		return s.embedDeclined(why)
	}
	s.trace(-1, "embd", s.x)
	return nil
}

// SetDeviceFallback decides what happens when the device faults mid-sequence
// (a driver error, a device lost): move every block and its history to the
// CPU device and finish there (true, the default), or return an error
// (false). A model whose caller did not list the CPU (WithDevices without cpu) takes
// the error either way, and a decline or a kernel that did not build is an
// error always (refusal).
//
// A service would rather serve slowly than fail; a benchmark wants the error,
// since a run silently finished on the CPU produces a meaningless number.
func (s *Instance) SetDeviceFallback(on bool) { s.devFallback = on }

// fallsBack reports whether a device failure moves the blocks to the CPU
// (demoteAll) rather than failing the step.
func (s *Instance) fallsBack() bool { return s.devFallback && s.cpuListed() }

// demoteAll moves every device block and its history to the CPU device.
//
// The history moves before the blocks do: releasing first would free the only
// copy of their keys and values.
func (s *Instance) demoteAll() bool {
	cause := s.devCause()
	// The set, not the prefix: a hybrid's prefix can be zero with blocks
	// placed.
	if s.ld == nil || s.devCount() == 0 {
		return true
	}
	for li := 0; li < len(s.onDev); li++ {
		if !s.onDev[li] {
			continue
		}
		if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
			return false
		}
	}
	s.head, s.headReady = nil, nil
	// Release every run, not one prefix: ReleaseLayers takes a range.
	for _, r := range s.devRuns() {
		s.ld.ReleaseLayers(r[0], r[1])
	}
	s.clearOnDev()
	s.moved = nil
	// Do not walk straight back into the card that just refused. Growing again
	// is a decision for whoever handles the report, not for the next token.
	s.gpuTarget, s.seam = 0, nil
	s.devDemoted++
	s.demotedBy[cause]++
	return true
}

// DeviceDemotions is how many times the device failed and its blocks were moved
// to the host. Non-zero means this session's rate is not comparable with one
// that kept the card.
func (s *Instance) DeviceDemotions() int { return s.devDemoted }

// TailRowChunks is how many few-row prompt chunks this session ran on the
// device as one ragged step of their rows rather than as a batched chunk
// (tailrows.go).
func (s *Instance) TailRowChunks() int { return s.tailRowChunks }

// SetRelocate chooses what happens when the device cannot grow its attention
// history to the next position: hand its highest block to the host and ask
// again, as often as it takes (true, the default), or let the token fail on
// the device and take SetDeviceFallback's answer, which demotes every block
// (false). Moving one block is strictly less host work than moving all of them.
//
// It is asked before the token rather than after a failure, which makes it
// safe on a hybrid (which cannot restart a token; see errRecurRetry): nothing
// has run when a block moves, so its history and summary come home at a token
// boundary.
func (s *Instance) SetRelocate(on bool) { s.relocate = on }

// Relocating reports whether SetRelocate is on.
func (s *Instance) Relocating() bool { return s.relocate }

// Relocations is how many blocks SetRelocate has handed to the host.
func (s *Instance) Relocations() int { return s.devRelocated }

// Spills is how many blocks moved onto a later device because the one holding
// them could not grow its history (gpu.GPU.SpillAfter).
func (s *Instance) Spills() int { return s.devSpilled }

// Reclaims is how many of those the device has since taken back.
func (s *Instance) Reclaims() int { return s.devReclaimed }

// relocateFor is SetRelocate's pre-flight: while the device cannot hold `pos`
// positions of history, a block comes home -- the highest one on the device
// that refused.
//
// one block per ask. The device refuses before it touches anything
// (devTier.regrowKV) and grows to whatever fits before refusing at all
// (devTier.kvFit), so an ask is cheap and a block moves only when not even one
// more page fits.
func (s *Instance) relocateFor(pos int) {
	if s.ld == nil {
		return
	}
	fits := func() bool { return s.ld.ReserveKV(pos) }
	if s.nseq > 1 {
		if sd, ok := s.ld.(nn.SeqKVDevice); ok && sd.PerSequenceKV() {
			// Each row is its own sequence: what has to fit is every row's next
			// position, not the batch's whole context.
			bases, ends := make([]int, s.nseq), make([]int, s.nseq)
			for r := range bases {
				bases[r], ends[r] = r*s.maxSeq, s.bpos[r]+1
			}
			fits = func() bool { return sd.ReserveKVSeqs(bases, ends) }
		} else {
			// A contiguous cache holds every row at r*maxSeq+pos
			// (nn.RowsDevice) and a rows step reserves all of it, whatever one
			// row's position is. The moves below carry every row
			// (migrateKVRows).
			pos = s.nseq * s.maxSeq
		}
	}
	s.relocateUntil(pos, fits)
}

// relocateUntil moves blocks off the device -- to a later card, or home under
// SetRelocate -- until fits says the history the next step writes has room.
// pos is the positions the step needs, for reclaim.
func (s *Instance) relocateUntil(pos int, fits func() bool) {
	// Before and after: a block a relocation takes home may part a KV-sharing
	// group (fixKVGroups).
	s.fixKVGroups()
	defer s.fixKVGroups()
	sp := s.ldSpill
	// The CPU is a destination only where the caller listed it.
	toCPU := s.relocate && s.cpuListed()
	if !toCPU && sp == nil {
		return
	}
	if toCPU {
		s.reclaim(pos)
	}
	defer func() { s.roomSeen = s.roomGen() }() // our own moves are not news
	for s.devCount() > 0 && !fits() {
		// A kernel that did not build is not room, and no block's move gives
		// it one: nothing moves, and the step that needs the kernel meets the
		// failure and returns it (refusal).
		if s.devCause() == nn.FailBuild {
			return
		}
		li := s.relocationVictim()
		if li < 0 {
			return // every device block is pinned: the token takes the refusal
		}
		// The history comes home before the weights go, as in SetGPULayers. A
		// block whose history cannot move stays, and the token takes the
		// device's refusal as it would have without this.
		if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
			return
		}
		// A later device first, whether or not SetRelocate is on: moving the
		// block onto the next card keeps every run contiguous and costs no
		// crossing. See gpu.GPU.SpillAfter.
		spilled := false
		if sp != nil && sp.SpillAfter(li) {
			s.unmarkOnDev(li)
			s.offerRange(li, li+1)
			if spilled = s.devAt(li); !spilled {
				s.markOnDev(li) // no later card took it; it never left
			} else if s.migrateKV(li, s.pos, true) && s.migrateRec(li, true) {
				s.devSpilled++
				continue
			}
		}
		// A block on its new card without its history would attend over
		// nothing, so that one comes home even without SetRelocate.
		if !toCPU && !spilled {
			return
		}
		s.relocateToCPU(li)
	}
}

// relocateToCPU moves block li to the CPU device for this instance and
// remembers it, so that reclaim offers it back once the device reports room.
// The block's history must already be in this session's cache (or never have
// been written).
func (s *Instance) relocateToCPU(li int) {
	s.ld.ReleaseLayers(li, li+1)
	s.unmarkOnDev(li)
	// -gpu-grow would adopt the block straight back next token.
	s.gpuTarget = min(s.gpuTarget, s.gpuLayers)
	if s.moved == nil {
		s.moved = make([]bool, len(s.m.Layers))
	}
	s.moved[li] = true
	s.devRelocated++
}

// reclaim offers the device back the blocks relocation took off it, lowest
// first, when the device reports room it did not have when they left.
//
// Only on a room report, never every token: offering a block the card cannot
// hold reads its page for nothing. The report moves when another session
// detaches, a budget is raised, or a shorter fresh prompt shrinks the history
// (relocateFresh), not when this session's own relocation frees room. A block
// that fits but leaves no room for pos is given straight back.
func (s *Instance) reclaim(pos int) {
	if s.moved == nil || s.roomGen() == s.roomSeen {
		return
	}
	for li, was := range s.moved {
		if !was {
			continue
		}
		// A KV-sharing group comes back whole (kvSpan), as it left.
		// Every block of the span the offer placed carries its history up,
		// moved here or brought home with its group (fixKVGroups).
		lo, hi := s.kvSpan(li)
		was := make([]bool, hi-lo)
		for b := lo; b < hi; b++ {
			was[b-lo] = s.devAt(b)
		}
		s.offerRange(lo, hi)
		if !s.devAt(li) {
			break
		}
		ok := true
		for b := lo; b < hi && ok; b++ {
			if !was[b-lo] && s.devAt(b) {
				ok = s.migrateKV(b, s.pos, true) && s.migrateRec(b, true)
			}
		}
		if !ok || !s.ld.ReserveKV(pos) {
			// The blocks' history is still on the host (MigrateKV copies), so
			// giving them back loses nothing.
			for b := lo; b < hi; b++ {
				if !was[b-lo] && s.devAt(b) {
					s.ld.ReleaseLayers(b, b+1)
					s.unmarkOnDev(b)
				}
			}
			break
		}
		for b := lo; b < hi; b++ {
			if !was[b-lo] && s.devAt(b) {
				s.moved[b] = false
				s.devReclaimed++
			}
		}
	}
	s.roomSeen = s.roomGen()
}

// relocateFresh is what a sequence starting at position zero with a prompt of
// total positions does under SetRelocate: when blocks were relocated, the
// history the previous sequence grew is shrunk to what this one needs, and the
// room goes back to those blocks before the first chunk runs.
//
// Only a prompt that needs less gives room back, so a chat turn (which
// re-prefills the whole conversation) does not bounce blocks.
func (s *Instance) relocateFresh(total int) {
	if !s.relocate || s.ld == nil || s.nseq != 1 || s.pos != 0 || !slices.Contains(s.moved, true) {
		return
	}
	if t, ok := s.ld.(nn.KVTrimmer); ok {
		t.TrimKV(total)
	}
	s.relocateFor(total)
}

// roomGen is the device's room report, or 0 from a device that makes none.
func (s *Instance) roomGen() uint64 {
	if s.ldRoom != nil {
		return s.ldRoom.RoomGen()
	}
	return 0
}

// relocationVictim is the block relocateFor hands to the host: the highest one
// on the device that refused, when the device can say which that was, and the
// highest placed block otherwise.
//
// On several cards the highest block is usually on the wrong one (blocks fill
// the fastest card first). Taking from the card that refused leaves a host
// block between two device runs, which costs no extra crossing: the residual
// already comes home between devices.
func (s *Instance) relocationVictim() int {
	li := -1
	if r, ok := s.ld.(nn.RefusalReporter); ok {
		for _, b := range r.Refused() {
			if s.devAt(b) && !s.pinnedOnDev(b) && b > li {
				li = b
			}
		}
	}
	if li < 0 {
		for li = len(s.onDev) - 1; li >= 0 && (!s.onDev[li] || s.pinnedOnDev(li)); li-- {
		}
	}
	return li
}

// SetHeadOnDevice chooses whether the output norm and the vocabulary projection
// run in the same submission as the blocks, so a paired A/B can alternate the
// two in one process; the device keeps the weights resident either way.
func (s *Instance) SetHeadOnDevice(on bool) {
	if on {
		s.head = s.headReady
		return
	}
	s.head = nil
}

// HeadOnDevice reports whether the device is running the token's tail.
func (s *Instance) HeadOnDevice() bool { return s.head != nil }

// AttnChunk is how many query heads one pool task claims in decode attention,
// and SetAttnChunk sets it. Zero means the shipping default of one.
func (s *Instance) AttnChunk() int {
	if s.attnChunk < 1 {
		return 1
	}
	return s.attnChunk
}

// SetAttnChunk sets how many query heads one pool task claims in decode
// attention (see AttnChunk); below one means the default.
func (s *Instance) SetAttnChunk(n int) { s.attnChunk = n }

// The KV layout and width are per model (modelOpts), set before the instances
// that use them are created, since the attention kernels bake the stride.

// KVIsF16 reports whether this state's cache is binary16. It can be false with
// the width forced to f16 (WithKVF16), when the model's head dimension has no
// generated attention kernel and the f32 fallbacks have to be able to read the
// cache.
func (s *Instance) KVIsF16() bool { return s.kvFmt == cpu.KVF16 }

// attnPairShipped is what ships: share both the K walk and the V walk between
// each pair of query heads. It is neutral at shallow depth and a win where
// attention dominates (see docs/engineering-history/cpu-kernels.md).
// WithAttnPair overrides it per model, so a scoreboard row can attribute it.
const attnPairShipped = 3

// AttnPair selects which paired attention kernels decode uses: 1 the scores, 2
// the accumulation, 3 both. The halves are separable because they walk
// different arrays (K and V) with different register costs. Zero means unset
// and takes the default; a negative value means explicitly none.
func (s *Instance) AttnPair() int {
	switch {
	case s.attnPair == 0:
		return s.opt.attnPair
	case s.attnPair < 0:
		return 0
	default:
		return s.attnPair
	}
}

// AttnPaired is how many head pairs the paired kernels have served. Zero after
// a decode means the paired path was never taken -- the normal answer for an
// odd gqa, and a defect for an even one. A test that does not check it passes
// vacuously on any model whose heads cannot pair.
func (s *Instance) AttnPaired() int64 { return s.attnPaired.Load() }

// SetAttnPair selects this instance's paired attention kernels, as AttnPair
// reads them: 1 the scores, 2 the accumulation, 3 both, 0 the model's
// default, negative none.
func (s *Instance) SetAttnPair(v int) { s.attnPair = v }

// attStride rounds a score row's stride up to a whole vector, so the padding
// the generated softmax writes stays inside the row it belongs to.
func attStride(maxSeq int) int { return (maxSeq + 15) &^ 15 }

// rowsStride is the stride of the many-row score buffers (battf for a prefill
// chunk, bathf for a batch step) when their rows reach keys below reach.
//
// Those buffers hold a row per token or per (row, head), so sized at maxSeq
// they were the largest thing an instance allocated: a 170-token prompt on a
// 131072-position context held 89 MB of scores for keys it could not reach,
// and a server keeps an instance per concurrent request. The reach they are sized
// for grows by doubling, from scoreReachMin, up to maxSeq, so a sequence
// re-sizes them a handful of times in its life and a warm step at a steady
// reach allocates nothing.
func (s *Instance) rowsStride(reach int) int {
	if reach > s.scoreReach {
		r := max(s.scoreReach, scoreReachMin)
		for r < reach {
			r *= 2
		}
		s.scoreReach = min(r, s.maxSeq)
	}
	return attStride(s.scoreReach + s.m.SinkSlot())
}

// scoreReachMin is the first reach rowsStride sizes for: a page's worth of
// positions, past which a short exchange never re-sizes.
const scoreReachMin = 256

// devLogits says the logits the device's head handed back carry every stage
// of the head past the projection (devApplied).
func (s *Instance) devLogits() error {
	var a nn.HeadStage
	if s.head != nil {
		a = s.head.Applied
	}
	return s.devApplied(a)
}

// devApplied says a device that applied the stages a of the head applied
// all of them: the composed head's form is the whole head, its bias, cap
// and scale in the model's order, and a device serves it whole or refuses
// the call (nn.Head.Applied).
func (s *Instance) devApplied(a nn.HeadStage) error {
	if left := s.m.HeadStages() &^ a; left != 0 {
		return fmt.Errorf("model: the device's head left the stages %03b of the head to the caller", left)
	}
	return nil
}

// JIT is this state's generated-code tier: its vision segment runs on it
// (Instance.Vision), so a model has one JIT and one worker pool a session.
// Pool is the CPU device's workers this instance runs on.
func (s *Instance) Pool() *cpu.Pool { return s.cpu.x.Pool() }

// GPULayers is how many blocks a device took, which is the set and not the
// prefix: on a hybrid the two differ and the count is what a caller means.
func (s *Instance) GPULayers() int { return s.devCount() }

// noteDecline records why one block did not go to a device.
func (s *Instance) noteDecline(why string) {
	if s.declines == nil {
		s.declines = map[string]int{}
	}
	s.declines[why]++
}

// DeviceDeclines is how many blocks each device refused, by reason, most
// numerous first. A placement is not explained by the blocks it took: without
// this, a capability gap and a memory gap look identical.
func (s *Instance) DeviceDeclines() []DeviceDecline {
	out := make([]DeviceDecline, 0, len(s.declines))
	for why, n := range s.declines {
		out = append(out, DeviceDecline{Why: why, Blocks: n})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Blocks != out[j].Blocks {
			return out[i].Blocks > out[j].Blocks
		}
		return out[i].Why < out[j].Why
	})
	return out
}

// DeviceDecline is one reason and how many blocks it cost.
type DeviceDecline struct {
	Why    string
	Blocks int
}

// DeviceBlocks names the placed blocks as the contiguous runs the submission
// uses; a count cannot say which ones, and whether the last block is placed
// decides whether the head can ride the same submission.
func (s *Instance) DeviceBlocks() [][2]int { return s.devRuns() }

// SetGPUTarget asks for n blocks on the device eventually, growing a little at
// a time instead of all at once.
//
// The seam walks toward the target one block per token, so the first token
// comes at CPU latency and each later one carries about one block's upload
// (just the upload, with PrewarmGPU). As measured it has not paid: each grown
// block also forces a graph re-capture, and the first-token cost it aimed at
// was not block preparation (see docs/engineering-history/placement.md). It
// may pay with a cold page cache, a slow disk or large blocks.
//
// Zero disables growth. It does not shrink -- SetGPULayers does that,
// immediately, because giving memory back should not be gradual.
func (s *Instance) SetGPUTarget(n int) {
	if n > s.hi {
		n = s.hi
	}
	s.gpuTarget = n
}

// growStep adopts at most one block, called once per token.
//
// One block, not as many as are ready, so no single token carries the whole
// upload.
func (s *Instance) growStep() {
	if s.ld == nil || s.gpuTarget <= s.gpuLayers {
		return
	}
	was := s.gpuLayers
	got := s.SetGPULayers(was + 1)
	if got == was {
		if s.opt.growDebug {
			msg := ""
			if e, ok := s.ld.(nn.ErrReporter); ok {
				msg = e.Err()
			}
			fmt.Fprintf(os.Stderr, "grow: stuck at %d (target %d): %s\n", was, s.gpuTarget, msg)
		}
		s.gpuTarget = was // stop trying; the card refused
	}
}

// PrewarmGPU packs, off the main loop, the weights that growing the device's
// prefix to n blocks would need. It returns a channel closed when the packing
// finishes.
//
// The goroutine packs and the main loop uploads: the tier holds its mutex for
// a whole token and CUDA forbids allocation inside a Session, so nothing that
// touches the device can overlap decoding, while packing (most of the cost)
// can. A later SetGPULayers finds the packs waiting and only uploads. It is
// optional.
func (s *Instance) PrewarmGPU(n int) <-chan struct{} {
	done := make(chan struct{})
	if s.ldCand == nil || n <= s.gpuLayers {
		close(done)
		return done
	}
	if n > s.hi {
		n = s.hi
	}
	lo, c := s.gpuLayers, s.c
	go func() {
		defer close(done)
		for li := lo; li < n; li++ {
			l := s.layer(li)
			qn, kn := s.qkNorms(li)
			w := nn.LayerWeights{
				Vectors:  l.Vecs,
				AttnNorm: l.AttnNorm, FFNNorm: l.FfnNorm,
				QNorm: qn, KNorm: kn,
				Wq: model.Wt(l.Wq), Wk: model.Wt(l.Wk), Wv: model.Wt(l.Wv), Wo: model.Wt(l.Wo),
				Gate: model.Wt(l.Gate), Up: model.Wt(l.Up), Down: model.Wt(l.Down),
			}
			if c.MoE() {
				w.Router = model.Wt(l.Router)
				// The same lazy marker offerRange sets: the routed bank is
				// unread, and this goroutine must not do the file I/O beside a
				// live submission, so the tier skips packing such a block and
				// the upload path ensures it.
				ow := s.offerWeights(li, false)
				w.Ensure, w.EnsureExperts = ow.Ensure, ow.EnsureExperts
				w.PrefetchExperts, w.SpeculateExperts = ow.PrefetchExperts, ow.SpeculateExperts
				w.ExpertUses = ow.ExpertUses
			}
			s.ldCand.PrewarmLayer(li, &s.plan, &w)
		}
	}()
	return done
}

// SetGPULayers moves the CPU/GPU seam at runtime: n blocks on the device, the
// rest on the host. It returns how many the device actually holds afterwards,
// which can be fewer than asked when growing.
//
// Shrinking is exact and growing is best-effort: giving memory back always
// succeeds, while growing needs room and PrepLayer may decline any block.
func (s *Instance) SetGPULayers(n int) int {
	if s.ld == nil {
		return 0
	}
	defer s.settleHist()
	// What an integrated GPU holds moved with the seam.
	defer s.followHost()
	total := s.hi
	if n < s.lo {
		n = s.lo
	}
	if n > total {
		n = total
	}
	// An explicit seam overrides what relocation remembers: the caller has
	// said where the blocks go.
	s.moved = nil
	// Every placed block from n up, not the prefix: relocation can leave a
	// host block between two runs. A pinned one refuses the move.
	for _, li := range s.placedFrom(n) {
		if s.pinnedOnDev(li) {
			return s.gpuLayers
		}
	}
	if from := s.placedFrom(n); len(from) > 0 {
		// The head goes first, then the history comes home while the
		// device buffers still exist, then the blocks are freed.
		s.head, s.headReady = nil, nil
		for _, li := range from {
			if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
				return s.gpuLayers // refuse rather than lose the history
			}
		}
		s.ld.ReleaseLayers(n, total)
		s.shrinkOnDev(n)
	}
	if n > s.gpuLayers {
		was := s.gpuLayers
		// Only the blocks this call places get their history sent up: one
		// already on a device past a hole holds the only copy of its own, and
		// the host's pages for it were never written.
		var fresh []int
		for li := was; li < n; li++ {
			if !s.devAt(li) {
				fresh = append(fresh, li)
			}
		}
		s.prepDeviceRange(was, n)
		if s.opt.growDebug && s.gpuLayers == was {
			fmt.Fprintf(os.Stderr, "grow: PrepLayer(%d) declined\n", was)
		}
		for i, li := range fresh {
			if !s.devAt(li) {
				continue
			}
			if !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
				if s.opt.growDebug {
					fmt.Fprintf(os.Stderr, "grow: MigrateKV(%d) failed at pos %d\n", li, s.pos)
				}
				// The block is resident but its history is not, so it would
				// attend over an empty cache. Give it straight back, and every
				// block placed after it, whose history has not moved either.
				for _, lj := range fresh[i:] {
					if s.devAt(lj) {
						s.ld.ReleaseLayers(lj, lj+1)
						s.unmarkOnDev(lj)
					}
				}
				break
			}
		}
	}
	return s.gpuLayers
}

// Workers is how many worker threads the fast tier uses.
func (s *Instance) Workers() int { return s.cpu.x.Pool().Workers() }

// Pos is the number of positions already in the cache.
func (s *Session) Pos() int { return s.pos }

// PrefillRows is how many prompt rows this instance has run through its blocks
// since it was made: a prompt restored from a prefix cache adds none, and a
// one-row prompt (decode's path) adds one.
func (s *Session) PrefillRows() int { return s.promptRows }

// stepDownKVToF32 narrows the cache from packed halves to f32 and rebuilds it.
// It is a named method so a gate can reach it without a real card. The caller
// owns the pos == 0 precondition: the pages are rebuilt, not reinterpreted.
func (s *Instance) stepDownKVToF32() { s.setKVFmt(cpu.KVF32) }

// setKVFmt rebuilds an empty cache in format f, and the CPU's forms for it:
// what a served instance takes from the pool's first (Pool.Serve), so a
// session's history reads the same on every instance it leases.
func (s *Instance) setKVFmt(f cpu.KVFmt) {
	s.kvFmt, s.kvl.fmt = f, f
	// Carry every caller-visible setting into the fresh cache (namespace and
	// budget); a forgotten one is silently lost.
	old := s.kv
	s.kv = newKVCacheRange(s.c, old.id, s.nseq, s.kvl, old.store, s.opt.kvPage, s.lo, s.hi, s.Session.cpuView(), s.eng.cpu.KVPool(s.m.Id))
	s.kv.reserve(s.maxSeq)
	s.kv.ns, s.kv.budget = old.ns, old.budget
	// The CPU's blocks read the new cache: its arenas, and forms for f32.
	if err := s.kv.useArenas(s.maxSeq, s.kvArenaFor); err != nil {
		s.cpu.x.Composer().Note(err)
	}
	s.cpuPlace()
	s.Session.kvFormat = f
}

// seqPos and advance are the two places that know a position may be per row.
func (s *Instance) seqPos(slot int) int { return s.bpos[slot] }

func (s *Instance) advance(slot, n int) {
	s.bpos[slot] += n
	if s.bpos[slot] > s.pos {
		s.pos = s.bpos[slot] // the high water mark; see bpos
	}
	// Sealing is a consequence of the position moving, so it lives here. It
	// runs against the slowest row, not the furthest: a page holds every
	// row's slots.
	low := s.lowPos()
	s.kv.seal(low)
	// The recurrent half: it can only be captured AT a boundary, so it goes
	// through the position rather than through the page walk.
	s.sealRecurrent(low)
	s.kv.trim(low)
	// After the seal: a windowed layer's pages behind every row's window go
	// to its spare once the store has been offered them.
	s.kv.release(s.liveLow())
}

// liveLow is the slowest position among the rows that have history. A row at
// zero has none -- retired, or not yet admitted -- and whatever it writes it
// writes into pages of its own positions, which grow fresh if they were
// released; counting it would keep every page of the batch forever.
func (s *Instance) liveLow() int {
	low := -1
	for _, p := range s.bpos {
		if p > 0 && (low < 0 || p < low) {
			low = p
		}
	}
	return max(low, 0)
}

// lowPos is the slowest row's position -- the point before which no row can
// still write. It is what seal and trim are allowed to consider immutable.
func (s *Instance) lowPos() int {
	low := s.bpos[0]
	for _, p := range s.bpos[1:] {
		if p < low {
			low = p
		}
	}
	return low
}

// Reset clears the session without reallocating.
//
// It clears the pages and, on a hybrid, the recurrent summary, not only the
// positions: either left behind would give the next sequence the previous
// one's history.
func (s *Instance) Reset() {
	s.pos = 0
	for i := range s.bpos {
		s.bpos[i] = 0
		s.rewindRope(i, -1)
	}
	for i := 0; i < s.nseq; i++ {
		s.ResetRecurrent(i)
	}
	// A placed linear block keeps its running state on the device, and the
	// next prompt would start from the last one's summary: zeroed there too,
	// as Retire zeroes a row's. A failure is the next call's error.
	if s.recurrent() && s.devLinear() {
		if len(s.allRows) != s.nseq {
			s.allRows = make([]int, s.nseq)
			for i := range s.allRows {
				s.allRows[i] = i
			}
		}
		if rr, ok := s.ld.(nn.RecRowsDevice); !ok || !rr.ResetRecRows(s.allRows) {
			why := ""
			if er, ok := s.ld.(nn.ErrReporter); ok && er.Err() != "" {
				why = ": " + er.Err()
			}
			s.retireErr = fmt.Errorf("model: the recurrent state on the device could not be reset%s", why)
		}
	}
	s.aliasMark(0, s.pos+1)
	s.kv.retireTail(s.pos)
	s.kv.reset()
	s.kv.err.Store(nil)
	s.kvRoom = 0
	s.kvSkipped = 0
}

// ForwardEmbd is Forward with the token's embedding supplied rather than looked
// up: the decode half of the embedding seam a vision-language model splices
// image embeddings through.
//
// e is the residual stream, not a vocabulary row: EmbdScale is not applied
// here, so projected image features are not scaled twice. A caller reproducing
// the token path applies the scale itself (TestForwardEmbdMatchesForward).
func (s *Instance) ForwardEmbd(e []float32) ([]float32, error) {
	if len(e) != s.c.NEmbd {
		return nil, fmt.Errorf("model: ForwardEmbd: embedding is %d wide, want %d", len(e), s.c.NEmbd)
	}
	s.tok = -1
	return s.forward(func() error {
		st := &s.inScr
		st.tok1[0], st.pos1[0] = 0, s.pos
		if why := s.givenRows(s.x, s.ple, e, st.tok1[:], st.pos1[:]); why != nn.DeclineServes {
			return s.embedDeclined(why)
		}
		s.trace(-1, "embd", s.x)
		return nil
	})
}

// Forward runs one token through the model and returns the logits. The token is
// appended at the current position and the position advances.
//
// The graph, in execution order (not the order the tensors appear in the
// file):
//
//	x  = embed(token) * embdScale
//	for each layer:
//	    h = rmsnorm(x, attn_norm) ; q,k,v = Wq h, Wk h, Wv h
//	    rope(q, pos) ; rope(k, pos) ; cache k,v at pos
//	    x += Wo · attention(q, K[0..pos], V[0..pos])
//	    h = rmsnorm(x, ffn_norm)
//	    x += Wdown · (act(Wgate h) * Wup h)
//	logits = Woutput · rmsnorm(x, output_norm)
func (s *Instance) Forward(token int32) ([]float32, error) {
	defer s.fr.Enter()()
	if err := s.marksHeld(); err != nil {
		return nil, err
	}
	if int(token) < 0 || int(token) >= s.c.NVocab {
		return nil, errToken{token, s.c.NVocab}
	}
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return nil, err
	}
	s.kv.note(s.bpos[0], token)
	s.tok = token
	s.tokFill = true
	l, err := s.forward(func() error { return s.embed(token) })
	s.tokFill = false
	return l, err
}

// embedOnDevice promises the token's row to the device that runs block 0,
// which gathers it itself (nn.EmbedDevice) where the front end states the
// step's input rows (nn.Head.Embed, which GatherRow answers). The host's
// fill runs only if the device fails the token.
func (s *Instance) embedOnDevice() bool {
	if !s.tokFill || s.ldEmbed == nil || s.head == nil || !s.head.GatherRow || s.lo != 0 || !s.devAt(0) {
		return false
	}
	// Input rows that write the per-layer inputs on the device leave the
	// host's unfilled: no block may run on the host.
	if s.c.PLEDim != 0 && s.c.Streams() > 1 && !s.allOnDevice() {
		return false
	}
	// The card reads the per-layer inputs it writes back into the token's
	// own, handed before it runs.
	s.layerInputs(s.ple)
	s.tok1[0] = s.tok
	return s.ldEmbed.EmbedRows(s.tok1[:], s.pos, s.resid())
}

// ForwardGreedy is Forward for a caller that only wants the greedy next token:
// the argmax of the logits, lowest index on a tie. Where the output projection
// runs on a device it asks that device for the token alone
// (nn.Head.ArgmaxOnly), so the logits never cross the bus; everywhere else it
// is Greedy over Forward's logits. Either way the returned token is the same.
//
// It works under a final softcap too: the device head caps before its argmax
// (tanh saturates in float32 and can tie logits), so its token is Greedy over
// exactly the logits Forward would return.
func (s *Instance) ForwardGreedy(token int32) (int32, error) {
	h := s.head
	onDev := h != nil
	if onDev {
		h.ArgmaxOnly, h.Token = true, -1
		defer func() { h.ArgmaxOnly = false }()
	}
	logits, err := s.Forward(token)
	if err != nil {
		return 0, err
	}
	if onDev && h.Token >= 0 {
		return h.Token, nil
	}
	if tok, ok := s.greedy(logits); ok {
		return tok, nil
	}
	return 0, s.sampleDeclined("the greedy token")
}

// ForwardSample is Forward for a caller that only wants smp's next token.
// Where the output projection runs on a device and the draw is a top-k one
// (Sampler.Bounded), it asks that device for the k candidates alone
// (nn.Head.SampleK) -- the penalty and the selection run there -- and draws
// from them on the host (Sampler.SampleFrom), so k pairs cross the bus in
// place of the vocabulary's logits; everywhere else it is smp.Sample over
// Forward's logits. Either way the token is the one Sample would draw from
// the logits Forward returns: the device selects exactly the host's
// candidates, and the softmax, cuts and draw are the host sampler's own.
//
// A greedy smp is ForwardGreedy.
func (s *Instance) ForwardSample(token int32, smp *Sampler) (int32, error) {
	if smp == nil || smp.Temp <= 0 {
		return s.ForwardGreedy(token)
	}
	h := s.head
	onDev := h != nil && h.Left()&(nn.HeadBias|nn.HeadScale) == 0 && smp.Bounded(s.c.NVocab) &&
		s.opt.devSample != DeviceSampleOff
	// The call's one uniform, whichever arm draws with it: the device's
	// walk, the device's candidates, the CPU's walk or the logits.
	u := float32(smp.uniform())
	if onDev && smp.TopK > 0 && smp.TopK <= MaxTopK {
		k := smp.TopK
		topp, minp := smp.cuts()
		h.SampleK, h.SamplePenalty, h.Sampled = k, smp.penalty(), false
		h.SampleWindow = nil
		if h.SamplePenalty != 1 {
			h.SampleWindow = smp.window(s.c.NVocab)
		}
		h.WalkArgs, h.Token = [5]float32{float32(k), float32(1 / smp.Temp), topp, minp, u}, -1
		defer func() { h.SampleK, h.Sampled = 0, false }()
	}
	logits, err := s.Forward(token)
	if err != nil {
		return 0, err
	}
	var tok int32
	switch {
	case onDev && h.Sampled:
		s.SampledOnDevice++
		tok = h.Token
	default:
		t, ok := s.draw(logits, smp, u)
		if !ok {
			return 0, s.sampleDeclined("the draw")
		}
		tok = t
	}
	return tok, nil
}

// forward runs one decode step over whatever fill() puts in s.x.
//
// fill is a closure because the device-failure path calls it again to restart
// the token on the host (Layers may have written part of s.x).
func (s *Instance) forward(fill func() error) ([]float32, error) {
	s.touch()
	// An instance that led a host step across sessions lent the model its gate
	// scratch (returnStep -> takeHostBatch), and the token it runs alone
	// afterwards reads one query-gate row it no longer holds. It is made once;
	// an instance that does not lead holds it from NewInstance.
	if c := s.c; c.AttnOutGate {
		qd, n := c.NHead*c.HeadDim, len(s.bpos)
		if cap(s.qgate) < n*2*qd {
			s.qgate = make([]float32, n*2*qd)
		}
		if cap(s.ogate) < n*qd {
			s.ogate = make([]float32, n*qd)
		}
	}
	// The pool spins or parks for the next token by what this one did. A token
	// that paged weights in waits on reads between its regions, and a spinning
	// worker would burn a core through each; a token that read nothing runs its
	// regions back to back, and parking would cost a wake-up at every one.
	// Deciding by the model's size against the budget parked Qwen3-30B in a
	// 16 GiB cgroup for a run that read nothing after warm-up, at well under
	// half the spinning rate (scheduling-and-measurement.md, "What was left:
	// the pool spinning through a blocking read").
	if pc := s.fr; pc != nil {
		in0, _ := pc.Faults()
		defer func() {
			in1, _ := pc.Faults()
			s.cpu.x.Pool().SetSpinning(in1 == in0)
		}()
	}

	c := s.c
	if s.pos >= s.maxSeq {
		return nil, errFull{s.maxSeq, s.reqSeq}
	}
	// AltUp's streams ride the fill as well (altup.go): every restart begins
	// from the embedding.
	if c.StreamHead() {
		inner := fill
		fill = func() error {
			if err := inner(); err != nil {
				return err
			}
			return s.expandStreams(s.xa, 1, 0)
		}
	}
	s.relocateFor(s.pos + 1)
	if !s.embedOnDevice() {
		if err := fill(); err != nil {
			return nil, err
		}
	}

	// Each contiguous run of device blocks is one call (one upload, one
	// download, one drain), not one call per block.
	// The row's rotary tables are the CPU's to make where a block runs
	// there (oneTable); a device makes its own.
	s.oneFresh = false
	// With the last block on the device the projection rides the same
	// submission (folded); otherwise it is its own call below. Device runs sit
	// inside the block loop, and the residual crosses the seam once per run.
	//
	// AltUp's head reads the streams' mean, which the host takes, so the
	// projection is never folded into the last run.
	folded := s.head != nil && s.routeAt(s.hi-1).run == nn.LayerRunner(s.ld) && !c.StreamHead()
	// Not a defer inside the loop: one there takes every defer in this
	// function off the open-coded path, and a heap defer record costs a decode
	// token an allocation whenever the runtime's per-P pool runs dry.
	var unhide func()
	defer func() {
		s.unfuse()
		if unhide != nil {
			unhide()
		}
	}()
	for li := s.lo; li < s.hi; li++ {
		// A run of blocks on one device is one call, whichever device: the
		// residual crosses to it and back once.
		r := s.routeAt(li)
		hi := li + 1
		for hi < s.hi && s.routeAt(hi) == r {
			hi++
		}
		var fold *nn.Head
		if folded && hi == s.hi {
			fold = s.head
			unhide = s.rows.hiddenOf(fold)
		}
		if err := s.openRun(r); err != nil {
			return nil, err
		}
		if !r.run.Layers(li, hi, s.pos, 1, s.resid(), s.cs[:c.RopeW()], s.swaTable(c.LocalNRot()), fold) {
			if err := s.runRefusal(r, li); err != nil {
				return nil, err
			}
			// A fault demotes everything and restarts the token on the
			// CPU device from the embeddings (Layers may have written part
			// of the residual), unless SetDeviceFallback(false) asked for
			// the error.
			if !s.fallsBack() {
				return nil, errDevice{li, s.devWhy()}
			}
			// A hybrid cannot restart once a placed linear block has
			// advanced its recurrence (the host would apply it twice). li is
			// where the run started, not where it failed, so the guard asks
			// the tier's step count rather than the position.
			if s.recurrent() && s.devLinear() && s.recStepped() != 0 {
				return nil, errRecurRetry{li, s.devWhy()}
			}
			if !s.demoteAll() {
				// The history could not be rescued, so falling back really
				// would attend to an empty cache. Fail loudly instead.
				return nil, errDevice{li, s.devWhy()}
			}
			if err := fill(); err != nil {
				return nil, err
			}
			// demoteAll cleared the placement, so the retry is pure host.
			// li = lo-1 and the loop's ++ restarts it at the first block.
			folded = false
			li = s.lo - 1
			continue
		}
		s.trace(hi-1, "ffn_resid", s.x)
		if fold != nil {
			// The device ran the output norm and the projection too, so the
			// token is finished and the residual never came home.
			//
			if err := s.devLogits(); err != nil {
				return nil, err
			}
			s.advance(0, 1)
			// The token is complete, so it steps what every token steps:
			// a head on the device is no reason for the seam tuner or the
			// grow path to stop counting.
			s.hot.endToken()
			s.reap()
			s.growStep()
			s.seamStep()
			// A page the store could not return fails the request.
			if err := s.kvCheck(); err != nil {
				return nil, err
			}
			return s.logits, nil
		}
		li = hi - 1
	}
	// A head over several streams reads their collapse, written into stream
	// 0. A device that collapses the
	// streams itself (nn.Head.Collapses) is handed them as they are, and the
	// host collapses them only where it refuses the call.
	devCollapse := s.head != nil && !folded && s.head.Collapses && s.rows == nil && s.lo < c.NLayer
	if c.StreamHead() && !devCollapse {
		if err := s.collapseStreams(s.xa, 1); err != nil {
			return nil, err
		}
	}

	// The projection on the device under a partial seam: an empty block range
	// with a head, so the tier runs only the output norm and projection (the
	// same tail the folded path uses). A refusal falls through to the host,
	// which has the norm and the weights either way.
	done := false
	if s.head != nil && !folded {
		s.unfuse()
		if !c.ResidStreams() { // its hidden state is the residual, taken in collapseStreams
			defer s.rows.hiddenOf(s.head)()
		}
		if s.ld.Layers(s.gpuLayers, s.gpuLayers, s.pos, 1, s.resid(), s.cs[:c.RopeW()], s.swaTable(c.LocalNRot()), s.head) {
			done = true
		} else if err := s.refusal(s.gpuLayers); err != nil {
			return nil, err
		} else if !s.fallsBack() {
			return nil, errDevice{s.gpuLayers, s.devWhy()}
		}
	}
	if devCollapse && !done {
		if err := s.collapseStreams(s.xa, 1); err != nil {
			return nil, err
		}
	}
	// The CPU's composed head applies every stage of the head's graph.
	if !done {
		var hidden []float32
		if s.rows != nil && !c.ResidStreams() {
			hidden = s.rows.hidden
		}
		if why := s.headRow(s.x, s.logits, hidden); why != nn.DeclineServes {
			return nil, cpuHeadRefused(why)
		}
	}
	s.unfuse()
	if done {
		if err := s.devLogits(); err != nil {
			return nil, err
		}
	}
	// Through advance, not s.pos++: Prefill reads bpos, so a Prefill after a
	// Forward must see this token.
	s.advance(0, 1)
	s.hot.endToken()
	s.reap()
	s.growStep()
	s.seamStep()
	// A page the store could not return fails the request; see kvCache.faultIn.
	if err := s.kvCheck(); err != nil {
		return nil, err
	}
	return s.logits, nil
}

// errFull is the end of the context. It names newState's clamp when the caller
// asked for more positions than the model's context.
type errFull struct{ max, asked int }

func (e errFull) Error() string {
	s := "model: kv cache is full (" + model.Itoa(e.max) + " positions)"
	if e.asked > e.max {
		s += "; " + model.Itoa(e.asked) + " were requested and the model's context is " +
			model.Itoa(e.max)
	}
	return s
}

type errToken struct {
	tok    int32
	nVocab int
}

func (e errToken) Error() string {
	return "model: token " + model.Itoa(int(e.tok)) + " out of range for vocab " + model.Itoa(e.nVocab)
}

// Cand is one candidate token and its logit.
type Cand struct {
	ID    int32
	Logit float32
}

// TopK returns the k highest-scoring tokens, best first. Diagnostic only: the
// margin between the top two is what distinguishes float reassociation from a
// real disagreement with another implementation.
func TopK(logits []float32, k int) []Cand {
	out := make([]Cand, 0, k)
	for i, v := range logits {
		if len(out) < k {
			out = append(out, Cand{int32(i), v})
		} else if v > out[len(out)-1].Logit {
			out[len(out)-1] = Cand{int32(i), v}
		} else {
			continue
		}
		for j := len(out) - 1; j > 0 && out[j].Logit > out[j-1].Logit; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// swaTable is the local rotary table for the device, or nil where the
// architecture has one base. The device gets a range of blocks in one
// submission, so it needs both tables and the period to choose between them;
// see nn.LayerPlan.SWAPeriod.
func (s *Instance) swaTable(nrot int) []float32 {
	if s.csSWA == nil {
		return nil
	}
	return s.csSWA[:nrot]
}

// windowFor is the sliding window a device must apply in a session of maxSeq
// positions: none when the window is no shorter than the session, since then it
// never truncates anything and the windowed kernels would only cost a select.
func windowFor(c *model.Config, maxSeq int) int {
	if c.SWAWindow <= 0 || c.SWAWindow >= maxSeq {
		return 0
	}
	return c.SWAWindow
}

// swaPeriodFor is the plan's copy of Config.SWA's pattern, and it is zero unless
// the architecture trains two rotary bases: a sliding window alone is honoured
// through the per-row key count and needs no second table (gemma2).
func swaPeriodFor(c *model.Config) int {
	if c.RopeBaseSWA == 0 {
		return 0
	}
	return int(c.SWAPeriod)
}

// narrow copies one run of f32 into the f32 cache.

// errDevice is a device failure the instance cannot continue past on the host.
// why is the device's own reason (nn.ErrReporter), so a refusal the backend
// names -- a kernel launched at a width it does not declare, say -- reaches the
// caller by that name rather than as "the accelerator failed".
type errDevice struct {
	layer int
	why   string
}

func (e errDevice) Error() string {
	return fmt.Sprintf("model: the accelerator failed on block %d; its KV cache is on the "+
		"device and cannot be continued on the CPU%s", e.layer, e.why)
}

// errRecurRetry is a device failure on a hybrid after a linear block advanced:
// restarting the token on the host would apply that block's recurrent update a
// second time. Snapshotting rconv/rstate at the head of each token would allow
// a restart; it is not built. why is as errDevice's.
type errRecurRetry struct {
	layer int
	why   string
}

func (e errRecurRetry) Error() string {
	return fmt.Sprintf("model: the accelerator failed on block %d and this model is a "+
		"hybrid: the linear blocks before it have already advanced their recurrent "+
		"state, so restarting on the host would apply them twice%s", e.layer, e.why)
}

// devWhy is the device's own reason for its last failure, as ": reason", or ""
// when it gives none.
func (s *Instance) devWhy() string {
	if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
		return ": " + e.Err()
	}
	return ""
}

// recurrent reports that this session carries a linear block's running state,
// which is what makes re-running a block from the embeddings unsound.
func (s *Instance) recurrent() bool { return s.view.Recurrent() }

// RecurrenceAdvanced reports whether the device call that just failed had
// already advanced one of this instance's linear blocks, so running its tokens
// again would apply them twice. It asks the instance's own device session, which
// is where the count lives: the GPU's own count is session 0's, and an instance
// steps on a session of its own. A joint step counts on the instance that led it,
// so a caller asks every instance in the step.
func (s *Instance) RecurrenceAdvanced() bool { return s.recurrent() && s.recStepped() != 0 }

// recStepped is how many placed linear blocks advanced during the submission
// that just failed. Zero means nothing moved and the host restart is sound; it
// is -1 when the tier cannot say, which is read as "assume it did". It is
// measured because both positional proxies (the run's start, or "any linear
// block placed") are wrong in one direction or the other.
func (s *Instance) recStepped() int {
	rs, ok := s.ld.(nn.RecStepper)
	if !ok {
		return -1 // a tier that cannot say; treat any failure as unsafe
	}
	return rs.RecSteps()
}

// devLinear reports that at least one linear block is on a device right now
// (the set, not the prefix), so a failed submission may have advanced a
// summary the host cannot rewind.
func (s *Instance) devLinear() bool {
	if !s.view.Recurrent() {
		return false
	}
	for li := 0; li < len(s.onDev) && li < s.view.Blocks(); li++ {
		if s.onDev[li] && s.view.Conv(li) != nil {
			return true
		}
	}
	return false
}

// MaxSeq is how many positions this session preallocated its KV cache for.
func (s *Instance) MaxSeq() int { return s.maxSeq }

// ContextClamped reports whether the model's context is shorter than what was
// asked for, and by how much, so a caller can refuse rather than run short.
func (s *Instance) ContextClamped() (asked, got int, clamped bool) {
	return s.reqSeq, s.maxSeq, s.reqSeq > s.maxSeq
}

// setLD sets the instance's layer device and the optional faces a token and a
// step ask it for, asserted once here: an interface-to-interface assertion
// caches its answer in a heap object at a random later call, which made a
// decode token's allocation count nondeterministic.
func (s *Instance) setLD(ld nn.LayerDevice) {
	s.ld = ld
	s.routes[0] = newRoute(ld)
	s.ldRoom, _ = ld.(nn.RoomReporter)
	s.ldSpill, _ = ld.(nn.Spiller)
	s.ldStep, _ = ld.(nn.SessionStepper)
	s.ldRows, _ = ld.(nn.RowsDevice)
	s.ldInputs, _ = ld.(nn.LayerInputDevice)
	s.ldPrompt, _ = ld.(nn.PromptDevice)
	s.ldMarks, _ = ld.(nn.RopeMarksDevice)
	s.rmarks.Sent = s.rmarks.Sent[:0]
	s.sendMarks()
	s.ldEmbed, _ = ld.(nn.EmbedDevice)
}

// layerInputs hands the device the rows' side inputs for the call about to
// be made (nn.LayerInputDevice): their per-layer embedding inputs, or their
// tap rows. Nothing where the model has none.
func (s *Instance) layerInputs(in []float32) {
	if s.ldInputs != nil && s.c.SideWidth() != 0 {
		s.ldInputs.SetLayerInputs(in)
	}
}
