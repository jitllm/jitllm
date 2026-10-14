package model

import (
	"fmt"
	"os"
	"runtime"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/cpu"
)

// Per-op timing, armed by WithProfile. The counters are process-wide (OpProfile
// and ResetProfile have no receiver); the arm is per model, so a model opened
// without WithProfile never records because another one was opened with it.
var opNanos [nOps]atomic.Int64

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

var opNames = [nOps]string{"matvec", "rmsnorm", "rope", "attention", "activation", "residual", "router"}

func (s *State) tick() int64 {
	if !s.prof {
		return 0
	}
	return time.Now().UnixNano()
}

func (s *State) tock(op opID, t0 int64) {
	if s.prof {
		opNanos[op].Add(time.Now().UnixNano() - t0)
	}
}

// OpProfile returns per-op nanoseconds and their names, largest first is the
// caller's problem.
func OpProfile() ([]string, []int64) {
	out := make([]int64, nOps)
	for i := range out {
		out[i] = opNanos[i].Load()
	}
	return opNames[:], out
}

// ResetProfile zeroes the per-op counters.
//
// The counters accumulate across both phases, and a long prompt's prefill
// swamps decode; reset between the two and each phase answers for itself.
func ResetProfile() {
	for i := range opNanos {
		opNanos[i].Store(0)
	}
}

// State is one generation session: the KV cache plus the scratch buffers the
// forward pass reuses. One State is one conversation; it is not safe to share.
type State struct {
	m *Model
	// pending is the move between devices PrepareMove started and no Commit
	// or Cancel has ended (moveblocks.go).
	pending *PendingMove
	// c is the configuration of the segment this State runs: the text model's
	// (m.Cfg) for every session a caller makes, the vision tower's for the
	// State an image is encoded in (vision.go). The blocks are the model's
	// either way; only their geometry differs.
	c   *Config
	pos int
	// sampleArgs, sampleVals and sampleIDs are ForwardSample's device
	// sampler block and the candidates it returns, reused across tokens.
	// SampledOnDevice counts the tokens whose candidates came home in place
	// of the logits.
	sampleArgs      []uint32
	sampleVals      []float32
	sampleIDs       []uint32
	SampledOnDevice int64
	// SampleProbes counts the tokens ForwardSample timed to decide between
	// the device and the host sampler (devsample.go).
	SampleProbes int64
	// parked and parkedSeam are Park's: the session is preempted, and the
	// seam it held on the device. ParkedOut and ParkedIn count the KV pages
	// Park sent to the store and Resume brought back (park.go).
	parked     bool
	parkedSeam int
	parkStore  *MemStore
	// parkPrev is the store ParkInto swapped out, given back by Resume.
	parkPrev            KVStore
	ParkedOut, ParkedIn int64
	// outNorm and outW are the head this State projects through: the model's
	// output norm and projection for a trunk, a prediction block's own head
	// norm (and the trunk's projection, or the block's own copy) for a draft.
	// outW points at the tensor rather than copying it: a block's own head
	// is paged, and a page-in rebinds the model's copy.
	outNorm []float32
	outW    *tensor
	// lo and hi are the blocks this State runs, [lo, hi): the trunk, [0,
	// NLayer), for every session a caller makes, and a multi-token-prediction
	// block for the draft State speculation drives (newDraftState). Block
	// indices stay the model's -- the KV cache, the device's placement and the
	// pager all address block li as li -- so a draft State and its trunk share
	// one device without colliding.
	lo, hi int

	// rg is the pool regions a decode token runs, with their functions built
	// once (see regions).
	rg regions
	// fuseOff runs this State's decode one pool region per op
	// (WithBlockFusion, SetBlockFusion); fused is whether a fused run of host
	// blocks is open (fuse.go), and FusedRuns counts the runs opened.
	fuseOff   bool
	fused     bool
	FusedRuns int64
	// grp is groupOf's call, reused; groupOff keeps a fused run's matvecs
	// one region each (SetMatVecGroups); Grouped counts the group calls.
	grp      nn.MVGroup
	groupOff bool
	// aheads is the fused runs' lookahead sets, one per (block, op), and
	// lookOff turns the lookahead off (fuse.go).
	aheads  []nn.AheadSet
	lookOff bool
	// residFold lets the o and down groups add their rows into the
	// residual (WithResidualFold).
	residFold bool
	Grouped   int64

	// kv is the paged KV history (see engine/model/kvpage.go). Pages are committed as
	// the sequence reaches them rather than reserved for the whole context.
	kv *kvCache
	// vis is the vision segment's run: set on a State over the tower's blocks
	// (State.Vision), nil on every text State.
	vis *visRun
	// visState is the vision State a text State encodes its pictures in
	// (Vision), nil until the first picture.
	visState *State
	// enc is an encoder segment's run: set on the State an Embedder places
	// its encoder's blocks through (encdev.go), nil everywhere else.
	enc *encRun
	// kvErr is the first page fault of the current step that could not be
	// recovered. It is set inside a pool closure, which has no error path, and
	// drained by kvCheck at the entry point.
	kvErr atomic.Pointer[error]
	// spanTok is every row's token of the mixed prompt being prefilled, -1
	// for a row that has none (tokensOf): what a model with per-layer
	// embeddings, or DeepSeek V4's hash blocks, read where the prefill has no
	// ids. nil outside one.
	spanTok []int32
	// kvSkipped is what the last PrefillCached restored rather than computed.
	kvSkipped int
	// promptRows counts the prompt rows this State ran through its blocks, over
	// its life (PrefillRows): what a benchmark checks to show a prompt was
	// computed, not restored.
	promptRows int
	// forwardPrompts counts the prompts run a token at a time through the
	// decode path (WithPrefillForwardRows).
	forwardPrompts int
	// interrupt, when set, is asked before every prefill chunk
	// (SetPrefillInterrupt).
	interrupt func() bool

	x, h, xb []float32 // residual stream and two same-width scratch vectors
	// hf is a parallel block's FFN input (C6): the block input normed for the
	// FFN before attention adds into the residual. nil on every sequential
	// model. bhf is its batched twin.
	hf, bhf []float32
	// prow is one row of scratch for a learned position table (starcoder);
	// nil on every model without one.
	prow   []float32
	q      []float32 // n_head * head_dim
	kc, vc []float32 // one position's K and V
	att    []float32 // attention scores for one head
	atth   []float32 // ... and one row per head, so heads can run in parallel
	// float32 staging for attention's query and weighted accumulation.
	qf, xbf []float32
	attf    []float32 // softmaxed weights, narrowed for the accumulate kernel
	// attStride is attf's per-head stride, maxSeq rounded up to a vector: the
	// generated softmax fills [n, pad(n)) with -inf, and at n == maxSeq an
	// exact stride would pad into the next head's row, which another worker
	// is reading.
	attStride int
	gate, up  []float32
	logits    []float32
	maxSeq    int
	reqSeq    int       // what the caller asked for; > maxSeq when the model is shorter
	kvFmt     cpu.KVFmt // the KV cache's element format

	// MoE scratch, allocated only for a mixture-of-experts model. moeProbs is
	// NExpert wide and moeGate/moeUp are the per-expert ffn width, which is not
	// NFFN -- gate and up are NFFN for every dense model here and NFFNExp for
	// this one.
	moeProbs, moeGate, moeUp, moeDown []float32
	// moeGate/moeUp hold every selected expert, not one, so the k gate and k
	// up projections go out in one pool region (nn.MatVecPackedMulti) and the
	// SwiGLU over all of them is one more. Expert i occupies
	// [i*moePad, i*moePad+NFFNExp); the slack between NFFNExp and moePad is
	// allocated zero, written by nothing but the activation (silu(0)*0 = 0) and
	// read by nothing, so it stays zero for the life of the State.
	// The per-expert paths take [:NFFNExp], which is what they always had.
	moePad int
	// Scratch for the batch: the k*2 sheets and their outputs, reused so a
	// decode token allocates nothing.
	moeW []*nn.Packed
	moeO [][]float32
	// moeOnes is NExpertUsed ones: the weights a mixture that applies its
	// routed weight to the expert's input (Llama 4) sums the downs at.
	moeOnes []float32
	// The shared expert's scratch. Its width is NFFNShExp, a different key
	// from the routed experts'.
	shGate, shUp, shOut []float32
	// The shared expert run behind the routed read (moe): the block and row
	// it is for, the read's join and error, and its weight once computed.
	shOverlap  *layer
	shOverlapH []float32
	expWG      sync.WaitGroup
	expReq     chan expRead
	expErr     error
	shW        float32
	// hyOrd and hyW are hostExperts' selection in ascending id and its
	// weights in that order, kept so a token allocates neither.
	hyOrd []int32
	hyW   []float32
	// hostOnly keeps every matvec on the host's kernels: set while the
	// device's own submission waits on hostExperts, when offering a matvec to
	// that device would wait on the session it is inside of.
	hostOnly bool
	// hostCalls counts the device's calls into hostExperts, so a token can
	// tell whether its submission suspended for the host's experts.
	hostCalls int
	// autoStreamed counts the blocks the last placement streamed because no
	// card could hold them resident (nn.AutoStreamer).
	autoStreamed int
	shReady      bool
	// ShOverlaps counts mixture layers whose shared expert ran behind the
	// routed read: the selection check for WithSharedOverlap.
	ShOverlaps int
	// sg is the shared expert's one-logit gate and a constant 1, the operands
	// of a one-element generated sigmoid (see sharedExpert).
	sg [2]float32
	// ple is this token's per-layer embedding inputs (Config.PLEDim), pleT
	// the table row they are built from, pleG and pleO a block's gate and
	// projection, and bple a batched chunk's inputs, one row each. See ple.go.
	ple, pleT, pleG, pleO, bple []float32
	// Gemma 3n's AltUp and LAuReL (altup.go): xa is the one-row residual's
	// streams (x is its stream 0), altPred the block's predictions, and the
	// rest the router's, the coefficients' and the branches' scratch; balt*
	// are a chunk's.
	xa, altPred, altH, altM, altC, altOnes []float32
	altInn, altLaur, altLT, altExp, altXC  []float32
	baltPred, baltLaur                     []float32
	// ds is DeepSeek V4's scratch (ds4.go); xa is its streams too. tok is
	// the token a decode step embeds, -1 for a supplied embedding: DeepSeek
	// V4's hash blocks route by it.
	ds *ds4Scratch
	// k3 is Kimi-K3's residual attention and latent mixture buffers (k3.go),
	// nil on every other model.
	k3 *k3Scratch
	// q4 is Qwen3.8-Flash-Next's scratch (q4e.go); xa is its streams too.
	q4  *q4eScratch
	tok int32
	// tok1 hands a decode step's token to the device (tokenIDs) without an
	// allocation, and rowx one row's streams in a prefill's row-by-row
	// device retry (rowStreams).
	tok1 [1]int32
	rowx []float32
	// ldInputs is the device's per-layer input face (nn.LayerInputDevice),
	// asserted once in setLD; nil where it has none. ldTokens is its token
	// face (nn.TokenDevice).
	ldInputs nn.LayerInputDevice
	ldTokens nn.TokenDevice
	// ldNGram is its n-gram face (nn.NGramDevice), Qwen3.8-Flash-Next's.
	ldNGram nn.NGramDevice
	// ldQSA is its QSA face (nn.QSADevice), Qwen3.8-Flash-Next's.
	ldQSA nn.QSADevice
	// dmoe is Gemma 4's mixture block's scratch (Config.DenseMoE): the
	// router's input and the mixture's sum, each NEmbd wide. See denseMoE.
	dmoe [2][]float32
	// rowsOfChunk says moe() is running one row of a batched chunk, which
	// counts no hotness: endToken() advances once per Forward. See
	// denseMoERows.
	rowsOfChunk bool
	// route is the router kernel's output: the selection in descending
	// probability, the same ids ascending, both orders' renormalised weights
	// and the divisor. One per State, reused every layer of every token, so a
	// decode token allocates nothing here either.
	route *nn.MoERoute
	// expHold pins a layer's routed expert pages while its FFN reads them;
	// reused, like route.
	expHold expertHold
	// spareHost is a closed State's prompt scratch, taken from the model with
	// its promptBuf, for growBatch and growMoEBatch to grow into.
	spareHost hostBatch
	// traceLayer is the block moe() is running, for traceExperts.
	traceLayer int
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
	dg            deltaGeom
	rconv, rstate [][]float32
	// The delta rule's scratch, allocated only for a hybrid. dOut is per
	// value head so the heads can run in parallel without sharing.
	dMixed, dZ, dBA              []float32
	dDecay, dBeta, dAlpha, dBRaw []float32
	// dLowRank is the rank-kDim temporary KDA's two low-rank gates project
	// through, and dGateWaste receives the half of DeltaGate32JIT's pair that
	// the call in question is not for. Both nil on qwen3next.
	dLowRank, dGateWaste []float32
	dOut, dOnes          []float32
	// mvCalls counts mv: each one reads its weight once, the count a gate
	// holds a batched path to (TestLinearRowsCounts).
	mvCalls int64
	// lin is a prompt chunk's linear-block projections, run as GEMMs before
	// the row loop (linrows.go).
	lin linRows
	// dSB and dSC are a Mamba-1 block's B and C, and dZeroA the zero rates
	// its dt goes through SSDGate32JIT against (only the softplus is kept).
	dSB, dSC, dZeroA []float32

	// res keeps the hot experts resident and drops the rest; nil when the model
	// fits its budget, which is every model but a large MoE on a small one.
	res *residency
	hot *Hotness
	// plan and ldCand are kept so the seam can be moved after load: growing
	// the device's prefix needs the same LayerPlan the first offer used.
	plan   nn.LayerPlan
	ldCand nn.LayerDevice
	// forgets says the device attached for lone matvecs is told when the
	// pager recycles a frame (Model.watch), so mv may offer it a page's
	// weights and not only the dense region's.
	forgets bool
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
	// devRowChunks counts prompt chunks the device refused whole and ran
	// again a row at a time (prefillSrc's per-row retry). See DeviceRowChunks.
	devRowChunks int
	// tailRowChunks counts few-row prompt chunks run as one ragged step
	// (tailrows.go). See TailRowChunks.
	tailRowChunks int
	// tailPos is a ragged tail step's positions, kept so the step allocates
	// nothing once grown.
	tailPos []int
	// rtTook, rtStart and rtSeen are rowsTake's scratch, kept from step to step.
	rtTook, rtStart []int
	rtSeen          []bool
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

	// nseq is how many independent sequences ForwardBatch steps at once. At 1
	// every offset below collapses to zero, so Forward and Prefill are the same
	// code they were. The KV cache is strided by sequence first: sequence j's
	// position p is k[li][(j*maxSeq+p)*kvDim].
	nseq int
	// bpos is a position per row, which is what makes a batch ragged. It is
	// always allocated, one entry even for a single sequence (NewBatch(1) is
	// legal). s.pos stays the high-water mark across rows: the maxSeq bound
	// and Pos() want the furthest row.
	bpos []int
	// rmarks is, per slot, where an image has moved the rotary position of
	// the rows after it off their cache position (mrope.go). Empty for every
	// sequence that has seen no image, and on every single-axis model.
	rmarks [][]ropeMark
	// ximg is, per slot, the cache position each picture starts at under
	// XD-RoPE, whose image rows carry the picture's ordinal (xdRope).
	ximg [][]int
	// batched is whether this State came from NewBatch, which nseq cannot say
	// at a width of one. It is what decides which API a session accepts.
	batched bool
	// kvl is where this State's KV cache puts things; see kvLayout. It is the
	// global layers' layout; kvlAt is any layer's.
	kvl kvLayout
	// attnG and attnL are the attention kernel sets of the global and the
	// sliding layers, one set (the JIT's default) wherever the two geometries
	// are one; see provisionAttn and attnAt.
	attnG, attnL *nn.AttnSet
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

	// Multi-head Latent Attention scratch; nil unless Config.MLA(). See mla.go
	// for what each holds and why the absorbed query is row-wide.
	mlaQA, mlaAbs, mlaAcc, mlaPE, mlaOut []float32
	mlaAbsO, mlaOutO                     [][]float32
	// The prefill chunk's copies of the same, sized by growMLABatch.
	bmlaQA, bmlaAbs, bmlaAcc, bmlaKV, bmlaOut []float32
	bmlaAbsO, bmlaOutO                        [][]float32
	mlaKB, mlaVB                              []*nn.Packed
	// mlaG and bmlaG are Kimi-K3's MLA output gate (mla.go), one row and a
	// chunk's; nil where no block has one.
	mlaG, bmlaG []float32
	// mlaAbsLooped counts absorb steps that fell back to one matvec per head.
	mlaAbsLooped atomic.Int64
	// DeepSeek V3.2's lightning indexer (indexer.go); nil unless
	// Config.Indexer(). idxAttn scores the cached rows' indexer tails.
	idxQ, idxW, idxTmp, idxScore, idxBias []float32
	bidxQ, bidxW, bidxK, bidxBias         []float32
	idxMasks                              [][]float32
	idxOrd                                nn.SampleOrder
	idxAttn                               *nn.AttnSet
	// MiniMax Sparse Attention (msa.go); nil unless Config.MSA(). attnM is
	// the attention kernel set of the MSA blocks, whose row carries the
	// indexer's key as one more head, and reads that head for the scores.
	msaQ, msaTmp, msaBias  []float32
	bmsaQ, bmsaK, bmsaBias []float32
	msaKept                []bool
	msaMasks               [][]float32
	msaOrd                 nn.SampleOrder
	attnM                  *nn.AttnSet
	// moeBatched and moeLooped count which MoE FFN dispatch shape ran: one
	// region per stage, or one per expert. Without them "made no difference"
	// and "never ran" read the same.
	moeBatched, moeLooped atomic.Int64
	// The down projection is a separate question from gate/up: it has its own
	// contract (one activation per expert) and its own way to decline, so one
	// counter for both would read "batched" while half the FFN looped.
	moeDownBatched, moeDownLooped atomic.Int64
	blogits                       []float32 // nseq * NVocab, ForwardBatch's output
	retireErr                     error     // a device reset Retire or Reset could not make; see Retire
	allRows                       []int     // 0..nseq-1, the rows Reset zeroes on a device
	// placing is true while SetDeviceLayers offers blocks, when WithPlacement's
	// map applies; placeErr is what it could not do.
	placing  bool
	placeErr error
	// Attention scores per (sequence, head). atth/attf are the nseq==1 case;
	// a batch parallelises over sequences and heads, so every pair needs a row.
	bathf []float32
	// scoreReach is the key reach bathf and battf are sized for (rowsStride).
	scoreReach int

	// Batched prefill scratch, allocated on first use and sized by
	// PrefillChunk rather than by the sequence length.
	bx, bh, bq, bxb []float32
	bk, bv          []float32
	bgate, bup      []float32
	// bclamp is a clipped linear's clamped input (Gemma 4's tower,
	// clampIn), grown on first use; nil on every other model.
	bclamp    []float32
	bqf, bxbf []float32
	battf     []float32 // per-token score rows, at rowsStride

	// jit is the generated-code tier.
	jit *nn.JIT

	// device is the device SetDeviceLayers was handed, before Attach made ld a
	// session of it: what a draft State attaches to (Speculate).
	device nn.Device
	// ld runs whole blocks on an accelerator; gpuLayers is how many it accepted,
	// always a prefix. cs is the per-position rotary table those blocks read.
	ld        nn.LayerDevice
	gpuLayers int
	// ldRoom, ldSpill, ldStep, ldRows and ldNamed are ld's optional
	// faces, nil where it has none; set with ld, because a step asks for them.
	ldRoom  nn.RoomReporter
	ldSpill nn.Spiller
	ldStep  nn.SessionStepper
	ldRows  nn.RowsDevice
	ldNamed nn.NamedDevice
	// The lists a step this State leads reuses (step.go): Step's runs, the
	// rows' sessions and positions, the wanted rows' logits and the result.
	stepRuns   []Run
	stepSess   []nn.LayerDevice
	stepPos    []int
	stepTok    []int32
	stepLogits []float32
	stepRes    [][]float32
	// A step across host sessions (stepHost) this State leads: each row's
	// session, whether the row is its session's furthest (the one that
	// faults its attention window in), and each row's slot, always 0.
	rowOwn []*State
	rowWin []bool
	rowSeq []int
	// stepb is the model's step scratch holder while this State leads a
	// host step (borrowStep).
	stepb *stepBuf
	// ra and raRun are a ragged step's attention fan-out (rowsHost).
	ra    rowsAttn
	raRun func(lo, hi int)
	// rowCausal is a ragged step's rows in position order (rowsHost).
	rowCausal []int
	// onDev[li] says block li runs on a device: the truth for execution, where
	// gpuLayers is the truth for the seam. Placement is a set and migration a
	// boundary: gpuLayers is the contiguous device prefix (what -gpu-layers
	// sizes and seamtune moves), while a hybrid whose kinds alternate can have
	// placed blocks with a prefix of zero.
	onDev []bool
	// hostHeld says this State is counted in m.hostUse for the blocks of
	// [lo, hi) it does not place, from its making until Close.
	hostHeld bool
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
	// ropeTable a pointer compare for every other model.
	csSWA  []float32
	bcsSWA []float32
	// bcs is cs for a whole prefill chunk: one nRot-wide rotary table per row,
	// which is what the device's batched RoPE reads.
	bcs []float32
	// rowsLog and rowsHid are where a speculation step on the device that wants
	// a tail of its rows has the head write every row (stepRowsDevice), the
	// tail then copied out; rowsTok is where a greedy one's tokens land.
	rowsLog, rowsHid []float32
	rowsTok          []int32

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

	// deep is set for the length of one prefill whose spans carry deepstack
	// rows (deep.go): where they add in. deepIdx is its row index scratch.
	deep    []deepRun
	deepIdx []int32

	// rows is set for the length of one speculation step (specrows.go): the
	// rows whose logits and hidden state the step wants back, and the
	// recurrent snapshots it takes. nil in every other call.
	rows *rowsOut

	// rowTap, when set, receives every prefill chunk's final residual rows --
	// after the last block, before the output norm -- with the position of the
	// first. Nil in every session but a gate's: it is how a comparison sees
	// every position of a prompt rather than only the last row's logits.
	rowTap func(base int, rows []float32)
}

// NewState allocates a session able to hold maxSeq positions.
func (m *Model) NewState(maxSeq int) *State { return m.newState(1, maxSeq) }

// NewBatch allocates nseq independent sessions that step together, one token
// each per ForwardBatch. They share one pass over the weights, which is the
// whole point: decode is memory-bound, so the second sequence is nearly free.
// Each row carries its own position (bpos).
func (m *Model) NewBatch(nseq, maxSeq int) *State {
	s := m.newState(max(nseq, 1), maxSeq)
	s.batched = true
	return s
}

func (m *Model) newState(nseq, maxSeq int) *State {
	return m.newStateRange(nseq, maxSeq, 0, m.Cfg.NLayer)
}

// newStateRange is newState over blocks [lo, hi) of the model; see State.lo.
func (m *Model) newStateRange(nseq, maxSeq, lo, hi int) *State {
	// An encoder has no decoder layers, no cache and no logits, so a State
	// over it would run zero blocks and project nothing: refused by name
	// rather than handed back empty. NewState has no error return, and this
	// is a caller's contract violation, not a runtime condition.
	if m.enc != nil {
		panic(fmt.Sprintf("model: %s is an encoder; embed with NewEmbedder, it has no decoder State", m.Cfg.Arch))
	}
	m.states.Add(1)
	c := m.Cfg
	// Clamp to the model's context, and record the request (reqSeq) so a
	// later "kv cache is full" can name the number the caller passed.
	req := maxSeq
	if maxSeq > c.NCtx {
		maxSeq = c.NCtx
	}
	s := &State{m: m, c: m.Cfg, lo: lo, hi: hi, gpuLayers: lo, maxSeq: maxSeq, reqSeq: req, nseq: nseq,
		devFallback: true, relocate: true, prof: m.opt.profile, outNorm: m.outNorm, outW: &m.output,
		fuseOff: m.opt.noFuse, groupOff: m.opt.noGroups, lookOff: m.opt.noLook,
		residFold: m.opt.residFold}
	s.holdHostRuns()
	// A State over a prediction block projects through that block's head.
	if lo < len(m.layers) {
		if w := m.layers[lo].mtp; w != nil {
			s.outNorm, s.outW = w.headNorm, &w.head
		}
	}
	s.kvl = kvLayout{headMajor: m.opt.kvHeadMajor, maxSeq: maxSeq, nKV: c.NKVHead,
		headDim: c.HeadDim}
	// MLA caches one row per position for the whole layer: the compressed
	// latent plus the shared rotary key (KVLoraRank + NRot), which every head
	// attends to. So nKV is genuinely 1.
	if c.KVLoraRank != 0 {
		s.kvl.nKV, s.kvl.headDim = 1, c.KVDim()
	}
	s.x = make([]float32, c.NEmbd)
	s.h = make([]float32, c.NEmbd)
	// A parallel block's FFN input, or a Falcon-H1 block's mixer output.
	if c.Parallel || c.SSDAttn() {
		s.hf = make([]float32, c.NEmbd)
	}
	if m.posEmbd.e != nil {
		s.prow = make([]float32, c.NEmbd)
	}
	// The attention output is exactly n_head*head_dim wide, which is not
	// n_embd on every model (Qwen3-30B-A3B, gemma-2); the output projection
	// requires len(x) == k.
	s.xb = make([]float32, c.MaxQDim())
	s.q = make([]float32, c.MaxQDim())
	s.kc = make([]float32, c.MaxKVDim())
	s.vc = make([]float32, c.MaxKVDim())
	s.att = make([]float32, maxSeq)
	s.initHotness()
	s.atth = make([]float32, c.NHead*maxSeq)
	s.qf = make([]float32, c.MaxQDim())
	s.xbf = make([]float32, c.MaxQDim())
	// A sink takes one slot past the last position, so the row grows by one.
	s.attStride = attStride(maxSeq + m.sinkSlot())
	s.attf = make([]float32, c.NHead*s.attStride)
	s.growPLE()
	s.allocAltUp()
	s.gate = make([]float32, c.MaxNFFN(), ffnPad(c.MaxNFFN()))
	s.up = make([]float32, c.MaxNFFN(), ffnPad(c.MaxNFFN()))
	s.logits = make([]float32, c.NVocab)
	// One rotary table per row: a ragged batch has rows at different
	// positions.
	s.cs = make([]float32, nseq*c.RopeW())
	if m.ropeSWA != nil {
		s.csSWA = make([]float32, nseq*c.NRotSWA)
	}
	s.bpos = make([]int, nseq)
	s.rmarks = make([][]ropeMark, nseq)
	s.allocRecurrent(nseq)
	s.allocMLA()
	s.allocMSA()
	if c.AttnOutGate {
		qd := c.NHead * c.HeadDim
		s.qgate = make([]float32, nseq*2*qd)
		s.ogate = make([]float32, nseq*qd)
	}
	if c.MoE() {
		// Capacity for the generated softmax's padding (see attStride); the
		// length stays NExpert, which is what the top-k reads.
		s.moeProbs = make([]float32, c.NExpert, attStride(c.NExpert))
		// route.Ord is route.Sel sorted ascending by expert id, the order the
		// FFN contributions are summed in (see moe()); the router kernel
		// writes both. It is sized for this model's gate, and the bias's
		// presence is baked into the kernel key, which is why expSelBias is a
		// model-level fact checked at load.
		s.route = nn.NewMoERouteFor(c.NExpert, c.NExpertUsed, nn.MoEGate{
			Sigmoid:      c.ExpertSigmoid,
			SqrtSoftplus: c.ExpertSqrtSoftplus,
			Bias:         m.expSelBias(),
			NGroup:       c.NExpertGroup,
			NGroupUsed:   c.NExpertGroupUsed,
			Norm:         !c.NoExpertNorm,
			Scale:        float32(c.routedScale()),
			SparseMixer:  c.SparseMixer,
		})
		s.moePad = ffnPad(c.NFFNExp)
		s.moeGate = make([]float32, c.NExpertUsed*s.moePad)
		s.moeUp = make([]float32, c.NExpertUsed*s.moePad)
		// moeDown holds every selected expert's down projection, so the k of
		// them come out of one region and the weighted sum is one more.
		s.moeDown = make([]float32, c.NExpertUsed*c.NEmbd)
		s.moeW = make([]*nn.Packed, 0, 2*c.NExpertUsed)
		s.moeO = make([][]float32, 0, 2*c.NExpertUsed)
		s.moeOnes = make([]float32, c.NExpertUsed)
		for i := range s.moeOnes {
			s.moeOnes[i] = 1
		}
		// The shared expert has its own width (NFFNShExp), not NFFNExp.
		if c.NFFNShExp > 0 {
			s.shGate = make([]float32, c.NFFNShExp, ffnPad(c.NFFNShExp))
			s.shUp = make([]float32, c.NFFNShExp, ffnPad(c.NFFNShExp))
			s.shOut = make([]float32, c.NEmbd)
		}
		if c.DenseMoE {
			s.dmoe = [2][]float32{make([]float32, c.NEmbd), make([]float32, c.NEmbd)}
		}
	}

	// One kernel per quantization type the model actually contains, emitted
	// eagerly — the shapes are all known from the header before the first token.
	s.jit = m.newJIT()
	// One pair of attention kernels per model: the head dimension and the KV
	// stride are the same for every layer, and the kernels bake the layout's
	// stride. The cache width is decided here, after the kernels exist: a
	// binary16 cache is only legal if the generated kernels read it.
	// WithKVType (or WithKVF16, SetKVType) forces it; otherwise
	// nn.KVWidthPaysOff emits both and asks whether f16 costs anything on
	// this host. A quantized cache is only ever forced (KVType).
	hdK, hdV := c.attnWidths()
	want := m.opt.kvType.fmt()
	if !m.opt.kvTypeSet {
		want = cpu.KVF32
		if s.jit.KVWidthPaysOff(hdK, s.kvl.Stride()) {
			want = cpu.KVF16
		}
	}
	// MiniMax Sparse Attention caches its indexer's key in the row (msa.go),
	// and the selection over it is discrete: a key rounded to binary16 moves a
	// block's best score by ~1e-3, which flips a near tie, and a flipped block
	// is a different attention, not a nearby one. The references keep the key
	// at f32 (llama.cpp's converter and cache force F32 for the indexer), and
	// so does every device here (the paged K pool is f32), so the host's
	// default is f32 too; WithKVF16(true) still forces binary16.
	if c.MSA() && !m.opt.kvTypeSet {
		want = cpu.KVF32
	}
	// DeepSeek V4's cached row is not a key alone: it carries the
	// compressor's pending projections, which the references hold at f32 and
	// which a softmax over positions pools, so the row is f32 whatever is
	// asked (ds4.go).
	if c.DSV4() {
		want = cpu.KVF32
	}
	// Qwen3.8-Flash-Next's QSA pools its indexer's raw cached keys, which
	// the references hold at f32 (qsa.go).
	if c.QSARatio != 0 {
		want = cpu.KVF32
	}
	if c.Hybrid() {
		// The delta rule's kernel, provisioned per model like the attention
		// ones beside it: the state width is a header field, so it is baked
		// rather than read per head.
		switch {
		case c.ShortConv():
			// A short convolution has no state beside its window.
		case c.Mamba1():
			s.jit.AddSelScan(int(c.SSM.StateSize))
		case c.SSD():
			s.jit.AddSSD(int(c.SSM.StateSize))
		case c.ChanDecay():
			s.jit.AddDeltaChan(int(c.SSM.StateSize))
		default:
			s.jit.AddDelta(int(c.SSM.StateSize))
		}
		g := c.delta()
		s.jit.AddConv1d(g.conv, g.chans)
	}
	// A forced q8 cache on a model that refuses one was refused at Open
	// (Config.KVTypeRefusal); this is the backstop for one forced afterwards.
	if want == cpu.KVQ8 && c.KVTypeRefusal(KVQ8_0) != nil {
		want = cpu.KVF32
	}
	s.jit.AddAttnKV(hdK, hdV, s.kvl.Stride(), want)
	s.kvFmt, s.kvl.fmt = want, want
	if want == cpu.KVQ8 {
		s.kvl.jit = s.jit
		s.jit.ReserveKVQuant(max(c.HeadDim, c.HeadDimSWA))
	}
	s.provisionAttn()
	s.allocDS4()
	s.allocK3()
	s.allocQ4E()
	// After AddAttnKV: kvl.fmt sizes every page.
	s.kv = newKVCacheRange(c, nextCacheID(), nseq, s.kvl, nil, s.m.opt.kvPage, s.lo, s.hi)
	s.kv.usePool(&m.kvPool)
	s.kv.reserve(s.maxSeq)
	// Specialize on the shapes this model actually has. A transformer reuses a
	// handful of (type, K) pairs across all its layers, so this is a few
	// kernels rather than one per layer.
	for i := range m.container.Entries() {
		if t, ok := containerShape(&m.container.Entries()[i]); ok {
			s.jit.AddShape(t.typ, t.k)
		}
	}
	// Read at construction too, because an embedded caller may never call
	// SetMemBudget; it overrides this with the caller's number.
	s.memBudget = sched.MemBudget()
	return s
}

// SetMemBudget tells the session how many bytes of weights may stay resident,
// so it can keep the experts the router keeps asking for and drop the rest. 0
// means track but never evict.
//
// It is callable at any time, which a service needs: tightening the budget
// demotes on the next token, loosening it stops evicting, and the hotness
// counts and resident set are kept either way.
func (s *State) SetMemBudget(bytes uint64) {
	s.memBudget, s.memBudgetAsked = bytes, true
	s.applyMemBudget()
}

// HostBudget is the weight budget actually in force on the host: what
// SetMemBudget was given, less the share of it a device on the host's own memory
// is allowed to fill. It is what a report should print, because it is the number
// the residency manager is working to.
func (s *State) HostBudget() uint64 {
	b, _ := s.hostBudget()
	return b
}

// hostBudget splits the requested budget into the host's share and the device's.
//
// An integrated GPU spends the host's bytes, so counting them twice would
// thrash rather than fail. The tier reports what it holds through HostReserved
// and this subtracts it. It is an optional interface so model/ need not know
// what a backend is; a discrete card leaves the budget unchanged.
func (s *State) hostBudget() (host, device uint64) {
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
func (s *State) applyMemBudget() {
	host, _ := s.hostBudget()
	if s.memBudgetAsked {
		s.initResidency(host) // recomputes the expert share of the new budget
	}
}

// dev is the attached accelerator, or nil. ldCand rather than ld because ld is
// this State's session view (nn.Attacher) and ldCand the device itself, which is
// what answers HostReserved.
func (s *State) dev() nn.LayerDevice {
	if s.ldCand != nil {
		return s.ldCand
	}
	return s.ld
}

// Close releases the generated kernels and detaches the device session. Every
// State must be closed before its Model (`defer m.Close()` then
// `defer st.Close()` gives that order).
func (s *State) Close() error {
	// A move prepared and never committed gives its target's copies back.
	if s.pending != nil {
		s.pending.Cancel()
	}
	if s.expReq != nil {
		close(s.expReq)
		s.expReq = nil
	}
	// The vision State runs on this State's JIT, so it goes first.
	if s.visState != nil {
		s.visState.Close()
		s.visState = nil
	}
	// A vision State's blocks run once per picture, so the card they were
	// placed on gets its room back with the State: its text sessions' blocks
	// run every token. Through the session view, so another session's hold on
	// the same blocks keeps them there.
	if (s.vis != nil || s.enc != nil) && s.ld != nil && s.devCount() > 0 {
		s.SetGPULayers(s.lo)
	}
	// The session's history on the device goes back to the tier, which keeps
	// the model's weights for whoever else is using them.
	if d, ok := s.ld.(nn.Session); ok {
		d.Detach()
	}
	// The history the device gave back was host memory on an integrated GPU,
	// and the model's pages get it back once no State holds the device.
	s.holdHost(nil)
	// A vision State's K/V pair is lent to its first block, so the pool takes
	// it back with the rest of the cache.
	if s.vis != nil && s.vis.kp != nil {
		s.lend(s.lo)
	}
	s.m.kvPool.put(s.kv)
	// The blocks this State ran on the host no longer need their pages for
	// it. Close may be called twice; hostHeld makes the second a no-op.
	if s.hostHeld {
		s.hostHeld = false
		for li := s.lo; li < s.hi; li++ {
			if !s.devAt(li) {
				s.m.hostRuns(li, li+1, -1)
			}
		}
	}
	// An encoder segment's JIT is its Embedder's to close.
	if s.enc != nil {
		return nil
	}
	// A vision State's prompt buffers are the tower's shape, which no text
	// State could take; and a JIT it borrowed is the text State's to close.
	if s.vis != nil {
		s.vis.release()
		if s.vis.borrowed {
			return nil
		}
		return s.jit.Close()
	}
	s.keepSpare()
	return s.jit.Close()
}

// SetDevice attaches an optional accelerator to this State's matvecs. Nil, the
// default, is the CPU-only engine.
//
// A Device that can run whole blocks is offered every layer (see offerRange),
// and what it takes is decided by the device's own memory rather than by a
// separate placement policy.
func (s *State) SetDevice(d nn.Device) error {
	// The bisection knob for a tower (WithTowerGPULayers): how many of its
	// blocks are offered.
	if s.vis != nil && s.m.opt.towerGPULayers >= 0 {
		return s.SetDeviceLayers(d, s.m.opt.towerGPULayers)
	}
	return s.SetDeviceLayers(d, -1)
}

// ropeTabPlanes is one rotary configuration's folded constant planes, or nil
// where there is no such configuration -- a model with one rotary base has no
// local one, and a vision block has no rotary at all.
//
// nil is a real answer: LayerPlan.RopeTab is an alternative source for a table
// the tier is handed anyway, so nil means "upload the host's table".
//
// A multi-axis rotary (M-RoPE) has no planes to offer: a device builds a table
// from a row's CACHE position, and an image moves a row's rotary position off
// it, so the host's per-row table is what goes up (mrope.go).
func ropeTabPlanes(r *nn.Rope, nrot int) []float32 {
	if r == nil || nrot <= 0 || len(r.Runs) > 0 {
		return nil
	}
	return r.TabPlanes(nrot / 2)
}

// SetDeviceLayers is SetDevice with a cap on how many blocks the device is
// offered. -1 offers all of them and lets the device's own memory decide, which
// is the useful default; a positive cap is for pinning the split by hand, and
// zero attaches the device for single matvecs only.
//
// WithPlacement's map is applied here. The error reports a placement that
// named a device not attached, a strict one that could not be met (nothing is
// placed then), and a device this State cannot move onto at all.
func (s *State) SetDeviceLayers(d nn.Device, max int) error {
	// A head-major host cache cannot migrate: MigrateKV's contract is
	// row-major [pos*kvDim + i], and copying would hand the device a
	// permutation of the history.
	if s.kvl.headMajor && d != nil {
		return fmt.Errorf("model: a head-major KV cache cannot move to a device")
	}
	// A binary16 cache cannot migrate either (MigrateKV moves float32, and
	// packed halves would be reinterpreted). The device wins the tie: the
	// cache steps down to f32, which is only possible while the history is
	// empty. A q8_0 cache migrates converted (State.migrateKVAs).
	if s.kvFmt == cpu.KVF16 && d != nil {
		if s.pos != 0 {
			return fmt.Errorf("model: a binary16 KV cache with history cannot move to a device")
		}
		s.stepDownKVToF32()
	}
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
	// A borrowed JIT is the text State's, attached to its device by it.
	if !s.borrowsJIT() {
		s.jit.SetDevice(d)
	}
	s.clearOnDev()
	s.moved = nil
	s.device = d
	// A device on the host's memory is counted against the model's pages from
	// here, for what it holds; once the offers below have placed what they
	// will, both budgets are re-divided for it.
	s.holdHost(d)
	defer s.followHost()
	// The vision State this one encodes pictures in follows it to the device.
	if s.visState != nil {
		if err := s.visState.SetDevice(d); err != nil {
			return err
		}
	}
	// A device the model tells when a frame is refilled may be offered a page's
	// weights (mv); one it cannot tell sees only the dense region.
	s.forgets = d != nil && s.m.watch(d)
	if d == nil {
		// Detaching gives the host budget back (an integrated GPU's share).
		s.setLD(nil)
		s.ldCand = nil
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
		Model: s.m.id,
		NEmbd: c.NEmbd, NHead: c.NHead, NKVHead: c.NKVHead, HeadDim: c.HeadDim,
		NRot: c.NRot, NFFN: c.NFFN, MaxSeq: s.maxSeq, Vocab: c.NVocab, FinalSoftcap: c.FinalSoftcap,
		RMSEps: c.RMSEps, RopeBase: c.RopeBase, Act: c.Act, RopeNeox: c.RopeNeox,
		NoPosEnc: c.NoPosEnc,
		QKNorm:   c.QKNorm, QKNormWide: c.QKNormWide, QKNormPost: c.QKNormPost, NoExpertNorm: c.NoExpertNorm,
		// The tier picks the local or global rotary table per block from
		// this; zero means one kind of layer.
		SWAPeriod: swaPeriodFor(c),
		// The folded constants the host's rotary kernel reads, so the device
		// builds its own table and the two tiers agree by construction.
		RopeTab:    ropeTabPlanes(&s.m.rope, c.NRot),
		RopeTabSWA: ropeTabPlanes(s.m.ropeSWA, c.NRotSWA),
		RopeSplit:  c.RopeXD,
		// The sliding window itself (TestSlidingWindowRunsOnTheDevice).
		SWAWindow: windowFor(c, s.maxSeq),
		SWALocal:  c.periodLocal,
		// DeepSeek V4's model-wide geometry (ds4dev.go).
		DS4: ds4PlanOf(c),
		// Read off the loaded weights, not the architecture name: the
		// post-norms are optional per-layer roles.
		PostNorm:  s.m.hasPostNorm(),
		NoPreNorm: s.m.hasNoPreNorm(),
		// BitNet's inner norms, read off the weights as the post-norms are.
		SubNorm:     s.m.hasSubNorm(),
		AttnSoftcap: c.AttnSoftcap,
		ActWin:      s.jit.ActWindow(),
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
		// DBRX's clamp on q, k and v.
		ClampKQV: c.ClampKQV,
		// Gemma 4's; the block's own geometry is planFor's.
		GeomSplit: c.GeomSplit(), HeadDimSWA: c.HeadDimSWA, NKVHeadSWA: c.NKVHeadSWA,
		NRotSWA: c.NRotSWA, VNorm: c.VNorm,
	}
	s.plan, s.ldCand = p, ld
	hi := s.hi
	if max >= 0 && s.lo+max < hi {
		hi = s.lo + max
	}
	// The device stays attached even when it took nothing, so the seam can
	// grow later. One session per State: a tier offering Attach hands back a
	// view carrying this State's own KV cache (see
	// docs/design/device-sessions.md); two States must not share one cache.
	if a, ok := ld.(nn.Attacher); ok {
		ld = a.Attach()
		// The single matvecs are the session's too: the tier's crossing
		// verdict is sampled per session, and offered through the tier itself
		// they land on its zero session, which outlives every State, so a
		// second State would inherit the first's sample and run the same
		// token with different kernels
		// (TestHybridSecondSessionMatchesTheFirst).
		if !s.borrowsJIT() {
			s.jit.SetDevice(ld)
		}
	}
	// Size the scratch before offering anything: the largest matvec is known
	// from the header, and reserving first never competes with weights for the
	// card's last megabyte.
	rows, k := s.m.maxMatVec()
	ld.Reserve(rows, k)
	// Before any block is offered: a device on host memory shrinks the host's
	// budget by what it already holds (another session's blocks, the scratch
	// reserved above); followHost takes off what the offers add.
	s.setLD(ld)
	s.applyMemBudget()
	// The whole offer sequence is one decision: bracketed, so two States
	// attaching at once do not interleave their offers against one budget.
	if pl, ok := ld.(nn.Placer); ok {
		pl.BeginPlacement()
		// The router spreads a model that fits over its devices instead of
		// filling the first to the brim (tier.GPU.PlanBlocks).
		if pb, ok := ld.(nn.BlockPlanner); ok {
			pb.PlanBlocks(hi, s.planExtra(hi))
		}
		s.placing, s.placeErr = true, nil
		s.autoStreamed = 0
		s.offerRange(s.lo, hi)
		s.placing = false
		pl.EndPlacement()
		// Blocks no card could hold went on streamed by default; whether that
		// beats the host is measured, not assumed (initStreamTrial).
		if s.autoStreamed > 0 && s.seam == nil && !s.m.opt.noStreamTrial {
			s.initStreamTrial()
		}
	} else {
		s.placing, s.placeErr = true, nil
		s.offerRange(s.lo, hi)
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
				// after it, exactly as the grow path does.
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
	if pw, ok := d.(nn.KVPrewarmer); ok && s.devCount() > 0 && !s.m.opt.noKVPrewarm {
		pw.PrewarmKV(min(s.maxSeq, nn.MaxDevicePrefillChunk))
	}
	return nil
}

// reserveRows makes the devices hold every row's history before any is
// written. A batch's rows sit at slot*maxSeq+pos in each block's cache, so it
// needs nseq*maxSeq positions where one sequence grows into maxSeq. The
// capacity is this State's own (a tier session's), so no other State's caches
// grow; what the cards cannot hold comes home for this State alone -- a
// session's release leaves blocks another State is using where they are -- and
// with nothing written yet nothing has to move. A strict placement fails
// instead, since that would move a block it named.
func (s *State) reserveRows() error {
	// A device that pages per sequence takes each row's pages as it writes
	// them; a whole-context reservation is the contiguous cache's.
	if sd, ok := s.ld.(nn.SeqKVDevice); ok && sd.PerSequenceKV() {
		return nil
	}
	total := s.nseq * s.maxSeq
	for s.nseq > 1 && s.devCount() > 0 && !s.ld.ReserveKV(total) {
		li := s.relocationVictim()
		if p := s.m.opt.place; li < 0 || p != nil && p.Strict {
			return fmt.Errorf("model: the devices cannot hold %d rows of %d positions "+
				"for the blocks placed on them", s.nseq, s.maxSeq)
		}
		s.relocateHome(li)
	}
	return nil
}

// placeOf is the placement WithPlacement names for block li.
func (s *State) placeOf(li int) (Place, bool) {
	if p := s.m.opt.place; p != nil {
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
func (s *State) pinnedOnDev(li int) bool {
	pl, ok := s.placeOf(li)
	return ok && pl.Pin && pl.On != "host" && s.devAt(li)
}

// finishPlacement checks a strict placement against where every named block
// and the head landed, undoing all of it when one missed, and pins what the
// map pins.
func (s *State) finishPlacement() error {
	p := s.m.opt.place
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
			return "host"
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
			if li < 0 || li >= len(s.m.layers) {
				s.SetGPULayers(0)
				return fmt.Errorf("model: the placement names block %d of %d", li, len(s.m.layers))
			}
		}
		for li := s.lo; li < s.hi; li++ {
			pl, named := s.placeOf(li)
			if !named {
				continue
			}
			if got := on(li); got != nn.DeviceName(pl.On) && !(pl.On == "host" && got == "host") {
				s.SetGPULayers(0)
				return fmt.Errorf("model: strict placement: block %d is on %s, not %s", li, got, pl.On)
			}
		}
		if p.Head != nil {
			got := "host"
			if s.head != nil {
				got = "a device"
				if hn, ok := s.ld.(nn.NamedDevice); ok {
					if n, ok := hn.HeadDeviceName(); ok {
						got = n
					}
				}
			}
			if got != nn.DeviceName(p.Head.On) && !(p.Head.On == "host" && got == "host") {
				s.SetGPULayers(0)
				return fmt.Errorf("model: strict placement: the head is on %s, not %s", got, p.Head.On)
			}
		}
	}
	// The device's pin is a residency pin -- never a victim -- which a
	// streamed block cannot have; its Pin keeps it on its device through this
	// State's moves instead (pinnedOnDev).
	pin, _ := s.ld.(nn.NamedDevice)
	for li := s.lo; li < s.hi; li++ {
		if pl, named := s.placeOf(li); named && pl.Pin && !pl.Stream && pl.On != "host" && s.devAt(li) && pin != nil {
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
// It outranks a block by reads per resident byte (tier.Place's ranking): the
// head is read in full every token, while a mixture block's expert banks are
// read a fraction of the time. Releasing from the end keeps the placed blocks a
// contiguous prefix, and a trailing block is the cheapest to give up.
func (s *State) placeHead() {
	// WithHeadPlacement(false) leaves the projection on the host -- the other
	// arm of the placement comparison, and the way to bisect a divergence to it.
	if s.ld == nil || s.head != nil || s.outW.data == nil || !s.m.opt.headPlace {
		return
	}
	// The placement's head, when it names one.
	if p := s.m.opt.place; p != nil && p.Head != nil {
		if p.Head.On == "host" {
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
	if hb := s.m.headReadBytes(); hb > seamCrossBytes {
		budget = hb - seamCrossBytes
	}
	// What is released is restored when the head still does not fit: the
	// budget is in bytes read, while the constraint is VRAM space, and the
	// loop cannot tell them apart. PrepHead is asked rather than the space
	// predicted, because the device's accounting has slack and fragmentation.
	was := s.gpuLayers
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
		// Only buy room when the device is already partial. With every block
		// resident the head is pure upside and a refusal means the card is
		// simply full; giving up a block there would trade a whole block's
		// bytes for the head's and add a seam that does not exist.
		if s.gpuLayers >= s.hi || s.gpuLayers == s.lo+1 {
			restore()
			return
		}
		cost := s.m.blockReadBytes(s.gpuLayers - 1)
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

// newHead is the output projection as a device takes it.
func (s *State) newHead() *nn.Head {
	c := s.c
	h := &nn.Head{Model: s.m.id, Norm: s.outNorm, W: wt(*s.outW), Logits: s.logits,
		Softcap: c.FinalSoftcap, Embeds: c.TiedEmbd, EmbdScale: float32(c.EmbdScale)}
	// The final softcap rides the head: a tier that takes it applies the cap on
	// the device (nn.Head.Softcap), which is what lets a gemma2 token end in
	// the device argmax instead of a 256,000-float readback.
	// A LayerNorm head (C6) carries its bias, zeros where the file has none.
	// The head's own bias (phi-2's output.bias) is added by finishLogits on
	// the host after either arm, so the device never sees it.
	if c.LayerNorm {
		h.NormB = s.m.lnBias(s.m.outNormB)
	}
	return h
}

// placeHeadOn puts the projection on the named device, as WithPlacement asks.
func (s *State) placeHeadOn(name string) bool {
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

// layerWeightsAt gathers block li's weights as the device wants them. It does
// not set Ensure -- the caller does, because Ensure calls back into here. It is
// shared by the offer and by every device page-in, which must agree.
func (s *State) layerWeightsAt(li int) nn.LayerWeights {
	if s.vis != nil {
		return s.visWeights(li)
	}
	if s.enc != nil {
		return s.encWeights(li)
	}
	l := &s.m.layers[li]
	qn, kn := s.qkNorms(li)
	w := nn.LayerWeights{
		AttnNorm: l.attnNorm, FFNNorm: l.ffnNorm,
		PostAttnNorm: l.postAttnNorm, PostFFNNorm: l.postFFNNorm,
		AttnSubNorm: l.attnSubNorm, FFNSubNorm: l.ffnSubNorm,
		QNorm: qn, KNorm: kn,
		Bq: l.bq, Bk: l.bk, Bv: l.bv, Bo: l.bo,
		Wq: wt(l.wq), Wk: wt(l.wk), Wv: wt(l.wv), Wo: wt(l.wo),
		// For a MoE file these three are the banks -- l.gate is the 3-D
		// ffn_gate_exps and l.experts[e] are views into it, so
		// nothing is copied and the device indexes the same mapping.
		Gate: s.m.bankWeight(l.gate), Up: s.m.bankWeight(l.up), Down: s.m.bankWeight(l.down),
	}
	w.Sinks = l.sinks
	// The classic block's vectors (C6). A LayerNorm always hands the device
	// a bias -- zeros where the file has none (command-r) -- because the apply
	// kernel takes one; an ungated FFN's biases ride up and down.
	w.BUp, w.BDown = l.upB, l.downB
	w.XIELU = l.xielu
	if s.c.LayerNorm {
		w.AttnNormB = s.m.lnBias(l.attnNormB)
		if l.ffnNorm != nil {
			w.FFNNormB = s.m.lnBias(l.ffnNormB)
		}
	}
	if s.c.MoE() {
		w.Router = wt(l.router)
		w.RouterB, w.ExpGateB, w.ExpUpB, w.ExpDownB = l.routerB, l.expGateB, l.expUpB, l.expDownB
		w.ExpSelB = l.expProbsB
		w.FFNNorm2, w.PostFFNNorm1, w.PostFFNNorm2 = l.ffnNorm2, l.postFFNNorm1, l.postFFNNorm2
		w.RouterNorm, w.ExpScale = l.routerNorm, l.expScale
		// Keyed on the shared expert, not its gate: DeepSeek's always-on
		// expert has no gate (shRouter) at all.
		if l.shGate.rows != 0 || l.shUp.rows != 0 {
			w.ShGate, w.ShUp, w.ShDown = wt(l.shGate), wt(l.shUp), wt(l.shDown)
			w.ShRouter = l.shRouter
		}
	}
	if s.c.PLEDim != 0 {
		w.PLEGate, w.PLEProj, w.PLEPost = wt(l.pleGate), wt(l.pleProj), l.plePost
		w.AltRouter, w.AltRouterNorm, w.AltPredT, w.AltCorrT, w.AltCorrScale =
			wt(l.altRouter), l.altRouterNorm, l.altPredT, l.altCorrT, l.altCorrScale
		w.LaurelL, w.LaurelR, w.LaurelPost = wt(l.laurelL), wt(l.laurelR), l.laurelPost
	}
	if s.c.LayerKind(li).Recurrent() {
		w.SSM = nn.SSMWeights{
			Gate: wt(l.ssmGate), BA: wt(l.ssmBA), Out: wt(l.ssmOut),
			Conv1d: l.ssmConv1d, A: l.ssmA, DtBias: l.ssmDtBias, Norm: l.ssmNorm,
			FA: wt(l.ssmFA), FB: wt(l.ssmFB), GA: wt(l.ssmGA), GB: wt(l.ssmGB),
			D: l.ssmD, ConvBias: l.ssmConvB, In: wt(l.ssmIn),
		}
		if s.m.Cfg.Mamba1() {
			w.SSM.FA, w.SSM.FB, w.SSM.GA, w.SSM.GB = wt(l.ssmXDt), wt(l.ssmDtProj), wt(l.ssmXB), wt(l.ssmXC)
			w.SSM.DtNorm, w.SSM.BNorm, w.SSM.CNorm = l.ssmDtNorm, l.ssmBNorm, l.ssmCNorm
		}
	}
	// MLA's five matrices and two norms. Wk and Wv stay zero-valued (MLA has
	// no k or v projection; the tier skips a 0x0 weight). The per-head views
	// are not carried: wkb and wvb are the banks, indexed by head on the
	// device as an expert bank is by expert.
	if s.c.MLA() {
		w.Wqa, w.Wqb = wt(l.wqa), wt(l.wqb)
		w.Wkva, w.Wkb, w.Wvb = wt(l.wkva), wt(l.wkb), wt(l.wvb)
		w.QANorm, w.KVANorm = l.qaNorm, l.kvaNorm
		w.IdxQB, w.IdxK, w.IdxProj = wt(l.idxQB), wt(l.idxK), wt(l.idxProj)
		w.IdxKNorm, w.IdxKNormB = l.idxKNorm, l.idxKNormB
	}
	if s.m.Cfg.MSAAt(li) {
		w.IdxQ, w.IdxK = wt(l.idxQ), wt(l.idxK)
		w.IdxQNorm, w.IdxKNorm = l.idxQNorm, l.idxKNorm
	}
	s.ds4Weights(li, &w)
	s.q4eWeights(li, &w)
	w.ResAttn, w.ResFFN = l.resAttn, l.resFFN
	w.MLAGate, w.RoutedDown, w.RoutedUp, w.RoutedNorm = wt(l.mlaGate), wt(l.routedDown), wt(l.routedUp), l.routedNorm
	return w
}

// ensureLayer builds the callback a device calls before reading block li's
// bytes: the page back in, the routed bank read, and every Weight in the
// caller's struct re-pointed at where those bytes are now. The re-pointing
// matters because a host page is a reused frame: slices kept from placement
// would read whichever block the host faulted into it last. pageIn comes first
// because ensureBlockExperts reads into the page.
func (s *State) ensureLayer(li int) func(*nn.LayerWeights) error {
	return func(dst *nn.LayerWeights) error {
		if err := s.m.pageIn(li); err != nil {
			return err
		}
		if s.c.MoE() {
			if err := s.m.ensureBlockExperts(li); err != nil {
				return err
			}
		}
		w := s.layerWeightsAt(li)
		// Carry every callback over: a field missed here is silently cleared
		// by the first Ensure (a missed PrefetchExperts disables the streamed
		// read/upload pipeline).
		w.Ensure, w.EnsureExperts = dst.Ensure, dst.EnsureExperts
		w.PrefetchExperts, w.HostExperts, w.HostExpertsRows = dst.PrefetchExperts, dst.HostExperts, dst.HostExpertsRows
		*dst = w
		return nil
	}
}

// ensureSelected is ensureLayer for a caller that knows which experts it wants.
// Same three steps -- page in, read, re-point -- with the bank read narrowed to
// the routed sheets. See nn.LayerWeights.EnsureExperts.
func (s *State) ensureSelected(li int) func(*nn.LayerWeights, []uint32) error {
	return func(dst *nn.LayerWeights, sel []uint32) error {
		// The device's callback: a placed block's own weights are on the card,
		// so pageInSelected reads only the routed sheets. See selectedRanges.
		if s.c.MoE() {
			if err := s.m.pageInSelected(li, sel); err != nil {
				return err
			}
		} else if err := s.m.pageIn(li); err != nil {
			return err
		}
		w := s.layerWeightsAt(li)
		// Carry every callback over; see ensureLayer.
		w.Ensure, w.EnsureExperts = dst.Ensure, dst.EnsureExperts
		w.PrefetchExperts, w.HostExperts = dst.PrefetchExperts, dst.HostExperts
		*dst = w
		return nil
	}
}

// prefetchSelected is ensureSelected's read and nothing else, for a caller that
// wants the next group of sheets on their way while it uploads the last.
//
// It must not bind (so it is not pageInSelected): it runs on another goroutine
// while the tier uses m.layers[li]'s spans. EnsureRanges is safe because
// jlm.File locks and makeRoom never evicts the block being faulted in.
func (s *State) prefetchSelected(li int) func([]uint32) error {
	return func(sel []uint32) error {
		if !s.c.MoE() {
			return nil
		}
		return s.m.ensureBlockSelected(li, sel, false)
	}
}

// offerRange offers blocks [lo, hi) to the device in order. A declined block
// stays on the host and the rest are still offered (placement is a set; see
// onDev); the seam is the prefix of what was accepted. A tier that cannot run a
// block says so by name through DeclinePlan.
func (s *State) offerRange(lo, hi int) {
	ld := s.ldCand
	if ld == nil {
		return
	}
	// Offers go through this State's own view (tier.GPU.Attach), which makes it
	// the device's current session: an offer is where a block takes the session's
	// history (KV pages, a linear block's recurrent state), filed under whichever
	// session is current. Offered through the shared tier, a block took it for
	// the last session to make any call -- with two States placing at once, the
	// other one -- and this State's first token found no recurrent state and
	// demoted every block to the host.
	adm := s.ld
	if adm == nil {
		adm = ld
	}
	// Each placed block gives its host pages back and the next block's land
	// in the same frames; once the range is offered nothing will take the rest,
	// so they go back to the OS (jlm.File.TrimFree).
	defer s.m.container.TrimFree()
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
		if named && place.On == "host" && (s.placing || place.Pin) {
			continue
		}
		// The block is read here, for the device, and its host page released
		// once it is on the card: one block in flight, not the whole model.
		// Ask before reading: a graph decline is a property of the plan, so a
		// block the device will refuse costs no read.
		if d, ok := ld.(nn.PlanDecliner); ok {
			if why := d.DeclinePlan(s.planFor(li)); why != "" {
				s.noteDecline(why)
				continue
			}
		}
		// Where the block's routed experts run, when a placement or the caller
		// says: set before the size check, so a block placed with its experts
		// off the card is not refused for a bank it will not hold.
		if ep, ok := ld.(nn.ExpertPlacer); ok && c.MoEAt(li) {
			where := s.m.opt.experts
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
			total, bank := s.m.blockBytes(li)
			if why := d.DeclineSize(li, total, bank); why != "" {
				// A mixture block too big for any card resident is streamed
				// there instead when the device offers it: the base on the
				// card, the routed experts sent per token. A placement that
				// keeps the block home is honoured above, and the device can
				// be told not to (tier.Config.NoAutoStream).
				as, ok := ld.(nn.AutoStreamer)
				if !ok || bank == 0 || !as.AutoStream(li, total, bank, c.NExpert, c.NExpertUsed, c.NLayer, s.m.meanBase()) {
					s.noteDecline(why)
					continue
				}
				s.autoStreamed++
			}
		}
		// A block whose experts are separate tensors has no bank to offer; say
		// why rather than let the tier report zero-value weights. See
		// convert.stackGGUFExperts.
		if s.m.layers[li].legacyExperts {
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
		if bh, ok := ld.(nn.BlockHolder); ok && !named && bh.HoldsBlock(li) {
			w := nn.LayerWeights{Ensure: s.ensureLayer(li), EnsureExperts: s.ensureSelected(li),
				PrefetchExperts: s.prefetchSelected(li), HostExperts: s.hostExpertsFor(li),
				HostExpertsRows: s.hostExpertsRowsFor(li)}
			plan := *s.planFor(li)
			if adm.PrepLayer(li, &plan, &w) {
				s.markOnDev(li)
				continue
			}
		}
		plan, w, err := s.blockOffer(li)
		if err != nil {
			return
		}
		took := false
		if named && place.On != "host" && s.placing {
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
			if !took && !s.m.opt.place.Strict {
				s.noteDecline(fmt.Sprintf("%s declined block %d, which the placement names; "+
					"the engine placed it instead", place.On, li))
				took = adm.PrepLayer(li, &plan, &w)
			}
		} else {
			took = adm.PrepLayer(li, &plan, &w)
		}
		// A mixture block refused for room on every card goes on streamed,
		// its base on a card and its experts off it, rather than to the host:
		// gpt-oss-20b on a 4 GB card held 2 of 24 blocks resident and decoded
		// at 4.84 tok/s, and with all 24 streamed and their experts on the
		// host at 9.56 (placement.md 16c-3). The stream trial measures it
		// against the host like any auto-streamed placement.
		if !took && c.MoEAt(li) && !(named && place.On == "host") {
			if as, ok := ld.(nn.AutoStreamer); ok {
				total, bank := s.m.blockBytes(li)
				if bank > 0 && as.AutoStream(li, total, bank, c.NExpert, c.NExpertUsed, c.NLayer, s.m.meanBase()) {
					if took = adm.PrepLayer(li, &plan, &w); took {
						s.autoStreamed++
					} else if ep, ok := ld.(nn.ExpertPlacer); ok && !(named && place.Experts != "") && s.m.opt.experts == "" {
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

// releasePlacedPage gives back the host page of block li once a device holds
// it, or tells the devices to drop their lone-matvec copies of a page that
// stays. offerRange and MoveBlocks share it.
func (s *State) releasePlacedPage(ld nn.LayerDevice, li int) {
	l := &s.m.layers[li]
	// The block is on the card, so the host gives up its page; pageIn
	// re-reads it if the seam later moves the block back. Asked, not
	// assumed: a unified device's buffers alias host memory.
	// A prediction block keeps its page: its eh_proj runs on the host every
	// draft step (Speculator.draftRows), and re-reading the page for it
	// would cost a block of I/O a step.
	if r, ok := ld.(nn.HostPageReleaser); ok && r.ReleasesHostPage(li) && l.mtp == nil {
		// Under bind, as pageIn binds: another State may be binding or
		// reading this block's spans while this one places it. A State
		// that runs the block on the host keeps it bound (hostUse): it
		// faults the block in once and then reads its spans unlocked
		// through every matvec of the block.
		s.m.bind.Lock()
		if s.m.hostUse[li] == 0 {
			keep := false
			if eh, ok := ld.(nn.ExpertHolder); ok {
				keep = !eh.HoldsExperts(li)
			}
			releaseLayer(s.m.container, li, l, keep)
		}
		s.m.bind.Unlock()
	} else if buf := s.m.container.Page(li); len(buf) > 0 {
		// The page stays (a streamed block's host side reads it), so no
		// recycle tells the devices to drop the lone-matvec copies they
		// made of its weights while the host ran it; the block is placed
		// now and they are dead weight. Said here instead.
		s.m.forgetCopies(unsafe.Pointer(&buf[0]), uintptr(len(buf)))
	}
}

// blockOffer reads block li's page and gathers what a device is offered for
// it: the block's plan and its weights with every callback. offerRange and
// MoveBlocks share it, so a block moved between devices is offered exactly as
// a placement offers it.
func (s *State) blockOffer(li int) (nn.LayerPlan, nn.LayerWeights, error) {
	w, err := s.blockWeights(li)
	if err != nil {
		return nn.LayerPlan{}, nn.LayerWeights{}, err
	}
	// The block is offered and the device decides (PrepLayer). Wq carries
	// the fused q|k|v projection for a linear block. planFor, not s.plan:
	// it adds what this block is (its kind, sinks, mixture biases).
	return *s.planFor(li), w, nil
}

// blockWeights is blockOffer's weights: the page read and every callback set.
// It reads the model and nothing of the State's placement, so a move's
// staging runs it beside the State's own tokens (PendingMove).
func (s *State) blockWeights(li int) (nn.LayerWeights, error) {
	if err := s.m.pageIn(li); err != nil {
		return nn.LayerWeights{}, err
	}
	c := s.c
	l := &s.m.layers[li]
	w := s.layerWeightsAt(li)
	// Every block, not only a mixture's: a device pager needs the host
	// page re-read on every swap.
	w.Ensure, w.EnsureExperts = s.ensureLayer(li), s.ensureSelected(li)
	w.PrefetchExperts = s.prefetchSelected(li)
	w.HostExperts, w.HostExpertsRows = s.hostExpertsFor(li), s.hostExpertsRowsFor(li)
	if c.MoE() {
		// The expert banks are fetched by the tier through Ensure once it
		// admits the block: pageIn does not read them (the host fetches
		// only the experts it routes to), and reading here would spend a
		// bank on every block the budget then refuses.
		w.Router = wt(l.router)
		// The expert, not its gate; see the twin above.
		if l.shGate.rows != 0 || l.shUp.rows != 0 {
			w.ShGate, w.ShUp, w.ShDown = wt(l.shGate), wt(l.shUp), wt(l.shDown)
			w.ShRouter = l.shRouter
		}
	}
	if k := c.LayerKind(li); k.Recurrent() && !k.Attends() {
		// A linear block has none of these; leaving the attention four set
		// would offer a device four matrices whose shapes do not match the
		// plan it was handed. (planFor clears AttnOutGate for the same
		// reason: s.plan sets it on every block of a qwen3next.)
		w.Wk, w.Wv, w.Wo = nn.Weight{}, nn.Weight{}, nn.Weight{}
		w.QNorm, w.KNorm = nil, nil
	}
	return w, nil
}

// planExtra is what a placement of blocks [0,hi) must leave room for besides
// their weights (nn.BlockPlanner): the history this State's context will hold
// on its attention blocks, counted as float32 K and V (MLA's latent row), and
// the output projection. A recurrent block's state is fixed and small.
func (s *State) planExtra(hi int) uint64 {
	c := s.c
	hdK, hdV := c.attnWidths()
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
	return n*perPos*uint64(s.maxSeq)*uint64(max(s.nseq, 1)) + s.m.headReadBytes()
}

// planFor is the block's plan with no weights in it: s.plan adjusted for what
// this block is. Nothing here reads the block's page, so nn.PlanDecliner can
// answer before the page-in.
func (s *State) planFor(li int) *nn.LayerPlan {
	c := s.c
	plan := s.plan
	if s.vis != nil {
		return s.visPlan(li)
	}
	if s.enc != nil {
		return s.encPlan(li)
	}
	// The sinks and the mixture biases are resident vectors read at build(),
	// not page bytes, so asking about them here still needs no page-in.
	l := &s.m.layers[li]
	plan.AttnSinks = l.sinks != nil
	plan.MoEBias = l.routerB != nil || l.expGateB != nil || l.expUpB != nil || l.expDownB != nil
	// An ungated FFN has no gate tensor, which build() recorded without a
	// page-in: the entry is bound at load whether or not the block is resident.
	plan.UngatedFFN = l.gate.e == nil && (l.router.e == nil || l.ungatedExp)
	plan.NoFFN = l.noFFN
	// AttnScale is carried for every architecture: it includes YaRN's
	// magnitude correction (DeepSeek), and under MLA HeadDim is the wrong
	// width to derive it from. MLA's geometry is here so a tier can decline.
	plan.AttnScale = c.AttnScale
	// The block's own attention geometry (Gemma 4's sliding layers differ
	// from its global ones), its v (no projection of its own) and its output
	// scalar.
	plan.HeadDim, plan.NKVHead, plan.NRot = c.HeadDimAt(li), c.NKVHeadAt(li), c.NRotAt(li)
	plan.Sliding = c.GeomSplit() && c.SWA(li)
	plan.BidirOff = c.BidirSWA && !c.SWA(li)
	plan.VFromK, plan.OutScale = l.vFromK, l.outScale
	plan.DenseMoE = c.DenseMoE && c.MoEAt(li)
	plan.PLEDim, plan.PLEWidth = c.PLEDim, c.NLayer*c.PLEDim
	plan.IdxHeads, plan.IdxHeadDim, plan.IdxTopK = c.IdxHeads, c.IdxHeadDim, c.IdxTopK
	plan.IdxBlock, plan.IdxLocal = c.IdxBlock, c.IdxLocal
	// MiniMax-M3's dense lead selects nothing: its row has no indexer head.
	if c.MSA() && !c.MSAAt(li) {
		plan.IdxHeads, plan.IdxHeadDim, plan.IdxTopK, plan.IdxBlock, plan.IdxLocal = 0, 0, 0, 0, 0
	}
	plan.AltUp, plan.Sparse, plan.SparseStd = c.AltUp, li < c.NSparse, c.SparseStd
	plan.LaurelRank = l.laurelL.rows
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
	plan.ExpertSelBias = s.m.expSelBias() != nil
	plan.ExpertSparseMixer = c.SparseMixer
	s.ds4Plan(li, &plan)
	if c.LayerKind(li).Recurrent() {
		g := c.delta()
		plan.Recurrent = nn.RecurrentPlan{
			Conv: g.conv, Chans: g.chans, KHeads: g.kHeads, VHeads: g.vHeads,
			KDim: g.kDim, VDim: g.vDim,
			StateLen: g.stateStride(), ConvState: g.convStateLen(),
			ChanDecay: c.ChanDecay(), KeyTiled: g.tiled,
		}
		if g.ssd {
			plan.Recurrent.SSD, plan.Recurrent.NormGroups = true, c.SSMNormGroups
			plan.Recurrent.NormBeforeGate, plan.Recurrent.NoNorm = c.SSMNormBeforeGate, l.ssmNorm == nil
		}
		// A linear block has no attention, so no out-gate.
		plan.AttnOutGate = false
		plan.Recurrent.WithAttn = c.LayerKind(li).Attends()
		plan.Recurrent.ShortConv = c.ShortConv()
		if g.mamba1 {
			// No gated norm: the block's output is y*silu(z) straight to ssm_out.
			plan.Recurrent.Mamba1, plan.Recurrent.Rank, plan.Recurrent.NoNorm = true, g.rank, true
		}
	}
	s.k3Plan(li, &plan)
	s.q4ePlan(li, &plan)
	return &plan
}

// markOnDev records that a device accepted block li, and recomputes gpuLayers
// as the contiguous placed prefix (with holes, "li + 1" is not a length).
func (s *State) markOnDev(li int) {
	if s.onDev == nil {
		s.onDev = make([]bool, len(s.m.layers))
	}
	if !s.onDev[li] {
		s.hostRun(li, -1)
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
func (s *State) unmarkOnDev(li int) {
	if s.onDev[li] {
		s.hostRun(li, 1)
	}
	s.onDev[li] = false
	s.gpuLayers = min(s.gpuLayers, li)
}

// placedFrom lists the placed blocks at index lo and above.
func (s *State) placedFrom(lo int) []int {
	var out []int
	for li := lo; li < len(s.onDev); li++ {
		if s.onDev[li] {
			out = append(out, li)
		}
	}
	return out
}

// clearOnDev forgets the whole placement.
func (s *State) clearOnDev() {
	for i := range s.onDev {
		if s.onDev[i] {
			s.hostRun(i, 1)
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
func (s *State) shrinkOnDev(n int) {
	for i := n; i < len(s.onDev); i++ {
		if s.onDev[i] {
			s.hostRun(i, 1)
		}
		s.onDev[i] = false
	}
	if s.gpuLayers > n {
		s.gpuLayers = n
	}
}

// holdHostRuns counts this State in m.hostUse for every block of [lo, hi): a
// new State runs them all on the host.
func (s *State) holdHostRuns() {
	s.m.hostRuns(s.lo, s.hi, 1)
	s.hostHeld = true
}

// hostRun moves this State's count on block li in m.hostUse by d, as the
// block leaves (-1) or returns to (+1) the host for it.
func (s *State) hostRun(li, d int) {
	if s.hostHeld && li >= s.lo && li < s.hi {
		s.m.hostRuns(li, li+1, int32(d))
	}
}

// devAt reports whether block li runs on a device.
func (s *State) devAt(li int) bool { return li < len(s.onDev) && s.onDev[li] }

// devRuns is the placement as maximal contiguous runs, the unit
// nn.LayerDevice.Layers takes. Each run is one seam crossing (the residual goes
// up and comes back), so alternating tiers is priced rather than refused.
func (s *State) devRuns() [][2]int {
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
func (s *State) devCount() int {
	n := 0
	for _, v := range s.onDev {
		if v {
			n++
		}
	}
	return n
}

// prepDeviceRange grows the device's prefix to n blocks.
func (s *State) prepDeviceRange(lo, n int) {
	// A binary16 host cache cannot migrate to the device (packed halves read
	// as float32). SetDeviceLayers steps it down first, so this is a
	// backstop.
	if s.kvFmt == cpu.KVF16 {
		return
	}
	s.offerRange(lo, n)
	s.placeHead()
}

// embed writes the token's embedding into the residual stream.
func (s *State) embed(token int32) error {
	m, c := s.m, s.c
	// On the host, unlike a prompt's rows: a device gather here measured
	// slower (one row is usually cache-hot, and the launch sits on the
	// token's critical path).
	if err := m.embedRow(s.x, int(token)); err != nil {
		return err
	}
	if c.EmbdScale != 1 {
		s.scale(s.x, float32(c.EmbdScale))
	}
	m.trace(-1, "embd", s.x)
	if c.PLEDim != 0 {
		return s.pleInputs(s.ple, token, s.x)
	}
	return nil
}

// SetDeviceFallback decides what happens when the device fails mid-sequence:
// bring the history home and finish on the host (true, the default), or return
// an error (false).
//
// A service would rather serve slowly than fail; a benchmark wants the error,
// since a run silently finished on the CPU produces a meaningless number.
func (s *State) SetDeviceFallback(on bool) { s.devFallback = on }

// demoteAll brings every device block's history home and hands the blocks back.
//
// The history moves before the blocks do: releasing first would free the only
// copy of their keys and values.
func (s *State) demoteAll() bool {
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
	return true
}

// DeviceDemotions is how many times the device failed and its blocks were moved
// to the host. Non-zero means this session's rate is not comparable with one
// that kept the card.
func (s *State) DeviceDemotions() int { return s.devDemoted }

// DeviceRowChunks is how many prompt chunks the device refused as one batch
// and ran again one row at a time. A batched-path gate reads it: a chunk the
// device ran whole and then refused (at its head, say) is re-run a row at a
// time, and its logits are then the row-by-row arm's, whatever the batched
// kernels computed.
func (s *State) DeviceRowChunks() int { return s.devRowChunks }

// TailRowChunks is how many few-row prompt chunks this session ran on the
// device as one ragged step of their rows rather than as a batched chunk
// (tailrows.go).
func (s *State) TailRowChunks() int { return s.tailRowChunks }

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
func (s *State) SetRelocate(on bool) { s.relocate = on }

// Relocating reports whether SetRelocate is on.
func (s *State) Relocating() bool { return s.relocate }

// Relocations is how many blocks SetRelocate has handed to the host.
func (s *State) Relocations() int { return s.devRelocated }

// Spills is how many blocks moved onto a later device because the one holding
// them could not grow its history (tier.GPU.SpillAfter).
func (s *State) Spills() int { return s.devSpilled }

// Reclaims is how many of those the device has since taken back.
func (s *State) Reclaims() int { return s.devReclaimed }

// relocateFor is SetRelocate's pre-flight: while the device cannot hold `pos`
// positions of history, a block comes home -- the highest one on the device
// that refused.
//
// one block per ask. The device refuses before it touches anything
// (devTier.regrowKV) and grows to whatever fits before refusing at all
// (devTier.kvFit), so an ask is cheap and a block moves only when not even one
// more page fits.
func (s *State) relocateFor(pos int) {
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
func (s *State) relocateUntil(pos int, fits func() bool) {
	// Before and after: a block a relocation takes home may part a KV-sharing
	// group (fixKVGroups).
	s.fixKVGroups()
	defer s.fixKVGroups()
	sp := s.ldSpill
	if !s.relocate && sp == nil {
		return
	}
	if s.relocate {
		s.reclaim(pos)
	}
	defer func() { s.roomSeen = s.roomGen() }() // our own moves are not news
	for s.devCount() > 0 && !fits() {
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
		// crossing. See tier.GPU.SpillAfter.
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
		if !s.relocate && !spilled {
			return
		}
		s.relocateHome(li)
	}
}

// relocateHome brings block li home for this State and remembers it, so that
// reclaim offers it back once the device reports room. The block's history
// must already be home (or never have been written).
func (s *State) relocateHome(li int) {
	s.ld.ReleaseLayers(li, li+1)
	s.unmarkOnDev(li)
	// -gpu-grow would adopt the block straight back next token.
	s.gpuTarget = min(s.gpuTarget, s.gpuLayers)
	if s.moved == nil {
		s.moved = make([]bool, len(s.m.layers))
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
func (s *State) reclaim(pos int) {
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
func (s *State) relocateFresh(total int) {
	if !s.relocate || s.ld == nil || s.nseq != 1 || s.pos != 0 || !slices.Contains(s.moved, true) {
		return
	}
	if t, ok := s.ld.(nn.KVTrimmer); ok {
		t.TrimKV(total)
	}
	s.relocateFor(total)
}

// roomGen is the device's room report, or 0 from a device that makes none.
func (s *State) roomGen() uint64 {
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
func (s *State) relocationVictim() int {
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

// splitBankData is the Data of a bank that lives in expert pages. It is not
// the weight's bytes -- there is no one span of them -- and it exists because
// every consumer of an nn.Weight reads an empty Data as "no weight here".
var splitBankData = []byte{0}

// bankWeight is wt for a mixture's bank. A bank in expert pages has no span to
// hand over, so the device gets Sheet instead: expert x's planes, held in their
// page until the device has copied them. A float bank -- which the device reads
// row-major out of Data -- is gathered into one buffer, since that path has no
// Sheet; only the synthetic fixtures have one.
func (m *Model) bankWeight(t tensor) nn.Weight {
	c := m.container
	if c == nil || t.e == nil || !c.ExpertPaged(t.e) {
		return wt(t)
	}
	sheets := int(t.e.Dims[2])
	nrow := t.rows / max(sheets, 1)
	sheet := func(x int) (nn.Packed, func(), error) {
		p := c.ExpertPage(t.e, x)
		if p < 0 {
			return nn.Packed{}, func() {}, fmt.Errorf("model: %v expert %d has no expert page", t.e.Role, x)
		}
		// The block's own page is held too: a device assembles a bank one
		// expert page at a time, each fault can evict, and the block page holds
		// the router and shared expert the same upload reads.
		var lb jlm.Lease
		if b := int(t.e.Block); b >= 0 && b < int(c.H.NBlocks) {
			// Held only when in, as ensureExperts does: claiming the
			// frame of a block whose base is on the card held its size of
			// host budget empty for every sheet sent.
			if l, ok := c.HoldFilled(b); ok {
				lb = l
			}
		}
		off, size := c.PageBounds(p)
		l, err := c.Hold(p, []jlm.Range{{Off: off, N: size}})
		if err != nil {
			lb.Release()
			return nn.Packed{}, func() {}, err
		}
		qs, d, sc := c.Sheet(t.e, x)
		return nn.Packed{QS: qs, D: d, SC: sc, Stride: nrow}, func() { l.Release(); lb.Release() }, nil
	}
	if t.packed == nil {
		var buf []byte
		for x := 0; x < sheets; x++ {
			p, release, err := sheet(x)
			if err != nil {
				release()
				return wt(tensor{})
			}
			buf = append(buf, p.QS...)
			release()
		}
		return nn.Weight{T: t.typ, Data: buf, Rows: t.rows, K: t.k}
	}
	return nn.Weight{T: t.typ, Data: splitBankData, Rows: t.rows, K: t.k,
		Packed: &nn.Packed{Sheet: sheet, Sheets: sheets}}
}

func wt(t tensor) nn.Weight {
	return nn.Weight{T: t.typ, Data: t.data, Rows: t.rows, K: t.k, Packed: t.packed}
}

// SetHeadOnDevice chooses whether the output norm and the vocabulary projection
// run in the same submission as the blocks, so a paired A/B can alternate the
// two in one process; the device keeps the weights resident either way.
func (s *State) SetHeadOnDevice(on bool) {
	if on {
		s.head = s.headReady
		return
	}
	s.head = nil
}

// HeadOnDevice reports whether the device is running the token's tail.
func (s *State) HeadOnDevice() bool { return s.head != nil }

// AttnChunk is how many query heads one pool task claims in decode attention,
// and SetAttnChunk sets it. Zero means the shipping default of one.
func (s *State) AttnChunk() int {
	if s.attnChunk < 1 {
		return 1
	}
	return s.attnChunk
}

// SetAttnChunk sets how many query heads one pool task claims in decode
// attention (see AttnChunk); below one means the default.
func (s *State) SetAttnChunk(n int) { s.attnChunk = n }

// The KV layout and width are per model (modelOpts), set before the States
// that use them are created, since the attention kernels bake the stride.

// SetKVF16 forces the cache width for States this model creates afterwards,
// overriding what KVWidthPaysOff decides. For the A/B harness and the gates.
func (m *Model) SetKVF16(on bool) {
	m.opt.kvType, m.opt.kvTypeSet = KVF32, true
	if on {
		m.opt.kvType = KVF16
	}
}

// ClearKVF16 hands the decision back to the tuner.
func (m *Model) ClearKVF16() { m.opt.kvTypeSet = false }

// KVIsF16 reports whether this state's cache is binary16. It can be false with
// the width forced to f16 (WithKVF16), when the model's head dimension has no
// generated attention kernel and the f32 fallbacks have to be able to read the
// cache.
func (s *State) KVIsF16() bool { return s.kvFmt == cpu.KVF16 }

// attnPairShipped is what ships: share both the K walk and the V walk between
// each pair of query heads. It is neutral at shallow depth and a win where
// attention dominates (see docs/engineering-history/cpu-kernels.md).
// WithAttnPair overrides it per model, so a scoreboard row can attribute it.
const attnPairShipped = 3

// AttnPair selects which paired attention kernels decode uses: 1 the scores, 2
// the accumulation, 3 both. The halves are separable because they walk
// different arrays (K and V) with different register costs. Zero means unset
// and takes the default; a negative value means explicitly none.
func (s *State) AttnPair() int {
	switch {
	case s.attnPair == 0:
		return s.m.opt.attnPair
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
func (s *State) AttnPaired() int64 { return s.attnPaired.Load() }

// SetAttnPair selects this State's paired attention kernels, as AttnPair
// reads them: 1 the scores, 2 the accumulation, 3 both, 0 the model's
// default, negative none.
func (s *State) SetAttnPair(v int) { s.attnPair = v }

// rmsnorm runs the generated norm.
func (s *State) rmsnorm(y, x, w []float32, eps float64) { nn.RMSNorm32JIT(y, x, w, eps) }

// actmulAll applies the activation across a whole FFN width, on the pool.
//
// The split is by vector: the ragged remainder rides with the last chunk, and
// the tail is bit-identical to the vector body (TestElementwiseEveryWidth), so
// Forward and Prefill compute the same FFN however the work is divided.
func (s *State) actmulAll(gate, up []float32, act nn.ActKind) {
	v := len(gate) / nn.ElemLanes
	if v == 0 {
		s.actmul(gate, up, act)
		return
	}
	j := &s.rg.elem
	j.op, j.a, j.b, j.act, j.n, j.v = elemActMul, gate, up, act, len(gate), v
	s.elemRun()
}

// actmul runs the generated activation.
func (s *State) actmul(gate, up []float32, act nn.ActKind) { nn.ActMul32JIT(gate, up, act) }

// actAll is actmulAll for an ungated FFN (C6): the activation alone, in
// place, split by vector exactly as actmulAll splits it.
func (s *State) actAll(x []float32, act nn.ActKind) {
	v := len(x) / nn.ElemLanes
	if v == 0 {
		nn.Act32JIT(x, act)
		return
	}
	j := &s.rg.elem
	j.op, j.a, j.act, j.n, j.v = elemAct, x, act, len(x), v
	s.elemRun()
}

// ungatedAct is an ungated FFN's activation over x: the segment's kind, or
// Apertus's xIELU with the block's own four numbers.
func (s *State) ungatedAct(l *layer, x []float32) {
	if l.xielu == nil {
		s.actAll(x, s.c.Act)
		return
	}
	v := len(x) / nn.ElemLanes
	if v == 0 {
		nn.XIELU32JIT(x, l.xielu)
		return
	}
	j := &s.rg.elem
	j.op, j.a, j.b, j.n, j.v = elemXIELU, x, l.xielu, len(x), v
	s.elemRun()
}

// lnBias is a LayerNorm's bias as a device takes it: b itself, or zeros of
// the model's width where the file carries none (command-r). The host's
// kernel bakes the absence instead and never asks for this.
func (m *Model) lnBias(b []float32) []float32 {
	if b != nil {
		return b
	}
	m.lnZerosOnce.Do(func() { m.lnZeros = make([]float32, m.Cfg.NEmbd) })
	return m.lnZeros
}

// norm is the text graph's norm: an RMSNorm, or on a classic block (C6) a
// LayerNorm with its bias (b may be nil -- command-r's has none).
func (s *State) norm(y, x, w, b []float32) {
	c := s.c
	// A block with no pre-norm (OLMo 2, EXAONE 4) feeds its input as it is;
	// the loader admits that only where the block's post-norms stand in.
	if w == nil {
		copy(y, x)
		return
	}
	if c.LayerNorm {
		nn.LayerNorm32JIT(y, x, w, b, c.RMSEps)
		return
	}
	s.rmsnorm(y, x, w, c.RMSEps)
}

// ffnPad rounds an FFN width up to a whole vector. The scratch is allocated at
// this length so a ragged tail has somewhere legal to go.
func ffnPad(n int) int { return (n + nn.ElemLanes - 1) &^ (nn.ElemLanes - 1) }

// softmax runs the generated softmax over the first n entries of row.
func (s *State) softmax(row []float32, n int) { nn.Softmax32JIT(row, n) }

// attStride rounds a score row's stride up to a whole vector, so the padding
// the generated softmax writes stays inside the row it belongs to.
func attStride(maxSeq int) int { return (maxSeq + 15) &^ 15 }

// rowsStride is the stride of the many-row score buffers (battf for a prefill
// chunk, bathf for a batch step) when their rows reach keys below reach.
//
// Those buffers hold a row per token or per (row, head), so sized at maxSeq
// they were the largest thing a State allocated: a 170-token prompt on a
// 131072-position context held 89 MB of scores for keys it could not reach,
// and a server keeps a State per concurrent request. The reach they are sized
// for grows by doubling, from scoreReachMin, up to maxSeq, so a sequence
// re-sizes them a handful of times in its life and a warm step at a steady
// reach allocates nothing.
func (s *State) rowsStride(reach int) int {
	if reach > s.scoreReach {
		r := max(s.scoreReach, scoreReachMin)
		for r < reach {
			r *= 2
		}
		s.scoreReach = min(r, s.maxSeq)
	}
	return attStride(s.scoreReach + s.m.sinkSlot())
}

// scoreReachMin is the first reach rowsStride sizes for: a page's worth of
// positions, past which a short exchange never re-sizes.
const scoreReachMin = 256

// sinkSlot is 1 when any layer carries attention sinks, and the score rows
// need room for the one extra logit.
func (m *Model) sinkSlot() int {
	for i := range m.layers {
		if m.layers[i].sinks != nil {
			return 1
		}
	}
	return 0
}

// softmaxSink normalises one head's scaled scores, af, with the head's sink
// joining the softmax when the layer has one: the sink is written one slot
// past the scores, takes part in the maximum and the denominator, and is then
// ignored -- it has no value to weight, so the probabilities over real
// positions sum to less than one. That is ggml's soft_max_ext with sinks and
// transformers' cat([scores, sink]) followed by dropping the last column.
//
// It runs after the scale: the sink is a logit in the scaled space.
func (s *State) softmaxSink(af []float32, sinks []float32, hh int) {
	if sinks == nil {
		s.softmax(af, len(af))
		return
	}
	row := af[:len(af)+1]
	row[len(af)] = sinks[hh]
	s.softmax(row, len(row))
}

// addInto is the residual add, on generated code and across the pool;
// elementwise, so parallelising it is bit-identical. It carries Granite's
// residual_scale as the alpha, which is why every block output goes through
// here; a mixture gets it through the routed scale and the shared expert's
// weight instead.
func (s *State) addInto(dst, src []float32) {
	n, a := len(dst), float32(s.c.ResidualScale)
	if n >= 4096 {
		j := &s.rg.elem
		j.op, j.a, j.b, j.alpha, j.n, j.v = elemAxpy, dst, src, a, n, n/nn.ElemLanes
		s.elemRun()
		return
	}
	s.axpy(dst, src, a)
}

// finishLogits applies whatever the model does to its logits after the head:
// gemma2's final softcap and Granite's logit_scale, which divides. No model
// has both. Every path that produces logits ends here, the device's included.
func (s *State) finishLogits(x []float32) {
	s.headBias(x)
	softcap(x, s.c.FinalSoftcap)
	s.scaleLogits(x)
}

// finishDevLogits is finishLogits for logits the device head produced, which
// already carry the final softcap (capping twice is a different function). No
// model has both a head bias and a final softcap, so the order is safe.
func (s *State) finishDevLogits(x []float32) {
	s.headBias(x)
	s.scaleLogits(x)
}

// headBias adds the output projection's bias (phi-2) to one or more whole rows
// of NVocab.
//
// It goes first, before any cap or scale, and is added here so every arm that
// makes logits (host, device head, batch) gets it exactly once.
func (s *State) headBias(x []float32) {
	if s.m.outB != nil {
		s.addBiasRows(x, s.m.outB, len(x)/s.c.NVocab)
	}
}

// scaleLogits is Granite's logit_scale, which divides.
func (s *State) scaleLogits(x []float32) {
	if c := s.c; c.LogitScale != 1 {
		s.scale(x, float32(1/c.LogitScale))
	}
}

// axpy runs the generated dst += alpha*src.
func (s *State) axpy(dst, src []float32, alpha float32) { nn.Axpy32JIT(dst, src, alpha) }

// scale multiplies a vector in place, on generated code.
func (s *State) scale(x []float32, alpha float32) { nn.Scale32JIT(x, alpha) }

// softcap applies x = c*tanh(x/c) in place, generated; a no-op at c == 0.
func softcap(x []float32, c float32) { nn.Softcap32JIT(x, c) }

// JIT is this state's generated-code tier: its vision segment runs on it
// (State.Vision), so a model has one JIT and one worker pool a session.
func (s *State) JIT() *nn.JIT { return s.jit }

// GPULayers is how many blocks a device took, which is the set and not the
// prefix: on a hybrid the two differ and the count is what a caller means.
func (s *State) GPULayers() int { return s.devCount() }

// noteDecline records why one block did not go to a device.
func (s *State) noteDecline(why string) {
	if s.declines == nil {
		s.declines = map[string]int{}
	}
	s.declines[why]++
}

// DeviceDeclines is how many blocks each device refused, by reason, most
// numerous first. A placement is not explained by the blocks it took: without
// this, a capability gap and a memory gap look identical.
func (s *State) DeviceDeclines() []DeviceDecline {
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
func (s *State) DeviceBlocks() [][2]int { return s.devRuns() }

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
func (s *State) SetGPUTarget(n int) {
	if n > s.hi {
		n = s.hi
	}
	s.gpuTarget = n
}

// growStep adopts at most one block, called once per token.
//
// One block, not as many as are ready, so no single token carries the whole
// upload.
func (s *State) growStep() {
	if s.ld == nil || s.gpuTarget <= s.gpuLayers {
		return
	}
	was := s.gpuLayers
	got := s.SetGPULayers(was + 1)
	if got == was {
		if s.m.opt.growDebug {
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
func (s *State) PrewarmGPU(n int) <-chan struct{} {
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
			l := &s.m.layers[li]
			qn, kn := s.qkNorms(li)
			w := nn.LayerWeights{
				AttnNorm: l.attnNorm, FFNNorm: l.ffnNorm,
				PostAttnNorm: l.postAttnNorm, PostFFNNorm: l.postFFNNorm,
				AttnSubNorm: l.attnSubNorm, FFNSubNorm: l.ffnSubNorm,
				QNorm: qn, KNorm: kn,
				Bq: l.bq, Bk: l.bk, Bv: l.bv, Bo: l.bo,
				Wq: wt(l.wq), Wk: wt(l.wk), Wv: wt(l.wv), Wo: wt(l.wo),
				Gate: wt(l.gate), Up: wt(l.up), Down: wt(l.down),
				XIELU: l.xielu,
			}
			w.Sinks = l.sinks
			if c.MoE() {
				w.Router = wt(l.router)
				w.RouterB, w.ExpGateB, w.ExpUpB, w.ExpDownB = l.routerB, l.expGateB, l.expUpB, l.expDownB
				w.ExpSelB = l.expProbsB
				// The same lazy marker offerRange sets: the routed bank is
				// unread, and this goroutine must not do the file I/O beside a
				// live submission, so the tier skips packing such a block and
				// the upload path ensures it.
				w.Ensure, w.EnsureExperts = s.ensureLayer(li), s.ensureSelected(li)
				w.PrefetchExperts, w.HostExperts = s.prefetchSelected(li), s.hostExpertsFor(li)
				w.HostExpertsRows = s.hostExpertsRowsFor(li)
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
func (s *State) SetGPULayers(n int) int {
	if s.ld == nil {
		return 0
	}
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
		if s.m.opt.growDebug && s.gpuLayers == was {
			fmt.Fprintf(os.Stderr, "grow: PrepLayer(%d) declined\n", was)
		}
		for i, li := range fresh {
			if !s.devAt(li) {
				continue
			}
			if !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
				if s.m.opt.growDebug {
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

// Accelerated reports which quantization types this session runs on generated
// code.
func (s *State) Accelerated() []quant.Type { return s.jit.Types() }

// Workers is how many worker threads the fast tier uses.
func (s *State) Workers() int { return s.jit.Workers() }

// Shapes is how many shape-specialized kernels were generated for this model.
func (s *State) Shapes() int { return s.jit.Shapes() }

// addBias adds b to out when b is present; the nil test is one branch per
// projection rather than per element.
func (s *State) addBias(out, b []float32) {
	if b == nil {
		return
	}
	s.axpy(out, b, 1)
}

// addBiasRows is addBias over a batch of n rows of width len(b).
func (s *State) addBiasRows(out, b []float32, n int) {
	if b == nil {
		return
	}
	w := len(b)
	// A picture's rows -- a thousand of them, six biases a block -- on the
	// pool, as the tower's own loop ran them; one at a time they were a tenth
	// of SmolVLM's encode.
	if s.vis != nil && n > 1 {
		s.axpyRows(out, b, nil, w, n, s.rowChunk(n, w), 1)
		return
	}
	for r := 0; r < n; r++ {
		s.axpy(out[r*w:(r+1)*w], b, 1)
	}
}

// mv runs one matvec on generated code. There is no fallback: a decline is an
// error naming the tensor.
func (s *State) mv(out []float32, w tensor, x []float32) error {
	s.mvCalls++
	t0 := s.tick()
	defer func() { s.tock(opMatVec, t0) }()
	// A container weight is read by the host in the device layout, so a block
	// can sit on either side of the seam without a repack or a second copy.
	if w.packed != nil {
		if s.jit.MatVecPacked(out, w.typ, w.packed, x, w.rows, w.k) {
			return nil
		}
		// nn.MatVecPacked declines for a missing emitter, an unbound weight or
		// a caller's buffer that is too small; say which contract failed.
		switch {
		case len(w.packed.QS) == 0:
			return fmt.Errorf("jitllm: %s is not bound (%s, %d rows of k=%d): its page is "+
				"out, or bindPacked does not carry it", w.name(), w.typ, w.rows, w.k)
		case len(out) < w.rows:
			return fmt.Errorf("jitllm: %s writes %d rows and the caller's buffer holds %d "+
				"(%s, k=%d): a scratch sized from the config against a weight sized from "+
				"the file", w.name(), w.rows, len(out), w.typ, w.k)
		}
		return fmt.Errorf("jitllm: no host kernel reads %s (%s, rows [%d,%d) of %d, k=%d) on %s",
			w.name(), w.typ, w.packed.Row, w.packed.Row+w.rows, max(w.packed.Stride, w.rows), w.k, runtime.GOARCH)
	}
	// The device keeps its copy of a weight keyed on the pointer (nn.Device).
	// The dense region never moves; a page's frame is reused for the next
	// block's tensors at the same offsets and shapes, so a page's weight is
	// offered only to a device the pager tells when it refills a frame
	// (nn.CopyForgetter, Model.forgetCopies). Offered to any other, it would
	// answer for the next block from its copy of the first
	// (TestPagedWeightsAreNeverServedStale).
	matvec := s.jit.MatVecHost
	// A routed expert's sheet is never offered: a device copy of it (a lone
	// matvec the device keeps keyed on the weight) is an upload for one row's
	// use and a buffer nothing frees when the block is placed -- a hybrid block
	// never uploads its bank to take the copies back, and they outlived its
	// return (TestBatchSeamMovesCarryEveryRowOffCard).
	if w.e != nil && s.m.container != nil && (s.forgets || s.m.container.InDense(w.e)) && !s.hostOnly &&
		!jlm.ExpertBank(w.e.Role) {
		matvec = s.jit.MatVec
	}
	if matvec(out, w.typ, w.data, x, w.rows, w.k) {
		return nil
	}
	return fmt.Errorf("jitllm: no host kernel reads %s (%s, %dx%d, row-major) on %s",
		w.name(), w.typ, w.rows, w.k, runtime.GOARCH)
}

// Pos is the number of positions already in the cache.
func (s *State) Pos() int { return s.pos }

// PrefillRows is how many prompt rows this State has run through its blocks
// since it was made: a prompt restored from a prefix cache adds none, and a
// one-row prompt (decode's path) adds one.
func (s *State) PrefillRows() int { return s.promptRows }

// stepDownKVToF32 narrows the cache from packed halves to f32 and rebuilds it.
// It is a named method so a gate can reach it without a real card. The caller
// owns the pos == 0 precondition: the pages are rebuilt, not reinterpreted.
func (s *State) stepDownKVToF32() {
	s.kvFmt, s.kvl.fmt, s.kvl.jit = cpu.KVF32, cpu.KVF32, nil
	// attnWidths, not Cfg.HeadDim: on MLA the key and value widths differ
	// from HeadDim (TestMLAStepsTheKVDownAtTheRightWidths).
	hdK, hdV := s.c.attnWidths()
	s.jit.AddAttnKV(hdK, hdV, s.kvl.Stride(), cpu.KVF32)
	s.provisionAttn()
	// Carry every caller-visible setting into the fresh cache (namespace and
	// budget); a forgotten one is silently lost.
	old := s.kv
	s.kv = newKVCacheRange(s.c, old.id, s.nseq, s.kvl, old.store, s.m.opt.kvPage, s.lo, s.hi)
	s.kv.usePool(&s.m.kvPool)
	s.kv.reserve(s.maxSeq)
	s.kv.ns, s.kv.budget = old.ns, old.budget
}

// seqPos and advance are the two places that know a position may be per row.
func (s *State) seqPos(slot int) int { return s.bpos[slot] }

func (s *State) advance(slot, n int) {
	s.bpos[slot] += n
	if s.bpos[slot] > s.pos {
		s.pos = s.bpos[slot] // the high water mark; see bpos
	}
	// Sealing is a consequence of the position moving, so it lives here. It
	// runs against the slowest row, not the furthest: a page holds every
	// row's slots, and a row admitted by PrefillSeq may still be writing a
	// page the others have left.
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
func (s *State) liveLow() int {
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
func (s *State) lowPos() int {
	low := s.bpos[0]
	for _, p := range s.bpos[1:] {
		if p < low {
			low = p
		}
	}
	return low
}

// SeqPos is row i's position. For a single-sequence session it is Pos().
func (s *State) SeqPos(i int) int { return s.bpos[i] }

// Reset clears the session without reallocating.
//
// It clears the pages and, on a hybrid, the recurrent summary, not only the
// positions: either left behind would give the next sequence the previous
// one's history.
func (s *State) Reset() {
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
			s.retireErr = fmt.Errorf("model: the recurrent state on the device could not be reset")
		}
	}
	s.kv.reset()
	s.kvErr.Store(nil)
	s.kvSkipped = 0
}

// Retire frees row i for a new sequence.
//
// It zeroes a position and copies no KV: attention reads only up to a row's
// own position, so stale keys past it are never addressed. A recurrent block
// has no position to rewind, so its running summary for the row is zeroed, on
// the host and on any device holding a linear block; a failure there is kept
// and returned by the next batch step.
func (s *State) Retire(i int) {
	if i < 0 || i >= s.nseq {
		return
	}
	if s.recurrent() {
		s.ResetRecurrent(i)
		if s.devLinear() {
			if rr, ok := s.ld.(nn.RecRowsDevice); !ok || !rr.ResetRecRows([]int{i}) {
				s.retireErr = fmt.Errorf("model: row %d's recurrent state on the device could not be reset", i)
			}
		}
	}
	s.bpos[i] = 0
	s.rewindRope(i, -1)
	s.pos = 0
	for _, p := range s.bpos {
		if p > s.pos {
			s.pos = p
		}
	}
	// The next occupant writes this row's slots from page 0 up, so every page
	// is mutable again: lower the seal mark with the slowest row (seal treats a
	// fall as a rollback). Left high, trim would evict pages the new row has
	// rewritten and a fault would bring back the store's older copy.
	s.kv.seal(s.lowPos())
}

// ForwardEmbd is Forward with the token's embedding supplied rather than looked
// up: the decode half of the embedding seam a vision-language model splices
// image embeddings through.
//
// e is the residual stream, not a vocabulary row: EmbdScale is not applied
// here, so projected image features are not scaled twice. A caller reproducing
// the token path applies the scale itself (TestForwardEmbdMatchesForward).
func (s *State) ForwardEmbd(e []float32) ([]float32, error) {
	if len(e) != s.c.NEmbd {
		return nil, fmt.Errorf("model: ForwardEmbd: embedding is %d wide, want %d", len(e), s.c.NEmbd)
	}
	s.tok = -1
	return s.forward(func() error {
		copy(s.x, e)
		s.m.trace(-1, "embd", s.x)
		if s.c.PLEDim != 0 {
			return s.pleInputs(s.ple, 0, s.x) // see ple.go
		}
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
func (s *State) Forward(token int32) ([]float32, error) {
	defer s.m.enterPager()()
	if int(token) < 0 || int(token) >= s.c.NVocab {
		return nil, errToken{token, s.c.NVocab}
	}
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return nil, err
	}
	s.kv.note(s.bpos[0], token)
	s.tok = token
	return s.forward(func() error { return s.embed(token) })
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
func (s *State) ForwardGreedy(token int32) (int32, error) {
	h := s.head
	// Not with a head bias, which is added on the host (finishLogits).
	onDev := h != nil && s.m.outB == nil
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
	return Greedy(logits), nil
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
// Not under a head bias or a logit scale, which are added on the host after
// the head (finishLogits) and would come after the device's penalty rather
// than before it. A greedy smp is ForwardGreedy.
func (s *State) ForwardSample(token int32, smp *Sampler) (int32, error) {
	if smp == nil || smp.Temp <= 0 {
		return s.ForwardGreedy(token)
	}
	h := s.head
	onDev := h != nil && s.m.outB == nil && s.c.LogitScale == 1 && smp.Bounded(s.c.NVocab) &&
		s.m.opt.devSample != DeviceSampleOff
	// Auto: the key's verdict, or the probe's arm for this token
	// (devsample.go). Both arms return the same token.
	var choice *sampleChoice
	probe := false
	if onDev && s.m.opt.devSample == DeviceSampleAuto {
		choice = s.sampleChoiceFor(s.sampleKeyOf(smp))
		cs := &s.m.sampleChoices
		cs.mu.Lock()
		onDev, probe = choice.arm()
		cs.mu.Unlock()
	}
	if onDev {
		k := smp.TopK
		s.sampleArgs = smp.DeviceArgs(s.sampleArgs)
		s.sampleVals = grow32(s.sampleVals, k)
		if cap(s.sampleIDs) < k {
			s.sampleIDs = make([]uint32, k)
		}
		h.SampleK, h.SampleArgs, h.SampleVals, h.SampleIDs, h.Sampled = k, s.sampleArgs, s.sampleVals, s.sampleIDs[:k], false
		defer func() { h.SampleK, h.Sampled = 0, false }()
	}
	t0 := time.Now()
	logits, err := s.Forward(token)
	if err != nil {
		return 0, err
	}
	var tok int32
	if onDev && h.Sampled {
		s.SampledOnDevice++
		tok = smp.SampleFrom(h.SampleVals, h.SampleIDs)
	} else {
		tok = smp.Sample(logits)
	}
	if choice != nil {
		d := time.Since(t0)
		cs := &s.m.sampleChoices
		cs.mu.Lock()
		choice.record(onDev, probe, d)
		cs.mu.Unlock()
		if probe {
			s.SampleProbes++
		}
	}
	return tok, nil
}

// forward runs one decode step over whatever fill() puts in s.x.
//
// fill is a closure because the device-failure path calls it again to restart
// the token on the host (Layers may have written part of s.x).
func (s *State) forward(fill func() error) ([]float32, error) {
	// Brackets one decode step for the host tuners, which time real tokens.
	// Not when the device ran the whole token: no CPU matvec ran, and a
	// settled answer would be cached under the CPU's key for later runs.
	if s.gpuLayers < s.hi || s.head == nil {
		s.jit.TokenStart()
		// A token that read expert pages off disk is not a sample: that I/O
		// dwarfs what the tuners measure.
		reads0 := atomic.LoadInt64(&s.m.expReads)
		defer func() {
			if atomic.LoadInt64(&s.m.expReads) != reads0 {
				s.jit.DiscardToken()
			}
			s.jit.TokenEnd()
		}()
	}

	// The pool spins or parks for the next token by what this one did. A token
	// that paged weights in waits on reads between its regions, and a spinning
	// worker would burn a core through each; a token that read nothing runs its
	// regions back to back, and parking would cost a wake-up at every one.
	// Deciding by the model's size against the budget parked Qwen3-30B in a
	// 16 GiB cgroup for a run that read nothing after warm-up, at well under
	// half the spinning rate (scheduling-and-measurement.md, "What was left:
	// the pool spinning through a blocking read").
	if pc := s.m.container; pc != nil {
		in0, _ := pc.Faults()
		calls0 := s.hostCalls
		defer func() {
			in1, _ := pc.Faults()
			s.jit.SetSpinning(in1 == in0)
			// A token whose device suspended for the host's experts waited on
			// the card between those calls: the workers spin across it.
			if in1 == in0 && s.hostCalls != calls0 {
				if d := s.m.opt.hybridSpinFor(); d > 0 {
					s.jit.SetSpinFor(d)
				}
			}
		}()
	}

	m, c := s.m, s.c
	if s.batched {
		return nil, errBatch{}
	}
	if s.pos >= s.maxSeq {
		return nil, errFull{s.maxSeq, s.reqSeq}
	}
	// A learned position table (starcoder) rides the fill, so a restarted
	// token carries its position too.
	if m.posEmbd.e != nil {
		inner, pos := fill, s.pos
		fill = func() error {
			if err := inner(); err != nil {
				return err
			}
			return m.addPos(s.x, s.prow, pos)
		}
	}
	// AltUp's streams ride the fill as well (altup.go): every restart begins
	// from the embedding.
	if c.AltUp != 0 {
		inner := fill
		fill = func() error {
			if err := inner(); err != nil {
				return err
			}
			return s.altExpand(s.xa, 1)
		}
	}
	if c.DSV4() {
		inner := fill
		fill = func() error {
			if err := inner(); err != nil {
				return err
			}
			s.ds4Expand(s.xa, 1)
			return nil
		}
	}
	if c.Q4E() {
		inner := fill
		fill = func() error {
			if err := inner(); err != nil {
				return err
			}
			s.q4eExpand(s.xa, 1)
			return nil
		}
	}
	// Kimi-K3's bank starts every token empty (k3.go).
	if c.ResAttn() {
		inner := fill
		fill = func() error {
			if err := inner(); err != nil {
				return err
			}
			s.k3Expand(s.xa, 1)
			return nil
		}
	}
	s.relocateFor(s.pos + 1)
	if err := fill(); err != nil {
		return nil, err
	}

	// Each contiguous run of device blocks is one call (one upload, one
	// download, one drain), not one call per block.
	// The row's rotary position, which an image earlier in the sequence has
	// moved off s.pos on an M-RoPE model (mrope.go).
	rp := s.ropeOf(0, s.pos)
	s.ropeFill(s.cs, rp)
	if m.ropeSWA != nil {
		s.jit.RopeTable(*m.ropeSWA, s.csSWA, rp)
	}
	// With the last block on the device the projection rides the same
	// submission (folded); otherwise it is its own call below. Device runs sit
	// inside the block loop, and the residual crosses the seam once per run.
	//
	// AltUp's head reads the streams' mean, which the host takes, so the
	// projection is never folded into the last run.
	folded := s.head != nil && s.devAt(s.hi-1) && !c.streamHead()
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
	// attnResid and ffnResid: a group already added the projection into the
	// residual (fuse.go).
	var attnResid, ffnResid bool
	for li := s.lo; li < s.hi; li++ {
		if s.devAt(li) {
			// A device run ends the fused host run (fuse.go).
			s.unfuse()
			hi := li + 1
			for hi < s.hi && s.devAt(hi) {
				hi++
			}
			var fold *nn.Head
			if folded && hi == s.hi {
				fold = s.head
				unhide = s.rows.hiddenOf(fold)
			}
			s.layerInputs(s.ple)
			s.tok1[0] = s.tok
			s.tokenIDs(s.tok1[:])
			// Qwen3.8-Flash-Next's n-gram rows, when the run holds its n-gram
			// block (q4edev.go).
			if err := s.ngramToDevice(li, hi, 1, nil, s.selfSlot0, s.tok1At); err != nil {
				return nil, err
			}
			if !s.ld.Layers(li, hi, s.pos, 1, s.resid(), s.cs[:c.RopeW()], s.swaTable(c.NRotSWA), fold) {
				s.ngramUndoRows()
				// A failed run demotes everything and restarts the token on the
				// host from the embeddings (Layers may have written part of the
				// residual), unless SetDeviceFallback(false) asked for the
				// error.
				if !s.devFallback {
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
			m.trace(hi-1, "ffn_resid", s.x)
			if fold != nil {
				// The device ran the output norm and the projection too, so the
				// token is finished and the residual never came home.
				//
				// The device applied the final softcap (nn.Head.Softcap).
				s.finishDevLogits(s.logits)
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
			continue
		}
		// The block is faulted in at first touch; with a budget that holds the
		// model this is one nil compare. A block that has to be read ends the
		// fused run before the read, and the run reopens after it.
		if s.fused && !m.blockResident(li) {
			s.unfuse()
		}
		if err := m.pageIn(li); err != nil {
			return nil, err
		}
		s.fuseRun()
		l := &m.layers[li]
		if c.DSV4() {
			s.ds.rslot[0], s.ds.rpos[0], s.ds.rid[0] = 0, s.pos, s.tok
			if err := s.ds4Block(li, l, s.xa, 1); err != nil {
				return nil, err
			}
			m.trace(li, "ffn_resid", s.x)
			continue
		}
		if c.AltUp != 0 {
			if err := s.altPredict(l, s.xa, s.altPred, 1); err != nil {
				return nil, err
			}
		}
		t0 := s.tick()
		// Kimi-K3's attention reads its mix over the bank (k3.go).
		attnIn := s.x
		if c.Q4E() {
			// Qwen3.8-Flash-Next's n-gram block adds its embedding, then
			// every sublayer reads its hyper-connection's mix (q4e.go).
			if c.NGramAt(li) {
				tok := s.tok
				if err := s.ngramApply(li, l, s.xa, 1, nil, func(int) (*State, int) { return s, 0 },
					func(int) int32 { return tok }); err != nil {
					return nil, err
				}
			}
			if err := s.hcMix(s.xa, 1, &l.q4e.attn); err != nil {
				return nil, err
			}
			attnIn = s.q4.mix[:c.NEmbd]
		}
		if c.ResAttn() {
			var err error
			if attnIn, err = s.k3AttnIn(li, l, s.xa, 1); err != nil {
				return nil, err
			}
		}
		s.norm(s.h, attnIn, l.attnNorm, l.attnNormB)
		// A parallel block's FFN reads the block input, so its norm is taken
		// here, before attention adds into s.x. With one shared norm (phi-2) it
		// is a copy of s.h, which attention reuses for its output.
		if c.Parallel {
			if l.ffnNorm != nil {
				s.norm(s.hf, s.x, l.ffnNorm, l.ffnNormB)
			} else {
				copy(s.hf, s.h)
			}
		}
		s.jit.NewInput()
		s.tock(opRMSNorm, t0)
		m.trace(li, "attn_norm", s.h)
		if c.AltUp != 0 {
			if err := s.laurel(l, s.h, s.altLaur); err != nil {
				return nil, err
			}
		}

		// The layer kind (from the container) chooses the block. Both paths
		// leave their result in s.h; linearAttn's input may alias its output
		// because it reads h in its projections before writing anything.
		// A block that attends as well (Falcon-H1) leaves the mixer's output
		// in s.hf and runs its attention from the same s.h.
		kind := c.LayerKind(li)
		if kind.Recurrent() {
			out := s.h
			if kind.Attends() {
				out = s.hf
			}
			if err := s.linearAttn(li, l, 0, s.h, out); err != nil {
				return nil, err
			}
			m.trace(li, "ssm_out", out)
		}
		if !kind.Attends() {
		} else if c.MLA() {
			// MLA: one call replaces q, k, v and attnPrep, leaving the row-wide
			// query in s.mlaAbs and this position's row in the cache.
			if err := s.mlaProject(li, l, s.h,
				s.ropeTable(li, s.cs, s.csSWA, 0), 0, s.pos); err != nil {
				return nil, err
			}
			m.trace(li, "k", s.kc[:c.KVDim()])
		} else {
			kvd := c.KVDimAt(li)
			// q, k and v as one region inside a fused run (fuse.go).
			s.aheadOf(li, 0, [nn.MaxGroup]*tensor{&l.wo})
			grouped := !c.AttnOutGate && !c.KVShared(li) && !l.vFromK && !c.extraKVHead(li) &&
				s.groupOf(s.h, [nn.MaxGroup]*tensor{&l.wq, &l.wk, &l.wv},
					[nn.MaxGroup][]float32{s.q, s.kc, s.vc}, 3)
			if grouped {
				s.addBiasRows(s.q, l.bq, 1)
			} else if _, err := s.projectQ(l, s.q, s.ogate, s.h, 1, false); err != nil {
				return nil, err
			}
			// A KV-sharing block projects q alone and attends to its
			// source's history.
			if c.KVShared(li) {
				s.attnPrep(l, s.q, nil, nil, s.ropeTable(li, s.cs, s.csSWA, 0), li, 0, s.pos)
				goto attend
			}
			if !grouped {
				if err := s.mv(s.kc, l.wk, s.h); err != nil {
					return nil, err
				}
			}
			s.addBias(s.kc, l.bk)
			if l.vFromK {
				// Gemma 4's global layers: v is k's projection, taken before
				// k's norm and rotary.
				copy(s.vc[:kvd], s.kc[:kvd])
			} else if !grouped {
				if err := s.mv(s.vc, l.wv, s.h); err != nil {
					return nil, err
				}
			}
			s.addBias(s.vc, l.bv)
			// MiniMax Sparse Attention: the indexer's key rides the row as one
			// more kv head (its value zero) and its query is this token's, one
			// head per kv group (msa.go).
			kvr := kvd
			if c.MSAAt(li) {
				kvr = c.KVRowAt(li)
				cs := s.ropeTable(li, s.cs, s.csSWA, 0)
				if err := s.msaKey(s.kc[kvd:kvr], l, s.h, cs); err != nil {
					return nil, err
				}
				clear(s.vc[kvd:kvr])
				if err := s.msaQuery(s.msaQ, l, s.h, cs); err != nil {
					return nil, err
				}
			}
			// QSA: the indexer's raw key rides the row the same way, padded
			// to a head (qsa.go).
			if c.QSAAt(li) {
				kvr = c.KVRowAt(li)
				g := s.qsa()
				if err := s.qsaProject(s.kc[kvd:kvr], g.q, l, s.h,
					s.ropeTable(li, s.cs, s.csSWA, 0)); err != nil {
					return nil, err
				}
				clear(s.vc[kvd:kvr])
			}

			// attnPrep applies qwen3's per-head q/k norm (before RoPE), the
			// rotary and the cache write.
			s.attnPrep(l, s.q, s.kc[:kvr], s.vc[:kvr], s.ropeTable(li, s.cs, s.csSWA, 0),
				li, 0, s.pos)
			m.trace(li, "k", s.kc[:kvd])
		}
	attend:
		if kind.Attends() {

			// Attention, across the pool: it is the one op that grows with the
			// context. With GQA, query head hh reads kv head hh/gqa (hh%nKVHead
			// also runs and is wrong). Heads are independent -- each writes its
			// own slice of xb and its own scores row -- so the split needs no
			// coordination.
			t0 = s.tick()
			// Config.AttnScale is 1/sqrt(HeadDim) times YaRN's magnitude
			// correction squared (DeepSeek), computed once at load.
			scale := c.AttnScale
			// MLA reads a wider query than it writes, and every head reads the
			// same cached row: qw is the whole row, ow its latent prefix, and
			// gqa = NHead makes hh/gqa 0 for every head.
			j := &s.rg.attn
			hd := c.HeadDimAt(li)
			// j.li is the history read: a KV-sharing block's source's.
			kvli := c.KVSource(li)
			j.l, j.li, j.gqa, j.hd, j.mla = l, kvli, c.GQAAt(li), hd, c.MLA()
			j.qw, j.ow, j.qbuf, j.obuf = hd, hd, s.q, s.xbf
			j.xb, j.atf, j.fast, j.astride = s.xb, s.attf, s.attnAt(li), s.attStride
			if j.mla {
				j.qw, j.ow = c.KVLoraRank+c.NRot, c.KVLoraRank
				j.qbuf, j.obuf, j.gqa = s.mlaAbs, s.mlaAcc, c.NHead
			}
			// Heads in pairs when they share a kv head: the paired kernels load
			// each K/V vector once for two queries. That needs hh and hh+1 on
			// the same kvh (even gqa, even hh) and in the same pool task, so
			// pairing forces a chunk of at least two. An A/B of pairing must
			// compare at the same chunk.
			j.pair = s.AttnPair()
			chunk := s.AttnChunk()
			if j.pair != 0 && chunk < 2 {
				chunk = 2
			}
			// A sliding or chunked window (gemma3, Llama 4) is a base w0 and a
			// count an, not a different kernel; without a window it is 0 and
			// pos+1. Config.AttnWindow is the one derivation for all paths.
			j.w0, j.an = c.AttnWindow(li, s.pos)
			// Llama 4's attention temperature scales q on its NoPE layers, and
			// the scores are linear in q, so it rides the softmax scale.
			scale *= c.AttnTemp(li, s.pos)
			j.scale, j.softcap = float32(scale), c.AttnSoftcap
			// Residency is decided before the fan-out; see kvEnsureWindow.
			s.kvEnsureWindow(kvli, j.w0, j.an)
			j.bias, j.bstride = nil, 0
			if c.Indexer() {
				j.bias = s.idxMask(kvli, 0, j.w0+j.an, s.idxQ, s.idxW, s.idxTmp, s.idxScore,
					s.idxBias, &s.idxOrd)
			}
			if c.MSAAt(li) {
				j.bias = s.msaMask(kvli, 0, j.w0+j.an, s.attStride, s.msaQ, s.msaTmp, s.msaBias,
					s.msaKept, &s.msaOrd)
				j.bstride = s.attStride
			}
			if c.QSAAt(li) {
				g := s.qsa()
				j.bias = s.qsaMask(s, kvli, 0, j.w0+j.an, g.q, g.bias, l)
			}
			if j.fn == nil {
				j.fn = s.attnHeads
			}
			s.jit.Parallel(c.NHead, chunk, j.fn)
			j.l, j.qbuf, j.obuf, j.xb, j.atf, j.bias = nil, nil, nil, nil, nil, nil
			s.tock(opAttn, t0)
			s.jit.NewInput()
			m.trace(li, "attn", s.xb)

			// The output gate goes before the projection (after wo would scale
			// the residual contribution instead).
			if c.MLA() {
				// W_v un-absorbs out of the latent and wo follows, both inside
				// mlaOutProject, with Kimi-K3's output gate between them.
				if err := s.mlaOutProject(s.h, l); err != nil {
					return nil, err
				}
			} else {
				src := s.xb
				if c.AttnOutGate {
					s.applyOutGate(s.ogate, s.xb, 1)
					s.jit.NewInput()
					src = s.ogate
				}
				// BitNet's norm over the attention output, before wo.
				if l.attnSubNorm != nil {
					t0 = s.tick()
					s.rmsnorm(src[:c.QDimAt(li)], src[:c.QDimAt(li)], l.attnSubNorm, c.RMSEps)
					s.jit.NewInput()
					s.tock(opRMSNorm, t0)
				}
				// With nothing between the projection and the residual, the
				// add rides the projection's rows (fuse.go).
				plain := s.residFold && l.bo == nil && !kind.Recurrent() && l.postAttnNorm == nil &&
					!c.ResAttn() && c.AltUp == 0 && !c.Q4E()
				if plain {
					s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
				}
				s.aheadOf(li, 1, [nn.MaxGroup]*tensor{&l.gate, &l.up})
				attnResid = plain && s.groupOf(src[:c.QDimAt(li)], [nn.MaxGroup]*tensor{&l.wo},
					[nn.MaxGroup][]float32{s.h}, 1)
				s.grp.Resid = nil
				if !attnResid && !s.groupOf(src[:c.QDimAt(li)], [nn.MaxGroup]*tensor{&l.wo},
					[nn.MaxGroup][]float32{s.h}, 1) {
					if err := s.mv(s.h, l.wo, src[:c.QDimAt(li)]); err != nil {
						return nil, err
					}
				}
			}
			// The bias goes before the residual add; after it s.h is dead.
			s.addBias(s.h, l.bo)
			if kind.Recurrent() {
				s.addInto(s.h, s.hf)
			}
		}
		// gemma2/gemma3 normalise the attention output before the residual.
		if l.postAttnNorm != nil {
			t0 = s.tick()
			s.rmsnorm(s.h, s.h, l.postAttnNorm, c.RMSEps)
			s.tock(opRMSNorm, t0)
		}
		t0 = s.tick()
		if c.Q4E() {
			s.hcPlace(s.xa, 1, s.h)
		} else if c.ResAttn() {
			s.k3Resid(li, s.x, s.h, 1)
		} else if !attnResid {
			s.addInto(s.x, s.h)
		}
		attnResid = false
		if c.AltUp != 0 {
			s.laurelJoin(s.x, s.altLaur)
		}
		s.tock(opResid, t0)
		m.trace(li, "attn_resid", s.x)
		// A block that is its mixer alone (plain Mamba-2, a Nemotron-H mixer
		// layer) ends at the first residual.
		if l.noFFN {
			continue
		}

		t0 = s.tick()
		ffnIn := s.h
		if c.Parallel {
			ffnIn = s.hf
		} else if c.ResAttn() {
			in, err := s.k3FFNIn(li, l, s.xa, 1)
			if err != nil {
				return nil, err
			}
			s.norm(s.h, in, l.ffnNorm, l.ffnNormB)
		} else if c.Q4E() {
			if err := s.hcMix(s.xa, 1, &l.q4e.ffn); err != nil {
				return nil, err
			}
			copy(s.h, s.q4.mix[:c.NEmbd])
		} else {
			s.norm(s.h, s.x, l.ffnNorm, l.ffnNormB)
		}
		s.jit.NewInput()
		s.tock(opRMSNorm, t0)
		// The presence of the router is the fact; the config field is a guess
		// about it.
		ffn := s.h
		if l.router.data != nil && c.DenseMoE {
			// Gemma 4's block: the dense MLP and the mixture side by side, each
			// normed on both sides, their sum normed before the residual.
			if err := s.denseMoE(li, l, s.h, s.x); err != nil {
				return nil, err
			}
			s.layerOutScale(l, s.x)
			m.trace(li, "ffn_resid", s.x)
			continue
		}
		if l.router.data != nil {
			// The mixture adds straight into the residual, as moeBatch does, so
			// every entry point builds the same float sum. A post-FFN norm
			// cannot be expressed that way, and no supported mixture has one,
			// so it is refused rather than skipped.
			if l.postFFNNorm != nil {
				return nil, fmt.Errorf("model: layer %d is a mixture with a "+
					"post-FFN norm, which moe() cannot apply before the residual", li)
			}
			if c.ExpertLatent != 0 {
				if err := s.k3MoE(li, l, ffnIn, s.x); err != nil {
					return nil, err
				}
			} else if c.Q4E() {
				out := s.q4.out[:c.NEmbd]
				clear(out)
				if err := s.moe(li, l, ffnIn, ffnIn, out); err != nil {
					return nil, err
				}
				s.hcPlace(s.xa, 1, out)
			} else if err := s.moe(li, l, ffnIn, ffnIn, s.x); err != nil {
				return nil, err
			}
			s.layerOutScale(l, s.x)
			m.trace(li, "ffn_resid", s.x)
			continue
		} else if l.gate.e == nil {
			// The ungated FFN (C6): up, its bias, the activation alone, down
			// and its bias. No gate matrix exists to multiply by.
			if err := s.mv(s.up, l.up, ffnIn); err != nil {
				return nil, err
			}
			s.addBias(s.up, l.upB)
			t0 = s.tick()
			s.ungatedAct(l, s.up)
			s.jit.NewInput()
			s.tock(opAct, t0)
			if err := s.mv(s.h, l.down, s.up); err != nil {
				return nil, err
			}
			s.addBias(s.h, l.downB)
		} else {
			// The block's own width: Gemma 4's KV-sharing blocks are twice
			// NFFN (Config.NFFNAt).
			ff := c.NFFNAt(li)
			gate, up, act := s.gate[:ff], s.up[:ff], c.Act
			s.aheadOf(li, 2, [nn.MaxGroup]*tensor{&l.down})
			if !s.groupOf(ffnIn, [nn.MaxGroup]*tensor{&l.gate, &l.up},
				[nn.MaxGroup][]float32{gate, up}, 2) {
				if err := s.mv(gate, l.gate, ffnIn); err != nil {
					return nil, err
				}
				if err := s.mv(up, l.up, ffnIn); err != nil {
					return nil, err
				}
			}
			t0 = s.tick()
			s.gaussTopK(li, gate, ff)
			// Inside a fused run the activation rides the down projection's
			// group: each worker forms act(gate)*up itself (fuse.go).
			s.grp.ActUp, s.grp.Act = up, act
			if s.residFold && l.postFFNNorm == nil {
				s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
			}
			s.aheadNext(li)
			if l.ffnSubNorm != nil {
				// BitNet normalises act(gate)*up before down, so the product
				// is formed whole rather than inside down's group.
				s.grp.ActUp, s.grp.Act = nil, 0
			}
			grouped := l.ffnSubNorm == nil &&
				s.groupOf(gate, [nn.MaxGroup]*tensor{&l.down}, [nn.MaxGroup][]float32{s.h}, 1)
			ffnResid = grouped && s.grp.Resid != nil
			s.grp.ActUp, s.grp.Resid = nil, nil
			if !grouped {
				s.actmulAll(gate, up, act)
				s.jit.NewInput()
				s.tock(opAct, t0)
				if l.ffnSubNorm != nil {
					t0 = s.tick()
					s.rmsnorm(gate, gate, l.ffnSubNorm, c.RMSEps)
					s.jit.NewInput()
					s.tock(opRMSNorm, t0)
				}
				if err := s.mv(s.h, l.down, gate); err != nil {
					return nil, err
				}
			}
		}
		if l.postFFNNorm != nil {
			t0 = s.tick()
			s.rmsnorm(ffn, ffn, l.postFFNNorm, c.RMSEps)
			s.tock(opRMSNorm, t0)
		}
		t0 = s.tick()
		if !ffnResid {
			s.addInto(s.x, ffn)
		}
		ffnResid = false
		s.tock(opResid, t0)
		if c.AltUp != 0 {
			if err := s.altCorrect(li, l, s.xa, s.altPred, s.ple, 1); err != nil {
				return nil, err
			}
		} else if c.PLEDim != 0 {
			if err := s.pleApply(li, l, s.x, s.ple); err != nil {
				return nil, err
			}
		}
		s.layerOutScale(l, s.x)
		m.trace(li, "ffn_resid", s.x)
	}
	// AltUp's head reads the streams' mean, written into stream 0 (altup.go);
	// DeepSeek V4's their collapse (ds4.go).
	if c.AltUp != 0 {
		if err := s.altCollapse(s.xa, 1); err != nil {
			return nil, err
		}
	}
	if c.DSV4() {
		if err := s.ds4Collapse(s.xa, 1); err != nil {
			return nil, err
		}
	}
	if c.ResAttn() {
		if err := s.k3Collapse(s.xa, 1); err != nil {
			return nil, err
		}
	}
	if c.Q4E() {
		if err := s.q4eCollapse(s.xa, 1); err != nil {
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
		defer s.rows.hiddenOf(s.head)()
		if s.ld.Layers(s.gpuLayers, s.gpuLayers, s.pos, 1, s.resid(), s.cs[:c.RopeW()], s.swaTable(c.NRotSWA), s.head) {
			done = true
		} else if !s.devFallback {
			return nil, errDevice{s.gpuLayers, s.devWhy()}
		}
	}
	if !done {
		s.norm(s.h, s.x, s.outNorm, m.outNormB)
		s.jit.NewInput()
		m.trace(-1, "out_norm", s.h)
		if err := s.mv(s.logits, *s.outW, s.h); err != nil {
			return nil, err
		}
		if s.rows != nil {
			copy(s.rows.hidden, s.h)
		}
	}
	s.unfuse()
	// After both arms: when the tier ran the head it also applied the final
	// softcap, so that arm takes only the scale.
	if done {
		s.finishDevLogits(s.logits)
	} else {
		s.finishLogits(s.logits)
	}
	// Through advance, not s.pos++: Prefill reads bpos, so a Prefill after a
	// Forward must see this token.
	s.advance(0, 1)
	s.hot.endToken()
	s.reap()
	s.growStep()
	s.seamStep()
	// A page the store could not return fails the request; see kvFault.
	if err := s.kvCheck(); err != nil {
		return nil, err
	}
	return s.logits, nil
}

// Greedy returns the argmax token. Ties go to the lower id, which is what
// llama.cpp's top-k=1 does, and reproducibility of greedy decoding is what makes
// everything else regression-testable.
func Greedy(logits []float32) int32 {
	// Generated on every tier, with no Go scan behind it. A decline means a
	// host with no code generator, which nn.Available already refuses.
	i, ok := nn.Argmax32JIT(logits)
	if !ok {
		panic(fmt.Sprintf("model: no argmax kernel for %d logits on this host -- "+
			"greedy sampling is generated code and has no Go path", len(logits)))
	}
	return i
}

// errFull is the end of the context. It names newState's clamp when the caller
// asked for more positions than the model's context.
type errFull struct{ max, asked int }

func (e errFull) Error() string {
	s := "model: kv cache is full (" + itoa(e.max) + " positions)"
	if e.asked > e.max {
		s += "; " + itoa(e.asked) + " were requested and the model's context is " +
			itoa(e.max)
	}
	return s
}

type errToken struct {
	tok    int32
	nVocab int
}

func (e errToken) Error() string {
	return "model: token " + itoa(int(e.tok)) + " out of range for vocab " + itoa(e.nVocab)
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
func (s *State) swaTable(nrot int) []float32 {
	if s.csSWA == nil {
		return nil
	}
	return s.csSWA[:nrot]
}

// windowFor is the sliding window a device must apply in a session of maxSeq
// positions: none when the window is no shorter than the session, since then it
// never truncates anything and the windowed kernels would only cost a select.
func windowFor(c *Config, maxSeq int) int {
	if c.SWAWindow <= 0 || c.SWAWindow >= maxSeq {
		return 0
	}
	return c.SWAWindow
}

// swaPeriodFor is the plan's copy of Config.SWA's pattern, and it is zero unless
// the architecture trains two rotary bases: a sliding window alone is honoured
// through the per-row key count and needs no second table (gemma2).
func swaPeriodFor(c *Config) int {
	if c.RopeBaseSWA == 0 {
		return 0
	}
	return int(c.SWAPeriod)
}

// ropeTable picks the rotary table layer li was trained with: the local one on
// a sliding layer of an architecture with two bases (gemma3), the global one
// otherwise. It is a per-layer choice.
func (s *State) ropeTable(li int, global, local []float32, tok int) []float32 {
	c := s.c
	if local != nil && c.SWA(li) {
		return local[tok*c.NRotSWA : (tok+1)*c.NRotSWA]
	}
	return global[tok*c.RopeW() : (tok+1)*c.RopeW()]
}

// attnPrep is the tail of one token's attention projections, shared by decode,
// prefill and batch so the three cannot drift: the clamp, the optional q/k
// norm, RoPE on q and k (never v), and the write into the KV cache. cs is the
// cos/sin table for this token's position; kvSlot and kvPos say which sequence
// row and position its k and v belong at.
//
// k and v nil is a KV-sharing block (Config.KVShared): q alone is normed and
// rotated, and nothing is cached.
func (s *State) attnPrep(l *layer, q, k, v []float32, cs []float32, li, kvSlot, kvPos int) {
	c := s.c
	hd, nkv := c.HeadDimAt(li), c.NKVHeadAt(li)
	if k == nil {
		if c.QKNormAt(li) {
			nq, _, wq, _ := c.QKNormShapeAt(li)
			for hh := 0; hh < nq; hh++ {
				t := q[hh*wq : (hh+1)*wq]
				s.rmsnorm(t, t, l.qNorm, c.RMSEps)
			}
		}
		if c.RopeAt(li) {
			t0 := s.tick()
			nn.RoPE32JIT(q[:c.NHead*hd], hd, cs, c.RopeNeox)
			s.tock(opRoPE, t0)
		}
		return
	}
	// DBRX's clip_qkv, first: after the projections and their biases, before
	// any norm or rotary.
	if cl := c.ClampKQV; cl != 0 {
		nn.Clamp32JIT(q[:c.NHead*c.HeadDim], cl)
		nn.Clamp32JIT(k[:c.NKVHead*c.HeadDim], cl)
		nn.Clamp32JIT(v[:c.NKVHead*c.HeadDimV], cl)
	}
	qkNorm := c.QKNormAt(li)
	if qkNorm {
		// qwen3 norms each head over HeadDim, olmoe the whole projection at
		// once; QKNormShape is the one place that decides which. Whether to
		// norm at all is QKNormAt, the derivation the device's offer reads.
		nq, nk, wq, wk := c.QKNormShapeAt(li)
		if !c.QKNormPost {
			for hh := 0; hh < nq; hh++ {
				t := q[hh*wq : (hh+1)*wq]
				s.rmsnorm(t, t, l.qNorm, c.RMSEps)
			}
		}
		// Under XD-RoPE k's norm follows the rotary too (below): that rotation
		// does not keep a head's RMS.
		for hh := 0; hh < nk && (!c.RopeXD || xdKNormFirst); hh++ {
			t := k[hh*wk : (hh+1)*wk]
			s.rmsnorm(t, t, l.kNorm, c.RMSEps)
		}
	}
	// Per layer: Llama 4's NoPE layers (and a NoPE model) carry no rotary.
	if c.RopeAt(li) {
		t0 := s.tick()
		if c.RopeXD {
			nn.RoPESplit32JIT(q[:c.NHead*hd], hd, cs)
			nn.RoPESplit32JIT(k[:nkv*hd], hd, cs)
		} else {
			nn.RoPE32JIT(q[:c.NHead*hd], hd, cs, c.RopeNeox)
			nn.RoPE32JIT(k[:nkv*hd], hd, cs, c.RopeNeox)
		}
		s.tock(opRoPE, t0)
		// Llama 4's L2 norm goes after the rotary and has no weight: the same
		// generated RMSNorm against a ones vector.
		if c.QKL2Norm {
			hd := c.HeadDim
			for hh := 0; hh < c.NHead; hh++ {
				t := q[hh*hd : (hh+1)*hd]
				s.rmsnorm(t, t, s.m.onesHD, c.RMSEps)
			}
			for hh := 0; hh < c.NKVHead; hh++ {
				t := k[hh*hd : (hh+1)*hd]
				s.rmsnorm(t, t, s.m.onesHD, c.RMSEps)
			}
		}
	}
	// Hunyuan's q norm, after the rotary (or where it would be), weighted by
	// w_q*w_k (jlm.FlagQKNormPostRope): k's weight is folded into it, so the
	// scores are the model's while k, normed before, carries ones.
	if qkNorm && c.QKNormPost {
		for hh := 0; hh < c.NHead; hh++ {
			t := q[hh*hd : (hh+1)*hd]
			s.rmsnorm(t, t, l.qNorm, c.RMSEps)
		}
		if c.RopeXD && !xdKNormFirst {
			for hh := 0; hh < nkv; hh++ {
				t := k[hh*hd : (hh+1)*hd]
				s.rmsnorm(t, t, l.kNorm, c.RMSEps)
			}
		}
	}
	// Gemma 4 norms each head of v with no weight: the generated RMSNorm
	// against a ones vector, as Llama 4's L2 norm is.
	if c.VNorm {
		for hh := 0; hh < nkv; hh++ {
			t := v[hh*hd : (hh+1)*hd]
			s.rmsnorm(t, t, s.m.onesHD[:hd], c.RMSEps)
		}
	}
	// The page is allocated here, not faulted: growth is known, not
	// discovered -- unless trim evicted it with other history in it.
	s.kvWritable(li, kvPos)
	s.kv.layers[li].write(s.kvlAt(li), kvSlot, kvPos, k, v)
}

// layerOutScale multiplies the block's output row by its scalar (Gemma 4's
// layer_scalar), on the generated scale kernel; a block without one is
// untouched.
func (s *State) layerOutScale(l *layer, x []float32) {
	if l.outScale != 0 {
		s.scale(x, l.outScale)
	}
}

// narrow copies one run of f32 into the f32 cache.
func narrow(dst []float32, src []float32) { copy(dst, src) }

// errDevice is a device failure the State cannot continue past on the host.
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
func (s *State) devWhy() string {
	if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
		return ": " + e.Err()
	}
	return ""
}

// recurrent reports that this session carries a linear block's running state,
// which is what makes re-running a block from the embeddings unsound.
func (s *State) recurrent() bool { return s.rconv != nil || s.rstate != nil }

// RecurrenceAdvanced reports whether the device call that just failed had
// already advanced one of this State's linear blocks, so running its tokens
// again would apply them twice. It asks the State's own device session, which
// is where the count lives: the GPU's own count is session 0's, and a State
// steps on a session of its own. A joint step counts on the State that led it,
// so a caller asks every State in the step.
func (s *State) RecurrenceAdvanced() bool { return s.recurrent() && s.recStepped() != 0 }

// recStepped is how many placed linear blocks advanced during the submission
// that just failed. Zero means nothing moved and the host restart is sound; it
// is -1 when the tier cannot say, which is read as "assume it did". It is
// measured because both positional proxies (the run's start, or "any linear
// block placed") are wrong in one direction or the other.
func (s *State) recStepped() int {
	rs, ok := s.ld.(nn.RecStepper)
	if !ok {
		return -1 // a tier that cannot say; treat any failure as unsafe
	}
	return rs.RecSteps()
}

// devLinear reports that at least one linear block is on a device right now
// (the set, not the prefix), so a failed submission may have advanced a
// summary the host cannot rewind.
func (s *State) devLinear() bool {
	if s.rconv == nil {
		return false
	}
	for li := 0; li < len(s.onDev) && li < len(s.rconv); li++ {
		if s.onDev[li] && s.rconv[li] != nil {
			return true
		}
	}
	return false
}

// MaxSeq is how many positions this session preallocated its KV cache for.
func (s *State) MaxSeq() int { return s.maxSeq }

// ContextClamped reports whether the model's context is shorter than what was
// asked for, and by how much, so a caller can refuse rather than run short.
func (s *State) ContextClamped() (asked, got int, clamped bool) {
	return s.reqSeq, s.maxSeq, s.reqSeq > s.maxSeq
}

// setLD sets the State's layer device and the optional faces a token and a
// step ask it for, asserted once here: an interface-to-interface assertion
// caches its answer in a heap object at a random later call, which made a
// decode token's allocation count nondeterministic.
func (s *State) setLD(ld nn.LayerDevice) {
	s.ld = ld
	s.ldRoom, _ = ld.(nn.RoomReporter)
	s.ldSpill, _ = ld.(nn.Spiller)
	s.ldStep, _ = ld.(nn.SessionStepper)
	s.ldRows, _ = ld.(nn.RowsDevice)
	s.ldNamed, _ = ld.(nn.NamedDevice)
	s.ldInputs, _ = ld.(nn.LayerInputDevice)
	s.ldTokens, _ = ld.(nn.TokenDevice)
	s.ldNGram, _ = ld.(nn.NGramDevice)
	s.ldQSA, _ = ld.(nn.QSADevice)
}

// layerInputs hands the device the rows' per-layer embedding inputs for the
// call about to be made (nn.LayerInputDevice). Nothing where the model has
// none.
func (s *State) layerInputs(in []float32) {
	if s.ldInputs != nil && s.c.PLEDim != 0 {
		s.ldInputs.SetLayerInputs(in)
	}
}
