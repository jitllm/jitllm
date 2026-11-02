package gpu

import (
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
)

// A whole transformer block on the device.
//
// The point is one Session per block with exactly one Read: serving matvecs one
// at a time costs a blocking round trip each, which alone was a quarter of a
// gemma token. That is also why every op between the matvecs is a kernel -- a
// single host-side RMSNorm would drag the residual back across the bus.

// layer is one block's device-resident state.
type layer struct {
	// graph is the block as ops from the model's front end
	// (nn.LayerPlan.Graph): what serving it whole forms (served.go), and
	// win and chunked the window its attention states.
	graph          *graph.Graph
	win            int
	chunked        bool
	wq, wk, wv, wo *resident
	gate, up, down *resident
	// qwen3next's shared expert: an always-on feed-forward added to the routed
	// sum, scaled by a sigmoid of one logit. Its router is a vector, uploaded as
	// plain floats; mvShRouter is a one-row RouterMatVec.
	shGate, shUp, shDown *resident
	mvsg, mvsu, mvsd     mv
	shRouter             backend.Buf
	// Gemma 4's per-layer embedding (nn.LayerPlan.PLEDim): the gate and the
	// projection back, the post-norm, and pleOff the block's li*PLEDim, the
	// offset of its slice in a row's inputs (kernels.PLESlice).
	pleGate, pleProj *resident
	mvpg, mvpp       mv
	nPLE, pleOff     backend.Buf
	// Gemma 3n's AltUp (nn.LayerPlan.AltUp, altup.go): the router as plain
	// floats and its norm, the coefficients transposed, the scale before the
	// per-layer gate; LAuReL's two matrices and norm; sparse says the FFN
	// gate takes the gaussian top-k.
	altRouter, altRouterNorm, altPredT, altCorrT, altCorrScale backend.Buf
	// altRouterW is altRouter as the composer binds a matrix; it owns
	// nothing (altRouter is freed with the block).
	altRouterW       *resident
	laurelL, laurelR *resident
	mvLL, mvLR       mv
	nLaurel          backend.Buf
	sparse           bool
	// kvSrc is the block whose history this one attends to: its own index, or
	// for a KV-sharing block (nn.LayerPlan.KVShared) the source's, whose
	// cache this block reads and never writes.
	kvSrc int
	// A linear block's three matrices and five vectors. Wq above carries the
	// fused q|k|v for such a block (the layer kind decides the consumer, as in
	// nn.SSMWeights), so these are the z projection, the beta/alpha head and the
	// output projection.
	//
	// ssmOnes is a unit weight of kDim: the per-key-head L2 norm is an RMSNorm with
	// a unit weight and eps/n, as engine/model/delta.go builds it.
	// linear says this block runs the gated delta rule instead of attention. It is
	// set from the plan, not inferred from a nil weight.
	linear                 bool
	ssmGate, ssmBA, ssmOut *resident
	// centred is a short convolution with no state over a whole non-causal
	// call (nn.RecurrentPlan.Centred).
	centred bool
	// KDA's two low-rank gates, nil on qwen3next.
	ssmFA, ssmFB, ssmGA, ssmGB *resident
	mvSG, mvBA, mvSO           mv
	mvFA, mvFB, mvGA, mvGB     mv
	ssmConv1d, ssmA, ssmDt     backend.Buf
	ssmNorm, ssmOnes           backend.Buf
	// ssmTaps is the convolution's taps as the file stores them, a
	// channel's contiguous: the composer's state convolution reads them so.
	ssmTaps resident
	// ssmD and ssmCB are a Mamba-2 block's skip and convolution bias
	// (nn.RecurrentPlan.SSD), and a Mamba-1 block's; nil on the gated delta
	// rule.
	ssmD, ssmCB backend.Buf
	// m1DtN, m1BN and m1CN are a Mamba-1 block's dt, B and C RMSNorm weights
	// (Jamba's; FalconMamba's ones); nil where it has none. Its four matrices
	// ride KDA's four (nn.SSMWeights).
	m1DtN, m1BN, m1CN backend.Buf
	// withAttn is a block that runs attention beside its Mamba-2 mixer
	// (nn.RecurrentPlan.WithAttn, Falcon-H1): it keeps a KV history and a
	// recurrent state both, and its mixer's projection is ssmIn rather than
	// wq, which is the attention's query.
	withAttn bool
	ssmIn    *resident
	mvSI     mv
	// noFFN is a block that is its mixer alone (nn.LayerPlan.NoFFN): the
	// emit ends it at the first residual.
	noFFN bool
	// Multi-head latent attention's five matrices and two norms. wkb and wvb are
	// banks, one sheet per head, so their matvecs carry slots like a mixture's and
	// the "expert" indexed is the head. wq carries the one-step query where the
	// architecture has one and is empty where wqa/wqb do; wk and wv are always
	// empty. mla is set from the plan, like linear.
	mla                      bool
	wqa, wqb, wkva, wkb, wvb *resident
	mvQA, mvQB, mvKVA        mv
	mvKB, mvVB               mv
	nQA, nKVA                backend.Buf
	// DeepSeek V3.2's lightning indexer (nn.LayerPlan.IdxHeads, indexer.go):
	// its query, key and weight projections and the key's LayerNorm.
	idxQB, idxK, idxProj *resident
	mvIQ, mvIK, mvIW     mv
	nIdxK, nIdxKB        backend.Buf
	// MiniMax Sparse Attention (nn.LayerPlan.MSA, msa.go): msa says the block
	// selects its key blocks; idxQ/mvMQ is its indexer query and nIdxQ that
	// query's per-head norm, the key being idxK/mvIK with nIdxK.
	msa bool
	// qsa is a Qwen3.8-Flash-Next attention block's QSA selection (qsa.go).
	qsa   bool
	idxQ  *resident
	mvMQ  mv
	nIdxQ backend.Buf
	// DeepSeek V4 (nn.LayerPlan.DS4, ds4.go): the block's vectors, the
	// grouped output's first half (a bank of OGroups sheets, slot 31) and its
	// twins at a batch's rows, and the compressors' projections (32..35).
	ds4                               *ds4Block
	woA, compKV, compGate             *resident
	idxCompKV, idxCompGate            *resident
	mvWoA, mvCKV, mvCG, mvICKV, mvICG mv
	// Kimi-K3 (nn.LayerPlan.K3, k3.go): the MLA output gate (slot 36) and the
	// latent mixture's down and up projections (37, 38), and the block's
	// vectors (the residual attention's scores, the routed sum's norm).
	mlaGate, routedDown, routedUp *resident
	mvMG, mvRD, mvRU              mv
	k3                            *k3Block
	// Qwen3.8-Flash-Next (nn.LayerPlan.Q4E, q4e.go): the two
	// hyper-connections' down, up and placement projections (slots 39..44),
	// the n-gram key and value (45, 46), and the block's vectors.
	hcAD, hcAU, hcAI, hcFD, hcFU, hcFI       *resident
	ngKey, ngVal                             *resident
	mvHAD, mvHAU, mvHAI, mvHFD, mvHFU, mvHFI mv
	mvNGK, mvNGV                             mv
	q4e                                      *q4eBlock
	// conv is a convolutional tower block's own buffers (convblock.go), nil
	// for every other block.
	conv *convBlock

	nAttn, nFFN backend.Buf // norm weights, f32
	// nPostAttn and nPostFFN are gemma2/gemma3's two extra RMSNorms, applied to
	// the attention and FFN outputs before their residual adds. nil elsewhere.
	nPostAttn, nPostFFN backend.Buf
	// nSubAttn and nSubFFN are BitNet's inner norms (nn.LayerPlan.SubNorm):
	// over the attention output before wo and over act(gate)*up before down,
	// each fused with the int8 quantize its projection reads (bs.subQA,
	// bs.subQF). nil elsewhere.
	nSubAttn, nSubFFN backend.Buf
	// clampB is a vision block's clipped-linear bounds (nn.LayerWeights.Clamp,
	// Gemma 4), 28 floats the clamp kernels read; nil elsewhere.
	clampB backend.Buf
	// Gemma 4's mixture block (nn.LayerPlan.DenseMoE): the experts' pre-norm,
	// the dense MLP's and the experts' post-norms, the router input's norm and
	// the per-expert weight factor. nil elsewhere.
	nFFN2, nPost1, nPost2, nRouter, expScale backend.Buf
	// LayerNorm biases, nil for every block that RMSNorms.
	nAttnB, nFFNB backend.Buf
	// nResid and nResidB are a post-norm block's norm over x + mlp(x)
	// (nn.LayerPlan.PostResidNorm, BERT's), nil elsewhere.
	nResid, nResidB backend.Buf
	nQ, nK          backend.Buf // per-head q/k RMSNorm weights (qwen3), nil otherwise
	// The KV cache is per sequence, not per block: nn.LayerDevice.Layers carries a
	// position and no state identity, so two model.instances sharing one pair would
	// silently corrupt each other. See docs/design/device-sessions.md.
	//
	// kv is keyed by session id; the weights, norms and biases above are the
	// model's and stay shared. [MaxSeq*KVDim] f32 per entry.
	kv map[uint64]*kvPair
	// rec is the recurrent state of a linear block, per session: which half
	// of pool the session reads. It is a separate map from kv because the two
	// never coexist: a block keeps an attention history or a running summary,
	// never both.
	rec map[uint64]*recPair
	// pool holds every session's recurrent state for a linear block, each at
	// its seat's slots (recpool.go); nil until a session needs it.
	pool *recPool
	// nonCausal says the block's rows attend over their call's key runs and
	// keep no history (nn.LayerPlan.NonCausal): a call over it is one whole
	// run, submitted in its geometry's scratch set (geoNonCausal).
	nonCausal bool
	// windowed says the block attends inside the windowed runs its call was
	// handed (nn.LayerPlan.Windowed, SetKeyRuns). Per block: the scratch's
	// plan is its first block's, and Qwen2.5-VL's blocks alternate.
	windowed bool
	// picPool is the device's transient pair as the pool a composed
	// non-causal block reads it through (towerStep).
	picPool kvLayerPool
	// A non-causal block's own MLP and rotary, which may differ from the
	// first block's the set was built from: an encoder's decision head
	// (Laya's) is an ungated ReLU MLP wider than the encoder's gated GELU,
	// and rotates nothing where the encoder rotates every head. nffn is its
	// width, ungated whether it has no gate, actK its activation where the
	// set's is another (nil: the set's bs.actMul), noRope that it rotates
	// nothing. Read on the non-causal path only.
	nffn    int
	ungated bool
	act     graph.ActKind
	actK    backend.Kernel
	noRope  bool
	// swa says the block attends at the sliding layers' geometry, so it runs
	// in that geometry's scratch set (gemma4.go). outScale is its output
	// scalar, one float, nil where it has none.
	geo      geoKey
	vFromK   bool
	outScale backend.Buf
	// xielu is an Apertus block's four activation numbers (kernels.XIELU's
	// pP), nil elsewhere.
	xielu         backend.Buf
	mvq, mvk, mvv mv
	// shadow is the shadow arm's state for the block (Config.ComposedShadow):
	// the worst drift of the form from the tier's kernels and the steps
	// compared, two floats the comparison keeps (shadow.go).
	shadow        backend.Buf
	mvo, mvg, mvu mv
	mvd           mv
	// MoE only: the F32 router matrix and its matvec. gate/up/down above hold
	// the banks in that case, and their mv carries slots > 0.
	router backend.Buf
	// routerF16 is whether l.router holds binary16 (Mixtral) rather than f32;
	// the batched mixture picks its router kernel by it.
	routerF16 bool
	// routerW is l.router as the composer binds a matrix: its buffer, not a
	// second copy of it.
	routerW resident
	// gpt-oss's per-head attention sinks and its router and expert biases, nil
	// for every other architecture (LayerPlan.AttnSinks, MoEBias).
	sinks, routerB             backend.Buf
	expGateB, expUpB, expDownB backend.Buf
	// expSelB is DeepSeek's e_score_correction_bias: a different tensor from
	// routerB, added to the scores after gating and only for the selection.
	expSelB  backend.Buf
	kvBytes  uint64 // this block's KV cache, so ReleaseLayers can refund it
	recBytes uint64 // and its recurrent state, refunded on the same walks
	recHist  uint64 // one session's recurrent pair; see devTier.reserved
	// auxBytes is everything else charged to the budget for this block (the
	// norms), so ReleaseLayers refunds it too.
	auxBytes  uint64
	biasBytes uint64
	// pg is this block's pageable weights -- the seven matrices and nothing else.
	// It is built for every block, on every model, because the budget rather than
	// the fit is what decides whether anything ever moves; see page.go.
	pg *page
	// stream is non-nil when this block's routed bank is not resident: gate, up
	// and down hold NExpertUsed sheets each, refilled every token out of the
	// router's own selection. nil is the ordinary block, where the whole bank
	// sits on the card and nothing crosses the bus inside a token.
	stream *streamBank
	ok     bool
}

// streamBank is the host side of a block whose routed experts are assembled per
// token, and the suspension that makes that possible.
//
// A mixture routing few of many experts can make a resident bank tens of times
// the bytes a token reads, so on a small card the choice is streaming or the
// host. The price is a host round trip mid-block: the selection must be read back
// before anyone knows which sheets to send, so the submission suspends after
// ExpertRank and resumes after the upload, and such a block cannot be captured
// into a graph (a recording has nowhere to put a readback).
type streamBank struct {
	// sh is the per-expert sheet size of each plane, in bytes, for gate, up and
	// down in that order: [matrix][qs, d, sc].
	sh [3][3]int
	// sel is the selection, read back each token. It is len NExpertUsed and is
	// reused rather than allocated per token.
	sel []uint32
	// w is the host side of this block. ensure re-points its spans at wherever
	// the host pager holds the block now, a different frame every token on a
	// model that pages (see nn.LayerWeights.Ensure).
	w      nn.LayerWeights
	ensure func(*nn.LayerWeights) error
	// sel2 is ensure narrowed to the routed sheets. When it is non-nil the
	// whole-bank read never happens; see nn.LayerWeights.EnsureExperts.
	sel2 func(*nn.LayerWeights, []uint32) error
	// pre is sel2's read half, safe to call from another goroutine. When it is
	// non-nil fill pipelines: group g uploads while group g+1 reads. See
	// nn.LayerWeights.PrefetchExperts.
	pre func([]uint32) error
	// rd and grp are fill's reads in flight and its groups, kept so a token
	// allocates neither; items and want are its work list and the experts
	// it reads.
	rd    []streamRead
	grp   [][2]int
	items []streamItem
	want  []uint32
	// The expert cache (initCache), nil without one: slotOf maps an expert
	// to the slot holding it or -1, owner a slot to its expert or -1, used a
	// slot to the tick it was last selected; slotSel is this token's
	// selection in slots.
	slotOf  []int32
	owner   []int32
	used    []uint64
	tick    uint64
	slotSel []uint32
	bank    *resident
	// plain is the bank a cache sized after placement replaced (cacheBlock),
	// kept so dropCaches can give the cache back.
	plain [3]*resident
	// nExpert is the bank's expert count, so a cache can be set up after
	// placement (GPU.sizeAutoCaches).
	nExpert int
	// spec hands the host's read service a guess and nil abandons the
	// unstarted ones (nn.LayerWeights.SpeculateExperts).
	spec func(sel []uint32)
	// host is the block's routed experts on the host (hybrid execution), and
	// hin, hw and hout its input, weights and output, reused every token.
	host   func(sel []uint32, w, in, out []float32) error
	hybrid bool // the experts run on the host (runHost), not sent
	li     int  // the block, for the session's host side (hostFor)
	// The split (hybrid with a cache): heat counts each expert's selections,
	// halved every heatHalf tokens; hit marks which of this token's
	// selection the cache holds; dw is the card's weights (a miss's zeroed),
	// and msel and mw the misses the host runs.
	heat []uint32
	own  []uint32
	// owned counts the tokens st.own has counted, for its halving.
	owned uint64
	hit   []bool
	// tab is a composed step's page table (servedPart.pagedRun): each
	// expert's sheet of the bank, reused every token; rsel the selections
	// of a call of several rows.
	tab, rsel []uint32
}

// growU32 returns s resized to n, reusing its array when it is big enough.
func growU32(s []uint32, n int) []uint32 {
	if cap(s) < n {
		return make([]uint32, n)
	}
	return s[:n]
}

// fill uploads the k sheets this token routed to, in selection order, so that
// compact slot i is selected expert i and bs.rw[i] is its weight. The order is
// the contract: ExpertRank writes rsel and rtop in descending-logit order and
// the combine pairs slot i with rw[i], so sorting by expert id would weight
// every expert with another's gate.
func (st *streamBank) fill(g *devTier, s backend.Session, l *layer, nExpert int) error {
	// ensure and the uploads below are one window over a reused host frame, so an
	// eviction in between would send another block's weights to the card.
	// model.enterPager holds the pager for the whole Forward, which closes this
	// window and the other three like it; jlm.Lease is the finer primitive.
	//
	// Every group's read is issued up front and group g uploads as soon as its
	// own read has landed, so the pager has the whole selection in flight and
	// the transfers overlap the reads still running. Only the first group goes
	// through sel2, which re-points st.w and binds; the reads go through pre,
	// which touches nothing shared -- two sel2 calls in flight would race on
	// the model's layer. A bank in expert pages gets nothing from sel2 (its
	// sheets are reached through Sheet), so the first group is read through pre
	// too: without it every page was faulted in by Sheet one at a time, inside
	// the upload loop (placement.md 16c).
	// The work is a list of (expert, slot) items: every selected expert into
	// slot i without a cache, and only the misses, into the slots they evict,
	// with one.
	st.items = st.items[:0]
	if st.cached() {
		// A bank paged out and back in is a new buffer with nothing in it, so
		// the map is only good for the buffer it was built over.
		if l.down != st.bank {
			st.forget()
			st.bank = l.down
		}
		st.lookup(g)
	} else {
		for i, e := range st.sel {
			st.items = append(st.items, streamItem{e: e, slot: i})
		}
	}
	if len(st.items) == 0 {
		return nil
	}
	st.want = st.want[:0]
	for _, it := range st.items {
		st.want = append(st.want, it.e)
	}
	grp := st.groups(g, len(st.items))
	t0 := time.Now()
	// The previous block's guesses that no reader has started are dropped:
	// whatever this fill needs, it reads itself below, and a guess already
	// being read is joined by the read of the same page rather than waited
	// for (jlm.File.Speculate), so a wrong guess delays nothing.
	if st.spec != nil {
		st.spec(nil)
	}
	if st.pre != nil {
		if cap(st.rd) < len(grp) {
			st.rd = make([]streamRead, len(grp))
		}
		st.rd = st.rd[:len(grp)]
		for gi := 1; gi < len(grp); gi++ {
			st.rd[gi].wg.Add(1)
			g.StreamOverlaps++
			go st.read(gi, grp[gi])
		}
	}
	var err error
	if st.pre != nil {
		err = st.pre(st.want[grp[0][0]:grp[0][1]])
	}
	switch {
	case err != nil:
	case st.sel2 != nil:
		// The narrow read is preferred: the whole bank is correct but tens of
		// times the bytes the token consumes.
		err = st.sel2(&st.w, st.want[grp[0][0]:grp[0][1]])
	case st.ensure != nil:
		err = st.ensure(&st.w)
	}
	g.TStreamRead += time.Since(t0)
	for gi := range grp {
		// Every read is joined, even after a failure: a goroutine still reading
		// into a frame this block is about to release is the eviction hazard
		// above.
		if gi > 0 && st.pre != nil {
			t2 := time.Now()
			st.rd[gi].wg.Wait()
			g.TStreamRead += time.Since(t2)
			if err == nil {
				err = st.rd[gi].err
			}
		}
		if err == nil {
			t1 := time.Now()
			err = st.put(g, s, l, nExpert, grp[gi])
			g.TStreamPut += time.Since(t1)
		}
	}
	return err
}

// hostSide is a session's host side for one hybrid block: a row at a time,
// and a chunk's rows expert-major.
type hostSide struct {
	one  func(sel []uint32, w, in, out []float32) error
	rows func(sel []uint32, w, in, out []float32, rows, k int) error
	// uses is the session's counts of this block's experts
	// (nn.LayerWeights.ExpertUses).
	uses []uint32
}

// streamItem is one sheet set a fill sends: expert e into compact slot slot.
type streamItem struct {
	e    uint32
	slot int
}

// initCache sets a block's bank up as an expert cache of n slots when n is
// more than the selection; at n == k there is no cache and fill sends every
// selected expert into slot i.
func (st *streamBank) initCache(n, nExpert int) {
	if n <= len(st.sel) {
		return
	}
	st.slotOf = make([]int32, nExpert)
	for i := range st.slotOf {
		st.slotOf[i] = -1
	}
	st.owner = make([]int32, n)
	for i := range st.owner {
		st.owner[i] = -1
	}
	st.used = make([]uint64, n)
	st.slotSel = make([]uint32, len(st.sel))
}

// uncache returns a block to its plain bank: no cache, nothing held.
func (st *streamBank) uncache() {
	st.slotOf, st.owner, st.used, st.slotSel = nil, nil, nil, nil
	st.heat, st.own, st.hit, st.bank = nil, nil, nil, nil
}

// cached reports whether this block's bank is an expert cache.
func (st *streamBank) cached() bool { return st.owner != nil }

// lookup maps this token's selection onto cache slots: a hit keeps its slot,
// a miss takes the least recently used slot this token does not need, and
// only the misses become items. slotSel is the selection in slots, which the
// kernels index by.
func (st *streamBank) lookup(g *devTier) {
	st.tick++
	for _, e := range st.sel {
		if s := st.slotOf[e]; s >= 0 {
			st.used[s] = st.tick
		}
	}
	for i, e := range st.sel {
		if s := st.slotOf[e]; s >= 0 {
			st.slotSel[i] = uint32(s)
			g.StreamCacheHits++
			continue
		}
		// The victim: the oldest slot not touched this token. There always is
		// one, since the cache holds more slots than a token selects.
		v := -1
		for s := range st.owner {
			if st.used[s] != st.tick && (v < 0 || st.used[s] < st.used[v]) {
				v = s
			}
		}
		if o := st.owner[v]; o >= 0 {
			st.slotOf[o] = -1
		}
		st.owner[v], st.slotOf[e], st.used[v] = int32(e), int32(v), st.tick
		st.slotSel[i] = uint32(v)
		st.items = append(st.items, streamItem{e: e, slot: v})
		g.StreamCacheMisses++
	}
}

// heatHalf is how many tokens a block's own selection counts last at full
// weight, for a session that hands the tier none (nn.LayerWeights.ExpertUses
// is the model's, which halves its own).
const heatHalf = 256

// count adds this token's selection, which the card routed, to uses -- the
// session's own counts of the block's experts (nn.LayerWeights.ExpertUses,
// model.Hotness), one source for the model's measure of its experts and for
// the expert cache -- or to the block's own counts when the session handed
// none, halved every heatHalf tokens. The cache keeps by st.heat.
func (st *streamBank) count(uses []uint32) {
	if len(uses) == st.nExpert && st.nExpert > 0 {
		st.heat = uses
	} else {
		if len(st.own) != st.nExpert {
			st.own = make([]uint32, st.nExpert)
		}
		st.heat = st.own
		st.owned++
		if st.owned%heatHalf == 0 {
			for i := range st.heat {
				st.heat[i] >>= 1
			}
		}
	}
	for _, e := range st.sel {
		if int(e) < len(st.heat) && st.heat[e] != ^uint32(0) {
			st.heat[e]++
		}
	}
}

// slotInFlight marks an expert whose sheet is on its way to a slot: a miss
// until the sheet has landed (finishAdmits), and not admitted again.
const slotInFlight = -2

// admJob is one block's admissions waiting to be sent: g.admItems[lo:hi].
type admJob struct {
	st      *streamBank
	l       *layer
	nExpert int
	lo, hi  int
}

// flushAdmits sends the admissions admit queued. It runs where the host waits
// on the card anyway -- before a block's router readback, and at the call's
// end -- so a sheet's page read and host copy are off the token's path. On a
// session with a copy queue (backend.Uploader) the sheets are copied into a
// page-locked ring and sent beside the launches, and the slots are taken
// into use at the call's end (finishAdmits), once the copies have landed;
// otherwise they are sent in order with the launches (put) and in use at once.
func (g *devTier) flushAdmits(s backend.Session) error {
	if len(g.adm) == 0 {
		return nil
	}
	defer func(t0 time.Time) { g.TAdmit += time.Since(t0) }(time.Now())
	up, _ := s.(backend.Uploader)
	if up != nil && !g.upRingReady() {
		up = nil
	}
	var err error
	for _, j := range g.adm {
		st := j.st
		items := g.admItems[j.lo:j.hi]
		st.want = st.want[:0]
		for _, it := range items {
			st.want = append(st.want, it.e)
		}
		switch {
		case st.pre != nil:
			err = st.pre(st.want)
		case st.ensure != nil:
			err = st.ensure(&st.w)
		}
		if err == nil && st.sel2 != nil {
			err = st.sel2(&st.w, st.want)
		}
		if err == nil && up != nil {
			err = st.upload(g, up, j.l, j.nExpert, items)
			if err == nil {
				g.admPub = append(g.admPub, j)
				g.SplitQueued += len(items)
			}
		} else if err == nil {
			st.items = append(st.items[:0], items...)
			if err = st.put(g, s, j.l, j.nExpert, [2]int{0, len(items)}); err == nil {
				st.publish(items)
			}
		}
		if err != nil {
			// A sheet half sent is not the expert the map names.
			st.forget()
			break
		}
		g.SplitAdmits += len(items)
	}
	g.adm = g.adm[:0]
	return err
}

// finishAdmits sends what is queued, waits for the copy queue and takes the
// landed slots into use. It runs at the end of every call that admitted,
// so no sheet is in flight between calls (dropCaches frees banks there).
func (g *devTier) finishAdmits(s backend.Session) error {
	err := g.flushAdmits(s)
	if len(g.admPub) > 0 {
		t0 := time.Now()
		if up, ok := s.(backend.Uploader); ok {
			if werr := up.UploadWait(); err == nil {
				err = werr
			}
		}
		for _, j := range g.admPub {
			if err != nil {
				j.st.forget()
				continue
			}
			j.st.publish(g.admItems[j.lo:j.hi])
		}
		g.admPub = g.admPub[:0]
		g.TAdmit += time.Since(t0)
	}
	if len(g.adm) == 0 {
		g.admItems = g.admItems[:0]
	}
	g.ringUsed = 0
	return err
}

// publish takes landed sheets into use: an expert whose slot is still its own
// (no forget or later eviction moved it) is a hit from the next token.
func (st *streamBank) publish(items []streamItem) {
	for _, it := range items {
		if st.owner != nil && it.slot < len(st.owner) && st.owner[it.slot] == int32(it.e) &&
			int(it.e) < len(st.slotOf) && st.slotOf[it.e] == slotInFlight {
			st.slotOf[it.e] = int32(it.slot)
		}
	}
}

// upRingBytes is the page-locked ring flushAdmits copies sheets through.
const upRingBytes = 64 << 20

// upRingReady allocates the ring on first use, and reports false where the
// device has no page-locked memory.
func (g *devTier) upRingReady() bool {
	if g.upRing != nil {
		return true
	}
	hp, ok := g.dev.(backend.HostPinner)
	if !ok || g.noPin || g.StreamNoPin {
		return false
	}
	b, err := hp.PinHost(upRingBytes)
	if err != nil {
		return false
	}
	g.upRing = b
	return true
}

// upload copies items' sheets through the ring onto the copy queue, at their
// slots. A sheet plane larger than the ring goes the ordered way.
func (st *streamBank) upload(g *devTier, up backend.Uploader, l *layer, nExpert int, items []streamItem) error {
	for m, pair := range [3]struct {
		r *resident
		w nn.Weight
	}{{l.gate, st.w.Gate}, {l.up, st.w.Up}, {l.down, st.w.Down}} {
		if m == 0 && pair.r == nil && pair.w.Packed == nil && pair.w.Data == nil {
			continue
		}
		pk := pair.w.Packed
		float := pk == nil && layout.IsFloat(pair.r.t) && len(pair.w.Data) > 0
		if pair.r == nil || !pair.r.ok || pk == nil && !float {
			return fmt.Errorf("tier: streamed bank %d has no host side", m)
		}
		for pl := 0; pl < 3; pl++ {
			n := st.sh[m][pl]
			if n == 0 {
				continue
			}
			if n > len(g.upRing) {
				return fmt.Errorf("tier: a %d-byte sheet plane is wider than the upload ring", n)
			}
			dst := [3]backend.Buf{pair.r.qs, pair.r.d, pair.r.sc}[pl]
			for _, it := range items {
				if int(it.e) >= nExpert {
					return fmt.Errorf("tier: the router selected expert %d of %d", it.e, nExpert)
				}
				if g.ringUsed+n > len(g.upRing) {
					// Every copy from the ring has landed before it is reused.
					if err := up.UploadWait(); err != nil {
						return err
					}
					g.ringUsed = 0
				}
				buf := g.upRing[g.ringUsed : g.ringUsed+n : g.ringUsed+n]
				if !float && pk.Sheet != nil {
					sp, rel, err := pk.Sheet(int(it.e))
					if err != nil {
						rel()
						return err
					}
					src := [3][]byte{sp.QS, sp.D, sp.SC}[pl]
					if len(src) != n {
						rel()
						return fmt.Errorf("tier: expert %d plane %d is %d bytes, want %d", it.e, pl, len(src), n)
					}
					copy(buf, src)
					rel()
				} else {
					// A float bank's rows are its one plane.
					whole := pair.w.Data
					if !float {
						whole = [3][]byte{pk.QS, pk.D, pk.SC}[pl]
					}
					off := int(it.e) * n
					if off+n > len(whole) {
						return fmt.Errorf("tier: expert %d plane %d wants bytes %d..%d of %d",
							it.e, pl, off, off+n, len(whole))
					}
					copy(buf, whole[off:off+n])
				}
				g.ringUsed += n
				if err := up.Upload(dst, it.slot*n, buf); err != nil {
					return err
				}
				g.StreamBytes += int64(n)
			}
		}
	}
	return nil
}

// forget empties the cache, for when the bank's bytes can no longer be
// trusted to be the experts the map says (a fill that failed midway).
func (st *streamBank) forget() {
	for i := range st.slotOf {
		st.slotOf[i] = -1
	}
	for i := range st.owner {
		st.owner[i] = -1
	}
}

// cacheSlotsFor is a streamed block's bank size in sheets: the selection, or
// want (Config.StreamCacheSlots) when that is bigger and the device has room for
// the block's own bank of that many beside what it already holds. A model
// whose expert kernels read the selection itself (DenseMoE's per-expert
// scale) keeps the plain bank: the cache rewrites the selection as slots.
func (g *devTier) cacheSlotsFor(p *nn.LayerPlan, ws []nn.Weight, want, slots, bank int, biased bool) int {
	// Nor one whose experts carry biases (gpt-oss) where the tier's own
	// kernels run it: they index the biases by the selection, which the cache
	// rewrites as slots. The composer's form reads only the banks in pages
	// through the page table.
	if p.DenseMoE || biased {
		return slots
	}
	var sheet uint64
	for i := 4; i <= 6; i++ {
		x := ws[i]
		if len(x.Data) == 0 && x.Packed == nil {
			continue
		}
		q, ok := quantOf(x.T)
		if !ok {
			return slots
		}
		pq, pd, psc, err := layout.PackedWords(q, x.Rows/slots, x.K)
		if err != nil {
			return slots
		}
		sheet += uint64(pq+pd+psc) * 4
	}
	n := min(want, bank)
	if n <= slots {
		return slots
	}
	need := sheet * uint64(n)
	if !g.room(need) {
		g.StreamCacheShort++
		return slots
	}
	g.StreamCacheSize = max(g.StreamCacheSize, n)
	return n
}

// streamRead is one group's read in flight.
type streamRead struct {
	wg  sync.WaitGroup
	err error
}

// read is group gi's read, run on its own goroutine. A method rather than a
// closure, so issuing it allocates nothing on the heap.
func (st *streamBank) read(gi int, gr [2]int) {
	defer st.rd[gi].wg.Done()
	st.rd[gi].err = st.pre(st.want[gr[0]:gr[1]])
}

// groups splits the selection into contiguous runs so a read can overlap an
// upload. One group is the unpipelined path and is what a host with no
// PrefetchExperts gets.
func (st *streamBank) groups(g *devTier, k int) [][2]int {
	n := g.streamGroups(g.StreamBlocks)
	if st.pre == nil || n < 2 || k < 2 {
		st.grp = append(st.grp[:0], [2]int{0, k})
		return st.grp
	}
	if n > k {
		n = k
	}
	st.grp = st.grp[:0]
	per := (k + n - 1) / n
	for lo := 0; lo < k; lo += per {
		hi := min(lo+per, k)
		st.grp = append(st.grp, [2]int{lo, hi})
	}
	return st.grp
}

// directSheet is the plane size from which a sheet is sent straight out of
// its host frame instead of being gathered first. A gather turns many small
// transfers into one, which pays below a megabyte; above it the per-call cost
// is noise and the gather is a second pass over the bytes -- on Kimi-K3 the
// largest term of the fill, 96 ms of 182 (placement.md 16c).
const directSheet = 1 << 20

// put uploads one group of sheets into the compact bank. The compact bank is k
// sheets back to back, so slots [lo, hi) are the byte range [lo*n, hi*n) of
// every plane. A plane of directSheet or more is sent sheet by sheet from the
// host frame; a smaller one is gathered into the device's staging buffer and
// sent once.
func (st *streamBank) put(g *devTier, s backend.Session, l *layer, nExpert int, gr [2]int) (err error) {
	items := st.items[gr[0]:gr[1]]
	// Without a cache the items are slots lo..hi-1 in order, so a small plane
	// can be gathered into one transfer; a cache's misses land in scattered
	// slots and every plane is sent sheet by sheet.
	lo, hi := items[0].slot, items[0].slot+len(items)
	contiguous := !st.cached()
	g.pieces = g.pieces[:0]
	// Every held sheet is released however put leaves, and the direct ones
	// are sent first.
	defer func() {
		if err == nil {
			err = g.sendPieces(s, g.pieces)
		}
		for i := range g.pieces {
			g.pieces[i].release()
			g.pieces[i] = sheetPiece{}
		}
		g.pieces = g.pieces[:0]
	}()
	for m, pair := range [3]struct {
		r *resident
		w nn.Weight
	}{{l.gate, st.w.Gate}, {l.up, st.w.Up}, {l.down, st.w.Down}} {
		// Ungated experts (Nemotron 3) have no gate bank to send.
		if m == 0 && pair.r == nil && pair.w.Packed == nil && pair.w.Data == nil {
			continue
		}
		pk := pair.w.Packed
		float := pk == nil && layout.IsFloat(pair.r.t) && len(pair.w.Data) > 0
		if pair.r == nil || !pair.r.ok || pk == nil && !float {
			return fmt.Errorf("tier: streamed bank %d has no host side", m)
		}
		for pl := 0; pl < 3; pl++ {
			n := st.sh[m][pl]
			if n == 0 {
				continue // this format has no such plane
			}
			dst := [3]backend.Buf{pair.r.qs, pair.r.d, pair.r.sc}[pl]
			direct := n >= g.directBytes() || !contiguous
			var buf []byte
			if !direct {
				buf = g.streamStage(m, pl, n*(hi-lo))
			}
			for _, it := range items {
				e, i := it.e, it.slot
				// The bound is checked because the index was read back from a device;
				// a garbage word would otherwise be a panic mid-token.
				if int(e) >= nExpert {
					return fmt.Errorf("tier: the router selected expert %d of %d", e, nExpert)
				}
				// A bank in expert pages hands over one sheet at a time; a
				// contiguous one is sliced at e*n.
				var src []byte
				release := func() {}
				ts := time.Now()
				if !float && pk.Sheet != nil {
					sp, rel, err := pk.Sheet(int(e))
					release = rel
					if err != nil {
						release()
						return err
					}
					src = [3][]byte{sp.QS, sp.D, sp.SC}[pl]
				} else {
					// A float bank's rows are its one plane.
					whole := pair.w.Data
					if !float {
						whole = [3][]byte{pk.QS, pk.D, pk.SC}[pl]
					}
					off := int(e) * n
					if off+n > len(whole) {
						return fmt.Errorf("tier: expert %d plane %d wants bytes %d..%d of %d",
							e, pl, off, off+n, len(whole))
					}
					src = whole[off : off+n]
				}
				if len(src) != n {
					release()
					return fmt.Errorf("tier: expert %d plane %d is %d bytes, want %d", e, pl, len(src), n)
				}
				tc := time.Now()
				g.TStreamSheet += tc.Sub(ts)
				if direct {
					// The frame stays held until sendPieces has copied it out.
					g.pieces = append(g.pieces, sheetPiece{dst: dst, off: i * n, src: src, release: release})
					continue
				}
				copy(buf[(i-lo)*n:(i-lo+1)*n], src)
				release()
				g.TStreamCopy += time.Since(tc)
			}
			if direct {
				continue
			}
			th := time.Now()
			if err := s.WriteAt(dst, lo*n, buf); err != nil {
				return err
			}
			g.TStreamH2D += time.Since(th)
			g.StreamBytes += int64(len(buf))
			g.StreamGathered++
		}
	}
	return nil
}

// sheetPiece is one direct sheet plane: where it goes on the card, its bytes
// in a held host frame, and the release of that hold.
type sheetPiece struct {
	dst     backend.Buf
	off     int
	src     []byte
	release func()
}

// pinHalf is the size of each page-locked half: two of Kimi-K3's 5.6 MiB
// planes. The first half of every group is copied before any transfer can
// start, so a smaller half is less of the group spent waiting; 32 MiB cost
// 29 s of copy wait over a 32-token run, and a sheet a group 62 s
// (placement.md 16c).
const pinHalf = 12 << 20

// copyChunk is the most one copy goroutine moves: a 5.6 MiB plane copied by
// one core ran under the link, so a half is copied by several.
const copyChunk = 1 << 20

// sendPieces puts direct sheets on the card. With page-locked memory it is a
// two-stage pipeline: the pieces are copied into one half (concurrently, one
// goroutine a piece, since a single memcpy from a remote node runs well under
// the link) while the other half is transferred, and a transfer out of
// page-locked memory runs at the link's rate where a pageable one went
// through the driver's bounce buffer at 5.76 GiB/s (placement.md 16c).
// Without it each piece is a pageable transfer from its frame.
func (g *devTier) sendPieces(s backend.Session, ps []sheetPiece) error {
	if len(ps) == 0 {
		return nil
	}
	pin := g.pinHalves(ps)
	if pin[0] == nil {
		for i := range ps {
			th := time.Now()
			if err := s.WriteAt(ps[i].dst, ps[i].off, ps[i].src); err != nil {
				return err
			}
			g.TStreamH2D += time.Since(th)
			g.StreamBytes += int64(len(ps[i].src))
			g.StreamDirect++
		}
		return nil
	}
	tc := time.Now()
	i, j := 0, g.packHalf(pin[0], ps, 0)
	g.TStreamCopy += time.Since(tc)
	cur := 0
	for i < len(ps) {
		next := j
		if j < len(ps) {
			g.packWG.Add(1)
			go g.packNext(pin[1-cur], ps, j)
		}
		off := 0
		var err error
		th := time.Now()
		for k := i; k < j && err == nil; k++ {
			n := len(ps[k].src)
			err = s.WriteAt(ps[k].dst, ps[k].off, pin[cur][off:off+n])
			off += n
			g.StreamBytes += int64(n)
			g.StreamDirect++
			g.StreamPinned++
		}
		g.TStreamH2D += time.Since(th)
		if j < len(ps) {
			// Joined even on failure: the copy reads frames put releases.
			tw := time.Now()
			g.packWG.Wait()
			g.TStreamCopy += time.Since(tw)
			next = g.packEnd
		}
		if err != nil {
			return err
		}
		i, j, cur = j, next, 1-cur
	}
	return nil
}

// pinHalves returns the two page-locked halves, each big enough for the
// largest piece, allocating them on first use; nils when the device has no
// page-locked memory or the caller turned it off.
func (g *devTier) pinHalves(ps []sheetPiece) [2][]byte {
	hp, ok := g.dev.(backend.HostPinner)
	if !ok || g.noPin || g.StreamNoPin {
		return [2][]byte{}
	}
	need := g.pinHalfBytes()
	for i := range ps {
		need = max(need, len(ps[i].src))
	}
	for h := range g.pin {
		if len(g.pin[h]) >= need {
			continue
		}
		if g.pin[h] != nil {
			hp.UnpinHost(g.pin[h])
			g.pin[h] = nil
		}
		b, err := hp.PinHost(need)
		if err != nil {
			// Page-locked memory is a speed, not a requirement: the pageable
			// path is correct, so a refusal turns this off for the device.
			g.noPin = true
			for k, b := range g.pin {
				if b != nil {
					hp.UnpinHost(b)
					g.pin[k] = nil
				}
			}
			return [2][]byte{}
		}
		g.pin[h] = b
	}
	return g.pin
}

// packNext is packHalf on its own goroutine, its end left in g.packEnd.
func (g *devTier) packNext(buf []byte, ps []sheetPiece, i int) {
	defer g.packWG.Done()
	g.packEnd = g.packHalf(buf, ps, i)
}

// packHalf copies pieces from i into buf, back to back, as many as fit, one
// goroutine a copyChunk, and returns the index after the last one copied. At
// least one always fits: pinHalves sized buf for the largest.
func (g *devTier) packHalf(buf []byte, ps []sheetPiece, i int) int {
	// The half in use is the tuned size, never more than was locked: a half
	// tuned down keeps its bigger buffer and uses a prefix of it.
	if h := g.pinHalfBytes(); h < len(buf) {
		big := 0
		for k := i; k < len(ps); k++ {
			big = max(big, len(ps[k].src))
		}
		buf = buf[:max(h, big)]
	}
	j, n := i, 0
	for j < len(ps) && n+len(ps[j].src) <= len(buf) {
		n += len(ps[j].src)
		j++
	}
	off := 0
	for k := i; k < j; k++ {
		src := ps[k].src
		for c := 0; c < len(src); c += copyChunk {
			e := min(c+copyChunk, len(src))
			g.copyWG.Add(1)
			go g.copyPiece(buf[off+c:off+e], src[c:e])
		}
		off += len(src)
	}
	g.copyWG.Wait()
	return j
}

// copyPiece is one of packHalf's copies. Only one packHalf runs at a time on
// a device, so the WaitGroup is the device's and nothing is allocated.
func (g *devTier) copyPiece(dst, src []byte) {
	defer g.copyWG.Done()
	copy(dst, src)
}

type mv struct {
	// pad is set on a matvec over a ragged tower's padded width
	// (nn.LayerPlan.KPad): its activation is the padded int8 rows, so a
	// batched twin that reads the float rows instead (an f16 conversion
	// at k) would read them at the wrong stride, and is not taken.
	pad  bool
	rows int
	// slots is 0 for an ordinary matvec and NExpertUsed for an indexed one, in
	// which case the launch is rows*slots*split threads and the kernel takes a
	// pSel parameter after pOut.
	slots int
	// bias is the per-row addend, nil for every projection that has none. The
	// kernel declares pBias after pOut, so the launch appends it.
	bias backend.Buf
	alt  *mv
	// q/k are the shape this was built from, so the batched twin can be
	// generated later without re-reading the weight.
	q layout.Quant
	k int
	// experts and slotAct are an indexed matvec's bank size and whether each
	// slot reads its own activation -- what retuneIndexed rebuilds it from.
	experts int
	slotAct bool
}

// streamStage is the reused host buffer one plane's gathered sheets go into.
// One set per device, not per block: a device fills its streamed blocks one
// after another, and a set per block was 24 GiB of heap on Kimi-K3 that the
// runtime gave back and faulted in again every token (placement.md 16c).
func (g *devTier) streamStage(m, pl, n int) []byte {
	i := m*3 + pl
	if len(g.sheetStage[i]) < n {
		g.sheetStage[i] = make([]byte, n)
	}
	return g.sheetStage[i][:n]
}

// kvPair is one sequence's attention history for one block.
type kvPair struct {
	kc, vc backend.Buf
	// bytes is what this pair charged, so growing it adjusts the ledger by the
	// difference rather than guessing.
	bytes uint64
	// paged marks history kept in the device's pool (pagedattn.go): kc and vc
	// are nil, the pages are the sequence's, and the pool carries the charge.
	paged bool
	// shared marks the one k/v pair every vision block of the device uses
	// (transientkv.go): it is released by reference count, never by one block.
	shared bool
}

// recPair is one session's recurrent state on one linear block: the gated
// delta rule's per-head matrices and the causal convolution's window, held at
// the session's seat in the block's pool (recpool.go), each in two halves.
// ir.Validate refuses a param that is both loaded and stored (clamped surplus
// threads would apply the update twice), so the state arrives through one
// half, leaves through the other, and the halves swap. Both are constant in
// the context length, so doubling them is cheap. In the shared form (one
// state per slot and the device's recNext, taken when two halves do not fit)
// only the window has halves.
type recPair struct {
	// cur is the half a submission reads; it writes the other and flips.
	cur int
	// steps counts the flips, so a rewind can tell one step from several
	// (RecRewind); mark is the half and count RecMark saw.
	steps     uint64
	mark      int
	markSteps uint64
	marked    bool
}

// ragLinear is one batch size's linear-block kernels (blockScratch.ragLin).
type ragLinear struct {
	conv, shift, delta backend.Kernel
	// ngConv is Qwen3.8-Flash-Next's n-gram convolution over the step's runs
	// (q4e.go), nil on every other block.
	ngConv backend.Kernel
}

// recOf returns this session's recurrent state for the block, or nil.
func (l *layer) recOf(sid uint64) *recPair {
	if l == nil || l.rec == nil {
		return nil
	}
	return l.rec[sid]
}

// permBytes is what admitting a fresh block charges beyond its weights and
// never gives back while it is resident: this session's history (a KV cache,
// or a linear block's recurrent pair) and the norms. prepLayer prices it into
// the room check, so a block is declined -- and the next device asked -- rather
// than admitted into room its own cache does not have. Callers hold g.mu, after
// initKVCap.
func (g *devTier) permBytes(p *nn.LayerPlan, w *nn.LayerWeights) uint64 {
	n := uint64((2*len(w.AttnNorm) + len(w.PostAttnNorm) + len(w.PostFFNNorm)) * 4)
	if r := p.Recurrent; r.Conv > 0 {
		return n + g.sessions()*recHist(p)
	}
	if g.pagedPlan(p) {
		return n + g.pagedLayerBytes(p)
	}
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	if p.NonCausal {
		// One pair for every non-causal block, charged once (transientkv.go).
		if g.tkv != nil && g.tkv.bytes >= uint64(cache+vcache) {
			return n
		}
		return n + uint64(cache+vcache)
	}
	return n + g.sessions()*uint64(cache+vcache)
}

// recHist is one session's recurrent pair for a linear block.
func recHist(p *nn.LayerPlan) uint64 {
	r := p.Recurrent
	return 2 * uint64(r.StateLen*4+r.ConvState*4)
}

func (g *devTier) sessions() uint64 { return uint64(max(1, g.Sessions)) }

// reserved is what the device holds back for sessions that have not attached
// yet: for every placed linear block, one recurrent pair per seat of
// Config.Sessions that no live session occupies. room() counts it, so neither
// a block nor a page can spend it; a session taking a seat (addSessionRec)
// spends its own share. An attention history holds no seat: every causal
// block's history is paged (initKVCap sets g.paged with the first text plan),
// and a page is taken as a sequence writes its rows, not when it attaches.
// Callers hold g.mu.
func (g *devTier) reserved() uint64 {
	if g.Sessions <= 1 {
		return 0
	}
	var n uint64
	for _, l := range g.layers {
		if l == nil || !l.ok || l.nonCausal || !l.linear {
			continue
		}
		if free := g.Sessions - len(l.rec); free > 0 {
			n += uint64(free) * l.recHist
		}
	}
	return n
}

// addSessionRec gives session sid the two-half recurrent state for a
// linear block: a seat in the device's pools (one slot, unless its batch has
// grown) and a pool for the block if it has none. Callers hold g.mu. The state
// is zeroed because a summary of garbage is a history the model never saw,
// and every token after it is fluent and wrong (RULE 13).
func (g *devTier) addSessionRec(sid uint64, l *layer, p *nn.LayerPlan) bool {
	r := p.Recurrent
	if r.Conv == 0 {
		return true // not a linear block; nothing to carry
	}
	g.recS, g.recC = r.StateLen, r.ConvState
	need := 2 * uint64(r.StateLen*4+r.ConvState*4)
	l.recHist = need
	if g.recSeats == nil {
		g.recSeats = map[uint64]recSeat{}
	}
	st, seated := g.recSeats[sid]
	if !seated {
		st = recSeat{base: g.seatPlace(sid, 1), rows: 1}
	}
	slots := max(g.recSlots, st.base+st.rows)
	// The device's first pool takes the form Config.SharedRec asks for; a
	// later one takes the pools' form.
	shared := g.recShared
	if g.recSlots == 0 {
		shared = g.SharedRec
	}
	slot := g.recSlotBytes(shared)
	cost := uint64(slots-g.recSlots) * slot * uint64(len(g.recPools()))
	if l.pool == nil {
		cost += uint64(slots) * slot
	}
	if cost > 0 && !g.roomSeat(cost, len(l.rec), need) {
		g.LastErr = "a session's recurrent state does not fit in this device's budget"
		g.SessionDeclines++
		return false
	}
	fail := func(err error) bool { g.LastErr = err.Error(); return false }
	if slots != g.recSlots || shared != g.recShared {
		if err := g.recResize(slots, shared); err != nil {
			return fail(err)
		}
	}
	if !seated {
		g.recSeats[sid] = st
		if err := g.recZero(nil, st.base, st.rows); err != nil {
			return fail(err)
		}
	}
	if l.pool == nil {
		np, err := g.newRecPool(g.recSlots, g.recShared, nil)
		if err != nil {
			return fail(err)
		}
		l.pool = np
		n := uint64(g.recSlots) * slot
		g.charge(n)
		l.recBytes += n
		g.RecBytes += n
		if g.recShared {
			g.RecShared++
		}
	} else if err := g.recZero(l, st.base, st.rows); err != nil {
		// The seat's slots in a block this session left and came back to
		// hold its old summary.
		return fail(err)
	}
	if g.recShared && !g.ensureRecNext(st.rows) {
		return false
	}
	if l.rec == nil {
		l.rec = map[uint64]*recPair{}
	}
	l.rec[sid] = &recPair{}
	return true
}

// ensureRecNext makes the shared form's next-state scratch hold rows
// sequences' states and compiles the copy that reads them back, charging the
// ledger for any growth. It runs outside a session (growSeat, LayersRows
// before submit), where Compile is allowed. Callers hold g.mu.
func (g *devTier) ensureRecNext(rows int) bool {
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	if g.recS == 0 {
		return true // a short convolution keeps no state beside its window
	}
	if bytes := uint64(rows * g.recS * 4); g.recNextBytes < bytes {
		b, err := g.dev.Alloc(int(bytes))
		if err != nil {
			return fail(err)
		}
		// Every recording, and every submission in flight, names the old one.
		g.dropGraph()
		if g.recNext != nil {
			g.recNext.Free()
			g.refund(g.recNextBytes)
		}
		g.recNext, g.recNextBytes = b, bytes
		g.charge(bytes)
	}
	return true
}

// freeRec releases a block's pool and every session's place in it, which is
// the same contract freeKV has and for the same reason: a release gives the
// block up entirely, so no sequence keeps a summary for it. The callers refund
// l.recBytes; a seat no block uses any more goes at recTidy.
func freeRec(l *layer) {
	if l.pool != nil {
		l.pool.free()
		l.pool = nil
	}
	clear(l.rec)
}

// addSessionKV gives session sid its own attention history for a block
// whose model-resident parts are already on the device. Callers hold g.mu.
func (g *devTier) addSessionKV(sid uint64, l *layer, p *nn.LayerPlan) bool {
	if p.NonCausal {
		if kvp := g.transientKV(p); kvp != nil {
			if l.kv == nil {
				l.kv = map[uint64]*kvPair{}
			}
			l.kv[sid] = kvp
			return true
		}
	}
	g.initKVCap(p)
	if g.pagedPlan(p) {
		// The pages are the sequence's, taken as its rows are written; the
		// block's layer is already in the pool.
		if l.kv == nil {
			l.kv = map[uint64]*kvPair{}
		}
		l.kv[sid] = &kvPair{paged: true}
		return true
	}
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	need := uint64(cache + vcache)
	if !g.roomSeat(need, len(l.kv), need) {
		g.LastErr = "a second session's KV cache does not fit in this device's budget"
		g.SessionDeclines++
		return false
	}
	g.charge(need)
	kvp := &kvPair{}
	var err error
	// One allocation under MLA, where vcache is zero; see kvPair.value.
	if kvp.kc, err = g.dev.Alloc(cache); err == nil && vcache > 0 {
		kvp.vc, err = g.dev.Alloc(vcache)
	}
	if err != nil {
		if kvp.kc != nil {
			kvp.kc.Free()
		}
		g.refund(need)
		g.LastErr = err.Error()
		return false
	}
	if l.kv == nil {
		l.kv = map[uint64]*kvPair{}
	}
	kvp.bytes = need
	l.kv[sid] = kvp
	// l.kvBytes is what ReleaseLayers refunds, so it moves with every charge;
	// without it a block released after a second session refunds only the first
	// session's share and the budget leaks.
	l.kvBytes += need
	g.KVBytes += need
	return true
}

// freeKV releases every session's history for a block. A release gives the block
// up entirely, so no sequence keeps a cache for it.
func (g *devTier) freeKV(l *layer) {
	for sid, p := range l.kv {
		if p.shared {
			g.transientKVPut()
			delete(l.kv, sid)
			continue
		}
		if p.kc != nil {
			p.kc.Free()
		}
		if p.vc != nil {
			p.vc.Free()
		}
		delete(l.kv, sid)
	}
}

// kvOf returns this session's cache for the block, or nil if it has none yet.
func (l *layer) kvOf(sid uint64) *kvPair {
	if l == nil || l.kv == nil {
		return nil
	}
	return l.kv[sid]
}

// blockScratch is the per-call staging every layer of a lane shares: a call
// holds its lane (devsess.go), so one block runs in it at a time.
type blockScratch struct {
	// headGraph is the head as ops from the model's front end (nn.Head.Graph):
	// what serving the head forms (served.go); headHidden is it writing its
	// normed rows too (nn.Head.HiddenGraph), a call reading them back's.
	headGraph, headHidden *graph.Graph
	// sampleGraph is the greedy token over the head's logits as ops from the
	// front end (nn.Head.Sample): what serving the argmax forms; sampleWalk
	// a top-k draw's token (nn.Head.SampleWalk), what serving SampleK forms;
	// pickGraph and pickHead the label pick over the logits and the head
	// projecting the picked rows alone (nn.Head.PickGraph, PickHead).
	sampleGraph, sampleWalk, pickGraph, pickHead *graph.Graph
	// headShared says head, mvHead, hNorm, hNormB, hBias and headCap are another
	// scratch's -- lane0's, borrowed by a clone (shareHead) -- and are not
	// freed or refunded with this one.
	headShared bool

	// walked is g.layerGen+1 at prepBatch's last successful layer walk for
	// this scratch: until a block is placed or released or the decode retune
	// rewrites their matvecs, the walk would find what it found then (0: never
	// walked).
	walked uint64
	// left marks, by block, the blocks whose prompt the composer serves at
	// this scratch's rows and whose own batched kernels the walk left
	// uncompiled: a decline's work (declineTwins).
	left []bool
	// prompt is the lane's instance of each form the composer serves a
	// prompt chunk's blocks with at this scratch's rows, shared by the blocks
	// of the form (served.go): freed and refunded with the scratch.
	prompt map[*partForm]*partInst
	// arena is the device buffers those instances' passing buffers are laid
	// on, one set for every form, since the blocks run one after another
	// (partArena).
	arena *partArena
	p     nn.LayerPlan
	// rows is how many token positions this scratch runs at once: 1 for decode
	// and the prefill chunk width for the batched arm. Every buffer is sized
	// rows-wide and every kernel is the Rows variant, which is byte-identical
	// at rows == 1 (TestDenseKernelsUnchanged).
	rows                int
	tok                 int
	qtile, ktile, atile int
	// bytes is what building this scratch allocated, by the device's count
	// (0 where it keeps none).
	bytes int64
	// pkv is the paged attention set; nil unless this scratch's history is
	// paged (devTier.pagedPlan), and then the staged, tiled and flash
	// kernels above that read a contiguous cache are not built.
	pkv *pagedScratch
	// ds4 is DeepSeek V4's set (ds4.go), nil on every other model.
	ds4 *ds4Scratch

	// The ragged-decode set (rows.go): attention whose rows are different
	// sequences, a per-row softmax, the head's batched twin and a per-row argmax.
	// ragBuilt says prepRowsHead built the set; the composer's sample
	// writes the tokens.
	ragBuilt          bool
	ragLogits, ragTok backend.Buf
	ragVocab          int // the rows of ragLogits a row holds
	sstride           int // the score row stride; padded at rows > 1
	// arows is how many query rows one pass of the attention kernels takes:
	// rows, except for a vision block whose score planes would be too large,
	// which runs its attention in query chunks (visionattn.go). qc and xc are a
	// chunk's queries and output, gathered from and scattered to q and xb at
	// qoffs[c], by qGather and xScatter.
	arows         int
	qoffs         []backend.Buf
	x, h          backend.Buf
	roff          backend.Buf // RoPE-q destination offsets, one per row
	hNorm, logits backend.Buf
	// embedGraph is the input rows as ops (nn.Head.Embed); embW and posW the
	// tables they read, the token embedding (the head where it is the
	// embedding) and a learned position table, resident for the model, and
	// embWhy why PrepHead could not hold them. embTok and embPos are the
	// rows' tokens and positions the composer's gather reads, embCast the
	// rows a promise no call takes are gathered into (materializeLocked).
	embedGraph     *graph.Graph
	embW, posW     *resident
	embWhy         partDecline
	embTok, embPos backend.Buf
	embCast        backend.Buf
	// embOuts is the buffers a step's input rows write (embedOuts).
	embOuts [partIO]backend.Buf
	// embSW and embSV are the tensors the input rows bind past the token
	// embedding and the position table (nn.Head.Streams), by role.
	embSW map[jlm.Role]*resident
	embSV map[jlm.Role]backend.Buf
	// collapse is the streams' collapse a call of the head alone runs
	// first (collapse.go), nil where the head reads one row.
	collapse *nn.RowsPart
	// foldReq is the request of the composer's row take that moves a batched
	// chunk's last residual row into this (decode) scratch's x, so the output
	// projection rides the chunk's own submission (layersOnce's folded head):
	// composer.RequestRows words and the row (servedRowTake).
	foldReq backend.Buf
	hNormB  backend.Buf // a LayerNorm head's bias (C6)
	// hBias is the output projection's bias (nn.Head.Bias), held where the
	// composer serves a head graph that binds it.
	hBias       backend.Buf
	head        *resident
	mvHead      mv
	hRaw        []byte
	hOut        []float32
	cs, n, zero backend.Buf
	// calls is a call's rows, one word: what a composed form of several
	// rows reads as the rows it runs (composer.CallRows); headCalls the rows a
	// call's head over rows projects.
	calls, headCalls backend.Buf
	// pics is a non-causal scratch's rows' descriptors (composer.RowWords
	// each) where the composer serves its blocks (towerStep), and picsW
	// the same rows each attending its window of the call; picStage and
	// picStageW are their bytes as a call stages them.
	pics, picsW         backend.Buf
	picStage, picStageW []byte
	// instNoRoom says a composed instance at the scratch's rows did not fit
	// the budget (promptInst): the reserved width narrows (fitReserved).
	instNoRoom bool
	// csSWA is the local layers' rotary table, allocated only for an
	// architecture that trains two bases (LayerPlan.SWAPeriod). nil elsewhere,
	// which is what keeps ropeTable a nil check for every other model.
	csSWA backend.Buf
	// ropeTab and ropeTabSWA are the per-model folded planes (nn.Rope.TabPlanes)
	// the composer's rotary tables read, uploaded once (initRopePlanes).
	ropeTab, ropeTabSWA backend.Buf
	// ropeTabAxes and rposAxes are what the composer's table over several
	// position axes reads (nn.LayerPlan.RopeTabAxes): its planes, and the
	// live row count then each row's four positions.
	ropeTabAxes, rposAxes backend.Buf
	// ropeRows is the call's rows as the composed rotary tables read them
	// (ropeRowsWords): two words a row past the count. rowPos is their
	// positions as a block reads them (rowPosWords).
	ropeRows, rowPos backend.Buf
	// Gemma 4's per-layer embedding (nn.LayerPlan.PLEDim): the rows' inputs,
	// staged per submission, a block's gate, its slice of the inputs, their
	// product and the projection back. nil elsewhere.
	pleIn backend.Buf
	// rowTok is the call's rows' tokens, which a block routing by token reads
	// (SetTokenIDs), a word a row.
	rowTok backend.Buf
	// streams is, by a step's rows, each residual stream of x as a view at
	// its offset (x is stream-major), the inputs and the outputs a block of
	// several streams binds (servedStreams).
	streams map[int][2][]backend.Buf
	// ragLin is a ragged step's linear block per row count (LayersRows): the
	// convolution and the fused rule over the step's runs, each a sequence's
	// rows at consecutive positions (kernels' run forms), built for the real
	// row count so a padded row never reaches a state.
	ragLin map[int]*ragLinear
	dRows  backend.Buf
	// cRows is the composer's recurrent descriptor in the shared form.
	cRows           backend.Buf
	rec             nn.RecurrentPlan
	hx, hcs, hcsSWA []float32
	hraw            []byte
	hout            []float32
}

// PrepLayer is prepLayerFor for the zero session: a caller that never attached.
func (g *devTier) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	return g.prepLayerFor(0, li, p, w)
}

// prepLayerFor uploads one block and compiles the kernels it needs, taking it only
// if it fits.
func (g *devTier) prepLayerFor(sid uint64, li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	return g.prepLayer(sid, li, p, w, false)
}

// residScaled reports a plan whose block outputs are scaled before their
// residual adds (Granite's residual_scale). Such a block takes neither the
// projection's fused residual (mvres: the bias slot has no scale) nor the
// add-and-norm launches: the output lands in bs.mvOut as for a post-norm, and
// bs.addS adds s times it. A mixture's routed weights carry no part of it on
// the device, unlike the host's routed scale (model.Config.routedScale): the
// scaled add covers the routed sum and the shared expert at once.
func residScaled(p *nn.LayerPlan) bool { return p.ResidualScale != 0 && p.ResidualScale != 1 }

// lnBlock reports whether the block's norms are LayerNorms: a vision tower's,
// or a classic text block's (C6, nn.LayerPlan.LayerNorm). Every other block
// is RMSNorm, whose fused norm+quantize kernels a LayerNorm cannot use.
func lnBlock(p *nn.LayerPlan) bool { return p.LayerNorm }

// ungatedFFN reports whether the block's FFN has no gate matrix: up, the
// activation alone, down -- a vision tower's MLP, or a classic text block's
// (C6: phi-2, starcoder, falcon, nemotron).
func ungatedFFN(p *nn.LayerPlan) bool { return p.UngatedFFN }

// declineWeights names a weight this tier cannot express, or "" when every one
// of them is expressible. declineReason is its twin for the plan: the plan says
// what graph the block runs, and this says whether its buffers can be uploaded
// as they are.
func declineWeights(w *nn.LayerWeights) string {
	for _, x := range []struct {
		name string
		w    nn.Weight
	}{{"q", w.Wq}, {"k", w.Wk}, {"v", w.Wv}, {"o", w.Wo},
		{"gate", w.Gate}, {"up", w.Up}, {"down", w.Down},
		// MLA's five. The upload reads QS/D/SC whole and has no row base, so an
		// offset weight would be indexed from row 0 and answer fluently; this
		// guard checks rather than assumes none is.
		{"q_a", w.Wqa}, {"q_b", w.Wqb}, {"kv_a", w.Wkva},
		{"k_b", w.Wkb}, {"v_b", w.Wvb}} {
		p := x.w.Packed
		if p == nil || len(x.w.Data) == 0 {
			continue
		}
		// Stride 0 means "the whole tensor", which is the ordinary case.
		if p.Row != 0 || (p.Stride != 0 && p.Stride != x.w.Rows) {
			return fmt.Sprintf("the %s projection is rows [%d,%d) of a packed tensor of %d "+
				"(a fused qkv, phi3-style); the upload has no row base",
				x.name, p.Row, p.Row+x.w.Rows, p.Stride)
		}
	}
	return ""
}

// prepLayer is PrepLayer with the pager allowed. mayPage lets a block that does
// not fit evict another one rather than be declined. It is a second pass, not a
// mode: GPU.PrepLayer offers every device with mayPage false first, so a second
// card with room still gets the block and only when no device has room does
// anybody swap.
func (g *devTier) prepLayer(sid uint64, li int, p *nn.LayerPlan, w *nn.LayerWeights, mayPage bool) (placed bool) {
	if p.Conv != nil {
		return g.prepConv(li, p, w, mayPage)
	}
	// Every placement of a block forms its composed part, a move's and a
	// re-offer's too: a step would otherwise meet it first. Deferred first,
	// it runs after the body's own deferred unlocks.
	defer func() {
		if placed {
			g.mu.Lock()
			g.prewarmComposedTables()
			// A block whose composed part, or prompt part at the narrowest
			// width fitReserved reached, does not fit beside the blocks placed
			// is declined here rather than met by its first step or prompt; a
			// mixture's is offered again streamed.
			why := g.prewarmComposedBlock(li, true)
			if why == "" {
				why = g.unserved(li)
			}
			if why == "" {
				why = g.promptRoomless(li)
			}
			if why != "" && !g.sharedBeyond(li, sid) {
				g.dropGraph()
				g.releaseBlock(li)
				g.afterRelease()
				g.Declined++
				g.LastErr = why
				placed = false
			}
			g.mu.Unlock()
		}
	}()
	// The host side of a hybrid block is the offering session's: a placed
	// block is shared by every session that attaches, and each runs its experts
	// on its own instance (runHost looks it up by the call's session). Recorded
	// before anything can decline, so a block another session placed still
	// learns this one's.
	if w != nil && w.HostExperts != nil {
		g.mu.Lock()
		if g.hostFns == nil {
			g.hostFns = map[uint64]map[int]hostSide{}
		}
		if g.hostFns[sid] == nil {
			g.hostFns[sid] = map[int]hostSide{}
		}
		g.hostFns[sid][li] = hostSide{one: w.HostExperts, rows: w.HostExpertsRows, uses: w.ExpertUses}
		g.mu.Unlock()
	}
	// Shapes this tier cannot express are declined first and by name (RULE 8a):
	// each would run wrong if skipped, so the block goes to the host whole. They
	// are checked before any allocation so a decline takes no VRAM from the next
	// block.
	// A row range of a packed tensor cannot be expressed: the device layout
	// interleaves rows, resident() uploads QS/D/SC whole, and MatVecShape has no
	// row base, so the block runs on the host instead of reading the wrong rows.
	if r := declineWeights(w); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	// A tower block the device could not take leaves nothing behind: the
	// first one built its set's scratch before the budget said no, and with
	// no tower block on the device that scratch is a ceiling nobody runs.
	if p.NonCausal {
		defer func() {
			if placed {
				return
			}
			g.dropIdleTower()
			// And on a device it was the first block of, the staging its
			// build sized goes too, as the last block out takes it.
			if len(g.layers) == 0 && (g.bs == nil || g.bs.head == nil) {
				g.dropGraph()
				g.dropAllScratch()
			}
		}()
	}
	// The block's own geometry's scratch set, and the home set again on the
	// way out (gemma4.go): Gemma 4's sliding layers, and a non-causal
	// segment's blocks -- a vision tower's, whose residual, heads and FFN are
	// not the text model's. sp is the plan that set is built from.
	sp := scratchPlanOf(p)
	if k := geoOf(p); k != 0 {
		g.geoUsed = true
		g.useGeom(k)
		defer g.useGeom(0)
	}

	// The KV ceiling is the largest context any session asked for, not the
	// first's, or a longer second session is refused at its next position.
	if !p.NonCausal && g.kvCap != 0 && p.MaxSeq > g.maxSeqAsked {
		g.maxSeqAsked = p.MaxSeq
	}
	if r := planConflict(g.bs, sp); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	if r := nonCausalConflict(g.bs, sp); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}

	// A block already on this device is shared and only the history is new: the
	// weights, norms, biases and router are the model's. Rebuilding would
	// re-upload them and replace the layer the first session is running against.
	if l := g.layers[li]; l != nil && l.ok {
		// The new session's plan still has to reach the scratch: a longer session
		// may need a sliding window the first did not (model.windowFor), so take
		// the union as the fresh-block path does.
		if !l.nonCausal && g.bs != nil {
			if want := g.bs.p; planUnion(&want, sp) {
				g.dropGraph()
				if !g.rebuildScratch(&want) {
					return false
				}
			}
		}
		// A linear block's history is a recurrent pair, not a KV cache; a
		// block that attends as well keeps both.
		if l.linear {
			if l.recOf(sid) == nil && !g.addSessionRec(sid, l, p) {
				return false
			}
			if !l.withAttn {
				return true
			}
		}
		if l.kvOf(sid) != nil || l.kvSrc != li {
			return true // this session already has a history here, or reads its source's
		}
		if !g.addSessionKV(sid, l, p) {
			return false
		}
		return true
	}
	// The q/k norm is launched when the block carries its weights; the plan's
	// flag builds the kernels. The model offers the weights by the same
	// derivation that sets the flag (jlm.QKNormOn), so a block where the two
	// disagree is a contract violation, refused rather than run without its
	// norm. A block another session placed is offered without them, held
	// above.
	if attn := !p.NonCausal && (p.Recurrent.Conv == 0 || p.Recurrent.WithAttn); attn && p.QKNorm != (w.QNorm != nil) {
		g.LastErr = fmt.Sprintf("block %d: the plan's q/k norm is %v and the block offers %d norm "+
			"weights", li, p.QKNorm, len(w.QNorm))
		return false
	}
	g.dropGraph()
	// The KV capacity is decided before the scratch is built: bs.p is the capped
	// plan, and the scratch's score buffers and MigrateKV's staging are sized
	// from its MaxSeq, so building first would size them for the whole context.
	g.initKVCap(p)
	// A non-causal set is built for the rows of the pictures it runs, not for
	// the tower's largest grid (visrows.go): the first block sets where it
	// starts, ReserveRows moves it.
	if p.NonCausal {
		g.visMax = max(g.visMax, p.MaxSeq)
		if g.visCap == 0 {
			g.visCap = visRowsFor(min(p.MaxSeq, visStart), p.MaxSeq)
		}
	}
	if g.bs == nil && p.NonCausal {
		// A non-causal call is one whole run, so its set's scratch is built
		// for the run's rows -- the rows the set is built for (visCap), which
		// capped hands the plan as its MaxSeq: there is no decode width.
		if !g.buildTowerScratch(sp) {
			return false
		}
	} else if g.bs == nil {
		g.shapeChanged()
		if g.bs = g.initScratch(sp, 1); g.bs == nil {
			return false
		}
	} else if want := g.bs.p; p.NonCausal && (sp.NFFN > want.NFFN || sp.Recurrent.Conv > 0 && want.Recurrent.Conv == 0 ||
		sp.QKNorm && !want.QKNorm) {
		// A non-causal block with a wider MLP than the set was built for (a
		// decision head after its encoder): the set grows to it, and every
		// block runs its own width inside it (layer.nffn). An LFM2 encoder's
		// convolution blocks and attention blocks share a set too: it carries
		// the convolution's kernels and the q/k norm's, whichever came first.
		want.NFFN, want.MaxSeq = max(want.NFFN, sp.NFFN), g.visMax
		if want.Recurrent.Conv == 0 {
			want.Recurrent = sp.Recurrent
		}
		want.QKNorm = want.QKNorm || sp.QKNorm
		if !g.towerAt(&want, g.visCap) {
			return false
		}
	} else if want := g.bs.p; !p.NonCausal && planUnion(&want, sp) {
		// g.bs is built from whichever block arrives first, and on a hybrid that
		// may be a linear block with no out-gate kernels. A hybrid's two kinds
		// alternate over one residual inside one submission, so they share one
		// scratch carrying both kernel sets, built from the union.
		if !g.rebuildScratch(&want) {
			return false
		}
	}
	if g.layers == nil {
		g.layers = map[int]*layer{}
	}
	l := &layer{nonCausal: p.NonCausal, windowed: p.Windowed, geo: geoOf(p), vFromK: p.VFromK, graph: p.Graph}
	l.win, l.chunked = attendWindow(p.Graph)
	if p.NonCausal {
		l.nffn, l.ungated, l.act = p.NFFN, ungatedFFN(p), p.Act
		l.noRope = p.NRot == 0 || p.NoPosEnc
	}
	mkw := func(slot int, x nn.Weight) *resident {
		q, ok := quantOf(x.T)
		if !ok || x.Rows <= 0 || len(x.Data) == 0 {
			return nil
		}
		return g.resident(li, slot, q, x.Data, x.Rows, x.K, x.Packed)
	}
	// mkk is a matrix's record: its rows, format and width, where its bias
	// goes (served.go binds both by role).
	mkk := func(x nn.Weight, r *resident, bias bool) (mv, bool) {
		q, ok := quantOf(x.T)
		if !ok {
			return mv{}, false
		}
		return mv{rows: x.Rows, q: q, k: x.K, pad: p.KPad > 0 && x.K == p.KPad}, true
	}
	mkkID := func(x nn.Weight, r *resident, experts, slots int, slotAct, bias bool) (mv, bool) {
		m, ok := mkk(x, r, bias)
		m.experts, m.slots, m.slotAct = experts, slots, slotAct
		return m, ok
	}
	// An expert bank is one matrix of NExpert*Rows rows: a GGUF 3-D expert tensor
	// is expert-major and contiguous, so it packs, uploads and indexes as one
	// tall matrix, and the kernel does the row offset from Experts and Slots.
	moe := w.Router.Data != nil
	bank := 1
	slots := 0
	if moe {
		if p.NExpert < 1 || p.NExpertUsed < 1 || p.NExpertUsed > p.NExpert {
			return false
		}
		bank, slots = p.NExpert, p.NExpertUsed
	}
	// One list: layer.tensors() derives the free walk and pageSizeWhy the price
	// from this order, so a matrix uploaded outside it would be neither freed
	// nor charged. A linear block leaves 1..3 empty (Wq carries its fused q|k|v),
	// an attention block leaves 10..12 empty, and pageSizeWhy skips zero-length
	// weights.
	ws := weightList(w)
	// A streamed bank is priced at its compact size by scaling the three bank
	// rows from NExpert sheets to NExpertUsed; every size derives from
	// Weight.Rows, so the charge and the refund cannot disagree. slots > 1 is a
	// kernel constraint: at experts <= 1 mkkID returns the plain matvec, which
	// has no pSel parameter.
	mode, auto := g.autoStream[li]
	// hostExp says the block's experts will run on the host (hybrid): the
	// placement's choice, else the tier's defaults. Such a block keeps no bank
	// on the card at all -- no compact bank, no expert kernels, no packed
	// layout needed -- and a top-1 router is no obstacle, since no indexed
	// matvec runs.
	// The composed block reads its experts in pages on the card: only a
	// placement's own "host" sends them to the host.
	hostExp := w.HostExperts != nil && auto && mode == expHost
	stream := moe && (g.StreamExperts || auto) && slots < bank
	// A hybrid block that can hold an expert cache (Config.NoHybridCache)
	// is placed as a streamed one -- the compact bank and its kernels on the
	// card -- marked hybrid, so the cache sized after placement has kernels
	// to run its experts with (streamBank.split). One whose experts carry
	// biases or a dense scale keeps no bank, as cacheSlotsFor refuses them,
	// and so does a float bank, which has no packed sheets to send.
	cacheable := !g.NoHybridCache && g.StreamCacheSlots >= 0 && slots > 1 && !p.DenseMoE &&
		w.ExpGateB == nil && w.ExpUpB == nil && w.ExpDownB == nil &&
		w.Up.Packed != nil && w.Down.Packed != nil && (w.Gate.Packed != nil || len(w.Gate.Data) == 0)
	hybrid := stream && hostExp && !cacheable
	// cslots is the streamed bank's size in sheets, decided at the first bank
	// matrix (cacheSlotsFor): slots without an expert cache, more with one.
	cslots := -1
	// price is ws with the three banks removed for a streamed block: the
	// compact bank is one buffer for the whole device (sharedCompact), so a
	// block costs its base and nothing else.
	price := ws
	if stream {
		for i := 4; i <= 6; i++ {
			if ws[i].Rows%bank != 0 {
				g.LastErr = fmt.Sprintf("block %d: %s has %d rows, not a multiple of %d experts",
					li, weightRole(i), ws[i].Rows, bank)
				return false
			}
			ws[i].Rows = ws[i].Rows / bank * slots
		}
		price = append([]nn.Weight(nil), ws...)
		for i := 4; i <= 6; i++ {
			price[i].Data, price[i].Packed = nil, nil
		}
	}
	// Decide on the whole block before uploading any of it: declining halfway
	// leaves earlier tensors resident, counted and unreachable. Every decline
	// sets LastErr.
	full, why, ok := g.pageSizeWhy(price)
	if !ok {
		g.LastErr = fmt.Sprintf("block %d: %s", li, why)
		return false
	}
	// The slot width is the widest block, not this one; see slots().
	if full > g.widest {
		g.widest = full
	}
	need, _ := g.blockBytes(price)
	// Include what admitting it charges for as long as it is resident: its KV
	// cache (or recurrent pair) and norms. Leaving them out admitted blocks that
	// pushed the card over budget and made trim() page every token.
	perm := g.permBytes(p, w)
	need += perm
	// Before the block's weights: the prompt's and the step's scratch are
	// budgeted first, with room left for this block, so a card that cannot
	// hold both takes a block fewer -- or a narrower chunk -- rather than
	// refusing the prompt later (scratch.go).
	g.reserveScratch(sp, need)
	// The KV pools' free pages go first: they hold nothing.
	g.kvCompact(need, nil)
	if !g.room(need) {
		// A block that does not fit pages another one out rather than falling to
		// the host. Below minSlots there is nowhere to swap through, so it
		// declines and LastErr says which case it is.
		// The slots are counted after this block's own permanent charge: a
		// streamed block whose norms and history take the last slot leaves
		// nothing to swap through, and the next page-in fails the device.
		if n := g.slotsIfPerm(perm); !mayPage || !g.stream[li] || n < minSlots {
			// Only the final pass counts as a decline: GPU.PrepLayer asks every
			// device with mayPage false first.
			if mayPage {
				g.Declined++
			}
			// Say which of the three it is: another device may still have room,
			// the block is placed resident, or the budget is too small to swap
			// through.
			why := "another device is asked first"
			if mayPage {
				why = "it is placed resident and does not fit, so it runs on the host " +
					"(a placement that streams it -- model.Place.Stream, -placement N=DEV~ -- " +
					"swaps it through the device instead)"
				if g.stream[li] {
					why = fmt.Sprintf("the budget holds %d block slot(s), below the %d a swap needs", n, minSlots)
				}
			}
			g.LastErr = fmt.Sprintf("block %d needs %d bytes, %d of %d used (pool %s: %d of %d); %s",
				li, need, g.used, g.limit, g.pool.Name(), g.pool.Used(), g.pool.Limit(), why)
			// Recorded on the final pass only, like Declined.
			if mayPage && g.DeclineWhy == "" {
				g.DeclineWhy = why
			}
			return false
		}
		// At load the scan ascends from here, so the victim is the block just
		// behind this one and a resident prefix remains: the set the run-time
		// policy settles on, so the first token starts warm. See victim().
		if !g.reclaim(need, li, 0, li+1) {
			g.Declined++
			g.LastErr = fmt.Sprintf("block %d needs %d bytes and nothing can be paged out (%d of %d used)",
				li, need, g.used, g.limit)
			if g.DeclineWhy == "" {
				g.DeclineWhy = "nothing can be paged out"
			}
			return false
		}
	}
	// The staging is sized after the block is admitted, from what is left
	// (need is what the block still has to spend). If that is not enough it is
	// not allocated and the covered tensors are packed on the host. It is
	// allocated once and survives into later uploads and page-ins.
	//
	// The bytes are fetched here, after the decision and before the first read:
	// nn.LayerWeights.Ensure makes the contents valid (a paged mixture sets it),
	// and calling it earlier would read a bank for every block the budget then
	// refuses.
	if w.Ensure != nil {
		if err := w.Ensure(w); err != nil {
			g.LastErr = fmt.Sprintf("block %d: %v", li, err)
			return false
		}
	}
	rs := l.tensors()
	ms := []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd,
		&l.mvsg, &l.mvsu, &l.mvsd, &l.mvSG, &l.mvBA, &l.mvSO,
		// MLA's five, in weightList/tensors() order.
		&l.mvQA, &l.mvQB, &l.mvKVA, &l.mvKB, &l.mvVB,
		// KDA's four, in weightList/tensors() order.
		&l.mvFA, &l.mvFB, &l.mvGA, &l.mvGB,
		// Gemma 4's per-layer embedding gate and projection.
		&l.mvpg, &l.mvpp,
		// DeepSeek V3.2's indexer, Gemma 3n's LAuReL, then Falcon-H1's mixer
		// projection, last.
		&l.mvIQ, &l.mvIK, &l.mvIW,
		&l.mvLL, &l.mvLR, &l.mvSI,
		// MiniMax-M3's indexer query (its key is slot 25's).
		&l.mvMQ,
		// DeepSeek V4's grouped output and compressors.
		&l.mvWoA, &l.mvCKV, &l.mvCG, &l.mvICKV, &l.mvICG,
		// Kimi-K3's MLA output gate and latent mixture projections.
		&l.mvMG, &l.mvRD, &l.mvRU,
		// Qwen3.8-Flash-Next's hyper-connections and n-gram projections.
		&l.mvHAD, &l.mvHAU, &l.mvHAI, &l.mvHFD, &l.mvHFU, &l.mvHFI, &l.mvNGK, &l.mvNGV}
	// Same order as rs/ms. Attention carries biases on every architecture that
	// has them; a vision block also on up and down. For every text model the
	// FFN entries stay nil, so their kernels are byte-identical to before
	// (TestDenseKernelsUnchanged).
	bias := [][]float32{w.Bq, w.Bk, w.Bv, w.Bo, w.BGate, w.BUp, w.BDown,
		nil, nil, nil, nil, nil, nil,
		// MLA's five carry no bias on any architecture that has them, and
		// neither do KDA's four, the per-layer embedding's two, the indexer's
		// three, LAuReL's two, the mixer's projection or MiniMax-M3's query.
		nil, nil, nil, nil, nil,
		nil, nil, nil, nil, nil, nil,
		nil, nil, nil, nil, nil, nil,
		nil,
		nil, nil, nil, nil, nil,
		nil, nil, nil,
		nil, nil, nil, nil, nil, nil, nil, nil}
	// The latent block's own slots: 0 (q) is the one-step query and is empty
	// where 13/14 are populated; 1 and 2 are always empty (MLA has no k or v
	// projection); 15..17 are always there. Each is an absent weight, not a
	// failed upload.
	l.mla = p.MLA()
	mlaSkip := func(i int) bool {
		if !l.mla {
			// No latent block, no latent matrices -- except Mamba-1's four,
			// which ride KDA's slots 18..21 (nn.SSMWeights).
			return i >= 13 && !(i >= 18 && p.Recurrent.Mamba1)
		}
		// A linear block of an MLA model (Kimi-Linear) has none of them either:
		// l.mla is a property of the model, not the block.
		if p.Recurrent.Conv > 0 && i >= 13 && i < 18 {
			return true
		}
		switch i {
		case 0:
			// A linear block's q is its fused q|k|v whatever the model's
			// query rank (Kimi-K3 compresses its MLA query; Kimi-Linear
			// does not).
			return p.QLoraRank != 0 && p.Recurrent.Conv == 0
		case 1, 2:
			return true
		case 13, 14:
			return p.QLoraRank == 0
		}
		return false
	}
	for i, x := range ws {
		// A DeepSeek V4 block names its own slots (ds4Slot): the generic
		// skips below are for every other architecture.
		if p.DS4 != nil {
			want := ds4Slot(p, i)
			if want && len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: a DeepSeek V4 block and %s is empty", li, weightRole(i))
				return false
			}
			if !want {
				continue
			}
		} else if i >= 31 && i <= 35 {
			continue // DeepSeek V4's slots (31..35), absent everywhere else
		}
		// Qwen3.8-Flash-Next's (39..46), on the blocks the plan says carry
		// them and nowhere else.
		if i >= 39 {
			if !q4eSlot(p, i) {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: a Qwen3.8-Flash-Next block and %s is empty", li, weightRole(i))
				return false
			}
		}
		// Kimi-K3's MLA output gate (36) and latent projections (37, 38), on
		// the blocks the plan says carry them.
		if i >= 36 && i <= 38 {
			want := i == 36 && p.MLA() && p.Recurrent.Conv == 0 && len(x.Data) != 0 ||
				i >= 37 && p.K3 != nil && p.K3.Latent != 0
			if !want {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has a latent mixture and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		// The per-layer embedding's two (22, 23), the indexer's three (24..26),
		// Gemma 3n's LAuReL (27, 28) and Falcon-H1's mixer projection (29),
		// past every other range.
		extra := i >= 22
		if i >= 24 && i <= 28 && p.DS4 == nil {
			// MiniMax-M3's indexer has a key (25) and its own query (30),
			// not DeepSeek V3.2's query latent (24) or weights (26).
			if (i < 27 && (p.IdxHeads == 0 || (p.MSA() || p.QSA()) && i != 25)) || (i >= 27 && p.AltUp == 0) {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has an indexer or LAuReL and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		ple := i == 22 || i == 23
		if ple {
			if p.PLEDim == 0 {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has per-layer embeddings and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		// The mixer's own projection beside the attention's four (Falcon-H1, 29),
		// absent by construction everywhere else.
		if i == 29 {
			if !p.Recurrent.WithAttn {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan runs attention beside a Mamba-2 mixer "+
					"and %s is empty", li, weightRole(i))
				return false
			}
		}
		// MiniMax-M3's indexer query (30), absent by construction elsewhere.
		if i == 30 {
			if !p.MSA() && !p.QSA() {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan selects key blocks and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		// KDA's four low-rank gate matrices, checked before mlaSkip because they
		// sit past MLA's range and are not MLA's.
		if i >= 18 && !extra && p.DS4 == nil {
			switch {
			case !p.Recurrent.ChanDecay && !p.Recurrent.Mamba1, p.Recurrent.Conv == 0:
				// Not a KDA model, or an attention block of one: absent by
				// construction rather than by a missing tensor.
				continue
			case (i == 20 || i == 21) && p.Recurrent.ChanDecay && len(w.SSM.Gate.Data) != 0:
				// Kimi-K3's full-rank output gate is slot 10's, and there is no
				// low-rank pair.
				continue
			case len(x.Data) == 0:
				g.LastErr = fmt.Sprintf("block %d: the plan is a Kimi Delta Attention "+
					"block and %s is empty", li, weightRole(i))
				return false
			}
			// A present weight falls through to the upload; a continue here would
			// leave l.ssmGA nil and crash inside the recorded session.
		}
		if p.DS4 == nil && !extra && mlaSkip(i) {
			continue
		}
		if i >= 13 && !extra {
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan is a latent attention block "+
					"(KVLoraRank=%d) and %s is empty", li, p.KVLoraRank, weightRole(i))
				return false
			}
		}
		// An ungated block has no gate matrix: slot 4 is empty by construction.
		if ungatedFFN(p) && i == 4 {
			continue
		}
		// A KV-sharing block has no k or v matrix (slots 1, 2): it attends to
		// its source's history.
		if p.KVShared && (i == 1 || i == 2) {
			continue
		}
		// Nor has a block whose v is k's projection (Gemma 4's global layers)
		// a v matrix: slot 2. The emit reads k's projection for v.
		if p.VFromK && i == 2 {
			if len(x.Data) != 0 {
				g.LastErr = fmt.Sprintf("block %d: v is k's projection and the block offers a v matrix", li)
				return false
			}
			continue
		}
		// A block that is its mixer alone has no FFN at all.
		if p.NoFFN && i >= 4 && i <= 6 {
			continue
		}
		// A linear block has no k, v or o matrix: its one projection is slot 0
		// (Wq carries the fused q|k|v|gate), and offerRange clears the other three.
		if p.Recurrent.Conv > 0 && !p.Recurrent.WithAttn && i >= 1 && i <= 3 {
			continue
		}
		if i >= 10 && i < 13 {
			// A linear block's three; an attention block leaves them empty.
			if p.Recurrent.Conv == 0 {
				continue
			}
			// Kimi-Linear's KDA has no ssm_gate: its output gate is the
			// low-rank ssm_g_a/ssm_g_b pair. Kimi-K3's is one full-rank matrix,
			// here.
			if i == 10 && p.Recurrent.ChanDecay && len(w.SSM.GA.Data) != 0 {
				continue
			}
			// Mamba-1 has no ssm_ba: dt comes off its x_proj (slot 18).
			if i == 11 && p.Recurrent.Mamba1 {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan is a linear block and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		if i >= 7 && i < 10 {
			if p.NFFNShExp == 0 {
				continue // no shared expert in this architecture at all
			}
			// An ungated shared expert (Nemotron 3) has no gate matrix.
			if i == 7 && ungatedFFN(p) {
				continue
			}
			// A plan with a shared expert and an empty weight is a decline: a
			// skip would run the FFN on an unwritten buffer.
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has a shared expert "+
					"(NFFNShExp=%d) and %s is empty", li, p.NFFNShExp,
					[...]string{"ShGate", "ShUp", "ShDown"}[i-7])
				return false
			}
		}
		e, sl := 1, 0
		// MLA's two absorb banks are one sheet per head, and the "expert" indexed
		// is the head: every head is selected on every token, so the selection is
		// the identity 0..NHead-1 and one indexed matvec replaces NHead, each slot
		// reading its own activation like the mixture's down projection.
		if l.mla && (i == 16 || i == 17) {
			e, sl = p.NHead, p.NHead
		}
		// DeepSeek V4's grouped output is one sheet per group, every group
		// selected: the identity, as MLA's heads are.
		if p.DS4 != nil && i == 31 {
			e, sl = p.DS4.OGroups, p.DS4.OGroups
		}
		// 4..6 only: the shared expert's matrices at 7..9 are dense.
		if moe && i >= 4 && i <= 6 { // gate, up and down are the banks
			e, sl = bank, slots
			if stream {
				// The kernel is told the bank is k experts and indexes it with
				// 0..k-1; backend.TestIndexedMatVecMatchesACompactBank is where
				// "the same answer, bit for bit" comes from. With an expert
				// cache it is told the cache's size and indexed by slot.
				if cslots < 0 {
					// An auto-streamed block is placed on the plain bank;
					// its cache comes after placement (sizeAutoCaches).
					cslots = g.cacheSlotsFor(p, ws, g.StreamCacheSlots, slots, bank,
						false)
				}
				e = cslots
			}
		}
		// Weight.Rows already counts the whole bank (model.get multiplies every
		// trailing dimension), which PackWeights and the upload want; the kernel is
		// built for one expert's rows and does the offset, so it is divided below.
		var r *resident
		if hybrid && i >= 4 && i <= 6 {
			// Nothing of the bank goes on the card; the host runs it.
			if l.stream == nil {
				l.stream = &streamBank{sel: make([]uint32, slots),
					ensure: w.Ensure, sel2: w.EnsureExperts,
					pre: w.PrefetchExperts, spec: w.SpeculateExperts, host: w.HostExperts, w: *w, hybrid: true, nExpert: bank, li: li}
				g.StreamBlocks++
			}
			continue
		}
		if stream && i >= 4 && i <= 6 {
			q, qok := quantOf(x.T)
			if !qok {
				g.LastErr = fmt.Sprintf("block %d: %s is %s, which has no device kernel",
					li, weightRole(i), x.T)
				return false
			}
			if x.Packed == nil && (!layout.IsFloat(q) || len(x.Data) == 0) {
				// A streamed bank sends a sheet at a time: a packed bank's planes,
				// or a float bank's own rows (its one plane, expert-major).
				g.LastErr = fmt.Sprintf("block %d: %s is not pre-packed, so its bank cannot be streamed",
					li, weightRole(i))
				return false
			}
			var sh [3]int
			var ok bool
			if cslots > slots {
				// The block's own bank of cslots sheets, kept across tokens:
				// it is the expert cache, so it cannot be the device's one
				// shared compact bank. Its charge is the block's.
				r, sh, ok = g.residentCompact(q, x.Rows/slots, x.K, cslots)
			} else {
				r, sh, ok = g.sharedCompact(i, q, x.Rows/slots, x.K, slots)
			}
			if !ok {
				return false
			}
			if l.stream == nil {
				// w is seeded here so a host side that does not page (Ensure nil)
				// still has valid spans; otherwise fill() sends nothing and the FFN
				// contributes zeros.
				l.stream = &streamBank{sel: make([]uint32, slots),
					ensure: w.Ensure, sel2: w.EnsureExperts,
					pre: w.PrefetchExperts, spec: w.SpeculateExperts, host: w.HostExperts, w: *w, li: li}
				l.stream.nExpert = bank
				// An auto-streamed block runs its experts on the host unless
				// told otherwise: on Kimi-K3 over the V100s that decoded 1.43x
				// the host's rate where sending the sheets lost to it
				// (placement.md 16c-2).
				switch {
				case l.stream.host == nil || auto && mode == expCard:
				case auto && mode == expHost:
					l.stream.hybrid = true
				}
				l.stream.initCache(cslots, bank)
				g.StreamBlocks++
			}
			l.stream.sh[i-4] = sh
		} else {
			r = mkw(i, x)
		}
		if r == nil || !r.ok {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("block %d tensor %d (%s %dx%d) not resident",
					li, i, x.T, x.Rows, x.K)
			}
			return false
		}
		*rs[i] = r
		if stream && i >= 4 && i <= 6 {
			// A streamed bank's rows were scaled to the selection (slots
			// sheets) above, whatever size its cache is.
			x.Rows /= slots
		} else if e > 1 {
			// e, not bank: they differ for a streamed bank, whose kernel is built for
			// k experts. This is rows per expert either way.
			if x.Rows%e != 0 {
				return false
			}
			x.Rows /= e
		}
		// slotAct: each slot reads its own activation vector. The mixture's
		// down projection (6) does, and so do MLA's two banks -- head h absorbs
		// its own nope half and un-absorbs its own attention result.
		k, ok := mkkID(x, r, e, sl, i == 6 || i == 16 || i == 17 || i == 31, bias[i] != nil)
		if !ok {
			return false
		}
		*ms[i] = k
	}
	if moe {
		// The router is unquantized, so it does not go through resident() at
		// all -- it is uploaded as the row-major floats it arrives as.
		if w.Router.Rows != p.NExpert || w.Router.K != p.NEmbd {
			return false
		}
		// F32 or F16, with the bytes counted against the type (Mixtral's router
		// is F16; reading it as F32 runs past the buffer). Any other type is
		// refused by name. Data may run past the matrix, so it is sliced to it.
		es := 0
		switch w.Router.T {
		case quant.F32:
			es = 4
		case quant.F16:
			es = 2
		}
		n := p.NExpert * p.NEmbd
		if es == 0 || len(w.Router.Data) < es*n {
			g.LastErr = fmt.Sprintf("block %d: the router is %v in %d bytes, and the "+
				"device reads an F32 or F16 router of %dx%d", li, w.Router.T,
				len(w.Router.Data), p.NExpert, p.NEmbd)
			return false
		}
		// Padded to a whole word: the f16 kernel loads the word an element
		// sits in, and an odd element count's last word runs past the matrix.
		rb := w.Router.Data[:es*n]
		if len(rb)%4 != 0 {
			rb = append(append([]byte(nil), rb...), make([]byte, 4-len(rb)%4)...)
		}
		b, e := g.dev.Alloc(len(rb))
		if e != nil {
			return false
		}
		if e = b.Write(rb); e != nil {
			return false
		}
		l.router, l.routerF16 = b, es == 2
		l.routerW = resident{qs: b, nrows: p.NExpert, k: p.NEmbd, t: layout.Float32, ok: true}
		if es == 2 {
			l.routerW.t = layout.Float16
		}
	}
	var err error
	// Norms and biases arrive as f32, which is what the device reads.
	up := func(dst *backend.Buf, v []float32) {
		// A KV-sharing block's k norm is absent: it caches nothing.
		if err != nil || len(v) == 0 {
			return
		}
		f := v
		var b backend.Buf
		if b, err = g.dev.Alloc(len(f) * 4); err != nil {
			return
		}
		err = b.Write(f32b(f))
		*dst = b
	}
	if p.NFFNShExp > 0 && w.ShRouter != nil {
		// The shared gate is a vector producing one logit, uploaded as plain
		// floats like a norm. A nil gate is an ungated shared expert (DeepSeek's,
		// weight 1), matching the host's `if l.shRouter != nil`; a present gate of
		// the wrong width is an error.
		if len(w.ShRouter) != p.NEmbd {
			g.LastErr = fmt.Sprintf("block %d: the shared expert's gate is %d wide, want %d",
				li, len(w.ShRouter), p.NEmbd)
			return false
		}
		up(&l.shRouter, w.ShRouter)
	}
	// A post-norm block (OLMo 2, EXAONE 4) has neither pre-norm: the emit
	// copies the residual where the norm would have written (bs.copyRows).
	noPre := w.AttnNorm == nil && w.PostAttnNorm != nil && w.PostFFNNorm != nil
	// Qwen3.8-Flash-Next's sublayers read their hyper-connection's mix, whose
	// grouped norm stands where the pre-norms would (q4e.go).
	noPre = noPre || p.Q4E != nil && w.AttnNorm == nil && w.FFNNorm == nil
	// ModernBERT's first block has no attention norm and keeps its FFN norm:
	// the attention reads the residual as it is, the same copy.
	noAttnNorm := w.AttnNorm == nil && w.FFNNorm != nil && p.NoPreNorm && p.NonCausal
	if noPre && !p.NoPreNorm && p.Q4E == nil {
		g.LastErr = fmt.Sprintf("block %d has no pre-norm on a plan that does not say so", li)
		return false
	}
	if w.AttnNorm != nil {
		up(&l.nAttn, w.AttnNorm)
	} else if !noPre && !noAttnNorm {
		g.LastErr = fmt.Sprintf("block %d has no attention norm", li)
		return false
	}
	// A parallel block sharing one norm has no FFN norm (phi-2) and the FFN reads
	// the attention's normed row; every other block needs one.
	switch {
	case w.FFNNorm != nil:
		up(&l.nFFN, w.FFNNorm)
	case noPre, p.NoFFN:
	case !p.Parallel:
		g.LastErr = fmt.Sprintf("block %d has no FFN norm and is not a parallel block", li)
		return false
	}
	// Lengths are checked because a wrong one runs: a short sink array is read
	// past its end, and an expert's bias slice would land on its neighbour's rows.
	for _, v := range []struct {
		name     string
		on       bool
		have     []float32
		want     int
		dst      *backend.Buf
		optional bool
	}{
		{"attention sinks", p.AttnSinks, w.Sinks, p.NHead, &l.sinks, false},
		{"router bias", p.MoEBias, w.RouterB, p.NExpert, &l.routerB, true},
		{"expert gate bias", p.MoEBias, w.ExpGateB, p.NExpert * p.NFFNExp, &l.expGateB, true},
		{"expert up bias", p.MoEBias, w.ExpUpB, p.NExpert * p.NFFNExp, &l.expUpB, true},
		{"expert down bias", p.MoEBias, w.ExpDownB, p.NExpert * p.NEmbd, &l.expDownB, true},
		// The block, not just the model, must be a mixture: DeepSeek's leading
		// dense blocks carry no exp_probs_b.
		{"expert selection bias", p.ExpertSelBias && p.NExpert > 0 && !p.HashExperts, w.ExpSelB, p.NExpert, &l.expSelB, false},
	} {
		if !v.on || (v.optional && v.have == nil) {
			continue
		}
		if len(v.have) != v.want {
			g.LastErr = fmt.Sprintf("block %d: %s is %d wide, want %d", li, v.name, len(v.have), v.want)
			return false
		}
		up(v.dst, v.have)
	}
	if w.PostAttnNorm != nil {
		up(&l.nPostAttn, w.PostAttnNorm)
	}
	// BitNet's inner norms: both, at their widths, on every such block.
	if p.SubNorm {
		if len(w.AttnSubNorm) != p.NHead*p.HeadDim || len(w.FFNSubNorm) != p.NFFN {
			g.LastErr = fmt.Sprintf("block %d: inner norms %d and %d wide, want %d and %d", li,
				len(w.AttnSubNorm), len(w.FFNSubNorm), p.NHead*p.HeadDim, p.NFFN)
			return false
		}
		up(&l.nSubAttn, w.AttnSubNorm)
		up(&l.nSubFFN, w.FFNSubNorm)
	} else if w.AttnSubNorm != nil || w.FFNSubNorm != nil {
		g.LastErr = fmt.Sprintf("block %d carries inner norms the plan does not name", li)
		return false
	}
	if w.PostFFNNorm != nil {
		up(&l.nPostFFN, w.PostFFNNorm)
	}
	if p.Clamps {
		if len(w.Clamp) != 28 {
			g.LastErr = fmt.Sprintf("block %d: clipped-linear bounds are %d floats, want 28", li, len(w.Clamp))
			return false
		}
		up(&l.clampB, w.Clamp)
	}
	// Gemma 4's mixture block: four norms and the expert factors, charged
	// with the norms.
	if p.DenseMoE {
		for _, v := range []struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{
			{"experts' pre-norm", w.FFNNorm2, p.NEmbd, &l.nFFN2},
			{"dense MLP's post-norm", w.PostFFNNorm1, p.NEmbd, &l.nPost1},
			{"experts' post-norm", w.PostFFNNorm2, p.NEmbd, &l.nPost2},
			{"router input's norm", w.RouterNorm, p.NEmbd, &l.nRouter},
			{"per-expert scale", w.ExpScale, p.NExpert, &l.expScale},
		} {
			if len(v.have) != v.want {
				g.LastErr = fmt.Sprintf("block %d: %s is %d wide, want %d", li, v.name, len(v.have), v.want)
				return false
			}
			up(v.dst, v.have)
			l.auxBytes += uint64(4 * v.want)
		}
	}
	// Gemma 4's per-layer embedding norm and the block's slice offset.
	if p.PLEDim != 0 {
		if len(w.PLEPost) != p.NEmbd {
			g.LastErr = fmt.Sprintf("block %d: the per-layer embedding norm is %d wide, want %d",
				li, len(w.PLEPost), p.NEmbd)
			return false
		}
		up(&l.nPLE, w.PLEPost)
		if err == nil {
			if l.pleOff, err = g.dev.Alloc(4); err == nil {
				err = l.pleOff.Write(g.u32b(uint32(li * p.PLEDim)))
			}
		}
		l.auxBytes += uint64(4*p.NEmbd + 4)
	}
	if why := g.prepAltUp(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	if why := g.prepDS4(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	if why := g.prepK3(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	if why := g.prepQ4E(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	// Gemma 4's output scalar, one float, charged with the norms.
	if p.OutScale != 0 {
		up(&l.outScale, []float32{p.OutScale})
		l.auxBytes += 4
	}
	// Apertus's xIELU numbers, four floats, the block's own.
	if p.Act == graph.ActXIELU {
		if len(w.XIELU) != 4 {
			g.LastErr = fmt.Sprintf("block %d: xIELU wants 4 numbers, the block carries %d", li, len(w.XIELU))
			return false
		}
		up(&l.xielu, w.XIELU)
		l.auxBytes += 16
	}
	// The shadow arm's state, the worst drift and the steps compared, both
	// zero: written here, since a device buffer is not.
	if g.shadowing() && p.Graph != nil {
		up(&l.shadow, []float32{0, 0})
		l.auxBytes += 8
	}
	if w.AttnNormB != nil {
		up(&l.nAttnB, w.AttnNormB)
	}
	// BERT's norms after the residual adds: the attention reads the block
	// input (no attention norm), FFNNorm normalises x + attention and the
	// residual takes it, ResidNorm normalises x + mlp.
	if p.PostResidNorm {
		if w.AttnNorm != nil || w.FFNNorm == nil || len(w.ResidNorm) != p.NEmbd ||
			(w.ResidNormB != nil && len(w.ResidNormB) != p.NEmbd) {
			g.LastErr = fmt.Sprintf("block %d: a post-norm block wants no attention norm and two "+
				"norms of %d after its adds", li, p.NEmbd)
			return false
		}
		up(&l.nResid, w.ResidNorm)
		if w.ResidNormB != nil {
			up(&l.nResidB, w.ResidNormB)
		}
	}
	if w.FFNNormB != nil {
		up(&l.nFFNB, w.FFNNormB)
	}
	// The attention biases, uploaded as float32 like every other aux vector.
	// The kernel was already generated with pBias for exactly these slots.
	for _, bp := range []struct {
		v []float32
		m *mv
	}{{w.Bq, &l.mvq}, {w.Bk, &l.mvk}, {w.Bv, &l.mvv}, {w.Bo, &l.mvo},
		// A vision block's MLP is biased too; skipping these runs and is wrong
		// in exactly the way the attention biases are. A gated one's gate as
		// well (Qwen2.5-VL); nil on every text block.
		{w.BGate, &l.mvg}, {w.BUp, &l.mvu}, {w.BDown, &l.mvd}} {
		if bp.v == nil {
			continue
		}
		up(&bp.m.bias, bp.v)
		if bp.m.alt != nil {
			bp.m.alt.bias = bp.m.bias
		}
		l.biasBytes += uint64(len(bp.v) * 4)
	}
	if w.QNorm != nil {
		up(&l.nQ, w.QNorm)
		up(&l.nK, w.KNorm)
	}
	// MiniMax-M3's indexer norms, each one head wide.
	l.qsa = p.QSA()
	if l.msa = p.MSA(); l.msa || l.qsa {
		if len(w.IdxQNorm) != p.IdxHeadDim || len(w.IdxKNorm) != p.IdxHeadDim {
			g.LastErr = fmt.Sprintf("block %d: the indexer norms are %d/%d wide, want %d",
				li, len(w.IdxQNorm), len(w.IdxKNorm), p.IdxHeadDim)
			return false
		}
		up(&l.nIdxQ, w.IdxQNorm)
		up(&l.nIdxK, w.IdxKNorm)
		l.auxBytes += uint64(2 * p.IdxHeadDim * 4)
	}
	// A linear block of an MLA model has neither latent norm (see mlaSkip).
	if l.mla && p.Recurrent.Conv == 0 {
		// Widths are checked because a wrong one runs: KVANorm covers the latent
		// only, and a whole-row weight would norm the rotary key too.
		if len(w.KVANorm) != p.KVLoraRank {
			g.LastErr = fmt.Sprintf("block %d: the kv latent norm is %d wide, want %d",
				li, len(w.KVANorm), p.KVLoraRank)
			return false
		}
		up(&l.nKVA, w.KVANorm)
		l.auxBytes += uint64(len(w.KVANorm) * 4)
		if p.QLoraRank != 0 {
			if len(w.QANorm) != p.QLoraRank {
				g.LastErr = fmt.Sprintf("block %d: the query latent norm is %d wide, want %d",
					li, len(w.QANorm), p.QLoraRank)
				return false
			}
			up(&l.nQA, w.QANorm)
			l.auxBytes += uint64(len(w.QANorm) * 4)
		}
		if p.IdxHeads != 0 {
			if len(w.IdxKNorm) != p.IdxHeadDim || len(w.IdxKNormB) != p.IdxHeadDim {
				g.LastErr = fmt.Sprintf("block %d: the indexer key norm is %d/%d wide, want %d",
					li, len(w.IdxKNorm), len(w.IdxKNormB), p.IdxHeadDim)
				return false
			}
			up(&l.nIdxK, w.IdxKNorm)
			up(&l.nIdxKB, w.IdxKNormB)
			l.auxBytes += uint64(2 * p.IdxHeadDim * 4)
		}
	}
	if r := p.Recurrent; r.Conv > 0 {
		// The convolution weight stays plane-major, W[u][c] with the host's stride,
		// so migrating the state is a memcpy rather than a shuffle.
		if len(w.SSM.Conv1d) != r.Conv*r.Chans {
			g.LastErr = fmt.Sprintf("block %d: conv1d is %d floats, want %d x %d",
				li, len(w.SSM.Conv1d), r.Conv, r.Chans)
			return false
		}
		up(&l.ssmConv1d, w.SSM.Conv1d)
		l.auxBytes += uint64(len(w.SSM.Conv1d) * 4)
		if !r.Centred {
			taps := make([]float32, len(w.SSM.Conv1d))
			for u := range r.Conv {
				for c := range r.Chans {
					taps[c*r.Conv+u] = w.SSM.Conv1d[u*r.Chans+c]
				}
			}
			up(&l.ssmTaps.qs, taps)
			l.ssmTaps = resident{qs: l.ssmTaps.qs, nrows: r.Chans, k: r.Conv, t: layout.Float32, ok: true}
			l.auxBytes += uint64(len(taps) * 4)
		}
	}
	// A short convolution (LFM2) has its window and nothing else.
	if r := p.Recurrent; r.Conv > 0 && !r.ShortConv {
		up(&l.ssmA, w.SSM.A)
		up(&l.ssmDt, w.SSM.DtBias)
		if !r.NoNorm {
			up(&l.ssmNorm, w.SSM.Norm)
		}
		if r.Mamba1 {
			// Mamba-1's D and convolution bias, one a channel, and its three
			// norms where the block has them; lengths checked because a short
			// one is read past its end.
			if len(w.SSM.D) != r.VHeads || len(w.SSM.ConvBias) != r.Chans || len(w.SSM.A) != r.VHeads*r.KDim ||
				len(w.SSM.DtBias) != r.VHeads {
				g.LastErr = fmt.Sprintf("block %d: Mamba-1's D is %d, its conv bias %d, A %d and dt bias %d wide, "+
					"want %d, %d, %d and %d", li, len(w.SSM.D), len(w.SSM.ConvBias), len(w.SSM.A),
					len(w.SSM.DtBias), r.VHeads, r.Chans, r.VHeads*r.KDim, r.VHeads)
				return false
			}
			up(&l.ssmD, w.SSM.D)
			up(&l.ssmCB, w.SSM.ConvBias)
			l.auxBytes += uint64((len(w.SSM.D) + len(w.SSM.ConvBias)) * 4)
			if w.SSM.DtNorm != nil {
				if len(w.SSM.DtNorm) != r.Rank || len(w.SSM.BNorm) != r.KDim || len(w.SSM.CNorm) != r.KDim {
					g.LastErr = fmt.Sprintf("block %d: Mamba-1's dt/B/C norms are %d, %d and %d wide, want %d, %d and %d",
						li, len(w.SSM.DtNorm), len(w.SSM.BNorm), len(w.SSM.CNorm), r.Rank, r.KDim, r.KDim)
					return false
				}
				up(&l.m1DtN, w.SSM.DtNorm)
				up(&l.m1BN, w.SSM.BNorm)
				up(&l.m1CN, w.SSM.CNorm)
				l.auxBytes += uint64((r.Rank + 2*r.KDim) * 4)
			}
		}
		if r.SSD {
			// Mamba-2's two vectors more; lengths checked because a short one
			// is read past its end.
			if len(w.SSM.D) != r.VHeads || len(w.SSM.ConvBias) != r.Chans ||
				(!r.NoNorm && len(w.SSM.Norm) != r.VHeads*r.VDim) {
				g.LastErr = fmt.Sprintf("block %d: Mamba-2's D is %d, its conv bias %d and its norm %d wide, "+
					"want %d, %d and %d", li, len(w.SSM.D), len(w.SSM.ConvBias), len(w.SSM.Norm),
					r.VHeads, r.Chans, r.VHeads*r.VDim)
				return false
			}
			up(&l.ssmD, w.SSM.D)
			up(&l.ssmCB, w.SSM.ConvBias)
			l.auxBytes += uint64((len(w.SSM.D) + len(w.SSM.ConvBias)) * 4)
		}
		// A unit weight: the per-key-head L2 norm is an RMSNorm with a weight of
		// ones (as engine/model/delta.go builds it), so no second reduction kernel.
		ones := make([]float32, r.KDim)
		for i := range ones {
			ones[i] = 1
		}
		up(&l.ssmOnes, ones)
		l.auxBytes += uint64((len(w.SSM.A) + len(w.SSM.DtBias) + len(w.SSM.Norm) + len(ones)) * 4)
	}
	// One extra KV position is a trash slot for the batched prefill: a ragged
	// last chunk runs the full grid width with surplus rows zeroed, and those rows
	// still write K and V. A chunk-width of slack costs too much VRAM and a real
	// row's slot would be clobbered, so they all write here and nothing reads it.
	//
	// A linear block takes no KV cache at all: it keeps a running summary, and
	// the plan's NKVHead/HeadDim are the model-wide attention shape.
	linear := p.Recurrent.Conv > 0
	l.linear = linear
	l.withAttn = p.Recurrent.WithAttn
	l.noFFN = p.NoFFN
	l.centred = p.Recurrent.Centred
	if linear && !l.centred {
		if !g.addSessionRec(sid, l, p) {
			return false
		}
	}
	// A block that runs attention keeps a history, beside a recurrent state
	// where it runs a mixer too.
	attends := !linear || l.withAttn
	var cache, vcache int
	var kvp *kvPair
	l.kvSrc = li
	if p.KVShared {
		// No history of its own: it reads its source's, which must be on
		// this device already (blocks are offered in order).
		src := g.layers[p.KVSource]
		if src == nil || !src.ok || p.KVSource >= li {
			g.LastErr = fmt.Sprintf("block %d attends to block %d's history, which is not on this device",
				li, p.KVSource)
			return false
		}
		l.kvSrc = p.KVSource
	}
	if p.KVShared {
		// Nothing to allocate.
	} else if attends && g.pagedPlan(p) {
		err = g.pagedLayer(sid, l, li, p)
	} else if attends {
		g.initKVCap(p)
		cp := g.capped(p)
		// The V cache is half the size when packed (KVF16), which can decide
		// whether a model fits on the card at all. Under MLA it is zero: the value
		// is the key row's own prefix (kvPair.value).
		cache, vcache = g.kvCacheBytes(p, cp.MaxSeq)
		// One cache per session. PrepLayer runs for a session (sid), so
		// this allocates for whichever sequence is placing blocks; a second
		// Instance placing the same block reuses the shared weights above and gets
		// its own history. A non-causal block takes the device's one shared
		// pair instead (transientkv.go), and charges nothing of its own.
		kvp = &kvPair{}
		if p.NonCausal {
			if sh := g.transientKV(p); sh != nil {
				kvp, cache, vcache = sh, 0, 0
			}
		}
		for i, dst := range []*backend.Buf{&kvp.kc, &kvp.vc} {
			if kvp.shared {
				break
			}
			n := cache
			if i == 1 {
				n = vcache
			}
			if err == nil && n > 0 {
				*dst, err = g.dev.Alloc(n)
			}
		}
		if l.kv == nil {
			l.kv = map[uint64]*kvPair{}
		}
		if old := l.kv[sid]; old != nil && old.shared {
			g.transientKVPut()
		} else if old != nil {
			// Re-preparing a block this session already holds: its previous
			// history goes, or it leaks (see dropLayerLocal).
			if old.kc != nil {
				old.kc.Free()
			}
			if old.vc != nil {
				old.vc.Free()
			}
		}
		l.kv[sid] = kvp
	}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	// WithPoisonKV fills the cache with NaN so an attention kernel reading a
	// position nothing wrote fails a gate instead of reading a lucky zero.
	if g.kb.poisonKV && kvp != nil && !kvp.shared {
		nan := make([]byte, cache)
		for i := 0; i+3 < len(nan); i += 4 {
			nan[i], nan[i+1], nan[i+2], nan[i+3] = 0x01, 0x00, 0xC0, 0x7F
		}
		kvp.kc.Write(nan)
		if kvp.vc != nil {
			kvp.vc.Write(nan[:vcache])
		}
	}
	// cache + vcache, not 2*cache: with KVF16 the V half is half the size.
	l.kvBytes = uint64(cache + vcache)
	if kvp != nil && !kvp.shared {
		kvp.bytes = l.kvBytes
	}
	// += because a linear block already charged its five vectors above, and
	// auxBytes is what ReleaseLayers refunds.
	l.auxBytes += uint64((2*len(w.AttnNorm)+len(w.PostAttnNorm)+len(w.PostFFNNorm)+len(w.Clamp))*4) + l.biasBytes
	g.charge(l.auxBytes + l.kvBytes)
	g.KVBytes += l.kvBytes
	l.ok = true
	// The page record is built for every block, including on a model that fits,
	// so a later demotion is expressible; the budget decides whether anything
	// moves. Its size is what the residents actually charged, so reclaim() asks
	// for exactly what a page-in will spend.
	var pb, pr uint64
	for _, rp := range l.tensors() {
		r := *rp
		pb += r.bytes()
		if n := r.bytes(); n > 0 {
			if f, err := g.planesFootprint(r.t, r.nrows, r.k); err == nil && f > n {
				pr += f - n
			}
		}
	}
	if stream {
		// A streamed block is pinned (pg == nil, which every pager path reads as
		// "resident, never a victim"): its base is small and its bank is not
		// resident, so there is nothing to relocate. Paging it would also be
		// wrong, since pageIn would refresh the bank at NExpert sheets against an
		// admitted shape of k.
		l.pg = nil
	} else {
		l.pg = &page{ws: append([]nn.Weight(nil), ws...), bytes: pb, round: pr, in: true,
			// Kept so a page-in can ask again: ws points into a reused host frame.
			ensure: w.Ensure}
		g.pagesIn++
		// Inside the else because ReleaseLayers refunds pageBytes only when
		// l.pg.in was true.
		g.pageBytes += pb + pr
	}
	// Re-preparing a block frees the layer it replaces. Only the per-layer
	// buffers go: the weights live in g.res keyed by their bytes and may be
	// shared with the layer being installed.
	if old := g.layers[li]; old != nil && old != l {
		g.dropLayerLocal(old)
	}
	g.layers[li] = l
	g.layerGen++
	g.placeGen.Add(1)
	// The reserved widths' scratches and the composer's prompt forms of the
	// block at those widths are the placement's (promptForms), so no prompt
	// or step builds them on a card the blocks have filled.
	if !p.NonCausal {
		was := g.prewarming
		g.prewarming = true
		for _, w := range []int{g.promptW, g.stepW} {
			if w > 0 {
				g.prepBatch(w)
			}
		}
		g.prewarming = was
	}
	// Admitting a block charges its KV cache forever, so the slot count just
	// fell; give back whatever no longer fits under it.
	g.trim()
	return true
}

// poisonBuf fills a freshly allocated device buffer with NaN when the knob is
// on. It is the one place the two poison sites agree on the pattern.
func (g *devTier) poisonBuf(b backend.Buf, n int) {
	if g.kb.poisonScratch && b != nil && n > 0 {
		b.Write(nanFill(n))
	}
}

// nanFill is n bytes of float32 NaN, for poisoning a scratch allocation.
func nanFill(n int) []byte {
	b := make([]byte, n)
	for i := 0; i+4 <= n; i += 4 {
		// 0x7FC00001, a quiet NaN, in little-endian.
		b[i], b[i+1], b[i+2], b[i+3] = 0x01, 0x00, 0xC0, 0x7F
	}
	return b
}

// initRopePlanes uploads this scratch's folded rotary planes, once, for the
// composer's rotary tables (tableBinds), and zeroes the tables: a table
// writes only its live rows, so the padded tail must hold zeros already.
// Nothing else writes bs.cs, so this deliberately un-poisons it under
// WithPoisonScratch.
func (g *devTier) initRopePlanes(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	if p.NonCausal || p.NRot <= 0 || len(p.RopeTab) == 0 {
		return nil
	}
	up := func(dst *backend.Buf, v []float32) error {
		b, err := g.dev.Alloc(len(v) * 4)
		if err != nil {
			return err
		}
		*dst = b
		return b.Write(f32b(v))
	}
	if err := up(&bs.ropeTab, p.RopeTab); err != nil {
		return err
	}
	if p.SWAPeriod > 0 && len(p.RopeTabSWA) > 0 {
		if err := up(&bs.ropeTabSWA, p.RopeTabSWA); err != nil {
			return err
		}
	}
	z := make([]byte, rows*p.NRot*4)
	for _, t := range []backend.Buf{bs.cs, bs.csSWA} {
		if t == nil {
			continue
		}
		if err := t.Write(z); err != nil {
			return err
		}
	}
	return nil
}

// initRopeAxes holds what the composer's table over several position axes
// reads where the plan has one (RopeTabAxes): its planes, uploaded once, and
// the rows' positions, staged every call.
func (g *devTier) initRopeAxes(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	if len(p.RopeTabAxes) == 0 || p.NonCausal || p.NRot <= 0 || bs.cs == nil {
		return nil
	}
	b, err := g.dev.Alloc(len(p.RopeTabAxes) * 4)
	if err != nil {
		return err
	}
	if err := b.Write(f32b(p.RopeTabAxes)); err != nil {
		b.Free()
		return err
	}
	a, err := g.dev.Alloc((1 + 4*rows) * 4)
	if err != nil {
		b.Free()
		return err
	}
	bs.ropeTabAxes, bs.rposAxes = b, a
	return nil
}

// initRopeRows holds the rows the composed rotary tables read (ropeRowsWords),
// staged every call, and makes the device's marks arena (markArena).
func (g *devTier) initRopeRows(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	if p.NonCausal || p.NRot <= 0 || bs.cs == nil {
		return nil
	}
	b, err := g.dev.Alloc((1 + 2*rows) * 4)
	if err != nil {
		return err
	}
	bs.ropeRows = b
	return g.growMarks(1)
}

// initRowPos holds the rows' positions a block reads (rowPosWords), staged
// every call.
func (g *devTier) initRowPos(bs *blockScratch, rows int) error {
	b, err := g.dev.Alloc((1 + rows) * 4)
	if err != nil {
		return err
	}
	bs.rowPos = b
	return nil
}

// initScratch allocates what a call over the plan's blocks binds and no block
// owns: the residual and its twin, the rotary tables and the rows they are
// computed from, the rows' counts, positions and tokens, the descriptors a
// picture's rows or a paged history are read through, and the host's
// staging. Every block's intermediate values are its composed part's own.
func (g *devTier) initScratch(p *nn.LayerPlan, rows int) (out *blockScratch) {
	// Everything the build allocates is the scratch's, charged (scratch.go).
	defer g.scratchWin().close()
	if rows < 1 {
		rows = 1
	}
	// The capped plan: every bound below reads bs.p, so the scratch is built
	// for what the device holds, not the full context.
	cp := g.capped(p)
	p = &cp
	if why := unblockedRow(p, rows); why != "" {
		g.LastErr = why
		return nil
	}
	// A picture's rows go in query chunks where its blocks' planes do not fit
	// one (visionChunk), so the scratch is rounded up to whole chunks.
	if p.NonCausal && rows > 1 {
		_, rows = visionChunk(rows, p.NHead, scoreStride(p, rows), g.tiles.qtile, g.tiles.atile, g.planeBudget())
	}
	bs := &blockScratch{p: cp, rows: rows, qtile: 1, ktile: 1, atile: 1}
	// A build that fails part-way frees what it allocated: on a full card --
	// the case that refuses -- what it leaked was what the next build needed.
	defer func() {
		if out == nil {
			freeScratch(bs)
		}
	}()
	// What the build allocated, by the device's own count: a tower's set is
	// resized from it (visrows.go).
	defer func(a0 int64) {
		if out != nil {
			out.bytes = g.allocated() - a0
		}
	}(g.allocated())
	if rows > 1 {
		bs.qtile, bs.ktile, bs.atile = g.tiles.qtile, g.tiles.ktile, g.tiles.atile
		for rows%bs.qtile != 0 {
			bs.qtile /= 2
		}
		for rows%bs.atile != 0 {
			bs.atile /= 2
		}
	}
	var err error
	al := func(dst *backend.Buf, n int) {
		// cuMemAlloc(0) is an error, not an empty buffer; an absent buffer is
		// right, since nothing reads it.
		if n == 0 || err != nil {
			return
		}
		// The device refusing scratch is the card being full, the same as a
		// refused weight upload: counted as NoRoom.
		if *dst, err = g.dev.Alloc(n); err != nil {
			g.NoRoom++
		}
		// RULE 13: zero is the identity for a sum, so a form reading scratch it
		// did not write is right on the first session and wrong on the next.
		// WithPoisonScratch fills with NaN so the read shows in the first token.
		if err == nil && g.kb.poisonScratch && *dst != nil {
			(*dst).Write(nanFill(n))
		}
	}
	// Every stream, stream-major (altup.go); one stream elsewhere.
	al(&bs.x, rows*p.ResidW()*4)
	al(&bs.h, rows*p.NEmbd*4)
	// The score row a picture's chunks are sized by (visionChunk).
	bs.sstride = scoreStride(p, rows)
	bs.arows = rows
	if p.NonCausal {
		bs.arows, _ = visionChunk(rows, p.NHead, bs.sstride, bs.qtile, bs.atile, g.planeBudget())
	}
	al(&bs.cs, rows*p.RopeW()*4)
	if p.SWAPeriod > 0 {
		al(&bs.csSWA, rows*p.RopeW()*4)
	}
	if err == nil {
		err = g.initRopePlanes(bs, p, rows)
	}
	if err == nil {
		err = g.initRopeAxes(bs, p, rows)
	}
	if err == nil {
		err = g.initRopeRows(bs, p, rows)
	}
	if err == nil {
		err = g.initRowPos(bs, rows)
	}
	// A linear block's run descriptors (composer.RunDescWords): its rows and
	// each sequence's slots in the block's pool, and the composer's, where
	// the pool is one shared half its forms step in place.
	if r := cp.Recurrent; r.Conv > 0 {
		bs.rec = r
		al(&bs.dRows, 4*composer.RunDescWords(rows))
		al(&bs.cRows, 4*composer.RunDescWords(rows))
	}
	al(&bs.n, (rows+1)*4)
	al(&bs.calls, 4)
	al(&bs.headCalls, 4)
	if p.NonCausal && p.Graph != nil {
		al(&bs.pics, rows*composer.RowWords*4)
		al(&bs.picsW, rows*composer.RowWords*4)
		bs.picStage = make([]byte, rows*composer.RowWords*4)
		bs.picStageW = make([]byte, rows*composer.RowWords*4)
	}
	al(&bs.roff, rows*4)
	al(&bs.zero, 4)
	// The n-gram block's gathered rows, its side input (SetNGramRows), or
	// the per-layer inputs.
	if q := p.Q4E; p.PLEDim == 0 && q != nil && q.NGramWidth != 0 {
		al(&bs.pleIn, rows*q.NGramWidth*4)
	}
	if p.PLEDim != 0 {
		al(&bs.pleIn, rows*p.PLEWidth*4)
	}
	al(&bs.rowTok, rows*4)
	if err != nil {
		g.LastErr = err.Error()
		return nil
	}
	if err = bs.zero.Write([]byte{0, 0, 0, 0}); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	// Each row's offset into a picture's query rows, which the tower's
	// forms read (towerserve.go).
	qdim := p.NHead * p.HeadDim
	roff := make([]byte, 0, rows*4)
	for r := range rows {
		roff = append(roff, u32one(uint32(r*qdim))...)
	}
	if err = bs.roff.Write(roff); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	if g.pagedPlan(p) {
		if err := g.initPagedScratch(bs); err != nil {
			g.LastErr = "paged attention: " + err.Error()
			return nil
		}
	}
	bs.hx = make([]float32, rows*p.ResidW())
	bs.hcs = make([]float32, rows*p.RopeW())
	if p.SWAPeriod > 0 {
		bs.hcsSWA = make([]float32, rows*p.RopeW())
	}
	bs.hraw = make([]byte, rows*p.ResidW()*4)
	bs.hout = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hraw[0])), rows*p.ResidW())
	return bs
}

// pickLanes settles, once per device, whether the kernels whose arithmetic
// needs a 32-lane subgroup (the butterfly softmax, the per-head norm) may be
// used. It asks the device for a promise: CUDA's warp and Apple's simdgroup are
// 32 by architecture; a Vulkan device promises 32 when its min and max subgroup
// sizes are both 32 or it lets the width be pinned at pipeline creation. The
// reported default subgroupSize is not a promise (see backend.GuaranteedLanes).
//
// A refusal selects the thread-per-head twins, which run every model correctly,
// so this is a selection rather than a capability gate. WithLanes(1) forces the
// scalar twins for a paired measurement; WithLanes(32) forces the kernel but not
// the promise, so a refusing backend still declines the pipeline and the block
// runs on the host. Reintroducing the bug is vulkan subgroup "force", which
// scripts/three-way.sh --violate subgroup-width sets.
func (g *devTier) pickLanes() int {
	if g.Lanes != 0 {
		return g.Lanes
	}
	ok, why := backend.GuaranteedLanes(g.dev, ir.SubgroupLanes)
	g.Lanes, g.LanesWhy = 1, why
	if ok {
		g.Lanes, g.LanesWhy = ir.SubgroupLanes, ""
	}
	switch g.kb.lanes {
	case 1:
		g.Lanes, g.LanesWhy = 1, "WithLanes(1)"
	case ir.SubgroupLanes:
		g.Lanes, g.LanesWhy = ir.SubgroupLanes, "WithLanes(32)"
		if why != "" {
			g.LanesWhy += ", over the device's own answer: " + why
		}
	}
	return g.Lanes
}

// The softmax is picked by correctness, not speed. A load-time timing probe
// runs at pos=1, where the choice does not matter; the warp reduction's gain
// grows with context. `verify -ab softmax` at a long -n is how it is checked.

// headOut is the buffer the head's answer is read from: the capped logits
// where the model has a final softcap, the projection's own otherwise.
// servedStreams is a block of n residual streams' inputs and outputs at a
// step of rows: stream k of x, a view at k*rows rows of the stream-major
// residual, each stream's input and output the same view, the per-layer
// inputs after the inputs. Made once a step width; a warm step makes
// nothing.
func (bs *blockScratch) servedStreams(rows, n int) (in, out []backend.Buf, err error) {
	if v, ok := bs.streams[rows]; ok && len(v[1]) == n {
		return v[0], v[1], nil
	}
	w := rows * bs.p.NEmbd * 4
	out = make([]backend.Buf, n)
	for k := range out {
		if out[k], err = backend.View(bs.x, k*w, w); err != nil {
			return nil, nil, fmt.Errorf("tier: stream %d of a residual of %d rows: %w", k, rows, err)
		}
	}
	in = append(append([]backend.Buf(nil), out...), bs.pleIn)
	if bs.streams == nil {
		bs.streams = map[int][2][]backend.Buf{}
	}
	bs.streams[rows] = [2][]backend.Buf{in, out}
	return in, out, nil
}

// headOut is the head's logits, every stage its graph states applied.
func (bs *blockScratch) headOut() backend.Buf { return bs.logits }

// ragOut is headOut for the per-row head.
func (bs *blockScratch) ragOut() backend.Buf { return bs.ragLogits }

// HeadResident reports whether the vocabulary projection is on the device.
//
// It is not derivable from the block count, which is the reason it exists:
// placeHead gives blocks up to make room for the projection, so the run with
// the better placement is the one showing fewer blocks.
func (g *devTier) HeadResident() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.bs != nil && g.bs.head != nil && g.bs.head.ok
}

// headScratch gives a device that holds no block the scratch the output
// projection runs in, built from the plan of the device that holds the last
// block (see GPU.PrepHead). A device that already has one keeps it.
func (g *devTier) headScratch(p *nn.LayerPlan) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.bs != nil {
		return true
	}
	g.initKVCap(p)
	g.bs = g.initScratch(p, 1)
	g.shapeChanged()
	return g.bs != nil
}

// dropHeadScratch frees what PrepHead allocated around the projection -- the
// norm, its bias, the logits and the softcap -- and refunds it, leaving the
// projection's resident to whoever holds it. Callers hold g.mu.
func (g *devTier) dropHeadScratch(bs *blockScratch) {
	g.shapeChanged()
	// So does a PrepHead after a seam gave its blocks back and grew again: the
	// norm, its bias and the logits are allocated there, and overwriting them
	// leaked all three (and their charge) on every round trip. bs.head is set
	// only where those charges were made.
	if bs.head != nil {
		n := uint64(bs.p.NEmbd*4 + bs.mvHead.rows*4)
		if bs.hNormB != nil {
			n += uint64(bs.p.NEmbd * 4)
		}
		if bs.hBias != nil {
			n += uint64(bs.mvHead.rows * 4)
		}
		g.refund(n)
	}
	for _, b := range []*backend.Buf{&bs.hNorm, &bs.hNormB, &bs.hBias, &bs.logits} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	bs.head = nil
	g.placeGen.Add(1)
	g.freeAltNorms()
}

// headGraphs records the head as ops from the model's front end, what the
// composer serves: the model's, so every lane's and every geometry set's. A
// later instance of the model, its graphs restated, offers them. Callers hold
// g.mu.
func (g *devTier) headGraphs(h *nn.Head) {
	g.eachLane(func(l *lane) {
		ds := []*blockScratch{l.bs}
		for _, set := range l.geoSets {
			ds = append(ds, set.bs)
		}
		for _, d := range ds {
			if d != nil {
				d.headGraph, d.headHidden, d.sampleGraph, d.embedGraph = h.Graph, h.HiddenGraph, h.Sample, h.Embed
				d.sampleWalk, d.pickGraph, d.pickHead = h.SampleWalk, h.PickGraph, h.PickHead
			}
		}
	})
}

// PrepHead uploads the output norm and the vocabulary projection.
func (g *devTier) PrepHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	defer g.placeGen.Add(1)
	g.dropGraph()
	bs := g.bs
	if bs == nil || h == nil {
		return false
	}
	g.headGraphs(h)
	h.GatherRow = h.Embed != nil
	g.prepCollapse(bs, h)
	// The head is the model's: a second instance on this tier finds it placed.
	// Building it again allocated a second norm and logits buffer over the
	// first ones, leaking both and charging the budget twice.
	if bs.head != nil && bs.head.ok && bs.mvHead.rows == h.W.Rows {
		return g.adoptHead(h)
	}
	g.dropHeadScratch(bs)
	q, ok := quantOf(h.W.T)
	if !ok || h.W.K != bs.p.NEmbd || h.W.Rows <= 0 || len(h.W.Data) == 0 {
		return false
	}
	// The reserved width's per-row logits are the head's (prepRowsReserved),
	// admitted with it: built after it without asking, they put the card past
	// its budget by a vocabulary a row, and a ragged step on a full card could
	// not build them.
	rowsNeed := func() uint64 {
		if bb := g.reservedFor(ComposedBlock); bb != nil && g.rowsHeld() {
			return g.rowsHeadNeed(bb, h.W.Rows, h.Softcap)
		}
		return 0
	}
	rows := rowsNeed()
	foot, ferr := g.planesFootprint(q, h.W.Rows, h.W.K)
	if ferr != nil {
		foot = uint64(len(h.W.Data))
	}
	// They shrink with the reserved width. A caller that did not list the CPU
	// has no other device for the head, so the width gives way first; with
	// the CPU listed the head goes home and the reservation holds.
	for h.NoCPU && rows > 0 && !g.room(foot+rows) && g.narrowReserved("the output projection's per-row logits") {
		rows = rowsNeed()
	}
	// Under paging the projection and a block slot compete for the same bytes,
	// and the head is permanent while a slot turns over every block. The head
	// runs once per token and a block NLayer times, so the projection is taken
	// only when it costs no slot.
	// A caller that did not list the CPU has no other device for it: the
	// projection is taken while a swap still has its slots, the streamed
	// blocks paging out to make the room.
	if want := foot + rows; len(g.stream) > 0 && h.NoCPU {
		after := g.slotsIfPerm(want)
		for after < minSlots && g.promptW > 0 && g.narrowReserved("the output projection and a block slot") {
			after = g.slotsIfPerm(want)
		}
		if after < minSlots {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and taking them "+
				"would leave %d block slot(s), below the %d a swap needs", want, after, minSlots)
			return false
		}
		if !g.reclaim(want, -1, 0, 0) {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and no streamed block "+
				"can be paged out for it", want)
			return false
		}
	} else if now, after := g.slots(), g.slotsIfPerm(want); len(g.stream) > 0 && after < now ||
		!h.NoCPU && g.holdsPages() && after < minSlots {
		// Paging is always on: a projection that leaves the blocks paged here
		// no slot to turn over in strands them.
		g.LastErr = fmt.Sprintf("the output projection needs %d bytes and "+
			"taking them would leave %d block slot(s) instead of %d for %d block(s) the card holds, so it "+
			"stays on the host: a slot turns over every block and the "+
			"projection runs once a token", want, after, now, g.pagesIn)
		return false
	}
	if rows > 0 && !g.room(foot+rows) {
		g.Declined++
		g.LastErr = fmt.Sprintf("the output projection needs %d bytes and its per-row logits %d more, %d of %d used",
			foot, rows, g.used, g.limit)
		return false
	}
	// -1: the output projection is not a block, so it is not released with one.
	r := g.resident(-1, 0, q, h.W.Data, h.W.Rows, h.W.K, h.W.Packed)
	if r == nil || !r.ok {
		return false
	}
	var err error
	// A head with no norm (Qwen3.8-Flash-Next's reads its mix as it is)
	// leaves hNorm nil, and the emit copies the row where the norm would
	// have written.
	if len(h.Norm) > 0 {
		f := make([]float32, len(h.Norm))
		for i, v := range h.Norm {
			f[i] = float32(v)
		}
		if bs.hNorm, err = g.dev.Alloc(len(f) * 4); err != nil {
			return false
		}
		if bs.hNorm.Write(f32b(f)) != nil {
			return false
		}
	}
	// A LayerNorm head needs its bias; an RMSNorm one has none.
	if lnBlock(&bs.p) {
		if len(h.NormB) != len(h.Norm) {
			g.LastErr = fmt.Sprintf("a LayerNorm head with a %d-wide bias for a %d-wide norm",
				len(h.NormB), len(h.Norm))
			return false
		}
		if bs.hNormB, err = g.dev.Alloc(len(h.NormB) * 4); err != nil {
			return false
		}
		if bs.hNormB.Write(f32b(h.NormB)) != nil {
			return false
		}
		g.charge(uint64(len(h.NormB) * 4))
	}
	if bs.logits, err = g.dev.Alloc(h.W.Rows * 4); err != nil {
		return false
	}
	bs.hRaw = make([]byte, h.W.Rows*4)
	bs.hOut = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hRaw[0])), h.W.Rows)
	bs.head, bs.mvHead = r, mv{rows: h.W.Rows, q: q, k: h.W.K}
	g.shapeChanged()
	g.headSrc, g.headNormHost = &h.W.Data[0], slices.Clone(h.Norm)
	g.charge(uint64(len(h.Norm)*4 + h.W.Rows*4))
	g.prepEmbed(bs, h)
	// The input rows' tables are the head's: a head whose tables have no room
	// is given back, so placement buys the room as it does for the head
	// (placeHead), rather than met by its first step.
	if bs.embWhy == partNoRoom {
		g.giveBackHead(bs)
		g.Declined++
		g.LastErr = "the input rows' tables have no room beside the head and the blocks placed: " + g.LastErr
		return false
	}
	// The output bias, for a head graph the composer serves that binds it. A
	// head without it (no room, a bias of another width) is declined by name
	// (partHeadBias) and the caller adds the bias.
	if spec, ok := headShape(h.Graph); ok && spec.hasBias && len(h.Bias) == h.W.Rows &&
		g.room(uint64(h.W.Rows*4)) {
		if b, e := g.dev.Alloc(h.W.Rows * 4); e == nil {
			if b.Write(f32b(h.Bias)) == nil {
				bs.hBias = b
				g.charge(uint64(h.W.Rows * 4))
			} else {
				b.Free()
			}
		}
	}
	// The greedy token on the device (nn.Head.ArgmaxOnly): the word the
	// composed sample writes. Without it the head still runs and every
	// caller gets its logits, so a failure is not one.
	if g.argmaxOut == nil && h.Sample != nil {
		w := g.scratchWin() // the device's own word (scratch.go)
		if b, e := g.dev.Alloc(4); e == nil {
			g.argmaxOut = b
		}
		w.close()
	}
	g.prewarmHeadKernels()
	// A head whose composed part has no room beside the blocks placed stays
	// home, as such a block does (prepLayer), rather than met by its first
	// step.
	if why := g.prewarmComposedHead(true); why != "" {
		g.giveBackHead(bs)
		g.Declined++
		g.LastErr = why
		return false
	}
	// A head the composer does not serve on this device runs on the CPU
	// device's composed head, never on the tier's own kernels.
	if why := bs.headServes(bs.hNorm, false, false); why != partServes {
		g.giveBackHead(bs)
		g.Declined++
		g.LastErr = fmt.Sprintf("the composer does not serve the head on this device (%s)", DeclineName(int(why)))
		return false
	}
	g.prepRowsReserved()
	return true
}

// Layers is layersFor for the zero session: a caller that never attached.
func (g *devTier) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersFor(0, lo, hi, pos, n, x, cs, csSWA, head)
}

// layersFor runs blocks [lo, hi) at position pos as one submission.
//
// The read at the end is the only synchronise: consecutive launches on one
// stream are already ordered, so the whole prefix issues without the device
// draining. Those launches are captured once and replayed (a CUDA graph pays the
// per-launch cost once); see graphKey for why a decode token's sequence is
// stable enough to capture, and cuda/graph.go for the mechanism.
//
// It runs in session sid's view of the device, holding a lane for the call
// (devsess.go); a call reached from inside another keeps the lane it holds.
func (g *devTier) layersFor(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	ok := v.layersCall(sid, lo, hi, pos, n, x, cs, csSWA, head)
	v.dropEmbed()
	return ok
}

// layersCall is Layers. The wrapper exists because an EmbedRows promise lives
// for exactly one Layers call: a call that refuses before submitting leaves the
// caller to fill x, and a surviving promise would be gathered into a later
// token's x.
func (g *devTier) layersCall(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	// Fault injection, present only in a `jitllmfault` build; see tier/fault.go.
	// In a release build injectedFail is `return false` and inlines away.
	if g.injectedFail() {
		return false
	}
	// A tower's range runs in lane0, whose tower set is the device's own
	// (visrows.go): a call holding a clone borrows lane0 for it.
	g.mu.Lock()
	if l := g.layers[lo]; lo < hi && l != nil && l.nonCausal && g.lane != g.lane0 {
		v0 := g.borrowLane0()
		g.mu.Unlock()
		ok := v0.layersCall(sid, lo, hi, pos, n, x, cs, csSWA, head)
		g.mu.Lock()
		g.returnLane0(v0)
		g.mu.Unlock()
		return ok
	}
	// Reset per call, because the caller reads it immediately after a false to
	// decide whether a host restart is sound.
	g.recSteps = 0
	// Gemma 4: the range's geometry's scratch set, and the caller's set again
	// on the way out (gemma4.go). GPU.runsInto hands one geometry per call.
	prev := g.geoCur
	if g.geoUsed {
		g.useGeom(g.geomFor(lo, hi))
	}
	g.mu.Unlock()
	defer g.restoreGeom(prev)
	// KV growth happens before anything is submitted, because it drops the
	// captured graph and dropGraph may not be called inside a backend Session
	// (a deadlock on CUDA). A non-causal range is decided first: its n is the
	// run's rows, not a position, and must not grow the causal capacity. It is
	// one whole run in its own geometry's scratch set, which the switch above
	// made current.
	// An empty range is the text head alone (PrepHead's call): its lo is the
	// block past the text model's last, which is a tower's first when a tower
	// follows, and says nothing about the call.
	if l := g.layers[lo]; lo < hi && l != nil && l.nonCausal {
		g.mu.Lock()
		bs := g.bs
		g.mu.Unlock()
		if bs == nil || n > bs.rows || head != nil {
			g.LastErr = fmt.Sprintf("non-causal n=%d scratch=%v head=%v", n, bs != nil, head != nil)
			return false
		}
		return g.submit(sid, bs, lo, hi, pos, x, cs, csSWA, nil)
	}
	if !g.ensureKVCap(pos + n) {
		return false
	}
	// Paged history takes the chunk's pages now, before any block is paged in
	// for the submission.
	g.pgRows = g.pagedRows(sid, g.pgRows, n, n, pos, nil)
	if !g.pagedAppendRows(sid, g.pgRows) {
		return false
	}
	// A prefill chunk, padded to a fixed width: every kernel bakes its row
	// count, so a ragged tail runs the full width. The surplus rows write K and V
	// at positions not yet reached, which decode overwrites, and no earlier row
	// attends to them because the scores mask past each row's causal count.
	if n != 1 {
		bw := batchWidth
		if n > bw {
			g.LastErr = fmt.Sprintf("batch n=%d width=%d head=%v", n, bw, head != nil)
			return false
		}
		// The width is the chunk's rounded up to the grain, so a short tail does
		// not run the full grid. A scratch per width costs VRAM, so the narrow one
		// is used only if it fits; otherwise the wide one pads.
		g.mu.Lock()
		grain := g.promptGrain() // f16GemmK probes once, under g.mu
		g.mu.Unlock()
		w := (n + grain - 1) / grain * grain
		if w > bw {
			return false
		}
		if g.NoTail {
			w = bw
		}
		g.mu.Lock()
		// The chunk's own width, or the width reserved at placement when the
		// narrow one does not fit (scratch.go); with no reservation, the full
		// width.
		// The chunk's own grain width first: batchFor rounds a width it has
		// not built to a power of two, which is right for a step loop whose
		// row count moves every step but ran a 415-token prompt over 512
		// rows, a quarter of its arithmetic on padding. prepBatch refuses a
		// width the budget has no room for, and the chunk then pads as before.
		//
		// Each exact width is budgeted beside the reservation. A few are
		// kept, the least recently used given back when the budget has no
		// room for the next or the lane holds maxExactWidths (keepExact);
		// a dropped width's kernels stay in the caches.
		//
		// A call the composer serves whole runs in the reserved width, its
		// rows the call's, and a longer one in chunks of that width: no
		// width is built, formed or compiled for a prompt's length.
		ww := 0
		composed := g.promptComposed(lo, hi)
		// A head over every row cannot be cut into the calls a streamed
		// block's pages hold.
		if composed && head != nil && head.RowLogits && n > g.pagedCall(lo, hi, n) {
			composed = false
		}
		switch {
		case composed && n <= g.promptW:
			ww = g.promptW
		case composed:
		default:
			if !g.NoExactPrompt && w != bw {
				if ww = g.keepExact(w); ww != 0 {
					g.ExactPrompts++
				}
			}
			if ww == 0 {
				ww = g.batchFor(w)
			}
			if ww == 0 && w != bw && g.prepBatch(bw) {
				ww = bw
			}
		}
		ok := ww > 0
		if ok {
			w = ww
		}
		bbs, dbs, sub := g.bbs[w], g.bs, max(g.promptW, nn.MaxPrefillChunk)
		if composed {
			sub = g.pagedCall(lo, hi, g.promptW)
		}
		// A head over every row (a speculative verification) runs only in
		// the one-submission path, with the per-row head built for it: every
		// fallback below splits the chunk and projects the last row alone.
		if ok && bbs != nil && head != nil && head.RowLogits {
			if dbs == nil || dbs.head == nil || !g.prepRowsHead(bbs, head, n) {
				ok = false
			} else if len(head.Tokens) < n {
				head.Tokens = make([]int32, n)
			}
		}
		// The fold copies the last row into the decode scratch and projects it
		// there (layersOnce), which a resident float head cannot take. Only a
		// resident head: the split runs the chunk before the head, and a head
		// that then failed would send a hybrid's advanced rows round again.
		// Never for a head over every row, which the per-row head above serves
		// and the split, projecting the last row alone, cannot.
		noFold := dbs != nil && dbs.head != nil && dbs.head.ok && bbs != nil &&
			!(head != nil && head.RowLogits) &&
			(layout.IsFloat(dbs.head.t) || dbs.p.NEmbd != bbs.p.NEmbd)
		g.mu.Unlock()
		if head != nil && head.RowLogits && (!ok || bbs == nil) {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("a %d-row chunk wanting every row's logits has no batched scratch", n)
			}
			return false
		}
		// Every fallback below cuts x by rows, which a stream-major residual
		// (AltUp's, DeepSeek V4's) is not: such a chunk is refused whole, and
		// the instance runs it on the host.
		if dbs != nil && dbs.p.Streams() > 1 && (head != nil && noFold || !ok || bbs == nil) {
			g.LastErr = fmt.Sprintf("a %d-row chunk of %d streams this device would have to cut by rows", n,
				dbs.p.Streams())
			return false
		}
		// A call the composer serves longer than the reserved width runs in
		// chunks of it, the last carrying the head, folded as a call of the
		// reserved width folds it.
		if composed && n > sub && dbs != nil && !(head != nil && head.RowLogits) && !noFold {
			nr, e := dbs.p.RopeW(), dbs.p.NEmbd
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			for off := 0; off < n; off += sub {
				m := min(sub, n-off)
				var rcs, rswa []float32
				if nr > 0 {
					rcs = cs[off*nr : (off+m)*nr]
					if len(csSWA) >= (off+m)*nr {
						rswa = csSWA[off*nr : (off+m)*nr]
					}
				}
				var h *nn.Head
				if off+m == n {
					h = head
				}
				if !g.layersFor(sid, lo, hi, pos+off, m, x[off*e:(off+m)*e], rcs, rswa, h) {
					return false
				}
			}
			return true
		}
		// A head rides only the one-submission path. Every fallback splits the
		// chunk, so there the head runs on the last row as its own submission;
		// so does a head the fold cannot take, rather than the whole chunk
		// going a row at a time.
		if head != nil && (!ok || bbs == nil || noFold) {
			// A head-only call (lo == hi: the blocks ran in a submission of
			// their own, GPU.layersCall's tail) has no blocks to run first;
			// asking for the empty range without the head is a refusal, which
			// sent a chunk whose blocks had all run round again a row at a
			// time (MiniMax-M3's selecting blocks are a geometry of their own,
			// so its float head always came here).
			if dbs == nil || lo < hi && !g.layersFor(sid, lo, hi, pos, n, x, cs, csSWA, nil) {
				return false
			}
			e, nr := dbs.p.NEmbd, dbs.p.RopeW()
			var rcs, rswa []float32
			if nr > 0 {
				rcs = cs[(n-1)*nr : n*nr]
				if len(csSWA) >= n*nr {
					rswa = csSWA[(n-1)*nr : n*nr]
				}
			}
			// The head's call resets recSteps (layersCall), and the chunk's
			// blocks have advanced by then: a head that fails must still
			// report them, or the caller restarts a hybrid from the embeddings
			// and applies its linear blocks twice.
			g.mu.Lock()
			adv := g.recSteps
			g.mu.Unlock()
			if !g.layersFor(sid, hi, hi, pos+n-1, 1, x[(n-1)*e:n*e], rcs, rswa, head) {
				g.mu.Lock()
				g.recSteps += adv
				g.mu.Unlock()
				return false
			}
			return true
		}
		if (!ok || bbs == nil) && n > sub && dbs != nil {
			// A device that cannot hold the wide scratch takes the chunk in narrower
			// submissions -- the width it reserved, when it reserved one --
			// consecutive sub-chunks being what the caller's own chunk loop would
			// have issued.
			sw := sub
			nr := dbs.p.RopeW()
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			for off := 0; off < n; off += sw {
				m := min(sw, n-off)
				var rcs, rswa []float32
				if nr > 0 {
					rcs = cs[off*nr : (off+m)*nr]
					if len(csSWA) >= (off+m)*nr {
						rswa = csSWA[off*nr : (off+m)*nr]
					}
				}
				e := dbs.p.NEmbd
				if !g.layersFor(sid, lo, hi, pos+off, m, x[off*e:(off+m)*e], rcs, rswa, nil) {
					return false
				}
			}
			return true
		}
		if (!ok || bbs == nil) && dbs == nil {
			return false
		}
		if (!ok || bbs == nil) && bidirCheck(g.bidir, pos, 1, &dbs.p, true) != "" && g.attendsIn(lo, hi) {
			// One row at a time cuts a bidirectional run the chunk holds,
			// which attention cannot read row by row: refused before any
			// block runs (nothing has advanced), so the chunk restarts on
			// the host whole.
			g.LastErr = "a bidirectional chunk this device has no batched scratch for"
			return false
		}
		if !ok || bbs == nil {
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			// A batch this device cannot prepare is not a device failure. prepBatch
			// refuses before submitting anything, so x is untouched and the rows go
			// one at a time. Returning false would restart the chunk on the host,
			// which on a hybrid re-runs linear blocks whose state already advanced.
			for i := 0; i < n; i++ {
				row := x[i*dbs.p.NEmbd : (i+1)*dbs.p.NEmbd]
				var rcs, rswa []float32
				if nr := dbs.p.RopeW(); nr > 0 {
					rcs = cs[i*nr : (i+1)*nr]
					if len(csSWA) >= (i+1)*nr {
						rswa = csSWA[i*nr : (i+1)*nr]
					}
				}
				// Each row is a token of the recurrence, and each linear block's
				// halves flip with it, so row 1's recording is keyed apart from
				// row 0's (rangeParity).
				if !g.submit(sid, dbs, lo, hi, pos+i, row, rcs, rswa, nil) {
					return false
				}
			}
			return true
		}
		return g.submit(sid, bbs, lo, hi, pos, x, cs, csSWA, head)
	}
	if g.PerLayerSubmit && (hi-lo > 1 || head != nil) {
		// The other arm of the submission comparison: one Session and one Read
		// per block. The head becomes one more (empty-range) submission. submit
		// honours PerLayerSubmit itself, since it is also what pages a block in.
		if !g.submit(sid, g.bs, lo, hi, pos, x, cs, csSWA, nil) {
			return false
		}
		if head != nil {
			return g.layersOnce(sid, g.bs, hi, hi, pos, x, cs, csSWA, head)
		}
		return true
	}
	return g.submit(sid, g.bs, lo, hi, pos, x, cs, csSWA, head)
}

// parityMask is the halves a submission's linear blocks read, one bit per
// linear block of its range in order: what a recording of that range is keyed
// by, since a recording names one half as source and the other as
// destination. Four words cover 256 linear blocks; a range with more is never
// recorded (rangeParity).
type parityMask [4]uint64

// rangeParity is the halves the linear blocks in [lo, hi) read next -- the
// running step's (stepPar), so a step across sessions keys on the half its
// rows were aligned to. It is the range's own rather than the device's: a
// device holding two runs of one model (an explicit placement) submits twice a
// token, and a device-wide flip per submission would come back to the same
// parity every token while each block's halves alternate -- replaying a
// recording that reads the stale half. Every block, not only the first: a
// block placed later, or a session that stepped apart from the others, can
// read a half its neighbours do not. ok is false for a range too long to key.
// Callers hold g.mu.
func (g *devTier) rangeParity(sid uint64, lo, hi int) (m parityMask, ok bool) {
	bit := 0
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.linear {
			continue
		}
		if bit == 64*len(m) {
			return m, false
		}
		if par, has := g.stepPar(sid, l); has && par == 1 {
			m[bit/64] |= 1 << (bit % 64)
		}
		bit++
	}
	return m, true
}

// replayRecurrent is emitLinear's bookkeeping for a submission that replayed a
// recording instead of emitting. The flip lives in Go, not in the graph, so a
// replay must swap each linear block's halves by hand; otherwise the same
// recording is chosen every token and the recurrence freezes. A token that
// records ran emit during the capture and has already flipped.
func (g *devTier) replayRecurrent(sid uint64, lo, hi int) {
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.linear {
			continue
		}
		if _, has := g.stepPar(sid, l); has {
			g.stepFlip(sid, l)
			g.recSteps++
			if g.rag != nil && g.rag.sid != nil {
				g.SessionLinear++
			}
		}
	}
}

// hasRecurrent reports whether any block in [lo, hi) keeps a recurrent state.
func (g *devTier) hasRecurrent(lo, hi int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.rec != nil {
			return true
		}
	}
	return false
}

// graphKey is everything about a token that decides which launch sequence it
// issues. Two tokens with the same key issue byte-identical submissions, so one
// captured graph serves both.
//
// It is short because the sequence barely varies: every buffer a decode launch
// names is allocated once and never moves, and every per-token scalar (the
// position, the page descriptors, the rotary rows) reaches the device inside a
// buffer.
type graphKey struct {
	lo, hi int
	scoreN int // the rounded position count the call's forms were asked at
	rows   int // the batch width; 1 for decode
	head   bool
	// argmax says the head ends in the device argmax (nn.Head.ArgmaxOnly):
	// one more launch, so a recording without it cannot serve a call with it.
	argmax bool
	// sampleK is the composed walk's k (nn.Head.SampleK), 0 for none: its
	// launches follow the head, so a recording without them cannot serve a
	// call with them, nor one of another k.
	sampleK int
	// pick says the head ends in the label pick (nn.Head.Pick).
	pick bool
	// rag is a LayersRows step's sequence count, 0 for any other call: ragged
	// attention and a per-row head, which a recording names by address.
	rag int
	// ragRuns is how many runs a hybrid's ragged step groups its rows into.
	ragRuns int
	// The recurrent parity: a linear block's state deliberately moves between
	// two halves, so a recording names one as source and the other as
	// destination. Replayed at the other parity it would read the stale half
	// and the state would never advance. Two recordings alternate instead,
	// as a partial seam alternates two lo/hi keys. One bit per linear block
	// (rangeParity).
	recParity parityMask
	// keys is the paged history's key count, rounded to the recording grain
	// (pagedPrep): a form over passes launches by it.
	keys int
	// ragHead is how many of a ragged step's rows the head projects
	// (nn.Head.LogitRows), and a step of the same rows wanting fewer logits
	// differs only here.
	ragHead int
	// hnorm is which norm the head reads (headNormFor): a recording names its
	// buffer by address.
	hnorm int
	// persist says the recording holds composed persistent kernels
	// (Config.Persist); persistDeps that they are ordered by dependency
	// counters (Config.PersistDeps) rather than grid barriers.
	persist, persistDeps bool
	// kvF16 is the binary16 history (Config.KVF16), which the forms read.
	kvF16 bool
	// sid is the session the recording was made for: it bakes that session's
	// page descriptors and recurrent seats, so another session never replays
	// it. Keying on it, rather than dropping every recording when the current
	// session changes, is what lets two sessions that take turns -- a
	// speculator's trunk and draft, two requests stepped one after another --
	// each keep replaying instead of recording again at every turn.
	sid uint64
	// headSrv, tableSrv, sampleSrv and embedSrv say the composer formed the
	// head, the step's tables, its token and its input rows; pickHeadSrv and
	// pickSrv that the composer's head projects the picked rows alone, or its
	// gather picks.
	headSrv, tableSrv, sampleSrv, embedSrv bool
	pickHeadSrv, pickSrv                   bool
}

// canRecord reports whether this device's sessions can capture a launch
// sequence, cached because it costs an ownership hop to ask.
func (g *devTier) canRecord() bool {
	if g.recCap == 0 {
		g.recCap = -1
		g.dev.Session(func(s backend.Session) {
			if _, ok := s.(backend.Recorder); ok {
				g.recCap = 1
			}
		})
	}
	return g.recCap == 1
}

// scoreGrain rounds the score grid's position count up to a multiple of itself.
//
// The rounding keeps the sequence stable for a run of tokens, re-capturing when
// it rolls over, as llama.cpp pads its KV length. It costs a little arithmetic
// and no correctness: AttnScores clamps its thread index with n read from a
// buffer, so surplus groups recompute and store the same last score. It rounds
// on both arms of the graph A/B so they do the same device work.
const scoreGrain = 128

func (g *devTier) layersOnce(sid uint64, bs *blockScratch, lo, hi, pos int, x, cs, csSWA []float32, head *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Every wait comes first. The prologue below reads what other sessions
	// change -- this session's pages, whether any are in host memory, the blocks --
	// into the submission's descriptors, and a wait after it would let one of
	// them change under the submission (another session's page-in or
	// eviction). Alone on the device: wait for every submission in flight.
	// Beside the others: wait for a quiesce in progress to have its turn. From
	// here to the submission nothing lets g.mu go (prologueHeld).
	// The question is asked again after every wait: another session's
	// prologue may have sent this session's history to host memory meanwhile.
	var excl bool
	for {
		if excl = g.exclusiveRange(lo, hi, g.rag, bs) || g.evictedFor(sid, g.rag); excl {
			// Running alone is right whatever changes while it waits.
			g.quiesce()
			break
		}
		if g.quiet == 0 {
			break
		}
		g.idle.Wait()
	}
	g.prologueHeld = true
	defer func() { g.prologueHeld = false }()
	// An empty range is legal only with a head: the projection under a partial
	// seam. Every refusal records why, since the instance's answer to false is a
	// silent restart on the host.
	refuse := func(f string, a ...any) bool {
		g.LastErr = fmt.Sprintf(f, a...)
		return false
	}
	switch {
	case bs == nil:
		return refuse("no scratch for this submission")
	case lo > hi || (lo == hi && head == nil):
		return refuse("an empty block range [%d,%d) with no head", lo, hi)
	case pos >= bs.p.MaxSeq:
		return refuse("position %d is past this device's capacity of %d (asked for %d)",
			pos, bs.p.MaxSeq, g.maxSeqAsked)
	case head != nil && lo < hi && bs.p.Streams() > 1:
		// AltUp's head reads the streams' mean and DeepSeek V4's their
		// collapse, which the host takes (altup.go, ds4.go): the blocks and
		// the head go as two calls.
		return refuse("a head folded into %d blocks of %d streams", hi-lo, bs.p.Streams())
	}
	// The head reads one residual vector and writes one logit row. A ragged
	// step's head is the decode scratch's; the batched one has none.
	rag := g.rag
	hb := bs
	// A chunk projecting every row (nn.Head.RowLogits) runs the ragged step's
	// per-row head, which reads the resident head through the decode scratch.
	if rag != nil || bs.rows > 1 && head != nil && head.RowLogits {
		hb = g.bs
	}
	// A batched chunk may carry the head, which then runs on the last row:
	// prefill wants logits for its final position only. The row is copied into
	// the decode scratch on the card, its norm, quantize and projection run
	// there, and only the logits are read back -- no second session, no readback of
	// the chunk.
	// A chunk whose head wants every row's logits (nn.Head.RowLogits: a
	// speculative verification) runs the ragged step's per-row head over the
	// chunk instead of folding its last row; prepRowsHead built it.
	allRows := bs.rows > 1 && head != nil && rag == nil && head.RowLogits
	fold := bs.rows > 1 && head != nil && rag == nil && !allRows
	if allRows && (!bs.ragBuilt || bs.ragLogits == nil) {
		return refuse("a %d-row chunk asked for every row's logits with no per-row head built", bs.rows)
	}
	// The per-row head caps where decode's does: a step's rows end in an
	// argmax and a readback of their own, and uncapped they would be a
	// different function from the session's decode.
	// The head's norm: the one it was built with, or a prediction block's own
	// (headnorm.go). A norm the device was never given is a refusal, never the
	// resident one in its place.
	var hNorm backend.Buf
	hNormID := 0
	if head != nil && g.bs != nil {
		var ok bool
		if hNorm, hNormID, ok = g.headNormFor(head); !ok {
			return refuse("the head brings a norm this device was not given (PrepHead)")
		}
	}
	if fold {
		hb = g.bs
		switch {
		case hb == nil || hb.head == nil || !hb.head.ok:
			return refuse("the head is not resident on this device")
		case hb.p.NEmbd != bs.p.NEmbd:
			return refuse("a batched submission of %d rows %d wide with a head %d wide", bs.rows, bs.p.NEmbd, hb.p.NEmbd)
		}
		if e := g.foldReqBuf(hb); e != nil {
			return refuse("the folded head's row take: %v", e)
		}
	}
	if head != nil && (hb == nil || hb.head == nil || !hb.head.ok) {
		return refuse("the head is not resident on this device")
	}
	if rag != nil && len(rag.pos) != len(x)/bs.p.ResidW() {
		return refuse("%d ragged rows for %d rows of x", len(rag.pos), len(x)/bs.p.ResidW())
	}
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.ok {
			return refuse("block %d is not on this device", li)
		}
		// A paged-out block's weight buffers are freed, and launching against one
		// reads whatever now lives there without faulting. submit() pages the
		// range in first; this asserts it, and is why nothing else may call
		// layersOnce.
		if l.pg != nil && !l.pg.in {
			g.LastErr = fmt.Sprintf("block %d is paged out", li)
			return false
		}
		// A ragged step's rows are different sequences, or different positions
		// of one, each reading its own history through its descriptor: only the
		// paged latent cache has one (the contiguous cache is one sequence's).
		if l.mla && rag != nil && bs.pkv == nil {
			return refuse("block %d: a ragged step over a latent block needs the paged cache", li)
		}
	}
	g.Submits++
	p := &bs.p
	// The cache row; see initScratch.

	R := bs.rows
	// The caller's row count comes from the slice and the grid does not: one
	// scratch serves one width, so a ragged chunk runs the full width with the
	// surplus rows zeroed (harmless; see Layers).
	nrow := len(x) / p.ResidW()
	if nrow < 1 || nrow > R || pos+nrow > p.MaxSeq {
		return refuse("%d rows at position %d on a scratch of %d rows and capacity %d",
			nrow, pos, R, p.MaxSeq)
	}
	hx := bs.hx[:R*p.ResidW()]
	// A promised embedding is gathered into the scratch, not uploaded (see
	// EmbedRows); the gather writes all R rows of bs.x, padding as zeros. Its
	// last workgroup must not run past R rows; otherwise the promise is made real
	// on the host.
	pe := g.takeEmbedLocked(x)
	embedSrv := false
	if pe != nil {
		// The composer's gather (ComposedEmbed) at the call's rows; a call
		// it cannot write them for (a ragged step's rows, each its own
		// sequence's) takes them made real by the same form.
		if len(pe.ids) == nrow && nrow <= R && rag == nil && g.bs.embedServes() == partServes && g.embedBufs(g.bs) &&
			g.bs.embW.k == p.NEmbd {
			embedSrv = true
		} else {
			g.materializeLocked(pe)
			pe = nil
		}
	}
	if pe == nil {
		for i := range hx {
			hx[i] = 0
		}
		// AltUp's streams are stream-major over the caller's nrow rows and
		// over the scratch's R (altup.go); one stream is a prefix either way.
		streamRows(hx, x, p, R, nrow)
	}
	// pN carries a header at R > 1: element 0 is the uniform score width every
	// grid is sized for, elements 1..R each row's own causal key count. At R == 1
	// it is the single count.
	n := uint32(pos + R)
	if p.NonCausal && g.ncPhase == 2 {
		// A row-chunked block's second phase reads every key the first
		// phase wrote, whatever chunk its queries are.
		n = uint32(g.ncKeys)
	}
	if int(n) > p.MaxSeq {
		// The score row is MaxSeq wide, so the uniform width cannot exceed it.
		// Clamping is safe: a real row needs at most pos+nrow of it.
		n = uint32(p.MaxSeq)
	}
	// The staging bytes are the devTier's, reused: they are written to the
	// device before the call returns and read by nothing after.
	pn := append(g.subBytes.pn[:0], u32one(n)...)
	if R == 1 {
		n = uint32(pos + 1)
		pn = binary.LittleEndian.AppendUint32(pn[:0], n)
	}
	// Only a backend that can record pays the rounding; elsewhere it would be
	// surplus attention work for nothing.
	nCap := int(n)
	if g.canRecord() && R == 1 {
		nCap = (int(n) + scoreGrain - 1) / scoreGrain * scoreGrain
	}
	// A call with no attention block in it (the head alone, or a hybrid's
	// linear blocks, which scan their rows in order) cuts no run: a run is
	// a key count, and only attention reads one.
	if len(g.bidir) > 0 && !p.NonCausal && g.attendsIn(lo, hi) {
		if why := bidirCheck(g.bidir, pos, nrow, p, g.pagedPlan(p)); why != "" {
			return refuse("rows [%d,%d): %s", pos, pos+nrow, why)
		}
	}
	if R > 1 {
		// The per-row counts go in after the cap, which is what the rounding
		// above would otherwise invalidate -- so a batched call does not round.
		pn = pn[:4]
		binary.LittleEndian.PutUint32(pn, uint32(nCap))
		for r := 0; r < R; r++ {
			// A padded row takes the last real row's count. It must be clamped,
			// because AttnAcc walks each row's own count and an unclamped one runs
			// past the cache (a fault on one model, a neighbour's buffer on
			// another). It must not be smaller than its neighbours', because
			// AttnAccTiled walks the count of its tile's last row, so a short one
			// truncates every real row in the tile.
			n := pos + r + 1
			if r >= nrow {
				n = pos + nrow
			}
			if p.NonCausal {
				// The whole call is one run: the mask is a per-row key count, so
				// every row gets the same count and needs no second score kernel. A
				// padded row still takes nrow, for the tile reason above.
				n = nrow
				if g.ncPhase == 2 {
					n = g.ncKeys
				}
			} else if r < nrow {
				// A row of a bidirectional run counts keys to the run's end
				// (bidir.go); bidirCheck has kept the run inside these rows.
				if e := bidirCount(g.bidir, pos+r); e > 0 && !p.BidirOff {
					n = e
				}
			}
			pn = append(pn, u32one(uint32(n))...)
		}
	}

	// A ragged step's rows are different sequences (LayersRows), each at its
	// own position with its own pages (pagedRows); what is staged here is the
	// longest history, which sizes the recording's key.
	if rag != nil {
		pn = pn[:4]
		mc := 1
		for r := 0; r < R; r++ {
			cnt := 1
			if r < nrow {
				if rag.slot[r]-rag.pos[r] < 0 {
					return refuse("row %d: slot %d at position %d", r, rag.slot[r], rag.pos[r])
				}
				cnt = rag.pos[r] + 1
			}
			mc = max(mc, cnt)
			pn = append(pn, u32one(uint32(cnt))...)
		}
		// Rounded as decode's is, so a recording serves scoreGrain steps.
		nCap = mc
		if g.canRecord() {
			nCap = min((mc+scoreGrain-1)/scoreGrain*scoreGrain, p.MaxSeq)
		}
		binary.LittleEndian.PutUint32(pn, uint32(nCap))
	}

	// Paged history: every row's descriptor and the tables, staged with the
	// other per-call writes. The pages were taken before the submission
	// (pagedAppend); here nothing may allocate one, since growing the pool can
	// page weights out from under the range about to run.
	if pk := bs.pkv; pk != nil {
		pk.rows = g.pagedRows(sid, pk.rows, R, nrow, pos, rag)
		if e := g.pagedPrep(sid, bs, pk.rows); e != nil {
			return refuse("paged attention: %v", e)
		}
		pk.st = nil
		if pk.held {
			g.pagedStage(pk, p, pk.rows)
			// A call over evicted history streams (pagedstream.go).
			if e := g.pagedStreamPrep(sid, bs, pk.rows); e != nil {
				return refuse("paged attention: %v", e)
			}
		}
	}

	// A scratch with a local rotary table (two bases, gemma3) needs it for
	// every row the global one has: short, the local layers would rotate by
	// whatever the buffer last held -- a table of other positions -- and stay
	// fluent.
	if bs.csSWA != nil && len(csSWA) < len(cs) {
		return refuse("the local layers' rotary table has %d floats against the global one's %d",
			len(csSWA), len(cs))
	}
	hcs := cs
	if R > 1 {
		hcs = bs.hcs[:R*p.RopeW()]
		for i := range hcs {
			hcs[i] = 0
		}
		copy(hcs, cs)
	}
	// The local table gets the identical treatment, and its own staging buffer:
	// sharing hcs would have the second copy overwrite the first.
	hcsSWA := csSWA
	if R > 1 && csSWA != nil {
		hcsSWA = bs.hcsSWA[:R*p.RopeW()]
		for i := range hcsSWA {
			hcsSWA[i] = 0
		}
		copy(hcsSWA, csSWA)
	}

	// rrows is how many rows the caller filled. The rotary kernel is clamped to
	// it, so the padded rows of a ragged chunk (a short prompt is all ragged)
	// keep the zeros initRopeTable wrote rather than a real rotation. The
	// position bound is checked here, where the kernel is launched. NRot is zero
	// on a vision block, so the division is guarded.
	rrows := 0
	if p.NRot > 0 {
		rrows = R
		if v := len(cs) / p.RopeW(); v < rrows {
			rrows = v
		}
		if bs.csSWA != nil {
			if v := len(csSWA) / p.RopeW(); v < rrows {
				rrows = v
			}
		}
	}
	top := pos + rrows
	if rag != nil {
		top = 0
		for _, v := range rag.pos {
			top = max(top, v+1)
		}
	}
	// The call's tables the composer serves (ComposedTable) at its rows,
	// which write cs and csSWA where the tier's own table would. A call of
	// the head alone (a model whose host collapses its streams between the
	// blocks and the head) turns nothing and is not asked for them, nor is a
	// non-causal segment's, whose rotary is the caller's table of its grid.
	tableSrv := false
	var raxes []byte
	if bs.cs != nil && p.NRot > 0 && !p.NonCausal && lo < hi && rrows >= 1 {
		if why := g.tableServes(bs, p, pos, rrows, rag); why != partServes {
			g.PartsDeclined[partRope]++
			g.StepDeclines[why]++
			return refuse("the composer does not serve the rotary tables: %s", DeclineName(int(why)))
		} else {
			tableSrv = true
		}
	}
	switch {
	case tableSrv:
		// Each row's cache position and its sequence's marks: the tables
		// turn it by its positions on the axes they compute from them.
		raxes = g.ropeRowsWords(sid, pos, rrows, R, rag)
	case bs.cs != nil:
		g.RopeTableUploads += R
	}

	// Retire recordings before the session opens: freeing one is a driver call
	// that cannot nest inside a session. See dropGraph.
	//
	// Retire on nCap, keep on lo/hi. nCap only increases, so a key differing
	// only there is dead once a longer one appears; lo/hi/head is a placement,
	// and a partial seam alternates two of them every token.
	//
	// The persistent path: decode only, and not over history streamed from
	// back, whose uploads sit between launches.
	persistOn := R == 1 && rag == nil && !(bs.pkv != nil && bs.pkv.st != nil) && g.ensurePersist()
	// The head the composer serves in this call, and so what the call leaves
	// to the caller after the logits (nn.Head.Applied): a call that serves the
	// head's graph applies all of its stages, any other the final softcap
	// alone. The token, the candidates and the picked labels are read before
	// the caller finishes the logits, so a request for them is declined while
	// a stage that bears on them is the caller's.
	var applied, left nn.HeadStage
	headWhy := partServes
	// Every head is the head scratch's graph at the rows it projects: a
	// decode step's or a prompt's one row, a chunk's folded last row, a
	// ragged step's wanted rows or every row of a chunk. A prompt's call,
	// its folded head too, asks only where the prompt is asked
	// (ComposedPrompt).
	composedHead := head != nil
	if composedHead {
		switch {
		case head.Graph != hb.headGraph || head.HiddenGraph != hb.headHidden:
			headWhy = partHeadOther
		default:
			headWhy = hb.headServes(hNorm, false, persistOn)
		}
		if headWhy == partServes {
			applied |= head.Stages
		}
	}
	if head != nil {
		left = head.Stages &^ applied
	}
	argmax := head != nil && head.ArgmaxOnly && g.argmaxOut != nil && R == 1 && left&nn.HeadBias == 0 &&
		rag == nil && !g.prompt && !fold && bs.sampleGraph != nil && head.Sample == bs.sampleGraph
	// A top-k draw's token (nn.Head.SampleK), the composer's walk written
	// where the argmax's is, where the front end states it.
	sampleK := 0
	walk := head != nil && !g.prompt && g.argmaxOut != nil && bs.sampleWalk != nil && head.SampleWalk == bs.sampleWalk
	if head != nil && !argmax && head.SampleK > 0 && R == 1 && rag == nil && !allRows && !fold &&
		left&(nn.HeadBias|nn.HeadScale) == 0 && walk && g.prepSample(head) {
		sampleK = head.SampleK
	}
	// The label pick: one row's logits, a decode's or a folded chunk's last.
	pickWant := head != nil && !argmax && sampleK == 0 && len(head.Pick) > 0 && rag == nil && !allRows &&
		(R == 1 || fold) && left&(nn.HeadBias|nn.HeadScale) == 0
	// The composer's pick, asked where the step's token is and a prompt's
	// where the prompt is: the head projecting the picked rows alone, where
	// it serves the head, no stage is left after the projection and the
	// picked rows' form is the head's own arithmetic at them, else the gather
	// over the logits.
	pickAsked := pickWant
	pickHeadSrv := pickAsked && composedHead && headWhy == partServes && hb.pickHead != nil && head.PickHead == hb.pickHead &&
		head.Stages&(nn.HeadBias|nn.HeadScale|nn.HeadCap) == 0 && hb.pickRowsExact()
	pickSrv := pickAsked && !pickHeadSrv && hb.pickGraph != nil && head.PickGraph == hb.pickGraph
	pick := pickWant && g.prepPick(head, pickHeadSrv || pickSrv)
	pickHeadSrv, pickSrv = pickHeadSrv && pick, pickSrv && pick
	// The token the composer selects (ComposedSample): a one-row step's
	// greedy token, written where the tier's argmax writes it, the sampler's
	// candidates, or the label pick.
	sampleSrv := false
	ragSample := head != nil && (rag != nil || allRows)
	if head != nil && (argmax || sampleK > 0 || pick || ragSample) {
		why := partServes
		switch {
		case ragSample && (hb.sampleGraph == nil || head.Sample != hb.sampleGraph):
			why = partNoSample
		case ragSample:
			sampleSrv = true
		case pick && pickAsked && (pickSrv || pickHeadSrv):
			sampleSrv = true
		case pick && pickAsked:
			why = partNoSample
		case sampleK > 0:
			sampleSrv = true
		case bs.sampleGraph == nil || head.Sample != bs.sampleGraph:
			why = partNoSample
		default:
			sampleSrv = true
		}
		if why != partServes {
			g.PartsDeclined[partSample]++
			g.StepDeclines[why]++
			return refuse("the composer does not serve the step's token: %s", DeclineName(int(why)))
		}
	}
	// A pick that projects its labels' rows alone leaves no logits.
	pickOnly := pick && !fold && pickHeadSrv
	headSrv := composedHead && headWhy == partServes
	// The composer is the only path: a head it does not serve is refused.
	if head != nil && !headSrv {
		reason := "the composer was not asked for it"
		if composedHead {
			reason = DeclineName(int(headWhy))
		}
		return refuse("the composer does not serve the head: %s", reason)
	}
	// A head alone over every stream collapses them into the first before it
	// (collapse.go): the caller handed them uncollapsed.
	collapse := head != nil && lo == hi && hb.collapse != nil && p.Streams() > 1
	ragN, ragRuns, ragHead := 0, 0, 0
	if rag != nil {
		ragN, ragRuns = len(rag.pos), len(rag.runs)
		if head != nil {
			ragHead = head.WantedRows(ragN)
		}
	}
	// Counted here rather than in emit, which a replayed recording skips. The
	// composer's head runs the wanted rows' count (servedHead's rows.n), and
	// ragHead is in the recording's key.
	if ragHead == 1 && headSrv {
		g.RagHeadOne++
	}
	par, parOK := g.rangeParity(sid, lo, hi)
	if allRows {
		ragHead = head.WantedRows(nrow)
	}
	keys := 0
	if bs.pkv != nil {
		keys = bs.pkv.keys
	}
	key := graphKey{lo: lo, hi: hi, scoreN: nCap, rows: R, head: head != nil, argmax: argmax, sampleK: sampleK, pick: pick,
		rag: ragN, ragRuns: ragRuns, recParity: par, keys: keys, ragHead: ragHead, hnorm: hNormID,
		persist: persistOn, persistDeps: persistOn && g.PersistDeps, kvF16: g.KVF16, sid: sid,
		headSrv: headSrv, tableSrv: tableSrv, sampleSrv: sampleSrv, embedSrv: embedSrv, pickHeadSrv: pickHeadSrv, pickSrv: pickSrv}
	if lo == hi {
		// The head alone reads no history, so the capacity is not part of
		// what it records: keyed by it, prompts on either side of a growth
		// would retire each other's recording on every call.
		key.scoreN = 0
	}
	if _, live := g.recs[key]; !live {
		for k, r := range g.recs {
			older := k
			older.scoreN = key.scoreN
			if older != key {
				continue // a different placement, still live
			}
			g.stale = append(g.stale, r)
			delete(g.recs, k)
		}
	}
	// A backstop, not the policy above, and per session: several sessions
	// taking turns each hold their own few keys, and a session over the bound
	// loses its least recently used.
	if bound := g.graphBound(); len(g.recs) > bound {
		var mine []graphKey
		for k := range g.recs {
			if k.sid == sid {
				mine = append(mine, k)
			}
		}
		if len(mine) > bound {
			slices.SortFunc(mine, func(a, b graphKey) int { return cmp.Compare(g.recUsed[a], g.recUsed[b]) })
			for _, k := range mine[:len(mine)-bound] {
				g.stale = append(g.stale, g.recs[k])
				delete(g.recs, k)
			}
		}
		for k := range g.recUsed {
			if _, live := g.recs[k]; !live {
				delete(g.recUsed, k)
			}
		}
	}
	g.freeStale()

	g.subBytes.pn = pn
	t0 := time.Now()
	g.sub = submitArgs{
		bs: bs, hb: hb, head: head, lo: lo, hi: hi, pos: pos, x: x,
		hx: hx, hcs: hcs, hcsSWA: hcsSWA, pn: pn,
		p: p, rag: rag, pe: pe, key: key, R: R, nrow: nrow,
		nCap: nCap, rrows: rrows,
		tableSrv: tableSrv, sampleSrv: sampleSrv, embedSrv: embedSrv, raxes: raxes,
		fold: fold, argmax: argmax, sampleK: sampleK, pick: pick, ragHead: ragHead, parOK: parOK,
		allRows: allRows, hNorm: hNorm, headWhy: headWhy, pickOnly: pickOnly,
		pickHeadSrv: pickHeadSrv, pickSrv: pickSrv,
		collapse: collapse,
	}
	if g.subFn == nil {
		g.subFn = g.layersSession
	}
	// Read here, under g.mu: a page-in another session makes changes it.
	g.sub.paging = g.pagesIn < len(g.layers)
	g.sub.persist = persistOn
	g.sessQueue()
	// The prologue streams this call's history only when some of it was at
	// in host memory before the prologue began (evictedFor), which made the call one
	// to run alone; nothing can have gone to host memory since, so this is a check of
	// that reasoning and not a path.
	if !excl && g.exclusiveRange(lo, hi, rag, bs) {
		return refuse("a submission that must run alone was prepared to run beside others")
	}
	g.prologueHeld = false
	if excl {
		// Alone on the device, as every submission was before sessions ran
		// at once: nothing else is in flight (the wait was before the
		// prologue), and nothing starts while g.mu is held across it.
		g.runSub(g.subFn)
	} else {
		// Beside any other session's: g.mu is let go for the submission, so
		// the bookkeeping of the others goes on meanwhile. What the
		// submission builds on the way takes g.mu for the build (subLock),
		// and its counters are its own until it completes.
		t := g.beginSub()
		g.inSub, g.Stats = true, &g.acc
		g.runUnlocked(g.subFn)
		g.inSub, g.Stats = false, &g.tot
		g.tot.add(g.acc)
		g.acc = Stats{}
		g.endSub(t)
		g.freeStale()
	}
	err, direct := g.sub.err, g.sub.direct
	g.sub = submitArgs{}
	if err != nil {
		g.LastErr = err.Error()
		if nn.IsBuildError(err) {
			g.noteCause(nn.FailBuild)
		} else {
			g.noteCause(nn.FailFault)
		}
		return false
	}
	if head != nil {
		head.Applied = applied
	}
	if head != nil && !direct {
		for i := range head.Logits {
			head.Logits[i] = hb.hOut[i]
		}
	} else if head == nil && !direct {
		streamRows(x, bs.hout, p, nrow, R)
	}
	g.TLayer += time.Since(t0)
	g.Blocks += hi - lo
	// The submission has completed (its readback waited for it), so pages
	// released before it are no longer read by anything in flight, and the
	// rows it wrote are history eviction may send to host memory.
	if pk := bs.pkv; pk != nil && pk.held && g.kvp != nil {
		g.kvp.wrote(pk.rows)
	}
	if g.kvp != nil {
		for _, l := range g.kvp.layers {
			l.fence()
		}
	}
	return true
}

// submitArgs is one layersOnce submission's state, handed to layersSession
// through the session's devSess rather than captured by a closure: the
// function a Session runs escapes, so a closure over the submission's locals
// was a heap object -- with every variable it shared with its own closures --
// on every token. A session runs one submission at a time on a device, so one
// set a session serves.
type submitArgs struct {
	// paging says a block of the device is paged out, read before the
	// submission: a recording names device addresses a page-in allocates
	// afresh.
	paging bool
	// persist says the submission composes its decode blocks (persist.go).
	persist            bool
	bs, hb             *blockScratch
	head               *nn.Head
	lo, hi, pos        int
	x, hx, hcs, hcsSWA []float32
	pn                 []byte
	p                  *nn.LayerPlan
	rag                *ragStep
	pe                 *embPend
	key                graphKey
	R, nrow            int
	nCap               int
	rrows              int
	ragHead            int
	fold, argmax       bool
	// tableSrv and sampleSrv say the composer serves the step's tables and
	// its token; raxes is a table over several axes' staged positions.
	tableSrv, sampleSrv bool
	// embedSrv says the composer gathers the decode row.
	embedSrv       bool
	raxes          []byte
	sampleK        int
	pick           bool
	parOK, allRows bool
	hNorm          backend.Buf
	// headWhy is why the composer is not asked for the head at this one-row
	// step, partServes where it is; pickOnly says the head projects the
	// picked labels alone.
	headWhy  partDecline
	pickOnly bool
	// collapse says the head reads the collapse of every stream (collapse.go).
	collapse bool
	// pickHeadSrv and pickSrv say the composer picks the labels: its head
	// over the picked rows alone, or its gather over the logits.
	pickHeadSrv, pickSrv bool
	// What the session hands back.
	err    error
	direct bool
}

// layersSession is the device side of layersOnce: the staging writes, the
// launches (or a graph replay) and the readback, in one Session. Its arguments
// are g.sub's; see submitArgs.
func (g *devTier) layersSession(s backend.Session) {
	// The persistent path (persist.go) runs the submission through psess,
	// which issues what the launcher holds before any other call on the
	// session. Entered again rather than s reassigned below: a reassigned
	// parameter that closures capture moves to the heap, five objects a token.
	if g.sub.persist && g.sub.R == 1 && g.sub.rag == nil && g.launchTo.ps != nil &&
		g.launchTo.ps.sync != nil && g.launchTo.psess.Session == nil {
		g.launchTo.psess = persistSession{Session: s, t: &g.launchTo, g: g}
		g.layersSession(&g.launchTo.psess)
		g.launchTo.psess = persistSession{}
		return
	}
	g.injectedPanic()
	a := &g.sub
	// The session the submission runs for, carried in its recording's key.
	sid := a.key.sid
	bs, hb, head, lo, hi, pos, x := a.bs, a.hb, a.head, a.lo, a.hi, a.pos, a.x
	hx, hcs, hcsSWA, pn := a.hx, a.hcs, a.hcsSWA, a.pn
	p, rag, pe, key := a.p, a.rag, a.pe, a.key
	R, nrow := a.R, a.nrow
	fold, argmax, sampleK, pick := a.fold, a.argmax, a.sampleK, a.pick
	tableSrv, sampleSrv, embedSrv, raxes := a.tableSrv, a.sampleSrv, a.embedSrv, a.raxes
	ragHead, parOK, allRows, hNorm := a.ragHead, a.parOK, a.allRows, a.hNorm
	pickOnly := a.pickOnly
	pickHeadSrv := a.pickHeadSrv
	collapse := a.collapse
	var err error
	direct := false
	defer func() { a.err, a.direct = err, direct }()
	// Runs first: no admitted sheet is in flight when the call returns.
	defer func() {
		if len(g.adm) > 0 || len(g.admPub) > 0 {
			if aerr := g.finishAdmits(s); err == nil {
				err = aerr
			}
		}
	}()
	w := func(b backend.Buf, p []byte) {
		if err == nil {
			t := time.Now()
			err = s.Write(b, p)
			g.TSubStage += time.Since(t)
		}
	}
	// lc is the submission's launch path; its la is the ordinary launch
	// (see launcher).
	g.launchTo.s = s
	if ps, ok := s.(*persistSession); ok {
		g.launchTo.s = ps.Session
	}
	persist := g.launchTo.psess.Session != nil
	defer func() { g.launchTo.s = nil }()
	lc := &launcher{to: &g.launchTo, err: &err, g: g}

	// Scanned rather than counted on the tier, so no release path can let a
	// counter drift.
	streamed := false
	for li := lo; li < hi && !streamed; li++ {
		// A batched mixture reads its selection back too (moegroup.go).
		if l := g.layers[li]; l != nil && (l.stream != nil || (R > 1 && l.router != nil)) {
			streamed = true
		}
	}
	// emit issues the whole token's launches. It is a closure so the same code
	// both runs the sequence and records it: one source for both means a graph
	// cannot differ from the launches it replaces.
	emit := func() {
		// The rotary table first: every block reads it and nothing writes it again.
		// Both launches are pure functions of buffers, so a replay rebuilds it from
		// whatever bs.rpos holds this token.
		switch {
		case tableSrv:
			for i := range bs.tableCount() {
				cs, planes, at := g.tableBinds(bs, i)
				sv, serr := g.servedTable(scratchTable(bs, p, i), R, p.RopeMarks, cs, planes, at)
				if serr == nil {
					serr = sv.run(g, lc, sv.inst, nil, nil)
				}
				if err == nil {
					err = serr
				}
			}
		}
		for li := lo; li < hi; li++ {
			l := g.layers[li]
			// The previous block's held launches go out; this block's are
			// held when it is one the path takes.
			lc.persistBlock(persist && !streamed && persistable(l, bs))
			// The block the composer serves whole: its graph from the model's
			// front end, formed as one unit and launched on the block's own
			// weights and history -- a decode step's, or a prompt's at its
			// rows (ComposedPrompt): a chunk, or a one-row call of the prompt
			// (nn.PromptDevice), never counted as a step.
			// A ragged step's rows are a decode step of several sessions,
			// each row at its own position and history (its descriptor):
			// its blocks are asked as decode's, at the step's rows, and a
			// prompt's ragged rows as the prompt's.
			kind := g.rowsKind(R, rag != nil)
			// A non-causal call is a picture's rows, one run: its blocks are
			// asked whole, at the call's rows.
			if l.nonCausal {
				kind = partTower
			}
			why := g.blockServes(l, li, p, bs, persist, kind == partPrompt)
			if why == partServes && R > 1 {
				why = l.expertsServe(nrow)
			}
			var sv, cp *servedPart
			var inst *partInst
			if why == partServes {
				var serr error
				sv, inst, cp, serr = g.servedStep(bs, l, li, p, R, rag == nil || rag.oneSequence())
				switch {
				case errors.Is(serr, errStreamForm):
					g.LastErr = serr.Error()
					sv, inst, cp, why = nil, nil, nil, partStream
				case serr != nil:
					if err == nil {
						err = serr
					}
					continue
				}
				switch {
				case why != partServes:
				case l.linear && !g.bindRec(sv, bs, sid, l):
					inst, why = nil, partRecurrent
				case inst == nil:
					why = partNoRoom
				}
				if R > 1 && sv != nil {
					sv.rows = nrow
					if cp != nil {
						cp.rows = nrow
					}
				}
			}
			if why != partServes {
				g.PartsDeclined[kind]++
				g.BlockDeclines[why]++
			}
			// The composer is the only path: a block it does not serve is
			// refused, never run on the tier's own kernels.
			if inst == nil {
				if err == nil {
					err = fmt.Errorf("tier: the composer does not serve block %d at %d rows: %s", li, R, DeclineName(int(why)))
					if why == partNoRoom && g.LastErr != "" {
						err = fmt.Errorf("%w (%s)", err, g.LastErr)
					}
				}
				continue
			}
			{
				out := bs.x
				if cp != nil {
					out = bs.h
				}
				// A second input is the rows' per-layer inputs (blockServes);
				// a block of several streams reads and writes each at its
				// offset in x.
				in := []backend.Buf{bs.x, bs.pleIn}
				outs := []backend.Buf{out}
				if len(l.graph.Out) > 1 {
					var verr error
					if in, outs, verr = bs.servedStreams(R, len(l.graph.Out)); verr != nil {
						if err == nil {
							err = verr
						}
						continue
					}
				}
				var serr error
				switch {
				case sv.f.split > 0:
					serr = sv.pagedRun(g, lc, s, inst, kind, l, p.NExpert, g.hostFns[sid][li].uses, in, outs)
				case sv.f.passHi > 0:
					serr = sv.streamRun(g, lc, s, inst, kind, bs, g.kvp.layers[l.kvSrc], in, outs)
				default:
					serr = sv.runAs(g, lc, inst, kind, in, outs)
				}
				if err == nil {
					err = serr
				}
				if cp != nil {
					if serr := cp.run(g, lc, cp.inst, []backend.Buf{bs.h}, []backend.Buf{bs.x}); err == nil {
						err = serr
					}
				}
				// The form wrote the other half: the session reads it next,
				// as after emitLinear.
				if l.linear && err == nil {
					g.stepFlip(sid, l)
					g.recSteps++
					if rag != nil && rag.sid != nil {
						g.SessionLinear++
					}
				}
				continue
			}
		}
		if collapse && err == nil {
			err = g.runCollapse(lc, hb.collapse, bs, R)
		}
		// The head the composer serves: the step fails if it does not
		// form, every stage its graph states applied.
		if head != nil && (rag != nil || allRows) {
			// The wanted rows (they lead) at their own count.
			out := bs.ragLogits
			rows := ragHead
			if rowsFaulted("headrow") {
				rows = 1
			}
			// The head of the step's scratch's capacity, its rows the wanted
			// ones: one form whatever the count.
			sv, serr := g.servedHead(hb, hNorm, bs.rows)
			if serr == nil {
				sv.rows, sv.sl.tables[tableKey(composer.TableCall)] = rows, bs.headCalls
				serr = sv.runHead(g, lc, bs.x, out, bs.h)
			}
			if err == nil {
				err = serr
			}
			if err == nil {
				err = g.ragTokens(lc, bs, hb, ragHead, sampleSrv)
			}
		} else if fold {
			// The chunk's last row, into the decode scratch, and decode's
			// head on it.
			if tk, terr := g.servedRowTake(hb, p.NEmbd, R); terr != nil {
				err = terr
			} else {
				g.rows.takeIn[0], g.rows.takeOut[0] = bs.x, hb.x
				err = tk.run(g, lc, tk.inst, g.rows.takeIn[:], g.rows.takeOut[:])
			}
			out := hb.logits
			in := hb.x
			if rowsFaulted("foldrow") {
				in = bs.x
			}
			if pickHeadSrv {
				if err == nil {
					err = g.runPickHead(lc, hb, hNorm, in)
				}
			} else {
				sv, serr := g.servedHead(hb, hNorm, 1)
				if serr == nil {
					serr = sv.runHead(g, lc, in, out, hb.h)
				}
				if err == nil {
					err = serr
				}
				if pick && err == nil {
					err = g.runPick(lc, hb)
				}
			}
			g.HeadFolds++
		} else if head != nil {
			// The output norm and the vocabulary projection, in the same
			// submission: the residual stream is never read back.
			if pickHeadSrv {
				if err == nil {
					err = g.runPickHead(lc, bs, hNorm, bs.x)
				}
			} else {
				out := bs.logits
				sv, serr := g.servedHead(bs, hNorm, 1)
				if serr == nil {
					serr = sv.runHead(g, lc, bs.x, out, bs.h)
				}
				if err == nil {
					err = serr
				}
			}
			switch {
			case argmax && sampleSrv && err == nil:
				sv, serr := g.servedSample(bs.sampleGraph)
				if serr == nil {
					serr = sv.run(g, lc, sv.inst, []backend.Buf{bs.headOut()}, []backend.Buf{g.argmaxOut})
				}
				err = serr
			}
			switch {
			case sampleK > 0 && sampleSrv && err == nil:
				sv, serr := g.servedSample(bs.sampleWalk)
				if serr == nil {
					serr = sv.run(g, lc, sv.inst, []backend.Buf{bs.headOut(), g.sampleWindow, g.samplePenalty, g.sampleWalk}, []backend.Buf{g.argmaxOut})
				}
				err = serr
			}
			if pick && !pickOnly && err == nil {
				err = g.runPick(lc, bs)
			}
		}
		// What the last block held goes out inside emit, so a recording
		// captures it.
		lc.persistFlush()
	}
	// submit gets the sequence onto the device, by replaying a captured graph
	// where the backend has one and by issuing the launches where it does not.
	// The staging writes and the readback stay outside it: a captured memcpy
	// would bake in a Go heap address.
	submit := func() {
		if err != nil {
			return
		}
		tl := time.Now()
		defer func() { g.TSubLaunch += time.Since(tl) }()
		rec, ok := g.recorderOf(s)
		if !ok {
			g.graphOff = true
		}
		// No recording when the key would not hold still or a replay cannot be
		// right:
		// - PerLayerSubmit changes the key every block, so -ab submit with a
		//   resident head compares two changes at once unless NoGraph is set on
		//   both arms.
		// - A prefill chunk's key steps with its position, so it would be
		//   captured every chunk and replayed never. A ragged step is recorded
		//   like decode.
		// - A paging device: a graph holds device addresses, and a page-in
		//   allocates the buffers afresh.
		// - A streamed block: the mid-FFN host readback and upload have no place
		//   in a replay.
		// - History streamed from host memory: its uploads and waits are per call.
		if (R > 1 && rag == nil) || !parOK || g.NoGraph || g.graphOff || g.PerLayerSubmit ||
			a.paging || streamed || bs.pkv != nil && bs.pkv.st != nil {
			te := time.Now()
			emit()
			g.TEmit += time.Since(te)
			return
		}
		cur, have := g.recs[key]
		if !have {
			tc := time.Now()
			parts := g.PartsServed
			g.capturing = true
			r, e := backend.Record(rec, emit)
			g.capturing = false
			g.TCapture += time.Since(tc)
			if errors.Is(err, errServedCold) {
				// A served slice not formed yet: its compile and its
				// constants' upload have no place in a capture, so this step
				// is issued unrecorded, forming it, and the next is captured.
				if r != nil {
					g.stale = append(g.stale, r)
				}
				// The capture ran no part: the issue below counts them.
				err, g.PartsServed = nil, parts
				emit()
				return
			}
			if e != nil || err != nil {
				// A capture executes nothing, so the work has still to be
				// issued -- the ordinary way, since a capture that failed
				// once will fail again. A recording that came back around a
				// failed launch is complete as far as the driver knows and
				// wrong as far as jitllm does, so it is retired unused.
				g.graphOff, err, g.PartsServed = true, nil, parts
				if e != nil {
					g.LastErr = "graphs disabled: " + e.Error()
				}
				if r != nil {
					g.stale = append(g.stale, r)
				}
				emit()
				return
			}
			if g.recs == nil {
				g.recs = make(map[graphKey]backend.Recording)
			}
			g.recs[key], cur = r, r
			g.Captures++
			if g.recParts == nil {
				g.recParts = map[graphKey][numParts]int{}
			}
			for i := range parts {
				parts[i] = g.PartsServed[i] - parts[i]
			}
			g.recParts[key] = parts
		}
		if e := rec.Replay(cur); e != nil {
			g.graphOff, g.LastErr = true, "graphs disabled: "+e.Error()
			g.stale = append(g.stale, cur)
			delete(g.recs, key)
			emit()
		} else {
			if g.recUsed == nil {
				g.recUsed = map[graphKey]uint64{}
			}
			g.recTick++
			g.recUsed[key] = g.recTick
			if have {
				g.replayRecurrent(sid, lo, hi)
				for i, n := range g.recParts[key] {
					g.PartsReplayed[i] += n
				}
			}
		}
	}

	// Position state is per token, not per block: written once here rather
	// than 54 times across 18 blocks.
	if fold {
		w(hb.foldReq, g.foldWords(nrow-1))
	}
	if embedSrv {
		tok, at := g.embedWords(pe, R)
		w(g.bs.embTok, tok)
		if g.bs.posW != nil {
			w(g.bs.embPos, at)
		}
		if pe.kind == partEmbedRows {
			g.EmbedLaunches++
		}
		sv, serr := g.servedEmbed(g.bs, R)
		var outs []backend.Buf
		if serr == nil {
			outs, serr = bs.embedOuts(g.bs.embStreams(), R)
		}
		if serr == nil && err == nil {
			serr = sv.runAs(g, lc, sv.inst, pe.kind, nil, outs)
		}
		if err == nil {
			err = serr
		}
	} else {
		w(bs.x, f32b(hx))
	}
	// Input rows over every stream write the per-layer inputs too
	// (embedOuts).
	if bs.rowTok != nil && len(g.tokIDs) > 0 {
		w(bs.rowTok, g.tokWords(R))
	}
	if bs.rowPos != nil {
		w(bs.rowPos, g.rowPosWords(pos, nrow, R, rag))
	}
	switch q := p.Q4E; {
	case bs.pleIn != nil && p.PLEDim == 0 && q != nil && q.NGramWidth != 0:
		w(bs.pleIn, f32b(g.sideRows(g.ngHost, R*q.NGramWidth)))
	case bs.pleIn != nil && !(embedSrv && g.bs.embStreams() > 1):
		w(bs.pleIn, f32b(g.pleRows(R*p.PLEWidth)))
	}
	switch {
	case raxes != nil:
		w(bs.ropeRows, raxes)
	default:
		if bs.cs != nil {
			// nil for a block with no rotary; nothing reads it and there is
			// no buffer to write into.
			w(bs.cs, f32b(hcs))
		}
		if bs.csSWA != nil {
			w(bs.csSWA, f32b(hcsSWA))
		}
	}
	w(bs.n, pn)
	if bs.calls != nil && R > 1 {
		binary.LittleEndian.PutUint32(g.subBytes.calls[:], uint32(nrow))
		w(bs.calls, g.subBytes.calls[:])
	}
	if bs.pics != nil {
		w(bs.pics, bs.picDesc(pos, nrow, g.ncPhase, g.ncKeys))
		w(bs.picsW, bs.picWinDesc(pos, nrow, g.ncPhase, g.ncKeys, g.winRuns, g.winPerRow))
	}
	if bs.headCalls != nil && R > 1 && ragHead > 0 {
		hr := ragHead
		if rowsFaulted("headrow") {
			hr = 1
		}
		binary.LittleEndian.PutUint32(g.subBytes.headCalls[:], uint32(hr))
		w(bs.headCalls, g.subBytes.headCalls[:])
	}
	switch {
	case sampleK > 0 && sampleSrv:
		// The composed sampler reads the penalty, the window where a penalty
		// applies (a penalty of 1 leaves the row whatever the window holds)
		// and the walk's words.
		binary.LittleEndian.PutUint32(g.subBytes.penalty[:], math.Float32bits(head.SamplePenalty))
		w(g.samplePenalty, g.subBytes.penalty[:])
		if head.SamplePenalty != 1 {
			w(g.sampleWindow, f32b(head.SampleWindow))
		}
		w(g.sampleWalk, f32b(head.WalkArgs[:]))
	}
	if pick {
		// The ids change every question; the recording reads them.
		w(g.pickIDs, g.pickStage(head))
	}
	if bs.dRows != nil {
		// The recurrent descriptor (see emitLinear), staged here outside any
		// recording: written inside emit it was a memcpy mid-capture, which
		// invalidates a CUDA stream capture. Its row count is nrow, the
		// caller's rows, not the grid R: padding rows must not go through
		// the convolution's shift and the delta rule as tokens. Its slots
		// are where each sequence's state is in the pools, which a
		// recording therefore does not bake.
		w(bs.dRows, g.recDesc(sid, nrow, false))
		if g.recShared && bs.cRows != nil {
			w(bs.cRows, g.recDesc(sid, nrow, true))
		}
	}
	if pk := bs.pkv; pk != nil && pk.held {
		w(pk.pRow, u32view(pk.desc))
		if len(pk.descW) > 0 {
			w(pk.pRowW, u32view(pk.descW))
		}
		for _, e := range pk.ents {
			w(e.buf, u32view(e.words))
		}
		if pk.st != nil {
			pk.st.writes(w)
		}
	}

	submit()

	if err != nil {
		return
	}
	// Read straight into the caller's slice when the lengths agree: copying
	// out of bs.hRaw was a measurable cost for a large vocabulary, and
	// cuMemcpyDtoH takes any host pointer.
	tr := time.Now()
	defer func() { g.TSubRead += time.Since(tr) }()
	// The head's input, before the reads below consume the session's
	// attention: the normed rows a prediction block reads (nn.Head.Hidden).
	if head != nil && head.Hidden != nil && err == nil {
		src, rows := bs.h, 1
		switch {
		case rag != nil || allRows:
			rows = ragHead
		case fold:
			src = hb.h
		}
		if len(head.Hidden) < rows*p.NEmbd {
			err = fmt.Errorf("a %d-float hidden destination for %d rows of %d", len(head.Hidden), rows, p.NEmbd)
		} else {
			err = s.Read(src, f32b(head.Hidden[:rows*p.NEmbd]))
		}
		if err == nil && key.headSrv {
			g.HiddenHeads++
			g.HiddenRows += rows
		}
	}
	switch {
	case (rag != nil || allRows) && head != nil:
		tb := slices.Grow(g.subBytes.tok[:0], 4*R)[:4*R]
		g.subBytes.tok = tb
		if err = s.Read(bs.ragTok, tb); err == nil {
			for r := range ragHead {
				head.Tokens[r] = int32(binary.LittleEndian.Uint32(tb[4*r:]))
			}
		}
		// The rows' logits are already in ragLogits, [R][vocab]; the first
		// ragHead rows are the wanted ones.
		if need := ragHead * g.bs.head.nrows; err == nil && head.RowLogits {
			if len(head.Logits) < need {
				err = fmt.Errorf("rows: %d logits for %d rows of %d", len(head.Logits), ragHead, g.bs.head.nrows)
			} else {
				err = s.Read(bs.ragOut(), f32b(head.Logits[:need]))
			}
		}
		direct = true
	case sampleK > 0:
		tok := g.subBytes.one[:]
		if err = s.Read(g.argmaxOut, tok); err == nil {
			head.Token = int32(binary.LittleEndian.Uint32(tok))
		}
		head.Sampled = err == nil
		// Counted here, not at the launches, which a replayed recording skips.
		g.SampleReads++
		direct = true
	case pick:
		err = s.Read(g.pickOut, f32b(head.PickVals[:len(head.Pick)]))
		head.Picked = err == nil
		// Counted here, not at the launch, which a replayed recording skips.
		g.PickReads++
		direct = true
	case argmax:
		tok := g.subBytes.one[:]
		if err = s.Read(g.argmaxOut, tok); err == nil {
			head.Token = int32(binary.LittleEndian.Uint32(tok))
		}
		direct = true
	case head != nil && len(head.Logits) == len(hb.hOut):
		err = s.Read(hb.headOut(), f32b(head.Logits))
		direct = true
	case head != nil:
		err = s.Read(hb.headOut(), hb.hRaw)
	case len(x) == nrow*p.ResidW() && (p.Streams() == 1 || nrow == R):
		err = s.Read(bs.x, f32b(x))
		direct = true
	case p.Streams() > 1:
		// Every stream is read back, stream-major at R rows (streamRows cuts it).
		err = s.Read(bs.x, bs.hraw[:R*p.ResidW()*4])
	default:
		err = s.Read(bs.x, bs.hraw[:nrow*p.NEmbd*4])
	}
}

// recorderOf is s as a backend.Recorder, asserted once per session object:
// the backends reuse their sessions, and an interface-to-interface assertion
// on every token caches its answer in a heap object at a random later call.
func (g *devTier) recorderOf(s backend.Session) (backend.Recorder, bool) {
	if g.recSess != s {
		g.recSess = s
		g.rec, _ = s.(backend.Recorder)
	}
	return g.rec, g.rec != nil
}

// stagingBytes is a devTier's host bytes for a submission's small writes,
// reused so a decode token allocates none of them.
type stagingBytes struct {
	pn               []byte
	tok              []byte // a ragged step's tokens, read back
	embIDs           []byte // a prompt chunk's ids for the gather on the card (embIDs)
	ropeRows         []byte // the rows as the rotary tables read them (ropeRowsWords)
	rowPos           []byte // the rows' positions a block reads (rowPosWords)
	rowTok           []byte // the rows' tokens a block routing by token reads (tokWords)
	embPos           []byte // the composer's gather's positions (embedWords)
	fold             []byte // the folded head's row take's request (foldWords)
	one              [4]byte
	calls, headCalls [4]byte // a call's rows (blockScratch.calls, headCalls)
	penalty          [4]byte // the sampler's penalty (nn.Head.SamplePenalty)
}

// u32b is v as four little-endian bytes in the devTier's own scratch, for a
// write that completes before the next one: every backend's Write copies p
// before it returns.
func (g *devTier) u32b(v uint32) []byte {
	binary.LittleEndian.PutUint32(g.subBytes.one[:], v)
	return g.subBytes.one[:]
}

func u32one(v uint32) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
}

// auxBufs is every per-layer device vector that a page-out and a release both
// drop. It is one list because separate copies drifted repeatedly, each time
// leaking a new kind of vector (vision MLP biases, post-norms, a linear block's
// five, gpt-oss's sinks and biases). Both callers are full teardowns, so
// nothing here is re-read without being uploaded again.
func (l *layer) auxBufs() []backend.Buf {
	return append(append([]backend.Buf(nil),
		l.nAttn, l.nFFN, l.nAttnB, l.nFFNB, l.nQ, l.nK,
		l.nResid, l.nResidB, // BERT's post-norm
		l.nPostAttn, l.nPostFFN, // gemma2/gemma3
		l.nSubAttn, l.nSubFFN, // BitNet
		l.clampB,      // Gemma 4's tower: its clipped linears' bounds
		l.nQA, l.nKVA, // MLA's two latent norms
		l.nIdxK, l.nIdxKB, l.nIdxQ, // the indexers' norms (DeepSeek V3.2, MiniMax-M3)
		l.altRouter, l.altRouterNorm, l.altPredT, l.altCorrT, l.altCorrScale, l.nLaurel, // Gemma 3n
		l.ssmConv1d, l.ssmA, l.ssmDt, l.ssmNorm, l.ssmOnes, l.ssmTaps.qs, // a linear block
		l.ssmD, l.ssmCB, // Mamba-2's and Mamba-1's
		l.m1DtN, l.m1BN, l.m1CN, // Mamba-1's
		l.router,
		l.sinks, l.routerB, l.expGateB, l.expUpB, l.expDownB, // gpt-oss
		l.expSelB,  // DeepSeek-V3
		l.shRouter, // qwen2moe's and qwen3next's shared-expert gate
		l.mvq.bias, l.mvk.bias, l.mvv.bias, l.mvo.bias,
		l.mvu.bias, l.mvd.bias,
		l.outScale,                                         // Gemma 4's output scalar
		l.xielu,                                            // Apertus's activation numbers
		l.shadow,                                           // the shadow arm's state
		l.nFFN2, l.nPost1, l.nPost2, l.nRouter, l.expScale, // Gemma 4's mixture block
		l.nPLE, l.pleOff, // Gemma 4's per-layer embedding
	), append(append(append(l.ds4.bufs(), l.k3.bufs()...), l.q4e.bufs()...), l.conv.bufs()...)...) // DeepSeek V4's, Kimi-K3's, Qwen3.8-Flash-Next's, a convolutional block's
}

// dropLayerLocal frees the buffers PrepLayer allocates fresh for one block and
// shares with nothing: the KV cache, the norms, the biases and the router. It
// does not touch the weights, which live in g.res keyed by their bytes and may
// be referenced elsewhere. Callers hold g.mu.
func (g *devTier) dropLayerLocal(l *layer) {
	for _, b := range l.auxBufs() {
		if b != nil {
			b.Free()
		}
	}
	g.convLeft(l)
	g.freeKV(l)
	freeRec(l)
	if l.actK != nil {
		l.actK.Close()
		l.actK = nil
	}
	if n := l.recBytes; n > 0 {
		g.refund(n)
		g.RecBytes -= n
		l.recBytes = 0
	}
	if n := l.kvBytes; n > 0 {
		g.refund(n)
		if g.KVBytes >= n {
			g.KVBytes -= n
		}
	}
	if n := l.auxBytes; n > 0 {
		g.refund(n)
	}
}

// ReleaseLayers hands blocks [lo, hi) back to the host and frees what they held
// on the device, which lets placement be undone (demote, park, re-adopt).
//
// It must drop the recording: a captured sequence holds device addresses, and
// replaying it after these buffers are freed reads memory that belongs to
// something else. Kernels are not closed, since g.kerns is shared by every layer
// of the same shape; only the per-layer router kernel is. The packed arena is
// not touched, so re-preparing the block is an upload rather than a repack;
// giving the host bytes back is GPU.DropArena.
//
// It takes the whole block, every session's history included: it is what a
// move between devices and a device's teardown use. A session giving blocks
// back goes through leaveLayers.
func (g *devTier) ReleaseLayers(lo, hi int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	for li := lo; li < hi; li++ {
		g.releaseBlock(li)
	}
	g.afterRelease()
}

// leaveLayers is ReleaseLayers for one session: session sid's history
// on blocks [lo, hi) goes, and a block goes with it only when no other
// session has history there -- the weights are the model's, and another
// session is still running against them. It returns the blocks it released.
func (g *devTier) leaveLayers(sid uint64, lo, hi int) []int {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	var gone []int
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil {
			continue
		}
		if g.sharedBeyond(li, sid) {
			g.dropHistory(li, l, sid)
			continue
		}
		g.releaseBlock(li)
		gone = append(gone, li)
	}
	g.afterRelease()
	return gone
}

// releaseBlock frees block li and every session's history on it. Callers
// hold g.mu and drop the graph.
func (g *devTier) releaseBlock(li int) {
	l := g.layers[li]
	g.freeRecSnaps(li)
	if l == nil {
		return
	}
	if l.graph != nil {
		g.dropServed(l.graph)
	}
	// The same free path as a page-out (layer.tensors()); a release also
	// gives up the KV cache and the block itself.
	freed := g.dropTensors(l)
	if l.pg != nil && l.pg.in {
		l.pg.in = false
		g.pagesIn--
		g.pageBytes -= min(g.pageBytes, freed+l.pg.round)
	}
	// The KV cache is charged to the budget at PrepLayer, so releasing has
	// to give it back or the card leaks budget every migration.
	if n := l.kvBytes; n > 0 {
		g.refund(n)
		if g.KVBytes >= n {
			g.KVBytes -= n
		}
	}
	if n := l.auxBytes; n > 0 {
		g.refund(n)
	}
	g.freeKV(l)
	freeRec(l)
	// The block's layer leaves the pool, and with it every session's pages
	// there.
	if g.kvp != nil {
		g.dropKVLayer(g.kvp, li)
	}
	if n := l.recBytes; n > 0 {
		g.refund(n)
		g.RecBytes -= n
		l.recBytes = 0
	}
	for _, b := range l.auxBufs() {
		if b != nil {
			b.Free()
		}
	}
	g.convLeft(l)
	delete(g.layers, li)
	g.layerGen++
	g.placeGen.Add(1)
}

// afterRelease is what every release ends with. Callers hold g.mu.
func (g *devTier) afterRelease() {
	// A pool no session holds a place in, and the seats of sessions that hold
	// none, go back to the budget.
	g.recTidy()
	// The last tower block out takes its set's scratch with it.
	g.dropIdleTower()
	if len(g.layers) == 0 {
		// And a device left with no block and no head holds nothing of its
		// own: the scratches and the staging go too, refunded (the next
		// placement builds and reserves them again).
		if g.bs == nil || g.bs.head == nil {
			g.dropGraph()
			g.dropAllScratch()
		}
	}
	// A decline is cached: resident() keeps a failed entry so a tensor is not
	// re-priced every matvec, and that entry says no forever. Giving budget
	// back is what clears those tombstones, or a seam grown after a shrink is
	// silently refused. Only failures are dropped; a live entry is another
	// block's weights.
	for k, r := range g.res {
		if r != nil && !r.ok {
			delete(g.res, k)
		}
	}
	g.freeStale()
}

// ReserveKV is reserveKV for the zero session: a caller that never attached.
func (g *devTier) ReserveKV(pos int) bool {
	return g.reserveKV(0, pos)
}

// reserveKV gives session sid's sequence the pages pos positions of
// its history need, from outside a submission: growing a layer drops the
// captured graph, which deadlocks CUDA inside a Session. A restored prefix
// arrives with no submission at all.
func (g *devTier) reserveKV(sid uint64, pos int) bool {
	// No scratch is read: the session's pages are all it touches, so it takes
	// no lane.
	g = g.bareView(sid)
	if pos <= 0 {
		return true
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.asking = true
	defer func() { g.asking = false }()
	// No paged history is no attention history here (a tower's keys live for
	// one encode): nothing to reserve.
	if !g.paged {
		return true
	}
	if err := g.pagedHold(sid, pos); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// TrimKV is trimKV for the zero session: a caller that never attached.
func (g *devTier) TrimKV(pos int) bool {
	return g.trimKV(0, pos)
}

// trimKV gives back session sid's pages past pos, and reports whether
// any went. The trimmed pages stay in the pool, for this session's next
// positions or anyone's; kvCompact gives them to the budget when something
// needs the room. They are room all the same, so it is announced.
func (g *devTier) trimKV(sid uint64, pos int) bool {
	g = g.bareView(sid)
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.kvp == nil || !g.kvp.trimSeq(seqID{sid, 0}, pos) {
		return false
	}
	g.roomGen.Add(1)
	return true
}

// dropSession frees everything session sid holds on this device: its history
// and its recurrent state on every block.
func (g *devTier) dropSession(sid uint64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	delete(g.hostFns, sid)
	freed := false
	for li, l := range g.layers {
		freed = g.dropHistory(li, l, sid) || freed
	}
	if freed {
		// A captured launch sequence names the buffers just freed.
		g.dropGraph()
	}
	// Its own recordings name nothing that will be read again either, in
	// whichever lane they were made, and its state on this device goes.
	g.forgetSess(sid)
	// Its seat in the recurrent pools, and a pool it was the last to hold.
	g.recTidy()
	// Its pages are free now and stay in the pool for the next session;
	// kvCompact gives them back when something needs the room. The scratch's
	// bound stays: paged history sizes nothing by context, so resetting it
	// only rebuilt the scratch -- the batched ones with it -- at the next
	// session's first call, inside its prompt: every prompt after a process's
	// first ran about a fifth slower (gpu-kernels.md, "The prompt lost 22%").
}

// dropHistory frees session sid's history on block l -- its KV cache or its
// recurrent pair -- and reports whether it had any. Callers hold g.mu and drop
// the graph when it did.
func (g *devTier) dropHistory(li int, l *layer, sid uint64) bool {
	if l == nil {
		return false
	}
	freed := false
	if kvp := l.kv[sid]; kvp != nil && kvp.paged {
		// Its pages in this layer go and the layer compacts: what the session
		// held here is room again, whoever keeps the block.
		delete(l.kv, sid)
		g.pagedLeave(li, sid, l)
		freed = true
	} else if kvp != nil && kvp.shared {
		delete(l.kv, sid)
		g.transientKVPut()
		freed = true
	} else if kvp != nil {
		if kvp.kc != nil {
			kvp.kc.Free()
		}
		if kvp.vc != nil {
			kvp.vc.Free()
		}
		delete(l.kv, sid)
		g.refund(kvp.bytes)
		l.kvBytes -= min(l.kvBytes, kvp.bytes)
		g.KVBytes -= min(g.KVBytes, kvp.bytes)
		freed = true
	}
	if l.rec[sid] != nil {
		// Its slots in the block's pool are free; recTidy gives the pool, or
		// the slots past the last seat, back to the budget.
		delete(l.rec, sid)
		freed = true
	}
	return freed
}

// MigrateRec is migrateRec for the zero session: a caller that never attached.
func (g *devTier) MigrateRec(li int, conv, state []float32, toDevice bool) bool {
	return g.migrateRec(0, li, conv, state, toDevice)
}

// migrateRec moves block li's recurrent summary between host and device.
// model.Instance.rconv/rstate are what every host consumer reads (including the
// -kv-cache seal), and a placed linear block never touches them, so without this
// a cached restore would resume from a summary that never saw the prompt. It
// reads half cur, the state as of the last completed token, at the session's
// seat in the block's pool.
func (g *devTier) migrateRec(sid uint64, li int, conv, state []float32, toDevice bool) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil {
		return false
	}
	rp := l.recOf(sid)
	if rp == nil {
		// Not a linear block, or no session state: nothing to move is success,
		// exactly as MigrateKV treats a block with no cache.
		return true
	}
	if l.pool == nil || g.recC == 0 {
		return false
	}
	// A batch hands every row's state back to back, row r at r*ConvState and
	// r*StateLen: the seat's own layout, one slot a row.
	rows := len(conv) / g.recC
	if rows < 1 || len(conv) != rows*g.recC || len(state) != rows*g.recS {
		g.LastErr = fmt.Sprintf("MigrateRec: block %d wants a multiple of %d conv and %d state floats, "+
			"the caller offered %d and %d", li, g.recC, g.recS, len(conv), len(state))
		return false
	}
	st := g.recSeats[sid]
	switch {
	case st.rows == rows:
	case st.rows < rows && !toDevice:
		// The seat was never grown to the batch, and it grows at the batch's
		// first step (LayersRows) or before a state is sent up (below): no
		// step has run these rows here, so the host's copy is the current one
		// and there is nothing to read back.
		return true
	case st.rows < rows:
		// Sent up for a batch the seat does not hold yet: grow it first.
		// growSeat discards the session's states, which is right, since a
		// batch's rows have no history on this device before its first step
		// and every row is written next.
		if !g.growSeat(sid, rows) {
			return false
		}
		st = g.recSeats[sid]
	default:
		g.LastErr = fmt.Sprintf("MigrateRec: block %d holds %d row(s), the caller offered %d",
			li, st.rows, rows)
		return false
	}
	sh := rp.cur
	if g.recShared {
		// One persistent state (s[1] is nil): the step reads s[0] whichever
		// half of the conv window is current, as emitLinear does.
		sh = 0
	}
	if !toDevice {
		rd := func(b backend.Buf, v []float32, w int, name string) bool {
			// A buffer reads from its start, so a seat above slot 0 is first
			// copied to a buffer of its own on the device: reading the pool
			// directly would bring every other session's state below it to the CPU
			// too.
			err := func() error {
				if len(v) == 0 {
					// A block with no state of this kind (a short
					// convolution has no delta-rule half): nothing to read,
					// and a zero-byte buffer is no allocation a driver takes.
					return nil
				}
				if st.base == 0 {
					return b.Read(f32b(v))
				}
				tmp, err := g.dev.Alloc(len(v) * 4)
				if err != nil {
					return err
				}
				defer tmp.Free()
				if err := g.dev.Copy(tmp, 0, b, st.base*w*4, len(v)*4); err != nil {
					return err
				}
				return tmp.Read(f32b(v))
			}()
			if err != nil {
				g.LastErr = fmt.Sprintf("MigrateRec: block %d %s: %v", li, name, err)
				return false
			}
			return true
		}
		return rd(l.pool.c[rp.cur], conv, g.recC, "conv") && rd(l.pool.s[sh], state, g.recS, "state")
	}
	// Both halves. Seeding cur alone is already correct (the next submission
	// writes 1-cur before anything reads it); writing both is defence, so no
	// future caller depends on that parity argument.
	for i := 0; i < 2; i++ {
		if err := l.pool.c[i].WriteAt(st.base*g.recC*4, f32b(conv)); err != nil {
			g.LastErr = fmt.Sprintf("MigrateRec: block %d conv half %d: %v", li, i, err)
			return false
		}
		if l.pool.s[i] == nil || len(state) == 0 {
			continue // a shared pool's second half is the device's scratch
		}
		if err := l.pool.s[i].WriteAt(st.base*g.recS*4, f32b(state)); err != nil {
			g.LastErr = fmt.Sprintf("MigrateRec: block %d state half %d: %v", li, i, err)
			return false
		}
	}
	return true
}

// MigrateKV is migrateKV for the zero session: a caller that never attached.
func (g *devTier) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool {
	return g.migrateKV(0, li, k, v, pos, toDevice)
}

// migrateKV moves block li's attention history between host and device.
// Without it a block that changes sides mid-generation attends over whatever
// was in the pages it arrived at: fluent, wrong text.
//
// k and v are the host arrays for this block, laid out [pos*kvDim + i] in
// float32; only positions [0, pos) are moved (pagedMigrate). A block with no
// paged history here -- a tower, whose keys live for one encode -- has
// nothing to move.
func (g *devTier) migrateKV(sid uint64, li int, k, v []float32, pos int, toDevice bool) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	kvp := l.kvOf(sid)
	if kvp == nil || !kvp.paged || pos <= 0 {
		return l != nil
	}
	if err := g.pagedMigrate(sid, li, 0, k, v, pos, toDevice); err != nil {
		g.LastErr = fmt.Sprintf("MigrateKV: block %d: %v", li, err)
		return false
	}
	return true
}

// PrewarmLayer packs a whole block off the main loop. See GPU.Prewarm.
func (g *devTier) PrewarmLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	bank := 1
	if w.Router.Data != nil && p.NExpert > 1 {
		bank = p.NExpert
	}
	_ = bank // the bank is packed as one tall matrix, exactly as PrepLayer does
	// A lazy weight is not prewarmable: this must be safe beside a running
	// Layers, and nn.LayerWeights.Ensure reads the file into the pager's frames.
	// Packing without it would pack whatever the frame holds. The upload calls
	// Ensure itself, so only the prewarm is lost.
	if w.Ensure != nil {
		return false
	}
	any := false
	// weightList, not a second copy of it: a matrix left off would silently
	// never prewarm.
	for _, x := range weightList(w) {
		q, ok := quantOf(x.T)
		if !ok || x.Rows <= 0 || x.K <= 0 || len(x.Data) == 0 {
			continue
		}
		if g.prewarm(li, q, x.Data, x.Rows, x.K) {
			any = true
		}
	}
	return any
}

// weightList is a block's pageable matrices in layer.tensors() order.
//
// One list, for tensors()' reason: prepLayer builds it to decide and upload,
// and pageIn rebuilds it after Ensure re-points the slices. Two copies of the
// order could upload one matrix into another's slot with every size right.
func weightList(w *nn.LayerWeights) []nn.Weight {
	return []nn.Weight{w.Wq, w.Wk, w.Wv, w.Wo, w.Gate, w.Up, w.Down,
		w.ShGate, w.ShUp, w.ShDown,
		w.SSM.Gate, w.SSM.BA, w.SSM.Out,
		w.Wqa, w.Wqb, w.Wkva, w.Wkb, w.Wvb,
		w.SSM.FA, w.SSM.FB, w.SSM.GA, w.SSM.GB,
		w.PLEGate, w.PLEProj,
		w.IdxQB, w.IdxK, w.IdxProj,
		w.LaurelL, w.LaurelR, w.SSM.In,
		w.IdxQ,
		w.WoA, w.CompKV, w.CompGate, w.IdxCompKV, w.IdxCompGate,
		w.MLAGate, w.RoutedDown, w.RoutedUp,
		w.HCAttn.Down, w.HCAttn.Up, w.HCAttn.Inject, w.HCFFN.Down, w.HCFFN.Up, w.HCFFN.Inject,
		w.NGram.Key, w.NGram.Value}
}

// BatchTok is how many token columns one thread of a batched matvec carries.
// A chunk of NTok tokens read one at a time reads the weights NTok times; with
// Tok columns per thread it reads them NTok/Tok times. Larger runs out of
// registers and occupancy. 16 won the sweep once attention was tiled; the best
// width moves with the kernel mix, so re-sweep after changing it.
// BatchShape.Tok overrides it.
const BatchTok = 16

// The batched prefill's tile geometry. Each is the fitted default; a tier's
// WithBatchShape and WithMMATile pins replace it for that tier (tilesFor).
//
// defBatchRowt is how many rows one thread carries. See MatVecShape.Rowt: Tok
// alone plateaus because the kernel issues one activation load per dot4, and a
// tile of rows shares those loads. 4 won by a wide margin in mvbench -ntok 128;
// 8 does not fit its accumulators.
//
// defMMAMT and defMMANT are the warp tile of the matrix-instruction matvec: MT
// blocks of 16 weight rows by NT blocks of 8 token columns. 4x4 beat narrower
// tiles and the dp4a tile in mvbench -mma -ntok 128 (see mmaTile).
//
// defAttnQTile and defAttnKTile are the query rows and key positions one thread
// of the batched scores kernel carries. maxKeyTile bounds the score-row
// padding, so it must not be smaller than the key tile. 4x2 won the end-to-end
// sweep: the query side is a broadcast (widening costs registers) while the key
// side is a strided walk (widening costs coalescing).
//
// defAttnATile is how many query rows one AttnAcc thread carries, so one V load
// serves all of them (see AttnAccTiled); 8 beat 1 end to end.
//
// defAttnNT is how many 8-position key tiles one warp of the tensor-core scores
// kernel carries, so one q fragment serves all of them; 2 beat 1, and 4 is no
// better.
//
// defBatchParts is the RMSNorm partial count for a batched chunk; see
// initScratch.
const (
	defBatchRowt               = 4
	defMMAMT, defMMANT         = 4, 4
	defAttnQTile, defAttnKTile = 4, 2
	defAttnATile               = 8
	defAttnNT                  = 2
	defBatchParts              = 64
)

// tiles is one tier's resolved batched geometry.
type tiles struct {
	rowt, mmaMT, mmaNT, qtile, ktile, atile, attnNT, parts int
}

// tilesFor applies a tier's pins to the defaults.
func tilesFor(kb knobs) tiles {
	t := tiles{rowt: defBatchRowt, mmaMT: defMMAMT, mmaNT: defMMANT, qtile: defAttnQTile,
		ktile: defAttnKTile, atile: defAttnATile, attnNT: defAttnNT, parts: defBatchParts}
	b := kb.batch
	if b.RowT >= 1 {
		t.rowt = b.RowT
	}
	for _, e := range []struct {
		v   int
		dst *int
	}{{b.QTile, &t.qtile}, {b.KTile, &t.ktile}, {b.ATile, &t.atile}, {b.Parts, &t.parts},
		{b.AttnNT, &t.attnNT}} {
		if e.v >= 1 && e.v <= 128 {
			*e.dst = e.v
		}
	}
	if kb.mmaMT > 0 && kb.mmaNT > 0 {
		t.mmaMT, t.mmaNT = kb.mmaMT, kb.mmaNT
	}
	return t
}

// maxKeyTile bounds how far past a row's causal width a kernel may write, so
// the score row is padded by it. The FMA tile writes up to ktile-1 past; the
// matrix kernel up to 8*attnNT-1; sm_70's AttnScoresMMA70 up to
// 32*volta70AttnMT-1.
//
// It must cover the widest tile: an overshoot writes -inf into the next row's
// scores, which softmaxes to NaN. volta70Attn refuses a tile wider than this.
const maxKeyTile = 64

// scoreStride is a block's score row: the context, padded for the batched
// arm -- AttnScoresTiled's last key tile can write up to maxKeyTile-1 past the
// causal width, which unpadded lands in the next row's scores -- and rounded
// to four floats, since AttnAccMMA70 reads a row's probabilities 16 bytes at a
// time and every batched kernel takes this one stride.
func scoreStride(p *nn.LayerPlan, rows int) int {
	if rows > 1 {
		return (p.MaxSeq + maxKeyTile + 3) &^ 3
	}
	return p.MaxSeq
}

// batchWidth is how many token positions one batched submission carries. It has
// to be a multiple of BatchTok, and larger only amortizes the launches -- the
// weight traffic is set by Tok alone.
const batchWidth = nn.MaxDevicePrefillChunk

// prepBatch builds the batched scratch and the composer's prompt forms at
// its width, on the first prompt that asks for one rather than at PrepLayer
// time.
//
// It is lazy because the scratch costs VRAM (mostly score buffers) that a
// decode-only process never uses, and separate because Alloc and Compile cannot
// run inside a Session.
func (g *devTier) prepBatch(width int) bool {
	// 8, not batchGrain: a ragged step on a device without a matrix
	// instruction runs at a multiple of 8 (see LayersRows).
	if g.bs == nil || width < 8 || width%8 != 0 {
		return false
	}
	if g.bbs == nil {
		g.bbs = map[int]*blockScratch{}
	}
	if g.bbs[width] == nil {
		if g.noRoomFor(width) {
			return false
		}
		tok := BatchTok
		if k := g.kb.batch.Tok; k >= 1 && batchWidth%k == 0 {
			tok = k
		}
		for tok > 1 && width%tok != 0 {
			tok /= 2
		}
		bs := g.initScratch(&g.bs.p, width)
		if bs == nil {
			return false
		}
		g.ScratchBuilds++
		bs.tok = tok
		g.bbs[width] = bs
		// A width built after placement filled the card must not take what
		// the blocks were budgeted: it is refused, and the caller runs the
		// reserved width (batchFor) or narrower submissions. Without the
		// reservation the device asks the card, as it did before it. Nor
		// may it take the history admitted rows were promised (PromiseKV).
		// Free pool pages are not a reservation (PrewarmKV grows them into
		// whatever room placement left): they are given back first.
		if !g.NoScratchReserve && !g.kvCompact(g.promisedBytes(), nil) && width != g.promptW && width != g.stepW && !g.reserving {
			g.dropBatch(width)
			g.markNoRoom(width)
			g.ScratchRefused++
			g.LastErr = fmt.Sprintf("a %d-row batched scratch does not fit the budget (%d of %d bytes charged)", width, g.used, g.limit)
			return false
		}
	}
	defer g.scratchWin().close()
	// The layer walk is not guarded by the scratch: ReleaseLayers deletes
	// blocks, so a moved seam brings back blocks whose batched kernels are not
	// compiled, and finding that mid-submission (where Compile is not allowed)
	// would refuse every later batched prompt. The compiled kernels live in
	// g.kerns keyed by shape, which holds every width; batKern is then a map
	// lookup, safe inside a Session.
	bb := g.bbs[width]
	// The walk is ~25 kernel lookups a block, every prompt (0.9 ms of a
	// 512-row Qwen3-0.6B prompt on a Volta-class card); it only has something
	// to find after a block was placed or released.
	if bb.walked == g.layerGen+1 {
		return true
	}
	// The composer's prompt forms at this width and their instances
	// (ComposedPrompt).
	g.promptForms(bb)
	bb.walked = g.layerGen + 1
	return true
}

// batchGrain is the coarsest granularity every batched kernel divides: the
// matrix instruction covers 8 token columns and the shipped tile takes four of
// them, so a chunk width is a multiple of 32. A ragged tail is rounded up to
// this rather than to the full width, which is the difference between 67% and
// 33% waste on a 42-row tail.
const batchGrain = 8 * 4

// promptGrain is the grain a prompt chunk's own width is rounded to: the
// batched forms' batchGrain. Callers hold g.mu.
func (g *devTier) promptGrain() int { return batchGrain }

// unblockedRow names the row of a rows-wide scratch for p that the tier's
// int8 activation blocks (32 wide) do not divide, which its quantizers refuse
// to build, or "".
func unblockedRow(p *nn.LayerPlan, rows int) string {
	const why = "a row of %d (%s) that the tier's int8 activation blocks of 32 do not divide"
	switch {
	case p.KPad == 0 && rows*p.NEmbd%32 != 0:
		return fmt.Sprintf(why, p.NEmbd, "the model's width")
	case rows*p.NFFN%32 != 0:
		return fmt.Sprintf(why, p.NFFN, "the MLP's width")
	}
	return ""
}

// attendsIn reports whether a block of [lo, hi) on this device attends: not
// a linear block's recurrence.
func (g *devTier) attendsIn(lo, hi int) bool {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && !l.linear {
			return true
		}
	}
	return false
}

// capped is the plan with MaxSeq replaced by what this device is actually built
// for. Every kernel build, every allocation and every bound goes through it. It
// also caps the score scratch (rows*NHead*sstride*4 twice), which at a long
// context is larger than the KV cache.
func (g *devTier) capped(p *nn.LayerPlan) nn.LayerPlan {
	q := *p
	// A vision plan's MaxSeq is the largest grid its tower takes, which a
	// dynamic-resolution tower puts far past any picture (65536 patches on
	// HunyuanOCR): the set is built for the rows ReserveRows last sized it to
	// (visrows.go), and a call is one submission of at most that many.
	if q.NonCausal {
		if g.visCap > 0 {
			q.MaxSeq = min(q.MaxSeq, max(g.visCap, g.visKeys))
		}
		return q
	}
	// The device's capacity, not the smaller of the two: every stride and slot
	// is kvCap wide, so a session with a shorter context still needs a cache
	// that wide.
	if g.kvCap > 0 {
		q.MaxSeq = g.kvCap
	}
	return q
}

// ensureKVCap makes the scratch cover the longest context a session has asked
// for (prepLayer raises maxSeqAsked). The history has no capacity -- it is
// paged -- so only the scratch's bound moves, and rebuilding it drops the
// graph, which cannot happen inside a Session (a CUDA deadlock): Layers calls
// this before entering one. Callers must NOT hold g.mu.
func (g *devTier) ensureKVCap(pos int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Per lane: a lane built before the growth still has the old bound,
	// whichever lane grew first.
	if !g.paged || g.bs == nil || g.kvCap >= g.maxSeqAsked && g.bs.p.MaxSeq >= g.maxSeqAsked {
		return true
	}
	g.kvCap = g.maxSeqAsked
	// The recordings made in this lane name the scratch about to go; no
	// other lane's changes.
	g.dropLaneGraph()
	// Every scratch set (gemma4.go) is built at the capacity, not only the
	// one in use: a set left at the old bound refuses the next position.
	home := g.geoCur
	defer g.useGeom(home)
	for _, k := range append([]geoKey{home}, g.geoKeys()...) {
		g.useGeom(k)
		// A non-causal set is built for its run, not for a context.
		if g.bs == nil || g.bs.p.NonCausal {
			continue
		}
		plan := g.bs.p
		plan.MaxSeq = g.maxSeqAsked
		if !g.rebuildScratch(&plan) {
			return false
		}
	}
	// The composer's parts follow the scratch: formed here, at the bound the
	// rebuild took, as a placement forms them, not inside the step.
	g.useGeom(home)
	g.prewarmComposed()
	return true
}

// rebuildScratch replaces every scratch with one built for plan, carrying the
// output projection across.
//
// The head is what a rebuild would otherwise lose: its matvec and its
// hNorm/logits buffers do not depend on the KV capacity or the block shape, but
// freeScratch frees them and a fresh scratch forgets the resident head, leaving
// the model believing the head is placed.
func (g *devTier) rebuildScratch(plan *nn.LayerPlan) bool {
	prev := g.bs
	// Same shape at the same bound is nothing a rebuild would change.
	if prev != nil && sameShape(prev.p, g.capped(plan)) && prev.p.MaxSeq == g.kvCap {
		return true
	}
	fresh := g.initScratch(plan, 1)
	if fresh == nil {
		return false
	}
	moveHead(fresh, prev)
	g.dropScratch(prev)
	for w := range g.bbs {
		g.dropBatch(w)
	}
	g.bs = fresh
	g.shapeChanged()
	// The reserved widths went with the old scratches: built again at the new
	// shape, here, because a capacity growth rebuilds outside any placement.
	g.reserveScratch(plan, 0)
	return true
}

// planUnion widens a to cover b, and reports whether it had to.
//
// Sessions and block kinds can offer one tier different plans that one scratch
// must serve: a hybrid's linear blocks (Recurrent set, AttnOutGate cleared by
// model.offerRange), a session needing a sliding window, a dense lead block
// before mixtures. Each widened field only adds kernels and buffers that the
// narrower blocks never launch.
func planUnion(a *nn.LayerPlan, b *nn.LayerPlan) bool {
	grew := false
	// Sinks and mixture biases only add work a block without them skips (a
	// sinkless block passes a -inf sink; an unbiased one skips the adds), so a
	// scratch built for them serves every block.
	if b.AttnSinks && !a.AttnSinks {
		a.AttnSinks, grew = true, true
	}
	if b.MoEBias && !a.MoEBias {
		a.MoEBias, grew = true, true
	}
	if b.AttnOutGate && !a.AttnOutGate {
		a.AttnOutGate, grew = true, true
	}
	if b.NoPreNorm && !a.NoPreNorm {
		a.NoPreNorm, grew = true, true
	}
	if b.Recurrent.Conv > 0 && a.Recurrent.Conv == 0 {
		a.Recurrent, grew = b.Recurrent, true
	}
	// A window is a mask: a session that never reaches it computes the same
	// thing through the windowed kernels.
	if b.SWAWindow > 0 && a.SWAWindow == 0 {
		a.SWAWindow, grew = b.SWAWindow, true
	}
	// The mixture: DeepSeek's first blocks are dense, so the first block this
	// tier sees may allocate no MoE buffers. A mixture scratch serves a dense
	// block too, since the two use different buffers; the one non-additive
	// kernel is the activation (baked element count), hence bs.actMulE.
	if b.NExpert > 0 && a.NExpert == 0 {
		a.NExpert, a.NExpertUsed, a.NFFNExp = b.NExpert, b.NExpertUsed, b.NFFNExp
		a.ExpertSigmoid, a.ExpertSelBias = b.ExpertSigmoid, b.ExpertSelBias
		a.ExpertGroups, a.ExpertGroupsUsed = b.ExpertGroups, b.ExpertGroupsUsed
		a.ExpertScale, a.NoExpertNorm = b.ExpertScale, b.NoExpertNorm
		a.ExpertSparseMixer = b.ExpertSparseMixer
		grew = true
	}
	// The shared expert and the dense FFN are widths, so the union is the max:
	// a DeepSeek mixture block carries no dense FFN at all and the dense lead
	// carries no experts, so neither alone sizes the scratch.
	if b.NFFNShExp > a.NFFNShExp {
		a.NFFNShExp, grew = b.NFFNShExp, true
	}
	if b.NFFN > a.NFFN {
		a.NFFN, grew = b.NFFN, true
	}
	// Kimi-K3: the streams, the banks and the checkpoints are per block, and
	// the scratch builds a mix for every bank and a placement for every
	// checkpoint (k3.go); the latent width is the mixture blocks', which the
	// dense lead does not carry.
	if b.K3 != nil && (a.K3 == nil || b.K3.Latent > a.K3.Latent) {
		k := *b.K3
		if a.K3 != nil {
			k = *a.K3
			k.Latent = b.K3.Latent
		}
		a.K3, grew = &k, true
	}
	// Gemma 3n's sparse lead: the gaussian's kernels serve the blocks that ask.
	if b.Sparse && !a.Sparse {
		a.Sparse, grew = true, true
	}
	// A block that is its mixer alone says nothing about the FFN the others
	// run, gated or not.
	if a.NoFFN && !b.NoFFN {
		a.NoFFN, a.UngatedFFN, grew = false, b.UngatedFFN, true
	}
	return grew
}

// planConflict names a plan this tier's scratch cannot serve alongside the one
// it holds, or "". Two different windows are the case: the scratch bakes one,
// and taking either would run the other session's local layers wrong.
func planConflict(bs *blockScratch, p *nn.LayerPlan) string {
	if bs == nil || p.NonCausal {
		return ""
	}
	if a, b := bs.p.SWAWindow, p.SWAWindow; a > 0 && b > 0 && a != b {
		return fmt.Sprintf("a sliding window of %d keys on a tier built for %d", b, a)
	}
	// Llama 4's attention is baked the same way (chunk, temperature table, norm,
	// input weighting), and planUnion widens none of it.
	if a, b := bs.p, *p; a.SWAChunked != b.SWAChunked || a.NoPEGlobal != b.NoPEGlobal ||
		a.QKL2Norm != b.QKL2Norm || a.ExpertWeightIn != b.ExpertWeightIn ||
		a.AttnTempScale != b.AttnTempScale || a.AttnTempFloor != b.AttnTempFloor ||
		a.AttnTempOffset != b.AttnTempOffset {
		return "Llama 4 attention of a different shape from the one this tier was built for"
	}
	// The activation kernels bake whether the FFN has a gate.
	if a, b := bs.p, *p; !a.NoFFN && !b.NoFFN && ungatedFFN(&a) != ungatedFFN(&b) {
		return "an ungated FFN beside a gated one on one tier"
	}
	// Where q's norm runs is baked into the scratch's launch sequence.
	if a, b := bs.p, *p; a.QKNorm != b.QKNorm || a.QKNormPost != b.QKNormPost {
		return fmt.Sprintf("a q/k norm (%v, after the rotary %v) on a tier built for (%v, %v)",
			b.QKNorm, b.QKNormPost, a.QKNorm, a.QKNormPost)
	}
	// The latent geometry is baked into every kernel of the scratch. No
	// architecture mixes them, which is why it is named rather than assumed.
	if a, b := bs.p, *p; a.KVLoraRank != b.KVLoraRank || a.QLoraRank != b.QLoraRank ||
		a.HeadDimV != b.HeadDimV {
		if a.MLA() || b.MLA() {
			return fmt.Sprintf("a latent attention geometry of (%d, %d, %d) on a tier "+
				"built for (%d, %d, %d)", b.KVLoraRank, b.QLoraRank, b.HeadDimV,
				a.KVLoraRank, a.QLoraRank, a.HeadDimV)
		}
	}
	// The classic block's switches (C6) are baked too: norm kernels, gating,
	// and the parallel residual read from the scratch's plan.
	if a, b := bs.p, *p; a.LayerNorm != b.LayerNorm || a.Parallel != b.Parallel ||
		a.UngatedFFN != b.UngatedFFN {
		return fmt.Sprintf("a classic block (LayerNorm %v, parallel %v, ungated %v) on a tier "+
			"built for (%v, %v, %v)", b.LayerNorm, b.Parallel, b.UngatedFFN,
			a.LayerNorm, a.Parallel, a.UngatedFFN)
	}
	// The score scale is baked into the attention kernels and the residual
	// scale into the scratch's one-element buffer: a model of another scale on
	// this scratch would run with this one's.
	if a, b := bs.p.Scale(), p.Scale(); a != b {
		return fmt.Sprintf("an attention score scale of %g on a tier built for %g", b, a)
	}
	if a, b := bs.p.ResidualScale, p.ResidualScale; residScaled(&bs.p) || residScaled(p) {
		if a != b {
			return fmt.Sprintf("a residual scale of %g on a tier built for %g", b, a)
		}
	}
	// The clamp's bound is baked into its kernel, and a scratch built without
	// one has neither the kernels nor the buffers.
	if bs.p.ClampKQV != p.ClampKQV {
		return fmt.Sprintf("a q/k/v clamp of %g on a tier built for %g", p.ClampKQV, bs.p.ClampKQV)
	}
	// The delta-rule kernel bakes one key-head pairing; a scratch built for
	// grouped heads runs a tiled block's value heads against the wrong keys,
	// and every shape still matches.
	if a, b := bs.p.Recurrent, p.Recurrent; a.Conv > 0 && b.Conv > 0 && a.KeyTiled != b.KeyTiled {
		return fmt.Sprintf("a delta rule with key heads tiled=%v on a tier built for tiled=%v",
			b.KeyTiled, a.KeyTiled)
	}
	return ""
}

// nonCausalConflict names a non-causal block the non-causal set, built from
// the first such block's plan, cannot run, or "". planConflict passes every
// non-causal plan, since a tower's blocks are one shape; an encoder with a
// decision head after it (Laya) is two. A block's own MLP width, gating and
// activation, and whether it rotates, are its own (layer.nffn, the set grown
// to the widest MLP); what the set's kernels bake for every block is the
// heads and their width, the norm kind, the rotary's width and pairing, a
// gate buffer and the copy where a norm is missing, so a block differing in
// one of those would run on the first one's kernels and is declined by name.
// A convolutional tower block runs its own program and is not compared.
func nonCausalConflict(bs *blockScratch, p *nn.LayerPlan) string {
	if bs == nil || !p.NonCausal || p.Conv != nil || bs.p.Conv != nil {
		return ""
	}
	a := &bs.p
	gated := map[bool]string{true: "ungated", false: "gated"}
	switch {
	case a.NEmbd != p.NEmbd || a.NHead != p.NHead || a.NKVHead != p.NKVHead || a.HeadDim != p.HeadDim:
		return fmt.Sprintf("a non-causal block of %d heads of %d (%d kv) over %d on a set built for %d of %d (%d kv) over %d",
			p.NHead, p.HeadDim, p.NKVHead, p.NEmbd, a.NHead, a.HeadDim, a.NKVHead, a.NEmbd)
	case !ungatedFFN(p) && ungatedFFN(a):
		// The set of an ungated MLP has no gate buffer; a block's own width,
		// gating and activation are otherwise its own (layer.nffn).
		return fmt.Sprintf("a non-causal block's MLP (%s %s, %d wide) on a set built for (%s %s, %d wide)",
			gated[ungatedFFN(p)], p.Act, p.NFFN, gated[ungatedFFN(a)], a.Act, a.NFFN)
	case p.LayerNorm != a.LayerNorm:
		return fmt.Sprintf("a non-causal block with LayerNorm %v on a set built with %v", p.LayerNorm, a.LayerNorm)
	case p.NRot > 0 && !p.NoPosEnc && (a.NRot == 0 || a.NoPosEnc):
		// A set that rotates nothing builds no rotary kernels; a block that
		// rotates nothing runs on one that does (layer.noRope).
		return fmt.Sprintf("a non-causal block rotating %d of each head on a set built to rotate none", p.NRot)
	case p.NRot > 0 && a.NRot > 0 && (p.NRot != a.NRot || p.RopeNeox != a.RopeNeox):
		return fmt.Sprintf("a non-causal block rotating %d (NEOX %v) on a set built for %d (NEOX %v)",
			p.NRot, p.RopeNeox, a.NRot, a.RopeNeox)
	case p.NoPreNorm && !a.NoPreNorm:
		return "a non-causal block with no pre-norm on a set built for blocks that have one"
	}
	return ""
}

// kvCacheBytes is one session's K and V cache for one block at capacity c.
func (g *devTier) kvCacheBytes(p *nn.LayerPlan, c int) (cache, vcache int) {
	cache = (c + 1) * p.KVRow() * 4
	// MLA has no V buffer: one row per position, the value its prefix.
	if p.MLA() {
		return cache, 0
	}
	vcache = cache
	if g.KVF16 {
		vcache = cache / 2
	}
	return cache, vcache
}

// restride copies rows rows of cols floats from src, srcStride apart, to dst,
// dstStride apart: one device copy where the run is contiguous, else a row a
// copy, run as one list. Every block of a growth has the same shape.
func (g *devTier) restride(src, dst backend.Buf, rows, cols, srcStride, dstStride int) bool {
	var err error
	if rows == 1 || cols == srcStride && cols == dstStride {
		err = g.dev.Copy(dst, 0, src, 0, rows*cols*4)
	} else {
		cs := g.gatherCopies[:0]
		for r := range rows {
			cs = append(cs, backend.CopyRange{Dst: dst, DstOff: r * dstStride * 4, Src: src, SrcOff: r * srcStride * 4, N: cols * 4})
		}
		g.gatherCopies = cs
		err = backend.CopyAll(g.dev, cs)
		clear(cs)
	}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// initKVCap sets the starting capacity the first time this device sees a plan:
// one page, or the whole context when that is smaller.
func (g *devTier) initKVCap(p *nn.LayerPlan) {
	if g.kvCap != 0 || p.NonCausal {
		// A tower carries no growing history, so it must not be what sets the
		// ceiling the text model grows toward.
		return
	}
	g.maxSeqAsked = p.MaxSeq
	// Paged history has no capacity to grow: no kernel bakes one, so the
	// scratch is built for the whole context once and every session shares it.
	g.paged, g.kvCap = true, p.MaxSeq
}

// foldReqBuf holds the decode scratch's request for the folded head's row
// take (servedRowTake), once.
func (g *devTier) foldReqBuf(hb *blockScratch) error {
	if hb.foldReq != nil {
		return nil
	}
	w := g.scratchWin() // the scratch's own words (scratch.go)
	defer w.close()
	b, err := g.dev.Alloc(4 * (composer.RequestRows + 1))
	if err != nil {
		return err
	}
	hb.foldReq = b
	return nil
}

// foldWords is the folded head's request: the row a chunk's head reads.
func (g *devTier) foldWords(row int) []byte {
	b := g.subBytes.fold[:0]
	for range composer.RequestRows {
		b = binary.LittleEndian.AppendUint32(b, 0)
	}
	b = binary.LittleEndian.AppendUint32(b, uint32(row))
	g.subBytes.fold = b
	return b
}
