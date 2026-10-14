package tier

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

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A whole transformer block on the device.
//
// The point is one Session per block with exactly one Read: serving matvecs one
// at a time costs a blocking round trip each, which alone was a quarter of a
// gemma token. That is also why every op between the matvecs is a kernel -- a
// single host-side RMSNorm would drag the residual back across the bus.

// layer is one block's device-resident state.
type layer struct {
	wq, wk, wv, wo *resident
	gate, up, down *resident
	// conv is a convolutional tower block's program and vectors (conv.go),
	// nil for every other block.
	conv *convLayer
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
	laurelL, laurelR                                           *resident
	mvLL, mvLR                                                 mv
	nLaurel                                                    backend.Buf
	sparse                                                     bool
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
	// KDA's two low-rank gates, nil on qwen3next.
	ssmFA, ssmFB, ssmGA, ssmGB *resident
	mvSG, mvBA, mvSO           mv
	mvFA, mvFB, mvGA, mvGB     mv
	ssmConv1d, ssmA, ssmDt     backend.Buf
	ssmNorm, ssmOnes           backend.Buf
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
	woAW                              map[int]mv
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
	// position and no state identity, so two model.States sharing one pair would
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
	// A non-causal block's own MLP and rotary, which may differ from the
	// first block's the set was built from: an encoder's decision head
	// (Laya's) is an ungated ReLU MLP wider than the encoder's gated GELU,
	// and rotates nothing where the encoder rotates every head. nffn is its
	// width, ungated whether it has no gate, actK its activation where the
	// set's is another (nil: the set's bs.actMul), noRope that it rotates
	// nothing. Read on the non-causal path only.
	nffn    int
	ungated bool
	act     kernels.ActKind
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
	// qkv is q, k and v in one launch (kernels.MatVecSegments), each still
	// writing its own buffer; nil where fuseQKV declined. qkvBlocks and qkvW
	// are its grid.
	qkv             backend.Kernel
	qkvBlocks, qkvW int
	qkvBias         [3]bool // which segments the kernel takes a bias for
	// qkvIn is qkv normalising and quantizing its own operand (fuseActIn).
	qkvIn backend.Kernel
	// actInKeys names the operand-prologue site each of q/k/v, the attention
	// output and down took (fuseActIn), for tuneActIn to group blocks by.
	actInKeys [4]string
	// mlaQK is MLA's query projection (q, or q_a under a q LoRA) and kv_a in
	// one launch, and shGU the shared expert's gate and up: two matvecs over
	// one activation each, built by fuseQKV like qkv. nil where declined.
	mlaQK, shGU *segLaunch
	// gu is a dense gated FFN's gate and up in one launch (Config.GateUpFuse,
	// fuseGateUp), and guDown the down projection forming act(gate)*up from
	// their rows and quantizing it in its prologue, with the residual add:
	// three launches where there were four. nil where declined.
	gu            *segLaunch
	guDown        backend.Kernel
	mvo, mvg, mvu mv
	mvd           mv
	// MoE only: the F32 router matrix and its matvec. gate/up/down above hold
	// the banks in that case, and their mv carries slots > 0.
	router   backend.Buf
	mvRouter backend.Kernel
	// routerF16 is whether l.router holds binary16 (Mixtral) rather than f32;
	// the batched mixture picks its router kernel by it.
	routerF16 bool
	// gpt-oss's per-head attention sinks and its router and expert biases, nil
	// for every other architecture (LayerPlan.AttnSinks, MoEBias).
	sinks, routerB             backend.Buf
	expGateB, expUpB, expDownB backend.Buf
	// moeGate, moeUp and moeDown are the expert matvecs with their epilogues
	// fused (fuseMoE): the gate with its expert bias, up with its bias AND the
	// gated activation (writing act(gate)*up itself), down with its bias. nil
	// where the plain kernel and the separate launches run instead.
	moeGate, moeUp, moeDown backend.Kernel
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
// host. The price is a host round trip mid-block: the selection must come home
// before anyone knows which sheets to send, so the submission suspends after
// ExpertRank and resumes after the upload, and such a block cannot be captured
// into a graph (a recording has nowhere to put a readback).
type streamBank struct {
	// sh is the per-expert sheet size of each plane, in bytes, for gate, up and
	// down in that order: [matrix][qs, d, sc].
	sh [3][3]int
	// sel is the selection, read home each token. It is len NExpertUsed and is
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
	// nExpert is the bank's expert count, so a cache can be set up after
	// placement (GPU.sizeAutoCaches).
	nExpert int
	// The cross-layer prefetch (prefetch): the experts it reads, whether a
	// read is in flight, its join and its error.
	pfWant []uint32
	pfOn   bool
	pfWG   sync.WaitGroup
	pfErr  error
	// host is the block's routed experts on the host (hybrid execution), and
	// hin, hw and hout its input, weights and output, reused every token.
	host          func(sel []uint32, w, in, out []float32) error
	hybrid        bool // the experts run on the host (runHost), not sent
	li            int  // the block, for the session's host side (hostFor)
	hin, hw, hout []float32
	hsel          []uint32
	// pred is this block's selection as the previous block's probe predicted
	// it (Config.StreamProbe), valid while havePred; score compares it with
	// the real one and clears it.
	pred     []uint32
	havePred bool
}

// score charges this token's selection to the probe's counters and hands it
// to Config.StreamSelLog. Both are measurement: a run without either pays a
// branch.
func (st *streamBank) score(g *devTier, li int) {
	if g.StreamSelLog != nil {
		g.StreamSelLog(li, st.sel)
	}
	if !st.havePred {
		return
	}
	st.havePred = false
	for _, e := range st.sel {
		for _, q := range st.pred {
			if q == e {
				g.ProbeHits++
				break
			}
		}
	}
	g.ProbeExperts += len(st.sel)
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
	// The previous block's prefetch of this one is joined first: its reads
	// are this fill's reads when the prediction was right, and a frame still
	// being read is the eviction hazard above.
	if st.pfOn {
		tw := time.Now()
		st.pfWG.Wait()
		st.pfOn = false
		g.TStreamPrefetchWait += time.Since(tw)
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

// prefetch starts reading this block's predicted experts into host frames
// while the block before it finishes (Config.StreamPrefetch). It reads and
// never uploads: a wrong guess costs disk bytes, not link bytes or cache
// slots, and the fill that follows still reads, maps and sends exactly the
// experts the router chose. An expert the cache already holds is not read.
func (st *streamBank) prefetch(g *devTier) {
	if st.pre == nil || st.pfOn {
		return
	}
	st.pfWant = st.pfWant[:0]
	for _, e := range st.pred {
		if st.cached() && int(e) < len(st.slotOf) && st.slotOf[e] >= 0 {
			continue
		}
		st.pfWant = append(st.pfWant, e)
	}
	if len(st.pfWant) == 0 {
		return
	}
	g.StreamPrefetched += len(st.pfWant)
	st.pfOn = true
	st.pfWG.Add(1)
	go st.prefetchRun()
}

// prefetchRun is prefetch's read on its own goroutine. Its error is kept
// and not returned: a failed guess is not a failed token, and the fill's own
// read of anything it needs reports a real failure.
func (st *streamBank) prefetchRun() {
	defer st.pfWG.Done()
	st.pfErr = st.pre(st.pfWant)
}

// runHostRows is runHost for a chunk of rows: the selections (row stride
// k+1, ExpertRank's layout), the weights (stride k) and the rows of in come
// home, the host runs each row's experts, and the rows of the sum go back.
func (st *streamBank) runHostRows(sid uint64, g *devTier, s backend.Session, sel, w, in, out backend.Buf, n, k, rows int) error {
	if cap(st.hin) < rows*n {
		st.hin = make([]float32, rows*n)
		st.hout = make([]float32, rows*n)
	}
	st.hin, st.hout = st.hin[:rows*n], st.hout[:rows*n]
	if cap(st.hw) < rows*k {
		st.hw = make([]float32, rows*k)
	}
	st.hw = st.hw[:rows*k]
	if cap(st.hsel) < rows*(k+1) {
		st.hsel = make([]uint32, rows*(k+1))
	}
	st.hsel = st.hsel[:rows*(k+1)]
	t0 := time.Now()
	if err := s.Sync(); err != nil {
		return err
	}
	for _, rd := range [3]struct {
		b backend.Buf
		p []byte
	}{{in, f32b(st.hin)}, {w, f32b(st.hw)}, {sel, u32b(st.hsel)}} {
		if err := s.Read(rd.b, rd.p); err != nil {
			return err
		}
	}
	if hr := g.hostFns[sid][st.li].rows; hr != nil {
		// Expert-major: each chosen expert read once for the chunk.
		if err := hr(st.hsel, st.hw, st.hin, st.hout, rows, k); err != nil {
			return err
		}
		err := s.WriteAt(out, 0, f32b(st.hout))
		g.THybrid += time.Since(t0)
		g.HybridRows += rows
		return err
	}
	host := g.hostFor(sid, st)
	if host == nil {
		return fmt.Errorf("tier: this session offered no host side for a hybrid block")
	}
	for r := 0; r < rows; r++ {
		if err := host(st.hsel[r*(k+1):r*(k+1)+k], st.hw[r*k:(r+1)*k],
			st.hin[r*n:(r+1)*n], st.hout[r*n:(r+1)*n]); err != nil {
			return err
		}
	}
	err := s.WriteAt(out, 0, f32b(st.hout))
	g.THybrid += time.Since(t0)
	g.HybridRows += rows
	return err
}

// hostFor is session sid's host side for st's block, nil when this
// session never offered one. Never the placing session's: that State may be
// closed, and its generated code with it. Callers hold g.mu.
func (g *devTier) hostFor(sid uint64, st *streamBank) func(sel []uint32, w, in, out []float32) error {
	return g.hostFns[sid][st.li].one
}

// hostSide is a session's host side for one hybrid block: a row at a time,
// and a chunk's rows expert-major.
type hostSide struct {
	one  func(sel []uint32, w, in, out []float32) error
	rows func(sel []uint32, w, in, out []float32, rows, k int) error
}

// runHost is the hybrid suspension: in (the experts' input, n wide -- the
// expert width, which is the latent on Kimi-K3) and the k routing weights
// come home, the host's experts sum into hout, and the sum is
// written to out, where the combine kernel would have put it. The Sync makes
// the launches that write in land first.
func (st *streamBank) runHost(sid uint64, g *devTier, s backend.Session, in, w, out backend.Buf, n, k int) error {
	if cap(st.hin) < n {
		st.hin = make([]float32, n)
		st.hout = make([]float32, n)
	}
	st.hin, st.hout = st.hin[:n], st.hout[:n]
	if cap(st.hw) < k+1 {
		st.hw = make([]float32, k+1)
	}
	st.hw = st.hw[:k+1]
	t0 := time.Now()
	if err := s.Sync(); err != nil {
		return err
	}
	if err := s.Read(in, f32b(st.hin)); err != nil {
		return err
	}
	if err := s.Read(w, f32b(st.hw)); err != nil {
		return err
	}
	host := g.hostFor(sid, st)
	if host == nil {
		return fmt.Errorf("tier: this session offered no host side for a hybrid block")
	}
	if err := host(st.sel, st.hw[:k], st.hin, st.hout); err != nil {
		return err
	}
	err := s.WriteAt(out, 0, f32b(st.hout))
	g.THybrid += time.Since(t0)
	return err
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
	// Nor one whose experts carry biases (gpt-oss): the bias kernels index
	// them by the selection, which the cache rewrites as slots. A second
	// buffer of true ids would lift it.
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
		pq, pd, psc, err := kernels.PackedWords(q, x.Rows/slots, x.K)
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
		if pair.r == nil || !pair.r.ok || pair.w.Packed == nil {
			return fmt.Errorf("tier: streamed bank %d has no host side", m)
		}
		pk := pair.w.Packed
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
				if pk.Sheet != nil {
					sp, rel, err := pk.Sheet(int(e))
					release = rel
					if err != nil {
						release()
						return err
					}
					src = [3][]byte{sp.QS, sp.D, sp.SC}[pl]
				} else {
					whole := [3][]byte{pk.QS, pk.D, pk.SC}[pl]
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

// mv is a matvec's compiled kernel plus the split it was chosen for. alt is the
// same matvec at the split the fitted table would have picked, kept so
// tuneSplit's answer can be checked in a paired A/B. Codegen is microseconds, so
// a second kernel is free.
// mvTail is a batched twin's second launch (mv.tail).
type mvTail struct {
	kern    backend.Kernel
	threads int
	width   int
}

type mv struct {
	kern backend.Kernel
	// tail is a second launch of the batched twin over the tokens the first
	// one's block does not divide (tileMV); it takes the same parameters.
	tail *mvTail
	red  backend.Kernel
	rows int
	// slots is 0 for an ordinary matvec and NExpertUsed for an indexed one, in
	// which case the launch is rows*slots*split threads and the kernel takes a
	// pSel parameter after pOut.
	slots int
	split int
	// bias is the per-row addend, nil for every projection that has none. The
	// kernel declares pBias after pOut, so the launch appends it.
	bias backend.Buf
	alt  *mv
	// q/k are the shape this was built from, so the batched twin can be
	// generated later without re-reading the weight.
	q kernels.Quant
	k int
	// groups is NTok/Tok: how many token columns one thread carries, expressed
	// as the number of thread groups the launch needs. Zero on a decode matvec.
	groups int
	// threads overrides rows*groups when the launch is not one thread per
	// output row, as for the matrix-instruction kernel (a warp per tile).
	threads int
	// width is a staged GEMM's workgroup (its tile's threads) where that is
	// not 128; zero is 128.
	width int
	// rowt is how many rows one decode thread carries, so the grid is
	// rows/rowt rather than rows. Zero and one both mean one row per thread.
	rowt int
	// group says the split reduces inside the threadgroup, so this matvec
	// writes its final row directly: no partial plane and no red kernel.
	group bool
	// experts and slotAct are an indexed matvec's bank size and whether each
	// slot reads its own activation -- what retuneIndexed rebuilds it from.
	experts int
	slotAct bool
	// res is this projection's kernel with a bias slot, cached like every
	// matvec kernel; the tier passes the residual there so the projection
	// writes x + W*a and the separate Add launch goes. nil where the weight
	// has a bias of its own, or is not the attention output or FFN down.
	res backend.Kernel
	// gated is the Gate variant of the FFN up projection: it writes
	// act(gate)*up into the activation buffer itself, so the gated FFN's
	// ActMul launch goes. nil elsewhere, and on a split whose last pass is a
	// reduce (which would have to apply it instead).
	gated backend.Kernel
	// resIn, gatedIn and kernIn are res, gated and kern building their own
	// operand (kernels.ActIn) from the float row: pIn [pNW] in pA/pAX's
	// place. Built by fuseActIn; nil where the block cannot take them.
	resIn, gatedIn, kernIn backend.Kernel
	// resActIn is res forming act(gate)*up and quantizing it in its prologue
	// (kernels.ActIn.Gated), for an FFN whose up projection has no gated form.
	resActIn backend.Kernel
	// wantGated marks the FFN up projection that takes a gated variant
	// whenever its split allows one, so a retune to an in-group split builds
	// it even where the first split had none.
	wantGated bool
	// redN is how many outputs red sums on a batched twin that splits k: every
	// row of every token column (ragMV).
	redN int
	// act is set on sm_70's tensor-core twin (voltaMV): the kernel that writes
	// the float activation as binary16 into g.f16Buf before kern reads it, over
	// actN threads.
	act  backend.Kernel
	actN int
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

// value is the buffer the accumulate reads. Under MLA it is the key buffer: a
// latent cache row is [latent | rotary key] and the value is its leading
// KVLoraRank floats, so storing them apart would write the same floats twice.
// engine/model/kvpage.go's latent flag is the host's half of the same
// aliasing. vc is left nil rather than set to kc so free walks never
// double-free.
func (p *kvPair) value() backend.Buf {
	if p.vc != nil {
		return p.vc
	}
	return p.kc
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
	// perRun is the scan's threads for one run. The kernels are compiled for
	// as many runs as rows and launched for the step's runs alone: a run's
	// state is in registers for all its rows, and the entries past the last
	// run repeat it (kernels' run descriptor), so a one-sequence step of four
	// rows launched for four runs stepped its whole chain four times over.
	perRun int
	// lanes is how many lanes the scan puts on one state row; 1 is its
	// one-thread form (Stats.LinearRowsScalar).
	lanes int
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
// sequences' states and compiles the copy that moves them home, charging the
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
	if _, err := g.slotCopy(g.recS, rows); err != nil {
		return fail(err)
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
	// headShared says head, mvHead, hNorm, hNormB and headCap are another
	// scratch's -- lane0's, borrowed by a clone (shareHead) -- and are not
	// freed or refunded with this one.
	headShared bool
	// walked is g.layerGen+1 at prepBatch's last successful layer walk for
	// this scratch: until a block is placed or released, the walk would find
	// what it found then (0: never walked).
	walked uint64
	p      nn.LayerPlan
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
	// mg is a batched scratch's mixture half, or nil (see moegroup.go).
	mg *moeGroup
	// mlab is a batched scratch's latent-attention half, or nil (mlabatch.go).
	mlab       *mlaBatch
	scoresMMA  backend.Kernel // the tensor-core scores kernel, or nil
	scoresWarp bool           // scores is MLA's warp-per-pair kernel: NHead*nCap groups of 32
	// scoresW and scoresMMAW are the same two kernels for a sliding-window
	// layer (LayerPlan.Window), built only when the plan has a window. The mask
	// is in the scores alone; softmax and accumulate are shared. See
	// kernels.windowed for why that is exact.
	// flashV is the decode attention's compiled key partitionings (flash.go),
	// sharing one partial buffer sized for the widest.
	flashV []flashVariant
	// pkv is the paged attention set; nil unless this scratch's history is
	// paged (devTier.pagedPlan), and then the staged, tiled and flash
	// kernels above that read a contiguous cache are not built.
	pkv                               *pagedScratch
	flashPart                         backend.Buf
	flashWidth                        int // the decode attention's workgroup
	flashChunk, flashForced, flashCap int // flashSplitFor's inputs
	// ds4 is DeepSeek V4's set (ds4.go), nil on every other model.
	ds4 *ds4Scratch
	// q4e is Qwen3.8-Flash-Next's set (q4e.go), nil on every other model.
	q4e *q4eScratch
	// k3 is Kimi-K3's set (k3.go), nil on every other model.
	k3 *k3Scratch

	scoresW, scoresMMAW backend.Kernel
	// scores70/scores70W/acc70 are the batched attention on sm_70's m8n8k4
	// (kernels.AttnScoresMMA70/AttnAccMMA70), for a device whose m16n8k16 f16
	// scores kernel does not lower. acc70 walks 32-row groups, so atile is 32.
	scores70, scores70W, acc70 backend.Kernel
	// mla70 says a batched latent block's attention runs on sm_70's m8n8k4
	// (mlabatch.go): its accumulate walks 32-row groups too, so the softmax
	// this scratch builds takes atile 32 for it (mla70Attn).
	mla70 bool
	// fp70/fp70W are the same attention as one kernel (kernels.FlashPrefill70),
	// the second for a sliding-window layer; nil where it does not take the
	// shape, and the three above run instead.
	fp70, fp70W backend.Kernel
	// fpT/fpTW are that one kernel on a collective matrix unit
	// (kernels.FlashPrefillTile, Metal), blocked by fpTile.
	fpT, fpTW backend.Kernel
	fpTile    kernels.FlashTile
	// The ragged-decode set (rows.go): attention whose rows are different
	// sequences, a per-row softmax, the head's batched twin and a per-row argmax.
	ragArgmax         backend.Kernel
	ragLogits, ragTok backend.Buf
	ragVocab          int // the rows of ragLogits a row holds
	ragHead           mv
	sstride           int // the score row stride; padded at rows > 1
	// arows is how many query rows one pass of the attention kernels takes:
	// rows, except for a vision block whose score planes would be too large,
	// which runs its attention in query chunks (visionattn.go). qc and xc are a
	// chunk's queries and output, gathered from and scattered to q and xb at
	// qoffs[c], by qGather and xScatter.
	arows             int
	qc, xc            backend.Buf
	qoffs             []backend.Buf
	qGather, xScatter backend.Kernel
	parts             int
	x, x2, h, mvOut   backend.Buf
	part, a, ax       backend.Buf
	qr, q, k, v       backend.Buf
	roff              backend.Buf // RoPE-q destination offsets, one per row
	kpos              backend.Buf // RoPE-k destination offsets: positions when K is transposed
	hNorm, logits     backend.Buf
	// headCap and logitsCap are the head's final softcap (nn.Head.Softcap):
	// the capped logits land in their own buffer, since no kernel may write
	// what it reads. Both nil on every model without one.
	headCap   backend.Kernel
	logitsCap backend.Buf
	// ragCapK and ragCap are the same for the per-row head: ragLogits capped
	// into a buffer of their own, which the per-row argmax and the readback
	// take. ragCapC is the cap ragCapK was built for.
	ragCapK backend.Kernel
	ragCap  backend.Buf
	ragCapC float32
	// headEmbeds and embScale are nn.Head.Embeds and EmbdScale: the resident
	// head is also the token embedding, and EmbedRows gathers from it. embK is
	// GetRowsPacked at embW rows, with its id and output buffers; all three are
	// made on first use.
	headEmbeds     bool
	embScale       float32
	embK           backend.Kernel
	embW           int
	embIds, embOut backend.Buf
	// spanK and spanOff move a batched chunk's last residual row into this
	// (decode) scratch's x, so the output projection can ride the chunk's own
	// submission (layersOnce's folded head). Made on first use.
	spanK         backend.Kernel
	spanOff       backend.Buf
	hNormB        backend.Buf // a LayerNorm head's bias (C6)
	head          *resident
	mvHead        mv
	hRaw          []byte
	hOut          []float32
	att, prob, xb backend.Buf
	// attCap holds the capped scores for an architecture with an attention
	// softcap (gemma2): out of place, since no kernel may write what it reads.
	attCap  backend.Buf
	softcap backend.Kernel
	// attWin holds a windowed vision block's masked scores (WindowMaskRows)
	// for one attention pass, vwins each pass's rows' windows as lo, hi pairs
	// (one buffer per query chunk, visionattn.go) and vwinRows how many rows
	// the windows cover (setWindows). Built on the first windows a non-causal
	// call hands this scratch; nil on every other.
	attWin   backend.Buf
	vwins    []backend.Buf
	winMask  backend.Kernel
	vwinRows int
	// vsegs is the packed sequences' runs as each pass's rows' windows,
	// which every unwindowed non-causal block masks with (SetSeqRuns), and
	// vsegRows the rows they cover; 0 is one sequence and no mask.
	vsegs    []backend.Buf
	vsegRows int
	// DBRX's clamp on q, k and v (LayerPlan.ClampKQV), out of place for the
	// same reason: the clamped copies are what everything after the
	// projections reads. nil for every other architecture.
	qClamp, kClamp, vClamp backend.Buf
	// Gemma 4's clipped linears (nn.LayerPlan.Clamps): clampK[2s] clamps
	// matrix s's input and clampK[2s+1] its output (q, k, v, o, gate, up,
	// down), out of place; cin and cout are a clamped input and output, and
	// gClamp/uClamp the clamped gate and up the activation reads. nil for
	// every other plan.
	clampK                    [14]backend.Kernel
	cin, cout, gClamp, uClamp backend.Buf
	clampQ, clampKV           backend.Kernel
	g, u, act                 backend.Buf
	cs, n, koff, zero         backend.Buf
	// csSWA is the local layers' rotary table, allocated only for an
	// architecture that trains two bases (LayerPlan.SWAPeriod). nil elsewhere,
	// which is what keeps ropeTable a nil check for every other model.
	csSWA backend.Buf
	// The rotary table is built on the device: ropeTab and ropeTabSWA are the
	// per-model folded planes (nn.Rope.TabPlanes) and ropeConst the constant
	// block, uploaded once; rpos is the R positions, the only input rewritten per
	// token. ropeTable is nil wherever the table is uploaded instead (a vision
	// block, a refused compile, or Config.RopeTableHost).
	ropeTab, ropeTabSWA    backend.Buf
	ropeConst, rpos        backend.Buf
	ropeTable              backend.Kernel
	ropeNPairs             int
	quantE, quantF, quantQ backend.Kernel
	// quantAct is ActMul and quantF in one launch (kernels.QuantizeAct), nil
	// for an ungated FFN or an activation ActMul has no gated form of.
	quantAct            backend.Kernel
	normPart, normApply backend.Kernel
	// A vision block normalises with LayerNorm, which is three kernels: the mean,
	// the variance about it, then the apply. The passes cannot fuse because
	// E[x^2]-E[x]^2 cancels in f32 on a ViT residual whose mean dwarfs its
	// variance (see kernels.LayerNormPart). normPart is the first pass either way
	// (sums of squares for RMSNorm, the mean for LayerNorm); normVar is
	// LayerNorm's second, nil otherwise.
	normVar backend.Kernel
	part2   backend.Buf // LayerNorm's variance partials, nil otherwise
	// copyK puts k in the cache for a block with no rotary, in the transposed
	// layout RoPERowsT would have written. nil unless NonCausal with no rotary.
	copyK, copyQ             backend.Kernel
	ropeQ, ropeK, copyV      backend.Kernel
	scores, softmax, attnAcc backend.Kernel
	// qNorm/kNorm normalise each head of q and k before RoPE; nil unless the
	// architecture has them. qn/kn are their out-of-place destinations, which
	// the IR requires (AGENTS.md RULE 13: no kernel reads a buffer it writes).
	qNorm, kNorm backend.Kernel
	// Llama 4's weightless q/k norm rides qNorm/kNorm with a weight of ones,
	// before the rotary rather than after: a weightless norm commutes with a
	// rotation, so the device keeps its one norm site. l4Ones is that
	// HeadDim-wide weight; nil unless QKL2Norm.
	l4Ones backend.Buf
	// qTemp scales an unrotated layer's query by Llama 4's temperature, looked
	// up in tempTab by the row's causal count; nil unless AttnTempScale.
	qTemp   backend.Kernel
	tempTab backend.Buf
	// qt is a rotated layer's query scaled by mistral3's temperature, which
	// applies on every layer (NoPEGlobal unset); nil otherwise.
	qt backend.Buf
	// copyRows copies rows of the residual into h where a block has no
	// pre-norm (nn.LayerPlan.NoPreNorm); nil otherwise.
	copyRows backend.Kernel
	// actMulW is ActMul for a mixture that weights the expert's input, and
	// wOnes the unit weights its down projections are then combined at.
	actMulW backend.Kernel
	wOnes   backend.Buf
	// qkPart/qkApply replace them for a wide q/k norm (olmoe: one RMSNorm over
	// all of q), where HeadNorm's single 32-lane group per vector is several
	// times slower than the block norm's partitioned pair. One pair serves q and
	// k, so it is built only where they are the same width: a GQA OLMo 2 takes
	// HeadNormRows.
	qkPart, qkApply backend.Kernel
	// qkRms is that wide norm as one whole-row RMSNorm launch per vector (the
	// kernel bs.rms is), two launches a layer where the pair took four.
	qkRms backend.Kernel
	// rms and addRms are the whole RMSNorm in one launch of R groups of
	// kernels.RMSNormGroup -- rms for the norm closure, addRms for a residual
	// add followed by the FFN norm. nil for a LayerNorm (vision) block.
	rms, addRms backend.Kernel
	// rmsQ and addRmsQ are rms and addRms that also write the quantized
	// activation quantE would (kernels.RMSNormQuantRows): one launch where the
	// norm and the quantize were two. nil where they cannot be built.
	rmsQ, addRmsQ backend.Kernel
	// subQA and subQF are BitNet's inner norms fused with the quantize wo and
	// down read (kernels.RMSNormQuantRows at qdim and NFFN, the row window),
	// writing the normed rows to sub. nil elsewhere.
	subQA, subQF backend.Kernel
	sub          backend.Buf
	qn, kn       backend.Buf
	// krot is XD-RoPE's k turned row-major, before its norm (RopeSplit): the
	// whole head, since XD-RoPE turns every dimension (model.build refuses a
	// partial one).
	krot, kroff backend.Buf
	ropeKRow    backend.Kernel
	// Gemma 4's: vn is v after its weightless norm (hdOnes is the weight),
	// and one is the 1.0 a block's output scalar copies back with. nil
	// elsewhere.
	vn, hdOnes, one backend.Buf
	// Gemma 4's per-layer embedding (nn.LayerPlan.PLEDim): the rows' inputs,
	// staged per submission, a block's gate, its slice of the inputs, their
	// product and the projection back. nil elsewhere.
	pleIn, pleG, pleS, pleA, pleO backend.Buf
	pleSlice, pleAct, pleQuant    backend.Kernel
	// Gemma 3n's AltUp and LAuReL (nn.LayerPlan.AltUp, altup.go): the
	// router's modalities, every stream's prediction, the corrected streams,
	// LAuReL's branch and its rank-wide middle; a sparse block's gate
	// statistics and its gaussian top-k (gParts partials a row). nil elsewhere.
	altM, altPred, altB, altLaur, altLT           backend.Buf
	gPartB, gVarB, gg                             backend.Buf
	altRoute, altPredictK, altCorrectK, altFinish backend.Kernel
	altMul, altCopy, laurelJoin, laurelQuant      backend.Kernel
	gPart, gVar, gApply                           backend.Kernel
	gParts                                        int
	// Gemma 4's mixture block (nn.LayerPlan.DenseMoE): hr is the router's
	// input rows, rwS the routed weights times each expert's factor
	// (ewScale). nil elsewhere.
	hr, rwS backend.Buf
	ewScale backend.Kernel
	// qwen3next's attention out-gate. The q projection is double width and its
	// second half, interleaved per head, gates the attention output through a
	// sigmoid before the o projection. splitQG deinterleaves bs.q into qg and
	// ogate; sigMul writes sigma(ogate)*xb into ogated, out of place (RULE 13).
	splitQG, sigMul   backend.Kernel
	qg, ogate, ogated backend.Buf
	// qwen3next's shared expert, whose gate is one logit per row. shsum is a
	// separate destination because the fold reads the routed sum in bs.mvOut
	// (RULE 13).
	shRouterK, shActMul, shQuant, shAdd backend.Kernel
	// routeWidth is bs.route's workgroup: one warp for kernels.ExpertRouteWarp,
	// ExpertRouteWidth(NExpert) for kernels.ExpertRoute.
	routeWidth                             int
	shg, shu, shact, shout, shlogit, shsum backend.Buf
	nShExp                                 int
	// A linear block's half of the submission. Every one is out of place, which
	// is what makes the recurrence legal: the conv reads the old window and
	// writes a new one, and the delta rule reads the old state and writes the
	// other half of its recPair. dMixed is the fused q|k|v the conv reads and
	// dConv what it writes, separate for the same reason.
	convRows, convShift, ssmSiLU backend.Kernel
	splitGates, deltaGate        backend.Kernel
	qkNorm, qkScale, deltaStep   backend.Kernel
	outNorm, outActMul           backend.Kernel
	sliceQ, sliceK, sliceV       backend.Kernel
	dMixed, dConv, dZ, dBA       backend.Buf
	// quantD quantizes the gated delta output for the SSM output projection.
	// It is a separate kernel from quantF because Quantize bakes its element
	// count and VHeads*VDim is not NFFN.
	quantD                       backend.Kernel
	dBRaw, dAlpha, dDecay, dBeta backend.Buf
	// KDA's extras: the rank-r temporary its two low-rank gates project
	// through with its own quantized activation, and the half of DeltaGate's
	// output pair each of its two launches is not for.
	dLowRank, dLRa, dLRax, dGateWaste backend.Buf
	quantLR, deltaGateChan            backend.Kernel
	deltaStepChan, outSigMul          backend.Kernel
	// deltaScan is the gated delta rule over a whole chunk in one launch
	// (kernels.GatedDeltaScan): every token of the chunk steps the state,
	// which GatedDeltaStep -- one token -- cannot. scanThreads is its grid.
	deltaScan   backend.Kernel
	scanThreads int
	// deltaFused is deltaScan with everything from the convolution's output to
	// the rule folded in (kernels.GatedDeltaFused); built where the device has
	// a subgroup, and then the eleven kernels it replaces are not launched.
	deltaFused backend.Kernel
	// ssdScan is Mamba-2's whole update at this scratch's rows
	// (kernels.GatedDeltaFused with SSD), from the convolution's raw output
	// and the dt projection to the output and the next state, at any row
	// count and any lanes; ssdNorm its grouped gated norm (ssdNormL the lanes
	// it was built at), and copyRes the copy a block with no FFN ends with,
	// moving the residual home from bs.x2.
	ssdScan, ssdNorm, copyRes backend.Kernel
	ssdNormL                  int
	// Mamba-1's (nn.RecurrentPlan.Mamba1): the convolution's bias and SiLU,
	// dt's quantization over the rank, the dt and B/C norms (built at
	// m1NormDtL and m1NormBCL lanes) and the whole scan; its buffers are dt's
	// normed bottleneck and B and C before and after their norms.
	m1Bias, m1Quant, m1NormDt, m1NormBC, m1Scan backend.Kernel
	m1NormDtL, m1NormBCL                        int
	m1LowN, m1B, m1C, m1BN, m1CN                backend.Buf
	// side is a block's mixer output where the block runs attention beside it
	// (Falcon-H1): the two halves are summed into bs.h before the residual.
	side backend.Buf
	// LFM2's short convolution: its two gates' rows (scB, scC), B*x (scBx),
	// C times the convolution (scY) and the plain product that forms both
	// (scMul, kernels.ActMul with the identity).
	scB, scC, scBx, scY backend.Buf
	scMul               backend.Kernel
	// ragLin is a ragged step's linear block per row count (LayersRows): the
	// convolution and the fused rule over the step's runs, each a sequence's
	// rows at consecutive positions (kernels' run forms), built for the real
	// row count so a padded row never reaches a state.
	ragLin               map[int]*ragLinear
	dQ, dQn, dK, dKn, dV backend.Buf
	dOut, dOutN, dAct    backend.Buf
	dRows, dScale        backend.Buf
	rec                  nn.RecurrentPlan
	// qkLanes is how many threads cooperate on one head in those two kernels:
	// 32 where the device guarantees a subgroup that wide, 1 where it does not
	// (devTier.pickLanes, as for the softmax). qkWidth is the launch width -- the
	// group is the subgroup at 32, a plain 64-wide grid at 1.
	qkLanes, qkWidth int
	// qNormL, kNormL, qkNormL and outNormL are the lanes each head norm was
	// built at, which is not always qkLanes: headLanes drops a head that is not
	// a whole number of subgroups to the scalar twin. The launch geometry has
	// to follow the kernel (headNormGeom), not the device.
	qNormL, kNormL, qkNormL, outNormL int
	// Multi-head latent attention's half of the submission, every buffer out of
	// place (RULE 13). The cache row is lat+rot wide, the absorbed query is that
	// same width per head, and the attention result is lat wide per head, so the
	// score and accumulate kernels are built for two widths over one buffer (see
	// kvPair.value).
	//
	// mlaAbsRaw is the absorb's slot-major output (NHead x lat) and mlaAbs the
	// row-wide query the scores read (NHead x row): an indexed matvec writes slot
	// j at j*Rows and cannot use stride row, so the copies interleave the rotary
	// halves in.
	mlaSliceNope, mlaSlicePE, mlaSliceKPE backend.Kernel
	mlaRopeQ, mlaRopeK                    backend.Kernel
	mlaQuantNope, mlaQuantAcc             backend.Kernel
	mlaQuantOut, mlaQuantQA               backend.Kernel
	mlaNormQAPart, mlaNormQAApply         backend.Kernel
	mlaNormKVPart, mlaNormKVApply         backend.Kernel
	mlaCopyAbs, mlaCopyPE, mlaCopyLat     backend.Kernel
	// mlaNoPEQ/mlaNoPEK stand in for mlaRopeQ/mlaRopeK on a NoPE model: the
	// same writes, unrotated. See nn.LayerPlan.NoPosEnc.
	mlaNoPEQ, mlaNoPEK                backend.Kernel
	mlaQA, mlaQAn, mlaQAPart          backend.Buf
	mlaKV, mlaKVn, mlaKVPart          backend.Buf
	mlaNope, mlaPE, mlaPErot, mlaKPE  backend.Buf
	mlaAbsRaw, mlaAbs, mlaAcc, mlaOut backend.Buf
	// mlaAbsOff and mlaPEOff are the per-head destinations inside mlaAbs --
	// h*row and h*row+lat -- and are written once, because they are geometry.
	// mlaPOff is the position's rotary-key slot in the cache and moves per token
	// exactly as bs.koff does.
	mlaAbsOff, mlaPEOff, mlaPOff backend.Buf
	// mlaIdent is 0..NHead-1: the "selection" both absorb banks are indexed
	// with, since every head runs on every token. It is bs.ident's twin and is a
	// separate buffer because a mixture's is NExpertUsed long.
	mlaIdent               backend.Buf
	mlaParts, mlaQAParts   int
	mlaRow, mlaLat, mlaRot int
	mlaNopeW               int
	// attnRed reduces attnAcc's partial sums when accSplit > 1.
	attnRed               backend.Kernel
	accSplit              int
	actMul, actMulE, addE backend.Kernel
	// addS is the residual add of a scaled block output, x + s*out, and
	// rscale the one-element s it reads (Granite's residual_scale); both nil
	// where the plan carries no residual scale (residScaled).
	addS   backend.Kernel
	rscale backend.Buf
	// scaleX multiplies a block's output row by its scalar (Gemma 4's
	// layer_scalar), out of place: x to h by the block's own scalar, h back
	// to x by bs.one. nil where the plan has no split geometry.
	scaleX backend.Kernel
	// MoE. eg/eu/eact are Slots x NFFNExp, edown is Slots x NEmbd, and rlogits
	// is NExpert wide. rsel/rtop are NExpertUsed+1 long: ExpertRank's spare
	// slot is where every unselected thread stores, since the IR has no
	// conditional store.
	rank, weights, combine  backend.Kernel
	route                   backend.Kernel // rank+weights in one launch, the plain softmax route
	quantX                  backend.Kernel
	rlogits, rsel, rtop, rw backend.Buf
	// rwProbe is the cross-layer probe's routing weights, kept off rw, which
	// holds the block's own until its combine reads them.
	rwProbe backend.Buf
	// The V3 family's grouped selection: gscore writes one score per expert
	// group into rgs, gmask writes the masked selection plane into rbm, and
	// rank then counts over rbm instead of over the logits. Both are nil on
	// every ungrouped router, which is every mixture but DeepSeek-V3's.
	gscore, gmask backend.Kernel
	rgs, rbm      backend.Buf
	// MoEBias only: the biased router logits and the three expert outputs with
	// their experts' biases added, each out of place (RULE 13).
	rlogitsB, egB, euB, edownB backend.Buf
	rbias, ebiasFF, ebiasD     backend.Kernel
	// noSink is NHead of -inf: the sink a block without sinks passes to a
	// scratch whose softmax takes one, since exp(-inf) adds nothing. sinkProbe
	// is the probe's own sink row (softmaxAgreesAt).
	noSink, sinkProbe backend.Buf
	// ident is 0, 1, .. NExpertUsed-1: the selection a compact bank is indexed
	// with, since its k sheets are already the k the router chose and in the
	// order it chose them. Written once here rather than per token, and nil on
	// a device with no streamed block.
	ident                   backend.Buf
	eg, eu, eact, edown     backend.Buf
	nExpert, nUsed, nFFNExp int
	hx, hcs, hcsSWA         []float32
	hraw                    []byte
	hout                    []float32
	// The softmax launch geometry, which is the one kernel whose shape is not
	// "one thread per output": 32 lanes per head when the device carries a
	// subgroup shuffle, one thread per head when it does not.
	smGroups, smWidth int
	// softmaxAlt is the thread-per-head kernel kept alongside the warp one, so
	// GPU.ScalarSoftmax can alternate them in one process on one State. It is nil
	// when the warp kernel did not win the check.
	softmaxAlt              backend.Kernel
	smAltGroups, smAltWidth int
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

// declineReason names the graph feature this tier cannot run, or "" when it can
// run the block's shape. A string rather than a bool so the reason survives: a
// full card, a missing kernel and an unknown graph want opposite responses.
func declineReason(p *nn.LayerPlan) string {
	// The router's gating shape, declined by name: a kernel gate on ExpertRank
	// proves the kernel, not that the tier asked for the right gating function.
	if p.NExpert > 0 {
		if r := routerWhyNot(p); r != "" {
			return r
		}
	}
	// MLA runs; what is left is a shape check (a tier that does not implement
	// MLA still declines on KVLoraRank alone).
	if r := mlaWhyNot(p); r != "" {
		return r
	}
	if r := llama4WhyNot(p); r != "" {
		return r
	}
	if r := gemma4WhyNot(p); r != "" {
		return r
	}
	if r := msaWhyNot(p); r != "" {
		return r
	}
	if r := ds4WhyNot(p); r != "" {
		return r
	}
	if r := k3WhyNot(p); r != "" {
		return r
	}
	if r := q4eWhyNot(p); r != "" {
		return r
	}
	switch {
	case p.NonCausal && (p.Parallel || p.Recurrent.Conv > 0 || p.MLA() || p.NExpert > 0 || p.AttnSinks):
		// A non-causal call is one submission of the block body's plain
		// attention over its runs; these features are built only on the causal
		// paths, and no non-causal model has them.
		return "a non-causal block with a parallel residual, a recurrence, latent attention, experts or sinks"
	case p.Windowed && !p.NonCausal:
		return "windowed key runs on a causal block"
	case p.PostResidNorm && (!p.NonCausal || !p.NoPreNorm || !p.LayerNorm):
		// The norms after the residual adds are wired into the non-causal
		// block (BERT's encoder): its attention reads the residual through
		// the no-pre-norm copy, and its norms are LayerNorms.
		return "post-residual norms on a block other than a non-causal LayerNorm one with no pre-norm"
	case p.Clamps && (!p.NonCausal || p.ClampKQV != 0 || p.AttnOutGate || ungatedFFN(p)):
		// The clipped linears are wired into the non-causal block's plain
		// q/k/v, its output projection and a gated MLP: Gemma 4's tower, the
		// one block that has them.
		return "clipped linears on a block other than a non-causal one with a gated MLP"
	case p.ClampKQV != 0 && (p.AttnOutGate || p.MLA()):
		// The clamp is wired into the plain q/k/v path; an out-gate splits the
		// double-width q first and MLA projects through its own sequence. No
		// architecture combines them.
		return "a q/k/v clamp with an attention out-gate or latent attention"
	case p.Recurrent.Conv > 0 && p.Recurrent.KHeads == 0 && !p.Recurrent.ShortConv:
		// The gated delta rule runs (recPair gives every kernel an out-of-place
		// state); what is left is a shape check.
		return "a linear-attention block with no key heads"
	case p.Recurrent.Mamba1 && (p.Recurrent.Rank <= 0 || p.Recurrent.Rank%32 != 0):
		// dt_proj reads a quantized activation of rank width, and
		// kernels.Quantize works in 32-element blocks.
		return fmt.Sprintf("a Mamba-1 dt rank of %d (the activation quantizer works in 32-element blocks)",
			p.Recurrent.Rank)
	case p.Recurrent.WithAttn && (!p.Recurrent.SSD || p.MLA() || p.NoFFN):
		// The sum is built for Falcon-H1's shape alone: Mamba-2 beside plain
		// attention, then an FFN.
		return "attention beside a recurrent mixer in a shape no architecture has"
	case p.Recurrent.SSD && (p.Recurrent.ChanDecay || p.Recurrent.KeyTiled):
		// Mamba-2's update has one decay a head and grouped keys; neither
		// switch exists on any Mamba-2, and the fused form refuses both.
		return "Mamba-2's update with a per-channel decay or tiled keys"
	case p.Recurrent.SSD && !p.Recurrent.NoNorm && (p.Recurrent.NormGroups <= 0 ||
		(p.Recurrent.VHeads*p.Recurrent.VDim)%p.Recurrent.NormGroups != 0):
		return fmt.Sprintf("a Mamba-2 gated norm of %d groups over %d channels",
			p.Recurrent.NormGroups, p.Recurrent.VHeads*p.Recurrent.VDim)
	case p.Recurrent.Conv > 0 && p.Recurrent.ChanDecay && p.Recurrent.KDim%32 != 0:
		// KDA's second low-rank matvec reads a quantized activation of rank
		// width, the rank is KDim, and kernels.Quantize works in 32-element
		// blocks. Declined by name rather than failing inside prepLayer.
		return fmt.Sprintf("a rank-%d low-rank gate (the activation quantizer "+
			"works in 32-element blocks)", p.Recurrent.KDim)
	case !actHasKernel(p):
		// Every activation the host bakes has a device kernel (quick-GELU
		// for CLIP towers, swiglu-oai for gpt-oss); what is left is a kind in a
		// shape it does not exist in -- a gated kind in an ungated block.
		return "the " + p.Act.String() + " activation in a " + map[bool]string{true: "ungated", false: "gated"}[ungatedFFN(p)] + " block"
	}
	return ""
}

// residScaled reports a plan whose block outputs are scaled before their
// residual adds (Granite's residual_scale). Such a block takes neither the
// projection's fused residual (mvres: the bias slot has no scale) nor the
// add-and-norm launches: the output lands in bs.mvOut as for a post-norm, and
// bs.addS adds s times it. A mixture's routed weights carry no part of it on
// the device, unlike the host's routed scale (model.Config.routedScale): the
// scaled add covers the routed sum and the shared expert at once.
func residScaled(p *nn.LayerPlan) bool { return p.ResidualScale != 0 && p.ResidualScale != 1 }

// scoreScale is the attention score multiplier every kernel of p's attention
// bakes: the plan's (nn.LayerPlan.Scale), which Granite's attention.scale and
// YaRN's magnitude correction replace, or 1/sqrt(HeadDim) under
// ScaleFaultAttn.
func (g *devTier) scoreScale(p *nn.LayerPlan) float32 {
	if g.kb.scaleFault == ScaleFaultAttn {
		return float32(1 / math.Sqrt(float64(p.HeadDim)))
	}
	return float32(p.Scale())
}

// llama4WhyNot names the Llama 4 attention or mixture feature this tier cannot
// run, or "". Each is fluent when skipped (a sliding mask is exact inside
// the first chunk, a rotated NoPE layer at position 0), so the block goes home
// by name rather than running wrong (RULE 8a). What remains are combinations no
// model ships: an input-weighted mixture with expert biases, and a chunk on an
// MLA or vision block. A temperature on every layer (mistral3's form) runs:
// the rotated layers scale q into qt before the rotary.
func llama4WhyNot(p *nn.LayerPlan) string {
	switch {
	case p.AttnTempScale != 0 && (p.AttnTempFloor < 1 || p.AttnTempOffset < 0 ||
		p.AttnTempFloor != float32(int(p.AttnTempFloor)) || p.AttnTempOffset != float32(int(p.AttnTempOffset))):
		return fmt.Sprintf("an attention temperature floor %g / offset %g that is not a whole count",
			p.AttnTempFloor, p.AttnTempOffset)
	case p.ExpertWeightIn && p.MoEBias:
		return "a routed weight on the input of a biased expert"
	case p.ExpertWeightIn && p.NExpert > 0 && p.Act != kernels.ActSiLU && p.Act != kernels.ActGELU:
		return "a routed weight on the input of a " + p.Act.String() + " expert"
	case (p.SWAChunked || p.NoPEGlobal || p.QKL2Norm) && (p.MLA() || p.NonCausal):
		return "Llama 4's attention on a latent or non-causal block"
	}
	return ""
}

// l4TempTab is Llama 4's attention temperature as a table over the integer
// k = floor((pos+off)/floor): entry k is 1 + s*ln(1+k). It covers every
// position a plan of maxSeq can reach, so the kernel's clamp never bites.
func l4TempTab(p *nn.LayerPlan) []float32 {
	fl, off := int(p.AttnTempFloor), int(p.AttnTempOffset)
	n := (max(p.MaxSeq, 1)-1+off)/fl + 1
	out := make([]float32, n)
	for k := range out {
		out[k] = float32(1 + float64(p.AttnTempScale)*math.Log(1+float64(k)))
	}
	return out
}

// windowArg is the window a windowed score kernel is built with: the sliding
// width, or kernels.ChunkWindow of it where the layers attend within a chunk.
func windowArg(p *nn.LayerPlan) int {
	if p.SWAChunked {
		return kernels.ChunkWindow(p.SWAWindow)
	}
	return p.SWAWindow
}

// routerWhyNot names the mixture gating this tier cannot express, or "". The
// answer comes from the kernels (kernels.MoERouteWhyNot), so a missing gating
// feature is a device kernel owed, not a refused capability.
func routerWhyNot(p *nn.LayerPlan) string {
	if p.NExpert == 0 {
		return ""
	}
	return kernels.MoERouteWhyNot(routeOf(p))
}

// routeOf is the plan's router as the kernels describe one: one translation,
// so the decline and the kernel cannot disagree.
func routeOf(p *nn.LayerPlan) kernels.MoERoute {
	// DeepSeek V4 selects through a per-row plane on every block (ds4.go):
	// its selection bias, or a hash block's table.
	ds4 := p.DS4 != nil
	return kernels.MoERoute{
		NExpert: p.NExpert, K: p.NExpertUsed,
		Sigmoid: p.ExpertSigmoid, Bias: p.ExpertSelBias || ds4,
		SqrtSoftplus: ds4 && p.DS4.ExpertSqrtSoftplus, RowBias: ds4,
		NGroup: p.ExpertGroups, NGroupUsed: p.ExpertGroupsUsed,
		Norm: !p.NoExpertNorm, Scale: float32(p.ExpertScale),
		SparseMixer: p.ExpertSparseMixer,
	}
}

// subFor is the activation sub-block a matvec of this format and width needs,
// or 0 for the format's own. It is float-only: a quantized format's scales are
// laid out against its sub-block (see kernels.MatVecShape.Sub), while a float
// weight has no scale plane. What reaches this is MLA's absorbed banks, whose k
// can be below 32 on a small fixture; the sub-block narrows to the width
// itself.
func subFor(q kernels.Quant, k int) int {
	if !kernels.IsFloat(q) || k >= 32 || k <= 0 || k%4 != 0 {
		return 0
	}
	return k
}

// mlaWhyNot names the MLA shape this tier cannot express, or "". Each is a
// width, so each is a kernel owed rather than a refusal (RULE 8): the
// 32-element granule of kernels.Quantize and of the norm's partial reduction.
// The batched arm is refused separately, at submission (see layersOnce), so a
// correct decode stays on the device.
func mlaWhyNot(p *nn.LayerPlan) string {
	if !p.MLA() {
		return ""
	}
	if p.IdxHeads != 0 && p.NoPosEnc {
		return "DeepSeek Sparse Attention's lightning indexer on a model with no rotary"
	}
	nope := p.HeadDim - p.NRot
	if nope <= 0 {
		return fmt.Sprintf("multi-head latent attention with %d rotary dimensions of a %d-wide "+
			"key head, which leaves no nope half to absorb W_k into", p.NRot, p.HeadDim)
	}
	if p.NRot%2 != 0 || p.NRot == 0 {
		return fmt.Sprintf("multi-head latent attention with %d rotary dimensions, which the "+
			"rotation needs to be an even non-zero count", p.NRot)
	}
	for _, v := range []struct {
		n    int
		what string
	}{
		{p.NHead * nope, "the compacted nope half of the query"},
		{p.NHead * p.KVLoraRank, "the per-head attention result"},
		{p.NHead * p.HeadDimV, "the un-absorbed attention output"},
		{p.QLoraRank, "the query latent"},
	} {
		// QLoraRank is 0 on the one-step query, which is not a shape at all.
		if v.n != 0 && v.n%32 != 0 {
			return fmt.Sprintf("multi-head latent attention where %s is %d elements: the "+
				"activation quantizer bakes a multiple of 32 and this tier has no narrower one",
				v.what, v.n)
		}
	}
	return ""
}

// lnBlock reports whether the block's norms are LayerNorms: a vision tower's,
// or a classic text block's (C6, nn.LayerPlan.LayerNorm). Every other block
// is RMSNorm, whose fused norm+quantize kernels a LayerNorm cannot use.
func lnBlock(p *nn.LayerPlan) bool { return p.LayerNorm }

// ungatedFFN reports whether the block's FFN has no gate matrix: up, the
// activation alone, down -- a vision tower's MLP, or a classic text block's
// (C6: phi-2, starcoder, falcon, nemotron).
func ungatedFFN(p *nn.LayerPlan) bool { return p.UngatedFFN }

// actHasKernel reports whether kernels.Act (an ungated FFN) or kernels.ActMul
// (every gated FFN) bakes the plan's activation.
func actHasKernel(p *nn.LayerPlan) bool {
	// A block with no FFN runs no activation.
	if p.NoFFN {
		return true
	}
	set := kernels.Gated[:]
	if ungatedFFN(p) {
		// xIELU reads its numbers from a buffer (kernels.XIELU), so it is in
		// no baked set.
		if p.Act == kernels.ActXIELU {
			return true
		}
		set = kernels.Ungated[:]
	}
	for _, k := range set {
		if k == p.Act {
			return true
		}
	}
	return false
}

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
	// The host side of a hybrid block is the offering session's: a placed
	// block is shared by every session that attaches, and each runs its experts
	// on its own State (runHost looks it up by the call's session). Recorded
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
		g.hostFns[sid][li] = hostSide{one: w.HostExperts, rows: w.HostExpertsRows}
		g.mu.Unlock()
	}
	// Shapes this tier cannot express are declined first and by name (RULE 8a):
	// each would run wrong if skipped, so the block goes to the host whole. They
	// are checked before any allocation so a decline takes no VRAM from the next
	// block.
	if p.Conv != nil {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.prepConv(li, p, w, mayPage)
	}
	if r := declineReason(p); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	// A row range of a packed tensor cannot be expressed: the device layout
	// interleaves rows, resident() uploads QS/D/SC whole, and MatVecShape has no
	// row base, so the block runs on the host instead of reading the wrong rows.
	if r := declineWeights(w); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	// The q/k norm is launched when the block carries its weights; the plan's
	// flag builds the kernels. The model offers the weights by the same
	// derivation that sets the flag (jlm.QKNormOn), so a block where the two
	// disagree is a contract violation, refused rather than run without its norm.
	if attn := !p.NonCausal && (p.Recurrent.Conv == 0 || p.Recurrent.WithAttn); attn && p.QKNorm != (w.QNorm != nil) {
		g.LastErr = fmt.Sprintf("block %d: the plan's q/k norm is %v and the block offers %d norm "+
			"weights", li, p.QKNorm, len(w.QNorm))
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
		// The softmax's subgroup width is a property of the device, which the
		// one-row scratch probes and every batched one inherits (pickSoftmax).
		// A device whose first block is a tower's has built no one-row
		// scratch, and a batched scratch built unprobed takes the
		// one-thread-per-head softmax: SmolVLM's tower ran 2.6x slower on CUDA
		// for it. So the probe runs here, on a one-row scratch of this plan
		// that is dropped again.
		if g.SoftmaxLanes == 0 {
			if probe := g.initScratch(sp, 1); probe != nil {
				g.dropScratch(probe)
			}
		}
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
	} else if want := g.bs.p; p.NonCausal && sp.NFFN > want.NFFN {
		// A non-causal block with a wider MLP than the set was built for (a
		// decision head after its encoder): the set grows to it, and every
		// block runs its own width inside it (layer.nffn).
		want.NFFN, want.MaxSeq = sp.NFFN, g.visMax
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
	l := &layer{nonCausal: p.NonCausal, windowed: p.Windowed, geo: geoOf(p), vFromK: p.VFromK}
	if p.NonCausal {
		l.nffn, l.ungated, l.act = p.NFFN, ungatedFFN(p), p.Act
		l.noRope = p.NRot == 0 || p.NoPosEnc
		if !g.ownActKern(l) {
			return false
		}
	}
	mkw := func(slot int, x nn.Weight) *resident {
		q, ok := quantOf(x.T)
		// A float format's block is the activation granule, not a file structure,
		// so a k below it is a width (see subFor); the k-quants keep the check.
		if !ok || x.Rows <= 0 || len(x.Data) == 0 {
			return nil
		}
		if x.K%q.Elems() != 0 && subFor(q, x.K) == 0 {
			return nil
		}
		return g.resident(li, slot, q, x.Data, x.Rows, x.K, x.Packed)
	}
	build := func(x nn.Weight, split int, bias bool) (mv, bool) {
		q, ok := quantOf(x.T)
		if !ok {
			return mv{}, false
		}
		// The decode row tile: its rows are adjacent, so a lane loads Rowt
		// contiguous u32 instead of one. It composes with the split (the tile buys
		// bytes per lane, the split buys threads). The row count must divide.
		rowt := 1
		if g.DecodeRowt > 1 && x.Rows%g.DecodeRowt == 0 {
			rowt = g.DecodeRowt
		}
		// Counted, so "it made no difference" and "it never ran" can be told
		// apart.
		if rowt > 1 {
			g.RowtTiles++
			if split > 1 {
				g.RowtSplitTiles++
			}
		} else {
			g.RowtPlain++
		}
		// The in-group split reduction, where tuneSplit measured it faster. It
		// writes the final row, so no reduction kernel and no partial plane; see
		// kernels.MatVecShape.GroupSplit.
		grp := !g.NoGroupSplit && split > 1 && rowt == 1 &&
			(g.ForceGroupSplit || g.groupSplit[splitKey{q, x.Rows, x.K}]) &&
			kernels.GroupSplitOK(x.Rows, split)
		kern := g.kernelMode(q, x.K, x.Rows, split, rowt, bias, grp)
		if kern == nil {
			// A refused tile is not a refused weight: fall back to one row.
			rowt, grp = 1, false
			if kern = g.kernel(q, x.K, x.Rows, split, 1, bias); kern == nil {
				return mv{}, false
			}
		}
		var red backend.Kernel
		if split > 1 && !grp {
			if red = g.reduceKernel(x.Rows, split); red == nil {
				split = 1
				if kern = g.kernel(q, x.K, x.Rows, 1, rowt, bias); kern == nil {
					return mv{}, false
				}
			}
		}
		if !grp && !g.sizePart(x.Rows*split) {
			return mv{}, false
		}
		return mv{kern: kern, red: red, rows: x.Rows, split: split, q: q, k: x.K,
			rowt: rowt, group: grp}, true
	}
	// bias is threaded into every kernel: a 6-buffer kernel launched with 7 is
	// refused on Vulkan and silently drops the bias on CUDA, which does not
	// check arity.
	mkk := func(x nn.Weight, r *resident, bias bool) (mv, bool) {
		q, ok := quantOf(x.T)
		if !ok {
			return mv{}, false
		}
		m, ok := build(x, g.tuneSplit(q, x.Rows, x.K, r), bias)
		if !ok {
			return mv{}, false
		}
		if tbl := chooseSplit(x.Rows, x.K/32, g.kb.split); tbl != m.split {
			if a, ok := build(x, tbl, bias); ok {
				m.alt = &a
			}
		}
		return m, true
	}
	// mkkID is mkk with the expert fields. A dense weight passes experts=1 and
	// gets exactly the kernel it always got -- MatVec is byte-identical there,
	// which TestDenseKernelsUnchanged holds it to.
	mkkID := func(x nn.Weight, r *resident, experts, slots int, slotAct, bias bool) (mv, bool) {
		if experts <= 1 {
			return mkk(x, r, bias)
		}
		q, ok := quantOf(x.T)
		if !ok {
			return mv{}, false
		}
		// The split is chosen from Rows*Slots, not Rows: an indexed launch is
		// rows*slots*split threads and MatVec lays its partials out at stride
		// Rows*Slots, so Rows alone would over-split and reduce the wrong stride.
		// It comes from the table, not the tuner, which has no selection to pass;
		// WithSplit still pins it.
		nout := x.Rows * slots
		mk := func(split int) (mv, bool) {
			// ForceIndexedGroup selects the in-group reduction here without
			// depending on what retuneIndexed happens to time fastest.
			grp := g.ForceIndexedGroup && !g.NoGroupSplit && split > 1 && !bias &&
				kernels.GroupSplitOK(x.Rows, split)
			c, e := g.idKernel(kernels.MatVecShape{Center: g.center, T: q, K: x.K, Rows: x.Rows,
				Split: split, Experts: experts, Slots: slots, SlotAct: slotAct, Bias: bias,
				Sub: subFor(q, x.K), GroupSplit: grp})
			if e != nil {
				g.LastErr = e.Error()
				return mv{}, false
			}
			m := mv{kern: c, rows: x.Rows, slots: slots, split: split, q: q, k: x.K, group: grp,
				experts: experts, slotAct: slotAct}
			if grp {
				g.IndexedGroup++
			}
			if split > 1 && !grp {
				if m.red = g.reduceKernel(nout, split); m.red == nil {
					return mv{}, false
				}
				if !g.sizePart(nout * split) {
					return mv{}, false
				}
			}
			return m, true
		}
		// A narrowed sub-block takes no split: chooseSplit is handed K/32, zero
		// below 32 elements, which MatVec would refuse.
		if subFor(q, x.K) == 0 {
			if m, ok := mk(chooseSplit(nout, x.K/32, g.kb.split)); ok {
				return m, true
			}
		}
		return mk(1)
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
	hostExp := w.HostExperts != nil && (auto && mode == expHost ||
		g.HybridExperts && !(auto && mode == expCard) || auto && mode == expAuto && !g.NoHybrid)
	stream := moe && (g.StreamExperts || auto) && slots < bank && (slots > 1 || hostExp)
	hybrid := stream && hostExp
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
	// The unpack staging never costs a block. On a device that cannot page the
	// staging would never be reused, so it is freed when the block does not
	// otherwise fit; admission is then what it was without staging, and the
	// block's covered tensors take the host packer (NoUnpackRoom). On a paging
	// device it stays, because without it every page-in is a slow host repack;
	// reclaim() and slots() both price it.
	// The KV pools' free pages go first: they hold nothing.
	g.kvCompact(need, nil)
	if !g.room(need) && !g.stream[li] {
		g.freeRaw()
	}
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
	g.ensureRaw(ws, need)
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
						w.ExpGateB != nil || w.ExpUpB != nil || w.ExpDownB != nil)
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
					pre: w.PrefetchExperts, host: w.HostExperts, w: *w, hybrid: true, nExpert: bank, li: li}
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
			if x.Packed == nil {
				// A streamed bank needs the packed layout, since fill() sends a
				// sheet at a time. A float bank (an F32 or F16 file) has none, so
				// such a mixture streams nothing and is declined here by name.
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
					pre: w.PrefetchExperts, host: w.HostExperts, w: *w, li: li}
				l.stream.nExpert = bank
				// An auto-streamed block runs its experts on the host unless
				// told otherwise: on Kimi-K3 over the V100s that decoded 1.43x
				// the host's rate where sending the sheets lost to it
				// (placement.md 16c-2).
				switch {
				case l.stream.host == nil || auto && mode == expCard:
				case auto && mode == expHost, g.HybridExperts:
					l.stream.hybrid = true
				default:
					l.stream.hybrid = auto && !g.NoHybrid
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
		if (i == 3 || i == 6) && e <= 1 && bias[i] == nil && !p.NonCausal && k.kern != nil {
			k.res = g.kernelMode(k.q, k.k, k.rows, k.split, max(k.rowt, 1), true, k.group)
		}
		if i == 5 && e <= 1 && bias[i] == nil && !ungatedFFN(p) && k.kern != nil && k.rowt <= 1 &&
			(p.Act == kernels.ActSiLU || p.Act == kernels.ActGELU) {
			k.gated = g.gatedKernel(k.q, k.k, k.rows, k.split, k.group, p.Act)
			// nil at a partial-writing split; the retune builds it once the
			// split it picks has a final-row form.
			k.wantGated = true
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
		rk, e := kernels.RouterMatVecOf(p.NExpert, p.NEmbd, es == 2)
		if e != nil {
			g.LastErr = e.Error()
			return false
		}
		if l.mvRouter, e = g.dev.Compile(rk); e != nil {
			g.LastErr = e.Error()
			return false
		}
		// Padded to a whole word: the f16 kernel loads the word an element
		// sits in, and an odd element count's last word runs past the matrix.
		rb := w.Router.Data[:es*n]
		if len(rb)%4 != 0 {
			rb = append(append([]byte(nil), rb...), make([]byte, 4-len(rb)%4)...)
		}
		var b backend.Buf
		if b, e = g.dev.Alloc(len(rb)); e != nil {
			return false
		}
		if e = b.Write(rb); e != nil {
			return false
		}
		l.router, l.routerF16 = b, es == 2
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
	if p.Act == kernels.ActXIELU {
		if len(w.XIELU) != 4 {
			g.LastErr = fmt.Sprintf("block %d: xIELU wants 4 numbers, the block carries %d", li, len(w.XIELU))
			return false
		}
		up(&l.xielu, w.XIELU)
		l.auxBytes += 16
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
	// After the biases: the fused kernel bakes which segments take one, and the
	// launch appends a buffer for every bias it finds. CUDA does not check the
	// count, so a mismatch shifts every later buffer.
	g.fuseQKV(l)
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
	if linear {
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
		// State placing the same block reuses the shared weights above and gets
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
	// A non-causal block's batched twins are built here, not lazily: every
	// call over it is batched, and a Compile from inside layersOnce would post
	// to the backend goroutine that is blocked running the session -- a
	// deadlock. prepBatch walks the causal layers eagerly for the same reason.
	if p.NonCausal && g.bs != nil {
		for _, m := range []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd} {
			if m.kern == nil {
				continue
			}
			if _, ok := g.batchMV(*m, g.bs.rows, g.bs.tok); !ok {
				g.LastErr = fmt.Sprintf("non-causal block %d: no batched twin at %d rows",
					li, g.bs.rows)
				return false
			}
		}
	}
	// The reserved widths' batched twins are built here, against this block,
	// as the non-causal twins are above. Their shared buffers -- the binary16
	// activation and the k-split partials of sm_70's tensor-core twin
	// (voltaMV) -- are sized by the block's matrices, which reserveScratch,
	// running before the first block is installed, cannot see. Left to the
	// first prompt or step, they were asked of a card the blocks had filled:
	// 12 MB for Llama-3.2-1B on a Volta-class card, a driver out-of-memory on a
	// full card and the last swap slot on a streaming one.
	if !p.NonCausal {
		for _, w := range []int{g.promptW, g.stepW} {
			if w > 0 {
				g.prepBatch(w)
			}
		}
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

// partsFor is how many partial sums a norm over n elements reduces through:
// initScratch's own 64, halved until it divides.
// It is a function because MLA norms other widths (KVLoraRank, QLoraRank)
// that 64 may not divide, and NormPart refuses those.
func partsFor(n int) int {
	parts := 64
	for parts > 1 && (n <= 0 || n%parts != 0) {
		parts /= 2
	}
	return parts
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

// initRopeTable builds the device's own rotary table for this scratch, or
// leaves it nil and records why in Stats.RopeTableWhy.
//
// It never fails the scratch: Layers is also given the host's finished
// cs/csSWA, so a device without the kernel uploads the table and computes the
// same logits. Only a silent absence would be wrong, hence the recorded reason.
// The planes depend on the model, not the position, so they go up once and a
// token writes four bytes a row instead of NRot*4. It is built even under
// Config.RopeTableHost so `jitllm verify -ab ropetab` can alternate both arms
// in one process.
func (g *devTier) initRopeTable(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	why := func(f string, a ...any) error {
		g.RopeTableWhy = fmt.Sprintf(f, a...)
		return nil
	}
	switch {
	case p.NonCausal || p.NRot <= 0:
		// Not an absence: a ViT's rotary, where it has one, is a per-row table
		// the host builds per grid and hands over as cs; with none, bs.cs is
		// nil too and there is no table for either tier to build.
		return nil
	case len(p.RopeTab) == 0:
		return why("the plan carries no folded rotary planes")
	}
	npairs := p.NRot / 2
	if want := kernels.RopeTabBlock(npairs); len(p.RopeTab) != want {
		return why("the plan's rotary planes are %d words, the layout wants %d",
			len(p.RopeTab), want)
	}
	if p.SWAPeriod > 0 && len(p.RopeTabSWA) != kernels.RopeTabBlock(npairs) {
		return why("SWAPeriod is %d and the LOCAL rotary planes are %d words",
			p.SWAPeriod, len(p.RopeTabSWA))
	}
	k, err := kernels.RopeTable(npairs, rows)
	if err != nil {
		return why("%v", err)
	}
	c, err := g.compileShared(k)
	if err != nil {
		return why("%s declined the rotary table: %v", g.dev.API(), err)
	}
	// The buffers are allocated after the compile, so a device that refuses the
	// kernel spends nothing; an allocation failure from here on is a real error.
	up := func(dst *backend.Buf, v []float32) error {
		b, err := g.dev.Alloc(len(v) * 4)
		if err != nil {
			return err
		}
		*dst = b
		return b.Write(f32b(v))
	}
	if err := up(&bs.ropeTab, p.RopeTab); err != nil {
		closeKern(c)
		return err
	}
	if p.SWAPeriod > 0 {
		if err := up(&bs.ropeTabSWA, p.RopeTabSWA); err != nil {
			closeKern(c)
			return err
		}
	}
	if err := up(&bs.ropeConst, kernels.RopeTabConsts()); err != nil {
		closeKern(c)
		return err
	}
	// (rows+1) words, the first being the valid row count as in bs.n, so a
	// ragged chunk's per-token staging is one small write (see kernels.RopeTable).
	b, err := g.dev.Alloc((rows + 1) * 4)
	if err != nil {
		closeKern(c)
		return err
	}
	// The table is zeroed once: the kernel writes only valid rows, so the padded
	// tail must already hold the zeros the host staging would have put there.
	// Nothing else writes bs.cs, so this deliberately un-poisons it under
	// WithPoisonScratch.
	z := make([]byte, rows*p.NRot*4)
	for _, t := range []backend.Buf{bs.cs, bs.csSWA} {
		if t == nil {
			continue
		}
		if err := t.Write(z); err != nil {
			closeKern(c)
			b.Free()
			return err
		}
	}
	bs.rpos, bs.ropeTable, bs.ropeNPairs = b, c, npairs
	return nil
}

// initScratch allocates and compiles everything that does not depend on which
// block is running.
func (g *devTier) initScratch(p *nn.LayerPlan, rows int) (out *blockScratch) {
	// Everything the build allocates is the scratch's, charged (scratch.go).
	defer g.scratchWin().close()
	if rows < 1 {
		rows = 1
	}
	// The capped plan: every bound, stride and score-row width below reads bs.p,
	// so the scratch is built for what the device holds, not the full context.
	cp := g.capped(p)
	p = &cp
	// A vision block's attention may go in query chunks (visionChunk), whose
	// scratch is rounded up to whole chunks before anything is sized by it.
	if p.NonCausal && rows > 1 {
		_, rows = visionChunk(rows, p.NHead, scoreStride(p, rows), g.tiles.qtile, g.tiles.atile, g.planeBudget())
	}
	bs := &blockScratch{p: cp, rows: rows, qtile: 1, ktile: 1, atile: 1}
	// A build that fails part-way frees what it allocated: every refusal below
	// is a plain return nil, and on a full card -- the case that refuses --
	// what it leaked was what the next build needed.
	defer func() {
		if out == nil {
			freeScratch(bs)
		}
	}()
	// What the build allocated, by the device's own count: a tower's set is
	// resized from it (visrows.go). Read here rather than from the window,
	// which measures only when it is the outermost.
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
		// A latent block on sm_70 takes both products on m8n8k4, and the
		// accumulate walks the softmax's row group, so the group is fixed
		// here, before the softmax is built.
		if p.MLA() && g.mla70Attn(p, rows) {
			bs.atile, bs.mla70 = 8*volta70AttnNT, true
		}
	}
	// The norm reduction wants enough parts to spread across the device and few
	// enough that NormApply's re-sum stays cheap. 64 wins even at 128 rows: the
	// re-read is cached, and the reduction's thread count is what binds (see
	// docs/engineering-history/gpu-kernels.md).
	bs.parts = 64
	if rows > 1 {
		bs.parts = g.tiles.parts
	}
	for bs.parts > 1 && p.NEmbd%bs.parts != 0 {
		bs.parts /= 2
	}
	qdim := p.NHead * p.HeadDim
	// The position split is for one token only: at rows > 1 the grid already
	// exceeds the card's slots, so AttnAccTiled runs unsplit.
	if rows == 1 {
		if sp := accSplitFor(qdim, g.dev.Slots(), g.kb.mvAccSplit); sp > 1 && !g.sizePart(qdim*sp) {
			return nil
		}
	}
	// The cache row, not NKVHead*HeadDim: they differ only under MLA, where a
	// position is one KVLoraRank+NRot row for the whole layer.
	kvDim := p.KVRow()
	// bs.a and bs.ax are shared by every quantize in the block, so they are
	// sized for the widest vector that gets quantized; a short buffer is a
	// device-side out-of-bounds write, not a Go panic. A mixture quantizes
	// NExpertUsed*NFFNExp, which is not p.NFFN (the dense width), so it is
	// computed rather than assumed.
	bs.nExpert, bs.nUsed, bs.nFFNExp = p.NExpert, p.NExpertUsed, p.NFFNExp
	moeWide := 0
	if p.NExpert > 0 {
		moeWide = p.NExpertUsed * p.NFFNExp
	}
	wide := p.NEmbd
	// The shared expert, the linear block's gated output (VHeads*VDim) and its
	// conv channels, and MLA's four widths below all quantize through the same
	// buffers, so all are in the max.
	inner, chans := 0, 0
	if r := cp.Recurrent; r.Conv > 0 {
		inner, chans = r.VHeads*r.VDim, r.Chans
	}
	mlaNopeW, mlaLatW, mlaOutW, mlaQAW := 0, 0, 0, 0
	if p.MLA() {
		bs.mlaLat, bs.mlaRot = p.KVLoraRank, p.NRot
		// The query's and kv_a's row: the latent and the rotary key. The cached
		// row (KVRow) also carries the indexer's key after them.
		bs.mlaRow, bs.mlaNopeW = p.KVLoraRank+p.NRot, p.HeadDim-p.NRot
		mlaNopeW = p.NHead * bs.mlaNopeW
		mlaLatW, mlaOutW, mlaQAW = p.NHead*p.KVLoraRank, p.NHead*p.HeadDimV, p.QLoraRank
	}
	// DeepSeek V4's query latent and grouped output (ds4.go).
	ds4QA, ds4OA := 0, 0
	if d := p.DS4; d != nil {
		ds4QA, ds4OA = p.QLoraRank, d.OGroups*d.OLoraRank
	}
	for _, n := range []int{p.NFFN, qdim, moeWide, p.NFFNShExp, inner, chans,
		mlaNopeW, mlaLatW, mlaOutW, mlaQAW, ds4QA, ds4OA} {
		if n > wide {
			wide = n
		}
	}
	var err error
	al := func(dst *backend.Buf, n int) {
		// cuMemAlloc(0) is an error, not an empty buffer; an absent buffer is
		// right, since no kernel reads it (e.g. a vision block's NRot is 0).
		if n == 0 {
			return
		}
		if err == nil {
			// The device refusing scratch is the card being full, the same
			// as a refused weight upload: counted as NoRoom, so a caller can
			// tell it from a shape this tier declines.
			if *dst, err = g.dev.Alloc(n); err != nil {
				g.NoRoom++
			}
		}
		// RULE 13: zero is the identity for a sum, so a kernel reading scratch it
		// did not write is right on the first session and wrong on the next.
		// WithPoisonScratch fills with NaN so the read shows in the first token.
		if err == nil && g.kb.poisonScratch && *dst != nil {
			(*dst).Write(nanFill(n))
		}
	}
	// Every AltUp stream, stream-major (altup.go); one stream elsewhere.
	al(&bs.x, rows*p.ResidW()*4)
	// x2 is every stream too where a hyper-connection or Kimi-K3 writes it.
	x2 := p.NEmbd
	if p.DS4 != nil || p.K3 != nil || p.Q4E != nil {
		x2 = p.ResidW()
	}
	al(&bs.x2, rows*x2*4)
	al(&bs.h, rows*p.NEmbd*4)
	al(&bs.mvOut, rows*p.NEmbd*4)
	al(&bs.part, rows*bs.parts*4)
	if lnBlock(p) {
		// LayerNorm's second pass needs partials of its own: the mean's must
		// survive into the apply, so the variance cannot overwrite them.
		al(&bs.part2, rows*bs.parts*4)
	}
	al(&bs.a, rows*wide)           // one byte per activation: packed int8
	al(&bs.ax, 3*rows*(wide/32)*4) // scale per 32, sum per 16
	if p.NExpert > 0 {
		k := p.NExpertUsed
		al(&bs.rlogits, p.NExpert*4)
		al(&bs.rsel, (k+1)*4)
		al(&bs.ident, k*4)
		al(&bs.rtop, (k+1)*4)
		al(&bs.rw, (k+1)*4) // slot k is ExpertRoute's bin
		al(&bs.rwProbe, (k+1)*4)
		if p.DenseMoE {
			al(&bs.hr, rows*p.NEmbd*4)
			al(&bs.rwS, k*4)
		}
		if p.ExpertGroups > 1 {
			al(&bs.rgs, p.ExpertGroups*4)
			al(&bs.rbm, p.NExpert*4)
		}
		al(&bs.eg, k*p.NFFNExp*4)
		al(&bs.eu, k*p.NFFNExp*4)
		al(&bs.eact, k*p.NFFNExp*4)
		al(&bs.edown, k*p.NEmbd*4)
		if p.MoEBias {
			al(&bs.rlogitsB, p.NExpert*4)
			al(&bs.egB, k*p.NFFNExp*4)
			al(&bs.euB, k*p.NFFNExp*4)
			al(&bs.edownB, k*p.NEmbd*4)
		}
	}
	if p.QKNorm || p.QKL2Norm {
		al(&bs.qn, rows*qdim*4)
		al(&bs.kn, rows*kvDim*4)
	}
	al(&bs.qr, rows*qdim*4)
	if p.RopeSplit {
		al(&bs.krot, rows*kvDim*4)
		al(&bs.kroff, rows*4)
	}
	if p.AttnTempScale != 0 && !p.NoPEGlobal {
		al(&bs.qt, rows*qdim*4)
	}
	if p.NFFNShExp > 0 {
		bs.nShExp = p.NFFNShExp
		al(&bs.shg, rows*p.NFFNShExp*4)
		al(&bs.shu, rows*p.NFFNShExp*4)
		al(&bs.shact, rows*p.NFFNShExp*4)
		al(&bs.shout, rows*p.NEmbd*4)
		al(&bs.shlogit, rows*4)
		al(&bs.shsum, rows*p.NEmbd*4)
	}
	if p.SubNorm {
		// BitNet's inner norms write their rows here, out of place (RULE 13).
		al(&bs.sub, rows*max(qdim, p.NFFN)*4)
	}
	if p.AttnOutGate {
		// Double width: with the gate folded in, l.wq has 2*qdim rows and the
		// matvec writes all of them.
		al(&bs.q, rows*2*qdim*4)
		al(&bs.qg, rows*qdim*4)
		al(&bs.ogate, rows*qdim*4)
		al(&bs.ogated, rows*qdim*4)
	} else {
		al(&bs.q, rows*qdim*4)
	}
	al(&bs.k, rows*kvDim*4)
	al(&bs.v, rows*kvDim*4)
	if p.ClampKQV != 0 || p.Clamps {
		al(&bs.qClamp, rows*qdim*4)
		al(&bs.kClamp, rows*kvDim*4)
		al(&bs.vClamp, rows*kvDim*4)
	}
	if p.Clamps {
		wide := max(p.NEmbd, qdim, p.NFFN)
		al(&bs.cin, rows*wide*4)
		al(&bs.cout, rows*wide*4)
		al(&bs.gClamp, rows*p.NFFN*4)
		al(&bs.uClamp, rows*p.NFFN*4)
	}
	// All three attention kernels share the score row (scoreStride).
	bs.sstride = scoreStride(p, rows)
	// Paged history reads through FlashDecodeKV, which keeps no score plane:
	// the planes, and every kernel below that reads a contiguous cache, are
	// not built (pagedSkip).
	paged := g.pagedPlan(p)
	// A non-causal block's attention may run in query chunks
	// (visionChunk), so its score planes are sized for a chunk rather than
	// the whole run.
	bs.arows = rows
	if p.NonCausal {
		bs.arows, _ = visionChunk(rows, p.NHead, bs.sstride, bs.qtile, bs.atile, g.planeBudget())
	}
	if !paged {
		al(&bs.att, bs.arows*p.NHead*bs.sstride*4)
	}
	if p.AttnSinks {
		al(&bs.noSink, p.NHead*4)
		al(&bs.sinkProbe, p.NHead*4)
		if err == nil {
			inf := make([]float32, p.NHead)
			for i := range inf {
				inf[i] = float32(math.Inf(-1))
			}
			err = bs.noSink.Write(f32b(inf))
		}
	}
	if p.AttnSoftcap != 0 && !paged {
		al(&bs.attCap, bs.arows*p.NHead*bs.sstride*4)
	}
	if !paged {
		al(&bs.prob, bs.arows*p.NHead*bs.sstride*4)
	}
	if bs.arows < rows {
		g.allocAttnChunks(bs, qdim, al)
	}
	al(&bs.xb, rows*qdim*4)
	// An ungated FFN (a vision block, C6) has no gate to write here; at a
	// 4096-patch image the plane is 70 MB nothing reads.
	if !ungatedFFN(p) {
		al(&bs.g, rows*p.NFFN*4)
	}
	al(&bs.u, rows*p.NFFN*4)
	al(&bs.act, rows*p.NFFN*4)
	al(&bs.cs, rows*p.RopeW()*4)
	if p.SWAPeriod > 0 {
		al(&bs.csSWA, rows*p.RopeW()*4)
	}
	if err == nil {
		err = g.initRopeTable(bs, p, rows)
	}
	// A linear block's scratch, sized from the plan's own geometry. al() is a
	// no-op at zero, so other architectures allocate nothing here.
	if r := cp.Recurrent; r.Conv > 0 {
		bs.rec = r
		if r.WithAttn {
			al(&bs.side, rows*p.NEmbd*4)
		}
		if r.ShortConv {
			for _, b := range []*backend.Buf{&bs.scB, &bs.scC, &bs.scBx, &bs.scY} {
				al(b, rows*r.Chans*4)
			}
		}
		al(&bs.dMixed, rows*r.Chans*4)
		al(&bs.dConv, rows*r.Chans*4)
		al(&bs.dZ, rows*r.VHeads*r.VDim*4)
		al(&bs.dOut, rows*r.VHeads*r.VDim*4)
		al(&bs.dBA, rows*2*r.VHeads*4)
		// The gate buffers are per channel on KDA and per head on qwen3next.
		nGate := r.VHeads
		if r.ChanDecay {
			nGate = r.VHeads * r.KDim
		}
		for _, b := range []*backend.Buf{&bs.dAlpha, &bs.dDecay} {
			al(b, rows*nGate*4)
		}
		for _, b := range []*backend.Buf{&bs.dBRaw, &bs.dBeta} {
			al(b, rows*r.VHeads*4)
		}
		if r.Mamba1 {
			// dt's bottleneck before and after its norm, its quantization (a
			// scale per 32 and a sum per 16, as bs.ax), and B and C before
			// and after theirs.
			al(&bs.dLowRank, rows*r.Rank*4)
			al(&bs.m1LowN, rows*r.Rank*4)
			al(&bs.dLRa, rows*r.Rank)
			al(&bs.dLRax, 3*rows*(r.Rank/32)*4)
			for _, b := range []*backend.Buf{&bs.m1B, &bs.m1C, &bs.m1BN, &bs.m1CN} {
				al(b, rows*r.KDim*4)
			}
		}
		if r.ChanDecay {
			// The rank IS KDim: KDA's f_a_proj and g_a_proj both project to
			// linear_head_dim, which is the state's key extent.
			al(&bs.dLowRank, rows*r.KDim*4)
			// The scale plane is three floats per 32-element group (a scale per 32
			// and a sum per 16), as bs.ax is sized. An undersized one ran on CUDA
			// and failed on Vulkan.
			al(&bs.dLRa, rows*r.KDim)
			al(&bs.dLRax, 3*rows*(r.KDim/32)*4)
			al(&bs.dGateWaste, rows*nGate*4)
		}
		// q and k ping-pong between two buffers each: every device kernel is out
		// of place, and slice -> norm -> scale needs only a swap per step.
		qkDim := r.KHeads * r.KDim
		for _, b := range []*backend.Buf{&bs.dQ, &bs.dQn, &bs.dK, &bs.dKn} {
			al(b, rows*qkDim*4)
		}
		for _, b := range []*backend.Buf{&bs.dV, &bs.dOutN, &bs.dAct} {
			al(b, rows*r.VHeads*r.VDim*4)
		}
		// 1/sqrt(kDim), written once: it is geometry, and kernels.Scale takes its
		// factor from a one-element buffer.
		al(&bs.dScale, 4)
		// The recurrent descriptor (kernels.RecDescWords): the real row count,
		// which Conv1dShift reads -- a padded chunk submits surplus zero rows
		// and a zero column shifted into the state is a column the model never
		// produced, and it stays there -- and each sequence's slots in the
		// block's pool; for a ragged step, the run descriptor its rows are
		// grouped by (kernels.RunDescWords, the longer of the two).
		al(&bs.dRows, 4*kernels.RunDescWords(rows))
	}
	// MLA's scratch, allocated only where the model has a latent block.
	if p.MLA() {
		lat, rot, row := bs.mlaLat, bs.mlaRot, bs.mlaRow
		// The norm reductions are over the latent and the query latent, not over
		// NEmbd, so they need their own partial counts: bs.parts is chosen to
		// divide NEmbd and a 16-wide latent is not divisible by 64.
		bs.mlaParts, bs.mlaQAParts = partsFor(lat), partsFor(p.QLoraRank)
		al(&bs.mlaKV, row*4)
		al(&bs.mlaKVn, lat*4)
		al(&bs.mlaKVPart, bs.mlaParts*4)
		al(&bs.mlaNope, mlaNopeW*4)
		al(&bs.mlaPE, p.NHead*rot*4)
		al(&bs.mlaPErot, p.NHead*rot*4)
		al(&bs.mlaKPE, rot*4)
		al(&bs.mlaAbsRaw, mlaLatW*4)
		al(&bs.mlaAbs, p.NHead*row*4)
		al(&bs.mlaAcc, mlaLatW*4)
		al(&bs.mlaOut, mlaOutW*4)
		al(&bs.mlaAbsOff, p.NHead*4)
		al(&bs.mlaPEOff, p.NHead*4)
		al(&bs.mlaPOff, 4)
		al(&bs.mlaIdent, p.NHead*4)
		if p.QLoraRank != 0 {
			al(&bs.mlaQA, p.QLoraRank*4)
			al(&bs.mlaQAn, p.QLoraRank*4)
			al(&bs.mlaQAPart, bs.mlaQAParts*4)
		}
	}
	al(&bs.n, (rows+1)*4)
	al(&bs.koff, rows*4)
	al(&bs.kpos, rows*4)
	al(&bs.roff, rows*4)
	al(&bs.zero, 4)
	var tempTab []float32
	if p.QKL2Norm {
		al(&bs.l4Ones, p.HeadDim*4)
	}
	if p.AttnTempScale != 0 {
		tempTab = l4TempTab(p)
		al(&bs.tempTab, len(tempTab)*4)
	}
	if p.ExpertWeightIn && p.NExpert > 0 {
		al(&bs.wOnes, p.NExpertUsed*4)
	}
	if residScaled(p) {
		al(&bs.rscale, 4)
	}
	// Gemma 4's v norm and output scalar: v normed out of place, against
	// ones, and the 1.0 the scaled residual is copied back with.
	if p.VNorm {
		al(&bs.vn, rows*p.NKVHead*p.HeadDim*4)
		al(&bs.hdOnes, p.HeadDim*4)
	}
	if p.GeomSplit {
		al(&bs.one, 4)
	}
	allocAltUp(bs, p, rows, al)
	if p.PLEDim != 0 {
		al(&bs.pleIn, rows*p.PLEWidth*4)
		al(&bs.pleG, rows*p.PLEDim*4)
		al(&bs.pleS, rows*p.PLEDim*4)
		al(&bs.pleA, rows*p.PLEDim*4)
		al(&bs.pleO, rows*p.NEmbd*4)
	}
	if err != nil {
		g.LastErr = err.Error()
		return nil
	}
	// Llama 4's three constant planes and Granite's residual scale, written once
	// because they are geometry.
	for _, c := range []struct {
		b backend.Buf
		v []float32
	}{{bs.l4Ones, ones32(p.HeadDim)}, {bs.tempTab, tempTab}, {bs.wOnes, ones32(p.NExpertUsed)},
		{bs.rscale, []float32{float32(p.ResidualScale)}}, {bs.hdOnes, ones32(p.HeadDim)},
		{bs.one, []float32{1}}} {
		if c.b == nil {
			continue
		}
		if err = c.b.Write(f32b(c.v)); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}
	if bs.ident != nil {
		id := make([]uint32, p.NExpertUsed)
		for i := range id {
			id[i] = uint32(i)
		}
		if err = bs.ident.Write(u32b(id)); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}
	// MLA's three constant buffers, written once: the two destination lists
	// interleave each head's absorbed latent and rotated key into one row-wide
	// query, and the identity is the selection both absorb banks are indexed with.
	if bs.mlaIdent != nil {
		id := make([]uint32, p.NHead)
		abs := make([]uint32, p.NHead)
		pe := make([]uint32, p.NHead)
		for h := 0; h < p.NHead; h++ {
			id[h] = uint32(h)
			if g.kb.mlaFault == MLAFaultSheetOff {
				// Head h absorbs through head h+1's sheet: in bounds, finite,
				// and a different model. See MLAFault.
				id[h] = uint32((h + 1) % p.NHead)
			}
			abs[h] = uint32(h * bs.mlaRow)
			pe[h] = uint32(h*bs.mlaRow + bs.mlaLat)
		}
		for _, w := range []struct {
			b backend.Buf
			v []uint32
		}{{bs.mlaIdent, id}, {bs.mlaAbsOff, abs}, {bs.mlaPEOff, pe}} {
			if err = w.b.Write(u32b(w.v)); err != nil {
				g.LastErr = err.Error()
				return nil
			}
		}
	}
	if err = bs.zero.Write([]byte{0, 0, 0, 0}); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	if bs.dScale != nil {
		sc := float32(1 / math.Sqrt(float64(cp.Recurrent.KDim)))
		if err = bs.dScale.Write(f32b([]float32{sc})); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}
	// The rotated q of row r lands at r*qdim, which never changes -- unlike the
	// K cache offsets, which move with the position. Written once here.
	roff := make([]byte, 0, rows*4)
	for r := 0; r < rows; r++ {
		roff = append(roff, u32one(uint32(r*qdim))...)
	}
	if err = bs.roff.Write(roff); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	// XD-RoPE's row-major k (bs.krot): row r at r*kvDim.
	if bs.kroff != nil {
		ko := make([]byte, 0, rows*4)
		for r := 0; r < rows; r++ {
			ko = append(ko, u32one(uint32(r*p.KVRow()))...)
		}
		if err = bs.kroff.Write(ko); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}

	r := cp.Recurrent
	actWin := p.ActWin
	if actWin < 32 {
		actWin = 32
	}
	// The plan's score scale, as every paged kernel reads it: the contiguous
	// tiles below serve only a vision block, where the two agree.
	scale := g.scoreScale(p)
	gqa := p.NHead / p.NKVHead
	// Settle the subgroup width before any kernel is built: it decides which
	// variant of two kernels this device gets, and it is one answer for the
	// device rather than one per kernel.
	bs.qkLanes = g.pickLanes()
	bs.qkWidth = bs.qkLanes
	if bs.qkLanes == 1 {
		bs.qkWidth = scalarWidth
	}
	// HeadNorm's warp form also needs a head that is a whole number of
	// subgroups; otherwise the kernel build fails and surfaces as a misleading
	// budget decline. headLanes drops such a head to the scalar form (the gated
	// delta rule's KDim/VDim can be any width).
	headLanes := func(width int) int {
		if bs.qkLanes == ir.SubgroupLanes && width%ir.SubgroupLanes != 0 {
			return 1
		}
		return bs.qkLanes
	}
	// The per-kernel lanes, set when each head norm is built; see qNormL.
	bs.qNormL, bs.kNormL, bs.qkNormL, bs.outNormL = bs.qkLanes, bs.qkLanes, bs.qkLanes, bs.qkLanes
	type job struct {
		dst  *backend.Kernel
		make func() (*ir.Kernel, error)
	}
	jobs := []job{
		// Each quantizer's window is clamped to its own row, as the host
		// clamps it: a session window wider than a row (a per-token model's)
		// must not take the next row's maximum.
		{&bs.quantE, func() (*ir.Kernel, error) {
			return kernels.Quantize(rows*p.NEmbd, kernels.ClampWindow(actWin, p.NEmbd))
		}},
		{&bs.quantF, func() (*ir.Kernel, error) {
			if p.NFFN == 0 {
				return nil, nil
			}
			return kernels.Quantize(rows*p.NFFN, kernels.ClampWindow(actWin, p.NFFN))
		}},
		{&bs.subQA, func() (*ir.Kernel, error) {
			if !p.SubNorm {
				return nil, nil
			}
			return g.rmsQuantRows()(qdim, rows, float32(p.RMSEps), false, false, kernels.ClampWindow(actWin, qdim))
		}},
		{&bs.subQF, func() (*ir.Kernel, error) {
			if !p.SubNorm {
				return nil, nil
			}
			return g.rmsQuantRows()(p.NFFN, rows, float32(p.RMSEps), false, false, kernels.ClampWindow(actWin, p.NFFN))
		}},
		{&bs.quantAct, func() (*ir.Kernel, error) {
			if ungatedFFN(p) || p.NFFN == 0 {
				return nil, nil
			}
			// Optional: an activation with no gated form keeps two launches.
			if k, err := kernels.QuantizeAct(rows*p.NFFN, kernels.ClampWindow(actWin, p.NFFN), p.Act); err == nil {
				return k, nil
			}
			return nil, nil
		}},
		// Quantize bakes its element count, so the gated delta output
		// (VHeads*VDim wide, not NFFN) and the attention output (nHead*headDim,
		// not NEmbd) each get their own quantizer. Borrowing a kernel of another
		// width quantizes the wrong number of elements and runs fluently wrong.
		{&bs.quantD, func() (*ir.Kernel, error) {
			r := cp.Recurrent
			if r.Conv == 0 {
				return nil, nil
			}
			return kernels.Quantize(rows*r.VHeads*r.VDim, kernels.ClampWindow(actWin, r.VHeads*r.VDim))
		}},
		{&bs.quantQ, func() (*ir.Kernel, error) { return kernels.Quantize(rows*qdim, kernels.ClampWindow(actWin, qdim)) }},
		// The norm is two kernels (RMSNorm) or three (LayerNorm: mean, variance
		// about it, apply); see blockScratch.normVar.
		{&bs.normPart, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return kernels.LayerNormPartRows(p.NEmbd, bs.parts, rows)
			}
			return kernels.NormPartRows(p.NEmbd, bs.parts, rows)
		}},
		{&bs.rms, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			return g.rmsRows()(p.NEmbd, rows, float32(p.RMSEps), false, false)
		}},
		{&bs.addRms, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			return g.rmsRows()(p.NEmbd, rows, float32(p.RMSEps), false, true)
		}},
		{&bs.rmsQ, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			// Optional: a width the window does not tile keeps the two
			// launches rather than failing the scratch.
			if k, err := g.rmsQuantRows()(p.NEmbd, rows, float32(p.RMSEps), false, false, kernels.ClampWindow(actWin, p.NEmbd)); err == nil {
				return k, nil
			}
			return nil, nil
		}},
		{&bs.addRmsQ, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			// Optional: a width the window does not tile keeps the two
			// launches rather than failing the scratch.
			if k, err := g.rmsQuantRows()(p.NEmbd, rows, float32(p.RMSEps), false, true, kernels.ClampWindow(actWin, p.NEmbd)); err == nil {
				return k, nil
			}
			return nil, nil
		}},
		{&bs.copyRows, func() (*ir.Kernel, error) {
			// A block with no pre-norm copies the residual into h instead.
			if !p.NoPreNorm && p.Q4E == nil {
				return nil, nil
			}
			return kernels.Restride(rows, p.NEmbd, p.NEmbd, p.NEmbd)
		}},
		{&bs.normApply, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return kernels.LayerNormApplyRows(p.NEmbd, bs.parts, float32(p.RMSEps), true, rows)
			}
			return kernels.NormApplyRows(p.NEmbd, bs.parts, float32(p.RMSEps), false, rows)
		}},
		// RoPE is not generated at all for a vision block with no rotary (NRot
		// is zero), so it cannot be launched by mistake. A ViT with a 2-D
		// rotary (Qwen2-VL, Pixtral) is the text block's rotation over a table
		// the host built per patch from its grid coordinates (nn.Rope.TableAt):
		// the kernels do not know a position has two axes.
		{&bs.ropeQ, func() (*ir.Kernel, error) {
			// MLA rotates through its own pair (mlaRopeQ/mlaRopeK): its rotary
			// halves are the tail of each query head and one shared key.
			if (p.NonCausal && p.NRot == 0) || p.MLA() {
				return nil, nil
			}
			if p.RopeSplit {
				return kernels.RoPERowsSplit(p.NHead, p.HeadDim, p.NRot, rows, 0)
			}
			return kernels.RoPERows(p.NHead, p.HeadDim, p.NRot, p.RopeNeox, rows)
		}},
		{&bs.ropeK, func() (*ir.Kernel, error) {
			if (p.NonCausal && p.NRot == 0) || p.MLA() {
				return nil, nil
			}
			if p.RopeSplit {
				return nil, nil // ropeKRow
			}
			return kernels.RoPERowsT(p.NKVHead, p.HeadDim, p.NRot, p.RopeNeox, rows, g.kStride(p))
		}},
		{&bs.ropeKRow, func() (*ir.Kernel, error) {
			// XD-RoPE's k is normed after the rotary, so it turns row-major
			// into bs.krot and reaches the cache, contiguous or paged, through
			// copyK.
			if !p.RopeSplit {
				return nil, nil
			}
			return kernels.RoPERowsSplit(p.NKVHead, p.HeadDim, p.NRot, rows, 0)
		}},

		{&bs.normVar, func() (*ir.Kernel, error) {
			if !lnBlock(p) {
				return nil, nil
			}
			return kernels.LayerNormVarRows(p.NEmbd, bs.parts, rows)
		}},
		{&bs.copyK, func() (*ir.Kernel, error) {
			// Needed wherever k reaches the cache unrotated or partly rotated: a
			// ViT with no rotary, a partial rotary (RoPERows leaves the tail
			// unwritten), Llama 4's NoPE layers, and a model with no rotary at all.
			// A non-causal set may hold a block that rotates nothing beside
			// ones that do (an encoder's decision head; layer.noRope).
			if p.MLA() || (!p.NonCausal && !partialRotary(p) && !p.NoPEGlobal && !p.NoPosEnc &&
				!p.RopeSplit) {
				return nil, nil
			}
			if st := g.kStride(p); st > 0 {
				return kernels.CopyAtRowsT(kvDim, rows, st)
			}
			return kernels.CopyAtRows(kvDim, rows)
		}},
		{&bs.qTemp, func() (*ir.Kernel, error) {
			if p.AttnTempScale == 0 {
				return nil, nil
			}
			return kernels.ScaleRowsByCount(rows, qdim, int(p.AttnTempFloor),
				int(p.AttnTempOffset), len(l4TempTab(p)))
		}},
		{&bs.actMulW, func() (*ir.Kernel, error) {
			if !p.ExpertWeightIn || p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ActMulWeighted(p.NExpertUsed*p.NFFNExp, p.NFFNExp, p.Act, false)
		}},
		{&bs.copyQ, func() (*ir.Kernel, error) {
			if p.MLA() || !partialRotary(p) {
				return nil, nil
			}
			return kernels.CopyAtRows(qdim, rows)
		}},
		{&bs.copyV, func() (*ir.Kernel, error) {
			// MLA writes no V at all: the value is the key row's own prefix.
			if p.MLA() {
				return nil, nil
			}
			if g.KVF16 {
				return kernels.CopyAtF16(kvDim)
			}
			return kernels.CopyAtRows(kvDim, rows)
		}},
		// MLA's own sequence, built from existing kernels: the absorb is the
		// indexed matvec, scores and accumulate are the ordinary two at two widths,
		// and the rest are slices, rotations and copies.
		{&bs.mlaSliceNope, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// q's nope half is the head of each head and its rotary half the
			// tail, at a stride of HeadDim. Both are compacted, because the
			// absorb wants one contiguous activation per head at stride nope
			// and the rotation wants nRot pairs from a base.
			return kernels.SliceRows(bs.mlaNopeW, p.HeadDim, 0, p.NHead)
		}},
		{&bs.mlaSlicePE, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.SliceRows(bs.mlaRot, p.HeadDim, bs.mlaNopeW, p.NHead)
		}},
		{&bs.mlaSliceKPE, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// The key's rotary half is the tail of the kv_a output, one for the
			// whole layer (the "mqa" in kv_a_proj_with_mqa).
			return kernels.SliceRows(bs.mlaRot, bs.mlaRow, bs.mlaLat, 1)
		}},
		{&bs.mlaRopeQ, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// NHead "heads" of exactly nRot, because the halves were compacted
			// above -- so the rotation is over the whole of each one.
			return kernels.RoPERows(p.NHead, bs.mlaRot, bs.mlaRot, p.RopeNeox, 1)
		}},
		{&bs.mlaRopeK, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// It rotates straight into the cache at this position's row plus the
			// latent's width; RoPERows takes its destination offset from a buffer.
			return kernels.RoPERows(1, bs.mlaRot, bs.mlaRot, p.RopeNeox, 1)
		}},
		{&bs.mlaCopyLat, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.CopyAt(bs.mlaLat)
		}},
		// A NoPE MLA model (Kimi-Linear) still writes both destinations the rope
		// kernels would, so these copies replace them rather than a skipped launch.
		{&bs.mlaNoPEQ, func() (*ir.Kernel, error) {
			if !p.MLA() || !p.NoPosEnc {
				return nil, nil
			}
			return kernels.CopyAt(p.NHead * bs.mlaRot)
		}},
		{&bs.mlaNoPEK, func() (*ir.Kernel, error) {
			if !p.MLA() || !p.NoPosEnc {
				return nil, nil
			}
			return kernels.CopyAt(bs.mlaRot)
		}},
		{&bs.mlaCopyAbs, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.CopyAtRows(bs.mlaLat, p.NHead)
		}},
		{&bs.mlaCopyPE, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.CopyAtRows(bs.mlaRot, p.NHead)
		}},
		{&bs.mlaQuantNope, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.Quantize(p.NHead*bs.mlaNopeW, actWin)
		}},
		{&bs.mlaQuantAcc, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.Quantize(p.NHead*bs.mlaLat, actWin)
		}},
		{&bs.mlaQuantOut, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.Quantize(p.NHead*p.HeadDimV, actWin)
		}},
		{&bs.mlaNormKVPart, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.NormPart(bs.mlaLat, bs.mlaParts)
		}},
		{&bs.mlaNormKVApply, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.NormApply(bs.mlaLat, bs.mlaParts, latentEps(p), false)
		}},
		{&bs.mlaQuantQA, func() (*ir.Kernel, error) {
			if !p.MLA() || p.QLoraRank == 0 {
				return nil, nil
			}
			return kernels.Quantize(p.QLoraRank, actWin)
		}},
		{&bs.mlaNormQAPart, func() (*ir.Kernel, error) {
			if !p.MLA() || p.QLoraRank == 0 {
				return nil, nil
			}
			return kernels.NormPart(p.QLoraRank, bs.mlaQAParts)
		}},
		{&bs.mlaNormQAApply, func() (*ir.Kernel, error) {
			if !p.MLA() || p.QLoraRank == 0 {
				return nil, nil
			}
			return kernels.NormApply(p.QLoraRank, bs.mlaQAParts, latentEps(p), false)
		}},
		{&bs.scores, func() (*ir.Kernel, error) {
			// MLA's scores are MQA over the whole cache row in the existing kernel's
			// terms: head width and cache stride are the row, and gqa is NHead so
			// h/gqa is 0 for every head (as engine/model/forward.go does). The scale is the
			// plan's, not 1/sqrt(row).
			if p.MLA() {
				// A decode row takes the warp-per-(head, position) kernel
				// where the device guarantees the width: the thread-per-pair
				// one walks the whole 576-wide row serially (see
				// kernels.AttnScoresWarp).
				if rows == 1 && g.pickLanes() == ir.SubgroupLanes {
					bs.scoresWarp = true
					return kernels.AttnScoresWarp(p.NHead, p.KVRow(), p.KVRow(), p.NHead,
						bs.sstride, g.scoreScale(p))
				}
				return kernels.AttnScoresTiled(p.NHead, p.KVRow(), p.KVRow(), p.NHead,
					bs.sstride, g.scoreScale(p), rows, bs.qtile, bs.ktile, 0)
			}
			// The matrix instruction where the shape allows it: q.k is a matrix
			// product and the transposed K cache stores B the way mma wants it. A
			// refusal falls through to the FMA tile.
			if kst := g.kStride(p); bs.arows > 1 && kst > 0 && !g.NoMMA && !g.kb.noAttnMMA &&
				p.HeadDim%16 == 0 && bs.arows%16 == 0 {
				k, err := kernels.AttnScoresMMA(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride,
					scale, bs.arows, kst, g.tiles.attnNT)
				if err == nil {
					if c, e := g.dev.Compile(k); e == nil {
						bs.scoresMMA, g.AttnMMA = c, true
						return nil, nil
					}
				}
			}
			// sm_70's m8n8k4 where m16n8k16 does not lower, for both products.
			// Not under query chunks: its kernels are built for the whole image.
			if bs.arows == rows && g.volta70Attn(bs, kvDim, gqa, scale) {
				return nil, nil
			}
			// The one-kernel form on Metal's matrix unit where it takes the
			// shape; the FMA tiles below stay built as everything else's path.
			g.flashTileAttn(bs, kvDim, gqa, scale)
			return kernels.AttnScoresTiled(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale,
				bs.arows, bs.qtile, bs.ktile, g.kStride(p))
		}},
		{&bs.scoresW, func() (*ir.Kernel, error) {
			// Built after bs.scores so it takes the same path: a model whose
			// global layers run the matrix instruction runs its local ones on
			// it too, and a device that refused it refuses both.
			if p.SWAWindow <= 0 {
				return nil, nil
			}
			if bs.scoresMMA != nil {
				k, err := kernels.AttnScoresMMAW(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride,
					scale, bs.arows, g.kStride(p), g.tiles.attnNT, windowArg(p))
				if err == nil {
					if c, e := g.dev.Compile(k); e == nil {
						bs.scoresMMAW = c
						return nil, nil
					}
				}
				// A refusal falls through to the windowed FMA tile, which the
				// launch takes whenever scoresMMAW is nil.
			}
			return kernels.AttnScoresTiledW(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale,
				bs.arows, bs.qtile, bs.ktile, g.kStride(p), windowArg(p))
		}},
		{&bs.attnAcc, func() (*ir.Kernel, error) {
			// MLA accumulates at the latent's width (what it produces) over a
			// row-wide cache (the stride it walks), so the value is the key row's
			// prefix. No split: the split's partial plane is sized from qdim.
			if p.MLA() {
				bs.accSplit = 1
				// gqa is NHead so every head reads the row's own prefix; the fault arm
				// makes it head-major instead.
				gqa := p.NHead
				if g.kb.mlaFault == MLAFaultValueHeadMajor {
					gqa = 1
				}
				return kernels.AttnAcc(p.NHead, p.KVLoraRank, p.KVRow(), gqa, p.MaxSeq)
			}
			// The position split is a constant, not a tuned parameter: each thread's
			// trip count is ceil((n - s)/S), zero for every split past n, so one S is
			// correct at every depth. It exists because the unsplit kernel is a long
			// loop-carried chain over too few threads to fill the card at depth. S is
			// capped so the grid does not overshoot the device, and never above 32.
			if rows > 1 {
				bs.accSplit = 1
				return kernels.AttnAccTiled(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, bs.arows, bs.atile)
			}
			bs.accSplit = accSplitFor(qdim, g.dev.Slots(), g.kb.mvAccSplit)
			// WithAttnAccSplit pins it, to tell a real difference from f32
			// reassociation.
			if k := g.kb.accSplit; k >= 1 {
				bs.accSplit = k
			}
			if g.KVF16 {
				// The packed kernel handles split 1 too, so there is no
				// unsplit variant to fall back to here.
				return kernels.AttnAccSplitF16(p.NHead, p.HeadDim, kvDim, gqa, p.MaxSeq, bs.accSplit)
			}
			if bs.accSplit < 2 {
				return kernels.AttnAcc(p.NHead, p.HeadDim, kvDim, gqa, p.MaxSeq)
			}
			return kernels.AttnAccSplit(p.NHead, p.HeadDim, kvDim, gqa, p.MaxSeq, bs.accSplit)
		}},
		{&bs.attnRed, func() (*ir.Kernel, error) {
			if bs.accSplit < 2 {
				return nil, nil // no split, no reduction
			}
			return kernels.Reduce(qdim, bs.accSplit)
		}},
		{&bs.shRouterK, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			return kernels.SharedRouterLogits(rows, p.NEmbd)
		}},
		{&bs.shActMul, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			if ungatedFFN(p) {
				return kernels.Act(rows*p.NFFNShExp, p.Act)
			}
			return kernels.ActMul(rows*p.NFFNShExp, p.Act)
		}},
		{&bs.shQuant, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			return kernels.Quantize(rows*p.NFFNShExp, actWin)
		}},
		{&bs.shAdd, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			return kernels.SharedExpertAdd(p.NEmbd, rows)
		}},
		{&bs.splitQG, func() (*ir.Kernel, error) {
			if !p.AttnOutGate {
				return nil, nil
			}
			return kernels.SplitHeadGate(p.NHead, p.HeadDim, rows)
		}},
		{&bs.softcap, func() (*ir.Kernel, error) {
			if p.AttnSoftcap == 0 {
				return nil, nil
			}
			return kernels.Softcap(bs.arows*p.NHead*bs.sstride, p.AttnSoftcap)
		}},
		{&bs.clampQ, func() (*ir.Kernel, error) {
			if p.ClampKQV == 0 {
				return nil, nil
			}
			return kernels.Clamp(rows*qdim, p.ClampKQV)
		}},
		{&bs.clampKV, func() (*ir.Kernel, error) {
			if p.ClampKQV == 0 {
				return nil, nil
			}
			return kernels.Clamp(rows*kvDim, p.ClampKQV)
		}},
		{&bs.sigMul, func() (*ir.Kernel, error) {
			if !p.AttnOutGate {
				return nil, nil
			}
			return kernels.SigmoidMul(rows * qdim)
		}},
		{&bs.qkPart, func() (*ir.Kernel, error) {
			if !p.QKNorm || !p.QKNormWide || p.NKVHead != p.NHead || (p.NHead*p.HeadDim)%bs.parts != 0 {
				return nil, nil
			}
			return kernels.NormPartRows(p.NHead*p.HeadDim, bs.parts, rows)
		}},
		{&bs.qkApply, func() (*ir.Kernel, error) {
			if !p.QKNorm || !p.QKNormWide || p.NKVHead != p.NHead || (p.NHead*p.HeadDim)%bs.parts != 0 {
				return nil, nil
			}
			return kernels.NormApplyRows(p.NHead*p.HeadDim, bs.parts, float32(p.RMSEps), false, rows)
		}},
		{&bs.qkRms, func() (*ir.Kernel, error) {
			if !p.QKNorm || !p.QKNormWide || p.NKVHead != p.NHead || g.NoQKRms {
				return nil, nil
			}
			return g.rmsRows()(p.NHead*p.HeadDim, rows, float32(p.RMSEps), false, false)
		}},
		// A whole-projection norm (olmoe) is HeadNormRows with one group of the
		// whole width, each of q and k its own projection's.
		{&bs.qNorm, func() (*ir.Kernel, error) {
			if !p.QKNorm && !p.QKL2Norm {
				return nil, nil
			}
			if p.QKNorm && p.QKNormWide {
				w := p.NHead * p.HeadDim
				bs.qNormL = headLanes(w)
				return kernels.HeadNormRows(1, w, p.RMSEps, rows, bs.qNormL)
			}
			bs.qNormL = headLanes(p.HeadDim)
			return kernels.HeadNormRows(p.NHead, p.HeadDim, p.RMSEps, rows, bs.qNormL)
		}},
		{&bs.kNorm, func() (*ir.Kernel, error) {
			if !p.QKNorm && !p.QKL2Norm {
				return nil, nil
			}
			if p.QKNorm && p.QKNormWide {
				w := p.NKVHead * p.HeadDim
				bs.kNormL = headLanes(w)
				return kernels.HeadNormRows(1, w, p.RMSEps, rows, bs.kNormL)
			}
			bs.kNormL = headLanes(p.HeadDim)
			return kernels.HeadNormRows(p.NKVHead, p.HeadDim, p.RMSEps, rows, bs.kNormL)
		}},
		{&bs.actMul, func() (*ir.Kernel, error) {
			if ungatedFFN(p) {
				// Ungated: the activation alone replaces ActMul's product, which
				// would read a bs.g nothing wrote. xIELU's numbers are each
				// block's buffer (layer.xielu).
				if p.Act == kernels.ActXIELU {
					return kernels.XIELU(rows * p.NFFN)
				}
				return kernels.Act(rows*p.NFFN, p.Act)
			}
			if p.NFFN == 0 {
				return nil, nil
			}
			return kernels.ActMul(rows*p.NFFN, p.Act)
		}},
		// A second activation kernel because the element count is baked and a
		// DeepSeek has both shapes in one model (dense lead blocks, then
		// mixtures), while the scratch is one per tier.
		{&bs.actMulE, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			// One SwiGLU over all the slots at once: they are contiguous
			// and elementwise, so the slot boundary does not matter. Ungated
			// experts (Nemotron 3) take the activation alone.
			if ungatedFFN(p) {
				return kernels.Act(p.NExpertUsed*p.NFFNExp, p.Act)
			}
			return kernels.ActMul(p.NExpertUsed*p.NFFNExp, p.Act)
		}},
		{&bs.rank, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ExpertRank(routeOf(p))
		}},
		{&bs.weights, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ExpertWeights(routeOf(p))
		}},
		{&bs.pleSlice, func() (*ir.Kernel, error) {
			if p.PLEDim == 0 {
				return nil, nil
			}
			return kernels.PLESlice(rows, p.PLEWidth, p.PLEDim)
		}},
		{&bs.pleAct, func() (*ir.Kernel, error) {
			if p.PLEDim == 0 {
				return nil, nil
			}
			return kernels.ActMul(rows*p.PLEDim, p.Act)
		}},
		{&bs.pleQuant, func() (*ir.Kernel, error) {
			if p.PLEDim == 0 {
				return nil, nil
			}
			return kernels.Quantize(rows*p.PLEDim, actWin)
		}},
		{&bs.ewScale, func() (*ir.Kernel, error) {
			if !p.DenseMoE {
				return nil, nil
			}
			return kernels.ExpertWeightScale(1, p.NExpertUsed)
		}},
		{&bs.route, func() (*ir.Kernel, error) {
			r := routeOf(p)
			// The fused route renormalises or softmaxes all; sparsemixer's weights
			// are their own kernel (kernels.ExpertWeights).
			if p.NExpert == 0 || r.Gated() || r.Bias || r.Grouped() || r.SparseMixer != 0 ||
				kernels.ExpertRouteWidth(p.NExpert) == 0 {
				return nil, nil
			}
			// One warp for the renormalised route where the device has one: the
			// thread-per-expert form walks every logit. Same selection and weights
			// (kernels.ExpertRouteWarp).
			if r.Norm && bs.qkLanes == ir.SubgroupLanes {
				if k, err := kernels.ExpertRouteWarp(r); err == nil {
					bs.routeWidth = ir.SubgroupLanes
					return k, nil
				}
			}
			bs.routeWidth = kernels.ExpertRouteWidth(p.NExpert)
			return kernels.ExpertRoute(r)
		}},
		{&bs.gscore, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || p.ExpertGroups <= 1 {
				return nil, nil
			}
			return kernels.ExpertGroupScore(routeOf(p))
		}},
		{&bs.gmask, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || p.ExpertGroups <= 1 {
				return nil, nil
			}
			return kernels.ExpertGroupMask(routeOf(p))
		}},
		{&bs.rbias, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || !p.MoEBias {
				return nil, nil
			}
			return kernels.Add(p.NExpert)
		}},
		{&bs.ebiasFF, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || !p.MoEBias {
				return nil, nil
			}
			return kernels.IndexedBiasAdd(p.NExpertUsed, p.NFFNExp)
		}},
		{&bs.ebiasD, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || !p.MoEBias {
				return nil, nil
			}
			return kernels.IndexedBiasAdd(p.NExpertUsed, p.NEmbd)
		}},
		{&bs.combine, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ExpertCombine(p.ExpWidth(), p.NExpertUsed)
		}},
		{&bs.quantX, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			// The SwiGLU product, quantized per slot: Slots independent
			// vectors of NFFNExp, laid out contiguously, which is exactly what
			// MatVecShape.SlotAct then reads.
			return kernels.Quantize(p.NExpertUsed*p.NFFNExp, actWin)
		}},
		{&bs.addE, func() (*ir.Kernel, error) { return kernels.Add(rows * p.NEmbd) }},
		{&bs.addS, func() (*ir.Kernel, error) {
			if !residScaled(p) {
				return nil, nil
			}
			return kernels.AddScaled(rows * p.NEmbd)
		}},
		{&bs.scaleX, func() (*ir.Kernel, error) {
			if !p.GeomSplit {
				return nil, nil
			}
			return kernels.Scale(rows * p.NEmbd)
		}},
		// Mamba-1's: the convolution's bias and SiLU, dt's quantization, the
		// three norms (dt over the rank, B and C over the state), and the
		// whole scan in one launch.
		{&bs.m1Bias, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			return kernels.BiasAct(r.Chans, rows, kernels.ActSiLU)
		}},
		{&bs.m1Quant, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			return kernels.Quantize(rows*r.Rank, actWin)
		}},
		{&bs.m1NormDt, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			bs.m1NormDtL = headLanes(r.Rank)
			return kernels.HeadNormRows(1, r.Rank, p.RMSEps, rows, bs.m1NormDtL)
		}},
		{&bs.m1NormBC, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			bs.m1NormBCL = headLanes(r.KDim)
			return kernels.HeadNormRows(1, r.KDim, p.RMSEps, rows, bs.m1NormBCL)
		}},
		{&bs.m1Scan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Mamba1: true})
		}},
		// Mamba-2's: the whole update in one launch, its grouped norm, and the
		// copy a mixer-only block ends with.
		{&bs.ssdScan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.SSD {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans, SSD: true})
		}},
		{&bs.ssdNorm, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.SSD || r.NoNorm {
				return nil, nil
			}
			w := r.VHeads * r.VDim / r.NormGroups
			bs.ssdNormL = headLanes(w)
			return kernels.GroupNormRows(r.NormGroups, w, p.RMSEps, rows, bs.ssdNormL)
		}},
		{&bs.copyRes, func() (*ir.Kernel, error) {
			if !r.SSD && !r.Mamba1 {
				return nil, nil
			}
			return kernels.Restride(rows, p.NEmbd, p.NEmbd, p.NEmbd)
		}},
		// The linear block's kernels, each nil when the model has no such block.
		{&bs.convRows, func() (*ir.Kernel, error) {
			if r.Conv == 0 {
				return nil, nil
			}
			return kernels.Conv1dRows(r.Conv, r.Chans, rows)
		}},
		{&bs.scMul, func() (*ir.Kernel, error) {
			if !r.ShortConv {
				return nil, nil
			}
			return kernels.ActMul(rows*r.Chans, kernels.ActIdentity)
		}},
		{&bs.convShift, func() (*ir.Kernel, error) {
			if r.Conv == 0 {
				return nil, nil
			}
			return kernels.Conv1dShift(r.Conv, r.Chans, rows)
		}},
		{&bs.ssmSiLU, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.Act(rows*r.Chans, kernels.ActSiLU)
		}},
		{&bs.splitGates, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SplitDeltaGatesRows(r.KHeads, r.VHeads/r.KHeads, rows)
		}},
		{&bs.deltaGate, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.DeltaGateRows(r.VHeads, rows)
		}},
		// KDA's gate kernels: two launches because decay runs over VHeads*KDim
		// channels and beta over VHeads heads. DeltaGate computes both, so each
		// launch keeps the half it wants and sends the other to dGateWaste.
		{&bs.deltaGateChan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !r.ChanDecay {
				return nil, nil
			}
			if r.DecayBound != 0 {
				// Kimi-K3's lower-bounded decay, the same params.
				return kernels.DeltaDecayBoundRows(r.VHeads*r.KDim, rows, r.DecayBound)
			}
			return kernels.DeltaGateRows(r.VHeads*r.KDim, rows)
		}},
		{&bs.quantLR, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !r.ChanDecay {
				return nil, nil
			}
			return kernels.Quantize(rows*r.KDim, actWin)
		}},
		{&bs.deltaStepChan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !r.ChanDecay {
				return nil, nil
			}
			return kernels.GatedDeltaStepChan(r.VHeads, r.VDim, r.KDim, r.VHeads/r.KHeads)
		}},
		{&bs.outSigMul, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !(r.ChanDecay || r.SigmoidGate) {
				return nil, nil
			}
			// SigmoidMul, not ActMul: KimiLinearRMSNormGated applies out*sigma(z)
			// where qwen3next applies out*SiLU(z), which differ by a factor of z.
			return kernels.SigmoidMul(rows * r.VHeads * r.VDim)
		}},
		{&bs.sliceQ, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SliceRows(r.KHeads*r.KDim, r.Chans, 0, rows)
		}},
		{&bs.sliceK, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SliceRows(r.KHeads*r.KDim, r.Chans, r.KHeads*r.KDim, rows)
		}},
		{&bs.sliceV, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SliceRows(r.VHeads*r.VDim, r.Chans, 2*r.KHeads*r.KDim, rows)
		}},
		{&bs.qkNorm, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			// The L2 norm is an RMSNorm with a unit weight and eps/n, one launch per
			// operand. KDA's l2norm hardcodes eps 1e-6 added to the sum (not
			// rms_norm_eps); dividing by KDim turns it into the mean form.
			// engine/model/delta.go's kdaL2Eps is the same number.
			eps := p.RMSEps
			if r.ChanDecay {
				eps = 1e-6
			}
			bs.qkNormL = headLanes(r.KDim)
			return kernels.HeadNormRows(r.KHeads, r.KDim, eps/float64(r.KDim),
				rows, bs.qkNormL)
		}},
		{&bs.qkScale, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.Scale(rows * r.KHeads * r.KDim)
		}},
		{&bs.deltaStep, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			if r.KeyTiled && r.TailLen > 0 {
				return kernels.GatedDeltaStepTiledStride(r.VHeads, r.VDim, r.KDim, r.KHeads, r.StateLen)
			}
			if r.KeyTiled {
				return kernels.GatedDeltaStepTiled(r.VHeads, r.VDim, r.KDim, r.KHeads)
			}
			return kernels.GatedDeltaStep(r.VHeads, r.VDim, r.KDim, r.VHeads/r.KHeads)
		}},
		// The chunk's delta rule: GatedDeltaStep steps one token, so a batched
		// submission needs the scan. Decode takes it too where the device has a
		// subgroup: the step is a thread per state row, so warp loads are
		// uncoalesced, while the scan puts L lanes on a row. Without a subgroup the
		// scan's one-thread form holds the whole row in registers, so decode keeps
		// the step.
		{&bs.deltaFused, func() (*ir.Kernel, error) {
			// Decode only: the fused kernel forms q, k, v and the gates once per
			// state row inside the per-token loop. At one row that beats eleven
			// launches; over a chunk it is serial work on the scan's critical path
			// and measured slower.
			if r.Conv == 0 || r.ShortConv || r.Mamba1 || rows != 1 || bs.qkLanes != ir.SubgroupLanes {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			if d.ScanLanes() == 1 {
				return nil, nil
			}
			// The two constants the unfused chain bakes: qkNorm's epsilon over
			// KDim (HeadNormRows' own argument, KDA's hardcoded 1e-6) and the
			// 1/sqrt(KDim) dScale holds.
			eps := p.RMSEps
			if r.ChanDecay {
				eps = 1e-6
			}
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans,
				Eps: float32(eps / float64(r.KDim)), Scale: float32(1 / math.Sqrt(float64(r.KDim))),
				BARep: r.VHeads / r.KHeads, Bound: r.DecayBound})
		}},
		{&bs.deltaScan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 || bs.deltaFused != nil {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			if rows == 1 && d.ScanLanes() == 1 {
				return nil, nil
			}
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaScan(d)
		}},
		{&bs.outNorm, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			bs.outNormL = headLanes(r.VDim)
			return kernels.HeadNormRows(r.VHeads, r.VDim, p.RMSEps, rows, bs.outNormL)
		}},
		{&bs.outActMul, func() (*ir.Kernel, error) {
			if r.Conv == 0 {
				return nil, nil
			}
			return kernels.ActMul(rows*r.VHeads*r.VDim, kernels.ActSiLU)
		}},
	}
	for _, j := range altJobs(bs, p, rows, actWin) {
		jobs = append(jobs, job{j.dst, j.make})
	}
	// Gemma 4's clipped linears: matrix s's input clamp and output clamp at
	// its widths, rows of each.
	if p.Clamps {
		in := [7]int{p.NEmbd, p.NEmbd, p.NEmbd, qdim, p.NEmbd, p.NEmbd, p.NFFN}
		out := [7]int{qdim, kvDim, kvDim, p.NEmbd, p.NFFN, p.NFFN, p.NEmbd}
		for s := range 7 {
			ni, no, off := rows*in[s], rows*out[s], 4*s
			jobs = append(jobs,
				job{&bs.clampK[2*s], func() (*ir.Kernel, error) { return kernels.ClampAt(ni, off) }},
				job{&bs.clampK[2*s+1], func() (*ir.Kernel, error) { return kernels.ClampAt(no, off+2) }})
		}
	}
	// The contiguous cache's writers and readers, which paged history
	// replaces (initPagedScratch).
	pagedSkip := map[*backend.Kernel]bool{&bs.scores: true, &bs.scoresW: true, &bs.attnAcc: true,
		&bs.attnRed: true, &bs.softcap: true, &bs.copyK: true, &bs.ropeK: true, &bs.copyV: true}
	for _, j := range jobs {
		if paged && pagedSkip[j.dst] {
			continue
		}
		k, e := j.make()
		if e != nil {
			g.LastErr = e.Error()
			return nil
		}
		if k == nil {
			continue // an optional kernel this architecture does not need
		}
		c, e := g.compileShared(k)
		if e != nil {
			g.LastErr = e.Error()
			return nil
		}
		*j.dst = c
	}
	if paged {
		if err := g.initPagedScratch(bs); err != nil {
			g.LastErr = "paged attention: " + err.Error()
			freeScratch(bs)
			return nil
		}
		if err := g.initDS4(bs); err != nil {
			g.LastErr = "DeepSeek V4: " + err.Error()
			freeScratch(bs)
			return nil
		}
	} else {
		if !g.pickSoftmax(bs) {
			return nil
		}
		if err := g.initFlash(bs); err != nil {
			// Compilation failure retains the staged implementation.
			g.LastErr = err.Error()
		}
	}
	bs.hx = make([]float32, rows*max(wide, p.ResidW()))
	bs.hcs = make([]float32, rows*p.RopeW())
	if p.SWAPeriod > 0 {
		bs.hcsSWA = make([]float32, rows*p.RopeW())
	}
	bs.hraw = make([]byte, rows*p.ResidW()*4)
	bs.hout = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hraw[0])), rows*p.ResidW())
	if rows > 1 && p.NExpert > 0 {
		bs.mg = g.newMoeGroup(p, rows)
	}
	if err := g.initK3(bs); err != nil {
		g.LastErr = "Kimi-K3: " + err.Error()
		freeScratch(bs)
		return nil
	}
	if err := g.initQ4E(bs); err != nil {
		g.LastErr = "Qwen3.8-Flash-Next: " + err.Error()
		freeScratch(bs)
		return nil
	}
	if rows > 1 && p.MLA() {
		bs.mlab = g.newMLABatch(bs, p, rows)
	}
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

// pickSoftmax takes the warp-per-head softmax when this device guarantees a
// 32-lane subgroup and the kernel then agrees with the host, and the
// one-thread-per-head kernel otherwise.
//
// It asks first (pickLanes) and probes second. The probe alone is not enough:
// an out-of-subgroup shuffle is undefined, and a device that picks its width
// per shader can pass on an idle device and produce NaN in a real submission
// (see docs/engineering-history/gpu-kernels.md). The probe stays because it
// catches a driver that promised a width and did not deliver, a shuffle
// lowered wrongly, and a deterministically narrow device. A device that
// refuses to compile the shuffle lands in the same fallback.
func (g *devTier) pickSoftmax(bs *blockScratch) bool {
	p := &bs.p
	// Every row of a chunk is an independent set of score rows, so a batched
	// softmax is simply nHeads*rows of them -- the kernel already takes the row
	// count as its first argument and needs no other change.
	heads := p.NHead * bs.arows
	build := func(lanes int) backend.Kernel {
		k, err := kernels.SoftmaxRowsSink(p.NHead, bs.sstride, lanes, bs.arows, bs.atile, p.AttnSinks)
		if err != nil {
			g.LastErr = err.Error()
			return nil
		}
		c, err := g.compileShared(k)
		if err != nil {
			g.LastErr = err.Error()
			return nil
		}
		return c
	}
	scalarGroups := (heads + scalarWidth - 1) / scalarWidth

	// The batched scratch inherits the decision rather than re-probing: the
	// subgroup width is a property of the device, and the rows==1 scratch is
	// always built first, so g.SoftmaxLanes is settled.
	if bs.rows > 1 {
		lanes, groups, width := g.SoftmaxLanes, heads, ir.SubgroupLanes
		if lanes != ir.SubgroupLanes {
			lanes, groups, width = 1, scalarGroups, scalarWidth
		}
		c := build(lanes)
		if c == nil {
			return false
		}
		bs.softmax, bs.smGroups, bs.smWidth = c, groups, width
		return true
	}

	// The wide kernel is the default on every device that guarantees the width;
	// WithLanes(1) forces the one-thread kernel. (An apparent Metal bug here was
	// the harness comparing two free-running greedy chains; diff at a fixed input,
	// see docs/engineering-history/gpu-kernels.md.)
	warpOK := g.pickLanes() == ir.SubgroupLanes
	g.SoftmaxWhy = "the warp reduction did not agree with the host"
	if !warpOK {
		g.SoftmaxWhy = g.LanesWhy
	}
	if c := build(ir.SubgroupLanes); warpOK && c != nil {
		// One workgroup per head, one warp wide: the group IS the warp, so the
		// kernel's lane == tid holds without assuming how a larger group is cut
		// into subgroups.
		if g.softmaxAgrees(bs, c, heads, 32) {
			bs.softmax, bs.smGroups, bs.smWidth = c, heads, ir.SubgroupLanes
			g.SoftmaxLanes = ir.SubgroupLanes
			bs.softmaxAlt = build(1)
			bs.smAltGroups, bs.smAltWidth = scalarGroups, scalarWidth
			return true
		}
		closeKern(c)
	} else if c != nil {
		closeKern(c)
	}
	c := build(1)
	if c == nil {
		return false
	}
	bs.softmax, bs.smGroups, bs.smWidth = c, scalarGroups, scalarWidth
	g.SoftmaxLanes = 1
	return true
}

// scalarWidth is the launch width of every thread-per-head kernel in the
// subgroup-width class: the softmax twin and the head-norm twin. It is a plain
// grid width with no cross-lane meaning at all, which is the whole point of
// those kernels.
const scalarWidth = 64

// headNormGeom is the launch of a head norm built at lanes over heads heads:
// one 32-lane group per head, or one thread per head in scalarWidth groups. It
// follows the kernel's lanes, not the device's: headLanes may drop a head to the
// scalar twin, and msl lowers NTID as the declared 64, so a 32-wide launch of
// it would leave half the heads unwritten.
func headNormGeom(lanes, heads int) (groups, width int) {
	if lanes == 1 {
		return (heads + scalarWidth - 1) / scalarWidth, scalarWidth
	}
	return heads, lanes
}

// The softmax is picked by correctness, not speed. A load-time timing probe
// runs at pos=1, where the choice does not matter; the warp reduction's gain
// grows with context. `verify -ab softmax` at a long -n is how it is checked.

// softmaxAgrees launches k on a synthetic score matrix and compares against a
// float64 softmax on the host.
//
// The scores span several units so that exp() separates them, and each head's
// maximum sits at a different position -- a reduction that sees only one lane's
// stride of the row picks the wrong maximum and normalises by the wrong sum,
// which shows up as an error of order 1, not of order 1e-6. n is deliberately
// not a multiple of 32: it exercises both the lanes that own two positions and
// the lanes that own none.
func (g *devTier) softmaxAgrees(bs *blockScratch, k backend.Kernel, groups, width int) bool {
	// Every n the first tokens use: at n=1 a single lane owns the whole row, a
	// different path through the reduction than a long row. 31, 32 and 33
	// straddle the warp boundary.
	for _, n := range []int{1, 2, 31, 32, 33, 37} {
		if n > bs.p.MaxSeq {
			continue
		}
		if !g.softmaxAgreesAt(bs, k, groups, width, n) {
			return false
		}
	}
	return true
}

func (g *devTier) softmaxAgreesAt(bs *blockScratch, k backend.Kernel, groups, width, n int) bool {
	p := &bs.p
	scores := make([]float32, p.NHead*p.MaxSeq)
	for h := 0; h < p.NHead; h++ {
		for t := 0; t < n; t++ {
			scores[h*p.MaxSeq+t] = float32(4 * math.Sin(0.7*float64(t)+float64(h)))
		}
	}
	got := make([]float32, p.NHead*p.MaxSeq)
	// A sink row that sits inside the scores' range, so it moves both the
	// maximum and the denominator.
	sinks := make([]float32, p.NHead)
	for h := range sinks {
		sinks[h] = float32(3 * math.Cos(float64(h)))
	}
	var err error
	g.dev.Session(func(s backend.Session) {
		if err = s.Write(bs.att, f32b(scores)); err != nil {
			return
		}
		if err = s.Write(bs.n, u32one(uint32(n))); err != nil {
			return
		}
		bufs := []backend.Buf{bs.att, bs.n, bs.prob}
		if p.AttnSinks {
			if err = s.Write(bs.sinkProbe, f32b(sinks)); err != nil {
				return
			}
			bufs = append(bufs, bs.sinkProbe)
		}
		if err = s.Launch(k, groups, width, bufs...); err != nil {
			return
		}
		err = s.Read(bs.prob, f32b(got))
	})
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	for h := 0; h < p.NHead; h++ {
		row := scores[h*p.MaxSeq : h*p.MaxSeq+n]
		mx := math.Inf(-1)
		if p.AttnSinks {
			mx = float64(sinks[h])
		}
		for _, v := range row {
			if float64(v) > mx {
				mx = float64(v)
			}
		}
		e := make([]float64, n)
		var sum float64
		if p.AttnSinks {
			sum = math.Exp(float64(sinks[h]) - mx)
		}
		for i, v := range row {
			e[i] = math.Exp(float64(v) - mx)
			sum += e[i]
		}
		for i := range e {
			// 1e-4 is loose against float32 and an approximate device exp, and
			// tight against every way a cross-lane reduction can be wrong.
			if math.Abs(float64(got[h*p.MaxSeq+i])-e[i]/sum) > 1e-4 {
				return false
			}
		}
	}
	return true
}

// headOut is the buffer the head's answer is read from: the capped logits
// where the model has a final softcap, the projection's own otherwise.
func (bs *blockScratch) headOut() backend.Buf {
	if bs.headCap != nil {
		return bs.logitsCap
	}
	return bs.logits
}

// ragOut is headOut for the per-row head: the rows' capped logits where the
// model has a final softcap, the projection's own otherwise.
func (bs *blockScratch) ragOut() backend.Buf {
	if bs.ragCapK != nil && !rowsFaulted("nocap") {
		return bs.ragCap
	}
	return bs.ragLogits
}

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
	// A head that was dropped (its resident failed) replaces the cap it
	// compiled, and must not leak the one before.
	if bs.headCap != nil {
		bs.headCap.Close()
		bs.headCap = nil
	}
	if bs.logitsCap != nil {
		bs.logitsCap.Free()
		bs.logitsCap = nil
		g.refund(uint64(bs.mvHead.rows * 4))
	}
	// So does a PrepHead after a seam gave its blocks back and grew again: the
	// norm, its bias and the logits are allocated there, and overwriting them
	// leaked all three (and their charge) on every round trip. bs.head is set
	// only where those charges were made.
	if bs.head != nil {
		n := uint64(bs.p.NEmbd*4 + bs.mvHead.rows*4)
		if bs.hNormB != nil {
			n += uint64(bs.p.NEmbd * 4)
		}
		g.refund(n)
	}
	for _, b := range []*backend.Buf{&bs.hNorm, &bs.hNormB, &bs.logits} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	bs.head = nil
	g.freeAltNorms()
}

// PrepHead uploads the output norm and the vocabulary projection.
func (g *devTier) PrepHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	bs := g.bs
	if bs == nil || h == nil {
		return false
	}
	// The head is the model's: a second State on this tier finds it placed.
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
	// Under paging the projection and a block slot compete for the same bytes,
	// and the head is permanent while a slot turns over every block. The head
	// runs once per token and a block NLayer times, so the projection is taken
	// only when it costs no slot.
	if len(g.stream) > 0 {
		want := uint64(len(h.W.Data))
		if now, after := g.slots(), g.slotsIfPerm(want); after < now {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and "+
				"taking them would leave %d block slot(s) instead of %d, so it "+
				"stays on the host: a slot turns over every block and the "+
				"projection runs once a token", want, after, now)
			return false
		}
	}
	// -1: the output projection is not a block, so it is not released with one.
	r := g.resident(-1, 0, q, h.W.Data, h.W.Rows, h.W.K, h.W.Packed)
	if r == nil || !r.ok {
		return false
	}
	split := g.tuneSplit(q, h.W.Rows, h.W.K, r)
	// HeadSplit pins the head's split alone, leaving every block matvec on the
	// tuner: the head's row count is far beyond any block matvec's, and a global
	// split cannot test it in isolation.
	if g.HeadSplit > 0 {
		split = g.HeadSplit
	}
	// The output projection has no bias in any architecture here; false is a
	// statement about nn.Head, not a default.
	kern := g.kernel(q, h.W.K, h.W.Rows, split, 1, false)
	if kern == nil {
		return false
	}
	var red backend.Kernel
	if split > 1 {
		if red = g.reduceKernel(h.W.Rows, split); red == nil {
			split = 1
			if kern = g.kernel(q, h.W.K, h.W.Rows, 1, 1, false); kern == nil {
				return false
			}
		}
	}
	if !g.sizePart(h.W.Rows * split) {
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
	} else if bs.copyRows == nil {
		g.LastErr = "a head with no norm, on a scratch with no copy for it"
		return false
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
	bs.head, bs.mvHead = r, mv{kern: kern, red: red, rows: h.W.Rows, split: split}
	g.shapeChanged()
	g.headSrc, g.headNormHost = &h.W.Data[0], slices.Clone(h.Norm)
	g.charge(uint64(len(h.Norm)*4 + h.W.Rows*4))
	bs.headEmbeds, bs.embScale = h.Embeds, h.EmbdScale
	if bs.embScale == 0 {
		bs.embScale = 1
	}
	// The final softcap, which the contract on nn.Head.Softcap makes this
	// tier's to apply once it accepts the head -- so a failure here is a
	// refusal of the head, not a head without its cap.
	if h.Softcap != 0 {
		ck, e := kernels.Softcap(h.W.Rows, h.Softcap)
		if e != nil {
			return false
		}
		c, e := g.dev.Compile(ck)
		if e != nil {
			return false
		}
		b, e := g.dev.Alloc(h.W.Rows * 4)
		if e != nil {
			closeKern(c)
			return false
		}
		bs.headCap, bs.logitsCap = c, b
		g.charge(uint64(h.W.Rows * 4))
	}
	// The greedy token on the device (nn.Head.ArgmaxOnly). Without it the head
	// still runs and every caller gets its logits, so a failure is not one.
	if g.argmaxK == nil {
		if ak, e := kernels.Argmax(h.W.Rows); e == nil {
			if c, e := g.dev.Compile(ak); e == nil {
				w := g.scratchWin() // the device's own word (scratch.go)
				if b, e := g.dev.Alloc(4); e == nil {
					g.argmaxK, g.argmaxOut = c, b
				} else {
					closeKern(c)
				}
				w.close()
			}
		}
	}
	g.prewarmHeadKernels()
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
		// The windows staged in the set are whichever session's picture
		// came last: this session's go in again.
		wok := g.setWindowsLocked(g.winRuns, g.winPerRow) && g.setSeqsLocked(g.seqRuns)
		g.mu.Unlock()
		if !wok {
			return false
		}
		if bs == nil || n > bs.rows || head != nil {
			g.LastErr = fmt.Sprintf("non-causal n=%d scratch=%v head=%v", n, bs != nil, head != nil)
			return false
		}
		return g.submit(sid, bs, lo, hi, pos, x, cs, csSWA, nil)
	}
	if n == 1 {
		g.retuneDecode()
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
		// Config.NoBatch makes the caller go one device call per prompt token:
		// the way to tell a batching bug from an f32 tie, and the A/B control.
		if g.NoBatch {
			g.LastErr = "batched prefill disabled"
			return false
		}
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
		ww := 0
		if !g.NoExactPrompt && w != bw && g.keepExact(w) {
			ww = w
			g.ExactPrompts++
		}
		if ww == 0 {
			ww = g.batchFor(w)
		}
		if ww == 0 && w != bw && g.prepBatch(bw) {
			ww = bw
		}
		ok := ww > 0
		if ok {
			w = ww
		}
		bbs, dbs, sub := g.bbs[w], g.bs, max(g.promptW, nn.MaxPrefillChunk)
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
			(kernels.IsFloat(dbs.head.t) || dbs.p.NEmbd != bbs.p.NEmbd)
		g.mu.Unlock()
		if head != nil && head.RowLogits && (!ok || bbs == nil) {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("a %d-row chunk wanting every row's logits has no batched scratch", n)
			}
			return false
		}
		// Every fallback below cuts x by rows, which a stream-major residual
		// (AltUp's, DeepSeek V4's) is not: such a chunk is refused whole, and
		// the State runs it on the host.
		if dbs != nil && dbs.p.Streams() > 1 && (head != nil && noFold || !ok || bbs == nil) {
			g.LastErr = fmt.Sprintf("a %d-row chunk of %d streams this device would have to cut by rows", n,
				dbs.p.Streams())
			return false
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
// position, the KV offset, the rotary table) reaches the device inside a
// buffer. The only grid that changes is the attention scores', over pos+1
// positions.
type graphKey struct {
	lo, hi   int
	scoreN   int // the rounded position count the scores grid was sized for
	rows     int // the batch width; 1 for decode
	head     bool
	tblSplit bool
	// The softmax arm: the two kernels' grids differ, and verify -ab softmax
	// flips it every round.
	scalarSM bool
	flash    bool
	// The packed V cache changes copyV's grid and AttnAcc's kernel entirely, so
	// a recording made under one arm cannot replay under the other.
	kvF16 bool
	// Whether the rotary table is built on the device, which adds launches at
	// the head of the sequence; verify -ab ropetab flips it every round.
	ropeDev bool
	// argmax says the head ends in the device argmax (nn.Head.ArgmaxOnly):
	// one more launch, so a recording without it cannot serve a call with it.
	argmax bool
	// sampleK is the device sampler's k (nn.Head.SampleK), 0 for none: its
	// launches follow the head, so a recording without them cannot serve a
	// call with them, nor one of another k.
	sampleK int
	// samplePen says the sampler's penalty launch runs: with no history it
	// is skipped and the first top-k pass reads the logits.
	samplePen bool
	// pick says the head ends in the label pick's gather (nn.Head.Pick).
	pick bool
	// rag is a LayersRows step's sequence count, 0 for any other call: ragged
	// attention and a per-row head, and on a hybrid the per-count linear
	// kernels (ragLin), which a recording names by address.
	rag int
	// ragRuns is how many runs a hybrid's ragged step groups its rows into:
	// the linear kernels' grids are sized by it (ragLinear.perRun).
	ragRuns int
	// The recurrent parity: a linear block's state deliberately moves between
	// two halves, so a recording names one as source and the other as
	// destination. Replayed at the other parity it would read the stale half
	// and the state would never advance. Two recordings alternate instead,
	// as a partial seam alternates two lo/hi keys. One bit per linear block
	// (rangeParity).
	recParity parityMask
	// pagedVar is the paged attention's variant: the decode plan a recording
	// launched, which the keys' count chose and nCap may not.
	pagedVar *pagedVariant
	// ragHead is how many of a ragged step's rows the head projects
	// (nn.Head.LogitRows): it picks the head's matvec, which a recording
	// names, and a step of the same rows wanting fewer logits differs only
	// here.
	ragHead int
	// hnorm is which norm the head reads (headNormFor): a recording names its
	// buffer by address.
	hnorm int
	// persist says the recording holds composed persistent kernels
	// (Config.Persist), which a per-op recording does not.
	persist bool
	// persistDeps says those kernels are ordered by dependency counters
	// (Config.PersistDeps) rather than grid barriers.
	persistDeps bool
	// gateUp says the dense FFNs ran gate and up as one launch
	// (Config.GateUpFuse).
	gateUp bool
	// sid is the session the recording was made for: it bakes that session's
	// page descriptors and recurrent seats, so another session never replays
	// it. Keying on it, rather than dropping every recording when the current
	// session changes, is what lets two sessions that take turns -- a
	// speculator's trunk and draft, two requests stepped one after another --
	// each keep replaying instead of recording again at every turn.
	sid uint64
	// noFuse is Config.NoFuse: the fused kernels and their parts are two
	// launch sequences, and verify -ab fuse flips it every round.
	noFuse bool
	// noActFuse is Config.NoActFuse, verify -ab actfuse's switch.
	noActFuse bool
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
	// change -- this session's pages, whether any are at home, the blocks --
	// into the submission's descriptors, and a wait after it would let one of
	// them change under the submission (another session's page-in or
	// eviction). Alone on the device: wait for every submission in flight.
	// Beside the others: wait for a quiesce in progress to have its turn. From
	// here to the submission nothing lets g.mu go (prologueHeld).
	// The question is asked again after every wait: another session's
	// prologue may have sent this session's history home meanwhile.
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
	// seam. Every refusal records why, since the State's answer to false is a
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
	// there, and only the logits come home -- no second session, no readback of
	// the chunk.
	// A chunk whose head wants every row's logits (nn.Head.RowLogits: a
	// speculative verification) runs the ragged step's per-row head over the
	// chunk instead of folding its last row; prepRowsHead built it.
	allRows := bs.rows > 1 && head != nil && rag == nil && head.RowLogits
	fold := bs.rows > 1 && head != nil && rag == nil && !allRows
	if allRows && (bs.ragHead.kern == nil || bs.ragLogits == nil) {
		return refuse("a %d-row chunk asked for every row's logits with no per-row head built", bs.rows)
	}
	// The per-row head caps where decode's does: a step's rows end in an
	// argmax and a readback of their own, and uncapped they would be a
	// different function from the session's decode.
	if (rag != nil || allRows) && head != nil && g.bs != nil && g.bs.headCap != nil && bs.ragCapK == nil {
		return refuse("a %d-row step of a model with a final logit softcap, with no per-row softcap built", bs.rows)
	}
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
		case kernels.IsFloat(hb.head.t) || hb.p.NEmbd != bs.p.NEmbd:
			return refuse("a batched submission of %d rows with a %v head", bs.rows, hb.head.t)
		}
		if e := g.spanKernel(hb); e != nil {
			return refuse("the folded head's row copy: %v", e)
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
		// A latent block without a batched scratch refuses the submission, not
		// the block: Layers then runs the chunk's rows one at a time on this
		// device, keeping decode here. The batched arm is emitMLARows.
		if l.mla && bs.rows > 1 && bs.mlab == nil {
			g.LastErr = fmt.Sprintf("block %d: multi-head latent attention has no batched "+
				"scratch here, so a %d-row chunk goes one row at a time", li, bs.rows)
			return false
		}
		// A ragged step's rows are different sequences, or different positions
		// of one, each reading its own history through its descriptor: only the
		// paged latent cache has one (the contiguous cache is one sequence's).
		if l.mla && rag != nil && bs.pkv == nil {
			return refuse("block %d: a ragged step over a latent block needs the paged cache", li)
		}
		// A windowed vision block over the whole image is fluent and wrong, so
		// it runs only under windows that cover exactly this image's rows.
		if l.windowed && (bs.winMask == nil || bs.vwinRows != len(x)/bs.p.NEmbd) {
			return refuse("block %d attends inside windows and the device holds windows for %d "+
				"rows, not %d (SetKeyRuns)", li, bs.vwinRows, len(x)/bs.p.NEmbd)
		}
		// Packed sequences over other rows than the call's would let a row
		// read another sequence's keys.
		if l.nonCausal && bs.vsegRows != 0 && bs.vsegRows != len(x)/bs.p.NEmbd {
			return refuse("block %d: the device holds packed sequences over %d rows, not %d (SetSeqRuns)",
				li, bs.vsegRows, len(x)/bs.p.NEmbd)
		}
	}
	g.Submits++
	p := &bs.p
	// The cache row; see initScratch.
	kvDim := p.KVRow()
	qdim := p.NHead * p.HeadDim

	// Which softmax this call runs, resolved once rather than per block.
	sm, smG, smW := bs.softmax, bs.smGroups, bs.smWidth
	if g.ScalarSoftmax && bs.softmaxAlt != nil {
		sm, smG, smW = bs.softmaxAlt, bs.smAltGroups, bs.smAltWidth
	}

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
	var gatherN int
	if pe != nil {
		hr := g.bs.head
		gatherN = kernels.GetRowsThreads(hr.t, hr.k, R)
		if !g.embedKernel(g.bs) || hr.k != p.NEmbd || len(pe.ids) > R || gatherN%128 != 0 {
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
	kst := g.kStride(p)
	koffs := g.subBytes.koffs[:0]
	kposs := g.subBytes.kposs[:0]
	for r := 0; r < R; r++ {
		at := pos + r
		if r >= nrow {
			at = p.MaxSeq // the trash slot; see PrepLayer
		}
		koffs = append(koffs, u32one(uint32(at*kvDim))...)
		kposs = append(kposs, u32one(uint32(at))...)
	}
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
	if len(g.bidir) > 0 && !p.NonCausal {
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
		pn, koffs, kposs = pn[:4], koffs[:0], kposs[:0]
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
			koffs = append(koffs, u32one(0)...)
			kposs = append(kposs, u32one(0)...)
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
	var pagedVar *pagedVariant
	if pk := bs.pkv; pk != nil {
		pk.rows = g.pagedRows(sid, pk.rows, R, nrow, pos, rag)
		if e := g.pagedPrep(sid, bs, pk.rows); e != nil {
			return refuse("paged attention: %v", e)
		}
		pk.st = nil
		if pk.cur != nil {
			g.pagedStage(pk, p, pk.rows)
			// A call over evicted history streams (pagedstream.go), through
			// the staged decode kernels rather than the prefill ones.
			pf := pk.pf != nil && rag == nil && R > 1 && nrow > 0
			if e := g.pagedStreamPrep(sid, bs, pk.rows, pf); e != nil {
				return refuse("paged attention: %v", e)
			}
			// The lightning indexer scores every position through the
			// table, which a streamed pass does not hold.
			if pk.st != nil && pk.idx != nil {
				return refuse("the lightning indexer reads the whole history and part of it is evicted")
			}
			// So do MiniMax-M3's and QSA's block selections (msa.go, qsa.go).
			if pk.st != nil && (pk.msa != nil || pk.qsa != nil) {
				return refuse("the block selection reads the whole history and part of it is evicted")
			}
		}
		pk.prefill = pk.pf != nil && pk.cur != nil && pk.st == nil && rag == nil && R > 1 && nrow > 0
		if pk.prefill {
			if e := g.prefillPrep(bs, nrow); e != nil {
				return refuse("paged prefill attention: %v", e)
			}
		}
		pagedVar = pk.cur
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
	ropeDev := bs.ropeTable != nil && !g.RopeTableHost && rrows >= 1 &&
		pos >= 0 && top <= kernels.RopeTabMaxPos
	// QSA's marks, each row's pictures (qsa.go).
	if bs.pkv != nil {
		if e := g.qsaStage(bs.pkv.qsa, R); e != nil {
			return refuse("QSA: %v", e)
		}
	}
	// DeepSeek V4's entries rotate at positions of their own, which only the
	// device's table builds; its keys are read through the paged pool.
	if bs.ds4 != nil && lo < hi {
		pk := bs.pkv
		switch {
		case !ropeDev:
			return refuse("DeepSeek V4's entries rotate at positions of their own, which needs the " +
				"device's rotary table")
		case pk == nil || pk.cur == nil:
			return refuse("DeepSeek V4 reads its keys through the paged pool, and this call holds none")
		case pk.st != nil:
			return refuse("DeepSeek V4's window and entries are read through their tables and part " +
				"of the history is evicted")
		}
		if e := g.ds4Stage(bs, lo, hi, pk.rows); e != nil {
			return refuse("%v", e)
		}
	}
	var rposs []byte
	if ropeDev {
		rposs = append(g.subBytes.rposs[:0], u32one(uint32(rrows))...)
		for r := 0; r < R; r++ {
			at := pos + r
			if r >= rrows {
				// Never read (the kernel clamps at rrows), but kept inside the
				// range the decomposition covers.
				at = pos
			} else if rag != nil {
				at = rag.pos[r]
			}
			rposs = append(rposs, u32one(uint32(at))...)
		}
		g.RopeTables += rrows
		if bs.csSWA != nil {
			g.RopeTables += rrows
		}
	} else if bs.cs != nil {
		g.RopeTableUploads += R
	}

	// Retire recordings before the session opens: freeing one is a driver call
	// that cannot nest inside a session. See dropGraph.
	//
	// Retire on nCap, keep on lo/hi. nCap only increases, so a key differing
	// only there is dead once a longer one appears; lo/hi/head is a placement,
	// and a partial seam alternates two of them every token.
	argmax := head != nil && head.ArgmaxOnly && g.argmaxK != nil && R == 1
	sampleK, samplePen := 0, false
	if head != nil && !argmax && head.SampleK > 0 && R == 1 && rag == nil && !allRows && !fold && g.prepSample(head) {
		sampleK, samplePen = head.SampleK, head.SampleArgs[0] > 0
	}
	// The label pick: one row's logits, a decode's or a folded chunk's last.
	pick := head != nil && !argmax && sampleK == 0 && len(head.Pick) > 0 && rag == nil && !allRows &&
		(R == 1 || fold) && g.prepPick(head)
	ragN, ragRuns, ragHead := 0, 0, 0
	if rag != nil {
		ragN, ragRuns = len(rag.pos), len(rag.runs)
		if head != nil {
			ragHead = head.WantedRows(ragN)
		}
	}
	// Counted here rather than in emit, which a replayed recording skips.
	if ragHead == 1 {
		if _, ok := g.headOneMV(R); ok {
			g.RagHeadOne++
		}
	}
	par, parOK := g.rangeParity(sid, lo, hi)
	if allRows {
		ragHead = head.WantedRows(nrow)
	}
	// The persistent path: decode only, and not over history streamed from
	// home, whose uploads sit between launches.
	persistOn := R == 1 && rag == nil && !(bs.pkv != nil && bs.pkv.st != nil) && g.ensurePersist()
	key := graphKey{lo, hi, nCap, R, head != nil, g.TableSplit, g.ScalarSoftmax, g.flashOn(), g.KVF16,
		ropeDev, argmax, sampleK, samplePen, pick, ragN, ragRuns, par, pagedVar, ragHead, hNormID, persistOn, persistOn && g.PersistDeps, g.GateUpFuse, sid, g.NoFuse, g.NoActFuse}
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

	g.subBytes.pn, g.subBytes.koffs, g.subBytes.kposs = pn, koffs, kposs
	if rposs != nil {
		g.subBytes.rposs = rposs
	}
	t0 := time.Now()
	g.sub = submitArgs{
		bs: bs, hb: hb, head: head, lo: lo, hi: hi, pos: pos, x: x,
		hx: hx, hcs: hcs, hcsSWA: hcsSWA, pn: pn, koffs: koffs, kposs: kposs, rposs: rposs,
		p: p, rag: rag, pe: pe, key: key, R: R, nrow: nrow, kvDim: kvDim, qdim: qdim,
		kst: kst, nCap: nCap, gatherN: gatherN, rrows: rrows, sm: sm, smG: smG, smW: smW,
		fold: fold, ropeDev: ropeDev, argmax: argmax, sampleK: sampleK, samplePen: samplePen, pick: pick, ragHead: ragHead, parOK: parOK,
		allRows: allRows, hNorm: hNorm,
	}
	if g.subFn == nil {
		g.subFn = g.layersSession
	}
	// Read here, under g.mu: a page-in another session makes changes it.
	g.sub.paging = g.pagesIn < len(g.layers)
	g.sub.persist = persistOn
	g.sessQueue()
	// The prologue streams this call's history only when some of it was at
	// home before the prologue began (evictedFor), which made the call one
	// to run alone; nothing can have gone home since, so this is a check of
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
		return false
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
	// rows it wrote are history eviction may send home.
	if pk := bs.pkv; pk != nil && pk.cur != nil && g.kvp != nil {
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
	persist                 bool
	bs, hb                  *blockScratch
	head                    *nn.Head
	lo, hi, pos             int
	x, hx, hcs, hcsSWA      []float32
	pn, koffs, kposs, rposs []byte
	p                       *nn.LayerPlan
	rag                     *ragStep
	pe                      *embPend
	key                     graphKey
	R, nrow, kvDim, qdim    int
	kst, nCap, gatherN      int
	rrows                   int
	sm                      backend.Kernel
	smG, smW                int
	ragHead                 int
	fold, ropeDev, argmax   bool
	sampleK                 int
	samplePen, pick         bool
	parOK, allRows          bool
	hNorm                   backend.Buf
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
	a := &g.sub
	// The session the submission runs for, carried in its recording's key.
	sid := a.key.sid
	bs, hb, head, lo, hi, pos, x := a.bs, a.hb, a.head, a.lo, a.hi, a.pos, a.x
	hx, hcs, hcsSWA, pn, koffs, kposs, rposs := a.hx, a.hcs, a.hcsSWA, a.pn, a.koffs, a.kposs, a.rposs
	p, rag, pe, key := a.p, a.rag, a.pe, a.key
	R, nrow, kvDim, qdim, kst, nCap, gatherN, rrows := a.R, a.nrow, a.kvDim, a.qdim, a.kst, a.nCap, a.gatherN, a.rrows
	sm, smG, smW := a.sm, a.smG, a.smW
	fold, ropeDev, argmax, sampleK, samplePen, pick := a.fold, a.ropeDev, a.argmax, a.sampleK, a.samplePen, a.pick
	ragHead, parOK, allRows, hNorm := a.ragHead, a.parOK, a.allRows, a.hNorm
	var err error
	direct := false
	defer func() { a.err, a.direct = err, direct }()
	w := func(b backend.Buf, p []byte) {
		if err == nil {
			t := time.Now()
			err = s.Write(b, p)
			g.TSubStage += time.Since(t)
		}
	}
	// fsrc is the float buffer the current int8 activation was quantized
	// from: a launch writing (bs.a, bs.ax) quantizes its first buffer, and
	// that shape is the record. A float weight reads it in place of a
	// scale plane (see kernels.MatVec), so it keeps its precision.
	var fsrc backend.Buf
	// actMemo is the ActF16 conversion the last batched matvec left in
	// g.f16Buf: which kernel (its layout and shape) and from which float
	// source. A launch through la or laRows may rewrite any buffer, so each
	// one forgets it; runBatched sets it after its own.
	var actMemoK backend.Kernel
	var actMemoSrc backend.Buf
	// lc is the submission's launch path; its la is the ordinary launch
	// (see launcher).
	g.launchTo.s = s
	if ps, ok := s.(*persistSession); ok {
		g.launchTo.s = ps.Session
	}
	persist := g.launchTo.psess.Session != nil
	defer func() { g.launchTo.s = nil }()
	lc := &launcher{to: &g.launchTo, err: &err, fsrc: &fsrc,
		memoK: &actMemoK, memoSrc: &actMemoSrc, a: bs.a, ax: bs.ax, g: g}

	// lamv launches a decode matvec at its own workgroup width: an in-group
	// split is 32 rows by Split segments (kernels.GroupSplitWidth), every
	// other matvec 128. The thread count is the same either way.
	lamv := func(m mv, k backend.Kernel, threads int, bufs ...backend.Buf) {
		w := 128
		if m.group {
			w = kernels.GroupSplitWidth(m.split)
		} else if m.width > 0 && k == m.kern {
			w = m.width
		}
		if err == nil {
			err = lc.launch(k, (threads+w-1)/w, w, bufs...)
		}
	}
	// dOf is what a matvec takes in the scale-plane slot.
	dOf := func(r *resident) backend.Buf {
		if !kernels.IsFloat(r.t) {
			return r.d
		}
		if fsrc == nil && err == nil {
			err = fmt.Errorf("tier: a %v matvec ran before any activation was quantized", r.t)
		}
		return fsrc
	}
	// runBatched launches a batched twin: straight into dst, or, when it
	// splits k (a ragged step on a card with no matrix instruction; see
	// ragMV), into the partials and a Reduce over every row of every token.
	runBatched := func(bm mv, r *resident, dst, a, ax, d, f backend.Buf) {
		if bm.act != nil {
			// sm_70's tensor-core twin reads the float activation as f16.
			//
			// Once per source and layout, not once per matvec: q, k and v read
			// one normed row, as do gate and up. A layout is the kernel (Q6_K's
			// step order is not Q4_K's).
			if bm.act != actMemoK || f != actMemoSrc {
				lc.la(bm.act, bm.actN, f, g.f16Buf)
				g.ActF16Launches++
			}
			memoK, memoSrc := bm.act, f
			defer func() { actMemoK, actMemoSrc = memoK, memoSrc }()
			out := dst
			if bm.split > 1 {
				out = g.partBuf
			}
			if bm.bias != nil {
				lc.law(bm.kern, bm.threads, bm.width, r.qs, d, r.sc, g.f16Buf, out, bm.bias)
			} else {
				lc.law(bm.kern, bm.threads, bm.width, r.qs, d, r.sc, g.f16Buf, out)
			}
			if t := bm.tail; t != nil {
				if bm.bias != nil {
					lc.law(t.kern, t.threads, t.width, r.qs, d, r.sc, g.f16Buf, out, bm.bias)
				} else {
					lc.law(t.kern, t.threads, t.width, r.qs, d, r.sc, g.f16Buf, out)
				}
			}
			if bm.split > 1 {
				lc.la(bm.red, bm.redN, g.partBuf, dst)
			}
			return
		}
		n := bm.threads
		if n == 0 {
			n = bm.rows * bm.groups * max(bm.split, 1)
		}
		out := dst
		if bm.split > 1 && !bm.group {
			out = g.partBuf
		}
		if bm.bias != nil {
			lamv(bm, bm.kern, n, r.qs, d, r.sc, a, ax, out, bm.bias)
		} else {
			lamv(bm, bm.kern, n, r.qs, d, r.sc, a, ax, out)
		}
		if bm.split > 1 && !bm.group {
			lc.la(bm.red, bm.redN, g.partBuf, dst)
		}
	}
	// mvrunAct is mvrun with the activation named rather than assumed, for a
	// chained matvec such as KDA's low-rank gates, whose second half reads the
	// first half's output. It takes the float source f too, because a float
	// weight reads the float activation (dOf) rather than a/ax.
	mvrunAct := func(m mv, r *resident, dst, a, ax, f backend.Buf) {
		dOfAct := func(r *resident) backend.Buf {
			if !kernels.IsFloat(r.t) {
				return r.d
			}
			return f
		}
		if R > 1 {
			bm, ok := g.batchMV(m, R, bs.tok)
			if rag != nil {
				bm, ok = g.ragStepMV(m, R, len(rag.pos))
			}
			if !ok {
				if err == nil {
					err = errNoBatch
				}
				return
			}
			runBatched(bm, r, dst, a, ax, dOfAct(r), f)
			return
		}
		if g.TableSplit && m.alt != nil {
			m = *m.alt
		}
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		n := m.rows * m.split
		if m.rowt > 1 {
			n /= m.rowt
		}
		if m.bias != nil {
			lamv(m, m.kern, n, r.qs, dOfAct(r), r.sc, a, ax, out, m.bias)
		} else {
			lamv(m, m.kern, n, r.qs, dOfAct(r), r.sc, a, ax, out)
		}
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
		}
	}
	mvrun := func(m mv, r *resident, dst backend.Buf) {
		if R > 1 {
			// The batched twin has no split arm: a prefill chunk already has token
			// parallelism, so MatVec refuses NTok > 1 with Split > 1.
			bm, ok := g.batchMV(m, R, bs.tok)
			if rag != nil {
				bm, ok = g.ragStepMV(m, R, len(rag.pos))
			}
			if !ok {
				if err == nil {
					err = errNoBatch
				}
				return
			}
			runBatched(bm, r, dst, bs.a, bs.ax, dOf(r), fsrc)
			return
		}
		if g.TableSplit && m.alt != nil {
			m = *m.alt
		}
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		// The grid divides by the row tile; an oversized one is silent, since
		// surplus threads clamp to the last work item.
		n := m.rows * m.split
		if m.rowt > 1 {
			n /= m.rowt
		}
		if m.bias != nil {
			lamv(m, m.kern, n, r.qs, dOf(r), r.sc, bs.a, bs.ax, out, m.bias)
		} else {
			lamv(m, m.kern, n, r.qs, dOf(r), r.sc, bs.a, bs.ax, out)
		}
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
		}
	}
	// mvres is mvrun with the residual in the bias slot: dst = resid + W*a,
	// replacing the Add that followed. Only a one-row decode with the
	// variant built; false means the caller runs mvrun and the Add.
	mvres := func(m mv, r *resident, dst, resid backend.Buf) bool {
		if rag != nil {
			gm, ok := g.groupKern(m, len(rag.pos), R, groupRes, 0)
			if ok {
				lamv(gm, gm.kern, gm.rows*gm.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, dst, resid)
				g.RagResFused++
			}
			return ok
		}
		if m.res == nil || R > 1 || (g.TableSplit && m.alt != nil) {
			return false
		}
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		n := m.rows * m.split
		if m.rowt > 1 {
			n /= m.rowt
		}
		lamv(m, m.res, n, r.qs, dOf(r), r.sc, bs.a, bs.ax, out, resid)
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
		}
		g.ResidualFused++
		return true
	}
	// mvgate launches the up projection's Gate variant, writing
	// act(gate)*up into dst; false where there is none, and the caller then
	// runs the matvec and the ActMul.
	mvgate := func(m mv, r *resident, dst, gate backend.Buf) bool {
		if rag != nil {
			gm, ok := g.groupKern(m, len(rag.pos), R, groupGate, p.Act)
			if ok {
				lamv(gm, gm.kern, gm.rows*gm.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, dst, gate)
				g.RagGateFused++
			}
			return ok
		}
		if m.gated == nil || R > 1 || (g.TableSplit && m.alt != nil) {
			return false
		}
		lamv(m, m.gated, m.rows*m.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, dst, gate)
		return true
	}
	// mvIn launches one of m's operand-prologue kernels (fuseActIn): the
	// float row src, and the norm weight nw where a norm leads, in the
	// operand's place; extra is the residual (res) or the gate (gated).
	mvIn := func(m mv, k backend.Kernel, r *resident, dst, src, nw, extra backend.Buf) {
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		n := m.rows * m.split
		if m.rowt > 1 && k != m.gatedIn {
			n /= m.rowt
		}
		var arr [8]backend.Buf
		bufs := append(arr[:0], r.qs, r.d, r.sc, src)
		if nw != nil {
			bufs = append(bufs, nw)
		}
		bufs = append(bufs, out)
		if extra != nil {
			bufs = append(bufs, extra)
		}
		lamv(m, k, n, bufs...)
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
		}
		g.ActInLaunches++
	}
	// mvid launches an indexed matvec: the same buffers plus the selection,
	// and rows*slots*split threads rather than rows. The kernel takes pSel
	// after pOut, so the argument order here is the parameter order there.
	// The reduction is over Rows*Slots rows, which is the launch's own
	// output geometry and not the bank's.
	// An in-group split (retuneIndexed) writes the final rows itself, at its
	// own workgroup width.
	mvid := func(m mv, r *resident, dst, sel backend.Buf) {
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		lamv(m, m.kern, m.rows*m.slots*m.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, out, sel)
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows*m.slots, g.partBuf, dst)
		}
	}
	// skip holds WithSkip's bisection probes (scores, softmax, acc, attn, ffn,
	// ...), which produce wrong tokens on purpose: skipping a phase and
	// re-timing the prompt prices it without serialising the stream.
	skip := map[string]bool{}
	if R > 1 {
		for k := range g.kb.skip {
			skip[k] = true
		}
	}
	skipAttn, skipFFN := skip["attn"], skip["ffn"]
	// laRows launches one RMSNormGroup-thread group per row.
	laRows := func(k backend.Kernel, bufs ...backend.Buf) {
		actMemoK, actMemoSrc = nil, nil
		if err == nil {
			err = lc.launch(k, R, kernels.RMSNormGroup, bufs...)
		}
	}
	// norm is RMSNorm or LayerNorm, decided by the block once; LayerNorm has
	// a pass in between, a bias, and an apply reading two partial arrays.
	norm := func(src, w, wb backend.Buf) {
		// No pre-norm (OLMo 2): the branch reads the residual as it is.
		if w == nil {
			lc.la(bs.copyRows, R*p.NEmbd, src, bs.h)
			return
		}
		if bs.rms != nil && bs.normVar == nil && wb == nil {
			laRows(bs.rms, src, w, bs.h)
			return
		}
		lc.la(bs.normPart, R*bs.parts, src, bs.part)
		if bs.normVar != nil {
			lc.la(bs.normVar, R*bs.parts, src, bs.part, bs.part2)
			lc.la(bs.normApply, R*p.NEmbd, src, w, bs.part, bs.part2, wb, bs.h)
			return
		}
		lc.la(bs.normApply, R*p.NEmbd, src, w, bs.part, bs.h)
	}
	// Scanned rather than counted on the tier, so no release path can let a
	// counter drift.
	streamed := false
	for li := lo; li < hi && !streamed; li++ {
		// A batched mixture reads its selection home too (moegroup.go).
		if l := g.layers[li]; l != nil && (l.stream != nil || (R > 1 && l.router != nil)) {
			streamed = true
		}
	}
	// addRes is a block's residual add, out = x + y, with y scaled first where
	// the model scales its block outputs (residScaled): never fused into a
	// projection or a norm, which carry no scale.
	scaled := residScaled(p)
	// resScaled keeps the projections from fusing the residual; only
	// ScaleFaultStepResidual lets a ragged step's fuse it, unscaled.
	resScaled := scaled && !(rag != nil && g.kb.scaleFault == ScaleFaultStepResidual)
	addRes := func(x, y, out backend.Buf) {
		if scaled {
			if g.kb.scaleFault == ScaleFaultResidual {
				lc.la(bs.addE, R*p.NEmbd, x, y, out)
				return
			}
			lc.la(bs.addS, R*p.NEmbd, x, y, bs.rscale, out)
			g.ResidScaled++
			return
		}
		lc.la(bs.addE, R*p.NEmbd, x, y, out)
	}
	// emit issues the whole token's launches. It is a closure so the same code
	// both runs the sequence and records it: one source for both means a graph
	// cannot differ from the launches it replaces.
	emit := func() {
		// The rotary table first: every block reads it and nothing writes it again.
		// Both launches are pure functions of buffers, so a replay rebuilds it from
		// whatever bs.rpos holds this token.
		if ropeDev {
			lc.la(bs.ropeTable, rrows*bs.ropeNPairs, bs.cs, bs.ropeTab, bs.ropeConst, bs.rpos)
			if bs.csSWA != nil {
				lc.la(bs.ropeTable, rrows*bs.ropeNPairs, bs.csSWA, bs.ropeTabSWA, bs.ropeConst, bs.rpos)
			}
			g.ds4Tables(lc, bs, R)
		}
		for li := lo; li < hi; li++ {
			attnFused, ffnFused := false, false
			l := g.layers[li]
			// The previous block's held launches go out; this block's are
			// held when it is one the path takes.
			lc.persistBlock(persist && !streamed && persistable(l, bs))
			// A block whose every batched matvec is on sm_70's f16 path never reads
			// the int8 activation, so it skips the quantize: the norm runs alone and
			// fsrc names the float row the ActF16 conversion reads.
			// Not an AltUp block, whose LAuReL branch quantizes between the norm
			// and the attention.
			voltaOnly := R > 1 && rag == nil && g.voltaBlock(l, R, bs.tok) && l.altRouter == nil && l.ds4 == nil &&
				l.k3 == nil && l.q4e == nil
			// Gemma 3n: every stream's prediction, the block run on the active
			// one (altup.go).
			if l.altRouter != nil {
				g.emitAltPredict(lc, bs, l, R, norm)
			}
			// A linear block replaces the attention half and ends where it ends, in
			// bs.mvOut, so everything below is shared (RULE 8a's block as a unit).
			if l.linear {
				g.emitLinear(sid, s, bs, l, li, R, rag != nil, p, &err, lc, mvrun, mvrunAct)
			}
			if l.ds4 != nil {
				// DeepSeek V4: the hyper-connection's mix, then its own
				// attention, ending in bs.mvOut (ds4.go).
				pl, ent := g.kvp.layers[li], g.kvp.layers[entKey(li)]
				if pl == nil || bs.pkv == nil || bs.pkv.cur == nil {
					if err == nil {
						err = fmt.Errorf("tier: block %d has no pages in this device's pool", li)
					}
					return
				}
				g.ds4Pre(lc, bs, l, R, bs.x, true, norm)
				pc := pagedCall{pk: bs.pkv, pl: pl, k: pl.k, tab: pl.tab, desc: bs.pkv.pRow}
				g.PagedLaunches++
				g.ds4Attn(s, lc, bs, l, R, &pc, ent, mvrun, mvid)
			}
			// A block that runs attention beside its mixer (Falcon-H1) runs the
			// attention half too, from the same norm, and the two outputs are
			// summed below.
			if (!l.linear || l.withAttn) && l.ds4 == nil {
				// This session's attention history for the block -- its source's,
				// for a KV-sharing block -- while the weights are the model's and
				// shared. See layer.kv.
				// A row-chunked block's second phase reads the k and v its
				// first phase wrote for every chunk, as a KV-sharing block
				// reads its source's: q alone.
				shared := l.kvSrc != li || p.NonCausal && g.ncPhase == 2
				kvp := g.layers[l.kvSrc].kvOf(sid)
				if kvp == nil {
					// A missing cache fails the submission rather than launching against
					// nil buffers: PrepLayer returns false when a second session's KV does
					// not fit, so this is reachable on a nearly full card.
					if err == nil {
						err = fmt.Errorf("tier: block %d has no KV cache for session %d",
							li, sid)
					}
					return
				}
				// --- attention
				// Kimi-K3's block reads the mix of its banked checkpoints and
				// the running residual (k3.go); every other block bs.x.
				ain := bs.x
				if l.k3 != nil {
					ain = g.k3AttnIn(lc, bs, l, R)
				}
				if l.q4e != nil {
					// The hyper-connection's mix (q4e.go); an attention
					// block carries no n-gram embedding, so no state.
					ain = g.q4eAttnIn(lc, bs, l, R, mvrunAct, nil, nil, nil)
				}
				// The operand prologue (fuseActIn): q/k/v normalise and quantize
				// the block input themselves, and the norm launch goes. Only
				// where q/k/v run as the fused launch below (qkvIn) or as
				// their three launches (qkvSep), and nothing else reads the
				// normed row (bs.h).
				qkvPro := R == 1 && rag == nil && g.actInOn() && l.k3 == nil && l.q4e == nil && !l.mla && l.clampB == nil &&
					l.kvSrc == li && !l.vFromK && !g.TableSplit && l.altRouter == nil && bs.normVar == nil &&
					l.nAttnB == nil && l.nAttn != nil && !voltaOnly && !skip["norm"] && !skip["quant"] && !skip["qkvo"]
				qkvIn := qkvPro && l.qkv != nil && l.qkvIn != nil &&
					l.qkvBias == [3]bool{l.mvq.bias != nil, l.mvk.bias != nil, l.mvv.bias != nil}
				qkvSep := qkvPro && l.qkv == nil && l.mvq.kernIn != nil
				if qkvIn || qkvSep {
					g.ActInSaved++
				} else if voltaOnly && !skip["norm"] {
					// The norm writes the first projection's binary16 operand
					// too, and the memo tells its GEMM so.
					nk, conv := backend.Kernel(nil), backend.Kernel(nil)
					if l.nAttnB == nil && l.nAttn != nil && l.altRouter == nil {
						nk, conv = g.rmsF16For(bs, l.mvq, R, false)
					}
					if nk != nil {
						laRows(nk, ain, l.nAttn, bs.h, g.f16Buf)
						actMemoK, actMemoSrc = conv, bs.h
						g.NormF16++
					} else {
						norm(ain, l.nAttn, l.nAttnB)
					}
					fsrc = bs.h
					g.QuantSkipped++
				} else if !skip["norm"] && !skip["quant"] && bs.rmsQ != nil && bs.normVar == nil && l.nAttnB == nil &&
					l.nAttn != nil && l.altRouter == nil {
					laRows(bs.rmsQ, ain, l.nAttn, bs.h, bs.a, bs.ax)
					fsrc = bs.h // what la records for a quantize: float weights read it
				} else {
					if !skip["norm"] {
						norm(ain, l.nAttn, l.nAttnB)
					}
					if !skip["quant"] {
						lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.h, bs.a, bs.ax)
					}
					if l.altRouter != nil {
						g.emitLaurel(lc, bs, l, R, mvrun, norm)
					}
				}
				// MLA takes the rest of the attention half and ends in bs.mvOut, the
				// same seam emitLinear sits on.
				var pc *pagedCall
				if bs.pkv != nil && l.mla {
					pl := g.kvp.layers[li]
					if pl == nil {
						if err == nil {
							err = fmt.Errorf("tier: block %d has no pages in this device's pool", li)
						}
						return
					}
					pc = &pagedCall{pk: bs.pkv, pl: pl, k: pl.k, tab: pl.tab, desc: bs.pkv.pRow}
					g.PagedLaunches++
					if !bs.pkv.prefill && bs.pkv.cur.plan.Path == kernels.PathStaged {
						g.PagedStagedLaunches++
					}
				}
				// Kimi-K3's MLA output gate reads the normed input's
				// quantization, which the query's second step replaces.
				if l.mla && l.mlaGate != nil {
					g.k3GateMLA(bs, l, mvrun)
				}
				if l.mla && R > 1 {
					g.emitMLARows(s, bs, l, kvp, p, &err, lc, mvrun, sm, smG, smW, nCap, pc)
				} else if l.mla {
					g.emitMLA(s, bs, l, kvp, p, &err, lc, mvrun, mvid, sm, smG, smW, nCap, skip, pc)
				} else {
					// A few-sequence step takes decode's one launch over its rows.
					var seg segLaunch
					segOK := false
					// A block with no v projection (Gemma 4's global layers:
					// v is k's) takes neither fused launch, both of which write
					// three outputs.
					vFromK := l.vFromK
					if rag != nil && !g.TableSplit && !vFromK && !shared {
						seg, segOK = g.groupQKV(l, len(rag.pos), R)
					}
					if !skip["qkvo"] {
						if shared {
							// q alone: k and v are the source's, already cached.
							mvrun(l.mvq, l.wq, bs.q)
						} else if vFromK {
							mvrun(l.mvq, l.wq, bs.q)
							mvrun(l.mvk, l.wk, bs.k)
						} else if segOK || R == 1 && l.qkv != nil && !g.TableSplit &&
							l.qkvBias == [3]bool{l.mvq.bias != nil, l.mvk.bias != nil, l.mvv.bias != nil} {
							kern, blocks, w := l.qkv, l.qkvBlocks, l.qkvW
							if segOK {
								kern, blocks, w = seg.kern, seg.blocks, seg.w
								g.RagQKVFused++
							}
							// Built in a stack array: two, then up to five a
							// projection; appending past a literal's length
							// grew it on the heap every block.
							var bufArr [17]backend.Buf
							bufs := append(bufArr[:0], bs.a, bs.ax)
							if qkvIn {
								kern, bufs = l.qkvIn, append(bufArr[:0], ain, l.nAttn)
								g.ActInLaunches++
							}
							for _, sg := range []struct {
								m   mv
								r   *resident
								out backend.Buf
							}{{l.mvq, l.wq, bs.q}, {l.mvk, l.wk, bs.k}, {l.mvv, l.wv, bs.v}} {
								bufs = append(bufs, sg.r.qs, sg.r.d, sg.r.sc, sg.out)
								if sg.m.bias != nil {
									bufs = append(bufs, sg.m.bias)
								}
							}
							if err == nil {
								err = lc.launch(kern, blocks, w, bufs...)
							}
						} else if l.clampB != nil {
							// Gemma 4's clipped linears: each projection reads its
							// own clamped copy of the normed rows, quantized for it,
							// and writes into a buffer its output clamp reads.
							for s, pr := range [3]struct {
								m   mv
								r   *resident
								out backend.Buf
								w   int
							}{{l.mvq, l.wq, bs.q, qdim}, {l.mvk, l.wk, bs.k, kvDim}, {l.mvv, l.wv, bs.v, kvDim}} {
								lc.la(bs.clampK[2*s], R*p.NEmbd, bs.h, l.clampB, bs.cin)
								lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.cin, bs.a, bs.ax)
								fsrc = bs.cin
								mvrun(pr.m, pr.r, pr.out)
							}
						} else if qkvSep {
							mvIn(l.mvq, l.mvq.kernIn, l.wq, bs.q, ain, l.nAttn, l.mvq.bias)
							mvIn(l.mvk, l.mvk.kernIn, l.wk, bs.k, ain, l.nAttn, l.mvk.bias)
							mvIn(l.mvv, l.mvv.kernIn, l.wv, bs.v, ain, l.nAttn, l.mvv.bias)
						} else {
							mvrun(l.mvq, l.wq, bs.q)
							mvrun(l.mvk, l.wk, bs.k)
							mvrun(l.mvv, l.wv, bs.v)
						}
					}
					// The per-head norm goes between the projection and RoPE, and only
					// there: a rotation preserves each pair's norm but not the per-head
					// scaling, so norming afterwards is a different function.
					qSrc, kSrc, vSrc := bs.q, bs.k, bs.v
					// DBRX clamps all three before anything reads them, as llama.cpp and
					// transformers both do.
					if bs.clampQ != nil {
						lc.la(bs.clampQ, R*qdim, bs.q, bs.qClamp)
						lc.la(bs.clampKV, R*kvDim, bs.k, bs.kClamp)
						lc.la(bs.clampKV, R*kvDim, bs.v, bs.vClamp)
						qSrc, kSrc, vSrc = bs.qClamp, bs.kClamp, bs.vClamp
					}
					if l.clampB != nil {
						lc.la(bs.clampK[1], R*qdim, bs.q, l.clampB, bs.qClamp)
						lc.la(bs.clampK[3], R*kvDim, bs.k, l.clampB, bs.kClamp)
						lc.la(bs.clampK[5], R*kvDim, bs.v, l.clampB, bs.vClamp)
						qSrc, kSrc, vSrc = bs.qClamp, bs.kClamp, bs.vClamp
					}
					if bs.splitQG != nil {
						// Deinterleave before anything reads q: the norm, RoPE and
						// the scores all want the query alone, qdim wide.
						lc.la(bs.splitQG, R*qdim, bs.q, bs.qg, bs.ogate)
						qSrc = bs.qg
					}
					// Gemma 4's global layers: v is k's projection, read before
					// k's norm writes bs.kn and before the rotary.
					if vFromK {
						vSrc = kSrc
					}
					kOff := bs.koff
					if kst > 0 {
						kOff = bs.kpos
					}
					qOff := bs.zero
					if R > 1 {
						qOff = bs.roff
					}
					// Llama 4's weightless norm runs on the rotating layers
					// only, against ones (see bs.l4Ones for why before RoPE).
					nQ, nK := l.nQ, l.nK
					if nQ == nil && p.QKL2Norm && p.RopeAt(li) {
						nQ, nK = bs.l4Ones, bs.l4Ones
					}
					if bs.qNorm != nil && nQ != nil {
						// The geometry follows the kernel's lanes (headNormGeom). A wide norm
						// is one head per row, whatever NHead is.
						qh, kh := R*p.NHead, R*p.NKVHead
						if p.QKNorm && p.QKNormWide {
							qh, kh = R, R
						}
						qg, qw := headNormGeom(bs.qNormL, qh)
						kg, kw := headNormGeom(bs.kNormL, kh)
						if bs.qkRms != nil {
							// One whole-row norm per vector: llama.cpp's
							// rms_norm_f32 shape, two launches not four.
							laRows(bs.qkRms, qSrc, l.nQ, bs.qn)
							laRows(bs.qkRms, kSrc, l.nK, bs.kn)
							g.QKRms++
						} else if bs.qkPart != nil && bs.qkApply != nil {
							w := p.NHead * p.HeadDim
							lc.la(bs.qkPart, R*bs.parts, qSrc, bs.part)
							lc.la(bs.qkApply, R*w, qSrc, nQ, bs.part, bs.qn)
							lc.la(bs.qkPart, R*bs.parts, kSrc, bs.part)
							lc.la(bs.qkApply, R*w, kSrc, nK, bs.part, bs.kn)
						} else {
							if err == nil && !p.QKNormPost {
								// qSrc, not bs.q: with an attention out-gate bs.q is
								// the double-width interleaved projection and the norm
								// wants the query alone.
								err = lc.launch(bs.qNorm, qg, qw, qSrc, nQ, bs.qn)
							}
							if err == nil && !shared && !p.RopeSplit {
								err = lc.launch(bs.kNorm, kg, kw, kSrc, nK, bs.kn)
							}
						}
						// Hunyuan's q is normed after the rotary instead (below),
						// and under XD-RoPE its k too.
						if !p.QKNormPost {
							qSrc = bs.qn
						}
						if !p.RopeSplit {
							kSrc = bs.kn
						}
						// Gemma 4's v: each head RMSNormed with no weight, on k's
						// head-norm kernel (the same heads and width) against ones.
						if p.VNorm && bs.vn != nil && err == nil && !shared {
							err = lc.launch(bs.kNorm, kg, kw, vSrc, bs.hdOnes, bs.vn)
							vSrc = bs.vn
						}
					}
					// Per layer: Llama 4 leaves every SWAPeriod'th layer unrotated.
					roped := bs.ropeQ != nil && !l.noRope && (p.RopeAt(li) || rag != nil && rowsFaulted("nope"))
					// mistral3's temperature reaches the rotated layers too. A row's
					// factor and its rotation are both linear in q, so q is scaled
					// first, into qt, and rotated from there.
					if roped && !p.NoPEGlobal && bs.qTemp != nil && bs.qt != nil &&
						!(rag != nil && rowsFaulted("notemp")) {
						lc.la(bs.qTemp, R*qdim, qSrc, bs.n, bs.tempTab, bs.qt)
						qSrc = bs.qt
					}
					// Where K and V go: this session's contiguous cache at the
					// row's offset, or the pool's layer at the page each row's
					// descriptor names (pagedattn.go), the offset unread.
					copyK, ropeK, copyV := bs.copyK, bs.ropeK, bs.copyV
					kDst, vDst, vOff := kvp.kc, kvp.vc, bs.koff
					var pgArgs []backend.Buf
					var pgTab backend.Buf // this layer's table arena
					pk := bs.pkv
					if pk != nil {
						pl := g.kvp.layers[l.kvSrc]
						if pl == nil {
							if err == nil {
								err = fmt.Errorf("tier: block %d has no pages in this device's pool", li)
							}
							return
						}
						copyK, ropeK, copyV = pk.copyK, pk.ropeK, pk.copyV
						kDst, vDst, kOff, vOff = pl.k, pl.v, bs.zero, bs.zero
						pgTab = pl.tab
						pgArgs = []backend.Buf{pl.tab, pk.pRow}
					}
					laKV := func(k backend.Kernel, threads int, bufs ...backend.Buf) {
						// Joined in a stack array: append to bufs itself would
						// grow it on the heap every launch.
						var all [8]backend.Buf
						lc.la(k, threads, append(append(all[:0], bufs...), pgArgs...)...)
					}
					// MiniMax-M3's selecting block: the indexer's key joins k's
					// heads in one row, which the pool's writers turn and store
					// together; its query waits for the selection (msa.go).
					kHeads, vThreads := p.NKVHead, R*p.NKVHead*p.HeadDim
					if l.msa {
						if pk == nil || pk.msa == nil {
							if err == nil {
								err = fmt.Errorf("tier: block %d selects its keys and has no paged scratch here", li)
							}
							return
						}
						if err == nil {
							err = g.msaKeyQuery(lc, bs, l, R, kSrc, bs.cs, mvrun)
						}
						kSrc, kHeads = pk.msa.kAll, kHeads+1
					}
					// Qwen3.8-Flash-Next's: the raw key joins the row,
					// which is copied whole; only k's heads turn (qsa.go).
					if l.qsa {
						if pk == nil || pk.qsa == nil {
							if err == nil {
								err = fmt.Errorf("tier: block %d selects its keys and has no paged scratch here", li)
							}
							return
						}
						if err == nil {
							err = g.qsaKeyQuery(lc, bs, l, R, kSrc, bs.cs, mvrun)
						}
						kSrc = pk.qsa.kAll
					}
					fusedKV := false
					if !skip["rope"] {
						// An unrotated layer still needs K and V in the cache: q is read where
						// it is (qAtt below) and k is copied where RoPERowsT would have
						// written it. A KV-sharing block writes neither: its source did.
						if !roped && !shared {
							laKV(copyK, R*kvDim, kSrc, kDst, kOff)
						} else if roped {
							// The table layer li was trained with: gemma3 puts five layers in
							// six on the local base (see nn.LayerPlan.SWAPeriod), ModernBERT
							// all but every third from the first (LocalRope).
							rcs := bs.cs
							if bs.csSWA != nil && p.LocalRope(li) {
								rcs = bs.csSWA
							}
							// The unrotated tail first: RoPERows writes only the nRot
							// dimensions it rotates, and the scores read the whole head. It
							// copies the normalized q and k, as the host passes the tail
							// through, rather than zeros.
							if bs.copyQ != nil {
								lc.la(bs.copyQ, R*qdim, qSrc, bs.qr, qOff)
							}
							if copyK != nil && (partialRotary(p) || l.qsa) && !shared {
								laKV(copyK, R*kvDim, kSrc, kDst, kOff)
							}
							if pk != nil && pk.ropeQKV != nil && !shared && bs.copyQ == nil && kHeads == p.NKVHead &&
								!g.NoFuse && !skip["fuse"] {
								// q's rotation, k's into its page and v's copy
								// in one launch (kernels.RoPEQKVPaged), the
								// three kernels below being its parts.
								lc.la(pk.ropeQKV, kernels.RoPEQKVPagedThreads(p.NHead, p.NKVHead, p.HeadDim, p.NRot, R, g.KVF16),
									qSrc, rcs, qOff, bs.qr, kSrc, kDst, vSrc, vDst, pgTab, pk.pRow)
								g.FusedLaunches++
								fusedKV = true
							} else {
								lc.la(bs.ropeQ, R*p.NHead*p.NRot/2, qSrc, rcs, qOff, bs.qr)
							}
							if fusedKV {
							} else if !shared && p.RopeSplit {
								// XD-RoPE: k turns row-major, is normed after
								// the rotary (the rotation does not keep a head's
								// RMS), and is copied into the cache.
								kro := bs.zero
								if R > 1 {
									kro = bs.kroff
								}
								lc.la(bs.ropeKRow, R*p.NKVHead*p.NRot/2, kSrc, rcs, kro, bs.krot)
								kg, kw := headNormGeom(bs.kNormL, R*p.NKVHead)
								if err == nil {
									err = lc.launch(bs.kNorm, kg, kw, bs.krot, l.nK, bs.kn)
								}
								laKV(copyK, R*kvDim, bs.kn, kDst, kOff)
							} else if !shared {
								laKV(ropeK, R*kHeads*p.NRot/2, kSrc, rcs, kOff, kDst)
							}
						}
						if !shared && !fusedKV {
							laKV(copyV, vThreads, vSrc, vDst, vOff)
						}
					}
					// Where the score kernel reads q: the rotated buffer, or q itself on
					// an unrotated layer (bs.qr would hold the last block's).
					qAtt := bs.qr
					if !roped {
						qAtt = qSrc
						// Llama 4's temperature, on exactly these layers:
						// q times a factor of its position. bs.qr is free
						// here, since nothing rotated into it.
						if bs.qTemp != nil && !(rag != nil && rowsFaulted("notemp")) {
							lc.la(bs.qTemp, R*qdim, qSrc, bs.n, bs.tempTab, bs.qr)
							qAtt = bs.qr
						}
					}
					// Hunyuan's q norm runs after the rotary, weighted by w_q*w_k
					// (jlm.FlagQKNormPostRope), into bs.qn -- which the pre-rotary
					// norm left unwritten for q, so it reads one buffer and writes
					// another (RULE 13).
					if p.QKNormPost && bs.qNorm != nil && l.nQ != nil {
						qg, qw := headNormGeom(bs.qNormL, R*p.NHead)
						if err == nil {
							err = lc.launch(bs.qNorm, qg, qw, qAtt, l.nQ, bs.qn)
						}
						qAtt = bs.qn
					}
					// A row-chunked block's first phase ends with its chunk's
					// k and v in the cache: the attention reads every chunk's,
					// so it runs in the second phase.
					if p.NonCausal && g.ncPhase == 1 {
						continue
					}
					var fv *flashVariant
					var flash backend.Kernel
					if len(bs.flashV) > 0 {
						fv = bs.flashFor(nCap)
						flash = fv.k
						if p.Window(li) > 0 {
							flash = fv.kW
						}
					}
					// A windowed vision block masks its scores between the score
					// kernel and the softmax, so it takes the three-kernel path.
					useFlash := g.flashOn() && flash != nil && !skip["scores"] && !skip["softmax"] && !skip["acc"] &&
						!l.windowed
					if !skipAttn && l.qsa {
						if err == nil {
							err = g.qsaAttend(s, lc, bs, R, pk, g.kvp.layers[l.kvSrc], l, qAtt, bs.n, bs.xb)
							g.PagedLaunches++
						}
					} else if !skipAttn && l.msa {
						// Each kv group's kept blocks, gathered, then the decode
						// plan over them, a chunk's rows included (msa.go).
						if err == nil {
							err = g.msaAttend(s, lc, R, pk, g.kvp.layers[l.kvSrc], qAtt, bs.n, bs.xb)
							g.PagedLaunches++
						}
					} else if !skipAttn && pk != nil && pk.prefill {
						// A prefill chunk: one sequence's rows in order, a K
						// or V load serving a whole query tile.
						var sk backend.Buf
						if p.AttnSinks {
							if sk = l.sinks; sk == nil {
								sk = bs.noSink
							}
						}
						win := p.Window(li) > 0 && !pagedFaulted("window")
						if err == nil {
							err = pk.pf.prefillAttn(s, lc, qAtt, kDst, vDst, bs.n, bs.xb, pgTab, win, sk)
							g.PagedPrefillLaunches++
							if pk.pf.flash != nil {
								g.FlashPrefills++ // the paged one-kernel form, on any backend
							}
							g.PagedPrefillPasses += pk.pf.passes(win)
							if pk.pf.on70() {
								g.PagedPrefill70++
							}
							if pk.pf.accOnMMA() {
								g.PagedAccMMA++
							}
							if pk.pf.on80() {
								g.FlashPrefill80++
							}
						}
					} else if !skipAttn && pk != nil {
						// A decode row or a batch's rows: each a descriptor, a
						// window its keyStart (pRowW), through the decode plan
						// chosen before the submission (pagedPlanFor).
						desc := pk.pRow
						if p.Window(li) > 0 && !pagedFaulted("window") && !(rag != nil && rowsFaulted("nowin")) {
							desc = pk.pRowW
						}
						var sk backend.Buf
						if p.AttnSinks {
							if sk = l.sinks; sk == nil {
								sk = bs.noSink
							}
						}
						if err == nil {
							err = pk.attend(s, lc, g.kvp.layers[l.kvSrc], qAtt, kDst, vDst, bs.n, bs.xb, pgTab, desc, sk)
							g.PagedLaunches++
							if pk.cur.plan.Path == kernels.PathStaged {
								g.PagedStagedLaunches++
							}
						}
					} else if !skipAttn && useFlash {
						dst := bs.xb
						if fv.splits > 1 {
							dst = bs.flashPart
						}
						args := []backend.Buf{qAtt, kvp.kc, kvp.vc, bs.n, dst}
						if p.AttnSinks {
							sk := l.sinks
							if sk == nil {
								sk = bs.noSink
							}
							args = append(args, sk)
						}
						if err == nil {
							err = lc.launch(flash, fv.groups, bs.flashWidth, args...)
							if err == nil && fv.splits > 1 {
								ma := []backend.Buf{bs.flashPart, bs.xb}
								if p.AttnSinks {
									ma = append(ma, args[5])
								}
								err = lc.launch(fv.merge, R*p.NHead, 32, ma...)
							}
							g.FlashLaunches++
						}
					} else if !skipAttn && bs.fp70 != nil && !skip["scores"] && !skip["softmax"] && !skip["acc"] && !l.windowed {
						// The chunk's attention as one kernel: scores,
						// softmax and the weighted V sum per key tile, the
						// scores never written (kernels.FlashPrefill70).
						fk := bs.fp70
						if p.Window(li) > 0 {
							fk = bs.fp70W
						}
						if err == nil {
							err = lc.launch(fk, kernels.FlashPrefill70Groups(kernels.FlashPrefill70Shape{Heads: p.NHead, Rows: R}),
								kernels.FlashPrefill70Threads, qAtt, kvp.kc, kvp.vc, bs.n, bs.xb)
						}
						g.FlashPrefills++
					} else if !skipAttn && bs.fpT != nil && !skip["scores"] && !skip["softmax"] && !skip["acc"] && !l.windowed {
						// The same, on Metal's matrix unit (kernels.FlashPrefillTile).
						fk := bs.fpT
						if p.Window(li) > 0 {
							fk = bs.fpTW
						}
						if err == nil {
							err = lc.launch(fk, kernels.FlashTileGroups(p.NHead, R, bs.fpTile),
								bs.fpTile.Threads(), qAtt, kvp.kc, kvp.vc, bs.n, bs.xb)
						}
						g.FlashPrefills++
					} else if !skipAttn {
						// One pass over the rows, or one per query chunk of a vision
						// block whose score planes are sized for a chunk (arows):
						// the chunk's queries gathered into qc, its output scattered
						// back from xc.
						run := func(qIn, out, win backend.Buf, R int) {
							if !skip["scores"] {
								// A local layer takes the windowed pair: bs.n's key counts are
								// causal and shared, while the window is per layer and baked in.
								scores, scoresMMA := bs.scores, bs.scoresMMA
								if p.Window(li) > 0 {
									scores, scoresMMA = bs.scoresW, bs.scoresMMAW
								}
								scores70 := bs.scores70
								if p.Window(li) > 0 {
									scores70 = bs.scores70W
								}
								if scoresMMA != nil {
									// One warp per 16 query rows x 8*attnNT key positions.
									kg := (nCap + 8*g.tiles.attnNT - 1) / (8 * g.tiles.attnNT)
									lc.la(scoresMMA, (R/16)*p.NHead*kg*32, qIn, kvp.kc, bs.n, bs.att)
								} else if scores70 != nil {
									lc.la(scores70, kernels.AttnScoresMMA70Warps(p.NHead, R, nCap, volta70AttnMT,
										volta70AttnNT)*32, qIn, kvp.kc, bs.n, bs.att)
								} else {
									// The key axis is covered by ceil(nCap/ktile)
									// threads, which is the same G the kernel derives
									// from pN.
									kg := (nCap + bs.ktile - 1) / bs.ktile
									lc.la(scores, (R/bs.qtile)*p.NHead*kg, qIn, kvp.kc, bs.n, bs.att)
								}
							}
							att := bs.att
							if bs.softcap != nil {
								lc.la(bs.softcap, R*p.NHead*bs.sstride, bs.att, bs.attCap)
								att = bs.attCap
							}
							// Qwen2.5-VL's windowed blocks: every key outside a
							// row's window to -inf before the softmax, the windows
							// of this pass's rows in win.
							// Packed sequences: every key outside a row's own
							// sequence, the same mask over the runs (SetSeqRuns).
							if win != nil {
								lc.la(bs.winMask, R*p.NHead*bs.sstride, att, win, bs.attWin)
								att = bs.attWin
							}
							if err == nil && !skip["softmax"] {
								// Not la(): softmax is the one kernel whose group is
								// not 128 wide. With 32 lanes to a head the group IS
								// the warp, which is what lets the shuffle assume
								// lane == tid.
								if bs.p.AttnSinks {
									sk := l.sinks
									if sk == nil {
										sk = bs.noSink
									}
									err = lc.launch(sm, smG, smW, att, bs.n, bs.prob, sk)
								} else {
									err = lc.launch(sm, smG, smW, att, bs.n, bs.prob)
								}
							}
							if bs.accSplit > 1 {
								lc.la(bs.attnAcc, qdim*bs.accSplit, bs.prob, kvp.vc, bs.n, g.partBuf)
								lc.la(bs.attnRed, qdim, g.partBuf, out)
							} else if !skip["acc"] && bs.acc70 != nil {
								lc.la(bs.acc70, kernels.AttnAccMMA70Warps(p.NHead, p.HeadDim, R,
									volta70AccMT, volta70AttnNT)*32, bs.prob, kvp.vc, bs.n, out)
							} else if !skip["acc"] {
								lc.la(bs.attnAcc, R*qdim/bs.atile, bs.prob, kvp.vc, bs.n, out)
							}
						}
						if bs.arows > 0 && bs.arows < R {
							for c := range bs.qoffs {
								lc.la(bs.qGather, bs.arows*qdim, qAtt, bs.qc, bs.qoffs[c])
								run(bs.qc, bs.xc, bs.winOf(l, c), bs.arows)
								lc.la(bs.xScatter, bs.arows*qdim, bs.xc, bs.xb, bs.qoffs[c])
							}
							g.VisionAttnChunks += len(bs.qoffs)
						} else {
							run(qAtt, bs.xb, bs.winOf(l, 0), R)
						}
					}
					// The out-gate goes before wo: gating after the projection would scale
					// the residual contribution instead (as engine/model/forward.go notes).
					aSrc := bs.xb
					if bs.sigMul != nil {
						lc.la(bs.sigMul, R*qdim, bs.ogate, bs.xb, bs.ogated)
						aSrc = bs.ogated
					}
					// The attention output projection quantizing its own
					// operand (fuseActIn), with the residual add it already
					// carries.
					oIn := R == 1 && rag == nil && g.actInOn() && l.mvo.resIn != nil && !voltaOnly && l.clampB == nil && l.nSubAttn == nil &&
						!skip["quant"] && !skip["qkvo"] && l.nPostAttn == nil && !l.withAttn && !resScaled && !skip["add"] &&
						l.altRouter == nil && l.k3 == nil && l.q4e == nil && !(g.TableSplit && l.mvo.alt != nil) && l.mvo.k == qdim
					if oIn {
						g.ActInSaved++
					} else if l.nSubAttn != nil {
						// BitNet: the norm over the attention output and the
						// quantize wo reads, one launch a row.
						laRows(bs.subQA, aSrc, l.nSubAttn, bs.sub, bs.a, bs.ax)
						fsrc = bs.sub
					} else if voltaOnly {
						fsrc = aSrc
						g.QuantSkipped++
					} else if l.clampB != nil {
						lc.la(bs.clampK[6], R*qdim, aSrc, l.clampB, bs.cin)
						lc.la(bs.quantQ, kernels.QuantizeThreads(R*qdim/32), bs.cin, bs.a, bs.ax)
						fsrc = bs.cin
					} else if !skip["quant"] {
						lc.la(bs.quantQ, kernels.QuantizeThreads(R*qdim/32), aSrc, bs.a, bs.ax)
					}
					if oIn {
						mvIn(l.mvo, l.mvo.resIn, l.wo, bs.x2, aSrc, nil, bs.x)
						attnFused = true
					} else if !skip["qkvo"] {
						// x2 = x + Wo*attn in one launch where nothing sits
						// between the projection and the residual add.
						if !(l.nPostAttn == nil && !l.withAttn && !resScaled && !skip["add"] && l.altRouter == nil && l.k3 == nil && l.q4e == nil &&
							mvres(l.mvo, l.wo, bs.x2, bs.x)) {
							mvrun(l.mvo, l.wo, bs.mvOut)
						} else {
							attnFused = true
						}
					}
				}
			}

			// gemma2/gemma3's post-attention norm, before the residual add: it
			// normalises the attention output, not the stream. It writes bs.h rather
			// than in place (RULE 13); bs.h is free here, consumed by q/k/v and
			// rewritten by the FFN pre-norm.
			aout := bs.mvOut
			if l.withAttn {
				// attention + mixer; bs.h is free here (see below).
				lc.la(bs.addE, R*p.NEmbd, bs.mvOut, bs.side, bs.h)
				aout = bs.h
			}
			if l.clampB != nil && !attnFused {
				lc.la(bs.clampK[7], R*p.NEmbd, aout, l.clampB, bs.cout)
				aout = bs.cout
			}
			if l.nPostAttn != nil && !skip["norm"] {
				lc.la(bs.normPart, R*bs.parts, aout, bs.part)
				lc.la(bs.normApply, R*p.NEmbd, aout, l.nPostAttn, bs.part, bs.h)
				aout = bs.h
			}

			// A block that is its mixer alone ends here: the residual add, and
			// the sum moved home to bs.x, where the next block reads it.
			if l.noFFN {
				if !skip["add"] && !attnFused {
					addRes(bs.x, aout, bs.x2)
					lc.la(bs.copyRes, R*p.NEmbd, bs.x2, bs.x)
				}
				continue
			}

			// --- feed-forward
			quantDone := false
			// The operand prologue (fuseActIn): gate and up normalise and
			// quantize the stream themselves, and the down projection
			// quantizes act(gate)*up; the FFN norm and its quantize go. Only
			// on a plain gated FFN, where nothing else reads the normed row.
			ffnIn := R == 1 && rag == nil && g.actInOn() && l.mvg.kernIn != nil && (l.mvu.gatedIn != nil || l.mvu.kernIn != nil) &&
				!voltaOnly && !skipFFN && !skip["norm"] && !skip["quant"] && !skip["actmul"] && bs.normVar == nil &&
				l.nFFNB == nil && l.nFFN != nil && l.nResid == nil && l.router == nil && l.clampB == nil &&
				!l.nonCausal && !l.sparse && l.k3 == nil && l.q4e == nil && l.ds4 == nil && l.altRouter == nil && !p.Parallel &&
				!ungatedFFN(p) && !(g.TableSplit && (l.mvg.alt != nil || l.mvu.alt != nil))
			// The FFN norm's fused form where the block runs binary16 GEMMs:
			// it writes the first FFN projection's operand too.
			var ffnNorm, ffnConv backend.Kernel
			if voltaOnly && l.nFFN != nil {
				first := l.mvg
				if first.kern == nil {
					first = l.mvu
				}
				ffnNorm, ffnConv = g.rmsF16For(bs, first, R, true)
			}
			switch {
			case l.q4e != nil:
				// Qwen3.8-Flash-Next: the output placed into every stream,
				// then the mixture's input is the mix over them (q4e.go).
				g.q4ePostAttn(lc, bs, l, R, aout)
				norm(g.q4eFFNIn(lc, bs, l, R, mvrunAct), l.nFFN, l.nFFNB)
			case l.k3 != nil:
				// Kimi-K3: every stream after the attention, then the FFN's
				// input is the mix over them (k3.go).
				g.k3PostAttn(lc, bs, l, R, aout)
				if !skip["norm"] {
					norm(g.k3FFNIn(lc, bs, l, R), l.nFFN, l.nFFNB)
				}
			case l.ds4 != nil:
				// The hyper-connection around the attention, then the
				// mixture's: its input normed into bs.h and quantized (ds4.go).
				g.ds4Post(lc, bs, R, bs.x, aout, bs.x2)
				g.ds4Pre(lc, bs, l, R, bs.x2, false, norm)
				quantDone = true
			case l.altRouter != nil:
				// LAuReL's branch joins the attention's residual add (altup.go),
				// and the FFN's norm takes the joined row.
				if !skip["add"] {
					lc.la(bs.laurelJoin, R*p.NEmbd, bs.x, aout, bs.altLaur, bs.x2)
				}
				if !skip["norm"] {
					norm(bs.x2, l.nFFN, l.nFFNB)
				}
			case p.Parallel:
				// The parallel residual (C6): the FFN reads the block input. bs.x is
				// still that input (attention wrote bs.x2); a block sharing the
				// attention's norm (phi-2) reads bs.h, which nothing rewrote.
				if !skip["add"] && !attnFused {
					addRes(bs.x, aout, bs.x2)
				}
				if l.nFFN != nil && !skip["norm"] {
					norm(bs.x, l.nFFN, l.nFFNB)
				}
			case !voltaOnly && !attnFused && !scaled && !skip["add"] && !skip["norm"] && !skip["quant"] && bs.addRmsQ != nil &&
				bs.normVar == nil && l.nFFNB == nil && aout != bs.h && l.nFFN != nil:
				// Not when aout is bs.h (a post-attention norm wrote it): the
				// quantize phase re-reads the input after the norm phase wrote h
				// (RULE 13). x2 = x + attention, h = RMSNorm(x2) and its quantized
				// form, one launch.
				laRows(bs.addRmsQ, bs.x, aout, l.nFFN, bs.x2, bs.h, bs.a, bs.ax)
				fsrc = bs.h // what la records for a quantize: float weights read it
				quantDone = true
			case voltaOnly && !attnFused && !scaled && !skip["add"] && !skip["norm"] && bs.addRms != nil &&
				bs.normVar == nil && l.nFFNB == nil && aout != bs.h && l.nFFN != nil && ffnNorm != nil:
				// The same launch, writing the gate's binary16 operand too.
				laRows(ffnNorm, bs.x, aout, l.nFFN, bs.x2, bs.h, g.f16Buf)
				actMemoK, actMemoSrc = ffnConv, bs.h
				g.NormF16++
			case !attnFused && !scaled && !skip["add"] && !skip["norm"] && bs.addRms != nil &&
				bs.normVar == nil && l.nFFNB == nil && aout != bs.h && l.nFFN != nil:
				// x2 = x + attention and h = RMSNorm(x2), one launch; not when aout
				// is bs.h either, for the reason above.
				laRows(bs.addRms, bs.x, aout, l.nFFN, bs.x2, bs.h)
			default:
				if !skip["add"] && !attnFused {
					addRes(bs.x, aout, bs.x2)
				}
				if ffnIn {
					g.ActInSaved++
					quantDone = true
				} else if !voltaOnly && !skip["norm"] && !skip["quant"] && bs.rmsQ != nil && bs.normVar == nil && l.nFFNB == nil &&
					l.nFFN != nil {
					laRows(bs.rmsQ, bs.x2, l.nFFN, bs.h, bs.a, bs.ax)
					fsrc = bs.h // what la records for a quantize: float weights read it
					quantDone = true
				} else if !skip["norm"] {
					norm(bs.x2, l.nFFN, l.nFFNB)
				}
			}
			// A post-norm block's residual is the norm of x + attention, the
			// row the FFN reads (nn.LayerPlan.PostResidNorm).
			if l.nResid != nil && !skip["norm"] {
				lc.la(bs.copyRows, R*p.NEmbd, bs.h, bs.x2)
			}
			if voltaOnly && !quantDone {
				fsrc = bs.h
				g.QuantSkipped++
			} else if !skip["quant"] && !quantDone {
				lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.h, bs.a, bs.ax)
			}
			// ffnOut is what the FFN produced: bs.mvOut for every model, and
			// the shared-expert sum where one exists.
			ffnOut := bs.mvOut
			if l.router != nil {
				// Gemma 4's block runs its dense MLP first, on the ffn_norm row
				// in bs.h, and then the mixture on two norms of its own of x2:
				// the router's into bs.hr, the experts' into bs.h. See
				// emitDenseMLP.
				rin := bs.h
				if p.DenseMoE {
					g.emitDenseMLP(lc, bs, l, R, bs.x2, mvrun, &err)
					rin = bs.hr
				}
				// The whole mixture with nothing crossing the bus: the router reads
				// the normed f32 vector (the block's one unquantized matrix) and its
				// output feeds ExpertRank, whose output indexes the expert matvecs.
				// Kimi-K3's experts run at the latent (k3.go): read from its
				// projection of the normed row, summed in x.latAcc and taken
				// out of the latent below.
				ein, eout := bs.h, bs.mvOut
				latent := l.k3 != nil && l.k3.latent != 0
				if latent {
					ein, eout = bs.k3.lat, bs.k3.latAcc
				}
				hybrid := false
				if R > 1 {
					if latent {
						g.k3LatentIn(lc, bs, l, R, mvrun)
					}
					g.emitGroupedMoE(sid, s, lc, bs, l, R, nrow, rin, ein, eout, &err)
					if latent {
						g.k3LatentOut(lc, bs, l, R, mvrun)
					}
				} else {
					// route ranks one block's router over rin into bs.rsel, bs.rtop and,
					// with weights, bs.rw. It is a closure so the streamed path's
					// cross-layer probe can run the NEXT block's router over this block's
					// row through exactly the launches that block will run itself.
					route := func(l *layer, weights bool) bool {
						lc.la(l.mvRouter, kernels.RouterThreads(bs.nExpert), l.router, rin, bs.rlogits)
						logits := bs.rlogits
						if l.routerB != nil {
							// gpt-oss biases the logits before the rank, so the
							// bias decides which experts are chosen.
							lc.la(bs.rbias, bs.nExpert, bs.rlogits, l.routerB, bs.rlogitsB)
							logits = bs.rlogitsB
						}
						// The V3 family's two extra passes exist because the IR has no nested
						// loops: ranking needs each group's mask, which needs the group's rank
						// among all groups, which needs a sum over members. Materialising the
						// group scores and then the masked plane keeps each pass flat.
						plane := logits
						if bs.gscore != nil {
							if l.expSelB != nil {
								lc.la(bs.gscore, p.ExpertGroups, logits, bs.rgs, l.expSelB)
								lc.la(bs.gmask, bs.nExpert, logits, bs.rgs, bs.rbm, l.expSelB)
							} else {
								lc.la(bs.gscore, p.ExpertGroups, logits, bs.rgs)
								lc.la(bs.gmask, bs.nExpert, logits, bs.rgs, bs.rbm)
							}
							plane = bs.rbm
						}
						// The arity follows the route. Vulkan refuses an undeclared buffer count;
						// CUDA ignores a surplus pointer and silently never reads the plane.
						// DeepSeek V4 selects through a plane per row (ds4.go).
						selB := l.expSelB
						if l.ds4 != nil {
							selB = g.ds4RouteBias(lc, bs, l, 1)
						}
						fusedRoute := bs.route != nil && bs.gscore == nil && selB == nil
						switch {
						case fusedRoute:
							// Rank and weights in one workgroup (kernels.ExpertRoute).
							// The probe's weights go to a scratch of their own:
							// this block's are already in bs.rw and still to
							// be read.
							rwDst := bs.rw
							if !weights {
								rwDst = bs.rwProbe
							}
							if err == nil {
								err = lc.launch(bs.route, 1, bs.routeWidth,
									logits, bs.rsel, bs.rtop, rwDst)
							}
						case bs.gscore != nil:
							lc.la(bs.rank, bs.nExpert, logits, bs.rsel, bs.rtop, plane)
						case selB != nil:
							lc.la(bs.rank, bs.nExpert, logits, bs.rsel, bs.rtop, selB)
						default:
							lc.la(bs.rank, bs.nExpert, logits, bs.rsel, bs.rtop)
						}
						if err == nil && !fusedRoute && weights {
							// One thread: k is 2 to 8 and this is three short loops.
							// The un-renormalised softmax kernel reads the full
							// logit array; the renormalised one does not declare
							// it, and the sigmoid one reads neither -- sigma()
							// already ran inside the rank.
							if p.NoExpertNorm && !p.ExpertSigmoid {
								err = lc.launch(bs.weights, 1, 1, bs.rtop, bs.rw, logits)
							} else {
								err = lc.launch(bs.weights, 1, 1, bs.rtop, bs.rw)
							}
						}
						return false
					}
					route(l, true)
					// The suspension, for a streamed bank: the selection comes home, k
					// sheets of gate/up/down are uploaded at their compact offsets, and the
					// submission resumes indexing 0..k-1, because the bank it reads is the
					// selection. The Sync is required: Launch does not wait, and the uploads
					// that follow must precede the launches that read them.
					sel := bs.rsel
					if l.stream != nil {
						sel = bs.ident
						if l.stream.cached() {
							// fill writes each selected expert's cache slot over
							// the selection, in selection order.
							sel = bs.rsel
						}
						if g.StreamFixedSel {
							// The probe spreads per block on purpose: a fixed 0..k-1 would keep
							// one small working set resident and measure that along with the
							// round trip. A block-dependent stride removes only the readback.
							for i := range l.stream.sel {
								l.stream.sel[i] = uint32((li*31 + i*7) % p.NExpert)
							}
						} else if err == nil {
							tw := time.Now()
							err = s.Sync()
							g.TStreamWait += time.Since(tw)
						}
						if err == nil && !g.StreamFixedSel {
							err = s.Read(bs.rsel, u32b(l.stream.sel))
						}
						if err == nil && !g.StreamFixedSel {
							l.stream.score(g, li)
						}
						hybrid = l.stream.hybrid && !g.StreamFixedSel
						if err == nil && hybrid {
							// Hybrid: the experts' input and the routing weights come
							// home, the host runs the selected experts, and their sum
							// goes back where the combine would have written it.
							if latent {
								g.k3LatentIn(lc, bs, l, 1, mvrun)
							}
							err = l.stream.runHost(sid, g, s, ein, bs.rw, eout, p.ExpWidth(), bs.nUsed)
						}
						if err == nil && !hybrid && (g.StreamProbe || g.StreamPrefetch) && li+1 < hi {
							// The cross-layer probe: the next block's router over this
							// block's normed row, ranked by that block's own launches.
							// It clobbers rsel and rtop, which nothing below reads on a
							// streamed block (it indexes bs.ident and the weights are
							// already in bs.rw).
							if nx := g.layers[li+1]; nx != nil && nx.stream != nil && nx.router != nil {
								if route(nx, false) {
									g.ProbeFused++
								} else if err = s.Sync(); err == nil {
									nx.stream.pred = growU32(nx.stream.pred, len(nx.stream.sel))
									err = s.Read(bs.rsel, u32b(nx.stream.pred))
									nx.stream.havePred = err == nil
									if err == nil && g.StreamPrefetch {
										nx.stream.prefetch(g)
									}
								}
							}
						}
						if err == nil && !hybrid {
							tf := time.Now()
							if err = l.stream.fill(g, s, l, p.NExpert); err != nil && l.stream.cached() {
								l.stream.forget()
							}
							g.fillDone(time.Since(tf))
						}
						if err == nil && l.stream.cached() {
							err = s.WriteAt(bs.rsel, 0, u32b(l.stream.slotSel))
						}
						if err != nil {
							return
						}
						if hybrid {
							g.HybridRuns++
						} else {
							g.StreamFills++
						}
					}
					if hybrid && latent {
						// The routed sum is already in eout.
						g.k3LatentOut(lc, bs, l, 1, mvrun)
					}
					if !hybrid {
						// gate and up read the one normed vector, so they are
						// indexed on the weight side only. bs.a/bs.ax still hold
						// its quantization from the launch above -- or, at
						// Kimi-K3's latent, its projection's.
						if latent {
							g.k3LatentIn(lc, bs, l, 1, mvrun)
						}
						ffnW := bs.nUsed * bs.nFFNExp
						// mvidE launches an expert matvec with fused epilogue
						// operands between its output and the selection -- the
						// kernel's parameter order (pOut, [pBias], [pGate], pSel).
						mvidE := func(k backend.Kernel, m mv, r *resident, dst backend.Buf, extra ...backend.Buf) {
							// Joined in a stack array: appending to a literal of
							// its own length grew it on the heap every launch.
							var all [12]backend.Buf
							args := append(all[:0], r.qs, dOf(r), r.sc, bs.a, bs.ax, dst)
							args = append(append(args, extra...), sel)
							lamv(m, k, m.rows*m.slots*m.split, args...)
							g.MoEFused++
						}
						rw := bs.rw
						if p.DenseMoE {
							// Each routed weight times its expert's own factor.
							lc.la(bs.ewScale, bs.nUsed, bs.rw, bs.rsel, l.expScale, bs.rwS)
							rw = bs.rwS
						}
						if ungatedFFN(p) {
							// Ungated experts (Nemotron 3): up, then the activation
							// alone. No bias exists on any such architecture.
							mvid(l.mvu, l.up, bs.eu, sel)
							lc.la(bs.actMulE, ffnW, bs.eu, bs.eact)
						} else if l.moeUp != nil {
							// act(gate)*up in the up projection's own epilogue, the
							// biases in each matvec's (fuseMoE).
							if l.moeGate != nil {
								mvidE(l.moeGate, l.mvg, l.gate, bs.eg, l.expGateB)
							} else {
								mvid(l.mvg, l.gate, bs.eg, sel)
							}
							if l.expUpB != nil {
								mvidE(l.moeUp, l.mvu, l.up, bs.eact, l.expUpB, bs.eg)
							} else {
								mvidE(l.moeUp, l.mvu, l.up, bs.eact, bs.eg)
							}
						} else {
							mvid(l.mvg, l.gate, bs.eg, sel)
							mvid(l.mvu, l.up, bs.eu, sel)
							eg, eu := bs.eg, bs.eu
							// The experts' biases are indexed by the true ids in rsel,
							// even where the weights are a compacted bank read 0..k-1.
							if l.expGateB != nil {
								lc.la(bs.ebiasFF, ffnW, bs.eg, bs.rsel, l.expGateB, bs.egB)
								eg = bs.egB
							}
							if l.expUpB != nil {
								lc.la(bs.ebiasFF, ffnW, bs.eu, bs.rsel, l.expUpB, bs.euB)
								eu = bs.euB
							}
							if bs.actMulW != nil {
								// Llama 4: the weight scales gate and up, and the
								// downs are summed at one.
								lc.la(bs.actMulW, ffnW, eg, eu, bs.rw, bs.eact)
								rw = bs.wOnes
							} else {
								lc.la(bs.actMulE, ffnW, eg, eu, bs.eact)
							}
						}
						// One Quantize over all the slots, writing a flat [scales][sums] pAX,
						// which is the layout down's SlotAct per-slot bases expect.
						lc.la(bs.quantX, kernels.QuantizeThreads(ffnW/32), bs.eact, bs.a, bs.ax)
						edown := bs.edown
						if l.moeDown != nil {
							mvidE(l.moeDown, l.mvd, l.down, bs.edown, l.expDownB)
						} else {
							mvid(l.mvd, l.down, bs.edown, sel)
							if l.expDownB != nil {
								lc.la(bs.ebiasD, bs.nUsed*p.NEmbd, bs.edown, bs.rsel, l.expDownB, bs.edownB)
								edown = bs.edownB
							}
						}
						lc.la(bs.combine, p.ExpWidth(), edown, rw, eout)
						if latent {
							g.k3LatentOut(lc, bs, l, 1, mvrun)
						}
					}
				}
				if p.DenseMoE {
					// The experts' post-norm, then the dense MLP's normed output
					// (bs.shsum) added: the two branches' sum, which the post-FFN
					// norm below takes like any block's FFN output.
					lc.la(bs.normPart, R*bs.parts, bs.mvOut, bs.part)
					lc.la(bs.normApply, R*p.NEmbd, bs.mvOut, l.nPost2, bs.part, bs.h)
					lc.la(bs.addE, R*p.NEmbd, bs.shsum, bs.h, bs.mvOut)
				}
				// The always-on expert is added after the routed sum, not into it: the
				// routed weights may be renormalised by their own sum (see engine/model/moe.go).
				if bs.shAdd != nil && !p.DenseMoE &&
					(l.shGate != nil || ungatedFFN(p)) && l.shUp != nil && l.shDown != nil {
					if l.shRouter != nil {
						lc.la(bs.shRouterK, kernels.SharedRouterThreads(R), l.shRouter, bs.h, bs.shlogit)
					}
					// Re-quantize h: quantX overwrote bs.a/bs.ax with the expert
					// activations.
					lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.h, bs.a, bs.ax)
					switch {
					case ungatedFFN(p):
						// Ungated (Nemotron 3): up and the activation alone.
						mvrun(l.mvsu, l.shUp, bs.shu)
						lc.la(bs.shActMul, R*bs.nShExp, bs.shu, bs.shact)
					case l.shGU != nil && R == 1 && !g.TableSplit:
						// gate and up in one launch (fuseSegs).
						if err == nil {
							err = lc.launch(l.shGU.kern, l.shGU.blocks, l.shGU.w, bs.a, bs.ax,
								l.shGate.qs, l.shGate.d, l.shGate.sc, bs.shg,
								l.shUp.qs, l.shUp.d, l.shUp.sc, bs.shu)
						}
						g.SegFused++
					default:
						mvrun(l.mvsg, l.shGate, bs.shg)
						mvrun(l.mvsu, l.shUp, bs.shu)
					}
					if !ungatedFFN(p) {
						lc.la(bs.shActMul, R*bs.nShExp, bs.shg, bs.shu, bs.shact)
					}
					lc.la(bs.shQuant, kernels.QuantizeThreads(R*bs.nShExp/32), bs.shact, bs.a, bs.ax)
					mvrun(l.mvsd, l.shDown, bs.shout)
					// An ungated shared expert is a plain add: no logit makes sigma()
					// exactly one, so the multiply goes rather than being neutralised.
					if l.shRouter != nil {
						lc.la(bs.shAdd, R*p.NEmbd, bs.mvOut, bs.shout, bs.shlogit, bs.shsum)
					} else {
						lc.la(bs.addE, R*p.NEmbd, bs.mvOut, bs.shout, bs.shsum)
					}
					ffnOut = bs.shsum
				}
			} else if !skipFFN {
				// A non-causal block's own MLP, which may not be the set's
				// (layer.nffn): its width, gating and activation.
				ungated, nffn, actK := ungatedFFN(p), p.NFFN, bs.actMul
				if l.nonCausal {
					ungated, nffn = l.ungated, l.nffn
					if l.actK != nil {
						actK = l.actK
					}
				}
				gIn, uIn := bs.g, bs.u
				guDone := false
				if l.clampB != nil && !ungated {
					// Gemma 4's clipped linears: gate and up each read their own
					// clamped copy of the normed rows, and the activation their
					// clamped outputs.
					for s, pr := range [2]struct {
						m       mv
						r       *resident
						out, cl backend.Buf
					}{{l.mvg, l.gate, bs.g, bs.gClamp}, {l.mvu, l.up, bs.u, bs.uClamp}} {
						lc.la(bs.clampK[8+2*s], R*p.NEmbd, bs.h, l.clampB, bs.cin)
						lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.cin, bs.a, bs.ax)
						fsrc = bs.cin
						mvrun(pr.m, pr.r, pr.out)
						lc.la(bs.clampK[9+2*s], R*p.NFFN, pr.out, l.clampB, pr.cl)
					}
					gIn, uIn = bs.gClamp, bs.uClamp
				} else if ffnIn {
					mvIn(l.mvg, l.mvg.kernIn, l.gate, bs.g, bs.x2, l.nFFN, nil)
				} else if guOK := g.GateUpFuse && l.gu != nil && l.guDown != nil && R == 1 && rag == nil && !voltaOnly && !ungated &&
					l.nSubFFN == nil && l.nPostFFN == nil && !resScaled && !skip["add"] && !skip["actmul"] && !skip["quantf"] &&
					!g.TableSplit && l.mvd.k == nffn && l.k3 == nil && l.q4e == nil && !l.nonCausal && !l.sparse; guOK {
					// Gate and up in one launch over one operand, and down
					// forming act(gate)*up as it quantizes, with the residual
					// add (fuseGateUp). Every row keeps its own reduction.
					if err == nil {
						err = lc.launch(l.gu.kern, l.gu.blocks, l.gu.w, bs.a, bs.ax,
							l.gate.qs, l.gate.d, l.gate.sc, bs.g, l.up.qs, l.up.d, l.up.sc, bs.u)
					}
					if gateUpSwapped {
						mvIn(l.mvd, l.guDown, l.down, bs.x, bs.u, bs.g, bs.x2)
					} else {
						mvIn(l.mvd, l.guDown, l.down, bs.x, bs.g, bs.u, bs.x2)
					}
					g.GateUpFused++
					ffnFused = true
					guDone = true
				} else if !ungated {
					mvrun(l.mvg, l.gate, bs.g)
					// Gemma 3n's sparse lead: the gate's gaussian top-k.
					gIn = g.emitGauss(lc, bs, l, R)
				}
				if !guDone {
					gated := false
					switch {
					case ffnIn && l.mvu.gatedIn != nil:
						mvIn(l.mvu, l.mvu.gatedIn, l.up, bs.act, bs.x2, l.nFFN, gIn)
						gated = true
					case ffnIn:
						mvIn(l.mvu, l.mvu.kernIn, l.up, bs.u, bs.x2, l.nFFN, nil)
					default:
						// BitNet normalises the product before down, so up does not
						// form it in its own store.
						gated = !ungated && l.clampB == nil && l.nSubFFN == nil && !skip["actmul"] && mvgate(l.mvu, l.up, bs.act, gIn)
						if !gated && l.clampB == nil {
							mvrun(l.mvu, l.up, bs.u)
						}
					}
					// The down projection building its own operand (fuseActIn):
					// quantizing act(gate)*up where up wrote it (downIn), or
					// forming it from gate and up as it quantizes (downAct).
					downOK := R == 1 && rag == nil && g.actInOn() && !voltaOnly && l.clampB == nil && l.nPostFFN == nil && l.nSubFFN == nil &&
						!resScaled && !skip["add"] && !skip["quantf"] && !skip["actmul"] && l.k3 == nil && l.q4e == nil && !l.nonCausal &&
						!(g.TableSplit && l.mvd.alt != nil) && l.mvd.k == nffn
					downIn := downOK && gated && l.mvd.resIn != nil
					downAct := downOK && ffnIn && !gated && l.mvd.resActIn != nil
					// The activation writes the down projection's operand where it
					// can: binary16 for a binary16 GEMM, int8 for the rest.
					var actConv backend.Kernel
					quantDoneF := downAct
					if !skip["actmul"] && !gated && !ungated && l.clampB == nil && !downAct {
						if voltaOnly {
							if ak, conv, n := g.actF16For(bs, l.mvd, R); ak != nil {
								lc.la(ak, n, gIn, uIn, g.f16Buf)
								actConv = conv
								g.ActF16Fused++
							}
						} else if bs.quantAct != nil && !skip["quantf"] && !g.NoOperandFuse && l.nSubFFN == nil {
							lc.la(bs.quantAct, kernels.QuantizeThreads(R*p.NFFN/32), gIn, uIn, bs.act, bs.a, bs.ax)
							fsrc = bs.act
							quantDoneF = true
							g.QuantActs++
						}
					}
					if !skip["actmul"] && !gated && actConv == nil && !quantDoneF {
						if ungated && l.xielu != nil {
							// Apertus: the block's four numbers ride as a buffer.
							lc.la(actK, R*nffn, bs.u, l.xielu, bs.act)
						} else if ungated {
							// One operand: the gate does not exist.
							lc.la(actK, R*nffn, bs.u, bs.act)
						} else {
							lc.la(actK, R*nffn, gIn, uIn, bs.act)
						}
					}
					if voltaOnly {
						fsrc = bs.act
						g.QuantSkipped++
						if actConv != nil {
							actMemoK, actMemoSrc = actConv, bs.act
						}
					} else if l.clampB != nil && !quantDoneF {
						lc.la(bs.clampK[12], R*p.NFFN, bs.act, l.clampB, bs.cin)
						lc.la(bs.quantF, kernels.QuantizeThreads(R*p.NFFN/32), bs.cin, bs.a, bs.ax)
						fsrc = bs.cin
					}
					if downIn || downAct {
						g.ActInSaved++
					} else if l.nSubFFN != nil {
						// BitNet: the norm over act(gate)*up and the quantize down
						// reads, one launch a row.
						laRows(bs.subQF, bs.act, l.nSubFFN, bs.sub, bs.a, bs.ax)
						fsrc = bs.sub
					} else if !voltaOnly && l.clampB == nil && !skip["quantf"] && !quantDoneF {
						lc.la(bs.quantF, kernels.QuantizeThreads(R*nffn/32), bs.act, bs.a, bs.ax)
					}
					if downIn {
						mvIn(l.mvd, l.mvd.resIn, l.down, bs.x, bs.act, nil, bs.x2)
						ffnFused = true
					} else if downAct {
						// pIn is the gate's row and the norm-weight slot carries up's.
						mvIn(l.mvd, l.mvd.resActIn, l.down, bs.x, gIn, uIn, bs.x2)
						ffnFused = true
					} else if l.clampB == nil && l.nPostFFN == nil && !resScaled && !skip["add"] && l.k3 == nil && l.q4e == nil && mvres(l.mvd, l.down, bs.x, bs.x2) {
						ffnFused = true
					} else {
						mvrun(l.mvd, l.down, bs.mvOut)
						if l.clampB != nil {
							lc.la(bs.clampK[13], R*p.NEmbd, bs.mvOut, l.clampB, bs.cout)
							ffnOut = bs.cout
						}
					}
				}
			}
			// The post-FFN norm, same shape and the same reasoning: bs.h
			// held the FFN pre-norm and the gate/up matvecs have read it.
			fout := ffnOut
			if l.nPostFFN != nil && !skip["norm"] {
				lc.la(bs.normPart, R*bs.parts, ffnOut, bs.part)
				lc.la(bs.normApply, R*p.NEmbd, ffnOut, l.nPostFFN, bs.part, bs.h)
				fout = bs.h
			}
			if l.ds4 != nil {
				g.ds4Post(lc, bs, R, bs.x2, fout, bs.x)
			} else if l.q4e != nil {
				g.q4ePostFFN(lc, bs, R, fout)
			} else if l.k3 != nil {
				g.k3PostFFN(lc, bs, R, fout)
			} else if !skip["add"] && !ffnFused {
				addRes(bs.x2, fout, bs.x)
			}
			// BERT's output norm, over x + mlp, out of place and back.
			if l.nResid != nil && !skip["norm"] {
				norm(bs.x, l.nResid, l.nResidB)
				lc.la(bs.copyRows, R*p.NEmbd, bs.h, bs.x)
			}
			// Gemma 4's per-layer embedding: x += post_norm(W_out ·
			// (gelu(W_gate · x) * pl[li])), the sum through x2 and back.
			if l.altRouter != nil {
				// Gemma 3n: every stream corrected and the per-layer input
				// gated into the rest (altup.go).
				g.emitAltCorrect(lc, bs, l, R, mvrun, norm)
			} else if l.pleGate != nil && bs.pleIn != nil {
				lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.x, bs.a, bs.ax)
				mvrun(l.mvpg, l.pleGate, bs.pleG)
				lc.la(bs.pleSlice, R*p.PLEDim, bs.pleIn, l.pleOff, bs.pleS)
				lc.la(bs.pleAct, R*p.PLEDim, bs.pleG, bs.pleS, bs.pleA)
				lc.la(bs.pleQuant, kernels.QuantizeThreads(R*p.PLEDim/32), bs.pleA, bs.a, bs.ax)
				mvrun(l.mvpp, l.pleProj, bs.pleO)
				lc.la(bs.normPart, R*bs.parts, bs.pleO, bs.part)
				lc.la(bs.normApply, R*p.NEmbd, bs.pleO, l.nPLE, bs.part, bs.h)
				addRes(bs.x, bs.h, bs.x2)
				lc.la(bs.scaleX, R*p.NEmbd, bs.x2, bs.one, bs.x)
			}
			// Gemma 4's layer_scalar on the whole output row, out of place:
			// x times the block's scalar into h (dead past the add), and back.
			if l.outScale != nil && bs.scaleX != nil {
				lc.la(bs.scaleX, R*p.NEmbd, bs.x, l.outScale, bs.h)
				lc.la(bs.scaleX, R*p.NEmbd, bs.h, bs.one, bs.x)
			}
		}
		if head != nil && (rag != nil || allRows) {
			// The output norm, the projection of the rows that want logits
			// (they lead) and a per-row argmax over them: their tokens and
			// logits come home. A chunk asking for every row's logits
			// (nn.Head.RowLogits on Layers) takes the same tail.
			hr := g.bs.head
			norm(bs.x, hNorm, g.bs.hNormB)
			lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.h, bs.a, bs.ax)
			one, oneOK := mv{}, false
			if ragHead == 1 {
				one, oneOK = g.headOneMV(R) // compiled by prepRagged
			}
			if m := one; oneOK {
				// One row: decode's head matvec at decode's split, reading
				// the activation's first row -- the wanted row's -- and
				// writing that row's logits, the first of ragLogits. A
				// batched twin would project every row of the step to read
				// back one.
				if rowsFaulted("headone") {
					// Decode's own kernel, which takes the sums to follow
					// the first row's scales: they follow all R rows'.
					m.kern = g.bs.mvHead.kern
				}
				out := bs.ragLogits
				if m.split > 1 && !m.group {
					out = g.partBuf
				}
				nt := m.rows * m.split
				if m.rowt > 1 {
					nt /= m.rowt
				}
				lamv(m, m.kern, nt, hr.qs, dOf(hr), hr.sc, bs.a, bs.ax, out)
				if m.split > 1 && !m.group {
					lc.la(m.red, m.rows, g.partBuf, bs.ragLogits)
				}
			} else {
				hm := bs.ragHead
				if gm, ok := g.groupMV(g.headMV(), ragHead, R); ok {
					hm = gm // the wanted rows only
				}
				runBatched(hm, hr, bs.ragLogits, bs.a, bs.ax, dOf(hr), fsrc)
			}
			// The final softcap, then the argmax over the capped rows: over
			// the wanted rows only, as the one-row and few-row heads project
			// nothing past them, and no row past them is read.
			if bs.ragCapK != nil && !rowsFaulted("nocap") {
				lc.la(bs.ragCapK, ragHead*hr.nrows, bs.ragLogits, bs.ragCap)
			}
			if err == nil {
				err = lc.launch(bs.ragArgmax, ragHead, kernels.ArgmaxGroup, bs.ragOut(), bs.ragTok)
			}
		} else if fold {
			// The chunk's last row, into the decode scratch, and the decode
			// scratch's head sequence on it -- mvrun's one-row arm, spelled
			// out because mvrun's R is the chunk's.
			hr := hb.head
			lc.la(hb.spanK, p.NEmbd, bs.x, hb.x, hb.spanOff)
			if hNorm == nil {
				lc.la(hb.copyRows, p.NEmbd, hb.x, hb.h)
			} else {
				lc.la(hb.normPart, hb.parts, hb.x, hb.part)
				lc.la(hb.normApply, p.NEmbd, hb.x, hNorm, hb.part, hb.h)
			}
			lc.la(hb.quantE, kernels.QuantizeThreads(p.NEmbd/32), hb.h, hb.a, hb.ax)
			// The label pick projects the picked rows alone (pickMV).
			pickOnly := pick && g.pickMV != nil && hb.headCap == nil
			if pickOnly {
				lc.la(g.pickMV, pickMax, hr.qs, hr.d, hr.sc, hb.a, hb.ax, g.pickOut, g.pickIDs)
				g.PickRows++
			}
			m := hb.mvHead
			if g.TableSplit && m.alt != nil {
				m = *m.alt
			}
			out := hb.logits
			if m.split > 1 && !m.group {
				out = g.partBuf
			}
			n := m.rows * m.split
			if m.rowt > 1 {
				n /= m.rowt
			}
			if !pickOnly {
				lamv(m, m.kern, n, hr.qs, hr.d, hr.sc, hb.a, hb.ax, out)
				if m.split > 1 && !m.group {
					lc.la(m.red, m.rows, g.partBuf, hb.logits)
				}
				if hb.headCap != nil {
					lc.la(hb.headCap, m.rows, hb.logits, hb.logitsCap)
				}
				if pick {
					lc.la(g.pickK, pickMax, hb.headOut(), g.pickIDs, g.pickOut)
				}
			}
			g.HeadFolds++
		} else if head != nil {
			// The output norm and the vocabulary projection, in the same
			// submission: the residual stream never comes home.
			if hNorm == nil {
				// No output norm: the row as it is.
				lc.la(bs.copyRows, R*p.NEmbd, bs.x, bs.h)
			} else if bs.normVar != nil {
				// A LayerNorm head (C6): its mean, variance and apply, with
				// the output norm's bias (zeros where the file has none).
				norm(bs.x, hNorm, bs.hNormB)
			} else {
				lc.la(bs.normPart, R*bs.parts, bs.x, bs.part)
			}
			// HeadNormRepeat is a probe, not a feature: normApply is idempotent, so
			// each repeat costs exactly one extra tiny dispatch, which prices whether
			// the head's fixed per-token cost is dispatch structure or work (see the
			// Metal entries in AGENTS.md).
			if bs.normVar == nil && hNorm != nil {
				for r := 0; r < g.HeadNormRepeat; r++ {
					lc.la(bs.normApply, p.NEmbd, bs.x, hNorm, bs.part, bs.h)
				}
				lc.la(bs.normApply, p.NEmbd, bs.x, hNorm, bs.part, bs.h)
			}
			lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.h, bs.a, bs.ax)
			// HeadMatvecRepeat is the same idempotent-repeat probe on the head
			// matvec, pricing it by repetition.
			pickOnly := pick && g.pickMV != nil && bs.headCap == nil && R == 1
			if pickOnly {
				// The label pick projects the picked rows alone (pickMV).
				lc.la(g.pickMV, pickMax, bs.head.qs, bs.head.d, bs.head.sc, bs.a, bs.ax, g.pickOut, g.pickIDs)
				g.PickRows++
			} else {
				for r := 0; r < g.HeadMatvecRepeat; r++ {
					mvrun(bs.mvHead, bs.head, bs.logits)
				}
				mvrun(bs.mvHead, bs.head, bs.logits)
			}
			if bs.headCap != nil {
				lc.la(bs.headCap, bs.mvHead.rows, bs.logits, bs.logitsCap)
			}
			if argmax && err == nil {
				err = lc.launch(g.argmaxK, 1, kernels.ArgmaxGroup, bs.headOut(), g.argmaxOut)
			}
			if sampleK > 0 && err == nil {
				err = g.launchSample(lc, bs.headOut(), bs.mvHead.rows, sampleK, samplePen)
			}
			if pick && !pickOnly {
				lc.la(g.pickK, pickMax, bs.headOut(), g.pickIDs, g.pickOut)
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
		// - History streamed from home: its uploads and waits are per call.
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
			r, e := backend.Record(rec, emit)
			g.TCapture += time.Since(tc)
			if e != nil || err != nil {
				// A capture executes nothing, so the work has still to be
				// issued -- the ordinary way, since a capture that failed
				// once will fail again. A recording that came back around a
				// failed launch is complete as far as the driver knows and
				// wrong as far as jitllm does, so it is retired unused.
				g.graphOff, err = true, nil
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
			}
		}
	}

	// Position state is per token, not per block: written once here rather
	// than 54 times across 18 blocks.
	if fold {
		w(hb.spanOff, g.u32b(uint32((nrow-1)*p.NEmbd)))
	}
	if pe != nil {
		hr := g.bs.head
		g.subBytes.embIDs = embIDs(g.subBytes.embIDs, pe.ids, R, hr.nrows)
		w(g.bs.embIds, g.subBytes.embIDs)
		if err == nil {
			err = lc.launch(g.bs.embK, gatherN/128, 128, hr.qs, hr.d, hr.sc, g.bs.embIds, bs.x)
		}
		g.EmbedLaunches++
	} else {
		w(bs.x, f32b(hx))
	}
	if bs.pleIn != nil {
		w(bs.pleIn, f32b(g.pleRows(R*p.PLEWidth)))
	}
	if ropeDev {
		// Four bytes a row in place of NRot*4; the table never crosses the bus.
		w(bs.rpos, rposs)
	} else {
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
	if samplePen {
		// The history changes every token; the recording reads its count.
		w(g.sampleArgs, u32b(head.SampleArgs))
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
		w(bs.dRows, g.recDesc(sid, nrow))
	}
	w(bs.koff, koffs)
	if kst > 0 {
		w(bs.kpos, kposs)
	}
	bs.ds4.ds4Writes(w)
	if bs.pkv != nil {
		bs.pkv.qsa.qsaWrites(w)
	}
	g.q4eWrites(bs.q4e, w)
	if pk := bs.pkv; pk != nil && pk.cur != nil {
		w(pk.pRow, u32view(pk.desc))
		if len(pk.descW) > 0 {
			w(pk.pRowW, u32view(pk.descW))
		}
		if pk.prefill {
			pk.pf.prefillWrites(w)
		}
		if pk.st != nil {
			pk.st.writes(w)
		}
	}
	if bs.mlab != nil && R > 1 {
		// The batched twin of mlaPOff below: each row's rotary key sits at
		// its own cache row plus KVLoraRank -- a padded row's in the trash
		// slot, where its latent went.
		po := g.subBytes.mlaPOff[:0]
		for r := 0; r < R; r++ {
			po = binary.LittleEndian.AppendUint32(po, binary.LittleEndian.Uint32(koffs[4*r:])+uint32(bs.mlab.lat))
		}
		g.subBytes.mlaPOff = po
		w(bs.mlab.pOff, po)
	}
	if bs.mlaPOff != nil {
		// The rotary key's slot inside this position's row: bs.koff holds
		// pos*row (the latent's slot) and the key follows at +KVLoraRank, so the
		// rotation writes straight into the cache.
		w(bs.mlaPOff, g.u32b(uint32(pos*kvDim+bs.mlaLat)))
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
		if err = s.Read(g.sampleV, f32b(head.SampleVals[:sampleK])); err == nil {
			err = s.Read(g.sampleI, u32b(head.SampleIDs[:sampleK]))
		}
		head.Sampled = err == nil
		// Counted here, not at the launches, which a replayed recording skips.
		g.SampleReads++
		g.SampleLaunches += 2
		if samplePen {
			g.SampleLaunches++
		}
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
		// Every stream comes home, stream-major at R rows (streamRows cuts it).
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
	pn, koffs, kposs, rposs []byte
	tok                     []byte // a ragged step's tokens, read home
	mlaPOff                 []byte // a batched latent attention's rotary key slots
	embIDs                  []byte // a prompt chunk's ids for the gather on the card (embIDs)
	one                     [4]byte
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

// accSplitFor picks how many threads share one attention output: enough to
// fill the device, rounded down to a power of two, capped at 32 (past that the
// partials and the reduction cost more than the occupancy buys). It is derived,
// not tuned, and not context-aware: each thread's trip count is ceil((n-s)/S),
// zero for splits past n, so a constant S is correct at every depth and no
// bucket boundary ever retires a CUDA graph.
//
// pin is WithMatVecAccSplit's value; see there for why.
func accSplitFor(qdim, slots, pin int) int {
	if n := pin; n >= 1 {
		return n
	}
	// 16384 only when the runtime would not say (Vulkan and Metal); CUDA
	// reports the real figure.
	if slots <= 0 {
		slots = 16384
	}
	if qdim <= 0 {
		return 1
	}
	sp := 1
	for sp < 32 && qdim*sp*2 <= slots {
		sp *= 2
	}
	return sp
}

// auxBufs is every per-layer device vector that a page-out and a release both
// drop. It is one list because separate copies drifted repeatedly, each time
// leaking a new kind of vector (vision MLP biases, post-norms, a linear block's
// five, gpt-oss's sinks and biases). Both callers are full teardowns, so
// nothing here is re-read without being uploaded again.
func (l *layer) auxBufs() []backend.Buf {
	return append(append(l.convBufs(),
		l.nAttn, l.nFFN, l.nAttnB, l.nFFNB, l.nQ, l.nK,
		l.nResid, l.nResidB, // BERT's post-norm
		l.nPostAttn, l.nPostFFN, // gemma2/gemma3
		l.nSubAttn, l.nSubFFN, // BitNet
		l.clampB,      // Gemma 4's tower: its clipped linears' bounds
		l.nQA, l.nKVA, // MLA's two latent norms
		l.nIdxK, l.nIdxKB, l.nIdxQ, // the indexers' norms (DeepSeek V3.2, MiniMax-M3)
		l.altRouter, l.altRouterNorm, l.altPredT, l.altCorrT, l.altCorrScale, l.nLaurel, // Gemma 3n
		l.ssmConv1d, l.ssmA, l.ssmDt, l.ssmNorm, l.ssmOnes, // a linear block
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
		l.nFFN2, l.nPost1, l.nPost2, l.nRouter, l.expScale, // Gemma 4's mixture block
		l.nPLE, l.pleOff, // Gemma 4's per-layer embedding
	), append(append(l.ds4.bufs(), l.k3.bufs()...), l.q4e.bufs()...)...) // DeepSeek V4's, Kimi-K3's, Qwen3.8-Flash-Next's
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
	if l.mvRouter != nil {
		l.mvRouter.Close()
	}
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
	if l == nil {
		return
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
	if l.mvRouter != nil {
		l.mvRouter.Close()
	}
	delete(g.layers, li)
	g.layerGen++
}

// afterRelease is what every release ends with. Callers hold g.mu.
func (g *devTier) afterRelease() {
	// A pool no session holds a place in, and the seats of sessions that hold
	// none, go back to the budget.
	g.recTidy()
	// The last tower block out takes its set's scratch with it.
	g.dropIdleTower()
	// The last block out takes the unpack staging with it, or every
	// relocation would leak one widest-tensor allocation.
	if len(g.layers) == 0 {
		g.freeRaw()
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
	// The crossing sample is this session's too, and session ids never repeat,
	// so a map left to grow is a leak of one small struct per closed State.
	delete(g.mv, sid)
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
// model.State.rconv/rstate are what every host consumer reads (including the
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
		// and there is nothing to bring home.
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
			// directly would bring every other session's state below it home
			// too.
			err := func() error {
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
		if l.pool.s[i] == nil {
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
		if !ok || x.Rows <= 0 || x.K <= 0 || len(x.Data) == 0 ||
			(x.K%q.Elems() != 0 && subFor(q, x.K) == 0) {
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

// errNoBatch is what a block reports when a matvec in it has no batched twin --
// a mixture of experts, where a token batch and an expert batch want the
// same slot machinery for two different things.
var errNoBatch = errors.New("tier: no batched matvec for this block")

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

// prepBatch builds the batched scratch and the batched twin of every matvec,
// on the first prompt that asks for one rather than at PrepLayer time.
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
	tok := g.bbs[width].tok
	bb := g.bbs[width]
	// The walk is ~25 kernel lookups a block, every prompt (0.9 ms of a
	// 512-row Qwen3-0.6B prompt on a Volta-class card); it only has something
	// to find after a block was placed or released.
	if bb.walked == g.layerGen+1 {
		return true
	}
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue
		}
		// Only this scratch set's blocks: another set's are prepared with its
		// own batched scratch (gemma4.go), which this one may not be able to
		// run -- MiniMax-M3's dense lead set has no mixture scratch for the
		// blocks after it.
		if g.geoUsed && l.geo != g.geoCur {
			continue
		}
		// A recurrent block batches through GatedDeltaScan (or the fused rule,
		// or Mamba-1's scan); a scratch that could not build one refuses here.
		if l.linear && !bb.rec.ShortConv && bb.deltaScan == nil && bb.deltaFused == nil && bb.m1Scan == nil {
			g.LastErr = "tier: batched prefill: no chunked delta rule for a recurrent block"
			return false
		}
		// Every matvec a batched block can launch (prepLayer's list), or the rest
		// would compile lazily inside the session.
		for _, m := range []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd,
			&l.mvsg, &l.mvsu, &l.mvsd, &l.mvSG, &l.mvBA, &l.mvSO,
			&l.mvQA, &l.mvQB, &l.mvKVA, &l.mvKB, &l.mvVB,
			&l.mvFA, &l.mvFB, &l.mvGA, &l.mvGB, &l.mvpg, &l.mvpp,
			&l.mvIQ, &l.mvIK, &l.mvIW, &l.mvLL, &l.mvLR, &l.mvSI, &l.mvMQ,
			&l.mvCKV, &l.mvCG, &l.mvICKV, &l.mvICG,
			&l.mvHAD, &l.mvHAU, &l.mvHAI, &l.mvHFD, &l.mvHFU, &l.mvHFI, &l.mvNGK, &l.mvNGV} {
			if m.kern == nil {
				continue
			}
			if m.slots > 0 && l.mla && (m == &l.mvKB || m == &l.mvVB) {
				// MLA's absorb banks run grouped with the head as the expert
				// (mlabatch.go): every head's run is the whole chunk. A float
				// bank reads the float activation (emitMLARows' grouped).
				if bb.mlab == nil {
					return false
				}
				if _, ok := g.groupedMV(*m, bb.mlab.nh, bb.mlab.nh*width); !ok {
					return false
				}
				continue
			}
			if m.slots > 0 {
				// A mixture runs grouped (moegroup.go): a streamed bank holds
				// only the selection.
				mg := bb.mg
				// A hybrid block routes the rows on the device and runs their
				// experts on the host (emitGroupedMoE): the routing scratch is
				// all it needs, and no grouped expert kernel.
				if mg != nil && l.stream != nil && l.stream.hybrid {
					continue
				}
				if mg == nil || l.stream != nil {
					// Named either way: a refusal with a stale LastErr reads as
					// some other block's failure.
					g.LastErr = "tier: batched mixture: a streamed bank"
					if mg == nil {
						g.LastErr = fmt.Sprintf("tier: batched mixture: a block has experts and this "+
							"%d-row scratch has no mixture set", width)
					}
					return false
				}
				// A float bank has no scales to quantize against: the grouped
				// matvec reads the sorted float rows (moeFloatHalf), on
				// sm_70 too, where the tensor-core twin takes no float.
				if kernels.IsFloat(m.q) {
					if !g.moeFloatHalf(mg, &bb.p) {
						return false
					}
					if _, ok := g.groupedMV(*m, mg.nExp, mg.p); !ok {
						return false
					}
					continue
				}
				// On sm_70 the tensor-core twin or nothing: the scratch holds no
				// dp4a half there (newMoeGroup), so a bank without a Volta form
				// refuses the batch and the chunk goes row by row.
				if mg.volta {
					bias, nsrc := moeVoltaArgs(l, m, mg)
					if _, ok := g.voltaGroupedMV(*m, mg.nExp, mg.p, mg.unit, bias, nsrc); !ok {
						return false
					}
					continue
				}
				if _, ok := g.groupedMV(*m, mg.nExp, mg.p); !ok {
					return false
				}
				continue
			}
			if _, ok := g.batchMV(*m, width, tok); !ok {
				return false
			}
		}
	}
	// DeepSeek V4's grouped output at this width (ds4.go).
	for _, l := range g.layers {
		if l != nil && l.ok && !g.ds4Twins(l, width) {
			return false
		}
	}
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
// batched kernels' batchGrain, or on a device that runs a chunk's attention
// as FlashPrefill80 the row block that kernel takes (64 rows; a 32-row grain
// sent every width that is not a multiple of 64 to the staged kernels and
// the int8 operands). Callers hold g.mu.
func (g *devTier) promptGrain() int {
	if !g.kb.noAttnMMA && g.f16GemmK() == 16 {
		return max(batchGrain, 16*4)
	}
	return batchGrain
}

// actWinFor is the activation window of the scratch set in use: a tier with a
// vision tower has a set per segment, each with its plan, and the window is a
// codegen input both tiers must agree on (nn.LayerPlan.ActWin).
func (g *devTier) actWinFor(ntok int) int {
	if g.bs != nil {
		return g.bs.p.ActWin
	}
	return 32
}

// batchMV compiles one matvec's batched twin for a prompt chunk: the staged
// int8 GEMM where Config.PreferGemmInt8 asks for it (int8Gemm, sm_80 on), the
// staged binary16 GEMM on the m16n8 instruction where the device has it
// (f16Gemm, sm_75 on), then intBatchMV's forms.
func (g *devTier) batchMV(m mv, ntok, tok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if g.PreferTiles {
		if bm, ok := g.tileMV(m, ntok); ok {
			return bm, true
		}
	}
	if bm, ok := g.int8Gemm(m, ntok); ok {
		return bm, true
	}
	if bm, ok := g.f16Gemm(m, ntok); ok {
		return bm, true
	}
	return g.intBatchMV(m, ntok, tok)
}

// intBatchMV is batchMV without the m16n8 GEMM: the integer matrix
// instruction wherever the device has one (MatVecMMA is faster than the dp4a
// tile and bit-identical to it), then the other tiled forms, then the dp4a
// tile. The widest admissible warp tile is a divisor calculation (mmaTile),
// not a search. A ragged step takes it directly (ragMV): its few rows a
// sequence are decode's, and keep decode's int8 arithmetic.
func (g *devTier) intBatchMV(m mv, ntok, tok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	// A float weight has no tensor-core kernel. Asking would fail, and a
	// failure here switches MMA off for every other matvec (mmaOff).
	if mt, nt, ok := g.mmaTile(m.rows, ntok); ok && !kernels.IsFloat(m.q) {
		kk := kernKey{m.q, m.k, m.rows, -(ntok*100 + mt*10 + nt), m.bias != nil, 1, false, false, 0}
		c, cached := g.kerns[kk]
		if !cached {
			ker, err := kernels.MatVecMMA(kernels.MatVecShape{Center: g.center,
				T: m.q, K: m.k, Rows: m.rows, NTok: ntok, MT: mt, NT: nt,
				Bias: m.bias != nil, ActWin: g.actWinFor(ntok)})
			if err == nil {
				c, err = g.dev.Compile(ker)
			}
			if err != nil {
				// A backend without a matrix instruction is a property of the
				// device, not the shape, so it is recorded once instead of every
				// matvec re-discovering it with a Compile call.
				g.mmaOff, g.MMAWhy = true, err.Error()
				c = nil
			}
			g.kerns[kk] = c
		}
		if c != nil {
			return mv{kern: c, rows: m.rows, split: 1, bias: m.bias, q: m.q, k: m.k,
				threads: (m.rows / (16 * mt)) * (ntok / (8 * nt)) * 32}, true
		}
	}
	if bm, ok := g.tileMV(m, ntok); ok {
		return bm, true
	}
	if bm, ok := g.voltaMV(m, ntok); ok {
		return bm, true
	}
	return g.batchMVDot4(m, ntok, tok)
}

// mmaTile is the widest warp tile this shape admits: MT blocks of 16 rows by NT
// blocks of 8 token columns, each capped at the measured optimum and reduced to
// a divisor of the shape.
//
// 4x4 is measured, not derived: wider tiles run out of registers (MT*NT*4
// accumulators a lane) and narrower ones read the weights more times per chunk.
// mvbench -mma has the per-shape isolated numbers.
func (g *devTier) mmaTile(rows, ntok int) (int, int, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if g.mmaOff || g.NoMMA {
		return 0, 0, false
	}
	mt, nt := g.tiles.mmaMT, g.tiles.mmaNT
	for mt > 1 && rows%(16*mt) != 0 {
		mt /= 2
	}
	for nt > 1 && ntok%(8*nt) != 0 {
		nt /= 2
	}
	// Below the smallest tile the shape simply cannot be covered by a warp: 16
	// rows and 8 columns are the instruction's own granularity.
	if rows%16 != 0 || ntok%8 != 0 {
		return 0, 0, false
	}
	return mt, nt, true
}

func (g *devTier) batchMVDot4(m mv, ntok, tok int) (mv, bool) {
	// Rows per thread is per shape because it only divides some; the
	// fallback is 2 and then 1.
	rowt := 1
	for _, r := range []int{g.tiles.rowt, 2} {
		if r > 1 && m.rows%r == 0 {
			rowt = r
			break
		}
	}
	// dot4Split passes the bias (this is every non-MMA backend's batched
	// path) and splits k where the tile leaves the grid thin.
	return g.dot4Split(m, ntok, tok, rowt, g.kb.batch.Split)
}

// kStride is the device K cache's transposed stride, or 0 for the row-major
// layout. See kernels.RoPERowsT: the scores kernel reads K with one thread per
// position, so [head][dim][position] coalesces where [position][head][dim] costs
// eight times the memory transactions.
//
// It is one value for the whole session because the rotation writes the layout
// and attention reads it; they cannot disagree. Config.NoKT pins it off, which
// is the bisection switch and the A/B arm.
func (g *devTier) kStride(p *nn.LayerPlan) int {
	if g.NoKT {
		return 0
	}
	// A latent cache is row-major: one row per position is written by two
	// kernels at two offsets and read whole by every head, so a transposed
	// layout would scatter both writes to help only the scores. Row-major
	// also keeps MigrateKV a straight copy.
	if p.MLA() {
		return 0
	}
	if c := g.capped(p); c.MaxSeq != p.MaxSeq {
		return c.MaxSeq + 1
	}
	// One slot past MaxSeq: PrepLayer allocates it for the batched prefill's
	// padded rows (their K goes to the trash slot), so the stride has to include
	// it or the last real position lands on top of another head's first.
	return p.MaxSeq + 1
}

// devKVPage is the granule the device's capacity grows in. It matches the
// host's page so a block's cache has the same shape on both sides of the seam
// and MigrateKV stays a straight copy of a position range.
const devKVPage = 256

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
		a.SWAWindow, a.SWALocal, grew = b.SWAWindow, b.SWALocal, true
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

// kvGrow is one session's cache on one text block, and the buffers a growth
// replaces it with.
type kvGrow struct {
	li     int
	kvp    *kvPair
	nk, nv backend.Buf
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

// copyKVAcrossStride moves the live history from a cache of oldCap positions
// into one of newCap, on the device, with kernels.Restride.
//
// K's transpose makes a growth a reshape (at stride S row e lives at e*S), a
// strided copy a generated kernel does without leaving the card. trans is
// g.kStride's answer rather than an assumption: under WithNoKT or MLA the cache
// is row-major, and a transposing copy would scramble the history.
//
// Positions past the live prefix are left as the fresh buffer holds them:
// attention is bounded by the position count, so nothing reads them.
func (g *devTier) copyKVAcrossStride(kvp *kvPair, nk, nv backend.Buf, kvDim, oldCap, newCap int, trans bool) bool {
	// A shrink keeps the positions the smaller cache has room for; TrimKV only
	// asks for one when none past it are live.
	keep := min(oldCap, newCap) + 1
	if kvp.kc != nil {
		rows, cols, ss, ds := 1, keep*kvDim, keep*kvDim, keep*kvDim
		if trans {
			rows, cols, ss, ds = kvDim, keep, oldCap+1, newCap+1
		}
		if !g.restride(kvp.kc, nk, rows, cols, ss, ds) {
			return false
		}
	}
	// Under MLA there is no V buffer; the value travelled with K.
	if nv == nil || kvp.vc == nil {
		return true
	}
	// V is row-major, so its live prefix is one contiguous run of words.
	words := keep * kvDim
	if g.KVF16 {
		words /= 2
	}
	return g.restride(kvp.vc, nv, 1, words, words, words)
}

// restride runs one kernels.Restride and waits for it. Every block of a growth
// has the same shape, so the kernel is compiled once per shape. One contiguous
// run is a device copy instead: no kernel compiled for every size a buffer
// grows to, and no grid to outgrow (an integrated GPU may dispatch at most
// 65535 groups, 33.5 MB of floats at 128 threads a group).
func (g *devTier) restride(src, dst backend.Buf, rows, cols, srcStride, dstStride int) bool {
	if rows == 1 || cols == srcStride && cols == dstStride {
		if err := g.dev.Copy(dst, 0, src, 0, rows*cols*4); err != nil {
			g.LastErr = err.Error()
			return false
		}
		return true
	}
	key := [4]int{rows, cols, srcStride, dstStride}
	k := g.restrides[key]
	if k == nil {
		ker, err := kernels.Restride(rows, cols, srcStride, dstStride)
		if err != nil {
			g.LastErr = err.Error()
			return false
		}
		if k, err = g.dev.Compile(ker); err != nil {
			g.LastErr = err.Error()
			return false
		}
		if g.restrides == nil {
			g.restrides = map[[4]int]backend.Kernel{}
		}
		g.restrides[key] = k
	}
	if err := k.Launch((rows*cols+127)/128, 128, src, dst); err != nil {
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

// devKVReserve is how many positions of history a device reserves up front,
// before growing (see initKVCap).
const devKVReserve = 2048

// DeclinePlan answers nn.PlanDecliner: would this tier refuse a block of this
// shape, before anyone reads its weights? Empty means it would not.
//
// It is declineReason itself, so the cheap check and the one after the upload
// cannot disagree.
func (g *GPU) DeclinePlan(p *nn.LayerPlan) string { return declineReason(p) }

// partialRotary reports that this block rotates only part of each head, so the
// untouched tail has to be copied into the rope's destination before the rope
// runs. qwen3next rotates 64 of a 256-wide head; every other model rotates the
// whole of it.
func partialRotary(p *nn.LayerPlan) bool { return p.NRot > 0 && p.NRot != p.HeadDim }

// deltaScanOf is the chunked delta rule's geometry for a linear block's plan:
// the key pairing, the decay's axis and the subgroup the device guarantees are
// the plan's and the device's, and nothing else about the scan varies.
func deltaScanOf(r nn.RecurrentPlan, rows, lanes int) kernels.DeltaScan {
	d := kernels.DeltaScan{VHeads: r.VHeads, VDim: r.VDim, KDim: r.KDim,
		Rep: r.VHeads / r.KHeads, PerChan: r.ChanDecay, Rows: rows, Lanes: lanes}
	if r.KeyTiled {
		d.TiledK = r.KHeads
	}
	if r.TailLen > 0 {
		d.Stride = r.StateLen
	}
	return d
}

// emitLinear is the gated delta rule's half of a block submission: everything
// between the pre-norm and bs.mvOut, which is exactly what the attention half
// spans.
//
// The order is engine/model/delta.go's, transcribed rather than re-derived. Where the
// host works in place this works out of place (ir.Validate), so q and k
// ping-pong between two buffers. recPair.cur is the half of the block's pool a
// submission reads; it writes the other, and the flip happens after the
// launches are issued so a captured graph replays the pointers it recorded.
// Every kernel that touches a state finds its sequence's slots in bs.dRows,
// the recurrent descriptor (recDesc). It takes the caller's la and
// mvrun so a linear block shares the launch accounting and batched-twin
// selection.
func (g *devTier) emitLinear(sid uint64, s backend.Session, bs *blockScratch, l *layer,
	li, R int, rag bool, p *nn.LayerPlan, errp *error,
	lc *launcher,
	mvrun func(mv, *resident, backend.Buf),
	mvrunAct func(mv, *resident, backend.Buf, backend.Buf, backend.Buf, backend.Buf)) {
	r := bs.rec
	// The real row count (bs.dRows) matters here where attention would mask:
	// a zero padding column shifted into the state stays there. layersOnce
	// stages it with the other per-call writes, outside emit, through the
	// session's Write (a Buf.Write would deadlock on the CUDA goroutine, and a
	// copy inside a recording invalidates the capture).
	par, has := g.stepPar(sid, l)
	pool := l.pool
	if !has || pool == nil {
		if *errp == nil {
			*errp = fmt.Errorf("tier: block %d has no recurrent state for session %d", li, sid)
		}
		return
	}
	inS, inC := pool.s[par], pool.c[par]
	outS, outC := pool.s[1-par], pool.c[1-par]
	if g.recShared {
		// One persistent state; the step writes the device's scratch.
		inS, outS = pool.s[0], g.recNext
	}
	seqs := 1
	qkDim := r.KHeads * r.KDim
	inner := r.VHeads * r.VDim

	// The pre-norm and its quantization, the same two the attention half runs,
	// over Kimi-K3's residual mix where the block has one.
	in := bs.x
	if l.k3 != nil {
		in = g.k3AttnIn(lc, bs, l, R)
	}
	if l.q4e != nil {
		conv := bs.q4e.ngConv
		if rag && l.q4e.ngram {
			if rk := bs.ragLin[len(g.rag.pos)]; rk != nil {
				conv = rk.ngConv
			}
		}
		in = g.q4eAttnIn(lc, bs, l, R, mvrunAct, inS, outS, conv)
	}
	if l.nAttn == nil {
		// No pre-norm (Qwen3.8-Flash-Next's mix is normed already).
		lc.la(bs.copyRows, R*p.NEmbd, in, bs.h)
	} else {
		lc.la(bs.normPart, R*bs.parts, in, bs.part)
		lc.la(bs.normApply, R*p.NEmbd, in, l.nAttn, bs.part, bs.h)
	}
	lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.h, bs.a, bs.ax)

	if r.SSD {
		g.emitSSD(sid, bs, l, li, R, rag, p, errp, lc, mvrun, pool, par, seqs)
		return
	}
	if r.ShortConv {
		g.emitShortConv(sid, bs, l, li, R, rag, p, errp, lc, mvrun, pool, par)
		return
	}
	if r.Mamba1 {
		g.emitMamba1(sid, bs, l, li, R, rag, p, errp, lc, mvrun, mvrunAct, pool, par, seqs)
		return
	}
	// The three projections. Wq carries the fused q|k|v of a linear block (the
	// layer kind decides the consumer; see nn.SSMWeights).
	mvrun(l.mvq, l.wq, bs.dMixed)
	if r.ChanDecay {
		// KDA's two gates are low-rank: two matvecs with a quantize between, the
		// second reading the rank-r temporary (hence mvrunAct). The output gate
		// goes first because both use dLowRank. Kimi-K3's output gate is one
		// full-rank matrix.
		if l.ssmGA == nil {
			mvrun(l.mvSG, l.ssmGate, bs.dZ)
		} else {
			mvrun(l.mvGA, l.ssmGA, bs.dLowRank)
			lc.la(bs.quantLR, kernels.QuantizeThreads(R*r.KDim/32), bs.dLowRank, bs.dLRa, bs.dLRax)
			mvrunAct(l.mvGB, l.ssmGB, bs.dZ, bs.dLRa, bs.dLRax, bs.dLowRank)
		}
		mvrun(l.mvFA, l.ssmFA, bs.dLowRank)
		lc.la(bs.quantLR, kernels.QuantizeThreads(R*r.KDim/32), bs.dLowRank, bs.dLRa, bs.dLRax)
		mvrunAct(l.mvFB, l.ssmFB, bs.dAlpha, bs.dLRa, bs.dLRax, bs.dLowRank)
		// ssmBA is b_proj alone here: beta's logits, one per value head, with
		// no alpha stream in it to deinterleave.
		mvrun(l.mvBA, l.ssmBA, bs.dBRaw)
	} else {
		mvrun(l.mvSG, l.ssmGate, bs.dZ)
		mvrun(l.mvBA, l.ssmBA, bs.dBA)
	}

	// The convolution and its shift are two kernels because they write two
	// buffers; folding the shift in would load and store the state (refused by
	// ir.Validate). A batched decode's rows are sequences, each with its own
	// window and state at its own seat (growSeat, recDesc), and every row
	// steps once.
	if rag {
		n, runs := len(g.rag.pos), len(g.rag.runs)
		seqs = n
		rk := bs.ragLin[n]
		if rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched linear step for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
		if r.ChanDecay {
			lc.la(rk.delta, runs*rk.perRun, inS, bs.dConv, bs.dAlpha, bs.dBRaw,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		} else {
			lc.la(rk.delta, runs*rk.perRun, inS, bs.dConv, bs.dBA,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		}
		g.LinearFused++
		if rk.lanes == 1 {
			g.LinearRowsScalar++
		}
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
	}
	// One launch from the convolution to the state where the device has a
	// subgroup (kernels.GatedDeltaFused): the SiLU, slices, L2 norms, scales and
	// gates are formed inside the scan, replacing eleven launches.
	if rag {
		// stepped above, one launch over every sequence
	} else if bs.deltaFused != nil {
		if r.ChanDecay {
			lc.la(bs.deltaFused, bs.scanThreads, inS, bs.dConv, bs.dAlpha, bs.dBRaw,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		} else {
			lc.la(bs.deltaFused, bs.scanThreads, inS, bs.dConv, bs.dBA,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		}
		g.LinearFused++
		if R > 1 {
			g.LinearBatched++
		}
	} else {
		lc.la(bs.ssmSiLU, R*r.Chans, bs.dConv, bs.dMixed)

		// The gates, from the projection the convolution did not touch.
		if r.ChanDecay {
			// Two launches at two widths: the decay over VHeads*KDim channels and
			// beta over VHeads heads, each discarding the other's half. The
			// arithmetic is qwen3next's unchanged -- the reference computes
			// exp(-exp(A_log)*softplus(g)) and ssm_a already holds -exp(A_log),
			// negated and broadcast per channel at conversion.
			if r.DecayBound != 0 {
				lc.la(bs.deltaGateChan, R*r.VHeads*r.KDim, bs.dAlpha, l.ssmDt, l.ssmA, bs.dDecay)
			} else {
				lc.la(bs.deltaGateChan, R*r.VHeads*r.KDim, bs.dAlpha, l.ssmDt, l.ssmA,
					bs.dAlpha, bs.dDecay, bs.dGateWaste)
			}
			lc.la(bs.deltaGate, R*r.VHeads, bs.dAlpha, l.ssmDt, l.ssmA, bs.dBRaw,
				bs.dGateWaste, bs.dBeta)
		} else {
			lc.la(bs.splitGates, R*r.VHeads, bs.dBA, bs.dBRaw, bs.dAlpha)
			lc.la(bs.deltaGate, R*r.VHeads, bs.dAlpha, l.ssmDt, l.ssmA, bs.dBRaw, bs.dDecay, bs.dBeta)
		}

		// q, k and v as three buffers, because a launch has no views.
		lc.la(bs.sliceQ, R*qkDim, bs.dMixed, bs.dQ)
		lc.la(bs.sliceK, R*qkDim, bs.dMixed, bs.dK)
		lc.la(bs.sliceV, R*inner, bs.dMixed, bs.dV)

		// The per-key-head L2 norm, then one scale each, then a second on q alone:
		// the host scales all 2*kHeads groups by 1/sqrt(kDim) and the query again,
		// so q ends at scale squared and k at scale.
		gw, gwid := headNormGeom(bs.qkNormL, R*r.KHeads)
		if *errp == nil {
			*errp = lc.launch(bs.qkNorm, gw, gwid, bs.dQ, l.ssmOnes, bs.dQn)
		}
		if *errp == nil {
			*errp = lc.launch(bs.qkNorm, gw, gwid, bs.dK, l.ssmOnes, bs.dKn)
		}
		lc.la(bs.qkScale, R*qkDim, bs.dKn, bs.dScale, bs.dK)
		lc.la(bs.qkScale, R*qkDim, bs.dQn, bs.dScale, bs.dQ)
		lc.la(bs.qkScale, R*qkDim, bs.dQ, bs.dScale, bs.dQn)

		// The rule, out of place: the state arrives through inS and leaves through
		// outS. The scan steps every one of the R real rows (bs.dRows) in order
		// inside one launch; a decode scratch on a device with no subgroup steps
		// its one row with GatedDeltaStep.
		if R > 1 || bs.deltaScan != nil {
			if bs.deltaScan == nil {
				if *errp == nil {
					*errp = fmt.Errorf("tier: block %d: %d rows and no chunked delta rule", li, R)
				}
				return
			}
			lc.la(bs.deltaScan, bs.scanThreads, inS, bs.dK, bs.dQn, bs.dV, bs.dDecay, bs.dBeta,
				bs.dOut, outS, bs.dRows)
			if R > 1 {
				g.LinearBatched++
			}
		} else {
			step := bs.deltaStep
			if r.ChanDecay {
				step = bs.deltaStepChan
			}
			lc.la(step, inner, inS, bs.dK, bs.dQn, bs.dV, bs.dDecay, bs.dBeta,
				bs.dOut, outS, bs.dRows)
		}
	}

	if g.recShared {
		// The new state home, before the next block reuses the scratch: each
		// sequence's row of recNext (its out slot) to its slot of the pool.
		// Compiled before the session (ensureRecNext): Compile inside one
		// deadlocks CUDA's owner goroutine.
		cp := g.recCopies[[2]int{r.StateLen, seqs}]
		if cp == nil && *errp == nil {
			*errp = fmt.Errorf("tier: block %d: no copy home for %d shared recurrent state(s)", li, seqs)
		}
		lc.la(cp, seqs*r.StateLen, g.recNext, pool.s[0], bs.dRows)
	}

	// The gated output norm: RMSNorm per value head against ssm_norm, times
	// SiLU of the z projection.
	vg, vwid := headNormGeom(bs.outNormL, R*r.VHeads)
	if *errp == nil {
		*errp = lc.launch(bs.outNorm, vg, vwid, bs.dOut, l.ssmNorm, bs.dOutN)
	}
	gated := bs.dAct
	if r.ChanDecay && l.k3 != nil && l.k3.fault == nn.K3FaultNoKDAGate {
		gated = bs.dOutN
	} else if r.ChanDecay || r.SigmoidGate {
		// KDA's gate and Qwen3.8-Flash-Next's are sigmoid(z).
		lc.la(bs.outSigMul, R*inner, bs.dZ, bs.dOutN, bs.dAct)
	} else {
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOutN, bs.dAct)
	}
	lc.la(bs.quantD, kernels.QuantizeThreads(R*inner/32), gated, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, bs.mvOut)

	// The flip is after the launches, so a captured graph replays the pointers
	// it recorded. It happens only while the submission is still good: la
	// latches the first error and skips later launches, so a block past a
	// failure must not flip, or the next token reads a half nothing wrote. This
	// does not make a failed submission resumable (blocks before the failure did
	// advance, which is why model.State.forward refuses a host restart).
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		// graphKey reads this block's half (rangeParity), so the next token
		// replays the other recording; with a constant parity one recording
		// would serve every token and the recurrence would freeze.
		g.recSteps++
	}
}

// emitSSD is a Mamba-2 block's half of the submission, emitLinear's twin from
// the projections on: engine/model/delta.go's ssdAttn, transcribed. Three
// projections (C|B|x, z and dt), the convolution and its shift as the delta
// rule runs them, one launch for the whole update -- the bias and SiLU, dt and
// the decay, the state over every row (or every run of a ragged step) and the
// D skip (kernels.GatedDeltaFused with SSD) -- then y*silu(z), its grouped
// norm, and ssm_out into bs.mvOut. The pre-norm and its quantization ran in
// emitLinear; the state is out of place exactly as there.
func (g *devTier) emitSSD(sid uint64, bs *blockScratch, l *layer, li, R int, rag bool, p *nn.LayerPlan,
	errp *error, lc *launcher, mvrun func(mv, *resident, backend.Buf), pool *recPool, par int, seqs int) {
	r := bs.rec
	inS, inC := pool.s[par], pool.c[par]
	outS, outC := pool.s[1-par], pool.c[1-par]
	if g.recShared {
		inS, outS = pool.s[0], g.recNext
	}
	inner := r.VHeads * r.VDim
	// A block that attends too keeps its mixer's projection apart from the
	// attention's query, and leaves its output in bs.side for the sum.
	in, inW, out := l.mvq, l.wq, bs.mvOut
	if l.withAttn {
		in, inW, out = l.mvSI, l.ssmIn, bs.side
	}
	mvrun(in, inW, bs.dMixed)
	mvrun(l.mvSG, l.ssmGate, bs.dZ)
	mvrun(l.mvBA, l.ssmBA, bs.dBRaw)
	if rag {
		n, runs := len(g.rag.pos), len(g.rag.runs)
		seqs = n
		rk := bs.ragLin[n]
		if rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched Mamba-2 step for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
		lc.la(rk.delta, runs*rk.perRun, inS, bs.dConv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmCB, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if rk.lanes == 1 {
			g.LinearRowsScalar++
		}
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
		lc.la(bs.ssdScan, bs.scanThreads, inS, bs.dConv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmCB, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if R > 1 {
			g.LinearBatched++
		}
	}
	if g.recShared {
		cp := g.recCopies[[2]int{r.StateLen, seqs}]
		if cp == nil && *errp == nil {
			*errp = fmt.Errorf("tier: block %d: no copy home for %d shared recurrent state(s)", li, seqs)
		}
		lc.la(cp, seqs*r.StateLen, g.recNext, pool.s[0], bs.dRows)
	}
	// The gated norm: y*silu(z) then the grouped RMSNorm, Mamba's order; or
	// norm(y)*silu(z) where the plan says so; or the gate alone with no norm.
	ng, nwid := headNormGeom(bs.ssdNormL, R*r.NormGroups)
	act := bs.dAct
	switch {
	case r.NoNorm:
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOut, bs.dAct)
	case r.NormBeforeGate:
		if *errp == nil {
			*errp = lc.launch(bs.ssdNorm, ng, nwid, bs.dOut, l.ssmNorm, bs.dOutN)
		}
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOutN, bs.dAct)
	default:
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOut, bs.dOutN)
		if *errp == nil {
			*errp = lc.launch(bs.ssdNorm, ng, nwid, bs.dOutN, l.ssmNorm, bs.dAct)
		}
	}
	lc.la(bs.quantD, kernels.QuantizeThreads(R*inner/32), act, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, out)
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		g.recSteps++
	}
}

// emitShortConv issues one LFM2 short-convolution block, from the quantized
// pre-norm to bs.mvOut: x, B and C, B*x through the convolution (its window is
// the block's whole recurrent state, in pool.c), C times that, out_proj. The
// two products are kernels.ActMul with the identity.
func (g *devTier) emitShortConv(sid uint64, bs *blockScratch, l *layer, li, R int, rag bool, p *nn.LayerPlan,
	errp *error, lc *launcher, mvrun func(mv, *resident, backend.Buf), pool *recPool, par int) {
	r := bs.rec
	inC, outC := pool.c[par], pool.c[1-par]
	mvrun(l.mvq, l.wq, bs.dMixed)
	mvrun(l.mvSG, l.ssmGate, bs.scB)
	mvrun(l.mvBA, l.ssmBA, bs.scC)
	lc.la(bs.scMul, R*r.Chans, bs.scB, bs.dMixed, bs.scBx)
	if rag {
		n, runs := len(g.rag.pos), len(g.rag.runs)
		rk := bs.ragLin[n]
		if rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched short convolution for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.scBx, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.scBx, outC, bs.dRows)
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.scBx, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.scBx, outC, bs.dRows)
		if R > 1 {
			g.LinearBatched++
		}
	}
	lc.la(bs.scMul, R*r.Chans, bs.scC, bs.dConv, bs.scY)
	lc.la(bs.quantE, kernels.QuantizeThreads(R*p.NEmbd/32), bs.scY, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, bs.mvOut)
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		g.recSteps++
	}
}

// emitMamba1 is a Mamba-1 block's half of the submission, engine/model/delta.go's
// mamba1Attn transcribed: x and z, the convolution and its shift as the delta
// rule runs them, x's bias and SiLU, x_proj's three outputs and their norms,
// dt_proj through a quantized bottleneck, the whole scan in one launch
// (kernels.GatedDeltaFused with Mamba1, every run of a ragged step), then
// y*silu(z) and ssm_out into bs.mvOut. The pre-norm and its quantization ran
// in emitLinear; the state is out of place exactly as there. Its four
// matrices ride KDA's slots (nn.SSMWeights): x_proj's dt rows in FA, dt_proj
// in FB, its B and C rows in GA and GB.
func (g *devTier) emitMamba1(sid uint64, bs *blockScratch, l *layer, li, R int, rag bool, p *nn.LayerPlan,
	errp *error, lc *launcher, mvrun func(mv, *resident, backend.Buf),
	mvrunAct func(mv, *resident, backend.Buf, backend.Buf, backend.Buf, backend.Buf),
	pool *recPool, par int, seqs int) {
	r := bs.rec
	inS, inC := pool.s[par], pool.c[par]
	outS, outC := pool.s[1-par], pool.c[1-par]
	if g.recShared {
		inS, outS = pool.s[0], g.recNext
	}
	inner := r.VHeads * r.VDim
	mvrun(l.mvq, l.wq, bs.dMixed)
	mvrun(l.mvSG, l.ssmGate, bs.dZ)
	var rk *ragLinear
	runs := 0
	if rag {
		n := len(g.rag.pos)
		runs, seqs = len(g.rag.runs), n
		if rk = bs.ragLin[n]; rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched Mamba-1 step for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
	}
	// x through its bias and SiLU, into dMixed (dead past the convolution),
	// and quantized for x_proj's three matvecs.
	lc.la(bs.m1Bias, R*r.Chans, bs.dConv, l.ssmCB, bs.dMixed)
	lc.la(bs.quantD, kernels.QuantizeThreads(R*inner/32), bs.dMixed, bs.a, bs.ax)
	mvrun(l.mvFA, l.ssmFA, bs.dLowRank)
	mvrun(l.mvGA, l.ssmGA, bs.m1B)
	mvrun(l.mvGB, l.ssmGB, bs.m1C)
	low, bv, cv := bs.dLowRank, bs.m1B, bs.m1C
	if l.m1DtN != nil {
		// Jamba's three RMSNorms (FalconMamba's of ones), out of place.
		dg, dw := headNormGeom(bs.m1NormDtL, R)
		if *errp == nil {
			*errp = lc.launch(bs.m1NormDt, dg, dw, bs.dLowRank, l.m1DtN, bs.m1LowN)
		}
		cg, cw := headNormGeom(bs.m1NormBCL, R)
		if *errp == nil {
			*errp = lc.launch(bs.m1NormBC, cg, cw, bs.m1B, l.m1BN, bs.m1BN)
		}
		if *errp == nil {
			*errp = lc.launch(bs.m1NormBC, cg, cw, bs.m1C, l.m1CN, bs.m1CN)
		}
		low, bv, cv = bs.m1LowN, bs.m1BN, bs.m1CN
	}
	// dt_proj reads the bottleneck's own quantization.
	lc.la(bs.m1Quant, kernels.QuantizeThreads(R*r.Rank/32), low, bs.dLRa, bs.dLRax)
	mvrunAct(l.mvFB, l.ssmFB, bs.dBRaw, bs.dLRa, bs.dLRax, low)
	if rag {
		lc.la(rk.delta, runs*rk.perRun, inS, bs.dMixed, bv, cv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if rk.lanes == 1 {
			g.LinearRowsScalar++
		}
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.m1Scan, bs.scanThreads, inS, bs.dMixed, bv, cv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if R > 1 {
			g.LinearBatched++
		}
	}
	if g.recShared {
		cp := g.recCopies[[2]int{r.StateLen, seqs}]
		if cp == nil && *errp == nil {
			*errp = fmt.Errorf("tier: block %d: no copy home for %d shared recurrent state(s)", li, seqs)
		}
		lc.la(cp, seqs*r.StateLen, g.recNext, pool.s[0], bs.dRows)
	}
	lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOut, bs.dAct)
	lc.la(bs.quantD, kernels.QuantizeThreads(R*inner/32), bs.dAct, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, bs.mvOut)
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		g.recSteps++
	}
}

// emitMLA issues one multi-head latent attention block's launches, from the
// quantized pre-norm to bs.mvOut, the span emitLinear and the ordinary attention
// half occupy.
//
// The cache holds one row per position for the layer. engine/model/mla.go has the
// derivation; the two identities are
//
//	score_h = q_nope_h . (W_k_h . c)  =  (W_k_hᵀ . q_nope_h) . c
//	out_h   = Σ a_j (W_v_h . c_j)     =  W_v_h . (Σ a_j c_j)
//
// so W_kᵀ folds into the query before the scores and W_v into the output after,
// and no key or value is materialised: a row-wide absorbed query against a
// row-wide cache for the scores, a latent-wide result from the same rows for the
// accumulation. Every launch reuses an existing kernel (the absorb is the
// indexed matvec with the head as the index).
//
// The order is engine/model/mla.go's; in particular kv_a runs before the query's
// second step, since both read bs.a/bs.ax and quantizing the query latent
// overwrites it.
func (g *devTier) emitMLA(s backend.Session, bs *blockScratch, l *layer, kvp *kvPair,
	p *nn.LayerPlan, errp *error,
	lc *launcher,
	mvrun func(mv, *resident, backend.Buf),
	mvid func(mv, *resident, backend.Buf, backend.Buf),
	sm backend.Kernel, smG, smW, nCap int, skip map[string]bool, pc *pagedCall) {
	lat, rot, nope := bs.mlaLat, bs.mlaRot, bs.mlaNopeW
	nh := p.NHead

	// ── the query, in one step or two ────────────────────────────────────
	//
	// One step or two is a property of the model, asked of the plan: V3
	// compresses the query through a latent, V2-Lite's q_lora_rank is null.
	// The query projection and kv_a go in one launch where fuseSegs built it
	// (decode only).
	qkFused := !skip["qkvo"] && l.mlaQK != nil && bs.rows == 1 && !g.TableSplit
	if qkFused {
		qw, qo := l.wq, bs.q
		if p.QLoraRank != 0 {
			qw, qo = l.wqa, bs.mlaQA
		}
		if *errp == nil {
			*errp = lc.launch(l.mlaQK.kern, l.mlaQK.blocks, l.mlaQK.w, bs.a, bs.ax,
				qw.qs, qw.d, qw.sc, qo, l.wkva.qs, l.wkva.d, l.wkva.sc, bs.mlaKV)
		}
		g.SegFused++
	} else if !skip["qkvo"] {
		if p.QLoraRank != 0 {
			mvrun(l.mvQA, l.wqa, bs.mlaQA)
		} else {
			mvrun(l.mvq, l.wq, bs.q)
		}
	}

	// ── the shared latent and the shared rotary key ──────────────────────
	//
	// One projection for both k and v: kv_a emits KVLoraRank+NRot floats, one
	// for the whole layer. It runs here, off the residual's quantization,
	// before the query's second step replaces it.
	if !skip["qkvo"] && !qkFused {
		mvrun(l.mvKVA, l.wkva, bs.mlaKV)
	}
	// The lightning indexer's key and weights read the same quantization.
	idx := pc != nil && pc.pk.idx != nil
	if idx {
		g.idxKeyAndWeights(lc, bs, l, 1, pc, mvrun)
	}
	if p.QLoraRank != 0 {
		// The norm covers the query latent only, out of place (RULE 13).
		if !skip["norm"] {
			lc.la(bs.mlaNormQAPart, bs.mlaQAParts, bs.mlaQA, bs.mlaQAPart)
			lc.la(bs.mlaNormQAApply, p.QLoraRank, bs.mlaQA, l.nQA, bs.mlaQAPart, bs.mlaQAn)
		}
		if !skip["quant"] {
			lc.la(bs.mlaQuantQA, kernels.QuantizeThreads(p.QLoraRank/32), bs.mlaQAn, bs.a, bs.ax)
		}
		if !skip["qkvo"] {
			mvrun(l.mvQB, l.wqb, bs.q)
		}
		if idx {
			g.idxQuery(lc, bs, l, 1, pc, mvrun)
		}
	}

	// ── the latent norm, over the first KVLoraRank floats only ───────────
	//
	// The rotary tail is not normed; the kernel is built for lat elements, so
	// it is untouched by construction.
	if !skip["norm"] {
		lc.la(bs.mlaNormKVPart, bs.mlaParts, bs.mlaKV, bs.mlaKVPart)
		lc.la(bs.mlaNormKVApply, lat, bs.mlaKV, l.nKVA, bs.mlaKVPart, bs.mlaKVn)
	}

	// ── the cache row: the normed latent, then the rotated key ───────────
	//
	// Two writes into one row at two offsets: bs.koff holds pos*row and
	// bs.mlaPOff pos*row+lat, so the rotation lands straight in the cache. The
	// row is the key and its prefix the value (kvPair.value).
	if !skip["rope"] && pc != nil {
		// Paged: the latent at the row's head of this position's page, and
		// the rotated key after it, rotated into staging first.
		lc.la(bs.mlaSliceKPE, rot, bs.mlaKV, bs.mlaKPE)
		pc.writeLatent(lc, 1, bs.mlaKVn, bs.mlaKPE, bs.zero, bs, !p.NoPosEnc)
	} else if !skip["rope"] {
		lc.la(bs.mlaCopyLat, lat, bs.mlaKVn, kvp.kc, bs.koff)
		lc.la(bs.mlaSliceKPE, rot, bs.mlaKV, bs.mlaKPE)
		if p.NoPosEnc {
			lc.la(bs.mlaNoPEK, rot, bs.mlaKPE, kvp.kc, bs.mlaPOff)
		} else {
			lc.la(bs.mlaRopeK, rot/2, bs.mlaKPE, bs.cs, bs.mlaPOff, kvp.kc)
		}
	}

	// ── the query's two halves, compacted ────────────────────────────────
	//
	// The nope half is the head of each head and the rotary half its tail, at
	// stride HeadDim, which neither the rotation nor the absorb can address, so
	// both are compacted (as the host does).
	lc.la(bs.mlaSliceNope, nh*nope, bs.q, bs.mlaNope)
	lc.la(bs.mlaSlicePE, nh*rot, bs.q, bs.mlaPE)
	if !skip["rope"] {
		if p.NoPosEnc {
			lc.la(bs.mlaNoPEQ, nh*rot, bs.mlaPE, bs.mlaPErot, bs.zero)
		} else {
			lc.la(bs.mlaRopeQ, nh*rot/2, bs.mlaPE, bs.cs, bs.zero, bs.mlaPErot)
		}
	}

	// ── absorb W_kᵀ into the query ───────────────────────────────────────
	//
	// One indexed launch, not NHead matvecs: slot h reads bs.mlaNope at h*nope
	// and the bank's sheet h (SlotAct), with the identity selection.
	if !skip["quant"] {
		lc.la(bs.mlaQuantNope, kernels.QuantizeThreads(nh*nope/32), bs.mlaNope, bs.a, bs.ax)
	}
	// Which bank is which is a container fact: attn_k_b is stored already
	// transposed and attn_v_b verbatim, and they can be the same shape. The
	// fault arm swaps them.
	absMV, absW, unMV, unW := l.mvKB, l.wkb, l.mvVB, l.wvb
	if g.kb.mlaFault == MLAFaultSwapBanks {
		absMV, absW, unMV, unW = l.mvVB, l.wvb, l.mvKB, l.wkb
	}
	if !skip["qkvo"] {
		mvid(absMV, absW, bs.mlaAbsRaw, bs.mlaIdent)
	}
	// The row-wide query is assembled by copies at the per-head offsets
	// initScratch wrote, since an indexed matvec writes slot h at h*Rows.
	lc.la(bs.mlaCopyAbs, nh*lat, bs.mlaAbsRaw, bs.mlaAbs, bs.mlaAbsOff)
	lc.la(bs.mlaCopyPE, nh*rot, bs.mlaPErot, bs.mlaAbs, bs.mlaPEOff)

	// ── attention, at two widths over one buffer ─────────────────────────
	if !skip["attn"] && idx {
		if *errp == nil {
			*errp = g.idxAttend(s, lc, 1, pc, bs.mlaAbs, bs.n, bs.mlaAcc)
		}
	} else if !skip["attn"] && pc != nil {
		if *errp == nil {
			*errp = pc.pk.attend(s, lc, pc.pl, bs.mlaAbs, pc.k, pc.k, bs.n, bs.mlaAcc, pc.tab, pc.desc, nil)
		}
	} else if !skip["attn"] {
		if !skip["scores"] {
			kg := (nCap + bs.ktile - 1) / bs.ktile
			if bs.scoresWarp {
				// One warp-wide workgroup per (head, position).
				if *errp == nil {
					*errp = lc.launch(bs.scores, p.NHead*nCap, ir.SubgroupLanes, bs.mlaAbs, kvp.kc, bs.n, bs.att)
				}
			} else {
				lc.la(bs.scores, p.NHead*kg, bs.mlaAbs, kvp.kc, bs.n, bs.att)
			}
		}
		if *errp == nil && !skip["softmax"] {
			// Not la(): the softmax group is not 128 wide. MLA carries no
			// attention sinks -- that is gpt-oss -- so this is the plain arm.
			*errp = lc.launch(sm, smG, smW, bs.att, bs.n, bs.prob)
		}
		// kvp.value() (which is kvp.kc) so the accumulate reads the value by
		// name.
		if !skip["acc"] {
			lc.la(bs.attnAcc, nh*lat, bs.prob, kvp.value(), bs.n, bs.mlaAcc)
		}
	}

	// ── un-absorb W_v and project out ────────────────────────────────────
	//
	// The second gather, with the first's contract: head h consumes its own
	// attention result (bs.mlaAcc at h*KVLoraRank) against its own sheet of
	// attn_v_b.
	if !skip["quant"] {
		lc.la(bs.mlaQuantAcc, kernels.QuantizeThreads(nh*lat/32), bs.mlaAcc, bs.a, bs.ax)
	}
	if !skip["qkvo"] {
		mvid(unMV, unW, bs.mlaOut, bs.mlaIdent)
	}
	// Kimi-K3's output gate (k3GatedMLA) and no bias: the o bias is added by
	// the caller on bs.mvOut.
	if !skip["quant"] {
		lc.la(bs.mlaQuantOut, kernels.QuantizeThreads(nh*p.HeadDimV/32), g.k3GatedMLA(lc, bs, l, 1, bs.mlaOut),
			bs.a, bs.ax)
	}
	if !skip["qkvo"] {
		mvrun(l.mvo, l.wo, bs.mvOut)
	}
}

// ones32 is n float32 ones.
func ones32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// volta70AttnMT and volta70AttnNT are AttnScoresMMA70's warp tile, 64 keys by 32
// queries, and AttnAccMMA70's query tile. The accumulate's 32 query rows are
// also the softmax's group (atile), since it walks its group's widest count.
const volta70AttnMT, volta70AttnNT = 2, 4

// volta70AccMT is AttnAccMMA70's dims per warp in 32s: one, so every 32 dims of
// a head is its own warp. The whole head per warp starved the card (too few
// warps, each walking its causal width serially); one per 32 dims roughly
// halved the accumulate on a Volta-class card.
const volta70AccMT = 1

// volta70LatMT is the same tile for a batched latent block (mlabatch.go): the
// whole row up to 128 dims a warp. It is zero below 32 dims, so every caller
// checks lat >= 32 first.
func volta70LatMT(dims int) int { return min(dims/32, 4) }

// volta70Attn builds bs's batched attention on sm_70's m8n8k4, both products,
// or neither: the scores alone would still need the accumulate's 32-row group
// in the softmax, and the pair is gated as a pair. It is for a device that
// lowers MMAVolta and not m16n8k16 (the caller tried that first), with the
// transposed f32 K cache these read; anything else keeps the FMA tiles.
func (g *devTier) volta70Attn(bs *blockScratch, kvDim, gqa int, scale float32) bool {
	p := &bs.p
	kst := g.kStride(p)
	if g.NoVolta || g.kb.noAttnMMA || g.KVF16 || bs.rows <= 1 || kst <= 0 || p.MLA() ||
		32*volta70AttnMT > maxKeyTile ||
		p.HeadDim%32 != 0 || bs.rows%(8*volta70AttnNT) != 0 || bs.sstride%4 != 0 {
		return false
	}
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			return nil
		}
		c, err := g.dev.Compile(k)
		if err != nil {
			return nil
		}
		return c
	}
	sk := comp(kernels.AttnScoresMMA70(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale, bs.rows, kst,
		volta70AttnMT, volta70AttnNT, 0))
	ak := comp(kernels.AttnAccMMA70(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, bs.rows,
		volta70AccMT, volta70AttnNT))
	var sw backend.Kernel
	if p.SWAWindow > 0 {
		sw = comp(kernels.AttnScoresMMA70(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale, bs.rows, kst,
			volta70AttnMT, volta70AttnNT, windowArg(p)))
	}
	if sk == nil || ak == nil || (p.SWAWindow > 0 && sw == nil) {
		for _, k := range []backend.Kernel{sk, ak, sw} {
			if k != nil {
				closeKern(k)
			}
		}
		return false
	}
	bs.scores70, bs.acc70, bs.scores70W = sk, ak, sw
	bs.atile = 8 * volta70AttnNT
	g.AttnMMA = true
	// The one-kernel form where it takes the shape (no sinks, no chunked
	// window); the pair above stays built as everyone else's path and
	// NoFlashPrefill's arm.
	if !g.NoFlashPrefill && !p.AttnSinks && !p.SWAChunked && bs.rows%kernels.FlashPrefill70Rows == 0 {
		fs := kernels.FlashPrefill70Shape{Heads: p.NHead, KVHeads: p.NHead / gqa, Dim: p.HeadDim, Rows: bs.rows,
			KStride: kst, Scale: scale, Softcap: p.AttnSoftcap}
		fk := comp(kernels.FlashPrefill70(fs))
		var fw backend.Kernel
		if fk != nil && p.SWAWindow > 0 {
			fs.Window = p.SWAWindow
			if fw = comp(kernels.FlashPrefill70(fs)); fw == nil {
				closeKern(fk)
				fk = nil
			}
		}
		bs.fp70, bs.fp70W = fk, fw
	}
	return true
}

// mla70Attn is whether a batched latent block of rows rows takes its scores
// and its accumulate on sm_70's m8n8k4 (mlabatch.go): the device is sm_70, the
// cache is float32, the chunk is a whole number of the kernels' 32-row query
// groups and the latent a whole number of the accumulate's 128-wide warps.
func (g *devTier) mla70Attn(p *nn.LayerPlan, rows int) bool {
	lat := p.KVLoraRank
	return !g.NoVolta && !g.kb.noAttnMMA && !g.KVF16 && rows%(8*volta70AttnNT) == 0 &&
		32*volta70AttnMT <= maxKeyTile && p.KVRow()%4 == 0 && lat >= 32 &&
		lat%(32*volta70LatMT(lat)) == 0 && g.sm70()
}

// voltaBlock reports whether every matvec of block l runs, at this batch width,
// as sm_70's f16 twin (voltaMV), which reads the float activation through
// ActF16 and never the int8 one. A lookup: batchMV's twins are compiled by
// prepBatch before any session.
func (g *devTier) voltaBlock(l *layer, ntok, tok int) bool {
	if l.router != nil || l.linear || l.mla || l.stream != nil {
		return false
	}
	n := 0
	for _, m := range []mv{l.mvq, l.mvk, l.mvv, l.mvo, l.mvg, l.mvu, l.mvd} {
		if m.kern == nil {
			continue
		}
		bm, ok := g.batchMV(m, ntok, tok)
		if !ok || bm.act == nil {
			return false
		}
		n++
	}
	return n > 0
}

// flashTileAttn builds kernels.FlashPrefillTile for a batched block where it
// takes the shape: a device that lowers the collective tiles (tilesOK), the
// transposed float32 K cache and float32 V, no sinks and no chunked window, a
// chunk that is a whole number of the kernel's query blocks. It leaves fpT nil
// otherwise, and the three-kernel path runs.
func (g *devTier) flashTileAttn(bs *blockScratch, kvDim, gqa int, scale float32) {
	p := &bs.p
	kst := g.kStride(p)
	if g.NoFlashPrefill || g.KVF16 || bs.rows <= 1 || kst <= 0 || !g.tileF32() ||
		p.MLA() || p.NonCausal || p.AttnSinks || p.SWAChunked || kvDim%4 != 0 {
		return
	}
	// Only at the 32-row blocking, as measured on Apple Silicon: a workgroup
	// re-stages K and V for every block of query rows, so a narrower block
	// (wide heads) lost to the three kernels. Staging K/V once per shared kv
	// head would fix the wide heads, but the float32 output in workgroup memory
	// leaves no room.
	t, ok := kernels.FlashTileFor(p.HeadDim)
	if !ok || t.SG < 4 || bs.rows%t.Rows() != 0 {
		return
	}
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			return nil
		}
		c, err := g.dev.Compile(k)
		if err != nil {
			return nil
		}
		return c
	}
	fs := kernels.FlashPrefill70Shape{Heads: p.NHead, KVHeads: p.NHead / gqa, Dim: p.HeadDim, Rows: bs.rows,
		KStride: kst, Scale: scale, Softcap: p.AttnSoftcap}
	fk := comp(kernels.FlashPrefillTile(fs, t))
	var fw backend.Kernel
	if fk != nil && p.SWAWindow > 0 {
		fs.Window = p.SWAWindow
		if fw = comp(kernels.FlashPrefillTile(fs, t)); fw == nil {
			closeKern(fk)
			fk = nil
		}
	}
	bs.fpT, bs.fpTW, bs.fpTile = fk, fw, t
}

// spanKernel builds the decode scratch's row copy the folded head reads (a
// chunk's last row into the decode scratch) if it has none. Callers hold g.mu.
func (g *devTier) spanKernel(hb *blockScratch) error {
	if hb.spanK != nil {
		return nil
	}
	kk, e := kernels.CopySpan(hb.p.NEmbd)
	var k backend.Kernel
	if e == nil {
		k, e = g.dev.Compile(kk)
	}
	var ob backend.Buf
	if e == nil {
		w := g.scratchWin() // the scratch's own word (scratch.go)
		if ob, e = g.dev.Alloc(4); e != nil {
			closeKern(k)
		}
		w.close()
	}
	if e != nil {
		return e
	}
	hb.spanK, hb.spanOff = k, ob
	return nil
}
