package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

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
	// (mainkernels.AttnScoresMMA70/AttnAccMMA70), for a device whose m16n8k16 f16
	// scores kernel does not lower. acc70 walks 32-row groups, so atile is 32.
	scores70, scores70W, acc70 backend.Kernel
	// mla70 says a batched latent block's attention runs on sm_70's m8n8k4
	// (mlabatch.go): its accumulate walks 32-row groups too, so the softmax
	// this scratch builds takes atile 32 for it (mla70Attn).
	mla70 bool
	// fp70/fp70W are the same attention as one kernel (mainkernels.FlashPrefill70),
	// the second for a sliding-window layer; nil where it does not take the
	// shape, and the three above run instead.
	fp70, fp70W backend.Kernel
	// fpT/fpTW are that one kernel on a collective matrix unit
	// (mainkernels.FlashPrefillTile, Metal), blocked by fpTile.
	fpT, fpTW backend.Kernel
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
	// aCap and axCap are a's and ax's sizes in bytes: the prologue grows them
	// to the widest operand a program quantizes (sizeOperands).
	aCap, axCap   int
	qr, q, k, v   backend.Buf
	roff          backend.Buf // RoPE-q destination offsets, one per row
	kpos          backend.Buf // RoPE-k destination offsets: positions when K is transposed
	hNorm, logits backend.Buf
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
	// headProg is the decode head's composed launches (PrepHead).
	headProg []headLaunch
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
	// quantAct is ActMul and quantF in one launch (mainkernels.QuantizeAct), nil
	// for an ungated FFN or an activation ActMul has no gated form of.
	quantAct            backend.Kernel
	normPart, normApply backend.Kernel
	// A vision block normalises with LayerNorm, which is three kernels: the mean,
	// the variance about it, then the apply. The passes cannot fuse because
	// E[x^2]-E[x]^2 cancels in f32 on a ViT residual whose mean dwarfs its
	// variance (see mainkernels.LayerNormPart). normPart is the first pass either way
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
	// mainkernels.RMSNormGroup -- rms for the norm closure, addRms for a residual
	// add followed by the FFN norm. nil for a LayerNorm (vision) block.
	rms, addRms backend.Kernel
	// rmsQ and addRmsQ are rms and addRms that also write the quantized
	// activation quantE would (mainkernels.RMSNormQuantRows): one launch where the
	// norm and the quantize were two. nil where they cannot be built.
	rmsQ, addRmsQ backend.Kernel
	// subQA and subQF are BitNet's inner norms fused with the quantize wo and
	// down read (mainkernels.RMSNormQuantRows at qdim and NFFN, the row window),
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
	// routeWidth is bs.route's workgroup: one warp for mainkernels.ExpertRouteWarp,
	// ExpertRouteWidth(NExpert) for mainkernels.ExpertRoute.
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
	// outNormGate is outNorm and its gate (outActMul or outSigMul) in one
	// launch (mainkernels.HeadNormGated, graph.NormGate).
	outNormGate            backend.Kernel
	sliceQ, sliceK, sliceV backend.Kernel
	dMixed, dConv, dZ, dBA backend.Buf
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
	// (mainkernels.GatedDeltaScan): every token of the chunk steps the state,
	// which GatedDeltaStep -- one token -- cannot. scanThreads is its grid.
	deltaScan   backend.Kernel
	scanThreads int
	// deltaFused is deltaScan with everything from the convolution's output to
	// the rule folded in (mainkernels.GatedDeltaFused); built where the device has
	// a subgroup, and then the eleven kernels it replaces are not launched.
	deltaFused backend.Kernel
	// ssdScan is Mamba-2's whole update at this scratch's rows
	// (mainkernels.GatedDeltaFused with SSD), from the convolution's raw output
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
	// (scMul, mainkernels.ActMul with the identity).
	scB, scC, scBx, scY backend.Buf
	scMul               backend.Kernel
	// convCentred is an LFM2 encoder's convolution (mainkernels.Conv3Centred),
	// and vwinAll every row's window in one buffer, the taps' bounds
	// (setWindowsLocked writes it beside the per-pass vwins).
	convCentred backend.Kernel
	vwinAll     backend.Buf
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
	// rwHit, esplit and hsum are the split's (streamBank.split): the card's
	// routing weights with the misses zeroed, the card's sum of the hits and
	// the host's of the misses.
	rwHit, esplit, hsum backend.Buf
	// rselPred is the next block's predicted selection (Config.SplitPredict).
	rselPred backend.Buf
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
