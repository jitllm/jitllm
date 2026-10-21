package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// A whole transformer block on the device.
//
// The point is one Session per block with exactly one Read: serving matvecs one
// at a time costs a blocking round trip each, which alone was a quarter of a
// gemma token. That is also why every op between the matvecs is a kernel -- a
// single host-side RMSNorm would drag the residual back across the bus.

// layer is one block's device-resident state.
type layer struct {
	// ran is the forms the block's last step took (ffnform.go, BlockForms).
	ran blockForms
	// prog is the block's device program (devprog.go), nil where the program
	// does not lower the block yet; progWhy says why.
	prog    *devProgram
	progWhy string
	// graph is the plan's block graph with its composite ops decomposed into
	// primitives (graph.Decompose): what the program, its matrices and its
	// vectors are bound from.
	graph *graph.Graph
	// progRun and progRunRows are why the last step of one row and of many
	// ran the block through the encoder, progRanProgram where it ran the
	// program.
	progRun, progRunRows progReason
	// progOne, progRows and progRag are prog with the rewrites that apply on
	// a scratch at one row, on a prompt chunk and on a ragged step
	// (devrewrite.go); progRagFew the ragged step's of 2 to ragGroupTok
	// live rows, whose matvecs carry every row in one thread.
	progOne, progRows, progRag progFused
	progRagFew                 [ragGroupTok - 1]progFused
	wq, wk, wv, wo             *resident
	// extra and mvExtra are the matrices the block holds beyond the
	// positional list (nn.LayerWeights.All, matched by memory), uploaded and
	// paged with it; roles names every uploaded matrix of the block, the
	// positional ones too, for the device program's binding by role.
	extra   []*resident
	mvExtra []mv
	roles   []roleAt
	bound   []jlm.Role // the roles the block's program binds (boundRoles)
	// rs and ms are the block's matrices and their matvecs in list order
	// (tensors, prep's ms): pointers to the layer's own fields, so a program
	// op indexes them without a switch.
	rs []**resident
	ms []*mv
	// vecs is every vector the program binds by role (prep: a positional
	// upload's buffer where it is the same host floats); vecOwned is those
	// uploaded for it alone, freed with the layer (auxBufs).
	vecs           []roleBuf
	vecOwned       []backend.Buf
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
	// offset of its slice in a row's inputs (oldkernels.PLESlice).
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
	// centred is a short convolution with no state over a whole non-causal
	// call (nn.RecurrentPlan.Centred).
	centred bool
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
	// graphed and rewrites are nn.LayerPlan.Graphed and Rewrites: the block
	// graph's matched rewrites, each asked at this device's Target (offer),
	// which decide a site in place of the structural checks in ffnKey.
	graphed  bool
	rewrites []graph.Offered
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
	// xielu is an Apertus block's four activation numbers (oldkernels.XIELU's
	// pP), nil elsewhere.
	xielu         backend.Buf
	mvq, mvk, mvv mv
	// qkv is q, k and v in one launch (oldkernels.MatVecSegments), each still
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
	// ffnWhy is why the block graph's gate2in rewrite was not offered here
	// at this device's Target (offer), nil where it was or the block has no
	// graph: the plan explanation's reason.
	ffnWhy error
	// normGateOffer is whether graph.NormGate is offered on this block at
	// decode ([0]) and at a prompt chunk's rows ([1]), decided outside a token
	// (prepNormGate); normGateKey the decode verdict's key, normGateWhy the
	// decode refusal.
	normGateOffer [2]bool
	normGateKey   string
	normGateWhy   error
	// mix is the mixture graph's rewrites decided per row class and
	// residency (prepMixture).
	mix mixOffer
	// actInWhy is ffnWhy for the q/k/v, o and down prologue sites.
	actInWhy      [3]error
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

// progFused is a block's program with the rewrites that apply on one
// scratch, and their names.
type progFused struct {
	bs *blockScratch
	// fault is the fault switches the variant was rewritten for (faultSig).
	fault uint32
	prog  *devProgram
	// rows is the rows its kernels were formed for: one for a decode
	// step, the scratch's chunk otherwise. A launch over fewer runs a
	// proportional share of the grid.
	rows int
	took rwSet
	err  error // the type error that refused every rewrite, if one did
	// valErr refuses the variant: its values could not be coloured, or
	// their buffers could not grow (colour.go). The step fails by it.
	valErr error
	// The op-span forms the program runs (BlockForms), noted each step.
	qkv   qkvForm
	gu    ffnGateUpForm
	down  ffnDownForm
	dense bool // a dense FFN, whose gate/up and down forms are noted
}
