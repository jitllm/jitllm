package tier

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// The device program (docs/design/composer.md, "The device program"): a
// block's graph lowered to a list of device ops, each one kernel launch over
// typed values, that rewrites fuse and the tier launches. It is the level the
// block encoder (layersSession's emit) skips; a block whose graph the program
// covers runs through it, every other block through the encoder, and the
// manifest says which (BlockForms "block").

// devElem is a value's element type on the device.
type devElem uint8

const (
	elemF32 devElem = iota
	// elemQ8 is the int8 activation with its per-32 scales (bs.a, bs.ax),
	// what an integer matvec reads.
	elemQ8
	// elemF16 is the binary16 operand a GEMM reads, in its layout.
	elemF16
	// elemU32 is an index: a router's selection of experts.
	elemU32
)

// devLayout is a value's memory layout. A consumer reads one layout; a
// rewrite that hands it another does not build.
type devLayout uint8

const (
	layoutRow    devLayout = iota // row-major, one row a token
	layoutKV                      // the paged KV history
	layoutTokF16                  // token-major binary16 (kernels.ActF16T)
	layoutFrag                    // the MMA's m16n8 fragments (kernels.ActF16)
	layoutPairs                   // one row a (token, selected expert) pair
	layoutHeads                   // over a chunk's rows, head-major: each head's rows together (devrewrite.go headMajor)
)

// devType is a value's type: element, layout and width a row.
type devType struct {
	Elem   devElem
	Layout devLayout
	Width  int
}

func (t devType) String() string {
	e := [...]string{"f32", "q8", "f16", "u32"}[t.Elem]
	l := [...]string{"row", "kv", "tok", "frag", "pairs", "heads"}[t.Layout]
	return fmt.Sprintf("%s.%s[%d]", e, l, t.Width)
}

// devKind is a device op's kind.
type devKind uint8

const (
	dkNorm            devKind = iota // RMSNorm of a row by a weight
	dkQuant                          // a float row to the int8 operand a matvec reads
	dkMatVec                         // an integer matvec over a weight
	dkRopeQ                          // q's rotary
	dkKVWrite                        // k's rotary into the history, and v's copy
	dkAttend                         // the paged decode attention
	dkAdd                            // a residual add
	dkActMul                         // act(gate) * up
	dkHeadNorm                       // each head's RMSNorm (q's or k's)
	dkPostNorm                       // a sublayer's output normed before its residual add (post-norm)
	dkRecurrent                      // a recurrent mixer with its pre-norm, from the block's row (emitLinear)
	dkCopy                           // a row moved to another slot (the sum home to the block's row)
	dkKVCopy                         // an unrotated key and its value copied into the history (NoPE)
	dkAct                            // an ungated FFN's activation of up's row
	dkQTemp                          // the query's temperature, a factor of each row's position
	dkSplitQG                        // a double-width query split into its heads and its output gate
	dkGateOut                        // the attention's output times the sigmoid of the gate
	dkPLESlice                       // this block's rows of the per-layer input table
	dkRopeKRow                       // k's rotary into a row of its own (XD-RoPE's k is normed after it)
	dkStreamRoute                    // a stream router's weights over a row (AltUp's)
	dkStreamPredict                  // each stream predicted from the others by the routed coefficients
	dkStreamCorrect                  // each stream corrected by the active one's change
	dkStreamScale                    // the active stream's correction scaled
	dkStreamFinish                   // every stream written back from the corrections and a row
	dkResidJoin3                     // a residual add with a branch beside it, over sqrt(2)
	dkGaussTopK                      // the gate's values above its mean plus a bound of deviations
	dkStreamNorm                     // every stream normed
	dkStreamGateMix                  // the streams averaged under a gate per channel: a sublayer's row
	dkPlaceWeights                   // a placement's weights from its projection (2*sigmoid)
	dkStreamPlace                    // a sublayer's output added into every stream at its weights
	dkIdxNorm                        // an indexer's query or key, each head normed
	dkIdxRope                        // an indexer's query turned
	dkIdxAssemble                    // the cached row: the attention's key and the indexer's as one more head
	dkIdxScore                       // the indexer's scores over the cached keys or pooled blocks
	dkSelectBlocks                   // the best blocks by score, compacted into a list
	dkGatherKV                       // the selected blocks' keys and values gathered, with their descriptors
	dkAttendGathered                 // the paged attention over the gathered keys
	dkStreamDot                      // per stream, the dot of two rows: a gate
	dkGatedVal                       // a value gated by a signed root of its gate
	dkNGramConv                      // a dilated convolution over the block's recurrent state
	dkNGramAdd                       // a gated value and its convolution added into every stream
	dkStreamScore                    // a bank of streams scored against a row's weights
	dkStreamWeighted                 // the bank's softmax-weighted sum: a sublayer's row
	dkIdxWrite                       // an indexer's turned key into this position's latent row
	dkIdxAttendLatent                // latent attention over the indexer's top positions, gathered
	dkRowBias                        // each row's selection bias: the block's, or a hash table's row for the token
	dkStreamFlatten                  // the streams laid out as one row
	dkFlatNorm                       // the flat row RMS-normed
	dkMixLogits                      // the mix's coefficients: the flat row against the sublayer's weights
	dkSinkhorn                       // the coefficients into read and placement weights, Sinkhorn-normalised
	dkStreamCollect                  // the streams weighed into the sublayer's row
	dkRowRMS                         // rows RMS-normed by a weight (per head where the width is a head's)
	dkTurn                           // a rotary over a row's tail dimensions by a table
	dkCacheWrite                     // a row's key (or a compressing block's pair) into the position's history row
	dkPoolEntries                    // every closed block of positions pooled into an entry
	dkEntryWrite                     // an entry turned into the block's second history
	dkEntrySelect                    // an indexer's top entries over the second history
	dkGatherEntries                  // the window's rows and the visible entries gathered for one attention
	dkGroupedMatVec                  // a grouped low-rank projection: each group's rows times its own matrix
	dkClampAt                        // a row clamped to a pair of the block's bounds table
	dkDWConvRows                     // a depthwise convolution over the call's rows, centred
	dkAffineRows                     // a per-channel scale and shift over the rows
	dkRelAttend                      // relative-position self-attention over the call's rows
	dkScale                          // the block's row times the layer's output scale
	dkClamp                          // q, k or v clamped to the plan's bound
	dkSubNorm                        // an inner norm before wo or down, with the quantize it feeds
	dkRouterLogits                   // a router's logits over the normed row (the router read as floats)
	dkRouterBias                     // a bias added to the router's logits before the choice
	dkGroupMask                      // the experts of the best-scoring groups kept, the rest masked
	dkTopK                           // the top k of the logits (or a masked or biased plane): the selection
	dkRouteWeights                   // the selection's weights: softmax, sigmoid or renormalised, per the plan
	dkExpertMatVec                   // a matvec per selected expert over a bank, indexed by the selection
	dkExpertBias                     // each selected expert's bias, indexed by its id
	dkScaleWeights                   // each routed weight times its expert's own factor
	dkCombine                        // the selection's weighted sum of the experts' outputs
	dkGateLogit                      // a shared expert's gate: one logit over the normed row
	dkGatedAdd                       // a branch added in, scaled by the sigmoid of its gate's logit
	dkExpertFetch                    // a paged bank's selected sheets brought to the card, and the index its matvecs read
	dkGroupByExpert                  // over rows, the selection sorted by expert: the order the grouped matvecs read
	dkSlice                          // a strided copy: each head's slice, or a row's tail
	dkView                           // a row's prefix read in place (no launch)
	dkRopeTail                       // the rotary over each head's tail, compacted
	dkLatentWrite                    // a latent row and its rotated key into this position's paged row
	dkConcatHeads                    // per-head parts assembled into one query row
	dkLatentAttend                   // attention over latent rows: the key the row, the value its prefix
	dkExpertMatVecE                  // an expert matvec with its epilogue: its bias, or act(gate)*up (the bank's fused kernel)
	dkConv1d                         // a causal convolution over the session's convolution state, and its shift
	dkScan                           // a recurrence's state update over the session's state, by the plan's rule
	dkGatedNorm                      // a recurrence's output: each head normed, gated by z
	dkMul                            // an elementwise product of two rows
	dkConvBias                       // the convolved row plus its bias, activated (Mamba-1's)
	dkSSMNorm                        // a selective scan's dt, B or C row normed (Mamba-1's)
	numBaseKinds                     // the fused kinds (devrewrite.go) count on from here
)

// devVal names a value of the program; its buffer is the scratch slot the
// lowering assigns (devSlot).
type devVal int

// devSlot is the block scratch buffer a value lives in.
type devSlot uint8

const (
	slotX        devSlot = iota // bs.x: the block's input and output
	slotX2                      // bs.x2: the residual after attention
	slotH                       // bs.h: a normed row
	slotQ8                      // bs.a / bs.ax: the int8 operand
	slotQ                       // bs.q
	slotK                       // bs.k
	slotV                       // bs.v
	slotQR                      // bs.qr: q rotated
	slotKV                      // the history
	slotXB                      // bs.xb: the attention's output
	slotOut                     // bs.mvOut: a projection's output
	slotG                       // bs.g: the gate
	slotU                       // bs.u: up
	slotAct                     // bs.act: act(gate)*up
	slotQN                      // bs.qn: q's heads normed
	slotKN                      // bs.kn: k's heads normed
	slotF16                     // g.f16Buf: a GEMM's binary16 operand
	slotQT                      // bs.qt: a rotated query scaled by its temperature
	slotQG                      // bs.qg: the query's heads, split from a gated query
	slotOGate                   // bs.ogate: the output gate
	slotOGated                  // bs.ogated: the attention's output, gated
	slotVN                      // bs.vn: v's heads normed
	slotQC                      // bs.qClamp
	slotKC                      // bs.kClamp
	slotVC                      // bs.vClamp
	slotSub                     // bs.sub: an inner norm's row
	slotSide                    // bs.side: a recurrent mixer's output beside the attention
	slotRLogits                 // bs.rlogits: a router's logits
	slotRLogitsB                // bs.rlogitsB: the logits biased
	slotRBM                     // bs.rbm: the group-masked selection plane
	slotSel                     // bs.rsel: the selection
	slotTop                     // bs.rtop: the selection's logits
	slotRW                      // bs.rw: the routed weights
	slotRWS                     // bs.rwS: the routed weights times the experts' factors
	slotEG                      // bs.eg: the experts' gate rows
	slotEU                      // bs.eu: the experts' up rows
	slotEGB                     // bs.egB: the gate rows biased
	slotEUB                     // bs.euB: the up rows biased
	slotEAct                    // bs.eact: the experts' activations
	slotEDown                   // bs.edown: the experts' down rows
	slotEDownB                  // bs.edownB: the down rows biased
	slotShG                     // bs.shg: a shared expert's gate row
	slotShU                     // bs.shu: its up row
	slotShAct                   // bs.shact: its activation
	slotShOut                   // bs.shout: its output
	slotShLogit                 // bs.shlogit: its gate's logit
	slotShSum                   // bs.shsum: the routed and shared sum
	slotSelW                    // the index the expert matvecs read: the selection, its cache slots, or the identity
	slotOnes                    // bs.wOnes: unit weights (a combine of experts weighted at their input)
	slotMQA                     // bs.mlaQA: the query latent
	slotMQAn                    // bs.mlaQAn: it normed
	slotMKV                     // bs.mlaKV: the kv latent and its rotary key
	slotMKVn                    // bs.mlaKVn: the latent normed
	slotMNope                   // bs.mlaNope: the query's unrotated parts, compacted
	slotMPE                     // bs.mlaPE: its rotary parts, compacted
	slotMPErot                  // bs.mlaPErot: them rotated
	slotMKPE                    // bs.mlaKPE: the shared rotary key
	slotMAbsRaw                 // bs.mlaAbsRaw: the absorbed query, head-major
	slotMAbs                    // bs.mlaAbs: the query row the latent attention reads
	slotMAcc                    // bs.mlaAcc: the attention over the latent
	slotMOut                    // bs.mlaOut: it un-absorbed
	slotMIdent                  // bs.mlaIdent: the identity selection 0..NHead-1
	slotMKVLat                  // the kv latent's prefix: bs.mlaKV read in place, copied out over rows
	slotHR                      // bs.hr: a router's own norm of the row
	slotDMixed                  // bs.dMixed: a recurrence's in-projection
	slotDZ                      // bs.dZ: its output gate
	slotDBA                     // bs.dBA: its b and a streams (the delta rule's)
	slotDBRaw                   // bs.dBRaw: its dt stream (Mamba-2's)
	slotDConv                   // bs.dConv: the in-projection convolved
	slotDOut                    // bs.dOut: the recurrence's output
	slotDAct                    // bs.dAct: it normed and gated
	slotScB                     // bs.scB: a short convolution's input gate
	slotScC                     // bs.scC: its output gate
	slotScBx                    // bs.scBx: the gated input
	slotScY                     // bs.scY: the gated output
	slotLRQ8                    // bs.dLRa, bs.dLRax: the low-rank projections' own int8 operand
	slotDLow                    // bs.dLowRank: a low-rank projection's rank-wide middle
	slotDAlpha                  // bs.dAlpha: the per-channel decay's logits (KDA's)
	slotM1B                     // bs.m1B: a selective scan's B
	slotM1C                     // bs.m1C: its C
	slotM1LowN                  // bs.m1LowN: its dt row normed
	slotM1BN                    // bs.m1BN: B normed
	slotM1CN                    // bs.m1CN: C normed
	slotPLEG                    // bs.pleG: the per-layer embedding's gate
	slotPLES                    // bs.pleS: this block's per-layer input
	slotPLEA                    // bs.pleA: the gate activated, times the input
	slotPLEO                    // bs.pleO: projected back to the row
	slotKRot                    // bs.krot: k rotated, before its norm (XD-RoPE)
	slotPred                    // bs.altPred: every stream's prediction
	slotAltM                    // bs.altM: a stream router's weights
	slotAltB                    // bs.altB: the streams corrected
	slotLT                      // bs.altLT: a low-rank branch's middle
	slotLaur                    // bs.altLaur: the branch
	slotGG                      // bs.gg: the gate after its gaussian top-k
	slotSXN                     // the streams normed
	slotSQ8                     // the streams' own int8 operand
	slotSQ8LR                   // the low-rank gate's own int8 operand
	slotSLo                     // the gate's rank-wide middle
	slotSLo2                    // it activated
	slotSGate                   // the gate
	slotSMix                    // the gated mix: a sublayer's row
	slotSInj                    // the placement's projection
	slotSW                      // the placement's weights
	slotIKRaw                   // an indexer's key, projected
	slotIQRaw                   // its query, projected
	slotIQn                     // the query normed
	slotIKn                     // the key normed
	slotIQRot                   // the query turned
	slotKAll                    // the cached row with the indexer's key
	slotISel                    // the selection: the gathered keys' list and descriptors
	slotNEmb                    // the call's n-gram table rows, staged
	slotSQ8W                    // their own int8 operand
	slotNKey                    // the n-gram key
	slotNVal                    // the n-gram value
	slotNKn                     // the key normed per stream
	slotNQn                     // the streams normed
	slotNDot                    // each stream's gate
	slotNGv                     // the gated value
	slotNGvn                    // it normed
	slotNConv                   // its convolution
	slotNXg                     // the streams with the n-gram embedding added
	slotBScores                 // a bank's scores
	slotBMix                    // the bank's weighted sum
	slotMGate                   // an output gate's projection beside latent attention
	slotMGated                  // the latent attention gated
	slotLat                     // a mixture's latent row
	slotLatAcc                  // the experts' sum in the latent
	slotLatN                    // it normed
	slotIdxW                    // an indexer's per-head weights
	slotIKRot                   // its key turned
	slotRBias                   // a row's selection bias
	slotHFlat                   // the streams flat
	slotHFlatN                  // normed
	slotHMix                    // the mix's coefficients
	slotHC                      // read and placement weights
	slotHColl                   // the streams weighed into a row
	slotCQA                     // the query's latent
	slotCQAN                    // normed
	slotCQ                      // the query
	slotCQN                     // normed per head
	slotCQR                     // turned
	slotCKRaw                   // the key
	slotCKN                     // normed
	slotCKR                     // turned
	slotCKV                     // a compressing block's pooled value
	slotCG                      // its pooling gate
	slotCIKV                    // the indexer's
	slotCIG                     // its gate
	slotCIW                     // the indexer's per-head weights
	slotCIQ                     // the indexer's query
	slotCIQR                    // turned
	slotCPE                     // the pooled entry
	slotCPEN                    // normed
	slotCPER                    // turned
	slotCIPE                    // the indexer's pooled entry
	slotCIPEN                   // normed
	slotCIPER                   // turned
	slotCSel                    // the selected entries
	slotCGK                     // the gathered rows
	slotCAO                     // the attention's heads
	slotCAOR                    // turned back
	slotCOA                     // the grouped projection's rows
	slotCIn                     // a linear's clamped input
	slotCOut                    // a linear's clamped output
	slotGC                      // the gate's, clamped
	slotUC                      // up's, clamped
	// slotPool0.. are a pool coloured by liveness: a convolution-path
	// program's values take the first free one (convgraph.go).
	slotPool0
	slotPool1
	slotPool2
	slotPool3
	slotPool4
	slotPool5
	slotPool6
	slotPool7
	slotPool8
	slotPool9
	slotPool10
	slotPool11
	// slotVal0.. hold the values the builder gives no slot of their own
	// (slotAuto): coloured by liveness once the program's rewrites are done
	// (colourVals), their buffers the lane's, grown in the prologue.
	slotVal0
	slotVal1
	slotVal2
	slotVal3
	slotVal4
	slotVal5
	slotVal6
	slotVal7
	slotVal8
	slotVal9
	slotVal10
	slotVal11
	slotVal12
	slotVal13
	slotVal14
	slotVal15
	slotConvIn  // a convolution-path block's input
	slotConvOut // its output
	slotConvPos // its positions' table (a second input)
	numSlots
)

func (s devSlot) String() string {
	if s >= slotPool0 && s < slotPool0+numPool {
		return fmt.Sprintf("pool%d", s-slotPool0)
	}
	if s >= slotVal0 && s < slotVal0+numVals {
		return fmt.Sprintf("val%d", s-slotVal0)
	}
	if s == slotAuto {
		return "auto"
	}
	switch s {
	case slotX:
		return "x"
	case slotX2:
		return "x2"
	case slotH:
		return "h"
	case slotQ8:
		return "q8"
	case slotQ:
		return "q"
	case slotK:
		return "k"
	case slotV:
		return "v"
	case slotQR:
		return "qr"
	case slotKV:
		return "kv"
	case slotXB:
		return "xb"
	case slotOut:
		return "out"
	case slotG:
		return "g"
	case slotU:
		return "u"
	case slotAct:
		return "act"
	case slotQN:
		return "qn"
	case slotKN:
		return "kn"
	case slotF16:
		return "f16"
	case slotQT:
		return "qt"
	case slotQG:
		return "qg"
	case slotOGate:
		return "ogate"
	case slotOGated:
		return "ogated"
	case slotVN:
		return "vn"
	case slotQC:
		return "qc"
	case slotKC:
		return "kc"
	case slotVC:
		return "vc"
	case slotSub:
		return "sub"
	case slotSide:
		return "side"
	case slotRLogits:
		return "rlogits"
	case slotRLogitsB:
		return "rlogits_b"
	case slotRBM:
		return "rbm"
	case slotSel:
		return "sel"
	case slotTop:
		return "top"
	case slotRW:
		return "rw"
	case slotRWS:
		return "rw_s"
	case slotEG:
		return "eg"
	case slotEU:
		return "eu"
	case slotEGB:
		return "eg_b"
	case slotEUB:
		return "eu_b"
	case slotEAct:
		return "eact"
	case slotEDown:
		return "edown"
	case slotEDownB:
		return "edown_b"
	case slotShG:
		return "shg"
	case slotShU:
		return "shu"
	case slotShAct:
		return "shact"
	case slotShOut:
		return "shout"
	case slotShLogit:
		return "shlogit"
	case slotShSum:
		return "shsum"
	case slotSelW:
		return "sel_w"
	case slotOnes:
		return "ones"
	case slotMQA:
		return "mla_qa"
	case slotMQAn:
		return "mla_qa_n"
	case slotMKV:
		return "mla_kv"
	case slotMKVn:
		return "mla_kv_n"
	case slotMNope:
		return "mla_nope"
	case slotMPE:
		return "mla_pe"
	case slotMPErot:
		return "mla_pe_rot"
	case slotMKPE:
		return "mla_kpe"
	case slotMAbsRaw:
		return "mla_abs_raw"
	case slotMAbs:
		return "mla_abs"
	case slotMAcc:
		return "mla_acc"
	case slotMOut:
		return "mla_out"
	case slotMIdent:
		return "mla_ident"
	case slotMKVLat:
		return "mla_kv_lat"
	case slotHR:
		return "hr"
	case slotDMixed:
		return "d_mixed"
	case slotDZ:
		return "d_z"
	case slotDBA:
		return "d_ba"
	case slotDBRaw:
		return "d_braw"
	case slotDConv:
		return "d_conv"
	case slotDOut:
		return "d_out"
	case slotDAct:
		return "d_act"
	case slotScB:
		return "sc_b"
	case slotScC:
		return "sc_c"
	case slotScBx:
		return "sc_bx"
	case slotScY:
		return "sc_y"
	case slotLRQ8:
		return "lr_q8"
	case slotDLow:
		return "d_low"
	case slotDAlpha:
		return "d_alpha"
	case slotM1B:
		return "m1_b"
	case slotM1C:
		return "m1_c"
	case slotM1LowN:
		return "m1_low_n"
	case slotM1BN:
		return "m1_b_n"
	case slotM1CN:
		return "m1_c_n"
	case slotPLEG:
		return "ple_g"
	case slotPLES:
		return "ple_s"
	case slotPLEA:
		return "ple_a"
	case slotPLEO:
		return "ple_o"
	case slotKRot:
		return "k_rot"
	case slotPred:
		return "pred"
	case slotAltM:
		return "alt_m"
	case slotAltB:
		return "alt_b"
	case slotLT:
		return "lt"
	case slotLaur:
		return "branch"
	case slotGG:
		return "gg"
	case slotSXN:
		return "s_xn"
	case slotSQ8:
		return "s_q8"
	case slotSQ8LR:
		return "s_q8_lr"
	case slotSLo:
		return "s_lo"
	case slotSLo2:
		return "s_lo2"
	case slotSGate:
		return "s_gate"
	case slotSMix:
		return "s_mix"
	case slotSInj:
		return "s_inj"
	case slotSW:
		return "s_w"
	case slotIKRaw:
		return "idx_k_raw"
	case slotIQRaw:
		return "idx_q_raw"
	case slotIQn:
		return "idx_q_n"
	case slotIKn:
		return "idx_k_n"
	case slotIQRot:
		return "idx_q_rot"
	case slotKAll:
		return "k_all"
	case slotISel:
		return "idx_sel"
	case slotNEmb:
		return "ng_emb"
	case slotSQ8W:
		return "ng_q8"
	case slotNKey:
		return "ng_key"
	case slotNVal:
		return "ng_val"
	case slotNKn:
		return "ng_kn"
	case slotNQn:
		return "ng_qn"
	case slotNDot:
		return "ng_dot"
	case slotNGv:
		return "ng_gv"
	case slotNGvn:
		return "ng_gvn"
	case slotNConv:
		return "ng_conv"
	case slotNXg:
		return "ng_xg"
	case slotBScores:
		return "b_scores"
	case slotBMix:
		return "b_mix"
	case slotMGate:
		return "m_gate"
	case slotMGated:
		return "m_gated"
	case slotLat:
		return "lat"
	case slotLatAcc:
		return "lat_acc"
	case slotLatN:
		return "lat_n"
	case slotIdxW:
		return "idx_w"
	case slotIKRot:
		return "idx_k_rot"
	case slotRBias:
		return "r_bias"
	case slotHFlat:
		return "hc_flat"
	case slotHFlatN:
		return "hc_flat_n"
	case slotHMix:
		return "hc_mix"
	case slotHC:
		return "hc"
	case slotHColl:
		return "hc_row"
	case slotCQA:
		return "c_qa"
	case slotCQAN:
		return "c_qa_n"
	case slotCQ:
		return "c_q"
	case slotCQN:
		return "c_q_n"
	case slotCQR:
		return "c_q_rot"
	case slotCKRaw:
		return "c_k"
	case slotCKN:
		return "c_k_n"
	case slotCKR:
		return "c_k_rot"
	case slotCKV:
		return "c_ckv"
	case slotCG:
		return "c_cg"
	case slotCIKV:
		return "c_ikv"
	case slotCIG:
		return "c_ig"
	case slotCIW:
		return "c_iw"
	case slotCIQ:
		return "c_iq"
	case slotCIQR:
		return "c_iq_rot"
	case slotCPE:
		return "c_pe"
	case slotCPEN:
		return "c_pe_n"
	case slotCPER:
		return "c_pe_rot"
	case slotCIPE:
		return "c_ipe"
	case slotCIPEN:
		return "c_ipe_n"
	case slotCIPER:
		return "c_ipe_rot"
	case slotCSel:
		return "c_sel"
	case slotCGK:
		return "c_gk"
	case slotCAO:
		return "c_ao"
	case slotCAOR:
		return "c_ao_rot"
	case slotCOA:
		return "c_oa"
	case slotCIn:
		return "clamp_in"
	case slotCOut:
		return "clamp_out"
	case slotGC:
		return "g_clamped"
	case slotUC:
		return "u_clamped"
	case slotConvIn:
		return "conv_in"
	case slotConvOut:
		return "conv_out"
	case slotConvPos:
		return "conv_pos"
	}
	return ""
}

// devOp is one launch of the program.
type devOp struct {
	Kind devKind
	// Src is the graph ops it lowers (indices into the graph's Ops): one,
	// or several where a lowering takes them together.
	Src  []int
	In   []devVal
	Out  []devVal
	Role jlm.Role // the weight a norm or matvec reads
	// W is a matvec's weight as the block holds it: its index in the
	// block's matrix list (layer.rs, layer.ms), resolved from Role once
	// when the program is built.
	W int
	// V and VB are a norm's weight and bias on the device, bound by role
	// when the program is built (layer.vecOf); a fused op carries the norm
	// it absorbed. nil for a weightless norm.
	V, VB backend.Buf
	// OneRow says the op runs for one row only: over rows, what reads it
	// produces it itself (an expert matvec's quantize).
	OneRow bool
	// Need overrides what the op reads on an input: a GEMM's binary16
	// operand is the layout its batched twin reads.
	Need []devType
	// Why is why its generator handed it to a former (genWhy).
	Why genWhy
	// FIn, where set, is 1 + the input holding the float row a matvec's
	// int8 operand was quantized from (floatSource): what a binary16
	// conversion and a float weight read.
	FIn int8
	// Heads, where set, is a selected matvec over this many heads that
	// reads its operands head-major over a chunk's rows (In[0] int8, In[2]
	// float): headMajor lowers the order into the program.
	Heads int
	// K and Conv are a binary16 writer's kernel and the conversion it
	// stands in for (rmsF16For, actF16For), chosen when it was rewritten.
	K, Conv backend.Kernel
	// Staged says the op is several launches that pass through scratch, so
	// reading and writing one slot is safe: no single launch does both.
	Staged bool
	N      int     // K's thread count, where it is not the rows'
	QS     *qkvSeg // the q/k/v grid's segments (qkv-gemm)
	// Alpha is an add's scale on its second operand (the graph's), and Act
	// an activation's kind: what the op's generated kernel is built from.
	Alpha float64
	Act   graph.Act
	// HD is a per-head op's head width (the graph's), and A its first
	// source graph op's attributes.
	HD int
	A  graph.Attrs
	// Gen is the op's generated launches (genop.go), and GC a constant it
	// owns (an add's scale), uploaded at admission.
	Gen []genLaunch
	GC  backend.Buf
}

// devProgram is a block's device program.
type devProgram struct {
	// f16 says the program's matvecs are binary16 GEMMs reading the float
	// rows (a prompt chunk on a card whose twins are all binary16): no
	// int8 operand, each GEMM converting its row or finding it written.
	f16 bool
	// attends says the program runs an attention: it reads the paged history;
	// ropes that it rotates q and k.
	attends, ropes bool
	// experts says the program runs routed experts, which it lowers for one
	// row: rows take the encoder until the grouped kernels are lowered.
	experts bool
	// streamed says an expert bank is paged in per token (dkExpertFetch).
	streamed bool
	// scans says the program steps a recurrence: its state's parity flips
	// once the block's launches are queued (devTier.stepFlip).
	scans bool
	// latent says the program attends over latent rows, which it lowers for
	// one row: rows take the encoder until latent prefill is lowered.
	latent bool
	vals   []devType
	slots  []devSlot
	names  []string
	ops    []devOp
}

// numVals is how many values a program colours by liveness at once.
const numVals = 16

// slotAuto marks a value the builder gives no slot: colourVals assigns it
// one of slotVal0.. by liveness once the rewrites are done.
const slotAuto = numSlots

func (p *devProgram) val(name string, t devType, s devSlot) devVal {
	p.vals = append(p.vals, t)
	p.slots = append(p.slots, s)
	p.names = append(p.names, name)
	return devVal(len(p.vals) - 1)
}

// String is the program, one op a line, each value with its type and slot.
func (p *devProgram) String() string {
	var b strings.Builder
	v := func(x devVal) string { return fmt.Sprintf("%%%s:%s@%s", p.names[x], p.vals[x], p.slots[x]) }
	for _, o := range p.ops {
		var outs, ins []string
		for _, x := range o.Out {
			outs = append(outs, v(x))
		}
		for _, x := range o.In {
			ins = append(ins, v(x))
		}
		fmt.Fprintf(&b, "  %s = %s(%s", strings.Join(outs, ", "), o.Kind, strings.Join(ins, ", "))
		if o.Role != 0 {
			fmt.Fprintf(&b, ", %s", o.Role)
		}
		gen := ""
		switch {
		case len(o.Gen) != 0 && o.Gen[0].former != 0 && o.Why != 0:
			gen = ", former (old kernels): " + o.Why.String()
		case len(o.Gen) != 0 && o.Gen[0].former != 0:
			gen = ", former (old kernels)"
		case len(o.Gen) != 0:
			gen = fmt.Sprintf(", generated: %d launch(es)", len(o.Gen))
		default:
			// What a generator saw: the shape it declined.
			lay := ""
			for _, x := range o.In {
				lay += p.vals[x].String() + " "
			}
			gen = fmt.Sprintf(", not generated [%sact %d v %v]", lay, o.Act, o.V != nil)
		}
		fmt.Fprintf(&b, ")  ; graph ops %v%s\n", o.Src, gen)
	}
	return b.String()
}

// wants is the type each kind reads on each input, and what it writes; the
// check holds every edge to it.
func (o devOp) wants(p *devProgram, i int) devType {
	in := p.vals[o.In[i]]
	// A selected expert's rows are rows to every elementwise and row kind:
	// what the pair layout changes is how many there are.
	row := layoutRow
	if in.Layout == layoutPairs {
		row = layoutPairs
	}
	if i < len(o.Need) && o.Need[i] != (devType{}) {
		return o.Need[i]
	}
	if in.Layout == layoutHeads && (o.Kind == dkQuant || o.Kind == dkExpertMatVec && o.Heads > 0 && i == 2) {
		return in // the head-major order the gather wrote, read in it
	}
	if o.Kind == dkGatherHeads {
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	}
	if p.f16 && (o.Kind == dkMatVec || o.Kind == dkQKV || o.Kind == dkQKVGemm) {
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	}
	switch o.Kind {
	case dkPoolEntries:
		// Pooling reads the position's history.
		return devType{Elem: elemF32, Layout: layoutKV, Width: in.Width}
	case dkRecurrent, dkCopy, dkKVCopy, dkAct, dkQTemp, dkSplitQG, dkGateOut, dkPLESlice, dkRopeKRow, dkScale, dkClamp, dkSubNorm:
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	case dkExpertMatVec:
		// The activation, int8; the selection, indices; the float rows the
		// grouped form sorts and quantizes itself.
		switch i {
		case 1:
			return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
		case 2:
			return devType{Elem: elemF32, Layout: row, Width: in.Width}
		}
		return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
	case dkExpertMatVecE:
		switch i {
		case 0:
			return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
		case 1:
			return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	case dkExpertBias, dkScaleWeights:
		if i == 1 {
			return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	case dkSlice, dkView, dkRopeTail, dkLatentWrite, dkConcatHeads, dkLatentAttend, dkConv1d, dkScan, dkGatedNorm, dkMul, dkConvBias, dkSSMNorm,
		dkStreamRoute, dkStreamPredict, dkStreamCorrect, dkStreamScale, dkStreamFinish, dkResidJoin3, dkGaussTopK,
		dkStreamNorm, dkStreamGateMix, dkPlaceWeights, dkStreamPlace,
		dkIdxNorm, dkIdxRope, dkIdxAssemble, dkIdxScore, dkSelectBlocks, dkGatherKV, dkAttendGathered,
		dkStreamDot, dkGatedVal, dkNGramConv, dkNGramAdd, dkStreamScore, dkStreamWeighted, dkIdxWrite, dkIdxAttendLatent,
		dkRowBias, dkStreamFlatten, dkFlatNorm, dkMixLogits, dkSinkhorn, dkStreamCollect, dkRowRMS, dkTurn,
		dkCacheWrite, dkEntryWrite, dkEntrySelect, dkGatherEntries, dkClampAt,
		dkDWConvRows, dkAffineRows, dkRelAttend:
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	case dkExpertFetch, dkGroupByExpert:
		return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
	case dkRouterLogits, dkRouterBias, dkGroupMask, dkTopK, dkRouteWeights, dkCombine, dkGateLogit, dkGatedAdd:
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	case dkMatVec, dkQKV, dkGate2:
		if i > 0 {
			return devType{Elem: elemF32, Layout: row, Width: in.Width} // a low-rank operand's float source
		}
		return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
	case dkMatVecRes, dkMatVecGate:
		if i == 0 {
			return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	case dkQuant, dkNorm, dkRopeQ, dkAdd, dkActMul, dkKVWrite, dkAttend, dkHeadNorm, dkPostNorm,
		dkNormQuant, dkAddNorm, dkAddNormQuant, dkActQuant, dkRopeQKV, dkGate2In,
		dkQKVIn, dkQKVSep, dkResIn, dkFFNIn, dkDownAct, dkNormF16, dkAddNormF16, dkActF16:
		return devType{Elem: elemF32, Layout: row, Width: in.Width}
	}
	return in
}

// check holds every op's inputs to the types the op reads: a value in a
// layout or element type its consumer does not read refuses the program.
func (p *devProgram) check() error {
	// Every value an op reads is written by an earlier op, is the block's
	// input or is a constant (constSlot): a rewrite that moves a launch above
	// what it reads is refused.
	written := make([]int, len(p.vals))
	for i := range written {
		written[i] = -1
	}
	for i, o := range p.ops {
		for _, y := range o.Out {
			if written[y] < 0 {
				written[y] = i
			}
		}
	}
	for i, o := range p.ops {
		for _, x := range o.In {
			if constSlot(p.slots[x]) {
				continue
			}
			if w := written[x]; w >= i {
				return fmt.Errorf("device program: op %d (%s) reads %%%s, which op %d writes after it", i, o.Kind, p.names[x], w)
			}
		}
	}
	for i, o := range p.ops {
		// RULE 13: no launch reads a buffer it writes (a staged op's
		// launches go through scratch).
		for _, x := range o.In {
			if o.Staged {
				break
			}
			for _, y := range o.Out {
				if p.slots[x] == p.slots[y] && p.slots[x] != slotKV && p.slots[x] != slotAuto {
					return fmt.Errorf("device program: op %d (%s) reads and writes %s", i, o.Kind, p.slots[x])
				}
			}
		}
		for j := range o.In {
			got, want := p.vals[o.In[j]], o.wants(p, j)
			if got != want {
				return fmt.Errorf("device program: op %d (%s) reads %%%s as %s, which is %s",
					i, o.Kind, p.names[o.In[j]], want, got)
			}
		}
	}
	return nil
}

// lowersOp reports the program lowers graph ops of kind k.
func lowersOp(k graph.OpKind) bool {
	switch k {
	case graph.OpNorm, graph.OpMatVec, graph.OpRope, graph.OpKVAppend, graph.OpAttend, graph.OpAdd,
		graph.OpActMul, graph.OpAct, graph.OpHeadNorm, graph.OpLayerNorm, graph.OpScale, graph.OpClamp,
		// A block-selecting indexer (its selection read by the attention).
		graph.OpIndex,
		// An n-gram embedding added into the streams.
		graph.OpNGram,
		// A stream mix and its placement (the low-rank gated form).
		graph.OpStreamMix, graph.OpStreamPlace,
		// Predicted and corrected streams, a low-rank branch, the sparse gate.
		graph.OpAltPredict, graph.OpAltCorrect, graph.OpLaurel, graph.OpGaussTopK,
		// An attention keeping pooled entries beside its rows.
		graph.OpCompressedAttend,
		// The mixture's ops.
		graph.OpRoute, graph.OpGather, graph.OpExpertMatVec, graph.OpCombine, graph.OpGateScalar,
		// A recurrent mixer, lowered whole with its pre-norm (emitLinear).
		graph.OpRecurrent,
		// Latent attention's own ops, lowered whole (emitMLA).
		graph.OpSlice, graph.OpLatentAppend, graph.OpAttendLatent:
		return true
	}
	return false
}

// buildDevProgram lowers p's graph to a device program, or says why it does
// not: an op or an attribute the program does not lower yet. The reason is
// what BlockProgram reports for a block left on the encoder.
func buildDevProgram(p *nn.LayerPlan, l *layer) (*devProgram, string) {
	g := p.Graph
	switch {
	case g == nil:
		return nil, "no graph"
	case !actHasKernel(p):
		return nil, "an activation with no kernel"
	}
	for _, o := range g.Ops {
		if !lowersOp(o.Kind) {
			return nil, "op " + o.Kind.String()
		}
		// A window or an aligned chunk is the plan's per-row key start (the
		// descriptors); a bidirectional sequence is a non-causal segment's,
		// over its contiguous history.
		if o.Kind == graph.OpAttend && o.Attr.Bidir && !p.NonCausal {
			return nil, "a bidirectional attention in a causal block"
		}
	}
	prog := &devProgram{}
	ne, qd, kv := p.NEmbd, p.NHead*p.HeadDim, p.KVRow()
	f32 := func(w int) devType { return devType{Elem: elemF32, Layout: layoutRow, Width: w} }
	// like is a float value of width w laid out as v is: an elementwise op
	// keeps its input's rows (a token's, or a selected expert's).
	like := func(v devVal, w int) devType { return devType{Elem: elemF32, Layout: prog.vals[v].Layout, Width: w} }
	// The block's row: every stream of it where it carries several.
	vals := map[graph.ValueID]devVal{g.In[0]: prog.val("x", f32(g.Values[g.In[0]].Width), slotX)}
	if len(g.In) > 1 && p.PLEDim != 0 {
		// The per-layer input: this block's rows of the model's table.
		pl := prog.val(g.Values[g.In[1]].Name, f32(p.PLEDim), slotPLES)
		prog.ops = append(prog.ops, devOp{Kind: dkPLESlice, Out: []devVal{pl}})
		vals[g.In[1]] = pl
	}
	// The int8 operand of a matvec: the float row it reads, quantized once
	// per source (q, k and v read one quantize of the normed row).
	//
	// One int8 operand buffer: the last quantize wins, so the cache is the
	// last source, its quantize and that op. A quantize only expert matvecs
	// read is one row's (OneRow): over rows they quantize their own sorted
	// operand.
	lastSrc, lastQ, lastOp := devVal(-1), devVal(-1), -1
	operand := func(src devVal, op int, expert bool) devVal {
		if src == lastSrc {
			if !expert && lastOp >= 0 {
				prog.ops[lastOp].OneRow = false
			}
			return lastQ
		}
		q := prog.val(prog.names[src]+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[src].Width}, slotQ8)
		prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{op}, In: []devVal{src}, Out: []devVal{q}, OneRow: expert})
		lastSrc, lastQ, lastOp = src, q, len(prog.ops)-1
		return q
	}
	matvecSlot := map[jlm.Role]devSlot{
		jlm.RoleAttnQ: slotQ, jlm.RoleAttnK: slotK, jlm.RoleAttnV: slotV, jlm.RoleAttnOut: slotOut,
		jlm.RoleFFNGate: slotG, jlm.RoleFFNUp: slotU, jlm.RoleFFNDown: slotOut,
		jlm.RoleShExpGate: slotShG, jlm.RoleShExpUp: slotShU, jlm.RoleShExpDown: slotShOut,
		jlm.RoleAttnQA: slotMQA, jlm.RoleAttnQB: slotQ, jlm.RoleAttnKVA: slotMKV,
		jlm.RolePLEGate: slotPLEG, jlm.RolePLEProj: slotPLEO,
		jlm.RoleAttnGate: slotMGate, jlm.RoleFFNRoutedDown: slotLat, jlm.RoleFFNRoutedUp: slotOut,
	}
	adds := 0
	// routeW is each selection's routed weights; gated is a shared expert's
	// output (by graph value) and its gate's logit.
	routeW := map[devVal]devVal{}
	// weightSel is a paged bank's weight-side index for a selection, and
	// weightIn marks a selection whose weights scale the experts' input.
	weightSel := map[devVal]devVal{}
	weightIn := map[devVal]bool{}
	gated := map[graph.ValueID]devVal{}
	// idxQuery is a block-selecting indexer's turned query, lowered with
	// the history write its key joins.
	idxQuery := devVal(-1)
	lightning := false
	var kRot graph.ValueID = -1
	mla := false
	for _, o := range g.Ops {
		if o.Kind == graph.OpAttendLatent {
			mla = true
		}
	}
	skipTo := 0
	for i, o := range g.Ops {
		if i < skipTo {
			continue // inside a region lowered whole (the mixture, latent attention)
		}
		in := func(j int) devVal { return vals[o.In[j]] }
		name := func(j int) string { return g.Values[o.Out[j]].Name }
		switch o.Kind {
		case graph.OpNorm, graph.OpLayerNorm:
			// A LayerNorm binds its bias second; the block's norm (norm) is
			// the form its plan was built with, and reads the bias.
			if len(o.Weight) == 0 || o.Kind == graph.OpNorm && len(o.Weight) != 1 {
				return nil, "a norm without its weight"
			}
			k := dkNorm
			switch o.Weight[0].Role {
			case jlm.RoleAttnSubNorm, jlm.RoleFFNSubNorm:
				// BitNet's inner norm, fused with the quantize its
				// projection reads: that projection takes the int8 row
				// written here.
				sub := prog.val(name(0), prog.vals[in(0)], slotSub)
				q := prog.val(name(0)+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[in(0)].Width}, slotQ8)
				prog.ops = append(prog.ops, devOp{Kind: dkSubNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{sub, q}, Role: o.Weight[0].Role})
				lastSrc, lastQ, lastOp = sub, q, -1
				vals[o.Out[0]] = sub
				continue
			case jlm.RoleAttnNorm, jlm.RoleFFNNorm:
			case jlm.RoleAttnOutNorm, jlm.RoleLayerOutNorm:
				// A norm after the residual add (post-norm): its output is the
				// residual. The norm into h, then h into the residual slot
				// the add did not write.
				h := prog.val(name(0)+".h", f32(ne), slotH)
				prog.ops = append(prog.ops, devOp{Kind: dkNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{h}, Role: o.Weight[0].Role})
				rs := slotX2
				if prog.slots[in(0)] == slotX2 {
					rs = slotX
				}
				out := prog.val(name(0), f32(ne), rs)
				prog.ops = append(prog.ops, devOp{Kind: dkCopy, Src: []int{i}, In: []devVal{h}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			case jlm.RolePostAttnNorm, jlm.RolePostFFNNorm:
				k = dkPostNorm
			case jlm.RoleFFNRoutedNorm:
				out := prog.val(name(0), prog.vals[in(0)], slotLatN)
				prog.ops = append(prog.ops, devOp{Kind: dkPostNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
				vals[o.Out[0]] = out
				continue
			case jlm.RolePostFFNNorm1, jlm.RolePostFFNNorm2, jlm.RoleRouterNorm, jlm.RoleFFNNorm2, jlm.RolePLEPostNorm:
				// A mixture beside a dense MLP: each branch's norms, out of
				// place, into the buffers the branches read (the router's
				// own row, the experts', and each branch's output).
				ns := slotH
				switch o.Weight[0].Role {
				case jlm.RolePostFFNNorm1:
					ns = slotShSum
				case jlm.RoleRouterNorm:
					ns = slotHR
				}
				out := prog.val(name(0), prog.vals[in(0)], ns)
				prog.ops = append(prog.ops, devOp{Kind: dkPostNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
				vals[o.Out[0]] = out
				continue
			case jlm.RoleAttnQANorm, jlm.RoleAttnKVANorm:
				// A latent's norm, at the latent's width, into its own buffer.
				ls := slotMQAn
				if o.Weight[0].Role == jlm.RoleAttnKVANorm {
					ls = slotMKVn
				}
				out := prog.val(name(0), prog.vals[in(0)], ls)
				prog.ops = append(prog.ops, devOp{Kind: dkNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
				vals[o.Out[0]] = out
				continue
			case jlm.RoleAttnQNorm, jlm.RoleAttnKNorm:
				// A whole-projection q/k norm (OLMo 2's, OLMoE's): the head
				// norm's lowering at the plan's wide geometry.
				if !p.QKNormWide {
					return nil, "a row norm of q or k on a plan that is not wide"
				}
				s := slotQN
				if prog.slots[in(0)] == slotK {
					s = slotKN
				}
				out := prog.val(name(0), prog.vals[in(0)], s)
				// A whole projection normed at once: one head of its width.
				prog.ops = append(prog.ops, devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role, HD: prog.vals[in(0)].Width})
				vals[o.Out[0]] = out
				continue
			default:
				return nil, "a norm over " + o.Weight[0].Role.String()
			}
			out := prog.val(name(0), f32(ne), slotH)
			prog.ops = append(prog.ops, devOp{Kind: k, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
			vals[o.Out[0]] = out
		case graph.OpMatVec:
			role := o.Weight[0].Role
			s, ok := matvecSlot[role]
			if !ok {
				return nil, "a matvec over " + role.String()
			}
			w := o.Weight[0].Rows
			if w == 0 {
				return nil, "a matvec with no stated width"
			}
			wi := l.matIndex(role)
			if wi < 0 {
				return nil, "a matvec over " + role.String() + ", which the block's device copy does not hold"
			}
			a := operand(in(0), i, false)
			out := prog.val(name(0), f32(w), s)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{a}, Out: []devVal{out}, Role: role, W: wi})
			vals[o.Out[0]] = out
		case graph.OpRecurrent:
			if rr := p.Recurrent; prog.slots[in(0)] == slotH && rr.ShortConv && !l.centred && l.k3 == nil && l.q4e == nil {
				// A short convolution (LFM2's): x, B and C projected, B*x
				// convolved over the session's state, gated by C, projected
				// out.
				for _, w := range [...]int{posSSMInProj, posSSMGate, posSSMBA, posSSMOut} {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a short convolution whose projections the block's device copy does not hold"
					}
				}
				ch := f32(rr.Chans)
				q := operand(in(0), i, false)
				mv := func(nm string, w int, s devSlot) devVal {
					v := prog.val(name(0)+nm, ch, s)
					prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{v}, W: w})
					return v
				}
				x, b, c := mv(".x", posSSMInProj, slotDMixed), mv(".b", posSSMGate, slotScB), mv(".c", posSSMBA, slotScC)
				bx := prog.val(name(0)+".bx", ch, slotScBx)
				prog.ops = append(prog.ops, devOp{Kind: dkMul, Src: []int{i}, In: []devVal{b, x}, Out: []devVal{bx}})
				conv := prog.val(name(0)+".conv", ch, slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{bx}, Out: []devVal{conv}})
				y := prog.val(name(0)+".y", ch, slotScY)
				prog.ops = append(prog.ops, devOp{Kind: dkMul, Src: []int{i}, In: []devVal{c, conv}, Out: []devVal{y}})
				qy := operand(y, i, false)
				out := prog.val(name(0), f32(ne), slotOut)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qy}, Out: []devVal{out}, W: posSSMOut})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
			if rr := p.Recurrent; prog.slots[in(0)] == slotH && rr.Mamba1 && !l.centred && l.k3 == nil && l.q4e == nil {
				// Mamba-1's selective scan: x and z projected, x convolved
				// over the session's state and biased, dt, B and C off x (dt
				// through its own low-rank operand), each optionally normed,
				// the scan, y*silu(z), projected out.
				for _, w := range [...]int{posSSMInProj, posSSMGate, posSSMOut, posLowA, posLowB, posGateA, posGateB} {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a selective scan whose projections the block's device copy does not hold"
					}
				}
				inner := rr.VHeads * rr.VDim
				mv := func(nm string, q devVal, w int, s devSlot, width int, extra ...devVal) devVal {
					v := prog.val(name(0)+nm, f32(width), s)
					prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: append([]devVal{q}, extra...), Out: []devVal{v}, W: w})
					return v
				}
				q := operand(in(0), i, false)
				x, z := mv(".x", q, posSSMInProj, slotDMixed, rr.Chans), mv(".z", q, posSSMGate, slotDZ, inner)
				conv := prog.val(name(0)+".conv", f32(rr.Chans), slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{x}, Out: []devVal{conv}})
				xa := prog.val(name(0)+".xa", f32(rr.Chans), slotDMixed)
				prog.ops = append(prog.ops, devOp{Kind: dkConvBias, Src: []int{i}, In: []devVal{conv}, Out: []devVal{xa}})
				qx := operand(xa, i, false)
				low := mv(".dt_low", qx, posLowA, slotDLow, l.ms[posLowA].rows)
				b, c := mv(".b", qx, posGateA, slotM1B, l.ms[posGateA].rows), mv(".c", qx, posGateB, slotM1C, l.ms[posGateB].rows)
				if l.m1DtN != nil {
					norm := func(v devVal, s devSlot) devVal {
						n := prog.val(prog.names[v]+".n", prog.vals[v], s)
						prog.ops = append(prog.ops, devOp{Kind: dkSSMNorm, Src: []int{i}, In: []devVal{v}, Out: []devVal{n}})
						return n
					}
					low, b, c = norm(low, slotM1LowN), norm(b, slotM1BN), norm(c, slotM1CN)
				}
				lq := prog.val(prog.names[low]+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[low].Width}, slotLRQ8)
				prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{i}, In: []devVal{low}, Out: []devVal{lq}})
				dt := mv(".dt", lq, posLowB, slotDBRaw, l.ms[posLowB].rows, low)
				st := prog.val(name(0)+".out", f32(inner), slotDOut)
				prog.ops = append(prog.ops, devOp{Kind: dkScan, Src: []int{i}, In: []devVal{xa, b, c, dt}, Out: []devVal{st}})
				act := prog.val(name(0)+".act", f32(inner), slotDAct)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedNorm, Src: []int{i}, In: []devVal{st, z}, Out: []devVal{act}, Staged: true})
				qa := operand(act, i, false)
				out := prog.val(name(0), f32(ne), slotOut)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qa}, Out: []devVal{out}, W: posSSMOut})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
			if rr := p.Recurrent; rr.ChanDecay && !l.centred && l.q4e == nil {
				// Kimi Delta Attention: the fused in-projection, the output
				// and decay gates each through a low-rank pair (the second
				// matvec reading its own int8 operand), beta's logits, the
				// convolution and the per-channel scan over the session's
				// state, the sigmoid-gated norm, projected out.
				fullGate := posGateA >= len(l.rs) || *l.rs[posGateA] == nil
				need := []int{posSSMInProj, posSSMBA, posSSMOut, posLowA, posLowB}
				if fullGate {
					need = append(need, posSSMGate) // one full-rank output gate
				} else {
					need = append(need, posGateA, posGateB)
				}
				for _, w := range need {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a delta rule whose projections the block's device copy does not hold"
					}
				}
				inner := rr.VHeads * rr.VDim
				q := operand(in(0), i, false)
				mv := func(nm string, q devVal, w int, s devSlot, width int, extra ...devVal) devVal {
					v := prog.val(name(0)+nm, f32(width), s)
					prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: append([]devVal{q}, extra...), Out: []devVal{v}, W: w})
					return v
				}
				lowRank := func(nm string, wa, wb int, s devSlot, width int) devVal {
					low := mv(nm+"_low", q, wa, slotDLow, l.ms[wa].rows)
					lq := prog.val(prog.names[low]+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[low].Width}, slotLRQ8)
					prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{i}, In: []devVal{low}, Out: []devVal{lq}})
					return mv(nm, lq, wb, s, width, low)
				}
				mixed := mv(".mixed", q, posSSMInProj, slotDMixed, rr.Chans)
				var z devVal
				if fullGate {
					z = mv(".z", q, posSSMGate, slotDZ, inner)
				} else {
					z = lowRank(".z", posGateA, posGateB, slotDZ, inner)
				}
				alpha := lowRank(".alpha", posLowA, posLowB, slotDAlpha, l.ms[posLowB].rows)
				beta := mv(".beta", q, posSSMBA, slotDBRaw, l.ms[posSSMBA].rows)
				conv := prog.val(name(0)+".conv", f32(rr.Chans), slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{mixed}, Out: []devVal{conv}})
				st := prog.val(name(0)+".out", f32(inner), slotDOut)
				prog.ops = append(prog.ops, devOp{Kind: dkScan, Src: []int{i}, In: []devVal{conv, alpha, beta}, Out: []devVal{st}})
				act := prog.val(name(0)+".act", f32(inner), slotDAct)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedNorm, Src: []int{i}, In: []devVal{st, z}, Out: []devVal{act}, Staged: true})
				qa := operand(act, i, false)
				out := prog.val(name(0), f32(ne), slotOut)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qa}, Out: []devVal{out}, W: posSSMOut})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
			if rr := p.Recurrent; !l.centred && l.k3 == nil && !rr.ShortConv && !rr.Mamba1 && !rr.ChanDecay {
				// The delta rule or Mamba-2's SSD as primitives: the
				// projections by role (a mixer beside an attention has its own
				// in-projection), the convolution and the scan over the
				// session's state, the gated norm, the out projection.
				// The tier's own slots for a recurrence's four projections
				// (layer.tensors' order), until the positional list goes.
				inW, zW, baW, oW, outSlot := posSSMInProj, posSSMGate, posSSMBA, posSSMOut, slotOut
				if rr.WithAttn {
					inW, outSlot = posSSMIn, slotSide
				}
				for _, w := range [...]int{inW, zW, baW, oW} {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a recurrence whose projections the block's device copy does not hold"
					}
				}
				inner := rr.VHeads * rr.VDim
				q := operand(in(0), i, false)
				mixed := prog.val(name(0)+".mixed", f32(rr.Chans), slotDMixed)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{mixed}, W: inW})
				z := prog.val(name(0)+".z", f32(inner), slotDZ)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{z}, W: zW})
				gs := slotDBA
				if rr.SSD {
					gs = slotDBRaw
				}
				gates := prog.val(name(0)+".gates", f32(l.ms[baW].rows), gs)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{gates}, W: baW})
				conv := prog.val(name(0)+".conv", f32(rr.Chans), slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{mixed}, Out: []devVal{conv}})
				st := prog.val(name(0)+".out", f32(inner), slotDOut)
				prog.ops = append(prog.ops, devOp{Kind: dkScan, Src: []int{i}, In: []devVal{conv, gates}, Out: []devVal{st}})
				act := prog.val(name(0)+".act", f32(inner), slotDAct)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedNorm, Src: []int{i}, In: []devVal{st, z}, Out: []devVal{act}, Staged: true})
				qa := operand(act, i, false)
				out := prog.val(name(0), f32(ne), outSlot)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qa}, Out: []devVal{out}, W: oW})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
			// emitLinear runs the block's pre-norm and quantize itself, from
			// the block's row, and writes the mixer's output: the norm the
			// graph states before it goes into this op.
			src := []int{i}
			x, normed := in(0), devVal(-1)
			if n := len(prog.ops); n > 0 && prog.ops[n-1].Kind == dkNorm && prog.ops[n-1].Out[0] == x {
				src = append(prog.ops[n-1].Src, i)
				normed, x = x, prog.ops[n-1].In[0]
				prog.ops = prog.ops[:n-1]
			}
			if prog.slots[x] != slotX {
				return nil, "a recurrent mixer not over the block's row"
			}
			// Beside an attention (the graph has both) the mixer leaves its
			// output in bs.side for the sum, and the normed row it wrote is
			// the attention's.
			s := slotOut
			if hasOp(g, graph.OpAttend) {
				s = slotSide
			}
			out := prog.val(name(0), f32(ne), s)
			outs := []devVal{out}
			if normed >= 0 {
				outs = append(outs, normed)
			}
			prog.ops = append(prog.ops, devOp{Kind: dkRecurrent, Src: src, In: []devVal{x}, Out: outs})
			vals[o.Out[0]] = out
		case graph.OpRoute:
			// The router as primitives: logits, a bias, a group mask, the
			// top k and their weights, each from the op's attributes and
			// bindings.
			switch {
			case l.stream != nil && l.stream.hybrid:
				return nil, "experts run on the host beside the card"
			case l.router == nil:
				return nil, "a router the block's device copy does not hold"
			}
			var selBias, hash jlm.Role
			for _, w := range o.Weight[1:] {
				switch w.Role {
				case jlm.RoleExpProbsB:
					selBias = w.Role
				case jlm.RoleHashExperts:
					hash = w.Role
				}
			}
			ex, k := o.Attr.Experts, o.Attr.TopK
			logits := prog.val(name(0)+".logits", f32(ex), slotRLogits)
			prog.ops = append(prog.ops, devOp{Kind: dkRouterLogits, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{logits}})
			for _, w := range o.Weight[1:] {
				if w.Role != jlm.RoleRouterBias {
					continue
				}
				lb := prog.val(name(0)+".logits_b", f32(ex), slotRLogitsB)
				prog.ops = append(prog.ops, devOp{Kind: dkRouterBias, Src: []int{i}, In: []devVal{logits}, Out: []devVal{lb}, Role: w.Role})
				logits = lb
			}
			rank := []devVal{logits}
			if o.Attr.Groups > 1 {
				plane := prog.val(name(0)+".plane", f32(ex), slotRBM)
				prog.ops = append(prog.ops, devOp{Kind: dkGroupMask, Src: []int{i}, In: []devVal{logits}, Out: []devVal{plane}, Role: selBias})
				rank = append(rank, plane)
			}
			if routeOf(p).RowBias {
				// The rank kernel reads a bias per row: the block's, or on a
				// hash-routed block the table's experts for the row's token.
				rb := prog.val(name(0)+".row_bias", f32(ex), slotRBias)
				prog.ops = append(prog.ops, devOp{Kind: dkRowBias, Src: []int{i}, Out: []devVal{rb}, Role: hash})
				rank = append(rank, rb)
			}
			sel := prog.val(name(0), devType{Elem: elemU32, Layout: layoutRow, Width: k}, slotSel)
			top := prog.val(name(0)+".top", f32(k), slotTop)
			prog.ops = append(prog.ops, devOp{Kind: dkTopK, Src: []int{i}, In: rank, Out: []devVal{sel, top}, Role: selBias})
			rw := prog.val(name(0)+".weights", f32(k), slotRW)
			prog.ops = append(prog.ops, devOp{Kind: dkRouteWeights, Src: []int{i}, In: []devVal{top, logits}, Out: []devVal{rw}})
			vals[o.Out[0]] = sel
			routeW[sel] = rw
			prog.ops = append(prog.ops, devOp{Kind: dkGroupByExpert, Src: []int{i}, In: []devVal{sel}})
			if l.stream != nil {
				// A paged bank: the selected sheets come to the card here,
				// and the matvecs index what they landed in.
				sw := prog.val(name(0)+".slots", devType{Elem: elemU32, Layout: layoutRow, Width: k}, slotSelW)
				prog.ops = append(prog.ops, devOp{Kind: dkExpertFetch, Src: []int{i}, In: []devVal{sel}, Out: []devVal{sw}, Staged: true})
				weightSel[sel] = sw
				prog.streamed = true
			}
		case graph.OpGather:
			// One row: the selection indexes the weight side, so the
			// experts read the row itself.
			// Weighted at their input (Llama 4's): the weight scales the
			// activation, and the combine sums at one.
			if o.Attr.WeightIn {
				weightIn[in(1)] = true
			}
			vals[o.Out[0]] = in(0)
		case graph.OpExpertMatVec:
			role := o.Weight[0].Role
			wi := l.matIndex(role)
			if wi < 0 {
				return nil, "an expert bank " + role.String() + ", which the block's device copy does not hold"
			}
			var s, sb devSlot
			switch role {
			case jlm.RoleExpGateBank:
				s, sb = slotEG, slotEGB
			case jlm.RoleExpUpBank:
				s, sb = slotEU, slotEUB
			case jlm.RoleExpDownBank:
				s, sb = slotEDown, slotEDownB
			default:
				return nil, "an expert matvec over " + role.String()
			}
			// The graph's value is a token's whole selection, k pairs; the
			// program's is a row a pair, so a pair's width.
			width := g.Values[o.Out[0]].Width
			if k := prog.vals[in(1)].Width; k > 0 && width%k == 0 {
				width /= k
			}
			a := operand(in(0), i, true)
			out := prog.val(name(0), devType{Elem: elemF32, Layout: layoutPairs, Width: width}, s)
			prog.experts = true
			ix := in(1)
			if w, ok := weightSel[ix]; ok {
				ix = w
			}
			prog.ops = append(prog.ops, devOp{Kind: dkExpertMatVec, Src: []int{i}, In: []devVal{a, ix, in(0)}, Out: []devVal{out}, Role: role, W: wi})
			if len(o.Weight) > 1 {
				b := prog.val(name(0)+".biased", like(out, width), sb)
				prog.ops = append(prog.ops, devOp{Kind: dkExpertBias, Src: []int{i}, In: []devVal{out, in(1)}, Out: []devVal{b}, Role: o.Weight[1].Role})
				out = b
			}
			vals[o.Out[0]] = out
		case graph.OpCombine:
			rw, ok := routeW[in(1)]
			if !ok {
				return nil, "a combine of no selection"
			}
			if weightIn[in(1)] {
				rw = prog.val("ones", prog.vals[rw], slotOnes)
			}
			if len(o.Weight) > 0 {
				ws := prog.val(name(0)+".weights", prog.vals[rw], slotRWS)
				prog.ops = append(prog.ops, devOp{Kind: dkScaleWeights, Src: []int{i}, In: []devVal{rw, in(1)}, Out: []devVal{ws}, Role: o.Weight[0].Role})
				rw = ws
			}
			cs := slotOut
			if g.Values[o.Out[0]].Width != ne {
				cs = slotLatAcc // experts in a latent: their sum there
			}
			out := prog.val(name(0), f32(g.Values[o.Out[0]].Width), cs)
			prog.ops = append(prog.ops, devOp{Kind: dkCombine, Src: []int{i}, In: []devVal{in(0), rw}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpGateScalar:
			// The gate's logit now; the scale rides the add that brings the
			// branch in (dkGatedAdd).
			lg := prog.val(name(0)+".logit", f32(1), slotShLogit)
			prog.ops = append(prog.ops, devOp{Kind: dkGateLogit, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{lg}, Role: o.Weight[0].Role})
			vals[o.Out[0]] = in(1)
			gated[o.Out[0]] = lg
		case graph.OpLatentAppend:
			hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: prog.vals[in(0)].Width + prog.vals[in(1)].Width}, slotKV)
			k := devOp{Kind: dkLatentWrite, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{hist}}
			if o.In[1] == kRot {
				k.N = 1 // rotate the key as it is stored
			}
			prog.ops = append(prog.ops, k)
		case graph.OpAttendLatent:
			if len(o.In) > 2 && !lightning {
				return nil, "an attention indexer"
			}
			kb, vb := l.matIndex(o.Weight[0].Role), l.matIndex(o.Weight[1].Role)
			if kb < 0 || vb < 0 {
				return nil, "a latent attention whose banks the block's device copy does not hold"
			}
			nh := o.Attr.NHead
			lat := o.Weight[0].Rows
			ident := prog.val("ident", devType{Elem: elemU32, Layout: layoutRow, Width: nh}, slotMIdent)
			// Absorb: each head's unrotated query against its sheet of the k
			// up-projection, one selected matvec over the heads.
			qa := operand(in(0), i, false)
			raw := prog.val(name(0)+".abs_raw", f32(nh*lat), slotMAbsRaw)
			prog.ops = append(prog.ops, devOp{Kind: dkExpertMatVec, Src: []int{i}, In: []devVal{qa, ident, in(0)}, Out: []devVal{raw}, Role: o.Weight[0].Role, W: kb, Heads: nh})
			abs := prog.val(name(0)+".query", f32(nh*(lat+o.Attr.NRot)), slotMAbs)
			prog.ops = append(prog.ops, devOp{Kind: dkConcatHeads, Src: []int{i}, In: []devVal{raw, in(1)}, Out: []devVal{abs}})
			acc := prog.val(name(0)+".acc", f32(nh*lat), slotMAcc)
			if len(o.In) > 2 {
				// Over the indexer's top positions, gathered.
				prog.ops = append(prog.ops, devOp{Kind: dkIdxAttendLatent, Src: []int{i}, In: []devVal{abs, in(2)}, Out: []devVal{acc}})
			} else {
				prog.ops = append(prog.ops, devOp{Kind: dkLatentAttend, Src: []int{i}, In: []devVal{abs}, Out: []devVal{acc}})
			}
			// Un-absorb: each head's attention through its sheet of the v
			// up-projection.
			qc := operand(acc, i, false)
			out := prog.val(name(0), f32(g.Values[o.Out[0]].Width), slotMOut)
			prog.ops = append(prog.ops, devOp{Kind: dkExpertMatVec, Src: []int{i}, In: []devVal{qc, ident, acc}, Out: []devVal{out}, Role: o.Weight[1].Role, W: vb, Heads: nh})
			vals[o.Out[0]] = out
			prog.latent = true
		case graph.OpNGram:
			if p.Q4E == nil {
				return nil, "an n-gram block with no device scratch"
			}
			role := func(r jlm.Role) jlm.Role {
				for _, w := range o.Weight {
					if w.Role == r {
						return r
					}
				}
				return jlm.RoleNone
			}
			kw, vw := l.matIndex(jlm.RoleNGramKey), l.matIndex(jlm.RoleNGramValue)
			if kw < 0 || vw < 0 {
				return nil, "an n-gram block whose projections the block's device copy does not hold"
			}
			// The call's table rows (staged) into their own operand, key
			// and value; the key normed against the normed streams, their
			// dot the gate; the gated value normed and convolved over the
			// block's state; both added into every stream.
			sw := prog.vals[in(0)]
			emb := prog.val("ng_emb", f32(p.Q4E.NGramWidth), slotNEmb)
			qe := prog.val("ng_emb.q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[emb].Width}, slotSQ8W)
			prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{i}, In: []devVal{emb}, Out: []devVal{qe}})
			key := prog.val("ng_key", sw, slotNKey)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qe, emb}, Out: []devVal{key}, W: kw})
			val := prog.val("ng_val", sw, slotNVal)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qe, emb}, Out: []devVal{val}, W: vw})
			norm := func(nm string, v devVal, r jlm.Role, s devSlot) devVal {
				out := prog.val(nm, prog.vals[v], s)
				prog.ops = append(prog.ops, devOp{Kind: dkStreamNorm, Src: []int{i}, In: []devVal{v}, Out: []devVal{out}, Role: r})
				return out
			}
			kn := norm("ng_kn", key, role(jlm.RoleNGramNormKey), slotNKn)
			qn := norm("ng_qn", in(0), role(jlm.RoleNGramNormQuery), slotNQn)
			dot := prog.val("ng_dot", f32(o.Attr.Streams), slotNDot)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamDot, Src: []int{i}, In: []devVal{kn, qn}, Out: []devVal{dot}})
			gv := prog.val("ng_gv", sw, slotNGv)
			prog.ops = append(prog.ops, devOp{Kind: dkGatedVal, Src: []int{i}, In: []devVal{dot, val}, Out: []devVal{gv}})
			gvn := norm("ng_gvn", gv, role(jlm.RoleNGramNormConv), slotNGvn)
			conv := prog.val("ng_conv", sw, slotNConv)
			prog.ops = append(prog.ops, devOp{Kind: dkNGramConv, Src: []int{i}, In: []devVal{gvn}, Out: []devVal{conv}, Role: role(jlm.RoleNGramConv)})
			xg := prog.val(name(0), sw, slotNXg)
			prog.ops = append(prog.ops, devOp{Kind: dkNGramAdd, Src: []int{i}, In: []devVal{in(0), gv, conv}, Out: []devVal{xg}})
			vals[o.Out[0]] = xg
			prog.scans = true
		case graph.OpCompressedAttend:
			// Over the normed row: a low-rank query normed per head and
			// turned; one key head, normed and turned, into the history row
			// (with a compressing block's pooled value and gate, and its
			// indexer's); every closed block pooled into an entry, normed,
			// turned, into the second history; an indexer's top entries where
			// it selects; the window and the entries gathered for one
			// attention with sinks; the heads turned back; the grouped
			// low-rank output.
			// Read off the bindings: a compressor's pooled value, and an
			// indexer's beside it.
			comp, csa := false, false
			for _, w := range o.Weight {
				comp = comp || w.Role == jlm.RoleCompKV
				csa = csa || w.Role == jlm.RoleIdxCompKV
			}
			idx := map[jlm.Role]int{}
			for _, w := range o.Weight {
				switch w.Role {
				case jlm.RoleAttnQA, jlm.RoleAttnQB, jlm.RoleAttnK, jlm.RoleAttnOut, jlm.RoleCompKV, jlm.RoleCompGate,
					jlm.RoleIdxQB, jlm.RoleIdxProj, jlm.RoleIdxCompKV, jlm.RoleIdxCompGate:
					// The matrices it multiplies by; the grouped output runs
					// its own plan, the rest are tables and vectors.
					if idx[w.Role] = l.matIndex(w.Role); idx[w.Role] < 0 {
						return nil, "a compressed attention whose projections the block's device copy does not hold"
					}
				}
			}
			h, H, hd := in(0), o.Attr.NHead, o.Attr.HeadDim
			mvc := func(nm string, src devVal, r jlm.Role, s devSlot, w int) devVal {
				out := prog.val(nm, f32(w), s)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{operand(src, i, false)}, Out: []devVal{out}, W: idx[r]})
				return out
			}
			un := func(kind devKind, nm string, src devVal, s devSlot, n int) devVal {
				out := prog.val(nm, prog.vals[src], s)
				prog.ops = append(prog.ops, devOp{Kind: kind, Src: []int{i}, In: []devVal{src}, Out: []devVal{out}, N: n})
				return out
			}
			width := func(r jlm.Role) int {
				for _, w := range o.Weight {
					if w.Role == r {
						return w.Rows
					}
				}
				return 0
			}
			qa := mvc("c_qa", h, jlm.RoleAttnQA, slotCQA, width(jlm.RoleAttnQA))
			kraw := mvc("c_k", h, jlm.RoleAttnK, slotCKRaw, hd)
			var ckv, cg, ikv, ig, iw devVal
			if comp {
				ckv = mvc("c_ckv", h, jlm.RoleCompKV, slotCKV, width(jlm.RoleCompKV))
				cg = mvc("c_cg", h, jlm.RoleCompGate, slotCG, width(jlm.RoleCompGate))
			}
			if csa {
				ikv = mvc("c_ikv", h, jlm.RoleIdxCompKV, slotCIKV, width(jlm.RoleIdxCompKV))
				ig = mvc("c_ig", h, jlm.RoleIdxCompGate, slotCIG, width(jlm.RoleIdxCompGate))
				iw = mvc("c_iw", h, jlm.RoleIdxProj, slotCIW, width(jlm.RoleIdxProj))
			}
			qan := un(dkRowRMS, "c_qa_n", qa, slotCQAN, 0)
			q := mvc("c_q", qan, jlm.RoleAttnQB, slotCQ, H*hd)
			var iqr devVal
			if csa {
				iq := mvc("c_iq", qan, jlm.RoleIdxQB, slotCIQ, width(jlm.RoleIdxQB))
				iqr = un(dkTurn, "c_iq_rot", iq, slotCIQR, 0)
			}
			qr := un(dkTurn, "c_q_rot", un(dkRowRMS, "c_q_n", q, slotCQN, 0), slotCQR, 0)
			kr := un(dkTurn, "c_k_rot", un(dkRowRMS, "c_k_n", kraw, slotCKN, 0), slotCKR, 0)
			hist := prog.val("c_hist", devType{Elem: elemF32, Layout: layoutKV, Width: hd}, slotKV)
			prog.ops = append(prog.ops, devOp{Kind: dkCacheWrite, Src: []int{i}, In: []devVal{kr}, Out: []devVal{hist}, Staged: true})
			var gIn []devVal
			if comp {
				prog.ops = append(prog.ops, devOp{Kind: dkCacheWrite, Src: []int{i}, In: []devVal{ckv, cg}, Out: []devVal{hist}, Staged: true, N: 1})
				if csa {
					prog.ops = append(prog.ops, devOp{Kind: dkCacheWrite, Src: []int{i}, In: []devVal{ikv, ig}, Out: []devVal{hist}, Staged: true, N: 2})
				}
				ent := prog.val("c_entries", devType{Elem: elemF32, Layout: layoutKV, Width: hd}, slotKV)
				pool := func(n int, nm string, s, sn, sr devSlot) {
					pe := prog.val(nm, f32(hd), s)
					prog.ops = append(prog.ops, devOp{Kind: dkPoolEntries, Src: []int{i}, In: []devVal{hist}, Out: []devVal{pe}, Staged: true, N: n})
					per := un(dkTurn, nm+"_rot", un(dkRowRMS, nm+"_n", pe, sn, 0), sr, 0)
					prog.ops = append(prog.ops, devOp{Kind: dkEntryWrite, Src: []int{i}, In: []devVal{per}, Out: []devVal{ent}, Staged: true, N: n})
				}
				pool(0, "c_pe", slotCPE, slotCPEN, slotCPER)
				if csa {
					pool(1, "c_ipe", slotCIPE, slotCIPEN, slotCIPER)
					sel := prog.val("c_sel", devType{Elem: elemU32, Layout: layoutRow, Width: 1}, slotCSel)
					prog.ops = append(prog.ops, devOp{Kind: dkEntrySelect, Src: []int{i}, In: []devVal{iqr, iw}, Out: []devVal{sel}, Staged: true})
					gIn = append(gIn, sel)
				}
			}
			gk := prog.val("c_gk", f32(hd), slotCGK)
			prog.ops = append(prog.ops, devOp{Kind: dkGatherEntries, Src: []int{i}, In: gIn, Out: []devVal{gk}, Staged: true,
				Need: []devType{{Elem: elemU32, Layout: layoutRow, Width: 1}}})
			ao := prog.val("c_ao", f32(H*hd), slotCAO)
			prog.ops = append(prog.ops, devOp{Kind: dkAttendGathered, Src: []int{i}, In: []devVal{qr, gk}, Out: []devVal{ao}})
			aor := un(dkTurn, "c_ao_rot", ao, slotCAOR, 0)
			oa := prog.val("c_oa", f32(o.Attr.Groups*width(jlm.RoleAttnOutA)), slotCOA)
			prog.ops = append(prog.ops, devOp{Kind: dkGroupedMatVec, Src: []int{i}, In: []devVal{operand(aor, i, false)}, Out: []devVal{oa}})
			vals[o.Out[0]] = mvc(name(0), oa, jlm.RoleAttnOut, slotOut, ne)
		case graph.OpStreamMix:
			if o.Attr.Sinkhorn && len(o.Out) == 2 {
				// The streams flat, RMS-normed, scored against the
				// sublayer's weights into coefficients, those Sinkhorn-
				// normalised into read and placement weights, the streams
				// weighed into the row. N marks the attention's weights.
				att := 0
				for _, w := range o.Weight {
					if w.Role == jlm.RoleHCAttnFn {
						att = 1
					}
				}
				n := o.Attr.Streams
				flat := prog.val(name(0)+".flat", f32(n*ne), slotHFlat)
				prog.ops = append(prog.ops, devOp{Kind: dkStreamFlatten, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{flat}})
				fn := prog.val(name(0)+".flat_n", f32(n*ne), slotHFlatN)
				prog.ops = append(prog.ops, devOp{Kind: dkFlatNorm, Src: []int{i}, In: []devVal{flat}, Out: []devVal{fn}})
				mx := prog.val(name(0)+".coeff", f32((2+n)*n), slotHMix)
				prog.ops = append(prog.ops, devOp{Kind: dkMixLogits, Src: []int{i}, In: []devVal{fn}, Out: []devVal{mx}, N: att})
				hc := prog.val(name(1), f32((2+n)*n), slotHC)
				prog.ops = append(prog.ops, devOp{Kind: dkSinkhorn, Src: []int{i}, In: []devVal{mx}, Out: []devVal{hc}, N: att})
				row := prog.val(name(0), f32(ne), slotHColl)
				prog.ops = append(prog.ops, devOp{Kind: dkStreamCollect, Src: []int{i}, In: []devVal{in(0), hc}, Out: []devVal{row}})
				vals[o.Out[0]], vals[o.Out[1]] = row, hc
				continue
			}
			if len(o.Out) == 1 && !o.Attr.Sinkhorn {
				// A banked read (residual attention): the bank's entries
				// scored against the row's weights, their softmax-weighted
				// sum the sublayer's row; an empty bank reads the row as it is.
				nb := o.Attr.Groups
				if nb == 0 {
					// The row as it is: the first stream of the streams,
					// which are stored stream-major, so a view of their
					// prefix at the row's width.
					vals[o.Out[0]] = prog.val(name(0), f32(ne), prog.slots[in(0)])
					continue
				}
				sc := prog.val(name(0)+".scores", f32(nb+1), slotBScores)
				prog.ops = append(prog.ops, devOp{Kind: dkStreamScore, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{sc}, Role: o.Weight[0].Role, N: nb})
				row := prog.val(name(0), f32(ne), slotBMix)
				prog.ops = append(prog.ops, devOp{Kind: dkStreamWeighted, Src: []int{i}, In: []devVal{in(0), sc}, Out: []devVal{row}, N: nb})
				vals[o.Out[0]] = row
				continue
			}
			if o.Attr.Sinkhorn || len(o.Out) != 2 || p.Q4E == nil {
				return nil, "a stream mix other than the low-rank gated one"
			}
			// The streams normed and quantized into their own operand; a
			// low-rank sigmoid gate (down, SiLU, its own operand, up); the
			// gated average, the sublayer's row; the placement's projection
			// and weights.
			ws := o.Weight
			rows := func(r jlm.Role) int {
				if w := l.matIndex(r); w >= 0 {
					return l.ms[w].rows
				}
				return 0
			}
			xn := prog.val(name(0)+".streams_normed", prog.vals[in(0)], slotSXN)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{xn}, Role: ws[0].Role})
			qx := prog.val(name(0)+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[xn].Width}, slotSQ8)
			prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{i}, In: []devVal{xn}, Out: []devVal{qx}})
			mv := func(nm string, q, src devVal, r jlm.Role, s devSlot) devVal {
				v := prog.val(name(0)+nm, f32(rows(r)), s)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q, src}, Out: []devVal{v}, W: l.matIndex(r)})
				return v
			}
			lo := mv(".lo", qx, xn, ws[1].Role, slotSLo)
			lo2 := prog.val(name(0)+".lo_act", prog.vals[lo], slotSLo2)
			// The gate's down projection through SiLU of its input over the
			// stream count: OpStreamMix's own definition (graph.OpStreamMix),
			// stated on the step as an activation and an input scale.
			prog.ops = append(prog.ops, devOp{Kind: dkAct, Src: []int{i}, In: []devVal{lo}, Out: []devVal{lo2}, Act: graph.ActSiLU,
				Alpha: float64(1 / float32(o.Attr.Streams))})
			ql := prog.val(name(0)+".lo_q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[lo2].Width}, slotSQ8LR)
			prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{i}, In: []devVal{lo2}, Out: []devVal{ql}})
			gate := mv(".gate", ql, lo2, ws[2].Role, slotSGate)
			row := prog.val(name(0), f32(ne), slotSMix)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamGateMix, Src: []int{i}, In: []devVal{xn, gate}, Out: []devVal{row}})
			inj := mv(".inj", qx, xn, ws[3].Role, slotSInj)
			w := prog.val(name(1), f32(o.Attr.Streams), slotSW)
			prog.ops = append(prog.ops, devOp{Kind: dkPlaceWeights, Src: []int{i}, In: []devVal{inj}, Out: []devVal{w}})
			vals[o.Out[0]], vals[o.Out[1]] = row, w
		case graph.OpStreamPlace:
			// Every stream with the sublayer's output added at its weights:
			// a residual add over the streams, into the slot it does not read.
			s := slotX2
			if prog.slots[in(0)] == slotX2 {
				s = slotX
			}
			out := prog.val(name(0), prog.vals[in(0)], s)
			ins := []devVal{in(0), in(1)}
			if len(o.In) > 2 {
				ins = append(ins, in(2)) // the gated form's weights
			}
			// A banked placement's checkpoint block banks its input (N).
			prog.ops = append(prog.ops, devOp{Kind: dkStreamPlace, Src: []int{i}, In: ins, Out: []devVal{out}, N: o.Attr.Block})
			vals[o.Out[0]] = out
			adds++
		case graph.OpAltPredict:
			// The active stream normed, the router's weights, every stream's
			// prediction, the active one's copied into the row.
			active := prog.val("x.active", f32(ne), slotX)
			prog.ops = append(prog.ops, devOp{Kind: dkView, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{active}, Staged: true})
			hn := prog.val(name(1)+".normed", f32(ne), slotH)
			prog.ops = append(prog.ops, devOp{Kind: dkPostNorm, Src: []int{i}, In: []devVal{active}, Out: []devVal{hn}, Role: o.Weight[0].Role})
			m := prog.val(name(0)+".route", f32(o.Attr.Streams), slotAltM)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamRoute, Src: []int{i}, In: []devVal{hn}, Out: []devVal{m}})
			pred := prog.val(name(0), prog.vals[in(0)], slotPred)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamPredict, Src: []int{i}, In: []devVal{in(0), m}, Out: []devVal{pred}})
			act := prog.val(name(1), f32(ne), slotX)
			prog.ops = append(prog.ops, devOp{Kind: dkCopy, Src: []int{i}, In: []devVal{pred}, Out: []devVal{act}})
			vals[o.Out[0]], vals[o.Out[1]] = pred, act
		case graph.OpLaurel:
			// in + norm(R(L(in))): the rank's middle, its own quantize, back
			// to the row, normed into the corrections' first plane, added.
			if p.AltUp == 0 {
				return nil, "a low-rank branch with no rank"
			}
			lt := prog.val(name(0)+".mid", f32(p.LaurelRank), slotLT)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{operand(in(0), i, false)}, Out: []devVal{lt}, W: l.matIndex(o.Weight[0].Role)})
			back := prog.val(name(0)+".back", f32(ne), slotOut)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{operand(lt, i, false)}, Out: []devVal{back}, W: l.matIndex(o.Weight[1].Role)})
			bn := prog.val(name(0)+".normed", f32(ne), slotAltB)
			prog.ops = append(prog.ops, devOp{Kind: dkPostNorm, Src: []int{i}, In: []devVal{back}, Out: []devVal{bn}, Role: o.Weight[2].Role})
			lr := prog.val(name(0), f32(ne), slotLaur)
			prog.ops = append(prog.ops, devOp{Kind: dkAdd, Src: []int{i}, In: []devVal{in(0), bn}, Out: []devVal{lr}})
			vals[o.Out[0]] = lr
		case graph.OpGaussTopK:
			if prog.slots[in(0)] != slotG {
				return nil, "a gaussian top-k not of the gate"
			}
			out := prog.val(name(0), prog.vals[in(0)], slotGG)
			prog.ops = append(prog.ops, devOp{Kind: dkGaussTopK, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpAltCorrect:
			// The block's active row normed, the router's weights, every
			// stream corrected, the active correction scaled and projected
			// through the per-layer input, every stream written back.
			ws := o.Weight
			hn := prog.val(name(0)+".normed", f32(ne), slotH)
			prog.ops = append(prog.ops, devOp{Kind: dkPostNorm, Src: []int{i}, In: []devVal{in(1)}, Out: []devVal{hn}, Role: ws[0].Role})
			m := prog.val(name(0)+".route", f32(o.Attr.Streams), slotAltM)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamRoute, Src: []int{i}, In: []devVal{hn}, Out: []devVal{m}})
			b := prog.val(name(0)+".corr", prog.vals[in(0)], slotAltB)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamCorrect, Src: []int{i}, In: []devVal{in(0), in(1), m}, Out: []devVal{b}})
			sc := prog.val(name(0)+".scaled", f32(ne), slotH)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamScale, Src: []int{i}, In: []devVal{b}, Out: []devVal{sc}})
			pg := prog.val(name(0)+".ple_gate", f32(p.PLEDim), slotPLEG)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{operand(sc, i, false)}, Out: []devVal{pg}, W: l.matIndex(ws[4].Role)})
			pa := prog.val(name(0)+".ple_act", f32(p.PLEDim), slotPLEA)
			prog.ops = append(prog.ops, devOp{Kind: dkActMul, Src: []int{i}, In: []devVal{pg, in(2)}, Out: []devVal{pa}})
			po := prog.val(name(0)+".ple_out", f32(ne), slotPLEO)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{operand(pa, i, false)}, Out: []devVal{po}, W: l.matIndex(ws[5].Role)})
			pn := prog.val(name(0)+".ple_normed", f32(ne), slotH)
			prog.ops = append(prog.ops, devOp{Kind: dkPostNorm, Src: []int{i}, In: []devVal{po}, Out: []devVal{pn}, Role: ws[6].Role})
			out := prog.val(name(0), prog.vals[in(0)], slotX)
			prog.ops = append(prog.ops, devOp{Kind: dkStreamFinish, Src: []int{i}, In: []devVal{b, pn}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpClamp:
			if len(o.Weight) != 0 {
				// A pair of the block's bounds table: an input into its own
				// slot, an output into its matrix's clamped slot.
				cs := slotCIn
				if o.Attr.Offset%4 == 2 {
					switch prog.slots[in(0)] {
					case slotQ:
						cs = slotQC
					case slotK:
						cs = slotKC
					case slotV:
						cs = slotVC
					case slotG:
						cs = slotGC
					case slotU:
						cs = slotUC
					default:
						cs = slotCOut
					}
				}
				out := prog.val(name(0), prog.vals[in(0)], cs)
				prog.ops = append(prog.ops, devOp{Kind: dkClampAt, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, N: o.Attr.Offset / 2, V: l.clampB})
				vals[o.Out[0]] = out
				continue
			}
			var cs devSlot
			switch prog.slots[in(0)] {
			case slotQ:
				cs = slotQC
			case slotK:
				cs = slotKC
			case slotV:
				cs = slotVC
			default:
				return nil, "a clamp of neither q, k nor v"
			}
			out := prog.val(name(0), prog.vals[in(0)], cs)
			prog.ops = append(prog.ops, devOp{Kind: dkClamp, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Alpha: o.Attr.Bound})
			vals[o.Out[0]] = out
		case graph.OpScale:
			// The layer's output scale over the block's row.
			if len(o.Weight) != 1 || o.Weight[0].Role != jlm.RoleLayerOutScale {
				return nil, "a scale the program does not lower"
			}
			out := prog.val(name(0), f32(ne), slotX)
			prog.ops = append(prog.ops, devOp{Kind: dkScale, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Staged: true, V: l.outScale})
			vals[o.Out[0]] = out
		case graph.OpSlice:
			if mla {
				// Latent attention's slices: each head's unrotated and
				// rotary parts of q, compacted; the kv latent's prefix, read
				// in place; its rotary key.
				src, w := prog.slots[in(0)], g.Values[o.Out[0]].Width
				var k devKind
				var ss devSlot
				switch {
				case src == slotQ && o.Attr.HeadDim == p.HeadDim && o.Attr.Offset == 0:
					k, ss = dkSlice, slotMNope
				case src == slotQ && o.Attr.HeadDim == p.HeadDim:
					k, ss = dkSlice, slotMPE
				case src == slotMKV && o.Attr.HeadDim == 0 && o.Attr.Offset == 0:
					k, ss = dkView, slotMKVLat
				case src == slotMKV && o.Attr.HeadDim == 0:
					k, ss = dkSlice, slotMKPE
				default:
					return nil, "a latent slice the program does not lower"
				}
				out := prog.val(name(0), f32(w), ss)
				prog.ops = append(prog.ops, devOp{Kind: k, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Staged: k == dkView})
				vals[o.Out[0]] = out
				continue
			}
			// A gated query's split: two slices of q's projection, its heads
			// and its gate, lowered as one launch at the first.
			if prog.slots[in(0)] != slotQ || o.Attr.HeadDim != 2*p.HeadDim || o.Attr.Width != p.HeadDim {
				return nil, "a slice the program does not lower"
			}
			if o.Attr.Offset != 0 {
				continue // the second half, written with the first
			}
			gate := graph.ValueID(-1)
			for k := i + 1; k < len(g.Ops); k++ {
				if n := g.Ops[k]; n.Kind == graph.OpSlice && n.In[0] == o.In[0] && n.Attr.Offset == p.HeadDim {
					gate = n.Out[0]
				}
			}
			if gate < 0 {
				return nil, "a gated query without its gate"
			}
			qh := prog.val(name(0), f32(qd), slotQG)
			gv := prog.val(g.Values[gate].Name, f32(qd), slotOGate)
			prog.ops = append(prog.ops, devOp{Kind: dkSplitQG, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{qh, gv}})
			vals[o.Out[0]], vals[gate] = qh, gv
		case graph.OpHeadNorm:
			var role jlm.Role
			if len(o.Weight) > 0 {
				role = o.Weight[0].Role
			}
			src := in(0)
			hd := o.Attr.HeadDim
			switch {
			case role == 0 && o.In[0] == kRot:
				// A weightless (L2) norm of k after its rotary: a rotation
				// keeps a head's norm, so it norms k before, and the rotary
				// still writes the history (the norm then the rotation).
				out := prog.val(name(0), prog.vals[src], slotKN)
				prog.ops = append(prog.ops, devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{src}, Out: []devVal{out}, HD: hd})
				kRot = o.Out[0]
				vals[o.Out[0]] = out
				continue
			case role == 0 && prog.slots[src] == slotQR:
				// The same for q: its rotary's input is normed first.
				r := len(prog.ops) - 1
				for r >= 0 && prog.ops[r].Kind != dkRopeQ {
					r--
				}
				if r < 0 || prog.ops[r].Out[0] != src {
					return nil, "an L2 norm of q not after its rotary"
				}
				qn := prog.val(name(0)+".pre", prog.vals[prog.ops[r].In[0]], slotQN)
				norm := devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{prog.ops[r].In[0]}, Out: []devVal{qn}, HD: hd}
				prog.ops = append(prog.ops[:r], append([]devOp{norm}, prog.ops[r:]...)...)
				prog.ops[r+1].In[0] = qn
				vals[o.Out[0]] = src
				continue
			}
			s := slotQN
			switch prog.slots[src] {
			case slotK, slotV, slotKC, slotVC, slotKRot:
				// k's norm; or v's (weightless, on k's heads), which reads
				// the projection k's norm does not overwrite where v is k's.
				s = slotKN
				if role == 0 && o.Out[0] != kRot && isValue(g, o.Out[0]) {
					s = slotVN
				}
			case slotQ, slotQG, slotQR, slotQC:
				// q's norm, before its rotary or after it (a weighted norm
				// after the rotary does not commute with it).
			default:
				return nil, "a head norm of neither q nor k"
			}
			out := prog.val(name(0), prog.vals[src], s)
			prog.ops = append(prog.ops, devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{src}, Out: []devVal{out}, Role: role, HD: hd})
			vals[o.Out[0]] = out
		case graph.OpRope:
			switch prog.slots[in(0)] {
			case slotMPE:
				// The query's rotary parts, compacted.
				out := prog.val(name(0), prog.vals[in(0)], slotMPErot)
				prog.ops = append(prog.ops, devOp{Kind: dkRopeTail, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			case slotMKPE:
				// The shared rotary key: rotated as the latent write stores it.
				kRot = o.Out[0]
				vals[o.Out[0]] = in(0)
				continue
			case slotK, slotKC:
				if p.RopeSplit {
					// XD-RoPE: k turns into a row of its own, which its norm
					// reads before the history copies it in.
					out := prog.val(name(0), prog.vals[in(0)], slotKRot)
					prog.ops = append(prog.ops, devOp{Kind: dkRopeKRow, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
					vals[o.Out[0]] = out
					continue
				}
			}
			if s := prog.slots[in(0)]; s == slotQ || s == slotQG || s == slotQN || s == slotQC {
				out := prog.val(name(0), f32(qd), slotQR)
				prog.ops = append(prog.ops, devOp{Kind: dkRopeQ, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
			} else {
				// k's rotary writes the history itself (the device has no
				// rotated-k row): it lowers with the kv_append that reads it.
				kRot = o.Out[0]
				vals[o.Out[0]] = in(0)
			}
		case graph.OpKVAppend:
			if x := selectOf(g); x != nil {
				// A block-selecting indexer: its key rides the cached row as
				// one more head, so its projections, norms and the row's
				// assembly come before the history write; its query turns
				// for the scores the attention's selection reads.
				h := vals[x.In[0]]
				kw, qw := l.matIndex(jlm.RoleIdxK), l.matIndex(jlm.RoleIdxQ)
				if kw < 0 || qw < 0 {
					return nil, "an indexer whose projections the block's device copy does not hold"
				}
				hq := operand(h, i, false)
				kRaw := prog.val("idx_k", f32(x.Attr.HeadDim), slotIKRaw)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{hq}, Out: []devVal{kRaw}, W: kw})
				qRaw := prog.val("idx_q", f32(x.Attr.NHead*x.Attr.HeadDim), slotIQRaw)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{hq}, Out: []devVal{qRaw}, W: qw})
				qn := prog.val("idx_q_normed", prog.vals[qRaw], slotIQn)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxNorm, Src: []int{i}, In: []devVal{qRaw}, Out: []devVal{qn}})
				key := kRaw
				if !x.Attr.Pooled {
					// The selecting form's key is normed; the pooled form's
					// is the raw projection, normed per block as it pools.
					key = prog.val("idx_k_normed", prog.vals[kRaw], slotIKn)
					prog.ops = append(prog.ops, devOp{Kind: dkIdxNorm, Src: []int{i}, In: []devVal{kRaw}, Out: []devVal{key}})
				}
				qr := prog.val("idx_q_rot", prog.vals[qn], slotIQRot)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxRope, Src: []int{i}, In: []devVal{qn}, Out: []devVal{qr}})
				all := prog.val("k_all", f32(kv), slotKAll)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxAssemble, Src: []int{i}, In: []devVal{in(0), key}, Out: []devVal{all}})
				hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: kv}, slotKV)
				prog.ops = append(prog.ops, devOp{Kind: dkKVWrite, Src: []int{i - 1, i}, In: []devVal{all, in(1)}, Out: []devVal{hist}})
				idxQuery = qr
				continue
			}
			if o.In[0] != kRot {
				// A layer without a rotary (NoPE): k and v copied in.
				hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: kv}, slotKV)
				prog.ops = append(prog.ops, devOp{Kind: dkKVCopy, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{hist}})
				continue
			}
			hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: kv}, slotKV)
			prog.ops = append(prog.ops, devOp{Kind: dkKVWrite, Src: []int{i - 1, i}, In: []devVal{in(0), in(1)}, Out: []devVal{hist}})
		case graph.OpIndex:
			if len(o.In) == 2 {
				// The lightning indexer over latent attention: its key
				// LayerNormed, turned and written into the position's latent
				// row; its per-head weights; its query off the query latent,
				// turned. The selection is the attention's to compute.
				kw, ww, qw := l.matIndex(jlm.RoleIdxK), l.matIndex(jlm.RoleIdxProj), l.matIndex(jlm.RoleIdxQB)
				if kw < 0 || ww < 0 || qw < 0 {
					return nil, "an indexer whose projections the block's device copy does not hold"
				}
				hq := operand(in(0), i, false)
				kRaw := prog.val("idx_k", f32(o.Attr.HeadDim), slotIKRaw)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{hq}, Out: []devVal{kRaw}, W: kw})
				wts := prog.val("idx_w", f32(o.Attr.NHead), slotIdxW)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{hq}, Out: []devVal{wts}, W: ww})
				kn := prog.val("idx_k_normed", prog.vals[kRaw], slotIKn)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxNorm, Src: []int{i}, In: []devVal{kRaw}, Out: []devVal{kn}})
				kr := prog.val("idx_k_rot", prog.vals[kn], slotIKRot)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxRope, Src: []int{i}, In: []devVal{kn}, Out: []devVal{kr}})
				hist := prog.val("idx_hist", devType{Elem: elemF32, Layout: layoutKV, Width: o.Attr.HeadDim}, slotKV)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxWrite, Src: []int{i}, In: []devVal{kr}, Out: []devVal{hist}, Staged: true})
				qRaw := prog.val("idx_q", f32(o.Attr.NHead*o.Attr.HeadDim), slotIQRaw)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{operand(in(1), i, false)}, Out: []devVal{qRaw}, W: qw})
				qr := prog.val("idx_q_rot", prog.vals[qRaw], slotIQRot)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxRope, Src: []int{i}, In: []devVal{qRaw}, Out: []devVal{qr}})
				vals[o.Out[0]] = qr
				lightning = true
				continue
			}
			// Lowered with the append (its key) and the attention (its
			// selection): the selection is the attention's to compute.
			if idxQuery < 0 || o.Attr.NHead == 0 || o.Attr.Block == 0 {
				return nil, "an attention indexer"
			}
			vals[o.Out[0]] = idxQuery
		case graph.OpAttend:
			q := in(0)
			if len(o.In) > 1 {
				// Attention over the indexer's selection: the scores, the
				// best blocks, their keys and values gathered, attended.
				sc := prog.val(name(0)+".scores", f32(1), slotISel)
				prog.ops = append(prog.ops, devOp{Kind: dkIdxScore, Src: []int{i}, In: []devVal{in(1)}, Out: []devVal{sc}, Staged: true})
				sel := prog.val(name(0)+".selection", f32(1), slotISel)
				prog.ops = append(prog.ops, devOp{Kind: dkSelectBlocks, Src: []int{i}, In: []devVal{sc}, Out: []devVal{sel}, Staged: true})
				gk := prog.val(name(0)+".gathered", f32(1), slotISel)
				prog.ops = append(prog.ops, devOp{Kind: dkGatherKV, Src: []int{i}, In: []devVal{sel}, Out: []devVal{gk}, Staged: true})
				out := prog.val(name(0), f32(qd), slotXB)
				prog.ops = append(prog.ops, devOp{Kind: dkAttendGathered, Src: []int{i}, In: []devVal{q, gk}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			if o.Attr.TempScale != 0 {
				// The query's temperature, a scalar a row: it commutes with
				// the rotary, so it scales whichever q the attention reads.
				ts := slotQR
				if prog.slots[q] == slotQR {
					ts = slotQT
				}
				qt := prog.val(prog.names[q]+".temp", prog.vals[q], ts)
				prog.ops = append(prog.ops, devOp{Kind: dkQTemp, Src: []int{i}, In: []devVal{q}, Out: []devVal{qt}})
				q = qt
			}
			out := prog.val(name(0), f32(qd), slotXB)
			prog.ops = append(prog.ops, devOp{Kind: dkAttend, Src: []int{i}, In: []devVal{q}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpAdd:
			if o.Attr.Scale != 0 {
				// A three-way join: the residual add just lowered with the
				// branch beside it, over sqrt(2), one launch.
				n := len(prog.ops) - 1
				if o.Attr.Scale != graph.LaurelJoin || n < 0 || prog.ops[n].Kind != dkAdd || prog.ops[n].Out[0] != in(0) {
					return nil, "a scaled add that is not a three-way join"
				}
				r := prog.ops[n]
				prog.ops[n] = devOp{Kind: dkResidJoin3, Src: append(r.Src, i), In: []devVal{r.In[0], r.In[1], in(1)}, Out: r.Out}
				vals[o.Out[0]] = r.Out[0]
				continue
			}
			if lg, ok := gated[o.In[1]]; ok {
				// The routed sum and a gated shared expert: one launch.
				out := prog.val(name(0), f32(ne), slotShSum)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedAdd, Src: []int{i}, In: []devVal{in(0), in(1), lg}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			// The add that is the block's output writes its row; another
			// (the attention's residual, a mixture's sum) the residual's
			// second slot.
			home := len(g.Out) == 1 && o.Out[0] == g.Out[0]
			reads := func(sl devSlot) bool { return prog.slots[in(0)] == sl || prog.slots[in(1)] == sl }
			if !reads(slotX) && !reads(slotX2) {
				// A sum of two branches (an attention's and a mixer's beside
				// it, a dense MLP's and a mixture's), not a residual add: into
				// the first of bs.h, bs.mvOut and bs.shsum it does not read.
				bsum := slotH
				for _, c := range [...]devSlot{slotH, slotOut, slotShSum} {
					if !reads(c) {
						bsum = c
						break
					}
				}
				out := prog.val(name(0), f32(ne), bsum)
				prog.ops = append(prog.ops, devOp{Kind: dkAdd, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			s := slotX2
			switch {
			case home && !reads(slotX):
				s = slotX
			case !home && reads(slotX2):
				// An add a later op ends the block after (a per-layer
				// embedding, a scale) writes the slot it does not read.
				s = slotX
			}
			out := prog.val(name(0), f32(ne), s)
			prog.ops = append(prog.ops, devOp{Kind: dkAdd, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}, Alpha: o.Attr.Alpha})
			if home && s != slotX {
				// The block's only add reads its row: the sum goes to x2 and
				// is moved home, as no launch reads what it writes (RULE 13).
				row := prog.val(name(0)+".home", f32(ne), slotX)
				prog.ops = append(prog.ops, devOp{Kind: dkCopy, Src: []int{i}, In: []devVal{out}, Out: []devVal{row}})
				out = row
			}
			vals[o.Out[0]] = out
			adds++
		case graph.OpAct:
			// An ungated FFN: the activation of up's row alone.
			out := prog.val(name(0), like(in(0), prog.vals[in(0)].Width), actSlot(prog.slots[in(0)]))
			prog.ops = append(prog.ops, devOp{Kind: dkAct, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Act: o.Attr.Act})
			vals[o.Out[0]] = out
		case graph.OpActMul:
			if o.Attr.Act == graph.ActSigmoid && prog.slots[in(0)] == slotMGate {
				// Latent attention's output gate: sigmoid(gate) * attention.
				out := prog.val(name(0), prog.vals[in(1)], slotMGated)
				prog.ops = append(prog.ops, devOp{Kind: dkGateOut, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			if o.Attr.Act == graph.ActSigmoid && prog.slots[in(0)] == slotOGate {
				// The attention's output gate: sigmoid(gate) * attention.
				out := prog.val(name(0), f32(qd), slotOGated)
				prog.ops = append(prog.ops, devOp{Kind: dkGateOut, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			out := prog.val(name(0), like(in(0), prog.vals[in(0)].Width), actSlot(prog.slots[in(0)]))
			ins := []devVal{in(0), in(1)}
			if s := actSlot(prog.slots[in(0)]); s == slotEAct {
				for sel := range weightIn {
					ins = append(ins, routeW[sel])
				}
			}
			prog.ops = append(prog.ops, devOp{Kind: dkActMul, Src: []int{i}, In: ins, Out: []devVal{out}, Act: o.Attr.Act})
			vals[o.Out[0]] = out
		}
	}
	// Pre-norms, post-norms or both (OLMo 2 has no pre-norm: its sublayers
	// read the residual), one residual add a sublayer (a recurrent mixer
	// alone has one; a per-layer embedding is a third), ending in the
	// block's row.
	// Every norm binds its weight and bias by role, from the graph op it
	// lowers: the device copy prep uploaded (layer.vecs).
	for i := range prog.ops {
		o := &prog.ops[i]
		switch o.Kind {
		case dkNorm, dkPostNorm, dkHeadNorm, dkSubNorm:
		case dkRouterBias, dkGroupMask, dkTopK, dkExpertBias, dkScaleWeights, dkGateLogit, dkStreamNorm, dkNGramConv, dkStreamScore:
			// A vector operand of the mixture's, by its role.
			if o.Role != jlm.RoleNone {
				if o.V = l.vecOf(o.Role); o.V == nil {
					return nil, "a vector " + o.Role.String() + ", which the block's device copy does not hold"
				}
			}
			continue
		default:
			continue
		}
		if o.Role == jlm.RoleNone {
			continue // a weightless norm
		}
		// A norm op's weight and bias are its bindings; a norm inside a
		// composite op (a branch's, a stream router's) binds the role its
		// lowering named.
		gop := g.Ops[o.Src[0]]
		ws := gop.Weight
		if gop.Kind != graph.OpNorm && gop.Kind != graph.OpLayerNorm && gop.Kind != graph.OpHeadNorm {
			ws = []graph.Binding{{Role: o.Role}}
		}
		if o.V = l.vecOf(ws[0].Role); o.V == nil {
			return nil, "a norm over " + ws[0].Role.String() + ", which the block's device copy does not hold"
		}
		if len(ws) > 1 {
			if o.VB = l.vecOf(ws[1].Role); o.VB == nil {
				return nil, "a norm bias " + ws[1].Role.String() + ", which the block's device copy does not hold"
			}
		}
		if g.Ops[o.Src[0]].Kind == graph.OpLayerNorm && o.VB == nil {
			// A LayerNorm kernel always reads a bias, and a file without
			// one gives the block zeros (the model's lnBias): until vectors
			// the model synthesises carry roles, the block's own.
			switch ws[0].Role {
			case jlm.RoleAttnNorm:
				o.VB = l.nAttnB
			case jlm.RoleFFNNorm:
				o.VB = l.nFFNB
			}
			if o.VB == nil {
				return nil, "a layer norm over " + ws[0].Role.String() + " with no bias on the device"
			}
		}
	}
	if adds == 0 || adds > 3 || len(g.Out) != 1 || prog.slots[vals[g.Out[0]]] != slotX {
		return nil, "not a block of one to three residual adds"
	}
	for _, o := range prog.ops {
		switch o.Kind {
		case dkAttend, dkLatentAttend, dkAttendGathered, dkIdxAttendLatent:
			prog.attends = true
		case dkRopeQ, dkKVWrite:
			prog.ropes = true
		}
	}
	// Every matrix an op binds is one the block holds.
	for _, o := range prog.ops {
		if (o.Kind == dkMatVec || o.Kind == dkExpertMatVec) && (o.W < 0 || o.W >= len(l.rs) || *l.rs[o.W] == nil) {
			return nil, "a matvec over a matrix the block's device copy does not hold"
		}
	}
	if err := prog.check(); err != nil {
		return nil, err.Error()
	}
	// Each op carries its first source graph op's attributes: what its
	// generated kernel is built from (genop.go).
	for i := range prog.ops {
		if src := prog.ops[i].Src; len(src) > 0 && src[0] < len(g.Ops) {
			prog.ops[i].A = g.Ops[src[0]].Attr
		}
	}
	return prog, ""
}

// compositeKind reports that kind k is not one of the program's primitives
// (primitiveKind): a mechanism of one family run whole, by an encoder
// sequence or by a generator written for that kind alone. It cannot be fused,
// reordered or re-tiled, and a model arranging the same pieces differently
// does not build, so a block holding one does not count as compiled
// (GPU.BlockComposite). The set only shrinks (TestCompositeKindsOnlyShrink),
// to empty as the builder lowers each graph op into primitives.
func compositeKind(k devKind) bool { return !primitiveKind(k) }

// primitiveKind is the program's primitive set (docs/design/composer.md,
// "Primitives only"), each stated by structure and its attributes so any
// model whose graph uses it runs: norms (a row's, a head's, a sublayer
// output's), the int8 operand, matvec plain and over a bank's selected
// sheets, the elementwise maps (add, mul, an activation, a gated activation,
// a scale, a clamp), the index maps (copy, slice, view, concatenation), the
// rotary, the paged history's write, copy and attention, top-k, the
// selection's combine, a bank's fetch (a runtime op inserted by property),
// and convolution's depthwise and affine forms. Every other kind is a
// mechanism of one family run whole: composite, to be lowered into these.
func primitiveKind(k devKind) bool {
	switch k {
	case dkNorm, dkHeadNorm, dkPostNorm, dkQuant, dkMatVec, dkExpertMatVec, dkAdd, dkMul, dkAct, dkActMul,
		dkScale, dkClamp, dkCopy, dkSlice, dkView, dkConcatHeads, dkRopeQ, dkKVWrite, dkKVCopy, dkAttend,
		dkTopK, dkCombine, dkExpertFetch, dkDWConvRows, dkAffineRows:
		return true
	}
	return false
}

// BlockComposite is the first composite op (compositeKind) in block li's
// program, by name, or "" where every op is a primitive or the device holds
// no program for it.
func (g *GPU) BlockComposite(li int) string {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return ""
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	l := d.layers[li]
	if l == nil || l.prog == nil {
		return ""
	}
	for _, o := range l.prog.ops {
		if compositeKind(o.Kind) {
			return o.Kind.String()
		}
	}
	return ""
}

// BlockProgram is block li's device program as text and, where its last step
// did not run it (or the program does not lower the block), why; both empty
// where the device does not hold the block.
func (g *GPU) BlockProgram(li int) (prog, why string) {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return "", ""
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	l := d.layers[li]
	if l == nil {
		return "", ""
	}
	if l.prog == nil {
		return "", l.progWhy
	}
	var out []string
	for _, f := range [3]progFused{l.progOne, l.progRows, l.progRag} {
		if f.prog != nil {
			head := "rewrites: " + f.took.names(d.rewrites)
			if f.err != nil {
				head += " (refused: " + f.err.Error() + ")"
			}
			out = append(out, head+"\n"+f.prog.String())
		}
	}
	if l.progRun != progRanProgram {
		why = l.progRun.String()
	}
	if l.progRunRows != progRanProgram {
		why += "; rows: " + l.progRunRows.String()
	}
	if out != nil {
		return strings.Join(out, ""), why
	}
	return l.prog.String(), why
}

// progReason is why a step ran a block through the encoder rather than its
// program: a byte the step writes, named only when reported.
type progReason uint8

const (
	progRanProgram     progReason = iota
	progNotLowered                // the program does not lower the block (progWhy)
	progNoProgram                 // Config.NoProgram
	progRowsNotPrefill            // rows that are not a prefill chunk
	progFault                     // a fault skips a kernel
	progNotPaged                  // not a paged decode or prefill
	progRotary                    // a rotary the program does not lower
	progTableSplit                // TableSplit
	progRowsExperts               // experts over several rows (grouped expert kernels not yet lowered)
	progStreamTuning              // a paged bank under a probe, prefetch or fixed selection
	progRowsLatent                // latent attention over several rows (latent prefill not yet lowered)
	numProgReasons
)

func (r progReason) String() string {
	switch r {
	case progNotLowered:
		return "not lowered"
	case progNoProgram:
		return "NoProgram"
	case progRowsExperts:
		return "experts over several rows"
	case progRowsLatent:
		return "latent attention over several rows"
	case progStreamTuning:
		return "a paged bank under a probe, prefetch or fixed selection"
	case progRowsNotPrefill:
		return "rows that are not a prefill chunk"
	case progFault:
		return "a fault skips a kernel"
	case progNotPaged:
		return "not a paged decode or prefill"
	case progRotary:
		return "a rotary the program does not lower"
	case progTableSplit:
		return "TableSplit"
	}
	return ""
}

// bindsRole reports that some op of g reads a weight of role r.
func bindsRole(g *graph.Graph, r jlm.Role) bool {
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if w.Role == r {
				return true
			}
		}
	}
	return false
}

// isValue reports that v is the value a history append stores (its second
// input): a norm writing it is v's.
func isValue(g *graph.Graph, v graph.ValueID) bool {
	for _, o := range g.Ops {
		if o.Kind == graph.OpKVAppend && len(o.In) > 1 && o.In[1] == v {
			return true
		}
	}
	return false
}

// readsRoleOut reports that op o reads the output of an op binding role r.
func readsRoleOut(g *graph.Graph, o graph.Op, r jlm.Role) bool {
	for _, p := range g.Ops {
		if len(p.Weight) == 0 || p.Weight[0].Role != r {
			continue
		}
		for _, x := range o.In {
			for _, y := range p.Out {
				if x == y {
					return true
				}
			}
		}
	}
	return false
}

// hasOp reports that g has an op of kind k.
func hasOp(g *graph.Graph, k graph.OpKind) bool {
	for _, o := range g.Ops {
		if o.Kind == k {
			return true
		}
	}
	return false
}

// matIndex is the index of block l's matrix of role r in its list (layer.rs,
// layer.ms), or -1 where the block holds none.
func (l *layer) matIndex(r jlm.Role) int {
	for _, a := range l.roles {
		if a.role == r {
			return a.i
		}
	}
	return -1
}

// actSlot is where an activation of a row in slot s goes: the experts' rows'
// to theirs, a shared expert's to its own, a dense FFN's to bs.act.
func actSlot(s devSlot) devSlot {
	switch s {
	case slotEG, slotEU, slotEGB, slotEUB:
		return slotEAct
	case slotShG, slotShU:
		return slotShAct
	case slotPLEG:
		return slotPLEA
	}
	return slotAct
}

// constSlot reports that slot s holds a constant the block's scratch is
// built with: read without a writer.
func constSlot(s devSlot) bool { return s == slotOnes || s == slotMIdent || s == slotNEmb }

// A recurrence's projections in the tier's positional list (layer.tensors'
// order): the in-projection (the fused q|k|v, or x|B|C), the output gate,
// the gate streams, the out projection, and a mixer's own in-projection
// beside an attention.
const (
	posSSMInProj = 0
	posSSMGate   = 10
	posSSMBA     = 11
	posSSMOut    = 12
	posSSMIn     = 29
	// The low-rank pairs (KDA's decay and output gates; Mamba-1's dt, B
	// and C off x ride the same four).
	posLowA  = 18
	posLowB  = 19
	posGateA = 20
	posGateB = 21
)

// selectOf is block graph g's block-selecting indexer op (a selection over
// GQA keys), or nil.
func selectOf(g *graph.Graph) *graph.Op {
	for i := range g.Ops {
		if o := &g.Ops[i]; o.Kind == graph.OpIndex && o.Attr.Block > 0 && len(o.In) == 1 {
			return o
		}
	}
	return nil
}

// Faults as program rewrites. A fault switch (a test's violation, set per
// instance) that leaves an op out of a block is applied to the block's
// program before its kernels are generated: the op is removed and what read
// its output reads its input, so the executor holds no fault at all and the
// program it runs is the program a dump prints. The switches name the
// mechanism they break, which is why they are matched here and nowhere in the
// executor.

// elide removes op i of p, its readers reading value keep in its output's
// place.
func (p *devProgram) elide(i int, keep devVal) {
	out := p.ops[i].Out[0]
	for j := i + 1; j < len(p.ops); j++ {
		for k, v := range p.ops[j].In {
			if v == out {
				p.ops[j].In[k] = keep
			}
		}
	}
	p.ops = append(p.ops[:i], p.ops[i+1:]...)
}

// faultSig is block l's fault switches as one number: what a fused variant
// was rewritten for, so a switch that changes rebuilds it.
func faultSig(l *layer, plan *nn.LayerPlan) uint32 {
	var s uint32
	if l.k3 != nil {
		s = uint32(l.k3.fault)
	}
	if plan.DS4 != nil {
		s |= uint32(plan.DS4.Fault) << 8
	}
	if plan.Q4E != nil {
		s |= uint32(plan.Q4E.Fault) << 16
	}
	return s
}

// swapLatentBanks is MLAFaultSwapBanks as a rewrite of the program: each
// latent attention absorbs its query through the value bank and un-absorbs
// through the key bank, the two bank steps (a selected matvec over the
// heads, the first before the attention and the second after it, from one
// graph op) exchanging their weights.
func swapLatentBanks(p *devProgram) {
	first := -1
	for i := range p.ops {
		o := &p.ops[i]
		if o.Kind != dkExpertMatVec || o.Heads == 0 {
			continue
		}
		if first < 0 || p.ops[first].Src[0] != o.Src[0] {
			first = i
			continue
		}
		a := &p.ops[first]
		a.W, o.W = o.W, a.W
		a.Role, o.Role = o.Role, a.Role
		first = -1
	}
}

// faultRewrite applies block l's fault switches to its program.
func faultRewrite(p *devProgram, l *layer, plan *nn.LayerPlan) {
	for i := 0; i < len(p.ops); i++ {
		o := p.ops[i]
		if len(o.Out) == 0 {
			continue
		}
		s := p.slots[o.Out[0]]
		switch {
		case l.k3 != nil && l.k3.fault == nn.K3FaultNoMLAGate && o.Kind == dkGateOut && s == slotMGated:
			p.elide(i, o.In[1]) // the latent attention's output, ungated
		case l.k3 != nil && l.k3.fault == nn.K3FaultNoLatentNorm && o.Kind == dkPostNorm && s == slotLatN:
			p.elide(i, o.In[0]) // the latent sum, unnormed
		case plan.DS4 != nil && plan.DS4.Fault == nn.DS4FaultNoDerope && o.Kind == dkTurn && s == slotCAOR:
			p.elide(i, o.In[0]) // the heads left turned
		case plan.Q4E != nil && plan.Q4E.Fault == nn.Q4EFaultUnscaledGate && o.Kind == dkAct && o.Alpha != 0:
			p.ops[i].Alpha = 0 // the stream gate's activation unscaled
			continue
		default:
			continue
		}
		i--
	}
}
