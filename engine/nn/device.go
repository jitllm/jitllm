//go:build amd64 || arm64

package nn

import (
	"errors"
	"math"
	"strings"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// Device is an optional accelerator that can serve a decode matvec.
//
// It is declared here and implemented in jit_old/gpu so that the core module keeps
// no dependency on the FFI layer: go.mod has one require and there is no cgo
// (goffi reaches the drivers through dlopen).
//
// MatVec returns false for anything it cannot serve (an unsupported quant type,
// a shape it has no kernel for, a weight it has not uploaded) and the caller
// runs the CPU path. A nil Device and a Device that declines are the same.
type Device interface {
	// MatVec computes out[0:nrows] = W x, or returns false to decline.
	// w is the raw GGUF tensor bytes, so the Device owns any residency and
	// repacking it needs, keyed on the pointer it was handed. The pointer is
	// the identity: a caller may offer only bytes that stay at that address,
	// unchanged, until it tells the Device otherwise (CopyForgetter). A
	// pager's reused frame does not stay.
	MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool
}

// Device is the accelerator SetDevice attached, nil for none.
func (f *JIT) Device() Device {
	if f == nil {
		return nil
	}
	return f.device
}

// SetDevice attaches an accelerator. Passing nil detaches it.
func (f *JIT) SetDevice(d Device) {
	if f == nil {
		return
	}
	f.device = d
	// A device that quantizes activations host-side has to use this session's
	// window or the two tiers will not agree on token ids. Optional, because a
	// device that never quantizes has no opinion.
	if w, ok := d.(ActWindower); ok {
		w.SetActWindow(f.actWindow)
	}
}

// Weight is one matvec's operand: the raw GGUF bytes plus the shape that reads
// them. The Device keys residency on the pointer, exactly as MatVec does.
type Weight struct {
	T    quant.Type
	Data []byte
	Rows int // output features
	K    int // input features
	// Role is what the container calls the tensor, which a graph's binding
	// names: how the device program finds a block's weight without a field
	// per mechanism. RoleNone where nothing named it.
	Role jlm.Role

	// Packed is this weight already in the device layout, or nil. When set the
	// device writes the spans straight to a buffer and skips the repack, which
	// dominates upload cost on a model that streams blocks every token. The
	// spans are read-only and not owned here.
	Packed *Packed
}

// Vector is a block's vector (a norm's weight, a bias, a scale) with its
// role.
type Vector struct {
	Role jlm.Role
	V    []float32
}

// Packed is a weight already in the device layout: three spans of a jlm
// container, read by every tier. SC is empty for formats with no per-sub-block
// scale table (Q4_0, Q8_0).
//
// The layout is [sub-block][word][row], so rows are interleaved and a row range
// is not a byte range: slicing a packed bank by rows hands the kernel another
// expert's weights. Stride is the row count of the whole buffer and Row is
// where this weight starts in it; both zero means the whole buffer.
type Packed struct {
	QS, D, SC   []byte
	Row, Stride int
	// Sheet, when set, stands in for QS/D/SC: the weight is a mixture's bank
	// whose Sheets experts live in separate container pages, so there is no
	// one span of it anywhere. Sheet(x) is expert x's three planes, readable
	// until the returned release is called; a device assembles the bank from
	// them at x times one sheet's bytes, which is the layout its kernels index.
	Sheet  func(x int) (Packed, func(), error)
	Sheets int
}

// LayerWeights is one transformer block.
//
// Many fields below are optional graph features. A device that cannot apply
// one must decline the block rather than skip it: skipping runs and produces
// fluent, wrong text.
type LayerWeights struct {
	// All is every matrix the block holds, each with its role: the named
	// fields below are views a hand-written path reads, All is what a
	// consumer binding by role (the device program) reads. Its backing
	// array is the caller's and is reused: a device copies what it keeps.
	All []Weight
	// Vecs is every vector the block holds (norms, biases, scales), each
	// with its role, for the same consumer. Read-only.
	Vecs []Vector

	AttnNorm, FFNNorm []float32 // tiny, always F32 in the file, kept expanded
	// QNorm and KNorm are head_dim wide and nil unless the architecture
	// RMSNorms each head of q and k before RoPE (qwen3).
	QNorm, KNorm []float32
	// Clamp is a block's clipped-linear bounds (LayerPlan.Clamps), 28 floats:
	// for q, k, v, o, gate, up and down, the input's minimum and maximum and
	// the output's. nil everywhere else.
	Clamp []float32
	// Conv is a convolutional tower block's vectors (LayerPlan.Conv): its
	// norms and depthwise filters. Its matrices are Up and Down (an edge or
	// inverted residual's two) or Wq, Wk, Wv and Wo (an attention block's).
	Conv *ConvWeights
	// Bq, Bk, Bv, Bo are the attention biases (qwen2 and others), nil when the
	// model has none.
	Bq, Bk, Bv, Bo []float32
	// PostAttnNorm and PostFFNNorm are gemma2/gemma3's extra RMSNorms on the
	// attention and FFN outputs, each applied before its residual add. They are
	// the same op as AttnNorm with a different weight. LayerPlan.PostNorm says
	// the model has them so a tier can decline before uploading anything.
	PostAttnNorm, PostFFNNorm []float32
	// AttnSubNorm and FFNSubNorm are BitNet's inner RMSNorms (LayerPlan.SubNorm):
	// over the attention output (QDim wide) before wo, and over act(gate)*up
	// (NFFN wide) before down. nil elsewhere.
	AttnSubNorm, FFNSubNorm []float32
	// ResidNorm and ResidNormB are a post-norm block's last LayerNorm, over x
	// + mlp(x) (LayerPlan.PostResidNorm): BERT's output norm. nil elsewhere.
	ResidNorm, ResidNormB []float32
	// Gemma 4's mixture block (LayerPlan.DenseMoE): the experts' pre-norm, the
	// dense MLP's and the experts' post-norms and the router input's norm,
	// NEmbd wide, and one weight factor per expert. nil elsewhere.
	FFNNorm2, PostFFNNorm1, PostFFNNorm2, RouterNorm, ExpScale []float32
	// Sinks is gpt-oss's learned logit per head (LayerPlan.AttnSinks), NHead
	// wide. RouterB is its router's bias, NExpert wide; ExpGateB, ExpUpB and
	// ExpDownB are the experts' biases, NExpert blocks of NFFNExp (gate, up)
	// or NEmbd (down), expert-major. All nil elsewhere (LayerPlan.MoEBias).
	Sinks, RouterB             []float32
	ExpGateB, ExpUpB, ExpDownB []float32
	// ExpSelB is DeepSeek's e_score_correction_bias, NExpert wide, non-nil only
	// when LayerPlan.ExpertSelBias is set. Unlike RouterB, which biases the
	// router logits before gating, this is added after gating and only for the
	// selection; a model can carry both.
	ExpSelB []float32

	// AttnNormB and FFNNormB are the LayerNorm biases (LayerPlan.LayerNorm):
	// an RMSNorm has no bias and a LayerNorm does.
	AttnNormB, FFNNormB []float32
	// BUp and BDown are the MLP biases: an ungated FFN's (C6, a ViT) or a
	// gated one's that carries them (Qwen2.5-VL's tower).
	BUp, BDown []float32
	// XIELU is an Apertus block's activation numbers (alpha_p, alpha_n, beta,
	// eps; jlm.RoleXIELU), where LayerPlan.Act is ActXIELU. They are the
	// block's own, so they ride with its weights and move when it moves.
	XIELU []float32
	// BGate is a gated MLP's gate bias, nil where it has none.
	BGate          []float32
	Wq, Wk, Wv, Wo Weight
	// Gate, Up and Down are one expert's matrices in a dense block and the whole
	// bank of LayerPlan.NExpert of them when Router.Data is non-nil. Rows stays
	// one expert's row count either way; the bank is NExpert*Rows rows of the
	// same buffer and the device does its own indexing.
	Gate, Up, Down Weight
	// Router is ffn_gate_inp: NExpert rows of NEmbd, F32 in every qwen3moe
	// file. Non-nil Data is the MoE branch: the tensor's presence decides it,
	// not the config.
	Router Weight
	// ShGate, ShUp, ShDown and ShRouter are the shared expert: an always-on FFN
	// whose gate is one weight per model dimension rather than a matrix, so the
	// gate is a single sigmoid of a dot product. nil unless LayerPlan.NFFNShExp
	// is non-zero.
	ShGate, ShUp, ShDown Weight
	ShRouter             []float32
	// PLEGate and PLEProj are Gemma 4's per-layer embedding gate (n_embd ->
	// PLEDim) and projection back, PLEPost its norm; zero unless
	// LayerPlan.PLEDim is set.
	PLEGate, PLEProj Weight
	PLEPost          []float32
	// Gemma 3n's AltUp and LAuReL (LayerPlan.AltUp): the router (AltUp rows of
	// n_embd, F32) and its norm, the prediction and correction coefficients
	// transposed, the active stream's scale before the per-layer gate, and
	// LAuReL's two matrices and norm.
	AltRouter                                       Weight
	AltRouterNorm, AltPredT, AltCorrT, AltCorrScale []float32
	LaurelL, LaurelR                                Weight
	LaurelPost                                      []float32
	// SSM is a linear-attention block's weights, and every field is zero for an
	// attention block. Wq carries the fused q|k|v projection for such a block;
	// the layer kind decides the consumer.
	SSM SSMWeights

	// Multi-head latent attention's matrices and norms, all zero unless
	// LayerPlan.KVLoraRank is set. MLA has no k or v projection: Wkva serves
	// both, and the up-projections are absorbed into the operands either side
	// of the softmax,
	//
	//	score_h = q_nope_h . (W_k_h . c)  =  (W_k_hᵀ . q_nope_h) . c
	//	out_h   = Σ a_j (W_v_h . c_j)     =  W_v_h . (Σ a_j c_j)
	//
	// so Wk and Wv are left empty.
	//
	// Wkb and Wvb are banks, one sheet per head (Wkb.Rows is NHead*KVLoraRank
	// of HeadDim-NRot, Wvb.Rows is NHead*HeadDimV of KVLoraRank), so the absorb
	// runs as one indexed launch with the head as the index. Wkb is stored
	// already transposed by the converter.
	//
	// Exactly one of Wq and (Wqa, Wqb) is populated, decided by QLoraRank.
	Wqa, Wqb, Wkva, Wkb, Wvb Weight
	// QANorm and KVANorm are the two latent RMSNorms: QANorm is QLoraRank wide
	// and nil where there is no query latent, KVANorm is KVLoraRank wide and
	// covers only the latent, not the cache row's rotary tail.
	QANorm, KVANorm []float32
	// DeepSeek V3.2's lightning indexer (LayerPlan.IdxHeads): its query from
	// the query latent, its key and the heads' weights from the block input,
	// and the key's LayerNorm. All zero without one.
	IdxQB, IdxK, IdxProj Weight
	IdxKNorm, IdxKNormB  []float32
	// MiniMax Sparse Attention (LayerPlan.MSA): the indexer's query from the
	// block's normed input and its per-head RMSNorm; the key is IdxK with
	// IdxKNorm (no bias). Zero without one.
	IdxQ     Weight
	IdxQNorm []float32
	// DeepSeek V4 (LayerPlan.DS4): the two hyper-connections' fn (F32, the
	// mixes by the flattened streams), per-mix scale and base; the grouped
	// output's first half (a bank of OGroups sheets of OLoraRank rows, the
	// second half Wo); the compressor's kv and gate projections, position
	// bias and norm (a compressed block), and the indexer compressor's (CSA,
	// whose query is IdxQB and weights IdxProj); and a hash block's token
	// table, NExpertUsed expert ids a token as floats. Wk is the one key
	// head (attn_kv), with KVANorm its norm. All zero elsewhere.
	HCAttnFn, HCFFNFn                          Weight
	HCAttnScale, HCAttnBase                    []float32
	HCFFNScale, HCFFNBase                      []float32
	WoA                                        Weight
	CompKV, CompGate, IdxCompKV, IdxCompGate   Weight
	CompAPE, CompNorm, IdxCompAPE, IdxCompNorm []float32
	HashExperts                                []float32
	// Kimi-K3 (LayerPlan.K3): the residual attention's two score vectors
	// (NEmbd, the reference's norm weight times its projection), the MLA
	// output gate (NHead*HeadDimV rows of NEmbd, on a full block of a model
	// that has one), and the latent mixture's down and up projections and
	// the routed sum's norm (ExpertLatent wide; nil where the reference has
	// none). All zero elsewhere. A linear block's full-rank output gate is
	// SSM.Gate, with SSM.GA and SSM.GB empty.
	ResAttn, ResFFN      []float32
	MLAGate              Weight
	RoutedDown, RoutedUp Weight
	RoutedNorm           []float32
	// Qwen3.8-Flash-Next (LayerPlan.Q4E): the two hyper-connections, and on
	// the n-gram block its weights.
	HCAttn, HCFFN HCWeights
	NGram         NGramWeights

	// Ensure makes the weight bytes valid, and is called by a tier once the
	// block is admitted and before anything is read. nil means they already
	// are, which is every caller but a paged mixture of experts.
	//
	// Admission reads only shapes and len(Data), so a mixture's routed bank is
	// not fetched until the tier has said yes. Ensure also refreshes w in
	// place: a tier that uploads a block more than once must call it again,
	// because the host pager may have moved the pages the earlier slices
	// pointed at. After it returns, every Weight in w is valid to read now.
	Ensure func(w *LayerWeights) error
	// EnsureExperts is Ensure for a caller that already knows which experts it
	// needs: the block's own weights in full, and of each routed bank only the
	// listed sheets. nil means the caller must use Ensure and take the whole
	// bank. sel is the selection in the order the kernels will index it, and
	// the callee must not retain it.
	EnsureExperts func(w *LayerWeights, sel []uint32) error
	// PrefetchExperts makes sel's sheets readable and does nothing else: it
	// does not re-point w or bind spans, so it is safe to call from another
	// goroutine while the caller uploads sheets it already has (EnsureExperts
	// writes through w and is not). nil means the caller must not overlap.
	PrefetchExperts func(sel []uint32) error
	// SpeculateExperts hands the host's read service a guess: sel's sheets
	// are read ahead in the background, never waited for, and replace any
	// earlier guess not yet started. A nil sel abandons the unstarted ones.
	// A demanded read (PrefetchExperts, EnsureExperts) of a guessed sheet
	// joins the guess's read and one that is not under way takes over its
	// place, so a wrong guess never delays the experts a token needs. nil
	// means the host does not speculate.
	SpeculateExperts func(sel []uint32)
	// HostExperts runs this block's routed experts on the host, for a device
	// that keeps the rest of the block (hybrid execution): out = sum over i
	// of w[i] * expert sel[i] applied to in, sel and w in selection order, in
	// the expert width the block's mixture reads (the latent on Kimi-K3). It
	// runs the host's generated kernels over the host's pages; only in and
	// out cross the bus. nil where the host cannot.
	HostExperts func(sel []uint32, w, in, out []float32) error
	// HostExpertsRows is HostExperts over rows of a chunk, expert-major: each
	// selected expert runs once over the rows that chose it. sel has a row
	// stride of k+1 (ExpertRank's layout), w of k, in and out of the expert
	// width.
	HostExpertsRows func(sel []uint32, w, in, out []float32, rows, k int) error
	// ExpertUses is the session's selection counts for this block's experts,
	// one a expert (model.Hotness): the host's routing adds to it, a device
	// routing the block adds its own selections, and a device's expert cache
	// keeps what it counts most. nil where nothing is counted.
	ExpertUses []uint32
}

// HCWeights are one Qwen3.8-Flash-Next hyper-connection's: the grouped
// norm (HCMult rows of NEmbd), the gate's down (HCMult*NEmbd -> HCLowRank)
// and up (back) projections, and the placement's projection (F32,
// HCMult*NEmbd -> HCMult).
type HCWeights struct {
	Norm     []float32
	Down, Up Weight
	Inject   Weight
}

// NGramWeights are Qwen3.8-Flash-Next's n-gram block's: the key (NGramWidth
// -> HCMult*NEmbd) and value (-> NEmbd) projections, three grouped norms
// (HCMult rows of NEmbd) and the convolution (NGramConv taps of
// HCMult*NEmbd channels, tap-major).
type NGramWeights struct {
	Key, Value                Weight
	NormKey, NormQuery, NormC []float32
	Conv                      []float32
}

// SSMWeights are the gated delta rule's tensors.
//
// The state is not a KV cache: it is a fixed VHeads x VDim x KDim matrix read
// and written every token. A device kernel may not read a buffer it writes, so
// a backend that runs this needs two state buffers and a swap.
type SSMWeights struct {
	Gate, BA, Out Weight    // the z projection, the beta/alpha head, the output
	Conv1d        []float32 // {Conv, Chans}, PLANE-MAJOR: see cpu.EmitConv1d
	A, DtBias     []float32 // per value head, or per CHANNEL when ChanDecay
	Norm          []float32 // one value head wide, shared by all of them
	// FA/FB and GA/GB are Kimi Delta Attention's two low-rank gates: the
	// forget gate whose softplus becomes the per-channel decay, and the output
	// gate that replaces Gate. All four are zero on qwen3next, which has one
	// matrix for the second and derives the first from BA's interleaved alpha.
	//
	// Mamba-1 (RecurrentPlan.Mamba1) carries its four matrices in the same
	// four: FA is x_proj's dt rows and FB dt_proj (the same bottleneck shape
	// as KDA's forget gate), GA x_proj's B rows and GB its C rows. Its dt, B
	// and C RMSNorm weights are DtNorm, BNorm and CNorm, nil where the block
	// has none.
	FA, FB, GA, GB       Weight
	DtNorm, BNorm, CNorm []float32
	// D and ConvBias are Mamba-2's skip (one per value head) and convolution
	// bias (one per channel); nil on the gated delta rule.
	D, ConvBias []float32
	// In is the mixer's x|B|C projection on a block that runs attention too
	// (RecurrentPlan.WithAttn), where Wq is the attention's query; empty
	// everywhere else, where Wq carries it.
	In Weight
}

// Head is the tail of a token: the output norm and the vocabulary projection.
// It runs in the same submission as the last block, so a token ends with one
// readback and the residual stream never comes home.
type Head struct {
	// Model is which model the head belongs to; see LayerPlan.Model.
	Model uint64
	Norm  []float32 // output_norm.weight
	// NormB is the output LayerNorm's bias (C6), non-nil exactly when the
	// model's norms are LayerNorms (LayerPlan.LayerNorm) -- zeros where the
	// file carries none.
	NormB []float32
	W     Weight // the lm_head, which is token_embd when the model ties them
	// Logits is the destination, NVocab wide. Prefer ArgmaxOnly, which keeps
	// the logits on the device and reads back one token id.
	Logits []float32
	// Softcap is the model's final logit softcap (gemma2), x = c*tanh(x/c)
	// with c this value; 0 is none. A device that accepts the head applies it
	// after the projection and before both the readback and the argmax, so
	// Logits come home capped and the caller must not cap them again.
	Softcap float32
	// Embeds says W is also the model's token embedding (a tied model), so a
	// tier holding the head can look prompt tokens up itself -- see
	// EmbedDevice. EmbdScale multiplies a looked-up row (gemma's sqrt(n_embd));
	// 0 means 1.
	Embeds    bool
	EmbdScale float32
	// ArgmaxOnly asks for the next greedy token alone. A tier that can serves
	// it on the device and sets Token, leaving Logits untouched; one that
	// cannot fills Logits as always and leaves Token as the caller set it (-1).
	// See model.State.ForwardGreedy.
	ArgmaxOnly bool
	Token      int32
	// SampleK, when positive, asks for the sampler's candidates in place of
	// the logits: the SampleK best of the row after the repeat penalty in
	// SampleArgs ([count, the penalty's bits, the history's ids], the layout
	// of kernels.SampleArgsWords), best first and the lower id on a tie --
	// exactly the host sampler's top-k (model.Sampler). A tier that serves it
	// fills SampleVals and SampleIDs, leaves Logits untouched and sets
	// Sampled; one that cannot fills Logits as always. See
	// model.State.ForwardSample.
	SampleK    int
	SampleArgs []uint32
	SampleVals []float32
	SampleIDs  []uint32
	Sampled    bool
	// Pick, when non-empty, asks for the logits at these vocabulary ids alone
	// in place of the row: a decision readout's label ids. A tier that serves
	// it fills PickVals (as long as Pick at least), leaves Logits untouched
	// and sets Picked; one that cannot fills Logits as always. The values are
	// the projection's own, capped where the head caps (Softcap): a head bias
	// and a logit scale are the caller's, which is why a caller with either
	// does not ask. See model.Decider.
	Pick     []int32
	PickVals []float32
	Picked   bool
	// Tokens is RowsDevice.LayersRows' output: each row's greedy token.
	Tokens []int32
	// RowLogits asks LayersRows to read every row's logits back as well, into
	// Logits, row-major and NVocab wide per row, for a caller that samples.
	// They are the projection's raw output: a head bias and a logit scale are
	// the caller's to apply (finishLogits), as for one sequence.
	//
	// On a Layers call of several rows (a prompt chunk) it asks for every
	// row's projection rather than the last row's alone: a speculative
	// verification needs the logits at each position it checks. A tier that
	// cannot refuses the call.
	RowLogits bool
	// Hidden, when non-nil, receives the head's input as well: the output
	// norm's result for each row the head projects (the last row of a folded
	// chunk, every row of a RowLogits call), NEmbd wide per row. It is what a
	// multi-token-prediction block reads of the trunk, and what it chains to
	// itself.
	Hidden []float32
	// LogitRows, when positive, says only the first LogitRows rows of a
	// LayersRows step want the head: their logits alone are read back (and
	// their tokens alone set), and a tier may project only them. A prompt chunk
	// riding in a decode step needs its last row's logits at most, and the
	// rest of its rows would otherwise each cost a vocabulary of readback.
	// Zero is every row.
	LogitRows int
}

// WantedRows is how many of a step's n rows want the head (LogitRows).
func (h *Head) WantedRows(n int) int {
	if h.LogitRows > 0 {
		return min(h.LogitRows, n)
	}
	return n
}

// MLA reports whether this block runs multi-head latent attention.
func (p *LayerPlan) MLA() bool { return p.KVLoraRank > 0 }

// RopeW is how many floats one row's rotary table holds: NRot, or two tables
// of NRot under RopeSplit.
func (p *LayerPlan) RopeW() int {
	if p.RopeSplit {
		return 2 * p.NRot
	}
	return p.NRot
}

// ResidW is one row's residual width in Layers' x: every AltUp stream, or
// every DeepSeek V4 hyper-connection stream.
func (p *LayerPlan) ResidW() int { return p.NEmbd * p.Streams() }

// Streams is how many residual streams Layers' x carries, stream-major: Gemma
// 3n's AltUp or DeepSeek V4's hyper-connections, one elsewhere.
func (p *LayerPlan) Streams() int {
	if p.DS4 != nil {
		return max(p.DS4.HCMult, 1)
	}
	if p.K3 != nil {
		return max(p.K3.Streams, 1)
	}
	if p.Q4E != nil {
		return max(p.Q4E.HCMult, 1)
	}
	return max(p.AltUp, 1)
}

// KVRow is one position's width in the attention cache: the latent plus the
// shared rotary key under MLA, and NKVHead heads of HeadDim otherwise. Every
// cache allocation, stride and migration must go through it.
func (p *LayerPlan) KVRow() int {
	if p.DS4 != nil {
		return p.DS4.RowAt(p.Comp, p.HeadDim)
	}
	if p.MLA() {
		return p.KVLoraRank + p.NRot + p.IdxHeadDim
	}
	if p.MSA() || p.QSA() {
		return (p.NKVHead + 1) * p.HeadDim
	}
	return p.NKVHead * p.HeadDim
}

// QSA reports Qwen3.8-Flash-Next's block selection on this block.
func (p *LayerPlan) QSA() bool { return p.Q4E != nil && p.Q4E.QSARatio != 0 }

// MSA reports MiniMax Sparse Attention's block selection on this block.
func (p *LayerPlan) MSA() bool { return p.IdxBlock != 0 && p.IdxHeads != 0 }

// Scale is the attention score multiplier this block runs.
func (p *LayerPlan) Scale() float64 {
	if p.AttnScale != 0 {
		return p.AttnScale
	}
	return 1 / math.Sqrt(float64(p.HeadDim))
}

// RopeAt reports whether block li rotates q and k. Llama 4's global layers do
// not (NoPEGlobal), and neither does any block of a NoPosEnc model.
func (p *LayerPlan) RopeAt(li int) bool {
	if p.NoPosEnc {
		return false
	}
	return !(p.NoPEGlobal && p.SWALocal != nil && !p.SWALocal(li))
}

// KVWindowSlack is how far behind its window a windowed layer's history is
// kept, in positions, on the host and on every device: a page wholly below
// the window start of the slowest row's position minus this is released.
// It is what a rewind may go back without losing its window -- a speculation
// round rewinds by its rejected rows, at most 2*specMaxDraft+2 -- and the
// host trusts history the device returns only from the window start of
// (its last position - 1 - KVWindowSlack), which this margin guarantees.
const KVWindowSlack = 64

// LocalRope reports whether block li rotates by the local table (csSWA):
// the blocks SWALocal names when it is set -- ModernBERT's are all but every
// SWAPeriod'th from the first, which the period's own expression does not
// say -- and otherwise the SWAPeriod pattern's.
func (p *LayerPlan) LocalRope(li int) bool {
	if p.SWAPeriod <= 0 {
		return false
	}
	if p.SWALocal != nil {
		return p.SWALocal(li)
	}
	return li%p.SWAPeriod < p.SWAPeriod-1
}

// Window is block li's sliding window in keys, or 0 when it attends its whole
// causal history.
func (p *LayerPlan) Window(li int) int {
	if p.SWAWindow <= 0 || p.SWALocal == nil || !p.SWALocal(li) {
		return 0
	}
	return p.SWAWindow
}

// DS4Plan is DeepSeek V4's model-wide geometry (model.Config.DSV4), the same
// on every block of the model: the hyper-connections, the grouped output
// projection, the compressors' rates and the lightning indexer's shape.
// LayerPlan.Comp and HashExperts are what differs per block.
//
// A block's cached row (KVRow) is its key, then on a compressed block the
// compressor's kv and gate projections of that position (one head on HCA, two
// on CSA), then on CSA the indexer compressor's (two heads of IdxHeadDim); the
// row is one latent head that is also the value, as MLA's. A compressed block
// keeps a second history of entries, EntRow wide, one per CompRate positions.
type DS4Plan struct {
	HCMult, HCIters      int
	HCEps                float32
	OGroups, OLoraRank   int
	RateCSA, RateHCA     int
	IdxHeads, IdxHeadDim int
	IdxTopK              int
	ExpertSqrtSoftplus   bool
	// Window is the model's sliding window, which every block's rows are
	// read through (LayerPlan.SWAWindow may be zero where the context fits
	// inside it).
	Window int
	// Fault breaks one piece of the graph on purpose, set only by a gate;
	// the host honours the same faults (model's ds4Fault).
	Fault DS4Fault
}

// DS4Fault names a piece of DeepSeek V4's graph a gate takes out.
type DS4Fault uint8

const (
	DS4FaultNone DS4Fault = iota
	DS4FaultCombSoftmax
	DS4FaultCombRows
	DS4FaultNoDerope
	DS4FaultWindowOnly
	DS4FaultNoOverlap
	DS4FaultAllEntries
	DS4FaultMainRope
	DS4FaultScoreRoute
	DS4FaultNoSinks
	DS4FaultHeadMean
)

// K3Plan is Kimi-K3's residual attention and latent mixture at one block
// (model's k3.go is the graph). The residual is Streams streams, stream-major
// in Layers' x: stream 0 the running residual, stream 1+j checkpoint j. The
// attention reads the softmax mix of the first BankAttn checkpoints and the
// running residual, scored by LayerWeights.ResAttn; on a checkpoint block
// (Push, the stream it banks into, non-zero) the block's raw input is banked
// first and the running residual restarts from the attention's output; the
// FFN reads the mix of the first BankFFN and the running residual, scored by
// ResFFN. The head's mix is the host's. Latent is the routed experts' width
// (zero for NEmbd): the experts read RoutedDown of the FFN input, their sum
// is normed by RoutedNorm and RoutedUp takes it back, beside the shared
// experts at NEmbd.
type K3Plan struct {
	Streams           int
	BankAttn, BankFFN int
	Push              int
	Latent            int
	// Fault breaks one piece of the graph on purpose, set only by a gate;
	// the host honours the same faults (model's k3Fault).
	Fault K3Fault
}

// Q4EPlan is Qwen3.8-Flash-Next at one block (model's q4e.go is the graph).
// The residual is HCMult streams, stream-major in Layers' x; each sublayer
// reads their mix through a HCLowRank gate (LayerWeights.HCAttn, HCFFN) and
// is added into every stream at its placement weight. On a QSA block
// (QSARatio non-zero) the indexer's raw key is one more kv head of the
// cached row (KVRow, padded to HeadDim) and attention reads the plan's
// IdxTopK/QSARatio best blocks of QSARatio positions and the tail, the
// indexer being IdxHeads heads of IdxHeadDim. On the n-gram block
// (NGramHere; NGramWidth is the model's, on every block, so a scratch built
// for any block serves it) the gathered table rows of each row
// (NGramDevice.SetNGramRows) are projected, gated per stream and convolved
// over NGramConv taps NGramDil positions apart, the history the block's
// recurrent tail (RecurrentPlan.TailLen). The head's mix is the host's.
type Q4EPlan struct {
	HCMult, HCLowRank int
	QSARatio          int
	NGramWidth        int
	NGramConv         int
	NGramDil          int
	NGramHere         bool
	// QSARope is the rotary's folded planes at a scalar position (Rope.
	// TabPlanes), which a QSA block's pooled keys turn by at their blocks'
	// first positions: an M-RoPE model's text rows turn by the same table.
	QSARope []float32
	// Fault breaks one piece of the graph on purpose, set only by a gate;
	// the host honours the same faults (model's q4eFault).
	Fault Q4EFault
}

// Q4EFault names a piece of Qwen3.8-Flash-Next's graph a gate takes out, in
// the host's numbering (model's q4eFault).
type Q4EFault uint8

// The host's violations a device runs too: the mix by the plain mean, the
// gate's low rank unscaled, every stream placed at one, the placement
// halved, no n-gram embedding, no n-gram convolution, every key read on a
// QSA block, a QSA block scored by its first key, the tail dropped, the
// delta rule's gate silu.
const (
	Q4EFaultNone Q4EFault = iota
	Q4EFaultNoGate
	Q4EFaultPlainPlace
	Q4EFaultNoNGram
	Q4EFaultNoNGramConv
	Q4EFaultNoEOSReset
	Q4EFaultDenseQSA
	Q4EFaultQSAFirstKey
	Q4EFaultNoTail
	Q4EFaultSiLUGate
	Q4EFaultUnscaledGate
	Q4EFaultHalfPlace
)

// NGramDevice is a device that runs Qwen3.8-Flash-Next's n-gram block: the
// host hashes each row's n-gram, reads the table rows and hands them over,
// rows*Q4EPlan.NGramWidth floats in the call's row order, before the call.
type NGramDevice interface {
	SetNGramRows(rows []float32)
}

// QSADevice is a device that runs Qwen3.8-Flash-Next's QSA blocks: before a
// call the host hands over each row's rotary marks (where a picture moved the
// positions after it, and by how much), in the call's row order: the row's
// count, then its (position, delta) pairs.
type QSADevice interface {
	SetRopeMarks(marks []int32)
}

// K3Fault names a piece of Kimi-K3's graph a gate takes out.
type K3Fault uint8

const (
	K3FaultNone K3Fault = iota
	K3FaultNoBank
	K3FaultNoRestart
	K3FaultRawScores
	K3FaultNoLatentNorm
	K3FaultNoMLAGate
	K3FaultNoKDAGate
)

// ExpWidth is the routed experts' input and output width: the latent on a
// Kimi-K3 mixture, NEmbd everywhere else.
func (p *LayerPlan) ExpWidth() int {
	if p.K3 != nil && p.K3.Latent != 0 {
		return p.K3.Latent
	}
	return p.NEmbd
}

// DS4 compression kinds, LayerPlan.Comp: jlm.CompKind's values.
const (
	DS4CompNone = 0
	DS4CompCSA  = 1
	DS4CompHCA  = 2
)

// RateAt is how many positions one entry of a comp block folds.
func (d *DS4Plan) RateAt(comp int) int {
	switch comp {
	case DS4CompCSA:
		return d.RateCSA
	case DS4CompHCA:
		return d.RateHCA
	}
	return 0
}

// RowAt is a comp block's cached row at head width hd.
func (d *DS4Plan) RowAt(comp, hd int) int {
	switch comp {
	case DS4CompHCA:
		return 3 * hd
	case DS4CompCSA:
		return 5*hd + 4*d.IdxHeadDim
	}
	return hd
}

// EntRowAt is one entry of a comp block: the compressed key, and on CSA the
// indexer's after it. Zero on a block with no compressor.
func (d *DS4Plan) EntRowAt(comp, hd int) int {
	switch comp {
	case DS4CompHCA:
		return hd
	case DS4CompCSA:
		return hd + d.IdxHeadDim
	}
	return 0
}

// LayerPlan is everything about a block that is not in its weights.
//
// Most flags below describe a graph feature that runs and produces fluent,
// wrong text when ignored, so a tier that cannot apply one must decline the
// block by name (see RULE 8a in AGENTS.md).
type LayerPlan struct {
	// Graph is the block's operation graph (engine/graph) when the plan was
	// lowered from one: the device program (tier's devprog.go) is built from
	// its ops. Nil for a plan nothing lowered.
	Graph *graph.Graph
	// KPad is the width the block's matrices that read the residual or the
	// attention's output were packed at, when NEmbd is not a whole number of
	// 32-element blocks (a tower 144 wide, packed at 160 with zero columns
	// past 144); their Weight.K is KPad, and the device quantizes those
	// activations as rows of KPad with zero tails (kernels.QuantizePadded),
	// as the host pads them. Zero everywhere else.
	KPad                                               int
	NEmbd, NHead, NKVHead, HeadDim, NRot, NFFN, MaxSeq int
	RMSEps, RopeBase                                   float64

	// Model is which model the block belongs to (ModelOwner): a device keys
	// its blocks by index, so it must not take a block of a second model
	// while the first's are there.
	Model uint64
	// Vocab is the output projection's rows. A device that reserves a ragged
	// step's per-row logits at placement sizes them from it, because the
	// head is offered after the blocks; 0 leaves them to the first step.
	Vocab int
	// FinalSoftcap is the head's final logit softcap (Head.Softcap), which a
	// device reserving those per-row logits reserves their capped twin for.
	FinalSoftcap float32
	// SWAPeriod is the repeating local/global layer pattern, 0 when the
	// architecture has one kind of layer. Layer il is local when
	// il%SWAPeriod < SWAPeriod-1 (model.Config.SWA's expression), or where
	// SWALocal is set when it says so (LocalRope). It picks
	// between the two rotary tables Layers is handed, because gemma3 trains its
	// local and global layers at different rotary bases.
	SWAPeriod int
	// RopeTab and RopeTabSWA are the folded rotary constant planes a device
	// builds its own table from (Rope.TabPlanes of the global and the local
	// configuration), so both tiers derive the table from the same constants.
	// nil for a block with no rotary, and RopeTabSWA is nil wherever SWAPeriod
	// is zero. A tier may ignore them and upload the finished cs/csSWA tables
	// instead; see tier.Config.RopeTableHost.
	RopeTab, RopeTabSWA []float32
	// RopeTabScalar is the rotary's folded planes at a scalar position
	// (Rope.TabPlanes), whatever table the call's rows turn by: what a
	// block selection's pooled keys turn by at their blocks' first
	// positions. Nil where nothing reads it.
	RopeTabScalar []float32
	// RopeSplit is XD-RoPE (HunyuanVL): each NEOX pair's two elements turn by
	// tables of their own, so a row's table is two of NRot floats (A then B;
	// RopeW) and the rotation does not keep a head's RMS -- k's norm follows
	// it, as q's does.
	RopeSplit bool
	// SWAWindow is how many keys a sliding-window layer attends to, and
	// SWALocal says which blocks those are: the blocks the SWAPeriod does not
	// leave global (model.Config.periodLocal), which are also the ones that
	// rotate under NoPEGlobal -- with SWAWindow 0 (SmolLM3) that is all they
	// mean. Both zero/nil for an architecture whose every layer sees its whole
	// history.
	// Separate from SWAPeriod, which is zero for single-base windowed models
	// such as gemma2 and phi3. See Window.
	SWAWindow int
	SWALocal  func(li int) bool

	// PostNorm says the block RMSNorms the attention output and the FFN output
	// before each residual add -- gemma2 and gemma3's two extra norms.
	PostNorm bool
	// SubNorm says the block RMSNorms the attention output before wo and the
	// gated FFN's product before down (LayerWeights.AttnSubNorm/FFNSubNorm) --
	// BitNet's two inner norms. A tier runs both or declines the block.
	SubNorm bool
	// PostResidNorm says the block's norms come after its residual adds, on
	// the residual itself (BERT's post-norm order): x = LN(x + attn(x)) with
	// FFNNorm, whose output is both the FFN's input and the residual, then
	// x = LN(x + mlp(x)) with LayerWeights.ResidNorm. The attention reads the
	// block input as it is (no AttnNorm; NoPreNorm). Built on the non-causal
	// path, the one encoders run.
	PostResidNorm bool
	// NoPreNorm says some block has no pre-norms (OLMo 2, EXAONE 4): attention
	// and the FFN read the residual as it is and only their outputs are
	// normalised (LayerWeights.AttnNorm and FFNNorm are nil). A tier copies the
	// row where the norm would have written it.
	NoPreNorm bool
	// AttnSoftcap is x = c*tanh(x/c) on the attention scores, gemma2's 50.0 and
	// zero everywhere else.
	AttnSoftcap float32
	// ActWin is how many activation elements share one quantization scale. It
	// must match what the CPU tier uses, because the gate between them is
	// identical token ids; see cpu.QuantizeQ8Window.
	ActWin int
	// Act is the feed-forward's non-linearity: SiLU (llama), GELU-tanh (gemma,
	// SmolVLM's tower) or quick-GELU (CLIP's). A device that cannot emit the
	// kind it is handed must decline.
	Act      ActKind
	RopeNeox bool // rotate (i, i+nRot/2), not (2i, 2i+1)
	// NoPosEnc: the attention blocks apply no positional encoding at all. On
	// MLA the rotary launches are replaced by a copy rather than skipped,
	// because those kernels write the KV cache row's tail and the rotary half
	// of the absorbed query.
	NoPosEnc bool
	// QKNorm says each head of q and k is RMSNormed before RoPE (qwen3).
	QKNorm bool
	// QKNormWide makes that one norm over the whole q (and k) projection rather
	// than one per head -- olmoe's shape.
	QKNormWide bool
	// QKNormPost runs q's norm after the rotary and k's before it, k's weight
	// folded into q's at conversion (Hunyuan; jlm.FlagQKNormPostRope).
	QKNormPost bool
	// Gemma 4. GeomSplit says the model's sliding and global layers attend at
	// two geometries; NKVHead, HeadDim and NRot are then THIS block's (planFor
	// sets them per block) and HeadDimSWA/NKVHeadSWA/NRotSWA the sliding
	// layers', so a tier sizes for both. VNorm RMSNorms each head of v with no
	// weight before it is cached; VFromK says the block has no v projection
	// and v is k's (LayerWeights.Wv is empty); OutScale multiplies the block's
	// output row after its last residual add (zero means none).
	GeomSplit                       bool
	HeadDimSWA, NKVHeadSWA, NRotSWA int
	// Sliding says this block attends at the sliding layers' geometry (its
	// NKVHead/HeadDim/NRot are HeadDimSWA's) and rotates by the local table.
	Sliding       bool
	VNorm, VFromK bool
	OutScale      float32
	// DenseMoE is Gemma 4's mixture block (model.Config.DenseMoE): a dense MLP
	// (LayerWeights.ShGate/ShUp/ShDown, ungated) beside the mixture, each with
	// its own norms, the router on its own norm and a per-expert weight factor.
	DenseMoE bool
	// Gemma 4's E2B/E4B: PLEDim is each block's per-layer embedding width
	// (zero for none), KVShared says the block attends to KVSource's history
	// and computes no k or v, and NFFN is then this block's own width.
	// PLEWidth is a row's whole per-layer input, NLayer*PLEDim.
	PLEDim, PLEWidth int
	// DeepSeek V3.2's lightning indexer (model.Config.Indexer): IdxHeads heads
	// of IdxHeadDim score every cached position, the key cached at the end of
	// the MLA row (KVRow), and attention reads the IdxTopK best. Zero heads is
	// none.
	IdxHeads, IdxHeadDim, IdxTopK int
	// MiniMax Sparse Attention (model.Config.MSA): with IdxBlock set the same
	// fields select blocks instead, one IdxHeadDim head per kv group: per
	// group the IdxTopK best blocks of IdxBlock positions, each block by its
	// best position, with the IdxLocal blocks ending at the query's own
	// forced in. The indexer's key is cached as one more kv head of the row
	// (KVRow). Set only on the blocks that select (past the dense lead).
	IdxBlock, IdxLocal int
	// Gemma 3n (model.Config.AltUp): AltUp residual streams, zero for one,
	// carried stream-major in Layers' x (ResidW); a gaussian top-k at
	// SparseStd deviations on the FFN gate of a block with Sparse set.
	AltUp     int
	Sparse    bool
	SparseStd float32
	// LaurelRank is LAuReL's middle width (Gemma 3n), zero elsewhere.
	LaurelRank int
	KVShared   bool
	KVSource   int
	// FFNWide says this block's NFFN is twice the model's (a KV-sharing
	// block under use_double_wide_mlp), which a tier that bakes the FFN width
	// into its scratch keeps apart.
	FFNWide bool
	// NoExpertNorm skips the division of the top-k mixture weights by their sum
	// (olmoe). Negated so the zero value renormalises, which is what every other
	// mixture here wants; see model.Config.
	NoExpertNorm bool
	// NExpert is 0 for a dense block. NExpertUsed is the top-k, and NFFNExp is
	// one expert's hidden width -- not NFFN, which for a MoE file is the dense
	// width the architecture would have used and is read by nothing here.
	NExpert, NExpertUsed, NFFNExp int
	// The router's gating shape; every zero value is the plain softmax top-k.
	//
	// ExpertSigmoid gates with sigma() instead of softmax. ExpertSelBias says a
	// per-expert correction bias moves the selection and not the weights (a
	// different tensor from MoEBias's router bias). ExpertGroups and
	// ExpertGroupsUsed are the grouped top-k, 0 or 1 group being ungrouped.
	// ExpertScale is routed_scaling_factor, where 0 and 1 both mean none.
	ExpertSigmoid                  bool
	ExpertSelBias                  bool
	ExpertGroups, ExpertGroupsUsed int
	ExpertScale                    float64
	// ExpertSparseMixer, when non-zero, is Phi-3.5-MoE's sparsemixer router
	// and its jitter_eps (MoEGate.SparseMixer): the plain logit rank keeps two
	// and the weights are its masked softmaxes, not renormalised.
	ExpertSparseMixer float32
	// NFFNShExp is the shared expert's hidden width, 0 when the model has none.
	// It runs for every token beside the routed ones and its gate is a vector
	// rather than a router.
	NFFNShExp int
	// NonCausal says the block's rows attend over the key runs of the call
	// that carries them, not over a history: every row of the call by
	// default (a vision tower's picture), or the windowed runs on a Windowed
	// block. Its keys live one call deep -- written, read by the block's own
	// attention, dead before the next block -- so it keeps no history, and a
	// call is one whole run, never a chunk of one. A vision segment's blocks
	// are this; their norms, MLP and rotary are the text kit's fields
	// (LayerNorm, UngatedFFN, NRot over Layers' cs or NoPosEnc).
	NonCausal bool
	// Windowed says a NonCausal block attends inside the windowed key runs
	// the call was handed (KeyRunDevice) rather than over every row
	// (Qwen2.5-VL's tower, every block but each n'th).
	Windowed bool
	// HeadSet says a NonCausal block is a decision head's that attends and
	// norms otherwise than the encoder before it (d1-omni: LayerNorm and 16
	// key heads after an RMSNorm trunk of 8): a device runs it in a scratch
	// set of its own rather than declining it for the trunk's geometry.
	HeadSet bool
	// AttnOutGate says the q projection is double width and its second half,
	// interleaved per head, gates the attention output through a sigmoid.
	AttnOutGate bool
	// AttnSinks says each head carries a learned sink logit that joins its
	// softmax denominator (gpt-oss).
	AttnSinks bool
	// MoEBias says the router and the experts' gate, up and down carry biases
	// (gpt-oss).
	MoEBias bool
	// KVLoraRank enables multi-head latent attention; 0 is every other
	// architecture, and a tier without MLA declines on this field alone.
	//
	// The cache holds one row per position for the whole layer,
	// [latent (KVLoraRank) | rotary key (NRot)], shared by every head; KVRow is
	// that width. The value is the row's leading KVLoraRank floats, so a tier
	// must not store K and V apart.
	//
	// HeadDimV is the value head's width, which is not HeadDim (192 and 128 on
	// DeepSeek-V3). QLoraRank is the query latent, 0 where the query is one
	// step.
	KVLoraRank, QLoraRank, HeadDimV int
	// AttnScale is the score multiplier, and 0 means the ordinary
	// 1/sqrt(HeadDim). It is carried because YaRN's magnitude correction reaches
	// the score (DeepSeek), and because MLA takes its scores over the
	// KVRow-wide absorbed query, where 1/sqrt(KVRow) is not the model's scale.
	AttnScale float64
	// Llama 4's attention.
	//
	// SWAChunked makes a windowed layer attend within an aligned chunk of
	// SWAWindow keys -- floor(pos/W)*W through pos -- rather than the last W.
	// NoPEGlobal takes the rotary off every layer SWALocal leaves global.
	// QKL2Norm RMSNorms each head of q and k with no weight on the layers that
	// rotate. AttnTemp* scale the query of an unrotated layer by
	// 1 + s*ln(1 + floor((pos+o)/f)).
	SWAChunked, NoPEGlobal, QKL2Norm             bool
	AttnTempScale, AttnTempFloor, AttnTempOffset float32
	// ResidualScale multiplies each block's attention and FFN output before its
	// residual add (Granite's residual_scale); 0 and 1 both mean none. On a
	// mixture block it multiplies the whole FFN output, the routed sum and the
	// shared expert together, where the host folds it into the routed weights:
	// ExpertScale does not carry it.
	ResidualScale float64
	// The classic block (C6). LayerNorm makes the block's norms mean-centred,
	// with LayerWeights.AttnNormB/FFNNormB as their biases when non-nil.
	// Parallel feeds attention and the FFN the same block input (the FFN norm
	// when FFNNorm is non-nil, the attention's normed row otherwise) and adds
	// both into the residual. UngatedFFN is up, its bias BUp, the activation
	// alone, down and BDown -- Gate is empty.
	LayerNorm, Parallel, UngatedFFN bool
	// ClampKQV clamps q, k and v to [-ClampKQV, ClampKQV] straight after their
	// projections (DBRX); zero means no clamp.
	ClampKQV float32
	// BidirOff says this block's attention ignores the bidirectional runs
	// (KeyRunDevice's full runs) and stays causal: Gemma 4's global layers,
	// on a model whose image rows see each other on the sliding layers only.
	BidirOff bool
	// Clamps says every matrix's input and output is clamped to bounds the
	// block carries (LayerWeights.Clamp): Gemma 4's clipped linears.
	Clamps bool
	// Conv is a convolutional tower's block (Gemma 3n's MobileNet-V5), nil
	// for every other: its rows are positions of a grid whose shape is the
	// block's own, and its body is convolutions, not a transformer block's.
	// Such a block runs through ConvDevice.ConvLayers, not Layers.
	Conv *ConvPlan
	// ExpertWeightIn applies a mixture's routed weight to the expert's input
	// (Llama 4), which with a SiLU expert is a different function from the
	// output-weighted sum every other mixture runs.
	ExpertWeightIn bool
	// DS4 is DeepSeek V4's model-wide geometry, nil on every other model;
	// Comp is this block's compressor (DS4CompNone, CSA or HCA) and
	// HashExperts says it routes by the token's row of a frozen table
	// (LayerWeights.HashExperts, the ids a TokenDevice is handed).
	DS4         *DS4Plan
	Comp        int
	HashExperts bool
	// K3 is Kimi-K3's residual attention and latent mixture at this block,
	// nil on every other model.
	K3 *K3Plan
	// Q4E is Qwen3.8-Flash-Next's hyper-connections, QSA and n-gram
	// embedding at this block, nil on every other model.
	Q4E *Q4EPlan
	// LatentNormEps is the MLA latent norms' epsilon (Kimi-K3 and
	// Kimi-Linear's 1e-6); zero means RMSEps.
	LatentNormEps float64
	// Recurrent describes a linear-attention block, and it is non-zero only for
	// one. Such a block has no q, k, v, o or KV cache at all -- see
	// LayerWeights.SSM. It is offered to the device like any other block so the
	// decision stays on the device's side of the seam.
	Recurrent RecurrentPlan
	// NoFFN is a block that is its mixer alone (plain Mamba-2, a Nemotron-H
	// mixer layer): it ends at the first residual add, and its FFN weights,
	// norm and kernels are absent.
	NoFFN bool
	// Graphed says the model built this block's graph (docs/design/composer.md):
	// a site a rewrite names is offered only where Rewrites holds a match
	// whose Offer passes at the tier's own Target. Without a graph a tier
	// decides a site from its own facts.
	Graphed bool
	// Rewrites are the rewrites matched on the block's graph, legal or not;
	// the tier asks each one's Offer with the Target it runs at. A block
	// that is not Graphed may still carry a sublayer's (a gated delta rule's,
	// graph.Delta): a rewrite found there is offered by it, and a site no
	// rewrite in it names is decided from the tier's own facts.
	Rewrites []graph.Offered
}

// RecurrentPlan is a linear-attention block's geometry. Every field is zero for
// an ordinary attention block, and Conv == 0 IS the test for one.
type RecurrentPlan struct {
	Conv      int // causal convolution width, in positions
	Chans     int // convolved channels: q + k + v
	KHeads    int // key/query heads
	VHeads    int // value heads, and the number of decay rates
	KDim      int // key/query head width, and the state's key extent
	VDim      int // value head width, and the state's value extent
	StateLen  int // VHeads*VDim*KDim floats and TailLen more, per sequence, CONSTANT in context
	ConvState int // (Conv-1)*Chans floats, per sequence
	// ChanDecay is Kimi Delta Attention rather than qwen3next's gated delta
	// rule: the state decays at a rate per channel instead of one per head,
	// both gates come off a low-rank bottleneck, and the output gate is a
	// sigmoid where qwen3next's is a SiLU. The geometry above is identical for
	// both, so no shape field can say it.
	ChanDecay bool
	// KeyTiled pairs value head vh with key head vh % KHeads instead of
	// vh / (VHeads/KHeads): the order llama.cpp's qwen35 converter writes
	// the value heads in (jlm.FlagDeltaKeyTiled). Like ChanDecay, no shape
	// field can say it -- both orders have identical geometry.
	KeyTiled bool
	// DecayBound is Kimi-K3's gate_lower_bound on a KDA block: the decay is
	// exp(DecayBound * sigma(-A*(alpha+dt))) in place of exp(A*softplus(.)),
	// A being -exp(A_log). Zero is the softplus decay.
	DecayBound float32
	// SSD is Mamba-2's selective update in place of the delta rule
	// (jlm.LayerSSD): the convolved channels are C | B | x (as q | k | v), the
	// convolution has a bias (SSMWeights.ConvBias), the update writes x*dt
	// along B with no key dot, D*x is added to the output, and the gated norm
	// is RMSNorm(y*silu(z)) over NormGroups groups of the whole VHeads*VDim
	// with a weight that wide -- or norm(y)*silu(z) where NormBeforeGate, or
	// y*silu(z) alone where NoNorm.
	SSD bool
	// WithAttn: the block runs softmax attention beside the recurrence, on
	// the same normed input, and the two outputs are summed before the
	// residual (Falcon-H1, jlm.LayerSSDAttn). The plan's attention shape is
	// the attention's, and the weights carry the mixer's projection in
	// SSMWeights.In.
	WithAttn bool
	// ShortConv is LFM2's gated short convolution (jlm.LayerShortConv): no
	// heads and no state beside the convolution's window over Chans channels.
	// Wq is x, SSMWeights.Gate is B and SSMWeights.BA is C, and the block is
	// out_proj(C * conv(B * x)).
	ShortConv bool
	// Centred is a ShortConv over a whole non-causal call (LFM2's encoder,
	// d1-omni's trunk): three taps on the rows before, at and after each
	// row, zero past either end, and no state. A tap reads a neighbour only
	// where the row's window holds it (nn.RowWindowDevice), so a media
	// prefix's last row does not read the text's first.
	Centred                bool
	NormGroups             int
	NormBeforeGate, NoNorm bool
	// SigmoidGate gates the output norm by sigmoid(z) where the gated delta
	// rule's is silu(z) (Qwen3.8-Flash-Next).
	SigmoidGate bool
	// TailLen is what the block's state carries per sequence after the delta
	// rule's, inside StateLen: room for Qwen3.8-Flash-Next's n-gram
	// predecessors and convolution history (model's recTailLen), the same on
	// every linear block so a device keeps one slot width. A sequence's slot
	// is [delta rule | tail]; the delta rule's kernels step its head.
	TailLen int
	// Mamba1 is Mamba-1's selective scan (jlm.LayerMamba1): VHeads channels
	// of one row each over a KDim-wide state, B and C shared, the decay a
	// vector exp(dt*A) per channel, dt off a rank-Rank bottleneck of the
	// convolved x. Wq is x, SSMWeights.Gate is z.
	Mamba1 bool
	Rank   int
}

// LayerDevice runs a whole transformer block without returning the residual
// stream to the host between its matvecs.
//
// The unit is a block because a per-matvec seam costs a blocking round trip per
// matvec, and because it keeps the graph in model/: the device knows nothing
// beyond one repeating unit (no embedding lookup, sampling or tied weights).
// PrepLayer is per block and may decline, so a model that does not fit runs
// its first N blocks on the device and the rest on the CPU with no separate
// mechanism for the seam.
type LayerDevice interface {
	Device

	// PrepLayer uploads block li's weights and scratch. It returns false if the
	// block will not fit, or a quant type has no kernel, in which case that
	// block and every later one runs on the CPU.
	PrepLayer(li int, p *LayerPlan, w *LayerWeights) bool

	// Layers runs blocks [lo, hi) for n consecutive positions starting at pos,
	// reading and writing n rows of the residual stream in x. n > 1 is the
	// prefill chunk; a device that cannot take it must return false and let
	// the host run the chunk. cs is the cos/sin table for those positions,
	// interleaved, NRot entries per position. The whole range is one
	// submission: x crosses the bus once each way.
	//
	// head, when non-nil, continues that same submission with the output norm
	// and the vocabulary projection, and fills head.Logits instead of reading x
	// back. It is only offered when the device took every block, since anything
	// left on the CPU needs the residual stream home anyway.
	// csSWA is the local layers' rotary table where SWAPeriod is set, and nil
	// for every architecture with a single base. A tier that is handed nil and
	// has a local block to run must decline rather than substitute cs.
	Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *Head) bool

	// PrewarmLayer packs block li's weights off the main loop, so that a later
	// PrepLayer only has to upload them. It must not touch the device and must
	// be safe to call from another goroutine while Layers is running, since
	// packing is most of preparing a block and is the only part that can
	// overlap a token.
	//
	// False means it did not stage anything: already resident, already staged,
	// or over the staging budget. None of those is an error; the work simply
	// happens on the main loop as before.
	PrewarmLayer(li int, p *LayerPlan, w *LayerWeights) bool

	// MigrateKV moves block li's attention history between host and device, so
	// that a block which changes sides keeps its keys and values. Without it a
	// migrated block attends over the cache it arrived at, which is fluent and
	// wrong rather than an error. pos is how many positions are live; k and v
	// are the host arrays, [pos*kvDim + i] float32. A batch's rows travel in
	// RowsDevice's layout, row r at positions r*maxSeq.., so pos is then the
	// end of the last row with history.
	MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool

	// MigrateRec is MigrateKV for a linear-attention block's recurrent summary,
	// which lives in the device's buffers while the block runs there. The host
	// copy is what -kv-cache seals and what a seam shrink restores.
	//
	// toDevice=false reads the state the device is currently reading (recPair
	// keeps two halves and flips per submission), so the answer is the summary
	// as of the last completed token.
	//
	// A batch passes every row's state back to back (len(conv) a multiple of
	// one row's), which is also the device's per-row layout; toDevice grows
	// the device's rows to match.
	MigrateRec(li int, conv, state []float32, toDevice bool) bool

	// ReserveKV makes the device able to hold pos positions of history before
	// any of it is written, and reports whether it can. A restored prompt
	// prefix arrives before any submission has grown the cache. It is separate
	// from MigrateKV because growth drops the captured graph and rebuilds the
	// scratch, which the device cannot do from inside a call that already
	// holds its lock: ask first, then move.
	ReserveKV(pos int) bool

	// ReleaseLayers hands blocks [lo, hi) back to the host and frees whatever
	// they held on the device.
	ReleaseLayers(lo, hi int)

	// PrepHead uploads the output norm and the vocabulary projection. False
	// means Layers must be called with a nil head.
	PrepHead(h *Head) bool

	// Reserve sizes the per-call scratch buffers once, for the largest matvec
	// the model contains, so that nothing grows on the serving path. The model
	// knows every shape from the header; the device sees one matvec at a time.
	//
	// Optional: a caller that never calls it still works, and so does a shape
	// past what was reserved.
	Reserve(maxRows, maxK int) bool
}

// EmbedDevice is an optional LayerDevice that fills residual rows from token
// ids on the card, out of the head it holds when that head is also the token
// embedding (Head.Embeds): dst's rows become EmbdScale * embedding[id]. True
// is a promise, not a write: the device gathers the rows into the next Layers
// call whose x is dst, without the rows crossing the bus, and makes them real
// in dst first if anything else reaches it. So the caller hands dst to Layers
// next and does not read it before. False means nothing was promised and the
// caller looks the rows up itself.
//
// It exists because a host lookup in the packed layout is a cache miss per
// word: one row's words sit a vocabulary's width apart.
type EmbedDevice interface {
	EmbedRows(ids []int32, dst []float32) bool
}

// KeyRun is a run of rows [Lo, Hi) that attend to each other in both
// directions. Every mask this engine runs is, per row, an interval of keys:
// causal is [0, p+1); a row inside a full run of a causal prompt (a Gemma 3
// image's tokens) sees keys up to the run's end, everything before the run
// staying causal and a sliding window still bounding the past from the row's
// own position; a row of a NonCausal call sees its whole call, or on a
// Windowed block its window's run.
type KeyRun struct{ Lo, Hi int }

type ConvKind uint8

const (
	// ConvEdge is an edge residual: a full 3x3 convolution at Stride, a
	// norm and GELU, a pointwise convolution and a norm.
	ConvEdge ConvKind = iota + 1
	// ConvInverted is a universal inverted residual: an optional depthwise
	// start and its norm, a pointwise expansion, a norm and GELU, an
	// optional depthwise middle with its norm and GELU, a pointwise
	// projection and its norm.
	ConvInverted
	// ConvAttention is multi-query attention: a norm, Heads query heads of
	// KD and one key and one value head of KD, each the normed input or a
	// stride-2 depthwise downsample of it with a norm, scores at
	// 1/sqrt(KD), then the output projection.
	ConvAttention
	// ConvConformer is a FastConformer block (d1-omni's audio tower): half a
	// SiLU feed-forward, Transformer-XL relative-position attention with
	// per-head u and v biases, the convolution module (LayerNorm, a GLU, a
	// depthwise convolution of KMid taps centred on the row, a folded
	// BatchNorm, SiLU, pointwise), the other half feed-forward and a
	// LayerNorm, every norm a LayerNorm with a bias and every matrix with a
	// bias. Rows are positions: HIn is the most a call takes, and a call of
	// fewer rows reads only its own (the keys and the convolution stop at the
	// call's rows). CIn is the width, CExp the feed-forward's, Heads of KD.
	// The matrices ride the slots: Wq, Wk, Wv, Wo, Gate (linear_pos), Up and
	// Down (the first feed-forward), ShGate (the GLU's value), ShUp and
	// ShDown (the second feed-forward), SSM.Gate (the GLU's gate) and SSM.BA
	// (the pointwise convolution).
	ConvConformer
)

// ConvPlan is one convolutional block's geometry at its picture's size. Rows
// are positions, channel-last; every convolution pads TF SAME (the smaller
// half before the first sample), and every weight is tap-major (ky, kx, c).
type ConvPlan struct {
	Kind             ConvKind
	HIn, WIn, CIn    int
	HOut, WOut, COut int
	// CExp is the widest intermediate: an edge or inverted residual's
	// expansion.
	CExp int
	// Stride is an edge residual's full convolution's.
	Stride int
	// The inverted residual's depthwise filters: start (KStart 0 when it has
	// none) over CIn writing HMid x WMid, middle (KMid 0) over CExp.
	KStart, SStart, HMid, WMid int
	KMid, SMid                 int
	// Attention: Heads of KD, and the key/value downsample's filter side
	// (KVK 0 when there is none) and the grid it writes.
	Heads, KD     int
	KVK, HKV, WKV int
	// Residual adds the block's input to its output.
	Residual bool
	// Eps is the norms' epsilon, and ActWin the activation quantizer's amax
	// window, which must be the host's.
	Eps    float64
	ActWin int
}

// ConvWeights is a convolutional block's vectors: the norms after its two
// matrices (an attention block's Norm1 is its input norm), and its depthwise
// filters with theirs, each filter K*K rows of channels.
type ConvWeights struct {
	Norm1, Norm2         []float32
	DwStart, DwStartNorm []float32
	DwMid, DwMidNorm     []float32
	KDown, KDownNorm     []float32
	VDown, VDownNorm     []float32
	// Conformer is a ConvConformer block's vectors, nil for every other.
	Conformer *ConformerVecs
}

// ConformerVecs is a FastConformer block's vectors: its norms and biases,
// the attention's u and v (heads x head width), the depthwise taps
// (tap-major), the BatchNorm folded into a scale and a shift (the depthwise
// bias folded into the shift), and PE, the relative positions' sinusoids
// from HIn-1 down to -(HIn-1): 2*HIn-1 rows of the width.
type ConformerVecs struct {
	FF1Norm, FF1NormB, FF1UpB, FF1DownB []float32
	AttnNorm, AttnNormB                 []float32
	QB, KB, VB, OB, U, V                []float32
	ConvNorm, ConvNormB                 []float32
	GLUValB, GLUGateB                   []float32
	DW, BNScale, BNShift, PW2B          []float32
	FF2Norm, FF2NormB, FF2UpB, FF2DownB []float32
	OutNorm, OutNormB                   []float32
	PE                                  []float32
}

// List is the vectors in a fixed order, the tier's.
func (c *ConformerVecs) List() [27][]float32 {
	return [27][]float32{c.FF1Norm, c.FF1NormB, c.FF1UpB, c.FF1DownB, c.AttnNorm, c.AttnNormB,
		c.QB, c.KB, c.VB, c.OB, c.U, c.V, c.ConvNorm, c.ConvNormB, c.GLUValB, c.GLUGateB,
		c.DW, c.BNScale, c.BNShift, c.PW2B, c.FF2Norm, c.FF2NormB, c.FF2UpB, c.FF2DownB,
		c.OutNorm, c.OutNormB, c.PE}
}

// ConvDevice is a LayerDevice that runs a convolutional tower's blocks.
// ConvLayers runs blocks [lo, hi) as one submission: x is block lo's input
// (its plan's HIn*WIn*CIn floats), out receives block hi-1's output, and when
// tap is a block of the range tapOut receives that block's output too (the
// fusion adapter's first input). False with nothing run is a device that does
// not hold the range.
type ConvDevice interface {
	ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool
}

// KeyRunDevice is an optional LayerDevice that takes the key runs of the
// Layers calls that follow, until replaced. full is the runs a causal call's
// rows widen to (positions of the sequence); windowed is the windows a
// NonCausal call's Windowed blocks attend inside ([Lo, Hi) row pairs that
// partition the call's rows). nil for both restores plain attention. A call
// whose rows cut a full run is refused, since no row of a run can be computed
// before every row's key is in the cache; false is a device that cannot take
// the runs, and the caller fails the call rather than running it unmasked.
type KeyRunDevice interface {
	SetKeyRuns(full, windowed []KeyRun) bool
}

// RowWindowDevice is an optional LayerDevice that takes one window a row for
// the NonCausal calls that follow, until replaced: row r attends inside
// wins[r] on its Windowed blocks. A symmetric sliding window (ModernBERT's
// local layers) is this; its rows' windows overlap, so it is not a partition
// KeyRunDevice can state. It replaces whatever runs SetKeyRuns gave, and
// false is a device that cannot take them.
type RowWindowDevice interface {
	SetRowWindows(wins []KeyRun) bool
}

// SeqRunsDevice is an optional LayerDevice that takes several sequences
// packed into the rows of the NonCausal calls that follow, until replaced:
// runs partition the rows, and every row of every non-causal block attends
// inside its run -- an encoder answering several questions in one pass. A
// windowed block's row windows (RowWindowDevice) are the caller's, inside
// the runs. nil or empty is one sequence over the call's rows. It is kept
// apart from SetRowWindows and SetKeyRuns, which it does not replace; false is
// a device that cannot take them.
type SeqRunsDevice interface {
	SetSeqRuns(runs []KeyRun) bool
}

// RowsReserver is an optional LayerDevice whose NonCausal blocks are built for
// the rows of the calls they run rather than for their plan's MaxSeq (the
// tower's largest grid): ReserveRows makes them able to take a call of n rows,
// moving what that costs from outside any call, and reports false -- having
// kept nothing it allocated -- when the device's budget cannot hold it. The
// caller moves blocks off the device until it can. A device that does not
// implement it is built for MaxSeq.
type RowsReserver interface {
	ReserveRows(n int) bool
}

// RowChunker is an optional RowsReserver whose non-causal set may take a
// picture in row chunks when the whole does not fit (paging is the design):
// TowerChunk, read after ReserveRows, is the rows a call takes (0: the
// picture goes whole). Such a picture runs block by block, every chunk's
// phase 1 (its k and v written at its rows of keys) before any chunk's phase
// 2 (its queries over every key, and the rest of the block on its rows).
type RowChunker interface {
	TowerChunk() int
	LayersPhase(lo, hi, pos, n, keys, phase int, x, cs, csSWA []float32) bool
}

// LayerInputDevice is an optional LayerDevice that takes a per-layer input
// for every row beside the residual: Gemma 4's per-layer embeddings
// (LayerPlan.PLEDim), NLayer*PLEDim floats a row, row-major, which block li
// reads its PLEDim slice of. The caller sets them before every Layers or
// LayersRows call whose rows they belong to; the device reads them during
// that call only.
type LayerInputDevice interface {
	SetLayerInputs(in []float32)
}

// EntDevice is an optional SeqKVDevice for DeepSeek V4 (LayerPlan.DS4):
// MigrateEnt moves block li's entries for the current session's sequence at
// base, as MigrateKVSeq moves its rows -- n of them, [n][EntRowAt] float32. A
// block's rows and its entries travel together.
type EntDevice interface {
	MigrateEnt(li, base int, ent []float32, n int, toDevice bool) bool
}

// TokenDevice is an optional LayerDevice that takes the token ids of the rows
// of the next Layers or LayersRows call: a block that routes by token
// (LayerPlan.HashExperts) reads its row of the table at the row's id. -1 is a
// row whose embedding was supplied, which such a block refuses.
type TokenDevice interface {
	SetTokenIDs(ids []int32)
}

// RowsDevice is an optional LayerDevice that runs ONE decode step for several
// independent sequences at once, so the weights are read once for all of them.
//
// It runs blocks [lo, hi), which may be a run of a split placement: the host or
// another device runs the blocks around it, and x carries the residual in and
// out. head is non-nil only when hi is the last block and this device holds
// the projection.
//
// Row r is a sequence at position pos[r] whose history occupies cache slots
// slot[r]-pos[r] .. slot[r]; the step writes its key and value at slot[r].
// seqLen is the slots one sequence spans: row r is sequence
// (slot[r]-pos[r])/seqLen of the session's batch, which is where a recurrent
// block keeps its state. x is len(pos) rows of the residual stream, cs one
// rotary table per row, and csSWA one of the local layers' per row where the
// architecture trains two rotary bases (LayerPlan.SWAPeriod; nil where it
// trains one). With a head, head.Tokens gets each row's greedy token,
// and with head.RowLogits set head.Logits gets each row's logits. Rows may
// belong to one sequence, as long as they are its next positions, each once
// and in any row order: a prompt laid out as rows at positions p..p+L-1 of the
// same slots is a causal prefill, because every row's key and value are
// written before any row attends and a recurrent block steps a sequence's
// rows in position order. False means nothing ran, and the device's LastErr
// says why.
type RowsDevice interface {
	LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *Head) bool
}

// RowsWidther is a RowsDevice that states how many rows one LayersRows call
// can take: RowsWidth(want) is the widest step of at most want rows whose
// batched scratch every device holding a block holds or can build in its
// budget now, 0 when not even the narrowest fits. A caller filling steps
// (model.Scheduler) keeps under it, since a step wider than the scratch is
// refused rather than split.
type RowsWidther interface {
	RowsWidth(want int) int
}

// RowsRoomer is a RowsDevice that says, before a step runs, whether every
// device holding a block has room for it: the batched scratch for rows rows
// and a recurrent state for each of seqs sequences, built or grown now. A step
// it has no room for is refused before anything runs, so a caller asks first
// and brings its blocks home rather than fail the step.
type RowsRoomer interface {
	RowsRoom(rows, seqs int) bool
}

// KVShareDevice is a device whose KV pages the prefix cache can name and map
// (jit_old/gpu/tier/kvshare.go): PublishKV names the session's page of block li
// by key, held by the session; MapKV puts the page key names into the
// session's page, the sequence then written to end at least, no byte moved,
// and is false when no page answers to key. A write into a named page copies
// it first.
type KVShareDevice interface {
	PublishKV(li, page int, key string) bool
	MapKV(li, page int, key string, end int) bool
	// PublishKVs is PublishKV for many names in one call: name i is block
	// lis[i]'s page pages[i] by keys[i]. False when any was refused.
	PublishKVs(lis, pages []int32, keys []string) bool
}

// RecShareDevice is a device that keeps a linear block's restore points where
// its state lives (jit_old/gpu/tier/recshare.go): SnapRec copies the session's
// state of block li into the snapshot named key, HasRec says one is kept, and
// LoadRec copies it into the session's state. Nothing crosses the bus.
type RecShareDevice interface {
	SnapRec(li int, key string) bool
	// SnapRecs is SnapRec for every block of lis at one point, in one call.
	SnapRecs(lis []int32, key string) bool
	HasRec(li int, key string) bool
	LoadRec(li int, key string) bool
}

// SeqKVDevice is a LayerDevice that keeps attention history per sequence --
// the device pages it (docs/design/device-kv-paging.md) -- rather than one
// cache laid out in RowsDevice's slots. A batch's rows then migrate one at a
// time, each by the base of its slots (row r's is r*maxSeq), and a batch needs
// no whole-context reservation: each row takes pages as it writes.
// PerSequenceKV is false on a device still running the contiguous cache.
type SeqKVDevice interface {
	PerSequenceKV() bool
	// MigrateKVSeq is MigrateKV for the sequence whose slots start at base:
	// k and v are that row's pos positions, [pos*kvDim + i] float32.
	MigrateKVSeq(li, base int, k, v []float32, pos int, toDevice bool) bool
	// ReserveKVSeqs gives each sequence whose slots start at bases[i] the
	// pages positions up to ends[i] need, on every device holding a block:
	// what a step is about to write, and nothing more. False is no room, and
	// the caller may move blocks and ask again.
	ReserveKVSeqs(bases, ends []int) bool
}

// NamedDevice is an optional LayerDevice made of several named devices (the
// names -devices gives them; see DeviceName). It places a block or the head on
// the device a placement names, says where each went, and pins a block where
// it is.
// KVPrewarmer is a device whose paged KV pool can grow ahead of the history
// written into it. PrewarmKV makes room for positions more positions in every
// placed block's pool where the device's budget has it free -- without paging
// weights out and without giving the pages to any sequence -- so the first
// prompt a State runs does not grow the pool inside its own time.
type KVPrewarmer interface {
	PrewarmKV(positions int)
}

// BlockMover is a NamedDevice that moves a placed block to another of its
// devices itself, the session's history copied from the source's pages into
// the target's without coming home (tier.GPU, move.go).
type BlockMover interface {
	// StageBlock prepares block li on the named device -- weights uploaded,
	// kernels built, room for positions of history -- without routing it
	// there. It is safe beside the
	// session's own steps on the block's current device, so a move's
	// preparation overlaps the tokens before it. False means not staged.
	StageBlock(li int, to string, p *LayerPlan, w *LayerWeights, positions int) bool
	// StageHead prepares the output projection on the named device the same
	// way, for a move that will leave none of the session's blocks beside it.
	StageHead(to string, h *Head) bool
	// Unstage frees what StageBlock prepared on blocks [lo, hi), and what
	// StageHead prepared, that no move committed.
	Unstage(lo, hi int)
	// MoveBlock routes block li to the named device with the session's
	// history at pos positions, between steps. Moved false with no error is
	// the target refusing the block, which stays where it was. An error that
	// is ErrNotDirect changed nothing: the caller moves the block itself. w
	// nil asks for the staged copy, and ErrNotStaged says there is none.
	MoveBlock(li int, to string, p *LayerPlan, w *LayerWeights, pos int) (BlockMoved, error)
	// RetireMoved frees what the MoveBlocks since the last call left on their
	// sources, after the caller has moved what it is moving.
	RetireMoved()
}

// ErrNotDirect is a BlockMover's answer for a move it cannot make itself --
// a device that keeps the contiguous cache, a batch's rows -- with nothing
// changed.
var ErrNotDirect = errors.New("the block's history cannot move between the devices directly")

// ErrNotStaged is a BlockMover's answer to a MoveBlock handed no weights for
// a block StageBlock did not prepare: the caller reads the block and asks
// again.
var ErrNotStaged = errors.New("the block was not staged on the target, and no weights came with the move")

// BlockMoved is what a BlockMover did with one block.
type BlockMoved struct {
	Moved bool
	// Path is how the history went: "peer" card to card, "staged" through a
	// host buffer, "host" through the device-independent row layout; "" when
	// the block had no history.
	Path string
	// Bytes is the history copied.
	Bytes uint64
}

type NamedDevice interface {
	PrepLayerOn(name string, li int, p *LayerPlan, w *LayerWeights) (bool, error)
	PrepHeadOn(name string, h *Head) (bool, error)
	// DeviceOf is the name of the device holding block li, and whether one does.
	DeviceOf(li int) (string, bool)
	HeadDeviceName() (string, bool)
	// Pin keeps block li on its device through seam moves, budget shrinks
	// and eviction; it reports false when no device holds the block.
	Pin(li int, on bool) bool
	// Stream marks block li's weights as streaming through a device's slots
	// rather than staying resident, before it is offered; false when no
	// device is attached.
	Stream(li int, on bool) bool
}

// The optional capabilities a LayerDevice may have, asked for by type
// assertion. One method each: test devices implement some and not others.

// ActWindower takes the activation amax window the session's scale choice
// uses (SetActWindow).
type ActWindower interface{ SetActWindow(n int) }

// CopyForgetter is a Device that keeps copies of the weights MatVec was handed,
// keyed on their address, and can be told that the bytes at an address range
// are no longer the ones it copied: a pager's frame refilled with another
// block, or memory given back. It drops every copy inside [p, p+n). A caller
// may offer a weight whose address is reused only to a Device it tells.
//
// Forget may be called on any goroutine, from inside a page-in the device
// itself started, and while no session is open; it must not block on the
// device's own work.
type CopyForgetter interface {
	Forget(p unsafe.Pointer, n uintptr)
}

// ModelOwner is a LayerDevice that holds one model's blocks at a time
// (LayerPlan.Model): a block of another model is refused while any of the
// first's blocks or its head is placed. ReleaseModel gives back everything
// model placed -- every block, every session's history on them and the head --
// so the device may take another; it does nothing for a model it does not hold.
type ModelOwner interface {
	ReleaseModel(model uint64)
}

// ErrReporter says why the device's last call refused.
type ErrReporter interface{ Err() string }

// HostReserver is how many bytes of the host's memory the device holds now (an
// integrated GPU's blocks, KV and scratch), which the host's page and expert
// budgets leave room for. It moves with placement, so the model re-reads it
// whenever a block moves (Model.SetPageBudget, State.SetMemBudget).
type HostReserver interface{ HostReserved() uint64 }

// Attacher hands out a view carrying one State's own history (a session).
type Attacher interface{ Attach() LayerDevice }

// Placer brackets a State's whole offer sequence, so two placing at once do
// not interleave against one budget.
type Placer interface {
	BeginPlacement()
	EndPlacement()
}

// BlockPlanner is told how many blocks a placement will offer, so a model that
// fits is spread across the devices rather than filling the first.
type BlockPlanner interface {
	// PlanBlocks says n blocks are about to be offered, and extra is what the
	// placement must leave room for besides their weights: the history the
	// session's context will hold on them, and the output projection.
	PlanBlocks(n int, extra uint64)
}

// RefusalReporter lists the blocks on the device whose last reservation
// refused, so room is made there.
type RefusalReporter interface{ Refused() []int }

// Spiller moves a block onto a later device when the one holding it is full.
type Spiller interface{ SpillAfter(li int) bool }

// RoomReporter moves whenever any device may have gained room.
type RoomReporter interface{ RoomGen() uint64 }

// KVTrimmer shrinks a session's device history to what pos needs.
type KVTrimmer interface{ TrimKV(pos int) bool }

// SessionStepper runs one step for rows that belong to different sessions of
// one device, row i being sess[i]'s sequence at pos[i]: every block's weights
// are read once for all of them. sess are the views Attach handed out; each
// row's history stays its own session's. A session may take several rows --
// a prompt chunk riding beside the others' decoding -- as long as they are its
// next positions, each once, in any row order; they see each other causally,
// as RowsDevice's rows of one sequence do. x is the rows' residual and cs and
// csSWA their rotary rows, as RowsDevice's; the head, when set, works as for RowsDevice.LayersRows,
// LogitRows included.
type SessionStepper interface {
	LayersSessions(lo, hi int, sess []LayerDevice, pos []int, x, cs, csSWA []float32, head *Head) bool
}

// Pipeliner says how many devices a prompt chunk crosses. A device that
// answers more than one takes chunks wider than MaxDevicePrefillChunk and runs
// them in pieces of it, each piece one device behind the next, so the cards
// work at once instead of in turn.
type Pipeliner interface{ PipelineDepth() int }

// LayerGrouper says a device pages its blocks, and how many device chunks a
// prompt chunk may then hold: the device runs the wide chunk one block group at
// a time, every piece through a group before the next pages in, so a block's
// weights cross once per wide chunk. 1 when no device pages.
type LayerGrouper interface{ LayeredChunks() int }

// ExpertHolder says whether the device holding block li holds its routed
// bank, so the host may give the bank's expert pages back with the block's
// page. A block whose experts run off the card (streamed, hybrid) does not.
type ExpertHolder interface{ HoldsExperts(li int) bool }

// HostPageReleaser says a placed block's host page can be given back.
type HostPageReleaser interface{ ReleasesHostPage(li int) bool }

// BlockHolder says block li is already placed for another session: offering it
// again shares the device's copy, so the caller need not read its weights.
type BlockHolder interface{ HoldsBlock(li int) bool }

// RecStepper counts the linear blocks whose recurrent state advanced in the
// last call, which decides whether a failed step may restart on the host.
type RecStepper interface{ RecSteps() int }

// Session is the device view one model.State holds (tier.GPU.Attach): the
// State's own history on the device, apart from the model's weights.
type Session interface {
	// Detach gives back everything the session holds on every device.
	Detach()
	// HeldBytes is what the session holds on every device now: its KV caches
	// and recurrent states. The weights are the model's and not counted.
	HeldBytes() uint64
}

// RecRowsDevice is a RowsDevice whose linear blocks keep a recurrent state per
// row. ResetRecRows zeroes those rows' state for a new occupant.
type RecRowsDevice interface {
	ResetRecRows(rows []int) bool
}

// RecRewinder is a device whose linear blocks can take back one step of a
// session's recurrent state: a speculative verification steps the summary
// over drafted tokens a rejection must undo, and a recurrence cannot be
// rewound by position as attention can. RecMark records where the session's
// state is; RecRewind returns it there, and is honest about when it cannot --
// more than one step since the mark, or a state the device keeps in one copy.
type RecRewinder interface {
	RecMark()
	RecRewind() bool
}

// PlanDecliner is an optional LayerDevice that can answer from the plan alone
// whether it would take a block, before the caller pays to read its weights.
// A graph decline is a property of the shape, and offering a block costs a
// page-in, so this lets placement skip reading blocks the device will refuse.
type PlanDecliner interface {
	DeclinePlan(p *LayerPlan) string
}

// SizeDecliner is an optional LayerDevice that can answer from a block's
// size alone that no device could hold it, before the caller reads it. total
// is every byte the block keeps resident; bank is the routed expert banks'
// share of it, which a device that streams the bank does not keep. A
// mixture block on a large model is gigabytes, so the read this saves is
// the whole cost of a refusal.
type SizeDecliner interface {
	DeclineSize(li int, total, bank uint64) string
}

// AutoStreamer is an optional LayerDevice that, offered a mixture block no
// device could hold resident (SizeDecliner), can take it streamed: the
// block's base on a card and its routed experts sent per token. It answers
// whether it will; blocks is how many blocks the model has and meanBase
// their mean bytes without routed banks, so the device can size what
// streaming keeps beside the bases from the whole model, not from whichever
// block was offered first.
type AutoStreamer interface {
	AutoStream(li int, total, bank uint64, nExpert, nUsed, blocks int, meanBase uint64) bool
}

// ExpertPlacer is an optional LayerDevice that can place a mixture block's
// routed experts off the card: "host" runs them on the host (hybrid
// execution), "card" streams their sheets to the card each token, "" leaves
// the device to decide. The choice holds on every device, so a block that
// moves keeps it.
type ExpertPlacer interface {
	PlaceExperts(li int, where string) bool
}

// ExpertModer is an optional LayerDevice that re-decides where every block
// placed with its experts off the card runs them ("host", "card", or "" for
// its defaults), for a measured trial; it takes effect when a block is next
// placed, and reports how many blocks it touched.
type ExpertModer interface {
	SetExpertMode(where string) int
}

// DeviceName is a device name as a placement compares it: lower case, and the
// API spellings (ptx, amdgcn, spirv, msl) under the backend ones -devices prints, so
// "PTX:0" and "cuda:0" are one device.
func DeviceName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	for _, r := range [][2]string{{"ptx", "cuda"}, {"amdgcn", "hip"}, {"spirv", "vulkan"}, {"msl", "metal"}} {
		if s == r[0] || strings.HasPrefix(s, r[0]+":") {
			s = r[1] + s[len(r[0]):]
		}
	}
	if s == "metal:0" {
		s = "metal"
	}
	return s
}
