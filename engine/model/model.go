// Package model loads a jlm container and runs its transformer graph on
// generated kernels, on the host and on whichever devices take its blocks.
package model

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok"
)

// Config is the hyperparameter set, after per-architecture defaults are applied.
//
// The defaults are the interesting part. Absent is not zero: gemma ships no
// rope.freq_base and no rope.dimension_count, stories15M ships no
// attention.head_count_kv, and every one of those defaulted wrong yields a model
// that loads, runs, and produces confident nonsense.
type Config struct {
	Arch string
	// TokenAct says every projection input is quantized to int8 per token
	// by its absolute maximum, as BitNet's BitLinear does: the session's
	// activation window is the widest row (nn.WithActWindow), so each 32-wide
	// block of a row carries the row's one scale. A property of the graph,
	// read from the arch.
	TokenAct bool
	// Pooling is how an EMBEDDING model reads its vector out of the last
	// residual, and PoolNone for every generator. NonCausal is an encoder's
	// attention: every position sees every other. Both come off the
	// container's flags; see jlm.FlagPoolMean and jlm.FlagNonCausal.
	Pooling   jlm.Pooling
	NonCausal bool
	// chanDecay is ChanDecay()'s backing field: Kimi Delta Attention rather
	// than qwen3next's gated delta rule. Unexported because it is derived from
	// the container's Arch and is not a knob.
	chanDecay bool
	NLayer    int
	NEmbd     int
	NHead     int
	NKVHead   int
	HeadDim   int
	NRot      int
	NFFN      int
	NVocab    int
	NCtx      int
	RMSEps    float64
	RopeBase  float64

	// Architecture quirks.
	EmbdScale float64 // gemma multiplies the embedding by sqrt(n_embd)
	// Act is the feed-forward's gated activation: SiLU (llama), GELU-tanh
	// (gemma) or gpt-oss's clamped swiglu-oai.
	Act      nn.ActKind
	TiedEmbd bool // no output.weight; the lm_head IS token_embd
	QKNorm   bool // qwen3 RMSNorms each head of q and k before RoPE
	// QKNormWide makes that ONE RMSNorm over the whole q (and k) projection
	// instead of one per head. olmoe's attn_q_norm is n_embd wide and is applied
	// before the reshape into heads; qwen3's is head_dim wide and is applied
	// after. Same tensor name, same position in the graph, different arithmetic.
	QKNormWide bool
	// QKNormPost runs q's norm after the rotary, weighted by the k weight
	// folded into it at conversion; k's (ones) still runs before. Hunyuan's
	// order; see jlm.FlagQKNormPostRope.
	QKNormPost bool
	// NoExpertNorm skips the division of the top-k mixture weights by their sum
	// (olmoe). It is negated so the zero value is the renormalising behaviour
	// most mixtures want.
	NoExpertNorm bool
	AttnFactor   float64 // phi3 LongRoPE scales cos and sin by this; 0 means 1
	RopeNeox     bool    // pairs are (i, i+nRot/2), not (2i, 2i+1)
	// RopeSections is Qwen2-VL's M-RoPE: a head's rotary PAIRS split between
	// the (t, h, w) coordinates of a row's position (jlm.Config.RopeSections).
	// A text row's three coordinates are equal, so only an image's rows turn
	// differently; see mrope.go. Zero everywhere else.
	RopeSections [4]int

	// RopeInterleaved says the sections are Qwen3-VL's interleaved split
	// (nn.IMRopeRuns), not contiguous runs (jlm.Arch.InterleavedMRope).
	RopeInterleaved bool

	// RopeXD is HunyuanVL's XD-RoPE (jlm.Arch.XDRope): four axes assigned per
	// element, so a row's table is two pair tables, A for each NEOX pair's
	// first half and B for its second (RopeW floats), the rotation
	// nn.RoPESplit32JIT, and k's norm after it as q's is.
	RopeXD bool

	// Mixture of experts. NExpert 0 is a dense FFN. NFFNExp is the per-expert
	// FFN width, distinct from NFFN, which a MoE file carries for a dense width
	// the model never uses.
	NExpert, NExpertUsed, NFFNExp int

	// NFFNShExp is the SHARED expert's width. A shared expert runs for every
	// token beside the routed ones and carries its own sigmoid gate; zero means
	// the model has none.
	NFFNShExp int

	// AttnOutGate: the q projection is double width and its second half gates
	// the attention output. Not a separate tensor, which is why it is a flag.
	AttnOutGate bool

	// LayerKinds is one entry per layer for a hybrid model, empty otherwise.
	// SSM is the recurrent geometry the linear layers use.
	LayerKinds []jlm.LayerKind
	SSM        jlm.SSMConfig
	// DeltaKeyTiled pairs value head vh with key head vh % key heads rather
	// than vh / (value heads per key head); see jlm.FlagDeltaKeyTiled.
	DeltaKeyTiled bool

	// ssd is SSD()'s backing field: the recurrent layers are Mamba-2's
	// (jlm.LayerSSD), derived from the container's LayerKinds.
	ssd bool
	// ssdAttn: see SSDAttn.
	ssdAttn bool
	// shortConv: see ShortConv.
	shortConv bool
	// mamba1: see Mamba1. dtRank is its dt bottleneck, read off the first
	// Mamba-1 block's x_proj at build(): the container states no rank.
	mamba1 bool
	dtRank int
	// SSMNormGroups is how many groups a Mamba-2 block's gated RMSNorm runs
	// over, each of SSM.Inner/SSMNormGroups channels with its own slice of the
	// weight: the container's SSM.Groups, as mamba_ssm and llama.cpp group it.
	// SSMNormBeforeGate is mamba_ssm's norm_before_gate -- norm(y)*silu(z)
	// instead of norm(y*silu(z)) -- false for every Mamba-2 that ships.
	SSMNormGroups     int
	SSMNormBeforeGate bool

	// Sliding-window attention. SWAWindow is how many keys a local layer may
	// attend to, counting back from the current position and including it;
	// SWAPeriod is the repeat length of the local/global pattern. Zero for
	// either means every layer attends to the whole history.
	//
	// Neither the pattern nor the local rope base is in a GGUF, so the
	// converter supplies them from llama.cpp's architecture knowledge.
	SWAWindow, SWAPeriod int

	// Logit softcapping: x = c*tanh(x/c). gemma2 caps the attention scores at
	// 50 and the output logits at 30. Zero means none. Omitting it gives fluent
	// wrong output that only an external oracle catches.
	AttnSoftcap, FinalSoftcap float32
	RopeBaseSWA               float64
	// SWARopePlain makes the sliding layers' rotary carry no YaRN magnitude
	// (AttnFactor) as well as no YaRN frequencies: OLMo 3's local layers
	// rotate at their base unscaled (llama.cpp's olmo2 builder passes
	// attn_factor 1 there, transformers builds them a default rotary).
	SWARopePlain bool

	// Rotary scaling; see jlm.Config. YarnFactor 0 is no YaRN and RopeLinear 0
	// no linear interpolation. Both scale the global rotary only.
	YarnFactor, YarnBetaFast, YarnBetaSlow float64
	YarnOrigCtx                            int
	YarnExact                              bool
	// NoPosEnc: the attention blocks apply no positional encoding at all.
	// Position still orders the causal mask and never reaches q or k.
	NoPosEnc   bool
	RopeLinear float64

	// Multi-head Latent Attention; see jlm.Config for the graph. KVLoraRank 0
	// is every architecture but DeepSeek's.
	//
	// HeadDimV is the value head's width, filled in even where the container
	// stores zero for "same as HeadDim".
	QLoraRank, KVLoraRank int
	HeadDimV              int

	// The DeepSeek router: sigmoid gating, a selection-only bias, grouped
	// scoring and a scale on the surviving weights. NExpertGroup <= 1 is the
	// ungrouped top-k every other mixture here uses.
	ExpertSigmoid                  bool
	ExpertScale                    float64
	NExpertGroup, NExpertGroupUsed int
	// SparseMixer, when non-zero, is Phi-3.5-MoE's router and its jitter_eps
	// (jlm.ArchPhiMoE carries it; nn.MoEGate.SparseMixer has the arithmetic).
	SparseMixer float32

	// AttnScale is the softmax's multiplier, already including YaRN's squared
	// magnitude correction. The converter folds one copy of the correction into
	// AttnFactor (which scales cos and sin); the score is scaled by its square
	// (DeepSeek's softmax_scale, llama.cpp's kq_scale), so it is its own field.
	AttnScale float64

	// NDenseLead is how many leading blocks have a dense feed-forward. The
	// forward graph branches on the router tensor instead; this is for the
	// loader, diagnostics and placement.
	NDenseLead int

	// Llama 4; see jlm.Config. MoEStep interleaves dense and mixture blocks,
	// SWAChunked makes the windowed layers attend within an aligned chunk,
	// NoPEGlobal takes the rotary off the unwindowed layers, QKL2Norm is a
	// weightless per-head RMSNorm of q and k after the rotary on the layers
	// that rotate, and ExpertWeightIn scales the expert's input by its routed
	// weight rather than its output.
	MoEStep                                    int
	SWAChunked, NoPEGlobal, QKL2Norm           bool
	ExpertWeightIn                             bool
	AttnTempScale, AttnTempFloor, AttnTempOffs float64

	// Granite's residual and logit scales; see jlm.Config. Both are one, never
	// zero, once loaded: the residual add multiplies by the first and the
	// logits are divided by the second on every path that produces them.
	ResidualScale, LogitScale float64

	// ClampKQV clamps q, k and v to [-ClampKQV, ClampKQV] after their
	// projections: DBRX's clip_qkv. Zero means none; see jlm.Config.
	ClampKQV float32

	// NMTP is how many multi-token-prediction blocks follow the trunk (blocks
	// NLayer..NLayer+NMTP-1); zero for a model that ships none. They run only
	// for speculative drafting (spec.go), never in the trunk.
	NMTP int

	// The sliding layers' attention geometry (jlm.Config.HeadDimSWA): equal to
	// HeadDim, NKVHead and NRot for every model but Gemma 4, whose global layers
	// are 512 wide where its sliding ones are 256. Read through HeadDimAt and
	// friends, never directly.
	HeadDimSWA, NKVHeadSWA, NRotSWA int
	// VNorm RMSNorms each head of v with no weight before it is cached
	// (jlm.Flag2VNorm, Gemma 4).
	VNorm bool
	// BidirSWA says an image's rows see each other on the sliding layers
	// only, the global ones staying causal: Gemma 4's 26B and 31B
	// (use_bidirectional_attention "vision"), where Gemma 3's rows see each
	// other on every layer. The E-models (per-layer embeddings) keep the
	// image causal, as transformers' config and llama.cpp's mtmd both have
	// them; the GGUF states neither, so it is read off the per-layer
	// embedding, the one difference between the two lines.
	BidirSWA bool
	// NKVShared, PLEDim and DoubleFFN are Gemma 4's E2B/E4B: the last
	// NKVShared layers attend to an earlier layer's history (KVSource) and,
	// under DoubleFFN, run an FFN twice NFFN wide (NFFNAt); PLEDim is each
	// layer's slice of the per-layer embeddings, zero for none.
	NKVShared, PLEDim int
	DoubleFFN         bool
	// IdxHeads, IdxHeadDim and IdxTopK are DeepSeek V3.2's lightning indexer
	// (indexer.go): IdxHeads heads of IdxHeadDim score every cached position,
	// and attention reads the IdxTopK best. Zero heads means none.
	IdxHeads, IdxHeadDim, IdxTopK int
	// IdxBlock and IdxLocal are MiniMax Sparse Attention (msa.go): the same
	// indexer fields, one head per kv group, selecting IdxTopK blocks of
	// IdxBlock positions with the IdxLocal ending at the query's forced in, on
	// every block past the dense lead. Zero IdxBlock is DeepSeek V3.2's
	// per-position selection.
	IdxBlock, IdxLocal int
	// idxFault breaks one piece of the indexer, for a gate's violation only.
	idxFault idxFault
	// msaFault breaks one piece of MiniMax-M3's block selection, for a gate's
	// violation only.
	msaFault msaFault
	// DeepSeek V4 (ds4.go): HCMult residual streams mixed by hyper-
	// connections (HCIters Sinkhorn rounds at HCEps), the attention output
	// projected through OGroups groups of OLoraRank, each block's compression
	// (CompKinds, at CompRateCSA or CompRateHCA), the first NHashLayers
	// blocks routed by token id, and the router's sqrt-softplus gate.
	HCMult, HCIters          int
	HCEps                    float64
	OGroups, OLoraRank       int
	CompRateCSA, CompRateHCA int
	NHashLayers              int
	CompKinds                []jlm.CompKind
	ExpertSqrtSoftplus       bool
	// ds4Fault breaks one piece of DeepSeek V4's graph, for a gate's
	// violation only.
	ds4Fault ds4Fault
	// Qwen3.8-Flash-Next (q4e.go): the hyper-connections' gate rank, QSA's
	// block size, the n-gram embedding (its one-based block, n, heads per n,
	// row width, convolution taps, the end-of-sequence and picture ids and
	// the hash's constants) and the delta rule's sigmoid output gate.
	HCLowRank, QSARatio              int
	NGramBlock, NGram, NGramHeadsPer int
	NGramRowDim, NGramConv           int
	NGramEOS, NGramImage             int32
	NGramMul, NGramOff, NGramVoc     []uint64
	DeltaSigmoidGate                 bool
	// q4eFault breaks one piece of Qwen3.8-Flash-Next's graph, for a gate's
	// violation only.
	q4eFault q4eFault
	// Kimi-K3 (k3.go): a residual checkpoint every AttnResBlock blocks, each
	// sublayer's input their softmax mix with the running residual (zero is
	// none); the routed experts at ExpertLatent (zero is n_embd); KDA's decay
	// bound (zero is the softplus decay); the MLA latent norms' epsilon.
	AttnResBlock  int
	ExpertLatent  int
	KDALowerBound float32
	LatentNormEps float64
	// k3Fault breaks one piece of Kimi-K3's graph, for a gate's violation
	// only.
	k3Fault k3Fault
	// AltUp is Gemma 3n's residual stream count (altup.go), zero for one
	// stream; NSparse blocks lead with a gaussian top-k on their FFN gate at
	// SparseStd deviations.
	AltUp, NSparse int
	SparseStd      float32
	// kvSrcOf replaces KVSource's choice, for a gate's violation only.
	kvSrcOf func(li int) int
	// noRouter marks the blocks of a mixture model that carry no router:
	// Nemotron 3's mixer-only layers and Jamba's dense blocks between its
	// mixtures. MoEAt is false there. Read off the container at build(), nil
	// when no block is one.
	noRouter []bool
	// DenseMoE is Gemma 4's mixture block (jlm.Flag2DenseMoE): a dense MLP and
	// the mixture side by side, each with its own norms, the router on its own
	// norm of the residual and each routed weight times its expert's scale.
	DenseMoE bool

	// The classic block (C6); see jlm.FlagLayerNorm and jlm.FlagParallel.
	// LayerNorm makes every text-graph norm mean-centred, with the norm's
	// bias when the block carries one; Parallel feeds attention and the FFN
	// the same block input and adds both into it. RMSEps is the LayerNorm's
	// epsilon when LayerNorm is set.
	LayerNorm, Parallel bool
}

// RopeAt reports whether layer il rotates q and k at all.
//
// Roping a NoPE layer anyway is exact at position 0, where the rotation is the
// identity, and wrong after it.
func (c *Config) RopeAt(il int) bool {
	if c.NoPosEnc {
		return false
	}
	return !(c.NoPEGlobal && !c.periodLocal(il))
}

// periodLocal reports whether layer il is not the last of its SWAPeriod:
// the windowed layers of a sliding pattern, and where no layer slides
// (SmolLM3's SWAWindow 0) the layers FlagNoPEGlobal leaves rotating. It is SWA
// wherever a window exists, so every windowed model reads as before.
func (c *Config) periodLocal(il int) bool {
	return c.SWAPeriod > 0 && il%c.SWAPeriod < c.SWAPeriod-1
}

// AttnWindow is the key range layer il attends to from position pos: w0 is
// the first key and an how many there are, so the keys are w0..w0+an-1 and
// the last is pos itself. A layer that sees the whole history gets 0, pos+1.
//
// Decode, prefill and the batch all call this rather than spelling the window
// themselves. A chunked layer's start is floor(pos/W)*W: at pos = W it sees
// one key, where a sliding one sees W.
func (c *Config) AttnWindow(il, pos int) (w0, an int) {
	an = pos + 1
	if !c.SWA(il) || c.SWAWindow <= 0 {
		return 0, an
	}
	if c.SWAChunked {
		w0 = pos / c.SWAWindow * c.SWAWindow
		return w0, pos + 1 - w0
	}
	if c.SWAWindow < an {
		return an - c.SWAWindow, c.SWAWindow
	}
	return 0, an
}

// AttnTemp is the factor layer il's query is scaled by at position pos: one
// everywhere but a layer with no positional encoding under Llama 4's
// temperature tuning, where it is 1 + s*ln(1 + floor((pos+o)/f)).
//
// It is folded into the softmax scale rather than applied to q, which is the
// same number since the scores are linear in q.
func (c *Config) AttnTemp(il, pos int) float64 {
	if c.AttnTempScale == 0 || (c.NoPEGlobal && c.RopeAt(il)) {
		return 1
	}
	return 1 + c.AttnTempScale*math.Log(1+math.Floor((float64(pos)+c.AttnTempOffs)/c.AttnTempFloor))
}

// SWA reports whether layer il attends to a window rather than the whole
// history.
//
// The last layer of each period is the global one, as in
// llama_hparams::set_swa_pattern:
//
//	is_swa_impl[il] = il % n_pattern < (n_pattern - 1)
//
// so at period 6 layers 0..4 are local and layer 5 is global.
func (c *Config) SWA(il int) bool {
	return c.SWAWindow > 0 && c.SWAPeriod > 0 && il%c.SWAPeriod < c.SWAPeriod-1
}

// MoE reports whether this model's FFN is a mixture of experts.
func (c *Config) MoE() bool { return c.NExpert > 0 }

// GQA is how many query heads share one key/value head.
func (c *Config) GQA() int { return c.NHead / c.NKVHead }

// KVDim is the width of one K or V vector: kv heads times head dim.
//
// Under MLA it is one row for the whole layer: the latent plus the shared
// rotary key, KVLoraRank + NRot, however many heads read it. Everything that
// sizes a page, a migration buffer or a geometry digest goes through here. MLA
// also aliases V onto K (kvPages.latent), so the width is charged once.
func (c *Config) KVDim() int {
	if c.KVLoraRank != 0 {
		// The indexer's key rides the same row, after the rotary key: it is
		// cached per position and per layer exactly as the latent is, so it
		// pages, relocates and is stored with it (indexer.go).
		return c.KVLoraRank + c.NRot + c.IdxHeadDim
	}
	return c.NKVHead * c.HeadDim
}

// MLA reports that this model's attention is over a compressed latent.
func (c *Config) MLA() bool { return c.KVLoraRank != 0 }

// Indexer reports that attention reads only the positions DeepSeek V3.2's
// lightning indexer selects.
func (c *Config) Indexer() bool { return c.IdxHeads != 0 && c.IdxBlock == 0 && !c.DSV4() && !c.Q4E() }

// MSA reports MiniMax Sparse Attention: attention on the blocks past the
// dense lead reads only the position blocks each kv group's indexer head
// selects (msa.go).
func (c *Config) MSA() bool { return c.IdxHeads != 0 && c.IdxBlock != 0 }

// MSAAt reports that block li selects its keys: every block past the dense
// lead of an MSA model.
func (c *Config) MSAAt(li int) bool { return c.MSA() && li >= c.NDenseLead }

// extraKVHead reports that layer li's cached row carries an indexer key as
// one more kv head: MiniMax-M3's (msa.go) and Qwen3.8-Flash-Next's QSA
// (qsa.go).
func (c *Config) extraKVHead(li int) bool { return c.MSAAt(li) || c.QSAAt(li) }

// KVRowAt is one cached position's width in layer li's history: KVDimAt,
// plus on an MSA block the indexer's key as one more kv head (msa.go). Every
// cache allocation, page and migration of a layer goes through it; the k and
// v projections are KVDimAt wide.
func (c *Config) KVRowAt(li int) int {
	if c.extraKVHead(li) {
		return (c.NKVHead + 1) * c.HeadDim
	}
	if c.DSV4() {
		// The key, and on a compressed block the compressor's pending
		// inputs (ds4.go).
		return c.ds4RowAt(li).width
	}
	return c.KVDimAt(li)
}

// GeomSplit reports that the sliding layers attend at another geometry than
// the global ones (Gemma 4): another head width, kv head count or rotary
// width. HeadDim, NKVHead and NRot are then the global layers', and every
// per-layer path reads HeadDimAt, NKVHeadAt and NRotAt.
func (c *Config) GeomSplit() bool {
	return c.HeadDimSWA != c.HeadDim || c.NKVHeadSWA != c.NKVHead || c.NRotSWA != c.NRot
}

// HeadDimAt, NKVHeadAt and NRotAt are layer li's attention geometry: the
// sliding layers' own where Config.SWA says li slides, the model's otherwise.
// For every model without a split they are HeadDim, NKVHead and NRot.
func (c *Config) HeadDimAt(li int) int {
	if c.SWA(li) {
		return c.HeadDimSWA
	}
	return c.HeadDim
}

// NKVHeadAt is layer li's kv head count (see HeadDimAt).
func (c *Config) NKVHeadAt(li int) int {
	if c.SWA(li) {
		return c.NKVHeadSWA
	}
	return c.NKVHead
}

// NRotAt is layer li's rotary width (see HeadDimAt).
func (c *Config) NRotAt(li int) int {
	if c.SWA(li) {
		return c.NRotSWA
	}
	return c.NRot
}

// KVDimAt is KVDim at layer li's geometry; MLA's one cached row is the same
// on every layer.
func (c *Config) KVDimAt(li int) int {
	if c.KVLoraRank != 0 {
		return c.KVDim()
	}
	return c.NKVHeadAt(li) * c.HeadDimAt(li)
}

// GQAAt is GQA at layer li's geometry.
func (c *Config) GQAAt(li int) int { return c.NHead / c.NKVHeadAt(li) }

// QDimAt is the query (and the attention output) width at layer li.
func (c *Config) QDimAt(li int) int { return c.NHead * c.HeadDimAt(li) }

// MaxHeadDim, MaxKVDim, MaxQDim and MaxNRot are the widest of the two
// geometries: what a buffer that serves every layer is sized by.
func (c *Config) MaxHeadDim() int { return max(c.HeadDim, c.HeadDimSWA) }

// KVShared reports that block li computes no k and v of its own and attends
// to KVSource(li)'s history (Gemma 4's E2B/E4B).
func (c *Config) KVShared(li int) bool { return c.NKVShared != 0 && li >= c.NLayer-c.NKVShared }

// KVSource is the block whose history block li attends to: li itself, or for
// a KV-sharing block the last block before the shared run of li's own kind
// (jlm.Config.KVSource). -1 when there is none, which load refuses.
func (c *Config) KVSource(li int) int {
	if !c.KVShared(li) {
		return li
	}
	if c.kvSrcOf != nil {
		return c.kvSrcOf(li)
	}
	for j := c.NLayer - c.NKVShared - 1; j >= 0; j-- {
		if c.SWA(j) == c.SWA(li) {
			return j
		}
	}
	return -1
}

// NFFNAt is block li's feed-forward width (twice NFFN on a KV-sharing block
// under DoubleFFN), and MaxNFFN the widest.
func (c *Config) NFFNAt(li int) int {
	if c.DoubleFFN && c.KVShared(li) {
		return 2 * c.NFFN
	}
	return c.NFFN
}

// MaxNFFN is the widest feed-forward width of any block: twice NFFN when the
// model has KV-sharing blocks under DoubleFFN, NFFN otherwise.
func (c *Config) MaxNFFN() int {
	if c.DoubleFFN && c.NKVShared != 0 {
		return 2 * c.NFFN
	}
	return c.NFFN
}

// MaxKVDim is the widest row any layer caches: MLA's latent row, an MSA
// block's k with the indexer's key after it, DeepSeek V4's widest KVRowAt, or
// else the wider of the two geometries' KV widths.
func (c *Config) MaxKVDim() int {
	if c.KVLoraRank != 0 {
		return c.KVDim()
	}
	if c.MSA() || c.QSARatio != 0 {
		// The row an MSA or QSA block writes: its k with the indexer's key
		// after it.
		return (c.NKVHead + 1) * c.HeadDim
	}
	if c.DSV4() {
		w := 0
		for li := 0; li < c.NLayer; li++ {
			w = max(w, c.KVRowAt(li))
		}
		return w
	}
	return max(c.KVDim(), c.NKVHeadSWA*c.HeadDimSWA)
}

// MaxQDim is the query width at the wider of the two head widths.
func (c *Config) MaxQDim() int { return c.NHead * c.MaxHeadDim() }

// MaxNRot is the wider of the two geometries' rotary widths.
func (c *Config) MaxNRot() int { return max(c.NRot, c.NRotSWA) }

// RopeW is how many floats one row's rotary table holds: NRot, or two tables
// of it under XD-RoPE (RopeXD).
func (c *Config) RopeW() int {
	if c.RopeXD {
		return 2 * c.NRot
	}
	return c.NRot
}

// MoEAt reports whether block li has a mixture feed-forward. MoE() is a
// property of the model and this of the block: DeepSeek's first NDenseLead
// blocks are dense, and Llama 4 interleaves with period MoEStep.
//
// The forward paths branch on the router tensor's presence instead; the loader
// asks this because it runs before the tensors are bound.
func (c *Config) MoEAt(li int) bool {
	if !c.MoE() || li < c.NDenseLead || li < len(c.noRouter) && c.noRouter[li] {
		return false
	}
	return c.MoEStep <= 1 || (li+1)%c.MoEStep == 0
}

// QKNormShape gives the q and k RMSNorm loops their group counts and widths.
// qwen3 norms each head over HeadDim; olmoe and OLMo 2 norm the whole
// projection once, and the k projection is NKVHead heads wide: a GQA OLMo 2
// (the 32B) norms q over NHead*HeadDim and k over NKVHead*HeadDim, the width
// of its attn_k_norm.
func (c *Config) QKNormShape() (nq, nk, wq, wk int) {
	if c.QKNormWide {
		return 1, 1, c.NHead * c.HeadDim, c.NKVHead * c.HeadDim
	}
	return c.NHead, c.NKVHead, c.HeadDim, c.HeadDim
}

// QKNormShapeAt is QKNormShape at layer li's geometry (HeadDimAt). A split
// geometry never has the whole-projection norm (configFrom refuses it).
func (c *Config) QKNormShapeAt(li int) (nq, nk, wq, wk int) {
	if c.QKNormWide {
		return c.QKNormShape()
	}
	hd := c.HeadDimAt(li)
	return c.NHead, c.NKVHeadAt(li), hd, hd
}

// QKNormAt is whether block li RMSNorms q and k: jlm.QKNormOn, the one
// derivation the converter, the reader, the host and the device all read. The
// tensors are there exactly when it is true -- jlm.Open refuses a container
// where they are not.
func (c *Config) QKNormAt(li int) bool { return jlm.QKNormOn(c.QKNorm, c.LayerKind(li)) }

// qkNorms is the pair block li offers a device: its weights when QKNormAt says
// the block norms q and k, and none when it does not, so the device's launch
// follows the same derivation as the host's.
func (s *State) qkNorms(li int) (q, k []float32) {
	if !s.c.QKNormAt(li) {
		return nil, nil
	}
	l := &s.m.layers[li]
	return l.qNorm, l.kNorm
}

// LayerKind is what layer i actually is. A model with no per-layer map is
// ordinary attention all the way down, which is every architecture but one.
//
// The container states it per layer; it is never derived from an architecture
// name.
func (c *Config) LayerKind(i int) jlm.LayerKind {
	if i < 0 || i >= len(c.LayerKinds) {
		return jlm.LayerFullAttn
	}
	return c.LayerKinds[i]
}

// Hybrid reports whether any layer is recurrent.
func (c *Config) Hybrid() bool { return len(c.LayerKinds) != 0 }

// ChanDecay reports that the linear blocks run Kimi Delta Attention rather
// than qwen3next's gated delta rule: the state decays at a rate per channel
// instead of one per head, and both gates come off a low-rank bottleneck. It is
// a property of the architecture, not of LayerKinds, which only says a block
// is recurrent.
func (c *Config) ChanDecay() bool { return c.chanDecay }

// SSD reports that the recurrent blocks are Mamba-2 mixers (jlm.LayerSSD):
// the delta rule's state stepped by the selective update, with a skip, a
// convolution bias and a grouped gated norm.
func (c *Config) SSD() bool { return c.ssd }

// SSDAttn reports whether some block runs attention and a Mamba-2 mixer side
// by side (Falcon-H1, jlm.LayerSSDAttn): the mixer's output needs a buffer of
// its own beside the attention's.
func (c *Config) SSDAttn() bool { return c.ssdAttn }

// ShortConv reports that the recurrent blocks are LFM2's gated short
// convolutions (jlm.LayerShortConv): a convolution window and no state beside
// it.
func (c *Config) ShortConv() bool { return c.shortConv }

// Mamba1 reports that the recurrent blocks are Mamba-1's selective scan
// (jlm.LayerMamba1): every channel a head of one row, the decay a vector over
// the state, dt off a low-rank bottleneck of the convolved x.
func (c *Config) Mamba1() bool { return c.mamba1 }

// configFrom translates the container's typed description into the engine's.
//
// Parsing and architecture knowledge live in convert.configOf; this copies
// fields and derives. The two types differ on purpose: the container stores
// what the file says and this carries what the engine derived.
func configFrom(c *jlm.Config) (*Config, error) {
	if c == nil {
		return nil, fmt.Errorf("model: the container has no config section")
	}
	if !c.Arch.Valid() || c.Arch == jlm.ArchCLIP {
		return nil, fmt.Errorf("model: %v is not a language model", c.Arch)
	}
	out := &Config{
		Arch:         c.Arch.String(),
		chanDecay:    c.Arch == jlm.ArchKimiLinear || c.Arch == jlm.ArchKimiK3,
		SparseMixer:  sparseMixerOf(c.Arch),
		NLayer:       int(c.NLayer),
		NMTP:         int(c.NMTP),
		NEmbd:        int(c.NEmbd),
		NHead:        int(c.NHead),
		NKVHead:      int(c.NKVHead),
		HeadDim:      int(c.HeadDim),
		NRot:         int(c.NRot),
		NFFN:         int(c.NFFN),
		NVocab:       int(c.NVocab),
		NCtx:         int(c.NCtx),
		RMSEps:       float64(c.RMSEps),
		RopeBase:     float64(c.RopeBase),
		EmbdScale:    float64(c.EmbdScale),
		AttnFactor:   float64(c.AttnFactor),
		NExpert:      int(c.NExpert),
		NExpertUsed:  int(c.NExpertUsed),
		NFFNExp:      int(c.NFFNExp),
		SWAWindow:    int(c.SWAWindow),
		SWAPeriod:    int(c.SWAPeriod),
		RopeBaseSWA:  float64(c.RopeBaseSWA),
		SWARopePlain: c.Arch == jlm.ArchOLMo3,
		AttnSoftcap:  c.AttnSoftcap,
		FinalSoftcap: c.FinalSoftcap,
		YarnFactor:   float64(c.YarnFactor),
		YarnBetaFast: float64(c.YarnBetaFast),
		YarnBetaSlow: float64(c.YarnBetaSlow),
		YarnOrigCtx:  int(c.YarnOrigCtx),
		YarnExact:    c.Flags.Has(jlm.FlagYarnExact),
		NoPosEnc:     c.Flags.Has(jlm.FlagNoPosEnc),
		RopeLinear:   float64(c.RopeLinear),
		TiedEmbd:     c.Flags.Has(jlm.FlagTiedEmbd),
		QKNorm:       c.Flags.Has(jlm.FlagQKNorm),
		QKNormWide:   c.Flags.Has(jlm.FlagQKNormWide),
		QKNormPost:   c.Flags.Has(jlm.FlagQKNormPostRope),
		NoExpertNorm: c.Flags.Has(jlm.FlagNoExpertNorm),
		RopeNeox:     c.Flags.Has(jlm.FlagRopeNeox),
		RopeSections: mropeSections(c),
		AttnOutGate:  c.Flags.Has(jlm.FlagAttnOutGate),
		NFFNShExp:    int(c.NFFNShExp),
		LayerKinds:   c.LayerKinds,
		SSM:          c.SSM,

		DeltaKeyTiled: c.Flags.Has(jlm.FlagDeltaKeyTiled),

		QLoraRank:        int(c.QLoraRank),
		KVLoraRank:       int(c.KVLoraRank),
		HeadDimV:         int(c.HeadDimV),
		ExpertSigmoid:    c.Flags.Has(jlm.FlagExpertSigmoid),
		ExpertScale:      float64(c.ExpertScale),
		NExpertGroup:     int(c.NExpertGroup),
		NExpertGroupUsed: int(c.NExpertGroupUsed),
		NDenseLead:       int(c.NDenseLead),
		Pooling:          c.Flags.Pooling(),
		NonCausal:        c.Flags.Has(jlm.FlagNonCausal),

		MoEStep:        int(c.MoEStep),
		SWAChunked:     c.Flags.Has(jlm.FlagSWAChunked),
		NoPEGlobal:     c.Flags.Has(jlm.FlagNoPEGlobal),
		QKL2Norm:       c.Flags.Has(jlm.FlagQKL2Norm),
		ExpertWeightIn: c.Flags.Has(jlm.FlagExpertWeightIn),
		AttnTempScale:  float64(c.AttnTempScale),
		AttnTempFloor:  float64(c.AttnTempFloor),
		AttnTempOffs:   float64(c.AttnTempOffset),
		ResidualScale:  float64(c.ResidualScale),
		LogitScale:     float64(c.LogitScale),
		ClampKQV:       c.ClampKQV,
		LayerNorm:      c.Flags.Has(jlm.FlagLayerNorm),
		Parallel:       c.Flags.Has(jlm.FlagParallel),

		HCMult:      int(c.HCMult),
		HCLowRank:   int(c.HCLowRank),
		HCIters:     int(c.HCIters),
		HCEps:       float64(c.HCEps),
		OGroups:     int(c.OGroups),
		OLoraRank:   int(c.OLoraRank),
		CompRateCSA: int(c.CompRateCSA),
		CompRateHCA: int(c.CompRateHCA),
		NHashLayers: int(c.NHashLayers),
		CompKinds:   c.CompKinds,
	}
	// Zero means the global layers' geometry; the engine wants the widths.
	out.HeadDimSWA, out.NKVHeadSWA, out.NRotSWA = out.HeadDim, out.NKVHead, out.NRot
	if c.HeadDimSWA != 0 {
		out.HeadDimSWA = int(c.HeadDimSWA)
	}
	if c.NKVHeadSWA != 0 {
		out.NKVHeadSWA = int(c.NKVHeadSWA)
	}
	if c.NRotSWA != 0 {
		out.NRotSWA = int(c.NRotSWA)
	}
	out.VNorm = c.Flags2.Has(jlm.Flag2VNorm)
	out.DenseMoE = c.Flags2.Has(jlm.Flag2DenseMoE)
	out.NKVShared, out.PLEDim = int(c.NKVShared), int(c.PLEDim)
	out.BidirSWA = c.Arch == jlm.ArchGemma4 && c.PLEDim == 0
	out.IdxHeads, out.IdxHeadDim, out.IdxTopK = int(c.IdxHeads), int(c.IdxHeadDim), int(c.IdxTopK)
	out.IdxBlock, out.IdxLocal = int(c.IdxBlock), int(c.IdxLocal)
	if out.MSA() && (out.MLA() || out.IdxHeads != out.NKVHead || out.IdxHeadDim != out.HeadDim ||
		out.IdxTopK <= 0 || out.IdxLocal < 1 || out.IdxLocal > out.IdxTopK || out.GeomSplit() ||
		out.NRot > out.IdxHeadDim || out.NKVShared != 0) {
		return nil, fmt.Errorf("model: %s: block selection over %d heads of %d (top %d blocks of %d, "+
			"%d local) on %d kv heads of %d", out.Arch, out.IdxHeads, out.IdxHeadDim, out.IdxTopK,
			out.IdxBlock, out.IdxLocal, out.NKVHead, out.HeadDim)
	}
	out.AltUp, out.NSparse, out.SparseStd = int(c.AltUp), int(c.NSparse), c.SparseStd
	if out.AltUp == 1 || (out.AltUp != 0 && (out.PLEDim == 0 || out.MLA() || out.Hybrid() || out.MoE() ||
		out.Parallel || out.GeomSplit())) || out.NSparse > out.NLayer {
		return nil, fmt.Errorf("model: %s: %d AltUp streams (per-layer input %d) and %d sparse blocks of %d",
			out.Arch, out.AltUp, out.PLEDim, out.NSparse, out.NLayer)
	}
	if out.Indexer() && (!out.MLA() || out.QLoraRank == 0 || out.IdxTopK <= 0 ||
		out.IdxHeadDim%8 != 0 || out.NRot > out.IdxHeadDim) {
		return nil, fmt.Errorf("model: %s: an indexer of %d heads of %d (top %d) on q_lora_rank %d, "+
			"kv_lora_rank %d, rotary %d", out.Arch, out.IdxHeads, out.IdxHeadDim, out.IdxTopK,
			out.QLoraRank, out.KVLoraRank, out.NRot)
	}
	out.DoubleFFN = c.Flags2.Has(jlm.Flag2DoubleFFN)
	if (out.NKVShared != 0 || out.PLEDim != 0) && (out.MLA() || out.Hybrid() || out.MoE()) {
		return nil, fmt.Errorf("model: %s: KV sharing or per-layer embeddings on MLA, a hybrid "+
			"or a mixture", out.Arch)
	}
	for li := out.NLayer - out.NKVShared; li < out.NLayer; li++ {
		if out.KVSource(li) < 0 {
			return nil, fmt.Errorf("model: %s: KV-sharing block %d has no earlier block of its kind",
				out.Arch, li)
		}
	}
	if out.DenseMoE && (out.NExpert == 0 || out.ExpertSigmoid || out.NoExpertNorm ||
		out.NExpertGroup > 1 || out.ExpertWeightIn) {
		return nil, fmt.Errorf("model: %s: a dense-and-mixture block on anything but a "+
			"renormalised softmax router", out.Arch)
	}
	// The split is Gemma 4's and is built for its block alone. A scale derived
	// from the head width would be two scales, and the one field is one.
	if out.GeomSplit() && (out.MLA() || out.Hybrid() || out.QKNormWide || out.AttnOutGate ||
		c.AttnScale == 0 || out.QKL2Norm || out.ClampKQV != 0) {
		return nil, fmt.Errorf("model: %s: a sliding-layer geometry on MLA, a hybrid, a whole-"+
			"projection q/k norm, an output gate, a clamp or a derived attention scale", out.Arch)
	}
	for _, k := range c.LayerKinds {
		switch k {
		case jlm.LayerSSD:
			out.ssd = true
		case jlm.LayerSSDAttn:
			out.ssd, out.ssdAttn = true, true
		case jlm.LayerShortConv:
			out.shortConv = true
		case jlm.LayerMamba1:
			out.mamba1 = true
		}
	}
	if out.mamba1 {
		for i, k := range c.LayerKinds {
			if k.Recurrent() && k != jlm.LayerMamba1 {
				return nil, fmt.Errorf("model: layer %d is %v in a Mamba-1 model", i, k)
			}
		}
	}
	if out.shortConv {
		for i, k := range c.LayerKinds {
			if k.Recurrent() && k != jlm.LayerShortConv {
				return nil, fmt.Errorf("model: layer %d is %v in a short-convolution model", i, k)
			}
		}
	}
	out.SSMNormGroups = max(int(c.SSM.Groups), 1)
	// One recurrence a model: every recurrent layer is a gated delta rule or
	// every one is Mamba-2's, since the two share one state geometry.
	if out.ssd {
		for i, k := range c.LayerKinds {
			if k == jlm.LayerLinearAttn {
				return nil, fmt.Errorf("model: layer %d is a gated delta rule in a Mamba-2 model", i)
			}
		}
	}
	if out.ResidualScale == 0 {
		out.ResidualScale = 1
	}
	if out.LogitScale == 0 {
		out.LogitScale = 1
	}
	// The container's zero means "the same as HeadDim"; the engine wants the
	// width.
	if out.HeadDimV == 0 {
		out.HeadDimV = out.HeadDim
	}
	if out.ExpertScale == 0 {
		out.ExpertScale = 1
	}
	// The score's scale, with YaRN's correction squared; without YaRN it is
	// 1/sqrt(HeadDim). A model that states its own scale (Granite's
	// attention.scale, gemma 27B's) is taken at its word; see jlm.Config.
	out.AttnScale = 1 / math.Sqrt(float64(out.HeadDim))
	if c.AttnScale != 0 {
		out.AttnScale = float64(c.AttnScale)
	}
	if c.YarnLogMul != 0 && c.YarnFactor > 1 {
		mscale := float64(c.YarnLogMul)*math.Log(float64(c.YarnFactor)) + 1
		out.AttnScale *= mscale * mscale
	}
	nact := 0
	for _, fl := range []jlm.Flags{jlm.FlagGELU, jlm.FlagSwiGLUOAI, jlm.FlagReLU2} {
		if c.Flags.Has(fl) {
			nact++
		}
	}
	switch {
	case nact > 1:
		return nil, fmt.Errorf("model: %s: the container claims two activations", out.Arch)
	case c.Flags.Has(jlm.FlagReLU2):
		out.Act = nn.ActReLU2
	case c.Flags.Has(jlm.FlagGELU):
		out.Act = nn.ActGELU
	case c.Flags.Has(jlm.FlagSwiGLUOAI):
		out.Act = nn.ActSwiGLUOAI
	}
	// BitNet quantizes every projection input per token (jlm.ArchBitNet).
	out.TokenAct = c.Arch == jlm.ArchBitNet
	// Apertus's activation is its arch's: a file states it only through the
	// per-block numbers (jlm.RoleXIELU).
	if c.Arch == jlm.ArchApertus {
		if nact > 0 {
			return nil, fmt.Errorf("model: %s: the container claims a second activation beside xIELU", out.Arch)
		}
		out.Act = nn.ActXIELU
	}
	if c.Flags2.Has(jlm.Flag2SwiGLUClamp) {
		if nact != 0 {
			return nil, fmt.Errorf("model: %s: the container claims two activations", out.Arch)
		}
		out.Act = nn.ActSwiGLUClamp
	}
	if err := ds4Config(out, c); err != nil {
		return nil, err
	}
	if err := q4eConfig(out, c); err != nil {
		return nil, err
	}
	if err := k3Config(out, c, nact); err != nil {
		return nil, err
	}
	// The after-rotary norm is FlagQKNorm's per-head norm moved (see
	// jlm.FlagQKNormPostRope); without the norm, or on a whole-projection one,
	// it describes nothing the engine runs.
	if out.QKNormPost && (!out.QKNorm || out.QKNormWide) {
		return nil, fmt.Errorf("model: %s: FlagQKNormPostRope needs the per-head FlagQKNorm "+
			"(QKNorm %v, QKNormWide %v)", out.Arch, out.QKNorm, out.QKNormWide)
	}
	// Zero means one for both scales; see jlm.Config.
	if out.EmbdScale == 0 {
		out.EmbdScale = 1
	}
	if out.AttnFactor == 0 {
		out.AttnFactor = 1
	}
	if out.NLayer == 0 || out.NEmbd == 0 || out.NHead == 0 {
		return nil, fmt.Errorf("model: %s: the container's config is empty", out.Arch)
	}
	if err := out.checkSparseMixer(); err != nil {
		return nil, err
	}
	return out, nil
}

// routedScale is what the router multiplies the selected weights by:
// DeepSeek's routed_scaling_factor times Granite's residual scale.
//
// The residual scale rides the routed one because each expert goes straight
// into the residual, so there is no add of its own to scale. The route's
// constructor and every call must agree on it; nn.MoERouteJIT refuses a gate
// whose scale differs from the one it was built for.
func (c *Config) routedScale() float64 { return c.ExpertScale * c.ResidualScale }

// tensor is one weight matrix, kept as the raw mapping plus its shape.
type tensor struct {
	// e is the container entry this weight came from: its role, its block and
	// its shape. Kept for diagnostics and for rows(), never for resolution --
	// a tensor is found through jlm.File.Find, by what it IS.
	e *jlm.Entry
	// typ is a copy of the entry's type, so the hot path never chases e.
	typ  quant.Type
	data []byte
	rows int // ne1, output features
	k    int // ne0, input features
	// kp is the width the weight was packed at when its k is not a whole
	// number of quantization blocks (a tower 144 wide): the converter pads
	// it with zero columns to kp, and every product quantizes its k-wide
	// activation as kp with a zero tail (nn's ragged quantizer on the host,
	// kernels.QuantizePadded on a device). Zero where k is the packed width.
	kp int
	// packed is this weight in the device layout, set for a packed container
	// type; nil for a verbatim one.
	packed *nn.Packed
}

type layer struct {
	attnNorm, ffnNorm []float32 // norms are tiny and always F32; kept expanded
	// attnNormB and ffnNormB are a LayerNorm's biases (C6), nil for RMSNorm
	// and for a bias-free LayerNorm. ffnNorm itself is nil on a parallel block
	// that shares one norm between its two branches (phi-2): the FFN reads the
	// attention's normed row.
	attnNormB, ffnNormB []float32
	// upB and downB are an ungated FFN's biases (starcoder, phi-2). An ungated
	// FFN is one whose gate tensor is absent.
	upB, downB []float32
	// gateB is a gated vision MLP's gate bias (Qwen2.5-VL), nil everywhere else.
	gateB []float32
	// mn is a MobileNet-V5 tower block's own description (mobilenet.go),
	// nil for every other block.
	mn *mnBlock
	// clamp is a vision block's clipped-linear bounds (jlm.RoleVClamp): for
	// q, k, v, o, gate, up and down, the input's and the output's minimum
	// and maximum. nil everywhere but Gemma 4's tower.
	clamp []float32
	// qNorm and kNorm are head_dim wide and nil unless the architecture has
	// them (qwen3). nil is the llama case, not an error path.
	qNorm, kNorm []float32
	// postAttnNorm and postFFNNorm are gemma2/gemma3's extra RMSNorms, applied
	// to the attention and FFN outputs before each is folded into the residual.
	// nil everywhere else.
	postAttnNorm, postFFNNorm []float32
	// attnSubNorm and ffnSubNorm are BitNet's inner RMSNorms: over the
	// attention output before wo, and over act(gate)*up before down. nil
	// everywhere else.
	attnSubNorm, ffnSubNorm []float32
	wq, wk, wv, wo          tensor
	// vFromK says the block has no v projection and v is k's (Gemma 4's
	// global layers): wv is unbound and every path copies k's projection
	// before k's norm and rotary run.
	vFromK bool
	// outScale multiplies the block's output row after its last residual add
	// (Gemma 4's layer_scalar). Zero means none.
	outScale float32
	// xielu is an Apertus block's activation numbers (jlm.RoleXIELU: alpha_p,
	// alpha_n, beta, eps), read where Config.Act is ActXIELU; nil elsewhere.
	xielu []float32
	// Gemma 4's mixture block (Config.DenseMoE): the experts' pre-norm, the
	// dense MLP's and the experts' post-norms, the router input's norm weight
	// and each expert's weight factor. The dense MLP is shGate/shUp/shDown.
	ffnNorm2, postFFNNorm1, postFFNNorm2, routerNorm, expScale []float32
	// Gemma 4's per-layer embedding (Config.PLEDim): the gate from the
	// residual, the projection back and its norm.
	pleGate, pleProj tensor
	plePost          []float32
	// Gemma 3n's AltUp and LAuReL (Config.AltUp, altup.go): the router and
	// its norm, the prediction and correction coefficients transposed, the
	// active stream's scale before the per-layer gate, and LAuReL's two
	// matrices and norm.
	altRouter                                       tensor
	altRouterNorm, altPredT, altCorrT, altCorrScale []float32
	laurelL, laurelR                                tensor
	laurelPost                                      []float32
	// Multi-head Latent Attention, all nil unless Config.KVLoraRank is set.
	// wq is then the one-step query (DeepSeek-V2-Lite, q_lora_rank null) and
	// wqa/wqb the two-step one; exactly one of those is populated.
	//
	// wkb and wvb are banks, one sheet per head, read through bankExpert as an
	// expert bank is, so the absorb runs as a single gather.
	wqa, wqb, wkva, wkb, wvb tensor
	qaNorm, kvaNorm          []float32
	// DeepSeek V3.2's lightning indexer (Config.Indexer): its query from the
	// query latent, its key and its LayerNorm, and the per-head weights.
	idxQB, idxK, idxProj tensor
	idxKNorm, idxKNormB  []float32
	// MiniMax Sparse Attention's query and its per-head norm (msa.go); the
	// key is idxK with idxKNorm.
	idxQ     tensor
	idxQNorm []float32
	// DeepSeek V4 (ds4.go): ds4 holds the block's hyper-connections,
	// compressors and grouped output projection; the query latent is
	// wqa/qaNorm/wqb, the key wk with kvaNorm, the output's second half wo.
	ds4 *ds4Layer
	// q4e is Qwen3.8-Flash-Next's hyper-connections and n-gram weights.
	q4e *q4eLayer
	// Kimi-K3 (k3.go): the residual attention's two score vectors and their
	// one-row F32 views (the generated matvec scores every stream with one
	// call), the MLA output gate (RoleAttnGate in a full block; a linear
	// block's is ssmGate), and the latent mixture's projections and norm.
	resAttn, resFFN      []float32
	resAttnT, resFFNT    tensor
	mlaGate              tensor
	routedDown, routedUp tensor
	routedNorm           []float32
	// kbHead and vbHead are wkb and wvb cut into one view per head, built once
	// at load as l.experts is.
	kbHead, vbHead []tensor
	// headsFrom is what wkb and wvb spanned when the heads were last cut, and
	// banksFrom the same for the expert bank views: bindPacked re-cuts only
	// when a bank moved, since a cut is a fresh nn.Packed per view and a
	// resident block is rebound every token.
	headsFrom [2]bankSpans
	banksFrom [3]bankSpans
	// The attention biases (qwen2 and others); nil where the model has none,
	// and the add is then skipped by one nil check.
	bq, bk, bv, bo []float32
	// sinks is gpt-oss's learned logit per query head, which joins that head's
	// softmax and carries no value. nil everywhere else.
	sinks          []float32
	gate, up, down tensor
	// routerB, expGateB, expUpB and expDownB are gpt-oss's mixture biases: one
	// per expert for the router, and one row per expert, expert-major, for each
	// bank -- expert e's gate bias is expGateB[e*NFFNExp:(e+1)*NFFNExp]. nil for
	// every other mixture.
	routerB, expGateB, expUpB, expDownB []float32
	// expProbsB is DeepSeek's selection bias -- a DIFFERENT tensor from
	// routerB, applied after the gating rather than to the logits. See
	// jlm.RoleExpProbsB.
	expProbsB []float32
	// router is nil unless this layer's FFN is a mixture of experts, in which
	// case gate/up/down hold the three 3-D banks and experts the per-expert
	// views of them. The router tensor's presence is the graph's branch.
	router tensor
	// legacyExperts marks the Mixtral-era layout: eight separate 2-D tensors
	// per role instead of one stacked 3-D bank. gate/up/down are then empty and
	// only l.experts is populated, so offerRange declines the block by name.
	// The converter no longer writes it for a uniform block; see build().
	legacyExperts bool
	experts       []expert
	// ungatedExp is a mixture whose experts have no gate bank (Nemotron 3):
	// each is down(act(up(h))), the activation alone, as an ungated FFN is.
	ungatedExp bool

	// The shared expert runs for every token beside the routed ones. Its gate
	// shRouter is a vector: sigma(dot(x, shRouter)) scales the whole
	// contribution. shRouter is nil where the shared expert is ungated.
	shGate, shUp, shDown tensor
	shRouter             []float32

	// The linear layer's weights. A hybrid's recurrent block projects the
	// residual into one fused q|k|v vector, which lives in wq; the layer kind
	// decides what reads it.
	//
	// ssmGate is the z projection whose SiLU gates the block's output; the full
	// attention layers of the same model fold their gate into wq instead and
	// activate it with sigmoid. nil on every non-hybrid.
	ssmGate, ssmBA, ssmOut tensor
	// ssmIn is a Mamba-2 mixer's x|B|C projection on a block that also runs
	// attention (Falcon-H1), where wq is the attention's query. Every other
	// recurrent block carries it in wq; mixIn picks.
	ssmIn tensor
	// ssmFA/ssmFB and ssmGA/ssmGB are Kimi Delta Attention's two low-rank
	// gates: the forget gate, whose softplus becomes the per-channel decay,
	// and the output gate that replaces ssmGate. nil on qwen3next, which has
	// one matrix for the second and derives the first from ssmBA's interleaved
	// alpha stream.
	ssmFA, ssmFB, ssmGA, ssmGB tensor
	// ssmConv1d is the convolution kernel transposed at load into conv planes
	// of channels, which is the layout the kernel reads; the file's is the
	// other way round. ssmA and ssmDtBias are per value head and ssmNorm is one
	// value head wide, shared by all of them.
	ssmConv1d, ssmA, ssmDtBias, ssmNorm []float32
	// ssmD and ssmConvB are a Mamba-2 block's skip (one per value head) and
	// convolution bias (one per channel, in q | k | v order); nil on the gated
	// delta rule. A Mamba-2 block's ssmNorm is all Inner channels, a weight
	// per group of Inner/Groups.
	ssmD, ssmConvB []float32
	// ssmXDt, ssmXB and ssmXC are a Mamba-1 block's x_proj (the dt
	// bottleneck, B and C, each of the convolved x), and ssmDtProj takes the
	// bottleneck back up to the channels. ssmDtNorm, ssmBNorm and ssmCNorm
	// are Jamba's RMSNorms on the three (FalconMamba's are ones); nil where
	// the block has none.
	ssmXDt, ssmXB, ssmXC, ssmDtProj tensor
	ssmDtNorm, ssmBNorm, ssmCNorm   []float32
	// noFFN is a block that is its mixer alone: plain Mamba-2's, and a
	// Nemotron-H mixer layer not followed by an MLP. It ends at the first
	// residual on every entry point and every tier.
	noFFN bool

	// mtp is a multi-token-prediction block's own weights, nil on every trunk
	// block. See jlm.Config.NMTP.
	mtp *mtpWeights
}

// mtpWeights is what a prediction block has beyond an ordinary block: the two
// input norms and eh_proj that make its residual from the next token's
// embedding and the trunk's hidden state, and its head norm. head and embd are
// the block's own copies where the container kept them (they differed from the
// trunk's); otherwise they are the trunk's, as headNorm is the trunk's output
// norm where the block has none.
type mtpWeights struct {
	ehProj                 tensor
	enorm, hnorm, headNorm []float32
	head, embd             tensor
}

// expert is one of a MoE layer's experts, as three aliasing views of its layer's
// three banks; no copy.
type expert struct{ gate, up, down tensor }

// Model is a loaded transformer.
type Model struct {
	// kpadViolation is a gate's violation of a ragged tower on a device: its
	// plans state no padded width (nn.LayerPlan.KPad), so the device
	// quantizes the rows at their own width under weights packed wider.
	kpadViolation bool
	Cfg           *Config
	Vocab         *tok.Vocab

	// pager serialises decode when the container evicts; see pagelock.go.
	pager sync.Mutex
	// pagerUnlock is pager.Unlock bound once, so enterPager hands back the
	// same func every token instead of allocating a method value.
	pagerUnlock func()
	// bind serialises re-pointing a block's weights at their frames. Rebinding
	// writes only what moved (see sameSpan), so when the model fits the writes
	// stop after the first token and concurrent sessions only ever read.
	bind sync.Mutex
	// sampleChoices is the device sampler's measured verdicts, per key
	// (devsample.go).
	sampleChoices sampleChoices
	// chatTemplates are the chat templates compiled, by source: a request
	// renders its model's template, not compiles it (chat.go).
	chatTemplates sync.Map
	// preloadStop, preloadDone and preloadErr are WithPreload's background
	// read: Close closes the first and waits on the second.
	preloadStop, preloadDone chan struct{}
	preloadErr               error
	// hostUse[li] counts the live States that run block li on the host, under
	// bind. A State placing a block gives its host page back only when the
	// count is zero: placement is per State and the binding per Model, so
	// another State running the block on the host would otherwise find it
	// unbound between its pageIn and its matvecs.
	hostUse []int32
	// pages is the page budget as the caller set it, and the devices on the
	// host's own memory whose holdings come off it (hostheld.go).
	pages pageAsk
	// spare is a closed State's prompt buffers, for the next State's first
	// prompt (promptBuf).
	spareMu sync.Mutex
	spare   *promptBuf
	// step is the host step scratch across sessions (stepBuf).
	step *stepBuf
	// kvPool is the host KV pages closed States gave back (kvPagePool).
	kvPool kvPagePool

	// TokErr is why Vocab is nil, when it is. The weights still loaded.
	TokErr error

	// Container reports that the weights came from a jlm container and are
	// therefore in the device layout, which both tiers read.
	Container bool

	container *jlm.File

	// basePlans is ensureBlock's read plan for every text block -- the block's
	// own weights, without its routed banks -- built once. The entries are
	// fixed at conversion, and building the plan per page-in was a fresh
	// slice, grown several times, for every block of every token.
	basePlanOnce sync.Once
	basePlans    [][]jlm.Range

	// The routed half of the pager's work, counted apart from the block's own
	// page-in: ensureExperts reads sheets nobody can name until the router has
	// run, and jlm.File's whole-run counters cannot tell the two apart. Atomic
	// deltas around the call.
	expWaitNs int64
	// expIONs is the part of expWaitNs that was actually I/O; the difference is
	// the plan (planRuns).
	expIONs  int64
	expReads int64
	// hostExperts is set once a device has run a block's routed experts on
	// the host from inside its submission (hybrid experts): from then on a
	// short prompt takes the decode path (forwardRowsFor).
	hostExperts atomic.Bool
	expBytes    int64
	expCalls    int64
	// hostStepRows is the rows StepRuns has run as host steps across
	// sessions (stepHost); HostStepRows reports it.
	hostStepRows int64

	embd tensor
	// hardEmbd is Gemma 3n's hard vision tokens' rows (mobilenet.go), nil
	// on every other model.
	hardEmbd *hardEmbd
	// rope is the model's rotary configuration, built once at load.
	rope nn.Rope
	// ropeSWA is the LOCAL layers' rotary where an architecture trains two
	// bases, and nil where it trains one -- which is every model but gemma3.
	ropeSWA *nn.Rope
	// ropeB is XD-RoPE's second table: the runs of each NEOX pair's second
	// half (nn.XDRopeRuns), rope's being the first halves'; nil elsewhere.
	ropeB *nn.Rope
	// ds4Head is DeepSeek V4's stream collapse before the head (ds4.go).
	ds4Head *ds4Head
	// q4eHead is Qwen3.8-Flash-Next's head mix and ngram its n-gram table.
	q4eHead *q4eHead
	ngram   *ngramTable
	// resOut is Kimi-K3's head score vector (RoleOutputResScore) and resOutT
	// its one-row view; nil without residual attention.
	resOut  []float32
	resOutT tensor
	// onesHD is a HeadDim-wide vector of ones: the weight Llama 4's weightless
	// q/k RMSNorm runs the generated norm against. nil unless QKL2Norm.
	onesHD  []float32
	outNorm []float32
	// clef is a Clef decision model's joint schema head, loaded by the first
	// Decider (decideclef.go) under bind; nil on every other model.
	clef *clefHead
	// clefDev is that head as a device reads it (decideclefdev.go), built
	// the first time a device runs it.
	clefDev *nn.ClefHead
	// outNormB and outB are the output LayerNorm's bias and the head's bias
	// (phi-2), nil everywhere else.
	outNormB, outB []float32
	// posEmbd is a learned absolute position table (starcoder), one row per
	// position, added to the token's embedding before block 0. Empty on every
	// model that encodes position with a rotary; its presence is the fact.
	posEmbd tensor
	// pleTok and pleProj are Gemma 4's per-layer embedding table and the
	// projection of the scaled embedding to the same NLayer*PLEDim width, and
	// pleNorm the RMSNorm each layer's PLEDim slice of the projection takes
	// (Config.PLEDim). Empty everywhere else.
	pleTok, pleProj tensor
	pleNorm         []float32
	// altProj and altUnembd are Gemma 3n's AltUp projections from the
	// embedding to the other streams and from each back (altup.go).
	altProj   tensor
	altUnembd []tensor
	// lnZeros is the zero bias a device's LayerNorm is handed where the file
	// has none; see lnBias.
	lnZeros     []float32
	lnZerosOnce sync.Once
	output      tensor
	layers      []layer
	tracer      func(layer int, name string, v []float32)
	// jit is what WithJITOptions collected, handed to every State.
	jit []nn.Option
	// opt is this model's resolved load options, a copy; see modelOpts.
	opt modelOpts
	// states counts the sessions built, so AddJITOptions can refuse to be late.
	// Sessions are built concurrently (a server's requests), so it is atomic.
	states atomic.Int64
	// live is the open States, which a prefill asks whether another decodes.
	live live
	// enc is the BERT-family encoder graph, and nil for every decoder. A model
	// with one has no layers, no output and no State: see encoder.go.
	enc *encoder
	// dense1 and dense2 are an embedding model's projection heads after
	// pooling (EmbeddingGemma's), empty for every other model.
	dense1, dense2 tensor
	// tower is the vision tower, nil for a container with no vision section.
	// It shares this model's container, budget and pager.
	tower *Tower
	// audio is the container's audio tower (audio.go), nil without one.
	audio *audioTower
	// imgCache is the vision segment's output rows by picture (imagekey.go),
	// shared by every State of the model.
	imgCache imageCache

	// id names this model to a device (nn.LayerPlan.Model), which holds one
	// model's blocks at a time.
	id uint64
	// forgetters are the devices a State attached that copy weights by
	// address: the pager tells them when a frame's bytes change (forgetCopies).
	// owners are the devices holding this model's blocks, released at Close.
	// Both guarded by devMu, since a page-in on any session reads forgetters.
	devMu sync.Mutex
	// hyState is the host executor a device's hybrid blocks run their routed
	// experts on (Model.hostExperts), made on first use, under hyMu.
	hyMu       sync.Mutex
	hyState    *State
	forgetters []nn.CopyForgetter
	owners     []nn.ModelOwner
}

// modelIDs hands each opened Model its id. Zero is never handed out, so a
// plan that forgot to say whose it is reads as nobody's.
var modelIDs atomic.Uint64

// watch registers d with this model: told when the pager recycles a frame, if
// it copies weights by address, and released at Close, if it holds blocks. It
// reports whether d is told, which is what lets a State offer it a weight in
// a page (State.mv).
func (m *Model) watch(d nn.Device) bool {
	m.devMu.Lock()
	defer m.devMu.Unlock()
	if o, ok := d.(nn.ModelOwner); ok && !slices.Contains(m.owners, o) {
		m.owners = append(m.owners, o)
	}
	f, ok := d.(nn.CopyForgetter)
	if ok && !slices.Contains(m.forgetters, f) {
		m.forgetters = append(m.forgetters, f)
	}
	return ok
}

// forgetCopies is the container's recycle callback (jlm.File.OnRecycle): every
// device that may hold a copy keyed inside [p, p+n) drops it, before the frame
// is read into. See placement.md 15v.
func (m *Model) forgetCopies(p unsafe.Pointer, n uintptr) {
	m.devMu.Lock()
	fs := m.forgetters
	m.devMu.Unlock()
	for _, f := range fs {
		f.Forget(p, n)
	}
}

// Open loads a model from a jlm container. A path that is not a container
// returns a NotConvertedError carrying the convert command.
func Open(path string, options ...Option) (*Model, error) {
	// Any case: the desktop catalog lists "X.JLM" as a container, and refusing it
	// here told the user a converted model still needed converting.
	if !strings.EqualFold(filepath.Ext(path), Ext) {
		return nil, &NotConvertedError{Path: path}
	}
	// Every op a token runs is generated code; with no emitter for this
	// architecture there is nothing to run it on.
	if !nn.Available() {
		return nil, fmt.Errorf("model: jitllm has no code generator for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	// A host the emitter cannot serve (an x86 below SSE4.1+SSSE3) is refused
	// here, naming what is missing, before any page is read.
	if err := nn.HostRefusal(); err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	return openContainer(path, options...)
}

// Option configures a load.
//
// Every placement and residency decision must be forcible by the caller rather
// than inferred. tok.Option is about the vocabulary, so this wraps it rather
// than replacing it.
type Option func(*loadOpts)

type loadOpts struct {
	// opt is the per-model knob set every Option writes into. It starts at the
	// shipping defaults, so an Option that is not passed leaves its knob alone
	// -- which is what a package variable with a default used to give, without
	// the sharing.
	opt modelOpts
	tok []tok.Option
	// jit is handed to nn.NewJIT by every State this model creates. It is how a
	// caller reaches the generated tier's knobs -- the pack width, the tuner,
	// the worker pool -- without the package reading an environment variable.
	jit []nn.Option
	// pageBudget is the host weight budget in BYTES, or zero for "ask the
	// process". See WithPageBudget.
	pageBudget uint64
	budgetSet  bool
	// place is the container's memory placement; see WithHugePages and
	// WithInterleave.
	place jlm.Placement
	// chunk overrides the container's fill granularity; see WithChunk.
	chunk uint64
	// preload is how many pages Open keeps in flight behind it; see
	// WithPreload.
	preload int
}

// WithPreload has Open start reading every page in the background, depth
// pages in flight, so a cold model's first token finds its blocks resident and
// the JIT compiles beside the reads instead of between them (jlm.File.Preload).
// Zero, the default, reads a page only when a token faults it in. It does
// nothing on a budget below the model, where the forward pass's own order
// decides residency.
func WithPreload(depth int) Option {
	return func(l *loadOpts) { l.preload = depth }
}

// WithTokenizerOption carries a tok.Option through Open, for a caller that
// already holds one.
func WithTokenizerOption(o tok.Option) Option {
	return func(l *loadOpts) { l.tok = append(l.tok, o) }
}

// WithTokenizer asserts the model's own tokenizer.json over the name the
// converter wrote -- RULE 7m's fourth and outranking authority.
func WithTokenizer(r io.Reader) Option { return WithTokenizerOption(tok.WithTokenizer(r)) }

// WithJITOptions passes options to the generated-code tier every State of this
// model builds. The JIT always runs (there is no other tier); these configure
// it. See nn.Option: the packages read no environment, so this (and
// cmd/jitllm, which fills it from the JITLLM_* names) is the only way in.
func WithJITOptions(o ...nn.Option) Option {
	return func(l *loadOpts) { l.jit = append(l.jit, o...) }
}

// AddJITOptions appends options for the States this model has NOT built yet.
//
// It exists because one decision cannot be made before Open and must be made
// before NewState: whether this run will page. A model that fits wants the
// worker pool spinning so a region starts the instant the previous one ends;
// one that streams every block off disk wants it parked, because the critical
// path is a synchronous read and spinning cores through it is only heat. The
// caller can weigh that once it knows the weight bytes, after the container is
// open.
//
// It panics rather than silently doing nothing if a State already exists: an
// option that reaches some of a model's sessions and not others is the shape
// RULE 10 is written against.
func (m *Model) AddJITOptions(o ...nn.Option) {
	if m.states.Load() > 0 {
		panic("model: AddJITOptions after a State was built -- the option would reach " +
			"some sessions and not others")
	}
	m.jit = append(m.jit, o...)
}

// WithPageBudget caps the bytes of block pages the container may hold resident.
//
// It has to be known at Open, not after it: when Open read every page before
// SetPageBudget narrowed the pool, a container larger than the machine's
// memory allocated its whole size first and was killed before producing a
// token (placement.md, "Measurements once cited in engine/model's comments").
//
// Zero means "ask the process": sched.MemBudget, which is MemAvailable or the
// cgroup limit, whichever binds, times eight tenths. A budget that holds every
// block never evicts, so one larger than the model is the fully-resident
// configuration and not a second code path.
func WithPageBudget(bytes uint64) Option {
	return func(l *loadOpts) { l.pageBudget, l.budgetSet = bytes, true }
}

// WithHugePages sets whether this model's page frames are advised onto
// transparent huge pages. On by default: the plane layout jumps a whole plane
// per payload word, and on 4 KB pages that walks the page tables on nearly
// every access (docs/engineering-history/cpu-kernels.md, "Huge pages for the
// frames").
func WithHugePages(on bool) Option {
	return func(l *loadOpts) { l.place.NoHugePages = !on }
}

// WithChunk sets this model's pager fill granularity in bytes, overriding the
// container's own: a power of two of at least 4096. Zero keeps the container's.
func WithChunk(bytes uint64) Option {
	return func(l *loadOpts) { l.chunk = bytes }
}

// WithInterleave spreads this model's weight memory page by page over the
// given NUMA nodes. Fewer than two leaves placement to first touch.
func WithInterleave(nodes []int) Option {
	return func(l *loadOpts) { l.place.Nodes = nodes }
}

// Ext is the only weight format the engine runs.
const Ext = jlm.Ext

// NotConvertedError is returned when something other than a container is handed
// to Open.
//
// The engine runs one weight format; GGUF and safetensors are the converter's
// inputs (convert.FromGGUFs, convert.FromSafetensors). The error carries the
// convert command, since that is the next line a caller wants to type.
type NotConvertedError struct{ Path string }

// Error names the file and the convert command that turns it into a
// container.
func (e *NotConvertedError) Error() string {
	out := strings.TrimSuffix(e.Path, filepath.Ext(e.Path)) + Ext
	return fmt.Sprintf("jitllm: %s is not a %s container, and inference reads nothing else.\n"+
		"  Convert it once:  jitllm convert %s %s", e.Path, Ext, e.Path, out)
}

// openContainer loads a converted model.
//
// Every tier reads these bytes: accelerators upload a page untouched and the
// host reads it in place through nn.MatVecPacked, so a block can live on either
// side of the seam with no repack. Pages are faulted in on demand; a budget
// that holds every block simply never evicts.
func openContainer(path string, options ...Option) (*Model, error) {
	lo := loadOpts{opt: defaultOpts()}
	for _, o := range options {
		o(&lo)
	}
	c, err := jlm.Open(path)
	if err != nil {
		return nil, err
	}
	// The budget is applied before the first page is read: with no frame pool,
	// every block gets its own page and nothing is ever evicted.
	budget := lo.pageBudget
	if !lo.budgetSet {
		budget = sched.MemBudget()
	}
	// The budget is bytes, since a container carries page arrays of different
	// sizes; see jlm.File.SetBudget.
	if budget > 0 {
		c.SetBudget(budget)
	}
	c.SetPlacement(lo.place)
	if lo.chunk > 0 {
		if err := c.SetChunk(lo.chunk); err != nil {
			c.Close()
			return nil, err
		}
	}
	// Pages come in during build(), one block at a time, so a container larger
	// than the host can still be opened.
	var m *Model
	if cf := c.Config(); cf != nil && cf.Arch.Encoder() {
		m, err = buildEncoderModel(c, lo.tok...)
	} else {
		if m, err = build(c, lo.tok...); err == nil {
			err = loadDenseHeads(c, m)
		}
	}
	if err != nil {
		c.Close()
		return nil, err
	}
	m.container, m.Container = c, true
	// The open-time budget is the caller's ask like any later one: a device
	// on the host's memory attached afterwards comes off it (hostheld.go).
	m.pages.bytes = budget
	m.jit, m.opt = lo.jit, lo.opt
	if lo.opt.kvTypeSet {
		if err := m.Cfg.KVTypeRefusal(lo.opt.kvType); err != nil {
			c.Close()
			return nil, err
		}
	}
	m.id = modelIDs.Add(1)
	c.OnRecycle(m.forgetCopies)
	for i := range m.layers {
		if err := bindPacked(c, m.Cfg, &m.layers[i]); err != nil {
			c.Close()
			return nil, err
		}
	}
	bindOne(c, &m.embd)
	bindOne(c, &m.posEmbd)
	bindOne(c, &m.output)
	bindOne(c, &m.dense1)
	bindOne(c, &m.dense2)
	bindOne(c, &m.pleTok)
	bindOne(c, &m.pleProj)
	bindOne(c, &m.altProj)
	for i := range m.altUnembd {
		bindOne(c, &m.altUnembd[i])
	}
	if m.q4eHead != nil {
		for _, t := range m.q4eHead.hc.tensors() {
			bindOne(c, t)
		}
	}
	// A container that carries a tower builds one. Nothing is faulted in: the
	// tower's norms and biases are in the dense region like the text model's,
	// and its matrices are bound by Tower.pageIn as Encode reaches them.
	if c.Vision() != nil && c.VisionBlocks() > 0 {
		tw, err := buildTower(c)
		if err != nil {
			c.Close()
			return nil, err
		}
		if tw.Cfg.ProjDim != m.Cfg.NEmbd {
			c.Close()
			return nil, fmt.Errorf("model: the tower projects to %d and this model wants %d; "+
				"the container was built from two files that do not belong together",
				tw.Cfg.ProjDim, m.Cfg.NEmbd)
		}
		tw.container, tw.jit, tw.opt, tw.model = c, lo.jit, &m.opt, m
		for _, w := range tw.weights() {
			bindOne(c, w)
		}
		// The vision blocks join the model's block space where the container
		// already puts them: block NBlocks+i is vision block i's page, so the
		// pager, the placement and the runner address it as any block.
		tw.base, tw.cfg = int(c.H.NBlocks), tw.Cfg.segConfig()
		if len(m.layers) != tw.base {
			c.Close()
			return nil, fmt.Errorf("model: %d text and prediction blocks, and the container's vision "+
				"blocks start at %d", len(m.layers), tw.base)
		}
		m.layers = append(m.layers, tw.pending...)
		tw.pending = nil
		for i := tw.base; i < len(m.layers); i++ {
			if err := bindPacked(c, m.Cfg, &m.layers[i]); err != nil {
				c.Close()
				return nil, err
			}
		}
		m.tower = tw
		m.imgCache.budget = defaultImageCache
		if m.audio, err = buildAudio(c, m); err != nil {
			c.Close()
			return nil, err
		}
		// The audio blocks take their places in the block space after the
		// vision blocks, as the container pages them; their weights are the
		// tower's own (audioPageIn).
		if m.audio != nil {
			m.layers = append(m.layers, make([]layer, len(m.audio.blocks))...)
		}
		if tw.mn != nil && tw.mn.hard != nil {
			m.hardEmbd = tw.mn.hardRows(m.Cfg.EmbdScale)
		}
	}
	if lo.preload > 0 {
		m.startPreload(lo.preload)
	}
	return m, nil
}

// startPreload runs the container's Preload behind Open. Close stops it and
// waits for it, so no read lands in a frame Close has unmapped.
func (m *Model) startPreload(depth int) {
	m.preloadStop, m.preloadDone = make(chan struct{}), make(chan struct{})
	go func() {
		defer close(m.preloadDone)
		m.preloadErr = m.container.Preload(depth, m.preloadStop)
	}()
}

// WaitPreload blocks until a WithPreload read has finished and returns its
// error; nil at once when there was none. A failed preload is not fatal: the
// page it could not read is read again, and its error returned, when a token
// faults it in.
func (m *Model) WaitPreload() error {
	if m.preloadDone == nil {
		return nil
	}
	<-m.preloadDone
	return m.preloadErr
}

// ReadConcurrency is the most read requests the container ever had
// outstanding at once, and ReadBytes the bytes they asked for.
func (m *Model) ReadConcurrency() int64 {
	if m.container == nil {
		return 0
	}
	return m.container.ReadConcurrency()
}

// PreloadDepth is the most pages a WithPreload read had in flight at once.
func (m *Model) PreloadDepth() int64 {
	if m.container == nil {
		return 0
	}
	return m.container.PreloadDepth()
}

func (m *Model) ReadBytes() int64 {
	if m.container == nil {
		return 0
	}
	return m.container.ReadBytes()
}

// Tower is the vision tower this container carries, or nil for a text-only
// model. It is not a second model: it shares this one's container, pager and
// budget.
func (m *Model) Tower() *Tower { return m.tower }

// PagerReads is how many read calls the pager has issued. A streamed mixture is
// priced in requests rather than bytes, since a routed bank asks for many small
// scattered ranges.
func (m *Model) PagerReads() int64 {
	if m.container == nil {
		return 0
	}
	return m.container.Reads()
}

// FreshFrames is how many frames the pager allocated rather than reused: on a
// steady run it stops growing once the budget is full, and a figure that keeps
// pace with the page-ins is frames being freed and remade.
func (m *Model) FreshFrames() int64 {
	if m.container == nil {
		return 0
	}
	return m.container.FreshFrames()
}

// StreamGroups is the container's own answer to how many pieces a streamed
// block's selection should be uploaded in. See jlm.streamGroupsFor.
func (m *Model) StreamGroups() int {
	if m.container == nil {
		return 0
	}
	return m.container.StreamGroups()
}

// ChunkBytes is the granularity this container was opened at.
func (m *Model) ChunkBytes() uint64 {
	if m.container == nil {
		return 0
	}
	return m.container.ChunkBytes()
}

// BytesRead is what those chunks actually cost, using this container's
// granularity rather than the package global. See jlm.File.ChunkBytes.
func (m *Model) BytesRead() uint64 {
	if m.container == nil {
		return 0
	}
	return uint64(m.container.ChunksIn()) * m.container.ChunkBytes()
}

// DirectIO reports whether this model's pages are read without going through
// the kernel's page cache, which decides how much of a memory ceiling a weight
// budget may claim. See jlm.File.Direct.
func (m *Model) DirectIO() bool { return m.container != nil && m.container.Direct() }

// HostResidentBlocks is how many of this model's block pages the host is
// currently holding. RSS cannot stand in for it: a released page shows only
// when the collector runs, and the KV cache dwarfs the weights anyway.
func (m *Model) HostResidentBlocks() int {
	if m.container == nil {
		return 0
	}
	n := 0
	for i := 0; i < len(m.layers); i++ {
		if m.container.Resident(i) {
			n++
		}
	}
	return n
}

// Blocks is the size of the model's block space: its text blocks, any
// prediction blocks, then its vision tower's, which are blocks like any other
// -- placed, paged and relocated the same way.
func (m *Model) Blocks() int { return len(m.layers) }

// BlockResident reports whether block li's page is in host memory now. A block
// that runs on the host and is not resident is read from the file on its next
// pass.
func (m *Model) BlockResident(li int) bool {
	return m.container != nil && m.container.Resident(li)
}

// SetPageBudget caps how many block pages may be resident, in bytes. Paging
// does not turn on or off: a budget that holds every block simply never
// evicts, so "fully resident" and "one page short" are the same code path.
//
// bytes is the host's share before a device on the host's own memory (an
// integrated GPU, Apple Silicon) takes what it holds: the model subtracts that
// itself, now and whenever a State's placement moves it (hostheld.go), so a
// caller never subtracts tier.GPU.HostReserved again.
func (m *Model) SetPageBudget(bytes uint64) int {
	if m.container == nil {
		return 0
	}
	m.pages.mu.Lock()
	m.pages.bytes = bytes
	m.applyPageBudget(m.pageNet())
	m.pages.mu.Unlock()
	return m.container.ResidentPages()
}

// PageSize is one block page's size in bytes, or zero for a model with no
// container behind it.
func (m *Model) PageSize() uint64 {
	if m.container == nil {
		return 0
	}
	return m.container.H.PageSize
}

// blockBytes is what block li's tensors occupy in the container's device
// layout, and the routed expert banks' share of that, from the tensor table
// alone: nothing is read. Zero for a model with no container.
func (m *Model) blockBytes(li int) (total, bank uint64) {
	if m.container == nil {
		return 0, 0
	}
	es := m.container.Entries()
	for i := range es {
		e := &es[i]
		if e.Block == jlm.DenseBlock || m.container.BlockOf(e) != li {
			continue
		}
		q, d, sc := jlm.SpanLenOf(e)
		total += q + d + sc
		if jlm.ExpertBank(e.Role) {
			bank += q + d + sc
		}
	}
	return total, bank
}

// meanBase is the mean of the mixture blocks' bytes without their routed
// banks, from the tensor table.
func (m *Model) meanBase() uint64 {
	var sum uint64
	n := 0
	for li := 0; li < m.Cfg.NLayer; li++ {
		total, bank := m.blockBytes(li)
		if bank == 0 {
			continue // only the mixture blocks stream
		}
		sum += total - bank
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / uint64(n)
}

// DenseBytes is what the non-block weights cost. They never page: the
// embedding and the output projection are read by the host on every token.
func (m *Model) DenseBytes() uint64 {
	if m.container == nil {
		return 0
	}
	var n uint64
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		// The same predicate jlm.alwaysResident uses: expanded tensors carry a
		// block number but live in the dense region.
		if e.Block != jlm.DenseBlock && !e.Role.Expanded() {
			continue
		}
		qs, d, sc := m.container.Span(e)
		n += uint64(len(qs) + len(d) + len(sc))
	}
	return n
}

// HostBytes is how many bytes of block pages this model is holding in host
// memory right now, and PageBudget the cap it is held against (0 when
// unlimited).
//
// Bytes, because page sizes differ per model and between a container's page
// arrays, so neither a page count nor frames * PageSize() is comparable. jlm
// keeps the running total against the budget it enforces.
func (m *Model) HostBytes() uint64 {
	if m.container == nil {
		return 0
	}
	return m.container.ResidentBytes()
}

// PageBudget is the byte cap the container holds its block pages against, 0
// when unlimited (see HostBytes).
func (m *Model) PageBudget() uint64 {
	if m.container == nil {
		return 0
	}
	return m.container.Budget()
}

// PagesFit reports whether the page budget holds every page of the container,
// so that nothing is ever evicted. A model with no container has nothing to
// page. It is the budget's answer, not the residency's: a model that fits may
// not have faulted a page in yet.
func (m *Model) PagesFit() bool {
	return m.container == nil || !m.container.CanEvict()
}

// PageBytes is block li's own page size, which differs between a text block
// and a vision block in the same container.
func (m *Model) PageBytes(li int) uint64 {
	if m.container == nil {
		return 0
	}
	return m.container.PageBytes(li)
}

// PageStats is the budget in pages and what it has cost so far.
func (m *Model) PageStats() (frames int, in, out int64) {
	if m.container == nil {
		return 0, 0, 0
	}
	in, out = m.container.Faults()
	return m.container.ResidentPages(), in, out
}

// pageIn faults block li in and re-points its weights at the frame it landed
// in. A resident block costs one bounds check and one nil compare.
//
// A frame is reused, so after an eviction block i's bytes are elsewhere and
// anything holding the old slice reads another block's weights.
func (m *Model) pageIn(li int) error {
	// Neither "no pool" nor Resident(li) means this block's bytes are in:
	// Resident only says a frame is allocated, and a partial read (a streamed
	// block's expert sheets) can claim one with the block's own weights unread.
	// So always fall through to ensureBlock, which reads nothing when every
	// chunk is filled; that is what makes a partial read composable.
	//
	// A model can have no block pages at all: an all-F32 model keeps
	// everything in the dense region (jlm.alwaysResident).
	if m.container == nil || li >= len(m.layers) || li >= m.container.NPages() {
		return nil
	}
	// An encoder's blocks are its own structs (encoder.go, modernbert.go);
	// a tower's after them (d1-omni's) are layers like any model's, and an
	// audio tower's its own (audio.go).
	if m.audio != nil && li >= m.audioFirst() {
		return m.audioPageIn(li - m.audioFirst())
	}
	if m.enc != nil && li < m.encBlocks() {
		return m.encBlockPageIn(li)
	}
	// The block's own weights, not its experts: moe() ensures those after the
	// router has chosen. EnsureRanges tracks which chunks hold real bytes, so an
	// expert nobody ensured is absent rather than stale, and every consumer of
	// a packed span has to ensure it.
	if err := m.ensureBlock(li); err != nil {
		return err
	}
	m.bind.Lock()
	defer m.bind.Unlock()
	return bindPacked(m.container, m.Cfg, &m.layers[li])
}

// pageInSelected is pageIn for the device's callback: the frame, the routed
// sheets and the bindings, but not the block's own weights, which a placed
// block already has on the card.
//
// The read comes before the bind: bindOne returns early on an absent span
// without clearing old pointers, so binding an unclaimed frame would leave the
// previous occupant's spans in place. EnsureRanges claims the frame.
//
// The consumer is tier.streamBank.fill. A block that demotes goes back through
// ensureLayer, the whole-block read.
func (m *Model) pageInSelected(li int, sel []uint32) error {
	if m.container == nil || li >= int(m.container.H.NBlocks) {
		return nil
	}
	// A bank in expert pages reaches the device through Sheet, which holds
	// each selected expert's own page; the block page has nothing it needs.
	if l := &m.layers[li]; l.expBank().e != nil && m.container.ExpertPaged(l.expBank().e) {
		return nil
	}
	rs, err := m.selectedRanges(li, sel, false)
	if err != nil {
		return err
	}
	// No bank to narrow to -- an attention-only block, or a model whose experts
	// are always-resident. There is nothing to save and the frame still has to
	// be claimed, so this is an ordinary page-in.
	if len(rs) == 0 {
		return m.pageIn(li)
	}
	if err := m.container.EnsureRanges(li, rs); err != nil {
		return err
	}
	m.bind.Lock()
	defer m.bind.Unlock()
	return bindPacked(m.container, m.Cfg, &m.layers[li])
}

// Close gives the model's blocks back to every device holding them, then
// closes the container and drops every copy keyed on it. Every State of the
// model must be closed first.
func (m *Model) Close() error {
	if m.container == nil {
		return nil
	}
	if m.preloadStop != nil {
		close(m.preloadStop)
		<-m.preloadDone
		m.preloadStop = nil
	}
	// The devices give this model's blocks back first, so a device another
	// model is offered next is free; then the container's memory goes, and
	// every copy keyed on it with it (forgetCopies).
	m.devMu.Lock()
	owners := m.owners
	m.devMu.Unlock()
	for _, o := range owners {
		o.ReleaseModel(m.id)
	}
	err := m.container.Close()
	m.devMu.Lock()
	m.forgetters, m.owners = nil, nil
	m.devMu.Unlock()
	return err
}

// name is what to call this weight in an error. It is the container's
// diagnostic label, and nothing resolves through it.
func (t tensor) name() string {
	if t.e == nil {
		return "a weight"
	}
	if t.e.Name != "" {
		return t.e.Name
	}
	return t.e.Role.String()
}

// bindOne points a tensor at its already-packed spans in the container.
//
// data is set to the payload span, not left nil: the tier keys residency on
// &data[0] plus the shape (resKey), and several paths read an empty Data as
// "no weight here".
func bindOne(c *jlm.File, t *tensor) {
	if t == nil || t.e == nil {
		return
	}
	e := t.e
	qs, d, sc := c.Span(e)
	if len(qs) == 0 {
		return
	}
	// Only what moved is written: on a model that fits the answer never
	// changes, and an unconditional write races another session's matvec
	// reading the same field (TestConcurrentDecodeOnAFittingMixture).
	if sameSpan(t.data, qs) && (t.packed == nil || !jlm.Packed(e.Type) ||
		(sameSpan(t.packed.QS, qs) && sameSpan(t.packed.D, d) && sameSpan(t.packed.SC, sc))) {
		return
	}
	t.data = qs
	// A type the container carries verbatim (F32/F16 norms, biases, routers)
	// is not packed; labelling it packed gives wrong numbers.
	if !jlm.Packed(e.Type) {
		return
	}
	// Reused, not reallocated: a fresh nn.Packed per fault would be garbage
	// proportional to the paging rate.
	if t.packed == nil {
		t.packed = &nn.Packed{}
	}
	t.packed.QS, t.packed.D, t.packed.SC = qs, d, sc
}

// sameSpan reports whether two slices are the same memory, not merely equal
// bytes: a rebind that lands on the frame it was already bound to.
func sameSpan(a, b []byte) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

// sameView is sameSpan for a whole weight: its payload and, if packed, its
// three planes and its shape.
func sameView(a, b tensor) bool {
	if a.rows != b.rows || a.k != b.k || !sameSpan(a.data, b.data) || (a.packed == nil) != (b.packed == nil) {
		return false
	}
	return a.packed == nil || (sameSpan(a.packed.QS, b.packed.QS) && sameSpan(a.packed.D, b.packed.D) &&
		sameSpan(a.packed.SC, b.packed.SC) && a.packed.Row == b.packed.Row && a.packed.Stride == b.packed.Stride)
}

// ensureBlock reads the parts of block li that every token needs: its
// attention or recurrent weights, its router, its shared expert -- everything
// but the routed expert banks.
func (m *Model) ensureBlock(li int) error {
	m.basePlanOnce.Do(m.buildBasePlans)
	if li >= 0 && li < len(m.basePlans) {
		return m.container.EnsureRanges(li, m.basePlans[li])
	}
	return m.ensureBlockRanges(li, false)
}

// buildBasePlans fills basePlans with ensureBlockRanges' plan for each block,
// the vision tower's included. EnsureRanges only reads a plan, so one slice
// serves every session.
func (m *Model) buildBasePlans() {
	c := m.container
	if c == nil {
		return
	}
	m.basePlans = make([][]jlm.Range, len(m.layers))
	for li := range m.basePlans {
		m.basePlans[li] = m.blockPlan(li, false)
	}
}

// ensureBlockExperts reads the whole block, routed expert banks included. A
// device is handed the entire bank at PrepLayer and cannot come back for an
// expert later, so it must be read first; otherwise the upload takes whatever
// the frame held (zeros, or another block's bytes). See
// docs/engineering-history/placement.md 15j.
func (m *Model) ensureBlockExperts(li int) error { return m.ensureBlockRanges(li, true) }

func (m *Model) ensureBlockRanges(li int, experts bool) error {
	return m.container.EnsureRanges(li, m.blockPlan(li, experts))
}

// blockPlan is the ranges of block li's page that ensureBlockRanges reads.
func (m *Model) blockPlan(li int, experts bool) []jlm.Range {
	c := m.container
	var rs []jlm.Range
	// A vision block's entries say Block i of the vision section; it is block
	// NBlocks+i of the one block space, the page they live in.
	vision, bi := li >= int(c.H.NBlocks), int32(li)
	if vision {
		bi = int32(li - int(c.H.NBlocks))
	}
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Role.Vision() != vision || e.Block != bi || e.Role.Expanded() {
			continue
		}
		// A bank in expert pages is not in this page at all: moe() holds the
		// sheets a token selects, and a device reads each through Sheet.
		if c.ExpertPaged(e) {
			continue
		}
		switch e.Role {
		case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank,
			jlm.RoleExpGate, jlm.RoleExpUp, jlm.RoleExpDown:
			if !experts {
				continue // routed experts: moe() ensures the ones it selected
			}
		}
		rs = appendSpans(rs, c, e)
	}
	return rs
}

// ensureBlockSelected is ensureBlockExperts narrowed to the experts a token
// actually routed to: the block's own weights in full, and of each bank only
// the listed sheets.
//
// A streamed device block asks for its bank right after the router has chosen,
// so reading only the selected sheets is what keeps the device path from
// reading the whole bank every token.
func (m *Model) ensureBlockSelected(li int, sel []uint32, base bool) error {
	// A bank in expert pages is warmed page by page: the reads land in frames
	// that stay cached across tokens, which is the point of splitting them.
	if m.container != nil && li < len(m.layers) && m.layers[li].expBank().e != nil &&
		m.container.ExpertPaged(m.layers[li].expBank().e) {
		bank := m.layers[li].expBank().e
		pages := make([]int, len(sel))
		for i, x := range sel {
			if pages[i] = m.container.ExpertPage(bank, int(x)); pages[i] < 0 {
				return fmt.Errorf("model: block %d: expert %d has no expert page", li, x)
			}
		}
		var wg sync.WaitGroup
		errs := make([]error, len(sel))
		for i, p := range pages {
			wg.Add(1)
			go func() {
				defer wg.Done()
				errs[i] = m.container.EnsurePage(p)
			}()
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				return err
			}
		}
		if base {
			return m.ensureBlock(li)
		}
		return nil
	}
	rs, err := m.selectedRanges(li, sel, base)
	if err != nil {
		return err
	}
	if rs == nil {
		return nil
	}
	return m.container.EnsureRanges(li, rs)
}

// selectedRanges is ensureBlockSelected's plan over the block page: the base
// when asked for, and a legacy per-expert tensor when selected. Since v26 a
// bank is never in it; its sheets are expert pages.
func (m *Model) selectedRanges(li int, sel []uint32, base bool) ([]jlm.Range, error) {
	c := m.container
	if c == nil || li >= int(c.H.NBlocks) {
		return nil, nil
	}
	var rs []jlm.Range
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Role.Vision() || e.Block != int32(li) || e.Role.Expanded() {
			continue
		}
		// A bank in expert pages is not in this block's page: its sheets are
		// held, one expert page each, by whoever reads them (Sheet).
		if c.ExpertPaged(e) {
			continue
		}
		switch e.Role {
		case jlm.RoleExpGate, jlm.RoleExpUp, jlm.RoleExpDown:
			// The legacy per-expert layout: each expert is its own tensor, so
			// "only the routed ones" is a filter on Index.
			if selected(sel, e.Index) {
				rs = appendSpans(rs, c, e)
			}
		default:
			// The base is skipped for a block on a device: it was uploaded at
			// placement, binding needs only addresses, and pageIn re-reads it
			// for a later host consumer.
			if base {
				rs = appendSpans(rs, c, e)
			}
		}
	}
	return rs, nil
}

func selected(sel []uint32, idx int32) bool {
	for _, x := range sel {
		if int32(x) == idx {
			return true
		}
	}
	return false
}

// appendSpans adds an entry's three spans to a plan.
func appendSpans(rs []jlm.Range, c *jlm.File, e *jlm.Entry) []jlm.Range {
	qs, d, sc := c.SpanLen(e)
	for _, sp := range []jlm.Range{{Off: e.QSOff, N: qs}, {Off: e.DOff, N: d}, {Off: e.SCOff, N: sc}} {
		if sp.N != 0 {
			rs = append(rs, sp)
		}
	}
	return rs
}

// ensureExperts reads the sheets of every selected expert of block li, all in
// flight at once, so an NVMe works at queue depth rather than at its depth-1
// rate (neurostream's "plan once per layer"). The pages are held in h until the
// caller's h.release, which it must call whatever this returns.
func (m *Model) ensureExperts(li int, sel []int32, h *expertHold) error {
	c, cfg := m.container, m.Cfg
	if c == nil || !cfg.MoEAt(li) || len(sel) == 0 {
		return nil
	}
	l := &m.layers[li]
	if l.legacyExperts {
		return m.ensureLegacyExperts(li, sel, h)
	}
	bank := l.expBank().e
	if bank == nil || !c.ExpertPaged(bank) {
		return nil
	}
	// The counters bracket the reads; see ExpertIO.
	t0 := time.Now()
	r0, b0, w0 := c.Reads(), c.ChunksIn(), c.ReadWait()
	defer func() {
		atomic.AddInt64(&m.expWaitNs, int64(time.Since(t0)))
		atomic.AddInt64(&m.expIONs, int64(c.ReadWait()-w0))
		atomic.AddInt64(&m.expReads, c.Reads()-r0)
		atomic.AddInt64(&m.expBytes, (c.ChunksIn()-b0)*int64(c.ChunkBytes()))
		atomic.AddInt64(&m.expCalls, 1)
	}()
	// Every page this layer reads is held until the caller is done, the block's
	// own page included: with every expert page pinned the victim rule falls
	// back to the most recently used block page, which is this layer's own. And
	// one selected expert's fault must not take another's page.
	if li < int(c.H.NBlocks) {
		// Only when it is in: a block whose base runs on a device gave its
		// page back, and claiming a frame for it here would hold a block's
		// worth of host budget empty for the experts' sake.
		if lb, ok := c.HoldFilled(li); ok {
			h.leases = append(h.leases, lb)
		}
	}
	// One read per expert page, all in flight at once. Every page is resolved
	// before any read starts, so a bad id cannot leave reads in flight holding
	// pins nobody releases.
	h.pages = h.pages[:0]
	for _, x := range sel {
		p := c.ExpertPage(bank, int(x))
		if p < 0 {
			return fmt.Errorf("model: block %d: expert %d has no expert page", li, x)
		}
		h.pages = append(h.pages, p)
	}
	// A page already resident is pinned inline. Only a cold one goes out as a
	// read of its own: on a model whose experts stay resident that is none of
	// them, and a goroutine per selected expert per layer was most of a
	// mixture token's allocations.
	h.held = slices.Grow(h.held[:0], len(sel))[:len(sel)]
	h.errs = slices.Grow(h.errs[:0], len(sel))[:len(sel)]
	for i, p := range h.pages {
		h.errs[i] = nil
		if lp, ok := c.HoldFilled(p); ok {
			h.held[i] = lp
			continue
		}
		off, size := c.PageBounds(p)
		h.wg.Add(1)
		go func() {
			defer h.wg.Done()
			h.held[i], h.errs[i] = c.Hold(p, []jlm.Range{{Off: off, N: size}})
		}()
	}
	h.wg.Wait()
	for i := range h.held {
		if h.errs[i] == nil {
			h.leases = append(h.leases, h.held[i])
		}
	}
	for _, err := range h.errs {
		if err != nil {
			return err
		}
	}
	m.bind.Lock()
	defer m.bind.Unlock()
	for _, x := range sel {
		v := &l.experts[x]
		bindSheet(c, &l.gate, cfg.NFFNExp, int(x), &v.gate)
		bindSheet(c, &l.up, cfg.NFFNExp, int(x), &v.up)
		bindSheet(c, &l.down, cfg.ExpWidth(), int(x), &v.down)
	}
	return nil
}

// expertHold is the pages one layer's routed experts are held under until the
// caller's FFN is done with them. A State keeps one and reuses its slices, so
// holding a token's selection allocates nothing once they have grown.
type expertHold struct {
	leases []jlm.Lease
	pages  []int
	held   []jlm.Lease
	errs   []error
	// wg is here and not on the stack because the reads capture it.
	wg sync.WaitGroup
}

// release drops every pin taken since the last release.
func (h *expertHold) release() {
	for i := range h.leases {
		h.leases[i].Release()
	}
	h.leases = h.leases[:0]
}

// ensureLegacyExperts is ensureExperts for a block whose experts are separate
// tensors in the block's own page.
//
// pageIn skips the routed experts, so without this nothing would read them. The
// converter no longer writes this layout for a uniform block; this keeps an
// old container correct.
func (m *Model) ensureLegacyExperts(li int, sel []int32, h *expertHold) error {
	c := m.container
	u := make([]uint32, len(sel))
	for i, x := range sel {
		u[i] = uint32(x)
	}
	rs, err := m.selectedRanges(li, u, false)
	if err != nil {
		return err
	}
	lb, err := c.Hold(li, rs)
	if err != nil {
		return err
	}
	h.leases = append(h.leases, lb)
	// Re-bound under the page just held: bindOne resolves a whole tensor, and
	// the frame may not be the one pageIn bound it to.
	m.bind.Lock()
	l := &m.layers[li]
	for _, x := range sel {
		bindOne(c, &l.experts[x].gate)
		bindOne(c, &l.experts[x].up)
		bindOne(c, &l.experts[x].down)
	}
	m.bind.Unlock()
	return nil
}

// expBank is the bank a mixture block's expert pages are addressed through:
// gate where the experts have one, up where they are ungated. Every bank of a
// block is split the same way, so any of them names the same pages.
func (l *layer) expBank() *tensor {
	if l.gate.e != nil {
		return &l.gate
	}
	return &l.up
}

// bindSheet points expert x's view v of bank b at the sheet in its expert
// page. The page must be held: the slices are a frame the pager reuses.
func bindSheet(c *jlm.File, b *tensor, nrow, x int, v *tensor) {
	if b.e == nil {
		return
	}
	qs, d, sc := c.Sheet(b.e, x)
	if v.rows == nrow && sameSpan(v.data, qs) && (b.packed == nil || (v.packed != nil &&
		sameSpan(v.packed.QS, qs) && sameSpan(v.packed.D, d) && sameSpan(v.packed.SC, sc))) {
		return // already bound to this page: a reader may be using it
	}
	v.e, v.typ, v.rows, v.k, v.data = b.e, b.typ, nrow, b.k, qs
	if b.packed == nil {
		return
	}
	if v.packed == nil {
		v.packed = &nn.Packed{}
	}
	*v.packed = nn.Packed{QS: qs, D: d, SC: sc, Row: 0, Stride: nrow}
}

// ExpertIO is what the routed expert read has cost: wall the decode path spent
// blocked on it, the I/O part of that, readAt calls, bytes, and how many
// layer-steps asked. On a model that fits it is a warm-up that converges to
// zero; see docs/engineering-history/scheduling-and-measurement.md.
func (m *Model) ExpertIO() (wait, io time.Duration, reads, bytes, calls int64) {
	if m == nil {
		return 0, 0, 0, 0, 0
	}
	return time.Duration(atomic.LoadInt64(&m.expWaitNs)),
		time.Duration(atomic.LoadInt64(&m.expIONs)),
		atomic.LoadInt64(&m.expReads), atomic.LoadInt64(&m.expBytes),
		atomic.LoadInt64(&m.expCalls)
}

// PagerReadWait is how long callers have been blocked in the pager overall,
// block page-ins and routed sheets. See jlm.File.ReadWait.
func (m *Model) PagerReadWait() time.Duration {
	if m == nil || m.container == nil {
		return 0
	}
	return m.container.ReadWait()
}

// hostRuns adds d to hostUse over blocks [lo, hi).
func (m *Model) hostRuns(lo, hi int, d int32) {
	m.bind.Lock()
	defer m.bind.Unlock()
	if m.hostUse == nil {
		m.hostUse = make([]int32, len(m.layers))
	}
	for li := max(lo, 0); li < min(hi, len(m.layers)); li++ {
		m.hostUse[li] += d
	}
}

// releaseLayer drops a block's host residency: the container's page and every
// slice of it this layer holds. DropPage alone frees nothing, since the
// layer's spans point into the page and keep its backing array alive.
//
// It is the inverse of bindPacked; pageIn rebinds on the next fault.
func releaseLayer(c *jlm.File, li int, l *layer, keepExperts bool) {
	for _, t := range []*tensor{
		&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down, &l.router,
		&l.shGate, &l.shUp, &l.shDown,
		&l.ssmGate, &l.ssmBA, &l.ssmOut, &l.ssmIn,
		&l.wqa, &l.wqb, &l.wkva, &l.wkb, &l.wvb,
		&l.mlaGate, &l.routedDown, &l.routedUp,
		&l.ssmFA, &l.ssmFB, &l.ssmGA, &l.ssmGB, &l.ssmXDt, &l.ssmXB, &l.ssmXC, &l.ssmDtProj,
		&l.pleGate, &l.pleProj,
		&l.idxQB, &l.idxK, &l.idxProj, &l.idxQ,
		&l.altRouter, &l.laurelL, &l.laurelR,
	} {
		t.data = nil
		if t.packed != nil {
			t.packed.QS, t.packed.D, t.packed.SC = nil, nil, nil
		}
	}
	if l.q4e != nil {
		for _, t := range l.q4e.tensors() {
			t.data = nil
			if t.packed != nil {
				t.packed.QS, t.packed.D, t.packed.SC = nil, nil, nil
			}
		}
	}
	if l.ds4 != nil {
		for _, t := range append(l.ds4.tensors(), &l.wqa, &l.wqb) {
			t.data = nil
			if t.packed != nil {
				t.packed.QS, t.packed.D, t.packed.SC = nil, nil, nil
			}
		}
		for i := range l.ds4.woAG {
			t := &l.ds4.woAG[i]
			t.data = nil
			if t.packed != nil {
				t.packed.QS, t.packed.D, t.packed.SC = nil, nil, nil
			}
		}
		l.ds4.woAFrom = bankSpans{}
	}
	if w := l.mtp; w != nil {
		for _, t := range []*tensor{&w.ehProj, &w.head, &w.embd} {
			if t.e == nil || t.e.Block == jlm.DenseBlock {
				continue // the trunk's, in the dense region
			}
			t.data = nil
			if t.packed != nil {
				t.packed.QS, t.packed.D, t.packed.SC = nil, nil, nil
			}
		}
	}
	// MLA's per-head views are slices of wkb and wvb, as the expert views
	// below are of the banks.
	for i := range l.kbHead {
		for _, t := range []*tensor{&l.kbHead[i], &l.vbHead[i]} {
			t.data = nil
			if t.packed != nil {
				t.packed.QS, t.packed.D, t.packed.SC = nil, nil, nil
			}
		}
	}
	for i := range l.experts {
		for _, t := range []*tensor{&l.experts[i].gate, &l.experts[i].up, &l.experts[i].down} {
			t.data = nil
			if t.packed != nil {
				t.packed.QS, t.packed.D, t.packed.SC = nil, nil, nil
			}
		}
	}
	// The views are gone, so the next bind cuts them again even if the banks
	// land where they were.
	l.headsFrom, l.banksFrom = [2]bankSpans{}, [3]bankSpans{}
	c.DropPage(li)
	// A bank in expert pages is not in the block's page: since v26 each expert
	// is a page of its own, and on a mixture those are nearly all of the
	// weights (Qwen3-30B-A3B: 18.0 of 18.5 GiB). The device holds the whole
	// bank, so they go too, and their frames serve the next block's upload --
	// unless the block's experts still run from the host's pages (keepExperts:
	// a streamed or hybrid block), which would read them all again.
	if e := l.expBank().e; e != nil && c.ExpertPaged(e) && !keepExperts {
		for x := 0; x < int(e.Dims[2]); x++ {
			c.DropPage(c.ExpertPage(e, x))
		}
	}
}

// addPos adds position pos's row of the learned position table to dst, and
// is a no-op for every model that has none (starcoder is the one that does).
// tmp is a scratch row of NEmbd; the caller owns it.
func (m *Model) addPos(dst, tmp []float32, pos int) error {
	e := &m.posEmbd
	if e.e == nil {
		return nil
	}
	if pos < 0 || pos >= e.rows {
		return fmt.Errorf("model: position %d is past the %d-row position table", pos, e.rows)
	}
	if e.packed != nil && len(e.packed.QS) != 0 {
		nn.RowPacked32JIT(tmp, e.typ, e.packed, pos, e.rows, e.k)
	} else {
		nn.Row32JIT(tmp, e.typ, e.data, pos, e.k)
	}
	nn.Axpy32JIT(dst, tmp[:len(dst)], 1)
	return nil
}

// embedRow writes token t's embedding into dst.
//
// The embedding is in the device layout like everything else, so the lookup is
// a strided read (one row's payload words sit rows*4 bytes apart). That buys a
// tied head the packed kernel at the cost of some cache misses once per token
// (cpu.EmitPackedRow).
func (m *Model) embedRow(dst []float32, token int) error {
	// Gemma 3n's hard vision tokens take the vision embedder's rows, which
	// Open divided by the embedding scale every caller applies after this.
	if h := m.hardEmbd; h != nil && token >= h.off && token < h.off+h.n {
		copy(dst, h.rows[(token-h.off)*len(dst):(token-h.off+1)*len(dst)])
		return nil
	}
	e := &m.embd
	if token < 0 || token >= e.rows {
		return fmt.Errorf("model: token %d is outside a vocabulary of %d", token, e.rows)
	}
	if e.packed != nil && len(e.packed.QS) != 0 {
		nn.RowPacked32JIT(dst, e.typ, e.packed, token, e.rows, e.k)
		return nil
	}
	nn.Row32JIT(dst, e.typ, e.data, token, e.k)
	return nil
}

// bindPacked binds every weight of one block.
//
// An expert of a bank is a view, and binding it with bindOne would overwrite it
// with the whole bank, making every expert expert 0 on an unpacked bank. The
// views are rebuilt from the freshly bound bank instead.
func bindPacked(c *jlm.File, cfg *Config, l *layer) error {
	// Every matrix a block holds. A tensor missing from this list keeps a stale
	// or empty span after a page-in and no host reader will accept it. This
	// list, the build() bindings and releaseLayer must agree about a new
	// matrix.
	for _, t := range []*tensor{
		&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down, &l.router,
		&l.shGate, &l.shUp, &l.shDown,
		&l.ssmGate, &l.ssmBA, &l.ssmOut, &l.ssmIn,
		// MLA's five.
		&l.wqa, &l.wqb, &l.wkva, &l.wkb, &l.wvb,
		// Kimi-K3's MLA output gate and latent mixture projections.
		&l.mlaGate, &l.routedDown, &l.routedUp,
		// Kimi Delta Attention's four.
		&l.ssmFA, &l.ssmFB, &l.ssmGA, &l.ssmGB, &l.ssmXDt, &l.ssmXB, &l.ssmXC, &l.ssmDtProj,
		// Gemma 4's per-layer embedding gate and projection.
		&l.pleGate, &l.pleProj,
		// DeepSeek V3.2's indexer, and MiniMax-M3's query.
		&l.idxQB, &l.idxK, &l.idxProj, &l.idxQ,
		// Gemma 3n's AltUp router and LAuReL.
		&l.altRouter, &l.laurelL, &l.laurelR,
	} {
		bindOne(c, t)
	}
	// DeepSeek V4's matrices, and the grouped projection's per-group views,
	// re-cut when the bank moved as the heads below are.
	if d := l.q4e; d != nil {
		for _, t := range d.tensors() {
			bindOne(c, t)
		}
	}
	if d := l.ds4; d != nil {
		for _, t := range d.tensors() {
			bindOne(c, t)
		}
		if from := spansOf(d.woA); from != d.woAFrom {
			gw := cfg.NHead / cfg.OGroups * cfg.HeadDim
			for g := range d.woAG {
				t, err := bankExpert(d.woA, jlm.RoleAttnOutA, cfg.OGroups, cfg.OLoraRank, gw, g)
				if err != nil {
					return err
				}
				if !sameView(d.woAG[g], t) {
					d.woAG[g] = t
				}
			}
			d.woAFrom = from
		}
	}
	// A prediction block's eh_proj, head and embedding. The head and the
	// embedding are copies of the trunk's tensors unless the block carries its
	// own, and a copy taken at build() predates binding, so it is bound here
	// too; a dense span never moves, so after the first time this writes
	// nothing.
	if w := l.mtp; w != nil {
		for _, t := range []*tensor{&w.ehProj, &w.head, &w.embd} {
			bindOne(c, t)
		}
	}
	// The per-head views are re-cut, as l.experts is below: they are slices of
	// wkb and wvb, so a rebind that moves those would leave them stale.
	if from := [2]bankSpans{spansOf(l.wkb), spansOf(l.wvb)}; len(l.kbHead) != 0 && from != l.headsFrom {
		nope := cfg.HeadDim - cfg.NRot
		for h := range l.kbHead {
			kb, err := bankExpert(l.wkb, jlm.RoleAttnKB, cfg.NHead, cfg.KVLoraRank, nope, h)
			if err != nil {
				return err
			}
			vb, err := bankExpert(l.wvb, jlm.RoleAttnVB, cfg.NHead, cfg.HeadDimV, cfg.KVLoraRank, h)
			if err != nil {
				return err
			}
			// Written only when the head moved, for bindOne's reason above.
			if !sameView(l.kbHead[h], kb) {
				l.kbHead[h] = kb
			}
			if !sameView(l.vbHead[h], vb) {
				l.vbHead[h] = vb
			}
		}
		l.headsFrom = from
	}
	if l.legacyExperts {
		// Separate tensors per expert: each has its own name, so bindOne is
		// resolving a whole weight rather than clobbering a view.
		for i := range l.experts {
			bindOne(c, &l.experts[i].gate)
			bindOne(c, &l.experts[i].up)
			bindOne(c, &l.experts[i].down)
		}
		return nil
	}
	// A bank in expert pages has no bank to cut views from, and its views are
	// left alone: ensureExperts binds the ones a token selects. Emptying them
	// here would race another session's MoE reading them.
	if l.expBank().e != nil && c.ExpertPaged(l.expBank().e) {
		return nil
	}
	from := [3]bankSpans{spansOf(l.gate), spansOf(l.up), spansOf(l.down)}
	if from == l.banksFrom {
		return nil
	}
	for e := range l.experts {
		x := &l.experts[e]
		var err error
		if !l.ungatedExp {
			if x.gate, err = bankExpert(l.gate, jlm.RoleExpGateBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
				return err
			}
		}
		if x.up, err = bankExpert(l.up, jlm.RoleExpUpBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
			return err
		}
		if x.down, err = bankExpert(l.down, jlm.RoleExpDownBank, cfg.NExpert, cfg.ExpWidth(), cfg.NFFNExp, e); err != nil {
			return err
		}
	}
	l.banksFrom = from
	return nil
}

// bankSpans is where a bank's bytes are: its payload and, packed, its three
// planes, by address and length. Two are equal only when every span is the
// same memory, which is what a cut view depends on.
type bankSpans [4]spanAt

// spanAt is a span's address and length; a zero one is no span.
type spanAt struct {
	p *byte
	n int
}

func spanOf(b []byte) spanAt {
	if len(b) == 0 {
		return spanAt{}
	}
	return spanAt{&b[0], len(b)}
}

func spansOf(t tensor) bankSpans {
	s := bankSpans{spanOf(t.data)}
	if t.packed != nil {
		s[1], s[2], s[3] = spanOf(t.packed.QS), spanOf(t.packed.D), spanOf(t.packed.SC)
	}
	return s
}

func build(c *jlm.File, options ...tok.Option) (*Model, error) {
	cfg, err := configFrom(c.Config())
	if err != nil {
		return nil, err
	}
	// A block of a mixture model with no router is not a mixture: its mixer
	// alone (Nemotron 3) or a dense FFN (Jamba). Read off the tensors: the
	// container states the mixture model-wide.
	if cfg.MoE() {
		absent := make([]bool, cfg.NLayer+cfg.NMTP)
		some := false
		for i := range absent {
			absent[i] = !c.Has(jlm.RoleRouter, int32(i), -1)
			some = some || absent[i]
		}
		if some {
			cfg.noRouter = absent
		}
	}
	// Mamba-1's dt rank is its x_proj's dt rows.
	if cfg.Mamba1() {
		for i, k := range c.Config().LayerKinds {
			if e, ok := c.Find(jlm.RoleSSMXDt, int32(i), -1); ok && k == jlm.LayerMamba1 {
				cfg.dtRank = int(e.Dims[1])
				break
			}
		}
		if cfg.dtRank == 0 {
			return nil, fmt.Errorf("model: a Mamba-1 model with no x_proj")
		}
	}
	m := &Model{Cfg: cfg}
	m.pagerUnlock = m.pager.Unlock
	// An unsupported tokenizer is not a reason to refuse the weights: Vocab
	// nil is a model that runs token ids but cannot turn text into them.
	// TokErr says why.
	if m.Vocab, m.TokErr = tok.New(c.Vocab(), options...); m.TokErr != nil {
		m.Vocab = nil
	}

	// Tensors are resolved by role, block and expert index, not by name.
	get := func(role jlm.Role, block, index int32) (tensor, error) {
		e, ok := c.Find(role, block, index)
		if !ok {
			return tensor{}, fmt.Errorf("model: missing %v (block %d, index %d)", role, block, index)
		}
		if e.NDim < 2 {
			return tensor{}, fmt.Errorf("model: %v has %d dimensions, want 2", role, e.NDim)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok || !quant.Dequantable(typ) {
			return tensor{}, fmt.Errorf("model: %v is %v, which has no dequantizer yet", role, e.Type)
		}
		// rows is the product of every trailing dimension: for a 3-D expert
		// bank ne1 alone would describe one expert, and a matvec would
		// silently compute expert 0.
		nrows := 1
		for d := 1; d < int(e.NDim); d++ {
			nrows *= int(e.Dims[d])
		}
		qs, _, _ := c.Span(e)
		tn := tensor{e: e, typ: typ, data: qs, rows: nrows, k: int(e.Dims[0])}
		// Marked packed before the spans exist, since bankExpert and rows()
		// need the layout; bindOne fills the struct later.
		if jlm.Packed(e.Type) {
			tn.packed = &nn.Packed{}
		}
		return tn, nil
	}
	vec := func(role jlm.Role, block int32) ([]float32, error) {
		e, ok := c.Find(role, block, -1)
		if !ok {
			return nil, fmt.Errorf("model: missing %v (block %d)", role, block)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok {
			return nil, fmt.Errorf("model: %v is %v, which has no dequantizer yet", role, e.Type)
		}
		n := 1
		for d := 0; d < int(e.NDim); d++ {
			n *= int(e.Dims[d])
		}
		qs, _, _ := c.Span(e)
		out := make([]float32, n)
		if err := quant.Dequant32(typ, qs, out); err != nil {
			return nil, fmt.Errorf("model: %v: %w", role, err)
		}
		return out, nil
	}
	// optVec is vec for a tensor that may legitimately be absent; nil is the
	// answer, not an error.
	optVec := func(role jlm.Role, block int32) ([]float32, error) {
		if !c.Has(role, block, -1) {
			return nil, nil
		}
		return vec(role, block)
	}

	if m.embd, err = get(jlm.RoleTokenEmbd, jlm.DenseBlock, -1); err != nil {
		return nil, err
	}
	// Qwen3.8-Flash-Next has no output norm: the head's hyper-connection mix
	// stands in for it (q4e.go).
	outNorm := vec
	if cfg.Q4E() {
		outNorm = optVec
	}
	if m.outNorm, err = outNorm(jlm.RoleOutputNorm, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if p := cfg.PLEDim; p != 0 {
		if m.pleTok, err = get(jlm.RolePLETokEmbd, jlm.DenseBlock, -1); err != nil {
			return nil, err
		}
		if m.pleProj, err = get(jlm.RolePLEModelProj, jlm.DenseBlock, -1); err != nil {
			return nil, err
		}
		if m.pleNorm, err = vec(jlm.RolePLEProjNorm, jlm.DenseBlock); err != nil {
			return nil, err
		}
		// The table may stop short of the vocabulary: Gemma 3n's ids past
		// it (its vision and audio tokens) read row 0 (pleInputs).
		w := cfg.NLayer * p
		if m.pleTok.k != w || m.pleTok.rows == 0 || m.pleTok.rows > cfg.NVocab || m.pleProj.k != cfg.NEmbd ||
			m.pleProj.rows != w || len(m.pleNorm) != p {
			return nil, fmt.Errorf("model: per-layer embeddings: a table %dx%d, a projection %dx%d and "+
				"a norm of %d, want %d wide over at most %d tokens from %d, and a norm of %d",
				m.pleTok.rows, m.pleTok.k, m.pleProj.k, m.pleProj.rows, len(m.pleNorm), w, cfg.NVocab,
				cfg.NEmbd, p)
		}
	}
	if cfg.AltUp != 0 {
		if err := loadAltUp(m, cfg, get); err != nil {
			return nil, err
		}
	}
	if cfg.DSV4() {
		if err := loadDS4Head(m, cfg, get, vec); err != nil {
			return nil, err
		}
	}
	if err := loadK3Head(m, cfg, vec); err != nil {
		return nil, err
	}
	if err := loadQ4EHead(m, cfg, c, get, vec); err != nil {
		return nil, err
	}
	if m.outNormB, err = optVec(jlm.RoleOutputNormBias, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if c.Has(jlm.RolePosEmbd, jlm.DenseBlock, -1) {
		if m.posEmbd, err = get(jlm.RolePosEmbd, jlm.DenseBlock, -1); err != nil {
			return nil, err
		}
		if m.posEmbd.k != cfg.NEmbd {
			return nil, fmt.Errorf("model: the position table is %d wide, want %d", m.posEmbd.k, cfg.NEmbd)
		}
		// A learned table ends; the context is clamped to it so NewState
		// refuses a later position rather than reading past the table.
		if m.posEmbd.rows < cfg.NCtx {
			cfg.NCtx = m.posEmbd.rows
		}
	}
	if m.outB, err = optVec(jlm.RoleOutputBias, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if m.outB != nil && len(m.outB) != cfg.NVocab {
		return nil, fmt.Errorf("model: the head's bias has %d values, want %d", len(m.outB), cfg.NVocab)
	}
	// An RMSNorm has no bias, so a norm bias without FlagLayerNorm is a
	// container that disagrees with itself.
	if m.outNormB != nil && !cfg.LayerNorm {
		return nil, fmt.Errorf("model: %s: a norm bias on an RMSNorm model", cfg.Arch)
	}
	cfg.RopeInterleaved = c.Config().Arch.InterleavedMRope()
	m.rope = nn.Rope{NRot: cfg.NRot, Base: cfg.RopeBase, Neox: cfg.RopeNeox,
		Scale: cfg.AttnFactor, Runs: nn.MRopeRuns(cfg.RopeSections[:])}
	if cfg.RopeInterleaved && len(m.rope.Runs) > 0 {
		m.rope.Runs = nn.IMRopeRuns(cfg.RopeSections[:], cfg.NRot/2)
	}
	if cfg.RopeXD = c.Config().Arch.XDRope(); cfg.RopeXD {
		// transformers' XD-RoPE turns the whole head; a partial one has no
		// reference to be held to.
		if !cfg.RopeNeox || len(m.rope.Runs) == 0 || cfg.NRot != cfg.HeadDim {
			return nil, fmt.Errorf("model: %s: XD-RoPE needs NEOX pairs, rope sections and the whole head "+
				"turned (NRot %d of %d)", cfg.Arch, cfg.NRot, cfg.HeadDim)
		}
		b := m.rope
		m.rope.Runs, b.Runs = nn.XDRopeRuns(cfg.RopeSections[:], cfg.NRot/2)
		m.ropeB = &b
	}
	if cfg.QKL2Norm || cfg.VNorm {
		m.onesHD = make([]float32, cfg.MaxHeadDim())
		for i := range m.onesHD {
			m.onesHD[i] = 1
		}
	}
	// A second rotary for architectures trained with two bases (gemma3's
	// sliding layers), built only when RopeBaseSWA is set.
	if cfg.RopeBaseSWA != 0 || cfg.NRotSWA != cfg.NRot {
		r := m.rope
		if cfg.RopeBaseSWA != 0 {
			r.Base = cfg.RopeBaseSWA
		}
		// Gemma 4's sliding layers turn their own width, the plain rotary over
		// all of a 256-wide head where the global layers turn 512.
		r.NRot = cfg.NRotSWA
		if cfg.SWARopePlain {
			r.Scale = 1
		}
		m.ropeSWA = &r
	}
	// Per-pair rotary factors are a tensor: llama 3.1+ ships rope_freqs and
	// phi3 rope_factors_short/long, dividing theta per pair. phi3 picks by
	// sequence length; short is right below its original context (4096), and
	// the long switch is not implemented.
	for _, role := range []jlm.Role{jlm.RoleRopeFreqs, jlm.RoleRopeFactorsShort} {
		if c.Has(role, jlm.DenseBlock, -1) {
			if m.rope.Freqs, err = vec(role, jlm.DenseBlock); err != nil {
				return nil, err
			}
			break
		}
	}
	// Rotary scaling (YaRN, linear) is a per-pair divisor too, so it rides the
	// same field. It is set after the local rotary was copied above, because
	// llama.cpp leaves gemma3's sliding layers unscaled.
	if cfg.YarnFactor > 0 || cfg.RopeLinear > 0 {
		if m.rope.Freqs != nil {
			return nil, fmt.Errorf("model: %s ships rotary factors AND a scaling type; "+
				"composing the two is not implemented", cfg.Arch)
		}
		if cfg.YarnFactor > 0 {
			m.rope.Freqs = nn.YarnFreqs(cfg.NRot, cfg.RopeBase, cfg.YarnFactor, cfg.YarnOrigCtx,
				cfg.YarnBetaFast, cfg.YarnBetaSlow, cfg.YarnExact)
		} else {
			m.rope.Freqs = make([]float32, cfg.NRot/2)
			for i := range m.rope.Freqs {
				m.rope.Freqs[i] = float32(cfg.RopeLinear)
			}
		}
	}
	if cfg.TiedEmbd {
		m.output = m.embd
	} else if m.output, err = get(jlm.RoleOutput, jlm.DenseBlock, -1); err != nil {
		return nil, err
	}

	m.layers = make([]layer, cfg.NLayer+cfg.NMTP)
	// bind names one tensor a layer must have. It is a named type rather than
	// an anonymous struct so a hybrid can substitute the attention four for the
	// recurrent four without restating the literal.
	type bind = struct {
		role jlm.Role
		dst  *tensor
	}
	for i := range m.layers {
		// Nothing is faulted in here: the norms live in the dense region
		// (jlm.alwaysResident), so build() touches no page and placement can
		// happen before any block is read.
		l := &m.layers[i]
		bi := int32(i)
		want := []bind{
			{jlm.RoleAttnQ, &l.wq}, {jlm.RoleAttnK, &l.wk},
			{jlm.RoleAttnV, &l.wv}, {jlm.RoleAttnOut, &l.wo},
			{jlm.RoleFFNGate, &l.gate}, {jlm.RoleFFNUp, &l.up},
			{jlm.RoleFFNDown, &l.down},
		}
		// A linear layer (per the layer map) replaces the four attention
		// bindings with its own; MLA (a whole-model property, from KVLoraRank)
		// replaces them with five. The two are mutually exclusive per block:
		// Kimi-Linear has both kinds, and each branch trims the original
		// leading four, so applying both to one block would compose wrongly.
		linear := cfg.LayerKind(i).Recurrent()
		mla := cfg.KVLoraRank != 0 && !linear
		if mla {
			q := bind{jlm.RoleAttnQ, &l.wq}
			if cfg.QLoraRank != 0 {
				q = bind{jlm.RoleAttnQB, &l.wqb}
			}
			// The FFN triple must stay last: the mixture branch below trims it
			// with want[:len(want)-3].
			head := []bind{
				q,
				{jlm.RoleAttnKVA, &l.wkva},
				{jlm.RoleAttnKB, &l.wkb},
				{jlm.RoleAttnVB, &l.wvb},
				{jlm.RoleAttnOut, &l.wo},
			}
			if cfg.QLoraRank != 0 {
				head = append(head, bind{jlm.RoleAttnQA, &l.wqa})
			}
			want = append(head, want[4:]...)
		}
		// DeepSeek V4: the query latent's two halves, the key, and the
		// grouped projection's second half; the rest is loadDS4's.
		if cfg.DSV4() {
			want = append([]bind{
				{jlm.RoleAttnQA, &l.wqa}, {jlm.RoleAttnQB, &l.wqb},
				{jlm.RoleAttnK, &l.wk}, {jlm.RoleAttnOut, &l.wo},
			}, want[4:]...)
		}
		if linear {
			head := []bind{
				{jlm.RoleAttnQKV, &l.wq},
				{jlm.RoleSSMBA, &l.ssmBA},
				{jlm.RoleSSMOut, &l.ssmOut},
			}
			// A block that attends as well keeps the attention four, and its
			// mixer's projection goes beside them.
			attends := cfg.LayerKind(i).Attends()
			if attends {
				head[0].dst = &l.ssmIn
			}
			switch {
			case cfg.Mamba1():
				// x convolves (wq), z gates the output (ssmGate); dt, B and C
				// come off the convolved x, and there is no ssm_ba.
				head = []bind{{jlm.RoleAttnQKV, &l.wq}, {jlm.RoleAttnGate, &l.ssmGate},
					{jlm.RoleSSMOut, &l.ssmOut}, {jlm.RoleSSMXDt, &l.ssmXDt},
					{jlm.RoleSSMXB, &l.ssmXB}, {jlm.RoleSSMXC, &l.ssmXC},
					{jlm.RoleSSMDtProj, &l.ssmDtProj}}
			case cfg.ShortConv():
				// x convolves (wq, the mixed projection's place), B and C gate
				// it before and after (the two gate slots).
				head = []bind{{jlm.RoleSCX, &l.wq}, {jlm.RoleSCB, &l.ssmGate},
					{jlm.RoleSCC, &l.ssmBA}, {jlm.RoleSSMOut, &l.ssmOut}}
			case cfg.ChanDecay() && !c.Has(jlm.RoleSSMGA, bi, -1) && c.Has(jlm.RoleAttnGate, bi, -1):
				// Kimi-K3's full-rank output gate: one matrix where Kimi-Linear
				// factors it, in the z projection's place.
				head = append(head,
					bind{jlm.RoleSSMFA, &l.ssmFA}, bind{jlm.RoleSSMFB, &l.ssmFB},
					bind{jlm.RoleAttnGate, &l.ssmGate})
			case cfg.ChanDecay():
				head = append(head,
					bind{jlm.RoleSSMFA, &l.ssmFA}, bind{jlm.RoleSSMFB, &l.ssmFB},
					bind{jlm.RoleSSMGA, &l.ssmGA}, bind{jlm.RoleSSMGB, &l.ssmGB})
			default:
				head = append(head, bind{jlm.RoleAttnGate, &l.ssmGate})
			}
			if attends {
				want = append(head, want...)
			} else {
				want = append(head, want[4:]...)
			}
		}
		// A block with no FFN at all (no up projection and no mixture) is its
		// mixer alone; the FFN triple and its norm are not bound.
		if !cfg.MoEAt(i) && !c.Has(jlm.RoleFFNUp, bi, -1) && !c.Has(jlm.RoleFFNDown, bi, -1) {
			l.noFFN = true
			want = want[:len(want)-3]
		}
		// There is no fused qkv or gate/up in a container: convert.unfuse
		// splits phi3's while the bytes are still row-major. A packed tensor
		// has no byte range per row.
		if cfg.MoEAt(i) {
			// The dense FFN triple becomes a router plus three expert banks,
			// each one 3-D and holding every expert. Loaded in their own loop
			// rather than appended to want, so nothing about the attention
			// tensors above can be skipped by getting this branch wrong.
			want = want[:len(want)-3]
			// Two layouts: stacked 3-D banks, the only one the converter writes
			// (convert.sourceOf stacks Mixtral-era files), and separate 2-D
			// tensors per expert. The legacy branch survives for older
			// containers and for experts quantized to different types; those
			// run on the host and offerRange declines them by name.
			if !c.Has(jlm.RoleExpGateBank, bi, -1) && c.Has(jlm.RoleExpGate, bi, 0) {
				l.legacyExperts = true
			}
			specs := []struct {
				role jlm.Role
				dst  *tensor
			}{
				{jlm.RoleRouter, &l.router},
				{jlm.RoleExpGateBank, &l.gate},
				{jlm.RoleExpUpBank, &l.up},
				{jlm.RoleExpDownBank, &l.down},
			}
			if l.legacyExperts {
				specs = specs[:1] // the router only; experts are read below
			}
			// Experts with no gate bank are ungated: down(act(up(h))), the
			// activation an ungated kind (Nemotron 3's squared ReLU).
			if !l.legacyExperts && !c.Has(jlm.RoleExpGateBank, bi, -1) && c.Has(jlm.RoleExpUpBank, bi, -1) {
				if !slices.Contains(kernels.Ungated[:], cfg.Act) {
					return nil, fmt.Errorf("model: block %d's experts have no gate, and %v is a gated activation",
						bi, cfg.Act)
				}
				l.ungatedExp = true
				specs = slices.Delete(specs, 1, 2)
			}
			for _, spec := range specs {
				if *spec.dst, err = get(spec.role, bi, -1); err != nil {
					return nil, err
				}
			}
			// A model has a shared expert if the container carries its
			// tensors.
			// An ungated one (Nemotron 3) has up and down and no gate.
			if c.Has(jlm.RoleShExpGate, bi, -1) || c.Has(jlm.RoleShExpUp, bi, -1) {
				for _, spec := range []struct {
					role jlm.Role
					dst  *tensor
				}{
					{jlm.RoleShExpGate, &l.shGate},
					{jlm.RoleShExpUp, &l.shUp},
					{jlm.RoleShExpDown, &l.shDown},
				} {
					if spec.role == jlm.RoleShExpGate && !c.Has(jlm.RoleShExpGate, bi, -1) {
						if !slices.Contains(kernels.Ungated[:], cfg.Act) {
							return nil, fmt.Errorf("model: block %d's shared expert has no gate, and %v is a gated activation",
								bi, cfg.Act)
						}
						continue
					}
					if *spec.dst, err = get(spec.role, bi, -1); err != nil {
						return nil, err
					}
				}
				// Optional: DeepSeek's shared expert has no gate. See
				// sharedExpert for why an absent one is weight 1 and not
				// sigma(0).
				if l.shRouter, err = optVec(jlm.RoleShRouter, bi); err != nil {
					return nil, err
				}
				if l.shRouter != nil && len(l.shRouter) != cfg.NEmbd {
					return nil, fmt.Errorf("model: block %d shared-expert gate is %d wide, want %d",
						bi, len(l.shRouter), cfg.NEmbd)
				}
			}
		}
		for _, spec := range want {
			// A KV-sharing block has no k or v projection: it attends to an
			// earlier block's history (Config.KVSource).
			if cfg.KVShared(i) && (spec.role == jlm.RoleAttnK || spec.role == jlm.RoleAttnV) {
				continue
			}
			// An ungated FFN is one without a gate tensor (starcoder, phi-2,
			// falcon, nemotron): up, the activation, down.
			if spec.role == jlm.RoleFFNGate && !c.Has(jlm.RoleFFNGate, bi, -1) &&
				c.Has(jlm.RoleFFNUp, bi, -1) {
				continue
			}
			// Gemma 4's global layers have no v projection
			// (attention_k_eq_v): v is k's projection, before k's norm and
			// rotary, under v's own weightless norm. Only a model that norms v
			// may omit it; anywhere else an absent v is a broken file.
			if spec.role == jlm.RoleAttnV && cfg.VNorm && !c.Has(jlm.RoleAttnV, bi, -1) &&
				c.Has(jlm.RoleAttnK, bi, -1) {
				l.vFromK = true
				continue
			}
			if *spec.dst, err = get(spec.role, bi, -1); err != nil {
				return nil, err
			}
		}
		// A post-norm block (OLMo 2, EXAONE 4) has no pre-norms at all: each
		// branch reads the residual as it is and normalises its output instead.
		postNorm := c.Has(jlm.RolePostAttnNorm, bi, -1) && c.Has(jlm.RolePostFFNNorm, bi, -1)
		if (postNorm || cfg.Q4E()) && !c.Has(jlm.RoleAttnNorm, bi, -1) {
			// Qwen3.8-Flash-Next's hyper-connection mixes are its norms.
			l.attnNorm = nil
		} else if l.attnNorm, err = vec(jlm.RoleAttnNorm, bi); err != nil {
			return nil, err
		}
		// A parallel block may share one norm between its branches, and then
		// has no ffn_norm. Every other sequential block needs its own.
		if cfg.Parallel || (postNorm && l.attnNorm == nil) || l.noFFN || cfg.Q4E() {
			l.ffnNorm, err = optVec(jlm.RoleFFNNorm, bi)
		} else {
			l.ffnNorm, err = vec(jlm.RoleFFNNorm, bi)
		}
		if err != nil {
			return nil, err
		}
		for _, b := range []struct {
			role jlm.Role
			dst  *[]float32
		}{
			{jlm.RoleAttnNormBias, &l.attnNormB}, {jlm.RoleFFNNormBias, &l.ffnNormB},
			{jlm.RoleFFNUpBias, &l.upB}, {jlm.RoleFFNDownBias, &l.downB},
		} {
			if *b.dst, err = optVec(b.role, bi); err != nil {
				return nil, err
			}
		}
		if (l.attnNormB != nil || l.ffnNormB != nil) && !cfg.LayerNorm {
			return nil, fmt.Errorf("model: %s: block %d has a norm bias on an RMSNorm model", cfg.Arch, bi)
		}
		if l.ffnNormB != nil && l.ffnNorm == nil {
			return nil, fmt.Errorf("model: block %d has an FFN norm bias and no FFN norm", bi)
		}
		// FFN biases belong only to a dense ungated FFN; a mixture's are the
		// experts' own (RoleExpUpBias). Refused rather than guessed.
		if (l.upB != nil || l.downB != nil) && (l.gate.e != nil || cfg.MoEAt(i)) {
			return nil, fmt.Errorf("model: block %d carries FFN biases on a gated or mixture FFN", bi)
		}
		// Apertus's xIELU: four numbers on every block, and only an ungated
		// dense FFN runs it. A block without them, or a model that is not
		// Apertus carrying them, is a file this engine would run wrong.
		if cfg.Act == nn.ActXIELU {
			if l.xielu, err = vec(jlm.RoleXIELU, bi); err != nil {
				return nil, err
			}
			if len(l.xielu) != 4 || l.gate.e != nil || cfg.MoEAt(i) {
				return nil, fmt.Errorf("model: block %d: xIELU wants four numbers on an ungated dense FFN "+
					"(%d numbers, gate %v, mixture %v)", bi, len(l.xielu), l.gate.e != nil, cfg.MoEAt(i))
			}
		} else if c.Has(jlm.RoleXIELU, bi, -1) {
			return nil, fmt.Errorf("model: block %d carries xIELU numbers and %s runs %v", bi, cfg.Arch, cfg.Act)
		}
		// gemma2/gemma3's post-norms, when the file carries them.
		if l.postAttnNorm, err = optVec(jlm.RolePostAttnNorm, bi); err != nil {
			return nil, err
		}
		// BitNet's two inner norms: both on every block of a BitNet, and on
		// no other model, each as wide as what it normalises.
		if l.attnSubNorm, err = optVec(jlm.RoleAttnSubNorm, bi); err != nil {
			return nil, err
		}
		if l.ffnSubNorm, err = optVec(jlm.RoleFFNSubNorm, bi); err != nil {
			return nil, err
		}
		if (l.attnSubNorm != nil || l.ffnSubNorm != nil) != cfg.TokenAct ||
			cfg.TokenAct && (len(l.attnSubNorm) != cfg.QDimAt(i) || len(l.ffnSubNorm) != cfg.NFFNAt(i) ||
				l.gate.e == nil || cfg.MoEAt(i) || cfg.Parallel) {
			return nil, fmt.Errorf("model: block %d: inner norms %d and %d wide on %s (a gated dense FFN of %d, "+
				"attention %d wide, per-token activations %v)", bi, len(l.attnSubNorm), len(l.ffnSubNorm),
				cfg.Arch, cfg.NFFNAt(i), cfg.QDimAt(i), cfg.TokenAct)
		}
		// Gemma 4's per-block output scalar, when the file carries one.
		if sc, err := optVec(jlm.RoleLayerOutScale, bi); err != nil {
			return nil, err
		} else if sc != nil {
			if len(sc) != 1 {
				return nil, fmt.Errorf("model: block %d's output scale has %d values, want 1", bi, len(sc))
			}
			l.outScale = sc[0]
		}
		if l.postFFNNorm, err = optVec(jlm.RolePostFFNNorm, bi); err != nil {
			return nil, err
		}
		// Gemma 4's per-layer embedding, gated in after the FFN.
		if p := cfg.PLEDim; p != 0 {
			if l.pleGate, err = get(jlm.RolePLEGate, bi, -1); err != nil {
				return nil, err
			}
			if l.pleProj, err = get(jlm.RolePLEProj, bi, -1); err != nil {
				return nil, err
			}
			if l.plePost, err = vec(jlm.RolePLEPostNorm, bi); err != nil {
				return nil, err
			}
			if l.pleGate.k != cfg.NEmbd || l.pleGate.rows != p || l.pleProj.k != p ||
				l.pleProj.rows != cfg.NEmbd || len(l.plePost) != cfg.NEmbd {
				return nil, fmt.Errorf("model: block %d's per-layer embedding gate is %dx%d, its "+
					"projection %dx%d and its norm %d, want %d and %d", bi, l.pleGate.k, l.pleGate.rows,
					l.pleProj.k, l.pleProj.rows, len(l.plePost), cfg.NEmbd, p)
			}
		}
		if cfg.AltUp != 0 {
			if err := loadAltUpLayer(l, bi, cfg, get, vec); err != nil {
				return nil, err
			}
		}
		if f := cfg.NFFNAt(i); !cfg.MoEAt(i) && l.down.k != 0 && (l.down.k != f || l.up.rows != f) {
			return nil, fmt.Errorf("model: block %d's FFN is %d wide (up %d), want %d", bi, l.down.k,
				l.up.rows, f)
		}
		if cfg.DenseMoE && cfg.MoEAt(i) {
			for _, v := range []struct {
				role jlm.Role
				dst  *[]float32
				want int
			}{
				{jlm.RoleFFNNorm2, &l.ffnNorm2, cfg.NEmbd},
				{jlm.RolePostFFNNorm1, &l.postFFNNorm1, cfg.NEmbd},
				{jlm.RolePostFFNNorm2, &l.postFFNNorm2, cfg.NEmbd},
				{jlm.RoleRouterNorm, &l.routerNorm, cfg.NEmbd},
				{jlm.RoleExpScale, &l.expScale, cfg.NExpert},
			} {
				if *v.dst, err = vec(v.role, bi); err != nil {
					return nil, err
				}
				if len(*v.dst) != v.want {
					return nil, fmt.Errorf("model: block %d %s has %d values, want %d",
						bi, v.role, len(*v.dst), v.want)
				}
			}
			if l.postFFNNorm == nil || l.ffnNorm == nil {
				return nil, fmt.Errorf("model: block %d is a dense-and-mixture block with no "+
					"ffn_norm or post_ffw_norm", bi)
			}
		}
		// Attention biases, when the file carries them.
		for _, b := range []struct {
			role jlm.Role
			dst  *[]float32
		}{
			{jlm.RoleAttnQBias, &l.bq}, {jlm.RoleAttnKBias, &l.bk},
			{jlm.RoleAttnVBias, &l.bv}, {jlm.RoleAttnOutBias, &l.bo},
			{jlm.RoleAttnSinks, &l.sinks},
			{jlm.RoleRouterBias, &l.routerB}, {jlm.RoleExpProbsB, &l.expProbsB},
			{jlm.RoleExpGateBias, &l.expGateB},
			{jlm.RoleExpUpBias, &l.expUpB}, {jlm.RoleExpDownBias, &l.expDownB},
		} {
			if *b.dst, err = optVec(b.role, bi); err != nil {
				return nil, err
			}
		}
		// A bias of the wrong length would still run, landing on a neighbour's
		// rows, so the lengths are checked here.
		for _, b := range []struct {
			name string
			v    []float32
			want int
		}{
			{"attn_sinks", l.sinks, cfg.NHead},
			{"router bias", l.routerB, cfg.NExpert},
			{"expert selection bias", l.expProbsB, cfg.NExpert},
			{"expert gate bias", l.expGateB, cfg.NExpert * cfg.NFFNExp},
			{"expert up bias", l.expUpB, cfg.NExpert * cfg.NFFNExp},
			{"expert down bias", l.expDownB, cfg.NExpert * cfg.NEmbd},
			{"attn_norm bias", l.attnNormB, cfg.NEmbd},
			{"ffn_norm bias", l.ffnNormB, cfg.NEmbd},
			{"ffn_up bias", l.upB, cfg.NFFN},
			{"ffn_down bias", l.downB, cfg.NEmbd},
		} {
			if b.v != nil && len(b.v) != b.want {
				return nil, fmt.Errorf("model: block %d %s has %d values, want %d",
					bi, b.name, len(b.v), b.want)
			}
		}
		// The recurrent vectors are small and dense. ssm_conv1d is a matrix by
		// shape and a per-channel window by use, so it is expanded here.
		if linear {
			for _, v := range []struct {
				role jlm.Role
				dst  *[]float32
			}{
				{jlm.RoleSSMConv1d, &l.ssmConv1d}, {jlm.RoleSSMA, &l.ssmA},
				{jlm.RoleSSMDtBias, &l.ssmDtBias}, {jlm.RoleSSMNorm, &l.ssmNorm},
			} {
				// A short convolution has its window and nothing else.
				if cfg.ShortConv() && v.role != jlm.RoleSSMConv1d {
					continue
				}
				// A Mamba-2 block may have no gated norm at all (Falcon-H1's
				// mamba_rms_norm false): y*silu(z) goes straight to ssm_out.
				if v.role == jlm.RoleSSMNorm && cfg.Mamba1() {
					continue // Mamba-1 has no gated norm
				}
				if v.role == jlm.RoleSSMNorm && cfg.SSD() {
					if *v.dst, err = optVec(v.role, bi); err != nil {
						return nil, err
					}
					continue
				}
				if *v.dst, err = vec(v.role, bi); err != nil {
					return nil, err
				}
			}
			g := cfg.delta()
			// Transposed once at load: the file stores taps contiguous per
			// channel, and plane-major makes the convolution elementwise
			// (see cpu.EmitConv1d).
			if n := len(l.ssmConv1d); n == g.conv*g.chans {
				wt := make([]float32, n)
				for c := 0; c < g.chans; c++ {
					for t := 0; t < g.conv; t++ {
						wt[t*g.chans+c] = l.ssmConv1d[c*g.conv+t]
					}
				}
				l.ssmConv1d = wt
			}
			if len(l.ssmConv1d) != g.conv*g.chans {
				return nil, fmt.Errorf("model: block %d convolution is %d wide, want %d (%d positions x %d channels)",
					bi, len(l.ssmConv1d), g.conv*g.chans, g.conv, g.chans)
			}
			gateWidth, dtWidth := g.vHeads, g.vHeads
			if cfg.ChanDecay() {
				gateWidth, dtWidth = g.vHeads*g.kDim, g.vHeads*g.kDim
			}
			normWidth, dWidth, cbWidth := g.vDim, 0, 0
			if g.mamba1 {
				// A is a row of the state per channel; D, dt's bias and the
				// convolution's bias one per channel; no gated norm.
				gateWidth, normWidth = g.vHeads*g.kDim, 0
				if l.ssmD, err = vec(jlm.RoleSSMD, bi); err != nil {
					return nil, err
				}
				if l.ssmConvB, err = optVec(jlm.RoleSSMConvBias, bi); err != nil {
					return nil, err
				}
				if l.ssmConvB == nil {
					l.ssmConvB = make([]float32, g.chans)
				}
				dWidth, cbWidth = g.vHeads, g.chans
				for _, v := range []struct {
					role jlm.Role
					dst  *[]float32
					want int
				}{{jlm.RoleSSMDtNorm, &l.ssmDtNorm, g.rank}, {jlm.RoleSSMBNorm, &l.ssmBNorm, g.kDim},
					{jlm.RoleSSMCNorm, &l.ssmCNorm, g.kDim}} {
					if *v.dst, err = optVec(v.role, bi); err != nil {
						return nil, err
					}
					if *v.dst != nil && len(*v.dst) != v.want {
						return nil, fmt.Errorf("model: block %d %v is %d wide, want %d", bi, v.role, len(*v.dst), v.want)
					}
				}
				// The three norms come together or not at all, as llama.cpp
				// applies them.
				if (l.ssmDtNorm == nil) != (l.ssmBNorm == nil) || (l.ssmDtNorm == nil) != (l.ssmCNorm == nil) {
					return nil, fmt.Errorf("model: block %d carries some of Mamba-1's dt/B/C norms and not all", bi)
				}
				for _, m := range []struct {
					name    string
					t       tensor
					k, rows int
				}{{"ssm_x.dt", l.ssmXDt, g.chans, g.rank}, {"ssm_x.b", l.ssmXB, g.chans, g.kDim},
					{"ssm_x.c", l.ssmXC, g.chans, g.kDim}, {"ssm_dt", l.ssmDtProj, g.rank, g.vHeads}} {
					if m.t.k != m.k || m.t.rows != m.rows {
						return nil, fmt.Errorf("model: block %d %s is %dx%d, want %dx%d",
							bi, m.name, m.t.k, m.t.rows, m.k, m.rows)
					}
				}
			}
			if g.ssd {
				// Mamba-2's two vectors more, and its gated norm's weight is
				// every channel of y, a group of Inner/Groups at a time.
				if l.ssmD, err = vec(jlm.RoleSSMD, bi); err != nil {
					return nil, err
				}
				if l.ssmConvB, err = optVec(jlm.RoleSSMConvBias, bi); err != nil {
					return nil, err
				}
				if l.ssmConvB == nil {
					l.ssmConvB = make([]float32, g.chans)
				}
				normWidth, dWidth, cbWidth = g.vHeads*g.vDim, g.vHeads, g.chans
				if l.ssmNorm == nil {
					normWidth = 0
				}
			}
			for _, v := range []struct {
				name string
				have int
				want int
			}{
				// KDA carries one decay rate and delta-t bias per (head,
				// channel); qwen3next one per head.
				{"ssm_a", len(l.ssmA), gateWidth}, {"ssm_dt.bias", len(l.ssmDtBias), dtWidth},
				{"ssm_norm", len(l.ssmNorm), normWidth},
				{"ssm_d", len(l.ssmD), dWidth}, {"ssm_conv1d.bias", len(l.ssmConvB), cbWidth},
			} {
				if v.have != v.want {
					return nil, fmt.Errorf("model: block %d %s is %d wide, want %d", bi, v.name, v.have, v.want)
				}
			}
		}
		if mla {
			// The kv latent's norm is always there; the q latent's exists only
			// in the two-step form, which is what QLoraRank says.
			if l.kvaNorm, err = vec(jlm.RoleAttnKVANorm, bi); err != nil {
				return nil, err
			}
			if cfg.QLoraRank != 0 {
				if l.qaNorm, err = vec(jlm.RoleAttnQANorm, bi); err != nil {
					return nil, err
				}
			}
			// A norm of the wrong length would still run, so both latent
			// norms are checked.
			for _, v := range []struct {
				name string
				have int
				want int
			}{
				{"attn_kv_a_norm", len(l.kvaNorm), cfg.KVLoraRank},
				{"attn_q_a_norm", len(l.qaNorm), cfg.QLoraRank},
			} {
				if v.want != 0 && v.have != v.want {
					return nil, fmt.Errorf("model: block %d %s is %d wide, want %d",
						bi, v.name, v.have, v.want)
				}
			}
			if cfg.Indexer() {
				if err := loadIndexer(l, bi, cfg, get, vec); err != nil {
					return nil, err
				}
			}
			// Cut per head once, at load: wkb and wvb are banks of NHead
			// sheets, and the absorb is nn.MatVecPackedGather over them with
			// the head as the "expert".
			nope := cfg.HeadDim - cfg.NRot
			for h := 0; h < cfg.NHead; h++ {
				t, err := bankExpert(l.wkb, jlm.RoleAttnKB, cfg.NHead, cfg.KVLoraRank, nope, h)
				if err != nil {
					return nil, err
				}
				l.kbHead = append(l.kbHead, t)
				if t, err = bankExpert(l.wvb, jlm.RoleAttnVB, cfg.NHead, cfg.HeadDimV, cfg.KVLoraRank, h); err != nil {
					return nil, err
				}
				l.vbHead = append(l.vbHead, t)
			}
		}
		if cfg.MSAAt(i) {
			if err := loadMSA(l, bi, cfg, get, vec); err != nil {
				return nil, err
			}
		}
		if cfg.DSV4() {
			if err := loadDS4(l, bi, cfg, get, vec); err != nil {
				return nil, err
			}
		}
		if err := loadK3(l, bi, cfg, c, get, vec); err != nil {
			return nil, err
		}
		if err := loadQ4E(l, bi, cfg, get, vec); err != nil {
			return nil, err
		}
		if cfg.QSAAt(i) {
			if err := loadQSA(l, bi, cfg, get, vec); err != nil {
				return nil, err
			}
		}
		if cfg.QKNormAt(i) {
			if l.qNorm, err = vec(jlm.RoleAttnQNorm, bi); err != nil {
				return nil, err
			}
			// A KV-sharing block norms only q: it caches nothing.
			if !cfg.KVShared(i) {
				if l.kNorm, err = vec(jlm.RoleAttnKNorm, bi); err != nil {
					return nil, err
				}
			}
			// A norm of the wrong width reads past its projection or
			// leaves part of it unnormed, fluently.
			_, _, wq, wk := cfg.QKNormShapeAt(i)
			if cfg.KVShared(i) {
				wk = 0
			}
			if len(l.qNorm) != wq || len(l.kNorm) != wk {
				return nil, fmt.Errorf("model: block %d's q/k norms are %d and %d wide, want %d and %d",
					i, len(l.qNorm), len(l.kNorm), wq, wk)
			}
		}
		if cfg.MoEAt(i) {
			if l.router.rows != cfg.NExpert || l.router.k != cfg.NEmbd {
				return nil, fmt.Errorf("model: block %d router is %dx%d, want %dx%d",
					bi, l.router.k, l.router.rows, cfg.NEmbd, cfg.NExpert)
			}
			l.experts = make([]expert, cfg.NExpert)
			if l.legacyExperts {
				// NFFNExp comes from the tensor, not from a ratio of other
				// keys.
				for e := range l.experts {
					x := &l.experts[e]
					if x.gate, err = get(jlm.RoleExpGate, bi, int32(e)); err != nil {
						return nil, err
					}
					if x.up, err = get(jlm.RoleExpUp, bi, int32(e)); err != nil {
						return nil, err
					}
					if x.down, err = get(jlm.RoleExpDown, bi, int32(e)); err != nil {
						return nil, err
					}
					if e == 0 {
						cfg.NFFNExp = x.gate.rows
					}
					if x.gate.rows != cfg.NFFNExp || x.gate.k != cfg.ExpWidth() {
						return nil, fmt.Errorf("model: block %d expert %d gate is %dx%d, want %dx%d",
							bi, e, x.gate.k, x.gate.rows, cfg.ExpWidth(), cfg.NFFNExp)
					}
					if x.down.rows != cfg.ExpWidth() || x.down.k != cfg.NFFNExp {
						return nil, fmt.Errorf("model: block %d expert %d down is %dx%d, want %dx%d",
							bi, e, x.down.k, x.down.rows, cfg.NFFNExp, cfg.ExpWidth())
					}
				}
				continue
			}
			for e := range l.experts {
				x := &l.experts[e]
				if !l.ungatedExp {
					if x.gate, err = bankExpert(l.gate, jlm.RoleExpGateBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
						return nil, err
					}
				}
				if x.up, err = bankExpert(l.up, jlm.RoleExpUpBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
					return nil, err
				}
				// down's shape is the transpose of gate's: [n_ff_exp, n_embd].
				if x.down, err = bankExpert(l.down, jlm.RoleExpDownBank, cfg.NExpert, cfg.ExpWidth(), cfg.NFFNExp, e); err != nil {
					return nil, err
				}
			}
		}
	}
	if err := buildMTP(c, m, get, vec, optVec); err != nil {
		return nil, err
	}
	return m, nil
}

// buildMTP binds what each multi-token-prediction block has beyond the
// ordinary block build() already bound (jlm.Config.NMTP).
func buildMTP(c *jlm.File, m *Model,
	get func(jlm.Role, int32, int32) (tensor, error),
	vec, optVec func(jlm.Role, int32) ([]float32, error)) error {
	cfg := m.Cfg
	for i := cfg.NLayer; i < cfg.NLayer+cfg.NMTP; i++ {
		bi := int32(i)
		w := &mtpWeights{head: m.output, embd: m.embd, headNorm: m.outNorm}
		var err error
		if w.ehProj, err = get(jlm.RoleNextnEHProj, bi, -1); err != nil {
			return err
		}
		if w.ehProj.k != 2*cfg.NEmbd || w.ehProj.rows != cfg.NEmbd {
			return fmt.Errorf("model: prediction block %d eh_proj is %dx%d, want %dx%d",
				i, w.ehProj.k, w.ehProj.rows, 2*cfg.NEmbd, cfg.NEmbd)
		}
		if w.enorm, err = vec(jlm.RoleNextnENorm, bi); err != nil {
			return err
		}
		if w.hnorm, err = vec(jlm.RoleNextnHNorm, bi); err != nil {
			return err
		}
		if hn, err := optVec(jlm.RoleNextnHeadNorm, bi); err != nil {
			return err
		} else if hn != nil {
			w.headNorm = hn
		}
		// Qwen3.8-Flash-Next's block reads the trunk's residual, every stream,
		// each normed against its own row of hnorm, and its head is the block's
		// own mix (q4eLayer.head), not a norm.
		hw, hn := cfg.NEmbd, cfg.NEmbd
		if cfg.Q4E() {
			hw, hn = cfg.ResidW(), len(w.headNorm)
		}
		for _, v := range []struct {
			name    string
			n, want int
		}{{"enorm", len(w.enorm), cfg.NEmbd}, {"hnorm", len(w.hnorm), hw},
			{"shared_head_norm", len(w.headNorm), hn}} {
			if v.n != v.want {
				return fmt.Errorf("model: prediction block %d %s is %d wide, want %d", i, v.name, v.n, v.want)
			}
		}
		// The block's own head and embedding exist only where they differed
		// from the trunk's (convert.nextnTensors dropped the copies).
		if c.Has(jlm.RoleNextnHead, bi, -1) {
			if w.head, err = get(jlm.RoleNextnHead, bi, -1); err != nil {
				return err
			}
		}
		if c.Has(jlm.RoleNextnEmbd, bi, -1) {
			if w.embd, err = get(jlm.RoleNextnEmbd, bi, -1); err != nil {
				return err
			}
		}
		for _, t := range []tensor{w.head, w.embd} {
			if t.rows != cfg.NVocab || t.k != cfg.NEmbd {
				return fmt.Errorf("model: prediction block %d's head or embedding is %dx%d, want %dx%d",
					i, t.k, t.rows, cfg.NEmbd, cfg.NVocab)
			}
		}
		m.layers[i].mtp = w
	}
	return nil
}

// bankExpert validates a 3-D expert bank and returns expert e's view of it.
//
// The check is the exact byte total, not the dimensions: a bank short by an
// expert would otherwise slice inside the buffer and return another expert's
// weights, which every tier would agree on.
func bankExpert(b tensor, role jlm.Role, nExpert, nrow, k, e int) (tensor, error) {
	if b.k != k || b.rows != nExpert*nrow {
		return tensor{}, fmt.Errorf("model: %v is %d x %d, want %d x %d (%d experts of %d)",
			role, b.k, b.rows, k, nExpert*nrow, nExpert, nrow)
	}
	be, bb := int(b.typ.BlockElems()), int(b.typ.BlockBytes())
	if be == 0 || b.k%be != 0 {
		return tensor{}, fmt.Errorf("model: %v rows are %d elements, not a multiple of %s's %d",
			role, b.k, b.typ, be)
	}
	// A packed bank is priced by the packer, not by the source block size:
	// b.data is the payload span only.
	if b.packed != nil {
		q, ok := kernels.QuantOf(b.typ)
		if !ok {
			return tensor{}, fmt.Errorf("model: %v: %s has no device layout", role, b.typ)
		}
		// An expert is a contiguous sheet: the container packs a bank as
		// n_expert independent matrices back to back (jlm.sheetsOf), so expert
		// e owns one range of each of the three spans.
		nq, nd, nsc, err := kernels.PackedWords(q, nrow, b.k)
		if err != nil {
			return tensor{}, fmt.Errorf("model: %v: %w", role, err)
		}
		// Empty spans at build time are not an error: build() needs the shape,
		// and pageIn re-runs bindPacked, which re-runs this, for the bytes.
		bound := len(b.data) != 0
		// The span is padded to jlm.Align, so the test is a range and not an
		// equality.
		if want := nq * 4 * nExpert; bound &&
			(len(b.data) < want || len(b.data)-want >= jlm.Align) {
			return tensor{}, fmt.Errorf("model: %v holds %d packed bytes, want %d (%d experts of %d words)",
				role, len(b.data), want, nExpert, nq)
		}
		cut := func(buf []byte, words int) ([]byte, error) {
			if words == 0 || len(buf) == 0 {
				return nil, nil
			}
			lo, hi := e*words*4, (e+1)*words*4
			if hi > len(buf) {
				return nil, fmt.Errorf("model: %v: expert %d wants bytes [%d,%d) of %d",
					role, e, lo, hi, len(buf))
			}
			return buf[lo:hi:hi], nil
		}
		qs, err := cut(b.packed.QS, nq)
		if err != nil {
			return tensor{}, err
		}
		dd, err := cut(b.packed.D, nd)
		if err != nil {
			return tensor{}, err
		}
		sc, err := cut(b.packed.SC, nsc)
		if err != nil {
			return tensor{}, err
		}
		p := &nn.Packed{QS: qs, D: dd, SC: sc, Row: 0, Stride: nrow}
		return tensor{e: b.e, typ: b.typ, data: qs, rows: nrow, k: b.k, packed: p}, nil
	}
	// Not bound yet is not an error here either (an all-F32 bank lives in a
	// page); build() needs the shape and pageIn re-runs this for the bytes.
	if len(b.data) == 0 {
		return tensor{e: b.e, typ: b.typ, rows: nrow, k: b.k}, nil
	}
	if want := nExpert * nrow * (b.k / be * bb); want != len(b.data) {
		return tensor{}, fmt.Errorf("model: %v holds %d bytes, want %d", role, len(b.data), want)
	}
	return rows(b, e*nrow, nrow)
}

// rows returns the sub-tensor made of n rows of t starting at from.
//
// It aliases rather than copies: a row-major matrix's row range is a contiguous
// byte range.
func rows(t tensor, from, n int) (tensor, error) {
	// A packed row range is not a byte range: the device layout is
	// [sub-block][word][row], so the row offset travels with the weight.
	if t.packed != nil {
		if from < 0 || n <= 0 || from+n > t.rows {
			return tensor{}, fmt.Errorf("model: %s: rows [%d,%d) of %d do not fit",
				t.name(), from, from+n, t.rows)
		}
		stride := t.packed.Stride
		if stride == 0 {
			stride = t.rows
		}
		p := &nn.Packed{QS: t.packed.QS, D: t.packed.D, SC: t.packed.SC,
			Row: t.packed.Row + from, Stride: stride}
		return tensor{e: t.e, typ: t.typ, data: t.data, rows: n, k: t.k, packed: p}, nil
	}
	be, bb := int(t.typ.BlockElems()), int(t.typ.BlockBytes())
	if be == 0 || t.k%be != 0 {
		return tensor{}, fmt.Errorf("model: %s rows are %d elements, not a multiple of %s's %d",
			t.name(), t.k, t.typ, be)
	}
	rowBytes := t.k / be * bb
	lo, hi := from*rowBytes, (from+n)*rowBytes
	if from < 0 || n <= 0 || hi > len(t.data) {
		return tensor{}, fmt.Errorf("model: %s: rows [%d,%d) of %d do not fit %d bytes",
			t.name(), from, from+n, t.rows, len(t.data))
	}
	return tensor{e: t.e, typ: t.typ, data: t.data[lo:hi], rows: n, k: t.k}, nil
}

// Trace installs a callback invoked with every intermediate vector, for diffing
// against an independent oracle layer by layer. nil disables it.
func (m *Model) Trace(fn func(layer int, name string, v []float32)) { m.tracer = fn }

func (m *Model) trace(l int, name string, v []float32) {
	if m.tracer != nil {
		m.tracer(l, name, v)
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// WeightBytes is every weight byte in the file, which is what has to be
// resident for decode not to touch the disk. BytesPerToken is what one token
// reads, and for a mixture of experts the two differ by a lot: Qwen3-30B-A3B is
// 18.5 GB of weights and reads 1.9 GB per token.
func (m *Model) WeightBytes() uint64 {
	var n uint64
	for i := range m.container.Entries() {
		qs, d, sc := m.container.SpanLen(&m.container.Entries()[i])
		n += qs + d + sc
	}
	return n
}

// BytesPerToken is how many bytes of weights one decode step reads: the
// denominator of every decode bandwidth claim.
//
// Every tensor is read once, except the embedding table, of which a step reads
// one row -- unless the embedding is tied, in which case it is also the head
// and is read in full.
func (m *Model) BytesPerToken() uint64 {
	// The container's own spans, which is what the engine reads; SpanLen is
	// arithmetic on the entry and needs no page resident.
	var total uint64
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		qs, d, sc := m.container.SpanLen(e)
		n := qs + d + sc
		// A MoE token reads NExpertUsed of NExpert experts.
		if m.Cfg.MoE() && e.NDim == 3 {
			n = n / uint64(m.Cfg.NExpert) * uint64(m.Cfg.NExpertUsed)
		}
		// The embedding is read one row per token, unless it is also the
		// head, in which case it is read in full plus a row for the lookup.
		if e.Role == jlm.RoleTokenEmbd {
			row := n / uint64(max(e.Rows(), 1))
			if m.Cfg.TiedEmbd {
				n += row
			} else {
				n = row
			}
		}
		total += n
	}
	return total
}

// maxMatVec is the largest row and column count of any matrix the model will
// hand to a matvec, including the output projection and the expert banks.
//
// The two maxima are taken independently: they size different buffers (rows
// the output and partial sums, k the activation staging), and the tallest and
// widest matrices are usually different tensors.
func (m *Model) maxMatVec() (rows, k int) {
	// The shape is the entry's, read whether or not the block is bound: a
	// block another State placed has given its host page back (releaseLayer)
	// and may be re-bound under this call, so its data is neither the answer
	// nor safe to read here.
	see := func(t *tensor) {
		if t.e == nil {
			return
		}
		if t.rows > rows {
			rows = t.rows
		}
		if t.k > k {
			k = t.k
		}
	}
	for i := range m.layers {
		l := &m.layers[i]
		for _, t := range []*tensor{&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down, &l.router,
			&l.pleGate, &l.pleProj, &l.idxQB, &l.idxK, &l.idxProj, &l.idxQ, &l.altRouter, &l.laurelL,
			&l.laurelR, &l.wqa, &l.wqb} {
			see(t)
		}
		if l.q4e != nil {
			for _, t := range l.q4e.tensors() {
				see(t)
			}
		}
		if l.ds4 != nil {
			for _, t := range l.ds4.tensors() {
				see(t)
			}
			for i := range l.ds4.woAG {
				see(&l.ds4.woAG[i])
			}
		}
	}
	see(&m.output)
	see(&m.embd)
	see(&m.pleProj)
	if m.q4eHead != nil {
		for _, t := range m.q4eHead.hc.tensors() {
			see(t)
		}
	}
	see(&m.altProj)
	for i := range m.altUnembd {
		see(&m.altUnembd[i])
	}
	return rows, k
}

// blockReadBytes is how many file bytes of block li a single token reads. On a
// mixture it is far less than the block's size: a placement removes the read
// from the host but costs the size on the device.
func (m *Model) blockReadBytes(li int) uint64 {
	l := &m.layers[li]
	var n uint64
	for _, t := range []tensor{l.wq, l.wk, l.wv, l.wo, l.router} {
		n += tensorBytes(t)
	}
	dense := tensorBytes(l.gate) + tensorBytes(l.up) + tensorBytes(l.down)
	if c := m.Cfg; c.MoE() && c.NExpert > 0 {
		dense = dense / uint64(c.NExpert) * uint64(c.NExpertUsed)
	}
	return n + dense
}

// headReadBytes is what a token reads for the output projection, or zero when
// the model ties its embedding (the projection is then the embedding table,
// which is already counted and already on the host).
func (m *Model) headReadBytes() uint64 { return tensorBytes(m.output) }

// tensorBytes is what this weight costs to READ once, which for a container is
// all three of its spans and not just the payload.
//
// It prices the tensor from its entry, not its residency: the spans are empty
// for a weight whose page is out.
func tensorBytes(t tensor) uint64 {
	if t.e != nil {
		qs, d, sc := jlm.SpanLenOf(t.e)
		return qs + d + sc
	}
	n := uint64(len(t.data))
	if t.packed != nil {
		n += uint64(len(t.packed.D) + len(t.packed.SC))
	}
	return n
}

// containerShape is the (type, k, rows) of one container entry, for the passes
// that size the JIT from every weight in the file.
//
// It reports false for anything the kernels cannot take: a 1-D norm, or a type
// written by a newer converter.
func containerShape(e *jlm.Entry) (tensor, bool) {
	if e.NDim < 2 {
		return tensor{}, false
	}
	typ, ok := jlm.SourceType(e.Type)
	if !ok {
		return tensor{}, false
	}
	rows := 1
	for d := 1; d < int(e.NDim); d++ {
		rows *= int(e.Dims[d])
	}
	return tensor{e: e, typ: typ, rows: rows, k: int(e.Dims[0])}, true
}

// hasPostNorm reports whether this model's blocks carry gemma2/gemma3's two
// extra RMSNorms. It asks the weights rather than the architecture, so a fourth
// architecture that ships them needs no entry anywhere.
func (m *Model) hasNoPreNorm() bool {
	for i := range m.layers {
		if m.layers[i].attnNorm == nil && m.layers[i].postAttnNorm != nil {
			return true
		}
	}
	return false
}

// hasSubNorm reports whether some block carries BitNet's inner norms.
func (m *Model) hasSubNorm() bool {
	for i := range m.layers {
		if l := &m.layers[i]; l.attnSubNorm != nil || l.ffnSubNorm != nil {
			return true
		}
	}
	return false
}

// hasPostNorm reports whether some block carries gemma2's post-norms.
func (m *Model) hasPostNorm() bool {
	// By index: a copy of each layer would read its bindings, which another
	// State's placement may be writing.
	for i := range m.layers {
		if l := &m.layers[i]; l.postAttnNorm != nil || l.postFFNNorm != nil {
			return true
		}
	}
	return false
}

// newJIT builds the generated-code tier a State runs on: one kernel per
// quantization type the container holds, sized for its widest matrix. The
// vision segment's matrices are in the same container, so one JIT serves the
// text blocks and the tower's; F32 joins the types where the model has a tower,
// for the bilinear preprocessors' resize (image.go).
func (m *Model) newJIT() *nn.JIT {
	maxK, maxRows := 0, m.Cfg.NVocab
	seen := map[quant.Type]bool{}
	var types []quant.Type
	add := func(t quant.Type) {
		if !seen[t] {
			seen[t] = true
			types = append(types, t)
		}
	}
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		t, ok := containerShape(e)
		if !ok {
			continue
		}
		maxK = max(maxK, t.k)
		maxRows = max(maxRows, t.rows)
		add(t.typ)
	}
	if m.tower != nil {
		add(quant.F32)
	}
	opts := append(m.jit[:len(m.jit):len(m.jit)], nn.WithServingSplit(!m.opt.noServingSplit))
	if m.Cfg.TokenAct {
		// One window over the widest row: each row's blocks share its scale.
		opts = append(append([]nn.Option(nil), opts...), nn.WithActWindow((maxK+31)/32*32))
	}
	return nn.NewJIT(maxK, maxRows, types, opts...)
}
