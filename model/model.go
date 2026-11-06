// Package model is a loaded model: a jlm container's weights, its
// configuration, its tokenizer and chat template, and the graphs it states
// for its blocks, head and vision tower. It runs nothing; package engine
// composes and runs its blocks on the devices an instance places them on.
package model

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/sched"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
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
	// RowWindowed says a non-causal block's rows attend inside windows of
	// their own, which its convolution's taps keep to as well: a media
	// prefix's rows the prefix alone (an LFM2 encoder's, rowWindows).
	RowWindowed bool
	// chanDecay is ChanDecay()'s backing field: a delta rule whose forget
	// gate is its own low-rank pair (RoleSSMFA, RoleSSMFB) decays per channel
	// rather than per head.
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
	TiedEmbd  bool    // no output.weight; the lm_head IS token_embd
	QKNorm    bool    // qwen3 RMSNorms each head of q and k before RoPE
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
	// graph.RopeSplit, and k's norm after it as q's is.
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
	// scan: see Scan. dtRank is its dt bottleneck, read off the first
	// scan block's x_proj at build(): the container states no rank.
	scan   bool
	DtRank int
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
	// SWAGlobalFirst makes the first layer of each period the global one
	// (ModernBERT's), not the last.
	SWAGlobalFirst bool

	// Logit softcapping: x = c*tanh(x/c). gemma2 caps the attention scores at
	// 50 and the output logits at 30. Zero means none. Omitting it gives fluent
	// wrong output that only an external oracle catches.
	AttnSoftcap, FinalSoftcap float32
	RopeBaseSWA               float64
	// RopeScaleSWA is the sliding layers' rotary magnitude where the
	// container states one of their own (jlm.Config.RopeScaleSWA); 0 is the
	// full layers'.
	RopeScaleSWA float64

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

	// geoms is each block's own geometry where the blocks differ
	// (jlm.Config.Geoms), nil when every block is the model's; mixed says
	// their attention geometries differ and localNRot is the rotary width the
	// blocks reading the second table turn. Set by setGeoms, read through
	// HeadDimAt and friends.
	geoms     []jlm.Geom
	mixed     bool
	localNRot int
	// VNorm RMSNorms each head of v with no weight before it is cached
	// (jlm.Flag2VNorm, Gemma 4).
	VNorm bool
	// bidir is, per layer, whether a picture's rows see each other in its
	// attention, as the container states it (jlm.Config.BidirLayers); nil
	// is every layer.
	bidir []bool
	// The last NKVShared layers attend to an earlier layer's history
	// (KVSource); PLEDim is each layer's slice of the per-layer embeddings,
	// zero for none.
	NKVShared, PLEDim int
	// DeepTaps is how many of the first blocks add a picture's tap rows into
	// its rows after they run (jlm.TowerTap, Qwen3-VL's deepstack): the
	// container's tower's taps, tap i after block i.
	DeepTaps int
	// sel is how the blocks' attention selects the history it reads, the
	// container's statement (jlm.Selection); nil for none.
	sel *jlm.Selection
	// HCMult is the residual streams a hyper-connected model carries.
	HCMult int
	// HCLowRank is the hyper-connections' gate rank, zero for none.
	HCLowRank int
	// AttnResBlock is a residual checkpoint every that many blocks, each
	// sublayer's input their softmax mix with the running residual (zero is
	// none).
	AttnResBlock int
	// DeltaDecay and DeltaGate are the delta rule's decay and output gate as
	// the container states them (jlm.Config.DeltaDecay), nil on a model with
	// no delta rule.
	DeltaDecay, DeltaGate *jlm.Program
	// stated is the residual's streams as the converter states them
	// (jlm.Streams, the container's own): N of them, each block's reads and
	// writes; nil for a config no container made.
	stated *jlm.Streams
	// expWidth is the routed experts' input and output width as their banks
	// state it (expertWidth), zero for n_embd.
	expWidth int
	// LatentNormEps is the MLA latent norms' epsilon.
	LatentNormEps float64
	// kvSrcAt replaces KVSource's choice for the shared blocks, a source a
	// block (nil: none), for a gate's violation only.
	KvSrcAt []int
	// noRouter marks the blocks of a mixture model that carry no router:
	// Nemotron 3's mixer-only layers and Jamba's dense blocks between its
	// mixtures. MoEAt is false there. Read off the container at build(), nil
	// when no block is one.
	noRouter []bool

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

// ropeLocal says block il turns by the sliding blocks' rotary where the
// model holds one: a sliding block.
func (c *Config) ropeLocal(il int) bool { return c.SWA(il) }

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
// so at period 6 layers 0..4 are local and layer 5 is global; under
// SWAGlobalFirst layer 0 is global and 1..5 local.
func (c *Config) SWA(il int) bool {
	if c.SWAGlobalFirst {
		return c.SWAWindow > 0 && c.SWAPeriod > 1 && il%c.SWAPeriod != 0
	}
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
		// A selection's key rides the same row, after the rotary key: it is
		// cached per position and per layer exactly as the latent is, so it
		// pages, relocates and is stored with it.
		return c.KVLoraRank + c.NRot + c.selKeyW()
	}
	return c.NKVHead * c.HeadDim
}

// MLA reports that this model's attention is over a compressed latent.
func (c *Config) MLA() bool { return c.KVLoraRank != 0 }

// Selection is how the blocks' attention selects the history it reads, nil
// where every block reads all of it.
func (c *Config) Selection() *jlm.Selection { return c.sel }

// SelectAt reports that block li's attention reads only the positions its
// indexer selects.
func (c *Config) SelectAt(li int) bool { return c.sel.At(li) }

// selKeyW is the width a selection's key takes in a latent attention's row,
// 0 for none.
func (c *Config) selKeyW() int {
	if c.sel == nil {
		return 0
	}
	return int(c.sel.KeyWidth)
}

// KeyHeadAt reports that layer li's cached row carries its selection's key
// as one more kv head after the attention's own.
func (c *Config) KeyHeadAt(li int) bool { return c.KVLoraRank == 0 && c.SelectAt(li) }

// KVRowAt is one cached position's width in layer li's history: KVDimAt,
// plus where it selects its keys the indexer's key as one more kv head.
// Every cache allocation, page and migration of a layer goes through it; the
// k and v projections are KVDimAt wide.
func (c *Config) KVRowAt(li int) int {
	if c.KeyHeadAt(li) {
		return (c.NKVHead + 1) * c.HeadDim
	}
	return c.KVDimAt(li)
}

// MixedGeom reports that the blocks do not all attend at one geometry: a
// block states its own head width, kv head count or rotary width. HeadDim,
// NKVHead and NRot are then the model's, and every per-layer path reads
// HeadDimAt, NKVHeadAt and NRotAt.
func (c *Config) MixedGeom() bool { return c.mixed }

// setGeoms takes the blocks' own geometries, gs one per block or none.
func (c *Config) setGeoms(gs []jlm.Geom) {
	c.geoms, c.mixed, c.localNRot = gs, false, 0
	for li := range gs {
		c.mixed = c.mixed || c.HeadDimAt(li) != c.HeadDim || c.NKVHeadAt(li) != c.NKVHead || c.NRotAt(li) != c.NRot
		if c.SWA(li) {
			c.localNRot = max(c.localNRot, c.NRotAt(li))
		}
	}
}

// LocalNRot is the rotary width the blocks reading the second table
// (Model.RopeSWA) turn: theirs where they state one, NRot otherwise.
func (c *Config) LocalNRot() int {
	if c.localNRot != 0 {
		return c.localNRot
	}
	return c.NRot
}

// HeadDimAt, NKVHeadAt and NRotAt are layer li's attention geometry: the
// block's own where the container states one, the model's otherwise.
func (c *Config) HeadDimAt(li int) int {
	if li >= 0 && li < len(c.geoms) {
		return int(c.geoms[li].HeadDim)
	}
	return c.HeadDim
}

// NKVHeadAt is layer li's kv head count (see HeadDimAt).
func (c *Config) NKVHeadAt(li int) int {
	if li >= 0 && li < len(c.geoms) {
		return int(c.geoms[li].NKVHead)
	}
	return c.NKVHead
}

// NRotAt is layer li's rotary width (see HeadDimAt).
func (c *Config) NRotAt(li int) int {
	if li >= 0 && li < len(c.geoms) {
		return int(c.geoms[li].NRot)
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

// MaxHeadDim, MaxKVDim, MaxQDim and MaxNRot are the widest of the blocks'
// geometries: what a buffer that serves every layer is sized by.
func (c *Config) MaxHeadDim() int {
	m := c.HeadDim
	for li := range c.geoms {
		m = max(m, c.HeadDimAt(li))
	}
	return m
}

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
	if li < len(c.KvSrcAt) {
		return c.KvSrcAt[li]
	}
	for j := c.NLayer - c.NKVShared - 1; j >= 0; j-- {
		if c.SWA(j) == c.SWA(li) {
			return j
		}
	}
	return -1
}

// NFFNAt is block li's feed-forward width: its own where the container
// states one, NFFN otherwise.
func (c *Config) NFFNAt(li int) int {
	if li >= 0 && li < len(c.geoms) {
		return int(c.geoms[li].NFFN)
	}
	return c.NFFN
}

// MaxNFFN is the widest feed-forward width of any block.
func (c *Config) MaxNFFN() int {
	m := c.NFFN
	for li := range c.geoms {
		m = max(m, c.NFFNAt(li))
	}
	return m
}

// MaxKVDim is the widest row any layer caches: MLA's latent row, an MSA
// block's k with the indexer's key after it, DeepSeek V4's widest KVRowAt, or
// else the widest of the blocks' KV widths.
func (c *Config) MaxKVDim() int {
	if c.KVLoraRank != 0 {
		return c.KVDim()
	}
	if c.sel != nil {
		// The row a selecting block writes: its k with the indexer's key
		// after it.
		return (c.NKVHead + 1) * c.HeadDim
	}
	m := c.KVDim()
	for li := range c.geoms {
		m = max(m, c.NKVHeadAt(li)*c.HeadDimAt(li))
	}
	return m
}

// MaxQDim is the query width at the widest head.
func (c *Config) MaxQDim() int { return c.NHead * c.MaxHeadDim() }

// MaxNRot is the widest rotary any block turns.
func (c *Config) MaxNRot() int {
	m := c.NRot
	for li := range c.geoms {
		m = max(m, c.NRotAt(li))
	}
	return m
}

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

// Attends reports whether any layer attends: none does in plain Mamba, whose
// one placeholder head is as wide as the embedding.
func (c *Config) Attends() bool {
	for i := range c.NLayer {
		if c.LayerKind(i).Attends() {
			return true
		}
	}
	return false
}

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

// Scan reports that the recurrent blocks are the per-element selective scan
// (jlm.LayerScan: Mamba, FalconMamba, Jamba): every channel a head of one
// row, the decay a vector over the state, dt off a low-rank bottleneck of the
// convolved x.
func (c *Config) Scan() bool { return c.scan }

// configFrom translates the container's typed description into the engine's.
//
// Parsing and architecture knowledge live in convert.configOf; this copies
// fields and derives. The two types differ on purpose: the container stores
// what the file says and this carries what the engine derived. stores says
// a block of the container stores a role.
func configFrom(c *jlm.Config, stores func(jlm.Role) bool) (*Config, error) {
	if c == nil {
		return nil, fmt.Errorf("model: the container has no config section")
	}
	// A container with no vocabulary is a tower alone.
	if !c.Arch.Valid() || c.NVocab == 0 {
		return nil, fmt.Errorf("model: %v is not a language model", c.Arch)
	}
	if c.HCMult != 0 && c.HCLowRank == 0 || len(c.CompKinds) != 0 || c.NHashLayers != 0 {
		return nil, fmt.Errorf("model: %v carries hyper-connections mixed by Sinkhorn rounds, compressed "+
			"attention or experts routed by token id, which have no composed form stated from the container", c.Arch)
	}
	out := &Config{
		Arch:         c.Arch.String(),
		chanDecay:    stores(jlm.RoleSSMFB),
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
		RopeScaleSWA: float64(c.RopeScaleSWA),
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

		HCMult:    int(c.HCMult),
		HCLowRank: int(c.HCLowRank),
	}
	out.setGeoms(c.Geoms)
	out.VNorm = c.Flags2.Has(jlm.Flag2VNorm)
	out.NKVShared, out.PLEDim = int(c.NKVShared), int(c.PLEDim)
	out.bidir = c.BidirLayers
	out.sel = c.Selection
	if s := out.sel; s != nil && !out.MLA() && (s.KeyWidth != uint32(out.HeadDim) ||
		s.Groups != 1 && s.Groups != uint32(out.NKVHead) || out.MixedGeom() || out.NKVShared != 0) {
		return nil, fmt.Errorf("model: %s: a selection of %d keys %d wide over %d kv heads of %d", out.Arch,
			s.Groups, s.KeyWidth, out.NKVHead, out.HeadDim)
	}
	if c.PredStreams != 0 {
		return nil, fmt.Errorf("model: %s: the container states %d predicted streams in fields of their own, which this "+
			"build reads only as the residual's stated streams: convert the model again", out.Arch, c.PredStreams)
	}
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
	// A scale derived from the head width would be one scale per geometry,
	// and the one field is one.
	if out.MixedGeom() && (out.MLA() || out.Hybrid() || out.QKNormWide || out.AttnOutGate ||
		c.AttnScale == 0 || out.QKL2Norm || out.ClampKQV != 0) {
		return nil, fmt.Errorf("model: %s: blocks of several geometries on MLA, a hybrid, a whole-"+
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
		case jlm.LayerScan:
			out.scan = true
		}
	}
	if out.scan {
		for i, k := range c.LayerKinds {
			if k.Recurrent() && k != jlm.LayerScan {
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
	out.TokenAct = c.ActQuant == jlm.ActQuantRowInt8
	out.HCLowRank = int(c.HCLowRank)
	out.DeltaDecay, out.DeltaGate = c.DeltaDecay, c.DeltaGate
	out.AttnResBlock = int(c.AttnResBlock)
	out.LatentNormEps = float64(c.LatentNormEps)
	if out.LatentNormEps == 0 {
		out.LatentNormEps = out.RMSEps
	}
	switch {
	case out.AttnResBlock < 0 || out.AttnResBlock != 0 && (out.HCMult != 0 || out.Parallel ||
		out.NMTP != 0):
		return nil, fmt.Errorf("model: %s: a checkpoint bank every %d blocks beside another residual "+
			"structure", out.Arch, out.AttnResBlock)
	}
	out.stated = &c.Streams
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
	return out, nil
}

// tensor is one weight matrix, kept as the raw mapping plus its shape.
// f32Tensor is vals as an F32 matrix of rows x k, a table the engine built.
func F32Tensor(vals []float32, rows, k int) Tensor {
	b := unsafe.Slice((*byte)(unsafe.Pointer(&vals[0])), 4*len(vals))
	return Tensor{Typ: quant.F32, Data: b, Rows: rows, K: k}
}

type Tensor struct {
	// e is the container entry this weight came from: its role, its block and
	// its shape. Kept for diagnostics and for rows(), never for resolution --
	// a tensor is found through jlm.File.Find, by what it IS.
	E *jlm.Entry
	// typ is a copy of the entry's type, so the hot path never chases e.
	Typ  quant.Type
	Data []byte
	Rows int // ne1, output features
	K    int // ne0, input features
	// kp is the width the weight was packed at when its k is not a whole
	// number of quantization blocks (a tower 144 wide): the converter pads
	// it with zero columns to kp, and every product quantizes its k-wide
	// activation as kp with a zero tail (nn's ragged quantizer on the host,
	// kernels.QuantizePadded on a device). Zero where k is the packed width.
	kp int
	// packed is this weight in the device layout, set for a packed container
	// type; nil for a verbatim one.
	Packed *nn.Packed
}

// vector is block l's vector the file stores under r, nil for none.
func (l *Layer) Vector(r jlm.Role) []float32 { return nn.VectorOf(l.Vecs, r) }

// roleTensor is one of a block's matrices and the role it was loaded under.
type roleTensor struct {
	Role jlm.Role
	T    *Tensor
}

// matrix is block l's matrix loaded under r, nil for none.
func (l *Layer) Matrix(r jlm.Role) *Tensor {
	for _, m := range l.Mats {
		if m.Role == r {
			return m.T
		}
	}
	return nil
}

type Layer struct {
	// vecs is every vector the file stores for the block, by role
	// (jlm.Role.Expanded), read once at load. The named vectors below are
	// these slices, not copies.
	Vecs []nn.RoleVector
	// mats is every matrix of the block's the loader bound, by the role it
	// was loaded under, pointing at the field that holds it: one role can
	// land in different fields by block kind (RoleAttnQKV in wq or ssmIn).
	Mats []roleTensor

	AttnNorm, FfnNorm []float32 // norms are tiny and always F32; kept expanded
	// attnNormB and ffnNormB are a LayerNorm's biases (C6), nil for RMSNorm
	// and for a bias-free LayerNorm. ffnNorm itself is nil on a parallel block
	// that shares one norm between its two branches (phi-2): the FFN reads the
	// attention's normed row.
	AttnNormB, FfnNormB []float32
	// clamp is a vision block's clipped-linear bounds (jlm.RoleVClamp): for
	// q, k, v, o, gate, up and down, the input's and the output's minimum
	// and maximum. nil everywhere but Gemma 4's tower.
	Clamp []float32
	// qNorm and kNorm are head_dim wide and nil unless the architecture has
	// them (qwen3). nil is the llama case, not an error path.
	QNorm, KNorm   []float32
	Wq, Wk, Wv, Wo Tensor
	// vFromK says the block has no v projection and v is k's (Gemma 4's
	// global layers): wv is unbound and every path copies k's projection
	// before k's norm and rotary run.
	VFromK bool
	// outScale multiplies the block's output row after its last residual add
	// (Gemma 4's layer_scalar). Zero means none. outScaleRow is it as the
	// container stores it, the vector a composed block binds.
	OutScale    float32
	OutScaleRow []float32
	// Multi-head Latent Attention, all nil unless Config.KVLoraRank is set.
	// wq is then the one-step query (DeepSeek-V2-Lite, q_lora_rank null) and
	// wqa/wqb the two-step one; exactly one of those is populated.
	//
	// wkb and wvb are banks, one sheet per head, read through bankExpert as an
	// expert bank is, so the absorb runs as a single gather.
	Wqa, Wqb, Wkva, Wkb, Wvb Tensor
	// kbHead and vbHead are wkb and wvb cut into one view per head, built once
	// at load as l.experts is.
	kbHead, vbHead []Tensor
	// headsFrom is what wkb and wvb spanned when the heads were last cut, and
	// banksFrom the same for the expert bank views: bindPacked re-cuts only
	// when a bank moved, since a cut is a fresh nn.Packed per view and a
	// resident block is rebound every token.
	headsFrom      [2]bankSpans
	banksFrom      [3]bankSpans
	Gate, Up, Down Tensor
	// router is nil unless this layer's FFN is a mixture of experts, in which
	// case gate/up/down hold the three 3-D banks and experts the per-expert
	// views of them. The router tensor's presence is the graph's branch.
	Router Tensor
	// legacyExperts marks the Mixtral-era layout: eight separate 2-D tensors
	// per role instead of one stacked 3-D bank. gate/up/down are then empty and
	// only l.experts is populated, so offerRange declines the block by name.
	// The converter no longer writes it for a uniform block; see build().
	LegacyExperts bool
	Experts       []expert
	// ungatedExp is a mixture whose experts have no gate bank (Nemotron 3):
	// each is down(act(up(h))), the activation alone, as an ungated FFN is.
	UngatedExp bool

	// The shared expert runs for every token beside the routed ones; its
	// gate, where it has one, is a vector in vecs.
	ShGate, ShUp, ShDown Tensor

	// The linear layer's weights. A hybrid's recurrent block projects the
	// residual into one fused q|k|v vector, which lives in wq; the layer kind
	// decides what reads it.
	//
	// ssmGate is the z projection whose SiLU gates the block's output; the full
	// attention layers of the same model fold their gate into wq instead and
	// activate it with sigmoid. nil on every non-hybrid.
	SsmGate, SsmBA, SsmOut Tensor
	// ssmIn is a Mamba-2 mixer's x|B|C projection on a block that also runs
	// attention (Falcon-H1), where wq is the attention's query. Every other
	// recurrent block carries it in wq; mixIn picks.
	SsmIn Tensor
	// ssmFA/ssmFB and ssmGA/ssmGB are Kimi Delta Attention's two low-rank
	// gates: the forget gate, whose softplus becomes the per-channel decay,
	// and the output gate that replaces ssmGate. nil on qwen3next, which has
	// one matrix for the second and derives the first from ssmBA's interleaved
	// alpha stream.
	SsmFA, SsmFB, SsmGA, SsmGB Tensor
	// ssmConv1d is the convolution kernel tap-major, as the file stores it.
	// The block's vectors (A, the dt bias, the norm, D, the convolution bias,
	// Mamba-1's three norms) are in vecs.
	SsmConv1d []float32
	// ssmConvTaps is the convolution kernel as the file stores it, the taps
	// of a channel contiguous: the composer's OpStateConv reads it so.
	SsmConvTaps Tensor
	// ssmXDt, ssmXB and ssmXC are a Mamba-1 block's x_proj (the dt
	// bottleneck, B and C, each of the convolved x), and ssmDtProj takes the
	// bottleneck back up to the channels.
	SsmXDt, SsmXB, SsmXC, SsmDtProj Tensor
	// noFFN is a block that is its mixer alone: plain Mamba-2's, and a
	// Nemotron-H mixer layer not followed by an MLP. It ends at the first
	// residual on every entry point and every tier.
	NoFFN bool

	// mtp is a multi-token-prediction block's own weights, nil on every trunk
	// block. See jlm.Config.NMTP.
	Mtp *mtpWeights
}

// mtpWeights is what a prediction block has beyond an ordinary block: the two
// input norms and eh_proj that make its residual from the next token's
// embedding and the trunk's hidden state, and its head norm. head and embd are
// the block's own copies where the container kept them (they differed from the
// trunk's); otherwise they are the trunk's, as headNorm is the trunk's output
// norm where the block has none.
type mtpWeights struct {
	EhProj                 Tensor
	Enorm, Hnorm, HeadNorm []float32
	Head, Embd             Tensor
	// ownEmbd says embd is the block's own table, not the trunk's.
	ownEmbd bool
}

// expert is one of a MoE layer's experts, as three aliasing views of its layer's
// three banks; no copy.
type expert struct{ Gate, Up, Down Tensor }

// Model is a loaded transformer.
type Model struct {
	// vecs is every vector the file stores for the model rather than a block
	// (jlm.DenseBlock, jlm.Role.Expanded), read once at load; the named ones
	// below (outNorm, outNormB, ...) are these slices.
	Vecs []nn.RoleVector
	// vecMats is each of those that is a table (the token-type rows) as a
	// float matrix over the same slice, which an embedding binds as one.
	VecMats []roleTensor
	// DenseMats is every matrix the model stores outside its blocks that no
	// field of Dense names, by role: what a stated head or the streams' mix
	// binds.
	DenseMats []roleTensor
	// splitBank is the Data of a bank held in expert pages (bankWeight).
	splitBank [1]byte
	// holdFree is the recycled sheet holds (sheetOf).
	holdMu   sync.Mutex
	holdFree []*sheetHold
	// xdKNormFirst is a gate's violation: k normed before XD-RoPE's
	// rotary, as every other q/k-normed model does. The rotation does not
	// keep a head's RMS, so HunyuanVL's post-rotary norm is load-bearing.
	xdKNormFirst bool
	// mnNoResidual is a gate's violation: Gemma 3n's MobileNet blocks with
	// their residual adds dropped from the graphs the front end states.
	Cfg   *Config
	Vocab *tok.Vocab

	// chatTemplates are the chat templates compiled, by source: a request
	// renders its model's template, not compiles it (chat.go).
	chatTemplates sync.Map
	// preloadStop, preloadDone and preloadErr are WithPreload's background
	// read: Close closes the first and waits on the second.

	// TokErr is why Vocab is nil, when it is. The weights still loaded.
	TokErr error

	Container *jlm.File
	// load is what the caller asked of the model's frames at Open
	// (NewFrames).
	load ModelLoad
	// front is the model's graphs as the composer reads them (front.go).
	Front frontState
	// tower is the container's stated vision tower lowered into graphs
	// (tower.go), nil for a model with none. Its blocks are layers from
	// towerBase on, after every text and prediction block; towerDense is its
	// entry's and head's tensors, which no block holds.
	Tower      *TowerGraphs
	TowerBase  int
	TowerDense Layer
	// towerTaps is each tap's tensors (jlm.TowerTap): the dense region's at
	// its block's index.
	TowerTaps []Layer

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
	expBytes int64
	expCalls int64

	Embd Tensor
	// rope is the model's rotary configuration, built once at load.
	Rope nn.Rope
	// ropeSWA is the LOCAL layers' rotary where an architecture trains two
	// bases, and nil where it trains one -- which is every model but gemma3.
	RopeSWA *nn.Rope
	// Ngram is the n-gram block's table.
	Ngram *ngramTable
	// onesHD is a HeadDim-wide vector of ones: the weight Llama 4's weightless
	// q/k RMSNorm runs the generated norm against. nil unless QKL2Norm.
	onesHD  []float32
	OutNorm []float32
	// outNormB and outB are the output LayerNorm's bias and the head's bias
	// (phi-2), nil everywhere else.
	OutNormB, OutB []float32
	// posEmbd is a learned absolute position table (starcoder), one row per
	// position, added to the token's embedding before block 0. Empty on every
	// model that encodes position with a rotary; its presence is the fact.
	PosEmbd Tensor
	// pleTok and pleProj are Gemma 4's per-layer embedding table and the
	// projection of the scaled embedding to the same NLayer*PLEDim width, and
	// pleNorm the RMSNorm each layer's PLEDim slice of the projection takes
	// (Config.PLEDim). Empty everywhere else.
	PleTok, PleProj Tensor
	PleNorm         []float32
	// lnZeros is the zero bias a device's LayerNorm is handed where the file
	// has none; see lnBias.
	lnZeros     []float32
	lnZerosOnce sync.Once
	Output      Tensor
	Layers      []Layer
	// actWindow is how many activation elements share one int8 scale, the
	// same on every device (actWindowOf).
	ActWindow int
	// gc is where the weights live and what their budget leaves the
	// collector (WithGCPlan).
	Gc sched.GCPlan
	// chatClock is the clock a chat template's strftime_now reads; nil is the
	// wall clock (WithChatClock).
	chatClock func() time.Time
	// dense1 and dense2 are an embedding model's projection heads after
	// pooling (EmbeddingGemma's), empty for every other model.
	Dense1, Dense2 Tensor

	// id names this model to a device (nn.LayerPlan.Model), which holds one
	// model's blocks at a time.
	Id uint64
}

// Close closes the container. It reaches no device: what a device keeps of
// the model went with the last instance holding it (nn.ModelOwner), and
// every instance of the model has closed first.
func (m *Model) Close() error {
	if m.Container == nil {
		return nil
	}
	return m.Container.Close()
}

// newModelID is a random id for an opened Model. Zero is never handed out, so
// a plan that forgot to say whose it is reads as nobody's. It is random so
// that a session's default cache name, which carries it, is not guessable
// across processes sharing a store.
func newModelID() uint64 {
	for {
		var b [8]byte
		if _, err := rand.Read(b[:]); err != nil {
			// A process that cannot read randomness must not fall back to a
			// guessable name; the timestamp is only a tiebreak.
			binary.LittleEndian.PutUint64(b[:], uint64(time.Now().UnixNano()))
		}
		if id := binary.LittleEndian.Uint64(b[:]); id != 0 {
			return id
		}
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
	m, err := openContainer(path, options...)
	if err != nil {
		return nil, err
	}
	// Where the composer serves, a container written before the converter
	// stated its activations would run every block on the tier's own kernels
	// and say nothing: refused, with the command that states them.
	if err := m.statesActivations(path); err != nil {
		m.Close()
		return nil, err
	}
	// Sliding layers with a scaled rotary or beside a tower read what a
	// container written before StatementRevision does not state.
	if c := m.Container.Config(); c.Revision < jlm.StatementRevision && c.SWAWindow > 0 &&
		(c.YarnFactor > 1 || m.Container.Vision() != nil) {
		m.Close()
		return nil, &UpgradeRequiredError{Path: path, What: "sliding layers"}
	}
	// A block of sub-norms before its projections reads the activation
	// quantization a container written before revision 2 does not state.
	if c := m.Container.Config(); c.Revision < 2 && m.storesAny(jlm.RoleAttnSubNorm, jlm.RoleFFNSubNorm) {
		m.Close()
		return nil, &UpgradeRequiredError{Path: path, What: "activation quantization"}
	}
	if (m.Cfg.AttnResBlock != 0 || m.Cfg.HCLowRank != 0) && m.Cfg.statedStreams() <= 1 {
		m.Close()
		return nil, &UpgradeRequiredError{Path: path, What: "residual streams"}
	}
	if cc := m.Container.Config(); cc.IdxHeads != 0 && cc.Selection == nil && len(cc.CompKinds) == 0 {
		m.Close()
		return nil, &UpgradeRequiredError{Path: path, What: "the history its attention selects"}
	}
	return m, nil
}

// statesActivations is an UpgradeRequiredError where the container states no
// activation and the front end refuses a block for wanting one.
func (m *Model) statesActivations(path string) error {
	if len(m.Container.Config().Acts) > 0 {
		return nil
	}
	for li := range m.Cfg.NLayer {
		if m.Uncovered(li) == UncoveredAct {
			return &UpgradeRequiredError{Path: path, What: "activations"}
		}
	}
	return nil
}

// UpgradeRequiredError is a container written before this build states what
// the composer reads of it. It carries the upgrade command.
type UpgradeRequiredError struct{ Path, What string }

// Error names the file, what it does not state and the command that states
// it, in place.
func (e *UpgradeRequiredError) Error() string {
	return fmt.Sprintf("jitllm: %s was written before this build states its %s, which the composer reads.\n"+
		"  Upgrade it once:  jitllm upgrade %s", e.Path, e.What, e.Path)
}

// Option configures a model's load: its vocabulary, where its weights
// live, and its chat template's clock.
//
// Every residency decision must be forcible by the caller rather than
// inferred. tok.Option is about the vocabulary, so this wraps it rather than
// replacing it.
type Option func(*ModelLoad)

type ModelLoad struct {
	tok []tok.Option
	// gc is WithGCPlan's, chatClock WithChatClock's.
	gc        sched.GCPlan
	ChatClock func() time.Time
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
	return func(l *ModelLoad) { l.preload = depth }
}

// WithTokenizerOption carries a tok.Option through Open, for a caller that
// already holds one.
func WithTokenizerOption(o tok.Option) Option {
	return func(l *ModelLoad) { l.tok = append(l.tok, o) }
}

// WithTokenizer asserts the model's own tokenizer.json over the name the
// converter wrote -- RULE 7m's fourth and outranking authority.
func WithTokenizer(r io.Reader) Option { return WithTokenizerOption(tok.WithTokenizer(r)) }

// WithGCPlan says where the model's weight memory lives and what a budget
// it sizes itself leaves the Go collector: the zero plan, the default, keeps
// the weights off the heap; OnHeap puts them on it (jlm.OnHeap), the control
// arm for a measurement, with Reserve's collector reserve (AGENTS.md RULE 2f).
func WithGCPlan(g sched.GCPlan) Option { return func(l *ModelLoad) { l.gc = g } }

// MappedBytes is how much of the model's weight memory is held off the Go
// heap (jlm.File.MappedBytes).
func (m *Model) MappedBytes(fr *jlm.Frames) uint64 {
	if fr == nil {
		return 0
	}
	return fr.MappedBytes()
}

// WithHugePages sets whether this model's page frames are advised onto
// transparent huge pages. On by default: the plane layout jumps a whole plane
// per payload word, and on 4 KB pages that walks the page tables on nearly
// every access (docs/engineering-history/cpu-kernels.md, "Huge pages for the
// frames").
func WithHugePages(on bool) Option {
	return func(l *ModelLoad) { l.place.NoHugePages = !on }
}

// WithChunk sets this model's pager fill granularity in bytes, overriding the
// container's own: a power of two of at least 4096. Zero keeps the container's.
func WithChunk(bytes uint64) Option {
	return func(l *ModelLoad) { l.chunk = bytes }
}

// WithInterleave spreads this model's weight memory page by page over the
// given NUMA nodes. Fewer than two leaves placement to first touch.
func WithInterleave(nodes []int) Option {
	return func(l *ModelLoad) { l.place.Nodes = nodes }
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
	var lo ModelLoad
	for _, o := range options {
		o(&lo)
	}
	// Refused at Open, where the caller asked, not when a device first makes
	// frames over the model.
	if lo.chunk > 0 {
		if err := jlm.CheckChunk(lo.chunk); err != nil {
			return nil, err
		}
	}
	c, err := jlm.Open(path)
	if err != nil {
		return nil, err
	}
	m, err := build(c, lo.tok...)
	if err == nil {
		err = loadDenseHeads(c, m)
	}
	if err != nil {
		c.Close()
		return nil, err
	}
	m.Container, m.load = c, lo
	m.Gc, m.chatClock = lo.gc, lo.ChatClock
	m.ActWindow = m.ActWindowOf()
	m.Id = newModelID()
	m.TowerBase = len(m.Layers)
	if v := c.Vision(); v != nil && v.Tower != nil {
		if err := m.bindTower(c); err != nil {
			c.Close()
			return nil, err
		}
	}
	// A tower is the container's statement of it (jlm.Tower), lowered into
	// graphs and verified here; an unstated one is refused by its family's
	// name.
	if c.Vision() != nil {
		tg, err := m.BuildTowerGraphs()
		if err != nil {
			c.Close()
			return nil, err
		}
		m.Tower = tg
	}
	return m, nil
}

// NewFrames is frames over the model's container as the caller asked at
// Open -- the placement, the fill granularity, the heap and the preload --
// for the device that pages the model's blocks to hold; the device budgets
// them by the quotas of the instances it pages them for. The model
// holds none: an instance binds every block and dense tensor it runs at the
// frames of the device running it.
func (m *Model) NewFrames(opts ...jlm.FramesOption) (*jlm.Frames, error) {
	lo := m.load
	if lo.gc.OnHeap {
		opts = append(opts, jlm.OnHeap())
	}
	fr, err := jlm.NewFrames(m.Container, opts...)
	if err != nil {
		return nil, err
	}
	fr.SetPlacement(lo.place)
	if lo.chunk > 0 {
		if err := fr.SetChunk(lo.chunk); err != nil {
			fr.Close()
			return nil, err
		}
	}
	if lo.preload > 0 {
		fr.StartPreload(lo.preload)
	}
	return fr, nil
}

// ReadConcurrency is the most read requests the container ever had
// outstanding at once, and ReadBytes the bytes they asked for.
func (m *Model) ReadConcurrency() int64 {
	if m.Container == nil {
		return 0
	}
	return m.Container.ReadConcurrency()
}

// PreloadDepth is the most pages a WithPreload read had in flight at once.
func (m *Model) PreloadDepth(fr *jlm.Frames) int64 {
	if m.Container == nil || fr == nil {
		return 0
	}
	return fr.PreloadDepth()
}

func (m *Model) ReadBytes() int64 {
	if m.Container == nil {
		return 0
	}
	return m.Container.ReadBytes()
}

// PagerReads is how many read calls the pager has issued. A streamed mixture is
// priced in requests rather than bytes, since a routed bank asks for many small
// scattered ranges.
func (m *Model) PagerReads() int64 {
	if m.Container == nil {
		return 0
	}
	return m.Container.Reads()
}

// FreshFrames is how many frames the pager allocated rather than reused: on a
// steady run it stops growing once the budget is full, and a figure that keeps
// pace with the page-ins is frames being freed and remade.
func (m *Model) FreshFrames(fr *jlm.Frames) int64 {
	if m.Container == nil || fr == nil {
		return 0
	}
	return fr.FreshFrames()
}

// StreamGroups is the container's own answer to how many pieces a streamed
// block's selection should be uploaded in. See jlm.streamGroupsFor.
func (m *Model) StreamGroups() int {
	if m.Container == nil {
		return 0
	}
	return m.Container.StreamGroups()
}

// ChunkBytes is the granularity this container was opened at.
func (m *Model) ChunkBytes(fr *jlm.Frames) uint64 {
	if m.Container == nil || fr == nil {
		return 0
	}
	return fr.ChunkBytes()
}

// BytesRead is what those chunks actually cost, using this container's
// granularity rather than the package global. See jlm.File.ChunkBytes.
func (m *Model) BytesRead(fr *jlm.Frames) uint64 {
	if m.Container == nil || fr == nil {
		return 0
	}
	return uint64(fr.ChunksIn()) * fr.ChunkBytes()
}

// PagedBytes is the bytes every device's frames have filled from the
// model's container, and PageFaults the pages they faulted.
func (m *Model) PagedBytes() uint64 {
	if m.Container == nil {
		return 0
	}
	return m.Container.PagedBytes()
}

func (m *Model) PageFaults() int64 {
	if m.Container == nil {
		return 0
	}
	return m.Container.PageFaults()
}

// DirectIO reports whether this model's pages are read without going through
// the kernel's page cache, which decides how much of a memory ceiling a weight
// budget may claim. See jlm.File.Direct.
func (m *Model) DirectIO() bool { return m.Container != nil && m.Container.Direct() }

// HostResidentBlocks is how many of this model's block pages the host is
// currently holding. RSS cannot stand in for it: a released page shows only
// when the collector runs, and the KV cache dwarfs the weights anyway.
func (m *Model) HostResidentBlocks(fr *jlm.Frames) int {
	if m.Container == nil || fr == nil {
		return 0
	}
	n := 0
	for i := 0; i < len(m.Layers); i++ {
		if fr.Resident(i) {
			n++
		}
	}
	return n
}

// Blocks is the size of the model's block space: its text blocks, any
// prediction blocks, then its vision tower's, which are blocks like any other
// -- placed, paged and relocated the same way.
func (m *Model) Blocks() int { return len(m.Layers) }

// BlockResident reports whether block li's page is in host memory now. A block
// that runs on the host and is not resident is read from the file on its next
// pass.
func (m *Model) BlockResident(fr *jlm.Frames, li int) bool {
	return m.Container != nil && fr != nil && fr.Resident(li)
}

// PageSize is one block page's size in bytes, or zero for a model with no
// container behind it.
func (m *Model) PageSize() uint64 {
	if m.Container == nil {
		return 0
	}
	return m.Container.H.PageSize
}

// blockBytes is what block li's tensors occupy in the container's device
// layout, and the routed expert banks' share of that, from the tensor table
// alone: nothing is read. Zero for a model with no container.
func (m *Model) BlockBytes(li int) (total, bank uint64) {
	if m.Container == nil {
		return 0, 0
	}
	es := m.Container.Entries()
	for i := range es {
		e := &es[i]
		if e.Block == jlm.DenseBlock || m.Container.BlockOf(e) != li {
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
func (m *Model) MeanBase() uint64 {
	var sum uint64
	n := 0
	for li := 0; li < m.Cfg.NLayer; li++ {
		total, bank := m.BlockBytes(li)
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
	if m.Container == nil {
		return 0
	}
	var n uint64
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		// The same predicate jlm.alwaysResident uses: expanded tensors carry a
		// block number but live in the dense region.
		if e.Block != jlm.DenseBlock && !e.Role.Expanded() {
			continue
		}
		qs, d, sc := jlm.SpanLenOf(e)
		n += qs + d + sc
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
func (m *Model) HostBytes(fr *jlm.Frames) uint64 {
	if m.Container == nil || fr == nil {
		return 0
	}
	return fr.ResidentBytes()
}

// PageBudget is the byte cap the container holds its block pages against, 0
// when unlimited (see HostBytes).
func (m *Model) PageBudget(fr *jlm.Frames) uint64 {
	if m.Container == nil || fr == nil {
		return 0
	}
	return fr.Budget()
}

// PagesFit reports whether the page budget holds every page of the container,
// so that nothing is ever evicted. A model with no container has nothing to
// page. It is the budget's answer, not the residency's: a model that fits may
// not have faulted a page in yet.
func (m *Model) PagesFit(fr *jlm.Frames) bool {
	return m.Container == nil || fr == nil || !fr.CanEvict()
}

// PageBytes is block li's own page size, which differs between a text block
// and a vision block in the same container.
func (m *Model) PageBytes(li int) uint64 {
	if m.Container == nil {
		return 0
	}
	return m.Container.PageBytes(li)
}

// PageStats is the budget in pages and what it has cost so far.
func (m *Model) PageStats(fr *jlm.Frames) (frames int, in, out int64) {
	if m.Container == nil || fr == nil {
		return 0, 0, 0
	}
	in, out = fr.Faults()
	return fr.ResidentPages(), in, out
}

// pageIn faults block li in and re-points its weights at the frame it landed
// in. A resident block costs one bounds check and one nil compare.
//
// A frame is reused, so after an eviction block i's bytes are elsewhere and
// anything holding the old slice reads another block's weights.
// PageIn reads block li's own weights into its page and binds l, a copy of
// the block (Layer.CloneTo), at them. An instance binds its own copy, so no
// other instance's binding moves; the caller serialises binds of one copy.
func (m *Model) PageIn(fr *jlm.Frames, li int, l *Layer) error {
	// Neither "no pool" nor Resident(li) means this block's bytes are in:
	// Resident only says a frame is allocated, and a partial read (a streamed
	// block's expert sheets) can claim one with the block's own weights unread.
	// So always fall through to ensureBlock, which reads nothing when every
	// chunk is filled; that is what makes a partial read composable.
	//
	// A model can have no block pages at all: an all-F32 model keeps
	// everything in the dense region (jlm.alwaysResident).
	if m.Container == nil || li >= len(m.Layers) || li >= m.Container.NPages() {
		return nil
	}
	// The block's own weights, not its experts: moe() ensures those after the
	// router has chosen. EnsureRanges tracks which chunks hold real bytes, so an
	// expert nobody ensured is absent rather than stale, and every consumer of
	// a packed span has to ensure it.
	if err := m.ensureBlock(fr, li); err != nil {
		return err
	}
	return bindPacked(fr, m.Cfg, l)
}

// BindBlock gives block li a frame in fr, claimed and unread, and binds l at
// it: every span the right length, no byte valid until PageIn (or an expert
// ensure) reads it. A device deciding whether to take the block reads shapes
// and lengths only, so it decides before anything is read.
func (m *Model) BindBlock(fr *jlm.Frames, li int, l *Layer) error {
	if m.Container == nil || li >= len(m.Layers) || li >= m.Container.NPages() {
		return nil
	}
	if err := fr.Claim(li); err != nil {
		return err
	}
	return bindPacked(fr, m.Cfg, l)
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
func (m *Model) PageInSelected(fr *jlm.Frames, li int, sel []uint32, l *Layer) error {
	if m.Container == nil || li >= int(m.Container.H.NBlocks) {
		return nil
	}
	// A bank in expert pages reaches the device through Sheet, which holds
	// each selected expert's own page; the block page has nothing it needs.
	if l.ExpBank().E != nil && m.Container.ExpertPaged(l.ExpBank().E) {
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
		return m.PageIn(fr, li, l)
	}
	if err := fr.EnsureRanges(li, rs); err != nil {
		return err
	}
	return bindPacked(fr, m.Cfg, l)
}

// name is what to call this weight in an error. It is the container's
// diagnostic label, and nothing resolves through it.
func (t Tensor) name() string {
	if t.E == nil {
		return "a weight"
	}
	if t.E.Name != "" {
		return t.E.Name
	}
	return t.E.Role.String()
}

// entryTensor is the matrix entry e holds, as a descriptor: its shape and
// entry, no bytes. bindOne points a copy of it at its spans in the frames of
// the device running it.
func entryTensor(e *jlm.Entry) (Tensor, error) {
	if e.NDim < 2 {
		return Tensor{}, fmt.Errorf("model: %v has %d dimensions, want 2", e.Role, e.NDim)
	}
	typ, ok := jlm.SourceType(e.Type)
	if !ok || !quant.Dequantable(typ) {
		return Tensor{}, fmt.Errorf("model: %v is %v, which has no dequantizer yet", e.Role, e.Type)
	}
	// rows is the product of every trailing dimension: for a 3-D expert
	// bank ne1 alone would describe one expert, and a matvec would
	// silently compute expert 0.
	nrows := 1
	for d := 1; d < int(e.NDim); d++ {
		nrows *= int(e.Dims[d])
	}
	tn := Tensor{E: e, Typ: typ, Rows: nrows, K: int(e.Dims[0])}
	// Marked packed before the spans exist, since bankExpert and rows()
	// need the layout; bindOne fills the struct later.
	if jlm.Packed(e.Type) {
		tn.Packed = &nn.Packed{}
	}
	return tn, nil
}

// entryVector is the vector entry e holds, read once from the file into
// floats of the model's own.
func entryVector(c *jlm.File, e *jlm.Entry) ([]float32, error) {
	typ, ok := jlm.SourceType(e.Type)
	if !ok {
		return nil, fmt.Errorf("model: %v is %v, which has no dequantizer yet", e.Role, e.Type)
	}
	n := 1
	for d := 0; d < int(e.NDim); d++ {
		n *= int(e.Dims[d])
	}
	qs, err := c.ReadPayload(e)
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	if err := quant.Dequant32(typ, qs, out); err != nil {
		return nil, fmt.Errorf("model: %v: %w", e.Role, err)
	}
	return out, nil
}

// bindOne points a tensor at its already-packed spans in the container.
//
// data is set to the payload span, not left nil: the tier keys residency on
// &data[0] plus the shape (resKey), and several paths read an empty Data as
// "no weight here".
func bindOne(c *jlm.Frames, t *Tensor) {
	if t == nil || t.E == nil {
		return
	}
	e := t.E
	qs, d, sc := c.Span(e)
	if len(qs) == 0 {
		return
	}
	// Only what moved is written: on a model that fits the answer never
	// changes, and an unconditional write races another session's matvec
	// reading the same field (TestConcurrentDecodeOnAFittingMixture).
	if SameSpan(t.Data, qs) && (t.Packed == nil || !jlm.Packed(e.Type) ||
		(SameSpan(t.Packed.QS, qs) && SameSpan(t.Packed.D, d) && SameSpan(t.Packed.SC, sc))) {
		return
	}
	t.Data = qs
	// A type the container carries verbatim (F32/F16 norms, biases, routers)
	// is not packed; labelling it packed gives wrong numbers.
	if !jlm.Packed(e.Type) {
		return
	}
	// Reused, not reallocated: a fresh nn.Packed per fault would be garbage
	// proportional to the paging rate.
	if t.Packed == nil {
		t.Packed = &nn.Packed{}
	}
	t.Packed.QS, t.Packed.D, t.Packed.SC = qs, d, sc
}

// sameSpan reports whether two slices are the same memory, not merely equal
// bytes: a rebind that lands on the frame it was already bound to.
func SameSpan(a, b []byte) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

// sameView is sameSpan for a whole weight: its payload and, if packed, its
// three planes and its shape.
func sameView(a, b Tensor) bool {
	if a.Rows != b.Rows || a.K != b.K || !SameSpan(a.Data, b.Data) || (a.Packed == nil) != (b.Packed == nil) {
		return false
	}
	return a.Packed == nil || (SameSpan(a.Packed.QS, b.Packed.QS) && SameSpan(a.Packed.D, b.Packed.D) &&
		SameSpan(a.Packed.SC, b.Packed.SC) && a.Packed.Row == b.Packed.Row && a.Packed.Stride == b.Packed.Stride)
}

// ensureBlock reads the parts of block li that every token needs: its
// attention or recurrent weights, its router, its shared expert -- everything
// but the routed expert banks.
func (m *Model) ensureBlock(fr *jlm.Frames, li int) error {
	m.basePlanOnce.Do(m.buildBasePlans)
	if li >= 0 && li < len(m.basePlans) {
		return fr.EnsureRanges(li, m.basePlans[li])
	}
	return m.ensureBlockRanges(fr, li, false)
}

// buildBasePlans fills basePlans with ensureBlockRanges' plan for each block,
// the vision tower's included. EnsureRanges only reads a plan, so one slice
// serves every session.
func (m *Model) buildBasePlans() {
	c := m.Container
	if c == nil {
		return
	}
	m.basePlans = make([][]jlm.Range, len(m.Layers))
	for li := range m.basePlans {
		m.basePlans[li] = m.blockPlan(li, false)
	}
}

// ensureBlockExperts reads the whole block, routed expert banks included. A
// device is handed the entire bank at PrepLayer and cannot come back for an
// expert later, so it must be read first; otherwise the upload takes whatever
// the frame held (zeros, or another block's bytes). See
// docs/engineering-history/placement.md 15j.
func (m *Model) EnsureBlockExperts(fr *jlm.Frames, li int) error {
	return m.ensureBlockRanges(fr, li, true)
}

func (m *Model) ensureBlockRanges(fr *jlm.Frames, li int, experts bool) error {
	return fr.EnsureRanges(li, m.blockPlan(li, experts))
}

// blockPlan is the ranges of block li's page that ensureBlockRanges reads.
func (m *Model) blockPlan(li int, experts bool) []jlm.Range {
	c := m.Container
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
func (m *Model) EnsureBlockSelected(fr *jlm.Frames, li int, sel []uint32, base bool) error {
	// A bank in expert pages is warmed page by page: the reads land in frames
	// that stay cached across tokens, which is the point of splitting them.
	if m.Container != nil && li < len(m.Layers) && m.Layers[li].ExpBank().E != nil &&
		m.Container.ExpertPaged(m.Layers[li].ExpBank().E) {
		// The cold pages are read together and a page a guess is already
		// reading is joined (jlm.File.EnsureExperts); a warm selection
		// allocates nothing.
		if err := fr.EnsureExperts(m.Layers[li].ExpBank().E, sel); err != nil {
			return fmt.Errorf("model: block %d: %w", li, err)
		}
		if base {
			return m.ensureBlock(fr, li)
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
	return fr.EnsureRanges(li, rs)
}

// speculateExperts queues the experts sel of block li's bank for the
// container's background readers, replacing the guesses not yet started.
func (m *Model) SpeculateExperts(fr *jlm.Frames, li int, sel []uint32) {
	if m.Container == nil || li >= len(m.Layers) {
		return
	}
	if bank := m.Layers[li].ExpBank().E; bank != nil && m.Container.ExpertPaged(bank) {
		fr.Speculate(bank, sel)
	}
}

// selectedRanges is ensureBlockSelected's plan over the block page: the base
// when asked for, and a legacy per-expert tensor when selected. Since v26 a
// bank is never in it; its sheets are expert pages.
func (m *Model) selectedRanges(li int, sel []uint32, base bool) ([]jlm.Range, error) {
	c := m.Container
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
// caller's h.release, which it must call whatever this returns. The views
// bound are l's, the caller's copy of the block.
func (m *Model) EnsureExperts(fr *jlm.Frames, li int, sel []int32, h *ExpertHold, l *Layer) error {
	c, cf, cfg := m.Container, fr, m.Cfg
	if c == nil || !cfg.MoEAt(li) || len(sel) == 0 {
		return nil
	}
	if l.LegacyExperts {
		return m.ensureLegacyExperts(fr, li, sel, h, l)
	}
	bank := l.ExpBank().E
	if bank == nil || !c.ExpertPaged(bank) {
		return nil
	}
	// The counters bracket the reads; see ExpertIO.
	t0 := time.Now()
	r0, b0, w0 := c.Reads(), cf.ChunksIn(), cf.ReadWait()
	defer func() {
		atomic.AddInt64(&m.expWaitNs, int64(time.Since(t0)))
		atomic.AddInt64(&m.expIONs, int64(cf.ReadWait()-w0))
		atomic.AddInt64(&m.expReads, c.Reads()-r0)
		atomic.AddInt64(&m.expBytes, (cf.ChunksIn()-b0)*int64(cf.ChunkBytes()))
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
		if lb, ok := cf.HoldFilled(li); ok {
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
		if lp, ok := cf.HoldFilled(p); ok {
			h.held[i] = lp
			continue
		}
		off, size := c.PageBounds(p)
		h.wg.Add(1)
		go func() {
			defer h.wg.Done()
			h.held[i], h.errs[i] = cf.Hold(p, []jlm.Range{{Off: off, N: size}})
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
	for _, x := range sel {
		v := &l.Experts[x]
		bindSheet(cf, &l.Gate, cfg.NFFNExp, int(x), &v.Gate)
		bindSheet(cf, &l.Up, cfg.NFFNExp, int(x), &v.Up)
		bindSheet(cf, &l.Down, cfg.ExpWidth(), int(x), &v.Down)
	}
	return nil
}

// expertHold is the pages one layer's routed experts are held under until the
// caller's FFN is done with them. An instance keeps one and reuses its slices, so
// holding a token's selection allocates nothing once they have grown.
type ExpertHold struct {
	leases []jlm.Lease
	pages  []int
	held   []jlm.Lease
	errs   []error
	// wg is here and not on the stack because the reads capture it.
	wg sync.WaitGroup
}

// release drops every pin taken since the last release.
func (h *ExpertHold) Release() {
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
func (m *Model) ensureLegacyExperts(fr *jlm.Frames, li int, sel []int32, h *ExpertHold, l *Layer) error {
	cf := fr
	u := make([]uint32, len(sel))
	for i, x := range sel {
		u[i] = uint32(x)
	}
	rs, err := m.selectedRanges(li, u, false)
	if err != nil {
		return err
	}
	lb, err := cf.Hold(li, rs)
	if err != nil {
		return err
	}
	h.leases = append(h.leases, lb)
	// Re-bound under the page just held: bindOne resolves a whole tensor, and
	// the frame may not be the one pageIn bound it to.
	for _, x := range sel {
		bindOne(cf, &l.Experts[x].Gate)
		bindOne(cf, &l.Experts[x].Up)
		bindOne(cf, &l.Experts[x].Down)
	}
	return nil
}

// expBank is the bank a mixture block's expert pages are addressed through:
// gate where the experts have one, up where they are ungated. Every bank of a
// block is split the same way, so any of them names the same pages.
func (l *Layer) ExpBank() *Tensor {
	if l.Gate.E != nil {
		return &l.Gate
	}
	return &l.Up
}

// bindSheet points expert x's view v of bank b at the sheet in its expert
// page. The page must be held: the slices are a frame the pager reuses.
func bindSheet(c *jlm.Frames, b *Tensor, nrow, x int, v *Tensor) {
	if b.E == nil {
		return
	}
	qs, d, sc := c.Sheet(b.E, x)
	if v.Rows == nrow && SameSpan(v.Data, qs) && (b.Packed == nil || (v.Packed != nil &&
		SameSpan(v.Packed.QS, qs) && SameSpan(v.Packed.D, d) && SameSpan(v.Packed.SC, sc))) {
		return // already bound to this page: a reader may be using it
	}
	v.E, v.Typ, v.Rows, v.K, v.Data = b.E, b.Typ, nrow, b.K, qs
	if b.Packed == nil {
		return
	}
	if v.Packed == nil {
		v.Packed = &nn.Packed{}
	}
	*v.Packed = nn.Packed{QS: qs, D: d, SC: sc, Row: 0, Stride: nrow}
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
func (m *Model) PagerReadWait(fr *jlm.Frames) time.Duration {
	if m == nil || m.Container == nil {
		return 0
	}
	return fr.ReadWait()
}

// releaseLayer drops a block's host residency: the container's page and every
// slice of it this layer holds. DropPage alone frees nothing, since the
// layer's spans point into the page and keep its backing array alive.
//
// It is the inverse of bindPacked; pageIn rebinds on the next fault.
func ReleaseLayer(c *jlm.Frames, li int, l *Layer, keepExperts bool) {
	for _, m := range l.Mats {
		// A matrix no entry names is a view of the block's own vector, not
		// of its page: nothing would bind it again.
		if m.T.E == nil {
			continue
		}
		m.T.Data = nil
		if m.T.Packed != nil {
			m.T.Packed.QS, m.T.Packed.D, m.T.Packed.SC = nil, nil, nil
		}
	}
	if w := l.Mtp; w != nil {
		for _, t := range []*Tensor{&w.EhProj, &w.Head, &w.Embd} {
			if t.E == nil || t.E.Block == jlm.DenseBlock {
				continue // the trunk's, in the dense region
			}
			t.Data = nil
			if t.Packed != nil {
				t.Packed.QS, t.Packed.D, t.Packed.SC = nil, nil, nil
			}
		}
	}
	// MLA's per-head views are slices of wkb and wvb, as the expert views
	// below are of the banks.
	for i := range l.kbHead {
		for _, t := range []*Tensor{&l.kbHead[i], &l.vbHead[i]} {
			t.Data = nil
			if t.Packed != nil {
				t.Packed.QS, t.Packed.D, t.Packed.SC = nil, nil, nil
			}
		}
	}
	for i := range l.Experts {
		for _, t := range []*Tensor{&l.Experts[i].Gate, &l.Experts[i].Up, &l.Experts[i].Down} {
			t.Data = nil
			if t.Packed != nil {
				t.Packed.QS, t.Packed.D, t.Packed.SC = nil, nil, nil
			}
		}
	}
	// The views are gone, so the next bind cuts them again even if the banks
	// land where they were.
	l.headsFrom, l.banksFrom = [2]bankSpans{}, [3]bankSpans{}
	// Another instance runs the block from host memory: its page and its
	// experts' stay for it.
	if c.Kept(li) {
		return
	}
	c.DropPage(li)
	// A bank in expert pages is not in the block's page: since v26 each expert
	// is a page of its own, and on a mixture those are nearly all of the
	// weights (Qwen3-30B-A3B: 18.0 of 18.5 GiB). The device holds the whole
	// bank, so they go too, and their frames serve the next block's upload --
	// unless the block's experts still run from the host's pages (keepExperts:
	// a streamed or hybrid block), which would read them all again.
	if e := l.ExpBank().E; e != nil && c.File().ExpertPaged(e) && !keepExperts {
		for x := 0; x < int(e.Dims[2]); x++ {
			c.DropPage(c.File().ExpertPage(e, x))
		}
	}
}

// bindPacked binds every weight of one block.
//
// An expert of a bank is a view, and binding it with bindOne would overwrite it
// with the whole bank, making every expert expert 0 on an unpacked bank. The
// views are rebuilt from the freshly bound bank instead.
func bindPacked(c *jlm.Frames, cfg *Config, l *Layer) error {
	for _, m := range l.Mats {
		bindOne(c, m.T)
	}
	// A prediction block's eh_proj, head and embedding. The head and the
	// embedding are copies of the trunk's tensors unless the block carries its
	// own, and a copy taken at build() predates binding, so it is bound here
	// too; a dense span never moves, so after the first time this writes
	// nothing.
	if w := l.Mtp; w != nil {
		for _, t := range []*Tensor{&w.EhProj, &w.Head, &w.Embd} {
			bindOne(c, t)
		}
	}
	// The per-head views are re-cut, as l.experts is below: they are slices of
	// wkb and wvb, so a rebind that moves those would leave them stale.
	if from := [2]bankSpans{spansOf(l.Wkb), spansOf(l.Wvb)}; len(l.kbHead) != 0 && from != l.headsFrom {
		nope := cfg.HeadDim - cfg.NRot
		for h := range l.kbHead {
			kb, err := bankExpert(l.Wkb, jlm.RoleAttnKB, cfg.NHead, cfg.KVLoraRank, nope, h)
			if err != nil {
				return err
			}
			vb, err := bankExpert(l.Wvb, jlm.RoleAttnVB, cfg.NHead, cfg.HeadDimV, cfg.KVLoraRank, h)
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
	if l.LegacyExperts {
		// Separate tensors per expert: each has its own name, so bindOne is
		// resolving a whole weight rather than clobbering a view.
		for i := range l.Experts {
			bindOne(c, &l.Experts[i].Gate)
			bindOne(c, &l.Experts[i].Up)
			bindOne(c, &l.Experts[i].Down)
		}
		return nil
	}
	// A bank in expert pages has no bank to cut views from, and its views are
	// left alone: ensureExperts binds the ones a token selects. Emptying them
	// here would race another session's MoE reading them.
	if l.ExpBank().E != nil && c.File().ExpertPaged(l.ExpBank().E) {
		return nil
	}
	from := [3]bankSpans{spansOf(l.Gate), spansOf(l.Up), spansOf(l.Down)}
	if from == l.banksFrom {
		return nil
	}
	for e := range l.Experts {
		x := &l.Experts[e]
		var err error
		if !l.UngatedExp {
			if x.Gate, err = bankExpert(l.Gate, jlm.RoleExpGateBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
				return err
			}
		}
		if x.Up, err = bankExpert(l.Up, jlm.RoleExpUpBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
			return err
		}
		if x.Down, err = bankExpert(l.Down, jlm.RoleExpDownBank, cfg.NExpert, cfg.ExpWidth(), cfg.NFFNExp, e); err != nil {
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

func spansOf(t Tensor) bankSpans {
	s := bankSpans{spanOf(t.Data)}
	if t.Packed != nil {
		s[1], s[2], s[3] = spanOf(t.Packed.QS), spanOf(t.Packed.D), spanOf(t.Packed.SC)
	}
	return s
}

// ropeScale is the rotary's per-pair divisors under cfg's scaling (YaRN,
// linear), times the factors the file ships where it ships them: each
// pair's theta divided by its factor, then scaled -- a ramp over the pair's
// index, so the divisors multiply (ggml_rope_ext's freq_factors under its
// YaRN). Once at open, a constant the rotary's table is formed with.
func ropeScale(cfg *Config, factors []float32) ([]float32, error) {
	var scale []float32
	if cfg.YarnFactor > 0 {
		scale = nn.YarnFreqs(cfg.NRot, cfg.RopeBase, cfg.YarnFactor, cfg.YarnOrigCtx,
			cfg.YarnBetaFast, cfg.YarnBetaSlow, cfg.YarnExact)
	} else {
		scale = make([]float32, cfg.NRot/2)
		for i := range scale {
			scale[i] = float32(cfg.RopeLinear)
		}
	}
	if factors == nil {
		return scale, nil
	}
	if len(factors) != len(scale) {
		return nil, fmt.Errorf("model: %d rotary factors for %d pairs", len(factors), len(scale))
	}
	for i := range scale {
		scale[i] *= factors[i]
	}
	return scale, nil
}

func build(c *jlm.File, options ...tok.Option) (*Model, error) {
	cfg, err := configFrom(c.Config(), func(r jlm.Role) bool {
		for bi := range c.Config().NLayer {
			if c.Has(r, int32(bi), -1) {
				return true
			}
		}
		return false
	})
	if err != nil {
		return nil, err
	}
	if cfg.expWidth, err = expertWidth(c, cfg); err != nil {
		return nil, err
	}
	if v := c.Vision(); v != nil && v.Tower != nil && len(v.Tower.Taps) > 0 {
		cfg.DeepTaps = len(v.Tower.Taps)
		if cfg.DeepTaps > cfg.NLayer || cfg.PLEDim != 0 {
			return nil, fmt.Errorf("model: %d tap rows added after the first of %d blocks, on a model of per-layer embeddings %v: "+
				"both would be the blocks' side input", cfg.DeepTaps, cfg.NLayer, cfg.PLEDim != 0)
		}
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
	if cfg.Scan() {
		for i, k := range c.Config().LayerKinds {
			if e, ok := c.Find(jlm.RoleSSMXDt, int32(i), -1); ok && k == jlm.LayerScan {
				cfg.DtRank = int(e.Dims[1])
				break
			}
		}
		if cfg.DtRank == 0 {
			return nil, fmt.Errorf("model: a Mamba-1 model with no x_proj")
		}
	}
	m := &Model{Cfg: cfg}
	// An unsupported tokenizer is not a reason to refuse the weights: Vocab
	// nil is a model that runs token ids but cannot turn text into them.
	// TokErr says why.
	if m.Vocab, m.TokErr = tok.New(c.Vocab(), options...); m.TokErr != nil {
		m.Vocab = nil
	}

	// Tensors are resolved by role, block and expert index, not by name.
	get := func(role jlm.Role, block, index int32) (Tensor, error) {
		e, ok := c.Find(role, block, index)
		if !ok {
			return Tensor{}, fmt.Errorf("model: missing %v (block %d, index %d)", role, block, index)
		}
		tn, err := entryTensor(e)
		if err != nil {
			return Tensor{}, err
		}
		return tn, nil
	}
	// cur is the block the loop below is loading; getTo binds a matrix of it
	// into dst and records dst under role (layer.mats).
	var cur *Layer
	getTo := func(dst *Tensor, role jlm.Role, block, index int32) error {
		t, err := get(role, block, index)
		if err != nil {
			return err
		}
		*dst = t
		if cur != nil {
			cur.Mats = append(cur.Mats, roleTensor{role, dst})
		}
		return nil
	}
	vec := func(role jlm.Role, block int32) ([]float32, error) {
		if block >= 0 && int(block) < len(m.Layers) {
			if v := nn.VectorOf(m.Layers[block].Vecs, role); v != nil {
				return v, nil
			}
		}
		if block == jlm.DenseBlock {
			if v := nn.VectorOf(m.Vecs, role); v != nil {
				return v, nil
			}
		}
		e, ok := c.Find(role, block, -1)
		if !ok {
			return nil, fmt.Errorf("model: missing %v (block %d)", role, block)
		}
		return entryVector(c, e)
	}
	// optVec is vec for a tensor that may legitimately be absent; nil is the
	// answer, not an error.
	optVec := func(role jlm.Role, block int32) ([]float32, error) {
		if !c.Has(role, block, -1) {
			return nil, nil
		}
		return vec(role, block)
	}
	var views []Tensor
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Index != -1 || !e.Role.Expanded() || e.Role.Vision() || e.Block != jlm.DenseBlock {
			continue
		}
		v, err := vec(e.Role, e.Block)
		if err != nil {
			return nil, err
		}
		m.Vecs = append(m.Vecs, nn.RoleVector{Role: e.Role, V: v})
		if e.NDim == 2 {
			t := F32Tensor(v, int(e.Dims[1]), int(e.Dims[0]))
			t.E = e
			views = append(views, t)
		}
	}
	for i := range views {
		m.VecMats = append(m.VecMats, roleTensor{views[i].E.Role, &views[i]})
	}

	if m.Embd, err = get(jlm.RoleTokenEmbd, jlm.DenseBlock, -1); err != nil {
		return nil, err
	}
	// A model without an output norm stores none (an encoder; a model whose
	// streams' mix stands in for it), and the head's graph states one only
	// where it is stored.
	if m.OutNorm, err = optVec(jlm.RoleOutputNorm, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if p := cfg.PLEDim; p != 0 {
		if m.PleTok, err = get(jlm.RolePLETokEmbd, jlm.DenseBlock, -1); err != nil {
			return nil, err
		}
		if m.PleProj, err = get(jlm.RolePLEModelProj, jlm.DenseBlock, -1); err != nil {
			return nil, err
		}
		if m.PleNorm, err = vec(jlm.RolePLEProjNorm, jlm.DenseBlock); err != nil {
			return nil, err
		}
		// The table may stop short of the vocabulary: Gemma 3n's ids past
		// it (its vision and audio tokens) read row 0 (pleInputs).
		w := cfg.NLayer * p
		if m.PleTok.K != w || m.PleTok.Rows == 0 || m.PleTok.Rows > cfg.NVocab || m.PleProj.K != cfg.NEmbd ||
			m.PleProj.Rows != w || len(m.PleNorm) != p {
			return nil, fmt.Errorf("model: per-layer embeddings: a table %dx%d, a projection %dx%d and "+
				"a norm of %d, want %d wide over at most %d tokens from %d, and a norm of %d",
				m.PleTok.Rows, m.PleTok.K, m.PleProj.K, m.PleProj.Rows, len(m.PleNorm), w, cfg.NVocab,
				cfg.NEmbd, p)
		}
	}
	if err := loadNGramTable(m, cfg, c); err != nil {
		return nil, err
	}
	if m.OutNormB, err = optVec(jlm.RoleOutputNormBias, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if c.Has(jlm.RolePosEmbd, jlm.DenseBlock, -1) {
		if m.PosEmbd, err = get(jlm.RolePosEmbd, jlm.DenseBlock, -1); err != nil {
			return nil, err
		}
		if m.PosEmbd.K != cfg.NEmbd {
			return nil, fmt.Errorf("model: the position table is %d wide, want %d", m.PosEmbd.K, cfg.NEmbd)
		}
		// A learned table ends; the context is clamped to it so NewInstance
		// refuses a later position rather than reading past the table.
		if m.PosEmbd.Rows < cfg.NCtx {
			cfg.NCtx = m.PosEmbd.Rows
		}
	}
	if m.OutB, err = optVec(jlm.RoleOutputBias, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if m.OutB != nil && len(m.OutB) != cfg.NVocab {
		return nil, fmt.Errorf("model: the head's bias has %d values, want %d", len(m.OutB), cfg.NVocab)
	}
	// An RMSNorm has no bias, so a norm bias without FlagLayerNorm is a
	// container that disagrees with itself.
	if m.OutNormB != nil && !cfg.LayerNorm {
		return nil, fmt.Errorf("model: %s: a norm bias on an RMSNorm model", cfg.Arch)
	}
	cfg.RopeInterleaved = c.Config().Arch.InterleavedMRope()
	m.Rope = nn.Rope{NRot: cfg.NRot, Base: cfg.RopeBase, Neox: cfg.RopeNeox,
		Scale: cfg.AttnFactor, Runs: nn.MRopeRuns(cfg.RopeSections[:])}
	if cfg.RopeInterleaved && len(m.Rope.Runs) > 0 {
		m.Rope.Runs = nn.IMRopeRuns(cfg.RopeSections[:], cfg.NRot/2)
	}
	if cfg.RopeXD = c.Config().Arch.XDRope(); cfg.RopeXD {
		// transformers' XD-RoPE turns the whole head; a partial one has no
		// reference to be held to.
		if !cfg.RopeNeox || len(m.Rope.Runs) == 0 || cfg.NRot != cfg.HeadDim {
			return nil, fmt.Errorf("model: %s: XD-RoPE needs NEOX pairs, rope sections and the whole head "+
				"turned (NRot %d of %d)", cfg.Arch, cfg.NRot, cfg.HeadDim)
		}
		m.Rope.Runs, _ = nn.XDRopeRuns(cfg.RopeSections[:], cfg.NRot/2)
	}
	if cfg.QKL2Norm || cfg.VNorm {
		m.onesHD = make([]float32, cfg.MaxHeadDim())
		for i := range m.onesHD {
			m.onesHD[i] = 1
		}
	}
	// A second rotary for architectures trained with two bases (gemma3's
	// sliding layers), built only when RopeBaseSWA is set.
	if cfg.RopeBaseSWA != 0 || cfg.LocalNRot() != cfg.NRot {
		r := m.Rope
		if cfg.RopeBaseSWA != 0 {
			r.Base = cfg.RopeBaseSWA
		}
		// The sliding layers turn their own width where their blocks state one.
		r.NRot = cfg.LocalNRot()
		if cfg.RopeScaleSWA != 0 {
			r.Scale = cfg.RopeScaleSWA
		}
		if len(r.Runs) > 0 {
			return nil, fmt.Errorf("model: %s: rotary sections on the sliding layers' own rotary, which the converter refuses",
				cfg.Arch)
		}
		m.RopeSWA = &r
	}
	// Per-pair rotary factors are a tensor: llama 3.1+ ships rope_freqs and
	// phi3 rope_factors_short/long, dividing theta per pair. phi3 picks by
	// sequence length; short is right below its original context (4096), and
	// the long switch is not implemented.
	for _, role := range []jlm.Role{jlm.RoleRopeFreqs, jlm.RoleRopeFactorsShort} {
		if c.Has(role, jlm.DenseBlock, -1) {
			if m.Rope.Freqs, err = vec(role, jlm.DenseBlock); err != nil {
				return nil, err
			}
			break
		}
	}
	// Rotary scaling (YaRN, linear) is a per-pair divisor too, so it rides the
	// same field. It is set after the local rotary was copied above, because
	// llama.cpp leaves gemma3's sliding layers unscaled.
	if cfg.YarnFactor > 0 || cfg.RopeLinear > 0 {
		if m.Rope.Freqs, err = ropeScale(cfg, m.Rope.Freqs); err != nil {
			return nil, err
		}
	}
	if cfg.TiedEmbd {
		m.Output = m.Embd
	} else if m.Output, err = get(jlm.RoleOutput, jlm.DenseBlock, -1); err != nil {
		return nil, err
	}
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Block != jlm.DenseBlock || e.Index != -1 || e.Role.Expanded() || e.Role.Vision() || e.NDim < 2 ||
			jlm.RowTable(e) || denseNamed(e.Role) {
			continue
		}
		t, err := get(e.Role, jlm.DenseBlock, -1)
		if err != nil {
			return nil, err
		}
		m.DenseMats = append(m.DenseMats, roleTensor{e.Role, &t})
	}

	// Room for the tower's blocks after the text's (bindTower): a layer's
	// matrices are recorded by pointers into it (layer.mats), so the slice
	// is never reallocated once a block is bound.
	nvis := 0
	if v := c.Vision(); v != nil && v.Tower != nil {
		nvis = int(v.NLayer)
	}
	m.Layers = make([]Layer, cfg.NLayer+cfg.NMTP, cfg.NLayer+cfg.NMTP+nvis)
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Index != -1 || !e.Role.Expanded() || e.Role.Vision() || e.Block < 0 || int(e.Block) >= len(m.Layers) {
			continue
		}
		v, err := vec(e.Role, e.Block)
		if err != nil {
			return nil, err
		}
		l := &m.Layers[e.Block]
		l.Vecs = append(l.Vecs, nn.RoleVector{Role: e.Role, V: v})
	}
	// bind names one tensor a layer must have. It is a named type rather than
	// an anonymous struct so a hybrid can substitute the attention four for the
	// recurrent four without restating the literal.
	type bind = struct {
		role jlm.Role
		dst  *Tensor
	}
	for i := range m.Layers {
		// Nothing is faulted in here: the norms live in the dense region
		// (jlm.alwaysResident), so build() touches no page and placement can
		// happen before any block is read.
		l := &m.Layers[i]
		cur = l
		bi := int32(i)
		want := []bind{
			{jlm.RoleAttnQ, &l.Wq}, {jlm.RoleAttnK, &l.Wk},
			{jlm.RoleAttnV, &l.Wv}, {jlm.RoleAttnOut, &l.Wo},
			{jlm.RoleFFNGate, &l.Gate}, {jlm.RoleFFNUp, &l.Up},
			{jlm.RoleFFNDown, &l.Down},
		}
		// A linear layer (per the layer map) replaces the four attention
		// bindings with its own; MLA (a whole-model property, from KVLoraRank)
		// replaces them with five. The two are mutually exclusive per block:
		// Kimi-Linear has both kinds, and each branch trims the original
		// leading four, so applying both to one block would compose wrongly.
		linear := cfg.LayerKind(i).Recurrent()
		mla := cfg.KVLoraRank != 0 && !linear
		if mla {
			q := bind{jlm.RoleAttnQ, &l.Wq}
			if cfg.QLoraRank != 0 {
				q = bind{jlm.RoleAttnQB, &l.Wqb}
			}
			// The FFN triple must stay last: the mixture branch below trims it
			// with want[:len(want)-3].
			head := []bind{
				q,
				{jlm.RoleAttnKVA, &l.Wkva},
				{jlm.RoleAttnKB, &l.Wkb},
				{jlm.RoleAttnVB, &l.Wvb},
				{jlm.RoleAttnOut, &l.Wo},
			}
			if cfg.QLoraRank != 0 {
				head = append(head, bind{jlm.RoleAttnQA, &l.Wqa})
			}
			want = append(head, want[4:]...)
		}
		if linear {
			head := []bind{
				{jlm.RoleAttnQKV, &l.Wq},
				{jlm.RoleSSMBA, &l.SsmBA},
				{jlm.RoleSSMOut, &l.SsmOut},
			}
			// A block that attends as well keeps the attention four, and its
			// mixer's projection goes beside them.
			attends := cfg.LayerKind(i).Attends()
			if attends {
				head[0].dst = &l.SsmIn
			}
			switch {
			case cfg.Scan():
				// x convolves (wq), z gates the output (ssmGate); dt, B and C
				// come off the convolved x, and there is no ssm_ba.
				head = []bind{{jlm.RoleAttnQKV, &l.Wq}, {jlm.RoleAttnGate, &l.SsmGate},
					{jlm.RoleSSMOut, &l.SsmOut}, {jlm.RoleSSMXDt, &l.SsmXDt},
					{jlm.RoleSSMXB, &l.SsmXB}, {jlm.RoleSSMXC, &l.SsmXC},
					{jlm.RoleSSMDtProj, &l.SsmDtProj}}
			case cfg.ShortConv():
				// x convolves (wq, the mixed projection's place), B and C gate
				// it before and after (the two gate slots).
				head = []bind{{jlm.RoleSCX, &l.Wq}, {jlm.RoleSCB, &l.SsmGate},
					{jlm.RoleSCC, &l.SsmBA}, {jlm.RoleSSMOut, &l.SsmOut}}
			case cfg.ChanDecay() && !c.Has(jlm.RoleSSMGA, bi, -1) && c.Has(jlm.RoleAttnGate, bi, -1):
				// Kimi-K3's full-rank output gate: one matrix where Kimi-Linear
				// factors it, in the z projection's place.
				head = append(head,
					bind{jlm.RoleSSMFA, &l.SsmFA}, bind{jlm.RoleSSMFB, &l.SsmFB},
					bind{jlm.RoleAttnGate, &l.SsmGate})
			case cfg.ChanDecay():
				head = append(head,
					bind{jlm.RoleSSMFA, &l.SsmFA}, bind{jlm.RoleSSMFB, &l.SsmFB},
					bind{jlm.RoleSSMGA, &l.SsmGA}, bind{jlm.RoleSSMGB, &l.SsmGB})
			default:
				head = append(head, bind{jlm.RoleAttnGate, &l.SsmGate})
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
			l.NoFFN = true
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
				l.LegacyExperts = true
			}
			specs := []struct {
				role jlm.Role
				dst  *Tensor
			}{
				{jlm.RoleRouter, &l.Router},
				{jlm.RoleExpGateBank, &l.Gate},
				{jlm.RoleExpUpBank, &l.Up},
				{jlm.RoleExpDownBank, &l.Down},
			}
			if l.LegacyExperts {
				specs = specs[:1] // the router only; experts are read below
			}
			// Experts with no gate bank are ungated: down(act(up(h))), the
			// activation an ungated kind (Nemotron 3's squared ReLU).
			if !l.LegacyExperts && !c.Has(jlm.RoleExpGateBank, bi, -1) && c.Has(jlm.RoleExpUpBank, bi, -1) {
				l.UngatedExp = true
				specs = slices.Delete(specs, 1, 2)
			}
			for _, spec := range specs {
				if err = getTo(spec.dst, spec.role, bi, -1); err != nil {
					return nil, err
				}
			}
			// A model has a shared expert if the container carries its
			// tensors.
			// An ungated one (Nemotron 3) has up and down and no gate.
			if c.Has(jlm.RoleShExpGate, bi, -1) || c.Has(jlm.RoleShExpUp, bi, -1) {
				for _, spec := range []struct {
					role jlm.Role
					dst  *Tensor
				}{
					{jlm.RoleShExpGate, &l.ShGate},
					{jlm.RoleShExpUp, &l.ShUp},
					{jlm.RoleShExpDown, &l.ShDown},
				} {
					if spec.role == jlm.RoleShExpGate && !c.Has(jlm.RoleShExpGate, bi, -1) {
						continue
					}
					if err = getTo(spec.dst, spec.role, bi, -1); err != nil {
						return nil, err
					}
				}
				// Optional: DeepSeek's shared expert has no gate. See
				// sharedExpert for why an absent one is weight 1 and not
				// sigma(0).
				if _, err = optVec(jlm.RoleShRouter, bi); err != nil {
					return nil, err
				}
				if l.Vector(jlm.RoleShRouter) != nil && len(l.Vector(jlm.RoleShRouter)) != cfg.NEmbd {
					return nil, fmt.Errorf("model: block %d shared-expert gate is %d wide, want %d",
						bi, len(l.Vector(jlm.RoleShRouter)), cfg.NEmbd)
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
				l.VFromK = true
				continue
			}
			if err = getTo(spec.dst, spec.role, bi, -1); err != nil {
				return nil, err
			}
		}
		// A post-norm block (OLMo 2, EXAONE 4) has no pre-norms at all: each
		// branch reads the residual as it is and normalises its output instead.
		// A block stores the pre-norms it has: none on a post-norm block (OLMo
		// 2's, an encoder's norms after each residual add), no FFN norm on a
		// parallel block sharing one; the front end states the norms stored.
		if l.AttnNorm, err = optVec(jlm.RoleAttnNorm, bi); err != nil {
			return nil, err
		}
		if l.FfnNorm, err = optVec(jlm.RoleFFNNorm, bi); err != nil {
			return nil, err
		}
		for _, b := range []struct {
			role jlm.Role
			dst  *[]float32
		}{
			{jlm.RoleAttnNormBias, &l.AttnNormB}, {jlm.RoleFFNNormBias, &l.FfnNormB},
		} {
			if *b.dst, err = optVec(b.role, bi); err != nil {
				return nil, err
			}
		}
		if (l.AttnNormB != nil || l.FfnNormB != nil) && !cfg.LayerNorm {
			return nil, fmt.Errorf("model: %s: block %d has a norm bias on an RMSNorm model", cfg.Arch, bi)
		}
		if l.FfnNormB != nil && l.FfnNorm == nil {
			return nil, fmt.Errorf("model: block %d has an FFN norm bias and no FFN norm", bi)
		}
		// FFN biases belong only to a dense ungated FFN; a mixture's are the
		// experts' own (RoleExpUpBias). Refused rather than guessed.
		if (l.Vector(jlm.RoleFFNUpBias) != nil || l.Vector(jlm.RoleFFNDownBias) != nil) && (l.Gate.E != nil || cfg.MoEAt(i)) {
			return nil, fmt.Errorf("model: block %d carries FFN biases on a gated or mixture FFN", bi)
		}
		// gemma2/gemma3's post-norms, when the file carries them.
		if _, err = optVec(jlm.RolePostAttnNorm, bi); err != nil {
			return nil, err
		}
		// BitNet's two inner norms: both on every block of a BitNet, and on
		// no other model, each as wide as what it normalises.
		if _, err = optVec(jlm.RoleAttnSubNorm, bi); err != nil {
			return nil, err
		}
		if _, err = optVec(jlm.RoleFFNSubNorm, bi); err != nil {
			return nil, err
		}
		if (l.Vector(jlm.RoleAttnSubNorm) != nil || l.Vector(jlm.RoleFFNSubNorm) != nil) != cfg.TokenAct ||
			cfg.TokenAct && (len(l.Vector(jlm.RoleAttnSubNorm)) != cfg.QDimAt(i) || len(l.Vector(jlm.RoleFFNSubNorm)) != cfg.NFFNAt(i) ||
				l.Gate.E == nil || cfg.MoEAt(i) || cfg.Parallel) {
			return nil, fmt.Errorf("model: block %d: inner norms %d and %d wide on %s (a gated dense FFN of %d, "+
				"attention %d wide, per-token activations %v)", bi, len(l.Vector(jlm.RoleAttnSubNorm)), len(l.Vector(jlm.RoleFFNSubNorm)),
				cfg.Arch, cfg.NFFNAt(i), cfg.QDimAt(i), cfg.TokenAct)
		}
		// Gemma 4's per-block output scalar, when the file carries one.
		if sc, err := optVec(jlm.RoleLayerOutScale, bi); err != nil {
			return nil, err
		} else if sc != nil {
			if len(sc) != 1 {
				return nil, fmt.Errorf("model: block %d's output scale has %d values, want 1", bi, len(sc))
			}
			l.OutScale, l.OutScaleRow = sc[0], sc
		}
		if _, err = optVec(jlm.RolePostFFNNorm, bi); err != nil {
			return nil, err
		}
		if f := cfg.NFFNAt(i); !cfg.MoEAt(i) && l.Down.K != 0 && (l.Down.K != f || l.Up.Rows != f) {
			return nil, fmt.Errorf("model: block %d's FFN is %d wide (up %d), want %d", bi, l.Down.K,
				l.Up.Rows, f)
		}
		// A mixture block whose experts read a pre-norm of their own runs a
		// dense MLP beside them, each branch with its norms.
		if cfg.MoEAt(i) && l.Vector(jlm.RoleFFNNorm2) != nil {
			for _, v := range []struct {
				role jlm.Role
				want int
			}{
				{jlm.RoleFFNNorm2, cfg.NEmbd},
				{jlm.RolePostFFNNorm1, cfg.NEmbd},
				{jlm.RolePostFFNNorm2, cfg.NEmbd},
				{jlm.RoleRouterNorm, cfg.NEmbd},
				{jlm.RoleExpScale, cfg.NExpert},
			} {
				if got := l.Vector(v.role); len(got) != v.want {
					return nil, fmt.Errorf("model: block %d %s has %d values, want %d",
						bi, v.role, len(got), v.want)
				}
			}
			if l.Vector(jlm.RolePostFFNNorm) == nil || l.FfnNorm == nil {
				return nil, fmt.Errorf("model: block %d is a dense-and-mixture block with no "+
					"ffn_norm or post_ffw_norm", bi)
			}
		}
		// A bias of the wrong length would still run, landing on a neighbour's
		// rows, so the lengths are checked here.
		for _, b := range []struct {
			name string
			v    []float32
			want int
		}{
			{"attn_sinks", l.Vector(jlm.RoleAttnSinks), cfg.NHead},
			{"router bias", l.Vector(jlm.RoleRouterBias), cfg.NExpert},
			{"expert selection bias", l.Vector(jlm.RoleExpProbsB), cfg.NExpert},
			{"expert gate bias", l.Vector(jlm.RoleExpGateBias), cfg.NExpert * cfg.NFFNExp},
			{"expert up bias", l.Vector(jlm.RoleExpUpBias), cfg.NExpert * cfg.NFFNExp},
			{"expert down bias", l.Vector(jlm.RoleExpDownBias), cfg.NExpert * cfg.NEmbd},
			{"attn_norm bias", l.AttnNormB, cfg.NEmbd},
			{"ffn_norm bias", l.FfnNormB, cfg.NEmbd},
			{"ffn_up bias", l.Vector(jlm.RoleFFNUpBias), cfg.NFFN},
			{"ffn_down bias", l.Vector(jlm.RoleFFNDownBias), cfg.NEmbd},
		} {
			if b.v != nil && len(b.v) != b.want {
				return nil, fmt.Errorf("model: block %d %s has %d values, want %d",
					bi, b.name, len(b.v), b.want)
			}
		}
		// The recurrent vectors are small and dense. ssm_conv1d is a matrix by
		// shape and a per-channel window by use, so it is expanded here.
		if linear {
			var discard []float32
			for _, v := range []struct {
				role jlm.Role
				dst  *[]float32
			}{
				{jlm.RoleSSMConv1d, &l.SsmConv1d}, {jlm.RoleSSMA, &discard},
				{jlm.RoleSSMDtBias, &discard}, {jlm.RoleSSMNorm, &discard},
			} {
				// A short convolution has its window and nothing else.
				if cfg.ShortConv() && v.role != jlm.RoleSSMConv1d {
					continue
				}
				// A Mamba-2 block may have no gated norm at all (Falcon-H1's
				// mamba_rms_norm false): y*silu(z) goes straight to ssm_out.
				if v.role == jlm.RoleSSMNorm && cfg.Scan() {
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
			g := cfg.Delta()
			// Transposed once at load: the file stores taps contiguous per
			// channel, and plane-major makes the convolution elementwise.
			if n := len(l.SsmConv1d); n == g.Conv*g.Chans {
				l.SsmConvTaps = F32Tensor(l.SsmConv1d, g.Chans, g.Conv)
				wt := make([]float32, n)
				for c := 0; c < g.Chans; c++ {
					for t := 0; t < g.Conv; t++ {
						wt[t*g.Chans+c] = l.SsmConv1d[c*g.Conv+t]
					}
				}
				l.SsmConv1d = wt
			}
			if len(l.SsmConv1d) != g.Conv*g.Chans {
				return nil, fmt.Errorf("model: block %d convolution is %d wide, want %d (%d positions x %d channels)",
					bi, len(l.SsmConv1d), g.Conv*g.Chans, g.Conv, g.Chans)
			}
			gateWidth, dtWidth := g.VHeads, g.VHeads
			if cfg.ChanDecay() {
				gateWidth, dtWidth = g.VHeads*g.KDim, g.VHeads*g.KDim
			}
			normWidth, dWidth, cbWidth := g.VDim, 0, 0
			if g.Scan {
				// A is a row of the state per channel; D, dt's bias and the
				// convolution's bias one per channel; no gated norm.
				gateWidth, normWidth = g.VHeads*g.KDim, 0
				if _, err = vec(jlm.RoleSSMD, bi); err != nil {
					return nil, err
				}
				if _, err = optVec(jlm.RoleSSMConvBias, bi); err != nil {
					return nil, err
				}
				dWidth, cbWidth = g.VHeads, g.Chans
				for _, v := range []struct {
					role jlm.Role
					want int
				}{{jlm.RoleSSMDtNorm, g.Rank}, {jlm.RoleSSMBNorm, g.KDim}, {jlm.RoleSSMCNorm, g.KDim}} {
					if got := l.Vector(v.role); got != nil && len(got) != v.want {
						return nil, fmt.Errorf("model: block %d %v is %d wide, want %d", bi, v.role, len(got), v.want)
					}
				}
				// The three norms come together or not at all, as llama.cpp
				// applies them.
				if (l.Vector(jlm.RoleSSMDtNorm) == nil) != (l.Vector(jlm.RoleSSMBNorm) == nil) || (l.Vector(jlm.RoleSSMDtNorm) == nil) != (l.Vector(jlm.RoleSSMCNorm) == nil) {
					return nil, fmt.Errorf("model: block %d carries some of Mamba-1's dt/B/C norms and not all", bi)
				}
				for _, m := range []struct {
					name    string
					t       Tensor
					k, rows int
				}{{"ssm_x.dt", l.SsmXDt, g.Chans, g.Rank}, {"ssm_x.b", l.SsmXB, g.Chans, g.KDim},
					{"ssm_x.c", l.SsmXC, g.Chans, g.KDim}, {"ssm_dt", l.SsmDtProj, g.Rank, g.VHeads}} {
					if m.t.K != m.k || m.t.Rows != m.rows {
						return nil, fmt.Errorf("model: block %d %s is %dx%d, want %dx%d",
							bi, m.name, m.t.K, m.t.Rows, m.k, m.rows)
					}
				}
			}
			if g.Ssd {
				// Mamba-2's two vectors more, and its gated norm's weight is
				// every channel of y, a group of Inner/Groups at a time.
				if _, err = vec(jlm.RoleSSMD, bi); err != nil {
					return nil, err
				}
				if _, err = optVec(jlm.RoleSSMConvBias, bi); err != nil {
					return nil, err
				}
				normWidth, dWidth, cbWidth = g.VHeads*g.VDim, g.VHeads, g.Chans
				if l.Vector(jlm.RoleSSMNorm) == nil {
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
				{"ssm_a", len(l.Vector(jlm.RoleSSMA)), gateWidth}, {"ssm_dt.bias", len(l.Vector(jlm.RoleSSMDtBias)), dtWidth},
				{"ssm_norm", len(l.Vector(jlm.RoleSSMNorm)), normWidth},
				{"ssm_d", len(l.Vector(jlm.RoleSSMD)), dWidth}, {"ssm_conv1d.bias", len(l.Vector(jlm.RoleSSMConvBias)), cbWidth},
			} {
				if v.have != v.want {
					return nil, fmt.Errorf("model: block %d %s is %d wide, want %d", bi, v.name, v.have, v.want)
				}
			}
		}
		if mla {
			// The kv latent's norm is always there; the q latent's exists only
			// in the two-step form, which is what QLoraRank says.
			if _, err = vec(jlm.RoleAttnKVANorm, bi); err != nil {
				return nil, err
			}
			if cfg.QLoraRank != 0 {
				if _, err = vec(jlm.RoleAttnQANorm, bi); err != nil {
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
				{"attn_kv_a_norm", len(l.Vector(jlm.RoleAttnKVANorm)), cfg.KVLoraRank},
				{"attn_q_a_norm", len(l.Vector(jlm.RoleAttnQANorm)), cfg.QLoraRank},
			} {
				if v.want != 0 && v.have != v.want {
					return nil, fmt.Errorf("model: block %d %s is %d wide, want %d",
						bi, v.name, v.have, v.want)
				}
			}
			// Cut per head once, at load: wkb and wvb are banks of NHead
			// sheets, and the absorb is nn.MatVecPackedGather over them with
			// the head as the "expert".
			nope := cfg.HeadDim - cfg.NRot
			for h := 0; h < cfg.NHead; h++ {
				t, err := bankExpert(l.Wkb, jlm.RoleAttnKB, cfg.NHead, cfg.KVLoraRank, nope, h)
				if err != nil {
					return nil, err
				}
				l.kbHead = append(l.kbHead, t)
				if t, err = bankExpert(l.Wvb, jlm.RoleAttnVB, cfg.NHead, cfg.HeadDimV, cfg.KVLoraRank, h); err != nil {
					return nil, err
				}
				l.vbHead = append(l.vbHead, t)
			}
		}
		// Every other matrix the block stores is loaded under its role, a
		// tensor of the block's own: what a stated graph binds that no field
		// above names.
		for k := range c.Entries() {
			e := &c.Entries()[k]
			if e.Block != bi || e.Index != -1 || e.Role.Expanded() || e.NDim < 2 || squeezed(e) ||
				l.Matrix(e.Role) != nil || l.Vector(e.Role) != nil {
				continue
			}
			if err := getTo(new(Tensor), e.Role, bi, -1); err != nil {
				return nil, err
			}
		}
		// A vector the block stores 2-D is a matrix too, a row of its
		// second dimension each, where no tensor views it already: a
		// convolution's taps a stated graph binds as one.
		for _, rv := range l.Vecs {
			e, ok := c.Find(rv.Role, bi, -1)
			if !ok || e.NDim != 2 || len(rv.V) == 0 || l.Matrix(rv.Role) != nil ||
				len(l.SsmConvTaps.Data) > 0 && &l.SsmConvTaps.Data[0] == (*byte)(unsafe.Pointer(&rv.V[0])) {
				continue
			}
			t := F32Tensor(rv.V, int(e.Dims[1]), int(e.Dims[0]))
			l.Mats = append(l.Mats, roleTensor{rv.Role, &t})
		}
		if cfg.QKNormAt(i) {
			if l.QNorm, err = vec(jlm.RoleAttnQNorm, bi); err != nil {
				return nil, err
			}
			// A KV-sharing block norms only q: it caches nothing.
			if !cfg.KVShared(i) {
				if l.KNorm, err = vec(jlm.RoleAttnKNorm, bi); err != nil {
					return nil, err
				}
			}
			// A norm of the wrong width reads past its projection or
			// leaves part of it unnormed, fluently.
			_, _, wq, wk := cfg.QKNormShapeAt(i)
			if cfg.KVShared(i) {
				wk = 0
			}
			if len(l.QNorm) != wq || len(l.KNorm) != wk {
				return nil, fmt.Errorf("model: block %d's q/k norms are %d and %d wide, want %d and %d",
					i, len(l.QNorm), len(l.KNorm), wq, wk)
			}
		}
		if cfg.MoEAt(i) {
			if l.Router.Rows != cfg.NExpert || l.Router.K != cfg.NEmbd {
				return nil, fmt.Errorf("model: block %d router is %dx%d, want %dx%d",
					bi, l.Router.K, l.Router.Rows, cfg.NEmbd, cfg.NExpert)
			}
			l.Experts = make([]expert, cfg.NExpert)
			if l.LegacyExperts {
				// NFFNExp comes from the tensor, not from a ratio of other
				// keys.
				for e := range l.Experts {
					x := &l.Experts[e]
					if x.Gate, err = get(jlm.RoleExpGate, bi, int32(e)); err != nil {
						return nil, err
					}
					if x.Up, err = get(jlm.RoleExpUp, bi, int32(e)); err != nil {
						return nil, err
					}
					if x.Down, err = get(jlm.RoleExpDown, bi, int32(e)); err != nil {
						return nil, err
					}
					if e == 0 {
						cfg.NFFNExp = x.Gate.Rows
					}
					if x.Gate.Rows != cfg.NFFNExp || x.Gate.K != cfg.ExpWidth() {
						return nil, fmt.Errorf("model: block %d expert %d gate is %dx%d, want %dx%d",
							bi, e, x.Gate.K, x.Gate.Rows, cfg.ExpWidth(), cfg.NFFNExp)
					}
					if x.Down.Rows != cfg.ExpWidth() || x.Down.K != cfg.NFFNExp {
						return nil, fmt.Errorf("model: block %d expert %d down is %dx%d, want %dx%d",
							bi, e, x.Down.K, x.Down.Rows, cfg.NFFNExp, cfg.ExpWidth())
					}
				}
				continue
			}
			for e := range l.Experts {
				x := &l.Experts[e]
				if !l.UngatedExp {
					if x.Gate, err = bankExpert(l.Gate, jlm.RoleExpGateBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
						return nil, err
					}
				}
				if x.Up, err = bankExpert(l.Up, jlm.RoleExpUpBank, cfg.NExpert, cfg.NFFNExp, cfg.ExpWidth(), e); err != nil {
					return nil, err
				}
				// down's shape is the transpose of gate's: [n_ff_exp, n_embd].
				if x.Down, err = bankExpert(l.Down, jlm.RoleExpDownBank, cfg.NExpert, cfg.ExpWidth(), cfg.NFFNExp, e); err != nil {
					return nil, err
				}
			}
		}
	}
	cur = nil
	if err := buildMTP(c, m, get, vec, optVec); err != nil {
		return nil, err
	}
	return m, nil
}

// buildMTP binds what each multi-token-prediction block has beyond the
// ordinary block build() already bound (jlm.Config.NMTP).
func buildMTP(c *jlm.File, m *Model,
	get func(jlm.Role, int32, int32) (Tensor, error),
	vec, optVec func(jlm.Role, int32) ([]float32, error)) error {
	cfg := m.Cfg
	for i := cfg.NLayer; i < cfg.NLayer+cfg.NMTP; i++ {
		bi := int32(i)
		w := &mtpWeights{Head: m.Output, Embd: m.Embd, HeadNorm: m.OutNorm}
		var err error
		if w.EhProj, err = get(jlm.RoleNextnEHProj, bi, -1); err != nil {
			return err
		}
		if w.EhProj.K != 2*cfg.NEmbd || w.EhProj.Rows != cfg.NEmbd {
			return fmt.Errorf("model: prediction block %d eh_proj is %dx%d, want %dx%d",
				i, w.EhProj.K, w.EhProj.Rows, 2*cfg.NEmbd, cfg.NEmbd)
		}
		if w.Enorm, err = vec(jlm.RoleNextnENorm, bi); err != nil {
			return err
		}
		if w.Hnorm, err = vec(jlm.RoleNextnHNorm, bi); err != nil {
			return err
		}
		if hn, err := optVec(jlm.RoleNextnHeadNorm, bi); err != nil {
			return err
		} else if hn != nil {
			w.HeadNorm = hn
		}
		// Over a residual of every stream (ResidStreams) the block reads the
		// trunk's, each stream normed against its own row of hnorm, and its
		// head is the streams' stated mix, not a norm.
		hw, hn := cfg.NEmbd, cfg.NEmbd
		if cfg.ResidStreams() {
			hw, hn = cfg.ResidW(), len(w.HeadNorm)
		}
		for _, v := range []struct {
			name    string
			n, want int
		}{{"enorm", len(w.Enorm), cfg.NEmbd}, {"hnorm", len(w.Hnorm), hw},
			{"shared_head_norm", len(w.HeadNorm), hn}} {
			if v.n != v.want {
				return fmt.Errorf("model: prediction block %d %s is %d wide, want %d", i, v.name, v.n, v.want)
			}
		}
		// The block's own head and embedding exist only where they differed
		// from the trunk's (convert.nextnTensors dropped the copies).
		if c.Has(jlm.RoleNextnHead, bi, -1) {
			if w.Head, err = get(jlm.RoleNextnHead, bi, -1); err != nil {
				return err
			}
		}
		if c.Has(jlm.RoleNextnEmbd, bi, -1) {
			if w.Embd, err = get(jlm.RoleNextnEmbd, bi, -1); err != nil {
				return err
			}
			w.ownEmbd = true
		}
		for _, t := range []Tensor{w.Head, w.Embd} {
			if t.Rows != cfg.NVocab || t.K != cfg.NEmbd {
				return fmt.Errorf("model: prediction block %d's head or embedding is %dx%d, want %dx%d",
					i, t.K, t.Rows, cfg.NEmbd, cfg.NVocab)
			}
		}
		m.Layers[i].Mtp = w
	}
	return nil
}

// bankExpert validates a 3-D expert bank and returns expert e's view of it.
//
// The check is the exact byte total, not the dimensions: a bank short by an
// expert would otherwise slice inside the buffer and return another expert's
// weights, which every tier would agree on.
func bankExpert(b Tensor, role jlm.Role, nExpert, nrow, k, e int) (Tensor, error) {
	if b.K != k || b.Rows != nExpert*nrow {
		return Tensor{}, fmt.Errorf("model: %v is %d x %d, want %d x %d (%d experts of %d)",
			role, b.K, b.Rows, k, nExpert*nrow, nExpert, nrow)
	}
	be, bb := int(b.Typ.BlockElems()), int(b.Typ.BlockBytes())
	if be == 0 || b.K%be != 0 {
		return Tensor{}, fmt.Errorf("model: %v rows are %d elements, not a multiple of %s's %d",
			role, b.K, b.Typ, be)
	}
	// A packed bank is priced by the packer, not by the source block size:
	// b.data is the payload span only.
	if b.Packed != nil {
		q, ok := layout.QuantOf(b.Typ)
		if !ok {
			return Tensor{}, fmt.Errorf("model: %v: %s has no device layout", role, b.Typ)
		}
		// An expert is a contiguous sheet: the container packs a bank as
		// n_expert independent matrices back to back (jlm.sheetsOf), so expert
		// e owns one range of each of the three spans.
		nq, nd, nsc, err := layout.PackedWords(q, nrow, b.K)
		if err != nil {
			return Tensor{}, fmt.Errorf("model: %v: %w", role, err)
		}
		// Empty spans at build time are not an error: build() needs the shape,
		// and pageIn re-runs bindPacked, which re-runs this, for the bytes.
		bound := len(b.Data) != 0
		// The span is padded to jlm.Align, so the test is a range and not an
		// equality.
		if want := nq * 4 * nExpert; bound &&
			(len(b.Data) < want || len(b.Data)-want >= jlm.Align) {
			return Tensor{}, fmt.Errorf("model: %v holds %d packed bytes, want %d (%d experts of %d words)",
				role, len(b.Data), want, nExpert, nq)
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
		qs, err := cut(b.Packed.QS, nq)
		if err != nil {
			return Tensor{}, err
		}
		dd, err := cut(b.Packed.D, nd)
		if err != nil {
			return Tensor{}, err
		}
		sc, err := cut(b.Packed.SC, nsc)
		if err != nil {
			return Tensor{}, err
		}
		p := &nn.Packed{QS: qs, D: dd, SC: sc, Row: 0, Stride: nrow}
		return Tensor{E: b.E, Typ: b.Typ, Data: qs, Rows: nrow, K: b.K, Packed: p}, nil
	}
	// Not bound yet is not an error here either (an all-F32 bank lives in a
	// page); build() needs the shape and pageIn re-runs this for the bytes.
	if len(b.Data) == 0 {
		return Tensor{E: b.E, Typ: b.Typ, Rows: nrow, K: b.K}, nil
	}
	if want := nExpert * nrow * (b.K / be * bb); want != len(b.Data) {
		return Tensor{}, fmt.Errorf("model: %v holds %d bytes, want %d", role, len(b.Data), want)
	}
	return rows(b, e*nrow, nrow)
}

// rows returns the sub-tensor made of n rows of t starting at from.
//
// It aliases rather than copies: a row-major matrix's row range is a contiguous
// byte range.
func rows(t Tensor, from, n int) (Tensor, error) {
	// A packed row range is not a byte range: the device layout is
	// [sub-block][word][row], so the row offset travels with the weight.
	if t.Packed != nil {
		if from < 0 || n <= 0 || from+n > t.Rows {
			return Tensor{}, fmt.Errorf("model: %s: rows [%d,%d) of %d do not fit",
				t.name(), from, from+n, t.Rows)
		}
		stride := t.Packed.Stride
		if stride == 0 {
			stride = t.Rows
		}
		p := &nn.Packed{QS: t.Packed.QS, D: t.Packed.D, SC: t.Packed.SC,
			Row: t.Packed.Row + from, Stride: stride}
		return Tensor{E: t.E, Typ: t.Typ, Data: t.Data, Rows: n, K: t.K, Packed: p}, nil
	}
	be, bb := int(t.Typ.BlockElems()), int(t.Typ.BlockBytes())
	if be == 0 || t.K%be != 0 {
		return Tensor{}, fmt.Errorf("model: %s rows are %d elements, not a multiple of %s's %d",
			t.name(), t.K, t.Typ, be)
	}
	rowBytes := t.K / be * bb
	lo, hi := from*rowBytes, (from+n)*rowBytes
	if from < 0 || n <= 0 || hi > len(t.Data) {
		return Tensor{}, fmt.Errorf("model: %s: rows [%d,%d) of %d do not fit %d bytes",
			t.name(), from, from+n, t.Rows, len(t.Data))
	}
	return Tensor{E: t.E, Typ: t.Typ, Data: t.Data[lo:hi], Rows: n, K: t.K}, nil
}

func Itoa(v int) string {
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
	for i := range m.Container.Entries() {
		qs, d, sc := m.Container.SpanLen(&m.Container.Entries()[i])
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
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		qs, d, sc := m.Container.SpanLen(e)
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
func (m *Model) MaxMatVec() (rows, k int) {
	// The shape is the entry's, read whether or not the block is bound: a
	// block another instance placed has given its host page back (releaseLayer)
	// and may be re-bound under this call, so its data is neither the answer
	// nor safe to read here.
	see := func(t *Tensor) {
		if t.E == nil {
			return
		}
		if t.Rows > rows {
			rows = t.Rows
		}
		if t.K > k {
			k = t.K
		}
	}
	for i := range m.Layers {
		for _, mt := range m.Layers[i].Mats {
			see(mt.T)
		}
	}
	see(&m.Output)
	see(&m.Embd)
	see(&m.PleProj)
	for _, d := range m.DenseMats {
		see(d.T)
	}
	return rows, k
}

// blockReadBytes is how many file bytes of block li a single token reads. On a
// mixture it is far less than the block's size: a placement removes the read
// from the host but costs the size on the device.
func (m *Model) BlockReadBytes(li int) uint64 {
	l := &m.Layers[li]
	var n uint64
	for _, t := range []Tensor{l.Wq, l.Wk, l.Wv, l.Wo, l.Router} {
		n += TensorBytes(t)
	}
	dense := TensorBytes(l.Gate) + TensorBytes(l.Up) + TensorBytes(l.Down)
	if c := m.Cfg; c.MoE() && c.NExpert > 0 {
		dense = dense / uint64(c.NExpert) * uint64(c.NExpertUsed)
	}
	return n + dense
}

// expertReadBytes is what a token reads of block li's routed experts: the
// selected share of its bank, what streaming the block sends per token.
func (m *Model) ExpertReadBytes(li int) uint64 {
	l, c := &m.Layers[li], m.Cfg
	if !c.MoE() || c.NExpert == 0 {
		return 0
	}
	return (TensorBytes(l.Gate) + TensorBytes(l.Up) + TensorBytes(l.Down)) / uint64(c.NExpert) * uint64(c.NExpertUsed)
}

// headReadBytes is what a token reads for the output projection, or zero when
// the model ties its embedding (the projection is then the embedding table,
// which is already counted and already on the host).
func (m *Model) HeadReadBytes() uint64 { return TensorBytes(m.Output) }

// tensorBytes is what this weight costs to READ once, which for a container is
// all three of its spans and not just the payload.
//
// It prices the tensor from its entry, not its residency: the spans are empty
// for a weight whose page is out.
func TensorBytes(t Tensor) uint64 {
	if t.E != nil {
		qs, d, sc := jlm.SpanLenOf(t.E)
		return qs + d + sc
	}
	n := uint64(len(t.Data))
	if t.Packed != nil {
		n += uint64(len(t.Packed.D) + len(t.Packed.SC))
	}
	return n
}

// containerShape is the (type, k, rows) of one container entry, for the passes
// that size the JIT from every weight in the file.
//
// It reports false for anything the kernels cannot take: a 1-D norm, or a type
// written by a newer converter.
func containerShape(e *jlm.Entry) (Tensor, bool) {
	if e.NDim < 2 {
		return Tensor{}, false
	}
	typ, ok := jlm.SourceType(e.Type)
	if !ok {
		return Tensor{}, false
	}
	rows := 1
	for d := 1; d < int(e.NDim); d++ {
		rows *= int(e.Dims[d])
	}
	return Tensor{E: e, Typ: typ, Rows: rows, K: int(e.Dims[0])}, true
}

// hasPostNorm reports whether this model's blocks carry gemma2/gemma3's two
// extra RMSNorms. It asks the weights rather than the architecture, so a fourth
// architecture that ships them needs no entry anywhere.
func (m *Model) HasNoPreNorm() bool {
	for i := range m.Layers {
		if m.Layers[i].AttnNorm == nil && m.Layers[i].Vector(jlm.RolePostAttnNorm) != nil {
			return true
		}
	}
	return false
}

// hasSubNorm reports whether some block carries BitNet's inner norms.
func (m *Model) HasSubNorm() bool {
	for i := range m.Layers {
		if l := &m.Layers[i]; l.Vector(jlm.RoleAttnSubNorm) != nil || l.Vector(jlm.RoleFFNSubNorm) != nil {
			return true
		}
	}
	return false
}

// hasPostNorm reports whether some block carries gemma2's post-norms.
func (m *Model) HasPostNorm() bool {
	// By index: a copy of each layer would read its bindings, which another
	// instance's placement may be writing.
	for i := range m.Layers {
		if l := &m.Layers[i]; l.Vector(jlm.RolePostAttnNorm) != nil || l.Vector(jlm.RolePostFFNNorm) != nil {
			return true
		}
	}
	return false
}

// actWindowOf is the model's activation window (composer.ActWindow) over the
// types of its matrices: for a model quantizing each row per token
// (Config.TokenAct), one window over its widest row.
func (m *Model) ActWindowOf() int {
	maxK := 0
	var types []quant.Type
	for i := range m.Container.Entries() {
		if t, ok := containerShape(&m.Container.Entries()[i]); ok {
			maxK = max(maxK, t.K)
			types = append(types, t.Typ)
		}
	}
	if m.Cfg.TokenAct {
		return (maxK + 31) / 32 * 32
	}
	return composer.ActWindow(types)
}

// streamHead says the head reads a collapse of several residual streams,
// which the host takes (the streams the converter states): a device runs the blocks and hands every stream back,
// and never folds the head into them.
func (c *Config) StreamHead() bool { return c.statedStreams() > 1 }

// specHidW is a speculation step's hidden state per row: what a prediction
// block reads of the trunk's. The output norm's result, n_embd wide; on
// a residual of every stream (ResidStreams) the residual before the head's
// mix, every stream.
func (c *Config) SpecHidW() int {
	if c.ResidStreams() {
		return c.ResidW()
	}
	return c.NEmbd
}

// ExpWidth is the routed experts' input and output width: their banks' own
// where they read a latent of their own, n_embd everywhere else.
func (c *Config) ExpWidth() int {
	if c.expWidth != 0 {
		return c.expWidth
	}
	return c.NEmbd
}

// expertWidth is the routed experts' width as the first mixture block's up
// bank states it (its rows' width), zero where no block holds one or it is
// n_embd.
func expertWidth(f *jlm.File, cfg *Config) (int, error) {
	for bi := range cfg.NLayer {
		e, ok := f.Find(jlm.RoleExpUpBank, int32(bi), -1)
		if !ok || e.NDim < 1 {
			continue
		}
		w := int(e.Dims[0])
		switch {
		case w == cfg.NEmbd:
			return 0, nil
		case w <= 0 || w > cfg.NEmbd || cfg.ExpertWeightIn:
			return 0, fmt.Errorf("model: %s: routed experts %d wide on n_embd %d (weights on their input %v)",
				cfg.Arch, w, cfg.NEmbd, cfg.ExpertWeightIn)
		}
		return w, nil
	}
	return 0, nil
}

// f32Bytes is a float32 slice addressed as the bytes a kernel reads: the same
// memory, no copy.
func f32Bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// BidirAt reports that a picture's rows see each other in layer li's
// attention.
func (c *Config) BidirAt(li int) bool { return len(c.bidir) == 0 || c.bidir[li] }

// storesAny reports that some block of the container stores one of rs.
func (m *Model) storesAny(rs ...jlm.Role) bool {
	for bi := range m.Container.Config().NLayer {
		for _, r := range rs {
			if m.Container.Has(r, int32(bi), -1) {
				return true
			}
		}
	}
	return false
}
