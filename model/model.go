// Package model is a loaded model: a jlm container's weights, its
// configuration, its tokenizer and chat template, and the graphs it states
// for its blocks, head and vision tower. It runs nothing; package engine
// composes and runs its blocks on the devices an instance places them on.
package model

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
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
	// Pooling is how an EMBEDDING model reads its vector out of the last
	// residual, and PoolNone for every generator. NonCausal is an encoder's
	// attention: every position sees every other. Both come off the
	// container's flags; see jlm.FlagPoolMean and jlm.FlagNonCausal.
	Pooling   jlm.Pooling
	NonCausal bool
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

	// RopePairs is the pairs a row's rotary table holds where the container's
	// pair map states them (jlm.Config.RopePairs), 0 for a rotary of one
	// position: more than NRot/2 where each element of a pair turns by an
	// entry of its own (graph.RopeSplit).
	RopePairs int

	// Mixture of experts. NExpert 0 is a dense FFN. Each expert's width is its
	// bank's (sheetRows), not NFFN, which a MoE file carries for a dense width
	// the model never uses.
	NExpert, NExpertUsed int

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

	// MarkPointer is a pointer readout (jlm.Flag2MarkerPointer).
	MarkPointer bool
	// Joint is a joint head's readout (jlm.JointHead).
	Joint bool

	// Logit softcapping: x = c*tanh(x/c). gemma2 caps the attention scores at
	// 50 and the output logits at 30. Zero means none. Omitting it gives fluent
	// wrong output that only an external oracle catches.
	AttnSoftcap, FinalSoftcap float32

	// YarnOrigCtx is the context past which a LongRoPE turns its long
	// factors. A scaled rotary's angles turn by the factors the container
	// states per pair, and YaRN's score magnitude is in AttnScale
	// (convert.StateRotary).
	YarnOrigCtx int
	// NoPosEnc: the attention blocks apply no positional encoding at all.
	// Position still orders the causal mask and never reaches q or k.
	NoPosEnc bool

	// Multi-head Latent Attention; see jlm.Config for the graph. KVLoraRank 0
	// is every architecture but DeepSeek's.
	//
	// HeadDimV is the value head's width, filled in even where the container
	// stores zero for "same as HeadDim".
	KVLoraRank int
	// RopeTail, KeyValue, QUnitNorm and OutUnrotated are the container's
	// attention statements (jlm.Flag2RopeTail, ...).
	RopeTail, KeyValue, QUnitNorm, OutUnrotated bool
	HeadDimV                                    int

	// The DeepSeek router: sigmoid gating, a selection-only bias, grouped
	// scoring and a scale on the surviving weights. NExpertGroup <= 1 is the
	// ungrouped top-k every other mixture here uses.
	ExpertSigmoid                  bool
	ExpertScale                    float64
	NExpertGroup, NExpertGroupUsed int

	// AttnScale is the softmax's multiplier, already including YaRN's squared
	// magnitude correction where a class's attention scales by it (DeepSeek's
	// softmax_scale), which the converter states in the container's.
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

	// QKVAct is the function q, k and v pass through after their
	// projections, as the container states it (jlm.Config.QKVAct); nil for
	// none.
	QKVAct *jlm.Expr

	// NMTP is how many multi-token-prediction blocks follow the trunk (blocks
	// NLayer..NLayer+NMTP-1); zero for a model that ships none. They run only
	// for speculative drafting (spec.go), never in the trunk.
	NMTP int

	// geoms is each block's own geometry where the blocks differ
	// (jlm.Config.Geoms), nil when every block is the model's; mixed says
	// their attention geometries differ. Set by setGeoms, read through
	// HeadDimAt and friends.
	geoms []jlm.Geom
	mixed bool
	// ownGeoms says the blocks state geometries of their own (setGeoms),
	// where mixed counts a marker head's too.
	ownGeoms bool
	// VNorm RMSNorms each head of v with no weight before it is cached
	// (jlm.Flag2VNorm, Gemma 4).
	VNorm bool
	// bidir is, per layer, whether a picture's rows see each other in its
	// attention, as the container states it (jlm.Config.BidirLayers); nil
	// is every layer.
	bidir []bool
	// The last NKVShared layers attend to an earlier layer's history
	// (KVShared); PLEDim is each layer's slice of the per-layer embeddings,
	// zero for none.
	NKVShared, PLEDim int
	// DeepTaps is how many of the first blocks add a picture's tap rows into
	// its rows after they run (jlm.TowerTap, Qwen3-VL's deepstack): the
	// container's tower's taps, tap i after block i.
	DeepTaps int
	// sel is how the blocks' attention selects the history it reads, the
	// container's statement (jlm.Selection); nil for none.
	sel *jlm.Selection
	// compKinds is each block's compression (the container's CompKinds)
	// and compRate the positions an entry of each kind pools.
	compKinds []jlm.CompKind
	compRate  [3]int
	// HCMult is the residual streams a hyper-connected model carries.
	HCMult int
	// DeltaDecay and DeltaGate are the delta rule's decay and output gate as
	// the container states them (jlm.Config.DeltaDecay), nil on a model with
	// no delta rule.
	DeltaDecay, DeltaGate *jlm.Program
	// stated is the residual's streams as the converter states them
	// (jlm.Streams, the container's own): N of them, each block's reads and
	// writes; nil for a config no container made.
	stated *jlm.Streams
	// LatentNormEps is the MLA latent norms' epsilon.
	LatentNormEps float64
	// routed marks the blocks that store a router, the mixture blocks
	// (MoEAt): read off the container at build(), nil on a model with no
	// mixture.
	routed []bool

	// The classic block (C6); see jlm.FlagLayerNorm and jlm.FlagParallel.
	// LayerNorm makes every text-graph norm mean-centred, with the norm's
	// bias when the block carries one; Parallel feeds attention and the FFN
	// the same block input and adds both into it. RMSEps is the LayerNorm's
	// epsilon when LayerNorm is set.
	LayerNorm, Parallel bool

	// Trunk is how many blocks come before a marker head's (jlm.Flag2HeadBlocks),
	// NLayer where there is none; Head is the head blocks' own config, nil for
	// none. Every per-block accessor answers a head block from Head.
	Trunk int
	Head  *Config
}

// at is the config block li runs at: Head for a marker head's block, c
// otherwise.
func (c *Config) at(li int) *Config {
	if c.Head != nil && li >= c.Trunk && li < c.NLayer {
		return c.Head
	}
	return c
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
func (c *Config) KeyHeadAt(li int) bool {
	_, rate := c.CompAt(li)
	return c.KVLoraRank == 0 && c.SelectAt(li) && rate == 0
}

// KVRowAt is one cached position's width in layer li's history: KVDimAt,
// plus where it selects its keys the indexer's key as one more kv head.
// Every cache allocation, page and migration of a layer goes through it; the
// k and v projections are KVDimAt wide.
func (c *Config) KVRowAt(li int) int {
	if c.KeyHeadAt(li) {
		return (c.NKVHead + 1) * c.HeadDim
	}
	if kind, rate := c.CompAt(li); rate > 0 {
		// The row carries the compressor's values and gates past the key,
		// two heads of each over overlapping windows, and an indexer's
		// beside them.
		w := c.HeadDimAt(li)
		if kind == jlm.CompCSA {
			w *= 2
		}
		if c.SelectAt(li) {
			w += 2 * int(c.sel.HeadDim)
		}
		return c.KVDimAt(li) + 2*w
	}
	return c.KVDimAt(li)
}

// CompAt is block li's compression and the positions an entry of it pools:
// CompNone and 0 for a block that keeps no entries.
func (c *Config) CompAt(li int) (jlm.CompKind, int) {
	if li < 0 || li >= len(c.compKinds) || int(c.compKinds[li]) >= len(c.compRate) {
		return jlm.CompNone, 0
	}
	k := c.compKinds[li]
	return k, c.compRate[k]
}

// MixedGeom reports that the blocks do not all attend at one geometry: a
// block states its own head width, kv head count or rotary width. HeadDim,
// NKVHead and NRot are then the model's, and every per-layer path reads
// HeadDimAt, NKVHeadAt and NRotAt.
func (c *Config) MixedGeom() bool { return c.mixed }

// OwnGeoms reports that the blocks state geometries of their own
// (jlm.Config.Geoms), the sliding ones turning by the second table as their
// own; a marker head's geometry, its blocks' set's, is not one.
func (c *Config) OwnGeoms() bool { return c.ownGeoms }

// setGeoms takes the blocks' own geometries, gs one per block or none.
func (c *Config) setGeoms(gs []jlm.Geom) {
	c.geoms, c.mixed = gs, false
	for li := range gs {
		c.mixed = c.mixed || c.HeadDimAt(li) != c.HeadDim || c.NKVHeadAt(li) != c.NKVHead || c.NRotAt(li) != c.NRot
	}
	c.ownGeoms = c.mixed
}

// HeadDimAt, NKVHeadAt and NRotAt are layer li's attention geometry: the
// block's own where the container states one, the model's otherwise.
func (c *Config) HeadDimAt(li int) int {
	if h := c.at(li); h != c {
		return h.HeadDim
	}
	if li >= 0 && li < len(c.geoms) {
		return int(c.geoms[li].HeadDim)
	}
	return c.HeadDim
}

// NKVHeadAt is layer li's kv head count (see HeadDimAt).
func (c *Config) NKVHeadAt(li int) int {
	if h := c.at(li); h != c {
		return h.NKVHead
	}
	if li >= 0 && li < len(c.geoms) {
		return int(c.geoms[li].NKVHead)
	}
	return c.NKVHead
}

// NHeadAt is layer li's query head count: the model's, or a marker head's
// block's own.
func (c *Config) NHeadAt(li int) int { return c.at(li).NHead }

// LayerNormAt says block li normalises by LayerNorm, its biases its own: a
// marker head's blocks do over a trunk that does not.
func (c *Config) LayerNormAt(li int) bool { return c.at(li).LayerNorm }

// Marks says the model reads its answer at a prompt's marker rows: a marker
// head's scorer or a pointer readout.
func (c *Config) Marks() bool { return c.Head != nil || c.MarkPointer }

// NRotAt is layer li's rotary width (see HeadDimAt).
func (c *Config) NRotAt(li int) int {
	if h := c.at(li); h != c {
		return h.NRot
	}
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
func (c *Config) GQAAt(li int) int { return c.NHeadAt(li) / c.NKVHeadAt(li) }

// QDimAt is the query (and the attention output) width at layer li.
func (c *Config) QDimAt(li int) int { return c.NHeadAt(li) * c.HeadDimAt(li) }

// MaxHeadDim, MaxQDim and MaxNRot are the widest of the blocks'
// geometries: what a buffer that serves every layer is sized by.
func (c *Config) MaxHeadDim() int {
	m := c.HeadDim
	for li := range c.geoms {
		m = max(m, c.HeadDimAt(li))
	}
	if c.Head != nil {
		m = max(m, c.Head.HeadDim)
	}
	return m
}

// KVShared reports that block li computes no k and v of its own and attends
// to an earlier block's history, the one its graph reads (Gemma 4's E2B/E4B;
// graph.Graph.ReadsHistoryOf). A recurrent block in the shared run keeps no
// history and is its own.
func (c *Config) KVShared(li int) bool {
	if h := c.at(li); h != c {
		return h.KVShared(li)
	}
	return c.NKVShared != 0 && li >= c.NLayer-c.NKVShared && li < c.NLayer && c.LayerKind(li).Attends()
}

// NFFNAt is block li's feed-forward width: its own where the container
// states one, NFFN otherwise.
func (c *Config) NFFNAt(li int) int {
	if h := c.at(li); h != c {
		return h.NFFN
	}
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
	if c.Head != nil {
		m = max(m, c.Head.NFFN)
	}
	return m
}

// MaxQDim is the query width at the widest head.
func (c *Config) MaxQDim() int {
	if c.Head != nil {
		return max(c.NHead*c.MaxHeadDim(), c.Head.NHead*c.Head.HeadDim)
	}
	return c.NHead * c.MaxHeadDim()
}

// MaxNRot is the widest rotary any block turns.
func (c *Config) MaxNRot() int {
	m := c.NRot
	for li := range c.geoms {
		m = max(m, c.NRotAt(li))
	}
	return m
}

// RopeW is how many floats one row's rotary table holds: two a pair of its
// table (RopePairs), NRot for a table of a head's rotated pairs.
func (c *Config) RopeW() int {
	if c.RopePairs > 0 {
		return 2 * c.RopePairs
	}
	return c.NRot
}

// MoEAt reports whether block li has a mixture feed-forward: it stores the
// router its stated graph routes its experts by. MoE() is a property of the
// model and this of the block: DeepSeek's leading blocks are dense, Llama 4
// interleaves, Nemotron 3's mixer-only layers and Jamba's dense blocks carry
// no router, and each is a block that stores none.
func (c *Config) MoEAt(li int) bool {
	return c.MoE() && li >= 0 && li < len(c.routed) && c.routed[li]
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
	if h := c.at(li); h != c {
		return h.QKNormShapeAt(li)
	}
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
func (c *Config) QKNormAt(li int) bool {
	h := c.at(li)
	return jlm.QKNormOn(h.QKNorm, h.LayerKind(li))
}

// LayerKind is what layer i actually is. A model with no per-layer map is
// ordinary attention all the way down, which is every architecture but one.
//
// The container states it per layer; it is never derived from an architecture
// name.
func (c *Config) LayerKind(i int) jlm.LayerKind {
	if h := c.at(i); h != c {
		return h.LayerKind(i)
	}
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
	if c.NVocab == 0 {
		return nil, fmt.Errorf("model: the container holds no vocabulary: it is not a language model")
	}
	if c.HCMult > 1 && c.Streams.N <= 1 {
		return nil, fmt.Errorf("model: the container carries %d hyper-connected streams it does not state: "+
			"convert the model again", c.HCMult)
	}
	for li, k := range c.CompKinds {
		if k == jlm.CompCSA && !c.Selection.At(li) {
			return nil, fmt.Errorf("model: block %d compresses overlapping windows with no indexer stated: "+
				"convert the model again", li)
		}
	}
	out := &Config{
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
		AttnSoftcap:  c.AttnSoftcap,
		FinalSoftcap: c.FinalSoftcap,
		YarnOrigCtx:  int(c.YarnOrigCtx),
		NoPosEnc:     c.Flags.Has(jlm.FlagNoPosEnc),
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

		RopeTail:         c.Flags2.Has(jlm.Flag2RopeTail),
		KeyValue:         c.Flags2.Has(jlm.Flag2KeyValue),
		QUnitNorm:        c.Flags2.Has(jlm.Flag2QUnitNorm),
		OutUnrotated:     c.Flags2.Has(jlm.Flag2OutUnrotated),
		MarkPointer:      c.Flags2.Has(jlm.Flag2MarkerPointer),
		Joint:            c.JointHead != nil,
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
		QKVAct:         c.QKVAct(),
		LayerNorm:      c.Flags.Has(jlm.FlagLayerNorm),
		Parallel:       c.Flags.Has(jlm.FlagParallel),

		HCMult: int(c.HCMult),
	}
	out.setGeoms(c.Geoms)
	out.VNorm = c.Flags2.Has(jlm.Flag2VNorm)
	out.NKVShared, out.PLEDim = int(c.NKVShared), int(c.PLEDim)
	out.bidir = c.BidirLayers
	out.sel = c.Selection
	out.compKinds, out.compRate = c.CompKinds, [3]int{0, int(c.CompRateCSA), int(c.CompRateHCA)}

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
	// The score's scale: 1/sqrt(HeadDim) where the container states none. A
	// stated one (Granite's attention.scale, gemma 27B's, YaRN's score
	// magnitude folded in by the converter) is taken at its word.
	out.AttnScale = 1 / math.Sqrt(float64(out.HeadDim))
	if c.AttnScale != 0 {
		out.AttnScale = float64(c.AttnScale)
	}

	out.DeltaDecay, out.DeltaGate = c.DeltaDecay, c.DeltaGate
	if out.DeltaDecay == nil {
		var err error
		if out.DeltaDecay, out.DeltaGate, err = olderDelta(c, stores); err != nil {
			return nil, err
		}
	}
	out.LatentNormEps = float64(c.LatentNormEps)
	if out.LatentNormEps == 0 {
		out.LatentNormEps = out.RMSEps
	}
	switch {
	}
	out.stated = &c.Streams
	// The after-rotary norm is FlagQKNorm's per-head norm moved (see
	// jlm.FlagQKNormPostRope); without the norm, or on a whole-projection one,
	// it describes nothing the engine runs.
	if out.QKNormPost && (!out.QKNorm || out.QKNormWide) {
		return nil, fmt.Errorf("model: FlagQKNormPostRope needs the per-head FlagQKNorm "+
			"(QKNorm %v, QKNormWide %v)", out.QKNorm, out.QKNormWide)
	}
	// Zero means one for both scales; see jlm.Config.
	if out.EmbdScale == 0 {
		out.EmbdScale = 1
	}
	if out.AttnFactor == 0 {
		out.AttnFactor = 1
	}
	if out.NLayer == 0 || out.NEmbd == 0 || out.NHead == 0 {
		return nil, fmt.Errorf("model: the container's config is empty")
	}
	out.Trunk = out.NLayer
	if c.Flags2.Has(jlm.Flag2HeadBlocks) {
		h, err := headConfigOf(c, out)
		if err != nil {
			return nil, err
		}
		out.Head, out.NLayer = h, out.NLayer+int(c.HeadBlocks())
	}
	return out, nil
}

// headConfigOf is a marker head's blocks as a config (jlm.Flag2HeadBlocks):
// every block pre-norm with LayerNorms over the trunk's width, attending
// both ways with no position encoding and no window. Its attention and its
// norms' epsilon are its blocks' stated graph's and its feed-forward width
// their tensors', which build reads (statedHead).
func headConfigOf(c *jlm.Config, trunk *Config) (*Config, error) {
	if c.HeadBlocks() == 0 {
		return nil, fmt.Errorf("model: the container states a marker head of no blocks")
	}
	return &Config{NLayer: trunk.NLayer + int(c.HeadBlocks()), NEmbd: int(c.NEmbd), NFFN: trunk.NFFN,
		NVocab: trunk.NVocab, NCtx: trunk.NCtx, EmbdScale: 1, AttnFactor: 1, ResidualScale: 1, LogitScale: 1,
		ExpertScale: 1, NonCausal: true, NoPosEnc: true, LayerNorm: true, Trunk: trunk.NLayer}, nil
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
	// Vecs is every vector the file stores for the block, by role
	// (jlm.Role.Expanded), read once at load.
	Vecs []nn.RoleVector
	// mats is every matrix of the block's the loader bound, by the role it
	// was loaded under, pointing at the field that holds it: one role can
	// land in different fields by block kind (RoleAttnQKV in wq or ssmIn).
	Mats []roleTensor

	Wq, Wk, Wv, Wo Tensor
	// vFromK says the block has no v projection and v is k's (Gemma 4's
	// global layers): wv is unbound and every path copies k's projection
	// before k's norm and rotary run.
	VFromK bool
	// banksFrom is what the expert banks spanned when their views were last
	// cut: bindPacked re-cuts only when a bank moved, since a cut is a fresh
	// nn.Packed per view and a resident block is rebound every token.
	banksFrom      [3]bankSpans
	Gate, Up, Down Tensor
	// On a mixture block (a matrix under jlm.RoleRouter) gate/up/down hold
	// the three 3-D banks and experts the per-expert views of them.
	Experts []expert
	// ungatedExp is a mixture whose experts have no gate bank (Nemotron 3):
	// each is down(act(up(h))), the activation alone, as an ungated FFN is.
	UngatedExp bool

	// noFFN is a block that is its mixer alone: plain Mamba-2's, and a
	// Nemotron-H mixer layer not followed by an MLP. It ends at the first
	// residual on every entry point and every tier.
	NoFFN bool

	// mtp is a multi-token-prediction block's own weights, nil on every trunk
	// block. See jlm.Config.NMTP.
	Mtp *mtpWeights
}

// mtpWeights is what a prediction block has beyond an ordinary block: the two
// eh_proj that makes its residual from the next token's embedding and the
// trunk's hidden state (its two input norms are the block's vectors), and its
// head norm. head and embd are
// the block's own copies where the container kept them (they differed from the
// trunk's); otherwise they are the trunk's, as headNorm is the trunk's output
// norm where the block has none.
type mtpWeights struct {
	EhProj     Tensor
	HeadNorm   []float32
	Head, Embd Tensor
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
	Cfg      *Config
	Vocab    *tok.Vocab

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
	// Audio is the container's stated audio tower lowered into graphs
	// (BuildAudioGraphs), nil for a model with none: a chained tower whose
	// blocks follow the vision tower's.
	Audio *TowerGraphs
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
	// Ropes is the model's rotaries past its own (Rope), rotary i+1 at [i]
	// (graph.Attrs.Rotary, nn.RopeTables): the sliding layers' where the
	// container states them a base, width or magnitude of their own.
	Ropes []nn.Rope
	// RopeLong is the rotary with the container's long factors
	// (rope_factors_long), which an instance whose context passes LongCtx
	// turns (RopeFor); nil where the container holds none. LongCtx is the
	// context the short factors were trained to.
	RopeLong *nn.Rope
	LongCtx  int
	// Ngram is the n-gram block's table.
	Ngram   *ngramTable
	OutNorm []float32
	// posEmbd is a learned absolute position table (starcoder), one row per
	// position, added to the token's embedding before block 0. Empty on every
	// model that encodes position with a rotary; its presence is the fact.
	PosEmbd Tensor
	Output  Tensor
	Layers  []Layer
	// actWindow is how many activation elements share one int8 scale, the
	// same on every device (actWindowOf).
	ActWindow int
	// gc is where the weights live and what their budget leaves the
	// collector (WithGCPlan).
	Gc sched.GCPlan
	// chatClock is the clock a chat template's strftime_now reads; nil is the
	// wall clock (WithChatClock).
	chatClock func() time.Time

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
		if ue := (*jlm.UpgradeError)(nil); errors.As(err, &ue) {
			return nil, &UpgradeRequiredError{Path: path, What: ue.What, Reconvert: ue.Reconvert}
		}
		return nil, err
	}
	if err := m.Unstated(); err != nil {
		m.Close()
		return nil, err
	}
	// The check leaves the graphs to be built when first asked, from the
	// config as it is then: the front set is keyed on the config's address,
	// and a test states another model by setting a field of it in place.
	m.Front.Set = nil
	return m, nil
}

// UpgradeRequiredError is a container written before this build states what
// the composer reads of it. It carries the upgrade command.
type UpgradeRequiredError struct {
	Path, What string
	// Reconvert says an upgrade cannot state it: the model is converted
	// again from its source.
	Reconvert bool
}

// Error names the file, what it does not state and the command that states
// it: an upgrade in place, or a conversion from the source where an upgrade
// cannot.
func (e *UpgradeRequiredError) Error() string {
	if e.Reconvert {
		return fmt.Sprintf("jitllm: %s was written before this build states %s, which the composer reads, "+
			"and an upgrade cannot state it from the container.\n  Convert it again from its source:  jitllm convert", e.Path, e.What)
	}
	return fmt.Sprintf("jitllm: %s was written before this build states %s, which the composer reads.\n"+
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
	if err := m.statedRotaries(); err != nil {
		c.Close()
		return nil, err
	}
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
		if a := c.Vision().Audio; a != nil && a.Tower != nil {
			if m.Audio, err = m.BuildAudioGraphs(); err != nil {
				c.Close()
				return nil, err
			}
		}
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

// CPUResidentBlocks is how many of this model's block pages the CPU device's
// frames currently hold. RSS cannot stand in for it: a released page shows only
// when the collector runs, and the KV cache dwarfs the weights anyway.
func (m *Model) CPUResidentBlocks(fr *jlm.Frames) int {
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

// RolesPlan is the ranges of block li's page holding its entries of roles,
// the routed banks never among them (moe() reads the sheets a token picks):
// what a part binding only those reads of the page.
func (m *Model) RolesPlan(li int, roles []jlm.Role) []jlm.Range {
	c := m.Container
	if c == nil || li >= int(c.H.NBlocks) {
		return nil
	}
	var rs []jlm.Range
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Role.Vision() || e.Block != int32(li) || e.Role.Expanded() || c.ExpertPaged(e) || !slices.Contains(roles, e.Role) {
			continue
		}
		switch e.Role {
		case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank:
			continue
		}
		rs = appendSpans(rs, c, e)
	}
	return rs
}

// PageInRanges is PageIn reading only ranges rs of block li's page, binding
// l at the whole page: a span outside rs is the right length and unread.
func (m *Model) PageInRanges(fr *jlm.Frames, li int, rs []jlm.Range, l *Layer) error {
	if m.Container == nil || li >= len(m.Layers) || li >= m.Container.NPages() {
		return nil
	}
	if err := fr.EnsureRanges(li, rs); err != nil {
		return err
	}
	return bindPacked(fr, m.Cfg, l)
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
		case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank:
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
// when asked for. Since v26 a bank is never in it; its sheets are expert
// pages.
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
		// The base is skipped for a block on a device: it was uploaded at
		// placement, binding needs only addresses, and pageIn re-reads it for
		// a later host consumer.
		if base {
			rs = appendSpans(rs, c, e)
		}
	}
	return rs, nil
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
		bindSheet(cf, &l.Gate, sheetRows(&l.Gate, cfg.NExpert), int(x), &v.Gate)
		bindSheet(cf, &l.Up, sheetRows(&l.Up, cfg.NExpert), int(x), &v.Up)
		bindSheet(cf, &l.Down, sheetRows(&l.Down, cfg.NExpert), int(x), &v.Down)
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
	*v.Packed = nn.Packed{QS: qs, D: d, SC: sc}
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
	l.banksFrom = [3]bankSpans{}
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
			if x.Gate, err = bankExpert(l.Gate, jlm.RoleExpGateBank, cfg.NExpert, e); err != nil {
				return err
			}
		}
		if x.Up, err = bankExpert(l.Up, jlm.RoleExpUpBank, cfg.NExpert, e); err != nil {
			return err
		}
		if x.Down, err = bankExpert(l.Down, jlm.RoleExpDownBank, cfg.NExpert, e); err != nil {
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
	// A marker head's feed-forward is as wide as its blocks store it.
	if h := cfg.Head; h != nil {
		e, ok := c.Find(jlm.RoleFFNUp, int32(cfg.Trunk), -1)
		if !ok {
			return nil, fmt.Errorf("model: the marker head's block %d stores no feed-forward", cfg.Trunk)
		}
		h.NFFN = e.Rows()
		if err := (&Model{Cfg: cfg, Container: c}).statedHead(h); err != nil {
			return nil, err
		}
		cfg.mixed = cfg.mixed || h.HeadDim != cfg.HeadDim || h.NKVHead != cfg.NKVHead || h.NRot != cfg.NRot
	}
	if v := c.Vision(); v != nil && v.Tower != nil && len(v.Tower.Taps) > 0 {
		cfg.DeepTaps = len(v.Tower.Taps)
		if cfg.DeepTaps > cfg.NLayer {
			return nil, fmt.Errorf("model: %d tap rows added after the first of %d blocks", cfg.DeepTaps, cfg.NLayer)
		}
	}
	// A block of a mixture model is a mixture where it stores a router: the
	// container states the mixture model-wide, and a block with none runs
	// its mixer alone (Nemotron 3) or a dense FFN (Jamba, DeepSeek's lead).
	if cfg.MoE() {
		cfg.routed = make([]bool, cfg.NLayer+cfg.NMTP)
		for i := range cfg.routed {
			cfg.routed[i] = c.Has(jlm.RoleRouter, int32(i), -1)
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
		// A model-level tensor of one dimension is a vector whatever its
		// role: a joint head's norms, biases and scales, a scorer stored as
		// the one row it is.
		if e.Index != -1 || !(e.Role.Expanded() || e.NDim == 1) || e.Role.Vision() || e.Block != jlm.DenseBlock {
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
		tok, err := get(jlm.RolePLETokEmbd, jlm.DenseBlock, -1)
		if err != nil {
			return nil, err
		}
		proj, err := get(jlm.RolePLEModelProj, jlm.DenseBlock, -1)
		if err != nil {
			return nil, err
		}
		norm, err := vec(jlm.RolePLEProjNorm, jlm.DenseBlock)
		if err != nil {
			return nil, err
		}
		// The table may stop short of the vocabulary: Gemma 3n's ids past
		// it (its vision and audio tokens) read row 0 (pleInputs).
		w := cfg.NLayer * p
		if tok.K != w || tok.Rows == 0 || tok.Rows > cfg.NVocab || proj.K != cfg.NEmbd ||
			proj.Rows != w || len(norm) != p {
			return nil, fmt.Errorf("model: per-layer embeddings: a table %dx%d, a projection %dx%d and "+
				"a norm of %d, want %d wide over at most %d tokens from %d, and a norm of %d",
				tok.Rows, tok.K, proj.K, proj.Rows, len(norm), w, cfg.NVocab,
				cfg.NEmbd, p)
		}
	}
	if err := loadNGramTable(m, cfg, c); err != nil {
		return nil, err
	}
	outNormB, err := optVec(jlm.RoleOutputNormBias, jlm.DenseBlock)
	if err != nil {
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
	outB, err := optVec(jlm.RoleOutputBias, jlm.DenseBlock)
	if err != nil {
		return nil, err
	}
	if outB != nil && len(outB) != cfg.NVocab {
		return nil, fmt.Errorf("model: the head's bias has %d values, want %d", len(outB), cfg.NVocab)
	}
	// An RMSNorm has no bias, so a norm bias without FlagLayerNorm is a
	// container that disagrees with itself.
	if outNormB != nil && !cfg.LayerNorm {
		return nil, fmt.Errorf("model: a norm bias on an RMSNorm model")
	}
	// The table's pairs are the container's: each pair's axis and frequency.
	pairs, err := c.Config().RopePairs()
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	m.Rope = nn.Rope{NRot: cfg.NRot, Base: cfg.RopeBase, Neox: cfg.RopeNeox,
		Scale: cfg.AttnFactor, Runs: nn.PairRuns(pairs)}
	if len(m.Rope.Runs) > 0 {
		cfg.RopePairs = m.Rope.TablePairs()
	}
	// The model's rotary divides each pair's angle by the factor the
	// container states for it (rope_freqs, or LongRoPE's short factors), the
	// long factors an instance past the original context turns (RopeLong):
	// the converter states a scaling method's divisors in them
	// (convert.StateRotary). A rotary of the blocks' own is the one their
	// stated rotary ops turn by (statedRotaries).
	for _, role := range []jlm.Role{jlm.RoleRopeFreqs, jlm.RoleRopeFactorsShort} {
		if c.Has(role, jlm.DenseBlock, -1) {
			if m.Rope.Freqs, err = vec(role, jlm.DenseBlock); err != nil {
				return nil, err
			}
			break
		}
	}
	if c.Has(jlm.RoleRopeFactorsLong, jlm.DenseBlock, -1) {
		long := m.Rope
		if long.Freqs, err = vec(jlm.RoleRopeFactorsLong, jlm.DenseBlock); err != nil {
			return nil, err
		}
		m.RopeLong, m.LongCtx = &long, cfg.YarnOrigCtx
		if m.LongCtx == 0 {
			m.LongCtx = cfg.NCtx
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
			c.RowTable(e) || denseNamed(e.Role) {
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
		nvis = int(v.NLayer) + audioBlocks(v)
	}
	m.Layers = make([]Layer, cfg.NLayer+cfg.NMTP, max(cfg.NLayer+cfg.NMTP, int(c.H.NBlocks))+nvis)
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
	// bind names one tensor a layer reads by name.
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
		// A block binds what it stores, each matrix under its role: its
		// stated graph names what it reads, and Open holds the graph to the
		// store (graph.Verify, graph.Unbound). Named here is what code reads
		// by name: the attention four and the FFN triple, a mixture's three
		// banks in the triple's place.
		ffn := []bind{{jlm.RoleFFNGate, &l.Gate}, {jlm.RoleFFNUp, &l.Up}, {jlm.RoleFFNDown, &l.Down}}
		if cfg.MoEAt(i) {
			// Every bank is stacked at conversion (convert.stackGGUFExperts, a
			// bank of mixed types as F32): experts one tensor each are a
			// container written before that.
			if !c.Has(jlm.RoleExpGateBank, bi, -1) && c.Has(jlm.RoleExpGate, bi, 0) {
				return nil, fmt.Errorf("model: block %d stores its experts one tensor each, as converters "+
					"before banks of mixed types did: re-convert the model (jitllm convert)", bi)
			}
			ffn = []bind{{jlm.RoleExpGateBank, &l.Gate}, {jlm.RoleExpUpBank, &l.Up}, {jlm.RoleExpDownBank, &l.Down}}
		}
		named := append([]bind{{jlm.RoleAttnQ, &l.Wq}, {jlm.RoleAttnK, &l.Wk}, {jlm.RoleAttnV, &l.Wv},
			{jlm.RoleAttnOut, &l.Wo}}, ffn...)
		for _, b := range named {
			if !c.Has(b.role, bi, -1) {
				continue
			}
			if err = getTo(b.dst, b.role, bi, -1); err != nil {
				return nil, err
			}
		}
		// What the block stores says how it runs: no FFN at all, experts
		// with no gate bank, v as k's projection under v's own norm.
		l.NoFFN = !cfg.MoEAt(i) && !c.Has(jlm.RoleFFNUp, bi, -1) && !c.Has(jlm.RoleFFNDown, bi, -1)
		l.UngatedExp = cfg.MoEAt(i) && !c.Has(jlm.RoleExpGateBank, bi, -1) && c.Has(jlm.RoleExpUpBank, bi, -1)
		l.VFromK = cfg.VNorm && !c.Has(jlm.RoleAttnV, bi, -1) && c.Has(jlm.RoleAttnK, bi, -1)
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
			if !ok || e.NDim != 2 || len(rv.V) == 0 || l.Matrix(rv.Role) != nil {
				continue
			}
			t := F32Tensor(rv.V, e.Rows(), e.K())
			l.Mats = append(l.Mats, roleTensor{rv.Role, &t})
		}
		if cfg.QKNormAt(i) {
			qn, err := vec(jlm.RoleAttnQNorm, bi)
			if err != nil {
				return nil, err
			}
			// A KV-sharing block norms only q: it caches nothing.
			var kn []float32
			if !cfg.KVShared(i) {
				if kn, err = vec(jlm.RoleAttnKNorm, bi); err != nil {
					return nil, err
				}
			}
			// A norm of the wrong width reads past its projection or
			// leaves part of it unnormed, fluently.
			_, _, wq, wk := cfg.QKNormShapeAt(i)
			if cfg.KVShared(i) {
				wk = 0
			}
			if len(qn) != wq || len(kn) != wk {
				return nil, fmt.Errorf("model: block %d's q/k norms are %d and %d wide, want %d and %d",
					i, len(qn), len(kn), wq, wk)
			}
		}
		if cfg.MoEAt(i) {
			if r := l.Matrix(jlm.RoleRouter); r.Rows != cfg.NExpert || r.K != cfg.NEmbd {
				return nil, fmt.Errorf("model: block %d router is %dx%d, want %dx%d",
					bi, r.K, r.Rows, cfg.NEmbd, cfg.NExpert)
			}
			l.Experts = make([]expert, cfg.NExpert)
			for e := range l.Experts {
				x := &l.Experts[e]
				if !l.UngatedExp {
					if x.Gate, err = bankExpert(l.Gate, jlm.RoleExpGateBank, cfg.NExpert, e); err != nil {
						return nil, err
					}
				}
				if x.Up, err = bankExpert(l.Up, jlm.RoleExpUpBank, cfg.NExpert, e); err != nil {
					return nil, err
				}
				// down's shape is the transpose of gate's: [n_ff_exp, width].
				if x.Down, err = bankExpert(l.Down, jlm.RoleExpDownBank, cfg.NExpert, e); err != nil {
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
		enorm, err := vec(jlm.RoleNextnENorm, bi)
		if err != nil {
			return err
		}
		hnorm, err := vec(jlm.RoleNextnHNorm, bi)
		if err != nil {
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
		}{{"enorm", len(enorm), cfg.NEmbd}, {"hnorm", len(hnorm), hw},
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
func bankExpert(b Tensor, role jlm.Role, nExpert, e int) (Tensor, error) {
	nrow := sheetRows(&b, nExpert)
	if nrow == 0 || b.Rows != nExpert*nrow {
		return Tensor{}, fmt.Errorf("model: %v is %d x %d, want %d experts of %d rows",
			role, b.K, b.Rows, nExpert, nrow)
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
		p := &nn.Packed{QS: qs, D: dd, SC: sc}
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
	// [sub-block][word][row]. A packed bank is cut by sheet in bankExpert.
	if t.Packed != nil {
		return Tensor{}, fmt.Errorf("model: %s is packed: its rows are not a byte range", t.name())
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
	for _, d := range m.DenseMats {
		see(d.T)
	}
	return rows, k
}

// blockReadBytes is how many file bytes of block li a single token reads:
// every matrix the block stores, a routed bank only for the experts a token
// picks. On a mixture it is far less than the block's size: a placement
// removes the read from the host but costs the size on the device.
func (m *Model) BlockReadBytes(li int) uint64 {
	c := m.Cfg
	var n uint64
	for _, mt := range m.Layers[li].Mats {
		b := TensorBytes(*mt.T)
		if jlm.ExpertBank(mt.Role) && c.NExpert > 0 {
			b = b / uint64(c.NExpert) * uint64(c.NExpertUsed)
		}
		n += b
	}
	return n
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

// actWindowOf is the model's activation window (composer.ActWindow) over the
// types of its matrices: for a model whose blocks quantize each row per
// token (quantizesRows), one window over its widest row, so each 32-wide
// block of a row carries the row's one scale.
func (m *Model) ActWindowOf() int {
	maxK := 0
	var types []quant.Type
	for i := range m.Container.Entries() {
		if t, ok := containerShape(&m.Container.Entries()[i]); ok {
			maxK = max(maxK, t.K)
			types = append(types, t.Typ)
		}
	}
	if m.quantizesRows() {
		return (maxK + 31) / 32 * 32
	}
	return composer.ActWindow(types)
}

// quantizesRows reports that a block's stated graph projects its rows' int8
// codes (graph.Graph.ReadsRowCodes), each graph the blocks share read once.
func (m *Model) quantizesRows() bool {
	of := m.Container.Config().Blocks.Of
	read := make([]bool, len(m.Container.Config().Blocks.Graphs))
	for li := range min(m.Cfg.NLayer, len(of)) {
		if read[of[li]] {
			continue
		}
		read[of[li]] = true
		if g, _, _ := m.blockGraph(li); g.ReadsRowCodes() {
			return true
		}
	}
	return false
}

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

// sheetRows is the rows each of a bank's n experts holds: its stored
// sheets' (the bank's second dimension, what the block's graph binds an
// expert at), b's rows over n for a bank stored as one matrix.
func sheetRows(b *Tensor, n int) int {
	if b.E != nil && b.E.NDim >= 3 {
		return int(b.E.Dims[1])
	}
	if n > 0 {
		return b.Rows / n
	}
	return 0
}

// BidirAt reports that a picture's rows see each other in layer li's
// attention.
func (c *Config) BidirAt(li int) bool {
	if h := c.at(li); h != c {
		return h.BidirAt(li)
	}
	return len(c.bidir) == 0 || c.bidir[li]
}
