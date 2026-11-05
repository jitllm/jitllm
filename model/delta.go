package model

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// The gated delta net: the other half of a hybrid's block loop.
//
// The graph is llama.cpp's running graph (layer 0 of Qwen3-Next-80B-A3B under
// `llama-eval-callback`), and six of its steps are ones a from-paper reading
// gets wrong:
//
//	(1) SiLU on the convolution's output, before it is split
//	(2) q and k L2-normalised: no weight, divided by the vector's own length
//	(3) 16 key heads and 32 value heads, interleaved: value head h reads h/2
//	(4) the decay softplus(a + dt_bias) * ssm_a, ssm_a already -exp(A_log)
//	(5) beta a plain sigmoid, no bias
//	(6) the output gate SiLU(z), where the full-attention layers use sigmoid
//
// The state is not a KV cache: it is the same size at every position, so
// residency, paging and placement cannot treat it as KV.
//
// The node-by-node transcript: docs/engineering-history/model-correctness.md,
// "model/delta.go".

// kdaL2Eps is the epsilon Kimi Delta Attention's q/k L2 normalisation adds to
// the sum of squares. It is the reference helper's own default, not the
// model's rms_norm_eps (1e-5 on the checkpoints shipped so far).
const kdaL2Eps = 1e-6

// deltaGeom is the recurrent block's derived geometry. Every field is a
// function of Config.SSM; they are computed once per call rather than stored so
// that a Config stays a description of the file.
type DeltaGeom struct {
	Conv    int // convolution width, in positions
	Chans   int // convolved channels: q + k + v
	KHeads  int // key/query heads
	VHeads  int // value heads, and the number of decay rates
	KDim    int // key/query head width, which is also the state's key extent
	VDim    int // value head width, which is also the state's value extent
	Rep     int // value heads per key head
	QkBytes int // q and k each occupy kHeads*kDim of the convolved channels
	// chanDecay is Kimi Delta Attention rather than qwen3next's gated delta
	// rule: the decay is a vector per value head instead of one rate, and both
	// gates come off a rank-kDim bottleneck (f_a_proj and g_a_proj project to
	// linear_head_dim, which is kDim).
	chanDecay bool
	Rank      int
	// tiled pairs value head vh with key head vh % kHeads instead of vh / rep:
	// the order llama.cpp's qwen35 converter writes the value heads in. See
	// keyHead.
	Tiled bool
	// ssd is Mamba-2's selective update in place of the delta rule (see
	// ssdAttn): the same state, k is B, q is C and v is x, and the convolved
	// channels arrive in that order because the converter put them there.
	Ssd bool
	// scale is 1/sqrt(kDim), which the L2 norm's rescale and the delta rule's
	// query scale both want.
	scale float32
	// mamba1 is Mamba-1's selective scan (mamba1Attn): every channel a value
	// head of one row (vHeads is Inner, vDim one), B and C one key head
	// shared by all, and only x convolved -- B and C come off it afterwards,
	// so chans is Inner. rank is its dt bottleneck.
	Mamba1 bool
	// tail is the recurrent state's tail per sequence (Config.recTailLen).
	tail int
}

func (c *Config) Delta() DeltaGeom {
	g := DeltaGeom{
		Conv:   int(c.SSM.ConvKernel),
		KHeads: int(c.SSM.Groups),
		VHeads: int(c.SSM.NHeadV),
		KDim:   int(c.SSM.StateSize),
	}
	g.QkBytes = g.KHeads * g.KDim
	g.Chans = int(c.SSM.Inner) + 2*g.QkBytes
	if g.VHeads > 0 {
		g.VDim = int(c.SSM.Inner) / g.VHeads
	}
	if g.KHeads > 0 {
		g.Rep = g.VHeads / g.KHeads
	}
	if g.KDim > 0 {
		g.scale = 1 / float32(math.Sqrt(float64(g.KDim)))
	}
	g.chanDecay, g.Rank = c.ChanDecay(), g.KDim
	g.Tiled = c.DeltaKeyTiled
	g.Ssd = c.SSD()
	g.tail = c.RecTailLen()
	if c.Mamba1() {
		g.Mamba1, g.Chans, g.QkBytes, g.Rank = true, int(c.SSM.Inner), 0, c.DtRank
	}
	return g
}

// mixer is a recurrent block's mixer by its structure, as the front end
// states it.
type mixer uint8

const (
	mixNone      mixer = iota
	mixDelta           // the gated delta rule (jlm.LayerLinearAttn)
	mixSelective       // the additive update, one decay a head (jlm.LayerSSD, LayerSSDAttn)
	mixScan            // the additive update, one decay a state element of each channel (jlm.LayerMamba1)
	mixShortConv       // the gated short convolution (jlm.LayerShortConv)
)

// mixerAt is block li's recurrent mixer, mixNone for a block with none.
func (c *Config) mixerAt(li int) mixer {
	switch c.LayerKind(li) {
	case jlm.LayerLinearAttn:
		return mixDelta
	case jlm.LayerSSD, jlm.LayerSSDAttn:
		return mixSelective
	case jlm.LayerMamba1:
		return mixScan
	case jlm.LayerShortConv:
		return mixShortConv
	}
	return mixNone
}

// keyHead is the key/query head value head vh reads.
//
// Two pairings ship, and both are fluent when wrong. qwen3next's GGUF keeps
// HuggingFace's grouped order (value heads kh*rep .. kh*rep+rep-1 share key
// head kh); llama.cpp's qwen35 converter reorders to tiled order (value head
// vh shares key head vh % kHeads). They differ only when rep > 1.
func (g DeltaGeom) KeyHead(vh int) int {
	if g.Tiled {
		return vh % g.KHeads
	}
	return vh / g.Rep
}

// convStateLen and deltaStateLen are one sequence's recurrent state for one
// linear layer. Both are constant in the context length.
func (g DeltaGeom) ConvStateLen() int  { return (g.Conv - 1) * g.Chans }
func (g DeltaGeom) DeltaStateLen() int { return g.VHeads * g.VDim * g.KDim }

// stateStride is one sequence's whole recurrent state row: the delta rule's
// and the tail after it (Config.recTailLen).
func (g DeltaGeom) StateStride() int { return g.DeltaStateLen() + g.tail }

// recTailLen is what each linear block's recurrent state carries per
// sequence after the delta rule's: room for Qwen3.8-Flash-Next's n-gram
// predecessors and convolution history (ngram.go), which its n-gram block
// fills and every other leaves unread, nothing on any other model. Every
// linear block has the same row, [delta rule | tail], so a device keeps one
// slot width for all of them (nn.RecurrentPlan.TailLen).
func (c *Config) RecTailLen() int {
	if c.NGramBlock != 0 {
		return c.ngramStateLen()
	}
	return 0
}
