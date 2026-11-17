package engine

import (
	"math"

	"github.com/jitllm/jitllm/model"
)

// deltaGeom is a recurrent block's geometry as the container's SSM
// fields and layer kinds give it: the oracle the gates hold the stated
// graphs' state ops to.
type deltaGeom struct {
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
	// Scan is the per-element selective scan (jlm.LayerScan): every channel a value
	// head of one row (vHeads is Inner, vDim one), B and C one key head
	// shared by all, and only x convolved -- B and C come off it afterwards,
	// so chans is Inner. rank is its dt bottleneck.
	Scan bool
	// tail is the recurrent state's tail per sequence (Config.recTailLen).
	tail int
}

func deltaOf(c *model.Config) deltaGeom {
	g := deltaGeom{
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
	if c.Scan() {
		g.Scan, g.Chans, g.QkBytes, g.Rank = true, int(c.SSM.Inner), 0, c.DtRank
	}
	return g
}

// convStateLen and deltaStateLen are one sequence's recurrent state for one
// linear layer. Both are constant in the context length.
func (g deltaGeom) ConvStateLen() int  { return (g.Conv - 1) * g.Chans }
func (g deltaGeom) DeltaStateLen() int { return g.VHeads * g.VDim * g.KDim }

// stateStride is one sequence's whole recurrent state row: the delta rule's
// and the tail after it (Config.recTailLen).
func (g deltaGeom) StateStride() int { return g.DeltaStateLen() + g.tail }
