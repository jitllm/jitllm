package engine

import (
	"math"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
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
	// tiled pairs value head vh with key head vh % kHeads instead of vh / rep:
	// the order a qwen35 GGUF holds the value heads in. See
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

func deltaOf(m *model.Model) deltaGeom {
	c := m.Cfg
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
	g.Tiled = keyTiled(m)
	g.Ssd = hasKind(c, jlm.LayerSSD, jlm.LayerSSDAttn)
	g.tail = c.RecTailLen()
	if hasKind(c, jlm.LayerScan) {
		g.Scan, g.Chans, g.QkBytes = true, int(c.SSM.Inner), 0
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

// hasKind reports whether some block of c is of one of kinds.
func hasKind(c *model.Config, kinds ...jlm.LayerKind) bool {
	for li := range c.NLayer {
		if slices.Contains(kinds, c.LayerKind(li)) {
			return true
		}
	}
	return false
}

// chanDecay reports whether m's recurrent blocks decay per channel: they
// store a forget gate's low-rank pair (RoleSSMFB).
func chanDecay(m *model.Model) bool {
	for i := range m.Layers {
		if m.Layers[i].Matrix(jlm.RoleSSMFB) != nil {
			return true
		}
	}
	return false
}

// dtRank is the dt rows of the x projection m's blocks store, 0 for none.
func dtRank(m *model.Model) int {
	for i := range m.Layers {
		if mt := m.Layers[i].Matrix(jlm.RoleSSMXDt); mt != nil {
			return mt.Rows
		}
	}
	return 0
}

// keyTiled reports whether m's container pairs value head vh with key head
// vh % key heads (jlm.FlagDeltaKeyTiled).
func keyTiled(m *model.Model) bool { return m.Container.Config().Flags.Has(jlm.FlagDeltaKeyTiled) }
