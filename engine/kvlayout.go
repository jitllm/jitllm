package engine

import (
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// kvLayout is where a (sequence slot, position, kv head) lives in the host KV
// cache, so the layout is one decision shared by every attention path, the
// cache write and the migrations: [slot][position][kv head][dim], a kernel
// advancing by Stride (kvDim) and reading HeadDim elements per step.
type kvLayout struct {
	maxSeq  int
	nKV     int
	headDim int
	// fmt is the cached elements' format. The cache stays a []float32 and
	// holds packed halves (f16) or q8_0 rows (nn.KVFmt) in its slots; every
	// offset handed out is a whole number of head rows, and a row of any
	// format fills whole float32 slots (nn.KVRowBytes).
	fmt nn.KVFmt
}

// slots is how many float32 slots a run of n elements occupies. n is always a
// whole number of head rows (positions times heads times headDim): a q8 row is
// the quantization unit and has no element-sized slot.
func (l kvLayout) slots(n int) int {
	switch l.fmt {
	case nn.KVF16:
		return n * 2 / 4
	case nn.KVQ8:
		return n / l.headDim * nn.KVRowBytes(nn.KVQ8, l.headDim) / 4
	}
	return n
}

// elemTag is the format as the prefix cache's geometry digest spells it: the
// element bytes it always printed for f32 and f16, so their stored pages keep
// their keys, and the format's name for a q8 cache.
func (l kvLayout) elemTag() string {
	switch l.fmt {
	case nn.KVF16:
		return "2"
	case nn.KVQ8:
		return l.fmt.String()
	}
	return "4"
}

func (l kvLayout) kvDim() int { return l.nKV * l.headDim }

// Stride is the element distance between one head's consecutive positions, and
// is exactly what the generated kernels bake as kvStride.
func (l kvLayout) Stride() int { return l.kvDim() }

// Base is the element offset of (slot, kvHead) at position zero. A kernel is
// handed cache[Base(...):] and advances by Stride.
func (l kvLayout) Base(slot, kvHead int) int {
	return l.slots(slot*l.maxSeq*l.kvDim() + kvHead*l.headDim)
}

// At is one (slot, position, kvHead) vector, for the scalar fallbacks.
func (l kvLayout) At(slot, pos, kvHead int) int {
	return l.Base(slot, kvHead) + l.slots(pos*l.Stride())
}

// kvlAt is layer li's layout: the instance's own at the layer's geometry, which
// differs from kvl only where a block attends at its own head width and kv
// heads (Config.MixedGeom). MLA's one cached row is the same on every layer.
func (s *Instance) kvlAt(li int) kvLayout { return s.kvl.atLayer(s.c, li) }

// atLayer is l at layer li's geometry; kvlAt has the cases. The page sizes
// (newKVCacheRange) read it too, since a q8 row's bytes depend on the layer's
// head width.
func (l kvLayout) atLayer(c *model.Config, li int) kvLayout {
	switch {
	case c.KeyHeadAt(li):
		l.nKV++
	case c.MixedGeom() && !c.MLA():
		l.nKV, l.headDim = c.NKVHeadAt(li), c.HeadDimAt(li)
	}
	return l
}
