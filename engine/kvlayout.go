package engine

import (
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/model"
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
	// holds packed halves (f16) or q8_0 rows (cpu.KVFmt) in its slots; every
	// offset handed out is a whole number of head rows, and a row of any
	// format fills whole float32 slots (cpu.KVRowBytes).
	fmt cpu.KVFmt
}

// slots is how many float32 slots a run of n elements occupies. n is always a
// whole number of head rows (positions times heads times headDim): a q8 row is
// the quantization unit and has no element-sized slot.
func (l kvLayout) slots(n int) int {
	switch l.fmt {
	case cpu.KVF16:
		return n * 2 / 4
	case cpu.KVQ8:
		return n / l.headDim * cpu.KVRowBytes(cpu.KVQ8, l.headDim) / 4
	}
	return n
}

// elemTag is the format as the prefix cache's geometry digest spells it: the
// element bytes it always printed for f32 and f16, so their stored pages keep
// their keys, and the format's name for a q8 cache.
func (l kvLayout) elemTag() string {
	switch l.fmt {
	case cpu.KVF16:
		return "2"
	case cpu.KVQ8:
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
// differs from kvl only where the sliding layers attend at their own head
// width and kv heads (Config.GeomSplit). MLA's one cached row is the same on
// every layer.
func (s *Instance) kvlAt(li int) kvLayout { return s.kvl.atLayer(s.c, li) }

// atLayer is l at layer li's geometry; kvlAt has the cases. The page sizes
// (newKVCacheRange) read it too, since a q8 row's bytes depend on the layer's
// head width.
func (l kvLayout) atLayer(c *model.Config, li int) kvLayout {
	switch {
	case c.ExtraKVHead(li):
		l.nKV++
	case c.GeomSplit() && !c.MLA() && c.SWA(li):
		l.nKV, l.headDim = c.NKVHeadSWA, c.HeadDimSWA
	}
	return l
}
