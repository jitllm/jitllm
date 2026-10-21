package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// kvLayout is where a (sequence slot, position, kv head) lives in the host KV
// cache, so the layout is one decision shared by every attention path, the
// cache write and the migrations. The attention kernels advance by Stride and
// read HeadDim elements per step; head-major makes that walk contiguous (see
// docs/engineering-history/cpu-kernels.md for the measurements). Both layouts
// occupy the same bytes and differ only in order.
//
//	row-major   [slot][position][kv head][dim]   Stride = kvDim
//	head-major  [slot][kv head][position][dim]   Stride = headDim
type kvLayout struct {
	headMajor bool
	maxSeq    int
	nKV       int
	headDim   int
	// fmt is the cached elements' format. The cache stays a []float32 and
	// holds packed halves (f16) or q8_0 rows (cpu.KVFmt) in its slots; every
	// offset handed out is a whole number of head rows, and a row of any
	// format fills whole float32 slots (cpu.KVRowBytes).
	fmt cpu.KVFmt
	// jit is the append's quantizer, set when fmt is q8 (State.kvl).
	jit *nn.JIT
	// q8Twin makes an f32 cache store every row rounded through q8_0 and
	// widened back: the values a q8 cache's kernels read, in the f32 cache's
	// format. It is the q8 gate's control arm (TestKVQ8IsTheF32CacheOfItsRows)
	// and nothing else sets it; jit is set with it.
	q8Twin []float32
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

// rowBytes is one head row's bytes at this layout's format.
func (l kvLayout) rowBytes() int { return cpu.KVRowBytes(l.fmt, l.headDim) }

func (l kvLayout) kvDim() int { return l.nKV * l.headDim }

// Stride is the element distance between one head's consecutive positions, and
// is exactly what the generated kernels bake as kvStride.
func (l kvLayout) Stride() int {
	if l.headMajor {
		return l.headDim
	}
	return l.kvDim()
}

// Base is the element offset of (slot, kvHead) at position zero. A kernel is
// handed cache[Base(...):] and advances by Stride, which is how both layouts
// reach the same kernel unchanged.
func (l kvLayout) Base(slot, kvHead int) int {
	s := slot * l.maxSeq * l.kvDim()
	if l.headMajor {
		s += kvHead * l.maxSeq * l.headDim
	} else {
		s += kvHead * l.headDim
	}
	return l.slots(s)
}

// At is one (slot, position, kvHead) vector, for the scalar fallbacks.
func (l kvLayout) At(slot, pos, kvHead int) int {
	return l.Base(slot, kvHead) + l.slots(pos*l.Stride())
}

// Write stores one position's whole kvDim-wide vector, narrowing to the
// cache's format.
//
// Row-major writes a position's heads in one store; head-major writes nKV
// separate ones, the layout's standing cost.
func (l kvLayout) Write(dst []float32, slot, pos int, src []float32) {
	if !l.headMajor {
		l.store(dst[l.slots(slot*l.maxSeq*l.kvDim()+pos*l.kvDim()):], src)
		return
	}
	for h := 0; h < l.nKV; h++ {
		l.store(dst[l.At(slot, pos, h):], src[h*l.headDim:(h+1)*l.headDim])
	}
}

// store copies one run of whole head rows at the cache's own format.
//
// At f16 this is the only place f32 becomes f16 on the host's KV path, so an
// f16 cache must match an f32 cache rounded through binary16 bit for bit: the
// generated narrowing (nn.NarrowF16) rounds as quant.EncodeHalf does.
// TestKVF16IsSelectedAndRuns holds the f16 cache end to end: that it is
// selected, and that its logits stay in the band rounding alone costs. At q8
// each head row goes through the generated quantizer (nn.QuantizeKVRows).
func (l kvLayout) store(dst []float32, src []float32) {
	switch l.fmt {
	case cpu.KVF16:
		nn.NarrowF16(unsafe.Slice((*uint16)(unsafe.Pointer(&dst[0])), len(src)), src)
	case cpu.KVQ8:
		l.jit.QuantizeKVRows(dst, src, l.headDim, len(src)/l.headDim)
	default:
		if l.q8Twin != nil {
			n := len(src) / l.headDim
			l.jit.QuantizeKVRows(l.q8Twin, src, l.headDim, n)
			l.jit.WidenKVRows(dst, l.q8Twin, l.headDim, n)
			return
		}
		narrow(dst, src)
	}
}

// kvlAt is layer li's layout: the State's own at the layer's geometry, which
// differs from kvl only where the sliding layers attend at their own head
// width and kv heads (Config.GeomSplit). MLA's one cached row is the same on
// every layer.
func (s *State) kvlAt(li int) kvLayout { return s.kvl.atLayer(s.c, li) }

// atLayer is l at layer li's geometry; kvlAt has the cases. The page sizes
// (newKVCacheRange) read it too, since a q8 row's bytes depend on the layer's
// head width.
func (l kvLayout) atLayer(c *Config, li int) kvLayout {
	switch {
	case c.extraKVHead(li):
		l.nKV++ // kvlMSA
	case c.DSV4():
		// One row a position, as wide as the block's (ds4.go).
		l.nKV, l.headDim = 1, c.KVRowAt(li)
	case c.GeomSplit() && !c.MLA() && c.SWA(li):
		l.nKV, l.headDim = c.NKVHeadSWA, c.HeadDimSWA // kvlSWA
	}
	return l
}

// kvlMSA is an MSA block's layout: the indexer's key is one more kv head
// after the attention's, so the row is (NKVHead+1)*HeadDim (msa.go).
func (s *State) kvlMSA() kvLayout {
	l := s.kvl
	l.nKV++
	return l
}

// kvlSWA is the sliding layers' layout.
func (s *State) kvlSWA() kvLayout {
	l := s.kvl
	l.nKV, l.headDim = s.c.NKVHeadSWA, s.c.HeadDimSWA
	return l
}

// provisionAttn picks the attention kernel sets after the cache width is
// decided: the JIT's default (the global layers' geometry, AddAttnKV) for
// every layer, and a second set baked at the sliding layers' head width and
// stride where the two geometries differ.
func (s *State) provisionAttn() {
	s.attnG = s.jit.Attn()
	s.attnL = s.attnG
	if c := s.c; c.GeomSplit() {
		s.attnL = s.jit.AttnSetFor(c.HeadDimSWA, c.HeadDimSWA, s.kvlSWA().Stride(), s.kvFmt)
	}
	// The indexer's scores read each cached row's IdxHeadDim tail at the row's
	// stride (indexer.go).
	if c := s.m.Cfg; c.Indexer() {
		s.idxAttn = s.jit.AttnSetFor(c.IdxHeadDim, c.IdxHeadDim, s.kvl.Stride(), s.kvFmt)
	}
	// An MSA block's row carries the indexer's key as one more head, so its
	// stride is a head wider; its indexer scores are this set's too, the key
	// being a head of the same width (msa.go).
	s.attnM = s.attnG
	if c := s.m.Cfg; c.MSA() || c.QSARatio != 0 {
		s.attnM = s.jit.AttnSetFor(c.HeadDim, c.HeadDim, s.kvlMSA().Stride(), s.kvFmt)
	}
}

// attnAt is the kernel set layer li attends with.
func (s *State) attnAt(li int) *nn.AttnSet {
	if s.c.extraKVHead(li) {
		return s.attnM
	}
	if s.c.SWA(li) {
		return s.attnL
	}
	return s.attnG
}
