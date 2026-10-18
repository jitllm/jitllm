package tier

import (
	"math"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// flashVariant is one compiled key partitioning of the decode attention: the
// kernel, its sliding-window twin, and the merge when splits > 1.
type flashVariant struct {
	splits, groups int
	k, kW, merge   backend.Kernel
}

// flashFor is the variant for a decode over n keys (the graph's nCap).
func (bs *blockScratch) flashFor(n int) *flashVariant {
	want := bs.flashSplitFor(n)
	for i := range bs.flashV {
		if bs.flashV[i].splits >= want {
			return &bs.flashV[i]
		}
	}
	return &bs.flashV[len(bs.flashV)-1]
}

// flashSplitFor is the partition count for n keys. A forced split (FlashSplit)
// is taken as given; otherwise it is the power of two nearest sqrt(n/8), capped
// at ~1.6 workgroups an SM. A partition's serial work falls as n/splits while
// the merge grows with splits, so the best count grows as a square root of n
// (measured with TestFlashAttentionAB). The floor is correctness:
// splits*FlashKVChunk must cover n.
func (bs *blockScratch) flashSplitFor(n int) int {
	s := bs.flashForced
	if s == 0 {
		s = 1 << int(math.Round(math.Log2(math.Sqrt(max(1, float64(n)/8)))))
		s = max(1, min(s, bs.flashCap))
	}
	return max(s, (n+bs.flashChunk-1)/bs.flashChunk)
}

// flashKV is whether decode attention runs as kernels.FlashDecodeKV: asked
// for, or the default on CUDA and Metal, where it measured faster than the
// staged kernels end to end (TestFlashAttentionAB at kernel level). Vulkan
// was not measured and stays staged.
func (g *devTier) flashKV() bool {
	api := g.dev.API()
	return g.FlashKV || ((api == "ptx" || api == "msl") && !g.NoFlashKV && !g.FlashAttention)
}

// flashOn is whether decode attention takes a fused kernel at all.
func (g *devTier) flashOn() bool { return g.FlashAttention || g.flashKV() }

// initFlash adds the optional decode schedule. Prefill keeps the staged query
// tiles: the initial fused schedule regresses long-context prefill on Metal.
func (g *devTier) initFlash(bs *blockScratch) (err error) {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	p := &bs.p
	if !g.flashOn() || bs.rows != 1 || p.MLA() || p.NonCausal || g.pickLanes() != 32 {
		return nil
	}
	shape := kernels.FlashShape{Heads: p.NHead, KVHeads: p.NKVHead, Dim: p.HeadDim, Rows: bs.rows,
		KStride: g.kStride(p), Scale: g.scoreScale(p), Softcap: p.AttnSoftcap, F16: g.KVF16, Sink: p.AttnSinks}
	build, width := kernels.FlashAttention, 32
	groups := func(s kernels.FlashShape) int { return p.NHead * max(1, s.Splits) }
	// FlashAttention holds no partition in shared memory: any split covers.
	bs.flashChunk, bs.flashForced, bs.flashCap = p.MaxSeq+1, max(1, g.FlashSplit), 1
	if g.flashKV() {
		if gqa := p.NHead / p.NKVHead; g.FlashGroup > 0 && gqa%g.FlashGroup == 0 {
			shape.Group = g.FlashGroup
		}
		shape.Warps = g.FlashWarps
		build, width, groups = kernels.FlashDecodeKV, kernels.FlashKVWidth(shape), kernels.FlashKVGroups
		shape.Splits = 1
		bs.flashChunk, bs.flashForced = kernels.FlashKVChunk(shape), g.FlashSplit
		slots := g.dev.Slots()
		if slots <= 0 {
			slots = 16384
		}
		bs.flashCap = max(1, slots/1280/max(1, kernels.FlashKVGroups(shape)))
	}
	bs.flashWidth = width
	// One variant per split any count up to MaxSeq selects.
	// flashSplitFor only grows with n, so walking every count collects them.
	var splits []int
	for n := 1; n <= max(1, p.MaxSeq); n++ {
		if sp := bs.flashSplitFor(n); len(splits) == 0 || sp > splits[len(splits)-1] {
			splits = append(splits, sp)
		}
	}
	// An unsupported compiler/layout leaves the complete staged schedule intact,
	// and releases every resource allocated for the unsuccessful fused schedule.
	defer func() {
		if err == nil {
			return
		}
		freeFlash(bs, g.devShared)
	}()
	compile := func(k *ir.Kernel, e error) (backend.Kernel, error) {
		if e != nil {
			return nil, e
		}
		return g.compileShared(k)
	}
	if top := splits[len(splits)-1]; top > 1 {
		shape.Splits = top
		n := kernels.FlashPartialFloats(shape) * 4
		bs.flashPart, err = g.dev.Alloc(n)
		if err != nil {
			return err
		}
		if g.kb.poisonScratch {
			if err = bs.flashPart.Write(nanFill(n)); err != nil {
				return err
			}
		}
	}
	for _, sp := range splits {
		shape.Splits = sp
		v := flashVariant{splits: sp, groups: groups(shape)}
		bs.flashV = append(bs.flashV, v)
		fv := &bs.flashV[len(bs.flashV)-1]
		if sp > 1 {
			if fv.merge, err = compile(kernels.FlashAttentionMerge(shape)); err != nil {
				return err
			}
		}
		if fv.k, err = compile(build(shape)); err != nil {
			return err
		}
		if p.SWAWindow > 0 {
			sw := shape
			sw.Window, sw.Chunked = p.SWAWindow, p.SWAChunked
			if fv.kW, err = compile(build(sw)); err != nil {
				return err
			}
		}
	}
	return nil
}

// freeFlash releases initFlash's kernels and partials; the scratch walk in
// freeScratch does not follow a slice.
func freeFlash(bs *blockScratch, g *devShared) {
	for _, v := range bs.flashV {
		for _, k := range []backend.Kernel{v.k, v.kW, v.merge} {
			if k != nil {
				g.closeKern(k)
			}
		}
	}
	bs.flashV = nil
	if bs.flashPart != nil {
		bs.flashPart.Free()
		bs.flashPart = nil
	}
}
