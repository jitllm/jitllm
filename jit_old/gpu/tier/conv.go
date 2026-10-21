package tier

import (
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// A convolutional tower's blocks on the device (nn.ConvPlan): Gemma 3n's
// MobileNet-V5. A block is a PROGRAM -- a list of launches over named
// buffers, built and compiled at admission, since a Session may not compile
// -- and ConvLayers runs the programs of a range in one submission, the
// activation staying on the card from block to block. The matrices are the
// block's ordinary slots (Up and Down, or Wq..Wo), so the pager pages a
// convolutional block exactly as it pages any other; the buffers between
// launches are one scratch the device's convolutional blocks share, sized
// for the widest of them.

// cbuf names a buffer a program launches against: the scratch's, the
// block's own vectors, and the block's input and output, which are the two
// activation buffers in turn.
type cbuf uint8

const (
	cX cbuf = iota // the block's input
	cY             // its output
	cT1
	cT2
	cE1
	cE2
	cPad
	cCol
	cQ
	cK
	cV
	cAO
	cSc
	cSc2
	cA  // the int8 activation, bytes
	cAX // its scales and sums
	// A FastConformer block's (ConvConformer): the call's row count (one
	// u32, written per call), the residual's three stages and a LayerNorm's
	// two partial planes.
	cTN
	cX1
	cX2
	cX3
	cLP
	cLV
	nConvScratch
)

const (
	cN1 cbuf = 32 + iota
	cN2
	cDwS
	cDwSN
	cDwM
	cDwMN
	cKD
	cKDN
	cVD
	cVDN
	cVecEnd
)

// A FastConformer block's vectors (nn.ConformerVecs.List order), from cC0.
const (
	cC0 cbuf = 64 + iota
	cCFF1NB
	cCFF1UB
	cCFF1DB
	cCAttN
	cCAttNB
	cCQB
	cCKB
	cCVB
	cCOB
	cCU
	cCV
	cCConvN
	cCConvNB
	cCGLUVB
	cCGLUGB
	cCDW
	cCBNS
	cCBNH
	cCPW2B
	cCFF2N
	cCFF2NB
	cCFF2UB
	cCFF2DB
	cCOutN
	cCOutNB
	cCPE
	cCEnd
)

// convStep is one launch. A matmul (mm set) is its batched twin over the
// block's weight in slot, reading the int8 activation quantized from src (and
// src itself where the twin or a float weight wants floats).
type convStep struct {
	k             backend.Kernel
	threads       int
	groups, width int // a launch of groups x width (the norms); 0 is la's 128
	bufs          []cbuf
	mm            *mv
	slot          int
	src, dst      cbuf
}

// convLayer is a convolutional block on the device.
type convLayer struct {
	p    nn.ConvPlan
	vecs [cVecEnd - cN1]backend.Buf
	// cvecs is a FastConformer block's vectors, from cC0.
	cvecs [cCEnd - cC0]backend.Buf
	// ts is the layer's weight slots (layer.tensors), taken once: the slice
	// is made on every call of tensors.
	ts   []**resident
	prog []convStep
	// graph is the block's verified graph where the plan carries one: its
	// launches are lowered from it (graphConvProgram) and dev is the
	// program they were recorded from. vecRole names the vectors by role.
	graph   *graph.Graph
	dev     *devProgram
	vecRole []vecAt
	// out is the block's output length in floats.
	out int
}

// convScratch is the buffers every convolutional block's program runs
// against; act holds the activation between blocks, act[cur] the current.
type convScratch struct {
	b    [nConvScratch]backend.Buf
	n    [nConvScratch]int // bytes
	act  [2]backend.Buf
	actN int
	cur  int
}

// shapeKern is the compiled kernel for key, built once per device: the one
// shape-keyed cache the convolution path and the program's generated ops share. nil is a
// shape the backend refused, and the reason is in LastErr.
func (g *devTier) shapeKern(key convKey, build func() (*ir.Kernel, error)) backend.Kernel {
	enc := []uint32{encTemplate, uint32(key.id), uint32(key.act), math.Float32bits(key.eps)}
	for _, a := range key.a {
		enc = append(enc, uint32(a))
	}
	k, _ := g.genKern(enc, build)
	return k
}

// same is TF SAME's padding along one side: the output length and the zeros
// before the first sample (the smaller half).
func same(n, k, s int) (out, before int) {
	out = (n + s - 1) / s
	return out, max((out-1)*s+k-n, 0) / 2
}

// convVecs is the block's vectors in cbuf order from cN1, with the lengths a
// plan wants (0 for one the block has none of).
func convVecs(p *nn.ConvPlan, w *nn.ConvWeights) ([cVecEnd - cN1][]float32, [cVecEnd - cN1]int) {
	have := [cVecEnd - cN1][]float32{w.Norm1, w.Norm2, w.DwStart, w.DwStartNorm, w.DwMid, w.DwMidNorm,
		w.KDown, w.KDownNorm, w.VDown, w.VDownNorm}
	var want [cVecEnd - cN1]int
	switch p.Kind {
	case nn.ConvEdge:
		want[0], want[1] = p.CExp, p.COut
	case nn.ConvInverted:
		want[0], want[1] = p.CExp, p.COut
		if p.KStart > 0 {
			want[2], want[3] = p.KStart*p.KStart*p.CIn, p.CIn
		}
		if p.KMid > 0 {
			want[4], want[5] = p.KMid*p.KMid*p.CExp, p.CExp
		}
	case nn.ConvAttention:
		want[0] = p.CIn
		if p.KVK > 0 {
			k := p.KVK * p.KVK * p.CIn
			want[6], want[7], want[8], want[9] = k, p.CIn, k, p.CIn
		}
	}
	return have, want
}

// convGrowth is what growing the scratch to sz costs.
func (g *devTier) convGrowth(sz [nConvScratch]int) uint64 {
	cs := &g.conv
	var n uint64
	for i, b := range sz {
		if cbuf(i) == cX {
			if b > cs.actN {
				n += 2 * uint64(b-cs.actN)
			}
			continue
		}
		if b > cs.n[i] {
			n += uint64(b - cs.n[i])
		}
	}
	return n
}

// convGrow grows the shared scratch to at least sz, every new buffer
// poisoned (RULE 13) and charged as scratch (regrow). Callers hold g.mu.
func (g *devTier) convGrow(sz [nConvScratch]int) bool {
	cs := &g.conv
	if sz[cX] > cs.actN {
		for i := range cs.act {
			n := cs.actN
			if !g.regrow(&cs.act[i], &n, sz[cX], 1, true, true) {
				return false
			}
		}
		cs.actN = sz[cX]
	}
	for i := range sz {
		if c := cbuf(i); c != cX && c != cY && sz[i] > 0 && !g.regrow(&cs.b[i], &cs.n[i], sz[i], 1, true, true) {
			return false
		}
	}
	return true
}

// convLeft is a convolutional block leaving the device; the last one takes
// the scratch with it. Callers hold g.mu.
func (g *devTier) convLeft(l *layer) {
	if l.conv == nil {
		return
	}
	l.conv = nil
	if g.convN--; g.convN > 0 {
		return
	}
	defer g.scratchWin().close()
	cs := &g.conv
	for _, b := range append(cs.b[:], cs.act[:]...) {
		if b != nil {
			b.Free()
		}
	}
	g.conv = convScratch{}
}

// convBufs is the block's own vectors, for auxBufs.
func (l *layer) convBufs() []backend.Buf {
	if l.conv == nil {
		return nil
	}
	return append(l.conv.vecs[:], l.conv.cvecs[:]...)
}

// convOut is block li's output length in floats, 0 where this device does
// not hold it as a convolutional block.
func (g *devTier) convOut(li int) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if l := g.layers[li]; l != nil && l.conv != nil {
		return l.conv.out
	}
	return 0
}

// convCallArgs is one convOnce call's arguments and its submission's state.
type convCallArgs struct {
	lo, hi      int
	first, last bool
	x, out      []float32
	tap         int
	tapOut      []float32
	rows        int // a conformer call's rows, 0 for another block's
	err         error
	fsrc, msrc  backend.Buf
	memoK       backend.Kernel
	lc          launcher
	tn          [4]byte
}

// ConvLayers is nn.ConvDevice's run of convolutional blocks on the card. The
// composer forms none yet, so the host runs them.
func (g *devTier) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	return false
}

// ConvLayers runs convolutional blocks [lo, hi) device by device, the
// activation crossing the host between devices.
func (g *GPU) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	g.convMu.Lock()
	defer g.convMu.Unlock()
	g.mu.Lock()
	rs, ok := g.runsInto(g.convRuns[:0], lo, hi)
	g.convRuns = rs
	g.mu.Unlock()
	if !ok || len(rs) == 0 {
		return false
	}
	in := x
	for i, r := range rs {
		dst := out
		if i < len(rs)-1 {
			n := r.dev.convOut(r.hi - 1)
			if n == 0 {
				return false
			}
			k := i % 2
			if cap(g.convTmp[k]) < n {
				g.convTmp[k] = make([]float32, n)
			}
			dst = g.convTmp[k][:n]
		}
		if !r.dev.ConvLayers(r.lo, r.hi, in, dst, tap, tapOut) {
			return false
		}
		in = dst
	}
	return true
}

// ConvLayers forwards to the tier (nn.ConvDevice).
func (s *gpuSession) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	defer s.leave(s.step())
	return s.g.ConvLayers(lo, hi, x, out, tap, tapOut)
}

// conformerVecLens is the lengths nn.ConformerVecs.List's vectors must be.
func conformerVecLens(p *nn.ConvPlan) [cCEnd - cC0]int {
	d, ff, r := p.CIn, p.CExp, p.HIn
	return [cCEnd - cC0]int{d, d, ff, d, d, d, d, d, d, d, d, d, d, d, d, d,
		p.KMid * d, d, d, d, d, d, ff, d, d, d, (2*r - 1) * d}
}
