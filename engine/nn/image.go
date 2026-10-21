package nn

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The image path on generated code (jit_old/cpu/imgproc_const.go has every
// kernel's contract): the decoded picture's rows into the processors'
// layouts, the antialiased resize's two passes, the normalisation's table
// lookup, the strided copies of the patch gather and the bilinear path, and
// the position tables built per grid -- a resampled table's taps and weights,
// MiniCPM-V's sin/cos. What stays in Go is geometry: windows, offsets and
// filter taps from the picture's sizes, and the tables of the tower's
// constants, built once and read here.
//
// Every entry point checks its operands' extents before the kernel reads or
// writes them, and panics on a short one: a programming error, which is the
// only way these can fail.

// PixFmt and PixOut are PixelRowJIT's source format and output, re-exported
// so engine/model need not import the assembler to name one.
type (
	PixFmt     = cpu.PixFmt
	PixOut     = cpu.PixOut
	AxisFilter = cpu.AxisFilter
)

const (
	PixRGBA   = cpu.PixRGBA
	PixNRGBA  = cpu.PixNRGBA
	PixGray   = cpu.PixGray
	PixRGBA64 = cpu.PixRGBA64
	PixYCbCr  = cpu.PixYCbCr
	PixOver8  = cpu.PixOver8
	PixPlanes = cpu.PixPlanes

	AxisBilinear = cpu.AxisBilinear
	AxisLinear   = cpu.AxisLinear
	AxisCubic    = cpu.AxisCubic
)

// AxisTapCount is the taps per output of AxisTapsJIT's filter.
func AxisTapCount(f AxisFilter) int { return cpu.AxisTapCount(f) }

// imgOp names one image kernel; with its shape it keys the kernel cache.
type imgOp uint8

const (
	opResample imgOp = iota
	opResampleH
	opPixLUT
	opCopy32
	opPixelRow
	opAxisTaps
	opLerpGrid
	opSinCos
)

type imgKey struct {
	t       cpu.Tier
	op      imgOp
	a, b, c int
}

// imgKernels is every image kernel this process has emitted, per tier and
// shape. A picture uses a handful, so the map stays small; the lookup takes
// no allocation once a kernel is in it.
var imgKernels struct {
	mu   sync.Mutex
	code map[imgKey]*cpu.Code
}

func imgKernel(op imgOp, a, b, c int) *cpu.Code {
	k := imgKey{cpu.HostTier(), op, a, b, c}
	imgKernels.mu.Lock()
	defer imgKernels.mu.Unlock()
	if code, ok := imgKernels.code[k]; ok {
		return code
	}
	if imgKernels.code == nil {
		imgKernels.code = map[imgKey]*cpu.Code{}
	}
	em := cpu.EmittersFor(k.t)
	var code *cpu.Code
	switch op {
	case opResample:
		code = mustEmit(fmt.Sprintf("resample_p%d", a))(em.Resample(a))
	case opResampleH:
		code = mustEmit(fmt.Sprintf("resample_h_p%d", a))(em.ResampleH(a))
	case opPixLUT:
		code = mustEmit("pixlut")(em.PixLUT())
	case opCopy32:
		code = mustEmit("copy32")(em.Copy32())
	case opPixelRow:
		code = mustEmit(fmt.Sprintf("pixelrow_%d_%d_%d", a, b, c))(em.PixelRow(cpu.PixFmt(a), b, cpu.PixOut(c)))
	case opAxisTaps:
		code = mustEmit(fmt.Sprintf("axistaps_%d", a))(em.AxisTaps(cpu.AxisFilter(a)))
	case opLerpGrid:
		code = mustEmit("lerpgrid")(em.LerpGrid())
	case opSinCos:
		code = mustEmit("sincostab")(em.SinCosTab())
	}
	imgKernels.code[k] = code
	return code
}

// resampleConsts is ResampleConsts per precision, built once each.
var resampleConsts struct {
	mu sync.Mutex
	m  map[uint][]int32
}

func resampleScr(prec uint) *byte {
	resampleConsts.mu.Lock()
	defer resampleConsts.mu.Unlock()
	c, ok := resampleConsts.m[prec]
	if !ok {
		if resampleConsts.m == nil {
			resampleConsts.m = map[uint][]int32{}
		}
		c = cpu.ResampleConsts(int(prec))
		resampleConsts.m[prec] = c
	}
	return (*byte)(unsafe.Pointer(&c[0]))
}

var (
	pixelConsts = cpu.PixelConsts()
)

func checkPrec(op string, prec uint) {
	if prec < 1 || prec > 30 {
		panic(fmt.Sprintf("nn: %s: precision %d", op, prec))
	}
}

// ResampleRowJIT is one output row of the vertical resampling pass, on
// generated code:
//
//	out[x] = min(max((1<<(prec-1) + sum_j taps[j]*in[j*stride+x]) >> prec, 0), 255)
//
// in holds the taps' source rows stride bytes apart, each at least len(out)
// long.
func ResampleRowJIT(out, in []uint8, stride int, taps []int32, prec uint) {
	n := len(out)
	if n == 0 {
		return
	}
	checkPrec("ResampleRowJIT", prec)
	if len(taps) == 0 || stride < 0 {
		panic(fmt.Sprintf("nn: ResampleRowJIT: %d taps, stride %d", len(taps), stride))
	}
	if need := (len(taps)-1)*stride + n; len(in) < need {
		panic(fmt.Sprintf("nn: ResampleRowJIT: %d source samples, %d taps of %d at stride %d need %d",
			len(in), len(taps), n, stride, need))
	}
	imgKernel(opResample, int(prec), 0, 0).Call(&cpu.Args{
		Out:    (*float32)(unsafe.Pointer(&out[0])),
		AScale: (*float32)(unsafe.Pointer(&in[0])),
		W:      (*byte)(unsafe.Pointer(&taps[0])),
		Cols:   int64(len(taps)),
		RowStr: int64(stride),
		Scr:    resampleScr(prec),
		K:      int64(n / cpu.ElemLanes),
		Rows:   int64(n % cpu.ElemLanes),
	})
}

// ResampleHJIT is the horizontal resampling pass over rows rows of RGB
// pixels, a pixel at a time: output pixel x of row r is window x of the
// input row, w[x*cols:(x+1)*cols] at the bytes from lo[x] on, three channels
// each. Every window lies inside its row.
func ResampleHJIT(out []uint8, outStr int, in []uint8, inStr, inW, rows int, w []int32, lo []int64, cols int, prec uint) {
	ow := len(lo)
	if rows == 0 || ow == 0 {
		return
	}
	checkPrec("ResampleHJIT", prec)
	switch {
	case cols < 1 || len(w) < ow*cols || cols > inW:
		panic(fmt.Sprintf("nn: ResampleHJIT: %d weights for %d windows of %d over %d pixels", len(w), ow, cols, inW))
	case len(in) < (rows-1)*inStr+3*inW || len(out) < (rows-1)*outStr+3*ow:
		panic(fmt.Sprintf("nn: ResampleHJIT: %d rows of %d -> %d pixels in %d and %d bytes", rows, inW, ow,
			len(in), len(out)))
	}
	for _, l := range lo {
		if l < 0 || int(l)+3*cols > 3*inW {
			panic(fmt.Sprintf("nn: ResampleHJIT: a window at byte %d runs past a %d-pixel row", l, inW))
		}
	}
	imgKernel(opResampleH, int(prec), 0, 0).Call(&cpu.Args{
		Out:     (*float32)(unsafe.Pointer(&out[0])),
		AScale:  (*float32)(unsafe.Pointer(&in[0])),
		W:       (*byte)(unsafe.Pointer(&w[0])),
		AScale2: (*float32)(unsafe.Pointer(&lo[0])),
		Cols:    int64(cols),
		K:       int64(ow),
		Rows:    int64(rows),
		RowStr:  int64(inStr),
		OutStr:  int64(outStr),
		Scr:     resampleScr(prec),
	})
}

// PixLUTJIT writes each sample's value from its channel's table: dst[i] =
// lut[i%3][src[i]].
func PixLUTJIT(dst []float32, src []uint8, lut *[3][256]float32) {
	n := len(src)
	if n == 0 {
		return
	}
	if len(dst) < n {
		lengthPanic("pixlut", n, len(dst))
	}
	imgKernel(opPixLUT, 0, 0, 0).Call(&cpu.Args{
		Out:    &dst[0],
		W:      &src[0],
		AScale: &lut[0][0],
		K:      int64(n / 3),
		Rows:   int64(n % 3),
	})
}

// Copy32JIT copies rows x n float32 with strides in elements:
//
//	dst[r*dRow + i*dEl] = src[r*sRow + i*sEl]
func Copy32JIT(dst []float32, dRow, dEl int, src []float32, sRow, sEl, rows, n int) {
	if rows == 0 || n == 0 {
		return
	}
	last := func(row, el int) int { return (rows-1)*row + (n-1)*el }
	if dRow < 0 || dEl < 0 || sRow < 0 || sEl < 0 || last(dRow, dEl) >= len(dst) || last(sRow, sEl) >= len(src) {
		panic(fmt.Sprintf("nn: Copy32JIT: %dx%d at strides (%d,%d) <- (%d,%d) over %d <- %d elements",
			rows, n, dRow, dEl, sRow, sEl, len(dst), len(src)))
	}
	imgKernel(opCopy32, 0, 0, 0).Call(&cpu.Args{
		Out:    &dst[0],
		AScale: &src[0],
		Rows:   int64(rows),
		K:      int64(n),
		RowStr: int64(4 * sRow),
		Cols:   int64(4 * sEl),
		OutStr: int64(4 * dRow),
		DStr:   int64(4 * dEl),
	})
}

// PixelRowJIT converts one row of n pixels of a decoded picture
// (jit_old/cpu/imgproc_const.go's PixelRow): src is the row in format f (the luma
// for PixYCbCr, whose cb and cr rows start at the chroma sample of the row's
// first x, x0, sub its horizontal chroma shift). PixOver8 writes 3n bytes to
// out8; PixPlanes writes n floats to each of planes.
func PixelRowJIT(f PixFmt, sub int, o PixOut, out8 []uint8, planes *[3][]float32, src, cb, cr []uint8, x0, n int) {
	if n == 0 {
		return
	}
	per := 1
	switch f {
	case PixRGBA, PixNRGBA:
		per = 4
	case PixRGBA64:
		per = 8
	}
	if len(src) < per*n {
		panic(fmt.Sprintf("nn: PixelRowJIT: %d source bytes for %d pixels", len(src), n))
	}
	args := cpu.Args{
		W:    &src[0],
		K:    int64(n),
		Cols: int64(x0),
		Scr:  (*byte)(unsafe.Pointer(&pixelConsts[0])),
	}
	if f == PixYCbCr {
		if x0 < 0 {
			panic(fmt.Sprintf("nn: PixelRowJIT: a chroma row at x %d", x0))
		}
		if need := (x0+n-1)>>sub - x0>>sub + 1; len(cb) < need || len(cr) < need {
			panic(fmt.Sprintf("nn: PixelRowJIT: %d and %d chroma samples, %d needed", len(cb), len(cr), need))
		}
		args.AScale = (*float32)(unsafe.Pointer(&cb[0]))
		args.Q32 = (*float32)(unsafe.Pointer(&cr[0]))
	}
	switch o {
	case PixOver8:
		if len(out8) < 3*n {
			panic(fmt.Sprintf("nn: PixelRowJIT: %d output bytes for %d pixels", len(out8), n))
		}
		args.Out = (*float32)(unsafe.Pointer(&out8[0]))
	case PixPlanes:
		for _, p := range planes {
			if len(p) < n {
				panic(fmt.Sprintf("nn: PixelRowJIT: a %d-float plane for %d pixels", len(p), n))
			}
		}
		args.Out, args.Out2, args.AScale2 = &planes[0][0], &planes[1][0], &planes[2][0]
	}
	imgKernel(opPixelRow, int(f), sub, int(o)).Call(&args)
}

// AxisTapsJIT fills one axis of a resampled position table (AxisTaps) under
// filter f: for n outputs, taps planes of n weights in w and n table indices
// in idx, the source coordinate the nine steps of chain, the indices clamped
// to hi.
func AxisTapsJIT(f AxisFilter, w []float32, idx []int32, n int, chain [9]float32, hi int) {
	if n == 0 {
		return
	}
	taps := cpu.AxisTapCount(f)
	if len(w) < taps*n || len(idx) < taps*n {
		panic(fmt.Sprintf("nn: AxisTapsJIT: %d weights and %d indices for %d taps of %d", len(w), len(idx), taps, n))
	}
	scr := cpu.AxisConsts(chain, int32(hi))
	imgKernel(opAxisTaps, int(f), 0, 0).Call(&cpu.Args{
		Out:  &w[0],
		Out2: (*float32)(unsafe.Pointer(&idx[0])),
		K:    int64(n),
		Scr:  (*byte)(unsafe.Pointer(&scr[0])),
	})
}

// LerpGridJIT fills one tap pair's plane of a 2-D table's weights and
// indices: w[y*len(ww)+x] = hw[y]*ww[x] and idx[y*len(ww)+x] = ht[y]*side +
// wt[x].
func LerpGridJIT(w []float32, idx []int32, hw []float32, ht []int32, ww []float32, wt []int32, side int) {
	gh, gw := len(hw), len(ww)
	if gh == 0 || gw == 0 {
		return
	}
	if len(ht) < gh || len(wt) < gw || len(w) < gh*gw || len(idx) < gh*gw {
		panic(fmt.Sprintf("nn: LerpGridJIT: a %dx%d grid in %d weights and %d indices", gh, gw, len(w), len(idx)))
	}
	imgKernel(opLerpGrid, 0, 0, 0).Call(&cpu.Args{
		Out:     &w[0],
		Out2:    (*float32)(unsafe.Pointer(&idx[0])),
		AScale:  &ww[0],
		AScale2: (*float32)(unsafe.Pointer(&wt[0])),
		Q32:     &hw[0],
		Q2:      (*float32)(unsafe.Pointer(&ht[0])),
		K:       int64(gw),
		Rows:    int64(gh),
		Cols:    int64(side),
	})
}

// sinCosConsts is SinCosConsts per frequency base, built once each.
var sinCosConsts struct {
	mu sync.Mutex
	m  map[float64][]float64
}

func sinCosScr(base float64) *byte {
	sinCosConsts.mu.Lock()
	defer sinCosConsts.mu.Unlock()
	c, ok := sinCosConsts.m[base]
	if !ok {
		if sinCosConsts.m == nil {
			sinCosConsts.m = map[float64][]float64{}
		}
		c = cpu.SinCosConsts(base)
		sinCosConsts.m[base] = c
	}
	return (*byte)(unsafe.Pointer(&c[0]))
}

// SinCosTabJIT fills MiniCPM-V's resampler position table along one axis
// (SinCosTab): rows positions of k frequencies of base, sin[p*k+i] and
// cos[p*k+i].
func SinCosTabJIT(sin, cos []float32, k, rows int, base float64) {
	if k == 0 || rows == 0 {
		return
	}
	if len(sin) < k*rows || len(cos) < k*rows {
		panic(fmt.Sprintf("nn: SinCosTabJIT: %d and %d floats for %d positions of %d", len(sin), len(cos), rows, k))
	}
	imgKernel(opSinCos, 0, 0, 0).Call(&cpu.Args{
		Out:  &sin[0],
		Out2: &cos[0],
		K:    int64(k),
		Rows: int64(rows),
		Scr:  sinCosScr(base),
	})
}
