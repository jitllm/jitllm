package cpu

import (
	"fmt"
	"math"
)

// The contracts of the image path's kernels, one copy for all three tiers
// (imgproc_sse.go is both x86 tiers' bodies, imgproc_a64.go NEON's). Every
// count is a runtime argument; nothing but a precision, a pixel format or a
// filter is baked.
//
// ResampleH -- the horizontal resampling pass, a pixel at a time (Pillow's
// ImagingResampleHorizontal_8bpc). For each row r < Rows, output pixel
// x < K and channel c < 3:
//
//	out[r*OutStr + 3x + c] = min(max((bias + sum_{j<Cols} w[x*Cols+j] *
//	                         in[r*RowStr + lo[x] + 3j + c]) >> prec, 0), 255)
//
//	Out     uint8 output rows        AScale   uint8 input rows
//	W       int32 w[K][Cols]         AScale2  int64 lo[K], each window's first byte
//	Cols    taps per window          K        output pixels
//	Rows    rows                     RowStr, OutStr  bytes between rows
//	Scr     ResampleConsts(prec)
//
// Every window is Cols taps inside its row: a shorter one is padded with zero
// weights and moved to fit (the sums are integers, so neither changes one).
// Exactly three bytes are read per tap and written per pixel.
//
// PixLUT -- a table per channel, read at each sample:
//
//	out[3p + c] = lut[256*c + in[3p + c]]     for the K whole pixels, then
//	out[3K + c] = lut[256*c + in[3K + c]]     for c < Rows (0, 1 or 2)
//
//	Out     float32 out              W        uint8 in
//	AScale  float32 lut[3][256]      K, Rows  pixels and the samples after them
//
// Copy32 -- a strided copy of 4-byte elements:
//
//	out[r*OutStr + i*DStr] = in[r*RowStr + i*Cols]    for r < Rows, i < K
//
// with every stride in bytes.
//
// PixelRow -- one row of a decoded picture, from the decoder's own layout to
// what the processors read, exactly as image.Image's At(x, y).RGBA() would
// have produced it pixel by pixel. Each pixel becomes its 16-bit
// alpha-premultiplied (r, g, b, a), from the format (PixFmt):
//
//	PixRGBA    *image.RGBA's bytes        c*257 (c|c<<8), a*257
//	PixNRGBA   *image.NRGBA's             c*257*a/255 (truncated), a*257
//	PixGray    *image.Gray's              y*257, 0xffff
//	PixRGBA64  *image.RGBA64's            each big-endian 16-bit sample
//	PixYCbCr   *image.YCbCr's planes      color.YCbCr.RGBA's fixed point:
//	           yy = y*0x10101, r = yy + 91881*(cr-128),
//	           g = yy - 22554*(cb-128) - 46802*(cr-128), b = yy + 116130*(cb-128),
//	           each clamped to 0..0xffffff and shifted down 8; a = 0xffff
//
// and is written (PixOut) as
//
//	PixOver8   three bytes, uint8((c + 0xffff - a) >> 8): composited over
//	           white, which is c>>8 for an opaque pixel
//	PixPlanes  three float32, float32(c), one to each of three planes
//
//	Out     Over8: the uint8 row;  Planes: the first plane's row (float32)
//	Out2, AScale2  Planes: the second and third planes' rows
//	W       the source row (PixYCbCr: the luma row)
//	AScale, Q32    PixYCbCr: the Cb and Cr rows
//	K       pixels
//	Cols    PixYCbCr: the row's first x, whose chroma sample is the rows'
//	        first: pixel i reads chroma ((Cols+i)>>sub) - (Cols>>sub)
//	Scr     PixelConsts()
//
// sub, baked, is the chroma's horizontal subsampling shift (0 for 4:4:4 and
// 4:4:0, 1 for 4:2:2 and 4:2:0, 2 for 4:1:1 and 4:1:0); the caller picks the
// chroma row.
//
// AxisTaps -- one axis of a resampled position table (Qwen3-VL's bilinear,
// GLM-4.xV's grid_sample bicubic, Kimi-VL's interpolate bicubic, HunyuanVL's
// interpolate bilinear): for each
// output index i < K the source coordinate, in float32 with one rounding per
// step,
//
//	x = ((((((((float32(i) + c0) * c1) / c2) * c3) + c4) + c5) * c6) + c7) / c8
//
// -- each reference's own sequence, the steps it does not take an exact
// identity (+0, *1, /1) -- then fl = floor(x), t = x - fl, and per tap k the
// weight and the table index min(max(int(fl) + k + off, 0), hi):
//
//	AxisBilinear  2 taps, off  0:  max(1 - |t|, 0), max(1 - |t - 1|, 0)
//	AxisLinear    2 taps, off  0:  x clamped below at 0 first; 1 - t, t
//	AxisCubic     4 taps, off -1:  conv2(t+1), conv1(t), conv1(1-t), conv2(2-t),
//	          conv1(x) = ((1.25x - 2.25)x)x + 1,
//	          conv2(x) = ((-0.75x + 3.75)x - 6)x + 3     (A = -0.75)
//
// every product and sum rounded on its own, as the references' float32 does.
//
//	Out   float32 weights [taps][K]     Out2  int32 indices [taps][K]
//	K     outputs                       Scr   AxisConsts(chain, hi)
//
// LerpGrid -- one tap pair's plane of a 2-D table's weights and indices, the
// outer product of two axes: for y < Rows and x < K,
//
//	Out[y*K + x] = hw[y] * ww[x]        Out2[y*K + x] = ht[y]*Cols + wt[x]
//
//	Q32  hw (float32 [Rows])   Q2  ht (int32 [Rows])
//	AScale  ww (float32 [K])   AScale2  wt (int32 [K])   Cols  the table's side

// pixLUTSize is the entries in one channel's PixLUT table: one per byte value.
const pixLUTSize = 256

// The AxisTaps constant block: float32 slots, int32 where named.
const (
	axisChainOff = 0  // c0..c8, the coordinate's nine steps
	axisHiOff    = 36 // hi, the last table index (int32)
	axisOneOff   = 40 // 1.0
	axisAbsOff   = 44 // 0x7fffffff, the absolute value's mask
	axisC1Off    = 48 // 1.25, 2.25: conv1's
	axisC2Off    = 56 // -0.75, 3.75, -6, 3: conv2's
	axisTwoOff   = 72 // 2.0
	axisFourOff  = 76 // 4.0, the index step of a vector
	axisIotaOff  = 80 // 0, 1, 2, 3: the first vector's indices
	axisOffsOff  = 96 // -1, 1, 2: the tap offsets added to floor (int32)
	axisConstLen = 112
)

// AxisBlock is the AxisTaps constant block, a value so a caller can keep it
// on its stack.
type AxisBlock [axisConstLen / 4]float32

// AxisConsts is the block for a coordinate's nine steps and a table's last
// index.
func AxisConsts(chain [9]float32, hi int32) (c AxisBlock) {
	copy(c[axisChainOff/4:], chain[:])
	c[axisHiOff/4] = math.Float32frombits(uint32(hi))
	c[axisOneOff/4] = 1
	c[axisAbsOff/4] = math.Float32frombits(0x7fffffff)
	fixed := [...]float32{1.25, 2.25, -0.75, 3.75, -6, 3, 2, 4, 0, 1, 2, 3}
	copy(c[axisC1Off/4:], fixed[:])
	for i, v := range [...]int32{-1, 1, 2} {
		c[axisOffsOff/4+i] = math.Float32frombits(uint32(v))
	}
	return c
}

// SinCosTab -- MiniCPM-V's resampler position table along one axis (the
// reference's get_1d_sincos_pos_embed_from_grid_new): for frequency i < K and
// position p < Rows,
//
//	om_i            = float32(base^(-float32(i)/float32(K)))   (the power in float64)
//	a               = float32(p) * om_i                         (float32)
//	Out[p*K + i]    = float32(sin(float64(a)))
//	Out2[p*K + i]   = float32(cos(float64(a)))
//
// two frequencies a register (float64 lanes), the rest one in lane 0. The
// float64 work is a Cody-Waite reduction by pi/2 in three parts, fdlibm's
// sin and cos kernels on |r| <= pi/4, and exp's degree-13 Taylor series on
// |r| <= ln2/2 after the reduction by ln2, every step its own rounding and
// none of it fused, so every tier computes the same float64 and rounds it the
// same way. Each float64 is within a few ulp of the true value, which lands
// every float32 on the correctly rounded one wherever the true value is
// further than that from a float32 rounding boundary -- at every table a real
// width builds (imgsincos_test.go measures the margin).
//
//	Out, Out2  float32 [Rows][K]      K  frequencies    Rows  positions
//	Scr        SinCosConsts(base)

// The SinCosTab constant block: float64 slots, float32 where named.
const (
	scLnHiOff    = 0   // ln(base), its high 28 bits
	scLnLoOff    = 8   // the rest of ln(base)
	scLog2eOff   = 16  // 1/ln2
	scLn2HiOff   = 24  // ln2, high part (fdlibm's ln2HI)
	scLn2LoOff   = 32  // ln2, low part
	scMagicOff   = 40  // 2^52 + 1023: 2^-k's exponent built in the mantissa
	scExpOff     = 48  // 1/13!, 1/12!, ..., 1/1!, 1/0!: e^r, highest first
	scExpTerms   = 14  //
	scTwoPiOff   = 160 // 2/pi
	scPio2Off    = 168 // pi/2 in three parts (fdlibm's pio2_1, pio2_2, pio2_2t)
	scSinOff     = 192 // S6, S5, ..., S1: fdlibm's __kernel_sin, highest first
	scCosOff     = 240 // C6, C5, ..., C1: fdlibm's __kernel_cos, highest first
	scHalfOff    = 288 // 0.5
	scQuarterOff = 296 // 0.25
	scOneOff     = 304 // 1.0
	scFourOff    = 312 // 4.0
	scOneF32Off  = 320 // 1.0 (float32)
	scTwoF32Off  = 324 // 2.0 (float32)
	scIotaF32Off = 328 // 0, 1 (float32): the first pair's frequencies
	scConstLen   = 336
	scLnLowBits  = 25 // the bits of ln(base) cleared for its high part
)

// SinCosConsts is the SinCosTab block for a frequency base, as float64 slots
// (the float32 ones packed two to a slot).
func SinCosConsts(base float64) []float64 {
	c := make([]float64, scConstLen/8)
	ln := math.Log(base)
	hi := math.Float64frombits(math.Float64bits(ln) &^ (1<<scLnLowBits - 1))
	c[scLnHiOff/8], c[scLnLoOff/8] = hi, ln-hi
	c[scLog2eOff/8] = 1 / math.Ln2
	c[scLn2HiOff/8] = 6.93147180369123816490e-01
	c[scLn2LoOff/8] = 1.90821492927058770002e-10
	c[scMagicOff/8] = 1<<52 + 1023
	copy(c[scExpOff/8:], []float64{1.0 / 6227020800, 1.0 / 479001600, 1.0 / 39916800, 1.0 / 3628800,
		1.0 / 362880, 1.0 / 40320, 1.0 / 5040, 1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, 1.0 / 2, 1, 1})
	c[scTwoPiOff/8] = 6.36619772367581382433e-01
	copy(c[scPio2Off/8:], []float64{1.57079632673412561417e+00, 6.07710050630396597660e-11, 2.02226624879595063154e-21})
	copy(c[scSinOff/8:], []float64{1.58969099521155010221e-10, -2.50507602534068634195e-08,
		2.75573137070700676789e-06, -1.98412698298579493134e-04, 8.33333333332248946124e-03,
		-1.66666666666666324348e-01})
	copy(c[scCosOff/8:], []float64{-1.13596475577881948265e-11, 2.08757232129817482790e-09,
		-2.75573143513906633035e-07, 2.48015872894767294178e-05, -1.38888888888741095749e-03,
		4.16666666666666019037e-02})
	c[scHalfOff/8], c[scQuarterOff/8], c[scOneOff/8], c[scFourOff/8] = 0.5, 0.25, 1, 4
	pack := func(lo, hi float32) float64 {
		return math.Float64frombits(uint64(math.Float32bits(lo)) | uint64(math.Float32bits(hi))<<32)
	}
	c[scOneF32Off/8] = pack(1, 2)
	c[scIotaF32Off/8] = pack(0, 1)
	return c
}

// AxisFilter is AxisTaps' filter.
type AxisFilter uint8

const (
	// AxisBilinear is transformers' aligned-corners bilinear (Qwen3-VL).
	AxisBilinear AxisFilter = iota
	// AxisLinear is F.interpolate's bilinear, align_corners False
	// (HunyuanVL): the coordinate clamped below at zero, weights 1-t and t.
	AxisLinear
	// AxisCubic is the cubic convolution at A = -0.75 (GLM-4.xV, Kimi-VL).
	AxisCubic
)

// AxisTapCount is the taps per output of AxisTaps' filter.
func AxisTapCount(f AxisFilter) int {
	if f == AxisCubic {
		return 4
	}
	return 2
}

// PixFmt is a decoded picture's pixel layout (PixelRow).
type PixFmt uint8

const (
	PixRGBA PixFmt = iota
	PixNRGBA
	PixGray
	PixRGBA64
	PixYCbCr
)

// PixOut is what PixelRow writes.
type PixOut uint8

const (
	PixOver8 PixOut = iota
	PixPlanes
)

// The PixelRow constant block, sixteen bytes a vector of four int32 (or
// float32) lanes.
const (
	pixK257Off   = 0   // 257 x4
	pixAlphaOff  = 16  // 0, 0, 0, 0xffff: the opaque alpha lane
	pixFFFFOff   = 32  // 0xffff x4
	pixFFOff     = 48  // 0xff x4
	pixF255Off   = 64  // 255.0 x4 (float32)
	pixBswapOff  = 80  // the bytes of four big-endian halfwords swapped
	pixK10101Off = 96  // 0x10101 x4
	pixKCbOff    = 112 // 0, -22554, 116130, 0
	pixKCrOff    = 128 // 91881, -46802, 0, 0
	pixK128Off   = 144 // 128 x4
	pixClampOff  = 160 // 0xffffff x4
	pixConstLen  = 176
)

// pixelShape refuses a PixelRow nothing calls for: an unknown format or
// output, or a chroma shift on a format with no chroma.
func pixelShape(f PixFmt, sub int, o PixOut) error {
	switch {
	case f > PixYCbCr || o > PixPlanes:
		return fmt.Errorf("jit: PixelRow: format %d, output %d", f, o)
	case sub < 0 || sub > 2 || (f != PixYCbCr && sub != 0):
		return fmt.Errorf("jit: PixelRow: chroma shift %d on format %d", sub, f)
	}
	return nil
}

// pixBytes is one pixel's bytes in format f's source row (the luma's for
// PixYCbCr).
func pixBytes(f PixFmt) int32 {
	switch f {
	case PixRGBA, PixNRGBA:
		return 4
	case PixRGBA64:
		return 8
	}
	return 1
}

// PixelConsts is the block, as 32-bit words.
func PixelConsts() []int32 {
	c := make([]int32, pixConstLen/4)
	splat := func(off int, v int32) {
		for i := 0; i < 4; i++ {
			c[off/4+i] = v
		}
	}
	splat(pixK257Off, 257)
	c[pixAlphaOff/4+3] = 0xffff
	splat(pixFFFFOff, 0xffff)
	splat(pixFFOff, 0xff)
	splat(pixF255Off, 0x437F0000) // 255.0
	// 0x80 in a PSHUFB index zeroes the byte.
	c[pixBswapOff/4+0] = 0x02030001
	c[pixBswapOff/4+1] = 0x06070405
	c[pixBswapOff/4+2] = -0x7F7F7F80 // 0x80808080
	c[pixBswapOff/4+3] = -0x7F7F7F80
	splat(pixK10101Off, 0x10101)
	copy(c[pixKCbOff/4:], []int32{0, -22554, 116130, 0})
	copy(c[pixKCrOff/4:], []int32{91881, -46802, 0, 0})
	splat(pixK128Off, 128)
	splat(pixClampOff, 0xffffff)
	return c
}
