package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
)

// f16At is where each format keeps its super-block float16 scales, from
// pack.go's own reader. A differential test does not need to decode the
// payload -- both kernels decode identical bytes identically -- but it does
// need the scales to be finite, because random bits hit f16's Inf/NaN exponent
// about one time in 32 and a NaN compares unequal to itself.
var f16At = map[layout.Quant][]int{
	layout.Q4_0: {0},
	layout.Q8_0: {0},
	layout.Q5_0: {0},
	layout.Q5_1: {0, 2}, // d, m
	layout.Q4_K: {0, 2},
	layout.Q5_K: {0, 2},
	layout.Q3_K: {108},
	layout.Q6_K: {208},
	layout.TQ2:  {64},
}

// f32At is where a format with an f32 scale keeps it (TQ2F).
var f32At = map[layout.Quant][]int{layout.TQ2F: {64}}

func randWeights(q layout.Quant, rows, k int, rng *rand.Rand) []byte {
	nblk := k / q.Elems()
	bb := q.BlockBytes()
	raw := make([]byte, rows*nblk*bb)
	if layout.IsFloat(q) { // random bits are NaNs and infinities
		exact := make([][]float64, rows)
		for r := range exact {
			exact[r] = make([]float64, k)
		}
		floatWeights(q, raw, exact, rows, k, rng)
		return raw
	}
	rng.Read(raw)
	for i := 0; i < rows*nblk; i++ {
		if q == layout.MXFP4 {
			// An E8M0 exponent the scale plane holds exactly, or the packer
			// refuses the tensor (random bytes reach 2^127).
			raw[i*bb] = byte(120 + rng.Intn(10))
		}
		if offs, ok := f32At[q]; ok {
			// An f32 scale near one, of no f16's precision.
			for _, off := range offs {
				binary.LittleEndian.PutUint32(raw[i*bb+off:], math.Float32bits(0.5+rng.Float32()))
			}
			continue
		}
		offs, ok := f16At[q]
		if !ok && q != layout.MXFP4 {
			// A format with no entry would keep random scales -- Inf or NaN
			// one time in 32 -- and a NaN compares unequal in both arms.
			panic("randWeights: no scale offsets for " + q.String())
		}
		for _, off := range offs {
			// exponent near 1.0, so products stay well inside float32
			binary.LittleEndian.PutUint16(raw[i*bb+off:], uint16(0x2000|rng.Intn(0x0C00)))
		}
	}
	return raw
}

func randActs(k int, rng *rand.Rand) []float32 {
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

// gpu is a scratch device context: allocate, launch, read back, free.
type scratchDev struct {
	d    backend.Device
	bufs []backend.Buf
}

func newGPU(t *testing.T, d backend.Device) *scratchDev { return &scratchDev{d: d} }

func (g *scratchDev) up(p []byte) backend.Buf {
	if len(p) == 0 { // a float format's empty scale plane
		p = make([]byte, 4)
	}
	b, err := g.d.Alloc(len(p))
	if err != nil {
		panic(err)
	}
	if err := b.Write(p); err != nil {
		panic(err)
	}
	g.bufs = append(g.bufs, b)
	return b
}

func (g *scratchDev) free() {
	for _, b := range g.bufs {
		b.Free()
	}
}

// nanFill is the poison RULE 13 asks for: a buffer nothing has written must be
// distinguishable from one written with zeros, and a fresh device allocation is
// usually already zero.
func nanFill(n int) []float32 {
	v := make([]float32, n)
	nan := float32(math.NaN())
	for i := range v {
		v[i] = nan
	}
	return v
}

func u32bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func f32bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// floatWeights fills raw with row-major float weights of format q and exact
// with the values they hold after rounding to it.
func floatWeights(q layout.Quant, raw []byte, exact [][]float64, nrows, k int, rng *rand.Rand) {
	for r := 0; r < nrows; r++ {
		for i := 0; i < k; i++ {
			v, o := float32(rng.NormFloat64()), r*k+i
			switch q {
			case layout.Float32:
				binary.LittleEndian.PutUint32(raw[4*o:], math.Float32bits(v))
			case layout.Float16:
				h := quant.EncodeHalf(v)
				binary.LittleEndian.PutUint16(raw[2*o:], h)
				v = layout.F16(h)
			default:
				h := uint16(math.Float32bits(v) >> 16)
				binary.LittleEndian.PutUint16(raw[2*o:], h)
				v = math.Float32frombits(uint32(h) << 16)
			}
			exact[r][i] = float64(v)
		}
	}
}
