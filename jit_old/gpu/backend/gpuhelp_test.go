package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// f16At is where each format keeps its super-block float16 scales, from
// pack.go's own reader. A differential test does not need to decode the
// payload -- both kernels decode identical bytes identically -- but it does
// need the scales to be finite, because random bits hit f16's Inf/NaN exponent
// about one time in 32 and a NaN compares unequal to itself.
var f16At = map[mainkernels.Quant][]int{
	mainkernels.Q4_0: {0},
	mainkernels.Q8_0: {0},
	mainkernels.Q5_0: {0},
	mainkernels.Q5_1: {0, 2}, // d, m
	mainkernels.Q4_K: {0, 2},
	mainkernels.Q5_K: {0, 2},
	mainkernels.Q3_K: {108},
	mainkernels.Q6_K: {208},
	mainkernels.TQ2:  {64},
}

// f32At is where a format with an f32 scale keeps it (TQ2F).
var f32At = map[mainkernels.Quant][]int{mainkernels.TQ2F: {64}}

func randWeights(q mainkernels.Quant, rows, k int, rng *rand.Rand) []byte {
	nblk := k / q.Elems()
	bb := q.BlockBytes()
	raw := make([]byte, rows*nblk*bb)
	if mainkernels.IsFloat(q) { // random bits are NaNs and infinities
		exact := make([][]float64, rows)
		for r := range exact {
			exact[r] = make([]float64, k)
		}
		floatWeights(q, raw, exact, rows, k, rng)
		return raw
	}
	rng.Read(raw)
	for i := 0; i < rows*nblk; i++ {
		if q == mainkernels.MXFP4 {
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
		if !ok && q != mainkernels.MXFP4 {
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
type gpu struct {
	d    backend.Device
	bufs []backend.Buf
}

func newGPU(t *testing.T, d backend.Device) *gpu { return &gpu{d: d} }

func (g *gpu) up(p []byte) backend.Buf {
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

func (g *gpu) free() {
	for _, b := range g.bufs {
		b.Free()
	}
}

// run compiles and launches one matvec, reducing the split partials, and
// returns rows*slots float32 results.
func (g *gpu) run(t *testing.T, s mainkernels.MatVecShape, rows, slots, split int,
	qs, dw, scw, av, ax, sel backend.Buf) []float32 {
	t.Helper()
	kk, err := mainkernels.MatVec(s)
	if err != nil {
		t.Skipf("shape rejected: %v", err)
	}
	kern, err := g.d.Compile(kk)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kern.Close()

	n := rows * slots
	out, err := g.d.Alloc(n * split * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := out.Write(make([]byte, n*split*4)); err != nil {
		t.Fatal(err)
	}

	const width = 128
	args := []backend.Buf{qs, dw, scw, av, ax, out}
	if sel != nil {
		args = append(args, sel)
	}
	threads := n * split
	if err := kern.Launch((threads+width-1)/width, width, args...); err != nil {
		t.Fatalf("launch: %v", err)
	}

	final := out
	if split > 1 {
		// Reduce bakes its row count, which for an indexed launch is
		// Rows*Slots (not Rows, not Experts*Rows); Rows would leave the other
		// slots holding un-summed partials.
		rk, err := mainkernels.Reduce(n, split)
		if err != nil {
			t.Fatal(err)
		}
		rkern, err := g.d.Compile(rk)
		if err != nil {
			t.Fatal(err)
		}
		defer rkern.Close()
		f, err := g.d.Alloc(n * 4)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Free()
		if err := rkern.Launch((n+width-1)/width, width, out, f); err != nil {
			t.Fatal(err)
		}
		final = f
	}
	raw := make([]byte, n*4)
	if err := final.Read(raw); err != nil {
		t.Fatal(err)
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return v
}
