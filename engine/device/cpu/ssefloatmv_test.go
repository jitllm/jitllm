//go:build amd64

package cpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// The SSE tier's float matvec gates: F32, F16 and BF16 routers, gates and
// verbatim float matrices. They execute directly (legacy SSE runs on an AVX2
// host).

var sseFloatTypes = []quant.Type{quant.F32, quant.F16, quant.BF16}

// sseFloatKs exercise each of the kernel's three loops alone and together:
// singles only (1, 3, 7), octets only (8, 16), groups only (32, 64), all three
// (13, 45, 100, 2047) -- TestFloatMatVecEveryK's list.
var sseFloatKs = []int{1, 3, 7, 8, 16, 32, 64, 13, 45, 100, 2047}

func sseFloatKernel(t *testing.T, typ quant.Type) []byte {
	t.Helper()
	code, err := EmitRowMajorSSE(Spec{W: typ, Rows: 1, Accs: BestAccs(typ), Cols: 1})
	if err != nil {
		t.Fatalf("%s: %v", typ, err)
	}
	if KernelTier(code) != TierSSE {
		t.Fatalf("%s: the float matvec is not declared SSE-tier", typ)
	}
	requireSSEKernel(t, "sse_floatmv_"+typ.String(), code)
	return code
}

// floatRows fills rows*k weights of typ starting at a byte offset that
// misaligns every row, and returns the bytes the kernel reads plus the values
// it must read them as. exact draws weights with at most 11 significant bits
// (8 for BF16) so a product with a 12-bit activation is exact.
func floatRows(rng *rand.Rand, typ quant.Type, rows, k int, exact bool) ([]byte, []float64) {
	es := int(typ.BlockBytes())
	raw := make([]byte, es+rows*k*es) // one element of misalignment in front
	w := raw[es:]
	wv := make([]float64, rows*k)
	for i := range wv {
		v := float32(rng.NormFloat64())
		if exact {
			v = float32(math.Ldexp(float64(rng.Intn(4095)-2047), -rng.Intn(16)-8))
		}
		switch typ {
		case quant.F32:
			*(*float32)(unsafe.Pointer(&w[4*i])) = v
			wv[i] = float64(v)
		case quant.F16:
			h := quant.EncodeHalf(v)
			if i%97 == 5 {
				h = uint16(rng.Intn(0x400)) | uint16(rng.Intn(2))<<15 // a subnormal half
			}
			*(*uint16)(unsafe.Pointer(&w[2*i])) = h
			wv[i] = quant.DecodeHalf(h)
		default:
			b := uint16(math.Float32bits(v) >> 16)
			*(*uint16)(unsafe.Pointer(&w[2*i])) = b
			wv[i] = float64(math.Float32frombits(uint32(b) << 16))
		}
	}
	return w, wv
}

func floatActs(rng *rand.Rand, k int, exact bool) []float32 {
	raw := make([]float32, k+1)
	x := raw[1:] // misaligned by one float
	for i := range x {
		x[i] = float32(rng.NormFloat64())
		if exact {
			x[i] = float32(math.Ldexp(float64(rng.Intn(8191)-4095), -rng.Intn(16)-6))
		}
	}
	return x
}

func callFloat(t *testing.T, code []byte, w []byte, x []float32, rows, k int, es int) []float32 {
	t.Helper()
	c := mustMap(t, code)
	defer c.Close()
	g := newGuarded(rows, float32(math.NaN()))
	c.Call(&Args{Out: &g.out()[0], W: &w[0], A: (*int8)(unsafe.Pointer(&x[0])),
		Rows: int64(rows), K: int64(k), RowStr: int64(k * es)})
	g.check(t, "float matvec")
	return append([]float32(nil), g.out()...)
}

// TestSSEFloatMatVecEveryK holds the SSE float matvec to a float64 sum over the
// weights exactly as stored, at every k shape, with misaligned rows and
// activations and a guard on each side of the output.
func TestSSEFloatMatVecEveryK(t *testing.T) {
	for _, typ := range sseFloatTypes {
		code := sseFloatKernel(t, typ)
		es := int(typ.BlockBytes())
		for _, k := range sseFloatKs {
			const rows = 5
			rng := rand.New(rand.NewSource(int64(k)))
			w, wv := floatRows(rng, typ, rows, k, false)
			x := floatActs(rng, k, false)
			out := callFloat(t, code, w, x, rows, k, es)
			for r := 0; r < rows; r++ {
				var want float64
				for i := 0; i < k; i++ {
					want += wv[r*k+i] * float64(x[i])
				}
				if d := math.Abs(float64(out[r]) - want); d > 1e-4*(1+math.Abs(want))*math.Sqrt(float64(k)) ||
					math.IsNaN(float64(out[r])) {
					t.Fatalf("%s k=%d row %d: %v, want %v", typ, k, r, out[r], want)
				}
			}
		}
	}
}

// TestSSEFloatMatVecMatchesAVX2BitForBit: with every product exact, the SSE
// kernel's only arithmetic difference from the AVX2 one -- a rounded product
// before the add, where AVX2 fuses them -- rounds nothing, so the two must agree
// to the bit. That makes this a gate on the chain, lane and fold order: the
// sums still round, and a chain summed in a different order is a different
// float.
func TestSSEFloatMatVecMatchesAVX2BitForBit(t *testing.T) {
	requireAVX2Twin(t)
	for _, typ := range sseFloatTypes {
		sse := sseFloatKernel(t, typ)
		avx, err := EmitNative(Spec{W: typ, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatal(err)
		}
		es := int(typ.BlockBytes())
		for _, k := range append(sseFloatKs, 4096, 777) {
			rows := 7
			rng := rand.New(rand.NewSource(int64(1000 + k)))
			w, _ := floatRows(rng, typ, rows, k, true)
			x := floatActs(rng, k, true)
			got := callFloat(t, sse, w, x, rows, k, es)
			want := callFloat(t, avx, w, x, rows, k, es)
			for r := range got {
				if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
					t.Fatalf("%s k=%d row %d: SSE %v (%#08x), AVX2 %v (%#08x) -- every product is "+
						"exact, so the chains or the fold are in a different order",
						typ, k, r, got[r], math.Float32bits(got[r]), want[r], math.Float32bits(want[r]))
				}
			}
		}
	}
}

// TestSSEFloatMatVecZeroRows: Rows=0 is no work, not 2^64 DEC/JNZ iterations.
func TestSSEFloatMatVecZeroRows(t *testing.T) {
	code := sseFloatKernel(t, quant.F32)
	c := mustMap(t, code)
	defer c.Close()
	out := []float32{-3}
	w, x := make([]byte, 64), make([]float32, 16)
	c.Call(&Args{Out: &out[0], W: &w[0], A: (*int8)(unsafe.Pointer(&x[0])), Rows: 0, K: 16, RowStr: 64})
	if out[0] != -3 {
		t.Fatalf("a zero-row call wrote %v", out[0])
	}
}

// TestSSERowMajorRefusesWhatItCannotServe: no quantized row-major kernel on
// this tier (a container never decodes through the GGUF layout), no
// interleaved float kernel, no multi-column one.
func TestSSERowMajorRefusesWhatItCannotServe(t *testing.T) {
	for _, g := range append([]quant.Type{}, quant.PackedTypes...) {
		if SupportedRowMajorSSE(g) {
			t.Errorf("%s: the SSE tier claims a row-major kernel", g)
		}
		if _, err := EmitRowMajorSSE(Spec{W: g, Rows: 1, Accs: 1, Cols: 1}); err == nil {
			t.Errorf("%s: the SSE tier emitted a row-major kernel", g)
		}
	}
	for _, typ := range sseFloatTypes {
		if !SupportedRowMajorSSE(typ) {
			t.Errorf("%s: the SSE tier has no float matvec", typ)
		}
		if _, err := EmitRowMajorSSE(Spec{W: typ, Rows: Interleave, Accs: 1, Cols: 1}); err == nil {
			t.Errorf("%s: an interleaved float kernel emitted", typ)
		}
		if _, err := EmitRowMajorSSE(Spec{W: typ, Rows: 1, Accs: 1, Cols: 2}); err == nil {
			t.Errorf("%s: a two-column float kernel emitted", typ)
		}
	}
	if EmittersFor(TierSSE).RowMajorGGUF {
		t.Error("the SSE table claims the GGUF row-major machinery")
	}
}
