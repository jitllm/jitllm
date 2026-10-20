package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestTileMatchesReference runs a collective matrix multiply on the device and
// compares it with a float64 evaluation of the same product. What a cooperative
// matrix supports (shapes, component types, accumulator width) is answered by
// the hardware, and this numeric probe answers it directly.
//
// The chain of MulAdds is the point: the accumulator must carry from one to the
// next as a value, as attention needs over a whole head dimension.
func TestTileMatchesReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const chain = 3
	// The shape list is the union of two disjoint ones: NVIDIA's cooperative
	// matrices have M=16, Metal's simdgroup_matrix is 8x8 only. No shape runs
	// on both, so a kernel must ask the device (vulkan.CooperativeMatrixShapes;
	// msl's tileT admits 8x8 only) rather than bake a shape in.
	cases := []struct {
		m, n, k int
		elem    ir.TileElem
	}{
		{16, 8, 16, ir.TileF16},  // spirv
		{16, 16, 16, ir.TileF16}, // spirv
		{16, 8, 32, ir.TileS8},   // spirv; Metal has no integer matrix at all
		{8, 8, 8, ir.TileF16},    // msl, the MIXED form: half operands, f32 accumulator
		{8, 8, 8, ir.TileF32},    // msl
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, c := range cases {
			kk, err := oldkernels.TileProbe(c.m, c.n, c.k, chain, c.elem)
			if err != nil {
				t.Fatalf("TileProbe: %v", err)
			}
			if err := kk.Validate(); err != nil {
				t.Fatalf("validate: %v", err)
			}
			kern, err := d.Compile(kk)
			if err != nil {
				// A decline is expected for most (backend, shape) pairs: ptx
				// has no cooperative-matrix lowering (its matrix op is OpMMA),
				// and spirv and msl refuse shapes their device does not
				// enumerate. The count below stops an all-declined run passing.
				t.Logf("%s %dx%dx%d %s: declined: %v", d.API(), c.m, c.n, c.k, c.elem, err)
				continue
			}
			t.Run(d.API()+"/"+itoa(c.m)+"x"+itoa(c.n)+"x"+itoa(c.k)+"/"+c.elem.String(), func(t *testing.T) {
				defer kern.Close()
				tileCase(t, d, kern, c.m, c.n, c.k, chain, c.elem)
			})
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no backend on this host lowers a cooperative matrix")
	}
}

func tileCase(t *testing.T, d backend.Device, kern backend.Kernel, m, n, k, chain int, elem ir.TileElem) {
	t.Helper()
	esz := elem.Bytes()
	// Small integers, exact in both f16 and s8, so the only permitted
	// difference is the accumulator's summation order.
	val := func(i int) float64 { return float64((i*7)%11 - 5) }

	aN, bN := chain*m*k, chain*k*n
	aBytes := make([]byte, aN*esz)
	bBytes := make([]byte, bN*esz)
	// The default is a failure, not a value: a missing arm would write
	// nothing and read like a broken lowering.
	put := func(dst []byte, i int, v float64) {
		switch elem {
		case ir.TileF16:
			binary.LittleEndian.PutUint16(dst[i*2:], f16bits(float32(v)))
		case ir.TileF32:
			binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(float32(v)))
		case ir.TileS8:
			dst[i] = byte(int8(v))
		default:
			t.Fatalf("no host encoder for %s; a tile element type added without "+
				"one here writes zeros and blames the device", elem)
		}
	}
	for i := 0; i < aN; i++ {
		put(aBytes, i, val(i))
	}
	for i := 0; i < bN; i++ {
		put(bBytes, i, val(i+3))
	}

	accSz := 4
	outBytes := make([]byte, m*n*accSz)
	bufs := make([]backend.Buf, 0, 3)
	for _, b := range [][]byte{aBytes, bBytes, outBytes} {
		buf, err := d.Alloc(len(b))
		if err != nil {
			t.Fatalf("alloc %d: %v", len(b), err)
		}
		defer buf.Free()
		if err := buf.Write(b); err != nil {
			t.Fatalf("write: %v", err)
		}
		bufs = append(bufs, buf)
	}
	// One subgroup: a cooperative matrix is distributed across exactly the
	// lanes of one, and a wider launch would compute the same tile repeatedly.
	if err := kern.Launch(1, 32, bufs[0], bufs[1], bufs[2]); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if err := bufs[2].Read(outBytes); err != nil {
		t.Fatalf("read: %v", err)
	}

	var worst float64
	for r := 0; r < m; r++ {
		for col := 0; col < n; col++ {
			var want float64
			for c := 0; c < chain; c++ {
				for i := 0; i < k; i++ {
					want += val(c*m*k+r*k+i) * val(c*k*n+i*n+col+3)
				}
			}
			var got float64
			if elem == ir.TileS8 {
				got = float64(int32(binary.LittleEndian.Uint32(outBytes[(r*n+col)*4:])))
			} else {
				got = float64(math.Float32frombits(binary.LittleEndian.Uint32(outBytes[(r*n+col)*4:])))
			}
			if dd := math.Abs(got - want); dd > worst {
				worst = dd
			}
		}
	}
	// Exact operands and an f32/i32 accumulator over 3*k products of small
	// integers: the result is exactly representable, so this is equality with
	// room only for f32 summation order.
	if worst > 1e-3 {
		t.Fatalf("%dx%dx%d %s: worst |device - reference| = %g over %d MulAdds; the "+
			"tile is not computing the product", m, n, k, elem, worst, chain)
	}
	t.Logf("%dx%dx%d %s: %d chained MulAdds, worst |d| %g", m, n, k, elem, chain, worst)
}

// f16bits rounds a float32 to IEEE binary16. Only the normal range is needed:
// every operand above is a small integer.
func f16bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32((b>>23)&0xFF) - 127 + 15
	man := uint16((b >> 13) & 0x3FF)
	if f == 0 {
		return sign
	}
	if exp <= 0 || exp >= 0x1F {
		panic("f16bits: out of the normal range")
	}
	return sign | uint16(exp)<<10 | man
}
