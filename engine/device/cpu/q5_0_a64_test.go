//go:build arm64

package cpu

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestA64Q5_0MatchesOracle gates the Q5_0 arm of emitA64MatVec. It is tagged
// `arm64` alone (not `arm64 && darwin` like TestA64KernelMatchesReference) so
// it also runs under qemu-aarch64 on an amd64 box.
//
// A qh plane is invisible to random data in aggregate, so the sub-cases pin
// each part of the mapping separately:
//
//	all-clear / all-set   the plane as a whole, and the zero point with it
//	                      (16, not Q4_0's 8 -- a kernel using 8 is wrong by
//	                      8*d on every element and still looks like a matvec)
//	low-half / high-half  which qh bytes feed elements 0..15 and 16..31, so a
//	                      TBL table reading the wrong half fails loudly
//	one-bit-per-byte      bits 0, 8, 16, 24 alone, so a wrong per-lane USHL
//	                      count lands the fifth bit on the wrong element
//	random                the shipping case
//
// The reference is internal/oracle.MatVec (quant.Dequant32 plus Neumaier
// compensation), a different method from the kernel's four-lane FMA.
func TestA64Q5_0MatchesOracle(t *testing.T) {
	code, err := EmitA64(Spec{W: quant.Q5_0, Rows: 1, Accs: 1, Cols: 1})
	if err != nil {
		t.Fatal(err)
	}
	kern := mustMap(t, code)
	defer kern.Close()

	konst := KernelConst(quant.Q5_0)
	if len(konst) < 32 {
		t.Fatalf("KernelConst(Q5_0) is %d bytes; the kernel reads two 16-byte vectors", len(konst))
	}
	scratch := make([]float32, 64)

	qhCases := []struct {
		name string
		of   func(b int, rnd uint32) uint32
	}{
		{"random", func(_ int, rnd uint32) uint32 { return rnd }},
		{"all-clear", func(int, uint32) uint32 { return 0 }},
		{"all-set", func(int, uint32) uint32 { return 0xFFFFFFFF }},
		{"low-half-set", func(int, uint32) uint32 { return 0x0000FFFF }},
		{"high-half-set", func(int, uint32) uint32 { return 0xFFFF0000 }},
		{"one-bit-per-byte", func(int, uint32) uint32 { return 0x01010101 }},
		{"walking", func(b int, _ uint32) uint32 { return 1 << uint(b%32) }},
	}
	dists := []struct {
		name string
		gen  func(rng *rand.Rand, i int) float64
	}{
		{"uniform", func(rng *rand.Rand, _ int) float64 { return rng.Float64()*2 - 1 }},
		{"student-t", func(rng *rand.Rand, _ int) float64 {
			return rng.NormFloat64() / (0.1 + math.Abs(rng.NormFloat64()))
		}},
		{"one-huge-per-block", func(rng *rand.Rand, i int) float64 {
			if i%32 == 7 {
				return 1000
			}
			return rng.Float64() * 0.01
		}},
		{"saturating", func(_ *rand.Rand, i int) float64 {
			if i%2 == 0 {
				return 1
			}
			return -1
		}},
	}

	for _, shape := range []struct{ rows, k int }{
		{1, 32}, {4, 32}, {1, 64}, {4, 64}, {1, 256}, {4, 256}, {4, 2048}, {3, 96},
	} {
		for _, qc := range qhCases {
			name := qc.name + "/" + itoa(shape.rows) + "x" + itoa(shape.k)
			t.Run(name, func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(shape.rows*100003 + shape.k)))
				nb := shape.k / 32
				packed, _ := buildWeights(rng, quant.Q5_0, shape.rows, shape.k)
				for b := 0; b < shape.rows*nb; b++ {
					off := b*22 + 2
					rnd := binary.LittleEndian.Uint32(packed[off:])
					binary.LittleEndian.PutUint32(packed[off:], qc.of(b, rnd))
				}

				for _, dist := range dists {
					x := make([]float32, shape.k)
					for i := range x {
						x[i] = float32(dist.gen(rng, i))
					}
					q := make([]int8, shape.k)
					pairs := make([]float32, 2*(shape.k/Q8Block))
					if err := oracle.QuantizeQ8(q, pairs, x, BiasC(quant.Q5_0)); err != nil {
						t.Fatal(err)
					}
					out := make([]float32, shape.rows)
					args := Args{
						Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
						Rows: int64(shape.rows), K: int64(BlocksPerRow(quant.Q5_0, shape.k)),
						RowStr: int64(RowBytes(quant.Q5_0, shape.k)),
						Scr:    &konst[0], Scratch: (*byte)(unsafe.Pointer(&scratch[0])),
					}
					kern.Call(&args)

					// The oracle is fed the dequantized activations, so the only
					// difference permitted between the two is summation order.
					xq := dequantAct(q, pairs)
					want := make([]float32, shape.rows)
					if err := oracle.MatVec(want, quant.Q5_0, packed, xq, shape.rows, shape.k); err != nil {
						t.Fatal(err)
					}

					var sse, sy2 float64
					for r := 0; r < shape.rows; r++ {
						d := float64(out[r]) - float64(want[r])
						sse += d * d
						sy2 += float64(want[r]) * float64(want[r])
					}
					// A non-finite or all-zero reference scores 0.00e+00 under a
					// naive ratio and reads as a perfect pass. Refuse it.
					if math.IsNaN(sse) || math.IsInf(sse, 0) || math.IsNaN(sy2) || math.IsInf(sy2, 0) {
						t.Fatalf("%s %s: the REFERENCE is non-finite (sse %g, sy2 %g) -- "+
							"a degenerate oracle is a failure, not a pass", name, dist.name, sse, sy2)
					}
					if sy2 == 0 {
						t.Fatalf("%s %s: the reference is identically zero", name, dist.name)
					}
					if nmse := sse / sy2; nmse > 1e-10 {
						t.Errorf("%s %s: NMSE %.3e exceeds 1e-10 (kernel %v, oracle %v)",
							name, dist.name, nmse, out, want)
					}
				}
			})
		}
	}
}

// TestA64Q5_0FixtureSeesTheBitPlane asserts the harness can tell the qh cases
// apart before anything is concluded from the kernel agreeing with it. The
// activations vary (under a flat x, 0x0000FFFF and its complement 0xFFFF0000
// give the same dot product), and the comparison is pairwise across seven
// distinct planes:
//
//	0 vs 0xFFFFFFFF          the plane is read at all
//	0x0000FFFF vs 0xFFFF0000 which qh bytes drive elements 0..15 vs 16..31
//	0x00000001 vs 0x00000100 which bit drives which element inside a byte
func TestA64Q5_0FixtureSeesTheBitPlane(t *testing.T) {
	const rows, k = 4, 64
	rng := rand.New(rand.NewSource(7))
	base, _ := buildWeights(rng, quant.Q5_0, rows, k)
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(1 + 0.5*float64(i)) // never flat: a flat x cancels a half swap
	}

	planes := []uint32{0, 0xFFFFFFFF, 0x0000FFFF, 0xFFFF0000, 0x01010101, 0x00000001, 0x00000100}
	refs := make([][]float32, len(planes))
	for pi, qh := range planes {
		w := append([]byte(nil), base...)
		for b := 0; b < rows*k/32; b++ {
			binary.LittleEndian.PutUint32(w[b*22+2:], qh)
		}
		refs[pi] = make([]float32, rows)
		if err := oracle.MatVec(refs[pi], quant.Q5_0, w, x, rows, k); err != nil {
			t.Fatal(err)
		}
	}
	for i := range planes {
		for j := i + 1; j < len(planes); j++ {
			sep := false
			for r := 0; r < rows; r++ {
				a, b := float64(refs[i][r]), float64(refs[j][r])
				if math.Abs(a-b) > 1e-3*(1+math.Abs(a)) {
					sep = true
					break
				}
			}
			if !sep {
				t.Errorf("qh %#08x and %#08x give the SAME reference on every row -- "+
					"the fixture cannot distinguish them, so a kernel that confused "+
					"them would pass", planes[i], planes[j])
			}
		}
	}
}

// dequantAct undoes QuantizeQ8 the way the kernel's own arithmetic does: the
// int8 code times its block's scale. The bias half of each pair is deliberately
// not read -- matvec_a64.go's SDOT path never reads it either.
func dequantAct(q []int8, pairs []float32) []float32 {
	x := make([]float32, len(q))
	for i := range q {
		x[i] = float32(q[i]) * pairs[2*(i/Q8Block)]
	}
	return x
}
