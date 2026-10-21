//go:build amd64 && linux

package cpu

import (
	"fmt"
	"github.com/jitllm/jitllm/internal/oracle"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestKernelMatchesReference is T1: a float64 evaluation of the same int8
// activations and per-block scales the kernel uses, so the only permitted
// difference is summation order (unquantized activations would need a ~1e-2
// tolerance).
func TestKernelMatchesReference(t *testing.T) {
	ran := 0
	for _, wt := range quant.PackedTypes {
		// This gate is about Emit, the row-major matvec, which not every
		// packed type has (Q5_0), so it asks Supported rather than keeping a
		// second list.
		if !Supported(wt) {
			t.Logf("%s: Emit has no kernel (the packed path does)", wt)
			continue
		}
		ran++
		t.Run(wt.String(), func(t *testing.T) { kernelVsReference(t, wt, 1) })
	}
	if ran == 0 {
		t.Fatal("no format reached Emit; this gate proved nothing")
	}
}

// TestKQuantAccumulatorChains holds every accumulator split to the same bar as
// the single-chain kernel.
//
// Splitting the chain changes only the summation order, so the result must
// still match the float64 evaluation; a mis-assigned chain register produces a
// plausible number that only the oracle catches.
//
// Q5_K spends Y15 on qh and so has one chain fewer; asking for 4 must clamp
// rather than emit a kernel that writes a register it does not own.
func TestKQuantAccumulatorChains(t *testing.T) {
	for _, wt := range []quant.Type{quant.Q4_K, quant.Q5_K} {
		for _, accs := range []int8{2, 4} {
			t.Run(fmt.Sprintf("%s/accs%d", wt, accs), func(t *testing.T) {
				kernelVsReference(t, wt, accs)
			})
		}
	}
}

func kernelVsReference(t *testing.T, wt quant.Type, accs int8) {
	code, err := Emit(Spec{W: wt, Rows: 1, Accs: accs, Cols: 1})
	if err != nil {
		t.Fatal(err)
	}
	kern := mustMap(t, code)
	defer kern.Close()
	t.Logf("%s matvec kernel, %d chain(s): %d bytes", wt, accs, len(code))

	mask := KernelConst(wt)
	shapes := []struct{ rows, k int }{{1, 32}, {4, 64}, {8, 256}, {32, 2048}, {3, 96}}
	if wt.BlockElems() == 256 {
		shapes = []struct{ rows, k int }{{1, 256}, {4, 512}, {8, 1024}, {32, 2048}, {3, 768}}
	}
	for _, shape := range shapes {
		rng := rand.New(rand.NewSource(int64(shape.rows*1000 + shape.k)))
		packed, exact := buildWeights(rng, wt, shape.rows, shape.k)

		// Four input distributions. Uniform alone is not enough: real activations
		// are heavy-tailed, and the interesting failures are at the extremes of
		// the quantization range.
		for _, dist := range []struct {
			name string
			gen  func(i int) float64
		}{
			{"uniform", func(int) float64 { return rng.Float64()*2 - 1 }},
			{"student-t", func(int) float64 { return rng.NormFloat64() / (0.1 + math.Abs(rng.NormFloat64())) }},
			{"one-huge-per-block", func(i int) float64 {
				if i%32 == 7 {
					return 1000
				}
				return rng.Float64() * 0.01
			}},
			{"saturating", func(i int) float64 {
				if i%2 == 0 {
					return 1
				}
				return -1
			}},
		} {
			x := make([]float32, shape.k)
			for i := range x {
				x[i] = float32(dist.gen(i))
			}
			q := make([]int8, shape.k)
			pairs := make([]float32, 2*(shape.k/32))
			if err := oracle.QuantizeQ8(q, pairs, x, BiasC(wt)); err != nil {
				t.Fatal(err)
			}
			// The 16-wide formats need per-16 sums as well.
			half := make([]float32, shape.k/16)
			oracle.QuantizeHalfSums(half, q, BiasC(wt))

			out := make([]float32, shape.rows)
			args := Args{
				Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
				Rows: int64(shape.rows), K: int64(BlocksPerRow(wt, shape.k)),
				RowStr: int64(RowBytes(wt, shape.k)), Scr: &mask[0], AHalf: &half[0],
			}
			kern.Call(&args)

			// Reference: the same arithmetic, in float64.
			var sse, sy2 float64
			for r := 0; r < shape.rows; r++ {
				var want float64
				for b := 0; b < shape.k/32; b++ {
					dx := float64(pairs[2*b])
					for i := 0; i < 32; i++ {
						want += exact[r][b*32+i] * float64(q[b*32+i]) * dx
					}
				}
				d := float64(out[r]) - want
				sse += d * d
				sy2 += want * want
			}
			nmse := 0.0
			if sy2 > 0 {
				nmse = sse / sy2
			}
			if nmse > 1e-10 || math.IsNaN(nmse) {
				t.Errorf("%s rows=%d k=%d %s: NMSE %.3e exceeds 1e-10", wt, shape.rows, shape.k, dist.name, nmse)
			}
		}
	}
}

// TestQ40KernelVsTrueDot reports how much accuracy activation quantization
// costs against the unquantized answer. Not a gate.
func TestQ40KernelVsTrueDot(t *testing.T) {
	if ggufDeclined(t, quant.Q4_0) {
		return
	}
	code, err := Emit(Spec{W: quant.Q4_0, Rows: 1, Accs: 1, Cols: 1})
	if err != nil {
		t.Fatal(err)
	}
	kern := mustMap(t, code)
	defer kern.Close()

	const rows, k = 16, 2048
	rng := rand.New(rand.NewSource(7))
	packed, exact := buildWeights(rng, quant.Q4_0, rows, k)
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, k)
	pairs := make([]float32, 2*(k/32))
	if err := oracle.QuantizeQ8(q, pairs, x, BiasC(quant.Q4_0)); err != nil {
		t.Fatal(err)
	}
	mask := KernelConst(quant.Q4_0)
	out := make([]float32, rows)
	args := Args{Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
		Rows: rows, K: k / 32, Scr: &mask[0]}
	kern.Call(&args)

	var sse, sy2 float64
	for r := 0; r < rows; r++ {
		var want float64
		for i := 0; i < k; i++ {
			want += exact[r][i] * float64(x[i])
		}
		d := float64(out[r]) - want
		sse += d * d
		sy2 += want * want
	}
	t.Logf("int8 activation quantization costs NMSE %.3e against the exact dot product", sse/sy2)
}

func TestEmitRejects(t *testing.T) {
	if _, err := Emit(Spec{W: quant.Q2_K, Rows: 1, Cols: 1}); err == nil {
		t.Error("Emit accepted a quant type with no kernel")
	}
	if _, err := Emit(Spec{W: quant.Q4_0, Rows: 3, Cols: 1}); err == nil {
		t.Error("Emit accepted an unimplemented interleave width")
	}
	if _, err := Emit(Spec{W: quant.Q4_0, Rows: 1, Cols: 4}); err == nil {
		t.Error("Emit accepted a batched shape; only decode matvec exists")
	}
	for _, rows := range []int8{1, Interleave} {
		if _, err := Emit(Spec{W: quant.Q4_0, Rows: rows, Cols: 1}); err != nil {
			t.Errorf("Emit(Rows=%d) failed: %v", rows, err)
		}
	}
	if n := MaxBlocksPerCall(); n < 1000 {
		t.Errorf("budget allows only %d blocks per call, which would make chunking dominate", n)
	}
	if r := RowsPerCall(64); r < 1 {
		t.Error("RowsPerCall must always allow at least one row")
	}
	t.Logf("execution budget: %d blocks per call, %d rows of 2048 elements",
		MaxBlocksPerCall(), RowsPerCall(64))
}

// TestInterleavedKernelMatchesReference holds the 4-row kernel to exactly the
// same bar as the 1-row one. Interleaving changes only the schedule, so any
// difference is a bug in the four cursors' pointer bookkeeping.
func TestInterleavedKernelMatchesReference(t *testing.T) {
	for _, wt := range []quant.Type{quant.Q4_0, quant.Q8_0} {
		t.Run(wt.String(), func(t *testing.T) {
			code, err := Emit(Spec{W: wt, Rows: Interleave, Accs: 1, Cols: 1})
			if err != nil {
				t.Fatal(err)
			}
			kern := mustMap(t, code)
			defer kern.Close()
			t.Logf("%s interleaved kernel: %d bytes", wt, len(code))

			konst := KernelConst(wt)
			for _, shape := range []struct{ rows, k int }{
				{4, 32}, {8, 64}, {64, 256}, {256, 2048}, {12, 96},
			} {
				rng := rand.New(rand.NewSource(int64(shape.rows*31 + shape.k)))
				packed, exact := buildWeights(rng, wt, shape.rows, shape.k)
				rowBytes := shape.k / 32 * int(wt.BlockBytes())

				for _, dist := range []struct {
					name string
					gen  func(i int) float64
				}{
					{"uniform", func(int) float64 { return rng.Float64()*2 - 1 }},
					{"student-t", func(int) float64 {
						return rng.NormFloat64() / (0.1 + math.Abs(rng.NormFloat64()))
					}},
					{"one-huge-per-block", func(i int) float64 {
						if i%32 == 3 {
							return 500
						}
						return rng.Float64() * 0.01
					}},
					{"saturating", func(i int) float64 {
						if i%2 == 0 {
							return 1
						}
						return -1
					}},
				} {
					x := make([]float32, shape.k)
					for i := range x {
						x[i] = float32(dist.gen(i))
					}
					q := make([]int8, shape.k)
					pairs := make([]float32, 2*(shape.k/32))
					if err := oracle.QuantizeQ8(q, pairs, x, BiasC(wt)); err != nil {
						t.Fatal(err)
					}
					out := make([]float32, shape.rows)
					args := Args{
						Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
						Rows:   int64(shape.rows / Interleave), // GROUPS, not rows
						K:      int64(shape.k / 32),
						RowStr: int64(rowBytes),
						Scr:    &konst[0],
					}
					kern.Call(&args)

					var sse, sy2 float64
					for r := 0; r < shape.rows; r++ {
						var want float64
						for b := 0; b < shape.k/32; b++ {
							dx := float64(pairs[2*b])
							for i := 0; i < 32; i++ {
								want += exact[r][b*32+i] * float64(q[b*32+i]) * dx
							}
						}
						d := float64(out[r]) - want
						sse += d * d
						sy2 += want * want
					}
					nmse := 0.0
					if sy2 > 0 {
						nmse = sse / sy2
					}
					if nmse > 1e-10 || math.IsNaN(nmse) {
						t.Errorf("%s x4 rows=%d k=%d %s: NMSE %.3e exceeds 1e-10",
							wt, shape.rows, shape.k, dist.name, nmse)
					}
				}
			}
		})
	}
}
