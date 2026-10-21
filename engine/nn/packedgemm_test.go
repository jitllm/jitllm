//go:build amd64 || arm64

package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// gemmShapes are the weight-stationary GEMM's gate shapes. The row counts are
// multiples of sixteen (MatMulPacked's own floor) that are odd multiples of
// eight groups where they can be, the token counts divide nothing, and the
// per-call token counts below make the last token block of a call ragged.
var gemmShapes = []struct{ nrows, k, ntok int }{
	{768, 768, 8},   // the tower's q/k/v/o shape
	{3072, 768, 5},  // fc1, an odd token count
	{768, 3072, 3},  // fc2
	{16, 512, 2},    // one fused group: two eight-row groups
	{48, 256, 37},   // six eight-row groups, a token count that divides nothing
	{512, 2048, 19}, // Llama-3.2-1B's k/v projection
	// Qwen2-VL-2B's: k/v at the model width, and the down projection, whose
	// k is 35 super-blocks -- an odd count no shape above has.
	{256, 1536, 3},
	{128, 8960, 3},
}

// runGEMMGate holds the weight-stationary GEMM to the per-token row loop bit
// for bit, at five tokens x rows per call settings, and reports how many shapes
// it compared. It is shared by the native gate and the forced pre-VNNI one.
// The integer sums are exact in any order and the float epilogue matches the
// fused kernel's operation for operation, so a tolerance would only hide a
// wrong token, slice, row block or a wrapped int16 partial sum.
func runGEMMGate(t *testing.T, mk func(maxK, maxRows int, ts []quant.Type, opts ...Option) *JIT) int {
	t.Helper()
	compared := 0
	// Per format, so a format whose int16 partials wrapped reads as itself
	// rather than being averaged into the others' ~1e-12.
	worstNMSE := map[quant.Type]float64{}
	defer func() {
		for _, gt := range quant.PackedTypes {
			if w, ok := worstNMSE[gt]; ok {
				t.Logf("%s: integer-accumulated, worst NMSE against the row loop %.3e", gt, w)
			}
		}
	}()
	for _, gt := range quant.PackedTypes {
		for _, sh := range gemmShapes {
			if sh.k%int(gt.BlockElems()) != 0 {
				continue
			}
			for _, sat := range []bool{false, true} {
				nrows, k, ntok := sh.nrows, sh.k, sh.ntok
				pk := packOne(t, gt, nrows, k)
				rng := rand.New(rand.NewSource(int64(nrows*31 + k*7 + ntok)))
				x := make([]float32, ntok*k)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				if sat {
					// The worst case the int16 bound is derived for: every weight
					// at its largest code and every activation at +127. Random
					// data never comes near the bound.
					pk = packMax(t, gt, nrows, k)
					for i := range x {
						x[i] = 1
					}
				}
				// The reference: the fused per-token kernel, one token at a time.
				// TuneOff: arm64's timed per-shape matvec pick would otherwise
				// choose a k-sliced arm, which reassociates the float sums.
				jr := mk(k, nrows, []quant.Type{gt}, WithGEMMTokens(-1), WithTune(TuneOff))
				if jr == nil {
					t.Fatal("no JIT")
				}
				want := make([]float32, ntok*nrows)
				for i := 0; i < ntok; i++ {
					jr.NewInput()
					if !jr.MatVecPacked(want[i*nrows:(i+1)*nrows], gt, pk, x[i*k:(i+1)*k], nrows, k) {
						jr.Close()
						t.Fatalf("%s: the row loop declined", gt)
					}
				}
				jr.Close()
				finite := 0
				for _, v := range want {
					if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
						finite++
					}
				}
				if finite == 0 {
					t.Fatalf("%s %dx%d: every reference value is non-finite -- a degenerate "+
						"oracle is a failure, not a pass", gt, nrows, k)
				}
				for _, cfg := range []struct{ per, rows int }{{0, 0}, {3, 0}, {1, 0}, {0, 64}, {5, 32}} {
					per := cfg.per
					j := mk(k, nrows, []quant.Type{gt}, WithGEMMTokens(per), WithGEMMRows(cfg.rows))
					if j == nil {
						t.Fatal("no JIT")
					}
					// A guard after the output: the kernel must write nothing past
					// ntok*nrows, which is where a ragged last token block would
					// spill.
					got := make([]float32, ntok*nrows+16)
					for i := ntok * nrows; i < len(got); i++ {
						got[i] = float32(math.NaN())
					}
					exact := !cpu.StationaryIntAcc(gt, j.actWindow, cpu.HostDotKind())
					if !j.MatMulPacked(got, gt, pk, x, nrows, k, ntok) {
						j.Close()
						t.Fatalf("%s %dx%d ntok=%d: MatMulPacked declined", gt, nrows, k, ntok)
					}
					if n := j.PackedGEMMCalls(); n != 1 {
						j.Close()
						t.Fatalf("%s %dx%d ntok=%d per=%d: the weight-stationary GEMM ran %d times, "+
							"want 1 -- the gate would be comparing another path with the row loop",
							gt, nrows, k, ntok, per, n)
					}
					if cfg.rows > 0 && j.BatchedChunkRows() != min(cfg.rows, nrows) {
						j.Close()
						t.Fatalf("%s %dx%d: asked for %d rows a call and ran %d -- the pin did not reach the GEMM",
							gt, nrows, k, cfg.rows, j.BatchedChunkRows())
					}
					j.Close()
					for i := ntok * nrows; i < len(got); i++ {
						if !math.IsNaN(float64(got[i])) {
							t.Fatalf("%s %dx%d ntok=%d per=%d: wrote past the output at %d",
								gt, nrows, k, ntok, per, i)
						}
					}
					diffs, worst, first := 0, 0.0, -1
					for i := range want {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							if first < 0 {
								first = i
							}
							diffs++
							worst = max(worst, math.Abs(float64(got[i])-float64(want[i])))
						}
					}
					// The integer-accumulating form is held to a bound, not
					// equality: it sums a super-block's sub-blocks exactly where
					// the fused kernel rounds after each. 1e-10 is two orders
					// above the measured worst (~1e-12) and far below any real
					// defect (a dropped scale half reads ~1.5).
					if !exact {
						var num, den float64
						for i := range want {
							d := float64(got[i]) - float64(want[i])
							num += d * d
							den += float64(want[i]) * float64(want[i])
						}
						nmse := num / den
						worstNMSE[gt] = max(worstNMSE[gt], nmse)
						if !(nmse <= 1e-10) {
							t.Errorf("%s %dx%d ntok=%d per=%d sat=%v: integer-accumulated GEMM at NMSE %.3e "+
								"against the row loop (bound 1e-10); first difference at token %d row %d "+
								"(got %v want %v)", gt, nrows, k, ntok, per, sat, nmse,
								first/nrows, first%nrows, got[first], want[first])
						}
					} else if diffs != 0 {
						t.Errorf("%s %dx%d ntok=%d per=%d sat=%v: %d of %d elements differ, worst |d| %g, "+
							"first at token %d row %d (got %v want %v)", gt, nrows, k, ntok, per, sat,
							diffs, len(want), worst, first/nrows, first%nrows, got[first], want[first])
					}
				}
				compared++
			}
		}
	}
	return compared
}

// TestPackedGEMMMatchesTheRowLoopExactly is the weight-stationary GEMM's gate
// on this host's dot sequence: VPDPBUSD on a VNNI host, the int16-accumulating
// VEX sequence on a pre-VNNI one (a pre-VNNI host runs it natively).
func TestPackedGEMMMatchesTheRowLoopExactly(t *testing.T) {
	if tr := cpu.HostTier(); tr != cpu.TierAVX2 && tr != cpu.TierNEON {
		t.Skipf("tier %v has no weight-stationary GEMM (sse_refuse.go says why)", tr)
	}
	n := runGEMMGate(t, NewJIT)
	if n == 0 {
		t.Fatal("no (format, shape) pair ran -- this gate proved nothing")
	}
	t.Logf("%d (format, shape) pair(s) bit-identical at five tokens x rows per call settings, dot %s",
		n, cpu.HostDotKind())
}

// packMax is packOne with every payload byte at the format's largest code:
// 0xFF for the integer formats, 0x77 for MXFP4, whose code 7 is the table's
// largest entry (+6, biased to 24). The scale fields are planted as packOne's
// are, so only the payload is extreme.
func packMax(t *testing.T, gt quant.Type, nrows, k int) *Packed {
	t.Helper()
	q, _ := kernels.QuantOf(gt)
	nb := uint64(nrows*k) / gt.BlockElems()
	src := make([]byte, nb*gt.BlockBytes())
	fill := byte(0xFF)
	if gt == quant.MXFP4 {
		fill = 0x77
	}
	for i := range src {
		src[i] = fill
	}
	if !quant.PlantScales(gt, src, 0) {
		t.Fatalf("%s: no scale planter", gt)
	}
	qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	return &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
}

// TestPackedScratchSurvivesTheGEMM runs the fused matvec on a JIT whose
// weight-stationary GEMM has already grown the shared scale scratch.
//
// The GEMM grows pdscr and never pmscr, so a length check on pdscr alone would
// let the matvec index an empty pmscr (a panic on the first decode after a
// prefill).
func TestPackedScratchSurvivesTheGEMM(t *testing.T) {
	const gt = quant.Q4_K
	const nrows, k, ntok = 1024, 2048, 64
	pk := packOne(t, gt, nrows, k)
	j := NewJIT(k, nrows, []quant.Type{gt})
	if j == nil {
		t.Fatal("no JIT")
	}
	defer j.Close()
	rng := rand.New(rand.NewSource(3))
	x := make([]float32, ntok*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	batch := make([]float32, ntok*nrows)
	if !j.MatMulPacked(batch, gt, pk, x, nrows, k, ntok) || j.PackedGEMMCalls() != 1 {
		t.Skip("no weight-stationary GEMM on this host, so there is nothing to grow the scratch")
	}
	j.NewInput()
	one := make([]float32, nrows)
	if !j.MatVecPacked(one, gt, pk, x[:k], nrows, k) {
		t.Fatal("the matvec declined")
	}
	var num, den float64
	for r := range one {
		d := float64(one[r]) - float64(batch[r])
		num += d * d
		den += float64(batch[r]) * float64(batch[r])
	}
	if !(num/den <= 1e-10) {
		t.Fatalf("the matvec after the GEMM disagrees with the GEMM's token 0 at NMSE %.3e", num/den)
	}
}
