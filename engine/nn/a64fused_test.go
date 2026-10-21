package nn

import (
	"math"
	"runtime"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestFusedMatchesTheTiledKernel holds the fused packed matvec to the same bits
// as the tiled one, on every format both serve. On arm64 the two are one
// emitter nested two ways (cpu.EmitA64PackedMatVecFused): the tiled kernel
// keeps Out in memory and walks super-blocks outside its tiles, the fused one
// keeps it in registers and walks them inside, and both add the same terms to
// the same running sum in the same order -- so any difference is a cursor that
// read another super-block's bytes, not a rounding.
func TestFusedMatchesTheTiledKernel(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("amd64's fused and tiled kernels are separate emitters with different reduction orders")
	}
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) {
			continue
		}
		step := kStep(gt)
		for _, sh := range []struct{ rows, k, bankRow int }{
			{64, 2 * step, 0},
			{192, 5 * step, 0},
			{72, 3 * step, 128}, // a tail, inside a mixture's bank
			{1024, max(2, 2048/step) * step, 0},
		} {
			stride := sh.rows + sh.bankRow
			pk := packOne(t, gt, stride, sh.k)
			pk.Row, pk.Stride = sh.bankRow, stride
			x := prevnniAct(sh.k)
			run := func(fused bool) []float32 {
				j := NewJIT(sh.k, stride, []quant.Type{gt}, WithFusedKSlices(-1), WithGEMMExact(true)) // the float forms
				if j == nil {
					t.Fatal("no JIT")
				}
				defer j.Close()
				if j.packedFused[gt] == nil {
					t.Fatalf("%s: PackedFusedSupported but the JIT built no fused kernel", gt)
				}
				if !fused {
					delete(j.packedFused, gt) // fusedAt still owns and closes it
				}
				out := make([]float32, sh.rows+1)
				out[sh.rows] = 12345
				if !j.MatVecPacked(out, gt, pk, x, sh.rows, sh.k) {
					t.Fatalf("%s %dx%d: MatVecPacked declined", gt, sh.rows, sh.k)
				}
				if out[sh.rows] != 12345 {
					t.Fatalf("%s %dx%d: wrote past the rows", gt, sh.rows, sh.k)
				}
				return out[:sh.rows]
			}
			want, got := run(false), run(true)
			for r := range want {
				if math.Float32bits(got[r]) != math.Float32bits(want[r]) || math.IsNaN(float64(want[r])) {
					t.Fatalf("%s %dx%d bank row %d: row %d is %v fused, %v tiled",
						gt, sh.rows, sh.k, sh.bankRow, r, got[r], want[r])
				}
			}
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no format has a fused kernel on arm64 -- this gate proved nothing")
	}
	t.Logf("%d (format, shape) pair(s) bit-identical, fused against tiled", ran)
}
