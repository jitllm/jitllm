package nn

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

func u32b(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// TestPackedMatVecTailAgrees holds every generated packed matvec to
// oracle.MatVecPacked over (rows, k) pairs that include ragged row counts (the
// main tile / tail / window split, where a stride bug hides) and real models'
// shapes, with varied scale planes, a poisoned output and a second call into
// the same buffer.
func TestPackedMatVecTailAgrees(t *testing.T) {
	ran := 0
	// Ragged counts exercise the tile/tail split, multiples of PackedWideGroup
	// the wide kernel, and the model shapes the (rows, k) combinations real
	// decodes use.
	type shape struct{ rows, k int }
	shapes := []shape{
		{200, 512}, {72, 256}, {8, 768}, {128, 1024}, {256, 1280}, {2048, 1536},
		// The real models' shapes, which is what a gate on a kernel this
		// engine actually runs has to contain.
		{288, 288}, {768, 288}, {288, 768}, {32000, 288}, // stories15M
		{960, 960}, {2560, 960}, {960, 2560}, // SmolLM2-360M
		// gemma-3-1b, whose k of 1152 is not a multiple of 256 (its weights
		// are Q5_0 rather than a k-quant).
		{256, 1152}, {1024, 1152}, {6912, 1152}, {1152, 1024}, {1152, 6912},
		// Row counts the tail kernel's group does not divide: served by a
		// gathered window, odd and even remainders (the narrow formats pair
		// rows in their d plane), with and without full groups before them.
		{1, 256}, {3, 1024}, {5, 512}, {13, 1152}, {67, 256}, {77, 1152},
	}
	maxK, maxRows := 0, 0
	for _, sh := range shapes {
		maxK = max(maxK, sh.k)
		maxRows = max(maxRows, sh.rows)
	}
	for _, sh := range shapes {
		nrows := sh.rows
		for _, gt := range quant.PackedTypes {
			if !cpu.PackedSupported(gt) {
				// continue, not Skipf: Skipf would end the whole test and hide
				// every format that does have a kernel.
				t.Logf("%s: no packed kernel on this architecture", gt)
				continue
			}
			k := sh.k
			// A shape whose k the format's block does not divide skips the
			// format, never the shape.
			if k%int(gt.BlockElems()) != 0 {
				continue
			}
			rng := rand.New(rand.NewSource(int64(nrows)))
			q, _ := kernels.QuantOf(gt)
			nb := uint64(nrows*k) / gt.BlockElems()
			bb := gt.BlockBytes()
			src := make([]byte, nb*bb)
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
			// Random bytes are not a valid block (a random f16 is Inf or NaN
			// about 1 time in 32). Scales must also vary per block -- a
			// constant scale plane hides a wrong-row or wrong-half scale read
			// -- and span subnormal to largest finite. quant.PlantScales does
			// both, and a format it does not know is a failure.
			if !quant.PlantScales(gt, src, 0) {
				t.Fatalf("%s: no scale planter, so its blocks would carry random "+
					"scales -- add it rather than testing noise", gt)
			}
			qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
			if err != nil {
				t.Fatal(err)
			}
			x := make([]float32, k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			// Sized like a model's JIT (the widest k and rows), not per case:
			// model.Open builds one JIT and serves every shape from it.
			j := NewJIT(maxK, maxRows, []quant.Type{gt}, WithTune(TuneOff)) // exact: no timed per-shape kernel choice
			if j == nil {
				t.Fatal("no JIT")
			}
			defer j.Close()
			// The output is poisoned: the kernels accumulate into out and the
			// caller owns the zero, which a fresh make() would hide.
			got := make([]float32, nrows)
			for i := range got {
				got[i] = float32(1e3 * (float64(i%7) - 3))
			}
			pk := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
			if !j.MatVecPacked(got, gt, pk, x, nrows, k) {
				t.Fatalf("%s nrows=%d: declined", gt, nrows)
			}
			want := make([]float32, nrows)
			if err := oracle.MatVecPacked(want, q, pk.QS, pk.D, pk.SC, x, nrows, k); err != nil {
				t.Fatal(err)
			}
			// Called twice into the same buffer: a missing clear gives
			// previous contents plus the right answer, which only a second
			// call can see.
			again := make([]float32, nrows)
			copy(again, got)
			if !j.MatVecPacked(got, gt, pk, x, nrows, k) {
				t.Fatalf("%s nrows=%d: declined on the second call", gt, nrows)
			}
			for r := 0; r < nrows; r++ {
				if got[r] != again[r] {
					t.Fatalf("%s %dx%d: NOT IDEMPOTENT -- row %d went %v -> %v on a "+
						"second call into the same buffer, so the kernel accumulates "+
						"into a buffer nothing cleared", gt, nrows, k, r, again[r], got[r])
				}
			}

			var num, den float64
			for r := 0; r < nrows; r++ {
				dv := float64(got[r]) - float64(want[r])
				num += dv * dv
				den += float64(want[r]) * float64(want[r])
			}
			if den == 0 || math.IsNaN(num) || math.IsNaN(den) {
				t.Fatalf("%s nrows=%d: degenerate oracle -- this gate proved nothing", gt, nrows)
			}
			ran++
			// Not below one group of rows: with int8 activations one dot with
			// cancellation can exceed 1e-3 alone. The rows of a ragged count
			// are held to bit equality just below.
			if nmse := num / den; nmse > 1e-3 && nrows >= cpu.PackedTail {
				t.Fatalf("%s nrows=%d: NMSE %.3e (%d main tile(s) + %d tail)",
					gt, nrows, nmse, nrows/cpu.PackedRows,
					(nrows%cpu.PackedRows)/cpu.PackedTail)
			}
			// A ragged remainder is served by a gathered window, which must
			// agree to the bit with the same rows as the first rows of a
			// one-group tensor run through the same JIT.
			if rem := nrows % cpu.PackedTail; rem != 0 {
				lo := nrows - rem
				rowB := int(uint64(k) / gt.BlockElems() * bb)
				one := make([]byte, cpu.PackedTail*rowB)
				copy(one, src[lo*rowB:nrows*rowB])
				for i := rem * rowB; i < len(one); i++ {
					one[i] = byte(rng.Intn(256))
				}
				quant.PlantScales(gt, one, 0)
				copy(one, src[lo*rowB:nrows*rowB]) // the planter touched these too
				qs1, d1, sc1, err := kernels.PackWeights(q, one, cpu.PackedTail, k)
				if err != nil {
					t.Fatal(err)
				}
				grp := make([]float32, cpu.PackedTail)
				if !j.MatVecPacked(grp, gt, &Packed{QS: u32b(qs1), D: u32b(d1), SC: u32b(sc1)}, x, cpu.PackedTail, k) {
					t.Fatalf("%s: the one-group tensor declined", gt)
				}
				for r := 0; r < rem; r++ {
					if math.Float32bits(got[lo+r]) != math.Float32bits(grp[r]) {
						t.Fatalf("%s nrows=%d: row %d is %v through the window and %v in place",
							gt, nrows, lo+r, got[lo+r], grp[r])
					}
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no format had a packed kernel -- this gate proved nothing")
	}
}
