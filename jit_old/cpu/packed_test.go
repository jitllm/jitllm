package cpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The kernel is gated against the packed reference here, and the reference
// against the GGUF dequantizer in oldkernels.TestPackedRefMatchesDequant; neither
// alone would catch a wrong layout shared by the packer and reference.
// packedFormats derives from quant.PackedTypes so no format is left out.
var packedFormats = func() (fs []struct {
	name string
	g    quant.Type
	k    kernels.Quant
}) {
	for _, g := range quant.PackedTypes {
		k, ok := kernels.QuantOf(g)
		if !ok {
			panic(g.String() + " is in quant.PackedTypes and has no device layout")
		}
		fs = append(fs, struct {
			name string
			g    quant.Type
			k    kernels.Quant
		}{g.String(), g, k})
	}
	return fs
}()

func u32b(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func TestPackedKernelMatchesReference(t *testing.T) {
	// 48 rows, three tiles: the kernel serves a tile but its stride is the
	// tensor's full width, which a one-tile run cannot tell apart.
	const nrows, k = 192, 512
	ran := 0
	for _, f := range packedFormats {
		t.Run(f.name, func(t *testing.T) {
			if !PackedSupported(f.g) {
				t.Fatalf("%s has no packed kernel -- the table and the emitter disagree", f.name)
			}
			// The host tier's kernel: EmitPackedMatVec with HostDotKind on the
			// primary tier, the SSE tier's tile (same Args) on an SSE host.
			code, err := Native().PackedMatVec(f.g, PackedRows)
			if err != nil {
				t.Fatal(err)
			}
			kern := mustMap(t, code)
			defer kern.Close()

			rng := rand.New(rand.NewSource(7))
			nb := uint64(nrows*k) / f.g.BlockElems()
			bb := f.g.BlockBytes()
			src := make([]byte, nb*bb)
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
			if !quant.PlantScales(f.g, src, 0) {
				t.Fatalf("%s: no scale planter -- add it rather than testing noise", f.name)
			}
			qs, d, sc, err := kernels.PackWeights(f.k, src, nrows, k)
			if err != nil {
				t.Fatal(err)
			}

			x := make([]float32, k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			q8 := make([]int8, k)
			pairs := make([]float32, 2*(k/Q8Block))
			if err := oracle.QuantizeQ8(q8, pairs, x, BiasC(f.g)); err != nil {
				t.Fatal(err)
			}
			// The 16-wide k-quants correct their bias from per-16 sums; the
			// 32-wide ones from the pair array. Both are filled always so a
			// kernel that reads the wrong one faults rather than reading zero.
			half := make([]float32, k/16)
			oracle.QuantizeHalfSums(half, q8, BiasC(f.g))
			scr := make([]byte, PackedScratchBytes)
			if err := PackedScratch(scr); err != nil {
				t.Fatal(err)
			}
			step := PackedOuterElems(f.g)
			if step == 0 || k%step != 0 {
				t.Fatalf("k=%d is not a multiple of %s's outer step %d", k, f.name, step)
			}

			qb, db, scb := u32b(qs), u32b(d), u32b(sc)
			// One call for every tile, which also gates the in-kernel row loop
			// (a kernel that ignored its tile counter would fail here).
			got := make([]float32, nrows)
			args := Args{
				Out: &got[0], W: &qb[0], A: &q8[0], AScale: &pairs[0],
				Rows: int64(nrows / PackedRows), RowStr: int64(nrows * 4),
				K: int64(k / step), Scr: &scr[0], PD: &db[0], AHalf: &half[0],
				// The d plane's super-block stride is not RowStr on a narrow
				// format: two rows share a word there. A harness that leaves it
				// zero pins every super-block to the first one's scales.
				DStr: int64(DSuperBytes(f.g, nrows)),
			}
			if len(scb) > 0 {
				args.PSC = &scb[0]
			}
			kern.Call(&args)

			ref := make([]float32, nrows)
			if err := oracle.MatVecPacked(ref, f.k, qb, db, scb, x, nrows, k); err != nil {
				t.Fatal(err)
			}
			var num, den float64
			for r := 0; r < nrows; r++ {
				dv := float64(got[r]) - float64(ref[r])
				num += dv * dv
				den += float64(ref[r]) * float64(ref[r])
			}
			if den == 0 || math.IsNaN(num) || math.IsNaN(den) {
				t.Fatalf("degenerate oracle (num=%v den=%v) -- this gate proved nothing", num, den)
			}
			// Not a bit comparison: the activations are quantized to int8, so
			// the bound is q8 error, not kernel error.
			if nmse := num / den; nmse > 1e-3 {
				t.Fatalf("%s: NMSE %.3e\n  got  %v\n  want %v", f.name, nmse, got[:8], ref[:8])
			} else {
				t.Logf("%s: NMSE %.2e over %d rows of %d", f.name, nmse, nrows, k)
			}
			ran++
		})
	}
	if ran != len(packedFormats) {
		t.Fatalf("%d of %d formats ran -- this gate proved less than it claims", ran, len(packedFormats))
	}
}
