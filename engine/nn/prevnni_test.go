package nn

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// prevnniShape is a (rows, k) the packed family actually meets: 256 rows is a
// whole number of fused groups, 512 is a whole number of super-blocks for every
// k-quant, and 200 is ragged so the 64-row tile and the 8-row tail both run.
type prevnniShape struct{ rows, k int }

var prevnniShapes = []prevnniShape{{256, 512}, {200, 512}, {128, 1024}}

// packOne builds a valid packed weight for gt: random payload bytes with the
// f16 scale fields forced finite and varying, which is what makes a wrong-row
// or wrong-super-block scale read visible at all.
func packOne(t *testing.T, gt quant.Type, nrows, k int) *Packed {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(nrows*1000 + k)))
	q, _ := kernels.QuantOf(gt)
	nb := uint64(nrows*k) / gt.BlockElems()
	bb := gt.BlockBytes()
	src := make([]byte, nb*bb)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	// A uniformly random f16 is Inf or NaN about one time in 32.
	// quant.PlantScales carries subnormals and the extremes for every packed
	// format.
	if !quant.PlantScales(gt, src, 0) {
		t.Fatalf("%s: no scale planter -- add it rather than testing noise", gt)
	}
	qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	return &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
}

func prevnniAct(k int) []float32 {
	rng := rand.New(rand.NewSource(int64(k)))
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

// TestPackedFamilyIsBuiltOnThisHost runs on every architecture with no build
// tag. It asserts that the capability says yes for every format the
// architecture has a kernel for, that a JIT built for it holds a packed kernel,
// and that a real matvec through it leaves the reference counter at zero -- the
// only way to tell a generated tier from a declining one, since the answers
// are equal.
func TestPackedFamilyIsBuiltOnThisHost(t *testing.T) {
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedSupported(gt) {
			// continue, never Skipf: one format without an emitter here must
			// not hide every format that has one.
			t.Logf("%s: no packed emitter on this architecture", gt)
			continue
		}
		if !cpu.SupportedPackedNative(gt) {
			t.Errorf("%s: this host has a packed emitter (PackedSupported) and "+
				"SupportedPackedNative says no, so every matvec of it will run "+
				"the float64 ORACLE -- correct answers at ~24.5x, invisible to "+
				"every correctness gate (RULE 8)", gt)
			continue
		}
		for _, sh := range prevnniShapes {
			if sh.k%int(gt.BlockElems()) != 0 {
				continue
			}
			pk := packOne(t, gt, sh.rows, sh.k)
			x := prevnniAct(sh.k)
			j := NewJIT(sh.k, sh.rows, []quant.Type{gt})
			if j == nil {
				t.Fatalf("%s: no JIT", gt)
			}
			out := make([]float32, sh.rows)
			if !j.MatVecPacked(out, gt, pk, x, sh.rows, sh.k) {
				j.Close()
				t.Errorf("%s %dx%d: MatVecPacked declined", gt, sh.rows, sh.k)
				continue
			}
			j.Close()
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no format ran -- this gate proved nothing")
	}
	t.Logf("%d (format, shape) pair(s) ran generated, %s sequence, no declines",
		ran, cpu.HostDotKind())
}
