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

// TestMatMulPackedMatchesTheRowLoopExactly gates the batched packed path with
// bit equality against the per-token loop: the arithmetic is unchanged by
// construction, and a tolerance would hide a wrong activation offset, a wrong
// output slice or a shared scratch.
func TestMatMulPackedMatchesTheRowLoopExactly(t *testing.T) {
	compared := 0
	// Q8_0 is every matrix of a SmolVLM tower; the k-quants cover the half-sums
	// path, which is a second activation buffer and the likeliest thing to get
	// wrong per token.
	for _, gt := range quant.PackedTypes {
		for _, sh := range []struct{ nrows, k, ntok int }{
			{768, 768, 8},  // the tower's q/k/v/o shape
			{3072, 768, 5}, // fc1, and an odd token count
			{768, 3072, 3}, // fc2
			{128, 512, 2},  // the minimum batch
			{256, 512, 37}, // a token count that divides nothing
		} {
			if !cpu.PackedSupported(gt) {
				t.Fatalf("%s lost its packed kernel", gt)
			}
			nrows, k, ntok := sh.nrows, sh.k, sh.ntok
			rng := rand.New(rand.NewSource(int64(nrows*31 + k*7 + ntok)))
			q, _ := kernels.QuantOf(gt)
			nb := uint64(nrows*k) / gt.BlockElems()
			bb := gt.BlockBytes()
			src := make([]byte, nb*bb)
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
			// A uniformly random f16 is Inf or NaN about 1 time in 32.
			if !quant.PlantScales(gt, src, 0) {
				t.Fatalf("%s: no scale planter, so its blocks would carry "+
					"random scales -- add it rather than testing noise", gt)
			}
			qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
			if err != nil {
				t.Fatal(err)
			}
			pk := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}

			// Every token gets a different activation:
			// one shared row would pass even if the batch read token 0 ntok times.
			x := make([]float32, ntok*k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}

			// The weight-stationary GEMM is held off here: this gate covers the
			// tiled and per-token paths behind it, which a pre-VNNI host or a
			// declined shape still takes. TestPackedGEMMMatchesTheRowLoopExactly
			// gates the GEMM itself.
			j := NewJIT(k, nrows, []quant.Type{gt}, WithGEMMTokens(-1), WithTune(TuneOff), WithGEMMExact(true)) // exact: no timed per-shape kernel choice, the float epilogue
			if j == nil {
				t.Fatal("no JIT")
			}
			batched := make([]float32, ntok*nrows)
			if !j.MatMulPacked(batched, gt, pk, x, nrows, k, ntok) {
				// A decline is logged, not passed; the count at the end fails
				// if every shape declined.
				j.Close()
				t.Logf("%s nrows=%d k=%d ntok=%d: the batch declined on this architecture",
					gt, nrows, k, ntok)
				continue
			}
			// The reference is the path it replaces, run exactly as vision.go
			// ran it: NewInput per row, because the activation cache is keyed by
			// generation and each token is a different vector.
			want := make([]float32, ntok*nrows)
			for i := 0; i < ntok; i++ {
				j.NewInput()
				if !j.MatVecPacked(want[i*nrows:(i+1)*nrows], gt, pk, x[i*k:(i+1)*k], nrows, k) {
					j.Close()
					t.Fatalf("%s: the row loop declined", gt)
				}
			}
			// A decline gives the same answer, so assert the batch actually ran
			// before comparing anything.
			if n := j.PackedBatchCalls(); n != 1 {
				j.Close()
				t.Fatalf("%s nrows=%d k=%d ntok=%d: the batch ran %d times, want 1 "+
					"-- the gate would be comparing the row loop with itself", gt, nrows, k, ntok, n)
			}
			// And the tiled kernel separately, since its decline into the
			// per-token path is also invisible to bit equality. One-sided
			// because the tile may narrow below the register maximum (the L1i
			// budget): at or above that maximum it must have run, and below two
			// tokens it must not.
			ran := j.TiledMatMulCalls() > 0
			maxTok := cpu.MaxTiledTokensNative(gt)
			switch {
			case maxTok >= 2 && ntok >= maxTok && !ran:
				j.Close()
				t.Fatalf("%s nrows=%d k=%d ntok=%d: the tile did NOT run at ntok >= maxTok=%d",
					gt, nrows, k, ntok, maxTok)
			case ntok < 2 && ran:
				j.Close()
				t.Fatalf("%s nrows=%d k=%d ntok=%d: the tile ran below two tokens", gt, nrows, k, ntok)
			}
			j.Close()
			compared++
			var diffs, finite int
			worst := 0.0
			for i := range want {
				if !math.IsNaN(float64(want[i])) && !math.IsInf(float64(want[i]), 0) {
					finite++
				}
				if batched[i] != want[i] {
					diffs++
					if d := math.Abs(float64(batched[i]) - float64(want[i])); d > worst {
						worst = d
					}
				}
			}
			if finite == 0 {
				t.Fatalf("%s nrows=%d k=%d ntok=%d: every reference value is "+
					"non-finite -- a degenerate oracle is a failure, not a pass",
					gt, nrows, k, ntok)
			}
			if diffs != 0 {
				t.Errorf("%s nrows=%d k=%d ntok=%d: %d of %d elements differ, worst |d| %g "+
					"-- the batch is not running the same arithmetic",
					gt, nrows, k, ntok, diffs, len(want), worst)
			}
		}
	}
	if compared == 0 {
		t.Fatal("the batch declined every shape -- this gate proved nothing")
	}
}

// TestMatMulPackedDeclinesRatherThanApproximating: a caller that gets false
// loops MatVecPacked and gets exactly what it had, so every decline must be a
// decline and never a wrong answer written into out.
func TestMatMulPackedDeclinesRatherThanApproximating(t *testing.T) {
	const nrows, k, ntok = 768, 768, 4
	gt := quant.Q8_0
	q, _ := kernels.QuantOf(gt)
	nb := uint64(nrows*k) / gt.BlockElems()
	src := make([]byte, nb*gt.BlockBytes())
	for b := uint64(0); b < nb; b++ {
		src[b*gt.BlockBytes()], src[b*gt.BlockBytes()+1] = 0x00, 0x3C
	}
	qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	pk := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
	x := make([]float32, ntok*k)
	j := NewJIT(k, nrows, []quant.Type{gt})
	if j == nil {
		t.Fatal("no JIT")
	}
	defer j.Close()
	out := make([]float32, ntok*nrows)
	for _, c := range []struct {
		what           string
		nrows, k, ntok int
		out, x         []float32
	}{
		{"one token is not a batch", nrows, k, 1, out, x},
		{"nrows not a multiple of the fused group", nrows + 8, k, ntok, out, x},
		{"out too small", nrows, k, ntok, out[:nrows], x},
		{"x too small", nrows, k, ntok, out, x[:k]},
	} {
		if j.MatMulPacked(c.out, gt, pk, c.x, c.nrows, c.k, c.ntok) {
			t.Errorf("%s: accepted, and the caller would trust the result", c.what)
		}
	}
	if j.MatMulPacked(nil, gt, nil, x, nrows, k, ntok) {
		t.Error("a nil weight was accepted")
	}
}
