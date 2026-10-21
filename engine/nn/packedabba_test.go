//go:build jitllmbench && (amd64 || arm64)

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a t.Skip that would put a green line in every run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// Any JITLLM_* variable the test reads still selects its parameters.

package nn

import (
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestPackedBatchABBA prices MatMulPacked against the per-row loop it replaces,
// on the shapes a SmolVLM tower actually runs, paired and interleaved in one
// process with the A/A self-control first.
func TestPackedBatchABBA(t *testing.T) {
	// The emitters are host-pure: EmitGEMM succeeds for a target this CPU
	// cannot run, so "it emitted" is not "it executes" and picking a tile by
	// emitter success walks straight into a SIGILL on a pre-VNNI host.
	requireRowMajorHost(t, quant.Q8_0)
	for _, sh := range []struct {
		what           string
		nrows, k, ntok int
	}{
		{"q/k/v/o  768x768", 768, 768, 1024},
		{"fc1     3072x768", 3072, 768, 1024},
		{"fc2     768x3072", 768, 3072, 1024},
	} {
		gt := quant.Q8_0
		nrows, k, ntok := sh.nrows, sh.k, sh.ntok
		rng := rand.New(rand.NewSource(1))
		q, _ := kernels.QuantOf(gt)
		nb := uint64(nrows*k) / gt.BlockElems()
		bb := gt.BlockBytes()
		src := make([]byte, nb*bb)
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		for b := uint64(0); b < nb; b++ {
			src[b*bb], src[b*bb+1] = 0x00, 0x3C
		}
		qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
		if err != nil {
			t.Fatal(err)
		}
		pk := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
		x := make([]float32, ntok*k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		j := NewJIT(k, nrows, []quant.Type{gt})
		if j == nil {
			t.Fatal("no JIT")
		}
		out := make([]float32, ntok*nrows)

		batch := func() time.Duration {
			s := time.Now()
			j.MatMulPacked(out, gt, pk, x, nrows, k, ntok)
			return time.Since(s)
		}
		loop := func() time.Duration {
			s := time.Now()
			for i := 0; i < ntok; i++ {
				j.NewInput()
				j.MatVecPacked(out[i*nrows:(i+1)*nrows], gt, pk, x[i*k:(i+1)*k], nrows, k)
			}
			return time.Since(s)
		}
		// Soak to thermal steady state before the first sample.
		for i := 0; i < 3; i++ {
			batch()
			loop()
		}
		run := func(name string, A, B func() time.Duration) (float64, float64) {
			const n = 24
			r := make([]float64, 0, n)
			for i := 0; i < n; i++ {
				a := A()
				b := B()
				r = append(r, float64(b)/float64(a)) // >1 means A is faster
			}
			sort.Float64s(r)
			med := r[len(r)/2]
			iqr := r[3*len(r)/4] - r[len(r)/4]
			t.Logf("  %-22s %s  ratio %.4f  IQR/med %.3f  n=%d", sh.what, name, med, iqr/med, n)
			return med, iqr / med
		}
		// The GGUF GEMM is here as a yardstick, not a path: it runs on src,
		// the same weight before packing, to price the token-tiled packed
		// kernels against it. Nothing in the engine may reach it with container
		// bytes.
		gout := make([]float32, ntok*nrows)
		gemm := func() time.Duration {
			s := time.Now()
			if !j.MatMul(gout, gt, src, x, nrows, k, ntok) {
				t.Fatalf("%s: the GGUF GEMM declined nrows=%d k=%d ntok=%d", sh.what, nrows, k, ntok)
			}
			return time.Since(s)
		}
		for i := 0; i < 3; i++ {
			gemm()
		}
		mac := float64(nrows) * float64(k) * float64(ntok)
		t.Logf("  %-22s %.1f Gmac | batch %.1f Gmac/s | gemm %.1f Gmac/s", sh.what, mac/1e9,
			mac/batch().Seconds()/1e9, mac/gemm().Seconds()/1e9)
		run("A/A self-control", batch, batch)
		run("batch / GGUF gemm", gemm, batch)
		med, rel := run("loop / batch", batch, loop)
		if rel > 0.10 {
			t.Logf("  %-22s REJECTED at IQR/median %.3f", sh.what, rel)
		} else {
			t.Logf("  %-22s => the batch is %.2fx the row loop", sh.what, med)
		}
		j.Close()
	}
}
