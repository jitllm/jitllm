//go:build amd64

package nn

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
)

// TestPackedGEMMRate prices the weight-stationary GEMM alone, on one worker,
// so a kernel change can be judged without a model around it: GMAC/s at a
// real projection shape, for a grid of tokens x rows per call. It is an
// instrument (JITLLM_GEMM_RATE=1), not a gate, and prints the ratio against
// the per-token fused kernel measured in the same process.
func TestPackedGEMMRate(t *testing.T) {
	if os.Getenv("JITLLM_GEMM_RATE") == "" {
		t.Skip("set JITLLM_GEMM_RATE=1 to price the weight-stationary GEMM")
	}
	gt := quant.Q4_K
	if s := os.Getenv("JITLLM_GEMM_RATE_TYPE"); s != "" {
		for _, q := range quant.PackedTypes() {
			if q.String() == s {
				gt = q
			}
		}
	}
	nrows, k, ntok := 2048, 2048, 128
	if s := os.Getenv("JITLLM_GEMM_RATE_K"); s != "" {
		k, _ = strconv.Atoi(s)
	}
	pk := packOne(t, gt, nrows, k)
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, ntok*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	cores := 1
	if s := os.Getenv("JITLLM_GEMM_RATE_CORES"); s != "" {
		cores, _ = strconv.Atoi(s)
	}
	if s := os.Getenv("JITLLM_GEMM_RATE_ROWS"); s != "" {
		nrows, _ = strconv.Atoi(s)
		pk = packOne(t, gt, nrows, k)
	}
	out := make([]float32, ntok*nrows)
	rate := func(opts ...Option) float64 {
		j := NewJIT(k, nrows, []quant.Type{gt}, append(opts, WithSched(sched.WithCores(cores)))...)
		defer j.Close()
		j.MatMulPacked(out, gt, pk, x, nrows, k, ntok) // emit and warm
		best := 0.0
		for r := 0; r < 5; r++ {
			t0 := time.Now()
			j.MatMulPacked(out, gt, pk, x, nrows, k, ntok)
			g := float64(nrows) * float64(k) * float64(ntok) / time.Since(t0).Seconds() / 1e9
			best = max(best, g)
		}
		return best
	}
	if s := os.Getenv("JITLLM_GEMM_RATE_ONLY"); s != "" {
		var T, R int
		fmt.Sscanf(s, "%d,%d", &T, &R)
		for r := 0; r < 20; r++ {
			rate(WithGEMMTokens(T), WithGEMMRows(R))
		}
		fmt.Printf("  T=%-3d R=%-4d %6.1f GMAC/s\n", T, R, rate(WithGEMMTokens(T), WithGEMMRows(R)))
		return
	}
	base := rate(WithGEMMTokens(-1))
	fmt.Printf("%s %dx%d ntok=%d %d worker(s): row loop / tiled %.1f GMAC/s\n", gt, nrows, k, ntok, cores, base)
	for _, T := range []int{8, 16, 32, 64} {
		for _, R := range []int{16, 64, 256} {
			g := rate(WithGEMMTokens(T), WithGEMMRows(R))
			fmt.Printf("  T=%-3d R=%-4d %6.1f GMAC/s  %.2fx\n", T, R, g, g/base)
		}
	}
}
