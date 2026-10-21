//go:build amd64 || arm64

package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestGEMMExactDecodeSumsTheWholeK holds the decode matvec under GEMMExact to
// the batched path bit for bit at a shape the pool splits over k.
//
// fusedSlices decides from the pool's width: a matrix of 32M weights or more
// whose balanced row chunk is short is summed as two k-halves added together.
// That is a different order from the batched path, which sums every row over
// the whole of k, so a batch decoded against its single-token path differs at
// NMSE ~1e-14 on a 14-worker two-socket pool (a 32000-row head splits there)
// and by 0 on a six-worker one (it does not).
// fusedSlices splits only on a pool of even width, so both JITs run the host's
// pool rounded down to even, and the shape -- 32M weights, 4096 rows -- leaves
// a balanced chunk under fusedSliceRows from two workers up: the split is
// reached on every host with two cores or more, and the control proves it was.
func TestGEMMExactDecodeSumsTheWholeK(t *testing.T) {
	const ntok = 2
	workers := len(sched.DecodeCores()) / 2 * 2
	if workers < 2 {
		t.Skipf("a %d-core pool never splits over k -- this gate proved nothing here", len(sched.DecodeCores()))
	}
	pool := WithSched(sched.WithCores(workers))
	for _, gt := range []quant.Type{quant.Q4_K, quant.Q6_K, quant.Q8_0} {
		t.Run(gt.String(), func(t *testing.T) {
			// 4096 x 8192 is 32M weights; at two workers or more a balanced
			// chunk is 2048 rows or fewer, under fusedSliceRows.
			nrows, k := 4096, 8192
			rng := rand.New(rand.NewSource(int64(gt) + 17))
			q, _ := kernels.QuantOf(gt)
			src := make([]byte, uint64(nrows*k)/gt.BlockElems()*gt.BlockBytes())
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
			if !quant.PlantScales(gt, src, 0) {
				t.Fatalf("%s: no scale planter", gt)
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

			// The control: the same shape left to decide does split, or this
			// gate compares the whole-k matvec with itself. On NEON the split
			// is the timed pick's (mvpick.go), which tuning off pins to the
			// tiled kernel, so the control pins the two-slice arm there.
			ctl := []Option{WithTune(TuneOff), pool}
			if cpu.HostTier() == cpu.TierNEON {
				ctl = append(ctl, WithMatVecPick(3))
			}
			free := NewJIT(k, nrows, []quant.Type{gt}, ctl...)
			defer free.Close()
			if !free.MatVecPacked(make([]float32, nrows), gt, pk, x[:k], nrows, k) {
				t.Fatalf("%s: the matvec declined", gt)
			}
			if free.KSliced() == 0 {
				t.Fatalf("%s %dx%d on a %d-worker pool was not split over k: the shape does not reach "+
					"fusedSlices' split here, so this gate proves nothing on this host", gt, nrows, k,
					free.Workers())
			}

			j := NewJIT(k, nrows, []quant.Type{gt}, WithTune(TuneOff), WithGEMMExact(true), pool)
			defer j.Close()
			batched := make([]float32, ntok*nrows)
			if !j.MatMulPacked(batched, gt, pk, x, nrows, k, ntok) {
				t.Fatalf("%s: the batch declined", gt)
			}
			want := make([]float32, ntok*nrows)
			for i := range ntok {
				j.NewInput()
				if !j.MatVecPacked(want[i*nrows:(i+1)*nrows], gt, pk, x[i*k:(i+1)*k], nrows, k) {
					t.Fatalf("%s: the matvec declined", gt)
				}
			}
			if n := j.KSliced(); n != 0 {
				t.Errorf("%s: GEMMExact split %d matvec(s) over k", gt, n)
			}
			diffs, finite := 0, 0
			for i := range want {
				if !math.IsNaN(float64(want[i])) && !math.IsInf(float64(want[i]), 0) {
					finite++
				}
				if batched[i] != want[i] {
					diffs++
				}
			}
			if finite == 0 {
				t.Fatalf("%s: every reference value is non-finite -- a degenerate oracle", gt)
			}
			t.Logf("%s %dx%d, %d workers: the free JIT split %d matvec(s); under GEMMExact %d of %d "+
				"elements differ from the batch", gt, nrows, k, free.Workers(), free.KSliced(), diffs, len(want))
			if diffs != 0 {
				t.Errorf("%s: %d of %d elements of decode differ from the batch under GEMMExact", gt, diffs,
					len(want))
			}
		})
	}
}
