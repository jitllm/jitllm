package nn

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestFusedPrefetchIsSelectedAndChangesNothing pins each prefetch distance the
// duel can choose and demands two things: the kernel the JIT dispatches really
// is that distance's (a pin that never reaches packedFused would make every
// arm of a duel the same kernel), and the answer is bit-identical to no
// prefetch, since a prefetch changes no arithmetic.
func TestFusedPrefetchIsSelectedAndChangesNothing(t *testing.T) {
	const nrows, k = 512, 2048
	if cpu.HostTier() == cpu.TierNEON {
		t.Skip("arm64 runs no prefetch duel: its fused kernel carries a two-group prefetch " +
			"(cpu.EmitPackedMatVecFused) and MatVecPacked times each shape instead (mvpick.go)")
	}
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) || k%int(gt.BlockElems()) != 0 {
			continue
		}
		plain, err := cpu.Native().PackedFused(gt)
		if err != nil {
			continue
		}
		if _, err := cpu.Native().PackedFusedAhead(gt, 4); err != nil {
			continue
		}
		rng := rand.New(rand.NewSource(7))
		q, _ := kernels.QuantOf(gt)
		src := make([]byte, uint64(nrows*k)/gt.BlockElems()*gt.BlockBytes())
		rng.Read(src)
		if !quant.PlantScales(gt, src, 0) {
			t.Fatalf("%s: no scale planter", gt)
		}
		qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
		if err != nil {
			t.Fatal(err)
		}
		pk := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		var want []float32
		for _, dist := range prefetchDistances {
			if dist > 0 {
				b, err := cpu.Native().PackedFusedAhead(gt, dist)
				if err != nil || bytes.Equal(b, plain) {
					t.Fatalf("%s: distance %d emitted the plain kernel (err %v)", gt, dist, err)
				}
			}
			j := NewJIT(k, nrows, []quant.Type{gt}, WithFusedPrefetch(dist), WithTune(TuneOff), WithQuietTuner(true))
			j.TokenStart()
			j.TokenEnd()
			if j.fpf == nil || j.fpf.cur != dist || j.packedFused[gt] != j.fusedAt[dist][gt] {
				t.Fatalf("%s: pinned distance %d is not what dispatches", gt, dist)
			}
			got := make([]float32, nrows)
			if !j.MatVecPacked(got, gt, pk, x, nrows, k) {
				t.Fatalf("%s: declined", gt)
			}
			j.Close()
			if want == nil {
				want = got
				continue
			}
			for r := range got {
				if got[r] != want[r] {
					t.Fatalf("%s distance %d: row %d is %v, %v without prefetch", gt, dist, r, got[r], want[r])
				}
			}
		}
		ran++
	}
	if ran == 0 {
		t.Skip("no fused packed kernel with a prefetch form on this host")
	}
}
