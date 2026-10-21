//go:build amd64 || arm64

package nn

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestMatVecPickArmsGiveTheirKernelsAnswer pins each of MatVecPacked's
// per-shape candidates (mvpick.go) and checks what it computed. The unsliced
// arms -- tiled, fused over whole k -- are the same sums in the same order, so
// they must match the tiled kernel bit for bit. A k-sliced arm adds partial
// sums afterwards, so it must be close AND differ somewhere: that half is the
// selection check, because an arm that silently ran whole k passes any bound.
// Then, where shapes are timed (arm64), the chooser must settle on one of them.
func TestMatVecPickArmsGiveTheirKernelsAnswer(t *testing.T) {
	gt := quant.Q4_K
	if !cpu.PackedFusedSupported(gt) {
		t.Skip("no fused packed kernel here, so there is nothing to choose between")
	}
	rows, k := 256, 8*cpu.PackedOuterElems(gt)
	pk := packOne(t, gt, rows, k)
	x := prevnniAct(k)
	run := func(pick int) ([]float32, *JIT) {
		j := NewJIT(k, rows, []quant.Type{gt}, WithMatVecPick(pick), WithGEMMExact(true)) // the float forms
		out := make([]float32, rows)
		if !j.MatVecPacked(out, gt, pk, x, rows, k) {
			t.Fatalf("pick %d declined", pick)
		}
		return out, j
	}
	ref, j0 := run(1)
	n := j0.pool.N()
	j0.Close()
	ran := 0
	for i, s := range pickArms[1:] {
		if s > 1 && n%s != 0 {
			continue
		}
		out, j := run(i + 2)
		j.Close()
		same, num, den := true, 0.0, 0.0
		for r := range out {
			if math.Float32bits(out[r]) != math.Float32bits(ref[r]) {
				same = false
			}
			d := float64(out[r] - ref[r])
			num, den = num+d*d, den+float64(ref[r])*float64(ref[r])
		}
		// Without VNNI the two kernels fold a format's minimum term in a
		// different order, so they agree to rounding rather than bit for bit
		// (Q4_K: rows differ at ~3e-6 relative, both at the oracle's NMSE).
		exact := cpu.HostDotKind() == cpu.DotVNNI
		switch {
		case s == 1 && !same && (exact || !(num/den < 1e-10)):
			t.Fatalf("the fused arm (whole k) differs from the tiled kernel: NMSE %.3e", num/den)
		case s > 1 && (same || !(num/den < 1e-10)):
			t.Fatalf("%d k-slices: bit-identical %v, NMSE %.3e -- it did not slice, or sliced wrong", s, same, num/den)
		}
		ran++
	}
	// A sliced arm runs only on a pool its slice count divides; a pool no
	// count divides (five workers) has the whole-k arm alone.
	want := 1
	for _, s := range pickArms[2:] {
		if n%s == 0 {
			want = 2
		}
	}
	if ran < want {
		t.Fatalf("only %d fused arm(s) ran on a %d-worker pool", ran, n)
	}

	j := NewJIT(k, rows, []quant.Type{gt})
	defer j.Close()
	if j.tier != cpu.TierNEON {
		// x86 times prefetch distances per shape, not kernels; see
		// TestPrefetchDistanceIsPickedPerShape. A tier whose fused kernel has
		// no prefetch form (SSE) has nothing to time.
		if _, err := cpu.EmittersFor(j.tier).PackedFusedAhead(gt, cpu.FusedAheadWords); err != nil {
			if p := j.pickFor(gt, rows, k, rows); p != nil {
				t.Fatalf("tier %v has no prefetch form, yet a chooser was made: %+v", j.tier, p)
			}
			return
		}
		if p := j.pickFor(gt, rows, k, rows); p == nil || !p.dist {
			t.Fatalf("tier %v: default chooser %+v, want the prefetch-distance one", j.tier, p)
		}
		return
	}
	for c := 0; c < (pickWarm+pickRounds)*len(pickArms); c++ {
		j.NewInput()
		j.MatVecPacked(make([]float32, rows), gt, pk, x, rows, k)
	}
	p := j.picks[pickKey{gt, rows, k, rows}]
	if p == nil || p.best < 0 || !slices.Contains(p.arms, p.best) {
		t.Fatalf("the chooser did not settle: %+v", p)
	}
	t.Logf("%d arm(s) checked; the chooser settled on %d of %v", ran+1, p.best, p.arms)
}
