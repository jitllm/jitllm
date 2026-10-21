//go:build amd64

package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestPrefetchDistanceIsPickedPerShape: on x86 each shape times the fused
// kernel's prefetch distances in place and keeps the fastest (JIT.distPick).
// A prefetch changes no arithmetic, so every call must equal a JIT pinned at
// distance 0 bit for bit. The selection check is that every distance's kernel
// was actually emitted and run -- a chooser whose arm the call ignores would
// leave only the default -- and that the larger shape's choice is what the
// batched paths run (packedFused).
func TestPrefetchDistanceIsPickedPerShape(t *testing.T) {
	gt := quant.Q4_K
	if !cpu.PackedFusedSupported(gt) {
		t.Skip("no fused packed kernel here")
	}
	step := cpu.PackedOuterElems(gt)
	type shape struct{ rows, k int }
	shapes := []shape{{512, 4 * step}, {256, 8 * step}}
	maxK, maxRows := 0, 0
	for _, sh := range shapes {
		maxK, maxRows = max(maxK, sh.k), max(maxRows, sh.rows)
	}
	j := NewJIT(maxK, maxRows, []quant.Type{gt})
	defer j.Close()
	ref := NewJIT(maxK, maxRows, []quant.Type{gt}, WithFusedPrefetch(0))
	defer ref.Close()
	// The probe decides which half this host runs: a tier whose fused kernel
	// has no prefetch form (SSE) must not pick at all, and its matvec is the
	// one fused kernel.
	if _, err := cpu.EmittersFor(cpu.HostTier()).PackedFusedAhead(gt, cpu.FusedAheadWords); err != nil {
		if j.distPick() {
			t.Fatalf("tier %v has no prefetch form (%v), yet the JIT picks a distance per shape", cpu.HostTier(), err)
		}
		for si, sh := range shapes {
			p := packSheet(t, gt, sh.rows, sh.k, si+1)
			x := make([]float32, sh.k)
			for i := range x {
				x[i] = float32(math.Sin(float64(i)))
			}
			want := make([]float32, sh.rows)
			ref.NewInput()
			if !ref.MatVecPacked(want, gt, p, x, sh.rows, sh.k) {
				t.Fatal("reference declined")
			}
			for c := 0; c < pickWarm+pickDistRounds+1; c++ {
				got := poison(sh.rows)
				j.NewInput()
				if !j.MatVecPacked(got, gt, p, x, sh.rows, sh.k) {
					t.Fatal("declined")
				}
				for r := range got {
					if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
						t.Fatalf("%dx%d call %d row %d: %v, want %v", sh.rows, sh.k, c, r, got[r], want[r])
					}
				}
			}
		}
		if pk := j.picks[pickKey{gt, shapes[0].rows, shapes[0].k, shapes[0].rows}]; pk != nil && pk.dist {
			t.Fatalf("tier %v timed prefetch distances it cannot emit: %+v", cpu.HostTier(), pk)
		}
		for d, m := range j.fusedAt {
			if d != 0 && m[gt] != nil {
				t.Fatalf("tier %v emitted a distance-%d kernel it refuses", cpu.HostTier(), d)
			}
		}
		t.Logf("tier %v: no prefetch form, no distance picked, the one fused kernel ran (%v)", cpu.HostTier(), err)
		return
	}
	if !j.distPick() || ref.distPick() {
		t.Fatalf("distPick: default %v, pinned %v", j.distPick(), ref.distPick())
	}
	calls := (pickWarm + pickDistRounds + 1) * len(prefetchDistances)
	for si, sh := range shapes {
		p := packSheet(t, gt, sh.rows, sh.k, si+1)
		x := make([]float32, sh.k)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)))
		}
		want := make([]float32, sh.rows)
		ref.NewInput()
		if !ref.MatVecPacked(want, gt, p, x, sh.rows, sh.k) {
			t.Fatal("reference declined")
		}
		for c := 0; c < calls; c++ {
			got := poison(sh.rows)
			j.NewInput()
			if !j.MatVecPacked(got, gt, p, x, sh.rows, sh.k) {
				t.Fatal("declined")
			}
			for r := range got {
				if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
					t.Fatalf("%dx%d call %d row %d: %v, pinned distance 0 gives %v", sh.rows, sh.k, c, r, got[r], want[r])
				}
			}
		}
		pk := j.picks[pickKey{gt, sh.rows, sh.k, sh.rows}]
		if pk == nil || !pk.dist || pk.best < 0 || !pk.applied {
			t.Fatalf("%dx%d: the chooser did not settle and apply: %+v", sh.rows, sh.k, pk)
		}
		t.Logf("%dx%d settled on distance %d", sh.rows, sh.k, pk.best)
	}
	for _, d := range prefetchDistances {
		if j.fusedAt[d][gt] == nil {
			t.Fatalf("distance %d was never emitted, so it never ran", d)
		}
	}
	big := j.picks[pickKey{gt, shapes[0].rows, shapes[0].k, shapes[0].rows}]
	if j.packedFused[gt] != j.fusedAt[big.best][gt] {
		t.Fatalf("packedFused is not the larger shape's choice (distance %d)", big.best)
	}
	// The rule itself, with distances the timing did not choose: the largest
	// shape sets the default and a smaller one cannot move it.
	j.settleDist(gt, 1<<30, 8)
	j.settleDist(gt, 1<<20, 2)
	if j.packedFused[gt] != j.fusedAt[8][gt] {
		t.Fatal("a smaller shape moved packedFused off the largest shape's distance")
	}
}
