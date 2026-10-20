package oldkernels_test

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestAttnScoresWarpMatchesTheReference gates the warp-per-(head, position)
// scores kernel against a float64 dot product, at MLA's shape (sixteen heads
// sharing one 576-wide latent row) and at an ordinary GQA shape, over the
// position counts where a clamped grid can go wrong (1, 31, 33) and a ragged
// head width (72 = 2*32 + 8) where the tail lanes must add nothing.
//
// The grid is launched at a capacity above n, as a captured decode graph
// launches it, so the surplus groups' clamp is exercised too; and every score
// row is pre-filled with a sentinel so a pair the kernel never wrote is seen.
func TestAttnScoresWarpMatchesTheReference(t *testing.T) {
	d := dev(t)
	for _, c := range []struct{ nh, hd, kvDim, gqa, n int }{
		{16, 576, 576, 16, 1}, {16, 576, 576, 16, 31}, {16, 576, 576, 16, 80},
		{4, 64, 128, 2, 33}, {4, 72, 144, 2, 5},
	} {
		const maxSeq, capN = 96, 90
		scale := float32(1 / math.Sqrt(float64(c.hd)))
		q := ramp(c.nh*c.hd, func(i int) float32 { return float32(math.Sin(float64(i) * 0.37)) })
		kc := ramp(maxSeq*c.kvDim, func(i int) float32 { return float32(math.Cos(float64(i) * 0.11)) })
		k, err := kernels.AttnScoresWarp(c.nh, c.hd, c.kvDim, c.gqa, maxSeq, scale)
		if err != nil {
			t.Fatal(err)
		}
		kern, err := d.Compile(k)
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		bq, _ := d.Alloc(len(q) * 4)
		bk, _ := d.Alloc(len(kc) * 4)
		bn, _ := d.Alloc(4)
		bo, _ := d.Alloc(c.nh * maxSeq * 4)
		bq.Write(f32b(q))
		bk.Write(f32b(kc))
		bn.Write(u32b1(uint32(c.n)))
		sentinel := make([]float32, c.nh*maxSeq)
		for i := range sentinel {
			sentinel[i] = 12345
		}
		bo.Write(f32b(sentinel))
		if err := kern.Launch(c.nh*capN, 32, bq, bk, bn, bo); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, c.nh*maxSeq*4)
		bo.Read(raw)
		got := make([]float32, c.nh*maxSeq)
		b2f(raw, got)
		kern.Close()
		bq.Free()
		bk.Free()
		bn.Free()
		bo.Free()
		for h := 0; h < c.nh; h++ {
			for p := 0; p < c.n; p++ {
				var s float64
				for i := 0; i < c.hd; i++ {
					s += float64(q[h*c.hd+i]) * float64(kc[p*c.kvDim+(h/c.gqa)*c.hd+i])
				}
				s *= float64(scale)
				g := float64(got[h*maxSeq+p])
				if math.IsNaN(g) || math.IsInf(g, 0) || math.Abs(g-s) > 1e-4*(1+math.Abs(s)) {
					t.Fatalf("%+v head %d pos %d: got %g, reference %g", c, h, p, g, s)
				}
			}
		}
	}
}
