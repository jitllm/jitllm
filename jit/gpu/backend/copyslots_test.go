package backend_test

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestCopySlotsMovesTheNamedSlots: CopySlots is how the shared recurrent form
// brings a compact next state home (each row's out slot of the scratch to its
// in slot of the pool), inside the step's submission. Each named destination
// slot must hold its source slot, bit for bit, and no other slot may move.
func TestCopySlotsMovesTheNamedSlots(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			const w, P = 300, 5
			rng := rand.New(rand.NewSource(5))
			src := make([]float32, P*w)
			for i := range src {
				src[i] = float32(rng.NormFloat64())
			}
			// dst slot <- src slot: 4 <- 0, 1 <- 3, 2 <- 2.
			pairs := [][2]int{{4, 0}, {1, 3}, {2, 2}}
			var desc []uint32
			for _, p := range pairs {
				desc = append(desc, uint32(p[0]), uint32(p[1]))
			}
			g := newGPU(t, d)
			defer g.free()
			bSrc, bDst := g.up(f32bytes(src)), g.up(f32bytes(poisonPool(P*w)))
			bMap := g.up(recDesc(0, desc...))
			k, err := oldkernels.CopySlots(w, len(pairs))
			if err != nil {
				t.Fatal(err)
			}
			if err := k.Validate(); err != nil {
				t.Fatal(err)
			}
			ck, err := d.Compile(k)
			if err != nil {
				t.Fatal(err)
			}
			defer ck.Close()
			n := len(pairs) * w
			if err := ck.Launch((n+127)/128, 128, bSrc, bDst, bMap); err != nil {
				t.Fatal(err)
			}
			got := readF32(t, bDst, P*w)
			var written []int
			for _, p := range pairs {
				written = append(written, p[0])
				for i := 0; i < w; i++ {
					if float32(got[p[0]*w+i]) != src[p[1]*w+i] {
						t.Fatalf("slot %d element %d = %g, want source slot %d's %g",
							p[0], i, got[p[0]*w+i], p[1], src[p[1]*w+i])
					}
				}
			}
			poisonKept(t, "copied pool", got, w, written...)
		})
	}
}
