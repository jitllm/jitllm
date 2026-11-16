package composer_test

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUPad widens rows of ragged widths with zeros to the next
// whole block (and a whole block to the next) on every GPU, at a step of one
// row and of 8, into a poisoned output: each row's elements its input's,
// then zeros, exactly.
func TestComposedGPUPad(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(41))
			for _, w := range []int{1, 7, 20, 31, 32, 33, 144} {
				p := (w + 31) / 32 * 32
				if p == w {
					p += 32
				}
				for _, R := range []int{1, 8} {
					b := graph.New()
					x := b.In("x", w)
					out := b.Pad(x, p)
					b.Out(out)
					fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{})
					if err != nil {
						t.Fatalf("%d wide to %d at %d rows: %v", w, p, R, err)
					}
					in := make([]float32, R*w)
					for i := range in {
						in[i] = float32(rng.NormFloat64())
					}
					fr.vals = map[graph.ValueID][]uint32{x: f32w(in), out: poisoned(R * p)}
					clear(fr.bufs)
					if err := fr.launch(); err != nil {
						fr.close()
						t.Fatalf("%d wide to %d at %d rows: %v", w, p, R, err)
					}
					got := wf32(fr.vals[out])
					fr.close()
					for r := range R {
						for i := range p {
							want := float32(0)
							if i < w {
								want = in[r*w+i]
							}
							if got[r*p+i] != want {
								t.Fatalf("%d wide to %d at %d rows: row %d element %d is %v, want %v", w, p, R, r, i, got[r*p+i], want)
							}
						}
					}
				}
			}
		})
	}
}
