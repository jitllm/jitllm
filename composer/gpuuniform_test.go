package composer_test

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUUniform holds OpUniform on every GPU device to its
// definition (composer.UniformRef), bit for bit, at ragged widths and several
// rows each under its own counter: the draw a sampler takes on a card is the
// host's. The violation: nine rounds, which the comparison must tell apart.
func TestComposedGPUUniform(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
	}
	rng := rand.New(rand.NewSource(19))
	for _, d := range devs {
		t.Run(d.API(), func(t *testing.T) {
			ran, nine := 0, 0
			for _, w := range []int{1, 3, 8, 40, 64, 129, 1024} {
				g, keys, ctr, u := composer.UniformGraph(w)
				for _, R := range []int{1, 3} {
					_, codes, err := composer.Default().Form(composer.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{},
						Step: composer.Step{Rows: R, Page: 64, Keys: 1}, Device: gpuDevice(d)})
					if err != nil {
						t.Fatalf("%d keys at %d rows: %v", w, R, err)
					}
					ks, cs := make([]uint32, R*w), make([]uint32, 2*R)
					for i := range ks {
						ks[i] = rng.Uint32()
					}
					for i := range cs {
						cs[i] = rng.Uint32()
					}
					vals := map[graph.ValueID][]uint32{keys: slices.Clone(ks), ctr: cs, u: poisoned(R * w)}
					run := newPassRunner(d)
					words, err := run.run(codes, 0, len(codes), vals, map[string][]uint32{"rows.n": {uint32(R)}}, nil, nil)
					run.close()
					if err != nil {
						t.Fatalf("%d keys at %d rows: %v", w, R, err)
					}
					if len(words) != R*w {
						t.Fatalf("%d keys at %d rows: %d words written", w, R, len(words))
					}
					for r := range R {
						want := composer.UniformRef(ks[r*w:(r+1)*w], cs[2*r], cs[2*r+1], 10)
						got := make([]float32, w)
						for i := range got {
							got[i] = math.Float32frombits(words[r*w+i])
						}
						if !slices.Equal(got, want) {
							t.Fatalf("%d keys, row %d of %d: %v, want %v", w, r, R, got[:min(4, w)], want[:min(4, w)])
						}
						if !slices.Equal(got, composer.UniformRef(ks[r*w:(r+1)*w], cs[2*r], cs[2*r+1], 9)) {
							nine++
						}
					}
					ran++
				}
			}
			if ran == 0 || nine == 0 {
				t.Fatalf("%d shapes, %d rows parted from nine rounds: the gate cannot see a round", ran, nine)
			}
			t.Logf("%d shapes held bit for bit; %d rows parted from nine rounds", ran, nine)
		})
	}
}
