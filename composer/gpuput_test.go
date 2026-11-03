package composer_test

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUPut holds OpPut on every GPU device to its definition
// (composer.PutRef), bit for bit, at ragged widths, index counts and rows,
// indices past the row and repeated among them. The violation: the row
// unput, which the comparison must tell apart.
func TestComposedGPUPut(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
	}
	rng := rand.New(rand.NewSource(29))
	words := func(f []float32) []uint32 {
		u := make([]uint32, len(f))
		for i, x := range f {
			u[i] = math.Float32bits(x)
		}
		return u
	}
	for _, d := range devs {
		t.Run(d.API(), func(t *testing.T) {
			ran, parted := 0, 0
			for _, w := range []int{1, 3, 8, 40, 129, 1000, 32768} {
				for _, k := range []int{1, 5, 16, 33, 256} {
					g, row, idx, vals, out := composer.PutGraph(w, k)
					for _, R := range []int{1, 3} {
						_, codes, err := composer.Default().Form(composer.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{},
							Step: composer.Step{Rows: R, Page: 64, Keys: 1}, Device: gpuDevice(d)})
						if err != nil {
							t.Fatalf("%d indices into %d at %d rows: %v", k, w, R, err)
						}
						var rows, vs []float32
						var ix []uint32
						for range R {
							r, i, v := composer.PutCase(rng, w, k)
							rows, ix, vs = append(rows, r...), append(ix, i...), append(vs, v...)
						}
						m := map[graph.ValueID][]uint32{row: words(rows), idx: slices.Clone(ix), vals: words(vs), out: poisoned(R * w)}
						run := newPassRunner(d)
						got, err := run.run(codes, 0, len(codes), m, map[string][]uint32{"rows.n": {uint32(R)}}, nil, nil)
						run.close()
						if err != nil {
							t.Fatalf("%d indices into %d at %d rows: %v", k, w, R, err)
						}
						if len(got) != R*w {
							t.Fatalf("%d indices into %d at %d rows: %d words written", k, w, R, len(got))
						}
						for r := range R {
							want := composer.PutRef(rows[r*w:(r+1)*w], ix[r*k:(r+1)*k], vs[r*k:(r+1)*k])
							for i := range w {
								if got[r*w+i] != math.Float32bits(want[i]) {
									t.Fatalf("%d indices into %d, row %d of %d: element %d is %v, want %v", k, w, r, R, i,
										math.Float32frombits(got[r*w+i]), want[i])
								}
							}
							if !slices.Equal(want, rows[r*w:(r+1)*w]) {
								parted++
							}
						}
						ran++
					}
				}
			}
			if ran == 0 || parted == 0 {
				t.Fatalf("%d shapes, %d rows parted from the row unput: the gate cannot see a put", ran, parted)
			}
			t.Logf("%d shapes held bit for bit; %d rows parted from the row unput", ran, parted)
		})
	}
}
