package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUFloatTiles forms a projection of float32 weights at a
// prompt chunk's rows on every device, at each of the profile's float tiles
// alone, at outputs and a capacity that do not fill the workgroups, and runs
// it at calls of 1 to the capacity's rows: it must be the floattiles form,
// each row held to the oracle within float32's rounding, and every word past
// the call's rows and the guard after them untouched. Two violations must
// fail: the weight's words moved by one, and the next row's reference.
func TestComposedGPUFloatTiles(t *testing.T) {
	testlock.GPU(t)
	devs := map[string]backend.Device{}
	for _, d := range backend.Open() {
		defer d.Close()
		devs[d.API()] = d
	}
	if d, err := backend.OpenCUDAWith(0, backend.Opts{CUDATarget: "sm_70"}); err == nil {
		defer d.Close()
		devs["ptx-sm_70"] = d
	}
	var tiles []composer.FloatTile
	var pass composer.FloatTiles
	for _, p := range composer.Default().Profiles[composer.DeviceGPU].Passes {
		if f, ok := p.(composer.FloatTiles); ok {
			pass, tiles = f, f.Tiles
		}
	}
	if len(tiles) == 0 {
		t.Fatal("the GPU profile has no float tiles")
	}
	const guard, capacity = 64, 100
	calls := []int{1, 15, 33, 64, 100}
	for _, name := range slices.Sorted(func(yield func(string) bool) {
		for n := range devs {
			if !yield(n) {
				return
			}
		}
	}) {
		d := devs[name]
		t.Run(name, func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(61))
			ran := 0
			for _, tile := range tiles {
				for _, sh := range []struct{ n, k int }{{128, 2048}, {100, 256}, {1056, 512}} {
					n, k := sh.n, sh.k
					what := fmt.Sprintf("%d outputs of %d at %v", n, k, tile)
					reg := poisonedRegistry()
					p := reg.Profiles[composer.DeviceGPU]
					p.Passes = slices.Clone(p.Passes)
					for i, ps := range p.Passes {
						if _, ok := ps.(composer.FloatTiles); ok {
							q := pass
							q.Tiles = []composer.FloatTile{tile}
							p.Passes[i] = q
						}
					}
					reg.Profiles[composer.DeviceGPU] = p
					w := make([]byte, 4*n*k)
					ws := make([]uint32, n*k)
					for i := range ws {
						ws[i] = math.Float32bits(float32(rng.NormFloat64()))
						binary.LittleEndian.PutUint32(w[4*i:], ws[i])
					}
					b := graph.New()
					in := b.In("x", k)
					out := b.MatVec(in, jlm.RoleRouter, n)
					b.Out(out)
					_, formed, err := reg.Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleRouter: jlm.TypeF32},
						Step: composer.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
					if err != nil {
						t.Fatalf("%s: forming: %v", what, err)
					}
					if !slices.ContainsFunc(formed, func(c composer.Code) bool { return slices.Contains(c.Passes, "floattiles") }) {
						t.Fatalf("%s: no kernel was formed in float tiles", what)
					}
					op, err := compileLone(d, formed)
					if err != nil {
						t.Fatalf("%s: %v", what, err)
					}
					x := make([]float32, capacity*k)
					for i := range x {
						x[i] = float32(rng.NormFloat64())
					}
					want := make([]float32, capacity*n)
					for r := range capacity {
						if err := oracle.MatVec(want[r*n:(r+1)*n], quant.F32, w, x[r*k:(r+1)*k], n, k); err != nil {
							t.Fatal(err)
						}
					}
					run := func(L int, planes [3][]uint32) []float32 {
						xs := make([]float32, capacity*k)
						copy(xs, x[:L*k])
						for i := L * k; i < len(xs); i++ {
							xs[i] = math.Float32frombits(guardWord)
						}
						res, tail, err := runGraphRows(d, formed, op.kerns, in, out, xs, capacity*n, guard, planes, L)
						if err != nil {
							t.Fatalf("%s: %v", what, err)
						}
						for i, v := range res[L*n:] {
							if math.Float32bits(v) != guardWord {
								t.Fatalf("%s: a call of %d rows wrote row %d's element %d", what, L, L+i/n, i%n)
							}
						}
						for i, v := range tail {
							if v != guardWord {
								t.Fatalf("%s: the guard word %d past the result was written", what, i)
							}
						}
						return res[:L*n]
					}
					planes := [3][]uint32{ws}
					worst := 0.0
					for _, L := range calls {
						e := gpuNMSE(run(L, planes), want[:L*n])
						if math.IsNaN(e) || !(e < 1e-11) {
							t.Fatalf("%s: a call of %d rows reads NMSE %.3e against the oracle", what, L, e)
						}
						worst = max(worst, e)
					}
					t.Logf("%s: worst NMSE %.3e", what, worst)
					if e := gpuNMSE(run(capacity, planes)[:(capacity-1)*n], want[n:]); !(e > 1e-2) {
						t.Fatalf("%s: against the next row's reference the NMSE is %.3e", what, e)
					}
					moved := [3][]uint32{append(ws[1:len(ws):len(ws)], ws[0])}
					if e := gpuNMSE(run(capacity, moved), want); !(e > 1e-4) {
						t.Fatalf("%s: with the weight's words moved by one the NMSE is %.3e", what, e)
					}
					op.close()
					ran++
				}
			}
			t.Logf("%s: %d projections formed in float tiles, held to the oracle at calls of %v rows", name, ran, calls)
		})
	}
}
