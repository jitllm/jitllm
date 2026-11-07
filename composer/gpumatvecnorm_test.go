package composer_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUMatVecNormRows forms a pointwise projection and the row
// norm over its outputs -- a Gemma 3n tower block's start -- over many rows
// on every card, its lanes the device's and four, with the card's matrix
// units and without, and holds the coarsened
// form to the one formed without the coarsen pass, run on the same card.
// The violation: a chain that reads the projection's outputs at another
// row's place parts from the members run apart.
func TestComposedGPUMatVecNormRows(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const K, N = 160, 160
	rng := rand.New(rand.NewSource(16))
	weights := map[jlm.Role][3][]uint32{jlm.RoleVMNPwExp: packed(t, quant.Q8_0, layout.Q8_0, N, K, rng)}
	w := make([]float32, N)
	for i := range w {
		w[i] = float32(1 + 0.3*rng.NormFloat64())
	}
	vecs := map[jlm.Role][]float32{jlm.RoleVMNPwExpNorm: w}
	formats := map[jlm.Role]jlm.Type{jlm.RoleVMNPwExp: jlm.TypeQ8, jlm.RoleVMNPwExpNorm: jlm.TypeF32}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, rows := range []int{64, 576, 2304} {
				for _, arm := range []struct {
					lanes  int
					matrix bool
				}{{0, false}, {4, false}, {4, true}} {
					lanes := arm.lanes
					name := fmt.Sprintf("%d rows/%d lanes/matrix %v", rows, lanes, arm.matrix)
					b := graph.New()
					in := b.In("x", K)
					o := b.Norm(b.MatVec(in, jlm.RoleVMNPwExp, N), jlm.RoleVMNPwExpNorm, 1e-6)
					b.Out(o)
					dev := gpuDevice(d)
					if lanes > 0 {
						dev.Lanes = lanes
					}
					if arm.matrix {
						if ok, _ := backend.LowersTile(d, backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32}); !ok {
							continue
						}
						dev.Matrix = [3]int{16, 16, 16}
					}
					form := func(r *composer.Registry) []composer.Code {
						_, codes, err := r.Form(composer.Request{Graph: b.G, Formats: formats,
							Step: composer.Step{Rows: rows, Page: 64, Keys: 1}, Device: dev, Apart: true})
						if err != nil {
							t.Fatalf("%s: %v", name, err)
						}
						return codes
					}
					x := make([]float32, rows*K)
					for i := range x {
						x[i] = float32(rng.NormFloat64())
					}
					run := func(codes []composer.Code) []float64 {
						res, err := runRoles(d, codes, map[graph.ValueID][]float32{in: x}, weights, vecs, o, rows*N)
						if err != nil {
							t.Fatalf("%s: %v", name, err)
						}
						return res
					}
					chained, apart := form(composer.Default()), form(without("coarsen"))
					e := nmseOf(run(chained), run(apart))
					t.Logf("%s: %d launches against %d apart, NMSE %.3e", name, len(chained), len(apart), e)
					if !(e < 1e-6) {
						t.Errorf("%s: the coarsened form parts from its members run apart: NMSE %.3e", name, e)
					}
				}
			}
		})
	}
}
