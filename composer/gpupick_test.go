package composer_test

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUPickedRowsAreTheRows forms a Q8_0 head of 4096 to 131072
// rows on every device present twice: whole, and projecting the rows 512
// ids pick (graph.Head.Pick, the decision models' label pick). Each picked
// row must be the whole head's row bit for bit: the selection's sums are cut
// as the whole form's (composer.Pick), not by the selection's own count. The
// violation: where the whole head's sums are cut, the picked form uncut (the
// split passes off) must part from the whole head's rows, at one row count
// at least.
func TestComposedGPUPickedRowsAreTheRows(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const guard, k, P = 64, 2048, 512
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(46))
			seen := 0
			x := make([]float32, k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			for _, n := range []int{4096, 32000, 65536, 131072} {
				w := tiledWeights(quant.Q8_0, layout.Q8_0, n, k, rng)
				qs, dw, sc, err := layout.PackWeights(layout.Q8_0, w, n, k)
				if err != nil {
					t.Fatal(err)
				}
				planes := [3][]uint32{qs, dw, sc}
				form := func(r *composer.Registry, g *graph.Graph) []composer.Code {
					_, codes, err := r.Form(composer.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jlm.TypeQ8},
						Step: composer.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
					if err != nil {
						t.Fatalf("%d rows: %v", n, err)
					}
					return codes
				}
				run := func(codes []composer.Code, g *graph.Graph, in, out graph.ValueID, width int, inputs map[graph.ValueID][]uint32) []float32 {
					op, err := compileLone(d, codes)
					if err != nil {
						t.Fatalf("%d rows: %v", n, err)
					}
					defer op.close()
					res, tail, err := runGraphRowsWith(d, codes, op.kerns, in, out, x, width, guard, planes, 0, inputs)
					if err != nil {
						t.Fatalf("%d rows: %v", n, err)
					}
					for _, v := range tail {
						if v != guardWord {
							t.Fatalf("%d rows: a word past the output was written", n)
						}
					}
					return res
				}
				b := graph.New()
				in := b.In("x", k)
				out := b.MatVec(in, jlm.RoleOutput, n)
				b.Out(out)
				wholeCodes := form(composer.Default(), b.G)
				cut := false
				for _, c := range wholeCodes {
					cut = cut || slices.Contains(c.Passes, "split")
				}
				whole := run(wholeCodes, b.G, in, out, n, nil)

				pb := graph.New()
				pin := pb.In("x", k)
				ids := pb.In("pick.ids", P)
				pb.G.Values[ids].Elem = graph.ElemU32
				pout := pb.Emit(graph.OpMatVec, []graph.ValueID{pin, ids}, []graph.Result{{Name: "picked", Width: P}},
					[]graph.Binding{{Role: jlm.RoleOutput, K: k, Rows: n}}, graph.Attrs{}, nil, nil)[0]
				pb.Out(pout)
				pick := make([]uint32, P)
				for i := range pick {
					pick[i] = uint32(rng.Intn(n))
				}
				inputs := map[graph.ValueID][]uint32{ids: pick}
				differ := func(got []float32) int {
					c := 0
					for i, id := range pick {
						if math.IsNaN(float64(whole[id])) || math.Float32bits(got[i]) != math.Float32bits(whole[id]) {
							c++
						}
					}
					return c
				}
				if c := differ(run(form(composer.Default(), pb.G), pb.G, pin, pout, P, inputs)); c != 0 {
					t.Errorf("%d rows: %d of %d picked rows are not the whole head's to the bit", n, c, P)
				}
				if !cut {
					t.Logf("%s, %d rows: %d picked rows bit for bit; the whole head's sums are not cut here", d.API(), n, P)
					continue
				}
				r := composer.Default()
				p := r.Profiles[composer.DeviceGPU]
				p.Off = map[string]bool{"split": true}
				r.Profiles[composer.DeviceGPU] = p
				c := differ(run(form(r, pb.G), pb.G, pin, pout, P, inputs))
				if c == 0 {
					t.Errorf("%d rows: the picked form uncut writes every row as the whole head's cut form: the gate cannot see the cut", n)
				}
				seen++
				t.Logf("%s, %d rows: %d picked rows bit for bit; uncut, %d of them part", d.API(), n, P, c)
			}
			if seen == 0 {
				t.Fatal("no row count where the whole head's sums are cut: the violation never ran")
			}
		})
	}
}
