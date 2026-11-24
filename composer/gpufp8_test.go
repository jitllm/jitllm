package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedGPUFP8Weights is TestComposedCPUFP8Weights on every GPU
// device present: a matvec and an expert matvec over NVFP4 and MXFP8 rows,
// the format's basic decoded kernel, at ragged widths and a checkpoint's
// 1152, held to the oracle; one sheet's bytes changed must move its picks.
func TestComposedGPUFP8Weights(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		testmodels.Missing(t, "no GPU backend on this host: the gate runs on one")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(31))
			ran := 0
			for _, ty := range []jlm.Type{jlm.TypeNV4, jlm.TypeMX8} {
				for _, sh := range [][2]int{{16, 5}, {48, 7}, {96, 33}, {1152, 9}} {
					K, rows := sh[0], sh[1]
					if ty == jlm.TypeMX8 && K%32 != 0 {
						continue
					}
					name := fmt.Sprintf("%v/%dx%d", ty, rows, K)
					words, w := composer.FP8Bank(t, ty, 1, rows, K, rng)
					b := graph.New()
					x := b.In("x", K)
					y := b.MatVec(x, jlm.RoleAttnQ, rows)
					b.Out(y)
					codes := formWith(t, composer.Default(), b.G, map[jlm.Role]jlm.Type{jlm.RoleAttnQ: ty}, dev)
					xs := make([]float32, K)
					for i := range xs {
						xs[i] = float32(rng.NormFloat64())
					}
					got, err := runRoles(d, codes, map[graph.ValueID][]float32{x: xs},
						map[jlm.Role][3][]uint32{jlm.RoleAttnQ: {words, {0}, {0}}}, nil, y, rows)
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					if e := nmseOf(got, widen(composer.FP8Ref(w, xs, rows, K))); math.IsNaN(e) || !(e < 1e-9) {
						t.Errorf("%s: matvec %.3e from the oracle", name, e)
					}

					const E, P = 3, 2
					bank, bw := composer.FP8Bank(t, ty, E, rows, K, rng)
					b = graph.New()
					x = b.In("x", P*K)
					sel := b.In("sel", P)
					b.G.Values[sel].Elem = graph.ElemU32
					y = b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: "y", Width: P * rows}},
						[]graph.Binding{{Role: jlm.RoleExpGateBank, K: K, Rows: rows, Sheets: E}}, graph.Attrs{}, nil, nil)[0]
					b.Out(y)
					codes = formWith(t, composer.Default(), b.G, map[jlm.Role]jlm.Type{jlm.RoleExpGateBank: ty}, dev)
					xs = make([]float32, P*K)
					for i := range xs {
						xs[i] = float32(rng.NormFloat64())
					}
					picks := []uint32{2, 1}
					selv := []float32{math.Float32frombits(picks[0]), math.Float32frombits(picks[1])}
					run := func(bank []uint32) []float64 {
						got, err := runRoles(d, codes, map[graph.ValueID][]float32{x: xs, sel: selv},
							map[jlm.Role][3][]uint32{jlm.RoleExpGateBank: {bank, {0}, {0}}}, nil, y, P*rows)
						if err != nil {
							t.Fatalf("%s: experts: %v", name, err)
						}
						return got
					}
					var want []float32
					for j, e := range picks {
						want = append(want, composer.FP8Ref(bw[int(e)*rows*K:], xs[j*K:], rows, K)...)
					}
					if e := nmseOf(run(bank), widen(want)); math.IsNaN(e) || !(e < 1e-9) {
						t.Errorf("%s: experts %.3e from the oracle", name, e)
					}
					mut := append([]uint32(nil), bank...)
					per := len(bank) / E
					for i := 2 * per; i < 3*per; i++ {
						mut[i] ^= 0x01010101
					}
					if e := nmseOf(run(mut), widen(want)); !(e > 1e-6) {
						t.Errorf("%s: sheet 2's bytes changed reads %.3e: the gate cannot see the sheet", name, e)
					}
					ran++
				}
			}
			if ran != 6 {
				t.Fatalf("%d arms ran, want 6", ran)
			}
		})
	}
}

func widen(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, f := range v {
		out[i] = float64(f)
	}
	return out
}
