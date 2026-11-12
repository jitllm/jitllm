package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUMatrixFoldedRows forms on the matrix unit a projection of
// rows a quarter of the step's -- four rows folded into one (graph.OpFold),
// as a tower's keys downsampled two by two are -- and runs it at calls of 1
// to the capacity's rows: the call's rows reach ceil(rows / 4) of the
// projection's, each held to the oracle, and nothing past them or the guard
// after the result is written. Gemma 3n's tower read 64 rows past its
// downsampled keys and values here, its matrix form's last row the step's.
func TestComposedGPUMatrixFoldedRows(t *testing.T) {
	testlock.GPU(t)
	devs, closeAll := matrixDevices(t)
	defer closeAll()
	if len(devs) == 0 {
		t.Skip("no device here multiplies binary16 tiles")
	}
	const guard, capacity, fold, n, k = 64, 512, 4, 64, 640
	rows := capacity / fold
	for name, d := range devs {
		t.Run(name, func(t *testing.T) {
			dev := gpuDevice(d)
			dev.Matrix = [3]int{matrixTile().M, matrixTile().N, matrixTile().K}
			rng := rand.New(rand.NewSource(59))
			for _, typ := range []quant.Type{quant.F16, quant.Q8_0} {
				for _, tile := range splitTiles(gpuMatrixTiles()) {
					what := fmt.Sprintf("%v/%d outputs of %d over %d rows of %d at %v", typ, n, k, rows, capacity, tile)
					jt, ok := jlm.TypeOf(typ)
					q, qok := layout.QuantOf(typ)
					if typ == quant.F16 {
						jt, q, ok, qok = jlm.TypeF16, layout.Float16, true, true
					}
					if !ok || !qok {
						t.Fatalf("%v has no container type or kernel format", typ)
					}
					w := matrixWeights(typ, q, n, k, rng)
					qs, dw, sc, err := layout.PackWeights(q, w, n, k)
					if err != nil {
						t.Fatal(err)
					}
					td := dev
					td.Tile = tile
					b := graph.New()
					in := b.In("x", k/fold)
					folded := b.Emit(graph.OpFold, []graph.ValueID{in}, []graph.Result{{Name: "folded", Width: k, Rows: rows}},
						nil, graph.Attrs{Span: fold}, nil, nil)[0]
					out := b.MatVec(folded, jlm.RoleAttnQ, n)
					b.Out(out)
					_, formed, err := poisonedRegistry().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt},
						Step: composer.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: td})
					if err != nil {
						t.Fatalf("%s: forming: %v", what, err)
					}
					matrix := 0
					for _, c := range formed {
						if c.Matrix != [2]int{} {
							matrix++
						}
					}
					if matrix == 0 {
						t.Fatalf("%s: no kernel was formed on the matrix unit", what)
					}
					op, err := compileLone(d, formed)
					if err != nil {
						t.Fatalf("%s: %v", what, err)
					}
					x := make([]float32, capacity*k/fold)
					for i := range x {
						x[i] = float32(rng.NormFloat64())
					}
					want := make([]float32, rows*n)
					for r := range rows {
						if err := oracle.MatVec(want[r*n:(r+1)*n], typ, w, x[r*k:(r+1)*k], n, k); err != nil {
							t.Fatal(err)
						}
					}
					for _, L := range []int{1, 5, 64, 300, capacity} {
						xs := make([]float32, len(x))
						copy(xs, x[:L*k/fold])
						for i := L * k / fold; i < len(xs); i++ {
							xs[i] = math.Float32frombits(guardWord)
						}
						live := (L + fold - 1) / fold
						res, tail, err := runGraphRows(d, formed, op.kerns, in, out, xs, rows*n, guard, [3][]uint32{qs, dw, sc}, L)
						if err != nil {
							t.Fatalf("%s: %v", what, err)
						}
						for i, v := range res[live*n:] {
							if math.Float32bits(v) != guardWord {
								t.Fatalf("%s: a call of %d rows wrote row %d's element %d past its %d rows", what, L, live+i/n, i%n, live)
							}
						}
						for i, v := range tail {
							if v != guardWord {
								t.Fatalf("%s: a call of %d rows wrote the guard word %d past the result", what, L, i)
							}
						}
						// A call's last folded row reads rows past the call's own, which
						// the guard words poison, unless the call fills it.
						whole := L / fold * n
						if e := gpuNMSE(res[:whole], want[:whole]); whole > 0 && (math.IsNaN(e) || !(e < 2e-6)) {
							t.Fatalf("%s: a call of %d rows reads NMSE %.3e against the oracle", what, L, e)
						}
					}
					op.close()
				}
			}
		})
	}
}
