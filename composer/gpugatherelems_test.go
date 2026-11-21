package composer_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedGPUGatherElems is an element gather on every device backend,
// at ragged heads and keys: a relative shift of runs of keys a head from
// runs of 2*keys-1 dots, every element the input's element its index names,
// exactly, and the words past the result untouched. The violation is the
// gather read one key over.
func TestComposedGPUGatherElems(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		testmodels.Missing(t, "no GPU backend on this host: the gate proved nothing")
	}
	const guard = 64
	cases := [][2]int{{1, 1}, {2, 3}, {3, 5}, {4, 7}, {5, 33}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := composer.Default()
			dev := gpuDevice(d)
			ran, moved := 0, false
			for _, cs := range cases {
				heads, keys := cs[0], cs[1]
				rows, q := 2*keys-1, keys
				name := fmt.Sprintf("%d heads/%d keys", heads, keys)
				b := graph.New()
				x0 := b.In("dots", heads*rows)
				b.G.Values[x0].Rows = q
				var ix graph.Index
				row, col := ix.In(graph.GatherRow), ix.In(graph.GatherCol)
				h := ix.Div(col, ix.K(keys))
				j := ix.Rem(col, ix.K(keys), h)
				at := ix.Add(ix.Mul(row, ix.K(heads*rows)), ix.Mul(h, ix.K(rows)))
				b.Out(b.Emit(graph.OpGatherElems, []graph.ValueID{x0}, []graph.Result{{Name: "bias", Width: heads * keys, Rows: q}}, nil,
					ix.Attrs(ix.Add(at, ix.Sub(ix.Add(ix.K(keys-1), j), row))), nil, nil)[0])
				_, codes, err := r.Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: q, Page: 64, Keys: 1}, Device: dev})
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				x := make([]float32, q*heads*rows)
				for i := range x {
					x[i] = float32(math.Sin(0.37*float64(i) + 0.1))
				}
				want := func(shift int) []float32 {
					out := make([]float32, 0, q*heads*keys)
					for i := range q {
						for h := range heads {
							for j := range keys {
								at := keys - 1 - i + j + shift
								v := float32(0)
								if at >= 0 && at < rows {
									v = x[i*heads*rows+h*rows+at]
								}
								out = append(out, v)
							}
						}
					}
					return out
				}
				res, tail, err := op.run(x, q*heads*keys, guard, nil)
				op.close()
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				for i, v := range tail {
					if v != guardWord {
						t.Errorf("%s: the guard word %d past the result was written", name, i)
						break
					}
				}
				off := func(w []float32) int {
					n := 0
					for i := range w {
						if res[i] != w[i] {
							n++
						}
					}
					return n
				}
				if n := off(want(0)); n != 0 {
					t.Errorf("%s: %d of %d elements are not the ones their index names", name, n, len(res))
					continue
				}
				if off(want(1)) > 0 {
					moved = true
				}
				ran++
				t.Logf("%s: every element its index's", name)
			}
			if ran != len(cases) {
				t.Fatalf("%d of %d cases ran", ran, len(cases))
			}
			if !moved {
				t.Fatal("VIOLATION: the gather read one key over reads the same on every case")
			}
		})
	}
}
