package composer_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedGPUHeadDots is OpHeadDots on every device backend, at ragged
// heads, head widths and table rows: each row's every head's dot with every
// table row's same head, and with the rows a relative index names
// (graph.RelRows, a key a table row, none past the table), within f32
// reduction order, and the words past the result untouched. The violation
// is the dots of the next head.
func TestComposedGPUHeadDots(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		testmodels.Missing(t, "no GPU backend on this host: the gate proved nothing")
	}
	const guard = 64
	cases := [][4]int{{1, 1, 1, 1}, {2, 3, 5, 3}, {3, 8, 9, 5}, {4, 7, 13, 2}, {8, 16, 31, 6}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := composer.Default()
			dev := gpuDevice(d)
			ran, moved := 0, false
			for k := range 2 * len(cases) {
				cs, rel := cases[k/2], k%2 == 1
				H, hd, T, Q := cs[0], cs[1], cs[2], cs[3]
				D := H * hd
				K, top := T, (T+1)/2
				a := graph.Attrs{NHead: H, HeadDim: hd}
				name := fmt.Sprintf("%d heads of %d/%d table rows/%d rows", H, hd, T, Q)
				if rel {
					K = Q + 1
					a = graph.RelRows(top)
					a.NHead, a.HeadDim, a.Span = H, hd, K
					name += fmt.Sprintf("/relative over %d keys", K)
				}
				b := graph.New()
				x0 := b.In("x", D)
				b.Out(b.Emit(graph.OpHeadDots, []graph.ValueID{x0}, []graph.Result{{Name: "dots", Width: H * K}},
					[]graph.Binding{{Role: jlm.RolePosEmbd, K: D, Rows: T}}, a, nil, nil)[0])
				_, codes, err := r.Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: Q, Page: 64, Keys: 1}, Device: dev,
					Formats: map[jlm.Role]jlm.Type{jlm.RolePosEmbd: jlm.TypeF32}})
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				x := make([]float32, Q*D)
				for i := range x {
					x[i] = float32(math.Sin(0.37*float64(i) + 0.1))
				}
				tab := make([]float32, T*D)
				for i := range tab {
					tab[i] = float32(math.Cos(0.23*float64(i) + 0.4))
				}
				want := func(next int) []float64 {
					out := make([]float64, 0, Q*H*K)
					for q := range Q {
						for h := range H {
							g := (h + next) % H
							for j := range K {
								tr := j
								if rel {
									tr = top - 1 - q + j
								}
								s := 0.0
								for e := range hd {
									if tr >= 0 && tr < T {
										s += float64(x[q*D+h*hd+e]) * float64(tab[tr*D+g*hd+e])
									}
								}
								out = append(out, s)
							}
						}
					}
					return out
				}
				res, tail, err := op.run(x, Q*H*K, guard, map[jlm.Role][]float32{jlm.RolePosEmbd: tab})
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
				off := func(w []float64) int {
					n := 0
					for i := range w {
						if d := math.Abs(float64(res[i]) - w[i]); math.IsNaN(d) || d > 1e-5*max(1, math.Abs(w[i])) {
							n++
						}
					}
					return n
				}
				if n := off(want(0)); n != 0 {
					t.Errorf("%s: %d of %d dots are not their head's", name, n, len(res))
					continue
				}
				if H > 1 && off(want(1)) > 0 {
					moved = true
				}
				ran++
				t.Logf("%s: every dot its head's", name)
			}
			if ran != 2*len(cases) {
				t.Fatalf("%d of %d cases ran", ran, 2*len(cases))
			}
			if !moved {
				t.Fatal("VIOLATION: the next head's dots read the same on every case")
			}
		})
	}
}
