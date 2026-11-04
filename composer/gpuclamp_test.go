package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUClamp is TestComposedCPUClamp on every device backend: the
// composer's clamp by a table's pair (jlm.RoleVClamp) at every pair of a
// block's 28 words and by a symmetric bound, at ragged widths and several
// rows, exact against the oracle with the words past the result untouched,
// and the table read one word over seen.
func TestComposedGPUClamp(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	const guard, pairs = 64, 14
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := composer.Default()
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(31))
			tab := make([]float32, 2*pairs)
			for i := range pairs {
				lo, hi := float32(-0.5-rng.Float64()), float32(0.5+rng.Float64())
				switch i % 4 {
				case 1:
					lo = float32(math.Inf(-1))
				case 2:
					hi = float32(math.Inf(1))
				}
				tab[2*i], tab[2*i+1] = lo, hi
			}
			right, seen := 0, 0
			for _, rows := range []int{1, 3, 9} {
				for _, w := range []int{1, 2, 3, 7, 9, 17, 33, 257, 1000} {
					for _, off := range []int{-1, 0, 2, 12, 26} {
						name := fmt.Sprintf("%d rows/width %d/pair at %d", rows, w, off)
						k, at, bound := len(tab), off, 0.0
						if off < 0 {
							k, at, bound = 0, 0, 0.75
						}
						b := graph.New()
						x0 := b.In("x", w)
						var bind []graph.Binding
						if k > 0 {
							bind = []graph.Binding{{Role: jlm.RoleVClamp, K: k}}
						}
						b.Out(b.Emit(graph.OpClamp, []graph.ValueID{x0}, []graph.Result{{Name: "x.clamp", Width: w}}, bind,
							graph.Attrs{Offset: at, Bound: bound}, nil, nil)[0])
						_, codes, err := r.Form(composer.Request{Graph: b.G,
							Formats: map[jlm.Role]jlm.Type{jlm.RoleVClamp: jlm.TypeF32},
							Step:    composer.Step{Rows: rows, Page: 64, Keys: 1}, Device: dev})
						if err != nil {
							t.Errorf("%s: %v", name, err)
							continue
						}
						op, err := compileLone(d, codes)
						if err != nil {
							t.Errorf("%s: %v", name, err)
							continue
						}
						x := make([]float32, rows*w)
						for i := range x {
							x[i] = float32(3 * rng.NormFloat64())
						}
						lo, hi := float32(-bound), float32(bound)
						if k > 0 {
							lo, hi = tab[at], tab[at+1]
						}
						want := make([]float32, len(x))
						oracle.Clamp32(want, x, lo, hi)
						run := func(tab []float32) ([]float32, error) {
							res, tail, err := op.run(x, len(x), guard, map[jlm.Role][]float32{jlm.RoleVClamp: tab})
							if err != nil {
								return nil, err
							}
							for i, v := range tail {
								if v != guardWord {
									return nil, fmt.Errorf("the guard word %d past the result was written", i)
								}
							}
							return res, nil
						}
						got, err := run(tab)
						if err != nil {
							op.close()
							t.Errorf("%s: %v", name, err)
							continue
						}
						bad := 0
						for i := range want {
							if got[i] != want[i] {
								bad++
							}
						}
						if bad != 0 {
							op.close()
							t.Errorf("%s: %d of %d elements differ from the oracle's clamp", name, bad, len(want))
							continue
						}
						right++
						if k > 0 {
							moved, err := run(append(append([]float32(nil), tab[1:]...), tab[0]))
							differ := false
							for i := range want {
								differ = differ || err == nil && moved[i] != want[i]
							}
							switch {
							case err != nil:
								t.Errorf("%s: the violation did not run: %v", name, err)
							case differ:
								seen++
							case len(want) >= 16:
								t.Errorf("%s: the table read one word over clamps the same %d elements: the gate cannot see it", name, len(want))
							}
						}
						op.close()
					}
				}
			}
			t.Logf("%s: %d clamps exact against the oracle; the table read one word over seen at %d", d.API(), right, seen)
			if right == 0 || seen == 0 {
				t.Fatal("no clamp held to the oracle, or no violation seen: this gate proved nothing")
			}
		})
	}
}
