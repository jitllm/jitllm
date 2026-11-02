package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUCrossAttend is TestComposedCPUCrossAttend on every device
// backend: query rows attending, both ways, version 0 of a history the
// graph does not write, at head widths of 128, 72 and 48, grouped and not,
// query and key counts ragged and apart, every output word written and held
// to the definition (binary16 keys and values, f64 softmax); a span cut by
// its last key or starting one key late must fail it.
func TestComposedGPUCrossAttend(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	const page = 32
	ran := 0
	for _, d := range devs {
		for _, sh := range [][3]int{{4, 4, 128}, {6, 2, 48}, {3, 3, 72}} {
			for _, qn := range [][2]int{{1, 1}, {7, 5}, {64, 33}, {9, 100}, {64, 300}} {
				heads, kvh, dim, Q, N := sh[0], sh[1], sh[2], qn[0], qn[1]
				t.Run(fmt.Sprintf("%s/%dq%dkv%dd/Q%d/N%d", d.API(), heads, kvh, dim, Q, N), func(t *testing.T) {
					pages := (N + page - 1) / page
					b := graph.New()
					b.Out(b.CrossAttend(b.In("q", heads*dim), graph.StateRef{State: graph.StateKV}, graph.Attrs{NHead: heads, NKVHead: kvh, HeadDim: dim,
						Scale: 1 / math.Sqrt(float64(dim))}))
					g := b.G
					_, codes, err := composer.Default().Form(composer.Request{Graph: g, Step: composer.Step{Rows: Q, Page: page, Keys: pages * page}, History: composer.History{Keys: composer.Pool{Half: true}, Values: composer.Pool{Half: true}},
						Device: gpuDevice(d)})
					if err != nil {
						t.Fatal(err)
					}
					rng := rand.New(rand.NewSource(int64(Q*1000 + N)))
					w, qw := kvh*dim, heads*dim
					rnd := func(n int) []float32 {
						x := make([]float32, n)
						for i := range x {
							x[i] = float32(rng.NormFloat64())
						}
						return x
					}
					q, ks, vs := rnd(Q*qw), rnd(N*w), rnd(N*w)
					pool := func(x []float32) []uint32 {
						out := poisoned(pages * page * w / 2)
						for i := range N * w / 2 {
							out[i] = uint32(quant.EncodeHalf(x[2*i])) | uint32(quant.EncodeHalf(x[2*i+1]))<<16
						}
						return out
					}
					round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
					scale := float64(float32(1 / math.Sqrt(float64(dim))))
					want := make([]float64, Q*qw)
					for r := range Q {
						for h := range heads {
							kv := h / (heads / kvh)
							sc := make([]float64, N)
							mx := math.Inf(-1)
							for p := range N {
								var s float64
								for e := range dim {
									s += float64(q[r*qw+h*dim+e]) * round(ks[p*w+kv*dim+e])
								}
								sc[p] = s * scale
								mx = math.Max(mx, sc[p])
							}
							var sum float64
							for p := range sc {
								sc[p] = math.Exp(sc[p] - mx)
								sum += sc[p]
							}
							for e := range dim {
								var o float64
								for p := range sc {
									o += sc[p] / sum * round(vs[p*w+kv*dim+e])
								}
								want[r*qw+h*dim+e] = o
							}
						}
					}
					pg := make([]uint32, pages)
					for i := range pg {
						pg[i] = uint32(i)
					}
					kp, vp := pool(ks), pool(vs)
					run := func(lo, hi uint32) float64 {
						rows := make([]uint32, Q*4)
						for r := range Q {
							copy(rows[r*4:], []uint32{0, lo, hi, hi - 1})
						}
						vals := map[graph.ValueID][]uint32{g.In[0]: f32w(q), g.Out[0]: poisoned(Q * qw)}
						tables := map[string][]uint32{"rows": rows, "rows.win": rows, "rows.n": {uint32(Q)}, "pages": pg,
							"kv.k": kp, "kv.v": vp}
						out, err := launchAttend(d, codes, vals, tables, nil, g.Out[0])
						if err != nil {
							t.Fatal(err)
						}
						got := wf32(out)
						var num, den float64
						for i := range want {
							e := float64(got[i]) - want[i]
							num += e * e
							den += want[i] * want[i]
						}
						return num / den
					}
					e := run(0, uint32(N))
					t.Logf("%d queries over %d keys: NMSE %.3e against the definition", Q, N, e)
					if math.IsNaN(e) || !(e < 1e-6) {
						t.Fatalf("NMSE %.3e against the definition", e)
					}
					ran++
					if N < 2 {
						return
					}
					if e := run(0, uint32(N-1)); !(e > 1e-6) {
						t.Errorf("with the span cut by its last key the NMSE is %.3e: the gate cannot see it", e)
					}
					if e := run(1, uint32(N)); !(e > 1e-6) {
						t.Errorf("with the span starting one key late the NMSE is %.3e: the gate cannot see it", e)
					}
				})
			}
		}
	}
	if ran == 0 {
		t.Fatal("no cross-attention held to the definition: the gate proved nothing")
	}
}
