package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUPrologueTakesWholeValues runs a projection of a value two
// kernels write -- a turned tail and the head before it copied -- on every
// GPU, against its form with no prologue: a prologue running one of the
// two in the projection's workgroups would leave the other's part of what
// it reads unwritten. Behind an op of the rows too, which the turn's
// kernels follow.
func TestComposedGPUPrologueTakesWholeValues(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const Q = 1024
	for _, d := range devs {
		defer d.Close()
		for _, sh := range []struct{ heads, hd, nrot int }{{4, 32, 16}, {8, 32, 16}, {4, 64, 16}, {8, 64, 32}} {
			for _, made := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%dx%d/rot%d/made%v", d.API(), sh.heads, sh.hd, sh.nrot, made), func(t *testing.T) {
					K := sh.heads * sh.hd
					rng := rand.New(rand.NewSource(7))
					x := make([]float32, K)
					for i := range x {
						x[i] = float32(rng.NormFloat64())
					}
					w := make([]float32, Q*K)
					for i := range w {
						w[i] = float32(rng.NormFloat64() / 16)
					}
					angles := make([]float32, sh.nrot)
					for p := range sh.nrot / 2 {
						th := 5 * math.Pow(10000, -2*float64(p)/float64(sh.nrot))
						angles[2*p], angles[2*p+1] = float32(math.Cos(th)), float32(math.Sin(th))
					}
					b := graph.New()
					in := b.In("x", K)
					src := in
					if made {
						src = b.Emit(graph.OpAdd, []graph.ValueID{in, in}, []graph.Result{{Name: "x2", Width: K}}, nil, graph.Attrs{}, nil, nil)[0]
					}
					a := graph.Attrs{HeadDim: sh.hd, NRot: sh.nrot, Base: 10000, Layout: graph.RopeNorm, Offset: sh.hd - sh.nrot, Inverse: true}
					turned := b.Emit(graph.OpRope, []graph.ValueID{src}, []graph.Result{{Name: "turned", Width: K}}, nil, a, nil, nil)[0]
					out := b.MatVec(turned, jlm.RoleAttnOut, Q)
					b.Out(out)
					run := func(r *composer.Registry) ([]float32, int) {
						_, codes, err := r.Form(composer.Request{Graph: b.G, Apart: true,
							Formats:  map[jlm.Role]jlm.Type{jlm.RoleAttnOut: jlm.TypeF32},
							Rotaries: []graph.Rotary{{NRot: sh.nrot, Base: 10000}},
							Step:     composer.Step{Rows: 1, Page: 16, Keys: 64, Seqs: 1}, Device: gpuDevice(d)})
						if err != nil {
							t.Fatal(err)
						}
						tables := map[string][]uint32{}
						for _, c := range codes {
							for _, bd := range c.Binds {
								if o := bd.Operand; o.Kind == composer.OperandTable && o.Table == composer.TableAngles {
									tables[o.Name] = f32w(angles)
								}
							}
						}
						fr := &recRun{d: d, codes: codes, vals: map[graph.ValueID][]uint32{in: f32w(x), turned: poisoned(K), out: poisoned(Q)},
							tables: tables, wts: map[jlm.Role][]uint32{jlm.RoleAttnOut: f32w(w)}, bufs: map[string]backend.Buf{}}
						if made {
							fr.vals[src] = poisoned(K)
						}
						for _, c := range codes {
							k, err := d.Compile(c.GPU)
							if err != nil {
								fr.close()
								t.Fatalf("compiling %s: %v", c.Name, err)
							}
							fr.kerns = append(fr.kerns, k)
						}
						defer fr.close()
						if err := fr.launch(); err != nil {
							t.Fatal(err)
						}
						return wf32(fr.vals[out]), len(codes)
					}
					got, n := run(composer.Default())
					want, m := run(without("prologue"))
					var num, den float64
					for i := range want {
						if math.IsNaN(float64(got[i])) || math.IsNaN(float64(want[i])) {
							t.Fatalf("element %d: %v, with no prologue %v", i, got[i], want[i])
						}
						e := float64(got[i] - want[i])
						num, den = num+e*e, den+float64(want[i])*float64(want[i])
					}
					if !(den > 0) || !(num/den < 1e-10) {
						t.Fatalf("%d launches part from the %d with no prologue: NMSE %.3g", n, m, num/den)
					}
					t.Logf("%d launches, %d with no prologue", n, m)
				})
			}
		}
	}
}
