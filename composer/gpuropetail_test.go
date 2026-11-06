package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPURopeTail turns each head's tail (Offset) on every GPU, both
// ways (Inverse), by the step's table, held to the definition row by row;
// the head before the tail is the input's.
func TestComposedGPURopeTail(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, sh := range []struct{ heads, hd, nrot int }{{4, 32, 16}, {8, 64, 16}, {2, 16, 16}} {
			for _, c := range []struct{ inv, made bool }{{false, false}, {true, false}, {false, true}, {true, true}} {
				inv := c.inv
				t.Run(fmt.Sprintf("%s/%dx%d/rot%d/inverse%v/made%v", d.API(), sh.heads, sh.hd, sh.nrot, inv, c.made), func(t *testing.T) {
					const R = 5
					W := sh.heads * sh.hd
					b := graph.New()
					x := b.In("x", W)
					// made: the turned value is an op's, which the card may
					// fuse the turn into, rather than an input.
					src := x
					if c.made {
						src = b.Emit(graph.OpAdd, []graph.ValueID{x, x}, []graph.Result{{Name: "x2", Width: W}}, nil, graph.Attrs{}, nil, nil)[0]
					}
					a := graph.Attrs{HeadDim: sh.hd, NRot: sh.nrot, Base: 10000, Layout: graph.RopeNorm, Offset: sh.hd - sh.nrot, Inverse: inv}
					y := b.Emit(graph.OpRope, []graph.ValueID{src},[]graph.Result{{Name: "y", Width: W}}, nil, a, nil, nil)[0]
					b.Out(y)
					_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Apart: true,
						Rotaries: []graph.Rotary{{NRot: sh.nrot, Base: 10000}},
						Step:     composer.Step{Rows: R, Page: 16, Keys: 64, Seqs: 1}, Device: gpuDevice(d)})
					if err != nil {
						t.Fatal(err)
					}
					rng := rand.New(rand.NewSource(3))
					in := make([]float32, R*W)
					for i := range in {
						in[i] = float32(rng.NormFloat64())
					}
					scale := 1.0
					if c.made {
						scale = 2
					}
					angles := make([]float32, R*sh.nrot)
					for r := range R {
						for p := range sh.nrot / 2 {
							th := float64(3+7*r) * math.Pow(10000, -2*float64(p)/float64(sh.nrot))
							angles[r*sh.nrot+2*p], angles[r*sh.nrot+2*p+1] = float32(math.Cos(th)), float32(math.Sin(th))
						}
					}
					tables := map[string][]uint32{}
					for _, c := range codes {
						for _, bd := range c.Binds {
							if o := bd.Operand; o.Kind == composer.OperandTable && o.Table == composer.TableAngles {
								tables[o.Name] = f32w(angles)
							}
						}
					}
					fr := &recRun{d: d, codes: codes, vals: map[graph.ValueID][]uint32{x: f32w(in), y: poisoned(R * W)},
						tables: tables, wts: nil, bufs: map[string]backend.Buf{}}
					if c.made {
						fr.vals[src] = poisoned(R * W)
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
					got := wf32(fr.vals[y])
					worst := 0.0
					for r := range R {
						for h := range sh.heads {
							row := in[r*W+h*sh.hd:]
							for i := range sh.hd {
								want := scale * float64(row[i])
								if j := i - (sh.hd - sh.nrot); j >= 0 {
									p := j / 2
									co, si := float64(angles[r*sh.nrot+2*p]), float64(angles[r*sh.nrot+2*p+1])
									if inv {
										si = -si
									}
									x0, x1 := scale*float64(row[sh.hd-sh.nrot+2*p]), scale*float64(row[sh.hd-sh.nrot+2*p+1])
									want = x0*co - x1*si
									if j%2 == 1 {
										want = x0*si + x1*co
									}
								}
								g := float64(got[r*W+h*sh.hd+i])
								if math.IsNaN(g) {
									t.Fatalf("row %d head %d element %d is NaN", r, h, i)
								}
								worst = math.Max(worst, math.Abs(g-want))
							}
						}
					}
					t.Logf("worst |diff| %.3g", worst)
					if worst > 1e-5 {
						t.Fatalf("the turned tail parts from the definition by %.3g", worst)
					}
				})
			}
		}
	}
}
