package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// squaresSummed is a row squared, then summed in spans of span: the
// reduction reads the squares as outputs rows of span, the map writes them
// as one row.
func squaresSummed(outputs, span int) *graph.Graph {
	b := graph.New()
	x := b.In("x", outputs*span)
	y := b.Map("sq", graph.Attrs{Fn: graph.MapMul}, nil, outputs*span, x, x)
	b.Out(b.Emit(graph.OpReduce, []graph.ValueID{y}, []graph.Result{{Name: "sum", Width: outputs}}, nil,
		graph.Attrs{Fn: graph.MapAdd, Span: span}, nil, nil)[0])
	return b.G
}

// TestComposedGPURecomputesThroughAView holds a map read by a reduction
// through another shape of its elements to running in the reduction: x
// squared, then summed in spans, its input and output apart, binds no
// buffer of the squares on a device of CUDA's statement, at ragged spans. The map wrote its row flat and the
// reduction read it as rows of a span, and Recompute, holding a reader to
// the map's own shape, left them two kernels, every square written and
// read back. On every card the sums match the host's.
func TestComposedGPURecomputesThroughAView(t *testing.T) {
	cuda := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32}
	shapes := [][2]int{{37, 7}, {1000, 32}, {3, 4096}, {129, 100}}
	for _, sh := range shapes {
		_, codes, err := composer.Default().Form(composer.Request{Graph: squaresSummed(sh[0], sh[1]), Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: cuda, Apart: true})
		if err != nil {
			t.Fatalf("%d spans of %d: %v", sh[0], sh[1], err)
		}
		for _, c := range codes {
			for _, bd := range c.Binds {
				if bd.Operand.Kind == composer.OperandValue && bd.Operand.Value == 1 {
					t.Fatalf("%d spans of %d: %s binds the squares (%d codes): the map was not run in the sum", sh[0], sh[1], c.Name, len(codes))
				}
			}
		}
		t.Logf("%d spans of %d: %d code(s), none binding the squares", sh[0], sh[1], len(codes))
	}
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		testmodels.Missing(t, "no GPU backend on this host: the card arm proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, sh := range shapes {
				t.Run(fmt.Sprintf("%dx%d", sh[0], sh[1]), func(t *testing.T) {
					outputs, span := sh[0], sh[1]
					g := squaresSummed(outputs, span)
					_, codes, err := composer.Default().Form(composer.Request{Graph: g, Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d), Apart: true})
					if err != nil {
						t.Fatal(err)
					}
					var bufs []backend.Buf
					defer func() {
						for _, b := range bufs {
							b.Free()
						}
					}()
					alloc := func(w []uint32) backend.Buf {
						t.Helper()
						b, err := d.Alloc(4 * len(w))
						if err != nil {
							t.Fatal(err)
						}
						bufs = append(bufs, b)
						p := make([]byte, 4*len(w))
						for i, v := range w {
							binary.LittleEndian.PutUint32(p[4*i:], v)
						}
						if err := b.Write(p); err != nil {
							t.Fatal(err)
						}
						return b
					}
					x := make([]float32, outputs*span)
					xw := make([]uint32, len(x))
					for i := range x {
						x[i] = float32(math.Sin(0.31*float64(i) + 0.2))
						xw[i] = math.Float32bits(x[i])
					}
					vals := map[graph.ValueID]backend.Buf{0: alloc(xw), g.Out[0]: alloc(poisoned(outputs))}
					own := map[composer.Operand]backend.Buf{}
					for _, c := range codes {
						k, err := d.Compile(c.GPU)
						if err != nil {
							t.Fatalf("compiling %s: %v", c.Name, err)
						}
						defer k.Close()
						args := make([]backend.Buf, len(c.Binds))
						for i, bd := range c.Binds {
							o := bd.Operand
							if b, ok := vals[o.Value]; o.Kind == composer.OperandValue && ok {
								args[i] = b
								continue
							}
							if o.Kind == composer.OperandValue {
								t.Fatalf("%s binds value %d, neither the row nor the sums", c.Name, o.Value)
							}
							b, ok := own[o]
							if !ok {
								w := bd.Words
								if w == nil {
									w = poisoned(int(max(bd.Bytes, 4) / 4))
								}
								b = alloc(w)
								own[o] = b
							}
							args[i] = b
						}
						if err := k.Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
							t.Fatalf("launching %s: %v", c.Name, err)
						}
					}
					raw := make([]byte, 4*outputs)
					if err := vals[g.Out[0]].Read(raw); err != nil {
						t.Fatal(err)
					}
					worst := 0.0
					for o := range outputs {
						want := 0.0
						for j := range span {
							v := float64(x[o*span+j])
							want += v * v
						}
						got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*o:])))
						if math.IsNaN(got) || math.IsInf(got, 0) {
							t.Fatalf("sum %d is %v", o, got)
						}
						worst = max(worst, math.Abs(got-want)/max(math.Abs(want), 1))
					}
					if worst > 1e-5 {
						t.Fatalf("worst relative error %.3g against the host's sums", worst)
					}
				})
			}
		})
	}
}
