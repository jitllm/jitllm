package composer_test

import (
	"encoding/binary"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUIterate holds an iteration on every card to its trips
// taken in order, as TestComposedCPUIterate holds the CPU device: the
// shift-and-xor chain exact, the counter's multiply-add within 1e-6, at
// every row width and trip count, each output word poisoned first.
func TestComposedGPUIterate(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, c := range composer.IterateCases() {
				t.Run(c.String(), func(t *testing.T) {
					g := composer.IterateGraph(c)
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
					x := composer.IterateInput(c)
					n := c.N * c.Taps()
					vals := map[graph.ValueID]backend.Buf{0: alloc(x), g.Out[0]: alloc(poisoned(n))}
					own := map[composer.Operand]backend.Buf{}
					for _, cd := range codes {
						k, err := d.Compile(cd.GPU)
						if err != nil {
							t.Fatalf("compiling %s: %v", cd.Name, err)
						}
						defer k.Close()
						args := make([]backend.Buf, len(cd.Binds))
						for i, bd := range cd.Binds {
							o := bd.Operand
							if b, ok := vals[o.Value]; o.Kind == composer.OperandValue && ok {
								args[i] = b
								continue
							}
							if o.Kind == composer.OperandValue {
								t.Fatalf("%s binds value %d, neither the row nor its values", cd.Name, o.Value)
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
						if err := k.Launch(cd.Groups, cd.GPU.Group[0], args...); err != nil {
							t.Fatalf("launching %s: %v", cd.Name, err)
						}
					}
					raw := make([]byte, 4*n)
					if err := vals[g.Out[0]].Read(raw); err != nil {
						t.Fatal(err)
					}
					got := make([]uint32, n)
					for i := range got {
						got[i] = binary.LittleEndian.Uint32(raw[4*i:])
					}
					if bad := composer.IterateMismatch(c, got, composer.IterateWant(c, x)); bad != "" {
						t.Fatalf("%d code(s): %s", len(codes), bad)
					}
				})
			}
		})
	}
}
