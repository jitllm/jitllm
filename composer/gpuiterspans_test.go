package composer_test

import (
	"encoding/binary"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUIterateSpans holds an iteration summing terms over runs of
// its row on every card, as TestComposedCPUIterateSpans holds the CPU
// device: runs of one element to a workgroup of 256, rows of one run to
// many, so a launch's last workgroup holds threads past the row, each
// output poisoned and guarded past its end. A run of at most a subgroup is
// asserted formed as exchanges across lanes on a card that guarantees
// them, any other as a sum over the workgroup.
func TestComposedGPUIterateSpans(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			for _, c := range composer.SpanCases(256, 256, 256) {
				t.Run(c.String(), func(t *testing.T) {
					g := composer.SpanGraph(c)
					req := composer.Request{Graph: g, Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: dev, Apart: true}
					ks, codes, err := composer.Default().FormKernels(req)
					if err != nil {
						t.Fatal(err)
					}
					f := composer.SpanFormOf(ks)
					lanes := c.Span <= ir.SubgroupLanes && dev.Lanes >= ir.SubgroupLanes
					switch {
					case c.Span > 1 && lanes && (f.Shuffles == 0 || f.GroupSums != 0):
						t.Fatalf("runs of %d formed %+v, not exchanges across lanes", c.Span, f)
					case c.Span > 1 && !lanes && f.GroupSums == 0:
						t.Fatalf("runs of %d formed %+v, not a sum over the workgroup", c.Span, f)
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
					in := composer.SpanInputs(c)
					vals := map[graph.ValueID]backend.Buf{}
					for _, v := range g.Out {
						vals[v] = alloc(poisoned(g.Values[v].Width + 16))
					}
					for id, v := range g.Values {
						if w, ok := in[v.Name]; ok {
							vals[graph.ValueID(id)] = alloc(w)
						}
					}
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
					var outs [][]uint32
					for _, v := range g.Out {
						n := g.Values[v].Width
						raw := make([]byte, 4*(n+16))
						if err := vals[v].Read(raw); err != nil {
							t.Fatal(err)
						}
						o := make([]uint32, n+16)
						for i := range o {
							o[i] = binary.LittleEndian.Uint32(raw[4*i:])
						}
						for _, w := range o[n:] {
							if w != guardWord {
								t.Fatalf("%s was written past its end: %#x", g.Values[v].Name, w)
							}
						}
						outs = append(outs, o)
					}
					if bad := composer.SpanMismatch(c, composer.SpanGot(c, outs), composer.SpanWant(c, in)); bad != "" {
						t.Fatalf("%d code(s), %+v: %s", len(codes), f, bad)
					}
				})
			}
		})
	}
}
