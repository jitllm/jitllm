package composer_test

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// chase is a chain of steps takes through a table of words: each take's
// index the one before's word, the first's the start.
func chase(words, steps int) *graph.Graph {
	b := graph.New()
	tab := b.In("table", words)
	b.G.Values[tab].Elem = graph.ElemU32
	at := b.In("at", 1)
	b.G.Values[at].Elem = graph.ElemU32
	v := at
	for range steps {
		v = b.Emit(graph.OpTake, []graph.ValueID{tab, v}, []graph.Result{{Name: "next", Width: 1}}, nil, graph.Attrs{}, nil, nil)[0]
		b.G.Values[v].Elem = graph.ElemU32
	}
	b.Out(v)
	return b.G
}

// TestComposedGPUBindsWhatItReaches forms a chain of takes, fused into one
// kernel, and holds its GPU code to binding the table, the start and the
// result alone: an intermediate value no access of the code names is no
// parameter of it. Each was a parameter the caller bound to a word, and a
// thousand of them passed CUDA's 4 KiB of parameters. On every card the
// chain at 64 and 1024 steps lands on the word a walk of the table on the
// host reaches.
func TestComposedGPUBindsWhatItReaches(t *testing.T) {
	cuda := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32}
	for _, steps := range []int{4, 1024} {
		_, codes, err := composer.Default().Form(composer.Request{Graph: chase(256, steps), Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: cuda})
		if err != nil {
			t.Fatalf("%d steps: %v", steps, err)
		}
		binds, params := 0, 0
		for _, c := range codes {
			binds += len(c.Binds)
			params += len(c.GPU.Params)
			if len(c.Binds) != len(c.GPU.Params) {
				t.Fatalf("%d steps: %s binds %d and declares %d parameters", steps, c.Name, len(c.Binds), len(c.GPU.Params))
			}
		}
		if len(codes) != 1 || binds != 3 {
			t.Fatalf("%d steps: %d code(s) binding %d buffers; want one binding the table, the start and the result", steps, len(codes), binds)
		}
		t.Logf("%d steps: one code, %d bindings, %d parameters", steps, binds, params)
	}
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		testmodels.Missing(t, "no GPU backend on this host: the card arm proved nothing")
	}
	const lines = 4096
	rng := rand.New(rand.NewSource(7))
	order := rng.Perm(lines)
	words := make([]uint32, lines*4)
	for i, a := range order {
		words[a*4] = uint32(order[(i+1)%lines] * 4)
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, steps := range []int{64, 1024} {
				t.Run(fmt.Sprint(steps), func(t *testing.T) {
					g := chase(len(words), steps)
					_, codes, err := composer.Default().Form(composer.Request{Graph: g, Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
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
					start := uint32(order[0] * 4)
					vals := map[graph.ValueID]backend.Buf{0: alloc(words), 1: alloc([]uint32{start}), g.Out[0]: alloc([]uint32{guardWord})}
					for _, c := range codes {
						k, err := d.Compile(c.GPU)
						if err != nil {
							t.Fatalf("compiling %s: %v", c.Name, err)
						}
						defer k.Close()
						args := make([]backend.Buf, len(c.Binds))
						for i, bd := range c.Binds {
							b, ok := vals[bd.Operand.Value]
							if bd.Operand.Kind != composer.OperandValue || !ok {
								t.Fatalf("%s binds %+v, neither the table, the start nor the result", c.Name, bd.Operand)
							}
							args[i] = b
						}
						if err := k.Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
							t.Fatalf("launching %s: %v", c.Name, err)
						}
					}
					raw := make([]byte, 4)
					if err := vals[g.Out[0]].Read(raw); err != nil {
						t.Fatal(err)
					}
					want := start
					for range steps {
						want = words[want]
					}
					if got := binary.LittleEndian.Uint32(raw); got != want {
						t.Fatalf("%d takes land on word %d, the host's walk on %d", steps, got, want)
					}
				})
			}
		})
	}
}
