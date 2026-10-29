package jit_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// runGraphValues runs codes on d over the graph's values: vals holds the
// inputs and receives every output, each value followed by guard words that
// must stay poisoned; scratch is poisoned and constant blocks written.
func runGraphValues(d backend.Device, codes []jit.Code, kerns []backend.Kernel, g *graph.Graph, vals map[graph.ValueID][]uint32) error {
	const guard = 64
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	alloc := func(w []uint32) (backend.Buf, error) {
		b, err := d.Alloc(max(4*len(w), 4))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		p := make([]byte, 4*len(w))
		for i, v := range w {
			binary.LittleEndian.PutUint32(p[4*i:], v)
		}
		if len(p) > 0 {
			err = b.Write(p)
		}
		return b, err
	}
	reach := map[jit.Operand]int64{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			reach[bd.Operand] = max(reach[bd.Operand], bd.Bytes)
		}
	}
	values := map[graph.ValueID]backend.Buf{}
	own := map[jit.Operand]backend.Buf{}
	for ci, c := range codes {
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == jit.OperandValue:
				b, ok := values[o.Value]
				if !ok {
					w := poisoned(g.Values[o.Value].Width + guard)
					if in, ok := vals[o.Value]; ok {
						w = append(append([]uint32(nil), in...), poisoned(guard)...)
					}
					var err error
					if b, err = alloc(w); err != nil {
						return err
					}
					values[o.Value] = b
				}
				args[i] = b
			case o.Kind == jit.OperandWeight || o.Kind == jit.OperandVector:
				return fmt.Errorf("%s binds %+v, which the sampler does not hold", c.Name, o)
			default:
				b, ok := own[o]
				if !ok {
					w := bd.Words
					if w == nil {
						w = poisoned(int(max(reach[o], 4)/4) + guard)
					}
					var err error
					if b, err = alloc(w); err != nil {
						return err
					}
					own[o] = b
				}
				args[i] = b
			}
		}
		if err := kerns[ci].Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
			return fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	for _, v := range g.Out {
		b, ok := values[v]
		if !ok {
			return fmt.Errorf("no kernel binds the output %s", g.Values[v].Name)
		}
		n := g.Values[v].Width
		raw := make([]byte, 4*(n+guard))
		if err := b.Read(raw); err != nil {
			return err
		}
		w := make([]uint32, n)
		for i := range w {
			w[i] = binary.LittleEndian.Uint32(raw[4*i:])
		}
		for i := range guard {
			if x := binary.LittleEndian.Uint32(raw[4*(n+i):]); x != guardWord {
				return fmt.Errorf("the output %s: word %#x written %d past its end", g.Values[v].Name, x, i)
			}
		}
		vals[v] = w
	}
	return nil
}

// sampleOnGPU forms the sampler's graph g on d under dv and runs c.
func sampleOnGPU(t *testing.T, d backend.Device, dv jit.Device, g *graph.Graph, c jit.SampleCase) ([]float32, []uint32) {
	t.Helper()
	_, codes, err := jit.Default().Form(jit.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: dv})
	if err != nil {
		t.Fatalf("%s: forming: %v", c.Name, err)
	}
	op, err := compileLone(d, codes)
	if err != nil {
		t.Fatalf("%s: %v", c.Name, err)
	}
	defer op.close()
	lw := make([]uint32, len(c.Logits))
	for i, v := range c.Logits {
		lw[i] = math.Float32bits(v)
	}
	vals := map[graph.ValueID][]uint32{g.In[0]: lw, g.In[1]: jit.SampleArgs(c), g.In[2]: {math.Float32bits(c.Pen)}}
	if err := runGraphValues(d, codes, op.kerns, g, vals); err != nil {
		t.Fatalf("%s: %v", c.Name, err)
	}
	gv := make([]float32, len(vals[g.Out[0]]))
	for i, w := range vals[g.Out[0]] {
		gv[i] = math.Float32frombits(w)
	}
	return gv, vals[g.Out[1]]
}

// TestComposedGPUSampleTopK forms the sampler's candidates (graph.SampleTopK)
// on every GPU device present, each at the division its device states, and
// holds every case (jit.SampleCases) to the host sampler's candidates bit
// for bit in value and id, guard words past every output, ties among the
// kept on some. The violation: the form with the penalty on the id after
// each history entry's (jit.SampleNeighbour) parts on every case with a
// history.
func TestComposedGPUSampleTopK(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	cases := jit.SampleCases()
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dv := gpuDevice(d)
			ran, ties := 0, 0
			for _, c := range cases {
				s := graph.SampleTopK{NVocab: c.V, K: c.K, Hist: c.B}
				g := s.Build()
				gv, gi := sampleOnGPU(t, d, dv, g, c)
				wv, wi := jit.SampleWant(c, false)
				if bad := jit.SampleMismatch(gv, gi, wv, wi); bad > 0 {
					t.Errorf("%s: %d of %d places part from the host's; first %v %v, want %v %v", c.Name, bad, len(wv), gv[:min(4, len(gv))], gi[:min(4, len(gi))], wv[:min(4, len(wv))], wi[:min(4, len(wi))])
					continue
				}
				if jit.SampleTieHigh(gi, wv) {
					ties++
				}
				if len(c.Hist) > 0 {
					bv, bi := sampleOnGPU(t, d, dv, jit.SampleNeighbour(g), c)
					if jit.SampleMismatch(bv, bi, wv, wi) == 0 {
						t.Errorf("%s: the form with the history moved to the next id matches the host: the gate cannot see it", c.Name)
					}
				}
				ran++
			}
			if ran != len(cases) {
				t.Fatalf("%d of %d cases held", ran, len(cases))
			}
			if ties == 0 {
				t.Fatal("no case keeps two places of one value: the ties to the lower id are not held")
			}
			t.Logf("%d cases held bit for bit (exact division %v), %d with ties among the kept", ran, dv.ExactDiv, ties)
		})
	}
}

// TestComposedGPUEqual is TestComposedCPUEqual on every GPU device present:
// MapEqual over two index rows (their integers, words a float would round
// together) and two float rows, at ragged widths, guard words past the
// output; the equality one id off must part.
func TestComposedGPUEqual(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			ran, parted := 0, 0
			for _, ints := range []bool{true, false} {
				for _, n := range []int{1, 2, 3, 7, 31, 32, 33, 257} {
					b := graph.New()
					elem := graph.ElemF32
					if ints {
						elem = graph.ElemU32
					}
					x, y := b.In("x", n), b.In("y", n)
					b.G.Values[x].Elem, b.G.Values[y].Elem = elem, elem
					b.Out(b.Emit(graph.OpMap, []graph.ValueID{x, y}, []graph.Result{{Name: "eq", Width: n}}, nil, graph.Attrs{Fn: graph.MapEqual}, nil, nil)[0])
					_, codes, err := jit.Default().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
					if err != nil {
						t.Fatalf("ints %v width %d: %v", ints, n, err)
					}
					op, err := compileLone(d, codes)
					if err != nil {
						t.Fatal(err)
					}
					xs, ys := make([]uint32, n), make([]uint32, n)
					fx := []float32{0, float32(math.Copysign(0, -1)), 1, float32(math.NaN())}
					for i := range xs {
						a, c := uint32((i*7+3)%4), uint32((i*5+1)%4)
						if ints {
							xs[i], ys[i] = a+1<<25, c+1<<25
						} else {
							xs[i], ys[i] = math.Float32bits(fx[a]), math.Float32bits(fx[c])
						}
					}
					vals := map[graph.ValueID][]uint32{x: xs, y: ys}
					err = runGraphValues(d, codes, op.kerns, b.G, vals)
					op.close()
					if err != nil {
						t.Fatalf("ints %v width %d: %v", ints, n, err)
					}
					for i, w := range vals[b.G.Out[0]] {
						got := math.Float32frombits(w)
						want := float32(0)
						if ints && xs[i] == ys[i] || !ints && math.Float32frombits(xs[i]) == math.Float32frombits(ys[i]) {
							want = 1
						}
						if got != want {
							t.Fatalf("ints %v width %d: element %d is %v, want %v (%#x, %#x)", ints, n, i, got, want, xs[i], ys[i])
						}
						if off := ints && xs[i]+1 == ys[i]; off != (got == 1) {
							parted++
						}
					}
					ran++
				}
			}
			if parted == 0 {
				t.Fatal("the equality one id off agrees with the form everywhere: the gate cannot see it")
			}
			t.Logf("%d widths held; one id off, %d elements part", ran, parted)
		})
	}
}

// TestComposedGPUSampleGreedy is TestComposedCPUSampleGreedy on every GPU
// device present, at widths whose argmax is cut into segments that divide
// it and into segments the last of which reads past it (a vocabulary of
// 201088), each row's best alone at its start and its end, and in its
// middle tied with an earlier one. The violation: the logits negated must
// part.
func TestComposedGPUSampleGreedy(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	widths := []int{1, 7, 33, 1531, 4096, 151936, 201088}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dv := gpuDevice(d)
			rng := rand.New(rand.NewSource(5))
			ran, parted := 0, 0
			for _, n := range widths {
				g := (&graph.Sample{NVocab: n}).Build()
				_, codes, err := jit.Default().Form(jit.Request{Graph: g, Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: dv})
				if err != nil {
					t.Fatalf("width %d: forming: %v", n, err)
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Fatalf("width %d: %v", n, err)
				}
				run := func(l []float32) int {
					lw := make([]uint32, n)
					for i, v := range l {
						lw[i] = math.Float32bits(v)
					}
					vals := map[graph.ValueID][]uint32{g.In[0]: lw}
					if err := runGraphValues(d, codes, op.kerns, g, vals); err != nil {
						t.Fatalf("width %d: %v", n, err)
					}
					return int(vals[g.Out[0]][0])
				}
				for _, c := range [][2]int{{0, -1}, {n / 2, n / 4}, {n - 1, -1}} {
					at, tie := c[0], c[1]
					l := make([]float32, n)
					for i := range l {
						l[i] = float32(rng.NormFloat64())
					}
					l[at] = 9
					want := at
					if tie >= 0 && tie < at {
						l[tie] = 9
						want = tie
					}
					if got := run(l); got != want {
						t.Errorf("width %d, best at %d: token %d, want %d", n, at, got, want)
						continue
					}
					for i := range l {
						l[i] = -l[i]
					}
					if n > 2 && run(l) != want {
						parted++
					}
					ran++
				}
				op.close()
			}
			if ran != 3*len(widths) {
				t.Fatalf("%d of %d rows held", ran, 3*len(widths))
			}
			if parted == 0 {
				t.Fatal("the negated logits give every token: the gate cannot see a wrong one")
			}
			t.Logf("%d rows held", ran)
		})
	}
}
