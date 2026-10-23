package board

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestComposedArgmaxRunsOnTheGPU runs the GPU sampling the composer forms
// -- the argmax split into segments and their combine -- on every device a
// backend opens, on logits whose answer the test knows: random ones, a
// maximum planted at a random place, at the first and at the last, and two
// equal maxima far apart and near, of which the lower index is the answer.
func TestComposedArgmaxRunsOnTheGPU(t *testing.T) {
	f := composing(t, *boardModel+".jlm")
	s := jit.Step{Rows: 1, Page: 64, Keys: 9}
	d := cost.Device{Device: sm70}
	ccs, err := cost.EstimateCodes(jit.Default(), f, s, func(jit.Part) cost.Device { return d }, true)
	if err != nil {
		t.Fatal(err)
	}
	parts, err := jit.Front(f, false)
	if err != nil {
		t.Fatal(err)
	}
	var sample []cost.CodeCost
	for _, cc := range ccs {
		if parts[cc.Part].Kind == jit.PartSample {
			sample = append(sample, cc)
		}
	}
	if len(sample) < 2 {
		t.Fatalf("the sampling formed %d kernels; the split forms a part and a combine", len(sample))
	}
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	ran := 0
	for _, dev := range devs {
		ran++
		t.Run(dev.API(), func(t *testing.T) { argmaxOn(t, dev, sample) })
	}
	if ran == 0 {
		t.Skip("no device")
	}
}

func argmaxOn(t *testing.T, dev backend.Device, codes []cost.CodeCost) {
	size := map[jit.Operand]int64{}
	var logits jit.Operand
	var token jit.Operand
	for _, cc := range codes {
		for j, b := range cc.Code.Binds {
			at := cc.Binds[j].Storage
			size[at] = max(size[at], cc.Binds[j].Reach)
			switch {
			case !b.Writes && b.Operand.Kind == jit.OperandValue:
				logits = at
			case b.Writes && b.Operand.Kind == jit.OperandValue && b.Operand.Elem == ir.I32:
				token = at
			}
		}
	}
	if size[logits] == 0 || size[token] == 0 {
		t.Fatalf("no logits (%d bytes) or token (%d bytes) among the sampling's bindings", size[logits], size[token])
	}
	bufs := map[jit.Operand]backend.Buf{}
	for at, n := range size {
		b, err := dev.Alloc(int(n) + 4096)
		if err != nil {
			t.Fatal(err)
		}
		defer b.Free()
		bufs[at] = b
	}
	type launch struct {
		k             backend.Kernel
		groups, width int
		bufs          []backend.Buf
	}
	var runs []launch
	for _, cc := range codes {
		k, err := dev.Compile(cc.Code.GPU)
		if err != nil {
			t.Fatalf("%s: %v", cc.Code.Name, err)
		}
		defer k.Close()
		var bs []backend.Buf
		for j := range cc.Code.Binds {
			bs = append(bs, bufs[cc.Binds[j].Storage])
		}
		runs = append(runs, launch{k, cc.Code.Groups, cc.Code.GPU.Group[0], bs})
	}
	n := int(size[logits] / 4)
	rng := rand.New(rand.NewSource(7))
	cases := []struct {
		name  string
		plant func(x []float32)
	}{
		{"random", func(x []float32) {}},
		{"planted", func(x []float32) { x[rng.Intn(len(x))] = 100 }},
		{"first", func(x []float32) { x[0] = 100 }},
		{"last", func(x []float32) { x[len(x)-1] = 100 }},
		{"ties", func(x []float32) {
			i, j := rng.Intn(len(x)/2), len(x)/2+rng.Intn(len(x)/2)
			x[i], x[j] = 50, 50
		}},
		{"near ties", func(x []float32) {
			i := rng.Intn(len(x) - 512)
			x[i], x[i+1+rng.Intn(511)] = 50, 50
		}},
	}
	for _, c := range cases {
		name, plant := c.name, c.plant
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		plant(x)
		want := 0
		for i, v := range x {
			if v > x[want] {
				want = i
			}
		}
		raw := make([]byte, size[logits]+4096)
		for i, v := range x {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		if err := bufs[logits].Write(raw); err != nil {
			t.Fatal(err)
		}
		var err error
		dev.Session(func(se backend.Session) {
			for _, r := range runs {
				if err = se.Launch(r.k, r.groups, r.width, r.bufs...); err != nil {
					return
				}
			}
			err = se.Sync()
		})
		if err != nil {
			t.Fatal(err)
		}
		out := make([]byte, size[token]+4096)
		if err := bufs[token].Read(out); err != nil {
			t.Fatal(err)
		}
		if got := int(int32(binary.LittleEndian.Uint32(out))); got != want {
			t.Errorf("%s: token %d (logit %g), the argmax of %d logits is %d (logit %g)", name, got, at(x, got), n, want, x[want])
		}
	}
}

func at(x []float32, i int) float32 {
	if i < 0 || i >= len(x) {
		return float32(math.NaN())
	}
	return x[i]
}
