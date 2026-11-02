package composer_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
)

// TestComposedGPUContractPairs runs a gated FFN's gate and up, 5632 rows of
// 2048 floats, and silu(gate)*up on every GPU device present: one launch
// whose element takes both rows, its value held to float64, and to the
// float64 of the operands swapped to show the gate tells them apart.
func TestComposedGPUContractPairs(t *testing.T) {
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const K, F = 2048, 5632
	rng := rand.New(rand.NewSource(11))
	w := map[jlm.Role][]byte{}
	for _, r := range []jlm.Role{jlm.RoleFFNGate, jlm.RoleFFNUp} {
		w[r] = tiledWeights(quant.F32, layout.Float32, F, K, rng)
	}
	x := make([]float32, K)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	dot := func(r jlm.Role, i int) float64 {
		s := 0.0
		for j := range K {
			s += float64(math.Float32frombits(binary.LittleEndian.Uint32(w[r][4*(i*K+j):]))) * float64(x[j])
		}
		return s
	}
	silu := func(v float64) float64 { return v / (1 + math.Exp(-v)) }
	want, swapped := make([]float64, F), make([]float64, F)
	for i := range F {
		g, u := dot(jlm.RoleFFNGate, i), dot(jlm.RoleFFNUp, i)
		want[i], swapped[i] = silu(g)*u, silu(u)*g
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			b := graph.New()
			in := b.In("x", K)
			g := b.MatVec(in, jlm.RoleFFNGate, F)
			u := b.MatVec(in, jlm.RoleFFNUp, F)
			var e jlm.ExprBuilder
			e.Fn(jlm.FnMul, e.SiLU(jlm.Operand(0)), jlm.Operand(1))
			act := jlm.Expr{Terms: e.T}
			h, err := b.Stated("act", &act, []graph.ValueID{g, u}, F, graph.Facts{})
			if err != nil {
				t.Fatal(err)
			}
			b.Out(h)
			formats := map[jlm.Role]jlm.Type{jlm.RoleFFNGate: jlm.TypeF32, jlm.RoleFFNUp: jlm.TypeF32}
			_, codes, err := composer.Default().WithoutInt8(composer.DeviceGPU).Form(composer.Request{Graph: b.G, Formats: formats,
				Step: composer.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: gpuDevice(d)})
			if err != nil {
				t.Fatal(err)
			}
			if len(codes) != 1 || !slices.Contains(codes[0].Passes, "contractpairs") {
				t.Fatalf("gate, up and their product formed as %d launches, want one paired", len(codes))
			}
			op, err := compileLone(d, codes)
			if err != nil {
				t.Fatal(err)
			}
			defer op.close()
			var bufs []backend.Buf
			defer func() {
				for _, bf := range bufs {
					bf.Free()
				}
			}()
			up := func(v []uint32) backend.Buf {
				p := make([]byte, max(4*len(v), 4))
				for i, u := range v {
					binary.LittleEndian.PutUint32(p[4*i:], u)
				}
				bf, err := d.Alloc(len(p))
				if err != nil {
					t.Fatal(err)
				}
				if err := bf.Write(p); err != nil {
					t.Fatal(err)
				}
				bufs = append(bufs, bf)
				return bf
			}
			xs := make([]uint32, K)
			for i, v := range x {
				xs[i] = math.Float32bits(v)
			}
			var out backend.Buf
			var args []backend.Buf
			for _, bd := range codes[0].Binds {
				o := bd.Operand
				var bf backend.Buf
				switch {
				case o.Kind == composer.OperandValue && o.Value == in:
					bf = up(xs)
				case o.Kind == composer.OperandValue:
					bf = up(poisoned(int(max(bd.Bytes/4, 1))))
					if o.Value == h {
						out = bf
					}
				case o.Kind == composer.OperandWeight:
					qs, dw, sc, err := layout.PackWeights(layout.Float32, w[o.Role], F, K)
					if err != nil {
						t.Fatal(err)
					}
					bf = up([][]uint32{qs, dw, sc}[o.Region])
				default:
					ws := bd.Words
					if ws == nil {
						ws = make([]uint32, int(max(bd.Bytes, 4)/4))
					}
					bf = up(ws)
				}
				args = append(args, bf)
			}
			if out == nil {
				t.Fatal("the product is not bound")
			}
			if err := op.kerns[0].Launch(codes[0].LiveGroups(1), codes[0].GPU.Group[0], args...); err != nil {
				t.Fatal(err)
			}
			p := make([]byte, 4*F)
			if err := out.Read(p); err != nil {
				t.Fatal(err)
			}
			nmse := func(ref []float64) float64 {
				var num, den float64
				for i := range F {
					got := float64(math.Float32frombits(binary.LittleEndian.Uint32(p[4*i:])))
					num, den = num+(got-ref[i])*(got-ref[i]), den+ref[i]*ref[i]
				}
				return num / den
			}
			if e := nmse(want); !(e < 1e-10) {
				t.Fatalf("NMSE %.3g against float64", e)
			}
			if e := nmse(swapped); !(e > 1e-3) {
				t.Fatalf("NMSE %.3g against gate and up swapped: the gate cannot tell them apart", e)
			}
			t.Logf("%s: gate and up one launch of %d workgroups of %d", d.API(), codes[0].LiveGroups(1), codes[0].GPU.Group[0])
		})
	}
}
