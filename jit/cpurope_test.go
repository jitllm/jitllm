package jit

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestComposedCPURope forms OpRope on the CPU, full and partial, both
// layouts, at one row and at a chunk of rows: each head's first NRot turned
// in pairs by the row's angles, held to the definition, and every element
// of the unturned tail equal to the input's, bit for bit.
func TestComposedCPURope(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	ran := 0
	for _, sh := range []struct{ heads, hd, nrot int }{{4, 128, 64}, {2, 256, 64}, {3, 64, 32}, {2, 128, 128}, {2, 80, 40}} {
		for _, layout := range []graph.RopeLayout{graph.RopeNorm, graph.RopeNeox} {
			for _, R := range []int{1, 5} {
				name := fmt.Sprintf("%dx%d/rot%d/layout%d/rows%d", sh.heads, sh.hd, sh.nrot, layout, R)
				w := sh.heads * sh.hd
				b := graph.New()
				x := b.In("x", w)
				y := b.Emit(graph.OpRope, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: w}}, nil,
					graph.Attrs{HeadDim: sh.hd, NRot: sh.nrot, Layout: layout}, nil, nil)[0]
				b.Out(y)
				capRows := R
				if R > 1 {
					capRows = 8
				}
				_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
					Step: Step{Rows: capRows, Page: 64, Keys: 64, Seqs: 1}, Device: cpuDevice()})
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				in := make([]float32, capRows*w)
				for i := range in {
					in[i] = float32(rng.NormFloat64())
				}
				cs := make([]float32, capRows*sh.nrot)
				for r := range capRows {
					for p := range sh.nrot / 2 {
						a := float64(r+3) * math.Pow(10000, -2*float64(p)/float64(sh.nrot))
						cs[r*sh.nrot+2*p], cs[r*sh.nrot+2*p+1] = float32(math.Cos(a)), float32(math.Sin(a))
					}
				}
				out := make([]float32, capRows*w)
				for i := range out {
					out[i] = float32(math.NaN())
				}
				bufs := map[string][]uint32{"x": f32Words(append([]float32(nil), in...)), b.G.Values[y].Name: f32Words(out), "rope.cs": f32Words(cs), "rows.n": {uint32(R)}}
				if R > 1 {
					err = runCPURows(codes, b.G, bufs, R)
				} else {
					err = runCPUCalls(codes, b.G, bufs)
				}
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				// The in-place pass may leave the output over the input or in a
				// scratch, where the last kernel writes it.
				for _, bd := range codes[len(codes)-1].Binds {
					switch o := bd.Operand; {
					case !bd.Writes:
					case o.Kind == OperandScratch:
						copy(out, wordsF32(bufs["own:"+o.Name]))
					case o.Kind == OperandValue && o.Value != y:
						copy(out, wordsF32(bufs[b.G.Values[o.Value].Name]))
					}
				}
				worst := 0.0
				for r := range R {
					for h := range sh.heads {
						base := r*w + h*sh.hd
						for p := range sh.nrot / 2 {
							i0, i1 := base+2*p, base+2*p+1
							if layout == graph.RopeNeox {
								i0, i1 = base+p, base+p+sh.nrot/2
							}
							c, s := float64(cs[r*sh.nrot+2*p]), float64(cs[r*sh.nrot+2*p+1])
							w0 := float64(in[i0])*c - float64(in[i1])*s
							w1 := float64(in[i0])*s + float64(in[i1])*c
							worst = max(worst, math.Abs(float64(out[i0])-w0), math.Abs(float64(out[i1])-w1))
							if math.IsNaN(float64(out[i0])) || math.IsNaN(float64(out[i1])) {
								t.Fatalf("%s: row %d head %d pair %d unwritten\n%s", name, r, h, p, describeCodes(codes))
							}
						}
						moved := sh.nrot == sh.hd
						for j := sh.nrot; j < sh.hd; j++ {
							if math.Float32bits(out[base+j]) != math.Float32bits(in[base+j]) {
								t.Fatalf("%s: row %d head %d tail element %d is %g, the input's %g", name, r, h, j, out[base+j], in[base+j])
							}
							// The tail moved by one element must part.
							moved = moved || math.Float32bits(out[base+j]) != math.Float32bits(in[base+j-1])
						}
						if !moved {
							t.Fatalf("%s: the tail moved by one element matches: the gate cannot see it", name)
						}
					}
				}
				if worst > 1e-5 {
					t.Fatalf("%s: worst turned element off by %.3e", name, worst)
				}
				ran++
			}
		}
	}
	t.Logf("%d forms held", ran)
}
