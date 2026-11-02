package composer

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestBindingsSayWhatACodeReadsAndWrites forms a residual add over a
// projection of a mixer's output (x plus W exp(V x)) and over a projection
// of x itself (x plus W x) on a card, with subgroups and without, at one row
// and at eight, over float and Q8_0 weights, at several widths -- the add a
// projection's epilogue, a kernel of its own, or a split's parts into
// planes and their combine. A binding a code holds but does not read or
// write says so (Binding.Read, Writes): no code writes x, some forms read it
// before the code writing the output and some in it, which the tier does
// not write over x (partForm.inPlace).
func TestBindingsSayWhatACodeReadsAndWrites(t *testing.T) {
	form := func(t *testing.T, dev Device, d, k, rows int, ty jlm.Type, mixed bool) (graph.ValueID, graph.ValueID, []Code) {
		b := graph.New()
		x := b.In("x", d)
		n := x
		if mixed {
			n = b.Emit(graph.OpMap, []graph.ValueID{b.MatVec(x, jlm.RoleAttnV, k)}, []graph.Result{{Name: "mix", Width: k}}, nil,
				graph.Attrs{Fn: graph.MapExp}, nil, nil)[0]
		}
		y := b.Add(x, b.MatVec(n, jlm.RoleAttnOut, d))
		b.Out(y)
		_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnOut: ty, jlm.RoleAttnV: ty},
			Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
		if err != nil {
			t.Fatal(err)
		}
		return x, y, codes
	}
	before, writer := 0, 0
	for _, dev := range []Device{{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 1}} {
		for _, sh := range [][2]int{{64, 64}, {768, 1536}, {2048, 512}, {80, 512}} {
			for _, rows := range []int{1, 8} {
				for _, ty := range []jlm.Type{jlm.TypeF32, jlm.TypeQ8} {
					if ty == jlm.TypeQ8 && sh[0]%32 != 0 {
						continue
					}
					for _, mixed := range []bool{true, false} {
						name := fmt.Sprintf("lanes %d %dx%d rows %d %v mixed %v", dev.Lanes, sh[0], sh[1], rows, ty, mixed)
						x, y, codes := form(t, dev, sh[0], sh[1], rows, ty, mixed)
						is := func(o Operand, v graph.ValueID) bool { return o.Kind == OperandValue && o.Value == v }
						lastRead, firstWrite := -1, len(codes)
						for i, c := range codes {
							for _, bd := range c.Binds {
								if is(bd.Operand, x) && bd.Read {
									lastRead = i
								}
								if is(bd.Operand, y) && bd.Writes {
									firstWrite = min(firstWrite, i)
								}
								if is(bd.Operand, x) && bd.Writes {
									t.Fatalf("%s: %s writes x, which no op of the graph writes", name, c.Name)
								}
							}
						}
						switch {
						case firstWrite == len(codes):
							t.Fatalf("%s: no code writes the output", name)
						case lastRead < 0:
							t.Fatalf("%s: no code reads x", name)
						case lastRead < firstWrite:
							before++
						case lastRead == firstWrite:
							writer++
						}
					}
				}
			}
		}
	}
	if before == 0 || writer == 0 {
		t.Fatalf("x read before the output's writer in %d forms, in it in %d: each must be asked", before, writer)
	}
	t.Logf("x read before the kernel writing the output in %d forms, in it in %d", before, writer)
}
