package jit

import (
	"fmt"
	"slices"
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
// write says so (Binding.Read, Writes), and an input read only at the
// elements an output is written at names it (Binding.AtElement): over the
// mixer's output some form reads x in the code writing the output, at its
// element, and names it (the output may take x's storage), and over x
// itself (float, so the projection reads x, not its int8 codes) no code
// reading x at every element names the output.
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
	before, named, apart, refused := 0, 0, 0, 0
	for _, dev := range []Device{{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 1}} {
		for _, sh := range [][2]int{{64, 64}, {768, 1536}, {2048, 512}} {
			for _, rows := range []int{1, 8} {
				for _, ty := range []jlm.Type{jlm.TypeF32, jlm.TypeQ8} {
					for _, mixed := range []bool{true, false} {
						name := fmt.Sprintf("lanes %d %dx%d rows %d %v mixed %v", dev.Lanes, sh[0], sh[1], rows, ty, mixed)
						x, y, codes := form(t, dev, sh[0], sh[1], rows, ty, mixed)
						is := func(o Operand, v graph.ValueID) bool { return o.Kind == OperandValue && o.Value == v }
						lastRead, firstWrite, atWrite := -1, len(codes), false
						for i, c := range codes {
							for _, bd := range c.Binds {
								if is(bd.Operand, x) && bd.Read {
									lastRead = i
									atWrite = slices.ContainsFunc(bd.AtElement, func(o Operand) bool { return is(o, y) })
								}
								if is(bd.Operand, y) && bd.Writes {
									firstWrite = min(firstWrite, i)
								}
							}
						}
						switch {
						case firstWrite == len(codes):
							t.Fatalf("%s: no code writes the output", name)
						case mixed && lastRead < firstWrite:
							before++
						case mixed && lastRead == firstWrite && atWrite:
							named++
						case mixed:
							apart++
						case ty == jlm.TypeF32 && lastRead == firstWrite && atWrite:
							t.Fatalf("%s: the code writing the output reads float x at every element, and names the output", name)
						case ty == jlm.TypeF32 && lastRead == firstWrite:
							refused++
						}
					}
				}
			}
		}
	}
	if named == 0 || refused == 0 {
		t.Fatalf("x read in the writer at its element in %d forms, at every element in %d: each must be asked", named, refused)
	}
	t.Logf("over the mixer the output takes x's storage in %d forms reading x before it, %d reading it at its element, and not in %d; over x the writer reads every element in %d", before, named, apart, refused)
}
