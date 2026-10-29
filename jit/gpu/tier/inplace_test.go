package tier

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// TestAKernelReadingXWhereItWritesIsNotInPlace forms a block with no MLP,
// x plus the projection of a row of the mixer's width, at mamba-130m's
// widths (768 over 1536), at a fixture's (64 over 128) and at 80, whose rows
// fill no workgroup and split into planes, and holds
// partForm.inPlace to the rule: a form whose kernel writing the output
// reads x is not written over x, though it reads x only at the elements it
// writes -- the launch clamps its surplus threads to the last element and
// each lane storing an element reads it, some after another's store, so
// the residual is added twice (mamba-130m on the V100 parted from its float
// reference by 3.3 times the tier's distance and changed under poisoned
// buffers). A form reading x before its first write is.
func TestAKernelReadingXWhereItWritesIsNotInPlace(t *testing.T) {
	over, before := 0, 0
	for _, sh := range [][2]int{{768, 1536}, {64, 128}, {2048, 512}, {80, 512}} {
		for _, ty := range []jlm.Type{jlm.TypeF32, jlm.TypeQ8} {
			if ty == jlm.TypeQ8 && sh[0]%32 != 0 {
				continue
			}
			for _, rows := range []int{1, 3} {
				name := fmt.Sprintf("%dx%d %v rows %d", sh[0], sh[1], ty, rows)
				b := graph.New()
				x := b.In("x", sh[0])
				mix := b.Emit(graph.OpMap, []graph.ValueID{b.MatVec(x, jlm.RoleAttnV, sh[1])}, []graph.Result{{Name: "mix", Width: sh[1]}}, nil,
					graph.Attrs{Fn: graph.MapExp}, nil, nil)[0]
				b.Out(b.Add(x, b.MatVec(mix, jlm.RoleAttnOut, sh[0])))
				_, codes, err := jit.Default().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnOut: ty, jlm.RoleAttnV: ty},
					Step:   jit.Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32},
					Device: jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}})
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				f := &partForm{codes: make([]partCode, len(codes))}
				lastRead, firstWrite, shared := -1, len(codes), false
				for ci, c := range codes {
					var args []partArg
					reads, writes := false, false
					for _, bd := range c.Binds {
						a := partArg{kind: partOwn, read: bd.Read, writes: bd.Writes}
						if o := bd.Operand; o.Kind == jit.OperandValue && o.Value == b.G.In[0] {
							a.kind, reads = partIn, reads || bd.Read
						} else if o.Kind == jit.OperandValue && o.Value == b.G.Out[0] {
							a.kind, writes = partOut, writes || bd.Writes
						}
						args = append(args, a)
					}
					if reads {
						lastRead = ci
					}
					if writes {
						firstWrite = min(firstWrite, ci)
					}
					shared = shared || reads && writes
					f.args = append(f.args, args)
				}
				got := f.inPlace(0, 0)
				switch {
				case shared && got:
					t.Fatalf("%s: a kernel reads x and writes the output, and the form is written over x", name)
				case got != (lastRead < firstWrite):
					t.Fatalf("%s: x last read by kernel %d, the output first written by %d, and the form written over x: %v", name, lastRead, firstWrite, got)
				case shared:
					over++
				case got:
					before++
				}
			}
		}
	}
	if over == 0 || before == 0 {
		t.Fatalf("%d forms read x where they write, %d before: the gate sees one side only", over, before)
	}
	t.Logf("%d forms read x in the kernel writing the output and are refused, %d read it before and are written over it", over, before)
}
