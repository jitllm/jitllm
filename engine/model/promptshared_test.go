package model

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestPromptFormsFitTheWorkgroupsMemory forms every block of each served
// state-space checkpoint (ssmServed) for a card at decode, at a chunk's rows
// and at the reserved width (512 rows), with subgroups and without, and holds
// each kernel's workgroup arrays, the group sums' the lowering adds among
// them, to the 48 KiB every card's static workgroup memory holds. Falcon-H1's
// scan at 512 rows ran its step map inside its workgroups (Prologue) with a
// budget that left the group sums out: 49280 bytes, which sm_70's compiler
// refused (its prompts formed again at their own widths) and sm_86 refused
// at launch.
func TestPromptFormsFitTheWorkgroupsMemory(t *testing.T) {
	for _, file := range ssmServed {
		t.Run(file, func(t *testing.T) {
			m := frontModel(t, file)
			f := m.Composing()
			formed, widest := 0, 0
			for li := range m.Cfg.NLayer {
				g := f.BlockGraph(li)
				if g == nil {
					t.Fatalf("block %d has no graph: %s", li, m.Uncovered(li))
				}
				formats := map[jlm.Role]jlm.Type{}
				for _, o := range g.Ops {
					for _, w := range o.Weight {
						formats[w.Role] = jlm.TypeF32
						if e, ok := f.Find(w.Role, int32(li), -1); ok && w.Rows > 0 && w.K > 4 {
							formats[w.Role] = e.Type
						}
					}
				}
				gr := *g
				gr.Block = 0
				for _, rows := range []int{1, 64, 512} {
					for _, lanes := range []int{32, 1} {
						_, codes, err := jit.Default().Form(jit.Request{Graph: &gr, Formats: formats,
							Step:   jit.Step{Rows: rows, Page: 64, Keys: 2048, ActWindow: 32},
							Device: jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: lanes}})
						if err != nil {
							t.Fatalf("block %d at %d rows, %d lanes: %v", li, rows, lanes, err)
						}
						for _, c := range codes {
							n := 0
							for _, op := range c.GPU.Ops {
								if op.Kind == ir.OpShared {
									n += int(op.Imm) * 4
								}
							}
							widest = max(widest, n)
							if n > 48<<10 {
								t.Fatalf("block %d at %d rows, %d lanes: %s declares %d bytes of workgroup memory, past 48 KiB",
									li, rows, lanes, c.Name, n)
							}
						}
						formed++
					}
				}
			}
			t.Logf("%d forms, the widest kernel declaring %d bytes of workgroup memory", formed, widest)
		})
	}
}
