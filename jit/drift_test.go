package jit

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
)

// TestDriftForms forms the shadow arm's comparison for a card, with the int8
// form and without it: every kernel has its GPU code, the three inputs are
// read and the state's output written, and nothing is bound that a block's
// tier would have to hold for it.
//
// The tier binds the state's input and output to one buffer, so the state is
// updated in place: no kernel that reads the state may launch at or after the
// first that writes the new one (partForm.inPlace). A kernel doing both would
// read a word a surplus thread, clamped to the last work item, had already
// written.
func TestDriftForms(t *testing.T) {
	for _, arm := range []struct {
		name string
		r    *Registry
	}{{"int8", Default()}, {"float", Default().WithoutInt8(DeviceGPU)}} {
		t.Run(arm.name, func(t *testing.T) {
			g := (&graph.Drift{NEmbd: 2048}).Build()
			_, codes, err := arm.r.Form(Request{Graph: g, Step: Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
				Device: Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true}})
			if err != nil {
				t.Fatal(err)
			}
			if len(codes) == 0 {
				t.Fatal("the comparison formed no kernel")
			}
			state, next := g.In[3], g.Out[0]
			bound, lastRead, firstWrite := map[graph.ValueID]bool{}, -1, len(codes)
			for i, c := range codes {
				if c.GPU == nil {
					t.Errorf("%s has no GPU code", c.Name)
				}
				var reads, writes []graph.ValueID
				for _, b := range c.Binds {
					switch b.Operand.Kind {
					case OperandValue:
						bound[b.Operand.Value] = true
						if b.Writes {
							writes = append(writes, b.Operand.Value)
						} else {
							reads = append(reads, b.Operand.Value)
						}
						if b.Operand.Value == state && !b.Writes {
							lastRead = i
						}
						if b.Operand.Value == next && b.Writes {
							firstWrite = min(firstWrite, i)
						}
					case OperandScratch:
					default:
						if len(b.Words) == 0 {
							t.Errorf("%s binds %+v, which only a block's tier holds", c.Name, b.Operand)
						}
					}
				}
				t.Logf("kernel %d %s reads %v writes %v", i, c.Name, reads, writes)
			}
			for i, v := range g.In[:3] {
				if !bound[v] {
					t.Errorf("no kernel reads input %d (%s)", i, g.Values[v].Name)
				}
			}
			if firstWrite == len(codes) {
				t.Errorf("no kernel writes the new state")
			}
			if lastRead < 0 {
				t.Errorf("no kernel reads the state")
			}
			if lastRead >= firstWrite {
				t.Errorf("kernel %d reads the state and kernel %d writes the new one: the tier's update in place would read what it wrote",
					lastRead, firstWrite)
			}
			t.Logf("%d kernels; the state is last read by kernel %d and first written by kernel %d", len(codes), lastRead, firstWrite)
		})
	}
}
