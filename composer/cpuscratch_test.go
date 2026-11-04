package composer

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestCPUPromptScratchMovesWithItsCalls holds a prompt's int8 form of its
// matvec input on the CPU to scratch each call writes at its own offset,
// a row's windows a run-time length the caller passes as its extent, so
// the caller runs the calls on the pool rather than one after another.
func TestCPUPromptScratchMovesWithItsCalls(t *testing.T) {
	jt, _ := jlm.TypeOf(quant.Q4_K)
	req := Request{
		Graph:   graph.MatVec(jlm.RoleOutput, 2048, 2048),
		Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
		Step:    Step{Rows: 128, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceCPU, Slots: 14, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true},
	}
	_, codes, err := Default().Form(req)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, c := range codes {
		for _, b := range c.Binds {
			if !b.Writes || b.Operand.Kind != OperandScratch || len(c.Calls) == 0 {
				continue
			}
			for _, a := range c.Args {
				if a.Operand != b.Operand {
					continue
				}
				checked++
				if len(a.Per) != len(c.Calls) || a.Per[0] <= 0 {
					t.Errorf("%s writes %s at one address for its %v calls (per %v)", c.Name, b.Operand.Name, c.Calls, a.Per)
				}
			}
		}
	}
	if checked < 6 {
		t.Fatalf("%d scratch writes checked: the gate proved nothing", checked)
	}
}
