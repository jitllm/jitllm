package composer

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUReachStatesEachCallsRows: a decode attention's kernels over
// heads reach their own heads alone; a matvec's calls read the int8 words
// whole, so none states a reach of them.
func TestComposedCPUReachStatesEachCallsRows(t *testing.T) {
	_, codes, err := Default().Form(Request{Graph: attendGraph(32, 8, 64),
		Step:   Step{Rows: 1, Page: 64, Keys: 128, ActWindow: 32, Seqs: 1},
		Device: Device{Kind: DeviceCPU, Slots: 14, Lanes: 8, ISA: cpuDevice().ISA, ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	heads := 0
	for _, c := range codes {
		if len(c.Calls) != 2 {
			continue
		}
		per := 32 / c.Calls[1]
		found := false
		for _, r := range c.Reach {
			if strings.HasSuffix(r.Operand.Name, ".dots") {
				found = true
				if len(r.Per) != 2 || r.Per[1] != per {
					t.Errorf("%s calls %v: reaches %v of %s, want %d heads a call", c.Name, c.Calls, r.Per, r.Operand.Name, per)
				}
			}
		}
		if !found {
			t.Errorf("%s calls %v states no reach of the scores: %+v", c.Name, c.Calls, c.Reach)
		}
		heads++
	}
	if heads < 3 {
		t.Fatalf("%d kernels over heads, want the dots, the softmax and the values", heads)
	}
	jt, _ := jlm.TypeOf(quant.Q4_K)
	_, codes, err = Default().Form(Request{Graph: graph.MatVec(jlm.RoleOutput, 2048, 2048),
		Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
		Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceCPU, Slots: 14, Lanes: 8, ISA: cpuDevice().ISA, ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	mv := 0
	for _, c := range codes {
		if len(c.Calls) == 0 || c.Calls[len(c.Calls)-1] == 1 {
			continue
		}
		mv++
		for _, r := range c.Reach {
			if strings.HasSuffix(r.Operand.Name, ".q8a") {
				t.Errorf("%s calls %v states a reach %v of the words every call reads whole", c.Name, c.Calls, r.Per)
			}
		}
	}
	if mv == 0 {
		t.Fatal("no matvec kernel over calls")
	}
}
