package composer

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUReadsEachRowsToken executes OpToken on the CPU: each row's
// token, the step's own, as an index, row for row.
func TestComposedCPUReadsEachRowsToken(t *testing.T) {
	const R = 5
	b := graph.New()
	tok := b.Emit(graph.OpToken, nil, []graph.Result{{Name: "tok", Width: 1, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(tok)
	g := b.G
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := Default().Form(Request{Graph: g, Step: Step{Rows: R, Page: 1, Keys: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	toks := []uint32{7, 0, 151643, 3, 1 << 20}
	out := make([]uint32, R)
	if err := runCPUCalls(hostArm(), codes, g, map[string][]uint32{"tokens": toks, g.Values[tok].Name: out}); err != nil {
		t.Fatal(err)
	}
	for r := range R {
		if out[r] != toks[r] {
			t.Fatalf("row %d reads token %d, the step's is %d", r, out[r], toks[r])
		}
	}
}
