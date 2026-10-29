package jit_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
)

// TestPrologueRunsWhereEveryWorkgroupReadsAll forms, on a GPU, a one-row
// matvec and an expert matvec of four pairs over MXFP4: the matvec's
// workgroups each read the whole int8 row, so its quantize runs inside them;
// an expert matvec's workgroup reads its pair's row alone, so its quantize
// runs once, a launch of its own.
func TestPrologueRunsWhereEveryWorkgroupReadsAll(t *testing.T) {
	const P, K, rows, E = 4, 2880, 2880, 32
	dev := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, Shared: 96 << 10,
		ExactDiv: true, ExactArith: true, HalfPairs: true, VectorLoads: true}
	jt, _ := jlm.TypeOf(quant.MXFP4)
	prologues := func(experts bool) int {
		b := graph.New()
		role := jlm.RoleFFNDown
		var y []graph.ValueID
		if experts {
			role = jlm.RoleExpGateBank
			x := b.In("x", P*K)
			sel := b.In("sel", P)
			b.G.Values[sel].Elem = graph.ElemU32
			y = b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: "y", Width: P * rows}},
				[]graph.Binding{{Role: role, K: K, Rows: rows, Sheets: E}}, graph.Attrs{}, nil, nil)
		} else {
			x := b.In("x", K)
			y = b.Emit(graph.OpMatVec, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: rows}},
				[]graph.Binding{{Role: role, K: K, Rows: rows}}, graph.Attrs{}, nil, nil)
		}
		b.Out(y[0])
		_, codes, err := jit.Default().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{role: jt},
			Step: jit.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
		if err != nil {
			t.Fatal(err)
		}
		n := 0
		for _, c := range codes {
			if slices.Contains(c.Passes, "prologue") {
				n++
			}
		}
		return n
	}
	if n := prologues(false); n == 0 {
		t.Fatal("the matvec's quantize does not run in its workgroups")
	}
	if n := prologues(true); n != 0 {
		t.Fatalf("%d expert launches run the quantize of every pair's row", n)
	}
}
