package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUPositionsPastTheTableReadRowZero executes a position
// table's OpEmbed over the step's positions on the CPU: a position past the
// table reads row 0, and every other its own row. Unbounded, the gather
// reads the guard row after the table. A position is a small word, below
// 2^31 (the unsigned pass's fact about the step's tables), unlike a token.
func TestComposedCPUPositionsPastTheTableReadRowZero(t *testing.T) {
	const R, K, N = 7, 13, 9
	b := graph.New()
	rows := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "rows", Width: K}},
		[]graph.Binding{{Role: jlm.RolePosEmbd, K: K, Rows: N}}, graph.Attrs{Positional: true}, nil, nil)[0]
	b.Out(rows)
	g := b.G
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{jlm.RolePosEmbd: jlm.TypeF32},
		Step:   Step{Rows: R, Page: 1, Keys: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	table := make([]float32, (N+1)*K)
	for i := range table {
		table[i] = float32(i) + 0.5
	}
	for i := range K {
		table[N*K+i] = -2
	}
	pos := []uint32{3, 1 << 20, N + 5, 8, N, N + 2, 0}
	out := make([]float32, R*K)
	bufs := map[string][]uint32{"rope.pos": append([]uint32{R}, pos...), jlm.RolePosEmbd.String(): f32Words(table),
		g.Values[rows].Name: f32Words(out)}
	if err := runCPUCalls(hostArm(), codes, g, bufs); err != nil {
		t.Fatal(err)
	}
	for r, p := range pos {
		row := int(p)
		if p >= N {
			row = 0
		}
		for i := range K {
			if w, v := table[row*K+i], out[r*K+i]; math.Float32bits(w) != math.Float32bits(v) {
				t.Fatalf("row %d (position %#x) element %d: %v, the table's row %d holds %v", r, p, i, v, row, w)
			}
		}
	}
	t.Logf("%d rows, positions past the table, each its row or row 0", R)
}
