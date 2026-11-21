package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUEmbedsARowNamingNoToken executes OpEmbed over the step's
// tokens on the CPU with tokens on both sides of 2^31: a row naming no token
// (-1, the all-ones word) and one past the table read row 0, as the bound
// says, and the rest their own rows. A signed bound lets -1 through to the
// row before the table.
func TestComposedCPUEmbedsARowNamingNoToken(t *testing.T) {
	const R, K, N = 6, 13, 9
	b := graph.New()
	rows := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "rows", Width: K}},
		[]graph.Binding{{Role: jlm.RoleTokenEmbd, K: K, Rows: N}}, graph.Attrs{}, nil, nil)[0]
	b.Out(rows)
	g := b.G
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{jlm.RoleTokenEmbd: jlm.TypeF32},
		Step:   Step{Rows: R, Page: 1, Keys: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	table := make([]float32, (N+1)*K)
	for i := range table {
		table[i] = float32(i) + 0.5
	}
	toks := []uint32{3, 0xFFFFFFFF, 0x80000000, 8, N + 2, 0}
	out := make([]float32, R*K)
	// The table sits one row into its buffer, a guard row before it: a
	// gather of row -1 reads the guard's values.
	guarded := append(make([]float32, K), table[:N*K]...)
	for i := range K {
		guarded[i] = -1
	}
	bufs := map[string][]uint32{"tokens": toks, jlm.RoleTokenEmbd.String(): f32Words(guarded[K:]), g.Values[rows].Name: f32Words(out)}
	if err := runCPUCalls(hostArm(), codes, g, bufs); err != nil {
		t.Fatal(err)
	}
	for r, tk := range toks {
		row := int(tk)
		if tk >= N {
			row = 0
		}
		for i := range K {
			if w, v := table[row*K+i], out[r*K+i]; math.Float32bits(w) != math.Float32bits(v) {
				t.Fatalf("row %d (token %#x) element %d: %v, the table's row %d holds %v", r, tk, i, v, row, w)
			}
		}
	}
	t.Logf("%d rows, tokens across 2^31 and past the table, each its row or row 0", R)
}
