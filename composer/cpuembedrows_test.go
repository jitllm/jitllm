package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUEmbedsIndexRows executes OpEmbed over an index row on the
// CPU: each row's H indices' rows of an f32 table side by side, the table's
// width ragged against the vector, an index past the table reading row 0.
// The same table read through the transposed indices must not agree: the
// gate sees which index fills which slot.
func TestComposedCPUEmbedsIndexRows(t *testing.T) {
	const R, H, K, N = 3, 4, 13, 9
	b := graph.New()
	idx := b.In("idx", H)
	b.G.Values[idx].Elem = graph.ElemU32
	rows := b.Emit(graph.OpEmbed, []graph.ValueID{idx}, []graph.Result{{Name: "rows", Width: H * K}},
		[]graph.Binding{{Role: jlm.RoleTokenEmbd, K: K, Rows: N}}, graph.Attrs{}, nil, nil)[0]
	b.Out(rows)
	g := b.G
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{jlm.RoleTokenEmbd: jlm.TypeF32},
		Step:   Step{Rows: R, Page: 1, Keys: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	table := make([]float32, N*K)
	for i := range table {
		table[i] = float32(rng.NormFloat64())
	}
	ids := make([]uint32, R*H)
	for i := range ids {
		ids[i] = uint32(rng.Intn(N))
	}
	ids[R*H-1] = N + 5
	run := func(ix []uint32) []float32 {
		out := make([]float32, R*H*K)
		bufs := map[string][]uint32{"idx": append([]uint32(nil), ix...), jlm.RoleTokenEmbd.String(): f32Words(table),
			g.Values[rows].Name: f32Words(out)}
		if err := runCPUCalls(codes, g, bufs); err != nil {
			t.Fatal(err)
		}
		return out
	}
	got := run(ids)
	for r := range R {
		for h := range H {
			row := int(ids[r*H+h])
			if row >= N {
				row = 0
			}
			for i := range K {
				if w, v := table[row*K+i], got[(r*H+h)*K+i]; math.Float32bits(w) != math.Float32bits(v) {
					t.Fatalf("row %d index %d (%d) element %d: %v, the table holds %v", r, h, ids[r*H+h], i, v, w)
				}
			}
		}
	}
	swapped := append([]uint32(nil), ids...)
	swapped[0], swapped[1] = swapped[1], swapped[0]
	if ids[0] != ids[1] {
		other := run(swapped)
		same := true
		for i := range other {
			same = same && other[i] == got[i]
		}
		if same {
			t.Fatal("two indices swapped read the same rows: the gate cannot see which index fills which slot")
		}
	}
}
