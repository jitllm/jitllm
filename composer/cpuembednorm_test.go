package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUNormsAPackedTablesRows executes, on the CPU, each row's
// token's row of a packed table RMS-normed: the row's gathered loads keep
// its sum of squares a loop of one lane, and the norm's own loop runs in
// lanes, so the scale the sum makes reaches every lane only broadcast.
// Each format's rows are held to its own decoder; the violation reads the
// scale in the first lane alone.
func TestComposedCPUNormsAPackedTablesRows(t *testing.T) {
	const R, K, N, eps = 3, 256, 9, 1e-6
	rng := rand.New(rand.NewSource(11))
	w := make([]float32, K)
	for i := range w {
		w[i] = float32(0.5 + rng.Float64())
	}
	ids := []uint32{4, 7, 1}
	dev := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	ran := 0
	for _, typ := range quant.PackedTypes() {
		q, ok := layout.QuantOf(typ)
		jt, jok := jlm.TypeOf(typ)
		if !ok || !jok || K%q.Elems() != 0 {
			continue
		}
		t.Run(fmt.Sprint(typ), func(t *testing.T) {
			raw := cpuTestWeights(typ, q, N, K, rng)
			qs, d, sc, err := layout.PackWeights(q, raw, N, K)
			if err != nil {
				t.Fatal(err)
			}
			table := make([]float32, N*K)
			if err := quant.Dequant32(typ, raw, table); err != nil {
				t.Fatal(err)
			}
			want := make([]float32, R*K)
			for r, id := range ids {
				row := table[int(id)*K : (int(id)+1)*K]
				var ss float64
				for _, v := range row {
					ss += float64(v) * float64(v)
				}
				inv := 1 / math.Sqrt(ss/K+eps)
				for i, v := range row {
					want[r*K+i] = float32(float64(v)*inv) * w[i]
				}
			}
			b := graph.New()
			e := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "e", Width: K}},
				[]graph.Binding{{Role: jlm.RoleTokenEmbd, K: K, Rows: N}}, graph.Attrs{}, nil, nil)[0]
			n := b.Norm(e, jlm.RoleAttnNorm, eps)
			b.Out(n)
			g := b.G
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{jlm.RoleTokenEmbd: jt, jlm.RoleAttnNorm: jlm.TypeF32},
				Step: Step{Rows: R, Page: 1, Keys: 1, ActWindow: 32}, Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			got, gotE := make([]float32, R*K), make([]float32, R*K)
			bufs := map[string][]uint32{"tokens": append([]uint32(nil), ids...), jlm.RoleTokenEmbd.String(): qs,
				jlm.RoleTokenEmbd.String() + "#1": d, jlm.RoleTokenEmbd.String() + "#2": sc, jlm.RoleAttnNorm.String(): f32Words(w),
				g.Values[e].Name: f32Words(gotE), g.Values[n].Name: f32Words(got)}
			if err := runCPUCalls(hostArm(), codes, g, bufs); err != nil {
				t.Fatal(err)
			}
			// The norm stores over the row it reads where nothing else does.
			stored := false
			for _, c := range codes {
				for _, bd := range c.Binds {
					stored = stored || bd.Writes && bd.Operand.Kind == OperandValue && bd.Operand.Value == n
				}
			}
			if !stored {
				got = gotE
			}
			if e := cpuNMSE(got, want); !(e < 1e-10) {
				t.Fatalf("NMSE %.3e: got %v, want %v", e, got[:8], want[:8])
			}
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no packed type ran")
	}
}
