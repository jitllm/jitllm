package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUSampleMirostat holds graph.SampleMirostat, both versions, on
// the CPU device to its definition (internal/oracle): every decisive draw is
// the definition's token and its next mu the definition's, and the
// truncation moves the token against the row's plain draw on a share of
// them, so the gate sees the cut.
func TestComposedCPUSampleMirostat(t *testing.T) {
	const n = 300
	for _, v1 := range []bool{false, true} {
		name := "v2"
		if v1 {
			name = "v1"
		}
		t.Run(name, func(t *testing.T) {
			g := (&graph.SampleMirostat{NVocab: n, V1: v1}).Build()
			if err := graph.Verify(g, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			in := map[string]graph.ValueID{}
			for _, v := range g.In {
				in[g.Values[v].Name] = v
			}
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(hostArm())})
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]uint32, n)
			for i := range ids {
				ids[i] = uint32(i)
			}
			f := func(x float64) uint32 { return math.Float32bits(float32(x)) }
			rng := rand.New(rand.NewSource(71))
			decisive, cases, moved := 0, 0, 0
			for c := range 300 {
				l := make([]float32, n)
				for i := range l {
					l[i] = float32(rng.NormFloat64() * 2.5)
				}
				invT := float32(1 / (0.5 + rng.Float64()))
				tau := float64(float32(2 + 6*rng.Float64()))
				eta := float64(float32(0.05 + 0.25*rng.Float64()))
				mu := float64(float32(tau * (1 + 2*rng.Float64())))
				u := float64(float32(rng.Float64()))
				lw := make([]uint32, n)
				for i, x := range l {
					lw[i] = math.Float32bits(x)
				}
				vals := map[graph.ValueID][]uint32{in["logits"]: lw, in["sample.window"]: make([]uint32, n),
					in["sample.penalty"]: {f(1), 0, 0}, in["sample.ids"]: append([]uint32(nil), ids...),
					in["sample.miro"]: {math.Float32bits(invT), f(tau), f(eta), f(mu), f(u)}, g.Out[0]: {0}, g.Out[1]: {0}}
				if err := runCPUValues(hostArm(), codes, g, vals, [3][]uint32{}); err != nil {
					t.Fatal(err)
				}
				want, mu2, margin := oracle.SampleMirostat(l, v1, invT, tau, eta, mu, u)
				plain, _ := oracle.SampleRow(l, nil, 1, invT, u)
				cases++
				if margin < 1e-4 {
					continue
				}
				decisive++
				if want != plain {
					moved++
				}
				if got := vals[g.Out[0]][0]; got != want {
					t.Fatalf("case %d (1/T %.3f, tau %.3f, mu %.3f, u %.6f): drew %d, want %d (margin %.3g)", c, invT, tau, mu, u, got, want, margin)
				}
				if got := float64(math.Float32frombits(vals[g.Out[1]][0])); math.Abs(got-mu2) > 1e-4*(1+math.Abs(mu2)) {
					t.Fatalf("case %d: next mu %.6f, want %.6f", c, got, mu2)
				}
			}
			if decisive < cases*7/10 || moved < decisive/10 {
				t.Fatalf("%d of %d draws decisive, %d moved by the truncation: the comparison sees too little", decisive, cases, moved)
			}
			t.Logf("%d of %d draws decisive and right, %d moved by the truncation", decisive, cases, moved)
		})
	}
}
