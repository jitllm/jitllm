package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUGridConcat runs a concatenation over a grid of rows of its
// own -- a stride-2 depthwise convolution's 9 rows of a 36-row step -- on the
// CPU: the convolution's rows beside a matrix of them, doubled, held to the
// oracle. The concatenation forced to the step's rows (Misrowed) is the
// violation.
func TestComposedCPUGridConcat(t *testing.T) {
	const H, W, C, K, S, E = 6, 6, 4, 3, 2, 6
	rng := rand.New(rand.NewSource(43))
	dev := cpuDevice(hostArm())
	Ho, Wo := (H+S-1)/S, (W+S-1)/S
	b := graph.New()
	x := b.In("x", C)
	conv := b.Emit(graph.OpConv, []graph.ValueID{x}, []graph.Result{{Name: "conv", Width: C, Rows: Ho * Wo}},
		[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: C, Rows: K * K}}, graph.Attrs{Taps: K, GridW: W, Stride: S}, nil, nil)[0]
	up := b.MatVec(conv, jlm.RoleVMNPwExp, E)
	cat := b.Emit(graph.OpConcat, []graph.ValueID{conv, up}, []graph.Result{{Name: "cat", Width: C + E}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(b.Add(cat, cat))
	gr := b.G
	in := RandRow(rng, H*W*C, 1)
	filt := RandRow(rng, K*K*C, 0.5)
	w := RandRow(rng, E*C, 1/math.Sqrt(C))
	cv := make([]float32, Ho*Wo*C)
	oracle.DWConv32(cv, in, H, W, C, K, S, filt)
	want := make([]float32, 0, Ho*Wo*(C+E))
	for r := range Ho * Wo {
		row := append([]float32(nil), cv[r*C:(r+1)*C]...)
		for e := range E {
			s := 0.0
			for c := range C {
				s += float64(w[e*C+c]) * float64(cv[r*C+c])
			}
			row = append(row, float32(s))
		}
		for _, v := range row {
			want = append(want, 2*v)
		}
	}
	roles := map[jlm.Role][]float32{jlm.RoleVMNDwMid: filt, jlm.RoleVMNPwExp: w}
	formats := map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32, jlm.RoleVMNPwExp: jlm.TypeF32}
	step := Step{Rows: H * W, Page: 64, Keys: 1, ActWindow: 32}
	run := func(r *Registry) ([]float32, error) {
		_, codes, err := r.Form(Request{Graph: gr, Formats: formats, Step: step, Device: dev})
		if err != nil {
			return nil, err
		}
		return gridRowsRun(hostArm(), codes, gr, step.Rows, in, roles, [3][]uint32{})
	}
	got, err := run(Default().WithoutInt8(DeviceCPU))
	if err != nil {
		t.Fatal(err)
	}
	e := rowsNMSE(got, want)
	t.Logf("a %dx%d grid's %d rows of a %d-row step, concatenated: NMSE %.3e against the oracle", Ho, Wo, Ho*Wo, step.Rows, e)
	if math.IsNaN(e) || !(e < 1e-10) {
		t.Fatalf("NMSE %.3e against the oracle", e)
	}
	for _, k := range []graph.OpKind{graph.OpConcat} {
		bad, err := run(Misrowed(Default().WithoutInt8(DeviceCPU), k, step.Rows))
		if err != nil {
			t.Logf("violation (%s at the step's rows): refused to form: %v", k, err)
			continue
		}
		be := rowsNMSE(bad, want)
		t.Logf("violation (%s at the step's %d rows in place of the grid's %d): NMSE %.3e", k, step.Rows, Ho*Wo, be)
		if !(be > 1e-6) && !math.IsNaN(be) {
			t.Fatalf("VIOLATION: %s at the step's rows reads the same", k)
		}
	}
}
