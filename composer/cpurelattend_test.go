package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPURelativeAttention holds graph.OpRowsAttend's relative form
// to its definition on the CPU: each live row's scores over the live keys
// are q.k plus its second query's dot with the table's row for the
// position q-j (row k the position top-1-k), scaled and softmaxed, the
// values' weighted sum. Two heads of four, six live rows of eight, a table
// for ten positions either way. The violation -- the position j-q -- fails
// it.
func TestComposedCPURelativeAttention(t *testing.T) {
	const R, H, hd, live, top = 8, 2, 4, 6, 10
	const D, PT, scale = H * hd, 2*top - 1, 0.5
	x := make([]float32, R*D)
	for i := range x {
		x[i] = float32(math.Sin(0.41*float64(i)+0.3) * (1 + float64(i%4)/3))
	}
	pe := make([]float32, PT*D)
	for i := range pe {
		pe[i] = float32(math.Sin(0.17*float64(i) + 1))
	}
	b := graph.NewBlock(-1)
	in := b.In("x", D)
	qv := b.Emit(graph.OpMap, []graph.ValueID{in}, []graph.Result{{Name: "qv", Width: D}}, nil,
		graph.Attrs{Fn: graph.MapMul, Const: 0.5}, nil, nil)[0]
	b.Out(b.Emit(graph.OpRowsAttend, []graph.ValueID{in, in, in, qv}, []graph.Result{{Name: "o", Width: D}},
		[]graph.Binding{{Role: jlm.RoleAPos, K: D, Rows: PT}},
		graph.Attrs{NHead: H, NKVHead: H, HeadDim: hd, Scale: scale}, nil, nil)[0])
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAPos: jlm.TypeF32},
		Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	pos := make([]uint32, R+1)
	pos[0] = live
	for i := range R {
		pos[1+i] = uint32(i)
	}
	// qv is value 1, the harness's output; the attention's, value 2, is read
	// from a buffer of the test's own.
	got := make([]float32, R*D)
	_, err = runCPUBound(hostArm(), codes, x, R*D, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleAPos: pe},
		func(o Operand) []uint32 {
			switch {
			case o.Name == "rope.pos":
				return pos
			case o.Kind == OperandValue && o.Value == 2:
				return f32Words(got)
			case o.Kind == OperandWeight && o.Role == jlm.RoleAPos:
				return f32Words(pe)
			}
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for q := range live {
		for h := range H {
			s := make([]float64, live)
			mx := math.Inf(-1)
			for j := range live {
				for e := range hd {
					el := h*hd + e
					s[j] += float64(x[q*D+el])*float64(x[j*D+el]) + 0.5*float64(x[q*D+el])*float64(pe[(top-1-q+j)*D+el])
				}
				s[j] *= scale
				mx = math.Max(mx, s[j])
			}
			l := 0.0
			for j := range s {
				s[j] = math.Exp(s[j] - mx)
				l += s[j]
			}
			for e := range hd {
				want := 0.0
				for j := range s {
					want += s[j] * float64(x[j*D+h*hd+e])
				}
				want /= l
				d := math.Abs(float64(got[q*D+h*hd+e]) - want)
				if math.IsNaN(d) || d > 1e-4*max(1, math.Abs(want)) {
					t.Fatalf("row %d head %d element %d: %g, want %g", q, h, e, got[q*D+h*hd+e], want)
				}
				worst = max(worst, d)
			}
		}
	}
	t.Logf("%d live rows over a %d-position table, worst %.2e", live, top, worst)
}
