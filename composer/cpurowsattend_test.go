package composer

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPURowsAttend runs a learned table's rows (graph.OpTableRows)
// attending over the step's live rows (graph.OpRowsAttend) on the CPU: three
// query rows, two heads of four, over seven live rows of ten, each output
// held to the softmax-weighted sum the formula names. Attending all ten rows
// -- the step's dead rows among them -- is the violation.
func TestComposedCPURowsAttend(t *testing.T) {
	const Q, H, hd, R, live = 3, 2, 4, 10, 7
	const D, scale = H * hd, 0.5
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	query := make([]float32, Q*D)
	for i := range query {
		query[i] = float32(math.Sin(0.7*float64(i) + 0.3))
	}
	x := make([]float32, R*D)
	for i := range x {
		x[i] = float32(math.Cos(0.29*float64(i)) * (1 + float64(i%5)/4))
	}
	b := graph.NewBlock(-1)
	in := b.In("x", D)
	q := b.Emit(graph.OpTableRows, nil, []graph.Result{{Name: "q", Width: D, Rows: Q}},
		[]graph.Binding{{Role: jlm.RoleVRsQuery, K: D, Rows: Q}}, graph.Attrs{}, nil, nil)[0]
	b.Out(b.Emit(graph.OpRowsAttend, []graph.ValueID{q, in, in}, []graph.Result{{Name: "o", Width: D, Rows: Q}}, nil,
		graph.Attrs{NHead: H, HeadDim: hd, Scale: scale}, nil, nil)[0])
	_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVRsQuery: jlm.TypeF32},
		Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	pos := make([]uint32, R+1)
	pos[0] = live
	for i := range R {
		pos[1+i] = uint32(i)
	}
	// The queries are value 1, which the harness binds as its output; the
	// attention's output, value 2, is read from a buffer of the test's own.
	res := make([]float32, Q*D)
	if _, err := runCPUBound(hostArm(), codes, x, Q*D, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleVRsQuery: query}, func(o Operand) []uint32 {
		switch {
		case o.Name == "rope.pos":
			return pos
		case o.Kind == OperandValue && o.Value == 2:
			return unsafe.Slice((*uint32)(unsafe.Pointer(&res[0])), len(res))
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	got := res
	want := func(rows int) []float64 {
		out := make([]float64, Q*D)
		for qi := range Q {
			for h := range H {
				s := make([]float64, rows)
				m := math.Inf(-1)
				for j := range rows {
					for e := range hd {
						s[j] += float64(query[qi*D+h*hd+e]) * float64(x[j*D+h*hd+e])
					}
					s[j] *= scale
					m = math.Max(m, s[j])
				}
				l := 0.0
				for j := range rows {
					s[j] = math.Exp(s[j] - m)
					l += s[j]
				}
				for e := range hd {
					v := 0.0
					for j := range rows {
						v += s[j] * float64(x[j*D+h*hd+e])
					}
					out[qi*D+h*hd+e] = v / l
				}
			}
		}
		return out
	}
	far := func(w []float64) float64 {
		d := 0.0
		for i, v := range got {
			d = math.Max(d, math.Abs(float64(v)-w[i]))
		}
		return d
	}
	if d := far(want(live)); d > 1e-5 {
		t.Fatalf("the rows attention is %.3g from the formula over the %d live rows", d, live)
	}
	all := far(want(R))
	t.Logf("%d query rows over %d live rows of %d: within 1e-5 of the formula; over every row %.3g from it", Q, live, R, all)
	if all < 1e-3 {
		t.Fatalf("VIOLATION: attending all %d rows reads the same (%.3g)", R, all)
	}
}
