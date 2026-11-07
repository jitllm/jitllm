package composer

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUUpsample runs a 3x4 grid of rows five wide upsampled two
// times each way (graph.OpUpsample) on the CPU: every output row its nearest
// input row's, exactly. The grid read transposed is the violation.
func TestComposedCPUUpsample(t *testing.T) {
	const H, W, C, S = 3, 4, 5, 2
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	x := make([]float32, H*W*C)
	for i := range x {
		x[i] = float32(math.Sin(0.37*float64(i) + 0.1))
	}
	b := graph.NewBlock(-1)
	in := b.In("x", C)
	b.Out(b.Emit(graph.OpUpsample, []graph.ValueID{in}, []graph.Result{{Name: "up", Width: C, Rows: H * W * S * S}}, nil,
		graph.Attrs{Span: S, GridW: W}, nil, nil)[0])
	_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: H * W, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	got, err := runCPUBound(hostArm(), codes, x, H*W*S*S*C, [3][]uint32{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := func(transposed bool) []float32 {
		out := make([]float32, 0, H*W*S*S*C)
		for y := range H * S {
			for c := range W * S {
				p := (y/S)*W + c/S
				if transposed {
					p = (c/S)%H*W + (y/S)%W
				}
				out = append(out, x[p*C:(p+1)*C]...)
			}
		}
		return out
	}
	off := func(w []float32) int {
		n := 0
		for i := range w {
			if got[i] != w[i] {
				n++
			}
		}
		return n
	}
	if n := off(want(false)); n != 0 {
		t.Fatalf("%d of %d values are not their nearest input row's", n, len(got))
	}
	v := off(want(true))
	t.Logf("a %dx%d grid upsampled %d times: every row its nearest input's; read transposed, %d values differ", H, W, S, v)
	if v == 0 {
		t.Fatal("VIOLATION: the transposed grid reads the same")
	}
}

// TestComposedCPURowsAttendGrouped runs a learned table's rows attending over
// the step's live rows with four query heads over two key/value heads
// (graph.OpRowsAttend, NKVHead): each output held to the softmax-weighted sum
// the formula names, query heads 0 and 1 reading key/value head 0 and 2 and
// 3 head 1. The heads grouped the other way -- head h reading h%2 -- is the
// violation.
func TestComposedCPURowsAttendGrouped(t *testing.T) {
	const Q, H, G, hd, R, live = 3, 4, 2, 4, 10, 7
	const D, KV, scale = H * hd, G * hd, 0.5
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	query := make([]float32, Q*D)
	for i := range query {
		query[i] = float32(math.Sin(0.7*float64(i) + 0.3))
	}
	x := make([]float32, R*KV)
	for i := range x {
		x[i] = float32(math.Cos(0.29*float64(i)) * (1 + float64(i%5)/4))
	}
	b := graph.NewBlock(-1)
	in := b.In("x", KV)
	q := b.Emit(graph.OpTableRows, nil, []graph.Result{{Name: "q", Width: D, Rows: Q}},
		[]graph.Binding{{Role: jlm.RoleVRsQuery, K: D, Rows: Q}}, graph.Attrs{}, nil, nil)[0]
	b.Out(b.Emit(graph.OpRowsAttend, []graph.ValueID{q, in, in}, []graph.Result{{Name: "o", Width: D, Rows: Q}}, nil,
		graph.Attrs{NHead: H, NKVHead: G, HeadDim: hd, Scale: scale}, nil, nil)[0])
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
	want := func(kvOf func(h int) int) []float64 {
		out := make([]float64, Q*D)
		for qi := range Q {
			for h := range H {
				g := kvOf(h)
				s := make([]float64, live)
				m := math.Inf(-1)
				for j := range live {
					for e := range hd {
						s[j] += float64(query[qi*D+h*hd+e]) * float64(x[j*KV+g*hd+e])
					}
					s[j] *= scale
					m = math.Max(m, s[j])
				}
				l := 0.0
				for j := range live {
					s[j] = math.Exp(s[j] - m)
					l += s[j]
				}
				for e := range hd {
					v := 0.0
					for j := range live {
						v += s[j] * float64(x[j*KV+g*hd+e])
					}
					out[qi*D+h*hd+e] = v / l
				}
			}
		}
		return out
	}
	far := func(w []float64) float64 {
		d := 0.0
		for i, v := range res {
			d = math.Max(d, math.Abs(float64(v)-w[i]))
		}
		return d
	}
	if d := far(want(func(h int) int { return h / (H / G) })); d > 1e-5 {
		t.Fatalf("the grouped rows attention is %.3g from the formula", d)
	}
	other := far(want(func(h int) int { return h % G }))
	t.Logf("%d query heads over %d key/value heads: within 1e-5 of the formula; grouped the other way %.3g from it", H, G, other)
	if other < 1e-3 {
		t.Fatalf("VIOLATION: the heads grouped the other way read the same (%.3g)", other)
	}
}
