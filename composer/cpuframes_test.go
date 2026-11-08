package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUFrames holds graph.OpFrames to its definition on the CPU:
// a clip of 50 samples framed 16 wide every 5 (eleven centred frames), each
// sample less 0.97 of the one before, zero past either end -- the first
// frames reach before the clip and the last past it. Framing without the
// centring offset is the violation.
func TestComposedCPUFrames(t *testing.T) {
	const n, N, hop = 50, 16, 5
	const R, pre = n/hop + 1, 0.97
	clip := make([]float32, n)
	for i := range clip {
		clip[i] = float32(math.Sin(0.3*float64(i)) + 0.1*float64(i%7))
	}
	b := graph.NewBlock(-1)
	b.Out(b.Emit(graph.OpFrames, nil, []graph.Result{{Name: "frames", Width: N}}, nil,
		graph.Attrs{Width: N, Stride: hop, Const: pre}, nil, nil)[0])
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	req := make([]uint32, RequestPicW+1)
	req[RequestPicW] = n
	pos := make([]uint32, R+1)
	pos[0] = R
	// The frames are value 0, which the harness binds as its input: they are
	// read from a buffer of the test's own.
	got := make([]float32, R*N)
	_, err = runCPUBound(hostArm(), codes, nil, R*N, [3][]uint32{}, nil, func(o Operand) []uint32 {
		if o.Kind == OperandValue && o.Value == 0 {
			return f32Words(got)
		}
		switch o.Name {
		case "request":
			return req
		case "rope.pos":
			return pos
		}
		if o.Name == TablePicture.op().Name {
			return f32Words(clip)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for tt := range R {
		for i := range N {
			k := tt*hop + i - N/2
			want := 0.0
			if k >= 0 && k < n {
				want = float64(clip[k])
				if k > 0 {
					want -= pre * float64(clip[k-1])
				}
			}
			if d := math.Abs(float64(got[tt*N+i]) - want); math.IsNaN(d) || d > 1e-6 {
				t.Fatalf("frame %d sample %d: %g, want %g", tt, i, got[tt*N+i], want)
			}
		}
	}
	t.Logf("%d frames of %d from a clip of %d", R, N, n)
}
