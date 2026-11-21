package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// lineConvGraph is a depthwise convolution over a line of r rows (graph.OpConv
// with no GridW), of k centred taps over c channels at stride s.
func lineConvGraph(r, c, k, s int) *graph.Graph {
	b := graph.New()
	x := b.In("x", c)
	out := graph.Result{Name: "x.conv", Width: c}
	if s > 1 {
		out.Rows = (r + s - 1) / s
	}
	b.Out(b.Emit(graph.OpConv, []graph.ValueID{x}, []graph.Result{out},
		[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: c, Rows: k}}, graph.Attrs{Taps: k, Stride: s}, nil, nil)[0])
	return b.G
}

// lineConvCases is the rows, channels, taps and strides the line gates run:
// a line of one row, lines shorter than the taps, ragged channel counts,
// and strides over even and odd lines.
func lineConvCases() [][4]int {
	return [][4]int{{1, 1, 3, 1}, {2, 3, 5, 1}, {7, 5, 3, 1}, {9, 33, 5, 1}, {16, 64, 3, 1}, {31, 17, 7, 1},
		{8, 17, 3, 2}, {7, 9, 5, 2}, {2, 3, 3, 2}, {33, 64, 5, 2}}
}

// TestComposedCPULineConv executes the composer's depthwise convolution over
// a line of rows on the CPU at every case of lineConvCases, held to the
// oracle's (float64 accumulation) with a guard of poisoned words after the
// result. The taps read reversed must fail it.
func TestComposedCPULineConv(t *testing.T) {
	const guard = 64
	reg := Default()
	rng := rand.New(rand.NewSource(41))
	dev := cpuDevice(hostArm())
	nan := float32(math.NaN())
	right := 0
	for _, cs := range lineConvCases() {
		r, c, k, s := cs[0], cs[1], cs[2], cs[3]
		name := fmt.Sprintf("%d rows/%d channels/%d taps/stride %d", r, c, k, s)
		_, codes, err := reg.Form(Request{Graph: lineConvGraph(r, c, k, s),
			Formats: map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32},
			Step:    Step{Rows: r, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		x := make([]float32, r*c)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		filt := make([]float32, k*c)
		for i := range filt {
			filt[i] = float32(rng.NormFloat64())
		}
		want := make([]float32, ((r+s-1)/s)*c)
		oracle.DWConvLine32(want, x, r, c, k, s, filt)
		run := func(filt []float32) ([]float32, error) {
			in := append(append([]float32(nil), x...), make([]float32, guard)...)
			for i := len(x); i < len(in); i++ {
				in[i] = nan
			}
			out, err := runCPUVectors(hostArm(), codes, in, len(want)+guard, [3][]uint32{f32Words(filt)}, nil)
			if err != nil {
				return nil, err
			}
			for i := len(want); i < len(out); i++ {
				if !math.IsNaN(float64(out[i])) {
					return nil, fmt.Errorf("the guard word %d past the result was written", i-len(want))
				}
			}
			return out[:len(want)], nil
		}
		got, err := run(filt)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if e := rowsNMSE(got, want); math.IsNaN(e) || !(e < 1e-12) {
			t.Errorf("%s: NMSE %.3e against the oracle", name, e)
			continue
		}
		right++
		if r < 2 {
			continue
		}
		rev := make([]float32, len(filt))
		for i := range k {
			copy(rev[(k-1-i)*c:(k-i)*c], filt[i*c:(i+1)*c])
		}
		if bad, err := run(rev); err != nil {
			t.Errorf("%s: the violation did not run: %v", name, err)
		} else if e := rowsNMSE(bad, want); !(e > 1e-6) {
			t.Errorf("%s: the taps reversed read %.3e: the gate cannot see it", name, e)
		}
	}
	t.Logf("%d of %d line convolutions held to the oracle", right, len(lineConvCases()))
	if right != len(lineConvCases()) {
		t.Fatalf("%d of %d line convolutions held to the oracle", right, len(lineConvCases()))
	}
}
