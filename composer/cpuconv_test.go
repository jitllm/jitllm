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

// gridConvGraph is a depthwise convolution over an h x w grid, of k x k
// taps over c channels at stride s (graph.OpConv with GridW and Stride),
// its output a grid of rows of its own where s is above one.
func gridConvGraph(h, w, c, k, s int) *graph.Graph {
	b := graph.New()
	x := b.In("x", c)
	out := graph.Result{Name: "x.conv", Width: c}
	if s > 1 {
		out.Rows = ((h + s - 1) / s) * ((w + s - 1) / s)
	}
	b.Out(b.Emit(graph.OpConv, []graph.ValueID{x}, []graph.Result{out},
		[]graph.Binding{{Role: jlm.RoleVMNDwMid, K: c, Rows: k * k}}, graph.Attrs{Taps: k, GridW: w, Stride: s}, nil, nil)[0])
	return b.G
}

// gridConvCases is the grids, channels, taps and strides the convolution
// gates run: ragged sides and channel counts, a grid of one row and of one
// column, MobileNet-V5's 3x3 and 5x5 taps, and its stride-2 downsamples
// over even and odd sides.
func gridConvCases() [][5]int {
	return [][5]int{{1, 1, 1, 3, 1}, {1, 7, 5, 3, 1}, {7, 1, 9, 3, 1}, {4, 6, 3, 3, 1}, {5, 5, 17, 5, 1}, {9, 7, 33, 3, 1},
		{8, 8, 64, 5, 1}, {3, 3, 1, 5, 1}, {16, 16, 128, 3, 1},
		{8, 8, 64, 3, 2}, {7, 9, 5, 5, 2}, {5, 5, 17, 3, 2}, {16, 16, 128, 5, 2}, {1, 7, 5, 3, 2}, {7, 1, 9, 3, 2}, {2, 2, 3, 3, 2}}
}

// TestComposedCPUGridConv executes the composer's depthwise convolution over
// a picture's grid on the CPU at every case of gridConvCases, held to the
// oracle's (float64 accumulation) with a guard of poisoned words after the
// result. Two violations must fail it: the filter read transposed (its
// taps' rows and columns swapped) and the grid read one column narrower.
func TestComposedCPUGridConv(t *testing.T) {
	const guard = 64
	convReg := Default()
	rng := rand.New(rand.NewSource(37))
	dev := cpuDevice(hostArm())
	nan := float32(math.NaN())
	right := 0
	for _, cs := range gridConvCases() {
		h, w, c, k, s := cs[0], cs[1], cs[2], cs[3], cs[4]
		name := fmt.Sprintf("%dx%d/%d channels/%dx%d/stride %d", h, w, c, k, k, s)
		form := func(w int) ([]Code, error) {
			_, codes, err := convReg.Form(Request{Graph: gridConvGraph(h*cs[1]/w, w, c, k, s),
				Formats: map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32},
				Step:    Step{Rows: h * cs[1], Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
			return codes, err
		}
		codes, err := form(w)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		x := make([]float32, h*w*c)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		filt := make([]float32, k*k*c)
		for i := range filt {
			filt[i] = float32(rng.NormFloat64())
		}
		want := make([]float32, ((h+s-1)/s)*((w+s-1)/s)*c)
		oracle.DWConv32(want, x, h, w, c, k, s, filt)
		run := func(codes []Code, filt []float32) ([]float32, error) {
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
		got, err := run(codes, filt)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if e := rowsNMSE(got, want); math.IsNaN(e) || !(e < 1e-12) {
			t.Errorf("%s: NMSE %.3e against the oracle", name, e)
			continue
		}
		right++
		if k == 1 || h*w < 4 {
			continue
		}
		tr := make([]float32, len(filt))
		for ky := range k {
			for kx := range k {
				copy(tr[(kx*k+ky)*c:(kx*k+ky+1)*c], filt[(ky*k+kx)*c:(ky*k+kx+1)*c])
			}
		}
		if bad, err := run(codes, tr); err != nil {
			t.Errorf("%s: the violation did not run: %v", name, err)
		} else if e := rowsNMSE(bad, want); !(e > 1e-6) {
			t.Errorf("%s: the filter transposed reads %.3e: the gate cannot see it", name, e)
		}
		if s == 1 && w > 1 && (h*w)%(w-1) == 0 {
			narrow, err := form(w - 1)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			if bad, err := run(narrow, filt); err != nil {
				t.Errorf("%s: the violation did not run: %v", name, err)
			} else if e := rowsNMSE(bad, want); !(e > 1e-6) {
				t.Errorf("%s: the grid read a column narrower reads %.3e: the gate cannot see it", name, e)
			}
		}
	}
	t.Logf("%d of %d grid convolutions held to the oracle", right, len(gridConvCases()))
	if right != len(gridConvCases()) {
		t.Fatalf("%d of %d grid convolutions held to the oracle", right, len(gridConvCases()))
	}
}
