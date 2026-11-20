package composer

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPURowConv holds graph.OpRowConv to its definition on the CPU:
// each channel's taps against the rows Taps/2 before to Taps/2 after, zero
// past the live rows, channel-major taps; a row past the live rows is the
// last live row's. Three and nine taps, four and 512 channels, nine live
// rows of twelve. Taps read past the live rows are the violation.
func TestComposedCPURowConv(t *testing.T) {
	const R, live = 12, 9
	for _, cs := range [][2]int{{3, 4}, {9, 4}, {3, 512}, {9, 512}} {
		T, C := cs[0], cs[1]
		t.Run(fmt.Sprintf("%d taps over %d", T, C), func(t *testing.T) {
			x := make([]float32, R*C)
			for i := range x {
				x[i] = float32(math.Sin(0.37*float64(i)) + 0.2)
			}
			w := make([]float32, C*T)
			for i := range w {
				w[i] = float32(math.Cos(0.11*float64(i)) / 2)
			}
			b := graph.NewBlock(-1)
			in := b.In("x", C)
			b.Out(b.Emit(graph.OpRowConv, []graph.ValueID{in}, []graph.Result{{Name: "conv", Width: C}},
				[]graph.Binding{{Role: jlm.RoleADW, K: T, Rows: C}}, graph.Attrs{Taps: T}, nil, nil)[0])
			_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleADW: jlm.TypeF32},
				Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: cpuDevice(hostArm()), Apart: true})
			if err != nil {
				t.Fatal(err)
			}
			pos := make([]uint32, R+1)
			pos[0] = live
			got, err := runCPUBound(hostArm(), codes, x, R*C, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleADW: w},
				func(o Operand) []uint32 {
					switch {
					case o.Name == "rope.pos":
						return pos
					case o.Kind == OperandWeight && o.Role == jlm.RoleADW:
						return f32Words(w)
					}
					return nil
				})
			if err != nil {
				t.Fatal(err)
			}
			for r := range R {
				rr := min(r, live-1)
				for c := range C {
					want := 0.0
					for k := range T {
						at := rr + k - T/2
						if at >= 0 && at < live {
							want += float64(x[at*C+c]) * float64(w[c*T+k])
						}
					}
					if d := math.Abs(float64(got[r*C+c]) - want); math.IsNaN(d) || d > 1e-5*max(1, math.Abs(want)) {
						t.Fatalf("row %d channel %d: %g, want %g", r, c, got[r*C+c], want)
					}
				}
			}
		})
	}
}

// TestComposedCPURowConvWindows holds a windowed graph.OpRowConv
// (RowWindows) to its definition on the CPU: each row's taps inside its
// window, which its descriptor states in positions, and inside the live
// rows. The call's rows start at position 0 and at 40, as a piece past a
// run's first does (gpu.GPU.phased): the first five rows' window starts
// before the call and ends at its sixth row, the rest's starts there. Taps
// compared against the window as rows of the call, not as positions, are
// the violation the call at 40 sees.
func TestComposedCPURowConvWindows(t *testing.T) {
	const R, live, T, C, split = 12, 9, 3, 4, 5
	for _, first := range []int{0, 40} {
		t.Run(fmt.Sprintf("from %d", first), func(t *testing.T) {
			x := make([]float32, R*C)
			for i := range x {
				x[i] = float32(math.Sin(0.37*float64(i)) + 0.2)
			}
			w := make([]float32, C*T)
			for i := range w {
				w[i] = float32(math.Cos(0.11*float64(i)) / 2)
			}
			b := graph.NewBlock(-1)
			in := b.In("x", C)
			b.Out(b.Emit(graph.OpRowConv, []graph.ValueID{in}, []graph.Result{{Name: "conv", Width: C}},
				[]graph.Binding{{Role: jlm.RoleADW, K: T, Rows: C}}, graph.Attrs{Taps: T, RowWindows: true}, nil, nil)[0])
			_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleADW: jlm.TypeF32},
				Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: cpuDevice(hostArm()), Apart: true})
			if err != nil {
				t.Fatal(err)
			}
			pos := make([]uint32, R+1)
			pos[0] = live
			win := func(r int) (lo, hi int) {
				if r < split {
					return max(first-3, 0), first + split
				}
				return first + split, first + live
			}
			desc := make([]uint32, R*RowWords)
			for r := range R {
				lo, hi := win(r)
				desc[r*RowWords+RowStart], desc[r*RowWords+RowEnd], desc[r*RowWords+RowWrite] = uint32(lo), uint32(hi), uint32(first+r)
			}
			got, err := runCPUBound(hostArm(), codes, x, R*C, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleADW: w},
				func(o Operand) []uint32 {
					switch {
					case o.Name == "rope.pos":
						return pos
					case o.Kind == OperandTable && o.Table == TableWindows:
						return desc
					case o.Kind == OperandWeight && o.Role == jlm.RoleADW:
						return f32Words(w)
					}
					return nil
				})
			if err != nil {
				t.Fatal(err)
			}
			for r := range R {
				rr := min(r, live-1)
				lo, hi := win(rr)
				for c := range C {
					want := 0.0
					for k := range T {
						at := rr + k - T/2
						if at >= 0 && at < live && first+at >= lo && first+at < hi {
							want += float64(x[at*C+c]) * float64(w[c*T+k])
						}
					}
					if d := math.Abs(float64(got[r*C+c]) - want); math.IsNaN(d) || d > 1e-5*max(1, math.Abs(want)) {
						t.Fatalf("row %d channel %d: %g, want %g", r, c, got[r*C+c], want)
					}
				}
			}
		})
	}
}
