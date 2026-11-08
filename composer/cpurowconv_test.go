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
