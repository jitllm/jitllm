package composer

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestALongReductionIsNotWrittenOutInItsMap holds the unrolled fusion to short
// reductions whatever reads them: a projection over 2304 columns and a map
// after it (a head and its softcap's scale) once wrote every trip of the row's
// dot into the map, a 55341-op kernel of 1.67 MB of SPIR-V that a V100's
// Vulkan driver refused to build. The forms hold their loop, with and without
// the int8 form, and the map still runs.
func TestALongReductionIsNotWrittenOutInItsMap(t *testing.T) {
	const k, rows = 2304, 256000
	for _, int8 := range []bool{true, false} {
		for _, typ := range []jlm.Type{jlm.TypeQ6S, jlm.TypeQ8, jlm.TypeF32} {
			b := graph.New()
			y := b.MatVec(b.In("x", k), jlm.RoleOutput, rows)
			b.Out(b.Emit(graph.OpMap, []graph.ValueID{y}, []graph.Result{{Name: "capped", Width: rows}}, nil,
				graph.Attrs{Fn: graph.MapMul, Const: 1.0 / 30}, nil, nil)[0])
			r := Default()
			if !int8 {
				r = r.WithoutInt8(DeviceGPU)
			}
			_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: typ},
				Step:   Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
				Device: Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80}})
			if err != nil {
				t.Fatalf("int8 %v, %v: %v", int8, typ, err)
			}
			largest := 0
			for _, c := range codes {
				if c.GPU == nil {
					t.Fatalf("int8 %v, %v: %s has no GPU code", int8, typ, c.Name)
				}
				largest = max(largest, len(c.GPU.Ops))
			}
			if largest > 2048 {
				t.Errorf("int8 %v, %v: a kernel of %d ops: the row's dot is written out, not looped", int8, typ, largest)
			}
		}
	}
}
