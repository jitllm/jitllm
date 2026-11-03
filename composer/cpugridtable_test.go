package composer

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
)

// TestComposedCPUGridTableForms forms a tower's grid rotary table
// (graph.RopeTable.Grid) for the CPU and for a card at a picture's rows.
func TestComposedCPUGridTableForms(t *testing.T) {
	r := Default()
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	for _, hd := range []int{64, 72, 80} {
		gr := TableGraph(graph.Rotary{NRot: hd, Grid: true}, 0)
		for _, rows := range []int{1, 16, 1024} {
			for _, d := range []Device{dev, {Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}} {
				if _, _, err := r.Form(Request{Graph: gr, Part: PartTable, Step: Step{Rows: rows, Page: 1, Keys: 1, ActWindow: 32}, Device: d, Apart: true}); err != nil {
					t.Errorf("head %d, %d rows on %v: %v", hd, rows, d.Kind, err)
				}
			}
		}
	}
}
