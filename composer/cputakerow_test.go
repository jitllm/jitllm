package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUTakeRow moves the row the request names out of a step of
// rows into a row of its own (graph.TakeRow, a folded head's take) on the
// CPU, at steps of one row to 512, the rows held to the input's bit for bit;
// a request past the step's rows takes its last, as the form clamps. The
// same graph forms for a card at every step: a one-row output is a grid of
// rows of its own, which a step of 512 must not refuse.
func TestComposedCPUTakeRow(t *testing.T) {
	const w = 37
	gr := (&graph.TakeRow{Width: w}).Build()
	r := Default()
	cpuDev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	for _, rows := range []int{1, 2, 5, 64, 512} {
		if _, _, err := r.Form(Request{Graph: gr, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32},
			Device: Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}}); err != nil {
			t.Fatalf("%d rows on a card: %v", rows, err)
		}
		_, codes, err := r.Form(Request{Graph: gr, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: cpuDev})
		if err != nil {
			t.Fatalf("%d rows: %v", rows, err)
		}
		x := make([]float32, rows*w)
		for i := range x {
			x[i] = float32(i%9973) + 0.25
		}
		take := func(at int) []float32 {
			req := make([]uint32, RequestRows+1)
			req[RequestRows] = uint32(at)
			got, err := runCPUBound(codes, x, w, [3][]uint32{}, nil, func(o Operand) []uint32 {
				if o.Name == "request" {
					return req
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%d rows, row %d: %v", rows, at, err)
			}
			return got
		}
		for _, at := range []int{0, rows / 2, rows - 1, rows + 3} {
			src := min(at, rows-1)
			got := take(at)
			for j := range w {
				if math.Float32bits(got[j]) != math.Float32bits(x[src*w+j]) {
					t.Fatalf("%d rows, row %d: element %d is %v, want row %d's %v", rows, at, j, got[j], src, x[src*w+j])
				}
			}
		}
	}
}
