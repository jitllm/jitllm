package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestSliceRowsMatchesTheHostSlice runs the field cut on every device against
// the Go slice expression it stands in for.
//
// Byte equality, because it is a copy and any tolerance would admit an
// off-by-one in the offset. Reading q where k belongs is the whole failure mode
// -- the same shape kernels.SplitDeltaGates and SplitHeadGate are written
// against -- and it produces finite, plausible numbers.
//
// The shapes are qwen3next-80B's three fields and a ragged one whose grid is not
// a whole number of workgroups, so the surplus threads clamp onto the last work
// item.
func TestSliceRowsMatchesTheHostSlice(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	for _, c := range []struct{ n, stride, off, rows int }{
		{2048, 8192, 0, 1},    // q
		{2048, 8192, 2048, 1}, // k
		{4096, 8192, 4096, 1}, // v
		{5, 17, 7, 3},         // ragged, several rows
	} {
		src := make([]float32, c.rows*c.stride)
		for i := range src {
			src[i] = float32(i%211) - 105
		}
		want := make([]float32, c.rows*c.n)
		for r := 0; r < c.rows; r++ {
			copy(want[r*c.n:][:c.n], src[r*c.stride+c.off:][:c.n])
		}
		k, err := kernels.SliceRows(c.n, c.stride, c.off, c.rows)
		if err != nil {
			t.Fatalf("SliceRows: %v", err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("validate: %v", err)
		}
		for _, d := range devs {
			kern, err := d.Compile(k)
			if err != nil {
				t.Fatalf("%s: compile: %v", d.API(), err)
			}
			bs, err := d.Alloc(len(src) * 4)
			if err != nil {
				t.Fatalf("alloc: %v", err)
			}
			defer bs.Free()
			bo, err := d.Alloc(len(want) * 4)
			if err != nil {
				t.Fatalf("alloc: %v", err)
			}
			defer bo.Free()
			if err := bs.Write(f32le(src)); err != nil {
				t.Fatalf("write: %v", err)
			}
			if err := kern.Launch((c.rows*c.n+127)/128, 128, bs, bo); err != nil {
				t.Fatalf("launch: %v", err)
			}
			kern.Close()
			raw := make([]byte, len(want)*4)
			if err := bo.Read(raw); err != nil {
				t.Fatalf("read: %v", err)
			}
			for i, w := range want {
				got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
				if got != w {
					t.Fatalf("%s n=%d stride=%d off=%d rows=%d out[%d] = %v, want %v",
						d.API(), c.n, c.stride, c.off, c.rows, i, got, w)
				}
			}
		}
	}
}
