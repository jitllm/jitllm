package tier

import (
	"fmt"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// TestPCIeBandwidth measures the number that decides where weights should live.
//
// Decode is memory-bound, so a weight byte costs whatever its slowest link
// costs. With host-to-device PCIe far slower than CPU reads of DRAM, and those
// far slower than the device reading its own
// VRAM, weights resident in VRAM are fastest, host DRAM next, and weights paged
// across PCIe every token last. A paged VRAM cache breaks even only near an
// 85% hit rate, so it cannot rescue a model many times the card's size;
// spilling whole blocks to the CPU (PrepLayer's decline) is better there, and
// overlapping upload with compute still runs at the upload rate.
func TestPCIeBandwidth(t *testing.T) {
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	for _, d := range devs {
		func() {
			defer d.Close()
			for _, mb := range []int{16, 64, 256} {
				n := mb << 20
				b, err := d.Alloc(n)
				if err != nil {
					t.Logf("  %s: alloc %d MB: %v", d.API(), mb, err)
					return
				}
				host := make([]byte, n)
				d.Session(func(s backend.Session) {
					s.Write(b, host) // warm
				})
				const reps = 8
				start := time.Now()
				d.Session(func(s backend.Session) {
					for i := 0; i < reps; i++ {
						s.Write(b, host)
					}
				})
				el := time.Since(start).Seconds()
				fmt.Printf("  %-6s host->device %4d MB x%d: %.2f GB/s\n",
					d.API(), mb, reps, float64(n)*reps/1e9/el)
				b.Free()
			}
		}()
	}
}
