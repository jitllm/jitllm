package probe

import (
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/cost"
)

// TestGPUMeasures probes every device the backends open and holds what it
// measures to the order physics allows: the last cache faster than memory
// and nearer, a launch and a kernel's floor under a millisecond, a float chain slower than the float rate of one thread.
func TestGPUMeasures(t *testing.T) {
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	for _, dev := range devs {
		t.Run(dev.API(), func(t *testing.T) {
			if _, ok := Timer(dev); !ok {
				t.Skipf("%s: %s has no device clock to time kernels on", dev.Name(), dev.API())
			}
			d, err := GPU(dev)
			if err != nil {
				t.Fatal(err)
			}
			if l1 := d.Caches[0]; l1.Bytes > 0 {
				t.Logf("L1 %d KiB a multiprocessor: %.0f GB/s, %.0f ns a read; %.1f loads in flight a thread", l1.Bytes>>10, l1.Rate.Total/1e9, l1.Latency*1e9, d.Overlap)
			}
			c := d.Caches[1]
			t.Logf("%s: %d slots, launch %.2f us, a kernel's floor %.2f us", dev.Name(), d.Slots, d.Launch*1e6, d.Floor*1e6)
			t.Logf("L2 %d KiB: %.0f GB/s, %.0f ns a read", c.Bytes>>10, c.Rate.Total/1e9, c.Latency*1e9)
			t.Logf("memory: %.0f GB/s, %.0f ns a read", d.Bandwidth/1e9, d.Latency*1e9)
			names := [cost.NumClasses]string{"float", "special", "int", "index", "lanes", "group", "mem"}
			for cl, r := range d.Rates {
				t.Logf("%-8s %8.0f G/s all, %6.2f G/s a thread, %6.2f ns before its result is used", names[cl], r.Total/1e9, r.Thread/1e9, d.Delay[cl]*1e9)
			}
			f := d.Rates[0]
			if c.Bytes == 0 || c.Rate.Total <= d.Bandwidth || c.Latency <= 0 || c.Latency >= d.Latency {
				t.Errorf("L2 %d bytes at %.0f GB/s and %.0f ns, memory at %.0f GB/s and %.0f ns", c.Bytes, c.Rate.Total/1e9, c.Latency*1e9, d.Bandwidth/1e9, d.Latency*1e9)
			}
			if d.Launch <= 0 || d.Launch > 1e-3 || d.Floor <= 0 || d.Floor > 1e-3 {
				t.Errorf("a launch %.2f us, a kernel's floor %.2f us", d.Launch*1e6, d.Floor*1e6)
			}
			if f.Thread <= 0 || f.Total <= f.Thread || 1/d.Delay[0] >= f.Thread {
				t.Errorf("float %.2f G/s a thread, %.0f all, a chain %.2f G/s", f.Thread/1e9, f.Total/1e9, 1/d.Delay[0]/1e9)
			}
		})
	}
}
