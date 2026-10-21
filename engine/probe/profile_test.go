package probe

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestProfileTravels holds a device profile to arriving whole: decoded, it
// is the device that was measured, and an estimate made from it on another
// node is the estimate made where it was measured, to the bit. A profile
// of another version is refused.
func TestProfileTravels(t *testing.T) {
	d := jit.Device{Kind: jit.DeviceGPU, Slots: 163840, Lanes: 32, Units: 80,
		Bandwidth: 869e9, Launch: 2.9e-6, Latency: 2.82e-7, Floor: 7.5e-6, Overlap: 2.2}
	d.Caches[0] = jit.Cache{Bytes: 96 << 10, Rate: jit.Rate{Total: 5.5e12}, Latency: 2.6e-8, Kernel: true}
	d.Caches[1] = jit.Cache{Bytes: 6 << 20, Rate: jit.Rate{Total: 3.1e12}, Latency: 1.46e-7, Retain: 0.31, Cycle: 0.05}
	for cl := range d.Rates {
		d.Rates[cl] = jit.Rate{Thread: 0.7e9 / float64(cl+1), Total: 7.8e12 / float64(cl+1)}
		d.Delay[cl] = 2.6e-9 * float64(cl+1)
	}
	b, err := Profile{Name: "Tesla V100-SXM2-16GB", API: "ptx", Device: d}.Encode()
	if err != nil {
		t.Fatal(err)
	}
	p, err := DecodeProfile(b)
	if err != nil {
		t.Fatal(err)
	}
	if p.Device != d || p.Name != "Tesla V100-SXM2-16GB" || p.API != "ptx" {
		t.Fatalf("decoded %+v, measured %+v", p.Device, d)
	}
	f, err := jlm.Open(testmodels.Path("synth-ernie45.jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s := jit.Step{Rows: 1, Page: 64, Keys: 9}
	here, err := jit.Default().Estimate(f, s, func(jit.Part) jit.Device { return d })
	if err != nil {
		t.Fatal(err)
	}
	there, err := jit.Default().Estimate(f, s, func(jit.Part) jit.Device { return p.Device })
	if err != nil {
		t.Fatal(err)
	}
	if here != there || here.Seconds <= 0 {
		t.Fatalf("measured there %v, priced here %v", here, there)
	}
	if _, err := DecodeProfile([]byte(strings.Replace(string(b), `"version":1`, `"version":2`, 1))); err == nil {
		t.Fatal("a profile of another version was read")
	}
}
