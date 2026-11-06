package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestDevicesListTheCPU holds WithDevices' specs to what they state about the
// CPU device: listed for auto (the default) and for a cpu entry, not for cards
// alone, its variants its Target's, and a bad spec refused at Open.
func TestDevicesListTheCPU(t *testing.T) {
	for _, c := range []struct {
		specs  []string
		listed bool
		opts   int
	}{
		{nil, true, 0},
		{[]string{"auto"}, true, 0},
		{[]string{"cpu"}, true, 0},
		{[]string{"cuda:0"}, false, 0},
		{[]string{"cuda:0", "vulkan:1"}, false, 0},
		{[]string{"cuda:0,cpu:sse"}, true, 1},
		{[]string{"auto", "cpu:perfmap"}, true, 1},
		{[]string{"cpu:novnni+nodotprod"}, true, 2},
	} {
		opts, listed, err := cpuOfDevices(c.specs)
		if err != nil {
			t.Fatalf("%q: %v", c.specs, err)
		}
		if listed != c.listed || len(opts) != c.opts {
			t.Errorf("%q: listed %v with %d options, want %v with %d", c.specs, listed, len(opts), c.listed, c.opts)
		}
	}
	for _, bad := range [][]string{{"cpu:avx9"}, {"cuda:x"}, {"auto", "cuda:0"}} {
		if _, _, err := cpuOfDevices(bad); err == nil {
			t.Errorf("%q was accepted", bad)
		}
	}
	m, err := model.Open(jlmOf(t, benchModel(t)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	in := testInstance(t, m, 64, WithDevices("cpu:novnni"))
	defer in.Close()
	if execOf(in).ISA()&cpu.ISAVNNI != 0 || in.opt.noCPU {
		t.Fatalf("an instance on cpu:novnni states ISA %v, CPU left out %v", execOf(in).ISA(), in.opt.noCPU)
	}
	if bad, err := NewInstance(m, 64, WithDevices("cpu:avx9")); err == nil {
		bad.Close()
		t.Fatal("an instance opened on cpu:avx9")
	}
}
