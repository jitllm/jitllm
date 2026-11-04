package model

import (
	"testing"

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
	m, err := Open(jlmOf(t, benchModel(t)), WithDevices("cpu:novnni"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.cpuISA()&cpu.ISAVNNI != 0 || m.opt.noCPU {
		t.Fatalf("a model opened on cpu:novnni states ISA %v, CPU left out %v", m.cpuISA(), m.opt.noCPU)
	}
	if _, err := Open(jlmOf(t, benchModel(t)), WithDevices("cpu:avx9")); err == nil {
		t.Fatal("a model opened on cpu:avx9")
	}
}
