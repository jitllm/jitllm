package main

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The -devices flag, at the CLI's own level: what it accepts, what it refuses,
// and what it prints about the budgets it handed out.

func TestOpenDevicesCPUOpensNothing(t *testing.T) {
	for _, spec := range []string{"cpu", " cpu ", "CPU"} {
		d, closeDev, name, err := openDevices(spec, 0, 8<<30, 0, 0)
		if err != nil {
			t.Fatalf("-devices %q: %v", spec, err)
		}
		closeDev()
		if d != nil || name != "cpu" {
			t.Fatalf("-devices %q gave device %v named %q, want no device at all", spec, d, name)
		}
	}
}

// TestOpenDevicesRefusalsNameBothSides: every rejection says what it got and
// what it wanted, because -devices has nine forms and "invalid" does not tell
// a user which of them to try.
func TestOpenDevicesRefusalsNameBothSides(t *testing.T) {
	for _, tc := range []struct{ spec, got, want string }{
		{"nvidia", "nvidia", "cuda"},
		{"cuda:0,", "entry 2", "empty"},
		{"vulkan:1=nonsense", "nonsense", "byte count"},
		// `cpu` is allowed anywhere -- the host is always a member -- so the
		// contradiction to catch is two device answers, not cpu beside one.
		{"gpu,cuda:0", "gpu", "alone"},
	} {
		_, _, _, err := openDevices(tc.spec, 0, 8<<30, 0, 0)
		if err == nil {
			t.Errorf("-devices %q was accepted", tc.spec)
			continue
		}
		if !strings.Contains(err.Error(), tc.got) || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("-devices %q said %q, which does not name both what it got (%s) and "+
				"what it wanted (%s)", tc.spec, err, tc.got, tc.want)
		}
	}
}

// TestOpenDevicesNamesTwoDevicesAndPoolsThem names two devices and asserts
// which pool each budget comes out of, not only the count: an integrated
// device's heap is the host's own RAM.
func TestOpenDevicesNamesTwoDevicesAndPoolsThem(t *testing.T) {
	sel := -1
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Skipf("no Vulkan here (%v): this gate needs a second backend", err)
	}
	for _, in := range infos {
		if in.Unified && in.Compute && !in.Software {
			sel = in.Index
		}
	}
	if n, err := backend.CUDACount(); err != nil || n == 0 || sel < 0 {
		t.Skipf("this box has %d CUDA device(s) (%v) and integrated Vulkan index %d; "+
			"CPU + iGPU + discrete needs one of each", n, err, sel)
	}
	t.Setenv("JITLLM_GPU_TUNE", "0")
	spec := "cpu,cuda:0=200M,vulkan:" + itoa(sel) + "=200M"
	d, closeDev, name, err := openDevices(spec, 0, 8<<30, 0, 0)
	if err != nil {
		t.Fatalf("-devices %q: %v", spec, err)
	}
	defer closeDev()
	g := d.(*tier.GPU)
	t.Logf("-devices %q opened %q", spec, name)
	t.Log(strings.TrimLeft(deviceBudgets(d), "\n"))
	if g.Devices() != 2 {
		t.Fatalf("%q opened %d device(s), want 2 (the host is the third member and is not "+
			"a device)", spec, g.Devices())
	}
	bs := g.Budgets()
	host := 0
	for _, b := range bs {
		if b.Limit != 200<<20 {
			t.Errorf("%s got %d bytes; the spec said 200M", b.Device, b.Limit)
		}
		if b.Why != "the spec" {
			t.Errorf("%s says its budget came from %q, want the spec that set it", b.Device, b.Why)
		}
		if b.Host {
			host++
		}
	}
	if host != 1 {
		t.Fatalf("%d of %d devices draw from the host's memory; exactly one here does "+
			"(the integrated GPU), and a report that cannot say which is a report that "+
			"lets the host spend the same bytes again", host, len(bs))
	}
	// The host loses what the integrated GPU holds -- nothing is placed here --
	// and never the 200 MiB it may grow into.
	var held uint64
	for _, b := range bs {
		if b.Host {
			held += b.Used
		}
	}
	if r := hostReserved(d); r != held {
		t.Fatalf("the host reserved %d bytes for a device holding %d of a 200 MiB budget on "+
			"its memory", r, held)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestRelocateNeedsTheHost: -relocate moves blocks to the CPU, so it is taken
// only where -devices lets the host run them.
func TestRelocateNeedsTheHost(t *testing.T) {
	for _, tc := range []struct {
		spec string
		ok   bool
	}{
		{"auto", true},
		{"cpu", true},
		{"gpu,cpu", true},
		{"cpu,cuda:0=3G", true},
		{"gpu", false},
		{"cuda:0", false},
		{"all", false},
	} {
		err := relocatable(tc.spec)
		if (err == nil) != tc.ok {
			t.Errorf("-relocate with -devices %q: err %v, want ok=%v", tc.spec, err, tc.ok)
		}
		if err != nil && !strings.Contains(err.Error(), "cpu") {
			t.Errorf("-devices %q: %q does not say how to admit the host", tc.spec, err)
		}
	}
}
