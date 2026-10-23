package backend

import (
	"github.com/jitllm/jitllm/internal/testdevices"
	"os/exec"
	"regexp"
	"strconv"
	"testing"
)

// TestVulkanMaxBufferIsTheStorageRange holds MaxBuffer to what vulkaninfo
// reports as maxStorageBufferRange for the same device. The limits are carried
// as raw words and the field is read by offset, so a wrong offset reads a
// neighbour -- the uniform-buffer range is 64 KiB on most cards -- and a KV
// pool sized from it would be refused, or worse, bound past the real limit.
// Without vulkaninfo it holds the spec's floor (2^27).
func TestVulkanMaxBufferIsTheStorageRange(t *testing.T) {
	infos, err := VulkanDevices()
	if err != nil || len(infos) == 0 {
		t.Skipf("no Vulkan device: %v -- this gate proved nothing", err)
	}
	ref := map[string]uint64{}
	if out, err := exec.Command("vulkaninfo").Output(); err == nil {
		re := regexp.MustCompile(`deviceName\s+=\s+(.+)|maxStorageBufferRange\s+=\s+(\d+)`)
		name := ""
		for _, m := range re.FindAllStringSubmatch(string(out), -1) {
			if m[1] != "" {
				name = m[1]
			} else if name != "" {
				if v, err := strconv.ParseUint(m[2], 10, 64); err == nil {
					if _, seen := ref[name]; !seen {
						ref[name] = v
					}
				}
			}
		}
	}
	checked := 0
	for _, in := range infos {
		if !testdevices.Has(t, "vulkan", in.Index) {
			continue
		}
		d, err := OpenVulkan(strconv.Itoa(in.Index))
		if err != nil {
			t.Logf("vulkan:%d would not open: %v", in.Index, err)
			continue
		}
		got := d.(BufferLimited).MaxBuffer()
		want, ok := ref[d.Name()]
		d.Close()
		switch {
		case ok && got != want:
			t.Errorf("%s: MaxBuffer %d, vulkaninfo's maxStorageBufferRange %d", d.Name(), got, want)
		case got < 1<<27:
			t.Errorf("%s: MaxBuffer %d is below the spec's floor of 2^27", d.Name(), got)
		default:
			t.Logf("%s: MaxBuffer %d (vulkaninfo %v)", d.Name(), got, ok)
			checked++
		}
	}
	if checked == 0 && !t.Failed() {
		t.Skip("no Vulkan device opened -- this gate proved nothing")
	}
}

// TestMetalMaxBufferIsRead holds MaxBuffer to what a Metal device must say:
// read (not the zero of a failed selector), at least the 256 MiB every Apple
// GPU allows, and no more than the working set the driver recommends.
func TestMetalMaxBufferIsRead(t *testing.T) {
	testdevices.Require(t, "metal")
	d, err := OpenMetalWith(Opts{})
	if err != nil || d == nil {
		t.Skipf("no Metal device: %v -- this gate proved nothing", err)
	}
	defer d.Close()
	got := d.(BufferLimited).MaxBuffer()
	_, total, err := d.Mem()
	if err != nil {
		t.Fatal(err)
	}
	if got < 256<<20 || got > total {
		t.Fatalf("%s: MaxBuffer %d, outside [256 MiB, the %d-byte working set]", d.Name(), got, total)
	}
	t.Logf("%s: MaxBuffer %d of a %d-byte working set", d.Name(), got, total)
}
