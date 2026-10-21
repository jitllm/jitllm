package vulkan

import (
	"bufio"
	"bytes"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

// The specification's floors on the two limits read from one word of
// VkPhysicalDeviceLimits (Table "Required Limits"). They are what a misread
// offset is least likely to satisfy by accident, and nothing outside this test
// uses them: the engine reads the device's own value.
const (
	specMinGroupCount  = 65535
	specMinSharedBytes = 16384
)

// TestWorkGroupCountIsReadFromTheDriver pins maxComputeWorkGroupCount[0] to the
// driver: every device's reading clears the specification's floor, the word's
// other half (maxComputeSharedMemorySize) clears its own, and where vulkaninfo
// is installed both match what it prints for the same enumeration index. A
// wrong offset compiles, runs and reads a plausible number, so an independent
// reader of the same struct is the gate.
func TestWorkGroupCountIsReadFromTheDriver(t *testing.T) {
	inst := openInstance(t)
	cands, err := enumerate(inst)
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	ref := vulkaninfoLimits(t)
	for _, c := range cands {
		var props physProps
		vkGetPhysicalDeviceProperties(c.phys, up(&props))
		groups, shared := props.maxComputeWorkGroupCount(), props.maxComputeSharedMemorySize()
		t.Logf("GPU%d %s: maxComputeWorkGroupCount[0] %d, maxComputeSharedMemorySize %d",
			c.idx, c.name, groups, shared)
		if c.maxGroups != groups {
			t.Errorf("GPU%d: enumerate kept %d, the properties say %d", c.idx, c.maxGroups, groups)
		}
		if groups < specMinGroupCount || shared < specMinSharedBytes {
			t.Errorf("GPU%d: %d groups / %d shared bytes is below the specification's floor "+
				"(%d / %d): the limits are read at the wrong offset",
				c.idx, groups, shared, specMinGroupCount, specMinSharedBytes)
		}
		if ref != nil {
			// A parse that found nothing must not read as agreement.
			r, ok := ref[c.idx]
			if !ok || r[0] == 0 || r[1] == 0 {
				t.Errorf("GPU%d: vulkaninfo is installed and its limits for this device were not parsed (%v)", c.idx, r)
			} else if r[0] != groups || r[1] != shared {
				t.Errorf("GPU%d: read %d groups / %d shared bytes, vulkaninfo prints %d / %d",
					c.idx, groups, shared, r[0], r[1])
			}
		}
	}
	if ref == nil {
		t.Log("vulkaninfo not installed: the readings are checked against the floors only")
	}
}

// vulkaninfoLimits is maxComputeWorkGroupCount[0] and maxComputeSharedMemorySize
// per GPU index, as vulkaninfo prints them, or nil when it is not installed.
func vulkaninfoLimits(t *testing.T) map[int][2]uint32 {
	path, err := exec.LookPath("vulkaninfo")
	if err != nil {
		return nil
	}
	out, err := exec.Command(path).Output()
	if err != nil {
		t.Logf("vulkaninfo: %v", err)
		return nil
	}
	ref := map[int][2]uint32{}
	gpu, countNext := -1, false
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		switch {
		case strings.HasPrefix(line, "GPU") && strings.HasSuffix(line, ":"):
			if n, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(line, "GPU"), ":")); err == nil {
				gpu = n
			}
		case gpu < 0:
		case strings.HasPrefix(line, "maxComputeSharedMemorySize"):
			if v, ok := afterEquals(line); ok {
				r := ref[gpu]
				r[1] = v
				ref[gpu] = r
			}
		case strings.HasPrefix(line, "maxComputeWorkGroupCount:"):
			countNext = true
		case countNext:
			countNext = false
			if v, err := strconv.ParseUint(line, 10, 32); err == nil {
				r := ref[gpu]
				r[0] = uint32(v)
				ref[gpu] = r
			}
		}
	}
	return ref
}

func afterEquals(line string) (uint32, bool) {
	i := strings.LastIndex(line, "=")
	if i < 0 {
		return 0, false
	}
	v, err := strconv.ParseUint(strings.TrimSpace(line[i+1:]), 10, 32)
	return uint32(v), err == nil
}
