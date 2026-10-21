//go:build darwin

package metal

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// TestDeviceMem checks that Mem and UnifiedMemory read the three MTLDevice
// properties they claim to, and not three plausible integers.
//
// Every assertion is a range or a delta, not "err == nil": a binding with the
// wrong return type does not fail, it reads whatever was in the register. So
// total is bounded by the machine's RAM (a quarter to all of hw.memsize), free
// must drop by about a 256 MiB allocation, and the BOOL must be exactly 1.
func TestDeviceMem(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer c.Close()

	free, total, err := c.Mem()
	if err != nil {
		t.Fatalf("Mem: %v", err)
	}
	const giB = 1 << 30
	t.Logf("device            %s", c.Name())
	t.Logf("recommended max working set   %13d  (%.2f GiB)", total, float64(total)/giB)
	t.Logf("free = recommended - current  %13d  (%.2f GiB)", free, float64(free)/giB)
	t.Logf("hasUnifiedMemory              %13v", c.UnifiedMemory())

	if free > total {
		t.Errorf("free %d > total %d", free, total)
	}
	// A floor and ceiling for any Mac, before the sysctl cross-check narrows
	// them.
	if total < 1*giB || total > 1024*giB {
		t.Errorf("recommendedMaxWorkingSetSize = %d (%.2f GiB), outside any plausible range for a Mac",
			total, float64(total)/giB)
	}

	// The real bound is the machine's own RAM, read in the same run: the
	// working-set budget cannot exceed hw.memsize, and under a quarter of it
	// would not be worth reporting.
	if ram := sysctlUint64(t, "hw.memsize"); ram > 0 {
		t.Logf("hw.memsize                    %13d  (%.2f GiB)", ram, float64(ram)/giB)
		if total > ram {
			t.Errorf("recommendedMaxWorkingSetSize %d exceeds hw.memsize %d", total, ram)
		}
		if total < ram/4 {
			t.Errorf("recommendedMaxWorkingSetSize %d is under a quarter of hw.memsize %d", total, ram)
		}
	}

	// currentAllocatedSize must track: 256 MiB is far above background noise
	// and far below the budget.
	const want = 256 << 20
	b, err := c.Alloc(want)
	if err != nil {
		t.Fatalf("Alloc(%d): %v", want, err)
	}
	afterAlloc, totalAfter, err := c.Mem()
	if err != nil {
		t.Fatalf("Mem after Alloc: %v", err)
	}
	t.Logf("free after a %d MiB buffer  %13d  (%.2f GiB)", want>>20, afterAlloc, float64(afterAlloc)/giB)
	if totalAfter != total {
		t.Errorf("recommendedMaxWorkingSetSize moved across an allocation: %d -> %d", total, totalAfter)
	}
	// Allow the rest of the process to allocate underneath in either
	// direction; free must still drop by most of the buffer.
	if drop := int64(free) - int64(afterAlloc); drop < want*7/8 {
		t.Errorf("free dropped by %d after allocating %d bytes; currentAllocatedSize is not tracking allocations",
			drop, want)
	}
	b.Free()
	afterFree, _, err := c.Mem()
	if err != nil {
		t.Fatalf("Mem after Free: %v", err)
	}
	t.Logf("free after releasing it       %13d  (%.2f GiB)", afterFree, float64(afterFree)/giB)
	if afterFree < afterAlloc {
		t.Errorf("free went DOWN after releasing the buffer: %d -> %d", afterAlloc, afterFree)
	}

	// The BOOL read raw: UnifiedMemory() is a cached Go bool and would hide a
	// wrong-width read.
	raw := sendBool(c.dev, sel("hasUnifiedMemory"))
	t.Logf("hasUnifiedMemory raw byte     %13d", uint8(raw))
	if raw != 0 && raw != 1 {
		t.Errorf("hasUnifiedMemory returned %d; BOOL is 0 or 1, so that register did not hold a BOOL", uint8(raw))
	}
	if raw.ok() != c.UnifiedMemory() {
		t.Errorf("UnifiedMemory() = %v but the selector says %d", c.UnifiedMemory(), uint8(raw))
	}
	if runtime.GOARCH == "arm64" && !c.UnifiedMemory() {
		t.Error("hasUnifiedMemory is false on Apple Silicon, where device and host share one pool")
	}

	// Why the two shapes are bound separately, demonstrated: an NSUInteger
	// read through the BOOL binding keeps only the low byte, a plausible
	// wrong number. A buffer is held so the true value exceeds a byte.
	t.Run("return shape", func(t *testing.T) {
		keep, err := c.Alloc(want)
		if err != nil {
			t.Fatalf("Alloc(%d): %v", want, err)
		}
		defer keep.Free()
		wide := sendU64(c.dev, sel("currentAllocatedSize"))
		narrow := sendBool(c.dev, sel("currentAllocatedSize"))
		t.Logf("currentAllocatedSize through the NSUInteger binding  0x%016x", wide)
		t.Logf("currentAllocatedSize through the BOOL binding        0x%016x", uint8(narrow))
		if wide <= 0xff {
			t.Fatalf("currentAllocatedSize = %d, too small for this to show anything", wide)
		}
		if uint8(narrow) != uint8(wide) {
			t.Errorf("the BOOL-shaped read gave %d, not the low byte %d of %d: "+
				"goffi is not truncating to the declared return size", uint8(narrow), uint8(wide), wide)
		}
		if narrow.ok() == (wide != 0) && uint8(wide) != 0 {
			t.Logf("(the low byte is non-zero this time, so the truth value survived by luck)")
		}
	})
}

// sysctlUint64 reads one integer sysctl. It shells out rather than binding
// sysctl(3): std's syscall.SysctlUint32 truncates hw.memsize, x/sys is not a
// dependency of this module, and the value is wanted once in a test.
func sysctlUint64(t *testing.T, name string) uint64 {
	t.Helper()
	out, err := exec.Command("/usr/sbin/sysctl", "-n", name).Output()
	if err != nil {
		t.Logf("sysctl %s: %v (skipping the physical-RAM cross-check)", name, err)
		return 0
	}
	v, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		t.Logf("sysctl %s: %v", name, err)
		return 0
	}
	return v
}
