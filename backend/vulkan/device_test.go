package vulkan

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// openInstance is the process's VkInstance alone, which is all the selection
// tests need -- they never create a logical device, so a host with a
// display-only card or a broken ICD still runs them.
func openInstance(t *testing.T) Instance {
	t.Helper()
	if err := load(); err != nil {
		t.Skipf("no Vulkan loader: %v", err)
	}
	inst, err := instance()
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	return inst
}

// TestEnumerateSeesEveryDevice is the precondition for everything below: if the
// loader only ever hands back one device, "selection" is untestable and a green
// line here would mean nothing (RULE 10).
func TestEnumerateSeesEveryDevice(t *testing.T) {
	cands, err := enumerate(openInstance(t))
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	for _, c := range cands {
		t.Logf("  %s", c.describe())
	}
	if len(cands) < 2 {
		t.Logf("NOTE: %d device(s) here, so TestPickIsDeliberate cannot "+
			"distinguish a real pick from taking index 0 on this box", len(cands))
	}
}

// TestPickIsDeliberate asserts the pick is the highest-ranked candidate with a
// compute queue, not the first one: on a host that enumerates the iGPU or
// llvmpipe ahead of a discrete card, taking index 0 fails here.
func TestPickIsDeliberate(t *testing.T) {
	inst := openInstance(t)
	cands, err := enumerate(inst)
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	got, err := pickDevice(inst, "", "")
	if err != nil {
		t.Fatalf("pickDevice: %v", err)
	}
	t.Logf("picked %s out of %d", got.describe(), len(cands))

	if got.family == noFamily {
		t.Fatalf("picked %s, which has no compute queue", got.describe())
	}
	for _, c := range cands {
		if c.family == noFamily {
			continue
		}
		if devTypeRank(c.devType) > devTypeRank(got.devType) {
			t.Errorf("picked %s but %s outranks it", got.describe(), c.describe())
		}
	}
	// Ties keep enumeration order, so the winner must be the FIRST candidate at
	// its rank. Anything else means the loop is not stable.
	for _, c := range cands {
		if c.family != noFamily && devTypeRank(c.devType) == devTypeRank(got.devType) {
			if c.idx != got.idx {
				t.Errorf("tie at rank %d broken towards [%d], not the earliest [%d]",
					devTypeRank(got.devType), got.idx, c.idx)
			}
			break
		}
	}
	if got.devType == devTypeCPU {
		// Allowed only when there is nothing else; see pickDevice.
		for _, c := range cands {
			if c.family != noFamily && c.devType != devTypeCPU {
				t.Errorf("picked the software rasteriser %s while %s was available",
					got.describe(), c.describe())
			}
		}
	}
}

// TestNoComputeQueueFallsThrough drives the fall-through with a synthetic
// candidate list, since a test host rarely has a device without a compute queue. The
// ranking is pure, so it can be exercised directly.
func TestNoComputeQueueFallsThrough(t *testing.T) {
	// The best device first, with no compute queue.
	cands := []candidate{
		{idx: 0, name: "display-only", devType: devTypeDiscrete, family: noFamily},
		{idx: 1, name: "iGPU", devType: devTypeIntegrated, family: 0},
		{idx: 2, name: "llvmpipe", devType: devTypeCPU, family: 0},
	}
	if best := bestOf(cands); best < 0 || cands[best].name != "iGPU" {
		t.Fatalf("fall-through picked %d (%v), want iGPU", best, cands)
	}

	// And a list where only a CPU device has a queue still yields that device,
	// because vulkan.Open must keep working on a lavapipe-only host.
	only := []candidate{
		{idx: 0, name: "display-only", devType: devTypeDiscrete, family: noFamily},
		{idx: 1, name: "llvmpipe", devType: devTypeCPU, family: 0},
	}
	if best := bestOf(only); best < 0 || only[best].name != "llvmpipe" {
		t.Fatalf("software-only list picked %d, want llvmpipe", best)
	}
}

// TestOverridePinsByIndexAndName opens every enumerated device through the
// selector and reports its name, type and memory. It fails if two indices
// produce the same device.
func TestOverridePinsByIndexAndName(t *testing.T) {
	cands, err := enumerate(openInstance(t))
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	seen := map[string]int{}
	for _, want := range cands {
		if want.family == noFamily {
			continue
		}
		t.Run(strconv.Itoa(want.idx)+"/"+want.name, func(t *testing.T) {
			c, err := OpenDeviceWith("", Config{Device: strconv.Itoa(want.idx)})
			if err != nil {
				t.Fatalf("Open pinned to %d: %v", want.idx, err)
			}
			defer c.Close()
			if !strings.HasPrefix(c.Name(), want.name) {
				t.Fatalf("pinned index %d opened %q, want %q", want.idx, c.Name(), want.name)
			}
			if c.Software() != (want.devType == devTypeCPU) {
				t.Errorf("Software()=%v for %s", c.Software(), devTypeName(want.devType))
			}
			// The literal, not softwareSuffix: a gate that compares a constant
			// with itself passes however the constant is mutated.
			if want.devType == devTypeCPU && !strings.Contains(c.Name(), "SOFTWARE RASTERISER") {
				t.Errorf("a CPU device is named %q, with no warning in it", c.Name())
			}
			if want := want.devType == devTypeIntegrated || want.devType == devTypeCPU; c.UnifiedMemory() != want {
				t.Errorf("UnifiedMemory()=%v for a %s, want %v", c.UnifiedMemory(), c.DeviceType(), want)
			}
			free, total, err := c.Mem()
			t.Logf("%-46s %-26s unified=%-5v  %s", c.Name(), c.DeviceType(), c.UnifiedMemory(),
				memLine(free, total, err))
			if total == 0 && err == nil {
				t.Errorf("total memory 0 with no error")
			}
			seen[c.Name()]++

			// The name form must reach the same device as the index form.
			c2, err := OpenDeviceWith("", Config{Device: want.name})
			if err != nil {
				t.Fatalf("Open pinned to %q: %v", want.name, err)
			}
			defer c2.Close()
			if c2.Name() != c.Name() {
				t.Errorf("name pin opened %q, index pin opened %q", c2.Name(), c.Name())
			}
		})
	}
	if len(seen) < 2 {
		t.Logf("NOTE: only %d distinct device(s) opened; selection is exercised but "+
			"not DISTINGUISHED on this box", len(seen))
	}
}

func memLine(free, total uint64, err error) string {
	if err != nil {
		return "mem unknown: " + err.Error()
	}
	return strconv.FormatFloat(float64(free)/(1<<30), 'f', 2, 64) + " of " +
		strconv.FormatFloat(float64(total)/(1<<30), 'f', 2, 64) + " GiB free"
}

// TestOverrideMissIsAnError: a pin that matches nothing must fail rather than
// quietly measure a different card.
func TestOverrideMissIsAnError(t *testing.T) {
	openInstance(t) // skip early if there is no loader
	for _, v := range []string{"no-such-device-xyzzy", "9999"} {
		c, err := OpenDeviceWith("", Config{Device: v})
		if err == nil {
			c.Close()
			t.Errorf("a pin of %q opened %q instead of failing", v, c.Name())
			continue
		}
		t.Logf("pin %q -> %v", v, err)
	}
}

// TestMemIsTheDeviceLocalHeap checks the number against core Vulkan's own heap
// table, and against the invariants a budget must satisfy.
//
// A discrete card also exposes host RAM as a heap, so total must equal one
// device-local heap, never the sum.
func TestMemIsTheDeviceLocalHeap(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer c.Close()

	var sum uint64
	local := -1
	for i := uint32(0); i < c.mem.heapCount; i++ {
		sum += c.mem.heaps[i].size
		t.Logf("heap[%d] %7.2f GiB flags %#x", i, float64(c.mem.heaps[i].size)/(1<<30), c.mem.heaps[i].flags)
		if c.mem.heaps[i].flags&heapDeviceLocal != 0 && local < 0 {
			local = int(i)
		}
	}
	free, total, err := c.Mem()
	t.Logf("%s (%s): %s, unified=%v, budget extension=%v",
		c.Name(), c.DeviceType(), memLine(free, total, err), c.UnifiedMemory(), c.budget)

	h := c.localHeap()
	if c.mem.heaps[h].flags&heapDeviceLocal == 0 && local >= 0 {
		t.Errorf("localHeap picked %d, which is not DEVICE_LOCAL, while %d is", h, local)
	}
	if total != c.mem.heaps[h].size {
		t.Errorf("total %d, heap[%d] is %d", total, h, c.mem.heaps[h].size)
	}
	if c.mem.heapCount > 1 && total == sum {
		t.Errorf("total %d equals the SUM of %d heaps -- heaps are being added up", total, c.mem.heapCount)
	}
	if err == nil && free > total {
		t.Errorf("free %d exceeds total %d", free, total)
	}
	if !c.budget && err == nil {
		t.Errorf("no %s, yet Mem reported free=%d with no error", extMemoryBudget, free)
	}
	if c.budget && err != nil {
		t.Errorf("%s is supported, yet Mem failed: %v", extMemoryBudget, err)
	}
	// Unified memory is a property of the device type, not of the heap flags.
	if c.UnifiedMemory() != (c.devType == devTypeIntegrated || c.devType == devTypeCPU) {
		t.Errorf("UnifiedMemory()=%v for a %s", c.UnifiedMemory(), c.DeviceType())
	}
}

// TestMemAgreesWithTheKernelDriver cross-checks the discrete card's numbers
// against nvidia-smi, which is an independent witness -- a wrong heapBudget
// offset reads a plausible number from the wrong place and no invariant above
// would notice.
func TestMemAgreesWithTheKernelDriver(t *testing.T) {
	// The card's free memory is a global, and the two samples below are taken
	// at different instants; see gpuLock.
	gpuLock(t)
	total, free, ok := nvidiaSMI()
	if !ok {
		t.Skip("nvidia-smi not available")
	}
	t.Setenv("JITLLM_VK_DEVICE", "NVIDIA")
	c, err := Open()
	if err != nil {
		t.Skipf("no NVIDIA Vulkan device: %v", err)
	}
	defer c.Close()
	vkFree, vkTotal, err := c.Mem()
	if err != nil {
		t.Fatalf("Mem: %v", err)
	}
	t.Logf("vulkan %6.2f of %6.2f GiB free", float64(vkFree)/(1<<30), float64(vkTotal)/(1<<30))
	t.Logf("smi    %6.2f of %6.2f GiB free", float64(free)/(1<<30), float64(total)/(1<<30))
	if vkTotal != total {
		t.Errorf("total: vulkan %d, nvidia-smi %d", vkTotal, total)
	}
	// Free moves between the two samples; 10% is slack for that, not for a
	// layout slip, which would be off by orders of magnitude or read a heap
	// index's worth of zeroes.
	if d := ratioOff(vkFree, free); d > 0.10 {
		t.Errorf("free: vulkan %d, nvidia-smi %d, %.1f%% apart", vkFree, free, d*100)
	}
}

func ratioOff(a, b uint64) float64 {
	if b == 0 {
		if a == 0 {
			return 0
		}
		return 1
	}
	d := float64(a) - float64(b)
	if d < 0 {
		d = -d
	}
	return d / float64(b)
}

// nvidiaSMI reads total and free bytes from the kernel driver's own accounting.
func nvidiaSMI() (total, free uint64, ok bool) {
	b, err := runSMI()
	if err != nil {
		return 0, 0, false
	}
	for _, ln := range strings.Split(string(b), "\n") {
		f := strings.Split(ln, ",")
		if len(f) != 2 {
			continue
		}
		t, e1 := strconv.ParseUint(strings.Fields(strings.TrimSpace(f[0]))[0], 10, 64)
		fr, e2 := strconv.ParseUint(strings.Fields(strings.TrimSpace(f[1]))[0], 10, 64)
		if e1 == nil && e2 == nil {
			return t << 20, fr << 20, true // MiB
		}
	}
	return 0, 0, false
}

func runSMI() ([]byte, error) {
	if _, err := os.Stat("/usr/bin/nvidia-smi"); err != nil {
		return nil, err
	}
	return exec.Command("/usr/bin/nvidia-smi",
		"--query-gpu=memory.total,memory.free", "--format=csv,noheader").Output()
}

// TestConfigIsPerDevice: contexts opened concurrently with different Configs
// each keep their own pin and subgroup override. Models load concurrently in a
// server, so a Config held in a package variable between an open's start and
// its device pick hands one open another's pin; run under -race.
func TestConfigIsPerDevice(t *testing.T) {
	cands, err := enumerate(openInstance(t))
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	var last candidate
	for _, c := range cands {
		if c.family != noFamily {
			last = c
		}
	}
	cfgs := []Config{{Device: strconv.Itoa(last.idx), Subgroup: "off"}, {Subgroup: "force"}}
	for round := 0; round < 4; round++ {
		var wg sync.WaitGroup
		errs := make(chan string, 8)
		for i := 0; i < 8; i++ {
			c := cfgs[(i+round)%2]
			wg.Add(1)
			go func() {
				defer wg.Done()
				ctx, err := OpenDeviceWith("", c)
				if err != nil {
					errs <- fmt.Sprintf("OpenDeviceWith(%+v): %v", c, err)
					return
				}
				defer ctx.Close()
				sg := ctx.Subgroups()
				if sg.Off != (c.Subgroup == "off") || sg.Forced != (c.Subgroup == "force") ||
					(c.Device != "") != (ctx.Index() == last.idx && last.idx != 0) {
					errs <- fmt.Sprintf("%s (index %d) opened with %+v reports off=%v forced=%v",
						ctx.Name(), ctx.Index(), c, sg.Off, sg.Forced)
				}
			}()
		}
		wg.Wait()
		close(errs)
		for e := range errs {
			t.Error(e)
		}
	}
}
