package vulkan

// Physical-device selection and the memory report.
//
// vkEnumeratePhysicalDevices returns every ICD the loader found, in the order
// it walked icd.d, which may put an iGPU or llvmpipe (a CPU software rasteriser)
// first. llvmpipe answers every call correctly and is merely slow, so nothing
// downstream would notice blocks placed on the host's own cores through a
// Vulkan queue. Hence the explicit ranking below.

import (
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// candidate is one enumerated physical device with everything the pick needs.
type candidate struct {
	phys    PhysDevice
	idx     int // enumeration index, which is vulkaninfo's GPUn
	name    string
	devType uint32
	family  uint32 // compute queue family, or noFamily
	queues  int    // how many queues that family has
	// apiVer is the device's VkPhysicalDeviceProperties.apiVersion, not the
	// instance's. A query that is core in 1.1 is invalid usage on a 1.0 device,
	// and the instance asking for 1.3 says nothing about what the device
	// answers to; see readSubgroups.
	apiVer uint32
	// driverVer is VkPhysicalDeviceProperties.driverVersion, vendor-encoded.
	driverVer uint32
	// uuid is VkPhysicalDeviceIDProperties.deviceUUID, and uuidOK is false when
	// the device cannot be asked for one. See readDeviceUUID.
	uuid   [uuidSize]byte
	uuidOK bool
	// maxStorage is VkPhysicalDeviceLimits.maxStorageBufferRange: the most
	// bytes one storage-buffer binding may address.
	maxStorage uint32
	// maxGroups is VkPhysicalDeviceLimits.maxComputeWorkGroupCount[0]: the
	// most workgroups one dispatch may ask for along x. See Ctx.MaxGroups.
	maxGroups uint32
}

const noFamily = ^uint32(0)

// devTypeRank orders VkPhysicalDeviceType by how much a model wants to run on
// it. Higher wins; ties keep enumeration order.
//
// OTHER outranks nothing but CPU: it is what a driver reports when it is real
// hardware the enum has no name for, so it is worth taking over nothing, and
// worth taking last. A CPU device is negative rather than 0, so that "the only
// candidate is a software rasteriser" is a case a caller can test for with a
// sign rather than by re-listing the enum; backend.openVulkan is that caller.
func devTypeRank(t uint32) int {
	switch t {
	case devTypeDiscrete:
		return 4
	case devTypeIntegrated:
		return 3
	case devTypeVirtual:
		return 2
	case devTypeOther:
		return 1
	default: // devTypeCPU
		return -1
	}
}

func devTypeName(t uint32) string {
	switch t {
	case devTypeDiscrete:
		return "discrete GPU"
	case devTypeIntegrated:
		return "integrated GPU"
	case devTypeVirtual:
		return "virtual GPU"
	case devTypeOther:
		return "other"
	case devTypeCPU:
		return "CPU (software rasteriser)"
	}
	return fmt.Sprintf("unknown type %d", t)
}

// softwareSuffix is appended to the device name of a CPU device, so that every
// consumer of Name() -- the hardware report, JITLLM_GPU_VERBOSE, the tune
// cache key -- carries the warning rather than one call site remembering it.
const softwareSuffix = " [SOFTWARE RASTERISER -- NOT A GPU]"

// ErrDevicesVanished is an enumeration that found fewer physical devices than
// an earlier one in the same process. A driver that stops loading partway
// through a process does not fail a call -- the loader drops its ICD and
// enumerates the rest -- so without this an open of a device that was there a
// moment ago reads as "this host has none", and a caller that skips on a
// missing device skips everything from then on.
var ErrDevicesVanished = errors.New("vulkan: a device an earlier enumeration in this process found is gone")

// seen is the first enumeration's device count in this process, and shrunk the
// count of the first enumeration that came back with fewer; see Vanished.
var seen struct {
	sync.Mutex
	first, shrunk int
}

// Vanished reports the device count this process first enumerated and the
// count of the first later enumeration that found fewer, or (0, 0) when none
// has. A test binary checks it after its tests, so a device that vanished
// fails the run even where a test skipped on it.
func Vanished() (first, now int) {
	seen.Lock()
	defer seen.Unlock()
	if seen.shrunk == 0 {
		return 0, 0
	}
	return seen.first, seen.shrunk
}

// noteCount records n against the process's first enumeration and refuses a
// count below it.
func noteCount(n int) error {
	seen.Lock()
	defer seen.Unlock()
	if seen.first == 0 {
		seen.first = n
	}
	if n >= seen.first {
		return nil
	}
	if seen.shrunk == 0 {
		seen.shrunk = n
	}
	return fmt.Errorf("%w: the loader enumerates %d device(s) where this process first enumerated %d "+
		"(VK_LOADER_DEBUG=error names the driver that failed to load)", ErrDevicesVanished, n, seen.first)
}

// enumerate reads every physical device's name, type and compute queue family.
func enumerate(inst Instance) ([]candidate, error) {
	var n uint32
	if err := check(vkEnumeratePhysicalDevices(inst, &n, nil), "vkEnumeratePhysicalDevices"); err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, fmt.Errorf("vulkan: no physical devices")
	}
	devs := make([]PhysDevice, n)
	if err := check(vkEnumeratePhysicalDevices(inst, &n, up(&devs[0])), "vkEnumeratePhysicalDevices"); err != nil {
		return nil, err
	}
	if err := noteCount(int(n)); err != nil {
		return nil, err
	}
	cands := make([]candidate, 0, n)
	for i, p := range devs {
		var props physProps
		vkGetPhysicalDeviceProperties(p, up(&props))
		c := candidate{phys: p, idx: i, name: gostr(props.name[:]), devType: props.devType,
			family: noFamily, apiVer: props.apiVer, driverVer: props.driverVer}
		c.uuid, c.uuidOK = readDeviceUUID(p, props.apiVer)
		c.maxStorage = props.maxStorageBufferRange()
		c.maxGroups = props.maxComputeWorkGroupCount()

		var nq uint32
		vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, nil)
		if nq > 0 {
			fams := make([]queueFamily, nq)
			vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, up(&fams[0]))
			// A compute-only family first: on NVIDIA the graphics family's
			// queues run one after another, and the compute-only one is where
			// two sessions' submissions run at once. Any compute family else.
			for _, only := range []bool{true, false} {
				for j, f := range fams {
					if f.flags&queueCompute != 0 && (!only || f.flags&queueGraphics == 0) {
						c.family, c.queues = uint32(j), int(f.count)
						break
					}
				}
				if c.family != noFamily {
					break
				}
			}
		}
		cands = append(cands, c)
	}
	return cands, nil
}

func (c candidate) describe() string {
	s := fmt.Sprintf("[%d] %s (%s", c.idx, c.name, devTypeName(c.devType))
	if c.family == noFamily {
		s += ", NO COMPUTE QUEUE"
	}
	return s + ")"
}

func describeAll(cands []candidate) string {
	parts := make([]string, len(cands))
	for i, c := range cands {
		parts[i] = c.describe()
	}
	return strings.Join(parts, ", ")
}

// pickDevice chooses among the enumerated devices.
//
// Order: discrete, then integrated, then virtual, then other, ties broken by
// enumeration order. A candidate with no compute queue falls through to the
// next one.
//
// A CPU device is taken only when there is nothing else. The refusal that
// matters is one layer up, in backend.openVulkan, and the split is deliberate:
//
//   - vulkan.Open is a SPIR-V runtime. On a host with only lavapipe it is the
//     only way to run SPIR-V at all, and the correctness tests rely on it.
//   - backend.Device is the list the seam tuner measures and places blocks
//     against, so openVulkan drops a software rasteriser unless a selector
//     named it.
//
// Either way Open appends softwareSuffix to the name of a CPU device.
//
// sel is the caller's selector; "" falls back to pin (Config.Device) and then
// to the ordering.
func pickDevice(inst Instance, sel, pin string) (candidate, error) {
	cands, err := enumerate(inst)
	if err != nil {
		return candidate{}, err
	}

	if sel == "" {
		sel = pin
	}
	if sel != "" {
		return pinned(cands, sel)
	}

	best := bestOf(cands)
	if best < 0 {
		return candidate{}, fmt.Errorf("vulkan: no device with a compute queue among %s",
			describeAll(cands))
	}
	return cands[best], nil
}

// bestOf returns the index of the winning candidate, or -1.
//
// A device with no compute queue falls through to the next one rather than
// failing the whole backend. Ties keep enumeration order, so the `>` is strict.
// It is a function so a test can drive it with a synthetic candidate list.
func bestOf(cands []candidate) int {
	best := -1
	for i, c := range cands {
		if c.family == noFamily {
			continue
		}
		if best < 0 || devTypeRank(c.devType) > devTypeRank(cands[best].devType) {
			best = i
		}
	}
	return best
}

// pinned resolves a selector -- OpenDevice's argument or Config.Device, which
// are the same grammar: an enumeration index, or a case-insensitive substring of
// the device name.
//
// A pin that misses is an error, not a fallback: silently using a different
// device defeats the pin.
func pinned(cands []candidate, want string) (candidate, error) {
	var got *candidate
	if i, err := strconv.Atoi(want); err == nil {
		if i < 0 || i >= len(cands) {
			return candidate{}, fmt.Errorf("vulkan: device %d is out of range, this host has %d: %s",
				i, len(cands), describeAll(cands))
		}
		got = &cands[i]
	} else {
		for i := range cands {
			if strings.Contains(strings.ToLower(cands[i].name), strings.ToLower(want)) {
				got = &cands[i]
				break
			}
		}
		if got == nil {
			return candidate{}, fmt.Errorf("vulkan: %q matches no device name: %s",
				want, describeAll(cands))
		}
	}
	if got.family == noFamily {
		return candidate{}, fmt.Errorf("vulkan: selected %s, which has no compute queue",
			got.describe())
	}
	return *got, nil
}

// readDeviceUUID asks the physical device for the sixteen bytes that identify
// the hardware, and reports false when it cannot be asked.
//
// A device below Vulkan 1.1 is a refusal, not a zero UUID: the query is invalid
// usage there, and an all-zero UUID would compare equal to every other device
// that could not be asked, merging distinct cards.
func readDeviceUUID(p PhysDevice, apiVer uint32) ([uuidSize]byte, bool) {
	var zero [uuidSize]byte
	if apiVer < apiVersion11 {
		return zero, false
	}
	id := idProps{sType: stPhysDevIDProps}
	p2 := physProps2{sType: stPhysDevProps2, pNext: uintptr(up(&id))}
	vkGetPhysicalDeviceProperties2(p, up(&p2))
	out := id.deviceUUID
	// id is reachable only through a uintptr pNext, which is not a reference.
	runtime.KeepAlive(id)
	return out, true
}

// hasExtension reports whether the physical device supports a device extension.
func hasExtension(p PhysDevice, name string) (bool, error) {
	var n uint32
	if err := check(vkEnumerateDeviceExtensionProperties(p, nil, &n, nil), "vkEnumerateDeviceExtensionProperties"); err != nil {
		return false, err
	}
	if n == 0 {
		return false, nil
	}
	exts := make([]extProps, n)
	if err := check(vkEnumerateDeviceExtensionProperties(p, nil, &n, up(&exts[0])), "vkEnumerateDeviceExtensionProperties"); err != nil {
		return false, err
	}
	for i := range exts {
		if gostr(exts[i].name[:]) == name {
			return true, nil
		}
	}
	return false, nil
}

// localHeap picks the heap whose size Mem reports.
//
// Summing heaps over-reports: a discrete card also exposes host RAM it can
// reach over PCIe. Alloc asks for memDeviceLocal, so the device-local heap is
// the only one a weight budget may be drawn against.
//
// Largest-device-local rather than first: a multi-heap driver may expose a
// small device-local BAR window beside the real one.
func (c *Ctx) localHeap() uint32 {
	best, bestSize := ^uint32(0), uint64(0)
	for i := uint32(0); i < c.mem.heapCount && i < maxMemoryHeaps; i++ {
		if c.mem.heaps[i].flags&heapDeviceLocal == 0 {
			continue
		}
		if best == ^uint32(0) || c.mem.heaps[i].size > bestSize {
			best, bestSize = i, c.mem.heaps[i].size
		}
	}
	if best != ^uint32(0) {
		return best
	}
	// No device-local heap at all. Alloc's memType falls back to host-visible
	// in exactly this case, so report the heap it will land in: the largest.
	for i := uint32(0); i < c.mem.heapCount && i < maxMemoryHeaps; i++ {
		if best == ^uint32(0) || c.mem.heaps[i].size > bestSize {
			best, bestSize = i, c.mem.heaps[i].size
		}
	}
	if best == ^uint32(0) {
		return 0
	}
	return best
}

// UnifiedMemory reports that the heap Mem describes is host RAM. When true, a
// device budget and a host budget are claims on the same bytes: a caller must
// subtract what it puts on the device from the host's figure rather than add
// the two.
//
// It is decided by device type and not by heap flags. An integrated GPU's
// device-local heap is system memory by construction, and so is a CPU device's;
// a discrete card with resizable BAR also exposes a device-local heap that is
// host-visible, and that memory is still VRAM. Flags cannot tell those apart,
// the type can.
func (c *Ctx) UnifiedMemory() bool {
	return c.devType == devTypeIntegrated || c.devType == devTypeCPU
}

// DeviceType is the VkPhysicalDeviceType this context opened, as text.
func (c *Ctx) DeviceType() string { return devTypeName(c.devType) }

// Software reports that this context is a CPU device -- llvmpipe/lavapipe or
// swiftshader -- executing SPIR-V on the host's own cores. It runs kernels
// correctly, which is why the Vulkan package's correctness tests are allowed to
// use one, and it must never be offered to the seam tuner as a place to put a
// block.
func (c *Ctx) Software() bool { return c.devType == devTypeCPU }

// Mem reports the free and total bytes of the device-local heap, satisfying the
// same optional interface cmd/jitllm reads on CUDA.
//
// Free comes from VK_EXT_memory_budget: heapBudget is what this process may
// still allocate (net of other processes), heapUsage is what it holds. Neither
// is the heap size.
//
// Without the extension this returns an error with the total still set: free
// memory is unknown, and inventing it would feed a placement decision.
func (c *Ctx) Mem() (free, total uint64, err error) {
	h := c.localHeap()
	if h < maxMemoryHeaps {
		total = c.mem.heaps[h].size
	}
	if !c.budget {
		return 0, total, fmt.Errorf("%s does not support %s, so free memory is unknown (heap %d is %.2f GiB)",
			c.name, extMemoryBudget, h, float64(total)/(1<<30))
	}
	var b memBudgetEXT
	b.sType = stMemBudgetEXT
	var p2 memProps2
	p2.sType = stPhysDevMemProps2
	p2.pNext = uintptr(up(&b))
	vkGetPhysicalDeviceMemoryProperties2(c.phys, up(&p2))

	// The budget query returns the core properties too; take the heap size from
	// the same call so size and budget cannot come from two different reads.
	if h < maxMemoryHeaps {
		total = p2.props.heaps[h].size
	}
	if b.usage[h] < b.budget[h] {
		free = b.budget[h] - b.usage[h]
	}
	// b is reachable only through a uintptr in p2.pNext, which is not a
	// reference; Open does the same for appInfo.
	runtime.KeepAlive(b)
	return free, total, nil
}
